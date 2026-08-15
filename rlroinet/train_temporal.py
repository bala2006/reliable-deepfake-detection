"""Fresh one-epoch temporal-quality experiment over frozen verdict features."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import default_config
from .data import load_index
from .data.temporal_data import SequenceMaskDataset
from .evaluate import calibration_curve, eer, save_report
from .models import focal_loss
from .precision import autocast, make_scaler
from .temporal_quality import TemporalQualityAgent
from .verdict import load_verdict

log = logging.getLogger("rlroinet.train_temporal")


def _metrics(labels, scores, cfg):
    labels_a, scores_a = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    preds = (scores_a >= cfg.eval.threshold).astype(int)
    both = len(np.unique(labels_a)) > 1
    out = {
        "schema_version": cfg.schema_version, "source": cfg.data.source,
        "n": int(len(labels_a)), "video_level": True,
        "acc": float((preds == labels_a).mean()),
        "auc": float(roc_auc_score(labels_a, scores_a)) if both else float("nan"),
        "f1": float(f1_score(labels_a, preds, zero_division=0)),
        "precision": float(precision_score(labels_a, preds, zero_division=0)),
        "recall": float(recall_score(labels_a, preds, zero_division=0)),
        "eer": eer(labels_a, scores_a),
        "ece": calibration_curve(labels_a, scores_a, cfg.eval.calibration_bins),
        "fp_rate": float(((preds == 1) & (labels_a == 0)).mean()),
        "fn_rate": float(((preds == 0) & (labels_a == 1)).mean()),
        "region_iou": float("nan"), "region_hit": float("nan"), "per_roi_iou": {},
    }
    fake, real = scores_a >= cfg.eval.fake_threshold, scores_a <= cfg.eval.real_threshold
    reviewed = ~(fake | real)
    out.update({"review_rate": float(reviewed.mean()), "reliable_coverage": float((fake | real).mean()),
                "confident_false_positive_rate": float((fake & (labels_a == 0)).sum() / max(1, (labels_a == 0).sum())),
                "confident_false_negative_rate": float((real & (labels_a == 1)).sum() / max(1, (labels_a == 1).sum()))})
    return out


def _evaluate(agent, dataset, cfg, batch_size):
    agent.eval()
    labels, scores = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
            faces = torch.stack([sample.faces for sample in batch]).to(cfg.train.device)
            with autocast(cfg):
                logits = agent(faces)["logits"].squeeze(-1)
            scores.extend(torch.sigmoid(logits).float().cpu().tolist())
            labels.extend(sample.label for sample in batch)
    return _metrics(labels, scores, cfg)


def _train_epoch(agent, dataset, cfg, optimizer, scaler, batch_size):
    agent.train()
    order = torch.randperm(len(dataset)).tolist()
    total_loss, total_count, started = 0.0, 0, time.perf_counter()
    for start in range(0, len(order), batch_size):
        batch = [dataset[i] for i in order[start:start + batch_size]]
        faces = torch.stack([sample.faces for sample in batch]).to(cfg.train.device)
        labels = torch.tensor([sample.label for sample in batch], device=cfg.train.device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with autocast(cfg):
            logits = agent(faces)["logits"].squeeze(-1)
            loss = focal_loss(logits.float(), labels, cfg.loss.focal_gamma, cfg.loss.focal_alpha)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
            optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        total_count += len(batch)
    return {"train_loss": total_loss / max(1, total_count),
            "train_seconds": round(time.perf_counter() - started, 2)}


def _checkpoint(cfg, verdict, agent, epoch, metrics, initial_checkpoint: str | None):
    return {"checkpoint_format": "temporal-quality-v1", "verdict": verdict,
            "temporal_state": agent.head.state_dict(), "cfg": cfg.to_dict(), "epoch": epoch,
            "metrics": metrics, "experiment": {
                "fresh": initial_checkpoint is None, "resume": initial_checkpoint is not None,
                "initial_checkpoint": initial_checkpoint,
                "optimizer": "fresh AdamW for this run",
                "localization_head": "not trained in temporal feasibility experiment",
                "augmentation": "none in temporal feasibility experiment"}}


def main():
    parser = argparse.ArgumentParser(description="Train or continue a frozen-backbone temporal-quality head")
    parser.add_argument("--ffpp-dir", default="data/FaceForensics++")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None,
                        help="optional temporal-quality-v1 checkpoint; head weights continue with a fresh optimizer")
    parser.add_argument("--epochs", type=int, default=1,
                        help="total epoch number; an initialized run starts at saved epoch + 1")
    parser.add_argument("--batch", type=int, default=8, help="videos per optimization step")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--manifest-frames-per-video", type=int, default=8)
    parser.add_argument("--train-sample", type=int, default=2400)
    parser.add_argument("--val-sample", type=int, default=600)
    parser.add_argument("--test-sample", type=int, default=600)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.55)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16",
                        help="mixed-precision compute dtype (bf16 is faster on Ada and needs no GradScaler)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.epochs < 1 or args.batch < 1 or args.sequence_length < 2 or not 0.0 < args.focal_alpha < 1.0:
        raise SystemExit("invalid temporal training arguments")
    if not torch.cuda.is_available():
        raise SystemExit("temporal-quality training requires CUDA")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; use --amp-dtype fp16")
    cfg = default_config()
    cfg.data.source, cfg.data.ffpp_dir = "ffpp", args.ffpp_dir
    cfg.data.sequence_length, cfg.data.manifest_frames_per_video = args.sequence_length, args.manifest_frames_per_video
    cfg.data.sample_train, cfg.data.sample_val, cfg.data.sample_test = args.train_sample, args.val_sample, args.test_sample
    cfg.loss.focal_alpha, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr_head = args.focal_alpha, args.epochs, args.batch, args.head_lr
    cfg.train.amp_dtype = args.amp_dtype
    cfg.train.checkpoint_dir = args.checkpoint_dir
    root = Path(args.checkpoint_dir).parent
    cfg.train.log_dir, cfg.train.metrics_path = str(root / "logs"), str(root / "metrics.json")
    cfg.resolve_paths()
    train_ds = SequenceMaskDataset(load_index(cfg, "train"), cfg)
    val_ds = SequenceMaskDataset(load_index(cfg, "val"), cfg)
    test_ds = SequenceMaskDataset(load_index(cfg, "test"), cfg)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("each temporal split must contain complete sequences")
    verdict = load_verdict("honi05", device="cuda")
    agent = TemporalQualityAgent(cfg, verdict).to("cuda")
    start_epoch = 1
    if args.init_checkpoint:
        initial_path = Path(args.init_checkpoint)
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        if not initial_path.is_file() or initial.get("checkpoint_format") != "temporal-quality-v1" or initial.get("verdict") != "honi05":
            raise ValueError("--init-checkpoint must be a compatible temporal-quality-v1 honi05 checkpoint")
        initial_cfg = initial.get("cfg", {})
        if (initial_cfg.get("data", {}).get("sequence_length") != args.sequence_length or
                initial_cfg.get("model", {}).get("temporal_hidden") != cfg.model.temporal_hidden or
                initial_cfg.get("model", {}).get("temporal_quality_hidden") != cfg.model.temporal_quality_hidden):
            raise ValueError("--init-checkpoint temporal configuration is incompatible")
        agent.head.load_state_dict(initial["temporal_state"])
        start_epoch = int(initial.get("epoch", 0)) + 1
    if start_epoch > args.epochs:
        raise ValueError("--epochs must exceed the initialized checkpoint epoch")
    log.info("temporal-quality run: train=%d val=%d test=%d sequences, steps=%d, batch=%d, start_epoch=%d",
             len(train_ds), len(val_ds), len(test_ds), cfg.data.sequence_length, args.batch, start_epoch)
    optimizer = torch.optim.AdamW(agent.trainable_parameters(), lr=args.head_lr, weight_decay=cfg.train.weight_decay)
    scaler = make_scaler(cfg)
    history, best_auc = [], -float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        train = _train_epoch(agent, train_ds, cfg, optimizer, scaler, args.batch)
        validation = _evaluate(agent, val_ds, cfg, args.batch)
        validation.update(train | {"epoch": epoch})
        history.append(validation)
        save_report(history, cfg.train.metrics_path)
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(_checkpoint(cfg, "honi05", agent, epoch, validation, args.init_checkpoint),
                       Path(cfg.train.checkpoint_dir) / "best.pt")
        log.info("epoch %d/%d loss=%.4f val AUC=%.4f ACC=%.4f P=%.4f R=%.4f",
                 epoch, args.epochs, train["train_loss"], validation["auc"], validation["acc"], validation["precision"], validation["recall"])
    test = _evaluate(agent, test_ds, cfg, args.batch)
    test.update(train | {"epoch": args.epochs, "initial_epoch": start_epoch - 1,
                 "selected_by": "validation_auc", "validation_best_auc": best_auc})
    torch.save(_checkpoint(cfg, "honi05", agent, args.epochs, test, args.init_checkpoint),
               Path(cfg.train.checkpoint_dir) / "final.pt")
    (root / "test_metrics.json").write_text(json.dumps(test, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(test, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
