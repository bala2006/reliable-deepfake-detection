"""Train the isolated parameter-efficient spatial forensic adapter candidate.

The frozen honi05 verdict model, data split, mask/ROI losses, and evaluator are
unchanged.  The candidate is written to its own directory and is never used by
production unless explicitly selected later.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

from .adapter_agent import AdapterRegionAgent
from .config import default_config
from .data import load_index, make_mask_dataset
from .data.maskdata import make_data_loader, mask_sample_collate
from .evaluate import evaluate, save_report
from .feature_cache import (CachedMaskDataset, build_feature_cache, cached_mask_collate,
                            load_feature_cache)
from .precision import make_scaler
from .region_agent import region_training_step_from_batch
from .snapshot import SnapshotScheduler, WarmupCosine
from .train import configure_reproducibility, write_run_config
from .verdict import load_verdict

log = logging.getLogger("rlroinet.train_adapter")


def _checkpoint(cfg, agent, optimizer, epoch, metrics):
    return {
        "checkpoint_format": "adapter-region-v1",
        "verdict": "honi05",
        "adapter_state": agent.adapter.state_dict(),
        "region_state": agent.region.state_dict(),
        "classifier_state": agent.classifier.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "experiment": {
            "backbone": "frozen honi05",
            "adapter": "bottleneck spatial residual",
            "classifier": "adapted avg+max feature plus frozen verdict logit",
            "localization": "existing mask and four-ROI decoder",
        },
    }


def _latest_adapter_checkpoint(checkpoint_dir):
    """Return ``(path, epoch)`` of the newest compatible adapter checkpoint, or None."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    best = None
    for p in ckpt_dir.glob("*.pt"):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if (not isinstance(ckpt, dict)
                or ckpt.get("checkpoint_format") != "adapter-region-v1"):
            continue
        ep = int(ckpt.get("epoch", 0))
        if best is None or ep > best[1]:
            best = (p, ep)
    return best


def _load_history(metrics_path, start_epoch):
    if not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path) as fh:
            hist = json.load(fh)
    except Exception:
        return []
    if not isinstance(hist, list):
        return []
    return [m for m in hist if m.get("epoch", 0) < start_epoch]


def _train_epoch(agent, loader, cfg, optimizer, scaler, use_cache):
    agent.train()
    totals = {"cls_loss": 0.0, "mask_loss": 0.0, "region_loss": 0.0, "loss": 0.0}
    count = 0
    for batch in loader:
        values = region_training_step_from_batch(agent, batch, cfg, optimizer, scaler)
        n = int(batch["labels"].shape[0])
        for key in totals:
            totals[key] += values[key] * n
        count += n
    return {key: value / max(1, count) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train the frozen-backbone spatial adapter candidate")
    parser.add_argument("--ffpp-dir", default="data/FaceForensics++")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="parallel CPU DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--train-sample", type=int, default=2400)
    parser.add_argument("--val-sample", type=int, default=600)
    parser.add_argument("--test-sample", type=int, default=600)
    parser.add_argument("--manifest-frames-per-video", type=int, default=8)
    parser.add_argument("--adapter-bottleneck", type=int, default=64)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.55)
    parser.add_argument("--methods", default="Deepfakes,Face2Face,FaceSwap,NeuralTextures")
    parser.add_argument(
        "--leakage-safe-split", action="store_true",
        help="use the official FF++ source-pair train/val/test lists",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16",
                        help="mixed-precision compute dtype (bf16 is faster on Ada and needs no GradScaler)")
    parser.add_argument("--feature-cache", default="",
                        help="directory for precomputed frozen-backbone features; "
                             "recomputing the backbone each epoch is skipped when present "
                             "(train split only; validation/test still use the real faces)")
    parser.add_argument("--snapshot-minutes", type=float, default=30.0,
                        help="wall-clock interval in minutes between snapshot checkpoints (0 disables)")
    parser.add_argument("--lr-min-factor", type=float, default=0.1,
                        help="cosine schedule decays head LR from --head-lr down to head-lr * this factor")
    parser.add_argument("--lr-warmup-epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the nearest saved checkpoint in --checkpoint-dir")
    parser.add_argument("--seed", type=int, default=None,
                        help="training and split seed (defaults to the config seed)")
    parser.add_argument("--deterministic", action="store_true", default=None,
                        help="use deterministic algorithms and disable cuDNN benchmarking")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.adapter_bottleneck < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise SystemExit("epochs, batch, adapter bottleneck, workers, and prefetch factor must be positive")
    if not 0.0 < args.focal_alpha < 1.0 or args.head_lr <= 0.0:
        raise SystemExit("focal alpha must be in (0,1) and head learning rate must be positive")
    if not 0.0 <= args.lr_min_factor <= 1.0 or args.lr_warmup_epochs < 0 or args.lr_warmup_epochs >= args.epochs:
        raise SystemExit("lr_min_factor must be in [0,1] and warmup epochs in [0, epochs)")
    if not torch.cuda.is_available():
        raise SystemExit("adapter training requires CUDA")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; use --amp-dtype fp16")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    if args.seed is not None:
        cfg.data.seed = args.seed
        cfg.train.seed = args.seed
    if args.deterministic is not None:
        cfg.train.deterministic = args.deterministic
    configure_reproducibility(cfg)
    cfg.data.source = "ffpp"
    cfg.data.ffpp_dir = args.ffpp_dir
    cfg.data.methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    cfg.data.sample_train = args.train_sample
    cfg.data.sample_val = args.val_sample
    cfg.data.sample_test = args.test_sample
    cfg.data.manifest_frames_per_video = args.manifest_frames_per_video
    cfg.data.leakage_safe_split = args.leakage_safe_split
    cfg.model.adapter_bottleneck = args.adapter_bottleneck
    cfg.loss.focal_alpha = args.focal_alpha
    cfg.train.device = "cuda"
    cfg.train.batch_size = args.batch
    cfg.train.epochs = args.epochs
    cfg.train.lr_head = args.head_lr
    cfg.train.amp = args.amp
    cfg.train.amp_dtype = args.amp_dtype
    cfg.train.checkpoint_dir = args.checkpoint_dir
    root = Path(args.checkpoint_dir).parent
    cfg.train.log_dir = str(root / "logs")
    cfg.train.metrics_path = str(root / "metrics.json")
    cfg.resolve_paths()
    write_run_config(root, cfg, trainer="train_adapter", epochs=int(args.epochs),
                     resume=bool(args.resume), feature_cache=bool(args.feature_cache))

    train_ds = make_mask_dataset(cfg, "train", device="cpu")
    val_ds = make_mask_dataset(cfg, "val", device="cpu")
    test_ds = make_mask_dataset(cfg, "test", device="cpu")
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("train, validation, and test datasets must all be non-empty")

    verdict = load_verdict("honi05", device="cuda")
    agent = AdapterRegionAgent(cfg, verdict).to("cuda")
    optimizer = torch.optim.AdamW(
        agent.trainable_parameters(), lr=args.head_lr, weight_decay=cfg.train.weight_decay)
    scaler = make_scaler(cfg)
    use_cache = bool(args.feature_cache)
    if use_cache:
        train_index = load_index(cfg, "train")
        cache = load_feature_cache(train_index, cfg, "honi05", args.feature_cache)
        if cache is None:
            cache = build_feature_cache(train_index, cfg, verdict, args.feature_cache)
        train_ds = CachedMaskDataset(train_index, cfg, cache, device="cpu")
        train_collate = cached_mask_collate
        log.info("adapter training on cached backbone features (train=%d)", len(train_ds))
    else:
        train_collate = mask_sample_collate
    train_loader = make_data_loader(train_ds, args.batch, True, args.workers,
                                    args.prefetch_factor, collate_fn=train_collate)
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshotter = SnapshotScheduler(checkpoint_dir, interval_minutes=args.snapshot_minutes)

    history, best_score, start_epoch = [], -float("inf"), 1
    if args.resume:
        found = _latest_adapter_checkpoint(checkpoint_dir)
        if found is not None:
            resume_path, saved_epoch = found
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            agent.adapter.load_state_dict(ckpt["adapter_state"])
            agent.region.load_state_dict(ckpt["region_state"])
            agent.classifier.load_state_dict(ckpt["classifier_state"])
            if isinstance(ckpt.get("optimizer_state"), dict):
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except (RuntimeError, ValueError) as exc:
                    log.warning("optimizer state incompatible with this run; "
                                "starting the optimizer fresh (%s)", exc)
            saved_metrics = ckpt.get("metrics") or {}
            best_score = float(saved_metrics.get("auc", -float("inf")))
            start_epoch = saved_epoch + 1
            history = _load_history(cfg.train.metrics_path, start_epoch)
            log.info("resuming adapter training from %s (epoch %d, best val AUC=%.4f)",
                     resume_path.name, saved_epoch, best_score)
        else:
            log.info("--resume requested but no compatible checkpoint found; starting fresh")
    scheduler = WarmupCosine(optimizer, args.head_lr, args.epochs,
                             warmup_epochs=args.lr_warmup_epochs,
                             min_lr=args.head_lr * args.lr_min_factor,
                             epoch=start_epoch - 1)
    if start_epoch > args.epochs:
        raise SystemExit(f"already trained through epoch {start_epoch - 1} >= --epochs {args.epochs}")

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        losses = _train_epoch(agent, train_loader, cfg, optimizer, scaler, use_cache)
        validation = evaluate(agent, val_ds, cfg, device="cuda", video_level=True)
        lr_now = scheduler.step()
        validation.update({"epoch": epoch, "lr": round(lr_now, 8),
                           **{f"train_{k}": v for k, v in losses.items()},
                           "elapsed_minutes": round((time.perf_counter() - started) / 60.0, 1)})
        history.append(validation)
        save_report(history, cfg.train.metrics_path)
        score = float(validation["auc"])
        log.info("epoch %d/%d val video AUC=%.4f ACC=%.4f P=%.4f R=%.4f IoU=%.4f hit=%.4f lr=%.2e elapsed=%.1fmin",
                 epoch, args.epochs, score, validation["acc"], validation["precision"],
                 validation["recall"], validation.get("region_iou", float("nan")),
                 validation.get("region_hit", float("nan")), lr_now, validation["elapsed_minutes"])
        if score > best_score:
            best_score = score
            torch.save(_checkpoint(cfg, agent, optimizer, epoch, validation), checkpoint_dir / "best.pt")
        snapshot_path = snapshotter.maybe_save(
            lambda ep, m: _checkpoint(cfg, agent, optimizer, ep, m), epoch, validation)
        if snapshot_path:
            log.info("snapshot saved: %s", snapshot_path.name)

    selected = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    agent.adapter.load_state_dict(selected["adapter_state"])
    agent.region.load_state_dict(selected["region_state"])
    agent.classifier.load_state_dict(selected["classifier_state"])
    test = evaluate(agent, test_ds, cfg, device="cuda", video_level=True)
    test.update({"epoch": int(selected["epoch"]), "selected_by": "validation_video_auc",
                 "validation_best_auc": best_score,
                 "train_contract": {
                     "methods": list(cfg.data.methods),
                     "manifest_frames_per_video": cfg.data.manifest_frames_per_video,
                     "sample_train": cfg.data.sample_train,
                     "sample_val": cfg.data.sample_val,
                     "sample_test": cfg.data.sample_test,
                     "adapter_bottleneck": args.adapter_bottleneck,
                 }})
    torch.save(_checkpoint(cfg, agent, optimizer, int(selected["epoch"]), test), checkpoint_dir / "final.pt")
    report_path = checkpoint_dir / "test_metrics.json"
    report_path.write_text(json.dumps(test, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(test, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
