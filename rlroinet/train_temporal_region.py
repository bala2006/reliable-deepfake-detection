"""Fresh joint temporal-video and ROI-localization training experiment."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import default_config
from .data import REGION_NAMES, load_index, roi_mask
from .data.temporal_data import SequenceMaskDataset, TrainSequenceAugment, sequence_collate
from .evaluate import calibration_curve, calibrate_decision_policy, eer, save_report
from .feature_cache import (CachedSequenceMaskDataset, build_feature_cache,
                            cached_sequence_collate, load_feature_cache)
from .models import focal_loss
from .precision import autocast, make_scaler
from .region_agent import _mask_loss
from .snapshot import SnapshotScheduler, WarmupCosine
from .temporal_region import TemporalRegionAgent
from .train import configure_reproducibility, write_run_config
from .verdict import load_verdict

log = logging.getLogger("rlroinet.train_temporal_region")


def _iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred > 0.5, target > 0.5
    union = np.logical_or(pred, target).sum()
    return float(np.logical_and(pred, target).sum() / union) if union else 1.0


def _metrics(labels, scores, cfg):
    labels_a, scores_a = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    preds = (scores_a >= cfg.eval.threshold).astype(int)
    both = len(np.unique(labels_a)) > 1
    fake, real = scores_a >= cfg.eval.fake_threshold, scores_a <= cfg.eval.real_threshold
    review = ~(fake | real)
    return {
        "schema_version": cfg.schema_version, "source": cfg.data.source, "n": int(len(labels_a)),
        "video_level": True, "acc": float((preds == labels_a).mean()),
        "auc": float(roc_auc_score(labels_a, scores_a)) if both else float("nan"),
        "f1": float(f1_score(labels_a, preds, zero_division=0)),
        "precision": float(precision_score(labels_a, preds, zero_division=0)),
        "recall": float(recall_score(labels_a, preds, zero_division=0)),
        "eer": eer(labels_a, scores_a),
        "ece": calibration_curve(labels_a, scores_a, cfg.eval.calibration_bins),
        "fp_rate": float(((preds == 1) & (labels_a == 0)).mean()),
        "fn_rate": float(((preds == 0) & (labels_a == 1)).mean()),
        "review_rate": float(review.mean()), "reliable_coverage": float((fake | real).mean()),
        "confident_false_positive_rate": float((fake & (labels_a == 0)).sum() / max(1, (labels_a == 0).sum())),
        "confident_false_negative_rate": float((real & (labels_a == 1)).sum() / max(1, (labels_a == 1).sum())),
    }


# Gate metrics shown in every checkpoint evaluation table (rows).
GATE_METRIC_ROWS = [
    ("auc", "val video AUC", "higher"),
    ("acc", "accuracy", "higher"),
    ("f1", "F1", "higher"),
    ("precision", "precision", "higher"),
    ("recall", "recall", "higher"),
    ("eer", "EER", "lower"),
    ("ece", "ECE (calibration)", "lower"),
    ("region_iou", "region IoU", "higher"),
    ("region_hit", "region hit@0.5", "higher"),
    ("fp_rate", "false positive rate", "lower"),
    ("fn_rate", "false negative rate", "lower"),
    ("confident_false_positive_rate", "confident FP rate", "lower"),
    ("confident_false_negative_rate", "confident FN rate", "lower"),
    ("review_rate", "review rate", "lower"),
    ("reliable_coverage", "reliable coverage", "higher"),
]

# Official acceptance bar for the paper's primary metrics. Every metrics table
# gets a PASS/FAIL column gated against these; metrics without an entry in the
# bar are still reported but not gated.
PASS_BAR = {
    "auc": 0.85,
    "region_iou": 0.75,
    "region_hit": 0.90,
    "confident_false_positive_rate": 0.01,
    "confident_false_negative_rate": 0.01,
    "reliable_coverage": 0.50,
    "review_rate": 0.50,
}

_BETTER = {"higher": max, "lower": min}


def _pass_fail(value, key, direction):
    """PASS/FAIL for ``key`` against PASS_BAR; '—' when no bar exists."""
    if key not in PASS_BAR:
        return "—"
    limit = PASS_BAR[key]
    if value is None or not isinstance(value, (int, float)) or value != value:
        return "—"
    passed = float(value) >= limit if direction == "higher" else float(value) <= limit
    return "PASS" if passed else "FAIL"


def _best_so_far(history, key, direction):
    """Best value for ``key`` across every entry in ``history`` (nan-aware)."""
    vals = [m.get(key) for m in history]
    vals = [float(v) for v in vals
            if isinstance(v, (int, float)) and v == v and not isinstance(v, bool)]
    return _BETTER[direction](vals) if vals else None


def _fmt_table_cell(value, width):
    if value is None:
        return " " * width
    if isinstance(value, float):
        return f"{value:.4f}".rjust(width)
    return str(value).rjust(width)


def _pct_vs_previous(prev, cur, direction):
    """Signed % change from previous to current, with an improvement marker.

    ``direction`` is "higher" (up is good) or "lower" (down is good). Returns a
    fixed-width string like ``+8.6% ^`` / ``-3.2% v`` / `` n/a``.
    """
    if prev is None or cur is None or not isinstance(prev, (int, float)) or not isinstance(cur, (int, float)):
        return "n/a".rjust(10)
    if prev == 0 or not prev == prev or not cur == cur:
        return "n/a".rjust(10)
    pct = (float(cur) - float(prev)) / abs(float(prev)) * 100.0
    improved = (direction == "higher" and pct >= 0) or (direction == "lower" and pct <= 0)
    marker = "^" if improved else "v"
    return f"{pct:+.1f}% {marker}".rjust(10)


def print_metrics_table(history, epoch=None):
    """Print Previous | Current | Best table for every gate metric.

    ``history`` is the list of per-epoch validation dicts (already including
    the just-finished epoch). The table is emitted after every validation,
    i.e. at every checkpoint decision point.
    """
    current = history[-1] if history else {}
    previous = history[-2] if len(history) > 1 else {}
    width = 12
    header = (f"metric".ljust(30) + "previous".rjust(width) + "current".rjust(width)
              + "% vs prev".rjust(10) + "best".rjust(width) + "pass?".rjust(8))
    log.info("=== checkpoint metrics%s ===", f" (epoch {epoch})" if epoch is not None else "")
    bar_parts = [f"{label} {'>=' if direction == 'higher' else '<='}{PASS_BAR[key]:.2f}"
                 for key, label, direction in GATE_METRIC_ROWS if key in PASS_BAR]
    if bar_parts:
        log.info("acceptance bar: %s", " | ".join(bar_parts))
    log.info(header)
    log.info("-" * len(header))
    for key, label, direction in GATE_METRIC_ROWS:
        cur = current.get(key)
        prev = previous.get(key) if previous else None
        best = _best_so_far(history, key, direction)
        log.info(label.ljust(30) + _fmt_table_cell(prev, width) + _fmt_table_cell(cur, width)
                 + _pct_vs_previous(prev, cur, direction) + _fmt_table_cell(best, width)
                 + _pass_fail(cur, key, direction).rjust(8))
    roi = current.get("per_roi_iou")
    if isinstance(roi, dict) and roi:
        for name, value in roi.items():
            roi_vals = [m.get("per_roi_iou", {}).get(name)
                        for m in history if isinstance(m.get("per_roi_iou"), dict)]
            best_roi = _best_so_far([{"x": v} for v in roi_vals], "x", "higher")
            prev_roi = previous.get("per_roi_iou", {}).get(name) if previous else None
            log.info(("  ROI " + name).ljust(30) + _fmt_table_cell(prev_roi, width)
                     + _fmt_table_cell(value, width) + _pct_vs_previous(prev_roi, value, "higher")
                     + _fmt_table_cell(best_roi, width) + "—".rjust(8))
    checked, passed = 0, True
    for key, _, direction in GATE_METRIC_ROWS:
        if key not in PASS_BAR:
            continue
        if _pass_fail(current.get(key), key, direction) != "PASS":
            passed = False
        checked += 1
    overall = ("PASS" if passed else "FAIL") if checked else "—"
    log.info("OVERALL acceptance".ljust(30) + " " * (width * 2 + 10 + width) + overall.rjust(8))
    log.info("-" * len(header))


def _fit_temperature(labels, scores) -> float:
    logits = torch.tensor(np.log(np.clip(scores, 1e-5, 1 - 1e-5) /
                                 (1 - np.clip(scores, 1e-5, 1 - 1e-5))), dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp().clamp(0.05, 20.0), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def _seed_worker(worker_id):
    """Keep worker-side random transforms reproducible and CPU oversubscription low."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def _make_loader(dataset, batch_size, shuffle, workers, prefetch_factor, collate_fn=sequence_collate):
    if workers < 0 or prefetch_factor < 1:
        raise ValueError("workers must be >= 0 and prefetch_factor must be >= 1")
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "worker_init_fn": _seed_worker,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


def _evaluate(agent, loader, cfg, temperature: float = 1.0):
    agent.eval()
    labels, scores, ious = [], [], []
    roi_values = {name: [] for name in REGION_NAMES.values() if name != "OTHER"}
    hits, fake_frames = 0, 0
    non_blocking = bool(loader.pin_memory)
    with torch.inference_mode():
        for batch in loader:
            faces = batch["faces"].to(cfg.train.device, non_blocking=non_blocking)
            with autocast(cfg):
                out = agent(faces)
            scores.extend(torch.sigmoid(out["video_logits"].squeeze(-1).float() / temperature).cpu().tolist())
            labels.extend(batch["labels"].to(torch.int64).tolist())
            masks = batch["masks"]
            predicted = F.interpolate(out["mask"].flatten(0, 1).float(), size=masks.shape[-2:],
                                      mode="bilinear", align_corners=False).squeeze(1).cpu().numpy()
            for sample_index, label in enumerate(batch["labels"].tolist()):
                if int(label) != 1:
                    continue
                layouts = batch["layouts"][sample_index]
                for step, layout in enumerate(layouts):
                    offset = sample_index * len(layouts) + step
                    pred, target = predicted[offset], masks[sample_index, step].numpy()
                    ious.append(_iou(pred, target))
                    hits += int(np.logical_and(pred > 0.5, target > 0.5).any())
                    fake_frames += 1
                    for region_id, roi in roi_mask(layout, cfg.data.face_size).items():
                        roi_array = np.asarray(roi)
                        roi_values[REGION_NAMES[region_id]].append(_iou(pred * roi_array, target * roi_array))
    metrics = _metrics(labels, scores, cfg)
    metrics.update({
        "region_iou": float(np.mean(ious)) if ious else float("nan"),
        "region_hit": float(hits / max(1, fake_frames)),
        "per_roi_iou": {name: float(np.mean(values)) if values else float("nan")
                        for name, values in roi_values.items()},
        "temperature": float(temperature),
    })
    return metrics, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _train_epoch(agent, loader, cfg, optimizer, scaler, grad_accum):
    agent.train()
    started = time.perf_counter()
    total = {name: 0.0 for name in ("temporal", "frame", "mask", "region", "loss")}
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        non_blocking = bool(loader.pin_memory)
        cached = "features" in batch
        faces = None if cached else batch["faces"].to(cfg.train.device, non_blocking=non_blocking)
        masks = batch["masks"].reshape(-1, 1, cfg.data.face_size, cfg.data.face_size).to(
            cfg.train.device, non_blocking=non_blocking)
        labels = batch["labels"].to(cfg.train.device, non_blocking=non_blocking)
        if cached:
            steps = int(batch["features"].shape[1])
            features = batch["features"].flatten(0, 1).to(cfg.train.device, non_blocking=non_blocking)
            verdict_probs = batch["verdict_probs"].flatten(0, 1).to(
                cfg.train.device, non_blocking=non_blocking)
            quality = batch["quality"].flatten(0, 1).to(cfg.train.device, non_blocking=non_blocking)
        else:
            steps = int(batch["faces"].shape[1])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16,
                                enabled=cfg.train.amp):
            if cached:
                out = agent.forward_from_cached(features, verdict_probs, quality,
                                                int(labels.shape[0]), steps)
            else:
                out = agent(faces)
        video_loss = focal_loss(out["video_logits"].float().squeeze(-1), labels,
                                cfg.loss.focal_gamma, cfg.loss.focal_alpha)
        frame_loss = focal_loss(out["frame_logits"].float().flatten(), labels.repeat_interleave(steps),
                                cfg.loss.focal_gamma, cfg.loss.focal_alpha)
        pred_mask = F.interpolate(out["mask"].flatten(0, 1).float(), size=masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        map_loss = _mask_loss(pred_mask, masks, cfg)
        region = out["region"].flatten(0, 1).float()
        region_size = region.shape[-1]
        targets = F.interpolate(masks, size=(region_size, region_size), mode="nearest")
        layouts = [layout for sample_layouts in batch["layouts"] for layout in sample_layouts]
        roi_targets = torch.stack([agent.frame.region.region_masks(region_size, layout).to(
            cfg.train.device, non_blocking=non_blocking) for layout in layouts])
        region_loss = F.binary_cross_entropy_with_logits(region, targets * roi_targets)
        loss = video_loss + cfg.loss.class_weight * frame_loss + map_loss + cfg.loss.region_weight * region_loss
        if scaler is not None:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()
        should_step = (step + 1) % grad_accum == 0 or step + 1 == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        values = {"temporal": video_loss, "frame": frame_loss, "mask": map_loss, "region": region_loss, "loss": loss}
        for name, value in values.items():
            total[name] += float(value.item()) * len(batch["labels"])
        count += len(batch["labels"])
    return {(f"train_{name}_loss" if name != "loss" else "train_loss"): value / max(1, count)
            for name, value in total.items()} | {
        "train_seconds": round(time.perf_counter() - started, 2)}


def _checkpoint(cfg, agent, epoch, metrics, augmentation, optimizer=None):
    ckpt = {
        "checkpoint_format": "temporal-region-v2", "verdict": "honi05", "epoch": epoch,
        "region_state": agent.frame.region.state_dict(),
        "classifier_state": agent.frame.classifier.state_dict(),
        "temporal_state": agent.temporal.state_dict(), "cfg": cfg.to_dict(), "metrics": metrics,
        "experiment": {"fresh": True, "resume": False, "localization_head": "jointly trained",
                       "temporal_fusion": "quality-gated residual",
                       "augmentation": "sequence-consistent photometric" if augmentation else "none"},
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["experiment"]["resume"] = True
    return ckpt


def _latest_temporal_region_checkpoint(checkpoint_dir):
    """Return ``(path, epoch)`` of the newest compatible checkpoint, or None.

    Considers ``best.pt``, ``final.pt``, and every time-based snapshot so an
    interrupted run resumes from the nearest saved point regardless of which
    filename carried it.
    """
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
                or ckpt.get("checkpoint_format") != "temporal-region-v2"
                or ckpt.get("verdict") != "honi05"):
            continue
        ep = int(ckpt.get("epoch", 0))
        if best is None or ep > best[1]:
            best = (p, ep)
    return best


def _load_region_history(metrics_path, start_epoch):
    """Re-read metrics.json so a resumed run keeps prior epochs in the report."""
    path = Path(metrics_path)
    if not path.exists():
        return []
    try:
        hist = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(hist, list):
        return []
    return [m for m in hist if m.get("epoch", 0) < start_epoch]


def _load_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_format") != "temporal-region-v2" or checkpoint.get("verdict") != "honi05":
        raise ValueError("temporal-region checkpoint is incompatible")
    agent.frame.region.load_state_dict(checkpoint["region_state"])
    agent.frame.classifier.load_state_dict(checkpoint["classifier_state"])
    agent.temporal.load_state_dict(checkpoint["temporal_state"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a fresh temporal classifier with ROI localization evidence")
    parser.add_argument("--ffpp-dir", default="data/FaceForensics++")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2, help="GPU micro-batch: videos per optimization input")
    parser.add_argument("--grad-accum", type=int, default=2, help="micro-batches accumulated before an optimizer step")
    parser.add_argument("--workers", type=int, default=4, help="parallel CPU DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--manifest-frames-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0,
                        help="split + training seed (feature caches are keyed per index, "
                             "so each seed builds its own cache automatically)")
    parser.add_argument("--deterministic", action="store_true", default=None,
                        help="use deterministic algorithms and disable cuDNN benchmarking")
    parser.add_argument("--train-sample", type=int, default=2400)
    parser.add_argument("--val-sample", type=int, default=600)
    parser.add_argument("--test-sample", type=int, default=600)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.55)
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16",
                        help="mixed-precision compute dtype (bf16 is faster on Ada and needs no GradScaler)")
    parser.add_argument("--feature-cache", default="",
                        help="directory for precomputed frozen-backbone features; "
                             "train loop skips the backbone forward when present. "
                             "Requires --no-train-augmentation because cached features "
                             "were computed on the original (unaugmented) faces.")
    parser.add_argument("--target-confident-fp", type=float, default=0.01)
    parser.add_argument("--target-confident-fn", type=float, default=0.01)
    parser.add_argument("--snapshot-minutes", type=float, default=30.0,
                        help="wall-clock interval in minutes between snapshot checkpoints (0 disables)")
    parser.add_argument("--lr-min-factor", type=float, default=0.1,
                        help="cosine schedule decays head LR from --head-lr down to head-lr * this factor")
    parser.add_argument("--lr-warmup-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=12,
                        help="stop after this many epochs without val-AUC improvement "
                             "(0 disables early stopping)")
    parser.add_argument("--early-stop-min-epochs", type=int, default=6,
                        help="never early-stop before this epoch")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the nearest saved checkpoint in --checkpoint-dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (args.epochs < 1 or args.batch < 1 or args.grad_accum < 1 or args.workers < 0 or
            args.prefetch_factor < 1 or args.sequence_length < 2 or not 0 < args.focal_alpha < 1 or
            not 0 <= args.target_confident_fp < 1 or not 0 <= args.target_confident_fn < 1 or
            not 0 <= args.lr_min_factor <= 1 or args.lr_warmup_epochs < 0 or
            args.lr_warmup_epochs >= args.epochs or args.early_stop_patience < 0 or
            args.early_stop_min_epochs < 1):
        raise SystemExit("invalid temporal-region training arguments")
    if args.feature_cache and not args.no_train_augmentation:
        raise SystemExit("--feature-cache requires --no-train-augmentation (cached features "
                         "were computed on unaugmented faces)")
    if not torch.cuda.is_available():
        raise SystemExit("temporal-region training requires CUDA")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; use --amp-dtype fp16")
    cfg = default_config()
    cfg.data.source, cfg.data.ffpp_dir = "ffpp", args.ffpp_dir
    cfg.data.seed = args.seed
    cfg.train.seed = args.seed
    cfg.data.sequence_length, cfg.data.manifest_frames_per_video = args.sequence_length, args.manifest_frames_per_video
    cfg.data.sample_train, cfg.data.sample_val, cfg.data.sample_test = args.train_sample, args.val_sample, args.test_sample
    cfg.loss.focal_alpha, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr_head = args.focal_alpha, args.epochs, args.batch, args.head_lr
    cfg.train.amp_dtype = args.amp_dtype
    cfg.train.checkpoint_dir = args.checkpoint_dir
    root = Path(args.checkpoint_dir).parent
    cfg.train.log_dir, cfg.train.metrics_path = str(root / "logs"), str(root / "metrics.json")
    if args.deterministic is not None:
        cfg.train.deterministic = args.deterministic
    cfg.resolve_paths()
    configure_reproducibility(cfg)
    write_run_config(root, cfg, trainer="train_temporal_region", epochs=int(args.epochs),
                     resume=bool(args.resume), sequence_length=int(args.sequence_length),
                     feature_cache=bool(args.feature_cache),
                     device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
                     early_stop_patience=int(args.early_stop_patience),
                     early_stop_min_epochs=int(args.early_stop_min_epochs), args=vars(args))
    train_index = load_index(cfg, "train")
    verdict = load_verdict("honi05", device="cuda")
    use_cache = bool(args.feature_cache)
    if use_cache:
        cache = load_feature_cache(train_index, cfg, "honi05", args.feature_cache)
        if cache is None:
            cache = build_feature_cache(train_index, cfg, verdict, args.feature_cache)
        train_ds = CachedSequenceMaskDataset(train_index, cfg, cache)
        log.info("temporal-region training on cached backbone features (train=%d sequences)", len(train_ds))
    else:
        train_base = SequenceMaskDataset(train_index, cfg)
        train_ds = train_base if args.no_train_augmentation else TrainSequenceAugment(train_base)
    val_ds = SequenceMaskDataset(load_index(cfg, "val"), cfg)
    test_ds = SequenceMaskDataset(load_index(cfg, "test"), cfg)
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError("each temporal split must contain complete same-video sequences")
    train_loader = _make_loader(train_ds, args.batch, True, args.workers, args.prefetch_factor,
                                collate_fn=cached_sequence_collate if use_cache else sequence_collate)
    val_loader = _make_loader(val_ds, args.batch, False, args.workers, args.prefetch_factor)
    test_loader = _make_loader(test_ds, args.batch, False, args.workers, args.prefetch_factor)
    agent = TemporalRegionAgent(cfg, verdict).to("cuda")
    optimizer = torch.optim.AdamW(agent.trainable_parameters(), lr=args.head_lr, weight_decay=cfg.train.weight_decay)
    scaler = make_scaler(cfg)
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshotter = SnapshotScheduler(checkpoint_dir, interval_minutes=args.snapshot_minutes)
    started = time.perf_counter()

    history, best_auc, best_epoch, final_metrics, start_epoch = [], -float("inf"), 0, None, 1
    if args.resume:
        found = _latest_temporal_region_checkpoint(checkpoint_dir)
        if found is not None:
            resume_path, saved_epoch = found
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            agent.frame.region.load_state_dict(ckpt["region_state"])
            agent.frame.classifier.load_state_dict(ckpt["classifier_state"])
            agent.temporal.load_state_dict(ckpt["temporal_state"])
            if isinstance(ckpt.get("optimizer_state"), dict):
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except (RuntimeError, ValueError) as exc:
                    log.warning("optimizer state incompatible with this run; "
                                "starting the optimizer fresh (%s)", exc)
            saved_metrics = ckpt.get("metrics") or {}
            best_auc = float(saved_metrics.get("auc", -float("inf")))
            best_epoch = saved_epoch
            start_epoch = saved_epoch + 1
            history = _load_region_history(cfg.train.metrics_path, start_epoch)
            if history:
                best_entry = max(
                    (m for m in history if isinstance(m.get("auc"), (int, float))
                     and m["auc"] == m["auc"]),
                    key=lambda m: m["auc"], default=None)
                if best_entry is not None and float(best_entry["auc"]) > best_auc:
                    best_auc, best_epoch = float(best_entry["auc"]), int(best_entry.get("epoch", 0))
            log.info("resuming temporal-region training from %s (epoch %d, best val AUC=%.4f)",
                     resume_path.name, saved_epoch, best_auc)
        else:
            log.info("--resume requested but no compatible checkpoint found; starting fresh")
    scheduler = WarmupCosine(optimizer, args.head_lr, args.epochs,
                             warmup_epochs=args.lr_warmup_epochs,
                             min_lr=args.head_lr * args.lr_min_factor,
                             epoch=start_epoch - 1)
    if start_epoch > args.epochs:
        raise SystemExit(f"already trained through epoch {start_epoch - 1} >= --epochs {args.epochs}")

    log.info("temporal-region run: train=%d val=%d test=%d sequences, sequence=%d, micro_batch=%d, effective_batch=%d, workers=%d, prefetch=%d, augmentation=%s, lr_schedule=warmup%d+cosine(min=%.1e), start_epoch=%d",
             len(train_ds), len(val_ds), len(test_ds), args.sequence_length, args.batch,
             args.batch * args.grad_accum, args.workers, args.prefetch_factor, not args.no_train_augmentation,
             args.lr_warmup_epochs, args.head_lr * args.lr_min_factor, start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        train = _train_epoch(agent, train_loader, cfg, optimizer, scaler, args.grad_accum)
        validation, _, _ = _evaluate(agent, val_loader, cfg)
        lr_now = scheduler.step()
        validation.update(train | {"epoch": epoch, "lr": round(lr_now, 8),
                                   "elapsed_minutes": round((time.perf_counter() - started) / 60.0, 1)})
        history.append(validation)
        save_report(history, cfg.train.metrics_path)
        final_metrics = validation
        print_metrics_table(history, epoch=epoch)
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            best_epoch = epoch
            torch.save(_checkpoint(cfg, agent, epoch, validation, not args.no_train_augmentation, optimizer), checkpoint_dir / "best.pt")
        snapshot_path = snapshotter.maybe_save(
            lambda ep, m: _checkpoint(cfg, agent, ep, m, not args.no_train_augmentation, optimizer), epoch, validation)
        log.info("epoch %d/%d loss=%.4f lr=%.2e val AUC=%.4f ACC=%.4f P=%.4f R=%.4f IoU=%.4f hit=%.4f elapsed=%.1fmin%s",
                 epoch, args.epochs, train["train_loss"], lr_now, validation["auc"], validation["acc"],
                 validation["precision"], validation["recall"], validation["region_iou"], validation["region_hit"],
                 validation["elapsed_minutes"], f" snapshot={snapshot_path.name}" if snapshot_path else "")
        if (args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs
                and epoch - best_epoch >= args.early_stop_patience):
            final_metrics["early_stopped"] = True
            final_metrics["early_stop_reason"] = (
                f"val AUC best {best_auc:.4f} at epoch {best_epoch}; "
                f"no improvement for {args.early_stop_patience} epochs")
            final_metrics["epochs_completed"] = epoch
            log.info("early stopping at epoch %d: %s", epoch, final_metrics["early_stop_reason"])
            break
    final_epoch = int(final_metrics.get("epoch", args.epochs))
    torch.save(_checkpoint(cfg, agent, final_epoch, final_metrics, not args.no_train_augmentation, optimizer),
               checkpoint_dir / "final.pt")
    selected = _load_checkpoint(agent, checkpoint_dir / "best.pt")
    _, val_labels, val_scores = _evaluate(agent, val_loader, cfg)
    temperature = _fit_temperature(val_labels, val_scores)
    validation, val_labels, val_scores = _evaluate(agent, val_loader, cfg, temperature)
    policy = calibrate_decision_policy(val_labels, val_scores, args.target_confident_fp, args.target_confident_fn)
    policy.update({"checkpoint": "best.pt", "checkpoint_epoch": int(selected["epoch"]),
                   "selection": "validation_video_auc", "temperature": temperature})
    (root / "decision_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    default_test, _, _ = _evaluate(agent, test_loader, cfg)
    for key in ("real_threshold", "fake_threshold", "review_enabled", "video_aggregation"):
        setattr(cfg.eval, key, policy[key])
    calibrated_validation, _, _ = _evaluate(agent, val_loader, cfg, temperature)
    calibrated_test, _, _ = _evaluate(agent, test_loader, cfg, temperature)
    result = default_test | {"seed": args.seed, "epoch": int(selected["epoch"]), "selected_by": "validation_video_auc",
                             "validation_best_auc": best_auc, "validation_calibrated_metrics": calibrated_validation,
                             "decision_policy": policy, "calibrated_policy_metrics": calibrated_test}
    (root / "test_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
