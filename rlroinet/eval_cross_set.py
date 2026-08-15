"""Post-training cross-set evaluation for a temporal-region v2 checkpoint.

Runs automatically after seed 0 passes the health gate (before seeds 1+2, so a
broken transfer is caught before more GPU hours are spent). Two checks:

  1. **Unseen-method FF++ breakdown** -- the held-out FF++ test videos grouped by
     manipulation method (FaceSwap / Deepfakes / Face2Face / NeuralTextures).
     This is the strongest signal of whether the frozen ``honi05`` backbone
     (trained on Celeb-DF v2) generalizes across manipulation methods.
  2. **Unseen-dataset Celeb-DF** -- raw Celeb-DF videos scored verdict-only,
     when a ``data/celebdf`` tree is present.

Report: ``<seed_dir>/eval_cross_set.json`` (also printed to stdout).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .config import Config, default_config
from .data import REGION_NAMES, load_index, roi_mask
from .data.temporal_data import SequenceMaskDataset
from .evaluate import calibration_curve, eer
from .precision import autocast, enable_tf32
from .predict import aggregate_video_confidence
from .temporal_region import TemporalRegionAgent
from .train_temporal_region import _iou, _make_loader, _metrics
from .verdict import load_verdict

log = logging.getLogger("rlroinet.eval_cross_set")

_CHECKPOINT_FORMAT = "temporal-region-v2"
_VERDICT = "honi05"


def _load_agent(checkpoint: Path, device: str) -> tuple:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("checkpoint_format") != _CHECKPOINT_FORMAT:
        raise SystemExit(f"expected a {_CHECKPOINT_FORMAT} checkpoint, got {checkpoint}")
    cfg = Config.from_dict(ckpt["cfg"]) if isinstance(ckpt.get("cfg"), dict) else default_config()
    cfg.train.device = device
    verdict = load_verdict(_VERDICT, device=device)
    agent = TemporalRegionAgent(cfg, verdict).to(device)
    agent.frame.region.load_state_dict(ckpt["region_state"])
    agent.frame.classifier.load_state_dict(ckpt["classifier_state"])
    agent.temporal.load_state_dict(ckpt["temporal_state"])
    return agent, cfg, ckpt


@torch.inference_mode()
def _collect(agent, loader, cfg, temperature: float = 1.0):
    """One pass over a SequenceMaskDataset: metrics + per-video labels/scores."""
    agent.eval()
    videos, labels, scores, ious, hits, fake_frames = [], [], [], [], 0, 0
    roi_values = {}
    for batch in loader:
        faces = batch["faces"].to(cfg.train.device)
        with autocast(cfg):
            out = agent(faces)
        scores.extend(torch.sigmoid(out["video_logits"].squeeze(-1).float()
                                   / temperature).cpu().tolist())
        labels.extend(batch["labels"].to(torch.int64).tolist())
        videos.extend(batch["videos"])
        if float(batch["labels"].sum()) < 1:
            continue
        masks = batch["masks"]
        predicted = torch.nn.functional.interpolate(
            out["mask"].flatten(0, 1).float(), size=masks.shape[-2:],
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
                    roi_values.setdefault(REGION_NAMES[region_id], []).append(
                        _iou(pred * np.asarray(roi), target * np.asarray(roi)))
    metrics = _metrics(labels, scores, cfg)
    metrics.update({
        "region_iou": float(np.mean(ious)) if ious else float("nan"),
        "region_hit": float(hits / max(1, fake_frames)),
        "per_roi_iou": {name: float(np.mean(v)) if v else float("nan")
                        for name, v in roi_values.items()},
        "temperature": float(temperature),
    })
    return metrics, videos, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def _per_method(videos, labels, scores, cfg) -> dict:
    """Group the single held-out FF++ test set by manipulation method."""
    real = [i for i, _ in enumerate(videos) if int(labels[i]) == 0]
    fakes = [i for i, _ in enumerate(videos) if int(labels[i]) == 1]
    methods = sorted({str(videos[i]).split("/")[0] for i in fakes})
    out = {}
    for m in methods:
        idx = real + [i for i in fakes if str(videos[i]).split("/")[0] == m]
        if not idx:
            continue
        metrics = _metrics(labels[idx], scores[idx], cfg)
        metrics.update({"method": m, "n": len(idx), "n_real": len(real),
                        "n_fake": len(idx) - len(real)})
        out[m] = metrics
        log.info("FF++ method %-16s ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f (n=%d fake=%d real=%d)",
                 m, metrics["acc"], metrics["auc"], metrics["f1"], metrics["eer"],
                 metrics["n"], metrics["n_fake"], metrics["n_real"])
    return out


@torch.inference_mode()
def _celebdf(agent, cfg, celebdf_dir, device, quick: int = 0, temperature: float = 1.0) -> dict:
    """Cross-dataset verdict-only check on raw Celeb-DF videos (no GT masks)."""
    from .generalize import _list_videos, _sample_frames
    videos = _list_videos(Path(celebdf_dir))
    if quick > 0:
        real = [v for v in videos if v["label"] == 0]
        fake = [v for v in videos if v["label"] == 1]
        rng = np.random.default_rng(0)
        rng.shuffle(real)
        rng.shuffle(fake)
        videos = real[:quick] + fake[:quick]
    labels, scores, missing = [], [], 0
    agent.eval()
    for item in videos:
        frames = _sample_frames(item["video"], cfg)
        if frames is None:
            missing += 1
            continue
        with autocast(cfg):
            out = agent(frames.unsqueeze(0).to(device))
        prob = torch.sigmoid(out["video_logits"].squeeze(-1).float() / temperature)
        frame_scores = prob.reshape(-1).cpu().numpy()
        scores.append(float(aggregate_video_confidence(frame_scores, cfg)))
        labels.append(item["label"])
    labels_a = np.asarray(labels)
    scores_a = np.asarray(scores)
    preds = (scores_a >= cfg.eval.threshold).astype(int)
    has_both = len(np.unique(labels_a)) > 1
    metrics = {
        "dataset": "celebdf_raw",
        "n_videos": int(len(labels_a)),
        "n_real": int((labels_a == 0).sum()),
        "n_fake": int((labels_a == 1).sum()),
        "skipped": missing,
        "acc": float((preds == labels_a).mean()) if len(labels_a) else float("nan"),
        "auc": float(roc_auc_score(labels_a, scores_a)) if has_both else float("nan"),
        "f1": float(f1_score(labels_a, preds, zero_division=0)) if has_both else float("nan"),
        "eer": eer(labels_a, scores_a) if len(labels_a) else float("nan"),
        "ece": (calibration_curve(labels_a, scores_a, cfg.eval.calibration_bins)
                if len(labels_a) else float("nan")),
    }
    log.info("Celeb-DF cross-dataset: ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f (n=%d real=%d fake=%d)",
             metrics["acc"], metrics["auc"], metrics["f1"], metrics["eer"],
             metrics["n_videos"], metrics["n_real"], metrics["n_fake"])
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Cross-set evaluation for temporal-region v2")
    ap.add_argument("--checkpoint", required=True, help="path to seed best.pt")
    ap.add_argument("--ffpp-dir", default=None)
    ap.add_argument("--celebdf-dir", default=None)
    ap.add_argument("--quick", type=int, default=0,
                    help="cap raw Celeb-DF videos per class (0 = unlimited)")
    ap.add_argument("--amp-dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/eval_cross_set.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("cross-set evaluation requires CUDA")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        log.warning("bf16 not natively supported on this GPU; falling back to fp16")
        args.amp_dtype = "fp16"
    enable_tf32(True)

    agent, cfg, ckpt = _load_agent(Path(args.checkpoint), args.device)
    cfg.train.amp_dtype = args.amp_dtype
    cfg.train.device = args.device
    if args.ffpp_dir:
        cfg.data.ffpp_dir = args.ffpp_dir
    cfg.resolve_paths()

    log.info("--- unseen methods (FF++ held-out test, per-method) ---")
    test_index = load_index(cfg, "test")
    test_ds = SequenceMaskDataset(test_index, cfg)
    test_loader = _make_loader(test_ds, 8, False, args.workers, 2)
    overall, videos, labels, scores = _collect(agent, test_loader, cfg)
    per_method = _per_method(videos, labels, scores, cfg)
    log.info("FF++ unseen-video test: ACC=%.4f AUC=%.4f F1=%.4f regionIoU=%.4f regionHit=%.4f (n=%d)",
             overall["acc"], overall["auc"], overall["f1"],
             overall.get("region_iou", float("nan")), overall.get("region_hit", float("nan")),
             overall["n"])

    log.info("--- unseen dataset (raw Celeb-DF, verdict only) ---")
    celebdf_metrics, celebdf_status = None, "not run: no --celebdf-dir"
    celebdf_root = Path(args.celebdf_dir) if args.celebdf_dir else None
    if celebdf_root is None:
        celebdf_root = Path(cfg.data.celebdf_dir)
    if celebdf_root and celebdf_root.exists():
        try:
            celebdf_metrics = _celebdf(agent, cfg, celebdf_root, args.device,
                                       quick=args.quick)
            celebdf_status = "ok"
        except Exception as exc:
            log.error("Celeb-DF evaluation failed: %s", exc)
            celebdf_status = f"failed: {exc}"
    else:
        celebdf_status = f"not run: no Celeb-DF tree at {celebdf_root}"

    per_method_aucs = [m["auc"] for m in per_method.values()
                       if isinstance(m.get("auc"), (int, float)) and m["auc"] == m["auc"]]
    summary = {
        "overall_test_auc": overall.get("auc"),
        "overall_test_acc": overall.get("acc"),
        "overall_test_region_iou": overall.get("region_iou"),
        "per_method_mean_auc": round(float(np.mean(per_method_aucs)), 4) if per_method_aucs else None,
        "per_method_min_auc": round(float(np.min(per_method_aucs)), 4) if per_method_aucs else None,
        "per_method_n": len(per_method),
        "celebdf_auc": celebdf_metrics.get("auc") if celebdf_metrics else None,
        "celebdf_acc": celebdf_metrics.get("acc") if celebdf_metrics else None,
        "celebdf_status": celebdf_status,
    }
    report = {
        "checkpoint": args.checkpoint,
        "format": _CHECKPOINT_FORMAT,
        "verdict": _VERDICT,
        "epoch": ckpt.get("epoch"),
        "overall_test": overall,
        "per_method": per_method,
        "celebdf": celebdf_metrics,
        "celebdf_status": celebdf_status,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=True, default=str), encoding="utf-8")
    log.info("cross-set report saved to %s", out)
    print(json.dumps(report, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
