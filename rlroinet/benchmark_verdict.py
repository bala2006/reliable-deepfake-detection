"""Verdict-model shootout: measure frozen detectors on raw Celeb-DF videos.

Runs each candidate verdict model over uniformly sampled frames of the raw
``data/celebdf`` videos, aggregates per-frame probabilities to a per-video
verdict, and reports ACC / AUC / F1 / EER / ECE. The winner becomes the frozen
backbone for region-head training.

Run in Docker on the RTX 4050:
    docker compose run --rm verdict
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .config import Config, default_config
from .verdict import _REGISTRY, load_verdict

log = logging.getLogger("rlroinet.benchmark_verdict")

CELEBDF_DIR = Path("data/celebdf")
FAKE_DIRS = ("celeb-synthesis",)
REAL_DIRS = ("celeb-real", "youtube-real")


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.searchsorted(edges[1:-1], scores, side="left"), 0, bins - 1)
    ece, n = 0.0, max(1, len(labels))
    for b in range(bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


def _list_videos(root: Path):
    items = []
    for d in root.iterdir():
        if not d.is_dir() or d.name.lower() not in FAKE_DIRS + REAL_DIRS:
            continue
        label = 1 if d.name.lower() in FAKE_DIRS else 0
        for v in sorted(d.glob("*.mp4")):
            items.append({"video": v, "label": label})
    return items


def _face_crop(frame_bgr: np.ndarray, size: int):
    """Detect largest face and return an aligned crop, or a center crop fallback."""
    from .data.video_utils import detect_face
    H, W = frame_bgr.shape[:2]
    bbox = detect_face(frame_bgr)
    if bbox is not None:
        x, y, w, h = bbox
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = frame_bgr[y0:y1, x0:x1]
    else:
        crop = frame_bgr
    crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _sample_frames(path: Path, cfg: Config, time_budget_s: float = 8.0):
    import time as _time
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(np.ceil(cfg.data.max_video_duration_seconds * fps)) + 1
    if total > 0 and total / fps > cfg.data.max_video_duration_seconds + 0.05:
        cap.release()
        return None
    idx = np.linspace(0, max(total, 1) - 1, cfg.data.frames_per_video, dtype=int)
    wanted = set(int(i) for i in idx)
    frames, pos, deadline = [], 0, _time.time() + time_budget_s
    while True:
        if _time.time() > deadline:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if pos in wanted:
            frames.append(_face_crop(frame, cfg.data.face_size))
        pos += 1
    cap.release()
    if not frames:
        return None
    t = torch.from_numpy(np.stack(frames)).float().div_(255.0).permute(0, 3, 1, 2)
    return t


@torch.no_grad()
def _score_video(model, frames: torch.Tensor, device: str) -> float:
    model.eval()
    # Batch every frame of one video into a single forward pass (one GPU
    # transfer instead of frames_per_video separate ones).
    x = frames.to(device)
    if x.shape[0] > 8:
        probs = []
        for i in range(0, x.shape[0], 8):
            probs.append(model.verdict(x[i:i + 8]))
        prob = torch.cat(probs)
    else:
        prob = model.verdict(x)
    return float(prob.mean().item())


def benchmark(name: str, cfg: Config, device: str, max_per_class: int = 0) -> dict:
    model = load_verdict(name, device=device)
    videos = _list_videos(CELEBDF_DIR)
    if max_per_class > 0:
        real = [v for v in videos if v["label"] == 0]
        fake = [v for v in videos if v["label"] == 1]
        rng = np.random.default_rng(0)
        rng.shuffle(real)
        rng.shuffle(fake)
        videos = real[:max_per_class] + fake[:max_per_class]
    labels, scores = [], []
    missing = 0
    for item in videos:
        frames = _sample_frames(item["video"], cfg)
        if frames is None:
            missing += 1
            continue
        scores.append(_score_video(model, frames, device))
        labels.append(item["label"])
    labels_a = np.asarray(labels)
    scores_a = np.asarray(scores)
    preds_a = (scores_a >= cfg.eval.threshold).astype(int)
    has_both = len(np.unique(labels_a)) > 1
    metrics = {
        "model": name,
        "n_videos": int(len(labels_a)),
        "n_real": int((labels_a == 0).sum()),
        "n_fake": int((labels_a == 1).sum()),
        "skipped": missing,
        "acc": float((preds_a == labels_a).mean()) if len(labels_a) else float("nan"),
        "auc": float(roc_auc_score(labels_a, scores_a)) if has_both else float("nan"),
        "f1": float(f1_score(labels_a, preds_a, zero_division=0)) if has_both else float("nan"),
        "eer": _eer(labels_a, scores_a),
        "ece": _ece(labels_a, scores_a, cfg.eval.calibration_bins) if len(labels_a) else float("nan"),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Verdict-model shootout on raw Celeb-DF")
    ap.add_argument("--models", nargs="+", default=list(_REGISTRY),
                    help="verdict models to evaluate")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--out", default="outputs/verdict_shootout.json")
    ap.add_argument("--quick", type=int, default=0,
                    help="cap videos per class for a fast estimate (e.g. 100)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("verdict benchmark requires the Docker CUDA runtime with an RTX 4050")
    if not CELEBDF_DIR.exists():
        raise SystemExit(f"raw Celeb-DF not found at {CELEBDF_DIR}")

    results = []
    for name in args.models:
        log.info("benchmarking '%s' ...", name)
        m = benchmark(name, cfg, args.device, max_per_class=args.quick)
        results.append(m)
        log.info("  %s: ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f ECE=%.4f (n=%d)",
                 name, m["acc"], m["auc"], m["f1"], m["eer"], m["ece"], m["n_videos"])

    winner = max(results, key=lambda r: r["auc"]) if results else None
    out = {"winner": winner["model"] if winner else None,
           "target_acc": 0.95, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    if winner:
        log.info("winner: %s (AUC=%.4f, ACC=%.4f)", winner["model"], winner["auc"], winner["acc"])
        if winner["acc"] < 0.95:
            log.warning("no model cleared the 0.95 ACC target; consider more candidates")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
