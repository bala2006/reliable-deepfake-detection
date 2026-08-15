"""Frozen-backbone shootout on the FF++ held-out split (frame-based, no raw videos).

Scores each frozen verdict model (``honi05``, ``deepfakebench_xception``) on the
*same* video-disjoint held-out test split used by training/eval, using the
FF++ manifest (``.rlroinet_index.json``) so no raw videos are required.  This is
the evidence step that decides which backbone becomes the frozen signal for the
region head in Phase 2.

Outputs overall, per-method, and frame-level ACC / AUC / F1 / EER / ECE for each
model, plus the winner.  Run inside Docker on the RTX 4050:
    docker compose run --rm backbones
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .config import Config, default_config
from .data import load_index
from .data.maskdata import _read_png
from .predict import aggregate_video_confidence
from .verdict import _REGISTRY, load_verdict

log = logging.getLogger("rlroinet.benchmark_backbones")

# The on-disk manifest records these exact values; loading with anything else
# forces a slow directory walk that produces a different index.
MANIFEST_CONTRACT = {
    "sample_train": 2400,
    "sample_val": 600,
    "sample_test": 600,
    "manifest_frames_per_video": 8,
}


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


def _metrics(labels, scores, cfg: Config) -> dict:
    labels_a = np.asarray(labels, dtype=int)
    scores_a = np.asarray(scores, dtype=float)
    preds = (scores_a >= cfg.eval.threshold).astype(int)
    both = len(np.unique(labels_a)) > 1
    return {
        "n": int(len(labels_a)),
        "n_real": int((labels_a == 0).sum()),
        "n_fake": int((labels_a == 1).sum()),
        "acc": float((preds == labels_a).mean()) if len(labels_a) else float("nan"),
        "auc": float(roc_auc_score(labels_a, scores_a)) if both else float("nan"),
        "f1": float(f1_score(labels_a, preds, zero_division=0)) if both else float("nan"),
        "eer": _eer(labels_a, scores_a),
        "ece": _ece(labels_a, scores_a, cfg.eval.calibration_bins) if len(labels_a) else float("nan"),
    }


@torch.no_grad()
def _score_frames(model, faces: torch.Tensor, device: str, cfg: Config) -> tuple:
    """Score one video's face crops -> (per-frame probs, video-level prob)."""
    model.eval()
    x = faces.to(device)
    probs = []
    for i in range(0, x.shape[0], 8):
        probs.append(model.verdict(x[i:i + 8]))
    frame_probs = torch.cat(probs).float().cpu().numpy()
    video_prob = aggregate_video_confidence(frame_probs, cfg)
    return frame_probs, float(video_prob)


def _group_test_videos(cfg: Config, ffpp_dir: str) -> dict:
    """The deterministic video-disjoint held-out test split, grouped by video.

    Uses the exact same ``load_index(cfg, 'test')`` path as evaluation so the
    shootout measures the same videos the region head is later scored on.
    """
    items = load_index(cfg, "test")
    groups: dict = defaultdict(list)
    for item in items:
        groups[str(item["video"])].append(item)
    return {video: sorted(frames, key=lambda i: str(i.get("frame", "")))
            for video, frames in groups.items()}


def shootout(models, cfg: Config, device: str, ffpp_dir: str,
             frame_level: bool = True) -> dict:
    videos = _group_test_videos(cfg, ffpp_dir)
    results, winners = [], []
    for name in models:
        log.info("scoring frozen backbone '%s' on %d held-out FF++ videos ...",
                 name, len(videos))
        model = load_verdict(name, device=device)
        video_labels, video_scores, frame_labels, frame_scores = [], [], [], []
        per_method: dict = defaultdict(lambda: {"labels": [], "scores": []})
        for video, frames in videos.items():
            label = int(frames[0]["label"])
            method = str(video).split("/", 1)[0]
            tensors = []
            for item in frames:
                face_path = item.get("face")
                if not face_path or not Path(face_path).exists():
                    log.warning("missing face for %s frame %s", video, item.get("frame"))
                    continue
                tensors.append(_read_png(Path(face_path), cfg.data.face_size))
            if not tensors:
                continue
            faces = torch.stack(tensors)
            frame_probs, video_prob = _score_frames(model, faces, device, cfg)
            video_labels.append(label)
            video_scores.append(video_prob)
            per_method[method]["labels"].append(label)
            per_method[method]["scores"].append(video_prob)
            if frame_level:
                frame_labels.extend([label] * len(frame_probs))
                frame_scores.extend(frame_probs.tolist())

        result = _metrics(video_labels, video_scores, cfg)
        result.update({
            "model": name,
            "n_videos": int(len(video_labels)),
            "per_method": {
                m: _metrics(d["labels"], d["scores"], cfg)
                for m, d in sorted(per_method.items())
            },
        })
        if frame_level:
            result["frame_level"] = _metrics(frame_labels, frame_scores, cfg)
        results.append(result)
        log.info("  %s: ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f ECE=%.4f (n=%d)",
                 name, result["acc"], result["auc"], result["f1"],
                 result["eer"], result["ece"], result["n_videos"])
        for method, m in result["per_method"].items():
            log.info("    method %-16s AUC=%.4f (n=%d)", method, m["auc"], m["n"])
        winners.append((name, float(result["auc"])))
    return {"videos": len(videos), "winner": max(winners, key=lambda w: w[1])[0],
            "results": results}


def main():
    ap = argparse.ArgumentParser(description="Frozen-backbone shootout on FF++ held-out (frame-based)")
    ap.add_argument("--models", nargs="+", default=list(_REGISTRY))
    ap.add_argument("--ffpp-dir", default="data/FaceForensics++")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--out", default="outputs/backbone_shootout.json")
    ap.add_argument("--no-frame-level", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    cfg.data.source = "ffpp"
    cfg.data.ffpp_dir = args.ffpp_dir
    cfg.data.methods = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    cfg.data.compression = "c23"
    cfg.data.sample_train = MANIFEST_CONTRACT["sample_train"]
    cfg.data.sample_val = MANIFEST_CONTRACT["sample_val"]
    cfg.data.sample_test = MANIFEST_CONTRACT["sample_test"]
    cfg.data.manifest_frames_per_video = MANIFEST_CONTRACT["manifest_frames_per_video"]
    cfg.resolve_paths()

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("backbone shootout requires the Docker CUDA runtime with an RTX 4050")
    manifest = Path(args.ffpp_dir) / ".rlroinet_index.json"
    if not manifest.exists():
        raise SystemExit(f"FF++ manifest not found at {manifest}")

    report = shootout(args.models, cfg, args.device, args.ffpp_dir,
                      frame_level=not args.no_frame_level)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
