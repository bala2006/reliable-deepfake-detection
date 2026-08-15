"""Generalization evaluation for the joint WHAT + WHERE detector.

Evaluates a trained region-head checkpoint across three levels of "never seen":

  1. **Unseen videos**  - FF++ held-out test faces (same methods, new videos).
  2. **Unseen methods** - per-method FF++ breakdown, including methods held out
     of training (cross-method memorization check).
  3. **Unseen dataset** - raw Celeb-DF videos (verdict-only, no GT masks there),
     the strongest real-world transfer test.

Run inside Docker on the RTX 4050:
    docker compose run --rm generalize
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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from .config import Config, default_config
from .data import make_mask_dataset
from .evaluate import (_REGION_FORMATS, calibration_curve, calibrate_decision_policy,
                       collect_scores, eer, evaluate, load_region_agent)
from .verdict import load_verdict

log = logging.getLogger("rlroinet.generalize")

CELEBDF_DIR = Path("data/celebdf")
FAKE_DIRS = ("celeb-synthesis",)
REAL_DIRS = ("celeb-real", "youtube-real")


def _per_method_metrics(agent, test_ds, cfg: Config, device: str) -> dict:
    """Group the **single** held-out test set by manipulation method.

    Filters the same video-level test dataset by each method's name (the
    ``video`` field is ``"<Method>/<video>"`` for fakes) so the per-method
    numbers come from the exact same held-out videos as the overall result.
    """
    from .data.maskdata import MaskDataset

    methods = sorted({d["video"].split("/")[0] for d in test_ds.index if d["label"] == 1})
    real_origins = {d["video"].split("/")[0] for d in test_ds.index if d["label"] == 0}
    out = {}
    for m in methods:
        idx = [d for d in test_ds.index if d["video"].split("/")[0] == m
               or (d["label"] == 0 and d["video"].split("/")[0] in real_origins)]
        if not idx:
            continue
        ds = MaskDataset(idx, cfg, device="cpu")
        metrics = evaluate(agent, ds, cfg, device=device, video_level=True)
        metrics["method"] = m
        out[m] = metrics
        log.info("method %-16s ACC=%.4f AUC=%.4f F1=%.4f regionIoU=%.4f regionHit=%.4f (n=%d)",
                 m, metrics["acc"], metrics["auc"], metrics["f1"],
                 metrics.get("region_iou", float("nan")),
                 metrics.get("region_hit", float("nan")), metrics["n"])
    return out


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
def _agent_score(agent, frames: torch.Tensor, device: str, cfg: Config) -> float:
    from .predict import aggregate_video_confidence
    agent.eval()
    x = frames.to(device)
    if x.shape[0] > 8:
        probs = []
        for i in range(0, x.shape[0], 8):
            o = agent(x[i:i + 8])
            probs.append(_classifier_prob(o))
        prob = torch.cat(probs)
    else:
        o = agent(x)
        prob = _classifier_prob(o)
    return aggregate_video_confidence(prob.detach().cpu().numpy(), cfg)


def _classifier_prob(out: dict) -> torch.Tensor:
    if "cls_logits" in out:
        return torch.sigmoid(out["cls_logits"]).squeeze(-1)
    return out["verdict_prob"]


def _celebdf_verdict(agent, cfg: Config, device: str, max_per_class: int = 0,
                     celebdf_dir: Path | None = None) -> dict:
    """Cross-dataset check on raw Celeb-DF videos (verdict only, no GT masks)."""
    videos = _list_videos(Path(celebdf_dir or CELEBDF_DIR))
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
        scores.append(_agent_score(agent, frames, device, cfg))
        labels.append(item["label"])
    labels_a = np.asarray(labels)
    scores_a = np.asarray(scores)
    preds_a = (scores_a >= cfg.eval.threshold).astype(int)
    has_both = len(np.unique(labels_a)) > 1
    metrics = {
        "dataset": "celebdf_raw",
        "n_videos": int(len(labels_a)),
        "n_real": int((labels_a == 0).sum()),
        "n_fake": int((labels_a == 1).sum()),
        "skipped": missing,
        "acc": float((preds_a == labels_a).mean()) if len(labels_a) else float("nan"),
        "auc": float(roc_auc_score(labels_a, scores_a)) if has_both else float("nan"),
        "f1": float(f1_score(labels_a, preds_a, zero_division=0)) if has_both else float("nan"),
        "eer": eer(labels_a, scores_a),
        "ece": calibration_curve(labels_a, scores_a, cfg.eval.calibration_bins) if len(labels_a) else float("nan"),
        "aggregation": cfg.eval.video_aggregation,
        "review_rate": float(((scores_a > cfg.eval.real_threshold) &
                               (scores_a < cfg.eval.fake_threshold)).mean()) if len(scores_a) else float("nan"),
    }
    log.info("cross-dataset Celeb-DF: ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f (n=%d)",
             metrics["acc"], metrics["auc"], metrics["f1"], metrics["eer"], metrics["n_videos"])
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Generalization evaluation (unseen videos/methods/dataset)")
    ap.add_argument("--source", choices=["deepfakebench", "ffpp", "celebdf_raw"], default=None)
    ap.add_argument("--ffpp-dir", default=None)
    ap.add_argument("--celebdf-dir", default=None)
    ap.add_argument("--methods", default=None,
                    help="comma-separated trained methods (used to label per-method breakdown)")
    ap.add_argument("--eval-methods", default=None,
                    help="comma-separated methods to build the test set from "
                         "(defaults to --methods; set to a held-out method for "
                         "cross-method evaluation)")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    ap.add_argument("--out", default="outputs/eval_generalization.json")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    ap.add_argument("--quick", type=int, default=0,
                    help="cap raw Celeb-DF videos per class for a fast cross-dataset estimate")
    ap.add_argument("--train-sample", type=int, default=None)
    ap.add_argument("--val-sample", type=int, default=None)
    ap.add_argument("--test-sample", type=int, default=None)
    ap.add_argument("--manifest-frames-per-video", type=int, default=None)
    ap.add_argument("--leakage-safe-split", action="store_true",
                    help="use official FF++ source-pair train/val/test lists")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrate REAL/FAKE bounds on the video-disjoint validation split")
    ap.add_argument("--target-fp", type=float, default=0.01,
                    help="maximum validation false-positive rate for the FAKE bound")
    ap.add_argument("--target-fn", type=float, default=0.05,
                    help="maximum validation false-negative rate for the REAL bound")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    if args.source:
        cfg.data.source = args.source
    if args.ffpp_dir:
        cfg.data.ffpp_dir = args.ffpp_dir
    if args.celebdf_dir:
        cfg.data.celebdf_dir = args.celebdf_dir
    if args.methods:
        cfg.data.methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("generalization eval requires the Docker CUDA runtime with an RTX 4050")
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    cfg.resolve_paths()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("checkpoint_format") not in _REGION_FORMATS:
        raise SystemExit("generalization eval expects a region-head checkpoint (train with --verdict)")

    # Use the checkpoint's split and decision policy so an evaluation run is
    # reproducible. Explicit CLI source/path/method overrides remain in force.
    saved_cfg_dict = ckpt.get("cfg")
    if isinstance(saved_cfg_dict, dict):
        saved_cfg = Config.from_dict(saved_cfg_dict)
        for name in ("sample_train", "sample_val", "sample_test", "frames_per_video",
                     "manifest_frames_per_video", "face_size", "seed", "compression",
                     "leakage_safe_split"):
            setattr(cfg.data, name, getattr(saved_cfg.data, name))
        cfg.eval = saved_cfg.eval

    for name, value in (("sample_train", args.train_sample),
                        ("sample_val", args.val_sample),
                        ("sample_test", args.test_sample),
                        ("manifest_frames_per_video", args.manifest_frames_per_video)):
        if value is not None:
            setattr(cfg.data, name, value)
    if args.leakage_safe_split:
        cfg.data.leakage_safe_split = True

    agent = load_region_agent(args.checkpoint, cfg, args.device)
    if hasattr(agent, "cfg"):
        cfg.eval = agent.cfg.eval
    saved_methods = ckpt.get("cfg", {}).get("data", {}).get("methods")
    trained_methods = list(saved_methods) if saved_methods else list(cfg.data.methods)
    report = {
        "checkpoint": args.checkpoint,
        "format": ckpt.get("checkpoint_format"),
        "verdict": ckpt.get("verdict"),
        "classifier_trained": bool(getattr(agent, "classifier_trained", False)),
        "trained_methods": trained_methods,
        "eval_methods": None,
        "decision_policy": {
            "aggregation": cfg.eval.video_aggregation,
            "real_threshold": cfg.eval.real_threshold,
            "fake_threshold": cfg.eval.fake_threshold,
            "review_enabled": cfg.eval.review_enabled,
            "min_fake_fraction": cfg.eval.min_fake_fraction,
        },
    }

    if args.eval_methods:
        report["eval_methods"] = [m.strip() for m in args.eval_methods.split(",") if m.strip()]
        log.info("evaluating held-out methods only: %s (trained on %s)",
                 report["eval_methods"], trained_methods)
        cfg.data.methods = tuple(report["eval_methods"])

    if args.calibrate:
        log.info("calibrating decision policy on video-disjoint validation split")
        val_ds = make_mask_dataset(cfg, "val", device="cpu")
        val_labels, val_scores = collect_scores(agent, val_ds, cfg, args.device,
                                                 video_level=True)
        policy = calibrate_decision_policy(val_labels, val_scores,
                                           target_fp=args.target_fp,
                                           target_fn=args.target_fn)
        cfg.eval.real_threshold = policy["real_threshold"]
        cfg.eval.fake_threshold = policy["fake_threshold"]
        cfg.eval.review_enabled = True
        policy["top_fraction"] = cfg.eval.top_fraction
        policy_path = Path(args.checkpoint).parent / "decision_policy.json"
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
        report["calibration"] = {"policy": policy, "path": str(policy_path)}
        report["decision_policy"].update({
            "real_threshold": cfg.eval.real_threshold,
            "fake_threshold": cfg.eval.fake_threshold,
            "review_enabled": cfg.eval.review_enabled,
        })

    log.info("--- unseen videos (FF++ held-out test) ---")
    test_ds = make_mask_dataset(cfg, "test", device="cpu")
    unseen = evaluate(agent, test_ds, cfg, device=args.device, video_level=True)
    unseen["level"] = "unseen_videos"
    report["unseen_videos"] = unseen
    log.info("unseen videos: ACC=%.4f AUC=%.4f F1=%.4f regionIoU=%.4f regionHit=%.4f (n=%d)",
             unseen["acc"], unseen["auc"], unseen["f1"],
             unseen.get("region_iou", float("nan")), unseen.get("region_hit", float("nan")),
             unseen["n"])

    log.info("--- unseen methods (per-method FF++ breakdown, same test videos) ---")
    report["per_method"] = _per_method_metrics(agent, test_ds, cfg, args.device)

    log.info("--- unseen dataset (raw Celeb-DF, verdict only) ---")
    if args.celebdf_dir:
        celebdf_root = Path(args.celebdf_dir)
    else:
        celebdf_root = CELEBDF_DIR
    if celebdf_root.exists():
        report["cross_dataset"] = _celebdf_verdict(agent, cfg, args.device,
                                                   max_per_class=args.quick,
                                                   celebdf_dir=celebdf_root)
    else:
        report["cross_dataset"] = None
        report["cross_dataset_status"] = f"not run: raw Celeb-DF not found at {celebdf_root}"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
