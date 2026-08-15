"""Supervised v2 evaluation for classification and mask localization."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from .config import Config, default_config
from .data import REGION_NAMES, make_dataset, make_mask_dataset, roi_mask
from .models import Agent
from .precision import autocast as _autocast

log = logging.getLogger("rlroinet.evaluate")

_REGION_FORMATS = ("region-head-v1", "region-head-v2", "adapter-region-v1")


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def calibration_curve(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_idx = np.clip(np.searchsorted(edges[1:-1], scores, side="left"), 0, bins - 1)
    ece, n = 0.0, max(1, len(labels))
    for b in range(bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


@torch.no_grad()
def collect_scores(agent: Agent, dataset, cfg: Config, device: str = "cuda",
                   video_level: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Collect classification scores without touching localization metrics."""
    agent.eval()
    samples = [dataset[i] for i in range(len(dataset))]
    labels, scores, groups = [], [], []
    faces = [s for s in samples if hasattr(s, "face")]
    videos = [s for s in samples if not hasattr(s, "face")]
    for sample in videos:
        labels.append(int(sample["label"]))
        scores.append(float(_video_result(agent, sample, cfg, device)))
        groups.append(str(sample.get("video", f"video-{len(groups)}")))
    chunk = 32
    while faces:
        batch = faces[:chunk]
        try:
            with _autocast(cfg, device):
                out = agent(torch.stack([s.face for s in batch]).to(device))
                if getattr(agent, "classifier_trained", False) and "cls_logits" in out:
                    batch_scores = torch.sigmoid(out["cls_logits"]).squeeze(-1).float().cpu().tolist()
                elif "verdict_prob" in out:
                    batch_scores = out["verdict_prob"].squeeze(-1).float().cpu().tolist()
                else:
                    batch_scores = torch.sigmoid(out["cls_logits"]).squeeze(-1).float().cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        for sample, score in zip(batch, batch_scores):
            labels.append(int(sample.label))
            scores.append(float(score))
            groups.append(str(getattr(sample, "video", f"face-{len(groups)}")))
        faces = faces[chunk:]
    if video_level and groups:
        from .predict import aggregate_video_confidence
        grouped = {}
        for label, score, group in zip(labels, scores, groups):
            grouped.setdefault(group, {"label": label, "scores": []})["scores"].append(score)
        labels = [row["label"] for row in grouped.values()]
        scores = [aggregate_video_confidence(np.asarray(row["scores"]), cfg)
                  for row in grouped.values()]
    return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32)


def calibrate_decision_policy(labels: np.ndarray, scores: np.ndarray,
                              target_fp: float = 0.01,
                              target_fn: float = 0.05) -> dict:
    """Choose conservative fake/real bounds on validation scores.

    The fake bound is the lowest threshold whose validation false-positive rate
    is within ``target_fp``. The real bound is the highest threshold whose
    confident-fake-as-real rate is within ``target_fn``. The review band is
    retained whenever these constraints conflict.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    real = labels == 0
    fake = labels == 1
    fake_candidates = [float(t) for t in candidates
                       if ((scores[real] >= t).mean() if real.any() else 0.0) <= target_fp]
    fake_threshold = min(fake_candidates) if fake_candidates else 1.0
    real_candidates = [float(t) for t in candidates
                       if ((scores[fake] <= t).mean() if fake.any() else 0.0) <= target_fn]
    real_threshold = max(real_candidates) if real_candidates else 0.0
    if real_threshold >= fake_threshold:
        real_threshold = max(0.0, fake_threshold - 0.05)
    return {
        "real_threshold": round(real_threshold, 6),
        "fake_threshold": round(fake_threshold, 6),
        "review_enabled": True,
        "video_aggregation": "robust",
        "target_fp": float(target_fp),
        "target_fn": float(target_fn),
        "validation_n": int(len(labels)),
    }


def _iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    union = float(np.logical_or(pred, gt).sum())
    return float(np.logical_and(pred, gt).sum()) / union if union else 1.0


def _face_result(agent: Agent, sample, cfg: Config, device: str):
    """Return score and localization data for one aligned face sample.

    Supports both the supervised ``Agent`` (``cls_logits``) and the head-only
    ``RegionAgent`` (``verdict_prob`` or its own trained ``cls_logits``). When
    the region agent has a trained classifier head, its logits win over the
    frozen verdict probability (that is the point of training the head).
    """
    x = sample.face.unsqueeze(0).to(device)
    with _autocast(cfg, device):
        out = agent(x)
        mask = F.interpolate(out["mask"], size=(cfg.data.face_size, cfg.data.face_size),
                             mode="bilinear", align_corners=False).squeeze().detach().cpu().numpy()
    if getattr(agent, "classifier_trained", False) and "cls_logits" in out:
        score = float(torch.sigmoid(out["cls_logits"]).squeeze().item())
    elif "verdict_prob" in out:
        score = float(out["verdict_prob"].squeeze().item())
    else:
        score = float(torch.sigmoid(out["cls_logits"]).squeeze().item())
    return score, mask, sample.mask.detach().cpu().numpy(), sample.layout


def _video_result(agent: Agent, sample, cfg: Config, device: str):
    """Score an FF++ video with the same robust pool used by the API."""
    from .predict import aggregate_video_confidence, predict_frames
    trace = predict_frames(agent, sample["video"], cfg, device=device)
    if not trace:
        raise RuntimeError("FF++ sample contained no decodable frames")
    return aggregate_video_confidence(
        np.asarray([row["confidence"] for row in trace], dtype=np.float32), cfg)


@torch.no_grad()
def evaluate(agent: Agent, dataset, cfg: Config, device: str = "cuda",
             video_level: bool = False) -> Dict:
    agent.eval()
    labels, scores, preds = [], [], []
    groups = []
    localization = []
    region_values = {name: [] for name in REGION_NAMES.values() if name != "OTHER"}
    region_hits = 0
    region_total = 0

    samples = [dataset[i] for i in range(len(dataset))]
    faces = [s for s in samples if hasattr(s, "face")]
    videos = [s for s in samples if not hasattr(s, "face")]

    for sample in videos:
        score = _video_result(agent, sample, cfg, device)
        labels.append(int(sample["label"]))
        scores.append(score)
        preds.append(int(score >= cfg.eval.threshold))
        groups.append(str(sample.get("video", f"video-{len(groups)}")))

    chunk = 32
    face_size = cfg.data.face_size
    while faces:
        batch = faces[:chunk]
        try:
            with _autocast(cfg, device):
                x = torch.stack([s.face for s in batch]).to(device)
                out = agent(x)
                mask_map = F.interpolate(
                    out["mask"], size=(face_size, face_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(1).detach().cpu().numpy()
                if getattr(agent, "classifier_trained", False) and "cls_logits" in out:
                    chunk_scores = torch.sigmoid(out["cls_logits"]).squeeze(-1).cpu().tolist()
                elif "verdict_prob" in out:
                    chunk_scores = out["verdict_prob"].squeeze(-1).cpu().tolist()
                else:
                    chunk_scores = torch.sigmoid(out["cls_logits"]).squeeze(-1).cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        for j, sample in enumerate(batch):
            score = float(chunk_scores[j])
            pred_mask = mask_map[j]
            gt_mask = sample.mask.detach().cpu().numpy()
            label = int(sample.label)
            if label == 1:
                localization.append(_iou(pred_mask, gt_mask))
                hit = bool(np.logical_and(pred_mask > 0.5, gt_mask > 0.5).any())
                region_hits += int(hit)
                region_total += 1
                for rid, roi in roi_mask(sample.layout, face_size).items():
                    name = REGION_NAMES[rid]
                    region_values[name].append(
                        _iou(pred_mask * roi, gt_mask * roi))
            labels.append(label)
            scores.append(score)
            preds.append(int(score >= cfg.eval.threshold))
            groups.append(str(getattr(sample, "video", f"face-{len(groups)}")))
        faces = faces[chunk:]

    if video_level and groups:
        from .predict import aggregate_video_confidence
        grouped = {}
        for label, score, group in zip(labels, scores, groups):
            grouped.setdefault(group, {"label": int(label), "scores": []})["scores"].append(score)
        labels = [row["label"] for row in grouped.values()]
        scores = [aggregate_video_confidence(np.asarray(row["scores"]), cfg)
                  for row in grouped.values()]
        preds = [int(score >= cfg.eval.threshold) for score in scores]

    labels_a, scores_a, preds_a = map(np.asarray, (labels, scores, preds))
    has_both = len(np.unique(labels_a)) > 1
    metrics = {
        "schema_version": cfg.schema_version,
        "source": cfg.data.source,
        "n": int(len(labels_a)),
        "acc": float((preds_a == labels_a).mean()) if len(labels_a) else float("nan"),
        "auc": float(roc_auc_score(labels_a, scores_a)) if has_both else float("nan"),
        "f1": float(f1_score(labels_a, preds_a, zero_division=0)) if has_both else float("nan"),
        "precision": float(precision_score(labels_a, preds_a, zero_division=0)) if has_both else float("nan"),
        "recall": float(recall_score(labels_a, preds_a, zero_division=0)) if has_both else float("nan"),
        "eer": eer(labels_a, scores_a),
        "ece": calibration_curve(labels_a, scores_a, cfg.eval.calibration_bins) if len(labels_a) else float("nan"),
        "fp_rate": float(((preds_a == 1) & (labels_a == 0)).mean()) if len(labels_a) else float("nan"),
        "fn_rate": float(((preds_a == 0) & (labels_a == 1)).mean()) if len(labels_a) else float("nan"),
        "video_level": bool(video_level),
    }
    if len(labels_a):
        reliable_fake = scores_a >= cfg.eval.fake_threshold
        reliable_real = scores_a <= cfg.eval.real_threshold
        reviewed = ~(reliable_fake | reliable_real) if cfg.eval.review_enabled else np.zeros_like(reliable_fake)
        certain = reliable_fake | reliable_real
        metrics.update({
            "review_rate": float(reviewed.mean()),
            "reliable_coverage": float(certain.mean()),
            "reliable_precision": float((labels_a[reliable_fake] == 1).mean()) if reliable_fake.any() else float("nan"),
            "reliable_recall": float((reliable_fake & (labels_a == 1)).sum() / max(1, (labels_a == 1).sum())),
            "confident_false_positive_rate": float((reliable_fake & (labels_a == 0)).sum() / max(1, (labels_a == 0).sum())),
            "confident_false_negative_rate": float((reliable_real & (labels_a == 1)).sum() / max(1, (labels_a == 1).sum())),
        })
    else:
        metrics.update({"review_rate": float("nan"), "reliable_coverage": float("nan"),
                        "reliable_precision": float("nan"), "reliable_recall": float("nan"),
                        "confident_false_positive_rate": float("nan"),
                        "confident_false_negative_rate": float("nan")})
    if localization:
        metrics["region_iou"] = float(np.mean(localization))
        metrics["region_hit"] = float(region_hits / max(1, region_total))
        metrics["per_roi_iou"] = {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in region_values.items()
        }
    else:
        metrics["region_iou"] = float("nan")
        metrics["region_hit"] = float("nan")
        metrics["per_roi_iou"] = {}
    return metrics


def save_report(metrics: List[Dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return path


def load_agent(checkpoint: str | Path, cfg: Config, device: str = "cuda") -> Agent:
    """Load only current supervised-v2 checkpoints and fail clearly on legacy files."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint is malformed; expected a supervised-v2 checkpoint object")
    if ckpt.get("checkpoint_format") == "region-head-v1":
        raise ValueError(
            "checkpoint is a region-head-v1 artifact; use load_region_agent() instead")
    if ckpt.get("checkpoint_format") != "supervised-v2":
        raise ValueError(
            "checkpoint is not a supervised-v2 artifact; remove the legacy checkpoint "
            "and run the Docker RTX 4050 training job")
    saved_cfg = ckpt.get("cfg")
    if not isinstance(saved_cfg, dict) or saved_cfg.get("schema_version") != 2:
        raise ValueError("checkpoint configuration is not supervised-v2")
    saved = Config.from_dict(saved_cfg)
    if not isinstance(ckpt.get("model"), dict) or not ckpt["model"]:
        raise ValueError("checkpoint is malformed; missing supervised-v2 model state")
    saved.train.device = device
    agent = Agent(saved).to(device)
    try:
        agent.load_state_dict(ckpt["model"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint model state is incompatible with its configuration") from exc
    agent.eval()
    return agent


def load_region_agent(checkpoint: str | Path, cfg: Config, device: str = "cuda"):
    """Load a region-head checkpoint into a :class:`RegionAgent`.

    Handles both ``region-head-v1`` (region head only, frozen verdict is the
    classifier) and ``region-head-v2`` (region head + trained classifier head).
    """
    from .region_agent import RegionAgent
    from .verdict import load_verdict
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("checkpoint_format") not in _REGION_FORMATS:
        raise ValueError("checkpoint is not a region-head artifact")
    saved_cfg = ckpt.get("cfg")
    if not isinstance(saved_cfg, dict) or saved_cfg.get("schema_version") != 2:
        raise ValueError("checkpoint configuration is not schema v2")
    saved = Config.from_dict(saved_cfg)
    saved.train.device = device
    if ckpt.get("checkpoint_format") == "adapter-region-v1":
        from .adapter_agent import AdapterRegionAgent
        verdict = load_verdict(ckpt["verdict"], device=device)
        agent = AdapterRegionAgent(
            saved, verdict, with_classifier="classifier_state" in ckpt).to(device)
        agent.adapter.load_state_dict(ckpt["adapter_state"])
        agent.region.load_state_dict(ckpt["region_state"])
        if agent.classifier is not None and ckpt.get("classifier_state"):
            agent.classifier.load_state_dict(ckpt["classifier_state"])
            agent.classifier_trained = True
        policy_path = Path(checkpoint).parent / "decision_policy.json"
        if policy_path.exists():
            try:
                policy = json.loads(policy_path.read_text(encoding="utf-8"))
                for name in ("real_threshold", "fake_threshold", "review_enabled",
                             "video_aggregation", "top_fraction", "min_fake_fraction"):
                    if name in policy:
                        setattr(agent.cfg.eval, name, policy[name])
            except (OSError, ValueError, TypeError) as exc:
                log.warning("ignoring invalid decision policy %s: %s", policy_path, exc)
        agent.eval()
        return agent
    verdict = load_verdict(ckpt["verdict"], device=device)
    agent = RegionAgent(saved, verdict,
                        with_classifier="classifier_state" in ckpt).to(device)
    agent.region.load_state_dict(ckpt["region_state"])
    if agent.classifier is not None and ckpt.get("classifier_state"):
        agent.classifier.load_state_dict(ckpt["classifier_state"])
        agent.classifier_trained = True
    policy_path = Path(checkpoint).parent / "decision_policy.json"
    if policy_path.exists():
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            for name in ("real_threshold", "fake_threshold", "review_enabled",
                         "video_aggregation", "top_fraction", "min_fake_fraction"):
                if name in policy:
                    setattr(agent.cfg.eval, name, policy[name])
        except (OSError, ValueError, TypeError) as exc:
            log.warning("ignoring invalid decision policy %s: %s", policy_path, exc)
    agent.eval()
    return agent


def main():
    ap = argparse.ArgumentParser(description="Evaluate a supervised ROI-Net v2 checkpoint")
    ap.add_argument("--source", choices=["deepfakebench", "ffpp", "celebdf_raw"], default=None)
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--celebdf-dir", default=None)
    ap.add_argument("--ffpp-dir", default=None)
    ap.add_argument("--checkpoint", default="outputs/checkpoints/final.pt")
    ap.add_argument("--out", default="outputs/eval_report.json")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    if args.source:
        cfg.data.source = args.source
    if args.mask_dir:
        cfg.data.mask_dir = args.mask_dir
    if args.celebdf_dir:
        cfg.data.celebdf_dir = args.celebdf_dir
    if args.ffpp_dir:
        cfg.data.ffpp_dir = args.ffpp_dir
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}; train the current v2 model first")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise SystemExit("evaluation requires the Docker CUDA runtime with an RTX 4050")
    cfg.resolve_paths()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_format = ckpt.get("checkpoint_format") if isinstance(ckpt, dict) else None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
        saved_cfg = Config.from_dict(ckpt["cfg"])
        if args.source is None:
            cfg.data.source = saved_cfg.data.source
        for name in ("sample_train", "sample_val", "sample_test", "frames_per_video",
                     "manifest_frames_per_video", "face_size", "seed", "compression",
                     "leakage_safe_split"):
            setattr(cfg.data, name, getattr(saved_cfg.data, name))
        if args.source is None:
            cfg.data.methods = saved_cfg.data.methods
        cfg.eval = saved_cfg.eval
    if ckpt_format in _REGION_FORMATS:
        dataset = make_mask_dataset(cfg, "test", device="cpu")
        agent = load_region_agent(args.checkpoint, cfg, args.device)
    else:
        dataset = make_dataset(cfg, "test", device="cpu")
        agent = load_agent(args.checkpoint, cfg, args.device)
    metrics = evaluate(agent, dataset, cfg, args.device)
    save_report([metrics], args.out)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
