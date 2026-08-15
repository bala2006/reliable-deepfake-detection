"""Single-video inference for supervised ROI-Net v2.

Decodes an uploaded video, samples frames, detects the face, and runs the
transfer-learning backbone + mask-head model per frame to produce:
  * a video-level verdict (probability of manipulation, configured robust pool of frame probs),
  * per-frame region predictions (forgery mask + primary ROI),
  * a transparent trace for the demo UI.

The output is a stable JSON object with a verdict, confidence, detections,
trace, and normalized bounding boxes for the website.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config, default_config
from .data import FaceLayout, detect_face, layout_from_face, REGION_NAMES
from .evaluate import _REGION_FORMATS, load_agent, load_region_agent
from .models import Agent


def aggregate_video_confidence(confidences: np.ndarray, cfg: Config) -> float:
    """Pool frame scores without allowing sparse evidence to vanish in a mean.

    The robust pool is the midpoint of the ordinary mean and the mean of the
    highest-scoring frames. This is deliberately less sensitive than max and
    is used consistently by evaluation and production inference.
    """
    scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if not len(scores):
        raise ValueError("at least one frame score is required")
    mean_score = float(scores.mean())
    if cfg.eval.video_aggregation == "mean":
        return mean_score
    k = max(1, int(np.ceil(len(scores) * float(cfg.eval.top_fraction))))
    top_score = float(np.sort(scores)[-k:].mean())
    return float(0.5 * mean_score + 0.5 * top_score)


def video_decision(confidences: np.ndarray, cfg: Config) -> tuple[str, float, dict]:
    """Return ``(verdict, score, evidence)`` using a conservative review band."""
    scores = np.asarray(confidences, dtype=np.float32).reshape(-1)
    score = aggregate_video_confidence(scores, cfg)
    high_fraction = float((scores >= cfg.eval.fake_threshold).mean())
    if cfg.eval.review_enabled:
        if score >= cfg.eval.fake_threshold and high_fraction >= cfg.eval.min_fake_fraction:
            verdict = "FAKE"
        elif score <= cfg.eval.real_threshold:
            verdict = "REAL"
        else:
            verdict = "REVIEW"
    else:
        verdict = "FAKE" if score >= cfg.eval.threshold else "REAL"
    evidence = {
        "aggregation": cfg.eval.video_aggregation,
        "mean_frame_confidence": round(float(scores.mean()), 4),
        "top_frame_confidence": round(float(np.max(scores)), 4),
        "high_confidence_frame_fraction": round(high_fraction, 4),
        "review_band": [cfg.eval.real_threshold, cfg.eval.fake_threshold],
    }
    return verdict, score, evidence

_ROI_ORDER = (1, 2, 3, 4)  # PERIOCULAR, JAWLINE, MOUTH, HAIRLINE


def bbox_from_loc(cx: float, cy: float, scale: float, w: int, h: int) -> tuple:
    """Pixel bounding box for a region at normalized (cx, cy) covering ``scale``."""
    side = max(8, int(round(scale * min(h, w))))
    x_c = cx * (w - 1)
    y_c = cy * (h - 1)
    half = side // 2
    x0 = int(np.clip(x_c - half, 0, w - side))
    y0 = int(np.clip(y_c - half, 0, h - side))
    return x0, y0, x0 + side, y0 + side


def _face_crop(frame_rgb: np.ndarray, size: int) -> tuple:
    """Detect the largest face; return (crop tensor, FaceLayout, crop bbox in frame)."""
    H, W = frame_rgb.shape[:2]
    # OpenCV detectors in data.video_utils consume BGR, while the model path
    # and returned crop are RGB. Keep the conversion local to the detector call.
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    bbox = detect_face(frame_bgr)
    if bbox is not None:
        x, y, w, h = bbox
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = frame_rgb[y0:y1, x0:x1]
        layout = layout_from_face(bbox, H, W)
        box_in_frame = (x0, y0, x1, y1)
    else:
        crop = frame_rgb
        layout = FaceLayout()
        box_in_frame = (0, 0, W, H)
    crop = cv2.resize(crop, (size, size))
    face = torch.from_numpy(crop).float().div_(255.0).permute(2, 0, 1)
    return face, layout, box_in_frame


@torch.no_grad()
def predict_frames(agent: Agent, frames: torch.Tensor, cfg: Config,
                   device: str = "cpu") -> List[Dict]:
    """Run the model on each sampled frame; return per-frame analysis dicts."""
    agent.eval()
    size = cfg.data.face_size
    out_frames = []
    for t in range(frames.shape[0]):
        frame_rgb = (frames[t].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        face, layout, box_in_frame = _face_crop(frame_rgb, size)
        x = face.unsqueeze(0).to(device)
        with torch.amp.autocast(
            "cuda",
            enabled=cfg.train.amp and str(device).startswith("cuda") and torch.cuda.is_available(),
            dtype=torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16,
        ):
            o = agent(x)
            if getattr(agent, "classifier_trained", False) and "cls_logits" in o:
                conf = float(torch.sigmoid(o["cls_logits"]).squeeze(0).item())
            elif "verdict_prob" in o:
                conf = float(o["verdict_prob"].squeeze(0).item())
            else:
                conf = float(torch.sigmoid(o["cls_logits"]).squeeze(0).item())
            mask_map = F.interpolate(o["mask"], size=(size, size), mode="bilinear",
                                     align_corners=False).squeeze(0).squeeze(0)
            region_map = F.interpolate(o["region"], size=(size, size), mode="bilinear",
                                       align_corners=False).squeeze(0)
        mask_np = mask_map.detach().cpu().numpy()
        # primary ROI: the region with the strongest positive mask / salience
        roi_id = 0
        if mask_np.max() > 0.5:
            salience = [float(region_map[r - 1].detach().cpu().max()) for r in _ROI_ORDER]
            roi_id = _ROI_ORDER[int(np.argmax(salience))]
        # bbox: bounding box around the predicted mask (fallback to the ROI center)
        ys, xs = np.where(mask_np > 0.5)
        if len(xs):
            cx, cy = float(xs.mean()) / size, float(ys.mean()) / size
            scale = float(max(0.3, (xs.max() - xs.min()) / size))
        else:
            L = layout
            anchors = {1: L.eye_left, 2: (L.fx, L.fy), 3: L.mouth, 4: (L.fx, L.hairline_y)}
            cx, cy = anchors.get(roi_id, (L.fx, L.fy))
            scale = 0.35
        out_frames.append({
            "frame_index": t,
            "confidence": round(conf, 4),
            "roi": REGION_NAMES.get(roi_id, "OTHER"),
            "cx": round(float(cx), 3),
            "cy": round(float(cy), 3),
            "scale": round(scale, 3),
            "mask": mask_np,                     # (size,size) float map
            "layout": layout,
            "box_in_frame": box_in_frame,
            "flagged": conf >= cfg.eval.fake_threshold,
        })
    return out_frames


def predict(agent: Agent, frames: torch.Tensor, cfg: Config,
            device: str = "cuda") -> Dict:
    """frames: (T, 3, H, W) float [0,1]. Video-level verdict + trace."""
    analysis = predict_frames(agent, frames, cfg, device)
    if not analysis:
        raise ValueError("no frames available for inference")
    confs = np.array([a["confidence"] for a in analysis], dtype=np.float32)
    verdict, confidence, evidence = video_decision(confs, cfg)
    return {
        "verdict": verdict,
        "label": verdict,
        "manipulated": verdict == "FAKE",
        "review_required": verdict == "REVIEW",
        "confidence": round(confidence, 4),
        "confidence_type": "robust video probability of manipulation",
        "frames_per_video": int(frames.shape[0]),
        "glimpses": len(analysis),
        "compute_savings": None,
        "decision_evidence": evidence,
        "trace": [{"step": i + 1, "frame_index": a["frame_index"], "roi": a["roi"],
                   "confidence": a["confidence"], "flagged": a["flagged"]}
                  for i, a in enumerate(analysis)],
    }


def analyze(agent: Agent, frames: torch.Tensor, cfg: Config,
            sample_idx: Optional[list] = None, fps: Optional[float] = None,
            disp_w: Optional[int] = None, disp_h: Optional[int] = None,
            device: Optional[str] = None) -> Dict:
    """Like predict, plus per-frame detections with normalized bboxes for the UI."""
    run_device = device or cfg.train.device
    analysis = predict_frames(agent, frames, cfg, run_device)
    confs = np.array([a["confidence"] for a in analysis], dtype=np.float32)
    verdict, confidence, evidence = video_decision(confs, cfg)
    w, h = disp_w or frames.shape[-1], disp_h or frames.shape[-2]

    detections = []
    trace = []
    for i, a in enumerate(analysis):
        vframe = int(sample_idx[i]) if sample_idx else i
        tsec = round(vframe / fps, 3) if fps else round(float(vframe), 3)
        x0, y0, x1, y1 = bbox_from_loc(a["cx"], a["cy"], a["scale"], w, h)
        d = {
            "step": i + 1, "frame_index": a["frame_index"], "video_frame": vframe,
            "time_sec": tsec, "roi": a["roi"], "confidence": a["confidence"],
            "cx": a["cx"], "cy": a["cy"], "scale": a["scale"],
            "flagged": a["flagged"],
            "bbox": [round(x0 / w, 4), round(y0 / h, 4),
                     round(x1 / w, 4), round(y1 / h, 4)],
        }
        trace.append(d)
        if a["flagged"]:
            detections.append(d)

    if verdict == "FAKE" and not detections:
        top = sorted(trace, key=lambda e: e["confidence"], reverse=True)[:3]
        detections = [dict(e, flagged=True) for e in top]

    return {
        "verdict": verdict, "label": verdict, "manipulated": verdict == "FAKE",
        "review_required": verdict == "REVIEW",
        "confidence": round(confidence, 4),
        "confidence_type": "robust video probability of manipulation",
        "decision_evidence": evidence,
        "glimpses": len(analysis),
        "frames_per_video": int(frames.shape[0]),
        "compute_savings": None,
        "fps": fps,
        "duration_sec": (round(sample_idx[-1] / fps, 2) if fps and sample_idx else None),
        "detections": detections,
        "trace": trace,
    }


def read_video_frames(path: str | Path, cfg: Config) -> tuple[torch.Tensor, np.ndarray, float, int]:
    """Read only uniformly sampled frames and reject clips over the v2 limit."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not decode video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or not np.isfinite(fps):
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(np.ceil(cfg.data.max_video_duration_seconds * fps)) + 1
    if total > 0 and total / fps > cfg.data.max_video_duration_seconds + 0.05:
        cap.release()
        raise ValueError(f"video must be {cfg.data.max_video_duration_seconds:.0f} seconds or shorter")
    sampled_idx = []
    if total <= 0:
        frames = []
        while len(frames) <= max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if len(frames) > max_frames:
            raise ValueError(f"video must be {cfg.data.max_video_duration_seconds:.0f} seconds or shorter")
        total = len(frames)
        if not total:
            raise ValueError("video contained no frames")
        idx = np.linspace(0, total - 1, cfg.data.frames_per_video, dtype=int)
        selected = [frames[int(i)] for i in idx]
        sampled_idx = idx.tolist()
    else:
        idx = np.linspace(0, total - 1, cfg.data.frames_per_video, dtype=int)
        wanted = set(int(i) for i in idx)
        selected = []
        pos = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if pos in wanted:
                selected.append(frame)
                sampled_idx.append(pos)
            pos += 1
        cap.release()
        if not selected:
            raise ValueError("video contained no decodable sampled frames")
    idx = np.asarray(sampled_idx, dtype=int)
    rgb = [cv2.cvtColor(cv2.resize(f, (cfg.data.face_size, cfg.data.face_size)), cv2.COLOR_BGR2RGB)
           for f in selected]
    tensor = torch.from_numpy(np.stack(rgb)).float().div_(255.0).permute(0, 3, 1, 2)
    return tensor, idx, fps, total


def main():
    ap = argparse.ArgumentParser(description="Run supervised ROI-Net v2 inference")
    ap.add_argument("--video", required=True, help="path to a video")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/final.pt")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"), choices=["cuda"])
    ap.add_argument("--out", default="outputs/prediction.json")
    args = ap.parse_args()

    cfg = default_config()
    cfg.train.device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("inference requires the Docker CUDA runtime with an RTX 4050")
    frames, sample_idx, fps, total = read_video_frames(args.video, cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ckpt.get("checkpoint_format") in _REGION_FORMATS:
        agent = load_region_agent(args.checkpoint, cfg, "cuda")
        cfg = agent.cfg
        cfg.train.device = "cuda"
    else:
        agent = load_agent(args.checkpoint, cfg, "cuda")
    result = analyze(agent, frames, cfg, sample_idx=sample_idx, fps=fps,
                     device="cuda")
    result["duration_sec"] = round(total / fps, 2) if fps else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()