"""FastAPI service exposing supervised ROI-Net v2 inference and the demo website.

POST /analyze  — upload a video, returns verdict + per-frame detections with
                 annotated ROI frames (base64 JPEG) + a URL to the original.
POST /predict  — upload a video, returns verdict + inspection trace (compat).
GET  /health   — container liveness + model readiness.
GET  /         — the demo website (index.html / demo.html / css / js).
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import threading
import time
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import default_config
from .data import FaceLayout, detect_face, layout_from_face
from .evaluate import _REGION_FORMATS, load_agent, load_region_agent
from .predict import analyze as analyze_video
from .predict import bbox_from_loc, predict as predict_video

CHECKPOINT = os.environ.get("CHECKPOINT", "outputs/checkpoints/best.pt")
DEVICE = os.environ.get("DEVICE", "cuda")
WEBSITE_DIR = os.environ.get("WEBSITE_DIR", "website")
UPLOADS_DIR = "uploads"
MAX_VIDEO_DURATION_SECONDS = float(os.environ.get("MAX_VIDEO_DURATION_SECONDS", "30"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))
UPLOAD_RETENTION_SECONDS = int(os.environ.get("UPLOAD_RETENTION_SECONDS", str(24 * 60 * 60)))
MAX_CONCURRENT_ANALYSES = max(1, int(os.environ.get("MAX_CONCURRENT_ANALYSES", "1")))

_ROI_COLORS = {
    "PERIOCULAR": (38, 78, 228),   # BGR red
    "JAWLINE": (20, 120, 244),     # BGR orange
    "MOUTH": (180, 40, 220),       # BGR pink
    "HAIRLINE": (210, 150, 30),    # BGR blue
    "OTHER": (60, 190, 90),        # BGR green
}

app = FastAPI(title="Supervised ROI-Net v2 Inference", version="2.0.0")

_cfg = default_config()
_cfg.train.device = DEVICE
_agent = None
_agent_lock = threading.Lock()
_inference_lock = threading.BoundedSemaphore(MAX_CONCURRENT_ANALYSES)


def _get_agent():
    global _agent, _cfg
    if not str(DEVICE).startswith("cuda") or not torch.cuda.is_available():
        raise HTTPException(status_code=503,
                            detail="CUDA RTX 4050 runtime is required for supervised v2 inference")
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                if not Path(CHECKPOINT).exists():
                    raise HTTPException(status_code=503,
                                        detail="model checkpoint not found — run training first")
                ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and ckpt.get("checkpoint_format") in _REGION_FORMATS:
                    _agent = load_region_agent(CHECKPOINT, _cfg, DEVICE)
                else:
                    _agent = load_agent(CHECKPOINT, _cfg, DEVICE)
                if hasattr(_agent, "cfg"):
                    _cfg = _agent.cfg
                    _cfg.train.device = DEVICE
    return _agent


def _downscale(frame, display_max: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= display_max:
        return frame
    sc = display_max / max(h, w)
    return cv2.resize(frame, (int(w * sc), int(h * sc)))


def _cleanup_stale_uploads() -> None:
    """Keep the local demo privacy-friendly and prevent unbounded disk growth."""
    root = Path(UPLOADS_DIR)
    if not root.exists():
        return
    cutoff = time.time() - UPLOAD_RETENTION_SECONDS
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _save_upload(file: UploadFile) -> tuple[Path, Path]:
    """Stream one upload to an isolated request directory with a strict limit."""
    _cleanup_stale_uploads()
    request_dir = Path(UPLOADS_DIR) / uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(file.filename or "").suffix.lower()
    if not suffix or len(suffix) > 16 or not suffix[1:].isalnum():
        suffix = ".video"
    source = request_dir / f"input{suffix}"
    written = 0
    try:
        with source.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"video is too large; maximum upload size is {limit_mb:.0f} MB",
                    )
                destination.write(chunk)
        if not written:
            raise HTTPException(status_code=400, detail="upload is empty")
        return request_dir, source
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise


def _transcode(source: Path, preview: Path) -> Path:
    """Make an isolated upload browser-playable with an H.264 preview copy.

    Celeb-DF clips are FMP4/MJPEG-in-MP4, which browsers cannot demux, so we
    remux+re-encode to H.264 (yuv420p) for the <video> player. Falls back to
    the raw file if ffmpeg is unavailable or fails.
    """
    preview.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             "-movflags", "+faststart", "-an", str(preview)],
            check=True, capture_output=True, timeout=120)
    except Exception:
        return source
    return preview if preview.exists() else source


def _probe_duration_seconds(source: Path) -> float | None:
    """Read duration with FFprobe before a fallback transcode when possible."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
            check=True, capture_output=True, text=True, timeout=20)
        duration = float(probe.stdout.strip())
        return duration if math.isfinite(duration) and duration >= 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _ensure_duration_limit(source: Path) -> None:
    duration = _probe_duration_seconds(source)
    if duration is not None and duration > MAX_VIDEO_DURATION_SECONDS + 0.05:
        raise HTTPException(status_code=413,
                            detail=f"video must be {MAX_VIDEO_DURATION_SECONDS:.0f} seconds or shorter")


def _decode_video(source: Path, target: int, frames_per_video: int,
                  display_max: int = 960):
    """Decode an uploaded video.

    Returns (tensor (T,3,H,W) float, FaceLayout, display dict, sample_idx,
    fps, total_frames). `display` maps each sampled index -> downscaled BGR
    frame (original aspect) used to draw ROI boxes. Memory stays bounded by
    only keeping the sampled frames.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="could not decode video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or not math.isfinite(fps):
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(math.ceil(MAX_VIDEO_DURATION_SECONDS * fps)) + 1
    if total > 0 and total / fps > MAX_VIDEO_DURATION_SECONDS + 0.05:
        cap.release()
        raise HTTPException(status_code=413,
                            detail=f"video must be {MAX_VIDEO_DURATION_SECONDS:.0f} seconds or shorter")

    # Some containers do not expose a frame count. Count a bounded first pass
    # so sampling stays uniform without retaining the whole video in memory.
    if total <= 0:
        cap.release()
        count_cap = cv2.VideoCapture(str(source))
        total = 0
        while total <= max_frames:
            ok, _ = count_cap.read()
            if not ok:
                break
            total += 1
        count_cap.release()
        if total > max_frames:
            raise HTTPException(status_code=413,
                                detail=f"video must be {MAX_VIDEO_DURATION_SECONDS:.0f} seconds or shorter")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="could not decode video")

    first = None
    sampled = []             # inference frames (RGB, target x target)
    sampled_idx = []
    display = {}             # index -> downscaled BGR display frame
    sample_idx = None

    idx = np.linspace(0, total - 1, frames_per_video, dtype=int)
    wanted = set(int(i) for i in idx)
    p = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        if p in wanted:
            sampled.append(cv2.cvtColor(cv2.resize(frame, (target, target)),
                                        cv2.COLOR_BGR2RGB))
            sampled_idx.append(p)
            display[p] = _downscale(frame, display_max).copy()
        p += 1
    cap.release()
    sample_idx = np.asarray(sampled_idx, dtype=int)

    if not sampled:
        raise HTTPException(status_code=400, detail="video contained no frames")

    tensor = torch.from_numpy(np.stack(sampled)).float().div_(255.0).permute(0, 3, 1, 2)

    layout = FaceLayout()
    if first is not None:
        bbox = detect_face(first)
        if bbox is not None:
            H, W = first.shape[:2]
            layout = layout_from_face(bbox, H, W)
    return tensor, layout, display, sample_idx, fps, total


def _decode_with_fallback(source: Path, target: int, frames_per_video: int,
                          preview_path: Path):
    """Decode the original stream, then use an FFmpeg-normalized fallback.

    The original is preferred for inference quality. If OpenCV cannot decode
    it but FFmpeg can, the normalized H.264 preview becomes both the playable
    asset and inference source.
    """
    _ensure_duration_limit(source)
    try:
        return _decode_video(source, target, frames_per_video), source
    except HTTPException as error:
        if error.status_code != 400:
            raise
        normalized = _transcode(source, preview_path)
        if normalized == source:
            raise error
        return _decode_video(normalized, target, frames_per_video), normalized


def _annotate(det: dict, display: dict, request_dir: Path) -> dict:
    """Write one annotated evidence frame and return its local static URL."""
    img = display.get(det["video_frame"])
    if img is None:
        img = display.get(det["frame_index"])
    if img is None:
        return {**det, "image_url": None}
    img = img.copy()
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox_from_loc(det["cx"], det["cy"], det["scale"], w, h)
    color = _ROI_COLORS.get(det["roi"], _ROI_COLORS["OTHER"])
    tag_y = max(14, y0 - 6)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    cv2.rectangle(img, (x0, y0), (x0 + 4, y0 + 4), color, -1)
    cv2.putText(img, f"{det['roi']} {det['confidence']:.2f}", (x0, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    evidence_dir = request_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    filename = f"frame_{int(det['step']):02d}.jpg"
    if not cv2.imwrite(str(evidence_dir / filename), img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        return {**det, "image_url": None}
    return {**det, "image_url": f"/{UPLOADS_DIR}/{request_dir.name}/evidence/{filename}"}


def _service_status() -> dict:
    checkpoint = Path(CHECKPOINT)
    cuda_available = bool(torch.cuda.is_available())
    ready = bool(str(DEVICE).startswith("cuda") and cuda_available and checkpoint.is_file())
    return {
        "status": "ready" if ready else "unavailable",
        "ready": ready,
        "model_loaded": _agent is not None,
        "cuda_available": cuda_available,
        "runtime": DEVICE,
        "checkpoint": CHECKPOINT,
        "checkpoint_exists": checkpoint.is_file(),
        "max_video_duration_seconds": MAX_VIDEO_DURATION_SECONDS,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "retention_seconds": UPLOAD_RETENTION_SECONDS,
    }


@app.get("/health")
def health():
    """Report whether the local inference service can accept a new analysis."""
    return _service_status()


def _acquire_inference_slot() -> None:
    if not _inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="another video is being analyzed; wait for it to finish and retry",
        )


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Compatibility endpoint; removes the upload after returning the compact result."""
    request_dir, source = _save_upload(file)
    acquired = False
    try:
        _acquire_inference_slot()
        acquired = True
        agent = _get_agent()
        decoded, _ = _decode_with_fallback(
            source, _cfg.data.face_size, _cfg.data.frames_per_video,
            source.parent / "normalized.mp4")
        frames, layout, _, _, _, _ = decoded
        result = predict_video(agent, frames, cfg=_cfg, device=DEVICE)
        result.update({
            "filename": file.filename,
            "checkpoint": Path(CHECKPOINT).name,
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return JSONResponse(content=result)
    finally:
        if acquired:
            _inference_lock.release()
        shutil.rmtree(request_dir, ignore_errors=True)


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    """Analyze one local video and retain its review assets until deleted or expired."""
    request_dir, source = _save_upload(file)
    acquired = False
    try:
        _acquire_inference_slot()
        acquired = True
        agent = _get_agent()
        decoded, inference_source = _decode_with_fallback(
            source, _cfg.data.face_size, _cfg.data.frames_per_video,
            request_dir / "preview.mp4")
        frames, layout, display, sample_idx, fps, total = decoded
        preview = inference_source if inference_source != source else _transcode(
            source, request_dir / "preview.mp4")
        disp_w = disp_h = None
        if display:
            first_key = next(iter(display))
            disp_w, disp_h = display[first_key].shape[1], display[first_key].shape[0]
        result = analyze_video(agent, frames, cfg=_cfg, sample_idx=sample_idx, fps=fps,
                               disp_w=disp_w, disp_h=disp_h, device=DEVICE)
        result["trace"] = [_annotate(d, display, request_dir) for d in result["trace"]]
        result["detections"] = [d for d in result["trace"] if d.get("flagged")]
        result.update({
            "request_id": request_dir.name,
            "video_url": f"/{UPLOADS_DIR}/{request_dir.name}/{preview.name}",
            "duration_sec": round(total / fps, 2) if fps else None,
            "filename": file.filename,
            "manipulated": result["verdict"] == "FAKE",
            "confidence_type": "robust video probability of manipulation",
            "checkpoint": Path(CHECKPOINT).name,
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "retention_seconds": UPLOAD_RETENTION_SECONDS,
        })
        return JSONResponse(content=result)
    except Exception:
        shutil.rmtree(request_dir, ignore_errors=True)
        raise
    finally:
        if acquired:
            _inference_lock.release()


@app.delete("/uploads/{request_id}", status_code=204)
def delete_upload(request_id: str):
    """Delete one review bundle immediately from this local service."""
    if not re.fullmatch(r"[0-9a-f]{32}", request_id):
        raise HTTPException(status_code=404, detail="analysis bundle not found")
    request_dir = Path(UPLOADS_DIR) / request_id
    if not request_dir.is_dir():
        raise HTTPException(status_code=404, detail="analysis bundle not found")
    shutil.rmtree(request_dir, ignore_errors=True)
    return Response(status_code=204)


Path(UPLOADS_DIR).mkdir(exist_ok=True)
app.mount(f"/{UPLOADS_DIR}", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Mounted last so API routes and explicitly deletable local review assets win.
app.mount("/", StaticFiles(directory=WEBSITE_DIR, html=True), name="website")
