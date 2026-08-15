"""Benchmark end-to-end eight-frame ROI-Net inference, including face detection.

This is intentionally inference-only. It accepts a supervised-v2 or region-head
checkpoint, falls back to CPU when CUDA is unavailable, prints a compact table,
and writes the same measurements to outputs/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlroinet.config import Config, default_config  # noqa: E402
from rlroinet.evaluate import _REGION_FORMATS, load_agent, load_region_agent  # noqa: E402
from rlroinet.predict import predict_frames, read_video_frames  # noqa: E402


def _effective_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        print("CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
    return "cpu"


def _load_checkpoint(checkpoint: Path, device: str):
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_cfg = raw.get("cfg") if isinstance(raw, dict) else None
    cfg = Config.from_dict(saved_cfg) if isinstance(saved_cfg, dict) else default_config()
    cfg.train.device = device
    cfg.data.frames_per_video = 8
    cfg.train.amp = bool(cfg.train.amp and device == "cuda")
    checkpoint_format = raw.get("checkpoint_format") if isinstance(raw, dict) else None
    if checkpoint_format in _REGION_FORMATS:
        agent = load_region_agent(checkpoint, cfg, device)
        cfg = agent.cfg
        cfg.train.device = device
        cfg.data.frames_per_video = 8
    else:
        agent = load_agent(checkpoint, cfg, device)
    return agent.eval(), cfg, checkpoint_format


def _synthetic_clip(size: int, seed: int) -> torch.Tensor:
    """Create a deterministic RGB clip when no source video is supplied."""
    rng = np.random.default_rng(seed)
    frames = rng.uniform(0.0, 1.0, (8, 3, size, size)).astype(np.float32)
    return torch.from_numpy(frames)


def _model_size_mb(agent) -> tuple[int, float]:
    parameters = sum(int(p.numel()) for p in agent.parameters())
    bytes_used = sum(int(p.numel()) * p.element_size() for p in agent.parameters())
    return parameters, bytes_used / (1024 * 1024)


def benchmark(checkpoint: Path, requested_device: str, video: Path | None,
              iterations: int, warmup: int, seed: int, output: Path) -> dict:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup must be non-negative")
    device = _effective_device(requested_device)
    agent, cfg, checkpoint_format = _load_checkpoint(checkpoint, device)
    if video is None:
        frames = _synthetic_clip(cfg.data.face_size, seed)
        input_source = "synthetic RGB clip"
    else:
        frames, _, _, _ = read_video_frames(video, cfg)
        if frames.shape[0] != 8:
            raise RuntimeError(f"expected an 8-frame clip, got {frames.shape[0]} frames")
        input_source = str(video.resolve())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            predict_frames(agent, frames, cfg, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(iterations):
            started = time.perf_counter()
            predict_frames(agent, frames, cfg, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)

    latency_ms = float(np.mean(latencies))
    parameters, model_size_mb = _model_size_mb(agent)
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_format": checkpoint_format,
        "device_requested": requested_device,
        "device_used": device,
        "input_source": input_source,
        "clip_frames": 8,
        "includes_face_detection": True,
        "iterations": iterations,
        "warmup_iterations": warmup,
        "latency_ms_per_clip": round(latency_ms, 3),
        "frames_per_second": round(8.0 / (latency_ms / 1000.0), 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
        if device == "cuda" else 0.0,
        "model_parameters": parameters,
        "model_size_mb": round(model_size_mb, 3),
        "checkpoint_size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 3),
        "target_latency_ms_rtx4050_fp16": 300.0,
        "target_met_on_this_run": bool(device == "cuda" and latency_ms <= 300.0),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--video", type=Path,
                        help="optional source video; otherwise use a deterministic synthetic 8-frame clip")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/diagnostics/latency_benchmark.json"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.video is not None and not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    result = benchmark(args.checkpoint, args.device, args.video, args.iterations,
                       args.warmup, args.seed, args.out)
    print("metric                         value")
    print("-----------------------------  ----------------")
    print(f"device                         {result['device_used']}")
    print(f"checkpoint format              {result['checkpoint_format']}")
    print(f"latency / 8-frame clip (ms)   {result['latency_ms_per_clip']:.3f}")
    print(f"frames / second                {result['frames_per_second']:.3f}")
    print(f"peak VRAM (MB)                 {result['peak_vram_mb']:.3f}")
    print(f"model size (MB, parameters)    {result['model_size_mb']:.3f}")
    print(f"checkpoint size (MB)           {result['checkpoint_size_mb']:.3f}")
    print(f"JSON                           {args.out.resolve()}")


if __name__ == "__main__":
    main()
