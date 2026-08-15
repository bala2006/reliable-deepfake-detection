"""Build the FF++ manifest consumed by Docker training.

Run from the repository root on Windows:
    python tools/build_ffpp_manifest.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path("data/FaceForensics++")
COMPRESSION = "c23"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
# Keep 32 video frames at inference. This cap only selects diverse,
# mask-supervised frames for each source clip in the training manifest.
FRAMES_PER_VIDEO = 8
INFERENCE_FRAMES_PER_VIDEO = 32
SAMPLE_TRAIN = 2400
SAMPLE_VAL = 600
SAMPLE_TEST = 600
# Extra, previously unused clips are marked train-only. They enlarge the
# continuation dataset without changing the established validation/test videos.
EXTRA_REAL_VIDEOS_PER_ORIGIN = 100
EXTRA_FAKE_VIDEOS_PER_METHOD = 50


def _dirs(path: Path) -> list[Path]:
    return sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)


def _append_frames(items: list[dict], video: Path, label: int, base: Path,
                   source: str, partition: str | None = None,
                   limit: int | None = None) -> int:
    frames = sorted(video.glob("*.png"), key=lambda p: p.name)
    if not frames:
        return 0
    step_f = max(1, len(frames) // FRAMES_PER_VIDEO)
    selected = frames[::step_f][:FRAMES_PER_VIDEO]
    if limit is not None:
        selected = selected[:max(0, limit)]
    for frame in selected:
        mask = base / "masks" / video.name / frame.name
        landmark = base / "landmarks" / video.name / (frame.stem + ".npy")
        item = {
            "label": label,
            "video": f"{source}/{video.name}",
            "frame": frame.stem,
            "face": frame.relative_to(ROOT).as_posix(),
            "mask": mask.relative_to(ROOT).as_posix() if mask.exists() else None,
            "landmark": landmark.relative_to(ROOT).as_posix() if landmark.exists() else None,
        }
        if partition:
            item["partition"] = partition
        items.append(item)
    return len(selected)


def _collect(items: list[dict], videos: list[Path], label: int,
             base: Path, source: str, cap: int) -> None:
    need = cap - sum(1 for item in items
                     if item["label"] == label
                     and item["video"].startswith(source + "/"))
    if need <= 0 or not videos:
        return
    n_videos = max(1, math.ceil(need / FRAMES_PER_VIDEO))
    step_v = max(1, len(videos) // n_videos)
    for video in videos[::step_v]:
        need -= _append_frames(items, video, label, base, source, limit=need)
        if need <= 0:
            return


def _collect_extra_train_videos(items: list[dict], videos: list[Path], label: int,
                                base: Path, source: str, count: int) -> int:
    existing = {item["video"] for item in items}
    available = [
        video for video in videos
        if f"{source}/{video.name}" not in existing
        and len(list(video.glob("*.png"))) >= FRAMES_PER_VIDEO
    ]
    if not available or count <= 0:
        return 0
    requested = min(count, len(available))
    step = max(1, len(available) // requested)
    chosen = available[::step][:requested]
    for video in chosen:
        _append_frames(items, video, label, base, source, partition="train")
    return len(chosen)


def main() -> None:
    per_label = SAMPLE_TRAIN + SAMPLE_VAL + SAMPLE_TEST
    items: list[dict] = []
    real_sources = []
    for origin in ("youtube", "actors"):
        base = ROOT / "original_sequences" / origin / COMPRESSION
        frames = base / "frames"
        if frames.is_dir():
            real_sources.append((origin, base, _dirs(frames)))
    real_cap = max(1, math.ceil(per_label / max(1, len(real_sources))))
    for origin, base, videos in real_sources:
        _collect(items, videos, 0, base, origin, real_cap)

    fake_sources = []
    fake_cap = max(1, math.ceil(per_label / len(METHODS)))
    for method in METHODS:
        base = ROOT / "manipulated_sequences" / method / COMPRESSION
        frames = base / "frames"
        if frames.is_dir():
            videos = _dirs(frames)
            fake_sources.append((method, base, videos))
            _collect(items, videos, 1, base, method, fake_cap)

    extra_real = sum(
        _collect_extra_train_videos(items, videos, 0, base, origin,
                                    EXTRA_REAL_VIDEOS_PER_ORIGIN)
        for origin, base, videos in real_sources
    )
    extra_fake = sum(
        _collect_extra_train_videos(items, videos, 1, base, method,
                                    EXTRA_FAKE_VIDEOS_PER_METHOD)
        for method, base, videos in fake_sources
    )
    payload = {
        "schema_version": 2,
        "compression": COMPRESSION,
        "methods": list(METHODS),
        "manifest_frames_per_video": FRAMES_PER_VIDEO,
        "inference_frames_per_video": INFERENCE_FRAMES_PER_VIDEO,
        "sample_train": SAMPLE_TRAIN,
        "sample_val": SAMPLE_VAL,
        "sample_test": SAMPLE_TEST,
        "continuation_train_only_videos": {
            "real": extra_real,
            "fake": extra_fake,
        },
        "items": items,
    }
    out = ROOT / ".rlroinet_index.json"
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(items)} entries to {out}; added {extra_real} real and "
          f"{extra_fake} fake train-only videos")


if __name__ == "__main__":
    main()
