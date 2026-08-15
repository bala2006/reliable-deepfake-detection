from pathlib import Path

import cv2
import numpy as np
import pytest

from rlroinet.config import Config, default_config
from rlroinet.predict import read_video_frames


def _video(path: Path, frames: int, fps: float = 10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    for i in range(frames):
        writer.write(np.full((32, 32, 3), i, dtype=np.uint8))
    writer.release()


def test_current_config_round_trips_methods():
    cfg = default_config()
    cfg.data.methods = ("DeepFakes", "FaceSwap")
    restored = Config.from_dict(cfg.to_dict())
    assert restored.schema_version == 2
    assert restored.data.methods == cfg.data.methods


def test_legacy_config_is_rejected():
    with pytest.raises(ValueError, match="unsupported checkpoint configuration"):
        Config.from_dict({"data": {"use_official_test_split": True}})


def test_cli_reader_rejects_overlong_video(tmp_path):
    path = tmp_path / "long.mp4"
    _video(path, frames=35, fps=1.0)
    cfg = default_config()
    cfg.data.face_size = 32
    with pytest.raises(ValueError, match="30 seconds"):
        read_video_frames(path, cfg)


def test_cli_reader_samples_valid_video(tmp_path):
    path = tmp_path / "short.mp4"
    _video(path, frames=8, fps=8.0)
    cfg = default_config()
    cfg.data.face_size = 32
    cfg.data.frames_per_video = 4
    frames, indices, fps, total = read_video_frames(path, cfg)
    assert frames.shape == (4, 3, 32, 32)
    assert len(indices) == 4
    assert fps > 0 and total == 8
