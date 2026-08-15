from pathlib import Path

import cv2
import numpy as np
import torch

from rlroinet.config import default_config
from rlroinet.feature_cache import _index_fingerprint, build_feature_cache


class _DummyVerdict:
    name = "dummy"

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def infer(self, faces):
        return (torch.ones(faces.shape[0], 2, 1, 1),
                torch.full((faces.shape[0],), 0.5))


def _face(path: Path, size: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size, size, 3), 120, dtype=np.uint8))


def test_cached_fake_missing_mask_uses_pseudo_mask(tmp_path):
    face = tmp_path / "face.png"
    _face(face)
    checkpoint = tmp_path / "backbone.pt"
    checkpoint.write_bytes(b"weights-v1")
    cfg = default_config()
    cfg.data.face_size = 16
    index = [{
        "label": 1,
        "video": "fake/clip",
        "frame": "0",
        "face": face,
        "mask": tmp_path / "missing-mask.png",
        "landmark": None,
    }]

    cached = build_feature_cache(
        index, cfg, _DummyVerdict(checkpoint), str(tmp_path / "cache"),
        device="cpu", chunk=1, workers=1,
    )
    assert bool(cached["masks"][0].any())


def test_cache_fingerprint_changes_for_mask_and_backbone_content(tmp_path):
    face = tmp_path / "face.png"
    mask = tmp_path / "mask.png"
    checkpoint = tmp_path / "backbone.pt"
    _face(face)
    mask.write_bytes(b"mask-v1")
    checkpoint.write_bytes(b"weights-v1")
    index = [{"label": 1, "video": "fake/clip", "frame": "0",
              "face": face, "mask": mask}]

    initial = _index_fingerprint(index, backbone_checkpoint=checkpoint)
    mask.write_bytes(b"mask-v2")
    changed_mask = _index_fingerprint(index, backbone_checkpoint=checkpoint)
    checkpoint.write_bytes(b"weights-v2")
    changed_backbone = _index_fingerprint(index, backbone_checkpoint=checkpoint)

    assert changed_mask != initial
    assert changed_backbone != changed_mask
