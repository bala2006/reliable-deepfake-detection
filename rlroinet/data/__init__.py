"""Supervised v2 data factories."""

from __future__ import annotations

from pathlib import Path

from .synthetic import (
    FaceLayout, label_roi, roi_mask, REGION_NAMES,
    PERIOCULAR, JAWLINE, MOUTH, HAIRLINE,
)
from .video_utils import detect_face, layout_from_face, default_layout
from .maskdata import (
    MaskFaceSample, MaskDataset, index_gt_faces, index_ffpp_masks,
    index_raw_celebdf_videos, split_index,
)
from .celebdf import CelebDFDataset, ensure_celebdf, locate_celebdf_root
from .dfdcp import index_dfdcp_faces
from .ffpp import FFPPDataset

__all__ = [
    "FaceLayout", "label_roi", "roi_mask", "REGION_NAMES",
    "PERIOCULAR", "JAWLINE", "MOUTH", "HAIRLINE",
    "detect_face", "layout_from_face", "default_layout",
    "MaskFaceSample", "MaskDataset", "index_gt_faces", "index_ffpp_masks",
    "index_raw_celebdf_videos", "index_dfdcp_faces", "split_index",
    "CelebDFDataset", "FFPPDataset", "ensure_celebdf", "locate_celebdf_root",
    "load_index", "make_dataset", "make_mask_dataset",
]



_INDEX_CACHE = {}


def _index_cache_key(cfg):
    """Return a stable key for the source pool used by all three splits."""
    d = cfg.data
    return (
        d.source,
        str(Path(d.mask_dir).resolve()),
        str(Path(d.ffpp_dir).resolve()),
        str(Path(d.celebdf_dir).resolve()),
        str(Path(d.dfdcp_dir).resolve()),
        d.compression,
        tuple(d.methods),
        int(d.frames_per_video),
        int(d.manifest_frames_per_video),
        int(d.sample_train),
        int(d.sample_val),
        int(d.sample_test),
    )


def load_index(cfg, split: str):
    """Build or reuse an aligned face/mask index for a supervised source."""
    key = _index_cache_key(cfg)
    if cfg.data.source == "dfdcp":
        if key not in _INDEX_CACHE:
            if not Path(cfg.data.dfdcp_dir).exists():
                raise FileNotFoundError(f"DFDCP data not found at {cfg.data.dfdcp_dir}")
            _INDEX_CACHE[key] = index_dfdcp_faces(Path(cfg.data.dfdcp_dir), cfg)
        return _INDEX_CACHE[key][split]
    if key not in _INDEX_CACHE:
        if cfg.data.source == "deepfakebench":
            index = index_gt_faces(Path(cfg.data.mask_dir),
                                   frames_per_video=cfg.data.frames_per_video)
        elif cfg.data.source == "ffpp":
            if not Path(cfg.data.ffpp_dir).exists():
                raise FileNotFoundError(f"FF++ data not found at {cfg.data.ffpp_dir}")
            index = index_ffpp_masks(
                Path(cfg.data.ffpp_dir), cfg,
                frames_per_video=cfg.data.manifest_frames_per_video,
            )
        elif cfg.data.source == "celebdf_raw":
            index = index_raw_celebdf_videos(Path(cfg.data.celebdf_dir))
        else:
            raise ValueError(f"unknown training source '{cfg.data.source}'")
        _INDEX_CACHE[key] = index
    else:
        index = _INDEX_CACHE[key]

    split_idx = split_index(
        index, cfg.data.sample_train, cfg.data.sample_test, cfg.data.seed,
        n_val=cfg.data.sample_val,
        stratify_methods=cfg.data.stratify_methods,
        ffpp_dir=Path(cfg.data.ffpp_dir),
        leakage_safe=cfg.data.leakage_safe_split,
    )
    return split_idx[split]


def make_dataset(cfg, split: str = "test", device: str = "cpu"):
    """Create the configured v2 dataset."""
    if cfg.data.source == "ffpp":
        if not Path(cfg.data.ffpp_dir).exists():
            raise FileNotFoundError(f"FF++ data not found at {cfg.data.ffpp_dir}")
        return FFPPDataset(cfg.data.ffpp_dir, cfg, split=split, device=device)
    return MaskDataset(load_index(cfg, split), cfg, device=device)


def make_mask_dataset(cfg, split: str = "train", device: str = "cpu") -> MaskDataset:
    """Aligned-face mask-supervised dataset (deepfakebench, ffpp, or celebdf_raw)."""
    return MaskDataset(load_index(cfg, split), cfg, device=device)
