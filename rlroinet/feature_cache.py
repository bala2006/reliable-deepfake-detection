"""Frozen-backbone feature cache: compute backbone features once, reuse across epochs.

The verdict backbone is frozen, so its feature map and verdict probability for a
given face are deterministic.  Caching them to disk lets every subsequent
training epoch skip the backbone forward entirely (the dominant cost of the
~430 s/epoch temporal runs).  The cache also stores the deterministic masks and
image-quality features, so cached epochs perform **zero** face decoding and zero
quality recomputation on the training thread — the GPU sees only precomputed
tensors.  The cache is keyed by a fingerprint of the exact index it was built
from, so a stale or incompatible cache is never silently reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from .config import Config
from .data.maskdata import MaskDataset
from .precision import amp_dtype

log = logging.getLogger("rlroinet.feature_cache")


_CHECKPOINT_FILENAMES = {
    "honi05": "best_model.pt",
    "deepfakebench_xception": "xception_best.pth",
}


def _path_identity(path) -> dict:
    """Return a content identity for a checkpoint or supervision file."""
    resolved = Path(path).expanduser().resolve()
    try:
        content = resolved.read_bytes()
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _checkpoint_identity(source) -> Optional[dict]:
    """Resolve a verdict model/object to a stable checkpoint identity."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        filename = _CHECKPOINT_FILENAMES.get(source)
        if filename is None:
            return {"name": source}
        return {"name": source,
                "checkpoint": _path_identity(Path("outputs/models") / filename)}
    if isinstance(source, Path):
        return {"checkpoint": _path_identity(source)}

    name = getattr(source, "name", source.__class__.__name__)
    checkpoint = getattr(source, "checkpoint_path", None)
    if checkpoint is None:
        weights_dir = getattr(source, "weights_dir", None)
        filename = _CHECKPOINT_FILENAMES.get(str(name))
        if weights_dir is not None and filename is not None:
            checkpoint = Path(weights_dir) / filename
    identity = {"name": str(name)}
    if checkpoint is not None:
        identity["checkpoint"] = _path_identity(checkpoint)
    return identity


def _file_content_digest(path, cache: dict) -> str:
    """Hash a supervision file, distinguishing missing files from empty paths."""
    if not path:
        return ""
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in cache:
        try:
            cache[resolved] = hashlib.sha256(Path(resolved).read_bytes()).hexdigest()
        except OSError:
            cache[resolved] = "missing"
    return cache[resolved]


def _index_fingerprint(index: List[dict], backbone_checkpoint=None) -> str:
    """Identity of index membership, masks/landmarks, and frozen weights."""
    file_digests = {}
    rows = []
    for item in index:
        mask = item.get("mask")
        landmark = item.get("landmark")
        rows.append((
            str(item.get("face")), int(item.get("label", 0)),
            str(item.get("video") or ""), str(item.get("frame") or ""),
            str(mask or ""), _file_content_digest(mask, file_digests),
            str(landmark or ""), _file_content_digest(landmark, file_digests),
        ))
    payload = {
        "rows": sorted(rows),
        "backbone": _checkpoint_identity(backbone_checkpoint),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_path(cache_dir, verdict_name: str, cfg: Config, index: List[dict],
                backbone_checkpoint=None, fingerprint: Optional[str] = None) -> Path:
    backbone = _checkpoint_identity(backbone_checkpoint or verdict_name)
    index_key = fingerprint or _index_fingerprint(index, backbone)
    key = {
        "verdict": verdict_name,
        "backbone": backbone,
        "face_size": cfg.data.face_size,
        "compression": cfg.data.compression,
        "methods": list(cfg.data.methods),
        "amp_dtype": str(getattr(cfg.train, "amp_dtype", "bf16")),
        "n": len(index),
        "fingerprint": index_key[:16],
    }
    name = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / f"features_{verdict_name}_{name}.pt"


@torch.no_grad()
def build_feature_cache(index: List[dict], cfg: Config, verdict, cache_dir: str,
                        device: str = "cuda", chunk: int = 64,
                        workers: int = 4) -> dict:
    """Compute and persist features, verdict probs, masks, and quality features.

    Returns a dict with:
      ``features``      (N,C,h,w) in the configured AMP dtype (bf16 default)
      ``verdict_probs`` (N,)
      ``masks``         (N,face_size,face_size) fp32
      ``quality``       (N,3) fp32 image-quality features (luma/std/sharpness)

    Face decoding and the backbone pass are parallelized across ``workers`` CPU
    threads, then run through the GPU in batches of ``chunk``.  Decoding is
    streamed in blocks so peak host RAM stays ~5 GB even for large splits
    (loading every train face at once would exceed a 16 GB Kaggle session).
    Rebuilding is roughly the cost of one training epoch; after that every
    epoch is heads-only.
    """
    size = cfg.data.face_size
    dt = amp_dtype(cfg)
    from .data.maskdata import _read_png

    # Reuse MaskDataset's mask/layout semantics so missing fake masks receive
    # the same pseudo-mask used by the non-cached training path.
    mask_dataset = MaskDataset(index, cfg, device="cpu", cache_size=0,
                               read_face=False)

    def load_one(position_item):
        position, item = position_item
        face = _read_png(Path(item["face"]), size)  # (3,H,W)
        luma = face.mean(dim=0)                             # (H,W)
        sharpness = (luma[1:, :] - luma[:-1, :]).abs().mean().item()
        quality = torch.tensor([luma.mean().item(), luma.std(unbiased=False).item(), sharpness])
        mask = mask_dataset[position].mask.cpu()
        return face, mask, quality

    from concurrent.futures import ThreadPoolExecutor
    features, verdict_probs, masks, quality = [], [], [], []
    block = max(chunk, 256)
    executor = ThreadPoolExecutor(max_workers=max(1, workers)) if workers > 1 else None
    try:
        for start in range(0, len(index), block):
            block_items = index[start:start + block]
            positioned_items = [(start + offset, item)
                                for offset, item in enumerate(block_items)]
            if executor is not None:
                rows = list(executor.map(load_one, positioned_items))
            else:
                rows = [load_one(item) for item in positioned_items]
            faces = torch.stack([r[0] for r in rows])       # one decode block
            masks.append(torch.stack([r[1] for r in rows]))
            quality.append(torch.stack([r[2] for r in rows]))
            for bstart in range(0, len(faces), chunk):
                batch = faces[bstart:bstart + chunk].to(device)
                with torch.amp.autocast("cuda", dtype=dt, enabled=torch.cuda.is_available()):
                    feat, prob = verdict.infer(batch)
                features.append(feat.float().to(dt).cpu())
                verdict_probs.append(prob.float().cpu())
            del faces, rows
    finally:
        if executor is not None:
            executor.shutdown()
    out = {
        "features": torch.cat(features),
        "verdict_probs": torch.cat(verdict_probs),
        "masks": torch.cat(masks),
        "quality": torch.cat(quality),
    }
    backbone_identity = _checkpoint_identity(verdict)
    fingerprint = _index_fingerprint(index, backbone_identity)
    payload = {
        "verdict": verdict.name,
        "backbone_identity": backbone_identity,
        "cfg": cfg.to_dict(),
        "fingerprint": fingerprint,
        "features": out["features"],
        "verdict_probs": out["verdict_probs"],
        "masks": out["masks"],
        "quality": out["quality"],
    }
    path = _cache_path(cache_dir, verdict.name, cfg, index,
                       backbone_checkpoint=backbone_identity,
                       fingerprint=fingerprint)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    log.info("built feature cache for %d faces -> %s (%.1f MB)",
             len(index), path.name, path.stat().st_size / 1e6)
    return out


def load_feature_cache(index: List[dict], cfg: Config, verdict_name: str,
                       cache_dir: str) -> Optional[dict]:
    """Load a compatible cache, or None when stale/missing."""
    backbone_identity = _checkpoint_identity(verdict_name)
    fingerprint = _index_fingerprint(index, backbone_identity)
    path = _cache_path(cache_dir, verdict_name, cfg, index,
                       backbone_checkpoint=backbone_identity,
                       fingerprint=fingerprint)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log.warning("unreadable feature cache %s (%s); will rebuild", path.name, exc)
        return None
    if (not isinstance(payload, dict) or payload.get("verdict") != verdict_name
            or payload.get("backbone_identity") != backbone_identity
            or payload.get("fingerprint") != fingerprint):
        log.warning("stale feature cache %s (index or backbone changed); will rebuild", path.name)
        return None
    # Older caches (features + probs only) are rebuilt with the extra tensors.
    if not (isinstance(payload.get("masks"), torch.Tensor)
            and isinstance(payload.get("quality"), torch.Tensor)):
        log.warning("legacy feature cache %s (missing masks/quality); will rebuild", path.name)
        return None
    return {"features": payload["features"], "verdict_probs": payload["verdict_probs"],
            "masks": payload["masks"], "quality": payload["quality"]}


class CachedMaskSample:
    """Face-level sample: cached features replace the face tensor entirely."""
    __slots__ = ("features", "verdict_prob", "mask", "label", "layout", "video", "frame")

    def __init__(self, features, verdict_prob, mask, label, layout, video, frame):
        self.features = features          # (C, h, w) cached backbone feature map
        self.verdict_prob = verdict_prob  # float frozen verdict probability
        self.mask = mask
        self.label = label
        self.layout = layout
        self.video = video
        self.frame = frame


class CachedMaskDataset(Dataset):
    """Face-level dataset over fully-cached tensors (no disk I/O per epoch)."""

    def __init__(self, index: List[dict], cfg: Config, cache: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.index = index
        self.features = cache["features"]
        self.verdict_probs = cache["verdict_probs"]
        self.masks = cache["masks"]
        self.size = cfg.data.face_size
        self.layout_dataset = MaskDataset(index, cfg, device="cpu", cache_size=0,
                                          read_face=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> CachedMaskSample:
        item = self.index[idx]
        layout = self.layout_dataset[idx].layout
        return CachedMaskSample(
            self.features[idx].to(self.device),
            float(self.verdict_probs[idx].item()),
            self.masks[idx].to(self.device),
            int(item["label"]), layout,
            item.get("video", ""), item.get("frame", ""),
        )


class CachedSequenceMaskSample:
    """Sequence sample carrying only cached tensors (no faces needed)."""
    __slots__ = ("features", "verdict_probs", "quality", "masks", "layouts",
                 "label", "video", "frames")

    def __init__(self, features, verdict_probs, quality, masks, layouts,
                 label, video, frames):
        self.features = features            # (steps, C, h, w)
        self.verdict_probs = verdict_probs  # (steps,)
        self.quality = quality              # (steps, 3)
        self.masks = masks
        self.layouts = layouts
        self.label = label
        self.video = video
        self.frames = frames


class CachedSequenceMaskDataset(Dataset):
    """Same-video sequences over fully-cached tensors (zero face decode)."""

    def __init__(self, index: List[dict], cfg: Config, cache: dict, device: str = "cpu"):
        import numpy as np
        self.cfg = cfg
        self.device = device
        self.index = index
        self.features = cache["features"]
        self.verdict_probs = cache["verdict_probs"]
        self.masks = cache["masks"]
        self.quality = cache["quality"]
        self.layout_dataset = MaskDataset(index, cfg, device="cpu", cache_size=0,
                                          read_face=False)
        length = int(cfg.data.sequence_length)
        if length < 2:
            raise ValueError("sequence_length must be at least 2")
        groups: dict = {}
        for position, item in enumerate(index):
            groups.setdefault(str(item.get("video") or position), []).append(position)
        self.sequences = []
        for video, positions in sorted(groups.items()):
            positions.sort(key=lambda pos: str(index[pos].get("frame", "")))
            labels = {int(index[pos]["label"]) for pos in positions}
            if len(labels) != 1 or len(positions) < length:
                continue
            chosen = np.linspace(0, len(positions) - 1, length, dtype=int)
            self.sequences.append((video, tuple(positions[int(i)] for i in chosen), labels.pop()))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> CachedSequenceMaskSample:
        video, positions, label = self.sequences[idx]
        return CachedSequenceMaskSample(
            torch.stack([self.features[pos].to(self.device) for pos in positions]),
            torch.stack([self.verdict_probs[pos].to(self.device) for pos in positions]),
            torch.stack([self.quality[pos].to(self.device) for pos in positions]),
            torch.stack([self.masks[pos].to(self.device) for pos in positions]),
            tuple(self.layout_dataset[pos].layout for pos in positions),
            label, video,
            tuple(str(self.index[pos].get("frame", "")) for pos in positions),
        )


def _default_layout():
    from .data.synthetic import FaceLayout
    return FaceLayout()


def cached_sequence_collate(samples: List[CachedSequenceMaskSample]) -> dict:
    return {
        "features": torch.stack([s.features for s in samples]),
        "verdict_probs": torch.stack([s.verdict_probs for s in samples]),
        "quality": torch.stack([s.quality for s in samples]),
        "masks": torch.stack([s.masks for s in samples]),
        "layouts": tuple(s.layouts for s in samples),
        "labels": torch.tensor([s.label for s in samples], dtype=torch.float32),
        "videos": tuple(s.video for s in samples),
        "frames": tuple(s.frames for s in samples),
    }


def cached_mask_collate(samples: List[CachedMaskSample]) -> dict:
    """DataLoader collate for face-level cached samples.

    Features live on CPU in the cache; the loader pins the batch so the
    training step transfers a single contiguous tensor with ``non_blocking``.
    """
    return {
        "features": torch.stack([s.features for s in samples]),
        "verdict_probs": torch.tensor([s.verdict_prob for s in samples],
                                      dtype=torch.float32),
        "masks": torch.stack([s.mask for s in samples]),
        "labels": torch.tensor([s.label for s in samples], dtype=torch.float32),
        "layouts": tuple(s.layout for s in samples),
        "videos": tuple(s.video for s in samples),
        "frames": tuple(s.frame for s in samples),
    }


__all__ = ["build_feature_cache", "load_feature_cache", "CachedMaskDataset",
           "CachedMaskSample", "CachedSequenceMaskDataset",
           "CachedSequenceMaskSample", "cached_sequence_collate",
           "cached_mask_collate"]
