"""Video-disjoint ordered face sequences for temporal experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .maskdata import MaskDataset


@dataclass(frozen=True)
class SequenceMaskSample:
    faces: torch.Tensor
    masks: torch.Tensor
    layouts: tuple
    label: int
    video: str
    frames: tuple[str, ...]


class SequenceMaskDataset(Dataset):
    """One deterministic, evenly-spaced same-video sequence per split video."""

    def __init__(self, index: List[dict], cfg, device: str = "cpu"):
        self.cfg = cfg
        # Temporal training uses worker processes, so do not retain a second
        # unbounded decoded-image cache in every worker.
        self.base = MaskDataset(index, cfg, device=device, cache_size=0)
        length = int(cfg.data.sequence_length)
        if length < 2:
            raise ValueError("sequence_length must be at least 2")
        groups: Dict[str, List[int]] = {}
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

    def __getitem__(self, index: int) -> SequenceMaskSample:
        video, positions, label = self.sequences[index]
        samples = [self.base[position] for position in positions]
        return SequenceMaskSample(
            torch.stack([sample.face for sample in samples]),
            torch.stack([sample.mask for sample in samples]),
            tuple(sample.layout for sample in samples), label, video,
            tuple(str(sample.frame) for sample in samples),
        )


def sequence_collate(samples: List[SequenceMaskSample]) -> dict:
    """Collate tensors in the worker, retaining layout metadata as tuples."""
    return {
        "faces": torch.stack([sample.faces for sample in samples]),
        "masks": torch.stack([sample.masks for sample in samples]),
        "layouts": tuple(sample.layouts for sample in samples),
        "labels": torch.tensor([sample.label for sample in samples], dtype=torch.float32),
        "videos": tuple(sample.video for sample in samples),
        "frames": tuple(sample.frames for sample in samples),
    }


class TrainSequenceAugment:
    """Non-mutating, sequence-consistent appearance augmentation for training only."""

    def __init__(self, base: SequenceMaskDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> SequenceMaskSample:
        sample = self.base[index]
        faces = sample.faces.clone()
        contrast = 0.9 + 0.2 * torch.rand((), dtype=faces.dtype)
        offset = -0.05 + 0.10 * torch.rand((), dtype=faces.dtype)
        center = faces.mean(dim=(1, 2, 3), keepdim=True)
        faces = (faces - center) * contrast + center + offset
        if torch.rand(()) < 0.35:
            faces = torch.nn.functional.avg_pool2d(faces, 3, stride=1, padding=1)
        if torch.rand(()) < 0.50:
            faces = faces + torch.randn_like(faces[:1]).expand_as(faces) * 0.015
        return SequenceMaskSample(faces.clamp_(0, 1), sample.masks, sample.layouts,
                                  sample.label, sample.video, sample.frames)
