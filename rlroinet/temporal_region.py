"""Joint temporal video classification and per-frame ROI localization."""

from __future__ import annotations

import torch
from torch import nn

from .config import Config
from .region_agent import RegionAgent
from .temporal_quality import TemporalQualityHead
from .verdict import VerdictModel


def quality_features(faces: torch.Tensor) -> torch.Tensor:
    """Return luminance, contrast, and vertical-detail quality per sequence frame."""
    luma = faces.mean(dim=2)
    sharpness = (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs().mean(dim=(2, 3))
    return torch.stack((luma.mean(dim=(2, 3)), luma.std(dim=(2, 3), unbiased=False), sharpness), dim=-1)


class TemporalRegionAgent(nn.Module):
    """Train temporal decisions while preserving mask and four-ROI evidence."""

    def __init__(self, cfg: Config, verdict: VerdictModel):
        super().__init__()
        self.cfg = cfg
        self.frame = RegionAgent(cfg, verdict, with_classifier=True)
        self.temporal = TemporalQualityHead(
            verdict.feature_channels, cfg.model.temporal_hidden, cfg.model.temporal_quality_hidden,
        )
        self.frame.classifier_trained = True

    def forward(self, faces: torch.Tensor) -> dict:
        batch, steps = faces.shape[:2]
        frame_out = self.frame(faces.flatten(0, 1))
        return self._compose(frame_out, quality_features(faces), batch, steps)

    def forward_from_features(self, faces: torch.Tensor, feat: torch.Tensor,
                              verdict_probs: torch.Tensor) -> dict:
        """Compose the temporal head from cached backbone features.

        ``feat`` is ``(B*steps, C, h, w)`` and ``verdict_probs`` is ``(B*steps,)``
        as produced by :func:`rlroinet.feature_cache.build_feature_cache`, aligned
        to the flattened sequence frames. Faces are still needed for the
        lightweight quality features but the frozen backbone is never run.
        """
        batch, steps = faces.shape[:2]
        quality = quality_features(faces).reshape(-1, 3)
        return self.forward_from_cached(feat, verdict_probs, quality, batch, steps)

    def forward_from_cached(self, feat: torch.Tensor, verdict_probs: torch.Tensor,
                            quality: torch.Tensor, batch: int, steps: int) -> dict:
        """Compose the temporal head from fully-cached inputs (no faces needed).

        ``quality`` is ``(B*steps, 3)`` image-quality features precomputed by the
        cache; this path never touches face tensors, so the training loop is pure
        tensor shuffling into the trainable heads.
        """
        frame_out = self.frame.forward_from_features(feat, verdict_probs)
        return self._compose(frame_out, quality, batch, steps)

    def _compose(self, frame_out: dict, quality: torch.Tensor, batch: int, steps: int) -> dict:
        """Assemble the temporal head output from a frame agent's forward pass.

        ``frame_out`` is the dict from :meth:`RegionAgent.forward` (mask, region,
        features, cls_logits) over flattened ``(B*steps, ...)`` faces, and
        ``quality`` is ``(B*steps, 3)`` per-frame quality features. Shared by
        the cached and uncached forward paths so both produce identical outputs.
        """
        features = self.frame.gap(frame_out["features"]).flatten(1).reshape(batch, steps, -1)
        frame_logits = frame_out["cls_logits"].reshape(batch, steps)
        temporal = self.temporal(features, quality.reshape(batch, steps, -1), frame_logits)
        return {
            "video_logits": temporal["logits"],
            "attention": temporal["attention"],
            "frame_logits": frame_logits,
            "mask": frame_out["mask"].reshape(batch, steps, *frame_out["mask"].shape[1:]),
            "region": frame_out["region"].reshape(batch, steps, *frame_out["region"].shape[1:]),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.frame.train(mode)
        return self

    def trainable_parameters(self):
        return [*self.frame.trainable_parameters(), *self.temporal.parameters()]
