"""Parameter-efficient spatial forensic adapter on a frozen verdict backbone.

This experiment keeps the pretrained verdict model frozen but learns a small
bottleneck residual adapter over its spatial feature map.  The adapted map is
shared by the existing mask/ROI decoder and an avg+max classification head,
with the frozen verdict logit retained as an explicit auxiliary signal.

The module intentionally matches the RegionAgent output contract so the
existing evaluator, generalization report, and inference helpers can score it
without changing production behavior.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import Config
from .models.classifier import ClassifierHead
from .models.region_head import RegionHead
from .verdict import VerdictModel


class SpatialForensicAdapter(nn.Module):
    """Low-rank residual adapter for a frozen CNN feature map.

    The zero-near initialization makes the experiment start close to the
    frozen representation while allowing the trainable heads to adapt local
    forensic evidence.  It is deliberately spatial rather than temporal so
    any gain can be attributed to representation adaptation, not frame order.
    """

    def __init__(self, channels: int, bottleneck: int = 64):
        super().__init__()
        if channels < 1 or bottleneck < 1:
            raise ValueError("channels and bottleneck must be positive")
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.norm = nn.GroupNorm(max(1, min(8, bottleneck)), bottleneck)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.act(self.norm(self.down(features))))
        return features + self.scale * residual


class AdapterRegionAgent(nn.Module):
    """Frozen verdict model plus trainable adapter, classifier, and ROI head."""

    def __init__(self, cfg: Config, verdict: VerdictModel, with_classifier: bool = True):
        super().__init__()
        self.verdict = verdict
        self.adapter = SpatialForensicAdapter(
            verdict.feature_channels,
            bottleneck=int(getattr(cfg.model, "adapter_bottleneck", 64)),
        )
        self.region = RegionHead(
            in_channels=verdict.feature_channels,
            out_channels=cfg.model.region_out_channels,
            n_regions=cfg.model.n_regions,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier = ClassifierHead(
            feat_dim=verdict.feature_channels * 2 + 1,
            hidden=cfg.model.classifier_hidden,
        ) if with_classifier else None
        self.classifier_trained = bool(with_classifier)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> dict:
        features, verdict_prob = self.verdict.infer(x)
        return self.forward_from_features(features, verdict_prob)

    def forward_from_features(self, features: torch.Tensor, verdict_prob: torch.Tensor) -> dict:
        """Run the trainable adapter + heads on cached backbone features."""
        adapted = self.adapter(features)
        mask, region = self.region(adapted)
        out = {
            "verdict_prob": verdict_prob,
            "mask": mask,
            "region": region,
            "features": adapted,
        }
        if self.classifier is not None:
            avg = self.gap(adapted).flatten(1)
            maximum = self.gmp(adapted).flatten(1)
            p = verdict_prob.float().clamp(1e-5, 1.0 - 1e-5)
            pooled = torch.cat((avg, maximum, torch.logit(p).unsqueeze(1)), dim=1)
            out["cls_logits"] = self.classifier(pooled)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        self.verdict.eval()
        return self

    def trainable_parameters(self):
        params = list(self.adapter.parameters()) + list(self.region.parameters())
        if self.classifier is not None:
            params += list(self.classifier.parameters())
        return params


__all__ = ["SpatialForensicAdapter", "AdapterRegionAgent"]
