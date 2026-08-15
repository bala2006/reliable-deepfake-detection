"""Agent: frozen backbone + Focal classification head + region (mask) head."""

from __future__ import annotations

import torch
from torch import nn

from ..config import Config
from .backbone import FeatureExtractor, PretrainedFeatureExtractor
from .classifier import ClassifierHead
from .region_head import RegionHead


class Agent(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.backbone = self._build_backbone(m)
        self.classifier = ClassifierHead(feat_dim=m.feat_dim, hidden=m.classifier_hidden)
        self.region = RegionHead(
            in_channels=self._feature_channels(m),
            out_channels=m.region_out_channels,
            n_regions=m.n_regions,
        )
        self.cfg = cfg

    def _feature_channels(self, m) -> int:
        """Channel depth of the backbone feature map consumed by the region head."""
        if m.backbone == "custom":
            return 256
        from .backbone import PretrainedFeatureExtractor
        return PretrainedFeatureExtractor._OUT_DIM.get(m.backbone, 1792)

    def _build_backbone(self, m):
        if m.backbone == "custom":
            return FeatureExtractor(in_ch=m.in_ch, widths=(32, 64, 128, 256),
                                    feat_dim=m.feat_dim)
        bb = PretrainedFeatureExtractor(
            name=m.backbone, pretrained=m.pretrained,
            feat_dim=m.feat_dim, in_ch=m.in_ch)
        bb.set_trainable(m.freeze_blocks)
        return bb

    def forward(self, x: torch.Tensor, layout=None):
        """x: (B, 3, H, W) aligned faces.

        Returns dict{cls_logits:(B,1), mask:(B,1,H',W'), region:(B,4,H',W')}.
        """
        feat_map = self.backbone.forward_features(x)
        pooled = self.backbone.gap(feat_map).flatten(1)
        pooled = self.backbone.proj(pooled)
        cls_logits = self.classifier(pooled)
        mask, region = self.region(feat_map)
        return {"cls_logits": cls_logits, "mask": mask, "region": region,
                "pooled": pooled}

    def param_groups(self):
        """Separate LR for the frozen-constrained backbone vs the trainable heads."""
        heads = list(self.classifier.parameters()) + list(self.region.parameters())
        backbone = self.backbone.trainable_params()
        return [
            {"params": backbone, "lr": self.cfg.train.lr_backbone},
            {"params": heads, "lr": self.cfg.train.lr_head},
        ]


__all__ = ["Agent"]