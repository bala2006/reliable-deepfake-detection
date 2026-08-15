"""Frozen-verdict, quality-aware temporal classification head."""

from __future__ import annotations

import torch
from torch import nn

from .config import Config
from .verdict import VerdictModel


class TemporalQualityHead(nn.Module):
    """Quality-aware temporal fusion with a conservative residual gate.

    The gate lets the head retain the frame representation when temporal
    context is unreliable, instead of always adding the depthwise-mixed
    signal.  It is deliberately small and uses the frozen verdict logit and
    image-quality features as inputs, so this experiment changes only the
    trainable temporal fusion and keeps the backbone frozen.
    """

    def __init__(self, channels: int, hidden: int, quality_hidden: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(channels + 1, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.depthwise = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.quality_context = nn.Sequential(
            nn.Linear(3, quality_hidden), nn.GELU(), nn.Linear(quality_hidden, hidden),
        )
        gate_hidden = max(8, hidden // 4)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2 + 1, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, hidden), nn.Sigmoid(),
        )
        # Start close to the prior residual behavior while allowing the gate
        # to learn suppression of noisy/contradictory frame context.
        nn.init.constant_(self.fusion_gate[2].bias, -1.5)
        self.quality = nn.Sequential(nn.Linear(3, quality_hidden), nn.GELU(), nn.Linear(quality_hidden, 1))
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, features: torch.Tensor, quality: torch.Tensor, verdict_logits: torch.Tensor) -> dict:
        base = self.project(torch.cat((features, verdict_logits.unsqueeze(-1)), dim=-1))
        mixed = self.pointwise(self.depthwise(base.transpose(1, 2))).transpose(1, 2)
        candidate = self.temporal_norm(mixed + self.quality_context(quality))
        gate_input = torch.cat((base, candidate, verdict_logits.unsqueeze(-1)), dim=-1)
        x = base + self.fusion_gate(gate_input) * candidate
        weights = torch.softmax((self.attention(x) + self.quality(quality)).squeeze(-1), dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return {"logits": self.classifier(pooled), "attention": weights}


class TemporalQualityAgent(nn.Module):
    """Train only temporal quality fusion; keep the verdict backbone frozen."""

    def __init__(self, cfg: Config, verdict: VerdictModel):
        super().__init__()
        self.cfg, self.verdict = cfg, verdict
        self.head = TemporalQualityHead(verdict.feature_channels, cfg.model.temporal_hidden,
                                        cfg.model.temporal_quality_hidden)

    @staticmethod
    def _quality(x: torch.Tensor) -> torch.Tensor:
        luma = x.mean(dim=2)
        sharpness = (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs().mean(dim=(2, 3))
        return torch.stack((luma.mean(dim=(2, 3)), luma.std(dim=(2, 3), unbiased=False), sharpness), dim=-1)

    def forward(self, faces: torch.Tensor) -> dict:
        batch, steps = faces.shape[:2]
        flat = faces.flatten(0, 1)
        with torch.no_grad():
            features, verdict_prob = self.verdict.infer(flat)
            features = features.mean(dim=(2, 3)).reshape(batch, steps, -1)
            verdict_logits = torch.logit(verdict_prob.clamp(1e-5, 1 - 1e-5)).reshape(batch, steps)
        return self.head(features, self._quality(faces), verdict_logits)

    def train(self, mode: bool = True):
        super().train(mode)
        self.verdict.eval()
        return self

    def trainable_parameters(self):
        return self.head.parameters()
