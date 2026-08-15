"""Focal-loss binary classification head.

The head consumes the projected backbone feature and produces a binary
fake/real logit. Focal loss (gamma, alpha) down-weights well-classified frames
so the minority (fake) class contributes more.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ClassifierHead(nn.Module):
    """Binary classifier over the projected backbone feature."""

    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.input_dim = int(feat_dim)
        # Keep the two-layer head shape stable for existing v2 checkpoints;
        # enhanced runs change the input features, not the learned head type.
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        """pooled_feature: (B, D) -> logits (B, 1)."""
        return self.fc(pooled_feature)

    def logit(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        return self.forward(pooled_feature).squeeze(-1)

    def probability(self, pooled_feature: torch.Tensor) -> torch.Tensor:
        """Probability of the 'fake' class."""
        return torch.sigmoid(self.logit(pooled_feature))


def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Focal loss for binary classification (targets in {0,1})."""
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    targets = targets.float().reshape_as(logits)
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, torch.full_like(probs, alpha),
                          torch.full_like(probs, 1.0 - alpha))
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (alpha_t * (1.0 - pt) ** gamma * ce).mean()


__all__ = ["ClassifierHead", "focal_loss"]