"""Shared CNN backbone for supervised ROI-Net v2.

EfficientNet/ResNet feature maps feed the classifier and dense forgery-mask head.
The lightweight custom backbone is reserved for hermetic tests.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Lightweight residual conv block reserved for hermetic tests."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        return self.relu(out + self.shortcut(x))


class FeatureExtractor(nn.Module):
    """Tiny custom CNN: (B, 3, P, P) -> (B, feat_dim). CPU-friendly fallback."""

    def __init__(self, in_ch: int = 3, widths=(32, 64, 128, 256), feat_dim: int = 256):
        super().__init__()
        blocks = []
        cin = in_ch
        for w in widths:
            blocks.append(nn.Conv2d(cin, w, 5, stride=2, padding=2, bias=False))
            blocks.append(nn.BatchNorm2d(w))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(ConvBlock(w, w))
            cin = w
        self.stem = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(widths[-1], feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.gap(h).flatten(1)
        return self.proj(h)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map (B, C, H', W') used by the region head."""
        return self.stem(x)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


class PretrainedFeatureExtractor(nn.Module):
    """torchvision backbone with ImageNet weights + feature-map access.

    ``set_trainable(freeze_blocks)`` freezes the first ``freeze_blocks`` feature
    blocks (blocks 0..freeze_blocks-1) and leaves the remaining blocks + the
    projection trainable.  This is the transfer-learning recipe validated on
    Celeb-DF v2 (freeze blocks 0-4, train 5-8 + heads).
    """

    _FAMILIES = {
        "efficientnet_b0": "efficientnet",
        "efficientnet_b1": "efficientnet",
        "efficientnet_b2": "efficientnet",
        "efficientnet_b3": "efficientnet",
        "efficientnet_b4": "efficientnet",
        "resnet18": "resnet",
    }
    _OUT_DIM = {
        "efficientnet_b0": 1280, "efficientnet_b1": 1280, "efficientnet_b2": 1408,
        "efficientnet_b3": 1536, "efficientnet_b4": 1792, "resnet18": 512,
    }

    def __init__(self, name: str = "efficientnet_b4", pretrained: bool = True,
                 feat_dim: int = 256, in_ch: int = 3):
        super().__init__()
        import torchvision.models as tvm

        if name not in self._FAMILIES:
            raise ValueError(f"unsupported pretrained backbone '{name}'; "
                             f"choose from {sorted(self._FAMILIES)}")
        builder = getattr(tvm, name)
        weights = "IMAGENET1K_V1" if pretrained else None
        self.net = builder(weights=weights)
        self._family = self._FAMILIES[name]
        self._name = name

        if in_ch != 3:
            self._replace_first_conv(in_ch)

        if self._family == "efficientnet":
            self.net.classifier = nn.Identity()
            self._feature_blocks = list(self.net.features)
        else:  # resnet
            self.net.fc = nn.Identity()
            self._feature_blocks = [m for nm, m in self.net.named_children()
                                    if nm not in ("avgpool", "fc")]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self._OUT_DIM[name], feat_dim)

    def _replace_first_conv(self, in_ch: int) -> None:
        if self._family == "efficientnet":
            old = self.net.features[0][0]
            self.net.features[0][0] = nn.Conv2d(
                in_ch, old.out_channels, old.kernel_size, old.stride,
                old.padding, bias=old.bias is not None)
        else:
            self.net.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)

    def set_trainable(self, freeze_blocks: int) -> None:
        """Freeze the first ``freeze_blocks`` feature blocks; train the rest + proj."""
        freeze_blocks = max(0, int(freeze_blocks))
        freeze_blocks = min(freeze_blocks, len(self._feature_blocks))
        for i, blk in enumerate(self._feature_blocks):
            trainable = i >= freeze_blocks
            for p in blk.parameters():
                p.requires_grad_(trainable)
        for p in self.proj.parameters():
            p.requires_grad_(True)

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature map (B, C, H', W') used by the region head.

        EfficientNet: forward through ``features`` (spatial pooling removed).
        ResNet: forward through the conv body before avgpool/fc.
        """
        if self._family == "efficientnet":
            return self.net.features(x)
        h = x
        for blk in self._feature_blocks:
            h = blk(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled feature (B, feat_dim) used by the classification head."""
        h = self.forward_features(x)
        h = self.gap(h).flatten(1)
        return self.proj(h)

    def spatial_size(self, input_size: int) -> int:
        """Spatial size of the feature map for a square input of ``input_size`` px."""
        eff = {"efficientnet_b0": 32, "efficientnet_b1": 32, "efficientnet_b2": 32,
               "efficientnet_b3": 32, "efficientnet_b4": 32, "resnet18": 32}
        return max(1, input_size // eff.get(self._name, 32))