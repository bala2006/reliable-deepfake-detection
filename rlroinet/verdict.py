"""Frozen deepfake verdict models.

Each model wraps an external pretrained detector and exposes two things for the
downstream pipeline:

  * ``features(face)`` -> ``(B, C, H, W)`` frozen backbone feature map, consumed
    by the trainable region head.
  * ``verdict(face)`` -> ``(B,)`` probability of manipulation, the frozen
    real/fake verdict.

The backbone and classifier are frozen at construction; only an external region
head trained on top of ``features()`` learns. Preprocessing (resize + ImageNet
normalization) is applied internally so callers pass standard ``[0,1]`` face
crops in the project's ``face_size`` resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("rlroinet.verdict")

WEIGHTS_DIR = Path("outputs/models")


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet normalize a ``[0,1]`` float tensor (no /255)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class VerdictModel(nn.Module):
    """Frozen verdict model: backbone features + verdict probability.

    Parameters
    ----------
    name : str
        ``"honi05"`` (EfficientNet-B4, MIT) or ``"deepfakebench_xception"``
        (Xception, CC BY-NC 4.0).
    device : str
        Target device.
    weights_dir : Path
        Directory containing the checkpoint files.
    """

    def __init__(self, name: str, device: str = "cpu", weights_dir: Path = WEIGHTS_DIR):
        super().__init__()
        self.name = name
        self.device = device
        self.weights_dir = Path(weights_dir)
        self._build()
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.to(device)

    # -- public interface ---------------------------------------------------

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, *, *)`` [0,1] face -> ``(B, C, h, w)`` frozen feature map."""
        x = self._preprocess(x)
        return self._features_forward(x)

    def verdict(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, *, *)`` [0,1] face -> ``(B,)`` frozen probability of fake."""
        x = self._preprocess(x)
        feat = self._features_forward(x)
        return self._verdict_from_features(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.verdict(x)

    def infer(self, x: torch.Tensor) -> tuple:
        """One backbone pass -> ``(features, verdict_prob)``.

        More efficient than calling ``features`` and ``verdict`` separately,
        which each run the backbone. Used by training and inference.
        """
        x = self._preprocess(x)
        feat = self._features_forward(x)
        return feat, self._verdict_from_features(feat)

    @property
    def feature_channels(self) -> int:
        return self._feature_channels

    @property
    def input_size(self) -> int:
        return self._input_size

    def spatial_size(self, face_size: int) -> int:
        return max(1, face_size // self._spatial_factor)

    # -- internals ----------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._input_size or x.shape[-2] != self._input_size:
            x = F.interpolate(x, size=(self._input_size, self._input_size),
                              mode="bilinear", align_corners=False)
        return _imagenet_normalize(x)

    def _build(self) -> None:
        if self.name == "honi05":
            self._build_honi05()
        elif self.name == "deepfakebench_xception":
            self._build_deepfakebench_xception()
        else:
            raise ValueError(f"unknown verdict model '{self.name}'; "
                             f"choose from {list(_REGISTRY)}")

    @torch.no_grad()
    def _verdict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _features_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# honi05 / deepfake-detection  (EfficientNet-B4, MIT)
# ---------------------------------------------------------------------------

class _Honi05(VerdictModel):
    def _build_honi05(self):
        import torchvision.models as tvm

        self._input_size = 224
        self._spatial_factor = 32
        self._feature_channels = 1792

        base = tvm.efficientnet_b4(weights=None)
        base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        ckpt = torch.load(self.weights_dir / "best_model.pt", map_location="cpu",
                          weights_only=True)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        base.load_state_dict(state)

        self.backbone = base.features
        self.classifier = base.classifier
        self.gap = nn.AdaptiveAvgPool2d(1)

    def _features_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def _verdict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.gap(feat).flatten(1)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DeepfakeBench Xception  (CC BY-NC 4.0)
# ---------------------------------------------------------------------------

class _SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class _XceptionBlock(nn.Module):
    def __init__(self, in_f, out_f, reps, strides=1, start_relu=True, grow_first=True):
        super().__init__()
        if out_f != in_f or strides != 1:
            self.skip = nn.Conv2d(in_f, out_f, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_f)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_f
        if grow_first:
            rep += [self.relu, _SeparableConv2d(in_f, out_f, 3, 1, 1), nn.BatchNorm2d(out_f)]
            filters = out_f
        for _ in range(reps - 1):
            rep += [self.relu, _SeparableConv2d(filters, filters, 3, 1, 1), nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [self.relu, _SeparableConv2d(in_f, out_f, 3, 1, 1), nn.BatchNorm2d(out_f)]
        if not start_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        skip = self.skipbn(self.skip(x)) if self.skip is not None else x
        return self.rep(x) + skip


class _Xception(nn.Module):
    def __init__(self, inc=3, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = _XceptionBlock(64, 128, 2, 2, start_relu=False, grow_first=True)
        self.block2 = _XceptionBlock(128, 256, 2, 2, start_relu=True, grow_first=True)
        self.block3 = _XceptionBlock(256, 728, 2, 2, start_relu=True, grow_first=True)
        self.block4 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block5 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block6 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block7 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block8 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block9 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block10 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block11 = _XceptionBlock(728, 728, 3, 1, start_relu=True, grow_first=True)
        self.block12 = _XceptionBlock(728, 1024, 2, 2, start_relu=True, grow_first=False)
        self.conv3 = _SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = _SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        # adjust_channel: present in the DeepfakeBench pretrained checkpoint.
        # Their forward applies it only in adjust_channel mode; this checkpoint's
        # last_linear is [2, 2048], so it is NOT on the active path (see classifier).
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=False))
        # The pretrained xception_best.pth saves last_linear as a plain Linear(2048, 2).
        self.last_linear = nn.Linear(2048, 2)

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block3(self.block2(self.block1(x)))
        x = self.block7(self.block6(self.block5(self.block4(x))))
        x = self.block11(self.block10(self.block9(self.block8(x))))
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        return self.bn4(self.conv4(x))

    def classifier(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.last_linear(x)


class _DeepfakeBenchXception(VerdictModel):
    def _build_deepfakebench_xception(self):
        self._input_size = 299
        self._spatial_factor = 32
        self._feature_channels = 2048

        net = _Xception(inc=3, dropout=0.0)
        ckpt = torch.load(self.weights_dir / "xception_best.pth", map_location="cpu",
                          weights_only=True)
        state = ckpt.get("backbone", ckpt) if isinstance(ckpt, dict) else ckpt
        # strip the 'backbone.' prefix DeepfakeBench saves with
        state = {k[len("backbone."):] if k.startswith("backbone.") else k: v
                 for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        self.backbone = net
        self.classifier = net.classifier

    def _features_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)

    @torch.no_grad()
    def _verdict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(feat)
        return torch.softmax(logits, dim=-1)[:, 1]


_REGISTRY: Dict[str, type] = {
    "honi05": _Honi05,
    "deepfakebench_xception": _DeepfakeBenchXception,
}


def load_verdict(name: str, device: str = "cpu", weights_dir: Path = WEIGHTS_DIR) -> VerdictModel:
    """Load a frozen verdict model by name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown verdict model '{name}'; choose from {list(_REGISTRY)}")
    log.info("loading frozen verdict model '%s' on %s", name, device)
    return _REGISTRY[name](name, device=device, weights_dir=weights_dir)


__all__ = ["VerdictModel", "load_verdict"]
