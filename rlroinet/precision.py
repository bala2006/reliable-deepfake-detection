"""Shared precision / autocast helpers so every trainer uses the same dtype rules.

The project trains on an RTX 4050 (Ada).  ``bf16`` runs the same tensor-core
math as ``fp16`` but keeps fp32-comparable exponent range, needs no
``GradScaler``, and never underflows small losses, so it is both faster and
more stable than fp16 for this task.  Master weights stay fp32 either way.
"""

from __future__ import annotations

import torch

from .config import Config


def amp_dtype(cfg: Config) -> torch.dtype:
    """Resolve the configured mixed-precision dtype (``fp16`` or ``bf16``)."""
    kind = str(getattr(cfg.train, "amp_dtype", "bf16")).lower()
    if kind == "bf16":
        return torch.bfloat16
    if kind == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be 'fp16' or 'bf16'")


def make_scaler(cfg: Config):
    """Return a GradScaler for fp16, or None for bf16 (which needs no scaling)."""
    enabled = bool(cfg.train.amp) and str(getattr(cfg.train, "amp_dtype", "bf16")).lower() == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled) if enabled else None


def autocast(cfg: Config, device: str = "cuda"):
    """Autocast context matching the configured dtype (enabled only on CUDA)."""
    enabled = bool(cfg.train.amp) and str(device).startswith("cuda") and torch.cuda.is_available()
    return torch.amp.autocast("cuda", enabled=enabled, dtype=amp_dtype(cfg))


def enable_tf32(tf32: bool = True) -> None:
    """Enable TF32 matmuls + cudnn (safe on Ada; ~1.5-2x fp16/bf16 conv gains)."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


__all__ = ["amp_dtype", "make_scaler", "autocast", "enable_tf32"]
