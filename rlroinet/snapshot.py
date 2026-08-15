"""Wall-clock snapshot checkpoints so a long training run can be inspected while running.

The champion trainers already evaluate validation metrics every epoch and keep
``best.pt`` / ``final.pt``.  This helper adds a *time-based* budget on top so a
run saves a labeled checkpoint roughly every ``interval_minutes`` regardless of
epoch length, which lets you watch whether validation AUC is still improving
without waiting for the run to finish.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch

BuildCheckpoint = Callable[[int, dict], dict]


class SnapshotScheduler:
    """Save a checkpoint approximately every ``interval_minutes`` of wall-clock time.

    Call :meth:`maybe_save` after each epoch with a callable that builds the
    checkpoint dict (so the trainer keeps ownership of its own format).  Each
    snapshot is named ``snapshot_t<minutes>min_epoch<NN>.pt`` and its metrics
    carry ``elapsed_minutes`` so progress is directly readable from the file.
    """

    def __init__(self, checkpoint_dir: str | Path, interval_minutes: float = 30.0,
                 started: float | None = None):
        if interval_minutes < 0:
            raise ValueError("snapshot interval cannot be negative")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.disabled = interval_minutes == 0
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = float(interval_minutes) * 60.0
        self.started = time.perf_counter() if started is None else started
        self.next_snapshot = (None if self.disabled
                              else self.started + self.interval_seconds)
        self.last_path: Path | None = None

    def elapsed_minutes(self) -> float:
        return (time.perf_counter() - self.started) / 60.0

    def due(self) -> bool:
        return not self.disabled and time.perf_counter() >= self.next_snapshot

    def maybe_save(self, build: BuildCheckpoint, epoch: int, metrics: dict) -> Path | None:
        if self.disabled or not self.due():
            return None
        self.next_snapshot = time.perf_counter() + self.interval_seconds
        minutes = int(round(self.elapsed_minutes()))
        snapshot_metrics = dict(metrics)
        snapshot_metrics["elapsed_minutes"] = round(self.elapsed_minutes(), 1)
        path = self.checkpoint_dir / f"snapshot_t{minutes:03d}min_epoch{epoch:02d}.pt"
        torch.save(build(epoch, snapshot_metrics), path)
        self.last_path = path
        return path


class WarmupCosine:
    """Linear warm-up then cosine decay to ``min_lr`` over a fixed epoch budget.

    Every training script currently uses a constant ``lr_head``; a decaying
    schedule is the single highest-leverage training change (standard for
    deepfake detection, and how the honi05 backbone itself was trained).
    """

    def __init__(self, optimizer, base_lr: float, total_epochs: int,
                 warmup_epochs: int = 1, min_lr: float = 0.0, epoch: int = 0):
        if total_epochs < 1 or warmup_epochs < 0 or warmup_epochs >= total_epochs:
            raise ValueError("total_epochs must be >= 1 and warmup_epochs in [0, total_epochs)")
        if epoch < 0 or epoch > total_epochs:
            raise ValueError("start epoch must be within [0, total_epochs]")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.epoch = int(epoch)

    def step(self) -> float:
        """Advance one epoch and return the new learning rate."""
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            fraction = self.epoch / max(1, self.warmup_epochs)
            lr = self.base_lr * fraction
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


__all__ = ["SnapshotScheduler", "WarmupCosine"]
