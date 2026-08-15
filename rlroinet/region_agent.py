"""Trainable region head on top of a frozen verdict backbone.

The verdict model (backbone + classifier) is frozen and provides both the
real/fake verdict and the feature map. Only the :class:`RegionHead` decoder is
trained, supervised by DeepfakeBench forgery masks (Dice + BCE + per-ROI BCE).

This is the head-only training regime: the backbone is never updated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import Config
from .models.classifier import ClassifierHead, focal_loss
from .models.region_head import RegionHead
from .precision import autocast
from .verdict import VerdictModel


class RegionAgent(nn.Module):
    """Frozen verdict backbone + trainable region head (+ optional classifier).

    ``forward`` returns the frozen verdict probability alongside the trainable
    mask and per-ROI logits, so the same module serves both training and
    inference. When a trainable classifier is present, ``cls_logits`` is its
    binary output and ``classifier_trained`` marks whether those logits are
    meaningful (set True after training/loading a classifier state).
    """

    def __init__(self, cfg: Config, verdict: VerdictModel, with_classifier: bool = True):
        super().__init__()
        self.verdict = verdict
        self.region = RegionHead(
            in_channels=verdict.feature_channels,
            out_channels=cfg.model.region_out_channels,
            n_regions=cfg.model.n_regions,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.classifier_use_verdict = bool(cfg.model.classifier_use_verdict)
        self.classifier_pooling = str(cfg.model.classifier_pooling)
        if self.classifier_pooling not in ("avg", "avg_max"):
            raise ValueError("classifier_pooling must be 'avg' or 'avg_max'")
        classifier_dim = verdict.feature_channels * (2 if self.classifier_pooling == "avg_max" else 1)
        if self.classifier_use_verdict:
            classifier_dim += 1
        self.classifier = ClassifierHead(
            feat_dim=classifier_dim,
            hidden=cfg.model.classifier_hidden,
        ) if with_classifier else None
        self.classifier_trained = False
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> dict:
        feat, verdict_prob = self.verdict.infer(x)
        return self.forward_from_features(feat, verdict_prob)

    def forward_from_features(self, feat: torch.Tensor, verdict_prob: torch.Tensor) -> dict:
        """Forward the trainable heads from cached backbone features.

        The verdict backbone is frozen, so its ``(features, verdict_prob)`` can
        be computed once per face and reused across epochs. This method runs
        only the trainable region head + classifier on that cached input,
        skipping the backbone entirely.
        """
        mask, region = self.region(feat)
        out = {
            "verdict_prob": verdict_prob,
            "mask": mask,
            "region": region,
            "features": feat,
        }
        if self.classifier is not None:
            pooled = [self.gap(feat).flatten(1)]
            if self.classifier_pooling == "avg_max":
                pooled.append(self.gmp(feat).flatten(1))
            if self.classifier_use_verdict:
                # The frozen detector is an additional calibrated forensic
                # signal; logit space is more useful than a saturated prob.
                p = verdict_prob.float().clamp(1e-5, 1.0 - 1e-5)
                pooled.append(torch.logit(p).unsqueeze(1))
            out["cls_logits"] = self.classifier(torch.cat(pooled, dim=1))
        return out

    def train(self, mode: bool = True):
        """Keep the frozen verdict backbone in eval mode.

        The verdict model is a fixed feature extractor; running its BatchNorm
        in train mode would drift the running statistics during region-head
        training, making evaluation (which reloads pristine weights) inconsistent
        with training-time metrics.
        """
        super().train(mode)
        self.verdict.eval()
        return self

    def trainable_parameters(self):
        params = list(self.region.parameters())
        if self.classifier is not None:
            params += list(self.classifier.parameters())
        return params


def _mask_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    probs = pred.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(probs, target)
    pf = probs.reshape(probs.shape[0], -1)
    tf = target.reshape(target.shape[0], -1)
    inter = (pf * tf).sum(1)
    denom = pf.sum(1) + tf.sum(1)
    dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return cfg.loss.dice_weight * dice + cfg.loss.bce_weight * bce


def region_training_step(agent: RegionAgent, batch, cfg: Config, optim, scaler=None) -> dict:
    """One optimization step for the region head (and classifier when trained).

    Loss = mask Dice+BCE + region_weight * per-ROI BCE, plus the classification
    loss (Focal) when the agent carries a trainable classifier. The verdict
    backbone is frozen and contributes no loss. When ``scaler`` (a CUDA
    :class:`torch.amp.GradScaler`) is provided, gradients are scaled so fp16
    mixed-precision training does not underflow small losses.
    """
    device = cfg.train.device
    agent.train()

    faces = torch.stack([b.face for b in batch]).to(device, non_blocking=True)
    masks = torch.stack([b.mask for b in batch]).unsqueeze(1).to(device, non_blocking=True)
    labels = torch.tensor([b.label for b in batch], device=device).float()

    optim.zero_grad(set_to_none=True)
    with autocast(cfg):
        out = agent(faces)

    return _finish_region_step(agent, out, masks, labels, [b.layout for b in batch],
                               cfg, optim, scaler)


def region_training_step_from_cache(agent: RegionAgent, batch, cfg: Config, optim,
                                    scaler=None) -> dict:
    """Training step over cached backbone features (no backbone forward).

    Batch samples expose ``features`` and ``verdict_prob`` produced once by
    :func:`rlroinet.feature_cache.build_feature_cache` instead of a ``face``.
    """
    device = cfg.train.device
    agent.train()

    features = torch.stack([b.features for b in batch]).to(device, non_blocking=True)
    verdict_probs = torch.tensor([b.verdict_prob for b in batch], device=device).float()
    masks = torch.stack([b.mask for b in batch]).unsqueeze(1).to(device, non_blocking=True)
    labels = torch.tensor([b.label for b in batch], device=device).float()

    optim.zero_grad(set_to_none=True)
    with autocast(cfg):
        out = agent.forward_from_features(features, verdict_probs)

    return _finish_region_step(agent, out, masks, labels, [b.layout for b in batch],
                               cfg, optim, scaler)


def region_training_step_from_batch(agent: RegionAgent, batch: dict, cfg: Config, optim,
                                    scaler=None) -> dict:
    """One optimization step from a DataLoader ``collate_fn`` dict batch.

    Parallel DataLoader workers decode faces/masks (or serve cached features)
    while the GPU computes; ``pin_memory`` + ``non_blocking`` keep host-to-device
    copies off the critical path.  Mirrors ``region_training_step``/``from_cache``
    but takes a single stacked dict so one forward/backward covers the micro-batch.
    """
    device = cfg.train.device
    agent.train()
    cached = batch.get("features") is not None
    non_blocking = bool(batch.get("pinned", False))

    if cached:
        features = batch["features"].to(device, non_blocking=non_blocking)
        verdict_probs = batch["verdict_probs"].to(device, non_blocking=non_blocking).float()
        masks = batch["masks"].unsqueeze(1).to(device, non_blocking=non_blocking)
        faces = None
    else:
        features = None
        verdict_probs = None
        faces = batch["faces"].to(device, non_blocking=non_blocking)
        masks = batch["masks"].unsqueeze(1).to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking).float()

    optim.zero_grad(set_to_none=True)
    with autocast(cfg):
        if cached:
            out = agent.forward_from_features(features, verdict_probs)
        else:
            out = agent(faces)

    return _finish_region_step(agent, out, masks, labels, batch["layouts"], cfg, optim, scaler)


def _finish_region_step(agent, out, masks, labels, layouts, cfg, optim, scaler) -> dict:
    """Shared loss + optimizer tail for region/adapter head training steps."""
    device = cfg.train.device
    mask = out["mask"].float()
    region = out["region"].float()
    pred_mask = F.interpolate(mask, size=masks.shape[-2:], mode="bilinear",
                              align_corners=False)
    map_loss = _mask_loss(pred_mask, masks, cfg)

    region_size = region.shape[-1]
    gt_small = F.interpolate(masks, size=(region_size, region_size), mode="nearest")
    roi_targets = torch.stack([
        agent.region.region_masks(region_size, layout).to(device, non_blocking=True)
        for layout in layouts
    ])
    region_targets = gt_small * roi_targets
    region_loss = F.binary_cross_entropy_with_logits(region, region_targets)

    loss = map_loss + cfg.loss.region_weight * region_loss
    cls_loss = torch.tensor(0.0, device=device)
    if agent.classifier is not None:
        cls_logits = out["cls_logits"].float()
        cls_loss = focal_loss(cls_logits, labels,
                              gamma=cfg.loss.focal_gamma,
                              alpha=cfg.loss.focal_alpha)
        loss = loss + cfg.loss.class_weight * cls_loss

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.train.grad_clip)
        optim.step()

    return {"cls_loss": cls_loss.item(), "mask_loss": map_loss.item(),
            "region_loss": region_loss.item(), "loss": loss.item()}


__all__ = ["RegionAgent", "region_training_step", "region_training_step_from_cache",
           "region_training_step_from_batch"]
