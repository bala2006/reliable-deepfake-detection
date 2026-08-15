"""Joint supervised training for ROI-Net v2.

Trains the partially frozen transfer-learning model on aligned face crops with focal classification,
dense Dice+BCE mask supervision, and explicit per-ROI supervision. The backbone
early blocks are frozen according to ``ModelConfig.freeze_blocks``.

Run inside Docker only:
    docker compose run --rm train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import Config, default_config
from .data import load_index, make_mask_dataset
from .data.maskdata import make_data_loader, mask_sample_collate
from .evaluate import evaluate, save_report
from .feature_cache import (CachedMaskDataset, build_feature_cache, cached_mask_collate,
                            load_feature_cache)
from .models import Agent, focal_loss
from .precision import enable_tf32, make_scaler
from .region_agent import RegionAgent, region_training_step_from_batch
from .verdict import load_verdict

log = logging.getLogger("rlroinet.train")


def configure_reproducibility(cfg: Config) -> None:
    """Seed all supported RNGs and apply the configured deterministic policy."""
    seed = int(cfg.train.seed)
    deterministic = bool(cfg.train.deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    enable_tf32(bool(cfg.train.tf32) and not deterministic)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _dataset_manifest_identity(cfg: Config) -> dict:
    roots = {
        "ffpp": Path(cfg.data.ffpp_dir),
        "dfdcp": Path(cfg.data.dfdcp_dir),
        "deepfakebench": Path(cfg.data.mask_dir),
        "celebdf_raw": Path(cfg.data.celebdf_dir),
    }
    root = roots.get(cfg.data.source)
    names = {
        "ffpp": (".rlroinet_index.json",),
        "dfdcp": ("dataset.json",),
        "deepfakebench": (".rlroinet_index.json", "dataset.json"),
        "celebdf_raw": (".rlroinet_index.json", "dataset.json"),
    }
    candidates = [root / name for name in names.get(cfg.data.source, ())] if root else []
    for path in candidates:
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        stat = path.stat()
        return {
            "source": cfg.data.source,
            "path": str(path.resolve()),
            "exists": True,
            "size_bytes": stat.st_size,
            "sha256": digest.hexdigest(),
        }
    return {
        "source": cfg.data.source,
        "path": str(candidates[0].resolve()) if candidates else None,
        "exists": False,
        "size_bytes": None,
        "sha256": None,
    }


def write_run_config(output_dir: str | Path, cfg: Config, **metadata) -> Path:
    """Write resolved configuration and environment identity before data loading."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": int(cfg.train.seed),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "dataset_manifest": _dataset_manifest_identity(cfg),
        "cfg": cfg.to_dict(),
        **metadata,
    }
    out = path / "run_config.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for a batch of probability maps."""
    pred_f = pred.reshape(pred.shape[0], -1)
    tgt_f = target.reshape(target.shape[0], -1)
    inter = (pred_f * tgt_f).sum(1)
    denom = pred_f.sum(1) + tgt_f.sum(1)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def mask_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Weighted Dice+BCE mask objective."""
    probs = pred.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(probs, target)
    return cfg.loss.dice_weight * dice_loss(probs, target) + cfg.loss.bce_weight * bce


def train_one_epoch(agent: Agent, loader, cfg: Config, optim, scaler) -> dict:
    device = cfg.train.device
    agent.train()
    total = {"cls": 0.0, "mask": 0.0, "n": 0}
    for batch in loader:
        faces = batch["faces"].to(device, non_blocking=True)
        masks = batch["masks"].unsqueeze(1).to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.train.amp,
                                dtype=torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16):
            out = agent(faces)

        cls_logits = out["cls_logits"].float()
        mask = out["mask"].float()
        region = out["region"].float()
        cls_loss = focal_loss(cls_logits, labels,
                              gamma=cfg.loss.focal_gamma,
                              alpha=cfg.loss.focal_alpha)
        pred_mask = F.interpolate(mask, size=masks.shape[-2:], mode="bilinear",
                                  align_corners=False)
        map_loss_value = mask_loss(pred_mask, masks, cfg)
        region_size = region.shape[-1]
        gt_small = F.interpolate(masks, size=(region_size, region_size), mode="nearest")
        roi_targets = torch.stack([
            agent.region.region_masks(region_size, layout).to(device, non_blocking=True)
            for layout in batch["layouts"]
        ])
        region_targets = gt_small * roi_targets
        region_loss = F.binary_cross_entropy_with_logits(region, region_targets)
        total_mask_loss = map_loss_value + cfg.loss.region_weight * region_loss
        loss = cfg.loss.class_weight * cls_loss + total_mask_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.grad_clip)
            optim.step()

        n = int(labels.shape[0])
        total["cls"] += cls_loss.item() * n
        total["mask"] += total_mask_loss.item() * n
        total["n"] += n

    n = max(1, total["n"])
    return {k: v / n for k, v in total.items() if k != "n"}


def train_one_epoch_region(agent: RegionAgent, loader, cfg: Config, optim, scaler,
                           epoch: int = 0, total_epochs: int = 0) -> dict:
    device = cfg.train.device
    total = {"cls": 0.0, "mask": 0.0, "region": 0.0, "loss": 0.0, "n": 0}
    n_batches = max(1, len(loader))
    last_pct = -1
    t_start = time.time()
    for bi, batch in enumerate(loader):
        step = region_training_step_from_batch(agent, batch, cfg, optim, scaler)
        n = int(batch["labels"].shape[0])
        total["cls"] += step["cls_loss"] * n
        total["mask"] += step["mask_loss"] * n
        total["region"] += step["region_loss"] * n
        total["loss"] += step["loss"] * n
        total["n"] += n

        pct = int(100 * (bi + 1) / n_batches)
        if pct >= last_pct + 5 or bi == n_batches - 1:
            last_pct = pct
            done = (bi + 1) / n_batches
            elapsed = time.time() - t_start
            eta = elapsed / done - elapsed if done > 0 else 0.0
            log.info("epoch %d/%d [%d/%d] %d%% cls=%.4f mask=%.4f region=%.4f loss=%.4f | %.1fs elapsed ETA %.1fs",
                     epoch, total_epochs, bi + 1, n_batches, pct,
                     total["cls"] / max(1, total["n"]),
                     total["mask"] / max(1, total["n"]),
                     total["region"] / max(1, total["n"]),
                     total["loss"] / max(1, total["n"]), elapsed, eta)
    n = max(1, total["n"])
    return {"cls_loss": total["cls"] / n, "mask_loss": total["mask"] / n,
            "region_loss": total["region"] / n, "loss": total["loss"] / n}


def _region_ckpt(cfg: Config, verdict_name: str, agent: RegionAgent, optim,
                 epoch: int, metrics: dict) -> dict:
    """Build a region-head checkpoint dict (includes optimizer for resume)."""
    ckpt = {"checkpoint_format": "region-head-v2", "verdict": verdict_name,
            "region_state": agent.region.state_dict(),
            "optimizer_state": optim.state_dict(),
            "cfg": cfg.to_dict(), "epoch": epoch, "metrics": metrics}
    if agent.classifier is not None:
        ckpt["classifier_state"] = agent.classifier.state_dict()
    return ckpt


_REGION_FORMATS = ("region-head-v1", "region-head-v2")


def _latest_region_checkpoint(checkpoint_dir: str, verdict_name: str):
    """Return ``(path, epoch)`` of the newest region-head checkpoint, or None."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    best = None
    for p in ckpt_dir.glob("*.pt"):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if not isinstance(ckpt, dict) or ckpt.get("checkpoint_format") not in _REGION_FORMATS:
            continue
        if ckpt.get("verdict") != verdict_name:
            continue
        ep = int(ckpt.get("epoch", 0))
        if best is None or ep > best[1]:
            best = (p, ep)
    return best


def _load_region_history(metrics_path: str, start_epoch: int):
    """Re-read metrics.json so a resumed run keeps prior epochs in the report."""
    if not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path) as fh:
            hist = json.load(fh)
    except Exception:
        return []
    if not isinstance(hist, list):
        return []
    return [m for m in hist if m.get("epoch", 0) < start_epoch]


def train_region_head(cfg: Config, verdict_name: str, epochs: int = None,
                      resume: bool = True,
                      snapshot_epochs: tuple = (5, 10, 15, 20),
                      with_classifier: bool = True,
                      select_video_validation: bool = False,
                      feature_cache_dir: str = "") -> dict:
    """Train only the region head on top of a frozen verdict backbone.

    The verdict model (backbone + classifier) stays frozen; its feature map
    feeds a trainable RegionHead. Supervised by forgery masks. Used for both
    DeepfakeBench GT-mask training and raw-data pseudo-mask smoke runs.

    With ``resume`` (default True) training continues from the most recent
    region-head checkpoint for this verdict instead of starting fresh.
    Snapshot checkpoints (``epoch_NN.pt``) are saved at ``snapshot_epochs``
    so interrupted runs can be resumed or each milestone evaluated.

    With ``with_classifier`` (default True) a small trainable classifier head
    is trained alongside the region head on the same frozen features, giving
    a joint WHAT (fake/real) + WHERE (forgery region) detector.
    """
    cfg.resolve_paths()
    configure_reproducibility(cfg)
    if cfg.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("region-head training requires CUDA on the RTX 4050 Docker runtime")
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    write_run_config(ckpt_dir, cfg, trainer="train_region_head", verdict=verdict_name,
                     epochs=int(epochs or cfg.train.epochs), resume=bool(resume),
                     with_classifier=bool(with_classifier),
                     select_video_validation=bool(select_video_validation))

    device = torch.device("cuda")
    verdict = load_verdict(verdict_name, device=device)
    agent = RegionAgent(cfg, verdict, with_classifier=with_classifier).to(device)
    if with_classifier:
        agent.classifier_trained = True

    # A continuation run starts from a verified region-head checkpoint but
    # deliberately creates a fresh optimizer in a separate experiment folder.
    # This prevents optimizer history and result artifacts from one dataset
    # contract being silently reused for another.
    init_checkpoint = os.environ.get("RLROINET_INIT_CHECKPOINT")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"initial checkpoint not found: {init_path}")
        initial = torch.load(init_path, map_location="cpu", weights_only=True)
        if (not isinstance(initial, dict)
                or initial.get("checkpoint_format") not in _REGION_FORMATS
                or initial.get("verdict") != verdict_name):
            raise ValueError("initial checkpoint is not compatible with this verdict")
        agent.region.load_state_dict(initial["region_state"])
        if agent.classifier is not None:
            classifier_state = initial.get("classifier_state")
            if not classifier_state:
                raise ValueError("initial checkpoint has no compatible classifier state")
            agent.classifier.load_state_dict(classifier_state)
        log.info("initialized region heads from %s (epoch %s)",
                 init_path, initial.get("epoch", "unknown"))

    train_ds = make_mask_dataset(cfg, "train", device="cpu")
    val_ds = make_mask_dataset(cfg, "val", device="cpu")
    test_ds = make_mask_dataset(cfg, "test", device="cpu")
    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise RuntimeError("training, validation, and test datasets must all contain samples")
    use_cache = bool(feature_cache_dir)
    if use_cache:
        train_index = load_index(cfg, "train")
        cache = load_feature_cache(train_index, cfg, verdict_name, feature_cache_dir)
        if cache is None:
            cache = build_feature_cache(train_index, cfg, verdict, feature_cache_dir)
        train_ds = CachedMaskDataset(train_index, cfg, cache, device="cpu")
        train_collate = cached_mask_collate
        log.info("region-head training on cached backbone features (train=%d)", len(train_ds))
    else:
        train_collate = mask_sample_collate
    workers = int(os.environ.get("RLROINET_WORKERS", cfg.train.num_workers))
    prefetch = max(1, int(os.environ.get("RLROINET_PREFETCH", cfg.train.prefetch_factor)))
    train_loader = make_data_loader(train_ds, cfg.train.batch_size, True, workers,
                                    prefetch, collate_fn=train_collate)
    log.info("region-head training: verdict=%s feat=%d cls=%s train=%d val=%d test=%d device=%s%s",
             verdict_name, verdict.feature_channels, with_classifier,
             len(train_ds), len(val_ds), len(test_ds), device,
             " (feature-cache)" if use_cache else "")

    head_lr = float(os.environ.get("RLROINET_HEAD_LR", cfg.train.lr_head))
    if head_lr <= 0.0:
        raise ValueError("RLROINET_HEAD_LR must be positive")
    cfg.train.lr_head = head_lr
    optim = torch.optim.AdamW(agent.trainable_parameters(), lr=head_lr,
                              weight_decay=cfg.train.weight_decay)
    scaler = make_scaler(cfg)

    epochs = epochs or cfg.train.epochs
    selection_key = "video_auc" if select_video_validation else "auc"
    selection_label = "validation_video_auc" if select_video_validation else "validation_auc"
    best_score = -float("inf")
    history = []
    start_epoch = 1

    if resume:
        found = _latest_region_checkpoint(cfg.train.checkpoint_dir, verdict_name)
        if found is not None:
            resume_path, saved_epoch = found
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
            agent.region.load_state_dict(ckpt["region_state"])
            if agent.classifier is not None and ckpt.get("classifier_state"):
                agent.classifier.load_state_dict(ckpt["classifier_state"])
            if isinstance(ckpt.get("optimizer_state"), dict):
                try:
                    optim.load_state_dict(ckpt["optimizer_state"])
                except (RuntimeError, ValueError) as exc:
                    log.warning("optimizer state incompatible after adding the classifier "
                                "head; starting the optimizer fresh (%s)", exc)
            start_epoch = saved_epoch + 1
            history = _load_region_history(cfg.train.metrics_path, start_epoch)
            if history:
                best_score = max(float(m.get(selection_key, -float("inf"))) for m in history)
            log.info("resuming region-head training for '%s' from epoch %d (%s)",
                     verdict_name, saved_epoch, resume_path.name)
            # If we now train a classifier but the saved checkpoint predates it,
            # extend the run so the new head is actually trained.
            has_cls = bool(ckpt.get("classifier_state"))
            if not has_cls and with_classifier and saved_epoch >= epochs:
                epochs = saved_epoch + epochs
                log.info("resuming into classifier training: extending run to %d epochs "
                         "(saved head at epoch %d has no classifier)", epochs, saved_epoch)
            if saved_epoch >= epochs and (not with_classifier or has_cls):
                log.info("checkpoint already at epoch %d >= requested %d; nothing to train",
                         saved_epoch, epochs)
                return history[-1] if history else {}
        else:
            log.info("no previous region-head checkpoint for '%s'; starting fresh", verdict_name)

    for ep in range(start_epoch, epochs + 1):
        ep_losses = train_one_epoch_region(agent, train_loader, cfg, optim, scaler,
                                           epoch=ep, total_epochs=epochs)
        val_metrics = evaluate(agent, val_ds, cfg, device="cuda")
        if select_video_validation:
            video_val_metrics = evaluate(agent, val_ds, cfg, device="cuda", video_level=True)
            val_metrics.update({
                "video_auc": video_val_metrics.get("auc", float("nan")),
                "video_acc": video_val_metrics.get("acc", float("nan")),
                "video_precision": video_val_metrics.get("precision", float("nan")),
                "video_recall": video_val_metrics.get("recall", float("nan")),
                "video_fp_rate": video_val_metrics.get("fp_rate", float("nan")),
                "video_fn_rate": video_val_metrics.get("fn_rate", float("nan")),
                "video_n": video_val_metrics.get("n", 0),
            })
        val_metrics["epoch"] = ep
        val_metrics.update({f"train_{k}": v for k, v in ep_losses.items()})
        history.append(val_metrics)
        save_report(history, cfg.train.metrics_path)
        selection_score = float(val_metrics.get(selection_key, -float("inf")))
        log.info("epoch %02d/%d train cls=%.4f mask=%.4f region=%.4f | val ACC=%.4f faceAUC=%.4f %s=%.4f regionIoU=%.4f regionHit=%.4f",
                 ep, epochs, ep_losses.get("cls_loss", 0.0), ep_losses["mask_loss"],
                 ep_losses["region_loss"], val_metrics["acc"], val_metrics.get("auc", 0.0),
                 selection_label, selection_score, val_metrics.get("region_iou", 0.0),
                 val_metrics.get("region_hit", 0.0))
        # Model selection is validation-only. The held-out test partition is
        # never consulted until the run is complete.
        if selection_score > best_score:
            best_score = selection_score
            torch.save(_region_ckpt(cfg, verdict_name, agent, optim, ep, val_metrics),
                       os.path.join(cfg.train.checkpoint_dir, "best.pt"))
        if ep in snapshot_epochs:
            torch.save(_region_ckpt(cfg, verdict_name, agent, optim, ep, val_metrics),
                       ckpt_dir / f"epoch_{ep:02d}.pt")

    test_metrics = evaluate(agent, test_ds, cfg, device="cuda")
    test_metrics["epoch"] = epochs
    test_metrics["selected_by"] = selection_label
    test_metrics["validation_best_metric"] = selection_key
    test_metrics["validation_best_score"] = best_score
    test_metrics.update({f"train_{k}": v for k, v in ep_losses.items()})
    torch.save(_region_ckpt(cfg, verdict_name, agent, optim, epochs, test_metrics),
               os.path.join(cfg.train.checkpoint_dir, "final.pt"))
    log.info("region-head training complete (held-out test): %s",
             json.dumps(test_metrics, indent=2, default=str))
    return test_metrics


def train(cfg: Config, epochs: int = None) -> dict:
    cfg.resolve_paths()
    configure_reproducibility(cfg)
    if cfg.train.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("supervised v2 training requires CUDA on the RTX 4050 Docker runtime")
    Path(cfg.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train.log_dir).mkdir(parents=True, exist_ok=True)
    write_run_config(cfg.train.checkpoint_dir, cfg, trainer="train", epochs=int(epochs or cfg.train.epochs))

    device = torch.device("cuda")
    train_ds = make_mask_dataset(cfg, "train", device="cpu")
    test_ds = make_mask_dataset(cfg, "test", device="cpu")
    if not len(train_ds) or not len(test_ds):
        raise RuntimeError("training and test datasets must both contain samples")
    log.info("train faces: %d | test faces: %d | device: %s", len(train_ds), len(test_ds), device)

    train_loader = make_data_loader(train_ds, cfg.train.batch_size, True,
                                    int(os.environ.get("RLROINET_WORKERS", cfg.train.num_workers)),
                                    max(1, int(os.environ.get("RLROINET_PREFETCH", cfg.train.prefetch_factor))),
                                    collate_fn=mask_sample_collate)

    agent = Agent(cfg).to(device)
    optim = torch.optim.AdamW(agent.param_groups(), weight_decay=cfg.train.weight_decay)
    scaler = make_scaler(cfg)

    epochs = epochs or cfg.train.epochs
    best_auc = -1.0
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        ep_losses = train_one_epoch(agent, train_loader, cfg, optim, scaler)
        metrics = evaluate(agent, test_ds, cfg, device="cuda")
        metrics["epoch"] = ep
        metrics["train_seconds"] = round(time.time() - t0, 1)
        metrics.update({f"train_{k}": v for k, v in ep_losses.items()})
        history.append(metrics)
        save_report(history, cfg.train.metrics_path)
        log.info("epoch %02d/%d train cls=%.4f mask=%.4f | test ACC=%.4f AUC=%.4f F1=%.4f EER=%.4f regionIoU=%.4f regionHit=%.4f ECE=%.4f",
                 ep, epochs, ep_losses["cls"], ep_losses["mask"], metrics["acc"],
                 metrics["auc"], metrics["f1"], metrics["eer"],
                 metrics.get("region_iou", 0.0), metrics.get("region_hit", 0.0), metrics.get("ece", 0.0))
        artifact = {"checkpoint_format": "supervised-v2", "model": agent.state_dict(),
                    "cfg": cfg.to_dict(), "epoch": ep, "metrics": metrics}
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(artifact, os.path.join(cfg.train.checkpoint_dir, "best.pt"))

    torch.save({"checkpoint_format": "supervised-v2", "model": agent.state_dict(),
                "cfg": cfg.to_dict(), "epoch": epochs, "metrics": metrics},
               os.path.join(cfg.train.checkpoint_dir, "final.pt"))
    log.info("training complete: %s", json.dumps(metrics, indent=2, default=str))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train supervised ROI-Net v2 on the RTX 4050")
    ap.add_argument("--source", choices=["deepfakebench", "ffpp", "celebdf_raw", "dfdcp"], default=None)
    ap.add_argument("--mask-dir", default=None, help="DeepfakeBench face+mask root")
    ap.add_argument("--celebdf-dir", default=None, help="raw Celeb-DF root")
    ap.add_argument("--ffpp-dir", default=None, help="FF++ preprocessed GT-mask root")
    ap.add_argument("--dfdcp-dir", default=None, help="DFDCP manifest + extracted face-frame root")
    ap.add_argument("--backbone", default=None, choices=["efficientnet_b4", "efficientnet_b3", "efficientnet_b1", "resnet18", "custom"])
    ap.add_argument("--verdict", default=None, choices=["honi05", "deepfakebench_xception"],
                    help="head-only region training on a frozen verdict backbone")
    ap.add_argument("--classifier", dest="with_classifier", action="store_true", default=None,
                    help="train a fake/real classifier head alongside the region head (default)")
    ap.add_argument("--no-classifier", dest="with_classifier", action="store_false",
                    help="train the region head only (frozen verdict stays the classifier)")
    ap.add_argument("--methods", default=None,
                    help="comma-separated FF++ methods to train on (e.g. --methods "
                         "Deepfakes,Face2Face,NeuralTextures to hold out FaceSwap)")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="save checkpoints/metrics here instead of outputs/checkpoints "
                         "(e.g. a per-experiment folder so runs do not overwrite each other)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel CPU DataLoader workers (decode overlaps the GPU)")
    ap.add_argument("--prefetch-factor", type=int, default=None)
    ap.add_argument("--freeze-blocks", type=int, default=None)
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="learning rate for trainable backbone blocks")
    ap.add_argument("--train-sample", type=int, default=None)
    ap.add_argument("--test-sample", type=int, default=None)
    ap.add_argument("--val-sample", type=int, default=None,
                    help="validation faces per class; used for checkpoint selection")
    ap.add_argument("--manifest-frames-per-video", type=int, default=None,
                    help="FF++ mask-supervised frames retained per source clip; inference remains 32 frames")
    ap.add_argument("--enhanced-classifier", action="store_true",
                    help="use avg+max spatial evidence and frozen verdict logit in the head")
    ap.add_argument("--select-video-validation", action="store_true",
                    help="select region-head checkpoints by video-level validation AUC")
    ap.add_argument("--focal-alpha", type=float, default=None,
                    help="fake-target focal-loss weight; must be between 0 and 1")
    ap.add_argument("--resume", dest="resume", action="store_true", default=None,
                    help="continue from the latest region-head checkpoint (default)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="start training fresh, ignoring previous checkpoints")
    ap.add_argument("--snapshot-epochs", default="5,10,15,20",
                    help="comma-separated epochs at which to save epoch_NN.pt checkpoints")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "cuda"), choices=["cuda"])
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16",
                    help="mixed-precision compute dtype (bf16 is faster on Ada and needs no GradScaler)")
    ap.add_argument("--seed", type=int, default=None,
                    help="training and split seed (defaults to the config seed)")
    ap.add_argument("--deterministic", action="store_true", default=None,
                    help="use deterministic algorithms and disable cuDNN benchmarking")
    ap.add_argument("--feature-cache", default="",
                    help="directory for precomputed frozen-backbone features; "
                         "region-head training skips the backbone forward when present")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = default_config()
    if args.seed is not None:
        cfg.data.seed = args.seed
        cfg.train.seed = args.seed
    if args.deterministic is not None:
        cfg.train.deterministic = args.deterministic
    if args.source:
        cfg.data.source = args.source
    if args.mask_dir:
        cfg.data.mask_dir = args.mask_dir
    if args.celebdf_dir:
        cfg.data.celebdf_dir = args.celebdf_dir
    if args.ffpp_dir:
        cfg.data.ffpp_dir = args.ffpp_dir
    if args.dfdcp_dir:
        cfg.data.dfdcp_dir = args.dfdcp_dir
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch is not None:
        cfg.train.batch_size = args.batch
    if args.workers is not None:
        if args.workers < 0:
            raise SystemExit("--workers must be non-negative")
        os.environ["RLROINET_WORKERS"] = str(args.workers)
    if args.prefetch_factor is not None:
        if args.prefetch_factor < 1:
            raise SystemExit("--prefetch-factor must be at least 1")
        os.environ["RLROINET_PREFETCH"] = str(args.prefetch_factor)
    if args.freeze_blocks is not None:
        if args.freeze_blocks < 0:
            raise SystemExit("--freeze-blocks must be non-negative")
        cfg.model.freeze_blocks = args.freeze_blocks
    if args.backbone_lr is not None:
        if args.backbone_lr <= 0.0:
            raise SystemExit("--backbone-lr must be positive")
        cfg.train.lr_backbone = args.backbone_lr
    if args.train_sample is not None:
        cfg.data.sample_train = args.train_sample
    if args.test_sample is not None:
        cfg.data.sample_test = args.test_sample
    if args.val_sample is not None:
        cfg.data.sample_val = args.val_sample
    if args.manifest_frames_per_video is not None:
        if args.manifest_frames_per_video < 1:
            raise SystemExit("--manifest-frames-per-video must be at least 1")
        cfg.data.manifest_frames_per_video = args.manifest_frames_per_video
    if args.enhanced_classifier:
        cfg.model.classifier_use_verdict = True
        cfg.model.classifier_pooling = "avg_max"
    if args.focal_alpha is not None:
        if not 0.0 < args.focal_alpha < 1.0:
            raise SystemExit("--focal-alpha must be between 0 and 1")
        cfg.loss.focal_alpha = args.focal_alpha
    if args.amp is not None:
        cfg.train.amp = args.amp
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 is not supported on this GPU; use --amp-dtype fp16")
    cfg.train.amp_dtype = args.amp_dtype

    if args.checkpoint_dir:
        cfg.train.checkpoint_dir = args.checkpoint_dir
        cfg.train.log_dir = os.path.join(args.checkpoint_dir, "logs")
        cfg.train.metrics_path = os.path.join(args.checkpoint_dir, "metrics.json")
    if args.verdict is not None:
        resume = True if args.resume is None else args.resume
        snapshots = tuple(int(x) for x in args.snapshot_epochs.split(",") if x.strip())
        with_classifier = True if args.with_classifier is None else args.with_classifier
        if args.methods:
            cfg.data.methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
        train_region_head(cfg, args.verdict, resume=resume,
                          snapshot_epochs=snapshots, with_classifier=with_classifier,
                          select_video_validation=args.select_video_validation,
                          feature_cache_dir=args.feature_cache)
        return

    if cfg.data.source == "deepfakebench" and not Path(cfg.data.mask_dir).exists():
        raise SystemExit(
            f"mask data not found at {cfg.data.mask_dir}; mount DeepfakeBench data into Docker first")
    train(cfg)


if __name__ == "__main__":
    main()