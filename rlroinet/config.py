"""Configuration for the supervised ROI-Net v2 detector.

This file only holds configuration. Nothing else runs on import.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple


@dataclass
class DataConfig:
    # --- source ---
    source: str = "deepfakebench"   # "deepfakebench" | "celebdf_raw" | "ffpp" | "dfdcp"
    mask_dir: str = "data/deepfakebench/celebdf-v2"   # DeepfakeBench preprocessed root (frames/masks/landmarks)
    celebdf_dir: str = "data/celebdf"                 # raw Celeb-DF videos for smoke/fallback experiments
    celebdf_zip: str = ""                             # explicit raw zip (auto-extract on first use)
    ffpp_dir: str = "data/FaceForensics++"   # FF++ preprocessed (frames/masks GT) root
    dfdcp_dir: str = "data/DFDCP"             # DFDCP manifest + extracted face frames
    compression: str = "c23"
    methods: Tuple[str, ...] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")

    # --- sampling ---
    frames_per_video: int = 32          # frames sampled per video for evaluation/inference
    # Number of mask-supervised frames retained per source FF++ clip in the
    # manifest. This is intentionally separate from inference sampling so a
    # diverse training pool does not weaken 32-frame video decisions.
    manifest_frames_per_video: int = 32
    # Ordered same-video frames consumed by the optional temporal-quality head.
    sequence_length: int = 8
    sample_train: int = 1800            # faces per class (training split)
    sample_val: int = 300               # faces per class (threshold/checkpoint validation split)
    sample_test: int = 300              # faces/videos per class (held-out split)
    stratify_methods: bool = True       # balance FF++ methods/origins in every split
    leakage_safe_split: bool = True     # FF++ official pair-grouped split by default
    face_size: int = 288                # aligned face crop size (DeepfakeBench default)
    max_video_duration_seconds: float = 30.0
    seed: int = 0

    # --- auto-mask fallback (only when source == 'celebdf_raw' and no GT masks) ---
    auto_generate_masks: bool = True    # build pseudo-masks from face-alignment + ROI layout when no GT exists
    min_artist_frames: int = 1          # frames that must contain forgery before a label is 'fake'


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b4"   # torchvision backbone
    pretrained: bool = True             # ImageNet weights (transfer learning)
    freeze_blocks: int = 4              # freeze the first N backbone blocks; train the rest + heads
    feat_dim: int = 256                 # classifier feature dimension
    classifier_hidden: int = 128
    # Optional robustness features for the head-only classifier. Disabled by
    # default so existing region-head-v2 checkpoints remain loadable.
    classifier_use_verdict: bool = False
    classifier_pooling: str = "avg"   # "avg" or "avg_max"
    # Lightweight sequence head; only used by temporal-quality-v1 experiments.
    temporal_hidden: int = 128
    temporal_quality_hidden: int = 16
    # Parameter-efficient spatial forensic adapter experiment. Existing
    # checkpoints omit this field and receive the backward-compatible default.
    adapter_bottleneck: int = 64
    # region head
    region_out_channels: Tuple[int, ...] = (256, 128, 64, 32)   # decoder channel ladder
    n_regions: int = 4                  # PERIOCULAR, JAWLINE, MOUTH, HAIRLINE
    in_ch: int = 3


@dataclass
class LossConfig:
    focal_gamma: float = 2.0            # Focal loss (class imbalance ~5:1 real:fake)
    focal_alpha: float = 0.75           # alpha for the minority (fake) class
    dice_weight: float = 1.0            # mask Dice loss weight
    bce_weight: float = 1.0             # mask BCE weight
    class_weight: float = 1.0           # classification loss weight
    region_weight: float = 0.5          # explicit per-ROI mask supervision weight


@dataclass
class TrainConfig:
    device: str = "cuda"              # RTX 4050 CUDA runtime; CPU training is unsupported
    epochs: int = 5                     # joint supervised fine-tune default
    batch_size: int = 1                 # physical video batch; accumulation supplies the effective batch
    gradient_accumulation_steps: int = 4
    num_workers: int = 4                # CPU decode workers; bounded by the training launcher
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    seed: int = 0
    deterministic: bool = False         # deterministic kernels are slower; seed is always recorded
    tf32: bool = True                   # safe throughput setting on RTX 4050/Ada
    amp: bool = True                    # mixed precision when CUDA
    amp_dtype: str = "bf16"             # "fp16" or "bf16"; bf16 is faster + no GradScaler on Ada
    lr_backbone: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True                    # mixed precision when CUDA
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    metrics_path: str = "outputs/metrics.json"


@dataclass
class EvalConfig:
    threshold: float = 0.5              # legacy binary threshold / metrics threshold
    real_threshold: float = 0.30        # below this, confidently REAL
    fake_threshold: float = 0.70        # at/above this, confidently FAKE
    review_enabled: bool = True         # scores in between require human review
    video_aggregation: str = "robust"   # mean + top-frame evidence, not mean alone
    top_fraction: float = 0.25          # fraction used for top-evidence pooling
    min_fake_fraction: float = 0.10     # minimum high-confidence frames for FAKE
    bboxes_per_region: int = 1          # top-N region boxes reported per flagged frame
    calibration_bins: int = 10          # expected-calibration-error histogram bins


@dataclass
class Config:
    schema_version: int = 2
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def resolve_paths(self) -> "Config":
        base = os.getcwd()
        t = self.train
        for attr in ("checkpoint_dir", "log_dir", "metrics_path"):
            val = getattr(t, attr)
            if val and not os.path.isabs(val):
                setattr(t, attr, os.path.join(base, val))
        for attr in ("mask_dir", "celebdf_dir", "celebdf_zip", "ffpp_dir", "dfdcp_dir"):
            val = getattr(self.data, attr)
            if val and not os.path.isabs(val):
                setattr(self.data, attr, os.path.join(base, val))
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        if not isinstance(d, dict):
            raise ValueError("unsupported checkpoint configuration: expected an object")

        allowed = {"schema_version", "data", "model", "loss", "train", "eval"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"unsupported checkpoint configuration fields: {unknown}")

        schema_version = int(d.get("schema_version", 2))
        if schema_version != 2:
            raise ValueError(f"unsupported checkpoint configuration schema: {schema_version}")

        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "loss": LossConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
        }
        values = {}
        for name, section_type in sections.items():
            section = d.get(name, {})
            if not isinstance(section, dict):
                raise ValueError(f"unsupported checkpoint configuration section: {name}")
            section_unknown = sorted(
                set(section) - {item.name for item in fields(section_type)}
            )
            if section_unknown:
                raise ValueError(
                    f"unsupported checkpoint configuration fields in {name}: "
                    f"{section_unknown}"
                )
            values[name] = dict(section)

        # Pre-validation v2 checkpoints used a two-way split. Preserve their
        # exact data contract when loaded; new configs explicitly serialize a
        # non-zero validation budget.
        if "sample_val" not in values["data"]:
            values["data"]["sample_val"] = 0
        if "fake_threshold" not in values["eval"]:
            legacy_threshold = float(values["eval"].get("threshold", 0.5))
            values["eval"].update({
                "real_threshold": legacy_threshold,
                "fake_threshold": legacy_threshold,
                "review_enabled": False,
                "video_aggregation": "mean",
            })

        if "methods" in values["data"]:
            values["data"]["methods"] = tuple(values["data"]["methods"])
        return cls(schema_version=schema_version, **{
            name: section_type(**values[name])
            for name, section_type in sections.items()
        })


def default_config() -> Config:
    return Config(schema_version=2)