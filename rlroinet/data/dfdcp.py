"""DFDCP manifest-backed face-frame loader.

DFDCP is used here as an external robustness-training source. Its extracted
frames are aligned face crops and it does not provide the pixel forgery masks
required for localization ground truth, so fake samples receive the existing
synthetic ROI mask fallback in ``MaskDataset``. Official train/test tags are
preserved and source groups are never split across train/validation/test.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

log = logging.getLogger("rlroinet.dfdcp")


def _source_group(meta: dict, stem: str) -> str:
    """Return the source-video identity used as the leakage barrier."""
    source = str(meta.get("source_video") or "").strip()
    if source:
        return source
    # Original-video names end in a three-digit clip suffix (for example
    # 1003254_A_001); fake records carry source_video explicitly.
    head, sep, tail = stem.rpartition("_")
    return head if sep and tail.isdigit() else stem


def _frame_dir(root: Path, record_key: str) -> Path:
    parts = Path(record_key).parts
    if len(parts) < 2:
        raise ValueError(f"invalid DFDCP manifest key: {record_key!r}")
    source_dir = "original_videos" if parts[0] == "original_videos" else parts[0]
    return root / source_dir / "frames" / Path(parts[-1]).stem


def _pick_frames(frame_dir: Path, limit: int) -> List[Path]:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        return []
    limit = max(1, int(limit))
    if len(frames) <= limit:
        return frames
    positions = np.linspace(0, len(frames) - 1, limit).astype(int)
    return [frames[int(i)] for i in positions]


def _select_group_keys(groups: Dict[str, List[dict]], target_per_class: int,
                       rng: np.random.Generator) -> List[str]:
    """Select complete source groups until both class budgets are reached."""
    if target_per_class <= 0:
        return []
    keys = list(groups)
    rng.shuffle(keys)
    counts = {0: 0, 1: 0}
    chosen: List[str] = []
    for key in keys:
        chosen.append(key)
        for item in groups[key]:
            counts[int(item["label"])] += 1
        if counts[0] >= target_per_class and counts[1] >= target_per_class:
            break
    if counts[0] < target_per_class or counts[1] < target_per_class:
        raise ValueError(
            "DFDCP cannot satisfy balanced class budget "
            f"{target_per_class}: selected real={counts[0]} fake={counts[1]}"
        )
    return chosen


def _flatten_balanced(groups: Dict[str, List[dict]], keys: List[str],
                      target_per_class: int, rng: np.random.Generator) -> List[dict]:
    """Materialize exact class budgets without changing selected source groups."""
    candidates: Dict[int, List[dict]] = {0: [], 1: []}
    for key in keys:
        for item in groups[key]:
            candidates[int(item["label"])].append(item)

    selected: List[dict] = []
    for label in (0, 1):
        if len(candidates[label]) < target_per_class:
            raise ValueError(
                "DFDCP cannot materialize balanced class budget "
                f"{target_per_class}: available label={label}={len(candidates[label])}"
            )
        order = np.arange(len(candidates[label]))
        rng.shuffle(order)
        selected.extend(candidates[label][int(i)] for i in order[:target_per_class])
    selected.sort(key=lambda item: (str(item["video"]), str(item["frame"])))
    return selected


def index_dfdcp_faces(root: Path, cfg) -> Dict[str, List[dict]]:
    """Build balanced, source-disjoint DFDCP train/val/test face indices.

    DFDCP's manifest describes videos while the extracted tree stores sampled
    face frames. We retain at most ``frames_per_video`` frames per video and
    select complete source groups according to the configured face budgets.
    """
    root = Path(root)
    manifest_path = root / "dataset.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"DFDCP manifest not found at {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DFDCP dataset.json must contain an object")

    records: List[dict] = []
    group_partitions: Dict[str, set] = defaultdict(set)
    for record_key, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        label_text = str(meta.get("label", "")).lower()
        if label_text not in {"real", "fake"}:
            continue
        stem = Path(record_key).stem
        group = _source_group(meta, stem)
        partition = "test" if str(meta.get("set", "train")).lower() == "test" else "train"
        frame_dir = _frame_dir(root, record_key)
        frames = _pick_frames(frame_dir, cfg.data.frames_per_video)
        if not frames:
            log.warning("DFDCP record has no extracted frames: %s", record_key)
            continue
        label = int(label_text == "fake")
        method = Path(record_key).parts[0]
        group_partitions[group].add(partition)
        for frame in frames:
            records.append({
                "label": label,
                "video": f"source/{group}",
                "frame": frame.stem,
                "face": frame,
                "mask": None,
                "landmark": root / method / "landmarks" / stem / f"{frame.stem}.npy",
                "method": method,
                "partition": partition,
                "dfdcp_synthetic_mask": True,
            })

    if not records:
        raise FileNotFoundError(f"no extracted DFDCP face frames found under {root}")

    # If any source group appears in both official partitions, the test side is
    # authoritative and the train-side duplicate is excluded from all fitting.
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    dropped = 0
    for item in records:
        group = item["video"].split("/", 1)[1]
        effective = "test" if "test" in group_partitions[group] else "train"
        if effective == "test" and item["partition"] == "train":
            dropped += 1
            continue
        grouped[(effective, group)].append(item)

    train_groups = {group: items for (part, group), items in grouped.items()
                    if part == "train"}
    test_groups = {group: items for (part, group), items in grouped.items()
                   if part == "test"}
    rng = np.random.default_rng(int(cfg.data.seed))

    val_keys = set(_select_group_keys(train_groups, int(cfg.data.sample_val), rng))
    remaining_train = {key: value for key, value in train_groups.items()
                       if key not in val_keys}
    train_keys = _select_group_keys(remaining_train, int(cfg.data.sample_train), rng)
    test_keys = _select_group_keys(test_groups, int(cfg.data.sample_test), rng)

    result = {
        "train": _flatten_balanced(remaining_train, train_keys,
                                   int(cfg.data.sample_train), rng),
        "val": _flatten_balanced(train_groups, sorted(val_keys),
                                 int(cfg.data.sample_val), rng),
        "test": _flatten_balanced(test_groups, test_keys,
                                  int(cfg.data.sample_test), rng),
    }
    log.info(
        "indexed DFDCP: train=%d val=%d test=%d source_groups=%d dropped_cross_partition=%d; "
        "localization masks are synthetic fallbacks",
        *(len(result[name]) for name in ("train", "val", "test")),
        len(group_partitions), dropped,
    )
    return result


__all__ = ["index_dfdcp_faces"]
