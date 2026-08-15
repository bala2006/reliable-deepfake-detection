import json
from pathlib import Path

import cv2
import numpy as np

from rlroinet.config import default_config
from rlroinet.data import index_dfdcp_faces
from rlroinet.data.maskdata import MaskDataset


def _png(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((24, 24, 3), value, dtype=np.uint8))


def test_dfdcp_manifest_loader_is_grouped_and_balanced(tmp_path):
    root = tmp_path / "DFDCP"
    records = {}
    for group_num in range(4):
        source = f"{1000 + group_num}_A"
        partition = "test" if group_num == 3 else "train"
        original_stem = f"{source}_001"
        fake_stem = f"{2000 + group_num}_{source}_001"
        records[f"original_videos/{source}/{original_stem}.mp4"] = {
            "label": "real", "set": partition,
        }
        records[f"method_A/{source}/{source}/{fake_stem}.mp4"] = {
            "label": "fake", "set": partition, "source_video": source,
        }
        for base, stem in (("original_videos", original_stem), ("method_A", fake_stem)):
            for frame in range(2):
                _png(root / base / "frames" / stem / f"{frame:03d}.png", 80 + frame)
    (root / "dataset.json").write_text(json.dumps(records), encoding="utf-8")

    cfg = default_config()
    cfg.data.dfdcp_dir = str(root)
    cfg.data.frames_per_video = 2
    cfg.data.sample_train = 2
    cfg.data.sample_val = 2
    cfg.data.sample_test = 2

    splits = index_dfdcp_faces(root, cfg)
    assert {split: len(items) for split, items in splits.items()} == {
        "train": 4, "val": 4, "test": 4,
    }
    groups = {
        split: {item["video"] for item in items}
        for split, items in splits.items()
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert all(item["mask"] is None for items in splits.values() for item in items)

    ds = MaskDataset(splits["train"], cfg)
    fake = [ds[i] for i, item in enumerate(ds.index) if item["label"] == 1]
    assert fake and all(bool(sample.mask.any()) for sample in fake)
