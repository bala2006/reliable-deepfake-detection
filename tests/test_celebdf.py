from pathlib import Path

import cv2
import numpy as np

from rlroinet.config import default_config
from rlroinet.data import (
    index_gt_faces, index_raw_celebdf_videos, make_dataset, make_mask_dataset, split_index,
)
from rlroinet.data.celebdf import locate_celebdf_root
from rlroinet.data.maskdata import MaskDataset
from rlroinet.data.synthetic import FaceLayout, label_roi, roi_mask

REAL_DIRS = ("celeb-real", "youtube-real")
FAKE_DIRS = ("celeb-synthesis",)


def _write_png(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def _make_video(path: Path, frames=4, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(frames):
        img = np.full((size, size, 3), 30 + 10 * i, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_split_index_balanced():
    items = [{"label": 1, "v": i} for i in range(10)] + [{"label": 0, "v": i} for i in range(10)]
    out = split_index(items, n_train=3, n_test=2, seed=0)
    for s in ("train", "test"):
        labels = [i["label"] for i in out[s]]
        assert labels.count(0) == labels.count(1)
    train_ids = {(i["label"], i["v"]) for i in out["train"]}
    test_ids = {(i["label"], i["v"]) for i in out["test"]}
    assert train_ids.isdisjoint(test_ids)


def test_index_gt_faces(tmp_path):
    root = tmp_path / "gt"
    for cls in ("Celeb-real", "Celeb-synthesis"):
        vdir = root / cls / "frames" / "v1"
        mdir = root / cls / "masks" / "v1"
        for i in range(6):
            _write_png(vdir / f"{i:04d}.png", np.full((64, 64, 3), 120, dtype=np.uint8))
            _write_png(mdir / f"{i:04d}.png", np.zeros((64, 64), dtype=np.uint8))
    idx = index_gt_faces(root, frames_per_video=3)
    labels = [i["label"] for i in idx]
    assert set(labels) == {0, 1}
    assert len(idx) == 6
    for i in idx:
        assert i["face"].suffix == ".png"
        assert i["mask"] is not None


def test_index_raw_celebdf_videos(tmp_path):
    root = tmp_path / "raw"
    _make_video(root / "Celeb-real" / "r1.mp4")
    _make_video(root / "Celeb-synthesis" / "f1.mp4")
    idx = index_raw_celebdf_videos(root)
    assert len(idx) == 2
    assert {i["label"] for i in idx} == {0, 1}


def test_locate_celebdf_root(tmp_path):
    root = tmp_path / "wrapped" / "Celeb-DF"
    (root / "Celeb-real").mkdir(parents=True)
    (root / "Celeb-synthesis").mkdir(parents=True)
    found = locate_celebdf_root(tmp_path)
    assert found == root


def test_mask_dataset_reads_gt(tmp_path):
    root = tmp_path / "gt"
    for cls, label in (("Celeb-real", 0), ("Celeb-synthesis", 1)):
        vdir = root / cls / "frames" / "v1"
        mdir = root / cls / "masks" / "v1"
        for i in range(2):
            _write_png(vdir / f"{i:04d}.png", np.full((64, 64, 3), 120, dtype=np.uint8))
            mask = np.zeros((64, 64), dtype=np.uint8)
            if label:
                mask[16:48, 16:48] = 255
            _write_png(mdir / f"{i:04d}.png", mask)
    cfg = default_config()
    cfg.data.face_size = 32
    cfg.data.source = "deepfakebench"
    cfg.data.mask_dir = str(root)
    idx = index_gt_faces(root, frames_per_video=2)
    ds = MaskDataset(idx, cfg, device="cpu")
    assert len(ds) == 4
    sample = ds[0]
    assert sample.face.shape == (3, 32, 32)
    assert sample.mask.shape == (32, 32)
    assert sample.label in (0, 1)
    # fake samples must carry non-empty masks (GT or pseudo)
    fake = [ds[i] for i in range(len(ds)) if ds.index[i]["label"] == 1]
    assert all(bool(s.mask.any()) for s in fake)


def test_make_mask_dataset_missing_dir_raises(tmp_path):
    cfg = default_config()
    cfg.data.source = "deepfakebench"
    cfg.data.mask_dir = str(tmp_path / "nope")
    try:
        make_mask_dataset(cfg, "train")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing mask dir")


def test_roi_label_and_masks():
    from rlroinet.data.synthetic import FaceLayout
    L = FaceLayout()
    assert label_roi(L.eye_left[0], L.eye_left[1], L) == 1
    assert label_roi(L.mouth[0], L.mouth[1], L) == 3
    assert label_roi(0.9, 0.9, L) == 0
    masks = roi_mask(L, 64)
    assert set(masks) == {1, 2, 3, 4}
    for m in masks.values():
        assert m.shape == (64, 64)
        assert bool(m.any())


def test_ffpp_factory_indexes_cross_domain_videos(tmp_path):
    root = tmp_path / "ffpp"
    _make_video(root / "original_sequences" / "youtube" / "c23" / "videos" / "r1.mp4")
    _make_video(root / "original_sequences" / "youtube" / "c23" / "videos" / "r2.mp4")
    _make_video(root / "manipulated_sequences" / "DeepFakes" / "c23" / "videos" / "f1.mp4")
    _make_video(root / "manipulated_sequences" / "DeepFakes" / "c23" / "videos" / "f2.mp4")
    cfg = default_config()
    cfg.data.source = "ffpp"
    cfg.data.ffpp_dir = str(root)
    cfg.data.methods = ("DeepFakes",)
    cfg.data.sample_train = 1
    cfg.data.sample_test = 1
    ds = make_dataset(cfg, "test", device="cpu")
    assert len(ds) == 2
    assert {ds[i]["label"] for i in range(len(ds))} == {0, 1}



def test_official_ffpp_split_groups_source_pairs(tmp_path):
    import json

    for name, pairs in {
        "train": [["000", "001"]],
        "val": [["002", "003"]],
        "test": [["004", "005"]],
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(pairs), encoding="utf-8")

    items = []
    for source, pair in (("youtube", "000"), ("youtube", "001"),
                         ("youtube", "002"), ("youtube", "003"),
                         ("youtube", "004"), ("youtube", "005")):
        items.append({"label": 0, "video": f"{source}/{pair}", "frame": "0"})
    for method in ("Deepfakes", "Face2Face"):
        for pair, label in (("000_001", 1), ("002_003", 1), ("004_005", 1)):
            items.append({"label": label, "video": f"{method}/{pair}", "frame": "0"})
    items.append({"label": 1, "video": "Deepfakes/998_999", "frame": "0"})
    items.append({
        "label": 1, "video": "FaceSwap/002_003", "frame": "0", "partition": "train",
    })

    out = split_index(
        items, n_train=1, n_test=1, seed=0, n_val=1,
        ffpp_dir=tmp_path, leakage_safe=True,
    )
    locations = {
        item["video"]: partition
        for partition, values in out.items()
        for item in values
    }
    assert locations["youtube/000"] == locations["Deepfakes/000_001"] == "train"
    assert locations["youtube/002"] == locations["Face2Face/002_003"] == "val"
    assert locations["FaceSwap/002_003"] == "val"
    assert locations["youtube/004"] == locations["Deepfakes/004_005"] == "test"
    assert locations["Deepfakes/998_999"] == "train"


def test_raw_celebdf_uses_clip_disjoint_nonempty_splits(tmp_path):
    root = tmp_path / "raw"
    for label_dir in ("Celeb-real", "Celeb-synthesis"):
        for number in range(6):
            _make_video(root / label_dir / f"clip_{number}.mp4")

    index = index_raw_celebdf_videos(root)
    assert len({item["video"] for item in index}) == len(index)
    assert all("/" in item["video"] for item in index)
    cfg = default_config()
    splits = split_index(
        index, n_train=2, n_test=2, n_val=2, seed=0,
        ffpp_dir=tmp_path / "no_ffpp_needed",
        leakage_safe=cfg.data.leakage_safe_split,
    )
    assert all(splits[name] for name in ("train", "val", "test"))
    videos = {name: {item["video"] for item in values}
              for name, values in splits.items()}
    assert videos["train"].isdisjoint(videos["val"])
    assert videos["train"].isdisjoint(videos["test"])
    assert videos["val"].isdisjoint(videos["test"])


def test_landmark_layout_is_used_and_missing_landmarks_fall_back(tmp_path):
    root = tmp_path / "gt"
    face_path = root / "face.png"
    _write_png(face_path, np.full((256, 256, 3), 120, dtype=np.uint8))
    points = np.zeros((81, 2), dtype=np.float32)
    points[:17] = np.column_stack((np.linspace(40, 215, 17),
                                   np.array([130, 150, 172, 195, 215, 230, 242, 248,
                                             250, 246, 238, 225, 207, 185, 165, 145, 128])))
    points[17:27] = np.array([
        [55, 118], [68, 108], [84, 104], [100, 106], [114, 112],
        [140, 112], [156, 106], [174, 104], [190, 108], [202, 118],
    ])
    points[27:36] = np.array([
        [128, 120], [128, 135], [128, 150], [128, 165], [114, 176],
        [121, 178], [128, 179], [135, 178], [142, 176],
    ])
    points[36:42] = np.array([
        [66, 128], [76, 122], [90, 122], [101, 128], [90, 133], [76, 133],
    ])
    points[42:48] = np.array([
        [150, 128], [161, 122], [175, 122], [187, 128], [175, 133], [161, 133],
    ])
    points[48:68] = np.array([
        [104, 198], [114, 188], [124, 184], [134, 184], [144, 188],
        [154, 198], [145, 210], [135, 216], [128, 218], [121, 216],
        [111, 210], [112, 204], [110, 198], [120, 194], [128, 193], [136, 194],
        [148, 198], [136, 202], [128, 203], [120, 202],
    ])
    points[68:75] = np.array([
        [55, 70], [75, 62], [100, 58], [128, 56], [160, 58], [184, 64], [202, 78],
    ])
    points[75:81] = np.array([
        [45, 100], [48, 82], [42, 128], [211, 124], [190, 75], [158, 58],
    ])
    landmark_path = root / "landmarks.npy"
    np.save(landmark_path, points)
    cfg = default_config()
    cfg.data.face_size = 64
    item = {"label": 1, "video": "fake/clip", "frame": "0",
            "face": face_path, "mask": None, "landmark": landmark_path}
    sample = MaskDataset([item], cfg)[0]
    fallback = FaceLayout()
    assert sample.layout.eye_left != fallback.eye_left
    assert not np.array_equal(roi_mask(sample.layout, 64)[1], roi_mask(fallback, 64)[1])

    missing = dict(item, landmark=root / "does-not-exist.npy")
    fallback_sample = MaskDataset([missing], cfg)[0]
    assert fallback_sample.layout == fallback

    malformed_path = root / "malformed.npy"
    np.save(malformed_path, np.zeros((81, 2), dtype=np.float32))
    malformed_sample = MaskDataset([dict(item, landmark=malformed_path)], cfg)[0]
    assert malformed_sample.layout == fallback


def test_default_leakage_safe_split_keeps_official_pairs_together(tmp_path):
    import json

    assert default_config().data.leakage_safe_split is True
    for name, pairs in {
        "train": [["000", "001"]],
        "val": [["002", "003"]],
        "test": [["004", "005"]],
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(pairs), encoding="utf-8")
    items = []
    for source, pair in (("youtube", "000"), ("youtube", "001"),
                         ("youtube", "002"), ("youtube", "003"),
                         ("youtube", "004"), ("youtube", "005")):
        items.append({"label": 0, "video": f"{source}/{pair}", "frame": "0"})
    for method in ("Deepfakes", "FaceSwap"):
        for pair in ("000_001", "002_003", "004_005"):
            items.append({"label": 1, "video": f"{method}/{pair}", "frame": "0"})

    out = split_index(items, n_train=1, n_test=1, n_val=1, seed=0,
                      ffpp_dir=tmp_path, leakage_safe=True)
    ownership = {}
    for partition, values in out.items():
        for item in values:
            name = item["video"].split("/", 1)[1]
            tokens = (name,) if item["video"].startswith("youtube/") else tuple(name.split("_", 2)[:2])
            for token in tokens:
                ownership.setdefault(token, set()).add(partition)
    assert all(len(partitions) == 1 for partitions in ownership.values())


def test_default_leakage_flag_preserves_non_ffpp_dataset_splits(tmp_path):
    root = tmp_path / "gt"
    for cls, label in (("Celeb-real", 0), ("Celeb-synthesis", 1)):
        for number in range(3):
            frame_dir = root / cls / "frames" / f"clip_{number}"
            mask_dir = root / cls / "masks" / f"clip_{number}"
            _write_png(frame_dir / "0000.png", np.full((32, 32, 3), 120, dtype=np.uint8))
            _write_png(mask_dir / "0000.png", np.full((32, 32), 255 if label else 0, dtype=np.uint8))
    cfg = default_config()
    cfg.data.source = "deepfakebench"
    cfg.data.mask_dir = str(root)
    cfg.data.frames_per_video = 1
    cfg.data.sample_train = 1
    cfg.data.sample_val = 1
    cfg.data.sample_test = 1
    cfg.data.face_size = 16
    dataset = make_mask_dataset(cfg, "test")
    assert len(dataset) == 2
    assert {dataset.index[i]["label"] for i in range(len(dataset))} == {0, 1}
