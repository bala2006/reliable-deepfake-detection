"""Mask-supervised dataset (DeepfakeBench preprocessed Celeb-DF v2).

Loads aligned face crops with forgery masks + landmarks, as distributed by
DeepfakeBench (SCLBD/DeepfakeBench).  Directory layout expected:

    <mask_dir>/Celeb-real|YouTube-real|Celeb-synthesis/
        frames/<video>/<frame>.png        # aligned face crop (RGB)
        masks/<video>/<frame>.png         # per-frame forgery mask
        landmarks/<video>/<frame>.npy     # 81-point dlib landmarks
        videos/<video>.mp4                # original clip (for inference/overlay)

If the true masks are unavailable (only raw Celeb-DF videos exist), set
``DataConfig.source='celebdf_raw'`` with ``auto_generate_masks=True``; the loader
then generates pseudo-masks from the detected face layout. These masks must be
identified as synthetic supervision in evaluation reports.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from .synthetic import FaceLayout

log = logging.getLogger("rlroinet.maskdata")

REAL_DIRS = ("celeb-real", "youtube-real")
FAKE_DIRS = ("celeb-synthesis",)


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")


def _read_png(path: Path, size: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1)


def _read_mask(path: Optional[Path], size: int) -> torch.Tensor:
    if path is None or not path.exists():
        return torch.zeros(size, size)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return torch.zeros(size, size)
    img = cv2.resize(img, (size, size))
    return torch.from_numpy(img).float().div_(255.0)


def _landmark_layout(path: Optional[Path], face_path=None) -> Optional[FaceLayout]:
    """Convert dlib-style landmarks into the normalized ROI layout.

    DeepfakeBench stores 81-point landmarks in the coordinate system of the
    aligned crop.  Invalid or unavailable files deliberately return ``None``
    so callers retain the historical centered ``FaceLayout`` fallback.
    """
    if path is None:
        return None
    try:
        points = np.asarray(np.load(Path(path), allow_pickle=False))
        while points.ndim > 2 and points.shape[0] == 1:
            points = points[0]
        if points.ndim != 2 or points.shape[0] < 68 or points.shape[1] < 2:
            return None
        points = points[:, :2].astype(np.float32, copy=False)
    except (OSError, ValueError, TypeError, OverflowError):
        return None
    if not np.isfinite(points).all():
        return None
    if np.ptp(points[:, 0]) <= 1e-6 or np.ptp(points[:, 1]) <= 1e-6:
        return None

    # Normalized landmarks are accepted for small synthetic/unit-test inputs;
    # normal DeepfakeBench landmarks use pixel coordinates.
    if float(np.max(np.abs(points))) <= 1.5:
        width = height = 1.0
    else:
        width = height = 0.0
        if face_path is not None:
            try:
                image = cv2.imread(str(face_path), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    height, width = image.shape[:2]
            except (OSError, TypeError):
                pass
        if width <= 0 or height <= 0:
            width = max(1.0, float(points[:, 0].max()) + 1.0)
            height = max(1.0, float(points[:, 1].max()) + 1.0)

    jaw = points[0:17]
    brows = points[17:27]
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    mouth_points = points[48:68]
    mouth = mouth_points.mean(axis=0)
    face_min, face_max = points.min(axis=0), points.max(axis=0)
    eye_y = float(np.mean(points[36:48, 1]) / height)
    jaw_y = float(np.mean(jaw[5:12, 1]) / height)

    if points.shape[0] >= 75:
        hair_points = points[68:75]
        if points.shape[0] >= 81:
            hair_points = np.concatenate((hair_points, points[79:81]), axis=0)
        hairline_y = float(np.mean(hair_points[:, 1]) / height)
        hairline_h = max(0.02, float(np.ptp(hair_points[:, 1]) / height * 0.75))
    else:
        hairline_y = max(0.02, float(np.min(brows[:, 1]) / height - 0.08))
        hairline_h = 0.03

    return FaceLayout(
        fx=float((face_min[0] + face_max[0]) / (2.0 * width)),
        fy=jaw_y,
        fw=max(0.10, float((face_max[0] - face_min[0]) / (2.0 * width))),
        fh=max(0.08, float((jaw_y - eye_y) * 0.30)),
        eye_left=(float(left_eye[0] / width), float(left_eye[1] / height)),
        eye_right=(float(right_eye[0] / width), float(right_eye[1] / height)),
        eye_w=max(0.025, float(max(np.ptp(points[36:42, 0]),
                                   np.ptp(points[42:48, 0])) / width)),
        eye_h=max(0.02, float(np.ptp(points[36:48, 1]) / height)),
        mouth=(float(mouth[0] / width), float(mouth[1] / height)),
        mouth_w=max(0.05, float(np.ptp(mouth_points[:, 0]) / width * 0.70)),
        mouth_h=max(0.035, float(np.ptp(mouth_points[:, 1]) / height * 0.70)),
        hairline_y=hairline_y,
        hairline_h=hairline_h,
    )


class MaskFaceSample:
    """One aligned face crop with its forgery mask + layout (+ optional landmarks)."""
    __slots__ = ("face", "mask", "label", "layout", "video", "frame")

    def __init__(self, face, mask, label, layout, video, frame):
        self.face = face        # (3, H, W) float
        self.mask = mask        # (H, W) float forgery probs (all-zeros for real)
        self.label = label      # 1 fake, 0 real
        self.layout = layout    # FaceLayout
        self.video = video
        self.frame = frame


def _find_gt_root(mask_dir: Path) -> Optional[Path]:
    """Root whose class dirs hold a frames branch."""
    mask_dir = Path(mask_dir)
    if not mask_dir.exists():
        return None
    for d in [mask_dir, *sorted(mask_dir.rglob("*"))]:
        if not d.is_dir():
            continue
        subs = {p.name.lower(): p for p in d.iterdir() if p.is_dir()}
        if any(r in subs for r in REAL_DIRS) and FAKE_DIRS[0] in subs:
            if any((subs[r] / "frames").is_dir() for r in (*REAL_DIRS, *FAKE_DIRS)):
                return d
    return None


def index_gt_faces(mask_dir: Path, frames_per_video: int = 32) -> List[dict]:
    """Enumerate up to ``frames_per_video`` (video, frame, label) per source video."""
    root = _find_gt_root(mask_dir)
    if root is None:
        raise FileNotFoundError(
            f"no DeepfakeBench frames/masks tree under {mask_dir}. "
            f"Place the preprocessed Celeb-DF there, or set DataConfig.source = "
            f"'celebdf_raw' to auto-generate masks.")
    items: List[dict] = []
    classes = [d for d in root.iterdir()
               if d.is_dir() and (d.name.lower() in FAKE_DIRS or d.name.lower() in REAL_DIRS)]
    for class_dir in classes:
        label = 1 if class_dir.name.lower() in FAKE_DIRS else 0
        fdir = class_dir / "frames"
        mdir = class_dir / "masks"
        ldir = class_dir / "landmarks"
        videos = sorted([p for p in fdir.iterdir() if p.is_dir()])
        for v in videos:
            frames = sorted(v.glob("*.png"))
            step = max(1, len(frames) // frames_per_video)
            for f in frames[::step][:frames_per_video]:
                    landmark_p = ldir / v.name / (f.stem + ".npy")
                    items.append({
                        "label": label,
                        "video": v.name,
                        "frame": f.stem,
                        "face": f,
                        "mask": (mdir / v.name / f.name) if (mdir / v.name / f.name).exists() else None,
                        "landmark": landmark_p if landmark_p.exists() else None,
                        "source": "deepfakebench",
                    })
    log.info("indexed %d aligned faces from %s", len(items), root)
    return items




def _load_ffpp_manifest(root: Path, cfg, frames_per_video: int) -> Optional[List[dict]]:
    """Load a host-generated FF++ manifest when the dataset is bind-mounted.

    Docker Desktop can make directory enumeration over a Windows bind mount
    extremely slow. The manifest contains only the selected face entries, so
    training still reads the same images but avoids repeatedly walking the
    thousands of FF++ video directories.
    """
    path = root / ".rlroinet_index.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "compression": cfg.data.compression,
            "methods": list(cfg.data.methods),
            "sample_train": int(cfg.data.sample_train),
            "sample_val": int(cfg.data.sample_val),
            "sample_test": int(cfg.data.sample_test),
        }
        # Schema v2 separates the per-video manifest cap from the 32-frame
        # inference setting. Schema v1 manifests remain usable for old
        # checkpoints, where the single frame count served both roles.
        frame_key = ("manifest_frames_per_video"
                     if int(payload.get("schema_version", 1)) >= 2
                     else "frames_per_video")
        expected[frame_key] = int(frames_per_video)
        if any(payload.get(key) != value for key, value in expected.items()):
            log.warning("ignoring stale FF++ manifest: %s", path)
            return None
        items = []
        for raw in payload.get("items", []):
            item = dict(raw)
            item.setdefault("source", "ffpp")
            for key in ("face", "mask", "landmark"):
                if item.get(key):
                    item[key] = root / item[key]
            items.append(item)
        if not items:
            return None
        log.info("loaded %d FF++ faces from manifest %s", len(items), path)
        return items
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable FF++ manifest %s (%s)", path, exc)
        return None


def index_ffpp_masks(ffpp_dir: Path, cfg, frames_per_video: int = 32) -> List[dict]:
    """Index DeepfakeBench-preprocessed FaceForensics++ GT-mask faces.

    Walks the FF++ preprocessed tree (real sequences + the configured
    manipulated methods at the requested compression level):

        <ffpp>/original_sequences/youtube|c23|.../frames/<video>/*.png   (real, no masks)
        <ffpp>/manipulated_sequences/<Method>/c23/.../frames/<video>/*.png (fake)
        .../masks/<video>/<frame>.png    (per-frame forgery mask)
        .../landmarks/<video>/<frame>.npy

    The index is capped at ``sample_train + sample_test`` faces per label,
    distributed **evenly across the configured methods** so training sees every
    manipulation type (not just the first method that fills the cap). The
    ``video`` field records ``"<Method>/<video>"`` (or ``"<origin>/<video>``"
    for real sequences) so per-method evaluation can be performed later.

    Returns the same item dicts as :func:`index_gt_faces` so ``MaskDataset``
    consumes them unchanged.
    """
    root = Path(ffpp_dir)
    manifest_items = _load_ffpp_manifest(root, cfg, frames_per_video)
    if manifest_items is not None:
        return manifest_items
    comp = cfg.data.compression
    methods = tuple(cfg.data.methods)
    per_label = max(1, cfg.data.sample_train + cfg.data.sample_val + cfg.data.sample_test)
    items: List[dict] = []

    def _landmark_for(base: Path, v: Path, f: Path):
        land = base / "landmarks" / v.name / (f.stem + ".npy")
        return land if land.exists() else None

    def _mask_for(base: Path, v: Path, f: Path):
        maskp = base / "masks" / v.name / f.name
        return maskp if maskp.exists() else None

    def collect_group(videos: List[Path], label: int, base: Path,
                      source_name: str, cap: int) -> None:
        """Gather up to ``cap`` faces, striding uniformly across ``videos``."""
        need = cap - sum(1 for i in items if i["label"] == label and
                         i["video"].startswith(source_name + "/"))
        if need <= 0 or not videos:
            return
        per_video = frames_per_video
        n_videos = max(1, math.ceil(need / per_video))
        vid_step = max(1, len(videos) // n_videos)
        for v in videos[::vid_step]:
            frames = sorted(v.glob("*.png"))
            if not frames:
                continue
            step = max(1, len(frames) // frames_per_video)
            for f in frames[::step][:frames_per_video]:
                items.append({"label": label, "video": f"{source_name}/{v.name}",
                              "frame": f.stem, "face": f,
                              "mask": _mask_for(base, v, f),
                              "landmark": _landmark_for(base, v, f),
                              "source": "ffpp"})
                need -= 1
                if need <= 0:
                    return

    fake_cap = max(1, math.ceil(per_label / max(1, len(methods))))

    real_sources = []
    for orig in ("youtube", "actors"):
        fbase = root / "original_sequences" / orig / comp / "frames"
        if not fbase.exists():
            continue
        videos = sorted(p for p in fbase.iterdir() if p.is_dir())
        real_sources.append((orig, root / "original_sequences" / orig / comp, videos))

    # Divide the real budget across available origins rather than giving the
    # full class cap to every origin. This keeps real/fake totals controlled
    # and prevents one origin from dominating the balanced split.
    real_cap = max(1, math.ceil(per_label / max(1, len(real_sources))))
    for orig, base, videos in real_sources:
        collect_group(videos, 0, base, orig, real_cap)

    for m in methods:
        b = root / "manipulated_sequences" / m / comp
        fbase = b / "frames"
        if not fbase.exists():
            log.warning("missing FF++ method tree: %s", fbase)
            continue
        videos = sorted(p for p in fbase.iterdir() if p.is_dir())
        collect_group(videos, 1, b, m, fake_cap)

    if not items:
        raise FileNotFoundError(
            f"no FF++ DeepfakeBench-preprocessed tree with masks under {root}; "
            f"expected original_sequences/youtube/{comp}/frames and "
            f"manipulated_sequences/<Method>/{comp}/{{frames,masks}}")
    log.info("indexed %d FF++ GT-mask faces from %s (%d methods, cap %d/label)",
             len(items), root, len(methods), per_label)
    return items


def index_raw_celebdf_videos(celebdf_dir: Path) -> List[dict]:
    """Enumerate raw Celeb-DF clips (for classification/auto-mask sampling)."""
    celebdf_dir = Path(celebdf_dir)
    items: List[dict] = []
    for d in celebdf_dir.rglob("*"):
        if d.is_file() and _is_video(d):
            cls = d.parent.name
            label = 1 if cls.lower() in FAKE_DIRS else 0
            if cls.lower() in FAKE_DIRS or cls.lower() in REAL_DIRS:
                # Include the class and every nested directory in the identity.
                # The old class-only value grouped all clips in a class into one
                # split unit, which could empty validation and test partitions.
                video = d.relative_to(celebdf_dir).with_suffix("").as_posix()
                items.append({"label": label, "video": video,
                              "frame": d.stem, "face": d, "mask": None,
                              "source": "celebdf_raw"})
    return items


def _region_primary(mask: torch.Tensor, layout: FaceLayout, size: int) -> int:
    """Primary ROI id (PERIOCULAR=1..HAIRLINE=4) containing the most forgery mass."""
    from .synthetic import PERIOCULAR, JAWLINE, MOUTH, HAIRLINE
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    xn, yn = xx.float() / size, yy.float() / size
    L = layout
    in_roi = {
        PERIOCULAR: ((xn - L.eye_left[0]).abs() < L.eye_w) & ((yn - L.eye_left[1]).abs() < 0.09)
                  | ((xn - L.eye_right[0]).abs() < L.eye_w) & ((yn - L.eye_right[1]).abs() < 0.09),
        JAWLINE: ((yn - L.fy).abs() < L.fh * 1.15) & (yn > L.eye_left[1]),
        MOUTH: ((xn - L.mouth[0]).abs() < L.mouth_w) & ((yn - L.mouth[1]).abs() < L.mouth_h),
        HAIRLINE: ((yn - L.hairline_y).abs() < L.hairline_h),
    }
    best, best_score = 0, 0.0
    for r, mask_roi in in_roi.items():
        s = float((mask * mask_roi.float()).sum())
        if s > best_score:
            best, best_score = r, s
    return best


class MaskDataset:
    """Face-level dataset with forgery-mask supervision.

    ``index`` is a list of dicts from ``index_gt_faces`` or
    ``index_raw_celebdf_videos``.  With GT data each item yields (face, mask).
    With raw data we read the video, take a designated frame, detect the face,
    and produce a pseudo-mask from the ROIs when ``auto_generate_masks`` is set.
    """

    def __init__(self, index: List[dict], cfg: Config, device: str = "cpu",
                 cache_size: Optional[int] = None, read_face: bool = True):
        self.index = index
        self.cfg = cfg
        self.size = cfg.data.face_size
        self.device = device
        self.read_face = bool(read_face)
        # ``None`` preserves the historical cache for existing callers;
        # temporal worker datasets pass 0 because each worker owns its cache.
        self.cache_size = None if cache_size is None else max(0, int(cache_size))
        self._cache: Dict[int, MaskFaceSample] = {}
        self._layout_cache: Dict[int, FaceLayout] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _layout(self, item) -> FaceLayout:
        """Load landmark-grounded geometry, falling back to centered defaults."""
        layout = item.get("layout")
        if layout is not None:
            return layout
        return _landmark_layout(item.get("landmark"), item.get("face")) or FaceLayout()

    def __getitem__(self, idx: int) -> MaskFaceSample:
        if idx in self._cache:
            return self._cache[idx]
        item = self.index[idx]
        layout = self._layout_cache.get(idx)
        if layout is None:
            layout = self._layout(item)
            self._layout_cache[idx] = layout
        label = int(item["label"])

        if not self.read_face:
            # Feature-cache datasets only need the mask/layout supervision.
            mask = _read_mask(item.get("mask"), self.size)
            if label == 1 and not bool(mask.any()):
                mask = _pseudo_mask(layout, self.size)
            sample = MaskFaceSample(torch.zeros(3, self.size, self.size),
                                    mask.to(self.device), label, layout,
                                    item.get("video", ""), item.get("frame", ""))
        elif item.get("face", "").suffix.lower() in (".png", ".jpg", ".jpeg"):
            # GT face crop
            face = _read_png(item["face"], self.size)
            mask = _read_mask(item.get("mask"), self.size)
            if label == 1 and not bool(mask.any()):
                # GT mask file missing for a fake -> pseudo-mask so region loss is defined
                mask = _pseudo_mask(layout, self.size)
            sample = MaskFaceSample(face.to(self.device), mask.to(self.device),
                                    label, layout, item.get("video", ""), item.get("frame", ""))
        else:
            # raw video: sample a frame, detect face, build crop + pseudo-mask
            face, mask = _crop_from_video(item["face"], layout, self.size,
                                          auto_mask=self.cfg.data.auto_generate_masks or label == 1)
            if label == 0 and self.cfg.data.auto_generate_masks is False:
                mask = torch.zeros(self.size, self.size)
            sample = MaskFaceSample(face.to(self.device), mask.to(self.device),
                                    label, layout, item.get("video", ""), item.get("frame", ""))
        if self.cache_size is None:
            self._cache[idx] = sample
        elif self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = sample
        return sample


def _pseudo_mask(layout: FaceLayout, size: int) -> torch.Tensor:
    """Heuristic forgery mask: a soft band over jawline + mouth + periocular."""
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    xn, yn = xx.float() / size, yy.float() / size
    L = layout
    m = torch.zeros(size, size)
    m += ((yn - L.fy).abs() < L.fh * 1.15).float() * 0.7          # jawline
    m += ((xn - L.eye_left[0]).abs() < L.eye_w).float() * 0.5      # periocular
    m = m.clamp(0, 1)
    return m


def _crop_from_video(path: Path, layout: FaceLayout, size: int, auto_mask: bool) -> tuple:
    from .video_utils import detect_face
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"cannot decode {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W = frame.shape[:2]
    bbox = detect_face(frame)
    if bbox is not None:
        x, y, w, h = bbox
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = frame[y0:y1, x0:x1]
    else:
        crop = frame
    crop = cv2.resize(crop, (size, size))
    face = torch.from_numpy(crop).float().div_(255.0).permute(2, 0, 1)
    mask = _pseudo_mask(layout, size) if auto_mask else torch.zeros(size, size)
    return face, mask


def _official_ffpp_split(index: List[dict], ffpp_dir: Path) -> Dict[str, List[dict]]:
    """Assign FF++ records using the official source-pair JSON files.

    Every YouTube source and every ``source_target`` manipulated pair is kept
    in the same partition, including all configured manipulation methods. Any
    record that cannot be matched to the official pair lists is training-only;
    this is safer than silently placing it in validation or test.
    """
    pair_partition: Dict[tuple[str, str], str] = {}
    source_partition: Dict[str, str] = {}
    for partition in ("train", "val", "test"):
        path = Path(ffpp_dir) / f"{partition}.json"
        if not path.exists():
            raise FileNotFoundError(f"official FF++ split file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"official FF++ split must be a list: {path}")
        for pair in payload:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"invalid FF++ source pair in {path}: {pair!r}")
            first, second = (str(pair[0]), str(pair[1]))
            key = tuple(sorted((first, second)))
            previous = pair_partition.setdefault(key, partition)
            if previous != partition:
                raise ValueError(f"FF++ pair appears in multiple partitions: {key}")
            for source in key:
                previous = source_partition.setdefault(source, partition)
                if previous != partition:
                    raise ValueError(
                        f"FF++ source appears in multiple partitions: {source}")

    def match(item: dict) -> str:
        explicit = item.get("partition")
        video = str(item.get("video") or "")
        _, separator, name = video.partition("/")
        source = video.split("/", 1)[0] if separator else ""
        official = None
        if separator and source == "youtube" and name in source_partition:
            official = source_partition[name]
        else:
            tokens = name.split("_", 2)
            if len(tokens) >= 2 and tokens[0].isdigit() and tokens[1].isdigit():
                pair_key = tuple(sorted((tokens[0], tokens[1])))
                official = pair_partition.get(pair_key)
        if official is not None:
            # A stale train-only manifest tag cannot override an official
            # validation/test pair; grouping the source pair is the safety rule.
            return official
        if explicit is not None and explicit != "train":
            raise ValueError("official FF++ mode only permits train-only additions")
        return "train"

    result = {"train": [], "val": [], "test": []}
    for item in index:
        result[match(item)].append(item)
    return result


def split_index(index: List[dict], n_train: int, n_test: int, seed: int,
                n_val: int = 0, stratify_methods: bool = True,
                ffpp_dir: Optional[Path] = None,
                leakage_safe: bool = False):
    """Deterministic, video-disjoint per-class train/validation/test split.

    When ``stratify_methods`` is enabled, fake videos are split independently
    for every manipulation method and real videos independently for every
    origin (for example ``youtube`` and ``actors``). This keeps each partition
    evenly distributed instead of allowing a random video split to over-select
    one manipulation family. Complete videos are always assigned together, and
    one video per stratum is reserved for training whenever possible.

    ``n_train`` remains part of the public API and documents the intended
    training budget; the indexed pool determines the number of leftover
    training faces. The test partition is selected first, then validation, so
    neither can share a video with training.
    When ``leakage_safe`` is enabled, the official FF++ train/val/test pair
    lists are authoritative. Matching source videos and all manipulated
    variants stay together; unmatched records are assigned to training only.
    ``n_train``, ``n_val``, and ``n_test`` then document the requested legacy
    budgets but do not override the official pair membership.
    """
    if leakage_safe and not (index and all(
            item.get("source") in {"celebdf_raw", "deepfakebench"}
            for item in index)):
        if ffpp_dir is None:
            raise ValueError("ffpp_dir is required for leakage-safe FF++ splitting")
        return _official_ffpp_split(index, Path(ffpp_dir))

    def _unit(item: dict) -> str:
        return str(item.get("video") or
                   f"{item.get('frame', '')}@{item.get('face', '')}")

    def _stratum(item: dict) -> str:
        if not stratify_methods:
            return "__all__"
        video = str(item.get("video") or "")
        # FF++ uses '<method>/<video>'; raw/unit-test records without a method
        # label stay in one stratum so legacy callers retain balanced classes.
        return video.split("/", 1)[0] if "/" in video else "__all__"

    def _quotas(target: int, keys: List[str]) -> Dict[str, int]:
        if not keys:
            return {}
        base, remainder = divmod(max(0, int(target)), len(keys))
        return {key: base + int(pos < remainder)
                for pos, key in enumerate(keys)}

    def _take(units: List[List[dict]], start: int, target: int,
              result_key: str) -> int:
        """Take whole video units, reserving the final unit for training."""
        count, pos = 0, start
        limit = max(0, len(units) - 1)
        while pos < limit and count < max(0, target):
            result[result_key].extend(units[pos])
            count += len(units[pos])
            pos += 1
        return pos

    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for label in (1, 0):
        group = [item for item in index if int(item["label"]) == label]
        strata: Dict[str, Dict[str, List[dict]]] = {}
        for item in group:
            key = _stratum(item)
            strata.setdefault(key, {}).setdefault(_unit(item), []).append(item)

        keys = sorted(strata)
        test_quota = _quotas(n_test, keys)
        val_quota = _quotas(n_val, keys)
        for key in keys:
            free_units = []
            for unit in strata[key].values():
                assignments = {str(item.get("partition")) for item in unit
                               if item.get("partition") is not None}
                if not assignments:
                    free_units.append(unit)
                    continue
                if len(assignments) != 1 or assignments - set(result):
                    raise ValueError("a manifest video must have one valid partition")
                result[assignments.pop()].extend(unit)

            # Train-only manifest additions do not participate in the seeded
            # video draw, preserving the established validation/test videos.
            rng.shuffle(free_units)
            pos = _take(free_units, 0, test_quota[key], "test")
            pos = _take(free_units, pos, val_quota[key], "val")
            # Every unassigned complete video remains in training.
            for unit in free_units[pos:]:
                result["train"].extend(unit)

    return result


def mask_sample_collate(samples: List[MaskFaceSample]) -> dict:
    """DataLoader collate: stack face-level samples into one GPU-ready dict.

    Workers decode PNGs in parallel and the loader pins the tensors, so the
    training step transfers a single contiguous block to the device instead of
    decoding each image synchronously on the main thread.
    """
    faces = torch.stack([s.face for s in samples])
    masks = torch.stack([s.mask for s in samples])
    return {
        "faces": faces,
        "masks": masks,
        "labels": torch.tensor([s.label for s in samples], dtype=torch.float32),
        "layouts": tuple(s.layout for s in samples),
        "videos": tuple(s.video for s in samples),
        "frames": tuple(s.frame for s in samples),
    }


def _seed_worker(worker_id):
    """Bound worker RNG to the torch seed and cap thread oversubscription."""
    import random
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_data_loader(dataset, batch_size: int, shuffle: bool, workers: int,
                     prefetch_factor: int = 2, collate_fn=None,
                     drop_last: bool = False) -> DataLoader:
    """Parallel DataLoader tuned for small GPU memory.

    ``workers>0`` decodes PNGs in separate processes (overlapping CPU with the
    GPU), ``pin_memory`` enables non-blocking host-to-device copies, and
    ``persistent_workers`` avoids respawning processes each epoch.
    """
    from torch.utils.data import DataLoader
    if workers < 0 or prefetch_factor < 1:
        raise ValueError("workers must be >= 0 and prefetch_factor must be >= 1")
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


__all__ = ["MaskFaceSample", "MaskDataset", "index_gt_faces", "index_ffpp_masks",
           "index_raw_celebdf_videos", "_read_png", "_read_mask", "_region_primary",
           "split_index", "mask_sample_collate", "make_data_loader"]