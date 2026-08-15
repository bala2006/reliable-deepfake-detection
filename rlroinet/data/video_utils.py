"""Shared video I/O, face detection and dataset-sampling helpers.

Used by both the FaceForensics++ (ffpp.py) and Celeb-DF (celebdf.py) loaders so
real-video handling is implemented exactly once.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .synthetic import FaceLayout

log = logging.getLogger("rlroinet.data")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# YuNet DNN face detector (bundled in the Docker image for OpenCV >= 5, where the
# legacy Haar CascadeClassifier was removed).
_YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
_detector_cache: Dict[str, object] = {}


def _get_detector():
    """Return a callable(frame_bgr) -> (x, y, w, h) | None across OpenCV versions."""
    if _detector_cache.get("detector") is not None:
        return _detector_cache["detector"]
    det = None
    if hasattr(cv2, "CascadeClassifier"):
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        def haar_detect(frame_bgr):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            faces = haar.detectMultiScale(
                gray, 1.1, 5, minSize=(max(20, w // 6), max(20, h // 6)))
            if len(faces) == 0:
                return None
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

        det = haar_detect
    elif hasattr(cv2, "FaceDetectorYN") and Path(_YUNET_MODEL).exists():
        yunet = cv2.FaceDetectorYN_create(_YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000)

        def yunet_detect(frame_bgr):
            h, w = frame_bgr.shape[:2]
            yunet.setInputSize((w, h))
            _, faces = yunet.detect(frame_bgr)
            if faces is None or len(faces) == 0:
                return None
            best = max(faces, key=lambda f: f[2] * f[3])
            return tuple(int(v) for v in best[:4])

        det = yunet_detect
    if det is None:
        log.warning("no face detector available (model missing) — using frame-centered layout")
    _detector_cache["detector"] = det
    return det


def detect_face(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Largest frontal face (x, y, w, h) in a BGR frame, or None."""
    det = _get_detector()
    if det is None:
        return None
    return det(frame_bgr)


def layout_from_face(bbox: Tuple[int, int, int, int], H: int, W: int) -> FaceLayout:
    """Map the synthetic ROI layout onto a detected face box (normalized)."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    fw = min(w / W, 0.48)
    fh = min(h / H, 0.45)
    return FaceLayout(
        fx=cx, fy=cy, fw=fw * 0.5, fh=fh * 0.5,
        eye_left=(cx - 0.55 * fw, cy - 0.54 * fh),
        eye_right=(cx + 0.55 * fw, cy - 0.54 * fh),
        eye_w=max(0.02, 0.45 * fw), eye_h=max(0.01, 0.10 * fh),
        mouth=(cx, cy + 0.54 * fh),
        mouth_w=0.60 * fw, mouth_h=0.19 * fh,
        hairline_y=cy - 0.92 * fh, hairline_h=0.12 * fh,
    )


def default_layout(H: int, W: int) -> FaceLayout:
    return FaceLayout()


class VideoFrameReader:
    """Reads videos into uniform tensors with an LRU cache."""

    def __init__(self, max_cache: int = 8):
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._max = max_cache

    def read(self, path: str | Path, frames_per_video: int, frame_size: int) -> torch.Tensor:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = frames_per_video
        idx = np.linspace(0, total - 1, frames_per_video).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames.append(frame[:, :, ::-1])          # BGR -> RGB
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames read from {path}")
        while len(frames) < frames_per_video:         # pad short clips
            frames.append(frames[-1])
        arr = np.stack(frames)                        # (T, H, W, 3) uint8
        tensor = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)
        if len(self._cache) >= self._max:
            self._cache.popitem(last=False)
        self._cache[key] = tensor
        return tensor


def split_per_class(real: List[dict], fake: List[dict],
                    n_train: int, n_test: int, seed: int, split: str,
                    official_test: Dict[int, set] | None = None) -> List[dict]:
    """Deterministic disjoint train/test sampling per class.

    official_test: {label: set of filenames} from the dataset's official test
    list. Official-test videos are reserved for the test split; the same seeded
    shuffle of the remaining pool is derived in both calls, so train/test stay
    disjoint even when the test split borrows extra videos from the pool.
    """
    rng = np.random.default_rng(seed)
    items: List[dict] = []
    for group, label in ((real, 0), (fake, 1)):
        group = list(group)
        official = [i for i in group
                    if official_test and i["name"] in official_test.get(label, set())]
        others = [i for i in group if i not in official]
        perm = rng.permutation(len(others))
        others = [others[i] for i in perm]

        if len(others) < n_train:
            log.warning("only %d non-official videos for class %d (need %d for train)",
                        len(others), label, n_train)
        if split == "train":
            chosen = others[:n_train]
        else:
            chosen = official[:n_test]
            if len(chosen) < n_test:
                fill = n_test - len(chosen)
                chosen += others[n_train:n_train + fill]
        items += chosen
    items.sort(key=lambda d: str(d["path"]))
    return items


class BaseVideoDataset:
    """Shared behaviour for real-video datasets (FF++, Celeb-DF)."""

    def __init__(self, root: str | Path, cfg, split: str = "train", device: str = "cpu"):
        self.root = Path(root)
        self.cfg = cfg
        self.device = device
        self.split = split
        self.frames_per_video = cfg.data.frames_per_video
        self.frame_size = cfg.data.face_size
        self.reader = VideoFrameReader()
        self._layout_cache: Dict[Path, FaceLayout] = {}
        self._items = self._build_index()
        self.layout = default_layout(self.frame_size, self.frame_size)
        self.manifest = {
            "layout": self.layout.to_dict(),
            "videos": [{"n_frames": self.frames_per_video} for _ in self._items],
        }

    def _class_items(self, label: int) -> List[dict]:
        raise NotImplementedError

    def _official_test(self) -> Optional[Dict[int, set]]:
        return None

    def _build_index(self) -> List[dict]:
        real = self._class_items(0)
        fake = self._class_items(1)
        c = self.cfg.data
        official = self._official_test()
        return split_per_class(real, fake, c.sample_train, c.sample_test,
                               c.seed, self.split, official_test=official)

    def _video_layout(self, path: Path) -> FaceLayout:
        if path in self._layout_cache:
            return self._layout_cache[path]
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            bbox = detect_face(frame)
            if bbox is not None:
                H, W = frame.shape[:2]
                self._layout_cache[path] = layout_from_face(bbox, H, W)
                return self._layout_cache[path]
        self._layout_cache[path] = default_layout(self.frame_size, self.frame_size)
        return self._layout_cache[path]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        item = self._items[idx]
        video = self.reader.read(item["path"], self.frames_per_video, self.frame_size)
        return {
            "video": video.to(self.device),
            "label": item["label"],
            "layout": self._video_layout(item["path"]),
            "method": item.get("method", ""),
        }

    def paths(self) -> List[Path]:
        return [i["path"] for i in self._items]
