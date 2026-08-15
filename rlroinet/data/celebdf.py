"""Celeb-DF v2 dataset loader (PRD §6.1) with automatic zip ingestion.

You download the dataset zip yourself and drop it anywhere under `data/`
(any filename works, e.g. `data/archive.zip`). On first use the loader
auto-extracts it, finds the dataset root regardless of the zip's inner layout,
and splits real (Celeb-real + YouTube-real) vs fake (Celeb-synthesis).

Disk: ~10 GB zip + ~10 GB extracted.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import List, Optional

from .video_utils import BaseVideoDataset, VIDEO_EXTS

log = logging.getLogger("rlroinet.celebdf")

REAL_DIRS = ("celeb-real", "youtube-real")
FAKE_DIRS = ("celeb-synthesis",)


def find_zip(data_dir: str | Path, hint: str = "") -> Optional[Path]:
    """Locate a Celeb-DF zip. Prefers an explicit hint, else any *.zip under data_dir."""
    if hint:
        p = Path(hint)
        if p.exists() and p.suffix.lower() == ".zip":
            return p
        log.warning("explicit celebdf zip not found: %s", p)
    data_dir = Path(data_dir)
    hits = sorted([p for p in data_dir.rglob("*.zip")])
    if not hits:
        return None
    if len(hits) > 1:
        log.info("multiple zips found under %s, using the first: %s", data_dir, hits[0])
    return hits[0]


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip_path into dest with progress logging (zip is kept afterwards)."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        total = len(zf.infolist())
        log.info("extracting %s (%d entries) into %s", zip_path.name, total, dest)
        for i, member in enumerate(zf.infolist()):
            zf.extract(member, dest)
            if i % 100 == 0 or i == total - 1:
                log.info("extract %6d / %d", i + 1, total)
    log.info("extraction done")


def locate_celebdf_root(root: str | Path) -> Optional[Path]:
    """Find the folder that contains the Celeb-DF class directories.

    Walks every subfolder (handles any zip layout / wrapper dirs) and returns the
    first dir that has a fake dir plus at least one real dir (case-insensitive).
    """
    root = Path(root)
    if not root.exists():
        return None
    candidates = [root, *sorted(root.rglob("*"))]
    for d in candidates:
        if not d.is_dir():
            continue
        sub = {p.name.lower(): p for p in d.iterdir() if p.is_dir()}
        if (FAKE_DIRS[0] in sub
                and any(r in sub for r in REAL_DIRS)):
            return d
    return None


def ensure_celebdf(celebdf_dir: str | Path, zip_hint: str = "") -> Path:
    """Return the Celeb-DF root, extracting the dataset zip on first use."""
    celebdf_dir = Path(celebdf_dir)
    root = locate_celebdf_root(celebdf_dir)
    if root is not None:
        return root
    if not celebdf_dir.exists():
        log.warning("no Celeb-DF data yet at %s", celebdf_dir)
    zip_path = find_zip(celebdf_dir.parent, zip_hint)
    if zip_path is None:
        raise FileNotFoundError(
            "Celeb-DF not extracted and no zip found. Download the dataset zip and "
            f"place it under {celebdf_dir.parent} (any filename), then re-run. "
            f"Looked for a zip under {celebdf_dir.parent}.")
    extract_zip(zip_path, celebdf_dir)
    root = locate_celebdf_root(celebdf_dir)
    if root is None:
        raise FileNotFoundError(
            f"extracted {zip_path} but could not find Celeb-DF class folders under "
            f"{celebdf_dir}. The zip kept in place; check its contents.")
    log.info("Celeb-DF ready at %s", root)
    return root


def _gather(d: Path, label: int, method: str) -> List[dict]:
    if not d.is_dir():
        return []
    return [{"path": p, "label": label, "method": method, "name": p.stem}
            for p in sorted(d.glob("*")) if p.suffix.lower() in VIDEO_EXTS]


class CelebDFDataset(BaseVideoDataset):
    def _class_items(self, label: int) -> List[dict]:
        items = []
        for d in self.root.rglob("*"):
            if not d.is_dir():
                continue
            name = d.name.lower()
            if label == 0 and name in REAL_DIRS:
                items += _gather(d, 0, d.name)
            elif label == 1 and name == FAKE_DIRS[0]:
                items += _gather(d, 1, FAKE_DIRS[0])
        return items


__all__ = ["CelebDFDataset", "ensure_celebdf", "locate_celebdf_root", "find_zip", "extract_zip"]
