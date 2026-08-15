"""FaceForensics++ cross-domain test loader.

Reads the official FF++ download structure for a held-out generalization check:

    FF++/
    ├── original_sequences/youtube/<compression>/videos/000.mp4 ...  (real)
    └── manipulated_sequences/<Method>/<compression>/videos/...      (fake)

Returns video samples so a model trained on Celeb-DF v2 can be scored on
unseen generators. Metrics are written only after an actual evaluation run.
"""

from __future__ import annotations

import logging
from typing import List

from .video_utils import BaseVideoDataset, VIDEO_EXTS, detect_face, layout_from_face, default_layout

log = logging.getLogger("rlroinet.ffpp")

METHODS = ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]


class FFPPDataset(BaseVideoDataset):
    def _class_items(self, label: int) -> List[dict]:
        c = self.cfg.data
        if label == 0:
            real_dir = self.root / "original_sequences" / "youtube" / c.compression / "videos"
            paths = sorted(real_dir.glob("*")) if real_dir.exists() else []
            return [{"path": p, "label": 0, "method": "original", "name": p.stem}
                    for p in paths if p.suffix.lower() in VIDEO_EXTS]
        fake = []
        for m in c.methods:
            d = self.root / "manipulated_sequences" / m / c.compression / "videos"
            if not d.exists():
                log.warning("missing FF++ method dir: %s", d)
                continue
            fake += [{"path": p, "label": 1, "method": m, "name": p.stem}
                     for p in sorted(d.glob("*")) if p.suffix.lower() in VIDEO_EXTS]
        return fake


__all__ = ["FFPPDataset", "METHODS", "detect_face", "layout_from_face", "default_layout"]