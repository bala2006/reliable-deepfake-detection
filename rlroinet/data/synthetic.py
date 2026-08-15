"""Face ROI geometry shared across the pipeline.

Defines the facial regions supervised ROI-Net v2 predicts:
PERIOCULAR, JAWLINE, MOUTH, and HAIRLINE, plus helpers to label a point and
build region masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Region ids used for ROI labelling and localization evaluation
PERIOCULAR = 1
JAWLINE = 2
MOUTH = 3
HAIRLINE = 4

REGION_NAMES: Dict[int, str] = {
    0: "OTHER",
    PERIOCULAR: "PERIOCULAR",
    JAWLINE: "JAWLINE",
    MOUTH: "MOUTH",
    HAIRLINE: "HAIRLINE",
}


@dataclass
class FaceLayout:
    """Normalized facial landmark layout (ratios of frame/face-crop size)."""
    fx: float = 0.50
    fy: float = 0.58
    fw: float = 0.20          # face half-width
    fh: float = 0.26          # face half-height
    eye_left: Tuple[float, float] = (0.39, 0.44)
    eye_right: Tuple[float, float] = (0.61, 0.44)
    eye_w: float = 0.09
    eye_h: float = 0.035
    mouth: Tuple[float, float] = (0.50, 0.72)
    mouth_w: float = 0.12
    mouth_h: float = 0.05
    hairline_y: float = 0.34
    hairline_h: float = 0.03

    def to_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy, "fw": self.fw, "fh": self.fh,
            "eye_left": list(self.eye_left), "eye_right": list(self.eye_right),
            "eye_w": self.eye_w, "eye_h": self.eye_h,
            "mouth": list(self.mouth), "mouth_w": self.mouth_w, "mouth_h": self.mouth_h,
            "hairline_y": self.hairline_y, "hairline_h": self.hairline_h,
        }


def label_roi(cx: float, cy: float, layout: FaceLayout) -> int:
    """Classify a normalized (cx, cy) coordinate into a facial ROI id."""
    dx_l = abs(cx - layout.eye_left[0])
    dx_r = abs(cx - layout.eye_right[0])
    if (dx_l < layout.eye_w or dx_r < layout.eye_w) and abs(cy - layout.eye_left[1]) < 0.09:
        return PERIOCULAR
    if abs(cx - layout.mouth[0]) < layout.mouth_w and abs(cy - layout.mouth[1]) < layout.mouth_h:
        return MOUTH
    if abs(cy - layout.hairline_y) < layout.hairline_h and abs(cx - layout.fx) < layout.fw * 1.3:
        return HAIRLINE
    if abs(cy - layout.fy) < layout.fh * 1.15 and cy > layout.eye_left[1]:
        return JAWLINE
    return 0


def roi_mask(layout: FaceLayout, size: int) -> Dict[int, np.ndarray]:
    """Binary region masks (n_regions, size, size) derived from a layout."""
    s = size
    yy, xx = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    xn, yn = xx.astype(float) / s, yy.astype(float) / s
    L = layout

    def rect(cx, cy, w, h):
        return (np.abs(xn - cx) <= w / 2) & (np.abs(yn - cy) <= h / 2)

    return {
        PERIOCULAR: rect((L.eye_left[0] + L.eye_right[0]) / 2, L.eye_left[1], L.eye_w * 2, L.eye_h * 3)
                    | ((np.abs(xn - L.eye_left[0]) <= L.eye_w) & (np.abs(yn - L.eye_left[1]) <= 0.09))
                    | ((np.abs(xn - L.eye_right[0]) <= L.eye_w) & (np.abs(yn - L.eye_right[1]) <= 0.09)),
        JAWLINE: (np.abs(yn - L.fy) <= L.fh * 1.15) & (yn > L.eye_left[1]),
        MOUTH: rect(L.mouth[0], L.mouth[1], L.mouth_w, L.mouth_h),
        HAIRLINE: (np.abs(yn - L.hairline_y) <= L.hairline_h) & (np.abs(xn - L.fx) <= L.fw * 1.3),
    }


__all__ = ["PERIOCULAR", "JAWLINE", "MOUTH", "HAIRLINE", "REGION_NAMES",
           "FaceLayout", "label_roi", "roi_mask"]