"""Region head: mask-supervised pixel localization (the thesis contribution).

Takes the backbone feature map (B, C, H, W) and decodes it into:
  * a dense per-pixel forgery probability map (B, 1, H, W) -- the "mask",
  * per-region ROI logits over the ROIs (PERIOCULAR, JAWLINE, MOUTH,
    HAIRLINE) using the existing FaceLayout ROI geometry.

The dense mask is supervised by DeepfakeBench forgery masks (Dice + BCE).  The
per-region branch emits four ROI-aligned logit maps, which are supervised
against the ground-truth mask restricted to each facial ROI.
"""

from __future__ import annotations

import torch
from torch import nn


class RegionHead(nn.Module):
    """Decoder feature map -> dense forgery mask + per-ROI forgery scores.

    ``in_channels`` is the backbone map depth (e.g. 1792 for EfficientNet-B4).
    Input map is (B, C, H, W); output mask is (B, 1, H, W) at the same spatial
    size (further upsampled to face resolution by the caller).
    """

    def __init__(self, in_channels: int, out_channels=(256, 128, 64, 32),
                 n_regions: int = 4):
        super().__init__()
        self.n_regions = n_regions
        ch = in_channels
        layers = []
        for oc in out_channels:
            layers += [
                nn.Conv2d(ch, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            ch = oc
        self.decoder = nn.Sequential(*layers)

        self.mask_head = nn.Conv2d(ch, 1, 1)            # dense forgery logit map
        self.region_head = nn.Conv2d(ch, n_regions, 1)  # per-ROI salient logits

    def forward(self, feat: torch.Tensor):
        """feat: (B, C, H, W) -> (mask, region) with mask (B,1,H,W) sigmoid probs."""
        h = self.decoder(feat)
        mask_logit = self.mask_head(h)                  # (B, 1, H, W)
        region_logit = self.region_head(h)              # (B, n_regions, H, W)
        return torch.sigmoid(mask_logit), region_logit

    def region_masks(self, size: int, layout) -> torch.Tensor:
        """Per-ROI pixel masks (n_regions, size, size) from a FaceLayout.

        Upsampled to ``size`` so they align with the decoder output spatial dims.
        ROIs: PERIOCULAR(1), JAWLINE(2), MOUTH(3), HAIRLINE(4) -> indexed 0..3.
        """
        from ..data.synthetic import FaceLayout, PERIOCULAR, JAWLINE, MOUTH, HAIRLINE

        s = size
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        xn, yn = xx.float() / s, yy.float() / s
        L = layout or FaceLayout()

        def rect_region(cx, cy, w, h):
            return ((xn - cx).abs() <= w / 2) & ((yn - cy).abs() <= h / 2)

        regions = {
            PERIOCULAR: rect_region(L.eye_left[0], L.eye_left[1], L.eye_w * 2, L.eye_h * 3)
                         | rect_region(L.eye_right[0], L.eye_right[1], L.eye_w * 2, L.eye_h * 3),
            JAWLINE: (yn - L.fy).abs() <= L.fh * 1.15,
            MOUTH: rect_region(L.mouth[0], L.mouth[1], L.mouth_w * 2, L.mouth_h * 2),
            HAIRLINE: (yn - L.hairline_y).abs() <= L.hairline_h,
        }
        out = torch.stack([regions[r].float() for r in (PERIOCULAR, JAWLINE, MOUTH, HAIRLINE)])
        return out.to(torch.float32)


__all__ = ["RegionHead"]