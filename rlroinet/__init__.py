"""ROI-Net v2: supervised deepfake classification and mask localization.

An ImageNet-pretrained EfficientNet backbone classifies aligned face frames with
Focal loss while a dense region head predicts forgery masks and facial ROI
localization for PERIOCULAR, JAWLINE, MOUTH, and HAIRLINE.
"""

__version__ = "2.0.0"

from .config import Config  # noqa: E402
