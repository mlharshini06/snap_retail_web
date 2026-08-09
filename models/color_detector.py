"""
models/color_detector.py
Combines pose keypoints + segmentation masks to compute:
  - skin tone HEX: 20x20 crop around the nose keypoint -> HSV -> KMeans
  - garment color HEX: pixels under the garment mask -> HSV -> KMeans
Pure numpy/OpenCV, no model loading here — this is local, synchronous,
and cheap enough to run every frame on the camera thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import settings
from models.pose_detector import PersonPose
from models.segmentation_detector import GarmentInstance
from utils.image_utils import safe_crop, dominant_color_bgr, bgr_to_hex
from utils.logger import get_logger

logger = get_logger("color_detector")


@dataclass
class ColorReadings:
    skin_tone_hex: Optional[str] = None
    garment_color_hex: Optional[str] = None


def detect_skin_tone(frame_bgr: np.ndarray, person: Optional[PersonPose]) -> Optional[str]:
    if person is None:
        return None
    nose = person.nose
    if nose is None:
        return None

    half = settings.models.skin_patch_size // 2
    patch = safe_crop(frame_bgr, nose[0], nose[1], half)
    if patch is None:
        return None

    dominant = dominant_color_bgr(patch)
    if dominant is None:
        return None
    return bgr_to_hex(dominant)


def detect_garment_color(frame_bgr: np.ndarray, garment: Optional[GarmentInstance]) -> Optional[str]:
    if garment is None:
        return None

    x1, y1, x2, y2 = (int(v) for v in garment.bbox_xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    mask_crop = garment.mask[y1:y2, x1:x2]
    if crop.size == 0 or mask_crop.sum() == 0:
        return None

    dominant = dominant_color_bgr(crop, mask=mask_crop)
    if dominant is None:
        return None
    return bgr_to_hex(dominant)


def detect_colors(
    frame_bgr: np.ndarray,
    person: Optional[PersonPose],
    garment: Optional[GarmentInstance],
) -> ColorReadings:
    return ColorReadings(
        skin_tone_hex=detect_skin_tone(frame_bgr, person),
        garment_color_hex=detect_garment_color(frame_bgr, garment),
    )
