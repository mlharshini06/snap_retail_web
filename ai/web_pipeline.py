"""
ai/web_pipeline.py
Stateless, single-frame pipeline used by the web backend (web/main.py).

This intentionally does NOT reuse ai/recommendation_engine.py, because
that pipeline is wired for the desktop app's background-executor model
and always does snapshot-to-disk + Fal.ai + QR + PDF + PostgreSQL. The
web prototype must not permanently store camera images or depend on
those extra systems (see project brief: no DB/QR/PDF/auth for this
prototype).

What IS reused unchanged: PoseDetector, SegmentationDetector,
detect_colors, classify, and get_style_recommendation — the actual
"working AI logic" from the desktop app.

Every function here is synchronous/blocking by design; the web layer
runs it inside FastAPI's threadpool (plain `def` route handlers) so
the async event loop is never blocked, and multiple users are served
concurrently without any shared mutable state between requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ai.openrouter_client import get_style_recommendation
from models.clothing_classifier import classify
from models.color_detector import detect_colors
from models.pose_detector import PoseDetector
from models.segmentation_detector import SegmentationDetector
from utils.logger import get_logger

logger = get_logger("web_pipeline")


@dataclass
class DetectionResult:
    person_detected: bool
    garment_detected: bool
    clothing_type: str
    skin_tone_hex: Optional[str]
    garment_color_hex: Optional[str]


@dataclass
class RecommendationPayload:
    ok: bool
    clothing_type: str = "Unknown"
    skin_tone: Optional[str] = None
    garment_color: Optional[str] = None
    recommended_colors: list = field(default_factory=list)
    recommended_outfits: list = field(default_factory=list)
    styling_tips: list = field(default_factory=list)
    message: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "clothing_type": self.clothing_type,
            "skin_tone": self.skin_tone,
            "garment_color": self.garment_color,
            "recommendations": {
                "recommended_colors": self.recommended_colors,
                "recommended_outfits": self.recommended_outfits,
                "styling_tips": self.styling_tips,
            },
            "message": self.message,
        }


def run_detection(
    frame_bgr: np.ndarray,
    pose_detector: PoseDetector,
    seg_detector: SegmentationDetector,
) -> DetectionResult:
    """Single-frame pose + segmentation + color detection. Pure function
    over the given frame — holds no state between calls."""
    pose_result = pose_detector.infer(frame_bgr)
    seg_result = seg_detector.infer(frame_bgr)

    primary_person = pose_result.primary_person
    primary_garment = seg_result.primary_garment

    colors = detect_colors(frame_bgr, primary_person, primary_garment)
    clothing_type = classify(primary_garment)

    return DetectionResult(
        person_detected=pose_result.person_count > 0,
        garment_detected=primary_garment is not None,
        clothing_type=clothing_type,
        skin_tone_hex=colors.skin_tone_hex,
        garment_color_hex=colors.garment_color_hex,
    )


def run_recommendation(
    frame_bgr: np.ndarray,
    pose_detector: PoseDetector,
    seg_detector: SegmentationDetector,
) -> RecommendationPayload:
    """Detect + call OpenRouter for a single captured frame. Never
    touches disk or any shared/global state, so concurrent requests
    from different browser sessions never interfere with each other."""
    detection = run_detection(frame_bgr, pose_detector, seg_detector)

    if not detection.person_detected:
        return RecommendationPayload(
            ok=False,
            clothing_type=detection.clothing_type,
            message="Please make sure a person is clearly visible.",
        )

    if not detection.garment_detected:
        return RecommendationPayload(
            ok=False,
            clothing_type=detection.clothing_type,
            skin_tone=detection.skin_tone_hex,
            message="Please make sure a person is clearly visible.",
        )

    try:
        rec = get_style_recommendation(
            detection.skin_tone_hex or "#000000",
            detection.clothing_type,
            detection.garment_color_hex or "#000000",
        )
    except Exception:
        logger.exception("Unexpected error calling OpenRouter")
        rec = None

    if rec is None or not rec.ok:
        if rec is not None:
            logger.warning("OpenRouter recommendation failed: %s", rec.error)
        return RecommendationPayload(
            ok=False,
            clothing_type=detection.clothing_type,
            skin_tone=detection.skin_tone_hex,
            garment_color=detection.garment_color_hex,
            message="We couldn't generate recommendations right now. Please try again.",
        )

    return RecommendationPayload(
        ok=True,
        clothing_type=detection.clothing_type,
        skin_tone=detection.skin_tone_hex,
        garment_color=detection.garment_color_hex,
        recommended_colors=rec.recommended_colors,
        recommended_outfits=rec.recommended_outfits,
        styling_tips=rec.styling_tips,
    )
