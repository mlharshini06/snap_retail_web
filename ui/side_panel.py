"""
ui/side_panel.py
Renders the recommendation side panel as its own BGR image, which
app.py horizontally concatenates next to the live camera frame. Kept
separate from overlay.py because this panel reflects *background*
pipeline state (last completed/failed recommendation) rather than
per-frame detection.
"""

from __future__ import annotations

import textwrap
from typing import Optional

import cv2
import numpy as np

from ai.recommendation_engine import PipelineResult
from config import settings

_BG = (32, 32, 32)
_WHITE = (235, 235, 235)
_GREY = (150, 150, 150)
_GREEN = (60, 200, 60)
_YELLOW = (0, 210, 255)
_RED = (50, 50, 235)
_ACCENT = (255, 180, 40)


def _wrap(text: str, width_chars: int = 34):
    return textwrap.wrap(text, width=width_chars) or [""]


def render_side_panel(height: int, last_result: Optional[PipelineResult], pipeline_running: bool) -> np.ndarray:
    width = settings.ui.side_panel_width
    panel = np.full((height, width, 3), _BG, dtype=np.uint8)

    y = 30
    cv2.putText(panel, "Style Recommendations", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _WHITE, 2, cv2.LINE_AA)
    y += 14
    cv2.line(panel, (16, y), (width - 16, y), (70, 70, 70), 1)
    y += 26

    if pipeline_running:
        cv2.putText(panel, "Working on it...", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA)
        y += 26

    if last_result is None:
        cv2.putText(
            panel, f"Press '{settings.ui.recommend_key.upper()}' to", (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1, cv2.LINE_AA,
        )
        y += 20
        cv2.putText(panel, "get a recommendation.", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1, cv2.LINE_AA)
        return panel

    # --- detected attributes -----------------------------------------
    cv2.putText(panel, f"Detected: {last_result.clothing_type}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _WHITE, 1, cv2.LINE_AA)
    y += 22
    cv2.putText(panel, f"Skin tone: {last_result.skin_tone_hex or 'N/A'}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(panel, f"Garment color: {last_result.clothing_color_hex or 'N/A'}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1, cv2.LINE_AA)
    y += 26
    cv2.line(panel, (16, y), (width - 16, y), (70, 70, 70), 1)
    y += 22

    rec = last_result.recommendation
    if rec is None or not rec.ok:
        err = rec.error if rec else "No recommendation available"
        cv2.putText(panel, "Recommendation failed:", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _RED, 1, cv2.LINE_AA)
        y += 20
        for line in _wrap(err or "unknown error"):
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
            y += 18
    else:
        y = _draw_section(panel, "Recommended Colors", rec.recommended_colors, y, width)
        y = _draw_section(panel, "Recommended Outfits", rec.recommended_outfits, y, width)
        y = _draw_section(panel, "Styling Tips", rec.styling_tips, y, width)

    y += 6
    cv2.line(panel, (16, y), (width - 16, y), (70, 70, 70), 1)
    y += 22

    # --- try-on / QR / db status --------------------------------------
    if last_result.fal_image_url:
        cv2.putText(panel, "Virtual Try-On: ready", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1, cv2.LINE_AA)
        y += 20
        for line in _wrap(last_result.fal_image_url, 40):
            cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREY, 1, cv2.LINE_AA)
            y += 16
    elif last_result.fal_error:
        cv2.putText(panel, "Virtual Try-On: failed", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _RED, 1, cv2.LINE_AA)
        y += 18
    else:
        cv2.putText(panel, "Virtual Try-On: skipped", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREY, 1, cv2.LINE_AA)
        y += 18

    y += 8
    if last_result.pdf_path:
        cv2.putText(panel, "PDF report: saved", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "PDF report: failed", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _RED, 1, cv2.LINE_AA)
    y += 20

    if last_result.recommendation_id is not None:
        cv2.putText(panel, f"Saved to DB (id={last_result.recommendation_id})", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, f"DB: {last_result.db_error or 'not saved'}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _RED, 1, cv2.LINE_AA)
    y += 20

    cv2.putText(panel, f"Took {last_result.elapsed_ms:.0f}ms", (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)

    return panel


def _draw_section(panel: np.ndarray, title: str, items, y: int, width: int) -> int:
    cv2.putText(panel, title, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, _ACCENT, 1, cv2.LINE_AA)
    y += 20
    if not items:
        cv2.putText(panel, "  (none)", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _GREY, 1, cv2.LINE_AA)
        y += 18
    for item in items:
        for i, line in enumerate(_wrap(str(item))):
            prefix = "- " if i == 0 else "  "
            cv2.putText(panel, prefix + line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
            y += 17
    return y + 8
