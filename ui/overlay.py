"""
ui/overlay.py
Draws everything onto the raw camera frame: pose skeleton, garment
bounding box, clothing label, skin-tone/garment color swatches, FPS,
device indicator, and a bottom status bar. Pure OpenCV drawing, no I/O,
safe to call every frame on the camera thread.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from config import settings
from models.pose_detector import PersonPose, COCO_SKELETON
from models.segmentation_detector import GarmentInstance
from utils.image_utils import hex_to_bgr

_GREEN = (60, 200, 60)
_CYAN = (255, 220, 40)
_WHITE = (255, 255, 255)
_YELLOW = (0, 210, 255)
_RED = (50, 50, 235)


def draw_pose(frame: np.ndarray, person: PersonPose, show: bool = True) -> None:
    if not show:
        return
    x1, y1, x2, y2 = (int(v) for v in person.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, settings.ui.line_thickness)

    pts = person.keypoints_xy
    confs = person.keypoints_conf
    for a, b in COCO_SKELETON:
        if a >= len(pts) or b >= len(pts):
            continue
        if confs[a] < 0.3 or confs[b] < 0.3:
            continue
        pa = tuple(int(v) for v in pts[a])
        pb = tuple(int(v) for v in pts[b])
        cv2.line(frame, pa, pb, _CYAN, 2)

    for i, (x, y) in enumerate(pts):
        if confs[i] < 0.3:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, _YELLOW, -1)


def draw_garment_mask(frame: np.ndarray, garment: GarmentInstance, show: bool = True, alpha: float = 0.35) -> None:
    if not show:
        return
    x1, y1, x2, y2 = (int(v) for v in garment.bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _CYAN, settings.ui.line_thickness)

    overlay = frame.copy()
    color_layer = np.zeros_like(frame)
    color_layer[garment.mask.astype(bool)] = (255, 180, 40)
    cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0, dst=overlay)
    mask_bool = garment.mask.astype(bool)
    frame[mask_bool] = overlay[mask_bool]

    label = f"{garment.class_name} {garment.confidence * 100:.0f}%"
    _put_label(frame, label, (x1, max(0, y1 - 8)))


def _put_label(frame: np.ndarray, text: str, org: tuple, color=_WHITE, scale: Optional[float] = None) -> None:
    scale = scale or settings.ui.font_scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 6), (x + tw + 4, y + 4), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_color_swatch(frame: np.ndarray, hex_color: Optional[str], label: str, top_left: tuple) -> None:
    x, y = top_left
    size = 26
    if hex_color:
        bgr = hex_to_bgr(hex_color)
        cv2.rectangle(frame, (x, y), (x + size, y + size), bgr, -1)
        cv2.rectangle(frame, (x, y), (x + size, y + size), _WHITE, 1)
        text = f"{label}: {hex_color}"
    else:
        cv2.rectangle(frame, (x, y), (x + size, y + size), (80, 80, 80), -1)
        text = f"{label}: --"
    cv2.putText(
        frame, text, (x + size + 8, y + size - 7),
        cv2.FONT_HERSHEY_SIMPLEX, settings.ui.font_scale, _WHITE, 1, cv2.LINE_AA,
    )


def draw_fps(frame: np.ndarray, fps: float) -> None:
    if not settings.ui.show_fps:
        return
    color = _GREEN if fps >= 20 else (_YELLOW if fps >= 12 else _RED)
    cv2.putText(
        frame, f"{fps:4.1f} FPS", (frame.shape[1] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def draw_device_indicator(frame: np.ndarray, device: str) -> None:
    label = "GPU" if "cuda" in device.lower() else "CPU"
    color = _GREEN if label == "GPU" else _YELLOW
    cv2.putText(
        frame, label, (frame.shape[1] - 150, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def draw_status_bar(frame: np.ndarray, statuses: dict) -> None:
    """statuses: dict like {'Cloud': 'idle'|'working'|'ok'|'error', 'DB': ..., 'Try-On': ...}"""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (25, 25, 25), -1)

    x = 10
    for name, state in statuses.items():
        color = {
            "idle": (150, 150, 150),
            "working": _YELLOW,
            "ok": _GREEN,
            "error": _RED,
        }.get(state, _WHITE)
        text = f"{name}: {state}"
        cv2.putText(frame, text, (x, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x += tw + 30


def draw_help_hint(frame: np.ndarray) -> None:
    ui = settings.ui
    text = (
        f"[{ui.recommend_key.upper()}] Recommend   "
        f"[{ui.snapshot_key.upper()}] Snapshot   "
        f"[{ui.toggle_pose_key.upper()}] Pose   "
        f"[{ui.toggle_seg_key.upper()}] Mask   "
        f"[{ui.quit_key.upper()}] Quit"
    )
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
