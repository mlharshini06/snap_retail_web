"""
utils/image_utils.py
Small, dependency-light image helpers shared by the color/pose/seg
models and the UI layer. Kept allocation-light since these run every
frame on the camera thread.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import settings
from utils.logger import get_logger

logger = get_logger("image_utils")


def safe_crop(frame: np.ndarray, cx: int, cy: int, half: int) -> Optional[np.ndarray]:
    """Crop a (2*half)x(2*half) patch centered at (cx, cy), clamped to
    the frame bounds. Returns None if the resulting patch is empty."""
    h, w = frame.shape[:2]
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def bgr_to_hex(bgr: np.ndarray) -> str:
    b, g, r = [int(round(float(c))) for c in bgr[:3]]
    return "#{:02X}{:02X}{:02X}".format(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def hex_to_bgr(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def dominant_color_bgr(
    patch_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Return the dominant BGR color of `patch_bgr` via KMeans clustering
    on the HSV representation (more perceptually stable than raw RGB for
    skin tone and fabric color), converted back to BGR for display/storage.

    If `mask` is provided (same H x W as patch_bgr, nonzero = keep),
    only masked pixels are clustered — used for garment-only coloring
    so background pixels never pollute the result.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    if mask is not None:
        mask_bool = mask.astype(bool)
        if mask_bool.sum() < 10:
            return None
        pixels_hsv = hsv[mask_bool].reshape(-1, 3)
        pixels_bgr = patch_bgr[mask_bool].reshape(-1, 3)
    else:
        pixels_hsv = hsv.reshape(-1, 3)
        pixels_bgr = patch_bgr.reshape(-1, 3)

    if len(pixels_hsv) < 5:
        return None

    k = n_clusters or settings.color.kmeans_clusters
    k = max(1, min(k, len(pixels_hsv)))

    try:
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        km = KMeans(
            n_clusters=k,
            random_state=settings.color.kmeans_random_state,
            n_init=4,
            max_iter=settings.color.kmeans_max_iter,
        )
        with warnings.catch_warnings():
            # Small, near-uniform-color crops (skin/garment patches) often
            # have fewer distinct colors than n_clusters — harmless, and
            # KMeans still returns a sensible dominant cluster.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            labels = km.fit_predict(pixels_hsv.astype(np.float32))
    except Exception:
        logger.exception("KMeans clustering failed, falling back to mean color")
        return pixels_bgr.mean(axis=0)

    counts = np.bincount(labels)
    dominant_label = int(np.argmax(counts))
    dominant_mask = labels == dominant_label
    # Average the *original BGR* pixels belonging to the dominant HSV
    # cluster, so the reported color matches what a human sees.
    return pixels_bgr[dominant_mask].mean(axis=0)


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    scale = target_width / float(w)
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def release_buffer(*arrays: Optional[np.ndarray]) -> None:
    """Explicitly drop references to large ndarrays so they're eligible
    for GC immediately rather than at end-of-scope, useful right after
    an array has been uploaded to a cloud API and is no longer needed."""
    for _ in arrays:
        pass  # references are dropped by the caller reassigning/deleting
    import gc

    gc.collect()
