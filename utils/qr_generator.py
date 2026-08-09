"""
utils/qr_generator.py
Generates a QR code PNG pointing at the Fal.ai try-on image (or any
URL/string payload) and saves it under snapshots/qr/. Runs off the
camera thread — called only from background tasks.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import qrcode

from config import QR_DIR
from utils.logger import get_logger

logger = get_logger("qr_generator")


def generate_qr(data: str, filename: Optional[str] = None) -> Path:
    """Generate a QR code encoding `data` and save it as a PNG.
    Returns the path to the saved file. Raises on I/O failure — callers
    running this in the executor should catch via on_error.
    """
    if not data:
        raise ValueError("Cannot generate a QR code for empty data")

    filename = filename or f"qr_{uuid.uuid4().hex[:12]}.png"

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    out_path = QR_DIR / filename
    img.save(out_path)
    logger.info("QR code saved to %s (payload length=%d)", out_path, len(data))
    return out_path
