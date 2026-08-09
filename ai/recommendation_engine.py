"""
ai/recommendation_engine.py
Orchestrates the full "press R" pipeline:

  snapshot -> OpenRouter recommendation -> Fal.ai virtual try-on
            -> QR code -> PDF report -> PostgreSQL row

This whole function is meant to be handed to utils.async_executor.executor
as a single task, so the camera thread only ever fires it and moves on;
every step here runs on a worker thread and is allowed to block.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from ai.fal_client import run_virtual_tryon, FalError
from ai.openrouter_client import get_style_recommendation, RecommendationResult
from config import settings, SNAPSHOTS_DIR
from database.postgres_pool import db_pool
from database import repository
from utils.device import free_gpu_memory
from utils.image_utils import encode_jpeg
from utils.logger import get_logger
from utils.pdf_generator import ReportData, generate_pdf_report
from utils.qr_generator import generate_qr

logger = get_logger("recommendation_engine")


@dataclass
class PipelineResult:
    started_at: str
    skin_tone_hex: Optional[str]
    clothing_type: str
    clothing_color_hex: Optional[str]
    recommendation: Optional[RecommendationResult] = None
    fal_image_url: Optional[str] = None
    fal_error: Optional[str] = None
    db_error: Optional[str] = None
    snapshot_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    qr_path: Optional[Path] = None
    recommendation_id: Optional[int] = None
    elapsed_ms: float = 0.0
    stage: str = "starting"


def _save_snapshot(frame_bgr) -> Path:
    filename = f"snapshot_{uuid.uuid4().hex[:12]}.jpg"
    path = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(path), frame_bgr)
    return path


def run_recommendation_pipeline(
    frame_bgr,
    skin_tone_hex: Optional[str],
    clothing_type: str,
    clothing_color_hex: Optional[str],
    session_token: str,
    garment_image_url: Optional[str] = None,
) -> PipelineResult:
    """Blocking, synchronous, meant to run inside the background executor.

    `garment_image_url` should point at the product image the shopper
    wants to try on (e.g. selected from the `products` table / a
    catalog UI). If omitted, `settings` has no safe default product to
    try on, so the virtual try-on stage is skipped rather than guessing
    a garment — everything else (recommendation, PDF, QR, DB) still runs.
    """
    t0 = time.perf_counter()
    result = PipelineResult(
        started_at=datetime.now().isoformat(timespec="seconds"),
        skin_tone_hex=skin_tone_hex,
        clothing_type=clothing_type,
        clothing_color_hex=clothing_color_hex,
    )

    # --- 1. Snapshot -------------------------------------------------
    result.stage = "snapshot"
    try:
        result.snapshot_path = _save_snapshot(frame_bgr)
        logger.info("Snapshot saved to %s", result.snapshot_path)
    except Exception:
        logger.exception("Failed to save snapshot")

    # Snapshot bytes are needed by Fal; encode once and drop the frame
    # reference immediately after so it isn't held for the whole pipeline.
    try:
        snapshot_bytes = encode_jpeg(frame_bgr)
    except Exception:
        logger.exception("Failed to JPEG-encode snapshot for upload")
        snapshot_bytes = None
    frame_bgr = None  # release large ndarray reference early

    # --- 2. OpenRouter recommendation --------------------------------
    result.stage = "openrouter"
    try:
        result.recommendation = get_style_recommendation(
            skin_tone_hex or "#000000", clothing_type, clothing_color_hex or "#000000",
        )
        if not result.recommendation.ok:
            logger.warning("OpenRouter recommendation failed: %s", result.recommendation.error)
    except Exception:
        logger.exception("Unexpected error calling OpenRouter")
        result.recommendation = RecommendationResult(error="Unexpected OpenRouter failure")

    # --- 3. Fal.ai virtual try-on -------------------------------------
    result.stage = "fal"
    if snapshot_bytes and garment_image_url:
        try:
            tryon = run_virtual_tryon(snapshot_bytes, garment_image_url)
            result.fal_image_url = tryon.image_url
        except FalError as exc:
            result.fal_error = str(exc)
            logger.warning("Fal virtual try-on failed: %s", exc)
        except Exception:
            result.fal_error = "Unexpected Fal.ai failure"
            logger.exception("Unexpected error calling Fal.ai")
    else:
        logger.info("Skipping Fal virtual try-on: no garment_image_url provided")

    # Snapshot bytes have been uploaded (or we gave up); drop the
    # reference and force a GC pass per the "release buffers after API
    # uploads" requirement, and clear any cached CUDA memory too.
    snapshot_bytes = None
    free_gpu_memory()

    # --- 4. QR code -----------------------------------------------------
    result.stage = "qr"
    qr_payload = result.fal_image_url or (str(result.snapshot_path) if result.snapshot_path else None)
    if qr_payload:
        try:
            result.qr_path = generate_qr(qr_payload)
        except Exception:
            logger.exception("QR generation failed")

    # --- 5. PDF report ----------------------------------------------
    result.stage = "pdf"
    try:
        rec = result.recommendation
        report = ReportData(
            timestamp=result.started_at,
            skin_tone_hex=skin_tone_hex or "N/A",
            clothing_type=clothing_type,
            clothing_color_hex=clothing_color_hex or "N/A",
            recommended_colors=rec.recommended_colors if rec else [],
            recommended_outfits=rec.recommended_outfits if rec else [],
            styling_tips=rec.styling_tips if rec else [],
            fal_image_url=result.fal_image_url,
            qr_code_path=result.qr_path,
            snapshot_path=result.snapshot_path,
        )
        result.pdf_path = generate_pdf_report(report)
    except Exception:
        logger.exception("PDF generation failed")

    # --- 6. PostgreSQL ------------------------------------------------
    result.stage = "database"
    try:
        if db_pool.initialize():
            user_id = repository.get_or_create_user(session_token)
            result.recommendation_id = repository.insert_recommendation(
                user_id=user_id,
                skin_tone_hex=skin_tone_hex,
                clothing_type=clothing_type,
                clothing_color_hex=clothing_color_hex,
                openrouter_response=result.recommendation.as_dict() if result.recommendation else {},
                fal_image_url=result.fal_image_url,
                snapshot_path=str(result.snapshot_path) if result.snapshot_path else None,
                pdf_path=str(result.pdf_path) if result.pdf_path else None,
                qr_path=str(result.qr_path) if result.qr_path else None,
            )
        else:
            result.db_error = "PostgreSQL unavailable"
    except Exception as exc:
        logger.exception("Failed to persist recommendation to PostgreSQL")
        result.db_error = str(exc)

    result.stage = "done"
    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info("Recommendation pipeline finished in %.0fms", result.elapsed_ms)
    return result
