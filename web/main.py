"""
web/main.py
FastAPI backend for the browser-accessible Snap Retail Mirror prototype.

Design constraints this file follows (see project brief):
- Reuse the existing YOLO-Pose / YOLO-Seg / color / OpenRouter pipeline
  unchanged (imported from models/ and ai/).
- Load both models exactly once at startup and reuse them for every
  request — never reload per-request.
- No global per-user state: every request is fully self-contained
  (the browser sends a frame, gets a JSON response back). Two people
  hitting the same public URL from different phones never see each
  other's results.
- No camera images are ever written to disk here.
- Friendly, non-technical error messages only; full tracebacks go to
  the server log via `logger.exception`.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ai.web_pipeline import run_detection, run_recommendation
from models.pose_detector import PoseDetector
from models.segmentation_detector import SegmentationDetector
from utils.logger import get_logger

logger = get_logger("web_main")

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"

# Longest side (px) any incoming frame is resized to before inference.
# Keeps CPU inference time + request payload bounded regardless of the
# camera/device resolution a shopper's phone happens to capture at.
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "960"))

# Comma-separated list, or "*" for any origin (fine for this prototype
# since there is no auth / per-user secret data involved).
_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = ["*"] if _cors_env.strip() == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

FRIENDLY_SERVER_ERROR = "Unable to connect to the AI service. Please try again."
FRIENDLY_INVALID_IMAGE = "Please make sure a person is clearly visible."


class ModelState:
    """Holds the two YOLO models, loaded once at startup, plus a lock.

    Ultralytics' YOLO.predict() is not guaranteed thread-safe when the
    same model instance is called from multiple threads at once. Since
    FastAPI's sync `def` routes each run on their own threadpool
    thread, we serialize inference through this lock so concurrent
    users are handled correctly rather than fast — appropriate for a
    small CPU deployment. (For heavier traffic, run several worker
    processes instead of removing this lock.)
    """

    def __init__(self) -> None:
        self.pose_detector: Optional[PoseDetector] = None
        self.seg_detector: Optional[SegmentationDetector] = None
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.pose_detector is not None and self.seg_detector is not None


model_state = ModelState()
def _ensure_models_loaded() -> None:
    """Load YOLO models only when they are first needed."""
    if model_state.ready:
        return

    with model_state.lock:
        if model_state.ready:
            return

        logger.info("Loading YOLO models on first request...")
        model_state.pose_detector = PoseDetector()
        model_state.seg_detector = SegmentationDetector()
        logger.info("Models loaded and ready.")
app = FastAPI(title="Snap Retail Mirror", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)




# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _decode_upload(raw: bytes) -> Optional[np.ndarray]:
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _resize_if_needed(frame: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


async def _read_frame(file: UploadFile) -> np.ndarray:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=FRIENDLY_INVALID_IMAGE)

    raw = await file.read()
    frame = _decode_upload(raw)
    if frame is None:
        raise HTTPException(status_code=400, detail=FRIENDLY_INVALID_IMAGE)

    return _resize_if_needed(frame)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": model_state.ready}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    """Fast detection-only pass: person/garment presence, clothing
    type, skin tone, garment color. No AI call — used to give the
    shopper immediate visual feedback right after they capture a
    frame, before they tap "Get Recommendations"."""
    try:
        _ensure_models_loaded()
    except Exception:
        logger.exception("Model loading failed in /analyze")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": FRIENDLY_SERVER_ERROR},
        )

    try:
        frame = await _read_frame(file)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"ok": False, "message": exc.detail})

    try:
        with model_state.lock:
            detection = run_detection(frame, model_state.pose_detector, model_state.seg_detector)
    except Exception:
        logger.exception("Detection failed in /analyze")
        return JSONResponse(status_code=500, content={"ok": False, "message": FRIENDLY_SERVER_ERROR})

    if not detection.person_detected:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "person_detected": False,
                "garment_detected": False,
                "clothing_type": detection.clothing_type,
                "skin_tone": detection.skin_tone_hex,
                "garment_color": detection.garment_color_hex,
                "message": FRIENDLY_INVALID_IMAGE,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "person_detected": detection.person_detected,
            "garment_detected": detection.garment_detected,
            "clothing_type": detection.clothing_type,
            "skin_tone": detection.skin_tone_hex,
            "garment_color": detection.garment_color_hex,
            "message": None if detection.garment_detected else "Person detected. Move closer so your outfit is fully visible.",
        },
    )


@app.post("/recommend")
async def recommend(file: UploadFile = File(...)) -> JSONResponse:
    """Full pipeline for a single captured frame: detect -> color ->
    OpenRouter styling recommendation. Stateless per request."""
    try:
        _ensure_models_loaded()
    except Exception:
        logger.exception("Model loading failed in /recommend")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": FRIENDLY_SERVER_ERROR},
        )

    try:
        frame = await _read_frame(file)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"ok": False, "message": exc.detail})

    try:
        with model_state.lock:
            payload = run_recommendation(frame, model_state.pose_detector, model_state.seg_detector)
    except Exception:
        logger.exception("Pipeline failed in /recommend")
        return JSONResponse(status_code=500, content={"ok": False, "message": FRIENDLY_SERVER_ERROR})

    status_code = 200 if payload.ok else (422 if "visible" in (payload.message or "") else 502)
    return JSONResponse(status_code=status_code, content=payload.as_dict())


# Serve the frontend last so it never shadows the API routes above.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
