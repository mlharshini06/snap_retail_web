"""
app.py
Snap Retail Mirror — main entrypoint.

The camera loop below is the ONLY thing allowed to touch: capture,
pose inference, seg inference, local color extraction, overlay
rendering, and cv2.imshow. Every other feature (DB, PDF, QR,
OpenRouter, Fal.ai, and even snapshot file writes) is dispatched to
`utils.async_executor.executor` and its results are picked up here
through a small thread-safe state object — never awaited inline.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import settings
from database.postgres_pool import db_pool
from database import repository
from models.clothing_classifier import classify
from models.color_detector import detect_colors
from models.pose_detector import PoseDetector, PoseResult
from models.segmentation_detector import SegmentationDetector, SegResult
from ai.recommendation_engine import run_recommendation_pipeline, PipelineResult
from ui import overlay
from ui.side_panel import render_side_panel
from utils.async_executor import executor
from utils.logger import get_logger

logger = get_logger("app")


@dataclass
class SharedState:
    """Everything the camera thread reads that background threads write.
    Guarded by one lock since updates are small and infrequent relative
    to the ~30fps camera loop."""
    lock: threading.Lock
    last_result: Optional[PipelineResult] = None
    pipeline_running: bool = False
    cloud_status: str = "idle"     # idle | working | ok | error
    db_status: str = "idle"
    tryon_status: str = "idle"


class CameraSource:
    """Owns the cv2.VideoCapture handle and knows how to reconnect if
    the camera drops mid-session, so a loose USB cable doesn't crash
    the app."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self) -> bool:
        cam = settings.camera
        self.cap = cv2.VideoCapture(cam.device_index)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
            self.cap.set(cv2.CAP_PROP_FPS, cam.target_fps)
        opened = self.cap is not None and self.cap.isOpened()
        if opened:
            logger.info("Camera %d opened at %dx%d", cam.device_index, cam.width, cam.height)
        else:
            logger.error("Failed to open camera %d", cam.device_index)
        return opened

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def reconnect(self) -> bool:
        logger.warning("Attempting camera reconnect...")
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        return self._open()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class FPSCounter:
    def __init__(self, smoothing: float = 0.9):
        self._smoothing = smoothing
        self._fps = 0.0
        self._last = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0:
            instant = 1.0 / dt
            self._fps = self._smoothing * self._fps + (1 - self._smoothing) * instant
        return self._fps


class SnapRetailMirrorApp:
    def __init__(self):
        self.session_token = uuid.uuid4().hex
        self.state = SharedState(lock=threading.Lock())
        self.show_pose = True
        self.show_mask = True
        self._running = True

        logger.info("Initializing models...")
        self.pose_detector = PoseDetector()
        self.seg_detector = SegmentationDetector()

        self.camera = CameraSource()
        self.fps_counter = FPSCounter()

        # Fire-and-forget: verify/create schema without blocking startup
        # or the camera loop. DB features degrade gracefully if this fails.
        executor.submit(
            self._init_database, task_name="db_schema_init",
            on_success=lambda _: self._set_db_status("idle"),
            on_error=lambda _: self._set_db_status("error"),
        )

    # ------------------------------------------------------------------
    # Background-safe helpers (never called from the camera loop body)
    # ------------------------------------------------------------------
    def _init_database(self):
        if db_pool.initialize():
            repository.initialize_schema()

    def _set_db_status(self, status: str) -> None:
        with self.state.lock:
            self.state.db_status = status

    def _on_pipeline_success(self, result: PipelineResult) -> None:
        with self.state.lock:
            self.state.last_result = result
            self.state.pipeline_running = False
            self.state.cloud_status = "ok" if (result.recommendation and result.recommendation.ok) else "error"
            self.state.tryon_status = "ok" if result.fal_image_url else ("error" if result.fal_error else "idle")
            self.state.db_status = "ok" if result.recommendation_id is not None else "error"
        logger.info("Recommendation pipeline completed (stage=%s)", result.stage)

    def _on_pipeline_error(self, exc: BaseException) -> None:
        with self.state.lock:
            self.state.pipeline_running = False
            self.state.cloud_status = "error"
        logger.error("Recommendation pipeline crashed: %s", exc)

    def _trigger_recommendation(self, frame: np.ndarray, skin_hex, clothing_type, garment_hex) -> None:
        with self.state.lock:
            if self.state.pipeline_running:
                logger.info("Recommendation already in progress, ignoring key press")
                return
            self.state.pipeline_running = True
            self.state.cloud_status = "working"
            self.state.tryon_status = "working"

        # Copy the frame — the camera loop will overwrite/reuse buffers
        # on the next iteration, and this task runs on another thread.
        frame_copy = frame.copy()
        executor.submit(
            run_recommendation_pipeline,
            frame_copy, skin_hex, clothing_type, garment_hex, self.session_token,
            task_name="recommendation_pipeline",
            on_success=self._on_pipeline_success,
            on_error=self._on_pipeline_error,
        )

    def _save_snapshot_async(self, frame: np.ndarray) -> None:
        frame_copy = frame.copy()

        def _write():
            import uuid as _uuid
            from config import SNAPSHOTS_DIR
            path = SNAPSHOTS_DIR / f"manual_{_uuid.uuid4().hex[:12]}.jpg"
            cv2.imwrite(str(path), frame_copy)
            return path

        executor.submit(
            _write, task_name="manual_snapshot",
            on_success=lambda p: logger.info("Manual snapshot saved to %s", p),
        )

    # ------------------------------------------------------------------
    # The sacred camera loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        cam_cfg = settings.camera
        reconnect_attempts = 0

        try:
            while self._running:
                frame = self.camera.read()
                if frame is None:
                    reconnect_attempts += 1
                    logger.warning("Camera read failed (attempt %d)", reconnect_attempts)
                    if cam_cfg.max_reconnect_attempts and reconnect_attempts > cam_cfg.max_reconnect_attempts:
                        logger.error("Exceeded max camera reconnect attempts, exiting")
                        break
                    time.sleep(cam_cfg.reconnect_delay_sec)
                    self.camera.reconnect()
                    continue
                reconnect_attempts = 0

                try:
                    self._process_frame(frame)
                except Exception:
                    # A bad frame or a transient inference hiccup must
                    # never kill the app — log it and keep the loop alive.
                    logger.exception("Frame processing error, continuing")

                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key, frame):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self._shutdown()

    def _process_frame(self, frame: np.ndarray) -> None:
        pose_result: PoseResult = self.pose_detector.infer(frame)
        seg_result: SegResult = self.seg_detector.infer(frame)

        primary_person = pose_result.primary_person
        primary_garment = seg_result.primary_garment
        colors = detect_colors(frame, primary_person, primary_garment)
        clothing_type = classify(primary_garment)

        display = frame.copy()

        for person in pose_result.people:
            overlay.draw_pose(display, person, show=self.show_pose)
        for garment in seg_result.garments:
            overlay.draw_garment_mask(display, garment, show=self.show_mask)

        overlay.draw_help_hint(display)
        overlay.draw_color_swatch(display, colors.skin_tone_hex, "Skin", (10, 40))
        overlay.draw_color_swatch(display, colors.garment_color_hex, "Garment", (10, 75))
        overlay.draw_fps(display, self.fps_counter.tick())
        overlay.draw_device_indicator(display, self.pose_detector.device)

        with self.state.lock:
            statuses = {
                "Cloud": self.state.cloud_status,
                "Try-On": self.state.tryon_status,
                "DB": self.state.db_status,
                "People": str(pose_result.person_count),
            }
            last_result = self.state.last_result
            pipeline_running = self.state.pipeline_running

        overlay.draw_status_bar(display, statuses)

        panel = render_side_panel(display.shape[0], last_result, pipeline_running)
        composite = np.hstack([display, panel])

        cv2.imshow(settings.ui.window_name, composite)

        # Stash the latest raw frame + detections for key handlers
        # (recommendation/snapshot triggers), read only on this thread.
        self._latest_frame = frame
        self._latest_colors = colors
        self._latest_clothing_type = clothing_type

    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        ui = settings.ui
        if key == 255:  # no key pressed
            return True

        ch = chr(key).lower() if 0 <= key < 256 else ""

        if ch == ui.quit_key:
            logger.info("Quit key pressed")
            return False
        if ch == ui.recommend_key:
            colors = getattr(self, "_latest_colors", None)
            clothing_type = getattr(self, "_latest_clothing_type", "Unknown")
            skin_hex = colors.skin_tone_hex if colors else None
            garment_hex = colors.garment_color_hex if colors else None
            self._trigger_recommendation(frame, skin_hex, clothing_type, garment_hex)
        elif ch == ui.snapshot_key:
            self._save_snapshot_async(frame)
        elif ch == ui.toggle_pose_key:
            self.show_pose = not self.show_pose
        elif ch == ui.toggle_seg_key:
            self.show_mask = not self.show_mask

        return True

    def _shutdown(self) -> None:
        logger.info("Shutting down Snap Retail Mirror...")
        self._running = False
        try:
            self.camera.release()
        except Exception:
            logger.exception("Error releasing camera")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.pose_detector.close()
            self.seg_detector.close()
        except Exception:
            pass
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            db_pool.close_all()
        except Exception:
            pass
        logger.info("Shutdown complete")


def main() -> int:
    try:
        app = SnapRetailMirrorApp()
    except Exception:
        logger.exception("Fatal error during startup")
        return 1

    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
