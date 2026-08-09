"""
models/pose_detector.py
Thin wrapper around Ultralytics YOLO11n-Pose. Owns the model instance,
device placement, and translation of raw Ultralytics results into a
small, stable dataclass the rest of the app depends on — so if the
Ultralytics API changes shape, only this file needs to change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import settings
from utils.device import resolve_device
from utils.logger import get_logger, LatencyTimer

logger = get_logger("pose_detector")

# COCO-pose 17-keypoint skeleton, used by the overlay to draw limbs.
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms + shoulders
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


@dataclass
class PersonPose:
    bbox_xyxy: tuple                 # (x1, y1, x2, y2) in pixel coords
    confidence: float
    keypoints_xy: np.ndarray         # shape (17, 2), pixel coords
    keypoints_conf: np.ndarray       # shape (17,)

    def keypoint(self, index: int) -> Optional[tuple]:
        if index < 0 or index >= len(self.keypoints_xy):
            return None
        if self.keypoints_conf[index] < 0.3:
            return None
        x, y = self.keypoints_xy[index]
        return (int(x), int(y))

    @property
    def nose(self) -> Optional[tuple]:
        return self.keypoint(settings.models.nose_keypoint_index)


@dataclass
class PoseResult:
    people: List[PersonPose] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def person_count(self) -> int:
        return len(self.people)

    @property
    def primary_person(self) -> Optional[PersonPose]:
        """Largest-bbox-area person, i.e. whoever is closest to the mirror."""
        if not self.people:
            return None
        return max(
            self.people,
            key=lambda p: (p.bbox_xyxy[2] - p.bbox_xyxy[0]) * (p.bbox_xyxy[3] - p.bbox_xyxy[1]),
        )


class PoseDetector:
    def __init__(self):
        from ultralytics import YOLO

        self.device = resolve_device()
        logger.info("Loading pose model '%s' on %s", settings.models.pose_model_path, self.device)
        self.model = YOLO(settings.models.pose_model_path)
        try:
            self.model.to(self.device)
        except Exception:
            logger.warning("Could not move pose model to %s, using default placement", self.device)
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one dummy inference so the first real frame isn't slow
        (CUDA kernel compilation / cudnn autotune happens here instead)."""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(
                dummy, device=self.device, verbose=False,
                conf=settings.models.pose_conf_threshold,
            )
            logger.info("Pose model warm-up complete")
        except Exception:
            logger.exception("Pose model warm-up failed (non-fatal)")

    def infer(self, frame_bgr: np.ndarray) -> PoseResult:
        start = time.perf_counter()
        with LatencyTimer(logger, "pose_inference"):
            results = self.model.predict(
                frame_bgr,
                device=self.device,
                conf=settings.models.pose_conf_threshold,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - start) * 1000.0

        people: List[PersonPose] = []
        if results:
            r = results[0]
            boxes = r.boxes
            kpts = r.keypoints
            if boxes is not None and kpts is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                kp_xy = kpts.xy.cpu().numpy()          # (N, 17, 2)
                kp_conf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(kp_xy.shape[:2])

                for i in range(len(boxes)):
                    people.append(PersonPose(
                        bbox_xyxy=tuple(float(v) for v in xyxy[i]),
                        confidence=float(confs[i]),
                        keypoints_xy=kp_xy[i],
                        keypoints_conf=kp_conf[i],
                    ))

        return PoseResult(people=people, inference_ms=inference_ms)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
