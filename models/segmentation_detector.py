"""
models/segmentation_detector.py
Thin wrapper around Ultralytics YOLO11n-Seg. Produces per-instance
masks + class labels, filtered down to garment classes so the rest of
the pipeline never has to think about non-clothing detections.

Note on the base model: stock yolo11n-seg.pt is trained on COCO, which
does not have fine-grained garment classes. In production you should
fine-tune / swap in a garment-segmentation checkpoint (e.g. trained on
DeepFashion2) and point `settings.models.seg_model_path` at it — the
wrapper itself is agnostic to the class list, it just filters by
`settings.models.garment_classes`. This is called out explicitly in
the README.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import settings
from utils.device import resolve_device
from utils.logger import get_logger, LatencyTimer

logger = get_logger("segmentation_detector")


@dataclass
class GarmentInstance:
    class_name: str
    confidence: float
    bbox_xyxy: tuple
    mask: np.ndarray  # binary mask, same H x W as the source frame


@dataclass
class SegResult:
    garments: List[GarmentInstance] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def primary_garment(self) -> Optional[GarmentInstance]:
        if not self.garments:
            return None
        return max(self.garments, key=lambda g: g.mask.sum())


class SegmentationDetector:
    def __init__(self):
        from ultralytics import YOLO

        self.device = resolve_device()
        logger.info("Loading segmentation model '%s' on %s", settings.models.seg_model_path, self.device)
        self.model = YOLO(settings.models.seg_model_path)
        try:
            self.model.to(self.device)
        except Exception:
            logger.warning("Could not move seg model to %s, using default placement", self.device)

        self.class_names = self.model.names  # dict[int, str]
        self._garment_ids = {
            idx for idx, name in self.class_names.items()
            if name.lower() in settings.models.garment_classes
        }
        if not self._garment_ids:
            logger.warning(
                "No model classes matched configured garment_classes; "
                "seg model will report zero garments until a garment-trained "
                "checkpoint is used. See README for fine-tuning notes."
            )

        self._warm_up()

    def _warm_up(self) -> None:
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(
                dummy, device=self.device, verbose=False,
                conf=settings.models.seg_conf_threshold,
            )
            logger.info("Segmentation model warm-up complete")
        except Exception:
            logger.exception("Segmentation model warm-up failed (non-fatal)")

    def infer(self, frame_bgr: np.ndarray) -> SegResult:
        h, w = frame_bgr.shape[:2]
        start = time.perf_counter()
        with LatencyTimer(logger, "seg_inference"):
            results = self.model.predict(
                frame_bgr,
                device=self.device,
                conf=settings.models.seg_conf_threshold,
                iou=settings.models.seg_iou_threshold,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - start) * 1000.0

        garments: List[GarmentInstance] = []
        if results:
            r = results[0]
            if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()
                masks = r.masks.data.cpu().numpy()  # (N, mh, mw), already ~frame-sized

                for i in range(len(cls_ids)):
                    class_id = int(cls_ids[i])
                    if self._garment_ids and class_id not in self._garment_ids:
                        continue
                    class_name = self.class_names.get(class_id, str(class_id))

                    mask = masks[i]
                    if mask.shape[0] != h or mask.shape[1] != w:
                        import cv2
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask > 0.5).astype(np.uint8)

                    garments.append(GarmentInstance(
                        class_name=class_name,
                        confidence=float(confs[i]),
                        bbox_xyxy=tuple(float(v) for v in xyxy[i]),
                        mask=mask_bin,
                    ))

        return SegResult(garments=garments, inference_ms=inference_ms)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
