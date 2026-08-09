"""
utils/device.py
Single place that decides whether models run on CUDA or CPU, so both
YOLO models agree and we never duplicate detection logic.
"""

from __future__ import annotations

import gc
from functools import lru_cache

from config import settings
from utils.logger import get_logger

logger = get_logger("device")


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """Return 'cuda', 'cuda:N', or 'cpu'. Honors config.FORCE_DEVICE if set."""
    if settings.models.force_device:
        logger.info("Device forced via config: %s", settings.models.force_device)
        return settings.models.force_device

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA GPU detected: %s", name)
            return "cuda:0"
    except Exception:
        logger.warning("torch.cuda check failed, falling back to CPU", exc_info=True)

    logger.info("No CUDA GPU available, using CPU")
    return "cpu"


def free_gpu_memory() -> None:
    """Release cached CUDA memory and force a Python GC pass. Call this
    after heavy cloud-API round trips or large-image bursts to keep
    VRAM/host memory flat over long sessions."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
