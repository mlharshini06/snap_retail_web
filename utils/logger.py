"""
utils/logger.py
Single source of truth for logging configuration. Every module calls
`get_logger(__name__)` instead of configuring logging itself.
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler

from config import settings

_lock = threading.Lock()
_configured = False


def _configure_root() -> None:
    global _configured
    with _lock:
        if _configured:
            return

        root = logging.getLogger("snap_retail_mirror")
        root.setLevel(settings.logging.level)
        root.propagate = False

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(threadName)-14s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            settings.logging.log_file,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(settings.logging.level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        console_handler.setLevel(settings.logging.level)

        root.addHandler(file_handler)
        root.addHandler(console_handler)

        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `snap_retail_mirror` root."""
    _configure_root()
    return logging.getLogger(f"snap_retail_mirror.{name}")


class LatencyTimer:
    """Context manager that logs how long a block took, in milliseconds.

    Usage:
        with LatencyTimer(logger, "pose_inference"):
            run_pose_model(frame)
    """

    def __init__(self, logger: logging.Logger, label: str, level: int = logging.DEBUG):
        self.logger = logger
        self.label = label
        self.level = level
        self._start = 0.0

    def __enter__(self):
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if exc_type is not None:
            self.logger.error("%s failed after %.1fms: %s", self.label, elapsed_ms, exc_val)
        else:
            self.logger.log(self.level, "%s took %.1fms", self.label, elapsed_ms)
        return False
