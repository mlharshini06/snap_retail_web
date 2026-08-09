"""
config.py
Central configuration for Snap Retail Mirror.

All tunables live here so the rest of the codebase never hardcodes
magic numbers or reads os.environ directly. Import this module and
use `settings` everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    # Optional: if python-dotenv is installed, load a local .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
LOGS_DIR = BASE_DIR / "logs"
QR_DIR = SNAPSHOTS_DIR / "qr"
PDF_DIR = SNAPSHOTS_DIR / "pdf"
TRYON_DIR = SNAPSHOTS_DIR / "tryon"

for _dir in (ASSETS_DIR, SNAPSHOTS_DIR, LOGS_DIR, QR_DIR, PDF_DIR, TRYON_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = _env_int("CAMERA_INDEX", 0)
    width: int = _env_int("CAMERA_WIDTH", 1280)
    height: int = _env_int("CAMERA_HEIGHT", 720)
    target_fps: int = _env_int("CAMERA_TARGET_FPS", 30)
    reconnect_delay_sec: float = 1.5
    max_reconnect_attempts: int = 0  # 0 = retry forever


@dataclass(frozen=True)
class ModelConfig:
    # Exactly two models are loaded by this application. Do not add more.
    pose_model_path: str = os.getenv("POSE_MODEL_PATH", "yolo11n-pose.pt")
    seg_model_path: str = os.getenv("SEG_MODEL_PATH", "yolo11n-seg.pt")

    pose_conf_threshold: float = _env_float("POSE_CONF_THRESHOLD", 0.5)
    seg_conf_threshold: float = _env_float("SEG_CONF_THRESHOLD", 0.45)
    seg_iou_threshold: float = _env_float("SEG_IOU_THRESHOLD", 0.5)

    # Force a device string ("cuda", "cuda:0", "cpu") or leave None to auto-detect.
    force_device: str | None = os.getenv("FORCE_DEVICE") or None

    # COCO-pose keypoint index for the nose, used for skin-tone sampling.
    nose_keypoint_index: int = 0
    skin_patch_size: int = 20  # 20x20 px crop around the nose

    # YOLO-Seg class names we treat as "garment" classes for clothing
    # classification / coloring. Anything else detected is ignored.
    garment_classes: tuple = (
        "shirt", "t-shirt", "tshirt", "hoodie", "sweater", "sweatshirt",
        "dress", "coat", "jacket", "blazer", "jeans", "pants", "trousers",
        "skirt", "shorts", "top", "blouse", "suit", "vest",
    )


@dataclass(frozen=True)
class ColorConfig:
    kmeans_clusters: int = _env_int("KMEANS_CLUSTERS", 3)
    kmeans_random_state: int = 42
    kmeans_max_iter: int = 100


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct")
    timeout_sec: float = _env_float("OPENROUTER_TIMEOUT_SEC", 30.0)
    max_retries: int = 2
    temperature: float = 0.7
    max_tokens: int = 900


@dataclass(frozen=True)
class FalConfig:
    api_key: str = os.getenv("FAL_KEY", "FAL_KEY")
    tryon_model: str = os.getenv("FAL_TRYON_MODEL", "fal-ai/kling/v1-5/kolors-virtual-try-on")
    submit_url: str = "https://queue.fal.run/{model}"
    status_url: str = "https://queue.fal.run/{model}/requests/{request_id}/status"
    result_url: str = "https://queue.fal.run/{model}/requests/{request_id}"
    poll_interval_sec: float = _env_float("FAL_POLL_INTERVAL_SEC", 2.0)
    poll_timeout_sec: float = _env_float("FAL_POLL_TIMEOUT_SEC", 120.0)
    request_timeout_sec: float = _env_float("FAL_REQUEST_TIMEOUT_SEC", 30.0)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = _env_int("POSTGRES_PORT", 5432)
    dbname: str = os.getenv("POSTGRES_DB", "snap_retail_mirror")
    user: str = os.getenv("POSTGRES_USER", "POSTGRES_USER")
    password: str = os.getenv("POSTGRES_PASSWORD", "POSTGRES_PASSWORD")
    min_connections: int = _env_int("POSTGRES_MIN_CONN", 1)
    max_connections: int = _env_int("POSTGRES_MAX_CONN", 8)
    connect_timeout_sec: int = _env_int("POSTGRES_CONNECT_TIMEOUT", 5)
    statement_timeout_ms: int = _env_int("POSTGRES_STATEMENT_TIMEOUT_MS", 8000)


@dataclass(frozen=True)
class ExecutorConfig:
    max_workers: int = _env_int("EXECUTOR_MAX_WORKERS", 6)
    task_timeout_sec: float = _env_float("EXECUTOR_TASK_TIMEOUT_SEC", 90.0)


@dataclass(frozen=True)
class UIConfig:
    window_name: str = "Snap Retail Mirror"
    side_panel_width: int = 420
    font_scale: float = 0.55
    line_thickness: int = 2
    show_fps: bool = True
    recommend_key: str = "r"
    quit_key: str = "q"
    snapshot_key: str = "s"
    toggle_pose_key: str = "p"
    toggle_seg_key: str = "m"


@dataclass(frozen=True)
class LoggingConfig:
    log_file: Path = LOGS_DIR / "snap_retail_mirror.log"
    level: str = os.getenv("LOG_LEVEL", "INFO")
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


@dataclass(frozen=True)
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    fal: FalConfig = field(default_factory=FalConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


settings = Settings()
