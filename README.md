# Snap Retail Mirror

A real-time smart retail mirror: person detection, pose estimation, clothing
segmentation/classification, skin-tone and garment-color extraction, AI
styling recommendations (OpenRouter), virtual try-on (Fal.ai Kolors), and
PostgreSQL-backed recommendation history with PDF/QR export — all running
locally on a laptop webcam without blocking the video feed.

> **Want a browser-based, phone/laptop/tablet-friendly version instead of
> the local webcam app below?** See [`WEB_DEPLOYMENT.md`](./WEB_DEPLOYMENT.md)
> for the FastAPI web prototype (`web/main.py`), which reuses the exact
> same detection/AI pipeline described here.

## Architecture at a glance

- **Camera thread (`app.py`)** does only: capture → YOLO-Pose → YOLO-Seg →
  local color extraction → overlay → display. Nothing else runs on this
  thread.
- **Everything else** (OpenRouter, Fal.ai, PostgreSQL, PDF, QR, file writes)
  runs on a shared `ThreadPoolExecutor` (`utils/async_executor.py`) and
  reports back through a lock-guarded `SharedState` object that the camera
  loop reads each frame. The UI never blocks waiting on a network call.
- Exactly **two** models are loaded: `yolo11n-pose.pt` and `yolo11n-seg.pt`.

## 1. Prerequisites

- Python 3.10–3.12
- A webcam
- (Optional but recommended) an NVIDIA GPU with a recent driver, for real-time FPS
- PostgreSQL 13+ (local install or Docker)
- API keys: [OpenRouter](https://openrouter.ai/keys), [Fal.ai](https://fal.ai/dashboard/keys)

## 2. Installation

```bash
git clone <your-repo-url> snap_retail_mirror
cd snap_retail_mirror
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### GPU (CUDA) setup

The `requirements.txt` pulls the default PyPI `torch` build, which is
CPU-only. For GPU acceleration, install the CUDA build that matches your
driver **before** or **instead of** the plain `torch` line, e.g. for CUDA 12.1:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check `nvidia-smi` for your driver's supported CUDA version and pick the
matching wheel from https://pytorch.org/get-started/locally/. The app
auto-detects CUDA at startup (`utils/device.py`) — no code changes needed.

### CPU-only setup

No extra steps: the default `pip install -r requirements.txt` already gives
you a working CPU pipeline. Expect noticeably lower FPS (the app targets 30
FPS on GPU and degrades gracefully on CPU — you'll typically see single- to
low-double-digit FPS depending on your machine).

### YOLO model weights

`ultralytics` downloads `yolo11n-pose.pt` / `yolo11n-seg.pt` automatically on
first run if they're not already present in the working directory. To
pre-download:

```python
from ultralytics import YOLO
YOLO("yolo11n-pose.pt")
YOLO("yolo11n-seg.pt")
```

> **Important note on garment classes:** the stock `yolo11n-seg.pt` is
> trained on COCO, which does not include fine-grained clothing categories
> (shirt/hoodie/dress/etc.) — COCO's `person` class would be the only
> match. For real garment segmentation + classification, fine-tune (or
> download a checkpoint already fine-tuned on) a fashion dataset such as
> DeepFashion2, and point `SEG_MODEL_PATH` (see below) at that checkpoint.
> `models/segmentation_detector.py` is written to work with any class list
> — it filters by name against `settings.models.garment_classes` in
> `config.py`, so swapping the checkpoint is a one-line config change.

## 3. PostgreSQL setup

```bash
# Docker, quickest path:
docker run --name srm-postgres -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=snap_retail_mirror -p 5432:5432 -d postgres:16

# or, local install:
createdb snap_retail_mirror
```

No manual schema step is needed — `database/repository.py::initialize_schema()`
runs `CREATE TABLE IF NOT EXISTS` for `users`, `products`, and
`recommendations` automatically on app startup, in the background, without
blocking the camera loop. If PostgreSQL is unreachable, the app logs a
warning and keeps running with DB features disabled (recommendations still
work — they just won't persist).

## 4. Environment variables

Create a `.env` file in the project root (auto-loaded via `python-dotenv`):

```ini
# --- Cloud APIs ---
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct
FAL_KEY=...
FAL_TRYON_MODEL=fal-ai/kling/v1-5/kolors-virtual-try-on

# --- PostgreSQL ---
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=snap_retail_mirror
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme

# --- Camera / performance (optional, sensible defaults exist) ---
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_TARGET_FPS=30
FORCE_DEVICE=            # leave empty to auto-detect; or "cpu" / "cuda:0"
EXECUTOR_MAX_WORKERS=6
LOG_LEVEL=INFO
```

See `config.py` for every other tunable (thresholds, timeouts, retry
counts) — all of them read from `os.environ` with safe defaults, so the
app runs even with an empty `.env`.

## 5. Running

```bash
python app.py
```

### Controls

| Key | Action |
|---|---|
| `R` | Capture snapshot → run OpenRouter recommendation (+ Fal.ai try-on if a garment image URL is wired up) → generate PDF/QR → store in PostgreSQL, all in the background |
| `S` | Save a manual snapshot to `snapshots/` (async, non-blocking) |
| `P` | Toggle pose skeleton overlay |
| `M` | Toggle garment mask overlay |
| `Q` | Quit (releases camera, closes DB pool, shuts down the executor) |

The right-hand side panel shows the last completed recommendation, cloud/DB
status, and elapsed pipeline time; the bottom status bar shows live
Cloud/Try-On/DB state so you can tell at a glance whether a background job
is running, succeeded, or failed — the video feed itself never freezes
while this happens.

### On virtual try-on and `garment_image_url`

Fal's Kolors Virtual Try-On needs both a person image (the webcam
snapshot, handled automatically) **and** a target garment product image
URL. This app doesn't ship a product catalog UI, so
`ai/recommendation_engine.run_recommendation_pipeline()` accepts
`garment_image_url` as a parameter — wire it up to whatever you use to let
a shopper pick a product (e.g. read it from the `products` table via
`database/repository.py`, or pass a fixed demo URL for testing). If it's
omitted, the pipeline still runs the OpenRouter recommendation, PDF, QR,
and DB write — it just skips the try-on stage rather than guessing a
garment.

## 6. Troubleshooting

**Camera won't open / `Failed to open camera 0`**
Try a different `CAMERA_INDEX` (1, 2, ...). On Linux, check `ls /dev/video*`
and permissions (`sudo usermod -aG video $USER`, then re-login). On macOS,
grant camera permission to your terminal/IDE in System Settings → Privacy.

**Low FPS on CPU**
Expected — YOLO11n is small but still real-time-GPU-oriented. Lower
`CAMERA_WIDTH`/`CAMERA_HEIGHT`, or install the CUDA `torch` build (see
above) if you have an NVIDIA GPU.

**`CUDA out of memory` over a long session**
The app calls `torch.cuda.empty_cache()` + `gc.collect()` after every
cloud-API round trip (`utils/device.py::free_gpu_memory`). If you still see
growth, lower `EXECUTOR_MAX_WORKERS` so fewer concurrent background jobs
hold image buffers at once.

**OpenRouter returns malformed JSON / recommendation panel shows an error**
`ai/openrouter_client.py` requests `response_format: json_object` and
defensively extracts the first `{...}` block even if the model adds stray
text, but some models still misbehave. Check `logs/snap_retail_mirror.log`
for the raw candidate that failed to parse, and consider switching
`OPENROUTER_MODEL` if it persists.

**Fal.ai job times out**
Increase `FAL_POLL_TIMEOUT_SEC`. Check `logs/snap_retail_mirror.log` for
the `request_id` and inspect it directly at
`https://queue.fal.run/<model>/requests/<request_id>/status`.

**`psycopg2.OperationalError` / recommendations aren't being saved**
Confirm PostgreSQL is reachable with the configured host/port/credentials
(`psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB`). The app
degrades gracefully (DB status shows `error` in the status bar) instead of
crashing, so the rest of the pipeline keeps working while you fix it.

**Window doesn't appear / `cv2.imshow` errors on a headless server**
This app requires a display (it's a physical mirror UI). Run it on a
machine with a GUI session, not a headless server/container.

## 7. Project structure

```
snap_retail_mirror/
├── app.py                          # main entrypoint / camera thread
├── config.py                       # all configuration, env vars
├── requirements.txt
├── models/
│   ├── pose_detector.py            # YOLO11n-Pose wrapper
│   ├── segmentation_detector.py    # YOLO11n-Seg wrapper
│   ├── color_detector.py           # skin tone + garment color (KMeans)
│   └── clothing_classifier.py      # label normalization
├── ai/
│   ├── openrouter_client.py        # styling recommendations
│   ├── fal_client.py               # Kolors virtual try-on
│   └── recommendation_engine.py    # orchestrates the full "press R" pipeline
├── database/
│   ├── postgres_pool.py            # SimpleConnectionPool wrapper
│   └── repository.py               # schema + all SQL
├── ui/
│   ├── overlay.py                  # per-frame drawing
│   └── side_panel.py               # recommendation panel
├── utils/
│   ├── async_executor.py           # shared background thread pool
│   ├── device.py                   # CUDA/CPU detection + memory cleanup
│   ├── image_utils.py              # crop/color helpers
│   ├── logger.py                   # rotating file logging
│   ├── pdf_generator.py            # reportlab report
│   └── qr_generator.py             # qrcode wrapper
├── assets/
└── snapshots/                      # jpg snapshots, qr/, pdf/, tryon/
```
