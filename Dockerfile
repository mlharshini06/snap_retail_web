# Snap Retail Mirror — web deployment image
# CPU-only, single container: FastAPI + YOLO11 (pose + seg) + OpenRouter.
FROM python:3.11-slim

# OpenCV (even the headless build) needs these shared libs on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes.
COPY requirements-web.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-web.txt

# App code + model weights (small: ~6MB each, baked in so there is no
# runtime download / no dependency on external model storage).
COPY config.py .
COPY ai ./ai
COPY models ./models
COPY utils ./utils
COPY web ./web
COPY yolo11n-pose.pt .
COPY yolo11n-seg.pt .

# Platforms like Render/Railway inject $PORT at runtime; default to
# 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Single worker: the two YOLO models are loaded once into this
# process's memory and reused for every request (see web/main.py).
# Scale by increasing the host's CPU/RAM and running more replicas,
# not more workers in one container, so models aren't loaded twice.
CMD ["sh", "-c", "uvicorn web.main:app --host 0.0.0.0 --port ${PORT}"]
