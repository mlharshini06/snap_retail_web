# Snap Retail Mirror — Web Prototype

A browser-accessible version of Snap Retail Mirror. Same AI pipeline as the
desktop app (YOLO11n-Pose, YOLO11n-Seg, OpenCV + K-Means color detection,
OpenRouter styling recommendations) — reachable from a phone, tablet,
laptop, or desktop over a public HTTPS URL, no Python install required for
the people using it.

```
Open URL → Allow Camera → See yourself → Capture & Analyze → Get Recommendations → See results
```

---

## 1. What changed vs. the desktop app

**Not touched** (same files, same logic):
- `models/pose_detector.py`, `models/segmentation_detector.py`,
  `models/color_detector.py`, `models/clothing_classifier.py`
- `ai/openrouter_client.py`
- `app.py` (the original desktop entrypoint still works exactly as before)

**Added** (new files only — nothing existing was rewritten):
- `web/main.py` — FastAPI backend. Loads both YOLO models once at startup
  and reuses them for every request.
- `web/static/` — the browser frontend (plain HTML/CSS/JS, no build step).
- `ai/web_pipeline.py` — a small, stateless wrapper that calls the same
  detectors/color/classifier/OpenRouter functions the desktop app uses,
  but skips snapshot-to-disk, PostgreSQL, PDF, QR, and Fal.ai (out of
  scope for this prototype, per the brief).
- `requirements-web.txt`, `Dockerfile`, `.dockerignore`, `render.yaml` —
  deployment plumbing.

**Not included in the web prototype** (unchanged desktop-only files, not
imported by `web/main.py` at all): `database/`, `utils/pdf_generator.py`,
`utils/qr_generator.py`, `ai/fal_client.py`, `ui/`.

## 2. Updated project structure

```
snap_retail-main/
├── app.py                    # desktop app (unchanged)
├── config.py                 # shared config (unchanged)
├── requirements.txt          # desktop deps (unchanged)
├── requirements-web.txt      # NEW — lean deps for the web deployment
├── Dockerfile                # NEW
├── .dockerignore             # NEW
├── render.yaml                # NEW — optional Render blueprint
├── .env.example               # NEW
├── ai/
│   ├── openrouter_client.py   # unchanged, reused as-is
│   ├── recommendation_engine.py  # unchanged (desktop-only pipeline)
│   ├── fal_client.py          # unchanged (desktop-only, not used by web)
│   └── web_pipeline.py        # NEW — stateless web pipeline
├── models/                    # unchanged, reused as-is
├── web/                       # NEW
│   ├── main.py                 # FastAPI app + API routes
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── database/, ui/, utils/pdf_generator.py, utils/qr_generator.py, ai/fal_client.py
│                              # unchanged, NOT used by the web app
└── yolo11n-pose.pt, yolo11n-seg.pt   # unchanged
```

## 3. API

| Method | Path         | Purpose                                                             |
|--------|--------------|----------------------------------------------------------------------|
| GET    | `/health`    | Liveness check — `{"status":"ok","models_loaded":true}`             |
| POST   | `/analyze`   | Multipart `file=<jpeg>` → detection only (no AI call)               |
| POST   | `/recommend` | Multipart `file=<jpeg>` → detection + OpenRouter recommendation      |

`/recommend` response shape:

```json
{
  "ok": true,
  "clothing_type": "Shirt",
  "skin_tone": "#8D684D",
  "garment_color": "#564C3C",
  "recommendations": {
    "recommended_colors": ["Navy", "Cream", "Rust"],
    "recommended_outfits": ["Navy chinos with a cream sweater"],
    "styling_tips": ["Roll the sleeves for a relaxed look"]
  },
  "message": null
}
```

Every request is fully self-contained — the backend keeps **no per-user or
global state** between requests, so many browsers can hit the same public
URL at once without seeing each other's results. No camera image is ever
written to disk.

## 4. Local testing

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-web.txt

cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in a browser on the same machine. Camera access
works on `localhost` without HTTPS (browsers special-case localhost). To
test from your **phone** on the same Wi‑Fi before deploying anywhere:

```bash
uvicorn web.main:app --host 0.0.0.0 --port 8000
```

then visit `http://<your-laptop-LAN-IP>:8000` from your phone — but note
most mobile browsers **require HTTPS** for `getUserMedia` (camera access)
except on `localhost`, so this LAN test may only work on some
devices/browsers. Deploy behind HTTPS (Section 6) for reliable phone
testing.

Quick check the API works without a browser:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend -F "file=@/path/to/photo.jpg"
```

## 5. Deployment recommendation

**Recommended: Render.com (Docker Web Service).** It runs arbitrary Docker
containers (so the full YOLO + torch + OpenCV stack works unmodified),
gives you free automatic HTTPS on a public `*.onrender.com` URL, needs no
server administration, and its Starter plan (512MB–1GB+ RAM) is enough for
two small YOLO11n models doing single-frame inference.

Why not the alternatives commonly suggested for "just deploy a Python app":
- **Vercel / Netlify / Streamlit Community Cloud (serverless/edge)** — not
  designed for multi-hundred-MB ML dependencies (torch + ultralytics) or
  for holding two models resident in memory across requests; builds
  frequently fail or cold-start unacceptably slowly.
- **Free-tier-only serverless functions** — typical request timeouts
  (10–30s) are too tight once you add YOLO inference + an OpenRouter round
  trip on a cold CPU.
- **Heroku free dynos** — no longer offered on the free tier at all.

Render's **free** web service tier also works for a class demo, with two
caveats: it spins down after ~15 minutes idle (the next request pays a
30–60s cold start while models reload into memory), and free-tier RAM
(512MB) is tight — if you hit out-of-memory errors, move to the $7/mo
Starter plan, which fixes both.

### 5a. Deployment architecture

```
Browser (phone/laptop/tablet/desktop)
   │  HTTPS
   ▼
Render Web Service (Docker container)
   │  uvicorn (1 process, N threadpool threads)
   │  ├─ PoseDetector   (loaded once at startup, held in memory)
   │  └─ SegmentationDetector (loaded once at startup, held in memory)
   ▼
OpenRouter API (only on "Get Recommendations")
```

One container instance serves every visitor concurrently; inference calls
are serialized through a lock (see `web/main.py`) so concurrent users are
handled correctly on a single CPU instance. If you need to serve many
simultaneous users, scale by adding more Render instances (each loads its
own copy of the models) rather than more processes in one container.

### 5b. Environment variables

Set these in the Render dashboard (Environment tab) or via `render.yaml`:

| Variable             | Required | Default                          | Notes                                   |
|----------------------|----------|-----------------------------------|------------------------------------------|
| `OPENROUTER_API_KEY` | Yes      | —                                  | From https://openrouter.ai/keys          |
| `OPENROUTER_MODEL`   | No       | `qwen/qwen-2.5-72b-instruct`      |                                            |
| `PORT`               | No       | injected by Render automatically  | Don't set manually on Render              |
| `MAX_IMAGE_SIDE`     | No       | `960`                              | Lower (e.g. `720`) if inference feels slow on the free tier |
| `CORS_ORIGINS`       | No       | `*`                                 | Fine for this prototype (no auth/secrets involved) |
| `LOG_LEVEL`          | No       | `INFO`                             |                                            |

### 5c. Build command

None needed — Render builds the `Dockerfile` directly. If asked for a
build command explicitly (e.g. on a non-Docker "native environment"
service instead of a Docker service), use:

```bash
pip install -r requirements-web.txt
```

### 5d. Start command

Render (Docker service) uses the Dockerfile's `CMD` automatically. If
configuring a native-environment service instead:

```bash
uvicorn web.main:app --host 0.0.0.0 --port $PORT
```

### 5e. Port configuration

The app reads `$PORT` at startup (`Dockerfile` passes it straight to
uvicorn). Render/Railway inject this automatically — don't hardcode a
port.

### 5f. HTTPS setup

Render terminates HTTPS for you automatically on the `*.onrender.com`
domain — nothing to configure. This is required: browsers block
`getUserMedia` (camera access) on any non-HTTPS origin other than
`localhost`.

### 5g. Public URL setup

1. Push this repo to GitHub.
2. On Render: **New → Web Service → connect the repo**.
3. Render detects `render.yaml`/`Dockerfile` automatically; confirm plan
   (Starter recommended) and region.
4. Add `OPENROUTER_API_KEY` in the Environment tab.
5. Deploy. Render gives you `https://snap-retail-mirror-XXXX.onrender.com`
   (or your chosen name) — that's the URL to hand out.
6. Optional: add a custom domain under Settings → Custom Domains (Render
   provisions HTTPS for it automatically too).

### 5h. How users access it

Anyone opens the HTTPS URL from Section 5g on **any device with a camera
and a modern browser** (Chrome, Safari, Edge, Firefox) — phone, laptop,
tablet, or desktop. No app install, no Python, no account. They tap
"Start Camera," allow the permission prompt, capture a frame, and tap
"Get Recommendations."

### 5i. Alternative: Railway

If you'd rather use Railway instead of Render, the same `Dockerfile`
works unmodified:

1. `railway init` → connect the repo.
2. Railway auto-detects the Dockerfile and builds it.
3. Set `OPENROUTER_API_KEY` under Variables.
4. Railway assigns a public HTTPS `*.up.railway.app` URL automatically
   (Settings → Networking → Generate Domain).

## 6. Error handling reference

| Situation                                   | What the user sees                                                    |
|----------------------------------------------|-------------------------------------------------------------------------|
| Camera permission denied                     | "Please allow camera access to use Snap Retail Mirror."                |
| No camera on device                          | "No camera was found on this device."                                  |
| Backend unreachable / crashed / 500          | "Unable to connect to the AI service. Please try again."               |
| OpenRouter call fails (rate limit, timeout)  | "We couldn't generate recommendations right now. Please try again."    |
| No person visible in the captured frame      | "Please make sure a person is clearly visible."                        |

Full technical errors (stack traces, exception details) are only ever
logged server-side via Python's `logging` module — never sent to the
browser. See `web/main.py`'s `logger.exception(...)` calls.

## 7. Performance notes

- Both models are loaded exactly once, at process startup
  (`web/main.py: load_models()`), and reused for every request — never
  reloaded per-request.
- The browser only sends **one frame per action** (on "Capture & Analyze"
  and again on "Get Recommendations"), never a continuous video stream, to
  keep server load and mobile data usage low.
- Uploaded frames are downscaled server-side to `MAX_IMAGE_SIDE` (default
  960px longest side) before inference, so a 12MP phone photo doesn't
  balloon inference time.
- The UI shows an "Analyzing…" / spinner state instead of freezing while
  the backend runs — see `web/static/app.js`.

## 8. Troubleshooting

**"Please allow camera access…" even after tapping Allow**
The page must be served over HTTPS (or `localhost`). Check the deployed
URL starts with `https://`.

**First request after idle time is very slow**
Expected on Render's free tier — the container spun down and is cold-
starting (reloading both YOLO models into memory takes ~10-20s, plus
container boot). Upgrade to the Starter plan to avoid spin-down, or just
warn users the first load may take a moment.

**`/analyze` or `/recommend` returns `503`**
Models are still loading (right after a cold start) or failed to load —
check the server logs for the `Loading YOLO models...` / model load error
lines.

**"We couldn't generate recommendations right now"**
Usually `OPENROUTER_API_KEY` is missing/invalid, or OpenRouter rate-
limited/timed out. Check server logs for the specific `openrouter_client`
warning/error line.

**Out of memory on the deployed instance**
Two YOLO11n models + torch comfortably fit in 512MB–1GB, but if you see
OOM kills in the platform logs, move up one plan tier, or lower
`MAX_IMAGE_SIDE` to reduce peak memory during inference.

**Works on desktop but not on my phone's browser**
Some browsers (older Android WebViews, in-app browsers like Instagram's)
restrict `getUserMedia`. Ask the user to open the URL directly in Chrome,
Safari, or Firefox rather than an embedded in-app browser.
