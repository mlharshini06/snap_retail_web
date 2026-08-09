"""
ai/fal_client.py
Synchronous client for Fal.ai's queue-based REST API, used to run
Kolors Virtual Try-On on the captured snapshot against a chosen
garment image. Blocking network calls — background executor only.

Flow:
  1. submit_tryon()  -> POST the request (image data is inlined as a
     base64 data: URI, so no separate upload endpoint is needed) and
     get back a request_id.
  2. poll_tryon()    -> poll the status endpoint until COMPLETED/ERROR,
     bounded by settings.fal.poll_timeout_sec.
  3. fetch_result()  -> GET the final payload and pull out the image URL.

run_virtual_tryon() wraps all three so callers (recommendation_engine)
get one function that returns a final URL or raises.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import settings
from utils.logger import get_logger

logger = get_logger("fal_client")


class FalError(RuntimeError):
    pass


@dataclass
class TryOnResult:
    image_url: Optional[str]
    request_id: str
    raw_response: dict


def _headers() -> dict:
    return {
        "Authorization": f"Key {settings.fal.api_key}",
        "Content-Type": "application/json",
    }


def _to_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def submit_tryon(person_image_bytes: bytes, garment_image_url: str) -> str:
    """Submit a try-on job. `garment_image_url` is a URL to the target
    garment product image (e.g. from your catalog/products table).
    Returns the Fal request_id."""
    if not settings.fal.api_key or settings.fal.api_key == "FAL_KEY":
        raise FalError("FAL_KEY is not configured")

    url = settings.fal.submit_url.format(model=settings.fal.tryon_model)
    body = {
        "human_image_url": _to_data_uri(person_image_bytes),
        "garment_image_url": garment_image_url,
    }

    resp = requests.post(url, headers=_headers(), json=body, timeout=settings.fal.request_timeout_sec)
    if resp.status_code >= 400:
        raise FalError(f"Fal submit failed ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    request_id = data.get("request_id")
    if not request_id:
        raise FalError(f"Fal submit response missing request_id: {data}")

    logger.info("Fal try-on submitted, request_id=%s", request_id)
    return request_id


def poll_tryon(request_id: str) -> dict:
    """Poll until the job reaches a terminal state or the poll timeout
    elapses. Returns the final status payload."""
    status_url = settings.fal.status_url.format(model=settings.fal.tryon_model, request_id=request_id)
    deadline = time.monotonic() + settings.fal.poll_timeout_sec

    while time.monotonic() < deadline:
        resp = requests.get(status_url, headers=_headers(), timeout=settings.fal.request_timeout_sec)
        if resp.status_code >= 400:
            raise FalError(f"Fal status check failed ({resp.status_code}): {resp.text[:300]}")

        data = resp.json()
        status = data.get("status")
        logger.debug("Fal request %s status=%s", request_id, status)

        if status == "COMPLETED":
            return data
        if status in ("ERROR", "FAILED"):
            raise FalError(f"Fal try-on job failed: {data}")

        time.sleep(settings.fal.poll_interval_sec)

    raise FalError(f"Fal try-on job {request_id} timed out after {settings.fal.poll_timeout_sec}s")


def fetch_result(request_id: str) -> dict:
    result_url = settings.fal.result_url.format(model=settings.fal.tryon_model, request_id=request_id)
    resp = requests.get(result_url, headers=_headers(), timeout=settings.fal.request_timeout_sec)
    if resp.status_code >= 400:
        raise FalError(f"Fal result fetch failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _extract_image_url(payload: dict) -> Optional[str]:
    # Kolors-style responses typically nest the output under "images": [{"url": ...}]
    images = payload.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            return first.get("url")
        if isinstance(first, str):
            return first
    # Some Fal models return a single "image" object instead.
    image = payload.get("image")
    if isinstance(image, dict):
        return image.get("url")
    return None


def run_virtual_tryon(person_image_bytes: bytes, garment_image_url: str) -> TryOnResult:
    """End-to-end: submit -> poll -> fetch -> extract URL. Blocking;
    run only via the background executor."""
    request_id = submit_tryon(person_image_bytes, garment_image_url)
    status_payload = poll_tryon(request_id)

    # The completed status payload sometimes already contains the output;
    # fall back to an explicit result fetch if it doesn't.
    image_url = _extract_image_url(status_payload)
    result_payload = status_payload
    if image_url is None:
        result_payload = fetch_result(request_id)
        image_url = _extract_image_url(result_payload)

    if image_url is None:
        raise FalError(f"Could not locate an image URL in Fal response: {result_payload}")

    logger.info("Fal try-on complete, request_id=%s image_url=%s", request_id, image_url)
    return TryOnResult(image_url=image_url, request_id=request_id, raw_response=result_payload)
