"""
ai/openrouter_client.py
Synchronous HTTP client for OpenRouter's chat-completions endpoint,
used to turn (skin tone, clothing type, clothing color) into a
structured styling recommendation. This module makes blocking network
calls — it must only ever run inside the background executor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from config import settings
from utils.logger import get_logger

logger = get_logger("openrouter_client")

_SYSTEM_PROMPT = (
    "You are a professional fashion stylist working inside a smart retail "
    "mirror. You receive a shopper's detected skin tone and current outfit, "
    "and you must respond with ONLY a single strict JSON object — no prose, "
    "no markdown fences, no commentary before or after it. The JSON object "
    "must have exactly these keys: "
    '"recommended_colors" (array of color name strings), '
    '"recommended_outfits" (array of short outfit description strings), '
    '"styling_tips" (array of short tip strings). '
    "Each array should contain 3 to 5 items."
)


@dataclass
class RecommendationResult:
    recommended_colors: List[str] = field(default_factory=list)
    recommended_outfits: List[str] = field(default_factory=list)
    styling_tips: List[str] = field(default_factory=list)
    raw_response: Optional[dict] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_dict(self) -> dict:
        return {
            "recommended_colors": self.recommended_colors,
            "recommended_outfits": self.recommended_outfits,
            "styling_tips": self.styling_tips,
        }


def _build_user_prompt(skin_tone_hex: str, clothing_type: str, clothing_color_hex: str) -> str:
    payload = {
        "skin_tone_hex": skin_tone_hex,
        "current_clothing_type": clothing_type,
        "current_clothing_color_hex": clothing_color_hex,
    }
    return (
        "Shopper attributes (JSON):\n"
        f"{json.dumps(payload)}\n\n"
        "Recommend complementary colors, 3-5 full outfit ideas, and styling "
        "tips tailored to this skin tone and current garment. Respond with "
        "ONLY the JSON object described in your instructions."
    )


def _extract_json_object(text: str) -> Optional[dict]:
    """Models occasionally wrap JSON in markdown fences or add stray
    whitespace/prose despite instructions. Extract the first top-level
    {...} block and parse it defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Failed to parse OpenRouter JSON payload, candidate=%r", candidate[:500])
        return None


def get_style_recommendation(
    skin_tone_hex: str,
    clothing_type: str,
    clothing_color_hex: str,
) -> RecommendationResult:
    """Blocking call. Run via the background executor only."""
    if not settings.openrouter.api_key or settings.openrouter.api_key == "OPENROUTER_API_KEY":
        return RecommendationResult(error="OPENROUTER_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {settings.openrouter.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://snap-retail-mirror.local",
        "X-Title": "Snap Retail Mirror",
    }
    body = {
        "model": settings.openrouter.model,
        "temperature": settings.openrouter.temperature,
        "max_tokens": settings.openrouter.max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(skin_tone_hex, clothing_type, clothing_color_hex)},
        ],
    }

    last_error: Optional[str] = None
    for attempt in range(1, settings.openrouter.max_retries + 2):
        try:
            resp = requests.post(
                settings.openrouter.base_url,
                headers=headers,
                json=body,
                timeout=settings.openrouter.timeout_sec,
            )
            if resp.status_code == 429:
                last_error = "Rate limited by OpenRouter"
                logger.warning("OpenRouter 429 on attempt %d, backing off", attempt)
                time.sleep(min(2 ** attempt, 8))
                continue
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            parsed = _extract_json_object(content)
            if parsed is None:
                last_error = "Could not parse strict JSON from model response"
                continue

            return RecommendationResult(
                recommended_colors=list(parsed.get("recommended_colors", []))[:5],
                recommended_outfits=list(parsed.get("recommended_outfits", []))[:5],
                styling_tips=list(parsed.get("styling_tips", []))[:5],
                raw_response=parsed,
            )

        except requests.Timeout:
            last_error = "OpenRouter request timed out"
            logger.warning("OpenRouter timeout on attempt %d", attempt)
        except requests.RequestException as exc:
            last_error = f"OpenRouter request failed: {exc}"
            logger.warning("OpenRouter request error on attempt %d: %s", attempt, exc)
        except (KeyError, IndexError, ValueError) as exc:
            last_error = f"Unexpected OpenRouter response shape: {exc}"
            logger.exception("Unexpected OpenRouter response shape")

    logger.error("OpenRouter recommendation failed after retries: %s", last_error)
    return RecommendationResult(error=last_error or "Unknown OpenRouter failure")
