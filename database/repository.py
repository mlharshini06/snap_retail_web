"""
database/repository.py
All SQL lives here. Every public function is synchronous (blocking) by
design — callers MUST invoke these through utils.async_executor.executor
from background tasks, never directly from the camera thread.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from database.postgres_pool import db_pool
from utils.logger import get_logger

logger = get_logger("repository")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    session_token VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    color_hex VARCHAR(7),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    skin_tone_hex VARCHAR(7),
    clothing_type VARCHAR(100),
    clothing_color_hex VARCHAR(7),
    openrouter_response JSONB,
    fal_image_url TEXT,
    snapshot_path TEXT,
    pdf_path TEXT,
    qr_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations (user_id);
"""


def initialize_schema() -> None:
    """Create tables/indexes if they don't already exist. Safe to call
    on every app startup (idempotent, IF NOT EXISTS everywhere)."""
    with db_pool.get_cursor(commit=True, dict_cursor=False) as cur:
        cur.execute(_SCHEMA_SQL)
    logger.info("Database schema verified/created")


def get_or_create_user(session_token: str) -> int:
    with db_pool.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO users (session_token)
            VALUES (%s)
            ON CONFLICT (session_token) DO UPDATE SET session_token = EXCLUDED.session_token
            RETURNING id
            """,
            (session_token,),
        )
        row = cur.fetchone()
        return row["id"]


def insert_product(name: str, category: str, color_hex: Optional[str], metadata: Optional[dict] = None) -> int:
    with db_pool.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO products (name, category, color_hex, metadata)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, category, color_hex, json.dumps(metadata or {})),
        )
        row = cur.fetchone()
        return row["id"]


def insert_recommendation(
    user_id: Optional[int],
    skin_tone_hex: Optional[str],
    clothing_type: Optional[str],
    clothing_color_hex: Optional[str],
    openrouter_response: Optional[dict],
    fal_image_url: Optional[str] = None,
    snapshot_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    qr_path: Optional[str] = None,
) -> int:
    with db_pool.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO recommendations (
                user_id, skin_tone_hex, clothing_type, clothing_color_hex,
                openrouter_response, fal_image_url, snapshot_path, pdf_path, qr_path
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id, skin_tone_hex, clothing_type, clothing_color_hex,
                json.dumps(openrouter_response or {}), fal_image_url,
                snapshot_path, pdf_path, qr_path,
            ),
        )
        row = cur.fetchone()
        logger.info("Recommendation %s stored for user_id=%s", row["id"], user_id)
        return row["id"]


def update_recommendation_fal_url(recommendation_id: int, fal_image_url: str) -> None:
    with db_pool.get_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE recommendations SET fal_image_url = %s WHERE id = %s",
            (fal_image_url, recommendation_id),
        )


def get_recent_recommendations(limit: int = 20) -> list[dict[str, Any]]:
    with db_pool.get_cursor() as cur:
        cur.execute(
            "SELECT * FROM recommendations ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return list(cur.fetchall())


def get_recommendation_history_for_user(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with db_pool.get_cursor() as cur:
        cur.execute(
            "SELECT * FROM recommendations WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        return list(cur.fetchall())
