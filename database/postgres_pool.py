"""
database/postgres_pool.py
Owns a single psycopg2 SimpleConnectionPool for the whole process.
Nothing outside this module should call psycopg2.connect() directly —
always borrow/return connections through get_connection().

All DB work happens on the background executor, never on the camera
thread, but the pool itself is thread-safe (psycopg2's SimpleConnectionPool
supports concurrent get/put from multiple threads).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

from config import settings
from utils.logger import get_logger

logger = get_logger("postgres_pool")


class PostgresPool:
    _instance: Optional["PostgresPool"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._pool: Optional[psycopg2.pool.SimpleConnectionPool] = None
        self._init_lock = threading.Lock()
        self._available = False

    @classmethod
    def instance(cls) -> "PostgresPool":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = PostgresPool()
            return cls._instance

    def initialize(self) -> bool:
        """Lazily create the pool. Returns True on success, False if the
        database is unreachable — callers must treat DB features as
        optional/degraded rather than crashing the app."""
        if self._available:
            return True

        with self._init_lock:
            if self._available:
                return True
            try:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    settings.database.min_connections,
                    settings.database.max_connections,
                    host=settings.database.host,
                    port=settings.database.port,
                    dbname=settings.database.dbname,
                    user=settings.database.user,
                    password=settings.database.password,
                    connect_timeout=settings.database.connect_timeout_sec,
                    options=f"-c statement_timeout={settings.database.statement_timeout_ms}",
                )
                self._available = True
                logger.info(
                    "PostgreSQL pool initialized (%d-%d connections) host=%s db=%s",
                    settings.database.min_connections, settings.database.max_connections,
                    settings.database.host, settings.database.dbname,
                )
                return True
            except Exception:
                logger.exception("Failed to initialize PostgreSQL pool; DB features disabled")
                self._pool = None
                self._available = False
                return False

    @property
    def available(self) -> bool:
        return self._available

    @contextmanager
    def get_connection(self):
        """Borrow a connection from the pool. Always returns it, even on
        exception, and rolls back any uncommitted work on failure so a
        bad transaction never poisons the pool."""
        if not self._available and not self.initialize():
            raise RuntimeError("PostgreSQL is not available")

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    logger.exception("Rollback failed")
            raise
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = False, dict_cursor: bool = True):
        with self.get_connection() as conn:
            cursor_factory = RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()

    def close_all(self) -> None:
        if self._pool is not None:
            try:
                self._pool.closeall()
                logger.info("PostgreSQL pool closed")
            except Exception:
                logger.exception("Error closing PostgreSQL pool")
            finally:
                self._available = False


db_pool = PostgresPool.instance()
