"""
utils/async_executor.py

Every operation that is not "capture -> infer -> render -> display" must
run through this executor: database writes, PDF/QR generation,
OpenRouter calls, Fal.ai calls. This guarantees the camera thread is
never blocked.

Design notes:
- A single shared ThreadPoolExecutor backs the whole app.
- submit() never raises synchronously; failures are delivered to the
  optional `on_error` callback so callers never need to babysit futures.
- Every task is wrapped in try/except/finally so a failing task can
  never leave the pool, a DB connection, or a file handle dangling.
- A soft timeout is enforced per task via `settings.executor.task_timeout_sec`
  using a watchdog wrapper (ThreadPoolExecutor itself has no per-task
  timeout, so we implement one with `Future.result(timeout=...)` inside
  the watchdog thread instead of blocking the caller).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger("async_executor")


class AsyncExecutor:
    """Thin, safe wrapper around ThreadPoolExecutor for fire-and-forget
    and callback-style background work."""

    def __init__(self, max_workers: Optional[int] = None):
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers or settings.executor.max_workers,
            thread_name_prefix="srm-worker",
        )
        self._active_lock = threading.Lock()
        self._active_count = 0
        self._shutdown = False

    @property
    def active_count(self) -> int:
        with self._active_lock:
            return self._active_count

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
        task_name: str = "task",
        **kwargs: Any,
    ) -> Optional[Future]:
        """Run fn(*args, **kwargs) on the background pool.

        on_success(result) fires on the worker thread if fn succeeds.
        on_error(exc) fires on the worker thread if fn raises or times out.
        Callbacks must be cheap/non-blocking (e.g. push to a queue),
        since they execute on the worker thread.
        """
        if self._shutdown:
            logger.warning("Executor is shut down, dropping task '%s'", task_name)
            return None

        def _runner() -> Any:
            with self._active_lock:
                self._active_count += 1
            result = None
            try:
                result = fn(*args, **kwargs)
                if on_success is not None:
                    try:
                        on_success(result)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_success callback failed for '%s'", task_name)
                return result
            except Exception as exc:  # noqa: BLE001
                logger.exception("Background task '%s' failed", task_name)
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:  # noqa: BLE001
                        logger.exception("on_error callback failed for '%s'", task_name)
                raise
            finally:
                with self._active_lock:
                    self._active_count -= 1
                logger.debug("Task '%s' finished (active=%d)", task_name, self.active_count)

        future = self._pool.submit(_runner)
        self._attach_timeout_watchdog(future, task_name)
        return future

    def _attach_timeout_watchdog(self, future: Future, task_name: str) -> None:
        """Log (not cancel — Python threads can't be force-killed safely)
        if a task runs past the configured soft timeout, so hangs are
        visible in logs instead of silently stalling the pool."""

        timeout = settings.executor.task_timeout_sec

        def _watch():
            try:
                future.result(timeout=timeout)
            except TimeoutError:
                logger.error(
                    "Task '%s' exceeded soft timeout of %.1fs (still running in background)",
                    task_name, timeout,
                )
            except Exception:
                # Already logged inside _runner; nothing more to do here.
                pass

        threading.Thread(target=_watch, name=f"srm-watchdog-{task_name}", daemon=True).start()

    def shutdown(self, wait: bool = True) -> None:
        logger.info("Shutting down async executor (wait=%s)", wait)
        self._shutdown = True
        self._pool.shutdown(wait=wait, cancel_futures=not wait)


# Process-wide singleton. Import this everywhere instead of constructing
# a new pool per module.
executor = AsyncExecutor()
