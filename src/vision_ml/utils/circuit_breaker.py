"""Reusable circuit breaker for protecting pipeline stages from cascading failures."""

from enum import Enum
from typing import Any, Callable
from ..logging import get_logger

logger = get_logger(__name__)

_LOG_INTERVAL = 30  # frames between repeated warnings for the same breaker


class BreakerState(Enum):
    CLOSED    = "closed"     # normal — calls pass through
    OPEN      = "open"       # tripped — calls short-circuit
    HALF_OPEN = "half_open"  # one trial call to test recovery


class CircuitBreaker:
    """Wraps a callable and protects against repeated failures.

    Usage:
        cb = CircuitBreaker("detect", failure_threshold=3, recovery_frames=300)
        result = cb.call(self.detector.detect, frame, frame_idx=frame_idx, fallback=None)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_frames: int = 300,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_frames = recovery_frames
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_frame: int = 0
        self._last_log_frame: int = -_LOG_INTERVAL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        fn: Callable,
        *args: Any,
        frame_idx: int = 0,
        fallback: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Execute fn(*args, **kwargs) with circuit breaker protection.

        Returns fallback immediately when the circuit is OPEN.
        Resets on first successful call after HALF_OPEN.
        """
        if self._state == BreakerState.OPEN:
            if frame_idx - self._last_failure_frame >= self.recovery_frames:
                self._state = BreakerState.HALF_OPEN
                logger.info("Circuit '%s' → HALF_OPEN (testing recovery at frame %d)",
                            self.name, frame_idx)
            else:
                return fallback  # fast-fail

        try:
            result = fn(*args, **kwargs)
            if self._state == BreakerState.HALF_OPEN:
                self._reset(frame_idx)
            return result
        except Exception as exc:
            return self._on_failure(exc, frame_idx, fallback)

    @property
    def is_open(self) -> bool:
        return self._state == BreakerState.OPEN

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_failure(self, exc: Exception, frame_idx: int, fallback: Any) -> Any:
        self._consecutive_failures += 1
        self._last_failure_frame = frame_idx

        if self._consecutive_failures >= self.failure_threshold:
            if self._state != BreakerState.OPEN:
                logger.error(
                    "Circuit '%s' OPENED after %d consecutive failures. "
                    "Last error: %s: %s",
                    self.name, self._consecutive_failures, type(exc).__name__, exc,
                )
            self._state = BreakerState.OPEN
        else:
            if self._should_log(frame_idx):
                logger.warning(
                    "Circuit '%s' failure %d/%d at frame %d — %s: %s",
                    self.name, self._consecutive_failures, self.failure_threshold,
                    frame_idx, type(exc).__name__, exc,
                )

        return fallback

    def _reset(self, frame_idx: int) -> None:
        logger.info("Circuit '%s' → CLOSED (recovered at frame %d)", self.name, frame_idx)
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0

    def _should_log(self, frame_idx: int) -> bool:
        if frame_idx - self._last_log_frame >= _LOG_INTERVAL:
            self._last_log_frame = frame_idx
            return True
        return False
