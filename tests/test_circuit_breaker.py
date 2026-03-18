"""Unit tests for CircuitBreaker."""

import pytest
from src.vision_ml.utils.circuit_breaker import CircuitBreaker, BreakerState


def _ok():
    return "ok"


def _fail():
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# CLOSED → OPEN transition
# ---------------------------------------------------------------------------

def test_transitions_to_open_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_frames=300)
    assert cb._state == BreakerState.CLOSED

    for i in range(3):
        result = cb.call(_fail, frame_idx=i, fallback="fallback")
        assert result == "fallback"

    assert cb._state == BreakerState.OPEN
    assert cb.is_open


def test_returns_fallback_before_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_frames=300)
    result = cb.call(_fail, frame_idx=0, fallback="fb")
    assert result == "fb"
    assert cb._state == BreakerState.CLOSED  # not yet open


# ---------------------------------------------------------------------------
# OPEN fast-fail (fn must not be called)
# ---------------------------------------------------------------------------

def test_open_fast_fails_without_calling_fn():
    calls = []

    def spy():
        calls.append(1)
        raise RuntimeError("should not be called")

    cb = CircuitBreaker("test", failure_threshold=2, recovery_frames=300)
    cb.call(spy, frame_idx=0, fallback=None)
    cb.call(spy, frame_idx=1, fallback=None)
    assert cb.is_open

    calls.clear()
    result = cb.call(spy, frame_idx=2, fallback="safe")
    assert result == "safe"
    assert calls == []  # fn was never invoked


# ---------------------------------------------------------------------------
# OPEN → HALF_OPEN → CLOSED recovery
# ---------------------------------------------------------------------------

def test_recovery_path():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_frames=10)

    # Trip the breaker
    cb.call(_fail, frame_idx=0, fallback=None)
    cb.call(_fail, frame_idx=1, fallback=None)
    assert cb.is_open

    # Still within recovery window — fast-fail
    result = cb.call(_ok, frame_idx=5, fallback="fb")
    assert result == "fb"
    assert cb._state == BreakerState.OPEN

    # Past recovery window → HALF_OPEN trial succeeds → CLOSED
    result = cb.call(_ok, frame_idx=12, fallback="fb")
    assert result == "ok"
    assert cb._state == BreakerState.CLOSED
    assert not cb.is_open


def test_half_open_failure_reopens():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_frames=10)

    cb.call(_fail, frame_idx=0, fallback=None)
    cb.call(_fail, frame_idx=1, fallback=None)
    assert cb.is_open

    # Past window → HALF_OPEN, but trial fails → re-OPEN
    result = cb.call(_fail, frame_idx=12, fallback="fb")
    assert result == "fb"
    assert cb._state == BreakerState.OPEN


# ---------------------------------------------------------------------------
# Successful calls don't accumulate failure count
# ---------------------------------------------------------------------------

def test_success_resets_consecutive_failures():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_frames=300)
    cb.call(_fail, frame_idx=0, fallback=None)
    cb.call(_fail, frame_idx=1, fallback=None)
    assert cb._consecutive_failures == 2

    cb.call(_ok, frame_idx=2, fallback=None)
    assert cb._consecutive_failures == 2  # success doesn't reset when CLOSED (only HALF_OPEN does)
    assert cb._state == BreakerState.CLOSED


# ---------------------------------------------------------------------------
# kwargs are forwarded correctly
# ---------------------------------------------------------------------------

def test_kwargs_forwarded():
    def fn(x, y=0):
        return x + y

    cb = CircuitBreaker("test")
    result = cb.call(fn, 3, frame_idx=0, fallback=-1, y=4)
    assert result == 7
