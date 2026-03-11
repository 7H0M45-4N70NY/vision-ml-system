"""In-memory implementations of EventPublisher, EventSubscriber, and JobQueue.

Used for:
  - Unit testing (no external dependencies required)
  - Local development without Docker (fallback when Redis/Kafka unavailable)
  - Integration tests with deterministic event ordering

These implementations are API-identical to Redis/Kafka versions.
Swapping is a one-line change in the dependency injection container.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .base import (
    Event,
    EventPublisher,
    EventSubscriber,
    EventType,
    Job,
    JobQueue,
    JobState,
)

logger = logging.getLogger(__name__)


class InMemoryJobQueue(JobQueue):
    """Thread-safe in-memory job queue for testing and local development.

    Stores jobs in a plain dict.  No TTL, no persistence, no horizontal
    scaling — but identical interface to RedisJobQueue.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._pending: List[str] = []

    def submit(self, job: Job) -> str:
        self._jobs[job.id] = job
        self._pending.append(job.id)
        logger.info(f"[InMemory] Job {job.id} submitted")
        return job.id

    def update_state(self, job_id: str, state: JobState, **kwargs) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        job.state = state
        if state == JobState.COMPLETED:
            job.completed_at = datetime.utcnow().isoformat()
        if "result" in kwargs:
            job.result = kwargs["result"]
        if "error" in kwargs:
            job.error = str(kwargs["error"])

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def complete(self, job_id: str, result: Dict[str, Any]) -> None:
        self.update_state(job_id, JobState.COMPLETED, result=result)
        logger.info(f"[InMemory] Job {job_id} completed")

    def fail(self, job_id: str, error: str) -> None:
        self.update_state(job_id, JobState.FAILED, error=error)
        logger.warning(f"[InMemory] Job {job_id} failed: {error}")

    def dequeue(self) -> Optional[str]:
        """Pop next pending job ID (non-blocking)."""
        if self._pending:
            return self._pending.pop(0)
        return None


class InMemoryEventPublisher(EventPublisher):
    """Stores published events in a list for inspection in tests."""

    def __init__(self):
        self.events: List[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)
        logger.info(f"[InMemory] Published {event.event_type.value}")

    def close(self) -> None:
        pass


class InMemoryEventSubscriber(EventSubscriber):
    """Synchronously dispatches events to handlers (no background thread).

    Call ``dispatch(event)`` directly in tests to simulate event arrival.
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        self._handlers[event_type].append(handler)

    def start(self) -> None:
        pass  # no-op for in-memory

    def stop(self) -> None:
        pass  # no-op for in-memory

    def dispatch(self, event: Event) -> None:
        """Manually dispatch an event (for testing)."""
        for handler in self._handlers.get(event.event_type, []):
            handler(event)
