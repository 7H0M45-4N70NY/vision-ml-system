# --- Vision ML System: In-Memory Events ---
"""Simple in-memory implementations of the event and job abstractions.

Suitable for local testing and lightweight coordination without external
dependencies like Kafka or Redis.
"""

from typing import List, Dict, Optional, Callable
from .base import Event, EventPublisher, EventSubscriber, Job, JobQueue, JobState


class InMemoryEventPublisher(EventPublisher):
    """Simple in-memory event publisher for decoupling system components."""

    def __init__(self):
        self._subscribers: Dict[str, List[EventSubscriber]] = {}

    def subscribe(self, event_type_name: str, subscriber: EventSubscriber) -> None:
        """Subscribes a component to a specific event type."""
        if event_type_name not in self._subscribers:
            self._subscribers[event_type_name] = []
        self._subscribers[event_type_name].append(subscriber)

    def publish(self, event: Event) -> None:
        """Delivers an event to all relevant subscribers."""
        event_type_name = event.event_type.name
        if event_type_name in self._subscribers:
            for subscriber in self._subscribers[event_type_name]:
                subscriber.handle_event(event)


class InMemoryJobQueue(JobQueue):
    """Synchronous in-memory job queue for lightweight task tracking."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def submit(self, job: Job) -> None:
        """Stores a job in the local dictionary."""
        self._jobs[job.job_id] = job

    def get_status(self, job_id: str) -> Optional[Job]:
        """Retrieves a job by ID from the local dictionary."""
        return self._jobs.get(job_id)

    def update_state(self, job_id: str, state: JobState, result: Optional[dict] = None) -> None:
        """Updates the state and result of a job in the queue."""
        if job_id in self._jobs:
            self._jobs[job_id].state = state
            if result is not None:
                self._jobs[job_id].result = result
