# --- Vision ML System: Events Package ---
"""Event-driven coordination and background task management.

Exports core event types, dataclasses, and in-memory publisher/queue
implementations for use throughout the Vision ML system.
"""

from .base import (
    Event,
    EventType,
    Job,
    JobState,
    EventPublisher,
    EventSubscriber,
    JobQueue
)
from .in_memory import (
    InMemoryEventPublisher,
    InMemoryJobQueue
)

__all__ = [
    'Event',
    'EventType',
    'Job',
    'JobState',
    'EventPublisher',
    'EventSubscriber',
    'JobQueue',
    'InMemoryEventPublisher',
    'InMemoryJobQueue'
]
