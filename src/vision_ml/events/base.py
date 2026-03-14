# --- Vision ML System: Events Base ---
"""Fundamental event and job abstractions for the Vision ML system.

This module defines the core data structures and interfaces for the
event-driven architecture, enabling decoupling between inference,
training, and other system components.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable


class EventType(Enum):
    """Supported event types in the system."""
    INFERENCE_COMPLETE = auto()
    DRIFT_DETECTED = auto()
    TRAINING_TRIGGERED = auto()
    LABELING_COMPLETE = auto()


@dataclass
class Event:
    """A discrete system event containing a payload and metadata."""
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class JobState(Enum):
    """Lifecycle states for background tasks."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class Job:
    """Represents a background task or long-running operation."""
    job_id: str
    task: str
    state: JobState = JobState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None


class EventPublisher(ABC):
    """Interface for publishing events to the system."""

    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publishes an event to all interested subscribers."""
        pass


class EventSubscriber(ABC):
    """Interface for receiving system events."""

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        """Processes a received event."""
        pass


class JobQueue(ABC):
    """Interface for managing background task execution."""

    @abstractmethod
    def submit(self, job: Job) -> None:
        """Submits a new job to the queue."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> Optional[Job]:
        """Retrieves the current status of a job."""
        pass
