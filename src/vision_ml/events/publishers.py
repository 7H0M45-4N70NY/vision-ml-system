"""Runtime event publisher adapters for pipeline observability.

These publishers are best-effort and must never break core inference/training flows.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..logging import get_logger

logger = get_logger(__name__)


class PipelineEventPublisher(ABC):
    """Transport-agnostic publisher for pipeline lifecycle events."""

    @abstractmethod
    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        """Publish an event payload in a best-effort manner."""


class NoopPipelineEventPublisher(PipelineEventPublisher):
    """Default publisher that intentionally does nothing."""

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        return


class KafkaPipelineEventPublisher(PipelineEventPublisher):
    """Kafka-backed publisher used when explicitly enabled via config/env."""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self._producer = None

        try:
            from kafka import KafkaProducer  # type: ignore

            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except ImportError:
            logger.warning("kafka-python is not installed. Falling back to no-op publisher.")
        except Exception as exc:
            logger.warning("Failed to initialize Kafka producer: %s", exc)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        if self._producer is None:
            return

        message = {
            "event": event_name,
            "payload": payload,
        }

        try:
            self._producer.send(self.topic, message)
        except Exception as exc:
            logger.warning("Kafka publish failed for event '%s': %s", event_name, exc)


def get_pipeline_event_publisher(config: Optional[Dict[str, Any]] = None) -> PipelineEventPublisher:
    """Return publisher based on config/env flags.

    Config contract (all optional):
      events:
        backend: noop|kafka
        kafka_bootstrap_servers: localhost:9092
        kafka_topic: vision-ml-events
    """
    cfg = config or {}
    events_cfg = cfg.get("events", {})

    backend = (
        os.getenv("VISION_ML_EVENT_BACKEND")
        or events_cfg.get("backend")
        or "noop"
    ).lower()

    if backend != "kafka":
        return NoopPipelineEventPublisher()

    bootstrap = (
        os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        or events_cfg.get("kafka_bootstrap_servers")
        or "localhost:9092"
    )
    topic = (
        os.getenv("VISION_ML_EVENT_TOPIC")
        or events_cfg.get("kafka_topic")
        or "vision-ml-events"
    )

    return KafkaPipelineEventPublisher(bootstrap_servers=bootstrap, topic=topic)
