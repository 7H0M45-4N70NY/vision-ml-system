"""Reference implementation of non-blocking, zero-latency logging in Python.

Companion to: specs/logging_best_practice/SPEC.md
Provides a drop-in pattern using standard library `logging.handlers.QueueHandler`
and `logging.handlers.QueueListener`.
"""

import queue
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import atexit
import sys
from typing import Optional


class AsyncLoggingManager:
    """Manages an asynchronous, non-blocking logging pipeline.
    
    The main/inference thread only writes to an in-memory Queue (< 5 µs overhead).
    A separate background daemon thread handles formatting and writing to disk/console.
    """

    _queue: queue.Queue = queue.Queue(maxsize=10000)
    _listener: Optional[QueueListener] = None
    _initialized: bool = False

    @classmethod
    def initialize(
        cls,
        log_file: Optional[str] = None,
        console: bool = True,
        level: int = logging.INFO
    ) -> None:
        """Initialize the background listener thread and sinks."""
        if cls._initialized:
            return

        handlers = []

        # Console sink
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
            handlers.append(console_handler)

        # File sink (Rotating to prevent unbounded disk growth)
        if log_file:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB per file
                backupCount=5
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
            handlers.append(file_handler)

        # Start the background listener thread
        cls._listener = QueueListener(cls._queue, *handlers, respect_handler_level=True)
        cls._listener.start()
        cls._initialized = True

        # Ensure all queued logs are flushed on process exit
        atexit.register(cls.shutdown)

    @classmethod
    def get_handler(cls) -> QueueHandler:
        """Returns the non-blocking QueueHandler for loggers."""
        if not cls._initialized:
            cls.initialize()
        return QueueHandler(cls._queue)

    @classmethod
    def shutdown(cls) -> None:
        """Cleanly stops the background listener and flushes remaining records."""
        if cls._listener:
            cls._listener.stop()
            cls._listener = None
            cls._initialized = False


def get_async_logger(name: str) -> logging.Logger:
    """Convenience function to get a non-blocking logger for high-FPS loops."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(AsyncLoggingManager.get_handler())
    return logger
