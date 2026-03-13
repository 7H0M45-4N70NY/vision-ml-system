"""Central logging configuration module.

This module provides a centralized logging setup following SOLID principles.
All modules in the codebase use get_logger() to obtain a configured logger instance.

Configuration is centralized here, making it easy to:
- Change log levels globally
- Add/remove handlers (console, file, cloud)
- Modify log format
- Switch backends without changing client code

Usage:
    from src.vision_ml.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")
    logger.warning("Resource low")
    logger.error("Failed to load model")
"""

import logging
import sys
import os
from typing import Optional


class LoggerConfig:
    """Centralized logging configuration."""

    # Global log level (can be overridden per handler)
    LEVEL = logging.INFO

    # Log format: timestamp | level | module | message
    FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Console output
    CONSOLE_ENABLED = True
    CONSOLE_LEVEL = logging.INFO

    # File logging (optional)
    FILE_ENABLED = False
    FILE_PATH = "logs/vision_ml.log"
    FILE_LEVEL = logging.DEBUG

    @classmethod
    def enable_file_logging(cls, log_path: str = "logs/vision_ml.log") -> None:
        """Enable file logging."""
        cls.FILE_ENABLED = True
        cls.FILE_PATH = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @classmethod
    def disable_file_logging(cls) -> None:
        """Disable file logging."""
        cls.FILE_ENABLED = False

    @classmethod
    def set_level(cls, level: int) -> None:
        """Set global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
        cls.LEVEL = level


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Module name, typically __name__

    Returns:
        Configured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Starting inference")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(LoggerConfig.LEVEL)
        # Prevent propagation to root logger (avoids duplicate output)
        logger.propagate = False

        # Console handler
        if LoggerConfig.CONSOLE_ENABLED:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(LoggerConfig.CONSOLE_LEVEL)
            formatter = logging.Formatter(
                LoggerConfig.FORMAT,
                datefmt=LoggerConfig.DATE_FORMAT
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler (optional)
        if LoggerConfig.FILE_ENABLED:
            try:
                file_handler = logging.FileHandler(LoggerConfig.FILE_PATH)
                file_handler.setLevel(LoggerConfig.FILE_LEVEL)
                formatter = logging.Formatter(
                    LoggerConfig.FORMAT,
                    datefmt=LoggerConfig.DATE_FORMAT
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except (IOError, OSError) as e:
                console_logger = logging.getLogger(__name__)
                console_logger.warning(f"Failed to setup file logging: {e}")

    return logger
