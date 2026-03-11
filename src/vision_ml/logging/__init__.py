"""Logging module - Centralized logging configuration.

This module provides a single source of truth for all logging in the codebase.

Usage:
    from vision_ml.logging import get_logger, LoggerConfig

    # Get logger in any module
    logger = get_logger(__name__)
    logger.info("Message")

    # Configure globally (affects all loggers)
    LoggerConfig.set_level(logging.DEBUG)
    LoggerConfig.enable_file_logging("logs/app.log")
"""

from .logger import get_logger, LoggerConfig

__all__ = ['get_logger', 'LoggerConfig']
