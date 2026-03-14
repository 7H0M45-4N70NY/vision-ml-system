# --- Vision ML System: Logging Package ---
"""Centralized logging configuration.

Usage inside packages:
    from ..logging import get_logger

Usage in scripts:
    from vision_ml.logging import get_logger
"""

from .logger import get_logger, LoggerConfig

__all__ = ['get_logger', 'LoggerConfig']
