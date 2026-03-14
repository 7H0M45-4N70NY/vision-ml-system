# --- Vision ML System: API Package ---
"""FastAPI-based REST and WebSocket interface for the Vision ML system.

This package exposes the core predictive endpoints, analytics, and
configuration management for the production frontend and external consumers.
"""

from .main import app

__all__ = ['app']
