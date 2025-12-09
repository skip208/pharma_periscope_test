"""
Vector store abstractions.
"""

from app.config import settings

DEFAULT_VECTOR_STORE_BACKEND = settings.vector_store_backend
DEFAULT_VECTOR_STORE_PATH = settings.vector_store_path

__all__ = ["DEFAULT_VECTOR_STORE_BACKEND", "DEFAULT_VECTOR_STORE_PATH"]

