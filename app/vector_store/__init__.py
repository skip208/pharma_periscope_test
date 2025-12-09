"""
Vector store abstractions and factories.
"""

from app.config import settings
from app.vector_store.chroma_store import ChromaVectorStore

DEFAULT_VECTOR_STORE_BACKEND = settings.vector_store_backend


def get_vector_store():
    """
    Factory to obtain configured VectorStore instance.
    Currently supports only Chroma backend.
    """
    backend = DEFAULT_VECTOR_STORE_BACKEND.lower()
    if backend == "chroma":
        return ChromaVectorStore()
    raise ValueError(f"Unsupported vector store backend: {backend}")


__all__ = ["DEFAULT_VECTOR_STORE_BACKEND", "get_vector_store", "ChromaVectorStore"]

