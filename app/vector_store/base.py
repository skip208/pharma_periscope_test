"""
Vector store interface and shared types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: List[float]


class VectorStore(Protocol):
    def clear(self) -> None:
        ...

    def upsert_documents(self, documents: List[DocumentChunk]) -> None:
        ...

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[DocumentChunk, float]]:
        ...


__all__ = ["DocumentChunk", "VectorStore"]

