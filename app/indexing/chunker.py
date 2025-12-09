"""
Text chunking utilities.
"""

from __future__ import annotations

from typing import Dict, List

from app.config import settings
from app.vector_store.base import DocumentChunk

CHUNK_SIZE_CHARS = settings.chunk_size_chars
CHUNK_OVERLAP_CHARS = settings.chunk_overlap_chars


def chunk_chapter_text(text: str, chapter_index: int, book_info: Dict[str, str]) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    step = max(1, CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS)
    chunk_index = 0

    for start in range(0, len(text), step):
        end = min(start + CHUNK_SIZE_CHARS, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{book_info['book_id']}_ch{chapter_index}_{chunk_index:04d}"
        metadata = {
            "book": book_info["book"],
            "book_id": book_info["book_id"],
            "book_part": book_info.get("book_part"),
            "chapter_title": book_info["chapter_title"],
            "chapter_index": chapter_index,
            "chunk_index": chunk_index,
            "position": start,
            "source_file": book_info["source_file"],
        }
        chunks.append(DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata, embedding=[]))
        chunk_index += 1
        if end >= len(text):
            break

    return chunks


__all__ = ["chunk_chapter_text", "CHUNK_SIZE_CHARS", "CHUNK_OVERLAP_CHARS"]

