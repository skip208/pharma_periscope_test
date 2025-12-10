"""
Text chunking utilities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

MAX_PARAGRAPH_OVERLAP_CHARS = 500
MIN_CORE_CHARS = 800  # минимальный размер основы чанка без учета оверлапа

from app.config import settings
from app.vector_store.base import DocumentChunk

CHUNK_SIZE_CHARS = settings.chunk_size_chars
CHUNK_OVERLAP_CHARS = settings.chunk_overlap_chars


def _split_paragraphs(text: str) -> List[Tuple[str, int]]:
    """
    Разбить текст на абзацы, фиксируя стартовый индекс каждого абзаца.
    Разделитель — пустая строка (двойной перевод строки).
    Пустые абзацы отфильтровываются.
    """
    paragraphs: List[Tuple[str, int]] = []
    cursor = 0
    length = len(text)

    while cursor < length:
        sep = text.find("\n\n", cursor)
        if sep == -1:
            paragraph = text[cursor:]
            if paragraph.strip():
                paragraphs.append((paragraph, cursor))
            break

        paragraph = text[cursor:sep]
        if paragraph.strip():
            paragraphs.append((paragraph, cursor))
        cursor = sep + 2

    return paragraphs


def _truncate_overlap(paragraph: str) -> str:
    if len(paragraph) <= MAX_PARAGRAPH_OVERLAP_CHARS:
        return paragraph
    return paragraph[:MAX_PARAGRAPH_OVERLAP_CHARS]


def chunk_chapter_text(text: str, chapter_index: int, book_info: Dict[str, str]) -> List[DocumentChunk]:
    paragraphs = _split_paragraphs(text)
    chunks: List[DocumentChunk] = []
    chunk_index = 0
    idx = 0

    while idx < len(paragraphs):
        start_idx = idx
        core: List[Tuple[str, int]] = []
        core_len = 0

        while idx < len(paragraphs):
            para_text, _ = paragraphs[idx]
            separator_len = 2 if core else 0  # account for "\n\n" between paragraphs
            candidate_len = core_len + separator_len + len(para_text)
            must_extend = core_len < MIN_CORE_CHARS

            # Если уже набрали минимум, не превышаем целевой размер
            if not must_extend and core and candidate_len > CHUNK_SIZE_CHARS:
                break

            # Добавляем параграф даже если перепрыгнем лимит, чтобы достигнуть минимума
            core.append(paragraphs[idx])
            core_len = candidate_len
            idx += 1

            # Когда достигли минимума и приблизились к лимиту — выходим
            if core_len >= MIN_CORE_CHARS and core_len >= CHUNK_SIZE_CHARS:
                break

        prev_para = paragraphs[start_idx - 1][0] if start_idx > 0 else None
        next_para = paragraphs[idx][0] if idx < len(paragraphs) else None

        chunk_parts: List[str] = []
        if prev_para:
            chunk_parts.append(_truncate_overlap(prev_para))
        chunk_parts.append("\n\n".join(p[0] for p in core))
        if next_para:
            chunk_parts.append(_truncate_overlap(next_para))

        chunk_text = "\n\n".join(chunk_parts)
        position = core[0][1]
        chunk_id = f"{book_info['book_id']}_ch{chapter_index}_{chunk_index:04d}"
        metadata = {
            "book": book_info["book"],
            "book_id": book_info["book_id"],
            "book_part": book_info.get("book_part"),
            "chapter_title": book_info["chapter_title"],
            "chapter_index": chapter_index,
            "chunk_index": chunk_index,
            "position": position,
            "source_file": book_info["source_file"],
        }
        chunks.append(DocumentChunk(id=chunk_id, text=chunk_text, metadata=metadata, embedding=[]))
        chunk_index += 1

    return chunks


__all__ = ["chunk_chapter_text", "CHUNK_SIZE_CHARS", "CHUNK_OVERLAP_CHARS"]

