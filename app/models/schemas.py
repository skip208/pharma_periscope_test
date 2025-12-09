from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


# Admin
class ReindexRequest(BaseModel):
    """Запрос на переиндексацию корпуса."""

    mode: Literal["full"] = Field(default="full", description="Режим переиндексации")


class ReindexResponse(BaseModel):
    """Ответ об окончании переиндексации."""

    status: Literal["completed"] = Field(default="completed")
    indexed_chunks: int = Field(..., ge=0, description="Сколько чанков проиндексировано")
    elapsed_sec: float | None = Field(None, ge=0, description="Сколько секунд заняла операция")


# RAG
class AskRequest(BaseModel):
    """Запрос на ответ по корпусу."""

    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    mode: Literal["default", "short_only", "debug"] = Field(default="default")
    max_context_chunks: int | None = Field(
        default=None,
        gt=0,
        description="Переопределить количество чанков контекста",
    )


class Citation(BaseModel):
    book: str
    book_id: str | None = None
    chapter_title: str | None = None
    chapter_index: int | None = None
    position: int | None = None
    quote: str | None = None
    chunk_id: str | None = None


class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, object]


class RetrievalScore(BaseModel):
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    answer_short: str
    answer_full: str
    can_answer: bool
    citations: List[Citation]
    context_chunks: List[ContextChunk]
    raw_scores: List[RetrievalScore] | None = None


__all__ = [
    "ReindexRequest",
    "ReindexResponse",
    "AskRequest",
    "Citation",
    "ContextChunk",
    "AskResponse",
    "RetrievalScore",
]
