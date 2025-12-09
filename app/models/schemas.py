from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):
    """Запрос на переиндексацию корпуса."""

    mode: Literal["full"] = Field(default="full", description="Режим переиндексации")


class ReindexResponse(BaseModel):
    """Ответ об окончании переиндексации."""

    status: Literal["completed"] = Field(default="completed")
    indexed_chunks: int = Field(..., ge=0, description="Сколько чанков проиндексировано")
    elapsed_sec: float | None = Field(None, ge=0, description="Сколько секунд заняла операция")


__all__ = ["ReindexRequest", "ReindexResponse"]
