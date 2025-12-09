from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, status

from app.config import settings
from app.embeddings.client import EmbeddingsClient
from app.indexing.pipeline import ReindexService
from app.llm.client import LLMClient
from app.models.schemas import AskRequest, AskResponse, ReindexRequest, ReindexResponse
from app.rag.pipeline import RAGService
from app.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


def _check_admin_token(x_admin_token: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin token is not configured",
        )
    if x_admin_token != settings.admin_token.get_secret_value():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@router.post("/admin/reindex", response_model=ReindexResponse, summary="Reindex corpus")
def admin_reindex(
    reindex_request: ReindexRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> ReindexResponse:
    _check_admin_token(x_admin_token)

    service = ReindexService(get_vector_store(), EmbeddingsClient())
    logger.info("Admin reindex requested", extra={"mode": reindex_request.mode})

    summary = service.run()
    response = ReindexResponse(
        status="completed",
        indexed_chunks=summary.indexed_chunks,
        elapsed_sec=round(summary.elapsed_sec, 2),
    )
    logger.info(
        "Admin reindex completed",
        extra={"indexed_chunks": response.indexed_chunks, "elapsed_sec": response.elapsed_sec},
    )
    return response


@router.post("/api/v1/ask", response_model=AskResponse, summary="Ask question about LOTR corpus")
def ask(request: AskRequest) -> AskResponse:
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question must not be empty")

    logger.info("Ask request", extra={"len": len(question)})
    service = RAGService(
        vector_store=get_vector_store(),
        embeddings_client=EmbeddingsClient(),
        llm_client=LLMClient(),
    )
    return service.answer_question(request)


__all__ = ["router"]
