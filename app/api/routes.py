from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, status

from app.config import settings
from app.embeddings.client import EmbeddingsClient
from app.indexing.pipeline import ReindexService
from app.models.schemas import ReindexRequest, ReindexResponse
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


__all__ = ["router"]
