"""
OpenAI embeddings client.
"""

from __future__ import annotations

from typing import List, Sequence

from openai import OpenAI

from app.config import settings

DEFAULT_EMBEDDING_MODEL = settings.embedding_model_name
DEFAULT_EMBED_BATCH_SIZE = 64


class EmbeddingsClient:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self.client = client or OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []


__all__ = ["EmbeddingsClient", "DEFAULT_EMBEDDING_MODEL"]

