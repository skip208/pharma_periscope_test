"""
Indexing pipeline: load corpus, chunk, embed, and upsert into vector store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

from tqdm import tqdm

from app.config import settings
from app.embeddings.client import EmbeddingsClient
from app.indexing.chunker import chunk_chapter_text
from app.indexing.parser import parse_books
from app.vector_store.base import DocumentChunk, VectorStore

logger = logging.getLogger(__name__)


def reindex_corpus(vector_store: VectorStore, embeddings_client: EmbeddingsClient, embed_batch: int = 64) -> int:
    started = time.time()
    vector_store.clear()

    books = parse_books()
    total_books = len(books)
    total_chapters = 0
    all_chunks: List[DocumentChunk] = []

    for book in books:
        book_id = book["book_id"]
        book_title = book["book"]
        source_file = book["source_file"]
        for chapter in book["chapters"]:
            total_chapters += 1
            chapter_index = chapter["chapter_index"]
            chapter_title = chapter["chapter_title"]
            chunks = chunk_chapter_text(
                chapter["text"],
                chapter_index,
                {
                    "book": book_title,
                    "book_id": book_id,
                    "chapter_title": chapter_title,
                    "book_part": chapter.get("book_part"),
                    "source_file": source_file,
                },
            )
            all_chunks.extend(chunks)

    total_chunks = len(all_chunks)
    logger.info(
        "Parsed corpus",
        extra={"books": total_books, "chapters": total_chapters, "chunks": total_chunks},
    )

    # Embed and upsert in batches with progress bar
    for i in tqdm(range(0, total_chunks, embed_batch), desc="Indexing", unit="chunks"):
        batch = all_chunks[i : i + embed_batch]
        texts = [c.text for c in batch]
        embeddings = embeddings_client.embed_texts(texts)
        for c, emb in zip(batch, embeddings):
            c.embedding = emb
        vector_store.upsert_documents(batch)
        logger.info("Upserted batch", extra={"count": len(batch), "offset": i})

    elapsed = time.time() - started
    logger.info(
        "Reindex completed",
        extra={"chunks_indexed": total_chunks, "elapsed_sec": round(elapsed, 2)},
    )
    return total_chunks


@dataclass
class ReindexSummary:
    indexed_chunks: int
    elapsed_sec: float


class ReindexService:
    """Сервисный класс для переиндексации корпуса."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings_client: EmbeddingsClient,
        embed_batch: int = 64,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings_client = embeddings_client
        self.embed_batch = embed_batch
        self.logger = logger_ or logging.getLogger(__name__)

    def run(self) -> ReindexSummary:
        started = time.time()
        indexed = reindex_corpus(self.vector_store, self.embeddings_client, embed_batch=self.embed_batch)
        elapsed = time.time() - started
        self.logger.info(
            "ReindexService completed",
            extra={"indexed_chunks": indexed, "elapsed_sec": round(elapsed, 2)},
        )
        return ReindexSummary(indexed_chunks=indexed, elapsed_sec=elapsed)


__all__ = ["reindex_corpus", "ReindexService", "ReindexSummary"]

