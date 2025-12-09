"""
Chroma-based VectorStore implementation.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import chromadb

from app.config import settings
from app.vector_store.base import DocumentChunk, VectorStore

CHROMA_COLLECTION = "lotr_corpus"
CHROMA_PERSIST_DIR = settings.vector_store_path

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_directory: str | None = None, collection_name: str = CHROMA_COLLECTION) -> None:
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        logger.info(
            "ChromaVectorStore initialised",
            extra={"persist_directory": self.persist_directory, "collection": self.collection_name},
        )

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        logger.info("Chroma collection cleared and recreated", extra={"collection": self.collection_name})

    def upsert_documents(self, documents: List[DocumentChunk]) -> None:
        if not documents:
            return

        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        texts = [doc.text for doc in documents]

        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        logger.info("Upserted documents into Chroma", extra={"count": len(documents), "collection": self.collection_name})

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[DocumentChunk, float]]:
        if top_k <= 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0] or []
        texts = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        chunks: List[Tuple[DocumentChunk, float]] = []
        for doc_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            chunk = DocumentChunk(id=doc_id, text=text, metadata=metadata or {}, embedding=[])
            score = float(distance)
            chunks.append((chunk, score))

        return chunks


__all__ = ["ChromaVectorStore", "CHROMA_COLLECTION", "CHROMA_PERSIST_DIR"]

