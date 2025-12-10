"""
RAG pipeline: normalize question, retrieve context, guardrails, LLM answer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Sequence, Dict, Set

from app.config import settings
from app.embeddings.client import EmbeddingsClient
from app.llm.client import LLMClient
from app.models.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    ContextChunk,
    RetrievalScore,
)
from app.vector_store.base import DocumentChunk, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_REFUSAL = "В загруженных текстах недостаточно информации для точного ответа."


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    distance: float


class RAGService:
    """Сервисный класс RAG-пайплайна."""

    def __init__(
        self,
        vector_store: VectorStore,
        embeddings_client: EmbeddingsClient,
        llm_client: LLMClient,
        logger_: logging.Logger | None = None,
        request_id: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings_client = embeddings_client
        self.llm_client = llm_client
        self.logger = logger_ or logging.getLogger(__name__)
        self.request_id = request_id

    # --- Public API ---
    def answer_question(self, request: AskRequest) -> AskResponse:
        """Главная точка входа для ответа на вопрос."""
        normalized_question = self.normalize_question(request.question)
        retrievals = self.retrieve_relevant_chunks(
            normalized_question,
            max_candidates=(request.max_context_chunks or settings.max_context_chunks) * 2,
        )

        if self._should_refuse(retrievals):
            self.logger.info(
                "Guardrails refusal before LLM",
                extra={"reason": "low_relevance", "request_id": self.request_id},
            )
            return self._refusal_response()

        context_limit = request.max_context_chunks or settings.max_context_chunks
        context = self._select_context(retrievals, limit=context_limit)
        messages = self._build_messages(question=normalized_question, context=context)

        raw_answer = self.llm_client.chat(messages, response_format={"type": "json_object"})
        parsed_primary = self._parse_llm_response(raw_answer)

        if not parsed_primary:
            self.logger.warning("LLM response parse failed, fallback to refusal")
            return self._refusal_response()

        if not parsed_primary.get("can_answer", False):
            self.logger.info("LLM indicated refusal, fallback")
            return self._refusal_response()

        citations_primary = parsed_primary.get("sources") or []
        if not citations_primary:
            self.logger.info("Citations missing, fallback")
            return self._refusal_response()

        expanded_context = self._expand_context_with_neighbors(
            citations_primary, retrievals=retrievals, fallback=context
        )

        parsed_final = parsed_primary
        context_used = context

        if expanded_context != context:
            messages_expanded = self._build_messages(question=normalized_question, context=expanded_context)
            raw_expanded = self.llm_client.chat(messages_expanded, response_format={"type": "json_object"})
            parsed_expanded = self._parse_llm_response(raw_expanded)

            if parsed_expanded and parsed_expanded.get("can_answer", False):
                parsed_final = parsed_expanded
                context_used = expanded_context
            else:
                self.logger.info("Expanded pass failed or refused; using primary answer")

        citations = self._map_citations(parsed_final.get("sources") or [], context_used)
        if not citations:
            self.logger.info("Citations missing after mapping, fallback")
            return self._refusal_response()

        answer_short = parsed_final.get("answer_short") or DEFAULT_REFUSAL
        answer_full = parsed_final.get("answer_full") or answer_short

        return AskResponse(
            answer_short=answer_short,
            answer_full=answer_full,
            can_answer=True,
            citations=citations,
            context_chunks=[
                ContextChunk(chunk_id=item.chunk.id, text=item.chunk.text, metadata=item.chunk.metadata)
                for item in context_used
            ],
            raw_scores=[
                RetrievalScore(chunk_id=item.chunk.id, score=item.score)
                for item in retrievals
            ],
        )

    # --- Steps ---
    @staticmethod
    def normalize_question(text: str) -> str:
        """Трим и схлопывание пробелов/переносов."""
        return " ".join(text.strip().split())

    def retrieve_relevant_chunks(self, question: str, max_candidates: int) -> List[RetrievedChunk]:
        embedding = self.embeddings_client.embed_text(question)
        raw_results = self.vector_store.search(embedding, top_k=max_candidates)
        processed: List[RetrievedChunk] = []

        for chunk, distance in raw_results:
            score = self._distance_to_score(distance)
            processed.append(RetrievedChunk(chunk=chunk, score=score, distance=distance))

        processed.sort(key=lambda x: x.score, reverse=True)
        self.logger.info(
            "Retrieved chunks",
            extra={
                "requested": max_candidates,
                "returned": len(processed),
                "top_score": round(processed[0].score, 3) if processed else None,
                "request_id": self.request_id,
                "results": [
                    {"chunk_id": r.chunk.id, "score": round(r.score, 3)}
                    for r in processed[: min(5, len(processed))]
                ],
            },
        )
        return processed

    def _should_refuse(self, results: Sequence[RetrievedChunk]) -> bool:
        if not results:
            return True
        threshold = settings.relevance_threshold
        good = [r for r in results if r.score >= threshold]
        max_score = max((r.score for r in results), default=0.0)
        if max_score < threshold:
            return True
        if len(good) < settings.min_good_chunks:
            return True
        return False

    @staticmethod
    def _select_context(results: Sequence[RetrievedChunk], limit: int) -> List[RetrievedChunk]:
        return list(results[:limit])

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        # Chroma возвращает дистанцию (меньше — лучше). Переводим в псевдо-сходство.
        try:
            return max(0.0, min(1.0, 1.0 - float(distance)))
        except Exception:
            return 0.0

    def _build_messages(self, question: str, context: Sequence[RetrievedChunk]) -> List[dict]:
        fragments: List[str] = []
        for idx, item in enumerate(context, start=1):
            meta = item.chunk.metadata
            fragments.append(
                "\n".join(
                    [
                        f"[Фрагмент {idx}]",
                        f"Книга: {meta.get('book')} | Глава: {meta.get('chapter_title')} | Позиция: {meta.get('position')}",
                        f"Текст:\n{item.chunk.text}",
                    ]
                )
            )

        system_message = {
            "role": "system",
            "content": (
                "Ты — помощник по книгам «Властелин колец». "
                "Отвечай строго по предоставленным фрагментам. "
                "Если данных недостаточно, честно отвечай: "
                f'"{DEFAULT_REFUSAL}". '
                "Не придумывай факты, ограничивайся 1500 символами для основного ответа. "
                "Используй понятный Markdown."
            ),
        }

        user_message = {
            "role": "user",
            "content": "\n\n".join(
                [
                    f'Вопрос пользователя: "{question}"',
                    "Ниже приведены фрагменты из корпуса. Используй только их.",
                    "\n\n".join(fragments),
                    "Требуемый формат ответа (JSON):",
                    json.dumps(
                        {
                            "answer_short": "Краткий ответ, не более 1500 символов.",
                            "answer_full": "Более развёрнутый ответ.",
                            "sources": [
                                {
                                    "book": "Название книги",
                                    "chapter": "Название или номер главы",
                                    "position": 123,
                                    "quote": "Короткая цитата 1-2 предложения.",
                                    "chunk_id": "chunk-id",
                                }
                            ],
                            "can_answer": True,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    f'Если информации недостаточно, верни JSON с can_answer=false и текстом "{DEFAULT_REFUSAL}" в обоих полях ответа.',
                ]
            ),
        }

        return [system_message, user_message]

    def _parse_llm_response(self, raw: str) -> dict | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _map_citations(self, sources: Sequence[dict], context: Sequence[RetrievedChunk]) -> List[Citation]:
        chunk_index = {c.chunk.id: c.chunk.metadata for c in context}
        citations: List[Citation] = []
        for src in sources:
            chunk_id = src.get("chunk_id")
            meta = chunk_index.get(chunk_id, {}) if chunk_id else {}
            citations.append(
                Citation(
                    book=src.get("book") or meta.get("book") or "Unknown",
                    book_id=meta.get("book_id"),
                    chapter_title=src.get("chapter") or meta.get("chapter_title"),
                    chapter_index=meta.get("chapter_index"),
                    position=src.get("position") or meta.get("position"),
                    quote=src.get("quote"),
                    chunk_id=chunk_id,
                )
            )
        return citations

    def _expand_context_with_neighbors(
        self,
        citations: Sequence[dict],
        retrievals: Sequence[RetrievedChunk],
        fallback: Sequence[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Для процитированных чанков добавляем соседние (предыдущий/следующий) в той же главе/книге,
        если они есть в выборке retrievals. Сохраняем порядок по убыванию релевантности.
        """
        if not citations:
            return list(fallback)

        id_lookup: Dict[str, RetrievedChunk] = {r.chunk.id: r for r in retrievals}
        key_lookup: Dict[tuple, RetrievedChunk] = {}
        for r in retrievals:
            meta = r.chunk.metadata
            key = (meta.get("book_id"), meta.get("chapter_index"), meta.get("chunk_index"))
            key_lookup[key] = r

        wanted: Set[str] = set()
        for src in citations:
            cid = src.get("chunk_id")
            base = id_lookup.get(cid)
            if not base:
                continue
            meta = base.chunk.metadata
            book_id = meta.get("book_id")
            chapter_index = meta.get("chapter_index")
            idx = meta.get("chunk_index")
            if book_id is None or chapter_index is None or idx is None:
                continue
            for delta in (-1, 0, 1):
                neighbor = key_lookup.get((book_id, chapter_index, idx + delta))
                if neighbor:
                    wanted.add(neighbor.chunk.id)

        if not wanted:
            return list(fallback)

        expanded: List[RetrievedChunk] = []
        seen: Set[str] = set()
        for r in retrievals:
            cid = r.chunk.id
            if cid in wanted and cid not in seen:
                expanded.append(r)
                seen.add(cid)

        return expanded or list(fallback)

    @staticmethod
    def _refusal_response() -> AskResponse:
        return AskResponse(
            answer_short=DEFAULT_REFUSAL,
            answer_full=DEFAULT_REFUSAL,
            can_answer=False,
            citations=[],
            context_chunks=[],
            raw_scores=[],
        )


__all__ = ["RAGService", "DEFAULT_REFUSAL", "RetrievedChunk"]
