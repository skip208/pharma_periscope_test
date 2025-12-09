"""
Простой smoke-тест RAG-пайплайна.

Пример:
    python -m scripts.rag_smoke --question "Кто такой Арагорн?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import setup_logging
from app.embeddings.client import EmbeddingsClient
from app.llm.client import LLMClient
from app.models.schemas import AskRequest
from app.rag.pipeline import RAGService
from app.vector_store import get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-тест RAG-пайплайна.")
    parser.add_argument("--question", "-q", required=True, help="Вопрос к корпусу LOTR")
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=None,
        help="Переопределить число чанков контекста",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    service = RAGService(
        vector_store=get_vector_store(),
        embeddings_client=EmbeddingsClient(),
        llm_client=LLMClient(),
        logger_=logger,
    )

    try:
        request = AskRequest(
            question=args.question,
            max_context_chunks=args.max_context_chunks,
        )
        response = service.answer_question(request)
    except Exception:
        logger.exception("RAG smoke failed")
        sys.exit(1)

    print("\n=== RAG Smoke Result ===")
    print(f"can_answer: {response.can_answer}")
    print(f"answer_short:\n{response.answer_short}")
    print("\nCitations:")
    for idx, c in enumerate(response.citations, start=1):
        print(
            f"#{idx} book={c.book} chapter={c.chapter_title} pos={c.position} chunk_id={c.chunk_id}"
        )
        if c.quote:
            print(f"   quote: {c.quote}")
    print("\nContext chunks used:", len(response.context_chunks))
    print("Raw scores:")
    if response.raw_scores:
        for s in response.raw_scores[:5]:
            print(f"  {s.chunk_id}: {s.score:.3f}")
    else:
        print("  <none>")


if __name__ == "__main__":
    main()

