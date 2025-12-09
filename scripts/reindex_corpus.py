"""
CLI для полной переиндексации корпуса.

Пример:
    python -m scripts.reindex_corpus --embed-batch 64
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import setup_logging
from app.embeddings.client import EmbeddingsClient
from app.indexing.pipeline import ReindexService
from app.vector_store import get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Полная переиндексация корпуса.")
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Размер батча для запроса эмбеддингов.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    service = ReindexService(
        get_vector_store(),
        EmbeddingsClient(),
        embed_batch=args.embed_batch,
        logger_=logger,
    )

    try:
        summary = service.run()
    except Exception:
        logger.exception("Reindex failed")
        sys.exit(1)

    print(f"Indexed chunks: {summary.indexed_chunks} (elapsed {summary.elapsed_sec:.2f}s)")


if __name__ == "__main__":
    main()

