"""
CLI для поиска по векторному индексу по текстовому запросу.

Пример:
    python -m scripts.search_query --query "Горлум исчез; Фродо..." --top-k 5
"""

from __future__ import annotations

import argparse

from app.embeddings.client import EmbeddingsClient
from app.vector_store import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed chunks by text query.")
    parser.add_argument("--query", "-q", required=True, help="Текст запроса")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть")
    parser.add_argument("--snippet", type=int, default=300, help="Длина сниппета текста")
    args = parser.parse_args()

    vs = get_vector_store()
    emb = EmbeddingsClient()

    q_vec = emb.embed_text(args.query)
    results = vs.search(q_vec, top_k=args.top_k)

    if not results:
        print("Нет результатов")
        return

    for idx, (doc, distance) in enumerate(results, start=1):
        snippet = doc.text[: args.snippet].replace("\n", " ")
        print(f"\n#{idx} distance={distance:.4f} id={doc.id}")
        print("metadata:", doc.metadata)
        print("text:", snippet + ("..." if len(doc.text) > args.snippet else ""))


if __name__ == "__main__":
    main()

