"""
CLI для вывода уникальных значений метадаты `book_part` из Chroma.

Пример:
    python -m scripts.list_book_parts --page-size 500
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Counter as CounterType, Tuple

from app.vector_store import get_vector_store


def collect_book_parts(page_size: int = 500) -> Tuple[CounterType[str], int]:
    """
    Собрать уникальные book_part и их частоту во всём индексе.
    """
    store = get_vector_store()
    collection = getattr(store, "collection", None)
    if collection is None:
        raise RuntimeError("Vector store backend не предоставляет raw collection (ожидается Chroma).")

    total = collection.count()
    counts: CounterType[str] = Counter()
    offset = 0

    while offset < total:
        result = collection.get(include=["metadatas"], limit=page_size, offset=offset)
        metas = result.get("metadatas", []) or []
        if not metas:
            break

        for meta in metas:
            if not meta:
                continue
            part = meta.get("book_part")
            if part is None:
                continue
            counts[str(part)] += 1

        offset += len(metas)

    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Показать все book_part, присутствующие в индексе.")
    parser.add_argument("--page-size", type=int, default=500, help="Размер страницы для обхода .get")
    args = parser.parse_args()

    counts, total = collect_book_parts(page_size=args.page_size)

    print(f"Всего документов в коллекции: {total}")
    if not counts:
        print("Поле book_part не найдено ни в одном документе.")
        return

    print("Найденные book_part -> количество документов:")
    for part in sorted(counts.keys(), key=lambda x: (len(x), x)):
        print(f"  {part}: {counts[part]}")


if __name__ == "__main__":
    main()
