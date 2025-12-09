"""
Utility script to inspect indexed chunks without embeddings.

Usage:
    python -m scripts.inspect_index --limit 5 --offset 0
"""

from __future__ import annotations

import argparse
import json

from app.vector_store.chroma_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stored chunks in Chroma.")
    parser.add_argument("--limit", type=int, default=5, help="Number of documents to show")
    parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    args = parser.parse_args()

    store = ChromaVectorStore()
    collection = store.collection
    total = collection.count()

    # Chroma get uses "ids" or "where" filters; to page we fetch by slicing "limit" with offset
    # using "offset" parameter on .get
    result = collection.get(
        include=["documents", "metadatas"],
        limit=args.limit,
        offset=args.offset,
    )

    ids = result.get("ids", [])  # may be empty if not included by backend
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])

    print(f"Total documents in collection: {total}")
    print(f"Showing {len(ids)} documents (offset={args.offset}, limit={args.limit})")
    for idx, (doc_id, doc, meta) in enumerate(zip(ids or [""] * len(docs), docs, metas), start=1):
        print(f"\n#{idx}: {doc_id or '<no-id>'}")
        ordered_meta = meta or {}
        if ordered_meta:
            # Define a stable order for known fields
            order = [
                "book",
                "book_id",
                "chapter_title",
                "chapter_index",
                "chunk_index",
                "position",
                "source_file",
            ]
            ordered_meta = {k: ordered_meta.get(k) for k in order if k in ordered_meta} | {
                k: v for k, v in ordered_meta.items() if k not in order
            }
        print("Metadata:", json.dumps(ordered_meta, ensure_ascii=False))
        snippet = doc[:400].replace("\n", " ")
        print("Text:", snippet + ("..." if len(doc) > 400 else ""))


if __name__ == "__main__":
    main()

