"""
Book parser utilities: loading corpus files and splitting into chapters.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

from app.config import settings

CORPUS_DIR = settings.corpus_dir

BOOK_FILE_MAP: Dict[str, Dict[str, str]] = {
    "lotr_fellowship.txt": {"book": "The Fellowship of the Ring", "book_id": "fellowship"},
    "lotr_two_towers.txt": {"book": "The Two Towers", "book_id": "two_towers"},
    "lotr_return_of_king.txt": {"book": "The Return of the King", "book_id": "return_of_king"},
}

# Match:
# - optional "Книга <num>" prefix
# - "Глава" (case-insensitive)
# - number as арабские или римские
CHAPTER_PATTERN = re.compile(
    r"^(?:книга\s*\d+\s*)?глава\s+(?:\d+|[ivxlcdm]+)[^\n]*",
    flags=re.IGNORECASE | re.MULTILINE,
)


def map_file_to_book_info(file_path: Path) -> Dict[str, str]:
    name = file_path.name
    info = BOOK_FILE_MAP.get(name, {"book": name.rsplit(".", 1)[0], "book_id": name.rsplit(".", 1)[0]})
    return {"book": info["book"], "book_id": info["book_id"], "source_file": name}


def load_book_files(corpus_dir: str | Path = CORPUS_DIR) -> List[Dict[str, str]]:
    base = Path(corpus_dir)
    if not base.exists():
        return []

    books: List[Dict[str, str]] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        info = map_file_to_book_info(path)
        books.append({**info, "text": text})
    return books


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_chapters(book_text: str) -> List[Dict[str, str]]:
    matches = list(CHAPTER_PATTERN.finditer(book_text))
    chapters: List[Dict[str, str]] = []

    if not matches:
        chapters.append({"chapter_index": 1, "chapter_title": "ГЛАВА 1", "text": clean_text(book_text)})
        return chapters

    # Track "book_part" (e.g., Книга 1, Книга 2) if present in chapter title
    current_book_part = 1

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(book_text)
        title = match.group(0).strip()

        # detect book part number if present in title (e.g., "Книга 2 Глава III")
        part_match = re.search(r"книга\s*(\d+)", title, flags=re.IGNORECASE)
        if part_match:
            try:
                current_book_part = int(part_match.group(1))
            except ValueError:
                current_book_part = current_book_part

        chunk_text = book_text[start:end]
        chapters.append(
            {
                "chapter_index": idx + 1,
                "chapter_title": title,
                "book_part": current_book_part,
                "text": clean_text(chunk_text),
            }
        )

    return chapters


def parse_books(corpus_dir: str | Path = CORPUS_DIR) -> List[Dict[str, object]]:
    books_raw = load_book_files(corpus_dir)
    parsed: List[Dict[str, object]] = []
    for entry in books_raw:
        chapters = split_into_chapters(entry["text"])
        parsed.append(
            {
                "book": entry["book"],
                "book_id": entry["book_id"],
                "source_file": entry["source_file"],
                "chapters": chapters,
            }
        )
    return parsed


__all__ = [
    "map_file_to_book_info",
    "load_book_files",
    "clean_text",
    "split_into_chapters",
    "parse_books",
    "CORPUS_DIR",
    "BOOK_FILE_MAP",
]

