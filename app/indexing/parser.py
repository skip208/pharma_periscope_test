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
    # Каждая .txt содержит две книги (части); фиксируем стартовые индексы book_part
    "lotr_fellowship.txt": {
        "book": "The Fellowship of the Ring",
        "book_id": "fellowship",
        "book_part_start": 1,  # Книга 1 и Книга 2
    },
    "lotr_two_towers.txt": {
        "book": "The Two Towers",
        "book_id": "two_towers",
        "book_part_start": 3,  # Книга 3 и Книга 4
    },
    "lotr_return_of_king.txt": {
        "book": "The Return of the King",
        "book_id": "return_of_king",
        "book_part_start": 5,  # Книга 5 и Книга 6
    },
}

# Match:
# - optional "Книга <num>" prefix
# - "Глава" (case-insensitive)
# - number as арабские или римские
CHAPTER_PATTERN = re.compile(
    r"^(?:книга\s*\d+\s*)?глава\s+(?:\d+|[ivxlcdm]+)[^\n]*",
    flags=re.IGNORECASE | re.MULTILINE,
)

BOOK_PART_PATTERN = re.compile(r"книга\s*(\d+)", flags=re.IGNORECASE)
MIN_CHAPTER_CHARS = 500  # эвристика, чтобы отсечь оглавление
BOOK_PART_LOOKBACK = 400  # сколько символов смотреть назад для book_part
GAP_THRESHOLD_CHARS = 1000  # минимальный разрыв между заголовками, чтобы считать началом основного текста
CHAPTER_NUM_PATTERN = re.compile(r"глава\s+([ivxlcdm]+|\d+)", flags=re.IGNORECASE)


def _roman_to_int(value: str) -> int:
    mapping = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(value.lower()):
        curr = mapping.get(ch, 0)
        if curr < prev:
            total -= curr
        else:
            total += curr
        prev = curr
    return total


def _parse_chapter_number(title: str) -> int | None:
    m = CHAPTER_NUM_PATTERN.search(title)
    if not m:
        return None
    token = m.group(1)
    if token.isdigit():
        return int(token)
    return _roman_to_int(token)


def map_file_to_book_info(file_path: Path) -> Dict[str, str]:
    name = file_path.name
    info = BOOK_FILE_MAP.get(
        name,
        {"book": name.rsplit(".", 1)[0], "book_id": name.rsplit(".", 1)[0], "book_part_start": 1},
    )
    return {"book": info["book"], "book_id": info["book_id"], "book_part_start": info["book_part_start"], "source_file": name}


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


def split_into_chapters(book_text: str, base_part: int = 1) -> List[Dict[str, str]]:
    matches = list(CHAPTER_PATTERN.finditer(book_text))
    chapters: List[Dict[str, str]] = []

    if not matches:
        chapters.append({"chapter_index": 1, "chapter_title": "ГЛАВА 1", "text": clean_text(book_text)})
        return chapters

    def detect_book_part(segment: str, fallback: int) -> int:
        """Ищем последнее упоминание 'Книга <n>' в хвосте кусочка текста."""
        part = fallback
        tail = segment[-BOOK_PART_LOOKBACK:]
        for m in BOOK_PART_PATTERN.finditer(tail):
            try:
                part = int(m.group(1))
            except ValueError:
                continue
        return part

    # Track "book_part" (e.g., Книга 1, Книга 2) if present in chapter title or предшествующем блоке
    current_book_part = base_part
    prev_heading_start = 0
    prev_chunk_end = 0
    started = False
    last_chapter_num: int | None = None

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(book_text)
        title = match.group(0).strip()

        # Пропускаем оглавление: ищем первую главу, после которой большой текстовый блок
        gap = start - prev_heading_start
        if not started and gap < GAP_THRESHOLD_CHARS:
            prev_heading_start = start
            prev_chunk_end = end
            continue
        started = True

        # detect book part number if present в заголовке или в тексте перед ним
        current_book_part = detect_book_part(title, current_book_part)
        before_segment = book_text[prev_chunk_end:start]
        current_book_part = detect_book_part(before_segment, current_book_part)

        # Обновляем book_part по сбросу нумерации глав (после Book I -> Book II и т.д.)
        chapter_num = _parse_chapter_number(title)
        if last_chapter_num is not None and chapter_num is not None and chapter_num < last_chapter_num:
            current_book_part += 1
        if chapter_num is not None:
            last_chapter_num = chapter_num

        chunk_text = book_text[start:end]
        cleaned = clean_text(chunk_text)

        # Простейшая защита от оглавления: пропускаем слишком короткие куски до первой реальной главы
        if len(cleaned) < MIN_CHAPTER_CHARS and not chapters:
            prev_heading_start = start
            prev_chunk_end = end
            continue

        chapters.append(
            {
                "chapter_index": len(chapters) + 1,
                "chapter_title": title,
                "book_part": current_book_part,
                "text": cleaned,
            }
        )
        prev_heading_start = start
        prev_chunk_end = end

    return chapters


def parse_books(corpus_dir: str | Path = CORPUS_DIR) -> List[Dict[str, object]]:
    books_raw = load_book_files(corpus_dir)
    parsed: List[Dict[str, object]] = []
    for entry in books_raw:
        chapters = split_into_chapters(entry["text"], base_part=entry.get("book_part_start", 1))
        parsed.append(
            {
                "book": entry["book"],
                "book_id": entry["book_id"],
                "book_part_start": entry.get("book_part_start", 1),
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

