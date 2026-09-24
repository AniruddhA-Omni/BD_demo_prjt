from __future__ import annotations

import re
from typing import Iterable

DEFAULT_MAX_CHARS = 1200

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_paragraphs(paragraphs: Iterable[str], max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Pack paragraphs into chunks of at most ``max_chars``, splitting only oversized paragraphs."""

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        for piece in _split_oversized(paragraph, max_chars):
            if current and current_len + len(piece) + 2 > max_chars:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces
