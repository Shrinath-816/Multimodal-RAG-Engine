"""
mmrag/rag/chunking.py
========================

Heading- and paragraph-aware text chunking with word-based overlap (the
Week 2 chunking principles, dependency-free). Chunks never straddle a
markdown heading, so each chunk carries ONE section label — useful for
citations ("returns_policy.md > Damaged items").
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text and the markdown section it came from (None if no heading seen)."""

    text: str
    section: str | None


def chunk_markdown(text: str, max_words: int = 120, overlap_words: int = 20) -> list[TextChunk]:
    """
    Split markdown/plain text into chunks of at most `max_words` words.

    Args:
        text: Source document.
        max_words: Maximum words per chunk.
        overlap_words: Words repeated from the end of one chunk at the start of
            the next WITHIN a section (never across a heading boundary).

    Returns:
        Chunks in document order. Empty input yields an empty list.

    Raises:
        ValueError: If overlap_words >= max_words (the loop could never advance).
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words.")

    chunks: list[TextChunk] = []
    section: str | None = None
    buffer: list[str] = []
    new_words = 0  # words added since the last flush (excludes carried-over overlap)

    def flush(carry_overlap: bool) -> None:
        nonlocal buffer, new_words
        if new_words > 0:
            chunks.append(TextChunk(text=" ".join(buffer), section=section))
        buffer = buffer[-overlap_words:] if (carry_overlap and overlap_words > 0 and new_words > 0) else []
        new_words = 0

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if paragraph.startswith("#"):
            lines = paragraph.splitlines()
            flush(carry_overlap=False)
            buffer = []
            section = lines[0].lstrip("#").strip() or section
            paragraph = "\n".join(lines[1:]).strip()
            if not paragraph:
                continue

        words = paragraph.split()
        # Keep a paragraph whole when it fits in a fresh chunk.
        if buffer and len(buffer) + len(words) > max_words and len(words) <= max_words:
            flush(carry_overlap=True)

        index = 0
        while index < len(words):
            space = max_words - len(buffer)
            if space <= 0:
                flush(carry_overlap=True)
                continue
            taken = words[index : index + space]
            buffer.extend(taken)
            new_words += len(taken)
            index += len(taken)
            if len(buffer) >= max_words and index < len(words):
                flush(carry_overlap=True)

    flush(carry_overlap=False)
    return chunks
