from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence-ending punctuation followed by space or newline,
        # but keep the punctuation with the sentence (use a lookbehind or re.split with capturing group)
        # Pattern: sentence boundaries are ". ", "! ", "? ", or ".\n"
        sentence_pattern = re.compile(r'(?<=[.!?])\s+|(?<=\.)\n')
        raw_sentences = sentence_pattern.split(text)

        # Filter out empty strings after splitting
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return [text.strip()] if text.strip() else []

        # Group sentences into chunks of at most max_sentences_per_chunk
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\\n\\n", "\\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        """Entry point: delegate to _split with the full separator list."""
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """
        Recursively split current_text using remaining_separators.

        Algorithm:
        1. If text fits within chunk_size, return it as-is.
        2. Try the first separator to split the text into pieces.
        3. Merge consecutive small pieces to maximise chunk size, then recurse
           on any piece that is still oversized using the next separator.
        4. If no separators remain (empty string separator), fall back to
           character-level slicing.
        """
        # Base case: text already fits
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # No separators left — character-level slice
        if not remaining_separators:
            results = []
            for start in range(0, len(current_text), self.chunk_size):
                results.append(current_text[start : start + self.chunk_size])
            return results

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Split on current separator (empty string handled below)
        if sep == "":
            # Same as no separator — character-level slice
            results = []
            for start in range(0, len(current_text), self.chunk_size):
                results.append(current_text[start : start + self.chunk_size])
            return results

        pieces = current_text.split(sep)

        # Merge pieces greedily so each merged segment is <= chunk_size,
        # then recurse on any segment that is still oversized.
        results: list[str] = []
        current_buffer: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            # +len(sep) for the separator that joins pieces
            join_len = len(sep) * (len(current_buffer)) + sum(len(p) for p in current_buffer) + piece_len

            if join_len <= self.chunk_size:
                current_buffer.append(piece)
                current_len = join_len
            else:
                # Flush current buffer
                if current_buffer:
                    merged = sep.join(current_buffer).strip()
                    if merged:
                        if len(merged) > self.chunk_size:
                            results.extend(self._split(merged, next_separators))
                        else:
                            results.append(merged)
                # Start fresh with this piece
                current_buffer = [piece]
                current_len = piece_len

        # Flush remaining buffer
        if current_buffer:
            merged = sep.join(current_buffer).strip()
            if merged:
                if len(merged) > self.chunk_size:
                    results.extend(self._split(merged, next_separators))
                else:
                    results.append(merged)

        return [r for r in results if r]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))

    # Zero-magnitude guard
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Run all three built-in strategies and return comparison statistics.

        Returns a dict with keys: 'fixed_size', 'by_sentences', 'recursive'.
        Each value is a dict with: 'count', 'avg_length', 'chunks'.
        """
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        results: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            results[name] = {
                "count": count,
                "avg_length": round(avg_length, 2),
                "chunks": chunks,
            }

        return results
