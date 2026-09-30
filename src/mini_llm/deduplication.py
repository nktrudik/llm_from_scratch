"""Консервативный поиск точных и почти точных копий обработанного текста."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

DeduplicationResult = Literal["unique", "exact_duplicate", "near_duplicate"]
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def canonicalize_for_deduplication(text: str) -> str:
    """Нормализовать поверхностные различия, не меняя сохраняемый исходный текст."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(WORD_PATTERN.findall(normalized))


def _stable_hash(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def _word_shingles(canonical_text: str) -> frozenset[int]:
    words = canonical_text.split()
    if len(words) < 3:
        return frozenset(_stable_hash(word) for word in words)
    return frozenset(
        _stable_hash(" ".join(words[index : index + 3])) for index in range(len(words) - 2)
    )


@dataclass(frozen=True, slots=True)
class _NearDuplicateEntry:
    canonical_text: str
    shingles: frozenset[int]


class ContentDeduplicator:
    """Находить реальные копии, не затрагивая обычные короткие ответы."""

    def __init__(self, *, minimum_characters: int = 80, similarity_threshold: float = 0.9) -> None:
        if minimum_characters <= 0:
            raise ValueError("minimum_characters must be positive")
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in the range (0, 1]")
        self.minimum_characters = minimum_characters
        self.similarity_threshold = similarity_threshold
        self._exact_texts: set[str] = set()
        self._entries: list[_NearDuplicateEntry] = []
        self._anchor_buckets: dict[int, list[int]] = {}

    def classify(self, text: str) -> DeduplicationResult:
        """Классифицировать и запомнить достаточно длинный уникальный текст."""

        canonical_text = canonicalize_for_deduplication(text)
        if len(canonical_text) < self.minimum_characters:
            return "unique"
        if canonical_text in self._exact_texts:
            return "exact_duplicate"

        shingles = _word_shingles(canonical_text)
        anchors = sorted(shingles)[:4]
        candidate_indices = {
            index for anchor in anchors for index in self._anchor_buckets.get(anchor, [])
        }
        for index in candidate_indices:
            candidate = self._entries[index]
            length_ratio = min(len(canonical_text), len(candidate.canonical_text)) / max(
                len(canonical_text), len(candidate.canonical_text)
            )
            if length_ratio < self.similarity_threshold:
                continue
            union_size = len(shingles | candidate.shingles)
            similarity = len(shingles & candidate.shingles) / union_size if union_size else 1.0
            if similarity >= self.similarity_threshold:
                return "near_duplicate"

        entry_index = len(self._entries)
        self._entries.append(_NearDuplicateEntry(canonical_text, shingles))
        self._exact_texts.add(canonical_text)
        for anchor in anchors:
            self._anchor_buckets.setdefault(anchor, []).append(entry_index)
        return "unique"
