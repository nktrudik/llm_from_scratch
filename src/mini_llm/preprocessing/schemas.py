"""Конфигурация и входные схемы preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PreprocessingError(RuntimeError):
    """Ошибка схемы исходного треда или чтения raw-данных."""


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Пути и консервативные настройки фильтрации preprocessing."""

    input_dir: Path = Path("data/raw/2ch")
    output_dir: Path = Path("data/processed")
    review_size: int = 100
    random_seed: int = 42
    dedup_min_characters: int = 80
    near_duplicate_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.review_size < 0:
            raise ValueError("review_size не может быть отрицательным")
        if self.dedup_min_characters <= 0:
            raise ValueError("dedup_min_characters должен быть положительным")
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold должен находиться в диапазоне (0, 1]")


@dataclass(frozen=True, slots=True)
class RawPost:
    """Поля поста, необходимые для восстановления графа ответов."""

    post_id: int
    text: str
    references: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawThread:
    """Проверенное неизменяемое представление raw-треда."""

    board: str
    thread_id: int
    source_path: str
    posts: tuple[RawPost, ...]
