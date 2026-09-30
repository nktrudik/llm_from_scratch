"""Конфигурация и схемы JSON-сборщика 2ch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TWO_CH_BASE_URL = "https://2ch.org"
TWO_CH_HOSTS = frozenset({"2ch.org", "www.2ch.org"})
USER_AGENT = (
    "mini-llm-data-collector/0.1 "
    "(educational sequential collector; +https://github.com/nktrudik/llm_from_scratch)"
)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ScraperError(RuntimeError):
    """Ожидаемая ошибка URL, HTTP-запроса или схемы JSON API."""


@dataclass(frozen=True, slots=True)
class ScraperConfig:
    """Сетевые настройки и каталог результатов одного запуска."""

    output_dir: Path = Path("data/raw")
    timeout_seconds: float = 15.0
    max_retries: int = 2
    backoff_factor: float = 1.0
    min_request_delay: float = 1.0
    max_request_delay: float = 2.5

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds должен быть положительным")
        if self.max_retries < 0:
            raise ValueError("max_retries не может быть отрицательным")
        if self.backoff_factor < 0:
            raise ValueError("backoff_factor не может быть отрицательным")
        if self.min_request_delay < 0 or self.max_request_delay < self.min_request_delay:
            raise ValueError("Некорректный диапазон задержки между запросами")


@dataclass(frozen=True, slots=True)
class TwoChTarget:
    """Разобранный URL доски или конкретного треда."""

    board: str
    thread_id: int | None = None


@dataclass(frozen=True, slots=True)
class PostRecord:
    """Текстовое представление одного непустого поста 2ch."""

    post_id: int
    text: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Вернуть JSON-совместимое представление поста."""

        return {
            "post_id": self.post_id,
            "text": self.text,
            "references": list(self.references),
        }


@dataclass(frozen=True, slots=True)
class ThreadDocument:
    """Текстовый тред, полученный через JSON API 2ch."""

    board: str
    thread_id: int
    source_url: str
    api_url: str
    title: str
    fetched_at: str
    posts: tuple[PostRecord, ...]

    def to_dict(self) -> dict[str, object]:
        """Вернуть JSON-совместимое представление треда."""

        return {
            "board": self.board,
            "thread_id": self.thread_id,
            "source_url": self.source_url,
            "api_url": self.api_url,
            "title": self.title,
            "fetched_at": self.fetched_at,
            "posts": [post.to_dict() for post in self.posts],
        }
