"""Единый формат сообщений и ограничение частоты terminal progress."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime


def format_elapsed(seconds: float) -> str:
    """Отформатировать длительность компактно и без потери длинных интервалов."""

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def terminal_log(category: str, message: str, *, elapsed: float | None = None) -> None:
    """Напечатать одно немедленно видимое сообщение с timestamp и категорией."""

    timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
    elapsed_suffix = "" if elapsed is None else f" | elapsed={format_elapsed(elapsed)}"
    print(f"[{timestamp}] [{category}] {message}{elapsed_suffix}", flush=True)


@contextmanager
def terminal_stage(category: str, description: str) -> Iterator[None]:
    """Логировать начало, успешное окончание или ошибку этапа."""

    started_at = time.perf_counter()
    terminal_log(category, f"Начало: {description}")
    try:
        yield
    except Exception:
        terminal_log(
            category,
            f"Ошибка: {description}",
            elapsed=time.perf_counter() - started_at,
        )
        raise
    terminal_log(
        category,
        f"Завершено: {description}",
        elapsed=time.perf_counter() - started_at,
    )


@dataclass(slots=True)
class ProgressThrottle:
    """Разрешать progress по числу элементов или прошедшему времени."""

    every_items: int
    every_seconds: float
    _last_items: int = 0
    _last_time: float = field(default_factory=time.perf_counter)

    def should_report(self, items: int, *, force: bool = False) -> bool:
        """Вернуть ``True`` для первого, периодического и финального сообщения."""

        now = time.perf_counter()
        due = (
            force
            or self._last_items == 0
            or items - self._last_items >= self.every_items
            or now - self._last_time >= self.every_seconds
        )
        if due:
            self._last_items = items
            self._last_time = now
        return due
