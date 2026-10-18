"""Лёгкие снимки обучения для UI; состояние optimizer и checkpoints не затрагиваются."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mini_llm.data.dialogue import write_json


class TrainingProgress(BaseModel):
    """Текущий шаг, метрики и приблизительная оценка оставшегося времени."""

    status: Literal["preparing", "running", "completed", "interrupted", "failed"] = "preparing"
    stage: str = "Подготовка обучения"
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    global_step: int = 0
    start_step: int = 0
    total_steps: int | None = None
    steps_per_epoch: int | None = None
    planned_epochs: int | None = None
    epoch: int = 0
    epoch_progress: float = 0.0
    samples_seen: int = 0
    tokens_seen: int = 0
    train_loss: float | None = None
    rolling_loss: float | None = None
    validation_loss: float | None = None
    learning_rate: float | None = None
    tokens_per_second: float | None = None
    recent_losses: list[float] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None


class TrainingProgressReporter:
    """Писать не чаще раза в секунду; смену этапа и завершение публиковать сразу."""

    def __init__(self, directory: Path) -> None:
        self.path = directory / "progress.json"
        self.snapshot = TrainingProgress()
        self._started = time.perf_counter()
        self._learning_started: float | None = None
        self._last_write = float("-inf")
        self._start_tokens = 0
        self.publish(force=True)

    def publish(self, *, force: bool = False) -> None:
        """Ошибка наблюдаемости не должна прерывать обучение."""

        now = time.perf_counter()
        if not force and now - self._last_write < 1.0:
            return
        self.snapshot.elapsed_seconds = now - self._started
        self.snapshot.updated_at = datetime.now(UTC).isoformat()
        completed = self.snapshot.global_step - self.snapshot.start_step
        total = self.snapshot.total_steps
        if self._learning_started is not None and completed >= 3 and total is not None:
            elapsed = max(now - self._learning_started, 1e-9)
            remaining = max(0, total - self.snapshot.global_step)
            self.snapshot.eta_seconds = elapsed / completed * remaining if remaining else None
            self.snapshot.tokens_per_second = (
                self.snapshot.tokens_seen - self._start_tokens
            ) / elapsed
        if self.snapshot.status == "completed":
            self.snapshot.eta_seconds = 0.0
        elif self.snapshot.status in {"failed", "interrupted"}:
            self.snapshot.eta_seconds = None
        for name in (
            "train_loss",
            "rolling_loss",
            "validation_loss",
            "learning_rate",
            "tokens_per_second",
            "eta_seconds",
        ):
            value = getattr(self.snapshot, name)
            if isinstance(value, float) and not math.isfinite(value):
                setattr(self.snapshot, name, None)
        self.snapshot.recent_losses = [
            value for value in self.snapshot.recent_losses if math.isfinite(value)
        ][-100:]
        try:
            write_json(self.path, self.snapshot.model_dump(mode="json"))
        except OSError as error:
            logging.getLogger(__name__).warning("Не удалось сохранить training progress: %s", error)
        self._last_write = now

    def start(self, *, step: int, tokens: int) -> None:
        """ETA учитывает только шаги текущего запуска, включая корректный resume."""

        self.snapshot.start_step = self.snapshot.global_step = step
        self.snapshot.tokens_seen = self._start_tokens = tokens
        self.snapshot.status = "running"
        self.snapshot.stage = "Обучение"
        self._learning_started = time.perf_counter()
        self.publish(force=True)

    def finish(self, *, interrupted: bool = False, error: str | None = None) -> None:
        """Зафиксировать финальное состояние без растущего после завершения таймера."""

        self.snapshot.error = error
        self.snapshot.status = (
            "failed" if error is not None else "interrupted" if interrupted else "completed"
        )
        self.snapshot.stage = (
            "Ошибка" if error is not None else "Прервано" if interrupted else "Обучение завершено"
        )
        self.publish(force=True)


_current: ContextVar[TrainingProgressReporter | None] = ContextVar(
    "training_progress", default=None
)


def current_training_progress() -> TrainingProgressReporter | None:
    """Получить наблюдатель только текущего training pipeline, а не глобальную модель."""

    return _current.get()


@contextmanager
def report_training(directory: Path) -> Iterator[TrainingProgressReporter]:
    """Ограничить наблюдатель одним запуском и сообщить UI об исключении."""

    reporter = TrainingProgressReporter(directory)
    token = _current.set(reporter)
    try:
        yield reporter
    except BaseException as error:
        reporter.finish(
            interrupted=isinstance(error, KeyboardInterrupt),
            error=None if isinstance(error, KeyboardInterrupt) else str(error),
        )
        raise
    finally:
        _current.reset(token)


@contextmanager
def report_stage(description: str) -> Iterator[None]:
    """Показать длительную операцию, затем вернуть предыдущий этап."""

    reporter = _current.get()
    previous = reporter.snapshot.stage if reporter else ""
    if reporter:
        reporter.snapshot.stage = description
        reporter.publish(force=True)
    try:
        yield
    finally:
        if reporter:
            reporter.snapshot.stage = previous
            reporter.publish(force=True)


def read_training_progress(directory: Path) -> TrainingProgress | None:
    """Читать атомарный снимок, не открывая веса, Dataset или tokenizer."""

    path = directory / "progress.json"
    if not path.is_file():
        return None
    try:
        return TrainingProgress.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
