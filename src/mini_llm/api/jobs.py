"""Потокобезопасный реестр фоновых задач локального API."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Literal
from uuid import uuid4

JobStatus = Literal["queued", "running", "completed", "failed"]


class JobAlreadyRunningError(RuntimeError):
    """Новая тяжёлая задача отклонена, пока выполняется предыдущая."""


@dataclass(slots=True)
class JobRecord:
    """Текущее состояние одной фоновой операции."""

    job_id: str
    kind: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: object | None = None
    error: str | None = None
    operation_progress: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Вернуть независимый JSON-совместимый снимок состояния."""

        return asdict(self)


class JobManager:
    """Последовательно выполнять ресурсоёмкие pipeline в одном процессе API."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._generation_lock = Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._active_job_id: str | None = None

    def create(self, kind: str) -> JobRecord:
        """Создать queued-задачу, если другой pipeline сейчас не активен."""

        if not self._generation_lock.acquire(blocking=False):
            raise JobAlreadyRunningError("Сейчас выполняется генерация; дождитесь её завершения")
        try:
            with self._lock:
                if self._active_job_id is not None:
                    raise JobAlreadyRunningError(
                        f"Уже выполняется задача {self._active_job_id}; дождитесь её завершения"
                    )
                job_id = uuid4().hex
                record = JobRecord(job_id, kind, "queued", datetime.now(UTC).isoformat())
                self._jobs[job_id] = record
                self._active_job_id = job_id
                return record
        finally:
            self._generation_lock.release()

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Не разрешать удаление файлов между проверкой занятости и генерацией."""

        with self._generation_lock:
            if self.active_job_id() is not None:
                raise JobAlreadyRunningError(
                    "Генерация недоступна, пока выполняется фоновая задача"
                )
            yield

    def report_operation(self, job_id: str, progress: dict[str, object]) -> None:
        """Обновить лёгкий статус скачивания/очистки без весов и больших объектов."""

        with self._lock:
            self._jobs[job_id].operation_progress = dict(progress)

    def run(self, job_id: str, operation: Callable[[], object]) -> None:
        """Выполнить операцию и сохранить её результат или диагностическую ошибку."""

        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = datetime.now(UTC).isoformat()
        try:
            result = operation()
        except Exception as error:
            with self._lock:
                record.status = "failed"
                record.error = f"{type(error).__name__}: {error}"
        else:
            with self._lock:
                record.status = "completed"
                record.result = result
        finally:
            with self._lock:
                record.finished_at = datetime.now(UTC).isoformat()
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def get(self, job_id: str) -> dict[str, object] | None:
        """Вернуть снимок задачи или ``None`` для неизвестного идентификатора."""

        with self._lock:
            record = self._jobs.get(job_id)
            return None if record is None else record.to_dict()

    def active_job_id(self) -> str | None:
        """Вернуть идентификатор текущей тяжёлой задачи."""

        with self._lock:
            return self._active_job_id
