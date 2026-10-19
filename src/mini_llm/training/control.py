"""Кооперативная остановка без прерывания optimizer step и записи checkpoint."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

_stop_event: ContextVar[Event | None] = ContextVar("training_stop_event", default=None)


class TrainingCancelled(RuntimeError):
    """Подготовка отменена до начала optimizer steps; checkpoint ещё не существует."""


@contextmanager
def training_control(event: Event) -> Iterator[None]:
    """Передать сигнал остановки только текущей фоновой задаче."""

    token = _stop_event.set(event)
    try:
        yield
    finally:
        _stop_event.reset(token)


def training_stop_requested() -> bool:
    """Проверить сигнал, не изменяя процесс и другие задачи."""

    event = _stop_event.get()
    return event is not None and event.is_set()
