"""Ленивая загрузка опциональных Hugging Face зависимостей."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


class PretrainedDependencyError(RuntimeError):
    """Не установлены библиотеки, необходимые для pretrained-режима."""


def require_module(name: str) -> ModuleType:
    """Импортировать optional dependency с понятной диагностикой."""

    try:
        return import_module(name)
    except ImportError as error:
        extra = "unsloth" if name == "unsloth" else "pretrained"
        raise PretrainedDependencyError(
            f"Установите optional dependencies: uv sync --locked --extra {extra}"
        ) from error
