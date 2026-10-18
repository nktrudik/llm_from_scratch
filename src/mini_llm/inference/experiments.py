"""Каталог совместимых checkpoints экспериментов без загрузки модели в GPU."""

from __future__ import annotations

import logging
import math

# Нужен только тип UnpicklingError; pickle.load/loads не используются.
import pickle  # nosec B403
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import torch

from mini_llm.inference.checkpoints import resolve_named_checkpoint
from mini_llm.pretrained.config import PretrainedConfig

EXPERIMENTS_ROOT = Path("checkpoints/experiments")


@dataclass(frozen=True, slots=True)
class ExperimentCheckpoint:
    """Краткие сведения о checkpoint для выбора в UI."""

    name: str
    global_step: int
    validation_loss: float | None


@dataclass(frozen=True, slots=True)
class Experiment:
    """Один запуск обучения; обновлённые запуски показываются первыми."""

    experiment_id: str
    checkpoints: tuple[ExperimentCheckpoint, ...]
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _CheckpointSummary:
    config: PretrainedConfig
    global_step: int
    validation_loss: float | None


def experiment_directory(experiment_id: str) -> Path:
    """Разрешить только непосредственный подкаталог experiments, без обхода через ссылки."""

    if (
        not experiment_id
        or experiment_id in {".", ".."}
        or len(experiment_id) > 255
        or any(char in experiment_id for char in '<>:"/\\|?*')
        or any(ord(char) < 32 for char in experiment_id)
        or experiment_id.endswith((".", " "))
    ):
        raise ValueError("Укажите имя эксперимента без пути")
    directory = EXPERIMENTS_ROOT / experiment_id
    if directory.resolve().parent != EXPERIMENTS_ROOT.resolve():
        raise ValueError("Эксперимент должен находиться внутри checkpoints/experiments")
    return directory


@lru_cache(maxsize=256)
def _read_summary(path: Path, modified_ns: int, size: int) -> _CheckpointSummary | None:
    """Кэшировать только метаданные по сигнатуре файла; mmap не копирует веса в RAM."""

    try:
        payload = cast(object, torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint должен содержать объект")
        metadata = payload.get("model_metadata")
        if not isinstance(metadata, dict) or metadata.get("backend") != "pretrained":
            return None
        config = PretrainedConfig.model_validate(metadata.get("pretrained_config"))
        step = payload.get("global_step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Некорректный global_step")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict) or not state:
            raise ValueError("Checkpoint не содержит model_state_dict")
        loss = payload.get("last_validation_loss")
        validation_loss = (
            float(loss)
            if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(loss)
            else None
        )
        return _CheckpointSummary(config, step, validation_loss)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        logging.getLogger(__name__).warning("Checkpoint пропущен %s: %s", path, str(error))
        return None


def _summary(path: Path) -> _CheckpointSummary | None:
    stat = path.stat()
    return _read_summary(path.resolve(), stat.st_mtime_ns, stat.st_size)


def list_experiments(config: PretrainedConfig) -> tuple[Experiment, ...]:
    """Показать только checkpoints выбранной модели, revision и режима адаптации."""

    experiments: list[Experiment] = []
    for directory in EXPERIMENTS_ROOT.iterdir() if EXPERIMENTS_ROOT.is_dir() else ():
        if not directory.is_dir():
            continue
        checkpoints: list[ExperimentCheckpoint] = []
        modified_ns = 0
        try:
            experiment_directory(directory.name)
        except ValueError:
            continue
        for path in directory.glob("*.pt"):
            try:
                path = resolve_named_checkpoint(directory, path.name)
                summary = _summary(path)
                if summary is None or summary.config != config:
                    continue
                checkpoints.append(
                    ExperimentCheckpoint(path.name, summary.global_step, summary.validation_loss)
                )
                modified_ns = max(modified_ns, path.stat().st_mtime_ns)
            except (OSError, RuntimeError, ValueError):
                continue
        if checkpoints:
            checkpoints.sort(
                key=lambda item: (
                    0 if item.name == "last.pt" else 1 if item.name == "best.pt" else 2,
                    -item.global_step,
                    item.name,
                )
            )
            experiments.append(Experiment(directory.name, tuple(checkpoints), modified_ns))
    return tuple(sorted(experiments, key=lambda item: (-item.modified_ns, item.experiment_id)))


def resolve_experiment_checkpoint(
    experiment_id: str, checkpoint_name: str, config: PretrainedConfig
) -> Path:
    """Проверить путь и совместимость ещё раз непосредственно перед генерацией."""

    path = resolve_named_checkpoint(experiment_directory(experiment_id), checkpoint_name)
    summary = _summary(path)
    if summary is None or summary.config != config:
        raise ValueError("Checkpoint эксперимента не совместим с выбранной моделью и конфигом")
    return path
