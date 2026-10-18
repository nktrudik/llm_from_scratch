"""Каталог совместимых checkpoints экспериментов без загрузки модели в GPU."""

from __future__ import annotations

import logging
import math

# Нужен только тип UnpicklingError; pickle.load/loads не используются.
import pickle  # nosec B403
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import torch

from mini_llm.inference.checkpoints import resolve_named_checkpoint
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.workspace.catalog import ModelRun, contained, get_run, list_runs
from mini_llm.workspace.paths import RUNS_ROOT

EXPERIMENTS_ROOT = Path("checkpoints/runs")


@dataclass(frozen=True, slots=True)
class ExperimentCheckpoint:
    """Краткие сведения о checkpoint для выбора в UI."""

    name: str
    global_step: int | None
    validation_loss: float | None


@dataclass(frozen=True, slots=True)
class Experiment:
    """Один запуск обучения; обновлённые запуски показываются первыми."""

    experiment_id: str
    checkpoints: tuple[ExperimentCheckpoint, ...]
    modified_ns: int
    title: str = ""
    directory: str = ""


@dataclass(frozen=True, slots=True)
class _CheckpointSummary:
    config: PretrainedConfig | None
    global_step: int
    validation_loss: float | None


def experiment_directory(experiment_id: str) -> Path:
    """Разрешить серверный ID запуска, не позволяя выход за пределы хранилища."""

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
    run = get_run(experiment_id)
    if run is not None:
        return contained(run.directory, RUNS_ROOT)
    if directory.resolve().parent != EXPERIMENTS_ROOT.resolve():
        raise ValueError("Эксперимент должен находиться внутри checkpoints/runs")
    return directory


@lru_cache(maxsize=256)
def _read_summary(path: Path, modified_ns: int, size: int) -> _CheckpointSummary | None:
    """Кэшировать только метаданные по сигнатуре файла; mmap не копирует веса в RAM."""

    try:
        payload = cast(object, torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint должен содержать объект")
        metadata = payload.get("model_metadata")
        if isinstance(metadata, dict) and metadata.get("backend") == "pretrained":
            config: PretrainedConfig | None = PretrainedConfig.model_validate(
                metadata.get("pretrained_config")
            )
        elif isinstance(payload.get("model_config"), dict):
            config = None
        else:
            return None
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


def list_experiments(config: PretrainedConfig | None) -> tuple[Experiment, ...]:
    """Показать только checkpoints выбранной модели, revision и режима адаптации."""

    experiments: list[Experiment] = []
    entries: list[tuple[Path, str, str, PretrainedConfig | None]] = []
    managed: dict[str, ModelRun] = {}
    for run in list_runs():
        if run.model_id != (config.model_id if config else "custom"):
            continue
        expected = (
            PretrainedConfig.load(run.pretrained_config_file)
            if run.pretrained_config_file
            else None
        )
        entries.append((run.directory, run.run_id, run.title, expected))
        managed[run.run_id] = run
    # Совместимость с внешними одноуровневыми каталогами без manifest.
    for directory in EXPERIMENTS_ROOT.iterdir() if EXPERIMENTS_ROOT.is_dir() else ():
        if directory.is_dir() and any(directory.glob("*.pt")):
            entries.append((directory, directory.name, directory.name, config))
    for directory, run_id, title, expected in entries:
        if not directory.is_dir():
            continue
        checkpoints: list[ExperimentCheckpoint] = []
        modified_ns = 0
        try:
            if run_id in managed:
                contained(directory, RUNS_ROOT)
            else:
                experiment_directory(run_id)
        except ValueError:
            continue
        for path in directory.glob("*.pt"):
            try:
                path = resolve_named_checkpoint(directory, path.name)
                stat = path.stat()
                if not stat.st_size:
                    continue
                if run_id in managed:
                    run = managed[run_id]
                    saved = run.checkpoints.get(path.name)
                    match = re.fullmatch(r"step_([0-9]+)\.pt", path.name)
                    step = (
                        run.global_step
                        if path.name == "last.pt"
                        else int(match[1])
                        if match
                        else None
                    )
                    loss = None
                    if saved and (saved.modified_ns, saved.size) == (
                        stat.st_mtime_ns,
                        stat.st_size,
                    ):
                        step, loss = saved.global_step, saved.validation_loss
                    checkpoints.append(ExperimentCheckpoint(path.name, step, loss))
                else:
                    summary = _summary(path)
                    if summary is None or summary.config != expected:
                        continue
                    checkpoints.append(
                        ExperimentCheckpoint(
                            path.name, summary.global_step, summary.validation_loss
                        )
                    )
                modified_ns = max(modified_ns, stat.st_mtime_ns)
            except (OSError, RuntimeError, ValueError):
                continue
        if checkpoints:
            checkpoints.sort(
                key=lambda item: (
                    0 if item.name == "last.pt" else 1 if item.name == "best.pt" else 2,
                    -(item.global_step if item.global_step is not None else -1),
                    item.name,
                )
            )
            experiments.append(
                Experiment(run_id, tuple(checkpoints), modified_ns, title, directory.as_posix())
            )
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
