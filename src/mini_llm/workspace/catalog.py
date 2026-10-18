"""Явная связь версии датасета, модели и отдельного запуска обучения."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mini_llm.data.dialogue import write_json
from mini_llm.workspace.paths import DATASETS_ROOT, RUNS_ROOT


def identifier(value: str) -> str:
    """Принимать только безопасные идентификаторы, а не пути."""

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,159}", value) or ".." in value:
        raise ValueError("Ожидался идентификатор без пути")
    if value.endswith("."):
        raise ValueError("Идентификатор не должен оканчиваться точкой")
    return value


def contained(path: Path, root: Path) -> Path:
    """Не разрешать выход из хранилища, в том числе через символьные ссылки."""

    if not path.resolve().is_relative_to(root.resolve()) or path.resolve() == root.resolve():
        raise ValueError(f"Путь должен находиться внутри {root}")
    return path


class DatasetVersion(BaseModel):
    """Опубликованная версия: samples неизменяемы, splits создаются отдельно."""

    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    version: str = Field(pattern=r"^v[1-9][0-9]*$")
    title: str
    description: str = ""
    samples: int = Field(ge=1)
    sha256: str
    source: str = ""

    @property
    def directory(self) -> Path:
        return contained(
            DATASETS_ROOT / identifier(self.dataset_id) / identifier(self.version), DATASETS_ROOT
        )

    @property
    def splits_ready(self) -> bool:
        return all(
            (self.directory / "splits" / f"{split}.jsonl").is_file()
            for split in ("train", "validation", "test")
        )

    def save(self) -> None:
        """Сохранить описание версии рядом с samples.jsonl."""

        write_json(self.directory / "dataset.json", self.model_dump())


class CheckpointMetadata(BaseModel):
    """Лёгкий индекс checkpoint: каталог UI не читает torch weights."""

    global_step: int
    validation_loss: float | None = None
    modified_ns: int
    size: int


class ModelRun(BaseModel):
    """Manifest запуска хранится отдельно от неизменяемого training checkpoint."""

    model_config = ConfigDict(extra="forbid")
    run_id: str
    model_id: str
    dataset_id: str
    dataset_version: str
    directory: Path
    title: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: Literal["imported", "running", "completed", "interrupted", "failed"] = "imported"
    pretrained_config_file: Path | None = None
    tokenizer_file: Path | None = None
    training_config: dict[str, object] = Field(default_factory=dict)
    source: str | None = None
    parent_run_id: str | None = None
    job_id: str | None = None
    global_step: int = 0
    planned_total_steps: int | None = None
    error: str | None = None
    checkpoints: dict[str, CheckpointMetadata] = Field(default_factory=dict)

    def save(self) -> None:
        """Атомарно сохранить метаданные, не затрагивая веса."""

        identifier(self.run_id)
        contained(self.directory, RUNS_ROOT)
        write_json(self.directory / "run.json", self.model_dump(mode="json"))


def list_datasets() -> list[DatasetVersion]:
    """Прочитать только manifest-файлы, без полного сканирования samples."""

    result = []
    for path in sorted(DATASETS_ROOT.glob("*/v*/dataset.json")):
        version = DatasetVersion.model_validate_json(path.read_text(encoding="utf-8"))
        if version.directory.resolve() != path.parent.resolve():
            raise ValueError(f"Manifest датасета не соответствует каталогу: {path}")
        result.append(version)
    return sorted(result, key=lambda item: (item.dataset_id, int(item.version[1:])))


def get_dataset(dataset_id: str, version: str) -> DatasetVersion:
    """Найти именно запрошенную версию, без fallback на другой корпус."""

    directory = contained(
        DATASETS_ROOT / identifier(dataset_id) / identifier(version), DATASETS_ROOT
    )
    path = directory / "dataset.json"
    if not path.is_file():
        raise ValueError("Версия датасета не найдена")
    record = DatasetVersion.model_validate_json(path.read_text(encoding="utf-8"))
    if record.dataset_id != dataset_id or record.version != version:
        raise ValueError("Manifest не соответствует версии датасета")
    return record


def list_runs() -> list[ModelRun]:
    """Читать реестр запусков без десериализации model weights."""

    records = []
    for path in RUNS_ROOT.glob("*/*/*/run-*/run.json"):
        contained(path, RUNS_ROOT)
        record = ModelRun.model_validate_json(path.read_text(encoding="utf-8"))
        if record.directory.resolve() != path.parent.resolve():
            raise ValueError(f"Manifest запуска не соответствует каталогу: {path}")
        records.append(record)
    return sorted(records, key=lambda item: (item.created_at, item.run_id), reverse=True)


def get_run(run_id: str) -> ModelRun | None:
    """Разрешить серверный ID в проверенный каталог запуска."""

    identifier(run_id)
    return next((record for record in list_runs() if record.run_id == run_id), None)


def fingerprint(path: Path) -> str:
    """Получить SHA256 для контроля неизменности корпуса."""

    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def reserve_run(model_slug: str, model_id: str, dataset: DatasetVersion) -> ModelRun:
    """Зарезервировать новый номер атомарным mkdir; существующий запуск не перезаписывается."""

    base = contained(
        RUNS_ROOT / identifier(model_slug) / dataset.dataset_id / dataset.version, RUNS_ROOT
    )
    base.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10000):
        name = f"run-{number:03d}"
        directory = base / name
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return ModelRun(
            run_id=f"{model_slug}--{dataset.dataset_id}-{dataset.version}--{name}",
            model_id=model_id,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            directory=directory,
            title=f"{model_id} · {dataset.dataset_id}/{dataset.version} — {dataset.title} · {name}",
        )
    raise ValueError("Закончились доступные номера запусков")


def update_progress(
    directory: Path,
    step: int,
    total: int | None = None,
    *,
    checkpoint_name: str | None = None,
    validation_loss: float | None = None,
) -> None:
    """Обновить только manifest; ошибка отображения не отменяет сохранённый checkpoint."""

    path = directory / "run.json"
    if not path.is_file():
        return
    try:
        run = ModelRun.model_validate_json(path.read_text(encoding="utf-8"))
        run.global_step = step
        if total is not None:
            run.planned_total_steps = total
        if checkpoint_name is not None:
            checkpoint = contained(directory / checkpoint_name, directory)
            stat = checkpoint.stat()
            run.checkpoints[checkpoint_name] = CheckpointMetadata(
                global_step=step,
                validation_loss=(
                    validation_loss
                    if validation_loss is not None and math.isfinite(validation_loss)
                    else None
                ),
                modified_ns=stat.st_mtime_ns,
                size=stat.st_size,
            )
        run.save()
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning("Не удалось обновить progress: %s", error)
