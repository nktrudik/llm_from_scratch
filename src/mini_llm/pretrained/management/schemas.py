"""Контракты каталога, загрузки и подтверждённой очистки моделей."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_llm.pretrained.workspace import ModelPaths


class DownloadRequest(BaseModel):
    """Достаточно model ID; существующий QLoRA-профиль подставляется сервером."""

    model_config = ConfigDict(extra="forbid")
    model_id: str

    @model_validator(mode="after")
    def validate_model_id(self) -> Self:
        if self.model_id.casefold() == "custom":
            raise ValueError("custom зарезервирован для собственной модели; укажите HF owner/model")
        ModelPaths.for_model(self.model_id)
        return self


class OperationProgress(BaseModel):
    """Текущий этап операции; процент загрузки не выдумывается."""

    stage: str
    elapsed_seconds: float = 0.0


class DeletionTarget(BaseModel):
    """Только серверные ID, никогда не произвольные пути пользователя."""

    model_config = ConfigDict(extra="forbid")
    scope: Literal["model", "run", "checkpoint"]
    model_id: str | None = None
    run_id: str | None = None
    checkpoint_name: str | None = None
    include_cache: bool = False

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.scope == "model":
            if not self.model_id or self.run_id or self.checkpoint_name:
                raise ValueError("Для удаления модели нужен только model_id")
            if self.model_id != "custom":
                ModelPaths.for_model(self.model_id)
            elif self.include_cache:
                raise ValueError("У Custom нет Hugging Face cache")
        elif not self.run_id or self.model_id or self.include_cache:
            raise ValueError("Для удаления запуска/checkpoint нужен run_id, без model_id/cache")
        elif (self.scope == "checkpoint") != (self.checkpoint_name is not None):
            raise ValueError("checkpoint_name нужен только для scope=checkpoint")
        return self


class DeletionPath(BaseModel):
    """Точный объект очистки и оценка его объёма на диске."""

    path: str
    size_bytes: int
    files: int


class DeletionPreview(BaseModel):
    """План защищён от изменения файлов между просмотром и подтверждением."""

    target: DeletionTarget
    paths: list[DeletionPath]
    size_bytes: int
    files: int
    preview_token: str
    warning: str = "Удаление необратимо. Датасеты и код не затрагиваются."


class DeletionRequest(BaseModel):
    """Подтверждение уже показанного пользователю плана."""

    model_config = ConfigDict(extra="forbid")
    target: DeletionTarget
    preview_token: str = Field(min_length=64, max_length=64)
    confirmation: Literal["УДАЛИТЬ"]


class ManagedRun(BaseModel):
    """Лёгкая запись запуска для очистки без torch.load."""

    run_id: str
    title: str
    model_id: str
    directory: str
    checkpoints: list[str]
    size_bytes: int


class ManagedModel(BaseModel):
    """Модель, её локальные пути и запуски."""

    model_id: str
    registered: bool
    available: bool
    active: bool = False
    revision: str | None = None
    config_file: str | None = None
    snapshot_path: str | None = None
    size_bytes: int = 0
    runs: list[ManagedRun] = Field(default_factory=list)


class ModelsCatalog(BaseModel):
    """Актуальный каталог на момент запроса, без фонового сканирования."""

    models: list[ManagedModel]
    active_job_id: str | None = None
    total_size_bytes: int = 0
    warnings: list[str] = Field(default_factory=list)
