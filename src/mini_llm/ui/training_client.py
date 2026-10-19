"""HTTP-клиент каталога версий и фонового обучения."""

from __future__ import annotations

from typing import Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, Field, ValidationError

from mini_llm.observability.training import TrainingProgress
from mini_llm.ui.config import API_BASE_URL, OPTIONS_TIMEOUT
from mini_llm.ui.http import APIError as GenerationAPIError
from mini_llm.ui.http import api_session, response_payload
from mini_llm.workspace.catalog import ModelRun


class DatasetOption(BaseModel):
    """Описание доступной версии без чтения её данных на стороне UI."""

    dataset_id: str
    version: str
    title: str
    description: str = ""
    samples: int
    splits_ready: bool


class ModelOption(BaseModel):
    """Локально доступная модель и базовый профиль."""

    model_id: str
    max_sequence_length: int
    adaptation_mode: str


class WorkspaceOptions(BaseModel):
    """Серверный каталог для формы обучения."""

    datasets: list[DatasetOption]
    models: list[ModelOption]
    runs: list[ModelRun] = Field(default_factory=list)
    active_job_id: str | None = None
    dataset_count: int | None = None
    total_samples: int | None = None
    dataset_warnings: list[str] = Field(default_factory=list)


class JobInfo(BaseModel):
    """Состояние задачи, но не локальный training thread UI."""

    job_id: str
    kind: str = "training"
    status: Literal["queued", "running", "completed", "failed"]
    error: str | None = None
    result: dict[str, object] | None = None
    progress: TrainingProgress | None = None
    stop_requested: bool = False


def _request(method: str, path: str, body: dict[str, object] | None = None) -> object:
    try:
        with api_session(API_BASE_URL) as session:
            response = session.request(
                method, f"{API_BASE_URL}{path}", json=body, timeout=OPTIONS_TIMEOUT
            )
        return response_payload(response)
    except requests.RequestException as error:
        raise GenerationAPIError(
            f"Не удалось обратиться к API ({API_BASE_URL}): {error}"
        ) from error


def get_workspace(*, refresh_datasets: bool = True) -> WorkspaceOptions:
    """Прочитать свежий каталог; мониторинг задач не регистрирует новые датасеты."""

    try:
        path = "/v1/workspace" if refresh_datasets else "/v1/workspace?refresh_datasets=false"
        return WorkspaceOptions.model_validate(_request("GET", path))
    except ValidationError as error:
        raise GenerationAPIError("API вернул некорректный каталог версий") from error


def start_training(body: dict[str, object]) -> JobInfo:
    """Отправить ровно один запрос; автоматических повторов запуска нет."""

    return JobInfo.model_validate(_request("POST", "/v1/workspace/training", body))


def prepare_splits(dataset_id: str, version: str) -> JobInfo:
    """Явно запросить подготовку ещё не существующих splits."""

    path = f"/v1/workspace/datasets/{quote(dataset_id, safe='')}/{quote(version, safe='')}/split"
    return JobInfo.model_validate(_request("POST", path))


def get_job(job_id: str) -> JobInfo:
    """Проверить состояние ранее запущенной задачи."""

    return JobInfo.model_validate(_request("GET", f"/v1/jobs/{quote(job_id, safe='')}"))


def stop_training(job_id: str) -> JobInfo:
    """Попросить backend завершить текущий шаг и сохранить checkpoint."""

    return JobInfo.model_validate(_request("POST", f"/v1/jobs/{quote(job_id, safe='')}/stop"))
