"""HTTP-клиент управления моделями; никаких весов на стороне Streamlit."""

from __future__ import annotations

from typing import Literal

import requests
from pydantic import BaseModel, ValidationError

from mini_llm.pretrained.management.schemas import (
    DeletionPreview,
    DeletionRequest,
    DeletionTarget,
    DownloadRequest,
    ModelsCatalog,
    OperationProgress,
)
from mini_llm.ui.config import API_BASE_URL, OPTIONS_TIMEOUT
from mini_llm.ui.http import APIError, api_session, response_payload


class ModelJob(BaseModel):
    """Статус операции из общей очереди backend."""

    job_id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: str
    started_at: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None
    operation_progress: OperationProgress | None = None


def _request[T: BaseModel](
    schema: type[T], method: str, path: str, body: dict[str, object] | None = None
) -> T:
    try:
        with api_session(API_BASE_URL) as session:
            response = session.request(
                method, f"{API_BASE_URL}{path}", json=body, timeout=OPTIONS_TIMEOUT
            )
        return schema.model_validate(response_payload(response))
    except requests.RequestException as error:
        raise APIError(f"Не удалось обратиться к API ({API_BASE_URL}): {error}") from error
    except ValidationError as error:
        raise APIError(
            "API вернул некорректный ответ управления моделями; перезапустите backend"
        ) from error


def get_models() -> ModelsCatalog:
    """Загрузить актуальный локальный каталог без кэширования в UI."""

    return _request(ModelsCatalog, "GET", "/v1/models")


def download_model(model_id: str) -> ModelJob:
    """Поставить явную загрузку в очередь; сервер выбирает cache и профиль."""

    try:
        request = DownloadRequest(model_id=model_id)
    except ValidationError as error:
        raise APIError(
            "Укажите Hugging Face ID вида owner/model, не URL и не путь к файлу"
        ) from error
    return _request(ModelJob, "POST", "/v1/models/download", request.model_dump())


def preview_deletion(target: DeletionTarget) -> DeletionPreview:
    """Получить список удаляемых объектов до подтверждения."""

    return _request(DeletionPreview, "POST", "/v1/models/deletion-preview", target.model_dump())


def delete_files(request: DeletionRequest) -> ModelJob:
    """Отправить подтверждённый план ровно один раз, без retry."""

    return _request(ModelJob, "POST", "/v1/models/delete", request.model_dump())


def get_model_job(job_id: str) -> ModelJob:
    """Прочитать статус задачи, созданной сервером."""

    if not job_id.isalnum():
        raise APIError("Некорректный ID задачи")
    return _request(ModelJob, "GET", f"/v1/jobs/{job_id}")
