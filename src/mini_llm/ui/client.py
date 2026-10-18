"""HTTP-клиент существующей ручки генерации без локальной загрузки модели."""

from pathlib import Path
from typing import cast

import requests
from pydantic import BaseModel, Field, ValidationError

from mini_llm.ui.config import (
    API_BASE_URL,
    OPTIONS_TIMEOUT,
    REQUEST_TIMEOUT,
    ModelBackend,
    PretrainedMode,
)
from mini_llm.ui.http import APIError, api_session, response_payload

UI_CLIENT_VERSION = 2
GenerationAPIError = APIError


class ExperimentCheckpointOptions(BaseModel):
    """Номер шага и loss файла, доступного на сервере."""

    name: str
    global_step: int | None
    validation_loss: float | None = None


class ExperimentOptions(BaseModel):
    """Один эксперимент с серверным списком совместимых checkpoints."""

    experiment_id: str
    checkpoints: list[ExperimentCheckpointOptions]
    title: str = ""
    directory: str = ""


class PretrainedModelOptions(BaseModel):
    """Доступность pretrained-режимов одной модели, полученная от сервера."""

    model_id: str
    before_sft_available: bool
    after_sft_available: bool
    reason: str | None = None
    experiments: list[ExperimentOptions] = Field(default_factory=list)


class GenerationOptions(PretrainedModelOptions):
    """Каталог моделей с совместимостью с прежним ответом без списка models."""

    models: list[PretrainedModelOptions] = Field(default_factory=list)
    custom_runs: list[ExperimentOptions] = Field(default_factory=list)


def get_generation_options() -> GenerationOptions:
    """Получить каталог, различая недоступность соединения и медленную индексацию."""

    try:
        with api_session(API_BASE_URL) as session:
            response = session.get(f"{API_BASE_URL}/v1/generate/options", timeout=OPTIONS_TIMEOUT)
        return GenerationOptions.model_validate(response_payload(response))
    except requests.ConnectTimeout as error:
        raise GenerationAPIError(
            f"Не удалось установить соединение с API ({API_BASE_URL}) за {OPTIONS_TIMEOUT[0]:g} с. "
            "Проверьте адрес, порт и MINI_LLM_API_URL."
        ) from error
    except requests.Timeout as error:
        raise GenerationAPIError(
            f"API ({API_BASE_URL}) не вернул каталог моделей/checkpoints "
            f"за {OPTIONS_TIMEOUT[1]:g} с. После обновления кода перезапустите backend. "
            "Проверьте терминал сервера и обновите страницу."
        ) from error
    except requests.ConnectionError as error:
        raise GenerationAPIError(
            f"Не удалось подключиться к API ({API_BASE_URL}). "
            "Проверьте, что backend запущен на этом адресе и порту."
        ) from error
    except requests.RequestException as error:
        raise GenerationAPIError(f"Ошибка запроса к API ({API_BASE_URL}): {error}") from error
    except ValidationError as error:
        raise GenerationAPIError("API вернул некорректный список режимов Qwen.") from error


def generate_reply(
    prompt: str,
    model_backend: ModelBackend,
    pretrained_mode: PretrainedMode | None = None,
    checkpoint_name: str | None = None,
    pretrained_model_id: str | None = None,
    experiment_id: str | None = None,
) -> str:
    """Передать prompt, backend и режим Qwen; пути и параметры выбирает API."""

    body = {"prompt": prompt, "model_backend": model_backend}
    if experiment_id is not None:
        if (model_backend == "pretrained" and pretrained_mode != "custom_checkpoint") or (
            model_backend == "custom" and pretrained_mode is not None
        ):
            raise GenerationAPIError("Эксперимент допустим только для checkpoint pretrained-модели")
        body["experiment_id"] = experiment_id
    if pretrained_model_id is not None:
        if model_backend != "pretrained":
            raise GenerationAPIError("Model ID допустим только для Qwen/pretrained backend")
        body["pretrained_model_id"] = pretrained_model_id
    if model_backend == "pretrained" and pretrained_mode is not None:
        body["pretrained_mode"] = pretrained_mode
    if pretrained_mode == "custom_checkpoint" or (
        model_backend == "custom" and experiment_id is not None
    ):
        if checkpoint_name is None:
            raise GenerationAPIError("Для режима «Свой чекпоинт» Qwen нужно имя checkpoint")
        name = checkpoint_name.strip()
        if not name or any(char in name for char in "/\\:") or Path(name).suffix != ".pt":
            raise GenerationAPIError(
                "Укажите только имя .pt файла, например step_00000500.pt, без пути"
            )
        body["checkpoint_name"] = name
    elif checkpoint_name is not None:
        raise GenerationAPIError("Имя checkpoint допустимо только в режиме «Свой чекпоинт»")
    try:
        with api_session(API_BASE_URL) as session:
            response = session.post(
                f"{API_BASE_URL}/v1/generate",
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
    except requests.Timeout as error:
        raise GenerationAPIError(
            "API не ответил за 10 минут. Проверьте терминал сервера; запрос не повторён."
        ) from error
    except requests.RequestException as error:
        raise GenerationAPIError(
            f"Не удалось связаться с API ({API_BASE_URL}). Убедитесь, что сервер запущен."
        ) from error

    payload = response_payload(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise GenerationAPIError("В ответе API отсутствует текст модели.")
    return cast(str, payload["text"])
