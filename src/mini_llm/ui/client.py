"""HTTP-клиент существующей ручки генерации без локальной загрузки модели."""

from contextlib import suppress
from ipaddress import ip_address
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, Field, ValidationError

from mini_llm.ui.config import (
    API_BASE_URL,
    OPTIONS_TIMEOUT,
    REQUEST_TIMEOUT,
    ModelBackend,
    PretrainedMode,
)


class GenerationAPIError(RuntimeError):
    """Ошибка API, которую можно показать пользователю чата."""


class ExperimentCheckpointOptions(BaseModel):
    """Номер шага и loss файла, доступного на сервере."""

    name: str
    global_step: int
    validation_loss: float | None = None


class ExperimentOptions(BaseModel):
    """Один эксперимент с серверным списком совместимых checkpoints."""

    experiment_id: str
    checkpoints: list[ExperimentCheckpointOptions]


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


def _api_session() -> requests.Session:
    """Не отправлять локальные запросы к API через системный proxy."""

    try:
        parsed = urlsplit(API_BASE_URL)
        hostname = parsed.hostname
    except ValueError as error:
        raise GenerationAPIError("Некорректный адрес API в MINI_LLM_API_URL.") from error
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise GenerationAPIError(
            "MINI_LLM_API_URL должен содержать полный http:// или https:// адрес."
        )
    local = hostname.rstrip(".").lower() == "localhost"
    with suppress(ValueError):
        local = local or ip_address(hostname).is_loopback
    session = requests.Session()
    if local:
        session.trust_env = False
    return session


def _response_payload(response: requests.Response) -> object:
    try:
        payload = cast(object, response.json())
    except ValueError as error:
        raise GenerationAPIError(
            f"API вернул некорректный JSON (HTTP {response.status_code})."
        ) from error
    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = detail if isinstance(detail, str) else "Проверьте терминал сервера и /docs."
        raise GenerationAPIError(f"Ошибка API (HTTP {response.status_code}): {message}")
    return payload


def get_generation_options() -> GenerationOptions:
    """Получить каталог, различая недоступность соединения и медленную индексацию."""

    try:
        with _api_session() as session:
            response = session.get(f"{API_BASE_URL}/v1/generate/options", timeout=OPTIONS_TIMEOUT)
        return GenerationOptions.model_validate(_response_payload(response))
    except requests.ConnectTimeout as error:
        raise GenerationAPIError(
            f"Не удалось установить соединение с API ({API_BASE_URL}) за {OPTIONS_TIMEOUT[0]:g} с. "
            "Проверьте адрес, порт и MINI_LLM_API_URL."
        ) from error
    except requests.Timeout as error:
        raise GenerationAPIError(
            f"API ({API_BASE_URL}) не вернул каталог моделей/checkpoints "
            f"за {OPTIONS_TIMEOUT[1]:g} с. Возможно, первое чтение metadata ещё продолжается. "
            "Проверьте терминал backend и обновите страницу."
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
        if model_backend != "pretrained" or pretrained_mode != "custom_checkpoint":
            raise GenerationAPIError("Эксперимент допустим только для checkpoint pretrained-модели")
        body["experiment_id"] = experiment_id
    if pretrained_model_id is not None:
        if model_backend != "pretrained":
            raise GenerationAPIError("Model ID допустим только для Qwen/pretrained backend")
        body["pretrained_model_id"] = pretrained_model_id
    if model_backend == "pretrained" and pretrained_mode is not None:
        body["pretrained_mode"] = pretrained_mode
    if pretrained_mode == "custom_checkpoint":
        if model_backend != "pretrained" or checkpoint_name is None:
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
        with _api_session() as session:
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

    payload = _response_payload(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise GenerationAPIError("В ответе API отсутствует текст модели.")
    return cast(str, payload["text"])
