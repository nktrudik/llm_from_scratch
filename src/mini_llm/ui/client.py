"""HTTP-клиент существующей ручки генерации без локальной загрузки модели."""

from pathlib import Path
from typing import cast

import requests
from pydantic import BaseModel, ValidationError

from mini_llm.ui.config import API_BASE_URL, REQUEST_TIMEOUT, ModelBackend, PretrainedMode


class GenerationAPIError(RuntimeError):
    """Ошибка API, которую можно показать пользователю чата."""


class GenerationOptions(BaseModel):
    """Доступность pretrained-режимов, полученная от сервера без загрузки модели."""

    model_id: str
    before_sft_available: bool
    after_sft_available: bool
    reason: str | None = None


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
    """Уточнить наличие best.pt на сервере, не предполагая общую файловую систему."""

    try:
        with requests.Session() as session:
            response = session.get(f"{API_BASE_URL}/v1/generate/options", timeout=(5.0, 10.0))
        return GenerationOptions.model_validate(_response_payload(response))
    except requests.RequestException as error:
        raise GenerationAPIError(
            "Не удалось проверить режимы Qwen: убедитесь, что API запущен."
        ) from error
    except ValidationError as error:
        raise GenerationAPIError("API вернул некорректный список режимов Qwen.") from error


def generate_reply(
    prompt: str,
    model_backend: ModelBackend,
    pretrained_mode: PretrainedMode | None = None,
    checkpoint_name: str | None = None,
) -> str:
    """Передать prompt, backend и режим Qwen; пути и параметры выбирает API."""

    body = {"prompt": prompt, "model_backend": model_backend}
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
        with requests.Session() as session:
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
