"""Общий HTTP-транспорт UI, независимый от схем чата и обучения."""

from contextlib import suppress
from ipaddress import ip_address
from typing import cast
from urllib.parse import urlsplit

import requests


class APIError(RuntimeError):
    """Ошибка API, которую можно показать пользователю интерфейса."""


def api_session(api_base_url: str) -> requests.Session:
    """Не отправлять локальные запросы через системный proxy."""

    try:
        parsed = urlsplit(api_base_url)
        hostname = parsed.hostname
    except ValueError as error:
        raise APIError("Некорректный адрес API в MINI_LLM_API_URL.") from error
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise APIError("MINI_LLM_API_URL должен содержать полный http:// или https:// адрес.")
    local = hostname.rstrip(".").lower() == "localhost"
    with suppress(ValueError):
        local = local or ip_address(hostname).is_loopback
    session = requests.Session()
    if local:
        session.trust_env = False
    return session


def response_payload(response: requests.Response) -> object:
    """Прочитать JSON и преобразовать HTTP-ошибку в понятное сообщение."""

    try:
        payload = cast(object, response.json())
    except ValueError as error:
        raise APIError(f"API вернул некорректный JSON (HTTP {response.status_code}).") from error
    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        message = detail if isinstance(detail, str) else "Проверьте терминал сервера и /docs."
        raise APIError(f"Ошибка API (HTTP {response.status_code}): {message}")
    return payload
