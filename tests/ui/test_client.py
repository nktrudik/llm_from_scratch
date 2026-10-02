"""Проверки HTTP-клиента без сети и загрузки весов."""

import json

import pytest
import requests

from mini_llm.ui.client import GenerationAPIError, generate_reply, get_generation_options
from mini_llm.ui.config import API_BASE_URL, REQUEST_TIMEOUT, ModelBackend, PretrainedMode


def make_response(payload: object, status_code: int = 200) -> requests.Response:
    """Собрать небольшой JSON-ответ для подменённого HTTP-запроса."""

    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return response


@pytest.mark.parametrize("backend", ["custom", "pretrained"])
def test_client_sends_only_prompt_and_backend(
    monkeypatch: pytest.MonkeyPatch, backend: ModelBackend
) -> None:
    calls: list[dict[str, str]] = []

    def post(
        session: requests.Session,
        url: str,
        *,
        json: dict[str, str],
        timeout: tuple[float, float],
    ) -> requests.Response:
        assert url == f"{API_BASE_URL}/v1/generate"
        assert timeout == REQUEST_TIMEOUT
        calls.append(json)
        return make_response({"text": "Ответ модели", "token_ids": [1, 2]})

    monkeypatch.setattr(requests.Session, "post", post)

    assert generate_reply("Привет", backend) == "Ответ модели"
    assert calls == [{"prompt": "Привет", "model_backend": backend}]


@pytest.mark.parametrize(
    ("payload", "status", "message"),
    [
        ({"detail": "Генерация недоступна: идёт обучение"}, 409, "идёт обучение"),
        ({"detail": "Checkpoint не найден"}, 400, "Checkpoint не найден"),
        ({"detail": [{"msg": "Неверный запрос"}]}, 422, "HTTP 422"),
        ({"token_ids": [1]}, 200, "отсутствует текст"),
        (["Ответ"], 200, "отсутствует текст"),
    ],
)
def test_client_reports_api_errors(
    monkeypatch: pytest.MonkeyPatch, payload: object, status: int, message: str
) -> None:
    def post(*args: object, **kwargs: object) -> requests.Response:
        return make_response(payload, status)

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(GenerationAPIError, match=message):
        generate_reply("Привет", "custom")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (requests.ConnectionError("offline"), "сервер запущен"),
        (requests.Timeout("timeout"), "запрос не повторён"),
    ],
)
def test_client_does_not_retry_failed_requests(
    monkeypatch: pytest.MonkeyPatch, error: requests.RequestException, message: str
) -> None:
    calls = 0

    def post(*args: object, **kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(GenerationAPIError, match=message):
        generate_reply("Привет", "pretrained")
    assert calls == 1


def test_client_handles_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(*args: object, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = 502
        response._content = b"Bad gateway"
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    with pytest.raises(GenerationAPIError, match="некорректный JSON.*502"):
        generate_reply("Привет", "custom")


@pytest.mark.parametrize("mode", ["before_sft", "after_sft"])
def test_qwen_request_contains_mode_but_no_model_paths(
    monkeypatch: pytest.MonkeyPatch, mode: PretrainedMode
) -> None:
    calls: list[object] = []

    def post(*args: object, **kwargs: object) -> requests.Response:
        calls.append(kwargs["json"])
        return make_response({"text": "Ответ", "token_ids": [1]})

    monkeypatch.setattr(requests.Session, "post", post)
    assert generate_reply("Текст", "pretrained", mode) == "Ответ"
    assert calls == [{"prompt": "Текст", "model_backend": "pretrained", "pretrained_mode": mode}]


def test_client_reads_availability_from_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(*args: object, **kwargs: object) -> requests.Response:
        assert args[1] == f"{API_BASE_URL}/v1/generate/options"
        return make_response(
            {
                "model_id": "org/instruct",
                "before_sft_available": True,
                "after_sft_available": False,
                "reason": "best.pt нет",
            }
        )

    monkeypatch.setattr(requests.Session, "get", get)
    options = get_generation_options()
    assert options.model_id == "org/instruct"
    assert options.before_sft_available
    assert not options.after_sft_available


def test_client_sends_only_checkpoint_name_without_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def post(*args: object, **kwargs: object) -> requests.Response:
        calls.append(kwargs["json"])
        return make_response({"text": "Ответ", "token_ids": [1]})

    monkeypatch.setattr(requests.Session, "post", post)
    assert (
        generate_reply("Текст", "pretrained", "custom_checkpoint", " step_00000500.pt ") == "Ответ"
    )
    assert calls == [
        {
            "prompt": "Текст",
            "model_backend": "pretrained",
            "pretrained_mode": "custom_checkpoint",
            "checkpoint_name": "step_00000500.pt",
        }
    ]


@pytest.mark.parametrize("name", [None, "", "../step.pt", "folder\\step.pt", "model.json"])
def test_client_rejects_invalid_checkpoint_without_sending_request(
    monkeypatch: pytest.MonkeyPatch, name: str | None
) -> None:
    def unexpected_request(*args: object, **kwargs: object) -> requests.Response:
        pytest.fail("Неверное имя checkpoint не должно отправляться в API")

    monkeypatch.setattr(requests.Session, "post", unexpected_request)
    with pytest.raises(GenerationAPIError):
        generate_reply("Текст", "pretrained", "custom_checkpoint", name)
