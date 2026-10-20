"""Потоковая генерация без весов, сети и GPU."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Sequence
from io import BytesIO
from threading import Event
from typing import cast

import pytest
import requests
import torch
from fastapi.testclient import TestClient

from mini_llm.api import app as api_module
from mini_llm.api import streaming
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import GenerationRequest
from mini_llm.inference.config import GenerationConfig
from mini_llm.inference.schemas import GenerationResult
from mini_llm.main import app
from mini_llm.pretrained.streaming import TokenTextStreamer
from mini_llm.ui import client


class Decoder:
    """Декодировать простые ID, имитируя незавершённый многобайтовый символ."""

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
        return {(): "", (1,): "\ufffd", (1, 2): "Я", (1, 2, 3): "Я!"}[tuple(token_ids)]


def test_streamer_skips_prompt_stop_and_buffers_unicode() -> None:
    snapshots: list[str] = []
    streamer = TokenTextStreamer(Decoder(), snapshots.append, [99])
    streamer.put(torch.tensor([[40, 41]]))
    for token in (1, 2, 3, 99, 50):
        streamer.put(torch.tensor([token]))
    streamer.end()
    assert snapshots == ["", "", "Я", "Я!", "Я!", "Я!", "Я!"]


def test_api_stream_sends_snapshots_and_holds_gpu_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = JobManager()
    monkeypatch.setattr(api_module, "job_manager", manager)

    def generate(
        prompt: str, config: GenerationConfig, *, on_text: Callable[[str], None]
    ) -> GenerationResult:
        assert prompt == "Привет"
        assert (config.temperature, config.top_k, config.max_new_tokens) == (0.3, 20, 1024)
        with pytest.raises(JobAlreadyRunningError):
            manager.create("training")
        on_text("Ответ")
        on_text("Ответ продолжается")
        return GenerationResult("Ответ продолжается", [1, 2])

    monkeypatch.setattr(streaming, "generate_response", generate)
    with TestClient(app) as http:
        response = http.post("/v1/generate/stream", json={"prompt": "Привет", "device": "cpu"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.iter_lines()]
    assert [event["text"] for event in events if event["type"] == "text"] == [
        "Ответ",
        "Ответ продолжается",
    ]
    assert events[-1] == {"type": "done", "text": "Ответ продолжается"}
    assert manager.create("training").kind == "training"


def test_api_stream_preserves_partial_answer_before_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_module, "job_manager", JobManager())

    def fail(
        prompt: str, config: GenerationConfig, *, on_text: Callable[[str], None]
    ) -> GenerationResult:
        on_text("Часть ответа")
        raise RuntimeError("Недостаточно памяти")

    monkeypatch.setattr(streaming, "generate_response", fail)
    with TestClient(app) as http:
        response = http.post("/v1/generate/stream", json={"prompt": "Тест", "device": "cpu"})
    events = [json.loads(line) for line in response.iter_lines()]
    assert events[-2] == {"type": "text", "text": "Часть ответа"}
    assert events[-1] == {"type": "error", "detail": "Недостаточно памяти"}


def test_api_stream_rejects_generation_during_training(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = JobManager()
    manager.create("training")
    monkeypatch.setattr(api_module, "job_manager", manager)
    with TestClient(app) as http:
        response = http.post("/v1/generate/stream", json={"prompt": "Тест", "device": "cpu"})
    assert response.status_code == 409


@pytest.mark.parametrize("ending", ["done", "error", "disconnect", "invalid"])
def test_ui_client_reads_incremental_text_and_detects_failure(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    frames = [{"type": "heartbeat"}, {"type": "text", "text": "Я"}]
    if ending == "done":
        frames.append({"type": "done", "text": "Я ответил"})
    if ending == "error":
        frames.append({"type": "error", "detail": "Ошибка GPU"})

    def post(*args: object, **kwargs: object) -> requests.Response:
        assert str(args[1]).endswith("/v1/generate/stream")
        assert kwargs["stream"] is True
        assert kwargs["json"] == {"prompt": "Тест", "model_backend": "custom"}
        response = requests.Response()
        response.status_code = 200
        response._content = (
            "\n".join(json.dumps(frame, ensure_ascii=False) for frame in frames)
            + ("\nnot json" if ending == "invalid" else "\n")
        ).encode("utf-8")
        response.raw = BytesIO(response._content)
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    iterator = client.stream_reply("Тест", "custom")
    assert next(iterator) == "Я"
    if ending == "done":
        assert list(iterator) == ["Я ответил"]
    else:
        with pytest.raises(client.GenerationAPIError):
            list(iterator)


def test_custom_streaming_does_not_change_sampled_tokens() -> None:
    from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig

    model = DecoderOnlyTransformer(ModelConfig(vocab_size=32, d_model=16, num_heads=4))
    model.eval()
    prompt = torch.tensor([[1, 2]])
    torch.manual_seed(42)
    expected = model.generate(prompt, max_new_tokens=4)
    chunks: list[list[int]] = []
    torch.manual_seed(42)
    actual = model.generate(
        prompt,
        max_new_tokens=4,
        on_token=lambda token: chunks.append(cast(list[int], token.flatten().tolist())),
    )
    assert torch.equal(actual, expected)
    assert chunks == [[token] for token in expected[0, 2:].tolist()]


def test_disconnected_stream_stops_worker_and_releases_gpu_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager()
    release, stopped = Event(), Event()

    def generate(
        prompt: str, config: GenerationConfig, *, on_text: Callable[[str], None]
    ) -> GenerationResult:
        try:
            on_text("Начало")
            assert release.wait(timeout=5)
            on_text("Не должно отправиться после отключения")
            pytest.fail("Отключённый клиент должен остановить генерацию")
        finally:
            stopped.set()

    monkeypatch.setattr(streaming, "generate_response", generate)

    async def disconnect() -> None:
        response = streaming.generation_stream(GenerationRequest(prompt="Тест"), manager)
        iterator = cast(AsyncGenerator[str], response.body_iterator)
        assert json.loads(await anext(iterator))["type"] == "status"
        assert json.loads(await anext(iterator))["text"] == "Начало"
        await iterator.aclose()
        release.set()
        assert await asyncio.to_thread(stopped.wait, 5)

    asyncio.run(disconnect())
    assert manager.create("training").kind == "training"
