"""Проверки компактного чата без модели, сети и сохранения истории."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.ui import client

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Открыть настоящий Streamlit-экран с серверным каталогом-заглушкой."""

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    checkpoints = [
        client.ExperimentCheckpointOptions(name="last.pt", global_step=200),
        client.ExperimentCheckpointOptions(name="best.pt", global_step=100),
        client.ExperimentCheckpointOptions(name="step_00000100.pt", global_step=100),
    ]
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="Qwen/Qwen3-0.6B",
            before_sft_available=True,
            after_sft_available=True,
            models=[
                client.PretrainedModelOptions(
                    model_id="Qwen/Qwen3-0.6B",
                    before_sft_available=True,
                    after_sft_available=True,
                    experiments=[
                        client.ExperimentOptions(
                            experiment_id="qwen-v2",
                            title="QA v2 · run-002",
                            checkpoints=checkpoints,
                        )
                    ],
                ),
                client.PretrainedModelOptions(
                    model_id="Qwen/Qwen2.5-0.5B-Instruct",
                    before_sft_available=True,
                    after_sft_available=False,
                ),
            ],
            custom_runs=[
                client.ExperimentOptions(
                    experiment_id="custom-v1",
                    title="Custom v1",
                    checkpoints=checkpoints,
                )
            ],
        ),
    )
    return AppTest.from_file(Path(__file__).parents[2] / "src" / "mini_llm" / "ui" / "app.py").run()


def test_only_one_exchange_and_no_automatic_generation(
    chat: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def generate(prompt: str, backend: str, mode: object = None, **kwargs: object) -> Iterator[str]:
        calls.append({"prompt": prompt, "backend": backend, "mode": mode, **kwargs})
        return iter(["Ответ: " + prompt])

    monkeypatch.setattr(client, "stream_reply", generate)
    assert not chat.exception and not calls
    assert len(chat.selectbox) == 3
    chat.chat_input[0].set_value("Привет").run()
    assert not chat.exception
    assert [item.value for item in chat.text] == ["Привет", "Ответ: Привет"]
    chat.chat_input[0].set_value("Ещё вопрос").run()
    assert len(chat.chat_message) == 2 and len(calls) == 2
    chat.selectbox[0].select("Qwen/Qwen3-0.6B").run()
    assert len(calls) == 2 and not chat.chat_message


@pytest.mark.parametrize("checkpoint", ["last.pt", "best.pt", "step_00000100.pt"])
def test_qwen_selects_run_and_checkpoint_without_paths(
    chat: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    calls: list[dict[str, object]] = []

    def generate(prompt: str, backend: str, mode: object = None, **kwargs: object) -> Iterator[str]:
        calls.append({"backend": backend, "mode": mode, **kwargs})
        return iter(["Ответ"])

    monkeypatch.setattr(client, "stream_reply", generate)
    chat.selectbox[0].select("Qwen/Qwen3-0.6B").run()
    assert not chat.exception and not chat.radio
    assert chat.selectbox[1].value == "qwen-v2"
    chat.selectbox[2].select(checkpoint).run()
    chat.chat_input[0].set_value("Привет").run()
    assert calls == [
        {
            "backend": "pretrained",
            "mode": "custom_checkpoint",
            "checkpoint_name": checkpoint,
            "pretrained_model_id": "Qwen/Qwen3-0.6B",
            "experiment_id": "qwen-v2",
        }
    ]


def test_base_model_and_models_without_runs(chat: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def generate(prompt: str, backend: str, mode: object = None, **kwargs: object) -> Iterator[str]:
        calls.append(mode)
        assert kwargs["checkpoint_name"] is None and kwargs["experiment_id"] is None
        return iter(["Привет!"])

    monkeypatch.setattr(client, "stream_reply", generate)
    chat.selectbox[0].select("Qwen/Qwen3-0.6B").run()
    chat.selectbox[1].select("").run()
    assert len(chat.selectbox) == 2
    chat.chat_input[0].set_value("Тест").run()
    chat.selectbox[0].select("Qwen/Qwen2.5-0.5B-Instruct").run()
    assert not chat.exception and len(chat.selectbox) == 2
    chat.chat_input[0].set_value("Тест").run()
    assert calls == ["before_sft", "before_sft"]


@pytest.mark.parametrize("reply", [">>123\nОтвет **без Markdown**", "<b>Текст</b> 😄", ""])
def test_literal_model_output(chat: AppTest, monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    monkeypatch.setattr(client, "stream_reply", lambda *args, **kwargs: iter([reply]))
    chat.chat_input[0].set_value("Тест").run()
    assert not chat.exception
    assert chat.text[-1].value == (reply or "Пустой ответ")
    assert not chat.markdown


def test_api_failure_is_shown_without_crash(chat: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> client.GenerationOptions:
        raise client.GenerationAPIError("API недоступен")

    monkeypatch.setattr(client, "get_generation_options", fail)
    chat.run()
    assert not chat.exception
    assert chat.error[0].value == "API недоступен"
    assert not chat.chat_input


def test_stream_updates_one_plain_text_block_and_preserves_partial_answer(
    chat: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stream(*args: object, **kwargs: object) -> Iterator[str]:
        yield ">>123"
        yield ">>123\nОтвет **как текст**"
        raise client.GenerationAPIError("Соединение прервано")

    monkeypatch.setattr(client, "stream_reply", stream)
    chat.chat_input[0].set_value("Тест").run()
    assert not chat.exception
    assert len(chat.chat_message) == 2
    assert chat.text[-1].value == ">>123\nОтвет **как текст**"
    assert chat.error[0].value == "Соединение прервано"
    assert not chat.markdown and not chat.checkbox


def test_no_weights_disables_chat(chat: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="",
            before_sft_available=False,
            after_sft_available=False,
        ),
    )
    chat.run()
    assert not chat.exception and chat.chat_input[0].disabled
