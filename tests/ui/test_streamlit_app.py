"""Проверки Streamlit-экрана с подменённой генерацией, без HTTP и модели."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.ui import client
from mini_llm.ui.config import ModelBackend, PretrainedMode

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_test(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """Запустить экран без браузера; optional-группа ui нужна только для этих тестов."""

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="org/instruct", before_sft_available=True, after_sft_available=True
        ),
    )
    return AppTest.from_file(Path(__file__).parents[2] / "src" / "mini_llm" / "ui" / "app.py").run()


def test_ui_shows_only_current_exchange(app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ModelBackend, PretrainedMode | None]] = []

    def generate(prompt: str, backend: ModelBackend, mode: PretrainedMode | None = None) -> str:
        calls.append((prompt, backend, mode))
        return f"Ответ: {prompt}"

    monkeypatch.setattr(client, "generate_reply", generate)
    assert not app_test.exception
    assert len(app_test.selectbox) == len(app_test.chat_input) == 1
    assert not app_test.chat_message
    assert not app_test.sidebar.children

    app_test.chat_input[0].set_value("Первое сообщение").run()
    assert not app_test.exception
    assert len(app_test.chat_message) == 2
    assert [block.value for block in app_test.markdown] == [
        "Первое сообщение",
        "Ответ: Первое сообщение",
    ]

    app_test.chat_input[0].set_value("Второе сообщение").run()
    assert len(app_test.chat_message) == 2
    assert [block.value for block in app_test.markdown] == [
        "Второе сообщение",
        "Ответ: Второе сообщение",
    ]
    assert calls == [("Первое сообщение", "custom", None), ("Второе сообщение", "custom", None)]


def test_ui_switches_backend_without_replaying_request(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ModelBackend, PretrainedMode | None]] = []

    def generate(prompt: str, backend: ModelBackend, mode: PretrainedMode | None = None) -> str:
        calls.append((prompt, backend, mode))
        return "Привет!"

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.selectbox[0].select("pretrained").run()
    assert not calls
    app_test.chat_input[0].set_value("Привет").run()
    assert not app_test.exception
    assert calls == [("Привет", "pretrained", "before_sft")]
    app_test.selectbox[0].select("custom").run()
    assert len(calls) == 1
    assert not app_test.chat_message


def test_ui_qwen_after_sft_is_unselectable_without_best_checkpoint(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="org/instruct",
            before_sft_available=True,
            after_sft_available=False,
            reason="После SFT недоступен: best.pt ещё не создан.",
        ),
    )
    assert app_test.selectbox[0].options == ["Custom", "Qwen"]
    app_test.selectbox[0].select("pretrained").run()
    assert not app_test.exception
    assert app_test.radio[0].options == ["До SFT — исходная модель"]
    assert not app_test.chat_input[0].disabled


def test_ui_can_select_after_sft_when_checkpoint_exists(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str | None] = []

    def generate(prompt: str, backend: ModelBackend, mode: PretrainedMode | None = None) -> str:
        calls.append(mode)
        return "Ответ"

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.selectbox[0].select("pretrained").run()
    app_test.radio[0].set_value("after_sft").run()
    app_test.chat_input[0].set_value("Тест").run()
    assert not app_test.exception
    assert calls == ["after_sft"]


def test_ui_requires_preparation_for_qwen(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(
            model_id="org/instruct",
            before_sft_available=False,
            after_sft_available=False,
            reason="Выполните setup",
        ),
    )
    app_test.selectbox[0].select("pretrained").run()
    assert app_test.chat_input[0].disabled
    app_test.selectbox[0].select("custom").run()
    assert not app_test.radio
    assert not app_test.chat_input[0].disabled


def test_ui_displays_api_failure(app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(prompt: str, backend: ModelBackend) -> str:
        raise client.GenerationAPIError("Запустите API")

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.chat_input[0].set_value("Привет").run()
    assert not app_test.exception
    assert app_test.error[0].value == "Запустите API"


def test_ui_ignores_blank_input(app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(prompt: str, backend: ModelBackend) -> str:
        pytest.fail("Пустое сообщение не должно отправляться в API")

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.chat_input[0].set_value("   ").run()
    assert not app_test.exception
    assert not app_test.chat_message


def test_ui_handles_empty_model_reply(app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(prompt: str, backend: ModelBackend) -> str:
        return ""

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.chat_input[0].set_value("Привет").run()
    assert not app_test.exception
    assert app_test.caption[0].value == "Модель вернула пустой ответ."
