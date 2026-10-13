"""Проверки Streamlit-экрана с подменённой генерацией, без HTTP и модели."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mini_llm.ui import client
from mini_llm.ui import config as ui_config
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
    assert [block.value for block in app_test.markdown] == ["Первое сообщение"]
    assert [block.value for block in app_test.text] == ["Ответ: Первое сообщение"]

    app_test.chat_input[0].set_value("Второе сообщение").run()
    assert len(app_test.chat_message) == 2
    assert [block.value for block in app_test.markdown] == ["Второе сообщение"]
    assert [block.value for block in app_test.text] == ["Ответ: Второе сообщение"]
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


def test_ui_handles_cached_labels_without_custom_checkpoint(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Воспроизвести старый словарь config, оставшийся после горячего обновления UI."""

    monkeypatch.delitem(ui_config.MODE_LABELS, "custom_checkpoint")
    app_test.selectbox[0].select("pretrained").run()
    assert not app_test.exception
    assert "Свой чекпоинт" in app_test.radio[0].options
    app_test.radio[0].set_value("custom_checkpoint").run()
    assert not app_test.exception
    assert len(app_test.text_input) == 1
    assert app_test.chat_input[0].disabled


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
    assert app_test.radio[0].options == ["До SFT — исходная модель", "Свой чекпоинт"]
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


@pytest.mark.parametrize("checkpoint_name", ["step_00000500.pt", "last.pt", "best.pt"])
def test_ui_sends_checkpoint_name_and_requires_nonempty_input(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch, checkpoint_name: str
) -> None:
    calls: list[tuple[str, ModelBackend, PretrainedMode | None, str | None]] = []

    def generate(
        prompt: str,
        backend: ModelBackend,
        mode: PretrainedMode | None = None,
        checkpoint_name: str | None = None,
    ) -> str:
        calls.append((prompt, backend, mode, checkpoint_name))
        return "Ответ checkpoint"

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.selectbox[0].select("pretrained").run()
    app_test.radio[0].set_value("custom_checkpoint").run()
    assert app_test.chat_input[0].disabled
    assert len(app_test.text_input) == 1
    app_test.text_input[0].set_value("   ").run()
    assert app_test.chat_input[0].disabled
    app_test.text_input[0].set_value(f" {checkpoint_name} ").run()
    assert not app_test.chat_input[0].disabled
    app_test.chat_input[0].set_value("Привет").run()
    assert not app_test.exception
    assert calls == [("Привет", "pretrained", "custom_checkpoint", checkpoint_name)]
    app_test.radio[0].set_value("before_sft").run()
    assert not app_test.text_input
    assert len(calls) == 1
    app_test.selectbox[0].select("custom").run()
    assert not app_test.radio
    assert not app_test.text_input


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


def test_ui_selects_qwen_model_and_keeps_its_checkpoint_modes_separate(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Смена модели не генерирует ответ и не использует режим best другой модели."""

    first = client.PretrainedModelOptions(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        before_sft_available=True,
        after_sft_available=True,
    )
    second = client.PretrainedModelOptions(
        model_id="Qwen/Qwen3-0.6B",
        before_sft_available=True,
        after_sft_available=False,
        reason="best.pt ещё не создан",
    )
    monkeypatch.setattr(
        client,
        "get_generation_options",
        lambda: client.GenerationOptions(**first.model_dump(), models=[first, second]),
    )
    calls: list[tuple[str | None, PretrainedMode | None, str | None]] = []

    def generate(
        prompt: str,
        backend: ModelBackend,
        mode: PretrainedMode | None = None,
        checkpoint_name: str | None = None,
        pretrained_model_id: str | None = None,
    ) -> str:
        assert backend == "pretrained"
        calls.append((pretrained_model_id, mode, checkpoint_name))
        return "Ответ"

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.selectbox[0].select("pretrained").run()
    assert not app_test.exception
    assert app_test.selectbox[1].options == [first.model_id, second.model_id]
    app_test.radio[0].set_value("after_sft").run()
    app_test.selectbox[1].select(second.model_id).run()
    assert not app_test.exception
    assert app_test.radio[0].value == "before_sft"
    assert app_test.radio[0].options == ["До SFT — исходная модель", "Свой чекпоинт"]
    assert not calls
    app_test.chat_input[0].set_value("Тест Qwen3").run()
    assert not app_test.exception
    assert calls == [(second.model_id, "before_sft", None)]
    app_test.radio[0].set_value("custom_checkpoint").run()
    assert app_test.chat_input[0].disabled
    app_test.text_input[0].set_value("last.pt").run()
    app_test.chat_input[0].set_value("Тест checkpoint").run()
    assert not app_test.exception
    assert calls[-1] == (second.model_id, "custom_checkpoint", "last.pt")
    app_test.selectbox[1].select(first.model_id).run()
    assert not app_test.exception
    assert "После SFT — best checkpoint" in app_test.radio[0].options
    assert len(calls) == 2


def test_ui_displays_api_failure(app_test: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(prompt: str, backend: ModelBackend) -> str:
        raise client.GenerationAPIError("Запустите API")

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.chat_input[0].set_value("Привет").run()
    assert not app_test.exception
    assert app_test.error[0].value == "Запустите API"


def test_ui_disables_qwen_when_options_request_fails(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ошибка проверки режимов не должна мешать переключиться обратно на Custom."""

    def options() -> client.GenerationOptions:
        raise client.GenerationAPIError("Не удалось проверить режимы Qwen")

    monkeypatch.setattr(client, "get_generation_options", options)
    app_test.selectbox[0].select("pretrained").run()
    assert not app_test.exception
    assert app_test.chat_input[0].disabled
    assert app_test.error[0].value == "Не удалось проверить режимы Qwen"
    app_test.selectbox[0].select("custom").run()
    assert not app_test.exception
    assert not app_test.chat_input[0].disabled


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


@pytest.mark.parametrize(
    "reply",
    [">>123\nОтвет **без Markdown**\n\nВторая строка", "<b>Текст</b> и `код` 😄"],
)
def test_ui_displays_model_reply_as_literal_text(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    """Ссылки на посты и разметка не должны превращаться в цитаты или HTML."""

    def generate(prompt: str, backend: ModelBackend) -> str:
        return reply

    monkeypatch.setattr(client, "generate_reply", generate)
    app_test.chat_input[0].set_value("Вопрос").run()
    assert not app_test.exception
    assert [block.value for block in app_test.markdown] == ["Вопрос"]
    assert [block.value for block in app_test.text] == [reply]
