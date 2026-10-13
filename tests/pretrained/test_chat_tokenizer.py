"""Проверки родного instruct-формата и response-only границы без Hugging Face."""

from collections.abc import Sequence

import pytest

from mini_llm.data.dialogue import DialogueSample
from mini_llm.data.interfaces import OversizedResponseError
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


class ChatTokenizer:
    """Маленький детерминированный tokenizer с явными role/end маркерами."""

    chat_template: str | dict[str, str] | None = "native-template"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "<end>"
    pad_token = "<pad>"

    def __len__(self) -> int:
        return 65536

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) + 100 for char in text]

    def decode(self, tokens: Sequence[int], *, skip_special_tokens: bool) -> str:
        return "".join(chr(token - 100) for token in tokens if token >= 100)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        assert tokenize
        assert not return_dict
        tokens: list[int] = []
        for message in messages:
            tokens.append(11 if message["role"] == "user" else 12)
            tokens.extend(self.encode(message["content"], add_special_tokens=False))
            tokens.append(self.eos_token_id)
        if add_generation_prompt:
            tokens.append(12)
        return tokens


def test_instruct_prompt_uses_native_chat_template() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    assert tokenizer.encode_prompt("Привет") == [11, *tokenizer.encode("Привет"), 2, 12]


def test_chat_window_preserves_response_and_masks_context() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    sample = DialogueSample("b", 1, ("старый" * 20, "ответ в context", "новый"), "response")
    window = tokenizer.encode_training_window(sample, max_length=22)
    prompt = tokenizer.encode_prompt("новый")
    assert window.token_ids[: window.response_start] == prompt
    assert window.token_ids[window.response_start :] == [*tokenizer.encode("response"), 2]
    assert len(window.token_ids) <= 22
    assert "старый" not in tokenizer.decode(window.token_ids)


def test_chat_window_does_not_truncate_oversized_response() -> None:
    tokenizer = HuggingFaceDialogueTokenizer(ChatTokenizer())
    with pytest.raises(OversizedResponseError):
        tokenizer.encode_training_window(
            DialogueSample("b", 1, ("context",), "r" * 40), max_length=16
        )


def test_tokenizer_without_chat_template_keeps_legacy_prompt() -> None:
    backend = ChatTokenizer()
    backend.chat_template = None
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("text") == [
        1,
        *tokenizer.encode("<USER>"),
        *tokenizer.encode("text"),
        *tokenizer.encode("<ASSISTANT>"),
    ]


class ThinkingTokenizer(ChatTokenizer):
    """Имитировать переключатель reasoning и записывать параметры шаблона."""

    def __init__(self) -> None:
        self.chat_template = "native-template-with-enable_thinking"
        self.calls: list[tuple[bool, bool | None]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        self.calls.append((add_generation_prompt, enable_thinking))
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )
        if add_generation_prompt and enable_thinking is False:
            # Пустой think-блок входит в prompt, а не расходует бюджет генерации.
            tokens.extend([70, 71])
        return tokens


@pytest.mark.parametrize(
    "template",
    ["native-enable_thinking", {"default": "native-enable_thinking"}],
)
def test_reasoning_is_disabled_in_inference_prompt(template: str | dict[str, str]) -> None:
    backend = ThinkingTokenizer()
    backend.chat_template = template
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("Привет") == [11, *tokenizer.encode("Привет"), 2, 12, 70, 71]
    assert backend.calls == [(True, False)]


def test_disabling_inference_reasoning_does_not_change_training_window() -> None:
    backend = ThinkingTokenizer()
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    tokenizer.encode_prompt("Привет")
    backend.calls.clear()
    sample = DialogueSample("b", 1, ("question",), "answer")
    window = tokenizer.encode_training_window(sample, max_length=64)
    expected = HuggingFaceDialogueTokenizer(ChatTokenizer()).encode_training_window(
        sample, max_length=64
    )
    assert window == expected
    assert backend.calls == [(True, None), (False, None)]


def test_unused_reasoning_template_does_not_change_default_template() -> None:
    backend = ThinkingTokenizer()
    backend.chat_template = {"default": "native-template", "tool_use": "enable_thinking"}
    tokenizer = HuggingFaceDialogueTokenizer(backend)
    assert tokenizer.encode_prompt("text") == [11, *tokenizer.encode("text"), 2, 12]
    assert backend.calls == [(True, None)]
