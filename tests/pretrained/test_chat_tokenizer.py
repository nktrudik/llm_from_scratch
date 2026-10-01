"""Проверки родного instruct-формата и response-only границы без Hugging Face."""

from collections.abc import Sequence

import pytest

from mini_llm.data.dialogue import DialogueSample
from mini_llm.data.interfaces import OversizedResponseError
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


class ChatTokenizer:
    """Маленький детерминированный tokenizer с явными role/end маркерами."""

    chat_template: str | None = "native-template"
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
        self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]:
        assert tokenize
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
