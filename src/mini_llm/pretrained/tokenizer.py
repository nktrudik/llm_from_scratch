"""Адаптер Hugging Face tokenizer к формату диалогового Dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from mini_llm.data.dialogue import (
    ASSISTANT_TOKEN,
    USER_TOKEN,
    DialogueSample,
    context_role_tokens,
)
from mini_llm.data.interfaces import EncodedDialogue, OversizedResponseError


class HuggingFaceTokenizerProtocol(Protocol):
    """Используемая часть API ``PreTrainedTokenizerBase``."""

    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    eos_token: str | None
    pad_token: str | None

    def __len__(self) -> int: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...

    def add_special_tokens(self, special_tokens_dict: dict[str, object]) -> int: ...

    def save_pretrained(self, save_directory: str) -> object: ...


class HuggingFaceDialogueTokenizer:
    """Форматировать dialogue samples tokenizer-ом pretrained-модели."""

    def __init__(self, tokenizer: object) -> None:
        self.backend = cast(HuggingFaceTokenizerProtocol, tokenizer)
        self._configure_special_tokens()

    def _configure_special_tokens(self) -> None:
        if self.backend.eos_token_id is None or self.backend.eos_token is None:
            raise RuntimeError("Pretrained tokenizer должен содержать EOS token")

        if self.backend.pad_token_id is None:
            self.backend.pad_token = self.backend.eos_token

    @property
    def vocab_size(self) -> int:
        """Вернуть полный размер словаря с добавленными role tokens."""

        return len(self.backend)

    @property
    def pad_token_id(self) -> int:
        """Вернуть настроенный padding ID."""

        token_id = self.backend.pad_token_id
        if token_id is None:
            raise RuntimeError("Hugging Face tokenizer не содержит pad_token_id")
        return token_id

    def encode(self, text: str) -> list[int]:
        """Закодировать текст без автоматического добавления special tokens."""

        return self.backend.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Декодировать token IDs в текст."""

        return self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _role_ids(self, role: str) -> list[int]:
        token_ids = self.encode(role)
        if not token_ids:
            raise RuntimeError(f"Не удалось закодировать role marker {role}")
        return token_ids

    def _bos_id(self) -> int:
        token_id = self.backend.bos_token_id
        if token_id is not None:
            return token_id
        return self._eos_id()

    def _eos_id(self) -> int:
        token_id = self.backend.eos_token_id
        if token_id is None:
            raise RuntimeError("Hugging Face tokenizer не содержит eos_token_id")
        return token_id

    @property
    def eos_token_id(self) -> int:
        """Вернуть завершающий token ID для остановки генерации."""

        return self._eos_id()

    def encode_prompt(self, prompt: str) -> list[int]:
        """Собрать простой single-turn prompt в проектном role-формате."""

        return [
            self._bos_id(),
            *self._role_ids(USER_TOKEN),
            *self.encode(prompt),
            *self._role_ids(ASSISTANT_TOKEN),
        ]

    def encode_training_window(
        self,
        sample: DialogueSample,
        *,
        max_length: int,
    ) -> EncodedDialogue:
        """Сохранить response и заполнить окно последними context messages."""

        context_segments = [
            [*self._role_ids(role), *self.encode(text)]
            for role, text in zip(
                context_role_tokens(len(sample.context)),
                sample.context,
                strict=True,
            )
        ]
        assistant_ids = self._role_ids(ASSISTANT_TOKEN)

        response_ids = [
            *assistant_ids,
            *self.encode(sample.response),
            self._eos_id(),
        ]
        if len(response_ids) + 1 > max_length:
            raise OversizedResponseError(response_tokens=len(response_ids), max_length=max_length)
        remaining = max_length - len(response_ids) - 1
        selected: list[list[int]] = []
        for segment in reversed(context_segments):
            if len(segment) <= remaining:
                selected.append(segment)
                remaining -= len(segment)
            else:
                break
        context_ids = [token_id for segment in reversed(selected) for token_id in segment]
        token_ids = [self._bos_id(), *context_ids, *response_ids]
        response_start = len(token_ids) - len(response_ids) + len(assistant_ids)
        return EncodedDialogue(token_ids, response_start)
