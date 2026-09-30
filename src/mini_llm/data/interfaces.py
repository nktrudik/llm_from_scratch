"""Контракты токенизации диалоговых данных."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mini_llm.data.dialogue import DialogueSample


@dataclass(frozen=True, slots=True)
class EncodedDialogue:
    """Токены окна и индекс первого токена ответа в полной последовательности."""

    token_ids: list[int]
    response_start: int


class OversizedResponseError(ValueError):
    """Response вместе с обязательными special tokens не помещается в окно."""

    def __init__(self, *, response_tokens: int, max_length: int) -> None:
        self.response_tokens = response_tokens
        self.required_tokens = response_tokens + 1
        self.max_length = max_length
        super().__init__(
            f"Response требует {self.required_tokens} tokens вместе с BOS, "
            f"но размер окна равен {max_length}"
        )


class DialogueTokenizer(Protocol):
    """Минимальный tokenizer-интерфейс, необходимый Dataset."""

    @property
    def vocab_size(self) -> int:
        """Вернуть размер словаря."""

    @property
    def pad_token_id(self) -> int:
        """Вернуть ID padding-токена."""

    def encode_training_window(
        self,
        sample: DialogueSample,
        *,
        max_length: int,
    ) -> EncodedDialogue:
        """Закодировать диалог и отметить начало response для маски loss."""
