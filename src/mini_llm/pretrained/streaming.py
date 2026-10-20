"""Декодирование token IDs в обновляемый текст без изменения sampling."""

from collections.abc import Callable, Sequence
from typing import Protocol, cast

from torch import Tensor


class TextDecoder(Protocol):
    """Общий контракт декодирования для собственного BPE и Hugging Face."""

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


class TokenTextStreamer:
    """Передавать снимки ответа, не включая prompt и завершающие токены."""

    def __init__(
        self, decoder: TextDecoder, on_text: Callable[[str], None], stop_ids: Sequence[int]
    ) -> None:
        self.decoder = decoder
        self.on_text = on_text
        self.stop_ids = set(stop_ids)
        self.token_ids: list[int] = []
        self.prompt_pending = True
        self.stopped = False
        self.last_text = ""

    def put(self, value: Tensor) -> None:
        """Принять prompt первым вызовом, затем очередные сгенерированные токены."""

        if self.prompt_pending:
            self.prompt_pending = False
            self.on_text("")
            return
        for token_id in cast(list[int], value.reshape(-1).tolist()):
            if token_id in self.stop_ids:
                self.stopped = True
            if not self.stopped:
                self.token_ids.append(token_id)
        text = self.decoder.decode(self.token_ids, skip_special_tokens=True)
        # Незаконченный UTF-8 символ покажется после получения остальных байтов.
        if not text.endswith("\ufffd"):
            self.last_text = text
        self.on_text(self.last_text)

    def end(self) -> None:
        """Передать окончательный текст, в том числе для пустого ответа."""

        self.on_text(self.decoder.decode(self.token_ids, skip_special_tokens=True))
