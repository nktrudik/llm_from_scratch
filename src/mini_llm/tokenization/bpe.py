"""Обучение, сохранение и использование byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.data.dialogue import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    USER_TOKEN,
    DialogueFormatError,
    DialogueSample,
    context_role_tokens,
    iter_training_texts,
)
from mini_llm.modeling.config import DEFAULT_MAX_SEQUENCE_LENGTH, ModelConfig
from mini_llm.tokenization.config import DEFAULT_TOKENIZER_PATH


class TokenizerError(RuntimeError):
    """Ошибка конфигурации, обучения или загрузки tokenizer."""


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


class BPETokenizer:
    """Тонкая типизированная оболочка над Hugging Face Tokenizers."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._validate_special_token_ids()

    @property
    def vocab_size(self) -> int:
        """Вернуть фактический размер vocabulary."""

        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> int:
        """Вернуть ID известного token или UNK ID."""

        token_id = self._tokenizer.token_to_id(token)
        return SPECIAL_TOKEN_IDS[UNK_TOKEN] if token_id is None else token_id

    def _validate_special_token_ids(self) -> None:
        config = ModelConfig()
        expected_ids = {
            PAD_TOKEN: config.pad_token_id,
            UNK_TOKEN: config.unk_token_id,
            BOS_TOKEN: config.bos_token_id,
            EOS_TOKEN: config.eos_token_id,
            USER_TOKEN: config.user_token_id,
            ASSISTANT_TOKEN: config.assistant_token_id,
        }
        actual_ids = {token: self._tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
        if actual_ids != expected_ids:
            raise TokenizerError(
                f"Special token IDs несовместимы с ModelConfig: {actual_ids}, "
                f"ожидались {expected_ids}"
            )

    def encode(self, text: str) -> list[int]:
        """Преобразовать Unicode-текст в token IDs без автоматических BOS/EOS."""

        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Декодировать token IDs обратно в строку."""

        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def encode_context(self, sample: DialogueSample) -> list[int]:
        """Закодировать context вместе с role tokens."""

        return [
            token_id for segment in self.encode_context_segments(sample) for token_id in segment
        ]

    def encode_context_segments(self, sample: DialogueSample) -> list[list[int]]:
        """Закодировать context отдельными role-prefixed messages."""

        segments: list[list[int]] = []
        roles = context_role_tokens(len(sample.context))
        for role, text in zip(roles, sample.context, strict=True):
            segments.append([self.token_to_id(role), *self.encode(text)])
        return segments

    def encode_response(self, sample: DialogueSample) -> list[int]:
        """Закодировать response с ASSISTANT и завершающим EOS."""

        return [
            self.token_to_id(ASSISTANT_TOKEN),
            *self.encode(sample.response),
            self.token_to_id(EOS_TOKEN),
        ]

    def encode_dialogue(self, sample: DialogueSample) -> list[int]:
        """Собрать полный sample в формате BOS + context + response."""

        return [
            self.token_to_id(BOS_TOKEN),
            *self.encode_context(sample),
            *self.encode_response(sample),
        ]

    def encode_dialogue_window(
        self, sample: DialogueSample, *, max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    ) -> list[int]:
        """Сохранить response целиком и заполнить окно последними context messages."""

        return self.build_dialogue_window(
            self.encode_context_segments(sample),
            self.encode_response(sample),
            max_length=max_length,
        )

    def build_dialogue_window(
        self,
        context_segments: Sequence[Sequence[int]],
        response_ids: Sequence[int],
        *,
        max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> list[int]:
        """Собрать окно из уже закодированных частей по единой политике truncation."""

        if max_length <= 0:
            raise ValueError("max_length должен быть положительным")
        bos_id = self.token_to_id(BOS_TOKEN)
        if len(response_ids) + 1 > max_length:
            raise OversizedResponseError(response_tokens=len(response_ids), max_length=max_length)

        remaining = max_length - len(response_ids) - 1
        selected: list[list[int]] = []
        for encoded_segment in reversed(context_segments):
            segment = list(encoded_segment)
            if len(segment) <= remaining:
                selected.append(segment)
                remaining -= len(segment)
                continue
            if remaining >= 2:
                selected.append([segment[0], *segment[-(remaining - 1) :]])
            break

        context_ids = [token_id for segment in reversed(selected) for token_id in segment]
        return [bos_id, *context_ids, *response_ids]

    def save(self, path: Path) -> None:
        """Сохранить tokenizer одним JSON-файлом."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path), pretty=True)

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        """Загрузить tokenizer без повторного обучения."""

        try:
            tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:
            raise TokenizerError(f"Не удалось загрузить tokenizer {path}: {error}") from error
        return cls(tokenizer)


def train_bpe_tokenizer(
    train_file: Path,
    output_file: Path = DEFAULT_TOKENIZER_PATH,
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> BPETokenizer:
    """Обучить byte-level BPE только на переданном train split и сохранить его."""

    if vocab_size < 262:
        raise ValueError("vocab_size должен вмещать 6 special tokens и byte alphabet")
    if min_frequency <= 0:
        raise ValueError("min_frequency должен быть положительным")
    if not train_file.is_file():
        raise TokenizerError(f"Train split не найден: {train_file}")

    backend = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    try:
        backend.train_from_iterator(iter_training_texts(train_file), trainer=trainer)
    except (DialogueFormatError, OSError) as error:
        raise TokenizerError(f"Не удалось обучить tokenizer: {error}") from error

    tokenizer = BPETokenizer(backend)
    tokenizer.save(output_file)
    return tokenizer


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI обучения BPE."""

    parser = argparse.ArgumentParser(description="Обучить byte-level BPE на train split.")
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/processed/splits/train.jsonl")
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить обучение tokenizer из командной строки."""

    args = build_argument_parser().parse_args(argv)
    try:
        tokenizer = train_bpe_tokenizer(
            args.train_file,
            args.output_file,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    except (TokenizerError, ValueError) as error:
        print(f"Tokenizer не обучен: {error}")
        return 1
    print(f"Tokenizer сохранён в {args.output_file}; vocabulary: {tokenizer.vocab_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
