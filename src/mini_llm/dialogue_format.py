"""Общий формат чтения и сериализации диалоговых samples."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# Это публичные маркеры vocabulary, а не секреты или пароли.
PAD_TOKEN = "<PAD>"  # nosec B105
UNK_TOKEN = "<UNK>"  # nosec B105
BOS_TOKEN = "<BOS>"  # nosec B105
EOS_TOKEN = "<EOS>"  # nosec B105
USER_TOKEN = "<USER>"  # nosec B105
ASSISTANT_TOKEN = "<ASSISTANT>"  # nosec B105

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
)
SPECIAL_TOKEN_IDS = {token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)}


class DialogueFormatError(ValueError):
    """Ошибка схемы или чтения dialogue JSONL."""


@dataclass(frozen=True, slots=True)
class DialogueSample:
    """Поля sample, необходимые следующим этапам pipeline."""

    board: str
    thread_id: int
    context: tuple[str, ...]
    response: str


def _as_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DialogueFormatError(f"{context}: ожидался JSON object")
    return cast(dict[str, object], value)


def parse_dialogue_sample(payload: object, context: str = "sample") -> DialogueSample:
    """Проверить минимальную схему и вернуть типизированное представление sample."""

    root = _as_object(payload, context)
    board = root.get("board")
    thread_id = root.get("thread_id")
    raw_context = root.get("context")
    response = _as_object(root.get("response"), f"{context}.response")
    response_text = response.get("text")
    if not isinstance(board, str) or not board:
        raise DialogueFormatError(f"{context}.board: ожидалась непустая строка")
    if not isinstance(thread_id, int) or isinstance(thread_id, bool):
        raise DialogueFormatError(f"{context}.thread_id: ожидалось целое число")
    if not isinstance(raw_context, list):
        raise DialogueFormatError(f"{context}.context: ожидался список")
    if not isinstance(response_text, str) or not response_text:
        raise DialogueFormatError(f"{context}.response.text: ожидалась непустая строка")

    messages: list[str] = []
    for index, raw_message in enumerate(cast(list[object], raw_context)):
        message = _as_object(raw_message, f"{context}.context[{index}]")
        text = message.get("text")
        if not isinstance(text, str) or not text:
            raise DialogueFormatError(f"{context}.context[{index}].text: ожидалась непустая строка")
        messages.append(text)
    if not messages:
        raise DialogueFormatError(f"{context}.context: нужен хотя бы один message")
    return DialogueSample(board, thread_id, tuple(messages), response_text)


def iter_dialogue_jsonl(path: Path) -> Iterator[tuple[dict[str, object], DialogueSample]]:
    """Потоково читать JSONL, сохраняя исходный object и проверенные поля."""

    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise DialogueFormatError(f"Не удалось открыть {path}: {error}") from error
    with source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                payload = cast(object, json.loads(line))
            except json.JSONDecodeError as error:
                raise DialogueFormatError(f"{context}: некорректный JSON: {error}") from error
            root = _as_object(payload, context)
            yield root, parse_dialogue_sample(root, context)


def context_role_tokens(message_count: int) -> tuple[str, ...]:
    """Назначить роли назад от последнего USER-сообщения перед response."""

    if message_count <= 0:
        raise ValueError("message_count должен быть положительным")
    return tuple(
        USER_TOKEN if (message_count - index) % 2 == 1 else ASSISTANT_TOKEN
        for index in range(message_count)
    )


def serialize_dialogue(sample: DialogueSample) -> str:
    """Сериализовать sample в единый role-token формат для tokenizer/training."""

    parts = [BOS_TOKEN]
    for role, text in zip(context_role_tokens(len(sample.context)), sample.context, strict=True):
        parts.extend((role, text))
    parts.extend((ASSISTANT_TOKEN, sample.response, EOS_TOKEN))
    return "".join(parts)


def iter_training_texts(path: Path) -> Iterator[str]:
    """Отдать тексты только из указанного split для обучения BPE."""

    for _, sample in iter_dialogue_jsonl(path):
        yield from sample.context
        yield sample.response


def count_jsonl_records(path: Path) -> int:
    """Посчитать непустые строки для progress bar без загрузки файла в RAM."""

    try:
        with path.open(encoding="utf-8") as source:
            return sum(1 for line in source if line.strip())
    except OSError as error:
        raise DialogueFormatError(f"Не удалось прочитать {path}: {error}") from error


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Атомарно записать небольшой JSON-отчёт."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def percentile(ordered_values: Sequence[int], percentile_value: float) -> int:
    """Вернуть дискретный percentile методом nearest rank."""

    if not ordered_values:
        return 0
    index = max(0, int(len(ordered_values) * percentile_value + 0.999999) - 1)
    return ordered_values[min(index, len(ordered_values) - 1)]
