"""Проверка ChatML JSONL без изменения содержимого сообщений."""

import hashlib
import json
from typing import cast


def parse_messages(payload: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """Принять необязательный system и чередующиеся user/assistant, завершая ответом."""

    raw = payload.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("messages должен содержать вопрос и ответ")
    messages: list[tuple[str, str]] = []
    expected = "user"
    for index, value in enumerate(cast(list[object], raw)):
        if not isinstance(value, dict):
            raise ValueError("Каждое сообщение должно быть объектом role/content")
        role, content = value.get("role"), value.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content должен быть непустой строкой")
        if role == "system" and index == 0:
            messages.append(("system", content))
            continue
        if role != expected:
            raise ValueError("После system нужны чередующиеся user/assistant сообщения")
        messages.append((expected, content))
        expected = "assistant" if expected == "user" else "user"
    if expected != "user" or len(messages) < 2:
        raise ValueError("Последнее сообщение должно иметь роль assistant")
    return tuple(messages)


def conversation_group(payload: dict[str, object]) -> tuple[str, int]:
    """Сохранить thread grouping; без thread_id группировать одинаковые разговоры."""

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    board = payload.get("board", metadata.get("board", "chatml"))
    thread_id = payload.get("thread_id", metadata.get("thread_id"))
    if not isinstance(board, str) or not board:
        raise ValueError("board должен быть непустой строкой")
    if thread_id is None:
        serialized = json.dumps(parse_messages(payload), ensure_ascii=False).encode("utf-8")
        thread_id = int.from_bytes(hashlib.sha256(serialized).digest()[:8])
    if not isinstance(thread_id, int) or isinstance(thread_id, bool):
        raise ValueError("thread_id должен быть целым числом")
    return board, thread_id
