"""Чтение и проверка raw JSON-файлов 2ch."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from mini_llm.data.preprocessing.schemas import PreprocessingError, RawPost, RawThread

REFERENCE_PATTERN = re.compile(r">>(\d+)")


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PreprocessingError(f"{context}: ожидался JSON-объект")
    return cast(dict[str, object], value)


def _require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise PreprocessingError(f"{context}: ожидался список")
    return cast(list[object], value)


def _require_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PreprocessingError(f"{context}: ожидалось целое число")
    return value


def load_raw_thread(path: Path, input_dir: Path) -> RawThread:
    """Прочитать и проверить один raw-тред, не изменяя исходный файл."""

    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreprocessingError(f"Не удалось прочитать {path}: {error}") from error
    root = _require_object(payload, str(path))
    board = root.get("board")
    if not isinstance(board, str) or not board:
        raise PreprocessingError(f"{path}: board должен быть непустой строкой")
    thread_id = _require_int(root.get("thread_id"), f"{path}: thread_id")

    posts: list[RawPost] = []
    for index, raw_post in enumerate(_require_list(root.get("posts"), f"{path}: posts")):
        post = _require_object(raw_post, f"{path}: posts[{index}]")
        post_id = _require_int(post.get("post_id"), f"{path}: posts[{index}].post_id")
        text = post.get("text")
        if not isinstance(text, str):
            raise PreprocessingError(f"{path}: posts[{index}].text должен быть строкой")
        raw_references = _require_list(post.get("references"), f"{path}: posts[{index}].references")
        references = tuple(
            int(match.group(1))
            for value in raw_references
            if isinstance(value, str) and (match := REFERENCE_PATTERN.fullmatch(value))
        )
        posts.append(RawPost(post_id, text, tuple(dict.fromkeys(references))))

    return RawThread(board, thread_id, path.relative_to(input_dir).as_posix(), tuple(posts))
