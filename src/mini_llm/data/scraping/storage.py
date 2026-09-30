"""Неизменяемое сохранение raw-тредов на диск."""

from __future__ import annotations

import json
from pathlib import Path

from mini_llm.data.scraping.schemas import ThreadDocument


def thread_output_path(output_dir: Path, board: str, thread_id: int) -> Path:
    """Вернуть канонический путь raw-файла одного треда."""

    return output_dir / "2ch" / board / f"{thread_id}.json"


def save_thread_document(document: ThreadDocument, output_dir: Path) -> Path | None:
    """Сохранить новый тред или вернуть ``None``, если файл уже существует."""

    output_path = thread_output_path(output_dir, document.board, document.thread_id)
    if output_path.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path
