"""Тесты детерминированного split по тредам."""

import json
from pathlib import Path
from typing import cast

from mini_llm.data.splitting import SPLIT_NAMES, SplitConfig, split_dataset


def _sample(board: str, thread_id: int, response_id: int) -> dict[str, object]:
    return {
        "board": board,
        "thread_id": thread_id,
        "source_post_ids": [thread_id, response_id],
        "context": [{"post_id": thread_id, "text": f"Вопрос треда {thread_id}"}],
        "response": {"post_id": response_id, "text": f"Ответ {response_id}"},
        "metadata": {"parent_strategy": "explicit_references"},
    }


def test_split_has_no_thread_leakage_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "dialogues.jsonl"
    with source.open("w", encoding="utf-8") as output:
        for thread_id in range(20):
            for offset in range(2):
                output.write(
                    json.dumps(_sample("b", thread_id, thread_id * 10 + offset), ensure_ascii=False)
                    + "\n"
                )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    report = split_dataset(SplitConfig(source, first_dir, random_seed=17))
    split_dataset(SplitConfig(source, second_dir, random_seed=17))

    thread_splits: dict[tuple[str, int], set[str]] = {}
    for split in SPLIT_NAMES:
        assert (first_dir / f"{split}.jsonl").read_bytes() == (
            second_dir / f"{split}.jsonl"
        ).read_bytes()
        for line in (first_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            payload = cast(dict[str, object], json.loads(line))
            key = (cast(str, payload["board"]), cast(int, payload["thread_id"]))
            thread_splits.setdefault(key, set()).add(split)

    assert all(len(splits) == 1 for splits in thread_splits.values())
    split_report = cast(dict[str, dict[str, object]], report["splits"])
    assert [split_report[name]["threads"] for name in SPLIT_NAMES] == [18, 1, 1]
    assert [split_report[name]["samples"] for name in SPLIT_NAMES] == [36, 2, 2]
