"""Версии ChatML проходят существующую публикацию и split без изменения исходников."""

import json
from pathlib import Path

import pytest

from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.workspace.datasets import publish_dataset
from mini_llm.workspace.training import prepare_version_splits, validate_splits


def test_chatml_thread_split_preserves_system_and_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [
        {
            "messages": [
                {"role": "system", "content": " Система\n"},
                {"role": "user", "content": f"Вопрос {thread} 😄  "},
                {"role": "assistant", "content": f"\nОтвет {reply}"},
            ],
            "metadata": {"board": "b", "thread_id": thread},
        }
        for thread in range(20)
        for reply in range(2)
    ]
    source = tmp_path / "chatml.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    original = source.read_bytes()
    dataset = publish_dataset(source, "chatml", "v1", "ChatML")
    prepare_version_splits("chatml", "v1")
    hashes = validate_splits(dataset)
    assert set(hashes) == {"train", "validation", "test"}
    sets = [
        {
            sample.thread_id
            for _, sample in iter_dialogue_jsonl(dataset.directory / "splits" / f"{split}.jsonl")
        }
        for split in ("train", "validation", "test")
    ]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sum(len(group) for group in sets) == 20
    assert source.read_bytes() == original
    assert (dataset.directory / "samples.jsonl").read_bytes() == original
