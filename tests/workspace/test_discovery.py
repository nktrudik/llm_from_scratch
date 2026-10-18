"""Обновление каталога вручную добавленных датасетов без изменения корпусов."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_llm.api.app import workspace_catalog
from mini_llm.workspace import discovery
from mini_llm.workspace.catalog import fingerprint, get_dataset


def add_version(version: str, count: int) -> Path:
    """Добавить только samples.jsonl, как при ручном копировании пользователем."""

    source = Path("data/datasets/2ch") / version / "samples.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "board": "b",
            "thread_id": index,
            "context": [{"text": f"Вопрос {index}"}],
            "response": {"text": f"Ответ {index}"},
        }
        for index in range(count)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source


def test_refresh_registers_new_folder_once_and_persists_sample_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = add_version("v4", 3)
    original = source.read_bytes()
    assert discovery.dataset_catalog(refresh=False) == ([], [])
    records, warnings = discovery.dataset_catalog()
    assert not warnings
    assert records[0].samples == 3
    assert get_dataset("2ch", "v4").sha256 == fingerprint(source)
    assert source.read_bytes() == original
    assert not (source.parent / "splits").exists()

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("Повторное обновление не должно сканировать уже учтённый корпус")

    monkeypatch.setattr(discovery, "iter_dialogue_jsonl", forbidden)
    monkeypatch.setattr(discovery, "fingerprint", forbidden)
    assert discovery.dataset_catalog() == (records, [])


def test_api_refresh_updates_counts_and_ignores_broken_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    add_version("v3", 2)
    first = workspace_catalog()
    assert first["dataset_count"] == 1
    assert first["total_samples"] == 2
    add_version("v4", 3)
    assert workspace_catalog(refresh_datasets=False)["dataset_count"] == 1
    updated = workspace_catalog()
    assert updated["dataset_count"] == 2
    assert updated["total_samples"] == 5
    broken = add_version("v5", 1)
    broken.write_text("not json\n", encoding="utf-8")
    invalid = workspace_catalog()
    assert invalid["dataset_count"] == 2
    assert "v5" in str(invalid["dataset_warnings"])
    assert not (broken.parent / "dataset.json").exists()


def test_existing_manifest_and_splits_are_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = add_version("v4", 2)
    discovery.dataset_catalog()
    record = get_dataset("2ch", "v4")
    record.title = "Ручное название"
    record.save()
    splits = source.parent / "splits"
    splits.mkdir()
    (splits / "train.jsonl").write_bytes(b"unchanged")
    manifest = (source.parent / "dataset.json").read_bytes()
    records, warnings = discovery.dataset_catalog()
    assert records[0].title == "Ручное название"
    assert not warnings
    assert (source.parent / "dataset.json").read_bytes() == manifest
    assert (splits / "train.jsonl").read_bytes() == b"unchanged"
