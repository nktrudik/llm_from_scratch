"""Версии, происхождение данных и безопасное выделение новых запусков."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mini_llm.api.schemas import GenerationRequest
from mini_llm.inference.experiments import list_experiments
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.workspace.catalog import get_dataset, get_run, list_datasets, list_runs, reserve_run
from mini_llm.workspace.datasets import publish_dataset
from mini_llm.workspace.migration import _move, migrate_workspace
from mini_llm.workspace.training import validate_splits


def _sample(thread_id: int, question: str = " Вопрос\n🙂 ") -> str:
    return (
        json.dumps(
            {
                "board": "b",
                "thread_id": thread_id,
                "context": [{"post_id": thread_id, "text": question}],
                "response": {"post_id": thread_id + 1, "text": "Ответ\n\n!"},
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def test_publication_preserves_bytes_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_bytes((_sample(1) + _sample(2)).encode())
    record = publish_dataset(source, "2ch", "v1", "Корпус")
    assert record.samples == 2
    assert (record.directory / "samples.jsonl").read_bytes() == source.read_bytes()
    assert list_datasets() == [get_dataset("2ch", "v1")]
    with pytest.raises(FileExistsError):
        publish_dataset(source, "2ch", "v1", "Не заменить")
    with pytest.raises(ValueError):
        publish_dataset(source, "../escape", "v2", "Ошибка")
    with pytest.raises(ValueError):
        publish_dataset(source, "2ch", "other", "Ошибка")


def test_run_numbers_isolate_model_dataset_and_previous_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(_sample(1), encoding="utf-8")
    dataset = publish_dataset(source, "2ch", "v1", "Пары")
    first = reserve_run("custom", "custom", dataset)
    first.save()
    (first.directory / "last.pt").write_bytes(b"unchanged")
    second = reserve_run("custom", "custom", dataset)
    second.save()
    assert first.directory.name == "run-001"
    assert second.directory.name == "run-002"
    assert get_run(first.run_id) == first
    assert len(list_runs()) == 2
    assert (first.directory / "last.pt").read_bytes() == b"unchanged"


def test_split_validation_rejects_leakage_and_foreign_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(_sample(1) + _sample(1, "Другой вопрос"), encoding="utf-8")
    dataset = publish_dataset(source, "2ch", "v1", "Пары")
    splits = dataset.directory / "splits"
    splits.mkdir()
    (splits / "train.jsonl").write_text(_sample(1), encoding="utf-8")
    (splits / "validation.jsonl").write_text(_sample(1, "Другой вопрос"), encoding="utf-8")
    (splits / "test.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(dataset)
    (splits / "validation.jsonl").write_text(_sample(2), encoding="utf-8")
    with pytest.raises(ValueError, match="не из выбранной"):
        validate_splits(dataset)


def test_migration_preserves_dataset_and_checkpoint_bytes_and_is_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    processed = Path("data/processed")
    processed.mkdir(parents=True)
    original = _sample(1).encode()
    (processed / "2ch_dialogues_qa_candidates.jsonl").write_bytes(original)
    experiment = Path("checkpoints/experiments/qwen3_qa_v1")
    experiment.mkdir(parents=True)
    config = PretrainedConfig(model_id="Qwen/Qwen3-0.6B", adaptation_mode="qlora")
    torch.save(
        {
            "global_step": 200,
            "training_config": {"batch_size": 2},
            "model_metadata": {
                "backend": "pretrained",
                "pretrained_config": config.model_dump(mode="json"),
            },
            "model_state_dict": {"weight": torch.ones(1)},
        },
        experiment / "last.pt",
    )
    weights = (experiment / "last.pt").read_bytes()
    report = migrate_workspace()
    assert report["status"] == "completed"
    assert migrate_workspace() == report
    dataset = get_dataset("2ch", "v3")
    assert (dataset.directory / "samples.jsonl").read_bytes() == original
    run = list_runs()[0]
    assert (run.directory / "last.pt").read_bytes() == weights
    assert run.pretrained_config_file is not None
    assert PretrainedConfig.load(run.pretrained_config_file) == config
    assert list_experiments(config)[0].experiment_id == run.run_id


def test_move_refuses_overwrite_and_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    with pytest.raises(ValueError, match="не перезапишет"):
        _move(source, target, [])
    with pytest.raises(ValueError, match="внутри"):
        _move(source, tmp_path.parent / "outside", [])
    assert source.is_dir()


def test_dataset_versions_sort_numerically_and_splits_are_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mini_llm.workspace.training import prepare_version_splits

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(_sample(1), encoding="utf-8")
    publish_dataset(source, "2ch", "v10", "Новая версия")
    older = publish_dataset(source, "2ch", "v2", "Старая версия")
    assert [item.version for item in list_datasets()] == ["v2", "v10"]
    splits = older.directory / "splits"
    splits.mkdir()
    (splits / "train.jsonl").write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="перезапись"):
        prepare_version_splits("2ch", "v2")
    assert (splits / "train.jsonl").read_bytes() == b"unchanged"


def test_custom_inference_uses_selected_run_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.jsonl"
    source.write_text(_sample(1), encoding="utf-8")
    dataset = publish_dataset(source, "2ch", "v1", "Пары")
    run = reserve_run("custom", "custom", dataset)
    run.tokenizer_file = Path("artifacts/tokenizers/test/v1/tokenizer.json")
    run.save()
    (run.directory / "best.pt").write_bytes(b"not-loaded")
    config = GenerationRequest(
        prompt="Вопрос",
        model_backend="custom",
        experiment_id=run.run_id,
        checkpoint_name="best.pt",
    ).to_config()
    assert config.checkpoint_file == run.directory / "best.pt"
    assert config.tokenizer_file == run.tokenizer_file
