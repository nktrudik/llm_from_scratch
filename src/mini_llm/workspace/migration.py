"""Одноразовая переносимая миграция старого хранилища без перезаписи весов и JSONL."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch

from mini_llm.data.dialogue import write_json
from mini_llm.observability import terminal_log
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.workspace import ModelPaths, ModelRegistration
from mini_llm.workspace.catalog import DatasetVersion, ModelRun, contained, fingerprint, get_dataset
from mini_llm.workspace.paths import DATASETS_ROOT, RUNS_ROOT

JOURNAL = Path("data/archive/layout-migration.json")


def _move(source: Path, destination: Path, moves: list[dict[str, str]]) -> None:
    """Проверить оба абсолютных пути до rename; никогда не заменять существующую цель."""

    root = Path.cwd().resolve()
    contained(source, root)
    contained(destination, root)
    if not source.exists():
        return
    if destination.exists():
        raise ValueError(f"Миграция не перезапишет {destination}; исходник: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    moves.append({"from": source.as_posix(), "to": destination.as_posix()})
    write_json(JOURNAL, {"status": "in_progress", "moves": moves})
    terminal_log("MIGRATION", f"{source} -> {destination}")


def _import_run(source: Path, version: str, number: int, moves: list[dict[str, str]]) -> None:
    checkpoint = source / "last.pt"
    if not checkpoint.is_file():
        return
    payload = cast(dict[str, object], torch.load(checkpoint, map_location="cpu", weights_only=True))
    metadata = payload.get("model_metadata")
    pretrained = metadata.get("pretrained_config") if isinstance(metadata, dict) else None
    config = PretrainedConfig.model_validate(pretrained) if isinstance(pretrained, dict) else None
    model_id = config.model_id if config else "custom"
    slug = ModelPaths.for_model(model_id).config_file.parent.name if config else "custom"
    dataset_id = "2ch" if version != "v0" else "legacy"
    dataset_title = get_dataset(dataset_id, version).title if version != "v0" else "Старый корпус"
    directory = RUNS_ROOT / slug / dataset_id / version / f"run-{number:03d}"
    _move(source, directory, moves)
    run = ModelRun(
        run_id=f"{slug}--{dataset_id}-{version}--run-{number:03d}",
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_version=version,
        directory=directory,
        title=f"{model_id} · {dataset_id}/{version} — {dataset_title} · run-{number:03d}",
        source=source.as_posix(),
        global_step=int(str(payload.get("global_step", 0))),
        created_at=datetime.fromtimestamp((directory / "last.pt").stat().st_mtime, UTC).isoformat(),
        training_config=cast(dict[str, object], payload.get("training_config", {})),
    )
    if config:
        run.pretrained_config_file = directory / "pretrained.json"
        config.save(run.pretrained_config_file)
    else:
        run.tokenizer_file = Path("artifacts/tokenizers/custom-bpe/v1/tokenizer.json")
    run.save()


def migrate_workspace() -> dict[str, object]:
    """Перенести известные версии и запуски; исторические .pt не модифицируются."""

    if JOURNAL.is_file():
        previous = json.loads(JOURNAL.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            return cast(dict[str, object], previous)
        raise ValueError(
            f"Миграция ранее прервалась; проверьте журнал {JOURNAL} перед продолжением"
        )
    for target in (DATASETS_ROOT, RUNS_ROOT, Path("artifacts/models"), Path("configs/models")):
        contained(target, Path.cwd())
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Перед миграцией целевой каталог должен быть пуст: {target}")
    moves: list[dict[str, str]] = []
    definitions = (
        ("v1", "2ch_dialogues.jsonl", "Исходные SFT-пары", "splits"),
        ("v2", "2ch_dialogues_qa_full.jsonl", "QA Full: переработанные пары", None),
        (
            "v3",
            "2ch_dialogues_qa_candidates.jsonl",
            "QA Candidates: отобранные пары",
            "splits_qa_candidates",
        ),
    )
    for version, filename, title, splits in definitions:
        source = Path("data/processed") / filename
        if not source.is_file():
            continue
        directory = DATASETS_ROOT / "2ch" / version
        sha256 = fingerprint(source)
        with source.open(encoding="utf-8") as stream:
            count = sum(bool(line.strip()) for line in stream)
        destination = directory / "samples.jsonl"
        _move(source, destination, moves)
        if fingerprint(destination) != sha256:
            raise RuntimeError("Контрольная сумма корпуса изменилась")
        DatasetVersion(
            dataset_id="2ch",
            version=version,
            title=title,
            samples=count,
            sha256=sha256,
            source=filename,
        ).save()
        if splits:
            _move(Path("data/processed") / splits, directory / "splits", moves)
            stats_path = directory / "splits/split_stats.json"
            if stats_path.is_file():
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                stats["source_file"] = destination.as_posix()
                for name, details in stats["splits"].items():
                    details["file"] = (directory / "splits" / f"{name}.jsonl").as_posix()
                write_json(stats_path, stats)

    roots = (
        ("checkpoints/experiments/qwen3_qa_v1", "v3", 1),
        ("checkpoints/experiments/qwen3_qa_v1_resume_1epoch_20261017_085135_199", "v3", 2),
        ("checkpoints/experiments/qwen3_qa_epoch1_v2", "v3", 3),
        ("checkpoints/pretrained/model--qwen--qwen3-0.6b/qlora", "v1", 1),
        (
            "checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora_clean_20261002_201738_776",
            "v1",
            1,
        ),
        ("checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora", "v0", 1),
        ("checkpoints/training", "v0", 1),
    )
    for run_source, version, number in roots:
        _import_run(Path(run_source), version, number, moves)

    for old in Path("configs/pretrained").glob("*.json"):
        config = PretrainedConfig.load(old)
        _move(old, ModelPaths.for_model(config.model_id, config.adaptation_mode).config_file, moves)
    _move(Path("artifacts/pretrained"), Path("artifacts/models"), moves)
    for path in Path("artifacts/models").rglob("*.json"):
        registration = ModelRegistration.model_validate_json(path.read_text(encoding="utf-8"))
        registration.config_file = ModelPaths.for_model(registration.model_id).config_file
        slug = registration.config_file.parent.name
        # Каноническая legacy-ссылка сохраняется; новые запуски выбираются по run.json.
        version = "v1" if "qwen3" in slug else "v0"
        dataset_id = "2ch" if version == "v1" else "legacy"
        registration.checkpoint_dir = RUNS_ROOT / slug / dataset_id / version / "run-001"
        registration.save(path)
    _move(
        Path("artifacts/tokenizer/2ch_bpe.json"),
        Path("artifacts/tokenizers/custom-bpe/v1/tokenizer.json"),
        moves,
    )
    _move(Path("data/raw/2ch"), Path("data/sources/2ch/threads"), moves)
    _move(Path("data/raw/2ch_clean"), Path("data/sources/2ch/sft-pairs"), moves)
    _move(Path("data/processed/reports"), Path("data/reports/2ch"), moves)
    _move(
        Path("data/processed/backups/before_sft_20261002_200348"),
        Path("data/archive/legacy-thread-dialogues"),
        moves,
    )
    # Неизвестные остатки сохраняются в архиве, а не удаляются.
    for remaining_source, remaining_target in (
        ("data/processed", "data/archive/processed-remainder"),
        ("data/raw", "data/archive/raw-remainder"),
        ("checkpoints/experiments", "checkpoints/archive/experiments-remainder"),
        ("checkpoints/pretrained", "checkpoints/archive/pretrained-remainder"),
        ("checkpoints/.vscode", "checkpoints/archive/editor-settings"),
        ("configs/pretrained", "configs/archive/pretrained-remainder"),
        ("artifacts/tokenizer", "artifacts/archive/tokenizer-remainder"),
    ):
        _move(Path(remaining_source), Path(remaining_target), moves)
    report: dict[str, object] = {"status": "completed", "moves": moves}
    write_json(JOURNAL, report)
    return report
