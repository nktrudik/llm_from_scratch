"""Управление моделями на крошечных временных файлах; никаких HF-загрузок."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.routing import APIRoute

from mini_llm.api import models as routes
from mini_llm.api.jobs import JobAlreadyRunningError, JobManager
from mini_llm.api.schemas import JobResponse
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.management import service, storage
from mini_llm.pretrained.management.catalog import models_catalog
from mini_llm.pretrained.management.schemas import DeletionRequest, DeletionTarget, DownloadRequest
from mini_llm.pretrained.workspace import (
    ACTIVE_MODEL_FILE,
    ModelPaths,
    ModelRegistration,
    load_active_model,
)
from mini_llm.workspace.catalog import ModelRun, list_runs


def make_model(model_id: str, *, active: bool = True) -> ModelRun:
    """Сохранить только конфиг, регистрацию и фиктивные байты файлов."""

    paths = ModelPaths.for_model(model_id)
    config = PretrainedConfig(model_id=model_id, adaptation_mode="qlora", revision="a" * 40)
    config.save(paths.config_file)
    cache = Path(".cache/huggingface") / ("models--" + model_id.replace("/", "--"))
    snapshot = cache / "snapshots" / config.revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"synthetic-base-weights")
    directory = Path("checkpoints/runs") / paths.config_file.parent.name / "2ch/v1/run-001"
    run = ModelRun(
        run_id=paths.config_file.parent.name + "--v1--run-001",
        model_id=model_id,
        dataset_id="2ch",
        dataset_version="v1",
        directory=directory,
        title=model_id,
        pretrained_config_file=directory / "pretrained.json",
    )
    run.save()
    config.save(directory / "pretrained.json")
    for name in ("last.pt", "best.pt", "step_00000100.pt"):
        (directory / name).write_bytes(name.encode())
    registration = ModelRegistration(
        model_id=model_id,
        revision=config.revision,
        config_file=paths.config_file,
        checkpoint_dir=directory,
        snapshot_path=snapshot,
    )
    registration.save(paths.registration_file)
    if active:
        registration.save(ACTIVE_MODEL_FILE)
    return run


@pytest.fixture
def local_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRun:
    pytest.importorskip("huggingface_hub")
    monkeypatch.chdir(tmp_path)
    return make_model("org/model")


def confirmed(target: DeletionTarget) -> DeletionRequest:
    return DeletionRequest(
        target=target,
        preview_token=storage.deletion_preview(target).preview_token,
        confirmation="УДАЛИТЬ",
    )


def test_catalog_and_preview_do_not_open_weights(
    local_model: ModelRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("Каталог не должен читать веса")

    monkeypatch.setattr(torch, "load", forbidden)
    model = next(item for item in models_catalog().models if item.model_id == "org/model")
    assert model.registered and model.available
    assert len(model.runs) == 1 and model.size_bytes > 0
    assert "last.pt" in model.runs[0].checkpoints
    preview = storage.deletion_preview(
        DeletionTarget(scope="model", model_id="org/model", include_cache=True)
    )
    assert any(".cache/huggingface/models--org--model" in path.path for path in preview.paths)
    assert all("data/" not in path.path for path in preview.paths)


def test_checkpoint_removal_keeps_other_weights_and_base_model(local_model: ModelRun) -> None:
    target = DeletionTarget(
        scope="checkpoint", run_id=local_model.run_id, checkpoint_name="step_00000100.pt"
    )
    service.delete_model_files(confirmed(target), lambda _: None)
    assert not (local_model.directory / "step_00000100.pt").exists()
    assert (local_model.directory / "last.pt").read_bytes() == b"last.pt"
    assert load_active_model() is not None
    assert list_runs()[0].run_id == local_model.run_id


def test_run_removal_keeps_base_config_and_repairs_legacy_pointer(local_model: ModelRun) -> None:
    service.delete_model_files(
        confirmed(DeletionTarget(scope="run", run_id=local_model.run_id)), lambda _: None
    )
    assert not local_model.directory.exists()
    registration = load_active_model()
    assert registration is not None and registration.snapshot_path.is_dir()
    assert registration.config_file.is_file()
    assert registration.checkpoint_dir == ModelPaths.for_model("org/model").checkpoint_dir


def test_full_model_removal_preserves_other_models_and_dataset(local_model: ModelRun) -> None:
    other = make_model("org/other", active=False)
    dataset = Path("data/datasets/protected.jsonl")
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"protected dataset")
    service.delete_model_files(
        confirmed(DeletionTarget(scope="model", model_id="org/model", include_cache=True)),
        lambda _: None,
    )
    paths = ModelPaths.for_model("org/model")
    assert not paths.config_file.parent.exists()
    assert not paths.registration_file.parent.exists()
    assert not local_model.directory.exists()
    assert not Path(".cache/huggingface/models--org--model").exists()
    assert (other.directory / "last.pt").read_bytes() == b"last.pt"
    assert dataset.read_bytes() == b"protected dataset"
    active = load_active_model()
    assert active is not None and active.model_id == "org/other"


def test_stale_preview_is_rejected_without_removing_anything(local_model: ModelRun) -> None:
    request = confirmed(DeletionTarget(scope="run", run_id=local_model.run_id))
    (local_model.directory / "last.pt").write_bytes(b"changed after preview")
    with pytest.raises(ValueError, match="изменились"):
        service.delete_model_files(request, lambda _: None)
    assert local_model.directory.is_dir()
    assert (local_model.directory / "last.pt").read_bytes() == b"changed after preview"


def test_model_can_be_removed_without_deleting_cache(local_model: ModelRun) -> None:
    service.delete_model_files(
        confirmed(DeletionTarget(scope="model", model_id="org/model")), lambda _: None
    )
    assert Path(".cache/huggingface/models--org--model").exists()
    orphan = next(item for item in models_catalog().models if item.model_id == "org/model")
    assert not orphan.registered and orphan.size_bytes > 0
    service.delete_model_files(
        confirmed(DeletionTarget(scope="model", model_id="org/model", include_cache=True)),
        lambda _: None,
    )
    assert not Path(".cache/huggingface/models--org--model").exists()


def test_preview_cannot_target_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    protected = tmp_path / "protected.txt"
    protected.write_bytes(b"keep")
    monkeypatch.chdir(project)
    monkeypatch.setattr(storage, "target_paths", lambda _: [(protected, tmp_path)])
    with pytest.raises(ValueError, match="внутри"):
        storage.deletion_preview(DeletionTarget(scope="model", model_id="custom"))
    assert protected.read_bytes() == b"keep"


def test_generation_and_background_mutation_are_mutually_exclusive() -> None:
    manager = JobManager()
    with manager.generation(), pytest.raises(JobAlreadyRunningError, match="генерация"):
        manager.create("model_delete")
    manager.create("training")
    with pytest.raises(JobAlreadyRunningError), manager.generation():
        pytest.fail("Генерация не должна начаться во время training")


def test_download_route_uses_setup_only_after_explicit_job_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []

    def setup(model_id: str, *, on_stage: Callable[[str], None]) -> ModelRegistration:
        calls.append(model_id)
        on_stage("Проверка synthetic tokenizer")
        make_model(model_id)
        registration = load_active_model()
        assert registration is not None
        return registration

    monkeypatch.setattr(service, "setup_pretrained_model", setup)
    monkeypatch.setattr(routes, "clear_pretrained_cache", lambda: None)
    manager = JobManager()
    router = routes.create_model_router(manager)
    route = next(
        item
        for item in router.routes
        if isinstance(item, APIRoute) and item.path.endswith("/download")
    )
    endpoint = cast(Callable[[DownloadRequest, BackgroundTasks], JobResponse], route.endpoint)
    tasks = BackgroundTasks()
    response = endpoint(DownloadRequest(model_id="org/new"), tasks)
    assert response.status == "queued" and not calls
    asyncio.run(tasks())
    assert calls == ["org/new"]
    snapshot = manager.get(response.job_id)
    assert snapshot is not None and snapshot["status"] == "completed"
    assert snapshot["operation_progress"] is not None


def test_delete_route_rejects_active_training(local_model: ModelRun) -> None:
    manager = JobManager()
    manager.create("training")
    router = routes.create_model_router(manager)
    route = next(
        item
        for item in router.routes
        if isinstance(item, APIRoute) and item.path.endswith("/delete")
    )
    endpoint = cast(Callable[[DeletionRequest, BackgroundTasks], JobResponse], route.endpoint)
    with pytest.raises(HTTPException) as error:
        endpoint(
            confirmed(DeletionTarget(scope="run", run_id=local_model.run_id)), BackgroundTasks()
        )
    assert error.value.status_code == 409
    assert local_model.directory.is_dir()


def test_failed_metadata_update_rolls_back_files_and_registrations(
    local_model: ModelRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = confirmed(DeletionTarget(scope="run", run_id=local_model.run_id))
    before = ACTIVE_MODEL_FILE.read_bytes()

    def fail(*args: object) -> None:
        ACTIVE_MODEL_FILE.write_text("changed", encoding="utf-8")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(service, "_repair_registrations", fail)
    with pytest.raises(RuntimeError, match="отменено"):
        service.delete_model_files(request, lambda _: None)
    assert (local_model.directory / "last.pt").read_bytes() == b"last.pt"
    assert ACTIVE_MODEL_FILE.read_bytes() == before


def test_running_cli_run_and_shared_cache_cannot_be_deleted(local_model: ModelRun) -> None:
    local_model.status = "running"
    local_model.save()
    with pytest.raises(ValueError, match="выполняющийся"):
        storage.deletion_preview(DeletionTarget(scope="model", model_id="org/model"))
    local_model.status = "completed"
    local_model.save()
    make_model("org/other", active=False)
    registration_path = ModelPaths.for_model("org/other").registration_file
    other = ModelRegistration.model_validate_json(registration_path.read_text(encoding="utf-8"))
    active = load_active_model()
    assert active is not None
    other.snapshot_path = active.snapshot_path
    other.save(registration_path)
    with pytest.raises(ValueError, match="также используется"):
        storage.deletion_preview(
            DeletionTarget(scope="model", model_id="org/model", include_cache=True)
        )


def test_directory_links_and_invalid_model_ids_are_rejected(
    local_model: ModelRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.is_junction
    monkeypatch.setattr(
        Path, "is_junction", lambda path: path == local_model.directory or original(path)
    )
    with pytest.raises(ValueError, match="ссылками"):
        storage.deletion_preview(DeletionTarget(scope="run", run_id=local_model.run_id))
    with pytest.raises(ValueError):
        DownloadRequest(model_id="../outside")
    with pytest.raises(ValueError, match="зарезервирован"):
        DownloadRequest(model_id="custom")
    with pytest.raises(ValueError):
        DeletionRequest.model_validate(
            {
                "target": {"scope": "run", "run_id": local_model.run_id},
                "preview_token": "a" * 64,
                "confirmation": "нет",
            }
        )


def test_shared_hf_blobs_used_by_other_model_survive_cleanup(local_model: ModelRun) -> None:
    """Временный cache повторяет текущий shared-store layout HF, без настоящих весов."""

    cache_root = Path(".cache/huggingface")
    from huggingface_hub import scan_cache_dir

    if not hasattr(scan_cache_dir(cache_root), "cache_dir"):
        pytest.skip("Эта версия Hub не использует общий blob-store")
    make_model("org/other", active=False)
    shared = cache_root / "blobs/ff" / ("f" * 64)
    unique = cache_root / "blobs/ee" / ("e" * 64)
    for path in (shared, unique):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic shared store")
    (cache_root / "blobs/.huggingface-shared-blobs").write_text("1\n", encoding="utf-8")
    links: list[Path] = []
    try:
        for repo_name in ("models--org--model", "models--org--other"):
            repo = cache_root / repo_name
            blob = repo / "blobs/shared"
            blob.parent.mkdir(parents=True)
            blob.symlink_to(os.path.relpath(shared, blob.parent))
            links.append(blob)
            snapshot_file = repo / "snapshots" / ("a" * 40) / "model.safetensors"
            snapshot_file.unlink()
            snapshot_file.symlink_to(os.path.relpath(blob, snapshot_file.parent))
        unique_link = cache_root / "models--org--model/blobs/unique"
        unique_link.symlink_to(os.path.relpath(unique, unique_link.parent))
        unique_snapshot = unique_link.parent.parent / "snapshots" / ("a" * 40) / "extra.safetensors"
        unique_snapshot.symlink_to(os.path.relpath(unique_link, unique_snapshot.parent))
    except OSError:
        pytest.skip("ОС не разрешает создание файловых symlinks")
    shared.with_suffix(".refs").write_text(
        "".join(path.relative_to(cache_root).as_posix() + "\n" for path in links), encoding="utf-8"
    )
    unique.with_suffix(".refs").write_text(
        unique_link.relative_to(cache_root).as_posix() + "\n", encoding="utf-8"
    )
    service.delete_model_files(
        confirmed(DeletionTarget(scope="model", model_id="org/model", include_cache=True)),
        lambda _: None,
    )
    assert shared.read_bytes() == b"synthetic shared store"
    assert links[1].read_bytes() == b"synthetic shared store"
    assert not unique.exists()
