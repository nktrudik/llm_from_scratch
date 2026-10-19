"""Явная загрузка и подтверждённая очистка принадлежащих модели файлов."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from mini_llm.data.dialogue import write_json
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.management.hf_cache import CacheDeletion, cache_deletion, is_repo
from mini_llm.pretrained.management.schemas import DeletionRequest, DownloadRequest
from mini_llm.pretrained.management.storage import CACHE_ROOT, deletion_preview
from mini_llm.pretrained.setup import setup_pretrained_model
from mini_llm.pretrained.workspace import (
    ACTIVE_MODEL_FILE,
    ModelPaths,
    ModelRegistration,
    list_registered_models,
    load_active_model,
)
from mini_llm.workspace.catalog import contained, get_run


def download_model(request: DownloadRequest, on_stage: Callable[[str], None]) -> dict[str, object]:
    """Переиспользовать setup; скачать файлы, но не создавать модель в GPU и не обучать."""

    model_id = next(
        (
            item.model_id
            for item in list_registered_models()
            if item.model_id.casefold() == request.model_id.casefold()
        ),
        request.model_id,
    )
    registration = setup_pretrained_model(model_id, on_stage=on_stage)
    return registration.model_dump(mode="json")


def _repair_registrations(request: DeletionRequest, removed_run_dir: Path | None) -> None:
    """Не оставлять active.json и legacy best-ссылки на удалённые файлы."""

    target = request.target
    active = load_active_model()
    if (
        target.scope == "model"
        and active
        and active.model_id.casefold() == (target.model_id or "").casefold()
    ):
        ACTIVE_MODEL_FILE.unlink()
        remaining = list_registered_models()
        if remaining:
            remaining[0].save(ACTIVE_MODEL_FILE)
    if removed_run_dir is not None:
        paths = list(ACTIVE_MODEL_FILE.parent.glob("model--*/*.json"))
        if ACTIVE_MODEL_FILE.is_file():
            paths.append(ACTIVE_MODEL_FILE)
        for path in paths:
            registration = ModelRegistration.model_validate_json(path.read_text(encoding="utf-8"))
            if registration.checkpoint_dir.resolve() == removed_run_dir.resolve():
                config = PretrainedConfig.load(registration.config_file)
                registration.checkpoint_dir = ModelPaths.for_model(
                    config.model_id, config.adaptation_mode
                ).checkpoint_dir
                registration.save(path)
    if target.scope == "checkpoint":
        run = get_run(target.run_id or "")
        if run:
            run.checkpoints.pop(target.checkpoint_name or "", None)
            run.save()


def delete_model_files(
    request: DeletionRequest, on_stage: Callable[[str], None]
) -> dict[str, object]:
    """Проверить план повторно, переместить с rollback, затем необратимо удалить своё staging."""

    preview = deletion_preview(request.target)
    if preview.preview_token != request.preview_token:
        raise ValueError(
            "Файлы изменились после просмотра. Обновите план и подтвердите удаление заново."
        )
    removed_run = get_run(request.target.run_id or "") if request.target.scope == "run" else None
    root = contained(Path(".cache/model-cleanup"), Path.cwd())
    root.mkdir(parents=True, exist_ok=True)
    staging = contained(Path(tempfile.mkdtemp(prefix="delete-", dir=root)), root)
    moved: list[tuple[Path, Path]] = []
    cache_plans: list[tuple[Path, CacheDeletion]] = []
    metadata = list(ACTIVE_MODEL_FILE.parent.glob("model--*/*.json"))
    if ACTIVE_MODEL_FILE.is_file():
        metadata.append(ACTIVE_MODEL_FILE)
    if request.target.scope == "checkpoint":
        run = get_run(request.target.run_id or "")
        if run:
            metadata.append(run.directory / "run.json")
    backups = {contained(path, Path.cwd()): path.read_bytes() for path in metadata}
    try:
        on_stage("Проверка завершена; перемещение выбранных файлов перед очисткой")
        for index, item in enumerate(preview.paths):
            source = contained(Path(item.path), Path.cwd())
            if is_repo(source, CACHE_ROOT):
                cache_plan = cache_deletion(source, CACHE_ROOT)
                if cache_plan:
                    cache_plans.append((source, cache_plan))
                    continue
            destination = contained(staging / str(index), staging)
            source.rename(destination)
            moved.append((source, destination))
            write_json(
                staging / "journal.json",
                {
                    "moves": [
                        {"source": src.as_posix(), "staging": dst.as_posix()} for src, dst in moved
                    ]
                },
            )
        _repair_registrations(request, removed_run.directory if removed_run else None)
    except Exception as error:
        rollback_errors: list[str] = []
        for source, destination in reversed(moved):
            if not source.exists() and destination.exists():
                try:
                    destination.rename(source)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
        if not rollback_errors:
            for path, content in backups.items():
                try:
                    temporary = path.with_suffix(path.suffix + ".restore")
                    temporary.write_bytes(content)
                    temporary.replace(path)
                except OSError as rollback_error:
                    rollback_errors.append(str(rollback_error))
        raise RuntimeError(
            f"Удаление отменено до очистки: {error}; журнал: {staging}. "
            f"Ошибки восстановления: {rollback_errors}"
        ) from error
    on_stage("Необратимое удаление подтверждённых файлов; освобождение места на диске")
    try:
        for repo, cache_plan in cache_plans:
            contained(repo, CACHE_ROOT)
            contained(repo, Path.cwd())
            cache_plan.execute()
            if repo.exists():
                raise OSError(f"HF cache не очищен полностью: {repo}")
        for _, destination in moved:
            contained(destination, staging)
            contained(destination, Path.cwd())
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        (staging / "journal.json").unlink(missing_ok=True)
        staging.rmdir()
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Не все файлы удалось очистить. Остатки и журнал: {staging}. {error}"
        ) from error
    return {
        "removed_paths": [item.path for item in preview.paths],
        "estimated_freed_bytes": preview.size_bytes,
        "removed_files": preview.files,
    }
