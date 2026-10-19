"""Расчёт принадлежащих модели путей и безопасного плана очистки."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mini_llm.inference.checkpoints import resolve_named_checkpoint
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.management.hf_cache import cache_deletion, is_repo
from mini_llm.pretrained.management.schemas import DeletionPath, DeletionPreview, DeletionTarget
from mini_llm.pretrained.workspace import ModelPaths, list_registered_models
from mini_llm.workspace.catalog import contained, get_run, list_runs
from mini_llm.workspace.paths import RUNS_ROOT

CACHE_ROOT = Path(".cache/huggingface")


def model_paths(model_id: str, *, include_cache: bool) -> list[tuple[Path, Path]]:
    """Вернуть только выделенные каталоги модели, никогда не общие корни."""

    if model_id == "custom":
        return [(RUNS_ROOT / "custom", RUNS_ROOT)]
    paths = ModelPaths.for_model(model_id)
    slug = paths.config_file.parent.name
    result = [
        (RUNS_ROOT / slug, RUNS_ROOT),
        (paths.config_file.parent, Path("configs/models")),
        (paths.registration_file.parent, Path("artifacts/models")),
        (paths.checkpoint_dir.parent, Path("checkpoints/base")),
    ]
    if not include_cache:
        return result
    repo_name = "models--" + model_id.replace("/", "--")
    caches: set[Path] = set()
    default_cache = CACHE_ROOT / repo_name
    if default_cache.exists():
        contained(default_cache, Path.cwd())
        caches.add(default_cache)
    configs = list(paths.config_file.parent.glob("*.json"))
    for run in list_runs():
        if run.model_id.casefold() == model_id.casefold() and run.pretrained_config_file:
            contained(run.pretrained_config_file, run.directory)
            configs.append(run.pretrained_config_file)
    for path in configs:
        config = PretrainedConfig.load(path)
        if config.model_id.casefold() != model_id.casefold():
            raise ValueError(f"Конфиг {path} принадлежит другой модели")
        cache = config.cache_dir / repo_name
        if cache.exists():
            contained(cache, CACHE_ROOT)
            caches.add(cache)
    for registration in list_registered_models():
        if registration.model_id.casefold() != model_id.casefold():
            continue
        cache = registration.snapshot_path.parent.parent
        if cache.name.casefold() != repo_name.casefold():
            raise ValueError("Не удалось безопасно определить HF cache модели")
        if cache.exists():
            contained(cache, CACHE_ROOT)
            caches.add(cache)
    for cache in sorted(caches):
        for other in list_registered_models():
            if (
                other.model_id.casefold() != model_id.casefold()
                and other.snapshot_path.resolve().is_relative_to(cache.resolve())
            ):
                raise ValueError(f"Cache также используется моделью {other.model_id}")
        for other_run in list_runs():
            if (
                other_run.model_id.casefold() != model_id.casefold()
                and other_run.pretrained_config_file
            ):
                config = PretrainedConfig.load(other_run.pretrained_config_file)
                if config.model_id.casefold() == model_id.casefold():
                    raise ValueError(f"Базовые веса нужны запуску {other_run.run_id}")
        result.extend(((cache, CACHE_ROOT), (cache.parent / ".locks" / cache.name, CACHE_ROOT)))
    return result


def target_paths(target: DeletionTarget) -> list[tuple[Path, Path]]:
    """Разрешить подтверждаемые ID в канонические серверные пути."""

    if target.scope == "model":
        model_id = target.model_id or ""
        if any(
            run.status == "running" and run.model_id.casefold() == model_id.casefold()
            for run in list_runs()
        ):
            raise ValueError("У модели есть выполняющийся запуск. Дождитесь его завершения.")
        known = {item.model_id.casefold() for item in list_registered_models()}
        known.update(item.model_id.casefold() for item in list_runs())
        paths = model_paths(model_id, include_cache=target.include_cache)
        if (
            model_id != "custom"
            and model_id.casefold() not in known
            and not any(path.exists() for path, _ in paths)
        ):
            raise ValueError("Модель не найдена в локальном каталоге")
        return paths
    run = get_run(target.run_id or "")
    if run is None:
        raise ValueError("Запуск не найден")
    if run.status == "running":
        raise ValueError("Этот запуск помечен как выполняющийся. Удаление запрещено.")
    contained(run.directory, RUNS_ROOT)
    if target.scope == "checkpoint":
        return [
            (resolve_named_checkpoint(run.directory, target.checkpoint_name or ""), run.directory)
        ]
    return [(run.directory, RUNS_ROOT)]


def inspect_path(
    path: Path, root: Path, *, count_link_targets: bool = True
) -> tuple[DeletionPath, list[str]]:
    """Проверить абсолютные границы и ссылки до любого перемещения/удаления."""

    contained(path, Path.cwd())
    contained(path, root)
    files = size = 0
    signature: list[str] = []
    counted: set[Path] = set()
    cache_repo = is_repo(path, CACHE_ROOT)
    stack = [path]
    while stack:
        node = stack.pop()
        if node.is_junction() or (node.is_symlink() and node.is_dir()):
            raise ValueError(f"Очистка каталогов со ссылками запрещена: {node}")
        if node != path:
            contained(node, CACHE_ROOT if cache_repo else path if path.is_dir() else path.parent)
            contained(node, Path.cwd())
        stat = node.lstat()
        signature.append(f"{node.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
        if node.is_dir():
            stack.extend(sorted(node.iterdir()))
        else:
            files += 1
            resolved = node.resolve()
            if (
                resolved not in counted
                and (count_link_targets or not node.is_symlink())
                and resolved.exists()
            ):
                resolved_stat = resolved.stat()
                size += resolved_stat.st_size
                signature.append(f"{resolved}:{resolved_stat.st_size}:{resolved_stat.st_mtime_ns}")
                counted.add(resolved)
    return DeletionPath(path=path.as_posix(), size_bytes=size, files=files), signature


def deletion_preview(target: DeletionTarget) -> DeletionPreview:
    """Оценить объём без чтения весов; подпись выявляет изменения плана."""

    paths: list[DeletionPath] = []
    signatures: list[str] = []
    seen: set[Path] = set()
    for path, root in target_paths(target):
        if not path.exists() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        entry, signature = inspect_path(path, root)
        if is_repo(path, CACHE_ROOT):
            strategy = cache_deletion(path, CACHE_ROOT)
            if strategy:
                entry.size_bytes = strategy.expected_freed_size
                signature.append(f"HF freed bytes:{strategy.expected_freed_size}")
            else:
                entry.size_bytes = inspect_path(path, root, count_link_targets=False)[0].size_bytes
        paths.append(entry)
        signatures.extend(signature)
    if not paths:
        raise ValueError("Нет локальных файлов для удаления")
    digest = hashlib.sha256(
        json.dumps(
            {"target": target.model_dump(), "files": sorted(signatures)}, sort_keys=True
        ).encode()
    ).hexdigest()
    warning = "Удаление необратимо. Датасеты и код не затрагиваются."
    if target.include_cache:
        warning += (
            " Общие HF blobs удаляются штатным менеджером только если больше не используются."
        )
    return DeletionPreview(
        target=target,
        paths=paths,
        size_bytes=sum(item.size_bytes for item in paths),
        files=sum(item.files for item in paths),
        preview_token=digest,
        warning=warning,
    )
