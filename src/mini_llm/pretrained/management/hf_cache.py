"""Штатная очистка HF cache, включая общий blob-store новых версий Hub."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from mini_llm.pretrained.dependencies import require_module
from mini_llm.workspace.catalog import contained


class CacheDeletion(Protocol):
    """Публичная часть DeleteCacheStrategy из huggingface_hub."""

    repos: frozenset[Path]
    snapshots: frozenset[Path]
    refs: frozenset[Path]
    blobs: frozenset[Path]
    expected_freed_size: int

    def execute(self) -> None: ...


class _Revision(Protocol):
    commit_hash: str


class _Repo(Protocol):
    repo_path: Path
    revisions: frozenset[_Revision]


class _CacheInfo(Protocol):
    repos: frozenset[_Repo]

    def delete_revisions(self, *revisions: str) -> CacheDeletion: ...


def cache_deletion(repo: Path, root: Path) -> CacheDeletion | None:
    """Получить план Hub и запретить удаление других репозиториев/внешних путей."""

    contained(repo, root)
    contained(repo, Path.cwd())
    hub = require_module("huggingface_hub")
    scan = cast(Callable[[str], _CacheInfo], hub.scan_cache_dir)
    info = scan(str(repo.parent))
    selected = next(
        (item for item in info.repos if item.repo_path.resolve() == repo.resolve()), None
    )
    if selected is None or not selected.revisions:
        return None
    # Fork-модели могут иметь одинаковые commit hashes: ограничиваем публичный
    # HFCacheInfo выбранным repo, а не удаляем совпадения во всём cache.
    if not is_dataclass(info):
        raise ValueError("Неизвестный формат HFCacheInfo")
    scoped = cast(_CacheInfo, replace(info, repos=frozenset({selected})))
    strategy = scoped.delete_revisions(*(item.commit_hash for item in selected.revisions))
    if {item.resolve() for item in strategy.repos} != {repo.resolve()}:
        raise ValueError("HF cache plan затрагивает другой репозиторий; удаление запрещено")
    for collection in (strategy.repos, strategy.snapshots, strategy.refs, strategy.blobs):
        for path in collection:
            contained(path, root)
            contained(path, Path.cwd())
    # Новые версии Hub также могут предлагать удаление отдельных snapshot entries.
    files = cast(object, getattr(strategy, "files", frozenset()))
    if not isinstance(files, (frozenset, set, tuple, list)):
        raise ValueError("Неизвестный формат HF cache plan")
    for path in files:
        if not isinstance(path, Path):
            raise ValueError("Некорректный путь HF cache plan")
        contained(path, root)
        contained(path, Path.cwd())
    return strategy


def is_repo(path: Path, root: Path) -> bool:
    """Отличить репозиторий от файлов locks и остальных артефактов проекта."""

    return (
        path.name.startswith("models--")
        and path.parent.name != ".locks"
        and path.resolve().is_relative_to(root.resolve())
    )
