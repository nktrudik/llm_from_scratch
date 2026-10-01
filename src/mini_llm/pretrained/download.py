"""Скачивание совместимых Transformers-файлов без создания модели и использования GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from mini_llm.observability import terminal_stage
from mini_llm.pretrained.dependencies import require_module


class _RepositoryFile(Protocol):
    rfilename: str


class _ModelInfo(Protocol):
    sha: str | None
    siblings: list[_RepositoryFile] | None


class _HubAPI(Protocol):
    def model_info(self, repo_id: str, *, revision: str) -> _ModelInfo: ...


@dataclass(frozen=True, slots=True)
class DownloadedSnapshot:
    """Локальные файлы конкретного commit, а не плавающей branch/tag."""

    path: Path
    revision: str


def select_model_files(filenames: list[str]) -> list[str]:
    """Предпочесть safetensors и не скачивать дубли весов для других runtime."""

    excluded_directories = {"onnx", "openvino", "original", "gguf", "ggml", "flax", "tf"}
    supported = [
        name
        for name in filenames
        if not any(part.lower() in excluded_directories for part in Path(name).parts[:-1])
    ]
    has_safetensors = any(name.endswith(".safetensors") for name in supported)
    selected = [
        name
        for name in supported
        if (
            name.endswith(
                (".json", ".model", ".txt", ".tiktoken", ".bpe", ".jinja", ".safetensors")
            )
            or (not has_safetensors and name.endswith(".bin"))
        )
        and not Path(name).name.startswith(("training_args", "optimizer", "scheduler"))
    ]
    if "config.json" not in selected:
        raise ValueError("В репозитории нет config.json для Transformers")
    if not any(name.endswith((".safetensors", ".bin")) for name in selected):
        raise ValueError("Нет Transformers-весов (.safetensors или .bin); GGUF не поддерживается")
    return selected


def download_model_snapshot(model_id: str, revision: str, cache_dir: Path) -> DownloadedSnapshot:
    """Разрешить revision в commit и скачать только модель и её tokenizer."""

    hub = require_module("huggingface_hub")
    api = cast(_HubAPI, hub.HfApi())
    downloader = cast(Callable[..., str], hub.snapshot_download)
    try:
        with terminal_stage("DOWNLOAD", f"model_id={model_id} revision={revision}"):
            info = api.model_info(model_id, revision=revision)
            if not info.sha or not info.siblings:
                raise RuntimeError("Hugging Face не вернул commit или список файлов модели")
            files = select_model_files([item.rfilename for item in info.siblings])
            path = downloader(
                repo_id=model_id,
                revision=info.sha,
                cache_dir=str(cache_dir),
                allow_patterns=files,
            )
        return DownloadedSnapshot(Path(path), info.sha)
    except Exception as error:
        # Внешняя граница Hub: ошибки авторизации/сети не должны давать успех setup.
        raise RuntimeError(f"Не удалось скачать {model_id}: {error}") from error
