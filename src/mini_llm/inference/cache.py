"""Одна подготовленная pretrained-модель в памяти процесса API."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

import torch

from mini_llm.observability import terminal_log, terminal_stage
from mini_llm.pretrained import PreparedPretrained

type FileSignature = tuple[Path, int, int]
type CacheKey = tuple[FileSignature, FileSignature | None, torch.device]


def _file_signature(path: Path) -> FileSignature:
    resolved = path.resolve()
    stat = resolved.stat()
    return resolved, stat.st_mtime_ns, stat.st_size


class PretrainedModelCache:
    """Повторно использовать веса и tokenizer, но никогда не prompt или ответы."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._key: CacheKey | None = None
        self._prepared: PreparedPretrained | None = None

    @contextmanager
    def use(
        self,
        config_file: Path,
        checkpoint_file: Path | None,
        device: torch.device,
        loader: Callable[[], PreparedPretrained],
    ) -> Iterator[PreparedPretrained]:
        """Загрузить при промахе и защитить общую модель до конца генерации."""

        # Lock удерживается и при генерации: параллельные запросы не удваивают VRAM.
        with self._lock:
            key = (
                _file_signature(config_file),
                _file_signature(checkpoint_file) if checkpoint_file is not None else None,
                device,
            )
            if key != self._key or self._prepared is None:
                self._clear_unlocked()
                source = str(checkpoint_file) if checkpoint_file is not None else "до SFT"
                with terminal_stage("INFERENCE", f"Загрузка модели: {config_file}, {source}"):
                    prepared = loader()
                self._prepared = prepared
                self._key = key
            else:
                terminal_log("INFERENCE", "Модель и tokenizer взяты из памяти, без загрузки весов")
            yield self._prepared

    def clear(self) -> None:
        """Освободить модель перед другим GPU pipeline или переключением backend."""

        with self._lock:
            self._clear_unlocked()

    def _clear_unlocked(self) -> None:
        if self._prepared is None:
            return
        was_cuda = self._key is not None and self._key[2].type == "cuda"
        self._prepared = None
        self._key = None
        gc.collect()
        if was_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        terminal_log("INFERENCE", "Кэш модели освобождён")
