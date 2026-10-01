"""Проверки кэша ресурсов без весов, Hugging Face и GPU."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
import torch

from mini_llm.inference.cache import PretrainedModelCache
from mini_llm.pretrained import PreparedPretrained


@pytest.fixture
def cache_files(tmp_path: Path) -> tuple[Path, Path]:
    """Создать только файлы-маркеры; подменённый loader не читает их содержимое."""

    config = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint.pt"
    config.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    return config, checkpoint


def test_cache_reuses_model_across_requests(cache_files: tuple[Path, Path]) -> None:
    cache = PretrainedModelCache()
    prepared = cast(PreparedPretrained, object())
    loads = 0

    def load() -> PreparedPretrained:
        nonlocal loads
        loads += 1
        return prepared

    for _ in range(2):
        with cache.use(*cache_files, torch.device("cpu"), load) as cached:
            assert cached is prepared
    assert loads == 1


@pytest.mark.parametrize("changed_file", [0, 1])
def test_cache_invalidates_after_file_update(
    cache_files: tuple[Path, Path], changed_file: int
) -> None:
    cache = PretrainedModelCache()
    loaded: list[PreparedPretrained] = []

    def load() -> PreparedPretrained:
        prepared = cast(PreparedPretrained, object())
        loaded.append(prepared)
        return prepared

    with cache.use(*cache_files, torch.device("cpu"), load):
        pass
    cache_files[changed_file].write_bytes(b"updated contents with different size")
    with cache.use(*cache_files, torch.device("cpu"), load) as cached:
        assert cached is loaded[-1]
    assert len(loaded) == 2
    assert loaded[0] is not loaded[1]


def test_cache_clear_forces_reload(cache_files: tuple[Path, Path]) -> None:
    cache = PretrainedModelCache()
    loaded: list[PreparedPretrained] = []

    def load() -> PreparedPretrained:
        prepared = cast(PreparedPretrained, object())
        loaded.append(prepared)
        return prepared

    with cache.use(*cache_files, torch.device("cpu"), load):
        pass
    cache.clear()
    with cache.use(*cache_files, torch.device("cpu"), load):
        pass
    assert len(loaded) == 2


def test_failed_load_does_not_poison_cache(cache_files: tuple[Path, Path]) -> None:
    cache = PretrainedModelCache()
    prepared = cast(PreparedPretrained, object())
    attempts = 0

    def load() -> PreparedPretrained:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Ошибка загрузки")
        return prepared

    with (
        pytest.raises(RuntimeError, match="Ошибка загрузки"),
        cache.use(*cache_files, torch.device("cpu"), load),
    ):
        pass
    with cache.use(*cache_files, torch.device("cpu"), load) as cached:
        assert cached is prepared
    assert attempts == 2


def test_parallel_requests_share_one_model(cache_files: tuple[Path, Path]) -> None:
    cache = PretrainedModelCache()
    prepared = cast(PreparedPretrained, object())
    loads = 0

    def load() -> PreparedPretrained:
        nonlocal loads
        loads += 1
        return prepared

    def request() -> PreparedPretrained:
        with cache.use(*cache_files, torch.device("cpu"), load) as cached:
            return cached

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(request) for _ in range(2)]
        assert all(future.result(timeout=5) is prepared for future in futures)
    assert loads == 1


def test_cache_separates_before_and_after_sft(cache_files: tuple[Path, Path]) -> None:
    cache = PretrainedModelCache()
    loads = 0

    def load() -> PreparedPretrained:
        nonlocal loads
        loads += 1
        return cast(PreparedPretrained, object())

    config, checkpoint = cache_files
    for selected_checkpoint in (None, None, checkpoint, checkpoint, None):
        with cache.use(config, selected_checkpoint, torch.device("cpu"), load):
            pass
    assert loads == 3
