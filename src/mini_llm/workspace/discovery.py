"""Обнаружение вручную добавленных версий и постоянный учёт количества samples."""

from __future__ import annotations

import re
from threading import Lock

from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.workspace.catalog import DatasetVersion, contained, fingerprint
from mini_llm.workspace.paths import DATASETS_ROOT

# Одновременные запросы UI не должны записывать один manifest через общий временный файл.
_catalog_lock = Lock()


def dataset_catalog(*, refresh: bool = True) -> tuple[list[DatasetVersion], list[str]]:
    """Учесть новые samples.jsonl один раз; существующие корпуса повторно не сканировать."""

    records: list[DatasetVersion] = []
    warnings: list[str] = []
    with _catalog_lock:
        for directory in sorted(DATASETS_ROOT.glob("*/v*")):
            if not directory.is_dir() or not re.fullmatch(r"v[1-9][0-9]*", directory.name):
                continue
            try:
                contained(directory, DATASETS_ROOT)
                manifest = directory / "dataset.json"
                if manifest.exists():
                    contained(manifest, directory)
                    record = DatasetVersion.model_validate_json(
                        manifest.read_text(encoding="utf-8")
                    )
                elif refresh:
                    source = contained(directory / "samples.jsonl", directory)
                    before = source.stat()
                    count = sum(1 for _ in iter_dialogue_jsonl(source))
                    checksum = fingerprint(source)
                    after = source.stat()
                    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                        raise ValueError(
                            "samples.jsonl изменяется; дождитесь окончания копирования"
                        )
                    record = DatasetVersion(
                        dataset_id=directory.parent.name,
                        version=directory.name,
                        title=f"{directory.parent.name} / {directory.name}",
                        samples=count,
                        sha256=checksum,
                        source=source.as_posix(),
                    )
                    record.save()
                else:
                    continue
                if record.directory.resolve() != directory.resolve():
                    raise ValueError("dataset.json не соответствует каталогу версии")
                if not (directory / "samples.jsonl").is_file():
                    raise ValueError("Отсутствует samples.jsonl")
                records.append(record)
            except (OSError, ValueError, UnicodeError) as error:
                warnings.append(f"{directory.as_posix()}: {error}")
    records.sort(key=lambda item: (item.dataset_id, int(item.version[1:])))
    return records, warnings
