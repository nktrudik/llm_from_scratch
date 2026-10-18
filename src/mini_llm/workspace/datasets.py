"""Публикация неизменяемых версий подготовленного корпуса."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.workspace.catalog import DatasetVersion, contained, fingerprint, identifier
from mini_llm.workspace.paths import DATASETS_ROOT


def publish_dataset(source: Path, dataset_id: str, version: str, title: str) -> DatasetVersion:
    """Скопировать проверенный JSONL без изменений; опубликованную версию не заменять."""

    identifier(dataset_id)
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("Версия должна иметь вид v1, v2, v3 и далее")
    destination = contained(DATASETS_ROOT / dataset_id / version, DATASETS_ROOT)
    count = sum(1 for _ in iter_dialogue_jsonl(source))
    if not count:
        raise ValueError("Пустой датасет нельзя опубликовать")
    checksum = fingerprint(source)
    destination.mkdir(parents=True, exist_ok=False)
    # Без dataset.json незавершённая публикация не появляется в каталоге.
    target = destination / "samples.jsonl"
    shutil.copyfile(source, target)
    if fingerprint(target) != checksum:
        raise ValueError("Исходник изменился во время публикации")
    record = DatasetVersion(
        dataset_id=dataset_id,
        version=version,
        title=title,
        samples=count,
        sha256=checksum,
        source=source.as_posix(),
    )
    record.save()
    return record
