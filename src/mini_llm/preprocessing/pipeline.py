"""Потоковый pipeline преобразования raw-тредов в dialogue dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

from mini_llm.deduplication import ContentDeduplicator
from mini_llm.preprocessing.parsing import load_raw_thread
from mini_llm.preprocessing.samples import iter_thread_samples
from mini_llm.preprocessing.schemas import PreprocessingConfig, PreprocessingError
from mini_llm.preprocessing_stats import PreprocessingStats


class _ReviewReservoir:
    """Детерминированная reservoir-выборка для ручной проверки."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self._seen = 0
        self._items: list[dict[str, object]] = []
        # Отдельный PRNG сохраняет воспроизводимость review sample.
        self._random = random.Random(seed)  # nosec B311

    @property
    def items(self) -> list[dict[str, object]]:
        """Вернуть накопленные элементы review sample."""

        return self._items

    def consider(self, sample: dict[str, object]) -> None:
        """Рассмотреть sample для включения в reservoir-выборку."""

        self._seen += 1
        if self.size == 0:
            return
        if len(self._items) < self.size:
            self._items.append(sample)
            return
        replacement_index = self._random.randrange(self._seen)
        if replacement_index < self.size:
            self._items[replacement_index] = sample


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def preprocess_dataset(config: PreprocessingConfig | None = None) -> PreprocessingStats:
    """Создать dataset, статистику и review JSONL потоковым проходом по raw-тредам."""

    active_config = config or PreprocessingConfig()
    if not active_config.input_dir.is_dir():
        raise PreprocessingError(f"Каталог raw-данных не существует: {active_config.input_dir}")
    active_config.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = active_config.output_dir / "2ch_dialogues.jsonl"
    review_path = active_config.output_dir / "2ch_review_sample.jsonl"
    stats_path = active_config.output_dir / "2ch_preprocessing_stats.json"
    temporary_dataset_path = dataset_path.with_suffix(".jsonl.tmp")
    stats = PreprocessingStats()
    deduplicator = ContentDeduplicator(
        minimum_characters=active_config.dedup_min_characters,
        similarity_threshold=active_config.near_duplicate_threshold,
    )
    review = _ReviewReservoir(active_config.review_size, active_config.random_seed)

    with temporary_dataset_path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for path in sorted(active_config.input_dir.glob("*/*.json")):
            stats.raw_threads_read += 1
            try:
                thread = load_raw_thread(path, active_config.input_dir)
            except PreprocessingError:
                stats.dropped["malformed_thread"] += 1
                continue
            stats.raw_posts_read += len(thread.posts)
            stats.boards[thread.board] += 1
            for sample in iter_thread_samples(thread, deduplicator, stats):
                dataset_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                review.consider(sample)
    temporary_dataset_path.replace(dataset_path)

    temporary_review_path = review_path.with_suffix(".jsonl.tmp")
    with temporary_review_path.open("w", encoding="utf-8", newline="\n") as review_file:
        for sample in review.items:
            review_file.write(json.dumps(sample, ensure_ascii=False, indent=None) + "\n")
    temporary_review_path.replace(review_path)
    stats.review_samples = len(review.items)
    _write_json(stats_path, stats.to_dict())
    return stats
