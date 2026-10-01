"""PyTorch Dataset и DataLoader для causal language modeling."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.data.dialogue import (
    DialogueFormatError,
    DialogueSample,
    parse_dialogue_sample,
)
from mini_llm.data.interfaces import DialogueTokenizer, OversizedResponseError
from mini_llm.data.splitting import SPLIT_NAMES
from mini_llm.modeling.config import DEFAULT_MAX_SEQUENCE_LENGTH, ModelConfig
from mini_llm.observability import ProgressThrottle, terminal_log
from mini_llm.tokenization import DEFAULT_TOKENIZER_PATH, BPETokenizer

IGNORE_INDEX = -100
INDEX_PROGRESS_SAMPLES = 5_000
INDEX_PROGRESS_SECONDS = 5.0


class CausalLMItem(TypedDict):
    """Одна shifted-пара variable-length tensors."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    targets: Tensor


class CausalLMBatch(TypedDict):
    """Один padded batch для универсального causal LM backend."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """Минимальные настройки batching и воспроизводимого shuffle."""

    batch_size: int = MAX_BATCH_SIZE
    num_workers: int = 0
    random_seed: int = 42
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size должен быть в диапазоне от 1 до {MAX_BATCH_SIZE}")
        if self.num_workers < 0:
            raise ValueError("num_workers не может быть отрицательным")


class DialogueDataset(Dataset[CausalLMItem]):
    """Map-style JSONL Dataset, хранящий только offsets пригодных samples."""

    def __init__(
        self,
        split_file: Path,
        tokenizer: DialogueTokenizer,
        *,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        max_samples: int | None = None,
    ) -> None:
        if max_sequence_length <= 1:
            raise ValueError("max_sequence_length должен быть больше единицы")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples должен быть положительным")
        if not split_file.is_file():
            raise DialogueFormatError(f"Split-файл не найден: {split_file}")
        self.split_file = split_file
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_samples = max_samples
        self.scanned_samples = 0
        self.oversized_response_count = 0
        self.oversized_responses_by_board: Counter[str] = Counter()
        self._offsets: list[int] = []
        self._build_index()

    def _parse_line(self, line: bytes, location: str) -> DialogueSample:
        try:
            payload = cast(object, json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DialogueFormatError(f"{location}: некорректный JSON: {error}") from error
        return parse_dialogue_sample(payload, location)

    def _build_index(self) -> None:
        started_at = time.perf_counter()
        try:
            file_size = self.split_file.stat().st_size
        except OSError:
            file_size = 0
        terminal_log(
            "DATASET",
            f"Начало индексации file={self.split_file} size_bytes={file_size}",
        )
        throttle = ProgressThrottle(INDEX_PROGRESS_SAMPLES, INDEX_PROGRESS_SECONDS)
        current_position = 0
        with self.split_file.open("rb") as source:
            line_number = 0
            while True:
                offset = source.tell()
                line = source.readline()
                current_position = source.tell()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                sample = self._parse_line(line, f"{self.split_file}:{line_number}")
                self.scanned_samples += 1
                try:
                    self.tokenizer.encode_training_window(
                        sample, max_length=self.max_sequence_length
                    )
                except OversizedResponseError:
                    self.oversized_response_count += 1
                    self.oversized_responses_by_board[sample.board] += 1
                else:
                    self._offsets.append(offset)
                if throttle.should_report(self.scanned_samples):
                    elapsed = max(time.perf_counter() - started_at, 1e-9)
                    percentage = 100.0 * current_position / file_size if file_size else 0.0
                    terminal_log(
                        "DATASET",
                        f"Индексация file={self.split_file.name} "
                        f"scanned={self.scanned_samples} usable={len(self._offsets)} "
                        f"oversized={self.oversized_response_count} "
                        f"progress={percentage:.2f}% "
                        f"speed={self.scanned_samples / elapsed:.1f} samples/sec",
                        elapsed=elapsed,
                    )
                if self.max_samples is not None and len(self._offsets) >= self.max_samples:
                    break
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        percentage = 100.0 * current_position / file_size if file_size else 100.0
        terminal_log(
            "DATASET",
            f"Индексация завершена file={self.split_file} scanned={self.scanned_samples} "
            f"usable={len(self._offsets)} oversized={self.oversized_response_count} "
            f"progress={percentage:.2f}% speed={self.scanned_samples / elapsed:.1f} samples/sec",
            elapsed=elapsed,
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def _read_sample(self, index: int) -> DialogueSample:
        try:
            offset = self._offsets[index]
        except IndexError as error:
            raise IndexError(f"Индекс sample вне Dataset: {index}") from error
        with self.split_file.open("rb") as source:
            source.seek(offset)
            line = source.readline()
        return self._parse_line(line, f"{self.split_file}@{offset}")

    def __getitem__(self, index: int) -> CausalLMItem:
        sample = self._read_sample(index)
        encoded = self.tokenizer.encode_training_window(sample, max_length=self.max_sequence_length)
        sequence = torch.tensor(encoded.token_ids, dtype=torch.long)
        labels = sequence[1:].clone()
        # labels[i] соответствует token_ids[i + 1]. Контекст и маркер ASSISTANT
        # нужны как вход, но первый обучающий target — текст ответа после маркера.
        labels[: max(0, encoded.response_start - 1)] = IGNORE_INDEX
        return {
            "input_ids": sequence[:-1],
            "attention_mask": torch.ones(sequence.numel() - 1, dtype=torch.long),
            "labels": labels,
            "targets": labels,
        }


def collate_causal_lm_batch(items: Sequence[CausalLMItem], *, pad_token_id: int) -> CausalLMBatch:
    """Дополнить variable-length samples справа до максимума текущего batch."""

    if not items:
        raise ValueError("Нельзя собрать пустой batch")
    max_length = max(item["input_ids"].numel() for item in items)
    input_ids = torch.full((len(items), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
    labels = torch.full((len(items), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, item in enumerate(items):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "targets": labels,
    }


def create_dataloader(
    dataset: DialogueDataset,
    config: DataLoaderConfig | None = None,
    *,
    shuffle: bool,
) -> DataLoader[CausalLMBatch]:
    """Создать DataLoader с deterministic generator и неполным последним batch."""

    active_config = config or DataLoaderConfig()
    generator = torch.Generator()
    generator.manual_seed(active_config.random_seed)
    collate = partial(
        collate_causal_lm_batch,
        pad_token_id=dataset.tokenizer.pad_token_id,
    )
    return DataLoader(
        dataset,
        batch_size=active_config.batch_size,
        shuffle=shuffle,
        num_workers=active_config.num_workers,
        collate_fn=collate,
        pin_memory=active_config.pin_memory,
        drop_last=False,
        generator=generator,
    )


def create_split_dataloader(
    split: str,
    *,
    splits_dir: Path = Path("data/processed/splits"),
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH,
    loader_config: DataLoaderConfig | None = None,
    max_sequence_length: int | None = None,
    max_samples: int | None = None,
) -> tuple[DialogueDataset, DataLoader[CausalLMBatch]]:
    """Загрузить готовый tokenizer и создать train/evaluation DataLoader."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"Неизвестный split: {split}")
    tokenizer = BPETokenizer.load(tokenizer_file)
    dataset = DialogueDataset(
        splits_dir / f"{split}.jsonl",
        tokenizer,
        max_sequence_length=(
            ModelConfig().max_sequence_length
            if max_sequence_length is None
            else max_sequence_length
        ),
        max_samples=max_samples,
    )
    loader = create_dataloader(dataset, loader_config, shuffle=split == "train")
    return dataset, loader


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI быстрой проверки Dataset/DataLoader."""

    parser = argparse.ArgumentParser(description="Проверить Dataset/DataLoader на малой выборке.")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="train")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--tokenizer-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Собрать один CPU batch без обучения модели."""

    args = build_argument_parser().parse_args(argv)
    try:
        dataset, loader = create_split_dataloader(
            args.split,
            splits_dir=args.splits_dir,
            tokenizer_file=args.tokenizer_file,
            loader_config=DataLoaderConfig(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                random_seed=args.seed,
            ),
            max_sequence_length=args.max_sequence_length,
            max_samples=args.max_samples,
        )
        batch = next(iter(loader))
    except (DialogueFormatError, OSError, RuntimeError, ValueError, StopIteration) as error:
        print(f"Проверка DataLoader не выполнена: {error}")
        return 1
    print(
        f"Dataset: {len(dataset)} usable samples; oversized пропущено: "
        f"{dataset.oversized_response_count}; batch shapes: "
        f"input_ids={tuple(batch['input_ids'].shape)}, targets={tuple(batch['targets'].shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
