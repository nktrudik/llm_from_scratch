"""Подготовка целых разговоров и несдвинутых response-only labels для SFTTrainer."""

from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor

from mini_llm.data.dialogue import iter_dialogue_jsonl
from mini_llm.data.interfaces import OversizedResponseError
from mini_llm.observability import ProgressThrottle, terminal_log, terminal_stage
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer
from mini_llm.training.control import TrainingCancelled, training_stop_requested


class TokenizedDataset(Protocol):
    """Минимальный контракт готового datasets.Dataset."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, list[int]]: ...


class _DatasetFactory(Protocol):
    def from_list(self, rows: list[dict[str, list[int]]]) -> TokenizedDataset: ...


def prepare_dataset(
    path: Path, tokenizer: HuggingFaceDialogueTokenizer, *, max_length: int, limit: int | None
) -> TokenizedDataset:
    """Пропускать oversized целиком, не обрезая вопрос или ответ и не меняя JSONL."""

    rows: list[dict[str, list[int]]] = []
    oversized = 0
    loss_tokens = 0
    throttle = ProgressThrottle(every_items=2000, every_seconds=5.0)
    with terminal_stage("SFT DATASET", f"токенизация {path}"):
        for scanned, (_, sample) in enumerate(iter_dialogue_jsonl(path), 1):
            if training_stop_requested():
                raise TrainingCancelled
            if "<think>" in sample.response or "</think>" in sample.response:
                raise ValueError(
                    f"{path}:{scanned}: assistant содержит thinking-разметку; "
                    "подготовьте non-thinking ответ без изменения исходного файла"
                )
            try:
                encoded = tokenizer.encode_training_window(sample, max_length=max_length)
            except OversizedResponseError:
                oversized += 1
                continue
            # Сдвиг causal labels выполнит сама HF-модель, не Dataset.
            labels = [-100] * encoded.response_start + encoded.token_ids[encoded.response_start :]
            rows.append({"input_ids": encoded.token_ids, "labels": labels})
            loss_tokens += len(encoded.token_ids) - encoded.response_start
            if throttle.should_report(scanned):
                terminal_log(
                    "SFT DATASET", f"scanned={scanned} usable={len(rows)} oversized={oversized}"
                )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Нет пригодных целых диалогов: {path}; oversized={oversized}")
    terminal_log(
        "SFT DATASET",
        f"file={path} usable={len(rows)} oversized={oversized} effective_loss_tokens={loss_tokens}",
    )
    factory = cast(_DatasetFactory, require_module("datasets").Dataset)
    return factory.from_list(rows)


class ResponseOnlyCollator:
    """Динамический padding; служебный prompt и padding не участвуют в loss."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, Tensor]:
        length = max(len(row["input_ids"]) for row in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            padding = length - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            masks.append([1] * len(row["input_ids"]) + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
