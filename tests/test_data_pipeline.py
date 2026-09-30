"""Тесты PyTorch Dataset, padding и совместимости с моделью."""

import json
from pathlib import Path

import pytest
import torch

from mini_llm.bpe_tokenizer import BPETokenizer, train_bpe_tokenizer
from mini_llm.config import ModelConfig
from mini_llm.data_pipeline import DataLoaderConfig, DialogueDataset, create_dataloader
from mini_llm.dialogue_format import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    DialogueSample,
)
from mini_llm.model import DecoderOnlyTransformer

MAX_LENGTH = 40


def test_batch_size_is_limited_to_four() -> None:
    assert DataLoaderConfig(batch_size=1).batch_size == 1
    assert DataLoaderConfig(batch_size=4).batch_size == 4
    with pytest.raises(ValueError, match="от 1 до 4"):
        DataLoaderConfig(batch_size=0)
    with pytest.raises(ValueError, match="от 1 до 4"):
        DataLoaderConfig(batch_size=5)


def _sample(thread_id: int, context: list[str], response: str) -> dict[str, object]:
    return {
        "board": "b",
        "thread_id": thread_id,
        "context": [
            {"post_id": thread_id * 100 + index, "text": text} for index, text in enumerate(context)
        ],
        "response": {"post_id": thread_id * 100 + 99, "text": response},
    }


def _prepare_dataset(tmp_path: Path) -> tuple[DialogueDataset, BPETokenizer]:
    oversized_response = "".join(chr(0x400 + index) for index in range(100))
    samples = [
        _sample(1, ["Коротко"], "Да"),
        _sample(2, ["Старый контекст " * 50, "ПОСЛЕДНЕЕ СООБЩЕНИЕ"], "Ответ целиком"),
        _sample(3, ["Контекст средней длины"], "Немного более длинный ответ"),
        _sample(4, ["Контекст"], oversized_response),
    ]
    split_file = tmp_path / "train.jsonl"
    split_file.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe_tokenizer(
        split_file,
        tokenizer_path,
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    return (
        DialogueDataset(split_file, tokenizer, max_sequence_length=MAX_LENGTH),
        tokenizer,
    )


def _contains_subsequence(values: list[int], expected: list[int]) -> bool:
    return any(
        values[index : index + len(expected)] == expected
        for index in range(len(values) - len(expected) + 1)
    )


def test_dataset_skips_oversized_response_and_builds_shifted_sequence(tmp_path: Path) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)

    assert len(dataset) == 3
    assert dataset.oversized_response_count == 1
    item = dataset[1]
    input_ids = item["input_ids"].tolist()
    targets = item["targets"].tolist()
    sequence = [*input_ids, targets[-1]]
    response_ids = tokenizer.encode_response(
        DialogueSample(
            "b",
            2,
            ("Старый контекст " * 50, "ПОСЛЕДНЕЕ СООБЩЕНИЕ"),
            "Ответ целиком",
        )
    )

    assert len(sequence) <= MAX_LENGTH
    assert input_ids[1:] == targets[:-1]
    assert sequence[0] == tokenizer.token_to_id(BOS_TOKEN)
    assert sequence[-1] == tokenizer.token_to_id(EOS_TOKEN)
    assert sequence[-len(response_ids)] == tokenizer.token_to_id(ASSISTANT_TOKEN)
    assert sequence[-len(response_ids) :] == response_ids
    assert tokenizer.token_to_id(USER_TOKEN) in sequence[: -len(response_ids)]
    assert _contains_subsequence(sequence, tokenizer.encode("ПОСЛЕДНЕЕ СООБЩЕНИЕ"))


def test_dataloader_pads_targets_and_keeps_incomplete_last_batch(tmp_path: Path) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)
    loader = create_dataloader(
        dataset,
        DataLoaderConfig(batch_size=2, num_workers=0, random_seed=7),
        shuffle=False,
    )
    batches = list(loader)
    first = batches[0]
    pad_id = tokenizer.token_to_id("<PAD>")

    assert first["input_ids"].shape == first["targets"].shape
    assert first["input_ids"].shape[0] == 2
    assert batches[1]["input_ids"].shape[0] == 1
    padding_mask = first["input_ids"] == pad_id
    assert padding_mask.any()
    assert torch.all(first["targets"][padding_mask] == pad_id)


def test_batch_runs_through_transformer_with_finite_loss(tmp_path: Path) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)
    loader = create_dataloader(
        dataset,
        DataLoaderConfig(batch_size=2, num_workers=0),
        shuffle=False,
    )
    batch = next(iter(loader))
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=MAX_LENGTH,
        num_layers=1,
        d_model=32,
        num_heads=4,
        dropout=0.0,
    )
    model = DecoderOnlyTransformer(config)

    logits, loss = model(batch["input_ids"], batch["targets"])

    assert logits.shape[:2] == batch["input_ids"].shape
    assert torch.isfinite(loss)
