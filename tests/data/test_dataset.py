"""Тесты PyTorch Dataset, padding и совместимости с моделью."""

import json
from pathlib import Path

import pytest
import torch

from mini_llm.data.dataset import IGNORE_INDEX, DataLoaderConfig, DialogueDataset, create_dataloader
from mini_llm.data.dialogue import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    DialogueSample,
)
from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.tokenization import BPETokenizer, train_bpe_tokenizer
from mini_llm.training.monitoring import evaluate_validation_loss
from mini_llm.training.progress import effective_train_tokens

MAX_LENGTH = 40


def test_batch_size_uses_shared_limit() -> None:
    assert DataLoaderConfig(batch_size=1).batch_size == 1
    assert DataLoaderConfig(batch_size=8).batch_size == 8
    with pytest.raises(ValueError, match="от 1 до 8"):
        DataLoaderConfig(batch_size=0)
    with pytest.raises(ValueError, match="от 1 до 8"):
        DataLoaderConfig(batch_size=9)


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


def test_dataset_skips_oversized_response_and_builds_shifted_sequence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)
    progress_output = capsys.readouterr().out

    assert len(dataset) == 3
    assert dataset.oversized_response_count == 1
    item = dataset[1]
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    sample = DialogueSample(
        "b",
        2,
        ("Старый контекст " * 50, "ПОСЛЕДНЕЕ СООБЩЕНИЕ"),
        "Ответ целиком",
    )
    encoded = tokenizer.encode_training_window(sample, max_length=MAX_LENGTH)
    sequence = encoded.token_ids
    response_ids = tokenizer.encode_response(sample)

    assert len(sequence) <= MAX_LENGTH
    assert input_ids == sequence[:-1]
    assert sequence[0] == tokenizer.token_to_id(BOS_TOKEN)
    assert sequence[-1] == tokenizer.token_to_id(EOS_TOKEN)
    assert sequence[-len(response_ids)] == tokenizer.token_to_id(ASSISTANT_TOKEN)
    assert sequence[-len(response_ids) :] == response_ids
    assert tokenizer.token_to_id(USER_TOKEN) in sequence[: -len(response_ids)]
    assert _contains_subsequence(sequence, tokenizer.encode("ПОСЛЕДНЕЕ СООБЩЕНИЕ"))
    assert labels[: encoded.response_start - 1] == [IGNORE_INDEX] * (encoded.response_start - 1)
    assert labels[encoded.response_start - 1 :] == sequence[encoded.response_start :]
    assert "[DATASET] Начало индексации" in progress_output
    assert "[DATASET] Индексация завершена" in progress_output
    assert "scanned=4 usable=3 oversized=1" in progress_output


def test_loss_labels_include_only_response_and_eos(tmp_path: Path) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)
    item = dataset[0]
    sample = DialogueSample("b", 1, ("Коротко",), "Да")
    encoded = tokenizer.encode_training_window(sample, max_length=MAX_LENGTH)

    assert int((item["labels"] != IGNORE_INDEX).sum().item()) == (
        len(encoded.token_ids) - encoded.response_start
    )
    assert item["labels"][-1].item() == tokenizer.token_to_id(EOS_TOKEN)


def test_missing_effective_token_statistics_does_not_scan_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _ = _prepare_dataset(tmp_path)
    capsys.readouterr()

    def unexpected_access(self: DialogueDataset, index: int) -> None:
        pytest.fail("effective_train_tokens не должен читать samples Dataset")

    monkeypatch.setattr(DialogueDataset, "__getitem__", unexpected_access)

    token_count, source = effective_train_tokens(
        tmp_path / "missing_statistics.json",
        dataset,
        max_sequence_length=MAX_LENGTH,
    )

    output = capsys.readouterr().out
    assert token_count is None
    assert source == "unknown"
    assert "effective_train_tokens=unknown" in output
    assert "Предварительный пересчёт отключён" in output
    assert "Fallback scan" not in output


def test_effective_tokens_reports_statistics_json_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _ = _prepare_dataset(tmp_path)
    statistics_file = tmp_path / "statistics.json"
    statistics_file.write_text(
        json.dumps(
            {
                "max_sequence_length": MAX_LENGTH,
                "training_objective": "response_only",
                "splits": {
                    "train": {
                        "usable_samples": len(dataset),
                        "training_loss_tokens": 123,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    def unexpected_access(self: DialogueDataset, index: int) -> None:
        pytest.fail("Готовая статистика не требует чтения samples Dataset")

    monkeypatch.setattr(DialogueDataset, "__getitem__", unexpected_access)

    token_count, source = effective_train_tokens(
        statistics_file,
        dataset,
        max_sequence_length=MAX_LENGTH,
    )

    output = capsys.readouterr().out
    assert token_count == 123
    assert source == str(statistics_file)
    assert "прочитаны из statistics JSON" in output
    assert "training_loss_tokens=123" in output


@pytest.mark.parametrize("case", ["invalid_json", "window", "objective", "samples", "tokens"])
def test_incompatible_statistics_do_not_trigger_dataset_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    dataset, _ = _prepare_dataset(tmp_path)
    train: dict[str, object] = {"usable_samples": len(dataset), "training_loss_tokens": 123}
    payload: dict[str, object] = {
        "max_sequence_length": MAX_LENGTH,
        "training_objective": "response_only",
        "splits": {"train": train},
    }
    if case == "window":
        payload["max_sequence_length"] = MAX_LENGTH + 1
    elif case == "objective":
        payload["training_objective"] = "all_tokens"
    elif case == "samples":
        train["usable_samples"] = len(dataset) + 1
    elif case == "tokens":
        train["training_loss_tokens"] = "unknown"
    path = tmp_path / "statistics.json"
    path.write_text("{" if case == "invalid_json" else json.dumps(payload), encoding="utf-8")

    def unexpected_access(self: DialogueDataset, index: int) -> None:
        pytest.fail("Несовместимая статистика не должна запускать scan Dataset")

    monkeypatch.setattr(DialogueDataset, "__getitem__", unexpected_access)
    assert effective_train_tokens(path, dataset, max_sequence_length=MAX_LENGTH) == (
        None,
        "unknown",
    )


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
    assert torch.all(first["attention_mask"][padding_mask] == 0)
    assert torch.all(first["labels"][padding_mask] == IGNORE_INDEX)


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


def test_validation_reports_periodic_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, tokenizer = _prepare_dataset(tmp_path)
    loader = create_dataloader(
        dataset,
        DataLoaderConfig(batch_size=2, num_workers=0),
        shuffle=False,
    )
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=MAX_LENGTH,
        num_layers=1,
        d_model=32,
        num_heads=4,
        dropout=0.0,
    )
    backend = CustomCausalLMBackend(DecoderOnlyTransformer(config), config)
    capsys.readouterr()

    loss = evaluate_validation_loss(
        backend,
        loader,
        device=torch.device("cpu"),
        use_amp=False,
        amp_dtype=torch.float16,
        max_batches=2,
    )

    output = capsys.readouterr().out
    assert torch.isfinite(torch.tensor(loss))
    assert "[VALIDATION] Расчёт validation loss начат planned_batches=2" in output
    assert "batch=1/2" in output
    assert "[VALIDATION] Расчёт завершён batches=2/2" in output
