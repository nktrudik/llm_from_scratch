"""Тесты raw/effective token statistics и oversized responses."""

import json
from pathlib import Path
from typing import cast

from mini_llm.bpe_tokenizer import train_bpe_tokenizer
from mini_llm.dataset_split import SPLIT_NAMES
from mini_llm.token_statistics import TokenStatisticsConfig, calculate_token_statistics


def _sample(board: str, thread_id: int, context: list[str], response: str) -> dict[str, object]:
    return {
        "board": board,
        "thread_id": thread_id,
        "context": [
            {"post_id": thread_id * 100 + index, "text": text} for index, text in enumerate(context)
        ],
        "response": {"post_id": thread_id * 100 + 99, "text": response},
    }


def _write_jsonl(path: Path, samples: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )


def test_token_statistics_separates_raw_effective_and_oversized(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    oversized_response = "".join(chr(0x400 + index) for index in range(100))
    split_samples = {
        "train": [
            _sample("b", 1, ["Очень длинный старый context " * 50, "Последний"], "Ответ"),
            _sample("b", 2, ["Контекст"], oversized_response),
        ],
        "validation": [_sample("b", 3, ["Validation context"], "Validation response")],
        "test": [_sample("po", 4, ["Короткий вопрос"], "Да")],
    }
    for split in SPLIT_NAMES:
        _write_jsonl(splits_dir / f"{split}.jsonl", split_samples[split])

    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe_tokenizer(
        splits_dir / "train.jsonl",
        tokenizer_path,
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    output_path = tmp_path / "stats.json"
    report = calculate_token_statistics(
        TokenStatisticsConfig(splits_dir, tokenizer_path, output_path, max_sequence_length=40)
    )

    split_report = cast(dict[str, dict[str, object]], report["splits"])
    train = split_report["train"]
    train_context = cast(dict[str, object], train["context"])
    train_response = cast(dict[str, object], train["response"])
    effective_lengths = cast(dict[str, object], train["effective_sequence_lengths"])

    assert output_path.is_file()
    assert report["model_parameters"] == 5_518_848
    assert cast(int, report["raw_full_tokens"]) > cast(int, report["effective_tokens"])
    assert train["unusable_oversized_response_samples"] == 1
    assert cast(int, train_context["dropped_tokens"]) > 0
    assert cast(float, train_context["lost_percent"]) > 0
    assert train_context["truncated_samples"] == 1
    assert cast(int, effective_lengths["max"]) <= 40
    assert cast(dict[str, object], train["effective_window_usage"])["full_window_samples"] == 1
    assert train_response["responses_over_max_sequence_length"] == 1
    assert cast(dict[str, object], report["oversized_responses_by_board"])["b"] == 1
    assert report["effective_train_tokens"] == train["effective_tokens"]
    expected_ratio = cast(int, train["effective_tokens"]) / 5_518_848
    assert report["effective_train_tokens_per_parameter"] == round(expected_ratio, 6)
    assert cast(int, split_report["validation"]["effective_tokens"]) > 0
    assert cast(int, split_report["test"]["effective_tokens"]) > 0
    assert report["total_tokens"] == sum(
        cast(int, split_report[split]["raw_full_tokens"]) for split in SPLIT_NAMES
    )
