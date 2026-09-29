"""Тесты отчёта token statistics."""

import json
from pathlib import Path
from typing import cast

from mini_llm.bpe_tokenizer import train_bpe_tokenizer
from mini_llm.dataset_split import SPLIT_NAMES
from mini_llm.token_statistics import TokenStatisticsConfig, calculate_token_statistics


def _sample(board: str, thread_id: int, context: str, response: str) -> dict[str, object]:
    return {
        "board": board,
        "thread_id": thread_id,
        "context": [{"post_id": thread_id, "text": context}],
        "response": {"post_id": thread_id + 1, "text": response},
    }


def test_token_statistics_contains_required_metrics(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    samples = {
        "train": _sample("b", 1, "Привет, мир!", "Ответ на русском."),
        "validation": _sample("b", 2, "Mixed text 123", "Validation response 😎"),
        "test": _sample("po", 3, "Короткий вопрос", "Да"),
    }
    for split in SPLIT_NAMES:
        (splits_dir / f"{split}.jsonl").write_text(
            json.dumps(samples[split], ensure_ascii=False) + "\n", encoding="utf-8"
        )

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
        TokenStatisticsConfig(splits_dir, tokenizer_path, output_path, max_sequence_length=512)
    )

    split_report = cast(dict[str, dict[str, object]], report["splits"])
    assert output_path.is_file()
    assert cast(int, report["vocabulary_size"]) >= 262
    assert report["model_parameters"] == 5_387_776
    assert report["total_tokens"] == sum(
        cast(int, split_report[split]["tokens"]) for split in SPLIT_NAMES
    )
    assert cast(float, report["tokens_per_parameter"]) > 0
    assert cast(dict[str, object], report["sample_token_lengths"])["count"] == 3
    assert cast(int, cast(dict[str, object], report["context"])["tokens"]) > 0
    assert cast(int, cast(dict[str, object], report["response"])["tokens"]) > 0
    unknown = cast(dict[str, object], report["unknown_tokens"])
    assert unknown["count"] == 0
    assert unknown["percent"] == 0.0
    assert set(cast(dict[str, object], report["by_board"])) == {"b", "po"}
