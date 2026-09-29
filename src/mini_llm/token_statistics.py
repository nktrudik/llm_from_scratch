"""Подсчёт token statistics для готовых split-файлов и BPE tokenizer."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mini_llm.bpe_tokenizer import DEFAULT_TOKENIZER_PATH, BPETokenizer, TokenizerError
from mini_llm.config import ModelConfig
from mini_llm.dataset_split import SPLIT_NAMES
from mini_llm.dialogue_format import (
    BOS_TOKEN,
    UNK_TOKEN,
    DialogueFormatError,
    DialogueSample,
    iter_dialogue_jsonl,
    percentile,
    write_json,
)


@dataclass(frozen=True, slots=True)
class TokenStatisticsConfig:
    """Пути и размер будущего training window."""

    splits_dir: Path = Path("data/processed/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    output_file: Path = Path("data/processed/token_statistics.json")
    max_sequence_length: int = 512

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length должен быть положительным")


@dataclass(slots=True)
class _Accumulator:
    samples: int = 0
    tokens: int = 0
    characters: int = 0
    content_tokens: int = 0
    unknown_tokens: int = 0
    within_window: int = 0
    sample_lengths: list[int] = field(default_factory=list)
    context_lengths: list[int] = field(default_factory=list)
    response_lengths: list[int] = field(default_factory=list)
    context_tokens: int = 0
    response_tokens: int = 0
    context_content_tokens: int = 0
    response_content_tokens: int = 0
    context_characters: int = 0
    response_characters: int = 0

    def add(
        self,
        sample: DialogueSample,
        *,
        full_ids: Sequence[int],
        context_ids: Sequence[int],
        response_ids: Sequence[int],
        content_token_count: int,
        unknown_id: int,
        max_sequence_length: int,
    ) -> None:
        """Добавить измерения одного sample."""

        context_characters = sum(len(text) for text in sample.context)
        response_characters = len(sample.response)
        self.samples += 1
        self.tokens += len(full_ids)
        self.characters += context_characters + response_characters
        self.content_tokens += content_token_count
        self.unknown_tokens += full_ids.count(unknown_id)
        self.within_window += len(full_ids) <= max_sequence_length
        self.sample_lengths.append(len(full_ids))
        self.context_lengths.append(len(context_ids))
        self.response_lengths.append(len(response_ids))
        self.context_tokens += len(context_ids)
        self.response_tokens += len(response_ids)
        self.context_content_tokens += len(context_ids) - len(sample.context)
        self.response_content_tokens += len(response_ids) - 2
        self.context_characters += context_characters
        self.response_characters += response_characters


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0,
            "p95": 0,
            "max": 0,
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 4),
        "median": statistics.median(ordered),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _model_parameter_count(config: ModelConfig) -> int:
    d_model = config.d_model
    ffn_size = config.ffn_size
    embedding_parameters = config.vocab_size * d_model
    position_parameters = config.max_sequence_length * d_model
    attention_weights = 4 * d_model * d_model
    mlp_weights = 2 * d_model * ffn_size
    linear_biases = (5 * d_model + ffn_size) if config.bias else 0
    block_norms = 4 * d_model
    final_norm = 2 * d_model
    return (
        embedding_parameters
        + position_parameters
        + config.num_layers * (attention_weights + mlp_weights + linear_biases + block_norms)
        + final_norm
    )


def _accumulator_report(accumulator: _Accumulator, max_sequence_length: int) -> dict[str, object]:
    over_window = accumulator.samples - accumulator.within_window
    return {
        "samples": accumulator.samples,
        "tokens": accumulator.tokens,
        "characters_per_token": round(accumulator.characters / accumulator.content_tokens, 6)
        if accumulator.content_tokens
        else 0.0,
        "sample_token_lengths": _distribution(accumulator.sample_lengths),
        "within_max_sequence_length": {
            "max_sequence_length": max_sequence_length,
            "samples": accumulator.within_window,
            "percent": _percentage(accumulator.within_window, accumulator.samples),
        },
        "over_max_sequence_length": {
            "samples": over_window,
            "percent": _percentage(over_window, accumulator.samples),
        },
        "context": {
            "tokens": accumulator.context_tokens,
            "content_tokens": accumulator.context_content_tokens,
            "characters_per_token": round(
                accumulator.context_characters / accumulator.context_content_tokens, 6
            )
            if accumulator.context_content_tokens
            else 0.0,
            "token_lengths": _distribution(accumulator.context_lengths),
        },
        "response": {
            "tokens": accumulator.response_tokens,
            "content_tokens": accumulator.response_content_tokens,
            "characters_per_token": round(
                accumulator.response_characters / accumulator.response_content_tokens, 6
            )
            if accumulator.response_content_tokens
            else 0.0,
            "token_lengths": _distribution(accumulator.response_lengths),
        },
        "unknown_tokens": {
            "count": accumulator.unknown_tokens,
            "percent": _percentage(accumulator.unknown_tokens, accumulator.tokens),
        },
    }


def calculate_token_statistics(
    config: TokenStatisticsConfig | None = None,
) -> dict[str, object]:
    """Потоково токенизировать три split и сохранить JSON-отчёт."""

    active_config = config or TokenStatisticsConfig()
    tokenizer = BPETokenizer.load(active_config.tokenizer_file)
    unknown_id = tokenizer.token_to_id(UNK_TOKEN)
    total = _Accumulator()
    split_accumulators = {split: _Accumulator() for split in SPLIT_NAMES}
    board_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)

    for split in SPLIT_NAMES:
        split_path = active_config.splits_dir / f"{split}.jsonl"
        for _, sample in iter_dialogue_jsonl(split_path):
            context_ids = tokenizer.encode_context(sample)
            response_ids = tokenizer.encode_response(sample)
            full_ids = [tokenizer.token_to_id(BOS_TOKEN), *context_ids, *response_ids]
            content_token_count = len(context_ids) - len(sample.context) + len(response_ids) - 2
            for accumulator in (
                split_accumulators[split],
                board_accumulators[sample.board],
                total,
            ):
                accumulator.add(
                    sample,
                    full_ids=full_ids,
                    context_ids=context_ids,
                    response_ids=response_ids,
                    content_token_count=content_token_count,
                    unknown_id=unknown_id,
                    max_sequence_length=active_config.max_sequence_length,
                )

    model_parameters = _model_parameter_count(ModelConfig())
    report: dict[str, object] = {
        "tokenizer_file": active_config.tokenizer_file.as_posix(),
        "vocabulary_size": tokenizer.vocab_size,
        "model_parameters": model_parameters,
        "total_tokens": total.tokens,
        "tokens_per_parameter": round(total.tokens / model_parameters, 6),
        **_accumulator_report(total, active_config.max_sequence_length),
        "splits": {
            split: _accumulator_report(accumulator, active_config.max_sequence_length)
            for split, accumulator in split_accumulators.items()
        },
        "by_board": {
            board: _accumulator_report(accumulator, active_config.max_sequence_length)
            for board, accumulator in sorted(board_accumulators.items())
        },
    }
    write_json(active_config.output_file, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI подсчёта token statistics."""

    parser = argparse.ArgumentParser(description="Посчитать token statistics по split-файлам.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--tokenizer-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--output-file", type=Path, default=Path("data/processed/token_statistics.json")
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить подсчёт статистики из командной строки."""

    args = build_argument_parser().parse_args(argv)
    try:
        report = calculate_token_statistics(
            TokenStatisticsConfig(
                args.splits_dir,
                args.tokenizer_file,
                args.output_file,
                args.max_sequence_length,
            )
        )
    except (DialogueFormatError, TokenizerError, OSError, ValueError) as error:
        print(f"Token statistics не созданы: {error}")
        return 1
    print(
        f"Token statistics сохранены в {args.output_file}; всего tokens: {report['total_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
