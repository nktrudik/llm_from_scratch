"""Raw и effective token statistics для split-файлов и текущего окна модели."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from mini_llm.data.dialogue import (
    BOS_TOKEN,
    UNK_TOKEN,
    DialogueFormatError,
    DialogueSample,
    iter_dialogue_jsonl,
    percentile,
    write_json,
)
from mini_llm.data.splitting import SPLIT_NAMES
from mini_llm.modeling.config import DEFAULT_MAX_SEQUENCE_LENGTH, ModelConfig
from mini_llm.tokenization import (
    DEFAULT_TOKENIZER_PATH,
    BPETokenizer,
    OversizedResponseError,
    TokenizerError,
)


@dataclass(frozen=True, slots=True)
class TokenStatisticsConfig:
    """Пути и размер окна, используемый текущей моделью."""

    splits_dir: Path = Path("data/processed/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    output_file: Path = Path("data/processed/token_statistics.json")
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length должен быть положительным")


@dataclass(slots=True)
class _Accumulator:
    samples: int = 0
    usable_samples: int = 0
    oversized_response_samples: int = 0
    raw_full_tokens: int = 0
    effective_tokens: int = 0
    training_loss_tokens: int = 0
    characters: int = 0
    content_tokens: int = 0
    unknown_tokens: int = 0
    raw_within_window: int = 0
    full_window_samples: int = 0
    shorter_window_samples: int = 0
    context_truncated_samples: int = 0
    raw_sequence_lengths: list[int] = field(default_factory=list)
    effective_sequence_lengths: list[int] = field(default_factory=list)
    raw_context_lengths: list[int] = field(default_factory=list)
    effective_context_lengths: list[int] = field(default_factory=list)
    response_lengths: list[int] = field(default_factory=list)
    raw_context_tokens: int = 0
    context_content_tokens: int = 0
    usable_raw_context_tokens: int = 0
    effective_context_tokens: int = 0
    response_tokens: int = 0
    response_content_tokens: int = 0
    responses_over_window: int = 0
    context_characters: int = 0
    response_characters: int = 0

    def add(
        self,
        sample: DialogueSample,
        *,
        raw_ids: Sequence[int],
        raw_context_ids: Sequence[int],
        response_ids: Sequence[int],
        effective_ids: Sequence[int] | None,
        unknown_id: int,
        max_sequence_length: int,
    ) -> None:
        """Добавить raw-измерения и, если sample пригоден, effective window."""

        context_characters = sum(len(text) for text in sample.context)
        response_characters = len(sample.response)
        context_content_tokens = len(raw_context_ids) - len(sample.context)
        response_content_tokens = len(response_ids) - 2
        self.samples += 1
        self.raw_full_tokens += len(raw_ids)
        self.characters += context_characters + response_characters
        self.content_tokens += context_content_tokens + response_content_tokens
        self.unknown_tokens += raw_ids.count(unknown_id)
        self.raw_within_window += len(raw_ids) <= max_sequence_length
        self.raw_sequence_lengths.append(len(raw_ids))
        self.raw_context_lengths.append(len(raw_context_ids))
        self.response_lengths.append(len(response_ids))
        self.raw_context_tokens += len(raw_context_ids)
        self.context_content_tokens += context_content_tokens
        self.response_tokens += len(response_ids)
        self.response_content_tokens += response_content_tokens
        self.responses_over_window += len(response_ids) > max_sequence_length
        self.context_characters += context_characters
        self.response_characters += response_characters

        if effective_ids is None:
            self.oversized_response_samples += 1
            return

        effective_context_length = len(effective_ids) - len(response_ids) - 1
        self.usable_samples += 1
        self.usable_raw_context_tokens += len(raw_context_ids)
        self.effective_tokens += len(effective_ids)
        # ASSISTANT служит входным разделителем; loss начинается с текста response и включает EOS.
        self.training_loss_tokens += max(0, len(response_ids) - 1)
        self.effective_context_tokens += effective_context_length
        self.effective_sequence_lengths.append(len(effective_ids))
        self.effective_context_lengths.append(effective_context_length)
        self.full_window_samples += len(effective_ids) == max_sequence_length
        self.shorter_window_samples += len(effective_ids) < max_sequence_length
        self.context_truncated_samples += effective_context_length < len(raw_context_ids)


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
    raw_over_window = accumulator.samples - accumulator.raw_within_window
    dropped_context_tokens = (
        accumulator.usable_raw_context_tokens - accumulator.effective_context_tokens
    )
    return {
        "samples": accumulator.samples,
        "usable_samples": accumulator.usable_samples,
        "unusable_oversized_response_samples": accumulator.oversized_response_samples,
        "raw_full_tokens": accumulator.raw_full_tokens,
        "effective_tokens": accumulator.effective_tokens,
        "training_loss_tokens": accumulator.training_loss_tokens,
        "tokens": accumulator.raw_full_tokens,
        "characters_per_token": round(accumulator.characters / accumulator.content_tokens, 6)
        if accumulator.content_tokens
        else 0.0,
        "raw_sequence_lengths": _distribution(accumulator.raw_sequence_lengths),
        "sample_token_lengths": _distribution(accumulator.raw_sequence_lengths),
        "effective_sequence_lengths": _distribution(accumulator.effective_sequence_lengths),
        "raw_within_max_sequence_length": {
            "max_sequence_length": max_sequence_length,
            "samples": accumulator.raw_within_window,
            "percent": _percentage(accumulator.raw_within_window, accumulator.samples),
        },
        "raw_over_max_sequence_length": {
            "samples": raw_over_window,
            "percent": _percentage(raw_over_window, accumulator.samples),
        },
        "within_max_sequence_length": {
            "max_sequence_length": max_sequence_length,
            "samples": accumulator.raw_within_window,
            "percent": _percentage(accumulator.raw_within_window, accumulator.samples),
        },
        "over_max_sequence_length": {
            "samples": raw_over_window,
            "percent": _percentage(raw_over_window, accumulator.samples),
        },
        "effective_window_usage": {
            "full_window_samples": accumulator.full_window_samples,
            "shorter_window_samples": accumulator.shorter_window_samples,
        },
        "context": {
            "tokens": accumulator.raw_context_tokens,
            "raw_tokens": accumulator.raw_context_tokens,
            "content_tokens": accumulator.context_content_tokens,
            "usable_raw_tokens": accumulator.usable_raw_context_tokens,
            "effective_tokens": accumulator.effective_context_tokens,
            "dropped_tokens": dropped_context_tokens,
            "lost_percent": _percentage(
                dropped_context_tokens, accumulator.usable_raw_context_tokens
            ),
            "truncated_samples": accumulator.context_truncated_samples,
            "raw_token_lengths": _distribution(accumulator.raw_context_lengths),
            "token_lengths": _distribution(accumulator.raw_context_lengths),
            "effective_token_lengths": _distribution(accumulator.effective_context_lengths),
            "characters_per_token": round(
                accumulator.context_characters / accumulator.context_content_tokens,
                6,
            )
            if accumulator.context_content_tokens
            else 0.0,
        },
        "response": {
            "tokens": accumulator.response_tokens,
            "content_tokens": accumulator.response_content_tokens,
            "token_lengths": _distribution(accumulator.response_lengths),
            "responses_over_max_sequence_length": accumulator.responses_over_window,
            "unusable_for_current_window": accumulator.oversized_response_samples,
            "characters_per_token": round(
                accumulator.response_characters / accumulator.response_content_tokens, 6
            )
            if accumulator.response_content_tokens
            else 0.0,
        },
        "unknown_tokens": {
            "count": accumulator.unknown_tokens,
            "percent": _percentage(accumulator.unknown_tokens, accumulator.raw_full_tokens),
        },
    }


def calculate_token_statistics(
    config: TokenStatisticsConfig | None = None,
) -> dict[str, object]:
    """Потоково измерить raw samples и effective окна без изменения split-файлов."""

    active_config = config or TokenStatisticsConfig()
    tokenizer = BPETokenizer.load(active_config.tokenizer_file)
    unknown_id = tokenizer.token_to_id(UNK_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    total = _Accumulator()
    split_accumulators = {split: _Accumulator() for split in SPLIT_NAMES}
    board_accumulators: dict[str, _Accumulator] = defaultdict(_Accumulator)

    for split in SPLIT_NAMES:
        split_path = active_config.splits_dir / f"{split}.jsonl"
        for _, sample in iter_dialogue_jsonl(split_path):
            context_segments = tokenizer.encode_context_segments(sample)
            context_ids = [token_id for segment in context_segments for token_id in segment]
            response_ids = tokenizer.encode_response(sample)
            raw_ids = [bos_id, *context_ids, *response_ids]
            try:
                effective_ids: list[int] | None = tokenizer.build_dialogue_window(
                    context_segments,
                    response_ids,
                    max_length=active_config.max_sequence_length,
                )
            except OversizedResponseError:
                effective_ids = None
            for accumulator in (
                split_accumulators[split],
                board_accumulators[sample.board],
                total,
            ):
                accumulator.add(
                    sample,
                    raw_ids=raw_ids,
                    raw_context_ids=context_ids,
                    response_ids=response_ids,
                    effective_ids=effective_ids,
                    unknown_id=unknown_id,
                    max_sequence_length=active_config.max_sequence_length,
                )

    model_parameters = _model_parameter_count(ModelConfig())
    train_accumulator = split_accumulators["train"]
    total_report = _accumulator_report(total, active_config.max_sequence_length)
    report: dict[str, object] = {
        "tokenizer_file": active_config.tokenizer_file.as_posix(),
        "vocabulary_size": tokenizer.vocab_size,
        "model_parameters": model_parameters,
        "max_sequence_length": active_config.max_sequence_length,
        "training_objective": "response_only",
        "total_tokens": total.raw_full_tokens,
        "raw_full_tokens": total.raw_full_tokens,
        "effective_tokens": total.effective_tokens,
        "tokens_per_parameter": round(total.raw_full_tokens / model_parameters, 6),
        "effective_train_tokens": train_accumulator.training_loss_tokens,
        "effective_train_tokens_per_parameter": round(
            train_accumulator.training_loss_tokens / model_parameters, 6
        ),
        "train_samples_full_window": train_accumulator.full_window_samples,
        "train_samples_shorter_than_window": train_accumulator.shorter_window_samples,
        "oversized_responses_by_board": {
            board: accumulator.oversized_response_samples
            for board, accumulator in sorted(board_accumulators.items())
            if accumulator.oversized_response_samples
        },
        **total_report,
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

    parser = argparse.ArgumentParser(
        description="Посчитать raw и effective token statistics по split-файлам."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--tokenizer-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--output-file", type=Path, default=Path("data/processed/token_statistics.json")
    )
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
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
        f"Token statistics сохранены в {args.output_file}; "
        f"effective train tokens: {report['effective_train_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
