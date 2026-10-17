"""Распределение длин полных пар с локальным pretrained tokenizer, без загрузки весов."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from mini_llm.data.dialogue import iter_dialogue_jsonl, percentile, write_json
from mini_llm.data.interfaces import DialogueTokenizer
from mini_llm.observability import terminal_log
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer


def analyze_pair_lengths(path: Path, tokenizer: DialogueTokenizer) -> dict[str, object]:
    """Считать длины без усечения и фильтрации; special tokens входят в полную пару."""

    lengths: list[int] = []
    threads: set[tuple[str, int]] = set()
    context_sizes: Counter[int] = Counter()
    for _, sample in iter_dialogue_jsonl(path):
        encoded = tokenizer.encode_training_window(sample, max_length=sys.maxsize)
        lengths.append(len(encoded.token_ids))
        threads.add((sample.board, sample.thread_id))
        context_sizes[len(sample.context)] += 1
        if len(lengths) % 500 == 0:
            terminal_log("LENGTHS", f"file={path.name} processed={len(lengths)} samples")
    if not lengths:
        raise ValueError(f"В {path} нет samples")
    lengths.sort()
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "file": path.as_posix(),
        "sha256": digest,
        "samples": len(lengths),
        "threads": len(threads),
        "context_message_counts": dict(sorted(context_sizes.items())),
        "full_pair_tokens": {
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": lengths[-1],
        },
        "fits_windows": {
            str(window): {
                "samples": sum(length <= window for length in lengths),
                "percentage": 100 * sum(length <= window for length in lengths) / len(lengths),
            }
            for window in (512, 768, 1024)
        },
    }


def load_local_tokenizer(config: PretrainedConfig) -> HuggingFaceDialogueTokenizer:
    """Загрузить только tokenizer закреплённой revision, без сети и model weights."""

    transformers = require_module("transformers")
    load = cast(Callable[..., object], transformers.AutoTokenizer.from_pretrained)
    return HuggingFaceDialogueTokenizer(
        load(
            config.model_id,
            revision=config.revision,
            cache_dir=str(config.cache_dir),
            local_files_only=True,
            trust_remote_code=config.trust_remote_code,
            use_fast=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Проанализировать JSONL только на CPU; сохранить отдельный JSON-отчёт по запросу."""

    parser = argparse.ArgumentParser(description="Посчитать длины полных non-thinking пар.")
    parser.add_argument("--pretrained-config", type=Path, required=True)
    parser.add_argument("--input-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args(argv)
    try:
        sources = {path.resolve() for path in args.input_files}
        if args.output_file is not None and args.output_file.resolve() in (
            sources | {args.pretrained_config.resolve()}
        ):
            raise ValueError("Файл отчёта не должен совпадать с исходным JSONL или конфигом")
        config = PretrainedConfig.load(args.pretrained_config)
        terminal_log(
            "LENGTHS", f"Локальный tokenizer: {config.model_id} revision={config.revision}"
        )
        tokenizer = load_local_tokenizer(config)
        report: dict[str, object] = {
            "model_id": config.model_id,
            "revision": config.revision,
            "enable_thinking": False,
            "length_definition": (
                "full prompt + full response + native special tokens; no truncation"
            ),
            "percentile_method": "nearest_rank",
            "datasets": [analyze_pair_lengths(path, tokenizer) for path in args.input_files],
        }
        if args.output_file is not None:
            write_json(args.output_file, report)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Анализ длин не выполнен: {error}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
