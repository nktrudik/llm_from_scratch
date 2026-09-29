"""Детерминированное разбиение dialogue dataset по тредам."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from mini_llm.dialogue_format import DialogueFormatError, iter_dialogue_jsonl, write_json

SPLIT_NAMES = ("train", "validation", "test")
ThreadKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Пути и seed этапа split."""

    input_file: Path = Path("data/processed/2ch_dialogues.jsonl")
    output_dir: Path = Path("data/processed/splits")
    random_seed: int = 42


def _thread_sort_key(thread: ThreadKey, seed: int) -> tuple[bytes, str, int]:
    board, thread_id = thread
    value = f"{seed}:{board}:{thread_id}".encode()
    digest = hashlib.blake2b(value, digest_size=16).digest()
    return digest, board, thread_id


def assign_thread_splits(threads: set[ThreadKey], seed: int) -> dict[ThreadKey, str]:
    """Распределить уникальные треды в точной пропорции 90/5/5."""

    ordered = sorted(threads, key=lambda thread: _thread_sort_key(thread, seed))
    thread_count = len(ordered)
    train_end = int(thread_count * 0.90)
    validation_end = train_end + int(thread_count * 0.05)
    assignments: dict[ThreadKey, str] = {}
    for index, thread in enumerate(ordered):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        assignments[thread] = split
    return assignments


def _open_temporary_outputs(output_dir: Path) -> tuple[dict[str, TextIO], dict[str, Path]]:
    handles: dict[str, TextIO] = {}
    paths: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        destination = output_dir / f"{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        handles[split] = temporary.open("w", encoding="utf-8", newline="\n")
        paths[split] = temporary
    return handles, paths


def split_dataset(config: SplitConfig | None = None) -> dict[str, object]:
    """Двумя потоковыми проходами создать split-файлы без thread leakage."""

    active_config = config or SplitConfig()
    threads = {
        (sample.board, sample.thread_id)
        for _, sample in iter_dialogue_jsonl(active_config.input_file)
    }
    assignments = assign_thread_splits(threads, active_config.random_seed)
    active_config.output_dir.mkdir(parents=True, exist_ok=True)

    sample_counts: Counter[str] = Counter()
    thread_sets: dict[str, set[ThreadKey]] = {name: set() for name in SPLIT_NAMES}
    board_samples: dict[str, Counter[str]] = defaultdict(Counter)
    board_threads: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {name: set() for name in SPLIT_NAMES}
    )
    handles, temporary_paths = _open_temporary_outputs(active_config.output_dir)
    try:
        for payload, sample in iter_dialogue_jsonl(active_config.input_file):
            thread = (sample.board, sample.thread_id)
            split = assignments[thread]
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            sample_counts[split] += 1
            thread_sets[split].add(thread)
            board_samples[sample.board][split] += 1
            board_threads[sample.board][split].add(sample.thread_id)
    finally:
        for handle in handles.values():
            handle.close()

    for split, temporary_path in temporary_paths.items():
        temporary_path.replace(active_config.output_dir / f"{split}.jsonl")

    per_board: dict[str, object] = {}
    for board in sorted(board_samples):
        per_board[board] = {
            split: {
                "threads": len(board_threads[board][split]),
                "samples": board_samples[board][split],
            }
            for split in SPLIT_NAMES
        }
    report: dict[str, object] = {
        "seed": active_config.random_seed,
        "ratios": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "source_file": active_config.input_file.as_posix(),
        "splits": {
            split: {
                "threads": len(thread_sets[split]),
                "samples": sample_counts[split],
                "file": (active_config.output_dir / f"{split}.jsonl").as_posix(),
            }
            for split in SPLIT_NAMES
        },
        "by_board": per_board,
    }
    write_json(active_config.output_dir / "split_stats.json", report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI этапа split."""

    parser = argparse.ArgumentParser(description="Разбить dialogue dataset по тредам.")
    parser.add_argument(
        "--input-file", type=Path, default=Path("data/processed/2ch_dialogues.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить split из командной строки."""

    args = build_argument_parser().parse_args(argv)
    try:
        report = split_dataset(SplitConfig(args.input_file, args.output_dir, args.seed))
    except (DialogueFormatError, OSError, ValueError) as error:
        print(f"Split не выполнен: {error}")
        return 1
    splits = report["splits"]
    print(f"Split-файлы созданы в {args.output_dir}: {splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
