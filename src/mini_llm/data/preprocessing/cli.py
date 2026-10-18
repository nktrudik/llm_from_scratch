"""CLI preprocessing исходных posts либо очищенных SFT-пар."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mini_llm.data.preprocessing.pipeline import preprocess_dataset
from mini_llm.data.preprocessing.schemas import PreprocessingConfig, PreprocessingError
from mini_llm.data.preprocessing.sft import SFTPreprocessingConfig, preprocess_sft_dataset


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить выбранный preprocessing; SFT также создаёт splits."""

    parser = argparse.ArgumentParser(description="Подготовить dialogue dataset из raw JSON.")
    parser.add_argument("--source-format", choices=("posts", "sft"), default="posts")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/staging/2ch"))
    parser.add_argument("--review-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.source_format == "sft":
            report = preprocess_sft_dataset(
                SFTPreprocessingConfig(
                    input_dir=args.input_dir or Path("data/sources/2ch/sft-pairs"),
                    output_dir=args.output_dir,
                    review_size=args.review_size,
                    random_seed=args.seed,
                )
            ).to_dict()
        else:
            report = preprocess_dataset(
                PreprocessingConfig(
                    input_dir=args.input_dir or Path("data/sources/2ch/threads"),
                    output_dir=args.output_dir,
                    review_size=args.review_size,
                    random_seed=args.seed,
                )
            ).to_dict()
    except (PreprocessingError, OSError, ValueError) as error:
        print(f"Preprocessing не выполнен: {error}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
