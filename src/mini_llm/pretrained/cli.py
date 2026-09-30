"""CLI подготовки pretrained-модели в локальном Hugging Face cache."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import PretrainedDependencyError
from mini_llm.pretrained.model import prepare_pretrained_model


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать parser команды ``pretrained prepare``."""

    parser = argparse.ArgumentParser(description="Подготовить pretrained causal LM.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Загрузить модель в cache и проверить setup")
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--mode", choices=("full", "lora", "qlora"), default="lora")
    prepare.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    prepare.add_argument("--output-config", type=Path, required=True)
    prepare.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    prepare.add_argument("--max-sequence-length", type=int, default=1024)
    prepare.add_argument("--lora-rank", type=int, default=16)
    prepare.add_argument("--lora-alpha", type=int, default=32)
    prepare.add_argument("--lora-dropout", type=float, default=0.05)
    prepare.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Сохранить конфигурацию, загрузить и проверить выбранную модель."""

    args = build_argument_parser().parse_args(argv)
    config = PretrainedConfig(
        model_id=args.model_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        adaptation_mode=args.mode,
        torch_dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        local_files_only=args.local_files_only,
    )
    config.save(args.output_config)
    try:
        prepared = prepare_pretrained_model(config)
    except (OSError, RuntimeError, ValueError, PretrainedDependencyError) as error:
        print(f"Pretrained model не подготовлена: {error}")
        return 1
    print(
        f"Pretrained model подготовлена: mode={config.adaptation_mode}, "
        f"trainable_parameters={prepared.trainable_parameters:,}, "
        f"total_parameters={prepared.total_parameters:,}, config={args.output_config}"
    )
    return 0
