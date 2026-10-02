"""CLI универсального training pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.training.config import PretrainedTrainingConfig, TrainingConfig
from mini_llm.training.pipeline import train_model


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать parser полного обучения или fine-tuning."""

    parser = argparse.ArgumentParser(description="Обучить custom или pretrained causal LM.")
    parser.add_argument("--backend", choices=("custom", "pretrained"), default="custom")
    parser.add_argument("--pretrained-config", type=Path)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument(
        "--tokenizer-file", type=Path, default=Path("artifacts/tokenizer/2ch_bpe.json")
    )
    parser.add_argument(
        "--token-statistics-file",
        type=Path,
        default=Path("data/processed/token_statistics.json"),
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/training"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--batch-size", type=int, choices=range(1, MAX_BATCH_SIZE + 1))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Лимит пригодных train samples; 0 — весь train; pretrained default: 30000",
    )
    duration = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        help="Лимит пригодных validation samples; 0 — весь split; pretrained default: 1000",
    )
    duration.add_argument("--epochs", type=int, default=3)
    duration.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить training pipeline из командной строки."""

    args = build_argument_parser().parse_args(argv)
    epochs = None if args.max_steps is not None else args.epochs
    try:
        defaults = (
            PretrainedTrainingConfig(pretrained_config_file=args.pretrained_config)
            if args.backend == "pretrained"
            else TrainingConfig(batch_size=4)
        )
        config_class = PretrainedTrainingConfig if args.backend == "pretrained" else TrainingConfig
        result = train_model(
            config_class(
                splits_dir=args.splits_dir,
                tokenizer_file=args.tokenizer_file,
                token_statistics_file=args.token_statistics_file,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from,
                model_backend=args.backend,
                pretrained_config_file=args.pretrained_config,
                batch_size=args.batch_size if args.batch_size is not None else defaults.batch_size,
                num_workers=args.num_workers
                if args.num_workers is not None
                else defaults.num_workers,
                max_train_samples=(
                    defaults.max_train_samples
                    if args.max_train_samples is None
                    else (None if args.max_train_samples == 0 else args.max_train_samples)
                ),
                max_validation_samples=(
                    defaults.max_validation_samples
                    if args.max_validation_samples is None
                    else (None if args.max_validation_samples == 0 else args.max_validation_samples)
                ),
                epochs=epochs,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                validation_interval=(
                    args.validation_interval
                    if args.validation_interval is not None
                    else defaults.validation_interval
                ),
                checkpoint_interval=(
                    args.checkpoint_interval
                    if args.checkpoint_interval is not None
                    else defaults.checkpoint_interval
                ),
                validation_batches=(
                    args.validation_batches
                    if args.validation_batches is not None
                    else defaults.validation_batches
                ),
                log_interval=args.log_interval
                if args.log_interval is not None
                else defaults.log_interval,
                random_seed=args.seed,
                device=args.device,
                mixed_precision=not args.no_amp,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Обучение не завершено: {error}")
        return 1
    print(f"Последний checkpoint: {result.last_checkpoint}")
    return 0
