"""CLI подготовки pretrained-модели в локальном Hugging Face cache."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mini_llm.data.config import MAX_BATCH_SIZE
from mini_llm.pretrained.config import PretrainedConfig
from mini_llm.pretrained.dependencies import PretrainedDependencyError
from mini_llm.pretrained.model import prepare_pretrained_model
from mini_llm.pretrained.setup import setup_pretrained_model
from mini_llm.pretrained.workspace import ModelPaths


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать parser автоматической загрузки и совместимой команды prepare."""

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
    setup = subparsers.add_parser("setup", help="Скачать и разложить модель по model ID")
    setup.add_argument("model_id")
    setup.add_argument("--revision", default="main")
    setup.add_argument("--mode", choices=("full", "lora", "qlora"), default="qlora")
    setup.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    setup.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default=None)
    setup.add_argument("--max-sequence-length", type=int, default=1024)
    setup.add_argument("--train", action="store_true", help="После загрузки явно запустить SFT")
    setup.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    setup.add_argument("--batch-size", type=int, choices=range(1, MAX_BATCH_SIZE + 1), default=1)
    setup.add_argument("--epochs", type=int, default=3)
    setup.add_argument("--learning-rate", type=float)
    setup.add_argument("--resume-from", type=Path)
    setup.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _run_setup(args: argparse.Namespace) -> int:
    """Запустить download/config pipeline; обучение возможно только по --train."""

    from mini_llm.training import TrainingConfig, train_model

    try:
        if args.resume_from is not None and not args.train:
            raise ValueError("--resume-from требует --train")
        training_config = None
        if args.train:
            for split in ("train", "validation"):
                if not (args.splits_dir / f"{split}.jsonl").is_file():
                    raise RuntimeError(f"Не найден {split} split в {args.splits_dir}")
            if args.resume_from is not None and not args.resume_from.is_file():
                raise RuntimeError(f"Resume checkpoint не найден: {args.resume_from}")
            paths = ModelPaths.for_model(args.model_id, args.mode)
            training_config = TrainingConfig(
                model_backend="pretrained",
                pretrained_config_file=paths.config_file,
                checkpoint_dir=paths.checkpoint_dir,
                splits_dir=args.splits_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=(
                    args.learning_rate
                    if args.learning_rate is not None
                    else (0.00001 if args.mode == "full" else 0.0002)
                ),
                resume_from=args.resume_from,
                device=args.device,
            )
        registration = setup_pretrained_model(
            args.model_id,
            revision=args.revision,
            mode=args.mode,
            cache_dir=args.cache_dir,
            torch_dtype=args.dtype,
            max_sequence_length=args.max_sequence_length,
        )
        print(
            f"Модель зарегистрирована: {registration.model_id} commit={registration.revision}\n"
            f"Файлы: {registration.snapshot_path}\nКонфиг: {registration.config_file}\n"
            f"Checkpoints: {registration.checkpoint_dir}"
        )
        if training_config is not None:
            result = train_model(training_config)
            print(f"Последний checkpoint: {result.last_checkpoint}")
    except (OSError, RuntimeError, ValueError, PretrainedDependencyError) as error:
        print(f"Подготовка/обучение не завершены: {error}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Сохранить конфигурацию, загрузить и проверить выбранную модель."""

    args = build_argument_parser().parse_args(argv)
    if args.command == "setup":
        return _run_setup(args)
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
