"""Ручной sanity-check переобучения модели на малой фиксированной выборке."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from mini_llm.bpe_tokenizer import DEFAULT_TOKENIZER_PATH
from mini_llm.config import ModelConfig
from mini_llm.data_pipeline import DataLoaderConfig, create_split_dataloader
from mini_llm.model import DecoderOnlyTransformer


@dataclass(frozen=True, slots=True)
class OverfitConfig:
    """Параметры короткого ручного overfit experiment."""

    splits_dir: Path = Path("data/processed/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    samples: int = 32
    batch_size: int = 4
    steps: int = 200
    learning_rate: float = 3e-4
    num_workers: int = 0
    random_seed: int = 42
    device: str = "cuda"
    log_interval: int = 10

    def __post_init__(self) -> None:
        positive_values = {
            "samples": self.samples,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "log_interval": self.log_interval,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} должен быть положительным")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate должен быть положительным")
        if self.num_workers < 0:
            raise ValueError("num_workers не может быть отрицательным")


def _select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; передайте --device cpu для явного CPU-запуска")
    return device


def run_overfit_test(config: OverfitConfig | None = None) -> tuple[float, float]:
    """Обучить модель заданное число steps и вернуть initial/final loss."""

    active_config = config or OverfitConfig()
    torch.manual_seed(active_config.random_seed)
    device = _select_device(active_config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(active_config.random_seed)

    dataset, loader = create_split_dataloader(
        "train",
        splits_dir=active_config.splits_dir,
        tokenizer_file=active_config.tokenizer_file,
        loader_config=DataLoaderConfig(
            batch_size=active_config.batch_size,
            num_workers=active_config.num_workers,
            random_seed=active_config.random_seed,
            pin_memory=device.type == "cuda",
        ),
        max_sequence_length=ModelConfig().max_sequence_length,
        max_samples=active_config.samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("В выбранной train-выборке нет пригодных samples")
    if dataset.tokenizer.vocab_size > ModelConfig().vocab_size:
        raise RuntimeError("Vocabulary tokenizer превышает vocabulary текущей модели")

    model_config = ModelConfig(vocab_size=dataset.tokenizer.vocab_size, dropout=0.0)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_config.learning_rate)
    iterator = iter(loader)
    initial_loss: float | None = None
    final_loss = float("nan")

    model.train()
    for step in range(1, active_config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(input_ids, targets)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if initial_loss is None:
            initial_loss = final_loss
        if step == 1 or step % active_config.log_interval == 0 or step == active_config.steps:
            print(f"step={step} loss={final_loss:.6f}")

    if initial_loss is None:
        raise RuntimeError("Overfit test не выполнил ни одного шага")
    print(
        f"Overfit test завершён: initial_loss={initial_loss:.6f}, "
        f"final_loss={final_loss:.6f}, samples={len(dataset)}, device={device}"
    )
    return initial_loss, final_loss


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI ручного overfit test."""

    parser = argparse.ArgumentParser(
        description="Проверить переобучение Transformer на малой train-выборке."
    )
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--tokenizer-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить только явно вызванный пользователем overfit test."""

    args = build_argument_parser().parse_args(argv)
    try:
        run_overfit_test(
            OverfitConfig(
                splits_dir=args.splits_dir,
                tokenizer_file=args.tokenizer_file,
                samples=args.samples,
                batch_size=args.batch_size,
                steps=args.steps,
                learning_rate=args.learning_rate,
                num_workers=args.num_workers,
                random_seed=args.seed,
                device=args.device,
                log_interval=args.log_interval,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Overfit test не выполнен: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
