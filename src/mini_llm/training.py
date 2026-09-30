"""Однопроцессное обучение decoder-only Transformer с CUDA, AMP и checkpoints."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mini_llm.bpe_tokenizer import DEFAULT_TOKENIZER_PATH, BPETokenizer
from mini_llm.config import ModelConfig
from mini_llm.data_pipeline import (
    CausalLMBatch,
    DataLoaderConfig,
    DialogueDataset,
    create_dataloader,
)
from mini_llm.model import DecoderOnlyTransformer
from mini_llm.training_checkpoint import TrainingState, load_checkpoint, save_checkpoint


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры полного однопроцессного training run."""

    splits_dir: Path = Path("data/processed/splits")
    tokenizer_file: Path = DEFAULT_TOKENIZER_PATH
    checkpoint_dir: Path = Path("checkpoints/training")
    resume_from: Path | None = None
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 3
    max_steps: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    validation_interval: int = 200
    checkpoint_interval: int = 500
    validation_batches: int = 50
    log_interval: int = 10
    random_seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 4:
            raise ValueError("batch_size должен быть в диапазоне от 1 до 4")
        positive_values = {
            "epochs": self.epochs,
            "validation_interval": self.validation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "log_interval": self.log_interval,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} должен быть положительным")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps должен быть положительным")
        if self.num_workers < 0 or self.validation_batches < 0:
            raise ValueError("num_workers и validation_batches не могут быть отрицательными")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate должен быть положительным")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay не может быть отрицательным")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm должен быть положительным")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Краткий результат завершившегося или прерванного запуска."""

    global_step: int
    best_validation_loss: float
    last_checkpoint: Path
    interrupted: bool


def _select_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна; проверьте CUDA-сборку PyTorch и драйвер NVIDIA")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Поддерживаются только устройства cuda и cpu")
    return device


def _training_config_payload(config: TrainingConfig) -> dict[str, object]:
    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "epochs": config.epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "validation_interval": config.validation_interval,
        "checkpoint_interval": config.checkpoint_interval,
        "validation_batches": config.validation_batches,
        "log_interval": config.log_interval,
        "random_seed": config.random_seed,
        "mixed_precision": config.mixed_precision,
    }


def _gpu_telemetry(device: torch.device) -> str:
    if device.type != "cuda":
        return "device=cpu"
    gibibyte = 1024**3
    allocated = torch.cuda.memory_allocated(device) / gibibyte
    reserved = torch.cuda.memory_reserved(device) / gibibyte
    peak = torch.cuda.max_memory_allocated(device) / gibibyte
    return f"VRAM allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"


@torch.no_grad()
def evaluate_validation_loss(
    model: DecoderOnlyTransformer,
    loader: DataLoader[CausalLMBatch],
    *,
    device: torch.device,
    use_amp: bool,
    pad_token_id: int,
    max_batches: int,
) -> float:
    """Посчитать token-weighted validation loss на полном или ограниченном числе batches."""

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(input_ids, targets)
        valid_tokens = int((targets != pad_token_id).sum().item())
        weighted_loss += loss.item() * valid_tokens
        token_count += valid_tokens
    model.train()
    if token_count == 0:
        raise RuntimeError("Validation DataLoader не содержит target tokens")
    return weighted_loss / token_count


def _save_named_checkpoint(
    name: str,
    *,
    config: TrainingConfig,
    model: DecoderOnlyTransformer,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    model_config: ModelConfig,
    state: TrainingState,
) -> Path:
    path = config.checkpoint_dir / name
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        training_config=_training_config_payload(config),
        state=state,
    )
    return path


def train_model(config: TrainingConfig | None = None) -> TrainingResult:
    """Запустить training/validation/checkpoint цикл на одном CUDA или CPU device."""

    active_config = config or TrainingConfig()
    device = _select_device(active_config.device)
    use_amp = active_config.mixed_precision and device.type == "cuda"
    torch.manual_seed(active_config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(active_config.random_seed)
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = BPETokenizer.load(active_config.tokenizer_file)
    model_config = ModelConfig()
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError(
            f"Vocabulary tokenizer ({tokenizer.vocab_size}) не совпадает с ModelConfig "
            f"({model_config.vocab_size})"
        )
    train_dataset = DialogueDataset(
        active_config.splits_dir / "train.jsonl",
        tokenizer,
        max_sequence_length=model_config.max_sequence_length,
    )
    validation_dataset = DialogueDataset(
        active_config.splits_dir / "validation.jsonl",
        tokenizer,
        max_sequence_length=model_config.max_sequence_length,
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise RuntimeError("Train и validation Dataset должны содержать пригодные samples")

    loader_config = DataLoaderConfig(
        batch_size=active_config.batch_size,
        num_workers=active_config.num_workers,
        random_seed=active_config.random_seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = create_dataloader(validation_dataset, loader_config, shuffle=False)
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    state = TrainingState()
    if active_config.resume_from is not None:
        state = load_checkpoint(
            active_config.resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            learning_rate=active_config.learning_rate,
        )

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(
        f"Training start: device={device} ({gpu_name}), AMP={use_amp}, "
        f"train_samples={len(train_dataset)}, validation_samples={len(validation_dataset)}, "
        f"batch_size={active_config.batch_size}, max_sequence_length="
        f"{model_config.max_sequence_length}, start_step={state.global_step}"
    )
    pad_token_id = model_config.pad_token_id
    last_validation_step = -1
    interrupted = False
    stop_requested = False
    loss_since_log = 0.0
    steps_since_log = 0
    tokens_since_log = 0
    log_started = time.perf_counter()

    try:
        for epoch in range(state.epoch, active_config.epochs):
            epoch_loader_config = DataLoaderConfig(
                batch_size=active_config.batch_size,
                num_workers=active_config.num_workers,
                random_seed=active_config.random_seed + epoch,
                pin_memory=device.type == "cuda",
            )
            train_loader = create_dataloader(train_dataset, epoch_loader_config, shuffle=True)
            skip_batches = state.batches_completed_in_epoch if epoch == state.epoch else 0
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if batch_index < skip_batches:
                    continue
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    _, loss = model(input_ids, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), active_config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                state.global_step += 1
                state.epoch = epoch
                state.batches_completed_in_epoch = batch_index + 1
                state.last_train_loss = loss.item()
                loss_since_log += state.last_train_loss
                steps_since_log += 1
                tokens_since_log += int((targets != pad_token_id).sum().item())

                should_log = state.global_step == 1 or (
                    state.global_step % active_config.log_interval == 0
                )
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = max(time.perf_counter() - log_started, 1e-9)
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    print(
                        f"epoch={epoch + 1}/{active_config.epochs} "
                        f"step={state.global_step} "
                        f"train_loss={loss_since_log / steps_since_log:.6f} "
                        f"lr={learning_rate:.6g} tokens/sec={tokens_since_log / elapsed:.1f} "
                        f"{_gpu_telemetry(device)}"
                    )
                    loss_since_log = 0.0
                    steps_since_log = 0
                    tokens_since_log = 0
                    log_started = time.perf_counter()

                if state.global_step % active_config.validation_interval == 0:
                    validation_loss = evaluate_validation_loss(
                        model,
                        validation_loader,
                        device=device,
                        use_amp=use_amp,
                        pad_token_id=pad_token_id,
                        max_batches=active_config.validation_batches,
                    )
                    state.last_validation_loss = validation_loss
                    last_validation_step = state.global_step
                    print(
                        f"validation step={state.global_step} "
                        f"validation_loss={validation_loss:.6f} {_gpu_telemetry(device)}"
                    )
                    if validation_loss < state.best_validation_loss:
                        state.best_validation_loss = validation_loss
                        best_path = _save_named_checkpoint(
                            "best.pt",
                            config=active_config,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            model_config=model_config,
                            state=state,
                        )
                        print(f"Best checkpoint сохранён: {best_path}")

                if state.global_step % active_config.checkpoint_interval == 0:
                    checkpoint_path = _save_named_checkpoint(
                        f"step_{state.global_step:08d}.pt",
                        config=active_config,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        model_config=model_config,
                        state=state,
                    )
                    print(f"Checkpoint сохранён: {checkpoint_path}")

                if (
                    active_config.max_steps is not None
                    and state.global_step >= active_config.max_steps
                ):
                    stop_requested = True
                    break

            if stop_requested:
                break
            state.epoch = epoch + 1
            state.batches_completed_in_epoch = 0
    except KeyboardInterrupt:
        interrupted = True
        print("Получен KeyboardInterrupt; сохраняется last checkpoint...")

    if not interrupted and state.global_step > 0 and last_validation_step != state.global_step:
        validation_loss = evaluate_validation_loss(
            model,
            validation_loader,
            device=device,
            use_amp=use_amp,
            pad_token_id=pad_token_id,
            max_batches=active_config.validation_batches,
        )
        state.last_validation_loss = validation_loss
        if validation_loss < state.best_validation_loss:
            state.best_validation_loss = validation_loss
            _save_named_checkpoint(
                "best.pt",
                config=active_config,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                state=state,
            )
        print(f"final validation_loss={validation_loss:.6f}")

    last_checkpoint = _save_named_checkpoint(
        "last.pt",
        config=active_config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        state=state,
    )
    print(
        f"Training завершён: step={state.global_step}, "
        f"best_validation_loss={state.best_validation_loss:.6f}, "
        f"last_checkpoint={last_checkpoint}, interrupted={interrupted}"
    )
    return TrainingResult(
        state.global_step,
        state.best_validation_loss,
        last_checkpoint,
        interrupted,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Создать CLI полноценного обучения."""

    parser = argparse.ArgumentParser(description="Обучить decoder-only Transformer.")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--tokenizer-file", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/training"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить обучение только по явной команде пользователя."""

    args = build_argument_parser().parse_args(argv)
    try:
        result = train_model(
            TrainingConfig(
                splits_dir=args.splits_dir,
                tokenizer_file=args.tokenizer_file,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=args.resume_from,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                epochs=args.epochs,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                validation_interval=args.validation_interval,
                checkpoint_interval=args.checkpoint_interval,
                validation_batches=args.validation_batches,
                log_interval=args.log_interval,
                random_seed=args.seed,
                device=args.device,
                mixed_precision=not args.no_amp,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Training не запущен или завершился с ошибкой: {error}")
        return 1
    return 130 if result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
