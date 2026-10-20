"""Совместимые inference-файлы и строгий resume native HF checkpoints."""

from pathlib import Path
from typing import cast

import torch

from mini_llm.pretrained.model import PreparedPretrained
from mini_llm.training.config import LEGACY_SFT_SETTINGS, TrainingConfig
from mini_llm.training.runtime import training_config_payload
from mini_llm.workspace.catalog import update_progress


def resolve_sft_resume(path: Path) -> Path:
    """Не переносить optimizer старого ручного trainer в TRL неявно."""

    payload = cast(object, torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    name = payload.get("sft_checkpoint") if isinstance(payload, dict) else None
    if not isinstance(name, str) or Path(name).name != name or not name.startswith("checkpoint-"):
        raise ValueError(
            "Этот checkpoint создан старым trainer. Для TRL начните новый запуск; "
            "прежние checkpoints остаются доступны для inference."
        )
    directory = path.parent / name
    if directory.resolve().parent != path.parent.resolve():
        raise ValueError("Native checkpoint должен находиться в каталоге запуска")
    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        if not (directory / filename).is_file():
            raise ValueError(f"Неполный checkpoint для resume: отсутствует {filename}")
    return directory


def resume_metadata(config: TrainingConfig, base: object) -> dict[str, object]:
    """Проверить воспроизводимые параметры до загрузки модели и optimizer."""

    if config.resume_from is None:
        return {}
    payload = cast(
        dict[str, object],
        torch.load(config.resume_from, map_location="cpu", weights_only=True, mmap=True),
    )
    saved = payload.get("training_config")
    expected = training_config_payload(config)
    if not isinstance(saved, dict):
        raise ValueError("В checkpoint отсутствует training_config")
    for name in (
        "batch_size",
        "random_seed",
        "learning_rate",
        "max_train_samples",
        "pretrained_engine",
        "splits_dir",
        *LEGACY_SFT_SETTINGS,
    ):
        if saved.get(name, LEGACY_SFT_SETTINGS.get(name)) != expected.get(name):
            raise ValueError(f"Native TRL resume требует исходный {name}")
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("pretrained_config") != base:
        raise ValueError("PretrainedConfig не соответствует native checkpoint")
    return payload


def export_checkpoint(
    config: TrainingConfig,
    prepared: PreparedPretrained,
    *,
    step: int,
    loss: float | None,
    best_loss: float,
    best: bool,
    samples_seen: int,
    tokens_seen: int,
    packing_enabled: bool = False,
) -> Path:
    """Оставить native optimizer/RNG и экспортировать существующий .pt inference-контракт."""

    payload: dict[str, object] = {
        "model_metadata": prepared.model.checkpoint_metadata,
        "model_state_dict": dict(prepared.model.checkpoint_state_dict()),
        "global_step": step,
        "last_validation_loss": loss,
        "best_validation_loss": best_loss,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "training_config": training_config_payload(config),
        "effective_packing": packing_enabled,
        "sft_checkpoint": f"checkpoint-{step}",
    }
    names = [f"step_{step:08d}.pt", "last.pt"]
    if best:
        names.append("best.pt")
    for name in names:
        destination = config.checkpoint_dir / name
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        update_progress(config.checkpoint_dir, step, checkpoint_name=name, validation_loss=loss)
    return config.checkpoint_dir / "last.pt"
