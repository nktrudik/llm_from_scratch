"""Последовательная подготовка модели по одному Hugging Face model ID."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from mini_llm.pretrained.config import AdaptationMode, PretrainedConfig, TorchDType
from mini_llm.pretrained.dependencies import require_module
from mini_llm.pretrained.download import download_model_snapshot
from mini_llm.pretrained.tokenizer import HuggingFaceDialogueTokenizer
from mini_llm.pretrained.workspace import ACTIVE_MODEL_FILE, ModelPaths, ModelRegistration


def setup_pretrained_model(
    model_id: str,
    *,
    revision: str = "main",
    mode: AdaptationMode = "qlora",
    cache_dir: Path = Path(".cache/huggingface"),
    torch_dtype: TorchDType | None = None,
    max_sequence_length: int = 512,
    gradient_checkpointing: bool | None = None,
) -> ModelRegistration:
    """Скачать файлы, проверить tokenizer и зарегистрировать модель; SFT не запускать."""

    paths = ModelPaths.for_model(model_id, mode)
    # Проверка параметров выполняется до обращения к Hub и загрузки файлов.
    config = PretrainedConfig(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        adaptation_mode=mode,
        torch_dtype=torch_dtype or ("float32" if mode == "full" else "bfloat16"),
        max_sequence_length=max_sequence_length,
        gradient_checkpointing=(
            gradient_checkpointing if gradient_checkpointing is not None else mode != "qlora"
        ),
        local_files_only=True,
    )
    snapshot = download_model_snapshot(model_id, revision, cache_dir)
    config = config.model_copy(update={"revision": snapshot.revision})
    transformers = require_module("transformers")
    load_config = cast(Callable[..., object], transformers.AutoConfig.from_pretrained)
    model_config = load_config(str(snapshot.path), local_files_only=True, trust_remote_code=False)
    if getattr(model_config, "is_encoder_decoder", False):
        raise ValueError("Нужна decoder-only causal language model, а не encoder-decoder")
    load_tokenizer = cast(Callable[..., object], transformers.AutoTokenizer.from_pretrained)
    HuggingFaceDialogueTokenizer(
        load_tokenizer(
            str(snapshot.path), local_files_only=True, trust_remote_code=False, use_fast=True
        )
    )
    if any(paths.checkpoint_dir.glob("*.pt")):
        if not paths.config_file.is_file():
            raise RuntimeError("Есть checkpoints без исходного конфига; setup не перезапишет их.")
        previous_config = PretrainedConfig.load(paths.config_file)
        if previous_config != config:
            raise RuntimeError(
                "Для этой модели и режима уже есть checkpoints с другим конфигом. "
                "Конфиг не перезаписан: выберите другой режим "
                "или сохраните старые артефакты отдельно."
            )
    config.save(paths.config_file)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registration = ModelRegistration(
        model_id=model_id,
        revision=snapshot.revision,
        config_file=paths.config_file,
        checkpoint_dir=paths.checkpoint_dir,
        snapshot_path=snapshot.path,
    )
    registration.save(paths.registration_file)
    registration.save(ACTIVE_MODEL_FILE)
    return registration
