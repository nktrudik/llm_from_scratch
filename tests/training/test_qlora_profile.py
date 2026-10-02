"""Профиль QLoRA, precision, optimizer и ограничение train без GPU и внешних весов."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from mini_llm.api.schemas import PretrainedPrepareRequest, TrainingRequest
from mini_llm.data.dialogue import DialogueSample
from mini_llm.data.interfaces import EncodedDialogue
from mini_llm.modeling import CustomCausalLMBackend, DecoderOnlyTransformer, ModelConfig
from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained.model import HuggingFaceCausalLMBackend
from mini_llm.training import (
    PretrainedTrainingConfig,
    TrainingConfig,
    cli,
    components,
    optimizers,
    pipeline,
)
from mini_llm.training.checkpoints import load_checkpoint, save_checkpoint
from mini_llm.training.components import TrainingComponents
from mini_llm.training.optimizers import create_optimizer, optimizer_name
from mini_llm.training.precision import create_grad_scaler, validate_pretrained_training_device
from mini_llm.training.schemas import TrainingResult, TrainingState


class TinyTokenizer:
    """Фиксированная causal-пара для synthetic smoke test."""

    vocab_size = 16
    pad_token_id = 0

    def encode_training_window(self, sample: DialogueSample, *, max_length: int) -> EncodedDialogue:
        return EncodedDialogue([1, 6, 7, 8, 2], response_start=3)


def tiny_backend() -> CustomCausalLMBackend:
    """Создать CPU-модель без pretrained downloads."""

    config = ModelConfig(
        vocab_size=16, max_sequence_length=512, num_layers=1, d_model=8, num_heads=2, dropout=0.0
    )
    return CustomCausalLMBackend(DecoderOnlyTransformer(config), config)


def test_pretrained_profile_and_explicit_overrides() -> None:
    config = PretrainedTrainingConfig(pretrained_config_file=Path("model.json"))
    assert config.batch_size == config.num_workers == 2
    assert config.max_train_samples == 30_000
    assert config.validation_interval == config.checkpoint_interval == 500
    assert config.validation_batches == 50
    assert config.log_interval == 100
    assert TrainingConfig().max_train_samples is None
    assert (
        PretrainedTrainingConfig(
            pretrained_config_file=Path("model.json"), max_train_samples=None
        ).max_train_samples
        is None
    )
    with pytest.raises(ValueError, match="max_train_samples"):
        TrainingConfig(max_train_samples=0)


@pytest.mark.parametrize("limit", [None, 0, 17])
def test_training_cli_uses_backend_defaults_and_full_dataset_flag(
    monkeypatch: pytest.MonkeyPatch, limit: int | None
) -> None:
    calls: list[TrainingConfig] = []

    def fake_train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        return TrainingResult(0, 0.0, Path("last.pt"), False)

    monkeypatch.setattr(cli, "train_model", fake_train)
    args = ["--backend", "pretrained", "--pretrained-config", "model.json"]
    if limit is not None:
        args += ["--max-train-samples", str(limit)]
    assert cli.main(args) == 0
    assert calls[0].batch_size == 2
    assert calls[0].num_workers == 2
    assert calls[0].max_train_samples == (30_000 if limit is None else limit or None)
    assert calls[0].validation_interval == 500
    assert calls[0].validation_batches == 50


def test_api_uses_pretrained_profile_without_changing_custom_defaults() -> None:
    config = TrainingRequest(model_backend="pretrained", pretrained_config_file=Path("model.json"))
    assert config.to_config().batch_size == 2
    assert config.to_config().max_train_samples == 30_000
    assert TrainingRequest().to_config().batch_size == TrainingConfig().batch_size
    full = TrainingRequest(
        model_backend="pretrained",
        pretrained_config_file=Path("model.json"),
        max_train_samples=None,
        num_workers=0,
        batch_size=1,
    )
    assert full.to_config().max_train_samples is None
    assert full.to_config().num_workers == 0
    assert full.to_config().batch_size == 1
    model_config = PretrainedPrepareRequest(
        model_id="org/model", adaptation_mode="qlora"
    ).to_config()
    assert model_config.max_sequence_length == 512
    assert model_config.torch_dtype == "bfloat16"
    assert not model_config.gradient_checkpointing


@pytest.mark.parametrize("supported", [True, False])
def test_bf16_support_is_checked_without_emulation(
    monkeypatch: pytest.MonkeyPatch, supported: bool
) -> None:
    checks: list[bool] = []

    def is_supported(*, including_emulation: bool) -> bool:
        checks.append(including_emulation)
        return supported

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", is_supported)
    config = PretrainedConfig(model_id="org/model", torch_dtype="bfloat16", adaptation_mode="qlora")
    if supported:
        validate_pretrained_training_device(config, torch.device("cuda"), mixed_precision=True)
    else:
        with pytest.raises(RuntimeError, match="fallback отключён"):
            validate_pretrained_training_device(config, torch.device("cuda"), mixed_precision=True)
    assert checks == [False]


def test_explicit_fp16_does_not_require_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_check(*args: object, **kwargs: object) -> bool:
        pytest.fail("FP16 не должен проверять BF16 или менять dtype")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected_check)
    config = PretrainedConfig(model_id="org/model", torch_dtype="float16", adaptation_mode="qlora")
    validate_pretrained_training_device(config, torch.device("cuda"), mixed_precision=True)
    with pytest.raises(RuntimeError, match="требует CUDA"):
        validate_pretrained_training_device(config, torch.device("cpu"), mixed_precision=True)


def test_unsupported_bf16_fails_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "model.json"
    PretrainedConfig(model_id="org/model", adaptation_mode="qlora").save(path)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: False)

    def unexpected_loading(*args: object, **kwargs: object) -> None:
        pytest.fail("Веса не должны загружаться до проверки BF16")

    monkeypatch.setattr(components, "prepare_pretrained_model", unexpected_loading)
    with pytest.raises(RuntimeError, match="fallback отключён"):
        components.create_training_components(
            PretrainedTrainingConfig(pretrained_config_file=path), torch.device("cuda")
        )


def test_bf16_disables_scaler_but_fp16_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not create_grad_scaler(use_amp=True, amp_dtype=torch.bfloat16).is_enabled()
    assert create_grad_scaler(use_amp=True, amp_dtype=torch.float16).is_enabled()
    assert not create_grad_scaler(use_amp=False, amp_dtype=torch.float16).is_enabled()


@pytest.mark.parametrize("mode", ["full", "lora", "qlora"])
def test_optimizer_is_8bit_only_for_qlora(
    monkeypatch: pytest.MonkeyPatch, mode: Literal["full", "lora", "qlora"]
) -> None:
    class AdamW8bit(AdamW):
        """Подмена внешнего optimizer для CPU-проверки выбора и параметров."""

    module = ModuleType("bitsandbytes.optim")
    module.__dict__["AdamW8bit"] = AdamW8bit
    monkeypatch.setattr(optimizers, "require_module", lambda name: module)
    backend = HuggingFaceCausalLMBackend(
        nn.Linear(8, 8), PretrainedConfig(model_id="org/model", adaptation_mode=mode)
    )
    config = TrainingConfig(learning_rate=0.0002, weight_decay=0.03)
    optimizer = create_optimizer(backend, config)
    assert type(optimizer).__name__ == ("AdamW8bit" if mode == "qlora" else "AdamW")
    assert optimizer.param_groups[0]["lr"] == config.learning_rate
    assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
    custom = tiny_backend()
    assert optimizer_name(custom) == "AdamW"
    assert type(create_optimizer(custom, config)) is AdamW


def test_train_limit_leaves_validation_and_jsonl_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "board": "b",
        "thread_id": 1,
        "context": [{"post_id": 1, "text": "Вопрос"}],
        "response": {"post_id": 2, "text": "Ответ"},
    }
    text = (json.dumps(payload, ensure_ascii=False) + "\n") * 5
    for split in ("train", "validation"):
        (tmp_path / f"{split}.jsonl").write_text(text, encoding="utf-8")
    backend = tiny_backend()
    monkeypatch.setattr(
        pipeline,
        "create_training_components",
        lambda config, device: TrainingComponents(backend, TinyTokenizer()),
    )
    config = TrainingConfig(
        splits_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        token_statistics_file=tmp_path / "missing.json",
        batch_size=2,
        max_train_samples=3,
        epochs=1,
        device="cpu",
        log_interval=1,
    )
    result = pipeline.train_model(config)
    output = capsys.readouterr().out
    assert result.global_step == 2
    assert "train_samples=3, validation_samples=5" in output
    assert "max_train_samples=3 num_workers=0" in output
    assert "seq_length=512 batch_size=2" in output
    assert "effective_train_tokens=6" in output
    for split in ("train", "validation"):
        assert (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8") == text
    resumed = pipeline.train_model(
        TrainingConfig(
            splits_dir=tmp_path,
            checkpoint_dir=tmp_path / "checkpoints",
            resume_from=result.last_checkpoint,
            token_statistics_file=tmp_path / "missing.json",
            batch_size=2,
            max_train_samples=3,
            epochs=2,
            device="cpu",
        )
    )
    assert resumed.global_step == 4
    state = cast(dict[str, object], torch.load(resumed.last_checkpoint, weights_only=True))
    assert state["samples_seen"] == 6


@pytest.mark.parametrize("changed", ["limit", "optimizer"])
def test_resume_rejects_incompatible_train_limit_or_optimizer(tmp_path: Path, changed: str) -> None:
    class AdamW8bit(AdamW):
        """Имитация другого типа optimizer без bitsandbytes или CUDA."""

    backend = tiny_backend()
    scaler = create_grad_scaler(use_amp=False, amp_dtype=torch.float16)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=backend,
        optimizer=AdamW(backend.trainable_parameters()),
        scaler=scaler,
        training_config={"batch_size": 2, "max_train_samples": 3},
        state=TrainingState(epoch=0, batches_completed_in_epoch=1, global_step=1),
    )
    optimizer = (AdamW8bit if changed == "optimizer" else AdamW)(backend.trainable_parameters())
    with pytest.raises(RuntimeError, match="Optimizer checkpoint|max_train_samples"):
        load_checkpoint(
            path,
            model=backend,
            optimizer=optimizer,
            scaler=scaler,
            learning_rate=0.0002,
            expected_batch_size=2,
            expected_max_train_samples=4 if changed == "limit" else 3,
        )


def test_oom_is_reported_without_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fail(config: TrainingConfig) -> TrainingResult:
        calls.append(config.batch_size)
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(pipeline, "_train_model", fail)
    with pytest.raises(RuntimeError, match="автоматический fallback отключён"):
        pipeline.train_model(PretrainedTrainingConfig(pretrained_config_file=Path("model.json")))
    assert calls == [2]
