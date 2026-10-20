"""Профиль SFT, упаковка response-only диалогов и защита параметров resume."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from mini_llm.pretrained.config import ATTENTION_MLP_TARGET_MODULES, PretrainedConfig
from mini_llm.training.config import LEGACY_SFT_SETTINGS, PretrainedTrainingConfig, TrainingConfig
from mini_llm.training.progress import calculate_training_plan
from mini_llm.training.runtime import training_config_payload
from mini_llm.training.sft.checkpoints import resume_metadata
from mini_llm.training.sft.config import sft_arguments
from mini_llm.training.sft.data import TokenizedDataset
from mini_llm.training.sft.packing import pack_training_dataset, packing_available
from mini_llm.workspace.training import TrainingParameters


def test_profile_file_and_cli_expose_new_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from mini_llm.training import cli
    from mini_llm.training.schemas import TrainingResult

    parameters = TrainingParameters.model_validate_json(
        Path("configs/training/qlora-qa.json").read_text(encoding="utf-8")
    )
    assert parameters.batch_size == 1 and parameters.gradient_accumulation_steps == 2
    assert parameters.lr_scheduler_type == "cosine" and parameters.warmup_steps == 30
    assert parameters.packing and parameters.lora_target_modules == "attention-mlp"
    calls: list[TrainingConfig] = []

    def train(config: TrainingConfig) -> TrainingResult:
        calls.append(config)
        return TrainingResult(0, 0.0, Path("last.pt"), False)

    monkeypatch.setattr(cli, "train_model", train)
    assert (
        cli.main(
            [
                "--backend",
                "pretrained",
                "--pretrained-config",
                "unused.json",
                "--checkpoint-dir",
                "unused-run",
                "--gradient-accumulation-steps",
                "3",
                "--lr-scheduler-type",
                "constant",
                "--warmup-steps",
                "0",
                "--no-packing",
            ]
        )
        == 0
    )
    assert calls[0].gradient_accumulation_steps == 3
    assert calls[0].lr_scheduler_type == "constant" and calls[0].warmup_steps == 0
    assert not calls[0].packing


def test_accumulation_plan_includes_partial_last_step() -> None:
    plan = calculate_training_plan(5, 1, epochs=2, max_steps=None, gradient_accumulation_steps=2)
    assert plan.steps_per_epoch == 3 and plan.planned_total_steps == 6
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        calculate_training_plan(5, 1, epochs=1, max_steps=None, gradient_accumulation_steps=0)


def test_explicit_legacy_targets_and_custom_behavior_are_preserved() -> None:
    base = PretrainedConfig(model_id="Qwen/Qwen3-0.6B", adaptation_mode="qlora")
    assert base.lora_target_modules == ATTENTION_MLP_TARGET_MODULES
    assert "lm_head" not in base.lora_target_modules
    assert (
        PretrainedConfig(
            model_id=base.model_id, adaptation_mode="qlora", lora_target_modules="all-linear"
        ).lora_target_modules
        == "all-linear"
    )
    assert TrainingConfig().gradient_accumulation_steps == 1
    with pytest.raises(ValueError, match="только для SFT"):
        TrainingConfig(gradient_accumulation_steps=2)


@pytest.mark.parametrize("setting", list(LEGACY_SFT_SETTINGS))
def test_resume_rejects_changed_sft_settings(tmp_path: Path, setting: str) -> None:
    config = PretrainedTrainingConfig(pretrained_config_file=tmp_path / "model.json")
    base = PretrainedConfig(model_id="local/model")
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {
            "training_config": training_config_payload(config),
            "model_metadata": {"pretrained_config": base.model_dump(mode="json")},
        },
        checkpoint,
    )
    original = checkpoint.read_bytes()
    changed = {
        "gradient_accumulation_steps": replace(config, gradient_accumulation_steps=1),
        "lr_scheduler_type": replace(config, lr_scheduler_type="constant"),
        "warmup_steps": replace(config, warmup_steps=0),
        "packing": replace(config, packing=False),
    }[setting]
    with pytest.raises(ValueError, match=setting):
        resume_metadata(
            replace(changed, resume_from=checkpoint),
            base.model_dump(mode="json"),
        )
    assert checkpoint.read_bytes() == original


def test_resume_understands_old_metadata_without_new_fields(tmp_path: Path) -> None:
    config = PretrainedTrainingConfig(
        pretrained_config_file=tmp_path / "model.json",
        gradient_accumulation_steps=1,
        lr_scheduler_type="constant",
        warmup_steps=0,
        packing=False,
    )
    saved = training_config_payload(config)
    for name in LEGACY_SFT_SETTINGS:
        saved.pop(name)
    base = PretrainedConfig(model_id="local/model").model_dump(mode="json")
    checkpoint = tmp_path / "last.pt"
    torch.save(
        {"training_config": saved, "model_metadata": {"pretrained_config": base}}, checkpoint
    )
    assert resume_metadata(replace(config, resume_from=checkpoint), base)


@pytest.mark.parametrize(
    ("implementation", "device", "expected"),
    [
        ("sdpa", "cuda", False),
        ("eager", "cuda", False),
        ("flash_attention_2", "cuda", True),
        ("flash_attention_2", "cpu", False),
    ],
)
def test_packing_requires_document_aware_attention(
    implementation: str, device: str, expected: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("trl")
    model = nn.Linear(1, 1)
    model.__dict__["config"] = SimpleNamespace(_attn_implementation=implementation)
    assert packing_available(model, requested=True, device=device) is expected
    assert f"enabled={expected}" in capsys.readouterr().out
    assert not packing_available(model, requested=False, device=device)


def test_bfd_preserves_complete_pairs_and_response_only_labels() -> None:
    datasets = pytest.importorskip("datasets")
    pytest.importorskip("trl")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    rows = [
        {"input_ids": [11, 12, 13], "labels": [-100, -100, 13]},
        {"input_ids": [21, 22, 23, 24], "labels": [-100, -100, 23, 24]},
        {"input_ids": [31, 32], "labels": [-100, 32]},
    ]
    data = cast(TokenizedDataset, datasets.Dataset.from_list(rows))
    packed = pack_training_dataset(data, max_length=8)
    assert len(packed) == 2
    recovered = []
    for index in range(len(packed)):
        row = packed[index]
        assert len(row["input_ids"]) <= 8
        start = 0
        for length in row["seq_lengths"]:
            recovered.append(
                (
                    tuple(row["input_ids"][start : start + length]),
                    tuple(row["labels"][start : start + length]),
                )
            )
            start += length
    assert Counter(recovered) == Counter((tuple(r["input_ids"]), tuple(r["labels"])) for r in rows)
    collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    batch = collator([packed[index] for index in range(len(packed))])
    assert "attention_mask" not in batch
    assert int((batch["position_ids"] == 0).sum()) == 3
    assert batch["labels"][batch["position_ids"] == 0].eq(-100).all()
    assert Counter(batch["labels"][batch["labels"] != -100].tolist()) == Counter([13, 23, 24, 32])
    with pytest.raises(ValueError, match="целые диалоги"):
        pack_training_dataset(data, max_length=2)


def test_sft_arguments_use_resolved_packing_mode() -> None:
    config = PretrainedTrainingConfig(pretrained_config_file=Path("model.json"))
    base = PretrainedConfig(model_id="Qwen/Qwen3-0.6B", adaptation_mode="qlora")
    args = sft_arguments(config, base, packing=False)
    assert not args["packing"] and not args["eval_packing"]
    assert args["lr_scheduler_type"] == "cosine" and args["warmup_steps"] == 30
    assert args["per_device_train_batch_size"] == 1 and args["gradient_accumulation_steps"] == 2
