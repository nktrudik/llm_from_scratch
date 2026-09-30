"""Тесты конфигурации pretrained-моделей без загрузки весов."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_llm.pretrained import PretrainedConfig


def test_pretrained_config_round_trip_preserves_revision_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "pretrained.json"
    config = PretrainedConfig(
        model_id="org/model",
        revision="0123456789abcdef",
        adaptation_mode="qlora",
        torch_dtype="float16",
        lora_rank=8,
        lora_target_modules=("q_proj", "v_proj"),
    )

    config.save(path)
    restored = PretrainedConfig.load(path)

    assert restored == config
    assert restored.revision == "0123456789abcdef"
    assert restored.cache_dir == Path(".cache/huggingface")


def test_pretrained_config_rejects_invalid_qlora_dtype() -> None:
    with pytest.raises(ValidationError, match="QLoRA"):
        PretrainedConfig(
            model_id="org/model",
            adaptation_mode="qlora",
            torch_dtype="float32",
        )
