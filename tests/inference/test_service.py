"""Тесты dispatch генерации без загрузки pretrained weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor

from mini_llm.inference import GenerationConfig
from mini_llm.inference import service as inference_service
from mini_llm.inference.schemas import GenerationResult
from mini_llm.pretrained import (
    PreparedPretrained,
    PretrainedConfig,
    PretrainedGenerationResult,
)


class _FakePretrainedBackend:
    def __init__(self, metadata: dict[str, object]) -> None:
        self.checkpoint_metadata = metadata
        self.loaded_state: Mapping[str, Tensor] | None = None
        self.device: torch.device | None = None
        self.eval_called = False

    def load_checkpoint_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.loaded_state = state

    def to(self, device: torch.device) -> None:
        self.device = device

    def eval(self) -> None:
        self.eval_called = True


def test_default_generation_backend_remains_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_custom(
        prompt: str,
        config: GenerationConfig,
        device: torch.device,
    ) -> GenerationResult:
        calls.append(f"{prompt}:{config.model_backend}:{device.type}")
        return GenerationResult("custom", [1])

    monkeypatch.setattr(inference_service, "_generate_custom", fake_custom)

    result = inference_service.generate_response(
        "Тест",
        GenerationConfig(device="cpu"),
    )

    assert calls == ["Тест:custom:cpu"]
    assert result == GenerationResult("custom", [1])


def test_pretrained_generation_loads_config_checkpoint_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrained_config = PretrainedConfig(
        model_id="org/model",
        revision="commit-hash",
        adaptation_mode="lora",
        local_files_only=True,
    )
    config_file = tmp_path / "pretrained.json"
    pretrained_config.save(config_file)
    metadata: dict[str, object] = {
        "backend": "pretrained",
        "pretrained_config": pretrained_config.model_dump(mode="json"),
    }
    checkpoint_file = tmp_path / "best.pt"
    expected_state = {"adapter.weight": torch.tensor([1.0, 2.0])}
    torch.save(
        {
            "format_version": 3,
            "model_metadata": metadata,
            "model_state_dict": expected_state,
        },
        checkpoint_file,
    )
    backend = _FakePretrainedBackend(metadata)
    prepared = cast(
        PreparedPretrained,
        SimpleNamespace(model=backend, tokenizer=object()),
    )
    loaded_configs: list[PretrainedConfig] = []

    def fake_prepare(config: PretrainedConfig) -> PreparedPretrained:
        loaded_configs.append(config)
        return prepared

    def fake_generate(
        prepared_model: PreparedPretrained,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> PretrainedGenerationResult:
        assert prepared_model is prepared
        assert prompt == "Тест"
        assert (max_new_tokens, temperature, top_k) == (7, 0.5, 4)
        assert backend.eval_called
        return PretrainedGenerationResult("ответ", [10, 11])

    monkeypatch.setattr(inference_service, "prepare_pretrained_model", fake_prepare)
    monkeypatch.setattr(inference_service, "generate_pretrained", fake_generate)

    result = inference_service.generate_response(
        "Тест",
        GenerationConfig(
            checkpoint_file=checkpoint_file,
            tokenizer_file=None,
            device="cpu",
            max_new_tokens=7,
            temperature=0.5,
            top_k=4,
            model_backend="pretrained",
            pretrained_config_file=config_file,
        ),
    )

    assert loaded_configs == [pretrained_config]
    assert backend.loaded_state is not None
    torch.testing.assert_close(
        backend.loaded_state["adapter.weight"], expected_state["adapter.weight"]
    )
    assert backend.device == torch.device("cpu")
    assert result.text == "ответ"
    assert result.token_ids == [10, 11]


def test_pretrained_generation_rejects_checkpoint_from_other_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pretrained_config = PretrainedConfig(model_id="org/model", local_files_only=True)
    config_file = tmp_path / "pretrained.json"
    pretrained_config.save(config_file)
    checkpoint_file = tmp_path / "wrong.pt"
    torch.save(
        {
            "format_version": 3,
            "model_metadata": {"backend": "custom"},
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        checkpoint_file,
    )
    metadata: dict[str, object] = {
        "backend": "pretrained",
        "pretrained_config": pretrained_config.model_dump(mode="json"),
    }
    prepared = cast(
        PreparedPretrained,
        SimpleNamespace(model=_FakePretrainedBackend(metadata), tokenizer=object()),
    )
    monkeypatch.setattr(inference_service, "prepare_pretrained_model", lambda _: prepared)

    with pytest.raises(RuntimeError, match="не совпадает"):
        inference_service.generate_response(
            "Тест",
            GenerationConfig(
                checkpoint_file=checkpoint_file,
                tokenizer_file=None,
                device="cpu",
                model_backend="pretrained",
                pretrained_config_file=config_file,
            ),
        )
