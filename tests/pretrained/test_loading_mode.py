"""Подготовка training/inference backend на маленькой модели без внешних весов."""

from types import ModuleType, SimpleNamespace

import pytest
from torch import nn

from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained import model as model_module


class TinyTokenizer:
    """Минимальная конфигурация tokenizer для проверки загрузчика."""

    eos_token_id = 2
    eos_token = "<eos>"
    pad_token_id = 0

    def __len__(self) -> int:
        return 16


class TinyModel(nn.Module):
    """Проверить флаги без forward, optimizer или training loop."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.config = SimpleNamespace(use_cache=False, pad_token_id=None)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled = True


@pytest.mark.parametrize("for_inference", [False, True])
def test_preparation_sets_cache_and_checkpointing_flags(
    monkeypatch: pytest.MonkeyPatch, for_inference: bool
) -> None:
    tiny = TinyModel()
    calls: list[dict[str, object]] = []

    def load_model(model_id: str, **kwargs: object) -> nn.Module:
        assert model_id == "org/instruct"
        calls.append(kwargs)
        return tiny

    module = ModuleType("transformers")
    module.__dict__["AutoTokenizer"] = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: TinyTokenizer()
    )
    module.__dict__["AutoModelForCausalLM"] = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setattr(model_module, "require_module", lambda name: module)
    config = PretrainedConfig(model_id="org/instruct", revision="a" * 40, local_files_only=True)
    prepared = model_module.prepare_pretrained_model(config, for_inference=for_inference)

    assert prepared.model.module is tiny
    assert tiny.config.use_cache == for_inference
    assert tiny.checkpointing_enabled != for_inference
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["local_files_only"] is True
