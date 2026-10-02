"""Подготовка training/inference backend на маленькой модели без внешних весов."""

from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from mini_llm.pretrained import PretrainedConfig
from mini_llm.pretrained import model as model_module
from mini_llm.pretrained.config import TorchDType


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


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_qlora_compute_dtype_and_lora_parameters_are_preserved(
    monkeypatch: pytest.MonkeyPatch, dtype: TorchDType
) -> None:
    tiny = TinyModel()
    quantization: dict[str, object] = {}
    adaptation: dict[str, object] = {}
    checkpointing: list[bool] = []

    def quantization_config(**kwargs: object) -> object:
        quantization.update(kwargs)
        return kwargs

    def lora_config(**kwargs: object) -> object:
        adaptation.update(kwargs)
        return kwargs

    def prepare_kbit(model: nn.Module, *, use_gradient_checkpointing: bool) -> nn.Module:
        checkpointing.append(use_gradient_checkpointing)
        return model

    transformers = ModuleType("transformers")
    transformers.__dict__.update(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tiny),
        BitsAndBytesConfig=quantization_config,
    )
    peft = ModuleType("peft")
    peft.__dict__.update(
        prepare_model_for_kbit_training=prepare_kbit,
        LoraConfig=lora_config,
        get_peft_model=lambda model, config: model,
    )
    monkeypatch.setattr(
        model_module,
        "require_module",
        lambda name: {"transformers": transformers, "peft": peft}[name],
    )
    config = PretrainedConfig(model_id="org/instruct", adaptation_mode="qlora", torch_dtype=dtype)
    prepared = model_module.prepare_pretrained_model(config)
    expected_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    assert quantization == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": expected_dtype,
    }
    assert adaptation == {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "bias": "none",
    }
    assert checkpointing == [False]
    assert not tiny.checkpointing_enabled
    assert prepared.model.autocast_dtype == expected_dtype
    assert prepared.model.max_sequence_length == 512
