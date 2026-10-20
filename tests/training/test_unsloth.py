"""Контракт опционального Unsloth без импорта GPU-ядер и загрузки весов."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn

from mini_llm.pretrained.config import ATTENTION_MLP_TARGET_MODULES, PretrainedConfig
from mini_llm.training.sft import unsloth


@pytest.mark.parametrize("quantized", [True, False])
def test_unsloth_preserves_model_revision_quantization_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
    quantized: bool,
) -> None:
    class Tiny(nn.Linear):
        is_loaded_in_4bit = quantized

    model = Tiny(2, 2)
    load_args: dict[str, object] = {}
    adapter_args: dict[str, object] = {}

    def load(**kwargs: object) -> tuple[object, object]:
        load_args.update(kwargs)
        return model, SimpleNamespace(eos_token_id=2, eos_token="eos", pad_token_id=0)

    def adapt(value: object, **kwargs: object) -> object:
        assert value is model
        adapter_args.update(kwargs)
        return model

    module = ModuleType("unsloth")
    module.__dict__["FastLanguageModel"] = SimpleNamespace(
        from_pretrained=load, get_peft_model=adapt
    )
    monkeypatch.setitem(sys.modules, "unsloth", module)
    monkeypatch.setattr(unsloth, "require_module", lambda _: module)
    config = PretrainedConfig(
        model_id="Qwen/Qwen3-0.6B",
        revision="a" * 40,
        adaptation_mode="qlora",
        local_files_only=True,
    )
    if not quantized:
        with pytest.raises(RuntimeError, match="fallback запрещён"):
            unsloth.prepare_unsloth(config, random_seed=42)
        return
    result = unsloth.prepare_unsloth(config, random_seed=42)
    assert result.model.module is model
    assert load_args["model_name"] == config.model_id
    assert load_args["revision"] == config.revision
    assert load_args["local_files_only"] is True
    assert load_args["use_exact_model_name"] is True
    assert load_args["load_in_4bit"] is True
    assert load_args["use_gradient_checkpointing"] is False
    assert adapter_args["r"] == 8 and adapter_args["lora_alpha"] == 16
    assert adapter_args["target_modules"] == ATTENTION_MLP_TARGET_MODULES
    assert adapter_args["random_state"] == 42


def test_unsloth_requires_clean_import_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)
    with pytest.raises(RuntimeError, match="Перезапустите API"):
        unsloth.prepare_unsloth(
            PretrainedConfig(model_id="org/model", adaptation_mode="qlora"), random_seed=42
        )
