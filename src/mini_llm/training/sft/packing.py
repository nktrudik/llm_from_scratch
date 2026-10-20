"""Packing целых диалогов только при поддержке изоляции документов attention-backend-ом."""

from collections.abc import Callable
from typing import cast

from torch import nn

from mini_llm.observability import terminal_log, terminal_stage
from mini_llm.pretrained.dependencies import require_module
from mini_llm.training.sft.data import TokenizedDataset


def packing_available(model: nn.Module, *, requested: bool, device: str) -> bool:
    """Не включать BFD на SDPA: иначе разные диалоги смогут видеть друг друга."""

    if not requested:
        terminal_log("SFT PACKING", "Выключен в конфигурации")
        return False
    trainer = require_module("trl.trainer.sft_trainer")
    variants = getattr(trainer, "FLASH_ATTENTION_VARIANTS", ())
    implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
    supported = isinstance(variants, (set, frozenset, tuple, list)) and implementation in variants
    enabled = device.startswith("cuda") and supported
    terminal_log(
        "SFT PACKING",
        f"requested=True enabled={enabled} attention={implementation}; "
        + ("BFD, целые диалоги" if enabled else "нужен совместимый FlashAttention; без packing"),
    )
    return enabled


def pack_training_dataset(data: TokenizedDataset, *, max_length: int) -> TokenizedDataset:
    """Упаковать готовые input_ids и labels, не отдавая их повторной токенизации TRL."""

    if any(len(data[index]["input_ids"]) > max_length for index in range(len(data))):
        raise ValueError("BFD принимает только целые диалоги в пределах max_length")
    pack = cast(Callable[..., TokenizedDataset], require_module("trl.data_utils").pack_dataset)
    with terminal_stage("SFT PACKING", f"BFD: исходных диалогов {len(data)}"):
        packed = pack(data, seq_length=max_length, strategy="bfd")
    terminal_log("SFT PACKING", f"Исходных диалогов={len(data)} packed_windows={len(packed)}")
    return packed
