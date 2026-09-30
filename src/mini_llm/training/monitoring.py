"""Validation и телеметрия процесса обучения."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mini_llm.data.dataset import IGNORE_INDEX, CausalLMBatch
from mini_llm.modeling import CausalLMBackend


def gpu_telemetry(device: torch.device) -> str:
    """Вернуть краткую строку использования памяти CUDA или имя CPU-устройства."""

    if device.type != "cuda":
        return "device=cpu"
    gibibyte = 1024**3
    allocated = torch.cuda.memory_allocated(device) / gibibyte
    reserved = torch.cuda.memory_reserved(device) / gibibyte
    peak = torch.cuda.max_memory_allocated(device) / gibibyte
    return f"VRAM allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"


@torch.no_grad()
def evaluate_validation_loss(
    model: CausalLMBackend,
    loader: DataLoader[CausalLMBatch],
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int,
) -> float:
    """Посчитать взвешенный по токенам validation loss."""

    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = model.forward_batch(input_ids, attention_mask, labels)
        valid_tokens = int((labels != IGNORE_INDEX).sum().item())
        weighted_loss += output.loss.item() * valid_tokens
        token_count += valid_tokens
    model.train()
    if token_count == 0:
        raise RuntimeError("Validation DataLoader не содержит target tokens")
    return weighted_loss / token_count
