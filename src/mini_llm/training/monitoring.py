"""Validation и телеметрия процесса обучения."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mini_llm.data_pipeline import CausalLMBatch
from mini_llm.model import DecoderOnlyTransformer


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
    model: DecoderOnlyTransformer,
    loader: DataLoader[CausalLMBatch],
    *,
    device: torch.device,
    use_amp: bool,
    pad_token_id: int,
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
        targets = batch["targets"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(input_ids, targets)
        valid_tokens = int((targets != pad_token_id).sum().item())
        weighted_loss += loss.item() * valid_tokens
        token_count += valid_tokens
    model.train()
    if token_count == 0:
        raise RuntimeError("Validation DataLoader не содержит target tokens")
    return weighted_loss / token_count
