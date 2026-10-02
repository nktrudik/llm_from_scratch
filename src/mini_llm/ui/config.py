"""Настройки HTTP-клиента и доступные backend интерфейса."""

import os
from typing import Literal

ModelBackend = Literal["custom", "pretrained"]
PretrainedMode = Literal["before_sft", "after_sft", "custom_checkpoint"]
MODEL_BACKENDS: tuple[ModelBackend, ...] = ("custom", "pretrained")
MODEL_LABELS = {"custom": "Custom", "pretrained": "Qwen"}
MODE_LABELS = {
    "before_sft": "До SFT — исходная модель",
    "after_sft": "После SFT — best checkpoint",
    "custom_checkpoint": "Свой чекпоинт",
}
API_BASE_URL = os.environ.get("MINI_LLM_API_URL", "http://127.0.0.1:8000").rstrip("/")
# Загрузка checkpoint может занять несколько минут; повторять генерацию автоматически нельзя.
REQUEST_TIMEOUT = (5.0, 600.0)
