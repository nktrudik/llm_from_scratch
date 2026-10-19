"""Настройки HTTP-клиента и доступные backend интерфейса."""

import os
from typing import Literal

ModelBackend = Literal["custom", "pretrained"]
PretrainedMode = Literal["before_sft", "after_sft", "custom_checkpoint"]
API_BASE_URL = os.environ.get("MINI_LLM_API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
# Каталог должен возвращаться быстро; зависший старый backend не ждём несколько минут.
OPTIONS_TIMEOUT = (5.0, 15.0)
# Загрузка checkpoint может занять несколько минут; повторять генерацию автоматически нельзя.
REQUEST_TIMEOUT = (5.0, 600.0)
