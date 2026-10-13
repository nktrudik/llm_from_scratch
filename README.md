# Mini LLM

Проект для создания YouTube-видео и практических экспериментов с Large Language Models. Содержит полный локальный пайплайн: от парсинга данных (2ch) и препроцессинга до обучения собственной модели с нуля на PyTorch и файнтюнинга готовых open-source моделей (LoRA/QLoRA). 

Управление этапами происходит через локальный HTTP API (FastAPI), а тестирование ответов — через веб-интерфейс на Streamlit.

## 🚀 Быстрый старт (Установка и Запуск)

Проект использует `uv` для управления зависимостями. Требуется Python 3.13+.

**1. Установка всех зависимостей (включая UI и Hugging Face):**
```powershell
uv sync --extra dev --extra pretrained --extra ui
```

**2. Запуск локального API-сервера:**
```powershell
.\.venv\Scripts\uvicorn.exe mini_llm.main:app --host 127.0.0.1 --port 8000
```
*Swagger UI с документацией всех эндпоинтов доступен по адресу: http://127.0.0.1:8000/docs*

**3. Запуск веб-интерфейса (чат):**
Откройте новый терминал и выполните:
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/mini_llm/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Чат доступен по адресу http://127.0.0.1:8501. Можно выбрать `Custom` или `Qwen`.
Для Qwen появляется второй выпадающий список с зарегистрированными model ID,
например `Qwen/Qwen2.5-0.5B-Instruct` и `Qwen/Qwen3-0.6B` после их setup.
Для Qwen доступны режимы «До SFT — исходная модель», «После SFT — best checkpoint»
(только при наличии `best.pt`) и «Свой чекпоинт». В последнем укажите имя файла,
например `step_00000500.pt`, `last.pt` или `best.pt`, без пути. API ищет его в
`checkpoint_dir` регистрации выбранной модели; новый каталог отдельного training-запуска
автоматически не выбирается. Выбор в UI не меняет `artifacts/pretrained/active.json`.
Без выбора model ID API по-прежнему использует активную модель. Параметры генерации и пути
берутся из существующих defaults API. История диалога не сохраняется.

---

## 🛠 Пайплайн данных

Все долгие операции выполняются асинхронно через API или CLI.

**1. Сбор данных (Scraper):**
```powershell
# Пример сбора новых тредов с доски
$body = @{ url = "https://2ch.org/b/"; max_threads = 10 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/scraper" -ContentType "application/json" -Body $body
```

**2. Препроцессинг и очистка датасета:**
```powershell
.\.venv\Scripts\python.exe -m mini_llm.data.preprocessing --source-format sft --input-dir data/raw/2ch_clean --output-dir data/processed --review-size 100 --seed 42
```

**3. Разбиение датасета (Train / Validation / Test):**
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/dataset/split" -ContentType "application/json" -Body "{}"
```

**4. Обучение собственного BPE токенизатора:**
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/tokenizer/train" -ContentType "application/json" -Body "{}"
```

---

## 🧠 Обучение моделей (Train & Fine-Tuning)

### Вариант А: Дообучение готовой модели (QLoRA)
Загрузка весов с Hugging Face, подготовка конфигов и запуск обучения одной командой.

```powershell
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen2.5-0.5B-Instruct" --train --dtype bfloat16 --batch-size 2 --epochs 3 --learning-rate 0.0002
```
*Доступные режимы адаптации (`--mode`): `qlora` (по умолчанию), `lora`, `full`.*

### Qwen3-0.6B: отдельная загрузка и SFT

Остановите API/UI перед синхронизацией окружения и освободите GPU перед обучением.
Все команды ниже выполняются вручную из корня проекта.

```powershell
uv sync --extra dev --extra pretrained --extra ui
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen3-0.6B" --mode qlora --dtype bfloat16 --max-sequence-length 512 --no-gradient-checkpointing
```

Setup скачивает модель и её tokenizer, закрепляет фактический commit и создаёт:

- `.cache/huggingface/models--Qwen--Qwen3-0.6B/` — cache Hugging Face;
- `configs/pretrained/model--qwen--qwen3-0.6b-qlora.json` — конфиг;
- `artifacts/pretrained/model--qwen--qwen3-0.6b/qlora.json` — регистрация;
- `checkpoints/pretrained/model--qwen--qwen3-0.6b/qlora/` — каталог адаптеров;
- `artifacts/pretrained/active.json` — указатель на последнюю подготовленную модель.

Файлы Qwen2.5 остаются в своих каталогах. Setup не запускает обучение и не загружает
веса в GPU. После setup обе модели доступны в списке UI; best проверяется отдельно.
Для одного model ID используется активный режим адаптации, если модель активна,
иначе приоритет регистрации — QLoRA, LoRA, full.

Это **SFT/fine-tuning**, а не pretraining с нуля. Запуск на существующих split JSONL:

```powershell
$checkpointDir = "checkpoints/pretrained/model--qwen--qwen3-0.6b/qlora"
if (Get-ChildItem -LiteralPath $checkpointDir -Filter "*.pt" -ErrorAction SilentlyContinue) {
    throw "В каталоге уже есть checkpoints. Используйте resume или отдельный каталог запуска."
}
.\.venv\Scripts\python.exe -m mini_llm.training --backend pretrained --pretrained-config configs/pretrained/model--qwen--qwen3-0.6b-qlora.json --splits-dir data/processed/splits --checkpoint-dir $checkpointDir --device cuda --batch-size 2 --epochs 3 --learning-rate 0.0002 --weight-decay 0.01 --gradient-clip-norm 1.0 --max-train-samples 30000 --max-validation-samples 1000 --num-workers 2 --log-interval 100 --validation-interval 500 --validation-batches 50 --checkpoint-interval 500 --seed 42
```

Профиль сохраняет BF16, NF4 + double quantization, AdamW8bit, окно 512,
LoRA `r=16`, `alpha=32`, `dropout=0.05`, `all-linear`, gradient checkpointing выключен.
На RTX 3050 4 GB вместимость не гарантируется; скрытого fallback при OOM нет.
Сохраняются `best.pt`, `last.pt` и periodic checkpoints только в каталоге Qwen3.
Родной chat template не меняется; Qwen3 может выдавать блоки `<think>`.

### Вариант Б: Обучение кастомной модели с нуля
Запускается через API (предварительно убедитесь, что датасет разбит и токенизатор обучен):

```powershell
$body = @{ device = "cuda"; batch_size = 4; epochs = 3; learning_rate = 0.0003 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/training" -ContentType "application/json" -Body $body
```

---

## 💬 Инференс (Генерация)

Для тестирования генерации используйте UI (`http://127.0.0.1:8501`) или обращайтесь напрямую к API:

```powershell
$body = @{
    model_backend = "pretrained"
    pretrained_model_id = "Qwen/Qwen3-0.6B"
    pretrained_mode = "after_sft"
    prompt = "Привет! Объясни простыми словами, что такое LLM."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/generate" -ContentType "application/json" -Body $body
```

`GET /v1/generate/options` возвращает доступность активной модели и массив `models`
с доступностью каждой зарегистрированной модели, без загрузки весов. В запросе
генерации `pretrained_model_id` выбирает эту регистрацию без изменения active.json;
его нельзя совмещать с явными путями к конфигу/checkpoint. Прежние запросы без model ID
и прямые запросы с путями продолжают работать.

---

## 🧪 Качество кода

Быстрые проверки (линтеры, форматирование, тесты):
```powershell
.\.venv\Scripts\pytest.exe
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe .
.\.venv\Scripts\bandit.exe -r src -q
```
