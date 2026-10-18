# Mini LLM / НейроДвач

Локальный учебный проект: собственный Transformer и fine-tuning Hugging Face моделей
(full / LoRA / QLoRA), FastAPI и Streamlit. Интерфейс: **Чат** и **Обучение**.
Response-only loss, non-thinking chat template, checkpoints/resume, AMP и GPU telemetry.
Custom: 4 блока, d_model=256, 4 heads, FFN=1024, vocab=8192, tied embeddings, context=1024.

## Установка и запуск (PowerShell, из корня проекта)

```powershell
uv sync --extra dev --extra pretrained --extra ui
# Терминал 1: один процесс API, без --reload во время обучения.
.\.venv\Scripts\python.exe -m uvicorn mini_llm.main:app --host 127.0.0.1 --port 8000
# Терминал 2:
.\.venv\Scripts\python.exe -m streamlit run src/mini_llm/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

UI: http://127.0.0.1:8501 · Swagger: http://127.0.0.1:8000/docs.
Другой API: задайте `$env:MINI_LLM_API_URL="http://127.0.0.1:8000"` перед запуском UI.
В чате выберите модель → версию/запуск → checkpoint. До SFT используется базовая модель.
Во вкладке обучения выберите датасет, модель, параметры и нажмите «Запустить обучение».
Новая папка `data/datasets/<id>/vN/samples.jsonl` обнаруживается при открытии страницы
или по кнопке «Обновить список датасетов». Backend сохраняет количество samples и SHA256
в `dataset.json`; JSONL и существующие splits не меняются. Фонового обновления каталога нет.
Открытие страницы не запускает обучение. Статус — в UI, подробный progress — в терминале API.
После старта UI обновляет шаги, эпоху, loss, график и приблизительный ETA каждые 2 секунды.
Метрики пишутся в `progress.json` отдельно от checkpoints; ETA уточняется после первых шагов.
Одновременно выполняется только одна тяжёлая задача; генерация на это время блокируется.
Закрытие браузера не останавливает обучение. Не останавливайте API до сохранения checkpoint.

## Данные и версии моделей

```text
data/sources/2ch/{threads,sft-pairs}/          исходники, не изменяются
data/staging/2ch/                            результат preprocessing до публикации
data/datasets/2ch/vN/                        dataset.json, samples.jsonl, splits/, reports/
checkpoints/runs/<model>/2ch/vN/run-NNN/      run.json, pretrained.json, *.pt
configs/models/<model>/<mode>.json          профиль base model
configs/training/qlora-qa.json               стартовые параметры обучения
artifacts/models/                           регистрации скачанных моделей
artifacts/tokenizers/custom-bpe/v1/          сохранённый собственный tokenizer
.cache/huggingface/                          неизменённый HF cache
data/archive/, checkpoints/archive/         исторические данные и остатки
```

Версии 2ch: **v1** — исходные SFT-пары (3638), **v2** — QA Full (3638),
**v3** — QA Candidates (1069; train/validation/test: 1036/19/14).
Неизвестное происхождение старых моделей обозначено `legacy/v0`.
Каждый запуск, включая resume, получает новый `run-NNN`: прежние веса не перезаписываются.
`run.json` связывает модель, данные, параметры, родительский запуск и SHA256 splits.
Миграция старой структуры выполняется один раз: `python -m mini_llm.workspace migrate`.
Она переносит файлы без изменения JSONL/.pt; журнал: `data/archive/layout-migration.json`.
Исторические пути внутри .pt оставлены нетронутыми; старые команды с путями нужно обновить.

## Подготовка и ручной запуск

```powershell
# Setup скачивает модель и tokenizer; обучение не запускается.
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen3-0.6B"
# SFT preprocessing использует только *_sft_raw.json и сохраняет staging.
.\.venv\Scripts\python.exe -m mini_llm.data.preprocessing --source-format sft
# Публикация копирует JSONL без изменения текста; v4 — пример нового корпуса.
.\.venv\Scripts\python.exe -m mini_llm.workspace publish data/staging/2ch/2ch_dialogues.jsonl --dataset 2ch --version v4 --title "Новые QA-пары"
.\.venv\Scripts\python.exe -m mini_llm.workspace split --dataset 2ch --version v4
.\.venv\Scripts\python.exe -m mini_llm.workspace list
# Альтернатива кнопке UI: обучение v3, всегда в новом каталоге.
.\.venv\Scripts\python.exe -m mini_llm.workspace train --dataset 2ch --version v3 --model "Qwen/Qwen3-0.6B" --parameters configs/training/qlora-qa.json
```

Resume: добавьте `--resume-run "<run_id из workspace list>" --checkpoint last.pt`;
увеличьте общий epochs/max_steps в JSON параметров. Модель, версия, batch и seed сохраняются.
Split — детерминированный 90/5/5 по тредам; существующие версии/splits не заменяются.
QLoRA: BF16, NF4+double quant, r=16, alpha=32, dropout=0.05, all-linear;
пример: batch=2, seq=512, LR=3e-5, AdamW8bit, workers=2, без gradient checkpointing.
Скрытого fallback при BF16/OOM нет. QLoRA требует совместимую CUDA GPU.
Preprocessing/split/BPE используют CPU/RAM; обучение и генерация по умолчанию — CUDA.
Scraper, BPE и token statistics доступны через Swagger; автоматически не запускаются.
Код в `src/mini_llm/`: data, modeling, pretrained, tokenization, training,
inference, workspace, api, ui, observability. Нет distributed training и dataset packing.
Обычные CLI/API training тоже создают новый run; явный checkpoint-dir оставлен для совместимости.

## Проверки (без скачивания весов и обучения корпуса)

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy .
.\.venv\Scripts\python.exe -m bandit -r src -q
```
