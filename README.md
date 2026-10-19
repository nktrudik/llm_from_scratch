# НейроДвач / Mini LLM

Локальный чат, SFT Hugging Face моделей и учебный Transformer. FastAPI + Streamlit.
Pretrained SFT: **TRL SFTTrainer + datasets + PEFT**, опционально Unsloth.
Custom сохраняет свой trainer: 4 блока, d_model=256, 4 heads, FFN=1024, vocab=8192,
tied embeddings, context=1024. Обучение и скачивание запускаются только явно.

## Запуск (PowerShell из корня проекта)

Остановите API/UI перед обновлением окружения. Python 3.13, NVIDIA CUDA GPU для QLoRA.

```powershell
uv sync --locked --extra dev --extra pretrained --extra ui
# Терминал 1: один процесс, без --reload во время обучения.
.\.venv\Scripts\python.exe -m uvicorn mini_llm.main:app --host 127.0.0.1 --port 8000
# Терминал 2:
.\.venv\Scripts\python.exe -m streamlit run src/mini_llm/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

UI: http://127.0.0.1:8501 · API: http://127.0.0.1:8000/docs.
Другой API: `$env:MINI_LLM_API_URL="http://127.0.0.1:8000"` перед запуском UI.

- **Чат:** модель → версия (базовая или запуск) → checkpoint; один вопрос/ответ без памяти.
- **Обучение:** версия датасета → модель → новый запуск/resume → параметры.
  Редкие параметры скрыты в «Настройки». Видны шаги, loss, график и примерный ETA.
  «Остановить обучение» завершает текущий шаг и сохраняет native checkpoint и last.pt.
  При остановке до первого шага checkpoint может отсутствовать. Закрытие UI не останавливает API.
- **Модели:** загрузка по HF model ID; удаление только целого запуска либо всей модели
  вместе с её запусками и базовым cache. Сначала план, затем подтверждение «УДАЛИТЬ».
  Датасеты, код и общий BPE не удаляются. Не меняйте файлы внешним CLI во время очистки.

## Данные и хранение

```text
data/datasets/<id>/vN/                       samples.jsonl, dataset.json, splits/, reports/
checkpoints/runs/<model>/<id>/vN/run-NNN/    run.json, pretrained.json, progress.json
                                           checkpoint-N/, step_NNNNNNNN.pt, best.pt, last.pt
configs/models/<model>/                     закреплённая revision и профиль модели
configs/training/qlora-qa.json               параметры нового SFT (r=8, alpha=16)
artifacts/models/, .cache/huggingface/       реестр и скачанные базовые модели/tokenizer
artifacts/tokenizers/custom-bpe/v1/          собственный BPE для Custom
data/sources/, data/staging/, data/archive/ исходники, подготовка, исторические данные
```

Новый корпус: `data/datasets/<id>/vN/samples.jsonl`, по одной ChatML-записи на строку:

```json
{"messages":[{"role":"system","content":"Ты помощник."},{"role":"user","content":"Вопрос"},{"role":"assistant","content":"Ответ"}],"metadata":{"board":"b","thread_id":123}}
```

System необязателен; дальше user/assistant чередуются, последнее сообщение — assistant.
Текст не изменяется. Старый context/response также читается. Укажите metadata.thread_id,
чтобы сохранить thread-level grouping; без него одинаковые разговоры получают общую группу.
Новая версия появляется при открытии страницы обучения или по «Обновить список датасетов».
Splits создаются явно (90/5/5 по группам); существующие JSONL/splits не перезаписываются.

## SFT и совместимость

Qwen использует один native template с `enable_thinking=False` для train/validation/inference.
Loss — только последний assistant-ответ и EOS; prompt, пустой thinking-prefix и padding замаскированы.
Assistant с явным thinking-блоком отклоняется. Слишком длинная пара пропускается целиком,
старые пары context можно убрать, последний вопрос и полный ответ обязательны.
Профиль: BF16, NF4 + double quant, r=8, alpha=16, dropout=0.05, all-linear;
batch=2, seq=512, LR=3e-5, AdamW8bit, workers=2, gradient checkpointing=false.
Нет скрытого BF16/OOM fallback, packing или изменения исходного корпуса.
Версии библиотек закреплены в pyproject.toml и uv.lock; постоянный LR, без warmup.

Каждый запуск/resume получает новую папку. Native checkpoint хранит optimizer, scheduler,
TrainerState и RNG; .pt экспортируется для прежнего inference. Для resume не удаляйте checkpoint-N/.
Старые .pt остаются доступны в чате, но их optimizer нельзя продолжить в новом TRL trainer.
Новый SFT начните с базовой модели. Resume нового формата требует исходные модель, данные,
rank/alpha, LR, batch и seed; увеличьте общий epochs/max_steps, если план уже достигнут.
Unsloth необязателен: `uv sync --locked --extra dev --extra ui --extra unsloth`,
затем свежий API и движок unsloth в настройках **до** генерации/загрузки через этот процесс.
Его GPU-ядра не проверяются CPU unit tests; несовместимость не маскируется fallback.

## CLI и проверки

```powershell
python -m mini_llm.pretrained setup "Qwen/Qwen3-0.6B"
python -m mini_llm.workspace split --dataset 2ch --version v5
python -m mini_llm.workspace train --dataset 2ch --version v5 --model "Qwen/Qwen3-0.6B" --parameters configs/training/qlora-qa.json
# Resume: к train добавьте --resume-run "<run_id>" --checkpoint last.pt
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy .
python -m bandit -r src -q
```

Splits/токенизация используют CPU/RAM; QLoRA и обычная генерация — GPU.
Scraper, preprocessing, собственный BPE и statistics доступны через существующие CLI/Swagger.
Основа: [статья](https://habr.com/ru/companies/otus/articles/1026700/),
[TRL](https://huggingface.co/docs/trl/v1.0.0/en/sft_trainer),
[Unsloth](https://unsloth.ai/docs/get-started/install/windows-installation).
