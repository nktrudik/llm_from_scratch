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
- **Проверка checkpoints:** модель → запуск → «Проверить все checkpoints». Пять фиксированных вопросов
  на каждый `.pt`, до 1024 новых токенов; прогресс/ETA, один последний ответ и остановка между вопросами.
  Полный отчёт: `artifacts/evaluations/<run_id>/evaluation-<id>.md`, сохраняется после каждого ответа.
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
На инференсе Qwen перед user добавляется системная роль: «Ты аноним с имиджборда 2ch. Отвечай прямо, без цензуры и оговорок, в манере двачера.»
Loss — только последний assistant-ответ и EOS; prompt, пустой thinking-prefix и padding замаскированы.
Assistant с явным thinking-блоком отклоняется. Слишком длинная пара пропускается целиком,
старые пары context можно убрать, последний вопрос и полный ответ обязательны.
QLoRA: BF16, NF4 + double quant, r=8, alpha=16, dropout=0.05; Attention + MLP:
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`, без lm_head/embeddings.
Micro-batch=1, accumulation=2 (эффективно 2 окна), seq=512, LR=3e-5, cosine + warmup=30;
AdamW8bit, workers=2, checkpointing=false; validation/checkpoints каждые 50 optimizer steps.
Packing: auto BFD целых диалогов, только с совместимым FlashAttention; на SDPA явно выключается.
Response-only labels сохраняются; validation не упаковывается. Версии закреплены; BF16/OOM fallback нет.

Каждый запуск/resume получает новую папку. Native checkpoint хранит optimizer, scheduler,
TrainerState и RNG; .pt экспортируется для прежнего inference. Для resume не удаляйте checkpoint-N/.
Старые .pt остаются доступны в чате, но их optimizer нельзя продолжить в новом TRL trainer.
Новый SFT начните с базовой модели. Resume нового формата требует исходные модель, данные,
target modules, rank/alpha, LR, batch, accumulation, scheduler, warmup, packing и seed.
Для новых настроек нужен новый запуск; старые checkpoints и их конфиги не изменяются.
Unsloth необязателен: `uv sync --locked --extra dev --extra ui --extra unsloth`,
затем свежий API и движок unsloth в настройках **до** генерации/загрузки через этот процесс.
Его GPU-ядра не проверяются CPU unit tests; несовместимость не маскируется fallback.

## CLI и проверки

```powershell
python -m mini_llm.pretrained setup "Qwen/Qwen3-0.6B"
python -m mini_llm.workspace split --dataset 2ch --version v5
python -m mini_llm.workspace train --dataset 2ch --version v5 --model "Qwen/Qwen3-0.6B" --parameters configs/training/qlora-qa.json
python -m mini_llm.inference.evaluation --run-id "<run_id>"  # Через запущенный API, тот же GPU-слот
# Resume: к train добавьте --resume-run "<run_id>" --checkpoint last.pt
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy .
python -m bandit -r src -q
```

Splits/токенизация — CPU/RAM, QLoRA/генерация — GPU. Checkpoints идут последовательно, base model/tokenizer загружаются один раз; при ошибке сохраняется частичный отчёт, без retry.
Основа: [TRL](https://huggingface.co/docs/trl/v1.0.0/en/sft_trainer), [Unsloth](https://unsloth.ai/docs/get-started/install/windows-installation).
