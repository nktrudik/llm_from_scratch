# Mini LLM с нуля

Учебный проект компактной decoder-only Transformer-модели на PyTorch и последовательного
pipeline подготовки текстовых диалогов из тредов 2ch. Репозиторий содержит сборщик JSON API,
preprocessing графа ответов, разбиение датасета по тредам, обучение собственного byte-level BPE
tokenizer, расчёт token statistics, универсальное обучение custom и pretrained causal LM,
LoRA/QLoRA, локальный FastAPI для ручного запуска этапов и минимальный чат на Streamlit.

В проекте пока нет distributed training, scheduler и production model evaluation.
Код обучения и HTTP-генерации реализован; качество ответов зависит от используемого checkpoint.

## Установка и VS Code

Выберите `.venv` как Python interpreter в VS Code и синхронизируйте зависимости:

```powershell
uv sync --extra dev
```

Для Hugging Face Transformers, PEFT и bitsandbytes нужна отдельная optional-группа:

```powershell
uv sync --extra dev --extra pretrained
```

Базовый from-scratch pipeline от неё не зависит.

Без `uv` можно использовать:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Проект рассчитан на Python 3.13 или новее. Все команды ниже выполняются из корня репозитория.

## Архитектура модели

`DecoderOnlyTransformer` — causal decoder-only Transformer со следующей конфигурацией по
умолчанию:

- 4 Transformer-блока;
- `d_model = 256`;
- 4 attention heads;
- FFN размером 1024;
- максимальный контекст 1024 token IDs;
- vocabulary size 8192;
- pre-LayerNorm, residual connections и настраиваемый dropout;
- общие веса token embedding и выходной проекции;
- 5 518 848 обучаемых параметров.

Special token IDs зафиксированы в `ModelConfig`:

| Token | ID |
|---|---:|
| `<PAD>` | 0 |
| `<UNK>` | 1 |
| `<BOS>` | 2 |
| `<EOS>` | 3 |
| `<USER>` | 4 |
| `<ASSISTANT>` | 5 |

Создание модели:

```python
from mini_llm.modeling import DecoderOnlyTransformer, ModelConfig

config = ModelConfig()
model = DecoderOnlyTransformer(config)
```

Модель принимает `input_ids` формы `[batch, sequence_length]`. Без `targets` она возвращает
logits, а с `targets` — `(logits, loss)`. Метод `generate()` работает с token IDs, а не с текстом.

## Структура проекта

```text
src/mini_llm/
├── main.py                  # единственная ASGI-точка входа
├── __init__.py              # обязательный маркер Python-пакета
├── api/                     # HTTP routes, Pydantic-схемы, сервисы и jobs
├── modeling/                # общий model interface и собственный Transformer
├── pretrained/              # setup/download, registry, Hugging Face, full/LoRA/QLoRA
├── tokenization/            # BPE tokenizer и его конфигурация
├── data/
│   ├── config.py            # общие ограничения Dataset/DataLoader
│   ├── dialogue.py          # схема dialogue JSONL и role tokens
│   ├── dataset.py           # PyTorch Dataset, padding и DataLoader
│   ├── splitting.py         # train/validation/test по тредам
│   ├── statistics.py        # raw/effective token statistics
│   ├── deduplication.py     # exact и near-duplicate detection
│   ├── preprocessing/       # очистка, reply graph, схемы и pipeline
│   └── scraping/            # JSON API 2ch, parsing и storage
├── training/                # универсальный trainer, monitoring, checkpoints и overfit
├── inference/               # config, схемы и генерация по checkpoint
└── ui/                      # Streamlit-экран, HTTP-клиент и настройки подключения

data/raw/2ch/<board>/         # неизменяемые raw JSON тредов
data/processed/               # dialogue dataset, split и отчёты
artifacts/tokenizer/          # обученный tokenizer JSON
artifacts/pretrained/         # регистрации скачанных моделей и active.json
configs/pretrained/           # конфиги по model ID и режиму адаптации
checkpoints/pretrained/       # отдельные SFT checkpoints каждой модели/режима
.cache/huggingface/           # Hub snapshots, tokenizer и исходные веса
tests/                        # быстрые unit tests
```

Корень пакета не содержит бизнес-логики: кроме обязательного `__init__.py`, там находится только
`main.py`. Предметные области изолированы в пакетах, а запуск долгих операций сосредоточен в API.
Ни один training pipeline не стартует при импорте модуля.

## Локальный FastAPI

Запуск из корня проекта, без `--workers` и без `--reload`:

```powershell
.\.venv\Scripts\uvicorn.exe mini_llm.main:app --host 127.0.0.1 --port 8000
```

Swagger UI доступен на `http://127.0.0.1:8000/docs`. Тяжёлые операции возвращают `job_id` и
выполняются последовательно в фоне одного процесса. Проверить задачу:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/v1/jobs/JOB_ID"
```

Это локальный учебный диспетчер, а не production-очередь: его jobs хранятся в RAM и теряются при
перезапуске API. Checkpoints и результаты pipeline сохраняются на диск обычными модулями проекта.

## Минимальный чат (Streamlit)

Один экран: выбор `Custom` (`custom`) или `Qwen` (`pretrained`),
поле сообщения и ответ. Сохраняется только отображение текущей пары: следующий запрос заменяет
предыдущую пару. Истории, памяти диалога и списка чатов нет. Только для Qwen есть выбор:
«До SFT — исходная модель» или «После SFT — best checkpoint».
Второй вариант отсутствует, пока у активной модели нет непустого `best.pt`; наличие periodic
checkpoint или `last.pt` его не включает. После обучения обновите страницу.
UI проверяет доступность через `GET /v1/generate/options`, не читает файлы модели самостоятельно.
До выполнения setup ввод для Qwen заблокирован, но Custom остаётся доступен.

`Qwen` — название выбора pretrained backend в интерфейсе; фактический Hugging Face model ID
показывается под ним и определяется последней успешной командой setup, а не жёстко задан в UI.

Установить UI и зависимости обоих backend (обучение и загрузка весов не запускаются):

```powershell
uv sync --extra dev --extra pretrained --extra ui
```

Если нужен только custom backend, можно не указывать `--extra pretrained`. Без `uv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,pretrained,ui]"
```

В первом терминале запустить API из корня проекта:

```powershell
.\.venv\Scripts\uvicorn.exe mini_llm.main:app --host 127.0.0.1 --port 8000
```

Во втором терминале из того же каталога запустить интерфейс:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/mini_llm/ui/app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Открыть `http://127.0.0.1:8501`. Streamlit работает на CPU, GPU ему не нужен. Генерацию выполняет
API на CUDA по существующим defaults. Для Custom нужны checkpoint и BPE tokenizer;
для Qwen сначала выполните setup из раздела pretrained ниже. До SFT checkpoint не требуется.

UI отправляет в `POST /v1/generate` только `prompt`, `model_backend` и, для Qwen, `pretrained_mode`.
Сервер подставляет пути:

| Backend | Checkpoint | Tokenizer / конфиг |
|---|---|---|
| `custom` | `checkpoints/training/best.pt` | `artifacts/tokenizer/2ch_bpe.json` |
| `pretrained`, до SFT | не используется | конфиг из `artifacts/pretrained/active.json` |
| `pretrained`, после SFT | `best.pt` из каталога активной модели | тот же конфиг |

Pretrained tokenizer выбирается из `model_id/revision/cache_dir` в конфиге, а не из BPE-файла.
Загрузка до SFT не создаёт случайный LoRA-адаптер: используются исходные instruct-веса в
заданной dtype. После SFT применяется сохранённый full/LoRA/QLoRA checkpoint с проверкой metadata.
Defaults HTTP-генерации находятся в
`GenerationRequest`: `device="cuda"`, `max_new_tokens=256`, `temperature=0.3`, `top_k=20`.
Для прямых API-запросов по-прежнему можно явно передавать другие пути и параметры.

Во время ожидания UI показывает индикатор; ошибки соединения, занятого API или загрузки
checkpoint выводятся на экране. Запросы автоматически не повторяются. Первая загрузка Qwen
может занять несколько минут; UI ожидает до 10 минут. Затем API держит подготовленную pretrained
модель и tokenizer в памяти и повторно использует их без чтения checkpoint и подготовки адаптера.
Это кэш ресурсов, а не память диалога: каждый prompt независим. В памяти хранится одна модель;
переключение до/после SFT, изменение config/checkpoint либо device вызывает перезагрузку.
После перезапуска API модель загружается заново; Hugging Face cache на диске при этом сохраняется.
Перед запуском обучения/подготовки pretrained через API и при выборе custom кэш очищается,
чтобы освободить RAM/VRAM. В терминале API видны загрузка модели и попадание в кэш.
Перед обучением через CLI остановите API: это другой процесс, его кэш занимает VRAM независимо
от процесса trainer. API и UI можно снова запустить после обучения.
Для другого адреса API задайте `$env:MINI_LLM_API_URL = "http://127.0.0.1:8000"` до запуска UI.

## Сбор тредов 2ch

Scraper использует JSON API, не скачивает вложения и не пытается обходить Cloudflare, CAPTCHA,
авторизацию или anti-bot защиту. В HTTP-клиенте есть один `requests.Session`, User-Agent, timeout,
последовательные запросы, случайная задержка, ограниченные retries, exponential backoff и обработка
`Retry-After` для HTTP 429.

Один тред:

```powershell
$body = @{ url = "https://2ch.org/b/res/THREAD_ID.html" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/scraper" -ContentType "application/json" -Body $body
```

Актуальные треды доски, не более десяти новых:

```powershell
$body = @{ url = "https://2ch.org/b/"; max_threads = 10 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/scraper" -ContentType "application/json" -Body $body
```

Каждый тред сохраняется в `data/raw/2ch/<board>/<thread_id>.json`. Существующий raw-файл никогда
не перезаписывается, а `max_threads` учитывает только новые треды.

## 1. Preprocessing

```powershell
$body = @{} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/preprocessing" -ContentType "application/json" -Body $body
```

Этап читает `data/raw/2ch/<board>/*.json`, восстанавливает связи только по явным валидным
`>>post_id`, строит reply chains и multi-reference samples, очищает markup и технические
идентификаторы, удаляет PII и консервативно удаляет дубли. Сообщения без достоверного parent
отбрасываются: автоматического `thread_root_fallback` больше нет.

Создаются:

- `data/processed/2ch_dialogues.jsonl` — context/response samples;
- `data/processed/2ch_preprocessing_stats.json` — причины фильтрации, длины, глубина цепочек,
  `messages_dropped_without_reliable_parent` и `thread_root_fallback_samples_prevented`;
- `data/processed/2ch_review_sample.jsonl` — детерминированная выборка для ручной проверки.

Raw-файлы не меняются. Длинные samples не удаляются и не обрезаются до 1024 tokens. Имеющийся
processed dataset, созданный старой версией, нужно пересоздать этой командой, иначе в нём останутся
samples с `thread_root_fallback`.

Ресурсы: CPU и RAM; GPU не используется и не требуется. Время зависит от числа raw-тредов и
объёма дедупликации.

## 2. Train / validation / test split

После завершения preprocessing:

```powershell
$body = @{} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/dataset/split" -ContentType "application/json" -Body $body
```

Этап дважды потоково читает `data/processed/2ch_dialogues.jsonl`, детерминированно перемешивает
уникальные `(board, thread_id)` с seed 42 и распределяет треды в пропорции 90% / 5% / 5%. Все
samples одного треда всегда находятся только в одном split.

Создаются:

- `data/processed/splits/train.jsonl`;
- `data/processed/splits/validation.jsonl`;
- `data/processed/splits/test.jsonl`;
- `data/processed/splits/split_stats.json` — threads и samples по split и board.

Ресурсы: CPU, последовательное чтение диска и небольшая RAM для множества thread IDs; GPU не
используется и не требуется.

## 3. Обучение BPE tokenizer

Tokenizer обучается строго на `train.jsonl`:

```powershell
$body = @{} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/tokenizer/train" -ContentType "application/json" -Body $body
```

Этап читает `data/processed/splits/train.jsonl`, обучает byte-level BPE с vocabulary size 8192 и
сохраняет `artifacts/tokenizer/2ch_bpe.json`. Validation и test при обучении не читаются. Byte-level
alphabet обеспечивает представление русского и английского текста, цифр, пунктуации, сленга,
эмодзи и смешанных строк без нормализации исходного стиля.

Повторно обучать tokenizer для использования не нужно:

```python
from pathlib import Path

from mini_llm.tokenization import BPETokenizer

tokenizer = BPETokenizer.load(Path("artifacts/tokenizer/2ch_bpe.json"))
token_ids = tokenizer.encode("Привет, world! 😎")
text = tokenizer.decode(token_ids)
```

Единый формат model sample:

```text
<BOS><USER>последнее сообщение context<ASSISTANT>response<EOS>
```

Для длинной цепочки более ранние context messages получают чередующиеся роли так, чтобы последнее
сообщение перед response всегда имело `<USER>`. `encode_dialogue_window(..., max_length=1024)`
сохраняет response целиком и заполняет остаток окна последними context messages. Если полный
response сам не помещается, метод возвращает ошибку вместо скрытого обрезания.

Ресурсы: CPU и RAM; GPU не используется и не требуется. Обучение BPE — самый ресурсоёмкий этап
подготовки после preprocessing, но выполняется Rust-реализацией библиотеки `tokenizers`.

## 4. Token statistics

После обучения tokenizer:

```powershell
$body = @{ max_sequence_length = 1024 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/statistics/tokens" -ContentType "application/json" -Body $body
```

Этап читает три файла из `data/processed/splits/` и готовый
`artifacts/tokenizer/2ch_bpe.json`, затем создаёт
`data/processed/token_statistics.json`.

Отчёт содержит:

- фактический vocabulary size;
- raw/full tokens до ограничения окна;
- effective tokens после формирования допустимых окон;
- raw и effective tokens отдельно для train, validation и test;
- `training_loss_tokens` и `effective_train_tokens_per_parameter`, рассчитанные только по
  response targets в train;
- characters per token;
- min, mean, median, p90, p95 и max raw/effective lengths;
- число полных и коротких effective windows;
- raw, effective и отброшенные context tokens, процент потерь и число сокращённых samples;
- отдельное распределение response lengths;
- unusable oversized responses и их распределение по board;
- количество и долю `<UNK>`;
- статистику по board.

`effective_tokens` описывает все token IDs пригодных окон после сокращения context до 1024, а
`effective_train_tokens` — только response и EOS, по которым реально считается training loss.
Validation/test не входят в `effective_train_tokens_per_parameter`. Oversized response не
обрезается, даёт ноль effective tokens и остаётся в исходном split JSONL.

Этап ничего не удаляет и не создаёт tokenized training dataset. Ресурсы: CPU и последовательное
чтение диска; RAM используется для массивов длин samples. GPU не используется и не требуется.

## Архитектура PyTorch data pipeline

```text
processed JSONL
    ↓
train / validation / test JSONL
    ↓
готовый BPE tokenizer
    ↓
DialogueDataset
    ├── пропуск oversized responses
    └── окно ≤ 1024 с приоритетом последних context messages
    ↓
shift: input_ids = sequence[:-1], targets = sequence[1:]
    ↓
collate + padding внутри batch
    ↓
DecoderOnlyTransformer(input_ids, targets)
```

`DialogueDataset` хранит в RAM только byte offsets пригодных JSONL-строк. Токенизация выполняется
при чтении sample, а весь датасет заранее в GPU memory не переносится. Responses, которые вместе с
обязательными `<BOS>`, `<ASSISTANT>` и `<EOS>` не помещаются в окно, безопасно пропускаются и
считаются в `oversized_response_count`.

`targets` сдвинуты на один token относительно `input_ids`. Collator дополняет обе последовательности
справа через `<PAD>`; padded targets игнорируются существующим `cross_entropy`, поскольку модель
использует `ignore_index=pad_token_id`. Train DataLoader включает deterministic shuffle,
validation/test создаются без shuffle, последний неполный batch сохраняется. Общий предел
`batch_size` задаётся в `data/config.py` и сейчас равен 8; для RTX 3050 4 GB в примерах обучения
используется более консервативное значение 4.

Быстрая проверка восьми пригодных train samples и одного CPU batch:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.data.dataset --split train --max-samples 8 --batch-size 2
```

Команда читает готовые `data/processed/splits/train.jsonl` и
`artifacts/tokenizer/2ch_bpe.json`, но не обучает модель. GPU не используется.

## Ручной overfit sanity-check

Отдельный модуль позволяет проверить, уменьшается ли loss при многократном обучении на первых 32
пригодных train samples. Это не production trainer: в нём нет checkpoint, scheduler, validation,
resume, distributed training или полного прохода по корпусу.

Ручной запуск на GPU:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training.overfit --device cuda --samples 16 --batch-size 1 --steps 50 --learning-rate 0.0003
```

Команда читает `train.jsonl` и готовый tokenizer, создаёт новую модель только в памяти и печатает
loss. Она использует CPU/RAM для Dataset и GPU/VRAM для модели и batch; автоматически не запускается
и ничего не сохраняет.

## Pretrained-модели, LoRA и QLoRA

Модуль `mini_llm.pretrained` загружает causal LM и tokenizer через Hugging Face model ID,
поддерживает закреплённую branch/tag/commit revision, отдельный cache и сохраняемую JSON-
конфигурацию. Логика Transformers/PEFT не смешана с собственной реализацией Transformer.

### Автоматическая загрузка по model ID

Текущая instruct-модель — [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
Для неё и других совместимых decoder-only моделей достаточно одной команды:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen2.5-0.5B-Instruct"
```

Команда использует [Hub snapshot download](https://huggingface.co/docs/huggingface_hub/guides/download):
разрешает `main` в конкретный commit, скачивает модель/tokenizer, предпочитает safetensors и не
скачивает дубли весов для ONNX, GGUF, TensorFlow или Flax. Проверяет конфиг и tokenizer без создания
модели в GPU, создаёт регистрацию и делает её активной для UI. Требуются сеть, место на диске,
CPU/RAM; GPU не нужен. Повторная загрузка использует дисковый cache.

Старые артефакты базовой модели удалены. Их checkpoints нельзя переносить на Instruct.
Название «До SFT» означает до **нашего** дообучения: исходная Instruct-модель уже обучена авторами.

Для указанного ID создаются:

```text
.cache/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<commit>/
configs/pretrained/model--qwen--qwen2.5-0.5b-instruct-qlora.json
artifacts/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora.json
artifacts/pretrained/active.json
checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora/
```

Конфиг сохраняется с закреплённым commit и `local_files_only=true`: training/inference используют
уже скачанные файлы. Конфиги/checkpoints разных model ID и adaptation mode изолированы.
Setup не перезаписывает конфиг, несовместимый с существующими checkpoints. Последняя успешная
регистрация определяет активную pretrained-модель; другие модели не удаляются.

Для другой модели замените аргумент на её `owner/model`. Можно указать `--revision <commit-or-tag>`,
`--mode full|lora|qlora`, `--dtype` и `--cache-dir`. По умолчанию: QLoRA, BF16, окно 512,
`gradient_checkpointing=false`; флаг `--gradient-checkpointing` включает его явно.
для full без явной dtype выбирается float32. Доступны causal LM, поддерживаемые установленным
`AutoModelForCausalLM`, и соответствующие tokenizer. GGUF, encoder-decoder и произвольные
multimodal-модели не поддерживаются. Для gated/private репозитория нужны права и `HF_TOKEN`;
удалённый Python-код автоматически не запускается (`trust_remote_code=false`).

Для instruct-моделей training и inference используют
[родной chat template](https://huggingface.co/docs/transformers/chat_templating).
Loss по-прежнему маскирует user/context и учитывает только assistant response и завершающие
маркеры. Старый context убирается первым; response не обрезается. Без chat template сохраняется
legacy role-формат. Если template не позволяет надёжно определить границу response, код выдаёт
понятную ошибку, а не обучается с неверной маской.

### Загрузка и SFT одной командой

Датасет уже подготовлен: нужны `data/processed/splits/train.jsonl` и `validation.jsonl`.
Pretrained использует свой tokenizer: обучать BPE и повторно выполнять preprocessing не нужно.
Перед CLI-обучением остановите API, чтобы освободить GPU от inference-кэша.

Стартовый профиль для RTX 3050 Laptop 4 GB — QLoRA и batch size 2:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen2.5-0.5B-Instruct" --train --dtype bfloat16 --max-sequence-length 512 --no-gradient-checkpointing --batch-size 2 --max-train-samples 30000 --num-workers 2 --log-interval 100 --validation-interval 500 --validation-batches 50 --checkpoint-interval 500 --epochs 3 --learning-rate 0.0002
```

Только флаг `--train` запускает существующий trainer после setup. Training читает split JSONL,
использует CPU/RAM для Dataset и CUDA/VRAM для обучения; создаёт periodic `step_XXXXXXXX.pt`,
`best.pt` и `last.pt` в каталоге модели. AMP, response-only loss, validation и resume
остаются в общем training pipeline. При отсутствии split команда завершается **до** скачивания.
VRAM зависит от модели и длины samples: QLoRA не гарантирует, что произвольная модель войдёт в 4 GB.

Профиль `PretrainedTrainingConfig` используется CLI и HTTP API для pretrained backend:
batch size 2, `num_workers=2`, максимум 30 000 пригодных train samples, log interval 100,
validation interval 500, до 50 validation batches, checkpoint interval 500. Явные параметры
имеют приоритет. Custom backend сохраняет прежние defaults и обычный `torch.optim.AdamW`.

QLoRA использует [bitsandbytes AdamW8bit](https://huggingface.co/docs/bitsandbytes/reference/optim/adamw),
а full/LoRA — обычный AdamW. LoRA: `r=16`, `alpha=32`, `dropout=0.05`, `all-linear`;
квантизация: NF4, double quantization включена. Compute dtype QLoRA и AMP — BF16.
GradScaler выключен для BF16 и сохраняется для FP16. Перед загрузкой весов проверяется
[аппаратная поддержка BF16](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
без эмуляции: неподдерживаемый GPU получает ошибку, не скрытый fallback. Для совместимости
явно выберите `--dtype float16` при setup либо `torch_dtype: "float16"` в отдельном конфиге.
`auto` при обучении явно разрешается в BF16; full с float32-весами использует BF16 AMP.

Лимит применяется только к первым пригодным samples train в порядке JSONL, перед shuffle.
Исходные JSONL не меняются; validation Dataset индексируется целиком, независимо от лимита
train. `validation_batches=50` ограничивает только число batches одного validation прохода.
`--max-train-samples 0` использует весь train; в программном API и HTTP вместо 0 задайте
`max_train_samples=None` / JSON `null`. Effective train tokens и план эпох рассчитываются
по фактически выбранному train Dataset. Если статистика всего корпуса не подходит, выполняется
сканирование только выбранных samples.

Перед загрузкой модели и обучением выводятся dtype, seq length, batch size,
gradient checkpointing, optimizer, max train samples и num workers. При CUDA OOM запуск
завершается с понятной ошибкой; batch size, окно и precision автоматически не меняются.
Фактическое размещение профиля в 4 GB следует проверить своим запуском: автоматических GPU
benchmark или обучения проект при изменении кода не запускает.

Если модель уже скачана, можно отдельно запустить trainer с дополнительными настройками:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training --backend pretrained --pretrained-config configs/pretrained/model--qwen--qwen2.5-0.5b-instruct-qlora.json --checkpoint-dir checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora --device cuda --batch-size 2 --epochs 3 --learning-rate 0.0002 --max-train-samples 30000 --num-workers 2 --log-interval 100 --validation-interval 500 --validation-batches 50 --checkpoint-interval 500
```

Для обычного LoRA и full fine-tuning:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen2.5-0.5B-Instruct" --mode lora --train --batch-size 1 --epochs 3 --learning-rate 0.0002
.\.venv\Scripts\python.exe -m mini_llm.pretrained setup "Qwen/Qwen2.5-0.5B-Instruct" --mode full --train --batch-size 1 --epochs 3 --learning-rate 0.00001
```

LoRA требует больше VRAM, чем QLoRA. Full fine-tuning 0.5B-модели с AdamW не рассчитан на 4 GB:
нужен GPU с большей памятью или меньшая модель. Checkpoints каждого режима сохраняются отдельно.

Продолжение QLoRA из `last.pt` до общего целевого числа шести эпох, без обращения к Hub:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training --backend pretrained --pretrained-config configs/pretrained/model--qwen--qwen2.5-0.5b-instruct-qlora.json --checkpoint-dir checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora --resume-from checkpoints/pretrained/model--qwen--qwen2.5-0.5b-instruct/qlora/last.pt --batch-size 2 --epochs 6 --learning-rate 0.0002 --max-train-samples 30000 --num-workers 2 --log-interval 100 --validation-interval 500 --validation-batches 50 --checkpoint-interval 500
```

Для resume из лучшего checkpoint замените `last.pt` на `best.pt`. Исходная команда `pretrained
prepare --model-id ... --output-config ...` тоже сохранена для ручного управления, но для
автоматической регистрации модели в UI используйте setup.

Revision, adaptation mode и PEFT-параметры входят в checkpoint metadata.
Resume отклоняется, если конфигурация модели отличается. Программный API предоставляет
`prepare_pretrained_model()` для training и `generate_pretrained()` для inference.

Для точного resume также должны совпадать batch size, optimizer и `max_train_samples`.
Новые checkpoints сохраняют тип optimizer и лимит train вместе с прежними состояниями.
Старые custom checkpoints без этих полей продолжают загружаться как AdamW / весь train.
Старый QLoRA checkpoint с FP16, окном 1024 и AdamW нельзя продолжить новым профилем:
для него сохраните исходный конфиг и окружение; новый профиль запускайте с нуля в отдельном
`--checkpoint-dir`, чтобы не перезаписать старые `best.pt` / `last.pt`. Сам setup также
не перезапишет несовместимый конфиг при наличии checkpoints. Файлы существующих checkpoints
не удаляются и не конвертируются автоматически.

## Полноценное обучение

`mini_llm.training` обучает модель на `train.jsonl` и периодически считает token-weighted loss на
`validation.jsonl`. Trainer работает через `CausalLMBackend` и не зависит от внутреннего класса
модели: один pipeline обслуживает собственный Transformer, Hugging Face causal LM, full
fine-tuning, LoRA и QLoRA. Реализованы CUDA, AdamW / AdamW8bit, automatic mixed precision, gradient clipping,
terminal progress, tokens/sec и GPU telemetry: имя GPU, allocated/reserved/peak VRAM. При старте
печатаются `steps_per_epoch`, `planned_total_steps`, число train samples и effective train tokens.
Во время обучения выводятся `samples_seen`, `tokens_seen`, процент текущей эпохи, текущий loss и
rolling average loss по последним 100 шагам.

Terminal output использует timestamp и категории `DEVICE`, `MODEL`, `PRETRAINED`, `PARAMS`, `DATASET`,
`TOKENS`, `DATALOADER`, `OPTIMIZER`, `TRAIN`, `VALIDATION` и `CHECKPOINT`. Долгая индексация JSONL,
fallback-пересчёт response tokens и validation периодически показывают процент, elapsed time и
скорость. Сохранение каждого checkpoint явно сообщает начало, путь и длительность записи.

Training objective — только assistant response. `input_ids` содержат BOS, последние сообщения
context, role-маркер `ASSISTANT`, response и EOS, но в `labels` context, role-маркер и padding
заменены на `-100`. Поэтому loss, validation loss и `tokens_seen` учитывают только текст response
и завершающий EOS.

Полное обучение с нуля на RTX 3050 4 GB и всём train split через API:

```powershell
$body = @{
    device = "cuda"
    batch_size = 4
    epochs = 3
    learning_rate = 0.0003
    validation_interval = 1000
    validation_batches = 200
    checkpoint_interval = 1000
    log_interval = 10
    num_workers = 0
} | ConvertTo-Json
$job = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/training" -ContentType "application/json" -Body $body
$job
```

`epochs = 3` означает ровно три полных прохода по train dataset. Альтернативный `max_steps` нужен
только для явно ограниченного запуска; HTTP-схема не позволяет одновременно передать оба поля,
поэтому лимит шагов не может незаметно оборвать заданные эпохи. Для custom по умолчанию validation
выполняется каждые 1000 шагов и использует до 200 batches; для pretrained — каждые 500 шагов
и до 50 batches.

Trainer создаёт в `checkpoints/training/`:

- `step_XXXXXXXX.pt` — периодические checkpoints;
- `best.pt` — checkpoint с минимальным validation loss;
- `last.pt` — последнее состояние при нормальном завершении или `Ctrl+C`.

Checkpoint содержит веса модели, AdamW state, AMP GradScaler, global step, точную epoch/batch
position, `samples_seen`, `tokens_seen`, окно последних loss, лучший validation loss и RNG state.
Для LoRA/QLoRA сохраняются только параметры PEFT-адаптера; при resume базовая revision снова
загружается из локального Hugging Face cache, затем восстанавливаются адаптер и optimizer state.
Для точного продолжения текущей эпохи `batch_size` должен совпадать с сохранённым. Продолжение из
последнего checkpoint до общего числа шести эпох:

```powershell
$body = @{
    device = "cuda"
    resume_from = "checkpoints/training/last.pt"
    batch_size = 4
    epochs = 6
    learning_rate = 0.0003
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/training" -ContentType "application/json" -Body $body
```

Продолжение из checkpoint с лучшим validation loss:

```powershell
$body = @{
    device = "cuda"
    resume_from = "checkpoints/training/best.pt"
    checkpoint_dir = "checkpoints/training_from_best"
    batch_size = 4
    epochs = 6
    learning_rate = 0.0003
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/training" -ContentType "application/json" -Body $body
```

Число эпох при resume — общий целевой номер, а не число дополнительных эпох. Learning rate из
запроса заменяет сохранённый learning rate optimizer. AMP включён по умолчанию; поле
`mixed_precision = false` оставлено для диагностики, но на GPU с 4 GB обычно не рекомендуется.
Если актуального
`token_statistics.json` нет, trainer один раз считает effective train tokens через Dataset перед
стартом обучения. Отдельный `training_from_best` сохраняет исходную ветку checkpoints без
перезаписи.

## Генерация ответа через API

После появления обученного checkpoint:

```powershell
$body = @{
    model_backend = "custom"
    prompt = "Привет! Объясни простыми словами, что такое Transformer."
    checkpoint_file = "checkpoints/training/best.pt"
    tokenizer_file = "artifacts/tokenizer/2ch_bpe.json"
    device = "cuda"
    max_new_tokens = 128
    temperature = 0.8
    top_k = 50
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/generate" -ContentType "application/json" -Body $body
```

Для активной pretrained-модели до нашего SFT:

```powershell
$body = @{
    model_backend = "pretrained"
    pretrained_mode = "before_sft"
    prompt = "Привет! Объясни простыми словами, что такое Transformer."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/generate" -ContentType "application/json" -Body $body
```

Для лучшего full/LoRA/QLoRA checkpoint активной модели:

```powershell
$body = @{
    model_backend = "pretrained"
    pretrained_mode = "after_sft"
    prompt = "Привет! Объясни простыми словами, что такое Transformer."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/generate" -ContentType "application/json" -Body $body
```

Для `custom` ручка использует `BPETokenizer` из `tokenizer_file`. Для `pretrained` передавать
`tokenizer_file` не требуется: Hugging Face tokenizer всегда определяется полями
`model_id/revision/cache_dir` из `pretrained_config_file`. Ручка восстанавливает model state из
training checkpoint только после SFT и возвращает текст вместе с новыми token IDs.
Для совместимости прямой API-запрос с явно переданным `checkpoint_file` и без `pretrained_mode`
по-прежнему выбирает after-SFT загрузку; пути можно переопределить. До SFT checkpoint не читается.
Отсутствующий checkpoint после SFT — ошибка, без молчаливого переключения на исходные веса.
Сам вызов inference модель не обучает.

## Параметры API и вспомогательные CLI

Полные HTTP-схемы и значения по умолчанию доступны в `/docs`. Основные endpoints:

- `POST /v1/training` — обучение или resume;
- `POST /v1/pretrained/prepare` — загрузка и подготовка full/LoRA/QLoRA;
- `POST /v1/tokenizer/train` — обучение tokenizer;
- `POST /v1/preprocessing` — preprocessing;
- `POST /v1/dataset/split` — split по тредам;
- `POST /v1/statistics/tokens` — token statistics;
- `POST /v1/scraper` — последовательный сбор тредов;
- `POST /v1/generate` — один ответ модели;
- `GET /v1/generate/options` — доступность режимов активной pretrained-модели без загрузки весов;
- `GET /v1/jobs/{job_id}` — состояние фоновой операции.

Для быстрых диагностических утилит сохранены прямые CLI:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.data.dataset --help
.\.venv\Scripts\python.exe -m mini_llm.pretrained --help
.\.venv\Scripts\python.exe -m mini_llm.training --help
.\.venv\Scripts\python.exe -m mini_llm.training.overfit --help
```

## Что уже реализовано

- decoder-only Transformer и генерация token IDs;
- JSON scraper 2ch с immutable raw storage;
- preprocessing достоверных explicit reply relations;
- multi-reference и reply chains;
- очистка PII, exact/near deduplication и отчёт preprocessing;
- детерминированный thread-level split без leakage;
- byte-level BPE, encode/decode, save/load и role tokens;
- подготовка окон: 1024 tokens для custom, 512 по умолчанию для pretrained;
- raw/effective token statistics по split и board;
- PyTorch Dataset/DataLoader, causal shift и dynamic padding;
- response-only labels/loss для custom и pretrained моделей;
- безопасный пропуск oversized responses;
- отдельный ручной overfit sanity-check;
- CUDA/AMP training (BF16 для pretrained), AdamW / AdamW8bit, gradient clipping и validation loss;
- periodic/best/last checkpoints и продолжение обучения;
- автоматический setup по Hugging Face model ID, закрепление commit и локальная регистрация;
- Hugging Face model ID/revision/cache, full fine-tuning, LoRA и QLoRA;
- родной instruct chat template в training/inference и режимы до/после нашего SFT;
- локальный FastAPI с последовательными background jobs и генерацией по checkpoint;
- минимальный Streamlit-чат с выбором custom/pretrained, без истории и памяти.

## Что ещё не реализовано

- scheduler и gradient accumulation;
- расширенные validation metrics и model evaluation;
- distributed/multi-GPU training;
- production job queue, аутентификация и постоянное хранение статусов API.

## Проверки качества

Быстрые unit tests используют только временные маленькие fixtures и не запускают полный pipeline:

Для проверок Streamlit-экрана установите optional-группу `ui` вместе с `dev`; без неё эти тесты
пропускаются. UI-тесты используют официальный `AppTest`, не запускают браузер, реальные
HTTP-запросы или загрузку моделей.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe .
.\.venv\Scripts\bandit.exe -r src -q
```
