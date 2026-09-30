# Mini LLM с нуля

Учебный проект компактной decoder-only Transformer-модели на PyTorch и последовательного
pipeline подготовки текстовых диалогов из тредов 2ch. Репозиторий содержит сборщик JSON API,
preprocessing графа ответов, разбиение датасета по тредам, обучение собственного byte-level BPE
tokenizer, расчёт token statistics и однопроцессное обучение модели на CUDA.

В проекте пока нет distributed training, scheduler, production model evaluation и текстового
интерфейса. Код обучения реализован, но сама модель в репозитории не обучена и не умеет осмысленно
отвечать.

## Установка и VS Code

Выберите `.venv` как Python interpreter в VS Code и синхронизируйте зависимости:

```powershell
uv sync --extra dev
```

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
from mini_llm import DecoderOnlyTransformer, ModelConfig

config = ModelConfig()
model = DecoderOnlyTransformer(config)
```

Модель принимает `input_ids` формы `[batch, sequence_length]`. Без `targets` она возвращает
logits, а с `targets` — `(logits, loss)`. Метод `generate()` работает с token IDs, а не с текстом.

## Структура проекта

```text
src/mini_llm/
├── config.py                # конфигурация модели и special token IDs
├── model.py                 # attention, MLP, Transformer-блок и модель
├── scraper.py               # сбор тредов через JSON API 2ch
├── preprocessing.py         # reply graph и dialogue samples
├── preprocessing_stats.py   # статистика preprocessing
├── deduplication.py         # точные и консервативные near-duplicates
├── dialogue_format.py       # схема JSONL и role-token serialization
├── dataset_split.py         # train/validation/test по тредам
├── bpe_tokenizer.py         # обучение, сохранение и загрузка BPE
├── token_statistics.py      # raw/effective token statistics
├── data_pipeline.py         # PyTorch Dataset, padding и DataLoader
├── overfit_test.py          # отдельный ручной sanity-check обучения
├── training_checkpoint.py   # checkpoint serialization и resume state
└── training.py              # CUDA/AMP training и validation

data/raw/2ch/<board>/         # неизменяемые raw JSON тредов
data/processed/               # dialogue dataset, split и отчёты
artifacts/tokenizer/          # обученный tokenizer JSON
tests/                        # быстрые unit tests
```

## Сбор тредов 2ch

Scraper использует JSON API, не скачивает вложения и не пытается обходить Cloudflare, CAPTCHA,
авторизацию или anti-bot защиту. В HTTP-клиенте есть один `requests.Session`, User-Agent, timeout,
последовательные запросы, случайная задержка, ограниченные retries, exponential backoff и обработка
`Retry-After` для HTTP 429.

Один тред:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://2ch.org/b/res/THREAD_ID.html"
```

Актуальные треды доски, не более десяти новых:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://2ch.org/b/" --max-threads 10
```

Каждый тред сохраняется в `data/raw/2ch/<board>/<thread_id>.json`. Существующий raw-файл никогда
не перезаписывается, а `--max-threads` учитывает только новые треды.

## 1. Preprocessing

```powershell
.\.venv\Scripts\python.exe -m mini_llm.preprocessing
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
.\.venv\Scripts\python.exe -m mini_llm.dataset_split
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
.\.venv\Scripts\python.exe -m mini_llm.bpe_tokenizer
```

Этап читает `data/processed/splits/train.jsonl`, обучает byte-level BPE с vocabulary size 8192 и
сохраняет `artifacts/tokenizer/2ch_bpe.json`. Validation и test при обучении не читаются. Byte-level
alphabet обеспечивает представление русского и английского текста, цифр, пунктуации, сленга,
эмодзи и смешанных строк без нормализации исходного стиля.

Повторно обучать tokenizer для использования не нужно:

```python
from pathlib import Path

from mini_llm.bpe_tokenizer import BPETokenizer

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
.\.venv\Scripts\python.exe -m mini_llm.token_statistics --max-sequence-length 1024
```

Этап читает три файла из `data/processed/splits/` и готовый
`artifacts/tokenizer/2ch_bpe.json`, затем создаёт
`data/processed/token_statistics.json`.

Отчёт содержит:

- фактический vocabulary size;
- raw/full tokens до ограничения окна;
- effective tokens после формирования допустимых окон;
- raw и effective tokens отдельно для train, validation и test;
- `effective_train_tokens_per_parameter`, рассчитанный только по train;
- characters per token;
- min, mean, median, p90, p95 и max raw/effective lengths;
- число полных и коротких effective windows;
- raw, effective и отброшенные context tokens, процент потерь и число сокращённых samples;
- отдельное распределение response lengths;
- unusable oversized responses и их распределение по board;
- количество и долю `<UNK>`;
- статистику по board.

`effective_train_tokens` — число token IDs в пригодных train windows после сохранения полного
response и сокращения context до лимита 1024. Validation/test не входят в
`effective_train_tokens_per_parameter`. Oversized response не обрезается, даёт ноль effective
tokens и остаётся в исходном split JSONL.

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
validation/test создаются без shuffle, последний неполный batch сохраняется. `batch_size` может
быть только от 1 до 4; большее значение отклоняется до запуска.

Быстрая проверка восьми пригодных train samples и одного CPU batch:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.data_pipeline --split train --max-samples 8 --batch-size 2
```

Команда читает готовые `data/processed/splits/train.jsonl` и
`artifacts/tokenizer/2ch_bpe.json`, но не обучает модель. GPU не используется.

## Ручной overfit sanity-check

Отдельный модуль позволяет проверить, уменьшается ли loss при многократном обучении на первых 32
пригодных train samples. Это не production trainer: в нём нет checkpoint, scheduler, validation,
resume, distributed training или полного прохода по корпусу.

Ручной запуск на GPU:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.overfit_test --device cuda --samples 16 --batch-size 1 --steps 50 --learning-rate 0.0003
```

Команда читает `train.jsonl` и готовый tokenizer, создаёт новую модель только в памяти и печатает
loss. Она использует CPU/RAM для Dataset и GPU/VRAM для модели и batch; автоматически не запускается
и ничего не сохраняет.

## Полноценное обучение

`mini_llm.training` обучает модель на `train.jsonl` и периодически считает token-weighted loss на
`validation.jsonl`. Реализованы CUDA, AdamW, automatic mixed precision, gradient clipping,
terminal progress, tokens/sec и GPU telemetry: имя GPU, allocated/reserved/peak VRAM. При старте
печатаются `steps_per_epoch`, `planned_total_steps`, число train samples и effective train tokens.
Во время обучения выводятся `samples_seen`, `tokens_seen`, процент текущей эпохи, текущий loss и
rolling average loss по последним 100 шагам.

Полное обучение с нуля на RTX 3050 4 GB и всём train split:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training --device cuda --batch-size 4 --epochs 3 --learning-rate 0.0003 --validation-interval 1000 --validation-batches 200 --checkpoint-interval 1000 --log-interval 10 --num-workers 0
```

`--epochs 3` означает ровно три полных прохода по train dataset. Альтернативный `--max-steps`
нужен только для явно ограниченного запуска; CLI не позволяет одновременно передать оба аргумента,
поэтому лимит шагов не может незаметно оборвать заданные эпохи. Значения по умолчанию:
`batch_size=4`, validation каждые 1000 шагов и до 200 validation batches.

Trainer создаёт в `checkpoints/training/`:

- `step_XXXXXXXX.pt` — периодические checkpoints;
- `best.pt` — checkpoint с минимальным validation loss;
- `last.pt` — последнее состояние при нормальном завершении или `Ctrl+C`.

Checkpoint содержит веса модели, AdamW state, AMP GradScaler, global step, точную epoch/batch
position, `samples_seen`, `tokens_seen`, окно последних loss, лучший validation loss и RNG state.
Для точного продолжения текущей эпохи `batch_size` должен совпадать с сохранённым. Продолжение из
последнего checkpoint до общего числа шести эпох:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training --device cuda --resume-from checkpoints/training/last.pt --batch-size 4 --epochs 6 --learning-rate 0.0003 --validation-interval 1000 --validation-batches 200 --checkpoint-interval 1000 --log-interval 10 --num-workers 0
```

Продолжение из checkpoint с лучшим validation loss:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.training --device cuda --resume-from checkpoints/training/best.pt --checkpoint-dir checkpoints/training_from_best --batch-size 4 --epochs 6 --learning-rate 0.0003 --validation-interval 1000 --validation-batches 200 --checkpoint-interval 1000 --log-interval 10 --num-workers 0
```

Число эпох при resume — общий целевой номер, а не число дополнительных эпох. Learning rate из
команды заменяет сохранённый learning rate optimizer. AMP включён по умолчанию; `--no-amp`
оставлен для диагностики, но на GPU с 4 GB обычно не рекомендуется. Если актуального
`token_statistics.json` нет, trainer один раз считает effective train tokens через Dataset перед
стартом обучения. Отдельный `training_from_best` сохраняет исходную ветку checkpoints без
перезаписи.

## Изменение путей и параметров

Все CLI поддерживают `--help`:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.preprocessing --help
.\.venv\Scripts\python.exe -m mini_llm.dataset_split --help
.\.venv\Scripts\python.exe -m mini_llm.bpe_tokenizer --help
.\.venv\Scripts\python.exe -m mini_llm.token_statistics --help
.\.venv\Scripts\python.exe -m mini_llm.data_pipeline --help
.\.venv\Scripts\python.exe -m mini_llm.overfit_test --help
.\.venv\Scripts\python.exe -m mini_llm.training --help
```

Основные параметры:

- preprocessing: `--input-dir`, `--output-dir`, `--review-size`, `--seed`;
- split: `--input-file`, `--output-dir`, `--seed`;
- tokenizer: `--train-file`, `--output-file`, `--vocab-size`, `--min-frequency`;
- statistics: `--splits-dir`, `--tokenizer-file`, `--output-file`,
  `--max-sequence-length`.
- Dataset smoke-check: `--split`, `--batch-size`, `--num-workers`, `--seed`, `--max-samples`;
- overfit test: `--samples`, `--batch-size`, `--steps`, `--learning-rate`, `--device`.
- training: `--batch-size`, `--epochs`, `--max-steps`, `--learning-rate`,
  `--validation-interval`, `--checkpoint-interval`, `--resume-from`, `--seed`.

## Что уже реализовано

- decoder-only Transformer и генерация token IDs;
- JSON scraper 2ch с immutable raw storage;
- preprocessing достоверных explicit reply relations;
- multi-reference и reply chains;
- очистка PII, exact/near deduplication и отчёт preprocessing;
- детерминированный thread-level split без leakage;
- byte-level BPE, encode/decode, save/load и role tokens;
- подготовка 1024-token window с приоритетом последних context messages;
- raw/effective token statistics по split и board;
- PyTorch Dataset/DataLoader, causal shift и dynamic padding;
- безопасный пропуск oversized responses;
- отдельный ручной overfit sanity-check;
- CUDA/AMP training, AdamW, gradient clipping и validation loss;
- periodic/best/last checkpoints и продолжение обучения.

## Что ещё не реализовано

- labels/loss masking по ролям;
- scheduler и gradient accumulation;
- расширенные validation metrics и model evaluation;
- distributed/multi-GPU training;
- преобразование пользовательского текста в dialogue prompt;
- CLI или веб-интерфейс для inference.

## Проверки качества

Быстрые unit tests используют только временные маленькие fixtures и не запускают полный pipeline:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe .
.\.venv\Scripts\bandit.exe -r src -q
```
