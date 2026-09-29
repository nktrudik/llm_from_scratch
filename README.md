# Mini LLM с нуля

Учебный проект компактной decoder-only Transformer-модели на PyTorch и последовательного
pipeline подготовки текстовых диалогов из тредов 2ch. Репозиторий содержит сборщик JSON API,
preprocessing графа ответов, разбиение датасета по тредам, обучение собственного byte-level BPE
tokenizer и расчёт token statistics.

В проекте пока нет training loop, checkpoint, optimizer, batching, оценки качества модели и
текстового интерфейса. Реализованная модель не обучена и не умеет осмысленно отвечать.

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
- максимальный контекст 512 token IDs;
- vocabulary size 8192;
- pre-LayerNorm, residual connections и настраиваемый dropout;
- общие веса token embedding и выходной проекции;
- 5 387 776 обучаемых параметров.

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
└── token_statistics.py      # отчёт по token lengths и vocabulary

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

Raw-файлы не меняются. Длинные samples не удаляются и не обрезаются до 512 tokens. Имеющийся
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
сообщение перед response всегда имело `<USER>`. `encode_dialogue_window(..., max_length=512)`
сохраняет response целиком и заполняет остаток окна последними context messages. Если полный
response сам не помещается, метод возвращает ошибку вместо скрытого обрезания.

Ресурсы: CPU и RAM; GPU не используется и не требуется. Обучение BPE — самый ресурсоёмкий этап
подготовки после preprocessing, но выполняется Rust-реализацией библиотеки `tokenizers`.

## 4. Token statistics

После обучения tokenizer:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.token_statistics
```

Этап читает три файла из `data/processed/splits/` и готовый
`artifacts/tokenizer/2ch_bpe.json`, затем создаёт
`data/processed/token_statistics.json`.

Отчёт содержит:

- фактический vocabulary size;
- tokens в train, validation, test и суммарно;
- tokens per parameter для текущей модели;
- characters per token;
- min, mean, median, p90, p95 и max длины samples;
- количество и долю samples `<= 512` и `> 512`;
- отдельные длины и объёмы context/response;
- количество и долю `<UNK>`;
- статистику по board.

Этап ничего не удаляет и не создаёт tokenized training dataset. Ресурсы: CPU и последовательное
чтение диска; RAM используется для массивов длин samples. GPU не используется и не требуется.

## Изменение путей и параметров

Все CLI поддерживают `--help`:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.preprocessing --help
.\.venv\Scripts\python.exe -m mini_llm.dataset_split --help
.\.venv\Scripts\python.exe -m mini_llm.bpe_tokenizer --help
.\.venv\Scripts\python.exe -m mini_llm.token_statistics --help
```

Основные параметры:

- preprocessing: `--input-dir`, `--output-dir`, `--review-size`, `--seed`;
- split: `--input-file`, `--output-dir`, `--seed`;
- tokenizer: `--train-file`, `--output-file`, `--vocab-size`, `--min-frequency`;
- statistics: `--splits-dir`, `--tokenizer-file`, `--output-file`,
  `--max-sequence-length`.

## Что уже реализовано

- decoder-only Transformer и генерация token IDs;
- JSON scraper 2ch с immutable raw storage;
- preprocessing достоверных explicit reply relations;
- multi-reference и reply chains;
- очистка PII, exact/near deduplication и отчёт preprocessing;
- детерминированный thread-level split без leakage;
- byte-level BPE, encode/decode, save/load и role tokens;
- подготовка 512-token window с приоритетом последних context messages;
- token statistics по split и board.

## Что ещё не реализовано

- PyTorch Dataset/DataLoader и окончательная подготовка training batches;
- labels/loss masking по ролям;
- training loop, optimizer, scheduler и mixed precision;
- checkpoints и возобновление обучения;
- validation metrics и model evaluation;
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
