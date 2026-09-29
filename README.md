# Mini LLM from scratch

A compact decoder-only Transformer architecture implemented in PyTorch, plus utilities for
collecting raw 2ch threads and converting their reply graphs into dialogue samples. The project
still has no tokenizer, training loop, checkpoints, or text interface.

The default model uses 4 Transformer blocks, a width of 256, 4 attention heads, an FFN width of
1024, a context window of 512 token IDs, and a vocabulary of 8192 token IDs. Token embeddings and
the language-model output projection share weights.

## Setup in VS Code

Select `.venv` as the Python interpreter, then install the project and development tools:

```powershell
uv sync --extra dev
```

If `uv` is unavailable, use `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

## Module map

- `mini_llm.config` defines the model hyperparameters and special token IDs. It is imported by
  Python code and has no command-line entry point.
- `mini_llm.model` contains causal attention, the MLP, Transformer blocks, the complete decoder,
  and token-ID generation. Import it through `mini_llm`; do not run the module directly.
- `mini_llm.scraper` is the collection CLI. It downloads only 2ch JSON metadata and post text into
  immutable files under `data/raw/2ch/`.
- `mini_llm.preprocessing` is the dataset-building CLI. It reads the completed raw snapshot and
  writes dialogue JSONL, statistics, and a review sample under `data/processed/`.
- `mini_llm.deduplication` and `mini_llm.preprocessing_stats` are internal helpers used by
  preprocessing; they are not standalone commands.

Show the CLI options without downloading or processing anything:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper --help
.\.venv\Scripts\python.exe -m mini_llm.preprocessing --help
```

## Create a model

```python
from mini_llm import DecoderOnlyTransformer, ModelConfig

config = ModelConfig()
model = DecoderOnlyTransformer(config)
```

The constructor prints the number of trainable parameters. The model accepts integer `input_ids`
with shape `[batch, sequence_length]`. Without targets it returns logits; with targets it returns a
`(logits, loss)` tuple. Generation operates on token IDs because a tokenizer is deliberately out of
scope for this stage.

```python
import torch

prompt = torch.tensor([[config.bos_token_id, 42, 108]], dtype=torch.long)
generated_ids = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=20)
```

## Collect text from 2ch

The collector uses the public 2ch JSON API. A thread URL downloads that complete thread and writes
one text-only JSON file to `data/raw/2ch/<board>/<thread_id>.json`:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://2ch.org/b/res/THREAD_ID.html"
```

A board URL downloads current threads sequentially. Use `--max-threads` to bound a run:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://2ch.org/b/" --max-threads 10
```

Each output contains thread metadata and a `posts` list with `post_id`, cleaned plain `text`, and
`references` such as `>>123`. Attachments are not downloaded or included. `--timeout` and
`--output-dir` override their defaults. Existing raw thread files are immutable: the collector skips
them, and `--max-threads N` counts only newly downloaded threads. The collector does not attempt to
bypass authentication, CAPTCHA, Cloudflare, or other anti-bot protection. Check the site's terms
and robots policy before collecting its content.

## Build the processed dialogue dataset

Preprocessing reads `data/raw/2ch/<board>/*.json` without modifying it and writes:

- `data/processed/2ch_dialogues.jsonl` — ordered context and response samples;
- `data/processed/2ch_preprocessing_stats.json` — filtering, length, deduplication, chain-depth,
  board, and multi-reference statistics;
- `data/processed/2ch_review_sample.jsonl` — deterministic random samples for manual review.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.preprocessing
```

The pipeline reconstructs reply ancestors from `post_id` and `references`, keeps conversational
style, removes technical identifiers and markup, excludes clearly attachment-dependent examples,
and conservatively removes long exact or near-copy responses across threads. It does not truncate
contexts to 512 tokens; token budgeting belongs to the future tokenizer stage. Input/output paths,
review size, and review seed can be changed with `--input-dir`, `--output-dir`, `--review-size`, and
`--seed`.

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
ruff check .
ruff format --check .
mypy .
bandit -r src
```
