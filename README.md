# Mini LLM from scratch

A compact decoder-only Transformer architecture implemented in PyTorch, plus a small utility for
collecting raw text from ordinary HTML pages. The project still has no tokenizer, training loop,
checkpoints, or text interface.

The default model uses 4 Transformer blocks, a width of 256, 4 attention heads, an FFN width of
1024, a context window of 512 token IDs, and a vocabulary of 8192 token IDs. Token embeddings and
the language-model output projection share weights.

## Setup in VS Code

Select `.venv` as the Python interpreter, then install the project and development tools:

```powershell
uv sync --extra dev
```

If `uv` is unavailable, use `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

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

## Collect one raw HTML document

The scraper accepts one public HTTP(S) page, extracts useful paragraphs, and writes a JSON document
containing `source_url`, `title`, `text`, and `fetched_at` to `data/raw/`:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://example.com/page"
```

For `2ch.org` board and thread pages, the extractor prioritizes the current
`.post .post__message` structure, so post controls, timestamps, and reply forms are not mixed into
the collected text. For example:

```powershell
.\.venv\Scripts\python.exe -m mini_llm.scraper "https://2ch.org/b/res/THREAD_ID.html"
```

Use `--min-text-length`, `--min-block-length`, `--timeout`, or `--output-dir` to override the
conservative defaults. The scraper is sequential and does not attempt to bypass authentication,
CAPTCHA, Cloudflare, or other anti-bot protection. Check a site's terms and robots policy before
collecting its content.

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
ruff check .
ruff format --check .
mypy .
bandit -r src
```
