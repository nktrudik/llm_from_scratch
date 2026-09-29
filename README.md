# Mini LLM from scratch

A compact decoder-only Transformer architecture implemented in PyTorch. This stage contains
the model only: it has no tokenizer, dataset, training loop, checkpoints, or text interface yet.

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

Run the checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest
ruff check .
ruff format --check .
mypy .
bandit -r src
```
