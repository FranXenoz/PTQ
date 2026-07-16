# Coding Rules: Guardrails for Any Agent Working on This Repo

These rules are binding for every commit. If a change conflicts with one of
these, the rule wins — flag the conflict instead of silently working around
it.

## 1. Banned Shortcuts
The point of this project is implementing the algorithms, not calling a
library that already did. Do not use:
- `bitsandbytes`, `auto-gptq`, `optimum` — implement quantization math by
  hand with core `torch` ops.
- `model.generate()` — write the autoregressive loop explicitly.
- HuggingFace `pipeline()` — load via `AutoModelForCausalLM.from_pretrained`
  only, then unpack into the custom/modified blocks.

## 2. Mandatory Docstring Format
Every function/method touching a tensor documents shape and dtype for each
argument and the return value. Example:

```python
def append_to_kv_cache(
    past_key: torch.Tensor,
    new_key: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate new_key onto past_key along the sequence dimension.

    Args:
        past_key: [batch, heads, seq_len, head_dim], float16 or float32.
        new_key:  [batch, heads, 1, head_dim], dtype must match past_key.

    Returns:
        [batch, heads, seq_len + 1, head_dim], same dtype as inputs.
    """
    assert past_key.dtype == new_key.dtype, "Dtype mismatch in KV cache append"
    return torch.cat([past_key, new_key], dim=2)
```

## 3. Memory & Hardware Rules
- Prefer in-place ops (`add_`, `mul_`, `clamp_`) inside the generation loop
  to avoid extra allocations.
- No hardcoded `'cuda'`/`'cpu'`. Use one centralized dispatcher:
  ```python
  device = torch.device(
      "cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available()
      else "cpu"
  )
  ```
- No `.item()` or `.cpu().numpy()` inside the hot generation loop — these
  force a GPU↔CPU sync and kill throughput. Keep everything as tensors
  until generation is finished.

## 4. Module Layout
One responsibility per file. Don't merge these:

| Path                       | Responsibility                                              |
|-----------------------------|--------------------------------------------------------------|
| `models/loader.py`          | Load raw weights, map to the custom architecture             |
| `quantization/quantizer.py` | `quantize_tensor`, `dequantize_tensor`, `compute_scales`, `INT8Linear` |
| `cache/kv_cache.py`         | Cache allocation, append, (eviction if ever needed)           |
| `engine/sampler.py`         | Temperature / Top-K / Top-P / multinomial draw                |
| `engine/generator.py`       | The step-by-step autoregressive orchestration loop            |
| `benchmarks/evaluate.py`    | Memory, tokens/sec, and perplexity measurement                |

## 5. Required Parity Tests
No module is done until it passes its parity test.

- **Quantization parity:** quantize→dequantize a random weight tensor →
  mean squared error must be **< 1e-3**.
- **KV-cache parity:** with temperature `T = 0.0` (greedy decoding),
  generating 50 tokens with the cache enabled must produce the **exact
  same token IDs** as recomputing the full sequence from scratch every
  step (no cache). Any divergence is a bug in the cache implementation,
  not acceptable numerical drift.

Both tests must be automated (e.g. `pytest`) and runnable independently of
the full pipeline — don't require a full model load and generation run
just to check the quantization math.
