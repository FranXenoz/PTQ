# PTQ — Transformer Inference Optimizer (INT8 PTQ + KV-Cache)

A from-scratch inference optimization engine for **GPT-2 Small (124M)**, built in pure PyTorch with no quantization or generation wrapper libraries. The point of this project was to implement the algorithms, not call a library that already did — no `bitsandbytes`/`auto-gptq`/`optimum`, no `model.generate()`, no HF `pipeline()`. Weights are loaded via `AutoModelForCausalLM.from_pretrained` only and then unpacked into a hand-written architecture.

## Key Features

1. **INT8 Post-Training Quantization (PTQ)** — Manual asymmetric, per-channel scale/zero-point linear quantization of every projection in the model (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc_in`, `fc_out`). Weights are stored as `int8` and dequantized on-the-fly to `float32` at each forward pass.
2. **KV-Cache** — Reduces autoregressive decoding from quadratic `O(N²)` to linear `O(N)` by caching past Key/Value states and only computing attention for the newest token each step.
3. **Sampling Engine** — Temperature scaling, Top-K, and Top-P (nucleus) sampling implemented as pure tensor ops, with greedy (`T=0`) decoding for deterministic/parity testing.

## Data Flow

```
input tokens
    -> token + position embeddings
    -> Transformer block (x12):
         INT8Linear Q/K/V/O projections (dequantized on the fly using S, Z)
         -> KV-cache: append new K, V to past K, V (sequence dim)
         -> causal attention over full K, V, using only the new Q
         -> MLP/FFN (also INT8Linear)
    -> logits
    -> sampler: temperature -> top-k -> top-p -> multinomial draw
    -> next token
```

Full math for quantization, the KV-cache update, and sampling — plus binding shape contracts for every module boundary — is in [`architecture.md`](architecture.md).

## Directory Structure

```
PTQ/
├── models/
│   └── loader.py          # Custom GPT-2 architecture & HF weight mapper
├── cache/
│   └── kv_cache.py        # KV-cache append operations
├── quantization/
│   └── quantizer.py       # Asymmetric quantize/dequantize & INT8Linear layer
├── engine/
│   ├── sampler.py         # Temperature, Top-K, Top-P sampling
│   └── generator.py       # Stateful O(N) autoregressive generation loop
├── benchmarks/
│   └── evaluate.py        # Accuracy (perplexity), throughput, and memory benchmarking
├── tests/
│   ├── test_baseline.py   # Verify logit parity with the HF reference model
│   ├── test_kv_cache.py   # Verify KV-cache decoding output parity
│   └── test_quantization.py # Verify INT8 weight reconstruction MSE (< 1e-3)
├── architecture.md        # Technical reference (formulas & shape contracts)
├── blueprint.md            # Project build specification and phase plan
├── coding-rules.md        # Implementation guardrails (banned shortcuts, module layout, parity tests)
└── .gitignore
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FranXenoz/PTQ.git
   cd PTQ
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers pytest
   ```

## Usage

### Running tests
Runs the two required parity tests (quantization round-trip MSE, and cached-vs-uncached generation producing identical token IDs) plus the baseline logit-parity check against Hugging Face's reference model:
```bash
PYTHONPATH=. pytest
```

### Running benchmarks
Compares FP32 (no cache), FP32 (with KV-cache), and INT8 (with KV-cache):
```bash
PYTHONPATH=. python3 benchmarks/evaluate.py
```
Each configuration is run in its own subprocess so that peak memory (`Peak RSS`) is measured independently per configuration rather than as a single running maximum across the whole script.

## Design Constraints

This project is built under a fixed set of rules (see [`coding-rules.md`](coding-rules.md) for the full list):

- **No shortcut libraries.** Quantization math, the generation loop, and sampling are all hand-implemented with core `torch` ops — no `bitsandbytes`, `auto-gptq`, `optimum`, `model.generate()`, or HF `pipeline()`.
- **Every tensor-touching function documents shape and dtype** for each argument and return value.
- **No hardcoded `'cuda'`/`'cpu'`** — device selection goes through one centralized dispatcher (`models/loader.py::get_device`) that checks CUDA → MPS → CPU.
- **One responsibility per module** (see the directory structure above).
- **Two required parity tests, both automated:**
  - *Quantization:* quantize → dequantize a random weight tensor must give MSE **< 1e-3**.
  - *KV-cache:* greedy (`T=0`) generation with the cache enabled must produce the **exact same token IDs** as recomputing the full sequence from scratch every step. Any divergence is treated as a cache bug, not acceptable drift.

## Benchmark Results

Run `PYTHONPATH=. python3 benchmarks/evaluate.py` on your target hardware and drop the resulting table here. Numbers are hardware- and device-dependent (CPU vs. CUDA vs. Apple Silicon MPS), so it's worth noting which device produced them, e.g.:

| Configuration | Weights Size (MB) | Peak RSS (MB) | TTFT (ms) | Throughput (tok/s) | Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FP32 Baseline (no cache) | | | | | |
| FP32 + KV-Cache | | | | | |
| INT8 + KV-Cache | | | | | |

What to look for:
- **Memory**: INT8 storage should shrink weight size by roughly the expected ~40-50% (weights only — the current `INT8Linear.forward` dequantizes to `float32` before the matmul, so this is a storage/footprint win rather than a reduction in compute-time activation memory).
- **Latency**: KV-cache should noticeably cut both TTFT and per-token generation time versus full recomputation.
- **Accuracy**: perplexity under INT8 should stay close to the FP32 baseline — a large jump would indicate a quantization bug rather than expected precision loss.

## Roadmap

See [`blueprint.md`](blueprint.md) for the full four-phase build plan (baseline → KV-cache → INT8 PTQ → sampling/full pipeline). A natural next step once the pipeline is verified end-to-end on GPT-2 is extending it to a larger target model (e.g. TinyLlama-1.1B).
