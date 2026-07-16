# Transformer Inference Optimizer (PTQ + KV-Cache)

I wrote a lightweight, from-scratch inference optimization engine for **GPT-2 Small (124M)**. This project implements core latency-reduction and memory-reduction techniques in pure PyTorch and NumPy, without wrapping libraries.

## Key Features

1. **INT8 Post-Training Quantization (PTQ)**: Shrinks weight memory footprint by ~40% via manual asymmetric, per-channel scale and zero-point linear quantization of linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `fc_in`, `fc_out`). Weights are stored in standard `int8` format and dequantized on-the-fly to float32.
2. **KV-Cache**: Reduces the autoregressive decoding complexity from quadratic $O(N^2)$ to linear $O(N)$ by caching past Key and Value attention states instead of recomputing them.
3. **Sampling Engine**: Implements Temperature scaling, Top-K, and Top-P (nucleus) sampling using core tensor operations.

---

## Directory Structure

```
qui/
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
│   ├── test_baseline.py   # Verify output parity with Hugging Face reference model
│   ├── test_kv_cache.py   # Verify KV-cache decoding output parity
│   └── test_quantization.py# Verify INT8 weight reconstruction MSE (< 1e-3)
├── architecture.md        # Technical reference (formulas & shape contracts)
├── blueprint.md           # Project build specifications and phases
├── coding-rules.md        # Code design conventions and shape requirements
└── .gitignore             # Standard git exclusions for Python & ML
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd qui
   ```

2. **Install dependencies**:
   Ensure you have PyTorch, Hugging Face Transformers, and Pytest installed:
   ```bash
   pip install torch transformers pytest
   ```

---

## Usage

### Running Tests
To run the automated parity tests verifying model correctness:
```bash
PYTHONPATH=. pytest
```

### Running Benchmarks
To run the performance benchmarks comparing FP32 (no cache), FP32 (with KV-cache), and INT8 (with KV-cache):
```bash
PYTHONPATH=. python3 benchmarks/evaluate.py
```

---

## Benchmark Results (Apple Silicon MPS)

The final performance evaluation of the custom loader on an Apple Silicon (MPS) device:

| Configuration | Weights Size (MB) | Peak RSS (MB) | TTFT (ms) | Throughput (tok/s) | Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline (no cache)** | 621.9 MB | 1529.0 MB | 15.4 ms | 67.5 tok/s | 131.93 |
| **FP32 + KV-Cache** | 621.9 MB | 1529.0 MB | 8.9 ms | 113.1 tok/s | 131.93 |
| **INT8 + KV-Cache** | 379.6 MB | 1529.0 MB | 11.2 ms | 103.5 tok/s | 133.76 |

### Core Observations
* **Memory savings**: Quantization of block linear projections achieves **nearly 40% memory reduction** of the total model weights (621.9 MB $\rightarrow$ 379.6 MB).
* **Latency reduction**: Caching past keys and values yields **1.67x generation speedup** and significantly cuts TTFT down to **8.9 ms** for prompt prefill.
* **Accuracy preservation**: Per-channel asymmetric linear quantization preserves perplexity within **1.4% change** compared to the FP32 baseline (131.93 $\rightarrow$ 133.76), keeping outputs high-quality.
