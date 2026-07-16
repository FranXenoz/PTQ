# Blueprint: Transformer Inference Optimizer (PTQ + KV-Cache)

## Objective
Build, from scratch, an inference optimization engine for a small open-source
Transformer that implements — without wrapper libraries — three core
techniques:

1. **INT8 Post-Training Quantization (PTQ)** — shrink weight memory ~50% via
   manual scale/zero-point linear quantization.
2. **KV-Cache** — cut autoregressive generation cost from O(N²) to O(N) by
   caching past keys/values instead of recomputing them.
3. **Sampling engine** — Temperature, Top-K, and Top-P (nucleus) decoding.

Full math for each is in `architecture.md`. All implementation constraints
(banned libraries, module layout, shape-doc rules, parity tests) are in
`coding-rules.md`. This file only defines *what* to build and in *what order*.

## Target Model
`GPT-2 Small (124M)` — fits on CPU, well-documented shapes, fast iteration.
(`TinyLlama-1.1B` is an optional stretch target once the pipeline is verified
end-to-end on GPT-2.)

## Build Order

Each phase must be fully working and verified before the next one starts.
Do not parallelize phases — later phases depend on earlier ones being correct.

### Phase 1 — Baseline (no cache, no quantization, FP32)
- Load pretrained weights via `AutoModelForCausalLM.from_pretrained` only
  (no `pipeline()`, no `.generate()`).
- Write a manual forward pass for one Transformer block: LayerNorm →
  Multi-Head Attention → MLP/FFN.
- Write a baseline benchmark script: time-to-first-token, tokens/sec, peak
  memory (MB).
- **Exit criteria:** manual forward pass produces logits matching the
  reference HF forward pass (no cache) to float precision.

### Phase 2 — KV-Cache
- Extend attention to accept/return `past_key_values` and append new K/V
  along the sequence dimension.
- **Exit criteria:** cached and non-cached forward passes match to `1e-5`
  tolerance on logits (see `coding-rules.md` §5 for the full parity test).

### Phase 3 — INT8 PTQ
- Implement quantize/dequantize primitives (asymmetric, per-tensor or
  per-channel scale/zero-point).
- Wrap every Linear layer in the model (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`, and the MLP's `fc_in`/`fc_out`) in a custom `INT8Linear` module.
- **Exit criteria:** quantize→dequantize round-trip on a random weight
  tensor has MSE < 1e-3 (see `coding-rules.md` §5).

### Phase 4 — Sampling + Full Pipeline + Benchmarks
- Implement Temperature scaling, Top-K, Top-P as pure tensor ops.
- Write the explicit autoregressive generation loop (quantized model +
  active KV-cache + sampler).
- Produce a final report: memory footprint, tokens/sec, and perplexity,
  each compared against the Phase-1 FP32 baseline.

## Definition of Done
- All four phases pass their exit criteria.
- `benchmarks/evaluate.py` produces one table: Baseline vs. KV-Cache-only
  vs. KV-Cache+INT8, with memory (MB), throughput (tok/s), and perplexity
  for each.
