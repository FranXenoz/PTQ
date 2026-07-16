# Architecture: Data Flow, Math, and Shape Contracts

Companion to `blueprint.md` (phase plan) and `coding-rules.md`
(implementation rules). This file is the technical reference — read the
relevant section before implementing that piece, don't re-derive it.

## 1. Data Flow (one generation step)

```
input tokens
    -> token embeddings
    -> Transformer block:
         INT8Linear Q/K/V/O projections (dequantized on the fly using S, Z)
         -> KV-cache: append new K,V to past K,V (sequence dim)
         -> attention over full K,V, using only the new Q
         -> MLP/FFN (also INT8Linear)
    -> logits
    -> sampler: temperature -> top-k -> top-p -> multinomial draw
    -> next token
```

## 2. Module 1 — INT8 Post-Training Quantization (Asymmetric)

Given a float weight tensor `W`, with `q_min = -128`, `q_max = 127`:

```
S = (max(W) - min(W)) / (q_max - q_min)
Z = round(-min(W) / S) + q_min
W_q = clamp(round(W / S) + Z, q_min, q_max)          # quantize
W~  = S * (W_q - Z)                                  # dequantize (inference)
```

**Rule:** `S` and `Z` are computed once at load time and stored alongside
`W_q`. Never recompute them at inference time — that reintroduces the
latency this module exists to remove.

## 3. Module 2 — KV-Cache

At generation step `t`, only one new token is processed (`L_new = 1`):

1. Compute `Q_t, K_t, V_t` for the new token.
2. Load cached `K_past, V_past` — shape
   `[batch, heads, seq_len_so_far, head_dim]`.
3. Concatenate on the sequence dimension:
   `K_total = concat(K_past, K_t)`, `V_total = concat(V_past, V_t)`.
4. Attend: `softmax(Q_t @ K_total^T / sqrt(head_dim)) @ V_total`.
5. Store `K_total, V_total` back as the new cache.

This turns each step's attention cost from `O(seq_len²)` into
`O(seq_len)`, since `Q_t` is a single position, not the whole sequence.

## 4. Module 3 — Sampling

Given raw logits `z` (vocab-size vector):

1. **Temperature:** `z' = z / T`
2. **Top-K:** keep the `K` largest entries of `z'`, set everything else to
   `-inf`.
3. **Top-P (nucleus):** sort descending, take softmax `p = softmax(z'')`,
   keep the smallest prefix of tokens whose cumulative probability ≥ `P`,
   set the rest to `-inf`, re-normalize.
4. Draw the next token by multinomial sampling from the final
   distribution (or argmax if `T = 0`, i.e. greedy decoding — required for
   the KV-cache parity test in `coding-rules.md` §5).

## 5. Shape Contracts

These are binding — any function violating them is a bug, not a style
choice.

| Component        | Input shape                                | Output shape                                   |
|-------------------|---------------------------------------------|--------------------------------------------------|
| Quantizer         | `[out_channels, in_channels]` (fp32)         | `[out_channels, in_channels]` (int8) + scalars `S, Z` |
| KV-cache append   | new K/V: `[batch, heads, 1, head_dim]`       | full cache: `[batch, heads, seq_len, head_dim]`   |
| Attention step    | `[batch, seq, embed_dim]`                    | `[batch, seq, embed_dim]`                         |
| Sampler           | `[batch, vocab_size]` (raw logits)           | `[batch, 1]` (next token id)                      |

Every function that touches a tensor above must document its shape and
dtype in its docstring — see `coding-rules.md` §2 for the required format.
