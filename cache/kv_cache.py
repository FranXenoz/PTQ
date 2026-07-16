import torch

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
