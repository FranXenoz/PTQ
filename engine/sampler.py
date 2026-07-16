import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> torch.Tensor:
    """
    Sample the next token ID from raw logits using temperature, Top-K, and Top-P filtering.

    Args:
        logits:      [batch, vocab_size], float32 raw logits from the model's head.
        temperature: Temperature scaling factor. 0.0 forces greedy argmax decoding.
        top_k:       Number of highest probability tokens to keep (0 disables Top-K).
        top_p:       Cumulative probability threshold for nucleus sampling (1.0 disables Top-P).

    Returns:
        [batch, 1], int64 next token IDs.
    """
    assert logits.dim() == 2, "Logits must be [batch, vocab_size]"
    
    if temperature == 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply temperature
    logits = logits / temperature

    # Apply Top-K
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Find values of top k elements
        v, _ = torch.topk(logits, top_k, dim=-1)
        # Mask out any value less than the min value in the top k
        min_values = v[:, [-1]]
        logits = logits.masked_fill(logits < min_values, float("-inf"))

    # Apply Top-P (nucleus sampling)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Create mask for tokens to remove (cumulative prob > top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift mask to the right to keep the first token that crossed the top_p threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Map the mask back to the original logits indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
