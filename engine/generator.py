import time
import torch
from engine.sampler import sample_next_token

def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cache: bool = True
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Generate new tokens autoregressively from a starting prompt.

    Args:
        model:          Custom GPT-2 model.
        prompt_ids:     [batch, prompt_len], int64 prompt token IDs.
        max_new_tokens: Number of tokens to generate.
        temperature:    Temperature for sampling (0.0 for greedy).
        top_k:          Top-K sampling threshold.
        top_p:          Top-P sampling threshold.
        use_cache:      Whether to use KV-cache to avoid recomputation.

    Returns:
        Tuple containing:
            - generated_ids: [batch, prompt_len + max_new_tokens], int64.
            - metrics: Dict containing 'ttft_ms' and 'tokens_per_sec'.
    """
    device = prompt_ids.device
    batch_size, prompt_len = prompt_ids.shape
    
    generated_ids = prompt_ids.clone()
    
    # 1. First step (Prompt Processing / Prefill)
    start_ttft = time.perf_counter()
    with torch.no_grad():
        logits, past_key_values = model(prompt_ids)
        
    next_token = sample_next_token(
        logits[:, -1, :], 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p
    )
    ttft_ms = (time.perf_counter() - start_ttft) * 1000.0
    
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 2. Autoregressive generation loop
    start_gen = time.perf_counter()
    tokens_generated = 1
    
    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            if use_cache:
                # Cache-based forward pass: input is just the last token generated
                input_ids = generated_ids[:, -1:]
                logits, past_key_values = model(input_ids, past_key_values=past_key_values)
            else:
                # Full recomputation: input is the entire generated sequence
                logits, _ = model(generated_ids)
                
            next_token = sample_next_token(
                logits[:, -1, :], 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            tokens_generated += 1
            
    gen_time = time.perf_counter() - start_gen
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0.0
    
    metrics = {
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec
    }
    
    return generated_ids, metrics
