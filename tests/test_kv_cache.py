import torch
import pytest
from transformers import AutoTokenizer
from models.loader import load_pretrained_gpt2, get_device
from engine.generator import generate

def test_kv_cache_parity():
    device = get_device()
    model = load_pretrained_gpt2()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The quick brown fox jumps over the lazy dog"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    max_new_tokens = 50
    
    # 1. Generate with cache
    out_with_cache, metrics_with_cache = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy
        use_cache=True
    )
    
    # 2. Generate without cache (recomputing from scratch every step)
    out_no_cache, metrics_no_cache = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy
        use_cache=False
    )
    
    # Check that generated token IDs are identical
    token_ids_with_cache = out_with_cache[0].tolist()
    token_ids_no_cache = out_no_cache[0].tolist()
    
    assert token_ids_with_cache == token_ids_no_cache, (
        f"KV-cache parity test failed!\n"
        f"With cache: {token_ids_with_cache}\n"
        f"No cache:   {token_ids_no_cache}"
    )
    print("KV-cache parity test passed! Token IDs are exactly identical.")
