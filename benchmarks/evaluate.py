import time
import torch
import torch.nn as nn
import resource
import gc
from transformers import AutoTokenizer

# Set device
from models.loader import load_pretrained_gpt2, get_device

def get_peak_memory_mb() -> float:
    """
    Get peak resident memory usage of the process in MB.
    
    Returns:
        float: Peak memory in Megabytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes. On Linux, it is in kilobytes.
    # macOS OS check can be done implicitly or we can check platform.
    # Since the USER's OS is macOS (mac), it is in bytes.
    import sys
    if sys.platform == 'darwin':
        return usage.ru_maxrss / (1024.0 * 1024.0)
    else:
        return usage.ru_maxrss / 1024.0

def calculate_perplexity(model, text: str, tokenizer) -> float:
    """
    Calculate perplexity of the model on a given text.
    
    Args:
        model: Custom GPT-2 model, float32 or INT8.
        text: Input evaluation text, string.
        tokenizer: Tokenizer to use.
        
    Returns:
        float: Perplexity value.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Forward pass without cache for perplexity calculation
        logits, _ = model(input_ids)
        # Shift logits and targets for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
    return torch.exp(loss).item()

def evaluate_generation(model, tokenizer, prompt: str, max_new_tokens: int = 50, use_cache: bool = True) -> tuple[float, float, str]:
    """
    Measure Time-to-First-Token (TTFT), tokens/sec, and generate text.
    
    Args:
        model: Custom GPT-2 model.
        tokenizer: GPT-2 tokenizer.
        prompt: Prompt string.
        max_new_tokens: Number of tokens to generate.
        use_cache: If True, use KV-cache.
        
    Returns:
        Tuple containing:
            - ttft_ms: Time-to-first-token in milliseconds.
            - tokens_per_sec: Generation throughput in tokens/second.
            - generated_text: The decoded generated string.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    
    # Import sampler and generator if available, else do a basic local loop
    try:
        from engine.generator import generate
        # Run generator
        start_time = time.perf_counter()
        
        # We will use greedy decoding (T=0.0) for deterministic evaluation
        output_ids, metrics = generate(
            model, 
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=0.0, 
            top_k=0, 
            top_p=1.0, 
            use_cache=use_cache
        )
        total_time = time.perf_counter() - start_time
        
        ttft_ms = metrics.get("ttft_ms", 0.0)
        tokens_per_sec = metrics.get("tokens_per_sec", 0.0)
        generated_text = tokenizer.decode(output_ids[0])
        
    except ImportError:
        # Fallback local naive generation loop if generator.py is not implemented yet
        # (This is useful for Phase 1 where generator and KV cache are not fully done)
        ttft_ms = 0.0
        tokens_per_sec = 0.0
        
        with torch.no_grad():
            # Measure TTFT
            start_ttft = time.perf_counter()
            logits, past_key_values = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ttft_ms = (time.perf_counter() - start_ttft) * 1000.0
            
            generated_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Subsequence generation (naive, recomputing or using cache if available)
            start_gen = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                if use_cache and past_key_values is not None:
                    # Input is just the last token
                    logits, past_key_values = model(generated_ids[:, -1:], past_key_values=past_key_values)
                else:
                    # Recompute entire sequence
                    logits, _ = model(generated_ids)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            gen_time = time.perf_counter() - start_gen
            if gen_time > 0:
                tokens_per_sec = (max_new_tokens - 1) / gen_time
            generated_text = tokenizer.decode(generated_ids[0])
            
    return ttft_ms, tokens_per_sec, generated_text

def get_model_size_mb(model: nn.Module) -> float:
    """
    Compute the memory size of all parameters and buffers in the model in MB.
    """
    total_bytes = 0
    # Add parameters (embeddings, biases, layer norms)
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    # Add buffers (important for INT8 quantized weights!)
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024.0 * 1024.0)

def run_benchmarks():
    device = get_device()
    print(f"Running benchmarks on device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence is"
    eval_text = "The quick brown fox jumps over the lazy dog. Programming in Python is fun and rewarding. Deep learning enables many applications."
    
    # We will test three configurations:
    # 1. FP32 Baseline (no cache)
    # 2. FP32 + KV-Cache
    # 3. INT8 + KV-Cache
    
    # Let's check what modules are available
    has_quantizer = False
    try:
        from quantization.quantizer import quantize_model
        has_quantizer = True
    except ImportError:
        pass

    results = []

    # Config 1: FP32 Baseline (no cache)
    print("\nEvaluating FP32 Baseline (no cache)...")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load model
    model = load_pretrained_gpt2()
    weight_size_fp32 = get_model_size_mb(model)
    mem_baseline = get_peak_memory_mb()
    ppl_baseline = calculate_perplexity(model, eval_text, tokenizer)
    ttft_baseline, tps_baseline, text_baseline = evaluate_generation(
        model, tokenizer, prompt, max_new_tokens=30, use_cache=False
    )
    results.append({
        "Config": "FP32 Baseline (no cache)",
        "Weight Size (MB)": f"{weight_size_fp32:.1f}",
        "Peak RSS (MB)": f"{mem_baseline:.1f}",
        "TTFT (ms)": f"{ttft_baseline:.1f}",
        "Throughput (tok/s)": f"{tps_baseline:.1f}",
        "Perplexity": f"{ppl_baseline:.2f}"
    })
    
    # Config 2: FP32 + KV-Cache
    print("\nEvaluating FP32 + KV-Cache...")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    ppl_cache = calculate_perplexity(model, eval_text, tokenizer) # Should match baseline
    ttft_cache, tps_cache, text_cache = evaluate_generation(
        model, tokenizer, prompt, max_new_tokens=30, use_cache=True
    )
    results.append({
        "Config": "FP32 + KV-Cache",
        "Weight Size (MB)": f"{weight_size_fp32:.1f}",
        "Peak RSS (MB)": f"{mem_baseline:.1f}", # Peak RSS stays at baseline peak
        "TTFT (ms)": f"{ttft_cache:.1f}",
        "Throughput (tok/s)": f"{tps_cache:.1f}",
        "Perplexity": f"{ppl_cache:.2f}"
    })

    # Config 3: INT8 + KV-Cache
    if has_quantizer:
        print("\nEvaluating INT8 + KV-Cache...")
        # Quantize the model
        model = quantize_model(model)
        weight_size_int8 = get_model_size_mb(model)
        
        # Move back to device if needed (already handled by quantize_model, but safe)
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        mem_int8 = get_peak_memory_mb()
        ppl_int8 = calculate_perplexity(model, eval_text, tokenizer)
        ttft_int8, tps_int8, text_int8 = evaluate_generation(
            model, tokenizer, prompt, max_new_tokens=30, use_cache=True
        )
        results.append({
            "Config": "INT8 + KV-Cache",
            "Weight Size (MB)": f"{weight_size_int8:.1f}",
            "Peak RSS (MB)": f"{mem_int8:.1f}",
            "TTFT (ms)": f"{ttft_int8:.1f}",
            "Throughput (tok/s)": f"{tps_int8:.1f}",
            "Perplexity": f"{ppl_int8:.2f}"
        })
    else:
        results.append({
            "Config": "INT8 + KV-Cache (Not Implemented)",
            "Weight Size (MB)": "N/A",
            "Peak RSS (MB)": "N/A",
            "TTFT (ms)": "N/A",
            "Throughput (tok/s)": "N/A",
            "Perplexity": "N/A"
        })

    # Print results table
    print("\n" + "="*95)
    print(f"{'Configuration':<30} | {'Weights (MB)':<12} | {'Peak RSS (MB)':<13} | {'TTFT (ms)':<10} | {'Speed (tok/s)':<13} | {'Perplexity':<10}")
    print("="*95)
    for res in results:
        print(f"{res['Config']:<30} | {res['Weight Size (MB)']:<12} | {res['Peak RSS (MB)']:<13} | {res['TTFT (ms)']:<10} | {res['Throughput (tok/s)']:<13} | {res['Perplexity']:<10}")
    print("="*95)

if __name__ == '__main__':
    run_benchmarks()
