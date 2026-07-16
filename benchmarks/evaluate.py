import sys
import json
import time
import argparse
import subprocess
import torch
import torch.nn as nn
import resource
import gc
from transformers import AutoTokenizer

# Set device
from models.loader import load_pretrained_gpt2, get_device

# Marker line used to smuggle a JSON result out of a subprocess's stdout,
# which may also contain ordinary print() noise (progress messages, warnings, etc).
RESULT_MARKER = "RESULT_JSON::"

# Each entry is run in its own fresh subprocess so that ru_maxrss (which is a
# monotonically increasing "peak so far" counter for the life of a process)
# actually reflects that configuration's memory usage, rather than the peak
# across all configurations run so far in one long-lived process.
CONFIGS = ["fp32_baseline", "fp32_cache", "int8_cache"]

def get_peak_memory_mb() -> float:
    """
    Get peak resident memory usage of the *current process* in MB.

    Only meaningful when called in a process that has run exactly one
    benchmark configuration, since ru_maxrss never decreases within a
    process's lifetime.

    Returns:
        float: Peak memory in Megabytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS, ru_maxrss is in bytes. On Linux, it is in kilobytes.
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
    from engine.generator import generate

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # We use greedy decoding (T=0.0) for deterministic, reproducible evaluation.
    output_ids, metrics = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        use_cache=use_cache
    )

    ttft_ms = metrics.get("ttft_ms", 0.0)
    tokens_per_sec = metrics.get("tokens_per_sec", 0.0)
    generated_text = tokenizer.decode(output_ids[0])

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

def run_single_config(config: str) -> dict:
    """
    Run exactly one benchmark configuration in the current process and
    return its metrics, including this process's peak RSS.

    This is meant to be invoked as a subprocess (one per config) by
    run_benchmarks(), so that get_peak_memory_mb() reflects only the
    memory used by this configuration rather than the running maximum
    across every configuration executed so far.

    Args:
        config: One of "fp32_baseline", "fp32_cache", "int8_cache".

    Returns:
        dict: Metrics for this configuration (weight size, peak RSS,
            TTFT, throughput, perplexity).
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence is"
    eval_text = "The quick brown fox jumps over the lazy dog. Programming in Python is fun and rewarding. Deep learning enables many applications."

    model = load_pretrained_gpt2()

    use_cache = config in ("fp32_cache", "int8_cache")
    label = {
        "fp32_baseline": "FP32 Baseline (no cache)",
        "fp32_cache": "FP32 + KV-Cache",
        "int8_cache": "INT8 + KV-Cache",
    }[config]

    if config == "int8_cache":
        from quantization.quantizer import quantize_model
        model = quantize_model(model)
        model.to(device)

    weight_size_mb = get_model_size_mb(model)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ppl = calculate_perplexity(model, eval_text, tokenizer)
    ttft_ms, tokens_per_sec, _ = evaluate_generation(
        model, tokenizer, prompt, max_new_tokens=30, use_cache=use_cache
    )

    # Measured last, after model load + quantization + inference, so it
    # captures this configuration's actual peak resident memory.
    peak_rss_mb = get_peak_memory_mb()

    return {
        "Config": label,
        "Weight Size (MB)": f"{weight_size_mb:.1f}",
        "Peak RSS (MB)": f"{peak_rss_mb:.1f}",
        "TTFT (ms)": f"{ttft_ms:.1f}",
        "Throughput (tok/s)": f"{tokens_per_sec:.1f}",
        "Perplexity": f"{ppl:.2f}",
    }


def run_benchmarks():
    """
    Run all benchmark configurations, each in its own fresh subprocess,
    then print a combined results table.

    Isolating each configuration in a subprocess is what makes the
    "Peak RSS" column meaningful: resource.getrusage's ru_maxrss is a
    high-water mark for the life of a process and never resets, so
    measuring multiple configs in one long-lived process just reports
    the same running maximum for every row.
    """
    print(f"Running benchmarks on device: {get_device()}")

    results = []
    for config in CONFIGS:
        print(f"\nEvaluating {config}...")
        proc = subprocess.run(
            [sys.executable, __file__, "--config", config],
            capture_output=True,
            text=True,
        )
        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_MARKER):
                result_line = line[len(RESULT_MARKER):]

        if proc.returncode != 0 or result_line is None:
            print(f"  Failed to evaluate {config}:")
            print(proc.stderr[-2000:])
            results.append({
                "Config": f"{config} (FAILED)",
                "Weight Size (MB)": "N/A",
                "Peak RSS (MB)": "N/A",
                "TTFT (ms)": "N/A",
                "Throughput (tok/s)": "N/A",
                "Perplexity": "N/A",
            })
            continue

        results.append(json.loads(result_line))

    # Print results table
    print("\n" + "=" * 95)
    print(f"{'Configuration':<30} | {'Weights (MB)':<12} | {'Peak RSS (MB)':<13} | {'TTFT (ms)':<10} | {'Speed (tok/s)':<13} | {'Perplexity':<10}")
    print("=" * 95)
    for res in results:
        print(f"{res['Config']:<30} | {res['Weight Size (MB)']:<12} | {res['Peak RSS (MB)']:<13} | {res['TTFT (ms)']:<10} | {res['Throughput (tok/s)']:<13} | {res['Perplexity']:<10}")
    print("=" * 95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=CONFIGS,
        default=None,
        help="Internal: run a single configuration in this process and print its result as JSON.",
    )
    args = parser.parse_args()

    if args.config is not None:
        # Child-process mode: run one config, emit result as marked JSON on stdout.
        result = run_single_config(args.config)
        print(RESULT_MARKER + json.dumps(result))
    else:
        # Parent/orchestrator mode: spawn one subprocess per config.
        run_benchmarks()
