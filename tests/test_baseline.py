import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.loader import load_pretrained_gpt2, get_device

def test_logits_parity():
    device = get_device()
    
    # Load custom and reference models
    custom_model = load_pretrained_gpt2()
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    custom_model.eval()
    ref_model.eval()
    
    # Sample input prompt
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "In the beginning, there was code."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits
        custom_logits, _ = custom_model(input_ids)
        
    diff = (ref_logits - custom_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check that they match to float precision (within 1e-3, typically < 2e-4)
    assert max_diff < 1e-3, f"Logits do not match reference model: max diff = {max_diff}"
    assert mean_diff < 1e-4, f"Logits mean difference is too high: mean diff = {mean_diff}"
