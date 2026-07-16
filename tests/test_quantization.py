import torch
import pytest
from quantization.quantizer import (
    compute_scales_and_zero_points,
    quantize_tensor,
    dequantize_tensor
)

def test_quantization_parity():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create a random weight tensor of shape representing typical GPT-2 layers
    # e.g., [768, 768] (o_proj) or [3072, 768] (fc_in)
    shapes = [(768, 768), (3072, 768), (768, 2304)]
    
    for shape in shapes:
        # Generate random weights from normal distribution (representing model weights)
        W = torch.randn(shape, dtype=torch.float32)
        
        # Test per-channel quantization
        scale_c, zero_point_c = compute_scales_and_zero_points(W, per_channel=True)
        W_q_c = quantize_tensor(W, scale_c, zero_point_c)
        W_dequant_c = dequantize_tensor(W_q_c, scale_c, zero_point_c)
        
        mse_c = torch.mean((W - W_dequant_c) ** 2).item()
        print(f"Per-channel quantization MSE for shape {shape}: {mse_c:.6f}")
        
        # Test per-tensor quantization
        scale_t, zero_point_t = compute_scales_and_zero_points(W, per_channel=False)
        W_q_t = quantize_tensor(W, scale_t, zero_point_t)
        W_dequant_t = dequantize_tensor(W_q_t, scale_t, zero_point_t)
        
        mse_t = torch.mean((W - W_dequant_t) ** 2).item()
        print(f"Per-tensor quantization MSE for shape {shape}: {mse_t:.6f}")
        
        # Check exit criteria: MSE must be < 1e-3
        assert mse_c < 1e-3, f"Per-channel quantization MSE exceeds threshold: {mse_c}"
        assert mse_t < 1e-3, f"Per-tensor quantization MSE exceeds threshold: {mse_t}"
        
        # Ensure quantized tensor only has values in range [-128, 127]
        assert W_q_c.min() >= -128, f"Quantized weight out of bounds (min = {W_q_c.min()})"
        assert W_q_c.max() <= 127, f"Quantized weight out of bounds (max = {W_q_c.max()})"
        assert W_q_t.min() >= -128, f"Quantized weight out of bounds (min = {W_q_t.min()})"
        assert W_q_t.max() <= 127, f"Quantized weight out of bounds (max = {W_q_t.max()})"
        
        # Ensure the dtype is torch.int8
        assert W_q_c.dtype == torch.int8
        assert W_q_t.dtype == torch.int8
