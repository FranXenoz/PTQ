import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_scales_and_zero_points(
    tensor: torch.Tensor,
    per_channel: bool = True,
    q_min: int = -128,
    q_max: int = 127
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scale and zero-point parameters for asymmetric INT8 quantization.

    Args:
        tensor:      [out_features, in_features] or any shape, float32 tensor to quantize.
        per_channel: If True, compute parameters per row (dim 0). If False, compute per-tensor.
        q_min:       Minimum quantization integer value (-128).
        q_max:       Maximum quantization integer value (127).

    Returns:
        Tuple containing:
            - scale: [out_features, 1] (per-channel) or [1] (per-tensor), float32.
            - zero_point: [out_features, 1] (per-channel) or [1] (per-tensor), float32.
    """
    if per_channel:
        # Compute min/max along the input features dimension (dim -1)
        min_val = tensor.min(dim=-1, keepdim=True).values
        max_val = tensor.max(dim=-1, keepdim=True).values
    else:
        min_val = tensor.min()
        max_val = tensor.max()

    scale = (max_val - min_val) / (q_max - q_min)
    # Clamp scale to prevent division by zero
    scale = torch.clamp(scale, min=1e-8)
    
    zero_point = torch.round(-min_val / scale) + q_min
    zero_point = torch.clamp(zero_point, q_min, q_max)
    
    return scale, zero_point

def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: int = -128,
    q_max: int = 127
) -> torch.Tensor:
    """
    Quantize a float32 tensor to int8 using the provided scale and zero-point.

    Args:
        tensor:      Float32 input tensor (e.g. [out_features, in_features]).
        scale:       Float32 scale tensor.
        zero_point:  Float32 zero-point tensor.
        q_min:       Minimum value of int8 (-128).
        q_max:       Maximum value of int8 (127).

    Returns:
        Int8 quantized tensor of the same shape as the input tensor.
    """
    q_tensor = torch.round(tensor / scale) + zero_point
    q_tensor = torch.clamp(q_tensor, q_min, q_max)
    return q_tensor.to(torch.int8)

def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize an int8 tensor back to float32.

    Args:
        q_tensor:   Int8 quantized tensor.
        scale:      Float32 scale tensor.
        zero_point: Float32 zero-point tensor.

    Returns:
        Float32 reconstructed tensor of the same shape.
    """
    return scale * (q_tensor.to(torch.float32) - zero_point)

class INT8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Buffers for quantized weights, scales, zero points (not parameters)
        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        self.register_buffer("zero_point", torch.zeros((out_features, 1), dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, per_channel: bool = True) -> "INT8Linear":
        """
        Create an INT8Linear module from a standard nn.Linear module.

        Args:
            float_linear: Standard nn.Linear module.
            per_channel:  Whether to use per-channel or per-tensor quantization.

        Returns:
            INT8Linear: Quantized version of the linear layer.
        """
        has_bias = float_linear.bias is not None
        int8_linear = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=has_bias
        )
        # Move to same device as the float layer
        device = float_linear.weight.device
        int8_linear.to(device)
        
        # Compute quantization parameters for the weight matrix
        W = float_linear.weight.data
        scale, zero_point = compute_scales_and_zero_points(W, per_channel=per_channel)
        W_q = quantize_tensor(W, scale, zero_point)
        
        # Load parameters and buffers
        int8_linear.weight_q.copy_(W_q)
        int8_linear.scale.copy_(scale)
        int8_linear.zero_point.copy_(zero_point)
        
        if has_bias:
            int8_linear.bias.data.copy_(float_linear.bias.data)
            
        return int8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantizes weight on-the-fly and performs linear projection.

        Args:
            x: [batch, seq_len, in_features], float32.

        Returns:
            [batch, seq_len, out_features], float32.
        """
        # Dequantize weight on the fly to float32
        # self.weight_q: [out_features, in_features]
        # self.scale:    [out_features, 1] or [1]
        # self.zero_point: [out_features, 1] or [1]
        weight_dequant = dequantize_tensor(self.weight_q, self.scale, self.zero_point)
        
        return F.linear(x, weight_dequant, self.bias)

def quantize_model(model: nn.Module, per_channel: bool = True) -> nn.Module:
    """
    Wrap all designated Linear layers in the GPT-2 model in our custom INT8Linear module.

    Args:
        model:       Custom GPT-2 model (GPT2LMHeadModel).
        per_channel: Whether to quantize weights per-channel or per-tensor.

    Returns:
        nn.Module: Model with quantized linear layers.
    """
    # Specifically target the 6 linear layers in each transformer block:
    # attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj, mlp.fc_in, mlp.fc_out
    for i, block in enumerate(model.blocks):
        block.attn.q_proj = INT8Linear.from_float(block.attn.q_proj, per_channel=per_channel)
        block.attn.k_proj = INT8Linear.from_float(block.attn.k_proj, per_channel=per_channel)
        block.attn.v_proj = INT8Linear.from_float(block.attn.v_proj, per_channel=per_channel)
        block.attn.o_proj = INT8Linear.from_float(block.attn.o_proj, per_channel=per_channel)
        block.mlp.fc_in = INT8Linear.from_float(block.mlp.fc_in, per_channel=per_channel)
        block.mlp.fc_out = INT8Linear.from_float(block.mlp.fc_out, per_channel=per_channel)
        
    return model
