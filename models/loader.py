import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM

def get_device() -> torch.device:
    """
    Get the optimal available hardware device.
    
    Returns:
        torch.device: Centralized device dispatcher.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    
    Args:
        x: [batch, seq_len, dim] or [batch, seq_len, 4 * dim], float32.
        
    Returns:
        [batch, seq_len, dim] or [batch, seq_len, 4 * dim], float32.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPT2Attention(nn.Module):
    def __init__(self, n_embd: int = 768, n_head: int = 12):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.o_proj = nn.Linear(n_embd, n_embd)

    def forward(
        self, 
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head causal self-attention.
        
        Args:
            x: [batch, seq_len, n_embd], float32.
            past_key_value: Optional tuple of (past_k, past_v) where
                past_k is [batch, n_head, past_seq_len, head_dim], float32
                past_v is [batch, n_head, past_seq_len, head_dim], float32.
                
        Returns:
            Tuple containing:
                - output: [batch, seq_len, n_embd], float32.
                - present_key_value: Tuple of (present_k, present_v), float32.
        """
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            from cache.kv_cache import append_to_kv_cache
            k = append_to_kv_cache(past_k, k)
            v = append_to_kv_cache(past_v, v)
            
        present_key_value = (k, v)
        
        # k shape: [batch, n_head, total_seq_len, head_dim]
        # q shape: [batch, n_head, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        # Apply causal mask only for the parts of the attention matrix that are computed
        # scores shape: [batch, n_head, seq_len, total_seq_len]
        total_seq_len = k.shape[2]
        
        # Create a causal mask for the query sequence length vs total key sequence length
        # The query token at position 'i' can only look at key tokens at positions '<= i' (plus past tokens)
        # So we mask out the upper triangle of the last seq_len x seq_len block.
        mask = torch.ones(seq_len, total_seq_len, dtype=torch.bool, device=x.device)
        # Causal constraint: query index q_idx can only attend to key index k_idx if k_idx <= q_idx + past_seq_len
        past_seq_len = total_seq_len - seq_len
        q_idx = torch.arange(seq_len, device=x.device).view(-1, 1)
        k_idx = torch.arange(total_seq_len, device=x.device).view(1, -1)
        mask = k_idx <= (q_idx + past_seq_len)
        
        mask = mask.view(1, 1, seq_len, total_seq_len)
        scores = scores.masked_fill(~mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_embd)
        return self.o_proj(attn_out), present_key_value

class GPT2MLP(nn.Module):
    def __init__(self, n_embd: int = 768):
        super().__init__()
        self.fc_in = nn.Linear(n_embd, 4 * n_embd)
        self.fc_out = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feed-forward network output with GELU activation.
        
        Args:
            x: [batch, seq_len, n_embd], float32.
            
        Returns:
            [batch, seq_len, n_embd], float32.
        """
        return self.fc_out(gelu_new(self.fc_in(x)))

class GPT2Block(nn.Module):
    def __init__(self, n_embd: int = 768, n_head: int = 12, ln_eps: float = 1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.attn = GPT2Attention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.mlp = GPT2MLP(n_embd)

    def forward(
        self, 
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for one Transformer block.
        
        Args:
            x: [batch, seq_len, n_embd], float32.
            past_key_value: Optional tuple of (past_k, past_v), float32.
            
        Returns:
            Tuple containing:
                - output: [batch, seq_len, n_embd], float32.
                - present_key_value: Tuple of (present_k, present_v), float32.
        """
        attn_out, present_kv = self.attn(self.ln_1(x), past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv

class GPT2LMHeadModel(nn.Module):
    def __init__(
        self, 
        n_embd: int = 768, 
        n_head: int = 12, 
        vocab_size: int = 50257, 
        n_layer: int = 12, 
        n_positions: int = 1024, 
        ln_eps: float = 1e-5
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList([GPT2Block(n_embd, n_head, ln_eps) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=ln_eps)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self, 
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the full GPT-2 model.
        
        Args:
            input_ids: [batch, seq_len], int64.
            past_key_values: Optional list of tuples (past_k, past_v) for each layer, float32.
            
        Returns:
            Tuple containing:
                - logits: [batch, seq_len, vocab_size], float32.
                - present_key_values: List of tuples (present_k, present_v) for each layer, float32.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Determine sequence position offsets
        if past_key_values is not None:
            # past_key_values[0][0] shape is [batch, n_head, past_seq_len, head_dim]
            past_seq_len = past_key_values[0][0].shape[2]
        else:
            past_seq_len = 0
            
        pos = torch.arange(past_seq_len, past_seq_len + seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        x = self.wte(input_ids) + self.wpe(pos)
        
        present_key_values = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_key_value=layer_past)
            present_key_values.append(present_kv)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, present_key_values

def load_pretrained_gpt2() -> GPT2LMHeadModel:
    """
    Load pre-trained weights from Hugging Face GPT-2 and map them to our custom architecture.
    
    Returns:
        GPT2LMHeadModel: Fully populated custom model on optimal device.
    """
    device = get_device()
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model.eval()
    
    # Read config parameters
    config = ref_model.config
    custom_model = GPT2LMHeadModel(
        n_embd=config.n_embd,
        n_head=config.n_head,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_positions=config.n_positions,
        ln_eps=config.layer_norm_epsilon
    )
    custom_model.eval()
    
    # Load weights
    with torch.no_grad():
        custom_model.wte.weight.copy_(ref_model.transformer.wte.weight)
        custom_model.wpe.weight.copy_(ref_model.transformer.wpe.weight)
        custom_model.ln_f.weight.copy_(ref_model.transformer.ln_f.weight)
        custom_model.ln_f.bias.copy_(ref_model.transformer.ln_f.bias)
        custom_model.lm_head.weight.copy_(ref_model.lm_head.weight)
        
        for i, block in enumerate(custom_model.blocks):
            ref_block = ref_model.transformer.h[i]
            block.ln_1.weight.copy_(ref_block.ln_1.weight)
            block.ln_1.bias.copy_(ref_block.ln_1.bias)
            block.ln_2.weight.copy_(ref_block.ln_2.weight)
            block.ln_2.bias.copy_(ref_block.ln_2.bias)
            
            # Split c_attn weight and bias (shape [n_embd, 3 * n_embd])
            q_w, k_w, v_w = ref_block.attn.c_attn.weight.chunk(3, dim=1)
            q_b, k_b, v_b = ref_block.attn.c_attn.bias.chunk(3, dim=0)
            
            block.attn.q_proj.weight.copy_(q_w.t())
            block.attn.q_proj.bias.copy_(q_b)
            block.attn.k_proj.weight.copy_(k_w.t())
            block.attn.k_proj.bias.copy_(k_b)
            block.attn.v_proj.weight.copy_(v_w.t())
            block.attn.v_proj.bias.copy_(v_b)
            
            # c_proj weight is [n_embd, n_embd]
            block.attn.o_proj.weight.copy_(ref_block.attn.c_proj.weight.t())
            block.attn.o_proj.bias.copy_(ref_block.attn.c_proj.bias)
            
            # MLP fc weight is [n_embd, 4 * n_embd]
            block.mlp.fc_in.weight.copy_(ref_block.mlp.c_fc.weight.t())
            block.mlp.fc_in.bias.copy_(ref_block.mlp.c_fc.bias)
            
            # MLP proj weight is [4 * n_embd, n_embd]
            block.mlp.fc_out.weight.copy_(ref_block.mlp.c_proj.weight.t())
            block.mlp.fc_out.bias.copy_(ref_block.mlp.c_proj.bias)
            
    custom_model.to(device)
    return custom_model
