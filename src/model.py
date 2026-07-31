import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max sequence length (Context Window)
    vocab_size: int = 50304  # Padded from 50,257 to the nearest multiple of 64 for optimal GPU Tensor Core alignment
    n_layer: int = 12  # Number of Transformer blocks
    n_head: int = 12  # Number of Attention heads
    n_embd: int = 768  # Embedding dimension (768 / 12 = 64 dimensions per head)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections combined into one single linear layer for speed
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Flag for scaled initialization (1/sqrt(2L) variance scaling)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Time (Sequence Length), Channels (Embedding Dim)

        # Calculate Query, Key, Values for all heads in batch simultaneously
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # HARDWARE OPTIMIZATION: FlashAttention (Optimized CUDA kernel execution)
        # Avoids materializing the massive (T, T) attention matrix in high-bandwidth memory (HBM)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Feed-forward network expands dimensionality by 4x
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)

        # GELU activation using the Tanh approximation (matches original GPT-2 exactly)
        self.gelu = nn.GELU(approximate='tanh')

        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pre-LayerNorm architecture (Norm is applied BEFORE attention and MLP)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connections prevent vanishing gradients in deep networks
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The core Transformer module dictionary
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        # Language Model Head (Classification head to predict next token)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # ARCHITECTURAL OPTIMIZATION: Weight Tying
        # The embedding table and the output projection layer share the exact same weights in memory.
        # This saves ~30% of total model parameters (38.6M params).
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights with standard normal distribution and scales residuals."""
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scaled initialization for residual projection layers to prevent variance explosion
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds max context length {self.config.block_size}"

        # Generate Position indices: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Compute embeddings
        tok_emb = self.transformer.wte(idx)  # Token embeddings (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings (T, n_embd)

        # Add position embeddings to token embeddings
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm before prediction head
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training Mode: Calculate loss over all tokens
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # Inference Mode Optimization: We only care about predicting the very last token
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        Custom optimizer configuration to separate parameters that should be decayed (weights)
        from parameters that shouldn't (biases, layernorms).
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # 2D Parameters (Linear weights, Embeddings) get weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]

        # 1D Parameters (Biases, LayerNorm scales) do NOT get weight decay
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Use Fused AdamW if executing on CUDA (massively speeds up parameter updates)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
