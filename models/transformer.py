
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. RMS Normalization
# --------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        return x * rms * self.weight


# --------------------------------------------------
# 2. Causal Self-Attention
# --------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.0):

        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        B, T, C = x.shape

        # Generate query, key and value
        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B, T, 3, self.num_heads, self.head_dim
        )

        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # Causal attention prevents looking at future tokens
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Combine attention heads
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, T, C)

        return self.proj(out)


# --------------------------------------------------
# 3. SwiGLU Feed Forward Network
# --------------------------------------------------

class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim):

        super().__init__()

        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):

        gate = F.silu(self.gate(x))
        up = self.up(x)

        return self.down(gate * up)


# --------------------------------------------------
# 4. Transformer Block
# --------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        dropout=0.0,
    ):

        super().__init__()

        self.attention_norm = RMSNorm(dim)
        self.attention = CausalSelfAttention(
            dim,
            num_heads,
            dropout,
        )

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x):

        # Attention residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # Feed-forward residual connection
        x = x + self.ffn(
            self.ffn_norm(x)
        )

        return x
