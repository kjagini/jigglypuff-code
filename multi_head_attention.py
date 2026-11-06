"""Multi-head attention in PyTorch, written as a plain function.

The function takes the projection weights explicitly so it is easy to use in
experiments (swap weights, change head counts, mask, etc.) without a module.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product multi-head attention.

    Args:
        query: (batch, seq_q, d_model)
        key:   (batch, seq_k, d_model)
        value: (batch, seq_k, d_model)
        w_q, w_k, w_v, w_o: (d_model, d_model) projection matrices
            (applied as ``x @ w``).
        num_heads: number of heads; d_model must be divisible by it.
        mask: optional bool/float mask broadcastable to
            (batch, num_heads, seq_q, seq_k). Bool: True = attend,
            False = blocked. Float: added to the scores.
        causal: if True, apply a lower-triangular causal mask.
        dropout_p: dropout probability on attention weights.
        training: whether dropout is active.

    Returns:
        output:  (batch, seq_q, d_model)
        weights: (batch, num_heads, seq_q, seq_k) attention probabilities
    """
    batch, seq_q, d_model = query.shape
    seq_k = key.shape[1]
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    d_head = d_model // num_heads

    # Project, then split into heads: (batch, heads, seq, d_head)
    def split_heads(x: torch.Tensor, seq: int) -> torch.Tensor:
        return x.view(batch, seq, num_heads, d_head).transpose(1, 2)

    q = split_heads(query @ w_q, seq_q)
    k = split_heads(key @ w_k, seq_k)
    v = split_heads(value @ w_v, seq_k)

    # Attention scores: (batch, heads, seq_q, seq_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)

    if causal:
        causal_mask = torch.tril(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    weights = F.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=training)

    # Combine heads: (batch, seq_q, d_model)
    out = (weights @ v).transpose(1, 2).reshape(batch, seq_q, d_model)
    return out @ w_o, weights


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, D, H = 2, 5, 16, 4
    x = torch.randn(B, S, D)
    ws = [torch.randn(D, D) / math.sqrt(D) for _ in range(4)]

    out, attn = multi_head_attention(x, x, x, *ws, num_heads=H, causal=True)
    print("output:", tuple(out.shape), "weights:", tuple(attn.shape))

    # Check against torch's built-in implementation.
    mha = torch.nn.MultiheadAttention(D, H, bias=False, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([ws[0].T, ws[1].T, ws[2].T], dim=0))
        mha.out_proj.weight.copy_(ws[3].T)
    ref_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    ref, ref_w = mha(x, x, x, attn_mask=ref_mask, average_attn_weights=False)
    print("matches nn.MultiheadAttention:",
          torch.allclose(out, ref, atol=1e-5) and torch.allclose(attn, ref_w, atol=1e-5))
