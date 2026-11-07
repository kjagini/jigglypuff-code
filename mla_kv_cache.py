"""Multi-head latent attention (MLA) with a KV cache.

Keys and values for all heads come from one shared latent vector per token:

    c_t   = x_t @ W_dkv          (d_latent)      <- the only thing cached
    k_t,h = c_t @ W_uk[:, h]     (d_head)
    v_t,h = c_t @ W_uv[:, h]     (d_head)

The up-projections are not applied to the whole cache at every step. Instead:
  - the key projection is folded into the query:  score = (q_h @ W_uk_h^T) . c_t
  - the value projection is applied after mixing the latents:
        out_h = (sum_t p_t c_t) @ W_uv_h

These rearrangements are exact. MHA with K-weights W_dkv @ W_uk and V-weights
W_dkv @ W_uv gives the same output.

Omitted: DeepSeek's decoupled RoPE key. Without it, position information is
not added to the keys here.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def init_mla_cache() -> Dict[str, Optional[torch.Tensor]]:
    """Empty cache. The tensor is (batch, seq_cached, d_latent) once filled."""
    return {"c": None}


def multi_latent_attention_cached(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_dkv: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    cache: Dict[str, Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Causal multi-head latent attention for new tokens `x`, using and updating `cache`.

    Args:
        x: (batch, seq_new, d_model).
        w_q: (d_model, d_model) query projection.
        w_dkv: (d_model, d_latent) down-projection to the shared latent.
        w_uk, w_uv: (d_latent, d_model) up-projections to keys and values
            (num_heads heads of size d_head).
        w_o: (d_model, d_model) output projection.
        num_heads: number of heads; d_model must be divisible by it.
        cache: dict from init_mla_cache(); updated in place and returned.

    Returns:
        output: (batch, seq_new, d_model)
        cache:  the updated cache
    """
    batch, seq_new, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    d_head = d_model // num_heads
    d_latent = w_dkv.shape[1]

    q = (x @ w_q).view(batch, seq_new, num_heads, d_head).transpose(1, 2)  # (B, H, S, dh)
    c_new = x @ w_dkv                                                       # (B, S, dc)

    if cache["c"] is None:
        seq_past = 0
        c_all = c_new
    else:
        seq_past = cache["c"].shape[1]
        c_all = torch.cat([cache["c"], c_new], dim=1)
    cache["c"] = c_all

    seq_total = c_all.shape[1]
    w_uk_h = w_uk.view(d_latent, num_heads, d_head)  # (dc, H, dh)
    w_uv_h = w_uv.view(d_latent, num_heads, d_head)

    # Fold W_uk into the query: score_t = (q_h W_uk_h^T) . c_t
    q_latent = torch.einsum("bhsd,chd->bhsc", q, w_uk_h)                  # (B, H, S, dc)
    scores = torch.einsum("bhsc,btc->bhst", q_latent, c_all) / math.sqrt(d_head)

    q_pos = torch.arange(seq_past, seq_past + seq_new, device=x.device)
    k_pos = torch.arange(seq_total, device=x.device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~allowed, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    # Mix latents first, then apply W_uv per head.
    mixed = torch.einsum("bhst,btc->bhsc", weights, c_all)                # (B, H, S, dc)
    out_h = torch.einsum("bhsc,chd->bhsd", mixed, w_uv_h)                 # (B, H, S, dh)
    out = out_h.transpose(1, 2).reshape(batch, seq_new, d_model)
    return out @ w_o, cache
