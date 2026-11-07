"""Multi-query attention (MQA) with a KV cache.

All query heads share one key head and one value head. Queries are projected
to d_model as usual, but keys and values are projected only to d_head. The
cache therefore stores 1 head of keys and values instead of num_heads.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def init_mqa_cache() -> Dict[str, Optional[torch.Tensor]]:
    """Empty cache. Tensors are (batch, 1, seq_cached, d_head) once filled."""
    return {"k": None, "v": None}


def multi_query_attention_cached(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    cache: Dict[str, Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Causal multi-query attention for new tokens `x`, using and updating `cache`.

    Args:
        x: (batch, seq_new, d_model).
        w_q: (d_model, d_model) query projection (num_heads query heads).
        w_k, w_v: (d_model, d_head) shared key/value projections (one head).
        w_o: (d_model, d_model) output projection.
        num_heads: number of query heads; d_model must be divisible by it.
        cache: dict from init_mqa_cache(); updated in place and returned.

    Returns:
        output: (batch, seq_new, d_model)
        cache:  the updated cache
    """
    batch, seq_new, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    d_head = d_model // num_heads

    q = (x @ w_q).view(batch, seq_new, num_heads, d_head).transpose(1, 2)  # (B, H, S, dh)
    k_new = (x @ w_k).view(batch, seq_new, 1, d_head).transpose(1, 2)      # (B, 1, S, dh)
    v_new = (x @ w_v).view(batch, seq_new, 1, d_head).transpose(1, 2)      # (B, 1, S, dh)

    if cache["k"] is None:
        seq_past = 0
        k_all, v_all = k_new, v_new
    else:
        seq_past = cache["k"].shape[2]
        k_all = torch.cat([cache["k"], k_new], dim=2)
        v_all = torch.cat([cache["v"], v_new], dim=2)
    cache["k"], cache["v"] = k_all, v_all

    seq_total = k_all.shape[2]
    # The single K/V head broadcasts across all query heads.
    scores = q @ k_all.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, S, T)

    q_pos = torch.arange(seq_past, seq_past + seq_new, device=x.device)
    k_pos = torch.arange(seq_total, device=x.device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~allowed, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = (weights @ v_all).transpose(1, 2).reshape(batch, seq_new, d_model)
    return out @ w_o, cache
