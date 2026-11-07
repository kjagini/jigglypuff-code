"""Grouped-query attention (GQA) with a KV cache.

Query heads are split into `num_kv_heads` groups. Each group shares one key
head and one value head. With num_kv_heads == num_heads this is MHA, and with
num_kv_heads == 1 it is MQA.

Query head h uses key/value head h // (num_heads // num_kv_heads).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def init_gqa_cache() -> Dict[str, Optional[torch.Tensor]]:
    """Empty cache. Tensors are (batch, num_kv_heads, seq_cached, d_head) once filled."""
    return {"k": None, "v": None}


def grouped_query_attention_cached(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    cache: Dict[str, Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Causal grouped-query attention for new tokens `x`, using and updating `cache`.

    Args:
        x: (batch, seq_new, d_model).
        w_q: (d_model, d_model) query projection (num_heads query heads).
        w_k, w_v: (d_model, num_kv_heads * d_head) key/value projections.
        w_o: (d_model, d_model) output projection.
        num_heads: number of query heads.
        num_kv_heads: number of key/value heads; must divide num_heads.
        cache: dict from init_gqa_cache(); updated in place and returned.

    Returns:
        output: (batch, seq_new, d_model)
        cache:  the updated cache
    """
    batch, seq_new, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    d_head = d_model // num_heads
    group = num_heads // num_kv_heads  # query heads per key/value head

    # Queries: (B, G, R, S, dh), where R = query heads per group.
    q = (x @ w_q).view(batch, seq_new, num_kv_heads, group, d_head).permute(0, 2, 3, 1, 4)
    # Keys and values: (B, G, S, dh)
    k_new = (x @ w_k).view(batch, seq_new, num_kv_heads, d_head).transpose(1, 2)
    v_new = (x @ w_v).view(batch, seq_new, num_kv_heads, d_head).transpose(1, 2)

    if cache["k"] is None:
        seq_past = 0
        k_all, v_all = k_new, v_new
    else:
        seq_past = cache["k"].shape[2]
        k_all = torch.cat([cache["k"], k_new], dim=2)
        v_all = torch.cat([cache["v"], v_new], dim=2)
    cache["k"], cache["v"] = k_all, v_all

    seq_total = k_all.shape[2]
    # Each group's keys are broadcast across its R query heads.
    scores = q @ k_all.unsqueeze(2).transpose(-2, -1) / math.sqrt(d_head)  # (B, G, R, S, T)

    q_pos = torch.arange(seq_past, seq_past + seq_new, device=x.device)
    k_pos = torch.arange(seq_total, device=x.device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~allowed, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = weights @ v_all.unsqueeze(2)  # (B, G, R, S, dh)
    out = out.reshape(batch, num_heads, seq_new, d_head).transpose(1, 2).reshape(batch, seq_new, d_model)
    return out @ w_o, cache
