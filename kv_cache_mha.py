"""Multi-head attention with a KV cache.

Only the new tokens are projected each call. Keys and values for earlier
tokens are read from the cache, so a decode step costs one new query against
the stored keys instead of recomputing attention over the whole sequence.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def init_kv_cache() -> Dict[str, Optional[torch.Tensor]]:
    """Empty cache. Tensors are (batch, heads, seq_cached, d_head) once filled."""
    return {"k": None, "v": None}


def multi_head_attention_cached(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    cache: Dict[str, Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    """Causal attention for new tokens `x`, using and updating `cache`.

    Args:
        x: (batch, seq_new, d_model). seq_new is the prompt length for prefill
            and 1 for each decode step.
        w_q, w_k, w_v, w_o: (d_model, d_model) projection matrices.
        num_heads: number of heads; d_model must be divisible by it.
        cache: dict from init_kv_cache(); updated in place and returned.

    Returns:
        output: (batch, seq_new, d_model)
        cache:  the updated cache
    """
    batch, seq_new, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    d_head = d_model // num_heads

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        return t.view(batch, -1, num_heads, d_head).transpose(1, 2)

    q = split_heads(x @ w_q)  # (batch, heads, seq_new, d_head)
    k_new = split_heads(x @ w_k)
    v_new = split_heads(x @ w_v)

    if cache["k"] is None:
        seq_past = 0
        k_all, v_all = k_new, v_new
    else:
        seq_past = cache["k"].shape[2]
        k_all = torch.cat([cache["k"], k_new], dim=2)
        v_all = torch.cat([cache["v"], v_new], dim=2)
    cache["k"], cache["v"] = k_all, v_all

    seq_total = k_all.shape[2]
    scores = q @ k_all.transpose(-2, -1) / math.sqrt(d_head)  # (batch, heads, seq_new, seq_total)

    # Query at absolute position seq_past + i may attend to keys at positions <= that.
    q_pos = torch.arange(seq_past, seq_past + seq_new, device=x.device)
    k_pos = torch.arange(seq_total, device=x.device)
    allowed = k_pos[None, :] <= q_pos[:, None]  # (seq_new, seq_total)
    scores = scores.masked_fill(~allowed, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    out = (weights @ v_all).transpose(1, 2).reshape(batch, seq_new, d_model)
    return out @ w_o, cache
