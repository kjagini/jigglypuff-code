"""
Analytical cost model (per layer): FLOPs, bytes moved, arithmetic intensity, roofline time.

A "phase" is described by
    B  batch size
    T  number of NEW tokens processed this step (prefill: T = prompt length, decode: T = 1)
    P  tokens already in the KV cache (prefill: 0, decode: context length)

FLOPs = 2 * multiply-adds.  Bytes = weights read once + KV cache read + KV cache written + in/out activations.
Attention score matrices are assumed to stay on-chip (FlashAttention style), i.e. S x S never hits HBM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from attention import Config

# Illustrative accelerator numbers (H100 SXM, dense bf16). Change freely; only ratios matter.
PEAK_FLOPS = 989e12
HBM_BW = 3.35e12
RIDGE = PEAK_FLOPS / HBM_BW          # FLOP/byte needed to be compute-bound  (~295)

# A "7B-class" configuration used for the analytical plots (not for the CPU timings).
BIG = Config(d_model=4096, n_heads=32, head_dim=128, n_kv_heads=8,
             kv_lora_rank=512, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128,
             d_ff=2048, n_experts=64, top_k=4)
BIG_DENSE_FF = 11008                 # dense SwiGLU hidden size to compare against MoE
N_LAYERS = 32


@dataclass
class Cost:
    flops: float
    weight_bytes: float
    kv_read_bytes: float
    kv_write_bytes: float
    act_bytes: float

    @property
    def bytes(self):
        return self.weight_bytes + self.kv_read_bytes + self.kv_write_bytes + self.act_bytes

    @property
    def intensity(self):
        return self.flops / self.bytes

    def time(self, peak=PEAK_FLOPS, bw=HBM_BW):
        return max(self.flops / peak, self.bytes / bw)

    @property
    def bound(self):
        return "compute" if self.flops / PEAK_FLOPS > self.bytes / HBM_BW else "memory"


def kv_per_token(name: str, cfg: Config) -> int:
    """number of cached scalars per token per layer"""
    return {"MHA": 2 * cfg.n_heads * cfg.head_dim,
            "GQA": 2 * cfg.n_kv_heads * cfg.head_dim,
            "MQA": 2 * cfg.head_dim,
            "MLA": cfg.kv_lora_rank + cfg.qk_rope_dim}[name]


def attention_cost(name, cfg: Config, B, T, P, bytes_per=2, causal_skip=True, mla_absorbed=None) -> Cost:
    """Cost of one attention layer. `causal_skip=False` mimics a dense (un-skipped) mask like our torch code."""
    d, H = cfg.d_model, cfg.n_heads
    pairs = T * P + (T * (T + 1) / 2 if causal_skip else T * T)      # (query,key) pairs actually scored
    if mla_absorbed is None:
        mla_absorbed = (T == 1)                                       # decode -> absorbed, prefill -> naive
    kvt = kv_per_token(name, cfg)

    if name in ("MHA", "GQA", "MQA"):
        G = {"MHA": H, "GQA": cfg.n_kv_heads, "MQA": 1}[name]
        dh = cfg.head_dim
        params = d * H * dh + 2 * d * G * dh + H * dh * d
        flops = 2 * B * T * params + 4 * B * H * pairs * dh
    else:  # MLA
        r, dn, dr, dv = cfg.kv_lora_rank, cfg.qk_nope_dim, cfg.qk_rope_dim, cfg.v_head_dim
        p_fixed = d * H * (dn + dr) + d * r + d * dr + H * dv * d          # wq, w_dkv, w_kr, wo
        params = p_fixed + r * H * dn + r * H * dv                          # + W_uk, W_uv
        flops = 2 * B * T * p_fixed
        if mla_absorbed:       # attention in latent space (MQA-like, width r+dr / r)
            flops += 2 * B * T * H * dn * r                                 # q_nope @ W_uk
            flops += 2 * B * H * pairs * (r + dr) + 2 * B * H * pairs * r   # scores + P@c
            flops += 2 * B * T * H * r * dv                                 # o_lat @ W_uv
        else:                  # decompress to full K,V, then ordinary MHA
            flops += 2 * B * (T + P) * r * H * (dn + dv)
            flops += 2 * B * H * pairs * (dn + dr) + 2 * B * H * pairs * dv
    return Cost(flops=flops,
                weight_bytes=params * bytes_per,
                kv_read_bytes=B * P * kvt * bytes_per,
                kv_write_bytes=B * T * kvt * bytes_per,
                act_bytes=2 * B * T * d * bytes_per)


def expected_active_experts(E, k, tokens):
    """E[#distinct experts hit] if every token picks k distinct experts uniformly at random."""
    return E * (1 - (1 - k / E) ** tokens)


def ffn_cost(kind, cfg: Config, B, T, bytes_per=2, dense_ff=BIG_DENSE_FF, n_active=None) -> Cost:
    """kind: 'dense' | 'moe'.  SwiGLU = 3 matrices of d x dff."""
    d, tokens = cfg.d_model, B * T
    if kind == "dense":
        w = 3 * d * dense_ff
        flops = 2 * tokens * w
    else:
        per_expert = 3 * d * cfg.d_ff
        flops = 2 * tokens * cfg.top_k * per_expert + 2 * tokens * d * cfg.n_experts   # experts + router
        if n_active is None:
            n_active = expected_active_experts(cfg.n_experts, cfg.top_k, tokens)
        w = n_active * per_expert + d * cfg.n_experts
    return Cost(flops=flops, weight_bytes=w * bytes_per, kv_read_bytes=0, kv_write_bytes=0,
                act_bytes=2 * tokens * d * bytes_per)
