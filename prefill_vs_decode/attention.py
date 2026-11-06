"""
From-scratch attention variants (MHA, MQA, GQA, MLA) and a MoE feed-forward block.

No pretrained model, no HF, no flash-attn: just torch tensors and matmuls, so every
FLOP / byte can be accounted for. Weights are random -- prefill-vs-decode behaviour
is a property of the *shapes*, not of trained values.

Every module exposes the same two entry points:

    out, cache = block.prefill(x)               # x: [B, S, d_model]   -> whole prompt at once
    out, cache = block.decode(x_t, cache)       # x_t: [B, 1, d_model] -> one new token

and `forward_full(x)` (no cache) used by the tests to prove that
prefill + N x decode == one big causal forward pass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    d_model: int = 1024
    n_heads: int = 16
    head_dim: int = 64
    n_kv_heads: int = 4          # GQA groups (MQA = 1, MHA = n_heads)
    # --- MLA (DeepSeek-V2 style) ---
    kv_lora_rank: int = 128      # r : size of the cached latent c_kv
    qk_nope_dim: int = 64        # per-head "content" part of q/k
    qk_rope_dim: int = 32        # per-head rotary part (k_rope is SHARED across heads)
    v_head_dim: int = 64
    # --- MoE FFN ---
    d_ff: int = 2048             # hidden dim of ONE expert (and of the dense FFN)
    n_experts: int = 16
    top_k: int = 2
    rope_theta: float = 10000.0


# --------------------------------------------------------------------------- RoPE
def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float):
    """positions: [S] (absolute positions). returns cos,sin: [S, dim/2]"""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = positions.float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float):
    """x: [..., S, dim] with S at dim -2. Rotates pairs (x0,x1),(x2,x3),..."""
    cos, sin = rope_cos_sin(positions, x.shape[-1], theta)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def causal_sdpa(q, k, v, q_offset: int, scale: float):
    """Plain softmax(QK^T * scale + causal mask) V.
    q: [B,H,Sq,dk]  k: [B,H,Sk,dk]  v: [B,H,Sk,dv].  Query i sits at absolute pos q_offset+i.
    Prefill: Sq == Sk (square, lower-triangular).  Decode: Sq == 1 (one row, no masking needed).
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale          # [B,H,Sq,Sk]
    Sq, Sk = scores.shape[-2:]
    if Sq > 1:
        qi = torch.arange(Sq)[:, None] + q_offset
        kj = torch.arange(Sk)[None, :]
        scores = scores.masked_fill(kj > qi, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


# --------------------------------------------------------------------------- MHA / MQA / GQA
class GroupedAttention(nn.Module):
    """One class for MHA (n_kv == n_heads), GQA (1 < n_kv < n_heads) and MQA (n_kv == 1).

    The ONLY thing that changes between them is how many K/V heads exist -> how big the KV cache is.
    """

    def __init__(self, cfg: Config, n_kv_heads: int):
        super().__init__()
        assert cfg.n_heads % n_kv_heads == 0
        self.cfg, self.H, self.G, self.dh = cfg, cfg.n_heads, n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.H * self.dh, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.G * self.dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.G * self.dh, bias=False)
        self.wo = nn.Linear(self.H * self.dh, cfg.d_model, bias=False)

    def _qkv(self, x, pos):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.H, self.dh).transpose(1, 2)   # [B,H,S,dh]
        k = self.wk(x).view(B, S, self.G, self.dh).transpose(1, 2)   # [B,G,S,dh]
        v = self.wv(x).view(B, S, self.G, self.dh).transpose(1, 2)
        return apply_rope(q, pos, self.cfg.rope_theta), apply_rope(k, pos, self.cfg.rope_theta), v

    def _attend(self, q, k, v, offset):
        rep = self.H // self.G
        # expand_as view: each KV head is shared by `rep` query heads (no extra memory in a fused kernel;
        # here repeat_interleave materialises it, which is fine for correctness/timing at small scale)
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = causal_sdpa(q, k, v, offset, 1.0 / math.sqrt(self.dh))   # [B,H,Sq,dh]
        B, _, Sq, _ = o.shape
        return self.wo(o.transpose(1, 2).reshape(B, Sq, self.H * self.dh))

    def forward_full(self, x):
        pos = torch.arange(x.shape[1])
        q, k, v = self._qkv(x, pos)
        return self._attend(q, k, v, 0)

    def prefill(self, x):
        out = self.forward_full(x)
        pos = torch.arange(x.shape[1])
        _, k, v = self._qkv(x, pos)
        return out, {"k": k, "v": v}           # cache: 2 * G * dh numbers per token

    def decode(self, x_t, cache):
        L = cache["k"].shape[2]
        q, k, v = self._qkv(x_t, torch.tensor([L]))
        k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)   # append to cache
        return self._attend(q, k, v, L), {"k": k, "v": v}


# --------------------------------------------------------------------------- MLA
class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2 style MLA (query is a plain projection; q-LoRA omitted for clarity).

        c_kv   = W_dkv x                     [r]        <- cached
        k_rope = RoPE(W_kr x)                [d_r]      <- cached, shared by all heads
        k_nope = W_uk c_kv   (per head)      [H, d_nope]  (recomputed, NOT cached)
        v      = W_uv c_kv   (per head)      [H, d_v]     (recomputed, NOT cached)
        score  = q_nope . k_nope + q_rope . k_rope

    Two mathematically identical ways to run it:
      naive    : decompress c_kv -> full K,V for every cached token, then ordinary MHA.
      absorbed : fold W_uk into the query and W_uv into the output projection, so attention runs
                 directly on the compressed latents (like MQA with ONE kv head of width r + d_r).
    Prefill is compute-bound so naive is fine; decode is memory-bound so `absorbed` wins.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg, self.H = cfg, cfg.n_heads
        self.dn, self.dr, self.dv, self.r = cfg.qk_nope_dim, cfg.qk_rope_dim, cfg.v_head_dim, cfg.kv_lora_rank
        self.wq = nn.Linear(cfg.d_model, self.H * (self.dn + self.dr), bias=False)
        self.w_dkv = nn.Linear(cfg.d_model, self.r, bias=False)
        self.w_kr = nn.Linear(cfg.d_model, self.dr, bias=False)
        self.w_uk = nn.Linear(self.r, self.H * self.dn, bias=False)
        self.w_uv = nn.Linear(self.r, self.H * self.dv, bias=False)
        self.wo = nn.Linear(self.H * self.dv, cfg.d_model, bias=False)
        self.scale = 1.0 / math.sqrt(self.dn + self.dr)

    def _q(self, x, pos):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.H, self.dn + self.dr).transpose(1, 2)    # [B,H,S,dn+dr]
        q_nope, q_rope = q[..., : self.dn], apply_rope(q[..., self.dn:], pos, self.cfg.rope_theta)
        return q_nope, q_rope

    def _latent(self, x, pos):
        c = self.w_dkv(x)                                                        # [B,S,r]
        kr = apply_rope(self.w_kr(x)[:, None], pos, self.cfg.rope_theta)[:, 0]   # [B,S,dr]
        return c, kr

    # ---- naive: decompress then run regular MHA ----
    def _attend_naive(self, q_nope, q_rope, c, kr, offset):
        B, Sk, _ = c.shape
        k_nope = self.w_uk(c).view(B, Sk, self.H, self.dn).transpose(1, 2)      # [B,H,Sk,dn]
        v = self.w_uv(c).view(B, Sk, self.H, self.dv).transpose(1, 2)           # [B,H,Sk,dv]
        k = torch.cat([k_nope, kr[:, None].expand(-1, self.H, -1, -1)], -1)     # [B,H,Sk,dn+dr]
        q = torch.cat([q_nope, q_rope], -1)
        o = causal_sdpa(q, k, v, offset, self.scale)
        Sq = o.shape[2]
        return self.wo(o.transpose(1, 2).reshape(B, Sq, self.H * self.dv))

    # ---- absorbed: attention directly in latent space ----
    def _attend_absorbed(self, q_nope, q_rope, c, kr, offset):
        B, Sk, _ = c.shape
        W_uk = self.w_uk.weight.view(self.H, self.dn, self.r)                    # [H,dn,r]
        W_uv = self.w_uv.weight.view(self.H, self.dv, self.r)                    # [H,dv,r]
        q_lat = torch.einsum("bhsn,hnr->bhsr", q_nope, W_uk)                     # absorb W_uk into q
        q = torch.cat([q_lat, q_rope], -1)                                       # [B,H,Sq,r+dr]
        k = torch.cat([c, kr], -1)[:, None]                                      # [B,1,Sk,r+dr]  ONE shared "head"
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale               # [B,H,Sq,Sk]
        Sq = scores.shape[2]
        if Sq > 1:
            mask = torch.arange(Sk)[None, :] > (torch.arange(Sq)[:, None] + offset)
            scores = scores.masked_fill(mask, float("-inf"))
        p = torch.softmax(scores, -1)
        o_lat = torch.matmul(p, c[:, None])                                      # [B,H,Sq,r]
        o = torch.einsum("bhsr,hvr->bhsv", o_lat, W_uv)                          # absorb W_uv after attention
        return self.wo(o.transpose(1, 2).reshape(B, Sq, self.H * self.dv))

    def forward_full(self, x, absorbed=False):
        pos = torch.arange(x.shape[1])
        qn, qr = self._q(x, pos)
        c, kr = self._latent(x, pos)
        f = self._attend_absorbed if absorbed else self._attend_naive
        return f(qn, qr, c, kr, 0)

    def prefill(self, x):
        pos = torch.arange(x.shape[1])
        qn, qr = self._q(x, pos)
        c, kr = self._latent(x, pos)
        return self._attend_naive(qn, qr, c, kr, 0), {"c": c, "kr": kr}   # cache: r + d_r per token

    def decode(self, x_t, cache, absorbed=True):
        L = cache["c"].shape[1]
        pos = torch.tensor([L])
        qn, qr = self._q(x_t, pos)
        c1, kr1 = self._latent(x_t, pos)
        c, kr = torch.cat([cache["c"], c1], 1), torch.cat([cache["kr"], kr1], 1)
        f = self._attend_absorbed if absorbed else self._attend_naive
        return f(qn, qr, c, kr, L), {"c": c, "kr": kr}


# --------------------------------------------------------------------------- FFN: dense & MoE
class DenseFFN(nn.Module):
    """SwiGLU FFN: down( silu(gate x) * up x )."""

    def __init__(self, d, dff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, dff, bias=False), nn.Linear(d, dff, bias=False), nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Top-k routed mixture of SwiGLU experts (no shared expert, no capacity limit).

    `router_skew` > 0 adds a fixed per-expert bias to the router logits so that a few experts are
    'popular' (real routers are skewed too); 0 = roughly uniform random routing.
    `last_active` records how many DISTINCT experts were touched by the last call: this is what
    decides how many expert weight bytes must be streamed from memory.
    """

    def __init__(self, cfg: Config, router_skew: float = 0.0):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.router = nn.Linear(cfg.d_model, self.E, bias=False)
        self.experts = nn.ModuleList(DenseFFN(cfg.d_model, cfg.d_ff) for _ in range(self.E))
        self.register_buffer("bias", router_skew * torch.randn(self.E))
        self.last_active = 0

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        logits = self.router(flat) + self.bias
        w, idx = torch.topk(torch.softmax(logits, -1), self.k, dim=-1)           # [T,k]
        w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        active = 0
        for e in range(self.E):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue                                                          # expert untouched -> weights never read
            active += 1
            out.index_add_(0, tok, self.experts[e](flat[tok]) * w[tok, slot, None])
        self.last_active = active
        return out.view(B, S, D)


def build_attention(name: str, cfg: Config) -> nn.Module:
    if name == "MHA":
        return GroupedAttention(cfg, cfg.n_heads)
    if name == "GQA":
        return GroupedAttention(cfg, cfg.n_kv_heads)
    if name == "MQA":
        return GroupedAttention(cfg, 1)
    if name == "MLA":
        return MultiHeadLatentAttention(cfg)
    raise ValueError(name)


def cache_bytes(cache: dict, ) -> int:
    return sum(t.numel() * t.element_size() for t in cache.values())
