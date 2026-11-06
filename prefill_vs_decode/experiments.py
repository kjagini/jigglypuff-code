"""
Run everything:  python experiments.py
Writes figures/*.png, results.json, results.md next to this file.

Part A  (analytical, 7B-class shapes, bf16, H100-like roofline)  -> fig1, fig2, fig3, fig5, fig6b
Part B  (measured with torch on this machine, fp32, small shapes) -> fig4, fig6a, fig6c, tables
        + validation: torch's FlopCounterMode and real tensor sizes vs the analytical formulas
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import analysis as A
from attention import Config, DenseFFN, MoEFFN, build_attention, cache_bytes

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
torch.manual_seed(0)
torch.set_grad_enabled(False)

VARIANTS = ["MHA", "GQA", "MQA", "MLA"]
# categorical slots 1-4 of the validated palette; marker = secondary (non-colour) encoding
COLOR = {"MHA": "#2a78d6", "GQA": "#eb6834", "MQA": "#1baf7a", "MLA": "#eda100",
         "MoE": "#6250d6", "Dense": "#52514e"}
MARK = {"MHA": "o", "GQA": "s", "MQA": "^", "MLA": "D", "MoE": "P", "Dense": "X"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    "lines.markeredgecolor": SURF, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  wrote", name)


def line(ax, x, y, v, **kw):
    ax.plot(x, y, color=COLOR[v], marker=MARK[v], markeredgecolor=SURF, markeredgewidth=1.2, label=v, **kw)


def fmt_bytes(b):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or u == "TB":
            return f"{b:.3g} {u}"
        b /= 1024


R = {}   # everything that goes in results.json
MD = []  # lines of results.md

# =========================================================================== PART A: analytical
cfg = A.BIG
print("Part A: analytical (7B-class shapes)")

# ---- fig0: what the attention matrix looks like in each phase
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), gridspec_kw={"width_ratios": [1, 1]})
n = 16
m = np.tril(np.ones((n, n)))
axs[0].imshow(m, cmap="Blues", vmin=0, vmax=1.6)
axs[0].set_title("Prefill: S queries x S keys")
axs[0].set_xlabel("key position"); axs[0].set_ylabel("query position")
axs[0].text(n * 0.62, n * 0.3, "masked", color=INK2, ha="center")
d = np.zeros((n, n)); d[-1, :] = 1
axs[1].imshow(d, cmap="Blues", vmin=0, vmax=1.6)
axs[1].set_title("Decode: 1 query x L cached keys")
axs[1].set_xlabel("key position (read from the KV cache)")
for a in axs:
    a.grid(False); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)
fig.suptitle("Same tokens, very different matmul shapes", x=0.12, ha="left", fontsize=11, fontweight="bold")
save(fig, "fig0_shapes.png")

# ---- fig1: KV cache size
kvt = {v: A.kv_per_token(v, cfg) * 2 for v in VARIANTS}   # bytes/token/layer (bf16)
R["kv_bytes_per_token_per_layer_bf16"] = kvt
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axs[0]
vals = [kvt[v] for v in VARIANTS]
bars = ax.bar(VARIANTS, vals, color=[COLOR[v] for v in VARIANTS], width=0.6)
for b_, v_ in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_/1024:.2g} KB", ha="center", va="bottom", color=INK)
ax.set_title("KV cache per token, per layer"); ax.set_ylabel("bytes (bf16)")
ax.set_ylim(0, max(vals) * 1.12)
ax = axs[1]
Ls = np.array([1e3, 4e3, 16e3, 64e3, 128e3])
for v in VARIANTS:
    line(ax, Ls, Ls * kvt[v] * A.N_LAYERS / 1e9, v)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Total KV cache, 1 sequence, 32 layers"); ax.set_xlabel("context length (tokens)"); ax.set_ylabel("GB")
ax.legend(loc="upper left")
fig.tight_layout(); save(fig, "fig1_kv_cache.png")

# ---- fig2: arithmetic intensity prefill vs decode (attention layer only)
S_ = 4096
rows = {}
for v in VARIANTS:
    pre = A.attention_cost(v, cfg, 1, S_, 0)
    dec = A.attention_cost(v, cfg, 1, 1, S_)
    rows[v] = {"prefill_flops": pre.flops, "prefill_bytes": pre.bytes, "prefill_intensity": pre.intensity,
               "decode_flops": dec.flops, "decode_bytes": dec.bytes, "decode_intensity": dec.intensity,
               "decode_kv_share_of_bytes": dec.kv_read_bytes / dec.bytes}
dec_naive = A.attention_cost("MLA", cfg, 1, 1, S_, mla_absorbed=False)
rows["MLA_naive_decode"] = {"decode_flops": dec_naive.flops, "decode_bytes": dec_naive.bytes,
                            "decode_intensity": dec_naive.intensity}
pre_abs = A.attention_cost("MLA", cfg, 1, S_, 0, mla_absorbed=True)
rows["MLA_absorbed_prefill"] = {"prefill_flops": pre_abs.flops, "prefill_intensity": pre_abs.intensity}
R["attention_phase_costs_B1_S4096"] = rows

fig, ax = plt.subplots(figsize=(7.5, 4))
x = np.arange(len(VARIANTS)); w = 0.36
for i, v in enumerate(VARIANTS):
    ax.bar(i - w / 2 - 0.01, rows[v]["prefill_intensity"], w, color=COLOR[v])
    ax.bar(i + w / 2 + 0.01, rows[v]["decode_intensity"], w, color=COLOR[v], hatch="////", edgecolor=SURF, linewidth=0)
    ax.text(i - w / 2, rows[v]["prefill_intensity"] * 1.12, f"{rows[v]['prefill_intensity']:.0f}", ha="center", fontsize=8.5)
    ax.text(i + w / 2, rows[v]["decode_intensity"] * 1.12, f"{rows[v]['decode_intensity']:.1f}", ha="center", fontsize=8.5)
ax.axhline(A.RIDGE, color=INK2, lw=1.2, ls="--")
ax.text(-0.45, A.RIDGE * 1.12, f"H100 ridge ~{A.RIDGE:.0f} FLOP/B (above = compute-bound)", ha="left", color=INK2, fontsize=8.5)
ax.set_yscale("log"); ax.set_ylim(0.5, 20000)
ax.set_xticks(x); ax.set_xticklabels(VARIANTS)
ax.set_ylabel("arithmetic intensity (FLOP / byte)")
ax.set_title("Attention layer @ 4096 tokens: solid = prefill, hatched = decode")
fig.tight_layout(); save(fig, "fig2_intensity.png")

# ---- fig3: roofline-estimated decode step time vs context (attention layer, batch 32)
Ls = np.array([1e3, 2e3, 4e3, 8e3, 16e3, 32e3, 64e3, 128e3])
fig, ax = plt.subplots(figsize=(7, 4))
est = {}
for v in VARIANTS:
    t = [A.attention_cost(v, cfg, 32, 1, int(L)).time() * 1e3 for L in Ls]
    est[v] = t
    line(ax, Ls, t, v)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("context length already in KV cache"); ax.set_ylabel("ms per layer per decode step")
ax.set_title("Decode step time (batch 32) is set by how many KV bytes must be read")
ax.legend()
fig.tight_layout(); save(fig, "fig3_decode_time_model.png")
R["decode_ms_per_layer_B32"] = {"L": Ls.tolist(), **est}

# ---- fig5: decode intensity vs batch size
Bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
fig, ax = plt.subplots(figsize=(7, 4))
for v in VARIANTS:
    line(ax, Bs, [A.attention_cost(v, cfg, int(b), 1, 4096).intensity for b in Bs], v)
ax.plot(Bs, [A.attention_cost("MLA", cfg, int(b), 1, 4096, mla_absorbed=False).intensity for b in Bs],
        color=COLOR["MLA"], ls=":", lw=2, label="MLA, naive decode")
ax.axhline(A.RIDGE, color=INK2, lw=1.2, ls="--")
ax.text(Bs[-1], A.RIDGE * 0.62, "ridge (compute-bound above)", color=INK2, fontsize=8.5, ha="right")
ax.set_ylim(0.6, 3e4)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("decode batch size (sequences, each at 4096 context)"); ax.set_ylabel("FLOP / byte")
ax.set_title("Batching does NOT rescue decode attention: KV bytes grow with the batch")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5)
fig.tight_layout(); save(fig, "fig5_batch_intensity.png")
R["decode_intensity_vs_batch_L4096"] = {"B": Bs.tolist(),
    **{v: [A.attention_cost(v, cfg, int(b), 1, 4096).intensity for b in Bs] for v in VARIANTS}}

# ---- fig6b: FFN intensity dense vs MoE vs tokens
Ts = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384])
dense_i = [A.ffn_cost("dense", cfg, 1, int(t)).intensity for t in Ts]
moe_i = [A.ffn_cost("moe", cfg, 1, int(t)).intensity for t in Ts]
R["ffn_intensity_vs_tokens"] = {"tokens": Ts.tolist(), "dense": dense_i, "moe": moe_i}

# =========================================================================== PART B: measured
print("Part B: measured on this machine")
scfg = Config()   # d_model 1024, 16 heads x 64, GQA 4 kv heads, MLA r=128, E=16 top-2
torch.set_num_threads(4)
R["measured_config"] = scfg.__dict__ | {"dtype": "float32", "threads": torch.get_num_threads()}


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3   # ms


def fake_cache(v, B, L):
    if v == "MLA":
        return {"c": torch.randn(B, L, scfg.kv_lora_rank), "kr": torch.randn(B, L, scfg.qk_rope_dim)}
    G = {"MHA": scfg.n_heads, "GQA": scfg.n_kv_heads, "MQA": 1}[v]
    return {"k": torch.randn(B, G, L, scfg.head_dim), "v": torch.randn(B, G, L, scfg.head_dim)}


models = {v: build_attention(v, scfg).eval() for v in VARIANTS}

# ---- validation 1: analytical FLOPs vs torch FlopCounterMode; cache bytes vs real tensors
val = []
for v in VARIANTS:
    S = 512
    x = torch.randn(1, S, scfg.d_model)
    with FlopCounterMode(display=False) as fc:
        _, cache = models[v].prefill(x)
    # prefill() in our code runs the attention once; GroupedAttention.prefill also recomputes k,v for the
    # cache (2 extra small matmuls), so we count forward_full for the like-for-like FLOP check:
    with FlopCounterMode(display=False) as fc:
        models[v].forward_full(x)
    meas_pre = fc.get_total_flops()
    ana_pre = A.attention_cost(v, scfg, 1, S, 0, bytes_per=4, causal_skip=False, mla_absorbed=False).flops
    cache0 = fake_cache(v, 1, S)
    xt = torch.randn(1, 1, scfg.d_model)
    with FlopCounterMode(display=False) as fc:
        _, c2 = (models[v].decode(xt, cache0, absorbed=True) if v == "MLA" else models[v].decode(xt, cache0))
    meas_dec = fc.get_total_flops()
    ana_dec = A.attention_cost(v, scfg, 1, 1, S, bytes_per=4, mla_absorbed=True).flops
    val.append({"variant": v, "prefill_flops_measured": meas_pre, "prefill_flops_model": ana_pre,
                "decode_flops_measured": meas_dec, "decode_flops_model": ana_dec,
                "cache_bytes_measured": cache_bytes(cache), "cache_bytes_model": S * A.kv_per_token(v, scfg) * 4})
R["validation"] = val

# ---- measurement 2: latency scaling
print("  timing prefill vs S ...")
S_list = [128, 256, 512, 1024, 2048]
pre_ms = {v: [] for v in VARIANTS}
for v in VARIANTS:
    for S in S_list:
        x = torch.randn(1, S, scfg.d_model)
        pre_ms[v].append(timeit(lambda: models[v].prefill(x), reps=3, warm=1))
print("  timing decode vs L ...")
L_list = [256, 512, 1024, 2048, 4096, 8192]
DB = 8
dec_ms = {v: [] for v in VARIANTS}
for v in VARIANTS:
    for L in L_list:
        cache = fake_cache(v, DB, L)
        xt = torch.randn(DB, 1, scfg.d_model)
        dec_ms[v].append(timeit(lambda: models[v].decode(xt, cache), reps=5, warm=2))
dec_ms_mla_naive = []
for L in L_list:
    cache = fake_cache("MLA", DB, L)
    xt = torch.randn(DB, 1, scfg.d_model)
    dec_ms_mla_naive.append(timeit(lambda: models["MLA"].decode(xt, cache, absorbed=False), reps=5, warm=2))
R["prefill_ms_vs_S_B1"] = {"S": S_list, **pre_ms}
R["decode_ms_vs_L_B8"] = {"L": L_list, **dec_ms, "MLA_naive": dec_ms_mla_naive}

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
ax = axs[0]
for v in VARIANTS:
    line(ax, S_list, pre_ms[v], v)
ref = np.array(S_list, float); ax.plot(ref, pre_ms["MHA"][0] * (ref / ref[0]) ** 2, color=INK2, lw=1, ls="--")
ax.text(ref[-2], pre_ms["MHA"][0] * (ref[-2] / ref[0]) ** 2 * 1.3, "slope 2 (S^2)", color=INK2, fontsize=8.5, ha="right")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_title("Prefill: time to process the whole prompt"); ax.set_xlabel("prompt length S (batch 1)"); ax.set_ylabel("ms (CPU, one layer)")
ax.legend(loc="upper left")
ax = axs[1]
for v in VARIANTS:
    line(ax, L_list, dec_ms[v], v)
ax.plot(L_list, dec_ms_mla_naive, color=COLOR["MLA"], ls=":", lw=2, label="MLA, naive decode")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_title("Decode: time to emit ONE token per sequence"); ax.set_xlabel("KV cache length L (batch 8)"); ax.set_ylabel("ms (CPU, one layer)")
ax.legend(loc="upper left")
fig.tight_layout(); save(fig, "fig4_measured_latency.png")

# per-token cost, prefill vs decode (S = L = 1024, batch 1)
per_tok = {}
for v in VARIANTS:
    x = torch.randn(1, 1024, scfg.d_model)
    p = timeit(lambda: models[v].prefill(x), reps=3, warm=1) / 1024
    cache = fake_cache(v, 1, 1024); xt = torch.randn(1, 1, scfg.d_model)
    d_ = timeit(lambda: models[v].decode(xt, cache), reps=10, warm=3)
    per_tok[v] = {"prefill_ms_per_token": p, "decode_ms_per_token": d_, "ratio": d_ / p}
R["per_token_ms_1024"] = per_tok

# ---- MoE experiments
print("  MoE ...")
tok_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
E, k = scfg.n_experts, scfg.top_k
act = {}
for name, skew in [("uniform router", 0.0), ("skewed router", 2.0)]:
    moe = MoEFFN(scfg, router_skew=skew).eval()
    ys = []
    for T in tok_list:
        trials = []
        for _ in range(20 if T < 64 else 5):
            moe(torch.randn(1, T, scfg.d_model)); trials.append(moe.last_active)
        ys.append(float(np.mean(trials)))
    act[name] = ys
theory = [A.expected_active_experts(E, k, T) for T in tok_list]
R["moe_active_experts"] = {"tokens": tok_list, "theory_uniform": theory, **act}

moe = MoEFFN(scfg).eval()
dense = DenseFFN(scfg.d_model, k * scfg.d_ff).eval()    # same ACTIVE params per token as the MoE
mt = {"tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024], "moe": [], "dense": []}
for T in mt["tokens"]:
    x = torch.randn(1, T, scfg.d_model)
    mt["moe"].append(timeit(lambda: moe(x), reps=5, warm=2) / T)
    mt["dense"].append(timeit(lambda: dense(x), reps=5, warm=2) / T)
R["moe_ms_per_token"] = mt

fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
ax = axs[0]
ax.plot(tok_list, theory, color=INK2, ls="--", lw=1.3, label="theory (uniform)")
for (name, ys), mk in zip(act.items(), ["P", "X"]):
    ax.plot(tok_list, ys, color=COLOR["MoE"], marker=mk, ls="-" if "uniform" in name else ":", label=name, markeredgecolor=SURF)
ax.axhline(E, color=GRID, lw=1)
ax.set_xscale("log", base=2)
ax.set_title(f"Experts whose weights get read (E={E}, top-{k})"); ax.set_xlabel("tokens in the forward pass"); ax.set_ylabel("distinct experts touched")
ax.legend(loc="center right")
ax = axs[1]
ax.plot(Ts, dense_i, color=COLOR["Dense"], marker=MARK["Dense"], label="dense FFN (11008)", markeredgecolor=SURF)
ax.plot(Ts, moe_i, color=COLOR["MoE"], marker=MARK["MoE"], label="MoE 64 experts, top-4", markeredgecolor=SURF)
ax.axhline(A.RIDGE, color=INK2, lw=1.2, ls="--")
ax.text(Ts[0], A.RIDGE * 1.1, "ridge", color=INK2, fontsize=8.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_title("FFN arithmetic intensity (7B-class shapes)"); ax.set_xlabel("tokens in the step (batch x new tokens)"); ax.set_ylabel("FLOP / byte")
ax.legend(loc="upper left")
ax = axs[2]
ax.plot(mt["tokens"], mt["dense"], color=COLOR["Dense"], marker=MARK["Dense"], label="dense (same active params)", markeredgecolor=SURF)
ax.plot(mt["tokens"], mt["moe"], color=COLOR["MoE"], marker=MARK["MoE"], label="MoE", markeredgecolor=SURF)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_title("Measured on CPU: cost per token"); ax.set_xlabel("tokens in the forward pass"); ax.set_ylabel("ms / token")
ax.legend(loc="upper right")
fig.tight_layout(); save(fig, "fig6_moe.png")

# =========================================================================== results.md
MD.append("# Auto-generated results (python experiments.py)\n")
MD.append("## Validation: cost model vs torch (config d=1024, H=16, dh=64, S=L=512, fp32)\n")
MD.append("| variant | prefill FLOPs torch | model | decode FLOPs torch | model | cache bytes torch | model |\n|---|---|---|---|---|---|---|")
for r in val:
    MD.append(f"| {r['variant']} | {r['prefill_flops_measured']:.3e} | {r['prefill_flops_model']:.3e} | {r['decode_flops_measured']:.3e} | "
              f"{r['decode_flops_model']:.3e} | {r['cache_bytes_measured']} | {r['cache_bytes_model']} |")
MD.append("\n## Attention layer, 7B-class shapes, bf16, B=1, S=L=4096 (analytical)\n")
MD.append("| variant | KV B/token/layer | prefill FLOP/B | decode FLOP/B | decode bytes | KV share of decode bytes |\n|---|---|---|---|---|---|")
for v in VARIANTS:
    r = rows[v]
    MD.append(f"| {v} | {kvt[v]} | {r['prefill_intensity']:.0f} | {r['decode_intensity']:.2f} | {fmt_bytes(r['decode_bytes'])} | {r['decode_kv_share_of_bytes']*100:.0f}% |")
MD.append(f"\nMLA with naive (decompress-every-step) decode: {rows['MLA_naive_decode']['decode_intensity']:.2f} FLOP/B, "
          f"{rows['MLA_naive_decode']['decode_flops']:.3e} FLOPs vs absorbed {rows['MLA']['decode_flops']:.3e}.  "
          f"MLA with absorbed prefill: {rows['MLA_absorbed_prefill']['prefill_flops']:.3e} FLOPs vs naive {rows['MLA']['prefill_flops']:.3e}.\n")
MD.append("## Measured CPU, ms per token at 1024 tokens (fp32, one layer, batch 1)\n")
MD.append("| variant | prefill ms/token | decode ms/token | decode / prefill |\n|---|---|---|---|")
for v in VARIANTS:
    r = per_tok[v]
    MD.append(f"| {v} | {r['prefill_ms_per_token']:.4f} | {r['decode_ms_per_token']:.3f} | {r['ratio']:.0f}x |")
(OUT / "results.md").write_text("\n".join(MD) + "\n")
(OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
print("\n".join(MD))
