"""Compare MHA, GQA, MQA and MLA, all with a KV cache, on the same inputs.

Correctness: MLA is checked against MHA whose key and value weights are the
products W_dkv @ W_uk and W_dkv @ W_uv. GQA is checked against MHA with
group-tiled K/V weights. MQA is checked against MHA with its single K/V head
tiled across all heads.

Timing: TTFT, ITL, throughput, final KV cache size (float32), and cost per
million tokens at --hourly-usd.
"""

import argparse
import math
import statistics
import time

import torch

from kv_cache_mha import init_kv_cache, multi_head_attention_cached
from mqa_kv_cache import init_mqa_cache, multi_query_attention_cached
from gqa_kv_cache import init_gqa_cache, grouped_query_attention_cached
from mla_kv_cache import init_mla_cache, multi_latent_attention_cached
from cost_model import cost_per_million_tokens


def rnd(*shape, device, fan_in):
    return torch.randn(*shape, device=device) / math.sqrt(fan_in)


def build_weights(D, H, G, dc, device):
    d_head = D // H
    R = H // G
    w_q = rnd(D, D, device=device, fan_in=D)
    w_o = rnd(D, D, device=device, fan_in=D)

    # GQA and its MHA equivalent
    w_k_g = rnd(D, G * d_head, device=device, fan_in=D)
    w_v_g = rnd(D, G * d_head, device=device, fan_in=D)
    gqa = (w_q, w_k_g, w_v_g, w_o)
    mha_from_gqa = (
        w_q,
        w_k_g.view(D, G, d_head).repeat_interleave(R, dim=1).reshape(D, D),
        w_v_g.view(D, G, d_head).repeat_interleave(R, dim=1).reshape(D, D),
        w_o,
    )

    # MQA and its MHA equivalent
    w_k_q = w_k_g[:, :d_head].contiguous()
    w_v_q = w_v_g[:, :d_head].contiguous()
    mqa = (w_q, w_k_q, w_v_q, w_o)

    # MLA and its MHA equivalent
    w_dkv = rnd(D, dc, device=device, fan_in=D)
    w_uk = rnd(dc, D, device=device, fan_in=dc)
    w_uv = rnd(dc, D, device=device, fan_in=dc)
    mla = (w_q, w_dkv, w_uk, w_uv, w_o)
    mha_from_mla = (w_q, w_dkv @ w_uk, w_dkv @ w_uv, w_o)

    # Full MHA baseline (independent K/V weights)
    w_k_m = rnd(D, D, device=device, fan_in=D)
    w_v_m = rnd(D, D, device=device, fan_in=D)
    mha = (w_q, w_k_m, w_v_m, w_o)

    return dict(mha=mha, gqa=gqa, mqa=mqa, mla=mla,
                mha_from_gqa=mha_from_gqa, mqa_as_mha=(w_q, w_k_q.repeat(1, H), w_v_q.repeat(1, H), w_o),
                mha_from_mla=mha_from_mla)


def run(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    D, H, G, B, dc = args.d_model, args.num_heads, args.num_kv_heads, args.batch, args.latent_dim
    d_head = D // H
    W = build_weights(D, H, G, dc, device)
    prompt = torch.randn(B, args.prompt_len, D, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    methods = {
        "MHA": (
            lambda x: multi_head_attention_cached(x, *W["mha"], num_heads=H, cache=init_kv_cache()),
            lambda x, c: multi_head_attention_cached(x, *W["mha"], num_heads=H, cache=c),
        ),
        f"GQA (G={G})": (
            lambda x: grouped_query_attention_cached(
                x, *W["gqa"], num_heads=H, num_kv_heads=G, cache=init_gqa_cache()),
            lambda x, c: grouped_query_attention_cached(
                x, *W["gqa"], num_heads=H, num_kv_heads=G, cache=c),
        ),
        "MQA": (
            lambda x: multi_query_attention_cached(x, *W["mqa"], num_heads=H, cache=init_mqa_cache()),
            lambda x, c: multi_query_attention_cached(x, *W["mqa"], num_heads=H, cache=c),
        ),
        f"MLA (latent={dc})": (
            lambda x: multi_latent_attention_cached(
                x, *W["mla"], num_heads=H, cache=init_mla_cache()),
            lambda x, c: multi_latent_attention_cached(x, *W["mla"], num_heads=H, cache=c),
        ),
    }

    # Correctness: each cached variant must match its MHA equivalent.
    def max_diff_vs(name_fast, fast_fns, ref_fns):
        with torch.no_grad():
            check = prompt[:1, : min(8, args.prompt_len)].clone()
            out_a, c_a = fast_fns[0](check)
            out_b, c_b = ref_fns[0](check)
            worst = (out_a - out_b).abs().max().item()
            last_a, last_b = out_a[:, -1:], out_b[:, -1:]
            for _ in range(8):
                out_a, c_a = fast_fns[1](last_a, c_a)
                out_b, c_b = ref_fns[1](last_b, c_b)
                worst = max(worst, (out_a - out_b).abs().max().item())
                last_a, last_b = out_a, out_b
        print(f"correctness: {name_fast} vs its MHA equivalent, max diff = {worst:.2e}")
        assert worst < 1e-4, f"{name_fast} does not match its MHA equivalent"

    mha_eq = lambda w: (
        lambda x: multi_head_attention_cached(x, *w, num_heads=H, cache=init_kv_cache()),
        lambda x, c: multi_head_attention_cached(x, *w, num_heads=H, cache=c),
    )
    max_diff_vs("GQA", methods[f"GQA (G={G})"], mha_eq(W["mha_from_gqa"]))
    max_diff_vs("MQA", methods["MQA"], mha_eq(W["mqa_as_mha"]))
    max_diff_vs("MLA", methods[f"MLA (latent={dc})"], mha_eq(W["mha_from_mla"]))

    def time_method(prefill, step):
        with torch.no_grad():
            for _ in range(args.warmup):
                prefill(prompt)
            sync()
            ttft = []
            for _ in range(args.repeats):
                sync()
                t0 = time.perf_counter()
                out, cache = prefill(prompt)
                sync()
                ttft.append((time.perf_counter() - t0) * 1000.0)

            _, cache = prefill(prompt)
            last = out[:, -1:, :]
            itl, total_gen = [], 0.0
            for _ in range(args.gen_len - 1):
                sync()
                t0 = time.perf_counter()
                out, cache = step(last, cache)
                sync()
                dt = time.perf_counter() - t0
                itl.append(dt * 1000.0)
                total_gen += dt
                last = out

        total_time = total_gen + ttft[-1] / 1000.0
        tput = (B * args.gen_len) / total_time
        cache_tensors = [t for t in cache.values() if t is not None]
        cache_kib = sum(t.numel() for t in cache_tensors) * 4 / 1024
        return {
            "ttft": statistics.median(ttft),
            "itl": statistics.mean(itl),
            "tput": tput,
            "cache": cache_kib,
            "cost": cost_per_million_tokens(tput, args.hourly_usd),
        }

    results = {name: time_method(p, s) for name, (p, s) in methods.items()}

    names = list(results)
    print(f"\nconfig: device={device} batch={B} prompt_len={args.prompt_len} "
          f"gen_len={args.gen_len} d_model={D} heads={H} kv_heads(GQA)={G} "
          f"latent_dim(MLA)={dc} d_head={d_head}")
    print(f"{'':24s}" + "".join(f"{n:>20s}" for n in names))
    rows = [
        ("Prefill (TTFT, ms)", "ttft", "{:.3f}"),
        ("ITL (ms/token)", "itl", "{:.3f}"),
        ("Throughput (tok/s)", "tput", "{:.1f}"),
        ("KV cache, final (KiB)", "cache", "{:.1f}"),
        ("Cost / 1M tokens (USD)", "cost", "{:.4f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:24s}" + "".join(f"{fmt.format(results[n][key]):>20s}" for n in names))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--hourly-usd", type=float, default=2.50,
                   help="assumed hardware price per hour, used for cost per token")
    run(p.parse_args())
