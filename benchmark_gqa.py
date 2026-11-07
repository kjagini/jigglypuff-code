"""Compare MHA, GQA and MQA, all with a KV cache, on the same inputs.

Correctness: MHA's key/value weights are built from the GQA weights by copying
each group's key/value projection into every head in that group. MHA on those
weights should match GQA exactly. MQA is GQA with one group, so the same is
checked for MQA using its own weights.

Timing: same TTFT / ITL / throughput metrics as the earlier benchmarks, plus
the final KV cache size in KiB (float32).
"""

import argparse
import math
import statistics
import time

import torch

from kv_cache_mha import init_kv_cache, multi_head_attention_cached
from mqa_kv_cache import init_mqa_cache, multi_query_attention_cached
from gqa_kv_cache import init_gqa_cache, grouped_query_attention_cached


def build_weights(D, H, G, device):
    d_head = D // H
    R = H // G
    w_q = torch.randn(D, D, device=device) / math.sqrt(D)
    w_o = torch.randn(D, D, device=device) / math.sqrt(D)
    w_k = torch.randn(D, G * d_head, device=device) / math.sqrt(D)
    w_v = torch.randn(D, G * d_head, device=device) / math.sqrt(D)
    # MHA weights: head h gets the key/value projection of its group h // R.
    w_k_mha = w_k.view(D, G, d_head).repeat_interleave(R, dim=1).reshape(D, D)
    w_v_mha = w_v.view(D, G, d_head).repeat_interleave(R, dim=1).reshape(D, D)
    # MQA weights: the single key/value head is the first group's projection.
    w_k_mqa = w_k[:, :d_head].contiguous()
    w_v_mqa = w_v[:, :d_head].contiguous()
    mha = (w_q, w_k_mha, w_v_mha, w_o)
    gqa = (w_q, w_k, w_v, w_o)
    mqa = (w_q, w_k_mqa, w_v_mqa, w_o)
    return mha, gqa, mqa


def run(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    D, H, G, B = args.d_model, args.num_heads, args.num_kv_heads, args.batch
    R = H // G
    d_head = D // H
    mha_w, gqa_w, mqa_w = build_weights(D, H, G, device)
    prompt = torch.randn(B, args.prompt_len, D, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Each method: (prefill(x) -> (out, cache), step(x, cache) -> (out, cache))
    methods = {
        "MHA + cache": (
            lambda x: multi_head_attention_cached(x, *mha_w, num_heads=H, cache=init_kv_cache()),
            lambda x, c: multi_head_attention_cached(x, *mha_w, num_heads=H, cache=c),
        ),
        f"GQA (G={G}) + cache": (
            lambda x: grouped_query_attention_cached(
                x, *gqa_w, num_heads=H, num_kv_heads=G, cache=init_gqa_cache()),
            lambda x, c: grouped_query_attention_cached(
                x, *gqa_w, num_heads=H, num_kv_heads=G, cache=c),
        ),
        "MQA + cache": (
            lambda x: multi_query_attention_cached(x, *mqa_w, num_heads=H, cache=init_mqa_cache()),
            lambda x, c: multi_query_attention_cached(x, *mqa_w, num_heads=H, cache=c),
        ),
    }

    # Correctness checks
    with torch.no_grad():
        mha_prefill, mha_step = methods["MHA + cache"]
        gqa_prefill, gqa_step = methods[f"GQA (G={G}) + cache"]
        check = prompt[:1, : min(8, args.prompt_len)].clone()
        out_m, c_m = mha_prefill(check)
        out_g, c_g = gqa_prefill(check)
        max_diff = (out_m - out_g).abs().max().item()
        last_m, last_g = out_m[:, -1:], out_g[:, -1:]
        for _ in range(8):
            out_m, c_m = mha_step(last_m, c_m)
            out_g, c_g = gqa_step(last_g, c_g)
            max_diff = max(max_diff, (out_m - out_g).abs().max().item())
            last_m, last_g = out_m, out_g
    print(f"correctness: max |MHA - GQA| over prefill + 8 decode steps = {max_diff:.2e}")
    assert max_diff < 1e-4, "GQA does not match MHA with group-tiled K/V weights"

    # Timing
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
            itl = []
            total_gen = 0.0
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
        tokens = B * args.gen_len
        cache_kib = (cache["k"].numel() + cache["v"].numel()) * 4 / 1024
        return {
            "ttft": statistics.median(ttft),
            "itl": statistics.mean(itl),
            "tput": tokens / total_time,
            "cache": cache_kib,
        }

    results = {name: time_method(p, s) for name, (p, s) in methods.items()}

    names = list(results)
    print(f"\nconfig: device={device} batch={B} prompt_len={args.prompt_len} "
          f"gen_len={args.gen_len} d_model={D} heads={H} kv_heads(GQA)={G} d_head={d_head}")
    header = f"{'':24s}" + "".join(f"{n:>22s}" for n in names)
    print(header)
    rows = [
        ("Prefill (TTFT, ms)", "ttft", "{:.3f}"),
        ("ITL (ms/token)", "itl", "{:.3f}"),
        ("Throughput (tok/s)", "tput", "{:.1f}"),
        ("KV cache, final (KiB)", "cache", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:24s}" + "".join(f"{fmt.format(results[n][key]):>22s}" for n in names))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    run(p.parse_args())
