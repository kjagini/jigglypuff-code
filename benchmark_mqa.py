"""Compare multi-query attention (MQA) with multi-head attention (MHA), both with a KV cache.

Same inputs and same decode loop for both.

Correctness: MHA's key and value weights are the MQA weights tiled across
all heads. In that setup every MHA head sees the same keys and values, so MHA
should give the same outputs as MQA. The script checks that.

Timing: the same TTFT / ITL / throughput metrics as benchmark_kv_cache.py,
plus the KV cache size in bytes (float32).
"""

import argparse
import math
import statistics
import time

import torch

from kv_cache_mha import init_kv_cache, multi_head_attention_cached
from mqa_kv_cache import init_mqa_cache, multi_query_attention_cached


def build_weights(D, H, device):
    d_head = D // H
    w_q = torch.randn(D, D, device=device) / math.sqrt(D)
    w_k = torch.randn(D, d_head, device=device) / math.sqrt(D)
    w_v = torch.randn(D, d_head, device=device) / math.sqrt(D)
    w_o = torch.randn(D, D, device=device) / math.sqrt(D)
    # MHA weights: the single K/V head copied into every head's slot.
    w_k_mha = w_k.repeat(1, H)
    w_v_mha = w_v.repeat(1, H)
    return (w_q, w_k, w_v, w_o), (w_q, w_k_mha, w_v_mha, w_o)


def run(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    D, H, B = args.d_model, args.num_heads, args.batch
    d_head = D // H
    mqa_w, mha_w = build_weights(D, H, device)
    prompt = torch.randn(B, args.prompt_len, D, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def mha_prefill(x):
        return multi_head_attention_cached(x, *mha_w, num_heads=H, cache=init_kv_cache())

    @torch.no_grad()
    def mqa_prefill(x):
        return multi_query_attention_cached(x, *mqa_w, num_heads=H, cache=init_mqa_cache())

    # Correctness: same inputs, same decode loop, outputs should match.
    with torch.no_grad():
        check = prompt[:1, : min(8, args.prompt_len)].clone()
        out_mha, c_mha = mha_prefill(check)
        out_mqa, c_mqa = mqa_prefill(check)
        max_diff = (out_mha - out_mqa).abs().max().item()
        last_mha, last_mqa = out_mha[:, -1:], out_mqa[:, -1:]
        for _ in range(8):
            out_mha, c_mha = multi_head_attention_cached(
                last_mha, *mha_w, num_heads=H, cache=c_mha)
            out_mqa, c_mqa = multi_query_attention_cached(
                last_mqa, *mqa_w, num_heads=H, cache=c_mqa)
            max_diff = max(max_diff, (out_mha - out_mqa).abs().max().item())
            last_mha, last_mqa = out_mha, out_mqa
    print(f"correctness: max |MHA - MQA| over prefill + 8 decode steps = {max_diff:.2e}")
    assert max_diff < 1e-4, "MQA does not match MHA with tiled K/V weights"

    # Timing
    def time_config(prefill, step, cache_init_fn=None):
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
        with torch.no_grad():
            for _ in range(args.gen_len - 1):
                sync()
                t0 = time.perf_counter()
                out, cache = step(last, cache)
                sync()
                dt = time.perf_counter() - t0
                itl.append(dt * 1000.0)
                total_gen += dt
                last = out
        ttft_ms = statistics.median(ttft)
        itl_ms = statistics.mean(itl)
        total_time = total_gen + ttft[-1] / 1000.0
        tokens = B * args.gen_len
        cache_bytes = (cache["k"].numel() + cache["v"].numel()) * 4
        return ttft_ms, itl_ms, tokens / total_time, cache_bytes

    mha_step = lambda x, c: multi_head_attention_cached(x, *mha_w, num_heads=H, cache=c)
    mqa_step = lambda x, c: multi_query_attention_cached(x, *mqa_w, num_heads=H, cache=c)

    mha = time_config(mha_prefill, mha_step)
    mqa = time_config(mqa_prefill, mqa_step)

    print(f"\nconfig: device={device} batch={B} prompt_len={args.prompt_len} "
          f"gen_len={args.gen_len} d_model={D} heads={H} d_head={d_head}")
    print(f"{'':28s}{'MHA + cache':>16s}{'MQA + cache':>16s}")
    rows = [
        ("Prefill (TTFT, ms)", mha[0], mqa[0], "{:.3f}"),
        ("ITL (ms/token)", mha[1], mqa[1], "{:.3f}"),
        ("Throughput (tok/s)", mha[2], mqa[2], "{:.1f}"),
        ("KV cache, final (KiB)", mha[3] / 1024, mqa[3] / 1024, "{:.1f}"),
    ]
    for name, a, b, fmt in rows:
        print(f"{name:28s}{fmt.format(a):>16s}{fmt.format(b):>16s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    run(p.parse_args())
