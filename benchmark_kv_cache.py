"""Benchmark attention with and without a KV cache, using the same experiment.

Same metrics and same simulated decode as benchmark_mha.py:
- TTFT: prefill over the prompt.
- ITL: mean time per decode step.
- Throughput: generated tokens per second, including prefill.

With the cache, prefill fills the cache and each decode step runs one new
token against the stored keys and values. Before timing, the script checks
that cached outputs match the uncached function on the same sequence.
"""

import argparse
import math
import statistics
import time

import torch

from kv_cache_mha import init_kv_cache, multi_head_attention_cached
from multi_head_attention import multi_head_attention


def run(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    D, H = args.d_model, args.num_heads
    ws = [(torch.randn(D, D, device=device) / math.sqrt(D)) for _ in range(4)]
    prompt = torch.randn(args.batch, args.prompt_len, D, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def uncached_forward(x):
        out, _ = multi_head_attention(x, x, x, *ws, num_heads=H, causal=True)
        return out

    @torch.no_grad()
    def cached_prefill(x):
        cache = init_kv_cache()
        out, cache = multi_head_attention_cached(x, *ws, num_heads=H, cache=cache)
        return out, cache

    @torch.no_grad()
    def cached_step(x, cache):
        return multi_head_attention_cached(x, *ws, num_heads=H, cache=cache)

    # Correctness: cached decode should match the uncached function on the same sequence.
    with torch.no_grad():
        check_seq = prompt[:1, : min(8, args.prompt_len)].clone()
        out_c, cache = cached_prefill(check_seq)
        last = out_c[:, -1:, :]
        for _ in range(4):
            check_seq = torch.cat([check_seq, last], dim=1)
            out_c, cache = cached_step(last, cache)
            full = uncached_forward(check_seq)
            if not torch.allclose(out_c, full[:, -1:, :], atol=1e-4):
                raise RuntimeError("cached output does not match uncached output")
            last = out_c
    print("correctness check: cached decode matches uncached attention")

    for _ in range(args.warmup):
        uncached_forward(prompt)
        cached_prefill(prompt)
    sync()

    # TTFT: prefill over the prompt
    ttft_samples = []
    for _ in range(args.repeats):
        sync()
        t0 = time.perf_counter()
        out, cache = cached_prefill(prompt)
        sync()
        ttft_samples.append((time.perf_counter() - t0) * 1000.0)

    # Decode with the cache
    _, cache = cached_prefill(prompt)
    last = out[:, -1:, :]
    itl_samples = []
    total_gen_time = 0.0
    for _ in range(args.gen_len - 1):
        sync()
        t0 = time.perf_counter()
        out, cache = cached_step(last, cache)
        sync()
        dt = time.perf_counter() - t0
        itl_samples.append(dt * 1000.0)
        total_gen_time += dt
        last = out

    ttft_ms = statistics.median(ttft_samples)
    itl_ms = statistics.mean(itl_samples) if itl_samples else float("nan")
    decode_ms = total_gen_time * 1000.0
    total_time = total_gen_time + ttft_samples[-1] / 1000.0
    tokens = args.batch * args.gen_len
    throughput = tokens / total_time

    print(f"[KV cache] device={device} batch={args.batch} prompt_len={args.prompt_len} "
          f"gen_len={args.gen_len} d_model={D} heads={H}")
    print(f"Prefill (TTFT, median over {args.repeats} runs): {ttft_ms:.3f} ms")
    print(f"Decode total ({len(itl_samples)} steps): {decode_ms:.3f} ms")
    print(f"ITL (mean per decode step): {itl_ms:.3f} ms/token")
    print(f"Throughput: {throughput:.1f} tokens/s "
          f"(batch={args.batch}, {tokens} tokens in {total_time * 1000:.1f} ms)")


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
