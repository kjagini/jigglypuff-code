"""Benchmark multi_head_attention with LLM-serving style metrics.

- TTFT (time to first token): latency of one forward pass over the prompt.
- ITL (inter-token latency): average latency of each subsequent decode step.
  The function has no KV cache, so each decode step re-runs attention over the
  full (prompt + generated) sequence. This makes ITL grow with sequence length.
- Throughput: generated tokens per second, over the whole generation.

Decoding is simulated: each new token is a random vector appended to the
sequence and fed back in. The output of the last position is used as the
next input vector, so the attention is actually on the data path.
"""

import argparse
import math
import statistics
import time

import torch

from multi_head_attention import multi_head_attention


def run(args):
    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.float32

    D, H = args.d_model, args.num_heads
    ws = [(torch.randn(D, D, device=device, dtype=dtype) / math.sqrt(D)) for _ in range(4)]

    prompt = torch.randn(args.batch, args.prompt_len, D, device=device, dtype=dtype)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def forward(x):
        out, _ = multi_head_attention(x, x, x, *ws, num_heads=H, causal=True)
        return out

    # Warmup
    for _ in range(args.warmup):
        forward(prompt)
    sync()

    # TTFT: prefill over the prompt, measured several times
    ttft_samples = []
    for _ in range(args.repeats):
        sync()
        t0 = time.perf_counter()
        out = forward(prompt)
        sync()
        ttft_samples.append((time.perf_counter() - t0) * 1000.0)

    # Decode: generate tokens one at a time, timing each step
    seq = prompt.clone()
    last = out[:, -1:, :]
    itl_samples = []
    total_gen_time = 0.0
    for _ in range(args.gen_len - 1):
        seq = torch.cat([seq, last], dim=1)
        sync()
        t0 = time.perf_counter()
        out = forward(seq)
        sync()
        dt = time.perf_counter() - t0
        itl_samples.append(dt * 1000.0)
        total_gen_time += dt
        last = out[:, -1:, :]

    ttft_ms = statistics.median(ttft_samples)
    itl_ms = statistics.mean(itl_samples) if itl_samples else float("nan")
    # Tokens generated = batch * gen_len; time includes the prefill for the first token
    total_time = total_gen_time + ttft_samples[-1] / 1000.0
    tokens = args.batch * args.gen_len
    throughput = tokens / total_time

    print(f"device={device} batch={args.batch} prompt_len={args.prompt_len} "
          f"gen_len={args.gen_len} d_model={D} heads={H}")
    print(f"TTFT (median over {args.repeats} runs): {ttft_ms:.3f} ms")
    print(f"ITL  (mean over {len(itl_samples)} decode steps): {itl_ms:.3f} ms/token")
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
