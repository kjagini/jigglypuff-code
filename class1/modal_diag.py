"""Why is decode on the Modal H100 slower than on an A100 or even a T4?  Hypothesis: it is CPU-launch-bound.

    python -m modal run modal_diag.py --gpus a100,h100,t4

Prints, per GPU: host CPU, a pure-Python speed test, kernel-launch cost, TinyLlama decode measured three ways
(per-step GPU events, per-step CPU launch time, total wall), GPU-busy time from the profiler, and an attempt at a
CUDA-graph decode step (no per-kernel launch overhead).
"""
import modal

app = modal.App("prefill-decode-diag")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "numpy"
).add_local_python_source("modal_diag")
GPUS = {"t4": "T4", "a100": "A100-80GB", "h100": "H100"}
REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _diag(label: str) -> str:
    import os
    import platform
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    lines = []

    def log(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s, flush=True)

    dev = torch.device("cuda")
    log(f"== {label}: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        cpu = platform.processor()
    log(f"host cpu: {cpu} | logical cpus: {os.cpu_count()}")

    t = time.perf_counter()
    s = 0
    for i in range(5_000_000):
        s += i
    log(f"python loop 5M iters: {(time.perf_counter() - t) * 1e3:.0f} ms")

    x = torch.zeros(8, device=dev)
    for _ in range(200):
        x.add_(1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3000):
        x.add_(1)
    cpu_launch_us = (time.perf_counter() - t) / 3000 * 1e6
    torch.cuda.synchronize()
    log(f"tiny kernel launch cost (CPU side): {cpu_launch_us:.1f} us each")

    try:
        m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float16, attn_implementation="sdpa")
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float16, attn_implementation="sdpa")
    m = m.to(dev).eval()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, m.config.vocab_size - 1, (1, 128), generator=g).to(dev)

    def pct(a):
        a = np.asarray(a, float)
        return f"p10 {np.percentile(a, 10):.1f} | p50 {np.percentile(a, 50):.1f} | p90 {np.percentile(a, 90):.1f}"

    @torch.inference_mode()
    def eager(n):
        out = m(input_ids=ids, use_cache=True, logits_to_keep=1)
        nxt, pkv = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        cpu_ms = []
        w0 = time.perf_counter()
        for s_, e_ in ev:
            c0 = time.perf_counter()
            s_.record()
            o = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
            nxt, pkv = o.logits[:, -1].argmax(-1, keepdim=True), o.past_key_values
            e_.record()
            cpu_ms.append((time.perf_counter() - c0) * 1e3)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) * 1e3
        return [s_.elapsed_time(e_) for s_, e_ in ev], cpu_ms, wall

    eager(10)
    for rep in range(2):
        gpu_ev, cpu_ms, wall = eager(80)
        log(f"eager run {rep + 1}: event ms/step [{pct(gpu_ev[5:])}] | CPU-launch ms/step [{pct(cpu_ms[5:])}] "
            f"| wall {wall / 80:.1f} ms/step")

    # GPU-busy time per step, from the profiler (sum of kernel durations)
    try:
        from torch.profiler import ProfilerActivity, profile
        with torch.inference_mode():
            out = m(input_ids=ids, use_cache=True, logits_to_keep=1)
            nxt, pkv = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
            for _ in range(3):
                o = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
                nxt, pkv = o.logits[:, -1].argmax(-1, keepdim=True), o.past_key_values
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    o = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
                    nxt, pkv = o.logits[:, -1].argmax(-1, keepdim=True), o.past_key_values
                torch.cuda.synchronize()
        busy = sum(e.self_device_time_total for e in prof.key_averages())
        log(f"GPU kernel-busy time: {busy / 10 / 1e3:.2f} ms/step (sum of kernel durations)")
    except Exception as e:
        log("profiler failed:", type(e).__name__, str(e)[:120])

    # CUDA-graph decode step (static cache): removes per-kernel launch cost
    try:
        from transformers import StaticCache
        L, steps = ids.shape[1], 100
        try:
            cache = StaticCache(config=m.config, max_batch_size=1, max_cache_len=L + steps + 8, device=dev, dtype=torch.float16)
        except TypeError:
            cache = StaticCache(config=m.config, max_cache_len=L + steps + 8)
        with torch.inference_mode():
            out = m(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L, device=dev), use_cache=True,
                    logits_to_keep=1)
            s_nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            s_pos = torch.tensor([L], device=dev)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    o = m(input_ids=s_nxt, past_key_values=cache, cache_position=s_pos, use_cache=True)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                o = m(input_ids=s_nxt, past_key_values=cache, cache_position=s_pos, use_cache=True)
                g_next = o.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
            w0 = time.perf_counter()
            for s_, e_ in ev:
                s_.record()
                s_nxt.copy_(g_next)
                s_pos += 1
                graph.replay()
                e_.record()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - w0) * 1e3
        itl = [s_.elapsed_time(e_) for s_, e_ in ev]
        log(f"CUDA-graph decode: event ms/step [{pct(itl[5:])}] | wall {wall / steps:.2f} ms/step")
    except Exception as e:
        log("CUDA-graph attempt failed:", type(e).__name__, str(e)[:200])
    return "\n".join(lines)


_fns = {}
for _label, _gpu in GPUS.items():
    def _make(label):
        def fn():
            return _diag(label)
        fn.__name__ = f"diag_{label}"
        return fn
    _fns[_label] = app.function(gpu=_gpu, image=image, timeout=1800, name=f"diag_{_label}", serialized=True)(_make(_label))


@app.local_entrypoint()
def main(gpus: str = "a100,h100,t4"):
    import pathlib
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    for g in [x.strip().lower() for x in gpus.split(",")]:
        txt = _fns[g].remote()
        (out / f"diag_{g}.txt").write_text(txt)
