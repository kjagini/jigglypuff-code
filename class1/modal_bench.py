"""E8 subset (prefill vs decode on a GPU) on Modal. Writes results/<gpu>_E8.csv in the same format as the
Colab notebook (prefill_decode_article.ipynb), so its "combine E8" cell can build Fig 10.

    python -m modal run modal_bench.py                     # a100 + h100
    python -m modal run modal_bench.py --gpus t4,a10g      # any subset of: t4, a10g, a100, h100

Spec-sheet peaks are dense FP16 tensor FLOP/s and memory bandwidth. Verify before quoting.
"""
import modal

app = modal.App("prefill-decode-e8")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "accelerate", "pandas", "numpy"
).add_local_python_source("modal_bench")

# label -> (modal gpu string, peak FLOP/s, peak bytes/s). A100 is pinned to the 80GB card so the bandwidth is known.
GPUS = {
    "t4": ("T4", 65e12, 320e9),
    "a10g": ("A10G", 70e12, 600e9),
    "a100": ("A100-80GB", 312e12, 2039e9),
    "h100": ("H100", 989e12, 3350e9),
}
MODELS = {"gpt2": "gpt2", "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}
WARMUP, REPEATS, PF_L = 3, 5, 512


def _e8(label: str) -> str:
    import time

    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM

    dev = torch.device("cuda")
    _, spec_flops, spec_bw = GPUS[label]
    print(f"[{label}] {torch.cuda.get_device_name(0)}", flush=True)

    def stats(xs):
        a = np.asarray(xs, float)
        return np.percentile(a, 50), np.percentile(a, 10), np.percentile(a, 90)

    def ids_for(m, n):
        g = torch.Generator().manual_seed(0)
        return torch.randint(1000, m.config.vocab_size - 1, (1, n), generator=g).to(dev)

    def fwd(m, **kw):
        try:
            return m(logits_to_keep=1, **kw)
        except TypeError:
            return m(**kw)

    @torch.inference_mode()
    def prefill_ms(m, ids):
        def once():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fwd(m, input_ids=ids, use_cache=True)
            out.logits[:, -1].argmax(-1)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e)
        for _ in range(WARMUP):
            once()
        return [once() for _ in range(REPEATS)]

    @torch.inference_mode()
    def decode_ms(m, ids, n):
        out = fwd(m, input_ids=ids, use_cache=True)
        nxt, pkv = out.logits[:, -1].argmax(-1, keepdim=True), out.past_key_values
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        for s, e in ev:
            s.record()
            o = m(input_ids=nxt, past_key_values=pkv, use_cache=True)
            nxt, pkv = o.logits[:, -1].argmax(-1, keepdim=True), o.past_key_values
            e.record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in ev]

    n = 4096
    a, b = torch.randn(n, n, device=dev, dtype=torch.float16), torch.randn(n, n, device=dev, dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
    m_flops = 2 * n ** 3 * 20 / (time.perf_counter() - t)
    x = torch.empty(2 ** 27, device=dev).normal_()      # 512 MB
    y = torch.empty_like(x)
    y.copy_(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    m_bw = 2 * x.numel() * 4 * 10 / (time.perf_counter() - t)
    del a, b, x, y
    print(f"[{label}] measured {m_flops / 1e12:.0f} TFLOP/s, {m_bw / 1e9:.0f} GB/s", flush=True)

    rows = []
    for key, repo in MODELS.items():
        try:
            m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, attn_implementation="sdpa")
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, attn_implementation="sdpa")
        m = m.to(dev).eval()
        P = sum(p.numel() for p in m.parameters())
        p50, p10, p90 = stats(prefill_ms(m, ids_for(m, PF_L)))
        decode_ms(m, ids_for(m, 128), 6)                     # shape warmup
        d50, d10, d90 = stats(decode_ms(m, ids_for(m, 128), 24)[3:])
        rows.append(dict(device=label, model=key, prefill_L=PF_L, prefill_ms=p50, prefill_ms_p10=p10, prefill_ms_p90=p90,
                         decode_ms=d50, decode_ms_p10=d10, decode_ms_p90=d90, params=P, weight_bytes=P * 2,
                         spec_flops=spec_flops, spec_bw=spec_bw, measured_flops=m_flops, measured_bw=m_bw))
        print(f"[{label}] {key}: prefill L={PF_L} {p50:.1f} ms | decode {d50:.1f} ms/token", flush=True)
        del m
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).to_csv(index=False)


_fns = {}
for _label, (_gpu, _, _) in GPUS.items():
    def _make(label):
        def fn():
            return _e8(label)
        fn.__name__ = f"e8_{label}"
        return fn
    _fns[_label] = app.function(gpu=_gpu, image=image, timeout=1800, name=f"e8_{_label}", serialized=True)(_make(_label))


@app.local_entrypoint()
def main(gpus: str = "a100,h100"):
    import pathlib
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    for g in [x.strip().lower() for x in gpus.split(",")]:
        csv = _fns[g].remote()
        (out / f"{g}_E8.csv").write_text(csv)
        print(f"wrote results/{g}_E8.csv")
