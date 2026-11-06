"""Sanity checks: the cached prefill/decode path must reproduce the plain full forward pass,
and MLA's absorbed decode must equal MLA's naive decode. Run: python test_equivalence.py"""
import torch

from attention import Config, MoEFFN, build_attention

torch.manual_seed(0)
cfg = Config(d_model=256, n_heads=8, head_dim=32, n_kv_heads=2, kv_lora_rank=64,
             qk_nope_dim=32, qk_rope_dim=16, v_head_dim=32, d_ff=128, n_experts=8, top_k=2)
B, S, N = 2, 24, 6
x = torch.randn(B, S + N, cfg.d_model)

with torch.no_grad():
    for name in ["MHA", "GQA", "MQA", "MLA"]:
        m = build_attention(name, cfg).eval()
        ref = m.forward_full(x)                                  # one big causal pass over S+N tokens
        out_p, cache = m.prefill(x[:, :S])                       # prompt
        outs = [out_p]
        for t in range(N):                                       # then token-by-token decode
            o, cache = m.decode(x[:, S + t: S + t + 1], cache)
            outs.append(o)
        got = torch.cat(outs, 1)
        err = (got - ref).abs().max().item()
        print(f"{name}: prefill+decode vs full forward  max|diff| = {err:.2e}")
        assert err < 1e-4

    mla = build_attention("MLA", cfg).eval()
    _, cache = mla.prefill(x[:, :S])
    a, _ = mla.decode(x[:, S:S + 1], dict(cache), absorbed=True)
    b, _ = mla.decode(x[:, S:S + 1], dict(cache), absorbed=False)
    print(f"MLA absorbed vs naive decode  max|diff| = {(a - b).abs().max().item():.2e}")
    assert (a - b).abs().max() < 1e-4

    moe = MoEFFN(cfg).eval()
    full = moe(x)
    tok = torch.cat([moe(x[:, i:i + 1]) for i in range(x.shape[1])], 1)   # MoE is per-token => same answer
    print(f"MoE batched vs token-by-token max|diff| = {(full - tok).abs().max().item():.2e}")
    assert (full - tok).abs().max() < 1e-4
print("ALL OK")
