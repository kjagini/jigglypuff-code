# Auto-generated results (python experiments.py)

## Validation: cost model vs torch (config d=1024, H=16, dh=64, S=L=512, fp32)

| variant | prefill FLOPs torch | model | decode FLOPs torch | model | cache bytes torch | model |
|---|---|---|---|---|---|---|
| MHA | 5.369e+09 | 5.369e+09 | 1.049e+07 | 1.049e+07 | 4194304 | 4194304 |
| GQA | 3.758e+09 | 3.758e+09 | 7.344e+06 | 7.344e+06 | 1048576 | 1048576 |
| MQA | 3.355e+09 | 3.355e+09 | 6.558e+06 | 6.558e+06 | 262144 | 262144 |
| MLA | 4.463e+09 | 4.463e+09 | 1.082e+07 | 1.082e+07 | 327680 | 327680 |

## Attention layer, 7B-class shapes, bf16, B=1, S=L=4096 (analytical)

| variant | KV B/token/layer | prefill FLOP/B | decode FLOP/B | decode bytes | KV share of decode bytes |
|---|---|---|---|---|---|
| MHA | 16384 | 2560 | 1.00 | 192 MB | 33% |
| GQA | 4096 | 2867 | 1.50 | 96 MB | 17% |
| MQA | 512 | 3041 | 1.91 | 68 MB | 3% |
| MLA | 1152 | 3371 | 3.76 | 97 MB | 5% |

MLA with naive (decompress-every-step) decode: 339.53 FLOP/B, 3.454e+10 FLOPs vs absorbed 3.823e+08.  MLA with absorbed prefill: 9.815e+11 FLOPs vs naive 5.691e+11.

## Measured CPU, ms per token at 1024 tokens (fp32, one layer, batch 1)

| variant | prefill ms/token | decode ms/token | decode / prefill |
|---|---|---|---|
| MHA | 0.1190 | 1.075 | 9x |
| GQA | 0.0874 | 0.802 | 9x |
| MQA | 0.0795 | 0.761 | 10x |
| MLA | 0.0879 | 0.918 | 10x |
