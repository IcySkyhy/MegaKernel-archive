# Qwen3-4B-Instruct-2507 decode: Hoid megakernel vs vLLM 0.29.0, SGLang 0.5.21, TRT-LLM 1.2.1

NVIDIA H200, BF16, TP1. Every request generates 1000 tokens greedily; tok/s is aggregate decode throughput (all rows) over the decode window, median of 4 fresh processes per engine.

| context | batch | Hoid tok/s | vLLM tok/s | SGLang tok/s | TRT-LLM tok/s | Hoid / vLLM | Hoid / SGLang | Hoid / TRT-LLM | Hoid TPOT p50 ms | vLLM TPOT p50 ms | SGLang TPOT p50 ms | TRT-LLM TPOT p50 ms | spread Hoid / vLLM / SGLang / TRT-LLM | gate (first 10 tokens vs FP32) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8k | 1 | 316.9 | 255.3 | 269.9 | 218.6 | **1.242x** | **1.174x** | **1.450x** | 3.154 | 3.917 | 3.705 | 4.558 | 0.1% / 0.2% / 0.0% / 0.7% | PASS |
| 8k | 2 | 565.4 | 471.9 | 481.1 | 396.0 | **1.198x** | **1.175x** | **1.428x** | 3.535 | 4.239 | 4.160 | 4.984 | 0.1% / 0.3% / 0.0% / 0.8% | PASS |
| 8k | 4 | 948.8 | 830.1 | 836.9 | 710.9 | **1.143x** | **1.134x** | **1.335x** | 4.217 | 4.818 | 4.785 | 5.602 | 0.2% / 0.4% / 0.0% / 0.2% | PASS |
| 8k | 8 | 1406.1 | 1324.0 | 1348.5 | 1142.9 | **1.062x** | **1.043x** | **1.230x** | 5.690 | 6.043 | 5.932 | 6.924 | 0.2% / 0.2% / 0.1% / 0.3% | PASS |

Hoid vs each engine on the timed benchmark rows (first differing generated token, or "same" for all 1000):

- 8k b1 vs vLLM: row0: same
- 8k b1 vs SGLang: row0: same
- 8k b1 vs TRT-LLM: row0: same
- 8k b2 vs vLLM: row0: same, row1: same
- 8k b2 vs SGLang: row0: same, row1: same
- 8k b2 vs TRT-LLM: row0: same, row1: same
- 8k b4 vs vLLM: row0: same, row1: same, row2: same, row3: same
- 8k b4 vs SGLang: row0: same, row1: same, row2: same, row3: same
- 8k b4 vs TRT-LLM: row0: same, row1: same, row2: same, row3: same
- 8k b8 vs vLLM: row0: same, row1: same, row2: same, row3: same, row4: same, row5: same, row6: same, row7: same
- 8k b8 vs SGLang: row0: same, row1: same, row2: same, row3: same, row4: same, row5: same, row6: same, row7: same
- 8k b8 vs TRT-LLM: row0: same, row1: same, row2: same, row3: same, row4: same, row5: same, row6: same, row7: same
