# Qwen3-4B-Instruct-2507 decode: Hoid megakernel vs vLLM 0.29.0, SGLang 0.5.21, TRT-LLM 1.2.1

NVIDIA H200, BF16, TP1. Every request generates 1000 tokens greedily; tok/s is aggregate decode throughput (all rows) over the decode window, median of 4 fresh processes per engine.

| context | batch | Hoid tok/s | vLLM tok/s | SGLang tok/s | TRT-LLM tok/s | Hoid / vLLM | Hoid / SGLang | Hoid / TRT-LLM | Hoid TPOT p50 ms | vLLM TPOT p50 ms | SGLang TPOT p50 ms | TRT-LLM TPOT p50 ms | spread Hoid / vLLM / SGLang / TRT-LLM | gate (first 10 tokens vs FP32) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8k | 1 | 317.4 | 252.9 | 289.0 | 219.0 | **1.255x** | **1.098x** | **1.449x** | 3.147 | 3.954 | 3.459 | 4.555 | 1.8% / 0.7% / 0.1% / 0.1% | PASS |
| 8k | 2 | 568.5 | 470.2 | 481.5 | 405.3 | **1.209x** | **1.181x** | **1.403x** | 3.517 | 4.256 | 4.153 | 4.896 | 0.6% / 0.4% / 0.0% / 0.4% | PASS |
| 8k | 4 | 950.9 | 825.0 | 836.8 | 717.5 | **1.153x** | **1.136x** | **1.325x** | 4.207 | 4.848 | 4.780 | 5.539 | 0.4% / 0.8% / 0.0% / 0.2% | PASS |
| 8k | 8 | 1403.7 | 1317.9 | 1355.1 | 1162.1 | **1.065x** | **1.036x** | **1.208x** | 5.697 | 6.068 | 5.897 | 6.865 | 0.3% / 0.5% / 0.1% / 0.9% | PASS |

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
