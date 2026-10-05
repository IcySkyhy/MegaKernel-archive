# Non-hollowness check (reference vs reference, different input)

- A: `reference/layer3_chunk_m64` extra={'layer_idx': 3, 'layer_type': 'full_attention', 'm': 64, 'dump_positions': [0, 63], 'input_mode': 'synthetic', 'pending': 'none', 'seed': 0, 'sigma': 0.05, 'moe_precision': 'fp32', 'start_pos': 0}
- B: `reference/layer3_chunk_m64_seed1` extra={'layer_idx': 3, 'layer_type': 'full_attention', 'm': 64, 'dump_positions': [0, 63], 'input_mode': 'synthetic', 'pending': 'none', 'seed': 1, 'sigma': 0.05, 'moe_precision': 'fp32', 'start_pos': 0}

| segment | changed elements | max abs delta | verdict |
|---|---|---|---|
| `attn_hc.block_input` | 163723/163840 | 4.03125 | moves |
| `attn_hc.hidden` | 654772/655360 | 0.330078 | moves |
| `attn_hc.injection` | 256/256 | 14.0625 | moves |
| `layer.out.block_output` | 163655/163840 | 0.400391 | moves |
| `layer.out.hidden` | 654728/655360 | 2.55078 | moves |
| `layer.out.injection` | 255/256 | 9.14648 | moves |
| `mlp_hc.block_input` | 163671/163840 | 10.5938 | moves |
| `mlp_hc.hidden` | 654728/655360 | 2.55078 | moves |
| `mlp_hc.injection` | 255/256 | 9.14648 | moves |
| `moe.block_input` | 163671/163840 | 10.5938 | moves |
| `moe.out` | 163655/163840 | 0.400391 | moves |
| `moe.routed_out` | 163687/163840 | 0.148193 | moves |
| `moe.router_logits` | 32732/32768 | 4.60687 | moves |
| `moe.shared_out` | 163636/163840 | 0.381836 | moves |
| `moe.topk_ids` | 639/640 | 476 | moves |
| `moe.topk_weights` | 640/640 | 0.188064 | moves |
| `qsa.attn_out` | 392883/393216 | 1.98242 | moves |
| `qsa.block_indices` | 412/32768 | 13 | moves |
| `qsa.compressed_key` | 2047/2048 | 6.17188 | moves |
| `qsa.gate` | 392855/393216 | 2.95703 | moves |
| `qsa.index_k` | 8188/8192 | 3.57031 | moves |
| `qsa.index_logits` | 496/1024 | 59.4033 [non-finite pairs skipped: 528/1024] | moves |
| `qsa.index_q` | 32738/32768 | 6.75 | moves |
| `qsa.k` | 32737/32768 | 7.76562 | moves |
| `qsa.q` | 392876/393216 | 10.2188 | moves |
| `qsa.token_indices` | 1648/131328 | 52 | moves |
| `qsa.v` | 32738/32768 | 3.05469 | moves |

**27/27 output segments react to the input change** (0 would mean the dump is hollow)
RESULT: OK (27/27 segments compared, 27 moved)
