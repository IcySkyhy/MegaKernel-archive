# Nemotron 3 Embed Megakernel
Stupidly fast for short inputs, handles batching well. Use Gigatoken with this for hilariously large document/code retrieval. Purely optimized for RTX 5090/Blackwell, some modifications necessary to get it running on any other devices. 


Architecture Diagram
```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ NVIDIA NEMOTRON-3-EMBED-1B-NVFP4 · PERSISTENT MEGAKERNEL (sm_120a)                               │
│ Grid: 160 CTAs x 512 threads (144 Compute CTAs + 16 Layer L+1 Asynchronous Weight Streamers)     │
│ Shared Memory: Unified reusable arena (NK_XPAD bank skewing / padded FP4 + scale layouts)        │
│ Inter-phase Sync: AtomicGridSync (Device-Scope Acq/Rel Barrier or Cooperative Launch)            │
└────────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                                 │
  [Token IDs (d_ids)]            │ (Launch Entry / Layer 0)
                 │               ▼
                 ├────────► ┌──────────────────────────────────────────────────────────────────────┐
                 │          │ Phase A1 / Layer 0 Embedding Lookup & Norm Staging                   │
                 │          │ • Layer 0: bf16 embed lookup (__ldg) + input RMSNorm (eps 1e-5)      │
                 │          │ • Dual-CTA Path: nk_input_rstd_pass (rstd vector + float residual)   │
                 │          │ • Base Path: Redundant per-CTA nk_norm_tile -> f16 s_x               │
                 │          └──────────────────────────────┬───────────────────────────────────────┘
                 │                                         │
┌────────────────┴─────────────────────────────────────────▼───────────────────────────────────────┐
│ 16 PERSISTENT TRANSFORMER LAYERS (MINISTRAL3: H=2048, FFN=6144, Q=24, KV=8, D=128)               │
│                                                                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
│ │ Phase A: QKV Projection (3072 Q / 1024 K / 1024 V)                                          │  │
│ │ • Quantization: Direct in-register PTX pack (cvt.rn.satfinite.e2m1x2.f32 + mov.b32)         │  │
│ │ • W4A4 Tensor Core: mma.m16n8k64.mxf4nvf4.scale_vec::4X against padded s_bq + s_sf          │  │
│ │ • W4A16 Fallback: mma.m16n8k16.f32 with in-register nk_dec_e2m1 * e4m3 scale decode         │  │
│ │ • K-Split Reduction: 16-warp KSPLIT combine -> g_q [f16], g_k [f16], v_cache [H, S, D]      │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ [AtomicGridSync]                                │
│ ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐  │
│ │ Phase B: K YaRN RoPE & Cache Staging                                                        │  │
│ │ • 1 warp per (token, kv_head) -> local position RoPE rotation                               │  │
│ │ • Coalesced strided write to k_cache [NUM_KV_HEADS, max_seq, HEAD_DIM]                      │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ [AtomicGridSync]                                │
│ ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐  │
│ │ Phase C: Bidirectional Attention Engine (3:1 GQA)                                           │  │
│ │ • Local Q YaRN RoPE + Llama-4 Position Temperature Scaling (nk_l4_temperature)              │  │
│ │ • Path 1 (Exact Fused GQA): 1 warp per (token, kv_head), loads K/V once, updates 3 Q heads  │  │
│ │ • Path 2 (TMA/MMA FlashAttn): 16-query CTA tiles, cp.async.bulk + mbarrier K/V staging,     │  │
│ │   ldmatrix.m8n8.x2.trans for P@V, online FP32 softmax -> g_attn [f16]                       │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ [AtomicGridSync]                                │
│ ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐  │
│ │ Phase D1: O-Projection & Residual Accumulation (K=3072 -> H=2048)                           │  │
│ │ • Staging: Single-stage packed FP4 tile (s_bq/s_sf) or T=2 supertile (32 tokens)            │  │
│ │ • MMA Engine: m16n8k64 (W4A4) / m16n8k16 (W4A16) over 192-K slices                          │  │
│ │ • Accumulation: KSPLIT combine + float g_residual += (v * o_eps)                            │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ [AtomicGridSync]                                │
│ ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐  │
│ │ Phase D2: Post-Norm & Fused SwiGLU MLP (Intermediate=6144)                                  │  │
│ │ • Staging: nk_residual_rstd_pass / post-norm -> on-the-fly quant to s_bq/s_sf               │  │
│ │ • Dual MMA Stream: Gate and Up GEMMs evaluated concurrently or via compact 1-KB gate bridge │  │
│ │ • In-Thread Activation: nk_silu(gate * gate_eps) * (up * up_eps) -> g_mlp [f16]             │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │ [AtomicGridSync]                                │
│ ┌──────────────────────────────────────────────▼──────────────────────────────────────────────┐  │
│ │ Phase D3: Down Projection & Residual Add (K=6144 -> H=2048)                                 │  │
│ │ • Staging: Full 6144 stage OR Dual-CTA 2x 3072-K half-stages (g_down_partial workspace)     │  │
│ │ • Optional Rowpair: 2x 8-warp groups process adjacent 16-row units concurrently             │  │
│ │ • Writeback: hidden_out [f16] = __float2half(v * down_eps + g_residual)                     │  │
│ └──────────────────────────────────────────────┬──────────────────────────────────────────────┘  │
│                                                │                                                 │
│       [Loop Layer 0..15: hidden_out feeds Phase A1 input norm of Layer L+1]                      │
│       [Background: CTAs 144..159 stream Layer L+1 weights to L2 via __ldg(uint4)]                │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │ [AtomicGridSync after Layer 15]
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Finalize: Pooling & Normalization                                                                │
│ • F1 (Token Norm): nk_rstd_pass calculates per-token RMSNorm scale factors (g_rstd)              │
│ • F2 (Masked Pool): 512-thread CTA per sequence accumulates w_final * sum(hidden * rstd)         │
│ • L2 Normalization: In-CTA warp reduction computes 1/sqrt(sum(acc^2)) -> Unit 2048-D Embedding   │
└────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                               [d_out / h_out: [Batch, 2048] FP32]
```

## Caveats

Heavily optimized mostly for sentence to paragraph sized chunks. Larger 4096 token workloads just barely beat TEI or VLLM, so dependent on how you're chunking documents this can be a boon or a curse. 

This is optimized for TPS in document embedding tasks, not for query time speedups. Use the fused GQA scalar path (NK_FUSED_GQA_ATTN = 1) and cooperative launch if you want speedy single queries.  

Pack documents continuously using the packed batch API (nk_embed_batch_dev_ex). Keep total_len near your allocated max_seq capacity to saturate all 144 compute blocks. Enable the T=2 supertile flags (NK_FP4_T2_OPROJ = 1 or NK_FP4_T2_GATEUP = 1) to double the token tile size per weight pass across large batches   

---
Lynn Hughes. © 2026. All rights reserved.
MIT License (https://opensource.org/licenses/MIT)
