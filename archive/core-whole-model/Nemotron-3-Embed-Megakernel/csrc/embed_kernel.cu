/**
 * Fused single-kernel NVFP4 text embedding for
 * nvidia/Nemotron-3-Embed-1B-NVFP4 on RTX 5090 (sm_120a).
 *
 * Everything — embedding lookup, 16 bidirectional transformer layers
 * (RMSNorm, QKV, YaRN RoPE, full-window attention, O-proj, SwiGLU MLP),
 * per-token final RMSNorm, masked mean pooling and L2 normalization — runs
 * inside one persistent kernel launch per (packed batch of) text(s).
 *
 * Weights stay in checkpoint NVFP4 form end-to-end: packed e2m1 nibble
 * pairs (uint8, low nibble = even element) with one e4m3 scale per
 * 16-element group and a per-tensor fp32 scale.  The mma A-fragment loader
 * reads the packed bytes directly from global memory and decodes them with
 * single-instruction cvt (byte -> f16x2), multiplies by the group scale in
 * f16 (exact: e2m1 x e4m3 products fit f16's mantissa), and feeds
 * mma.m16n8k16.f32.f16.f16.f32.  The per-tensor scale folds into the
 * combine epilogue.  The 16-element NVFP4 group size aligning exactly with
 * the mma k-chunk is what makes this cheap: one scale byte per fragment row.
 *
 * Model: Ministral3 encoder, hidden 2048, ffn 6144, 24 Q / 8 KV heads,
 * head_dim 128, 16 layers, vocab 131072 (embed table bf16, unquantized),
 * rms eps 1e-5.
 * 
 * MIT License (https://opensource.org/licenses/MIT)
 * Developed By: Lynn Hughes
 * 2026, All rights reserved.
 */

#ifndef NK_SAFE_GRID_SYNC
#define NK_SAFE_GRID_SYNC 0
#endif

#include <cmath>
#include <cstdio>
#include <new>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// Engine select: 1 = native block-scaled fp4 tensor cores (W4A4, activations
// quantized to NVFP4 per 16-group on the fly; ~6x fewer mainloop
// instructions), 0 = f16 mma with in-register NVFP4 dequant (W4A16, highest
// fidelity to the bf16 reference).
#ifndef NK_FP4MMA
#define NK_FP4MMA 1
#endif

// Experimental CTA-tiled FP16 MMA attention.  Keep the scalar warp path as
// the default until this path clears the accuracy and throughput gates.
#ifndef NK_MMA_ATTN
#define NK_MMA_ATTN 0
#endif
#ifndef NK_MMA_ATTN_FROM_LAYER
#define NK_MMA_ATTN_FROM_LAYER 12
#endif
#ifndef NK_MMA_ATTN_MIN_SEQ
#define NK_MMA_ATTN_MIN_SEQ 1024
#endif
#ifndef NK_MMA_ATTN_KN
#define NK_MMA_ATTN_KN 16
#endif
#ifndef NK_MMA_ATTN_Q_WRITE
#define NK_MMA_ATTN_Q_WRITE 1
#endif
#ifndef NK_MMA_ATTN_TMA
#define NK_MMA_ATTN_TMA 0
#endif

// Exact scalar-attention path: one warp owns a (token, KV-head) task
// and advances all three grouped-query heads while loading K/V once.  Small
// solo launches retain the wider warp-per-Q-head schedule for parallelism.
#ifndef NK_FUSED_GQA_ATTN
#define NK_FUSED_GQA_ATTN 0
#endif
#ifndef NK_FUSED_GQA_MIN_TOTAL
#define NK_FUSED_GQA_MIN_TOTAL 256
#endif
#ifndef NK_FUSED_GQA_Q_ROUNDTRIP
#define NK_FUSED_GQA_Q_ROUNDTRIP 1
#endif

// =============================================================================
// Model constants (Nemotron-3-Embed-1B / Ministral3 pruned encoder)
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 2048;
constexpr int INTERMEDIATE_SIZE = 6144;
constexpr int NUM_Q_HEADS = 24;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 3072
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024
constexpr float NK_RMS_EPS = 1e-5f;
constexpr float NK_L4_BETA = 0.1f;        // llama_4_scaling_beta
constexpr float NK_L4_ORIG_MAX = 16384.f; // original_max_position_embeddings

#ifndef NK_NUM_BLOCKS
#define NK_NUM_BLOCKS 160
#endif
#ifndef NK_BLOCK_SIZE
#define NK_BLOCK_SIZE 512
#endif
#ifndef NK_STREAMER_BLOCKS
#define NK_STREAMER_BLOCKS 16
#endif
#ifndef NK_STREAM_AHEAD
#define NK_STREAM_AHEAD -1
#endif
#ifndef NK_MIN_BLOCKS_PER_SM
#define NK_MIN_BLOCKS_PER_SM 1
#endif
static_assert(NK_STREAM_AHEAD >= -1,
              "NK_STREAM_AHEAD < -1 deadlocks the final layer streamer");
#ifndef NK_TOKEN_TILE
#define NK_TOKEN_TILE 16
#endif
#ifndef NK_FP4_T2_OPROJ
#define NK_FP4_T2_OPROJ 0
#endif
#ifndef NK_FP4_T2_GATEUP
#define NK_FP4_T2_GATEUP 0
#endif
#ifndef NK_PACKED_WEIGHT_LAYOUT
#define NK_PACKED_WEIGHT_LAYOUT 0
#endif
#ifndef NK_DUAL_CTA
#define NK_DUAL_CTA 0
#endif
#ifndef NK_COOPERATIVE_LAUNCH
#define NK_COOPERATIVE_LAUNCH NK_SAFE_GRID_SYNC
#endif
#ifndef NK_PARALLEL_COMPACT_COMBINE
#define NK_PARALLEL_COMPACT_COMBINE NK_DUAL_CTA
#endif
#ifndef NK_PARALLEL_PROJECTION_COMBINE
#define NK_PARALLEL_PROJECTION_COMBINE NK_DUAL_CTA
#endif
#ifndef NK_PARALLEL_COMBINE_WARPS
#define NK_PARALLEL_COMBINE_WARPS 8
#endif
#ifndef NK_DUAL_DOWN_ROWPAIR
#define NK_DUAL_DOWN_ROWPAIR NK_DUAL_CTA
#endif
#ifndef NK_PTX_PACK_QUANT
#define NK_PTX_PACK_QUANT 1
#endif

constexpr int NK_NUM_WARPS = NK_BLOCK_SIZE / WARP_SIZE;
constexpr int NK_COMPUTE_BLOCKS = NK_NUM_BLOCKS - NK_STREAMER_BLOCKS;
constexpr int TOKEN_TILE = NK_TOKEN_TILE;

// f16 elements of padding per activation row in shared memory: row stride in
// 4-byte words == 4 (mod 32) for K = 1536/2048, so per-thread mma B-fragment
// loads cover all 32 banks conflict-free (same rule as harrier).
constexpr int NK_XPAD = 8;

static_assert(TOKEN_TILE == 16,
              "mma path: 16-token tile == two n=8 halves of m16n8k16");
static_assert(NK_BLOCK_SIZE == 512,
              "pool finalization requires exactly 512 threads");
static_assert(TOKEN_TILE <= NK_NUM_WARPS,
              "norm phase maps one warp per tile token");
static_assert(NK_FP4_T2_OPROJ == 0 || NK_FP4_T2_OPROJ == 1,
              "NK_FP4_T2_OPROJ must be 0 or 1");
static_assert(NK_FP4_T2_GATEUP == 0 || NK_FP4_T2_GATEUP == 1,
              "NK_FP4_T2_GATEUP must be 0 or 1");
static_assert(NK_MMA_ATTN_KN == 16 || NK_MMA_ATTN_KN == 32,
              "NK_MMA_ATTN_KN must be 16 or 32");
static_assert(NK_MMA_ATTN_Q_WRITE == 0 || NK_MMA_ATTN_Q_WRITE == 1,
              "NK_MMA_ATTN_Q_WRITE must be 0 or 1");
static_assert(NK_MMA_ATTN_TMA == 0 || NK_MMA_ATTN_TMA == 1,
              "NK_MMA_ATTN_TMA must be 0 or 1");
static_assert(!NK_MMA_ATTN_TMA || NK_MMA_ATTN,
              "TMA attention requires the tiled MMA path");
static_assert(!NK_MMA_ATTN_TMA || NK_MMA_ATTN_KN == 16,
              "TMA attention prototype is shape-locked to KN=16");
static_assert(NK_FUSED_GQA_ATTN == 0 || NK_FUSED_GQA_ATTN == 1,
              "NK_FUSED_GQA_ATTN must be 0 or 1");
static_assert(NK_FUSED_GQA_MIN_TOTAL >= 1,
              "NK_FUSED_GQA_MIN_TOTAL must be positive");
static_assert(NK_FUSED_GQA_Q_ROUNDTRIP == 0 ||
                  NK_FUSED_GQA_Q_ROUNDTRIP == 1,
              "NK_FUSED_GQA_Q_ROUNDTRIP must be 0 or 1");
static_assert(!NK_PACKED_WEIGHT_LAYOUT || NK_FP4MMA,
              "packed weight layout is only implemented by the W4A4 engine");
static_assert(NK_DUAL_CTA == 0 || NK_DUAL_CTA == 1,
              "NK_DUAL_CTA must be 0 or 1");
static_assert(NK_SAFE_GRID_SYNC == 0 || NK_SAFE_GRID_SYNC == 1,
              "NK_SAFE_GRID_SYNC must be 0 or 1");
static_assert(NK_COOPERATIVE_LAUNCH == 0 || NK_COOPERATIVE_LAUNCH == 1,
              "NK_COOPERATIVE_LAUNCH must be 0 or 1");
static_assert(!NK_SAFE_GRID_SYNC || NK_COOPERATIVE_LAUNCH,
              "safe grid sync requires cooperative launch admission");
static_assert(NK_PARALLEL_COMPACT_COMBINE == 0 ||
                  NK_PARALLEL_COMPACT_COMBINE == 1,
              "NK_PARALLEL_COMPACT_COMBINE must be 0 or 1");
static_assert(NK_PARALLEL_PROJECTION_COMBINE == 0 ||
                  NK_PARALLEL_PROJECTION_COMBINE == 1,
              "NK_PARALLEL_PROJECTION_COMBINE must be 0 or 1");
static_assert(NK_PARALLEL_COMBINE_WARPS == 1 ||
                  NK_PARALLEL_COMBINE_WARPS == 2 ||
                  NK_PARALLEL_COMBINE_WARPS == 4 ||
                  NK_PARALLEL_COMBINE_WARPS == 8,
              "NK_PARALLEL_COMBINE_WARPS must be 1, 2, 4, or 8");
static_assert(NK_DUAL_DOWN_ROWPAIR == 0 || NK_DUAL_DOWN_ROWPAIR == 1,
              "NK_DUAL_DOWN_ROWPAIR must be 0 or 1");
static_assert(!NK_DUAL_DOWN_ROWPAIR || NK_DUAL_CTA,
              "down row pairing requires the dual-CTA layout");
static_assert(NK_PTX_PACK_QUANT == 0 || NK_PTX_PACK_QUANT == 1,
              "NK_PTX_PACK_QUANT must be 0 or 1");
static_assert(!NK_DUAL_CTA || NK_FP4MMA,
              "dual-CTA compact staging requires the W4A4 engine");
static_assert(!(NK_DUAL_CTA && NK_FP4_T2_OPROJ),
              "dual-CTA arena is incompatible with the T=2 O projection");
static_assert(NK_COMPUTE_BLOCKS > 16, "need compute blocks");

// NVFP4 linear: packed nibble pairs + per-16-group e4m3 scales.  The
// per-tensor fp32 scale (weight_scale_2) lives in NKLayerWeights and folds
// into the epilogue.
struct NKQuantW {
  const unsigned char *w; // [rows][K/2]  e2m1 pairs, low nibble first
  const unsigned char *s; // [rows][K/16] e4m3 group scales
};

struct NKLayerWeights {
  const __nv_bfloat16 *input_ln;  // [2048] bf16
  const __nv_bfloat16 *post_ln;   // [2048] bf16
  NKQuantW q, k, v, o, gate, up, down;
  float q_ws2, k_ws2, v_ws2, o_ws2, gate_ws2, up_ws2, down_ws2;
  // W4A4 engine parameters, derived host-side from ws2 + input_scale:
  // one activation-quant alpha per B tile (fused tiles share max alpha, the
  // TRT-LLM fused-QKV convention), epilogue scale = ws2 * alpha.
  float q_eps, k_eps, v_eps, o_eps, gate_eps, up_eps, down_eps;
  float aqkv, aqkv_i6;   // alpha, 1/(6*alpha) for the qkv input tile
  float ao, ao_i6;
  float agu, agu_i6;
  float adown, adown_i6;
};

// =============================================================================
// Atomic barrier for the persistent kernel. The promoted continuous build
// combines device-scope acquire/release operations with cooperative launch
// admission; legacy configurations retain their historical barrier.
// =============================================================================

struct AtomicGridSync {
  unsigned int *counter;
  unsigned int *generation;
  unsigned int nblocks;
  unsigned int local_gen;

  __device__ void sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int my_gen = local_gen;
#if NK_SAFE_GRID_SYNC
      // Each acq_rel RMW acquires the prior CTA's release, so the final CTA
      // transitively observes every participant before publishing completion.
      unsigned int arrived = __nv_atomic_fetch_add(
          counter, 1u, __NV_ATOMIC_ACQ_REL, __NV_THREAD_SCOPE_DEVICE);
      if (arrived == nblocks - 1) {
        __nv_atomic_store_n(counter, 0u, __NV_ATOMIC_RELAXED,
                            __NV_THREAD_SCOPE_DEVICE);
        __nv_atomic_store_n(generation, my_gen + 1, __NV_ATOMIC_RELEASE,
                            __NV_THREAD_SCOPE_DEVICE);
      } else {
        while (__nv_atomic_load_n(generation, __NV_ATOMIC_ACQUIRE,
                                  __NV_THREAD_SCOPE_DEVICE) <= my_gen) {
        }
      }
#else
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned int arrived = atomicAdd(counter, 1);
      if (arrived == nblocks - 1) {
        *counter = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(generation, 1);
      } else {
        volatile unsigned int *vgen = (volatile unsigned int *)generation;
        while (*vgen <= my_gen) {
        }
      }
#endif
      local_gen = my_gen + 1;
    }
    __syncthreads();
  }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float nk_l4_temperature(int pos) {
  // The Llama-4 factor is exactly 1 below the original context limit. Avoid
  // paying for floor/log in the entire currently supported 8K context range.
  if (pos < (int)NK_L4_ORIG_MAX)
    return 1.0f;
  return 1.0f + NK_L4_BETA *
                    logf(1.0f + floorf(pos * (1.0f / NK_L4_ORIG_MAX)));
}

__device__ __forceinline__ float nk_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

constexpr float LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_exp(float x) {
  return ptx_exp2(x * LOG2E);
}

__device__ __forceinline__ float nk_silu(float x) {
  return x * ptx_rcp(1.0f + fast_exp(-x));
}

// =============================================================================
// NVFP4 tensor-core dot engine.  Warp shape and combine identical to harrier
// v3 (each warp: one 16-row unit x 16-token tile x one K-slice;
// mma.m16n8k16 accumulates over K in-thread; KSPLIT per-warp partials
// combined via smem in fixed slice order = deterministic).  What changes is
// the A path: raw packed e2m1 bytes + e4m3 group scales ride the prefetch
// ring (6 bytes per thread-chunk instead of 16), decoded to f16 fragments
// at mma time:
//   frag reg a0 = cvt(byte{k pair 2b+0,2b+1 of row r0}) * scale(r0, chunk)
// Weight bytes/scales are launch-immutable -> __ldg (and sectors get 4-way
// reuse across the ring, so let them allocate in L1).
// =============================================================================

struct NKRawA {
  uchar4 w;  // bytes: {r0 k+0..1, r1 k+0..1, r0 k+8..9, r1 k+8..9} nibbles
  uchar2 s;  // e4m3 group scales {r0, r1}
};

struct NKFragA {
  unsigned a0, a1, a2, a3; // f16x2: {r0,klo} {r1,klo} {r0,khi} {r1,khi}
};

// Raw A load for the 16-row unit at w/s, k-chunk starting at element k0.
// PTX m16n8k16 A thread layout: rows lane/4 and lane/4+8, k pairs
// (lane%4)*2 and (lane%4)*2+8 -> packed bytes (k0/2)+(lane&3) and +4.
template <int K_DIM>
__device__ __forceinline__ NKRawA nk_load_raw(const NKQuantW &qw, int k0,
                                              int lane) {
  constexpr int KB = K_DIM / 2;  // packed bytes per row
  constexpr int SB = K_DIM / 16; // scale bytes per row
  const unsigned char *r0 = qw.w + (size_t)(lane >> 2) * KB + (k0 >> 1) +
                            (lane & 3);
  const unsigned char *r1 = r0 + (size_t)8 * KB;
  const unsigned char *s0 = qw.s + (size_t)(lane >> 2) * SB + (k0 >> 4);
  NKRawA r;
  r.w.x = __ldg(r0);
  r.w.y = __ldg(r1);
  r.w.z = __ldg(r0 + 4);
  r.w.w = __ldg(r1 + 4);
  r.s.x = __ldg(s0);
  r.s.y = __ldg(s0 + (size_t)8 * SB);
  return r;
}

__device__ __forceinline__ __half2 nk_dec_e2m1(unsigned char b) {
  __half2_raw r = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,
                                             __NV_E2M1);
  return *reinterpret_cast<__half2 *>(&r);
}

__device__ __forceinline__ NKFragA nk_decode(const NKRawA &raw) {
  unsigned short spair =
      (unsigned short)raw.s.x | ((unsigned short)raw.s.y << 8);
  __half2_raw sr =
      __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)spair, __NV_E4M3);
  __half2 s = *reinterpret_cast<__half2 *>(&sr);
  __half2 s00 = __half2half2(__low2half(s));   // row r0 scale
  __half2 s11 = __half2half2(__high2half(s));  // row r1 scale
  __half2 a0 = __hmul2(nk_dec_e2m1(raw.w.x), s00);
  __half2 a1 = __hmul2(nk_dec_e2m1(raw.w.y), s11);
  __half2 a2 = __hmul2(nk_dec_e2m1(raw.w.z), s00);
  __half2 a3 = __hmul2(nk_dec_e2m1(raw.w.w), s11);
  NKFragA f;
  f.a0 = *reinterpret_cast<unsigned *>(&a0);
  f.a1 = *reinterpret_cast<unsigned *>(&a1);
  f.a2 = *reinterpret_cast<unsigned *>(&a2);
  f.a3 = *reinterpret_cast<unsigned *>(&a3);
  return f;
}

__device__ __forceinline__ void nk_mma16816(float *c, const NKFragA &a,
                                            unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.a0), "r"(a.a1), "r"(a.a2), "r"(a.a3), "r"(b0), "r"(b1));
}

// Ring depth: divides the chunk count so ring indices stay compile-time.
__host__ __device__ constexpr int nk_ring_for(int nchunk) {
  return nchunk % 8 == 0 ? 8
         : nchunk % 6 == 0 ? 6
         : nchunk % 4 == 0 ? 4
         : nchunk % 3 == 0 ? 3
         : nchunk % 2 == 0 ? 2
                           : 1;
}

// Mainloop: NA independent 16-row NVFP4 A streams over a K window of
// SLICE_CHUNKS 16-element chunks against the padded f16 activation tile.
// a_off/b_off are element offsets of the window start in the weight rows /
// smem tile (they differ when the tile is staged in halves/quarters).
template <int K_DIM, int XSTRIDE, int SLICE_CHUNKS, int NA>
__device__ __forceinline__ void
nk_mma_slice(const NKQuantW (&wb)[NA], const __half *__restrict__ s_x,
             int a_off, int b_off, float (&c)[NA][2][4], int lane) {
  constexpr int RING = nk_ring_for(SLICE_CHUNKS);

  const __half *xb = s_x + (size_t)(lane >> 2) * XSTRIDE + (lane & 3) * 2 +
                     b_off;
  const __half *xb8 = xb + (size_t)8 * XSTRIDE;

  NKRawA ring[NA][RING];
#pragma unroll
  for (int i = 0; i < RING; i++)
#pragma unroll
    for (int s = 0; s < NA; s++)
      ring[s][i] = nk_load_raw<K_DIM>(wb[s], a_off + i * 16, lane);

#pragma unroll 1
  for (int kc0 = 0; kc0 < SLICE_CHUNKS; kc0 += RING) {
#pragma unroll
    for (int i = 0; i < RING; i++) {
      int kc = kc0 + i;
      // B fragments for both n-halves (tokens 0-7 / 8-15)
      unsigned b00 = *reinterpret_cast<const unsigned *>(xb + kc * 16);
      unsigned b01 = *reinterpret_cast<const unsigned *>(xb + kc * 16 + 8);
      unsigned b10 = *reinterpret_cast<const unsigned *>(xb8 + kc * 16);
      unsigned b11 = *reinterpret_cast<const unsigned *>(xb8 + kc * 16 + 8);
#pragma unroll
      for (int s = 0; s < NA; s++) {
        NKFragA a = nk_decode(ring[s][i]);
        if (kc + RING < SLICE_CHUNKS)
          ring[s][i] = nk_load_raw<K_DIM>(wb[s], a_off + (kc + RING) * 16,
                                          lane);
        nk_mma16816(c[s][0], a, b00, b01);
        nk_mma16816(c[s][1], a, b10, b11);
      }
    }
  }
}

// Shared epilogue index math: value j of n-half h sits at
// token h*8 + (lane%4)*2 + (j&1), local row (lane/4) + ((j>>1)*8).
__device__ __forceinline__ int nk_c_token(int h, int j, int lane) {
  return h * 8 + (lane & 3) * 2 + (j & 1);
}
__device__ __forceinline__ int nk_c_row(int j, int lane) {
  return (lane >> 2) + ((j >> 1) << 3);
}

// Block-centric phase shape: each unit (16 output rows) goes to one block
// whose 16 warps split K into NK_KSPLIT slices; per-warp partial C
// fragments are combined through shared memory in ascending slice order
// (deterministic), and the combining warp runs the epilogue.
#ifndef NK_KSPLIT
#define NK_KSPLIT 16
#endif
constexpr int NK_RG = NK_NUM_WARPS / NK_KSPLIT; // row-subgroups per unit
constexpr int NK_UNIT_ROWS = NK_RG * 16;
static_assert(NK_NUM_WARPS % NK_KSPLIT == 0, "warps = RG x KSPLIT");

template <typename StoreFn>
__device__ __forceinline__ void
nk_combine_partials(const float *__restrict__ s_part, int crg, int row0,
                    int tcount, int lane, StoreFn store) {
  for (int i = lane; i < 256; i += WARP_SIZE) {
    float v = 0.0f;
#pragma unroll
    for (int s = 0; s < NK_KSPLIT; s++)
      v += s_part[(size_t)(crg * NK_KSPLIT + s) * 256 + i];
    int src_lane = i >> 3;
    int h = (i >> 2) & 1;
    int j = i & 3;
    int t = nk_c_token(h, j, src_lane);
    if (t < tcount)
      store(row0 + crg * 16 + nk_c_row(j, src_lane), t, v);
  }
}

// NK_RG=1 kernels have 256 independent fragment outputs per unit. Spread
// those outputs over several warps while preserving each output's ascending
// K-split accumulation order exactly.
template <typename StoreFn>
__device__ __forceinline__ void nk_combine_partials_parallel(
    const float *__restrict__ s_part, int row0, int tcount, StoreFn store) {
  static_assert(NK_RG == 1, "parallel combine assumes one row group");
  constexpr int COMBINE_THREADS = NK_PARALLEL_COMBINE_WARPS * WARP_SIZE;
  if (threadIdx.x < COMBINE_THREADS) {
    for (int i = threadIdx.x; i < 256; i += COMBINE_THREADS) {
      float v = 0.0f;
#pragma unroll
      for (int s = 0; s < NK_KSPLIT; s++)
        v += s_part[(size_t)s * 256 + i];
      int src_lane = i >> 3;
      int h = (i >> 2) & 1;
      int j = i & 3;
      int t = nk_c_token(h, j, src_lane);
      if (t < tcount)
        store(row0 + nk_c_row(j, src_lane), t, v);
    }
  }
}

// =============================================================================
// W4A4 engine: mma.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X with
// ue4m3 scale factors — weights AND activations ride the tensor cores as
// packed e2m1, scales applied in-instruction.  Per 64-K chunk a thread
// issues 4 coalesced u32 weight loads + 1 scale u32 instead of v1's 16 byte
// loads + ~40 decode ALU ops.  Fragment mappings (verified on hardware by
// tests/test_smoke_fp4.py against the Python dequant):
//   A regs:  a0 = row lane/4,   k nibbles (lane%4)*8..+7  (u32 of the row)
//            a1 = row lane/4+8; a2/a3 = same at k+32
//   B regs:  col lane/4 (+8 for the 2nd n-half), same k mapping
//   SFA u32: 4 ue4m3 bytes (k-groups g0..g0+3) of row (lane&1)*8 + lane/4
//   SFB u32: 4 bytes of col lane/4 — lane&3 duplicates
//   C:       same 16x8 layout as the f16 engine -> combine code reused.
// =============================================================================

struct NKRaw4 {
  uint4 w;       // a0..a3 packed e2m1 u32 fragments
  unsigned sfa;  // 4 ue4m3 group scales for this thread's SF row
};

template <int K_DIM>
__device__ __forceinline__ NKRaw4 nk_load_raw4(const NKQuantW &qw, int k0,
                                               int lane) {
#if NK_PACKED_WEIGHT_LAYOUT
  // One 16-row output group is stored as [k64][xyzw][lane].  Each of
  // the four warp loads is therefore one contiguous 128-byte request.
  constexpr int WORDS_PER_CHUNK = 4 * WARP_SIZE;
  int kc = k0 >> 6;
  const unsigned *wp = reinterpret_cast<const unsigned *>(qw.w) +
                       kc * WORDS_PER_CHUNK + lane;
  NKRaw4 r;
  r.w.x = __ldg(wp + 0 * WARP_SIZE);
  r.w.y = __ldg(wp + 1 * WARP_SIZE);
  r.w.z = __ldg(wp + 2 * WARP_SIZE);
  r.w.w = __ldg(wp + 3 * WARP_SIZE);

  // Scales are [k64][row16].  The MMA fragment mapping uses every row
  // twice, so load the compact 16-row table and broadcast by source lane.
  const unsigned *sp = reinterpret_cast<const unsigned *>(qw.s) + kc * 16;
  unsigned own_sfa = __ldg(sp + (lane & 15));
  int sfrow = (lane & 1) * 8 + (lane >> 2);
  r.sfa = __shfl_sync(0xffffffff, own_sfa, sfrow);
  return r;
#else
  constexpr int KB = K_DIM / 2;
  constexpr int SB = K_DIM / 16;
  const unsigned char *r0 =
      qw.w + (size_t)(lane >> 2) * KB + (k0 >> 1) + (lane & 3) * 4;
  const unsigned char *r1 = r0 + (size_t)8 * KB;
  NKRaw4 r;
  r.w.x = __ldg(reinterpret_cast<const unsigned *>(r0));
  r.w.y = __ldg(reinterpret_cast<const unsigned *>(r1));
  r.w.z = __ldg(reinterpret_cast<const unsigned *>(r0 + 16));
  r.w.w = __ldg(reinterpret_cast<const unsigned *>(r1 + 16));
  int sfrow = (lane & 1) * 8 + (lane >> 2);
  r.sfa = __ldg(reinterpret_cast<const unsigned *>(
      qw.s + (size_t)sfrow * SB + (k0 >> 4)));
  return r;
#endif
}

__device__ __forceinline__ void nk_mma4(float *c, const NKRaw4 &a, unsigned b0,
                                        unsigned b1, unsigned sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.w.x), "r"(a.w.y), "r"(a.w.z), "r"(a.w.w), "r"(b0), "r"(b1),
        "r"(a.sfa), "h"((unsigned short)0), "h"((unsigned short)0), "r"(sfb),
        "h"((unsigned short)0), "h"((unsigned short)0));
}

// Mainloop over a K window of SLICE_CHUNKS 64-element chunks against the
// packed-NVFP4 activation tile (BPITCH bytes/row) and its scale tile
// (SFPITCH bytes/row).  a_off/b_off are element offsets of the window start.
template <int K_DIM, int BPITCH, int SFPITCH, int SLICE_CHUNKS, int NA>
__device__ __forceinline__ void
nk_mma4_slice(const NKQuantW (&wb)[NA], const unsigned char *__restrict__ s_bq,
              const unsigned char *__restrict__ s_sf, int a_off, int b_off,
              float (&c)[NA][2][4], int lane) {
  constexpr int RING = nk_ring_for(SLICE_CHUNKS);

  const unsigned char *xb =
      s_bq + (size_t)(lane >> 2) * BPITCH + (lane & 3) * 4 + (b_off >> 1);
  const unsigned char *xb8 = xb + (size_t)8 * BPITCH;
  const unsigned char *sp = s_sf + (size_t)(lane >> 2) * SFPITCH + (b_off >> 4);
  const unsigned char *sp8 = sp + (size_t)8 * SFPITCH;

  NKRaw4 ring[NA][RING];
#pragma unroll
  for (int i = 0; i < RING; i++)
#pragma unroll
    for (int s = 0; s < NA; s++)
      ring[s][i] = nk_load_raw4<K_DIM>(wb[s], a_off + i * 64, lane);

#pragma unroll 1
  for (int kc0 = 0; kc0 < SLICE_CHUNKS; kc0 += RING) {
#pragma unroll
    for (int i = 0; i < RING; i++) {
      int kc = kc0 + i;
      unsigned b00 = *reinterpret_cast<const unsigned *>(xb + kc * 32);
      unsigned b01 = *reinterpret_cast<const unsigned *>(xb + kc * 32 + 16);
      unsigned b10 = *reinterpret_cast<const unsigned *>(xb8 + kc * 32);
      unsigned b11 = *reinterpret_cast<const unsigned *>(xb8 + kc * 32 + 16);
      unsigned sfb0 = *reinterpret_cast<const unsigned *>(sp + kc * 4);
      unsigned sfb1 = *reinterpret_cast<const unsigned *>(sp8 + kc * 4);
#pragma unroll
      for (int s = 0; s < NA; s++) {
        NKRaw4 a = ring[s][i];
        if (kc + RING < SLICE_CHUNKS)
          ring[s][i] = nk_load_raw4<K_DIM>(wb[s], a_off + (kc + RING) * 64,
                                           lane);
        nk_mma4(c[s][0], a, b00, b01, sfb0);
        nk_mma4(c[s][1], a, b10, b11, sfb1);
      }
    }
  }
}

#if NK_FP4_T2_OPROJ
// T=2 mainloop for the O-projection experiment.  Four n=8 activation halves
// share each weight fragment load, producing 16 output rows x 32 tokens.
template <int K_DIM, int BPITCH, int SFPITCH, int SLICE_CHUNKS>
__device__ __forceinline__ void nk_mma4_slice_t2(
    const NKQuantW (&wb)[1], const unsigned char *__restrict__ s_bq,
    const unsigned char *__restrict__ s_sf, int a_off, int b_off,
    float (&c)[1][4][4], int lane) {
  constexpr int RING = nk_ring_for(SLICE_CHUNKS);

  const unsigned char *xb =
      s_bq + (size_t)(lane >> 2) * BPITCH + (lane & 3) * 4 + (b_off >> 1);
  const unsigned char *sp =
      s_sf + (size_t)(lane >> 2) * SFPITCH + (b_off >> 4);

  NKRaw4 ring[RING];
#pragma unroll
  for (int i = 0; i < RING; i++)
    ring[i] = nk_load_raw4<K_DIM>(wb[0], a_off + i * 64, lane);

#pragma unroll 1
  for (int kc0 = 0; kc0 < SLICE_CHUNKS; kc0 += RING) {
#pragma unroll
    for (int i = 0; i < RING; i++) {
      int kc = kc0 + i;
      NKRaw4 a = ring[i];
      if (kc + RING < SLICE_CHUNKS)
        ring[i] =
            nk_load_raw4<K_DIM>(wb[0], a_off + (kc + RING) * 64, lane);
#pragma unroll
      for (int h = 0; h < 4; h++) {
        const unsigned char *xbh = xb + (size_t)h * 8 * BPITCH;
        const unsigned char *sph = sp + (size_t)h * 8 * SFPITCH;
        unsigned b0 = *reinterpret_cast<const unsigned *>(xbh + kc * 32);
        unsigned b1 =
            *reinterpret_cast<const unsigned *>(xbh + kc * 32 + 16);
        unsigned sfb = *reinterpret_cast<const unsigned *>(sph + kc * 4);
        nk_mma4(c[0][h], a, b0, b1, sfb);
      }
    }
  }
}
#endif

#if NK_FP4_T2_GATEUP
// T=2 gate/up mainloop.  Both A streams remain fused so four n=8 activation
// halves share each gate and up weight load as well as each B-fragment load.
template <int K_DIM, int BPITCH, int SFPITCH, int SLICE_CHUNKS>
__device__ __forceinline__ void nk_mma4_slice_t2_na2(
    const NKQuantW (&wb)[2], const unsigned char *__restrict__ s_bq,
    const unsigned char *__restrict__ s_sf, int a_off, int b_off,
    float (&c)[2][4][4], int lane) {
  constexpr int RING = nk_ring_for(SLICE_CHUNKS);

  const unsigned char *xb =
      s_bq + (size_t)(lane >> 2) * BPITCH + (lane & 3) * 4 + (b_off >> 1);
  const unsigned char *sp =
      s_sf + (size_t)(lane >> 2) * SFPITCH + (b_off >> 4);

  NKRaw4 ring[2][RING];
#pragma unroll
  for (int i = 0; i < RING; i++)
#pragma unroll
    for (int s = 0; s < 2; s++)
      ring[s][i] = nk_load_raw4<K_DIM>(wb[s], a_off + i * 64, lane);

#pragma unroll 1
  for (int kc0 = 0; kc0 < SLICE_CHUNKS; kc0 += RING) {
#pragma unroll
    for (int i = 0; i < RING; i++) {
      int kc = kc0 + i;
#pragma unroll
      for (int h = 0; h < 4; h++) {
        const unsigned char *xbh = xb + (size_t)h * 8 * BPITCH;
        const unsigned char *sph = sp + (size_t)h * 8 * SFPITCH;
        unsigned b0 = *reinterpret_cast<const unsigned *>(xbh + kc * 32);
        unsigned b1 =
            *reinterpret_cast<const unsigned *>(xbh + kc * 32 + 16);
        unsigned sfb = *reinterpret_cast<const unsigned *>(sph + kc * 4);
#pragma unroll
        for (int s = 0; s < 2; s++)
          nk_mma4(c[s][h], ring[s][i], b0, b1, sfb);
      }
#pragma unroll
      for (int s = 0; s < 2; s++)
        if (kc + RING < SLICE_CHUNKS)
          ring[s][i] =
              nk_load_raw4<K_DIM>(wb[s], a_off + (kc + RING) * 64, lane);
    }
  }
}
#endif

// ---- on-the-fly activation quantization (the modelopt/vLLM W4A4 recipe:
// static per-tensor alpha from calibration, dynamic per-16-group ue4m3
// scale = e4m3(amax/(6*alpha)), values = rn-satfinite e2m1 of x/(sf*alpha)).

// Quantize one 16-element group given as 8 half2.
__device__ __forceinline__ void nk_quant16(const __half2 (&h)[8], float inv6a,
                                           float a, uint2 &packed,
                                           unsigned char &sfb) {
  float amax = 0.0f;
  float2 f[8];
#pragma unroll
  for (int p = 0; p < 8; p++) {
    f[p] = __half22float2(h[p]);
    amax = fmaxf(amax, fmaxf(fabsf(f[p].x), fabsf(f[p].y)));
  }
  sfb = __nv_cvt_float_to_fp8(amax * inv6a, __NV_SATFINITE, __NV_E4M3);
  __half_raw hr = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
  float sf_f = __half2float(*reinterpret_cast<__half *>(&hr));
  float inv = (sf_f > 0.0f) ? 1.0f / (sf_f * a) : 0.0f;
#if NK_PTX_PACK_QUANT
  // Convert and pack inside one PTX scope.  The CUDA intrinsic expresses the
  // same cvt instruction eight times, but returning eight scalar bytes and
  // reassembling them in C++ leaves extra merge/permute glue for ptxas.
  // mov.b32's brace form packs q0..q3 in low-to-high byte order.
  asm volatile(
      "{ .reg .b8 q0, q1, q2, q3, q4, q5, q6, q7;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q0, %3, %2;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q1, %5, %4;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q2, %7, %6;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q3, %9, %8;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q4, %11, %10;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q5, %13, %12;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q6, %15, %14;\n"
      "  cvt.rn.satfinite.e2m1x2.f32 q7, %17, %16;\n"
      "  mov.b32 %0, {q0, q1, q2, q3};\n"
      "  mov.b32 %1, {q4, q5, q6, q7}; }\n"
      : "=r"(packed.x), "=r"(packed.y)
      : "f"(f[0].x * inv), "f"(f[0].y * inv),
        "f"(f[1].x * inv), "f"(f[1].y * inv),
        "f"(f[2].x * inv), "f"(f[2].y * inv),
        "f"(f[3].x * inv), "f"(f[3].y * inv),
        "f"(f[4].x * inv), "f"(f[4].y * inv),
        "f"(f[5].x * inv), "f"(f[5].y * inv),
        "f"(f[6].x * inv), "f"(f[6].y * inv),
        "f"(f[7].x * inv), "f"(f[7].y * inv));
#else
  unsigned b[8];
#pragma unroll
  for (int p = 0; p < 8; p++)
    b[p] = __nv_cvt_float2_to_fp4x2(make_float2(f[p].x * inv, f[p].y * inv),
                                    __NV_E2M1, cudaRoundNearest);
  packed.x = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
  packed.y = b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24);
#endif
}

// Quantize a K-element f16 row (in shared memory or any readable source)
// into packed + scale rows.  One warp per row; lane g handles groups
// g, g+32, ... — 16 contiguous halves per group, conflict-free.
__device__ __forceinline__ void
nk_quant_row(const __half *__restrict__ row, int k16,
             unsigned char *__restrict__ bq_row,
             unsigned char *__restrict__ sf_row, float inv6a, float a,
             int lane) {
  for (int g = lane; g < k16; g += WARP_SIZE) {
    alignas(16) __half2 h[8];
    const uint4 *src = reinterpret_cast<const uint4 *>(row + g * 16);
    uint4 u0 = src[0], u1 = src[1];
    *reinterpret_cast<uint4 *>(&h[0]) = u0;
    *reinterpret_cast<uint4 *>(&h[4]) = u1;
    uint2 packed;
    unsigned char sfb;
    nk_quant16(h, inv6a, a, packed, sfb);
    *reinterpret_cast<uint2 *>(bq_row + g * 8) = packed;
    sf_row[g] = sfb;
  }
}

// Zero a packed+scale row pair (tokens beyond seq_len).
__device__ __forceinline__ void nk_zero_qrow(unsigned char *bq_row, int k16,
                                             unsigned char *sf_row, int lane) {
  for (int g = lane; g < k16; g += WARP_SIZE) {
    *reinterpret_cast<uint2 *>(bq_row + g * 8) = make_uint2(0, 0);
    sf_row[g] = 0;
  }
}

__device__ unsigned int nk_stream_sink;

// =============================================================================
// Dedicated weight streamers: pull layer L+1's NVFP4 bytes through L2 with
// real loads while compute blocks work on layer L.  Byte-granular version of
// harrier's streamer.
// =============================================================================

#if NK_STREAMER_BLOCKS > 0
__device__ __forceinline__ void nk_stream_range(const unsigned char *base,
                                                size_t bytes) {
  const uint4 *p = reinterpret_cast<const uint4 *>(base);
  size_t n = bytes / 16;
  size_t i = threadIdx.x;
  constexpr int STRIDE = NK_BLOCK_SIZE;
  unsigned int acc[8] = {};
  for (; i + 7 * STRIDE < n; i += 8 * STRIDE) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      uint4 a = __ldg(p + i + u * STRIDE);
      acc[u] ^= a.x ^ a.y ^ a.z ^ a.w;
    }
  }
  for (; i < n; i += STRIDE) {
    uint4 a = __ldg(p + i);
    acc[0] ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  unsigned int all = (acc[0] ^ acc[1]) ^ (acc[2] ^ acc[3]) ^
                     ((acc[4] ^ acc[5]) ^ (acc[6] ^ acc[7]));
  if (all == 0x8badf00du)
    nk_stream_sink = all;
}

__device__ __forceinline__ void nk_stream_slice(const unsigned char *base,
                                                size_t bytes, int sid) {
  size_t chunk =
      ((bytes + NK_STREAMER_BLOCKS - 1) / NK_STREAMER_BLOCKS + 63) & ~(size_t)63;
  size_t start = sid * chunk;
  if (start < bytes)
    nk_stream_range(base + start, min(chunk, bytes - start));
}

__device__ __forceinline__ void nk_stream_qw(const NKQuantW &qw, size_t rows,
                                             size_t k, int sid) {
  nk_stream_slice(qw.w, rows * (k / 2), sid);
  nk_stream_slice(qw.s, rows * (k / 16), sid);
}

__device__ void nk_streamer(const NKLayerWeights *__restrict__ layer_weights,
                            int num_layers,
                            unsigned int *__restrict__ kv_flag) {
  int sid = blockIdx.x - NK_COMPUTE_BLOCKS;
  for (int layer = 0; layer < num_layers; layer++) {
    if (threadIdx.x == 0) {
      volatile int *vf = (volatile int *)kv_flag;
      while (*vf < layer - 1 - NK_STREAM_AHEAD) {
      }
    }
    __syncthreads();
    const NKLayerWeights &w = layer_weights[layer];
    nk_stream_qw(w.q, Q_SIZE, HIDDEN_SIZE, sid);
    nk_stream_qw(w.k, KV_SIZE, HIDDEN_SIZE, sid);
    nk_stream_qw(w.v, KV_SIZE, HIDDEN_SIZE, sid);
    nk_stream_qw(w.o, HIDDEN_SIZE, Q_SIZE, sid);
    if (threadIdx.x == 0) {
      volatile int *vf = (volatile int *)kv_flag;
      while (*vf < layer - NK_STREAM_AHEAD) {
      }
    }
    __syncthreads();
    nk_stream_qw(w.gate, INTERMEDIATE_SIZE, HIDDEN_SIZE, sid);
    nk_stream_qw(w.up, INTERMEDIATE_SIZE, HIDDEN_SIZE, sid);
    nk_stream_qw(w.down, HIDDEN_SIZE, INTERMEDIATE_SIZE, sid);
    if (threadIdx.x < 2) {
      const __nv_bfloat16 *nrm = threadIdx.x ? w.post_ln : w.input_ln;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(nrm));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(nrm + 64));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(nrm + 128));
    }
  }
}
#endif // NK_STREAMER_BLOCKS

// =============================================================================
// Phase A1: input RMSNorm for one token tile -> f16 tile in shared memory.
// One warp per tile token; every block computes the tile redundantly.  The
// unit-0 owner also saves the fp32 residual.  Layer 0 reads bf16 embedding
// rows (launch-immutable -> __ldg); later layers read the f16 hidden buffer
// written under a grid barrier (plain loads ONLY — the nc path is not
// coherent with same-launch writes).
// =============================================================================

__device__ void nk_norm_tile(int tile_base, int seq_len,
                             const __half *__restrict__ hidden_in,
                             const __nv_bfloat16 *__restrict__ embed_weight,
                             const int *__restrict__ d_ids,
                             const __nv_bfloat16 *__restrict__ norm_weight,
                             float *__restrict__ g_residual, bool save_residual,
                             __half *__restrict__ s_x) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  if (warp_id >= TOKEN_TILE)
    return;
  int t = tile_base + warp_id;
  __half *row_out = s_x + (size_t)warp_id * (HIDDEN_SIZE + NK_XPAD);

  if (t >= seq_len) {
    for (int i = lane_id; i < HIDDEN_SIZE; i += WARP_SIZE)
      row_out[i] = __float2half(0.0f);
    return;
  }

  constexpr int PER_LANE = HIDDEN_SIZE / WARP_SIZE; // 64
  float vals[PER_LANE];
  float ss = 0.0f;
  if (hidden_in) {
    const __half *src = hidden_in + (size_t)t * HIDDEN_SIZE;
#pragma unroll
    for (int j = 0; j < PER_LANE; j++) {
      float v = __half2float(src[lane_id + j * WARP_SIZE]); // plain load
      vals[j] = v;
      ss += v * v;
    }
  } else {
    const __nv_bfloat16 *src =
        embed_weight + (size_t)__ldg(d_ids + t) * HIDDEN_SIZE;
#pragma unroll
    for (int j = 0; j < PER_LANE; j++) {
      float v = __bfloat162float(__ldg(src + lane_id + j * WARP_SIZE));
      vals[j] = v;
      ss += v * v;
    }
  }
  ss = nk_warp_reduce_sum(ss);
  float rstd = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
  rstd = __shfl_sync(0xffffffff, rstd, 0);

#pragma unroll
  for (int j = 0; j < PER_LANE; j++) {
    int i = lane_id + j * WARP_SIZE;
    float w = __bfloat162float(__ldg(norm_weight + i));
    row_out[i] = __float2half(vals[j] * rstd * w);
    if (save_residual)
      g_residual[(size_t)t * HIDDEN_SIZE + i] = vals[j];
  }
}

// =============================================================================
// Phase A: input norm + QKV projection, tile-parallel flat pair list in
// balanced contiguous chunks per block.  Unit rows never straddle Q/K/V
// boundaries (3072 | 1024 | 1024, all multiples of 16).
// =============================================================================

__device__ void nk_qkv_mma(int seq_len, int n_tiles,
                           const __half *__restrict__ layer_in,
                           const __nv_bfloat16 *__restrict__ embed_weight,
                           const int *__restrict__ d_ids,
                           const __nv_bfloat16 *__restrict__ in_norm_w,
                           float *__restrict__ g_residual,
                           __half *__restrict__ s_x, float *__restrict__ s_part,
                           const NKLayerWeights &w, __half *__restrict__ g_q,
                           __half *__restrict__ g_k,
                           __half *__restrict__ v_cache, int max_seq) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = (Q_SIZE + KV_SIZE + KV_SIZE) / NK_UNIT_ROWS; // 320
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT; // 128
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads(); // prior tile's readers must finish before restage
      nk_norm_tile(tile_base, seq_len, layer_in, embed_weight, d_ids,
                   in_norm_w, g_residual, /*save_residual=*/u == 0, s_x);
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wsrc;
    float ws2;
    int out_base;
    if (row0 < Q_SIZE) {
      wsrc.w = w.q.w + (size_t)row0 * (HIDDEN_SIZE / 2);
      wsrc.s = w.q.s + (size_t)row0 * (HIDDEN_SIZE / 16);
      ws2 = w.q_ws2;
      out_base = 0;
    } else if (row0 < Q_SIZE + KV_SIZE) {
      int r = row0 - Q_SIZE;
      wsrc.w = w.k.w + (size_t)r * (HIDDEN_SIZE / 2);
      wsrc.s = w.k.s + (size_t)r * (HIDDEN_SIZE / 16);
      ws2 = w.k_ws2;
      out_base = 1;
    } else {
      int r = row0 - Q_SIZE - KV_SIZE;
      wsrc.w = w.v.w + (size_t)r * (HIDDEN_SIZE / 2);
      wsrc.s = w.v.s + (size_t)r * (HIDDEN_SIZE / 16);
      ws2 = w.v_ws2;
      out_base = 2;
    }
    NKQuantW wb[1] = {{wsrc.w + (size_t)rg * 16 * (HIDDEN_SIZE / 2),
                       wsrc.s + (size_t)rg * 16 * (HIDDEN_SIZE / 16)}};
    float c[1][2][4] = {};
    nk_mma_slice<HIDDEN_SIZE, HIDDEN_SIZE + NK_XPAD, SLICE / 16, 1>(
        wb, s_x, ks * SLICE, ks * SLICE, c, lane);

#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

    int local0 = row0 - (out_base == 0 ? 0 : out_base == 1 ? Q_SIZE
                                                           : Q_SIZE + KV_SIZE);
#if NK_PARALLEL_PROJECTION_COMBINE
    nk_combine_partials_parallel(
        s_part, local0, tcount, [&](int r, int t, float v) {
          v *= ws2;
          if (out_base == 0)
            g_q[(size_t)(tile_base + t) * Q_SIZE + r] = __float2half(v);
          else if (out_base == 1)
            g_k[(size_t)(tile_base + t) * KV_SIZE + r] = __float2half(v);
          else
            v_cache[((size_t)(r / HEAD_DIM) * max_seq + tile_base + t) *
                        HEAD_DIM +
                    (r % HEAD_DIM)] = __float2half(v);
        });
#else
    if (warp_id < NK_RG) {
      nk_combine_partials(
          s_part, warp_id, local0, tcount, lane, [&](int r, int t, float v) {
            v *= ws2;
            if (out_base == 0)
              g_q[(size_t)(tile_base + t) * Q_SIZE + r] = __float2half(v);
            else if (out_base == 1)
              g_k[(size_t)(tile_base + t) * KV_SIZE + r] = __float2half(v);
            else
              v_cache[((size_t)(r / HEAD_DIM) * max_seq + tile_base + t) *
                          HEAD_DIM +
                      (r % HEAD_DIM)] = __float2half(v);
          });
    }
#endif
    __syncthreads();
  }
}

// =============================================================================
// Phase B: K RoPE + K-cache write (no K-norm in Ministral3).  One warp per
// (token, kv_head) unit across the grid.  RoPE angle uses the token's LOCAL
// position inside its packed sequence.
// =============================================================================

__device__ void nk_k_rope(int seq_len, const int *__restrict__ seq_start,
                          const __half *__restrict__ g_k,
                          const __half *__restrict__ cos_table,
                          const __half *__restrict__ sin_table,
                          __half *__restrict__ k_cache, int max_seq) {
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;

  for (int u = warp_gid; u < seq_len * NUM_KV_HEADS; u += TOTAL_WARPS) {
    int t = u / NUM_KV_HEADS;
    int kvh = u % NUM_KV_HEADS;
    int pos = t - (seq_start ? __ldg(seq_start + t) : 0);
    const __half *k_head = g_k + (size_t)t * KV_SIZE + kvh * HEAD_DIM;
    __half *kc = k_cache + ((size_t)kvh * max_seq + t) * HEAD_DIM;
    const __half *cos_pos = cos_table + (size_t)pos * HEAD_DIM;
    const __half *sin_pos = sin_table + (size_t)pos * HEAD_DIM;

    float kl[HEAD_DIM / WARP_SIZE];
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
      kl[j] = __half2float(k_head[i]); // plain load: same-launch data
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      float cv = __half2float(__ldg(cos_pos + i));
      float sv = __half2float(__ldg(sin_pos + i));
      int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
      int pi = i + po, pj = pi / WARP_SIZE;
      float pv = __shfl_sync(0xffffffff, kl[pj], pi % WARP_SIZE);
      float kf =
          (i < HEAD_DIM / 2) ? kl[j] * cv - pv * sv : pv * sv + kl[j] * cv;
      kc[i] = __float2half(kf);
    }
  }
}

// =============================================================================
// Phase C: bidirectional attention.  One warp per (token, q_head) unit:
// Q RoPE + llama-4 temperature at entry (round-tripped through g_q), then
// online-softmax over the token's WHOLE sequence window [s0, e).
// =============================================================================

__device__ void nk_attention(int seq_len, const int *__restrict__ seq_start,
                             const int *__restrict__ seq_end,
                             __half *__restrict__ g_q,
                             const __half *__restrict__ cos_table,
                             const __half *__restrict__ sin_table,
                             const __half *__restrict__ k_cache,
                             const __half *__restrict__ v_cache,
                             __half *__restrict__ g_attn, int max_seq,
                             float attn_scale, int mma_min_seq) {
#if NK_FUSED_GQA_ATTN
  if (seq_len >= NK_FUSED_GQA_MIN_TOTAL)
    return;
#endif
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;

  for (int u = warp_gid; u < seq_len * NUM_Q_HEADS; u += TOTAL_WARPS) {
    int t = u / NUM_Q_HEADS;
    int qh = u % NUM_Q_HEADS;
    int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
    int s0 = seq_start ? __ldg(seq_start + t) : 0;
    int e = seq_end ? __ldg(seq_end + t) : seq_len;
    if (mma_min_seq > 0 && e - s0 >= mma_min_seq)
      continue;
    int pos_local = t - s0;
    __half *q_head = g_q + (size_t)t * Q_SIZE + qh * HEAD_DIM;
    const __half *cos_pos = cos_table + (size_t)pos_local * HEAD_DIM;
    const __half *sin_pos = sin_table + (size_t)pos_local * HEAD_DIM;

    // Q RoPE + llama-4 attention temperature (identity below pos 16384),
    // strided-register layout, written back to g_q.
    {
      float l4 = nk_l4_temperature(pos_local);
      float ql[HEAD_DIM / WARP_SIZE];
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        ql[j] = __half2float(q_head[i]); // plain load
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cv = __half2float(__ldg(cos_pos + i));
        float sv = __half2float(__ldg(sin_pos + i));
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        int pi = i + po, pj = pi / WARP_SIZE;
        float pv = __shfl_sync(0xffffffff, ql[pj], pi % WARP_SIZE);
        float qf =
            (i < HEAD_DIM / 2) ? ql[j] * cv - pv * sv : pv * sv + ql[j] * cv;
        q_head[i] = __float2half(qf * l4);
      }
      __syncwarp();
    }

    // Online softmax over the full window [s0, e)
    int q_idx = lane_id * 4;
    float q_local[4];
    {
      const __half *qp = q_head + q_idx;
      q_local[0] = __half2float(qp[0]);
      q_local[1] = __half2float(qp[1]);
      q_local[2] = __half2float(qp[2]);
      q_local[3] = __half2float(qp[3]);
    }

    float max_score = __int_as_float(0xff800000); // -inf
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const __half *k_base = k_cache + (size_t)kv_head * max_seq * HEAD_DIM;
    const __half *v_base = v_cache + (size_t)kv_head * max_seq * HEAD_DIM;

    uint2 k_u2 = *reinterpret_cast<const uint2 *>(
        k_base + (size_t)s0 * HEAD_DIM + q_idx);
    uint2 v_u2 = *reinterpret_cast<const uint2 *>(
        v_base + (size_t)s0 * HEAD_DIM + q_idx);
    for (int pos = s0; pos < e; pos++) {
      int npos = pos + 1;
      uint2 k_nxt = k_u2, v_nxt = v_u2;
      if (npos < e) {
        k_nxt = *reinterpret_cast<const uint2 *>(
            k_base + (size_t)npos * HEAD_DIM + q_idx);
        v_nxt = *reinterpret_cast<const uint2 *>(
            v_base + (size_t)npos * HEAD_DIM + q_idx);
      }

      __half *k_ptr = reinterpret_cast<__half *>(&k_u2);
      float score = 0.0f;
      score += q_local[0] * __half2float(k_ptr[0]) +
               q_local[1] * __half2float(k_ptr[1]) +
               q_local[2] * __half2float(k_ptr[2]) +
               q_local[3] * __half2float(k_ptr[3]);
      score = nk_warp_reduce_sum(score) * attn_scale;
      score = __shfl_sync(0xffffffff, score, 0);

      float old_max = max_score;
      max_score = fmaxf(old_max, score);
      float exp_diff = fast_exp(old_max - max_score);
      float weight = fast_exp(score - max_score);
      sum_exp = sum_exp * exp_diff + weight;
      __half *v_ptr = reinterpret_cast<__half *>(&v_u2);
      out_acc[0] = out_acc[0] * exp_diff + weight * __half2float(v_ptr[0]);
      out_acc[1] = out_acc[1] * exp_diff + weight * __half2float(v_ptr[1]);
      out_acc[2] = out_acc[2] * exp_diff + weight * __half2float(v_ptr[2]);
      out_acc[3] = out_acc[3] * exp_diff + weight * __half2float(v_ptr[3]);

      k_u2 = k_nxt;
      v_u2 = v_nxt;
    }

    float inv_sum = 1.0f / sum_exp;
    __half *out = g_attn + (size_t)t * Q_SIZE + qh * HEAD_DIM + q_idx;
    out[0] = __float2half(out_acc[0] * inv_sum);
    out[1] = __float2half(out_acc[1] * inv_sum);
    out[2] = __float2half(out_acc[2] * inv_sum);
    out[3] = __float2half(out_acc[3] * inv_sum);
  }
}

#if NK_FUSED_GQA_ATTN
// Exact scalar attention variant.  Each warp owns one (token, KV-head) task,
// retains the scalar path's per-head arithmetic order, and reuses every K/V
// vector across the associated three query heads.  The register-Q path applies
// the same explicit half rounding as the former g_q store/reload round trip.
__device__ void nk_attention_gqa(
    int seq_len, const int *__restrict__ seq_start,
    const int *__restrict__ seq_end, __half *__restrict__ g_q,
    const __half *__restrict__ cos_table,
    const __half *__restrict__ sin_table,
    const __half *__restrict__ k_cache,
    const __half *__restrict__ v_cache, __half *__restrict__ g_attn,
    int max_seq, float attn_scale, int mma_min_seq) {
  if (seq_len < NK_FUSED_GQA_MIN_TOTAL)
    return;

  constexpr int GQA = NUM_Q_HEADS / NUM_KV_HEADS;
  static_assert(GQA == 3, "fused scalar attention assumes 3:1 GQA");
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;

  for (int u = warp_gid; u < seq_len * NUM_KV_HEADS; u += TOTAL_WARPS) {
    int t = u / NUM_KV_HEADS;
    int kv_head = u % NUM_KV_HEADS;
    int s0 = seq_start ? __ldg(seq_start + t) : 0;
    int e = seq_end ? __ldg(seq_end + t) : seq_len;
    if (mma_min_seq > 0 && e - s0 >= mma_min_seq)
      continue;
    int pos_local = t - s0;
    const __half *cos_pos = cos_table + (size_t)pos_local * HEAD_DIM;
    const __half *sin_pos = sin_table + (size_t)pos_local * HEAD_DIM;
    float l4 = nk_l4_temperature(pos_local);

    float q_local[GQA][4];
#pragma unroll
    for (int lqh = 0; lqh < GQA; lqh++) {
      int qh = kv_head * GQA + lqh;
      __half *q_head = g_q + (size_t)t * Q_SIZE + qh * HEAD_DIM;
#if NK_FUSED_GQA_Q_ROUNDTRIP
      float ql[HEAD_DIM / WARP_SIZE];
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        ql[j] = __half2float(q_head[i]);
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cv = __half2float(__ldg(cos_pos + i));
        float sv = __half2float(__ldg(sin_pos + i));
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        int pi = i + po, pj = pi / WARP_SIZE;
        float pv = __shfl_sync(0xffffffff, ql[pj], pi % WARP_SIZE);
        float qf = (i < HEAD_DIM / 2) ? ql[j] * cv - pv * sv
                                      : pv * sv + ql[j] * cv;
        q_head[i] = __float2half(qf * l4);
      }
      __syncwarp();

      int q_idx = lane_id * 4;
      const __half *qp = q_head + q_idx;
      q_local[lqh][0] = __half2float(qp[0]);
      q_local[lqh][1] = __half2float(qp[1]);
      q_local[lqh][2] = __half2float(qp[2]);
      q_local[lqh][3] = __half2float(qp[3]);
#else
      int q_idx = lane_id * 4;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int i = q_idx + j;
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        float qf = __half2float(q_head[i]);
        float pv = __half2float(q_head[i + po]);
        float cv = __half2float(__ldg(cos_pos + i));
        float sv = __half2float(__ldg(sin_pos + i));
        float rotated = (i < HEAD_DIM / 2) ? qf * cv - pv * sv
                                            : pv * sv + qf * cv;
        q_local[lqh][j] = __half2float(__float2half(rotated * l4));
      }
#endif
    }

    float max_score[GQA];
    float sum_exp[GQA];
    float out_acc[GQA][4];
#pragma unroll
    for (int lqh = 0; lqh < GQA; lqh++) {
      max_score[lqh] = __int_as_float(0xff800000);
      sum_exp[lqh] = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++)
        out_acc[lqh][j] = 0.0f;
    }

    int q_idx = lane_id * 4;
    const __half *k_base = k_cache + (size_t)kv_head * max_seq * HEAD_DIM;
    const __half *v_base = v_cache + (size_t)kv_head * max_seq * HEAD_DIM;
    uint2 k_u2 = *reinterpret_cast<const uint2 *>(
        k_base + (size_t)s0 * HEAD_DIM + q_idx);
    uint2 v_u2 = *reinterpret_cast<const uint2 *>(
        v_base + (size_t)s0 * HEAD_DIM + q_idx);

    for (int pos = s0; pos < e; pos++) {
      int npos = pos + 1;
      uint2 k_nxt = k_u2, v_nxt = v_u2;
      if (npos < e) {
        k_nxt = *reinterpret_cast<const uint2 *>(
            k_base + (size_t)npos * HEAD_DIM + q_idx);
        v_nxt = *reinterpret_cast<const uint2 *>(
            v_base + (size_t)npos * HEAD_DIM + q_idx);
      }

      __half *k_ptr = reinterpret_cast<__half *>(&k_u2);
      __half *v_ptr = reinterpret_cast<__half *>(&v_u2);
      float score[GQA];
#pragma unroll
      for (int lqh = 0; lqh < GQA; lqh++) {
        score[lqh] = 0.0f;
        score[lqh] +=
            q_local[lqh][0] * __half2float(k_ptr[0]) +
            q_local[lqh][1] * __half2float(k_ptr[1]) +
            q_local[lqh][2] * __half2float(k_ptr[2]) +
            q_local[lqh][3] * __half2float(k_ptr[3]);
        score[lqh] = nk_warp_reduce_sum(score[lqh]) * attn_scale;
        score[lqh] = __shfl_sync(0xffffffff, score[lqh], 0);
      }

#pragma unroll
      for (int lqh = 0; lqh < GQA; lqh++) {
        float old_max = max_score[lqh];
        max_score[lqh] = fmaxf(old_max, score[lqh]);
        float exp_diff = fast_exp(old_max - max_score[lqh]);
        float weight = fast_exp(score[lqh] - max_score[lqh]);
        sum_exp[lqh] = sum_exp[lqh] * exp_diff + weight;
        out_acc[lqh][0] =
            out_acc[lqh][0] * exp_diff + weight * __half2float(v_ptr[0]);
        out_acc[lqh][1] =
            out_acc[lqh][1] * exp_diff + weight * __half2float(v_ptr[1]);
        out_acc[lqh][2] =
            out_acc[lqh][2] * exp_diff + weight * __half2float(v_ptr[2]);
        out_acc[lqh][3] =
            out_acc[lqh][3] * exp_diff + weight * __half2float(v_ptr[3]);
      }

      k_u2 = k_nxt;
      v_u2 = v_nxt;
    }

#pragma unroll
    for (int lqh = 0; lqh < GQA; lqh++) {
      float inv_sum = 1.0f / sum_exp[lqh];
      int qh = kv_head * GQA + lqh;
      __half *out =
          g_attn + (size_t)t * Q_SIZE + qh * HEAD_DIM + q_idx;
      out[0] = __float2half(out_acc[lqh][0] * inv_sum);
      out[1] = __float2half(out_acc[lqh][1] * inv_sum);
      out[2] = __float2half(out_acc[lqh][2] * inv_sum);
      out[3] = __float2half(out_acc[lqh][3] * inv_sum);
    }
  }
}
#endif

#if NK_MMA_ATTN
// Load an m16k16 row-major FP16 A fragment from shared memory in the register
// layout consumed by mma.m16n8k16.  This is the unquantized counterpart of
// nk_decode(nk_load_raw(...)).
__device__ __forceinline__ NKFragA
nk_load_smem_a16816(const __half *__restrict__ a, int ld, int k0, int lane) {
  const __half *r0 = a + (size_t)(lane >> 2) * ld + k0 + (lane & 3) * 2;
  const __half *r1 = r0 + (size_t)8 * ld;
  NKFragA f;
  f.a0 = *reinterpret_cast<const unsigned *>(r0);
  f.a1 = *reinterpret_cast<const unsigned *>(r1);
  f.a2 = *reinterpret_cast<const unsigned *>(r0 + 8);
  f.a3 = *reinterpret_cast<const unsigned *>(r1 + 8);
  return f;
}

#if NK_MMA_ATTN_TMA
struct NKAttnMatrixX2 {
  unsigned x, y;
};

__device__ __forceinline__ unsigned nk_attn_smem_address(
    const void *pointer) {
  return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void nk_attn_mbarrier_init(uint64_t *barrier) {
  unsigned address = nk_attn_smem_address(barrier);
  unsigned count = 1;
  asm volatile("mbarrier.init.shared.b64 [%0], %1;"
               :
               : "r"(address), "r"(count)
               : "memory");
}

__device__ __forceinline__ void nk_attn_mbarrier_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void nk_attn_mbarrier_arrive_expect_tx(
    uint64_t *barrier, unsigned bytes) {
  unsigned address = nk_attn_smem_address(barrier);
  unsigned long long state;
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
      : "=l"(state)
      : "r"(address), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void nk_attn_cp_async_bulk(
    void *destination, const void *source, unsigned bytes,
    uint64_t *barrier) {
  unsigned destination_address = nk_attn_smem_address(destination);
  unsigned barrier_address = nk_attn_smem_address(barrier);
  unsigned long long source_address =
      static_cast<unsigned long long>(__cvta_generic_to_global(source));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
       :
      : "r"(destination_address), "l"(source_address), "r"(bytes),
        "r"(barrier_address)
      : "memory");
}

__device__ __forceinline__ void nk_attn_mbarrier_wait(
    uint64_t *barrier, unsigned parity) {
  unsigned address = nk_attn_smem_address(barrier);
  unsigned ready;
  do {
    asm volatile(
        "{ .reg .pred complete;\n"
        "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "complete, [%1], %2;\n"
        "  selp.b32 %0, 1, 0, complete; }"
        : "=r"(ready)
        : "r"(address), "r"(parity)
        : "memory");
  } while (!ready);
}

__device__ __forceinline__ void nk_attn_mbarrier_invalidate(
    uint64_t *barrier) {
  unsigned address = nk_attn_smem_address(barrier);
  asm volatile("mbarrier.inval.shared.b64 [%0];"
               :
               : "r"(address)
               : "memory");
}

__device__ __forceinline__ void nk_attn_issue_kv(
    __half *__restrict__ s_k, __half *__restrict__ s_v,
    const __half *__restrict__ g_k, const __half *__restrict__ g_v,
    uint64_t *barrier) {
  constexpr unsigned ONE_BYTES = NK_MMA_ATTN_KN * HEAD_DIM * sizeof(__half);
  nk_attn_mbarrier_arrive_expect_tx(barrier, 2 * ONE_BYTES);
  nk_attn_cp_async_bulk(s_k, g_k, ONE_BYTES, barrier);
  nk_attn_cp_async_bulk(s_v, g_v, ONE_BYTES, barrier);
}

// V is TMA-staged row-major [keys, d].  The transposed matrix load creates the
// column-major B fragment consumed by P@V without a producer-side transpose.
__device__ __forceinline__ NKAttnMatrixX2 nk_attn_load_pv_b_ldmatrix(
    const __half *__restrict__ v, int key0, int d0, int lane) {
  unsigned address = nk_attn_smem_address(
      v + (size_t)(key0 + (lane & 15)) * HEAD_DIM + d0);
  NKAttnMatrixX2 fragment;
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
      : "=r"(fragment.x), "=r"(fragment.y)
      : "r"(address));
  return fragment;
}
#endif

// CTA-tiled bidirectional FlashAttention experiment. A task is one
// sequence-local 16-query tile x one KV head and owns all three GQA heads.
__device__ __noinline__ void nk_attention_mma(
    int seq_len, const int *__restrict__ seq_start,
    const int *__restrict__ seq_end, __half *__restrict__ g_q,
    const __half *__restrict__ cos_table,
    const __half *__restrict__ sin_table,
    const __half *__restrict__ k_cache,
    const __half *__restrict__ v_cache, __half *__restrict__ g_attn,
    int max_seq, float attn_scale, char *__restrict__ arena) {
  constexpr int GQA = NUM_Q_HEADS / NUM_KV_HEADS;
  constexpr int QM = 16;
  constexpr int KN = NK_MMA_ATTN_KN;
  static_assert(GQA == 3, "mma attention maps one GQA head per QK warp");
  static_assert(NK_NUM_WARPS == 16,
                "mma attention maps one output dblock per warp");

  // Baseline uses 25,728 bytes for KN=16 (38,528 for KN=32).  The TMA
  // prototype uses 33,936 bytes: two row-major K/V stages plus two barriers.
  constexpr int STAGES = NK_MMA_ATTN_TMA ? 2 : 1;
  __half *s_q = reinterpret_cast<__half *>(arena);                   // [3,16,128]
  __half *s_k_bank = s_q + GQA * QM * HEAD_DIM;                      // [stage,KN,128]
  __half *s_v_bank = s_k_bank + STAGES * KN * HEAD_DIM;              // stage x 4 KiB
  float *s_score = reinterpret_cast<float *>(
      s_v_bank + STAGES * HEAD_DIM * KN);                            // [3,16,16]
  __half *s_p = reinterpret_cast<__half *>(s_score + GQA * QM * KN); // [3,16,16]
  float *s_alpha = reinterpret_cast<float *>(s_p + GQA * QM * KN);   // [3,16]
  float *s_m = s_alpha + GQA * QM;                                   // [3,16]
  float *s_l = s_m + GQA * QM;                                       // [3,16]
  float *s_l4 = s_l + GQA * QM;                                      // [16]
#if NK_MMA_ATTN_TMA
  uint64_t *s_full = reinterpret_cast<uint64_t *>(s_l4 + QM);         // [2]
#endif

  int tid = threadIdx.x;
  int warp = tid / WARP_SIZE;
  int lane = tid & (WARP_SIZE - 1);

  // Only sequence-local positions divisible by 16 become CTA tasks.
  for (int candidate = blockIdx.x; candidate < seq_len * NUM_KV_HEADS;
       candidate += NK_COMPUTE_BLOCKS) {
    int q0 = candidate / NUM_KV_HEADS;
    int kvh = candidate % NUM_KV_HEADS;
    int s0 = seq_start ? __ldg(seq_start + q0) : 0;
    int local0 = q0 - s0;
    if ((local0 & (QM - 1)) != 0)
      continue;
    int e = seq_end ? __ldg(seq_end + q0) : seq_len;
    if (e - s0 < NK_MMA_ATTN_MIN_SEQ)
      continue;
    int qcount = min(QM, e - q0);

    const __half *k_base = k_cache + (size_t)kvh * max_seq * HEAD_DIM;
    const __half *v_base = v_cache + (size_t)kvh * max_seq * HEAD_DIM;

#if NK_MMA_ATTN_TMA
    if (tid < 2)
      nk_attn_mbarrier_init(&s_full[tid]);
    if (tid == 0)
      nk_attn_mbarrier_fence_init();
#endif
    if (tid < QM) {
      int pos = local0 + tid;
      s_l4[tid] = nk_l4_temperature(pos);
    }
    if (tid < GQA * QM) {
      s_m[tid] = __int_as_float(0xff800000);
      s_l[tid] = 0.0f;
    }
    __syncthreads();

#if NK_MMA_ATTN_TMA
    // Start the first full K/V transfer before Q-RoPE.  The two stages do not
    // alias Q or the online-softmax state, so this hides first-tile latency
    // behind useful cooperative work.
    if (tid == 0 && e - s0 >= KN)
      nk_attn_issue_kv(s_k_bank, s_v_bank,
                       k_base + (size_t)s0 * HEAD_DIM,
                       v_base + (size_t)s0 * HEAD_DIM, &s_full[0]);
#endif

    // Pair d with d+64 for local RoPE, then round-trip through g_q.
    for (int i = tid; i < GQA * QM * (HEAD_DIM / 2);
         i += NK_BLOCK_SIZE) {
      int lqh = i / (QM * (HEAD_DIM / 2));
      int rem = i - lqh * QM * (HEAD_DIM / 2);
      int qr = rem / (HEAD_DIM / 2);
      int d = rem & (HEAD_DIM / 2 - 1);
      int sq_base = (lqh * QM + qr) * HEAD_DIM;
      if (qr < qcount) {
        int t = q0 + qr;
        int qh = kvh * GQA + lqh;
        __half *q = g_q + (size_t)t * Q_SIZE + qh * HEAD_DIM;
        const __half *cp = cos_table + (size_t)(local0 + qr) * HEAD_DIM;
        const __half *sp = sin_table + (size_t)(local0 + qr) * HEAD_DIM;
        float q0f = __half2float(q[d]);
        float q1f = __half2float(q[d + HEAD_DIM / 2]);
        float l4 = s_l4[qr];
        float r0 = (q0f * __half2float(__ldg(cp + d)) -
                    q1f * __half2float(__ldg(sp + d))) * l4;
        float r1 = (q0f * __half2float(__ldg(sp + d + HEAD_DIM / 2)) +
                    q1f * __half2float(__ldg(cp + d + HEAD_DIM / 2))) * l4;
        __half h0 = __float2half(r0);
        __half h1 = __float2half(r1);
        s_q[sq_base + d] = h0;
        s_q[sq_base + d + HEAD_DIM / 2] = h1;
#if NK_MMA_ATTN_Q_WRITE
        q[d] = h0;
        q[d + HEAD_DIM / 2] = h1;
#endif
      } else {
        s_q[sq_base + d] = __float2half(0.0f);
        s_q[sq_base + d + HEAD_DIM / 2] = __float2half(0.0f);
      }
    }
    __syncthreads();

    float out[GQA][4];
#pragma unroll
    for (int h = 0; h < GQA; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        out[h][j] = 0.0f;

    int key_tile = 0;
    for (int kb = s0; kb < e; kb += KN, ++key_tile) {
      int nkeys = min(KN, e - kb);
      int stage = 0;
#if NK_MMA_ATTN_TMA
      stage = key_tile & 1;
#endif
      __half *s_k = s_k_bank + (size_t)stage * KN * HEAD_DIM;
      __half *s_vt = s_v_bank + (size_t)stage * KN * HEAD_DIM;
#if NK_MMA_ATTN_TMA
      if (nkeys == KN) {
        // Each stage is reused every other tile, so its expected phase is
        // derivable from the tile index.  Keeping it out of an indexed local
        // array avoids a persistent-kernel stack spill.
        unsigned phase = (key_tile >> 1) & 1;
        nk_attn_mbarrier_wait(&s_full[stage], phase);
      } else {
        // The final partial tile copies cooperatively to avoid reading beyond
        // a packed cache allocation.
        for (int i = tid; i < KN * HEAD_DIM; i += NK_BLOCK_SIZE) {
          int kr = i / HEAD_DIM;
          int d = i - kr * HEAD_DIM;
          __half kh = __float2half(0.0f);
          __half vh = __float2half(0.0f);
          if (kr < nkeys) {
            kh = k_base[(size_t)(kb + kr) * HEAD_DIM + d];
            vh = v_base[(size_t)(kb + kr) * HEAD_DIM + d];
          }
          s_k[kr * HEAD_DIM + d] = kh;
          s_vt[kr * HEAD_DIM + d] = vh;
        }
        __syncthreads();
      }

      int next_kb = kb + KN;
      if (tid == 0 && next_kb + KN <= e) {
        int next_stage = stage ^ 1;
        nk_attn_issue_kv(
            s_k_bank + (size_t)next_stage * KN * HEAD_DIM,
            s_v_bank + (size_t)next_stage * KN * HEAD_DIM,
            k_base + (size_t)next_kb * HEAD_DIM,
            v_base + (size_t)next_kb * HEAD_DIM, &s_full[next_stage]);
      }
#else
      for (int i = tid; i < KN * HEAD_DIM; i += NK_BLOCK_SIZE) {
        int kr = i / HEAD_DIM;
        int d = i - kr * HEAD_DIM;
        __half kh = __float2half(0.0f);
        __half vh = __float2half(0.0f);
        if (kr < nkeys) {
          kh = k_base[(size_t)(kb + kr) * HEAD_DIM + d];
          vh = v_base[(size_t)(kb + kr) * HEAD_DIM + d];
        }
        s_k[kr * HEAD_DIM + d] = kh;
        s_vt[d * KN + kr] = vh;
      }
      __syncthreads();
#endif

      // Three warps form the three 16xKN QK score tiles.  Keep only two
      // n=8 accumulators live at once so KN=32 does not inflate registers.
      if (warp < GQA) {
        const __half *qa = s_q + warp * QM * HEAD_DIM;
#pragma unroll
        for (int nb = 0; nb < KN; nb += 16) {
          float c[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
          for (int kk = 0; kk < HEAD_DIM; kk += 16) {
            NKFragA a = nk_load_smem_a16816(qa, HEAD_DIM, kk, lane);
            const __half *b0 =
                s_k + (size_t)(nb + (lane >> 2)) * HEAD_DIM + kk +
                (lane & 3) * 2;
            const __half *b1 = b0 + (size_t)8 * HEAD_DIM;
            unsigned b00 = *reinterpret_cast<const unsigned *>(b0);
            unsigned b01 = *reinterpret_cast<const unsigned *>(b0 + 8);
            unsigned b10 = *reinterpret_cast<const unsigned *>(b1);
            unsigned b11 = *reinterpret_cast<const unsigned *>(b1 + 8);
            nk_mma16816(c[0], a, b00, b01);
            nk_mma16816(c[1], a, b10, b11);
          }
#pragma unroll
          for (int nh = 0; nh < 2; nh++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
              int row = nk_c_row(j, lane);
              int col = nb + nk_c_token(nh, j, lane);
              s_score[(warp * QM + row) * KN + col] =
                  c[nh][j] * attn_scale;
            }
        }
      }
      __syncthreads();

      // FP32 online softmax; FP16 P is the only attention approximation.
      if (tid < GQA * QM) {
        int lqh = tid / QM;
        int qr = tid - lqh * QM;
        __half *prow = s_p + (lqh * QM + qr) * KN;
        if (qr < qcount) {
          const float *sr = s_score + (lqh * QM + qr) * KN;
          float tile_max = __int_as_float(0xff800000);
#pragma unroll
          for (int j = 0; j < KN; j++)
            if (j < nkeys)
              tile_max = fmaxf(tile_max, sr[j]);
          float old_m = s_m[lqh * QM + qr];
          float new_m = fmaxf(old_m, tile_max);
          float alpha = fast_exp(old_m - new_m);
          float tile_sum = 0.0f;
#pragma unroll
          for (int j = 0; j < KN; j++) {
            float p = (j < nkeys) ? fast_exp(sr[j] - new_m) : 0.0f;
            __half ph = __float2half(p);
            prow[j] = ph;
            tile_sum += __half2float(ph);
          }
          s_alpha[lqh * QM + qr] = alpha;
          s_l[lqh * QM + qr] = s_l[lqh * QM + qr] * alpha + tile_sum;
          s_m[lqh * QM + qr] = new_m;
        } else {
          s_alpha[lqh * QM + qr] = 0.0f;
#pragma unroll
          for (int j = 0; j < KN; j++)
            prow[j] = __float2half(0.0f);
        }
      }
      __syncthreads();

      // Warp w owns output dimensions [w*8,w*8+8) for all three heads.
#pragma unroll
      for (int lqh = 0; lqh < GQA; lqh++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int row = nk_c_row(j, lane);
          out[lqh][j] *= s_alpha[lqh * QM + row];
        }
#pragma unroll
        for (int pk = 0; pk < KN; pk += 16) {
          NKFragA a = nk_load_smem_a16816(
              s_p + lqh * QM * KN, KN, pk, lane);
#if NK_MMA_ATTN_TMA
          NKAttnMatrixX2 b = nk_attn_load_pv_b_ldmatrix(
              s_vt, pk, warp * 8, lane);
          nk_mma16816(out[lqh], a, b.x, b.y);
#else
          const __half *bp = s_vt +
              (warp * 8 + (lane >> 2)) * KN + pk + (lane & 3) * 2;
          unsigned b0 = *reinterpret_cast<const unsigned *>(bp);
          unsigned b1 = *reinterpret_cast<const unsigned *>(bp + 8);
          nk_mma16816(out[lqh], a, b0, b1);
#endif
        }
      }
      __syncthreads();
    }

#if NK_MMA_ATTN_TMA
    if (tid < 2)
      nk_attn_mbarrier_invalidate(&s_full[tid]);
    __syncthreads();
#endif

#pragma unroll
    for (int lqh = 0; lqh < GQA; lqh++) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int qr = nk_c_row(j, lane);
        if (qr < qcount) {
          int d = warp * 8 + (lane & 3) * 2 + (j & 1);
          int qh = kvh * GQA + lqh;
          __half *dst = g_attn + (size_t)(q0 + qr) * Q_SIZE +
                        qh * HEAD_DIM + d;
          *dst = __float2half(out[lqh][j] *
                             ptx_rcp(s_l[lqh * QM + qr]));
        }
      }
    }
    __syncthreads();
  }
}
#endif // NK_MMA_ATTN

// =============================================================================
// Phase D1: O-projection + residual.  K = 3072 exceeds the arena next to the
// combine scratch, so the attention tile is staged in two K-halves of
// [16][1536+pad]; the warps' C fragments accumulate across both halves.
// =============================================================================

__device__ void nk_oproj_mma(int seq_len, int n_tiles,
                             const __half *__restrict__ g_attn,
                             __half *__restrict__ s_attn,
                             float *__restrict__ s_part, const NKLayerWeights &w,
                             float *__restrict__ g_residual) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS; // 128
  constexpr int KHALF = Q_SIZE / 2;                     // 1536
  constexpr int SLICE = KHALF / NK_KSPLIT;              // 96
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[1] = {{w.o.w + (size_t)(row0 + rg * 16) * (Q_SIZE / 2),
                       w.o.s + (size_t)(row0 + rg * 16) * (Q_SIZE / 16)}};
    float c[1][2][4] = {};

#pragma unroll 1
    for (int half = 0; half < 2; half++) {
      if (staged != tile * 2 + half) {
        __syncthreads();
        for (int t = 0; t < tcount; t++) {
          const uint4 *src = reinterpret_cast<const uint4 *>(
              g_attn + (size_t)(tile_base + t) * Q_SIZE + half * KHALF);
          uint4 *dst =
              reinterpret_cast<uint4 *>(s_attn + (size_t)t * (KHALF + NK_XPAD));
          for (int i = threadIdx.x; i < KHALF / 8; i += NK_BLOCK_SIZE)
            dst[i] = src[i]; // plain load: same-launch data
        }
        __syncthreads();
        staged = tile * 2 + half;
      }
      nk_mma_slice<Q_SIZE, KHALF + NK_XPAD, SLICE / 16, 1>(
          wb, s_attn, half * KHALF + ks * SLICE, ks * SLICE, c, lane);
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

    if (warp_id < NK_RG)
      nk_combine_partials(s_part, warp_id, row0, tcount, lane,
                          [&](int r, int t, float v) {
                            g_residual[(size_t)(tile_base + t) * HIDDEN_SIZE +
                                       r] += v * w.o_ws2;
                          });
    __syncthreads();
  }
}

// =============================================================================
// Phase D2a: post-attention RMSNorm tile (fp32 residual -> f16 tile).
// =============================================================================

__device__ void nk_postnorm_tile(int tile_base, int seq_len,
                                 const float *__restrict__ g_residual,
                                 const __nv_bfloat16 *__restrict__ norm_weight,
                                 __half *__restrict__ s_x) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  if (warp_id >= TOKEN_TILE)
    return;
  int t = tile_base + warp_id;
  __half *row_out = s_x + (size_t)warp_id * (HIDDEN_SIZE + NK_XPAD);

  if (t >= seq_len) {
    for (int i = lane_id; i < HIDDEN_SIZE; i += WARP_SIZE)
      row_out[i] = __float2half(0.0f);
    return;
  }

  const float *src = g_residual + (size_t)t * HIDDEN_SIZE;
  constexpr int PER_LANE = HIDDEN_SIZE / WARP_SIZE; // 64
  float vals[PER_LANE];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < PER_LANE; j++) {
    float v = src[lane_id + j * WARP_SIZE];
    vals[j] = v;
    ss += v * v;
  }
  ss = nk_warp_reduce_sum(ss);
  float rstd = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
  rstd = __shfl_sync(0xffffffff, rstd, 0);

#pragma unroll
  for (int j = 0; j < PER_LANE; j++) {
    int i = lane_id + j * WARP_SIZE;
    float w = __bfloat162float(__ldg(norm_weight + i));
    row_out[i] = __float2half(vals[j] * rstd * w);
  }
}

// =============================================================================
// Phase D2b: gate+up+SiLU.  Each warp runs the gate unit and its matching up
// unit as two mma A-streams so silu(gate*ws2_g) * (up*ws2_u) combines
// in-thread (identical fragment coordinates).
// =============================================================================

__device__ void nk_gateup_mma(int seq_len, int n_tiles,
                              const float *__restrict__ g_residual,
                              const __nv_bfloat16 *__restrict__ post_norm_w,
                              __half *__restrict__ s_x,
                              float *__restrict__ s_part,
                              const NKLayerWeights &w,
                              __half *__restrict__ g_mlp) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = INTERMEDIATE_SIZE / NK_UNIT_ROWS; // 384
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT;
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      nk_postnorm_tile(tile_base, seq_len, g_residual, post_norm_w, s_x);
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[2] = {
        {w.gate.w + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 2),
         w.gate.s + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 16)},
        {w.up.w + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 2),
         w.up.s + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 16)}};
    float c[2][2][4] = {};
    nk_mma_slice<HIDDEN_SIZE, HIDDEN_SIZE + NK_XPAD, SLICE / 16, 2>(
        wb, s_x, ks * SLICE, ks * SLICE, c, lane);

#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          s_part[((size_t)warp_id * 2 + s) * 256 + lane * 8 + h * 4 + j] =
              c[s][h][j];
    __syncthreads();

    if (warp_id < NK_RG) {
      int crg = warp_id;
      for (int i = lane; i < 256; i += WARP_SIZE) {
        float g = 0.0f, up = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++) {
          size_t wp = (size_t)(crg * NK_KSPLIT + s) * 2;
          g += s_part[wp * 256 + i];
          up += s_part[(wp + 1) * 256 + i];
        }
        int src_lane = i >> 3;
        int h = (i >> 2) & 1;
        int j = i & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + crg * 16 + nk_c_row(j, src_lane);
          g_mlp[(size_t)(tile_base + t) * INTERMEDIATE_SIZE + r] =
              __float2half(nk_silu(g * w.gate_ws2) * (up * w.up_ws2));
        }
      }
    }
    __syncthreads();
  }
}

// =============================================================================
// Phase D3: down projection + residual -> next layer hidden (f16).
// K = 6144 staged in four quarters of [16][1536+pad].
// =============================================================================

__device__ void nk_down_mma(int seq_len, int n_tiles,
                            const __half *__restrict__ g_mlp,
                            __half *__restrict__ s_mlp,
                            float *__restrict__ s_part,
                            const NKLayerWeights &w,
                            const float *__restrict__ g_residual,
                            __half *__restrict__ hidden_out) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS; // 128
  constexpr int KQUARTER = INTERMEDIATE_SIZE / 4;       // 1536
  constexpr int SLICE = KQUARTER / NK_KSPLIT;           // 96
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[1] = {
        {w.down.w + (size_t)(row0 + rg * 16) * (INTERMEDIATE_SIZE / 2),
         w.down.s + (size_t)(row0 + rg * 16) * (INTERMEDIATE_SIZE / 16)}};
    float c[1][2][4] = {};

#pragma unroll 1
    for (int quarter = 0; quarter < 4; quarter++) {
      if (staged != tile * 4 + quarter) {
        __syncthreads();
        for (int t = 0; t < tcount; t++) {
          const uint4 *src = reinterpret_cast<const uint4 *>(
              g_mlp + (size_t)(tile_base + t) * INTERMEDIATE_SIZE +
              quarter * KQUARTER);
          uint4 *dst = reinterpret_cast<uint4 *>(
              s_mlp + (size_t)t * (KQUARTER + NK_XPAD));
          for (int i = threadIdx.x; i < KQUARTER / 8; i += NK_BLOCK_SIZE)
            dst[i] = src[i]; // plain load: same-launch data
        }
        __syncthreads();
        staged = tile * 4 + quarter;
      }
      nk_mma_slice<INTERMEDIATE_SIZE, KQUARTER + NK_XPAD, SLICE / 16, 1>(
          wb, s_mlp, quarter * KQUARTER + ks * SLICE, ks * SLICE, c, lane);
    }

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

    if (warp_id < NK_RG)
      nk_combine_partials(s_part, warp_id, row0, tcount, lane,
                          [&](int r, int t, float v) {
                            size_t gt =
                                (size_t)(tile_base + t) * HIDDEN_SIZE + r;
                            hidden_out[gt] =
                                __float2half(v * w.down_ws2 + g_residual[gt]);
                          });
    __syncthreads();
  }
}

// =============================================================================
// W4A4 phase functions.  Identical pair-list scaffolding to the f16 engine;
// what changes is staging (activations quantized to packed NVFP4 + scale
// tiles) and the mainloop (block-scaled fp4 mma).  K = 3072 / 6144 now fit
// a single packed stage — no half/quarter restaging.
//
// Smem pitches: packed rows padded so the row stride in 32-bit words is
// == 4 (mod 32) (B-fragment u32 loads cover all banks); scale rows padded
// to odd word counts.
// =============================================================================

#if NK_FP4MMA

constexpr int NK_BQ_PITCH_H = HIDDEN_SIZE / 2 + 16;        // 1040
constexpr int NK_SF_PITCH_H = HIDDEN_SIZE / 16 + 4;        // 132
constexpr int NK_BQ_PITCH_O = Q_SIZE / 2 + 16;             // 1552
constexpr int NK_SF_PITCH_O = Q_SIZE / 16 + 4;             // 196
constexpr int NK_BQ_PITCH_D = INTERMEDIATE_SIZE / 2 + 16;  // 3088
constexpr int NK_SF_PITCH_D = INTERMEDIATE_SIZE / 16 + 4;  // 388

// Region offsets for the norm-sourced phases (QKV / gate-up): packed tile,
// scale tile, then a region shared by the f16 staging tile (dead once the
// tile is quantized) and the combine scratch.
constexpr int NK_Q_SF_OFF = TOKEN_TILE * NK_BQ_PITCH_H;              // 16640
constexpr int NK_Q_R3_OFF = NK_Q_SF_OFF + TOKEN_TILE * NK_SF_PITCH_H; // 18752

#if NK_DUAL_CTA
// Compute layer-input RMSNorm scalars once per token instead of redundantly
// in every weight-stationary CTA. The residual copy is also uniquely owned by
// this pass. Lane iteration and warp reduction match nk_norm_tile exactly.
__device__ void nk_input_rstd_pass(
    int seq_len, const __half *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const int *__restrict__ d_ids, float *__restrict__ g_residual,
    float *__restrict__ g_rstd) {
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;
  for (int t = warp_gid; t < seq_len; t += TOTAL_WARPS) {
    const __half *hsrc = hidden_in ? hidden_in + (size_t)t * HIDDEN_SIZE
                                   : nullptr;
    const __nv_bfloat16 *bsrc = hidden_in
        ? nullptr
        : embed_weight + (size_t)__ldg(d_ids + t) * HIDDEN_SIZE;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < HIDDEN_SIZE / WARP_SIZE; j++) {
      int i = lane + j * WARP_SIZE;
      float v = hsrc ? __half2float(hsrc[i])
                     : __bfloat162float(__ldg(bsrc + i));
      ss += v * v;
      g_residual[(size_t)t * HIDDEN_SIZE + i] = v;
    }
    ss = nk_warp_reduce_sum(ss);
    if (lane == 0)
      g_rstd[t] = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
  }
}

// Post-attention residual RMSNorm scalar pass. This preserves the scalar
// accumulation order of nk_postnorm_tile but runs once per token per layer.
__device__ void nk_residual_rstd_pass(
    int seq_len, const float *__restrict__ g_residual,
    float *__restrict__ g_rstd) {
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;
  for (int t = warp_gid; t < seq_len; t += TOTAL_WARPS) {
    const float *src = g_residual + (size_t)t * HIDDEN_SIZE;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < HIDDEN_SIZE / WARP_SIZE; j++) {
      float v = src[lane + j * WARP_SIZE];
      ss += v * v;
    }
    ss = nk_warp_reduce_sum(ss);
    if (lane == 0)
      g_rstd[t] = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
  }
}

// One-pass normalized input packing after nk_input_rstd_pass. Each CTA still
// stages its weight-stationary activation tile, but no longer needs the 64-KB
// f16 intermediary or a redundant RMS reduction.
__device__ __forceinline__ void nk_inputnorm_quant_tile(
    int tile_base, int seq_len, const __half *__restrict__ hidden_in,
    const __nv_bfloat16 *__restrict__ embed_weight,
    const int *__restrict__ d_ids,
    const __nv_bfloat16 *__restrict__ norm_weight,
    const float *__restrict__ g_rstd,
    unsigned char *__restrict__ s_bq, unsigned char *__restrict__ s_sf,
    float inv6a, float a) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (warp >= TOKEN_TILE)
    return;
  int t = tile_base + warp;
  unsigned char *bq_row = s_bq + (size_t)warp * NK_BQ_PITCH_H;
  unsigned char *sf_row = s_sf + (size_t)warp * NK_SF_PITCH_H;
  if (t >= seq_len) {
    nk_zero_qrow(bq_row, HIDDEN_SIZE / 16, sf_row, lane);
    return;
  }
  const __half *hsrc = hidden_in ? hidden_in + (size_t)t * HIDDEN_SIZE
                                 : nullptr;
  const __nv_bfloat16 *bsrc = hidden_in
      ? nullptr
      : embed_weight + (size_t)__ldg(d_ids + t) * HIDDEN_SIZE;
  float rstd = g_rstd[t];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int group = lane + q * WARP_SIZE;
    int i0 = group * 16;
    __half2 h[8];
#pragma unroll
    for (int p = 0; p < 8; p++) {
      int i = i0 + p * 2;
      float x0 = hsrc ? __half2float(hsrc[i])
                      : __bfloat162float(__ldg(bsrc + i));
      float x1 = hsrc ? __half2float(hsrc[i + 1])
                      : __bfloat162float(__ldg(bsrc + i + 1));
      float w0 = __bfloat162float(__ldg(norm_weight + i));
      float w1 = __bfloat162float(__ldg(norm_weight + i + 1));
      h[p] = __floats2half2_rn(x0 * rstd * w0, x1 * rstd * w1);
    }
    uint2 packed;
    unsigned char sfb;
    nk_quant16(h, inv6a, a, packed, sfb);
    *reinterpret_cast<uint2 *>(bq_row + group * 8) = packed;
    sf_row[group] = sfb;
  }
}
#endif

// Quantize the staged f16 tile (s_x rows, XPAD-strided) into packed+scale
// tiles.  Warp w quantizes row w; call after a __syncthreads that makes all
// norm rows visible.
__device__ __forceinline__ void
nk_quant_tile(const __half *__restrict__ s_x, unsigned char *__restrict__ s_bq,
              unsigned char *__restrict__ s_sf, float inv6a, float a) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (warp_id >= TOKEN_TILE)
    return;
  nk_quant_row(s_x + (size_t)warp_id * (HIDDEN_SIZE + NK_XPAD),
               HIDDEN_SIZE / 16, s_bq + (size_t)warp_id * NK_BQ_PITCH_H,
               s_sf + (size_t)warp_id * NK_SF_PITCH_H, inv6a, a, lane);
}

#if NK_FP4_T2_GATEUP
// Fast exact post-attention RMSNorm + NVFP4 packing for two token tiles.
// Pass 1 preserves nk_postnorm_tile's strided RMS summation order.  Pass 2
// rereads the row as contiguous 16-value groups, avoiding the shuffle-heavy
// register transpose while retaining identical f16 and NVFP4 quantization.
__device__ __forceinline__ void nk_postnorm_quant_t2(
    int tile_base, int seq_len, const float *__restrict__ g_residual,
    const __nv_bfloat16 *__restrict__ norm_weight,
    unsigned char *__restrict__ s_bq, unsigned char *__restrict__ s_sf,
    float inv6a, float a) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (warp_id >= TOKEN_TILE)
    return;

#pragma unroll
  for (int th = 0; th < 2; th++) {
    int tr = warp_id + th * TOKEN_TILE;
    int t = tile_base + tr;
    unsigned char *bq_row = s_bq + (size_t)tr * NK_BQ_PITCH_H;
    unsigned char *sf_row = s_sf + (size_t)tr * NK_SF_PITCH_H;
    if (t >= seq_len) {
      nk_zero_qrow(bq_row, HIDDEN_SIZE / 16, sf_row, lane);
      continue;
    }

    const float *src = g_residual + (size_t)t * HIDDEN_SIZE;
    constexpr int PER_LANE = HIDDEN_SIZE / WARP_SIZE; // 64
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < PER_LANE; j++) {
      float v = src[lane + j * WARP_SIZE];
      ss += v * v;
    }
    ss = nk_warp_reduce_sum(ss);
    float rstd = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
    rstd = __shfl_sync(0xffffffff, rstd, 0);

#pragma unroll
    for (int q = 0; q < 4; q++) {
      int group = lane + q * WARP_SIZE;
      const float4 *vsrc =
          reinterpret_cast<const float4 *>(src + group * 16);
      __half2 h[8];
#pragma unroll
      for (int v = 0; v < 4; v++) {
        float4 x = vsrc[v];
        int i = group * 16 + v * 4;
        float w0 = __bfloat162float(__ldg(norm_weight + i));
        float w1 = __bfloat162float(__ldg(norm_weight + i + 1));
        float w2 = __bfloat162float(__ldg(norm_weight + i + 2));
        float w3 = __bfloat162float(__ldg(norm_weight + i + 3));
        h[v * 2] = __floats2half2_rn(x.x * rstd * w0, x.y * rstd * w1);
        h[v * 2 + 1] =
            __floats2half2_rn(x.z * rstd * w2, x.w * rstd * w3);
      }
      uint2 packed;
      unsigned char sfb;
      nk_quant16(h, inv6a, a, packed, sfb);
      *reinterpret_cast<uint2 *>(bq_row + group * 8) = packed;
      sf_row[group] = sfb;
    }
  }
}
#endif

// --- Phase A (W4A4): input norm + QKV ---
__device__ void nk_qkv_mma4(int seq_len, int n_tiles,
                            const __half *__restrict__ layer_in,
                            const __nv_bfloat16 *__restrict__ embed_weight,
                            const int *__restrict__ d_ids,
                            float *__restrict__ g_residual,
                            char *__restrict__ s_arena,
                            const NKLayerWeights &w, __half *__restrict__ g_q,
                            __half *__restrict__ g_k,
                            __half *__restrict__ v_cache, int max_seq,
                            const float *__restrict__ g_rstd) {
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf = reinterpret_cast<unsigned char *>(s_arena) + NK_Q_SF_OFF;
#if !NK_DUAL_CTA
  __half *s_x = reinterpret_cast<__half *>(s_arena + NK_Q_R3_OFF);
#endif
  float *s_part = reinterpret_cast<float *>(s_arena + NK_Q_R3_OFF);

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = (Q_SIZE + KV_SIZE + KV_SIZE) / NK_UNIT_ROWS; // 320
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT; // 128 elems = 2 k64 chunks
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
#if NK_DUAL_CTA
      nk_inputnorm_quant_tile(tile_base, seq_len, layer_in, embed_weight,
                              d_ids, w.input_ln, g_rstd, s_bq, s_sf,
                              w.aqkv_i6, w.aqkv);
#else
      nk_norm_tile(tile_base, seq_len, layer_in, embed_weight, d_ids,
                   w.input_ln, g_residual, /*save_residual=*/u == 0, s_x);
      __syncthreads();
      nk_quant_tile(s_x, s_bq, s_sf, w.aqkv_i6, w.aqkv);
#endif
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wsrc;
    float eps;
    int out_base;
    if (row0 < Q_SIZE) {
      wsrc.w = w.q.w + (size_t)row0 * (HIDDEN_SIZE / 2);
      wsrc.s = w.q.s + (size_t)row0 * (HIDDEN_SIZE / 16);
      eps = w.q_eps;
      out_base = 0;
    } else if (row0 < Q_SIZE + KV_SIZE) {
      int r = row0 - Q_SIZE;
      wsrc.w = w.k.w + (size_t)r * (HIDDEN_SIZE / 2);
      wsrc.s = w.k.s + (size_t)r * (HIDDEN_SIZE / 16);
      eps = w.k_eps;
      out_base = 1;
    } else {
      int r = row0 - Q_SIZE - KV_SIZE;
      wsrc.w = w.v.w + (size_t)r * (HIDDEN_SIZE / 2);
      wsrc.s = w.v.s + (size_t)r * (HIDDEN_SIZE / 16);
      eps = w.v_eps;
      out_base = 2;
    }
    NKQuantW wb[1] = {{wsrc.w + (size_t)rg * 16 * (HIDDEN_SIZE / 2),
                       wsrc.s + (size_t)rg * 16 * (HIDDEN_SIZE / 16)}};
    float c[1][2][4] = {};
    nk_mma4_slice<HIDDEN_SIZE, NK_BQ_PITCH_H, NK_SF_PITCH_H, SLICE / 64, 1>(
        wb, s_bq, s_sf, ks * SLICE, ks * SLICE, c, lane);

#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

    int local0 = row0 - (out_base == 0 ? 0 : out_base == 1 ? Q_SIZE
                                                           : Q_SIZE + KV_SIZE);
#if NK_PARALLEL_PROJECTION_COMBINE
    nk_combine_partials_parallel(
        s_part, local0, tcount, [&](int r, int t, float v) {
          v *= eps;
          if (out_base == 0)
            g_q[(size_t)(tile_base + t) * Q_SIZE + r] = __float2half(v);
          else if (out_base == 1)
            g_k[(size_t)(tile_base + t) * KV_SIZE + r] = __float2half(v);
          else
            v_cache[((size_t)(r / HEAD_DIM) * max_seq + tile_base + t) *
                        HEAD_DIM +
                    (r % HEAD_DIM)] = __float2half(v);
        });
#else
    if (warp_id < NK_RG) {
      nk_combine_partials(
          s_part, warp_id, local0, tcount, lane, [&](int r, int t, float v) {
            v *= eps;
            if (out_base == 0)
              g_q[(size_t)(tile_base + t) * Q_SIZE + r] = __float2half(v);
            else if (out_base == 1)
              g_k[(size_t)(tile_base + t) * KV_SIZE + r] = __float2half(v);
            else
              v_cache[((size_t)(r / HEAD_DIM) * max_seq + tile_base + t) *
                          HEAD_DIM +
                      (r % HEAD_DIM)] = __float2half(v);
          });
    }
#endif
    __syncthreads();
  }
}

// --- Phase D1 (W4A4): O projection, K=3072 in ONE packed stage ---
__device__ void nk_oproj_mma4(int seq_len, int n_tiles,
                              const __half *__restrict__ g_attn,
                              char *__restrict__ s_arena,
                              const NKLayerWeights &w,
                              float *__restrict__ g_residual) {
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + TOKEN_TILE * NK_BQ_PITCH_O;
  float *s_part = reinterpret_cast<float *>(
      s_arena + TOKEN_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O));

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS; // 128
  constexpr int SLICE = Q_SIZE / NK_KSPLIT;             // 192 = 3 chunks
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      if (warp_id < TOKEN_TILE) {
        unsigned char *bq_row = s_bq + (size_t)warp_id * NK_BQ_PITCH_O;
        unsigned char *sf_row = s_sf + (size_t)warp_id * NK_SF_PITCH_O;
        if (warp_id < tcount)
          nk_quant_row(g_attn + (size_t)(tile_base + warp_id) * Q_SIZE,
                       Q_SIZE / 16, bq_row, sf_row, w.ao_i6, w.ao, lane);
        else
          nk_zero_qrow(bq_row, Q_SIZE / 16, sf_row, lane);
      }
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[1] = {{w.o.w + (size_t)(row0 + rg * 16) * (Q_SIZE / 2),
                       w.o.s + (size_t)(row0 + rg * 16) * (Q_SIZE / 16)}};
    float c[1][2][4] = {};
    nk_mma4_slice<Q_SIZE, NK_BQ_PITCH_O, NK_SF_PITCH_O, SLICE / 64, 1>(
        wb, s_bq, s_sf, ks * SLICE, ks * SLICE, c, lane);

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

#if NK_PARALLEL_PROJECTION_COMBINE
    nk_combine_partials_parallel(
        s_part, row0, tcount, [&](int r, int t, float v) {
          g_residual[(size_t)(tile_base + t) * HIDDEN_SIZE + r] +=
              v * w.o_eps;
        });
#else
    if (warp_id < NK_RG)
      nk_combine_partials(s_part, warp_id, row0, tcount, lane,
                          [&](int r, int t, float v) {
                            g_residual[(size_t)(tile_base + t) * HIDDEN_SIZE +
                                       r] += v * w.o_eps;
                          });
#endif
    __syncthreads();
  }
}

#if NK_FP4_T2_OPROJ
// Experimental batch-throughput path: serve two 16-token tiles per O-weight
// pass.  The caller retains T=1 below 64 packed tokens, where 128 units from a
// single supertile do not fill the persistent compute grid.
__device__ void nk_oproj_mma4_t2(int seq_len,
                                  const __half *__restrict__ g_attn,
                                  char *__restrict__ s_arena,
                                  const NKLayerWeights &w,
                                  float *__restrict__ g_residual) {
  constexpr int T2_TILE = 2 * TOKEN_TILE;
  constexpr int T2_FRAG_FLOATS = 512;
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + T2_TILE * NK_BQ_PITCH_O;
  float *s_part = reinterpret_cast<float *>(
      s_arena + T2_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O));

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS; // 128
  constexpr int SLICE = Q_SIZE / NK_KSPLIT;             // 192 = 3 chunks
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int n_tiles = (seq_len + T2_TILE - 1) / T2_TILE;
  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * T2_TILE;
    int tcount = min(T2_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      for (int tr = warp_id; tr < T2_TILE; tr += TOKEN_TILE) {
        unsigned char *bq_row = s_bq + (size_t)tr * NK_BQ_PITCH_O;
        unsigned char *sf_row = s_sf + (size_t)tr * NK_SF_PITCH_O;
        if (tr < tcount)
          nk_quant_row(g_attn + (size_t)(tile_base + tr) * Q_SIZE,
                       Q_SIZE / 16, bq_row, sf_row, w.ao_i6, w.ao, lane);
        else
          nk_zero_qrow(bq_row, Q_SIZE / 16, sf_row, lane);
      }
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[1] = {{w.o.w + (size_t)(row0 + rg * 16) * (Q_SIZE / 2),
                       w.o.s + (size_t)(row0 + rg * 16) * (Q_SIZE / 16)}};
    float c[1][4][4] = {};
    nk_mma4_slice_t2<Q_SIZE, NK_BQ_PITCH_O, NK_SF_PITCH_O, SLICE / 64>(
        wb, s_bq, s_sf, ks * SLICE, ks * SLICE, c, lane);

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * T2_FRAG_FLOATS + lane * 16 + h * 4 + j] =
            c[0][h][j];
    __syncthreads();

    if (warp_id < NK_RG) {
      for (int i = lane; i < T2_FRAG_FLOATS; i += WARP_SIZE) {
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          v += s_part[(size_t)(warp_id * NK_KSPLIT + s) *
                          T2_FRAG_FLOATS +
                      i];
        int src_lane = i >> 4;
        int hj = i & 15;
        int h = hj >> 2;
        int j = hj & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + warp_id * 16 + nk_c_row(j, src_lane);
          g_residual[(size_t)(tile_base + t) * HIDDEN_SIZE + r] +=
              v * w.o_eps;
        }
      }
    }
    __syncthreads();
  }
}
#endif

// --- Phase D2 (W4A4): post-norm + gate/up/SiLU ---
__device__ void nk_gateup_mma4(int seq_len, int n_tiles,
                               const float *__restrict__ g_residual,
                               char *__restrict__ s_arena,
                               const NKLayerWeights &w,
                               __half *__restrict__ g_mlp) {
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf = reinterpret_cast<unsigned char *>(s_arena) + NK_Q_SF_OFF;
  __half *s_x = reinterpret_cast<__half *>(s_arena + NK_Q_R3_OFF);
  float *s_part = reinterpret_cast<float *>(s_arena + NK_Q_R3_OFF);

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = INTERMEDIATE_SIZE / NK_UNIT_ROWS; // 384
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT;
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      nk_postnorm_tile(tile_base, seq_len, g_residual, w.post_ln, s_x);
      __syncthreads();
      nk_quant_tile(s_x, s_bq, s_sf, w.agu_i6, w.agu);
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[2] = {
        {w.gate.w + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 2),
         w.gate.s + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 16)},
        {w.up.w + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 2),
         w.up.s + (size_t)(row0 + rg * 16) * (HIDDEN_SIZE / 16)}};
    float c[2][2][4] = {};
    nk_mma4_slice<HIDDEN_SIZE, NK_BQ_PITCH_H, NK_SF_PITCH_H, SLICE / 64, 2>(
        wb, s_bq, s_sf, ks * SLICE, ks * SLICE, c, lane);

#pragma unroll
    for (int s = 0; s < 2; s++)
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          s_part[((size_t)warp_id * 2 + s) * 256 + lane * 8 + h * 4 + j] =
              c[s][h][j];
    __syncthreads();

    if (warp_id < NK_RG) {
      int crg = warp_id;
      for (int i = lane; i < 256; i += WARP_SIZE) {
        float g = 0.0f, up = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++) {
          size_t wp = (size_t)(crg * NK_KSPLIT + s) * 2;
          g += s_part[wp * 256 + i];
          up += s_part[(wp + 1) * 256 + i];
        }
        int src_lane = i >> 3;
        int h = (i >> 2) & 1;
        int j = i & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + crg * 16 + nk_c_row(j, src_lane);
          g_mlp[(size_t)(tile_base + t) * INTERMEDIATE_SIZE + r] =
              __float2half(nk_silu(g * w.gate_eps) * (up * w.up_eps));
        }
      }
    }
    __syncthreads();
  }
}

#if NK_DUAL_CTA
// One-pass post-attention normalized packing after nk_residual_rstd_pass.
__device__ __forceinline__ void nk_postnorm_quant_tile_compact(
    int tile_base, int seq_len, const float *__restrict__ g_residual,
    const __nv_bfloat16 *__restrict__ norm_weight,
    const float *__restrict__ g_rstd,
    unsigned char *__restrict__ s_bq, unsigned char *__restrict__ s_sf,
    float inv6a, float a) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (warp >= TOKEN_TILE)
    return;
  int t = tile_base + warp;
  unsigned char *bq_row = s_bq + (size_t)warp * NK_BQ_PITCH_H;
  unsigned char *sf_row = s_sf + (size_t)warp * NK_SF_PITCH_H;
  if (t >= seq_len) {
    nk_zero_qrow(bq_row, HIDDEN_SIZE / 16, sf_row, lane);
    return;
  }
  const float *src = g_residual + (size_t)t * HIDDEN_SIZE;
  float rstd = g_rstd[t];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int group = lane + q * WARP_SIZE;
    const float4 *vsrc =
        reinterpret_cast<const float4 *>(src + group * 16);
    __half2 h[8];
#pragma unroll
    for (int v = 0; v < 4; v++) {
      float4 x = vsrc[v];
      int i = group * 16 + v * 4;
      float w0 = __bfloat162float(__ldg(norm_weight + i));
      float w1 = __bfloat162float(__ldg(norm_weight + i + 1));
      float w2 = __bfloat162float(__ldg(norm_weight + i + 2));
      float w3 = __bfloat162float(__ldg(norm_weight + i + 3));
      h[v * 2] = __floats2half2_rn(x.x * rstd * w0, x.y * rstd * w1);
      h[v * 2 + 1] =
          __floats2half2_rn(x.z * rstd * w2, x.w * rstd * w3);
    }
    uint2 packed;
    unsigned char sfb;
    nk_quant16(h, inv6a, a, packed, sfb);
    *reinterpret_cast<uint2 *>(bq_row + group * 8) = packed;
    sf_row[group] = sfb;
  }
}

// Compact exact gate/up path. Gate and up use the same activation tile and
// float partial buffer sequentially; a 1-KB gate tile bridges the two passes.
// This trades a second shared-B read for lower shared memory and register use.
__device__ void nk_gateup_mma4_compact(
    int seq_len, int n_tiles, const float *__restrict__ g_residual,
    const float *__restrict__ g_rstd, char *__restrict__ s_arena,
    const NKLayerWeights &w, __half *__restrict__ g_mlp) {
  static_assert(NK_RG == 1, "compact gate/up assumes one row group");
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf = reinterpret_cast<unsigned char *>(s_arena) + NK_Q_SF_OFF;
  float *s_part = reinterpret_cast<float *>(s_arena + NK_Q_R3_OFF);
  float *s_gate = s_part + NK_NUM_WARPS * 256;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = INTERMEDIATE_SIZE / NK_UNIT_ROWS;
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT;
  int ks = warp % NK_KSPLIT;
  int staged = -1;
  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int row0 = (tp % NUM_UNITS) * NK_UNIT_ROWS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);
    if (tile != staged) {
      __syncthreads();
      nk_postnorm_quant_tile_compact(
          tile_base, seq_len, g_residual, w.post_ln, g_rstd, s_bq, s_sf,
          w.agu_i6, w.agu);
      __syncthreads();
      staged = tile;
    }

    NKQuantW gate_w[1] = {{
        w.gate.w + (size_t)row0 * (HIDDEN_SIZE / 2),
        w.gate.s + (size_t)row0 * (HIDDEN_SIZE / 16)}};
    float gate_c[1][2][4] = {};
    nk_mma4_slice<HIDDEN_SIZE, NK_BQ_PITCH_H, NK_SF_PITCH_H,
                  SLICE / 64, 1>(gate_w, s_bq, s_sf, ks * SLICE,
                                 ks * SLICE, gate_c, lane);
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp * 256 + lane * 8 + h * 4 + j] =
            gate_c[0][h][j];
    __syncthreads();
#if NK_PARALLEL_COMPACT_COMBINE
    constexpr int COMBINE_THREADS = NK_PARALLEL_COMBINE_WARPS * WARP_SIZE;
    if (threadIdx.x < COMBINE_THREADS) {
      for (int i = threadIdx.x; i < 256; i += COMBINE_THREADS) {
        float value = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          value += s_part[(size_t)s * 256 + i];
        s_gate[i] = value;
      }
    }
#else
    if (warp == 0) {
      for (int i = lane; i < 256; i += WARP_SIZE) {
        float value = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          value += s_part[(size_t)s * 256 + i];
        s_gate[i] = value;
      }
    }
#endif
    __syncthreads();

    NKQuantW up_w[1] = {{
        w.up.w + (size_t)row0 * (HIDDEN_SIZE / 2),
        w.up.s + (size_t)row0 * (HIDDEN_SIZE / 16)}};
    float up_c[1][2][4] = {};
    nk_mma4_slice<HIDDEN_SIZE, NK_BQ_PITCH_H, NK_SF_PITCH_H,
                  SLICE / 64, 1>(up_w, s_bq, s_sf, ks * SLICE,
                                 ks * SLICE, up_c, lane);
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp * 256 + lane * 8 + h * 4 + j] =
            up_c[0][h][j];
    __syncthreads();
#if NK_PARALLEL_COMPACT_COMBINE
    if (threadIdx.x < COMBINE_THREADS) {
      for (int i = threadIdx.x; i < 256; i += COMBINE_THREADS) {
        float up = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          up += s_part[(size_t)s * 256 + i];
        int src_lane = i >> 3;
        int h = (i >> 2) & 1;
        int j = i & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + nk_c_row(j, src_lane);
          g_mlp[(size_t)(tile_base + t) * INTERMEDIATE_SIZE + r] =
              __float2half(nk_silu(s_gate[i] * w.gate_eps) *
                           (up * w.up_eps));
        }
      }
    }
#else
    if (warp == 0) {
      for (int i = lane; i < 256; i += WARP_SIZE) {
        float up = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          up += s_part[(size_t)s * 256 + i];
        int src_lane = i >> 3;
        int h = (i >> 2) & 1;
        int j = i & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + nk_c_row(j, src_lane);
          g_mlp[(size_t)(tile_base + t) * INTERMEDIATE_SIZE + r] =
              __float2half(nk_silu(s_gate[i] * w.gate_eps) *
                           (up * w.up_eps));
        }
      }
    }
#endif
    __syncthreads();
  }
}

#endif

#if NK_FP4_T2_GATEUP
// Exact T=2 gate/up prototype.  A compact gate result lets gate and up reuse
// one 32-KB partial buffer, keeping the complete layout below the base arena.
__device__ void nk_gateup_mma4_t2(int seq_len,
                                  const float *__restrict__ g_residual,
                                  char *__restrict__ s_arena,
                                  const NKLayerWeights &w,
                                  __half *__restrict__ g_mlp) {
  constexpr int T2_TILE = 2 * TOKEN_TILE;
  constexpr int T2_FRAG_FLOATS = 512;
  static_assert(NK_RG == 1, "T2 gate/up compact combine assumes one row group");

  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + T2_TILE * NK_BQ_PITCH_H;
  float *s_part = reinterpret_cast<float *>(
      s_arena + T2_TILE * (NK_BQ_PITCH_H + NK_SF_PITCH_H));
  float *s_gate = s_part + NK_NUM_WARPS * T2_FRAG_FLOATS;

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = INTERMEDIATE_SIZE / NK_UNIT_ROWS; // 384
  constexpr int SLICE = HIDDEN_SIZE / NK_KSPLIT;              // 128
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int n_tiles = (seq_len + T2_TILE - 1) / T2_TILE;
  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * T2_TILE;
    int tcount = min(T2_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      nk_postnorm_quant_t2(tile_base, seq_len, g_residual, w.post_ln, s_bq,
                           s_sf, w.agu_i6, w.agu);
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[2] = {
        {w.gate.w + (size_t)row0 * (HIDDEN_SIZE / 2),
         w.gate.s + (size_t)row0 * (HIDDEN_SIZE / 16)},
        {w.up.w + (size_t)row0 * (HIDDEN_SIZE / 2),
         w.up.s + (size_t)row0 * (HIDDEN_SIZE / 16)}};
    float c[2][4][4] = {};
    nk_mma4_slice_t2_na2<HIDDEN_SIZE, NK_BQ_PITCH_H, NK_SF_PITCH_H,
                          SLICE / 64>(wb, s_bq, s_sf, ks * SLICE, ks * SLICE,
                                      c, lane);

#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * T2_FRAG_FLOATS + lane * 16 + h * 4 + j] =
            c[0][h][j];
    __syncthreads();

    if (warp_id == 0) {
      for (int i = lane; i < T2_FRAG_FLOATS; i += WARP_SIZE) {
        float g = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          g += s_part[(size_t)s * T2_FRAG_FLOATS + i];
        s_gate[i] = g;
      }
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * T2_FRAG_FLOATS + lane * 16 + h * 4 + j] =
            c[1][h][j];
    __syncthreads();

    if (warp_id == 0) {
      for (int i = lane; i < T2_FRAG_FLOATS; i += WARP_SIZE) {
        float up = 0.0f;
#pragma unroll
        for (int s = 0; s < NK_KSPLIT; s++)
          up += s_part[(size_t)s * T2_FRAG_FLOATS + i];
        int src_lane = i >> 4;
        int hj = i & 15;
        int h = hj >> 2;
        int j = hj & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + nk_c_row(j, src_lane);
          g_mlp[(size_t)(tile_base + t) * INTERMEDIATE_SIZE + r] =
              __float2half(nk_silu(s_gate[i] * w.gate_eps) *
                           (up * w.up_eps));
        }
      }
    }
    __syncthreads();
  }
}
#endif

// --- Phase D3 (W4A4): down projection, K=6144 in ONE packed stage ---
__device__ void nk_down_mma4(int seq_len, int n_tiles,
                             const __half *__restrict__ g_mlp,
                             char *__restrict__ s_arena,
                             const NKLayerWeights &w,
                             const float *__restrict__ g_residual,
                             __half *__restrict__ hidden_out) {
  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + TOKEN_TILE * NK_BQ_PITCH_D;
  float *s_part = reinterpret_cast<float *>(
      s_arena + TOKEN_TILE * (NK_BQ_PITCH_D + NK_SF_PITCH_D));

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS; // 128
  constexpr int SLICE = INTERMEDIATE_SIZE / NK_KSPLIT;  // 384 = 6 chunks
  int rg = warp_id / NK_KSPLIT;
  int ks = warp_id % NK_KSPLIT;
  int staged = -1;

  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp0 = b * q + min(b, rem);
  int tp_end = tp0 + q + (b < rem ? 1 : 0);
#pragma unroll 1
  for (int tp = tp0; tp < tp_end; tp++) {
    int tile = tp / NUM_UNITS;
    int u = tp % NUM_UNITS;
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

    if (tile != staged) {
      __syncthreads();
      if (warp_id < TOKEN_TILE) {
        unsigned char *bq_row = s_bq + (size_t)warp_id * NK_BQ_PITCH_D;
        unsigned char *sf_row = s_sf + (size_t)warp_id * NK_SF_PITCH_D;
        if (warp_id < tcount)
          nk_quant_row(g_mlp + (size_t)(tile_base + warp_id) *
                                   INTERMEDIATE_SIZE,
                       INTERMEDIATE_SIZE / 16, bq_row, sf_row, w.adown_i6,
                       w.adown, lane);
        else
          nk_zero_qrow(bq_row, INTERMEDIATE_SIZE / 16, sf_row, lane);
      }
      __syncthreads();
      staged = tile;
    }

    int row0 = u * NK_UNIT_ROWS;
    NKQuantW wb[1] = {
        {w.down.w + (size_t)(row0 + rg * 16) * (INTERMEDIATE_SIZE / 2),
         w.down.s + (size_t)(row0 + rg * 16) * (INTERMEDIATE_SIZE / 16)}};
    float c[1][2][4] = {};
    nk_mma4_slice<INTERMEDIATE_SIZE, NK_BQ_PITCH_D, NK_SF_PITCH_D, SLICE / 64,
                  1>(wb, s_bq, s_sf, ks * SLICE, ks * SLICE, c, lane);

    __syncthreads();
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int j = 0; j < 4; j++)
        s_part[(size_t)warp_id * 256 + lane * 8 + h * 4 + j] = c[0][h][j];
    __syncthreads();

    if (warp_id < NK_RG)
      nk_combine_partials(s_part, warp_id, row0, tcount, lane,
                          [&](int r, int t, float v) {
                            size_t gt =
                                (size_t)(tile_base + t) * HIDDEN_SIZE + r;
                            hidden_out[gt] =
                                __float2half(v * w.down_eps + g_residual[gt]);
                          });
    __syncthreads();
  }
}

#if NK_DUAL_CTA
// Compact down projection. The 6144-wide activation is staged in two 3072-K
// halves. A coalesced float workspace carries the exact sum of K slices 0..7
// while the same 43.31-KB shared arena is reused for slices 8..15.
__device__ void nk_down_mma4_compact(
    int seq_len, const __half *__restrict__ g_mlp,
    char *__restrict__ s_arena, const NKLayerWeights &w,
    const float *__restrict__ g_residual,
    float *__restrict__ g_down_partial,
    __half *__restrict__ hidden_out) {
  static_assert(NK_RG == 1, "compact down assumes one row group");
  static_assert((NK_KSPLIT & 1) == 0, "compact down needs even K split");
  constexpr int KHALF = INTERMEDIATE_SIZE / 2;
  constexpr int SLICE = INTERMEDIATE_SIZE / NK_KSPLIT;
  constexpr int HALF_SPLITS = NK_KSPLIT / 2;
  constexpr int NUM_UNITS = HIDDEN_SIZE / NK_UNIT_ROWS;

  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + TOKEN_TILE * NK_BQ_PITCH_O;
  float *s_part = reinterpret_cast<float *>(
      s_arena + TOKEN_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O));
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int ks = warp % NK_KSPLIT;

  int n_tiles = (seq_len + TOKEN_TILE - 1) / TOKEN_TILE;
  int total = n_tiles * NUM_UNITS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp = b * q + min(b, rem);
  int tp_end = tp + q + (b < rem ? 1 : 0);
  while (tp < tp_end) {
    int tile = tp / NUM_UNITS;
    int tile_end = min(tp_end, (tile + 1) * NUM_UNITS);
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

#pragma unroll
    for (int half = 0; half < 2; half++) {
      __syncthreads();
      if (warp < TOKEN_TILE) {
        unsigned char *bq_row = s_bq + (size_t)warp * NK_BQ_PITCH_O;
        unsigned char *sf_row = s_sf + (size_t)warp * NK_SF_PITCH_O;
        if (warp < tcount) {
          nk_quant_row(g_mlp + (size_t)(tile_base + warp) *
                                   INTERMEDIATE_SIZE +
                                   half * KHALF,
                       KHALF / 16, bq_row, sf_row, w.adown_i6,
                       w.adown, lane);
        } else {
          nk_zero_qrow(bq_row, KHALF / 16, sf_row, lane);
        }
      }
      __syncthreads();

      for (int work = tp; work < tile_end; work++) {
        int row0 = (work % NUM_UNITS) * NK_UNIT_ROWS;
        bool active = ks / HALF_SPLITS == half;
        if (active) {
          NKQuantW wb[1] = {{
              w.down.w + (size_t)row0 * (INTERMEDIATE_SIZE / 2),
              w.down.s + (size_t)row0 * (INTERMEDIATE_SIZE / 16)}};
          float c[1][2][4] = {};
          int local_ks = ks - half * HALF_SPLITS;
          nk_mma4_slice<INTERMEDIATE_SIZE, NK_BQ_PITCH_O,
                        NK_SF_PITCH_O, SLICE / 64, 1>(
              wb, s_bq, s_sf, ks * SLICE, local_ks * SLICE, c, lane);
#pragma unroll
          for (int h = 0; h < 2; h++)
#pragma unroll
            for (int j = 0; j < 4; j++)
              s_part[(size_t)warp * 256 + lane * 8 + h * 4 + j] =
                  c[0][h][j];
        }
        __syncthreads();

#if NK_PARALLEL_COMPACT_COMBINE
        constexpr int COMBINE_THREADS =
            NK_PARALLEL_COMBINE_WARPS * WARP_SIZE;
        if (threadIdx.x < COMBINE_THREADS) {
          for (int i = threadIdx.x; i < 256; i += COMBINE_THREADS) {
            int src_lane = i >> 3;
            int h = (i >> 2) & 1;
            int j = i & 3;
            int t = nk_c_token(h, j, src_lane);
            if (t < tcount) {
              int r = row0 + nk_c_row(j, src_lane);
              size_t gt = (size_t)(tile_base + t) * HIDDEN_SIZE + r;
              float value = half ? g_down_partial[gt] : 0.0f;
#pragma unroll
              for (int s = 0; s < HALF_SPLITS; s++)
                value +=
                    s_part[(size_t)(half * HALF_SPLITS + s) * 256 + i];
              if (half == 0)
                g_down_partial[gt] = value;
              else
                hidden_out[gt] = __float2half(
                    value * w.down_eps + g_residual[gt]);
            }
          }
        }
#else
        if (warp == 0) {
          for (int i = lane; i < 256; i += WARP_SIZE) {
            int src_lane = i >> 3;
            int h = (i >> 2) & 1;
            int j = i & 3;
            int t = nk_c_token(h, j, src_lane);
            if (t < tcount) {
              int r = row0 + nk_c_row(j, src_lane);
              size_t gt = (size_t)(tile_base + t) * HIDDEN_SIZE + r;
              float value = half ? g_down_partial[gt] : 0.0f;
#pragma unroll
              for (int s = 0; s < HALF_SPLITS; s++)
                value += s_part[(size_t)(half * HALF_SPLITS + s) * 256 + i];
              if (half == 0)
                g_down_partial[gt] = value;
              else
                hidden_out[gt] = __float2half(
                    value * w.down_eps + g_residual[gt]);
            }
          }
        }
#endif
        __syncthreads();
      }
    }
    tp = tile_end;
  }
}

#if NK_DUAL_DOWN_ROWPAIR
// Exact two-row-group down path. Each 3072-K half has eight K slices, so the
// two eight-warp groups compute adjacent 16-row units concurrently. The
// reduction keeps the former slice order and the existing global half-sum.
__device__ void nk_down_mma4_compact_rowpair(
    int seq_len, const __half *__restrict__ g_mlp,
    char *__restrict__ s_arena, const NKLayerWeights &w,
    const float *__restrict__ g_residual,
    float *__restrict__ g_down_partial,
    __half *__restrict__ hidden_out) {
  static_assert(NK_RG == 1, "compact down rowpair assumes NK_RG=1");
  static_assert((NK_KSPLIT & 1) == 0,
                "compact down rowpair needs even K split");
  constexpr int KHALF = INTERMEDIATE_SIZE / 2;
  constexpr int SLICE = INTERMEDIATE_SIZE / NK_KSPLIT;
  constexpr int HALF_SPLITS = NK_KSPLIT / 2;
  constexpr int ROW_GROUPS = NK_NUM_WARPS / HALF_SPLITS;
  constexpr int ROWS_PER_PAIR = ROW_GROUPS * 16;
  constexpr int NUM_PAIRS = HIDDEN_SIZE / ROWS_PER_PAIR;
  static_assert(ROW_GROUPS == 2, "down rowpair expects two warp groups");
  static_assert(HIDDEN_SIZE % ROWS_PER_PAIR == 0,
                "down output rows must form complete pairs");

  unsigned char *s_bq = reinterpret_cast<unsigned char *>(s_arena);
  unsigned char *s_sf =
      reinterpret_cast<unsigned char *>(s_arena) + TOKEN_TILE * NK_BQ_PITCH_O;
  float *s_part = reinterpret_cast<float *>(
      s_arena + TOKEN_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O));
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int row_group = warp / HALF_SPLITS;
  int local_ks = warp % HALF_SPLITS;

  int n_tiles = (seq_len + TOKEN_TILE - 1) / TOKEN_TILE;
  int total = n_tiles * NUM_PAIRS;
  int q = total / NK_COMPUTE_BLOCKS;
  int rem = total % NK_COMPUTE_BLOCKS;
  int b = (int)blockIdx.x;
  int tp = b * q + min(b, rem);
  int tp_end = tp + q + (b < rem ? 1 : 0);
  while (tp < tp_end) {
    int tile = tp / NUM_PAIRS;
    int tile_end = min(tp_end, (tile + 1) * NUM_PAIRS);
    int tile_base = tile * TOKEN_TILE;
    int tcount = min(TOKEN_TILE, seq_len - tile_base);

#pragma unroll
    for (int half = 0; half < 2; half++) {
      __syncthreads();
      if (warp < TOKEN_TILE) {
        unsigned char *bq_row = s_bq + (size_t)warp * NK_BQ_PITCH_O;
        unsigned char *sf_row = s_sf + (size_t)warp * NK_SF_PITCH_O;
        if (warp < tcount) {
          nk_quant_row(g_mlp + (size_t)(tile_base + warp) *
                                   INTERMEDIATE_SIZE +
                                   half * KHALF,
                       KHALF / 16, bq_row, sf_row, w.adown_i6,
                       w.adown, lane);
        } else {
          nk_zero_qrow(bq_row, KHALF / 16, sf_row, lane);
        }
      }
      __syncthreads();

      for (int work = tp; work < tile_end; work++) {
        int row0 = (work % NUM_PAIRS) * ROWS_PER_PAIR;
        int row_base = row0 + row_group * 16;
        NKQuantW wb[1] = {{
            w.down.w + (size_t)row_base * (INTERMEDIATE_SIZE / 2),
            w.down.s + (size_t)row_base * (INTERMEDIATE_SIZE / 16)}};
        float c[1][2][4] = {};
        int ks = half * HALF_SPLITS + local_ks;
        nk_mma4_slice<INTERMEDIATE_SIZE, NK_BQ_PITCH_O,
                      NK_SF_PITCH_O, SLICE / 64, 1>(
            wb, s_bq, s_sf, ks * SLICE, local_ks * SLICE, c, lane);
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
          for (int j = 0; j < 4; j++)
            s_part[(size_t)warp * 256 + lane * 8 + h * 4 + j] =
                c[0][h][j];
        __syncthreads();

        int flat = threadIdx.x;
        int combine_group = flat >> 8;
        int i = flat & 255;
        int src_lane = i >> 3;
        int h = (i >> 2) & 1;
        int j = i & 3;
        int t = nk_c_token(h, j, src_lane);
        if (t < tcount) {
          int r = row0 + combine_group * 16 + nk_c_row(j, src_lane);
          size_t gt = (size_t)(tile_base + t) * HIDDEN_SIZE + r;
          float value = half ? g_down_partial[gt] : 0.0f;
#pragma unroll
          for (int s = 0; s < HALF_SPLITS; s++)
            value += s_part[(size_t)(combine_group * HALF_SPLITS + s) *
                                256 +
                            i];
          if (half == 0)
            g_down_partial[gt] = value;
          else
            hidden_out[gt] =
                __float2half(value * w.down_eps + g_residual[gt]);
        }
        __syncthreads();
      }
    }
    tp = tile_end;
  }
}
#endif
#endif

#endif // NK_FP4MMA

// =============================================================================
// Finalize.  Mean pooling needs the per-token final RMSNorm (rstd varies per
// token, so it cannot fold into L2 normalization like last-token pooling):
//   F1: warp per token: g_rstd[t] = rsqrt(mean(hidden[t]^2) + eps)
//   F2: block per sequence: pooled[i] = w_norm[i] * sum_t hidden[t][i] *
//       rstd[t]  (sequential token loop = deterministic), then L2-normalize.
//       The 1/T of the mean cancels in the L2 norm.
// =============================================================================

__device__ void nk_rstd_pass(int seq_len, const __half *__restrict__ hidden,
                             float *__restrict__ g_rstd) {
  int warp_gid = blockIdx.x * NK_NUM_WARPS + threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  constexpr int TOTAL_WARPS = NK_COMPUTE_BLOCKS * NK_NUM_WARPS;
  for (int t = warp_gid; t < seq_len; t += TOTAL_WARPS) {
    const uint4 *row =
        reinterpret_cast<const uint4 *>(hidden + (size_t)t * HIDDEN_SIZE);
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < HIDDEN_SIZE / (WARP_SIZE * 8); j++) { // 8 halves/uint4
      uint4 u = row[lane_id + j * WARP_SIZE];
      const __half2 *h2 = reinterpret_cast<const __half2 *>(&u);
#pragma unroll
      for (int p = 0; p < 4; p++) {
        float2 f = __half22float2(h2[p]);
        ss += f.x * f.x + f.y * f.y;
      }
    }
    ss = nk_warp_reduce_sum(ss);
    if (lane_id == 0)
      g_rstd[t] = rsqrtf(ss / float(HIDDEN_SIZE) + NK_RMS_EPS);
  }
}

__device__ void nk_pool_finalize(int first, int count,
                                 const __half *__restrict__ hidden,
                                 const float *__restrict__ g_rstd,
                                 const __nv_bfloat16 *__restrict__ final_norm_w,
                                 float *__restrict__ g_out) {
  __shared__ float s_red[NK_NUM_WARPS];
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  constexpr int PER_THREAD = HIDDEN_SIZE / NK_BLOCK_SIZE; // 4
  int d0 = threadIdx.x * PER_THREAD;
  float acc[PER_THREAD] = {};
  for (int t = first; t < first + count; t++) {
    float r = g_rstd[t];
    uint2 u = *reinterpret_cast<const uint2 *>(hidden + (size_t)t * HIDDEN_SIZE +
                                               d0);
    const __half2 *h2 = reinterpret_cast<const __half2 *>(&u);
    float2 f0 = __half22float2(h2[0]);
    float2 f1 = __half22float2(h2[1]);
    acc[0] += f0.x * r;
    acc[1] += f0.y * r;
    acc[2] += f1.x * r;
    acc[3] += f1.y * r;
  }
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < PER_THREAD; j++) {
    acc[j] *= __bfloat162float(__ldg(final_norm_w + d0 + j));
    ss += acc[j] * acc[j];
  }
  ss = nk_warp_reduce_sum(ss);
  if (lane_id == 0)
    s_red[warp_id] = ss;
  __syncthreads();
  if (warp_id == 0) {
    float sum = (lane_id < NK_NUM_WARPS) ? s_red[lane_id] : 0.0f;
    sum = nk_warp_reduce_sum(sum);
    if (lane_id == 0)
      s_red[0] = 1.0f / sqrtf(fmaxf(sum, 1e-24f));
  }
  __syncthreads();
  float inv_norm = s_red[0];
#pragma unroll
  for (int j = 0; j < PER_THREAD; j++)
    g_out[d0 + j] = acc[j] * inv_norm;
  __syncthreads(); // s_red reused when a block finalizes several sequences
}

// =============================================================================
// The embedding megakernel
// =============================================================================

// Shared-memory arena (bytes).
// f16 engine (disjoint lifetimes, rows padded by NK_XPAD f16):
//   A:    f16 x-tile   16*(2048+8)*2 = 64.25 KB + s_part 16 KB
//   D1/D3 f16 k-tile   16*(1536+8)*2 = 48.25 KB + s_part 16 KB
//   D2:   f16 x-tile   64.25 KB + s_part 32 KB  <- max, 96.25 KB
// W4A4 engine:
//   A/D2: packed 16.25 KB + sf 2.06 KB + (f16 x-tile 64.25 KB, dead after
//         quant, unioned with s_part) <- max, 82.56 KB
//   D2 T=2: packed 32.50 + sf 4.13 + s_part 32 + gate 2 = 70.63 KB
//   D1:   packed 24.25 + sf 3.06 + s_part 16 = 43.31 KB (T=1)
//         T=2 experiment: packed 48.50 + sf 6.13 + s_part 32 = 86.63 KB
//   D3:   packed 48.25 + sf 6.06 + s_part 16 = 70.31 KB
#if NK_FP4MMA
constexpr int NK_FP4_BASE_ARENA_BYTES =
    TOKEN_TILE * (HIDDEN_SIZE / 2 + 16) + TOKEN_TILE * (HIDDEN_SIZE / 16 + 4) +
    TOKEN_TILE * (HIDDEN_SIZE + NK_XPAD) * 2;
#if NK_DUAL_CTA
// Compact QKV/gate fit below this; O and half-staged down both use 43.31 KB.
constexpr int NK_ARENA_BYTES =
    TOKEN_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O) +
    NK_NUM_WARPS * 256 * 4;
#elif NK_FP4_T2_OPROJ
constexpr int NK_FP4_T2_OPROJ_ARENA_BYTES =
    2 * TOKEN_TILE * (NK_BQ_PITCH_O + NK_SF_PITCH_O) +
    NK_NUM_WARPS * 512 * 4;
constexpr int NK_ARENA_BYTES =
    NK_FP4_T2_OPROJ_ARENA_BYTES > NK_FP4_BASE_ARENA_BYTES
        ? NK_FP4_T2_OPROJ_ARENA_BYTES
        : NK_FP4_BASE_ARENA_BYTES;
#else
constexpr int NK_ARENA_BYTES = NK_FP4_BASE_ARENA_BYTES;
#endif
#else
constexpr int NK_ARENA_BYTES =
    TOKEN_TILE * (HIDDEN_SIZE + NK_XPAD) * 2 + NK_NUM_WARPS * 2 * 256 * 4;
#endif

#if NK_MMA_ATTN
constexpr int NK_MMA_ATTN_STAGES = NK_MMA_ATTN_TMA ? 2 : 1;
constexpr int NK_MMA_ATTN_ARENA_BYTES =
    sizeof(__half) *
        ((NUM_Q_HEADS / NUM_KV_HEADS) * 16 * HEAD_DIM +
         2 * NK_MMA_ATTN_STAGES * NK_MMA_ATTN_KN * HEAD_DIM +
         (NUM_Q_HEADS / NUM_KV_HEADS) * 16 * NK_MMA_ATTN_KN) +
    sizeof(float) *
        ((NUM_Q_HEADS / NUM_KV_HEADS) * 16 * NK_MMA_ATTN_KN +
         3 * (NUM_Q_HEADS / NUM_KV_HEADS) * 16 + 16) +
    (NK_MMA_ATTN_TMA ? 2 * sizeof(uint64_t) : 0);
static_assert(!NK_MMA_ATTN_TMA || NK_MMA_ATTN_ARENA_BYTES == 33936,
              "TMA attention shared layout changed unexpectedly");
static_assert(NK_MMA_ATTN_ARENA_BYTES <= NK_ARENA_BYTES,
              "MMA attention workspace exceeds persistent shared arena");
#endif

__global__ void __launch_bounds__(NK_BLOCK_SIZE, NK_MIN_BLOCKS_PER_SM)
nk_embed_kernel(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const NKLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __half *__restrict__ cos_table, const __half *__restrict__ sin_table,
    const int *__restrict__ d_ids, int seq_len,
    // Packed batching: seq_start[t] / seq_end[t] = packed index of token
    // t's sequence start / one-past-end (null = one sequence spanning all),
    // seq_first[b] + seq_count[b] describe sequence b for pooling.
    const int *__restrict__ seq_start, const int *__restrict__ seq_end,
    const int *__restrict__ seq_first, const int *__restrict__ seq_count,
    int batch, int max_item_len, int mma_attention_from_layer,
    __half *__restrict__ k_cache, __half *__restrict__ v_cache,
    __half *__restrict__ g_q, __half *__restrict__ g_k,
    __half *__restrict__ g_attn, float *__restrict__ g_residual,
    __half *__restrict__ g_mlp, __half *__restrict__ hidden,
    float *__restrict__ g_rstd, float *__restrict__ g_out,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, int num_layers, int max_seq,
    float attn_scale) {
  extern __shared__ __align__(16) char s_arena[];
  int block_id = blockIdx.x;

#if NK_STREAMER_BLOCKS > 0
  if (block_id >= NK_COMPUTE_BLOCKS) {
    nk_streamer(layer_weights, num_layers, kv_flag);
    return;
  }
#endif

  AtomicGridSync grid{barrier_counter, barrier_sense,
                      (unsigned int)NK_COMPUTE_BLOCKS, 0};

  int n_tiles = (seq_len + TOKEN_TILE - 1) / TOKEN_TILE;
#if NK_DUAL_CTA
  float *g_down_partial = g_rstd + max_seq;
#endif

  for (int layer = 0; layer < num_layers; layer++) {
    const NKLayerWeights &w = layer_weights[layer];
    const __half *layer_in = (layer == 0) ? nullptr : hidden;

    // --- Phase A: input norm + QKV projection ---
#if !defined(NK_SKIP_QKV)
#if NK_DUAL_CTA
    nk_input_rstd_pass(seq_len, layer_in, embed_weight, d_ids, g_residual,
                       g_rstd);
    grid.sync();
#endif
#if NK_FP4MMA
    nk_qkv_mma4(seq_len, n_tiles, layer_in, embed_weight, d_ids, g_residual,
                s_arena, w, g_q, g_k, v_cache, max_seq, g_rstd);
#else
    {
      // arena: [ s_x f16 16x(2048+8) | s_part fp32 16x256 ]
      __half *s_x = reinterpret_cast<__half *>(s_arena);
      float *s_part = reinterpret_cast<float *>(
          s_arena + (size_t)TOKEN_TILE * (HIDDEN_SIZE + NK_XPAD) * 2);
      nk_qkv_mma(seq_len, n_tiles, layer_in, embed_weight, d_ids, w.input_ln,
                 g_residual, s_x, s_part, w, g_q, g_k, v_cache, max_seq);
    }
#endif
#endif
    grid.sync();
    if (block_id == 0 && threadIdx.x == 0) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      atomicExch(kv_flag, (unsigned int)(layer + 1));
    }

    // --- Phase B: K RoPE + K cache ---
#if !defined(NK_SKIP_ATTN)
    nk_k_rope(seq_len, seq_start, g_k, cos_table, sin_table, k_cache, max_seq);
#endif
    grid.sync();

    // --- Phase C: bidirectional attention (Q RoPE at warp entry) ---
#if !defined(NK_SKIP_ATTN)
#if NK_MMA_ATTN
    if (max_item_len >= NK_MMA_ATTN_MIN_SEQ &&
        layer >= mma_attention_from_layer) {
      nk_attention(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                   k_cache, v_cache, g_attn, max_seq, attn_scale,
                   NK_MMA_ATTN_MIN_SEQ);
#if NK_FUSED_GQA_ATTN
      nk_attention_gqa(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                       k_cache, v_cache, g_attn, max_seq, attn_scale,
                       NK_MMA_ATTN_MIN_SEQ);
#endif
      nk_attention_mma(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                       k_cache, v_cache, g_attn, max_seq, attn_scale, s_arena);
    } else {
      nk_attention(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                   k_cache, v_cache, g_attn, max_seq, attn_scale, 0);
#if NK_FUSED_GQA_ATTN
      nk_attention_gqa(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                       k_cache, v_cache, g_attn, max_seq, attn_scale, 0);
#endif
    }
#else
    nk_attention(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                 k_cache, v_cache, g_attn, max_seq, attn_scale, 0);
#if NK_FUSED_GQA_ATTN
    nk_attention_gqa(seq_len, seq_start, seq_end, g_q, cos_table, sin_table,
                     k_cache, v_cache, g_attn, max_seq, attn_scale, 0);
#endif
#endif
#endif
    grid.sync();

    // --- Phase D1: O projection + residual ---
#if !defined(NK_SKIP_OPROJ)
#if NK_FP4MMA
#if NK_FP4_T2_OPROJ
    if (seq_len >= 64)
      nk_oproj_mma4_t2(seq_len, g_attn, s_arena, w, g_residual);
    else
      nk_oproj_mma4(seq_len, n_tiles, g_attn, s_arena, w, g_residual);
#else
    nk_oproj_mma4(seq_len, n_tiles, g_attn, s_arena, w, g_residual);
#endif
#else
    {
      // arena: [ s_attn f16 16x(1536+8) | s_part fp32 16x256 ]
      __half *s_attn = reinterpret_cast<__half *>(s_arena);
      float *s_part = reinterpret_cast<float *>(
          s_arena + (size_t)TOKEN_TILE * (Q_SIZE / 2 + NK_XPAD) * 2);
      nk_oproj_mma(seq_len, n_tiles, g_attn, s_attn, s_part, w, g_residual);
    }
#endif
#endif
    grid.sync();

    // --- Phase D2: post-attention norm + gate/up/SiLU ---
#if !defined(NK_SKIP_GATEUP)
#if NK_FP4MMA
#if NK_DUAL_CTA
    nk_residual_rstd_pass(seq_len, g_residual, g_rstd);
    grid.sync();
    nk_gateup_mma4_compact(seq_len, n_tiles, g_residual, g_rstd, s_arena,
                           w, g_mlp);
#elif NK_FP4_T2_GATEUP
    if (seq_len >= 32)
      nk_gateup_mma4_t2(seq_len, g_residual, s_arena, w, g_mlp);
    else
      nk_gateup_mma4(seq_len, n_tiles, g_residual, s_arena, w, g_mlp);
#else
    nk_gateup_mma4(seq_len, n_tiles, g_residual, s_arena, w, g_mlp);
#endif
#else
    {
      // arena: [ s_x f16 16x(2048+8) | s_part fp32 16x2x256 ]
      __half *s_x = reinterpret_cast<__half *>(s_arena);
      float *s_part = reinterpret_cast<float *>(
          s_arena + (size_t)TOKEN_TILE * (HIDDEN_SIZE + NK_XPAD) * 2);
      nk_gateup_mma(seq_len, n_tiles, g_residual, w.post_ln, s_x, s_part, w,
                    g_mlp);
    }
#endif
#endif
    grid.sync();

    // --- Phase D3: down projection + residual -> next layer hidden ---
#if !defined(NK_SKIP_DOWN)
#if NK_FP4MMA
#if NK_DUAL_CTA
#if NK_DUAL_DOWN_ROWPAIR
    nk_down_mma4_compact_rowpair(seq_len, g_mlp, s_arena, w, g_residual,
                                  g_down_partial, hidden);
#else
    nk_down_mma4_compact(seq_len, g_mlp, s_arena, w, g_residual,
                          g_down_partial, hidden);
#endif
#else
    nk_down_mma4(seq_len, n_tiles, g_mlp, s_arena, w, g_residual, hidden);
#endif
#else
    {
      // arena: [ s_mlp f16 16x(1536+8) | s_part fp32 16x256 ]
      __half *s_mlp = reinterpret_cast<__half *>(s_arena);
      float *s_part = reinterpret_cast<float *>(
          s_arena + (size_t)TOKEN_TILE * (INTERMEDIATE_SIZE / 4 + NK_XPAD) * 2);
      nk_down_mma(seq_len, n_tiles, g_mlp, s_mlp, s_part, w, g_residual,
                  hidden);
    }
#endif
#endif
    grid.sync();
  }

  // --- Finalize: per-token final-norm rstd, then mean pool + L2 ---
  nk_rstd_pass(seq_len, hidden, g_rstd);
  grid.sync();
  for (int b = block_id; b < batch; b += NK_COMPUTE_BLOCKS) {
    int first = seq_first ? __ldg(seq_first + b) : 0;
    int count = seq_count ? __ldg(seq_count + b) : seq_len;
    nk_pool_finalize(first, count, hidden, g_rstd, final_norm_weight,
                     g_out + (size_t)b * HIDDEN_SIZE);
  }
}

// =============================================================================
// C API (ctypes-friendly; no torch dependency)
// =============================================================================

#define NK_EXPORT extern "C" __declspec(dllexport)

static char g_err[512] = "";

static void set_err(const char *where, cudaError_t e) {
  snprintf(g_err, sizeof(g_err), "%s: %s", where, cudaGetErrorString(e));
}

#define NK_CHECK(expr)                                                         \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      set_err(#expr, _e);                                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

namespace {
struct NKState {
  const __nv_bfloat16 *embed_weight = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  const __half *cos_table = nullptr;
  const __half *sin_table = nullptr;
  NKLayerWeights *d_layer_weights = nullptr;
  int num_layers = 0;
  int max_seq = 0;

  __half *k_cache = nullptr;
  __half *v_cache = nullptr;
  __half *g_q = nullptr;
  __half *g_k = nullptr;
  __half *g_attn = nullptr;
  float *g_residual = nullptr;
  __half *g_mlp = nullptr;
  __half *hidden = nullptr;
  float *g_rstd = nullptr;
  float *g_out = nullptr;
  int *d_ids = nullptr;
  unsigned int *d_sync = nullptr;
  int *h_ids_pinned = nullptr;
  float *h_out_pinned = nullptr;
  bool ready = false;
};
NKState S;

void nk_release_allocations(NKState &state) {
  if (state.d_layer_weights)
    cudaFree(state.d_layer_weights);
  if (state.k_cache)
    cudaFree(state.k_cache);
  if (state.v_cache)
    cudaFree(state.v_cache);
  if (state.g_q)
    cudaFree(state.g_q);
  if (state.g_k)
    cudaFree(state.g_k);
  if (state.g_attn)
    cudaFree(state.g_attn);
  if (state.g_residual)
    cudaFree(state.g_residual);
  if (state.g_mlp)
    cudaFree(state.g_mlp);
  if (state.hidden)
    cudaFree(state.hidden);
  if (state.g_rstd)
    cudaFree(state.g_rstd);
  if (state.g_out)
    cudaFree(state.g_out);
  if (state.d_ids)
    cudaFree(state.d_ids);
  if (state.d_sync)
    cudaFree(state.d_sync);
  if (state.h_ids_pinned)
    cudaFreeHost(state.h_ids_pinned);
  if (state.h_out_pinned)
    cudaFreeHost(state.h_out_pinned);
  state = NKState{};
}
} // namespace

NK_EXPORT const char *nk_last_error() { return g_err; }

NK_EXPORT int nk_token_tile() { return TOKEN_TILE; }

// layer_ptrs: num_layers * 16 device pointers per layer in order:
//   input_ln, post_ln, q_w, q_s, k_w, k_s, v_w, v_s, o_w, o_s,
//   gate_w, gate_s, up_w, up_s, down_w, down_s
// layer_scales: num_layers * 14 floats per layer:
//   ws2 (weight_scale_2) for q, k, v, o, gate, up, down, then
//   input_scale for q, k, v, o, gate, up, down.
NK_EXPORT int nk_init(const void *embed_weight, const void *final_norm_weight,
                      const void *cos_table, const void *sin_table,
                      const void **layer_ptrs, const float *layer_scales,
                      int num_layers, int max_seq) {
  if (S.ready) {
    snprintf(g_err, sizeof(g_err), "nk_init already called for this DLL");
    return -1;
  }
  if (!embed_weight || !final_norm_weight || !cos_table || !sin_table ||
      !layer_ptrs || !layer_scales) {
    snprintf(g_err, sizeof(g_err), "nk_init received a null required pointer");
    return -1;
  }
  if (num_layers <= 0 || max_seq <= 0) {
    snprintf(g_err, sizeof(g_err),
             "nk_init requires num_layers > 0 and max_seq > 0 (got %d, %d)",
             num_layers, max_seq);
    return -1;
  }
  for (int l = 0; l < num_layers; l++) {
    const void **p = layer_ptrs + (size_t)l * 16;
    const float *sc = layer_scales + (size_t)l * 14;
    for (int i = 0; i < 16; i++) {
      if (!p[i]) {
        snprintf(g_err, sizeof(g_err),
                 "nk_init layer %d pointer %d is null", l, i);
        return -1;
      }
    }
    for (int i = 0; i < 14; i++) {
      if (!(sc[i] > 0.0f) || !std::isfinite(sc[i])) {
        snprintf(g_err, sizeof(g_err),
                 "nk_init layer %d scale %d must be positive and finite", l,
                 i);
        return -1;
      }
    }
  }

  NKState next{};
  next.embed_weight = (const __nv_bfloat16 *)embed_weight;
  next.final_norm_weight = (const __nv_bfloat16 *)final_norm_weight;
  next.cos_table = (const __half *)cos_table;
  next.sin_table = (const __half *)sin_table;
  next.num_layers = num_layers;
  next.max_seq = max_seq;

  NKLayerWeights *h_lw = new (std::nothrow) NKLayerWeights[num_layers];
  if (!h_lw) {
    snprintf(g_err, sizeof(g_err),
             "nk_init could not allocate the host layer table");
    return -1;
  }
#define NK_INIT_CHECK(expr)                                                    \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      delete[] h_lw;                                                           \
      nk_release_allocations(next);                                            \
      set_err(#expr, _e);                                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

  for (int l = 0; l < num_layers; l++) {
    const void **p = layer_ptrs + (size_t)l * 16;
    const float *sc = layer_scales + (size_t)l * 14;
    const float *is = sc + 7;
    h_lw[l].input_ln = (const __nv_bfloat16 *)p[0];
    h_lw[l].post_ln = (const __nv_bfloat16 *)p[1];
    h_lw[l].q = {(const unsigned char *)p[2], (const unsigned char *)p[3]};
    h_lw[l].k = {(const unsigned char *)p[4], (const unsigned char *)p[5]};
    h_lw[l].v = {(const unsigned char *)p[6], (const unsigned char *)p[7]};
    h_lw[l].o = {(const unsigned char *)p[8], (const unsigned char *)p[9]};
    h_lw[l].gate = {(const unsigned char *)p[10], (const unsigned char *)p[11]};
    h_lw[l].up = {(const unsigned char *)p[12], (const unsigned char *)p[13]};
    h_lw[l].down = {(const unsigned char *)p[14], (const unsigned char *)p[15]};
    h_lw[l].q_ws2 = sc[0];
    h_lw[l].k_ws2 = sc[1];
    h_lw[l].v_ws2 = sc[2];
    h_lw[l].o_ws2 = sc[3];
    h_lw[l].gate_ws2 = sc[4];
    h_lw[l].up_ws2 = sc[5];
    h_lw[l].down_ws2 = sc[6];
    // W4A4 parameters: fused tiles (qkv, gate/up) share max input_scale.
    float aqkv = fmaxf(is[0], fmaxf(is[1], is[2]));
    float ao = is[3];
    float agu = fmaxf(is[4], is[5]);
    float adown = is[6];
    h_lw[l].q_eps = sc[0] * aqkv;
    h_lw[l].k_eps = sc[1] * aqkv;
    h_lw[l].v_eps = sc[2] * aqkv;
    h_lw[l].o_eps = sc[3] * ao;
    h_lw[l].gate_eps = sc[4] * agu;
    h_lw[l].up_eps = sc[5] * agu;
    h_lw[l].down_eps = sc[6] * adown;
    h_lw[l].aqkv = aqkv;
    h_lw[l].aqkv_i6 = 1.0f / (6.0f * aqkv);
    h_lw[l].ao = ao;
    h_lw[l].ao_i6 = 1.0f / (6.0f * ao);
    h_lw[l].agu = agu;
    h_lw[l].agu_i6 = 1.0f / (6.0f * agu);
    h_lw[l].adown = adown;
    h_lw[l].adown_i6 = 1.0f / (6.0f * adown);
  }
  NK_INIT_CHECK(
      cudaMalloc(&next.d_layer_weights, num_layers * sizeof(NKLayerWeights)));
  NK_INIT_CHECK(cudaMemcpy(next.d_layer_weights, h_lw,
                           num_layers * sizeof(NKLayerWeights),
                           cudaMemcpyHostToDevice));
  delete[] h_lw;
  h_lw = nullptr;

  size_t ms = (size_t)max_seq;
  NK_INIT_CHECK(
      cudaMalloc(&next.k_cache, (size_t)NUM_KV_HEADS * ms * HEAD_DIM * 2));
  NK_INIT_CHECK(
      cudaMalloc(&next.v_cache, (size_t)NUM_KV_HEADS * ms * HEAD_DIM * 2));
  NK_INIT_CHECK(cudaMalloc(&next.g_q, ms * Q_SIZE * 2));
  NK_INIT_CHECK(cudaMalloc(&next.g_k, ms * KV_SIZE * 2));
  NK_INIT_CHECK(cudaMalloc(&next.g_attn, ms * Q_SIZE * 2));
  NK_INIT_CHECK(
      cudaMalloc(&next.g_residual, ms * HIDDEN_SIZE * sizeof(float)));
  NK_INIT_CHECK(cudaMalloc(&next.g_mlp, ms * INTERMEDIATE_SIZE * 2));
  NK_INIT_CHECK(cudaMalloc(&next.hidden, ms * HIDDEN_SIZE * 2));
#if NK_DUAL_CTA
  NK_INIT_CHECK(cudaMalloc(&next.g_rstd,
                           ms * (HIDDEN_SIZE + 1) * sizeof(float)));
#else
  NK_INIT_CHECK(cudaMalloc(&next.g_rstd, ms * sizeof(float)));
#endif
  // Single-row staging belongs only to nk_embed_host().  The batched device
  // API receives a caller-owned batch * HIDDEN_SIZE output allocation and
  // nk_pool_finalize offsets within that buffer at launch time.
  NK_INIT_CHECK(cudaMalloc(&next.g_out, HIDDEN_SIZE * sizeof(float)));
  NK_INIT_CHECK(cudaMalloc(&next.d_ids, ms * sizeof(int)));
  NK_INIT_CHECK(cudaMalloc(&next.d_sync, 3 * sizeof(unsigned int)));
  NK_INIT_CHECK(cudaMemset(next.d_sync, 0, 3 * sizeof(unsigned int)));
  NK_INIT_CHECK(cudaHostAlloc(&next.h_ids_pinned, ms * sizeof(int),
                              cudaHostAllocDefault));
  NK_INIT_CHECK(cudaHostAlloc(&next.h_out_pinned,
                              HIDDEN_SIZE * sizeof(float),
                              cudaHostAllocDefault));

  NK_INIT_CHECK(cudaFuncSetAttribute(
      nk_embed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      NK_ARENA_BYTES));
  NK_INIT_CHECK(cudaFuncSetAttribute(
      nk_embed_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  int device = 0;
  int active_blocks = 0;
  cudaDeviceProp props{};
  NK_INIT_CHECK(cudaGetDevice(&device));
  NK_INIT_CHECK(cudaGetDeviceProperties(&props, device));
#if NK_COOPERATIVE_LAUNCH
  if (!props.cooperativeLaunch) {
    nk_release_allocations(next);
    snprintf(g_err, sizeof(g_err),
             "cooperative launch is not supported by this CUDA device");
    return -1;
  }
#endif
  NK_INIT_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, nk_embed_kernel, NK_BLOCK_SIZE, NK_ARENA_BYTES));
#undef NK_INIT_CHECK
  if ((size_t)active_blocks * props.multiProcessorCount < NK_NUM_BLOCKS) {
    nk_release_allocations(next);
    snprintf(g_err, sizeof(g_err),
             "persistent grid needs %d resident blocks, device permits %d "
             "(%d/SM x %d SMs)", NK_NUM_BLOCKS,
             active_blocks * props.multiProcessorCount, active_blocks,
             props.multiProcessorCount);
    return -1;
  }
  next.ready = true;
  S = next;
  g_err[0] = '\0';
  return 0;
}

// Batched entry: `total_len` token ids packed back-to-back on device.
// d_seq_start[t] / d_seq_end[t]: packed first / one-past-last index of token
// t's sequence; d_seq_first[b] / d_seq_count[b]: sequence b's span.
// d_out = [batch][2048] fp32 on device.
// `max_item_len` is the host-known maximum d_seq_count value. It lets short
// packed batches bypass per-task MMA eligibility checks without a device sync.
// `mma_attention_from_layer` is uniform for the launch so cutoff A/B variants
// share identical compiled device code.
static int nk_embed_batch_dev_impl(
    const int *d_token_ids, const int *d_seq_start, const int *d_seq_end,
    const int *d_seq_first, const int *d_seq_count, int total_len, int batch,
    int max_item_len, int mma_attention_from_layer, float *d_out,
    void *stream_) {
  if (!S.ready) {
    snprintf(g_err, sizeof(g_err), "nk_init not called");
    return -1;
  }
  if (!d_token_ids || !d_out) {
    snprintf(g_err, sizeof(g_err),
             "nk_embed_batch_dev requires non-null token and output buffers");
    return -1;
  }
  if (total_len < 1 || total_len > S.max_seq) {
    snprintf(g_err, sizeof(g_err), "total_len %d out of range [1, %d]",
             total_len, S.max_seq);
    return -1;
  }
  if (batch < 1 || batch > total_len) {
    snprintf(g_err, sizeof(g_err), "batch %d out of range [1, %d]", batch,
             total_len);
    return -1;
  }
  if (max_item_len < 1 || max_item_len > total_len) {
    snprintf(g_err, sizeof(g_err),
             "max_item_len %d out of range [1, %d]", max_item_len,
             total_len);
    return -1;
  }
  if (mma_attention_from_layer < 0 || mma_attention_from_layer >= 16) {
    snprintf(g_err, sizeof(g_err),
             "mma_attention_from_layer %d out of range [0, 15]",
             mma_attention_from_layer);
    return -1;
  }
  bool any_meta = d_seq_start || d_seq_end || d_seq_first || d_seq_count;
  bool all_meta = d_seq_start && d_seq_end && d_seq_first && d_seq_count;
  if (any_meta != all_meta || (!all_meta && batch != 1)) {
    snprintf(g_err, sizeof(g_err),
             "packed launches require all four metadata buffers; unpacked "
             "launches require batch == 1");
    return -1;
  }
  cudaStream_t stream = (cudaStream_t)stream_;
  float attn_scale = 0.08838834764831845f; // 1/sqrt(128)
  NK_CHECK(cudaMemsetAsync(S.d_sync, 0, 3 * sizeof(unsigned int), stream));
#if NK_COOPERATIVE_LAUNCH
  unsigned int *barrier_counter = S.d_sync;
  unsigned int *barrier_sense = S.d_sync + 1;
  unsigned int *stream_ready = S.d_sync + 2;
  void *kernel_args[] = {
      &S.embed_weight,      &S.d_layer_weights, &S.final_norm_weight,
      &S.cos_table,         &S.sin_table,        &d_token_ids,
      &total_len,           &d_seq_start,        &d_seq_end,
      &d_seq_first,         &d_seq_count,        &batch,
      &max_item_len,        &mma_attention_from_layer,
      &S.k_cache,           &S.v_cache,
      &S.g_q,               &S.g_k,               &S.g_attn,
      &S.g_residual,        &S.g_mlp,             &S.hidden,
      &S.g_rstd,            &d_out,               &barrier_counter,
      &barrier_sense,       &stream_ready,        &S.num_layers,
      &S.max_seq,
      &attn_scale,
  };
  NK_CHECK(cudaLaunchCooperativeKernel(
      (const void *)nk_embed_kernel, dim3(NK_NUM_BLOCKS),
      dim3(NK_BLOCK_SIZE), kernel_args, NK_ARENA_BYTES, stream));
#else
  nk_embed_kernel<<<NK_NUM_BLOCKS, NK_BLOCK_SIZE, NK_ARENA_BYTES, stream>>>(
      S.embed_weight, S.d_layer_weights, S.final_norm_weight, S.cos_table,
      S.sin_table, d_token_ids, total_len, d_seq_start, d_seq_end, d_seq_first,
      d_seq_count, batch, max_item_len, mma_attention_from_layer, S.k_cache,
      S.v_cache, S.g_q, S.g_k, S.g_attn, S.g_residual, S.g_mlp, S.hidden,
      S.g_rstd, d_out, S.d_sync, S.d_sync + 1, S.d_sync + 2, S.num_layers,
      S.max_seq, attn_scale);
#endif
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    set_err("nk_embed_kernel launch", e);
    return -1;
  }
  g_err[0] = '\0';
  return 0;
}

// Compatibility entry conservatively treats the packed total as the maximum
// item length. New callers should use the extended entry below.
NK_EXPORT int nk_embed_batch_dev(const int *d_token_ids,
                                 const int *d_seq_start, const int *d_seq_end,
                                 const int *d_seq_first,
                                 const int *d_seq_count, int total_len,
                                 int batch, float *d_out, void *stream_) {
  return nk_embed_batch_dev_impl(d_token_ids, d_seq_start, d_seq_end,
                                 d_seq_first, d_seq_count, total_len, batch,
                                 total_len, NK_MMA_ATTN_FROM_LAYER, d_out, stream_);
}

NK_EXPORT int nk_embed_batch_dev_ex(
    const int *d_token_ids, const int *d_seq_start, const int *d_seq_end,
    const int *d_seq_first, const int *d_seq_count, int total_len, int batch,
    int max_item_len, int mma_attention_from_layer, float *d_out,
    void *stream_) {
  return nk_embed_batch_dev_impl(d_token_ids, d_seq_start, d_seq_end,
                                 d_seq_first, d_seq_count, total_len, batch,
                                 max_item_len, mma_attention_from_layer,
                                 d_out, stream_);
}

NK_EXPORT int nk_embed_dev(const int *d_token_ids, int seq_len, float *d_out,
                           void *stream_) {
  return nk_embed_batch_dev(d_token_ids, nullptr, nullptr, nullptr, nullptr,
                            seq_len, 1, d_out, stream_);
}

// Host-side convenience entry: copies ids up, embeds, copies the 2048-float
// embedding back, synchronizes.
NK_EXPORT int nk_embed_host(const int *h_token_ids, int seq_len, float *h_out) {
  if (!S.ready) {
    snprintf(g_err, sizeof(g_err), "nk_init not called");
    return -1;
  }
  if (!h_token_ids || !h_out) {
    snprintf(g_err, sizeof(g_err),
             "nk_embed_host requires non-null token and output buffers");
    return -1;
  }
  if (seq_len < 1 || seq_len > S.max_seq) {
    snprintf(g_err, sizeof(g_err), "seq_len %d out of range [1, %d]", seq_len,
             S.max_seq);
    return -1;
  }
  for (int i = 0; i < seq_len; i++)
    S.h_ids_pinned[i] = h_token_ids[i];
  NK_CHECK(cudaMemcpyAsync(S.d_ids, S.h_ids_pinned, seq_len * sizeof(int),
                           cudaMemcpyHostToDevice, 0));
  if (nk_embed_dev(S.d_ids, seq_len, S.g_out, nullptr) != 0)
    return -1;
  NK_CHECK(cudaMemcpyAsync(S.h_out_pinned, S.g_out, HIDDEN_SIZE * sizeof(float),
                           cudaMemcpyDeviceToHost, 0));
  NK_CHECK(cudaStreamSynchronize(0));
  for (int i = 0; i < HIDDEN_SIZE; i++)
    h_out[i] = S.h_out_pinned[i];
  return 0;
}
