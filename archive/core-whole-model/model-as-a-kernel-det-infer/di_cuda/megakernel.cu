// det-infer: an entire llama-family decode step as one kernel launch.
// Phase interpreter: a program (int64 [n_phases, 16]) lists the forward
// pass; all blocks run each phase cooperatively, then cross a software
// grid barrier. A per-phase launch mode runs the identical device code
// one phase per kernel (same grid, same arithmetic: bitwise identical).
// Arithmetic mirrors transformers eager bf16 semantics: bf16 storage,
// fp32 accumulation, one rounding per op boundary, fp32 softmax with the
// probability cast to bf16 before PV, scores bf16-rounded around the
// 1/sqrt(D) scale. Fixed reduction trees: outputs bitwise run to run.
// Torch-free translation unit; interface in mak_kernels.h.

#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <math.h>

#include "det_math.cuh"
#include "mak_kernels.h"

#define BLOCK_THREADS 256
#define MAX_HEAD_DIM 512
// GEMV weight tiles stream through a per-warp cp.async shared ring; the
// stage count is a per-launch power of two from the shared-memory budget,
// pure scheduling. One stage = 8 warps x 32 lanes x 16B = 4KB per block.
#define GEMV_RING_STAGE_BYTES ((BLOCK_THREADS / 32) * 32 * 16)

namespace mak {

using bf16 = __nv_bfloat16;

enum PhaseOp : int {
  OP_EMBED = 0,
  OP_GEMV = 1,
  OP_QKV_POST = 2,
  OP_ATTN = 3,
  OP_ARGMAX_PART = 4,
  OP_ARGMAX_FIN = 5,
  OP_KV_APPEND = 6,  // prefill chunks only: rope+append K/V for all m
  OP_NORMRES = 7,    // gemma post-sublayer norm + residual add
  OP_PLEMIX = 8,     // gemma per-layer-input mix (norm ctx, add tok, scale)
  // batched-decode transform phases: write the transformed [B][K] input to
  // a global scratch so the following OP_GEMV_PLAIN needs no shared staging,
  // decoupling the batch size from the projection width.
  OP_GEMV_PLAIN = 9, // GEMV over a global [B][K] input, no fused transform
  OP_NORMB = 10,     // RMSNorm (llama or gemma) [B][K] -> scratch
  OP_GLUB = 11,      // SwiGLU / gelu-glu gating [B][2K] -> scratch [B][K]
  OP_ATTNFINB = 12,  // finalize split-S attention [B] -> scratch [B][qdim]
  OP_GEMV_Q4 = 13,   // GEMV with nf4-packed weights, dequantized in-kernel
};

// NF4 codebook (bitsandbytes). Dequantized weight = code[nibble] *
// per-block absmax, rounded to bf16; exact lookup, IEEE multiply,
// reproducible on every card.
__device__ __constant__ float NF4_CODE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f,
    -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
    -0.09105003625154495f, 0.0f, 0.07958029955625534f, 0.16093020141124725f,
    0.24611230194568634f, 0.33791524171829224f, 0.44070982933044434f,
    0.5626170039176941f, 0.7229568362236023f, 1.0f};

enum InputTransform : int {
  IT_NONE = 0,
  IT_RMSNORM = 1,
  IT_SWIGLU = 2,
  IT_ATTNFIN = 3,   // finalize split-S attention partials on the fly
  IT_RMSNORM_G = 4, // gemma norm: fp32 throughout, raw weight, one rounding
  IT_GELU_GLU = 5,  // gelu_tanh(gate) * up
};
// slot-9 flag bits above the base transform
#define ITF_BASE_MASK 7
#define ITF_REDUCE 8      // input is fp32 k-slice partials: sum 4 slices
#define ITF_WRITEBACK 16  // write the reduced bf16 vector to p6

enum Epilogue : int {
  EP_STORE = 0,
  EP_RESID = 1,
  EP_F32 = 2,
  EP_F32_AMAX = 3,   // fp32 store + per-block argmax partial (lm head)
  EP_PARTIAL = 4,    // fp32 k-slice partials [NSLICE][N], no rounding
  EP_GELU_PLE = 5,   // gemma: gelu_tanh(out) * per-layer-input element
  EP_F32_AMAX_CAP = 6,  // lm head with final logit softcapping
};

#define ATTN_CHUNK 128
#define NSLICE 4

// Phase slots (int64 each):
//  0 op | 1 in | 2 w | 3 out | 4 a | 5 b | 6 c
//  7 N/max_seq | 8 K | 9 itrans/flags | 10 epi | 11 Hq | 12 Hkv | 13 D
// 14 f0 (fp32 bits in low 32) | 15 i0 / f1 (fp32 bits)

__device__ __forceinline__ float bits2f(long long v) {
  return __int_as_float((int)v);
}
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16(v); }

// torch gelu(approximate="tanh") in fp32
__device__ __forceinline__ float gelu_tanh_f(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  const float t = detm::tanhf_det(c * (x + 0.044715f * x * x * x));
  return 0.5f * x * (1.f + t);
}

__device__ __forceinline__ unsigned dyn_smem_size() {
  unsigned r;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(r));
  return r;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long v) {
  for (int o = 16; o > 0; o >>= 1) {
    unsigned long long w = __shfl_down_sync(0xffffffffu, v, o);
    v = (w > v) ? w : v;
  }
  return v;
}

// Valid on thread 0 after return; safe to call repeatedly (trailing sync).
__device__ float block_sum(float v, float* s_red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) s_red[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  v = (threadIdx.x < nw) ? s_red[threadIdx.x] : 0.f;
  if (w == 0) v = warp_sum(v);
  __syncthreads();
  return v;
}
__device__ float block_max(float v, float* s_red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) s_red[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  v = (threadIdx.x < nw) ? s_red[threadIdx.x] : -INFINITY;
  if (w == 0) v = warp_max(v);
  __syncthreads();
  return v;
}
__device__ unsigned long long block_max_u64(unsigned long long v,
                                            unsigned long long* s_red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_max_u64(v);
  if (lane == 0) s_red[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  v = (threadIdx.x < nw) ? s_red[threadIdx.x] : 0ull;
  if (w == 0) v = warp_max_u64(v);
  __syncthreads();
  return v;
}

// ---------------------------------------------------------------------------
// Phase bodies. Each is grid-cooperative: work is striped over all blocks,
// and any block with nothing to do simply falls through to the barrier.
// ---------------------------------------------------------------------------

__device__ unsigned long long block_max_u64(unsigned long long v,
                                            unsigned long long* s_red);
__device__ __forceinline__ unsigned long long amax_key(float v, unsigned idx);

// Prompt positions read the staged prompt buffer; beyond it each block
// reduces the previous step's argmax partials itself (closed loop) and
// records the token. prompt_len == -1 selects the external token buffer.
// pos_b != nullptr (batch mode) gives every row its own sequence position.
__device__ __forceinline__ int pm_of(int pos, int m, const int* pos_b,
                                     int step) {
  return (pos_b != nullptr) ? (pos_b[m] + step) : (pos + m);
}

__device__ void ph_embed(const long long* P, int pos, int M, int prompt_len,
                         int step_slot, const int* pos_b) {
  const bf16* tab = reinterpret_cast<const bf16*>(P[2]);
  bf16* out = reinterpret_cast<bf16*>(P[3]);
  const unsigned long long* parts =
      reinterpret_cast<const unsigned long long*>(P[4]);
  const int* prompt = reinterpret_cast<const int*>(P[5]);
  const int* tok = reinterpret_cast<const int*>(P[6]);
  int* toks_out = reinterpret_cast<int*>(P[7]);
  const long long K = P[8];

  __shared__ int s_tok;
  __shared__ int s_tokb[8];
  const bool need_feedback = (pos + M - 1 >= prompt_len);
  if (pos_b != nullptr) {
    // batch decode: every row reads its own sequence's fed-back token
    if (threadIdx.x < M) s_tokb[threadIdx.x] = tok[threadIdx.x];
    __syncthreads();
  } else if (need_feedback) {
    if (prompt_len >= 0 && parts != nullptr) {
      unsigned long long best = amax_key(-INFINITY, 0xFFFFFFFFu);
      for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) {
        const unsigned long long kk = parts[i];
        best = (kk > best) ? kk : best;
      }
      __shared__ unsigned long long s64[32];
      best = block_max_u64(best, s64);
      if (threadIdx.x == 0) {
        const int idx = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFull));
        s_tok = idx;
        const int prev_slot = step_slot - M;
        if (toks_out != nullptr && prev_slot >= 0 && blockIdx.x == 0)
          toks_out[prev_slot] = idx;
      }
    } else {
      if (threadIdx.x == 0) s_tok = tok[0];
    }
    __syncthreads();
  }

  const float hsc = bits2f(P[14]);  // gemma embed scale (0 bits = off)
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < (long long)M * K; i += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(i / K);
    const long long k = i % K;
    const int pm = pos + m;
    const long long t =
        (pos_b != nullptr)
            ? (long long)s_tokb[m]
            : ((pm < prompt_len && prompt != nullptr)
                   ? (long long)prompt[pm]
                   : (long long)s_tok);
    out[i] = (hsc != 0.f) ? f2bf(bf2f(tab[t * K + k]) * hsc)
                          : tab[t * K + k];
  }

  // gemma: gather this token's per-layer-input row, scaled (decode only;
  // the gemma builder pins chunk_m to 1)
  const bf16* ptab = reinterpret_cast<const bf16*>(P[1]);
  bf16* pout = reinterpret_cast<bf16*>(P[9]);
  if (ptab != nullptr && pout != nullptr && M == 1) {
    const long long PL = P[12];
    const float esc = bits2f(P[15]);
    const long long t = (pos < prompt_len && prompt != nullptr)
                            ? (long long)prompt[pos]
                            : (long long)s_tok;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < PL; i += (long long)gridDim.x * blockDim.x)
      pout[i] = f2bf(bf2f(ptab[t * PL + i]) * esc);
  }
}

__device__ __forceinline__ unsigned long long amax_key(float v, unsigned idx) {
  unsigned u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) |
         (unsigned long long)(0xFFFFFFFFu - idx);
}

// Finalize one element of split-S attention: combine per-chunk partials
// (max, sum, weighted V) into the bf16 attention output, matching eager's
// bf16 rounding of the attention output before the O projection.
__device__ __forceinline__ float attn_finalize(const float* mpart, int h,
                                               int d, int D, int maxch,
                                               int nch, int nch_lo) {
  const float* base = mpart + ((long long)h * maxch) * (D + 2);
  float M = -INFINITY;
  for (int c = nch_lo; c < nch; ++c)
    M = fmaxf(M, base[(long long)c * (D + 2) + D]);
  float L = 0.f, acc = 0.f;
  for (int c = nch_lo; c < nch; ++c) {
    const float* pc = base + (long long)c * (D + 2);
    const float w = detm::expf_det(pc[D] - M);
    L = fmaf(pc[D + 1], w, L);
    acc = fmaf(pc[d], w, acc);
  }
  return bf2f(f2bf(acc / L));
}

__device__ __forceinline__ float mk_load_x(const bf16* x, const bf16* gamma,
                                           long long k, long long K, int itrans,
                                           float inv, const float* part,
                                           int D, int maxch, int nch) {
  if (itrans == IT_RMSNORM)
    // transformers RMSNorm: (x_f32 * invrms) rounds to bf16 first, then the
    // bf16 weight multiply computes in fp32 and rounds again.
    return bf2f(f2bf(bf2f(gamma[k]) * bf2f(f2bf(bf2f(x[k]) * inv))));
  if (itrans == IT_SWIGLU) {
    // silu(gate) rounded to bf16, then bf16 multiply by up, rounded again.
    const float g = bf2f(x[k]);
    const float sig = 1.f / (1.f + detm::expf_det(-g));
    const float sg = bf2f(f2bf(g * sig));
    return bf2f(f2bf(sg * bf2f(x[K + k])));
  }
  if (itrans == IT_ATTNFIN)
    return attn_finalize(part, (int)(k / D), (int)(k % D), D, maxch, nch,
                         0);
  return bf2f(x[k]);
}

// One GEMV output element for the non-argmax epilogues.
__device__ __forceinline__ void gemv_store(const long long* P, int epi,
                                           long long N, const bf16* resid,
                                           int srcm, long long row, float a) {
  if (epi == EP_RESID)
    reinterpret_cast<bf16*>(P[3])[(long long)srcm * N + row] =
        f2bf(bf2f(resid[(long long)srcm * N + row]) + a);
  else if (epi == EP_F32)
    reinterpret_cast<float*>(P[3])[(long long)srcm * N + row] = a;
  else if (epi == EP_GELU_PLE) {
    // gemma per-layer-input gate: bf16 linear output, gelu in fp32
    // rounded back, multiplied by the layer's per-layer-input element
    const bf16* ple = reinterpret_cast<const bf16*>(P[6]);
    const float g = bf2f(f2bf(a));
    const float act = bf2f(f2bf(gelu_tanh_f(g)));
    reinterpret_cast<bf16*>(P[3])[(long long)srcm * N + row] =
        f2bf(act * bf2f(ple[row]));
  } else
    reinterpret_cast<bf16*>(P[3])[(long long)srcm * N + row] = f2bf(a);
}

// GEMV over an M-token chunk. Weights are read once per chunk: each warp's
// 16-byte weight tile feeds M accumulators against M staged input rows, so
// prefill weight traffic is amortized by the chunk size. mmode (slot 12)
// restricts the phase to the chunk's last row (LM head). Buffers are laid
// out [M][len]; the residual stream and outputs address row m.
template <int NB>
__device__ void ph_gemv(const long long* P, int pos, int M, int rs,
                        const int* pos_b, int step) {
  const bf16* W = reinterpret_cast<const bf16*>(P[2]);
  const bf16* gamma = reinterpret_cast<const bf16*>(P[4]);
  const bf16* resid = reinterpret_cast<const bf16*>(P[5]);
  const long long N = P[7], K = P[8];
  const int itrans = ((int)P[9]) & ITF_BASE_MASK;
  const bool tiled = ((P[9] >> 20) & 1) != 0;  // row-tile-of-8 weights
  const int epi = (int)P[10];
  const int mmode = ((int)P[12]) & 0xFFFF;
  const int chbits = (int)(P[12] >> 16) & 0xFFFF;
  const int CH = (chbits > 0 && chbits <= ATTN_CHUNK) ? chbits : ATTN_CHUNK;
  const float eps = bits2f(P[14]);
  const float* part = reinterpret_cast<const float*>(P[1]);
  const bf16* xin = reinterpret_cast<const bf16*>(P[1]);
  const int D = (int)P[13];
  const int maxch = (int)(P[15] & 0xFFFF);
  const int fin_win = (int)((P[15] >> 16) & 0xFFFF);  // 0 = unwindowed

  // mmode restricts to the chunk's last row (prompt LM head); in batch
  // mode every row is a live sequence needing logits.
  const int m0 = (mmode && pos_b == nullptr) ? (M - 1) : 0;
  const int Ma = M - m0;  // staged rows

  // dynamic shared: [cp.async weight ring (rs stages) | staged inputs xs]
  extern __shared__ unsigned char smem_raw[];
  int4* rng = reinterpret_cast<int4*>(smem_raw);
  bf16* xs = reinterpret_cast<bf16*>(
      smem_raw + (long long)rs * GEMV_RING_STAGE_BYTES);
  __shared__ float s_red[32];
  __shared__ float s_scale;
  __shared__ unsigned long long s64[32];
  __shared__ float s_fine[1024 + 128];  // attn-finalize combine weights

  const int lane = threadIdx.x & 31;
  const int wib = threadIdx.x >> 5;

  // Weight streaming starts before input staging: the first tile groups
  // fly while the staging transform computes.
  const long long gwarp =
      ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  const int rmask = rs - 1;  // rs is a power of two
  int4* ringw = rng + (wib * rs) * 32 + lane;
  const long long kbase = (long long)lane * 8;
  const long long ntiles = (kbase < K) ? (K - kbase + 255) / 256 : 0;
  long long irow = gwarp, itile = 0;  // next (row, tile) to issue
  long long ig = 0, cg = 0;           // groups issued / consumed
  // Tiled weights ([N/8][K/8][8][8]): eight warps hold eight consecutive
  // rows, one contiguous 4KB stream per block (HBM row-buffer locality);
  // the plain layout keeps per-warp sequential streams (GDDR). Same bytes
  // per lane in the same order either way; addresses only.
  const long long K8 = K >> 3;
  auto issue_one = [&]() {
    if (irow < N && ntiles > 0) {
      const bf16* src =
          tiled ? W + (((irow >> 3) * K8 + (kbase >> 3) + itile * 32) << 6) +
                      ((irow & 7) << 3)
                : W + irow * K + kbase + itile * 256;
      __pipeline_memcpy_async(ringw + ((int)ig & rmask) * 32, src, 16);
      __pipeline_commit();
      ++ig;
      if (++itile == ntiles) {
        itile = 0;
        irow += nwarps;
      }
    }
  };
  for (int i = 0; i < rs; ++i) issue_one();

  // Stage transformed inputs for all chunk rows: [Ma][K] bf16 in shared
  // (every transform value is bf16-rounded by semantics: staging exact).
  // Reads are 16-byte vectorized; RMSNORM lands the raw row in shared on
  // the first pass and transforms in place, so global x is read once.
  {
    for (int m = 0; m < Ma; ++m) {
      const int srcm = m0 + m;
      const long long G = K / 8;  // 16-byte groups (K % 8 == 0 by contract)
      if (itrans == IT_RMSNORM) {
        const bf16* xr = xin + (long long)srcm * K;
        float ss = 0.f;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          const int4 x4 = *reinterpret_cast<const int4*>(xr + g * 8);
          *reinterpret_cast<int4*>(xs + (long long)m * K + g * 8) = x4;
          const __nv_bfloat162* xp =
              reinterpret_cast<const __nv_bfloat162*>(&x4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float lo = __low2float(xp[j]);
            const float hi = __high2float(xp[j]);
            ss = __fadd_rn(ss, __fmul_rn(lo, lo));
            ss = __fadd_rn(ss, __fmul_rn(hi, hi));
          }
        }
        ss = block_sum(ss, s_red);
        if (threadIdx.x == 0) s_scale = detm::rsqrtf_det(ss / (float)K + eps);
        __syncthreads();
        const float inv = s_scale;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          bf16* xsp = xs + (long long)m * K + g * 8;
          int4 x4 = *reinterpret_cast<const int4*>(xsp);
          const int4 g4 = *reinterpret_cast<const int4*>(gamma + g * 8);
          __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(&x4);
          const __nv_bfloat162* gp =
              reinterpret_cast<const __nv_bfloat162*>(&g4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            xp[j].x = f2bf(__low2float(gp[j]) *
                           bf2f(f2bf(__low2float(xp[j]) * inv)));
            xp[j].y = f2bf(__high2float(gp[j]) *
                           bf2f(f2bf(__high2float(xp[j]) * inv)));
          }
          *reinterpret_cast<int4*>(xsp) = x4;
        }
        __syncthreads();
      } else if (itrans == IT_RMSNORM_G) {
        // gemma norm: fp32 throughout, raw weight, one rounding at the end
        const bf16* xr = xin + (long long)srcm * K;
        float ss = 0.f;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          const int4 x4 = *reinterpret_cast<const int4*>(xr + g * 8);
          *reinterpret_cast<int4*>(xs + (long long)m * K + g * 8) = x4;
          const __nv_bfloat162* xp =
              reinterpret_cast<const __nv_bfloat162*>(&x4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float lo = __low2float(xp[j]);
            const float hi = __high2float(xp[j]);
            ss = __fadd_rn(ss, __fmul_rn(lo, lo));
            ss = __fadd_rn(ss, __fmul_rn(hi, hi));
          }
        }
        ss = block_sum(ss, s_red);
        if (threadIdx.x == 0) s_scale = detm::rsqrtf_det(ss / (float)K + eps);
        __syncthreads();
        const float inv = s_scale;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          bf16* xsp = xs + (long long)m * K + g * 8;
          int4 x4 = *reinterpret_cast<const int4*>(xsp);
          const int4 g4 = *reinterpret_cast<const int4*>(gamma + g * 8);
          __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(&x4);
          const __nv_bfloat162* gp =
              reinterpret_cast<const __nv_bfloat162*>(&g4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            xp[j].x = f2bf((__low2float(xp[j]) * inv) * __low2float(gp[j]));
            xp[j].y =
                f2bf((__high2float(xp[j]) * inv) * __high2float(gp[j]));
          }
          *reinterpret_cast<int4*>(xsp) = x4;
        }
        __syncthreads();
      } else if (itrans == IT_GELU_GLU) {
        // gemma MLP gating: gelu_tanh(gate) rounded to bf16, times up
        const bf16* xr = xin + (long long)srcm * 2 * K;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          const int4 g4 = *reinterpret_cast<const int4*>(xr + g * 8);
          const int4 u4 = *reinterpret_cast<const int4*>(xr + K + g * 8);
          int4 o4;
          const __nv_bfloat162* gp =
              reinterpret_cast<const __nv_bfloat162*>(&g4);
          const __nv_bfloat162* up =
              reinterpret_cast<const __nv_bfloat162*>(&u4);
          __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            {
              const float gv = __low2float(gp[j]);
              const float act = bf2f(f2bf(gelu_tanh_f(gv)));
              op[j].x = f2bf(act * __low2float(up[j]));
            }
            {
              const float gv = __high2float(gp[j]);
              const float act = bf2f(f2bf(gelu_tanh_f(gv)));
              op[j].y = f2bf(act * __high2float(up[j]));
            }
          }
          *reinterpret_cast<int4*>(xs + (long long)m * K + g * 8) = o4;
        }
      } else if (itrans == IT_SWIGLU) {
        const bf16* xr = xin + (long long)srcm * 2 * K;
        for (long long g = threadIdx.x; g < G; g += blockDim.x) {
          const int4 g4 = *reinterpret_cast<const int4*>(xr + g * 8);
          const int4 u4 = *reinterpret_cast<const int4*>(xr + K + g * 8);
          int4 o4;
          const __nv_bfloat162* gp =
              reinterpret_cast<const __nv_bfloat162*>(&g4);
          const __nv_bfloat162* up =
              reinterpret_cast<const __nv_bfloat162*>(&u4);
          __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            {
              const float gv = __low2float(gp[j]);
              const float sig = 1.f / (1.f + detm::expf_det(-gv));
              const float sg = bf2f(f2bf(gv * sig));
              op[j].x = f2bf(sg * __low2float(up[j]));
            }
            {
              const float gv = __high2float(gp[j]);
              const float sig = 1.f / (1.f + detm::expf_det(-gv));
              const float sg = bf2f(f2bf(gv * sig));
              op[j].y = f2bf(sg * __high2float(up[j]));
            }
          }
          *reinterpret_cast<int4*>(xs + (long long)m * K + g * 8) = o4;
        }
      } else if (itrans == IT_ATTNFIN) {
        const int pm = pm_of(pos, srcm, pos_b, step);
        const int nch = (pm + CH) / CH;
        const int s_lo = (fin_win > 0 && pm + 1 > fin_win)
                             ? pm + 1 - fin_win : 0;
        const int nch_lo = s_lo / CH;
        const float* mpart =
            part + (long long)srcm * ((long long)P[11] * maxch * (D + 2));
        const int Hq_ = (int)P[11];
        if (Hq_ * nch <= 1024 && Hq_ <= 128) {
          // Per-(head, chunk) softmax-combine weights computed once per
          // block: same expf inputs and fma chains as attn_finalize, so
          // values are bit-identical; exp count drops K*nch -> Hq*nch.
          float* ew = s_fine;
          float* lw = s_fine + 1024;
          for (int h = threadIdx.x; h < Hq_; h += blockDim.x) {
            const float* base = mpart + ((long long)h * maxch) * (D + 2);
            float Mx = -INFINITY;
            for (int c = nch_lo; c < nch; ++c)
              Mx = fmaxf(Mx, base[(long long)c * (D + 2) + D]);
            float L = 0.f;
            for (int c = nch_lo; c < nch; ++c) {
              const float* pc = base + (long long)c * (D + 2);
              const float w = detm::expf_det(pc[D] - Mx);
              ew[h * nch + c] = w;
              L = fmaf(pc[D + 1], w, L);
            }
            lw[h] = L;
          }
          __syncthreads();
          if (D % 4 == 0) {
            // four elements per thread, 8-byte partial loads; combine
            // chains unchanged
            for (long long g = threadIdx.x; g < K / 4; g += blockDim.x) {
              const long long k = g * 4;
              const int h = (int)(k / D), d0 = (int)(k % D);
              const float* base = mpart + ((long long)h * maxch) * (D + 2);
              float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
              for (int c = nch_lo; c < nch; ++c) {
                const float* pc = base + (long long)c * (D + 2) + d0;
                const float2 pa = *reinterpret_cast<const float2*>(pc);
                const float2 pb = *reinterpret_cast<const float2*>(pc + 2);
                const float w = ew[h * nch + c];
                a0 = fmaf(pa.x, w, a0);
                a1 = fmaf(pa.y, w, a1);
                a2 = fmaf(pb.x, w, a2);
                a3 = fmaf(pb.y, w, a3);
              }
              const float L = lw[h];
              bf16* xp = xs + (long long)m * K + k;
              xp[0] = f2bf(bf2f(f2bf(a0 / L)));
              xp[1] = f2bf(bf2f(f2bf(a1 / L)));
              xp[2] = f2bf(bf2f(f2bf(a2 / L)));
              xp[3] = f2bf(bf2f(f2bf(a3 / L)));
            }
          } else {
            for (long long k = threadIdx.x; k < K; k += blockDim.x) {
              const int h = (int)(k / D), d = (int)(k % D);
              const float* base = mpart + ((long long)h * maxch) * (D + 2);
              float acc = 0.f;
              for (int c = nch_lo; c < nch; ++c)
                acc = fmaf(base[(long long)c * (D + 2) + d],
                           ew[h * nch + c], acc);
              xs[(long long)m * K + k] = f2bf(bf2f(f2bf(acc / lw[h])));
            }
          }
          __syncthreads();
        } else {
          for (long long k = threadIdx.x; k < K; k += blockDim.x)
            xs[(long long)m * K + k] = f2bf(attn_finalize(
                mpart, (int)(k / D), (int)(k % D), D, maxch, nch, nch_lo));
        }
      } else {
        const bf16* xr = xin + (long long)srcm * K;
        for (long long g = threadIdx.x; g < G; g += blockDim.x)
          *reinterpret_cast<int4*>(xs + (long long)m * K + g * 8) =
              *reinterpret_cast<const int4*>(xr + g * 8);
      }
    }
    __syncthreads();
  }

  // Warp-per-row GEMV over the cp.async ring primed above: rs tile
  // groups in flight per lane with no register dependence, (row, tile)
  // stream continuous across rows. Each lane consumes its tiles in
  // order, so the fma chain and reduction tree are exactly those of the
  // register schedule: bit-identical results.
  unsigned long long best = amax_key(-INFINITY, 0xFFFFFFFFu);
  {
    for (long long row = gwarp; row < N; row += nwarps) {
      float acc[NB];
#pragma unroll
      for (int m = 0; m < NB; ++m) acc[m] = 0.f;
      auto consume = [&](const int4& w4, long long t) {
        const long long k = kbase + t * 256;
        const __nv_bfloat162* wp =
            reinterpret_cast<const __nv_bfloat162*>(&w4);
#pragma unroll
        for (int m = 0; m < NB; ++m) {
          if (m < Ma) {
            const __nv_bfloat162* xp =
                reinterpret_cast<const __nv_bfloat162*>(xs +
                                                        (long long)m * K + k);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              acc[m] = fmaf(__low2float(wp[j]), __low2float(xp[j]), acc[m]);
              acc[m] = fmaf(__high2float(wp[j]), __high2float(xp[j]),
                            acc[m]);
            }
          }
        }
      };
      for (long long t = 0; t < ntiles; ++t) {
        __pipeline_wait_prior((int)(ig - cg - 1));
        const int4 w4 = ringw[((int)cg & rmask) * 32];
        ++cg;
        issue_one();
        consume(w4, t);
      }
#pragma unroll
      for (int m = 0; m < NB; ++m) {
        if (m < Ma) {
          const float a = warp_sum(acc[m]);
          if (lane == 0) {
            const int srcm = m0 + m;
            if (epi == EP_F32_AMAX) {
              reinterpret_cast<float*>(P[3])[row] = a;
              const unsigned long long kk = amax_key(bf2f(f2bf(a)),
                                                     (unsigned)row);
              best = (kk > best) ? kk : best;
            } else if (epi == EP_F32_AMAX_CAP) {
              // final logit softcapping, mirroring the eager bf16 op chain
              const float cap = bits2f(P[15]);
              const bf16 t1 = f2bf(a);
              const bf16 t2 = f2bf(bf2f(t1) / cap);
              const bf16 t3 = f2bf(detm::tanhf_det(bf2f(t2)));
              const float v = bf2f(f2bf(bf2f(t3) * cap));
              reinterpret_cast<float*>(P[3])[row] = v;
              const unsigned long long kk = amax_key(v, (unsigned)row);
              best = (kk > best) ? kk : best;
            } else {
              gemv_store(P, epi, N, resid, srcm, row, a);
            }
          }
        }
      }
    }
  }
  if (epi == EP_F32_AMAX || epi == EP_F32_AMAX_CAP) {
    __syncthreads();
    best = block_max_u64(best, s64);
    if (threadIdx.x == 0)
      reinterpret_cast<unsigned long long*>(P[6])[blockIdx.x] = best;
  }
}

// Norm/rope K and copy V into the cache for one (kv-head, position);
// block-cooperative, semantics identical to the eager reference.
__device__ void kv_norm_rope_append(const bf16* qkv, bf16* kc, bf16* vc,
                                    const bf16* knw, int qk_flag, float eps,
                                    const float* invf, int Hq, int Hkv, int kvh,
                                    int D, int pm, float* s_red,
                                    float* s_inv, float* hbuf, float ascale) {
  const int half = D >> 1;
  const bf16* ksrc = qkv + (long long)(Hq + kvh) * D;
  const bf16* vsrc = qkv + (long long)(Hq + Hkv + kvh) * D;
  if (qk_flag & 1) {
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      const float x = bf2f(ksrc[d]);
      ss = __fadd_rn(ss, __fmul_rn(x, x));
    }
    ss = block_sum(ss, s_red);
    if (threadIdx.x == 0) *s_inv = detm::rsqrtf_det(ss / (float)D + eps);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x)
      hbuf[d] = (qk_flag & 2)
                    ? bf2f(f2bf((bf2f(ksrc[d]) * (*s_inv)) * bf2f(knw[d])))
                    : bf2f(f2bf(bf2f(knw[d]) *
                                bf2f(f2bf(bf2f(ksrc[d]) * (*s_inv)))));
  }
  // without k-norm the rope reads the source row directly (identical
  // values), skipping the staging pass and its barrier
  if (qk_flag & 1) __syncthreads();
  bf16* krow = kc + (long long)pm * D;
  bf16* vrow = vc + (long long)pm * D;
  for (int d = threadIdx.x; d < half; d += blockDim.x) {
    const float ang = (float)pm * invf[d];
    float sv, cv;
    detm::sincosf_det(ang, &sv, &cv);
    const float c = bf2f(f2bf(cv * ascale));
    const float s = bf2f(f2bf(sv * ascale));
    const float x1 = (qk_flag & 1) ? hbuf[d] : bf2f(ksrc[d]);
    const float x2 = (qk_flag & 1) ? hbuf[d + half] : bf2f(ksrc[d + half]);
    krow[d] = f2bf(bf2f(f2bf(x1 * c)) + bf2f(f2bf(-x2 * s)));
    krow[d + half] = f2bf(bf2f(f2bf(x2 * c)) + bf2f(f2bf(x1 * s)));
  }
  if (qk_flag & 4) {
    // gemma v-norm: weightless RMS over v, fp32, one rounding
    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      const float x = bf2f(vsrc[d]);
      ss = __fadd_rn(ss, __fmul_rn(x, x));
    }
    ss = block_sum(ss, s_red);
    if (threadIdx.x == 0) *s_inv = detm::rsqrtf_det(ss / (float)D + eps);
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x)
      vrow[d] = f2bf(bf2f(vsrc[d]) * (*s_inv));
  } else {
    for (int d = threadIdx.x; d < D; d += blockDim.x) vrow[d] = vsrc[d];
  }
  __threadfence_block();
  __syncthreads();
}

// Attention with rope fused in, positions split across chunks, M-token
// prefill loop. Slots: 1 qkv [M rows], 2 kcache, 3 vcache, 4 fp32
// partials [M][Hq][maxch][D+2], 5 q-norm w, 6 k-norm w, 7 max_seq,
// 8 qk flag, 9 eps bits, 10 inv_freq ptr, 11 Hq, 12 Hkv, 13 D, 14 scale,
// 15 maxch. Decode (M == 1): the chunk owning `pos` norms/ropes K and
// appends K/V (duplicate GQA writes are identical, benign); prefill:
// OP_KV_APPEND has already appended all M positions.
__device__ void ph_attn(const long long* P, int pos, int M,
                        const int* pos_b, int step, long long kv_bstride) {
  const long long max_seq = P[7];
  // slot 8: bit0 q/k norm, bit1 gemma norm semantics, bit2 v-norm;
  // bits 16+ sliding window length (0 = unwindowed)
  const int qk_flag = ((int)P[8]) & 0xFFFF;
  const int win = (int)((P[8] >> 16) & 0xFFFFFFFFll);
  const float eps = bits2f(P[9]);
  const float* invf = reinterpret_cast<const float*>(P[10]);
  const int Hq = (int)P[11], Hkv = ((int)P[12]) & 0xFFFF;
  const int D = (int)P[13];
  // chunk length rides in slot 12 bits 16+ (0 = compile-time default);
  // shared staging is sized for the maximum, so CH <= ATTN_CHUNK.
  const int chbits = (int)(P[12] >> 16) & 0xFFFF;
  const int CH = (chbits > 0 && chbits <= ATTN_CHUNK) ? chbits : ATTN_CHUNK;
  const float scale = bits2f(P[14]);
  const int maxch = (int)(P[15] & 0xFFFF);
  // rope attention scaling in slot 15 bits 32+ (0 bits = 1.0)
  const int asbits = (int)(P[15] >> 32);
  const float ascale = (asbits != 0) ? __int_as_float(asbits) : 1.0f;
  const int rep = Hq / Hkv;
  const long long KT = (long long)(Hq + 2 * Hkv) * D;

  const bf16* qnw = reinterpret_cast<const bf16*>(P[5]);
  const bf16* knw = reinterpret_cast<const bf16*>(P[6]);

  __shared__ float qs[MAX_HEAD_DIM];
  __shared__ float hbuf[MAX_HEAD_DIM];
  __shared__ float sc[ATTN_CHUNK];
  __shared__ float pv[BLOCK_THREADS];
  __shared__ float s_red[32];
  __shared__ float s_inv, s_m;

  // (m, head, chunk) flattened into one block-stride space so prefill
  // chunks use M times the parallelism; per-m chunk counts vary, so
  // out-of-range (m, ch) pairs skip uniformly.
  int pm_max = pos + M - 1;
  if (pos_b != nullptr) {
    int mx = pos_b[0];
    for (int m = 1; m < M; ++m) mx = (pos_b[m] > mx) ? pos_b[m] : mx;
    pm_max = mx + step;
  }
  const int nch_max = (pm_max + 1 + CH - 1) / CH;
  {
    const int total = M * Hq * nch_max;
    for (int v = blockIdx.x; v < total; v += gridDim.x) {
      const int m = v / (Hq * nch_max);
      const int rem = v % (Hq * nch_max);
      const int head = rem / nch_max;
      const int ch = rem % nch_max;
      const int pm = pm_of(pos, m, pos_b, step);
      const int S = pm + 1;
      const int nch = (S + CH - 1) / CH;
      if (ch >= nch) continue;
      const int s_lo = (win > 0 && S > win) ? S - win : 0;
      if (ch < s_lo / CH) continue;  // fully outside the sliding window
      const int own_ch = pm / CH;
      const bf16* qkv =
          reinterpret_cast<const bf16*>(P[1]) + (long long)m * KT;
      float* partm = reinterpret_cast<float*>(P[4]) +
                     (long long)m * ((long long)Hq * maxch * (D + 2));
      const int kvh = head / rep;
      const long long kvb = (pos_b != nullptr) ? (long long)m * kv_bstride : 0;
      bf16* kc = reinterpret_cast<bf16*>(P[2]) + kvb +
                 (long long)kvh * max_seq * D;
      bf16* vc = reinterpret_cast<bf16*>(P[3]) + kvb +
                 (long long)kvh * max_seq * D;
      float* part = partm + ((long long)head * maxch + ch) * (D + 2);
      const int half = D >> 1;

      // decode: the owning chunk appends K/V before the chunk copy below
      // sees it; shared-KV layers (flag bit 3) append nothing
      if ((M == 1 || pos_b != nullptr) && ch == own_ch && !(qk_flag & 8)) {
        kv_norm_rope_append(qkv, kc, vc, knw, qk_flag, eps, invf, Hq, Hkv,
                            kvh, D, pm, s_red, &s_inv, hbuf, ascale);
      }

      const int s0 = ch * CH;
      const int s1 = (s0 + CH < S) ? s0 + CH : S;
      const int cn = s1 - s0;
      const int si0 = (s0 < s_lo) ? (s_lo - s0) : 0;  // window mask

      // Copy this chunk's K/V into the dynamic-shared region when it
      // fits; cp.async overlaps the q staging below. Values unchanged:
      // both paths compute identical bits.
      extern __shared__ unsigned char smem_raw[];
      bf16* ks_sh = reinterpret_cast<bf16*>(smem_raw);
      bf16* vs_sh = ks_sh + (long long)CH * D;
      const bool kv_staged =
          (D % 8 == 0) &&
          (2u * (unsigned)CH * (unsigned)D * 2u <= dyn_smem_size());
      if (kv_staged) {
        const int ncpy = cn * D / 8;
        for (int i = threadIdx.x; i < ncpy; i += blockDim.x)
          __pipeline_memcpy_async(
              ks_sh + (long long)i * 8,
              kc + (long long)s0 * D + (long long)i * 8, 16);
        for (int i = threadIdx.x; i < ncpy; i += blockDim.x)
          __pipeline_memcpy_async(
              vs_sh + (long long)i * 8,
              vc + (long long)s0 * D + (long long)i * 8, 16);
        __pipeline_commit();
      }

      // stage rope'd q (bf16-rounded values) into shared
      {
        const bf16* q = qkv + (long long)head * D;
        if (qk_flag & 1) {
          float ss = 0.f;
          for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const float x = bf2f(q[d]);
            ss = __fadd_rn(ss, __fmul_rn(x, x));
          }
          ss = block_sum(ss, s_red);
          if (threadIdx.x == 0) s_inv = detm::rsqrtf_det(ss / (float)D + eps);
          __syncthreads();
          for (int d = threadIdx.x; d < D; d += blockDim.x)
            hbuf[d] = (qk_flag & 2)
                          ? bf2f(f2bf((bf2f(q[d]) * s_inv) * bf2f(qnw[d])))
                          : bf2f(f2bf(bf2f(qnw[d]) *
                                      bf2f(f2bf(bf2f(q[d]) * s_inv))));
        }
        // without q-norm the rope reads q directly (identical values),
        // skipping the staging pass and its barrier
        if (qk_flag & 1) __syncthreads();
        for (int d = threadIdx.x; d < half; d += blockDim.x) {
          const float ang = (float)pm * invf[d];
          float sv, cv;
          detm::sincosf_det(ang, &sv, &cv);
          const float c = bf2f(f2bf(cv * ascale));
          const float s = bf2f(f2bf(sv * ascale));
          const float x1 = (qk_flag & 1) ? hbuf[d] : bf2f(q[d]);
          const float x2 =
              (qk_flag & 1) ? hbuf[d + half] : bf2f(q[d + half]);
          qs[d] = bf2f(f2bf(bf2f(f2bf(x1 * c)) + bf2f(f2bf(-x2 * s))));
          qs[d + half] = bf2f(f2bf(bf2f(f2bf(x2 * c)) + bf2f(f2bf(x1 * s))));
        }
        if (kv_staged) __pipeline_wait_prior(0);
        __syncthreads();
      }

      // score this chunk: eager rounds QK^T to bf16, scales as a bf16 op,
      // upcasts to fp32 for the softmax
      const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
      const int nw = blockDim.x >> 5;
      for (int si = wid; si < cn; si += nw) {
        if (si < si0) continue;
        const bf16* kr = kv_staged ? ks_sh + (long long)si * D
                                   : kc + (long long)(s0 + si) * D;
        float acc = 0.f;
        for (int d = lane; d < D; d += 32) acc = fmaf(qs[d], bf2f(kr[d]), acc);
        acc = warp_sum(acc);
        if (lane == 0) {
          float x = bf2f(f2bf(acc));
          sc[si] = bf2f(f2bf(x * scale));
        }
      }
      __syncthreads();

      float mx = -INFINITY;
      for (int si = threadIdx.x; si < cn; si += blockDim.x)
        if (si >= si0) mx = fmaxf(mx, sc[si]);
      mx = block_max(mx, s_red);
      if (threadIdx.x == 0) s_m = mx;
      __syncthreads();

      float lsum = 0.f;
      for (int si = threadIdx.x; si < cn; si += blockDim.x) {
        if (si < si0) continue;
        const float p = detm::expf_det(sc[si] - s_m);
        sc[si] = p;
        lsum = __fadd_rn(lsum, p);
      }
      lsum = block_sum(lsum, s_red);
      __syncthreads();

      // PV: long chunks split positions across NG thread groups; fp32
      // partials combine in group order. The tree depends only on cn and
      // D: identical on every device and launch mode.
      const int NG = (BLOCK_THREADS / D < 8) ? BLOCK_THREADS / D : 8;
      if (NG > 1 && cn >= 64) {
        const int g = threadIdx.x / D;
        const int d = threadIdx.x - g * D;
        if (g < NG) {
          const int csz = (cn + NG - 1) / NG;
          int a0 = g * csz;
          if (a0 < si0) a0 = si0;
          const int a1 = ((g * csz) + csz < cn) ? (g * csz) + csz : cn;
          float acc = 0.f;
          for (int si = a0; si < a1; ++si)
            acc = fmaf(sc[si],
                       bf2f(kv_staged ? vs_sh[(long long)si * D + d]
                                      : vc[(long long)(s0 + si) * D + d]),
                       acc);
          pv[g * D + d] = acc;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
          float acc = 0.f;
          for (int g2 = 0; g2 < NG; ++g2) acc += pv[g2 * D + d];
          part[d] = acc;
        }
      } else {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
          float acc = 0.f;
          for (int si = si0; si < cn; ++si)
            acc = fmaf(sc[si],
                       bf2f(kv_staged ? vs_sh[(long long)si * D + d]
                                      : vc[(long long)(s0 + si) * D + d]),
                       acc);
          part[d] = acc;
        }
      }
      if (threadIdx.x == 0) {
        part[D] = s_m;
        part[D + 1] = lsum;
      }
      __syncthreads();
    }
  }
}

// Prefill chunks: norm/rope K and copy V into the cache for every position
// of the chunk, one (m, kv-head) pair per block iteration. Runs before the
// attention phase so causal reads inside the chunk see complete K/V.
__device__ void ph_kv_append(const long long* P, int pos, int M) {
  const long long max_seq = P[7];
  const int qk_flag = ((int)P[8]) & 0xFFFF;
  const float eps = bits2f(P[9]);
  const float* invf = reinterpret_cast<const float*>(P[10]);
  const int Hq = (int)P[11], Hkv = ((int)P[12]) & 0xFFFF;
  const int D = (int)P[13];
  const int asbits = (int)(P[15] >> 32);
  const float ascale = (asbits != 0) ? __int_as_float(asbits) : 1.0f;
  const long long KT = (long long)(Hq + 2 * Hkv) * D;
  const bf16* knw = reinterpret_cast<const bf16*>(P[6]);

  __shared__ float s_red[32];
  __shared__ float s_inv;
  __shared__ float hbuf[MAX_HEAD_DIM];

  const int total = M * Hkv;
  for (int v = blockIdx.x; v < total; v += gridDim.x) {
    const int m = v / Hkv;
    const int kvh = v % Hkv;
    const int pm = pos + m;
    const bf16* qkv = reinterpret_cast<const bf16*>(P[1]) + (long long)m * KT;
    bf16* kc = reinterpret_cast<bf16*>(P[2]) + (long long)kvh * max_seq * D;
    bf16* vc = reinterpret_cast<bf16*>(P[3]) + (long long)kvh * max_seq * D;
    kv_norm_rope_append(qkv, kc, vc, knw, qk_flag, eps, invf, Hq, Hkv, kvh,
                        D, pm, s_red, &s_inv, hbuf, ascale);
  }
}
// gemma post-sublayer norm + residual: hid = hid + norm(t) (+ optional
// layer scale). One vector; block 0 computes while the rest fall through.
__device__ void ph_normres(const long long* P) {
  if (blockIdx.x != 0) return;
  const bf16* t = reinterpret_cast<const bf16*>(P[1]);
  const bf16* w = reinterpret_cast<const bf16*>(P[2]);
  bf16* hid = reinterpret_cast<bf16*>(P[3]);
  const long long n = P[7];
  const float eps = bits2f(P[14]);
  const float lscale = bits2f(P[15]);
  __shared__ float s_red[32];
  __shared__ float s_inv;
  float ss = 0.f;
  for (long long k = threadIdx.x; k < n; k += blockDim.x) {
    const float v = bf2f(t[k]);
    ss = __fadd_rn(ss, __fmul_rn(v, v));
  }
  ss = block_sum(ss, s_red);
  if (threadIdx.x == 0) s_inv = detm::rsqrtf_det(ss / (float)n + eps);
  __syncthreads();
  for (long long k = threadIdx.x; k < n; k += blockDim.x) {
    const float nv = bf2f(f2bf((bf2f(t[k]) * s_inv) * bf2f(w[k])));
    float h = bf2f(f2bf(bf2f(hid[k]) + nv));
    if (lscale != 1.f) h = bf2f(f2bf(h * lscale));
    hid[k] = f2bf(h);
  }
}

// gemma per-layer-input mix: for each layer row, scale the context
// projection (bf16 round), gemma-norm it over the 256-wide block, add the
// token-identity row, and scale by 1/sqrt(2), mirroring the eager op
// order exactly.
__device__ void ph_plemix(const long long* P) {
  const bf16* ctx = reinterpret_cast<const bf16*>(P[1]);
  const bf16* w = reinterpret_cast<const bf16*>(P[2]);
  bf16* out = reinterpret_cast<bf16*>(P[3]);
  const bf16* tok = reinterpret_cast<const bf16*>(P[4]);
  const long long L = P[7], C = P[8];
  const float eps = bits2f(P[14]);
  const float pscale = bits2f(P[15]);
  __shared__ float s_red[32];
  __shared__ float s_inv;
  __shared__ float xbuf[512];
  for (long long l = blockIdx.x; l < L; l += gridDim.x) {
    float ss = 0.f;
    for (int j = threadIdx.x; j < (int)C; j += blockDim.x) {
      const float v = bf2f(f2bf(bf2f(ctx[l * C + j]) * pscale));
      xbuf[j] = v;
      ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    ss = block_sum(ss, s_red);
    if (threadIdx.x == 0) s_inv = detm::rsqrtf_det(ss / (float)C + eps);
    __syncthreads();
    for (int j = threadIdx.x; j < (int)C; j += blockDim.x) {
      const float nv = bf2f(f2bf((xbuf[j] * s_inv) * bf2f(w[j])));
      const float mixed = bf2f(f2bf(nv + bf2f(tok[l * C + j])));
      out[l * C + j] = f2bf(mixed * 0.70710678118654752440f);
    }
    __syncthreads();
  }
}

__device__ void ph_argmax_part(const long long* P, int M) {
  const float* lg = reinterpret_cast<const float*>(P[1]);
  unsigned long long* parts = reinterpret_cast<unsigned long long*>(P[3]);
  const long long V = P[7];
  __shared__ unsigned long long s64[32];
  for (int m = 0; m < M; ++m) {
    unsigned long long best = amax_key(-INFINITY, 0xFFFFFFFFu);
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < V; i += (long long)gridDim.x * blockDim.x) {
      // Greedy parity with transformers: argmax over the bf16-rounded
      // logits, lowest index on ties (torch.argmax convention).
      const unsigned long long k =
          amax_key(bf2f(f2bf(lg[(long long)m * V + i])), (unsigned)i);
      best = (k > best) ? k : best;
    }
    best = block_max_u64(best, s64);
    if (threadIdx.x == 0)
      parts[(long long)m * gridDim.x + blockIdx.x] = best;
    if (m + 1 < M) __syncthreads();
  }
}

// Row m finalizes on block m; slot 8 carries the per-row output stride
// (0 for the single-sequence programs, whose only row is m = 0).
__device__ void ph_argmax_fin(const long long* P, int step_slot, int M) {
  if ((int)blockIdx.x >= M) return;
  const int m = (int)blockIdx.x;
  const unsigned long long* parts =
      reinterpret_cast<const unsigned long long*>(P[1]) +
      (long long)m * gridDim.x;
  int* tok = reinterpret_cast<int*>(P[3]);
  int* toks_out = reinterpret_cast<int*>(P[4]);
  const long long tstride = P[8];
  unsigned long long best = amax_key(-INFINITY, 0xFFFFFFFFu);
  for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
    const unsigned long long k = parts[i];
    best = (k > best) ? k : best;
  }
  __shared__ unsigned long long s64[32];
  best = block_max_u64(best, s64);
  if (threadIdx.x == 0) {
    const int idx = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFull));
    tok[m] = idx;
    if (toks_out != nullptr && step_slot >= 0)
      toks_out[(long long)m * tstride + step_slot] = idx;
  }
}

// ---- batched-decode transform phases: transformed [B][K] input -> scratch
// Reductions and op chains reproduce ph_gemv's fused staging exactly, so
// OP_GEMV_PLAIN is bit-identical to a fused OP_GEMV: a batched sequence's
// logits equal its batch-1 run.

// RMSNorm one row per block (llama IT_RMSNORM or gemma IT_RMSNORM_G).
__device__ void ph_normb(const long long* P, int M) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const long long K = P[8];
  const bf16* xin = reinterpret_cast<const bf16*>(P[1]) + (long long)m * K;
  const bf16* gamma = reinterpret_cast<const bf16*>(P[2]);
  bf16* xg = reinterpret_cast<bf16*>(P[3]) + (long long)m * K;
  const int variant = (int)P[9] & ITF_BASE_MASK;
  const float eps = bits2f(P[14]);
  __shared__ float s_red[32];
  __shared__ float s_scale;
  const long long G = K / 8;  // K % 8 == 0 by contract
  float ss = 0.f;
  for (long long g = threadIdx.x; g < G; g += blockDim.x) {
    const int4 x4 = *reinterpret_cast<const int4*>(xin + g * 8);
    const __nv_bfloat162* xp =
        reinterpret_cast<const __nv_bfloat162*>(&x4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float lo = __low2float(xp[j]);
      const float hi = __high2float(xp[j]);
      ss = __fadd_rn(ss, __fmul_rn(lo, lo));
      ss = __fadd_rn(ss, __fmul_rn(hi, hi));
    }
  }
  ss = block_sum(ss, s_red);
  if (threadIdx.x == 0) s_scale = detm::rsqrtf_det(ss / (float)K + eps);
  __syncthreads();
  const float inv = s_scale;
  for (long long g = threadIdx.x; g < G; g += blockDim.x) {
    const int4 x4 = *reinterpret_cast<const int4*>(xin + g * 8);
    const int4 g4 = *reinterpret_cast<const int4*>(gamma + g * 8);
    const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&x4);
    const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&g4);
    int4 o4;
    __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (variant == IT_RMSNORM_G) {
        op[j].x = f2bf((__low2float(xp[j]) * inv) * __low2float(gp[j]));
        op[j].y = f2bf((__high2float(xp[j]) * inv) * __high2float(gp[j]));
      } else {
        op[j].x = f2bf(__low2float(gp[j]) *
                       bf2f(f2bf(__low2float(xp[j]) * inv)));
        op[j].y = f2bf(__high2float(gp[j]) *
                       bf2f(f2bf(__high2float(xp[j]) * inv)));
      }
    }
    *reinterpret_cast<int4*>(xg + g * 8) = o4;
  }
}

// SwiGLU / gelu-glu gating: xg[m][k] = act(gate[m][k]) * up[m][k].
__device__ void ph_glub(const long long* P, int M) {
  const long long K = P[8];
  const bf16* xin = reinterpret_cast<const bf16*>(P[1]);  // [M][2K]
  bf16* xg = reinterpret_cast<bf16*>(P[3]);               // [M][K]
  const int variant = (int)P[9] & ITF_BASE_MASK;
  const long long tot = (long long)M * K;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < tot; i += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(i / K);
    const long long k = i % K;
    const bf16* gate = xin + (long long)m * 2 * K;
    const float g = bf2f(gate[k]);
    const float up = bf2f(gate[K + k]);
    if (variant == IT_GELU_GLU) {
      const float act = bf2f(f2bf(gelu_tanh_f(g)));
      xg[i] = f2bf(act * up);
    } else {
      const float sig = 1.f / (1.f + detm::expf_det(-g));
      const float sg = bf2f(f2bf(g * sig));
      xg[i] = f2bf(sg * up);
    }
  }
}

// Finalize split-S attention for the O-projection input.
__device__ void ph_attnfinb(const long long* P, int M, const int* pos_b,
                            int step) {
  const long long K = P[8];  // qdim = Hq * D
  const float* part = reinterpret_cast<const float*>(P[1]);
  bf16* xg = reinterpret_cast<bf16*>(P[3]);
  const int Hq = (int)P[11];
  const int D = (int)P[13];
  const int chbits = (int)(P[12] >> 16) & 0xFFFF;
  const int CH = (chbits > 0 && chbits <= ATTN_CHUNK) ? chbits : ATTN_CHUNK;
  const int maxch = (int)(P[15] & 0xFFFF);
  const int fin_win = (int)((P[15] >> 16) & 0xFFFF);
  const long long tot = (long long)M * K;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < tot; i += (long long)gridDim.x * blockDim.x) {
    const int m = (int)(i / K);
    const long long k = i % K;
    const int pm = pm_of(0, m, pos_b, step);
    const int nch = (pm + CH) / CH;
    const int s_lo = (fin_win > 0 && pm + 1 > fin_win) ? pm + 1 - fin_win : 0;
    const int nch_lo = s_lo / CH;
    const float* mpart =
        part + (long long)m * ((long long)Hq * maxch * (D + 2));
    xg[i] = f2bf(attn_finalize(mpart, (int)(k / D), (int)(k % D), D, maxch,
                               nch, nch_lo));
  }
}

// GEMV over a global [M][K] input already transformed into scratch: no
// shared staging, so batch width is bounded by the NB accumulators only.
// Ring, tile order, and warp reduction are ph_gemv's: bit-identical.
template <int NB>
__device__ void ph_gemv_plain(const long long* P, int M, int rs) {
  const bf16* W = reinterpret_cast<const bf16*>(P[2]);
  const bf16* resid = reinterpret_cast<const bf16*>(P[5]);
  const bf16* xin = reinterpret_cast<const bf16*>(P[1]);
  const long long N = P[7], K = P[8];
  const bool tiled = ((P[9] >> 20) & 1) != 0;
  const int epi = (int)P[10];

  extern __shared__ unsigned char smem_raw[];
  int4* rng = reinterpret_cast<int4*>(smem_raw);
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const long long gwarp =
      ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  const int rmask = rs - 1;
  int4* ringw = rng + (wib * rs) * 32 + lane;
  const long long kbase = (long long)lane * 8;
  const long long ntiles = (kbase < K) ? (K - kbase + 255) / 256 : 0;
  const long long K8 = K >> 3;
  long long irow = gwarp, itile = 0, ig = 0, cg = 0;
  auto issue_one = [&]() {
    if (irow < N && ntiles > 0) {
      const bf16* src =
          tiled ? W + (((irow >> 3) * K8 + (kbase >> 3) + itile * 32) << 6) +
                      ((irow & 7) << 3)
                : W + irow * K + kbase + itile * 256;
      __pipeline_memcpy_async(ringw + ((int)ig & rmask) * 32, src, 16);
      __pipeline_commit();
      ++ig;
      if (++itile == ntiles) {
        itile = 0;
        irow += nwarps;
      }
    }
  };
  for (int i = 0; i < rs; ++i) issue_one();

  for (long long row = gwarp; row < N; row += nwarps) {
    float acc[NB];
#pragma unroll
    for (int m = 0; m < NB; ++m) acc[m] = 0.f;
    for (long long t = 0; t < ntiles; ++t) {
      __pipeline_wait_prior((int)(ig - cg - 1));
      const int4 w4 = ringw[((int)cg & rmask) * 32];
      ++cg;
      issue_one();
      const long long k = kbase + t * 256;
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w4);
#pragma unroll
      for (int m = 0; m < NB; ++m) {
        if (m < M) {
          const __nv_bfloat162* xp =
              reinterpret_cast<const __nv_bfloat162*>(xin +
                                                      (long long)m * K + k);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            acc[m] = fmaf(__low2float(wp[j]), __low2float(xp[j]), acc[m]);
            acc[m] = fmaf(__high2float(wp[j]), __high2float(xp[j]), acc[m]);
          }
        }
      }
    }
#pragma unroll
    for (int m = 0; m < NB; ++m) {
      if (m < M) {
        const float a = warp_sum(acc[m]);
        if (lane == 0) gemv_store(P, epi, N, resid, m, row, a);
      }
    }
  }
}

// GEMV with nf4-packed weights [N][K/2] (two 4-bit indices per byte, row
// major), per-64-element fp32 absmax. Each lane dequantizes its own tile
// (code[nibble] * absmax, bf16-rounded) and accumulates in the dense
// path's order: bit-identical to a dense GEMV over the dequantized
// weights. Input is dense bf16 in global scratch; weights stay packed.
template <int NB>
__device__ void ph_gemv_q4(const long long* P, int M) {
  const unsigned char* W = reinterpret_cast<const unsigned char*>(P[2]);
  const float* absmax = reinterpret_cast<const float*>(P[6]);
  const bf16* resid = reinterpret_cast<const bf16*>(P[5]);
  const bf16* xin = reinterpret_cast<const bf16*>(P[1]);
  const long long N = P[7], K = P[8];
  const int epi = (int)P[10];
  const int lane = threadIdx.x & 31;
  const long long gwarp =
      ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long nwarps = ((long long)gridDim.x * blockDim.x) >> 5;
  const long long kbase = (long long)lane * 8;
  const long long ntiles = (kbase < K) ? (K - kbase + 255) / 256 : 0;
  const long long Kh = K >> 1;  // packed bytes per row

  for (long long row = gwarp; row < N; row += nwarps) {
    float acc[NB];
#pragma unroll
    for (int m = 0; m < NB; ++m) acc[m] = 0.f;
    for (long long t = 0; t < ntiles; ++t) {
      const long long k = kbase + t * 256;
      const unsigned int pw =
          *reinterpret_cast<const unsigned int*>(W + row * Kh + (k >> 1));
      const float am = absmax[(row * K + k) >> 6];
      float w8[8];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const unsigned int by = (pw >> (b * 8)) & 0xFFu;
        // bitsandbytes packs the earlier element in the high nibble
        w8[2 * b] = bf2f(f2bf(NF4_CODE[by >> 4] * am));
        w8[2 * b + 1] = bf2f(f2bf(NF4_CODE[by & 0xF] * am));
      }
#pragma unroll
      for (int m = 0; m < NB; ++m) {
        if (m < M) {
          const bf16* xp = xin + (long long)m * K + k;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            acc[m] = fmaf(w8[j], bf2f(xp[j]), acc[m]);
        }
      }
    }
#pragma unroll
    for (int m = 0; m < NB; ++m) {
      if (m < M) {
        const float a = warp_sum(acc[m]);
        if (lane == 0) gemv_store(P, epi, N, resid, m, row, a);
      }
    }
  }
}

template <int NB>
__device__ __forceinline__ void dispatch_phase(const long long* P, int pos,
                                               int M, int step_slot,
                                               int prompt_len,
                                               int ring_stages,
                                               const int* pos_b, int step,
                                               long long kv_bstride) {
  switch ((int)P[0]) {
    case OP_EMBED: ph_embed(P, pos, M, prompt_len, step_slot, pos_b); break;
    case OP_GEMV: ph_gemv<NB>(P, pos, M, ring_stages, pos_b, step); break;
    case OP_ATTN: ph_attn(P, pos, M, pos_b, step, kv_bstride); break;
    case OP_ARGMAX_PART: ph_argmax_part(P, M); break;
    case OP_ARGMAX_FIN: ph_argmax_fin(P, step_slot, M); break;
    case OP_KV_APPEND: ph_kv_append(P, pos, M); break;
    case OP_NORMRES: ph_normres(P); break;
    case OP_PLEMIX: ph_plemix(P); break;
    case OP_GEMV_PLAIN: ph_gemv_plain<NB>(P, M, ring_stages); break;
    case OP_GEMV_Q4: ph_gemv_q4<NB>(P, M); break;
    case OP_NORMB: ph_normb(P, M); break;
    case OP_GLUB: ph_glub(P, M); break;
    case OP_ATTNFINB: ph_attnfinb(P, M, pos_b, step); break;
    default: break;
  }
}

// Sense-reversing grid barrier over co-resident blocks; the launcher
// sizes the grid from the occupancy API, so the spin cannot deadlock.
// Counters return to zero per crossing; the sense word persists across
// launches. State (int32): [0..31] striped arrival counters, [32]
// level-2 counter, [33] sense word.
#define BAR_STRIPES 32

__device__ void grid_barrier(int* bar, int* local_sense) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const int ls = *local_sense ^ 1;
    *local_sense = ls;
    __threadfence();
    // Two-level arrival: one shared line costs ~2us of serialized RMWs;
    // 32 striped counters + a 32-wide second level cut it to tens of ns.
    const int nb = (int)gridDim.x;
    const int stripe = (int)blockIdx.x & (BAR_STRIPES - 1);
    const int in_stripe =
        nb / BAR_STRIPES + ((stripe < (nb % BAR_STRIPES)) ? 1 : 0);
    int* sense = bar + BAR_STRIPES + 1;
    if (atomicAdd(bar + stripe, 1) == in_stripe - 1) {
      atomicExch(bar + stripe, 0);
      __threadfence();
      const int nstripes = (nb < BAR_STRIPES) ? nb : BAR_STRIPES;
      if (atomicAdd(bar + BAR_STRIPES, 1) == nstripes - 1) {
        atomicExch(bar + BAR_STRIPES, 0);
        __threadfence();
        atomicExch(sense, ls);
      } else {
        const volatile int* vs = sense;
        while (*vs != ls) {
        }
      }
    } else {
      const volatile int* vs = sense;
        while (*vs != ls) {
        }
    }
    __threadfence();
  }
  __syncthreads();
}

// One launch consumes `total` positions: prompt in chunks of up to
// `chunk_m` (weights amortized across the chunk), generation one at a
// time with each step's argmax feeding the next embedding read; the grid
// barrier carries across every boundary. OP_KV_APPEND runs only in
// multi-token chunks; decode skips dispatch and barrier uniformly.
// Two blocks per SM (H200 at one block/SM measures 1.5x slower), so the
// register budget is pinned to 128; the ring keeps the hot loop's frame
// under it.
template <int NB>
__global__ void __launch_bounds__(BLOCK_THREADS, 2)
    mega_kernel(const long long* __restrict__ prog, int n_phases,
                int pos0, int total, int slot0, int prompt_len,
                int chunk_m, int ring_stages, int* bar,
                long long* __restrict__ ts,
                const int* __restrict__ pos_b, int B,
                long long kv_bstride) {
  __shared__ int s_sense;
  if (threadIdx.x == 0) s_sense = atomicAdd(bar + BAR_STRIPES + 1, 0);
  __syncthreads();
  // diagnostic: block 0 stamps its SM clock at entry and after each
  // barrier crossing of the first step (pure observation, no data effect)
  if (ts != nullptr && blockIdx.x == 0 && threadIdx.x == 0)
    ts[0] = (long long)clock64();
  const bool batch = (pos_b != nullptr);
  int pos = pos0;
  const int pos_end = pos0 + total;
  while (pos < pos_end) {
    int M = 1;
    if (batch) {
      M = B;  // one decode step for every sequence; positions per row
    } else if (pos < prompt_len) {
      M = prompt_len - pos;
      if (M > chunk_m) M = chunk_m;
      if (pos + M > pos_end) M = pos_end - pos;
    }
    const int step = pos - pos0;
    const int slot = batch ? (slot0 + step) : (slot0 + (pos + M - 1 - pos0));
    const bool final_step = batch ? (pos + 1 >= pos_end)
                                  : (pos + M >= pos_end);
    for (int p = 0; p < n_phases; ++p) {
      const long long* P = prog + (long long)p * 16;
      const int op = (int)P[0];
      if (op == OP_KV_APPEND && (M == 1 || batch)) continue;  // uniform skip
      if (op == OP_ARGMAX_FIN && !final_step && !batch)
        continue;  // embed derives it; batch feeds tokens through it
      dispatch_phase<NB>(P, pos, M, slot, batch ? -1 : prompt_len,
                         ring_stages, pos_b, step, kv_bstride);
      if (!(final_step && p + 1 == n_phases)) {
        // Warm the next GEMV's first weight tiles into L2 while the
        // barrier settles; weights are constant, no memory semantics.
        for (int q = p + 1; q < n_phases; ++q) {
          const long long* Pn = prog + (long long)q * 16;
          const int opn = (int)Pn[0];
          if (opn == OP_KV_APPEND && (M == 1 || batch)) continue;
          if (opn == OP_GEMV || opn == OP_GEMV_PLAIN) {
            const long long w =
                ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
            const long long kb = (long long)(threadIdx.x & 31) * 8;
            if (w < Pn[7] && kb < Pn[8]) {
              const long long off =
                  ((Pn[9] >> 20) & 1)
                      ? (((w >> 3) * (Pn[8] >> 3) + (kb >> 3)) << 6) +
                            ((w & 7) << 3)
                      : w * Pn[8] + kb;
              const char* wr =
                  reinterpret_cast<const char*>(Pn[2]) + off * 2;
              asm volatile("prefetch.global.L2 [%0];" ::"l"(wr));
              asm volatile("prefetch.global.L2 [%0];" ::"l"(wr + 512));
            }
          }
          break;
        }
        grid_barrier(bar, &s_sense);
      }
      if (ts != nullptr && pos == pos0 && blockIdx.x == 0 &&
          threadIdx.x == 0)
        ts[p + 1] = (long long)clock64();
    }
    pos += batch ? 1 : M;
  }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 2)
    phase_kernel(const long long* __restrict__ prog, int p,
                 int pos, int step_slot, int prompt_len,
                 int ring_stages) {
  const long long* P = prog + (long long)p * 16;
  if ((int)P[0] == OP_KV_APPEND) return;
  dispatch_phase<8>(P, pos, 1, step_slot, -1, ring_stages, nullptr, 0, 0);
}

// Batch width of the global-scratch decode kernel: acc[MAK_MAXB_BIG] per
// lane. 16 keeps the register frame under the launch_bounds(256, 2) budget.
#define MAK_MAXB_BIG 16

}  // namespace mak

#include <stdlib.h>

// cp.async ring depth: power-of-two stages fitting the shared-memory
// budget at two blocks per SM (a stage is 4KB). Pure scheduling: any
// value computes identical bits.
int mak_ring_stages_impl(int stage_bytes) {
  static int cached_dev = -1;
  static int cached_stage = -1;
  static int cached_rs = -1;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != cached_dev || stage_bytes != cached_stage || cached_rs < 0) {
    int shm_sm = 0, optin = 0;
    cudaDeviceGetAttribute(&shm_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                           dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           dev);
    cudaFuncAttributes fa{};
    cudaFuncGetAttributes(&fa, mak::mega_kernel<8>);
    const int stat = (int)fa.sharedSizeBytes;
    long long by_occ = ((long long)shm_sm / 2 - stat - stage_bytes) /
                       GEMV_RING_STAGE_BYTES;
    long long by_win = ((long long)optin - stat - stage_bytes) /
                       GEMV_RING_STAGE_BYTES;
    if (by_occ < 4) by_occ = 4;  // never starve the ring for occupancy
    long long rs = (by_occ < by_win) ? by_occ : by_win;
    if (rs > 4) rs = 4;  // deeper rings measure flat-to-worse on every card
    if (rs < 1) rs = 1;
    while (rs & (rs - 1)) rs &= rs - 1;  // power-of-two floor
    const char* env = getenv("MAK_RING");
    if (env != nullptr) {
      long long r = atoll(env);
      if (r >= 1 && r <= 16 && !(r & (r - 1)) &&
          (long long)stat + stage_bytes + r * GEMV_RING_STAGE_BYTES <= optin)
        rs = r;
    }
    cached_rs = (int)rs;
    cached_dev = dev;
    cached_stage = stage_bytes;
  }
  return cached_rs;
}

int mak_grid_blocks_impl(int smem_bytes) {
  static int cached_dev = -1;
  static int cached_smem = -1;
  static int cached_grid = -1;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != cached_dev || smem_bytes != cached_smem || cached_grid < 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    // staging + weight ring can exceed the default 48KB dynamic window;
    // the settable maximum is the opt-in limit minus the kernel's static
    // shared allocation
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           dev);
    cudaFuncAttributes fa{};
    cudaFuncGetAttributes(&fa, mak::mega_kernel<8>);
    cudaFuncSetAttribute(mak::mega_kernel<8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         optin - (int)fa.sharedSizeBytes);
    cudaFuncGetAttributes(&fa, mak::mega_kernel<MAK_MAXB_BIG>);
    cudaFuncSetAttribute(mak::mega_kernel<MAK_MAXB_BIG>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         optin - (int)fa.sharedSizeBytes);
    cudaFuncGetAttributes(&fa, mak::phase_kernel);
    cudaFuncSetAttribute(mak::phase_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         optin - (int)fa.sharedSizeBytes);
    cudaGetLastError();  // never leave a failed attribute set sticky
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mak::mega_kernel<8>,
                                                  BLOCK_THREADS, smem_bytes);
    if (nb < 1) nb = 1;
    int grid = nb * prop.multiProcessorCount;
    const char* env = getenv("MAK_GRID");
    if (env != nullptr) {
      const int g = atoi(env);
      if (g > 0 && g <= grid) grid = g;
    }
    cached_grid = grid;
    cached_dev = dev;
    cached_smem = smem_bytes;
  }
  return cached_grid;
}

// Grid for the global-scratch batch kernel, sized from its own occupancy
// (acc[MAK_MAXB_BIG] raises the register frame, which can lower blocks per
// SM; the grid must match so every block stays co-resident for the barrier).
int mak_grid_blocks_big_impl(int smem_bytes) {
  static int cached_dev = -1;
  static int cached_smem = -1;
  static int cached_grid = -1;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != cached_dev || smem_bytes != cached_smem || cached_grid < 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, mak::mega_kernel<MAK_MAXB_BIG>, BLOCK_THREADS, smem_bytes);
    if (nb < 1) nb = 1;
    int grid = nb * prop.multiProcessorCount;
    const char* env = getenv("MAK_GRID");
    if (env != nullptr) {
      const int g = atoi(env);
      if (g > 0 && g <= grid) grid = g;
    }
    cached_grid = grid;
    cached_dev = dev;
    cached_smem = smem_bytes;
  }
  return cached_grid;
}

int mak_batch_maxb_impl() { return MAK_MAXB_BIG; }

cudaError_t mak_launch_mega(const long long* prog, int n_phases, int pos0,
                            int total, int slot0, int prompt_len,
                            int chunk_m, int ring_stages, int* bar, int grid,
                            int smem_bytes, cudaStream_t stream,
                            long long* ts) {
  mak::mega_kernel<8><<<grid, BLOCK_THREADS, smem_bytes, stream>>>(
      prog, n_phases, pos0, total, slot0, prompt_len, chunk_m, ring_stages,
      bar, ts, nullptr, 1, 0);
  return cudaGetLastError();
}

cudaError_t mak_launch_mega_batch(const long long* prog, int n_phases,
                                  int steps, int slot0, int ring_stages,
                                  int* bar, int grid, int smem_bytes,
                                  const int* pos_b, int B,
                                  long long kv_bstride,
                                  cudaStream_t stream) {
  mak::mega_kernel<8><<<grid, BLOCK_THREADS, smem_bytes, stream>>>(
      prog, n_phases, 0, steps, slot0, -1, 1, ring_stages, bar, nullptr,
      pos_b, B, kv_bstride);
  return cudaGetLastError();
}

cudaError_t mak_launch_mega_batch_big(const long long* prog, int n_phases,
                                      int steps, int slot0, int ring_stages,
                                      int* bar, int grid, int smem_bytes,
                                      const int* pos_b, int B,
                                      long long kv_bstride,
                                      cudaStream_t stream) {
  mak::mega_kernel<MAK_MAXB_BIG><<<grid, BLOCK_THREADS, smem_bytes, stream>>>(
      prog, n_phases, 0, steps, slot0, -1, 1, ring_stages, bar, nullptr,
      pos_b, B, kv_bstride);
  return cudaGetLastError();
}

cudaError_t mak_launch_phase(const long long* prog, int phase, int pos,
                             int step_slot, int ring_stages, int grid,
                             int smem_bytes, cudaStream_t stream) {
  mak::phase_kernel<<<grid, BLOCK_THREADS, smem_bytes, stream>>>(
      prog, phase, pos, step_slot, 0, ring_stages);
  return cudaGetLastError();
}







