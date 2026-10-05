#pragma once
// ============================================================================
// mk/gemv.cuh -- the matrix-vector cores.
//
// At batch 1 every projection in a transformer is a gemv, so these four
// functions are where essentially all of a megakernel's time goes.  Each
// computes R output rows with one warp:
//
//     acc[r] = sum_k  W[row0+r][k] * xs[k]
//
// `xs` is the activation vector already in shared memory in the padded [G][33]
// layout (see mk/common.cuh).  R is the row-blocking factor: it amortises the
// shared-memory reads of xs across R rows, at the cost of dividing the number
// of parallel work items.  The planner picks it from measured rates.
// ============================================================================
#include "common.cuh"

namespace mk {

// ------------------------------------------------------------ dense (bf16/f16)
// R consecutive rows of a row-major [rows][K] matrix with leading dimension ldw.
//
// The loop is split into a fixed-trip-count body plus a tail, rather than the
// obvious `for (k0 = lane*VEC; k0 < K; k0 += 32*VEC)`.  That form has a
// lane-dependent trip count whenever K is not a multiple of 32*VEC (2880 is
// not), so the compiler cannot unroll it, and each warp then has only R loads
// in flight.  At batch 1 a per-layer stage is a single wave of warps, so
// memory-level parallelism per warp IS the achieved bandwidth: `ncu` showed
// 44% of DRAM peak at 18% SM throughput, i.e. idle memory pipes and plenty of
// issue headroom.  With a compile-time trip count, U unrolled steps put U*R
// loads in flight.
// U is chosen so that U*R vector loads are in flight regardless of R: the
// memory-level parallelism a warp needs is a property of the machine, while R is
// a property of the schedule.  Letting them multiply is how a kernel ends up at
// 128 registers and one block per SM.
template<class T, int K, int R, int U = (8 + R - 1) / R>
__device__ __forceinline__ void gemv_dense(const T* __restrict__ W, int ldw,
                                           const float* __restrict__ xs,
                                           int lane, float acc[R]) {
    constexpr int VEC  = 16 / sizeof(T);         // 16 B per lane per step
    constexpr int STEP = 32 * VEC;
    constexpr int FULL = K / STEP;               // every lane runs all of these
    constexpr int TAIL = K - FULL * STEP;        // a multiple of VEC
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    #pragma unroll U
    for (int i = 0; i < FULL; ++i) {
        const int k0 = lane * VEC + i * STEP;
        float4 v[R];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            v[r] = *reinterpret_cast<const float4*>(W + (size_t)r * ldw + k0);
        const float* xp = xs + pad33(k0);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float xv = xp[j];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const T* hb = reinterpret_cast<const T*>(&v[r]);
                acc[r] = fmaf(to_f<T>(hb[j]), xv, acc[r]);
            }
        }
    }
    if (TAIL) {
        const int k0 = FULL * STEP + lane * VEC;
        if (k0 < K) {
            float4 v[R];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                v[r] = *reinterpret_cast<const float4*>(W + (size_t)r * ldw + k0);
            const float* xp = xs + pad33(k0);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float xv = xp[j];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const T* hb = reinterpret_cast<const T*>(&v[r]);
                    acc[r] = fmaf(to_f<T>(hb[j]), xv, acc[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
}

// ------------------------------------------------------------ mxfp4
// W is stored as blocks u8[rows][G][16] plus scales u8[rows][G]; G = K/32.
// One lane owns a whole 32-value group, so the block scale folds into the
// accumulator once per group rather than once per value.
// One 32-value mxfp4 group of R rows, accumulated.  Factored out so the single
// and multi-expert gemvs are provably the same arithmetic.
template<int G, int R>
__device__ __forceinline__ void mxfp4_group(const uint4* __restrict__ blk,
                                            const uint8_t* __restrict__ scl,
                                            const float* __restrict__ xs,
                                            int g, float acc[R]) {
    uint4 pk[R]; float sf[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        pk[r] = blk[(size_t)r * G + g];
        sf[r] = mx_scale(scl[(size_t)r * G + g]);
    }
    const float* xp = xs + g * 33;
    float part[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) part[r] = 0.f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        __half2 h[R][4];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            mxfp4_8(reinterpret_cast<const uint32_t*>(&pk[r])[q], h[r]);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float x0 = xp[q * 8 + t * 2 + 0];
            const float x1 = xp[q * 8 + t * 2 + 1];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                float2 f = __half22float2(h[r][t]);
                part[r] = fmaf(f.x, x0, part[r]);
                part[r] = fmaf(f.y, x1, part[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = fmaf(part[r], sf[r], acc[r]);
}

template<int G, int R>
__device__ __forceinline__ void gemv_mxfp4(const uint4* __restrict__ blk,
                                           const uint8_t* __restrict__ scl,
                                           const float* __restrict__ xs,
                                           int lane, float acc[R]) {
    // Fixed-trip-count body plus tail, for the same reason as gemv_dense: G is
    // 90 for a 2880-wide model, so `g = lane; g < G; g += 32` runs 2 or 3 times
    // depending on the lane and cannot unroll.
    constexpr int FULL = G / 32;
    constexpr int TAIL = G - FULL * 32;
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    #pragma unroll (R >= 4 ? 1 : 2)
    for (int gi = 0; gi < FULL; ++gi)
        mxfp4_group<G, R>(blk, scl, xs, lane + gi * 32, acc);
    if (TAIL && lane < TAIL)
        mxfp4_group<G, R>(blk, scl, xs, lane + FULL * 32, acc);
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
}

// ------------------------------------------------------------ mxfp4, many experts
// The top-k expert down-projections are independent gemvs over the SAME output
// rows.  Running them as k separate calls gives a warp only R loads in flight
// and measured 779 GB/s of a 2800 GB/s core; interleaving them in one group
// loop puts k*R loads in flight for the same arithmetic and the same registers
// per accumulator.
template<int G, int R, int NE>
__device__ __forceinline__ void gemv_mxfp4_multi(const uint4* const* __restrict__ blk,
                                                 const uint8_t* const* __restrict__ scl,
                                                 const float* const* __restrict__ xs,
                                                 int lane, float acc[NE][R]) {
    constexpr int FULL = G / 32;
    constexpr int TAIL = G - FULL * 32;
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[e][r] = 0.f;
    // NE already supplies the loads in flight; unrolling the group loop on top
    // would multiply the live registers again.
    #pragma unroll 1
    for (int gi = 0; gi < FULL; ++gi) {
        const int g = lane + gi * 32;
        #pragma unroll
        for (int e = 0; e < NE; ++e) mxfp4_group<G, R>(blk[e], scl[e], xs[e], g, acc[e]);
    }
    if (TAIL && lane < TAIL) {
        const int g = lane + FULL * 32;
        #pragma unroll
        for (int e = 0; e < NE; ++e) mxfp4_group<G, R>(blk[e], scl[e], xs[e], g, acc[e]);
    }
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[e][r] = warp_sum(acc[e][r]);
}

// ------------------------------------------------------------ dense, many experts
template<class T, int K, int R, int NE>
__device__ __forceinline__ void gemv_dense_multi(const T* const* __restrict__ W,
                                                 const float* const* __restrict__ xs,
                                                 int lane, float acc[NE][R]) {
    constexpr int VEC  = 16 / sizeof(T);
    constexpr int STEP = 32 * VEC;
    constexpr int FULL = K / STEP;
    constexpr int TAIL = K - FULL * STEP;
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[e][r] = 0.f;
    #pragma unroll 1
    for (int i = 0; i < FULL; ++i) {
        const int k0 = lane * VEC + i * STEP;
        #pragma unroll
        for (int e = 0; e < NE; ++e) {
            float4 v[R];
            #pragma unroll
            for (int r = 0; r < R; ++r)
                v[r] = *reinterpret_cast<const float4*>(W[e] + (size_t)r * K + k0);
            const float* xp = xs[e] + pad33(k0);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const float xv = xp[j];
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const T* hb = reinterpret_cast<const T*>(&v[r]);
                    acc[e][r] = fmaf(to_f<T>(hb[j]), xv, acc[e][r]);
                }
            }
        }
    }
    if (TAIL) {
        const int k0 = FULL * STEP + lane * VEC;
        if (k0 < K) {
            #pragma unroll
            for (int e = 0; e < NE; ++e) {
                float4 v[R];
                #pragma unroll
                for (int r = 0; r < R; ++r)
                    v[r] = *reinterpret_cast<const float4*>(W[e] + (size_t)r * K + k0);
                const float* xp = xs[e] + pad33(k0);
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    const float xv = xp[j];
                    #pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const T* hb = reinterpret_cast<const T*>(&v[r]);
                        acc[e][r] = fmaf(to_f<T>(hb[j]), xv, acc[e][r]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int e = 0; e < NE; ++e)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[e][r] = warp_sum(acc[e][r]);
}

// ------------------------------------------------------------ fp8, block scales
// W is fp8 [rows][K]; S is fp32 [ceil(rows/BN)][SK], one scale per (BN x BK)
// tile.  BN=1, BK=K expresses the per-output-channel form that
// compressed-tensors emits, and BN=BK=128 the DeepSeek form, so one core covers
// both -- which is the point of normalising formats in the IR rather than in
// the kernel.
//
// A lane's 16 consecutive values never straddle a BK boundary (BK is a multiple
// of 16 in every real checkpoint), so the scale is one scalar load per step.
template<int K, int R, int BN, int BK>
__device__ __forceinline__ void gemv_fp8(const __nv_fp8_e4m3* __restrict__ W,
                                         const float* __restrict__ S, int sk, int n0,
                                         const float* __restrict__ xs,
                                         int lane, float acc[R]) {
    constexpr int VEC  = 16;
    constexpr int STEP = 32 * VEC;
    constexpr int FULL = K / STEP;
    constexpr int TAIL = K - FULL * STEP;
    // one scale row per output row: R consecutive rows are R consecutive scale
    // rows, not R uses of the same one
    const float* srow = S + (size_t)(n0 / BN) * sk;
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    #pragma unroll 2
    for (int i = 0; i < FULL + (TAIL ? 1 : 0); ++i) {
        const int k0 = lane * VEC + i * STEP;
        if (i == FULL && !(TAIL && k0 < K)) break;
        float4 v[R];
        #pragma unroll
        for (int r = 0; r < R; ++r)
            v[r] = *reinterpret_cast<const float4*>(W + (size_t)r * K + k0);
        float sc[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) sc[r] = srow[(size_t)r * sk + k0 / BK];
        const float* xp = xs + pad33(k0);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            const float xv = xp[j];
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const __nv_fp8_e4m3* hb = reinterpret_cast<const __nv_fp8_e4m3*>(&v[r]);
                acc[r] = fmaf(to_f<__nv_fp8_e4m3>(hb[j]) * sc[r], xv, acc[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
}

// ------------------------------------------------------------ int4, group scales
// Canonical layout, produced by the loader from AWQ or GPTQ: Q is uint8
// [rows][K/2] with value k in the low nibble of byte k/2 for even k, S and Z are
// half [rows][K/G].  Normalising at load time means one core serves both
// checkpoint conventions instead of one core per convention.
template<int K, int R, int G, bool SYM>
__device__ __forceinline__ void gemv_int4(const uint8_t* __restrict__ Q,
                                          const __half* __restrict__ S,
                                          const __half* __restrict__ Z,
                                          const float* __restrict__ xs,
                                          int lane, float acc[R]) {
    constexpr int VEC  = 32;            // 16 bytes = 32 nibbles
    constexpr int STEP = 32 * VEC;
    constexpr int FULL = K / STEP;
    constexpr int TAIL = K - FULL * STEP;
    constexpr int NG   = K / G;
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    #pragma unroll 2
    for (int i = 0; i < FULL + (TAIL ? 1 : 0); ++i) {
        const int k0 = lane * VEC + i * STEP;
        if (i == FULL && !(TAIL && k0 < K)) break;
        uint4 q[R]; float sc[R], zp[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            q[r] = *reinterpret_cast<const uint4*>(Q + (size_t)r * (K / 2) + k0 / 2);
            sc[r] = __half2float(S[(size_t)r * NG + k0 / G]);
            zp[r] = SYM ? 8.f : __half2float(Z[(size_t)r * NG + k0 / G]);
        }
        const float* xp = xs + pad33(k0);
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                // one word = 8 values; int4_8 yields the pairs (0,4)(1,5)(2,6)(3,7)
                __half2 h[4];
                int4_8(reinterpret_cast<const unsigned*>(&q[r])[w], zp[r], h);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float2 f = __half22float2(h[t]);
                    acc[r] = fmaf(f.x * sc[r], xp[w * 8 + t + 0], acc[r]);
                    acc[r] = fmaf(f.y * sc[r], xp[w * 8 + t + 4], acc[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = warp_sum(acc[r]);
}

// ------------------------------------------------------------ whole-block gemv
// One row per BLOCK rather than per warp.  Used where the row count is small
// relative to the grid (the MoE router: 128 rows, 132 blocks) -- one block per
// row computes each logit exactly once, instead of every block recomputing all
// of them and turning 0.7 MB into 97 MB of L2 traffic.
template<class T, int K>
__device__ __forceinline__ float gemv_dense_block(const T* __restrict__ W,
                                                  const float* __restrict__ xs,
                                                  float* scr) {
    constexpr int VEC = 16 / sizeof(T);
    float acc = 0.f;
    for (int k = threadIdx.x * VEC; k < K; k += blockDim.x * VEC) {
        const float4 v = *reinterpret_cast<const float4*>(W + k);
        const T* hb = reinterpret_cast<const T*>(&v);
        const float* xp = xs + pad33(k);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) acc = fmaf(to_f<T>(hb[j]), xp[j], acc);
    }
    return block_sum(acc, scr);
}

} // namespace mk
