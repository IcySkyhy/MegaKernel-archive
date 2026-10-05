#pragma once
// ============================================================================
// mk/common.cuh -- primitives shared by every generated megakernel.
//
// This file is hand-written and fixed: the compiler never emits into it, it
// only calls it.  Everything here is exhaustively tested (tests/) so that a
// generated kernel can be wrong only in its *composition*, never in its
// arithmetic.
// ============================================================================
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>

namespace mk {

// ---------------------------------------------------------------- rounding
// The reference implementation (HuggingFace transformers) materialises bf16 at
// specific points: linear outputs, RoPE, norm outputs, the LM head.  Matching
// exactly those roundings is what keeps end-to-end divergence at reduction
// noise instead of bf16 epsilon.  Being *more* accurate than the reference is
// not a virtue here; being faithful is.
__device__ __forceinline__ float rbf(float v) {
    return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ float rh(float v) {
    return __half2float(__float2half(v));
}

// Activation element type -> the rounding the reference applies after a linear.
template<class T> struct ActRound;
template<> struct ActRound<__nv_bfloat16> {
    __device__ static __forceinline__ float r(float v) { return rbf(v); }
};
template<> struct ActRound<__half> {
    __device__ static __forceinline__ float r(float v) { return rh(v); }
};
template<> struct ActRound<float> {
    __device__ static __forceinline__ float r(float v) { return v; }
};
// Weight-only formats never carry activations, so their "rounding" is identity.
template<> struct ActRound<__nv_fp8_e4m3> {
    __device__ static __forceinline__ float r(float v) { return v; }
};

template<class T> __device__ __forceinline__ float to_f(T v);
template<> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template<> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template<> __device__ __forceinline__ float to_f<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) {
    return __half2float(__half(v));
}
template<> __device__ __forceinline__ float to_f<__nv_fp8_e5m2>(__nv_fp8_e5m2 v) {
    return __half2float(__half(v));
}

template<class T> __device__ __forceinline__ T from_f(float v);
template<> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template<> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half(v); }
template<> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_fp8_e4m3 from_f<__nv_fp8_e4m3>(float v) {
    return __nv_fp8_e4m3(v);
}

// ---------------------------------------------------------------- reductions
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide sum.  `scr` must hold >= blockDim.x/32 floats.  Both syncs are
// required: the second publishes scr[0] to every thread.
__device__ __forceinline__ float block_sum(float v, float* scr) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scr[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nw) ? scr[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) scr[0] = v;
    __syncthreads();
    return scr[0];
}

// ---------------------------------------------------------------- activations
__device__ __forceinline__ float act_silu(float g)  { return g / (1.f + __expf(-g)); }
__device__ __forceinline__ float act_gelu_tanh(float g) {
    const float c = 0.7978845608028654f;           // sqrt(2/pi)
    return 0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g)));
}
__device__ __forceinline__ float act_gelu(float g) {
    return 0.5f * g * (1.f + erff(g * 0.7071067811865476f));
}
__device__ __forceinline__ float act_relu2(float g) { float r = fmaxf(g, 0.f); return r * r; }

// gpt-oss clamped swiglu: gate is clamped above, the linear branch on both
// sides, sigmoid gets an alpha, and the linear branch is offset by +1.
__device__ __forceinline__ float swiglu_clamped(float g, float l, float alpha, float limit) {
    g = fminf(g, limit);
    l = fmaxf(fminf(l, limit), -limit);
    return (g / (1.f + __expf(-alpha * g))) * (l + 1.f);
}

// ---------------------------------------------------------------- mxfp4
// fp16 high bytes of the eight e2m1 magnitudes {0,.5,1,1.5,2,3,4,6}.
__device__ __constant__ uint32_t MXLUT_A = 0x3E3C3800u;   // codes 0..3
__device__ __constant__ uint32_t MXLUT_B = 0x46444240u;   // codes 4..7

// Decode 8 packed e2m1 nibbles into 4 half2.  Exact: every e2m1 value is
// representable in fp16, so this is a bit permutation, not arithmetic.  The
// shared block scale is deliberately NOT applied here -- it folds into the
// accumulator once per 32-value group.
//
// Signs: nibble j's sign is at bit 4j+3, so in (w<<4) the EVEN nibbles' signs
// land on byte MSBs and in w itself the ODD ones do.  One PRMT per output pair
// gathers the right two bytes into positions 1 and 3 -- exactly bits 15 and 31,
// the two fp16 sign bits.  23 integer ops -> 9.
__device__ __forceinline__ void mxfp4_8(uint32_t w, __half2 out[4]) {
    const uint32_t mag = w & 0x77777777u;
    uint32_t h03 = __byte_perm(MXLUT_A, MXLUT_B,  mag        & 0xFFFFu);
    uint32_t h47 = __byte_perm(MXLUT_A, MXLUT_B, (mag >> 16) & 0xFFFFu);
    uint32_t p01 = __byte_perm(h03, 0u, 0x1404u);
    uint32_t p23 = __byte_perm(h03, 0u, 0x3424u);
    uint32_t p45 = __byte_perm(h47, 0u, 0x1404u);
    uint32_t p67 = __byte_perm(h47, 0u, 0x3424u);
    const uint32_t t = w << 4;
    p01 |= __byte_perm(t, w, 0x4000u) & 0x80008000u;
    p23 |= __byte_perm(t, w, 0x5010u) & 0x80008000u;
    p45 |= __byte_perm(t, w, 0x6020u) & 0x80008000u;
    p67 |= __byte_perm(t, w, 0x7030u) & 0x80008000u;
    out[0] = *reinterpret_cast<__half2*>(&p01);
    out[1] = *reinterpret_cast<__half2*>(&p23);
    out[2] = *reinterpret_cast<__half2*>(&p45);
    out[3] = *reinterpret_cast<__half2*>(&p67);
}
// e8m0 exponent byte -> 2^(s-127), as a bit pattern.
__device__ __forceinline__ float mx_scale(int s) { return __int_as_float(s << 23); }

// ---------------------------------------------------------------- work split
// How a stage's work items are handed to warps.
//
// The obvious grid-stride form, `it = blockIdx.x*NW + wid; it += gridDim.x*NW`,
// gives every block NW consecutive items -- and when a stage has fewer items
// than the grid has warps (which at batch 1 is most of them), it packs all the
// work into the first items/NW blocks and leaves the remaining SMs completely
// idle.  An idle SM issues no memory requests, so this shows up directly as
// lost bandwidth: qkv on gpt-oss ran on 160 of 264 blocks.
//
// Instead give every block an equal CONTIGUOUS chunk.  No SM is idle while work
// remains, and a block's rows stay contiguous, which is what the DRAM page
// locality wants.
__device__ __forceinline__ int ws_lo(int nitem) {
    return blockIdx.x * ((nitem + gridDim.x - 1) / gridDim.x);
}
__device__ __forceinline__ int ws_hi(int nitem) {
    const int chunk = (nitem + gridDim.x - 1) / gridDim.x;
    return min(blockIdx.x * chunk + chunk, nitem);
}

// ---------------------------------------------------------------- int4
// Eight packed 4-bit values -> eight halves, without touching the ALU eight
// times.  An unsigned nibble n and the fp16 bit pattern 0x6400 (=1024.0) share
// no bits, so `(nibbles & 0x000F000F) | 0x64006400` IS the pair of halves
// {1024+n_lo, 1024+n_hi} with no arithmetic at all; one hsub2 then removes the
// bias and the zero point together.  Four masks, four ors, four subs for eight
// values, against roughly forty shift/and/convert/subtract ops the obvious way.
//
// The four results are the value pairs (0,4), (1,5), (2,6), (3,7) -- the shift
// amount picks the nibble, the two halves of the word pick the byte.
__device__ __forceinline__ void int4_8(uint32_t w, float z, __half2 out[4]) {
    const uint32_t M = 0x000F000Fu, E = 0x64006400u;
    const __half2 bias = __float2half2_rn(1024.f + z);
    uint32_t t;
    t = ((w      ) & M) | E; out[0] = __hsub2(*reinterpret_cast<__half2*>(&t), bias);
    t = ((w >>  4) & M) | E; out[1] = __hsub2(*reinterpret_cast<__half2*>(&t), bias);
    t = ((w >>  8) & M) | E; out[2] = __hsub2(*reinterpret_cast<__half2*>(&t), bias);
    t = ((w >> 12) & M) | E; out[3] = __hsub2(*reinterpret_cast<__half2*>(&t), bias);
}

// ---------------------------------------------------------------- timing
// %globaltimer is a device-wide nanosecond clock, unlike clock64() which is
// per-SM.  One read per grid barrier on one block is free, and it is what lets
// the compiler compare its predicted schedule against the real one.
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

// The shared-memory activation layout.  A 32-value group occupies 33 floats so
// that lane g of a warp reading group g hits 33 distinct banks -- conflict free
// for both the mxfp4 path (one group per lane) and the dense path (8 contiguous
// values per lane, never crossing a group because k0 % 8 == 0).
__device__ __host__ __forceinline__ constexpr int pad33(int k) { return (k >> 5) * 33 + (k & 31); }
__device__ __host__ __forceinline__ constexpr int pad33_size(int n) { return ((n + 31) / 32) * 33; }

} // namespace mk
