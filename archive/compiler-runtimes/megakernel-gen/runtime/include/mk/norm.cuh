#pragma once
// ============================================================================
// mk/norm.cuh -- normalisation, with the reference's rounding discipline.
//
// HuggingFace's RMSNorm is
//     x32 = x.float(); x32 *= rsqrt(mean(x32^2) + eps)
//     return weight * x32.to(input_dtype)
// which rounds TWICE: once when the normalised activation returns to bf16, and
// once on the product with the (bf16) weight.  Reproducing exactly that moved
// per-op error from bf16 epsilon to reduction-order noise, which is the
// difference between a correctness gate that works and one that drowns in
// false positives.  Gemma's variant is computed wholly in fp32 and rounds once,
// and multiplies by (1 + w); that is a different kind, not a different eps.
// ============================================================================
#include "common.cuh"

namespace mk {

enum : int { NORM_RMS = 0, NORM_RMS_1P = 1, NORM_LAYER = 2 };

// Normalise `x` (length N) and leave the result in the padded [G][33] shared
// layout, ready for a gemv.  Every block recomputes this from x in L2 rather
// than synchronising on one producer: N*4 bytes per block is far cheaper than
// a grid-wide barrier.
template<int N, class T, int NT, int KIND>
__device__ __forceinline__ void norm_to_smem(const T* __restrict__ x,
                                             const float* __restrict__ w,
                                             const float* __restrict__ b,
                                             float eps, float* xs, float* scr) {
    constexpr int VEC = 16 / sizeof(T);        // 16 B per thread per step
    constexpr int NV  = N / VEC;
    constexpr int CAP = (NV + NT - 1) / NT;    // vectors held per thread

    // The hidden state is read ONCE, into registers, and reused for both the
    // reduction and the scaled write-out.  Reading it twice -- the obvious way
    // -- costs a second full memory round trip on the critical path of every
    // block, and every block runs this: it measured ~5 us per stage on a
    // 2880-wide model, which is a quarter of the stage it prefixes.
    float4 v[CAP];
    float ss = 0.f;
    #pragma unroll
    for (int c = 0; c < CAP; ++c) {
        const int i = threadIdx.x + c * NT;
        if (CAP == 1 ? (i < NV) : (i < NV)) {
            v[c] = *reinterpret_cast<const float4*>(x + (size_t)i * VEC);
            const T* p = reinterpret_cast<const T*>(&v[c]);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) { const float t = to_f<T>(p[j]); ss = fmaf(t, t, ss); }
        }
    }
    ss = block_sum(ss, scr);
    const float inv = rsqrtf(ss / N + eps);
    #pragma unroll
    for (int c = 0; c < CAP; ++c) {
        const int i = threadIdx.x + c * NT;
        if (i < NV) {
            const T* p = reinterpret_cast<const T*>(&v[c]);
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                const int k = i * VEC + j;
                const float t = to_f<T>(p[j]);
                float o;
                if (KIND == NORM_RMS)          o = ActRound<T>::r(w[k] * ActRound<T>::r(t * inv));
                else if (KIND == NORM_RMS_1P)  o = ActRound<T>::r((t * inv) * (1.f + w[k]));
                else                           o = ActRound<T>::r(w[k] * (t * inv) + (b ? b[k] : 0.f));
                xs[pad33(k)] = o;
            }
        }
    }
    // zero the pad lane of each group so a vector load never reads garbage
    for (int g = threadIdx.x; g < (N + 31) / 32; g += NT) xs[g * 33 + 32] = 0.f;
    __syncthreads();
}

// Tail handling for a hidden size that is not a multiple of 16 bytes: rare, so
// it gets the simple two-pass form rather than complicating the fast path.
template<int N, class T, int NT, int KIND>
__device__ __forceinline__ void norm_to_smem_slow(const T* __restrict__ x,
                                                  const float* __restrict__ w,
                                                  const float* __restrict__ b,
                                                  float eps, float* xs, float* scr) {
    float mean = 0.f;
    if (KIND == NORM_LAYER) {
        float sm = 0.f;
        for (int i = threadIdx.x; i < N; i += NT) sm += to_f<T>(x[i]);
        mean = block_sum(sm, scr) / N;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += NT) {
        const float t = to_f<T>(x[i]) - mean;
        ss = fmaf(t, t, ss);
    }
    ss = block_sum(ss, scr);
    const float inv = rsqrtf(ss / N + eps);
    for (int i = threadIdx.x; i < N; i += NT) {
        const float t = to_f<T>(x[i]) - mean;
        float o;
        if (KIND == NORM_RMS)          o = ActRound<T>::r(w[i] * ActRound<T>::r(t * inv));
        else if (KIND == NORM_RMS_1P)  o = ActRound<T>::r((t * inv) * (1.f + w[i]));
        else                           o = ActRound<T>::r(w[i] * (t * inv) + (b ? b[i] : 0.f));
        xs[pad33(i)] = o;
    }
    for (int g = threadIdx.x; g < (N + 31) / 32; g += NT) xs[g * 33 + 32] = 0.f;
    __syncthreads();
}

// Copy an already-normalised fp32 vector into the padded shared layout.
template<int N, int NT>
__device__ __forceinline__ void vec_to_smem(const float* __restrict__ v, float* xs) {
    constexpr int NV = N / 4;
    #pragma unroll
    for (int c = 0; c < (NV + NT - 1) / NT; ++c) {
        const int i = threadIdx.x + c * NT;
        if (i < NV) {
            const float4 t = *reinterpret_cast<const float4*>(v + (size_t)i * 4);
            xs[pad33(i * 4 + 0)] = t.x; xs[pad33(i * 4 + 1)] = t.y;
            xs[pad33(i * 4 + 2)] = t.z; xs[pad33(i * 4 + 3)] = t.w;
        }
    }
    for (int i = NV * 4 + threadIdx.x; i < N; i += NT) xs[pad33(i)] = v[i];
    for (int g = threadIdx.x; g < (N + 31) / 32; g += NT) xs[g * 33 + 32] = 0.f;
    __syncthreads();
}

// Warp-local RMS norm of a head vector whose HD elements are spread over the
// 32 lanes as E = HD/32 registers.  Used for Qwen3-style q/k norms, where the
// warp that is about to consume the head already holds it -- so the norm costs
// one warp reduction and no extra pass over memory.
template<int HD, class T>
__device__ __forceinline__ void head_rmsnorm(float v[HD / 32],
                                             const float* __restrict__ w,
                                             float eps, int lane) {
    constexpr int E = HD / 32;
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < E; ++e) ss = fmaf(v[e], v[e], ss);
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / HD + eps);
    #pragma unroll
    for (int e = 0; e < E; ++e)
        v[e] = ActRound<T>::r(w[e * 32 + lane] * ActRound<T>::r(v[e] * inv));
}

} // namespace mk
