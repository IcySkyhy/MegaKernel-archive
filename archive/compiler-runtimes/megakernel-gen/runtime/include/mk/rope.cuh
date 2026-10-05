#pragma once
// ============================================================================
// mk/rope.cuh -- rotary embedding, applied in registers.
//
// The cos/sin tables are built once on the host, in fp32, then stored in the
// activation dtype -- transformers returns `cos.to(x.dtype)`, so a bf16 model
// rotates with bf16 trig and matching that keeps the gate meaningful.
//
// Layout: `HALVES` (HF "neox") pairs element d with d + RD/2.  A warp holding
// element e*32+lane in register e therefore finds its partner in register
// e + E/2 of the SAME lane, so the rotation needs no shuffle at all -- provided
// RD is a multiple of 64, which is checked at compile time.
// ============================================================================
#include "common.cuh"

namespace mk {

template<int HD, int RD, class T>
__device__ __forceinline__ void rope_warp(float v[HD / 32],
                                          const T* __restrict__ cos_row,
                                          const T* __restrict__ sin_row,
                                          int lane) {
    constexpr int E = HD / 32;
    constexpr int ER = RD / 32;
    static_assert(RD % 64 == 0, "rotary dim must be a multiple of 64 for the register-local rotation");
    static_assert(RD <= HD, "rotary dim exceeds head dim");
    #pragma unroll
    for (int e = 0; e < ER / 2; ++e) {
        const int d = e * 32 + lane;                 // < RD/2
        const float c = to_f<T>(cos_row[d]);
        const float s = to_f<T>(sin_row[d]);
        const float lo = v[e], hi = v[e + ER / 2];
        v[e]          = ActRound<T>::r(ActRound<T>::r(lo * c) - ActRound<T>::r(hi * s));
        v[e + ER / 2] = ActRound<T>::r(ActRound<T>::r(hi * c) + ActRound<T>::r(lo * s));
    }
    (void)E;
}

// Head dims that are not a multiple of 64 (Phi-3's 96, for instance) pair
// element d with d + RD/2 in a DIFFERENT lane, and the register index of the
// partner is lane-dependent -- which in CUDA means indexed register access,
// which means local memory.  Route those through shared memory instead: one
// head vector per warp, a __syncwarp, and the same arithmetic.
template<int HD, int RD, class T>
__device__ __forceinline__ void rope_warp_smem(float v[HD / 32],
                                               const T* __restrict__ cos_row,
                                               const T* __restrict__ sin_row,
                                               int lane, float* __restrict__ sh) {
    constexpr int E = HD / 32;
    constexpr int H2 = RD / 2;
    #pragma unroll
    for (int e = 0; e < E; ++e) sh[e * 32 + lane] = v[e];
    __syncwarp();
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        const int d = e * 32 + lane;
        if (d < RD) {
            const int lo = (d < H2) ? d : d - H2;
            const float c = to_f<T>(cos_row[lo]);
            const float s = to_f<T>(sin_row[lo]);
            const float self = sh[d];
            const float other = (d < H2) ? sh[d + H2] : sh[d - H2];
            v[e] = (d < H2)
                 ? ActRound<T>::r(ActRound<T>::r(self * c) - ActRound<T>::r(other * s))
                 : ActRound<T>::r(ActRound<T>::r(self * c) + ActRound<T>::r(other * s));
        }
    }
    __syncwarp();
}

// GPT-J style interleaved rotation: pairs are (2i, 2i+1).  Both members live in
// the same 32-lane row only when consecutive elements share a lane, which they
// do not in this layout, so it is done with one shuffle per pair.
template<int HD, int RD, class T>
__device__ __forceinline__ void rope_warp_interleaved(float v[HD / 32],
                                                      const T* __restrict__ cos_row,
                                                      const T* __restrict__ sin_row,
                                                      int lane) {
    constexpr int ER = RD / 32;
    #pragma unroll
    for (int e = 0; e < ER; ++e) {
        const int d = e * 32 + lane;
        const int pair = d >> 1;
        const float c = to_f<T>(cos_row[pair]);
        const float s = to_f<T>(sin_row[pair]);
        const float other = __shfl_xor_sync(0xffffffffu, v[e], 1);
        const float lo = (d & 1) ? other : v[e];
        const float hi = (d & 1) ? v[e] : other;
        const float r = (d & 1) ? ActRound<T>::r(ActRound<T>::r(hi * c) + ActRound<T>::r(lo * s))
                                : ActRound<T>::r(ActRound<T>::r(lo * c) - ActRound<T>::r(hi * s));
        v[e] = r;
    }
}

} // namespace mk
