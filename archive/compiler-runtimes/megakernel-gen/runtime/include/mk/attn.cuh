#pragma once
// ============================================================================
// mk/attn.cuh -- the parts of single-query attention that are model-independent.
//
// The attention LOOP itself is generated (codegen.rs, T_ATTN): it fuses the
// q/k norm, the rotation and the write of K[pos] into the warp that consumes
// them, and which of those exist is a property of the model.  What is left here
// is the part that is the same for every model and worth unit-testing on its
// own: the logit softcap and the cross-split softmax reduction.
//
// KV cache layout is [kv_head][pos][head_dim]: a decode step reads one head's
// whole history, and this makes that read one contiguous run.
// ============================================================================
#include "common.cuh"

namespace mk {

struct AttnParams {
    int   slen;          // keys that exist (pos + 1)
    int   pos;           // current position
    int   window;        // sliding window span, 0 = full attention
    float scale;         // 1/sqrt(head_dim), or the model's own
    float softcap;       // 0 = off
    int   max_ctx;       // stride of the kv cache
};

__device__ __forceinline__ float apply_softcap(float s, float cap) {
    return cap * tanhf(s / cap);
}

// ---------------------------------------------------------------- reduce
// Combine the per-split partial softmaxes.
//
// The parallelism has to come from the INPUT.  The output is only
// n_heads*head_dim floats, so a warp-per-output-chunk decomposition has
// n_heads*head_dim/32 work items -- 128 of them on an 8B model -- while the
// input it must read is n_heads*SPLITS*head_dim.  At 128 key splits that left
// 3% of the resident warps with anything to do and the stage cost more than the
// attention it was reducing.
//
// So a whole BLOCK takes one (head, 32-lane chunk) and its warps divide the
// SPLIT axis between them.  The trick that makes the combine cheap is to
// compute the softmax normaliser first: (M, L) depend only on `pml`, which is
// 2*SPLITS floats and is read by every warp as a broadcast, so once they are
// known each warp's contribution is a plain weighted SUM and the warps combine
// through shared memory -- a __syncthreads, not a grid barrier.
//
// The combine sums the per-warp partials in warp order, not with atomicAdd.
// Float addition is not associative, so an atomic combine makes the kernel's
// output depend on the order the warps happen to arrive in: the guard caught
// exactly that, as a weight that did not restore bit-exactly after being
// zeroed.  Determinism is a promise this compiler makes.
//
// An attention sink is a learned logit with no value vector: it enters the
// denominator only.
//
// `scr` needs 32 floats per warp plus two, and gets the activation arena:
// between the attention stage that produced `psum` and the output projection
// that consumes `out`, nothing else is live in it.
template<class T, int HD, int SPLITS, bool SINKS>
__device__ __forceinline__ void attn_reduce(const float* __restrict__ pml,
                                            const float* __restrict__ psum,
                                            const float* __restrict__ sinks,
                                            float* __restrict__ out,
                                            int n_heads, float* scr) {
    constexpr int CH = HD / 32;
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5, wid = threadIdx.x >> 5;
    const int nitem = n_heads * CH;
    for (int it = blockIdx.x; it < nitem; it += gridDim.x) {
        const int h = it / CH, c = it - (it / CH) * CH;
        const int d = c * 32 + lane;
        const float* pm = pml + (size_t)2 * h * SPLITS;

        // The normaliser is a property of the HEAD, not of the output element,
        // so one warp computes it for the whole block and publishes it.  Every
        // warp computing it independently is 2*SPLITS scalar loads and SPLITS
        // exponentials each, thirty-two times over, to get the same two floats.
        //
        // Branch-free.  An empty split carries (-inf, 0) and an all-zero psum,
        // so exp(-inf - M) = 0 makes it contribute nothing; skipping it with a
        // data-dependent `continue` instead would serialise the split loop and
        // was measured at a sixth of the achievable bandwidth.
        if (wid == 0) {
            float M = -INFINITY;
            #pragma unroll 8
            for (int s = lane; s < SPLITS; s += 32) M = fmaxf(M, pm[2 * s]);
            M = warp_max(M);
            float L = 0.f;
            if (SINKS) { const float sk = sinks[h]; M = fmaxf(M, sk); }
            if (M > -INFINITY) {
                #pragma unroll 8
                for (int s = lane; s < SPLITS; s += 32)
                    L = fmaf(pm[2 * s + 1], __expf(pm[2 * s] - M), L);
                L = warp_sum(L);
                if (SINKS) L += __expf(sinks[h] - M);
            }
            if (lane == 0) { scr[nw * 32] = M; scr[nw * 32 + 1] = L; }
        }
        __syncthreads();
        const float M = scr[nw * 32], L = scr[nw * 32 + 1];
        const bool empty = (M == -INFINITY);

        float acc = 0.f;
        if (!empty) {
            for (int s = wid; s < SPLITS; s += nw)
                acc = fmaf(psum[(size_t)(h * SPLITS + s) * HD + d], __expf(pm[2 * s] - M), acc);
        }
        scr[wid * 32 + lane] = acc;      // lane picks the bank: conflict free
        __syncthreads();
        if (wid == 0) {
            float t = 0.f;
            for (int w = 0; w < nw; ++w) t += scr[w * 32 + lane];
            out[h * HD + d] = ActRound<T>::r(empty ? 0.f : t / L);
        }
        __syncthreads();
    }
}

} // namespace mk
