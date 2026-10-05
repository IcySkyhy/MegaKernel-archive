#pragma once
// ============================================================================
// mk/moe.cuh -- router top-k.
//
// Recomputed per block from the expert logits already in L2 (a few hundred
// bytes) rather than given its own stage and grid barrier.  The selection runs
// on ONE warp with shuffle reductions: the naive single-thread scan issues
// n_experts serially dependent L2 loads and measured 51 us on gpt-oss, a third
// of the whole layer, for 0.7 MB of work.
//
// Ties break towards the lower expert index, matching torch.topk.
// ============================================================================
#include "common.cuh"

namespace mk {

enum : int { SCORE_SOFTMAX = 0, SCORE_SIGMOID = 1 };

// NE experts, TOPK selected.  `gate` holds NE fp32 logits.
// Writes the chosen indices and their (normalised) weights to shared memory.
template<int NE, int TOPK, int SCORE, bool SOFTMAX_AFTER_TOPK, bool NORM_TOPK>
__device__ __forceinline__ void topk_warp(const float* __restrict__ gate,
                                          const float* __restrict__ corr,
                                          int* __restrict__ idx,
                                          float* __restrict__ wgt,
                                          float routed_scale) {
    if (threadIdx.x < 32) {
        constexpr int PER = (NE + 31) / 32;
        const int lane = threadIdx.x;
        float v[PER]; int id[PER];
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            id[i] = lane + i * 32;
            float g = (id[i] < NE) ? gate[id[i]] : -INFINITY;
            // DeepSeek-style selection bias affects the choice, not the weight
            g = (corr && id[i] < NE) ? g + corr[id[i]] : g;
            // A NaN loses every comparison below, so a NaN-poisoned gate would
            // leave the sentinel index in place and index the expert table out
            // of bounds.  Map it to -inf and it simply never wins.
            v[i] = isnan(g) ? -INFINITY : g;
        }
        float bv[TOPK]; int bi[TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float lb = -INFINITY; int li = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < PER; ++i)
                if (v[i] > lb || (v[i] == lb && id[i] < li)) { lb = v[i]; li = id[i]; }
            #pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, lb, o);
                const int   oi = __shfl_xor_sync(0xffffffffu, li, o);
                if (ov > lb || (ov == lb && oi < li)) { lb = ov; li = oi; }
            }
            if (li >= NE) li = 0;      // cannot happen once NaNs are mapped out
            bv[k] = lb; bi[k] = li;
            #pragma unroll
            for (int i = 0; i < PER; ++i) if (id[i] == li) v[i] = -INFINITY;
        }
        if (lane == 0) {
            float w[TOPK];
            if (SOFTMAX_AFTER_TOPK) {
                // gpt-oss: softmax over the selected logits only
                float mx = -INFINITY;
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) mx = fmaxf(mx, gate[bi[k]]);
                float sum = 0.f;
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) { w[k] = __expf(gate[bi[k]] - mx); sum += w[k]; }
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) w[k] /= sum;
            } else if (SCORE == SCORE_SOFTMAX) {
                // softmax over ALL experts, then take the selected entries
                float mx = -INFINITY;
                for (int e = 0; e < NE; ++e) mx = fmaxf(mx, gate[e]);
                float sum = 0.f;
                for (int e = 0; e < NE; ++e) sum += __expf(gate[e] - mx);
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) w[k] = __expf(gate[bi[k]] - mx) / sum;
            } else {
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) w[k] = 1.f / (1.f + __expf(-gate[bi[k]]));
            }
            if (NORM_TOPK && !SOFTMAX_AFTER_TOPK) {
                float s = 0.f;
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) s += w[k];
                const float inv = 1.f / (s + 1e-20f);
                #pragma unroll
                for (int k = 0; k < TOPK; ++k) w[k] *= inv;
            }
            #pragma unroll
            for (int k = 0; k < TOPK; ++k) { idx[k] = bi[k]; wgt[k] = w[k] * routed_scale; }
        }
    }
    __syncthreads();
}

} // namespace mk
