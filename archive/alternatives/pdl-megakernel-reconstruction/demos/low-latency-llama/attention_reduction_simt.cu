#pragma once

#include "llama.cuh"

#include <cuda_bf16.h>

// Matched two-partial attention reduction.  This opcode is dormant in the
// production context-1 schedule; the kernel deliberately matches the same
// synthetic two-partial contract used to establish the extracted-TK target.
namespace megakernel {
namespace attention_reduction_simt {

using globals = llama_1b_globals;
using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

template <int GridCtas, int WarpsPerCta>
struct config {
    static constexpr int ctas = GridCtas;
    static constexpr int warps = WarpsPerCta;
    static constexpr int threads = warps * 32;
    static_assert(ctas * warps == globals::num_attention_heads);
    static_assert(ctas == 1 || ctas == 4 || ctas == 8 || ctas == 32);
};

using reduction_1cta_32w = config<1, 32>;
using reduction_4cta_8w = config<4, 8>;
using reduction_8cta_4w = config<8, 4>;
using reduction_32cta_1w = config<32, 1>;

template <typename Config>
__launch_bounds__(Config::threads, 1) __global__ void
two_partial_reduction(const __grid_constant__ globals g) {
    static_assert(globals::head_dim == 64);
    static_assert(globals::num_attention_heads == 32);

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int q_head = static_cast<int>(blockIdx.x) * Config::warps + warp;
    const int lse_stride = static_cast<int>(g.attn_lse_intermediates.cols());
    const int partial_stride = globals::head_dim;
    const int head_stride =
        static_cast<int>(g.attn_out_intermediates.rows()) * partial_stride;

    float factor0 = 0.0f;
    float factor1 = 0.0f;
    if (lane == 0) {
        const float lse0 =
            g.attn_lse_intermediates.raw_ptr[q_head * lse_stride];
        const float lse1 =
            g.attn_lse_intermediates.raw_ptr[q_head * lse_stride + 1];
        const float max_lse = fmaxf(lse0, lse1);
        const float exp0 = exp2f(lse0 - max_lse);
        const float exp1 = exp2f(lse1 - max_lse);
        const float inv_sum = __frcp_rn(exp0 + exp1);
        factor0 = exp0 * inv_sum;
        factor1 = exp1 * inv_sum;
    }
    factor0 = __shfl_sync(0xffffffffu, factor0, 0);
    factor1 = __shfl_sync(0xffffffffu, factor1, 0);

    const float *head_base =
        g.attn_out_intermediates.raw_ptr + q_head * head_stride;
    const float2 out0 = reinterpret_cast<const float2 *>(head_base)[lane];
    const float2 out1 = reinterpret_cast<const float2 *>(
        head_base + partial_stride)[lane];
    const float reduced0 = fmaf(out0.x, factor0, out1.x * factor1);
    const float reduced1 = fmaf(out0.y, factor0, out1.y * factor1);
    reinterpret_cast<bf16x2 *>(
        g.attn_out.raw_ptr + q_head * globals::head_dim)[lane] =
        __floats2bfloat162_rn(reduced0, reduced1);
}

} // namespace attention_reduction_simt
} // namespace megakernel

