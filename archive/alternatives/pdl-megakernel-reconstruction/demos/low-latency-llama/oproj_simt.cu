#pragma once

#include "mlp_simt.cu"

#include <cuda_bf16.h>

// Pure-SIMT batch-1 O projection + residual controls.  Every launch is one
// H100 wave: either 128 equal CTAs or 132 balanced CTAs (one per SM).
namespace megakernel {
namespace oproj_simt {

using globals = llama_1b_globals;
using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

enum class input_cache_mode : int {
    direct = 0,
    synchronous = 1,
    cp_async = 2,
};

template <int GridCtas, int WarpsPerCta, int RowsPerWarp,
          input_cache_mode CacheMode>
struct config {
    static constexpr int ctas = GridCtas;
    static constexpr int warps = WarpsPerCta;
    static constexpr int rows_per_warp = RowsPerWarp;
    static constexpr int threads = warps * 32;
    static constexpr input_cache_mode cache_mode = CacheMode;
    static_assert(ctas == 128 || ctas == globals::sm_count);
    static_assert(warps * rows_per_warp == 16);
    static_assert(warps == 8 || warps == 16);
};

using oproj_128cta_16w =
    config<128, 16, 1, input_cache_mode::direct>;
using oproj_128cta_8w2r =
    config<128, 8, 2, input_cache_mode::direct>;
using oproj_132cta_16w =
    config<globals::sm_count, 16, 1, input_cache_mode::direct>;
using oproj_132cta_16w_sync =
    config<globals::sm_count, 16, 1, input_cache_mode::synchronous>;
using oproj_132cta_16w_cp_async =
    config<globals::sm_count, 16, 1, input_cache_mode::cp_async>;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ int instruction_layer(const globals &g) {
    return g.instructions.raw_ptr[1];
}

__device__ __forceinline__ void cp_async_16(void *shared_dst,
                                            const void *global_src) {
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" : :
                 "r"(shared_address), "l"(global_src));
}

template <typename Config>
__launch_bounds__(Config::threads, 1) __global__ void
oproj_residual(const __grid_constant__ globals g) {
    // The direct specialization is reduced to a tiny, unused allocation by
    // compile-time pruning.  Cached variants stage the 4 KiB activation once
    // per CTA; weights remain direct LDG because each element is consumed once.
    __shared__ __align__(16)
        bf16 input_cache[Config::cache_mode == input_cache_mode::direct
                             ? 1
                             : globals::hidden_dim];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const bf16 *input = g.attn_out.raw_ptr;

    if constexpr (Config::cache_mode == input_cache_mode::synchronous) {
        constexpr int input_quads = globals::hidden_dim / 4;
        for (int quad = tid; quad < input_quads; quad += Config::threads) {
            reinterpret_cast<unsigned long long *>(input_cache)[quad] =
                reinterpret_cast<const unsigned long long *>(input)[quad];
        }
        __syncthreads();
        input = input_cache;
    } else if constexpr (Config::cache_mode == input_cache_mode::cp_async) {
        constexpr int input_chunks =
            globals::hidden_dim * sizeof(bf16) / 16;
        for (int chunk = tid; chunk < input_chunks;
             chunk += Config::threads) {
            cp_async_16(reinterpret_cast<char *>(input_cache) + chunk * 16,
                        reinterpret_cast<const char *>(input) + chunk * 16);
        }
        asm volatile("cp.async.commit_group;" : :);
        asm volatile("cp.async.wait_group 0;" : :);
        __syncthreads();
        input = input_cache;
    }

    const int block = static_cast<int>(blockIdx.x);
    int row_begin;
    int row_count;
    if constexpr (Config::ctas == globals::sm_count) {
        // 2048 rows = 68 CTAs * 16 rows + 64 CTAs * 15 rows.
        row_begin = block * 15 + (block < 68 ? block : 68);
        row_count = 15 + (block < 68);
    } else {
        row_begin = block * 16;
        row_count = 16;
    }

    constexpr int input_pairs = globals::hidden_dim / 2;
    const bf16 *weight_layer =
        g.o_weights.raw_ptr + layer * globals::hidden_dim * globals::hidden_dim;
    float sums[Config::rows_per_warp] = {};

    for (int pair = lane; pair < input_pairs; pair += 32) {
        const float2 x = mlp_simt::load_bf16x2(input, pair);
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
            const int local_row = warp + row_iter * Config::warps;
            if (local_row < row_count) {
                const int row = row_begin + local_row;
                mlp_simt::accumulate_pair(
                    x,
                    mlp_simt::load_bf16x2(
                        weight_layer + row * globals::hidden_dim, pair),
                    sums[row_iter]);
            }
        }
    }

#pragma unroll
    for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
        const int local_row = warp + row_iter * Config::warps;
        float sum = warp_sum(sums[row_iter]);
        if (lane == 0 && local_row < row_count) {
            const int row = row_begin + local_row;
            const float residual = __bfloat162float(g.hidden_states.raw_ptr[row]);
            g.hidden_states.raw_ptr[row] = __float2bfloat16_rn(residual + sum);
        }
    }
}

// Cold-weight vector-load family.  The 128-CTA mappings have no tail, so the
// row addresses and two-row ownership are fully compile-time regular.  b64
// loads halve loop/address instructions while preserving pure SIMT FP32 dots.
template <typename Config, int WeightPolicy, bool PdlTrigger = false,
          bool PdlWait = false, int PdlTriggerQuad = -1,
          int PdlPrefetchLines = 0,
          int PdlPrefetchPolicy = mlp_simt::prefetch_policy_l2>
__launch_bounds__(Config::threads, 1) __global__ void
oproj_residual_v4(const __grid_constant__ globals g) {
    static_assert(Config::ctas == 128);
    static_assert(Config::cache_mode == input_cache_mode::direct);
    static_assert(PdlTriggerQuad == -1 || PdlTriggerQuad == 256 ||
                  PdlTriggerQuad == 384 || PdlTriggerQuad == 448 ||
                  PdlTriggerQuad == 480 || PdlTriggerQuad == 512);
    static_assert(PdlPrefetchLines == 0 || PdlPrefetchLines == 1 ||
                  PdlPrefetchLines == 2 || PdlPrefetchLines == 4 ||
                  PdlPrefetchLines == 8 || PdlPrefetchLines == 16 ||
                  PdlPrefetchLines == 32);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int layer = instruction_layer(g);
    const int row_begin = static_cast<int>(blockIdx.x) * 16;
    const bf16 *input = g.attn_out.raw_ptr;
    const bf16 *weight_layer =
        g.o_weights.raw_ptr + layer * globals::hidden_dim * globals::hidden_dim;
    float sums[Config::rows_per_warp] = {};
    constexpr int input_quads = globals::hidden_dim / 4;

    if constexpr (PdlWait && PdlPrefetchLines > 0) {
        constexpr int values_per_l2_line = 128 / sizeof(bf16);
        const int row = row_begin + warp;
        const bf16 *weight_row =
            weight_layer + row * globals::hidden_dim;
#pragma unroll
        for (int line = lane; line < PdlPrefetchLines; line += 32) {
            mlp_simt::prefetch_global<PdlPrefetchPolicy>(
                weight_row + line * values_per_l2_line);
        }
    }
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger && PdlTriggerQuad < 0) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    for (int quad = lane; quad < input_quads; quad += 32) {
        if constexpr (PdlTrigger && PdlTriggerQuad >= 0 &&
                      PdlTriggerQuad < input_quads) {
            if (threadIdx.x == 0 && quad == PdlTriggerQuad) {
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }
        const mlp_simt::float4_split x =
            mlp_simt::load_bf16x4(input, quad);
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
            const int row = row_begin + warp + row_iter * Config::warps;
            mlp_simt::accumulate_quad(
                x,
                mlp_simt::load_weight_bf16x4<WeightPolicy>(
                    weight_layer + row * globals::hidden_dim, quad),
                sums[row_iter]);
        }
    }
    if constexpr (PdlTrigger && PdlTriggerQuad == input_quads) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

#pragma unroll
    for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
        const float sum = warp_sum(sums[row_iter]);
        if (lane == 0) {
            const int row = row_begin + warp + row_iter * Config::warps;
            const float residual =
                __bfloat162float(g.hidden_states.raw_ptr[row]);
            g.hidden_states.raw_ptr[row] =
                __float2bfloat16_rn(residual + sum);
        }
    }
}

} // namespace oproj_simt
} // namespace megakernel
