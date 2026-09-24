#pragma once

#include "llama.cuh"

#include <cuda_bf16.h>

namespace megakernel {
namespace qkv_simt {

using globals = llama_1b_globals;
using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

constexpr int q_rows = globals::num_attention_heads * globals::head_dim;
constexpr int kv_rows = globals::num_kv_heads * globals::head_dim;
constexpr int qkv_rows = q_rows + 2 * kv_rows;
constexpr int qkv_pairs = qkv_rows / 2;
constexpr int grid_ctas = globals::sm_count;
constexpr int kv_only_grid_ctas = globals::sm_count;
constexpr int pair_units_per_cta = 12;

static_assert(q_rows == 2048);
static_assert(kv_rows == 512);
static_assert(qkv_rows == 3072);
static_assert(qkv_pairs == 1536);
static_assert(qkv_pairs == 84 * 12 + (grid_ctas - 84) * 11);
static_assert(kv_rows == 116 * 4 + (kv_only_grid_ctas - 116) * 3);

template <int WarpsPerPair, int WeightPolicy,
          int PairUnits = pair_units_per_cta>
struct qkv_config {
    static_assert(WarpsPerPair == 1 || WarpsPerPair == 2);
    static_assert(PairUnits > 0 && PairUnits <= pair_units_per_cta);
    static constexpr int warps_per_pair = WarpsPerPair;
    static constexpr int weight_policy = WeightPolicy;
    static constexpr int pair_units = PairUnits;
    static constexpr int warps = pair_units * warps_per_pair;
    static constexpr int threads = warps * 32;
};

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

template <bool WriteContext1Attention = false>
__device__ __forceinline__ void
store_qkv_pair(const globals &g, int layer, int row0, float value0,
               float value1) {
    if (row0 < q_rows + kv_rows) {
        const int dim = row0 % globals::head_dim;
        const int rope_offset = static_cast<int>(g.pos_id) *
                                    globals::head_dim +
                                dim;
        const float cos0 = g.rope_cos.raw_ptr[rope_offset];
        const float cos1 = g.rope_cos.raw_ptr[rope_offset + 1];
        const float sin0 = g.rope_sin.raw_ptr[rope_offset];
        const float sin1 = g.rope_sin.raw_ptr[rope_offset + 1];
        const float original0 = value0;
        value0 = original0 * cos0 - value1 * sin0;
        value1 = value1 * cos1 + original0 * sin1;
    }

    const bf16x2 packed = __floats2bfloat162_rn(value0, value1);
    if (row0 < q_rows) {
        reinterpret_cast<bf16x2 *>(g.q_post_rope.raw_ptr)[row0 / 2] = packed;
        return;
    }

    const bool is_key = row0 < q_rows + kv_rows;
    const int local_row = row0 - (is_key ? q_rows : q_rows + kv_rows);
    const int head = local_row / globals::head_dim;
    const int dim = local_row % globals::head_dim;
    const int cache_offset =
        ((layer * static_cast<int>(g.k_cache.depth()) +
          static_cast<int>(g.pos_id)) *
             static_cast<int>(g.k_cache.rows()) +
         head) *
            globals::head_dim +
        dim;
    bf16 *cache = is_key ? g.k_cache.raw_ptr : g.v_cache.raw_ptr;
    reinterpret_cast<bf16x2 *>(cache + cache_offset)[0] = packed;
    if constexpr (WriteContext1Attention) {
        if (!is_key) {
            constexpr int q_heads_per_kv =
                globals::num_attention_heads / globals::num_kv_heads;
#pragma unroll
            for (int group = 0; group < q_heads_per_kv; ++group) {
                const int q_head = head * q_heads_per_kv + group;
                reinterpret_cast<bf16x2 *>(
                    g.attn_out.raw_ptr + q_head * globals::head_dim + dim)[0] =
                    packed;
            }
        }
    }
}

template <typename Config, bool PdlWait = false, bool PdlTrigger = false,
          int PrefetchLines = 0, int PdlTriggerPair = -1,
          bool SignalVReady = false,
          bool WriteContext1Attention = false,
          bool KvOnly = false,
          int PdlPrefetchPolicy = mlp_simt::prefetch_policy_l2,
          bool SignalQReady = false,
          int PdlTriggerTailIterations = 0>
__launch_bounds__(Config::threads, 1) __global__ void
qkv_rope_append(const __grid_constant__ globals g) {
    static_assert(PrefetchLines == 0 || PrefetchLines == 1 ||
                  PrefetchLines == 2 || PrefetchLines == 4 ||
                  PrefetchLines == 8);
    static_assert(PdlTriggerPair == -1 || PdlTriggerPair == 0 ||
                  PdlTriggerPair == 512 || PdlTriggerPair == 768 ||
                  PdlTriggerPair == 896 || PdlTriggerPair == 960 ||
                  PdlTriggerPair == 1024);
    static_assert(PdlTriggerTailIterations == 0 ||
                  PdlTriggerTailIterations == 1 ||
                  PdlTriggerTailIterations == 2 ||
                  PdlTriggerTailIterations == 4);
    static_assert(PdlTriggerTailIterations == 0 || PdlTrigger);
    static_assert(PdlTriggerTailIterations == 0 || PdlTriggerPair == -1);
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];
    __shared__ float pair_partials[Config::pair_units]
                                  [Config::warps_per_pair][2];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.attn_norm_weights.raw_ptr + layer * globals::hidden_dim;
    constexpr int hidden_pairs = globals::hidden_dim / 2;

    if constexpr (PdlWait && PrefetchLines > 0) {
        const int block = static_cast<int>(blockIdx.x);
        constexpr int base_pairs = KvOnly ? 3 : 11;
        constexpr int large_ctas = KvOnly ? 116 : 84;
        constexpr int row_pair_offset = KvOnly ? q_rows / 2 : 0;
        const int pair_begin =
            block * base_pairs + (block < large_ctas ? block : large_ctas);
        const int pair_count = base_pairs + (block < large_ctas);
        const int unit = warp / Config::warps_per_pair;
        const int reduction_part = warp % Config::warps_per_pair;
        if (unit < pair_count && reduction_part == 0 &&
            lane < PrefetchLines) {
            constexpr int values_per_l2_line = 128 / sizeof(bf16);
            const int row0 = 2 * (row_pair_offset + pair_begin + unit);
            const bf16 *weight_row0 =
                g.qkv_weights.raw_ptr +
                layer * qkv_rows * globals::hidden_dim +
                row0 * globals::hidden_dim;
            mlp_simt::prefetch_global<PdlPrefetchPolicy>(
                weight_row0 + lane * values_per_l2_line);
            mlp_simt::prefetch_global<PdlPrefetchPolicy>(
                weight_row0 + globals::hidden_dim +
                lane * values_per_l2_line);
        }
    }
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger && PdlTriggerPair < 0) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += Config::threads) {
        const float2 value = mlp_simt::load_bf16x2(hidden, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < Config::warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(
                block_sum / static_cast<float>(globals::hidden_dim) +
                g.rms_norm_eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int pair = tid; pair < hidden_pairs; pair += Config::threads) {
        const float2 value = mlp_simt::load_bf16x2(hidden, pair);
        const float2 scale = mlp_simt::load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    // 1536 pair units = 84 CTAs * 12 + 48 CTAs * 11.
    const int block = static_cast<int>(blockIdx.x);
    constexpr int base_pairs = KvOnly ? 3 : 11;
    constexpr int large_ctas = KvOnly ? 116 : 84;
    constexpr int row_pair_offset = KvOnly ? q_rows / 2 : 0;
    const int pair_begin =
        block * base_pairs + (block < large_ctas ? block : large_ctas);
    const int pair_count = base_pairs + (block < large_ctas);
    const int unit = warp / Config::warps_per_pair;
    const int reduction_part = warp % Config::warps_per_pair;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    int row0 = 0;
    if constexpr (PdlTriggerTailIterations > 0) {
        constexpr int pair_stride = 32 * Config::warps_per_pair;
        constexpr int iterations = hidden_pairs / pair_stride;
        static_assert(hidden_pairs % pair_stride == 0);
        static_assert(PdlTriggerTailIterations < iterations);
        constexpr int main_iterations =
            iterations - PdlTriggerTailIterations;
        float2 tail_x[PdlTriggerTailIterations];
        float2 tail_weight0[PdlTriggerTailIterations];
        float2 tail_weight1[PdlTriggerTailIterations];

        if (unit < pair_count) {
            row0 = 2 * (row_pair_offset + pair_begin + unit);
            const bf16 *weight_layer =
                g.qkv_weights.raw_ptr +
                layer * qkv_rows * globals::hidden_dim;
            const bf16 *weight_row0 =
                weight_layer + row0 * globals::hidden_dim;
            const bf16 *weight_row1 =
                weight_row0 + globals::hidden_dim;

#pragma unroll
            for (int iteration = 0; iteration < main_iterations;
                 ++iteration) {
                const int pair = lane + reduction_part * 32 +
                                 iteration * pair_stride;
                const float2 x = mlp_simt::load_bf16x2(normalized, pair);
                mlp_simt::accumulate_pair(
                    x,
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row0, pair),
                    sum0);
                mlp_simt::accumulate_pair(
                    x,
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row1, pair),
                    sum1);
            }

            // Retain the final weight loads in registers.  The CTA barrier
            // ensures every warp has issued its tail LDGs before thread 0
            // publishes programmatic launch completion.  The dependent
            // attention grid can then prefetch historical KV while this CTA
            // consumes the retained values and runs reduction/RoPE/stores.
#pragma unroll
            for (int tail = 0; tail < PdlTriggerTailIterations; ++tail) {
                const int pair = lane + reduction_part * 32 +
                                 (main_iterations + tail) * pair_stride;
                tail_x[tail] = mlp_simt::load_bf16x2(normalized, pair);
                tail_weight0[tail] =
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row0, pair);
                tail_weight1[tail] =
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row1, pair);
            }
        }

        __syncthreads();
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }

        if (unit < pair_count) {
#pragma unroll
            for (int tail = 0; tail < PdlTriggerTailIterations; ++tail) {
                mlp_simt::accumulate_pair(tail_x[tail], tail_weight0[tail],
                                          sum0);
                mlp_simt::accumulate_pair(tail_x[tail], tail_weight1[tail],
                                          sum1);
            }
            sum0 = warp_sum(sum0);
            sum1 = warp_sum(sum1);
        }
    } else if (unit < pair_count) {
        row0 = 2 * (row_pair_offset + pair_begin + unit);
        const bf16 *weight_layer =
            g.qkv_weights.raw_ptr + layer * qkv_rows * globals::hidden_dim;
        const bf16 *weight_row0 =
            weight_layer + row0 * globals::hidden_dim;
        const bf16 *weight_row1 = weight_row0 + globals::hidden_dim;
        {
            for (int pair = lane + reduction_part * 32;
                 pair < hidden_pairs;
                 pair += 32 * Config::warps_per_pair) {
                if constexpr (PdlTrigger && PdlTriggerPair >= 0 &&
                              PdlTriggerPair < hidden_pairs) {
                    if (tid == 0 && pair == PdlTriggerPair) {
                        cudaTriggerProgrammaticLaunchCompletion();
                    }
                }
                const float2 x = mlp_simt::load_bf16x2(normalized, pair);
                mlp_simt::accumulate_pair(
                    x,
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row0, pair),
                    sum0);
                mlp_simt::accumulate_pair(
                    x,
                    mlp_simt::load_weight_bf16x2<Config::weight_policy>(
                        weight_row1, pair),
                    sum1);
            }
        }
        sum0 = warp_sum(sum0);
        sum1 = warp_sum(sum1);
    }

    if constexpr (Config::warps_per_pair == 1) {
        if (unit < pair_count && lane == 0) {
            store_qkv_pair<WriteContext1Attention>(g, layer, row0, sum0,
                                                   sum1);
        }
    } else {
        if (unit < pair_count && lane == 0) {
            pair_partials[unit][reduction_part][0] = sum0;
            pair_partials[unit][reduction_part][1] = sum1;
        }
        __syncthreads();
        if (unit < pair_count && reduction_part == 0 && lane == 0) {
            sum0 = pair_partials[unit][0][0] + pair_partials[unit][1][0];
            sum1 = pair_partials[unit][0][1] + pair_partials[unit][1][1];
            store_qkv_pair<WriteContext1Attention>(g, layer, row0, sum0,
                                                   sum1);
        }
    }
    if constexpr (PdlTrigger && PdlTriggerPair == hidden_pairs) {
        __syncthreads();
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    if constexpr (SignalVReady) {
        // Context-1 attention consumes only V.  Publish each 64-element V
        // head after all 32 BF16x2 pair units for that head are visible,
        // without making the consumer wait for unrelated Q/K rows.
        constexpr int v_pair_begin = (q_rows + kv_rows) / 2;
        constexpr int pairs_per_head = globals::head_dim / 2;
        constexpr int v_pair_end = v_pair_begin +
                                   globals::num_kv_heads * pairs_per_head;
        if (pair_begin + pair_count > v_pair_begin) {
            __syncthreads();
            if (tid == 0) {
                constexpr int v_flag_offset = 4096;
                const int overlap_begin = max(pair_begin, v_pair_begin);
                const int overlap_end = min(pair_begin + pair_count,
                                            v_pair_end);
                const int first_head =
                    (overlap_begin - v_pair_begin) / pairs_per_head;
                const int first_head_end =
                    v_pair_begin + (first_head + 1) * pairs_per_head;
                const int first_count =
                    min(overlap_end, first_head_end) - overlap_begin;
                __threadfence();
                atomicAdd(g.Bar.raw_ptr + v_flag_offset +
                              layer * globals::num_kv_heads + first_head,
                          static_cast<unsigned>(first_count));
                if (overlap_end > first_head_end) {
                    atomicAdd(g.Bar.raw_ptr + v_flag_offset +
                                  layer * globals::num_kv_heads +
                                  first_head + 1,
                              static_cast<unsigned>(overlap_end -
                                                    first_head_end));
                }
            }
        }
    }
    if constexpr (SignalQReady) {
        // Publish each complete 64-element Q head independently.  The
        // short-context attention kernel can then consume Q and historical
        // KV while this grid is still producing current-token K/V rows.
        constexpr int pairs_per_head = globals::head_dim / 2;
        constexpr int q_pair_end = q_rows / 2;
        if (pair_begin < q_pair_end) {
            __syncthreads();
            if (tid == 0) {
                constexpr int q_flag_offset = 4096;
                const int overlap_begin = pair_begin;
                const int overlap_end = min(pair_begin + pair_count,
                                            q_pair_end);
                const int first_head = overlap_begin / pairs_per_head;
                const int first_head_end =
                    (first_head + 1) * pairs_per_head;
                const int first_count =
                    min(overlap_end, first_head_end) - overlap_begin;
                __threadfence();
                atomicAdd(g.Bar.raw_ptr + q_flag_offset +
                              layer * globals::num_attention_heads +
                              first_head,
                          static_cast<unsigned>(first_count));
                if (overlap_end > first_head_end) {
                    atomicAdd(g.Bar.raw_ptr + q_flag_offset +
                                  layer * globals::num_attention_heads +
                                  first_head + 1,
                              static_cast<unsigned>(overlap_end -
                                                    first_head_end));
                }
            }
        }
    }
}

using qkv_12w_default = qkv_config<1, 0>;
using qkv_12w_stream = qkv_config<1, -5>;
using qkv_24w_default = qkv_config<2, 0>;
using qkv_24w_stream = qkv_config<2, -5>;
using qkv_4w_default = qkv_config<1, 0, 4>;
using qkv_8w_default = qkv_config<2, 0, 4>;
using qkv_12w_pair = qkv_config<2, 0, 6>;
using qkv_16w_pair = qkv_config<2, 0, 8>;

} // namespace qkv_simt
} // namespace megakernel
