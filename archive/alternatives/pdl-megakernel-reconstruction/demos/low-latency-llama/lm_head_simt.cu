#pragma once

#include "mlp_simt.cu"

#include <cuda_bf16.h>

// Pure-SIMT final RMSNorm + 128256-row LM head.  The fixed 132-CTA grid is
// exactly one H100 wave; each CTA owns 971 or 972 contiguous vocabulary rows.
namespace megakernel {
namespace lm_head_simt {

using globals = llama_1b_globals;
using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

constexpr int vocab_rows = 128256;
constexpr int grid_ctas = globals::sm_count;
constexpr int warps_per_cta = 32;
constexpr int threads_per_cta = warps_per_cta * 32;
static_assert(vocab_rows == 84 * 972 + 48 * 971);

template <int RowsPerUnit, int WeightPolicy, int ValuesPerLoad = 2>
struct config {
    static constexpr int rows_per_unit = RowsPerUnit;
    static constexpr int weight_policy = WeightPolicy;
    static constexpr int values_per_load = ValuesPerLoad;
    static constexpr int ctas = grid_ctas;
    static constexpr int warps = warps_per_cta;
    static constexpr int threads = threads_per_cta;
    static_assert(rows_per_unit == 1 || rows_per_unit == 2 ||
                  rows_per_unit == 4 || rows_per_unit == 8);
    static_assert(values_per_load == 2 || values_per_load == 4 ||
                  values_per_load == 8);
};

using lm_1r_default = config<1, 0>;
using lm_2r_default = config<2, 0>;
using lm_4r_default = config<4, 0>;
using lm_2r_stream = config<2, -5>;
using lm_4r_stream = config<4, -5>;
using lm_8r_stream = config<8, -5>;
using lm_1r_stream_v4 = config<1, -5, 4>;
using lm_2r_default_v4 = config<2, 0, 4>;
using lm_2r_stream_v4 = config<2, -5, 4>;
using lm_4r_stream_v4 = config<4, -5, 4>;
using lm_2r_stream_v8 = config<2, -5, 8>;

struct float8_split {
    float2 pair0;
    float2 pair1;
    float2 pair2;
    float2 pair3;
};

template <int WeightPolicy>
__device__ __forceinline__ mlp_simt::float4_split
load_weight_bf16x4(const bf16 *ptr, int quad_index) {
    union packed_bf16x4 {
        unsigned long long bits;
        bf16x2 pairs[2];
    } packed;
    const unsigned long long *address =
        reinterpret_cast<const unsigned long long *>(ptr) + quad_index;
    if constexpr (WeightPolicy == 0) {
        packed.bits = *address;
    } else if constexpr (WeightPolicy == -1) {
        asm volatile("ld.global.cg.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else if constexpr (WeightPolicy == -2) {
        asm volatile("ld.global.cs.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else if constexpr (WeightPolicy == -3) {
        asm volatile("ld.global.L1::no_allocate.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else if constexpr (WeightPolicy == -4) {
        asm volatile(
            "ld.global.L1::no_allocate.L2::128B.b64 %0, [%1];"
            : "=l"(packed.bits)
            : "l"(address));
    } else if constexpr (WeightPolicy == -6) {
        asm volatile("ld.global.nc.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else if constexpr (WeightPolicy == -7) {
        asm volatile("ld.global.nc.L1::no_allocate.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else if constexpr (WeightPolicy == -8) {
        asm volatile("ld.global.L1::evict_first.b64 %0, [%1];"
                     : "=l"(packed.bits)
                     : "l"(address));
    } else {
        static_assert(WeightPolicy == -5);
        asm volatile(
            "ld.global.L1::no_allocate.L2::256B.b64 %0, [%1];"
            : "=l"(packed.bits)
            : "l"(address));
    }
    return {__bfloat1622float2(packed.pairs[0]),
            __bfloat1622float2(packed.pairs[1])};
}

template <int WeightPolicy>
__device__ __forceinline__ float8_split
load_weight_bf16x8(const bf16 *ptr, int octet_index) {
    const unsigned long long *address =
        reinterpret_cast<const unsigned long long *>(ptr) + 2 * octet_index;
    union packed_bf16x4 {
        unsigned long long bits;
        bf16x2 pairs[2];
    } lo, hi;
    if constexpr (WeightPolicy == 0) {
        lo.bits = address[0];
        hi.bits = address[1];
    } else {
        static_assert(WeightPolicy == -5);
        asm volatile(
            "ld.global.L1::no_allocate.L2::256B.b64 %0, [%1];"
            : "=l"(lo.bits)
            : "l"(address));
        asm volatile(
            "ld.global.L1::no_allocate.L2::256B.b64 %0, [%1];"
            : "=l"(hi.bits)
            : "l"(address + 1));
    }
    return {__bfloat1622float2(lo.pairs[0]),
            __bfloat1622float2(lo.pairs[1]),
            __bfloat1622float2(hi.pairs[0]),
            __bfloat1622float2(hi.pairs[1])};
}

__device__ __forceinline__ float8_split
load_bf16x8(const bf16 *ptr, int octet_index) {
    const mlp_simt::float4_split lo =
        mlp_simt::load_bf16x4(ptr, 2 * octet_index);
    const mlp_simt::float4_split hi =
        mlp_simt::load_bf16x4(ptr, 2 * octet_index + 1);
    return {lo.lo, lo.hi, hi.lo, hi.hi};
}

__device__ __forceinline__ void accumulate_octet(float8_split x,
                                                   float8_split weight,
                                                   float &sum) {
    mlp_simt::accumulate_pair(x.pair0, weight.pair0, sum);
    mlp_simt::accumulate_pair(x.pair1, weight.pair1, sum);
    mlp_simt::accumulate_pair(x.pair2, weight.pair2, sum);
    mlp_simt::accumulate_pair(x.pair3, weight.pair3, sum);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ void prefetch_l2(const void *address) {
    asm volatile("prefetch.global.L2 [%0];" : : "l"(address));
}

template <typename Config>
__launch_bounds__(Config::threads, 1) __global__ void
rms_lm_head(const __grid_constant__ globals g) {
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int hidden_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += Config::threads) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = warp_sums[lane];
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
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        const float2 scale =
            mlp_simt::load_bf16x2(g.lm_head_norm_weights.raw_ptr, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    // 128256 = 84 * 972 + 48 * 971.
    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 971 + (block < 84 ? block : 84);
    const int row_count = 971 + (block < 84);
    const int row_end = row_begin + row_count;
    int unit_count;
    if constexpr (Config::rows_per_unit == 1) {
        unit_count = row_count;
    } else if constexpr (Config::rows_per_unit == 2) {
        unit_count = 486;
    } else if constexpr (Config::rows_per_unit == 4) {
        unit_count = 243;
    } else {
        unit_count = 122;
    }

    const bf16 *weights = g.lm_head_weights.raw_ptr;
    for (int unit = warp; unit < unit_count; unit += Config::warps) {
        const int first_row =
            row_begin + unit * Config::rows_per_unit;
        float sums[Config::rows_per_unit] = {};
        if constexpr (Config::values_per_load == 2) {
#pragma unroll 4
            for (int pair = lane; pair < hidden_pairs; pair += 32) {
                const float2 x = mlp_simt::load_bf16x2(normalized, pair);
#pragma unroll
                for (int row_iter = 0; row_iter < Config::rows_per_unit;
                     ++row_iter) {
                    const int row = first_row + row_iter;
                    if (row < row_end) {
                        mlp_simt::accumulate_pair(
                            x,
                            mlp_simt::load_weight_bf16x2<
                                Config::weight_policy>(
                                weights + row * globals::hidden_dim, pair),
                            sums[row_iter]);
                    }
                }
            }
        } else if constexpr (Config::values_per_load == 4) {
            constexpr int hidden_quads = globals::hidden_dim / 4;
#pragma unroll 4
            for (int quad = lane; quad < hidden_quads; quad += 32) {
                const mlp_simt::float4_split x =
                    mlp_simt::load_bf16x4(normalized, quad);
#pragma unroll
                for (int row_iter = 0; row_iter < Config::rows_per_unit;
                     ++row_iter) {
                    const int row = first_row + row_iter;
                    if (row < row_end) {
                        mlp_simt::accumulate_quad(
                            x,
                            load_weight_bf16x4<Config::weight_policy>(
                                weights + row * globals::hidden_dim, quad),
                            sums[row_iter]);
                    }
                }
            }
        } else {
            constexpr int hidden_octets = globals::hidden_dim / 8;
#pragma unroll 4
            for (int octet = lane; octet < hidden_octets; octet += 32) {
                const float8_split x = load_bf16x8(normalized, octet);
#pragma unroll
                for (int row_iter = 0; row_iter < Config::rows_per_unit;
                     ++row_iter) {
                    const int row = first_row + row_iter;
                    if (row < row_end) {
                        accumulate_octet(
                            x,
                            load_weight_bf16x8<Config::weight_policy>(
                                weights + row * globals::hidden_dim, octet),
                            sums[row_iter]);
                    }
                }
            }
        }
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_unit;
             ++row_iter) {
            const int row = first_row + row_iter;
            const float sum = warp_sum(sums[row_iter]);
            if (lane == 0 && row < row_end) {
                g.logits.raw_ptr[row] = __float2bfloat16_rn(sum);
            }
        }
    }
}

template <int WeightPolicy, bool FullUnroll = false>
__device__ __forceinline__ void
compute_two_rows_stream_v4(const bf16 *normalized, const bf16 *weights,
                           bf16 *logits, int row0, int row1, int lane) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    constexpr int hidden_quads = globals::hidden_dim / 4;
    if constexpr (FullUnroll) {
#pragma unroll
        for (int quad = lane; quad < hidden_quads; quad += 32) {
            const mlp_simt::float4_split x =
                mlp_simt::load_bf16x4(normalized, quad);
            mlp_simt::accumulate_quad(
                x, load_weight_bf16x4<WeightPolicy>(
                       weights + row0 * globals::hidden_dim, quad),
                sum0);
            mlp_simt::accumulate_quad(
                x, load_weight_bf16x4<WeightPolicy>(
                       weights + row1 * globals::hidden_dim, quad),
                sum1);
        }
    } else {
#pragma unroll 4
        for (int quad = lane; quad < hidden_quads; quad += 32) {
            const mlp_simt::float4_split x =
                mlp_simt::load_bf16x4(normalized, quad);
            mlp_simt::accumulate_quad(
                x, load_weight_bf16x4<WeightPolicy>(
                       weights + row0 * globals::hidden_dim, quad),
                sum0);
            mlp_simt::accumulate_quad(
                x, load_weight_bf16x4<WeightPolicy>(
                       weights + row1 * globals::hidden_dim, quad),
                sum1);
        }
    }
    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (lane == 0) {
        logits[row0] = __float2bfloat16_rn(sum0);
        logits[row1] = __float2bfloat16_rn(sum1);
    }
}

// Balanced-tail specialization for the best 2-row/v4 stream.  Every warp
// computes exactly 15 full row-pairs (960 rows/CTA).  The remaining 11 or 12
// rows are each split across two warps, replacing the original six-warp tail
// with a broad half-row tail while preserving total bytes and one-wave launch.
template <int WeightPolicy, bool FullUnroll = false, bool StripedRows = false>
__launch_bounds__(threads_per_cta, 1) __global__ void
rms_lm_head_balanced_tail_v4(const __grid_constant__ globals g) {
    static_assert(WeightPolicy == 0 ||
                  (WeightPolicy >= -8 && WeightPolicy <= -1));
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[warps_per_cta];
    __shared__ float tail_partials[12][2];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int hidden_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = warp_sums[lane];
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(
                block_sum / static_cast<float>(globals::hidden_dim) +
                g.rms_norm_eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        const float2 scale =
            mlp_simt::load_bf16x2(g.lm_head_norm_weights.raw_ptr, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 971 + (block < 84 ? block : 84);
    const int row_count = 971 + (block < 84);
    const bf16 *weights = g.lm_head_weights.raw_ptr;
    bf16 *logits = g.logits.raw_ptr;

#pragma unroll 1
    for (int batch = 0; batch < 15; ++batch) {
        const int pair_unit = warp + batch * warps_per_cta;
        const int local_row0 = 2 * pair_unit;
        const int row0 = StripedRows ? local_row0 * grid_ctas + block
                                     : row_begin + local_row0;
        const int row1 = StripedRows ? (local_row0 + 1) * grid_ctas + block
                                     : row0 + 1;
        compute_two_rows_stream_v4<WeightPolicy, FullUnroll>(
            normalized, weights, logits, row0, row1, lane);
    }

    const int tail_rows = row_count - 960;
    if (warp < 2 * tail_rows) {
        const int tail_row = warp >> 1;
        const int reduction_half = warp & 1;
        const int row = StripedRows
                            ? (960 + tail_row) * grid_ctas + block
                            : row_begin + 960 + tail_row;
        const bf16 *weight_row = weights + row * globals::hidden_dim;
        constexpr int hidden_quads = globals::hidden_dim / 4;
        float sum = 0.0f;
        if constexpr (FullUnroll) {
#pragma unroll
            for (int quad = lane + reduction_half * 32; quad < hidden_quads;
                 quad += 64) {
                mlp_simt::accumulate_quad(
                    mlp_simt::load_bf16x4(normalized, quad),
                    load_weight_bf16x4<WeightPolicy>(weight_row, quad), sum);
            }
        } else {
#pragma unroll 4
            for (int quad = lane + reduction_half * 32; quad < hidden_quads;
                 quad += 64) {
                mlp_simt::accumulate_quad(
                    mlp_simt::load_bf16x4(normalized, quad),
                    load_weight_bf16x4<WeightPolicy>(weight_row, quad), sum);
            }
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            tail_partials[tail_row][reduction_half] = sum;
        }
    }
    __syncthreads();
    if (warp < tail_rows && lane == 0) {
        const int row = StripedRows ? (960 + warp) * grid_ctas + block
                                    : row_begin + 960 + warp;
        logits[row] = __float2bfloat16_rn(
            tail_partials[warp][0] + tail_partials[warp][1]);
    }
}

template <int TileCols>
__device__ __forceinline__ void
issue_cp_async_two_rows(bf16 *stage, const bf16 *weights, int row0,
                        int row1, int col_begin, int lane) {
    static_assert(TileCols == 256 || TileCols == 512);
    constexpr int values_per_copy = 16 / sizeof(bf16);
    constexpr int copies_per_row = TileCols / values_per_copy;
    for (int copy = lane; copy < copies_per_row; copy += 32) {
        const int col = copy * values_per_copy;
        mlp_simt::cp_async_cg_16(
            stage + col,
            weights + row0 * globals::hidden_dim + col_begin + col);
        mlp_simt::cp_async_cg_16(
            stage + TileCols + col,
            weights + row1 * globals::hidden_dim + col_begin + col);
    }
    mlp_simt::cp_async_commit();
}

template <int TileCols>
__device__ __forceinline__ void
compute_two_rows_cp_async(const bf16 *normalized, const bf16 *weights,
                          bf16 *logits, bf16 *stages, int warp, int row0,
                          int row1, int lane) {
    static_assert(globals::hidden_dim % TileCols == 0);
    constexpr int tile_count = globals::hidden_dim / TileCols;
    constexpr int stage_stride =
        warps_per_cta * 2 * TileCols;
    constexpr int warp_stride = 2 * TileCols;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    bf16 *stage0 = stages + warp * warp_stride;
    issue_cp_async_two_rows<TileCols>(stage0, weights, row0, row1, 0,
                                     lane);

#pragma unroll
    for (int tile = 0; tile < tile_count; ++tile) {
        const int stage_idx = tile & 1;
        bf16 *current = stages + stage_idx * stage_stride +
                        warp * warp_stride;
        if (tile + 1 < tile_count) {
            const int next_stage_idx = stage_idx ^ 1;
            bf16 *next = stages + next_stage_idx * stage_stride +
                         warp * warp_stride;
            issue_cp_async_two_rows<TileCols>(
                next, weights, row0, row1, (tile + 1) * TileCols, lane);
            mlp_simt::cp_async_wait_one();
        } else {
            mlp_simt::cp_async_wait_all();
        }
        __syncwarp();

        constexpr int tile_pairs = TileCols / 2;
#pragma unroll
        for (int pair = lane; pair < tile_pairs; pair += 32) {
            const float2 x = mlp_simt::load_bf16x2(
                normalized, tile * tile_pairs + pair);
            mlp_simt::accumulate_pair(
                x, mlp_simt::load_bf16x2(current, pair), sum0);
            mlp_simt::accumulate_pair(
                x,
                mlp_simt::load_bf16x2(current + TileCols, pair), sum1);
        }
        __syncwarp();
    }

    sum0 = warp_sum(sum0);
    sum1 = warp_sum(sum1);
    if (lane == 0) {
        logits[row0] = __float2bfloat16_rn(sum0);
        logits[row1] = __float2bfloat16_rn(sum1);
    }
}

template <int TileCols>
__launch_bounds__(threads_per_cta, 1) __global__ void
rms_lm_head_cp_async_balanced_tail(const __grid_constant__ globals g) {
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[warps_per_cta];
    __shared__ float tail_partials[12][2];
    extern __shared__ __align__(16) bf16 stages[];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int hidden_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = warp_sums[lane];
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(
                block_sum / static_cast<float>(globals::hidden_dim) +
                g.rms_norm_eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        const float2 scale =
            mlp_simt::load_bf16x2(g.lm_head_norm_weights.raw_ptr, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 971 + (block < 84 ? block : 84);
    const int row_count = 971 + (block < 84);
    const bf16 *weights = g.lm_head_weights.raw_ptr;
    bf16 *logits = g.logits.raw_ptr;

#pragma unroll 1
    for (int batch = 0; batch < 15; ++batch) {
        const int pair_unit = warp + batch * warps_per_cta;
        const int row0 = row_begin + 2 * pair_unit;
        compute_two_rows_cp_async<TileCols>(
            normalized, weights, logits, stages, warp, row0, row0 + 1,
            lane);
    }

    const int tail_rows = row_count - 960;
    if (warp < 2 * tail_rows) {
        const int tail_row = warp >> 1;
        const int reduction_half = warp & 1;
        const int row = row_begin + 960 + tail_row;
        const bf16 *weight_row = weights + row * globals::hidden_dim;
        constexpr int hidden_quads = globals::hidden_dim / 4;
        float sum = 0.0f;
#pragma unroll
        for (int quad = lane + reduction_half * 32; quad < hidden_quads;
             quad += 64) {
            mlp_simt::accumulate_quad(
                mlp_simt::load_bf16x4(normalized, quad),
                load_weight_bf16x4<-3>(weight_row, quad), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            tail_partials[tail_row][reduction_half] = sum;
        }
    }
    __syncthreads();
    if (warp < tail_rows && lane == 0) {
        logits[row_begin + 960 + warp] = __float2bfloat16_rn(
            tail_partials[warp][0] + tail_partials[warp][1]);
    }
}

__device__ __forceinline__ void
compute_four_rows_na_v4_full(const bf16 *normalized, const bf16 *weights,
                             bf16 *logits, int row0, int lane) {
    float sums[4] = {};
    constexpr int hidden_quads = globals::hidden_dim / 4;
#pragma unroll
    for (int quad = lane; quad < hidden_quads; quad += 32) {
        const mlp_simt::float4_split x =
            mlp_simt::load_bf16x4(normalized, quad);
#pragma unroll
        for (int row_iter = 0; row_iter < 4; ++row_iter) {
            mlp_simt::accumulate_quad(
                x,
                load_weight_bf16x4<-3>(
                    weights + (row0 + row_iter) * globals::hidden_dim,
                    quad),
                sums[row_iter]);
        }
    }
#pragma unroll
    for (int row_iter = 0; row_iter < 4; ++row_iter) {
        const float sum = warp_sum(sums[row_iter]);
        if (lane == 0) {
            logits[row0 + row_iter] = __float2bfloat16_rn(sum);
        }
    }
}

// Four independent rows give each warp more memory-level parallelism.  With
// 24 warps, 960 rows divide exactly into ten four-row groups per warp; the
// final 11/12 rows again use two-warp half reductions.
__launch_bounds__(24 * 32, 1) __global__ void
rms_lm_head_24w4r_balanced_tail(const __grid_constant__ globals g) {
    constexpr int warps = 24;
    constexpr int threads = warps * 32;
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[warps];
    __shared__ float tail_partials[12][2];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int hidden_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += threads) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(
                block_sum / static_cast<float>(globals::hidden_dim) +
                g.rms_norm_eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int pair = tid; pair < hidden_pairs; pair += threads) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        const float2 scale =
            mlp_simt::load_bf16x2(g.lm_head_norm_weights.raw_ptr, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 971 + (block < 84 ? block : 84);
    const int row_count = 971 + (block < 84);
    const bf16 *weights = g.lm_head_weights.raw_ptr;
    bf16 *logits = g.logits.raw_ptr;

#pragma unroll 1
    for (int batch = 0; batch < 10; ++batch) {
        const int unit = warp + batch * warps;
        compute_four_rows_na_v4_full(normalized, weights, logits,
                                     row_begin + 4 * unit, lane);
    }

    const int tail_rows = row_count - 960;
    if (warp < 2 * tail_rows) {
        const int tail_row = warp >> 1;
        const int reduction_half = warp & 1;
        const int row = row_begin + 960 + tail_row;
        const bf16 *weight_row = weights + row * globals::hidden_dim;
        constexpr int hidden_quads = globals::hidden_dim / 4;
        float sum = 0.0f;
#pragma unroll
        for (int quad = lane + reduction_half * 32; quad < hidden_quads;
             quad += 64) {
            mlp_simt::accumulate_quad(
                mlp_simt::load_bf16x4(normalized, quad),
                load_weight_bf16x4<-3>(weight_row, quad), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            tail_partials[tail_row][reduction_half] = sum;
        }
    }
    __syncthreads();
    if (warp < tail_rows && lane == 0) {
        logits[row_begin + 960 + warp] = __float2bfloat16_rn(
            tail_partials[warp][0] + tail_partials[warp][1]);
    }
}

__device__ __forceinline__ void
compute_three_rows_na_v4_full(const bf16 *normalized, const bf16 *weights,
                              bf16 *logits, int row0, int lane) {
    float sums[3] = {};
    constexpr int hidden_quads = globals::hidden_dim / 4;
#pragma unroll
    for (int quad = lane; quad < hidden_quads; quad += 32) {
        const mlp_simt::float4_split x =
            mlp_simt::load_bf16x4(normalized, quad);
#pragma unroll
        for (int row_iter = 0; row_iter < 3; ++row_iter) {
            mlp_simt::accumulate_quad(
                x,
                load_weight_bf16x4<-3>(
                    weights + (row0 + row_iter) * globals::hidden_dim,
                    quad),
                sums[row_iter]);
        }
    }
#pragma unroll
    for (int row_iter = 0; row_iter < 3; ++row_iter) {
        const float sum = warp_sum(sums[row_iter]);
        if (lane == 0) {
            logits[row0 + row_iter] = __float2bfloat16_rn(sum);
        }
    }
}

// 960 = 32 warps * 10 batches * 3 rows, so this variant keeps all 32 warps
// resident while exposing three independent weight streams without a base
// workload imbalance.
template <int PrefetchLines, bool PdlWait = false,
          int PdlPrefetchPolicy = mlp_simt::prefetch_policy_l2>
__launch_bounds__(threads_per_cta, 1) __global__ void
rms_lm_head_32w3r_balanced_tail(const __grid_constant__ globals g) {
    static_assert(PrefetchLines == 0 || PrefetchLines == 1 ||
                  PrefetchLines == 2 || PrefetchLines == 4 ||
                  PrefetchLines == 8 || PrefetchLines == 12 ||
                  PrefetchLines == 16 || PrefetchLines == 24 ||
                  PrefetchLines == 32);
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[warps_per_cta];
    __shared__ float tail_partials[12][2];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int hidden_pairs = globals::hidden_dim / 2;
    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 971 + (block < 84 ? block : 84);
    const int row_count = 971 + (block < 84);
    const bf16 *weights = g.lm_head_weights.raw_ptr;
    bf16 *logits = g.logits.raw_ptr;

    if constexpr (PrefetchLines > 0) {
        constexpr int values_per_l2_line = 128 / sizeof(bf16);
        if (lane < PrefetchLines) {
#pragma unroll
            for (int row_iter = 0; row_iter < 3; ++row_iter) {
                const int row = row_begin + 3 * warp + row_iter;
                mlp_simt::prefetch_global<PdlPrefetchPolicy>(
                    weights + row * globals::hidden_dim +
                    lane * values_per_l2_line);
            }
        }
    }
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }

    float square_sum = 0.0f;
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        square_sum = fmaf(value.x, value.x, square_sum);
        square_sum = fmaf(value.y, value.y, square_sum);
    }
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = warp_sums[lane];
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(
                block_sum / static_cast<float>(globals::hidden_dim) +
                g.rms_norm_eps);
        }
    }
    __syncthreads();

    const float inv_rms = warp_sums[0];
    for (int pair = tid; pair < hidden_pairs; pair += threads_per_cta) {
        const float2 value =
            mlp_simt::load_bf16x2(g.hidden_states.raw_ptr, pair);
        const float2 scale =
            mlp_simt::load_bf16x2(g.lm_head_norm_weights.raw_ptr, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

#pragma unroll 1
    for (int batch = 0; batch < 10; ++batch) {
        const int unit = warp + batch * warps_per_cta;
        compute_three_rows_na_v4_full(normalized, weights, logits,
                                      row_begin + 3 * unit, lane);
    }

    const int tail_rows = row_count - 960;
    if (warp < 2 * tail_rows) {
        const int tail_row = warp >> 1;
        const int reduction_half = warp & 1;
        const int row = row_begin + 960 + tail_row;
        const bf16 *weight_row = weights + row * globals::hidden_dim;
        constexpr int hidden_quads = globals::hidden_dim / 4;
        float sum = 0.0f;
#pragma unroll
        for (int quad = lane + reduction_half * 32; quad < hidden_quads;
             quad += 64) {
            mlp_simt::accumulate_quad(
                mlp_simt::load_bf16x4(normalized, quad),
                load_weight_bf16x4<-3>(weight_row, quad), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            tail_partials[tail_row][reduction_half] = sum;
        }
    }
    __syncthreads();
    if (warp < tail_rows && lane == 0) {
        logits[row_begin + 960 + warp] = __float2bfloat16_rn(
            tail_partials[warp][0] + tail_partials[warp][1]);
    }
}

} // namespace lm_head_simt
} // namespace megakernel
