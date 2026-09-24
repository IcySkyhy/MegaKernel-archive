#pragma once

#include "llama.cuh"

#include <cuda_bf16.h>

// Low-resource batch-1 MLP controls.  These deliberately do not reuse the
// persistent VM page allocator, TMA pipeline, or tensor-core matvec path.  The
// existing TK standalone kernels remain the correctness/performance baseline.
namespace megakernel {
namespace mlp_simt {

using globals = llama_1b_globals;
using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

template <int WarpsPerCta, int RowsPerWarp, int OutputRows, int GridCtas = 0>
struct gemv_config {
    static constexpr int warps = WarpsPerCta;
    static constexpr int rows_per_warp = RowsPerWarp;
    static constexpr int threads = warps * 32;
    static constexpr int rows_per_cta = warps * rows_per_warp;
    static constexpr bool balanced_grid = GridCtas != 0;
    static constexpr int ctas =
        balanced_grid ? GridCtas : OutputRows / rows_per_cta;
    static constexpr int min_blocks_per_sm = threads == 1024 ? 1 : 2;

    static_assert(warps == 4 || warps == 8 || warps == 16 || warps == 32);
    static_assert(balanced_grid || OutputRows % rows_per_cta == 0);
    static_assert(!balanced_grid || rows_per_warp <= 2);
};

// The aliases name the launch-shape hypotheses included in one H100 build.
// The 8-warp variants remain the stable default entry points.
using upgate_4w8r = gemv_config<4, 8, globals::intermediate_dim>;
using upgate_8w4r = gemv_config<8, 4, globals::intermediate_dim>;
using upgate_8w8r = gemv_config<8, 8, globals::intermediate_dim>;
using upgate_16w2r = gemv_config<16, 2, globals::intermediate_dim>;
using upgate_16w4r = gemv_config<16, 4, globals::intermediate_dim>;
using upgate_32w1r = gemv_config<32, 1, globals::intermediate_dim>;
using upgate_32w2r = gemv_config<32, 2, globals::intermediate_dim>;
using upgate_32w1r_264 =
    gemv_config<32, 1, globals::intermediate_dim, 264>;
using upgate_32w2r_132 =
    gemv_config<32, 2, globals::intermediate_dim, 132>;

using down_4w4r = gemv_config<4, 4, globals::hidden_dim>;
using down_8w2r = gemv_config<8, 2, globals::hidden_dim>;
using down_8w4r = gemv_config<8, 4, globals::hidden_dim>;
using down_16w1r = gemv_config<16, 1, globals::hidden_dim>;

static_assert(upgate_8w4r::ctas == 256);
static_assert(down_8w2r::ctas == 128);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ int instruction_layer(const globals &g) {
    // Every standalone graph node owns one immutable [SM, 1, 32]
    // instruction tensor.  instruction[1] is layer_idx for both op5 and op6.
    return g.instructions.raw_ptr[1];
}

__device__ __forceinline__ float2 load_bf16x2(const bf16 *ptr,
                                               int pair_index) {
    const bf16x2 packed = reinterpret_cast<const bf16x2 *>(ptr)[pair_index];
    return __bfloat1622float2(packed);
}

template <int L2PrefetchBytes>
__device__ __forceinline__ float2 load_weight_bf16x2(const bf16 *ptr,
                                                      int pair_index) {
    if constexpr (L2PrefetchBytes == 0) {
        return load_bf16x2(ptr, pair_index);
    } else if constexpr (L2PrefetchBytes == -1 || L2PrefetchBytes == -2) {
        const unsigned *address =
            reinterpret_cast<const unsigned *>(ptr) + pair_index;
        const unsigned packed_bits = L2PrefetchBytes == -1
                                         ? __ldcg(address)
                                         : __ldcs(address);
        union packed_bf16x2 {
            unsigned bits;
            bf16x2 value;
        } packed{packed_bits};
        return __bfloat1622float2(packed.value);
    } else if constexpr (L2PrefetchBytes == -3 || L2PrefetchBytes == -4 ||
                         L2PrefetchBytes == -5) {
        const bf16x2 *address =
            reinterpret_cast<const bf16x2 *>(ptr) + pair_index;
        unsigned packed_bits;
        if constexpr (L2PrefetchBytes == -3) {
            asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];"
                         : "=r"(packed_bits)
                         : "l"(address));
        } else if constexpr (L2PrefetchBytes == -4) {
            asm volatile(
                "ld.global.L1::no_allocate.L2::128B.b32 %0, [%1];"
                : "=r"(packed_bits)
                : "l"(address));
        } else {
            asm volatile(
                "ld.global.L1::no_allocate.L2::256B.b32 %0, [%1];"
                : "=r"(packed_bits)
                : "l"(address));
        }
        union packed_bf16x2 {
            unsigned bits;
            bf16x2 value;
        } packed{packed_bits};
        return __bfloat1622float2(packed.value);
    } else {
        static_assert(L2PrefetchBytes == 128 || L2PrefetchBytes == 256);
        const bf16x2 *address =
            reinterpret_cast<const bf16x2 *>(ptr) + pair_index;
        unsigned packed_bits;
        if constexpr (L2PrefetchBytes == 128) {
            asm volatile("ld.global.L2::128B.b32 %0, [%1];"
                         : "=r"(packed_bits)
                         : "l"(address));
        } else {
            asm volatile("ld.global.L2::256B.b32 %0, [%1];"
                         : "=r"(packed_bits)
                         : "l"(address));
        }
        union packed_bf16x2 {
            unsigned bits;
            bf16x2 value;
        } packed{packed_bits};
        return __bfloat1622float2(packed.value);
    }
}

__device__ __forceinline__ unsigned
load_weight_bf16x2_bits_na256(const bf16 *ptr, int pair_index) {
    const bf16x2 *address =
        reinterpret_cast<const bf16x2 *>(ptr) + pair_index;
    unsigned packed_bits;
    asm volatile(
        "ld.global.L1::no_allocate.L2::256B.b32 %0, [%1];"
        : "=r"(packed_bits)
        : "l"(address)
        : "memory");
    return packed_bits;
}

__device__ __forceinline__ float2
bf16x2_bits_to_float2(unsigned packed_bits) {
    union packed_bf16x2 {
        unsigned bits;
        bf16x2 value;
    } packed{packed_bits};
    return __bfloat1622float2(packed.value);
}

__device__ __forceinline__ void accumulate_pair(float2 x, float2 weight,
                                                 float &sum) {
    sum = fmaf(x.x, weight.x, sum);
    sum = fmaf(x.y, weight.y, sum);
}

struct float4_split {
    float2 lo;
    float2 hi;
};

__device__ __forceinline__ float4_split load_bf16x4(const bf16 *ptr,
                                                     int quad_index) {
    union packed_bf16x4 {
        unsigned long long bits;
        bf16x2 pairs[2];
    } packed;
    packed.bits =
        reinterpret_cast<const unsigned long long *>(ptr)[quad_index];
    return {__bfloat1622float2(packed.pairs[0]),
            __bfloat1622float2(packed.pairs[1])};
}

template <int WeightPolicy>
__device__ __forceinline__ float4_split
load_weight_bf16x4(const bf16 *ptr, int quad_index) {
    union packed_bf16x4 {
        unsigned long long bits;
        bf16x2 pairs[2];
    } packed;
    const unsigned long long *address =
        reinterpret_cast<const unsigned long long *>(ptr) + quad_index;
    if constexpr (WeightPolicy == 0) {
        packed.bits = *address;
    } else if constexpr (WeightPolicy == -3) {
        asm volatile("ld.global.L1::no_allocate.b64 %0, [%1];"
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

enum prefetch_policy : int {
    prefetch_policy_l1 = 1,
    prefetch_policy_l2 = 2,
    prefetch_policy_l2_evict_last = 3,
};

template <int Policy>
__device__ __forceinline__ void prefetch_global(const void *address) {
    static_assert(Policy == prefetch_policy_l1 ||
                  Policy == prefetch_policy_l2 ||
                  Policy == prefetch_policy_l2_evict_last);
    if constexpr (Policy == prefetch_policy_l1) {
        asm volatile("prefetch.global.L1 [%0];" : : "l"(address));
    } else if constexpr (Policy == prefetch_policy_l2) {
        asm volatile("prefetch.global.L2 [%0];" : : "l"(address));
    } else {
        asm volatile("prefetch.global.L2::evict_last [%0];"
                     :
                     : "l"(address));
    }
}

__device__ __forceinline__ void prefetch_l2(const void *address) {
    prefetch_global<prefetch_policy_l2>(address);
}

__device__ __forceinline__ void prefetch_l2_evict_last(const void *address) {
    prefetch_global<prefetch_policy_l2_evict_last>(address);
}

template <int FetchBytes>
__device__ __forceinline__ unsigned demand_warm_l2(const void *address) {
    static_assert(FetchBytes == 128 || FetchBytes == 256);
    unsigned sink;
    if constexpr (FetchBytes == 128) {
        asm volatile(
            "ld.global.L1::no_allocate.L2::128B.b32 %0, [%1];"
            : "=r"(sink)
            : "l"(address)
            : "memory");
    } else {
        asm volatile(
            "ld.global.L1::no_allocate.L2::256B.b32 %0, [%1];"
            : "=r"(sink)
            : "l"(address)
            : "memory");
    }
    return sink;
}

__device__ __forceinline__ void accumulate_quad(float4_split x,
                                                 float4_split weight,
                                                 float &sum) {
    accumulate_pair(x.lo, weight.lo, sum);
    accumulate_pair(x.hi, weight.hi, sum);
}

template <typename Config, bool InterleaveRows, int L2PrefetchBytes = 0,
          bool AdjacentRows = false>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
upgate(
    const __grid_constant__ globals g) {
    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;

    const int norm_pairs = globals::hidden_dim / 2;
    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    int first_row = 0;
    int row_end = globals::intermediate_dim;
    if constexpr (Config::balanced_grid) {
        const int row_begin = static_cast<int>(blockIdx.x) *
                              globals::intermediate_dim / Config::ctas;
        row_end = (static_cast<int>(blockIdx.x) + 1) *
                  globals::intermediate_dim / Config::ctas;
        if (warp >= row_end - row_begin) {
            return;
        }
        first_row = row_begin + warp;
    } else {
        first_row = static_cast<int>(blockIdx.x) * Config::rows_per_cta +
                    (AdjacentRows ? warp * Config::rows_per_warp : warp);
    }
    const int hidden_pairs = globals::hidden_dim / 2;
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;

    if constexpr (InterleaveRows && Config::balanced_grid &&
                  Config::rows_per_warp == 2) {
        const int row0 = first_row;
        const int row1 = first_row + Config::warps;
        const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
        const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
        float up_sum0 = 0.0f;
        float gate_sum0 = 0.0f;

        if (row1 < row_end) {
            const bf16 *up_row1 = up_layer + row1 * globals::hidden_dim;
            const bf16 *gate_row1 =
                gate_layer + row1 * globals::hidden_dim;
            float up_sum1 = 0.0f;
            float gate_sum1 = 0.0f;
            for (int pair = lane; pair < hidden_pairs; pair += 32) {
                const float2 x = load_bf16x2(normalized, pair);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(up_row0, pair),
                    up_sum0);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(gate_row0, pair),
                    gate_sum0);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(up_row1, pair),
                    up_sum1);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(gate_row1, pair),
                    gate_sum1);
            }
            up_sum1 = warp_sum(up_sum1);
            gate_sum1 = warp_sum(gate_sum1);
            if (lane == 0) {
                const float silu1 =
                    gate_sum1 / (1.0f + __expf(-gate_sum1));
                g.silu_out.raw_ptr[row1] =
                    __float2bfloat16_rn(up_sum1 * silu1);
            }
        } else {
            for (int pair = lane; pair < hidden_pairs; pair += 32) {
                const float2 x = load_bf16x2(normalized, pair);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(up_row0, pair),
                    up_sum0);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(gate_row0, pair),
                    gate_sum0);
            }
        }
        up_sum0 = warp_sum(up_sum0);
        gate_sum0 = warp_sum(gate_sum0);
        if (lane == 0) {
            const float silu0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
            g.silu_out.raw_ptr[row0] =
                __float2bfloat16_rn(up_sum0 * silu0);
        }
    } else if constexpr (InterleaveRows) {
        float up_sums[Config::rows_per_warp] = {};
        float gate_sums[Config::rows_per_warp] = {};
        for (int pair = lane; pair < hidden_pairs; pair += 32) {
            const float2 x = load_bf16x2(normalized, pair);
#pragma unroll
            for (int row_iter = 0; row_iter < Config::rows_per_warp;
                 ++row_iter) {
                const int row = first_row +
                                row_iter *
                                    (AdjacentRows ? 1 : Config::warps);
                if constexpr (Config::balanced_grid) {
                    if (row >= row_end) {
                        continue;
                    }
                }
                const bf16 *up_row =
                    up_layer + row * globals::hidden_dim;
                const bf16 *gate_row =
                    gate_layer + row * globals::hidden_dim;
                accumulate_pair(x,
                                load_weight_bf16x2<L2PrefetchBytes>(up_row,
                                                                    pair),
                                up_sums[row_iter]);
                accumulate_pair(x,
                                load_weight_bf16x2<L2PrefetchBytes>(gate_row,
                                                                    pair),
                                gate_sums[row_iter]);
            }
        }
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp;
             ++row_iter) {
            up_sums[row_iter] = warp_sum(up_sums[row_iter]);
            gate_sums[row_iter] = warp_sum(gate_sums[row_iter]);
        }
        if (lane == 0) {
#pragma unroll
            for (int row_iter = 0; row_iter < Config::rows_per_warp;
                 ++row_iter) {
                const int row = first_row +
                                row_iter *
                                    (AdjacentRows ? 1 : Config::warps);
                if constexpr (Config::balanced_grid) {
                    if (row >= row_end) {
                        continue;
                    }
                }
                const float gate_sum = gate_sums[row_iter];
                const float silu =
                    gate_sum / (1.0f + __expf(-gate_sum));
                g.silu_out.raw_ptr[row] =
                    __float2bfloat16_rn(up_sums[row_iter] * silu);
            }
        }
    } else {
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp;
             ++row_iter) {
            const int row = first_row +
                            row_iter * (AdjacentRows ? 1 : Config::warps);
            if constexpr (Config::balanced_grid) {
                if (row >= row_end) {
                    continue;
                }
            }
            const bf16 *up_row = up_layer + row * globals::hidden_dim;
            const bf16 *gate_row = gate_layer + row * globals::hidden_dim;

            float up_sum = 0.0f;
            float gate_sum = 0.0f;
            for (int pair = lane; pair < hidden_pairs; pair += 32) {
                const float2 x = load_bf16x2(normalized, pair);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(up_row, pair),
                    up_sum);
                accumulate_pair(
                    x, load_weight_bf16x2<L2PrefetchBytes>(gate_row, pair),
                    gate_sum);
            }
            up_sum = warp_sum(up_sum);
            gate_sum = warp_sum(gate_sum);
            if (lane == 0) {
                const float silu =
                    gate_sum / (1.0f + __expf(-gate_sum));
                g.silu_out.raw_ptr[row] =
                    __float2bfloat16_rn(up_sum * silu);
            }
        }
    }
}

// Wider-load control for the one-wave 32-warp x 2-row launch.  Arithmetic and
// output ownership match the interleaved kernel, but every lane consumes four
// adjacent BF16 values per LDG instead of two.  This halves the dot-loop and
// address-instruction counts while preserving FP32 accumulation.
template <typename Config>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
upgate_v4(const __grid_constant__ globals g) {
    static_assert(!Config::balanced_grid);
    static_assert(Config::rows_per_warp == 2);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    const int norm_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    const int first_row =
        static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const int row0 = first_row;
    const int row1 = first_row + Config::warps;
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;
    const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
    const bf16 *up_row1 = up_layer + row1 * globals::hidden_dim;
    const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
    const bf16 *gate_row1 = gate_layer + row1 * globals::hidden_dim;
    const int hidden_quads = globals::hidden_dim / 4;

    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    for (int quad = lane; quad < hidden_quads; quad += 32) {
        const float4_split x = load_bf16x4(normalized, quad);
        accumulate_quad(x, load_bf16x4(up_row0, quad), up_sum0);
        accumulate_quad(x, load_bf16x4(gate_row0, quad), gate_sum0);
        accumulate_quad(x, load_bf16x4(up_row1, quad), up_sum1);
        accumulate_quad(x, load_bf16x4(gate_row1, quad), gate_sum1);
    }
    up_sum0 = warp_sum(up_sum0);
    up_sum1 = warp_sum(up_sum1);
    gate_sum0 = warp_sum(gate_sum0);
    gate_sum1 = warp_sum(gate_sum1);
    if (lane == 0) {
        const float silu0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        g.silu_out.raw_ptr[row0] =
            __float2bfloat16_rn(up_sum0 * silu0);
        g.silu_out.raw_ptr[row1] =
            __float2bfloat16_rn(up_sum1 * silu1);
    }
}

__device__ __forceinline__ void cp_async_cg_16(void *smem_dst,
                                                const void *gmem_src) {
    const uint32_t smem_address =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(smem_address), "l"(gmem_src)
                 : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_one() {
    asm volatile("cp.async.wait_group 1;" ::: "memory");
}

template <int TileCols>
__device__ __forceinline__ void
cp_async_upgate_tile(bf16 *stage, const bf16 *up_row0,
                     const bf16 *gate_row0, const bf16 *up_row1,
                     const bf16 *gate_row1, int col_begin, int lane) {
    static_assert(TileCols == 64 || TileCols == 128 || TileCols == 256);
    constexpr int bf16_per_copy = 16 / sizeof(bf16);
    constexpr int copies_per_stream = TileCols / bf16_per_copy;
    for (int copy = lane; copy < copies_per_stream; copy += 32) {
        const int col = col_begin + copy * bf16_per_copy;
        const int stage_col = copy * bf16_per_copy;
        cp_async_cg_16(stage + 0 * TileCols + stage_col, up_row0 + col);
        cp_async_cg_16(stage + 1 * TileCols + stage_col, gate_row0 + col);
        cp_async_cg_16(stage + 2 * TileCols + stage_col, up_row1 + col);
        cp_async_cg_16(stage + 3 * TileCols + stage_col, gate_row1 + col);
    }
    cp_async_commit();
}

// Warp-private cp.async weight staging for the one-wave 32-warp x 2-row
// mapping.  The next weight tile is copied while SIMT FP32 accumulation uses
// the current tile.  The shared allocation is 32 KiB for TileCols=64 and
// 64 KiB for TileCols=128, and 128 KiB for TileCols=256, in addition to the
// 4 KiB normalized activation.
template <typename Config, int TileCols, int StageCount = 2,
          bool PdlWait = false, bool PdlTrigger = false>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
upgate_cp_async(const __grid_constant__ globals g) {
    static_assert(!Config::balanced_grid);
    static_assert(Config::warps == 32);
    static_assert(Config::rows_per_warp == 2);
    static_assert(StageCount == 2 || StageCount == 3);
    static_assert(globals::hidden_dim % TileCols == 0);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];
    // Keep the weight stages dynamic so TileCols=128 can opt in above CUDA's
    // default 48-KiB static-shared limit on Hopper.
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    bf16 *weight_stages = reinterpret_cast<bf16 *>(dynamic_smem);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    const int norm_pairs = globals::hidden_dim / 2;

    // Weight addresses are activation-independent.  A programmatically
    // admitted CTA can therefore fill the initial SMEM stages before waiting
    // for the upstream O-projection.  The async-copy wait remains below the
    // RMSNorm, so both the upstream kernel and the post-dependency activation
    // work contribute useful latency-hiding time.
    const int row0 = static_cast<int>(blockIdx.x) * Config::rows_per_cta +
                     warp;
    const int row1 = row0 + Config::warps;
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;
    const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
    const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
    const bf16 *up_row1 = up_layer + row1 * globals::hidden_dim;
    const bf16 *gate_row1 = gate_layer + row1 * globals::hidden_dim;

    constexpr int stage_stride = Config::warps * 4 * TileCols;
    constexpr int warp_stride = 4 * TileCols;
    bf16 *warp_stage0 = weight_stages + warp * warp_stride;
    cp_async_upgate_tile<TileCols>(warp_stage0, up_row0, gate_row0,
                                   up_row1, gate_row1, 0, lane);
    if constexpr (StageCount == 3) {
        bf16 *warp_stage1 = weight_stages + stage_stride +
                            warp * warp_stride;
        cp_async_upgate_tile<TileCols>(
            warp_stage1, up_row0, gate_row0, up_row1, gate_row1,
            TileCols, lane);
    }
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    float up_sum0 = 0.0f;
    float gate_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum1 = 0.0f;
    if constexpr (StageCount == 3) {
        // Group zero is ready; group one may remain in flight.
        cp_async_wait_one();
    } else {
        cp_async_wait_all();
    }
    __syncwarp();

    constexpr int tiles = globals::hidden_dim / TileCols;
#pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage_index = tile % StageCount;
        constexpr int prefetch_distance = StageCount - 1;
        if (tile + prefetch_distance < tiles) {
            const int prefetch_tile = tile + prefetch_distance;
            const int prefetch_stage = prefetch_tile % StageCount;
            bf16 *next_stage = weight_stages +
                               prefetch_stage * stage_stride +
                               warp * warp_stride;
            cp_async_upgate_tile<TileCols>(
                next_stage, up_row0, gate_row0, up_row1, gate_row1,
                prefetch_tile * TileCols, lane);
        }

        const bf16 *stage = weight_stages + stage_index * stage_stride +
                            warp * warp_stride;
        constexpr int tile_pairs = TileCols / 2;
#pragma unroll
        for (int pair = lane; pair < tile_pairs; pair += 32) {
            const float2 x = load_bf16x2(
                normalized, tile * tile_pairs + pair);
            accumulate_pair(x, load_bf16x2(stage + 0 * TileCols, pair),
                            up_sum0);
            accumulate_pair(x, load_bf16x2(stage + 1 * TileCols, pair),
                            gate_sum0);
            accumulate_pair(x, load_bf16x2(stage + 2 * TileCols, pair),
                            up_sum1);
            accumulate_pair(x, load_bf16x2(stage + 3 * TileCols, pair),
                            gate_sum1);
        }
        if (tile + 1 < tiles) {
            if constexpr (StageCount == 3) {
                if (tile + prefetch_distance < tiles) {
                    cp_async_wait_one();
                } else {
                    // At the tail only the immediately-next group remains;
                    // wait_group 1 would be allowed to return too early.
                    cp_async_wait_all();
                }
            } else {
                cp_async_wait_all();
            }
            __syncwarp();
        }
    }

    up_sum0 = warp_sum(up_sum0);
    gate_sum0 = warp_sum(gate_sum0);
    up_sum1 = warp_sum(up_sum1);
    gate_sum1 = warp_sum(gate_sum1);
    if (lane == 0) {
        const float silu0 =
            __fdividef(gate_sum0, 1.0f + __expf(-gate_sum0));
        const float silu1 =
            __fdividef(gate_sum1, 1.0f + __expf(-gate_sum1));
        g.silu_out.raw_ptr[row0] =
            __float2bfloat16_rn(up_sum0 * silu0);
        g.silu_out.raw_ptr[row1] =
            __float2bfloat16_rn(up_sum1 * silu1);
    }
}

template <int L2PrefetchBytes, bool PdlTrigger = false,
          int PdlTriggerPair = -1>
__device__ __forceinline__ void
upgate_compute_two_rows(const globals &g, const bf16 *normalized,
                        const bf16 *up_layer, const bf16 *gate_layer,
                        int row0, int row1, int lane, bool trigger_owner = false) {
    static_assert(PdlTriggerPair == -1 || PdlTriggerPair == 512 ||
                  PdlTriggerPair == 768 || PdlTriggerPair == 896 ||
                  PdlTriggerPair == 960 || PdlTriggerPair == 1024);
    const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
    const bf16 *up_row1 = up_layer + row1 * globals::hidden_dim;
    const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
    const bf16 *gate_row1 = gate_layer + row1 * globals::hidden_dim;
    constexpr int hidden_pairs = globals::hidden_dim / 2;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    for (int pair = lane; pair < hidden_pairs; pair += 32) {
        if constexpr (PdlTrigger && PdlTriggerPair >= 0 &&
                      PdlTriggerPair < hidden_pairs) {
            if (trigger_owner && lane == 0 && pair == PdlTriggerPair) {
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }
        const float2 x = load_bf16x2(normalized, pair);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(up_row0, pair),
            up_sum0);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(gate_row0, pair),
            gate_sum0);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(up_row1, pair),
            up_sum1);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(gate_row1, pair),
            gate_sum1);
    }
    if constexpr (PdlTrigger && PdlTriggerPair == hidden_pairs) {
        if (trigger_owner && lane == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    up_sum0 = warp_sum(up_sum0);
    up_sum1 = warp_sum(up_sum1);
    gate_sum0 = warp_sum(gate_sum0);
    gate_sum1 = warp_sum(gate_sum1);
    if (lane == 0) {
        const float silu0 =
            __fdividef(gate_sum0, 1.0f + __expf(-gate_sum0));
        const float silu1 =
            __fdividef(gate_sum1, 1.0f + __expf(-gate_sum1));
        g.silu_out.raw_ptr[row0] =
            __float2bfloat16_rn(up_sum0 * silu0);
        g.silu_out.raw_ptr[row1] =
            __float2bfloat16_rn(up_sum1 * silu1);
    }
}

template <int L2PrefetchBytes>
__device__ __forceinline__ void
upgate_compute_one_row(const globals &g, const bf16 *normalized,
                       const bf16 *up_layer, const bf16 *gate_layer,
                       int row, int lane) {
    const bf16 *up_row = up_layer + row * globals::hidden_dim;
    const bf16 *gate_row = gate_layer + row * globals::hidden_dim;
    constexpr int hidden_pairs = globals::hidden_dim / 2;
    float up_sum = 0.0f;
    float gate_sum = 0.0f;
    for (int pair = lane; pair < hidden_pairs; pair += 32) {
        const float2 x = load_bf16x2(normalized, pair);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(up_row, pair), up_sum);
        accumulate_pair(
            x, load_weight_bf16x2<L2PrefetchBytes>(gate_row, pair),
            gate_sum);
    }
    up_sum = warp_sum(up_sum);
    gate_sum = warp_sum(gate_sum);
    if (lane == 0) {
        const float silu =
            __fdividef(gate_sum, 1.0f + __expf(-gate_sum));
        g.silu_out.raw_ptr[row] =
            __float2bfloat16_rn(up_sum * silu);
    }
}

template <int WeightPolicy>
__device__ __forceinline__ void
upgate_compute_two_rows_v4(const globals &g, const bf16 *normalized,
                           const bf16 *up_layer, const bf16 *gate_layer,
                           int row0, int row1, int lane) {
    static_assert(WeightPolicy == 0 || WeightPolicy == -3 ||
                  WeightPolicy == -5);
    const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
    const bf16 *up_row1 = up_layer + row1 * globals::hidden_dim;
    const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
    const bf16 *gate_row1 = gate_layer + row1 * globals::hidden_dim;
    constexpr int hidden_quads = globals::hidden_dim / 4;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    for (int quad = lane; quad < hidden_quads; quad += 32) {
        const float4_split x = load_bf16x4(normalized, quad);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(up_row0, quad), up_sum0);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(gate_row0, quad), gate_sum0);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(up_row1, quad), up_sum1);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(gate_row1, quad), gate_sum1);
    }
    up_sum0 = warp_sum(up_sum0);
    up_sum1 = warp_sum(up_sum1);
    gate_sum0 = warp_sum(gate_sum0);
    gate_sum1 = warp_sum(gate_sum1);
    if (lane == 0) {
        const float silu0 =
            __fdividef(gate_sum0, 1.0f + __expf(-gate_sum0));
        const float silu1 =
            __fdividef(gate_sum1, 1.0f + __expf(-gate_sum1));
        g.silu_out.raw_ptr[row0] =
            __float2bfloat16_rn(up_sum0 * silu0);
        g.silu_out.raw_ptr[row1] =
            __float2bfloat16_rn(up_sum1 * silu1);
    }
}

template <int WeightPolicy>
__device__ __forceinline__ void
upgate_compute_one_row_v4(const globals &g, const bf16 *normalized,
                          const bf16 *up_layer, const bf16 *gate_layer,
                          int row, int lane) {
    static_assert(WeightPolicy == 0 || WeightPolicy == -3 ||
                  WeightPolicy == -5);
    const bf16 *up_row = up_layer + row * globals::hidden_dim;
    const bf16 *gate_row = gate_layer + row * globals::hidden_dim;
    constexpr int hidden_quads = globals::hidden_dim / 4;
    float up_sum = 0.0f;
    float gate_sum = 0.0f;
    for (int quad = lane; quad < hidden_quads; quad += 32) {
        const float4_split x = load_bf16x4(normalized, quad);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(up_row, quad), up_sum);
        accumulate_quad(
            x, load_weight_bf16x4<WeightPolicy>(gate_row, quad), gate_sum);
    }
    up_sum = warp_sum(up_sum);
    gate_sum = warp_sum(gate_sum);
    if (lane == 0) {
        const float silu =
            __fdividef(gate_sum, 1.0f + __expf(-gate_sum));
        g.silu_out.raw_ptr[row] =
            __float2bfloat16_rn(up_sum * silu);
    }
}

// Exact one-wave 132-CTA mapping.  For the common 62-row slice, warps 0..30
// each own two rows and warp 31 exits after the final block barrier.  The first
// eight CTAs own 63 rows and use warp 31 for the single remainder.  This keeps
// the two-row warps at four independent weight streams instead of spreading
// the remainders across two low-ILP warps.
template <typename Config, int L2PrefetchBytes = 0, bool AllWarps = false,
          bool PdlTrigger = false, bool PdlWait = false,
          int PdlTriggerPair = -1, int PdlPrefetchLines = 0,
          int PdlPrefetchMatrices = 3, int PdlDemandFetchBytes = 0,
          bool SignalSplitReady = false,
          int PdlPrefetchPolicy = prefetch_policy_l2,
          bool WideLoad = false>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
upgate_132_pairwarp(const __grid_constant__ globals g) {
    static_assert(Config::balanced_grid);
    static_assert(Config::ctas == 132);
    static_assert(Config::warps == 32);
    static_assert(Config::rows_per_warp == 2);
    static_assert(PdlPrefetchLines == 0 || PdlPrefetchLines == 1 ||
                  PdlPrefetchLines == 2 || PdlPrefetchLines == 4 ||
                  PdlPrefetchLines == 8 || PdlPrefetchLines == 16);
    static_assert(PdlPrefetchMatrices >= 1 && PdlPrefetchMatrices <= 3);
    static_assert(PdlDemandFetchBytes == 0 ||
                  PdlDemandFetchBytes == 128 ||
                  PdlDemandFetchBytes == 256);
    static_assert(!WideLoad || L2PrefetchBytes == 0 ||
                  L2PrefetchBytes == -3 || L2PrefetchBytes == -5);
    static_assert(!WideLoad || PdlTriggerPair == -1);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];
    __shared__ volatile unsigned demand_warm_sinks[Config::warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 62 + (block < 8 ? block : 8);
    const int row_count = 62 + (block < 8);
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;

    if constexpr (PdlWait && PdlPrefetchLines > 0) {
        int prefetch_row0 = -1;
        int prefetch_row1 = -1;
        if (warp < 30) {
            prefetch_row0 = row_begin + 2 * warp;
            prefetch_row1 = prefetch_row0 + 1;
        } else if (warp == 30) {
            prefetch_row0 = row_begin + 60;
            if (row_count == 63) {
                prefetch_row1 = row_begin + 62;
            }
        } else {
            prefetch_row0 = row_begin + 61;
        }
        constexpr int fetch_bytes = PdlDemandFetchBytes == 0
                                        ? 128
                                        : PdlDemandFetchBytes;
        constexpr int values_per_l2_line = fetch_bytes / sizeof(bf16);
        unsigned demand_bits = 0;
#pragma unroll
        for (int line = lane; line < PdlPrefetchLines; line += 32) {
            const int value_offset = line * values_per_l2_line;
            if constexpr ((PdlPrefetchMatrices & 1) != 0) {
                const bf16 *address =
                    up_layer + prefetch_row0 * globals::hidden_dim +
                    value_offset;
                if constexpr (PdlDemandFetchBytes == 0) {
                    prefetch_global<PdlPrefetchPolicy>(address);
                } else {
                    demand_bits ^=
                        demand_warm_l2<PdlDemandFetchBytes>(address);
                }
            }
            if constexpr ((PdlPrefetchMatrices & 2) != 0) {
                const bf16 *address =
                    gate_layer + prefetch_row0 * globals::hidden_dim +
                    value_offset;
                if constexpr (PdlDemandFetchBytes == 0) {
                    prefetch_global<PdlPrefetchPolicy>(address);
                } else {
                    demand_bits ^=
                        demand_warm_l2<PdlDemandFetchBytes>(address);
                }
            }
            if (prefetch_row1 >= 0) {
                if constexpr ((PdlPrefetchMatrices & 1) != 0) {
                    const bf16 *address =
                        up_layer + prefetch_row1 * globals::hidden_dim +
                        value_offset;
                    if constexpr (PdlDemandFetchBytes == 0) {
                        prefetch_global<PdlPrefetchPolicy>(address);
                    } else {
                        demand_bits ^=
                            demand_warm_l2<PdlDemandFetchBytes>(address);
                    }
                }
                if constexpr ((PdlPrefetchMatrices & 2) != 0) {
                    const bf16 *address =
                        gate_layer + prefetch_row1 * globals::hidden_dim +
                        value_offset;
                    if constexpr (PdlDemandFetchBytes == 0) {
                        prefetch_global<PdlPrefetchPolicy>(address);
                    } else {
                        demand_bits ^=
                            demand_warm_l2<PdlDemandFetchBytes>(address);
                    }
                }
            }
        }
        if constexpr (PdlDemandFetchBytes != 0) {
#pragma unroll
            for (int delta = 16; delta > 0; delta >>= 1) {
                demand_bits ^=
                    __shfl_down_sync(0xffffffffu, demand_bits, delta);
            }
            if (lane == 0) {
                demand_warm_sinks[warp] = demand_bits;
            }
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
    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    const int norm_pairs = globals::hidden_dim / 2;

    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    // 8192 = 8 * 63 + 124 * 62.  This removes the two integer divisions and
    // generic row-end predicates from the balanced-grid implementation.
    if constexpr (!AllWarps) {
        if (warp < 31) {
            const int row0 = row_begin + 2 * warp;
            upgate_compute_two_rows<L2PrefetchBytes>(
                g, normalized, up_layer, gate_layer, row0, row0 + 1,
                lane);
        } else if (row_count == 63) {
            upgate_compute_one_row<L2PrefetchBytes>(
                g, normalized, up_layer, gate_layer, row_begin + 62,
                lane);
        }
    } else {
        if (warp < 30) {
            const int row0 = row_begin + 2 * warp;
            if constexpr (WideLoad) {
                upgate_compute_two_rows_v4<L2PrefetchBytes>(
                    g, normalized, up_layer, gate_layer, row0, row0 + 1,
                    lane);
            } else {
                upgate_compute_two_rows<L2PrefetchBytes, PdlTrigger,
                                        PdlTriggerPair>(
                    g, normalized, up_layer, gate_layer, row0, row0 + 1,
                    lane, warp == 0);
            }
        } else if (warp == 30) {
            if (row_count == 63) {
                if constexpr (WideLoad) {
                    upgate_compute_two_rows_v4<L2PrefetchBytes>(
                        g, normalized, up_layer, gate_layer, row_begin + 60,
                        row_begin + 62, lane);
                } else {
                    upgate_compute_two_rows<L2PrefetchBytes>(
                        g, normalized, up_layer, gate_layer, row_begin + 60,
                        row_begin + 62, lane);
                }
            } else {
                if constexpr (WideLoad) {
                    upgate_compute_one_row_v4<L2PrefetchBytes>(
                        g, normalized, up_layer, gate_layer, row_begin + 60,
                        lane);
                } else {
                    upgate_compute_one_row<L2PrefetchBytes>(
                        g, normalized, up_layer, gate_layer, row_begin + 60,
                        lane);
                }
            }
        } else {
            if constexpr (WideLoad) {
                upgate_compute_one_row_v4<L2PrefetchBytes>(
                    g, normalized, up_layer, gate_layer, row_begin + 61,
                    lane);
            } else {
                upgate_compute_one_row<L2PrefetchBytes>(
                    g, normalized, up_layer, gate_layer, row_begin + 61,
                    lane);
            }
        }
    }

    // Publish the contiguous K slices produced by this CTA.  The standalone
    // CUDA Graph initializes every Bar element to one million.  Each layer
    // owns four otherwise-unused flag words, and a split becomes ready after
    // exactly 2048 scalar SiLU outputs have been published.
    if constexpr (SignalSplitReady) {
        __syncthreads();
        if (tid == 0) {
            constexpr int split_cols = globals::hidden_dim;
            constexpr int split_flag_offset = globals::hidden_dim;
            __threadfence();
#pragma unroll
            for (int split = 0; split < 4; ++split) {
                const int split_begin = split * split_cols;
                const int split_end = split_begin + split_cols;
                const int overlap_begin = max(row_begin, split_begin);
                const int overlap_end = min(row_begin + row_count, split_end);
                const int produced = max(0, overlap_end - overlap_begin);
                if (produced > 0) {
                    atomicAdd(g.Bar.raw_ptr + split_flag_offset + layer * 4 +
                                  split,
                              static_cast<unsigned>(produced));
                }
            }
        }
    }
}

// Fuse the accepted one-wave up+gate and down-projection mappings.  The
// 132x1024 launch is exactly one resident CTA per H100 SM.  A software grid
// barrier is therefore safe and replaces the opcode-5 -> opcode-6 kernel
// boundary while preserving the original BF16/FP32 arithmetic on both sides.
template <bool TriggerDownstreamAtEpilogue>
__launch_bounds__(1024, 1) __global__ void
upgate_down_fused_132(const __grid_constant__ globals g) {
    constexpr int grid_ctas = 132;
    constexpr int warps = 32;
    constexpr int barrier_heads =
        globals::num_attention_heads + 2 * globals::num_kv_heads;
    constexpr int barrier_opcode_pages = 10;
    constexpr int fused_barrier_page = 9;

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 62 + (block < 8 ? block : 8);
    const int row_count = 62 + (block < 8);
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;

    // Retain the accepted 128-byte, both-matrix L2 hint before the incoming
    // opcode-4 dependency wait.
    int prefetch_row0 = -1;
    int prefetch_row1 = -1;
    if (warp < 30) {
        prefetch_row0 = row_begin + 2 * warp;
        prefetch_row1 = prefetch_row0 + 1;
    } else if (warp == 30) {
        prefetch_row0 = row_begin + 60;
        if (row_count == 63) {
            prefetch_row1 = row_begin + 62;
        }
    } else {
        prefetch_row0 = row_begin + 61;
    }
    if (lane == 0) {
        prefetch_global<prefetch_policy_l2>(
            up_layer + prefetch_row0 * globals::hidden_dim);
        prefetch_global<prefetch_policy_l2>(
            gate_layer + prefetch_row0 * globals::hidden_dim);
        if (prefetch_row1 >= 0) {
            prefetch_global<prefetch_policy_l2>(
                up_layer + prefetch_row1 * globals::hidden_dim);
            prefetch_global<prefetch_policy_l2>(
                gate_layer + prefetch_row1 * globals::hidden_dim);
        }
    }
    cudaGridDependencySynchronize();

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    constexpr int norm_pairs = globals::hidden_dim / 2;
    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += 1024) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += 1024) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    if (warp < 30) {
        const int row0 = row_begin + 2 * warp;
        upgate_compute_two_rows<-5>(
            g, normalized, up_layer, gate_layer, row0, row0 + 1, lane);
    } else if (warp == 30) {
        if (row_count == 63) {
            upgate_compute_two_rows<-5>(
                g, normalized, up_layer, gate_layer, row_begin + 60,
                row_begin + 62, lane);
        } else {
            upgate_compute_one_row<-5>(
                g, normalized, up_layer, gate_layer, row_begin + 60, lane);
        }
    } else {
        upgate_compute_one_row<-5>(
            g, normalized, up_layer, gate_layer, row_begin + 61, lane);
    }

    // The harness resets this otherwise-unused Bar page to zero before every
    // forward.  All 132 blocks are simultaneously resident, so the last
    // arrival can publish the completed SiLU vector without a second launch.
    __syncthreads();
    unsigned *fused_barrier =
        g.Bar.raw_ptr +
        layer * barrier_opcode_pages * barrier_heads +
        fused_barrier_page * barrier_heads;
    if (tid == 0) {
        __threadfence();
        const unsigned old = atomicAdd(fused_barrier, 1u);
        if (old == grid_ctas - 1) {
            __threadfence();
            atomicExch(fused_barrier + 1, 1u);
        } else {
            while (atomicAdd(fused_barrier + 1, 0u) == 0u) {
                __nanosleep(64);
            }
            __threadfence();
        }
    }
    __syncthreads();

    if constexpr (!TriggerDownstreamAtEpilogue) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    const int down_row_begin =
        block * globals::hidden_dim / grid_ctas;
    const int down_row_end =
        (block + 1) * globals::hidden_dim / grid_ctas;
    const int down_row_count = down_row_end - down_row_begin;
    float down_sum = 0.0f;
    if (warp < down_row_count) {
        const int row = down_row_begin + warp;
        const bf16 *weight_row =
            g.down_weights.raw_ptr +
            layer * globals::hidden_dim * globals::intermediate_dim +
            row * globals::intermediate_dim;
        constexpr int input_quads = globals::intermediate_dim / 4;
        for (int quad = lane; quad < input_quads; quad += 32) {
            accumulate_quad(
                load_bf16x4(g.silu_out.raw_ptr, quad),
                load_weight_bf16x4<-3>(weight_row, quad), down_sum);
        }
    }

    if constexpr (TriggerDownstreamAtEpilogue) {
        __syncthreads();
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    if (warp < down_row_count) {
        down_sum = warp_sum(down_sum);
        if (lane == 0) {
            const int row = down_row_begin + warp;
            const float residual =
                __bfloat162float(g.hidden_states.raw_ptr[row]);
            g.hidden_states.raw_ptr[row] =
                __float2bfloat16_rn(residual + down_sum);
        }
    }
}

// Persistent-style temporal ordering for the split down-projection edge.
// Every CTA contributes 15 or 16 rows to each 2048-row split, and all CTAs
// finish split N before moving to N+1.  This is deliberately different from
// the fast contiguous pair-warp mapping above: it makes split 0 visible after
// roughly one quarter of opcode5 so opcode6 can overlap useful GEMV work with
// the remaining three producer phases.
template <typename Config, int L2PrefetchBytes = -5>
__launch_bounds__(Config::threads, 1) __global__ void
upgate_132_split4_pipeline(const __grid_constant__ globals g) {
    static_assert(Config::ctas == 132);
    static_assert(Config::threads == 1024);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    constexpr int split_count =
        globals::intermediate_dim / globals::hidden_dim;
    constexpr int rows_floor = globals::hidden_dim / Config::ctas;
    constexpr int rows_remainder = globals::hidden_dim % Config::ctas;
    constexpr int split_flag_offset = globals::hidden_dim;
    static_assert(split_count == 4);
    static_assert(rows_floor == 15);
    static_assert(rows_remainder == 68);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int block = static_cast<int>(blockIdx.x);
    const int rows_this_cta = rows_floor + (block < rows_remainder);
    const int row_in_split_begin =
        block * rows_floor + min(block, rows_remainder);
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;

    // Preserve the winning one-line hint coverage from the contiguous
    // producer: issue one up and one gate hint for every output row before
    // waiting on opcode4, even though those rows are consumed in four phases.
    if (warp < rows_this_cta && lane == 0) {
#pragma unroll
        for (int split = 0; split < split_count; ++split) {
            const int row = split * globals::hidden_dim + row_in_split_begin +
                            warp;
            prefetch_l2(up_layer + row * globals::hidden_dim);
            prefetch_l2(gate_layer + row * globals::hidden_dim);
        }
    }
    cudaGridDependencySynchronize();
    if (tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    constexpr int norm_pairs = globals::hidden_dim / 2;
    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

#pragma unroll
    for (int split = 0; split < split_count; ++split) {
        if (warp < rows_this_cta) {
            const int row = split * globals::hidden_dim +
                            row_in_split_begin + warp;
            upgate_compute_one_row<L2PrefetchBytes>(
                g, normalized, up_layer, gate_layer, row, lane);
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
            atomicAdd(g.Bar.raw_ptr + split_flag_offset +
                          layer * split_count + split,
                      static_cast<unsigned>(rows_this_cta));
        }
    }
}

// Two-phase compromise: all 32 warps remain useful in each phase, while the
// first half of the intermediate vector still releases two down-projection
// splits before the second half of opcode5.  PairRows=false assigns one row
// per warp (31/32 active warps); PairRows=true keeps four independent weight
// streams in 16 warps as an ILP-versus-occupancy control.
template <typename Config, bool PairRows, int L2PrefetchBytes = -5>
__launch_bounds__(Config::threads, 1) __global__ void
upgate_132_split2_pipeline(const __grid_constant__ globals g) {
    static_assert(Config::ctas == 132);
    static_assert(Config::threads == 1024);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    constexpr int split_count =
        globals::intermediate_dim / globals::hidden_dim;
    constexpr int phase_count = 2;
    constexpr int phase_rows = globals::intermediate_dim / phase_count;
    constexpr int rows_floor = phase_rows / Config::ctas;
    constexpr int rows_remainder = phase_rows % Config::ctas;
    constexpr int split_flag_offset = globals::hidden_dim;
    static_assert(split_count == 4);
    static_assert(rows_floor == 31);
    static_assert(rows_remainder == 4);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int block = static_cast<int>(blockIdx.x);
    const int rows_this_cta = rows_floor + (block < rows_remainder);
    const int row_in_phase_begin =
        block * rows_floor + min(block, rows_remainder);
    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;

    // Match the original 8192-row one-line hint footprint.
    if (lane == 0) {
#pragma unroll
        for (int phase = 0; phase < phase_count; ++phase) {
            for (int local_row = warp; local_row < rows_this_cta;
                 local_row += Config::warps) {
                const int row = phase * phase_rows + row_in_phase_begin +
                                local_row;
                prefetch_l2(up_layer + row * globals::hidden_dim);
                prefetch_l2(gate_layer + row * globals::hidden_dim);
            }
        }
    }
    cudaGridDependencySynchronize();
    if (tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    constexpr int norm_pairs = globals::hidden_dim / 2;
    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

#pragma unroll
    for (int phase = 0; phase < phase_count; ++phase) {
        const int phase_begin = phase * phase_rows + row_in_phase_begin;
        if constexpr (PairRows) {
            const int pair_begin = 2 * warp;
            if (pair_begin + 1 < rows_this_cta) {
                upgate_compute_two_rows<L2PrefetchBytes>(
                    g, normalized, up_layer, gate_layer,
                    phase_begin + pair_begin,
                    phase_begin + pair_begin + 1, lane);
            } else if (pair_begin < rows_this_cta) {
                upgate_compute_one_row<L2PrefetchBytes>(
                    g, normalized, up_layer, gate_layer,
                    phase_begin + pair_begin, lane);
            }
        } else if (warp < rows_this_cta) {
            upgate_compute_one_row<L2PrefetchBytes>(
                g, normalized, up_layer, gate_layer, phase_begin + warp,
                lane);
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence();
#pragma unroll
            for (int split_in_phase = 0; split_in_phase < 2;
                 ++split_in_phase) {
                const int split = phase * 2 + split_in_phase;
                const int split_begin = split * globals::hidden_dim;
                const int split_end = split_begin + globals::hidden_dim;
                const int overlap_begin = max(phase_begin, split_begin);
                const int overlap_end =
                    min(phase_begin + rows_this_cta, split_end);
                const int produced = max(0, overlap_end - overlap_begin);
                if (produced > 0) {
                    atomicAdd(g.Bar.raw_ptr + split_flag_offset +
                                  layer * split_count + split,
                              static_cast<unsigned>(produced));
                }
            }
        }
    }
}

// Retain only the first 64 or 128 K values in registers across the incoming
// PDL wait.  Unlike whole-matrix SMEM staging, this adds no SMEM weight pass:
// the early LDG result is consumed directly by the dot product, and the rest
// of the row keeps the proven direct SIMT load path.
template <typename Config, int PrefixRounds, bool PdlTrigger = true>
__maxnreg__(48) __global__ void
upgate_132_register_prefix_pdl(const __grid_constant__ globals g) {
    static_assert(Config::balanced_grid);
    static_assert(Config::ctas == 132);
    static_assert(Config::warps == 32);
    static_assert(Config::rows_per_warp == 2);
    static_assert(PrefixRounds == 1 || PrefixRounds == 2);

    __shared__ __align__(16) bf16 normalized[globals::hidden_dim];
    __shared__ float warp_sums[Config::warps];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int block = static_cast<int>(blockIdx.x);
    const int row_begin = block * 62 + (block < 8 ? block : 8);
    const int row_count = 62 + (block < 8);

    int row0;
    int row1 = 0;
    bool has_row1 = false;
    if (warp < 30) {
        row0 = row_begin + 2 * warp;
        row1 = row0 + 1;
        has_row1 = true;
    } else if (warp == 30) {
        row0 = row_begin + 60;
        if (row_count == 63) {
            row1 = row_begin + 62;
            has_row1 = true;
        }
    } else {
        row0 = row_begin + 61;
    }

    const bf16 *up_layer = g.up_weights.raw_ptr +
                           layer * globals::intermediate_dim *
                               globals::hidden_dim;
    const bf16 *gate_layer = g.gate_weights.raw_ptr +
                             layer * globals::intermediate_dim *
                                 globals::hidden_dim;
    const bf16 *up_row0 = up_layer + row0 * globals::hidden_dim;
    const bf16 *gate_row0 = gate_layer + row0 * globals::hidden_dim;
    const bf16 *up_row1 = has_row1
                              ? up_layer + row1 * globals::hidden_dim
                              : up_row0;
    const bf16 *gate_row1 = has_row1
                                ? gate_layer + row1 * globals::hidden_dim
                                : gate_row0;

    unsigned prefix_up0[PrefixRounds];
    unsigned prefix_gate0[PrefixRounds];
    unsigned prefix_up1[PrefixRounds];
    unsigned prefix_gate1[PrefixRounds];
#pragma unroll
    for (int round = 0; round < PrefixRounds; ++round) {
        const int pair = round * 32 + lane;
        prefix_up0[round] =
            load_weight_bf16x2_bits_na256(up_row0, pair);
        prefix_gate0[round] =
            load_weight_bf16x2_bits_na256(gate_row0, pair);
        if (has_row1) {
            prefix_up1[round] =
                load_weight_bf16x2_bits_na256(up_row1, pair);
            prefix_gate1[round] =
                load_weight_bf16x2_bits_na256(gate_row1, pair);
        } else {
            prefix_up1[round] = 0;
            prefix_gate1[round] = 0;
        }
    }

    cudaGridDependencySynchronize();
    if constexpr (PdlTrigger) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    const bf16 *hidden = g.hidden_states.raw_ptr;
    const bf16 *norm =
        g.mlp_norm_weights.raw_ptr + layer * globals::hidden_dim;
    constexpr int norm_pairs = globals::hidden_dim / 2;
    float square_sum = 0.0f;
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
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
    for (int pair = tid; pair < norm_pairs; pair += Config::threads) {
        const float2 value = load_bf16x2(hidden, pair);
        const float2 scale = load_bf16x2(norm, pair);
        reinterpret_cast<bf16x2 *>(normalized)[pair] =
            __floats2bfloat162_rn(value.x * scale.x * inv_rms,
                                 value.y * scale.y * inv_rms);
    }
    __syncthreads();

    if (has_row1) {
        float up_sum0 = 0.0f;
        float gate_sum0 = 0.0f;
        float up_sum1 = 0.0f;
        float gate_sum1 = 0.0f;
#pragma unroll
        for (int round = 0; round < PrefixRounds; ++round) {
            const int pair = round * 32 + lane;
            const float2 x = load_bf16x2(normalized, pair);
            accumulate_pair(x, bf16x2_bits_to_float2(prefix_up0[round]),
                            up_sum0);
            accumulate_pair(x,
                            bf16x2_bits_to_float2(prefix_gate0[round]),
                            gate_sum0);
            accumulate_pair(x, bf16x2_bits_to_float2(prefix_up1[round]),
                            up_sum1);
            accumulate_pair(x,
                            bf16x2_bits_to_float2(prefix_gate1[round]),
                            gate_sum1);
        }
        for (int pair = PrefixRounds * 32 + lane; pair < norm_pairs;
             pair += 32) {
            const float2 x = load_bf16x2(normalized, pair);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(up_row0, pair), up_sum0);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(gate_row0, pair), gate_sum0);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(up_row1, pair), up_sum1);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(gate_row1, pair), gate_sum1);
        }
        up_sum0 = warp_sum(up_sum0);
        gate_sum0 = warp_sum(gate_sum0);
        up_sum1 = warp_sum(up_sum1);
        gate_sum1 = warp_sum(gate_sum1);
        if (lane == 0) {
            const float silu0 =
                __fdividef(gate_sum0, 1.0f + __expf(-gate_sum0));
            const float silu1 =
                __fdividef(gate_sum1, 1.0f + __expf(-gate_sum1));
            g.silu_out.raw_ptr[row0] =
                __float2bfloat16_rn(up_sum0 * silu0);
            g.silu_out.raw_ptr[row1] =
                __float2bfloat16_rn(up_sum1 * silu1);
        }
    } else {
        float up_sum = 0.0f;
        float gate_sum = 0.0f;
#pragma unroll
        for (int round = 0; round < PrefixRounds; ++round) {
            const int pair = round * 32 + lane;
            const float2 x = load_bf16x2(normalized, pair);
            accumulate_pair(x, bf16x2_bits_to_float2(prefix_up0[round]),
                            up_sum);
            accumulate_pair(x,
                            bf16x2_bits_to_float2(prefix_gate0[round]),
                            gate_sum);
        }
        for (int pair = PrefixRounds * 32 + lane; pair < norm_pairs;
             pair += 32) {
            const float2 x = load_bf16x2(normalized, pair);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(up_row0, pair), up_sum);
            accumulate_pair(
                x, load_weight_bf16x2<-5>(gate_row0, pair), gate_sum);
        }
        up_sum = warp_sum(up_sum);
        gate_sum = warp_sum(gate_sum);
        if (lane == 0) {
            const float silu =
                __fdividef(gate_sum, 1.0f + __expf(-gate_sum));
            g.silu_out.raw_ptr[row0] =
                __float2bfloat16_rn(up_sum * silu);
        }
    }
}

template <typename Config>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
downproj_cached(
    const __grid_constant__ globals g) {
    __shared__ __align__(16) bf16 input_cache[globals::intermediate_dim];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);

    for (int col = tid; col < globals::intermediate_dim;
         col += Config::threads) {
        input_cache[col] = g.silu_out.raw_ptr[col];
    }
    __syncthreads();

    const int first_row =
        static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const int input_pairs = globals::intermediate_dim / 2;
    const bf16 *weight_layer = g.down_weights.raw_ptr +
                               layer * globals::hidden_dim *
                                   globals::intermediate_dim;

#pragma unroll
    for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
        const int row = first_row + row_iter * Config::warps;
        const bf16 *weight_row =
            weight_layer + row * globals::intermediate_dim;
        float sum = 0.0f;
        for (int pair = lane; pair < input_pairs; pair += 32) {
            accumulate_pair(load_bf16x2(input_cache, pair),
                            load_bf16x2(weight_row, pair), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            const float residual =
                __bfloat162float(g.hidden_states.raw_ptr[row]);
            g.hidden_states.raw_ptr[row] =
                __float2bfloat16_rn(residual + sum);
        }
    }
}

template <typename Config>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
downproj(
    const __grid_constant__ globals g) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);

    const int first_row =
        static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const int input_pairs = globals::intermediate_dim / 2;
    const bf16 *input = g.silu_out.raw_ptr;
    const bf16 *weight_layer = g.down_weights.raw_ptr +
                               layer * globals::hidden_dim *
                                   globals::intermediate_dim;

    float sums[Config::rows_per_warp] = {};
    for (int pair = lane; pair < input_pairs; pair += 32) {
        const float2 x = load_bf16x2(input, pair);
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
            const int row = first_row + row_iter * Config::warps;
            const bf16 *weight_row =
                weight_layer + row * globals::intermediate_dim;
            accumulate_pair(x, load_bf16x2(weight_row, pair), sums[row_iter]);
        }
    }
#pragma unroll
    for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
        sums[row_iter] = warp_sum(sums[row_iter]);
    }
    if (lane == 0) {
#pragma unroll
        for (int row_iter = 0; row_iter < Config::rows_per_warp; ++row_iter) {
            const int row = first_row + row_iter * Config::warps;
            const float residual =
                __bfloat162float(g.hidden_states.raw_ptr[row]);
            g.hidden_states.raw_ptr[row] =
                __float2bfloat16_rn(residual + sums[row_iter]);
        }
    }
}

// Cold-weight control for the one-wave 16-warp x 1-row mapping.  A b64 load
// halves the dot-loop/address instruction count while every warp still issues
// naturally coalesced contiguous requests.  Optional L2 prefetching covers a
// compile-time prefix of each 16-KiB row before the consuming loop.
template <typename Config, int WeightPolicy, int PrefetchLines = 0,
          bool PdlWait = false, bool PdlTrigger = false,
          int PdlTriggerQuad = -1,
          int PdlPrefetchPolicy = prefetch_policy_l2>
__launch_bounds__(Config::threads, Config::min_blocks_per_sm) __global__ void
downproj_v4(const __grid_constant__ globals g) {
    static_assert(Config::rows_per_warp == 1);
    static_assert(PrefetchLines == 0 || PrefetchLines == 1 ||
                  PrefetchLines == 2 || PrefetchLines == 4 ||
                  PrefetchLines == 8 || PrefetchLines == 16 ||
                  PrefetchLines == 32 || PrefetchLines == 64 ||
                  PrefetchLines == 128);
    static_assert(PdlTriggerQuad == -1 || PdlTriggerQuad == 1024 ||
                  PdlTriggerQuad == 1536 || PdlTriggerQuad == 1792 ||
                  PdlTriggerQuad == 1920 || PdlTriggerQuad == 2048);

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int layer = instruction_layer(g);
    const int row = static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const bf16 *input = g.silu_out.raw_ptr;
    const bf16 *weight_row =
        g.down_weights.raw_ptr +
        layer * globals::hidden_dim * globals::intermediate_dim +
        row * globals::intermediate_dim;

    if constexpr (PrefetchLines > 0) {
        constexpr int values_per_l2_line = 128 / sizeof(bf16);
#pragma unroll
        for (int line = lane; line < PrefetchLines; line += 32) {
            prefetch_global<PdlPrefetchPolicy>(
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

    float sum = 0.0f;
    constexpr int input_quads = globals::intermediate_dim / 4;
    for (int quad = lane; quad < input_quads; quad += 32) {
        if constexpr (PdlTrigger && PdlTriggerQuad >= 0 &&
                      PdlTriggerQuad < input_quads) {
            if (threadIdx.x == 0 && quad == PdlTriggerQuad) {
                cudaTriggerProgrammaticLaunchCompletion();
            }
        }
        accumulate_quad(load_bf16x4(input, quad),
                        load_weight_bf16x4<WeightPolicy>(weight_row, quad),
                        sum);
    }
    if constexpr (PdlTrigger && PdlTriggerQuad == input_quads) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        const float residual = __bfloat162float(g.hidden_states.raw_ptr[row]);
        g.hidden_states.raw_ptr[row] =
            __float2bfloat16_rn(residual + sum);
    }
}

// Fine-grained opcode5->opcode6 pipeline matching the persistent VM's four
// reduction splits.  PDL starts all 132 CTAs while opcode5 is still resident;
// each CTA waits only for its own 2048-column SiLU slice, computes that
// partial dot, and publishes it.  The fourth arrival for an output row adds
// the partials in deterministic split order and applies the residual.
//
// 32 registers x 512 threads leaves exactly half of an H100 register file for
// the 48-register x 1024-thread opcode5 producer.  No shared memory is used.
__maxnreg__(32) __global__ void
downproj_split4_pdl(const __grid_constant__ globals g) {
    constexpr unsigned barrier_base = 1000000u;
    constexpr int split_cols = globals::hidden_dim;
    constexpr int split_count =
        globals::intermediate_dim / globals::hidden_dim;
    constexpr int split_flag_offset = globals::hidden_dim;
    static_assert(split_count == 4);

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int *instruction =
        g.instructions.raw_ptr + static_cast<int>(blockIdx.x) * 32;
    const int layer = instruction[1];
    const int start_block = instruction[2];
    const int end_block = instruction[3];
    const int split = instruction[4];

    if (tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
        const volatile unsigned *ready =
            g.Bar.raw_ptr + split_flag_offset + layer * split_count + split;
        while (*ready < barrier_base + split_cols) {
            __nanosleep(64);
        }
    }
    __syncthreads();

    const bf16 *input = g.silu_out.raw_ptr + split * split_cols;
    const bf16 *weight_layer =
        g.down_weights.raw_ptr +
        layer * globals::hidden_dim * globals::intermediate_dim;
    float *partials = g.attn_out_intermediates.raw_ptr;

    for (int output_block = start_block; output_block < end_block;
         ++output_block) {
        const int row = output_block * 16 + warp;
        const bf16 *weight_row =
            weight_layer + row * globals::intermediate_dim +
            split * split_cols;

        float sum = 0.0f;
        constexpr int split_quads = split_cols / 4;
        for (int quad = lane; quad < split_quads; quad += 32) {
            accumulate_quad(load_bf16x4(input, quad),
                            load_weight_bf16x4<-3>(weight_row, quad), sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            partials[split * globals::hidden_dim + row] = sum;
            __threadfence();
            unsigned *counter = g.Bar.raw_ptr + row;
            const unsigned old = atomicAdd(counter, 1u);
            const unsigned final_old = barrier_base + layer * split_count +
                                       (split_count - 1);
            if (old == final_old) {
                __threadfence();
                float total = partials[row];
#pragma unroll
                for (int other = 1; other < split_count; ++other) {
                    total += partials[other * globals::hidden_dim + row];
                }
                const float residual =
                    __bfloat162float(g.hidden_states.raw_ptr[row]);
                g.hidden_states.raw_ptr[row] =
                    __float2bfloat16_rn(residual + total);
            }
        }
    }
}

template <int TileCols>
__device__ __forceinline__ void
cp_async_down_tile(bf16 *stage, const bf16 *weight_row, int col_begin,
                   int lane) {
    static_assert(TileCols == 256 || TileCols == 512 || TileCols == 1024 ||
                  TileCols == 2048);
    constexpr int bf16_per_copy = 16 / sizeof(bf16);
    constexpr int copies = TileCols / bf16_per_copy;
    for (int copy = lane; copy < copies; copy += 32) {
        const int col = copy * bf16_per_copy;
        cp_async_cg_16(stage + col, weight_row + col_begin + col);
    }
    cp_async_commit();
}

// Load a single down-weight prefix into warp-private SMEM before the incoming
// programmatic dependency is ready.  Only that prefix uses cp.async; after
// the wait it is consumed once from SMEM and the proven direct-SIMT path
// handles the remaining columns.  This keeps the consumer at 32 registers so
// it can reside alongside the 48-register, 1024-thread opcode5 producer.
template <typename Config, int PrefixCols>
__maxnreg__(32) __global__ void
downproj_smem_prefix_pdl(const __grid_constant__ globals g) {
    static_assert(Config::rows_per_warp == 1);
    static_assert(PrefixCols == 256 || PrefixCols == 512 ||
                  PrefixCols == 1024 || PrefixCols == 2048);

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    bf16 *weight_prefix = reinterpret_cast<bf16 *>(dynamic_smem);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int layer = instruction_layer(g);
    const int row = static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const bf16 *input = g.silu_out.raw_ptr;
    const bf16 *weight_row =
        g.down_weights.raw_ptr +
        layer * globals::hidden_dim * globals::intermediate_dim +
        row * globals::intermediate_dim;
    bf16 *warp_prefix = weight_prefix + warp * PrefixCols;

    cp_async_down_tile<PrefixCols>(warp_prefix, weight_row, 0, lane);
    cp_async_wait_all();
    __syncwarp();
    cudaGridDependencySynchronize();
    if (tid == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }

    float sum = 0.0f;
    constexpr int prefix_quads = PrefixCols / 4;
    constexpr int input_quads = globals::intermediate_dim / 4;
    for (int quad = lane; quad < prefix_quads; quad += 32) {
        accumulate_quad(load_bf16x4(input, quad),
                        load_bf16x4(warp_prefix, quad), sum);
    }
    for (int quad = prefix_quads + lane; quad < input_quads; quad += 32) {
        accumulate_quad(load_bf16x4(input, quad),
                        load_weight_bf16x4<-3>(weight_row, quad), sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        const float residual = __bfloat162float(g.hidden_states.raw_ptr[row]);
        g.hidden_states.raw_ptr[row] =
            __float2bfloat16_rn(residual + sum);
    }
}

// Warp-private, double/triple-buffered cp.async staging for cold down weights.
// Unlike opcode 5 this has only one weight stream per warp, so useful 512- and
// 1024-column stages fit in 32--96 KiB for all 16 warps.
template <typename Config, int TileCols, int StageCount = 2>
__launch_bounds__(Config::threads, 1) __global__ void
downproj_cp_async(const __grid_constant__ globals g) {
    static_assert(Config::rows_per_warp == 1);
    static_assert(StageCount == 2 || StageCount == 3);
    static_assert(globals::intermediate_dim % TileCols == 0);

    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    bf16 *weight_stages = reinterpret_cast<bf16 *>(dynamic_smem);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int layer = instruction_layer(g);
    const int row = static_cast<int>(blockIdx.x) * Config::rows_per_cta + warp;
    const bf16 *input = g.silu_out.raw_ptr;
    const bf16 *weight_row =
        g.down_weights.raw_ptr +
        layer * globals::hidden_dim * globals::intermediate_dim +
        row * globals::intermediate_dim;

    constexpr int stage_stride = Config::warps * TileCols;
    bf16 *warp_stage0 = weight_stages + warp * TileCols;
    cp_async_down_tile<TileCols>(warp_stage0, weight_row, 0, lane);
    if constexpr (StageCount == 3) {
        bf16 *warp_stage1 = weight_stages + stage_stride + warp * TileCols;
        cp_async_down_tile<TileCols>(warp_stage1, weight_row, TileCols, lane);
        cp_async_wait_one();
    } else {
        cp_async_wait_all();
    }
    __syncwarp();

    float sum = 0.0f;
    constexpr int tiles = globals::intermediate_dim / TileCols;
    constexpr int tile_quads = TileCols / 4;
#pragma unroll
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage_index = tile % StageCount;
        constexpr int prefetch_distance = StageCount - 1;
        if (tile + prefetch_distance < tiles) {
            const int prefetch_tile = tile + prefetch_distance;
            const int prefetch_stage = prefetch_tile % StageCount;
            bf16 *next_stage = weight_stages +
                               prefetch_stage * stage_stride +
                               warp * TileCols;
            cp_async_down_tile<TileCols>(next_stage, weight_row,
                                         prefetch_tile * TileCols, lane);
        }

        const bf16 *stage = weight_stages + stage_index * stage_stride +
                            warp * TileCols;
#pragma unroll
        for (int quad = lane; quad < tile_quads; quad += 32) {
            accumulate_quad(
                load_bf16x4(input, tile * tile_quads + quad),
                load_bf16x4(stage, quad), sum);
        }
        if (tile + 1 < tiles) {
            if constexpr (StageCount == 3) {
                if (tile + prefetch_distance < tiles) {
                    cp_async_wait_one();
                } else {
                    cp_async_wait_all();
                }
            } else {
                cp_async_wait_all();
            }
            __syncwarp();
        }
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        const float residual = __bfloat162float(g.hidden_states.raw_ptr[row]);
        g.hidden_states.raw_ptr[row] =
            __float2bfloat16_rn(residual + sum);
    }
}

} // namespace mlp_simt
} // namespace megakernel
