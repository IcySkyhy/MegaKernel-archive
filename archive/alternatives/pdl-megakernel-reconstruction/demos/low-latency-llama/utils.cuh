#pragma once

#include "llama.cuh"

template <typename Config, kittens::ducks::sv::all sv_t>
__device__ static inline auto
rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem,
         float rms_norm_eps, void *scratch_memory) {
    using rv_t = kittens::rv_fl<sv_t::length>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;

    kittens::warp::load(activations_vec, activations_smem);
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = kittens::warp::sum(sq_activations_vec);

    float *smem_rms_partial_sums = (float *)scratch_memory;
    if (kittens::laneid() == 0) {
        smem_rms_partial_sums[kittens::warpid()] = partial_sum;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

    float full_sum = 0;
#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += smem_rms_partial_sums[i];
    }

    float variance = full_sum / 2048.0f;
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

#ifdef KITTENS_BLACKWELL
template <bool RegisterBufferedStage = false,
          kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations,
                                     kittens::semaphore *weights_finished =
                                         nullptr) {
    using rt_t = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename kittens::rt_fl<16, 16>::col_vec;
    // using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    if constexpr (RegisterBufferedStage) {
        // Once every consumer warp has copied its private weight fragment into
        // registers, the sole SMEM stage can be refilled while arithmetic runs.
        kittens::warp::sync();
        kittens::warp::arrive(*weights_finished);
    }
    kittens::rt_fl<16, 16> out_activations;
    kittens::warp::zero(out_activations);
    kittens::warp::mma_ABt(out_activations, weights, broadcast_activations,
                  out_activations);
    rcv_t sum_col_vec;
    kittens::warp::row_max(sum_col_vec, out_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#else
template <kittens::ducks::rt::all rt_t>
__device__ static inline void keep_register_tile(rt_t &tile) {
#pragma unroll
    for (int i = 0; i < rt_t::height; i++) {
#pragma unroll
        for (int j = 0; j < rt_t::width; j++) {
#pragma unroll
            for (int k = 0; k < rt_t::packed_per_tile; k++) {
                asm volatile("" : "+f"(tile.tiles[i][j].data[k].x) :: "memory");
                asm volatile("" : "+f"(tile.tiles[i][j].data[k].y) :: "memory");
            }
        }
    }
}

template <int MicrotileCols,
          kittens::ducks::st::all st_t,
          kittens::ducks::sv::all activation_sv_t>
__device__ static inline void
matvec_smem_microtile(kittens::sv_fl<st_t::rows> &out_smem,
                      st_t &weights_smem,
                      activation_sv_t &activations_smem) {
    static_assert(st_t::rows == 16);
    static_assert(MicrotileCols == 16 || MicrotileCols == 32 ||
                  MicrotileCols == 64 || MicrotileCols == 128);
    static_assert(st_t::cols % MicrotileCols == 0);
    static_assert(activation_sv_t::length == st_t::cols);

    using micro_rt_t = kittens::rt_fl<st_t::rows, MicrotileCols>;
    using packed_t = typename micro_rt_t::dtype;
    using out_rv_t = kittens::rv_fl<st_t::rows>;

    packed_t accum_top_row{0.0f, 0.0f};
    packed_t accum_bottom_row{0.0f, 0.0f};
    const uint32_t activation_shared_addr =
        static_cast<uint32_t>(
            __cvta_generic_to_shared(&activations_smem.data[0]));

    // Keep this as a real loop. The source-level fully-unrolled candidate let
    // ptxas hoist several later LDS operations and rebuilt a wide live range.
#pragma unroll 1
    for (int chunk = 0; chunk < st_t::cols / MicrotileCols; chunk++) {
        micro_rt_t weight_chunk;
        auto weight_smem_chunk =
            weights_smem.template subtile<st_t::rows, MicrotileCols>(
                {0, chunk});
        {
            using src_t = typename decltype(weight_smem_chunk)::dtype;
            using src_packed_t =
                typename kittens::base_types::packing<src_t>::packed_type;
            const int lane = kittens::laneid();
            const int row = lane % 16;
            const uint32_t shared_addr =
                static_cast<uint32_t>(
                    __cvta_generic_to_shared(&weight_smem_chunk.data[0]));

#pragma unroll
            for (int subtile = 0; subtile < micro_rt_t::width; subtile++) {
                src_packed_t tmp0, tmp1, tmp2, tmp3;
                const int col = subtile * 16 + (lane / 16) * 8;
                kittens::move<src_packed_t>::ldsm4(
                    tmp0, tmp1, tmp2, tmp3,
                    weight_smem_chunk.idx(shared_addr, {row, col}));
                auto &base = weight_chunk.tiles[0][subtile];
                base.data[0] =
                    kittens::base_types::convertor<
                        packed_t, src_packed_t>::convert(tmp0);
                base.data[1] =
                    kittens::base_types::convertor<
                        packed_t, src_packed_t>::convert(tmp1);
                base.data[2] =
                    kittens::base_types::convertor<
                        packed_t, src_packed_t>::convert(tmp2);
                base.data[3] =
                    kittens::base_types::convertor<
                        packed_t, src_packed_t>::convert(tmp3);
            }
        }

#pragma unroll
        for (int subtile = 0; subtile < micro_rt_t::width; subtile++) {
            auto &base = weight_chunk.tiles[0][subtile];
            packed_t activation_left{0.0f, 0.0f};
            packed_t activation_right{0.0f, 0.0f};
            const int lane_in_quad = kittens::laneid() & 3;
            if (kittens::laneid() < 4) {
                const int activation_base =
                    chunk * MicrotileCols + subtile * 16 +
                    2 * lane_in_quad;
                kittens::move<packed_t>::lds(
                    activation_left,
                    activation_shared_addr +
                        sizeof(float) * activation_base);
                kittens::move<packed_t>::lds(
                    activation_right,
                    activation_shared_addr +
                        sizeof(float) * (activation_base + 8));
            }
            activation_left = kittens::packed_shfl_sync(
                kittens::MASK_ALL, activation_left, lane_in_quad);
            activation_right = kittens::packed_shfl_sync(
                kittens::MASK_ALL, activation_right, lane_in_quad);
#pragma unroll
            for (int packed = 0;
                 packed < micro_rt_t::packed_per_tile / 2; packed++) {
                // Spell out TK's col_map in-place. Passing the same register
                // tile as both dst and const src to warp::mul_col made ptxas
                // preserve half of the tile in a local-memory stack frame.
                // Each element is independent, so direct element-wise updates
                // retain the arithmetic while ending that false live range.
                base.data[packed] =
                    kittens::base_ops::mul::template op<packed_t>(
                        base.data[packed],
                        activation_left);
                base.data[packed + 2] =
                    kittens::base_ops::mul::template op<packed_t>(
                        base.data[packed + 2],
                        activation_right);
            }
            accum_top_row =
                kittens::base_ops::sum::template op<packed_t>(
                    accum_top_row, base.data[0]);
            accum_top_row =
                kittens::base_ops::sum::template op<packed_t>(
                    accum_top_row, base.data[2]);
            accum_bottom_row =
                kittens::base_ops::sum::template op<packed_t>(
                    accum_bottom_row, base.data[1]);
            accum_bottom_row =
                kittens::base_ops::sum::template op<packed_t>(
                    accum_bottom_row, base.data[3]);
        }
    }

    packed_t accum_packed;
    accum_packed.x = accum_top_row.x + accum_top_row.y;
    accum_packed.y = accum_bottom_row.x + accum_bottom_row.y;
    accum_packed = kittens::base_ops::sum::template op<packed_t>(
        accum_packed,
        kittens::packed_shfl_down_sync(
            kittens::MASK_ALL, accum_packed, 2));
    accum_packed = kittens::base_ops::sum::template op<packed_t>(
        accum_packed,
        kittens::packed_shfl_down_sync(
            kittens::MASK_ALL, accum_packed, 1));
    const int leader = threadIdx.x & 0x1C;
    accum_packed = kittens::packed_shfl_sync(
        kittens::MASK_ALL, accum_packed, leader);

    typename micro_rt_t::col_vec sum_col_vec;
    sum_col_vec[0][0] = accum_packed;
    out_rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);
    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}

template <bool RegisterBufferedStage = false,
          bool AliasOperands = false,
          int ChunkCols = 0,
          kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations,
                                     kittens::semaphore *weights_finished =
                                         nullptr) {
    using rt_t = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rcv_t sum_col_vec;
    if constexpr (ChunkCols > 0) {
        static_assert(st_t::rows == 16);
        static_assert(ChunkCols == 16);
        static_assert(st_t::cols % ChunkCols == 0);
        using chunk_rt_t = kittens::rt_fl<st_t::rows, ChunkCols>;
        using chunk_rrv_t = typename chunk_rt_t::row_vec;
        using packed_t = typename chunk_rt_t::dtype;

        packed_t accum_top_row, accum_bottom_row;
#pragma unroll
        for (int chunk = 0; chunk < st_t::cols / ChunkCols; chunk++) {
            chunk_rrv_t activation_chunk;
            activation_chunk[0][0] = row_activations[chunk][0];

            chunk_rt_t broadcast_chunk, weight_chunk;
            kittens::warp::broadcast_col(broadcast_chunk, activation_chunk);
            auto weight_smem_chunk =
                weights_smem.template subtile<st_t::rows, ChunkCols>(
                    {0, chunk});
            kittens::warp::load(weight_chunk, weight_smem_chunk);
            kittens::warp::mul(broadcast_chunk, broadcast_chunk, weight_chunk);

            auto &base = broadcast_chunk.tiles[0][0];
            if (chunk == 0) {
                accum_top_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        base.data[0], base.data[2]);
                accum_bottom_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        base.data[1], base.data[3]);
            } else {
                accum_top_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        accum_top_row, base.data[0]);
                accum_top_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        accum_top_row, base.data[2]);
                accum_bottom_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        accum_bottom_row, base.data[1]);
                accum_bottom_row =
                    kittens::base_ops::sum::template op<packed_t>(
                        accum_bottom_row, base.data[3]);
            }
        }
        if constexpr (RegisterBufferedStage) {
            kittens::warp::sync();
            kittens::warp::arrive(*weights_finished);
        }

        packed_t accum_packed;
        accum_packed.x = accum_top_row.x + accum_top_row.y;
        accum_packed.y = accum_bottom_row.x + accum_bottom_row.y;
        accum_packed = kittens::base_ops::sum::template op<packed_t>(
            accum_packed,
            kittens::packed_shfl_down_sync(
                kittens::MASK_ALL, accum_packed, 2));
        accum_packed = kittens::base_ops::sum::template op<packed_t>(
            accum_packed,
            kittens::packed_shfl_down_sync(
                kittens::MASK_ALL, accum_packed, 1));
        const int leader = threadIdx.x & 0x1C;
        accum_packed = kittens::packed_shfl_sync(
            kittens::MASK_ALL, accum_packed, leader);
        sum_col_vec[0][0] = accum_packed;
    } else if constexpr (AliasOperands) {
        // Keep the original activation broadcast in the instruction stream,
        // but terminate its live range before the weight tile is allocated.
        // The empty asm is a compiler barrier only; it emits no arithmetic or
        // memory instruction. The subsequent weight-square deliberately
        // reuses one register tile while retaining one mul per element.
        {
            rt_t reused_tile;
            kittens::warp::broadcast_col(reused_tile, row_activations);
            keep_register_tile(reused_tile);
        }
        {
            rt_t reused_tile;
            kittens::warp::load(reused_tile, weights_smem);
            if constexpr (RegisterBufferedStage) {
                kittens::warp::sync();
                kittens::warp::arrive(*weights_finished);
            }
            kittens::warp::mul(reused_tile, reused_tile, reused_tile);
            kittens::warp::row_sum(sum_col_vec, reused_tile);
        }
    } else {
        rt_t broadcast_activations, weights;
        kittens::warp::broadcast_col(broadcast_activations, row_activations);
        kittens::warp::load(weights, weights_smem);
        if constexpr (RegisterBufferedStage) {
            // The FP32 register tile is now independent of SMEM. Release before
            // mul/reduction so the loader can overlap the next TMA transfer.
            kittens::warp::sync();
            kittens::warp::arrive(*weights_finished);
        }
        kittens::warp::mul(broadcast_activations, broadcast_activations, weights);
        kittens::warp::row_sum(sum_col_vec, broadcast_activations);
    }

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#endif

template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    rv_t part_vec;
    kittens::warp::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {

        // TODO: for now, deliberately not using sizeof(sv_t) here because we've
        // had alignment issues before.
        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));

        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}
