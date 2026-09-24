#pragma once

#include "llama.cuh"

// This is the production PartialAttention consumer math with the persistent
// VM shell removed.  P32/D128 uses one partial, so the reduction is skipped
// and one CTA can write the four GQA heads for one KV head directly.
namespace megakernel {
namespace attention_tk_short {

using globals = llama_1b_globals;
using config = default_config;
using op = attention_partial<config, globals>;
using q_st_t = op::q_st;
using kv_st_t = op::kv_st;
using o_sv_t = op::o_sv;
using o_sv_bf_t = op::o_sv_bf;
using q_rt_t = op::q_rt;
using k_rt_t = op::k_rt;
using v_rt_t = op::v_rt;
using o_rt_t = op::o_rt;
using attn_fl_rt_t = op::attn_fl_rt;
using attn_bf_rt_t = op::attn_bf_rt;
using max_vec_rv_t = op::max_vec_rv;
using norm_vec_rv_t = op::norm_vec_rv;

constexpr int GQA_RATIO =
    LLAMA_1B_NUM_ATTENTION_HEADS / LLAMA_1B_NUM_KV_HEADS;
constexpr int NUM_KV_HEADS = LLAMA_1B_NUM_KV_HEADS;
constexpr int KV_BLOCK = LLAMA_1B_KV_BLOCK_SIZE;
constexpr int HEAD_DIM = LLAMA_1B_HEAD_DIM;
constexpr int grid_ctas = NUM_KV_HEADS;
constexpr int threads = 32;
constexpr int dynamic_smem_bytes = 16384;

template <int Stages>
constexpr int dynamic_smem_bytes_for = 4096 * Stages + 4096;

__device__ __forceinline__ int instruction_layer(const globals &g) {
    return g.instructions.raw_ptr[1];
}

template <int Stages = 3, bool LateCurrentKvWait = false,
          bool HistoricalPrefetchWait = false,
          bool UseCpAsync = false,
          int HistoricalPrefetchMinBlocks = 0,
          bool HistoricalRegisterPrefetchWait = false>
__device__ __forceinline__ void run_attention(const globals &g, int layer) {
    using namespace kittens;
    static_assert(!(HistoricalPrefetchWait &&
                    HistoricalRegisterPrefetchWait));

    const int kv_head_idx = static_cast<int>(blockIdx.x);
    extern __shared__ int dynamic_smem[];
    shared_allocator allocator(dynamic_smem);

    q_st_t &q_smem = allocator.template allocate<q_st_t>();
    kv_st_t (&k_smem)[Stages] =
        allocator.template allocate<kv_st_t, Stages>();
    kv_st_t (&v_smem)[Stages] =
        allocator.template allocate<kv_st_t, Stages>();
    o_sv_t (&o_smem)[GQA_RATIO] =
        allocator.template allocate<o_sv_t, GQA_RATIO>();

    __shared__ semaphore k_arrived[Stages];
    __shared__ semaphore v_arrived[Stages];
    if constexpr (!UseCpAsync) {
        if (laneid() == 0) {
#pragma unroll
            for (int stage = 0; stage < Stages; ++stage) {
                init_semaphore(k_arrived[stage], 0, 1);
                init_semaphore(v_arrived[stage], 0, 1);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        warp::sync();
    }

    const int q_head_start = kv_head_idx * GQA_RATIO;
    const int q_head_local = (q_head_start % 16) / 4;
    const int seq_len = static_cast<int>(g.pos_id) + 1;
    const int total_blocks = (seq_len + KV_BLOCK - 1) / KV_BLOCK;
    const float softmax_temp = g.attn_scale * 1.44269504089f;
    bool historical_prefetch_active = false;
    if constexpr (HistoricalPrefetchWait) {
        historical_prefetch_active =
            HistoricalPrefetchMinBlocks == 0 ||
            total_blocks >= HistoricalPrefetchMinBlocks;
    }
    const bool historical_register_prefetch_active =
        HistoricalRegisterPrefetchWait && total_blocks > 1;
    const int pipeline_stages =
        HistoricalPrefetchWait && HistoricalPrefetchMinBlocks > 0 &&
                !historical_prefetch_active
        ? min(3, Stages)
        : Stages;

    q_rt_t q_reg;
    k_rt_t k_reg;
    v_rt_t v_reg;
    o_rt_t o_reg;
    attn_fl_rt_t attn_fl_reg;
    attn_bf_rt_t attn_bf_reg;
    max_vec_rv_t max_vec_reg;
    max_vec_rv_t scaled_max_vec_reg;
    max_vec_rv_t last_scaled_max_vec_reg;
    max_vec_rv_t diff_scaled_max_vec_reg;
    norm_vec_rv_t norm_vec_reg;
    warp::neg_infty(max_vec_reg);
    warp::zero(last_scaled_max_vec_reg);
    warp::zero(norm_vec_reg);
    warp::zero(o_reg);

    if constexpr (HistoricalRegisterPrefetchWait) {
        if (historical_register_prefetch_active) {
            // Block zero is immutable history for every P32/D128 decode
            // position.  Load it directly into the exact TK MMA register
            // layouts, retain it across the dependency wait, and consume it
            // without a global->SMEM->register round trip.
            warp::load(k_reg, g.k_cache,
                       {layer, 0, kv_head_idx, 0});
            warp::load(v_reg, g.v_cache,
                       {layer, 0, kv_head_idx, 0});
        }
    }

    const int initially_safe_blocks =
        historical_register_prefetch_active
        ? 0
        : (LateCurrentKvWait || historical_prefetch_active)
              ? min(total_blocks - 1, pipeline_stages)
              : min(total_blocks, pipeline_stages);
    if constexpr (UseCpAsync) {
        for (int block = 0; block < initially_safe_blocks; ++block) {
            // Each tile is one cp.async group.  Drain every three KV pairs so
            // the warp never exceeds Hopper's eight outstanding groups; keep
            // the final batch live so it can overlap the incoming PDL wait.
            if (block > 0 && block % 3 == 0) {
                warp::load_async_wait();
            }
            warp::load_async(
                k_smem[block], g.k_cache,
                {layer, block, kv_head_idx, 0});
            warp::load_async(
                v_smem[block], g.v_cache,
                {layer, block, kv_head_idx, 0});
        }
    } else {
        if (laneid() == 0) {
            for (int block = 0; block < initially_safe_blocks; ++block) {
                tma::expect(k_arrived[block], k_smem[block]);
                tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                    k_smem[block], g.k_cache,
                    {layer, block, kv_head_idx, 0}, k_arrived[block]);
                tma::expect(v_arrived[block], v_smem[block]);
                tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                    v_smem[block], g.v_cache,
                    {layer, block, kv_head_idx, 0}, v_arrived[block]);
            }
        }
    }

    if constexpr (HistoricalPrefetchWait) {
        if (historical_prefetch_active) {
            cudaGridDependencySynchronize();
            // If the current/final block fits in the initial stage set, it
            // was deliberately omitted above and becomes safe only after the
            // producer grid completes.
            if constexpr (UseCpAsync) {
                for (int block = initially_safe_blocks;
                     block < min(total_blocks, pipeline_stages); ++block) {
                    warp::load_async(
                        k_smem[block], g.k_cache,
                        {layer, block, kv_head_idx, 0});
                    warp::load_async(
                        v_smem[block], g.v_cache,
                        {layer, block, kv_head_idx, 0});
                }
                const int initial_target =
                    min(total_blocks, pipeline_stages);
                const int pending_history_groups =
                    initially_safe_blocks == 0
                    ? 0
                    : 2 * (((initially_safe_blocks - 1) % 3) + 1);
                const int post_wait_groups =
                    2 * (initial_target - initially_safe_blocks);
                if (pending_history_groups + post_wait_groups + 1 > 8) {
                    // Q below contributes one more group.  Drain only when
                    // that would exceed Hopper's eight-group limit.
                    warp::load_async_wait();
                }
            } else {
                if (laneid() == 0) {
                    for (int block = initially_safe_blocks;
                         block < min(total_blocks, pipeline_stages); ++block) {
                        tma::expect(k_arrived[block], k_smem[block]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            k_smem[block], g.k_cache,
                            {layer, block, kv_head_idx, 0},
                            k_arrived[block]);
                        tma::expect(v_arrived[block], v_smem[block]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            v_smem[block], g.v_cache,
                            {layer, block, kv_head_idx, 0},
                            v_arrived[block]);
                    }
                }
            }
        }
    }

    if constexpr (HistoricalRegisterPrefetchWait) {
        if (historical_register_prefetch_active) {
            cudaGridDependencySynchronize();
            // Block zero is already live in k_reg/v_reg.  Fill the remaining
            // initial ring slots only after the QKV producer is complete,
            // since the final/current slot may have just been written.
            if constexpr (UseCpAsync) {
                for (int block = 1;
                     block < min(total_blocks, pipeline_stages); ++block) {
                    warp::load_async(
                        k_smem[block], g.k_cache,
                        {layer, block, kv_head_idx, 0});
                    warp::load_async(
                        v_smem[block], g.v_cache,
                        {layer, block, kv_head_idx, 0});
                }
            } else {
                if (laneid() == 0) {
                    for (int block = 1;
                         block < min(total_blocks, pipeline_stages);
                         ++block) {
                        tma::expect(k_arrived[block], k_smem[block]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            k_smem[block], g.k_cache,
                            {layer, block, kv_head_idx, 0},
                            k_arrived[block]);
                        tma::expect(v_arrived[block], v_smem[block]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            v_smem[block], g.v_cache,
                            {layer, block, kv_head_idx, 0},
                            v_arrived[block]);
                    }
                }
            }
        }
    }

    if constexpr (LateCurrentKvWait) {
        if (laneid() == 0) {
            constexpr unsigned ready_base = 1'000'000u;
            constexpr unsigned pairs_per_head = globals::head_dim / 2;
            constexpr int q_flag_offset = 4096;
#pragma unroll
            for (int q_offset = 0; q_offset < GQA_RATIO; ++q_offset) {
                const volatile unsigned *ready =
                    g.Bar.raw_ptr + q_flag_offset +
                    layer * globals::num_attention_heads +
                    q_head_start + q_offset;
                while (*ready < ready_base + pairs_per_head) {
                    __nanosleep(64);
                }
            }
            __threadfence();
        }
        warp::sync();
    }

    op::load_Q_async(q_smem, g.q_post_rope, q_head_start);
    warp::load_async_wait();
    warp::load(q_reg, q_smem);

    for (int i = 0; i < total_blocks; ++i) {
        const int buffer = i % pipeline_stages;
        // With register-prefetched block zero, SMEM buffer zero's first TMA
        // transaction is block `pipeline_stages`, not block zero.
        const int phase =
            historical_register_prefetch_active && buffer == 0
            ? ((i / pipeline_stages) - 1) & 1
            : (i / pipeline_stages) & 1;

        if constexpr (LateCurrentKvWait) {
            if (i + 1 == total_blocks) {
                cudaGridDependencySynchronize();
                if constexpr (UseCpAsync) {
                    warp::load_async(
                        k_smem[buffer], g.k_cache,
                        {layer, i, kv_head_idx, 0});
                    warp::load_async(
                        v_smem[buffer], g.v_cache,
                        {layer, i, kv_head_idx, 0});
                } else {
                    if (laneid() == 0) {
                        tma::expect(k_arrived[buffer], k_smem[buffer]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            k_smem[buffer], g.k_cache,
                            {layer, i, kv_head_idx, 0}, k_arrived[buffer]);
                        tma::expect(v_arrived[buffer], v_smem[buffer]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            v_smem[buffer], g.v_cache,
                            {layer, i, kv_head_idx, 0}, v_arrived[buffer]);
                    }
                }
            }
        }

        warp::zero(attn_fl_reg);
        const bool use_register_prefetch =
            historical_register_prefetch_active && i == 0;
        if (!use_register_prefetch) {
            if constexpr (UseCpAsync) {
                if (i >= pipeline_stages) {
                    if (pipeline_stages == 3) {
                        // Two groups belong to each future KV block.  With
                        // the 3-stage ring, leaving four groups guarantees
                        // that the oldest K/V pair is complete while
                        // retaining overlap.
                        warp::load_async_wait<4>();
                    } else {
                        warp::load_async_wait();
                    }
                }
            } else {
                warp::wait(k_arrived[buffer], phase);
            }
            warp::load(k_reg, k_smem[buffer]);
            if constexpr (!UseCpAsync) {
                warp::wait(v_arrived[buffer], phase);
            }
            warp::load(v_reg, v_smem[buffer]);
        }

        const int next = i + pipeline_stages;
        if (next < total_blocks) {
            if (!LateCurrentKvWait || next + 1 < total_blocks) {
                const int next_buffer = next % pipeline_stages;
                if constexpr (UseCpAsync) {
                    warp::load_async(
                        k_smem[next_buffer], g.k_cache,
                        {layer, next, kv_head_idx, 0});
                    warp::load_async(
                        v_smem[next_buffer], g.v_cache,
                        {layer, next, kv_head_idx, 0});
                } else {
                    if (laneid() == 0) {
                        tma::expect(k_arrived[next_buffer],
                                    k_smem[next_buffer]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            k_smem[next_buffer], g.k_cache,
                            {layer, next, kv_head_idx, 0},
                            k_arrived[next_buffer]);
                        tma::expect(v_arrived[next_buffer],
                                    v_smem[next_buffer]);
                        tma::load_async<dim::DEPTH,
                                        cache_policy::EVICT_FIRST>(
                            v_smem[next_buffer], g.v_cache,
                            {layer, next, kv_head_idx, 0},
                            v_arrived[next_buffer]);
                    }
                }
            }
        }

        warp::mma_ABt(attn_fl_reg, q_reg, k_reg, attn_fl_reg);
        warp::sync();
        if ((i + 1) * KV_BLOCK > seq_len) {
            op::right_fill(attn_fl_reg, attn_fl_reg, seq_len % KV_BLOCK,
                           -999999999999.f);
        }

        warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);
        warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
        warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);
        warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
        warp::exp2(attn_fl_reg, attn_fl_reg);
        warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg,
                  scaled_max_vec_reg);
        warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);

        warp::mul_row(o_reg, o_reg, diff_scaled_max_vec_reg);
        warp::copy(attn_bf_reg, attn_fl_reg);
        warp::mma_AB(o_reg, attn_bf_reg, v_reg, o_reg);
        warp::sync();
        warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
        warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
        warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
    }

    warp::div_row(o_reg, o_reg, norm_vec_reg);
    op::store_4_rows(o_smem, o_reg, q_head_local);
    warp::sync();

    rv_bf<HEAD_DIM> out_bf;
#pragma unroll
    for (int q_offset = 0; q_offset < GQA_RATIO; ++q_offset) {
        auto &smem_fl = o_smem[q_offset];
        auto &smem_bf =
            *reinterpret_cast<o_sv_bf_t *>(&smem_fl);
        warp::load(out_bf, smem_fl);
        warp::sync();
        warp::store(smem_bf, out_bf);
        warp::sync();
    }
    if (laneid() == 0) {
#pragma unroll
        for (int q_offset = 0; q_offset < GQA_RATIO; ++q_offset) {
            auto &smem_bf =
                *reinterpret_cast<o_sv_bf_t *>(&o_smem[q_offset]);
            tma::store_async<cache_policy::EVICT_LAST>(
                g.attn_out, smem_bf, {q_head_start + q_offset});
        }
        tma::store_async_wait();
    }
    warp::sync();
}

template <bool PdlWait = false, bool PdlTrigger = false,
          bool LateCurrentKvWait = false, int Stages = 3,
          bool HistoricalPrefetchWait = false,
          bool UseCpAsync = false,
          int HistoricalPrefetchMinBlocks = 0,
          bool HistoricalRegisterPrefetchWait = false>
__global__ __launch_bounds__(threads, 1) void
short_attention(const __grid_constant__ globals g) {
    if constexpr (PdlWait && !LateCurrentKvWait) {
        if constexpr (!HistoricalPrefetchWait &&
                      !HistoricalRegisterPrefetchWait) {
            cudaGridDependencySynchronize();
        } else if constexpr (HistoricalPrefetchMinBlocks > 0) {
            const int total_blocks =
                (static_cast<int>(g.pos_id) + 1 + KV_BLOCK - 1) / KV_BLOCK;
            if (total_blocks < HistoricalPrefetchMinBlocks) {
                cudaGridDependencySynchronize();
            }
        } else if constexpr (HistoricalRegisterPrefetchWait) {
            const int total_blocks =
                (static_cast<int>(g.pos_id) + 1 + KV_BLOCK - 1) / KV_BLOCK;
            if (total_blocks <= 1) {
                cudaGridDependencySynchronize();
            }
        }
    }
    if constexpr (PdlTrigger) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    run_attention<Stages, LateCurrentKvWait, HistoricalPrefetchWait,
                  UseCpAsync, HistoricalPrefetchMinBlocks,
                  HistoricalRegisterPrefetchWait>(
        g, instruction_layer(g));
}

} // namespace attention_tk_short
} // namespace megakernel
