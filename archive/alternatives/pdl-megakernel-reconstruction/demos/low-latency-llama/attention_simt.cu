#pragma once

#include "llama.cuh"

#include <cuda_bf16.h>
#include <cfloat>

// Context-1 partial attention.  A one-element softmax is exactly one, so each
// query head's result is its grouped-query value head; Q and K provably cancel
// from the result.  These launch-shape controls are all single-wave kernels.
namespace megakernel {
namespace attention_simt {

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

using attention_1cta_32w = config<1, 32>;
using attention_4cta_8w = config<4, 8>;
using attention_8cta_4w = config<8, 4>;
using attention_32cta_1w = config<32, 1>;

struct attention_bypass_1cta_1w {
    static constexpr int ctas = 1;
    static constexpr int warps = 1;
    static constexpr int threads = 32;
};

// B1/P32/D128 decode never exceeds 159 visible KV tokens. One CTA owns a KV
// head and its four grouped-query heads; one warp computes one query head.
// Scores are materialized in a small per-warp SMEM row so exp is evaluated
// once per token and reused by all 32 output-dimension lanes.
constexpr int short_context_max_seq = 160;
using short_context_8cta_4w = attention_8cta_4w;

__device__ __forceinline__ int instruction_layer(const globals &g) {
    return g.instructions.raw_ptr[1];
}

template <typename Config, bool PdlWait = false, bool PdlTrigger = false,
          int PdlTriggerStage = -1, bool VReadyWait = false>
__launch_bounds__(Config::threads, 1) __global__ void
context1_attention(const __grid_constant__ globals g) {
    static_assert(globals::head_dim == 64);
    static_assert(globals::num_attention_heads == 32);
    static_assert(globals::num_kv_heads == 8);
    static_assert(PdlTriggerStage == -1 || PdlTriggerStage == 0);
    constexpr int q_heads_per_kv =
        globals::num_attention_heads / globals::num_kv_heads;

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int q_head = static_cast<int>(blockIdx.x) * Config::warps + warp;
    const int kv_head = q_head / q_heads_per_kv;
    const int layer = instruction_layer(g);

    if constexpr (VReadyWait) {
        constexpr unsigned barrier_base = 1000000u;
        constexpr unsigned pairs_per_head = globals::head_dim / 2;
        constexpr int v_flag_offset = 4096;
        if (threadIdx.x == 0) {
            const volatile unsigned *ready =
                g.Bar.raw_ptr + v_flag_offset +
                layer * globals::num_kv_heads + kv_head;
            while (*ready < barrier_base + pairs_per_head) {
                __nanosleep(64);
            }
            __threadfence();
        }
        __syncthreads();
    } else if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger && PdlTriggerStage < 0) {
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    const int cache_offset =
        ((layer * static_cast<int>(g.v_cache.depth()) +
          static_cast<int>(g.pos_id)) *
             static_cast<int>(g.v_cache.rows()) +
         kv_head) *
        globals::head_dim;

    const bf16x2 value =
        reinterpret_cast<const bf16x2 *>(g.v_cache.raw_ptr + cache_offset)[lane];
    reinterpret_cast<bf16x2 *>(
        g.attn_out.raw_ptr + q_head * globals::head_dim)[lane] = value;
    if constexpr (PdlTrigger && PdlTriggerStage == 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
}

template <typename Config, bool PdlWait = false>
__launch_bounds__(Config::threads, 1) __global__ void
context1_attention_bypass(const __grid_constant__ globals g) {
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
}

__device__ __forceinline__ float warp_sum_short(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

__device__ __forceinline__ float warp_sum_broadcast_short(float value) {
    value = warp_sum_short(value);
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ void cp_async_cg_16_short(void *shared_dst,
                                                     const void *global_src) {
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" : :
        "r"(shared_address), "l"(global_src) : "memory");
}

template <bool PdlWait = false, bool PdlTrigger = false,
          bool PrefetchHistoricalKv = false, bool UseCpAsync = true>
__launch_bounds__(short_context_8cta_4w::threads, 1) __global__ void
short_context_attention(const __grid_constant__ globals g) {
    static_assert(globals::head_dim == 64);
    static_assert(globals::num_attention_heads == 32);
    static_assert(globals::num_kv_heads == 8);
    constexpr int q_heads_per_kv =
        globals::num_attention_heads / globals::num_kv_heads;

    // A KV head is shared by all four GQA consumer warps.  Stage it once per
    // CTA instead of making each warp issue its own K/V global loads.  At the
    // largest P32/D128 context this is 40 KiB for K+V plus 2.5 KiB of scores.
    __shared__ __align__(16)
        bf16x2 k_smem[short_context_max_seq][globals::head_dim / 2];
    __shared__ __align__(16)
        bf16x2 v_smem[short_context_max_seq][globals::head_dim / 2];
    __shared__ float scores[q_heads_per_kv][short_context_max_seq];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int q_head = kv_head * q_heads_per_kv + warp;
    const int layer = instruction_layer(g);
    const int position = static_cast<int>(g.pos_id);
    const int seq_len = position + 1;

    if (seq_len > short_context_max_seq) {
        if (tid == 0) {
            asm volatile("trap;\n");
        }
        return;
    }

    if constexpr (PdlWait && PrefetchHistoricalKv) {
        // Every historical K or V head row is exactly one 128-byte L2 line.
        // Distribute those cache hints across the four consumer warps before
        // waiting for the current QKV projection to publish Q/K/V.
        const int historical_lines = position * 2;
        for (int line = tid; line < historical_lines;
             line += short_context_8cta_4w::threads) {
            const bool is_v = line >= position;
            const int token = line - (is_v ? position : 0);
            const bf16 *cache =
                is_v ? g.v_cache.raw_ptr : g.k_cache.raw_ptr;
            const int cache_offset =
                ((layer * static_cast<int>(g.k_cache.depth()) + token) *
                     static_cast<int>(g.k_cache.rows()) +
                 kv_head) *
                globals::head_dim;
            asm volatile("prefetch.global.L2 [%0];\n" ::
                             "l"(cache + cache_offset)
                         : "memory");
        }
    }
    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    // One 64-wide BF16 cache row is eight 16-byte chunks.  Treat K and V as
    // one combined copy domain, distribute it over all 128 threads, and wait
    // only once before the consumer warps start the dot products.
    constexpr int chunks_per_row =
        globals::head_dim * static_cast<int>(sizeof(bf16)) / 16;
    const int chunks_per_cache = seq_len * chunks_per_row;
    const int total_chunks = 2 * chunks_per_cache;
    const int cache_token_stride =
        static_cast<int>(g.k_cache.rows()) * globals::head_dim;
    const int cache_layer_base =
        layer * static_cast<int>(g.k_cache.depth()) * cache_token_stride;
    for (int chunk = tid; chunk < total_chunks;
         chunk += short_context_8cta_4w::threads) {
        const bool is_v = chunk >= chunks_per_cache;
        const int local_chunk = chunk - (is_v ? chunks_per_cache : 0);
        const int token = local_chunk / chunks_per_row;
        const int row_chunk = local_chunk % chunks_per_row;
        const bf16 *global_cache =
            is_v ? g.v_cache.raw_ptr : g.k_cache.raw_ptr;
        const char *global_src = reinterpret_cast<const char *>(
            global_cache + cache_layer_base + token * cache_token_stride +
            kv_head * globals::head_dim) + row_chunk * 16;
        char *shared_dst = reinterpret_cast<char *>(
            is_v ? v_smem[token] : k_smem[token]) + row_chunk * 16;
        if constexpr (UseCpAsync) {
            cp_async_cg_16_short(shared_dst, global_src);
        } else {
            *reinterpret_cast<uint4 *>(shared_dst) =
                *reinterpret_cast<const uint4 *>(global_src);
        }
    }
    if constexpr (UseCpAsync) {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();

    const bf16x2 q = reinterpret_cast<const bf16x2 *>(
        g.q_post_rope.raw_ptr + q_head * globals::head_dim)[lane];
    const float2 qf = __bfloat1622float2(q);
    float max_score = -FLT_MAX;

    for (int token = 0; token < seq_len; ++token) {
        const bf16x2 k = k_smem[token][lane];
        const float2 kf = __bfloat1622float2(k);
        float dot = fmaf(qf.x, kf.x, qf.y * kf.y);
        dot = warp_sum_short(dot);
        if (lane == 0) {
            const float score = dot * g.attn_scale;
            scores[warp][token] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    max_score = __shfl_sync(0xffffffffu, max_score, 0);
    __syncwarp();

    float sum = 0.0f;
    for (int token = lane; token < seq_len; token += 32) {
        const float probability = __expf(scores[warp][token] - max_score);
        scores[warp][token] = probability;
        sum += probability;
    }
    sum = warp_sum_short(sum);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    __syncwarp();

    float out0 = 0.0f;
    float out1 = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const bf16x2 v = v_smem[token][lane];
        const float2 vf = __bfloat1622float2(v);
        const float probability = scores[warp][token] / sum;
        out0 = fmaf(probability, vf.x, out0);
        out1 = fmaf(probability, vf.y, out1);
    }
    reinterpret_cast<bf16x2 *>(
        g.attn_out.raw_ptr + q_head * globals::head_dim)[lane] =
        __floats2bfloat162_rn(out0, out1);
}

// Direct short-context GQA attention for B1/P32/D128.  One warp owns one KV
// head and all four query heads that share it.  Each lane owns one adjacent
// BF16x2 dimension pair, so every K/V pair is loaded once and reused across
// the four queries entirely in registers.  Online softmax avoids score
// storage; this path has no TMA, mbarrier, tensor-core, or shared-memory state.
template <bool PdlWait = false, bool PdlTrigger = false,
          int WarpsPerKv = 1, int PrefetchTokens = 0>
__launch_bounds__(32 * WarpsPerKv, 1) __global__ void
short_context_attention_direct_gqa(const __grid_constant__ globals g) {
    static_assert(globals::head_dim == 64);
    static_assert(globals::num_attention_heads == 32);
    static_assert(globals::num_kv_heads == 8);
    constexpr int q_heads_per_kv =
        globals::num_attention_heads / globals::num_kv_heads;
    static_assert(q_heads_per_kv == 4);
    static_assert(WarpsPerKv == 1 || WarpsPerKv == 2 || WarpsPerKv == 4);
    static_assert(PrefetchTokens == 0 || PrefetchTokens == 8 ||
                  PrefetchTokens == 16 || PrefetchTokens == 32);
    static_assert(PrefetchTokens == 0 || PdlWait);
    constexpr int q_heads_per_warp = q_heads_per_kv / WarpsPerKv;
    constexpr int prefetch_storage =
        PrefetchTokens == 0 ? 1 : PrefetchTokens;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int q_head_start =
        kv_head * q_heads_per_kv + warp * q_heads_per_warp;
    const int layer = instruction_layer(g);
    const int position = static_cast<int>(g.pos_id);
    const int seq_len = position + 1;

    if (seq_len > short_context_max_seq) {
        if (lane == 0) {
            asm volatile("trap;\n");
        }
        return;
    }

    const int cache_token_stride =
        static_cast<int>(g.k_cache.rows()) * globals::head_dim;
    const int cache_layer_base =
        layer * static_cast<int>(g.k_cache.depth()) * cache_token_stride;
    const int cache_head_offset = kv_head * globals::head_dim;
    const bf16 *k_layer = g.k_cache.raw_ptr + cache_layer_base;
    const bf16 *v_layer = g.v_cache.raw_ptr + cache_layer_base;

    // Historical cache rows are independent of the current QKV producer. A
    // PDL successor may therefore demand-load a bounded prefix before its
    // dependency wait and retain the packed BF16 pairs in registers. The
    // current-position row is deliberately excluded because QKV is writing
    // it. Fixed, unrolled indices prevent these arrays from becoming local
    // memory; ptxas resource output is the promotion gate for every size.
    bf16x2 k_prefetched[prefetch_storage];
    bf16x2 v_prefetched[prefetch_storage];
    if constexpr (PrefetchTokens > 0) {
#pragma unroll
        for (int token = 0; token < PrefetchTokens; ++token) {
            if (token < position) {
                const int cache_offset =
                    token * cache_token_stride + cache_head_offset;
                k_prefetched[token] = reinterpret_cast<const bf16x2 *>(
                    k_layer + cache_offset)[lane];
                v_prefetched[token] = reinterpret_cast<const bf16x2 *>(
                    v_layer + cache_offset)[lane];
            }
        }
    }

    if constexpr (PdlWait) {
        cudaGridDependencySynchronize();
    }
    if constexpr (PdlTrigger) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    float q0[q_heads_per_warp];
    float q1[q_heads_per_warp];
    float out0[q_heads_per_warp] = {};
    float out1[q_heads_per_warp] = {};
    float max_score[q_heads_per_warp];
    float normalizer[q_heads_per_warp] = {};
#pragma unroll
    for (int q_offset = 0; q_offset < q_heads_per_warp; ++q_offset) {
        const bf16x2 q = reinterpret_cast<const bf16x2 *>(
            g.q_post_rope.raw_ptr +
            (q_head_start + q_offset) * globals::head_dim)[lane];
        const float2 qf = __bfloat1622float2(q);
        q0[q_offset] = qf.x;
        q1[q_offset] = qf.y;
        max_score[q_offset] = -FLT_MAX;
    }

    const auto accumulate_token = [&](const float2 kf, const float2 vf) {
        float score[q_heads_per_warp];
#pragma unroll
        for (int q_offset = 0; q_offset < q_heads_per_warp; ++q_offset) {
            float dot = fmaf(q0[q_offset], kf.x,
                             q1[q_offset] * kf.y);
            dot = warp_sum_broadcast_short(dot);
            score[q_offset] = dot * g.attn_scale;
        }

#pragma unroll
        for (int q_offset = 0; q_offset < q_heads_per_warp; ++q_offset) {
            // score is warp-broadcast, so this branch is warp-uniform.  The
            // usual two-scale online-softmax update evaluates exp(0) for one
            // of its scales on every token.  Select the nontrivial scale
            // explicitly instead: this preserves the same FP32 recurrence
            // while issuing exactly one exponential per query/token.
            if (score[q_offset] <= max_score[q_offset]) {
                const float new_scale =
                    __expf(score[q_offset] - max_score[q_offset]);
                normalizer[q_offset] += new_scale;
                out0[q_offset] =
                    fmaf(new_scale, vf.x, out0[q_offset]);
                out1[q_offset] =
                    fmaf(new_scale, vf.y, out1[q_offset]);
            } else {
                const float old_scale =
                    __expf(max_score[q_offset] - score[q_offset]);
                normalizer[q_offset] =
                    normalizer[q_offset] * old_scale + 1.0f;
                out0[q_offset] =
                    fmaf(1.0f, vf.x, out0[q_offset] * old_scale);
                out1[q_offset] =
                    fmaf(1.0f, vf.y, out1[q_offset] * old_scale);
                max_score[q_offset] = score[q_offset];
            }
        }
    };

    const int prefetched_tokens =
        position < PrefetchTokens ? position : PrefetchTokens;
    if constexpr (PrefetchTokens > 0) {
#pragma unroll
        for (int token = 0; token < PrefetchTokens; ++token) {
            if (token < prefetched_tokens) {
                accumulate_token(__bfloat1622float2(k_prefetched[token]),
                                 __bfloat1622float2(v_prefetched[token]));
            }
        }
    }

#pragma unroll 1
    for (int token = prefetched_tokens; token < seq_len; ++token) {
        const int cache_offset =
            token * cache_token_stride + cache_head_offset;
        const float2 kf = __bfloat1622float2(
            reinterpret_cast<const bf16x2 *>(k_layer + cache_offset)[lane]);
        const float2 vf = __bfloat1622float2(
            reinterpret_cast<const bf16x2 *>(v_layer + cache_offset)[lane]);
        accumulate_token(kf, vf);
    }

#pragma unroll
    for (int q_offset = 0; q_offset < q_heads_per_warp; ++q_offset) {
        const float inv_normalizer = 1.0f / normalizer[q_offset];
        reinterpret_cast<bf16x2 *>(
            g.attn_out.raw_ptr +
            (q_head_start + q_offset) * globals::head_dim)[lane] =
            __floats2bfloat162_rn(out0[q_offset] * inv_normalizer,
                                 out1[q_offset] * inv_normalizer);
    }
}

// Four-warps-per-KV-head variant that follows the production TK attention's
// 16-token online-softmax recurrence.  The complete first two historical
// blocks are demand-loaded into registers before the incoming PDL wait.  QK
// and the normalization state remain FP32; as in the TK MMA path, each
// exponentiated attention weight is rounded to BF16 before the BF16xBF16 PV
// product is accumulated in FP32.
template <bool PdlTrigger = true>
__launch_bounds__(128, 1) __global__ void
short_context_attention_direct_gqa_block16_reg32(
    const __grid_constant__ globals g) {
    static_assert(globals::head_dim == 64);
    static_assert(globals::num_attention_heads == 32);
    static_assert(globals::num_kv_heads == 8);
    constexpr int q_heads_per_kv =
        globals::num_attention_heads / globals::num_kv_heads;
    static_assert(q_heads_per_kv == 4);
    constexpr int block_tokens = globals::kv_block_size;
    static_assert(block_tokens == 16);
    constexpr int register_tokens = 32;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int q_head = kv_head * q_heads_per_kv + warp;
    const int layer = instruction_layer(g);
    const int position = static_cast<int>(g.pos_id);
    const int seq_len = position + 1;

    // This candidate is specialized to the formal P32/D128 decode interval,
    // whose first position is 32.  Token 32 and later may be producer-owned;
    // only tokens 0..31 are safe to demand-load before the wait.
    if (position < register_tokens || seq_len > short_context_max_seq) {
        if (tid == 0) {
            asm volatile("trap;\n");
        }
        return;
    }

    const int cache_token_stride =
        static_cast<int>(g.k_cache.rows()) * globals::head_dim;
    const int cache_layer_base =
        layer * static_cast<int>(g.k_cache.depth()) * cache_token_stride;
    const int cache_head_offset = kv_head * globals::head_dim;
    const bf16 *k_layer = g.k_cache.raw_ptr + cache_layer_base;
    const bf16 *v_layer = g.v_cache.raw_ptr + cache_layer_base;

    bf16x2 k_prefetched[register_tokens];
    bf16x2 v_prefetched[register_tokens];
#pragma unroll
    for (int token = 0; token < register_tokens; ++token) {
        const int cache_offset =
            token * cache_token_stride + cache_head_offset;
        k_prefetched[token] = reinterpret_cast<const bf16x2 *>(
            k_layer + cache_offset)[lane];
        v_prefetched[token] = reinterpret_cast<const bf16x2 *>(
            v_layer + cache_offset)[lane];
    }

    cudaGridDependencySynchronize();
    if constexpr (PdlTrigger) {
        if (tid == 0) {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }

    const float2 qf = __bfloat1622float2(
        reinterpret_cast<const bf16x2 *>(
            g.q_post_rope.raw_ptr + q_head * globals::head_dim)[lane]);
    const float softmax_temp = g.attn_scale * 1.44269504089f;
    float running_max = -FLT_MAX;
    float normalizer = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;

    // The first two complete blocks use only values retained in registers.
    // Both loops are fixed and fully unrolled so ptxas can scalarize the
    // arrays instead of lowering them to local memory.
#pragma unroll 2
    for (int block = 0; block < register_tokens / block_tokens; ++block) {
        float scores[block_tokens];
        float next_max = running_max;
#pragma unroll
        for (int offset = 0; offset < block_tokens; ++offset) {
            const int token = block * block_tokens + offset;
            const float2 kf =
                __bfloat1622float2(k_prefetched[token]);
            float score = fmaf(qf.x, kf.x, qf.y * kf.y);
            score = warp_sum_broadcast_short(score);
            scores[offset] = score;
            next_max = fmaxf(next_max, score);
        }

        const float old_scale =
            block == 0
                ? 0.0f
                : exp2f((running_max - next_max) * softmax_temp);
        float block_norm = 0.0f;
        float block_out0 = 0.0f;
        float block_out1 = 0.0f;
#pragma unroll
        for (int offset = 0; offset < block_tokens; ++offset) {
            const int token = block * block_tokens + offset;
            const float weight =
                exp2f((scores[offset] - next_max) * softmax_temp);
            block_norm += weight;
            const bf16 weight_bf16 = __float2bfloat16_rn(weight);
            const float pv_weight = __bfloat162float(weight_bf16);
            const float2 vf =
                __bfloat1622float2(v_prefetched[token]);
            block_out0 = fmaf(pv_weight, vf.x, block_out0);
            block_out1 = fmaf(pv_weight, vf.y, block_out1);
        }
        normalizer = fmaf(normalizer, old_scale, block_norm);
        out0 = fmaf(out0, old_scale, block_out0);
        out1 = fmaf(out1, old_scale, block_out1);
        running_max = next_max;
    }

    // The current token and later historical blocks are read only after QKV
    // completion.  Their K scores are retained for the block's PV pass, so K
    // is not loaded twice.
#pragma unroll 1
    for (int block_start = register_tokens; block_start < seq_len;
         block_start += block_tokens) {
        float scores[block_tokens];
        float next_max = running_max;
#pragma unroll
        for (int offset = 0; offset < block_tokens; ++offset) {
            const int token = block_start + offset;
            float score = -FLT_MAX;
            if (token < seq_len) {
                const int cache_offset =
                    token * cache_token_stride + cache_head_offset;
                const float2 kf = __bfloat1622float2(
                    reinterpret_cast<const bf16x2 *>(
                        k_layer + cache_offset)[lane]);
                score = fmaf(qf.x, kf.x, qf.y * kf.y);
                score = warp_sum_broadcast_short(score);
                next_max = fmaxf(next_max, score);
            }
            scores[offset] = score;
        }

        const float old_scale =
            exp2f((running_max - next_max) * softmax_temp);
        float block_norm = 0.0f;
        float block_out0 = 0.0f;
        float block_out1 = 0.0f;
#pragma unroll
        for (int offset = 0; offset < block_tokens; ++offset) {
            const int token = block_start + offset;
            if (token < seq_len) {
                const float weight =
                    exp2f((scores[offset] - next_max) * softmax_temp);
                block_norm += weight;
                const bf16 weight_bf16 = __float2bfloat16_rn(weight);
                const float pv_weight = __bfloat162float(weight_bf16);
                const int cache_offset =
                    token * cache_token_stride + cache_head_offset;
                const float2 vf = __bfloat1622float2(
                    reinterpret_cast<const bf16x2 *>(
                        v_layer + cache_offset)[lane]);
                block_out0 = fmaf(pv_weight, vf.x, block_out0);
                block_out1 = fmaf(pv_weight, vf.y, block_out1);
            }
        }
        normalizer = fmaf(normalizer, old_scale, block_norm);
        out0 = fmaf(out0, old_scale, block_out0);
        out1 = fmaf(out1, old_scale, block_out1);
        running_max = next_max;
    }

    const float inv_normalizer = 1.0f / normalizer;
    reinterpret_cast<bf16x2 *>(
        g.attn_out.raw_ptr + q_head * globals::head_dim)[lane] =
        __floats2bfloat162_rn(out0 * inv_normalizer,
                             out1 * inv_normalizer);
}

} // namespace attention_simt
} // namespace megakernel
