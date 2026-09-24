#include "common.cuh"
#include "convert.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// Precision alias matching llama.cpp's CUDA conventions
using half_t = half;
#define H2F(x) __half2float(x)
#define F2H(x) __float2half(x)

// =============================================================================
// Model constants (Qwen 3.5-0.8B)
// =============================================================================

constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3584;
constexpr int NUM_LAYERS = 24;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 8;
constexpr int FA_NUM_KV_HEADS = 2;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 16;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;
constexpr int NUM_WARPS = 512 / WARP_SIZE; // 16 warps for 512 block size

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight pointer packing structures
// =============================================================================

struct FullAttnWeights {
    const half_t *input_layernorm_weight;
    const half_t *q_proj_weight;
    const half_t *k_proj_weight;
    const half_t *v_proj_weight;
    const half_t *q_norm_weight;
    const half_t *k_norm_weight;
    const half_t *o_proj_weight;
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
    half_t *k_cache;
    half_t *v_cache;
};

struct DeltaNetWeights {
    const half_t *input_layernorm_weight;
    const half_t *qkv_proj_weight;
    const half_t *z_proj_weight;
    const half_t *beta_proj_weight;
    const half_t *alpha_proj_weight;
    const half_t *conv1d_weight;
    const half_t *a_log;
    const half_t *dt_bias;
    const half_t *norm_weight;
    const half_t *out_proj_weight;
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
    float *dn_state;
    float *conv_state;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Grid sync and Math helpers
// =============================================================================

struct AtomicGridSync {
    unsigned int * counter;
    unsigned int * generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() {
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int my_gen = local_gen;
            #if __CUDA_ARCH__ >= 700
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            #else
            asm volatile("membar.gl;" ::: "memory");
            #endif

            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                #if __CUDA_ARCH__ >= 700
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                #else
                asm volatile("membar.gl;" ::: "memory");
                #endif
                atomicAdd(generation, 1);
            } else {
                volatile unsigned int *vgen = (volatile unsigned int *)generation;
                while (*vgen <= my_gen) {}
            }
            local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        #if __CUDA_ARCH__ >= 700
        val += __shfl_down_sync(0xffffffff, val, offset);
        #else
        val += __shfl_down(val, offset);
        #endif
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    return expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
    #if __CUDA_ARCH__ >= 800
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    #elif __CUDA_ARCH__ >= 700
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    #else
    out = __ldg(ptr);
    #endif
    return out;
}

__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const half_t *act) {
    const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        sum += H2F(w[i]) * H2F(act[i]);
    return sum;
}

// =============================================================================
// Layer blocks implementations
// =============================================================================

__device__ void rmsnorm_redundant(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
        float v = H2F(__ldg(input + i));
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * w);
    }
    __syncthreads();
}

__device__ void rmsnorm_from_bf16(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
        float v = H2F(input[i]);
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * w);
    }
    __syncthreads();
}

__device__ void matvec_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0) output[m] = sum;
        }
    }
}

__device__ void matvec_gate_up_silu_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ gate_weight,
    const half_t *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *g_row = gate_weight + m * in_dim;
            const half_t *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
            #pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                output[m] = fast_silu(gate_sum) * up_sum;
        }
    }
}

__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(sum + H2F(residual[m]));
        }
    }
}

// =============================================================================
// Layers execution
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_q,
    float *__restrict__ g_kv,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    half_t *__restrict__ hidden_out,
    int position, int max_seq_len,
    int v_trans,
    half_t *__restrict__ shmem)
{
    half_t * k_cache = w.k_cache;
    half_t * v_cache = w.v_cache;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + RoPE + KV cache update
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            half_t *kc = k_cache + (position * FA_NUM_KV_HEADS + h) * FA_HEAD_DIM;
            half_t *vc = v_cache + (position * FA_NUM_KV_HEADS + h) * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            #if __CUDA_ARCH__ >= 700
            sc = __shfl_sync(0xffffffff, sc, 0);
            #else
            sc = __shfl(sc, 0);
            #endif
            for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                float normed = kh[i] * sc * H2F(__ldg(w.k_norm_weight + i));
                if (i < FA_ROTARY_DIM) {
                    float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                    float pv = kh[p]*sc*(1.0f+H2F(__ldg(w.k_norm_weight+p)));
                    float rotated = (i < FA_ROTARY_DIM/2) ? (normed*cv - pv*sv) : (pv*sv + normed*cv);
                    kc[i] = F2H(rotated);
                } else { kc[i] = F2H(normed); }
                if (v_trans) {
                    v_cache[i * (FA_NUM_KV_HEADS * max_seq_len) + h * max_seq_len + position] = F2H(vh[i]);
                } else {
                    vc[i] = F2H(vh[i]);
                }
            }
        }
    }
    // Q norm + RoPE
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                #if __CUDA_ARCH__ >= 700
                sc = __shfl_sync(0xffffffff, sc, 0);
                #else
                sc = __shfl(sc, 0);
                #endif
                for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    float normed = qh_ptr[i]*sc*H2F(__ldg(w.q_norm_weight+i));
                    if (i < FA_ROTARY_DIM) {
                        float fe = float(2*(i%(FA_ROTARY_DIM/2))) / float(FA_ROTARY_DIM);
                        float freq = float(position) / powf(FA_ROPE_THETA, fe);
                        float cv = cosf(freq), sv = sinf(freq);
                        int p = (i < FA_ROTARY_DIM/2) ? i+FA_ROTARY_DIM/2 : i-FA_ROTARY_DIM/2;
                        float pv = qh_ptr[p]*sc*H2F(__ldg(w.q_norm_weight+p));
                        qh_ptr[i] = (i < FA_ROTARY_DIM/2) ? (normed*cv-pv*sv) : (pv*sv+normed*cv);
                    } else { qh_ptr[i] = normed; }
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0;
            float out_acc[EPL], q_local[EPL];
            for (int e = 0; e < EPL; e++) { out_acc[e] = 0; q_local[e] = q_head[lane_id*EPL+e]; }

            for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *k_pos = k_cache + (pos * FA_NUM_KV_HEADS + kvh) * FA_HEAD_DIM;
                const half_t *v_pos = v_cache + (pos * FA_NUM_KV_HEADS + kvh) * FA_HEAD_DIM;
                float score = 0;
                for (int e = 0; e < EPL; e++) score += q_local[e] * H2F(__ldg(k_pos + lane_id*EPL+e));
                score = warp_reduce_sum(score) * attn_scale;
                #if __CUDA_ARCH__ >= 700
                score = __shfl_sync(0xffffffff, score, 0);
                #else
                score = __shfl(score, 0);
                #endif
                float old_max = max_score; max_score = fmaxf(max_score, score);
                float exp_diff = fast_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                float wt = fast_exp(score - max_score);
                for (int e = 0; e < EPL; e++) {
                    int i = lane_id * EPL + e;
                    float v_val;
                    if (v_trans) {
                        v_val = H2F(__ldg(v_cache + i * (FA_NUM_KV_HEADS * max_seq_len) + kvh * max_seq_len + pos));
                    } else {
                        v_val = H2F(__ldg(v_pos + i));
                    }
                    out_acc[e] = out_acc[e]*exp_diff + wt*v_val;
                }
            }
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            for (int e = 0; e < EPL; e++) g_activations[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max_score[ww] > -INFINITY) gm = fmaxf(gm, s_max_score[ww]);
                float ts = 0; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = fast_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                        for (int e = 0; e < EPL; e++) fo[e] += g_activations[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id*EPL+e;
                    out_head[idx] = fo[e]*rcp * fast_sigmoid(gate_ptr[idx]);
                }
            }
            __syncthreads();
        }
    }
    grid.sync();

    // Phase 4: O projection + residual
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += 512) s_attn[i] = g_attn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += 512) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    half_t *__restrict__ hidden_out,
    int dn_layer_idx,
    half_t *__restrict__ shmem)
{
    float * dn_state = w.dn_state;
    float * conv_buf = w.conv_state;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    // Phase 2+3: Conv1d + recurrence
    if (block_id < DN_NUM_HEADS) {
        int h = block_id;
        float *layer_conv = conv_buf;

        // Conv1d + SiLU
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        int head_ch[3] = {h*DN_KEY_DIM, DN_QK_SIZE+h*DN_KEY_DIM, 2*DN_QK_SIZE+h*DN_VALUE_DIM};
        for (int region = 0; region < 3; region++) {
            int ch_base = head_ch[region], ch_count = (region < 2) ? DN_KEY_DIM : DN_VALUE_DIM;
            float *dst = (region == 0) ? s_q : (region == 1) ? s_k : s_v;
            for (int c = threadIdx.x; c < ch_count; c += 512) {
                int ch = ch_base + c;
                float h0 = layer_conv[ch*3];
                float h1 = layer_conv[ch*3+1];
                float h2 = layer_conv[ch*3+2];
                float h3 = g_qkv[ch];
                float co = h0 * H2F(__ldg(w.conv1d_weight + ch*4))
                         + h1 * H2F(__ldg(w.conv1d_weight + ch*4+1))
                         + h2 * H2F(__ldg(w.conv1d_weight + ch*4+2))
                         + h3 * H2F(__ldg(w.conv1d_weight + ch*4+3));
                layer_conv[ch*3]   = h1;
                layer_conv[ch*3+1] = h2;
                layer_conv[ch*3+2] = h3;
                dst[c] = fast_silu(co);
            }
        }

        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = 1.0f / (1.0f + expf(-g_beta[h]));
            float a_log_val = H2F(__ldg(w.a_log + h));
            float dt_b = H2F(__ldg(w.dt_bias + h));
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));
            float decay = expf(-expf(a_log_val) * sp);
            if (isnan(decay) || isinf(decay)) {
                decay = 0.0f;
            }
            g_alpha[h] = decay;
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.31370849f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            #if __CUDA_ARCH__ >= 700
            n = __shfl_sync(0xffffffff,n,0);
            #else
            n = __shfl(n,0);
            #endif
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            #if __CUDA_ARCH__ >= 700
            n = __shfl_sync(0xffffffff,n,0);
            #else
            n = __shfl(n,0);
            #endif
            for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot product
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Recurrence step
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

        #pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
            #pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            #if __CUDA_ARCH__ >= 700
            stk = __shfl_sync(0xffffffff,stk,0); sqv = __shfl_sync(0xffffffff,sqv,0);
            #else
            stk = __shfl(stk,0); sqv = __shfl(sqv,0);
            #endif
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = o_j;
            #pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += 512) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += 512) {
                float normed = out_head[i] * rstd * H2F(__ldg(w.norm_weight + i));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = normed * gate;
            }
        }
    }
    grid.sync();

    // Phase 4: Out projection + residual
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += 512) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += 512) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// Forward declaration for F16→F32 helper (defined after decode_kernel)
__global__ void ggml_cuda_f16_to_f32(const half * __restrict__ src, float * __restrict__ dst, int n);

// =============================================================================
// Main decode kernel
// =============================================================================

__global__ void __launch_bounds__(512, 1)
decode_kernel(
    const half_t *__restrict__ embed_weight,
    const half_t *__restrict__ final_norm_weight,
    const half_t *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    half_t *__restrict__ fa_k_cache,
    half_t *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    half_t *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int input_token_id, int position, int max_seq_len,
    int v_trans, int lm_head_trans)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;



    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};

    // Shared memory: large enough for max(HIDDEN_SIZE bf16, INTERMEDIATE_SIZE f32)
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    // Populate hidden_buffer with input token embedding (always contiguous/non-transposed)
    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += blockDim.x) {
            hidden_buffer[i] = __ldg(embed_weight + input_token_id * HIDDEN_SIZE + i);
        }
    }
    grid.sync();

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                hidden_buffer, dn_layer_idx, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                position, max_seq_len, v_trans, shmem_bf16);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
            float v = H2F(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += 512) {
            float wt = H2F(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
    grid.sync();

    // Phase 6: LM Head Projection + Repetition Penalty
    int vocab_per_block = (VOCAB_SIZE + num_blocks - 1) / num_blocks;
    int v_start = block_id * vocab_per_block;
    int v_end = min(v_start + vocab_per_block, VOCAB_SIZE);

    for (int v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            int idx = lm_head_trans ? (i * VOCAB_SIZE + v) : (v * HIDDEN_SIZE + i);
            sum += g_normalized[i] * H2F(__ldg(lm_head_weight + idx));
        }

        hidden_buffer[v] = F2H(sum);
    }
}

// =============================================================================
// C++ Operator Hook
// =============================================================================

void ggml_cuda_op_fused_megakernel_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    // Unpack 64-bit pointers from op_params
    auto unpack_ptr = [&](int idx) -> void * {
        uint64_t val = ((uint64_t) dst->op_params[idx+1] << 32) | (uint32_t) dst->op_params[idx];
        return (void *) val;
    };

    // weights_flat is a host array of ggml_tensor* (one per weight slot)
    const ggml_tensor ** weights_flat = (const ggml_tensor **) unpack_ptr(0);
    // final_norm, embed, lm_head are also ggml_tensor* packed as addresses
    const ggml_tensor * final_norm_t  = (const ggml_tensor *) unpack_ptr(2);
    const ggml_tensor * embed_t       = (const ggml_tensor *) unpack_ptr(4);
    const ggml_tensor * lm_head_t     = (const ggml_tensor *) unpack_ptr(6);

    int n_vocab = dst->op_params[8];
    int input_token_id = dst->op_params[9];
    int position = dst->op_params[10];
    int max_seq_len = dst->op_params[11];
    float repetition_penalty = 1.0f;
    std::memcpy(&repetition_penalty, &dst->op_params[12], sizeof(float));
    int v_trans = dst->op_params[14];
    static bool printed_v_trans = false;
    if (!printed_v_trans) {
        FILE * f = fopen("megakernel_debug.log", "w");
        if (f) {
            fprintf(f, "v_trans = %d, n_vocab = %d, input_token_id = %d, position = %d\n", v_trans, n_vocab, input_token_id, position);
            fclose(f);
        }
        printed_v_trans = true;
    }

    // Allocate persistent GPU buffers for recurrent states, KV cache, and packed weights
    static LayerWeights * d_layer_weights = nullptr;
    static float * dn_states = nullptr;
    static float * conv_bufs = nullptr;
    static float * seen_token_mask = nullptr;
    static std::vector<void *> d_allocations;
    static const ggml_tensor * last_embed_t = nullptr;

    if (embed_t != last_embed_t) {
        for (void * ptr : d_allocations) {
            cudaFree(ptr);
        }
        d_allocations.clear();
        if (d_layer_weights) { cudaFree(d_layer_weights); d_layer_weights = nullptr; }
        if (dn_states) { cudaFree(dn_states); dn_states = nullptr; }
        if (conv_bufs) { cudaFree(conv_bufs); conv_bufs = nullptr; }
        if (seen_token_mask) { cudaFree(seen_token_mask); seen_token_mask = nullptr; }
        last_embed_t = embed_t;
    }

    // Helper: dequantize a ggml_tensor to a new FP16 device buffer (or alias if already F16)
    auto dequant_to_f16 = [&](const ggml_tensor * t, int64_t nelems) -> const half_t * {
        if (!t) return nullptr;

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, t->data);
        bool is_device = false;
        #if CUDART_VERSION >= 10000
        is_device = (err == cudaSuccess && attr.type == cudaMemoryTypeDevice);
        #else
        is_device = (err == cudaSuccess && attr.memoryType == cudaMemoryTypeDevice);
        #endif

        FILE * df = fopen("megakernel_debug.log", "a");
        if (df) {
            fprintf(df, "dequant_to_f16: tensor type=%d, nelems=%lld, is_device=%d\n", (int)t->type, (long long)nelems, (int)is_device);
            fclose(df);
        }

        if (t->type == GGML_TYPE_F16 && is_device) {
            return (const half_t *) t->data;
        }

        fprintf(stderr, "MEGALLOC: nelems=%lld size=%lld\n", (long long)nelems, (long long)(nelems * sizeof(half_t)));
        fflush(stderr);
        half_t * buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, nelems * sizeof(half_t)));
        d_allocations.push_back(buf);

        if (t->type == GGML_TYPE_F16) {
            CUDA_CHECK(cudaMemcpyAsync(buf, t->data, nelems * sizeof(half_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            if (is_device) {
                to_fp16_cuda_t fn = ggml_get_to_fp16_cuda(t->type);
                GGML_ASSERT(fn != nullptr);
                fn(t->data, buf, nelems, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            } else {
                size_t size = ggml_row_size(t->type, nelems);
                void * temp_dev = nullptr;
                CUDA_CHECK(cudaMalloc(&temp_dev, size));
                CUDA_CHECK(cudaMemcpyAsync(temp_dev, t->data, size, cudaMemcpyHostToDevice, stream));
                to_fp16_cuda_t fn = ggml_get_to_fp16_cuda(t->type);
                GGML_ASSERT(fn != nullptr);
                fn(temp_dev, buf, nelems, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaFree(temp_dev));
            }
        }
        return buf;
    };

    // Dequantize per-layer weights on first call
    static const half_t * d_embed_weight      = nullptr;
    static const half_t * d_final_norm_weight = nullptr;
    static const half_t * d_lm_head_weight    = nullptr;

    static LayerWeights h_weights[24];

    // Detect model reload (e.g. across llama-bench iterations) and invalidate weight caches
    {
        static const ggml_tensor * prev_embed_t = nullptr;
        static const ggml_tensor * prev_final_norm_t = nullptr;
        static const ggml_tensor * prev_lm_head_t = nullptr;
        bool changed = (embed_t != prev_embed_t || final_norm_t != prev_final_norm_t || lm_head_t != prev_lm_head_t);
        // fprintf(stderr, "MEGACHECK: embed_t=%p prev=%p final_norm_t=%p prev=%p lm_head_t=%p prev=%p changed=%d d_lw=%p\n",
        //         (void*)embed_t, (void*)prev_embed_t,
        //         (void*)final_norm_t, (void*)prev_final_norm_t,
        //         (void*)lm_head_t, (void*)prev_lm_head_t,
        //         (int)changed, (void*)d_layer_weights);
        // fflush(stderr);
        if (changed) {
            d_layer_weights      = nullptr;
            d_embed_weight       = nullptr;
            d_final_norm_weight  = nullptr;
            d_lm_head_weight     = nullptr;
            dn_states            = nullptr;
            conv_bufs            = nullptr;
            seen_token_mask      = nullptr;
            printed_v_trans      = false;
            prev_embed_t         = embed_t;
            prev_final_norm_t    = final_norm_t;
            prev_lm_head_t       = lm_head_t;
        }
    }

    if (d_layer_weights == nullptr) {
        fprintf(stderr, "MEGAINIT: initializing weight caches (d_layer_weights was null)\n");
        fflush(stderr);
        CUDA_CHECK(cudaMalloc(&seen_token_mask, 4096 * sizeof(float)));
        CUDA_CHECK(cudaMemset(seen_token_mask, 0, 4096 * sizeof(float)));

        // Dequantize global weights using actual tensor element counts
        auto ne = [](const ggml_tensor * t) -> int64_t { return t ? ggml_nelements(t) : 1; };

        d_embed_weight      = dequant_to_f16(embed_t,      ne(embed_t));
        d_final_norm_weight = dequant_to_f16(final_norm_t, ne(final_norm_t));
        d_lm_head_weight    = dequant_to_f16(lm_head_t,    ne(lm_head_t));

        // Dequantize per-layer weights
        const int layer_types[24] = { 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1 };
        for (int il = 0; il < 24; ++il) {
            h_weights[il].layer_type = layer_types[il];
            const ggml_tensor ** L = &weights_flat[il * 16];
            if (layer_types[il] == 0) {
                h_weights[il].dn.input_layernorm_weight     = dequant_to_f16(L[0],  ne(L[0]));
                h_weights[il].dn.qkv_proj_weight            = dequant_to_f16(L[1],  ne(L[1]));
                h_weights[il].dn.z_proj_weight               = dequant_to_f16(L[2],  ne(L[2]));
                h_weights[il].dn.beta_proj_weight            = dequant_to_f16(L[3],  ne(L[3]));
                h_weights[il].dn.alpha_proj_weight           = dequant_to_f16(L[4],  ne(L[4]));
                h_weights[il].dn.conv1d_weight               = dequant_to_f16(L[5],  ne(L[5]));
                h_weights[il].dn.a_log                       = dequant_to_f16(L[6],  ne(L[6]));
                h_weights[il].dn.dt_bias                     = dequant_to_f16(L[7],  ne(L[7]));
                h_weights[il].dn.norm_weight                 = dequant_to_f16(L[8],  ne(L[8]));
                h_weights[il].dn.out_proj_weight             = dequant_to_f16(L[9],  ne(L[9]));
                h_weights[il].dn.post_attn_layernorm_weight  = dequant_to_f16(L[10], ne(L[10]));
                h_weights[il].dn.gate_proj_weight            = dequant_to_f16(L[11], ne(L[11]));
                h_weights[il].dn.up_proj_weight              = dequant_to_f16(L[12], ne(L[12]));
                h_weights[il].dn.down_proj_weight            = dequant_to_f16(L[13], ne(L[13]));
            } else {
                h_weights[il].fa.input_layernorm_weight      = dequant_to_f16(L[0],  ne(L[0]));
                h_weights[il].fa.q_proj_weight               = dequant_to_f16(L[1],  ne(L[1]));
                h_weights[il].fa.k_proj_weight               = dequant_to_f16(L[2],  ne(L[2]));
                h_weights[il].fa.v_proj_weight               = dequant_to_f16(L[3],  ne(L[3]));
                h_weights[il].fa.q_norm_weight               = dequant_to_f16(L[4],  ne(L[4]));
                h_weights[il].fa.k_norm_weight               = dequant_to_f16(L[5],  ne(L[5]));
                h_weights[il].fa.o_proj_weight               = dequant_to_f16(L[6],  ne(L[6]));
                h_weights[il].fa.post_attn_layernorm_weight  = dequant_to_f16(L[7],  ne(L[7]));
                h_weights[il].fa.gate_proj_weight            = dequant_to_f16(L[8],  ne(L[8]));
                h_weights[il].fa.up_proj_weight              = dequant_to_f16(L[9],  ne(L[9]));
                h_weights[il].fa.down_proj_weight            = dequant_to_f16(L[10], ne(L[10]));
            }
        }

        CUDA_CHECK(cudaMalloc(&d_layer_weights, sizeof(LayerWeights) * 24));
        fprintf(stderr, "MEGAINIT: done dequantizing all weights, d_layer_weights=%p\n", (void*)d_layer_weights);
        fflush(stderr);
    } else {
        // fprintf(stderr, "MEGASKIP: using cached weights, d_layer_weights=%p\n", (void*)d_layer_weights);
        // fflush(stderr);
    }

    // Update dynamic cache/state pointers for all layers
    int head = dst->op_params[13];
    for (int il = 0; il < 24; ++il) {
        const ggml_tensor ** L = &weights_flat[il * 16];
        if (h_weights[il].layer_type == 0) {
            h_weights[il].dn.dn_state   = L[14] ? ((float *) L[14]->data + head * L[14]->ne[0]) : nullptr;
            h_weights[il].dn.conv_state = L[15] ? ((float *) L[15]->data + head * L[15]->ne[0]) : nullptr;
        } else {
            h_weights[il].fa.k_cache    = L[11] ? (half_t *) L[11]->data : nullptr;
            h_weights[il].fa.v_cache    = L[12] ? (half_t *) L[12]->data : nullptr;
        }
    }

    // Copy updated LayerWeights to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_layer_weights, h_weights, sizeof(LayerWeights) * 24, cudaMemcpyHostToDevice, stream));

    const half_t * embed_weight      = d_embed_weight;
    const half_t * final_norm_weight = d_final_norm_weight;
    const half_t * lm_head_weight    = d_lm_head_weight;

    const LayerWeights * layer_weights = d_layer_weights;

    // Allocate persistent grid sync barriers
    static unsigned int * d_barrier_counter    = nullptr;
    static unsigned int * d_barrier_generation = nullptr;
    if (d_barrier_counter == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_barrier_counter,    sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_barrier_generation, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_barrier_counter,    0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_barrier_generation, 0, sizeof(unsigned int)));
    }

    // Determine SM count for this device
    static int num_blocks  = 0;
    static int block_size  = 512;
    if (num_blocks == 0) {
        cudaDeviceProp prop;
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        num_blocks = prop.multiProcessorCount; // 68 on RTX 3080 Laptop
    }

    // Allocate per-token intermediate scratchpads (persistent across decode calls)
    static float   * d_activations  = nullptr;
    static half_t  * d_residual     = nullptr;
    static float   * d_qkv_scratch  = nullptr;
    static float   * d_kv_scratch   = nullptr;
    static float   * d_attn_out     = nullptr;
    static float   * d_mlp_inter    = nullptr;
    static float   * d_z_scratch    = nullptr;
    static float   * d_beta_scratch = nullptr;
    static float   * d_alpha_scratch= nullptr;
    static float   * d_normalized   = nullptr;
    // Separate half_t staging buffer for kernel logit output (VOCAB_SIZE elements)
    static half_t  * d_logits_h     = nullptr;

    if (d_activations == nullptr) {
        int max_scratch = (HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE);
        CUDA_CHECK(cudaMalloc(&d_activations,   max_scratch        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residual,      HIDDEN_SIZE        * sizeof(half_t)));
        CUDA_CHECK(cudaMalloc(&d_qkv_scratch,   DN_CONV_CHANNELS   * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kv_scratch,    FA_KV_SIZE * 2     * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_attn_out,      FA_Q_SIZE          * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mlp_inter,     INTERMEDIATE_SIZE  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z_scratch,     DN_V_SIZE          * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta_scratch,  DN_NUM_HEADS        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_alpha_scratch, DN_NUM_HEADS        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_normalized,    HIDDEN_SIZE        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits_h,      VOCAB_SIZE         * sizeof(half_t)));
    }

    // Reset barrier state before each call so local_gen=0 is always correct
    CUDA_CHECK(cudaMemsetAsync(d_barrier_counter,    0, sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_barrier_generation, 0, sizeof(unsigned int), stream));

    int lm_head_trans = dst->op_params[15];

    // Launch persistent decode megakernel — outputs F16 logits into d_logits_h
    decode_kernel<<<num_blocks, block_size, 0, stream>>>(
        embed_weight, final_norm_weight, lm_head_weight, layer_weights,
        nullptr, nullptr, dn_states, conv_bufs,
        d_logits_h, d_activations, d_residual, d_qkv_scratch, d_kv_scratch,
        d_attn_out, d_mlp_inter, d_z_scratch, d_beta_scratch, d_alpha_scratch,
        d_normalized, d_barrier_counter, d_barrier_generation,
        seen_token_mask, repetition_penalty, input_token_id, position, max_seq_len,
        v_trans, lm_head_trans
    );
    CUDA_CHECK(cudaGetLastError());

    // Quick sync + peek at first logit disabled for max performance
    /*
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
    if (captureStatus == cudaStreamCaptureStatusNone) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            half first[2];
            CUDA_CHECK(cudaMemcpy(first, d_logits_h, 2*sizeof(half), cudaMemcpyDeviceToHost));
            fprintf(stderr, "MEGAKERNEL: pos=%d tok=%d logits[0]=%f logits[1]=%f\n",
                    position, input_token_id,
                    __half2float(first[0]),
                    __half2float(first[1]));
        }
    }
    */

    // Convert F16 logits → F32 into dst->data (what llama-context.cpp expects)
    float  * dst_f32   = (float *) dst->data;
    int      n_vocab_i = n_vocab;
    int n_threads_cv   = 256;
    int n_blocks_cv    = (n_vocab_i + n_threads_cv - 1) / n_threads_cv;
    ggml_cuda_f16_to_f32<<<n_blocks_cv, n_threads_cv, 0, stream>>>(d_logits_h, dst_f32, n_vocab_i);
    CUDA_CHECK(cudaGetLastError());
}

// Helper kernel for F16 → F32 conversion used by the megakernel launcher
__global__ void ggml_cuda_f16_to_f32(const half * __restrict__ src, float * __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __half2float(src[idx]);
}
