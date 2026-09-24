#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// 1. EMBEDDING LAYER (wte, wpe)
// ============================================================================

__device__ void embedding_lookup_kernel(
    const int* tokens,        // [batch, seq_len]
    const float* embed_table, // [vocab_size, embed_dim]
    float* output,            // [batch, seq_len, embed_dim]
    int seq_len,
    int embed_dim,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int batch = idx / (seq_len * embed_dim);
    int remainder = idx % (seq_len * embed_dim);
    int pos = remainder / embed_dim;
    int dim = remainder % embed_dim;
    
    int token_id = tokens[batch * seq_len + pos];
    output[idx] = embed_table[token_id * embed_dim + dim];
}

__device__ void add_position_embedding_kernel(
    float* x,                    // [batch, seq_len, embed_dim] - modified in place
    const float* pos_embed,      // [max_pos, embed_dim]
    int seq_len,
    int embed_dim,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int pos = (idx / embed_dim) % seq_len;
    int dim = idx % embed_dim;
    
    x[idx] += pos_embed[pos * embed_dim + dim];
}

// ============================================================================
// 2. LAYER NORM
// ============================================================================

__device__ void layernorm_kernel(
    const float* x,      // [batch, seq_len, dim]
    float* out,          // [batch, seq_len, dim]
    const float* gamma,  // [dim]
    const float* beta,   // [dim]
    int seq_len,
    int dim,
    float eps,
    int bidx
) {
    int batch_seq = bidx;  // each block handles one (batch, seq) position
    int tid = threadIdx.x;       // thread handles subset of dimensions
    
    // Shared memory for reduction
    __shared__ float mean_shared;
    __shared__ float var_shared;
    
    // Calculate mean
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += x[batch_seq * dim + i];
    }
    
    // Reduce within block
    __shared__ float sums[256];
    sums[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sums[tid] += sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean_shared = sums[0] / dim;
    }
    __syncthreads();
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = x[batch_seq * dim + i] - mean_shared;
        var_sum += diff * diff;
    }
    
    sums[tid] = var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sums[tid] += sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        var_shared = sums[0] / dim;
    }
    __syncthreads();
    
    // Normalize and apply affine transform
    float std = sqrtf(var_shared + eps);
    for (int i = tid; i < dim; i += blockDim.x) {
        int idx = batch_seq * dim + i;
        float normalized = (x[idx] - mean_shared) / std;
        out[idx] = gamma[i] * normalized + beta[i];
    }
}

// ============================================================================
// 3. LINEAR LAYER (Conv1D in GPT-2 is just a linear layer)
// ============================================================================

__device__ void linear_kernel(
    const float* x,      // [batch * seq_len, in_dim]
    const float* weight, // [out_dim, in_dim] - transposed
    const float* bias,   // [out_dim]
    float* out,          // [batch * seq_len, out_dim]
    int batch_seq,
    int in_dim,
    int out_dim,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int row = idx / out_dim;  // which input vector
    int col = idx % out_dim;  // which output dimension
    
    if (row >= batch_seq) return;
    
    float sum = 0.0f;
    for (int k = 0; k < in_dim; k++) {
        sum += x[row * in_dim + k] * weight[col * in_dim + k];
    }
    
    if (bias != nullptr) {
        sum += bias[col];
    }
    
    out[idx] = sum;
}

// ============================================================================
// 4. ATTENTION SOFTMAX
// ============================================================================

__device__ void softmax_kernel(
    float* x,        // [batch, num_heads, seq_len, seq_len]
    int seq_len,
    int bidx
) {
    int batch_head_row = bidx;  // each block handles one row
    int tid = threadIdx.x;
    
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    // Find max for numerical stability
    float local_max = -1e10f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = x[batch_head_row * seq_len + i];
        local_max = fmaxf(local_max, val);
    }
    
    __shared__ float maxs[256];
    maxs[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            maxs[tid] = fmaxf(maxs[tid], maxs[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        max_val = maxs[0];
    }
    __syncthreads();
    
    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        int idx = batch_head_row * seq_len + i;
        x[idx] = expf(x[idx] - max_val);
        local_sum += x[idx];
    }
    
    __shared__ float sums[256];
    sums[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sums[tid] += sums[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sum_exp = sums[0];
    }
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        x[batch_head_row * seq_len + i] /= sum_exp;
    }
}

// ============================================================================
// 5. ATTENTION MASK (Causal Mask)
// ============================================================================

__device__ void apply_causal_mask_kernel(
    float* scores,   // [batch, num_heads, seq_len, seq_len]
    int seq_len,
    float mask_value,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int q_pos = (idx / seq_len) % seq_len;
    int k_pos = idx % seq_len;
    
    // Causal mask: can only attend to previous positions
    if (k_pos > q_pos) {
        scores[idx] = mask_value;
    }
}

// ============================================================================
// 6. GELU ACTIVATION
// ============================================================================

__device__ void gelu_kernel(
    const float* x,
    float* out,
    int size,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = x[idx];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cube = val * val * val;
    float inner = 0.7978845608f * (val + 0.044715f * cube);
    out[idx] = 0.5f * val * (1.0f + tanhf(inner));
}

// ============================================================================
// 7. DROPOUT (Training only - simple version)
// ============================================================================

__device__ void dropout_kernel(
    float* x,
    const float* rand_vals,  // random values [0, 1]
    float dropout_prob,
    float scale,
    int size,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    if (rand_vals[idx] < dropout_prob) {
        x[idx] = 0.0f;
    } else {
        x[idx] *= scale;  // scale = 1.0 / (1.0 - dropout_prob)
    }
}

// ============================================================================
// 8. RESIDUAL CONNECTION
// ============================================================================

__device__ void residual_add_kernel(
    float* x,           // [batch, seq_len, dim] - modified in place
    const float* residual, // [batch, seq_len, dim]
    int size,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    x[idx] += residual[idx];
}

// ============================================================================
// 9. MATRIX MULTIPLY (for Q@K^T and attn@V)
// ============================================================================

__device__ void matmul_kernel(
    const float* A,  // [batch, m, k]
    const float* B,  // [batch, k, n]
    float* C,        // [batch, m, n]
    int batch,
    int m,
    int n,
    int k,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int b = idx / (m * n);
    int remainder = idx % (m * n);
    int row = remainder / n;
    int col = remainder % n;
    
    if (b >= batch) return;
    
    float sum = 0.0f;
    int a_offset = b * m * k;
    int b_offset = b * k * n;
    
    for (int i = 0; i < k; i++) {
        sum += A[a_offset + row * k + i] * B[b_offset + i * n + col];
    }

    C[b * m * n + row * n + col] = sum;

}

// ============================================================================
// 10. TRANSPOSE (for reshaping in attention)
// ============================================================================

__device__ void transpose_kernel(
    const float* in,  // [batch, seq, dim]
    float* out,       // [batch, dim, seq]
    int batch,
    int seq,
    int dim,
    int bidx
) {
    int idx = bidx * blockDim.x + threadIdx.x;
    int b = idx / (seq * dim);
    int remainder = idx % (seq * dim);
    int s = remainder / dim;
    int d = remainder % dim;
    
    if (b >= batch) return;
    
    int in_idx = b * seq * dim + s * dim + d;
    int out_idx = b * dim * seq + d * seq + s;
    out[out_idx] = in[in_idx];
}

// ============================================================================
// FULL FORWARD PASS - GPT-2 124M
// ============================================================================

// void gpt2_forward(
//     const int* tokens,              // [batch, seq_len] - input token IDs
//     float* logits,                  // [batch, seq_len, vocab_size] - output
    
//     // Embedding weights
//     const float* wte,               // [vocab_size, n_embd] = [50257, 768]
//     const float* wpe,               // [n_positions, n_embd] = [1024, 768]
    
//     // Layer weights (12 layers)
//     const float* ln_1_gamma,        // [12, n_embd]
//     const float* ln_1_beta,         // [12, n_embd]
//     const float* ln_2_gamma,        // [12, n_embd]
//     const float* ln_2_beta,         // [12, n_embd]
//     const float* c_attn_weight,     // [12, 3*n_embd, n_embd] = [12, 2304, 768]
//     const float* c_attn_bias,       // [12, 3*n_embd]
//     const float* c_proj_weight,     // [12, n_embd, n_embd]
//     const float* c_proj_bias,       // [12, n_embd]
//     const float* c_fc_weight,       // [12, 4*n_embd, n_embd] = [12, 3072, 768]
//     const float* c_fc_bias,         // [12, 4*n_embd]
//     const float* c_proj_mlp_weight, // [12, n_embd, 4*n_embd] = [12, 768, 3072]
//     const float* c_proj_mlp_bias,   // [12, n_embd]
    
//     // Final layer norm
//     const float* ln_f_gamma,        // [n_embd]
//     const float* ln_f_beta,         // [n_embd]
    
//     // LM head
//     const float* lm_head_weight,    // [vocab_size, n_embd] (tied with wte)
    
//     // Temporary buffers (pre-allocated on device)
//     float* temp_buffer,             // Large buffer for intermediate results
    
//     // Dimensions
//     int batch,
//     int seq_len
// ) {
//     // GPT-2 124M hyperparameters
//     const int n_embd = 768;
//     const int n_head = 12;
//     const int n_layer = 12;
//     const int vocab_size = 50257;
//     const int head_dim = n_embd / n_head;  // 64
//     const int n_inner = 4 * n_embd;        // 3072
//     const float eps = 1e-5f;
//     const int threads = 256;
    
//     // Allocate sections of temp buffer
//     int offset = 0;
//     float* x = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* residual = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* ln_out = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* qkv = temp_buffer + offset; offset += batch * seq_len * 3 * n_embd;
//     float* q = qkv;
//     float* k = qkv + batch * seq_len * n_embd;
//     float* v = qkv + 2 * batch * seq_len * n_embd;
//     float* scores = temp_buffer + offset; offset += batch * n_head * seq_len * seq_len;
//     float* attn_out = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* attn_proj = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* mlp_in = temp_buffer + offset; offset += batch * seq_len * n_embd;
//     float* mlp_hidden = temp_buffer + offset; offset += batch * seq_len * n_inner;
//     float* mlp_out = temp_buffer + offset; offset += batch * seq_len * n_embd;
    
//     // ========================================================================
//     // 1. EMBEDDING LAYER
//     // ========================================================================
    
//     // Token embeddings
//     {
//         int total = batch * seq_len * n_embd;
//         int blocks = (total + threads - 1) / threads;
//         embedding_lookup_kernel<<<blocks, threads>>>(
//             tokens, wte, x, seq_len, n_embd
//         );
//     }
    
//     // Add position embeddings
//     {
//         int total = batch * seq_len * n_embd;
//         int blocks = (total + threads - 1) / threads;
//         add_position_embedding_kernel<<<blocks, threads>>>(
//             x, wpe, seq_len, n_embd
//         );
//     }
    
//     // ========================================================================
//     // 2. TRANSFORMER LAYERS (repeat 12 times)
//     // ========================================================================
    
//     for (int layer = 0; layer < n_layer; layer++) {
        
//         // Save residual
//         cudaMemcpy(residual, x, batch * seq_len * n_embd * sizeof(float), 
//                    cudaMemcpyDeviceToDevice);
        
//         // --------------------------------------------------------------------
//         // 2a. LAYER NORM 1
//         // --------------------------------------------------------------------
//         {
//             int blocks = batch * seq_len;
//             layernorm_kernel<<<blocks, threads>>>(
//                 x, ln_out, 
//                 ln_1_gamma + layer * n_embd,
//                 ln_1_beta + layer * n_embd,
//                 seq_len, n_embd, eps
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2b. ATTENTION: QKV Projection
//         // --------------------------------------------------------------------
//         {
//             int batch_seq = batch * seq_len;
//             int total = batch_seq * 3 * n_embd;
//             int blocks = (total + threads - 1) / threads;
//             linear_kernel<<<blocks, threads>>>(
//                 ln_out,
//                 c_attn_weight + layer * 3 * n_embd * n_embd,
//                 c_attn_bias + layer * 3 * n_embd,
//                 qkv,
//                 batch_seq, n_embd, 3 * n_embd
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2c. ATTENTION: Compute Scores (Q @ K^T / sqrt(head_dim))
//         // --------------------------------------------------------------------
        
//         // For each head, compute Q @ K^T
//         for (int h = 0; h < n_head; h++) {
//             // Extract Q and K for this head
//             // Q: [batch, seq_len, head_dim], K: [batch, seq_len, head_dim]
//             // scores: [batch, seq_len, seq_len]
            
//             int batch_total = batch;
//             int m = seq_len;
//             int n = seq_len;
//             int kd = head_dim;
            
//             int total = batch_total * m * n;
//             int blocks = (total + threads - 1) / threads;
            
//             // Simple matmul Q @ K^T for this head
//             matmul_kernel<<<blocks, threads>>>(
//                 q + h * head_dim,           // Q for head h
//                 k + h * head_dim,           // K for head h  
//                 scores + h * seq_len * seq_len,  // output for head h
//                 batch_total, m, n, kd
//             );
//         }
        
//         // Scale by sqrt(head_dim)
//         {
//             int total = batch * n_head * seq_len * seq_len;
//             int blocks = (total + threads - 1) / threads;
//             float scale = 1.0f / sqrtf((float)head_dim);
            
//             // Simple scaling kernel
//             auto scale_kernel = [] __device__ (float* data, float s, int size) {
//                 int idx = blockIdx.x * blockDim.x + threadIdx.x;
//                 if (idx < size) data[idx] *= s;
//             };
//             // Note: This lambda won't compile, use actual kernel below
//         }
        
//         // Apply causal mask
//         {
//             int total = batch * n_head * seq_len * seq_len;
//             int blocks = (total + threads - 1) / threads;
//             apply_causal_mask_kernel<<<blocks, threads>>>(
//                 scores, seq_len, -1e10f
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2d. ATTENTION: Softmax
//         // --------------------------------------------------------------------
//         {
//             int blocks = batch * n_head * seq_len;  // one block per row
//             softmax_kernel<<<blocks, threads>>>(scores, seq_len);
//         }
        
//         // --------------------------------------------------------------------
//         // 2e. ATTENTION: Multiply by V
//         // --------------------------------------------------------------------
        
//         for (int h = 0; h < n_head; h++) {
//             int batch_total = batch;
//             int m = seq_len;
//             int n = head_dim;
//             int kd = seq_len;
            
//             int total = batch_total * m * n;
//             int blocks = (total + threads - 1) / threads;
            
//             matmul_kernel<<<blocks, threads>>>(
//                 scores + h * seq_len * seq_len,  // attention weights
//                 v + h * head_dim,                 // V for head h
//                 attn_out + h * head_dim,          // output for head h
//                 batch_total, m, n, kd
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2f. ATTENTION: Output Projection
//         // --------------------------------------------------------------------
//         {
//             int batch_seq = batch * seq_len;
//             int total = batch_seq * n_embd;
//             int blocks = (total + threads - 1) / threads;
//             linear_kernel<<<blocks, threads>>>(
//                 attn_out,
//                 c_proj_weight + layer * n_embd * n_embd,
//                 c_proj_bias + layer * n_embd,
//                 attn_proj,
//                 batch_seq, n_embd, n_embd
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2g. RESIDUAL CONNECTION 1
//         // --------------------------------------------------------------------
//         {
//             int total = batch * seq_len * n_embd;
//             int blocks = (total + threads - 1) / threads;
//             residual_add_kernel<<<blocks, threads>>>(
//                 attn_proj, residual, total
//             );
//         }
        
//         // Copy result to x for next stage
//         cudaMemcpy(x, attn_proj, batch * seq_len * n_embd * sizeof(float),
//                    cudaMemcpyDeviceToDevice);
        
//         // Save residual again for MLP
//         cudaMemcpy(residual, x, batch * seq_len * n_embd * sizeof(float),
//                    cudaMemcpyDeviceToDevice);
        
//         // --------------------------------------------------------------------
//         // 2h. LAYER NORM 2
//         // --------------------------------------------------------------------
//         {
//             int blocks = batch * seq_len;
//             layernorm_kernel<<<blocks, threads>>>(
//                 x, mlp_in,
//                 ln_2_gamma + layer * n_embd,
//                 ln_2_beta + layer * n_embd,
//                 seq_len, n_embd, eps
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2i. MLP: First Linear + GELU
//         // --------------------------------------------------------------------
//         {
//             int batch_seq = batch * seq_len;
//             int total = batch_seq * n_inner;
//             int blocks = (total + threads - 1) / threads;
//             linear_kernel<<<blocks, threads>>>(
//                 mlp_in,
//                 c_fc_weight + layer * n_inner * n_embd,
//                 c_fc_bias + layer * n_inner,
//                 mlp_hidden,
//                 batch_seq, n_embd, n_inner
//             );
//         }
        
//         // GELU activation
//         {
//             int total = batch * seq_len * n_inner;
//             int blocks = (total + threads - 1) / threads;
//             gelu_kernel<<<blocks, threads>>>(
//                 mlp_hidden, mlp_hidden, total
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2j. MLP: Second Linear
//         // --------------------------------------------------------------------
//         {
//             int batch_seq = batch * seq_len;
//             int total = batch_seq * n_embd;
//             int blocks = (total + threads - 1) / threads;
//             linear_kernel<<<blocks, threads>>>(
//                 mlp_hidden,
//                 c_proj_mlp_weight + layer * n_embd * n_inner,
//                 c_proj_mlp_bias + layer * n_embd,
//                 mlp_out,
//                 batch_seq, n_inner, n_embd
//             );
//         }
        
//         // --------------------------------------------------------------------
//         // 2k. RESIDUAL CONNECTION 2
//         // --------------------------------------------------------------------
//         {
//             int total = batch * seq_len * n_embd;
//             int blocks = (total + threads - 1) / threads;
//             residual_add_kernel<<<blocks, threads>>>(
//                 mlp_out, residual, total
//             );
//         }
        
//         // Copy result to x for next layer
//         cudaMemcpy(x, mlp_out, batch * seq_len * n_embd * sizeof(float),
//                    cudaMemcpyDeviceToDevice);
//     }
    
//     // ========================================================================
//     // 3. FINAL LAYER NORM
//     // ========================================================================
//     {
//         int blocks = batch * seq_len;
//         layernorm_kernel<<<blocks, threads>>>(
//             x, ln_out, ln_f_gamma, ln_f_beta, seq_len, n_embd, eps
//         );
//     }
    
//     // ========================================================================
//     // 4. LM HEAD (Final projection to vocabulary)
//     // ========================================================================
//     {
//         int batch_seq = batch * seq_len;
//         int total = batch_seq * vocab_size;
//         int blocks = (total + threads - 1) / threads;
//         linear_kernel<<<blocks, threads>>>(
//             ln_out,
//             lm_head_weight,  // typically tied with wte
//             nullptr,          // no bias
//             logits,
//             batch_seq, n_embd, vocab_size
//         );
//     }
    
//     cudaDeviceSynchronize();
// }

// ============================================================================
// HELPER: Scale kernel (needed for attention scaling)
// ============================================================================

__device__ void scale_kernel(float* data, float scale, int size, int bidx) {
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}