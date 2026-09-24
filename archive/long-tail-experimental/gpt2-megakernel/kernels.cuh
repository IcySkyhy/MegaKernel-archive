

/**

1. embedding_kernel  
2. positional_embedding_kernel  
3. dropout_kernel  
4. layernorm_kernel  
5. qkv_kernel  
6. softmax_kernel  
7. attention_matmul_kernel  
8. attn_proj_kernel  
9. fc_kernel  
10. gelu_kernel  
11. fc_proj_kernel


*/

// -----------------------------------------------------------------------------
// UTILS
// -----------------------------------------------------------------------------

__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// -----------------------------------------------------------------------------
// 1. TOKEN EMBEDDING
// -----------------------------------------------------------------------------

__device__ void embedding_kernel(
    float* out,        // output activations (flattened)
    const int* tokens, // token IDs
    float* wte,        // (V, C)
    int total_elems,   // equals B*T*C
    int C ,             // hidden dim,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    int t = idx / C;
    int c = idx % C;

    int tok = tokens[t];
    out[idx] = wte[tok * C + c];
}




// -----------------------------------------------------------------------------
// 2. POSITIONAL EMBEDDING
// -----------------------------------------------------------------------------

__device__ void positional_embedding_kernel(
    float* out,
    float* wpe,
    int total_elems,
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    int t = idx / C;
    int c = idx % C;

    out[idx] += wpe[t * C + c];
}

// -----------------------------------------------------------------------------
// 3. DROPOUT (no RNG here, assume mask passed)
// -----------------------------------------------------------------------------

__device__ void dropout_kernel(
    float* out,
    float* in,
    float* mask,
    float scale,
    int total_elems,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    out[idx] = in[idx] * mask[idx] * scale;
}

// -----------------------------------------------------------------------------
// 4. LAYERNORM
// -----------------------------------------------------------------------------

__device__ void layernorm_kernel(
    float* out,
    float* in,
    float* gamma,
    float* beta,
    int total_elems,
    int C,
    float eps,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    int t = idx / C;

    float mean = 0.0f;
    float var  = 0.0f;

    int start = t * C;
    for(int i = 0; i < C; i++) mean += in[start + i];
    mean /= C;

    for(int i = 0; i < C; i++){
        float d = in[start + i] - mean;
        var += d * d;
    }
    var = var / C + eps;
    float inv = rsqrtf(var);

    out[idx] = (in[idx] - mean) * inv * gamma[idx % C] + beta[idx % C];
}

// -----------------------------------------------------------------------------
// 5. QKV PROJECTION (linear)
// -----------------------------------------------------------------------------

__device__ void qkv_kernel(
    float* out,  // (3*C)
    float* in,   // (C)
    float* w,    // (3*C, C)
    float* b,    // (3*C)
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= 3 * C) return;

    float sum = b[idx];
    int row_start = idx * C;

    for(int j = 0; j < C; j++){
        sum += w[row_start + j] * in[j];
    }
    out[idx] = sum;
}

// -----------------------------------------------------------------------------
// 6. SOFTMAX
// -----------------------------------------------------------------------------

__device__ void softmax_kernel(
    float* out,
    float* in,
    int size,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float maxv = -1e9;
    for(int i = 0; i < size; i++)
        if(in[i] > maxv) maxv = in[i];

    float denom = 0.0f;
    for(int i = 0; i < size; i++)
        denom += expf(in[i] - maxv);

    out[idx] = expf(in[idx] - maxv) / denom;
}

// -----------------------------------------------------------------------------
// 7. ATTENTION APPLY (softmax(QK^T) * V)
// -----------------------------------------------------------------------------

__device__ void attention_matmul_kernel(
    float* out,     // C
    float* softA,   // seq_len
    float* V,       // seq_len * C
    int seq_len,
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float sum = 0.0f;
    for(int t = 0; t < seq_len; t++){
        sum += softA[t] * V[t * C + idx];
    }
    out[idx] = sum;
}

// -----------------------------------------------------------------------------
// 8. ATTENTION PROJ (linear)
// -----------------------------------------------------------------------------

__device__ void attn_proj_kernel(
    float* out,
    float* in,
    float* w,
    float* b,
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float sum = b[idx];
    for(int j = 0; j < C; j++){
        sum += w[idx * C + j] * in[j];
    }
    out[idx] = sum;
}

// -----------------------------------------------------------------------------
// 9. MLP: FC
// -----------------------------------------------------------------------------

__device__ void fc_kernel(
    float* out,
    float* in,
    float* w,
    float* b,
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    int outC = 4 * C;
    if (idx >= outC) return;

    float sum = b[idx];
    int row_start = idx * C;
    for(int j = 0; j < C; j++){
        sum += w[row_start + j] * in[j];
    }
    out[idx] = sum;
}

// -----------------------------------------------------------------------------
// 10. GELU
// -----------------------------------------------------------------------------

__device__ void gelu_kernel(
    float* out,
    float* in,
    int total,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = gelu(in[idx]);
}

// -----------------------------------------------------------------------------
// 11. FC PROJECTION (4*C -> C)
// -----------------------------------------------------------------------------

__device__ void fc_proj_kernel(
    float* out,
    float* in,
    float* w,
    float* b,
    int C,
int bidx
){
    int idx = bidx * blockDim.x + threadIdx.x;
    if (idx >= C) return;

    float sum = b[idx];
    for(int j = 0; j < 4*C; j++){
        sum += w[idx * (4*C) + j] * in[j];
    }
    out[idx] = sum;
}


