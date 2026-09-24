#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>
#include "header.cuh"

// #include "kernels.cuh"
#include "t.cuh"

#define MAX_SYNC 1000 

__device__ int g_block_counter[MAX_SYNC];  // must be zero-initialized

__device__ void syncblocks(int total_blocks, int stage) {
    // Stage corresponds to g_block_counter[stage]

    __syncthreads();

    // One thread per block increments stage counter
    if (threadIdx.x == 0) {
        atomicAdd(&g_block_counter[stage], 1);
    }

    __syncthreads();

    // Spin until all blocks arrive
    while (atomicAdd(&g_block_counter[stage], 0) < total_blocks) {
        __threadfence();  // ensure visibility across SMs
    }

    __syncthreads();
}



// typedef struct {
//     floatX* wte;      // (V, C)
//     floatX* wpe;      // (maxT, C)
//     floatX* ln1w;     // (L, C)
//     floatX* ln1b;     // (L, C)
//     floatX* qkvw;     // (L, 3*C, C)
//     floatX* qkvb;     // (L, 3*C)
//     floatX* attprojw; // (L, C, C)
//     floatX* attprojb; // (L, C)
//     floatX* ln2w;     // (L, C)
//     floatX* ln2b;     // (L, C)
//     floatX* fcw;      // (L, 4*C, C)
//     floatX* fcb;      // (L, 4*C)
//     floatX* fcprojw;  // (L, C, 4*C)
//     floatX* fcprojb;  // (L, C)
//     floatX* lnfw;     // (C)
//     floatX* lnfb;     // (C)
// } ParameterTensors;


// typedef struct {
//     // Input token IDs
//     int* tokens;              // (B, T)

//     // Embedding outputs
//     floatX* x;                // (B, T, C) — main activations buffer

//     // Attention buffers
//     floatX* qkv;              // (L, B, T, 3*C)
//     floatX* q;                // (L, B, T, C)
//     floatX* k;                // (L, B, T, C)
//     floatX* v;                // (L, B, T, C)
//     floatX* att_scores;       // (L, B, H, T, T)
//     floatX* att_probs;        // (L, B, H, T, T)
//     floatX* att_out;          // (L, B, T, C)

//     // MLP buffers
//     floatX* fc1;              // (L, B, T, 4*C)
//     floatX* fc2;              // (L, B, T, C)

//     // Final logits
//     floatX* logits;           // (B, T, V)

//     // Optional dropout mask
//     floatX* dropout_mask;     // (B, T, C)
// } ActivationTensors;


const int n_embd = 768;
const int n_head = 12;
const int n_layer = 12;
const int vocab_size = 50257;
const int head_dim = n_embd / n_head;  // 64
const int n_inner = 4 * n_embd;        // 3072
const float eps = 1e-5f;
const int threads = 256;
const int num_blocks = 512;
const int batch = 8;



__global__ void kernel(ActivationTensors* act, ParameterTensors* params) {
    // const int n_embd = 768;
    // const int n_head = 12;
    // const int n_layer = 12;
    // const int vocab_size = 50257;
    // const int head_dim = n_embd / n_head;  // 64
    // const int n_inner = 4 * n_embd;        // 3072
    // const float eps = 1e-5f;
    // const int threads = 1024;
    // const int num_blocks = 512;
    // const int batch = 8;
    int phase = 0;
    // __syncthreads();
    // embedding_kernel(act, params, B, 1, C);
    
    // -----------------------------------------
    // 1. Token Embedding + Positional Embedding
    // -----------------------------------------
    int seq_len = 256;

    {
        int total = batch * seq_len * n_embd;
        int blocks = (total + threads - 1) / threads;
        for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            embedding_lookup_kernel(
                act->tokens, params->wte, act->x, seq_len, n_embd, i
            );
        }
    }

    syncblocks(gridDim.x, phase++);


    // __syncthreads();

    // Add position embeddings
    {
        int total = batch * seq_len * n_embd;
        int blocks = (total + threads - 1) / threads;
        for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            add_position_embedding_kernel(
                act->x, params->wpe, seq_len, n_embd, i
            );
        }
    }
    
    syncblocks(gridDim.x, phase++);

    // // dropout_kernel(
    // //     act.x, act.x, B*T*C, blockIdx.x
    // // );

    // -----------------------------------------
    // 2. Loop over transformer layers
    // -----------------------------------------
    for(int layer = 0; layer < 12; layer++)
    {
        // --------------------------------------------------------------------
        // 2a. LAYER NORM 1
        // --------------------------------------------------------------------
        {
            int blocks = batch * seq_len;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                layernorm_kernel(
                    act->x + layer * batch * seq_len * n_embd, 
                    act->x + (layer + 1) * batch * seq_len * n_embd, 
                    params->ln1w + layer * n_embd,
                    params->ln1b + layer * n_embd,
                    seq_len, n_embd, eps, i
                );
            }
        }
        
        syncblocks(gridDim.x, phase++);

        // --------------------------------------------------------------------
        // 2b. ATTENTION: QKV Projection
        // --------------------------------------------------------------------
        {
            int batch_seq = batch * seq_len;
            int total = batch_seq * 3 * n_embd;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                linear_kernel(
                    act->x + (layer + 1) * batch * seq_len * n_embd,
                    params->qkvw + layer * 3 * n_embd * n_embd,
                    params->qkvb + layer * 3 * n_embd,
                    act->qkv + layer * batch_seq * 3 * n_embd,
                    batch_seq, n_embd, 3 * n_embd, i
                );
            }
        }

        if()

        syncblocks(gridDim.x, phase++);


        for (int h = 0; h < n_head; h++) {
            // Extract Q and K for this head
            // Q: [batch, seq_len, head_dim], K: [batch, seq_len, head_dim]
            // scores: [batch, seq_len, seq_len]
            
            int batch_total = batch;
            int m = seq_len;
            int n = seq_len;
            int kd = head_dim;
            
            int total = batch_total * m * n; //  8 * 2k * 2k
            int blocks = (total + threads - 1) / threads; // 8 * 2k * 2k / 256
            // Simple matmul Q @ K^T for this head
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                matmul_kernel(
                    act->qkv + layer * batch_total * 3 * n_embd + h * head_dim,           // Q for head h
                    act->qkv + layer * batch_total * 3 * n_embd + batch * seq_len * n_embd + h * head_dim,           // K for head h  
                    act->att_scores + layer * batch_total * n_head * seq_len * seq_len + h * seq_len * seq_len,  // output for head h
                    batch_total, m, n, kd, i
                );
            }
            syncblocks(gridDim.x, phase++);

        }
        // if(threadIdx.x == 0){
        //     printf("Layer %d\n", layer);
        // }

        // Scale by sqrt(head_dim)
        {
            int total = batch * n_head * seq_len * seq_len;
            int blocks = (total + threads - 1) / threads;
            float scale = 1.0f / sqrtf((float)head_dim);
            
            // Simple scaling kernel
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            // auto scale_kernel = [] __device__ (float* data, float s, int size) {
            //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
            //     if (idx < size) data[idx] *= s;
            // };
                scale_kernel(
                    act->att_scores + layer * batch * n_head * seq_len * seq_len,
                    scale,
                    batch * n_head * seq_len * seq_len,
                    i
                );
            }
            syncblocks(gridDim.x, phase++);

            // Note: This lambda won't compile, use actual kernel below
        }


         // Apply causal mask
        {
            int total = batch * n_head * seq_len * seq_len;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                apply_causal_mask_kernel(
                    act->att_scores + layer * batch * n_head * seq_len * seq_len,
                    seq_len, -1e10f, i
                );
            }
        }

        syncblocks(gridDim.x, phase++);


        // --------------------------------------------------------------------
        // 2d. ATTENTION: Softmax
        // --------------------------------------------------------------------
        {
            int blocks = batch * n_head * seq_len;  // one block per row
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                softmax_kernel(act->att_scores + layer * batch * n_head * seq_len * seq_len, seq_len, i);
            }
        }


        syncblocks(gridDim.x, phase++);

        // --------------------------------------------------------------------
        // 2e. ATTENTION: Multiply by V
        // --------------------------------------------------------------------
        
        for (int h = 0; h < n_head; h++) {
            int batch_total = batch;
            int m = seq_len;
            int n = head_dim;
            int kd = seq_len;
            
            int total = batch_total * m * n;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            // matmul_kernel<<<blocks, threads>>>(
                matmul_kernel(
                    act->att_scores + layer * batch_total * n_head * seq_len * seq_len + h * seq_len * seq_len,  // attention weights
                    act->qkv + layer * batch_total * 3 * n_embd + 2 * batch * seq_len * n_embd + h * head_dim,                 // V for head h
                    act->att_out + layer * batch_total * n_embd + h * head_dim,          // output for head h
                    batch_total, m, n, kd, i
                );
            }
        }
        
        syncblocks(gridDim.x, phase++);

        // --------------------------------------------------------------------
        // 2f. ATTENTION: Output Projection
        // --------------------------------------------------------------------
        {
            int batch_seq = batch * seq_len;
            int total = batch_seq * n_embd;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                linear_kernel(
                    act->att_out + layer * batch_seq * n_embd,
                    params->attprojw + layer * n_embd * n_embd,
                    params->attprojb + layer * n_embd,
                    act->x + (layer + 1) * batch_seq * n_embd, // Using x as attn_proj output
                    batch_seq, n_embd, n_embd, i
                );
            }
        }

        syncblocks(gridDim.x, phase++);
        
        
        // --------------------------------------------------------------------
        // 2g. RESIDUAL CONNECTION 1
        // --------------------------------------------------------------------
        {
            int total = batch * seq_len * n_embd;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                residual_add_kernel(
                    act->x + (layer + 1) * batch * seq_len * n_embd,
                    act->x + layer * batch * seq_len * n_embd,
                    total, i
                );
            }
        }
        syncblocks(gridDim.x, phase++);
        
        // Copy result to x for next stage
        // cudaMemcpy(x, attn_proj, batch * seq_len * n_embd * sizeof(float),
        //            cudaMemcpyDeviceToDevice);
        
        // Save residual again for MLP
        // cudaMemcpy(residual, x, batch * seq_len * n_embd * sizeof(float),
        //            cudaMemcpyDeviceToDevice);
        
        // --------------------------------------------------------------------
        // 2h. LAYER NORM 2
        // --------------------------------------------------------------------
        {
            int blocks = batch * seq_len;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                layernorm_kernel(
                    act->x + (layer + 1) * batch * seq_len * n_embd, // Input is the output of previous residual
                    act->fc1 + layer * batch * seq_len * n_inner,    // Output to fc1 buffer
                    params->ln2w + layer * n_embd,
                    params->ln2b + layer * n_embd,
                    seq_len, n_embd, eps, i
                );
            }
        }
        syncblocks(gridDim.x, phase++);

        
        // --------------------------------------------------------------------
        // 2i. MLP: First Linear + GELU
        // --------------------------------------------------------------------
        {
            int batch_seq = batch * seq_len;
            int total = batch_seq * n_inner;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                linear_kernel(
                    act->fc1 + layer * batch_seq * n_inner, // Input is the output of LN2
                    params->fcw + layer * n_inner * n_embd,
                    params->fcb + layer * n_inner,
                    act->fc1 + layer * batch_seq * n_inner, // Output to fc1 buffer (in-place for GELU)
                    batch_seq, n_embd, n_inner, i
                );
            }
        }
        syncblocks(gridDim.x, phase++);
        
        // GELU activation
        {
            int total = batch * seq_len * n_inner;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                gelu_kernel(
                    act->fc1 + layer * batch * seq_len * n_inner, // Input is fc1
                    act->fc1 + layer * batch * seq_len * n_inner, // Output to fc1 (in-place)
                    total, i
                );
            }
        }
        syncblocks(gridDim.x, phase++);

        
        // --------------------------------------------------------------------
        // 2j. MLP: Second Linear
        // --------------------------------------------------------------------
        {
            int batch_seq = batch * seq_len;
            int total = batch_seq * n_embd;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                linear_kernel(
                    act->fc1 + layer * batch_seq * n_inner, // Input is fc1 (after GELU)
                    params->fcprojw + layer * n_embd * n_inner,
                    params->fcprojb + layer * n_embd,
                    act->fc2 + layer * batch_seq * n_embd, // Output to fc2 buffer
                    batch_seq, n_inner, n_embd, i
                );
            }
        }
        syncblocks(gridDim.x, phase++);

        
        // --------------------------------------------------------------------
        // 2k. RESIDUAL CONNECTION 2
        // --------------------------------------------------------------------
        {
            int total = batch * seq_len * n_embd;
            int blocks = (total + threads - 1) / threads;
            for(int i = blockIdx.x ; i < blocks; i += num_blocks){
                residual_add_kernel(
                    act->fc2 + layer * batch * seq_len * n_embd, // Output of MLP
                    act->x + (layer + 1) * batch * seq_len * n_embd, // Input to MLP (residual)
                    total, i
                );
            }
        }
        
        syncblocks(gridDim.x, phase++);
        
        // Copy result to x for next layer
        // The output of the current layer (after residual) becomes the input for the next layer
        // We are using `act->x` as the main activation buffer, indexed by layer.
        // So, `act->x + (layer + 2) * batch * seq_len * n_embd` would be the input for the next layer.
        // For the last layer, this will be the input to the final layer norm.
        // The `mlp_out` buffer (act->fc2) already contains the result of the current layer's MLP + residual.
        // So we just need to make sure the next layer's input points to this.
        // This is implicitly handled by how `act->x` is indexed in the next iteration.
        // For now, let's assume `act->x` is updated in place for the current layer's output.
        // This means `act->x + (layer + 1) * ...` should hold the output of the current layer.
        // The `residual_add_kernel` above writes to `act->fc2`, so we need to copy `act->fc2` to `act->x + (layer + 1) * ...`
        // if `act->x` is meant to be the primary activation buffer for the next layer's input.
        // Given the structure, it seems `act->x + (layer + 1) * ...` is meant to be the output of the current layer.
        // So, the `residual_add_kernel` should write directly to `act->x + (layer + 1) * ...`
        // Let's adjust the `residual_add_kernel` call to reflect this.
        // The previous `residual_add_kernel` wrote to `attn_proj` (which was `act->x + (layer + 1) * ...`).
    }


    // ========================================================================
    // 3. FINAL LAYER NORM
    // ========================================================================
    {
        int blocks = batch * seq_len;
        for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            layernorm_kernel(
                act->x + (n_layer) * batch * seq_len * n_embd, // Input is the output of the last layer
                act->x + (n_layer + 1) * batch * seq_len * n_embd, // Output to the same buffer (in-place)
                params->lnfw,
                params->lnfb,
                seq_len, n_embd, eps, i
            );
        }
    }

        syncblocks(gridDim.x, phase++);

    
    // ========================================================================
    // 4. LM HEAD (Final projection to vocabulary)
    // ========================================================================
    {
        int batch_seq = batch * seq_len;
        int total = batch_seq * vocab_size;
        int blocks = (total + threads - 1) / threads;
        for(int i = blockIdx.x ; i < blocks; i += num_blocks){
            linear_kernel(
                act->x + (n_layer + 1) * batch * seq_len * n_embd, // Input is the output of final LN
                params->wte,  // tied with wte
                nullptr,          // no bias
                act->logits,
                batch_seq, n_embd, vocab_size, i
            );
        }
    }
        syncblocks(gridDim.x, phase++);

  
    if(blockIdx.x == 0 && threadIdx.x == 0){
        for(int i = 0; i < 13; i++){
            printf("%f \n", (act->x + i * batch * seq_len * n_embd)[0]);
        }
    }




}

int main() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    int num_sms = prop.multiProcessorCount;
    int threads_per_sm = prop.maxThreadsPerMultiProcessor;

    printf("SMs: %d\n", num_sms);
    printf("Threads per SM: %d\n", threads_per_sm);

    // Launch exactly one block per SM
    int blocks = num_sms;
    int threads = threads_per_sm;

    printf("Launching kernel <<<%d, %d>>>\n", blocks, threads);


    
    int maxT = 256;

    ParameterTensors* params_dev = allocate_parameters(vocab_size, n_embd, n_layer, maxT);
    
    int* tokens = (int*)malloc(256 * sizeof(int));
    tokens[0] = 15496;
    tokens[1] = 11;
    tokens[2] = 995;
    tokens[3] = 0;

    int toks[] = {49488,   314,   481,   407,  2740,    11,  4249, 49671,    26,  1114,
           616,  5848,   318,  2626,   832,   220, 0};
    for(int i = 0; i < sizeof(toks)/sizeof(int); i++){
        tokens[i] = toks[i];
    }


    load_gpt2_weights("./gpt2_weights.bin", params_dev, vocab_size, n_embd, n_layer, maxT);
    ActivationTensors host = allocate_activations(batch, maxT, n_embd, vocab_size, n_layer, n_head);
    // Move struct to device
    ActivationTensors* act_dev;
    CUDA_CHECK(cudaMalloc(&act_dev, sizeof(ActivationTensors)));
    CUDA_CHECK(cudaMemcpy(act_dev, &host, sizeof(ActivationTensors), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(host.tokens, tokens, 256 * sizeof(int), cudaMemcpyHostToDevice));

    kernel<<<num_sms, 256>>>(act_dev, params_dev);
    cudaDeviceSynchronize();


    // dump_activations_npy("acts", act_dev, batch, maxT, n_embd, vocab_size, n_layer, n_head);

    float* logits_cpu = (float*)malloc(batch * maxT * vocab_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(logits_cpu, host.logits, batch * maxT * vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < 12; i++){
        float mx = -1e100;
        int idx = 0;
        for(int j = 0; j < vocab_size; j++){
            if(mx < logits_cpu[i * vocab_size + j]){
                mx = logits_cpu[i * vocab_size + j];
                idx = j;
            }
        }
        printf("%d %f \n", idx, mx);
    }


    free_parameters(params_dev);
    return 0;
}
