#include <stdio.h>

typedef float floatX; // modify if needed

typedef struct {
    floatX* wte;      // (V, C)
    floatX* wpe;      // (maxT, C)
    floatX* ln1w;     // (L, C)
    floatX* ln1b;     // (L, C)
    floatX* qkvw;     // (L, 3*C, C)
    floatX* qkvb;     // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w;     // (L, C)
    floatX* ln2b;     // (L, C)
    floatX* fcw;      // (L, 4*C, C)
    floatX* fcb;      // (L, 4*C)
    floatX* fcprojw;  // (L, C, 4*C)
    floatX* fcprojb;  // (L, C)
    floatX* lnfw;     // (C)
    floatX* lnfb;     // (C)
} ParameterTensors;


#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)


ParameterTensors* allocate_parameters(int V, int C, int L, int maxT) {
    ParameterTensors host;

    size_t size_wte      = V * C * sizeof(floatX);
    size_t size_wpe      = maxT * C * sizeof(floatX);
    size_t size_ln       = L * C * sizeof(floatX);
    size_t size_qkvw     = L * 3*C * C * sizeof(floatX);
    size_t size_qkvb     = L * 3*C * sizeof(floatX);
    size_t size_attprojw = L * C * C * sizeof(floatX);
    size_t size_attprojb = L * C * sizeof(floatX);
    size_t size_fcw      = L * 4*C * C * sizeof(floatX);
    size_t size_fcb      = L * 4*C * sizeof(floatX);
    size_t size_fcprojw  = L * C * 4*C * sizeof(floatX);
    size_t size_fcprojb  = L * C * sizeof(floatX);
    size_t size_lnf      = C * sizeof(floatX);

    CUDA_CHECK(cudaMalloc(&host.wte,      size_wte));
    CUDA_CHECK(cudaMalloc(&host.wpe,      size_wpe));
    CUDA_CHECK(cudaMalloc(&host.ln1w,     size_ln));
    CUDA_CHECK(cudaMalloc(&host.ln1b,     size_ln));
    CUDA_CHECK(cudaMalloc(&host.qkvw,     size_qkvw));
    CUDA_CHECK(cudaMalloc(&host.qkvb,     size_qkvb));
    CUDA_CHECK(cudaMalloc(&host.attprojw, size_attprojw));
    CUDA_CHECK(cudaMalloc(&host.attprojb, size_attprojb));
    CUDA_CHECK(cudaMalloc(&host.ln2w,     size_ln));
    CUDA_CHECK(cudaMalloc(&host.ln2b,     size_ln));
    CUDA_CHECK(cudaMalloc(&host.fcw,      size_fcw));
    CUDA_CHECK(cudaMalloc(&host.fcb,      size_fcb));
    CUDA_CHECK(cudaMalloc(&host.fcprojw,  size_fcprojw));
    CUDA_CHECK(cudaMalloc(&host.fcprojb,  size_fcprojb));
    CUDA_CHECK(cudaMalloc(&host.lnfw,     size_lnf));
    CUDA_CHECK(cudaMalloc(&host.lnfb,     size_lnf));

    // allocate device ParameterTensors struct
    ParameterTensors* device_ptr;
    CUDA_CHECK(cudaMalloc(&device_ptr, sizeof(ParameterTensors)));

    // copy host struct pointers → device copy of struct
    CUDA_CHECK(cudaMemcpy(device_ptr, &host, sizeof(ParameterTensors), cudaMemcpyHostToDevice));

    return device_ptr;
}



void free_parameters(ParameterTensors* params_dev) {
    ParameterTensors host;
    cudaMemcpy(&host, params_dev, sizeof(ParameterTensors), cudaMemcpyDeviceToHost);

    cudaFree(host.wte);
    cudaFree(host.wpe);
    cudaFree(host.ln1w);
    cudaFree(host.ln1b);
    cudaFree(host.qkvw);
    cudaFree(host.qkvb);
    cudaFree(host.attprojw);
    cudaFree(host.attprojb);
    cudaFree(host.ln2w);
    cudaFree(host.ln2b);
    cudaFree(host.fcw);
    cudaFree(host.fcb);
    cudaFree(host.fcprojw);
    cudaFree(host.fcprojb);
    cudaFree(host.lnfw);
    cudaFree(host.lnfb);

    cudaFree(params_dev);
}


typedef struct {
    // Input token IDs
    int* tokens;              // (B, T)

    // Embedding outputs
    floatX* x;                // (B, T, C) — main activations buffer

    // Attention buffers
    floatX* qkv;              // (L, B, T, 3*C)
    floatX* q;                // (L, B, T, C)
    floatX* k;                // (L, B, T, C)
    floatX* v;                // (L, B, T, C)
    floatX* att_scores;       // (L, B, H, T, T)
    floatX* att_probs;        // (L, B, H, T, T)
    floatX* att_out;          // (L, B, T, C)

    // MLP buffers
    floatX* fc1;              // (L, B, T, 4*C)
    floatX* fc2;              // (L, B, T, C)

    // Final logits
    floatX* logits;           // (B, T, V)

    // Optional dropout mask
    floatX* dropout_mask;     // (B, T, C)
} ActivationTensors;


ActivationTensors allocate_activations(int B, int T, int C, int V, int L, int H) {
    ActivationTensors host;

    int BT = B * T;
    int BTC = B * T * C;
    int BT3C = B * T * 3*C;
    int BT4C = B * T * 4*C;
    int BTCC = B * T * C * C;
    int BTTV = B * T * V;
    int BT_att = B * H * T * T;

    CUDA_CHECK(cudaMalloc(&host.tokens, BT * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&host.x, (L + 2) * BTC * sizeof(floatX)));

    CUDA_CHECK(cudaMalloc(&host.qkv, L * BT3C * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.q,   L * BTC * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.k,   L * BTC * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.v,   L * BTC * sizeof(floatX)));

    CUDA_CHECK(cudaMalloc(&host.att_scores, L * BT_att * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.att_probs,  L * BT_att * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.att_out,    L * BTC * sizeof(floatX)));

    CUDA_CHECK(cudaMalloc(&host.fc1,  L * BT4C * sizeof(floatX)));
    CUDA_CHECK(cudaMalloc(&host.fc2,  L * BTC * sizeof(floatX)));

    CUDA_CHECK(cudaMalloc(&host.logits, BTTV * sizeof(floatX)));

    CUDA_CHECK(cudaMalloc(&host.dropout_mask, BTC * sizeof(floatX)));

    

    return host;
}


void free_activations(ActivationTensors* act_dev) {
    ActivationTensors host;
    cudaMemcpy(&host, act_dev, sizeof(ActivationTensors), cudaMemcpyDeviceToHost);

    cudaFree(host.tokens);
    cudaFree(host.x);

    cudaFree(host.qkv);
    cudaFree(host.q);
    cudaFree(host.k);
    cudaFree(host.v);

    cudaFree(host.att_scores);
    cudaFree(host.att_probs);
    cudaFree(host.att_out);

    cudaFree(host.fc1);
    cudaFree(host.fc2);

    cudaFree(host.logits);
    cudaFree(host.dropout_mask);

    cudaFree(act_dev);
}


#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

// Helper to write a tensor to file with metadata
void write_tensor(FILE* f, const char* name, void* device_ptr, size_t num_elements, 
                  size_t element_size, const char* dtype) {
    if (device_ptr == NULL) {
        printf("  Skipping %s (NULL pointer)\n", name);
        return;
    }
    
    // Write metadata header
    int name_len = strlen(name);
    fwrite(&name_len, sizeof(int), 1, f);
    fwrite(name, sizeof(char), name_len, f);
    
    size_t total_bytes = num_elements * element_size;
    fwrite(&num_elements, sizeof(size_t), 1, f);
    fwrite(&element_size, sizeof(size_t), 1, f);
    
    int dtype_len = strlen(dtype);
    fwrite(&dtype_len, sizeof(int), 1, f);
    fwrite(dtype, sizeof(char), dtype_len, f);
    
    // Allocate host buffer and copy from device
    void* host_buffer = malloc(total_bytes);
    if (!host_buffer) {
        printf("Error: Could not allocate host buffer for %s\n", name);
        exit(1);
    }
    
    CUDA_CHECK(cudaMemcpy(host_buffer, device_ptr, total_bytes, cudaMemcpyDeviceToHost));
    
    // Write data
    fwrite(host_buffer, element_size, num_elements, f);
    
    free(host_buffer);
    printf("  ✓ Wrote %s: %zu elements (%.2f MB)\n", name, num_elements, 
           (float)total_bytes / (1024.0f * 1024.0f));
}

// Dump all activations to a binary file
void dump_activations(const char* filename, ActivationTensors* acts, 
                      int B, int T, int C, int V, int L, int H) {
    printf("\nDumping activations to %s...\n", filename);
    
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        exit(1);
    }
    
    // Write file header with dimensions
    const char* magic = "GPT2ACT";
    fwrite(magic, sizeof(char), 7, f);
    fwrite(&B, sizeof(int), 1, f);
    fwrite(&T, sizeof(int), 1, f);
    fwrite(&C, sizeof(int), 1, f);
    fwrite(&V, sizeof(int), 1, f);
    fwrite(&L, sizeof(int), 1, f);
    fwrite(&H, sizeof(int), 1, f);
    
    int num_tensors = 13;  // Total number of tensors we'll write
    fwrite(&num_tensors, sizeof(int), 1, f);
    
    // Calculate sizes
    int BT = B * T;
    int BTC = B * T * C;
    int BT3C = B * T * 3 * C;
    int BT4C = B * T * 4 * C;
    int BTTV = B * T * V;
    int BT_att = B * H * T * T;
    
    // Write each tensor
    write_tensor(f, "tokens", acts->tokens, BT, sizeof(int), "int32");
    write_tensor(f, "x", acts->x, (L + 2) * BTC, sizeof(floatX), "float32");
    write_tensor(f, "qkv", acts->qkv, L * BT3C, sizeof(floatX), "float32");
    write_tensor(f, "q", acts->q, L * BTC, sizeof(floatX), "float32");
    write_tensor(f, "k", acts->k, L * BTC, sizeof(floatX), "float32");
    write_tensor(f, "v", acts->v, L * BTC, sizeof(floatX), "float32");
    write_tensor(f, "att_scores", acts->att_scores, L * BT_att, sizeof(floatX), "float32");
    write_tensor(f, "att_probs", acts->att_probs, L * BT_att, sizeof(floatX), "float32");
    write_tensor(f, "att_out", acts->att_out, L * BTC, sizeof(floatX), "float32");
    write_tensor(f, "fc1", acts->fc1, L * BT4C, sizeof(floatX), "float32");
    write_tensor(f, "fc2", acts->fc2, L * BTC, sizeof(floatX), "float32");
    write_tensor(f, "logits", acts->logits, BTTV, sizeof(floatX), "float32");
    write_tensor(f, "dropout_mask", acts->dropout_mask, BTC, sizeof(floatX), "float32");
    
    fclose(f);
    printf("✓ Successfully dumped all activations!\n");
}

// Dump a single tensor (useful for debugging specific layers)
void dump_single_tensor(const char* filename, const char* tensor_name, 
                        void* device_ptr, size_t num_elements, 
                        size_t element_size, const char* dtype) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        exit(1);
    }
    
    write_tensor(f, tensor_name, device_ptr, num_elements, element_size, dtype);
    fclose(f);
    printf("✓ Dumped %s to %s\n", tensor_name, filename);
}

// Dump activations for a specific layer only
void dump_layer_activations(const char* filename, ActivationTensors* acts,
                            int layer, int B, int T, int C, int H) {
    printf("\nDumping layer %d activations to %s...\n", layer, filename);
    
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        exit(1);
    }
    
    // Write header
    const char* magic = "GPT2LAY";
    fwrite(magic, sizeof(char), 7, f);
    fwrite(&layer, sizeof(int), 1, f);
    fwrite(&B, sizeof(int), 1, f);
    fwrite(&T, sizeof(int), 1, f);
    fwrite(&C, sizeof(int), 1, f);
    fwrite(&H, sizeof(int), 1, f);
    
    int BT = B * T;
    int BTC = B * T * C;
    int BT3C = B * T * 3 * C;
    int BT4C = B * T * 4 * C;
    int BT_att = B * H * T * T;
    
    // Offsets for this layer
    floatX* qkv_layer = acts->qkv + layer * BT3C;
    floatX* q_layer = acts->q + layer * BTC;
    floatX* k_layer = acts->k + layer * BTC;
    floatX* v_layer = acts->v + layer * BTC;
    floatX* att_scores_layer = acts->att_scores + layer * BT_att;
    floatX* att_probs_layer = acts->att_probs + layer * BT_att;
    floatX* att_out_layer = acts->att_out + layer * BTC;
    floatX* fc1_layer = acts->fc1 + layer * BT4C;
    floatX* fc2_layer = acts->fc2 + layer * BTC;
    floatX* x_layer = acts->x + layer * BTC;
    
    int num_tensors = 10;
    fwrite(&num_tensors, sizeof(int), 1, f);
    
    write_tensor(f, "x_in", x_layer, BTC, sizeof(floatX), "float32");
    write_tensor(f, "qkv", qkv_layer, BT3C, sizeof(floatX), "float32");
    write_tensor(f, "q", q_layer, BTC, sizeof(floatX), "float32");
    write_tensor(f, "k", k_layer, BTC, sizeof(floatX), "float32");
    write_tensor(f, "v", v_layer, BTC, sizeof(floatX), "float32");
    write_tensor(f, "att_scores", att_scores_layer, BT_att, sizeof(floatX), "float32");
    write_tensor(f, "att_probs", att_probs_layer, BT_att, sizeof(floatX), "float32");
    write_tensor(f, "att_out", att_out_layer, BTC, sizeof(floatX), "float32");
    write_tensor(f, "fc1", fc1_layer, BT4C, sizeof(floatX), "float32");
    write_tensor(f, "fc2", fc2_layer, BTC, sizeof(floatX), "float32");
    
    fclose(f);
    printf("✓ Successfully dumped layer %d activations!\n", layer);
}

// Print statistics about a tensor (min, max, mean) - useful for debugging
void print_tensor_stats(const char* name, floatX* device_ptr, size_t num_elements) {
    floatX* host_buffer = (floatX*)malloc(num_elements * sizeof(floatX));
    CUDA_CHECK(cudaMemcpy(host_buffer, device_ptr, num_elements * sizeof(floatX), 
                          cudaMemcpyDeviceToHost));
    
    floatX min_val = host_buffer[0];
    floatX max_val = host_buffer[0];
    double sum = 0.0;
    
    for (size_t i = 0; i < num_elements; i++) {
        floatX val = host_buffer[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    double mean = sum / num_elements;
    
    printf("%s: min=%.6f, max=%.6f, mean=%.6f\n", name, min_val, max_val, mean);
    
    free(host_buffer);
}


void write_npy_header(FILE* f, int ndim, const size_t* shape, const char* dtype) {
    // Write magic number
    fwrite("\x93NUMPY", 1, 6, f);
    
    // Version 1.0
    unsigned char version[2] = {1, 0};
    fwrite(version, 1, 2, f);
    
    // Build header dict
    char header[256];
    char shape_str[128] = "";
    
    // Format shape as tuple
    if (ndim == 1) {
        snprintf(shape_str, sizeof(shape_str), "(%zu,)", shape[0]);
    } else if (ndim == 2) {
        snprintf(shape_str, sizeof(shape_str), "(%zu,%zu)", shape[0], shape[1]);
    } else if (ndim == 3) {
        snprintf(shape_str, sizeof(shape_str), "(%zu,%zu,%zu)", shape[0], shape[1], shape[2]);
    } else if (ndim == 4) {
        snprintf(shape_str, sizeof(shape_str), "(%zu,%zu,%zu,%zu)", shape[0], shape[1], shape[2], shape[3]);
    } else if (ndim == 5) {
        snprintf(shape_str, sizeof(shape_str), "(%zu,%zu,%zu,%zu,%zu)", 
                 shape[0], shape[1], shape[2], shape[3], shape[4]);
    }
    
    snprintf(header, sizeof(header), 
             "{'descr': '%s', 'fortran_order': False, 'shape': %s}", 
             dtype, shape_str);
    
    // Pad to 64-byte boundary
    unsigned short header_len = strlen(header) + 1;  // +1 for newline
    unsigned short padding = (64 - (10 + header_len) % 64) % 64;
    header_len += padding;
    
    fwrite(&header_len, 2, 1, f);
    fwrite(header, 1, strlen(header), f);
    for (int i = 0; i < padding; i++) fwrite(" ", 1, 1, f);
    fwrite("\n", 1, 1, f);
}

void dump_tensor_npy(const char* filename, void* device_ptr, int ndim, 
                     const size_t* shape, const char* dtype, size_t elem_size) {
    if (device_ptr == NULL) {
        printf("  Skipping %s (NULL pointer)\n", filename);
        return;
    }
    
    // Calculate total elements
    size_t total_elements = 1;
    for (int i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }
    
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open %s\n", filename);
        return;
    }
    
    // Write .npy header
    write_npy_header(f, ndim, shape, dtype);
    
    // Copy data from device and write
    void* host_buffer = malloc(total_elements * elem_size);
    CUDA_CHECK(cudaMemcpy(host_buffer, device_ptr, total_elements * elem_size, 
                          cudaMemcpyDeviceToHost));
    fwrite(host_buffer, elem_size, total_elements, f);
    free(host_buffer);
    
    fclose(f);
    
    float size_mb = (float)(total_elements * elem_size) / (1024.0f * 1024.0f);
    printf("  ✓ Saved %s: ", filename);
    printf("shape=(");
    for (int i = 0; i < ndim; i++) {
        printf("%zu%s", shape[i], i < ndim-1 ? "," : "");
    }
    printf(") %.2f MB\n", size_mb);
}

// ============================================================================
// DUMP ALL ACTIVATIONS AS SEPARATE .NPY FILES
// ============================================================================

void dump_activations_npy(const char* output_dir, ActivationTensors* acts,
                          int B, int T, int C, int V, int L, int H) {
    printf("\nDumping activations to %s/ ...\n", output_dir);
    
    // Create output directory (platform-specific)
    #ifdef _WIN32
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir \"%s\" 2>nul", output_dir);
    system(cmd);
    #else
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", output_dir);
    system(cmd);
    #endif
    
    char filepath[512];
    size_t shape[5];
    
    // Tokens: (B, T)
    snprintf(filepath, sizeof(filepath), "%s/tokens.npy", output_dir);
    shape[0] = B; shape[1] = T;
    dump_tensor_npy(filepath, acts->tokens, 2, shape, "<i4", sizeof(int));
    
    // X (embeddings + layer outputs): (L+2, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/x.npy", output_dir);
    shape[0] = L + 2; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->x, 4, shape, "<f4", sizeof(floatX));
    
    // QKV: (L, B, T, 3*C)
    snprintf(filepath, sizeof(filepath), "%s/qkv.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = 3*C;
    dump_tensor_npy(filepath, acts->qkv, 4, shape, "<f4", sizeof(floatX));
    
    // Q: (L, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/q.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->q, 4, shape, "<f4", sizeof(floatX));
    
    // K: (L, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/k.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->k, 4, shape, "<f4", sizeof(floatX));
    
    // V: (L, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/v.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->v, 4, shape, "<f4", sizeof(floatX));
    
    // Attention scores: (L, B, H, T, T)
    snprintf(filepath, sizeof(filepath), "%s/att_scores.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = H; shape[3] = T; shape[4] = T;
    dump_tensor_npy(filepath, acts->att_scores, 5, shape, "<f4", sizeof(floatX));
    
    // Attention probs: (L, B, H, T, T)
    snprintf(filepath, sizeof(filepath), "%s/att_probs.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = H; shape[3] = T; shape[4] = T;
    dump_tensor_npy(filepath, acts->att_probs, 5, shape, "<f4", sizeof(floatX));
    
    // Attention output: (L, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/att_out.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->att_out, 4, shape, "<f4", sizeof(floatX));
    
    // MLP fc1: (L, B, T, 4*C)
    snprintf(filepath, sizeof(filepath), "%s/fc1.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = 4*C;
    dump_tensor_npy(filepath, acts->fc1, 4, shape, "<f4", sizeof(floatX));
    
    // MLP fc2: (L, B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/fc2.npy", output_dir);
    shape[0] = L; shape[1] = B; shape[2] = T; shape[3] = C;
    dump_tensor_npy(filepath, acts->fc2, 4, shape, "<f4", sizeof(floatX));
    
    // Logits: (B, T, V)
    snprintf(filepath, sizeof(filepath), "%s/logits.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = V;
    dump_tensor_npy(filepath, acts->logits, 3, shape, "<f4", sizeof(floatX));
    
    // Dropout mask: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/dropout_mask.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, acts->dropout_mask, 3, shape, "<f4", sizeof(floatX));
    
    // Save metadata as text
    snprintf(filepath, sizeof(filepath), "%s/metadata.txt", output_dir);
    FILE* meta = fopen(filepath, "w");
    if (meta) {
        fprintf(meta, "GPT-2 Activation Dump\n");
        fprintf(meta, "====================\n\n");
        fprintf(meta, "Dimensions:\n");
        fprintf(meta, "  Batch size (B): %d\n", B);
        fprintf(meta, "  Sequence length (T): %d\n", T);
        fprintf(meta, "  Embedding dim (C): %d\n", C);
        fprintf(meta, "  Vocab size (V): %d\n", V);
        fprintf(meta, "  Num layers (L): %d\n", L);
        fprintf(meta, "  Num heads (H): %d\n", H);
        fprintf(meta, "\nFiles:\n");
        fprintf(meta, "  tokens.npy        - Input token IDs (B, T)\n");
        fprintf(meta, "  x.npy             - Hidden states (L+2, B, T, C)\n");
        fprintf(meta, "  qkv.npy           - QKV projections (L, B, T, 3*C)\n");
        fprintf(meta, "  q.npy             - Query (L, B, T, C)\n");
        fprintf(meta, "  k.npy             - Key (L, B, T, C)\n");
        fprintf(meta, "  v.npy             - Value (L, B, T, C)\n");
        fprintf(meta, "  att_scores.npy    - Attention scores (L, B, H, T, T)\n");
        fprintf(meta, "  att_probs.npy     - Attention probs (L, B, H, T, T)\n");
        fprintf(meta, "  att_out.npy       - Attention output (L, B, T, C)\n");
        fprintf(meta, "  fc1.npy           - MLP hidden (L, B, T, 4*C)\n");
        fprintf(meta, "  fc2.npy           - MLP output (L, B, T, C)\n");
        fprintf(meta, "  logits.npy        - Final logits (B, T, V)\n");
        fprintf(meta, "  dropout_mask.npy  - Dropout mask (B, T, C)\n");
        fclose(meta);
        printf("  ✓ Saved metadata.txt\n");
    }
    
    printf("\n✓ Successfully dumped all activations!\n");
    printf("  Load in Python with: np.load('%s/logits.npy')\n", output_dir);
}

// ============================================================================
// DUMP SINGLE LAYER (for debugging specific layers)
// ============================================================================

void dump_layer_npy(const char* output_dir, ActivationTensors* acts,
                    int layer, int B, int T, int C, int H) {
    printf("\nDumping layer %d to %s/ ...\n", layer, output_dir);
    
    #ifdef _WIN32
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir \"%s\" 2>nul", output_dir);
    system(cmd);
    #else
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", output_dir);
    system(cmd);
    #endif
    
    char filepath[512];
    size_t shape[5];
    
    int BT = B * T;
    int BTC = B * T * C;
    int BT3C = B * T * 3 * C;
    int BT4C = B * T * 4 * C;
    int BT_att = B * H * T * T;
    
    // Layer-specific pointers
    floatX* qkv_layer = acts->qkv + layer * BT3C;
    floatX* q_layer = acts->q + layer * BTC;
    floatX* k_layer = acts->k + layer * BTC;
    floatX* v_layer = acts->v + layer * BTC;
    floatX* att_scores_layer = acts->att_scores + layer * BT_att;
    floatX* att_probs_layer = acts->att_probs + layer * BT_att;
    floatX* att_out_layer = acts->att_out + layer * BTC;
    floatX* fc1_layer = acts->fc1 + layer * BT4C;
    floatX* fc2_layer = acts->fc2 + layer * BTC;
    
    // QKV: (B, T, 3*C)
    snprintf(filepath, sizeof(filepath), "%s/qkv.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = 3*C;
    dump_tensor_npy(filepath, qkv_layer, 3, shape, "<f4", sizeof(floatX));
    
    // Q: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/q.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, q_layer, 3, shape, "<f4", sizeof(floatX));
    
    // K: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/k.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, k_layer, 3, shape, "<f4", sizeof(floatX));
    
    // V: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/v.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, v_layer, 3, shape, "<f4", sizeof(floatX));
    
    // Attention scores: (B, H, T, T)
    snprintf(filepath, sizeof(filepath), "%s/att_scores.npy", output_dir);
    shape[0] = B; shape[1] = H; shape[2] = T; shape[3] = T;
    dump_tensor_npy(filepath, att_scores_layer, 4, shape, "<f4", sizeof(floatX));
    
    // Attention probs: (B, H, T, T)
    snprintf(filepath, sizeof(filepath), "%s/att_probs.npy", output_dir);
    shape[0] = B; shape[1] = H; shape[2] = T; shape[3] = T;
    dump_tensor_npy(filepath, att_probs_layer, 4, shape, "<f4", sizeof(floatX));
    
    // Attention output: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/att_out.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, att_out_layer, 3, shape, "<f4", sizeof(floatX));
    
    // MLP fc1: (B, T, 4*C)
    snprintf(filepath, sizeof(filepath), "%s/fc1.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = 4*C;
    dump_tensor_npy(filepath, fc1_layer, 3, shape, "<f4", sizeof(floatX));
    
    // MLP fc2: (B, T, C)
    snprintf(filepath, sizeof(filepath), "%s/fc2.npy", output_dir);
    shape[0] = B; shape[1] = T; shape[2] = C;
    dump_tensor_npy(filepath, fc2_layer, 3, shape, "<f4", sizeof(floatX));
    
    printf("✓ Successfully dumped layer %d!\n", layer);
}





#define CUDA_CHECK(x) do { cudaError_t err = x; if (err != cudaSuccess) { \
    printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

// Helper function to read binary data from file
void fread_check(void* ptr, size_t size, size_t count, FILE* file, const char* name) {
    size_t result = fread(ptr, size, count, file);
    if (result != count) {
        printf("Error reading %s from file. Expected %zu elements, got %zu\n", 
               name, count, result);
        exit(1);
    }
}

// Load GPT-2 weights from a binary file
// Expected file format (all in float32, contiguous):
// 1. wte: [V, C]
// 2. wpe: [maxT, C]
// For each layer l in [0, L):
//   3. ln1w[l]: [C]
//   4. ln1b[l]: [C]
//   5. qkvw[l]: [3*C, C]
//   6. qkvb[l]: [3*C]
//   7. attprojw[l]: [C, C]
//   8. attprojb[l]: [C]
//   9. ln2w[l]: [C]
//   10. ln2b[l]: [C]
//   11. fcw[l]: [4*C, C]
//   12. fcb[l]: [4*C]
//   13. fcprojw[l]: [C, 4*C]
//   14. fcprojb[l]: [C]
// Finally:
// 15. lnfw: [C]
// 16. lnfb: [C]

void load_gpt2_weights(const char* filename, ParameterTensors* params_device, 
                       int V, int C, int L, int maxT) {
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open weights file %s\n", filename);
        exit(1);
    }
    
    printf("Loading GPT-2 weights from %s...\n", filename);
    printf("Model config: V=%d, C=%d, L=%d, maxT=%d\n", V, C, L, maxT);
    
    // Get device pointers from the device struct
    ParameterTensors params_host;
    CUDA_CHECK(cudaMemcpy(&params_host, params_device, sizeof(ParameterTensors), 
                          cudaMemcpyDeviceToHost));
    
    // Temporary host buffers for reading
    floatX* temp_buffer = NULL;
    size_t max_size = 0;
    
    // Calculate maximum buffer size needed
    size_t sizes[] = {
        V * C,           // wte
        maxT * C,        // wpe
        C,               // ln1w/b
        3 * C * C,       // qkvw
        3 * C,           // qkvb
        C * C,           // attprojw
        C,               // attprojb
        4 * C * C,       // fcw
        4 * C,           // fcb
        C * 4 * C,       // fcprojw
        C                // fcprojb
    };
    
    for (int i = 0; i < sizeof(sizes) / sizeof(size_t); i++) {
        if (sizes[i] > max_size) max_size = sizes[i];
    }
    
    temp_buffer = (floatX*)malloc(max_size * sizeof(floatX));
    if (!temp_buffer) {
        printf("Error: Could not allocate temporary buffer\n");
        exit(1);
    }
    
    // ========================================================================
    // 1. Load token embeddings (wte)
    // ========================================================================
    printf("Loading wte [%d, %d]...\n", V, C);
    fread_check(temp_buffer, sizeof(floatX), V * C, file, "wte");
    CUDA_CHECK(cudaMemcpy(params_host.wte, temp_buffer, 
                          V * C * sizeof(floatX), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // 2. Load position embeddings (wpe)
    // ========================================================================
    printf("Loading wpe [%d, %d]...\n", maxT, C);
    fread_check(temp_buffer, sizeof(floatX), maxT * C, file, "wpe");
    CUDA_CHECK(cudaMemcpy(params_host.wpe, temp_buffer, 
                          maxT * C * sizeof(floatX), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // 3. Load transformer layers
    // ========================================================================
    for (int layer = 0; layer < L; layer++) {
        printf("Loading layer %d/%d...\n", layer + 1, L);
        
        // Layer Norm 1 weights
        fread_check(temp_buffer, sizeof(floatX), C, file, "ln1w");
        CUDA_CHECK(cudaMemcpy(params_host.ln1w + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // Layer Norm 1 bias
        fread_check(temp_buffer, sizeof(floatX), C, file, "ln1b");
        CUDA_CHECK(cudaMemcpy(params_host.ln1b + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // QKV projection weights
        fread_check(temp_buffer, sizeof(floatX), 3 * C * C, file, "qkvw");
        CUDA_CHECK(cudaMemcpy(params_host.qkvw + layer * 3 * C * C, temp_buffer, 
                              3 * C * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // QKV projection bias
        fread_check(temp_buffer, sizeof(floatX), 3 * C, file, "qkvb");
        CUDA_CHECK(cudaMemcpy(params_host.qkvb + layer * 3 * C, temp_buffer, 
                              3 * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // Attention output projection weights
        fread_check(temp_buffer, sizeof(floatX), C * C, file, "attprojw");
        CUDA_CHECK(cudaMemcpy(params_host.attprojw + layer * C * C, temp_buffer, 
                              C * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // Attention output projection bias
        fread_check(temp_buffer, sizeof(floatX), C, file, "attprojb");
        CUDA_CHECK(cudaMemcpy(params_host.attprojb + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // Layer Norm 2 weights
        fread_check(temp_buffer, sizeof(floatX), C, file, "ln2w");
        CUDA_CHECK(cudaMemcpy(params_host.ln2w + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // Layer Norm 2 bias
        fread_check(temp_buffer, sizeof(floatX), C, file, "ln2b");
        CUDA_CHECK(cudaMemcpy(params_host.ln2b + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // MLP first layer (fc) weights
        fread_check(temp_buffer, sizeof(floatX), 4 * C * C, file, "fcw");
        CUDA_CHECK(cudaMemcpy(params_host.fcw + layer * 4 * C * C, temp_buffer, 
                              4 * C * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // MLP first layer (fc) bias
        fread_check(temp_buffer, sizeof(floatX), 4 * C, file, "fcb");
        CUDA_CHECK(cudaMemcpy(params_host.fcb + layer * 4 * C, temp_buffer, 
                              4 * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // MLP second layer (fcproj) weights
        fread_check(temp_buffer, sizeof(floatX), C * 4 * C, file, "fcprojw");
        CUDA_CHECK(cudaMemcpy(params_host.fcprojw + layer * C * 4 * C, temp_buffer, 
                              C * 4 * C * sizeof(floatX), cudaMemcpyHostToDevice));
        
        // MLP second layer (fcproj) bias
        fread_check(temp_buffer, sizeof(floatX), C, file, "fcprojb");
        CUDA_CHECK(cudaMemcpy(params_host.fcprojb + layer * C, temp_buffer, 
                              C * sizeof(floatX), cudaMemcpyHostToDevice));
    }
    
    // ========================================================================
    // 4. Load final layer norm
    // ========================================================================
    printf("Loading final layer norm...\n");
    
    // Final LayerNorm weights
    fread_check(temp_buffer, sizeof(floatX), C, file, "lnfw");
    CUDA_CHECK(cudaMemcpy(params_host.lnfw, temp_buffer, 
                          C * sizeof(floatX), cudaMemcpyHostToDevice));
    
    // Final LayerNorm bias
    fread_check(temp_buffer, sizeof(floatX), C, file, "lnfb");
    CUDA_CHECK(cudaMemcpy(params_host.lnfb, temp_buffer, 
                          C * sizeof(floatX), cudaMemcpyHostToDevice));
    
    // Cleanup
    free(temp_buffer);
    fclose(file);
    
    printf("✓ Successfully loaded all weights!\n");
}

// Alternative: Load from PyTorch state dict (requires parsing .pt file)
// This would need a more complex parser, but here's a skeleton:

void load_gpt2_weights_from_pytorch(const char* checkpoint_path, 
                                     ParameterTensors* params_device,
                                     int V, int C, int L, int maxT) {
    // This would require:
    // 1. Reading PyTorch's pickle format or using torch::load
    // 2. Extracting tensors by name (e.g., "transformer.wte.weight")
    // 3. Converting to the expected layout
    
    printf("Error: PyTorch loader not implemented. Use binary format or convert weights first.\n");
    printf("To convert PyTorch weights to binary format, use the companion Python script.\n");
    exit(1);
}

// Helper: Calculate total model size
void print_model_size(int V, int C, int L, int maxT) {
    size_t total_params = 0;
    
    total_params += V * C;              // wte
    total_params += maxT * C;           // wpe
    total_params += L * C * 2;          // ln1w, ln1b
    total_params += L * 3 * C * C;      // qkvw
    total_params += L * 3 * C;          // qkvb
    total_params += L * C * C;          // attprojw
    total_params += L * C;              // attprojb
    total_params += L * C * 2;          // ln2w, ln2b
    total_params += L * 4 * C * C;      // fcw
    total_params += L * 4 * C;          // fcb
    total_params += L * C * 4 * C;      // fcprojw
    total_params += L * C;              // fcprojb
    total_params += C * 2;              // lnfw, lnfb
    
    double size_mb = (total_params * sizeof(floatX)) / (1024.0 * 1024.0);
    double size_gb = size_mb / 1024.0;
    
    printf("\nModel Statistics:\n");
    printf("  Total parameters: %zu (%.2f M)\n", total_params, total_params / 1e6);
    printf("  Memory footprint: %.2f MB (%.3f GB)\n", size_mb, size_gb);
}
