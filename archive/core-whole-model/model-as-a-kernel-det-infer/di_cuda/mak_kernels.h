// C++ interface to the megakernel launches. Torch-free so the .cu translation
// unit never sees libtorch headers (nvcc host-pass friendliness on all
// platforms); the torch binding layer owns tensors, checks, and timing.
#pragma once

#include <cuda_runtime.h>

int mak_grid_blocks_impl(int smem_bytes);

// Grid and max batch width for the global-scratch batched-decode kernel.
int mak_grid_blocks_big_impl(int smem_bytes);
int mak_batch_maxb_impl();

// cp.async ring stages for a launch with this staging footprint (power of
// two in [1, 16], chosen from the device's shared-memory budget).
int mak_ring_stages_impl(int stage_bytes);

// ts (nullable): block 0 stamps clock64() at entry (slot 0) and after each
// phase's barrier crossing of the first step (slot p+1); size n_phases+1.
cudaError_t mak_launch_mega(const long long* prog, int n_phases, int pos0,
                            int total, int slot0, int prompt_len,
                            int chunk_m, int ring_stages, int* bar, int grid,
                            int smem_bytes, cudaStream_t stream,
                            long long* ts);

cudaError_t mak_launch_phase(const long long* prog, int phase, int pos,
                             int step_slot, int ring_stages, int grid,
                             int smem_bytes, cudaStream_t stream);

// Batched decode: one step advances every sequence. pos_b[B] holds each
// sequence's base position (its row appends KV at pos_b[b] + step);
// kv_bstride is the per-sequence element stride of the K and V caches.
cudaError_t mak_launch_mega_batch(const long long* prog, int n_phases,
                                  int steps, int slot0, int ring_stages,
                                  int* bar, int grid, int smem_bytes,
                                  const int* pos_b, int B,
                                  long long kv_bstride,
                                  cudaStream_t stream);

// Batch decode for B beyond the fused staging cap: transformed input in a
// global scratch, so B is limited by mak_batch_maxb(), not the width K.
cudaError_t mak_launch_mega_batch_big(const long long* prog, int n_phases,
                                      int steps, int slot0, int ring_stages,
                                      int* bar, int grid, int smem_bytes,
                                      const int* pos_b, int B,
                                      long long kv_bstride,
                                      cudaStream_t stream);
