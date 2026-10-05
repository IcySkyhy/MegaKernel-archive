#pragma once
#include <cooperative_groups.h>
#include "cell.cuh"
// The whole time loop in one kernel, with B0's launch kept: grid=(120,150,1), block=(120,1,1),
// thread (x,y,z) = (threadIdx.x, blockIdx.x, blockIdx.y). Each step reads neighbours written by
// other blocks in the previous step, so between steps every block of the grid must have finished.
// With separate launches the kernel boundary provides that; inside one kernel it must be written.
namespace onekernel {
namespace cg = cooperative_groups;

// bigkern1: kern(src,dst); kern(dst,src) in one kernel, no barrier between them.
__global__ void bigkern1(float *src, float *dst) {
    performStreamCollide_cell(src, dst, threadIdx.x, blockIdx.x, blockIdx.y);
    performStreamCollide_cell(dst, src, threadIdx.x, blockIdx.x, blockIdx.y);
}

// bigkern2: the time loop around bigkern1, launched once, no barrier.
__global__ void bigkern2(float *src, float *dst, int steps) {
    for (int i = 0; i < steps / 2; ++i) {
        performStreamCollide_cell(src, dst, threadIdx.x, blockIdx.x, blockIdx.y);
        performStreamCollide_cell(dst, src, threadIdx.x, blockIdx.x, blockIdx.y);
    }
}

// bigkern2 with grid.sync() after each step; needs a cooperative launch.
__global__ void bigkern2_grid_sync(float *src, float *dst, int steps) {
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < steps / 2; ++i) {
        performStreamCollide_cell(src, dst, threadIdx.x, blockIdx.x, blockIdx.y);
        grid.sync();
        performStreamCollide_cell(dst, src, threadIdx.x, blockIdx.x, blockIdx.y);
        grid.sync();
    }
}

// A global barrier for an ordinary launch: a block counter plus a generation number. A block that
// waits longer than timeout_ns sets abort (recording how many blocks had arrived), and every block
// then returns instead of hanging the GPU.
__device__ unsigned int count, generation, arrived_at_abort;
__device__ int abort_flag;

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ bool grid_barrier(unsigned int blocks, unsigned long long timeout_ns) {
    __shared__ int ok;
    __syncthreads();
    if (threadIdx.x == 0) {
        ok = !*(volatile int *)&abort_flag;
        if (ok) {
            const unsigned int gen = *(volatile unsigned int *)&generation;
            __threadfence();
            if (atomicAdd(&count, 1) + 1 == blocks) {
                atomicExch(&count, 0);
                __threadfence();
                atomicAdd(&generation, 1);
            } else {
                const unsigned long long t0 = global_ns();
                while (*(volatile unsigned int *)&generation == gen) {
                    if (*(volatile int *)&abort_flag) { ok = 0; break; }
                    if (global_ns() - t0 > timeout_ns) {
                        if (atomicCAS(&abort_flag, 0, 1) == 0)
                            arrived_at_abort = *(volatile unsigned int *)&count;
                        ok = 0;
                        break;
                    }
                }
            }
            __threadfence();
        }
    }
    __syncthreads();
    return ok;
}

// bigkern2 with grid_barrier() after each step; ordinary launch.
__global__ void bigkern2_spin(float *src, float *dst, int steps, unsigned long long timeout_ns) {
    const unsigned int blocks = gridDim.x * gridDim.y;
    for (int i = 0; i < steps / 2; ++i) {
        performStreamCollide_cell(src, dst, threadIdx.x, blockIdx.x, blockIdx.y);
        if (!grid_barrier(blocks, timeout_ns)) return;
        performStreamCollide_cell(dst, src, threadIdx.x, blockIdx.x, blockIdx.y);
        if (!grid_barrier(blocks, timeout_ns)) return;
    }
}

// grid_barrier() alone, for a grid small enough to be resident at once.
__device__ unsigned int finished;
__global__ void barrier_only(int barriers, unsigned long long timeout_ns) {
    for (int i = 0; i < barriers; ++i)
        if (!grid_barrier(gridDim.x * gridDim.y, timeout_ns)) return;
    if (threadIdx.x == 0) atomicAdd(&finished, 1);
}

// The smallest change that is correct: launch only the blocks that fit, and let each one loop over
// B0's (y,z) rows. Block size and x = threadIdx.x stay; the grid and the block-to-row binding change.
__global__ void bigkern2_resident(float *src, float *dst, int steps) {
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < steps / 2; ++i) {
        for (int r = blockIdx.x; r < SIZE_Y * SIZE_Z; r += gridDim.x)
            performStreamCollide_cell(src, dst, threadIdx.x, r % SIZE_Y, r / SIZE_Y);
        grid.sync();
        for (int r = blockIdx.x; r < SIZE_Y * SIZE_Z; r += gridDim.x)
            performStreamCollide_cell(dst, src, threadIdx.x, r % SIZE_Y, r / SIZE_Y);
        grid.sync();
    }
}
}  // namespace onekernel
