#pragma once
#include "cell.cuh"
// Logical tile dimensions are independent of CUDA block dimensions.
// One launch is one timestep.
template<int TX, int TY, int TZ>
__device__ __forceinline__ void tile_step(float *src, float *dst) {
    for (int i = int(threadIdx.x); i < TX * TY * TZ; i += int(blockDim.x)) {
        const int x = int(blockIdx.x) * TX + i % TX;
        const int y = int(blockIdx.y) * TY + (i / TX) % TY;
        const int z = int(blockIdx.z) * TZ + i / (TX * TY);
        if (x < SIZE_X && y < SIZE_Y && z < SIZE_Z)
            performStreamCollide_cell(src, dst, x, y, z);
    }
}
