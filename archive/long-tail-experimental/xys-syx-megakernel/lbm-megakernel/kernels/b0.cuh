#pragma once
#include "cell.cuh"
// B0: one physical X row per block, block=(120,1,1), grid=(120,150,1).
__global__ void b0(float *src, float *dst) {
    performStreamCollide_cell(src, dst, threadIdx.x, blockIdx.x, blockIdx.y);
}
