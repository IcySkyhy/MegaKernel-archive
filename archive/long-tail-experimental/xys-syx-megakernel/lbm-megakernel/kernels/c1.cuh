#pragma once
#include "cell.cuh"
// C1: one site per worker; launch block=(32,4,1). No temporal fusion.
__global__ void c1(float *src, float *dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= SIZE_X || y >= SIZE_Y || z >= SIZE_Z) return;
    performStreamCollide_cell(src, dst, x, y, z);
}
