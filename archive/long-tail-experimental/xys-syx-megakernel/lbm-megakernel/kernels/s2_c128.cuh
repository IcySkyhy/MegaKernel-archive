#pragma once
#include "spatial.cuh"
// Logical tile 8x8x8; block=(128,1,1) or (256,1,1).
// The same kernel is also used for the matched 256-worker control.
__global__ void s2_c128(float *src, float *dst) {
    tile_step<8,8,8>(src, dst);
}
