#pragma once
#include "spatial.cuh"
// Logical tile 16x8x4; block=(128,1,1) or (256,1,1).
// The same kernel is also used for the matched 256-worker control.
__global__ void s2_d128(float *src, float *dst) {
    tile_step<16,8,4>(src, dst);
}
