#pragma once
#include "spatial.cuh"
// Logical tile 30x4x4; block=(128,1,1) or (256,1,1).
// The same kernel is also used for the matched 256-worker control.
__global__ void t30_s1(float *src, float *dst) {
    tile_step<30,4,4>(src, dst);
}
