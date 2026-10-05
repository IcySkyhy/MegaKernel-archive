#pragma once
#include "collision.cuh"
#ifndef SCATTER
#error "Temporal kernels require the original SCATTER layout"
#endif
static_assert(PADDED_X > SIZE_X && PADDED_Y == SIZE_Y,
              "Revisit flat-address halo semantics if the layout changes");
namespace temporal {
#define LBM_DIRECTIONS(F) \
    F(C, 0, 0, 0) F(N, 0, 1, 0) F(S, 0,-1, 0) \
    F(E, 1, 0, 0) F(W,-1, 0, 0) F(T, 0, 0, 1) F(B, 0, 0,-1) \
    F(NE, 1, 1, 0) F(NW,-1, 1, 0) F(SE, 1,-1, 0) F(SW,-1,-1, 0) \
    F(NT, 0, 1, 1) F(NB, 0, 1,-1) F(ST, 0,-1, 1) F(SB, 0,-1,-1) \
    F(ET, 1, 0, 1) F(EB, 1, 0,-1) F(WT,-1, 0, 1) F(WB,-1, 0,-1)

__device__ __forceinline__ int site(int x, int y, int z) {
    return x + PADDED_X * (y + PADDED_Y * z);
}

// A producer exists iff this *flat address* is a physical baseline site.
// With PADDING_Y=0, y=-1 and y=SIZE_Y can alias physical sites in adjacent
// Z planes. Geometric XYZ clipping here would silently change the baseline.
__device__ __forceinline__ bool has_producer(int s) {
    return s >= 0 && s < PADDED_X * PADDED_Y * SIZE_Z &&
           s % PADDED_X < SIZE_X;
}

__device__ __forceinline__ unsigned char flags_at(const float *src, int s) {
    return *reinterpret_cast<const unsigned char*>(src + FLAGS * TOTAL_PADDED_CELLS + s);
}

__device__ __forceinline__ D128Populations load(const float *src, int s) {
    D128Populations p;
#define LOAD(q, dx, dy, dz) p.temp##q = src[q * TOTAL_PADDED_CELLS + s];
    LBM_DIRECTIONS(LOAD)
#undef LOAD
    return p;
}

} // namespace temporal
