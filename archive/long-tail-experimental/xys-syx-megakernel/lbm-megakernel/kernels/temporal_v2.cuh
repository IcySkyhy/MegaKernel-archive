#pragma once
#include "temporal_common.cuh"
namespace temporal {
// src/dst must be distinct. Immutable flags and population slots with no physical
// producer must have matching initial values and remain invariant. The evolving
// population state need not match: successive launches hold t and t-2.
// No first-step global state is materialized. Each block redundantly collides
// its one-site halo; there is exactly one unconditional block barrier.
template<int TX, int TY, int TZ>
__global__ void v2(const float *__restrict__ src, float *__restrict__ dst) {
    constexpr int HX = TX + 2, HY = TY + 2, HZ = TZ + 2;
    constexpr int HALO_SITES = HX * HY * HZ, CORE_SITES = TX * TY * TZ;
    extern __shared__ float intermediate_R[];
    const int ox = int(blockIdx.x) * TX, oy = int(blockIdx.y) * TY;
    const int oz = int(blockIdx.z) * TZ;
    for (int i = int(threadIdx.x); i < HALO_SITES; i += int(blockDim.x)) {
        const int lx = i % HX - 1, ly = (i / HX) % HY - 1;
        const int lz = i / (HX * HY) - 1;
        const int s = site(ox + lx, oy + ly, oz + lz);
        const bool valid = has_producer(s);
        D128Populations p;
        if (valid) {
            p = load(src, s);
            p.collide(flags_at(src, s));
        }
        // Fixed q maps producer -> destination injectively. Each physical (r,q)
        // in R has exactly one logical producer in this halo box. A missing
        // physical producer writes the invariant fallback through the same owner;
        // there is no separate initialization pass racing with these writes.
#define STREAM_INTO_R(q, dx, dy, dz) { \
        const int rx = lx + (dx), ry = ly + (dy), rz = lz + (dz); \
        if (rx >= 0 && rx < TX && ry >= 0 && ry < TY && rz >= 0 && rz < TZ && \
            ox + rx < SIZE_X && oy + ry < SIZE_Y && oz + rz < SIZE_Z) { \
            const int dest = rx + TX * (ry + TY * rz); \
            if (valid) \
                intermediate_R[q * CORE_SITES + dest] = p.temp##q; \
            else \
                intermediate_R[q * CORE_SITES + dest] = \
                    src[q * TOTAL_PADDED_CELLS + site(ox + rx, oy + ry, oz + rz)]; \
        } \
    }
        LBM_DIRECTIONS(STREAM_INTO_R)
#undef STREAM_INTO_R
    }
    __syncthreads();

    for (int i = int(threadIdx.x); i < TX * TY * TZ; i += int(blockDim.x)) {
        const int lx = i % TX, ly = (i / TX) % TY, lz = i / (TX * TY);
        const int x = ox + lx, y = oy + ly, z = oz + lz;
        if (x >= SIZE_X || y >= SIZE_Y || z >= SIZE_Z) continue;
        const int s = site(x, y, z);
        D128Populations p;
#define LOAD_R(q, dx, dy, dz) p.temp##q = intermediate_R[q * CORE_SITES + i];
        LBM_DIRECTIONS(LOAD_R)
#undef LOAD_R
        p.collide(flags_at(src, s));
        // Preserve every baseline scatter, including writes to padding/margins.
        // For fixed q this translation is injective: no cross-block write races.
#define SCATTER_OUT(q, dx, dy, dz) dst[q * TOTAL_PADDED_CELLS + s + site(dx, dy, dz)] = p.temp##q;
        LBM_DIRECTIONS(SCATTER_OUT)
#undef SCATTER_OUT
    }
}
} // namespace temporal
