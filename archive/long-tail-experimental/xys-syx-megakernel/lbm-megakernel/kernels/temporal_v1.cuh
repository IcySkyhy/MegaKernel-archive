#pragma once
#include "temporal_common.cuh"
namespace temporal {
// src/dst must be distinct. Immutable flags and population slots with no physical
// producer must have matching initial values and remain invariant. The evolving
// population state need not match: successive launches hold t and t-2.
// No first-step global state is materialized. Each block redundantly collides
// its one-site halo; there is exactly one unconditional block barrier.
template<int TX, int TY, int TZ>
__global__ void v1(const float *__restrict__ src, float *__restrict__ dst) {
    constexpr int HX = TX + 2, HY = TY + 2, HZ = TZ + 2;
    constexpr int HALO_SITES = HX * HY * HZ, CORE_SITES = TX * TY * TZ;
    extern __shared__ float producer_state[];
    const int ox = int(blockIdx.x) * TX, oy = int(blockIdx.y) * TY;
    const int oz = int(blockIdx.z) * TZ;
    for (int i = int(threadIdx.x); i < HALO_SITES; i += int(blockDim.x)) {
        const int x = ox + i % HX - 1;
        const int y = oy + (i / HX) % HY - 1;
        const int z = oz + i / (HX * HY) - 1;
        const int s = site(x, y, z);
        if (has_producer(s)) {
            D128Populations p = load(src, s);
            p.collide(flags_at(src, s));
#define STORE_SHARED(q, dx, dy, dz) producer_state[q * HALO_SITES + i] = p.temp##q;
            LBM_DIRECTIONS(STORE_SHARED)
#undef STORE_SHARED
        }
    }
    __syncthreads();

    for (int i = int(threadIdx.x); i < TX * TY * TZ; i += int(blockDim.x)) {
        const int lx = i % TX, ly = (i / TX) % TY, lz = i / (TX * TY);
        const int x = ox + lx, y = oy + ly, z = oz + lz;
        if (x >= SIZE_X || y >= SIZE_Y || z >= SIZE_Z) continue;
        const int s = site(x, y, z);
        const int h = (lx + 1) + HX * ((ly + 1) + HY * (lz + 1));
        D128Populations p;
        // Entries with no producer are never overwritten by the baseline.
        // Both global buffers start identical, so their initial value survives
        // all odd/even timesteps and can be read directly from src at this site.
#define GATHER(q, dx, dy, dz) \
        p.temp##q = has_producer(s - site(dx, dy, dz)) \
            ? producer_state[q * HALO_SITES + h - ((dx) + HX * ((dy) + HY * (dz)))] \
            : src[q * TOTAL_PADDED_CELLS + s];
        LBM_DIRECTIONS(GATHER)
#undef GATHER
        p.collide(flags_at(src, s));
        // Preserve every baseline scatter, including writes to padding/margins.
        // For fixed q this translation is injective: no cross-block write races.
#define SCATTER_OUT(q, dx, dy, dz) dst[q * TOTAL_PADDED_CELLS + s + site(dx, dy, dz)] = p.temp##q;
        LBM_DIRECTIONS(SCATTER_OUT)
#undef SCATTER_OUT
    }
}
} // namespace temporal
