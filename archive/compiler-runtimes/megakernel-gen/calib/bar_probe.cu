// ============================================================================
// bar_probe -- is cooperative groups' grid barrier as cheap as it can be, and
// is anything cheaper still a BARRIER?
//
// A megakernel pays one grid-wide rendezvous per stage per layer: 170 of them
// on a 28-layer dense model, 338 on a 48-layer MoE.  At ~1 us each that is
// 8-12% of a token, so 300 ns off a barrier is real throughput.
//
// Why this file checks before it times.  A counting barrier whose spin read the
// counter with `ld.global.cg` -- which bypasses L1 and is measurably cheaper
// than a volatile load -- counted arrivals perfectly and ordered nothing: 90%
// of the cross-block reads that followed it came back stale, and the generated
// kernel reported it as NaN logits.  That barrier would have won this table.
// A rendezvous that does not publish the writes before it is not a fast
// barrier, it is a fast bug, so every variant here must first pass:
//
//   thread 0 writes slot[blockIdx.x] = round; rendezvous; EVERY thread reads
//   ALL of slot[0..gridDim.x) with PLAIN loads and counts the entries that are
//   not `round`; rendezvous again.  A correct barrier gives zero stale reads.
//
// The plain loads are the whole point: they are what generated stage code does,
// and they hit a per-SM L1 that arrival counting alone never invalidates.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <mk/common.cuh>
namespace cg = cooperative_groups;

// The counting barrier this file exists to judge.  It lives here and not in
// `runtime/include/mk` because the answer was no: it is 0.2-0.5 us per barrier
// slower than `cg::grid.sync()` at every geometry, because every block's
// arrival serialises on one contended cache line where cooperative groups uses
// a tree.  Kept so the next person can re-run the comparison rather than
// re-derive it -- and so the correctness check below has something to catch.
namespace probe {
__device__ __forceinline__ void grid_barrier(unsigned int* c, unsigned int target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(c, 1u);
        volatile unsigned int* p = c;
        while ((int)(*p - target) < 0) {}
        __threadfence();
    }
    __syncthreads();
}
}  // namespace probe

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("cuda %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

enum { V_CG = 0, V_MK, V_SLEEP, V_WARP, NVAR };
static const char* NAME[NVAR] = { "cg", "mk", "mk+slp", "mk+warp" };

// (3) probe::grid_barrier with a backoff: the spinners stop hammering the counter's
// L2 line, at the price of noticing the last arrival up to `ns` late.
__device__ __forceinline__ void bar_sleep(unsigned int* c, unsigned int target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(c, 1u);
        volatile unsigned int* p = c;
        while ((int)(*p - target) < 0) __nanosleep(32);
        __threadfence();
    }
    __syncthreads();
}

// (4) the whole of warp 0 spins instead of thread 0 alone.  All 32 lanes read
// the same word, so the SM issues one request per poll either way -- what
// changes is that the exit is a warp vote, which lets any lane that saw the
// release wake the other 31 instead of each of them re-reading.
__device__ __forceinline__ void bar_warp(unsigned int* c, unsigned int target) {
    __syncthreads();
    if (threadIdx.x < 32) {
        // Lane 0 announces; the others may start polling before it does, which
        // costs them a few wasted reads and nothing else.
        if (threadIdx.x == 0) { __threadfence(); atomicAdd(c, 1u); }
        volatile unsigned int* p = c;
        while (!__any_sync(0xffffffffu, (int)(*p - target) >= 0)) {}
        __threadfence();
    }
    __syncthreads();
}

template<int V> __device__ __forceinline__
void rendezvous(cg::grid_group& g, unsigned int* c, unsigned int target) {
    if constexpr (V == V_CG)         { (void)c; (void)target; g.sync(); }
    else if constexpr (V == V_MK)    { (void)g; probe::grid_barrier(c, target); }
    else if constexpr (V == V_SLEEP) { (void)g; bar_sleep(c, target); }
    else                             { (void)g; bar_warp(c, target); }
}

// ---------------------------------------------------------------- correctness
// `target` counts ARRIVALS, and there are two rendezvous per round, so round r
// ends at 2r*gridDim.x arrivals.
template<int V>
__global__ void k_check(unsigned int* c, unsigned int* slot, int rounds,
                        unsigned long long* stale) {
    cg::grid_group g = cg::this_grid();
    const unsigned int nblk = gridDim.x;
    unsigned long long bad = 0;
    for (int r = 1; r <= rounds; ++r) {
        if (threadIdx.x == 0) slot[blockIdx.x] = (unsigned int)r;
        rendezvous<V>(g, c, (unsigned int)(2 * r - 1) * nblk);
        for (unsigned int j = 0; j < nblk; ++j) bad += (slot[j] != (unsigned int)r);
        // The second rendezvous is what stops a fast block from overwriting its
        // slot for round r+1 while a slow one is still reading round r.
        rendezvous<V>(g, c, (unsigned int)(2 * r) * nblk);
    }
    if (bad) atomicAdd(stale, bad);
}

// ---------------------------------------------------------------- cost
// Nothing but barriers, so the number is the barrier.  The clock starts at
// iteration WARM because the first few pay for cold instruction cache and for
// the blocks that have not been scheduled yet.
static const int WARM = 9;

template<int V>
__global__ void k_time(unsigned int* c, int iters, unsigned long long* out) {
    cg::grid_group g = cg::this_grid();
    unsigned long long t0 = 0;
    for (int i = 1; i <= iters; ++i) {
        if (i == WARM && blockIdx.x == 0 && threadIdx.x == 0) t0 = mk::gtime();
        rendezvous<V>(g, c, (unsigned int)i * gridDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = mk::gtime() - t0;
}

static void* CHECK_K[NVAR] = { (void*)k_check<V_CG>, (void*)k_check<V_MK>,
                               (void*)k_check<V_SLEEP>, (void*)k_check<V_WARP> };
static void* TIME_K[NVAR]  = { (void*)k_time<V_CG>,  (void*)k_time<V_MK>,
                               (void*)k_time<V_SLEEP>,  (void*)k_time<V_WARP> };

// A hand-rolled barrier DEADLOCKS if the grid is not co-resident, so the probe
// asks the occupancy API rather than finding out by burning the job's walltime.
static int occ_min(void* const* k, int n, int nt) {
    int lo = 1 << 30;
    for (int i = 0; i < n; ++i) {
        int b = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, k[i], nt, 0));
        if (b < lo) lo = b;
    }
    return lo;
}

int main(int argc, char** argv) {
    const int iters  = argc > 1 ? atoi(argv[1]) : 2000;
    const int rounds = argc > 2 ? atoi(argv[2]) : 8;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    const int nsm = pr.multiProcessorCount;
    printf("bar_probe: %s, %d SMs, %d timed iterations, %d checked rounds\n",
           pr.name, nsm, iters, rounds);
    printf("  stale = cross-block reads that missed the barrier's ordering; "
           "0 is the only passing value\n");

    unsigned int *cnt, *slot;
    unsigned long long *stale, *out;
    CK(cudaMalloc(&cnt, 4));
    CK(cudaMalloc(&slot, 4096u * 4));      // >= 3x the SM count of any device
    CK(cudaMalloc(&stale, 8));
    CK(cudaMalloc(&out, 8));

    printf("  %-5s %-6s", "NT", "grid");
    for (int v = 0; v < NVAR; ++v) {
        char h[32];
        snprintf(h, sizeof h, "%s.stale", NAME[v]); printf(" %12s", h);
        snprintf(h, sizeof h, "%s.us", NAME[v]);    printf(" %9s", h);
    }
    printf("\n");

    bool any_bad = false;
    for (int nt : {256, 512, 1024}) {
        for (int mult : {1, 2, 3}) {
            const int grid = nsm * mult;
            if (occ_min(CHECK_K, NVAR, nt) * nsm < grid) continue;
            if (occ_min(TIME_K,  NVAR, nt) * nsm < grid) continue;

            unsigned long long bad[NVAR];
            double us[NVAR];
            for (int v = 0; v < NVAR; ++v) {
                int it = rounds; unsigned int* sl = slot;
                CK(cudaMemset(cnt, 0, 4));
                CK(cudaMemset(slot, 0, 4096u * 4));
                CK(cudaMemset(stale, 0, 8));
                void* ac[] = {&cnt, &sl, &it, &stale};
                CK(cudaLaunchCooperativeKernel(CHECK_K[v], grid, nt, ac, 0, 0));
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(&bad[v], stale, 8, cudaMemcpyDeviceToHost));

                // Fresh counter: the timing kernel's targets start from zero.
                int ti = iters;
                CK(cudaMemset(cnt, 0, 4));
                void* at[] = {&cnt, &ti, &out};
                CK(cudaLaunchCooperativeKernel(TIME_K[v], grid, nt, at, 0, 0));
                CK(cudaDeviceSynchronize());
                unsigned long long ns;
                CK(cudaMemcpy(&ns, out, 8, cudaMemcpyDeviceToHost));
                us[v] = (double)ns / 1000.0 / (iters - WARM + 1);
                any_bad |= (bad[v] != 0);
            }
            printf("  %-5d %-6d", nt, grid);
            for (int v = 0; v < NVAR; ++v) {
                char t[16];
                // A timing next to a nonzero stale count is not a barrier cost,
                // it is the cost of skipping the work, hence the marker.
                snprintf(t, sizeof t, "%.3f%s", us[v], bad[v] ? "!" : "");
                printf(" %12llu %9s", bad[v], t);
            }
            printf("\n");
            fflush(stdout);
        }
    }
    if (any_bad)
        printf("  ! = the variant failed the ordering check; its timing means nothing\n");
    CK(cudaFree(cnt)); CK(cudaFree(slot)); CK(cudaFree(stale)); CK(cudaFree(out));
    return 0;
}
