// ============================================================================
// mkc calibrate -- the machine model.
//
// Run once per GPU type.  It measures the primitive rates that every later
// compile is planned against, so that compiling a model is a search-free,
// deterministic function of (model, machine file).  Nothing else in the
// compiler ever touches a GPU.
//
// What it measures, and why each one exists:
//
//  [1] pure streaming read vs COLD footprint.  The advertised HBM number is
//      not reachable, and this GPU has a 52 MB L2, so a naive loop over a
//      32 MB footprint measures the cache and reports a rate the kernel will
//      never see.  Every repetition here reads memory it has not just read.
//
//  [2] every gemv core at every (block size, row blocking), cold, at a large
//      footprint.  This is the core's ceiling.
//
//  [2b] the same cores in the shape a megakernel actually runs them: inside a
//      persistent cooperative grid, one stage of B bytes and N items, ending
//      at a grid barrier.  A stage is often ONE partial wave, where the limit
//      is memory-level parallelism (how many items are in flight), not
//      bandwidth.  Conflating the two is why a naive "bandwidth vs size" curve
//      mispredicts by 4x.  The planner fits BW = min(peak, items * c) to this.
//
//  [3] cooperative grid barrier cost vs grid size.  barriers x stages x layers
//      is a fixed tax that decides how coarse the stages have to be.
//
//  [4] kernel launch overhead -- the thing a megakernel exists to delete.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "mk/common.cuh"
#include "mk/gemv.cuh"

namespace cg = cooperative_groups;
using namespace mk;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1);} } while(0)


static int    g_sms;
static char*  g_buf;
static size_t g_bytes;
static float* g_out;

// ---------------------------------------------------------------- kernels
__global__ __launch_bounds__(512) void k_stream(const float4* __restrict__ p,
                                                size_t n4, float* out) {
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = p[i];
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    if (acc.x == 1e30f) out[0] = acc.x + acc.y + acc.z + acc.w;
}

template<class T, int KDIM, int NT, int R>
__global__ __launch_bounds__(NT) void k_gemv_dense(const T* __restrict__ W,
                                                   long long rows, int stages,
                                                   int coop, float* out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < pad33_size(KDIM); i += NT) xs[i] = 1.0f / (1 + (i & 31));
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = NT >> 5;
    const long long nitem = rows / R;
    float sink = 0.f;
    for (int s = 0; s < stages; ++s) {
        const T* Ws = W + (size_t)s * rows * KDIM;
        const long long lo = (long long)blockIdx.x * ((nitem + gridDim.x - 1) / gridDim.x);
        const long long hi = min(lo + (nitem + gridDim.x - 1) / gridDim.x, nitem);
        for (long long it = lo + wid; it < hi; it += nw) {
            float acc[R];
            gemv_dense<T, KDIM, R>(Ws + (size_t)(it * R) * KDIM, KDIM, xs, lane, acc);
            #pragma unroll
            for (int r = 0; r < R; ++r) sink += acc[r];
        }
        if (coop) cg::this_grid().sync();
    }
    if (sink == 1e30f) out[0] = sink;
}

template<int KDIM, int NT, int R>
__global__ __launch_bounds__(NT) void k_gemv_mxfp4(const uint4* __restrict__ blk,
                                                   const uint8_t* __restrict__ scl,
                                                   long long rows, int stages,
                                                   int coop, float* out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < pad33_size(KDIM); i += NT) xs[i] = 1.0f / (1 + (i & 31));
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = NT >> 5;
    const long long nitem = rows / R;
    float sink = 0.f;
    for (int s = 0; s < stages; ++s) {
        constexpr int GDIM = KDIM / 32;
        const uint4*   bs = blk + (size_t)s * rows * GDIM;
        const uint8_t* ss = scl + (size_t)s * rows * GDIM;
        const long long lo = (long long)blockIdx.x * ((nitem + gridDim.x - 1) / gridDim.x);
        const long long hi = min(lo + (nitem + gridDim.x - 1) / gridDim.x, nitem);
        for (long long it = lo + wid; it < hi; it += nw) {
            float acc[R];
            gemv_mxfp4<KDIM / 32, R>(bs + (size_t)(it * R) * (KDIM / 32),
                                     ss + (size_t)(it * R) * (KDIM / 32), xs, lane, acc);
            #pragma unroll
            for (int r = 0; r < R; ++r) sink += acc[r];
        }
        if (coop) cg::this_grid().sync();
    }
    if (sink == 1e30f) out[0] = sink;
}

template<int KDIM, int NT, int R>
__global__ __launch_bounds__(NT) void k_gemv_fp8(const __nv_fp8_e4m3* __restrict__ W,
                                                 const float* __restrict__ S,
                                                 long long rows, int stages,
                                                 int coop, float* out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < pad33_size(KDIM); i += NT) xs[i] = 1.0f / (1 + (i & 31));
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = NT >> 5;
    const long long nitem = rows / R;
    constexpr int SK = KDIM / 128;
    float sink = 0.f;
    for (int s = 0; s < stages; ++s) {
        const __nv_fp8_e4m3* Ws = W + (size_t)s * rows * KDIM;
        const long long lo = (long long)blockIdx.x * ((nitem + gridDim.x - 1) / gridDim.x);
        const long long hi = min(lo + (nitem + gridDim.x - 1) / gridDim.x, nitem);
        for (long long it = lo + wid; it < hi; it += nw) {
            float acc[R];
            gemv_fp8<KDIM, R, 1, 128>(Ws + (size_t)(it * R) * KDIM, S, SK, 0, xs, lane, acc);
            #pragma unroll
            for (int r = 0; r < R; ++r) sink += acc[r];
        }
        if (coop) cg::this_grid().sync();
    }
    if (sink == 1e30f) out[0] = sink;
}

template<int KDIM, int NT, int R>
__global__ __launch_bounds__(NT) void k_gemv_int4(const uint8_t* __restrict__ Q,
                                                  const __half* __restrict__ S,
                                                  const __half* __restrict__ Z,
                                                  long long rows, int stages,
                                                  int coop, float* out) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < pad33_size(KDIM); i += NT) xs[i] = 1.0f / (1 + (i & 31));
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = NT >> 5;
    const long long nitem = rows / R;
    float sink = 0.f;
    for (int s = 0; s < stages; ++s) {
        const uint8_t* Qs = Q + (size_t)s * rows * (KDIM / 2);
        const long long lo = (long long)blockIdx.x * ((nitem + gridDim.x - 1) / gridDim.x);
        const long long hi = min(lo + (nitem + gridDim.x - 1) / gridDim.x, nitem);
        for (long long it = lo + wid; it < hi; it += nw) {
            float acc[R];
            gemv_int4<KDIM, R, 128, false>(Qs + (size_t)(it * R) * (KDIM / 2),
                                           S + (size_t)(it * R) * (KDIM / 128),
                                           Z + (size_t)(it * R) * (KDIM / 128),
                                           xs, lane, acc);
            #pragma unroll
            for (int r = 0; r < R; ++r) sink += acc[r];
        }
        if (coop) cg::this_grid().sync();
    }
    if (sink == 1e30f) out[0] = sink;
}

__global__ void k_barrier(int iters, float* out) {
    cg::grid_group grid = cg::this_grid();
    float v = 0.f;
    for (int i = 0; i < iters; ++i) { grid.sync(); v += 1.f; }
    if (v == 1e30f) out[0] = v;
}
__global__ void k_empty(float* out) { if (threadIdx.x == (1 << 30)) out[0] = 1.f; }

// ---------------------------------------------------------------- timing
struct Timer {
    cudaEvent_t a, b;
    Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start() { CK(cudaEventRecord(a)); }
    float stop() {
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms;
    }
};
static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// ---------------------------------------------------------------- records
struct GemvRow { std::string quant; int nt, r; double gbs; int bps, regs; };
struct StageRow { std::string quant; int nt, r, k; size_t bytes; size_t items; double us, gbs; };
static std::vector<GemvRow>  g_gemv;
static std::vector<StageRow> g_stage;
static std::vector<std::pair<int, double>> g_bars;

static double barrier_at(int grid) {
    if (g_bars.empty()) return 1.1;
    double best = g_bars[0].second; int bd = abs(g_bars[0].first - grid);
    for (auto& b : g_bars) { int d = abs(b.first - grid); if (d < bd) { bd = d; best = b.second; } }
    return best;
}

// ============================================================================
// One benchmark family per quant, instantiated by macro so that NT and R stay
// compile-time constants (they must: they set register blocking and unrolling).
// ============================================================================
#define DENSE_CASE(TY, QN, KDIM, NT, R)                                                       \
{                                                                                        \
    auto kern = k_gemv_dense<TY, KDIM, NT, R>;                                                 \
    const size_t smem = pad33_size(KDIM) * sizeof(float);                                \
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));\
    int bps = 0;                                                                         \
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NT, smem));             \
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));                          \
    const int grid = std::max(1, bps * g_sms);                                           \
    /* --- ceiling: whole buffer, one pass --- */                                        \
    long long rows = (long long)(g_bytes / (KDIM * sizeof(TY)));                         \
    rows -= rows % (8 * R);                                                              \
    double bb = (double)rows * KDIM * sizeof(TY);                                        \
    Timer t; int one = 1, zero = 0;                                                      \
    kern<<<grid, NT, smem>>>((const TY*)g_buf, rows, one, zero, g_out);                  \
    CK(cudaDeviceSynchronize());                                                         \
    std::vector<float> ms;                                                               \
    for (int i = 0; i < 5; ++i) { t.start();                                             \
        kern<<<grid, NT, smem>>>((const TY*)g_buf, rows, one, zero, g_out);              \
        ms.push_back(t.stop()); }                                                        \
    CK(cudaGetLastError());                                                              \
    double peak = bb / (med(ms) * 1e-3) / 1e9;                                           \
    if (KDIM == 4096) g_gemv.push_back({QN, NT, R, peak, bps, fa.numRegs});                                \
    printf("  gemv %-8s K=%-5d NT=%-5d R=%-2d  %7.0f GB/s  %d blk/SM  %d regs\n",       \
           QN, KDIM, NT, R, peak, bps, fa.numRegs);                                            \
    /* --- stage curve: persistent grid, fresh slice per stage --- */                    \
    for (long long rws = 8 * R; ; rws *= 4) {                                            \
        double sb = (double)rws * KDIM * sizeof(TY);                                     \
        if (sb > 1.6e9) break;                                                           \
        int stages = (int)std::min(400.0, (double)g_bytes / sb);                         \
        if (stages < 8) break;                                                           \
        void* args[] = {(void*)&g_buf, (void*)&rws, (void*)&stages, (void*)&one, (void*)&g_out}; \
        CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));           \
        CK(cudaDeviceSynchronize());                                                     \
        std::vector<float> sms_;                                                         \
        for (int i = 0; i < 5; ++i) { t.start();                                         \
            CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));       \
            sms_.push_back(t.stop()); }                                                  \
        double us = med(sms_) * 1000.0 / stages - barrier_at(grid);                      \
        if (us < 1e-3) us = 1e-3;                                                        \
        g_stage.push_back({QN, NT, R, KDIM, (size_t)sb, (size_t)(rws / R), us, sb / (us * 1e-6) / 1e9});\
        printf("    stage %8.2f MB %8lld items %8.2f us %7.0f GB/s\n",                   \
               sb / 1e6, (long long)(rws / R), us, sb / (us * 1e-6) / 1e9);              \
    }                                                                                    \
}

#define FP8_CASE(KDIM, NT, R)                                                            \
{                                                                                        \
    auto kern = k_gemv_fp8<KDIM, NT, R>;                                                 \
    const size_t smem = pad33_size(KDIM) * sizeof(float);                                \
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));\
    int bps = 0;                                                                         \
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NT, smem));             \
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));                          \
    const int grid = std::max(1, bps * g_sms);                                           \
    size_t wb = (g_bytes * 15 / 16) & ~(size_t)4095;                                     \
    const __nv_fp8_e4m3* W = (const __nv_fp8_e4m3*)g_buf;                                \
    const float* S = (const float*)(g_buf + wb);                                         \
    long long rows = (long long)(wb / KDIM); rows -= rows % (8 * R);                      \
    double bb = (double)rows * KDIM;                                                     \
    Timer t; int one = 1, zero = 0;                                                      \
    kern<<<grid, NT, smem>>>(W, S, rows, one, zero, g_out);                              \
    CK(cudaDeviceSynchronize());                                                         \
    std::vector<float> ms;                                                               \
    for (int i = 0; i < 5; ++i) { t.start();                                             \
        kern<<<grid, NT, smem>>>(W, S, rows, one, zero, g_out); ms.push_back(t.stop()); } \
    CK(cudaGetLastError());                                                              \
    double peak = bb / (med(ms) * 1e-3) / 1e9;                                           \
    if (KDIM == 4096) g_gemv.push_back({"fp8", NT, R, peak, bps, fa.numRegs});          \
    printf("  gemv %-8s K=%-5d NT=%-5d R=%-2d  %7.0f GB/s  %d blk/SM  %d regs\n",        \
           "fp8", KDIM, NT, R, peak, bps, fa.numRegs);                                    \
    for (long long rws = 8 * R; ; rws *= 4) {                                            \
        double sb = (double)rws * KDIM;                                                  \
        if (sb > 1.6e9) break;                                                           \
        int stages = (int)std::min(400.0, (double)wb / sb);                              \
        if (stages < 8) break;                                                           \
        void* args[] = {(void*)&W, (void*)&S, (void*)&rws, (void*)&stages,               \
                        (void*)&one, (void*)&g_out};                                     \
        CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));           \
        CK(cudaDeviceSynchronize());                                                     \
        std::vector<float> sms_;                                                         \
        for (int i = 0; i < 5; ++i) { t.start();                                         \
            CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));       \
            sms_.push_back(t.stop()); }                                                  \
        double us = med(sms_) * 1000.0 / stages - barrier_at(grid);                      \
        if (us < 1e-3) us = 1e-3;                                                        \
        g_stage.push_back({"fp8", NT, R, KDIM, (size_t)sb, (size_t)(rws / R), us,      \
                           sb / (us * 1e-6) / 1e9});                                     \
    }                                                                                    \
}

#define INT4_CASE(KDIM, NT, R)                                                           \
{                                                                                        \
    auto kern = k_gemv_int4<KDIM, NT, R>;                                                \
    const size_t smem = pad33_size(KDIM) * sizeof(float);                                \
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));\
    int bps = 0;                                                                         \
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NT, smem));             \
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));                          \
    const int grid = std::max(1, bps * g_sms);                                           \
    size_t wb = (g_bytes * 7 / 8) & ~(size_t)4095;                                       \
    const uint8_t* Q = (const uint8_t*)g_buf;                                            \
    const __half* S = (const __half*)(g_buf + wb);                                       \
    const __half* Z = (const __half*)(g_buf + wb + (g_bytes - wb) / 2);                  \
    long long rows = (long long)(wb / (KDIM / 2)); rows -= rows % (8 * R);                \
    double bb = (double)rows * (KDIM / 2 + KDIM / 128 * 4);                              \
    Timer t; int one = 1, zero = 0;                                                      \
    kern<<<grid, NT, smem>>>(Q, S, Z, rows, one, zero, g_out);                           \
    CK(cudaDeviceSynchronize());                                                         \
    std::vector<float> ms;                                                               \
    for (int i = 0; i < 5; ++i) { t.start();                                             \
        kern<<<grid, NT, smem>>>(Q, S, Z, rows, one, zero, g_out); ms.push_back(t.stop()); }\
    CK(cudaGetLastError());                                                              \
    double peak = bb / (med(ms) * 1e-3) / 1e9;                                           \
    if (KDIM == 4096) g_gemv.push_back({"int4", NT, R, peak, bps, fa.numRegs});      \
    printf("  gemv %-8s K=%-5d NT=%-5d R=%-2d  %7.0f GB/s  %d blk/SM  %d regs\n",        \
           "int4", KDIM, NT, R, peak, bps, fa.numRegs);                                   \
    for (long long rws = 8 * R; ; rws *= 4) {                                            \
        double sb = (double)rws * (KDIM / 2 + KDIM / 128 * 4);                           \
        if (sb > 1.6e9) break;                                                           \
        int stages = (int)std::min(400.0, (double)wb / ((double)rws * (KDIM / 2)));      \
        if (stages < 8) break;                                                           \
        void* args[] = {(void*)&Q, (void*)&S, (void*)&Z, (void*)&rws, (void*)&stages,    \
                        (void*)&one, (void*)&g_out};                                     \
        CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));           \
        CK(cudaDeviceSynchronize());                                                     \
        std::vector<float> sms_;                                                         \
        for (int i = 0; i < 5; ++i) { t.start();                                         \
            CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));       \
            sms_.push_back(t.stop()); }                                                  \
        double us = med(sms_) * 1000.0 / stages - barrier_at(grid);                      \
        if (us < 1e-3) us = 1e-3;                                                        \
        g_stage.push_back({"int4", NT, R, KDIM, (size_t)sb, (size_t)(rws / R), us,  \
                           sb / (us * 1e-6) / 1e9});                                     \
    }                                                                                    \
}

#define MX_CASE(KDIM, NT, R)                                                                   \
{                                                                                        \
    auto kern = k_gemv_mxfp4<KDIM, NT, R>;                                                     \
    const size_t smem = pad33_size(KDIM) * sizeof(float);                                \
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));\
    int bps = 0;                                                                         \
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, NT, smem));             \
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));                          \
    const int grid = std::max(1, bps * g_sms);                                           \
    size_t blkb = (g_bytes * 16 / 17) & ~(size_t)4095;                                   \
    const uint4*   blk = (const uint4*)g_buf;                                            \
    const uint8_t* scl = (const uint8_t*)(g_buf + blkb);                                 \
    long long rows = (long long)(blkb / ((KDIM / 32) * 16));                                    \
    rows -= rows % (8 * R);                                                              \
    double bb = (double)rows * KDIM * 17.0 / 32.0;                                       \
    Timer t; int one = 1, zero = 0;                                                      \
    kern<<<grid, NT, smem>>>(blk, scl, rows, one, zero, g_out);                          \
    CK(cudaDeviceSynchronize());                                                         \
    std::vector<float> ms;                                                               \
    for (int i = 0; i < 5; ++i) { t.start();                                             \
        kern<<<grid, NT, smem>>>(blk, scl, rows, one, zero, g_out);                      \
        ms.push_back(t.stop()); }                                                        \
    CK(cudaGetLastError());                                                              \
    double peak = bb / (med(ms) * 1e-3) / 1e9;                                           \
    if (KDIM == 4096) g_gemv.push_back({"mxfp4", NT, R, peak, bps, fa.numRegs});                        \
    printf("  gemv %-8s K=%-5d NT=%-5d R=%-2d  %7.0f GB/s  %d blk/SM  %d regs\n",       \
           "mxfp4x32", KDIM, NT, R, peak, bps, fa.numRegs);                                    \
    for (long long rws = 8 * R; ; rws *= 4) {                                            \
        double sb = (double)rws * KDIM * 17.0 / 32.0;                                    \
        if (sb > 1.6e9) break;                                                           \
        size_t sblk = (size_t)rws * (KDIM / 32) * 16;                                           \
        int stages = (int)std::min(400.0, (double)blkb / (double)sblk);                  \
        if (stages < 8) break;                                                           \
        void* args[] = {(void*)&blk, (void*)&scl, (void*)&rws, (void*)&stages,           \
                        (void*)&one, (void*)&g_out};                                     \
        CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));           \
        CK(cudaDeviceSynchronize());                                                     \
        std::vector<float> sms_;                                                         \
        for (int i = 0; i < 5; ++i) { t.start();                                         \
            CK(cudaLaunchCooperativeKernel((void*)kern, grid, NT, args, smem, 0));       \
            sms_.push_back(t.stop()); }                                                  \
        double us = med(sms_) * 1000.0 / stages - barrier_at(grid);                      \
        if (us < 1e-3) us = 1e-3;                                                        \
        g_stage.push_back({"mxfp4", NT, R, KDIM, (size_t)sb, (size_t)(rws / R), us,   \
                           sb / (us * 1e-6) / 1e9});                                     \
        printf("    stage %8.2f MB %8lld items %8.2f us %7.0f GB/s\n",                   \
               sb / 1e6, (long long)(rws / R), us, sb / (us * 1e-6) / 1e9);              \
    }                                                                                    \
}

int main(int argc, char** argv) {
    const char* outpath = (argc > 1) ? argv[1] : "hw.json";
    double budget_gb = (argc > 2) ? atof(argv[2]) : 16.0;

    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    g_sms = pr.multiProcessorCount;
    printf("device: %s  sm_%d%d  %d SMs  L2=%.0f MB  smem/SM=%d KB\n",
           pr.name, pr.major, pr.minor, g_sms, pr.l2CacheSize / 1e6,
           (int)(pr.sharedMemPerMultiprocessor / 1024));
    const double dram_peak = 2.0 * pr.memoryClockRate * 1e3 * (pr.memoryBusWidth / 8) / 1e9;

    size_t freeb = 0, totb = 0; CK(cudaMemGetInfo(&freeb, &totb));
    g_bytes = (size_t)(budget_gb * 1e9);
    if (g_bytes > freeb * 3 / 4) g_bytes = freeb * 3 / 4;
    g_bytes &= ~(size_t)((1 << 20) - 1);
    printf("buffer: %.2f GB (free %.1f GB)\n", g_bytes / 1e9, freeb / 1e9);
    CK(cudaMalloc(&g_buf, g_bytes));
    CK(cudaMemset(g_buf, 0x3c, g_bytes));
    CK(cudaMalloc(&g_out, 4));

    // ------------------------------------------------------------ [3] first:
    // the barrier cost is subtracted from every stage measurement below.
    printf("\n[3] cooperative grid barrier\n");
    {
        int bps = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, k_barrier, 256, 0));
        for (int mul = 1; mul <= bps; mul *= 2) {
            int grid = mul * g_sms;
            const int iters = 2000;
            void* args[] = {(void*)&iters, (void*)&g_out};
            Timer t;
            CK(cudaLaunchCooperativeKernel((void*)k_barrier, grid, 256, args, 0, 0));
            CK(cudaDeviceSynchronize());
            std::vector<float> ms;
            for (int i = 0; i < 5; ++i) {
                t.start();
                CK(cudaLaunchCooperativeKernel((void*)k_barrier, grid, 256, args, 0, 0));
                ms.push_back(t.stop());
            }
            double us = med(ms) * 1000.0 / iters;
            g_bars.push_back({grid, us});
            printf("  grid=%-5d  %.3f us / barrier\n", grid, us);
        }
    }

    // ------------------------------------------------------------ [1] stream
    printf("\n[1] streaming read (cold footprint -> achieved GB/s)\n");
    std::vector<std::pair<size_t, double>> ramp;
    {
        Timer t;
        for (size_t mb = 2; mb * 1000000 <= g_bytes; mb *= 2) {
            size_t n = mb * 1000000 / 16;
            int grid = g_sms * 16;
            size_t span = g_bytes - mb * 1000000;
            k_stream<<<grid, 512>>>((const float4*)g_buf, n, g_out);
            CK(cudaDeviceSynchronize());
            std::vector<float> ms;
            for (int i = 0; i < 9; ++i) {
                size_t off = span ? ((size_t)((double)span * i / 9) & ~(size_t)255) : 0;
                t.start();
                k_stream<<<grid, 512>>>((const float4*)(g_buf + off), n, g_out);
                ms.push_back(t.stop());
            }
            double gbs = (double)(n * 16) / (med(ms) * 1e-3) / 1e9;
            ramp.push_back({mb * 1000000, gbs});
            printf("  %8zu MB  %7.0f GB/s\n", mb, gbs);
        }
    }
    const double stream_peak = ramp.back().second;

    // ------------------------------------------------------------ [2]/[2b]
    printf("\n[2] gemv cores: ceiling, then stage curve inside a persistent grid\n");
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 256, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 256, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 256, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 256, 8)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 512, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 512, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 512, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 512, 8)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 1024, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 1024, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 1024, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 4096, 1024, 8)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 256, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 256, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 256, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 256, 8)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 512, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 512, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 512, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 512, 8)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 1024, 1)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 1024, 2)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 1024, 4)
    DENSE_CASE(__nv_bfloat16, "bf16", 1024, 1024, 8)
    DENSE_CASE(__half, "f16", 4096, 512, 2)
    DENSE_CASE(__half, "f16", 4096, 512, 4)
    DENSE_CASE(__half, "f16", 1024, 512, 2)
    DENSE_CASE(__half, "f16", 1024, 512, 4)
    MX_CASE(4096, 256, 1)
    MX_CASE(4096, 256, 2)
    MX_CASE(4096, 256, 4)
    MX_CASE(4096, 256, 8)
    MX_CASE(4096, 512, 1)
    MX_CASE(4096, 512, 2)
    MX_CASE(4096, 512, 4)
    MX_CASE(4096, 512, 8)
    MX_CASE(4096, 1024, 1)
    MX_CASE(4096, 1024, 2)
    MX_CASE(4096, 1024, 4)
    MX_CASE(4096, 1024, 8)
    MX_CASE(1024, 256, 1)
    MX_CASE(1024, 256, 2)
    MX_CASE(1024, 256, 4)
    MX_CASE(1024, 256, 8)
    MX_CASE(1024, 512, 1)
    MX_CASE(1024, 512, 2)
    MX_CASE(1024, 512, 4)
    MX_CASE(1024, 512, 8)
    MX_CASE(1024, 1024, 1)
    MX_CASE(1024, 1024, 2)
    MX_CASE(1024, 1024, 4)
    MX_CASE(1024, 1024, 8)
    FP8_CASE(4096, 256, 1)
    FP8_CASE(4096, 256, 2)
    FP8_CASE(4096, 256, 4)
    FP8_CASE(4096, 512, 1)
    FP8_CASE(4096, 512, 2)
    FP8_CASE(4096, 512, 4)
    FP8_CASE(4096, 1024, 1)
    FP8_CASE(4096, 1024, 2)
    FP8_CASE(4096, 1024, 4)
    FP8_CASE(1024, 256, 1)
    FP8_CASE(1024, 256, 2)
    FP8_CASE(1024, 256, 4)
    FP8_CASE(1024, 512, 1)
    FP8_CASE(1024, 512, 2)
    FP8_CASE(1024, 512, 4)
    FP8_CASE(1024, 1024, 1)
    FP8_CASE(1024, 1024, 2)
    FP8_CASE(1024, 1024, 4)
    INT4_CASE(4096, 256, 1)
    INT4_CASE(4096, 256, 2)
    INT4_CASE(4096, 256, 4)
    INT4_CASE(4096, 512, 1)
    INT4_CASE(4096, 512, 2)
    INT4_CASE(4096, 512, 4)
    INT4_CASE(4096, 1024, 1)
    INT4_CASE(4096, 1024, 2)
    INT4_CASE(4096, 1024, 4)
    INT4_CASE(1024, 256, 1)
    INT4_CASE(1024, 256, 2)
    INT4_CASE(1024, 256, 4)
    INT4_CASE(1024, 512, 1)
    INT4_CASE(1024, 512, 2)
    INT4_CASE(1024, 512, 4)
    INT4_CASE(1024, 1024, 1)
    INT4_CASE(1024, 1024, 2)
    INT4_CASE(1024, 1024, 4)

    // ------------------------------------------------------------ [4] launch
    double launch_us;
    {
        const int n = 2000;
        Timer t; k_empty<<<1, 32>>>(g_out); CK(cudaDeviceSynchronize());
        std::vector<float> ms;
        for (int i = 0; i < 5; ++i) {
            t.start();
            for (int j = 0; j < n; ++j) k_empty<<<132, 256>>>(g_out);
            ms.push_back(t.stop());
        }
        launch_us = med(ms) * 1000.0 / n;
        printf("\n[4] kernel launch: %.3f us\n", launch_us);
    }

    // ------------------------------------------------------------ emit
    FILE* f = fopen(outpath, "w");
    if (!f) { perror(outpath); return 1; }
    // The schema version.  `mkc` refuses an older machine file rather than
    // silently planning every stage from a proxy curve: bump this whenever the
    // meaning of a field changes.  2 = the tables are keyed by gemv CORE
    // ("bf16", "f16", "mxfp4", "fp8", "int4"), not by the format's full tag.
    fprintf(f, "{\n  \"schema\": 2,\n");
    fprintf(f, "  \"device\": \"%s\",\n  \"sm_arch\": %d,\n  \"sms\": %d,\n",
            pr.name, pr.major * 10 + pr.minor, g_sms);
    fprintf(f, "  \"clock_ghz\": %.4f,\n  \"l2_bytes\": %d,\n", pr.clockRate / 1e6, pr.l2CacheSize);
    fprintf(f, "  \"smem_per_sm\": %zu,\n  \"smem_per_block_max\": %zu,\n",
            pr.sharedMemPerMultiprocessor, pr.sharedMemPerBlockOptin);
    fprintf(f, "  \"max_threads_per_sm\": %d,\n  \"regs_per_sm\": %d,\n",
            pr.maxThreadsPerMultiProcessor, pr.regsPerMultiprocessor);
    fprintf(f, "  \"dram_peak_gbs\": %.1f,\n  \"stream_peak_gbs\": %.1f,\n", dram_peak, stream_peak);
    fprintf(f, "  \"ramp\": [\n");
    for (size_t i = 0; i < ramp.size(); ++i)
        fprintf(f, "    {\"bytes\": %zu, \"gbs\": %.1f}%s\n", ramp[i].first, ramp[i].second,
                i + 1 < ramp.size() ? "," : "");
    fprintf(f, "  ],\n  \"gemv\": [\n");
    for (size_t i = 0; i < g_gemv.size(); ++i)
        fprintf(f, "    {\"quant\": \"%s\", \"nt\": %d, \"r\": %d, \"gbs\": %.1f, "
                   "\"blocks_per_sm\": %d, \"regs\": %d}%s\n",
                g_gemv[i].quant.c_str(), g_gemv[i].nt, g_gemv[i].r, g_gemv[i].gbs,
                g_gemv[i].bps, g_gemv[i].regs, i + 1 < g_gemv.size() ? "," : "");
    fprintf(f, "  ],\n  \"stage\": [\n");
    for (size_t i = 0; i < g_stage.size(); ++i)
        fprintf(f, "    {\"quant\": \"%s\", \"nt\": %d, \"r\": %d, \"k\": %d, \"bytes\": %zu, "
                   "\"items\": %zu, \"us\": %.4f, \"gbs\": %.1f}%s\n",
                g_stage[i].quant.c_str(), g_stage[i].nt, g_stage[i].r, g_stage[i].k,
                g_stage[i].bytes, g_stage[i].items, g_stage[i].us, g_stage[i].gbs,
                i + 1 < g_stage.size() ? "," : "");
    fprintf(f, "  ],\n  \"barrier_us\": [");
    for (size_t i = 0; i < g_bars.size(); ++i)
        fprintf(f, "[%d, %.4f]%s", g_bars[i].first, g_bars[i].second,
                i + 1 < g_bars.size() ? ", " : "");
    fprintf(f, "],\n  \"launch_us\": %.4f,\n  \"calibrated_at\": \"%s\",\n"
               "  \"mkc_version\": \"0.1.0\"\n}\n", launch_us, __DATE__ " " __TIME__);
    fclose(f);
    printf("\nwrote %s\n", outpath);
    return 0;
}
