// ============================================================================
// tp_probe -- what a tensor-parallel megakernel would actually cost.
//
// At batch 1 a TP megakernel is not limited by the bytes it exchanges: the
// hidden state is a few kilobytes.  It is limited by how long it takes N
// persistent kernels on N GPUs to AGREE, twice per layer.  So the number that
// decides whether TP is worth building is the latency of one cross-device
// rendezvous plus one small all-reduce, measured between kernels that are
// already resident -- not the bandwidth of a NCCL call.
//
// One process, N devices, peer access, one persistent cooperative kernel per
// device, each on its own stream.  The kernels rendezvous through peer-mapped
// counters exactly the way a real TP megakernel would.
//
// STATUS: world=2 measures cleanly (rendezvous ~5.7 us, all-reduce ~52 us --
// and that 52 is this file's fault, not the machine's: three intra-GPU
// rendezvous and a __threadfence_system() per all-reduce).  world=4 HANGS and
// has not been diagnosed; suspect the peer-mapped flag array, since every rank
// waits on its own row and each peer bumps one column.  Fix that before
// believing any 4-way number from here.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("cuda %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

constexpr int MAXW = 8;

struct Peers {
    float*        buf[MAXW];      // buf[p] is rank p's staging area, peer-mapped
    unsigned int* flag[MAXW];     // buf[p]'s arrival counters
    int rank, world, h;
};

// One all-reduce of `h` floats, the way a megakernel would do it:
//   1. every rank writes its partial into every peer's slot for this rank
//   2. a system-scope fence, then one counter bump per peer
//   3. every rank waits for all `world` counters to reach this epoch
//   4. every rank sums the `world` slots
// Steps 1 and 4 are bandwidth (kilobytes); steps 2 and 3 are the latency that
// decides whether TP pays.
__device__ __forceinline__ void allreduce(Peers p, float* mine, unsigned int epoch,
                                          cg::grid_group& grid) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;
    for (int i = tid; i < p.h; i += nth) {
        const float v = mine[i];
        for (int r = 0; r < p.world; ++r) p.buf[r][p.rank * p.h + i] = v;
    }
    __threadfence_system();
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int r = 0; r < p.world; ++r) atomicAdd(p.flag[r] + p.rank, 1u);
    if (threadIdx.x == 0) {
        for (int r = 0; r < p.world; ++r) {
            volatile unsigned int* f = p.flag[p.rank] + r;
            while (*f < epoch) __nanosleep(64);
        }
    }
    grid.sync();
    for (int i = tid; i < p.h; i += nth) {
        float s = 0.f;
        for (int r = 0; r < p.world; ++r) s += p.buf[p.rank][r * p.h + i];
        mine[i] = s;
    }
    grid.sync();
}

__global__ void k_tp(Peers p, float* mine, int iters, unsigned long long* out) {
    cg::grid_group grid = cg::this_grid();
    unsigned long long t0 = 0;
    for (int it = 1; it <= iters; ++it) {
        if (it == 9 && blockIdx.x == 0 && threadIdx.x == 0)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
        allreduce(p, mine, (unsigned)it, grid);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1) :: "memory");
        *out = t1 - t0;
    }
}

// The same rendezvous with no data: the floor a TP schedule cannot go below.
__global__ void k_bar(Peers p, int iters, unsigned long long* out) {
    cg::grid_group grid = cg::this_grid();
    unsigned long long t0 = 0;
    for (int it = 1; it <= iters; ++it) {
        if (it == 9 && blockIdx.x == 0 && threadIdx.x == 0)
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
        grid.sync();
        if (blockIdx.x == 0 && threadIdx.x == 0)
            for (int r = 0; r < p.world; ++r) atomicAdd(p.flag[r] + p.rank, 1u);
        if (threadIdx.x == 0)
            for (int r = 0; r < p.world; ++r) {
                volatile unsigned int* f = p.flag[p.rank] + r;
                while (*f < (unsigned)it) __nanosleep(64);
            }
        grid.sync();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1) :: "memory");
        *out = t1 - t0;
    }
}

int main(int argc, char** argv) {
    int world = argc > 1 ? atoi(argv[1]) : 4;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (world > ndev) world = ndev;
    printf("tp_probe: world=%d of %d devices, %d iterations\n", world, ndev, iters);

    for (int i = 0; i < world; ++i) {
        CK(cudaSetDevice(i));
        for (int j = 0; j < world; ++j) {
            if (i == j) continue;
            int can = 0; CK(cudaDeviceCanAccessPeer(&can, i, j));
            if (!can) { printf("  device %d cannot peer with %d -- TP would go over PCIe\n", i, j); }
            else { cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                   if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
        }
    }

    for (int h : {1024, 2880, 4096, 8192}) {
        std::vector<float*> buf(world), mine(world);
        std::vector<unsigned int*> flag(world);
        std::vector<unsigned long long*> out(world);
        std::vector<cudaStream_t> st(world);
        for (int i = 0; i < world; ++i) {
            CK(cudaSetDevice(i));
            CK(cudaMalloc(&buf[i], (size_t)world * h * sizeof(float)));
            CK(cudaMalloc(&mine[i], (size_t)h * sizeof(float)));
            CK(cudaMalloc(&flag[i], (size_t)world * sizeof(unsigned int)));
            CK(cudaMalloc(&out[i], sizeof(unsigned long long)));
            CK(cudaMemset(flag[i], 0, world * sizeof(unsigned int)));
            CK(cudaMemset(mine[i], 0, h * sizeof(float)));
            CK(cudaStreamCreate(&st[i]));
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < world; ++i) {
                CK(cudaSetDevice(i));
                CK(cudaMemset(flag[i], 0, world * sizeof(unsigned int)));
                Peers p{}; p.rank = i; p.world = world; p.h = h;
                for (int r = 0; r < world; ++r) { p.buf[r] = buf[r]; p.flag[r] = flag[r]; }
                int bps = 0;
                void* kern = pass ? (void*)k_bar : (void*)k_tp;
                CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kern, 256, 0));
                cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, i));
                const int grid = bps * pr.multiProcessorCount;
                int it = iters;
                void* a0[] = {(void*)&p, (void*)&mine[i], (void*)&it, (void*)&out[i]};
                void* a1[] = {(void*)&p, (void*)&it, (void*)&out[i]};
                CK(cudaLaunchCooperativeKernel(kern, grid, 256, pass ? a1 : a0, 0, st[i]));
            }
            unsigned long long ns = 0;
            for (int i = 0; i < world; ++i) {
                CK(cudaSetDevice(i)); CK(cudaStreamSynchronize(st[i]));
                unsigned long long v; CK(cudaMemcpy(&v, out[i], 8, cudaMemcpyDeviceToHost));
                if (v > ns) ns = v;
            }
            const double us = (double)ns / 1000.0 / (iters - 8);
            if (pass) printf("  rendezvous only                      %6.2f us\n", us);
            else      printf("  all-reduce h=%-5d (%5.1f KB/rank)   %6.2f us\n",
                             h, world * h * 4 / 1024.0, us);
        }
        for (int i = 0; i < world; ++i) {
            CK(cudaSetDevice(i));
            cudaFree(buf[i]); cudaFree(mine[i]); cudaFree(flag[i]); cudaFree(out[i]);
            cudaStreamDestroy(st[i]);
        }
    }
    return 0;
}
