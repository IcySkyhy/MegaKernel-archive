#pragma once
#ifdef LBM_Z2_DEBUG
#include <cassert>
namespace z2_debug {
enum Counter { VISIT0, VISIT1, VALID0, VALID1, LOCAL, REMOTE,
    FALLBACK_LOCAL, FALLBACK_REMOTE, REMOTE0, REMOTE1,
    INTERIOR_REMOTE0, INTERIOR_REMOTE1, COUNT };
__device__ unsigned *writers, *cores, *producers;
__device__ unsigned long long counts[COUNT];
__device__ __forceinline__ void enter(cooperative_groups::cluster_group cluster) {
    const dim3 dims=cluster.dim_blocks();
    assert(dims.x==1 && dims.y==1 && dims.z==2);
    assert(blockDim.x==256 && blockDim.y==1 && blockDim.z==1);
    assert(cluster.block_rank()==cluster.block_index().z);
}
__device__ __forceinline__ void producer(int px,int py,int pz,int rank,int cz,bool valid) {
    const int cluster_id=int(blockIdx.x)+8*(int(blockIdx.y)+15*(cz/2));
    const int candidate=(px+1)+18*((py+1)+10*(pz+1));
    atomicAdd(producers+cluster_id*1800+candidate,1u);
    atomicAdd(counts+VISIT0+rank,1ull);
    if(valid) atomicAdd(counts+VALID0+rank,1ull);
}
__device__ __forceinline__ void store(int q,int x,int y,int z,bool valid,bool remote,int rank,int cz) {
    atomicAdd(writers+q*TOTAL_CELLS+x+SIZE_X*(y+SIZE_Y*z),1u);
    atomicAdd(counts+(valid?(remote?REMOTE:LOCAL):(remote?FALLBACK_REMOTE:FALLBACK_LOCAL)),1ull);
    if(remote) {
        atomicAdd(counts+REMOTE0+rank,1ull);
        if(blockIdx.x==1 && blockIdx.y==1 && cz==2)
            atomicAdd(counts+INTERIOR_REMOTE0+rank,1ull);
    }
}
}
#define Z2_ENTER(c) z2_debug::enter(c)
#define Z2_PRODUCER(...) z2_debug::producer(__VA_ARGS__)
#define Z2_STORE(...) z2_debug::store(__VA_ARGS__)
#define Z2_CORE(x,y,z) atomicAdd(z2_debug::cores+(x)+SIZE_X*((y)+SIZE_Y*(z)),1u)
#else
#define Z2_ENTER(c) ((void)0)
#define Z2_PRODUCER(...) ((void)0)
#define Z2_STORE(...) ((void)0)
#define Z2_CORE(x,y,z) ((void)0)
#endif
