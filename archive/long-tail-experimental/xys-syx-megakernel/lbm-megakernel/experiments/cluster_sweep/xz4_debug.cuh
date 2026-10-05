#pragma once
#ifdef SWEEP_XZ4_DEBUG
namespace xz4_debug {
enum { PRODUCER=0, VALID=4, LOCAL=8, REMOTE=9, FALLBACK_LOCAL=10,
       FALLBACK_REMOTE=11, REMOTE_X=12, REMOTE_Z=13, REMOTE_XZ=14, COUNT=15 };
__device__ unsigned *writers,*cores,*producers;
__device__ unsigned long long counts[COUNT];
__device__ __forceinline__ void enter(cooperative_groups::cluster_group c) {
    const dim3 d=c.dim_blocks(),b=c.block_index();
    assert(d.x==2&&d.y==1&&d.z==2);
    assert(blockDim.x==256&&blockDim.y==1&&blockDim.z==1);
    assert(c.block_rank()==b.x+2*(b.y+b.z));
}
__device__ __forceinline__ void producer(int px,int py,int pz,unsigned rank,int cx,int cz,bool valid) {
    const int cluster_id=cx/2+4*(int(blockIdx.y)+15*(cz/2));
    atomicAdd(producers+cluster_id*3400+(px+1)+34*((py+1)+10*(pz+1)),1u);
    atomicAdd(counts+PRODUCER+rank,1ull);
    if(valid)atomicAdd(counts+VALID+rank,1ull);
}
__device__ __forceinline__ void store(int q,int x,int y,int z,bool valid,unsigned from,unsigned to) {
    atomicAdd(writers+q*TOTAL_CELLS+x+SIZE_X*(y+SIZE_Y*z),1u);
    const bool remote=from!=to;
    atomicAdd(counts+(valid?(remote?REMOTE:LOCAL):(remote?FALLBACK_REMOTE:FALLBACK_LOCAL)),1ull);
    if(remote) {
        const unsigned crossed=from^to;
        atomicAdd(counts+(crossed==1?REMOTE_X:crossed==2?REMOTE_Z:REMOTE_XZ),1ull);
    }
}
}
#define XZ_ENTER(c) xz4_debug::enter(c)
#define XZ_PRODUCER(...) xz4_debug::producer(__VA_ARGS__)
#define XZ_STORE(...) xz4_debug::store(__VA_ARGS__)
#define XZ_CORE(x,y,z) atomicAdd(xz4_debug::cores+(x)+SIZE_X*((y)+SIZE_Y*(z)),1u)
#else
#define XZ_ENTER(c) ((void)0)
#define XZ_PRODUCER(...) ((void)0)
#define XZ_STORE(...) ((void)0)
#define XZ_CORE(x,y,z) ((void)0)
#endif
