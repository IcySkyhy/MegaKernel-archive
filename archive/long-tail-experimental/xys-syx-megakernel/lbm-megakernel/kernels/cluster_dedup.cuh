#pragma once
#include "cluster_compat.cuh"
#include "temporal_common.cuh"
#include "cluster_debug.cuh"
namespace temporal {
namespace cg=cooperative_groups;
// C2 / D-V3-Z2-256. Per-block R remains 16x8x4, FP32, 38912 shared bytes.
// src != dst; immutable flags and never-produced slots retain the V2 contract.
// Each consumed (r,q) has one logical producer p=r-cq, owned by exactly one
// rank. Only Phase 1 stores can be remote. Phase 2 reads local shared only.
__global__ void z2_dedup(const float *__restrict__ src,float *__restrict__ dst) {
    constexpr int TX=16,TY=8,TZ=4,HX=18,HY=10;
    constexpr int CORE_SITES=512, PRODUCERS_PER_BLOCK=900;
    extern __shared__ float intermediate_R[];
    auto cluster=cg::this_cluster();
    Z2_ENTER(cluster);
    const dim3 cluster_block=cluster.block_index();
    const int rank_z=int(cluster_block.z);
    const int cluster_block_z=int(blockIdx.z)-rank_z;
    const int cox=int(blockIdx.x)*TX, coy=int(blockIdx.y)*TY;
    const int coz=cluster_block_z*TZ;
    const int ox=cox, oy=coy, oz=int(blockIdx.z)*TZ;

    // All participating blocks must exist before any mapped shared access.
    cluster.sync();
    for(int i=int(threadIdx.x);i<PRODUCERS_PER_BLOCK;i+=int(blockDim.x)) {
        const int px=i%HX-1, py=(i/HX)%HY-1;
        const int zslot=i/(HX*HY);
        const int pz=rank_z==0?zslot-1:zslot+4;
        const int s=site(cox+px,coy+py,coz+pz);
        const bool valid=has_producer(s);
        Z2_PRODUCER(px,py,pz,rank_z,cluster_block_z,valid);
        D128Populations p;
        if(valid) {
            p=load(src,s);
            p.collide(flags_at(src,s));
        }
#define CLUSTER_STREAM(q,dx,dy,dz) { \
        const int rx=px+(dx), ry=py+(dy), rz=pz+(dz); \
        if(rx>=0 && rx<TX && ry>=0 && ry<TY && rz>=0 && rz<2*TZ && \
           cox+rx<SIZE_X && coy+ry<SIZE_Y && coz+rz<SIZE_Z) { \
            const int dst_rank_z=rz/TZ, local_z=rz-dst_rank_z*TZ; \
            const int dest=rx+TX*(ry+TY*local_z); \
            float *target=intermediate_R; \
            if(dst_rank_z!=rank_z) \
                target=cluster.map_shared_rank(intermediate_R,unsigned(dst_rank_z)); \
            if(valid) \
                target[q*CORE_SITES+dest]=p.temp##q; \
            else { \
                const int rs=site(cox+rx,coy+ry,coz+rz); \
                target[q*CORE_SITES+dest]=src[q*TOTAL_PADDED_CELLS+rs]; \
            } \
            Z2_STORE(q,cox+rx,coy+ry,coz+rz,valid,dst_rank_z!=rank_z,rank_z,cluster_block_z); \
        } \
    }
        LBM_DIRECTIONS(CLUSTER_STREAM)
#undef CLUSTER_STREAM
    }
    // Finish remote accesses before any block reads its local state or exits.
    cluster.sync();

    for (int i = int(threadIdx.x); i < TX * TY * TZ; i += int(blockDim.x)) {
        const int lx = i % TX, ly = (i / TX) % TY, lz = i / (TX * TY);
        const int x = ox + lx, y = oy + ly, z = oz + lz;
        if (x >= SIZE_X || y >= SIZE_Y || z >= SIZE_Z) continue;
        Z2_CORE(x,y,z);
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
