#pragma once
#include "../../kernels/cluster_compat.cuh"
#include "../../kernels/temporal_common.cuh"
#include "xz4_debug.cuh"
namespace temporal {
namespace cg=cooperative_groups;
// C1: full duplicated 18x10x6 producer box per block; no DSM accesses.
__global__ void xz4_sync(const float *__restrict__ src,float *__restrict__ dst) {
    constexpr int TX=16,TY=8,TZ=4,HX=18,HY=10,HZ=6;
    constexpr int HALO_SITES=1080, CORE_SITES=512;
    extern __shared__ float intermediate_R[];
    auto cluster=cg::this_cluster();
    XZ_ENTER(cluster);
    cluster.sync();
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
    cluster.sync();

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

// Rank linearization for a (2,1,2) cluster. X is the fastest block coordinate.
__device__ __forceinline__ unsigned xz4_rank(int bx,int by,int bz) {
    return unsigned(bx+2*(by+bz));
}
// Each block owns a 17x10x5 sub-box of the cluster's full 34x10x10 producers.
// Exactly two cluster synchronizations; remote communication is Phase 1 only.
__global__ void xz4_dedup(const float *__restrict__ src,float *__restrict__ dst) {
    constexpr int TX=16,TY=8,TZ=4,CORE_SITES=512;
    constexpr int LOCAL_PX=17,LOCAL_PY=10,PRODUCERS_PER_BLOCK=850;
    extern __shared__ float intermediate_R[];
    auto cluster=cg::this_cluster();
    XZ_ENTER(cluster);
    const dim3 bc=cluster.block_index();
    const int bx=int(bc.x),bz=int(bc.z);
    const unsigned rank=cluster.block_rank();
    const int cluster_bx=int(blockIdx.x)-bx,cluster_bz=int(blockIdx.z)-bz;
    const int cox=cluster_bx*TX,coy=int(blockIdx.y)*TY,coz=cluster_bz*TZ;
    const int ox=int(blockIdx.x)*TX,oy=coy,oz=int(blockIdx.z)*TZ;
    cluster.sync();
    for(int i=int(threadIdx.x);i<PRODUCERS_PER_BLOCK;i+=int(blockDim.x)) {
        const int ix=i%LOCAL_PX,iy=(i/LOCAL_PX)%LOCAL_PY,iz=i/(LOCAL_PX*LOCAL_PY);
        const int px=bx==0?ix-1:ix+16,py=iy-1,pz=bz==0?iz-1:iz+4;
        const int s=site(cox+px,coy+py,coz+pz);
        const bool valid=has_producer(s);
        XZ_PRODUCER(px,py,pz,rank,cluster_bx,cluster_bz,valid);
        D128Populations p;
        if(valid) {p=load(src,s);p.collide(flags_at(src,s));}
#define XZ_STREAM(q,dx,dy,dz) { \
        const int rx=px+(dx),ry=py+(dy),rz=pz+(dz); \
        if(rx>=0&&rx<2*TX&&ry>=0&&ry<TY&&rz>=0&&rz<2*TZ&& \
           cox+rx<SIZE_X&&coy+ry<SIZE_Y&&coz+rz<SIZE_Z) { \
            const int dst_bx=rx/TX,dst_bz=rz/TZ; \
            const int local_x=rx-dst_bx*TX,local_z=rz-dst_bz*TZ; \
            const unsigned dst_rank=xz4_rank(dst_bx,0,dst_bz); \
            const int dest=local_x+TX*(ry+TY*local_z); \
            float *target=intermediate_R; \
            if(dst_rank!=rank)target=cluster.map_shared_rank(intermediate_R,dst_rank); \
            if(valid)target[q*CORE_SITES+dest]=p.temp##q; \
            else {const int rs=site(cox+rx,coy+ry,coz+rz); \
                  target[q*CORE_SITES+dest]=src[q*TOTAL_PADDED_CELLS+rs];} \
            XZ_STORE(q,cox+rx,coy+ry,coz+rz,valid,rank,dst_rank); \
        } \
    }
        LBM_DIRECTIONS(XZ_STREAM)
#undef XZ_STREAM
    }
    cluster.sync();

    for (int i = int(threadIdx.x); i < TX * TY * TZ; i += int(blockDim.x)) {
        const int lx = i % TX, ly = (i / TX) % TY, lz = i / (TX * TY);
        const int x = ox + lx, y = oy + ly, z = oz + lz;
        if (x >= SIZE_X || y >= SIZE_Y || z >= SIZE_Z) continue;
        XZ_CORE(x,y,z);
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
