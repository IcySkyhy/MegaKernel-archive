#pragma once
#include "../kernels/b0.cuh"
#include "../kernels/c1.cuh"
#include "../kernels/c2.cuh"
#include "../kernels/c3.cuh"
#include "../kernels/s2_c128.cuh"
#include "../kernels/s2_d128.cuh"
#include "../kernels/t30_s1.cuh"
#include "../kernels/temporal_v1.cuh"
#include "../kernels/temporal_v2.cuh"

#ifdef LBM_Z2_EXPERIMENT
#include "../kernels/cluster_sync.cuh"
#include "../kernels/cluster_dedup.cuh"
#endif

struct Variant {
    const char *name;
    const void *kernel;
    dim3 grid, block;
    int shared, steps;
#ifdef LBM_Z2_EXPERIMENT
    int cluster_z; // 0 = ordinary launch, 2 = runtime Z2 cluster.
#endif
};
#define KERNEL(k) reinterpret_cast<const void *>(k)
#define TILE(x,y,z) dim3((SIZE_X+x-1)/x,(SIZE_Y+y-1)/y,(SIZE_Z+z-1)/z)
#define CONTROL(name,k,x,y,z,w) {name,KERNEL(k),TILE(x,y,z),dim3(w),0,1}
#define TEMPORAL(name,v,x,y,z,w,sites) \
    {name,KERNEL((temporal::v<x,y,z>)),TILE(x,y,z),dim3(w),19*(sites)*4,2}
static const Variant variants[] = {
    {"B0",KERNEL(b0),dim3(SIZE_Y,SIZE_Z),dim3(SIZE_X),0,1},
    {"C1",KERNEL(c1),TILE(32,4,1),dim3(32,4,1),0,1},
    {"C2",KERNEL(c2),TILE(32,2,2),dim3(32,2,2),0,1},
    {"C3",KERNEL(c3),TILE(64,2,1),dim3(64,2,1),0,1},
    CONTROL("S2-C128",s2_c128,8,8,8,128),
    CONTROL("S2-D128",s2_d128,16,8,4,128),
    CONTROL("S2-C256",s2_c128,8,8,8,256),
    CONTROL("S2-D256",s2_d128,16,8,4,256),
    TEMPORAL("C-V1-128",v1,8,8,8,128,1000),
    TEMPORAL("C-V1-256",v1,8,8,8,256,1000),
    TEMPORAL("C-V2-128",v2,8,8,8,128,512),
    TEMPORAL("C-V2-256",v2,8,8,8,256,512),
    TEMPORAL("D-V1-128",v1,16,8,4,128,1080),
    TEMPORAL("D-V1-256",v1,16,8,4,256,1080),
    TEMPORAL("D-V2-128",v2,16,8,4,128,512),
    TEMPORAL("D-V2-256",v2,16,8,4,256,512),
    CONTROL("T30-S1-128",t30_s1,30,4,4,128),
    CONTROL("T30-S1-256",t30_s1,30,4,4,256),
    TEMPORAL("T30-V2-128",v2,30,4,4,128,480),
    TEMPORAL("T30-V2-256",v2,30,4,4,256,480),
#ifdef LBM_Z2_EXPERIMENT
    // C0 reuses the exact D-V2 kernel entry: only its runtime launch differs.
    {"D-C0-Z2-256",KERNEL((temporal::v2<16,8,4>)),TILE(16,8,4),dim3(256),38912,2,2},
    {"D-C1-Z2-256",KERNEL(temporal::z2_sync),TILE(16,8,4),dim3(256),38912,2,2},
    {"D-V3-Z2-256",KERNEL(temporal::z2_dedup),TILE(16,8,4),dim3(256),38912,2,2},
#endif
};
#undef TEMPORAL
#undef CONTROL
#undef TILE
#undef KERNEL
