#pragma once
#include "../../kernels/b0.cuh"
#include "../../kernels/temporal_v2.cuh"
#include "../../kernels/cluster_sync.cuh"
#include "../../kernels/cluster_dedup.cuh"
struct ClusterShape { unsigned x,y,z; };
struct Variant {
    const char *name;
    const void *kernel;
    dim3 grid,block;
    int shared,steps;
    ClusterShape cluster;
    cudaClusterSchedulingPolicy policy;
};
#define SWEEP_KERNEL(k) reinterpret_cast<const void *>(k)
#define SWEEP_ROW(name,k,x,y,z,p) {name,SWEEP_KERNEL(k),dim3(8,15,38),dim3(256),38912,2,{x,y,z},p}
#define SWEEP_C0(shape,x,y,z) \
 SWEEP_ROW("D-C0-" shape "-Spread",(temporal::v2<16,8,4>),x,y,z,cudaClusterSchedulingPolicySpread), \
 SWEEP_ROW("D-C0-" shape "-LoadBalancing",(temporal::v2<16,8,4>),x,y,z,cudaClusterSchedulingPolicyLoadBalancing)
#ifndef SWEEP_EXTRA_ROWS
#define SWEEP_EXTRA_ROWS
#endif
static const Variant variants[]={
    {"B0",SWEEP_KERNEL(b0),dim3(SIZE_Y,SIZE_Z),dim3(SIZE_X),0,1,{0,0,0},cudaClusterSchedulingPolicyDefault},
    SWEEP_ROW("D-V2-256",(temporal::v2<16,8,4>),0,0,0,cudaClusterSchedulingPolicyDefault),
    SWEEP_C0("Z2",1,1,2),
    SWEEP_C0("XZ4",2,1,2),
    SWEEP_C0("YZ6",1,3,2),
    SWEEP_ROW("D-C1-Z2-Spread",temporal::z2_sync,1,1,2,cudaClusterSchedulingPolicySpread),
    SWEEP_ROW("D-C1-Z2-LoadBalancing",temporal::z2_sync,1,1,2,cudaClusterSchedulingPolicyLoadBalancing),
    SWEEP_ROW("D-V3-Z2-Spread",temporal::z2_dedup,1,1,2,cudaClusterSchedulingPolicySpread),
    SWEEP_ROW("D-V3-Z2-LoadBalancing",temporal::z2_dedup,1,1,2,cudaClusterSchedulingPolicyLoadBalancing),
    SWEEP_EXTRA_ROWS
};
