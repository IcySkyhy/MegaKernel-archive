#pragma once
static const Variant *selected=nullptr;
static const char *policy_name(cudaClusterSchedulingPolicy p) {
    return p==cudaClusterSchedulingPolicySpread?"Spread":
           p==cudaClusterSchedulingPolicyLoadBalancing?"LoadBalancing":"None";
}
static cudaLaunchConfig_t cluster_config(const Variant &v,cudaLaunchAttribute (&attrs)[2]) {
    cudaLaunchConfig_t c{};
    c.gridDim=v.grid;c.blockDim=v.block;c.dynamicSmemBytes=v.shared;
    if(v.cluster.x) {
        attrs[0].id=cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x=v.cluster.x;
        attrs[0].val.clusterDim.y=v.cluster.y;
        attrs[0].val.clusterDim.z=v.cluster.z;
        attrs[1].id=cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[1].val.clusterSchedulingPolicyPreference=v.policy;
        c.attrs=attrs;c.numAttrs=2;
    }
    return c;
}
static void setup_kernel() {
    const Variant &v=*selected;
    int device,limit,max_threads,blocks,supported;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};CHECK(cudaGetDeviceProperties(&prop,device));
    printf("DEVICE name=%s cc=%d.%d sms=%d\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    CHECK(cudaDeviceGetAttribute(&max_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device));
    CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,device));
    if(!supported) {fprintf(stderr,"UNSUPPORTED device cluster launch\n");exit(77);}
    cudaFuncAttributes attr{};CHECK(cudaFuncGetAttributes(&attr,v.kernel));
    if(v.shared+attr.sharedSizeBytes>size_t(limit)) {fprintf(stderr,"Shared capacity exceeded\n");exit(1);}
    if(v.cluster.x && (!v.cluster.y||!v.cluster.z||v.grid.x%v.cluster.x||
       v.grid.y%v.cluster.y||v.grid.z%v.cluster.z||v.block.x!=256||v.block.y!=1||v.block.z!=1)) {
        fprintf(stderr,"Invalid cluster geometry\n");exit(1);
    }
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,v.kernel,v.block.x,v.shared));
    cudaLaunchAttribute attrs[2]{};auto c=cluster_config(v,attrs);
    // Query singleton cluster capacity only; ordinary launches have no attributes.
    if(!v.cluster.x) {
        attrs[0].id=cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x=attrs[0].val.clusterDim.y=attrs[0].val.clusterDim.z=1;
        c.attrs=attrs;c.numAttrs=1;
    }
    int active=0,potential=0;
    cudaError_t occupancy_status=cudaOccupancyMaxActiveClusters(&active,v.kernel,&c);
    if(occupancy_status==cudaErrorInvalidValue||occupancy_status==cudaErrorInvalidConfiguration||occupancy_status==cudaErrorNotSupported||
       (occupancy_status==cudaSuccess&&active==0)) {
        fprintf(stderr,"UNSUPPORTED name=%s cluster=%ux%ux%u policy=%s occupancy=%s active=%d\n",
                v.name,v.cluster.x,v.cluster.y,v.cluster.z,policy_name(v.policy),cudaGetErrorString(occupancy_status),active);
        exit(77);
    }
    CHECK(occupancy_status);CHECK(cudaOccupancyMaxPotentialClusterSize(&potential,v.kernel,&c));
    printf("RESOURCE name=%s threads=%u registers=%d static_shared=%zu dynamic_shared=%d local_bytes=%zu "
           "active_blocks_per_sm=%d grid=%ux%ux%u block=%ux%ux%u steps_per_launch=%d theoretical_occupancy_pct=%.6f\n",
           v.name,v.block.x,attr.numRegs,attr.sharedSizeBytes,v.shared,attr.localSizeBytes,blocks,
           v.grid.x,v.grid.y,v.grid.z,v.block.x,v.block.y,v.block.z,v.steps,
           100.0*blocks*((v.block.x+31)/32)*32/max_threads);
    printf("CLUSTER runtime_cluster=%d dim=%ux%ux%u size=%u policy=%s active_clusters_device=%d potential_cluster_size=%d\n",
           v.cluster.x!=0,v.cluster.x,v.cluster.y,v.cluster.z,v.cluster.x*v.cluster.y*v.cluster.z,
           policy_name(v.policy),active,potential);
}
static void launch(const Variant &v,float *src,float *dst) {
    void *args[]={&src,&dst};cudaLaunchAttribute attrs[2]{};
    auto c=cluster_config(v,attrs);
    CHECK(cudaLaunchKernelExC(&c,v.kernel,args));
}
