#define LBM_Z2_DEBUG 1
#define main reused_driver_main
#include "cluster_main.cu"
#undef main

int main(int argc,char **argv) {
    for(const auto &v:variants) if(!strcmp(v.name,"D-V3-Z2-256")) selected=&v;
    setup_kernel();
    const bool bad_threads=argc==2&&!strcmp(argv[1],"--bad-threads");
    const bool bad_cluster=argc==2&&!strcmp(argv[1],"--bad-cluster");
    if(bad_threads||bad_cluster) {
        cudaLaunchAttribute attr{}; auto cfg=cluster_config(*selected,attr);
        if(bad_threads) cfg.blockDim.x=192;
        if(bad_cluster) attr.val.clusterDim.z=1;
        const float *src=nullptr; float *dst=nullptr;
        CHECK(cudaLaunchKernelEx(&cfg,temporal::z2_dedup,src,dst));
        cudaError_t e=cudaDeviceSynchronize();
        fprintf(stderr,"EXPECTED_ASSERT status=%s\n",cudaGetErrorString(e));
        return e==cudaErrorAssert?1:2;
    }
    if(argc!=2) {fprintf(stderr,"Usage: z2-ownership OBSTACLE_FILE | --bad-threads | --bad-cluster\n");return 1;}
    float *h; LBM_allocateGrid(&h);
    for(size_t j=0;j<bytes/4;++j) (h-REAL_MARGIN)[j]=0.01f+float((j*17)%251)*0.00001f;
    initialize(h,true,argv[1]);
    const int dx[19]={0,0,0,1,-1,0,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1};
    const int dy[19]={0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0};
    const int dz[19]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
    for(int z=0;z<SIZE_Z;++z) for(int y=0;y<SIZE_Y;++y) for(int x=0;x<SIZE_X;++x)
        for(int q=0;q<19;++q) {
            const int pred=(x-dx[q])+PADDED_X*((y-dy[q])+PADDED_Y*(z-dz[q]));
            if(!(pred>=0&&pred<PADDED_X*PADDED_Y*SIZE_Z&&pred%PADDED_X<SIZE_X))
                h[CALC_INDEX(x,y,z,q)]*=1.0f+float((x*7+y*13+z*19+q*23)%31)*0.001f;
        }
    std::vector<float> initial(h-REAL_MARGIN,h-REAL_MARGIN+bytes/4),got(bytes/4),ref(bytes/4);
    float *d[2],*oracle[2];
    for(int i=0;i<2;++i) {
        CHECK(cudaMalloc(&d[i],bytes));d[i]+=REAL_MARGIN;
        CHECK(cudaMalloc(&oracle[i],bytes));oracle[i]+=REAL_MARGIN;
        CHECK(cudaMemcpy(d[i]-REAL_MARGIN,initial.data(),bytes,cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(oracle[i]-REAL_MARGIN,initial.data(),bytes,cudaMemcpyHostToDevice));
    }
    unsigned *writers,*cores,*producers;
    CHECK(cudaMalloc(&writers,19*TOTAL_CELLS*sizeof(unsigned)));
    CHECK(cudaMalloc(&cores,TOTAL_CELLS*sizeof(unsigned)));
    CHECK(cudaMalloc(&producers,2280*1800*sizeof(unsigned)));
    CHECK(cudaMemset(writers,0,19*TOTAL_CELLS*sizeof(unsigned)));
    CHECK(cudaMemset(cores,0,TOTAL_CELLS*sizeof(unsigned)));
    CHECK(cudaMemset(producers,0,2280*1800*sizeof(unsigned)));
    CHECK(cudaMemcpyToSymbol(z2_debug::writers,&writers,sizeof(writers)));
    CHECK(cudaMemcpyToSymbol(z2_debug::cores,&cores,sizeof(cores)));
    CHECK(cudaMemcpyToSymbol(z2_debug::producers,&producers,sizeof(producers)));
    unsigned long long zero[z2_debug::COUNT]={};
    CHECK(cudaMemcpyToSymbol(z2_debug::counts,zero,sizeof(zero)));
    pair(d[0],d[1]); kern(oracle[0],oracle[1]);kern(oracle[1],oracle[0]);
    CHECK(cudaDeviceSynchronize());
    read_state(got,d[1]);read_state(ref,oracle[0]);
    bool ok=compare(got,ref,2,"cluster-current");
    read_state(got,d[0]);ok &= compare(got,initial,0,"cluster-input-unchanged");
    unsigned long long counters[z2_debug::COUNT];
    CHECK(cudaMemcpyFromSymbol(counters,z2_debug::counts,sizeof(counters)));
    unsigned *arrays[]={writers,cores,producers};
    const size_t sizes[]={19*TOTAL_CELLS,TOTAL_CELLS,2280*1800};
    size_t wrong[3]={};
    for(int a=0;a<3;++a) {
        std::vector<unsigned> counts(sizes[a]);
        CHECK(cudaMemcpy(counts.data(),arrays[a],counts.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
        for(auto n:counts) wrong[a]+=n!=1;
        ok &= wrong[a]==0;
    }
    printf("OWNERSHIP_JSON {\"passed\":%s,\"wrong_writers\":%zu,\"wrong_cores\":%zu,\"wrong_producers\":%zu,\"counts\":[",
           ok?"true":"false",wrong[0],wrong[1],wrong[2]);
    for(int i=0;i<z2_debug::COUNT;++i) printf("%s%llu",i?",":"",counters[i]);
    puts("]}");
    for(auto p:arrays) CHECK(cudaFree(p));
    for(int i=0;i<2;++i) { CHECK(cudaFree(d[i]-REAL_MARGIN));CHECK(cudaFree(oracle[i]-REAL_MARGIN)); }
    LBM_freeGrid(&h);
    return ok?0:1;
}
