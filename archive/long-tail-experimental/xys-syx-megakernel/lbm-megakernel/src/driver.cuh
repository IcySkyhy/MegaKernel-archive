#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
static const size_t bytes = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float)
                            + 2 * TOTAL_MARGIN * sizeof(float);

static void initialize(float *h, bool patterned, const char *obstacle) {
    LBM_initializeGrid(h);
    if (obstacle) LBM_loadObstacleFile(h, obstacle);
    LBM_initializeSpecialCellsForLDC(h);
    if (!patterned) return;
    // Nonuniform populations expose wrong direction/halo selection even at step 2.
    // Extra obstacles cross internal tile faces and the partial Z end tiles.
    for (int z=0; z<SIZE_Z; ++z) for (int y=0; y<SIZE_Y; ++y)
        for (int x=0; x<SIZE_X; ++x) {
            for (int q=0; q<19; ++q) {
                int j = CALC_INDEX(x,y,z,q);
                int k = (x*17 + y*31 + z*43 + q*13) % 31 - 15;
                h[j] *= 1.0f + float(k) * 0.0001f;
            }
            if ((x*3 + y*5 + z*7) % 29 == 0)
                SET_FLAG(h,x,y,z,OBSTACLE);
        }
}

static const Variant *selected = nullptr;

#ifdef LBM_Z2_EXPERIMENT
static cudaLaunchConfig_t cluster_config(const Variant &v,cudaLaunchAttribute &attr) {
    cudaLaunchConfig_t config{};
    config.gridDim=v.grid; config.blockDim=v.block; config.dynamicSmemBytes=v.shared;
    if(v.cluster_z) {
        attr.id=cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x=1; attr.val.clusterDim.y=1; attr.val.clusterDim.z=v.cluster_z;
        config.attrs=&attr; config.numAttrs=1;
    }
    return config;
}
#endif

static void setup_kernel() {
    const Variant &v = *selected;
    int device, limit, max_threads, blocks;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    CHECK(cudaDeviceGetAttribute(&max_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device));
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr,v.kernel));
    if (v.shared + attr.sharedSizeBytes > size_t(limit)) {
        fprintf(stderr,"%s exceeds shared memory capacity (%d bytes)\n",v.name,limit);
        exit(1);
    }
    if (v.shared > 48*1024)
        CHECK(cudaFuncSetAttribute(v.kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,v.shared));
    int threads=v.block.x*v.block.y*v.block.z;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,v.kernel,threads,v.shared));
    printf("RESOURCE name=%s threads=%d registers=%d static_shared=%zu dynamic_shared=%d "
           "local_bytes=%zu active_blocks_per_sm=%d grid=%ux%ux%u block=%ux%ux%u "
           "steps_per_launch=%d theoretical_occupancy_pct=%.6f\n",
           v.name,threads,attr.numRegs,attr.sharedSizeBytes,v.shared,attr.localSizeBytes,
           blocks,v.grid.x,v.grid.y,v.grid.z,v.block.x,v.block.y,v.block.z,v.steps,
           100.0*blocks*((threads+31)/32)*32/max_threads);
#ifdef LBM_Z2_EXPERIMENT
    int supported=0,active_clusters=0,potential_size=0;
    CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,device));
    if(!supported) { fprintf(stderr,"Device does not support cluster launch\n"); exit(1); }
    if(v.cluster_z && (v.grid.z%v.cluster_z || threads!=256 || v.block.y!=1 || v.block.z!=1)) {
        fprintf(stderr,"Invalid Z2 launch geometry\n"); exit(1);
    }
    cudaLaunchAttribute launch_attr{};
    auto config=cluster_config(v,launch_attr);
    // Query singleton clusters for the ordinary reference; the actual launch
    // still has no cluster attribute. The occupancy API needs explicit sizing.
    if(!v.cluster_z) {
        launch_attr.id=cudaLaunchAttributeClusterDimension;
        launch_attr.val.clusterDim.x=1;
        launch_attr.val.clusterDim.y=1;
        launch_attr.val.clusterDim.z=1;
        config.attrs=&launch_attr; config.numAttrs=1;
    }
    CHECK(cudaOccupancyMaxActiveClusters(&active_clusters,v.kernel,&config));
    CHECK(cudaOccupancyMaxPotentialClusterSize(&potential_size,v.kernel,&config));
    if(active_clusters<=0) { fprintf(stderr,"No active clusters for this launch\n"); exit(1); }
    printf("CLUSTER runtime_cluster=%d dim=1x1x%d active_clusters_device=%d potential_cluster_size=%d\n",
           v.cluster_z!=0,v.cluster_z?v.cluster_z:1,active_clusters,potential_size);
#endif
}

static void launch(const Variant &v, float *src, float *dst) {
    void *args[] = {&src,&dst};
#ifdef LBM_Z2_EXPERIMENT
    // Same extended launch API for all experiment rows. The unclustered
    // D-V2 reference has zero attributes; C0/C1/C2 specify runtime Z2.
    cudaLaunchAttribute attr{};
    auto config=cluster_config(v,attr);
    CHECK(cudaLaunchKernelExC(&config,v.kernel,args));
#else
    CHECK(cudaLaunchKernel(v.kernel,v.grid,v.block,args,v.shared,0));
#endif
}
static void pair(float *src, float *dst) { launch(*selected,src,dst); }
static void kern(float *src, float *dst) { launch(variants[0],src,dst); }

// Timing uses the same initialized grids and event protocol for all controls.
static void benchmark(const char *mode, int steps, float **d) {
    const bool temporal=selected->steps==2;
    float *snap[2];
    for (int i=0;i<2;++i) {
        CHECK(cudaMalloc(&snap[i],bytes));
        CHECK(cudaMemcpy(snap[i],d[i]-REAL_MARGIN,bytes,cudaMemcpyDeviceToDevice));
    }
    cudaEvent_t start,end;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    std::vector<double> times;
    for (int trial=-5;trial<20;++trial) {
        for (int i=0;i<2;++i)
            CHECK(cudaMemcpy(d[i]-REAL_MARGIN,snap[i],bytes,cudaMemcpyDeviceToDevice));
        int current=0;
        CHECK(cudaEventRecord(start));
        for (int launch=0;launch<(temporal?steps/2:steps);++launch) {
            pair(d[current],d[1-current]);
            current=1-current;
        }
        CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
        float ms; CHECK(cudaEventElapsedTime(&ms,start,end));
        if (trial>=0) {
            times.push_back(ms*1000.0/steps);
            printf("TRIAL %d us_per_step=%.9f\n",trial,times.back());
        }
    }
    std::sort(times.begin(),times.end());
    double median=(times[9]+times[10])/2;
    printf("BENCH mode=%s warmups=5 trials=20 steps=%d launches=%d median_us=%.9f "
           "min_us=%.9f p90_us=%.9f mlups=%.9f equivalent_population_TB_s=%.9f\n",
           selected->name,steps,temporal?steps/2:steps,median,times.front(),times[17],
           TOTAL_CELLS/median,(TOTAL_CELLS*152.0/1e6)/median);
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(end));
    for (int i=0;i<2;++i) CHECK(cudaFree(snap[i]));
}

static void read_state(std::vector<float> &host, const float *device) {
    CHECK(cudaMemcpy(host.data(), device-REAL_MARGIN, bytes, cudaMemcpyDeviceToHost));
}

static bool compare(const std::vector<float> &got_base, const std::vector<float> &ref_base,
                    int step, const char *role) {
    const float *got = got_base.data()+REAL_MARGIN, *ref = ref_base.data()+REAL_MARGIN;
    size_t mismatches=0, flag_mismatches=0, nonfinite=0, near_zero=0;
    double max_abs=0, sum_sq=0, max_rel=0, near_abs=0;
    for (int z=0; z<SIZE_Z; ++z) for (int y=0; y<SIZE_Y; ++y)
        for (int x=0; x<SIZE_X; ++x) {
            for (int q=0; q<19; ++q) {
                int j = CALC_INDEX(x,y,z,q);
                uint32_t gb, rb;
                memcpy(&gb,got+j,4); memcpy(&rb,ref+j,4);
                mismatches += gb != rb;
                if ((gb & 0x7f800000u) == 0x7f800000u ||
                    (rb & 0x7f800000u) == 0x7f800000u) { ++nonfinite; continue; }
                double err = fabs(double(got[j])-double(ref[j]));
                max_abs=std::max(max_abs,err); sum_sq+=err*err;
                if (fabs(double(ref[j])) >= 1e-12)
                    max_rel=std::max(max_rel,err/fabs(double(ref[j])));
                else { ++near_zero; near_abs=std::max(near_abs,err); }
            }
            int j = CALC_INDEX(x,y,z,FLAGS);
            flag_mismatches += *reinterpret_cast<const unsigned char*>(got+j) !=
                               *reinterpret_cast<const unsigned char*>(ref+j);
        }
    bool full = memcmp(got_base.data(),ref_base.data(),bytes)==0;
    printf("CHECK step=%d role=%s populations=%d bit_mismatches=%zu flag_byte_mismatches=%zu "
           "nonfinite=%zu full_allocation_equal=%d max_abs=%.17g rms=%.17g max_rel=%.17g "
           "relative_epsilon=1e-12 near_zero_count=%zu near_zero_max_abs=%.17g\n",
           step,role,TOTAL_CELLS*19,mismatches,flag_mismatches,nonfinite,full,max_abs,
           sqrt(sum_sq/(TOTAL_CELLS*19)),max_rel,near_zero,near_abs);
    if (!full) {
        const unsigned char *g = reinterpret_cast<const unsigned char*>(got_base.data());
        const unsigned char *r = reinterpret_cast<const unsigned char*>(ref_base.data());
        for (size_t i=0;i<bytes;++i) if (g[i]!=r[i]) {
            fprintf(stderr,"First differing allocation byte: %zu (got=%u ref=%u)\n",i,g[i],r[i]);
            break;
        }
    }
    return !mismatches && !flag_mismatches && !nonfinite && full;
}

static void usage() {
    fprintf(stderr,"Usage: lbm VARIANT [--check | --bench run] EVEN_STEPS [--patterned] [--sentinels] [-i OBSTACLES] [-o VELOCITY]\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--list")) {
        for (const auto &v: variants) printf("%s\n",v.name);
        return 0;
    }
    if (argc<3) usage();
    for (const auto &v: variants) if (!strcmp(argv[1],v.name)) selected=&v;
    if (!selected) usage();
    bool check=false, patterned=false, sentinels=false;
    int steps=0;
    const char *output=nullptr, *obstacle=nullptr, *bench=nullptr;
    for (int i=2;i<argc;++i) {
        if (!strcmp(argv[i],"--check")) check=true;
        else if (!strcmp(argv[i],"--bench") && i+1<argc) bench=argv[++i];
        else if (!strcmp(argv[i],"--patterned")) patterned=true;
        else if (!strcmp(argv[i],"--sentinels")) sentinels=true;
        else if (!strcmp(argv[i],"-o") && i+1<argc) output=argv[++i];
        else if (!strcmp(argv[i],"-i") && i+1<argc) obstacle=argv[++i];
        else {
            char *end;
            errno=0;
            long n=strtol(argv[i],&end,10);
            if (steps || errno || *end || n<=0 || n>2147483647L || n%2) usage();
            steps=int(n);
        }
    }
    if (!steps) usage();
    if (bench && (check || patterned || sentinels || obstacle || output ||
        strcmp(bench,"run"))) usage();
    if (obstacle) {
        FILE *f=fopen(obstacle,"rb");
        if (!f) { perror(obstacle); return 1; }
        if (fseek(f,0,SEEK_END) || ftell(f)!=SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z) {
            fprintf(stderr,"Invalid obstacle file length\n"); fclose(f); return 1;
        }
        fclose(f);
    }
    setup_kernel();
    float *h;
    LBM_allocateGrid(&h);
    if (sentinels) for (size_t j=0;j<bytes/4;++j)
        (h-REAL_MARGIN)[j]=0.01f+float((j*17)%251)*0.00001f;
    initialize(h,patterned,obstacle);
    if (sentinels) {
        // Independently enumerate baseline predecessors, including flat Y aliases.
        const int dx[19]={0,0,0,1,-1,0,0,1,-1,1,-1,0,0,0,0,1,1,-1,-1};
        const int dy[19]={0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,0,0,0,0};
        const int dz[19]={0,0,0,0,0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1};
        size_t invariant=0, aliased=0;
        for (int z=0;z<SIZE_Z;++z) for (int y=0;y<SIZE_Y;++y)
            for (int x=0;x<SIZE_X;++x) for (int q=0;q<19;++q) {
                const int pred=(x-dx[q])+PADDED_X*((y-dy[q])+PADDED_Y*(z-dz[q]));
                const bool valid=pred>=0 && pred<PADDED_X*PADDED_Y*SIZE_Z && pred%PADDED_X<SIZE_X;
                if (!valid) {
                    h[CALC_INDEX(x,y,z,q)]*=1.0f+float((x*7+y*13+z*19+q*23)%31)*0.001f;
                    ++invariant;
                } else if (y-dy[q]<0 || y-dy[q]>=SIZE_Y) ++aliased;
            }
        printf("SENTINELS invariant_slots=%zu y_alias_slots=%zu\n",invariant,aliased);
        if (invariant!=324236 || aliased!=178204) return 1;
    }
    float *d[2], *oracle[2]={nullptr,nullptr};
    for (int i=0;i<2;++i) {
        CHECK(cudaMalloc(&d[i],bytes)); d[i]+=REAL_MARGIN;
        CHECK(cudaMemcpy(d[i]-REAL_MARGIN,h-REAL_MARGIN,bytes,cudaMemcpyHostToDevice));
        if (check) {
            CHECK(cudaMalloc(&oracle[i],bytes)); oracle[i]+=REAL_MARGIN;
            CHECK(cudaMemcpy(oracle[i]-REAL_MARGIN,h-REAL_MARGIN,bytes,cudaMemcpyHostToDevice));
        }
    }
    if (bench) {
        benchmark(bench,steps,d);
        for (int i=0;i<2;++i) CHECK(cudaFree(d[i]-REAL_MARGIN));
        LBM_freeGrid(&h);
        return 0;
    }
    std::vector<float> got, ref, previous;
    if (check) { got.resize(bytes/4); ref.resize(bytes/4); previous.resize(bytes/4); }
    int current=0;
    bool ok=true;
    for (int launch=0;launch<steps/2;++launch) {
        const int step=2*(launch+1);
        const bool checkpoint=check && (step<=6 || step==steps);
        if (checkpoint) read_state(previous,oracle[0]);
        pair(d[current],d[1-current]);
        current=1-current;
        if(selected->steps==1) { pair(d[current],d[1-current]); current=1-current; }
        if (check) {
            // Unchanged B0 push kernels; oracle[0] is the even state, [1] odd.
            kern(oracle[0],oracle[1]); kern(oracle[1],oracle[0]);
            CHECK(cudaGetLastError());
        }
        if (checkpoint) {
            read_state(got,d[current]); read_state(ref,oracle[0]);
            ok &= compare(got,ref,step,"current-even");
            read_state(got,d[1-current]);
            if(selected->steps==1) read_state(previous,oracle[1]);
            ok &= compare(got,previous,step-selected->steps,"inactive");
            if (!ok) break;
        }
    }
    CHECK(cudaDeviceSynchronize());
    if (ok) {
        CHECK(cudaMemcpy(h-REAL_MARGIN,d[current]-REAL_MARGIN,bytes,cudaMemcpyDeviceToHost));
        LBM_showGridStatistics(h);
        if (output) LBM_storeVelocityField(h,output,TRUE);
        printf("%s physical_steps=%d kernel_launches=%d final_buffer=%c patterned=%d\n",
               check?"PASS":"DONE",steps,steps/selected->steps,'A'+current,patterned);
    }
    for (int i=0;i<2;++i) {
        CHECK(cudaFree(d[i]-REAL_MARGIN));
        if (check) CHECK(cudaFree(oracle[i]-REAL_MARGIN));
    }
    LBM_freeGrid(&h);
    return ok?0:1;
}
