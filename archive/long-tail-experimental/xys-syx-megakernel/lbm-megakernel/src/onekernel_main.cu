// One kernel for the whole time loop (kernels/onekernel.cuh) against B0's one launch per step.
// Every variant keeps B0's launch shape and mapping except bigkern2_resident, the smallest change
// that is correct. Results are compared bit for bit with B0 after the same number of steps.
// Usage: lbm-onekernel EVEN_STEPS [-i OBSTACLES]
#include <algorithm>
#include <cstdint>
#include <vector>
#include "grid.cuh"
#include "../kernels/b0.cuh"
#include "../kernels/onekernel.cuh"

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
static const size_t bytes = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float)
                            + 2 * TOTAL_MARGIN * sizeof(float);
static const unsigned long long timeout_ns = 300ull * 1000 * 1000;
static const dim3 grid(SIZE_Y, SIZE_Z), block(SIZE_X);
static float *h, *d[2];
static std::vector<float> ref(bytes / 4), got(bytes / 4);

static void reset() {
    for (int i = 0; i < 2; ++i)
        CHECK(cudaMemcpy(d[i] - REAL_MARGIN, h - REAL_MARGIN, bytes, cudaMemcpyHostToDevice));
    unsigned int zero = 0; int izero = 0;
    CHECK(cudaMemcpyToSymbol(onekernel::count, &zero, sizeof zero));
    CHECK(cudaMemcpyToSymbol(onekernel::generation, &zero, sizeof zero));
    CHECK(cudaMemcpyToSymbol(onekernel::arrived_at_abort, &zero, sizeof zero));
    CHECK(cudaMemcpyToSymbol(onekernel::finished, &zero, sizeof zero));
    CHECK(cudaMemcpyToSymbol(onekernel::abort_flag, &izero, sizeof izero));
    CHECK(cudaDeviceSynchronize());
}

// B0 one launch per step; after an even number of steps the state is back in d[0].
static void b0_steps(int steps) {
    for (int s = 0; s < steps; ++s) b0<<<grid, block>>>(d[s & 1], d[1 - (s & 1)]);
}

// Bit mismatches of d[0] against the B0 reference, over all populations.
static size_t mismatches() {
    CHECK(cudaMemcpy(got.data(), d[0] - REAL_MARGIN, bytes, cudaMemcpyDeviceToHost));
    const float *g = got.data() + REAL_MARGIN, *r = ref.data() + REAL_MARGIN;
    size_t n = 0;
    for (int z = 0; z < SIZE_Z; ++z) for (int y = 0; y < SIZE_Y; ++y)
        for (int x = 0; x < SIZE_X; ++x) for (int q = 0; q < 19; ++q) {
            const int j = CALC_INDEX(x, y, z, q);
            uint32_t gb, rb;
            memcpy(&gb, g + j, 4); memcpy(&rb, r + j, 4);
            n += gb != rb;
        }
    return n;
}

static void result(const char *name, int launches) {
    const size_t n = mismatches();
    printf("RUN name=%s launches=%d result=%s bit_mismatches=%zu populations=%d mismatch_pct=%.1f\n",
           name, launches, n ? "WRONG" : "correct", n, TOTAL_CELLS * 19, 100.0 * n / (TOTAL_CELLS * 19));
}

int main(int argc, char **argv) {
    int steps = 0;
    const char *obstacle = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) obstacle = argv[++i];
        else steps = atoi(argv[i]);
    }
    if (steps <= 0 || steps % 2) {
        fprintf(stderr, "Usage: lbm-onekernel EVEN_STEPS [-i OBSTACLES]\n");
        return 1;
    }
    LBM_allocateGrid(&h);
    LBM_initializeGrid(h);
    if (obstacle) LBM_loadObstacleFile(h, obstacle);
    LBM_initializeSpecialCellsForLDC(h);
    for (int i = 0; i < 2; ++i) { CHECK(cudaMalloc(&d[i], bytes)); d[i] += REAL_MARGIN; }

    int device, sms;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto resident = [&](const void *k) {
        int n; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, SIZE_X, 0)); return n * sms;
    };
    const int fit_spin = resident((const void *)onekernel::bigkern2_spin);
    const int fit_sync = resident((const void *)onekernel::bigkern2_grid_sync);
    const int fit_resident = resident((const void *)onekernel::bigkern2_resident);
    const unsigned int blocks = grid.x * grid.y;
    printf("RESOURCE grid=%ux%u block=%u blocks=%u sms=%d resident_blocks bigkern2_grid_sync=%d "
           "bigkern2_spin=%d bigkern2_resident=%d\n", grid.x, grid.y, block.x, blocks, sms, fit_sync,
           fit_spin, fit_resident);

    reset(); b0_steps(steps); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(ref.data(), d[0] - REAL_MARGIN, bytes, cudaMemcpyDeviceToHost));

    reset();
    for (int i = 0; i < steps / 2; ++i) onekernel::bigkern1<<<grid, block>>>(d[0], d[1]);
    CHECK(cudaDeviceSynchronize());
    result("bigkern1", steps / 2);

    reset();
    onekernel::bigkern2<<<grid, block>>>(d[0], d[1], steps);
    CHECK(cudaDeviceSynchronize());
    result("bigkern2", 1);

    reset();
    {
        void *args[] = {&d[0], &d[1], &steps};
        cudaError_t e = cudaLaunchCooperativeKernel((const void *)onekernel::bigkern2_grid_sync, grid,
                                                    block, args, 0, 0);
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        cudaGetLastError();
        printf("RUN name=bigkern2_grid_sync launch=cooperative blocks=%u status=%s error=\"%s\"\n", blocks,
               e == cudaSuccess ? "ran" : "refused", cudaGetErrorString(e));
        if (e == cudaSuccess) result("bigkern2_grid_sync", 1);
    }

    reset();
    {
        cudaEvent_t start, end;
        CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
        onekernel::bigkern2_spin<<<grid, block>>>(d[0], d[1], steps, timeout_ns);
        CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, end));
        int aborted; unsigned int arrived;
        CHECK(cudaMemcpyFromSymbol(&aborted, onekernel::abort_flag, sizeof aborted));
        CHECK(cudaMemcpyFromSymbol(&arrived, onekernel::arrived_at_abort, sizeof arrived));
        if (aborted)
            printf("RUN name=bigkern2_spin launch=ordinary status=deadlock arrived_at_first_barrier=%u "
                   "blocks=%u timeout_ms=%.0f elapsed_ms=%.3f\n", arrived, blocks, timeout_ns / 1e6, ms);
        else
            result("bigkern2_spin", 1);
    }

    reset();
    {
        const int barriers = 1000;
        onekernel::barrier_only<<<fit_spin, SIZE_X>>>(barriers, timeout_ns);
        CHECK(cudaDeviceSynchronize());
        int aborted; unsigned int finished;
        CHECK(cudaMemcpyFromSymbol(&aborted, onekernel::abort_flag, sizeof aborted));
        CHECK(cudaMemcpyFromSymbol(&finished, onekernel::finished, sizeof finished));
        printf("CHECK name=barrier_only blocks=%d barriers=%d finished_blocks=%u status=%s\n", fit_spin,
               barriers, finished, aborted ? "failed" : "works");
    }

    // bigkern2_resident: correctness once, then timed against B0, interleaved, 5 warm-ups + 20 trials.
    void *args[] = {&d[0], &d[1], &steps};
    reset();
    CHECK(cudaLaunchCooperativeKernel((const void *)onekernel::bigkern2_resident, dim3(fit_resident),
                                      block, args, 0, 0));
    CHECK(cudaDeviceSynchronize());
    char name[64];
    snprintf(name, sizeof name, "bigkern2_resident(grid=%d)", fit_resident);
    result(name, 1);
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&end));
    std::vector<double> t_b0, t_res;
    for (int trial = -5; trial < 20; ++trial) {
        for (int arm = 0; arm < 2; ++arm) {
            reset();
            CHECK(cudaEventRecord(start));
            if (arm == 0) b0_steps(steps);
            else CHECK(cudaLaunchCooperativeKernel((const void *)onekernel::bigkern2_resident,
                                                   dim3(fit_resident), block, args, 0, 0));
            CHECK(cudaEventRecord(end)); CHECK(cudaEventSynchronize(end));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, end));
            if (trial >= 0) (arm ? t_res : t_b0).push_back(ms * 1000.0 / steps);
        }
    }
    std::sort(t_b0.begin(), t_b0.end()); std::sort(t_res.begin(), t_res.end());
    const double m_b0 = (t_b0[9] + t_b0[10]) / 2, m_res = (t_res[9] + t_res[10]) / 2;
    printf("BENCH name=B0 steps=%d launches=%d median_us=%.3f\n", steps, steps, m_b0);
    printf("BENCH name=%s steps=%d launches=1 median_us=%.3f speedup_over_B0=%.3f\n", name, steps, m_res,
           m_b0 / m_res);

    for (int i = 0; i < 2; ++i) CHECK(cudaFree(d[i] - REAL_MARGIN));
    LBM_freeGrid(&h);
    return 0;
}
