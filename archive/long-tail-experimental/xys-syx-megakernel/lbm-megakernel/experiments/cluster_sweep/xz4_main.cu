#include "../../src/grid.cuh"
#include "xz4.cuh"
#define SWEEP_EXTRA_ROWS \
 SWEEP_ROW("D-C1-XZ4-LoadBalancing",temporal::xz4_sync,2,1,2,cudaClusterSchedulingPolicyLoadBalancing), \
 SWEEP_ROW("D-V3-XZ4-LoadBalancing",temporal::xz4_dedup,2,1,2,cudaClusterSchedulingPolicyLoadBalancing),
#include "variants.cuh"
#include "../../build/sweep/driver.cuh"
