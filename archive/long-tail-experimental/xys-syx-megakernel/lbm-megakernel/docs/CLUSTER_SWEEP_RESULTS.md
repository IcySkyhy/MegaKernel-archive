# Cluster profitability sweep: scheduling policy, shape, then DSMEM

Fresh stages on RTX 5090; FP32 D3Q19, fixed 120×120×150, sm_120.
Each row uses 256 threads, a 16×8×4 per-block core, 38,912 B shared/block,
grid 8×15×38 and two physical timesteps/launch. Explicit policy preferences
are launch attributes. Ordinary D-V2 has no cluster attribute.

The [decision sequence](../experiments/cluster_sweep/DECISIONS.md) follows
the supplied guide. The [implementation notes](CLUSTER_SWEEP_DESIGN.md)
describe code reuse, correctness gates and reproduction. Older Z2 timings
are not pooled into these measurements.

**Outcome: stop this Tt=2 cluster tuning branch under the guide’s criterion.**

The best clustered finalist is D-V3-Z2-LoadBalancing, still 2.24% slower than ordinary D-V2 in the fresh Stage D campaign.
LoadBalancing materially improves C0, so the required Z2 C1/V3 rerun was
completed before the shape sweep. XZ4 had acceptable C0 cost and was then
implemented and fully validated, but its DSM variant is slower. Stage E
pointer hoisting was not entered; deeper temporal blocking remains future work.

| Stage D finalist | µs/physical step | Time / ordinary |
| --- | --- | --- |
| D-V2-256 | 136.569 | 1.00000 |
| D-V3-Z2-LoadBalancing | 139.632 | 1.02243 |
| D-V3-XZ4-LoadBalancing | 153.386 | 1.12314 |


## Prospective producer work — independent flat-address model

C0 still computes independent halos. The combined-box counts below describe
a possible deduplicated implementation, not work saved by C0 itself.

| Shape | Launch sweep? | Combined candidates | Candidate reduction | Valid producers | Valid reduction |
| --- | --- | --- | --- | --- | --- |
| Z2 | yes | 4,104,000 | 16.67% | 3,766,500 | 16.96% |
| X2 | model only | 4,651,200 | 5.56% | 4,267,200 | 5.93% |
| Y3 | model only | 4,268,160 | 13.33% | 3,931,200 | 13.33% |
| XZ4 | yes | 3,876,000 | 21.30% | 3,543,300 | 21.88% |
| YZ6 | yes | 3,556,800 | 27.78% | 3,264,300 | 28.04% |
| X4Z2 | model only | 3,762,000 | 23.61% | 3,431,700 | 24.35% |

Independent halos have 4,924,800 candidates and 4,536,000 valid first
collisions per fused launch. Phase 2 remains 2,160,000 collision sites.

## Stage A — Z2 scheduling policy

Event campaign started 2026-09-25T04:12:37.353181+00:00.
Four seeded shuffled 100-step rounds (80 trials/variant), two 1000-step
rounds (40 trials/variant), five warm-ups and twenty trials/process.
Both buffers restore outside event timing. Times include launch gaps;
allocation, validation, output and restores are excluded. Default LDC input.

| Variant | Cluster | Requested policy | Registers/thread | Shared B/block | Ordinary blocks/SM | Ordinary theoretical occupancy % | Max active clusters/device |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 0x0x0 | None | 56 | 38912 | 2 | 33.333333 | 340 |
| D-C0-Z2-Spread | 1x1x2 | Spread | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C0-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 170 |

The occupancy API capacity is device-wide. The ordinary row queries singleton
clusters only for that diagnostic; its actual launch remains unclustered.

### 100 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 135.723 | 135.507 | 135.813 | 15914.78 | 1.00000 | 135.721 / 135.733 / 135.733 / 135.721 |
| D-C0-Z2-Spread | 143.591 | 143.422 | 143.690 | 15042.77 | 1.05797 | 143.611 / 143.516 / 143.632 / 143.582 |
| D-C0-Z2-LoadBalancing | 137.484 | 137.279 | 137.586 | 15710.92 | 1.01298 | 137.464 / 137.489 / 137.517 / 137.392 |


### 1000 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 135.680 | 135.627 | 135.701 | 15919.79 | 1.00000 | 135.678 / 135.680 |
| D-C0-Z2-Spread | 143.785 | 143.547 | 143.864 | 15022.44 | 1.05973 | 143.576 / 143.841 |
| D-C0-Z2-LoadBalancing | 137.475 | 137.408 | 137.549 | 15711.98 | 1.01323 | 137.435 / 137.508 |


| 1000-step comparison (numerator / denominator) | Pooled ratio | Same-round ratios |
| --- | --- | --- |
| D-C0-Z2-Spread / D-V2-256 | 1.05973 | 1.05821 / 1.06015 |
| D-C0-Z2-LoadBalancing / D-V2-256 | 1.01323 | 1.01295 / 1.01347 |
| D-C0-Z2-LoadBalancing / D-C0-Z2-Spread | 0.95611 | 0.95723 / 0.95597 |

These are complete-kernel ablations, not additive phase timings. Process
rounds expose drift; twenty trials in one process are not independent
process replicates. Ratios and ranges are descriptive, not confidence intervals.

### NSYS and NCU diagnostics

NSYS checks 50 launches over 100 physical steps, grid/block/shared and output
against B0. Cluster dimensions/policy are verified independently through NCU
launch metadata. NSYS 2025.1.3 SQLite has no cluster-dimension columns.

| Variant | Kernels | Kernel sum ms | GPU span ms | Steady µs/step | Median gap µs | Median launch API µs |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 50 | 13.530 | 13.608 | 135.474 | 1.536 | 2.674 |
| D-C0-Z2-Spread | 50 | 14.387 | 14.470 | 144.098 | 1.824 | 2.709 |
| D-C0-Z2-LoadBalancing | 50 | 13.765 | 13.842 | 137.810 | 1.472 | 2.724 |

NCU 2025.2.1 kernel replay, base clocks, launch 11 (t=20→22), both cache
policies. Additive counters/duration divide by two; ratios, resource counts
and percentages do not. Event timing remains the performance authority.
No-flush is a replay diagnostic. All profiler outputs match B0.

| Variant | NCU policy field | Flush read MB/step | Flush write MB/step | Flush total MB/step | No-flush total MB/step | Flush µs/step | No-flush µs/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | PolicySpread | 91.703 | 67.394 | 159.097 | 177.539 | 139.312 | 144.432 |
| D-C0-Z2-Spread | PolicySpread | 91.544 | 67.266 | 158.810 | 177.211 | 148.176 | 154.080 |
| D-C0-Z2-LoadBalancing | PolicyLoadBalancing | 91.596 | 67.074 | 158.669 | 177.092 | 141.952 | 147.248 |


| Flush-replay metric | D-V2-256 | D-C0-Z2-Spread | D-C0-Z2-LoadBalancing |
| --- | --- | --- | --- |
| Registers/thread | 56.000 | 56.000 | 56.000 |
| Achieved occupancy % | 33.300 | 28.380 | 31.870 |
| Eligible warps/scheduler/active cycle | 0.550 | 0.510 | 0.540 |
| Issue active % | 33.400 | 32.150 | 33.160 |
| L2 hit % | 70.840 | 70.730 | 70.880 |
| L2 throughput % | 67.450 | 63.550 | 66.240 |
| SM throughput % | 32.460 | 30.440 | 31.840 |
| Global ld requests/step | 1618741.500 | 1618741.500 | 1618741.500 |
| Local ld requests/step | 0.000 | 0.000 | 0.000 |
| Global ld sectors/step | 11570283.000 | 11570283.000 | 11570283.000 |
| Local ld sectors/step | 0.000 | 0.000 | 0.000 |
| Shared ld wavefronts/step | 722559.500 | 724844.500 | 724850.000 |
| Shared ld conflicts/step | 39981.500 | 40706.000 | 42682.000 |
| Global st requests/step | 684000.000 | 684000.000 | 684000.000 |
| Local st requests/step | 0.000 | 0.000 | 0.000 |
| Global st sectors/step | 3285000.000 | 3285000.000 | 3285000.000 |
| Local st sectors/step | 0.000 | 0.000 | 0.000 |
| Shared st wavefronts/step | 997402.500 | 991490.000 | 993386.500 |
| Shared st conflicts/step | 79343.500 | 72587.500 | 75395.500 |
| Stall long_scoreboard per issue-active ratio | 3.870 | 3.370 | 3.910 |
| Stall short_scoreboard per issue-active ratio | 0.770 | 0.650 | 0.740 |
| Stall barrier per issue-active ratio | 0.790 | 0.720 | 0.770 |
| Stall lg_throttle per issue-active ratio | 1.360 | 1.110 | 1.300 |
| Stall mio_throttle per issue-active ratio | 0.750 | 0.580 | 0.690 |
| Stall wait per issue-active ratio | 1.660 | 1.660 | 1.660 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 |
| l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 |
| l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 |

Shared bank conflicts are access serialization, not races. Stall ratios are
not time percentages. DSM interface bytes are not logical float payload.
The SASS useful-bytes/sector heuristic is deliberately omitted: its mixed
global/DSM kernel needs numerator/denominator verification (the earlier Z2 C2
report could exceed 32 B/sector). Raw metric values remain in the CSV.

## Stage A2 — existing Z2 synchronization and DSM under both policies

Event campaign started 2026-09-25T04:19:22.634751+00:00.
Four seeded shuffled 100-step rounds (80 trials/variant), two 1000-step
rounds (40 trials/variant), five warm-ups and twenty trials/process.
Both buffers restore outside event timing. Times include launch gaps;
allocation, validation, output and restores are excluded. Default LDC input.

| Variant | Cluster | Requested policy | Registers/thread | Shared B/block | Ordinary blocks/SM | Ordinary theoretical occupancy % | Max active clusters/device |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 0x0x0 | None | 56 | 38912 | 2 | 33.333333 | 340 |
| D-C0-Z2-Spread | 1x1x2 | Spread | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C0-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C1-Z2-Spread | 1x1x2 | Spread | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C1-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 170 |
| D-V3-Z2-Spread | 1x1x2 | Spread | 64 | 38912 | 2 | 33.333333 | 170 |
| D-V3-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 64 | 38912 | 2 | 33.333333 | 170 |

The occupancy API capacity is device-wide. The ordinary row queries singleton
clusters only for that diagnostic; its actual launch remains unclustered.

### 100 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.566 | 136.447 | 136.662 | 15816.52 | 1.00000 | 136.590 / 136.551 / 136.574 / 136.540 |
| D-C0-Z2-Spread | 144.155 | 144.007 | 144.252 | 14983.83 | 1.05557 | 144.190 / 144.144 / 144.207 / 144.108 |
| D-C0-Z2-LoadBalancing | 137.889 | 137.708 | 137.979 | 15664.78 | 1.00969 | 137.877 / 137.909 / 137.836 / 137.900 |
| D-C1-Z2-Spread | 146.454 | 146.290 | 146.569 | 14748.63 | 1.07241 | 146.500 / 146.407 / 146.552 / 146.383 |
| D-C1-Z2-LoadBalancing | 142.368 | 142.163 | 142.440 | 15171.92 | 1.04249 | 142.337 / 142.388 / 142.382 / 142.368 |
| D-V3-Z2-Spread | 141.222 | 141.040 | 141.324 | 15295.09 | 1.03409 | 141.184 / 141.200 / 141.291 / 141.242 |
| D-V3-Z2-LoadBalancing | 139.551 | 139.356 | 139.655 | 15478.24 | 1.02185 | 139.467 / 139.628 / 139.516 / 139.598 |


### 1000 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.598 | 136.555 | 136.641 | 15812.80 | 1.00000 | 136.587 / 136.607 |
| D-C0-Z2-Spread | 144.149 | 144.065 | 144.251 | 14984.48 | 1.05528 | 144.120 / 144.229 |
| D-C0-Z2-LoadBalancing | 138.120 | 137.859 | 138.301 | 15638.63 | 1.01114 | 137.984 / 138.281 |
| D-C1-Z2-Spread | 146.603 | 146.423 | 146.946 | 14733.66 | 1.07324 | 146.463 / 146.828 |
| D-C1-Z2-LoadBalancing | 142.536 | 142.327 | 142.734 | 15154.03 | 1.04347 | 142.473 / 142.720 |
| D-V3-Z2-Spread | 141.251 | 141.212 | 141.328 | 15291.91 | 1.03406 | 141.250 / 141.258 |
| D-V3-Z2-LoadBalancing | 139.632 | 139.515 | 139.673 | 15469.21 | 1.02221 | 139.644 / 139.588 |


| 1000-step comparison (numerator / denominator) | Pooled ratio | Same-round ratios |
| --- | --- | --- |
| D-C0-Z2-Spread / D-V2-256 | 1.05528 | 1.05515 / 1.05579 |
| D-C0-Z2-LoadBalancing / D-V2-256 | 1.01114 | 1.01023 / 1.01225 |
| D-C1-Z2-Spread / D-V2-256 | 1.07324 | 1.07231 / 1.07482 |
| D-C1-Z2-LoadBalancing / D-V2-256 | 1.04347 | 1.04310 / 1.04475 |
| D-V3-Z2-Spread / D-V2-256 | 1.03406 | 1.03414 / 1.03405 |
| D-V3-Z2-LoadBalancing / D-V2-256 | 1.02221 | 1.02239 / 1.02182 |
| D-C0-Z2-LoadBalancing / D-C0-Z2-Spread | 0.95817 | 0.95743 / 0.95876 |
| D-C1-Z2-LoadBalancing / D-C1-Z2-Spread | 0.97226 | 0.97276 / 0.97202 |
| D-V3-Z2-LoadBalancing / D-V3-Z2-Spread | 0.98854 | 0.98863 / 0.98817 |
| D-C1-Z2-Spread / D-C0-Z2-Spread | 1.01702 | 1.01626 / 1.01802 |
| D-C1-Z2-LoadBalancing / D-C0-Z2-LoadBalancing | 1.03198 | 1.03253 / 1.03210 |
| D-V3-Z2-Spread / D-C1-Z2-Spread | 0.96349 | 0.96441 / 0.96206 |
| D-V3-Z2-LoadBalancing / D-C1-Z2-LoadBalancing | 0.97963 | 0.98014 / 0.97805 |

These are complete-kernel ablations, not additive phase timings. Process
rounds expose drift; twenty trials in one process are not independent
process replicates. Ratios and ranges are descriptive, not confidence intervals.

### NSYS and NCU diagnostics

NSYS checks 50 launches over 100 physical steps, grid/block/shared and output
against B0. Cluster dimensions/policy are verified independently through NCU
launch metadata. NSYS 2025.1.3 SQLite has no cluster-dimension columns.

| Variant | Kernels | Kernel sum ms | GPU span ms | Steady µs/step | Median gap µs | Median launch API µs |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 50 | 13.528 | 13.610 | 135.730 | 1.600 | 3.045 |
| D-C0-Z2-Spread | 50 | 14.362 | 14.444 | 143.618 | 1.760 | 3.320 |
| D-C0-Z2-LoadBalancing | 50 | 13.731 | 13.815 | 137.506 | 1.696 | 3.195 |
| D-C1-Z2-Spread | 50 | 14.571 | 14.639 | 145.954 | 1.216 | 2.955 |
| D-C1-Z2-LoadBalancing | 50 | 14.165 | 14.238 | 141.938 | 1.376 | 2.749 |
| D-V3-Z2-Spread | 50 | 14.036 | 14.112 | 140.482 | 1.568 | 2.795 |
| D-V3-Z2-LoadBalancing | 50 | 13.861 | 13.940 | 138.754 | 1.696 | 2.755 |

NCU 2025.2.1 kernel replay, base clocks, launch 11 (t=20→22), both cache
policies. Additive counters/duration divide by two; ratios, resource counts
and percentages do not. Event timing remains the performance authority.
No-flush is a replay diagnostic. All profiler outputs match B0.

| Variant | NCU policy field | Flush read MB/step | Flush write MB/step | Flush total MB/step | No-flush total MB/step | Flush µs/step | No-flush µs/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | PolicySpread | 91.714 | 67.175 | 158.889 | 176.868 | 139.120 | 143.600 |
| D-C0-Z2-Spread | PolicySpread | 91.567 | 67.355 | 158.922 | 177.131 | 147.824 | 154.064 |
| D-C0-Z2-LoadBalancing | PolicyLoadBalancing | 91.591 | 67.139 | 158.730 | 177.378 | 141.856 | 148.128 |
| D-C1-Z2-Spread | PolicySpread | 91.544 | 67.190 | 158.734 | 176.736 | 153.680 | 161.024 |
| D-C1-Z2-LoadBalancing | PolicyLoadBalancing | 91.571 | 66.902 | 158.472 | 176.922 | 150.048 | 154.592 |
| D-V3-Z2-Spread | PolicySpread | 91.512 | 66.808 | 158.320 | 175.736 | 148.736 | 153.536 |
| D-V3-Z2-LoadBalancing | PolicyLoadBalancing | 91.498 | 67.039 | 158.538 | 176.907 | 149.504 | 152.560 |


| Flush-replay metric | D-V2-256 | D-C0-Z2-Spread | D-C0-Z2-LoadBalancing | D-C1-Z2-Spread | D-C1-Z2-LoadBalancing | D-V3-Z2-Spread | D-V3-Z2-LoadBalancing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Registers/thread | 56.000 | 56.000 | 56.000 | 56.000 | 56.000 | 64.000 | 64.000 |
| Achieved occupancy % | 32.750 | 29.200 | 31.800 | 33.080 | 32.410 | 32.290 | 32.600 |
| Eligible warps/scheduler/active cycle | 0.550 | 0.510 | 0.530 | 0.480 | 0.490 | 0.520 | 0.540 |
| Issue active % | 33.250 | 32.570 | 32.710 | 30.530 | 30.980 | 33.010 | 33.410 |
| L2 hit % | 70.820 | 70.740 | 70.880 | 71.730 | 71.790 | 67.930 | 67.980 |
| L2 throughput % | 67.550 | 63.630 | 66.290 | 64.740 | 66.380 | 61.430 | 61.070 |
| SM throughput % | 32.490 | 30.150 | 31.880 | 29.230 | 30.330 | 31.890 | 33.050 |
| Global ld requests/step | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 | 1390569.500 | 1390569.500 |
| Local ld requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global ld sectors/step | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 | 9649337.500 | 9649337.500 |
| Local ld sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared ld wavefronts/step | 725073.500 | 723814.500 | 726411.000 | 717812.000 | 727006.000 | 725477.000 | 735853.500 |
| Shared ld conflicts/step | 39804.000 | 39697.000 | 41846.500 | 33558.000 | 41626.000 | 42057.000 | 50728.500 |
| Global st requests/step | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 |
| Local st requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global st sectors/step | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 |
| Local st sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared st wavefronts/step | 998699.000 | 991302.500 | 993972.500 | 991488.500 | 992176.000 | 906705.000 | 906621.500 |
| Shared st conflicts/step | 79502.000 | 72138.500 | 75067.000 | 72338.000 | 74308.500 | 49330.500 | 50518.000 |
| Stall long_scoreboard per issue-active ratio | 3.830 | 3.500 | 3.900 | 3.290 | 3.480 | 3.140 | 3.230 |
| Stall short_scoreboard per issue-active ratio | 0.770 | 0.650 | 0.760 | 0.690 | 0.710 | 0.420 | 0.420 |
| Stall barrier per issue-active ratio | 0.790 | 0.710 | 0.780 | 2.320 | 1.930 | 1.960 | 1.690 |
| Stall lg_throttle per issue-active ratio | 1.350 | 1.130 | 1.300 | 1.210 | 1.190 | 1.370 | 1.330 |
| Stall mio_throttle per issue-active ratio | 0.780 | 0.580 | 0.680 | 0.540 | 0.540 | 0.230 | 0.210 |
| Stall wait per issue-active ratio | 1.660 | 1.660 | 1.660 | 1.690 | 1.690 | 1.720 | 1.720 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 59992.500 | 59992.500 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 209475.000 | 209475.000 |
| l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7177440.000 | 7177440.000 |
| l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7177440.000 | 7177440.000 |

Shared bank conflicts are access serialization, not races. Stall ratios are
not time percentages. DSM interface bytes are not logical float payload.
The SASS useful-bytes/sector heuristic is deliberately omitted: its mixed
global/DSM kernel needs numerator/denominator verification (the earlier Z2 C2
report could exceed 32 B/sector). Raw metric values remain in the CSV.

## Stage B — C0 shape sweep

Event campaign started 2026-09-25T04:25:56.156247+00:00.
Four seeded shuffled 100-step rounds (80 trials/variant), two 1000-step
rounds (40 trials/variant), five warm-ups and twenty trials/process.
Both buffers restore outside event timing. Times include launch gaps;
allocation, validation, output and restores are excluded. Default LDC input.

| Variant | Cluster | Requested policy | Registers/thread | Shared B/block | Ordinary blocks/SM | Ordinary theoretical occupancy % | Max active clusters/device |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 0x0x0 | None | 56 | 38912 | 2 | 33.333333 | 340 |
| D-C0-Z2-Spread | 1x1x2 | Spread | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C0-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 170 |
| D-C0-XZ4-Spread | 2x1x2 | Spread | 56 | 38912 | 2 | 33.333333 | 85 |
| D-C0-XZ4-LoadBalancing | 2x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 85 |
| D-C0-YZ6-Spread | 1x3x2 | Spread | 56 | 38912 | 2 | 33.333333 | 52 |
| D-C0-YZ6-LoadBalancing | 1x3x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 52 |

The occupancy API capacity is device-wide. The ordinary row queries singleton
clusters only for that diagnostic; its actual launch remains unclustered.

### 100 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.561 | 136.429 | 136.634 | 15817.08 | 1.00000 | 136.561 / 136.576 / 136.575 / 136.535 |
| D-C0-Z2-Spread | 144.189 | 143.974 | 144.275 | 14980.34 | 1.05586 | 144.108 / 144.184 / 144.190 / 144.220 |
| D-C0-Z2-LoadBalancing | 137.892 | 137.719 | 137.986 | 15664.45 | 1.00974 | 137.831 / 137.887 / 137.914 / 137.924 |
| D-C0-XZ4-Spread | 140.366 | 140.074 | 140.472 | 15388.35 | 1.02786 | 140.268 / 140.382 / 140.419 / 140.410 |
| D-C0-XZ4-LoadBalancing | 138.348 | 138.221 | 138.454 | 15612.75 | 1.01309 | 138.334 / 138.331 / 138.370 / 138.417 |
| D-C0-YZ6-Spread | 142.137 | 141.958 | 142.248 | 15196.61 | 1.04083 | 142.091 / 142.177 / 142.101 / 142.182 |
| D-C0-YZ6-LoadBalancing | 139.971 | 139.777 | 140.073 | 15431.73 | 1.02497 | 139.976 / 139.980 / 139.996 / 139.899 |


### 1000 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.574 | 136.517 | 136.617 | 15815.59 | 1.00000 | 136.563 / 136.593 |
| D-C0-Z2-Spread | 144.212 | 144.039 | 144.238 | 14977.90 | 1.05593 | 144.221 / 144.181 |
| D-C0-Z2-LoadBalancing | 138.092 | 137.826 | 138.279 | 15641.79 | 1.01111 | 137.940 / 138.261 |
| D-C0-XZ4-Spread | 140.646 | 140.419 | 140.739 | 15357.65 | 1.02982 | 140.467 / 140.726 |
| D-C0-XZ4-LoadBalancing | 138.629 | 138.367 | 138.743 | 15581.18 | 1.01504 | 138.541 / 138.708 |
| D-C0-YZ6-Spread | 142.188 | 142.099 | 142.461 | 15191.14 | 1.04111 | 142.125 / 142.449 |
| D-C0-YZ6-LoadBalancing | 140.111 | 139.899 | 140.295 | 15416.30 | 1.02590 | 139.982 / 140.275 |


| 1000-step comparison (numerator / denominator) | Pooled ratio | Same-round ratios |
| --- | --- | --- |
| D-C0-Z2-Spread / D-V2-256 | 1.05593 | 1.05607 / 1.05555 |
| D-C0-Z2-LoadBalancing / D-V2-256 | 1.01111 | 1.01008 / 1.01221 |
| D-C0-XZ4-Spread / D-V2-256 | 1.02982 | 1.02858 / 1.03026 |
| D-C0-XZ4-LoadBalancing / D-V2-256 | 1.01504 | 1.01448 / 1.01548 |
| D-C0-YZ6-Spread / D-V2-256 | 1.04111 | 1.04072 / 1.04287 |
| D-C0-YZ6-LoadBalancing / D-V2-256 | 1.02590 | 1.02503 / 1.02695 |
| D-C0-Z2-LoadBalancing / D-C0-Z2-Spread | 0.95756 | 0.95645 / 0.95894 |
| D-C0-XZ4-LoadBalancing / D-C0-XZ4-Spread | 0.98565 | 0.98629 / 0.98566 |
| D-C0-YZ6-LoadBalancing / D-C0-YZ6-Spread | 0.98539 | 0.98492 / 0.98473 |

These are complete-kernel ablations, not additive phase timings. Process
rounds expose drift; twenty trials in one process are not independent
process replicates. Ratios and ranges are descriptive, not confidence intervals.

### NSYS and NCU diagnostics

NSYS checks 50 launches over 100 physical steps, grid/block/shared and output
against B0. Cluster dimensions/policy are verified independently through NCU
launch metadata. NSYS 2025.1.3 SQLite has no cluster-dimension columns.

| Variant | Kernels | Kernel sum ms | GPU span ms | Steady µs/step | Median gap µs | Median launch API µs |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 50 | 13.532 | 13.612 | 135.778 | 1.696 | 2.814 |
| D-C0-Z2-Spread | 50 | 14.361 | 14.443 | 143.858 | 1.792 | 2.904 |
| D-C0-Z2-LoadBalancing | 50 | 13.733 | 13.809 | 137.649 | 1.504 | 3.175 |
| D-C0-XZ4-Spread | 50 | 13.984 | 14.057 | 140.065 | 1.440 | 3.440 |
| D-C0-XZ4-LoadBalancing | 50 | 13.768 | 13.849 | 137.970 | 1.728 | 3.255 |
| D-C0-YZ6-Spread | 50 | 14.123 | 14.208 | 141.393 | 1.792 | 3.480 |
| D-C0-YZ6-LoadBalancing | 50 | 13.943 | 14.021 | 139.874 | 1.664 | 4.447 |

NCU 2025.2.1 kernel replay, base clocks, launch 11 (t=20→22), both cache
policies. Additive counters/duration divide by two; ratios, resource counts
and percentages do not. Event timing remains the performance authority.
No-flush is a replay diagnostic. All profiler outputs match B0.

| Variant | NCU policy field | Flush read MB/step | Flush write MB/step | Flush total MB/step | No-flush total MB/step | Flush µs/step | No-flush µs/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | PolicySpread | 91.709 | 67.396 | 159.104 | 176.968 | 138.752 | 145.616 |
| D-C0-Z2-Spread | PolicySpread | 91.542 | 67.135 | 158.677 | 177.214 | 148.288 | 153.888 |
| D-C0-Z2-LoadBalancing | PolicyLoadBalancing | 91.687 | 67.174 | 158.861 | 176.916 | 141.888 | 147.072 |
| D-C0-XZ4-Spread | PolicySpread | 91.578 | 67.158 | 158.736 | 177.027 | 146.752 | 151.744 |
| D-C0-XZ4-LoadBalancing | PolicyLoadBalancing | 91.584 | 67.069 | 158.653 | 176.637 | 143.440 | 148.528 |
| D-C0-YZ6-Spread | PolicySpread | 91.586 | 67.228 | 158.814 | 176.958 | 148.848 | 154.272 |
| D-C0-YZ6-LoadBalancing | PolicyLoadBalancing | 91.559 | 67.298 | 158.857 | 176.837 | 146.448 | 151.584 |


| Flush-replay metric | D-V2-256 | D-C0-Z2-Spread | D-C0-Z2-LoadBalancing | D-C0-XZ4-Spread | D-C0-XZ4-LoadBalancing | D-C0-YZ6-Spread | D-C0-YZ6-LoadBalancing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Registers/thread | 56.000 | 56.000 | 56.000 | 56.000 | 56.000 | 56.000 | 56.000 |
| Achieved occupancy % | 32.690 | 28.700 | 31.610 | 31.040 | 31.330 | 29.010 | 30.260 |
| Eligible warps/scheduler/active cycle | 0.550 | 0.510 | 0.540 | 0.520 | 0.530 | 0.510 | 0.520 |
| Issue active % | 33.080 | 32.440 | 32.960 | 32.010 | 32.680 | 32.180 | 32.410 |
| L2 hit % | 70.830 | 70.740 | 70.860 | 70.880 | 70.800 | 70.860 | 70.900 |
| L2 throughput % | 67.710 | 63.490 | 66.280 | 64.220 | 65.600 | 63.320 | 64.220 |
| SM throughput % | 32.600 | 30.290 | 31.490 | 30.870 | 31.390 | 30.210 | 30.750 |
| Global ld requests/step | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 | 1618741.500 |
| Local ld requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global ld sectors/step | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 | 11570283.000 |
| Local ld sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared ld wavefronts/step | 724031.500 | 725112.000 | 726949.500 | 728473.000 | 728128.500 | 725128.500 | 726285.000 |
| Shared ld conflicts/step | 38444.000 | 39381.000 | 42541.000 | 42207.000 | 43130.000 | 40834.000 | 41120.500 |
| Global st requests/step | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 |
| Local st requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global st sectors/step | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 |
| Local st sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared st wavefronts/step | 997433.500 | 991036.000 | 993904.000 | 994914.000 | 993305.500 | 992120.500 | 994489.000 |
| Shared st conflicts/step | 78340.000 | 70977.500 | 74524.000 | 74462.000 | 76298.500 | 72918.000 | 73613.000 |
| Stall long_scoreboard per issue-active ratio | 3.800 | 3.480 | 3.920 | 3.820 | 3.840 | 3.510 | 3.790 |
| Stall short_scoreboard per issue-active ratio | 0.770 | 0.670 | 0.770 | 0.740 | 0.760 | 0.670 | 0.720 |
| Stall barrier per issue-active ratio | 0.790 | 0.720 | 0.780 | 0.760 | 0.760 | 0.730 | 0.730 |
| Stall lg_throttle per issue-active ratio | 1.340 | 1.120 | 1.300 | 1.300 | 1.310 | 1.160 | 1.280 |
| Stall mio_throttle per issue-active ratio | 0.760 | 0.580 | 0.690 | 0.660 | 0.640 | 0.550 | 0.590 |
| Stall wait per issue-active ratio | 1.660 | 1.660 | 1.660 | 1.660 | 1.660 | 1.660 | 1.660 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Shared bank conflicts are access serialization, not races. Stall ratios are
not time percentages. DSM interface bytes are not logical float payload.
The SASS useful-bytes/sector heuristic is deliberately omitted: its mixed
global/DSM kernel needs numerator/denominator verification (the earlier Z2 C2
report could exceed 32 B/sector). Raw metric values remain in the CSV.

## Stage D — selected larger DSM design

Event campaign started 2026-09-25T04:34:38.002567+00:00.
Four seeded shuffled 100-step rounds (80 trials/variant), two 1000-step
rounds (40 trials/variant), five warm-ups and twenty trials/process.
Both buffers restore outside event timing. Times include launch gaps;
allocation, validation, output and restores are excluded. Default LDC input.

| Variant | Cluster | Requested policy | Registers/thread | Shared B/block | Ordinary blocks/SM | Ordinary theoretical occupancy % | Max active clusters/device |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 0x0x0 | None | 56 | 38912 | 2 | 33.333333 | 340 |
| D-V3-Z2-LoadBalancing | 1x1x2 | LoadBalancing | 64 | 38912 | 2 | 33.333333 | 170 |
| D-C0-XZ4-LoadBalancing | 2x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 85 |
| D-C1-XZ4-LoadBalancing | 2x1x2 | LoadBalancing | 56 | 38912 | 2 | 33.333333 | 85 |
| D-V3-XZ4-LoadBalancing | 2x1x2 | LoadBalancing | 64 | 38912 | 2 | 33.333333 | 85 |

The occupancy API capacity is device-wide. The ordinary row queries singleton
clusters only for that diagnostic; its actual launch remains unclustered.

### 100 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.551 | 136.345 | 136.634 | 15818.28 | 1.00000 | 136.552 / 136.568 / 136.542 / 136.527 |
| D-V3-Z2-LoadBalancing | 139.548 | 139.368 | 139.644 | 15478.56 | 1.02195 | 139.484 / 139.548 / 139.536 / 139.578 |
| D-C0-XZ4-LoadBalancing | 138.369 | 138.231 | 138.445 | 15610.44 | 1.01331 | 138.333 / 138.379 / 138.410 / 138.373 |
| D-C1-XZ4-LoadBalancing | 145.092 | 144.836 | 145.267 | 14887.07 | 1.06255 | 145.015 / 144.984 / 145.215 / 145.203 |
| D-V3-XZ4-LoadBalancing | 153.226 | 152.986 | 153.366 | 14096.79 | 1.12212 | 153.115 / 153.253 / 153.201 / 153.331 |


### 1000 steps/trial — µs per physical timestep

| Variant | Median | Min | P90 | MLUPS | Time / ordinary | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 136.569 | 136.502 | 136.622 | 15816.18 | 1.00000 | 136.550 / 136.589 |
| D-V3-Z2-LoadBalancing | 139.632 | 139.517 | 139.665 | 15469.20 | 1.02243 | 139.620 / 139.639 |
| D-C0-XZ4-LoadBalancing | 138.393 | 138.272 | 138.520 | 15607.69 | 1.01336 | 138.310 / 138.505 |
| D-C1-XZ4-LoadBalancing | 145.282 | 145.130 | 145.351 | 14867.67 | 1.06380 | 145.175 / 145.327 |
| D-V3-XZ4-LoadBalancing | 153.386 | 153.273 | 153.453 | 14082.09 | 1.12314 | 153.341 / 153.424 |


| 1000-step comparison (numerator / denominator) | Pooled ratio | Same-round ratios |
| --- | --- | --- |
| D-V3-Z2-LoadBalancing / D-V2-256 | 1.02243 | 1.02248 / 1.02233 |
| D-C0-XZ4-LoadBalancing / D-V2-256 | 1.01336 | 1.01289 / 1.01403 |
| D-C1-XZ4-LoadBalancing / D-V2-256 | 1.06380 | 1.06316 / 1.06398 |
| D-V3-XZ4-LoadBalancing / D-V2-256 | 1.12314 | 1.12297 / 1.12325 |
| D-C1-XZ4-LoadBalancing / D-C0-XZ4-LoadBalancing | 1.04977 | 1.04963 / 1.04926 |
| D-V3-XZ4-LoadBalancing / D-C1-XZ4-LoadBalancing | 1.05579 | 1.05625 / 1.05571 |

These are complete-kernel ablations, not additive phase timings. Process
rounds expose drift; twenty trials in one process are not independent
process replicates. Ratios and ranges are descriptive, not confidence intervals.

### Four-rank ownership gate

Every physical shared (r,q) slot, Phase 2 core site and producer candidate
has exactly one visit/writer. CPU Cartesian and GPU production-loop counts
match; the debug launch rejects wrong worker count and cluster shape.

| Quantity per fused launch | CPU = GPU count |
| --- | --- |
| producer_rank0 | 969,000 |
| producer_rank1 | 969,000 |
| producer_rank2 | 969,000 |
| producer_rank3 | 969,000 |
| valid_rank0 | 944,700 |
| valid_rank1 | 846,002 |
| valid_rank2 | 924,600 |
| valid_rank3 | 827,998 |
| local | 37,307,300 |
| remote | 3,408,464 |
| fallback_local | 313,180 |
| fallback_remote | 11,056 |
| remote_x | 683,520 |
| remote_z | 2,699,520 |
| remote_xz | 36,480 |

Local/remote count valid producers; fallback categories are separate.
X/Z/XZ remote categories include fallbacks and sum to all remote stores.
Source-level float payload and producer visits are not hardware transactions.
Both XZ4 Phase 2 bodies match V2; C1 Phase 1 is unchanged. Production SASS
has two cluster arrive/wait pairs and no debug ATOM/RED/assertions.

### NSYS and NCU diagnostics

NSYS checks 50 launches over 100 physical steps, grid/block/shared and output
against B0. Cluster dimensions/policy are verified independently through NCU
launch metadata. NSYS 2025.1.3 SQLite has no cluster-dimension columns.

| Variant | Kernels | Kernel sum ms | GPU span ms | Steady µs/step | Median gap µs | Median launch API µs |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 50 | 13.534 | 13.613 | 135.762 | 1.568 | 2.423 |
| D-V3-Z2-LoadBalancing | 50 | 13.860 | 13.934 | 138.834 | 1.440 | 2.474 |
| D-C0-XZ4-LoadBalancing | 50 | 13.780 | 13.854 | 138.066 | 1.376 | 2.739 |
| D-C1-XZ4-LoadBalancing | 50 | 14.446 | 14.524 | 144.770 | 1.696 | 2.669 |
| D-V3-XZ4-LoadBalancing | 50 | 15.195 | 15.278 | 152.114 | 1.760 | 2.704 |

NCU 2025.2.1 kernel replay, base clocks, launch 11 (t=20→22), both cache
policies. Additive counters/duration divide by two; ratios, resource counts
and percentages do not. Event timing remains the performance authority.
No-flush is a replay diagnostic. All profiler outputs match B0.

| Variant | NCU policy field | Flush read MB/step | Flush write MB/step | Flush total MB/step | No-flush total MB/step | Flush µs/step | No-flush µs/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | PolicySpread | 91.682 | 67.116 | 158.797 | 177.115 | 138.784 | 144.208 |
| D-V3-Z2-LoadBalancing | PolicyLoadBalancing | 91.498 | 66.916 | 158.413 | 176.329 | 147.968 | 152.496 |
| D-C0-XZ4-LoadBalancing | PolicyLoadBalancing | 91.588 | 67.028 | 158.616 | 167.583 | 144.576 | 144.736 |
| D-C1-XZ4-LoadBalancing | PolicyLoadBalancing | 91.508 | 67.057 | 158.564 | 176.959 | 154.272 | 159.984 |
| D-V3-XZ4-LoadBalancing | PolicyLoadBalancing | 91.453 | 66.852 | 158.305 | 176.444 | 160.096 | 165.344 |


| Flush-replay metric | D-V2-256 | D-V3-Z2-LoadBalancing | D-C0-XZ4-LoadBalancing | D-C1-XZ4-LoadBalancing | D-V3-XZ4-LoadBalancing |
| --- | --- | --- | --- | --- | --- |
| Registers/thread | 56.000 | 64.000 | 56.000 | 56.000 | 64.000 |
| Achieved occupancy % | 33.510 | 32.780 | 31.930 | 32.180 | 32.070 |
| Eligible warps/scheduler/active cycle | 0.550 | 0.540 | 0.530 | 0.470 | 0.450 |
| Issue active % | 33.060 | 33.730 | 32.890 | 30.260 | 29.500 |
| L2 hit % | 70.760 | 67.940 | 70.760 | 72.370 | 61.570 |
| L2 throughput % | 67.720 | 61.730 | 65.060 | 64.390 | 55.900 |
| SM throughput % | 32.290 | 32.510 | 31.280 | 29.310 | 27.790 |
| Global ld requests/step | 1618741.500 | 1390569.500 | 1618741.500 | 1618741.500 | 1297752.000 |
| Local ld requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global ld sectors/step | 11570283.000 | 9649337.500 | 11570283.000 | 11570283.000 | 7273033.500 |
| Local ld sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared ld wavefronts/step | 722999.000 | 736387.500 | 728156.000 | 728549.000 | 737079.500 |
| Shared ld conflicts/step | 39380.000 | 53129.500 | 42784.500 | 44470.500 | 53075.500 |
| Global st requests/step | 684000.000 | 684000.000 | 684000.000 | 684000.000 | 684000.000 |
| Local st requests/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Global st sectors/step | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 | 3285000.000 |
| Local st sectors/step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Shared st wavefronts/step | 997994.000 | 906764.500 | 994901.000 | 992829.500 | 862111.000 |
| Shared st conflicts/step | 79219.000 | 50632.500 | 73319.500 | 75087.500 | 40073.000 |
| Stall long_scoreboard per issue-active ratio | 3.830 | 3.240 | 3.840 | 3.470 | 3.620 |
| Stall short_scoreboard per issue-active ratio | 0.760 | 0.420 | 0.730 | 0.670 | 0.480 |
| Stall barrier per issue-active ratio | 0.790 | 1.680 | 0.750 | 2.290 | 2.750 |
| Stall lg_throttle per issue-active ratio | 1.350 | 1.350 | 1.300 | 1.210 | 1.560 |
| Stall mio_throttle per issue-active ratio | 0.760 | 0.210 | 0.650 | 0.530 | 0.240 |
| Stall wait per issue-active ratio | 1.660 | 1.720 | 1.660 | 1.690 | 1.660 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 59992.500 | 0.000 | 0.000 | 262485.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum / step | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum / step | 0.000 | 209475.000 | 0.000 | 0.000 | 568905.000 |
| l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum / step | 0.000 | 7177440.000 | 0.000 | 0.000 | 18688320.000 |
| l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum / step | 0.000 | 7177440.000 | 0.000 | 0.000 | 18688320.000 |


### What the final controlled experiment supports

XZ4 V3 changes time by +5.58% versus its matched C1. It saves producer work but loses even within the same cluster topology and policy.
Both temporal variants use 64 registers/thread, 38,912 shared bytes/block
and the same ordinary two-block/SM resource bound. Local-memory traffic
is zero; the results do not support a spill explanation.

| Quantity per fused launch (unless noted) | Z2 V3 LB | XZ4 V3 LB |
| --- | --- | --- |
| Valid Phase 1 collisions | 3,766,500 | 3,543,300 |
| Logical remote FP32 payload MB | 10.944000 | 13.678080 |
| DSM store requests | 119985 | 524970 |
| DSM store sectors | 418950 | 1137810 |
| DSM interface write MB | 14.354880 | 37.376640 |
| Global-load sectors/physical step | 9649337.5 | 7273033.5 |
| DRAM MB/physical step | 158.413 | 158.305 |

The logical remote payload grows only 25%, but measured DSM interface
bytes grow about 2.6× and store requests about 4.4×. Global-load sectors
fall while DRAM bytes stay nearly unchanged. This is consistent with a
more expensive communication realization, not evidence that source-level
halo savings were absent. Aggregate counters do not isolate X versus Z
transaction costs or prove a single timing cause.

Static SASS inspection retains 19 generic Phase 1 stores, 19 local shared
loads and 19 final global stores in each V3. XZ4 has more static bit/shift
instructions (LOP3 44→69, SHF 23→37), but these are static code counts,
not dynamic execution or an instruction-level timing attribution. The
full disassembly and `codegen.json` are retained. No pointer-hoisting
variant was implemented after the measured stop condition was reached.
Shared bank conflicts are access serialization, not races. Stall ratios are
not time percentages. DSM interface bytes are not logical float payload.
The SASS useful-bytes/sector heuristic is deliberately omitted: its mixed
global/DSM kernel needs numerator/denominator verification (the earlier Z2 C2
report could exceed 32 B/sector). Raw metric values remain in the CSV.

## Validation and evidence

Every timed configuration passes full-allocation bitwise comparisons against B0
at 2 steps, sentinel/pattern/obstacle at 6, default 1000/1002, patterned 1000
and obstacles 100/102. Current N and inactive N−2, byte flags, unused flag-slot
bytes, padding/margins and invariant/Y-alias cases are preserved. Independent
velocity outputs match at 2/6/100; invalid counts are rejected. Four sanitizers
run six patterned/sentinel/obstacle steps. No tolerances are introduced.
A previously checked configuration may reuse its complete gate evidence only
when binary, compiled sources and obstacle bytes match; manifests name and hash
the reused gate. Event and profiler measurements are fresh in every stage.
C0 and ordinary D-V2 share the exact device function; its SASS hash matches
the original Z2 campaign. Source gates prove the timing and comparison driver
bodies are reused verbatim. Stage manifests bind source/binary hashes to results.

Exact prospective producer counts are saved in each stage's `producer-model.json`.
These are scalar flat-address model counts; C0 does not perform deduplication.
Raw event trials/order, pre-process telemetry, NSYS JSON and NCU CSV/details are
under `results/cluster-sweep/`. Large profiler binaries are retained locally
and ignored by Git. Report generation performs no GPU workloads.

## Recorded stage decisions

`decision-a.json`:

```json
{
  "stage": "A",
  "material": true,
  "threshold": "at least 1% lower pooled time at both lengths; every 1000-step round improves",
  "ratios": {
    "100": 0.9574724282206043,
    "1000": 0.9561139040732425
  },
  "next": "A2: rerun Z2 C1/V3 under both policies before shape sweep"
}
```

`decision-a2.json`:

```json
{
  "stage": "A2",
  "best_z2_policy": "LoadBalancing",
  "best_z2_us": 139.63225555399998,
  "ordinary_us": 136.5981826785,
  "v3_vs_ordinary_ratio": 1.0222116635522234,
  "v3_lb_vs_spread_ratio": 0.9885386658490278,
  "next": "B: measure C0 XZ4/YZ6 before deciding whether larger DSM is worth implementing"
}
```

`decision-c.json`:

```json
{
  "stage": "C",
  "selected": "XZ4",
  "policy": "LoadBalancing",
  "event_medians_1000": {
    "D-V2-256": 136.574081421,
    "D-C0-Z2-LoadBalancing": 138.0915985105,
    "D-C0-XZ4-LoadBalancing": 138.628746033,
    "D-C0-YZ6-LoadBalancing": 140.1114654545
  },
  "xz4_vs_z2_c0_ratio": 1.0038897914738756,
  "reason": "XZ4 C0 cost is close to Z2 and below YZ6, with greater prospective halo savings than Z2; use the guide's default XZ4 ownership partition.",
  "next": "D: implement XZ4 C1 and dedup, then run full ownership/numerical gates and fresh matched measurements."
}
```

`decision-d.json`:

```json
{
  "stage": "D",
  "ordinary_us": 136.5690078735,
  "finalists": {
    "D-V3-Z2-LoadBalancing": {
      "us": 139.6323013305,
      "ratio_to_ordinary": 1.0224303705847921
    },
    "D-V3-XZ4-LoadBalancing": {
      "us": 153.386352539,
      "ratio_to_ordinary": 1.1231417356496975
    }
  },
  "best_cluster": "D-V3-Z2-LoadBalancing",
  "best_cluster_ratio": 1.0224303705847921,
  "stop_t2_cluster_tuning": true,
  "reason": "Best controlled Tt=2 cluster still loses by more than 2%, within the supplied guide's strong stop condition. XZ4 dedup is slower even than matched XZ4 C1.",
  "stage_E": "Not executed: no clustered V3 beat ordinary D-V2. No pointer-hoisting or async-barrier variant added.",
  "future_hypothesis": "Deeper temporal blocking (Tt>2), a separate experiment, not implemented here."
}
```

