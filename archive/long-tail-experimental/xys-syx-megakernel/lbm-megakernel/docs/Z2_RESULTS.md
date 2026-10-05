# D-V3-Z2-256: cluster/DSMEM experiment

Measured 2026-09-25 (UTC) on RTX 5090 GPU 0, driver 575.57.08, CUDA 12.9.
Standalone Clang 23.0.0git (`c813428fb1b5678a6e4c541f4aee82a7977e95ea`),
`sm_120`, original `-O3 -ffast-math` numerical flags. All four variants were
freshly measured together; older README measurements are not pooled here.

**C2 / D-V2 time = 1.03509× (+3.51%).**
D-V2 takes 135.681 µs/physical step; C2 takes 140.442.
C2 changes time by -3.82% versus C1.
The deduplication helps within the clustered design, but does not beat the
ordinary D-V2 launch in this campaign.

See the [implementation and ownership contract](Z2_DESIGN.md) for kernel
links, the boundary argument, build requirements, and reproduction commands.

## Implementations and runtime resources

All variants own 16×8×4 per block, use 256 threads, grid 8×15×38,
38,912 B dynamic shared/block, and two physical timesteps/launch.
Static application shared and runtime local bytes/thread are zero. C0 uses
the exact D-V2 device function; C1 changes synchronization; C2 adds deduplication
and DSM stores. C0/C1/C2 here are unrelated to the older spatial C1/C2/C3 names.

| Variant | Cluster | Reg/thread | Ordinary blocks/SM | Max active clusters/device | Potential max cluster size |
| --- | --- | --- | --- | --- | --- |
| D-V2-256 | ordinary (query 1×1×1) | 56 | 2 | 340 | 8 |
| D-C0-Z2-256 | 1×1×2 | 56 | 2 | 170 | 8 |
| D-C1-Z2-256 | 1×1×2 | 56 | 2 | 170 | 8 |
| D-V3-Z2-256 | 1×1×2 | 64 | 2 | 170 | 8 |

Cluster occupancy is queried through `cudaOccupancyMaxActiveClusters`;
the result is device-wide capacity, not measured residency or clusters/SM.
The ordinary reference is queried with explicit singleton dimensions, but
its actual launch has no cluster attribute. All rows use `cudaLaunchKernelExC`
in this experiment; the default executable remains unchanged.

## Correctness gates completed before timing

- One fused launch equals two B0 launches across all initialized bytes.
- Default histories 1000/1002, patterned 1000, external obstacles 100/102 pass for all four variants.
- Current N and inactive N−2 are compared to the corresponding B0 states; short checkpoints 2/4/6 are also checked.
- Nonzero sentinels, never-produced population perturbations, flags, unused flag-slot bytes, padding and margins pass bitwise.
- Physical boundaries and internal interfaces z=3/4, 67/68, 147/148 are exercised by external obstacles.
- Independent B0 velocity output is byte-identical at 2/6/100 steps; malformed/odd/nonpositive step counts are rejected.
- Memcheck, racecheck, synccheck and global initcheck pass six-step patterned/sentinel/obstacle runs for all four variants.
- Debug assertions reject a 192-thread block and a singleton cluster.
- CPU/GPU ownership counts match exactly, with one visit per producer candidate/core and one writer per consumed shared slot.

No numerical tolerance was introduced. Every compared physical population
is finite; bit mismatches, maximum absolute/RMS/relative errors, flag mismatches
and full-allocation differences are zero. Global initcheck is not a shared
initialization proof; writer counts, racecheck and full-state comparisons
provide the additional evidence.

Production SASS has no debug ATOM/RED/assertion instructions. D-V2/C0
contains one block barrier. C1 and C2 each contain two `UCGABAR_ARV` /
`UCGABAR_WAIT` pairs, with their block barriers and memory ordering.
Source checks preserve the collision bodies and exact V2 Phase 2. Binary
and source hashes bind all timed/profiled workloads to completed gates.

## Ownership and source-level work

| Quantity per fused launch | Count |
| --- | --- |
| Original / C2 producer candidates | 4,924,800 / 4,104,000 |
| Original / C2 valid Phase 1 collisions | 4,536,000 / 3,766,500 |
| Phase 2 collisions / shared writers | 2,160,000 / 41,040,000 |
| producer_rank0 | 2,052,000 |
| producer_rank1 | 2,052,000 |
| valid_rank0 | 1,903,502 |
| valid_rank1 | 1,862,998 |
| local | 37,988,884 |
| remote | 2,726,880 |
| fallback_local | 315,116 |
| fallback_remote | 9,120 |
| remote_rank0 | 1,368,000 |
| remote_rank1 | 1,368,000 |
| interior_remote_rank0 | 640 |
| interior_remote_rank1 | 640 |

Local/remote rows count valid producers; fallback rows are separate and
disjoint. Rank remote totals include fallbacks. Remote stores total 2,736,000
floats = 10.944 MB/fused launch = 5.472 MB/physical step. A representative
full interior interface has 640 stores each way. These are logical FP32
payload and source visits, not hardware sectors or executed instruction counts.

## Unprofiled CUDA-event timing

Seed 20260924. Four shuffled 100-step rounds and two shuffled 1000-step
rounds; each process has five warm-ups and twenty trials. Both buffers are
restored from identical snapshots outside every timed interval. Allocation,
validation, restoration and output are excluded; GPU launch gaps are included.
All times below are µs/physical timestep, with nearest-rank P90.
MLUPS = 2,160,000 / µs_per_step. Default LDC, no external obstacles.

### 100 physical steps/trial

| Variant | Trials | Median | Minimum | P90 | MLUPS | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 80 | 135.729 | 135.567 | 135.813 | 15914.07 | 135.732 / 135.721 / 135.720 / 135.729 |
| D-C0-Z2-256 | 80 | 143.617 | 143.452 | 143.710 | 15040.01 | 143.581 / 143.608 / 143.657 / 143.678 |
| D-C1-Z2-256 | 80 | 145.803 | 145.563 | 145.927 | 14814.51 | 145.834 / 145.770 / 145.834 / 145.792 |
| D-V3-Z2-256 | 80 | 140.382 | 140.136 | 140.514 | 15386.58 | 140.309 / 140.437 / 140.463 / 140.376 |


### 1000 physical steps/trial

| Variant | Trials | Median | Minimum | P90 | MLUPS | Round medians |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 40 | 135.681 | 135.619 | 135.708 | 15919.68 | 135.679 / 135.688 |
| D-C0-Z2-256 | 40 | 143.653 | 143.602 | 143.682 | 15036.23 | 143.649 / 143.664 |
| D-C1-Z2-256 | 40 | 146.022 | 145.836 | 146.078 | 14792.25 | 145.889 / 146.051 |
| D-V3-Z2-256 | 40 | 140.442 | 140.382 | 140.476 | 15379.99 | 140.462 / 140.429 |


### Complete-kernel time ratios — below one favors the numerator

| Steps | Comparison | Pooled median ratio | Same-round median ratios |
| --- | --- | --- | --- |
| 100 | D-V3-Z2-256 / D-V2-256 | 1.03428 | 1.03372 / 1.03474 / 1.03495 / 1.03424 |
| 100 | D-C0-Z2-256 / D-V2-256 | 1.05812 | 1.05783 / 1.05811 / 1.05848 / 1.05857 |
| 100 | D-C1-Z2-256 / D-C0-Z2-256 | 1.01522 | 1.01569 / 1.01506 / 1.01516 / 1.01471 |
| 100 | D-V3-Z2-256 / D-C1-Z2-256 | 0.96282 | 0.96212 / 0.96341 / 0.96317 / 0.96285 |
| 1000 | D-V3-Z2-256 / D-V2-256 | 1.03509 | 1.03525 / 1.03494 |
| 1000 | D-C0-Z2-256 / D-V2-256 | 1.05876 | 1.05874 / 1.05879 |
| 1000 | D-C1-Z2-256 / D-C0-Z2-256 | 1.01649 | 1.01560 / 1.01661 |
| 1000 | D-V3-Z2-256 / D-C1-Z2-256 | 0.96179 | 0.96280 / 0.96151 |

Twenty trials in one process are not twenty independent process replicates.
Same-round ratios compare separately shuffled processes, not simultaneous
paired trials. Round ranges are descriptive, not confidence intervals.

## NSYS: actual launches, runtime geometry and gaps

NSYS 2025.1.3 CUDA tracing; no CPU sampling/context-switch capture. Every
launch in each 100-physical-step run passes identity/grid/block/shared checks.
All profiled velocity outputs match B0. Steady medians exclude launch one.

| Variant | Launches | Reg/thread | Kernel sum ms | GPU span ms | Steady µs/kernel | Steady µs/step | Median gap µs | Median launch API µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 50 | 56 | 13.605 | 13.684 | 272.388 | 136.194 | 1.504 | 2.840 |
| D-C0-Z2-256 | 50 | 56 | 14.336 | 14.419 | 287.075 | 143.537 | 1.792 | 2.924 |
| D-C1-Z2-256 | 50 | 56 | 14.563 | 14.641 | 291.684 | 145.842 | 1.536 | 2.875 |
| D-V3-Z2-256 | 50 | 64 | 14.001 | 14.080 | 280.388 | 140.194 | 1.408 | 2.835 |

Grid/block/shared are the common configuration above. This NSYS SQLite
schema does not expose cluster dimensions; those come from the runtime
launch configuration and debug assertions, with NCU launch metadata retained.
Host API time overlaps GPU execution and is not added to kernel duration.
Trace timing is auxiliary; event medians determine speed comparisons.

## NCU: matched t=20→22, two replay cache policies

NCU 2025.2.1, kernel replay, `--clock-control base`. Each variant samples
launch 11. Additive counters and duration are divided by two for physical-step
reporting; resource counts, percentages, hit rates and ratios are not.
CSV base units are converted explicitly. No-flush is replay sensitivity,
not an unprofiled steady-state timing estimate. All outputs match B0.

### Flush before replay

| Variant | Read MB/step | Write MB/step | Total MB/step | µs/kernel | µs/step | Actual DRAM TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 91.695 | 67.126 | 158.822 | 277.984 | 138.992 | 1.143 |
| D-C0-Z2-256 | 91.570 | 67.232 | 158.802 | 295.840 | 147.920 | 1.074 |
| D-C1-Z2-256 | 91.575 | 67.195 | 158.769 | 309.760 | 154.880 | 1.025 |
| D-V3-Z2-256 | 91.667 | 67.378 | 159.045 | 300.928 | 150.464 | 1.057 |


### No flush before replay

| Variant | Read MB/step | Write MB/step | Total MB/step | µs/kernel | µs/step | Actual DRAM TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| D-V2-256 | 91.374 | 86.169 | 177.543 | 288.416 | 144.208 | 1.231 |
| D-C0-Z2-256 | 91.210 | 86.365 | 177.574 | 309.504 | 154.752 | 1.147 |
| D-C1-Z2-256 | 91.209 | 85.880 | 177.090 | 319.328 | 159.664 | 1.109 |
| D-V3-Z2-256 | 91.203 | 85.899 | 177.103 | 310.880 | 155.440 | 1.139 |


### NCU launch metadata — verified for both replay policies

| Variant | Cluster dimensions | Cluster size | Max active clusters/device | Scheduling policy |
| --- | --- | --- | --- | --- |
| D-V2-256 | 0×0×0 | 0 | 0 | PolicySpread |
| D-C0-Z2-256 | 1×1×2 | 2 | 170 | PolicySpread |
| D-C1-Z2-256 | 1×1×2 | 2 | 170 | PolicySpread |
| D-V3-Z2-256 | 1×1×2 | 2 | 170 | PolicySpread |

NCU reports zero cluster dimensions/size/capacity for the ordinary launch.
The runtime occupancy diagnostic above explicitly queries singleton clusters,
so its 340-cluster capacity has a different configuration. The cluster variants
use the default scheduling policy; no scheduling preference was set.

Actual bandwidth uses measured DRAM bytes and NCU duration, not the
equivalent 152-B useful population convention. Event timing remains primary.

### Flush replay: memory, resources and scheduling

| Metric | D-V2-256 | D-C0-Z2-256 | D-C1-Z2-256 | D-V3-Z2-256 |
| --- | --- | --- | --- | --- |
| Registers/thread | 56.000 | 56.000 | 56.000 | 64.000 |
| Allocated registers/thread | 56.000 | 56.000 | 56.000 | 64.000 |
| Block limit: shared memory | 2.000 | 2.000 | 2.000 | 2.000 |
| Block limit: registers | 4.000 | 4.000 | 4.000 | 4.000 |
| Theoretical occupancy % | 33.330 | 33.330 | 33.330 | 33.330 |
| Achieved occupancy % | 32.700 | 29.100 | 32.660 | 31.970 |
| Active warps/SM | 15.700 | 13.970 | 15.670 | 15.350 |
| Eligible warps/scheduler/active cycle | 0.550 | 0.510 | 0.470 | 0.520 |
| Issue active % | 33.300 | 32.440 | 30.460 | 32.940 |
| Global load useful B/sector | 15.240 | 15.240 | 15.240 | 15.210 |
| Global store useful B/sector | 24.990 | 24.990 | 24.990 | 49.970 |
| L1/TEX hit % | 3.990 | 4.510 | 4.260 | 4.110 |
| L2 hit % | 70.820 | 70.870 | 71.720 | 67.940 |
| DRAM throughput % | 64.750 | 60.840 | 58.090 | 59.900 |
| L1/TEX throughput % | 32.040 | 30.100 | 28.580 | 28.200 |
| L2 throughput % | 67.590 | 63.650 | 64.230 | 60.740 |
| SM throughput % | 32.320 | 30.500 | 29.060 | 31.690 |
| Global load requests/step | 1618741.5 | 1618741.5 | 1618741.5 | 1390569.5 |
| Global load sectors/step | 11570283.0 | 11570283.0 | 11570283.0 | 9649337.5 |
| Local load requests/step | 0.0 | 0.0 | 0.0 | 0.0 |
| Local load sectors/step | 0.0 | 0.0 | 0.0 | 0.0 |
| Shared load bank conflicts/step | 38722.5 | 40460.5 | 32954.5 | 41474.5 |
| Shared load wavefronts/step | 724336.0 | 725287.0 | 717727.0 | 725646.0 |
| Global store requests/step | 684000.0 | 684000.0 | 684000.0 | 684000.0 |
| Global store sectors/step | 3285000.0 | 3285000.0 | 3285000.0 | 3285000.0 |
| Local store requests/step | 0.0 | 0.0 | 0.0 | 0.0 |
| Local store sectors/step | 0.0 | 0.0 | 0.0 | 0.0 |
| Shared store bank conflicts/step | 78622.0 | 71460.0 | 72543.0 | 50975.5 |
| Shared store wavefronts/step | 998220.0 | 990691.0 | 990372.0 | 906511.0 |
| Stall long_scoreboard per issue-active ratio | 3.860 | 3.480 | 3.300 | 3.070 |
| Stall short_scoreboard per issue-active ratio | 0.770 | 0.670 | 0.690 | 0.420 |
| Stall barrier per issue-active ratio | 0.790 | 0.710 | 2.350 | 2.000 |
| Stall lg_throttle per issue-active ratio | 1.370 | 1.130 | 1.200 | 1.350 |
| Stall mio_throttle per issue-active ratio | 0.760 | 0.570 | 0.550 | 0.230 |
| Stall wait per issue-active ratio | 1.660 | 1.660 | 1.690 | 1.720 |
| SM clock GHz | 1.980 | 1.971 | 1.989 | 1.987 |
| DRAM clock GHz | 13.786 | 13.786 | 13.787 | 13.786 |

Shared conflicts measure access serialization, not races. Local traffic is
measured, not inferred from register count. Achieved occupancy uses active
cycles; stall ratios are not elapsed-time percentages or an additive time budget.

### Installed DSM-specific counters — per physical step, flush replay

| Metric (installed name) | D-V2-256 | D-C0-Z2-256 | D-C1-Z2-256 | D-V3-Z2-256 |
| --- | --- | --- | --- | --- |
| `l1tex__t_requests_pipe_lsu_mem_dshared_op_ld.sum` | 0.0 | 0.0 | 0.0 | 0.0 |
| `l1tex__t_requests_pipe_lsu_mem_dshared_op_st.sum` | 0.0 | 0.0 | 0.0 | 59992.5 |
| `l1tex__t_sectors_pipe_lsu_mem_dshared_op_ld.sum` | 0.0 | 0.0 | 0.0 | 0.0 |
| `l1tex__t_sectors_pipe_lsu_mem_dshared_op_st.sum` | 0.0 | 0.0 | 0.0 | 209475.0 |
| `l1tex__m_l1tex2xbar_write_bytes_mem_dshared_op_st.sum` | 0.0 | 0.0 | 0.0 | 7177440.0 |
| `l1tex__m_xbar2l1tex_read_bytes_mem_dshared_op_st.sum` | 0.0 | 0.0 | 0.0 | 7177440.0 |

D-V2, C0 and C1 have zero DSM counters. C2 has nonzero remote stores
and zero remote loads in both replay policies, consistent with Phase 1-only
remote writes and local Phase 2 reads. Its hardware interface reports
14.354880 MB/fused launch,
versus 10.944 MB of logical remote float payload; these are different quantities.

Selected metrics were found in the installed query output; names and descriptions
are retained in `dsm-metrics.json`. DSM request/sector/interface-byte counters
describe hardware interfaces and must not be identified with the source-level
remote float-store count. The full base-unit CSV includes both cache policies.

## What the experiment supports

- D-V2-256 → D-C0-Z2-256: +5.88% event time at 1000 steps.
- D-C0-Z2-256 → D-C1-Z2-256: +1.65% event time at 1000 steps.
- D-C1-Z2-256 → D-V3-Z2-256: -3.82% event time at 1000 steps.
- C2 changes flush-replay DRAM bytes/step by +0.14% versus D-V2.
- Fewer duplicated producer collisions can help the clustered kernel, while cluster placement and synchronization still affect the complete result.
- Identical shared capacity and ordinary block occupancy do not establish identical cluster placement, scheduler behavior or communication cost.
- C1→C2 changes computation, DSM addressing/communication and generated resources together; these controls are not additive phase timings.

The current fastest hand-written configuration remains D-V2-256 in this
comparison. The result argues for a compiler profitability decision, rather
than unconditional replacement of redundant halo work with cluster communication.

Scope remains one GPU/toolchain, fixed 120×120×150, even-step FP32 evolution,
immutable flags and matching invariant initial slots. External obstacles extend
correctness coverage; performance uses default LDC. Two long-run process rounds
do not establish a broad statistical claim. No larger cluster, TMA, rolling
storage, asynchronous barrier, precision or worker-count change was included.

## Reproduce and inspect

See [Z2_DESIGN.md](Z2_DESIGN.md#build-validate-measure). Run from the repo folder:

```sh
make cluster-report  # saved evidence only
LBM_Z2_RESULTS="$PWD/results/z2-local" CUDA_VISIBLE_DEVICES=0 make cluster-all
```

Evidence is in `results/z2/`: source/binary/SASS hashes, independent model and
GPU ownership counts, numerical/sanitizer logs, every event trial and execution
order, pre-process telemetry, NSYS timeline JSON, NCU base-unit CSV and details.
Large `.nsys-rep`, `.sqlite` and `.ncu-rep` files are retained locally and ignored
by Git. The report generator performs no GPU workloads.
