# Cluster profitability sweep

This is the next experiment after [Z2](Z2_DESIGN.md). It asks which cluster
launch configuration has low enough fixed cost for halo deduplication to pay.
The supplied guide is implemented in measured stages, with ordinary D-V2 in
every event/profiler campaign. See the [decision record](../experiments/cluster_sweep/DECISIONS.md)
and [results](CLUSTER_SWEEP_RESULTS.md).

## A: scheduling policy before new kernels

`D-C0-Z2-Spread` and `D-C0-Z2-LoadBalancing` use the **same function pointer**
as ordinary `D-V2-256`. Only launch attributes differ. Z2 is 1×1×2 blocks.
The explicit preference uses `cudaLaunchAttributeClusterSchedulingPolicyPreference`.
CUDA defines Spread as distributing cluster blocks to SMs and LoadBalancing
as permitting hardware load balancing. A preference is not proof of a
particular physical placement; NCU's reported policy is checked in both
replay policies. See the [CUDA runtime enum](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__TYPES.html).

If LoadBalancing lowers event time by at least 1% at both trial lengths and
improves every long-run round, A2 measures existing Z2 C1 and V3 with both
policies before any larger DSM implementation. This threshold is a practical
experiment decision, not a statistical significance test.

## B: launch-only topology sweep

The first three shapes are Z2 (1×1×2), XZ4 (2×1×2), and YZ6 (1×3×2), each
under both scheduling preferences. Every dimension divides the unchanged
8×15×38 block grid. Each block still has 256 threads, 16×8×4 physical core,
38,912 bytes of streamed-R shared state and two physical timesteps/launch.
C0 adds no indexing, synchronization or DSM access to the D-V2 kernel.

[launch.cuh](../experiments/cluster_sweep/launch.cuh) takes a `ClusterShape`
and policy from the variant descriptor and sets two runtime attributes.
Ordinary D-V2 uses **zero launch attributes**. The occupancy diagnostic uses
an explicit singleton cluster only when querying the ordinary reference.
It records device-wide `cudaOccupancyMaxActiveClusters`, potential cluster
size, ordinary block occupancy, registers and shared bytes for every actual
configuration. Unsupported configurations receive no timing value.

[model.cpp](../experiments/cluster_sweep/model.cpp) independently enumerates
flat-address-valid producers for all six shapes in the guide. The first
three measured shapes are distinguished from model-only X2/Y3/X4Z2. These
are prospective deduplication counts: C0 continues doing independent halos.
They are not DRAM byte or performance predictions.

## Code reuse and preservation

The new code is under `experiments/cluster_sweep/`, with a separate executable
`build/sweep/lbm` and separate build file. Original Z2 kernels, launch wrapper,
source/binary hashes and result files are preserved.

[prepare_driver.py](../experiments/cluster_sweep/prepare_driver.py) reads the
existing `src/driver.cuh` and replaces only its selected-variant/setup/launch
section with the new launch helper in a generated build-directory header.
All initialization, numerical comparison, live-buffer tracking, event timing,
argument validation and output code is reused **verbatim**. A source gate
checks that exact splice; it does not silently maintain a second copy of
the benchmark. The numerical kernel source is imported directly, and C0's
compiled SASS hash must match the previously validated D-V2 body.

The Python runner reuses the Z2 timing and profiling functions, supplies new
variant lists and output paths, then checks the actual NCU cluster dimensions
and scheduling-policy fields. NSYS 2025.1.3's SQLite kernel table does not
expose cluster dimensions. Its kernel count/grid/block/shared checks are
supplemented by NCU metadata rather than inferred as a trace measurement.

## Validation and measurement

Before each stage's timing, every supported row passes the existing bitwise
full-allocation contract at two steps, sentinel/pattern/obstacle input at six,
default histories 1000/1002, patterned 1000, and obstacles 100/102. Current N
and inactive N−2 are compared against corresponding B0 states. Velocity
outputs at 2/6/100 match B0; malformed/odd/nonpositive counts are rejected.
Identical previously validated configurations reuse their gate only when the
compiled sources, binary and obstacle bytes all match; the manifest records
the prior gate and its hash. Event/profiler data is fresh in every stage.
Memcheck, racecheck, synccheck and global initcheck run six patterned,
sentinel and obstacle steps. Obstacles cover block/cluster interfaces,
including X 15/16 and 31/32, Z 3/4 and 7/8, their intersections and the
partial physical-domain end tiles.

The protocol is unchanged: four shuffled 100-step rounds and two 1000-step
rounds; five warm-ups and twenty trials/process, with both device buffers
restored outside each CUDA-event interval. Times are per physical step and
include launch gaps. Performance input is default LDC. Numerical obstacle
coverage is not an obstacle-density performance sweep.

NSYS checks every launch in 100 physical steps. NCU samples launch 11
(t=20→22), kernel replay, base-clock control, flush and no-flush cache
policies. Installed DSM metrics are queried; none is assumed to exist.
Additive counters and durations divide by two. Rates, occupancy, registers
and ratios do not. Event measurements determine performance rankings.

## Reproduce from the repository root

```sh
export CUDA_CXX=/path/to/clang++
export CUDA_PATH=/usr/local/cuda-12.9
export CUDA_VISIBLE_DEVICES=0
make -f experiments/cluster_sweep/Makefile

# Use a new directory; completed evidence is not overwritten.
export LBM_SWEEP_RESULTS="$PWD/results/cluster-sweep-local"
python3 experiments/cluster_sweep/run.py a campaign
# Inspect A; if material, run A2 before proceeding:
python3 experiments/cluster_sweep/run.py a2 campaign
python3 experiments/cluster_sweep/run.py b campaign
python3 experiments/cluster_sweep/report.py
```

A stage can also run `gates`, `bench`, `profile`, and `summary` separately.
The event/profiler stages require completed gates matching source and binary
hashes. A file lock serializes campaign processes. Large profiler binaries,
SQLite exports, metric-query output and temporary files are retained locally
but ignored by Git; compact logs, CSV and JSON remain reviewable.

C/D is a measured decision: XZ4 is preferred when its fixed cost is comparable
to Z2. If a larger shape is selected, its DSM ownership and full validation
are a separate stage. E (such as pointer hoisting) is a separate variant only
after a clustered implementation proves profitable. No TMA, rolling storage,
precision changes, asynchronous barriers or deeper temporal depth is mixed
into this sweep.

## D: selected XZ4 implementation

Stage B selected XZ4 with LoadBalancing: its C0 time is only 0.39% above
Z2 with that policy, and below YZ6. The decision is recorded before the
larger kernel was implemented in `results/cluster-sweep/decision-c.json`.

[xz4.cuh](../experiments/cluster_sweep/xz4.cuh) provides a matched C1 and V3.
C1 retains independent full 18×10×6 halos with two cluster syncs. V3 owns a
32×8×8 cluster core through four separate 38,912-byte per-block arrays.
Each block retains its original 16×8×4 consumer tile and final scatter.

For block coordinates `(bx,0,bz)`, the rank is `bx + 2*bz`. Producer X is
−1…15 for bx=0 and 16…32 for bx=1; producer Z is −1…3 for bz=0 and 4…8
for bz=1. Every block traverses 17×10×5 = 850 candidates. Destinations choose
rank `(rx/16) + 2*(rz/4)` and use the original 512-site local shared index.
Valid producers collide once; invalid producers explicitly copy `src[r,q]`
into the destination's local or remote shared slot. Flat-address producer
validity, flags and all collision expressions are unchanged.

Two unconditional cluster barriers protect DSM lifetime and complete all
Phase 1 remote stores. Phase 2 reads local shared only. Source gates compare
both Phase 2 bodies to V2 verbatim after removing debug-only counters, and
compare C1's complete Phase 1 with the original duplicated-halo code.

[xz4_model.cpp](../experiments/cluster_sweep/xz4_model.cpp) uses Cartesian
enumeration and inverse `r-cq` checks independently of the GPU decoder.
The [debug harness](../experiments/cluster_sweep/xz4_ownership.cu) verifies
all 3,876,000 candidate visits, 2,160,000 consumer visits and 41,040,000 shared
writers equal one. Per-rank producer/valid counts, valid local/remote stores,
fallback local/remote stores and X/Z/XZ remote categories must match the CPU.
The XZ category specifically covers ET/EB/WT/WB crossing both interfaces.
Invalid 192-thread and Z2-shaped launches must trigger debug assertions.
Production SASS must contain two cluster arrive/wait pairs and no debug
atomics/assertions. The same long histories and sanitizers precede timing.

The second executable keeps the launch-sweep binary and its hashes intact:

```sh
make -f experiments/cluster_sweep/Makefile.xz4 sweep xz4
python3 experiments/cluster_sweep/run.py d campaign
python3 experiments/cluster_sweep/report.py
```

D freshly compares ordinary D-V2, the best Z2 V3 policy, and XZ4 C0/C1/V3
under LoadBalancing. Code-generation optimization is conditional on the
measured outcome, as specified in the guide.

## Measured stop decision

The fresh D campaign measured ordinary D-V2 at 136.569 µs/step, Z2 V3
LoadBalancing at 139.632 (+2.24%), and XZ4 V3 LoadBalancing at 153.386
(+12.31%). XZ4 C1 was 145.282, so deduplication plus DSM also loses against
its matched synchronization control. All numerical and ownership gates pass.

This reaches the guide's stop condition: the best controlled Tt=2 cluster
still loses by 2–3%. Stage E is intentionally not entered. The repository
contains no pointer-hoisted, asynchronous-barrier or deeper-temporal variant
from this campaign. The exact decision and its inputs are saved in
`results/cluster-sweep/decision-d.json`; future Tt>2 work is not implemented.
