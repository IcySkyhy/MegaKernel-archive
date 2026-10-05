# D-V3-Z2-256: replace an internal halo overlap with DSM communication

This experiment extends `D-V2-256` without changing its numerical contract.
Each block still owns a 16×8×4 second-collision tile, uses 256 threads and
38,912 bytes of dynamic shared memory, and advances two physical timesteps.
The grid is 8×15×38 blocks. Runtime clusters contain 1×1×2 blocks, so the
cluster owns a 16×8×8 region, with 77,824 bytes distributed across two
separate shared allocations.

## Four complete-kernel controls

| Executable variant | Phase 1 producer work | Communication / synchronization | Source |
| --- | --- | --- | --- |
| `D-V2-256` | Each block computes its 18×10×6 box | Local shared, original block barrier; ordinary launch | [temporal_v2.cuh](../kernels/temporal_v2.cuh) |
| `D-C0-Z2-256` | Identical to D-V2, including overlap | Exact same kernel function, launched in Z2 clusters | [variants.cuh](../src/variants.cuh) |
| `D-C1-Z2-256` | Identical to D-V2, including overlap | Cluster sync before Phase 1 and after all local shared stores; no DSM access | [cluster_sync.cuh](../kernels/cluster_sync.cuh) |
| `D-V3-Z2-256` (C2) | Each block owns 900 candidates from the cluster's 18×10×10 box | Two cluster syncs; Phase 1 stores into local or remote shared | [cluster_dedup.cuh](../kernels/cluster_dedup.cuh) |

Here C0/C1/C2 are **cluster experiment controls**. They are unrelated to the
older one-step `C1`, `C2`, and `C3` optimization names in the main README.

All four rows use `cudaLaunchKernelExC` in the separate `build/lbm-z2`
executable. The ordinary reference has no launch attributes; the other
three have a runtime cluster-dimension attribute. This keeps the host launch
API consistent within the fresh campaign. The default `build/lbm` retains
its original launch API and variant list.

## Ownership and data flow

Let cluster-local second-collision Z be 0…7. Rank 0 owns producer planes
−1…3; rank 1 owns 4…8. Each rank evaluates 18×10×5 candidates. A producer
`p` streams direction `q` to `r=p+c_q`. A physical destination inside the
cluster core selects rank `r.z/4` and local index
`r.x + 16*(r.y + 8*(r.z%4))`. Its shared slot is `q*512 + local_index`.

```text
cluster.sync()                         ensure both shared allocations exist
  rank 0: producers z = −1…3  ─┐
  rank 1: producers z =  4…8  ─┴─ collision → local/remote streamed f(t+1)[R]
cluster.sync()                         complete all remote accesses
  each block: local shared reads → unchanged collision → original global scatter
```

For each `(r,q)`, the unique producer is `r-c_q`, assigned to exactly one
rank. Consequently no production atomic is needed. Only Phase 1 accesses
remote shared memory. Phase 2 is source-identical to D-V2, apart from
instrumentation that exists only in the debug executable. Every thread
reaches both cluster barriers; physical-boundary exclusions happen inside
loops. After the second barrier no block accesses another block's shared
allocation, so blocks can finish Phase 2 independently.

The first barrier establishes DSM lifetime; the second orders DSM writes
and keeps participating shared allocations alive until all accesses finish.
These follow NVIDIA's [distributed shared memory contract](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#distributed-shared-memory).
Cluster origin uses `blockIdx.z - cluster.block_index().z`; ordinary block
indexing and the per-block Phase 2 origin are preserved.

## Preserved numerical contract

Collision expressions, arithmetic order, FP32 state, byte-accessed flags,
SoA layout, and the final SCATTER writes are unchanged. `has_producer` tests
the flattened predecessor address, including the inherited Y-edge aliases;
it does not clip geometric producer coordinates. An invalid producer writes
`src[r,q]`, including across DSM. It never reads an uninitialized population
object, skips a consumed slot, or substitutes zero.

Source and destination allocations must be distinct. Immutable flags and
slots with no physical producer must have matching initial values and remain
invariant. An even-step temporal run holds states N and N−2 in its current
and inactive buffers; it does not materialize the odd intermediate globally.
The B0 oracle compares each buffer to its corresponding time, rather than
comparing inactive N−2 with the one-step implementation's inactive N−1.

## Mechanism counts, independently checked

The full box changes from 2×1080 to 1800 producer candidates per cluster,
a 16.67% reduction. The full physical domain has 2,280 clusters:

| Source-level quantity per fused launch | Count |
| --- | ---: |
| Independent D-V2 halo candidates | 4,924,800 |
| Deduplicated cluster candidates | 4,104,000 |
| Independent D-V2 valid first collisions | 4,536,000 |
| Deduplicated valid first collisions | 3,766,500 |
| Second collisions | 2,160,000 |
| Physical shared `(r,q)` slots | 41,040,000 |
| Valid local / remote stores | 37,988,884 / 2,726,880 |
| Fallback local / remote stores | 315,116 / 9,120 |
| All remote stores, each direction | 1,368,000 |

A full 16×8 internal interface has five crossing directions each way:
640 floats per direction, 1,280 total, or 5,120 bytes of source payload.
The whole-domain remote payload is 10,944,000 bytes per fused launch.
These logical counts are not hardware sectors, network traffic, or a speed
prediction. The full producer box is retained, including unused corners.

The independent [CPU checker](../experiments/z2/model.cpp) enumerates producer
ownership, inverse dependencies, destination ownership, and local/remote
fallback categories without calling the kernel coordinate decoder. The
[debug GPU harness](../src/cluster_ownership.cu) counts actual production-loop
visits and all 41,040,000 shared writers; it also checks representative
interior traffic in both directions. All array entries must equal one and
all category counts must match the CPU model.

## Build, validate, measure

Use the same Clang/CUDA toolchain as the original measurements. The selected
GPU must support cluster launches. With this Clang 23 snapshot and CUDA 12.9,
[cluster_compat.cuh](../kernels/cluster_compat.cuh) exposes the cooperative
cluster API using `_CG_CLUSTER_INTRINSICS_AVAILABLE`: Clang already provides
the native cluster intrinsics, but the installed CUDA header otherwise gates
the API on NVCC/NVRTC. Missing device intrinsics produce a compile-time error.
No numerical compiler flag or architecture target is changed.

```sh
export CUDA_CXX=/path/to/clang++
export CUDA_PATH=/usr/local/cuda-12.9
export CUDA_VISIBLE_DEVICES=0
make cluster
./build/lbm-z2 --list
./build/lbm-z2 D-V3-Z2-256 --check 6 --patterned --sentinels

# Use a fresh output directory to preserve shipped measurements.
export LBM_Z2_RESULTS="$PWD/results/z2-local"
make cluster-check
make cluster-bench
make cluster-profile
make cluster-report
# Or: make cluster-all
```

`cluster-check` runs full-allocation comparisons (populations, flag bytes,
unused flag-slot bytes, padding and margins), input immutability, sentinel
and flat-alias cases, default histories 1000/1002, patterned 1000, external
obstacles 100/102, and independent B0 velocity outputs at 2/6/100 steps.
Obstacles cover interfaces including z=3/4, 67/68, 147/148 and physical
boundaries. Memcheck, racecheck, synccheck and global initcheck each run six
physical steps with patterned, sentinel and obstacle input. Debug assertions
reject 192-thread and singleton-cluster launches. Production SASS is checked
for synchronization and absence of debug atomics/assertions.

Timing and profiling require hashes matching the completed gates. Event
trials use four shuffled 100-step rounds and two 1000-step rounds, with five
warm-ups and twenty trials/process. Both buffers are restored outside each
event interval. Times include launch gaps and are divided by physical steps.
The scripts serialize experiment stages with a file lock; run only one GPU
campaign on the selected device at a time.

Runtime resources include the cluster-specific occupancy API's **device-wide
maximum active cluster count** and ordinary blocks/SM. The unclustered
reference queries a singleton cluster for this diagnostic but still launches
without clustering. See NVIDIA's [cluster occupancy APIs](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__OCCUPANCY.html).

NSYS verifies each actual kernel's identity, count, grid, block and shared
allocation. NCU samples launch 11 (t=20→22) under flush and no-flush replay,
with installed DSM metrics discovered through `--query-metrics`. Additive
counters and duration are divided by two; resource counts, ratios and
percentages are not. No-flush replay is a sensitivity diagnostic, not event
timing. Large profiler binaries stay on disk but are ignored by Git; compact
CSV/JSON/log evidence is retained. Report generation reads saved evidence
without running GPU workloads.

The primary comparison is C2 / D-V2 time. D-V2→C0, C0→C1 and C1→C2 are
complete-kernel controls for placement, synchronization, and deduplication
plus DSM communication. Their time differences do not isolate additive
phase costs. TMA, rolling planes, asynchronous barriers, precision changes,
new tiles, worker counts and larger clusters are outside this experiment.
