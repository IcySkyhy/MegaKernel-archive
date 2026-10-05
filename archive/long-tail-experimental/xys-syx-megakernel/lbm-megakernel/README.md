# How far can temporal fusion take LBM?

A CUDA worklog on a concrete optimization: **keep the state between two dependent LBM timesteps on chip instead of materializing it in global memory.** The goal is to establish a correct, profitable hand-written transformation and identify what an MLIR compiler would need to reproduce it.

The workload is D3Q19 FP32 Lattice Boltzmann on a fixed **120 × 120 × 150** lid-driven cavity, derived from the [Enzyme-GPU-Tests LBM implementation](https://github.com/wsmoses/Enzyme-GPU-Tests/tree/mlir/LBM). This repository builds and runs independently of that project and Enzyme.

**Original 20-variant campaign: best measured `D-V2-256` at 135.82 µs per physical timestep—2.14× speedup over the original `B0` and 2.07× over its matched one-step control, `S2-D256`.** For 1000 timesteps, the measured interval is 135.82 ms rather than B0's 290.13 ms, including GPU launch gaps but excluding setup and output.

This is **two-step temporal fusion**, not a whole-simulation megakernel. A 1000-step simulation still uses 500 ordered temporal launches. The result demonstrates a useful transformation, not an optimal schedule or a measured performance ceiling.

## What changes—and what does not

B0 already fuses collision and streaming **within one timestep**. The new transformation crosses the boundary **between two timesteps**:

```text
Two ordinary launches

    global f(t)
        -> collision + scatter
        -> global f(t+1)
        -> collision + scatter
        -> global f(t+2)

One two-step launch

    global src = f(t)
        -> first collision over a tile's required producers
        -> block-local shared intermediate
        -> block barrier
        -> second collision over the owned tile
        -> original scatter into global dst = f(t+2)
```

Both temporal representations remove the global odd-step intermediate. Neighboring blocks recompute overlapping first-step halo producers so that each block can complete its second collisions independently. The collision equations, FP32 storage, boundary behavior, and final scatter are preserved.

The final `dst` is distinct from `src`: a block must not overwrite input that another block may still need. The host advances by two physical timesteps and updates the buffer roles between launches. The intermediate is local to each block, not a third full-grid global state.

The main experimental questions are therefore:

| Question | Comparison | Observation in this campaign |
|---|---|---|
| Is spatial remapping alone enough? | C1–C3 versus B0 | The best of these controls reduces time by 0.72%. |
| Does two-step fusion help? | S2-D256 → D-V2-256 | 281.69 → 135.82 µs/step; 2.07× speedup. |
| Does a smaller shared representation give a proportional gain? | D-V1-256 → D-V2-256 | 82,080 → 38,912 B/block, but only 1.01% less time. |
| Does the smallest halo produce the fastest tile? | Cube versus elongated V2 | The cube has fewer nominal producers, but its best measured configuration is slower. |
| Is a warp-wide producer row sufficient? | T30 versus elongated V2 | T30 helps at 128 workers, but does not beat the best elongated configuration. |

These comparisons evaluate complete implementations. They do not isolate the cost of an individual instruction, memory transaction, or synchronization mechanism.

## Measurements

**Platform:** NVIDIA GeForce RTX 5090, driver 575.57.08, CUDA 12.9, Clang 23.0.0git, target `sm_120`, with `-O3 -ffast-math`. Measured on 2026-09-24 using the default cavity without an obstacle file.

The tables report pooled medians from **40 measured trials per variant at 1000 physical timesteps per trial**, across two seeded shuffled process rounds. Each process performs five warm-ups and twenty measured trials. Both device buffers are restored outside each timed interval. Allocation, restoration, validation, and output are excluded; GPU launch gaps are included.

All numbers below come from the **fresh standalone-repository campaign**. They are not mixed with earlier prototype timings or profiler counters. Consolidating kernels into templates can change code generation, so the variants were rebuilt and retimed together.

### One-step controls

| Kernel | µs / physical step ↓ | Speedup over B0 ↑ |
|---|---:|---:|
| B0 | 290.13 | 1.00× |
| C1 | 289.42 | 1.00× |
| C2 | 290.59 | 1.00× |
| C3 | 288.04 | 1.01× |
| S2-C128 | 302.87 | 0.96× |
| S2-D128 | 290.34 | 1.00× |
| S2-D256 | 281.69 | 1.03× |

### Two-step fusion: selected configurations

| Kernel | µs / physical step ↓ | Speedup over B0 ↑ |
|---|---:|---:|
| D-V1-128 | 149.03 | 1.95× |
| D-V1-256 | 137.21 | 2.11× |
| D-V2-128 | 149.34 | 1.94× |
| **D-V2-256** | **135.82** | **2.14×** |
| C-V2-256 | 142.12 | 2.04× |
| T30-V2-128 | 144.94 | 2.00× |

Speedup over B0 is `B0 time / candidate time`; a value below 1 means slower. For the temporal transformation, also compare against the **same-shape, same-worker one-step control**. The 2.14× headline is the overall improvement; the 2.07× matched comparison keeps the nominal tile shape and worker count fixed, but still includes all changes introduced by temporal fusion.

The [complete benchmark table](docs/BENCHMARKS.md) includes both worker counts and the 100-step cross-check. [Raw event trials](results/rtx5090/events.json) retain execution order, rounds, build identity, and GPU telemetry. Twenty trials within one process are not twenty independent process replicates. Treat small percentage differences as descriptive; two process rounds do not establish statistical significance.

### Variant names

`B0` is the original one-step kernel. `C1`–`C3` are direct thread/block remappings. In tile-based names, `C` denotes the **8×8×8 cube**, `D` the **16×8×4 elongated tile**, and `T30` the **30×4×4 tile**. A worker suffix such as `128` or `256` gives the CUDA thread count per block.

**`S2` names an experiment stage, not two timesteps.** S2 kernels are one-step matched tile controls. `V1` and `V2` both execute two timesteps; they differ in shared-memory representation.

## 1. Preserve the baseline, then change spatial ownership

### B0: one thread per cell

[`b0.cuh`](kernels/b0.cuh) assigns one block to each X row: 120 threads per block and a `120×150×1` grid. Each thread reads its cell's 19 populations, applies the original bounce-back or fluid-collision behavior, and scatters the results to directional neighbors. All one-step variants call the same [cell body](kernels/cell.cuh).

The inherited layout is padded FP32 **structure of arrays (SoA)**: for a fixed population direction, consecutive X cells are consecutive values. Flags are accessed as bytes.

The one-step useful population payload is:

```text
19 FP32 reads + 19 FP32 writes
    = 152 B / cell / physical step
    = 328.32 MB / full-domain update
```

This excludes flags and is **not measured DRAM traffic**. Cache behavior, transactions, and the transformed schedule affect actual traffic. An equivalent useful-payload bandwidth must not be reported as physical DRAM bandwidth.

### C1–C3: direct spatial remapping

| Kernel | CUDA block | Grid | Ownership change |
|---|---|---|---|
| [C1](kernels/c1.cuh) | 32×4×1 | 4×30×150 | One warp-wide X segment across four Y rows. |
| [C2](kernels/c2.cuh) | 32×2×2 | 4×60×75 | Distribute the rows across Y and Z. |
| [C3](kernels/c3.cuh) | 64×2×1 | 2×60×150 | Two warp-wide X segments across two Y rows. |

All use 128 workers and one nominal site per worker, with physical-domain guards. There is no shared intermediate or temporal reuse. The last X tile has 24 valid columns for C1/C2 and 56 for C3; changing the launch geometry does not eliminate edge waste.

These controls give small timing changes, not the twofold improvement seen with temporal fusion.

### S2: separate tile shape from worker count

[`S2-C128`](kernels/s2_c128.cuh) assigns an `8×8×8` tile to 128 workers; [`S2-D128`](kernels/s2_d128.cuh) assigns `16×8×4`. Their grids are `15×15×19` and `8×15×38`, respectively.

Both tiles contain 512 nominal sites. An X-fast strided loop assigns four sites per worker with 128 threads, or two with 256 threads, before physical-boundary masking. **The logical tile describes cell ownership; the CUDA block describes the workers processing it.**

These kernels still read and write global populations every timestep. Their purpose is to measure the mapping cost before adding a local intermediate. In particular, `S2-D256` is the matched one-step control for `D-V1-256` and `D-V2-256`.

## 2. The temporal transformation: own the second collision

A tile `R` denotes the cells whose **second-step collisions** belong to a block. It does not denote all final output populations at those geometric cell locations.

For an interior cell, the original scatter update has the form:

```text
f_q(t+1, p + c_q) = collision_q(f(t, p), flags[p])
```

To perform the second collision at `r ∈ R`, the block needs the 19 intermediate populations at `r`. Population `q` comes from first-step producer `r − c_q`. The required first-step producer region is therefore:

```text
P = union over q of (R - c_q)
```

A one-cell-expanded box contains this region. For `R = 16×8×4`, the implemented producer box is `18×10×6`. Both temporal versions retain the full box, including its eight unused corners; physical-boundary cases follow the inherited flattened-address rules described below.

Each block recomputes its required first-step producers. That duplication removes the need to obtain intermediate values from neighboring blocks. After a block barrier, the second collisions can run locally. Ordered kernel launches still separate successive two-step updates.

## 3. V1: store post-collision producer state

[`temporal_v1.cuh`](kernels/temporal_v1.cuh) stores the first collision's results at their **producer locations**:

```text
Phase 1: load first-step producers from global src
         -> collide in registers
         -> store post-collision populations in shared producer storage

Block barrier

Phase 2: for each r in R, gather population q from producer r - c_q
         -> second collision
         -> original scatter into global dst
```

For the elongated tile, the shared allocation is:

```text
19 populations × 18 × 10 × 6 producer positions × 4 B
    = 82,080 B / block
```

This permits only one resident block per SM on the tested GPU. Nevertheless, `D-V1-256` reaches **137.21 µs/step**. The major gain is already present before reducing the shared footprint: the odd timestep is no longer materialized as a global array.

V1 does not remove halo work. It accepts redundant first-step computation in exchange for block-local communication between the two updates.

## 4. V2: store the streamed intermediate on R

[`temporal_v2.cuh`](kernels/temporal_v2.cuh) uses the same producer region but stores only the populations needed by the second collisions:

```text
Phase 1: collide a first-step producer p in registers
         -> for each q, compute destination r = p + c_q
         -> when r is in the valid core, write intermediate_R[q][r]

Block barrier

Phase 2: read all 19 populations locally at r from intermediate_R
         -> second collision
         -> original scatter into global dst
```

Missing physical producers use the invariant-slot fallback rather than leaving a consumed shared slot uninitialized.

The shared allocation is now:

```text
19 populations × 16 × 8 × 4 core positions × 4 B
    = 38,912 B / block
```

The complete input halo is not staged in shared memory: input values pass through registers, and only the streamed intermediate on `R` persists between phases.

| Elongated representation | Shared bytes/block | Reported residency limit | 128-worker µs/step | 256-worker µs/step |
|---|---:|---:|---:|---:|
| V1: post-collision producer state | 82,080 | 1 block/SM | 149.03 | 137.21 |
| V2: streamed intermediate on R | 38,912 | 2 blocks/SM | 149.34 | 135.82 |

V2 reduces time by **1.01%** at 256 workers; it is slightly slower at 128 workers. Both versions already remove the same global intermediate. V2 changes addressing, store predicates, register use, and shared transactions as well as capacity.

**A smaller footprint enables more residency; it does not determine execution time.** This comparison neither isolates occupancy's contribution nor proves that V1's shared accesses were unimportant.

## 5. Tile shape: balance producer work and execution mapping

The shape experiments keep two-step FP32 evolution but change the region owned by a block and the resulting producer traversal.

| Shape | Owned second-collision tile | Nominal core sites | Producer box | V2 shared bytes/block | Fastest measured V2 workers | µs/step |
|---|---|---:|---|---:|---:|---:|
| D: elongated | 16×8×4 | 512 | 18×10×6 | 38,912 | 256 | **135.82** |
| C: cube | 8×8×8 | 512 | 10×10×10 | 38,912 | 256 | 142.12 |
| T30 | 30×4×4 | 480 | 32×6×6 | 36,480 | 128 | 144.94 |

### Cube: fewer halo producers are not sufficient

The cube's full producer box contains 1000 positions rather than the elongated tile's 1080, for the same 512-site core. That is a favorable geometric tradeoff, but its fastest V2 configuration is still slower.

The X extent also shrinks from 16 to 8 in an X-contiguous SoA layout. The shape changes the address pattern, partial tiles, and worker mapping together. The result is evidence against choosing a tile from halo size alone—not proof that one particular transaction pattern caused the entire difference.

### T30: a warp-wide halo row is not the same as aligned access

T30 chooses a 30-wide core so that its producer box is 32 sites wide. The halo traversal places one producer row within a warp, but **the starting global address is not automatically sector-aligned**. The flat second-phase traversal also splits 30-site core rows across warps.

At 128 workers, T30 V2 reduces time by **2.95%** relative to D V2. At 256 workers, it increases time by **8.83%**. Its best configuration is therefore the 128-worker version, which still does not beat `D-V2-256`.

The practical conclusion is to choose **tile shape and worker mapping together**. Minimum halo size, minimum shared storage, and a warp-wide producer row are useful design considerations, but none alone selects the fastest measured implementation.

<!-- Z2_SUMMARY_BEGIN -->
## 6. Z2 clusters: replace duplicate halo work with DSM stores

This follow-up keeps D-V2’s 16×8×4 tile, 256 workers, FP32 arithmetic and
38,912 B shared/block. Two neighboring Z blocks form a 16×8×8 cluster tile.
C2 assigns each first-step producer to one block and streams crossing populations
through DSMEM. Two cluster barriers protect and order those writes; Phase 2 stays local.

**Fresh matched campaign; 1000-step event medians:**

| Variant | Change | µs/physical step | Time / D-V2 |
| --- | --- | --- | --- |
| D-V2-256 | Ordinary D-V2 | 135.681 | 1.000× |
| D-C0-Z2-256 | C0: cluster launch only | 143.653 | 1.059× |
| D-C1-Z2-256 | C1: two cluster syncs | 146.022 | 1.076× |
| D-V3-Z2-256 | C2: deduplication + DSMEM | 140.442 | 1.035× |

C2 removes 16.7% of candidate producer visits and improves on C1 by
3.82%, but takes **3.51% longer than D-V2**.
All four variants pass full-allocation bitwise, long-history and sanitizer gates.
The counts prove DSM communication replaces duplicated work; the timing shows
why a compiler still needs a profitability model. These are complete-kernel
controls, not additive phase timings. Cluster C1/C2 names are separate from
the older one-step C1/C2 variants.

[Implementation and reproduction](docs/Z2_DESIGN.md) · [Full evidence](docs/Z2_RESULTS.md)
<!-- Z2_SUMMARY_END -->

<!-- CLUSTER_SWEEP_SUMMARY_BEGIN -->
## 7. Scheduling policy and XZ4: measure cluster cost first

The follow-up first compares explicit Spread and LoadBalancing, then runs
C0 for Z2, XZ4 and YZ6. Every C0 uses the unchanged D-V2 device function.
LoadBalancing materially reduces Z2 C0 cost; the required Z2 C1/V3 rerun
still does not beat ordinary D-V2. XZ4’s C0 cost is close enough to Z2 to
justify a four-block prototype, so it was implemented with 850 uniquely
owned producers/block and explicit X/Z/XZ remote-write validation.

**Fresh final campaign; all cluster rows below use LoadBalancing:**

| Variant | µs/physical step | Time / ordinary D-V2 |
| --- | --- | --- |
| D-V2-256 | 136.569 | 1.000× |
| D-V3-Z2-LoadBalancing | 139.632 | 1.022× |
| D-C0-XZ4-LoadBalancing | 138.393 | 1.013× |
| D-C1-XZ4-LoadBalancing | 145.282 | 1.064× |
| D-V3-XZ4-LoadBalancing | 153.386 | 1.123× |

The best cluster still takes 2.24% longer than ordinary D-V2.
XZ4 saves more producer work, but its DSM interface bytes grow about 2.6×
versus Z2 for only 25% more logical remote payload. All correctness gates
pass. Following the guide’s stop condition, Tt=2 cluster tuning stops here;
pointer hoisting was not added. Deeper temporal blocking is a future experiment.

[Full staged results](docs/CLUSTER_SWEEP_RESULTS.md) · [Kernels and reproduction](docs/CLUSTER_SWEEP_DESIGN.md)
<!-- CLUSTER_SWEEP_SUMMARY_END -->

## Correctness is part of the transformation

The numerical expressions and their order are preserved under the recorded compiler settings. A source gate checks the [temporal collision body](kernels/collision.cuh) against B0. Full-allocation comparisons include populations, flag bytes, unused flag-slot bytes, padding, and margins. No error tolerance is used.

The implementation must preserve four obligations:

- **Immutable input during a fused launch.** Source and destination storage are distinct; final stores must not overwrite another block's unread halo inputs.
- **Inherited boundary semantics.** With zero Y padding, an out-of-range Y coordinate can alias a physical site in an adjacent Z plane. Producer validity uses the flattened address, not geometric XYZ clipping. Never-produced slots retain matching initial values in both buffers, and flags remain immutable.
- **Complete local intermediate.** Every physical `(r, q)` shared slot has one writer, including the fallback for a missing producer. All workers reach the block barrier before consumption.
- **Unique final writes.** For fixed `q`, scatter is injective. The owner of collision site `r` writes its result at `r + c_q`, even when that destination lies outside the geometric tile. Restricting final writes to destinations inside `R` would drop required outputs.

The shipped campaign passed all **20 variants**: full-allocation bitwise comparisons, velocity-output parity, invalid-count checks, default histories at 1000/1002 steps, patterned input at 1000, and external obstacles at 102. All variants also passed memcheck, racecheck, synccheck, and global-memory initcheck on six-step sentinel/obstacle inputs. B0 velocity output at 2/6/100 steps matches an independently built original executable.

See the [validation contract](docs/VALIDATION.md) and [saved gates](results/rtx5090/validation.json). These tests cover the recorded cases; they are not a proof for arbitrary inputs, horizons, or changed boundary rules. Obstacle tests extend correctness coverage; the reported performance uses the default cavity.

## Build and reproduce

Requirements: Linux, Python 3.9+ with the standard library, CUDA 12.9, a CUDA-capable Clang supporting `sm_120`, and a compatible NVIDIA GPU. The recorded compiler is Clang 23.0.0git at LLVM commit `c813428fb1b5678a6e4c541f4aee82a7977e95ea`. Other toolchains and architectures require their own validation and measurements.

Nsight is not required for building or event timing. Compute Sanitizer is needed only for the optional sanitizer checks.

Run from `lbm-megakernel/`:

```bash
export CUDA_CXX=/path/to/clang++
export CUDA_PATH=/usr/local/cuda-12.9
export CUDA_VISIBLE_DEVICES=0

make
./build/lbm --list
./build/lbm S2-D128 --check 100
./build/lbm D-V2-256 --check 1000 --patterned
./build/lbm D-V2-256 100 -o velocity.dat

make check
make bench

# Full history matrix and four sanitizers used for the shipped validation:
python3 scripts/validate.py --full --sanitizers

# Generate a report from local measurements:
python3 scripts/report.py --results results/local --output docs/LOCAL_RESULTS.md
```

The scripts default to ignored `results/local/` and preserve shipped `results/rtx5090/` measurements. Benchmarking requires passing validation, rejects a changed binary/source snapshot, and refuses to overwrite existing event trials. Use a new `--out` directory for a new campaign. `make report` regenerates the shipped summary without running GPU work.

## What should an MLIR compiler learn?

The target is not simply to concatenate kernels or move the host loop onto the device. It is a dependence-aware **two-timestep temporal tiling transformation**:

```text
Identify two dependent collision–streaming updates
    -> tile the second update's collision sites
    -> derive their required first-update producers
    -> duplicate halo work deliberately
    -> replace global intermediate communication with workgroup storage
    -> insert the block-local synchronization
    -> preserve boundary values, final scatter ownership, and live-outs
    -> choose a profitable tile, representation, and worker mapping
```

The hand-written kernels provide executable references and matched controls. The [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) supplies launches, workgroup memory, and barriers; establishing legality and selecting a profitable realization are the compiler work. See the [pass roadmap](docs/MLIR_FUSION.md).

A whole-simulation megakernel requires an additional dependency and synchronization strategy. The Z2 cluster/DSMEM experiment above tests replacing duplicated halo work with communication. Longer temporal tiles, rolling storage, input staging, and submission controls remain separate experiments. Reduced precision changes the numerical contract. None of those follow-ups is credited with the original two-step speedup.

**The central result is that localizing an intermediate between dependent timesteps can matter much more than changing a one-step launch geometry.** The remaining choices are how much producer work to duplicate, what to retain on chip, and how to map that work efficiently—not simply how many kernels to launch.

## Provenance

The worklog format is inspired by [Si Boehm's CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM): show the kernel change, the result, and the lesson. Upstream notices are preserved. [NOTICE](NOTICE) and [provenance](docs/provenance.json) identify the imported code and source snapshot.
