# blackwell-fp4-ffn

Pushing FP4 matrix throughput to its ceiling on a consumer Blackwell GPU
(RTX 5070 Ti, sm120). Accuracy is a low priority, lower precision accumulate is
acceptable. The measuring stick is the residual double GEMM at 4096 square
(D2 = W2*(W1*X) + C); the end goal is a fused transformer block (residual,
RMSNorm, SwiGLU) at the peak rate.

## Environment

Hardware: RTX 5070 Ti, GB203, sm120a, 70 SM, 16 GB GDDR7 at ~896 GB/s, max SM
clock 3210 MHz, 100 KB shared memory per block, 15 GB system RAM (max 3
parallel CUTLASS compiles, ~3.2 GB each).
Software: CUDA 13.1 (V13.1.115) and CUTLASS 4.4.2 in a sibling checkout, referred
to below as `$CUT` (not vendored), plus a 13.3 ptxas from pip for the sparse
path (recipe below).

## Results (2026-07-02)

The headline of this repo is the fused row, and its re-measured value is 858,
not the 862 shown below. On 2026-08-31 the dense, sparse and fused d=8 rows
were re-measured from a clean build under a clock lock, and the fused d=8 row
came back at 858.6 / 857.3 / 857.0, lower than the 862 first claimed here. 858
is the number the repo description quotes. The FFN block figure further down
was re-measured in the same session, and no other throughput number on this
page has a committed log. The two marked rows below, and the sections after
this one, are exploratory runs at a different epilogue configuration, a
different residual precision, a different shape or a different tile geometry.
[Reproducing the headline numbers](#reproducing-the-headline-numbers) has the
re-measured table, the conditions and the raw stdout.

Residual FP4 double GEMM, 4096 square, 3090 MHz lock, interleaved same-thermal
A/B:

| variant | ms/pair | TFLOPS | note |
| --- | --- | --- | --- |
| dense chain, raster AlongM | 0.404 | 680 | best dense config |
| 2:4 sparse chain, row-major C/D | 0.336 | **819** | dense-equivalent, +20% |
| fused sparse chain, one kernel, d=8 | 0.320 | 862 | +4.2% vs champion same session (827) |
| fused chain, deep-C epilogue (2026-07-03) | 0.3147 | 873 | StagesC=3/StagesD=1, bf16 C. Not reproduced on 2026-08-31 |
| fused chain, fp4 residual stream | 0.3131 | 878 | -DC_FP4, StagesC=8, C=e2m1. Not re-measured |

Bigger shapes, same kernel (fresh 200/120-iter runs; ON = residual on, OFF =
beta2off wall):

| shape | ON bf16-C | ON fp4-C | OFF wall | ON/wall |
| --- | --- | --- | --- | --- |
| 6144sq | 909 | **918.6** | 939.8 | 97.7% |
| 8192sq | 896 | **912.3** | 924.5 | 98.7% |

1000-iter soaks (fp4-C, heat-soaked sustained): 895 at 6144, 852 at 4096.

### The 256-tile unlock (2026-07-03, same day, hours later)

The walls above fell too. ncu on the deep-C wall kernel showed L2 at 93% of
peak with tensor pipes at 59): the real ceiling was L2 *bandwidth* (every tile
re-reads its A-row/B-col through L2; sparse metadata+SF ride the same port),
not MMA rate and not the epilogue. The fix is arithmetic intensity: TBM=256
halves B re-reads. The "tile fixed at 128x128x256" claim was a cute template
bug, not a hardware limit: bw_coalesce on the two-atom (M=256) metadata layout
instantiates `int == cute::R<1,16>` inside a statically-false branch.
`cute_scaledbasis_eq.patch` (one guarded if-constexpr in cute's ScaledBasis
operator==, semantics preserved) unblocks it; correctness verified via CUTLASS
example 80b reference check at 4096/6144 with the 256 tile.

Fused chain, 256x128x256, fp4-C sC=16/sD=1, d=2, 3090 lock, fresh:

| shape | ON | OFF wall | ON/wall | soak ON |
| --- | --- | --- | --- | --- |
| 6144sq | **1069.5** | 1092.0 | 98% | 1033.9 (1000 it) |
| 8192sq | **1045.0** | 1079.4 | 97% | 1024.0 (400 it) |

At 4096 the fat tile loses to wave quantization (14.6 waves of 2x-long tiles):
831 vs 878 - keep 128x128x256 there. Tile is now a per-shape knob: -DTBM=256
for >=6144. bf16-C at 256-tile: 1066.7 @6144 (C stream stops mattering once L2
has headroom) but 960 @8192 (it matters again at 128MB C); fp4-C dominates.
Geometry frontier, all measured to fail: 256x256x256 (smem, StagesB>=2 assert),
128x256x256 (B-side make_tma_copy break), 256x256x128 + 384x128x256 (smem-atom
tile_to_shape divisibility). Mem-clock down-lock (power shift): flat. After the
tile, ncu shows L2 76% / tensor 66% - neither saturated; the next wall is a
mixed feed-rate regime, and tensor@66% says ~1600 is what a perfectly-fed
OMMA.SF.SP could do at this clock.

Meta-lesson, standing bias: every "unreachable/fixed/ceiling" in this file is a
claim about assumptions tried, not physics. Two such claims fell in one day
(901-unreachable -> StagesC default; tile-fixed -> cute bug). Re-derive the
limiting assumption before repeating a verdict.

The 2026-07-03 "~870 ceiling at 4096" claim is retired. The flat +20 us
residual read was never DRAM bandwidth (the chain runs at ~250 of 896 GB/s and
C2 hits L2 at 94%): it was the epilogue C pipeline. The sm120 CollectiveBuilder
hardcodes StagesC=2, so the epi-load warp can only run 2 subtiles (of 8 per
tile) ahead and every GEMM2 tile pays exposed C latency inside its epilogue
window. Instantiating Sm120TmaBuilderImpl directly with StagesC=3/StagesD=1
(D is fp4, one store stage is free; 101376B smem budget exactly holds
87040 mainloop + 14336 epi) spreads C loads across the mainloop window:
tax 18.6 -> ~13 us at 4096, 52 -> 35 at 6144, 153 -> 114 at 8192. Shrinking the
stream to fp4 C (-DC_FP4, StagesC=8 = full-tile run-ahead, stage = 1KB) buys
the rest at scale: tax ~13 us at 8192. Failed routes, measured: persisting-L2
window on C2 is a no-op (TMA descriptors ignore the access policy; hit rate
94.57 vs 94.56), fp8/fp4 C at StagesC=2 is *slower* than bf16 (convert ALU in
an epilogue that already does the FP4+SF quantize; -15/-11 TF), locking 3210
MHz loses to 3090 (clock flutter under FP4 load). What remains vs the wall is
~10-15 us of epilogue ALU floor (the C add shares issue slots with the MMA
consumer warps) plus the sparse structure wall itself. d-interleave is flat
1..32 with the deep pipeline (d=1 no longer pathological); d=2 default.

Sparse single GEMM (dense-equivalent): 782 TFLOPS at 4096 cube, 955 at 8192.
Dense single GEMM best: 710 at 8192 cube. Dense knob space is exhausted at
~80% of the ~870 TFLOPS clock-adjusted dense peak; sparsity is the only physics
above that wall and it works now.

Fused FFN block (`fp4_ffn_block.cu`, dense, T 4096, d_model 4096, d_ff 14336,
clock locked 2700): 551 TFLOPS effective, gated by the 234 MB bf16 gate+up
intermediate (the SwiGLU reread runs at 90% of the memory wall). Fusing SiLU
into the GEMM epilogue is the fix. Relative Frobenius error 0.24 vs fp32.

### N-layer forward: the rate holds through depth (2026-07-03)

The 1069/1045 champion is ONE residual double-GEMM (one layer). Was the >1000
number ever over N layers? No — the 8-layer chain lived only in the fp32 cuBLAS
reference trainer. `fp4_nlayer_fwd.cu` chains L fused residual double-GEMMs
(distinct weights/layer = L2-cold, ping-pong activations, residual = layer input,
per-layer gate check). Locked 3090, fp4-C, dense-equiv TFLOPS:

| S | L=1 | L=4 | L=8 | L8/L1 |
| --- | --- | --- | --- | --- |
| 4096 (128t) | 866 | 841 | 837 | 96.6% |
| 6144 (256t) | 1099 | 1050 | 1041 | 94.7% |
| 8192 (256t) | 1056 | 1023 | 1019 | 96.5% |

Per-layer time is ~constant with depth (8192: 2.08 -> 2.16 ms/layer over 8x); the
small L=1->L=2 dip is losing single-weight L2 residency, flat from L=4. An
8-layer forward sustains 95-97% of the single-layer rate (rate is preserved: 8
identical layers = 8x work in ~8x time). This chains fp4->fp4: a valid THROUGHPUT
PROXY (exact FLOPs+traffic) but see the deployable section for real numerics.
Build: `build_sparse.sh fp4_nlayer_fwd.cu /tmp/nlayer -DTBM=256 -DC_FP4` (argv:
n iters L swz gate d).

### The numerically-valid deployable N-layer forward (2026-07-04)

`fp4_nlayer_deploy.cu`. Two earlier claims fall here.

FIRST: the fp4->fp4 chain is NOT numerically invalid. The old 0.44 rel-err was
(a) row-major D putting the output block-scale on the wrong axis + (b) a vec16
assumption (the sparse nvf4 sm120 SF vector is **32**, sm1xx_common.inl:170).
With **column-major D** the epilogue col-store writes SF via
`Sm1xxBlockScaledOutputConfig<VS,Major::MN>`, whose layout is byte-identical to
the mainloop SFB layout (`layout_probe.cu`: SFD_MN[m,n]==SFB[n,m], 0 mismatch).
`probe_chain.cu` (fixed to vec32) then passes CHAIN D2 rel_frob **0.10** = fp4
noise floor.

SECOND: fusing the requant into the epilogue (col-major D, `fp4_nlayer_valid.cu`)
is SLOWER, not faster: 627/729 @6144/8192 vs the proxy 1044/1019 -- the sub-byte
col-major fp4 store defeats the MMA pipe. (Col-major *bf16* out is free,
1202/1248 == row-major.) So the fast valid path is bf16 GEMMs + a dedicated
coalesced fp4 requant kernel + a bf16 residual stream (raw fp4 residual is
unscaled => invalid). The GEMM's fp4 B is K-contiguous (stride N,1) so requant is
transpose-free: one thread per (n,32-k block), uint4 load + ue4m3 SF + uint4
packed store, ~0.4ms/pass @8192.

Locked 3090, L=8, dense-equiv (requant included), validated vs fp32 (relPJ grows
~sqrt(L): L=1 0.086 -> L=8 0.239):

| S | valid deploy | invalid proxy | valid/proxy |
| --- | --- | --- | --- |
| 4096 (128t) | 584 | 837 | 70% |
| 6144 (256t) | **773** | 1033 | 75% |
| 8192 (256t) | **860** | 1016 | 85% |

Flat with depth. The gap = the 2 requant passes/layer (each GEMM output must be
re-quantized with a fresh block scale for the next GEMM; memory-bound, exposed
serially -- the irreducible deployable tax of a valid fp4 chain). Build:
`build_sparse.sh fp4_nlayer_deploy.cu /tmp/nd -DTBM=256` (argv: S iters L
[validate]; validate=1 at small S runs the fp32 host check).

## Training the flipped identity (2026-07-03)

Can we hold the lead while *training* the 8-layer residual double GEMM on the
flip (output[i] = input[S-1-i])? `train_flip_ref.cu` (cuBLAS fp32 + NVFP4 sim),
`probe_chain.cu` (chain numerics), `tput.cu` (throughput components),
`build_sparse.sh` (reusable splice).

Accuracy — the net is linear so P = ∏(I + W2_l W1_l); want P = J (reversal).
- L=1 solves exactly (W2W1 = J−I). L=8 plateaus (loss ~0.02) despite equal
  expressivity: a determinant-sign barrier. J's 2D blocks are swaps (det −1);
  the balanced "each layer does J^{1/8}" solution is a real 8th root of a
  reflection, which does not exist, so GD stalls short.
- Fix = homotopy: ramp the target I→J over training. relPJ 0.0006 (exact).
  SOTA recipe: AdamW 0.9/0.95 wd 0, init 1/√S + zero residual branch, warmup +
  cosine, grad-clip, homotopy (the key ingredient).
- fp4: PTQ relPJ 0.234 → QAT (fp4 fwd + STE, fp32 master) 0.131, b=0.981. Flip
  is structurally perfect; residual is the e2m1 noise floor.

Throughput (locked 3090, 8192, dense-equiv): sparse GEMM bf16-out 1205, sparse
fp4-out 1132, dense fp4 712, 2:4 compress 0.45ms/weight (47% of a GEMM).
- Correct-numerics forward (bf16 intermediate + requant) is FASTER than the
  fused champion (1205 > 1045) and valid — the fused fp4→fp4 chain is
  numerically invalid (SFD≠SFB layout, D row/col mismatch; `probe_chain` = 0.44
  rel err). Chaining requires requant between layers.
### The backward / full training step, MEASURED (2026-07-04)

`bwd_probe.cu` composes the step from real kernels; `train_flip_24.cu` adds real
2:4 pruning (per-row/block/transposable) to measure accuracy. This corrects the
earlier ~930 *projection*. Graph: `backward_bound.html`.

Components (S=8192, 3090 lock, dense-equiv): sparse GEMM 1268, dense dW 711,
**dense dW half-N = 1386** (the structured-sparse-dW 2× ceiling, 1.95×), weight
compress 0.46ms, activation requant 0.21ms (799 GB/s). Step / layer (6 GEMMs):

| scenario | GEMM-only | +compress | DEPLOYABLE (+4 requant) |
| --- | --- | --- | --- |
| plain 2:4 (bwd dense) | 833 | — | 716 |
| transposable 2:4 | 1006 | 940 | **839** |
| + structured-sparse dW | 1305 | 1196 | **1038** |

Three findings overturn the projection:
1. The structured-sparse dW 2× (1386) is real but needs a **block-2:4** mask
   (per-row output-sparsity can't feed a shared B tile — proven). Block-2:4
   **collapses the flip**: relPJ 0.82 @blk128 vs 0.34 per-row / 0.40 transposable
   / 0.17 dense (b 0.42 vs 0.90). Channel permutation can't save a permutation
   target; may hold on real redundant nets (untested).
2. The requant tax is real and does **not** overlap (measured 3% hidden — the
   bf16-out write + 799 GB/s requant saturate memory). It knocks transposable
   940 → 839.
3. So the **binding constraint is the dense dW gradient GEMM** (711, the dense-fp4
   ceiling, ~40% of the step). The accuracy-viable path (transposable, relPJ 0.40)
   caps at **~840 < 900**. >900 needs a sparse dW (block masks → accuracy break) OR
   fused fp4-out GEMMs to kill requant (~894, still short: dW stays dense). A
   genuine accuracy↔throughput tension at the 900 line; forward/inference >1000
   stands, training is a harder profile.

## The sparse unlock

CUDA 13.1 ptxas refuses the sm120 sparse FP4 MMA (a toolkit gap, not silicon).
The 13.3 ptxas from the pip wheel assembles it and the 13.1-era driver runs the
cubin. Two traps cost real throughput:

1. Only block-scaled sparse (`OpClassBlockScaledSparseTensorOp`, SASS
   `OMMA.SF.SP...4X`, CUTLASS example 80b type) runs at the marketed rate. The
   plain sparse path (`MMA.SP`, example 83 type) is quarter-rate on GeForce:
   325 dense-equiv, half of dense. Measured, do not use.
2. Example 80b's column-major C/D costs 10% in the chain epilogue. Row-major
   gives 819 vs 743.

The sparse tile default is 128x128x256; 256x128x256 works after
`cute_scaledbasis_eq.patch` (see the 256-tile section), other geometries fail
(measured list above). stream-K loses ~4%, raster and swizzle are near flat.

## Files

- `fp4_sparse_chain.cu`: the two-kernel champion. Sparse residual double GEMM,
  weights as the compressed sparse A operand, GEMM1 emits FP4 + scales, GEMM2
  reads that buffer as B and adds the residual via beta. Throughput-only
  (prepared input scales, accuracy parked). argv: n iters swizzle raster.
- `fp4_fused_chain.cu` + `make_fused_kernel.py`: the fused best. Epilogue is
  built via Sm120TmaBuilderImpl directly: `-DSTAGESC/-DSTAGESD` override the C/D
  pipeline depths (defaults 3/1 bf16-C, 8/1 with `-DC_FP4` fp4-C). Both GEMMs in
  one persistent kernel: the patcher copies CUTLASS's sm120 sparse kernel and
  doubles the scheduler N domain, second half is GEMM2 with its own param set;
  GEMM2 B-loads gate on per n-stripe counters GEMM1 bumps after its store
  (ordering is real, gate measured free). Launch order must put producers
  before consumers (CLC steals are strict launch order): AlongM forced, and
  stripe interleave needs swz=1. argv: n iters swz ras gate d beta2off pf,
  best d=8.
- `fp4_chain_cutlass.cu`: the dense residual double GEMM baseline. `-DRESIDUAL`
  for the residual, tile via `-DTBM/-DTBN/-DTBK`, argv: n iters swizzle raster.
- `fp4_ffn_block.cu`: the fused SwiGLU FFN block, the end-goal deliverable.
  Builds with `-DFFN_BLOCK`, flags `--t --dm --dff`.
- `fp4_cutlass.cu`: single dense NVFP4 GEMM benchmark. `--verify=0` for sweeps.
- `fp4_quant_probe.cu`: device side NVFP4 quantizer (CUTLASS ships only a host
  one), validated at relative Frobenius 0.145. Builds with `-DQUANT_PROBE`.

Exploration harnesses (gemm2 bf16/fp8/fp4 sticks, chain pipe/overlap/tower,
cuDNN and cuBLASLt probes, plain sparse probe) were pruned 2026-07-02; they
live in git history and their findings are in the recap below.

## Prerequisites

Three things, none vendored:

1. CUDA 13.1 or newer, and an sm120 (consumer Blackwell) GPU.
2. A CUTLASS 4.4.2 checkout at `4ca61d06`, with `cute_scaledbasis_eq.patch`
   from this repo applied to it (needed only for the 256-tile builds).
3. For any sparse build, a 13.3 `ptxas` in a venv — recipe below. `build_sparse.sh`
   exits with `13.3 ptxas missing` if it is not there.

Verified 2026-08-31 from a fresh clone on an account with none of the author's
files present: with the three above in place, `build_sparse.sh fp4_sparse_chain.cu
/tmp/out -DTBM=256 -DC_FP4` builds and the binary runs (842.9 TFLOPS at 4096,
unlocked clocks).

## Build and run

```
CUT=/path/to/cutlass          # 4.4.2 at 4ca61d06, not vendored
REPO=/path/to/this/repo       # the directory this README is in
export CUT REPO               # build_sparse.sh reads $CUT from the environment
cd $CUT
INC="-I include -I tools/util/include -I examples/common"
FLAGS="-O3 -arch=sm_120a -std=c++17 --expt-relaxed-constexpr -DNDEBUG"

# dense chain (raster 1 = AlongM, best dense config)
nvcc $FLAGS -DRESIDUAL $INC $REPO/fp4_chain_cutlass.cu -o /tmp/fp4chain
/tmp/fp4chain 4096 300 1 1

# fused FFN block
nvcc $FLAGS -DFFN_BLOCK $INC $REPO/fp4_ffn_block.cu -o /tmp/fp4_ffn
/tmp/fp4_ffn --t=4096 --dm=4096 --dff=14336 --verify=0 --iterations=300
#   --verify defaults to 1 and runs a host-side reference that takes many
#   minutes at this shape. Pass --verify=0 for timing runs.
#   --iterations defaults to 10, which is visibly noisy; 300 is stable.
```

True peak numbers need a clock lock (machine wide, revert after):

```
sudo nvidia-smi -pl 330 && sudo nvidia-smi -lgc 3090
# run benchmarks
sudo nvidia-smi -rgc && sudo nvidia-smi -pl 300
```

### Sparse build recipe (the 13.3 ptxas splice)

nvcc offers no ptxas override, so splice via `--dryrun`. Build 120a-only: the
compute_120 fallback pass rejects family-specific sparse instructions.

```
python3 -m venv ~/.local/share/double-gemm/ptxvenv
~/.local/share/double-gemm/ptxvenv/bin/pip install nvidia-cuda-nvcc==13.3.73
P33=$(find ~/.local/share/double-gemm/ptxvenv -path '*/site-packages/nvidia/cu13/bin/ptxas' -type f | head -n1)

cd $CUT
nvcc --dryrun -O3 -gencode arch=compute_120a,code=sm_120a -std=c++17 \
  --expt-relaxed-constexpr -DNDEBUG $INC \
  $REPO/fp4_sparse_chain.cu -o /tmp/sparse_chain 2>&1 \
  | sed 's/^#\$ //' > /tmp/dry.sh
{ echo 'set -e'; echo "cd $CUT"
  sed -e "1,13s|^\([A-Za-z_][A-Za-z0-9_]*\)=\(.*\)|export \1='\2'|" -e '14,18d' \
      -e "s|^ptxas -arch=sm_120a|$P33 -arch=sm_120a|" \
      -e 's|^rm /tmp/tmpxft|rm -f /tmp/tmpxft|' /tmp/dry.sh
} > /tmp/hyb.sh
bash /tmp/hyb.sh
/tmp/sparse_chain 4096 300 1 1
```

The fused chain builds the same way after generating its kernel header next to
the source (regenerate after bumping the CUTLASS checkout; anchored patches
fail loudly on drift):

```
python3 $REPO/make_fused_kernel.py   # writes sm120_fused_chain.hpp
# then the dryrun splice above with fp4_fused_chain.cu
/tmp/fused_chain 4096 300 1 1 1 2   # argv: n iters swz ras gate d [beta2off pf]

# 256-tile build (>=6144 shapes): apply the cute patch once to the CUTLASS
# checkout, then add -DTBM=256 (and -DC_FP4 for the fp4 residual stream):
#   cd $CUT && git apply $REPO/cute_scaledbasis_eq.patch
#   XFLAGS="-DTBM=256 -DC_FP4" <splice build>
/tmp/fused_chain_256 6144 200 1 1 1 2
```

## Reproducing the headline numbers

The dense, sparse and fused d=8 rows in the table at the top of this file were
re-measured from a clean build on 2026-08-31, along with the FFN block, and the
raw stdout is committed under `bench/repro-2026-08-31/`. The deep-C 873 row and
the fp4-residual 878 row were not reproduced in that session, and they are
marked as such in that table. Methodology is interleaved same-session A/B:
dense, sparse and fused alternate inside one run at one clock lock, so no
variant gets a thermal advantage.

| variant | claimed above | re-measured, 3 rounds |
| --- | --- | --- |
| dense chain | 0.404 ms · 680 | 0.3995 / 0.3997 / 0.3994 ms · **688.0 / 687.7 / 688.3** |
| 2:4 sparse chain | 0.336 ms · 819 | 0.3338 / 0.3342 / 0.3343 ms · **823.4 / 822.5 / 822.3** |
| fused, one kernel, d=8 | 0.320 ms · 862 | 0.3202 / 0.3206 / 0.3207 ms · **858.6 / 857.3 / 857.0** |
| fused, one kernel, d=2 | — | 0.3188 / 0.3191 / 0.3197 ms · **862.3 / 861.5 / 859.7** |
| fused, deep-C epilogue | 0.3147 ms · 873 | not reproduced; the d=8 and d=2 rounds above run that epilogue (`sC=3 sD=1 C=16b`) and read 857.0 to 862.3 |
| fused, fp4 residual stream | 0.3131 ms · 878 | not re-measured; no `-DC_FP4` round was run in this session |
| FFN block, 2700 lock | 551 | **561.0 / 561.4 / 560.9** (10 it), **558.8 / 558.5 / 558.0** (300 it) |

Conditions: RTX 5070 Ti, driver 595.71.05, `nvidia-smi -pl 330 -lgc 3090`
(2700 for the FFN row), sustained 3022 MHz, 46-64 C, no other compute process on
the device. nvcc 13.1.115 front end; CUTLASS 4.4.2 at `4ca61d06` with
`cute_scaledbasis_eq.patch` applied; the sparse and fused binaries use the 13.3
ptxas splice below.

**An idle GPU is part of the measurement.** `bench/repro-2026-08-31/contention.log`
is an A/B/A: one competing GPU process running the dense chain in a loop takes
the FFN block's first GEMM from 1.4452 ms to 3.1099 ms, and it returns to
1.4278 ms when the competitor stops. A number read off a busy device can be 40%
low with no other symptom.

## Findings recap

- Thermal methodology: cold runs read 2 to 3% higher than soaked runs
  (sustained 3037 vs 2970 MHz at a 3090 lock). Only interleaved same-session
  A/Bs are valid comparisons.
- Dense knob graveyard at 4096 square: MXFP4 -6%, FP8 residual -4.5%,
  cooperative schedule -8%, 2 stages -9%, 3 stages -2%, stream-K unsupported on
  pingpong, CUDA graphs flat, swizzle flat to negative, 64 and 256 wide tiles
  broken or pathological, EPI subtiles and 5 stages fail to compile. Raster
  AlongM +1.6% was the only dense win.
- Naive GEMM chaining does not amortize and 2-stream cross-iteration overlap
  runs slower than serial (654 vs 675). Depth wins need in-kernel fusion.
- Real FFN shapes (deep K, wide N) give ~7% more GEMM throughput than the
  square stick; the square number is the conservative one.
- The chain pair is not bandwidth bound (~200 GB/s of a ~896 GB/s wall); the
  FFN block is, through its bf16 intermediate.
- On sm120 with CUDA 13.1 only dense block-scaled NVFP4 is emittable; plain
  f8f6f4 and all sparse FP4 need the 13.3 ptxas splice.
- Seam findings (2026-07-03): the kernel boundary at 4096 square costs ~10 us
  (two tail waves + launch gap); tile-granular fusion recovers it and stripe
  interleave adds ~3 us more. Dependency gating over CLC launch order is free
  (gate on equals gate off). d=1 interleave is too tight (-13%), d 2..32 is
  flat within 1%, d=8 best by a hair.
- The GEMM2 bf16 residual read is a flat 20 us at every schedule. Warming C2
  into L2 via prefetch.global.L2 from the idle epi-load warp made it 15 TFLOPS
  worse: the default L2 policy was already optimal, the eviction pressure was
  not worth it.

## Open levers

1. Sparsify `fp4_ffn_block.cu` the way the chain was sparsified (weights as the
   sparse A operand) + carry the 256-tile and deep-C epilogue there. The block
   sits at 551 dense; the chain techniques are worth ~2x combined.
2. Fuse the SiLU times up epilogue into the gate+up GEMM to delete the 234 MB
   intermediate and lift the block toward the GEMM rate.
3. Above ~1070: tensor pipes are at 66% with L2 at 76% (256-tile wall). Feed
   rate is the game: a smem-atom rework could unlock 256x256x128 / 384-M tiles
   (the current failures are layout-atom divisibility, not hardware), or attack
   L1/smem-side traffic (l1tex 45%). ~1600 is the perfectly-fed bound at 3090.
4. 4096 square is quantization-bound with fat tiles: a tail-aware schedule
   (mixed tile sizes or stream-K on the last wave) could close 872 -> ~900.
5. A full CUDA 13.3+ toolkit install would retire the dryrun splice.
