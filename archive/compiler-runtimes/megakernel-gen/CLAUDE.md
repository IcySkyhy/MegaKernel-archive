# megakernel-gen -- working notes for Claude

## What this is
`mkc` is a **megakernel compiler**: HuggingFace checkpoint -> a single persistent
cooperative CUDA kernel that decodes one token per launch, plus its loader and
C ABI.  One calibration per GPU type; after that compilation is a pure function
of (model IR, machine model).  No hill-climbing, no token burning.

## Goal (from the user, verbatim intent)
Robust, maintainable, concise, clean, extensible, simple **and** frontier-fast:
faster than vLLM/SGLang and competitive with cloud providers on open models.
Cleanliness is not tradeable against features -- it was called out twice.

## Layout
```
mkc/src/
  main.rs     CLI: analyze | plan | compile | build (+ --verify)
  hf.rs       safetensors header + config.json accessors
  arch.rs     ONE generic decoder-only frontend -> ir::Model
  ir.rs       the model IR (Quant, Weight, Xform, Attention, Ffn, MoeCfg...)
  quant.rs    THE single place a weight format is described (4-edit extension pt)
  hw.rs       machine model: measured gemv/stage curves, barrier cost
  plan.rs     cost model + scheduler -> Plan (struct Build = one method/stage)
  search.rs   `mkc build`: enumerate schedules, compile them all in parallel,
              keep what the assembler places, optionally time them
  codegen.rs  literal CUDA templates with @NAME@ holes -> stages.cuh/megakernel.cu
  layout.rs   compile-time weight table (which file bytes -> which device rows)
  emit.rs     runtime.cu, Makefile, mkbench.cpp, the output tree
  validate.rs refuses what it cannot compile, with a reason
runtime/include/mk/*.cuh   hand-written, unit-tested primitives (gemv, norm,
                           rope, attn, moe, common)
calib/           the only places measurement enters:
  calib.cu         the machine model -- once per GPU type
  bar_probe.cu     is the grid barrier the cheapest CORRECT one?  (it is)
  tp_probe.cu      what a cross-GPU rendezvous would cost
harness/         mkrun.py (engine+gate), mkguard.py (adversarial), run.py, ...
tests/           run_tests.sh: exhaustive primitive tests + compile-all-models
bench/           vLLM and SGLang on the identical workload
scripts/         zoo.sh, table.py (regenerates the README table from the TSV),
                 delta.py (compares two runs row by row),
                 slurm_{calib,final,occ,bar,tp}.sh
```

## Non-negotiables
- **Never let the framework cheat.** `harness/mkguard.py` zeroes weights /
  rope tables and checks the output moves; `mkrun.py` gates teacher-forced
  logits against transformers.  Any perf idea must keep both PASS.
- **Nothing derived is ever written into the repository or under `$HOME`.**
  On a cluster `$HOME` is usually quota-limited and read-only from the compute
  nodes, so a cache under it kills a benchmark halfway through.  Everything goes
  to `$MKGEN_WORK` -- see `env.sh` and `env.local.sh.example`.  The repository
  itself is under 700 KB and stays that way.
- **The interactive GPUs are shared with other people's work.**  Never kill a
  process you did not start; keep local use to compiles and short profiles; put
  every number that will be reported on a batch job with an exclusive GPU.
- Timing numbers must all come from ONE frozen `mkc` build.  Rebuilding while a
  slurm zoo is running mixes binaries: re-run the zoo afterwards.

## Environment
`source env.sh` -- modules, GPU auto-select, and every path derived from
`$MKGEN_WORK`, which `env.local.sh` (untracked) sets for this machine.
Binary: `$CARGO_TARGET_DIR/release/mkc`.  Machine model: `hw/nvidia_h100_80gb_hbm3.json`.
Python: `$MKGEN_PY`.  Checkpoints: `$MKGEN_MODELS`.  Slurm reads `SBATCH_ACCOUNT`
and `SBATCH_OUTPUT` from the environment, so no batch script hardcodes either.

## Hard-won facts (do not rediscover)
- Calibration must be COLD and offset-cycled or it measures L2 (3268 GB/s lie).
- A stage's cost is item-starvation + bandwidth, not bytes/BW: measure it as a
  persistent cooperative stage and subtract the barrier.
- Extrapolating the stage curve must clamp slope >= 1 and floor at gemv_best,
  or the model predicts 10 TB/s.
- `attn_reduce` must be branch-free; the `#pragma unroll 1` data-dependent
  version was 49% of the token.  It is now a BLOCK per (head, 32-lane chunk)
  with its warps over the split axis, and one warp computes the softmax
  normaliser for the whole block -- warp-per-chunk left 3% of the GPU busy at
  128 key splits, and every warp recomputing the normaliser cost another 64% of
  the stage.
- The attention loop stages AU=2 keys' K AND V before the softmax consumes them;
  one key per iteration is 138 GB/s and 17% of a MoE token.
- L2 prefetch measured a 4.4% LOSS -> removed; see DESIGN.md for why.
- Forcing 32 registers spills and is 23% slower.  `__launch_bounds__(NT,BPS)`
  plus the back-off search in `mkc build` is the right lever.
- The Makefile must depend on `$(MKINC)/mk/*.cuh` or you validate a stale .so.
- Never name a harness file after a stdlib module (`profile.py` broke transformers).
- libmk.so must not be loaded before torch: `_preload_reference_stack()`.
- Reference must match the checkpoint's activation dtype (fp16 vs bf16) or a
  correct kernel looks like a 52-ULP bug.
- Weight-only-quantised checkpoints gate against DEQUANTISED reference
  (`--ref-dequant`), else the reference's own activation quantisation dominates.
- SGLang re-imports `__main__` (spawn): benchmark scripts need a real file with
  an `if __name__ == "__main__":` guard, not a heredoc.

## Roadmap
Two features would each change what this thing IS, and both have a clear shape.

**Tensor parallelism.**  Shard heads and the FFN intermediate across N GPUs; the
existing planner then works unchanged on the smaller shapes, and what is added is
two all-reduces of H floats per layer plus a cross-GPU rendezvous.  One process,
N devices, `cudaDeviceEnablePeerAccess`, direct peer pointers -- no IPC needed
inside one process.  The parts to write: a `tp` degree in the IR that divides
`n_heads`/`n_kv_heads`/`intermediate`/`vocab`; per-rank row ranges in
`layout.rs` (and a source-COLUMN slice, which the current `EPart` cannot express,
for the row-parallel matrices `o_proj` and `down_proj`); an `AllReduce` stage;
a world of contexts in `runtime.cu`.  See the probe numbers above -- the
all-reduce is the part that has to be good.

**Chunked prefill.**  Today prefill runs the decode kernel once per prompt token,
so a 1024-token prompt reads every weight 1024 times and vLLM wins TTFT by 35x.
The natural fix keeps the same kernel: make the gemv compute P outputs per row
instead of one (`acc[r][p] += W[r][k] * x[p][k]`, P activation vectors staged in
shared memory), which reads the weights once per P tokens.  R=1, P=8 is eight
accumulators -- affordable.  Attention is the hard half: the P tokens attend to
different prefixes and write P KV entries.

## Discipline learned the hard way
- **Never `cargo build` while a slurm job is running.**  The target directory is
  shared; the job picks up the new binary mid-run and the results table mixes
  two compilers.  The slurm scripts now copy `mkc` to
  `build/mkc-$SLURM_JOB_ID` and export `MKC_BIN`, but a rebuild still changes
  what the NEXT job copies.
- **The repository is read-only on compute nodes**, so `mkc calibrate` cannot
  write `hw/*.json` from a batch job.  Calibrate to `$MKGEN_WORK/hw.json`, point
  compiles at it with `MKC_HW`, and copy it into `hw/` from a login node.
- **Never calibrate or time on a shared GPU.**  A neighbour at 50% occupancy
  moves every number.  `scripts/slurm_calib.sh` and `scripts/slurm_occ.sh` exist
  for exactly this.

## Measured facts that shaped the design (2026-08-31, H100 SXM)
- Across (block size, blocks/SM) the PREDICTED cost spans ~10% and the MEASURED
  cost spans ~60%.  The cost model cannot rank occupancies; `--verify` times
  every candidate the assembler accepted, and that is why it exists.
- Omitting `minBlocksPerMultiprocessor` from `__launch_bounds__` is its own,
  often winning, schedule: NT=1024 with no minimum beat every explicit
  occupancy on Qwen3-1.7B (2.215 vs 2.538 ms), gpt-oss-20b (2.464 vs 2.612) and
  Qwen3-8B.  Pinning it to the value the assembler would have picked anyway
  still changed the allocation -- 16 B of spill became 368 B on gpt-oss-20b.
- Backing off occupancy on a tiny spill is usually wrong: 48 B of spill at two
  blocks/SM beat a clean kernel at one.  Let the measurement decide.
- **No float `atomicAdd` in the kernel.**  Addition is not associative, so an
  atomic combine makes the output depend on warp arrival order and the compiler
  stops being deterministic.  The guard catches it as
  `weight_sensitivity: RESTORE-NOT-BIT-EXACT` -- a weight that does not restore
  bit-exactly after being zeroed.  Combine through shared memory in a fixed
  order instead.

## Where things stand (2026-08-31 ~22:45)
- Compiler is clean and green: `tests/run_tests.sh` passes, and Qwen3-0.6B
  passes the gate with all 7 guard checks.
- `mkc build` compiles every candidate schedule in parallel (`.cand-*` trees
  under the output dir) and `--verify` builds them all in parallel then times
  them serially.  Qwen3-0.6B: 25 s for the search.
- Machine model is schema 2, all five gemv cores, calibrated per node.
- **v2 vs v1: every one of the 13 zoo models is faster, 1.6% to 23.6%, all
  PASS/PASS.**  Regenerate with
  `python scripts/delta.py $MKGEN_WORK/results_v1.tsv $MKGEN_WORK/results.tsv`.  Headline: Qwen3-8B 5.8859 ms at
  **81.4%** of the node's measured streaming ceiling (1.23x roofline);
  Qwen3-0.6B-FP8 0.9455 ms = 1058 tok/s.  `README.md`'s table and
  `docs/RESULTS.md`'s delta table are both GENERATED -- rerun `scripts/table.py`
  and `scripts/delta.py` rather than editing them.
- `tests/run_tests.sh` passes end to end, including two new checks: the same
  inputs produce byte-identical output, and no `@NAME@` template hole survives
  into the generated source.
- **v1 = job 868182 (done, 45 min).**  **v2 = job 868243** adds, over v1:
  the attention key-loop unroll (AU=2), the cross-split reduction's
  once-per-block softmax normaliser (-11.6% on a whole Qwen3-0.6B token), the
  audit fixes below, the zoo staleness guard, a `floor_ms` column, an offline
  MXFP4 reference, and working serving-engine baselines.
- **The hand-rolled grid barrier was measured and REJECTED.**  `cg::grid.sync()`
  is the cheapest CORRECT barrier at every geometry (132 blocks 1.11 vs 1.29 us,
  396 blocks 1.51 vs 2.05): a counting barrier serialises every block's arrival
  on one contended cache line where cooperative groups uses a tree.  The
  comparison lives in `calib/bar_probe.cu`, which now checks a barrier before
  it times one.
- **Audit fixes (from a review agent), all in v2:** the FFN activation buffer
  was sized `top_k*inter` and ignored a wider shared expert (Qwen1.5-MoE passed
  by exact coincidence) -- it now comes from `Plan::scratch`; `validate.rs`
  refuses MoE in a format with no `gemv_multi` (fp8/int4) and MoE with a
  post-FFN norm, both of which used to miscompile; one shared-memory formula
  instead of two that disagreed; `real_bps` now bounds occupancy by shared
  memory too; `with_splits` rebuilds the schedule instead of patching two of
  three fields; a model whose arena exceeds an SM is refused.
- **v1 zoo is complete**, all PASS/PASS except gpt-oss-20b (stale row, fixed for
  v2).  Headline: Qwen3-8B 6.1624 ms at **80.1% of the node's measured streaming
  ceiling** (1.25x roofline).  Full list, vs the previous run:
  Qwen3-0.6B-FP8 1.2381, Qwen3-0.6B 1.3577 (-2.4%), TinyLlama 1.4301 (-2.7%),
  Qwen2.5-1.5B-AWQ 1.7371 (was a FAIL), Qwen2.5-1.5B 2.0696 (-5.7%),
  Qwen3-1.7B 2.1378 (-9.2%), Qwen3-1.7B-FP8 1.7750, SmolLM2 1.9883,
  gemma-2-2b 2.9042, Phi-3-mini 3.5981 (-4.5%),
  Qwen1.5-MoE 2.5096 (-6.8%), Qwen3-8B 6.1624 (-4.8%, 80.1% bw util).
  gpt-oss-20b's row is STALE in v1 (see the staleness note above) -- fixed for v2.
- Best numbers to beat overall: gpt-oss-120b 3.4582 ms/tok vs vLLM 4.4585.

## Where the time actually goes (exclusive H100, v2 job 868243, ctx 1024)
Measured per-stage, in-kernel, with the compiler's own `%globaltimer` marks.

- **Qwen3-8B, 5844 us:** mlp1 42% @ 2948 GB/s, mlp2 23% @ 2679, qkv 12% @ 2527,
  oproj 9% @ 2278, lm_head 7% @ 3161, attn 4.8% @ 542, attn_red 1.9%.  The big
  gemvs track the machine's own cold-ramp curve almost exactly (128 MB -> 2789
  GB/s, 256 MB -> 2938), so **for a dense model this decomposition is at its
  limit**; the remaining ~19% is the DRAM ramp on 30-200 MB stages.
- **Qwen3-30B-A3B, 3891 us:** moe1 31% @ 1986, moe2 19% @ 1669, qkv 13% @ 2020,
  attn 12% @ 212, oproj 11% @ 1951, router 4.7%, attn_red 4.6%, lm_head 5%.
  Every gemv runs at 1600-2000 GB/s, not 2900, because a 48-layer MoE with a
  small hidden dim reads only 20-50 MB per stage.
- **gpt-oss-120b, 3103 us:** moe1 23% @ 1792, moe2 20% @ 1020, qkv 16% @ 2184,
  oproj 14% @ 1949, lm_head 12% @ 3103, attn 7.4% @ 185, router 4.6%, attn_red 3.3%.
- The attention key unroll moved attn by -31% (Qwen3-8B), -35% (30B), -31% (120b).
  **What is left is the per-stage DRAM ramp on MoE models, and TP is the lever.**

## Multi-GPU: what the probe says (job 868197, 4xH100 NVLink NV6)
- A cross-GPU rendezvous between resident persistent kernels costs **~5.7 us**
  at world=2 (that number includes an intra-GPU grid.sync at grid 1056, which is
  ~3 us on its own, so the cross-device part is ~2.7 us).
- `calib/tp_probe.cu`'s all-reduce costs ~52 us regardless of size (8-64 KB), so
  it is latency, not bandwidth -- and it is a BAD implementation: three
  intra-GPU rendezvous plus a `__threadfence_system()` per all-reduce.  A
  one-shot all-reduce that fuses the exchange into the rendezvous should land
  near 6-10 us.  Do not conclude TP is expensive from the 52 us number.
- Arithmetic for gpt-oss-120b at TP=4: bytes/GPU 1.26 GB -> ~0.87 ms at today's
  efficiency, plus 2 all-reduces x 36 layers.  At 10 us each that is 1.59 ms
  (2.2x faster than one GPU); at 6 us, 1.30 ms (2.7x).  Worth building; the
  all-reduce is the part that has to be good.
- `calib/tp_probe.cu` **hangs at world=4** (world=2 is fine).  Undiagnosed;
  suspect the peer-mapped flag array.  Fix before trusting a 4-way number.
- **A benchmark row must never be able to be stale.**  `zoo.sh` deletes
  `result.json` before the run and says RUN FAILED if it is not recreated.  It
  had been silently reusing an old one whenever `harness/run.py` died -- which
  it did for gpt-oss-20b on every compute-node run, because transformers'
  MXFP4 quantiser fetches a Triton kernel package from the hub and the node has
  no network.  `reference_logits` now asks for `Mxfp4Config(dequantize=True)`,
  which is both offline-safe and the more honest reference.
- **Slurm jobs re-read `scripts/*.sh` from /home on every invocation**, so an
  edit to `zoo.sh` lands mid-run.  Behaviour fixes are fine; do not change the
  results TSV schema while a job is running.
- **13 of 15 models are GATED; Qwen3-30B-A3B and gpt-oss-120b were guard-only.**
  The reference and the engine both want the whole model resident, and above
  ~8 B they do not fit together.  `harness/run.py --ref-first` now computes the
  reference and frees it before the engine loads, which makes Qwen3-30B-A3B
  gateable (60 GB, then 60 GB).  gpt-oss-120b still cannot be: dequantised to
  bf16 it is 240 GB, more than the card and more than the node's RAM.  Its
  architecture is gated on gpt-oss-20b, which shares every code path.  The
  results table must say "--" for it, never "PASS".
- **Other sessions may submit jobs under the same account.**  Never cancel a job
  you did not submit, and expect to queue behind someone else's array jobs.
- v2 big two: Qwen3-30B-A3B **3.9489 ms** (v1 4.3045, -8.3%, 49.1% bw util) and
  **gpt-oss-120b 3.1378 ms = 318.7 tok/s** (v1 3.2730, this morning 3.4582;
  50.4% bw util, predicted 3.1066 -- a 1% cost-model error).  vLLM on the same
  workload was 4.4585, so **1.42x**.  All guard checks PASS on both.  Neither big model is gated in v2: the `--ref-first` and
  `row.py` edits to `slurm_final.sh` postdate its submission (slurm copies the
  batch script at submit time), so the NEXT run gates Qwen3-30B-A3B and appends
  both big models to the results TSV automatically.  For this run, append them
  by hand: `python scripts/row.py $OUT/<name> <name> 0 $MKGEN_RESULTS`.
- **flashinfer JIT-compiles kernels and locks a file in its OWN cache dir**,
  which ignores `XDG_CACHE_HOME` and defaults under `$HOME`.  On a compute node
  that is read-only, the lock's `ftruncate` fails, and vLLM's engine core dies
  before it loads a model -- twice, in two runs, as a bare `PermissionError`.
  `env.sh` now sets `FLASHINFER_{WORKSPACE_BASE,CACHE_DIR,JIT_DIR}` and
  `VLLM_USE_FLASHINFER_SAMPLER=0`.  Baselines live in their own job now
  (`scripts/slurm_base.sh`) so a failure there does not cost a zoo run.
- Baselines, same workload (batch 1, ctx 1024, 128 steps, prefill subtracted),
  from `scripts/slurm_base.sh`, regenerate the README table with
  `python scripts/h2h.py <base-job.out> [more.out ...] --md README.md`:
  Qwen3-8B: vLLM 6.5016 / 6.5031 (two runs), SGLang 6.4358, mkc 5.8859 ->
  **1.09-1.10x**.  gpt-oss-20b: vLLM 3.3042 / 3.3019, SGLang 3.3053,
  mkc 2.1834 -> **1.51x**.  gpt-oss-120b: vLLM 4.4585 from an earlier exclusive
  run and 4.7433 / 4.7482 on these two, SGLang 4.7671, mkc 3.1378 ->
  **1.42-1.52x**.  The engines reproduce to within 0.1% across runs, which is the
  check that the N-vs-1 differencing is sound; the only number that moves is
  vLLM on the 120 B, and it moves between NODES, not between configurations.
- SGLang JIT-compiles into `~/.cache/sglang/jit` and also ignores
  `XDG_CACHE_HOME`.  Rather than chase one env var per library,
  `scripts/slurm_base.sh` now points **HOME itself** at
  `$MKGEN_WORK/fakehome` for the engine processes; the virtualenvs
  are absolute paths so nothing else notices.
- The two engines want **different** static memory fractions on the same model:
  SGLang's MXFP4 swizzle needs the unswizzled and swizzled copies live at once
  (so a smaller pool), while vLLM sizes its KV cache out of what is left and
  fails outright if that is too small.  One fraction for both is how the 120 B
  baseline failed twice, in opposite directions.  `slurm_base.sh` now carries
  `model:vllm-frac:sglang-frac`.

- Two baseline jobs in flight wrote the same `sgl_<model>.log`, so the loser
  could overwrite the winner.  `slurm_base.sh` now logs under
  `logs/base-$SLURM_JOB_ID/`.  The per-job stdout still carries the RESULT line,
  which is what `scripts/h2h.py` reads.
