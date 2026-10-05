# How mkc works

A megakernel is not a clever kernel; it is a *schedule*. The arithmetic of a
transformer decode step is boring — a dozen matrix-vector products, a softmax, a
couple of norms — and all of it is memory bound at batch 1. What makes a
megakernel fast is the ordering: which work items exist, which warp gets each
one, where the grid barriers fall, what stays in shared memory across what used
to be a kernel boundary, and how many loads each warp has in flight.

That is exactly the shape of thing a compiler should produce, and exactly the
shape of thing people currently produce by hand, one model at a time, by
sweeping. `mkc` replaces the sweep with a cost model over measured primitives.

---

## 1. The split: primitives are written, schedules are generated

`runtime/include/mk/*.cuh` is hand-written and fixed. It contains the gemv cores
(bf16/f16 dense, MXFP4, FP8, INT4, multi-expert), the normalisations, the rotary
embedding, the cross-split softmax reduction, the router top-k and the warp and
block reductions. Nothing in the compiler emits arithmetic.

`mkc` emits the composition: `stages.cuh` (one device function per stage),
`megakernel.cu` (the persistent loop over layers with barriers between stages),
`stages.cu` (each stage also as an ordinary kernel, so `ncu` can see it),
`weights.h` (the weight table) and `runtime.cu` (loader, decode loop, C ABI).

The consequence is worth stating plainly: **a generated kernel can be wrong in
its schedule, but not in its arithmetic.** The arithmetic is tested once, for
all models, and the schedule is what the planner already decided.

---

## 2. The IR

One architecture-neutral `Model`: shapes, an `Attention` (heads, kv heads, head
dim, per-layer sliding windows, optional q/k norm, optional sinks, optional logit
soft-cap), an `Ffn` that is either `Dense` or `Moe` (experts, top-k, scoring
function, whether the softmax is over all experts or only the selected ones,
shared experts, biases), three `Norm`s with a *kind* (RMS, Gemma's `1 + w` RMS,
LayerNorm), a `Rope` with its scaling variant, and a `Quant` per matrix class.

Everything downstream sees only this. Adding a model family is a frontend change;
adding a weight format is one `Quant` variant plus one gemv core.

### The frontend probes, it does not switch

There is one generic decoder-only reader, not a family of per-model ones. Config
keys are looked up with fallbacks (`num_key_value_heads` or `num_attention_heads`;
`head_dim` or `hidden/heads`; four different conventions for per-layer sliding
windows). Tensor roles are resolved against candidate name lists, so a fused
`qkv_proj` and separate `q_proj/k_proj/v_proj` are the same thing to the planner,
and so are a fused `gate_up_proj`, a pair of `gate_proj/up_proj`, and 128
per-expert `w1/w3` matrices. Unresolved roles are collected and reported
together, by name — a model the compiler cannot read says so, instead of
producing a kernel that computes something else.

### Weight formats live in one file

`quant.rs` is the whole surface of a weight format: which tensors a checkpoint
ships for it, what dtype each is stored in, which device pointers the kernel
needs, and what call a stage body emits. `layout.rs` builds the weight table and
`emit.rs` the loader and the layer struct from it; `codegen.rs` asks a `Mat` for its gemv. Nothing else
branches on the format.

Two ideas make that possible:

**Normalise at load time, not in the inner loop.** AWQ packs eight *output* rows
into one 32-bit word in the order {0,2,4,6,1,3,5,7}, with the reduction axis down
the rows; GPTQ packs the other axis; DeepSeek's fp8 stores one scale per 128x128
tile while compressed-tensors stores one per output row. The kernel sees none of
this: the loader repacks int4 into a canonical `[rows][K/2]` nibble layout, and
expands block scales to one per row. A format's weirdness belongs in the loader's
row mapping, and the row mapping is a compile-time table.

**Keep the transforms orthogonal.** A gate/up matrix in a block-scaled format is
*both* expanded (one scale row per weight row) *and* interleaved (so one warp
computes a channel pair). Those compose as two independent fields — a row map and
a repeat count — rather than as a combinatorial enum.

### The weight table is resolved at compile time

Every logical weight becomes a list of (file, byte offset, dtype, source rows) →
(destination row, stride) records, baked into `weights.h`. Concatenating q/k/v,
interleaving gate and up so that one warp computes both halves of a channel, and
dtype casts are all instances of the same row-mapping primitive. The generated
loader performs no name lookup and makes no layout decision at run time.

---

## 3. The machine model: the only measurement

`mkc calibrate` runs once per GPU type. Its output is the only thing in the
compiler that comes from a GPU, and after it exists, compiling a model is pure
computation.

Three of its measurements changed a design decision:

**Cold footprints.** This H100 has a 52 MB L2. A naive "bandwidth vs size" sweep
re-reads the same buffer and reports 3268 GB/s for a 32 MB footprint that will
actually run at 2500. Every repetition in the calibration reads memory it has not
just read.

**Stage shape, not kernel shape.** A megakernel stage is not a kernel launch: the
grid is already resident, the work is one or a few waves, and it ends at a grid
barrier. Measuring a gemv as a standalone launch conflates *item starvation* with
the DRAM ramp and mispredicts by 4x. The calibration runs each core inside a
persistent cooperative grid, N items over B cold bytes, barrier cost subtracted,
at two reduction lengths so the item axis and the bytes-per-item axis can be
separated.

**The barrier.** 1.10 µs at 132 blocks, 2.36 µs at 1056. Barriers × stages ×
layers is a fixed tax; on a 24-layer MoE model it is 8% of the token. It is what
decides how coarse the stages have to be, and it is why `mkc` works hard to
remove them (see §5).

That number is large enough to be worth not taking on faith, so
`calib/bar_probe.cu` asks whether the library's barrier is the cheapest correct
one. **It is.** A hand-rolled counting barrier — fence, atomic, spin, fence —
costs 0.2 µs more per barrier at 132 blocks and 0.5 µs more at 396, because
every block's arrival serialises on one contended cache line where cooperative
groups uses a tree. That is 30–170 µs of extra latency per token, so the
library stays.

The measurement came with a lesson attached. The first version of that counting
barrier spun on `ld.global.cg`, measured *half* the cost of `grid.sync()`, and
was **wrong**: it counted arrivals correctly and ordered nothing. In a direct
test — every block writes a slot, rendezvous, every block reads every slot —
90% of the reads came back stale. The generated kernel showed it as NaN logits
and the gate caught it in a minute. The probe now checks a barrier before it
times one, and prints the stale count next to the microseconds. Cheapest and
correct are different questions and a benchmark that only asks one of them will
answer it confidently.

The cost model is a two-dimensional interpolation of those measured points in
log-log space, with two guards: extrapolation above the measured item range uses
a slope of at least 1 (past saturation, twice the work takes at least twice the
time), and the result is clamped from below by the core's own measured ceiling.
Without those, extrapolation cheerfully predicts 10 TB/s.

---

## 4. The planner

Every decision that a human would reach by sweeping compiled binaries is a
closed-form evaluation here.

**Row blocking `R`.** Larger R amortises the shared-memory reads of the
activation vector and raises achieved bandwidth, but divides the item count — and
once items fall below the resident warp count, most of the GPU idles. Both halves
are in the measured table.

**Key splits.** More splits means more parallel items in the attention stage, but
the reduction that recombines them reads `n_heads × splits × head_dim` floats and
needs its own barrier. The planner prices both sides and picks the minimum; on
gpt-oss at 512 context it chooses 32, and choosing 16 instead measured 75% slower.

**Expert interleaving.** The top-k expert down-projections are independent gemvs
over the same output rows. Interleaving F of them into one group loop multiplies
the loads in flight per warp by F at no cost in item count. It is priced exactly
like row blocking — a fused group of F is a stage with F times the bytes per item.

**Occupancy is a decision, not an outcome.** Blocks per SM sets the register
budget every stage must fit in, and the register budget decides how much row
blocking each stage may use — while occupancy simultaneously sets how many warps
are resident, which is what a single-wave stage's bandwidth depends on. The two
pull in opposite directions, so `mkc build` enumerates the small space of (block
size, blocks per SM, fusion degree), compiles each candidate once to find out
what the assembler will actually give it, and keeps the cheapest schedule that
compiles without spilling.

That last step is a *feasibility* search over register allocation, not a
performance search: no candidate is ever run, let alone timed. It keeps every
occupancy the assembler accepted, not only the highest — more blocks per SM
means more resident warps but fewer registers each, and which side wins is not
something the calibration can see.

**`--verify` closes that loop with a measurement, and it is the one place the
compiler admits its cost model is not enough.** The predicted cost across
(block size, blocks per SM) spans about 10% while the measured cost spans 60%:
on Qwen3-1.7B one block per SM at 128 registers beats two at 64 by 10%, and
nothing in a bytes-and-items model sees why. So `--verify` builds each candidate
the assembler accepted, times it on *synthetic* weights (timing depends on the
layout, not the values, so this needs no checkpoint and costs seconds) and keeps
the measured winner, then sweeps the key splits around it. It is bounded — the
candidate set is whatever the register search already compiled — and
deterministic. Without it the build is still one shot and touches no GPU; with
it the schedule is chosen by the machine.

---

## 5. What the schedule looks like, and what was removed from it

Per layer:

```
qkv | attn | attn_reduce | oproj | router · topk | moe1 | moe2 |
                                        ^ no barrier
```

Three barriers that a naive lowering would have are not there:

**RoPE and the q/k norm do not get their own stage.** A Qwen3-style per-head norm
needs a reduction over the whole head, and the qkv projection spreads a head
across many warps — so the obvious lowering inserts a barrier. Instead the norm
and the rotation happen at the *start of the attention stage*, in the registers of
the warp that is about to consume the head. The current position's key is
computed there too, from the raw projection, so the KV cache is only ever read for
positions written by earlier tokens and the write of `K[pos]` races with nothing.

**The router's norm is not recomputed by the MoE stage.** A grid barrier does not
clear shared memory, and the top-k reads only the gate logits, so `ffn_norm(x)` is
still sitting in the arena when the expert gemv starts.

**The attention loop reads two keys at a time.** One key per iteration gives a
warp `2·head_dim/32` loads in flight and then stalls on a warp reduction and an
exponential before it may issue the next: on a 48-layer MoE that stage measured
138 GB/s and 17% of the token for 2 MB a layer. Staging two keys' K *and* V into
registers first doubles the independent loads; the online softmax that consumes
them is unchanged and still runs in a fixed order, so the kernel stays
bit-deterministic. The unroll factor is small because it costs `2·U·head_dim/32`
live registers, and registers are what set occupancy for every other stage.

**The top-k has no barrier at all.** A few hundred bytes of gate logits are in L2;
every block recomputes the selection on one warp with shuffle reductions.

And one that *is* there, but had to be made worth its cost. The cross-split
softmax reduction reads `n_heads x splits x head_dim` floats and writes only
`n_heads x head_dim`, so a warp-per-output-chunk decomposition has a few dozen
work items for the whole GPU: at 128 key splits it left 3% of the resident warps
with anything to do and cost more than the attention it was reducing. The
parallelism has to come from the input, so a whole *block* takes one output
chunk and its warps divide the split axis. What makes the combine cheap is
computing the softmax normaliser first — `(M, L)` depend only on the per-split
`(max, sum)` pairs, a broadcast read — after which each warp's contribution is a
plain weighted sum and the warps combine with an add in shared memory. A
`__syncthreads`, not a grid barrier.

---

## 6. What the guard is for

The gate answers "does it match the reference". The guard answers the question
the gate cannot: "could it have matched without doing the work?" Every one of
these was a real bug in a kernel that compiled, ran, and produced plausible
logits:

* the loader's dtype dispatch had no case for fp8, so any fp8 matrix that needed
  row interleaving silently loaded as **zeros** — the MLP produced nothing, and
  the model still looked input-sensitive because attention worked;
* a tied-embedding checkpoint that also ships `lm_head.weight` loaded 300 MB that
  the kernel never reads;
* the router's top-k kept its sentinel index when a corrupted weight produced
  NaN, and indexed the expert table out of bounds.

The last one only appears because the guard corrupts weights on purpose. The
first only appears because it zeroes *whole* tensors and requires the output to
move.

The guard has also been wrong, twice, and both times the fix was to make the
check sharper rather than to loosen it. Its original "position sensitivity" test
fed the same token at two positions and demanded different logits — but attention
over identical value vectors returns that value whatever the scores are, so a
repeated token is genuinely position-invariant and the reference agrees bit for
bit. It was replaced by zeroing the rotary tables, which asks the question that
was actually meant.

## 7. Findings the profiler produced

The compiler emits its own profiler: one `%globaltimer` read per barrier from
block 0, which costs nothing and gives the real per-stage schedule to compare
against the predicted one. Every number below is a fused, in-kernel measurement.

| finding | effect |
|---|---|
| `attn_reduce` looped over splits with a data-dependent `continue`, serialising 64 loads | 49% of the token → 4% |
| `norm_to_smem` read the hidden state twice, once for the reduction and once for the write | ~5 µs per stage, on every stage that has a norm |
| the grid-stride item mapping gave block *b* items `b·NW … b·NW+NW`, leaving 40% of SMs with nothing | qkv ran on 160 of 264 blocks |
| the gemv loop `for (k0 = lane*8; k0 < K; k0 += 256)` has a lane-dependent trip count for K=2880 and cannot unroll | one load in flight per warp |
| `LayerW` copied into registers pinned ~26 pointers live across the whole layer | fused kernel took 128 registers when no stage needs 64 |
| the top-k selection kept a sentinel index when the gate contained NaN | out-of-bounds expert read — found by the guard corrupting a weight |
| the cross-split reduction combined its per-warp partials with `atomicAdd` | float addition is not associative, so the output depended on warp arrival order — found by the guard restoring a zeroed weight and not getting the same logits back |

One measured result is worth recording as a negative: prefetching the next
stage's weights into L2 during the stages that leave the memory system 95% idle
looks free and is **not** — it cost 4.4% on gpt-oss-20b. The prefetch
instructions compete for issue slots, and 24 MB of speculative lines evict a
52 MB L2 faster than the next stage can consume them. The code
is gone; this paragraph is what remains, because it is the obvious idea and the
next person will have it.

The generated Makefile lists the runtime headers as prerequisites. Leaving them
out is the classic way to "validate" a kernel rewrite that was never compiled;
it cost an hour here before the dependency was added.
