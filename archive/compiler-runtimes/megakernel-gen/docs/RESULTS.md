# Results

Every number here is batch 1, greedy, a 1024-token prompt and 128 decode steps on
one H100 80GB HBM3, with prefill subtracted by the N-vs-1 difference so that
nothing in the measurement is prefill or prefix cache.

`bw util` is bytes-that-must-be-read per token divided by measured time, over the
machine's **measured** streaming ceiling — 3189 GB/s on the node these numbers
come from. The 3350 GB/s on the spec sheet is not reachable by any kernel, and
using it as the denominator flatters every result by 5%. Nor is the ceiling the
same on two nodes: three runs of this table, on identical H100 SXM cards,
measured 3190, 3095 and 3189 GB/s, so the calibration travels with the job and
every reported utilisation is against the card it actually ran on.

## The compiler's own claim

The interesting number is not any single latency; it is that **one command, with
no per-model work of any kind, produces a correct megakernel for every one of
these architectures.** The gate is a teacher-forced comparison against
HuggingFace `transformers` on the same checkpoint, and the guard is seven
adversarial checks that a fast-but-wrong kernel cannot pass.

## What is exercised

| | |
|---|---|
| attention | MHA (SmolLM2), GQA at ratios 2, 6, 8 (Qwen3, Qwen2.5, TinyLlama), sliding window (Phi-3, gpt-oss), attention sinks (gpt-oss) |
| norms | RMSNorm, Gemma's `1 + w` variant, LayerNorm, per-head QK-norm (Qwen3), Gemma-2's four-norms-per-layer placement |
| rotary | plain, YaRN (gpt-oss), linear and Llama-3 scaling, head_dim 64 / 96 / 128 |
| FFN | dense SwiGLU, MoE top-4 of 32/128 with softmax-after-topk (gpt-oss), MoE top-8 of 128 (Qwen3-MoE) |
| weights | bf16, f16, MXFP4 (4.25 bit), FP8 e4m3 (128x128 block scales and per-output-channel), INT4 AWQ (group 128) |
| layout | split and fused qkv, split and fused gate/up, per-expert and fused expert tensors, tied and untied embeddings |

None of these are special cases in the compiler; they are fields in the IR that
the planner and the codegen read.

## Method notes

**Two build modes.** `mkc build` is pure prediction: it compiles
candidate schedules to find out what the assembler will give them in registers,
but it never runs one. `mkc build --verify` additionally times *every* schedule
the assembler accepted, on *synthetic* weights — timing depends on the layout,
not the values, so this needs no checkpoint and costs seconds — and keeps the
measured winner.

It times all of them rather than the few the cost model liked because the cost
model cannot rank them: across (block size, blocks per SM) the *predicted* cost
spans about 10% and the *measured* cost spans 60%. On Qwen3-1.7B the ten
schedules the assembler accepted measured 2.15 to 2.51 ms and the predictions
put the slowest of them first. The search is still bounded and deterministic --
the candidate set is exactly what the register search already compiled, and it
compiles them in parallel -- but the choice among them is a measurement, and
saying otherwise would be a lie about what the cost model can do.

**The gate.** Teacher-forced logits against `transformers`, on a token sequence
drawn randomly and seeded from the clock. Thirteen of the fifteen models are
gated this way. The reference and the engine both want the whole model resident,
and above about 8 B parameters they no longer fit side by side; `--ref-first`
computes the reference and frees it *before* the engine loads, which is what
makes a 30 B model gateable at all. gpt-oss-120b cannot be gated on this machine
by any arrangement — dequantised to bf16 for the reference it is 240 GB, more
than the card and more than the node's RAM — so its row says so, and its
architecture is gated on gpt-oss-20b, which is the same code path end to end. Reported as top-1 agreement with
genuine ties excluded, and as maximum error in units of one bf16 ULP of the peak
logit. Typical end-to-end drift is 4–21 ULP depending on depth, which is about
half a ULP per layer — honest reduction-order noise, not a bug.

**Two references, for weight-only quantised checkpoints.** A w8a8 or w4a16
checkpoint means "quantise the weights *and* the activations". `transformers`
does both; mkc keeps activations in the model's own float type and quantises only
the weights. Comparing against their path therefore measures *their activation
quantisation*, which is 20–50 ULP and drowns out any real bug. So those models are
gated twice:

* against the stock `transformers` path — loose, but fully independent, and the
  only thing that can catch a wrong scale convention (a reversed one gives a
  correlation near zero, not near one);
* against the same stored weights dequantised by the format's own definition and
  run through the ordinary float model — tight, because the only difference left
  is reduction order. FP8 and AWQ both land at 4–5 ULP there, the same as an
  unquantised model.

Getting this wrong in the *harness* cost real time twice: once by running the
dequantised reference in bf16 when the checkpoint's activations are fp16 (which
looked like a 52-ULP kernel bug and was a harness bug), and once by "fixing" a
`transformers` crash in a way that routed a dense model into a MoE kernel.

**The guard.** See `harness/mkguard.py`. The load-bearing one is
`weight_sensitivity`: it zeroes each weight tensor in turn, requires the output
to move, then restores the bytes from the checkpoint and asserts the restored
model is bit-identical. The zoo runs it on a sample for speed;
`scripts/full_guard.sh` runs it over **every** tensor in a model, which is the
strongest statement available without a reference: every byte the compiler said
to load is a byte the kernel reads. That is the only cheap check that catches a kernel which
silently skips a layer or an expert branch — which is exactly the shortcut an
optimiser rewarded on latency alone would find.

## The loop, run once, on this table

Everything in this document exists so that "make it faster" is a measurement
rather than an opinion. Here is one turn of that loop, start to finish, on the
table above.

**The compiler's own profiler said where the time was.** On an exclusive GPU,
per stage, in-kernel: on a 48-layer mixture-of-experts model *attention* was 17%
of the token at 138 GB/s, and the cross-split softmax reduction another 4%. On a
dense model the big matrix-vector stages tracked the machine's cold-ramp curve to
within a few percent — meaning they were done, and the attention stages were not.

**Two of the fixes were decompositions, not tricks.** The reduction had been
giving one output chunk to one warp, which is a few dozen work items for the
whole GPU; a block per chunk with its warps over the split axis fixed the
parallelism, and computing the softmax normaliser once per block instead of once
per warp removed thirty-two-fold redundant work. The attention loop had been
issuing one key's loads and then stalling on a warp reduction and an exponential;
staging two keys' K *and* V into registers first doubled the loads in flight.

Measured in-kernel on an exclusive GPU, the second of those is what moved the
table: attention fell from 407 to 279 µs on Qwen3-8B (−31%) and from 728 to 474
on Qwen3-30B-A3B (−35%). The reduction's rework is worth stating separately and
honestly — it was worth 64% of that stage on a *contended* card and close to
nothing on an idle one, because once the parallelism is there the stage is bound
by reading the partial sums, not by recomputing two floats. The redundancy was
real and removing it was right; the speedup was mostly not where it looked.

**The third measurement said to change nothing.** A hand-rolled counting barrier
looked like a third off the biggest fixed tax in the kernel. Checked, it was
0.2–0.5 µs per barrier *slower* than the library's, and the version that had
looked faster was silently wrong. That is a saved regression, and it cost one
probe.

**Then the same command, on the same models, on an exclusive node.** No model was
touched, no schedule was hand-picked, and the gate and the guard were re-run on
every one. (gpt-oss-20b's "before" is not comparable: its reference had been
failing to load offline, and the harness was silently reusing an older result
file. Both are fixed — the reference now asks `transformers` for the dequantised
MXFP4 path, and a run that dies leaves no row at all.)

<!-- BEGIN DELTA TABLE -->
| model | before | after | change | bw util before → after |
|---|---|---|---|---|
| Qwen3-0.6B-FP8 | 1.2381 ms | **0.9455 ms** | -23.6% | 22.7% → 28.8% |
| Qwen3-0.6B | 1.3577 ms | **1.0440 ms** | -23.1% | 31.2% → 39.3% |
| TinyLlama-1.1B-Chat-v1.0 | 1.4301 ms | **1.2049 ms** | -15.7% | 47.3% → 54.4% |
| Qwen2.5-1.5B-Instruct-AWQ | 1.7371 ms | **1.4555 ms** | -16.2% | 22.2% → 25.7% |
| Qwen2.5-1.5B-Instruct | 2.0696 ms | **1.7418 ms** | -15.8% | 48.7% → 56.1% |
| Qwen3-1.7B | 2.1378 ms | **1.7834 ms** | -16.6% | 53.8% → 62.6% |
| Qwen3-1.7B-FP8-dynamic | 1.7750 ms | **1.4604 ms** | -17.7% | 39.2% → 46.2% |
| SmolLM2-1.7B-Instruct | 1.9883 ms | **1.7406 ms** | -12.5% | 58.9% → 65.3% |
| gemma-2-2b-it | 2.9042 ms | **2.7612 ms** | -4.9% | 59.4% → 60.6% |
| gpt-oss-20b | 2.6799 ms | **2.1834 ms** | -18.5% | 43.6% → 53.6% |
| Phi-3-mini-4k-instruct | 3.5981 ms | **3.2988 ms** | -8.3% | 70.5% → 74.6% |
| Qwen1.5-MoE-A2.7B | 2.5096 ms | **2.4706 ms** | -1.6% | 63.8% → 62.9% |
| Qwen3-8B | 6.1624 ms | **5.8859 ms** | -4.5% | 80.1% → 81.4% |
| Qwen3-30B-A3B | 4.3045 ms | **3.9489 ms** | -8.3% | 46.4% → 49.1% |
| gpt-oss-120b | 3.2730 ms | **3.1378 ms** | -4.1% | 49.8% → 50.4% |
<!-- END DELTA TABLE -->

## Where the time goes now

Measured in-kernel on an exclusive GPU, per stage, at 1024 context.

| | Qwen3-8B (dense) | Qwen3-30B-A3B (MoE) | gpt-oss-120b (MoE) |
|---|---|---|---|
| token | 5844 µs | 3891 µs | 3103 µs |
| the two big FFN stages | 65% @ 2679–2948 GB/s | 50% @ 1669–1986 | 43% @ 1020–1792 |
| qkv + o_proj | 21% @ 2278–2527 | 23% @ 1951–2020 | 30% @ 1949–2184 |
| attention + its reduction | 6.7% | 17% | 11% |
| lm head | 6.7% @ 3161 | 5.2% @ 3089 | 12% @ 3103 |

Two different stories. On the dense model the big matrix-vector stages run within a
few percent of the machine's own cold-ramp curve — 128 MB reads at 2789 GB/s,
256 MB at 2938 — so **that decomposition is at its limit**, and the remaining
19% is the ramp, not the schedule. On the mixture-of-experts models the same
stages run at 1000–2000 GB/s for the same reason inverted: a 48-layer model with
a small hidden dimension reads only 20–50 MB per stage, and no arrangement of
one GPU's warps makes a 20 MB read go at 3 TB/s.

That is the honest boundary of this design. Beyond it the lever is not a better
schedule; it is fewer bytes per GPU, which means tensor parallelism.

## The barrier, and why it is still the library's

A megakernel pays for one grid-wide rendezvous per stage per layer — 170 on a
28-layer model, 338 on a 48-layer MoE — so at ~1.1 µs each it is 8–12% of a
token, and `calib/bar_probe.cu` exists to check whether `cg::grid.sync()` is the
cheapest correct way to have one. Measured on an exclusive H100:

| grid | `cg::grid.sync` | counting barrier | + `__nanosleep` | warp-wide spin |
|---|---|---|---|---|
| 132 | **1.11 µs** | 1.29 | 1.30 | 1.30 |
| 264 | **1.34 µs** | 1.75 | 1.73 | 1.72 |
| 396 | **1.51 µs** | 2.06 | 2.07 | 2.04 |

Every block's arrival serialises on one contended cache line; cooperative groups
uses a tree, and its lead grows with the grid. The hand-rolled version stays in
the probe, not in the runtime.

The probe reports a *stale-read count* beside every timing, and that is the
point of it: the first counting barrier tried here spun on `ld.global.cg`,
measured 0.50 µs — less than half of `grid.sync()` — and ordered nothing. It
counted arrivals correctly while 90% of cross-block reads came back stale. A
benchmark that had only measured microseconds would have reported a 55%
improvement in the barrier and a NaN in the model.

## The machine file

`mkc calibrate` measures each gemv core's ceiling and its behaviour inside a
persistent grid. On this H100 (measured streaming ceiling 3190 GB/s):

| core | bytes/value | peak GB/s | at | R=1 → R=4 | note |
|---|---|---|---|---|---|
| bf16 / f16 | 2 | 3217 | any R | 3207 → 3204 | bandwidth bound at every geometry |
| fp8 e4m3 | 1 + scales | 3217 | R=4 | 3127 → 3201 | bandwidth bound; the conversion is free |
| mxfp4 | 0.53 | 2958 | R=4 | 2258 → 2889 | issue bound: row blocking is worth 28% |
| int4 (awq) | 0.53 | 2782 | R=4 | 2290 → 2703 | issue bound; the `0x6400` dequant trick is what gets it here |

The machine file is keyed by *core*, not by the format's exact spelling: fp8 with
a 128×128 block scale and fp8 with a per-output-channel scale run the same inner
loop and differ only in where one scalar comes from, and AWQ and GPTQ int4 are
the same nibbles once the loader has repacked them. Calibrating each spelling
separately would multiply the calibration for nothing — and, worse, leave any
format the calibration happened not to spell exactly right silently falling back
to a proxy curve. It carries a schema version and `mkc` refuses an older file
rather than doing that quietly.

The two issue-bound cores are why the planner has to choose row blocking at all,
and why it has to weigh it against the registers that blocking costs: on those,
R=4 is 28% faster per byte than R=1 but halves the resident warps.

## What the guard actually caught

Not hypothetical. Each of these was a real bug that produced a kernel which
compiled, ran, and looked plausible:

| symptom the guard reported | the bug |
|---|---|
| a weight did not restore bit-exactly after being zeroed | the cross-split softmax reduction combined its per-warp partials with `atomicAdd`, and float addition is not associative: the output depended on warp arrival order |
| every probe NaN, the gate at 0% | a nineteenth `take()` in the runtime's scratch arena ran past an allocation whose slack covered only eighteen roundings, into the weight arena |
| every probe NaN, the gate at 0% | a hand-rolled grid barrier whose spin used `ld.global.cg`: it counted arrivals correctly and ordered nothing |
| every probe insensitive, rope and history too | the loader's dtype dispatch had no FP8 case, so any FP8 matrix needing row interleaving loaded as zeros |
| `gu_w`, `dn_s`, `ffn_norm` insensitive; attention fine | the same, localised to the MLP because only it needed interleaving |
| `lm_head` insensitive | a tied-embedding checkpoint that also ships `lm_head.weight`: 300 MB loaded and never read |
| kernel faulted during the probe | the router's top-k kept a sentinel index when a corrupted weight produced NaN, and indexed the expert table out of bounds |

The last one is worth dwelling on: the guard corrupts weights *on purpose*, and
in doing so found an out-of-bounds read that a well-formed checkpoint would never
trigger but a truncated download might.

## What the profiler caught

The compiler emits its own per-stage timeline (one `%globaltimer` read per
barrier, from block 0). Every number below is a fused, in-kernel measurement, and
none of these were guesses:

| finding | effect |
|---|---|
| `attn_reduce` looped over splits with a data-dependent `continue`, serialising 64 loads | 49% of the token -> 4% |
| `norm_to_smem` read the hidden state twice | ~5 us on every stage that has a norm |
| the grid-stride item map left 40% of SMs idle when items < warps | qkv ran on 160 of 264 blocks |
| a lane-dependent loop bound stopped the gemv unrolling | one load in flight per warp |
| `LayerW` copied into registers | 128 registers where no stage needs 64 |
| L2-prefetching the next stage during the idle ones | measured, **rejected**: 4.4% slower |
| a warp-per-output-chunk cross-split reduction gave the whole GPU 128 work items at 128 key splits | 3% of the resident warps busy; a block per chunk with its warps over the split axis fixed it |
| every warp in that block then recomputed the same softmax normaliser | 2·SPLITS scalar loads and SPLITS exponentials, thirty-two times over, for two floats: the stage fell 64% and the whole token 11.6% when one warp computed it and published it |
| the attention loop issued one key's loads, then stalled on a warp reduction and an exponential | 138 GB/s and 17% of a 48-layer MoE token; two keys at a time doubles the loads in flight |
| the register search stopped at the highest occupancy that fit | one block per SM at 128 registers beat two at 64 by 10%, and no occupancy at all -- letting the assembler choose -- beat both |
