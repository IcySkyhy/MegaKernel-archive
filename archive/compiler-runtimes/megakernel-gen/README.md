# mkc — a megakernel compiler

`mkc` turns a HuggingFace checkpoint into a **single persistent CUDA kernel** that
runs the model's entire forward pass — every layer, the LM head and the sampler —
in one cooperative launch.

```
mkc build /path/to/model -o out/model     # HF checkpoint -> megakernel, one shot
```

No search over timings, no per-model tuning, no tokens burned. Compiling a model
is a **pure function** of the checkpoint and a machine file that is measured once
per GPU type. The same inputs always produce byte-identical CUDA.

## Scope, and what this is not

Read this before the numbers below, because it decides what they mean.

* **One GPU.** No tensor, pipeline or expert parallelism. The largest model here
  (gpt-oss-120b, MXFP4) fits on a single 80 GB card; nothing larger runs at all.
* **Batch 1, decode.** This is a latency system, not a throughput one. There is
  no paged KV cache, no continuous batching, no prefix caching, no speculative
  decoding, and no scheduler. A serving engine is a different program.
* **Prefill is not optimised and vLLM wins it decisively.** Prompt processing
  runs the decode kernel once per prompt token, so a 1024-token prompt reads
  every weight 1024 times -- roughly 35x worse TTFT. Every latency number in this
  README is steady-state decode with prefill differenced out, and that
  measurement is the only one this project has ever tried to win.
* **A research artifact.** One person, one cluster, one GPU type calibrated.

### Prior art

This is not the first megakernel compiler, and it is worth being precise about
who got there first. [**Mirage Persistent Kernel**](https://github.com/mirage-project/mirage)
([arXiv 2512.22219](https://arxiv.org/abs/2512.22219)) is the reference point:
it compiles a tensor program into a megakernel automatically, it is *multi-GPU*,
and it replaces global barriers entirely with an event-driven task graph
scheduled at SM granularity, which lets communication overlap computation instead
of serialising behind a barrier. On all three of those axes MPK is ahead of this
project, and the third is a genuine architectural difference rather than a
missing feature: `mkc` decomposes a token into stages separated by grid-wide
barriers, which is simpler to reason about and strictly less general.

What is less usual here, and is the reason this repository exists separately:

* the input is a **HuggingFace checkpoint directory**, not a traced program, and
  the compiler either emits a kernel or refuses and names the feature it lacks;
* **five weight formats** are calibrated end to end -- bf16, f16, MXFP4, FP8
  e4m3 (block and per-channel), INT4 AWQ -- rather than one;
* compilation is a **pure function** of (checkpoint, machine file): the same
  inputs produce byte-identical CUDA, which `tests/run_tests.sh` checks;
* correctness is defended by an **adversarial guard**, not only a comparison --
  see the last section. That part is independent of this compiler and would work
  against any megakernel exposing the same ABI.

## Why a megakernel

At batch 1 every projection in a transformer is a matrix-*vector* product. Weights
have zero reuse, nothing is compute bound, and the entire forward pass is pure
streaming: the floor is *bytes that must be read / achievable bandwidth*. A normal
engine spends the gap on 100–350 kernel launches per token, each with a ramp-up
and a drain during which the memory system idles. A megakernel deletes the
launches and lets stages that used to be separate kernels share shared memory.

## What it does

```
      HuggingFace checkpoint                  hw/<gpu>.json   (measured once)
      config.json + *.safetensors                   |
               |                                    |
        [ architecture frontend ]                   |
               |  Model IR                          |
        [ planner ] <------------------------------ +
               |  Plan: stages, barriers, row blocking, occupancy, arena
        [ codegen ]
               |
   megakernel.cu  stages.cu  runtime.cu  weights.h  model.h  Makefile
```

* **frontend** — one generic decoder-only reader, not a family of per-model ones.
  Config keys, tensor names and numerical conventions are all *probed*, so a model
  it has never seen lands if it follows the house style, and fails loudly naming
  the missing role if it does not.
* **planner** — every decision a human would reach by sweeping compiled binaries:
  block size, row blocking per stage, how many ways to split the key axis, how many
  expert gemvs to interleave, occupancy, where the barriers fall, arena size. Each
  one is a closed-form evaluation over *measured* primitive rates — with one
  honest exception, marked as such: the cost model cannot rank block size against
  occupancy (its predictions span 10% where the machine spans 60%), so `--verify`
  settles that one by timing every schedule the assembler accepted. See §4.
* **codegen** — emits the composition only. Every gemv core, norm, rotation and
  softmax lives in a hand-written, individually tested runtime library
  (`runtime/include/mk`), so a generated kernel can be wrong in its schedule but
  never in its arithmetic.
* **weights.h** — the weight table, resolved at compile time: which bytes of which
  safetensors file become which device rows. The loader does no name lookup and
  makes no layout decision at run time.

## What it covers

One generic frontend, one planner, one backend. Nothing below is special-cased:
they are fields in the IR that the templates read.

| | |
|---|---|
| architectures | Llama, Qwen2, Qwen3, Qwen2-MoE, Qwen3-MoE, Phi-3, Gemma-2, gpt-oss, and anything else that follows the same tensor-naming conventions |
| attention | MHA, GQA at any ratio, sliding-window (uniform, alternating, or per-layer), attention sinks, logit soft-capping, per-head QK-norm |
| rotary | plain, linear, YaRN, Llama-3; head dims 64 / 96 / 128 / 256; halves and interleaved layouts |
| FFN | dense SwiGLU / GeGLU, mixture-of-experts with softmax or sigmoid routing, softmax-after-topk, shared experts (gated and ungated) |
| norms | RMSNorm, Gemma's `1 + w` variant, LayerNorm, Gemma's four-norms-per-layer placement |
| weights | bf16, f16, **MXFP4**, **FP8 e4m3** (128x128 block scales and per-output-channel), **INT4 AWQ** |
| layout | split or fused qkv, split or fused gate/up, per-expert or fused expert tensors, tied or untied embeddings |

Two combinations are refused rather than lowered, and named as such: a
mixture-of-experts model whose weights are FP8 or INT4 (the top-k expert gemv
interleaves several matrices in one loop and only the bf16/f16 and MXFP4 cores
have that form), and a mixture-of-experts model with a post-FFN norm (the expert
down-projection adds straight into the residual, so there is nowhere to
normalise between).

The contract is that mkc either produces a kernel that passes the gate, or it
**refuses and names the feature it lacks** — never a third thing. `validate.rs`
enforces that, and `tests/run_tests.sh` checks both directions.

## Results

One `mkc build --verify` per model, no tuning, no per-model code. H100 80GB
HBM3, batch 1, 1024-token prompt, 128 decode steps, prefill removed by N-vs-1
differencing, on an exclusive node. `bw util` is bytes-that-must-be-read per
token over **that node's own measured** streaming ceiling — which is not the
3350 GB/s on the spec sheet, and is not even the same on two nodes: three runs of
this table measured 3190, 3095 and 3189 GB/s on identical H100 SXM cards, so the
calibration travels with the job.

The table is generated from the measurements by `scripts/table.py`; it is not
transcribed.

<!-- BEGIN RESULTS TABLE -->
| model | arch | MB/token | weights | ms/token | tok/s | bw util | vs roofline | gate | guard |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-FP8 | Qwen3 | 869 | fp8 block | 0.9455 | 1057.7 | 28.8% | 3.47x | PASS | PASS |
| Qwen3-0.6B | Qwen3 | 1310 | bf16 | 1.0440 | 957.9 | 39.3% | 2.54x | PASS | PASS |
| TinyLlama-1.1B | Llama | 2092 | bf16 | 1.2049 | 830.0 | 54.4% | 1.84x | PASS | PASS |
| Qwen2.5-1.5B-AWQ | Qwen2 | 1192 | int4 awq | 1.4555 | 687.0 | 25.7% | 3.89x | PASS | PASS |
| Qwen2.5-1.5B | Qwen2 | 3117 | bf16 | 1.7418 | 574.1 | 56.1% | 1.78x | PASS | PASS |
| Qwen3-1.7B | Qwen3 | 3559 | bf16 | 1.7834 | 560.7 | 62.6% | 1.60x | PASS | PASS |
| Qwen3-1.7B-FP8 | Qwen3 | 2152 | fp8 channel | 1.4604 | 684.7 | 46.2% | 2.16x | PASS | PASS |
| SmolLM2-1.7B | Llama (MHA) | 3624 | bf16 | 1.7406 | 574.5 | 65.3% | 1.53x | PASS | PASS |
| gemma-2-2b | Gemma-2 | 5338 | bf16 | 2.7612 | 362.2 | 60.6% | 1.65x | PASS | PASS |
| gpt-oss-20b | GptOss MoE | 3735 | mxfp4 | 2.1834 | 458.0 | 53.6% | 1.86x | PASS | PASS |
| Phi-3-mini | Phi-3 | 7848 | bf16 | 3.2988 | 303.1 | 74.6% | 1.34x | PASS | PASS |
| Qwen1.5-MoE-A2.7B | Qwen2-MoE + shared | 4957 | bf16 | 2.4706 | 404.8 | 62.9% | 1.59x | PASS | PASS |
| Qwen3-8B | Qwen3 | 15288 | bf16 | 5.8859 | 169.9 | 81.4% | 1.23x | PASS | PASS |
| Qwen3-30B-A3B | Qwen3-MoE | 6185 | bf16 | 3.9489 | 253.2 | 49.1% | 2.04x | — | PASS |
| gpt-oss-120b | GptOss MoE | 5043 | mxfp4 | **3.1378** | 318.7 | 50.4% | 1.98x | — | PASS |
<!-- END RESULTS TABLE -->

The gate column is a teacher-forced comparison against `transformers` on the same
checkpoint, and the two largest models have no entry rather than a claim.

The reference and the engine both want the whole model resident, and above about
8 B parameters they stop fitting side by side. For Qwen3-30B-A3B that is
solvable — `harness/run.py --ref-first` computes the reference and frees it
*before* the engine loads — and the next run will carry it. For gpt-oss-120b it
is not solvable on this machine at all: dequantised to bf16 for the reference it
is 240 GB, more than the card and more than the node's memory. Both get the full
guard, and both architectures are gated on their smaller siblings — gpt-oss-20b
is the same code path end to end (same MXFP4 core, same fused expert tensors,
same attention sinks, same YaRN rope, same sliding-window pattern) at 5.9 ulp.
Writing "PASS" in those two cells would have been a claim about a measurement
nobody took.

The pattern is the one the roofline predicts: utilisation rises with model size,
because a bigger model means longer streams per stage, and a stage that is one
partial wave of warps cannot reach steady-state bandwidth. Qwen3-8B, whose MLP
stages are 100–200 MB, runs at 81% of the measured ceiling; Qwen3-0.6B, whose
stages are 4–12 MB, runs at 39%. The mixture-of-experts models sit in between for
the same reason — a 48-layer MoE with a small hidden dimension reads only
20–50 MB per stage — which is exactly what a per-stage profile shows.

### Head to head

Same GPU, same 1024-token prompt, same 128 decode steps, batch 1, prefill removed
by N-vs-1 differencing so that nothing in the measurement is prefill or a prefix
cache. The last column is the engine's latency over the megakernel's.

<!-- BEGIN H2H TABLE -->
| model | engine | ms/token | tok/s | |
|---|---|---|---|---|
| Qwen3-8B  *dense, bf16* | vLLM | 6.5016 | 153.8 | 1.10× |
|  | SGLang | 6.4358 | 155.4 | 1.09× |
|  | **mkc** | **5.8859** | **169.9** | — |
| gpt-oss-20b  *MoE, MXFP4* | vLLM | 3.3019 | 302.9 | 1.51× |
|  | SGLang | 3.2932 | 303.7 | 1.51× |
|  | **mkc** | **2.1834** | **458.0** | — |
| gpt-oss-120b  *MoE, MXFP4* | vLLM | 4.7433 | 210.8 | 1.51× |
|  | SGLang | 4.7671 | 209.8 | 1.52× |
|  | **mkc** | **3.1378** | **318.7** | — |
<!-- END H2H TABLE -->

Two engines, two models, two independent runs each: vLLM and SGLang land within
0.1% of themselves across runs, which is the check that the N-vs-1 differencing
is sound. The one number that moves is vLLM on gpt-oss-120b — 4.74 ms on these
nodes at two different memory fractions, 4.46 ms on another node in an earlier
run — so the honest range against it is **1.42–1.51×**, and the conservative
figure is the one to quote.

For gpt-oss-120b there is also a longer history, because this compiler
generalises a hand-written megakernel for exactly that model:

| engine | ms/token | tok/s | notes |
|---|---|---|---|
| PyTorch reference | 113.37 | 8.8 | recomputes the sequence every token |
| hand-written per-op CUDA kernels | 4.31 | 232 | ~320 launches per token |
| hand-tuned megakernel | 3.68 | 272 | weeks of sweeping, this model only |
| **mkc, one shot** | **3.14** | **319** | `mkc build --verify`, no model-specific work |

The comparison is not apples to apples and it is not in mkc's favour to pretend
otherwise: vLLM is a general serving engine with paged KV, continuous batching
and arbitrary sampling, and this is a fixed batch-1 greedy decode path. Giving up
that generality is the entire point of a megakernel. Prefill here is *not*
optimised — it runs the decode kernel once per prompt token — and vLLM wins it
decisively. The rows that matter for the claim are the last two: a generated
kernel matching, and slightly beating, a hand-tuned one on the model the
hand-tuning was done for.

## Layout

```
mkc/            the compiler (Rust, serde only)
  hf.rs           safetensors + config ingestion
  arch.rs         HuggingFace -> Model IR
  ir.rs           the architecture-neutral IR
  hw.rs           the machine model and its cost interpolation
  plan.rs         the planner: cost model + schedule
  search.rs       `mkc build`: which schedules the assembler will place
  codegen.rs      CUDA emission
  layout.rs       the weight table: file bytes -> device rows, at compile time
  emit.rs         output tree: host runtime, Makefile, bench driver
runtime/include/mk/   the primitive library the generated code calls
calib/                the only places measurement enters:
  calib.cu              the machine model -- run once per GPU type
  bar_probe.cu          is the grid barrier as cheap as it can be?
  tp_probe.cu           what a cross-GPU rendezvous would cost
harness/              the frozen correctness gate, the benchmark, the guard
bench/                vLLM and SGLang on the identical workload
hw/                   machine files, one per GPU type
scripts/              zoo.sh (compile+gate+bench a list of models), table.py
                      (rewrites the table above from the TSV), delta.py
                      (compares two runs), and the slurm jobs that produce every
                      number: slurm_calib, slurm_final, slurm_occ, slurm_bar,
                      slurm_tp
```

Nothing that will be reported is ever timed on a shared GPU: the local cards
have other people's work on them, and a neighbour at 50% occupancy moves every
number. That is what the slurm scripts are for.

## Running

```bash
cp env.local.sh.example env.local.sh   # set MKGEN_WORK; nothing else is site-specific
source env.sh
cargo build --release --manifest-path mkc/Cargo.toml

# once per GPU type
nvcc -O3 -arch=sm_90a -Iruntime/include -o calib calib/calib.cu
./calib hw/nvidia_h100_80gb_hbm3.json

mkc analyze $MKGEN_MODELS/Qwen3-8B                      # parse + roofline, no GPU
mkc build   $MKGEN_MODELS/Qwen3-8B -o out/Qwen3-8B      # compile; register fixpoint
mkc build   $MKGEN_MODELS/Qwen3-8B -o out/Qwen3-8B --verify   # + time every candidate

python harness/run.py out/Qwen3-8B --model $MKGEN_MODELS/Qwen3-8B --guard
python harness/mkprofile.py out/Qwen3-8B --model $MKGEN_MODELS/Qwen3-8B   # per-stage schedule
tests/run_tests.sh

# every stage is ALSO emitted as an ordinary kernel, so a profiler can see one
# in isolation instead of a single 6 ms launch:
ncu --set full --kernel-name kst_mlp1 \
    python harness/ncu_stage.py out/Qwen3-8B --model $MKGEN_MODELS/Qwen3-8B --stages mlp1
```

`mkc build` never runs a kernel; it compiles candidate schedules only to learn
what the assembler gives them in registers. `--verify` additionally times each of
them on *synthetic* weights — timing depends on the layout, not the values, so it
needs no checkpoint and costs seconds — and keeps the measured winner.

That flag is where the compiler admits the limit of its cost model, and the size
of the admission is worth stating: across (block size, blocks per SM) the
predicted cost spans about 10% while the measured cost spans 60%. A bytes-and-
items model cannot see why one block per SM at 128 registers beats two at 64, or
why omitting `minBlocksPerMultiprocessor` entirely — letting the assembler pick
the occupancy — beats every explicit choice on three of four models measured. So
the search enumerates those schedules, the assembler decides which are legal, and
`--verify` decides which is fastest. Bounded, deterministic, and still one shot.

## End to end

The gate is teacher-forced, which is the right way to measure numerical
agreement but never exercises the sampler or the token feedback path.
`harness/generate.py` does: it tokenises a prompt, runs the megakernel with
greedy sampling so that each step's argmax becomes the next step's input
entirely on the GPU, and detokenises.

```
$ python harness/generate.py out/Qwen2.5-1.5B-AWQ --model $MKGEN_MODELS/Qwen2.5-1.5B-Instruct-AWQ \
      --chat --prompt "Write one sentence about the ocean." --n 28
prompt   : '<|im_start|>system\nYou are Qwen ... <|im_start|>assistant\n'
generated: "The ocean is a vast and mysterious expanse of water covering most of the
            Earth's surface, supporting diverse ecosystems and shaping the planet's climate"
36 prompt + 28 decode tokens in 88 ms
```

`--compare` also decodes greedily with `transformers` and reports where the two
streams diverge. On Qwen1.5-MoE they are token-identical for the whole
generation. That is a demonstration, not a gate: a single near-tied argmax sends
free-running decoders down different paths, which is exactly why the gate is
teacher-forced.

## Documentation

| | |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | how it works, and why each decision is where it is |
| [docs/EXTENDING.md](docs/EXTENDING.md) | adding an architecture, a weight format, a stage, or a GPU |
| [docs/RESULTS.md](docs/RESULTS.md) | measurement method, and what the gate, guard and profiler each caught |

## Correctness, and why it is not negotiable

Latency is trivially cheatable: the fastest kernel is the one that computes
nothing. So the metric is guarded twice.

**The gate.** Teacher-forced logit comparison against HuggingFace `transformers`
running the same checkpoint — third-party code this compiler did not write and
cannot influence. Both are fed the *same* fixed token sequence, drawn randomly and
seeded from the clock, and their logits are compared step by step. Free-running
greedy decode is not a gate: one near-tied argmax on step 3 makes every later token
differ and reports catastrophic failure for a 1-ulp difference.

**The guard** (`harness/mkguard.py`) attacks the ways a fast kernel could pass the
gate without doing the work:

| check | what it would catch |
|---|---|
| `weight_sensitivity` | zeroes each weight tensor in turn and requires the output to move — a layer that is never read, an expert branch that is skipped |
| `history_sensitivity` | the KV cache is actually consulted |
| `rope_applied` | zeroes the rotary tables, requires the output to move, and requires the model back bit-for-bit when they are rebuilt |
| `input_sensitivity` | no memoisation |
| `determinism` | bit-identical across runs; a difference means a race, and a race means the correctness result was luck |
| `timing_floor` | measured ms/token cannot be below the roofline of the bytes the model must read |
| `no_reference_link` | the shared object links no torch/transformers and the sources contain no captured output |

The weight probe restores the bytes by re-reading them from the checkpoint and
asserts the restored model is bit-identical, so the check cannot corrupt the run
it is checking.

## The one measurement

`mkc calibrate` runs once per GPU type and writes `hw/<gpu>.json`:

1. **streaming read vs cold footprint** — every repetition reads memory it has not
   just read, because this GPU has a 52 MB L2 and a naive loop over a 32 MB
   footprint measures the cache, not the memory.
2. **every gemv core at every (block size, row blocking)** — its ceiling.
3. **the same cores in the shape a megakernel runs them**: inside a persistent
   cooperative grid, one stage of B bytes and N items, ending at a grid barrier,
   at two reduction lengths. A stage is usually one partial wave, where the limit
   is memory-level parallelism rather than bandwidth; conflating the two
   mispredicts by 4x.
4. **cooperative grid barrier cost vs grid size** — barriers × stages × layers is a
   fixed tax that decides how coarse the stages must be.
5. **kernel launch overhead** — the thing a megakernel exists to delete.

After that file exists, compiling a model touches no GPU.

## License

Apache-2.0. See [LICENSE](LICENSE).
