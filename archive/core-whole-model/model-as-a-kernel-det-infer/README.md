---
library_name: kernels
license: apache-2.0
---

# det-infer

Architecture-invariant bitwise LLM inference, loadable through `kernels`:
the [model-as-a-kernel](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel)
engine with every architecture-dependent operation removed, so logits and
greedy token streams are bit-identical on every CUDA card. The reference
standard is a set of pinned SHA-256 digests reproduced across nine
card-torch-toolchain combinations.

The same model, the same prompt, and the same seed do not normally produce
the same bits on two different GPUs: reduction orders and the vendor math
library differ per architecture, so results drift and nothing is exactly
reproducible. This kernel removes both sources, fixing every reduction tree
and running the transcendentals through IEEE-only implementations, so an
inference result becomes a permanent object: an L4, an H200, and a Blackwell
workstation emit identical bytes, indefinitely, and a batched sequence's
logits are identical to running it alone.

![The fingerprint stream of 96 hashed logit steps, the live digest matching the pinned certificate character for character, and a row of architectures reading same bytes](https://huggingface.co/kernels/phanerozoic/det-infer/resolve/main/media/hero.gif)

*The certificate reproduced live: 96 teacher-forced steps of SmolLM2-135M
hashed to SHA-256, and the digest equals the pinned constant minted for the
cross-architecture set (L4, H200, RTX PRO 6000 all report the same bytes),
at 404 deterministic tokens per second.*

## Usage

```python
from kernels import get_kernel

di = get_kernel("phanerozoic/det-infer", version=1, trust_remote_code=True)

mm = di.MegaModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokens = mm.generate(prompt_ids, max_new=128)   # one kernel launch, total
logits = mm.decode_step(token_id, pos)          # one launch, fp32 [V] logits
streams = mm.generate_batch([ids_a, ids_b, ids_c], max_new=128)
```

`version` selects the release branch; `trust_remote_code` is required by
`kernels` for publishers without the trusted-publisher mark. The API is
identical to model-as-a-kernel.

## API

| Symbol | Purpose |
|---|---|
| `MegaModel.from_pretrained(model_or_id, device, max_seq, max_gen)` | pack a checkpoint into a program |
| `MegaModel.from_gguf(path)` | load a GGUF checkpoint, dequantized to bf16 at load |
| `decode_step(token, pos)` | one decode step, live fp32 logits `[V]` |
| `prefill(ids)` / `generate(prompt_ids, max_new, single_launch=True)` | chunked prompts; one-launch generation |
| `generate_batch(prompts, max_new)` / `decode_batch(...)` | batched greedy decode, per-sequence invariant |
| `decode_step_timed(token, pos)` | phase-per-launch mode, per-phase ms |
| `ops.mak_*` | raw program launches |

Supported: llama-family architectures with GQA, optional qk-norm, plain
rope, plus the gemma-4-E2B architecture; dense bf16 or bitsandbytes nf4
weights; GGUF checkpoints of any quant type via the gguf reader.

## Method

GPU floating-point add, multiply, fma, divide, and sqrt are IEEE bit-exact
on every architecture; what varies is reduction order and libdevice. The
engine fixes reduction order by construction (fixed trees, lane-striped
accumulation, launch geometry that never touches arithmetic), and det-infer
replaces the math library: exp runs in double-float fp32 pairs (Dekker
products, Knuth sums, a Cody-Waite ln2 split, one rounding at the collapse);
tanh builds on that exp; sin and cos use an fp64 two-term reduction with
double-float polynomials; rsqrt runs through fp64 sqrt and divide.
Accumulations use explicit round-per-op intrinsics so ptxas cannot contract
them differently per target. Rope frequency tables come from a fixed
50-digit decimal recipe rather than any host math library. nf4 weights
dequantize inside the GEMV through integer indexing and IEEE multiplication.

## Measured

The deterministic math costs about 8 percent of decode throughput on
SmolLM2-135M (824 vs 898 tok/s closed loop against model-as-a-kernel).
Consumer cards issue fp64 at 1/64 rate, so the hot transcendentals run in
double-float arithmetic on the full-rate fp32 units. Batched decode scales
Llama-3.2-1B from 256 tok/s at batch 1 to 1673 at batch 16 (6.5x), with
per-sequence bitwise invariance to batch composition.

## Correctness

Pinned logits and greedy-token digests for SmolLM2-135M, Qwen3-0.6B,
Llama-3.2-1B, Llama-3.2-3B, zephyr-7b-beta, and gemma-4-E2B-it, minted on
RTX 6000 Ada (sm89), reproduce byte-for-byte on L4 (sm89), H200 (sm90), and
RTX PRO 6000 Blackwell (sm120), under torch 2.11 through 2.13, across
compiled variants spanning cu126 to cu132: nine combinations against one
digest set. The digests are further invariant to launch geometry (64 to 284
blocks, ring depths 1 to 8), to concurrent device load, and along 1500-token
generations. Batch invariance is exact: a sequence's fp32 logits are bitwise
identical alone or inside a batch of 16. Against transformers eager,
teacher-forced argmax agreement matches model-as-a-kernel (Llama-3.2-1B
64/64). Fused, phase-per-launch, and repeated runs are mutually bitwise
identical.

## Requirements and limits

- One CUDA device, compute capability 8.0+; greedy in-kernel, external
  sampling via the fp32 logits.
- Positions to one million (the Cody-Waite rope reduction bound); batch to
  16 on the llama-family path, batch 1 on gemma.
- About 8 percent decode cost against the non-invariant engine.

## References

Dekker (1971) and Knuth error-free transformations; Cody-Waite range
reduction; IEEE 754 bit-exactness of basic GPU operations; the pinned-digest
certification method.

## License

Apache-2.0.
