---
library_name: kernels
license: apache-2.0
---

# model-as-a-kernel

A llama-family language model as one kernel launch, loadable through
`kernels`. The reference baseline is transformers eager, matched op-bitwise
where semantics are exactly defined and within transformers' own
execution-path spread on whole-model logits.

An eager greedy generation of a small model is roughly 180 kernel launches
per token, and at 135M parameters the launches, not the arithmetic, are the
wall. This kernel runs the full forward pass, embedding, norms, projections,
rope, attention over the KV cache, MLP, LM head, and greedy argmax, inside
one persistent phase-interpreter kernel that crosses a software grid barrier
between phases instead of returning to the host, and an in-kernel step loop
carries the barrier across token boundaries. An entire generation, prompt
consumption included, is a single `cudaLaunchKernel`, and decode runs an
order of magnitude past the eager loop.

![A typing race: the eager lane crawls with a dense launch comb while the one-launch lane finishes its whole generation](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel/resolve/main/media/hero.gif)

*SmolLM2-135M decoding live at true relative speed (clock slowed 6x): the
eager loop types at 39 tok/s behind thousands of kernel launches; the
persistent kernel types at 805 tok/s behind exactly one, with repeated runs
token-identical.*

## Usage

```python
from kernels import get_kernel

mak = get_kernel("phanerozoic/model-as-a-kernel", version=1, trust_remote_code=True)

mm = mak.MegaModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")

tokens = mm.generate(prompt_ids, max_new=128)     # one kernel launch, total
logits = mm.decode_step(token_id, pos)            # one launch, fp32 [V] logits
logits = mm.prefill(prompt_ids)                   # chunked prompt consumption
streams = mm.generate_batch([ids_a, ids_b], max_new=128)   # batched decode
```

`version` selects the release branch; `trust_remote_code` is required by
`kernels` for publishers without the trusted-publisher mark.

## API

| Symbol | Purpose |
|---|---|
| `MegaModel.from_pretrained(model_or_id, device, max_seq, max_gen)` | pack a checkpoint into a program |
| `decode_step(token, pos, phased=False)` | one decode step, live fp32 logits `[V]` |
| `prefill(ids)` | chunked prompt consumption |
| `generate(prompt_ids, max_new, single_launch=True)` | greedy generation; one launch total, or one per token |
| `generate_batch(prompts, max_new)` / `decode_batch(tokens, positions, steps)` | batched greedy decode, up to `batch_max()` sequences |
| `decode_step_timed(token, pos)` | phase-per-launch mode, per-phase ms |
| `ops.mak_*` | raw program launches |

Supported: RMSNorm decoder blocks with rotary attention (GQA, optional
Qwen3-style qk RMSNorm), SwiGLU MLP, plain unscaled rope, single device;
SmolLM2, TinyLlama, Qwen3 dense, and Llama-class checkpoints without rope
scaling. Weights dense bf16 or bitsandbytes nf4.

## Method

Model load packs the weights and compiles the architecture into an int64
`[n_phases, 16]` program naming an op per row with its pointers and sizes.
One persistent kernel interprets the program; all resident blocks execute
each phase cooperatively and cross a sense-reversing global barrier, with
the grid sized from the occupancy API so the barrier cannot deadlock.
Arithmetic reproduces transformers eager bf16 semantics op for op, with
fixed reduction trees and no floating-point atomics, so outputs are bitwise
deterministic. GEMV weights stream through a per-warp `cp.async`
shared-memory ring; attention splits cached positions into 128-token chunks
with the softmax combine folded into the O-projection's staging; prompts are
consumed in chunks of up to eight tokens, bitwise identical to
token-by-token consumption. A phase-per-launch debug mode is bitwise
identical to the fused mode.

## Measured

Greedy decode, 128 tokens with a 32-token prompt, against transformers eager
and a compiled decode loop (torch.compile reduce-overhead, StaticCache, CUDA
graphs):

SmolLM2-135M bf16:

| GPU | this kernel | eager | compiled static |
|---|---|---|---|
| RTX 6000 Ada (sm89) | 902 tok/s | 35 | n/a (Windows) |
| H200 (sm90) | 715 tok/s | 52 | 586 |
| RTX PRO 6000 (sm120) | 726 tok/s | 55 | 721 |

Llama-3.2-1B bf16:

| GPU | this kernel | eager | compiled static |
|---|---|---|---|
| H200 (sm90) | 490 tok/s | 99 | 622 |
| RTX PRO 6000 (sm120) | 368 tok/s | 98 | 403 |

Steady-state decode at position 512 on Llama-3.2-1B: 1.85 ms/token on H200
(1.62 TB/s effective weight bandwidth). Prefill: a 2040-token prompt costs
0.73 s against 2.7 s token by token, bitwise-identical logits either way.
Batched decode scales Llama-3.2-1B from 256 tok/s at batch 1 to 1673 at
batch 16 (6.5x aggregate), with each sequence's logits bitwise identical to
decoding alone.

## Correctness

Verified against transformers eager through `get_kernel` on L4, H200, and
RTX PRO 6000, and from source on RTX 6000 Ada. Bitwise (`torch.equal`): the
embedding read, fused rmsnorm+GEMV, rope tables and rotated K, attention at
S=1, and the layer-0 KV cache along a 48-token sequence against
`DynamicCache`. Whole-model teacher-forced logits sit within transformers'
own execution-path spread (mean 0.12-0.15, argmax agreement 93-96 of 96,
matching the spread between transformers' own eager and sdpa paths). Fused,
phase-per-launch, and repeated runs are mutually bitwise identical, and the
single-launch generation emits the same tokens as one launch per token.

## Requirements and limits

- One CUDA device; greedy sampling in-kernel (route the fp32 logits to an
  external sampler for anything else).
- Llama-family architectures without rope attention scaling or attention
  biases; prompts prefill in chunks of at most eight tokens.
- The program bakes device pointers, so the `MegaModel` owns its weights and
  buffers for its lifetime.

## References

Persistent-kernel programming and software grid barriers (cooperative
groups); the launch-overhead analysis of small-model decode; transformers
eager semantics as the reference.

## License

Apache-2.0.
