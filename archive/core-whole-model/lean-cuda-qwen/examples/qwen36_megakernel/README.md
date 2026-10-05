# Qwen3.6-27B Lean CUDA megakernel

This example is the first exact-geometry slice of a plain-Lean CUDA implementation of
[`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B). It does not use Tyr, LibTorch, or a
handwritten C++ kernel.

The checkpoint's language model has 64 layers in a repeating three-linear/one-full-attention
schedule. The 48 linear-attention layers use a recurrent Gated DeltaNet with 16 query/key heads,
48 value heads, and 128 channels per head. `Qwen36Megakernel.lean` implements that exact recurrent
decode rule as a 48-block `@[cuda_grid_persistent]` worker:

- query and key remain in the checkpoint's compact 16-head layout;
- three resident value-head blocks reuse each compact query/key head;
- the FP32 recurrent state has shape `[48, 128, 128]`;
- queued token descriptors update the state and publish `[48, 128]` outputs without ending the
  launch;
- all tiling, reductions, state traversal, rank-one updates, and synchronization are Lean.

The executable runs three tokens through one resident launch and through three conventional
launches of the same Lean device body. It requires byte-identical recurrent state and output
between those routes, then compares every FP32 element with an independent host Lean replay. The
runner also asks ptxas for resource usage; the initial GB10 implementation uses no stack or spills.

`Qwen36TrainingMegakernel.lean` adds the analytic VJP for that recurrent step in the same
resident launch. It differentiates query, key, value, log-decay, beta, and the incoming recurrent
state; the GPU result is checked against both ordinary Lean CUDA launches and a host Lean oracle.
Compact Q/K gradients correctly accumulate the three value heads with FP32 atomics.

`Qwen36LinearLayerMegakernel.lean` connects the inference recurrence to the exact 10,240-channel
depthwise causal convolution, sigmoid/softplus gates, persistent convolution cache, and per-head
gated RMSNorm. Three tokens stay in one cooperative-grid launch and match the split-launch route
plus an independent host replay.

`Qwen36ProjectionMegakernel.lean` provides a descriptor-driven BF16 matrix-vector engine and
validates it at the checkpoint's full `[10240, 5120]` recurrent input-projection shape. The same
resident body accepts rows, columns, and arena offsets for the other decode projections.

`Qwen36MLPMegakernel.lean` executes post-attention RMSNorm, both full `[17408, 5120]`
projections, SwiGLU, the `[5120, 17408]` down projection, and residual add in one descriptor. Its
validation reads about 510 MiB of BF16 matrix storage at the exact checkpoint dimensions.

`Qwen36AttentionMegakernel.lean` covers the 24-query/4-KV-head full-attention decode core,
head-local RMSNorm, 64-channel partial RoPE, six-way grouped-query mapping, sigmoid output gate,
and a persistent BF16 KV cache over four tokens.

`Qwen36TokenMegakernel.lean` covers the exact `[248320, 5120]` embedding address space, final
RMSNorm, stable 248320-way softmax cross-entropy and its logit VJP, and greedy sampling. The
resident and finite cooperative routes are byte-identical and checked against a host Lean oracle.

`Qwen36Checkpoint.lean` is a native Lean safetensors loader. It validates sharded index/header/file
sizes without loading shard payloads. Model-scale loads use at most a 64 MiB host chunk and copy it
straight into the correct byte range of one CUDA allocation with `Cuda.Buffer.copyFromAt`; the
loader never materializes a whole multi-gigabyte shard as a `ByteArray`. Its opt-in LeanTest gate
checks all 15 published shards and 1199 tensors plus bytes from a real BF16 checkpoint tensor.

`Qwen36ModelMegakernel.lean` resolves the published weights for all 64 layers and runs a real
first-position forward pass, loss, and greedy sample. It repeats the complete GPU execution and
requires byte-identical logits, loss, and sample. On the GB10 validation host, bounded loading held
process RSS near 697 MiB through shard 15; the former whole-shard loader reached about 68.8 GiB RSS
and was killed by the host OOM killer. The bounded run returned sample 19 and loss bits 1087776610
for token/target 1234, with the first model execution taking 360 ms.

Run it with:

```bash
export LEAN_CUDA_ROOT=/path/to/lean4-cuda-backend
./run.sh
./run-training.sh
./run-linear-layer.sh
./run-projection.sh
./run-mlp.sh
./run-attention.sh
./run-token.sh
QWEN36_MODEL_DIR=/path/to/Qwen3.6-27B ./run-model.sh
```

The model runner requires a Lean CUDA backend containing `Cuda.Buffer.copyFromAt` (commit
`714873fa0b` or a descendant). The backend's `feat-cuda-moe-abstractions` branch already contains
that primitive.

Set `LEAN_CUDA_ARCH=sm_90a` for Hopper compile validation. Set
`LEAN_CUDA_COMPUTE_SANITIZER=1` to add a memcheck run.

## Scope still required for the full objective

The example now has a real 64-layer first-position inference path, but it is not yet an
autoregressive chat runtime or a full training implementation. The remaining work is explicit:

- tokenizer-backed prompt ingestion, a chunked/prefill DeltaNet schedule, and multi-token
  autoregressive decode;
- numerical logit/loss parity against Transformers for the published checkpoint, beyond the
  current repeated-execution determinism gate;
- projection, convolution, norm, attention, MLP, embedding, and LM-head VJPs beyond the
  now-validated recurrent and exact-vocabulary loss VJPs;
- optimizer state and an end-to-end training loop;
- performance work beyond the current correctness-first descriptor schedule.

## LeanTest suite

The component GPU correctness gates plus the native safetensors parser and bounded-payload tests
are registered through the pinned LeanTest framework under `tests/leantest`. Run them sequentially
with the CUDA-toolchain wrapper:

```bash
tests/leantest/run.sh --fail-fast
```

The real 52 GiB checkpoint metadata/tensor gate is intentionally opt-in:

```bash
QWEN36_MODEL_DIR=/path/to/Qwen3.6-27B \
  tests/leantest/run.sh --filter realCheckpointManifestAndTensorStream --ignored --fail-fast
```

The full first-position model gate is separately opt-in and requires the newer backend:

```bash
LEAN_CUDA_ROOT=/path/to/lean4-cuda-backend-with-copyFromAt \
QWEN36_MODEL_DIR=/path/to/Qwen3.6-27B \
  lake test -- --filter realCheckpointInferenceMegakernel --ignored --fail-fast
```

The implementation deliberately starts with the recurrent core because it is used by three
quarters of the language layers and its persistent state is the defining difference from a
conventional decoder-only transformer.
