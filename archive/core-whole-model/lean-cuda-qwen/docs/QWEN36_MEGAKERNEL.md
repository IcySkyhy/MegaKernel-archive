<!--
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-->

# Qwen3.8-27B megakernel training and inference (design ledger)

Goal: implement the Qwen3.8-27B text model (hybrid Gated DeltaNet + gated full attention)
as Lean-written CUDA kernels on the `feat-cuda-backend` compiler, with a persistent
single-launch decode megakernel for inference and a fully device-resident training step
(batched forward + backward + optimizer update, including mixed-adapter LoRA), following the
`cuda_moe_shakespeare` trainer pattern. Small configurations are gated against a NumPy oracle;
the 27B shape is the deployment target.

This file is the coordination contract for all workstreams: module layout, math
reference, fixture formats, tolerances, and the build recipe. Keep it current when
any of these change.

## Source of truth

- HF repo: `https://huggingface.co/Qwen/Qwen3.8-27B` (revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; `config.json`; architecture
  `Qwen3_5ForConditionalGeneration`, text config model_type `qwen3_5_text`).
- Reference implementation: `transformers.models.qwen3_5.modeling_qwen3_5`
  (vendored excerpt in `tests/workload_regressions/cuda_qwen36_reference/modeling_qwen3_5_excerpt.py`).
- Text-only scope. The vision tower, mrope interleaving (collapses to standard RoPE for
  text), and the MTP layer (`mtp_num_hidden_layers=1`) are out of scope for the first
  implementation; record follow-ups in `dev/issues.md`.

## Architecture facts (text model)

- 64 decoder layers; `layer_types[i] == "full_attention"` iff `i % 4 == 3`
  (pattern: 3 x GatedDeltaNet then 1 x full attention, x16).
- hidden 5120, vocab 248320 (untied embeddings), SwiGLU FFN intermediate 17408,
  `rms_norm_eps = 1e-6`, weights bf16, SSM state fp32 (`mamba_ssm_dtype`).
- RMSNorm is the zero-init `(1 + w)` variant: `y = (x * rsqrt(mean(x^2)+eps) * (1+w)).to(dtype)`.
- Decoder layer: `h = x + mixer(norm(x)); h = h + mlp(norm(h))` (pre-norm, both norms
  the `(1+w)` variant). Final `norm` then `lm_head`.

### Gated DeltaNet (linear_attention layer)

Dims: `num_k_heads=16, head_k=128 -> key_dim=2048`; `num_v_heads=48, head_v=128 ->
value_dim=6144`; `conv_dim = 2*key_dim + value_dim = 10240`; conv kernel 4, depthwise,
no bias, SiLU activation.

Forward (per token / sequence):
1. `qkv = W_qkv x` (10240), `z = W_z x` (6144), `b = W_b x` (48), `a = W_a x` (48).
2. Causal depthwise conv1d(k=4) over `qkv` channel sequence, then SiLU.
3. Split `q,k` -> 16 heads x 128, `v` -> 48 heads x 128. L2-normalize `q,k` per head
   (eps 1e-6). `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)` (per v-head,
   fp32). `repeat_interleave` q,k by 3 to 48 head slots.
4. Delta rule with scale `1/sqrt(128)`, fp32 state `S[h] : [128 k x 128 v]`:
   recurrent (decode): `S *= exp(g); kv = S^T k; S += k (beta (v - kv))^T; o = S^T q`.
   Chunked form (prefill/training, chunk 64) follows `torch_chunk_gated_delta_rule`
   exactly (in-chunk cumsum decay, UT-transform forward substitution, v_new, state
   update with per-chunk decay).
5. Output per head: gated RMSNorm over head_v=128 (unit-init weight, NOT the (1+w)
   variant): `o = rmsnorm(o) * silu(z)`; flatten 6144; `out = W_out o` (5120).

### Gated full attention (full_attention layer)

- `head_dim=256`, `n_q=24`, `n_kv=4` (GQA ratio 6), scaling `256^-0.5`.
- `W_q : hidden -> 2 * 24 * 256` (second half is the output gate), `W_k, W_v : hidden ->
  4 * 256`, no biases.
- Per-head RMSNorm `(1+w)` variant on q and k (head_dim 256) BEFORE RoPE.
- Partial RoPE: rotary dim = `256 * 0.25 = 64` (first 64 of 256), theta 1e7,
  GPT-NeoX half-rotation layout (`rotate_half`, interleaved mrope is a no-op for text).
- Causal attention (fp32 softmax), GQA repeat, output `*= sigmoid(gate)`, then `o_proj`.

## Fully megakernel-shaped training and batched LoRA

Training is not complete when individual kernels compile or when a host submits a graph of
per-operation launches. One persistent device-graph launch owns a complete `batch x sequence`
step: ingress, forward, loss, backward, gradient reduction, optimizer update, precision
publication, and the device-tail transition to the next step. Host code may initialize buffers,
start the graph, and stop it; it does not sequence layers or operations between steps.

LoRA is part of that same graph rather than a separate fallback path:

- Each sequence carries an `adapterId`; a batch may mix adapters, repeat one adapter, or use a
  no-adapter sentinel. Reordering the batch must not change adapter-local results.
- Base checkpoint weights remain frozen. For each enabled projection, the fused projection is
  `y = W x + scale * B[adapterId] * (A[adapterId] * x)`, with `scale = alpha / rank`; the dense
  delta matrix is never materialized. Rank is runtime-configured but must satisfy the checked
  tile contract (positive and a multiple of 16 for the production kernels).
- The initial production family covers DeltaNet qkv/z/a/b/out, full-attention q/k/v/o, MLP
  gate/up/down, and LM head. Embeddings and normalization vectors remain frozen in the first
  LoRA profile.
- Backward includes the LoRA contribution to `dx` and accumulates `dA`/`dB` by adapter without
  cross-adapter contamination. Adapter gradients and optimizer state stay device-resident;
  AdamW is the default LoRA optimizer while full-model matrices retain tile-Muon where enabled.
- DeltaNet recurrent/conv state and attention KV state are indexed by batch row, never by
  adapter, so two sequences using the same adapter still have independent model state.

The tiny acceptance gate uses batch 4 with two adapters in an interleaved `0,1,0,1` assignment.
It must match the NumPy oracle for forward values, `dx`, `dA`, `dB`, and one optimizer update;
prove batch-permutation invariance after undoing the permutation; prove no update to frozen base
weights or the other adapter when one adapter is masked; and lower mean CE over a short recurrent
training slice. The megakernel result must match the sequential launch of the identical device
bodies byte-for-byte.

The production lifecycle is owned by reusable library abstractions rather than test setup:
`Projection.AdapterF32.allocate` creates routing, rank scratch, gradients, masks, and AdamW
moments, initializes A deterministically with Philox, and zeros B so the initial delta is exactly
zero. Adapter and schedule snapshots retain parameters plus optimizer state for exact resume.
`Training.ResidentTrainerF32` instantiates the complete step with its device-tail node and consumes
`Descriptor.steps` from one host launch; its completion check verifies both the tail counter and
the device-resident AdamW step.

Pretraining and GRPO now have distinct cooperative CUDA entry programs instead of a runtime
objective switch. `PretrainMegakernel.pretrainingProjectionStepF32Kernel` performs LoRA forward,
stable mean cross-entropy, the logit VJP, deterministic A/B VJPs, and masked AdamW in one launch.
`GRPOMegakernel.grpoProjectionStepF32Kernel` performs current-policy scoring, group-relative reward
normalization, clipped importance ratios, sampled direct KL, the completion-masked logit VJP,
the same adapter VJPs, and AdamW in one launch. Both consume nested launch structures deriving
`Cuda.POD` and share only device-phase bodies from `TrainingMegakernelCore`; neither generated
program contains the other objective. The CUDA gate runs each as a two-block cooperative grid and
compares all logits, losses, gradients, parameters, and optimizer moments with the independent
modular launch sequence.

The complete-model boundary is implemented separately from the projection acceptance kernels.
`FullTrainingMegakernelCore` lowers an arbitrary checked hybrid layer stack to fixed-width UInt32
operation records plus typed activation, BF16 matrix, adapter, and index pointer tables. One
cooperative grid interprets embedding gather, all 64 mixer/MLP layers, the LM head, the objective,
the exact reverse schedule, all 497 LoRA reductions, AdamW schedule advance, and all masked
parameter/moment publications. Entry kernels stay objective-specific:

- `FullPretrainMegakernel.fullPretrainingStepF32Kernel` embeds stable mean CE directly between the
  shared forward and reverse phases.
- `FullGRPOMegakernel.fullGRPOStepF32Kernel` embeds current-policy scoring, grouped reward
  normalization, clipping, sampled direct KL, and the masked policy VJP. Old-policy and frozen
  reference scoring remain batch preparation; each current-policy optimizer update is one launch.

Both entries query their compiled occupancy and cap the cooperative grid to the device's resident
block capacity. The production worker lowers and uploads the immutable tape once, then reuses it
for every SFT or GRPO update. DPO retains its reusable full-model CUDA graph until a dedicated DPO
entry program is added.

The real-checkpoint bootstrap exposes the same routing boundary through
`CheckpointLoRA.Descriptor`: fixed `batch x sequence` rows, one adapter ID per sequence, and one
optimizer update mask entry per adapter. The legacy `CheckpointLoRA.Trainer` retains the focused
LM-head gate. The production `CheckpointLoRA.ProjectionTrainer` maps zero-copy views over the
already loaded aligned BF16 buffers into all 497 DeltaNet, attention, MLP, and LM-head projections,
then reuses the complete model forward/reverse graph to update every adapter. Its bounded device
reductions report finite/nonzero/maxAbs/L2/RMS statistics for every A/B tensor without copying full
gradients to the host. Versioned `LCQPROJ2` snapshots retain canonical projection order, adapter
parameters, routing, mask, AdamW moments, and the bias-correction schedule for exact trainer-state
resume. They also carry model/geometry/objective compatibility metadata and a streaming checksum;
resume validates the complete file before restoring device state. The
`Megakernel.TrainableCheckpoint.forwardBatch` bridge executes sequence pairs in the
two-row retained-training kernel
(`qwen36TrainingBatch2Model`), which shares every projection weight read across the row pair
while keeping recurrent, convolution, and KV state isolated per row; only an odd tail row uses
the single-sequence kernel. A real-checkpoint gate checks the paired kernel against the
sequential single-sequence forward row by row. Wider lockstep batching beyond two rows remains
a throughput follow-up.

### Real Qwen3.8-27B projection-wide acceptance

The public-release gate on 2026-08-31 loaded all 851 text tensors, lowered exactly 497 rank-16
LoRA projections, and completed batch-1, sequence-length-1 SFT at learning rate `1e-5` in the
full pretraining megakernel. The worker returned finite loss `10.583307` and accepted every
projection's gradient diagnostics as finite before atomically publishing a 1.4 GB `LCQPROJ2`
checkpoint. A fresh process completed the full checksum/compatibility preflight, restored the
checkpoint, and ran another update with pre-update loss `10.554998`, proving the v2 save/resume
path preserves an effective real-model update.

The complete-kernel gate on 2026-08-25 loaded all 851 text tensors (53,792,000,000 packed BF16
bytes), lowered 64 layers and exactly 497 LoRA projections, and ran rank-16, batch-1,
sequence-length-1 SFT at learning rate `1e-5`. One full forward/CE/reverse/AdamW launch took
31.813 seconds, returned finite loss `6.623944`, and produced finite gradients for every adapter
tensor. Saving the resulting legacy `LCQPROJ1` state and replaying the same token pair in a fresh
process gave pre-update loss `6.601563`, demonstrating an effective real-checkpoint update. A
separate historical grouped-rollout gate (group 2, sequence length 15) completed the former
current-adapter old/reference preparation and one full GRPO optimizer launch with finite objective
`-0.000873`; it predates mandatory collector-recorded behavior log probabilities and is not an
acceptance result for the `LCQGRP2` offline objective.

The earlier reusable-graph acceptance remains useful as an independent production baseline:

The 2026-08-25 plain-Lean-CUDA gate used the published 18-shard checkpoint, rank 16, batch 1,
sequence length 128, and learning rate `1e-4`:

- Two resident-graph steps completed with losses `0.338788 -> 0.305122`; each full step took about
  185 seconds on the GB10 host.
- Step 1 produced finite, nonzero B gradients across all 497 projections while A gradients were
  exactly zero, as required by zero-initialized B. Step 2 produced finite, nonzero A and B
  gradients across the first and last layers and LM head, proving an effective optimizer update.
- The process held 85,040 MiB of device memory and stayed at about 96% GPU utilization during the
  reverse graph; maximum host RSS was 905,780 KiB.
- A 1.4 GB legacy `LCQPROJ1` snapshot recorded version 1, all 497 projections, and optimizer step 1. A
  fresh process restored it and completed another full step with finite loss `0.311806`.

## Live token streaming

The persistent inference kernel publishes a `{position, token}` event after each final projection
through a mapped system-scope `HostIO.Queue`; it does not wait for kernel shutdown or copy the
position-major logits tensor first. `Megakernel.InferenceStream` separates `enqueue`, `nextToken`,
`finish`, and full-output `collect`. A loaded `Checkpoint` can start repeated sessions without
reloading shards, and `Checkpoint.generateStreaming` feeds each sampled token back into the same
live persistent launch while invoking the host callback immediately.

The real-checkpoint LeanTest loads the published 18-shard Qwen3.8-27B model, checks ordered streamed
events against the sampled buffer, and performs a two-token autoregressive stream in one launch.
The tiny LoRA gate runs three complete device-tail steps and compares every adapter parameter,
moment, schedule value, and final loss byte-for-byte with three sequential steps.

## Tiny gate configuration

All correctness gates run at this shape (tile-friendly, exercises every code path):

| field | tiny | 27B |
| --- | --- | --- |
| hidden_size | 256 | 5120 |
| num_hidden_layers | 4 (lin,lin,lin,full) | 64 |
| linear_num_key_heads x key_head_dim | 2 x 64 | 16 x 128 |
| linear_num_value_heads x value_head_dim | 4 x 64 | 48 x 128 |
| linear_conv_kernel_dim | 4 | 4 |
| num_attention_heads / kv heads / head_dim | 4 / 2 / 64 | 24 / 4 / 256 |
| partial_rotary_factor | 0.25 (rotary 16) | 0.25 (rotary 64) |
| intermediate_size | 512 | 17408 |
| vocab_size | 512 | 248320 |
| rope_theta | 1e7 | 1e7 |

Weights for gates are seeded random (seed 0xC0FFEE) with HF-style init
(`A_log = log(U(0,16))`, `dt_bias = 1`, `(1+w)` norms at 0, gated norm at 1).

## Module layout and ownership

Library (built by Lake from `lib/LeanCudaQwen`):

- `lib/LeanCudaQwen/Qwen36.lean` — umbrella (config workstream owns).
- `lib/LeanCudaQwen/Qwen36/Config.lean` — `Config` structure, `Config.check`, derived dims.
- `lib/LeanCudaQwen/Qwen36/ConfigProvider.lean` — `hfconfig_type_provider "config.json" as NS`
  command (tyr `Tyr/SafeTensors/TypeProvider.lean` pattern: command elaborator that
  introspects the JSON at elaboration time via `Lean.Data.Json` and emits a typed
  `Config` value + derived constants + `layerTypes` table).
- `lib/LeanCudaQwen/Qwen36/Primitives.lean` — device bodies: RMSNorm `(1+w)` fwd/bwd, gated
  RMSNorm fwd/bwd, RoPE table + partial apply fwd/bwd, causal depthwise conv1d fwd/bwd,
  l2norm fwd/bwd, SiLU/SwiGLU fwd/bwd, embedding gather, fused softmax cross-entropy
  (reuse `LeanCudaQwen.MoE.Training` node where possible), BF16<->FP32 casts.
- `lib/LeanCudaQwen/Qwen36/Linear.lean` — generic dense-projection numerical bodies and VJPs;
  production schedules route tile-compatible BF16 calls through the native GEMM runners.
- `lib/LeanCudaQwen/Qwen36/DeltaNet.lean` — projections, conv, recurrent decode step,
  chunked prefill, gated output norm, VJP bundle.
- `lib/LeanCudaQwen/Qwen36/Attention.lean` — q/k norm + partial RoPE + GQA flash bodies
  (adapt `LeanCudaQwen.GB10.Attention256`), swish output gate, KV cache, VJP bundle.
- `lib/LeanCudaQwen/Qwen36/MLP.lean` — sequential RMSNorm/SwiGLU projection forward and VJP
  baseline.
- `lib/LeanCudaQwen/Qwen36/LoRA.lean` — checked mixed-adapter descriptors, fused projection
  contributions/VJPs, adapter-local gradient reduction, and optimizer state.
- `lib/LeanCudaQwen/Qwen36/Model.lean` — arbitrary hybrid layer stack, stable mean
  cross-entropy, complete residual reverse graph, and deterministic embedding VJP. This is the
  independent sequential numerical baseline for persistent-schedule parity.
- `lib/LeanCudaQwen/Qwen36/SafeTensors.lean` — Lean-side safetensors header parse + bulk
  device loads + HF-name -> weight-family registry.
- `lib/LeanCudaQwen/Qwen36/Megakernel.lean` — persistent multi-token prefill/decode composition.
- `lib/LeanCudaQwen/Qwen36/CheckpointLoRA.lean` — real-checkpoint LM-head gate plus the
  projection-wide 497-adapter trainer, diagnostics, streamed snapshots, and optimizer lifecycle.
- `lib/LeanCudaQwen/Qwen36/Training.lean` — fully resident batched forward/backward/update graph,
  including full-model and mixed-adapter LoRA profiles.
- `lib/LeanCudaQwen/Qwen36/TrainingMegakernelCore.lean` — shared POD descriptors and device phases
  for one-launch projection training.
- `lib/LeanCudaQwen/Qwen36/PretrainMegakernel.lean` — dedicated cooperative pretraining entry.
- `lib/LeanCudaQwen/Qwen36/GRPOMegakernel.lean` — dedicated cooperative clipped-GRPO entry.
- `lib/LeanCudaQwen/Qwen36/FullTrainingMegakernelCore.lean` — typed model-wide operation tape,
  forward/reverse interpreter, projection registry, and shared optimizer publication.
- `lib/LeanCudaQwen/Qwen36/FullPretrainMegakernel.lean` — complete-model CE training entry.
- `lib/LeanCudaQwen/Qwen36/FullGRPOMegakernel.lean` — complete-model clipped-GRPO policy entry.

Suites under `tests/workload_regressions` (each owns its directory and `run_test.sh` gate):

- `cuda_qwen36_config` — provider gates: real HF `config.json` (vendored) + tiny fixture;
  emitted constants checked; malformed configs rejected.
- `cuda_qwen36_reference` — NumPy oracle (`oracle.py`), checked-in generated fixtures,
  layout README. Pure NumPy fp32; no torch dependency.
- `cuda_qwen36_primitives` — per-primitive device gates vs fixtures.
- `cuda_qwen36_weights` — safetensors loader vs synthetic fixture.
- `cuda_qwen36_deltanet` / `cuda_qwen36_attention` / `cuda_qwen36_mlp` — stage gates.
- `cuda_qwen36_model` — full sequential embedding/layer/logit/loss/reverse parity vs oracle,
  including every trainable weight family across the four-layer tiny stack.
- `cuda_qwen36_megakernel` — persistent decode megakernel; byte-exact vs sequential.
- `cuda_qwen36_lora` — mixed-adapter forward/backward/update isolation and permutation gates.
- `cuda_qwen36_training` — fully resident device-graph training step; loss/update and sequential
  parity gates for both full-model and batched-LoRA profiles.

Run one or more suites through `tests/workload_regressions/run.sh`; the wrapper builds the workload library and dispatches each selected `run_test.sh`.

## Fixture format (oracle <-> Lean gates)

One file per tensor: `<name>.<dim1>x<dim2>x....f32.bin`, raw little-endian fp32,
row-major, plus `manifest.json` (`{name: {shape, file, sha256?}}`). Tiny shapes only;
fixtures are checked in so gates never need Python. Oracle fixtures live in
`tests/workload_regressions/cuda_qwen36_reference/fixtures/`.

## Tolerance policy

- fp32 paths: max abs err <= 1e-5 (primitives), <= 3e-5 (stage outputs, fp32 accumulation).
- bf16 storage paths: relative L2 <= 2e-2 per stage output vs fp32 oracle.
- Megakernel vs sequential launch of identical kernels: byte-exact.
- Training: loss curve parity vs sequential baseline; mean CE must fall below the
  uniform-softmax start within one epoch slice.
- Batched LoRA: forward/gradient/update parity vs NumPy within the applicable fp32/bf16 policy;
  frozen base weights and masked adapters remain byte-exact.

## Build and test recipe

Set `LEAN_CUDA_ROOT` to a checkout with a staged CUDA compiler at `build/release/stage1`. The suite wrapper builds `LeanCudaQwen`, exports its Lake output through `LEAN_PATH`, and compiles workload modules from `lib/LeanCudaQwen` while taking runtime sources and headers from `$LEAN_CUDA_ROOT/src`. CUDA execution gates require CUDA 13 and the selected device architecture; `LEAN_CUDA_COMPUTE_SANITIZER=1` enables memcheck where supported.

## Conventions

- File header: `Copyright (c) 2026 Ranvier Systems. All rights reserved.` + Apache 2.0
  line, per existing `lib/LeanCudaQwen` files.
- Comments: concise, non-obvious facts only (see root AGENTS.md).
