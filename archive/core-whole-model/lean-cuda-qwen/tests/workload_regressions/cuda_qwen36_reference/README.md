<!--
Copyright (c) 2026 Ranvier Systems. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-->

# cuda_qwen36_reference — NumPy oracle + fixtures

Numerical correctness anchor for the Qwen3.6-27B megakernel workstreams
(`docs/QWEN36_MEGAKERNEL.md`). Pure NumPy fp32; no torch or GPU dependency at
runtime.

Files:

- `modeling_qwen3_5_excerpt.py` — vendored HF reference excerpt (torch),
  provenance in its header. Development-time cross-check reference only.
- `oracle.py` — pure NumPy reimplementation: seeded init, full prefill forward
  with per-stage dumps, decode-step case, hand-derived analytic backward
  (finite-difference-free), fixture dump/compare CLI, self-test.
- `fixtures/` — checked-in generated fixtures (188 tensors + `manifest.json`).
- `torch_reference_check.py` — optional dev gate: replays every fixture against
  the vendored torch reference (needs a python with torch; skipped otherwise).
- `run_test.sh` — the ctest gate (regeneration determinism + self-test).

## Tiny gate configuration

| field | value |
| --- | --- |
| hidden_size H | 256 |
| layer_types | [linear_attention x3, full_attention] (layers 0..3) |
| linear key heads x head_k | 2 x 64 (key_dim 128) |
| linear value heads x head_v | 4 x 64 (value_dim 256) |
| conv_dim / conv kernel | 2*128 + 256 = 512 / 4 (depthwise, no bias, SiLU) |
| delta chunk size | 64 (seq 8 exercises padding) |
| attention heads / kv heads / head_dim | 4 / 2 / 64 (GQA ratio 2) |
| partial_rotary_factor | 0.25 (rotary dim 16, theta 1e7) |
| intermediate_size | 512 |
| vocab_size V | 512 (untied embeddings) |
| rms_norm_eps / l2norm eps | 1e-6 / 1e-6 |
| seq_len T | 8 |

## Weight init (seed 0xC0FFEE)

Single `numpy.random.default_rng(0xC0FFEE)`; draws in this exact order
(`standard_normal(shape, dtype=float32) * 0.02` unless noted):

1. `model.embed_tokens.weight` (512, 256)
2. per layer i = 0..3, in order (constant tensors draw nothing):
   - `input_layernorm.weight` = zeros(256) — the `(1+w)` norm, so norm == identity scale
   - linear_attention layers: `in_proj_qkv.weight` (512,256), `in_proj_z.weight` (256,256),
     `in_proj_b.weight` (4,256), `in_proj_a.weight` (4,256), `conv1d.weight` (512,4),
     `A_log` = log(uniform(0,16, size 4)), `dt_bias` = ones(4),
     `norm.weight` = ones(64) (gated norm, unit init), `out_proj.weight` (256,256)
   - full_attention layer: `q_proj.weight` (512,256), `k_proj.weight` (128,256),
     `v_proj.weight` (128,256), `o_proj.weight` (256,256),
     `q_norm.weight`/`k_norm.weight` = zeros(64)
   - `post_attention_layernorm.weight` = zeros(256)
   - `mlp.gate_proj.weight` (512,256), `mlp.up_proj.weight` (512,256), `mlp.down_proj.weight` (256,512)
3. `model.norm.weight` = zeros(256); `lm_head.weight` (512,256)
4. inputs, in order: `tokens` = integers(0,512, size 8); `targets` = integers(0,512, size 8);
   `probe_hidden` = standard_normal((8,256), float32); `decode_token` = integers(0,512, size 1)

All Linear weights are stored HF-style `(out_features, in_features)`, applied as
`y = x @ W.T`. Token/target/decode ids are stored as fp32 values (exact small ints).

## Equations (compact; matches modeling_qwen3_5_excerpt.py exactly)

Decoder layer: `h = x + mixer(norm1(x))`, `y = h + mlp(norm2(h))`; final
`norm` then `logits = norm_out @ W_lm.T`. All norms fp32.

- RMSNorm `(1+w)` variant (input/post_attention/final/q_norm/k_norm):
  `y = x * rsqrt(mean(x^2) + eps) * (1 + w)`
- Gated DeltaNet layer (x (T,256)):
  1. `qkv = W_qkv x` (T,512), `z = W_z x` (T,256), `b = W_b x` (T,4), `a = W_a x` (T,4)
  2. causal depthwise conv1d k=4 (left pad 3), then SiLU; split
     `q (T,128) | k (T,128) | v (T,256)`; head views q,k (T,2,64), v (T,4,64)
  3. `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)` (fp32)
  4. repeat_interleave q,k by 2 -> (T,4,64) (slot j of head m -> slot 2m+j);
     `q = l2norm(q)`, `k = l2norm(k)` per head (`x * rsqrt(sum(x^2)+1e-6)`)
  5. chunked delta rule, chunk 64, scale `64^-0.5` applied to q after l2norm;
     state `S (4, 64k, 64v)` fp32, zero init:
     - pad T to 64; `g = cumsum(g)` in-chunk; `decay[i,j] = exp(g_i - g_j)` for j<=i else 0
     - `A[i,j] = -(beta_i k_i . k_j) decay[i,j]` (strictly lower);
       `M_inv = (I - A)^-1` by forward substitution
     - `value = M_inv (v beta)`; `k_cum = M_inv (k beta e^g)`
     - per chunk: `att = (q k^T) decay`; `v_new = value - k_cum S`;
       `out = (q e^g) S + att v_new`;
       `S <- S e^{g_last} + (k e^{g_last - g})^T v_new`
  6. per (token, head): `o = (w_gn * rmsnorm(o)) * silu(z)` (unit-init weight,
     NOT `(1+w)`); flatten (T,256); `mixer = W_out o`
  - recurrent (decode) form per token: `S *= e^g; kv = S^T k;
    S += k (beta (v - kv))^T; o = S^T q` (q pre-scaled); mathematically
    identical to the chunked form (self-test: <= 7e-10).
- Full attention layer (x (T,256)):
  1. `qg = W_q x` (T,512) -> heads (T,4,128); `q = qg[..., :64]`,
     `gate = qg[..., 64:]` (per-head split, gate flattened (T,256));
     `k = W_k x` (T,2,64); `v = W_v x` (T,2,64)
  2. per-head RMSNorm `(1+w)` on q, k (dim 64), then partial RoPE on the first
     16 of 64 dims (GPT-NeoX half rotation): `inv_freq[i] = 1e7^(-2i/16)`,
     i<8; `cos[t,i] = cos(t*inv_freq[i mod 8])` (and sin); for the 16-dim rotary
     block `rot(x) = cat(-x[8:16], x[0:8])`; `x' = x*cos + rot(x)*sin`, dims
     16..63 pass through
  3. GQA: kv head m serves q heads {2m, 2m+1}; `scores = q k^T * 64^-0.5` +
     additive causal mask; softmax fp32; `o = probs v`; flatten (T,256)
  4. `o *= sigmoid(gate)`; `mixer = W_o o`
- MLP: `down(silu(gate(x)) * up(x))` (SwiGLU)
- Loss: `L = mean_t CE(logits_t, targets_t)` (fp32 log-softmax); all `grad.*`
  fixtures are `dL/d(.)`. Backward is hand-derived reverse mode over the same
  equations (UT-transform adjoint via `dA = tril(M_inv^T G M_inv^T, -1)`,
  reverse-cumsum for the in-chunk decay, etc.); validated by central finite
  differences (eps 1e-3, fp64, max err ~1e-7) in `oracle.py selftest` and by
  torch autograd in `torch_reference_check.py`.

## Fixture format

`fixtures/<name>.<dim1>x<dim2>x....f32.bin` — raw little-endian fp32, C
(row-major) order, last dim contiguous; 1-D tensors use a single dim
(`loss.1.f32.bin`). `fixtures/manifest.json` =
`{"meta": {...}, "tensors": {name: {"shape", "file", "sha256"}}}`.
Tensor axes: `(t, h, d)` = (sequence, head, channel); `S` is `(head, k_dim,
v_dim)`; `conv_state` rows are time-ordered raw pre-conv qkv vectors.

### Weights (HF names)

`model.embed_tokens.weight` (512x256), `lm_head.weight` (512x256),
`model.norm.weight` (256), and per layer `model.layers.{i}.`:
`input_layernorm.weight` (256), `post_attention_layernorm.weight` (256),
`mlp.{gate,up}_proj.weight` (512x256), `mlp.down_proj.weight` (256x512);
linear layers (i=0,1,2) `linear_attn.`: `in_proj_qkv.weight` (512x256),
`in_proj_z.weight` (256x256), `in_proj_b.weight`/`in_proj_a.weight` (4x256),
`conv1d.weight` (512x4 — HF stores (512,1,4); fixtures store it squeezed),
`A_log` (4), `dt_bias` (4), `norm.weight` (64), `out_proj.weight` (256x256);
attention layer (i=3) `self_attn.`: `q_proj.weight` (512x256),
`k_proj.weight`/`v_proj.weight` (128x256), `o_proj.weight` (256x256),
`q_norm.weight`/`k_norm.weight` (64).

### Inputs

- `tokens` (8) — prefill token ids (fp32-encoded ints)
- `targets` (8) — CE target ids
- `decode_token` (1) — the decode-case token id
- `probe_hidden` (8x256) — fixed random probe tensor for primitive gates
  (e.g. RMSNorm/conv/attention inputs); not consumed by the model forward

### Prefill forward dumps (T=8)

- `embed_out` (8x256); `rope.cos`/`rope.sin` (8x16)
- per layer i: `layer{i}.norm1_out` (8x256), `layer{i}.mlp_out` (8x256),
  `layer{i}.mixer_out` (8x256), `layer{i}.hidden_out` (8x256)
- per linear layer i in {0,1,2}: `layer{i}.qkv_pre_conv` (8x512);
  `layer{i}.q_post_conv`/`k_post_conv` (8x128), `layer{i}.v_post_conv` (8x256);
  `layer{i}.beta` (8x4), `layer{i}.g` (8x4);
  `layer{i}.q_l2`/`k_l2` (8x4x64 — post l2norm, post repeat_interleave, pre-scale);
  `layer{i}.v_heads` (8x4x64); `layer{i}.delta_out` (8x4x64 — delta-rule output,
  pre gated norm); `layer{i}.S_final` (4x64x64);
  `layer{i}.gated_out` (8x256 — post gated RMSNorm, flattened)
- attention layer 3: `layer3.q_rope` (8x4x64), `layer3.k_rope` (8x2x64),
  `layer3.v_heads` (8x2x64), `layer3.gate` (8x256 — pre-sigmoid),
  `layer3.attn_probs` (4x8x8 — heads-first),
  `layer3.attn_out_pre_gate`/`attn_out_post_gate` (8x256)
- `final_hidden` (8x256 — final norm output, lm_head input); `logits` (8x512);
  `loss` (1)

### Backward probes (dL/d.)

- `grad.logits` (8x512) = (softmax(logits) - onehot(targets)) / 8
- `grad.final_hidden` (8x256)
- `grad.embed_out` (8x256) and deterministic
  `grad.model.embed_tokens.weight` (512x256) scatter
- every per-layer MLP and mixer activation VJP emitted by the hand-written reverse graph
- every trainable DeltaNet, full-attention, MLP, input/post-attention norm, final norm, and
  language-model-head weight gradient. `manifest.json` is the authoritative complete list;
  the four finite-difference families remain the smaller analytical self-test subset.

### Decode case (one token after the 8-token prefill)

Per linear layer i in {0,1,2} (decode chains layers 0->1->2 only; the
full_attention layer is not part of this case):

- `decode.layer{i}.hidden_in` (256) — layer input (layer 0: embedding row of
  `decode_token`; layers 1/2: previous layer's `hidden_out`)
- `decode.layer{i}.norm1_out` (256); `decode.layer{i}.qkv_pre_conv` (512)
- `decode.layer{i}.conv_state_in` (3x512) — last 3 raw pre-conv qkv rows of the
  prefill (t = 5,6,7); `decode.layer{i}.conv_state_out` (3x512) — rows t = 6,7,8
- `decode.layer{i}.q_post_conv` (128), `k_post_conv` (128), `v_post_conv` (256)
- `decode.layer{i}.q_l2`/`k_l2` (4x64), `v_heads` (4x64), `beta` (4), `g` (4)
- `decode.layer{i}.S_in` (4x64x64) — equals `layer{i}.S_final`;
  `decode.layer{i}.delta_out` (4x64); `decode.layer{i}.S_out` (4x64x64)
- `decode.layer{i}.gated_out` (256), `mixer_out` (256), `mlp_out` (256),
  `hidden_out` (256)

## Gates (`run_test.sh`)

1. `oracle.py selftest` — chunked-vs-recurrent delta-rule parity, decode-step
   parity vs a 9-token recurrent reference, central-finite-difference backward
   check (eps 1e-3, tol 1e-3) on `in_proj_qkv` (layer 0), `q_proj` (layer 3),
   `in_proj_a`/`in_proj_b` (layer 0), fp32-vs-fp64 analytic gradient sanity.
2. Regeneration determinism: fixtures regenerated into a temp dir must be
   byte-identical to the checked-in ones (`oracle.py compare`).
3. Optional, explicit skip when unavailable: `torch_reference_check.py`
   (needs a python with torch; forward tol 1e-5, backward tol 1e-4).

Python selection: `python3` if it has numpy; override with `ORACLE_PYTHON`.

Regenerate fixtures after any intentional oracle change with:

```bash
python3 oracle.py dump fixtures   # then re-run run_test.sh
```
