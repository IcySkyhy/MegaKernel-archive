# Qwen3.8 train

This local CLI trains LoRA adapters on every documented projection of the published
Qwen3.8-27B checkpoint using plain Lean CUDA. The 53.792 GB of BF16 tensors are packed into aligned
GPU buffers once per process; the trainer takes zero-copy views over those buffers and never
updates the frozen base. All 497 transformer/LM-head adapters and their AdamW state are updated,
checkpointed, and resumed. The worker supports ordinary next-token SFT, completion-masked DPO,
and clipped outcome-GRPO against a frozen base-model reference. SFT and each current-policy GRPO
update use separate complete-model cooperative megakernels; DPO uses the reusable full-model CUDA
graph. Progress is emitted as NDJSON.

From the repository root:

```bash
export QWEN_MODEL_DIR=/path/to/Qwen3.8-27B

# 1. Tokenize a plain-text corpus into packed UInt32 streams (uses uv for the tokenizer).
uv run --no-project --with 'tokenizers==0.23.1' \
  python examples/qwen38_train/prepare_dataset.py \
  --model-dir "$QWEN_MODEL_DIR" --input corpus.txt \
  --train-out train.bin --val-out val.bin

# 2. Train. The worker loads the 52 GB checkpoint once per invocation.
./examples/qwen38_train/run.sh \
  --dataset train.bin --val-dataset val.bin \
  --steps 200 --batch-size 1 --sequence-length 128 \
  --learning-rate 0.001 --val-every 25 \
  --checkpoint-out adapter.lcqproj --save-every 50

# 3. Resume exactly (adapter parameters, AdamW moments, and schedule).
./examples/qwen38_train/run.sh \
  --dataset train.bin --steps 200 --resume adapter.lcqproj \
  --checkpoint-out adapter.lcqproj
```

For DPO, start from JSONL preference pairs:

```json
{"system":"Be concise.","prompt":"Why is the sky blue?","chosen":"Rayleigh scattering preferentially scatters blue light.","rejected":"Because the ocean reflects onto it."}
```

`messages` may replace `system`/`prompt`; `chosen` and `rejected` are always response strings.
Prepare a fixed-shape `LCQDPO1` dataset and train with pair batch size one:

```bash
uv run --no-project --with 'tokenizers==0.23.1' \
  python examples/qwen38_train/prepare_preference_dataset.py \
  --model-dir "$QWEN_MODEL_DIR" --input preferences.jsonl \
  --sequence-length 64 --train-out preferences.train.bin \
  --val-out preferences.val.bin --val-records 32

./examples/qwen38_train/run.sh \
  --objective dpo --dataset preferences.train.bin \
  --val-dataset preferences.val.bin --batch-size 1 --sequence-length 64 \
  --dpo-beta 0.1 --steps 200 --learning-rate 0.0001 \
  --val-every 25 --checkpoint-out adapter.dpo.lcqproj
```

The implemented loss is the pair mean
`-log sigmoid(beta * ((log pi(chosen) - log pi(rejected)) -
(log pi_ref(chosen) - log pi_ref(rejected))))`. `pi_ref` is the frozen checkpoint with every
adapter route disabled. Prompt and padding tokens are excluded from all four sequence scores and
from the policy VJP.

Set `QWEN_TRAIN_REBUILD=1` to force a worker rebuild. `--model-dir` overrides
`QWEN_MODEL_DIR` when both are given.
For outcome-GRPO, each JSONL record is one prompt with exactly one sampled rollout group. The
collector must record the exact behavior policy's SHA-256 identity and one log probability per
trainable generated token. This schematic assumes each shown response plus `<|im_end|>` tokenizes
to two trainable tokens:

```jsonc
{"system":"You may call tools.","prompt":"What is 17 * 23?","behavior_policy_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","rollouts":[{"response":"391","behavior_logprobs":[-0.08,-0.01],"reward":1.0},{"response":"381","behavior_logprobs":[-0.11,-0.01],"reward":0.0}]}
```

Use the log probabilities emitted during sampling; the preparer deliberately rejects missing,
mis-sized, out-of-range, or nonfinite behavior evidence.

The `messages` form accepted by the DPO preparer may replace `system`/`prompt`. A rollout can
instead contain ordered `segments`; this preserves tool observations as model context while
training only assistant/tool-call tokens:

```jsonc
{"prompt":"Find the weather, then answer.","behavior_policy_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","rollouts":[{"segments":[{"text":"<tool_call>{\"city\":\"Berlin\"}</tool_call>","train":true},{"text":"<tool_result>18 C, rain</tool_result>","train":false},{"text":"It is 18 C and raining.","train":true}],"behavior_logprobs":[/* one finite value per token in the two train:true spans */],"reward":1.0},{"response":"I cannot check.","behavior_logprobs":[/* one finite value per response token */],"reward":0.0}]}
```

Prepare complete adjacent groups and train with one prompt group per step:

```bash
uv run --no-project --with 'tokenizers==0.23.1' \
  python examples/qwen38_train/prepare_grpo_dataset.py \
  --model-dir "$QWEN_MODEL_DIR" --input rollouts.jsonl \
  --group-size 2 --sequence-length 64 --train-out rollouts.train.bin

./examples/qwen38_train/run.sh \
  --objective grpo --dataset rollouts.train.bin \
  --batch-size 1 --grpo-group-size 2 --sequence-length 64 \
  --grpo-clip-epsilon 0.2 --grpo-kl-beta 0.04 \
  --grpo-updates 1 --steps 200 --learning-rate 0.0001 \
  --checkpoint-out adapter.grpo.lcqproj
```

For group member `i` and masked target token `t`, the implemented loss is:

```text
A_i       = (reward_i - group_mean) / sqrt(group_variance + advantage_epsilon)
ratio_i,t = exp(log pi_i,t - log pi_old_i,t)
KL_i,t    = exp(log pi_ref_i,t - log pi_i,t)
            - (log pi_ref_i,t - log pi_i,t) - 1
loss_i,t  = -min(ratio_i,t * A_i,
                 clamp(ratio_i,t, 1 - clip_epsilon, 1 + clip_epsilon) * A_i)
            + kl_beta * KL_i,t
```

Tokens are averaged within each member and members are averaged across the batch. Rewards are
population-normalized independently in every adjacent group. `log pi_old` comes from the recorded
behavior-policy scores in the `LCQGRP2` file and remains fixed for all `--grpo-updates` optimizer
updates. The reference is the frozen checkpoint with every adapter route disabled. Prompt,
padding, and `train:false` tool-observation spans have exactly zero objective and logit gradient.

The preparer consumes already generated, scored trajectories; it does not generate, score, or
reward them. The required per-token scores make offline self-play, verifier, tool, and distillation
collectors mathematically valid even when the trainer's starting adapter differs from the behavior
policy. `behavior_policy_sha256` is retained per group for provenance and should identify the exact
base-plus-adapter artifact used by the collector.


## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--model-dir` | `$QWEN_MODEL_DIR` | published checkpoint directory |
| `--dataset` | required | SFT UInt32 stream, `LCQDPO1` pairs, or `LCQGRP2` rollout groups |
| `--objective` | `sft` | `sft`, `dpo`, or `grpo` |
| `--val-dataset` | none | validation data in the selected objective's format |
| `--steps` | 100 | optimizer steps |
| `--batch-size` | 1 | SFT sequences, DPO pairs, or complete GRPO groups per step |
| `--sequence-length` | 128 | tokens per sequence (max 256) |
| `--rank` | 16 | LoRA rank (positive multiple of 16) |
| `--alpha` | 16 | LoRA alpha; scale is `alpha / rank` |
| `--dpo-beta` | 0.1 | positive finite DPO preference temperature |
| `--grpo-group-size` | 2 | adjacent sampled completions per prompt group |
| `--grpo-updates` | 1 | optimizer updates using one fixed old-policy rollout batch |
| `--grpo-clip-epsilon` | 0.2 | PPO ratio clipping radius in `(0, 1)` |
| `--grpo-kl-beta` | 0.04 | nonnegative sampled-KL coefficient |
| `--grpo-advantage-epsilon` | 1e-6 | positive reward-normalization stabilizer |
| `--learning-rate` | 0.001 | AdamW learning rate |
| `--seed` | 13878 | adapter A initialization seed |
| `--log-every` | 1 | steps between `step` records |
| `--val-every` | 0 | steps between `val` records (0 disables) |
| `--val-batches` | 4 | validation batches averaged per record |
| `--checkpoint-out` | none | final versioned adapter checkpoint path |
| `--resume` | none | checkpoint to resume before step one |
| `--save-every` | 0 | steps between intermediate saves |

## NDJSON records

- `start` — dataset size, projection count, and resolved training geometry.
- `step` — `step`, `epoch`, pre-update objective `loss`, wall-clock `ms`, and per
  projection/adapter `a`/`b` gradient summaries (`finite`, `nonzero`, `maxAbs`, `l2`, `rms`).
- DPO `step`/`val` records also report `rewardAccuracy`, `meanChosenReward`,
  `meanRejectedReward`, and `meanRewardMargin`.
- GRPO `step`/`val` records report `meanReward`, `rewardStd`, `meanKL`,
  `clipFraction`, and `meanAbsoluteAdvantage`; steps also report `optimizerUpdates`.
- `val` — mean validation objective with the current adapter, no state updated.
- `checkpoint` — a versioned adapter/AdamW snapshot was written.
- `resume` — a snapshot was loaded before training.
- `done` — training finished; reports the final training loss.

Nonfinite losses, metrics, or gradients abort training before another checkpoint can be published;
lower the learning rate and resume from the last known-good checkpoint.

## Notes

- SFT lowers the 64-layer, 497-projection program once and reuses one cooperative pretraining
  launch per forward/CE/reverse/AdamW step. GRPO uses a separately compiled objective entry for
  each current-policy update. DPO retains the reusable complete forward/reverse/update CUDA graph.
- SFT activation bounds require `batch-size * sequence-length <= 128` and
  `batch-size * (sequence-length + 1) <= 132`.
- A DPO pair occupies two sequences, so the corresponding bounds use `2 * batch-size`; the
  practical default is one pair at sequence length 64.
- A DPO step runs a frozen-reference forward before the policy forward/backward/update graph.
  Reference scores are recomputed per batch and never receive gradients.
- A GRPO group occupies `grpo-group-size` sequences, so its bounds use
  `batch-size * grpo-group-size`. The default group of two supports sequence length 64.
- A GRPO batch uploads collector-recorded behavior scores, computes frozen-reference scores once,
  then launches the complete policy forward/objective/reverse/update megakernel `grpo-updates`
  times. Only the current policy receives gradients.
- Gradient diagnostics are reduced on device, so logging copies four scalars per A/B tensor,
  not the full gradient allocation.
- Checkpoints are streamed through a temporary file and atomically published as
  self-validating `LCQPROJ2` binaries. They include a checksum plus the base model fingerprint,
  geometry, LoRA/AdamW parameters, and objective configuration. Resume performs a complete
  compatibility/integrity preflight before restoring canonical projection order, parameters,
  moments, routes, masks, or schedule state.
- Snapshots restore trainer state exactly; the CLI's dataset cursor is process-local, so a job
  scheduler should persist its own data-shard/cursor metadata when exact sample order matters.
- The chat inference endpoint (`examples/qwen38_chat`) and this trainer both need the full
  checkpoint resident; run them one at a time.
