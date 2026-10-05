# Ascend incremental findings (2026-10-05)

Scope: incremental evidence found between 2026-09-05 and 2026-10-05. This note records provenance, license, implementation entry points, and execution boundaries. It does not perform security or integrity auditing.

## 1. Ascend/DeepEP — official Ascend MegaMoE integration

- Canonical upstream: <https://gitcode.com/Ascend/DeepEP>
- Archived at: `archive/distributed-moe/Ascend-DeepEP`
- Acquisition used:

  ```bash
  git clone --depth 1 --single-branch https://gitcode.com/Ascend/DeepEP.git archive/distributed-moe/Ascend-DeepEP
  ```

- Archive result: shallow Git checkout retained (`.git` present), **379 tracked files**. The archived default-branch tip has author/committer time **2026-09-30 18:11:01 +08:00** and subject `docs: 补充 Dispatch/Combine 使用示例`.
- Increment evidence: the archived default branch contains the complete `experiments/megamoe/` implementation and exposes the experimental Python entry `deep_ep_experimental.megamoe.fp8_fp4_mega_moe()`.
- Hardware/software scope: Linux, **Ascend950**, **CANN 9.2.0**, matching Torch/torch_npu. The root README lists MegaMoE Prefill for EP32/64/128 and Decode for EP64/128.
- Execution boundary: a **single distributed-MoE forward operator**, not a transformer/model-wide megakernel. One mixed-core launch fuses token dispatch/routing, GMM1, activation and quantization, GMM2, combine/unpermute, and shared-expert work. It does not include attention, transformer-layer scheduling, embeddings, LM head, sampling, or a device-resident token loop.
- Device mechanism: `mega_moe.cpp` defines one `__schedmode__(1) __global__ __mix__(1, 2) mega_moe_kernel` and dispatches one launch per call. `mega_moe_arch35.hpp` coordinates AIC and AIV roles, routed/shared expert waves, global-buffer readiness flags, token dispatch, and unpermute/combine inside that launch.
- Key implementation locations:
  - `experiments/megamoe/README.md`
  - `experiments/megamoe/csrc/kernels/mega_moe/mega_moe.cpp`
  - `experiments/megamoe/csrc/kernels/mega_moe/mega_moe_arch35.hpp`
  - `experiments/megamoe/csrc/kernels/mega_moe/stages/`
  - `experiments/megamoe/csrc/kernels/mega_moe/catlass_gmm/`
  - `experiments/megamoe/python/deep_ep_experimental/megamoe/`
  - `experiments/megamoe/scripts/build.sh`
- Build entry: after loading the matching CANN environment, `bash experiments/megamoe/scripts/build.sh`.
- License: root `LICENSE` is **BSD-2-Clause**, copyright 2026 Lu Lu. Root README lines 439–443 clarify that content owned by Lu Lu is BSD-2-Clause and code originating elsewhere retains its respective copyright and license. Catalog as open source, while preserving component-specific notices.
- Maturity caveat: upstream labels MegaMoE experimental; the README states that device compilation, accuracy, and performance qualification remain to be completed. Source presence is established, but no unverified performance claim should be inferred.

## 2. liruixin_dvc/ascend_mega_kernel — Ascend950 layer-scale megakernel prototype

- Canonical upstream: <https://gitcode.com/liruixin_dvc/ascend_mega_kernel>
- Archived at: `archive/operator-scale/ascend_mega_kernel`
- Acquisition used:

  ```bash
  git clone --depth 1 --single-branch https://gitcode.com/liruixin_dvc/ascend_mega_kernel.git archive/operator-scale/ascend_mega_kernel
  ```

- Archive result: shallow Git checkout retained (`.git` present), **5,366 tracked files**. The archived default-branch tip has author/committer time **2026-10-05 02:32:47 UTC** and subject `Merge branch 'feat/m203-docs-20-router-wo-a2-scope-and-owne'`.
- Temporal evidence: root README records the project as an Ascend950 mega-kernel effort, while the current checkout contains dated increments through **2026-10-05**. Earlier milestones in `m15_layer_loop/README.md` identify the 48-layer four-phase chain integration on 2026-09-26 and attention-cache layout work on 2026-09-27.
- Hardware/software/model scope: one **Ascend950PR** card, **CANN 9.1.0**, target **Qwen3.8-Flash-Next-MXFP4**.
- Proven execution boundaries:
  - `m13_moe_layer/`: one `__mix__(1,2)` launch covers MoE stages S1–S10.
  - `m14_gdn_layer/`: one `__mix__(1,2)` launch covers GDN stages S1–S7.
  - `m15_layer_loop/`: the current checkout goes beyond the older README summary. Its **host loop launches once per layer** across a 48-layer decode chain. Each layer launch combines `hc(attn) -> GDN or attention arm -> hc(mlp) -> MoE`; long-lived H/BO/IJ state is handed off through GM, and a separate final-mixer launch follows the 48 layers.
  - The 48-layer chain is **not one device-resident launch**: `m15_chain_host.h::H_ChRunOnce` contains the host `for (L...)` loop and calls `H_LaunchChainLayer` once for each layer, synchronizing after each launch.
  - It is also not yet a semantically complete model path. In the archived checkout, decode full-attention layers still use pass-through semantics; QSA is not integrated. The layer-1 PLE body remains a placeholder, prefill is only partially wired, and the attention prefill dense causal core is explicitly absent. Embedding, LM head, sampling, and a persistent token loop are outside this project boundary.
- Scheduling/mechanism: full-core mixed AIC/AIV launch (`__mix__(1,2)`), AIC Cube plus two AIV Vector roles, explicit BufferID/CrossCore synchronization, phase barriers within each layer, and GM-backed cross-layer state. The project is therefore a **genuine layer-scale megakernel prototype and host-orchestrated layer chain**, not a whole-model persistent megakernel.
- Key implementation and design locations:
  - `m13_moe_layer/m13_moe_layer.asc`
  - `m14_gdn_layer/m14_gdn_layer.asc`
  - `m15_layer_loop/m15_layer_loop.asc`
  - `m15_layer_loop/m15_chain_host.h`
  - `m15_layer_loop/m15_layer_resources.h`
  - `m15_layer_loop/README.md`
  - `docs/05-megakernel-design.md`
  - `docs/12-layer-integration.md`
  - `docs/14-hyperconnection-ple-indexer-spec.md`
  - `docs/15-prefill-design.md`
  - `docs/16-vllm-ascend-qwen4exp-plan.md`
- License: **no root `LICENSE`, `COPYING`, or `NOTICE` file** is present in the archived default branch. Treat as source-visible with license not declared; do not redistribute as if it had an OSI license unless upstream publishes one.

## Classification summary

| Project | What is actually fused | Host-free scope | License classification | Archive class |
|---|---|---|---|---|
| Ascend/DeepEP MegaMoE | Distributed MoE forward: dispatch/routing, two GMMs, activation/quantization, combine/unpermute, shared expert | One MoE operator call | BSD-2-Clause at root; preserve source-specific terms | `distributed-moe` |
| ascend_mega_kernel | Full GDN or MoE blocks; current decode chain combines four phases per layer | One layer; 48-layer traversal remains a host loop and attention/PLE are incomplete | No declared root license | `operator-scale` / source-visible |

## Evidence URLs

- <https://gitcode.com/Ascend/DeepEP>
- <https://gitcode.com/Ascend/DeepEP/blob/main/LICENSE>
- <https://gitcode.com/Ascend/DeepEP/blob/main/README.md>
- <https://gitcode.com/Ascend/DeepEP/blob/main/experiments/megamoe/README.md>
- <https://gitcode.com/Ascend/DeepEP/blob/main/experiments/megamoe/csrc/kernels/mega_moe/mega_moe.cpp>
- <https://gitcode.com/liruixin_dvc/ascend_mega_kernel>
- <https://gitcode.com/liruixin_dvc/ascend_mega_kernel/blob/main/README.md>
- <https://gitcode.com/liruixin_dvc/ascend_mega_kernel/blob/main/m15_layer_loop/README.md>
- <https://gitcode.com/liruixin_dvc/ascend_mega_kernel/blob/main/m15_layer_loop/m15_chain_host.h>
- <https://gitcode.com/liruixin_dvc/ascend_mega_kernel/blob/main/docs/05-megakernel-design.md>
