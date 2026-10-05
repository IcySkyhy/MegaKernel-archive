# /workspace 仓库调研：算子仓库、推理框架与模型目录

> 调研时间：2026-09-26。调研方式为只读操作（`ls`、`find`、`git log/remote/branch`、`head`、Grep、`du -sh`、python 读取 safetensors 文件头），未运行任何构建、未修改任何文件、未加载模型权重。环境详情见 [01-environment.md](01-environment.md)。

本调研覆盖 /workspace 下全部关键目录：六个华为 CANN 官方开源仓（asc-devkit、cannbot-knowledge、cannbot-skills、ops-math、ops-nn、ops-transformer）、vLLM 上游源码与 vllm-ascend fork、目标模型目录 Qwen3.8-Flash-Next-MXFP4（下载进行中），以及只读共享资源目录 shared_assets。

---

## 一、昇腾官方算子与知识仓库

### 1. 总览

六个目录全部是华为 **CANN 官方开源仓**（remote 均为 `https://gitcode.com/cann/<repo>.git`，License 为 CANN Open Software License 2.0），本地均检出 `master` 分支、工作区干净、最近提交集中在 2026-09-24/25，非常活跃：

```text
asc-devkit        master  648a60182 2026-09-24 17:50:40 +0800 fix(hccl): use direct local copy for parallel AllGather mesh
cannbot-knowledge master  1e015cdc 2026-09-25 10:33:27 +0800 feat: 检索子词分词+IDF 加权与 Ascend C API outline triage
cannbot-skills    master  edcd4b94 2026-09-24 17:25:34 +0800 infra(pre-commit): 引入 pre-commit/OAT 检查体系
ops-math          master  0463a6407 2026-09-25 07:09:02 +0800 [Fix] 拒绝 aclnnSumV2 的空 Tensor 列表
ops-nn            master  fed553bea 2026-09-24 18:59:06 +0800 更改头文件注释
ops-transformer   master  72b1459d6 2026-09-24 18:27:59 +0800 fix: 修复 FlashAttn/FlashMLA TTK 比较与 LSE 打印
```

| 仓库 | 用途 | 文件总数 | 主要语言 | 对 mega kernel 参考价值 |
|---|---|---|---|---|
| asc-devkit | Ascend C 语言官方仓（API 实现 + 示例 + 文档） | ~14,800 | C++/h/cpp/md（.asc 内核 382 个） | **高** |
| cannbot-knowledge | CANNBot 知识仓（OKF 知识卡） | ~7,900 | Markdown（7,802 张） | **高** |
| cannbot-skills | CANNBot 技能仓（Agent Skills/Plugins） | ~6,300 | md/py/h/cpp | **高** |
| ops-math | CANN 数学基础算子库 | ~14,000 | cpp/h | 中 |
| ops-nn | CANN NN 算子库（matmul 等） | ~24,700 | cpp/h | **高** |
| ops-transformer | CANN Transformer 算子库（attention/moe/mc2） | ~14,300 | cpp/h/hpp/py | **高** |

来源：各仓 `git remote -v`、`git log -1`、`find <dir> -type f -not -path '*/.git/*' | wc -l` 及扩展名统计。

### 2. asc-devkit —— Ascend C 语言官方仓

**用途与来源**：Ascend C 官方仓库（即 Ascend C 编程语言的 API 实现 + 样例 + 文档，原 AscendC 社区仓）。README 自述："Ascend C 是 CANN 面向昇腾 AI 处理器打造的专用算子开发编程语言"，API 分四层：Basic API（C++ Tensor）、Tensor API（Layout 代数）、SIMD C API（原生指针）、SIMT API。`impl/` 下是该语言的运行时/API 实现本体。属**官方仓**。

**顶层结构**（`ls /workspace/asc-devkit`）：

```text
CHANGELOG.md  CMakeLists.txt  LICENSE  README.md  build.sh  cmake/
docs/         # zh/en API 文档 + guide（programming_guide、operator_practice、cross_gen_migration_guide）
examples/     # 01_simd_cpp_api  02_simd_c_api  03_simt_api  04_aicpu  05_simd_simt_hybrid
impl/         # adv_api  aicpu_api  basic_api  c_api  simt_api  tensor_api  utils
include/      # 同上各层头文件 + kernel_operator.h
scripts/  tests/  tools/
```

语言与规模：约 14,789 个文件；h 3,452、cpp 1,699、py 784、**.asc（Ascend C 内核实现）382**、cc 305（`find ... | sed 's/.*\.//' | sort | uniq -c`）。

**与 kernel 开发相关的重点内容**：

- `examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_mxfp4_high_performance/` —— **MxFP4 Matmul 性能调优样例**（Ascend 950PR/DT，CANN ≥ 9.1.0），`matmul_mx.asc` 内含 2 个优化 case：多核 MDL 常量化 tiling、scale 随 A/B 多倍搬运（`mxTypePara`）。这是与 Qwen3.8-Flash-Next-MXFP4 最直接相关的单文件参考。
- `examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_mxfp4_basic_api_high_performance/` —— 同上场景的 Basic API 版本（`mmad_mx.asc`）。
- `examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/matmul_gelu_high_performance/` —— matmul+GELU 融合最佳实践；同目录还有 `quant_group_matmul_high_performance`。
- `examples/01_simd_cpp_api/07_tensor_api/matmul_mxfp4_tensor_api_high_performance/`、`matmul_tensor_api`、`matmul_bias_fusion`、`matmul_split_k`、`batch_matmul_tensor_api` —— Tensor API 体系的 matmul 样例全集。
- `examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/load_data_2dmx_l12l0/` —— **950 新特性 Load2DMX**（FP4 GM→L1→L0 搬运格式图解与样例）。
- `examples/01_simd_cpp_api/06_compatibility_guide/` —— Ascend 950 兼容性专题：`matmul_s4`（稀疏 4:2）、`data_copy_l1togm`、`set_loaddata_boundary` 等。
- `impl/tensor_api/`（arch/atom/algorithm 三层）、`impl/adv_api/tiling` —— Matmul 高阶 API 的源码级实现，可做 tiling 模板参考。
- `docs/zh/guide/`（getting_started / programming_guide / operator_practice / **cross_gen_migration_guide**）—— 含跨代迁移指南。

**参考价值：高**。理由：MXFP4 matmul 的一手调优样例（恰好 950 平台）、融合算子最佳实践、SIMT/SIMD 双范式与 950 新硬件特性（mmad_mx、load2dmx）样例、以及 API 源码本体，是写 mega kernel 的语法/范式基座。

### 3. cannbot-knowledge —— CANNBot 知识仓

**用途与来源**：CANN 社区 Infra 智能体层 CANNBot 的**知识仓**（官方）。把官方文档/固定版本源码/实验观测蒸馏成带 YAML Frontmatter 的 Markdown 知识卡（Google OKF v0.2 格式），供 Agent 检索。仓内 `AGENTS.md` 规定了严格的知识生产/治理规则（知识正文只在 `knowledge/`，来源须固定到 40 位 commit 等）。纯**官方**维护。

**顶层结构**（`ls /workspace/cannbot-knowledge`）：`knowledge/`（唯一知识 Bundle 根）、`governance/`、`evals/`、`recipes/`、`docs/`、`logs/`、`.agents/skills/`。语言与规模：约 7,900 个文件，其中 **Markdown 7,802**、py 75——纯知识库。`knowledge/` 下分 `ops/`（ascendc、pypto、tilelang、triton、shared）、`model/`、`graph/`、`runtime/`、`common/`。

**与 kernel 开发相关的重点内容**：

- `knowledge/ops/ascendc/` —— Ascend C 专属知识：`apis/`（simd/simt/ai_cpu/utils API 卡）、`operators/aclnn/<cv|math|nn|transformer>/...`（API→Kernel 链路算子卡，**其中 950 平台专属卡 369 张**，命名 `*_950.md`，如 `operators/aclnn/transformer/mc2/quant_matmul_all_reduce_950.md`、`operators/aclnn/transformer/gmm/grouped_matmul_swiglu_quant_950.md`）、`optimizations/`（含 `optimizations/matmul/` 10 张 matmul 专项优化卡，如 `resident_operand_and_scale_reuse.md`、`static_tiling_and_runtime_shapes.md`）、`runbooks/`（精度问题排查）。
- `knowledge/model/inference/concepts/quantization/hybrid_mxfp8_mxfp4.md` —— **DeepSeek-V4 在 Ascend 950PR/DT 上的 FP8/MXFP8+MXFP4 混合量化方案**知识卡（OCP MXFP4 E2M1、MoE 路由专家 W4A8、KV Cache C8 伪量化、950DT 16 卡 9.84ms/1625TPS 数据），是 MXFP4 推理设计的核心参考。
- `knowledge/model/inference/concepts/kernels/index.md` —— "融合 Kernel" 索引，逐卡介绍 Compressor、Deepseek Indexer Attention（明确称其为 **PyPTO "MegaKernel" 能力**的典型实例）、FIA v1/v2、Lightning Indexer、MLA Prolog、SparseAttnSharedKV 等，含 A3 与 950PR/DT 的双平台实现差异。
- `knowledge/model/inference/guides/models/qwen3_next_atlas_a3_deployment.md` 及 `concepts/models/qwen3_next.md`、`runbooks/runtime/qwen3_next_block_size_128_exception_chunked_prefill.md` —— Qwen3-Next（用户目标的同架构前代）部署/优化知识（注意：现有卡面向 Atlas A3，非 950/MXFP4）。
- `recipes/operator_development.md`、`recipes/api_search.md` —— 算子开发与 API 检索的场景化最佳实践。

**参考价值：高**。理由：950 平台 369 张算子链路卡 + MXFP4 混合量化决策卡 + "MegaKernel" 融合 kernel 方法论，正是 mega kernel 开发的背景知识库；检索这些卡可避免重推导。注意：知识卡 sources 指向 `cann-recipes-infer` 等外部仓，本 workspace 未包含该仓正文。

### 4. cannbot-skills —— CANNBot 技能仓

**用途与来源**：CANNBot 的**技能仓**（官方），按 Agent Skills 开放标准提供可安装 Skill/Plugin/Agent，覆盖 AscendC/PyPTO/TileLang/Triton 四类 DSL 的算子开发、模型迁移与推理优化。仓内 `AGENTS.md` 给出五层架构（Plugins→Agents→Skills→References→Infra）。

**顶层结构**（`ls /workspace/cannbot-skills`）：

```text
ops/          # 59 个算子 Skill（正式版）
model/        # 21 个模型推理/训练优化 Skill
graph/  runtime/  infra/  tools/
plugins-official/    # 11 个官方编排插件
plugins-community/   # 16 个社区插件
scripts/  tests/  tools/  docs/
```

语言与规模：约 6,284 个文件；md 3,104、py 1,668、h 306、cpp 136。

**与 kernel 开发相关的重点内容**：

- `ops/` 下 Ascend C 技能全集：`ascendc-tiling-design`、`ascendc-st-design`、`ascendc-perf-optimize`、`ascendc-simt-tiling-design`、`ascendc-blaze-best-practice`（BLAZE 模板库）、`ascendc-regbase-best-practice`、`ascendc-mc2-best-practice`、`catlass-op-design/develop/perf-tune`（Cube/Matmul 高阶模板拼装）、`triton-op-designer/coding/verifier/latency-optimizer`（triton_ascend）、`pypto-op-*` 系列、`tilelang-op-*` 系列、`npu-arch`（含 `references/npu-arch-guide.md`、`npu-hardware-params.md` 硬件参数手册）。
- `plugins-official/catlass-op-generator/`、`triton-op-generator/`、`pypto-op-orchestrator/`、`tilelang-op-orchestrator/`、`ops-direct-invoke(-flash)/`、`ops-registry-invoke/` —— 四条 DSL + 两种调用方式的端到端算子生成编排插件。
- `plugins-community/cuda2ascend/` —— **CUDA→AscendC 移植**的多 Agent 工作流（architect/developer/qa 角色 + hooks），是全仓唯一直接面向 CUDA 移植的插件。
- `plugins-community/ascendc-port-orchestrator/` —— **跨代际移植编排**：arch22（910B/V220）算子自动移植到 arch35/A5（950PR/V300），含 KernelBench 风格 golden 精度比对流程，与"CUDA 移植 + 950 适配"高度对口。
- `plugins-community/tilelang2ascendc-ops-generator/`、`collaborative-agent-kernel-evolution/`（kernel 演化协作 Agent）、`ops-perf-evolution`。
- `model/model-infer-superkernel/SKILL.md` —— SuperKernel 算子二进制融合适配（ge_graph + A3 + decode）；`model-infer-quantization/`（含量化融合资料）、`model-infer-fusion/`、`model-infer-kvcache/` 等推理优化技能。
- 全仓仅 2 个 `.cu` 文件（`runtime/runtime_migration/evals/inputs/vector_add_cuda.cu`、`apiRuntimeCoverage_cuda.cu`，均为 runtime API 对比评测输入，非算子样例）。

**参考价值：高**。理由：几乎覆盖 mega kernel 开发所需的全部方法论技能（tiling 设计、性能调优、Catlass 模板、跨代移植、CUDA 移植编排），且官方保证了知识来源可信（AGENTS.md 明令禁止编造 API）。

### 5. ops-math —— CANN 数学基础算子库

**用途与来源**：CANN 算子库中数值计算基础算子库（官方，SIG ops-basic），含 conversion/math/random 三类，覆盖张量形态变换、基础数学、随机数。**2025/09 首次上线**，已支持 Ascend 950PR/DT（2026/07 引入 FLOAT8/FLOAT4 低精度类型、Philox PRNG、Reg 接口迁移）。

**顶层结构**：`math/`（abs、add、ada_cast…约数百个算子目录）、`conversion/`、`random/`、`common/`、`experimental/`、`spack/`、`docs/`、`examples/`、`tests/`。语言与规模：约 13,996 个文件；cpp 5,576、h 3,275、md 989、py 558。无 matmul/gemm 类算子（`find -maxdepth 2 -iname '*matmul*'` 无结果）。

**与 kernel 开发相关的内容**：

- `examples/add_example/`、`add_example_c_api/`（SIMD C API 范式）、`add_example_aicpu/`、`add_example_pypto/` —— 同一算子的四种 DSL 入门样例，适合作为新算子工程脚手架参照。
- `examples/fast_kernel_launch_example/` —— `<<<>>>` 直调 kernel 异构调用示例（与 ops-nn/ops-transformer 同机制）。
- `docs/QUICKSTART.md`、`docs/zh/develop/aicore_develop_guide.md`（与 ops-nn/transformer 共享同一套开发指南体系）。

**参考价值：中**。理由：无 matmul/attention 内核，但对"算子最小交付件"（op_host/op_kernel/op_api/op_graph/tests 目录骨架、CMake 工程、<<<>>>直调样例、QUICKSTART 流程）有直接参考价值；FLOAT4 类型支持的新算子（如 SignBitsUnpack）可作低精度数据搬运参考。

### 6. ops-nn —— CANN NN 算子库（matmul 主场）

**用途与来源**：CANN NN 高阶算子库（官方，SIG ops_nn），含 matmul、activation、conv、norm、quant、loss、rnn 等。2026/03 已支持 Ascend 950PR；亮点包括低 bit 融合算子（fp8/mxfp8/hifp8/mxfp4，pertensor/perchannel/pertoken/pergroup 量化组合）、SIMD/SIMT 同构编程算子、`<<<>>>` 直调样例。

**顶层结构**：`matmul/`（**34 个算子**）、`activation/`、`conv/`、`norm/`、`quant/`、`index/`、`loss/`、`foreach/`、`rnn/`、`vfusion/`、`torch_extension/`、`experimental/`、`common/`、`docs/`、`tests/`。语言与规模：约 24,731 个文件（六仓中最大）；cpp 8,259、h 6,424、py 1,299、md 1,406。

**与 kernel 开发相关的重点内容**（`ls /workspace/ops-nn/matmul/`）：

- `quant_batch_matmul_v4/` —— 全量化 BatchMatmul（README 明示支持 fp8/mxfp8/**mxfp4** 等组合；`tests` 中有 `arch35` 平台用例，如 `quant_batch_matmul_v3/examples/arch35/test_aclnn_quant_matmul_weight_nz_mxfp4.cpp`），W4A8 类 MoE 线性层实现参考。
- `weight_quant_batch_matmul_v2/` —— 伪量化（weight-only）融合 matmul，README 列为低 bit 代表作。
- `gemm_v3/`、`batch_mat_mul_v3/`、`mat_mul_v3/`、`fused_mat_mul/`、`fused_quant_mat_mul/`、`quant_matmul_activation_quant/`、`sparse4to2quant_matmul/`（稀疏 4:2 硬件加速）等。
- 每个算子目录均为完整交付件：`op_kernel/`（含 `arch35` 子目录，即 950 内核实现）、`op_host/`（tiling）、`op_api/`、`op_graph/`、`tests/`、`docs/`。
- `docs/zh/develop/aicore_develop_guide.md`、`docs/QUICKSTART.md`、`docs/zh/debug/npu_sim.md`（NPU Simulator 仿真调试）。

**参考价值：高**。理由：MXFP4/W4A8 量化 matmul 的生产级 AscendC 内核源码（含 950 arch35 分支）是 mega kernel 中 GEMM 部分的直接对标对象；目录骨架也是算子工程模板。

### 7. ops-transformer —— CANN Transformer 算子库（attention / MoE / 通算融合）

**用途与来源**：CANN Transformer 进阶算子库（官方，SIG ops_transformer），覆盖 attention、moe、mc2（通算融合）、gmm、mamba、mhc、posembedding。2025/12 起支持 Ascend 950PR/DT。

**顶层结构**：`attention/`（**80+ 算子目录**）、`mc2/`（通信+计算融合，45 个算子）、`moe/`、`gmm/`、`ffn/`、`mamba/`、`mhc/`、`posembedding/`、`common/`、`experimental/`（50+ 实验算子）、`examples/`、`docs/`、`torch_extension/`。语言与规模：约 14,269 个文件；h 4,872、cpp 3,465、py 1,331、hpp 810、md 773。

**与 kernel 开发相关的重点内容**：

- `mc2/mega_moe/` —— **MegaMoE 算子**：把 MoE 层 Dispatch + Linear1 + Activation + Linear2 + Combine 全流程融合为单算子实现通信计算掩盖，支持 950PR/DT——这是全 workspace 名字和形态上都最接近 "mega kernel" 的生产实现。
- `attention/quant_flash_attn/` —— Dense Attention 全量化算子；README 记载 A5（950）上支持 MxFP8 全量化，`experimental/attention/quant_flash_attn/` 的 A5 版本支持 **MxFP4 全量化**（`op_kernel/arch35/vf/vf_attenOut_dn_mxfp4.h`、`vf_computeScale_dn_mxfp4.h`）。
- `attention/flash_attn/` —— A5 非量化 Dense Attention FlashAttention 实现（支持 head_dim 64/72/128/256 及 MLA 非吸收形态 192/128）。
- `attention/fused_infer_attention_score/`（FIA，Paged KVCache 通用 FA）、`flash_mla_with_kvcache/`、`mla_prolog_v2/v3`、`incre_flash_attention/`、`prompt_flash_attention/`。
- `mc2/` 通算融合矩阵：`matmul_all_reduce(+_add_rms_norm)`、`allto_all_matmul(_v2)`、`grouped_mat_mul_all_reduce`、`moe_distribute_dispatch/combine(v3)`、`engram_fetch` 等——**通信与 matmul 融合**是 mega kernel 跨卡形态的直接参考。
- `gmm/grouped_matmul_swiglu_quant_v2/` 等——MoE 分组 matmul + SwiGLU + 量化融合。
- `examples/fast_kernel_launch_example/csrc/grouped_matmul` —— `<<<>>>` 直调 grouped_matmul 示例（README 记载 2026/01 新增）。
- `docs/zh/ascend950_op_list.md` —— Ascend 950 全量算子支持矩阵表；`docs/zh/develop/aicore_develop_guide.md` 开发指南。
- `experimental/attention/block_sparse_attention/op_kernel/attn_infra/...mxfp4*.hpp` —— arch35 MXFP4 attention epilogue（softmax、rescale、mask 全套）。

**参考价值：高**（六仓中对 mega kernel 最直接）。理由：MegaMoE 单算子融合通算全流程、MXFP4 全量化 attention 内核、matmul×集合通信融合族，三者正好构成"Qwen3.8-Flash-Next-MXFP4 推理 mega kernel"的三个核心拼图。

---

## 二、vLLM 与 vllm-ascend

### 1. /workspace/vllm

#### 1.1 git remote 与 checkout

来源：`git -C /workspace/vllm log/branch/remote/describe`

```bash
$ git -C /workspace/vllm log -1 --format='%H %ci %s'
8a2364605c0b0581ea5d0d3720cb1125b47abc6f 2026-09-26 00:40:22 +0000 [Core] Bound draft-token RPC waits by the execute-model timeout (#58779)
$ git -C /workspace/vllm remote -v
origin  https://gitcode.com/GitHub_Trending/vl/vllm.git (fetch/push)   # gitcode 的 GitHub 镜像
$ git -C /workspace/vllm describe --tags
v0.30.1rc0-189-g8a2364605c        # main 分支，v0.30.1rc0 之后 189 个提交
$ git -C /workspace/vllm status -s   # 空 —— 工作区干净，补丁未应用
```

当前检出：**main 分支 @ 8a2364605c（2026-09-26）**，版本约 v0.30.1rc0+。注意与 vllm-ascend README 要求的 **v0.23.0（pin commit 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665）不一致**（见 §2.6 风险提示）。

#### 1.2 声明的 Ascend 硬件 / CANN 版本

vLLM 上游仓库本身不声明 Ascend 要求（Ascend 支持由 vllm-ascend 插件提供）。但**该 main 分支已原生包含 Qwen3.8 的模型架构 `qwen4_exp`**（`vllm/models/qwen4_exp/`，来源 `grep -n Qwen4Exp vllm/model_executor/models/registry.py`）：

```python
# vllm/model_executor/models/registry.py:113/601/697
"Qwen4ExpForCausalLM":              ("vllm.models.qwen4_exp", "Qwen4ExpForCausalLM"),
"Qwen4ExpForConditionalGeneration": ("vllm.models.qwen4_exp", "Qwen4ExpForConditionalGeneration"),
"Qwen4ExpMTP":                      ("vllm.models.qwen4_exp", "Qwen4ExpMTP"),
```

`vllm/models/qwen4_exp/` 目录结构：`config.py` + `common/`（hyperconnection、ngram_embedding、ple、qsa_cache）+ **`amd/`** 与 **`nvidia/`** 两套厂商实现（model.py、mtp.py、qsa.py、low_latency_gemm.py、ops/ 等），**无 ascend/npu 专用子目录** —— NPU 路径由 vllm-ascend 插件以通用机制（"ascend" 量化方法、GDN 算子、融合 MoE）接入。

#### 1.3 目录结构要点

标准 vLLM 源码树（`vllm/`、`csrc/`、`docs/`、`tests/`、`docker/` 等）。与本次任务相关的 MXFP4 代码（来源 `grep -rli mxfp4 /workspace/vllm/vllm --include='*.py'`，节选）：

- `vllm/model_executor/layers/quantization/mxfp4.py` — 上游 MXFP4 量化配置（`quant_method` 注册了 `"mxfp4"`、`"gpt_oss_mxfp4"`，见 `layers/quantization/__init__.py:36,152,176`）
- `vllm/model_executor/kernels/linear/mxfp4/` — GPU 各厂商 MXFP4 GEMM 内核：`b12x.py`、`marlin.py`、`flashinfer.py`、`aiter.py`、`humming.py`、`xpu.py`、`emulation.py`
- `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py`、`aiter_mxfp4_w4a8_moe.py`
- `vllm/config/attention.py`、`vllm/config/kernel.py`、`vllm/envs.py` 等也有 mxfp4 开关

注意：上游 mxfp4 内核面向 GPU（marlin/flashinfer/xpu），**不含 Ascend 实现**；Ascend 侧实现在 vllm-ascend（见下节）。

### 2. /workspace/vllm-ascend

#### 2.1 git remote 与 checkout

来源：`git -C /workspace/vllm-ascend log/branch/remote/tag/rev-parse`

```bash
$ git -C /workspace/vllm-ascend log -1 --format='%H %ci %s'
35d606dfe2392122824cd00c802f7864d427e2dc 2026-09-06 21:34:33 +0000 examples+docs: add 'easy' preset with trivially predictable long outputs
$ git -C /workspace/vllm-ascend remote -v
origin  https://gitcode.com/liruixin_dvc/vllm-ascend.git   # 个人 fork（gitcode）
$ git -C /workspace/vllm-ascend rev-parse v0.23.0+ascend950 qwen3.8-950
35d606dfe2392122824cd00c802f7864d427e2dc
35d606dfe2392122824cd00c802f7864d427e2dc        # 分支与 tag 指向同一 commit
$ git -C /workspace/vllm-ascend log --oneline -8
35d606df examples+docs: add 'easy' preset ...
216e42fc examples: long-form preset ...
316511cc docs+examples: reorder dep installs ...
154c236d docs: pin exact vllm v0.23.0 commit ...
...（均为 docs/examples 提交）
```

当前检出：自定义分支 **`qwen3.8-950`**，与 fork 的 tag **`v0.23.0+ascend950`** 同一 commit（35d606df）。即版本基线是 **vllm-ascend v0.23.0 + Ascend 950 定制**。

#### 2.2 README 声明的硬件与 CANN 版本

来源：`/workspace/vllm-ascend/README.md`

上游通用部分（README.md:60-68）：

```text
- Hardware: Atlas 800I A2 Inference series, Atlas A2 Training series,
            Atlas 800I A3 Inference series, Atlas A3 Training series,
            Atlas 300I Duo (Experimental)
- OS: Linux
- Software: Python >= 3.10, < 3.13
            CANN == 9.1.0
            PyTorch == 2.10.0, TorchNPU == 2.10.0.post4
            vLLM (the same version as vllm-ascend)
```

fork 专属章节 **"Ascend 950 Quick Start (this fork, tag v0.23.0+ascend950)"**（README.md:78-85）：

```text
This fork adds MXFP4 (single-level and dual-level) W4A4 quantization, fused GDN
attention kernels and DFlash2 speculative decoding for Ascend 950 series, and ships
with pre-quantized Qwen3.8-27B models.
Tested environment: Ascend 950PR, CANN 9.1.0, PyTorch 2.10.0,
torch_npu 2.10.0.post4, Python 3.12.
```

与本机环境（Ascend950PR + CANN 9.1.0 + python3.12）完全吻合。

#### 2.3 目录结构要点（vllm_ascend/）

顶层（`ls /workspace/vllm-ascend/vllm_ascend/`）：`ops/`、`attention/`、`quantization/`、`models/`、`spec_decode/`、`compilation/`、`model_loader/`、`distributed/`、`eplb/`、`kv_offload/`、`device/`、`platform.py`、`envs.py`、`_cann_ops_custom/`、`xlite/` 等。

**ops/（算子）** — `gdn.py`、`gdn_attn_builder.py`（融合 GDN 线性注意力内核，Qwen3.8 的 linear_attention 层）、`mla.py`、`dsa.py`、`rope_dsv4.py`、`mhc.py`、`bailing_moe_linear_attn.py`、`fused_moe/`（moe_mlp.py、token_dispatcher.py、prepare_finalize.py 等）、`triton/`、`register_custom_ops.py` 等。

**attention/（attention 后端）** — `attention_v1.py`（后端注册入口）、`fa3_v1.py`、`mla_v1.py`、`dsa_v1.py`（DeepSeek Sparse Attention）、`sfa_v1.py`、`kvcomp_attn/`、`context_parallel/`（attention_cp.py、mla_cp.py、dsa_cp.py、sfa_cp.py）。

**quantization/methods/（量化方法）** — 这是 MXFP4 的核心：

```text
w4a16_mxfp4.py  w4a4_mxfp4.py  w4a4_mxfp4_dual.py  w4a4_mxfp4_flatquant.py
w4a8_mxfp4.py   w8a8_mxfp8.py  w4a4_flatquant.py  w4a4_laos_dynamic.py
fp8.py  w4a16.py  w4a8.py  w8a16.py  w8a8_block_fp8.py  w8a8_dynamic.py  ...
```

**models/** — `deepseek_v4.py`、`deepseek_v4_mtp.py`、`qwen3_dflash2.py`（**DFlash2 草稿模型**，经 `vllm_ascend/models/__init__.py` 注册为 `DFlash2DraftModel`）、`llama_eagle3_vwn.py`、`layer/`。**无 qwen4_exp 专用文件**（`grep -rn 'qwen4_exp|Qwen4Exp' vllm_ascend/` 无命中）——Qwen3.8 语言模型本体走 vLLM 上游 `vllm.models.qwen4_exp`，插件只提供草稿模型与算子。

**spec_decode/** — `dflash_proposer.py`、`dflash2_proposer.py`（README 所称 DFlash2 投机解码）、`eagle_proposer.py`、`ngram_proposer_npu.py` 等。

**csrc/（AscendC 自定义算子）** — `online_mxfp4_gemm/`、`mxfp4_nz/`、`quant_fusion/`、`moe/`、`mla_preprocess/`、`mc2/`、`gmm/`、`block_fp8_gemm/` 等；`_cann_ops_custom/vendors/custom_transformer` 即 README 要求 `export ASCEND_CUSTOM_OPP_PATH` 指向的算子包。

#### 2.4 MXFP4 相关代码 / 配置

来源：`grep -rli mxfp4 /workspace/vllm-ascend`（去掉了 .po 翻译文件），关键命中：

| 类别 | 路径 |
|---|---|
| AscendC 算子 | `csrc/online_mxfp4_gemm/`（CMakeLists、test/、work/verify_*.py）、`csrc/mxfp4_nz/build_op.py` |
| 量化方法 | `vllm_ascend/quantization/methods/w4a4_mxfp4.py`、`w4a4_mxfp4_dual.py`、`w4a4_mxfp4_flatquant.py`、`w4a16_mxfp4.py`、`w4a8_mxfp4.py` |
| 设备兼容 | `vllm_ascend/device/mxfp_compat.py`（`ensure_mxfp4_dtype_available`、`float4_e2m1fn_x2` 等） |
| 环境变量 | `vllm_ascend/envs.py:93-100`：`VLLM_ASCEND_MXFP4_NZ`（默认 1，packed FP4 权重转 FRACTAL_NZ） |
| 图融合 | `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`、`graph_fusion_pass_manager.py` |
| MoE | `vllm_ascend/ops/fused_moe/moe_mlp.py`、`moe_runtime_args.py`、`token_dispatcher.py`、`prepare_finalize.py` |
| 补丁 | `vllm_ascend/patch/worker/patch_qwen3_5.py` |
| 文档 | `docs/source/tutorials/models/DeepSeek-V4-Flash.md`、`DeepSeek-V4-Pro.md`、`GLM5.md`、`Qwen3.5-397B-A17B.md` 等；`supported_models.md` 的 "Ascend 950 Products" tab（DeepSeek V4-Flash/Pro 标注 "Native mixed MXFP8/MXFP4 weights"） |
| 测试 | `tests/ut/quantization/methods/test_w4a4_mxfp4.py`、`test_w4a16_mxfp4.py`、`test_w4a4_mxfp4_flatquant_dynamic.py`、`tests/ut/test_mxfp_compat.py`、`tests/e2e/.../test_*_mx_quant_mxfp4.py` |
| 示例 | `examples/ascend950_speed_demo.py` |

`w4a4_mxfp4.py` 要点（head + grep class/def）：

```python
class AscendW4A4MXFP4DynamicLinearMethod(AscendLinearScheme): ...   # W4A4 MXFP4 线性层
class AscendW4A4MXFP4DynamicFusedMoEMethod(AscendMoEScheme): ...    # W4A4 MXFP4 MoE
# MXFP4_NZ_LAYER_SUFFIXES = ("mlp.gate_up_proj", "lm_head") —— 仅这些形状转 FRACTAL_NZ
```

另：`vllm_ascend/platform.py:190-198` 把 `"ascend"` 量化方法注入 `vllm serve --quantization` 的 choices（`ASCEND_QUANTIZATION_METHOD = "ascend"`，见 `vllm_ascend/utils.py:49`），对应模型 config 里 `quant_method: "ascend"`。

#### 2.5 patches/ 与上游 vLLM 的关系

`patches/vllm-v0.23.0-ascend950.patch`（291 行）：修改 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`，为 Qwen3.5/3.8 的 GDN 层增加 `fuse_in_proj_qkvzba`（融合 qkvz+ba 投影为单个 GEMM，MXFP4 路径下启用）。**当前 /workspace/vllm（main 工作区）未应用该补丁**（`git status` 干净；`grep in_proj_qkvzba vllm/model_executor/layers/mamba/gdn/` 无命中）。

#### 2.6 重要风险提示

1. **版本错配**：fork README 要求 vLLM **v0.23.0**（pin `0fc695fc`），而 /workspace/vllm 检出的是 **main（v0.30.1rc0-189）**。两者差多个大版本，README 的安装/补丁流程不能直接套用，需决定：切 vllm 到 v0.23.0，或验证 main 分支与 vllm-ascend v0.23.0 基线的兼容性（上游 main 已自带 `qwen4_exp`，但 `in_proj_qkvzba` 补丁对应的是 v0.23.0 的 GDN 代码）。
2. **Python 环境**：`python3` 默认 3.11.6，README 要求 3.12；需用 `/usr/local/python3.12.13/bin/python3.12`（modelscope 下载进程即用它）。
3. **未安装**：vllm、vllm-ascend、torch、torch_npu、triton-ascend 均未安装；仅 `cann_ops_transformer 1.0.0`、`cannsim 0.1.0`。

---

## 三、目标模型目录：Qwen3.8-Flash-Next-MXFP4

### 1. 目录现状：下载进行中

```bash
$ ps aux | grep modelscope   # 有活动进程
root 9214 ... modelscope download --model lenlrx/Qwen3.8-Flash-Next-MXFP4 --local_dir ./Qwen3.8-Flash-Next-MXFP4

$ ls /workspace/Qwen3.8-Flash-Next-MXFP4/ | grep -v safetensors
LICENSE  chat_template.jinja  config.json  configuration.json  generation_config.json  merges.txt

$ ls *.safetensors | wc -l          # 完整分片：18/131（调研期间从 16 增至 18，仍在增长）
$ ls *.incomplete | wc -l           # 4 个未完成分片（00016-00019 等）
$ du -sh /workspace/Qwen3.8-Flash-Next-MXFP4/    # 51G（其中完整分片 47.1 GB）
```

**缺失文件**：`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`model.safetensors.index.json` 均尚不存在（只有 `merges.txt` 3.3MB 和 `chat_template.jinja`）——应随下载完成补齐。**按主流分片 3.2GB × 131 估算，完整模型约 400GB 量级**。分片大小样例（`du -sh`）：00001=992M、00002=851M、00003=566M、00005=2.0G、00006-00015=3.2G。

### 2. config.json 关键字段

来源：`Read /workspace/Qwen3.8-Flash-Next-MXFP4/config.json`（全文 417 行）

```json
{
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "model_type": "qwen4_exp",                        // 文本子配置 model_type: qwen4_exp_text
  "transformers_version": "5.8.0.dev0",
  "text_config": {
    "hidden_size": 2560, "num_hidden_layers": 48,
    "layer_types": ["linear_attention"×3 + "full_attention"]×12,   // 36 线性注意力 + 12 全注意力
    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "num_experts": 512, "num_experts_per_tok": 10,
    "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
    "linear_num_key_heads": 16,  "linear_key_head_dim": 128,       // GDN 线性注意力
    "linear_num_value_heads": 48,"linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "ngram_vocab_size_base": 20000000, "split_ngram_parts": 128, "ngram_size": 3,  // 大 n-gram embedding
    "mtp_num_hidden_layers": 1,                                   // MTP 层（投机解码草稿）
    "vocab_size": 248320, "max_position_embeddings": 262144,
    "rope_parameters": {"mrope_interleaved": true, "partial_rotary_factor": 0.25, "rope_theta": 10000000},
    "quantize_linear_attn": false,
    ...
  },
  "vision_config": { "depth": 27, "hidden_size": 1152, ... },       // 多模态视觉塔
  "quantization_config": {
    "quant_method": "ascend", "format": "mxfp4-pack-quantized-e8m0",
    "bits": 4, "element": "e2m1", "group_size": 32, "scale_dtype": "e8m0",
    "packing": "uint8-nibble-lohi",
    "npu_reference": "torch_npu.npu_dynamic_mx_quant(dst_type=float4_e2m1fn_x2, round_mode=\"round\")",
    "quantized_tensors": [ "...layers.{0..47}.mlp.experts.down_proj", "...experts.gate_up_proj",
                           "...shared_expert.{down_proj,gate_proj,up_proj}.weight" ],
    "cpu_rtn_provenance": { "tool": "qwen3.8-flash-next-cpu-rtn-mxfp4", "source_dtype": "bfloat16",
                            "ngram_bf16": true, "mtp_bf16": true }
  }
}
```

要点：48 层混合架构（3:1 的 GDN 线性注意力 : 全注意力），MoE 512 专家 top-10 + 1 个共享专家；**仅 MLP 专家/共享专家做 MXFP4 量化**（e2m1 + e8m0 scale、group 32、uint8 nibble 打包），注意力/ngram embedding/MTP 保持 bf16。`generation_config.json`：`do_sample=true, top_k=20, top_p=0.95, temperature=1.0, eos=[248046, 248044]`。

### 3. safetensors 头验证（只读 header，未加载权重）

用 python 读取分片前 8 字节长度 + JSON 头：

```text
== model-00001-of-00131.safetensors  tensors: 349   dtype: 全部 BF16
   （hyper_connection_mixer、layers.0.linear_attn.in_proj_a/b、conv1d 等）
== model-00005-of-00131.safetensors  tensors: 21    dtype: BF16×12, U8×8, I64×1
   model.language_model.layers.1.mlp.experts.down_proj           U8 [512, 2560, 320]
   model.language_model.layers.1.mlp.experts.down_proj.weight_scale U8 [512, 2560, 20]
   model.language_model.layers.1.mlp.shared_expert.gate_proj.weight_scale U8 [640, 80]
```

与 config 完全一致：512 专家、每专家 down_proj 输入 640→packed 320 字节（4bit×2/byte），scale 按 group_size=32 → 20 个 e8m0 scale/行。**磁盘格式即 OCP MXFP4（e2m1 + e8m0 block scale）的 uint8 打包形式**。

---

## 四、shared_assets

来源：`cat README.md`、`ls`、`du -sh`、`find`

README 自述："本目录存放共享的公共资源"，**只读**，由 hidevlab@huawei.com 通过工单维护；结构为 `models/`（公开模型权重）+ `datasets/`（公开数据集）。

**models/（约 430G）**：

```text
Qwen/    246G  Qwen2.5-7B、Qwen2.5-VL-3B-Instruct、Qwen3-0.6B、Qwen3-30B-A3B、Qwen3-Reranker-8B、
              Qwen3-VL-30B-A3B-Instruct、Qwen3.5-2B/4B、Qwen3.6-27B、Qwen3.6-35B-A3B
OpenBMB/  91G  MiniCPM-o-4_5、MiniCPM-o-4_5-gguf
Wan-AI/   87G  Wan2.1-T2V-1.3B(-Diffusers)、Wan2.2-TI2V-5B(-Diffusers)
BAAI/    6.3G  bge-large-zh-v1.5、bge-reranker-large
```

**datasets/（约 92G）**：`MTEB/`（4.5G，含 Daily-Omni）、`lmms-lab/`（80G，含 Video-MME）、`MMMU___mmmu/`（2.9G）、`LLaVA-Instruct-150K/`（1.6G）、`CowboyZ/`（2.4G，seed-tts-eval）、`cais/`（258M）、`tatsu-lab/`（63M）。

**HC2026/**：`find -type f` 结果为 **0 个文件**，只有空目录骨架（`skills/model/`、`skills/other/megatron_vs_hf/engine/dumped/qwen3_5_a3b_4l/{replay_seq0,...}` 等），疑似预留给 HC2026（Huawei Connect 2026）的 skill/回放数据挂载点，当前无实际内容。

**对项目的用途**：可直接作为 mega kernel 开发的评测/对照资源——精度评测数据集（MMMU、MTEB、Video-MME、LLaVA-Instruct、GSM8K 需另下，vllm-ascend 的 speed demo 会从 ModelScope 自动拉取）、基线模型权重（Qwen 系列可做小模型上算子原型验证）；注意整个目录只读，不应写入。

---

## 补充说明与缺口（针对六个官方算子/知识仓）

1. **未发现的内容**：六仓中无 `.cu` CUDA kernel 样例（仅 cannbot-skills 有 2 个 runtime API 对比评测输入文件）；无直接针对 "Qwen3.8-Flash-Next-MXFP4" 的知识卡（最接近的是 cannbot-knowledge 中 Qwen3-Next 于 Atlas A3 的部署卡，和 DeepSeek-V4 于 950 的 MXFP8-MXFP4 量化卡）；cannbot-knowledge 大量卡片来源指向 `gitcode.com/cann/cann-recipes-infer`（含 DeepSeek-V4/Qwen3-Next 推理 recipe 正文与 PyPTO MegaKernel 源码），该仓**不在本 workspace**，如需可建议后续拉取。
2. **版本提示**：各仓 README 均声明源码随 CANN 版本发布、master 分支可能与已装 CANN 版本不匹配（README："使用 master 分支可能存在版本不匹配的风险"），引用具体 API 时建议核对本地 CANN 版本。
3. 本次仅做只读盘点，未运行任何构建/测试，未验证示例可在本机编译。

---

## 综合结论

1. **目标模型** `lenlrx/Qwen3.8-Flash-Next-MXFP4` 正在下载（modelscope 进程活跃），目前 18/131 分片、约 47GB，预计全量约 400GB；tokenizer 与 index json 尚未就位，**现在不能启动推理**。
2. **软件栈未就绪**：vllm（main @ 8a2364605c）与 vllm-ascend（v0.23.0+ascend950 基线，分支 qwen3.8-950）源码已在 /workspace，但存在**版本错配**（插件要求 vLLM v0.23.0 + patch，当前是 main/v0.30.1rc0 且未打补丁），且均未安装；Python 需用 3.12。
3. **MXFP4 能力已齐备**：vllm-ascend 含 W4A4/W4A8/W4A16 MXFP4 量化方法、online_mxfp4_gemm 与 mxfp4_nz 自定义算子、融合 GDN 内核、DFlash2 投机解码（含 Qwen3.8 DFlash2 草稿模型）；上游 vLLM main 已原生支持 `qwen4_exp` 架构（仅 amd/nvidia 专用实现，NPU 走插件通用路径）。
4. 模型量化方案 = **MLP 专家/共享专家 MXFP4（e2m1+e8m0, group32, uint8 nibble 打包，quant_method=ascend）**，其余 bf16——与 vllm-ascend 的 `AscendW4A4MXFP4DynamicFusedMoEMethod` 对应，是 mega kernel 开发（量化 GEMM/ MoE 算子）的直接切入点。
