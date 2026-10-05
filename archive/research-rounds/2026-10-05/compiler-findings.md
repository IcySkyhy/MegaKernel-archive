# 2026-10-05 增量调研：编译器、运行时与 whole-model megakernel

## 结论先行

本轮先与 `archive/CATALOG.md` 和 `archive/MEGAKERNEL_OPEN_SOURCE_LANDSCAPE_2026-08-26.md` 去重；下列八个重点仓库在两份旧索引中均无命中。按“技术含量、源码可复现性、时间窗口”综合排序，建议主报告优先加入：

1. [cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel)：本轮最强的生产形态 decode megakernel，Apache-2.0，核心 CUDA 与调度器完整公开。仓库创建于窗口前一天，但 9 月 8 日公开发布和推送都落在窗口内，应作为“窗口内首次公开”收录，并显式保留日期口径说明。
2. [kiddyboots216/training-megakernel](https://github.com/kiddyboots216/training-megakernel)：目前少见的整训练步常驻 megakernel；前向、反向、分布式梯度工作、全局范数裁剪和 AdamW 都在每 GPU 一次 cooperative CUfunction 的 resident run 内。创建于 9 月 1 日，9 月 26 日形成 0.1.0 实质发布，属于“窗口前创建、窗口内重大公开成熟”。
3. [pengjh0111/TileMega](https://github.com/pengjh0111/TileMega)：从参数化 Coupling Graph、符号依赖分析、放置/搜索到 L1/L2 persistent CUDA executor 的完整编译器骨架，并在 10 月 4–5 日仍有实质 runtime/backend 改动。技术价值高，但仓库没有许可证文件、GitHub 也未识别到许可证，因此只能记为公开源码，不能按严格开源软件归类。
4. [WilliamZhang20/megakernel-gen](https://github.com/WilliamZhang20/megakernel-gen)：高价值的此前遗漏项。Apache-2.0，Rust 编译器直接从 Hugging Face checkpoint 生成一次 cooperative launch 的整模型前向、LM head 和 sampler；创建及最后推送都在 9 月 3 日，不属于严格窗口新增。
5. [thebasedcapital/latticemk](https://github.com/thebasedcapital/latticemk)：10 月 2 日新建，MIT；最终有真实 INT4 persistent decode megakernel，但项目名中的 lattice 路线本身失败，早期 persistent 版本也输给 CUDA Graph，主报告应把“最终 INT4 结果”与“失败的 lattice 假设”分开写。
6. [ranvier-labs/lean-cuda-qwen](https://github.com/ranvier-labs/lean-cuda-qwen)：9 月 9 日新建，Apache-2.0；Qwen3.6/3.8 的 persistent decode、聊天和训练程序源码公开，但所需 staged CUDA Lean 编译器只给公共 nightly 二进制，编译器源码需 companion repository 权限，故是应用源码开放、关键工具链不完全开放。
7. [Parth-Badgujar/transformer-megakernels](https://github.com/Parth-Badgujar/transformer-megakernels)：MIT，CuTe DSL/CUTLASS 的多层 transformer body megakernel和静态 SM 调度器；是高价值旧遗漏，但不是窗口内新项目，也没有充分证据把它写成包含 embedding、LM head、sampling 的完整 token pipeline。
8. [hoid-ai/hoid-megakernel-qwen](https://github.com/hoid-ai/hoid-megakernel-qwen)：10 月 3 日新建，Apache-2.0，但核心 worker/controls 是按 batch 预编译的 cubin，execution plan 也是二进制；公开的是 vLLM adapter、scheduler、加载器和复现包装层。应收为 `C-PARTIAL`，不能作为完整开源 megakernel 实现。

## 时间与许可证矩阵

所有时间为 GitHub 公共元数据的 UTC 时间。`pushed_at` 只能说明仓库引用被推送，不能单独证明技术内容变化；实质性判断另结合提交主题、README 和源码。

| 仓库 | created_at | 最后推送/实质变化 | 许可证 | 本轮判断 |
| --- | --- | --- | --- | --- |
| cohere-ai/cohere-megakernel | 2026-09-04 18:26 | 2026-09-08 16:27；9 月 8 日 README/博客发布 | Apache-2.0 | 窗口内首次公开；完整核心源码 |
| pengjh0111/TileMega | 2026-08-28 16:28 | 2026-10-05 03:23；10 月 4 日仍有 backend/runtime 实改 | 未声明 | 已有仓库、窗口内重大更新；公开源码但非严格 OSS |
| hoid-ai/hoid-megakernel-qwen | 2026-10-03 18:27 | 2026-10-03 18:47 | Apache-2.0 | 真新增，但核心为 cubin/二进制计划，`C-PARTIAL` |
| thebasedcapital/latticemk | 2026-10-02 00:12 | 2026-10-02 16:40 | MIT | 真新增；实验型 INT4 whole-decode persistent kernel |
| ranvier-labs/lean-cuda-qwen | 2026-09-09 00:55 | 2026-09-09 00:59 | Apache-2.0 | 真新增；应用源码开放、编译器后端不完整开放 |
| kiddyboots216/training-megakernel | 2026-09-01 20:20 | 2026-09-30 09:21；9 月 26 日 0.1.0 主发布 | MIT | 窗口前创建、窗口内实质发布 |
| Parth-Badgujar/transformer-megakernels | 2026-06-03 22:16 | GitHub pushed_at 为 2026-09-06；未检出窗口内主分支提交 | MIT | 旧遗漏；不要把 pushed_at 当成新实现 |
| WilliamZhang20/megakernel-gen | 2026-09-03 01:34 | 2026-09-03 01:34 | Apache-2.0 | 旧遗漏；严格窗口外 |

## TileMega 源码归档范围

- 本地位置：`archive/compiler-runtimes/TileMega`
- 上游与分支：[pengjh0111/TileMega](https://github.com/pengjh0111/TileMega)，`tilemega`
- 归档方式：浅层、partial clone、稀疏检出；`.git` 保留，工作树与上游 tip 一致。
- 本地已物化 508 个文件，约 10.36 MiB。
- 明确排除 `docs/experiments/**`。上游该目录含大量原始 trace、构建日志、生成计划和实验结果；GitHub recursive tree 返回 53,620 项且标记 `truncated=true`，因此不能用那份 API 返回值证明仓库全树规模或文件缺失。
- 保留了根文档、`configs/`、`include/`、`lib/`、`python/`、`scripts/`、`test/`、`tools/` 等源码与常规文档。`third_party/cutlass` 和 `third_party/barvinok` 保持上游 gitlink，未展开子模块。
- 根目录没有 `LICENSE`、`LICENSE.md` 或 `COPYING`；GitHub license 字段为空。源码文件中个别 SPDX 头不能替代整个仓库的明确许可。
- 归档未运行构建或基准；本轮只做源码结构与机制核验。

## TileMega：源码所证实的编译与调度机制

### 1. 从模型图到 Coupling Graph

README 给出的主路径是：`torch.export` → 稳定 export JSON → MLIR Coupling Graph → 生成 CUDA 或共享库。关键入口包括：

- `python/tilemega/export_bridge.py`
- `include/tilemega/Frontend/TorchExportImporter.h`
- `lib/Frontend/SemanticLifting.cpp`
- `lib/Frontend/ServingSemanticLifting.cpp`
- `include/tilemega/Dialect/CouplingGraph/CGDialect.td`
- `include/tilemega/Dialect/CouplingGraph/CGOps.td`
- `lib/Dialect/CouplingGraph/CGDialect.cpp`

Coupling Graph 不只是普通 operator DAG。它显式表达 task space、tensor access、placement、event tensor，以及每条 coupling 的 wait、fanout、volume 和 count。`lib/Dialect/CouplingGraph/CGDialect.cpp` 会用关系的 cardinality 验证 wait/fanout，并检查 `sum(wait) == sum(fanout)`；这与 README 所述 ISL 参数化关系和 barvinok 参数化计数相符。

### 2. 符号分析、成本模型和放置

关键实现分布在：

- `lib/Analysis/CouplingDerivation.cpp`
- `lib/Analysis/CouplingRelation.cpp`
- `lib/Analysis/TaskWork.cpp`
- `lib/Solver/BackendCostQuery.cpp`
- `lib/Solver/BalancedPlacement.cpp`
- `lib/Solver/ListScheduler.cpp`
- `lib/Solver/SkeletonSearch.cpp`
- `lib/Solver/VariantSchedule.cpp`
- `include/tilemega/Dialect/CouplingGraph/PlacementSolvePass.h`

`ListScheduler.cpp` 按 DAG level、剩余高度和节点次序生成稳定的拓扑顺序；更完整的 solver 同时包含 balanced placement、chain/skeleton search、backend cost query、resource/co-residency 和 runtime projection。IR verifier 强制非 legacy 映射携带 `resident_only=true`，并可用符号关系证明 `0 < grid <= resident_limit`，说明调度结果是为常驻 grid 约束生成，而不是事后把普通图简单包进一个 kernel。

### 3. 两级 persistent executor

核心源码：

- `include/tilemega/Codegen/tasks/ModelHarness.cuh`
- `include/tilemega/Codegen/executor/ServingPages.cuh`
- `include/tilemega/Codegen/tasks/ServingRuntime.cuh`
- `include/tilemega/Codegen/tasks/EventSync.cuh`
- `include/tilemega/Codegen/tasks/Placement.cuh`
- `lib/Codegen/Codegen.cpp`

源码同时实现两个主要执行层级：

- **L1 stage-major 模式**：`tilemega_l1_kernel` 在一个 persistent launch 内运行运行期 stage table。每个 resident CTA 对当前 stage 调用 `RunStage`，再用 grid event/barrier 进入下一 stage；`tilemega_l1_loop_kernel` 可在同一 launch 内连续跑多个 decode step。
- **L2 task-event 模式**：`tilemega_l2_kernel` 为每个 worker/CTA 读取 `schedule_offsets[worker:worker+1]`，逐个执行预先分配的 `TaskRef`。任务先根据生成的 dependency slice 等待 event，再调用 `RunTask`，最后 `NotifyTask` 发布细粒度或聚合事件。`TILEMEGA_EVENT_KAPPA` 控制把多少 producer task 聚为一组事件；slot window 允许在有限窗口中选择已就绪任务，而不必严格按一个全局 stage barrier 前进。

这两条路径很重要：TileMega 不是只实现一种全栅栏 megakernel，也不是单纯通用设备队列。它保留较简单的 L1 stage-loop，同时用 L2 per-worker 静态任务表 + 细粒度 event 实现跨 stage 重叠。

### 4. 同步、预取与 handoff

- `include/tilemega/Codegen/tasks/EventSync.cuh`：device-scope event load/publish 与 graded wait。
- `include/tilemega/Codegen/executor/ServingPrefetch.cuh`：stage 级 arrival/epoch 协议和下一 stage 预取。
- `include/tilemega/Codegen/executor/Prefetch.cuh`：L2 只读 frontier operand 的异步预取。
- `include/tilemega/Codegen/executor/PageRing.cuh`：带 sequence tag 的 page ring。
- `include/tilemega/Codegen/executor/DirectHandoff.cuh`：同 CTA/page 内 producer-consumer 直接交接，可省全局中间缓冲和事件。
- `include/tilemega/Codegen/executor/LastArriver.cuh`：last-arriver 类型的 reduction handoff。
- `lib/Dialect/CouplingGraph/HandoffAccess.cpp` 与 `HandoffPass.cpp`：依据精确 tensor access 与 placement 关系选择/重写 handoff，而不是在 CUDA 模板里硬编码。
- `include/tilemega/Codegen/executor/Async.cuh` 与 `ServingLaunch.cuh`：按目标能力使用 async copy、L2 prefetch、Programmatic Dependent Launch 等路径。

10 月 4 日最近一批实质提交主题也与源码一致，包括：从 packed tile stages 加载 nonpaged weights、拆分 paged loop roles 以缩短 cursor 生命周期、用 slim control 去掉冗余 L2 barriers、补 tiled GEMV tail/deep-copy pipeline 覆盖，以及公开 registered lookahead cache policy 控制。10 月 5 日的 tip 主要是最终 canary review 文档发布。

### 5. 算子与硬件范围

生成/运行时已有 embedding、RMSNorm、QK norm、RoPE、KV append、GEMM 与 combine、paged/fused attention、attention merge、elementwise/add、argmax reduce 等 task body；服务配置覆盖 Llama 和 Qwen 的 batch-1/nonpaged/paged decode。目标配置包括 `sm_80`、`sm_89`、`sm_90`、`sm_100` 和 `sm_120`，README 记录的 `sm_120` 构建使用 CUDA 12.8。其现阶段仍明显偏 NVIDIA CUDA/CuTe/CUTLASS，不是 AMD/ROCm compiler backend。

## 其他重点候选的源码判断

### cohere-ai/cohere-megakernel — 推荐主档案

- 核心路径：`src/decode/megakernel.cuh`、`src/decode/schedule.py`、`src/decode/runtime.cu`、`src/decode/abi.h`、`src/decode/launch.cuh`、`src/decode/gemm-n8-wgmma.cuh`。
- 范围：North Mini Code 30B MoE，单 H100/sm90a，BF16，batch 1–8；公开服务层支持 continuous batching、ragged sequences、paged KV、sliding window、prefix cache 和 preemption。
- 调度：每 SM 一个 resident CTA；host 为每个 SM 构造 instruction/task list。规则性 wave 采用静态顺序，full-attention 与 MoE 使用动态队列/窃取；global-memory counters 表达依赖，producer/storer/consumer warpgroups 配合权重预取。
- 覆盖：opcode 包含 QKV、attention decode/combine/drain、O-proj、router/top-k、MoE gather/up/down/combine、add+RMSNorm 和 LM head。
- 边界：prefill 仍是独立 PyTorch kernels，并会暂停 decode；源码还把 embedding lookup 和最初 RMSNorm 放在 megakernel 外部。宜写“完整 decode forward 的主计算图”，不要扩大为整个请求生命周期或训练系统。
- 作者结果可记录为“作者报告”：BS1 decode 相对 vLLM 1.58×，端到端约 1.25–1.41×；不要把作者基准写成本轮复测。
- 关联一手说明：[Cohere Megakernels 博客](https://cohere.com/blog/megakernels)。

### kiddyboots216/training-megakernel — 推荐主档案

- 核心路径：`kernel/program/training_program.py`、`resident_step.py`、`tile_schedulers.py`、`decoder_layer.py`、`attention.py`、`fa4_forward_kernel.py`、`fa4_backward_kernel.py`、`communication/decoder_fabric.py`、`grid_barrier.py`、`clipped_adamw.py`；host 协议在 `src/training_megakernel/circular_refill_runtime.py`、`resident_protocol.py`、`schedule.py`。
- 范围：Qwen3-8B 宽度、2–82 层、8×H100 NVLink；每 rank 进入一个 cooperative CUfunction，跨 optimizer steps 常驻。
- 单次 resident step 内含完整 forward、backward、分布式 gradient work、global-norm clipping 和 AdamW；host 只负责 refill token slot、checkpoint 和 telemetry。
- 这不是“把一个训练 block 融合”那么窄，而是当前开源图景中最接近 whole-training-program megakernel 的项目。MIT 许可明确，引用的 FA4/Quack 也附第三方许可文本。
- 关联说明：[作者项目博客](https://kiddyboots216.github.io/megakernel/)。

### WilliamZhang20/megakernel-gen — 高价值旧遗漏

- 核心路径：`mkc/src/hf.rs`、`arch.rs`、`ir.rs`、`plan.rs`、`search.rs`、`codegen.rs`、`emit.rs`、`validate.rs`；primitive library 在 `runtime/include/mk/{common,gemv,norm,rope,attn,moe}.cuh`。
- 输入是 Hugging Face checkpoint 目录，不是 traced program；编译器探测 config/tensor role，建立 Model IR，结合一次测得的 machine JSON 决定 grid、occupancy、row blocking、key split、barrier 和 arena，再生成 CUDA/host runtime/weight table。
- 生成的 `mk_megakernel` 使用 cooperative launch 和 resident grid，stage 之间以 `cg::this_grid().sync()` 风格的 grid barrier 连接；同一 launch 包含所有 layer、LM head 和 sampler。
- 支持 Llama/Qwen/Phi/Gemma/gpt-oss 等 decoder-only 变体，以及 BF16/F16/MXFP4/FP8/INT4 AWQ；项目明确限定单 GPU、batch 1 decode，没有 paged KV、continuous batching、prefix cache 或高效 prefill。
- Apache-2.0 且核心编译器/primitive 均有源码，严格开源评级高；但 created/pushed 均为 9 月 3 日，应列“此前遗漏”，不是 9 月 5 日以后新增。

### thebasedcapital/latticemk — 实验型真新增

- 关键实现：`kernels/megakernel_v2/mega2.cu`、`kernels/megakernel_mt/mega_mt.cu`、`kernels/megakernel_attn/mega_mt.cu`、`kernels/megakernel_kv/megakv.cu`、`kernels/megakernel_kvc/kvc.cu`、`kernels/megakernel_sync/mega3.cu`、`kernels/megakernel_scale/mega_scale.cu`、`kernels/fused/fused.cu`，以及对应 `sched_*.py`/JSON schedule。
- 实际可保留的贡献是面向 Turing `sm_75`、Qwen3-0.6B/1.7B、batch-1 的 GPTQ INT4 decode megakernel；源码显示持久 grid、每层多次 grid barrier 和静态生成 schedule。
- README 诚实记录 lattice code 在目标质量/速度点失败，第一版 persistent kernel 也输给 CUDA Graph；后续 occupancy 修正后的 INT4 版本才形成最终结果。因此不要把它描述成“lattice 压缩成功的 megakernel”。
- 作者基准可作为作者报告引用，但硬件、模型、量化和 batch 都很窄，且大量 schedule JSON 是静态特化结果。

### ranvier-labs/lean-cuda-qwen — 源语言项目，工具链部分闭合

- 核心路径：`lib/LeanCudaQwen/Qwen36/Megakernel.lean`、`FullTrainingMegakernelCore.lean`、`TrainingMegakernelCore.lean`；应用入口在 `examples/qwen36_megakernel/Qwen36ModelMegakernel.lean`、`Qwen36TokenMegakernel.lean`、`Qwen36TrainingMegakernel.lean`，另有 `qwen38_chat` 与 `qwen38_train`。
- Qwen3.6-27B recurrent Gated DeltaNet decode 被表达为 cooperative-grid persistent launch，并与 separate launches/host Lean oracle 对照；Qwen3.8 还提供真实 checkpoint chat worker，以及 LoRA SFT/DPO/GRPO 路径。
- Apache-2.0 覆盖仓库内的 Lean 程序与应用代码；README 要求安装固定版本的公共 Lean CUDA nightly 二进制，并明确完整 backend setup 只对有 companion compiler repository 权限的用户开放。
- 因此可将它记为“开源 megakernel 程序/模型库”，但不能称为完整开源 Lean→CUDA compiler stack。

### hoid-ai/hoid-megakernel-qwen — 只收 partial

- Python/host 关键路径：`src/vllm_qwen3_megakernel/adapter.py`、`execution.py`、`native.py`、`scheduler.py`、`artifacts.py`。
- device payload 位于 `src/vllm_qwen3_megakernel/cubins/b{1,2,4,8}/worker.cubin`、`controls.cubin` 和 `execution.bin`。
- `execution.py` 把上述三项作为 payload；`native.py` 使用 CUDA driver API 加载 module、定位 function 并启动 kernel。仓库未发现产生这些 cubin 的 CUDA/CuTe/Triton 源码。
- 适合研究 H200/Qwen3-4B 与 vLLM plugin 的四种 batch artifact 组织、host scheduling 和复现接口，但核心设备程序不可审阅或再编译，故 `C-PARTIAL`。

### Parth-Badgujar/transformer-megakernels — 多层 body，而非完整 token 生命周期

- 核心路径：`src/transformer_megakernel/megakernel.py`、`scheduler.py`、`operators/attention.py`、`operators/matmul.py`、`operators/rmsnorm.py`。
- 面向 `sm120a`，用 CuTe DSL/CUTLASS 实现 Llama3/Qwen2.5 风格的 RMSNorm、QKV、attention、O-proj 和 SwiGLU 多层 body。
- scheduler 为每层的算子 tile 计算代价，分配到当前负载最低的 SM，生成每 SM schedule；kernel 以 `grid=(num_sms,)` 常驻 CTA 读取自己的 work list，并用 counters/atomics 表达跨算子依赖。
- 当前证据更适合标为深度 subgraph/multi-layer transformer-body megakernel；没有足够源码证据把 embedding、LM head 和 sampling 一并归入一个 token launch。

## 已有仓库在本窗口的增量

- [mirage-project/mirage](https://github.com/mirage-project/mirage)：已有档案，9 月新增 SM100 temperature/top-k/top-p sampling，随后有 split-launch scheduler packing、Hopper multi-token paged attention、Unified KV Cache Pool/Hybrid KV Spec 等实改。属于既有 MPK 的 serving 能力扩张，不是新架构。
- [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)：已有档案，10 月初 MegaMoE 扩到 SM90 native BF16、SM100/SM103 FP4/MXFP8 和更大的 token route，并出现 Rubin generation/combine 相关路径。属于 MegaMoE backend 扩张。
- [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)：9 月有 fused MXFP8 intra-node dispatch、可配置 SM budget 和 AMD MoRI CI；仍是分布式通信/算子融合底座，不应改写成 whole-model megakernel。
- TeraMoE：9 月出现新的 BF16/FP8 autograd API 与 sort-map 演进；属于已有 MoE 项目成熟。
- DeepGEMM：9 月公开发布及 Mega-MoE task-info slot release ordering 修正，机制类别未变。
- Luminal：9 月到 10 月有 CUDA graph shared arena、dynamic bucket、SPMD/ShapeVar/SSA 等编译器更新，但未找到本窗口新增 megakernel backend 的直接证据。
- AutoMegaKernel：GitHub `pushed_at` 有 9 月值，但本轮未获得主分支窗口内实质提交证据；不应仅据 pushed_at 写成技术更新。
- ROCm/fleet-chiplet-megakernel：本窗口未见 megakernel 核心技术变化。AMD 方向本轮也未找到一个同时满足“新建/实改、whole-model 或 multi-op persistent、核心源码与许可证完整”的新强候选。

## 论文有更新、代码未闭合

- [Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)：2026-09-18 提交。论文描述 4×H100 上的 persistent MoE megakernel，根据 routing 后的实际负载用 kernel 内 cost model 动态分配每层/每 GPU 的 SM，并结合 spatial/temporal scheduler。本轮未验证到作者官方开源实现，先记 paper-only。
- [MegaFlux](https://arxiv.org/abs/2610.00671)：2026-09-30 提交。把 TensorRT-LLM/CuTe DSL MegaMoE 扩到 pipelined expert replication，并给出 backward megakernel；本轮未验证到官方公开仓库，先记 paper-only。

## 排除、降级与避免误判

- Hoid：不是“Apache-2.0 即核心开源”；许可证存在，但设备核心只有 cubin。
- TileMega：不是“GitHub public 即 OSS”；没有仓库级许可证。
- Lean CUDA Qwen：不是“应用源码公开即编译器也公开”；required compiler backend 只有 pinned nightly/受限 companion source。
- latticemk：不能把项目名当技术结果；lattice 路线失败，最终亮点是 INT4 persistent decode。
- transformer-megakernels：不能从“多层 transformer”外推为完整 embedding→sampling token pipeline。
- `pushed_at`/`updated_at` 不能替代提交或源码证据；Parth 项目、AutoMegaKernel 等必须按这个原则降级日期断言。
- `ForgeKernel`、若干个人 `megakernel`/`vibe-megakernel` 仓库缺少足够实现或许可；本轮不升主档案。
- AMD 检索中的若干 exllamav3 ROCm port、Fleet 作业/复刻仓库没有新 whole-model megakernel 机制，不应重复收录上游或算作新库。

## 检索覆盖与可证实的时间限制

本轮在 2026-10-05（Asia/Shanghai）执行，目标时间窗按 UTC 取 2026-09-05 至 2026-10-05。覆盖了：

- GitHub repository search/API：`megakernel`、`mega kernel`、`single kernel transformer GPU`、`persistent megakernel CUDA LLM`、`whole-model CUDA kernel LLM`、`persistent task runtime CUDA compiler`、`megakernel ROCm AMD` 等组合；
- 已知主仓库的最近提交与默认分支源码树；
- GitHub Topics、项目 README/博客、arXiv 新论文及代码链接；
- 旧 `CATALOG.md` 与主报告去重；
- GitLab/Hugging Face 的公开网页与搜索结果交叉查询，但这些平台没有发现比上述候选更强、且能同时核实许可证和核心源码的新 NVIDIA/AMD 项目。

限制如下：

- GitHub 搜索只覆盖公开、可索引的默认分支；私有仓库、刚创建但尚未索引、重命名仓库和只在非默认分支出现的代码可能漏检。
- GitHub API 未认证额度在后段耗尽；关键候选已用此前保存的 API 证据、GitHub HTML/raw 文件和本地 sparse archive交叉核验。
- TileMega recursive tree 被 GitHub 截断，因此只对本地明确物化的 508 个源码/文档文件和被排除的 `docs/experiments/**` 范围负责，不声称完整列举上游所有实验制品。
- 2026-10-05 当天仍在发生提交；本结论只能证明检索截止时可见的状态。
- 所有性能数字均是上游作者报告，本轮没有硬件复测。
