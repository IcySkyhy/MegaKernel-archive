# MegaKernel 增量研究：MoE、跨 GPU 融合与 2026-09-05 至 2026-10-05 新论文/代码

> 调研日期：2026-10-05  
> 范围：在 2026-08-26 总报告与 archive/CATALOG.md 基础上去重，重点核查 MoE、跨 GPU 计算—通信融合、persistent task runtime，以及 2026-09-05 至 2026-10-05 的论文和研究代码。  
> 证据原则：优先作者仓库、官方项目仓库、arXiv 正文和官方技术博客；仓库创建日、提交 authored/committed date 与实质代码内容交叉判断，不把 pushed_at 当作发布或创新日期。性能数字均为作者报告，本轮不做硬件复测或性能审计。

## 一、结论摘要

这一轮最重要的新事实不是“又出现了一批把 kernel fusion 称作 MegaKernel 的小项目”，而是 MegaKernel 正沿三个彼此不同的方向成熟：

1. **MoE 层级的分布式算子 MegaKernel**：TensorRT-LLM CuTeDSL MegaMoE 与 Meta Dist-MoE 已经给出真实、可读、可调用的开源实现；MegaFlux 与 Weave 则把研究前沿推进到动态专家复制、路由偏斜和细粒度 SM 调度，但两者截至 2026-10-05 仍未公开作者代码。
2. **整模型/整解码/整训练程序**：Cohere、Inferact TPU MegaKernels 和 training-megakernel 把单次常驻程序的边界扩展到完整 decode 或完整 training step，已经不是传统“算子库”的范畴。
3. **自动生成与编译器**：ForgeMegakernel、TileMega 分别探索智能体生成和编译器生成；Forge 仅论文，TileMega 有公开源码但无明确许可证。

新增或需要更新的重点如下。

| 项目 | 时间证据 | 源码状态（截至 2026-10-05） | 严格分类 | 与当前归档关系 |
|---|---|---|---|---|
| [Meta Dist-MoE](https://github.com/meta-pytorch/dist_moe) | 首个实质提交 2026-10-03 UTC | BSD-3-Clause，完整源码 | 分布式 MoE 算子库；Mega 路径是严格 MoE-core megakernel | 新增候选 |
| [TensorRT-LLM CuTeDSL MegaMoE](https://github.com/NVIDIA/TensorRT-LLM/tree/b5011b14415d45c2f0cfeb31cf50e2ab68377867/tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe) | 对应提交 2026-09-18 | Apache-2.0，完整源码 | 严格分布式 MoE forward megakernel | 本轮新增归档；与 FlashInfer CuTeDSL MegaMoE 属同一技术谱系，不应写成完全独立的新算法 |
| [Weave](https://arxiv.org/abs/2609.21483) | arXiv v1 2026-09-18 | 论文称录用后开源；当前无作者仓库 | 严格分布式 MoE megakernel，动态 SM 调度 | 新论文，无代码 |
| [MegaFlux](https://arxiv.org/abs/2610.00671) | arXiv v1 2026-09-30 | 无作者代码；仅有第三方讲解仓库 | 严格 MoE forward/backward megakernel，动态专家复制 | 新论文，无代码；上游 TensorRT-LLM forward 可用 |
| [ForgeMegakernel](https://arxiv.org/abs/2609.12379) | arXiv v1 2026-09-11 | 无作者代码 | 生成式整模型 decode megakernel 框架 | 新论文，无代码 |
| [mKernel](https://github.com/uccl-project/mKernel) | 论文 2026-09-11；代码 9/15、10/1 有实质更新 | 已开源 | 跨节点 persistent 计算—通信融合运行时 | 已归档，更新结论 |
| [Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens) | 论文 2026-09-28 | 代码早于本轮；窗口内主要为文档更新 | 分布式 MoE training megakernel | 已归档，新论文而非新源码 |
| MonoMoE / FlashInfer | 论文实为 2026-08-19；代码更早 | 已在本地 FlashInfer 子树 | 单 GPU MoE 算子 megakernel | 已归档但 CATALOG/报告应单列命名 |
| [training-megakernel](https://github.com/kiddyboots216/training-megakernel) | 首个实质发布提交 2026-09-26 | MIT，完整源码 | 整训练程序 persistent megakernel | 新增候选 |
| [Inferact TPU MegaKernels](https://github.com/Inferact/tpu-megakernels) | 首个实现 2026-09-23 | Apache-2.0，完整源码 | 跨 TPU、整模型 decode megakernel | 新增候选 |
| [Cohere MegaKernel](https://github.com/cohere-ai/cohere-megakernel) | 初始提交 2026-09-04；公告 9/8 | Apache-2.0，完整源码 | 单 H100、整模型 decode megakernel serving engine | 本轮补录 |
| [TileMega](https://github.com/pengjh0111/TileMega) | 初始源码 2026-08-31 | 源码公开，但仓库未见许可证 | MegaKernel 编译器/代码生成器 | 本轮补录，不能称为标准意义 OSS |

## 二、分类口径：哪些算“严格 MegaKernel”

本报告采用以下边界，避免把所有融合算子都混在一起：

- **严格 MegaKernel**：一次持久化或超大 kernel launch 内包含多个原本独立的模型阶段，并在设备侧完成跨阶段调度、依赖或资源复用。它可以是单层 MoE，也可以是完整 decode/training 程序。
- **Persistent kernel / task runtime**：线程块或工作组长期驻留，通过任务队列、指令流或依赖计数器循环取活。它是 MegaKernel 的常见执行机制，但一个仅执行单一算子的 persistent kernel 不自动等于模型 MegaKernel。
- **Kernel fusion**：若只是 epilogue 融合、单一 GEMM+activation 或固定短链路融合，仍是融合算子，不应上升为 MegaKernel 平台。
- **Graph compiler / CUDA Graph**：主要减少 host launch 或生成多个 kernel；除非最终产物确实是一个跨算子常驻 kernel，否则属于相邻技术。
- **通信基础库**：只提供 NVSHMEM/NVLink/RDMA primitives 或 collectives，而没有把通信与专家计算封装进同一持久化程序，不算 MoE MegaKernel。

因此，Dist-MoE 的完整项目是“分布式 MoE 算子库”，其中 block-scaled Mega 路径才是严格 megakernel；TensorRT-LLM CuTeDSL MegaMoE 是严格 MoE forward megakernel；Weave 和 MegaFlux 是严格研究型 MoE megakernel；mKernel 是更通用的 persistent 计算—通信融合运行时。

## 三、Meta Dist-MoE：本轮最完整的新分布式 MoE 算子库

### 一手来源与开放状态

- 仓库：[meta-pytorch/dist_moe](https://github.com/meta-pytorch/dist_moe)
- 许可证：BSD-3-Clause。
- 仓库创建于 2026-09-18，但首个实质源码提交为 [ab56bced262e60b2bff70dde5cef214d24da4793](https://github.com/meta-pytorch/dist_moe/commit/ab56bced262e60b2bff70dde5cef214d24da4793)，author date 为 2026-10-02 19:06:51 -0700，即 2026-10-03 UTC。不能用仓库 pushed_at 代替这个时间。
- 面向 NVIDIA SM100+；README 明示 BF16、MXFP8、NVFP4，多卡 expert parallel，支持 forward/backward、训练/推理的不同组合。

### 真实融合边界

公共调用接受输入激活、调用方已经算好的 top-k expert IDs 和 scores、W13/W2 权重及可复用 context。也就是说，router logits 和 top-k 选择不在库内。

- BF16 路径是多个通信/计算 kernel 的流水，不应称为单 launch MegaKernel。
- block-scaled staged 路径采用两个分布式主 kernel。
- block-scaled Mega forward 的核心路径把 W13、SwiGLU/量化、W2、跨 peer combine 的主体合入一个 chunked topology。
- backward 提供成对融合的 DGRAD/WGRAD 路径，但整个训练 MoE 公共调用仍不是字面意义的单 launch。
- metadata 发布、部分 barrier/规划与最终 postprocess 会以周边 launch 存在。

因此最准确的表述是：**Dist-MoE 是完整的分布式 MoE operator library，其中 MXFP8/NVFP4 Mega pipeline 实现了严格的 MoE-core megakernel；它不是整层 router-to-output 的单 launch，也不是整模型 MegaKernel。**

### 关键源码路径

- dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm.py
- dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm_kernel.py
- dist_moe/kernels/mega_blockscaled_grouped_gemm.py
- dist_moe/kernels/mega_blockscaled_grouped_gemm_kernel.py
- dist_moe/kernels/dist_blockscaled_grouped_gemm.py
- dist_moe/kernels/dist_blockscaled_grouped_gemm_kernel.py
- dist_moe/kernels/triton/dist_dispatch_routing_kernel.py
- dist_moe/_blockscaled_ops.py
- docs/mxfp8_execution.md
- docs/mxfp8_kernel_design.md
- docs/bf16_execution.md

### 技术取舍

- 优点：同时覆盖 forward/backward、训练/推理、低精度和 graph-stable PyTorch integration；设备侧 planning 与对称通信缓冲使动态路由仍可被 CUDA Graph 捕获。
- 代价：固定 physical token shape、SM100+、强依赖 CUTLASS DSL/Triton/CUDA 版本；context 非重入；router/top-k 不包含在融合边界内。

## 四、TensorRT-LLM CuTeDSL MegaMoE：MegaFlux 的真实开源上游

### 一手来源与开放状态

- 官方快照：[NVIDIA/TensorRT-LLM@b5011b1 / cutedsl_megamoe](https://github.com/NVIDIA/TensorRT-LLM/tree/b5011b14415d45c2f0cfeb31cf50e2ab68377867/tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe)
- 对应实质提交：[b5011b14415d45c2f0cfeb31cf50e2ab68377867](https://github.com/NVIDIA/TensorRT-LLM/commit/b5011b14415d45c2f0cfeb31cf50e2ab68377867)，2026-09-18，提交说明为更新 Blackwell/Rubin MegaMoE kernels。
- TensorRT-LLM 根仓采用 Apache-2.0；该子树提供可读 CuTe DSL 源码，不是二进制 wrapper。
- 本地已按稀疏方式归档到 archive/distributed-moe/TensorRT-LLM-MegaMoE，保留 .git；sparse-checkout 唯一目标为 tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe，未下载依赖或完整仓库。
- 当前归档 HEAD 为 [bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60](https://github.com/NVIDIA/TensorRT-LLM/commit/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60)，2026-10-05。该 HEAD 是 nightly lock-file 更新，不能把它当作 MegaMoE 新贡献日期；MegaMoE 的实质时间证据仍以前述 9/18 提交为准。
- MegaMoE 目标子树共 54 个文件、约 1.47 MiB；整个稀疏工作树 140 个文件、约 7.99 MiB，含 .git 约 12.87 MiB。

### 真实融合边界

源码对核心 kernel 的描述是组合 pull dispatch、persistent FC12、token-back 和 reduction。调用方仍提供 top-k indices/scores，metadata-push routing 是单独 launch；核心持久 kernel 负责：

1. 从 peer 拉取被路由 token；
2. 执行 FC1、gated activation、FC2；
3. 把结果返回 token owner；
4. 在启用 reduce_topk_in_kernel 时完成 kernel 内 top-k reduction。

它应归为**严格的分布式 MoE forward operator megakernel**，但不是整模型 kernel，也不包含 router logits/top-k selection。

### 关键源码路径

- tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/api.py
- tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py
- .../kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop.py
- .../kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_epilogue.py
- .../kernel_src/blackwell/inference/mega/block_scaled_swap_ab_fc12_extension.py
- .../kernel_src/blackwell/inference/mega/dynamic_mainloop.py
- .../kernel_src/blackwell/inference/mega/topk_reduce.py
- .../kernel_src/schedulers/fc12_scheduler.py
- .../kernel_src/schedulers/fc12_mapping.py
- .../kernel_src/schedulers/non_clc_mixed_cga.py
- .../kernel_src/schedulers/work_id_claim.py
- .../communication/nvlink_domain/token_comm.py
- .../communication/nvlink_domain/symmetric_buffer.py

当前快照覆盖 Blackwell SM100 与 Rubin SM107：除分布式 NVLink-domain mega 路径外，还包含 Rubin inference/local_mega；量化定义见 cutedsl_megamoe/quant_def.py，代码覆盖 NVFP4、MXFP4、MXFP8 E4M3/E5M2 及混合格式。

### 与 FlashInfer 的关系

当前归档已包含 FlashInfer 自身的 cutedsl_megamoe/相关 MoE 代码。TensorRT-LLM 这一份是 NVIDIA 官方服务框架内的实现，也是 MegaFlux 论文明确扩展的 forward baseline。两者应在谱系图中标为同类 CuTeDSL/CUTLASS DSL 分布式 MegaMoE 路径，而不是宣称两个互不相关的全新算法。

## 五、Weave、MegaFlux、ForgeMegakernel：论文已公开，作者源码未公开

### 5.1 Weave

- 论文：[Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)，v1 于 2026-09-18。
- arXiv 正文明确写明 implementation will be open-sourced upon acceptance；截至 2026-10-05 未检索到作者官方仓库。
- 严格范围：dispatch、GEMM0、SiLU、GEMM1、combine 被组织进 persistent megakernel。
- 调度：
  - spatial scheduler 在 routing 结果可知后，用 kernel 内轻量 cost model 为每层、每 GPU 决定 communication SM 与 compute SM 的比例；
  - temporal scheduler 对 chunk 执行次序进行调度；
  - bubble stealing 允许暂时空闲的 communication workers 接管可执行 GEMM tiles。
- 与静态切分 persistent runtime 的主要差别：Weave 不把通信/计算 SM 比例固定在 host 配置或离线搜索结果中，而是在当前路由负载已知后设备侧决定。
- 作者报告：4×H100、6 个 MoE 模型，MoE layer 几何平均 2.89×、端到端 1.33×。本轮未复测。

### 5.2 MegaFlux

- 论文：[MegaFlux: Skew-Resilient MoE Megakernels via Pipelined Expert Replication](https://arxiv.org/abs/2610.00671)，v1 于 2026-09-30。
- 论文明确说明 forward 基于 TensorRT-LLM CuTeDSL MegaMoE，并新增 backward megakernel。
- 截至 2026-10-05，正文没有代码/项目链接或开源承诺；按题名、作者和项目名搜索未发现官方仓库。
- [Gin-Sin/megaflux-explained](https://github.com/Gin-Sin/megaflux-explained) 创建于 2026-10-02，是教学可视化/讲解与第三方复现入口，不是论文作者正式源码。
- 关键机制：
  - on-device planner 在每 GPU replica budget 下选择 hot expert replicas，并按 tile 对齐的 token blocks 分配；
  - forward 将 replica weight transfer 与 FC1/FC2 执行流水化；
  - backward 将 weight transfer、DGrad/WGrad/requantization 与 replica gradient reduction 交叠；
  - router 输出本身保持不变，metadata routing 与部分 final sum 仍是周边 launch。
- 作者报告：8×B200，147 个配置/方向，forward 几何平均 1.45×、backward 1.28×；集成 vLLM 的 DeepSeek-V4-Pro prefill 为 1.13–1.26×。本轮未复测。
- 开源判断：**MegaFlux 改动和 backward 当前仅论文；可直接使用的只是其上游 TensorRT-LLM forward baseline。**

### 5.3 ForgeMegakernel

- 论文：[ForgeMegakernel: A General Framework for Efficient Auto-Regressive Model Decode Megakernels](https://arxiv.org/abs/2609.12379)，v1 于 2026-09-11。
- 截至 2026-10-05，arXiv 页面和正文未给作者仓库或 artifact 链接，GitHub 精确题名/作者搜索未发现官方代码。
- [kamahori/vibe-megakernel](https://github.com/kamahori/vibe-megakernel) 中的 reproductions/ForgeMegakernel 是第三方复现，不是作者正式仓库。
- 生成流程：十个通用 milestones 约束每 SM 指令流、依赖计数器替代 grid barrier、共享内存 buffer pool 等结构；独立 mid-state oracle 检查中间状态；coding agents 逐步生成模型特化 decode megakernel。
- 最终产物语义属于严格整模型 decode megakernel，但框架本身仍是论文描述，当前不能当作可归档的开源编译器。
- 作者报告：14 个 decode cells、8 个模型家族、0.6B–13B、H100；50.5–85.9% MBU，几何平均相对 SGLang 1.21×、相对 MPK 1.54×。本轮未复测。

## 六、旧项目的实质更新：mKernel 与 Mixture-of-Kittens

### 6.1 mKernel

- 仓库：[uccl-project/mKernel](https://github.com/uccl-project/mKernel)，已在 2026-08-26 归档。
- 新论文：[mKernel: Enabling Compute-Communication Overlap in Distributed DNN Training and Inference](https://arxiv.org/abs/2609.13585)，v1 于 2026-09-11。
- 论文把项目定位为跨节点 persistent execution：SM 被划分为 compute/communication workers；GPU 内 controller 根据 shape/kernel 调整比例；network commands 经轻量 command queue 交给 host proxy/raw RDMA verbs；同一路径覆盖 InfiniBand 与 AWS EFA。
- 公开代码包含 AG+GEMM、GEMM+AR、MoE dispatch+GEMM、MoE dispatch+FFN+combine、Ring Attention、GEMM+RS，通信路径包括 include/comm/internode/session.h 与 session_efa.h。
- 本轮窗口内有两项实质更新，而不只是 pushed_at：
  - [888517d5a68e10e326a032ce3ae1931bb4ac68ef](https://github.com/uccl-project/mKernel/commit/888517d5a68e10e326a032ce3ae1931bb4ac68ef)，2026-09-15：AG-GEMM dispatch 泛化到更多 N shape；
  - [97e916b3a731689e4ea98db60bfe5f8adcaf685a](https://github.com/uccl-project/mKernel/commit/97e916b3a731689e4ea98db60bfe5f8adcaf685a)，2026-10-01：AG-GEMM 增加 CUDA Graph 支持。
- 结论：归档工件无需重复新增，但总报告应补论文、EFA/IB 范围和 9/15、10/1 功能更新。

### 6.2 Mixture-of-Kittens

- 仓库：[cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens)，已归档。
- 新论文：[Mixture-of-Kittens](https://arxiv.org/abs/2609.36070)，v1 于 2026-09-28。
- 论文描述单个 deterministic training megakernel 融合 dispatch、shared/routed FFN、combine，并覆盖 forward/backward；在 GB200/GB300 NVL72 上按 operator 选择 push/pull 并重组 overlap。
- 代码历史显示本轮窗口内主要为 9/29、10/1 README/文档更新，最近的功能代码变更仍在 8/14。应记为“新增论文与叙述证据”，不能把它计作本轮新源码工件。

## 七、MonoMoE：不是本轮新增，而是归档命名缺口

- 论文：[MonoMoE: A Single-Kernel Mixture-of-Experts Implementation](https://arxiv.org/abs/2609.04244)。尽管编号为 2609，实际首发时间为 2026-08-19，早于 8/26 基线报告。
- 源码已经包含在本地归档的 FlashInfer 中，无需再次下载：
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/monomoe_binding.cu
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/monomoe_wrapper.cuh
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/src/moe.cuh
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/src/moe_routing.cuh
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/src/moe_up_projection.cuh
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/src/moe_down_projection.cuh
  - archive/distributed-moe/flashinfer/csrc/fused_moe/monomoe/src/moe_tma.cu
- 它采用 weight-major persistent grid，把 routing/top-k、quantization、W13、activation、W2、reduction 放入单 GPU kernel；属于严格单 GPU MoE operator megakernel。
- 正确的增量动作是 CATALOG/总报告单列“MonoMoE（随 FlashInfer 已归档）”，而不是新增仓库计数。

## 八、整模型/整训练方向的新开源工件

### 8.1 training-megakernel

- 仓库：[kiddyboots216/training-megakernel](https://github.com/kiddyboots216/training-megakernel)，MIT。
- 仓库虽创建于 2026-09-01，首个实质版本提交为 [942f86a9fc2b5166acff14818f137b6d1b141848](https://github.com/kiddyboots216/training-megakernel/commit/942f86a9fc2b5166acff14818f137b6d1b141848)，2026-09-26 UTC。
- 一个 cooperative CUfunction/GPU 在多个 optimizer steps 间常驻，覆盖 Qwen3-8B forward、backward、分布式梯度工作、global-norm clipping、AdamW、token refill/checkpoint/telemetry loop。
- 关键路径：
  - kernel/program/training_program.py
  - kernel/program/resident_step.py
  - kernel/program/clipped_adamw.py
  - kernel/program/tile_schedulers.py
  - kernel/program/communication/decoder_fabric.py
  - kernel/program/communication/embedding_route.py
  - kernel/program/communication/symmetric_memory.py
  - kernel/tools/patch_step_backedge.py
- 分类：严格的整训练/application persistent megakernel，不是通用 operator library。
- 重要取舍：README 的匹配算子对比中，常驻 kernel 与 staged CUDA Graph 的差距约 1%，说明主要价值是设备常驻程序形态和控制边界，而不应把所有相对 Megatron 的收益都解释为“单 kernel 本身”。

### 8.2 Inferact TPU MegaKernels

- 仓库：[Inferact/tpu-megakernels](https://github.com/Inferact/tpu-megakernels)，Apache-2.0。
- 官方博客：[700 TPS on Kimi K3: A Case for TPU Megakernels](https://inferact.ai/blog/tpu-megakernels)，2026-09-23。
- 首个实现提交 [aa0094ef9add6a1f21b1697fc7371ffccf68e8ea](https://github.com/Inferact/tpu-megakernels/commit/aa0094ef9add6a1f21b1697fc7371ffccf68e8ea)，2026-09-23。
- Kimi K3 路径在一个 Pallas call 中覆盖 92 个 MoE layers、KDA/MLA/MoE/residual/norm/collectives，面向 16 个 TPU v7 chips（32 TensorCores、4 hosts）；Qwen3.8-27B 路径覆盖 GDN/GQA dense decode，可叠加 DFlash2。
- 关键路径：
  - kimi/decode_megakernel.py
  - kimi/dspark.py
  - kimi/load.py
  - qwen/decode_megakernel.py
  - qwen/dflash.py
  - qwen/load.py
  - collectives32.py
  - pool_alias.py
- 技术特点是 grid-less Pallas program、scoped VMEM、HBM→VMEM async DMA、跨层 weight prefetch 和拓扑特化 collectives。属于严格分布式整模型 decode megakernel。

### 8.3 Cohere MegaKernel

- 仓库：[cohere-ai/cohere-megakernel](https://github.com/cohere-ai/cohere-megakernel)，Apache-2.0。
- 初始提交 [cfb29771a7815197d1ffdfef30e13af85b18da81](https://github.com/cohere-ai/cohere-megakernel/commit/cfb29771a7815197d1ffdfef30e13af85b18da81)，2026-09-04；官方博客 [Inside Cohere's MegaKernel](https://cohere.com/blog/megakernels)，2026-09-08。
- 针对 North Mini Code、单 H100、batch 1–8；每 SM 一个常驻 block，host 生成静态 per-SM task list 和依赖计数器，attention/MoE 使用动态队列，统一 warp roles/calling convention。
- 核心路径 src/decode/megakernel.cuh；服务入口 src/serving/server.py。
- 完整 decode 属严格 megakernel；prefill 仍是独立 PyTorch kernels，并会暂停 decode，因此不是完整请求生命周期单 kernel。

## 九、编译器与较低成熟度工件

### 9.1 TileMega

- 仓库：[pengjh0111/TileMega](https://github.com/pengjh0111/TileMega)。
- 首个源码提交 2026-08-31；截至 2026-10-05 仓库未见 LICENSE，GitHub license metadata 也为空，所以只能写“公开源码”，不能写“开源许可证完备”。
- 以 Parameterized Coupling Graph 表达依赖，结合 ISL/barvinok 符号分析、MLIR dialect、hardware-aware search 和 CuTe/CUTLASS codegen，把 torch.export 图降为 persistent megakernel/shared object。
- 主要实现目录：include/tilemega/{Analysis,Backend,Codegen,Dialect,Frontend,Runtime,Solver,Support,Target}、lib、tools、python/tilemega。
- 无论文或独立复现实证；应列入实验性编译器，而不是与成熟运行时并列。

### 9.2 Lean CUDA Qwen

- 仓库：[ranvier-labs/lean-cuda-qwen](https://github.com/ranvier-labs/lean-cuda-qwen)，Apache-2.0，2026-09-09 创建。
- 公开 Lean 源码中包含 Qwen3.6/Qwen3.8 persistent decode/training experiments：
  - lib/LeanCudaQwen/Qwen36/Megakernel.lean
  - lib/LeanCudaQwen/Qwen36/FullGRPOMegakernel.lean
  - lib/LeanCudaQwen/Qwen36/FullPretrainMegakernel.lean
  - lib/LeanCudaQwen/Qwen36/FullTrainingMegakernelCore.lean
  - examples/qwen36_megakernel/
  - examples/qwen38_chat/Worker.lean
- 但公开说明在“完整 chat/training”与“仍非完整 autoregressive/chat/training”之间存在示例层面的差异，且 CUDA-enabled Lean compiler 的完整可获得性受限。适合列为 source-available/低成熟度实验，不宜计入成熟库。

### 9.3 hoid-megakernel-qwen

- 仓库：[hoid-ai/hoid-megakernel-qwen](https://github.com/hoid-ai/hoid-megakernel-qwen)，2026-10-03 创建。
- Python host/runtime 与 vLLM plugin 可读，但真实 device kernels 位于 src/vllm_qwen3_megakernel/cubins/b1、b2、b4、b8，仅发布编译 cubins/index，没有 CUDA/CuTe 核心源码。
- 应标记为“wrapper/runtime source + binary kernel distribution”，而不是完整开源 MegaKernel。

## 十、边界项目：不要误计为新 MegaKernel 算法

| 项目 | 正确归类 | 原因 |
|---|---|---|
| [msaroufim/megakernels-vs-cuda-graphs](https://github.com/msaroufim/megakernels-vs-cuda-graphs) | 对比/复现 harness | vendored HazyResearch MegaKernel，并从 DeepSeek DeepSpec 衍生 DSpark；不是新的核心实现谱系 |
| [jiazhihao/mpk-apple](https://github.com/jiazhihao/mpk-apple) | Metal 多 dispatch 静态设备程序 | README 明确是 self-advancing chain of bounded whole-GPU dispatches，而非单个 never-returning kernel |
| [Gin-Sin/megaflux-explained](https://github.com/Gin-Sin/megaflux-explained) | 教学可视化/第三方解释 | 非 MegaFlux 作者正式实现 |
| [kamahori/vibe-megakernel](https://github.com/kamahori/vibe-megakernel) | 第三方复现集合 | reproductions/ForgeMegakernel 不是 Forge 作者仓库 |
| Beomi/husky-megakernel | 实验性 Metal 单 dispatch 分支 | 多 threadgroup 正确性/性能尚未解决，正常路径不是该 megakernel |
| dazuozcy/megakernel | 教程/外部代码索引 | 主要指向外部 TileFoundry Ascend 代码，不是原创算子库 |

## 十一、横向技术图景

### 调度机制的差异

- **TensorRT-LLM MegaMoE**：面向 Blackwell/Rubin 的固定框架内 persistent FC12 和通信调度；router/top-k 在外，核心负责 pull dispatch—expert compute—token-back—可选 reduction。
- **Dist-MoE**：以完整 operator library 和可复用 context 为中心；不同精度选择 BF16/staged/Mega，强调 training/backward、图捕获和内存规划，不追求所有公共调用都缩成单 launch。
- **Weave**：重点不在新 GEMM，而在路由结果已知后的设备内动态空间/时间调度；通信与计算 worker 的比例逐层、逐 GPU 改变。
- **MegaFlux**：重点不在 SM 比例，而在路由偏斜导致的 hot expert；运行时复制专家，并把额外权重传输/梯度归并藏进 forward/backward persistent pipeline。
- **mKernel**：更底层、更通用，提供跨 IB/EFA 的 compute-communication persistent runtime；MoE 只是多种融合 pattern 之一。
- **MonoMoE**：单 GPU、weight-major persistent grid；解决单卡 MoE 小批量低利用率，不处理跨 GPU EP 通信。
- **整模型项目**：Cohere、Inferact、training-megakernel 通过 per-SM task lists、Pallas 程序或常驻 training loop 跨越整个模型/step，代价是模型和硬件特化更强。

### 开源成熟度分层

1. **可直接审阅核心源码**：Dist-MoE、TensorRT-LLM CuTeDSL MegaMoE、mKernel、Mixture-of-Kittens、FlashInfer MonoMoE、Cohere、Inferact、training-megakernel。
2. **论文有明确架构、当前无作者代码**：Weave、MegaFlux、ForgeMegakernel。
3. **公开源码但许可证或工具链/证据不完整**：TileMega、Lean CUDA Qwen。
4. **wrapper/二进制/教学复现**：hoid-megakernel-qwen、megaflux-explained、vibe-megakernel、megakernels-vs-cuda-graphs。

## 十二、对 2026-08-26 总图景的修订建议

1. 把“MoE megakernel 多为论文原型”的描述改为：**至少 TensorRT-LLM、FlashInfer、Meta Dist-MoE、mKernel、Mixture-of-Kittens 已提供可读核心源码；Weave/MegaFlux 代表仍未公开代码的下一代调度研究。**
2. 在归档中把 TensorRT-LLM CuTeDSL MegaMoE 作为本轮新工件，但在谱系图里连接 FlashInfer/CuTeDSL MegaMoE，避免把框架集成版写成完全独立算法。
3. 给 MonoMoE 单列目录索引/报告条目，并注明“随 FlashInfer 已归档”；不要重复下载或重复计数。
4. 将 Dist-MoE 同时归为“分布式 MoE operator library”和“含严格 Mega pipeline”，不要笼统宣称整个库每种模式都是单 kernel。
5. 把 Weave、MegaFlux、ForgeMegakernel 明确标注 paper-only；第三方讲解/复现仓库不能替代作者代码。
6. 把 whole-model decode、whole-training persistent program 与通用 operator library 分栏，否则 Cohere/Inferact/training-megakernel 会与 Dist-MoE/MonoMoE 产生误导性横比。

本轮研究仅整理一手来源、时间证据、源码边界和实现路径；未下载模型权重、未安装依赖，也未进行硬件性能复现。
