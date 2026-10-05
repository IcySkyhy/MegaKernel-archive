# MegaKernel 第二轮增量调研 · 2026-10-05

本轮重点窗口为 **2026-09-05 至 2026-10-05**，并补检第一轮 2026-08-26 之后的遗漏。9 月 24 日只是本地成果上传 GitHub 的日期，不能作为第一轮源码调研的截止日。

**结论：有一批值得增补的项目。真正重要的变化是整模推理开始接入服务框架、MoE 大核开始形成训练/推理算子 API、常驻范围扩展到完整训练循环，以及 TPU、Apple GPU、Ascend 的实现继续分化。尚不能据此认定已经出现统一、成熟、跨硬件的 MegaKernel 算子库。**

本轮实际归档 **22 个新增工件，总数由107增至129**：21个实现/相邻路线工件，1个agent方法参考。完整目录见[总索引第12节](CATALOG.md#12-第二轮增补222026-10-05)，机器可读清单见 [artifacts.json](research-rounds/2026-10-05/artifacts.json)。

## 1. 检索口径与可复查材料

- 9 组 GitHub repository API 查询得到 **98 个去重候选**；每组结果均不超过 100 项，响应未标记搜索结果截断。查询覆盖名称/描述、README、created 与 pushed 时间，以及 persistent、MoE、whole-model 等邻近词。
- 另检索 GitCode、arXiv、作者项目页和已有上游提交。98 是 API 候选集规模，不是 98 个有效 MegaKernel，也不是穷尽所有托管平台后的项目数。
- 时间分为仓库创建、首次实质代码发布、近期主线提交、论文首次提交。`pushed_at` 仅用于发现候选，不能证明本月有新的实现。
- [查询与元数据](research-rounds/2026-10-05/github-search-results.json)、[逐仓 README/文件树/近期提交](research-rounds/2026-10-05/candidate-evidence/)、[源码下载记录](research-rounds/2026-10-05/downloaded-snapshots.json)保存在本地。
- 本轮通过公开源码与执行入口判断覆盖范围，没有运行性能测试，也未连接 B300。下文性能讨论均是上游报告，不能视为本次实测。没有进行安全、哈希或边界条件审计。

第一批 18 个 GitHub 源码快照中，12 个仓库创建于重点窗口，5 个创建于 8 月 27 日至 9 月 4 日，1 个属于更早遗漏。这个划分不等于发布日期：例如 Cohere 创建于 9 月 4 日、9 月 8 日发布项目文章；training-megakernel 创建于 9 月 1 日，而首次实质代码提交在 9 月 26 日。

## 2. 最值得优先阅读的新增项目

| 项目 | 时间证据 | 新增价值 | 实际范围与限制 |
|---|---|---|---|
| [Cohere Megakernel](https://github.com/cohere-ai/cohere-megakernel) | 创建 09-04；[作者文章](https://cohere.com/blog/megakernels) 09-08 | 单 H100 上将任务图大核与 continuous batching、paged KV、prefix caching、preemption 和 HTTP 服务结合 | North Mini Code 专用，BF16、batch 1–8；decode 主计算图大核，embedding和首个RMSNorm在外；prefill分离且会暂停decode。Apache-2.0 |
| [Inferact TPU Megakernels](https://github.com/Inferact/tpu-megakernels) | 创建/首发 09-23 | TPU Pallas 的跨层 Kimi K3、Qwen3.8-27B 实现，并有 DSpark/DFlash2 相关路径 | Kimi 主核覆盖 decoder stack 与 collective，但 embedding/LM head 在外；Qwen 主核可含 embedding、全层、LM head 与 argmax。Apache-2.0 |
| [Dist-MoE](https://github.com/meta-pytorch/dist_moe) | 创建 09-18；首次实质提交 10-03 | Meta/PyTorch 组织发布可复用分布式 MoE API，包含训练、量化、context/activation 管理、staged/Mega 选择 | SM100+；Mega 是核心计算通信融合路径，不是一次 public call 的全部工作都在一个 launch。BSD-3-Clause |
| [Training Megakernel](https://github.com/kiddyboots216/training-megakernel) | 创建 09-01；首次实质提交 09-26 | 每 GPU 一个 cooperative CUfunction 跨多个 optimizer step 常驻，涵盖 forward、backward、分布式梯度、clip 与 AdamW | 8×H100、Qwen3-8B 宽度及受限 shape；源码构建涉及多个算子编译产物的组合。MIT |
| [TileMega](https://github.com/pengjh0111/TileMega) | 创建 08-28；首次实质提交 08-31；本月持续开发 | ISL/barvinok 符号依赖分析、MLIR Coupling Graph、硬件搜索和 CuTe/CUTLASS 代码生成；补充 MPK 之外的编译器路线 | 源码可读，根许可证未声明；依赖较重、缺少独立复现。归档排除大型 `docs/experiments` 原始结果，保留编译器源码与主要文档 |
| [Ascend DeepEP / MegaMoE](https://gitcode.com/Ascend/DeepEP) | 默认分支源码快照 09-30；含近期 MegaMoE 集成 | 官方 Ascend 仓库新增 W4A8 MegaMoE：dispatch、两次 GMM、激活/量化、combine | Ascend950/CANN9.2；根 LICENSE 为 BSD-2-Clause，外来代码保留各自条款；仍是实验性路径，与 DeepSeek DeepEP 分开计工件 |
| [TensorRT-LLM CuTeDSL MegaMoE](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe) | 09-18 提交更新 Blackwell/Rubin MegaMoE kernels | NVIDIA 官方前向大核、FC12 persistent scheduler 与 NVLink 通信原语 | Apache-2.0；接收已有 top-k，metadata push 在外围；核心融合 pull dispatch、FC1/激活/FC2、token-back，可选核内 top-k reduce。与 FlashInfer 相关后端有谱系关系 |

### Cohere：服务能力的增量大于又一个 batch-1 demo

`src/decode/megakernel.cuh` 与 `schedule.py` 将算子拆成 tile/task，SM 读取任务列表，通过依赖计数器推进。原生 C++ decode 服务循环与 Python 请求管理、prefill 配合。这表明 whole-model megakernel 与动态请求服务可以结合，但模型、精度和 batch 范围仍很窄。

原有结论“动态 serving 是短板”需要收窄为：**已经出现开源的动态服务实现，通用多模型、混合 prefill/decode 和跨硬件能力仍未形成。**

### TPU：同一仓库中的不同模型也要分别描述

关键文件为 `kimi/decode_megakernel.py`、`kimi/dspark.py`、`qwen/decode_megakernel.py`、`qwen/dflash.py`、`collectives32.py`。Pallas 的显式 VMEM、异步搬运与跨层循环构成其技术重点。

Kimi 部署文档里的 **32 TPU devices/TensorCores 对应 16 TPU v7 chips**，不能把二者当作不同规模结果。作者的 709/1515 tok/s 标题包含 speculative decoding 和指定 acceptance length，不能直接当作不带 speculation 的常规 decode 速度。[作者说明](https://inferact.ai/blog/tpu-megakernels)

### Dist-MoE：最接近“算子库”的新项目，但 API 不等于单 launch

调用方提供 top-k expert IDs/scores。BF16 使用多 kernel 流水线；MXFP8 和 NVFP4 可选择 staged/Mega，NVFP4 当前只支持推理。

`dist_moe/kernels/chunked_mega_blockscaled_grouped_gemm*.py` 的 Mega forward 融合 W13、SwiGLU/量化、W2 和 peer combine stores；routing metadata、publication 与最终 top-k sum 仍有外围 launch。Mega backward 也融合多个梯度阶段，而非把全部训练层工作压成一个 kernel。应按“可复用算子 API + 核心 MegaKernel 后端”理解它。

### 训练常驻：跨度增大，收益仍受基线影响

`training-megakernel/kernel/program/training_program.py`、`resident_step.py`、`clipped_adamw.py` 与 `communication/` 展示了跨 optimizer steps 常驻。host 仍负责输入槽补充、checkpoint 落盘和遥测。

仓库报告相对 Megatron 的收益更大，但使用相同算子组成的 staged CUDA Graph 基线，step time 约为 4141 ms 对 4092 ms，差距约 1%。这项工作的研究价值是执行范围与构建方法，不能把不同基线的收益合并成“大核天然更快”。[源码与作者说明](https://github.com/kiddyboots216/training-megakernel)

## 3. 其他已归档新增项目

| 本地目录 | 上游与时间 | 归类与关键证据 |
|---|---|---|
| [megakernel-gen](compiler-runtimes/megakernel-gen) | [WilliamZhang20/megakernel-gen](https://github.com/WilliamZhang20/megakernel-gen)，09-03 | Apache-2.0；Rust `mkc/src/{hf,plan,codegen,emit}.rs` 将 HF checkpoint/硬件配置转成 cooperative CUDA forward、LM head、sampler；阶段间 grid barrier，单 GPU/batch-1。早于重点窗口，属于补漏 |
| [latticemk](core-whole-model/latticemk) | [thebasedcapital/latticemk](https://github.com/thebasedcapital/latticemk)，10-02 | MIT；Turing sm75/Quadro RTX4000 上的 Qwen3 INT4 decode。保留了压缩、occupancy、同步、KV 与 speculation 的实验记录；不要用最初 GEMV 阶段或 CUDA Graph 的数据代替最终常驻 decode 路径 |
| [lean-cuda-qwen](core-whole-model/lean-cuda-qwen) | [ranvier-labs/lean-cuda-qwen](https://github.com/ranvier-labs/lean-cuda-qwen)，09-09 | Apache-2.0覆盖应用源码，含 `examples/qwen36_megakernel`、`qwen38_chat`、`qwen38_train`。所需 staged Lean CUDA compiler 提供nightly二进制，但完整backend源码需要companion仓库权限；不算完整开放工具链。SFT/GRPO与DPO Graph边界不同 |
| [mlx-lm-unified](core-whole-model/mlx-lm-unified) | [pierre427/mlx-lm-unified](https://github.com/pierre427/mlx-lm-unified)，09-10 | MIT；`mlx_lm/models/qwen4_megakernel_{body,runtime,schedule}.py` 提供可选单 dispatch Metal decode。embedding/PLE lookup 在 launch 前，层与 LM head 在内；默认关闭、依赖设备校准，不等于上游 MLX-LM 已正式支持 |
| [transformer-megakernels](operator-scale/transformer-megakernels) | [Parth-Badgujar/transformer-megakernels](https://github.com/Parth-Badgujar/transformer-megakernels)，创建06-03、元数据推送09-06 | MIT；CuTe DSL 的多层 Transformer stack、SM120，`megakernel.py`、`scheduler.py`。未检出窗口内主分支实质提交，是更早项目漏收，不是九月新项目，也不是完整生成服务 |
| [blackwell-fp4-ffn](operator-scale/blackwell-fp4-ffn) | [theProgrammingBox/blackwell-fp4-ffn](https://github.com/theProgrammingBox/blackwell-fp4-ffn)，09-01 | 根 LICENSE 为 MIT，CUTLASS 派生部分 BSD-3；GitHub 自动字段 NOASSERTION 不代表没有许可证。`fp4_fused_chain.cu` 等研究双 GEMM persistent fusion；多层可部署路径含额外 requant launch，不应写成整模单核 |
| [persist-decode](operator-scale/persist-decode) | [anishesg/persist-decode](https://github.com/anishesg/persist-decode)，09-01 | 未声明根许可证；单 CTA 的共享内存层融合，另有 `src/persistent_multi_layer.cu` 的跨层循环；输出 hidden state，非完整 sampling/token loop。仅据代码收录，不背书 README 中的性能推导 |
| [husky-megakernel](long-tail-experimental/husky-megakernel) | [Beomi/husky-megakernel](https://github.com/Beomi/husky-megakernel)，09-23 | 无根许可证；Swift/Metal、Woof4B。`Sources/husky/megakernel.metal` 是单 dispatch 实验，上游明确仅单 threadgroup 路径正确且明显慢于普通执行；有价值的负结果 |
| [mpk-apple](alternatives/mpk-apple) | [jiazhihao/mpk-apple](https://github.com/jiazhihao/mpk-apple)，09-20 | Apache-2.0；当前名称 lithos-metal。其机制是预编码、自推进的有限 whole-GPU dispatch 链，包含设备侧生成/speculation 控制；属于 MPK 思想在 Metal 的替代执行路线，不计作单一 CUDA 式 persistent launch |
| [megakernels-vs-cuda-graphs](alternatives/megakernels-vs-cuda-graphs) | [msaroufim/megakernels-vs-cuda-graphs](https://github.com/msaroufim/megakernels-vs-cuda-graphs)，09-15 | 无统一根许可；Hazy/DeepSpec 派生子树保留各自许可。Llama 与 DSpark 对比材料，Graph+PDL 与大核的差距约 0.7–4.6%，且数值行为存在差异；适合选择基线，不构成普遍优劣结论 |
| [ascend_mega_kernel](operator-scale/ascend_mega_kernel) | [liruixin_dvc/ascend_mega_kernel](https://gitcode.com/liruixin_dvc/ascend_mega_kernel)，文档探索日期 09-26 | 未声明根许可证；Ascend950PR/CANN9.1；`m13_moe_layer`、`m14_gdn_layer` 在混合 AIC/AIV launch 内同步；当前 `m15_layer_loop` 已有48层 host 循环、逐层四阶段 launch 和末层 mixer，但 decode attention passthrough、QSA/PLE 尚不完整，不是单次常驻全模型 |

## 4. 收录但不能作为成熟核心库计数的项目

| 项目 | 具体判断 |
|---|---|
| [hoid-megakernel-qwen](core-whole-model/hoid-megakernel-qwen) | 创建 10-03。Apache-2.0 的 vLLM 插件与调用代码，但发布包的核心是 `cubins/b{1,2,4,8}/worker.cubin` 和 `execution.bin`。`decoder.py` 的提交序列包括 validator/reset/worker/finalizer，stock embedding、输出处理与 vLLM 保持分工。标 **C-PARTIAL**，不能因仓库有许可证就算完整开源核心 |
| [amandeepsp-megakernels](compiler-runtimes/amandeepsp-megakernels) | 创建 09-12，未声明根许可证。FX/IR 分析原型；`megakernels/backend.py` 最终返回 `make_boxed_func(gm.forward)`，尚无生成设备常驻大核的实现。保留作原型观察，不计成熟 compiler |
| [vibe-megakernel](agentic-kernel-design/vibe-megakernel) | 创建 09-28，未声明根许可证。包含 `megabench/` 和 `reproductions/ForgeMegakernel/`，是第三方评测/复现，不是 ForgeMegakernel 作者官方代码；放入 agent 方法参考组 |
| [xys-syx-megakernel](long-tail-experimental/xys-syx-megakernel) | 创建 09-24，未声明根许可证。LBM/数值计算域的 temporal fusion/cluster 实验。`lbm-megakernel/README.md` 明说主要结果是两时间步融合，1000步仍500次launch；按跨领域相邻工件收录 |

其他未归档候选：`DaTouJun/ForgeKernel` 仅有少量说明/占位文件；`debashis-das/MegaKernel` 为空仓；`ezgamehost/unlimited-ocr-kernel` 当前默认分支是 Rust CPU/NEON OCR；`xuefenghao5121/megakernel-fft` 是 x86 FFTW/AVX2 复合算子 JIT。这些不属于本轮设备常驻 GPU/NPU/TPU 核心增量。

`pierre427/rapid-mlx-decode-lane-deps` 是相同 Metal 实现的依赖片段，明确不含完整 host wiring；本轮已收 `mlx-lm-unified`，不再按独立方案重复下载。`deepseek-ai/DeepSpec` 本体是 draft-model 训练/评测栈，未发现其默认分支发布 MegaKernel 核心；不能因为 DSpark 被别的项目融合，就把 DeepSpec 整仓当作大核库。

## 5. 本月论文有进展，但尚不能变成新增源码仓库

| 工作 | 一手时间 | 贡献与代码状态 |
|---|---|---|
| [ForgeMegakernel](https://arxiv.org/abs/2609.12379) | 09-11 | coding agent、渐进 milestones、独立中间状态 oracle，生成 per-SM instruction-stream decode kernel。截至本轮未找到作者正式实现；第三方 reproduction 单列 |
| [Weave](https://arxiv.org/abs/2609.21483) | v1 09-18，v3 09-25 | 设备内按 routing 结果选择通信/计算 SM 比例和 chunk 数，并允许通信 worker 填补计算空隙。论文写明接受后开源，目前按 paper-only 记录 |
| [MegaFlux](https://arxiv.org/abs/2610.00671) | 首次提交 09-30 | 动态 expert replication、tile 对齐任务分派、权重搬运与 replica-gradient reduction 的流水重叠。截至本轮未找到作者正式仓库；`Gin-Sin/megaflux-explained` 是教学页面，不是实现 |

上述日期以 arXiv 首次提交记录为准；论文编号前缀、发布队列日期和项目文章日期可能不同。对“未找到代码”的结论限定为本轮公开检索，不能推断作者没有内部实现。

## 6. 已有项目的增量：不重复加仓库数

| 既有项目 | 本月证据 | 本轮处理 |
|---|---|---|
| [Mirage](https://github.com/mirage-project/mirage) | 09-12 SM100 temperature/top-k/top-p sampling；09-17 至20 split-launch scheduler packing；09-28 hybrid KV pool 与 SM100 attention token ceiling | 实质工程增量。保留8月本地快照；新增证据目录记录本月主线提交，不把远端更新冒充已全量更新的本地代码 |
| [mKernel](https://arxiv.org/abs/2609.13585) | 09-11 论文，进一步解释 on-GPU controller、计算/通信 SM 分配与 RDMA proxy | 仓库已在第一轮归档，补论文和机制描述 |
| [Mixture-of-Kittens](https://arxiv.org/abs/2609.36070) | 09-28 论文；本窗口主线提交主要为 README | 训练路线的论文证据增加，不声称新实现仓库诞生 |
| [FlashInfer MonoMoE](https://github.com/flashinfer-ai/flashinfer/tree/main/csrc/fused_moe/monomoe) | 论文 [2609.04244](https://arxiv.org/abs/2609.04244) 首次提交实际为08-19；代码更早存在 | 原本地 `csrc/fused_moe/monomoe/` 已有源码。补记 weight-major/persistent 路径，不算本月新代码 |
| [Fleet](https://github.com/ROCm/fleet-chiplet-megakernel) | 09-28 工作流提交 | 不计为算法或 runtime 新能力 |
| [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel)、[Machete](https://github.com/b-albar/machete) | 本轮默认分支按提交日期查询未见窗口内实质提交，虽前者 pushed 时间变化 | 不能仅凭 pushed_at 增加“本月活跃实现”数量；也不代表所有分支/PR 均无活动 |
| [Awesome-MegaKernel](https://github.com/qhy991/Awesome-MegaKernel) | 09-23 添加 Weave | 更新了文献目录，不是新增可执行实现 |

## 7. 对开源图景的修正

1. **服务集成变得更具体。** Cohere 提供连续批处理和 KV 管理；Hoid 展示插件集成，但核心未完整开放；两者应分别计数。
2. **训练方向从单层扩展到循环。** Dist-MoE 将前反向核心做成可复用 API；training-megakernel 将常驻生命周期延伸到 optimizer steps。通用训练框架替代能力尚无充分证据。
3. **硬件差异决定实现形式。** TPU 更容易表达跨层显式存储/DMA；Apple 既有单 dispatch 实验，也有有限 dispatch 链；Ascend 的 Cube/Vector 混合角色形成层级融合。不能只数 persistent 字样来认定同一机制。
4. **编译器与 agent 两条自动化路线继续分化。** TileMega 走符号依赖图，megakernel-gen 走 checkpoint/代价模型驱动的阶段编排，ForgeMegakernel 则是 agent 生成工作流，公开程度也不同。
5. **比较基线已成为主要研究问题。** 新资料中同算子 CUDA Graph/PDL 与常驻核接近的例子增加。减少 launch 本身并不足以预测收益；资源占用、任务依赖、权重流水和模型结构依然决定结果。

接下来若只选择少量源码深读，推荐顺序为 **Cohere → Dist-MoE → TPU Megakernels → Training Megakernel → TileMega**。前四项分别覆盖服务、算子库、异构整模和训练循环，最后一项用于比较编译抽象。若重点是 MegaMoE，则并读 TensorRT-LLM 和 Ascend DeepEP；Hoid、空壳原型和论文待开源项不进入完整源码能力排名。

## 8. 归档方式

新 GitHub 工件以源码快照保存于原有分类目录，下载记录保留上游、branch、时间与方式；不递归下载 submodule、模型权重或运行依赖。原有107个工件保留，新增项目与原项目不会互相覆盖。TileMega 使用排除大规模实验原始输出的稀疏源码归档；TensorRT-LLM 仅保存 MegaMoE 子目录及根说明/许可；GitCode 项目单独获取。网页索引与本地源码不一致时采用本地源码，例如 Ascend DeepEP 的许可证与个人 Ascend 项目的 m15 进度均经此修正。

“归档工件数”包含完整源码、原型、部分开放后端和相邻路线；不能将总数直接称为“成熟开源 MegaKernel 库数量”。各项目开放状态与真实执行范围见 [总索引](CATALOG.md)。
