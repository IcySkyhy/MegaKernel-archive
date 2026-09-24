# MegaKernel 开源图景：从整模型单核到分布式 MoE 设备内运行时

调研快照：2026-08-26  
本地归档：85 个浅克隆仓库/源码工件  
完整逐项索引：[CATALOG.md](CATALOG.md)

## 一、结论先行

截至 2026-08-26，开源 MegaKernel 已经不是少数“把整个小模型手写进一个 CUDA kernel”的孤立实验，而是形成了四层生态：

1. **编译器与设备内运行时**：Mirage/MPK、Machete、Hazy Megakernels、AutoMegaKernel、Triton-distributed/DITRON、Luminal、Fleet；
2. **整模型/整 token 工件**：model-as-a-kernel、AWS NKI Transformer TKG、NKI MoE megakernel、Lucebox、Bonsai、qwen_megakernel、DSV2-236B、flint 等；
3. **分布式与 MoE 大核**：DeepGEMM MegaMoE、FlashInfer MegaMoE、FlashMoE、Mixture-of-Kittens、mKernel、TeraMoE、FastAFD、SGLang NPU/TPU、CANN PTO；
4. **算子级与底座**：TIRx、ThunderKittens、TileLang、NKI、PTO/PyPTO、ARK、cuSync、DeepEP、FLUX、TileScale，以及 CUDA Graph/PDL/SuperKernel 等替代路线。

但如果把问题严格问成“现在是否已经有一个类似 CUTLASS、可跨模型/shape/硬件复用、接口稳定、核心完整开源的 MegaKernel 算子库”，答案仍然是：**没有**。

目前最清晰的开源图景是：

- **整模型路线多，通用库少。** 很多项目证明单 launch 可以覆盖完整 forward、完整 token，甚至完整生成循环，但通常绑定 batch 1、固定模型和具体 GPU 代际。
- **MoE 是库化最快的方向。** dispatch、GEMM1、activation、GEMM2、combine 与通信之间存在天然的 tile-level overlap，融合边界可控，也更容易形成可被 serving/training 框架调用的 operator backend。
- **可组合框架正在成形。** Machete 的 Op/instruction stream、MPK 的 tGraph/event runtime、DITRON 的多层 tiling/scoreboard、TIRx 的 Python/TIR 编译栈，开始把“单个英雄 kernel”变成可编程系统。
- **异构硬件已不再是空白。** AMD 有官方 Fleet/MI350；AWS 有 NKI/Trainium；Ascend 同时出现 AscendC/PTO、PyPTO、TorchAir SuperKernel、Triton-distributed port；TPU 有 SGLang-JAX Pallas FusedEPMoE。
- **开放程度必须独立评价。** TileRT 是公开工具层加 binary backend；PTO/PyPTO 使用限制用途的 CANN OSL；DSV2、flint、MegaGDN、MegaQwen、FlashFormer 等源码可读但根仓未声明许可证；这些不能与 Apache/MIT/BSD 的完整源码项目混称“严格 OSS”。

一句话概括：**2026 年的 MegaKernel 已进入“设备内任务运行时 + 分布式 MoE 大核”阶段，但还没有完成从研究原型到统一、可移植算子库的收敛。**

## 二、什么才算 MegaKernel

本报告采用下面的核心纳入标准：

> 在一个 persistent launch 或等价的设备常驻执行底座中，调度原本通常分属于多个算子、模型阶段或通信阶段的 tile/task。

关键不是 kernel 大不大，也不是名称里是否含 mega，而是原来会被 host/runtime 分开调度的工作，是否真的被移入了一个设备内执行域。

| 概念 | 与 MegaKernel 的关系 |
|---|---|
| Persistent kernel/thread | 常见实现机制，但不是充分条件；一个长驻 GEMM 仍只是单算子。 |
| Kernel fusion | 更宽的上位概念；普通 elementwise/epilogue fusion 不一定有跨算子设备调度。 |
| CUDA Graph | 一次 host 提交/回放多个 kernel，降低 launch setup，但 kernel-wide 边界仍在。[CUDA 官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) |
| Programmatic Dependent Launch | 可在设备侧串联多个 kernel，仍保留多 kernel 边界；是重要替代路线。 |
| Graph compiler | TVM/XLA/Inductor 通常输出多个 fusion group；只有显式生成 persistent task runtime/single kernel 的 backend 才进入本报告核心。 |
| Operator library | CUTLASS、FlashAttention、Liger、FlashInfer 整体不是 MegaKernel；只能按其中真实 MegaMoE/MonoMoE 路径单列。 |
| Single-kernel LLM | Whole-model MegaKernel 的强子类，但还要逐项核实 embedding、全部层、LM head、sampling、prefill、token loop。 |
| Ascend SuperKernel | 把 child kernel 编入一个大 binary/scope，可多流执行；与源码级 persistent interpreter 不同，单列为 binary fusion 路线。 |

这个边界也解释了为什么 2013 年 NVIDIA 的 [Megakernels Considered Harmful](https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus) 仍值得读：图形学早已说明 divergence、寄存器压力和 instruction footprint 会让 wavefront 多 kernel 更优。MegaKernel 不是普遍优越的定律，而是针对 launch gap、跨算子 HBM 往返、wave quantization 和粗粒度通信屏障的一种执行策略。

## 三、检索范围与证据方法

本次不是只搜 GitHub topic，而是采用“关键词扩展 + 论文回溯 + 源码边界核验 + 重复谱系去重”：

- GitHub 对 exact megakernel 相关结果做了百量级仓库筛查；topic 页本身只有极少项目，远不足以代表生态。
- 扩展词包括 mega-kernel、mega kernel、whole-model kernel、single-launch transformer、persistent model、persistent MoE、MegaMoE、single kernel decode、compute communication persistent kernel 等。
- 交叉覆盖 GitHub、GitLab、Hugging Face kernel artifact、GitCode/CANN、项目官网、arXiv、USENIX、MLSys、NeurIPS 和官方技术博客。
- 使用 [Awesome-MegaKernel](https://github.com/qhy991/Awesome-MegaKernel) 作为公开基线，但独立补入其快照之后或遗漏的 Machete、Fleet 代码、DeepGEMM MegaMoE、FastAFD、FlashInfer 新 MegaMoE backend、TIRx、TeraMoE、SGLang NPU/TPU、NKI MoE 等。
- 对每个重点项目至少核验 README/文档、根许可证、关键 scheduler/orchestrator/kernel 源码路径，以及一次 launch 的真实覆盖范围。
- 不以 stars、营销标题或 benchmark 图作为技术分类依据；性能数字只在原项目配置内视为作者结果。
- 对 redirect、fork、vendored 代码和算法移植显式去重。

归档包含 85 个工件，但它们不是 85 个独立 MegaKernel 算法，也不是 85 个严格开源库。目录分布如下：

| 类别 | 数量 | 说明 |
|---|---:|---|
| 编译器/runtime | 7 | 生成或承载设备内跨算子执行 |
| 整模型/整 token | 16 | 含别名和同谱系变体 |
| 分布式/MoE | 12 | 当前最活跃的核心实现群 |
| 子图/算子级 | 12 | block、MLP、GDN、GDPA、TTS 等 |
| 基础设施 | 22 | DSL、通信、同步、框架接线 |
| 替代路线 | 5 | PDL、Graph、multi-persistent、SuperKernel |
| 长尾/实验 | 9 | 个人项目、fork、未完成和误命中 |
| 目录/课程 | 2 | 资料，不计执行系统 |

## 四、五层开源图景

| 层级 | 代表项目 | 当前成熟度 | 主要限制 |
|---|---|---|---|
| 通用编译/runtime | Mirage/MPK、Machete、Triton-distributed/DITRON、Luminal、AutoMegaKernel | 研究系统进入可复用框架阶段 | frontend、动态 workload、硬件可移植性和稳定 API 尚未收敛 |
| 整模型/整 token | Hazy、model-as-a-kernel、NKI TKG、Lucebox、Bonsai、DSV2、flint | 已证明从小模型到 236B、多 GPU都可做 | 高度绑定模型、batch、shape、精度和架构 |
| 分布式/MoE | DeepGEMM、FlashInfer、FlashMoE、Mixture-of-Kittens、TeraMoE、SGLang NPU/TPU | 最接近工程“算子库” | topology/quantization/transport 专用，接口仍碎片化 |
| 子图/层级大核 | Meta GDPA、MegaGDN、MLP megakernel、ClusterFusion、Cherimoya | 易集成、边界清晰 | 不能外推成 whole-model |
| 基础/替代路线 | ThunderKittens、TileLang/TIRx、NKI/PTO、ARK、PDL、StreamEP、TorchAir | 技术供给丰富 | 不满足同一种 launch 语义，需避免概念混并 |

### 4.1 最接近“库”的通用系统

| 系统 | 编程/编译入口 | 设备内调度 | 硬件 | 判断 |
|---|---|---|---|---|
| [Mirage/MPK](https://github.com/mirage-project/mirage) | tensor program → SM-level tGraph/task/event | worker + scheduler、分散队列、event-driven | NVIDIA 单/多 GPU；Fleet 扩展 AMD | 当前最完整的通用 compiler/runtime 之一；Apache-2.0。 |
| [Machete](https://github.com/b-albar/machete) | CuTe DSL 可组合 Op、named buffer 和 tile | build-time scheduler 生成扁平 instruction stream，persistent CTA replay | Hopper/Blackwell | 当前最像“MegaKernel operator framework”的项目；Apache-2.0，但年轻且硬件窄。 |
| [Hazy Megakernels](https://github.com/HazyResearch/Megakernels) | 手工/生成的 per-SM instruction sequence | GPU interpreter、paged SMEM、counter dependencies | H100/B200 | 开创性且源码清楚；更像研究 runtime + 高度特化 demo。 |
| [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) | Triton tasks、distributed IR、多层 tiling | compile-time scoreboard 与通信 swizzle | NVIDIA/AMD，多 GPU/节点 | 同时承载 dense TP、MoE EP 和 DITRON 研究；公开接口完整度仍分层。 |
| [AutoMegaKernel](https://github.com/RightNow-AI/AutoMegaKernel) | agent-facing schedule IR | cooperative static forward kernel | sm75–sm120 | 适合自动合成/验证实验；当前动态 batching 与 MoE 仍非主完成面。 |
| [Luminal](https://github.com/luminal-ai/luminal) | Rust compute graph → BlockOps/work queue | 单 kernel symbolic queue/interpreter | NVIDIA CUDA 路线 | megakernel backend 真实存在但仍演进，暂不与 MPK/Machete 同成熟度。 |
| [Fleet](https://github.com/ROCm/fleet-chiplet-megakernel) | MPK task graph + Chiplet-task | XCD-local worker、L2 counter、分层 signal | AMD MI350/gfx950 | 当前少见的官方 AMD 完整源码路径；解决 flat-SM scheduler 在多 die GPU 上的局部性问题。 |

这里有两条截然不同的“通用化”路线：

- **静态可组合 instruction stream**：Machete、Hazy、部分 DITRON。构建期决定大量调度，运行时开销低，但 shape/模型/硬件绑定强。
- **设备内任务图/动态事件调度**：MPK、Luminal、Event Tensor、TileRT。对 SM jitter、动态依赖、MoE imbalance 更有弹性，但需要 queue、atomics、scheduler SM 和更复杂的内存序。

Event Tensor 的论文提出把 task 依赖提升为可带 symbolic shape、index expression、wait count 的一等 IR，并可生成静态队列或动态 ready queue；截至快照日仍未发现官方源码。[MLSys 论文页](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html)

### 4.2 整模型/整 token：边界比项目名更重要

| 项目 | 单次设备执行覆盖 | 未覆盖或限制 | 开放状态 |
|---|---|---|---|
| [model-as-a-kernel](https://huggingface.co/kernels/phanerozoic/model-as-a-kernel) | embedding、全部层、LM head、greedy argmax、prompt 消费、完整 generation loop | 单 GPU、batch 1、greedy、有限 Llama/RoPE 变体 | Apache-2.0 |
| [AWS NKI Transformer TKG](https://github.com/aws-neuron/nki-library) | 多层 attention、MLP、residual、跨层 collective 的一个 token-generation invocation | Trainium/Inferentia 专用 | Apache-2.0 |
| [NKI MoE megakernel](https://github.com/KevGomes1403/nki-moe-megakernel) | decoder stack、MoE、vocab、greedy、TP all-reduce；Qwen3.6 还含 draft/verify/replay round | 具体模型/Trainium2/3 | Apache-2.0 |
| [Hazy Megakernels](https://github.com/HazyResearch/Megakernels) | Llama whole-forward；另有 TP Llama-70B 路线 | compiler/env、模型和配置高度特化 | MIT |
| [Lucebox](https://github.com/Luce-Org/lucebox) | Qwen3.5-0.8B 24 层 DeltaNet/attention 单 persistent dispatch | prefill 分离、batch 1/模型专用 | Apache-2.0 |
| [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) | embedding、28 层、final norm | LM head/argmax 分离 | MIT |
| [DSV2-236B](https://github.com/SwayamInSync/DSV2-236B-MegaKernel) | 8×B200 上 60 层、TP/EP collective、LM head、argmax 的一枚 CuTe kernel | 固定模型/拓扑/精度；无根许可证 | source-visible |
| [flint](https://github.com/asmit383/flint) | Granite-4.1-3B int4 whole decode，并在其上做 speculative decode | H100/模型专用；无根许可证 | source-visible |
| [Bonsai Turbo](https://github.com/RightNow-AI/bonsai-turbo) | Bonsai 27B ternary 的 cooperative whole-model path | 模型/精度/H100 专用 | Apache-2.0 |
| [Talos microGPT](https://github.com/AlexCheema/talos-vs-macbook) | 单 block 内 forward、sampling 与多 token loop | 4192 参数教学模型 | MIT |

“single-kernel LLM”本身也有四级：

1. 只把连续 Transformer layers 放进一个 kernel；
2. 再包含 embedding/final norm；
3. 再包含 LM head/argmax 或 sampling；
4. prompt 消费和多 token generation loop 也不返回 host。

如果不写清这一级别，qwen_megakernel 与 model-as-a-kernel 会被错误地描述成同一种覆盖边界。

## 五、为什么 MoE 是当前最成熟的开源方向

MoE 层的典型流水线是：

router/metadata → dispatch/all-to-all → expert linear1 → activation/quantization → expert linear2 → combine/all-to-all → weighted reduce

普通实现会在每个阶段形成 kernel-wide/collective-wide 边界。MegaMoE 可以让通信 worker、GEMM tile、ready counter 和 combine worker 在同一设备执行域中并行推进，收益不仅是省 launch，更重要的是消除“必须等整批 dispatch 完成才能启动 GEMM”的粗粒度屏障。

| 项目 | 单核/设备执行边界 | 训练/推理 | 硬件与传输 | 开放性与成熟度 |
|---|---|---|---|---|
| [DeepGEMM MegaMoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe) | EP dispatch → FP8×FP4 linear1 → SwiGLU → linear2 → EP combine | 推理为主 | SM100、NVLink/symmetric memory | MIT；代码路径短而清楚，是当前最重要的 MegaMoE 核心之一。 |
| [FlashInfer MegaMoE](https://github.com/flashinfer-ai/flashinfer) | 多种 SM90/SM100 mega backend，含 DeepGEMM、BF16、NVFP4、MXFP8 | serving/operator backend | Hopper/Blackwell，多 rank | Apache-2.0；已进入生产型 kernel library，而不是论文孤岛。 |
| [FlashMoE](https://github.com/osayamenja/FlashMoE) | dispatch + expert FFN + combine + device-side communication | 推理 | NVIDIA、NVSHMEM/CUTLASS/cuBLASDx | BSD-3-Clause；完整 persistent kernel 研究实现。 |
| [Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens) | deterministic forward + backward megakernels，含 activation replay | 训练 | GB200/GB300 NVL72 | Apache-2.0；把开源前沿从 forward 扩展到 backward。 |
| [TeraMoE](https://github.com/PFCCLab/TeraMoE) | dispatch/scheduler/compute/combine/gather 五类 worker 共居一个 cooperative kernel | 跨节点训练 | SM100、NVSHMEM/RDMA | MIT 为主；早期但源码完整，派生文件另有许可。 |
| [mKernel](https://github.com/uccl-project/mKernel) | 完整 MoE 或 collective+GEMM/ring attention 等多种 persistent kernel | 推理/系统原语 | Hopper、NVLink、CX7/EFA | MIT；覆盖多节点，是“通信计算大核库”而非一个固定 kernel。 |
| [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) | EP 常见为 dispatch+GroupGEMM 与 GroupGEMM+combine 两类大核；另有 task-level system | 训练/推理 | NVIDIA/AMD、多级互联 | MIT 为主；论文、tutorial 和当前公开路径要分别表述。 |
| [FastAFD](https://github.com/hao-ai-lab/FastAFD) | AFD 的 M2N 角色路径；expert 侧 persistent kernel 可跨层/多 microbatch lane | disaggregated serving | GB200 NVL72 | MIT；不是“整个 AFD 系统一枚核”，但其跨层 expert path 是独立贡献。 |
| [Alpha-MoE](https://github.com/Aleph-Alpha/Alpha-MoE) | TP W8A8/FP8 MoE layer megakernel | 推理 | Hopper | Apache-2.0；仓库已只读归档，偏 shape-specific 历史实现。 |
| [SGLang Kernel NPU](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/doc/FUSED_DEEP_MOE.md) | AscendC 单核 dispatch+2×GMM+activation+combine；增强模式含 routing+AllToAll/HCCL | 推理 | Ascend A3/A5 | MIT；目前最明确的 Ascend 严格 OSS MegaMoE 路径之一。 |
| [CANN PTO dispatch_mega_combine](https://pto-isa.gitcode.com/kernels/manual/a2a3/dispatch_mega_combine/) | reorder、dispatch、GMM1、SwiGLU、GMM2、combine、unpermute 的混合 AIC/AIV 大核 | 推理 | Ascend A2/A3/A5、HCCL window | 源码公开但 CANN OSL 限制用途，不属于 OSI OSS。 |
| [SGLang-JAX FusedEPMoE](https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/08-pallas-kernels.md) | routing、A2A scatter、expert FFN、A2A gather、shared expert、accumulation 的一次 Pallas call | serving | TPU v6e/v7 | Apache-2.0；证明 MegaMoE 并非 CUDA 独占。 |

这组项目说明了三个趋势：

- **MegaKernel 的工程主战场从“省几十次 launch”转向“细粒度通信计算重叠”。**
- **训练正在追上推理。** Mixture-of-Kittens、TeraMoE、Triton-distributed、Primus-Turbo 等开始处理 backward、determinism、activation memory 和 autograd。
- **MegaMoE 正在被普通 operator library 吸收。** FlashInfer 和 DeepGEMM 的路径比单独研究仓更接近现有 serving 栈的实际采用方式。

需要避免的反向误判：

- DeepEP/MoonEP 的 persistent dispatch 或 combine 如果不包含 expert FFN，就不是完整 MegaMoE。
- SonicMoE 是高性能 grouped MoE operator/persistent tile scheduler，不等于跨节点 dispatch+FFN+combine 一枚核。
- AMD Primus-Turbo “Mega MoE”是两枚通信-GEMM融合核，中间仍有 SwiGLU。
- Triton-distributed EP 当前常见公开路径也是两类大核，不能一概写成“整个 MoE 一枚核”。

## 六、硬件开源版图

### 6.1 NVIDIA：数量最多，但碎片化也最严重

NVIDIA 路线覆盖：

- 消费卡：RTX 3090 的 MegaQwen、RTX 5090 的 qwen_megakernel/Nemotron、TU102 的 sinter 实验；
- 数据中心：H100 的 Hazy、flint、Machete、FlashMoE；B200/GB200/GB300 的 MPK、DeepGEMM、FlashInfer、DSV2、Mixture-of-Kittens、FastAFD、TeraMoE；
- 分布式：NVLink/symmetric memory、NVSHMEM、multimem、RDMA/CX7/EFA；
- 编程栈：CUDA C++、CuTe DSL、Triton、ThunderKittens、TIRx、Rust/Rust-CUDA 式实验。

问题是“支持 NVIDIA”往往只说明整个仓支持某架构，不代表具体 mega 子路径支持。例如 DeepGEMM 整仓支持多代 GPU，不应由此推断其公开 MegaMoE 已支持 H100；当前关键实现是 SM100。

### 6.2 AMD：从相邻基础设施进入官方 whole-model runtime

- [ROCm Fleet](https://github.com/ROCm/fleet-chiplet-megakernel) 是最重要变化：官方 Apache-2.0 源码已公开，针对 MI350/gfx950 的多 XCD 私有 L2 和分层 signal 调度。
- [Primus-Turbo Mega MoE](https://github.com/AMD-AGI/Primus-Turbo/blob/main/docs/README_Mega_MoE.md) 在 MI355X 上实现两枚 FlyDSL 通信-GEMM大核，但严格说不是完整单核 MoE。
- Triton-distributed、TileLang、ARK 等具 AMD backend/路径，但必须以具体 demo 的可运行性为准。
- 非官方 ThunderKittens AMD port 应按 fork/移植看，不代表形成了独立成熟框架。

AMD 目前项目数量仍少于 NVIDIA，但 Fleet 已把生态从“只有基础工具/移植”推进到官方、完整、拓扑感知的 MegaKernel runtime。

### 6.3 Ascend：路线最分叉的非 CUDA 生态

| 路线 | 代表 | 技术语义 |
|---|---|---|
| 源码级混合 AIC/AIV 大核 | SGLang Fused Deep MoE、PTO dispatch_mega_combine、MegaGDN | 多阶段直接写入一个 AscendC/PTO kernel，最接近经典 MegaKernel。 |
| MPMD/图编译 | PyPTO | 将 tensor/MPMD 图映射到设备执行；目前公开模型路径多是局部 fusion。 |
| Triton task scoreboard port | Triton-distributed-ascend feature/megakernel | 已有 two-matmul/MLP opgraph，大模型公开 demo 尚未闭环。 |
| child-binary SuperKernel | TorchAir + LongCat NPU | 将多个 child kernel 编成大 binary/scope，可 stream-fusion；不等于 source-level persistent monolith。 |
| tile DSL/primitive | tilelang-ascend、pto-kernels | 构建底座，不应整体计作 MegaKernel。 |

Ascend 生态的技术多样性很高，但许可从 MIT/BSD，到 CANN OSL，再到 MegaGDN 无根许可证并存；“公开可读”与“开放使用”必须拆开。

### 6.4 Trainium/Inferentia：官方 whole-model token generation

[AWS NKI Library](https://github.com/aws-neuron/nki-library) 的 Transformer TKG 是非 GPU 路线中最明确的官方完整开源实现之一：attention、MLP、residual 和 collective 跨层执行于一次 NKI invocation。[NeuronX Distributed Inference](https://github.com/aws-neuron/neuronx-distributed-inference) 提供框架接线，第三方 [nki-moe-megakernel](https://github.com/KevGomes1403/nki-moe-megakernel) 则把边界推进到完整 MoE decoder、vocab/argmax 和 speculative round。

它的限制也同样明显：硬件和编译环境完全锁定 Neuron/NKI，但这恰恰说明 MegaKernel 的抽象并不依赖 CUDA。

### 6.5 TPU：Pallas MegaMoE

[SGLang-JAX](https://github.com/sgl-project/sglang-jax) 的 FusedEPMoE 使用 Pallas 的 BlockSpec、VMEM/SMEM、DMA 和 MXU，把 routing、A2A、专家 FFN 和 gather 放在一次调用中，面向 TPU v6e/v7。它是目前归档中最明确的 TPU 大核算子库路径。

### 6.6 尚未形成成熟开源路线的区域

- Apple Metal/MLX 有高性能 whole-model/graph 工程，但本次未找到与 MPK/Machete 同级、边界明确且独立开源的设备内跨算子 runtime。
- 韩国/日本定向检索没有发现与上述同级的公开一体化 MegaKernel 库；ES-MoE/FastMoE 等属于 offload/pipeline/模块化 MoE，而非单核。
- MI300X 有 [Kog.ai single-kernel engine](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/) 的技术文章，但核心未开源。

## 七、开放程度：不要把“能下载”当作同一种开源

| 层级 | 代表 | 应如何表述 |
|---|---|---|
| 完整、明确许可的核心源码 | MPK、Machete、Hazy、Fleet、DeepGEMM、FlashInfer、FlashMoE、Mixture-of-Kittens、TeraMoE、SGLang NPU/TPU、NKI | 可称 OSS，但仍需注明硬件/模型/子路径限制。 |
| 公开源码、功能/论文路径不完整 | FlashFormer、Triton Ascend Qwen stub、Luminal evolving backend | 只能称 prototype/partial artifact。 |
| open-core/binary backend | TileRT | 公开仓库是 MIT，但核心 runtime/compiler 不能完整审阅。 |
| 厂商限制用途许可 | PTO-ISA、PyPTO | source-available/vendor-licensed，不写成 OSI open source。 |
| 无根许可证 | DSV2、flint、MegaQwen、Maruthi、MegaGDN、ClusterFusion、DeepFusionKernel、FlashRT artifacts、VDCores | 可研究源码，但默认没有再分发/使用授权；不纳入严格 OSS 主集。 |
| 重复/别名/移植 | Luce redirect、det-infer、TIRx MegaMoE、ThunderKittens AMD port、vendored DeepGEMM | 记录实现价值，但方案数按 lineage 去重。 |

这一分层只用于回答“严格开源”与“公开可读源码”是否相同：一个项目是否能被复用，与代码是否在公网可读，是两件不同的事。

## 八、五种主要实现范式

### 8.1 静态单体 cooperative kernel

代表：AutoMegaKernel、Lucebox、qwen_megakernel、DSV2、flint、Bonsai。

优点：

- 几乎没有设备内解释器开销；
- 编译器能对固定 shape、权重布局、SM 数做极致 specialization；
- 易于把 residual、KV update 和中间状态留在寄存器/SMEM。

代价：

- 代码体积、寄存器、共享内存和 divergence 压力大；
- 模型升级、batch/sequence 变化或硬件迁移常需重写/重编译；
- 大量 grid sync 可能取代 host launch 成为新瓶颈。

### 8.2 指令流解释器与 paged shared memory

代表：Hazy Megakernels、Machete、model-as-a-kernel。

把多个 op 分解为更小 handler，利用 per-SM instruction stream、paged SMEM、barrier formula 或 phase interpreter 在一个 launch 内执行。它在可组合性和单体性能之间折中，但也引入 instruction dispatch、页管理和同步计数成本。

### 8.3 task/event 图与动态 ready queue

代表：MPK、DITRON、Event Tensor、Luminal、TileRT。

工作单位从 operator 下降到 tile/task，依赖通过 event、scoreboard、queue、counter 表达。它最适合动态 shape、continuous batching、MoE routing imbalance 和跨 GPU overlap，也是最可能形成通用 runtime 的方向。

关键难点是：调度器本身会占用 SM、L2、atomics 和 memory-fence 带宽。把调度移到 GPU 并不意味着调度免费。

### 8.4 role-specialized communication-compute megakernel

代表：FlashMoE、DeepGEMM、Mixture-of-Kittens、TeraMoE、FastAFD、SGLang NPU、PTO。

一部分 CTA/SM 做 dispatch 或通信，另一部分做 GEMM/activation/combine，通过 ready signals 以 tile/token 粒度接力。这是当前最成功的工程范式，因为 MoE pipeline 本身就有清晰角色和大量可隐藏通信。

### 8.5 保留多 kernel 边界的低开销替代

代表：PDL reconstruction、StreamEP、Tencent HPC-Ops、Primus-Turbo、CUDA Graph、TorchAir SuperKernel。

这些方案往往更容易维护、调试和复用，并可能达到接近单核的性能。它们不是“失败的 MegaKernel”，而是在 code footprint、并发、硬件异步单元和工程复杂度之间做不同选择。

## 九、当前技术取舍与真正瓶颈

### 9.1 何时最有收益

MegaKernel 最有吸引力的区域通常是：

- batch 1/小 batch decode；
- 小模型或每个 op 很短、launch gap 占比高的模型；
- memory-bound 的 GEMV/低算术强度路径；
- MoE 的 dispatch/compute/combine；
- 多 GPU 中需要 tile-level compute/communication overlap 的场景；
- 固定 shape、固定模型、长期重复执行，足以摊销编译和专用化成本的服务。

当 batch/模型足够大而进入纯 tensor-core compute-bound 区域时，kernel launch gap 的占比下降，单体化的边际收益也会下降。

### 9.2 消除 launch 不等于消除 bubble

host gap 被消掉后，瓶颈会转移到：

- grid-wide barrier；
- event/counter atomics；
- spin wait 与 memory fence；
- scheduler SM 和 worker queue；
- code/instruction cache footprint；
- 寄存器与 shared-memory occupancy；
- 各角色 work imbalance；
- 跨 rank 内存序与 fabric latency。

MegaQwen 的 grid-sync 经验、Hazy 的 profile、MPK/Fleet 的调度设计、TeraMoE 的 worker 划分都说明：MegaKernel 的核心问题不是“把函数粘起来”，而是重新设计整个设备内调度系统。

### 9.3 静态与动态的张力

- 静态计划更快、更容易做 deadlock proof 和精确 buffer reuse，但被 shape/model/hardware 固化。
- 动态 ready queue 能处理 continuous batching、数据依赖 routing 和 SM jitter，但 global queue/atomics 可能成为热点。
- 实际系统正在走 hybrid：编译期固定 task graph 和大部分资源分配，运行时只对 ready order、负载均衡和通信到达做有限动态选择。

### 9.4 单体化与库化的张力

完整模型越单体化，越容易拿到极致性能，也越难复用。可持续的库形态更可能不是“一个万能的 10 万行 kernel”，而是：

- 一套可组合 tile-level operator contract；
- 明确的 buffer lifetime 和 barrier/event IR；
- 可插拔的静态/动态 scheduler；
- 硬件特定的 task handler；
- 对 serving framework 暴露稳定 operator/runtime API。

从这个角度看，Machete、MPK、DITRON/Triton-distributed、TIRx，以及 FlashInfer/DeepGEMM 的 MegaMoE backend，比多数固定 Qwen/Llama 单核更接近长期库化。

### 9.5 多 die 与多 GPU 不能只看“SM 数”

Fleet 表明，即使在一张 GPU 内，MI350 的 XCD 私有 L2 也要求 chiplet-local task 和分层 signal。多 GPU 更需要把 NVLink/xGMI、NVSHMEM/RDMA、symmetric memory、proxy progress 和拓扑编入调度。一个在单 GPU 上成立的 flat task scheduler，不能自动扩展到多 die/多节点。

## 十、中国/亚洲开源图景

中国/亚洲生态并非只有“跟随 CUDA whole-model demo”，已经形成三层：

### 核心大核

- [ByteDance Triton-distributed/MegaTritonKernel](https://github.com/ByteDance-Seed/Triton-distributed)：当前最明确的中国团队 Qwen3 dense TP whole-model 路线，并承载 DITRON 与 EP 大核。
- [DeepSeek DeepGEMM MegaMoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)：SM100 的完整 EP dispatch—两层专家—combine 单核。
- [SGLang Kernel NPU](https://github.com/sgl-project/sgl-kernel-npu)：A3/A5 AscendC Fused Deep MoE。
- [CANN PTO-ISA](https://gitcode.com/cann/pto-isa)：A2/A3/A5 dispatch_mega_combine；厂商限制许可。
- [PFCCLab TeraMoE](https://github.com/PFCCLab/TeraMoE)：跨节点训练 persistent kernel。
- [SGLang-JAX](https://github.com/sgl-project/sglang-jax)：TPU Pallas FusedEPMoE。

### 编译与厂商基础设施

- Triton-distributed-ascend feature/megakernel；
- PyPTO MPMD graph；
- TorchAir SuperKernel；
- TileLang/TileScale/TileOPs/tilelang-ascend；
- PTO-kernels、DeepEP、MoonEP、FLUX。

### 层级深融合与相邻路径

- MegaGDN：GDN/KDA 六阶段 Ascend 单核；
- ClusterFusion/DeepFusionKernel：attention/MLP 子图；
- AMD-AGI Primus-Turbo：两枚 communication-GEMM 大核；
- Tencent HPC-Ops：PDL 多 kernel pipeline，MegaKernel 仍在 roadmap；
- LongCat NPU：TorchAir SuperKernel scope。

这一生态最突出的特点是 **Ascend 的实现语义并不统一**：源码级大核、MPMD 图、Triton scoreboard、child-binary SuperKernel 共存。因此报告或 benchmark 必须说明具体是哪一种“mega”。

论文侧仍有明显“研究公开、代码未开”的缺口：

- [Ada-MK](https://arxiv.org/abs/2605.11581)：百度广告系统的 MLIR DAG-search decode megakernel；
- [Event Tensor](https://arxiv.org/abs/2604.13327)：CMU/SJTU/NVIDIA/清华/北大的动态抽象；
- [FlashFuser](https://arxiv.org/abs/2512.12949)：Hopper DSM 深融合 compiler；
- [ClusterFusion++](https://arxiv.org/abs/2604.23553)：完整 Transformer block cluster kernel；
- [UniEP](https://arxiv.org/abs/2604.19241)：没有可独立核验的完整 artifact。

## 十一、成熟度判断

### 11.1 已达到“可作为库研究/采用”的项目

| 用途 | 优先观察项目 | 原因 |
|---|---|---|
| 通用编译/runtime | MPK、Machete、Triton-distributed/DITRON | 有明确 task/op abstraction、scheduler 和非单模型价值 |
| NVIDIA MegaMoE backend | DeepGEMM、FlashInfer、FlashMoE | 关键跨阶段源码完整，已有框架/API 接入方向 |
| 训练 MegaMoE | Mixture-of-Kittens、TeraMoE、Triton-distributed | 覆盖 backward/determinism/autograd/跨节点中的不同问题 |
| AMD | Fleet | 官方、Apache-2.0、明确解决 multi-XCD locality |
| Ascend | SGLang Kernel NPU；PTO 作为受限许可补充 | 前者是严格 OSS 的一体化 AscendC 路径 |
| Trainium | AWS NKI Library + NxDI | 官方 library/API/框架接线完整 |
| TPU | SGLang-JAX | Pallas FusedEPMoE 有明确文档与实现 |

### 11.2 最有教学价值的整模型工件

- model-as-a-kernel：单 launch 边界最彻底、代码/概念相对直接；
- Hazy Megakernels：理解 instruction interpreter、paged SMEM 和 device scheduling；
- qwen_megakernel/MegaQwen：理解消费卡 batch-1 decode 和 grid sync；
- Lucebox：理解 hybrid DeltaNet/attention 的模型专用单 dispatch；
- Talos microGPT：最小化的 full forward/sampling/token-loop 示例；
- DSV2/flint：理解大模型/多 GPU/int4 的前沿，但注意无根许可证。

### 11.3 尚未达到统一库的原因

1. **输入 IR 不统一**：PyTorch graph、Triton task、CuTe Op、NKI program、Pallas call、AscendC/PTO 各自为政。
2. **调度 ABI 不统一**：instruction stream、event tensor、scoreboard、worker role、binary scope 语义不同。
3. **算子 contract 不统一**：tensor layout、quantization scale、KV cache、top-k metadata 和 symmetric buffer 都高度特定。
4. **可移植性差**：多数关键路径只覆盖一代 GPU、固定 SM 数或特定 fabric。
5. **动态 serving 不完整**：continuous batching、cancellation、variable sequence、speculation、prefill/decode 混跑仍是少数系统的研究面。
6. **调试/验证成本高**：跨 CTA/rank deadlock、memory ordering 和 progress bug 很难用普通 kernel 工具定位。
7. **性能不可横比**：项目使用不同模型、精度、batch、context、拓扑和 baseline，不能按 README speedup 做排行榜。
8. **开放性碎片化**：open-core、无许可证、厂商限制许可与完整 OSS 并存。

## 十二、目前最值得关注的演进方向

### 12.1 任务 IR 标准化

如果未来出现事实标准，它很可能描述：

- tile/task 的资源需求；
- 输入输出 buffer 和 lifetime；
- event/barrier dependency；
- compute/communication engine affinity；
- static schedule 与 dynamic-ready 两种 lowering；
- topology/chiplet/rank awareness。

MPK tGraph、Event Tensor、DITRON multi-level IR 和 Machete Op contract 都在探索这组问题。

### 12.2 MegaMoE backend 进入主流 serving API

FlashInfer 已经展示了一个现实路径：不要求 serving 系统整体变成 MegaKernel，而是在 MoE layer API 下根据 dtype、architecture、topology 选择 split 或 mega backend。这个渐进式集成方式比替换整个执行引擎更容易落地。

### 12.3 training-first 大核

Mixture-of-Kittens、TeraMoE、Primus-Turbo 将问题扩展到 backward、router gradient、deterministic ordering、activation replay/recompute 和 autograd。未来“算子库”的竞争力可能更多来自训练闭环，而不是又一个 batch-1 decode demo。

### 12.4 chiplet/topology-aware scheduler

Fleet 证明层次化内存和 chiplet topology 必须成为 IR/调度的一等属性。随着多 die GPU、NVL72、跨节点 RDMA 成为常态，“把 SM 看成平坦池”会越来越不够。

### 12.5 自动生成与自动搜索

AutoMegaKernel、Ada-MK、Mirage superoptimizer、编译期 schedule search 都在试图降低手工编排成本。真正难点不是生成语法正确的巨型 kernel，而是同时满足：

- 无死锁的 dependency；
- 可接受的寄存器/SMEM；
- shape/architecture specialization；
- 与 baseline 对齐的数值语义；
- 稳定性能而不是单点 benchmark。

## 十三、实践选型建议

如果目标是“从源码理解 MegaKernel”：

1. 先读 model-as-a-kernel 或 Talos，建立 single-launch 直觉；
2. 再读 Hazy，理解 instruction stream、paged SMEM 和 counter；
3. 读 Machete，理解可组合 Op 如何变成 schedule；
4. 读 MPK，理解 task/event compiler-runtime；
5. 读 DeepGEMM/FlashMoE/TeraMoE，理解通信计算 role scheduling。

如果目标是“建设一个可复用库”：

- NVIDIA Hopper/Blackwell：优先比较 Machete、MPK、Triton-distributed、TIRx；MoE 直接研究 DeepGEMM/FlashInfer；
- AMD MI350：以 Fleet 为 whole-model runtime 基线，Primus-Turbo/TileScale 为通信-GEMM相邻基线；
- Ascend：严格 OSS 优先 SGLang Kernel NPU/Triton-distributed-ascend，PTO/PyPTO 需接受 CANN OSL 条件；
- Trainium：NKI Library + NxDI；
- TPU：SGLang-JAX/Pallas。

如果目标是“做性能对比”：

- 不要混用 whole-model、operator-level、CUDA Graph 和 two-kernel fusion；
- 固定模型、精度、batch、context、KV cache、sampling、GPU 拓扑；
- 分别测 cold compile、warm steady-state、单 token、完整 request；
- 把 router、pre-staging、LM head、sampling 和通信是否计时写清楚；
- 用同一个 correctness baseline，但不把本调研变成广泛硬件审计。

## 十四、关于 B300 验证

用户提供了 B300 环境作为必要时的定点核验资源。本次未连接该机器，原因是：

1. 主要任务是开源生态与源码边界调研，关键结论已能从 launch site、scheduler/orchestrator 和 kernel source 静态确认；
2. 85 个工件横跨 H100/B200/GB300、MI350/MI355、Ascend A2/A3/A5、Trainium2/3、TPU v6e/v7 和多节点 fabric，单台 B300不能代表整体；
3. 很多项目需要固定模型权重、NVL72、4/8 rank symmetric memory、RDMA/NVSHMEM 或专用编译器，强行在 B300 上测试会把“环境不匹配”误当作项目结论。

B300 更适合后续做三个有边界的验证：

- Machete/MPK/Hazy 的 Blackwell launch/runtime 对比；
- DeepGEMM 或 FlashInfer 某一 MegaMoE shape 的单/多卡正确性与 profiler；
- qwen_megakernel/model-as-a-kernel 的 whole-model launch 边界复现。

## 十五、最终判断

当前开源 MegaKernel 不是一个单一市场，而是五个相互连接的子生态：

1. whole-model 手工专用 kernel；
2. 可组合 instruction interpreter；
3. task/event compiler-runtime；
4. 分布式/MoE communication-compute megakernel；
5. 保留多 kernel 的低开销替代执行模型。

其中：

- **研究完整性最高**：MPK、Hazy、Machete、DITRON/Triton-distributed；
- **工程采用潜力最高**：DeepGEMM MegaMoE、FlashInfer、SGLang NPU、NKI；
- **训练前沿最强**：Mixture-of-Kittens、TeraMoE；
- **异构突破最明显**：Fleet、SGLang NPU/TPU、AWS NKI；
- **最大短板**：统一 IR/API、dynamic serving、跨硬件 portability、可比 benchmark 和完整开放性。

所以，更准确的 2026 开源图景不是“已经出现成熟 MegaKernel 算子库”，而是：

> **MegaKernel 已从模型专用技巧演化为一种新的设备内执行层；MoE 率先形成可复用 operator backend，通用 whole-model runtime 正在竞争抽象，但生态仍处于快速分化、尚未标准化的阶段。**

---

附：

- [85 个工件逐项索引](CATALOG.md)
- [归档目录说明](README.md)
- [MPK 论文](https://arxiv.org/abs/2512.22219)
- [AutoMegaKernel 论文](https://arxiv.org/abs/2606.09682)
- [Fleet 论文](https://arxiv.org/abs/2604.15379)
- [DITRON 论文](https://arxiv.org/abs/2605.02953)
- [FlashMoE 论文](https://arxiv.org/abs/2506.04667)
- [VDCores 论文](https://arxiv.org/abs/2605.03190)
