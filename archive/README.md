# MegaKernel 开源归档

MegaKernel 实现图景快照：2026-08-26；agentic kernel design 扩展：2026-09-01；第二轮增量：2026-10-05。

本目录收录 **129 个源码仓库或工件**：106 个用于刻画 MegaKernel 实现与相邻路线，另有23个 kernel-agent、知识库、评测与数据系统。第一轮及9月初扩展为107项，本轮新增22项。归档基于源码边界而非项目命名；因此“位于 archive”不等于“已经被认定为成熟开源 MegaKernel 库”。

## 阅读入口

- [开源图景总报告](MEGAKERNEL_OPEN_SOURCE_LANDSCAPE_2026-08-26.md)：结论、定义、技术路线、硬件分布、成熟度和空白。
- [第二轮增量报告](MEGAKERNEL_ROUND2_2026-10-05.md)：近一个月新仓库、旧项目更新、待开源论文与图景修正。
- [完整归档索引](CATALOG.md)：原107项及第二轮22项的开放状态、执行范围和源码入口。
- [KernelWiki 与 Skill 设计](../kernelwiki/KERNELWIKI_DISTILLATION_DESIGN_2026-09-01.md)：如何把仓库证据转成知识图谱、可执行 Skill、训练数据与经验回流。
- [本页](README.md)：目录导航和使用说明。

## 归档结构

| 目录 | 数量 | 含义 |
|---|---:|---|
| compiler-runtimes | 10 | 跨算子设备常驻编译器、解释器、runtime 及明确标注的未完成原型 |
| core-whole-model | 23 | 整模型、整 forward、整 token、完整生成/训练循环及其部分开放工件 |
| distributed-moe | 15 | 将通信、专家计算、combine 等阶段放入大核的分布式/MoE 实现 |
| operator-scale | 16 | Transformer block、MLP、GDN、GDPA、TTS predictor 等子图/算子级大核 |
| foundations | 22 | tile DSL、通信、同步、persistent grid、图编译和服务框架等底座 |
| alternatives | 7 | CUDA Graph、PDL、多 persistent kernel、Metal dispatch 链等相邻路线 |
| long-tail-experimental | 11 | 教学、个人实验、fork、跨领域或未完成的项目 |
| catalogs | 2 | 公开目录和课程资料 |
| agentic-kernel-design | 23 | kernel design agent、Wiki、benchmark/evaluator、trajectory 与数据生成系统；不计入 MegaKernel 实现成熟度统计 |
| **合计** | **129** | 106 个实现图景/相邻工件 + 23 个蒸馏/agent 参考工件；research-rounds 仅存检索证据，不计作仓库 |

## 本调研采用的核心定义

MegaKernel 是：在一个 persistent launch 或等价的设备常驻执行底座中，调度原本通常分属于多个算子、模型阶段或通信阶段的 tile/task。

因此：

- 单个 persistent GEMM 或 attention 不自动成为 MegaKernel。
- CUDA Graph 是一次 host 提交多个 kernel，不等于一个 kernel。
- 普通 kernel fusion 是更宽的上位概念。
- “完整模型”还需分别注明 embedding、全部层、LM head、sampling、prefill 和 token loop 是否真的位于同一 launch。
- “公开源码”不自动等于“开源”：无根许可证、厂商限制许可、binary-only 核心都单独标注。

## 快速结论

当前没有一个同时满足“类似 CUTLASS 的复用性、动态 workload、跨模型、跨硬件、完整源码、稳定 API”的统一 MegaKernel 算子库。第二轮新增的 Cohere、Dist-MoE、TPU Megakernels 和 Training Megakernel 已分别推进动态服务、可复用 MoE、异构整模和训练常驻；第一轮的三条路线仍然适用：

1. Mirage/MPK、Machete、Triton-distributed/DITRON、Luminal 等任务图或可组合运行时；
2. DeepGEMM MegaMoE、FlashInfer、FlashMoE、Mixture-of-Kittens、TeraMoE、SGLang NPU/TPU 等 MoE 跨阶段大核；
3. ThunderKittens、TileLang/TIRx、NKI、PTO/PyPTO 等可复用编程底座。

整模型开源实现已经很多样，但仍以固定模型、batch 1、固定 shape 和固定 GPU 代际的研究 artifact 为主。MoE 层反而是目前工程化、训练支持和跨设备扩展最活跃的区域。

## 归档说明

- 原有 Git 仓库以浅克隆方式保存；第二轮同时使用源码 ZIP 快照、浅克隆和稀疏克隆。ZIP 工件无嵌套 `.git`，上游与时间见 `research-rounds/2026-10-05/`。
- TileMega 不包含大型 `docs/experiments` 原始输出；TensorRT-LLM 工件仅包含 CuTeDSL MegaMoE 子目录和根说明/许可。submodule 内容、模型权重和运行依赖未递归获取。
- Hugging Face kernel artifact 也以 Git 工件形式保存。
- 别名、重定向和同谱系实现会在 CATALOG.md 中显式标记，不以文件夹数冒充独立方案数。
- 用户提供的 B300 可作为后续 NVIDIA Blackwell 定点复现环境；本次没有使用，因为跨 NVIDIA、AMD、Ascend、Trainium、TPU 的图景无法由单台 B300公平验证，而关键 launch 边界均可由公开源码判定。
