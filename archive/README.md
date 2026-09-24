# MegaKernel 开源归档

MegaKernel 实现图景快照：2026-08-26；agentic kernel design 扩展：2026-09-01。

本目录收录 107 个浅克隆仓库或可下载源码工件：85 个用于刻画 MegaKernel 实现图景，另有 22 个 kernel-agent、知识库、评测与数据系统，作为后续 KernelWiki/Skill 蒸馏的方法参考。归档基于源码边界而非项目命名；因此“位于 archive”不等于“已经被认定为成熟开源 MegaKernel 库”。

## 先看这四个文件

- [开源图景总报告](MEGAKERNEL_OPEN_SOURCE_LANDSCAPE_2026-08-26.md)：结论、定义、技术路线、硬件分布、成熟度和空白。
- [完整归档索引](CATALOG.md)：107 个本地工件逐项说明；前 85 项记录许可证/开放状态和 MegaKernel 归类理由，后 22 项记录其在蒸馏、搜索、评测或训练中的用途。
- [KernelWiki 与 Skill 设计](../kernelwiki/KERNELWIKI_DISTILLATION_DESIGN_2026-09-01.md)：如何把仓库证据转成知识图谱、可执行 Skill、训练数据与经验回流。
- [本页](README.md)：目录导航和使用说明。

## 归档结构

| 目录 | 数量 | 含义 |
|---|---:|---|
| compiler-runtimes | 7 | 能生成或承载跨算子设备常驻执行的编译器、解释器和 runtime |
| core-whole-model | 16 | 整模型、整 forward、整 token 或完整生成循环的实现及其变体 |
| distributed-moe | 12 | 将通信、路由、专家计算、combine 等阶段放入大核的分布式/MoE 实现 |
| operator-scale | 12 | Transformer block、MLP、GDN、GDPA、TTS predictor 等子图/算子级大核 |
| foundations | 22 | tile DSL、通信、同步、persistent grid、图编译和服务框架等底座 |
| alternatives | 5 | CUDA Graph、PDL、多 persistent kernel、binary SuperKernel 等相邻路线 |
| long-tail-experimental | 9 | 教学、个人实验、fork、未完成或需谨慎核验的项目 |
| catalogs | 2 | 公开目录和课程资料 |
| agentic-kernel-design | 22 | kernel design agent、Wiki、benchmark/evaluator、trajectory 与数据生成系统；不计入 MegaKernel 实现成熟度统计 |
| **合计** | **107** | 85 个实现图景工件 + 22 个蒸馏/agent 参考工件 |

## 本调研采用的核心定义

MegaKernel 是：在一个 persistent launch 或等价的设备常驻执行底座中，调度原本通常分属于多个算子、模型阶段或通信阶段的 tile/task。

因此：

- 单个 persistent GEMM 或 attention 不自动成为 MegaKernel。
- CUDA Graph 是一次 host 提交多个 kernel，不等于一个 kernel。
- 普通 kernel fusion 是更宽的上位概念。
- “完整模型”还需分别注明 embedding、全部层、LM head、sampling、prefill 和 token loop 是否真的位于同一 launch。
- “公开源码”不自动等于“开源”：无根许可证、厂商限制许可、binary-only 核心都单独标注。

## 快速结论

当前没有一个同时满足“类似 CUTLASS 的复用性、动态 workload、跨模型、跨硬件、完整源码、稳定 API”的统一 MegaKernel 算子库。最接近可持续库化的三类工作是：

1. Mirage/MPK、Machete、Triton-distributed/DITRON、Luminal 等任务图或可组合运行时；
2. DeepGEMM MegaMoE、FlashInfer、FlashMoE、Mixture-of-Kittens、TeraMoE、SGLang NPU/TPU 等 MoE 跨阶段大核；
3. ThunderKittens、TileLang/TIRx、NKI、PTO/PyPTO 等可复用编程底座。

整模型开源实现已经很多样，但仍以固定模型、batch 1、固定 shape 和固定 GPU 代际的研究 artifact 为主。MoE 层反而是目前工程化、训练支持和跨设备扩展最活跃的区域。

## 归档说明

- Git 仓库以浅克隆方式保存，保留上游 origin，便于后续更新或深入阅读。
- Hugging Face kernel artifact 也以 Git 工件形式保存。
- 别名、重定向和同谱系实现会在 CATALOG.md 中显式标记，不以文件夹数冒充独立方案数。
- 用户提供的 B300 可作为后续 NVIDIA Blackwell 定点复现环境；本次没有使用，因为跨 NVIDIA、AMD、Ascend、Trainium、TPU 的图景无法由单台 B300公平验证，而关键 launch 边界均可由公开源码判定。
