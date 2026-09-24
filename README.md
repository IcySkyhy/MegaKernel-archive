# MegaKernel Archive

MegaKernel 开源生态调研与源码快照。

本仓库包含：

- archive/：MegaKernel 编译器、whole-model kernel、分布式/MoE kernel、算子级大核、基础设施、替代路线和长尾实验的源码归档；
- archive/MEGAKERNEL_OPEN_SOURCE_LANDSCAPE_2026-08-26.md：开源图景总报告；
- archive/CATALOG.md：逐项索引、开放状态、真实 launch 边界与关键源码说明；
- archive/README.md：归档结构和使用说明；
- kernelwiki/：配套的内核知识与检索材料。

主报告的系统性快照日期为 2026-08-26；本仓库上传的是 2026-09-24 的完整本地工作区快照，包含报告完成后继续补充的实验材料。

## 结论摘要

当前尚不存在一个同时具备 CUTLASS 式复用性、动态 workload、跨模型与跨硬件能力、稳定 API 和完整开源核心的统一 MegaKernel 算子库。

目前最清晰的三条路线是：

1. Mirage/MPK、Machete、Triton-distributed/DITRON 等设备内任务图与可组合运行时；
2. DeepGEMM MegaMoE、FlashInfer、FlashMoE、Mixture-of-Kittens、TeraMoE、SGLang NPU/TPU 等分布式/MoE 大核；
3. ThunderKittens、TileLang/TIRx、AWS NKI、Ascend PTO/PyPTO 等编程和编译底座。

整模型实现已经覆盖 NVIDIA、AMD、Ascend、Trainium 和 TPU，但仍普遍绑定具体模型、shape、精度、拓扑和设备代际。

## 归档方式

GitHub 仓库保存的是各项目工作树的扁平化源码快照，不包含嵌套项目自身的 .git 对象数据库。各项目的原始上游地址、许可证与谱系关系记录在 archive/CATALOG.md 中。

部分目录属于公开可读但无明确根许可证、厂商限制许可、open-core、论文配套或替代路线；进入本归档不代表它们都属于严格意义上的开源软件。
