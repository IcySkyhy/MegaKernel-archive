# 昇腾 950 架构与 Mega Kernel 技术调研报告

> 调研日期：2026-09-26。本报告信息分三级标注：**[官方]** = 华为官方文档/keynote/白皮书；**[社区官方]** = 昇腾社区/CANN 官方频道文章；**[第三方]** = 媒体/券商/个人博客，数字可能存在出入。文末附"未能获取的信息"清单。

---

## 1. 昇腾 950 / Atlas 950 硬件架构

### 1.1 产品定位与节奏 [官方]

2025-09-18 华为全联接大会（徐直军 keynote）公布昇腾路线图：950PR（2026Q1，Prefill/推荐）、950DT（2026Q4，Decode/训练）、960（2027Q4）、970（2028Q4）。来源：[华为官网 keynote 全文](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech)、[华为英文稿](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech)。

2026 年华为发布了**《昇腾950 NPU 架构白皮书》**（版权页 ©2026），这是目前最权威的官方架构资料（PDF）：
`https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950 NPU架构白皮书.pdf`
（检索时可用编码 URL：<https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf>）

### 1.2 计算架构 [官方：白皮书 + keynote]

- **Chiplet UMA 合封**：整芯片 = 2 个 AI Die + 2 个 IO Die + 8 个（950PR）或 4 个（950DT）高速片上内存模块，经 D2D Clink 互连，构成统一内存访问（UMA）整体。950PR/950DT 共 Die、仅内存不同。[白皮书 §3]
- **AI 子系统**：共 **36 个基于第三代达芬奇架构的 AI 子系统，每个含 1 个 Cube Core + 2 个 Vector Core**（即 36 Cube + 72 Vector）。[白皮书 §3]
  ```text
  ⚫ 36个基于第三代Davinci架构的AI子系统，每个AI子系统包括1个Cube Core和2个Vector Core。
  ⚫ 4个AI CPU Cluster，每个包括2个Linx816 CPU（ARMv8-A、双线程）和4MB L3 Cache
  ⚫ 4个DVPP子系统：4×VPC、4×JPEGE、8×JPEGD
  ⚫ 128MB统一访问的L2 Cache；STARS2.0任务调度系统
  ```
- **Cube Core**：原生支持 HiF8/MXFP8/FP8/MXFP4；同频下 HiF8/MXFP8/FP8 为 FP16 的 2 倍 TFLOPS、MXFP4 为 4 倍；**L0C Buffer 增大到 256KB，支持 256×256 tile**（[CANN 社区文](https://ascendai.csdn.net/69dce43f0a2f6a37c59f57ec.html)）；支持 L0C→UB 随路量化（FP32/INT32→BF16/FP16/FP8/INT8）与 NZ→ND/DN 排布转换。[白皮书 §4.1.1]
- **Vector Core**：从传统 SIMD 升级为**双发射 Register-Based SIMD 新架构 + 首创 SIMD/SIMT 混合编程**；单核 FP16/FP32 算力较上代 **+100%**；UB 与 Vector ALU 之间新增 RegFile 寄存器级存储；新增 BF16 原生支持。整体 **Cube:Vector 算力配比达 8:1**（对比：上代向量算力明显偏弱是 FA 类融合算子的短板）。[白皮书 §4.1.2 / CANN 社区文]
- **算力口径** [官方 keynote]：950 系列 FP8 = 1 PFLOPS、FP4 = 2 PFLOPS（稀疏口径下 PR 单卡 FP4 1.56 PFLOPS、112GB、1.4TB/s 是衍生降配版本 [第三方：<https://baike.baidu.com/en/item/Huawei%20Ascend%20950/3989140>]）。
- **精度格式全集**：TF32/FP16/BF16/FP8/MXFP8/HiF8/INT8/MXFP4（960 起再加 HiF4）。**MXFP4 原生支持**——这是 Qwen3.8-Flash-Next-MXFP4 类量化模型落地的硬件前提。[白皮书 §3 / keynote]

### 1.3 存储体系 [官方]

- **128MB 全局 L2 Cache**（整芯片统一访问，Chiplet UMA）：128Byte Sector 管理粒度（上代为 512B）、按 Way 的 Cache Lock 与驻留策略、算子可控 Cache Hint、CMO 精细管理；离散小包/随机访存同带宽性能较上代 **提升 2 倍以上**。[白皮书 §"存储体系"、§4.3.2；佐证：<https://www.sohhu.com/a/1034057245_121752158>（搜狐转载白皮书解读）、CANN 社区文]
- **片上内存（HBM）**：
  | | 950PR | 950DT |
  |---|---|---|
  | 容量 | 128GB | 144GB |
  | 带宽 | 1.6 TB/s | 4 TB/s |
  | 自研 HBM | HiBL 1.0（低成本） | HiZQ 2.0（高带宽） |
  | 场景 | Prefill/推荐 | 训练/Decode/全生命周期 |
  来源：[白皮书 §3]、[keynote]。从 950PR 起改用**华为自研 HBM**（keynote 明确提到受制裁无法用先进 HBM 的语境）。
- **RAS** 特性完备，面向大规模集群。[白皮书]

### 1.4 片间互联：灵衢 UB 2.0（HCCS 的下一代）[官方]

- 整芯片 **72 Lane HiLink SerDes，分 18 个 x4 Port**，单 Port 4×112Gbps，**整芯片对外 IO 峰值 2TB/s**；UB 双向带宽 2016GB/s，UBoE 200GB/s（与 UB Link 复用 SerDes）。[白皮书 §3；CANN 社区文]
- **UB 2.0 双语义**：同步语义 **UB Memory**（Load/Store/Atomic，支持最高 128TB Host-Device/Device-Device 内存共享）；异步语义 **URMA**（异步内存访问 + 消息语义）。[白皮书 §3/§4.6]
- **CCU（集合通信加速单元）**：硬件卸载集合通信，计算/通信深度并行，释放 AI Core。[白皮书]
- **PCIe 5.0 x16**（EP/RC 双模）+ 2×400Gbps UBoE。[白皮书]
- **与上代对比**：keynote 称"互联带宽相比 Ascend 910C 提升 2.5 倍达到 2TB/s"——即 910C 约 800GB/s（HCCS）；910B 每芯片 7 条 HCCS 链路、理论最大 392GB/s（[华为 Atlas 800T A2 官方支持页](https://support.huawei.com/enterprise/zh/doc/EDOC1100317202/f3dba488)）。950 代际起互联协议从私有 HCCS 切换为**开放协议灵衢 2.0**（华为已开放 2.0 规范）。

### 1.5 超节点与集群 [官方 keynote]

- **Atlas 950 SuperPoD**：8192 卡（950DT），128 计算柜 + 32 互联柜，全光互联；FP8 8 EFLOPS / FP4 16 EFLOPS；内存 1152TB；互联带宽 16.3PB/s；训练 4.91M TPS、推理 19.6M TPS；2026Q4 上市。
- 单柜形态（Atlas 850E/950）：14U、8×950DT + 2 鲲鹏 CPU，14.27 PFLOPS MXFP4、768GB HBM（[EE Times 分析](https://www.eet-china.com/mp/a521876.html)，[第三方]）。
- **Atlas 950 SuperCluster**：64 个超节点 ≈ 52 万卡，FP8 524 EFLOPS；集群规模 >128K 卡 [白皮书]。
- **STARS 2.0**：软硬协同任务调度系统 [白皮书 §4.4]。

### 1.6 与昇腾 910B/910C 的关键差异汇总

| 维度 | 910B（2023） | 910C（2025） | 950 系列（2026） |
|---|---|---|---|
| 架构 | 达芬奇 2.x，纯 SIMD | 双 Die | **达芬奇 3.0，SIMD/SIMT 混合**，双 Die UMA Chiplet |
| AI 核 | [第三方]约 20–32 核/片 | 2×910B 类 Die | **36 子系统 = 36 Cube + 72 Vector（1:2）** |
| 向量算力 | 弱（FA 融合短板） | 增强 | **单核 FP16/FP32 ×2，Cube:Vector=8:1，RegFile** |
| 低精度 | FP16/BF16/INT8 | +FP8 | **+MXFP8/MXFP4/HiF8，MXFP4=4×BF16 算力** |
| L2 Cache | 小容量/无统一大 L2*[第三方，未获官方确认] | — | **128MB 统一 L2，128B Sector** |
| HBM | 64GB @ ~1.6TB/s [第三方] | 128GB @ ~3.2TB/s [第三方] | PR 128GB@1.6TB/s / DT 144GB@4TB/s（自研 HBM） |
| 互联 | HCCS 392GB/s [官方支持页] | HCCS ~800GB/s（推算自 keynote "×2.5"） | **灵衢 UB 2.0，2TB/s，URMA/UB Mem/CCU** |
| 超节点 | — | Atlas 900：384 卡 | **Atlas 950：8192 卡** |
| 访存粒度 | 512B | 512B | **128B Sector（keynote 明确"512→128 字节"）** |

*910B 具体核数与 L2 配置各第三方来源互相矛盾（如 [arksight 汇总](https://www.arksight.cn/posts/b13b8d70/)、[omniyq 对比](http://omniyq.com/sys-nd/153.html)、[smzdm 评测](https://post.smzdm.com/p/apqpdzvx/) 对 FP16 算力给出 256~376 TFLOPS 不等），**建议以华为企业支持网站对应型号的《技术规格》页为准**，本表不采信单一数字。

**关键架构变化（开发视角）**：① 向量单元大幅强化 + SIMD/SIMT 混合，类 CUDA 表达成为可能；② Cube↔Vector 核内高速直连通道（CV 融合），FA 类算子单核提升 1.5~2 倍；③ NDDMA 多维搬运指令；④ 新同步机制 BufferID；⑤ 128MB L2 提供跨 Die 数据复用与 Cache Hint 控制；⑥ 自研 HBM 带宽分层（PR 偏算力、DT 偏带宽）。

---

## 2. AscendC 编程模型现状

### 2.1 Kernel 启动方式 [官方文档]

- 核函数：`extern "C" __global__ __aicore__ void kernel_name(GM_ADDR...)`；启动用 CUDA 风格内核调用符（[华为开发者官网：核函数](https://developer.huawei.com/consumer/cn/doc/hiai-Guides/cannkit-kernel-function-0000002334159509)、[昇腾社区 Kernel 直调](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0052.html)）：
  ```cpp
  add_custom<<<BLOCK_NUM, nullptr, stream>>>(x, y, z);   // blockDim ∈ [1, 65535]，逻辑核
  ```
- SPMD 模型：blockDim 决定逻辑核数，核内用 `GetBlockIdx()` 取逻辑 ID；通常设为物理核数或其倍数。
- **CANN 9.0 起新增 KERNEL_TASK_TYPE 机制**（950 的混合核调度关键）：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 允许一次启动按 1:2 配比的 AIC（Cube）/AIV（Vector）核协同执行，还有 `KERNEL_TYPE_AIC_ONLY`、`KERNEL_TYPE_MIX_AIC_1_0/1_1` 等；tilelang-ascend 生成的 kernel 已在使用（[tilelang-ascend issue #110](https://github.com/tile-ai/tilelang-ascend/issues/110)、[KERNEL_TASK_TYPE 约束分析](https://hwcomputing.csdn.net/6a93f5aa2b83d06f0ec9ea9b.html)）。torchair SuperKernel 的 debug 选项也暴露 `MIX_AIC_1_0/1_1/1_2` 三种配比。

### 2.2 内存层级 [官方]

`GM(HBM) → L2 Cache(128MB, 950) → 核内 L1/L0A/L0B/L0C(Cube) / UB(Vector) → RegFile(950 新增)`。官方 API `PlatformAscendC::GetCoreMemSize` 的枚举即层级定义（[CANN 9.0 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1034.html)）：

```cpp
enum class CoreMemType { L0_A, L0_B, L0_C, L1, L2, UB, HBM, FB, BT };
```

- 搬运靠 MTE 类指令显式完成（DataCopy），典型三段式流水 CopyIn→Compute→CopyOut，`TPipe` + `TQue`（VECIN/VECOUT）管理 buffer 与同步，双缓冲隐藏搬运延迟（[昇腾社区编程范式](https://www.hiascend.com/dev/forum/thread-0239124507827469022-1-1.html)、[华为云实战](https://bbs.huaweicloud.com/blogs/469234)）。
- **950 新增 RegBase 编程模型**：AIV 内新增一级 SIMD Register File（VF Reg），`__simd_vf__` Vector Function + `AscendC::Reg` 命名空间（LoadAlign/StoreAlign），中间结果留在寄存器不落 UB；VL=256B（950PR）；数据流必须 GM→UB→Reg→算→Reg→UB→GM（[51CTO：REGBASE 详解](https://blog.51cto.com/u_16120231/14952973)，[第三方但对标官方 CANN NEXT 资料]）。多步链式融合（Cast→Mul→Add→Cast）是最大收益场景。

### 2.3 单核/多核编程与同步原语

- **单核内同步**：传统 `set_flag/wait_flag`（PIPE 间事件同步）、`PipeBarrier<PIPE_ALL>()`；**950 新增 BufferID 同步**（`get_buf()`/`rel_buf()`，互斥锁语义，替代 set/wait 配对，内聚性更强、流水线间解耦）[白皮书 §4.1.6 检索片段 + CANN 社区文]。
- **多核（跨核）同步**：`SyncAll` 全核同步指令（SuperKernel 文档中的 feed-sync-all 选项专门处理子算子 SyncAll 次数匹配问题）；**FFTS（核间同步内存）**：`AscendC::SetSyncBaseAddr(fftsAddr)` + 跨核 flag（tilelang-ascend 模板在用）。
- **Stream**：host 侧 `aclrtStream` 维护异步执行顺序（stream 内严格保序），`aclrtSynchronizeStream`/event 做同步；torchair 支持图内多流表达，SuperKernel 内可 `stream-fusion=1` 让纯 Cube 与纯 Vector 算子跨流并行。
- **AI CPU 子系统**：4×Linx816（ARMv8-A 双线程）跑 NPU 侧 OS、页表管理、性能监控等控制类任务，可与 AI Core 协同 [白皮书 §4.2]。

### 2.4 950 代编程模型的整体变化

SIMD/SIMT 混合 VF（`__simd_vf__` 标记，SIMD 为主承担 90%+ 算力、SIMT 处理 gather/scatter/分支）、NDDMA 一行指令完成 transpose/stride/broadcast/slice、CV 融合通道、CANN Next 提供 **CUDA 兼容抽象**（降低 CUDA 代码迁移成本）（[CANN 社区文](https://ascendai.csdn.net/69dce43f0a2f6a37c59f57ec.html)、[51CTO 博客](https://blog.51cto.com/u_16120231/14952973)）。

---

## 3. Mega kernel / persistent kernel / 融合大算子在昇腾上的实践

### 3.1 公开、成体系的官方能力：CANN **SuperKernel**（算子二进制融合）

- **定义**（[Ascend/torchair 官方文档](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)）："SuperKernel 是一种**算子二进制融合**技术，在已编译的二进制代码基础上融合创建一个超级 Kernel 函数，以调用子函数方式调用多个内核函数，达到优化任务调度等待、降低算子头开销的目的。"
- 用户接口：torchair 图模式 `with torchair.scope.super_kernel("sp1", "options")`，将范围内算子编译为一个大 Kernel 内顺序调用 + 自动插同步：
  ```cpp
  __global__ __aicore__ void sk_start_xxx_stop_yyy(...) {
      dynamic_quant(...);   Sync();
      grouped_matmul(...);  Sync();
      dequant_swiglu_quant(...); Sync();
      grouped_matmul(...);
  }
  ```
  （代码骨架来自 [CANN 开发者社区：SuperKernel 技术综述](https://cann.csdn.net/69afcc5554b52172bc603633.html)）
- 支持范围：Atlas A2（910B 系）/A3（910C 系）训练与推理产品；可融合通信算子 AllReduce/ReduceScatter/AllGather/AlltoAll；编译选项含 `feed-sync-all`、`stream-fusion`、`dcci-before/after-kernel-end`（缓存一致性指令精细控制）、`debug-aic-num/debug-aiv-num`（MIX_AIC_配比）。torchair 文档引用了《CANN Ascend C 算子开发》"附录>算子入图（GE 图）开发>SuperKernel 开发"章节，说明 CANN 主线文档已收录（本次未直接抓到该章节页面）。
- **实测收益**（[CANN 社区 SuperKernel 综述](https://cann.csdn.net/69afcc5554b52172bc603633.html)，[社区官方]）：DeepSeek-V3 61 层中 58 层融合为 1 个 SuperKernel，消除 task 调度等待（bs96/3K 场景每层约 18µs × 61 层 ≈ 1.1ms）、task 结束 Cache Flush 浪费、同地址访问排队与 scalar 初始化开销，**整网收益 10–20%**。配套优化：子 kernel 代码段**多副本**缓解多核同地址 ICache/L2 争用；"提前一个 kernel"的 **ICache Preload**（2KB 对齐）。

### 3.2 社区 mega kernel 实践：绑核常驻 + 大算子融合

- 昇腾社区文章《昇腾平台大模型算子高性能优化实践：从绑核到动态融合》（[hiascend forum](https://www.hiascend.com/dev/forum/thread-0297204111998064411-1-1.html)，2026-01）明确提出：**"MegaKernel 通过整合多算子逻辑为单一 Kernel，结合绑核策略让任务常驻物理核，可大幅提升硬件资源利用率"**——基于 910B（16 物理核）+ MindSpore 2.3 + DLCompiler/DLBlas，案例是 **Qwen3-Next AttentionProlog（Matmul+RmsNorm+RoPE）融合为 MegaKernel，实现 MLA（Cube）与 VPU（Vector）并行**。注意：该页正文被 cookie 墙拦截，以上取自搜索摘要，细节未全文核验。
- tilelang-ascend（[GitHub](https://github.com/tile-ai/tilelang-ascend)）把 TileLang DSL 编译到 AscendC，生成 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` 的混合核 kernel——说明"类 Triton 编写融合大算子→AscendC"链路已打通。

### 3.3 AscendC 能否做 persistent kernel 风格？——结论与边界

| 能力 | 昇腾现状 | 依据 |
|---|---|---|
| 单 kernel 内 tile 循环（无重启处理多 tile） | **支持，且是标准范式**（Process() 内 while 循环 + 双缓冲队列） | AscendC 编程范式文档/教程 |
| 多算子合并为一次 kernel 启动（跨 kernel 边界不返回 host） | **支持 = SuperKernel**（GE 图模式，二进制级融合） | torchair 官方文档 |
| 任务常驻物理核、绑核 | **社区实践存在**（绑核 + 核内循环），无独立公开 API 名称 | hiascend 论坛 MegaKernel 帖 |
| CUDA 式 persistent kernel（kernel 常驻 + device 侧全局 barrier + host 仅投喂工作） | **无公开的一等公民 API/官方范式**；最接近的是 SuperKernel + SyncAll 全核同步 + UB Memory/URMA 的 device 侧 load/store/atomic（950 提供硬件原语） | 综合白皮书 §4.1.6/§4.6 与 CANN 文档，未发现官方 "persistent kernel" 条目 |

即：**"一次启动、核内长驻循环、tile/层间无重启"在单算子粒度上是 AscendC 原生能力；"整网一个 megakernel"在昇腾上的官方路径是 SuperKernel（图级、二进制融合），而非手写常驻 kernel**。950 的 UB Memory（load/store/atomic，128TB 共享内存语义）+ CCU + BufferID 同步为 device 侧协作提供了新原语，但公开资料中尚无基于它们的通用 persistent-kernel 框架案例。

### 3.4 案例清单（均可公开查证）

1. torchair `scope.super_kernel`（A2/A3，含通信算子）— [官方文档](https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)
2. CANN SuperKernel × DeepSeek-V3 58 层融合，+10~20% — [CANN 社区](https://cann.csdn.net/69afcc5554b52172bc603633.html)
3. 910B + DLBlas MegaKernel（绑核常驻、Qwen3-Next AttentionProlog）— [昇腾论坛](https://www.hiascend.com/dev/forum/thread-0297204111998064411-1-1.html)
4. tilelang-ascend 生成 MIX_AIC 混合核融合 kernel — [GitHub](https://github.com/tile-ai/tilelang-ascend)

---

## 4. CUDA 生态 megakernel 代表作（对照参考）

### 4.1 可确认的代表作

| 作品 | 时间/出处 | 核心思想 | 来源 |
|---|---|---|---|
| **H100/B200 Megakernel**（斯坦福） | 2025-05 | 把整个 LLM forward（>1B 参数、16bit）放进**单个 CUDA kernel**，一个 warp 负责一层，前向传播 H100 <1ms、B200 ≈680µs；比 vLLM 快 2.5~3.5× | [51CTO 报道](https://www.51cto.com/article/817024.html)、[GitHub (HazyResearch)](https://github.com/HazyResearch/ThunderKittens) 生态 |
| **Mirage Persistent Kernel (MPK)** | 2025-06 | 编译器+运行时把多 GPU LLM 推理自动编译成**单个 megakernel**：一次 kernel launch 内完成全部计算与通信；自动搜索最优 kernel 图 | [mirage-project/mirage mpk 分支](https://github.com/mirage-project/mirage/tree/mpk) |
| **Cohere Megakernel** | 2026 | 单 H100 上单 kernel decode serving（MoE），面向低 batch，消除 per-kernel launch 开销 | [ai-tldr 分析](https://ai-tldr.dev/tools/cohere-megakernel/)、[GitHub cohere-ai/megakernel](https://github.com/cohere-ai/megakernel) |
| **Ada-MK**（TensorRT-LLM MegaKernel 学术版） | 2026-05 | 对 TRT-LLM 开启 MegaKernel 模式，自动 DAG 搜索融合方案，batch-1 低延迟场景收益显著 | [arXiv:2605.11581](https://arxiv.org/html/2605.11581v1) |
| **AutoMegaKernel (AMK)** | 2026 | 把 HF Llama 家族编译成**单个 persistent cooperative megakernel**；VM+micro-kernel 分层 + 静态验证保证无死锁/竞争；支持量化与 agent 驱动的 schedule 搜索 | [Emergent Mind 条目](https://www.emergentmind.com/topics/automegakernel-amk) |
| **UniEP** | 2026-04 | MoE 训练 EP 通信 megakernel：device 侧信号 + 单 kernel 完成 dispatch/combine 与专家计算重叠 | [arXiv:2604.19241](https://arxiv.org/html/2604.19241) |
| **FlashInfer**（融合 kernel 路线） | 2024– | 非整网 megakernel，代表**激进单算子融合 + JIT 生成**：fused RoPE kernel 带宽利用率提升 1.6–3.7×；v0.2  fused MLA decode（Matrix Absorption）；fused sampling（sorting-free）；plan/run 分离 API；split-KV 负载均衡 decode | [FlashInfer 论文 arXiv:2501.01005](https://arxiv.org/pdf/2501.01005)、[v0.2 release blog](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)、[GitHub](https://github.com/flashinfer-ai/flashinfer) |

**关于任务点名的 "single-batch fused decoding"**：FlashInfer 官方口径是 single-request decode/prefill 与 batch decode 全场景 kernel + 融合 RoPE/采样/MLA-decode（见上表来源）；未检索到名为 "single-batch fused decoding" 的独立特性条目，推测主 agent 指的是 FlashInfer 面向单请求/batch-1 decode 的融合 kernel 家族，建议按上表引用。

### 4.2 NanoGate —— 未检索到，如实说明

以 "NanoGate megakernel"、"NanoGate LLM inference"、"nanogate kernel github/arxiv" 等多组关键词搜索（WebSearch 约 4 轮），**均未找到与该名称对应的公开论文/项目**（仅命中生物学 nanogate 等无关结果）。可能原因：名称记误、过于新未入库、或非公开项目。建议主 agent 与用户核对名称；如需替补对照，可用上表 Ada-MK / AMK / Cohere Megakernel。

### 4.3 CUDA  megakernel 的共同设计要点（对昇腾开发的映射）

1. **消除 kernel 间空隙**：launch 开销、调度等待、cache flush——昇腾对应物 = SuperKernel（官方）/绑核 MegaKernel（社区）。
2. **device 侧自调度**：warp-specialization / layer-per-warp / device-side semaphore——昇腾对应物 = SPMD blockIdx 分工 + BufferID/SyncAll + 950 的 UB Memory atomic。
3. **指令供给**：megakernel 代码体积大，ICache 成为瓶颈——CANN SuperKernel 已有同款解法（ICache Preload、代码多副本）。
4. **通信进 kernel**：MPK/UniEP 把 NCCL 级通信融进 kernel——昇腾对应物 = SuperKernel 融合 AllReduce/AllGather/AllToAll + CCU 硬件卸载。
5. **风险**：编译时间长、调试困难、收益集中于低 batch/小模型 decode（大 batch 下 launch 开销占比小）——Ada-MK 论文与社区评论均指出此点，对 950 上跑 Qwen3.8-Flash（decode 带宽敏感型）尤其相关。

---

## 5. 与本项目（Ascend mega kernel + Qwen3.8-Flash-Next-MXFP4）的直接关联

- 目标模型为 MXFP4 量化：**950 Cube 原生 MXFP4（4×BF16 算力）+ L0C→UB 随路反量化**，权重带宽省 4 倍，与 DT 的 4TB/s HBM 或 PR 的大容量 HBM 匹配；Qwen3-Next 系（80B-A3B 及 -Coder-Next）已有 MXFP4 公开 checkpoint（[AMD Quark 版](https://huggingface.co/amd/Qwen3-Coder-Next-MXFP4)）与 vLLM-Ascend 适配记录（[vllm-ascend release notes](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/release_notes.html)），说明"Qwen3.8-Flash-Next-MXFP4"大概率是下一代同类模型，软件栈路径可复用。
- 950 上实现 mega kernel 的现实路线：**torchair SuperKernel（图级）→ 自定义 AscendC 大算子（MIX_AIC_1_2 混合核 + CV 融合 + RegBase 链式向量计算 + NDDMA）→（探索级）基于 UB Memory/SyncAll 的跨核 device 侧调度**。

## 6. 信息来源与获取失败说明

**直接抓取失败的项**（已用替代来源）：
1. 《昇腾950 NPU 架构白皮书》PDF 正文经 FetchURL 两次尝试均截断于 §4.1.5（NDDMA）；§4.1.6（BufferID 同步）及之后的 L2/STARS2.0/URMA/CCU 细节改由搜索引擎摘要（白皮书原文片段，[示例](https://hifloat.gccorg.com/pdf/昇腾950 NPU架构白皮书.pdf)）与[搜狐转载解读](https://www.sohu.com/a/1034057245_121752158)交叉补齐。
2. 昇腾论坛 MegaKernel 帖（thread-0297204111998064411）正文被 cookie 墙拦截，仅获得搜索摘要（§3.2 已注明）。
3. NanoGate 无任何公开命中（§4.2）。

**官方 vs 间接来源分界**：§1.2–§1.5 的 950 规格、§2.1/§2.2 的 API、§3.1 的 SuperKernel 均为官方/官方文档；§1.6 中 910B/910C 的算力/HBM 数字、§1.1 的 PR 衍生版本（112GB/1.4TB/s）为第三方来源且彼此存在出入，已在表中标注。950 的 **UB 双向带宽 2016GB/s、L0C 256KB、Cube:Vector=8:1、UBoE 200GB/s** 来自 CANN 官方社区频道文章，可信度高于一般媒体但仍非白皮书正文原句。
