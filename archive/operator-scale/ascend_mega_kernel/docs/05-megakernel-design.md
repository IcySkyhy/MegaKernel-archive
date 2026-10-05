# Mega Kernel 设计方案 v1.2（已确认决策 + 开发策略）

> 记录日期：2026-09-26（v1 提出设计 → 当日经两轮用户确认形成 v1.2）。
> 硬件事实来源：cannbot-knowledge `target_ascend950pr.md` 平台卡、asc-devkit 头文件。

## 1. 设计目标

提升 **decode TPS**（最终兼顾 prefill；目标模型 Qwen3.8-Flash-Next-MXFP4，单卡 Ascend950PR）。

## 2. 总体形态（已确认）

- **粒度**：不是整个网络一个 kernel，而是**每个 transformer layer 一个 kernel**（48 层 → 48 次启动/decode step）。
- **核配置**：整个 persistent kernel 以 **mix mode（AIC:AIV = 1:2）** 启动，用满所有核。
  - 本机 28 AIC + 56 AIV（PG 降频 binning 版，满配 die 为 32 AIC）——**核数必须 API 获取，禁止硬编码**：`PlatformAscendC::GetCoreNumAic()` / `GetCoreNumAiv()`，blockDim 用 `CalcTschNumBlocks(aicCoreNum, aivCoreNum)` 计算；mix 启动 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`。
  - **kernel 内部的分工是自由的**：同一 kernel 内可以划分 AIC-only 区域、AIV-only 区域、mix 区域。
- **m 的范围**：prefill 与 decode 都要照顾；**验收测试 m=1（decode）与 m=4097（prefill）**。**注意（`docs/17` §7 附则 1）**：m=4097 与 decode-ctx4096 的数值验收基准是**自建稠密 causal 参考**，**与官方 QSA 输出存在固有差异**（官方只 attend 约一半历史）——详见 §8 注*。
  - schedule 与 m 无关：token 维度按 tile 循环（tile 大小静态），tile 数由 main scalar 在 kernel 内根据 m 动态计算，m=1 走单 tile 快速路径。
- **同步约束（硬性要求）**：
  - 核内 pipeline 同步**只用 BufferID**（`get_buf`/`rls_buf`），**禁用 set_flag / wait_flag 系列**表达跨 pipe 数据可见性（理由：set/wait flag 是单向同步；BufferID 是 mutex 语义、双向同步）。**实测精化（M1 reviewer 真机裁决，2026-09-26，待用户追认）**：① 跨 pipe 数据交接（如 MTE2→V）用 **BufferID release**（`RlsBufInternal<pipe,false>` + 对侧 get）即可承载数据可见性（M1 reviewer 真机 10/10 PASS 实证，当时档位为 `RlsBufInternal<pipe,true>`）——`false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」、`true` = `NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**、`true` 额外等此前同 id 的释放（更保守），**全项目 release 一律 `false`**——但公开 `Mutex::Lock/Unlock` 把 mode 硬编码为 0，必须用 Internal 原语；② **同 pipe buffer 复用**（背靠背同类拷贝）任何模式 BufferID 都不保序，**必须 `PipeBarrier<PIPE>`**；③ 确无法用 BufferID 表达的残留场合（如有）再逐例上报裁决，不默认放开 set/wait flag。
  - **PIPE_S 同步限制（硬性要求）**：除**有值依赖**的场景外（如 main scalar 需要消费某个计算结果做标量运算——读回数值、地址计算、tiling 计算），**禁止把同步挂在 PIPE_S 上**。理由：PIPE_S 同步会阻塞 main scalar 继续发射异步指令，摧毁"scalar 跑在指令前面"的跨 op 预取重叠机制。数据搬运与计算之间的次序一律挂在对应数据通路 pipe（MTE1/MTE2/VEC/CUBE/FIXP/MTE3）或用 BufferID 表达。**例外（M10 实证）**：跨核链式同步中"wait 之后紧接 set"时，wait 必须挂 PIPE_S 才能保证 set 的次序（CrossCoreSetFlag 是 drain 触发；CANN SyncAllImpl 同款做法）。
  - **资源全静态自管理（硬性要求）**：尽量不使用 AscendC 资源管理函数（TPipe / TBuf / TQue / TBufPool / AllocTensor 等）——所有 **BufferID、CrossCore flagId、buffer 地址全部由我们自己管理，编译期静态分配**（一份静态资源表：BufferID 枚举、flagId 枚举、UB/L1 地址偏移常量）。动态 shape 不引入动态分配，只用循环 + mask 处理。**基础 API 一律用裸指针（`__ubuf__` / `__cbuf__` 地址）调用**，不用 LocalTensor 封装。
  - 核间同步用 **CrossCore 系列**，模式选用规则（用户明确）：
    - **mode 2**：单个 AIC ↔ 它所属 AICore 内的 2 个 AIV（及反向）——适用于 **AICore 内 CV 直连的数据交接**，典型如 attention 的 **FIXP L0C→UB**、cube 输入的 **UB→L1** 这类点对点（核组内）同步；1 set(AIC) 配 2 wait(AIV)，或 2 set(AIV) 配 1 wait(AIC) 算配对；
    - **mode 0**：**GM 介导的 op 间交接必须用 mode 0**——前一个任务输出到 GM、后一个任务从 GM 读时，不是特定一个核对一个核（前后 op 的 tiling 不一定一一对应），本质是 all-to-all。**mode 0 仅同类型**（"全部 AIC 之间"或"全部 AIV 之间"二选一，不支持 AIC 组 set、AIV 组 wait 的跨类型用法）；跨类型 all-to-all 的标准组合（SyncAll mix 序列）：AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）→ AIC→AIVs（mode 2）→（如需）全体 AIV barrier（mode 0）。
  - **CrossCore 硬约束（M2 调研实证）**：
    - **Set 侧 pipe：模式 0/1/2 禁止 PIPE_S/PIPE_ALL**，只能 PIPE_V/M/MTE1/MTE2/MTE3/FIX（模式 4 可加 PIPE_S）——"同步挂在计算 pipe 上"是强制的；
    - **Wait 侧 pipe 在 950 上生效**：只阻塞指定 pipe 的后续指令下发（已下发照常执行）——无依赖的 MTE2 预取可从同步旁流过，但必须用窄 pipe（PIPE_M/PIPE_FIX 等），不用默认的 PIPE_S；
    - 同一核连续多个 CrossCoreSetFlag **硬件不保证执行顺序** → 相邻不同同步点用不同 flagId；
    - 同 flagId 跨模式复用（同核）须前一模式全部 set/wait drain 完；numBlocks ≤ 物理核数；模式 0 建议 batchmode 独占核。
- **API 层级**：**不用任何高阶 API**（不用 Matmul 高阶 API），cube 侧也用基础 API（mmad 系列基础指令）手写。**指令级 vector intrinsic 例外清单（tower 2026-09-26 裁定，取代旧白名单）**：**只有 `Sort32` 与 `MrgSort`** 属例外（排序单元 primitive，性质等同搬运/矩阵类；依据：CANN 9.1.0 头文件穷举无 `Reg::` 等价物 + 官方 donor 同为 memory-based，见 §6.1 规则 ⓒ），每处使用必须附一行注释指明该依据。**旧白名单已收窄/撤回**：`WholeReduceMax`（改用 `Reg::Reduce`）、`Concat`（3510 经典 `Concat` 是 no-op）、`Extract`（厂商 3510 的经典 `Extract` 本身即 `__simd_vf__`，官方 donor 也已手写 VF ⇒ 须改造）、`MrgSort4`（3510 为 deprecated 空函数体，禁用）——一律**移出**；`Reg::LocalMemBar` 属寄存器侧屏障，本就在 §6.1 合规范围内。**完整规范与豁免清单见 §6.1**。被禁的只有框架级资源管理（TPipe/TBuf/TQue/TBufPool/AllocTensor）与 Matmul 高阶 API。
- **数据 layout**：**kernel 直接支持原始 layout**——权重直接读取 HF safetensors 的原始排布（uint8 nibble 打包 MXFP4 + e8m0 scale，参考 CANN 官方 mxfp4 样例的消费方式），输出也保持原始 layout；**不做离线转换**。

## 3. 收益来源

1. 减少 host 下发 kernel 开销（aclgraph 理论上也能做到，非唯一收益）；
2. 减少 device 侧每个 op 的启动/结束低效时段——kernel 开头 scalar tiling、MTE2 预热 ramp、结尾 MTE3/FIXP drain 的带宽效率都低。融合成 layer kernel 后，这些低效段只出现在 layer 边界（48×/step）而非 op 边界（数百×/step）；
3. 目标：层内通过跨 op 预取流水，全程保持 HBM 带宽效率。

## 4. 核心原理：main scalar 异步发射 + 跨 op 预取

昇腾架构中 main scalar 负责下发异步指令；异步指令的**同步在其对应 pipeline 内完成**，不阻塞 main scalar 继续执行。op1 输出期间 main scalar 即可发射 op2 的无依赖输入（权重）搬入指令。

示例（C=op1(A,B)，E=op2(C,D)，op2 依赖 C、D 无依赖）：

```text
MTE2 COPY IN A
MTE2 COPY IN B
OP1
MTE3 COPY OUT C
MTE2 COPY IN D // 在 MTE3 COPY OUT 之后发射，但 MTE3 等 OP1 依赖，
               // COPY D 只需 buffer 空间足够即可执行
SYNC
MTE2 COPY IN C
OP2
MTE3 COPY OUT E
```

硬件依据：950 = 6 pipeline（CUBE/VEC/MTE1/MTE2/MTE3/FIXP）+ 独立 Scalar 单元（乱序、每周期最多发射 5 条指令）；BufferID 见白皮书 §4.1.6（get_buf=acquire、rel_buf=release）；CrossCore 同步 2.0 由 STARS2.0 调度器支撑（flagId API 窗口 0–10、每 id 计数 ≤15，硬件 flag 池远大于此窗口）。

**推论**：GM 边界的 mode 0 同步应插在计算 pipe 上，这样无依赖的 MTE2 权重预取指令可以从同步旁边流过去（同步只阻塞依赖方，不阻塞搬运）。

### 4.1 同步语义实证结论（2026-09-26，M2 survey，来源：CANN 9.1.0 头文件 + cannbot API 文档 + 生产 donor）

- **BufferID 范式**：`Mutex::Lock/Unlock<pipe>(id)`（= get_buf/rls_buf）；token=(核,bufId) 粒度，AIC/AIV 空间独立；惯用法：互斥交接 `get(false); ...; rls(false)`，跨 pipe 交接同步点 `get(true); rls(true)`（档位为 M2 调查时点；**两种模式都等本 pipe 已发射指令落地**，`true` 额外等此前同 id 的释放 ⇒ 更保守 —— 见 §2；**全项目 release 现一律 `false`**）；同 pipe `rls→get` 重获取立即可得（同 pipe FIFO 本已保序）；同 id 连续 get 不 rls = 自死锁。**pbid（release_pbid）是编译器内部机制，用户不可见、不接触**。
- **FIXP L0C→UB 直写（CV 融合数据面）**：`Fixpipe(dstUB, srcL0C, FixpipeParamsArch3510)` 存在；AIC 的 FIXP 直写配对 AIV 的 UB；**dualDstCtl=1 一次按 M 维拆分同时写两个 AIV 的 UB**（M 必须为偶数），单目标用 subBlockId；支持 NZ→ND + 随路量化（ROW_MAJOR 配置）。完整交接范式（donor `kv_quant_scfa_block_cube.h:300-320`）：`get_buf(PIPE_FIX,id,false);rls_buf(...)` 等 FIXP 上空 → CrossCore mode 2 等 AIV 消费完 → Fixpipe → `get_buf(PIPE_FIX,id,true);rls_buf(...)` 等写完 → CrossCore mode 2 通知 AIV。
- **SSBuf**：`GetSsbufBaseAddr()` 返回专用窗口；MIX 1:2 下 AIC+2 AIV 共享 3KB；仅 32B 对齐读写；数据有脏；无内建通知需配 CrossCore。定位：**小控制信息**（tiling/标量/指针）传递，不适合数据面。
- **核函数类型约束**：AIC_ONLY/AIV_ONLY 不开启调度模块、CrossCore 不可用——layer kernel 必须 MIX 类型（本设计满足）。
- **官方同 id 跨两 pipe 用法（M128 补 pin，非设备、非自指）**：官方双缓冲指南的同步关系表给同一 `outputMutexId` 在 PIPE_V（写）与 PIPE_MTE3（搬）上交接（`asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md:189-190`）；样例见 `asc-devkit/examples/02_simd_c_api/02_features/01_reg_vector_compute/00_add_double_buffer/add_double_buffer.asc:151-169` 与 `asc-devkit/examples/01_simd_cpp_api/03_basic_api/05_sync_control/mutex/mutex.asc:76-100`；框架 impl 同一 `bufId` 跨两 pipe 见 `asc-devkit/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h:832-835`。收尾 release 落在搬出 pipe（UB→GM 为 PIPE_MTE3，`enable_double_buffer.md:172-174`），官方释放语义逐字见 `asc-devkit/docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_unlock.md:34`（「指定流水的前序指令执行完成后，根据`mutex_id`释放对应Mutex。」）。**命名窄结论**：官方文档树内以 `grep -rn -i "drain"` / `grep -rn "排空"` 检索未命中名为 “drain release” 的 API ⇒ 本仓 `BufRelease<PIPE>(id)` 是本仓私有包装、不是官方 API 名。完整 `文件:行` 与逐字引文见 `docs/06-m0-bringup.md` §4.1–§4.2（其中「官方 `asc_mutex_execute_mode` 的 BLOCK/NON_BLOCK 命名与 true=drain 相反」这一未决项已由 M126 真机对照结掉：底层映射**一一对应、无反转**（`BLOCK`=`false`、`NON_BLOCK`=`true`），见 `probe_ub_bufid_tail/README.md` §0.1 第 11 条 / §8.1；两种模式都等本 pipe 已发射指令落地，全项目 release 一律 `false`）。

### 4.2 aclgraph（图捕获）：排期与已知约束（M87 登记，2026-09-27）

**用户裁决（逐字，两条）**：「**要再套一层 acl graph**」；「**aclgraph可以记到文档中，但是不需要现在做，这个是最后做的**」。
⇒ **排期：最后做**。本节只登记**约束与开放问题**，不实现、也不改变当前「host 循环 48 层」的方案 —— §7 的「aclgraph 层后续再套」、§8 的 M4「aclgraph 后补」是同一裁决的早期落点。

**它捕获什么（官方定义与官方 example）**：

- **CANN 官方文档《ACLGraph》**（AscendCL 运行时；链接由 asc-devkit example 给出 = `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/programug/acldevg/runtime_doc_dev_0045.html`）：在 `aclmdlRICaptureBegin` 与 `aclmdlRICaptureEnd` 之间的任务**不立即执行**、被暂存进模型运行实例，只有 `aclmdlRIExecuteAsync` 才真正执行。原文：「**当前提供了“捕获Stream任务到模型中、再执行模型”的acl接口**，简称ACL Graph，在aclmdlRICaptureBegin和aclmdlRICaptureEnd接口之间，所有在指定Stream上下发的任务不会立即执行，而是被暂存在模型的运行实例中，只有在调用aclmdlRIExecuteAsync接口执行模型时这些任务才会被真正执行。」
  同页 `须知`（**规格性约束**）：「**本功能为试验特性，后续版本可能会存在变更，不支持应用于商用产品中。**」
- **官方 example（asc-devkit，`/workspace/asc-devkit/examples/01_simd_cpp_api/02_features/00_framework/04_aclgraph/`）**：`README.md` 原文「本样例以Add算子为例，**展示如何捕获Ascend C `<<<>>>` 核函数调用及其前后的数据拷贝任务**，并分别演示单流线性任务捕获和双流事件依赖捕获。」；`add.asc` 的 `RunSingleStreamCapture` 在 `aclmdlRICaptureBegin`/`aclmdlRICaptureEnd` 之间提交 `aclrtMemcpyAsync` 与 `add_custom<<<numBlocks, 0, mainStream>>>`；双流场景用 `aclrtRecordEvent` + `aclrtStreamWaitEvent` 建依赖（README：「两个流之间通过 `aclrtRecordEvent` 和 `aclrtStreamWaitEvent` 建立依赖。」）。
- **API 面**（本机 CANN 9.1.0 头，`/usr/local/Ascend/cann-9.1.0/x86_64-linux/include/acl/acl_rt.h`）：`aclmdlRICaptureBegin` / `aclmdlRICaptureGetInfo` / `aclmdlRICaptureEnd` / `aclmdlRICaptureThreadExchangeMode` / `aclmdlRICaptureTaskGrpBegin|TaskGrpEnd|TaskUpdateBegin|TaskUpdateEnd` / `aclmdlRICaptureToModelRIBegin`；capture mode 枚举 `ACL_MODEL_RI_CAPTURE_MODE_{GLOBAL,THREAD_LOCAL,RELAXED}`。runtime 层同名接口在 `/usr/local/Ascend/cann-9.1.0/x86_64-linux/pkg_inc/runtime/runtime/stream.h`：`rtStreamBeginCapture` / `rtStreamEndCapture` / `rtStreamGetCaptureInfo`。

**已知需要遵守的约束（只写有出处者；出处分「官方规格/example」与「社区经验库」两类，勿混）**：

1. **捕获粒度 = Stream 上的运行时任务，含整个 `<<<>>>` kernel 启动**（官方 example）：核函数启动被整体作为一个 task 捕获，kernel 内的 `PipeBarrier<PIPE_ALL>()`（`add.asc` 的 `add_custom`）**随 kernel 一起被记录、不在图里单独建模**。⇒ 我们「一层一 kernel」的形态天然可被捕获（一次 launch = 一个 task），**不需要**为了入图把层内拆成多个 op。
2. **跨流依赖必须显式表达**（官方 example）：单流是线性任务链；跨流**必须**用事件（`aclrtRecordEvent` + `aclrtStreamWaitEvent`）建依赖 —— README 原文「`aclmdlRI` 不只记录单个流上的任务顺序，也记录跨流的事件依赖」。⇒ 将来若把层 kernel 拆多流，每条「生产者→消费者」跨流边都要在 host 侧显式建事件。
3. **捕获区内不得有 host 侧同步 / host 数据依赖**（**社区经验库**，非官方规格）：`.item()`、`.tolist()`、`.cpu()`、`.nonzero()` 这类 device→host 同步会**打断图捕获**（`/workspace/cannbot-knowledge` 的 `graph_break_elimination_host_sync_unsupported_ops.md`）。⇒ 层 kernel 的 host 启动器在捕获区内**不得**依赖动态 tiling / 读回设备数据。
4. **同步范围要收窄到当前流**（**社区经验库 + vllm-ascend 落地**）：replay 是当前流上的重放；跨图步串行化应同步**当前流**，不要用设备级 `synchronize()`（后者会把无关背景流一并卡住）。出处：`/workspace/cannbot-knowledge/knowledge/model/inference/guides/experience/interventions/graph_mode_multi_stream_cross_stream_sync_api.md` §「跨"步"串行化的 replay 前同步：收窄到当前流，别用设备级同步」（原文：「**设备级 `torch.npu.synchronize()` 过宽**：它等**整个 device 上所有流**完成」「**当前流同步 `torch.npu.current_stream().synchronize()` 恰好够用**」）；落点 = `vllm-ascend` 的 `ACLGraphWrapper.__call__`（`vllm_ascend/compilation/acl_graph.py`，`entry.aclgraph.replay()` 前调 `torch.npu.current_stream().synchronize()`；本地检出在第 256 行，行号随代码变动）。

**开放问题（以下在本地 CANN 文档 / 官方 example / 社区知识库里均未找到明文，禁止当事实用）**：

- **aclgraph 对“kernel 内同步”的交互要求**：具体到本项目的 **CrossCore flagId**（§4.1 / §6 表）、**BufferID（`Mutex`）**（§4.1 / §6 表）、`SyncAll`，**没有找到**任何出处说明图捕获/重放对它们有额外要求。已查渠道：CANN 9.1.0 本机头（`acl_rt.h` / `runtime/stream.h`）、asc-devkit 官方 example `04_aclgraph`、`cann_ops_transformer/docs/zh/lightning_indexer.md`（**本机已安装包 `cann_ops_transformer/docs/` 内唯一**提到 aclgraph 的一处，只写「该接口支持单算子模式和TorchAir图模式（aclgraph）调用」；源码仓 `ops-transformer` 另有多处含同一句 `aclgraph` 的算子文档，同样不谈核内同步交互）、`/workspace/cannbot-knowledge`（社区知识库）——**均只讲 Stream 上任务的捕获/重放，没有讲与核内同步的交互**。
  官方 example 只演示了「捕获一个内部用 `PipeBarrier` 的核函数」这一个**正例**，**不能**据此推断 CrossCore / `Mutex`(BufferID) 也一定无约束。⇒ **实施 aclgraph 前必须在设备上就这几个同步点单独验证**；本 mission 未做该验证，也不在文档里替它下结论。

## 5. 分工与并行策略（已确认）

- **单卡**；MoE 中**一个专家的 GEMM 不必一个 AIC 完成**——把 x 个专家分到 y 个核（专家维 × K/N 维二维切分）。
- decode bs=1 时 top-10/512 专家活跃：静态 schedule 按**专家槽位**（第 i 个被激活的专家）划分而非按专家 id，运行时先用 AIV/SIMT 按路由结果做 token 重排，再 grouped GEMM（donor：`ops-transformer` gmm/mega_moe 的"先重排、再静态分组"结构）。
- 层内区域划分自由：GDN 层（36/48 层，vector-heavy）中 AIC 可用于权重预取或闲置；full-attn 层中 AIC 承担 QK^T/PV 等 cube 工作。
- prefill（m 大）时 GDN 线性注意力按 chunk 顺序扫描（recurrent 依赖），full attention 按 FA 风格 m-tile 流水——都属于 op 内部实现细节，不影响 layer kernel 框架。

### 5.1 Op 模块契约：暴露 pipeline 依赖（用户提出）

每个 op 模块对外暴露三类信息，供顶层 schedule 使用：

- **输入依赖**：本 op 从哪些 buffer 读、这些 buffer 由哪些 **pipeline** 生产（如 MTE2 搬入的权重、上游 op 的 VEC/FIXP 输出）；
- **输出声明**：本 op 向哪些 buffer 写、由哪些 pipeline 完成（如 MTE3 写 GM、FIXP 写 UB）;
- **使用的 pipeline 集合**：模块内部用到 MTE1/MTE2/VEC/CUBE/FIXP/MTE3 中的哪些。

顶层依据这些声明做两件事：**分配同步 BufferID**（每个跨 pipeline 的生产者-消费者交接一个 mutex id）和**排布指令顺序**（无依赖的搬运指令提前发射，实现跨 op 预取）。

### 5.2 资源分配：全局打平，不做完全模块化（用户提出并已确认）

前后两个 op 融合的前提是后一个 op 知道前者的内存分配布局，否则后者的搬运会踩前者还没消费完的数据。因此：

- **内存不模块化**：单一**全局编译期资源表**覆盖整个 layer kernel；每个模块只声明 footprint（大小/对齐/生命周期），buffer 地址与 BufferID 由顶层资源表静态分配、以参数形式注入模块，模块不自持有内存；
- 模块化的边界 = op 的计算逻辑 + pipeline 依赖声明；资源层面全部打平考虑；
- 后续 op 复用前 op 的 buffer 位置（节省片上内存）由全局表按生命周期规划，等价于一个编译期 linear allocator。

### 5.3 host ↔ kernel 边界契约：确定性要一起定义（M36 教训，2026-09-26）

**"多读的行由 host 保证可读"这类契约，必须连"多算出的列/结果的确定性"一起定义**，否则非确定性会从 L1 残留里渗进来——M36（hc mixer）实测：`down GEMM` 最后一个 N-tile 只搬 4 行 `wInj` 时，`OH` 的 padding 列取自 L1 残留（有限但随调度变化），**同一二进制的 dump 不是逐字节一致的**；改为搬满 16 行 + host 契约「`wInj` 行 [4,16) 为 0」后 padding 列恒为 0，15/15 dump 逐字节一致。⇒ 写任何"读到的多余行/多余列由 host 置零"式契约时，**必须同时给出该契约下"多算出的列/结果"的确定性依据**（这里是"padding 列恒 0"），否则会破坏 `docs/17` §4 的确定性前提。

## 6. 关键硬件约束

> **本节已经用户逐条审核（2026-09-26）并重构**：6.1=规格/API 约束（非硬件 bug）、6.2=实测硬件行为（必须遵守）、6.3=待复核（禁止当事实引用）。原"实测 quirk/bug"清单中大部分条目实为 API 语义、约束或我方代码缺陷，已按用户裁决重新归类。

| 资源 | 规格 | 含义 |
|---|---|---|
| L1 Buffer / AIC | 512KB（1 AIC + 2 AIV 共享） | cube 权重预取主 buffer；预取按 tile 滚动 |
| UB / AIV | 256KB 物理 / 248KB 可用（`GetCoreMemSize(UB)` 获取，禁硬编码） | vector 侧激活 + 预取 tile 预算 |
| L0A/L0B / AIC | 64KB + 4KB MX | MXFP4 走 MX buffer |
| L0C / AIC | 256KB（256×256 tile） | cube 累加驻留 |
| SSBuf | 3KB，AIC↔AIV 核间通信 | 小控制信息传递 |
| Scalar I-cache | AIC 32KB / AIV 16KB | 当前阶段暂不考虑，用循环结构自然规避 |
| CrossCore flagId | **每核 16 个（0-15），模式 4 下 AIC 侧 0-31**；"每 id 计数 15 次" = 每（核,flagId) 一个 4bit 计数器（0-15），未配平累计超 15 报错。SyncAll 占 11-14(+28/29)、Matmul 高阶 API 占 0-7；**本项目不用 SyncAll/高阶 API → 0-15 全可用**（早期资料"0-10"是保守窗口，已修正） | 层内多处同步需规划复用；相邻同步点必须用不同 flagId |
| BufferID | **每核用户可用 0-27**（MAX_TBUFID=31，28-31 保留给框架）；token 初始全部空闲；release 的 mode：`false` = CANN `ASC_LOCK_BLOCK`（默认，「阻塞」）、`true` = `ASC_LOCK_NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**（不是「立即生效」），`true` 额外等此前同 id 的释放 ⇒ `true` 更保守；**全项目 release 一律 `false`**；不可重入 | AIC 与 AIV 的 token 空间独立，pipe 不命名空间——不同 pipe 并发用同一 id 会互踩 |

### 6.1 规格 / API 约束（非硬件 bug；经用户逐条审核，2026-09-26）

| 项 | 内容 | 依据 |
|---|---|---|
| DMA/UB 对齐 | 大部分 API 要求 32B 对齐；**例外**：gather/scatter、unaligned load/store、单元素 load/store 支持非对齐 | M9/M16 实测 + 用户裁决 |
| **计算路径一律走 vector function（`__VEC_SCOPE__` / `__simd_vf__`）；禁止 memory-based 的向量计算 API**（用户裁决，2026-09-26） | 见本节末「**§6.1 计算路径规范：六条 + 豁免清单**」——六条规则（ⓐ 计算必须走 vector function；ⓑ 搬运/矩阵类例外；ⓒ 无寄存器等价物的原语例外＝仅 `Sort32`/`MrgSort`；ⓓ 控制路径不在禁令范围；ⓔ 落盘通路＝落 GM 一律走 DMA、标量 pipe 直写 GM 不构成可见性契约；ⓕ 计算三分口径＝矩阵乘法一律 cube、其余一律 VF、scalar 只做控制流）+ 探针/host 豁免清单。**判据落在被调函数上，不按 API 名**；`LocalMemBar` 属寄存器侧屏障，随 ⓐ 在 VF 内使用 | 用户原话："**你应该都用VEC_SCOPE aka simd vector function，不应该使用memory base的API**"。理由：① 用户指出的"**SIMD Vector function 的 load store 支持非 32B 对齐**（gather/scatter、unaligned、单元素）"那一族能力只在 register 侧；② memory-based 的中间量必须落 UB，白吃 UB 带宽与缓冲压力（M36 的 hc 段 4 条流整段驻 UB 才有收益）；③ M24 实测：经典 `Cast` + 标量↔向量交接出过真 bug，换 VF 后消失（见下方位宽转换行）；④ 出 507015 的那次 `Duplicate(…,1)` 本身就是 memory-based ⇒ 按本标准它根本不该出现在计算路径里。**存量模块先审计后分批改造**（审计见 `docs/18-vector-api-audit.md`；实际存量仅 `m3` 生产级 + `m0` 探针 + `m8` 一处存疑，分工见 M47/M48） |
| BufferID 用法 | 公开 `Mutex::Lock/Unlock` 的 mode 硬编码为 0——**就用 mode 0** | M1 reviewer 源码 + 用户裁决 |
| **BufferID 只覆盖核内**（用户澄清，2026-09-26） | **BufferID（get/rls）是核内 token，管不了跨核可见性**。凡"AIV 产出 → AIC 消费"的交接（典型：P 的 UB→L1→L0A 路径），**必须用 CrossCore 同步**（AIV 侧在拷贝完成后 set、AIC 侧 wait），漏了就会"AIC 读太早、数据不在"（表现为下游恒 0，且难以从数值上定位）。对照：Q/K 这类 GM→L1、无跨核作者的路径不需要这道同步 | 跨核数据交接一律 CrossCore；核内 buffer 生命周期才用 BufferID |
| `LocalTensor(pos, addr, size)` | addr 单位=字节，size=元素数 | M1 实测 + 用户裁决 |
| `GetBlockNum()` 在 mix kernel 内的语义（M22 实测） | **`__mix__(1,2)` 下 AIV 视角 `GetBlockNum()` 返回 AIC 数（28），不是 AIV 数**——AIV 侧要用核数需 `×2`（或在 host 用 `GetCoreNumAiv()` 传入） | 复制自单核 AIV 算件的代码必须先改此处；M22 的 m9/m4 段即因此改造 |
| **`LoadL0_2D` 类装载的目的偏移**（M24 实测，2026-09-26） | **该 API 只接受"源偏移"，目的恒从 `l0Dst` 的位置 0 开始写**——若想让数据落到 L0A/L0B 的某个槽位（如 Q 常驻头部、P 进 8KB 起），**必须通过 dst tensor 切片表达**（传 `l0Buf[off]`），否则会静默覆盖 dst 头部。M24 的症状是 P 覆盖 Q、mmad 从槽位读到未初始化内存 → 下游恒 0，且"换任何 B 侧参数都不出数" | dst 偏移用 `l0Dst[off]` 切片；装载类 API 的参数里先确认"哪些是源、哪些是目的" |
| **装载系参数的"静默不生效"模式与通用判据**（M24 首记 → **M27 标定后订正归因**，2026-09-26） | **订正：没有任何字段被证实「配了也无效」。** 原现象按两类归因（+ 一类通路问题）重写：① **参数语义/单位与文档的写法不同**——参数**有效果**，只是文档的 per-axis 单位（`M 轴 16 元素` / `K 轴 32 字节`）在 bf16 上**正好等于**「1 个 M 分形行 / 1 个 K 分形列」，实测的单位就是**分形**（`mStep`/`kStep` = 分形个数；与官方文档的对照口径 = **1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节）**，唯一真分歧 = `srcStride`/`dstStride` 的轴向措辞，见本节表后的「M27 标定 4 条」）；此前那句「目的偏移不生效」同属此类：`LoadL0_2D` 的目的偏移只能用 `l0Dst[off]` 切片表达，见上一行；② **越界或 NOP 组合读的是"压根没写过的槽位"**——**越界组合的读数一律不可用于推断**（前提）：其中越界数据落进回读窗口的那些在 5 次独立运行里内容非确定（如 `srcStride=3`、`mStart=16`；逐对分组见 `m16_load_geom/evidence/geom2d_determinism.txt`），落在窗口外/稳定零区的那些读数**确定**但仍在源范围之外、同样不可推断（如 `kStep=4` —— 本探针源只有 2 个列分形）；`mStep=kStep=0` 则是 NOP（读回全 0）；③（**不是参数问题而是通路问题**）**3D `LoadData3DParamsV2` → L0B 在 M27 的 29 组 conf 的回读窗口内未观察到任何写入**（限度：观察窗口只到 L0B 行 0–3 即 mmad `n<64`，29 组里没有一组"已知能写"的正对照 ⇒ 严格说只能断言"未观察到写入"；文档的两条硬约束——该通路自动转置、L0B 下 `enTranspose` 无效——是"这条路不该走"的独立依据；见 `m16_load_geom/README.md` §3.6）⇒ M24「3D 源偏移不生效」应改述为"**回读窗口内未观察到该通路写入**" | **通用判据仍成立，但要加前提**：① 对可疑参数**传两个不同值，看结果是否变**；② **只有"参数保证落在源范围内"的良构 conf 才能用于推断语义**（越界组合的读数不可解释）；③ 读目的 buffer 前必须清零（见 §6.2 新增行）。装载几何不要靠"外推 donor 参数"，先做最小标定。标定源：`m16_load_geom/README.md` §3.3–§3.6（**M27 标定，已随 `eeab864` 合入 main**；引用时的分支 tip 为 `de68fe3`，其后另有 `2e54a89`/`cdbfa20` 两次复评修正） |
| SIMD VL | **固定 256B**；element 数按位宽：F32/S32=64、S16=128 …（此前"Arange 只填 64 lane"的表述按此修正） | 用户裁决 |
| 位宽转换 Cast | 不同位宽间转换是 1↔2 寄存器的组合：bf16→fp32 需 even/odd 各一条 cast（需保持原序时再 deinterleave）；fp32→bf16 需先 interleave、even/odd 各一条输出、再 select+mask merge（**或用 store 的 pack 模式——2026-09-26 由 M35 确认**：官方在同类位置（norm+partial-RoPE 的 bf16 落盘）用的就是 `Cast<T, float, CAST_FP32_TO_FP16>` + `Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>` 一条完成，见 `ops-transformer/posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/kv_rms_norm_rope_cache_regbase_base.h:226-231`；**手写整数 RNE + `DeInterleave` 是多余的自造动作**，M35 曾因此把 ck 写成 NaN——那条"`DeInterleave` 语义与文档不符"的结论**已撤回**，属"**选错落盘指令**"而非指令语义问题）。**实证教训（M24，2026-09-26）**：AIV 上的 fp32↔bf16 转换与逐行规约**不要用经典 API**——经典路径下 softmax 的 sum 高 1.086~1.111×、P 出现"偶位 0/奇位 >1"交错；换 **RegBase VF**（`__VEC_SCOPE__` + `Reg::LoadAlign/StoreAlign/Reduce/Cast/LocalMemBar`，行状态用 `DIST_FIRST_ELEMENT_B32` 存、`DIST_BRC_B32` 广播取，标量完全退出该通路）后两者**同时消失**，max/sum/P 与 fp32 参考逐元素相等（maxErr 1.9e-6） | AIV 侧位宽转换与逐行规约一律 RegBase VF（m4/m5/m12/M24 同款）；经典 `Cast(` 在 AIV 路径上视为禁用 |
| Unaligned 系列（LoadUnAlign/StoreUnAlign） | 有状态操作（硬件 U register 做拼接，需 init/post）；**大部分场景不应使用**，确需使用时按文档+例子详细分析 | 用户裁决 |
| ND2NZ 行数=1 | 退化为 1D 拷贝是**性能特化**，不做该特殊优化即可（不是必须规避的坑） | 用户裁决 |
| fp32 次正规 FTZ | 硬件模式，**不在规避范围** | 用户裁决 |
| CrossCore 同步设计 | **不要把 SyncAllImpl 的通用做法机械套用**——它同步所有东西，我们使用时按场景更精细地设计（set 挂生产 pipe、wait 用尽可能窄的 pipe）；此前"M10 链式 wait 必须挂 PIPE_S"是特定场景结论，非通则 | 用户裁决 |
| 元素计数 DataCopy 重载 / `FixpipeParamsArch3510` 未初始化成员 | **我方代码缺陷**（传了未初始化结构体）；集成前统一显式清零。**归因修正（M21 reviewer 源码核实，2026-09-26）**：CANN 9.1.0 上 `DataCopyParams` 每个成员都有 NSDMI（`kernel_struct_data_copy.h:48-84`），count 重载内部 `struct DataCopyParams repeatParams;` 会跑默认构造 → **不会读未初始化字段**，M16 的"栈垃圾翻倍"机制无法复现（m9 README 的该归因很可能是误归因）。因此该改写为**代码规范**：新代码一律用显式块参数（`{1, len_bytes/32, 0, 0}`，blockLen 单位 32B）或 `DataCopyPad`；存量点位不必按"紧急 bug"处理 | M16/M19 实测 + M21 reviewer 源码；用户：暂不深究 |
| `TQue<VECIN,2>`（CANN 9.0） | 我们不用资源管理 API，不涉及 | 用户裁决 |

**§6.1 计算路径规范：六条 + 豁免清单**（2026-09-26 用户裁决 + tower 裁定，取代旧的「指令级白名单」；审计依据 `docs/18-vector-api-audit.md`；**ⓔ 落盘通路由 M113 新增，2026-09-27**；**ⓕ 计算三分口径由 M122 新增，2026-09-27**）

> **引用约定（全仓写作规则）**：引用代码位置一律以**符号**为准（函数名 / 常量名 / 结构体成员名），行号只作查阅提示、须注明"行号随代码变动"。理由：M48 只改了注释，本节 ⓒ 里原先的裸行号引用（`m7_router_topk.asc:352/:450`）就漂到了错位置 —— 一条指向无关代码的"现使用点"比没有更糟。**归档证据**（日志 / dump 清单 / sha256 文件）里的行号不适用本条，那是"那一刻的事实"。

- **ⓐ 计算路径必须走 vector function。** 逐元素算术 / 广播 / 比较 / 选择 / **位宽转换** / **归约** / 超越函数 / gather / interleave / pack 等计算，必须在 `__VEC_SCOPE__` 块内、或调用标了 `__simd_vf__` 的函数——**两者等价**：`__VEC_SCOPE__` 是词法入口、`__simd_vf__` 是函数属性（`__attribute__((cce_simd_vf))`）。**判据落在被调函数上**：必须 `__simd_vf__`，且函数体内**只用寄存器 API**（`RegTensor`/`MaskReg` + `AscendC::Reg::`）。**不要求**调用点有词法 `__VEC_SCOPE__`，**不要求** `asc_vf_call`（本仓 0 命中）。**禁止 memory-based 的向量计算 API**——在 `LocalTensor` / `__ubuf__ T*` 上直接做算术 / 超越 / 比较 / 归约 / 类型转换。
  - **判据按操作数类型 + 被调函数属性，不按 API 名**（M44 方法学教训）：本仓普遍 `using namespace AscendC::Reg;`，VEC_SCOPE 内裸写 `Add`/`Cast`/`Select` 解析到 `Reg::` 重载，**是合规的**；同名 API 作用于 `LocalTensor` 才违规。**按名字 grep 会得出相反结论**。
  - 对应实测：`m4_gdn_recurrent.asc` 的 `EgExpAll`（原 `:214`）与 `GdnHeadRecurrence`（原 `:279`）、`m14_gdn_layer.asc` 的同名两处（原 `:1084`/`:1149`）都是 `__simd_vf__` 裸调用形态，**合规**（不要求包 `__VEC_SCOPE__`；括注行号为 M44 审计时口径、随代码变动，**以符号为准**）；但 m14 / `m15_gdn_layer.h` 里"全部在 `__VEC_SCOPE__` 内调用"的注释是**事实错误**，由 M48 / M40 改。
- **ⓑ 搬运 / 矩阵类仍走 memory-based，不在禁令内。** GM↔UB `DataCopy`/`DataCopyPad`/`Copy`、L1→L0 `LoadData`（2D/3D；本仓文档旧称 `LoadL0_2D`/`LoadL0_3D`，这两个符号在 CANN 9.1.0 **不存在**）、`Mmad`/`MmadMx`、`Fixpipe`、`Nd2Nz`/`Dn2Nz`，以及同步原语（`Mutex`/`CrossCore*`/`PipeBarrier`）。理由：无 VF 等价物。
- **ⓒ 无寄存器等价物的原语例外：仅 `Sort32` 与 `MrgSort`。** 依据 = CANN 9.1.0 头文件穷举无 `Reg::` 等价物（`Sort*`/`MrgSort*` 在 reg_compute 系列头文件 0 命中）+ 官方 donor 同样 memory-based（`ops-transformer/moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h:187` = `Sort32`、`:225`/`:243` = `MrgSort`）⇒ 性质等同搬运/矩阵类，不是"把向量计算走在 memory-based 上"。**每处使用必须附一行注释指明本依据**（现使用点：`m7_router_topk.asc` 的 `SoftmaxTopkRenormRow`（`Sort32` 调用）与 `Merge2`（`MrgSort` 调用，由 `MergeTree` 逐级调用）、`m13_moe_layer.asc` 的 `RouterStage::SoftmaxTopkRow`（`Sort32`/`Extract`）——**以符号为准**；M44 审计时的行号口径是 `:352`/`:450`/`:974`，行号随代码改动而漂移，不得用行号定位）。例外**只许出现在排序段**，不得扩散。
  - **白名单不含**：`WholeReduceMax`（有 `Reg::Reduce`，改用）、`Concat`（3510 经典 `Concat` 是 no-op）、`Extract`（厂商 3510 的经典 `Extract` 本身就是 `__simd_vf__`、官方 donor 已手写 VF ⇒ **须改造**，M48 处理）。
  - **`MrgSort4` 在 3510 是 `[[deprecated]]` 空函数体**（不下发指令、与 `MrgSort` 参数不可互换），**不得使用**——用 `MrgSort(dst, srcList, MrgSort4Info)`。
- **ⓓ 不在禁令范围（控制路径，非计算）**：标量 index 生成 / 数据相关控制路径 / 索引与地址类 `GetValue`/`SetValue`。例：`m13_moe_layer.asc` 的 `IndexGenStage::Run`（计数排序的 offset 写入）、`m8_permute.asc` 的 `MoePermuteKernel::Process`（标量读 GM `counts` 求和，喂标量分条控制）与 `MoePermuteKernel::CopyIn` / `MoeUnpermuteKernel::CopyIn` 里的索引类 `GetValue`——均**不必为此改代码**（**以符号为准**；M44 审计时的行号口径是 `:157-159`/`:181`/`:269`）。
  - **边界**：用 `SetValue` 把**计算出的结果**写进 UB（如 `m3_grouped_gemm` `QuantRow` 的 scale 字节与 nibble）属**计算落盘**，**在禁令内**（M47 处理）。
- **ⓔ 落盘通路：设备侧把数据 / 标志落 GM 一律走 DMA；标量 pipe 直写 GM 不构成可见性契约。** 落 GM 的写侧按核型选通路 —— **AIV 用 MTE3 `DataCopy`/`DataCopyPad`**；**AIC 用 `Fixpipe`**。**AIC 这一支另有根因**（人类逐字纠正，2026-10-04）：「**AIC 就访问不了 UB，当然不能 UB->GM**」——即 **AIC 不能「直接」访问 UB**（人类再纠正，逐字：「**这是L1 to UB，不是直接访问UB**」），**UB→GM 对 AIC 不可路由**（故 AIC 落 GM 只能经 L0C 的 `Fixpipe`）；AIC 侧 `L1→UB`（MTE1）是 **pipeline 搬移**、不是直接访问 UB、不构成反证（`probe_aic_gm_dma/README.md` §1.5）。M141 探针的三个读数是这条根因的**后果 / 一致性证据**（同 README §1 / §1.4；主证据 = AIC 标量访问 UB → `507015` / plog `271`「The address for scalar to access the internal buffer is out of bounds」）。人类原话（逐字）：「**本来就不应该用 scalar pipe 写 GM，为什么要这么做**」；**AIC 这一支**另有逐字原话「**AIC 写 GM 用的是 FIXP，而不是 MTE3**，所以你这个同步本来就是错的」，与仓内硬规则 §6.2「**AIC 写 GM 必须走 `Fixpipe`，不能走 MTE3 DMA**」同源（内容锚点：该行标题里的 `AIC 写 GM 必须走`）。**标量直写 GM**（`GlobalTensor::SetValue`，以及裸 `__gm__ T*` 标量 store）**一律禁用** —— 依据：`docs/06-m0-bringup.md` §5.3 的表格行逐字「**标量 `SetValue` 写 GM ｜ kernel 退出时可见性无保证，元数据一律走 MTE3**」（内容锚点 `标量 SetValue 写 GM`；行号随代码变动，M113 时点 `:91`）＋同章 §5.2 逐字「（**标量 `SetValue` 直写 GM 同样不可靠，角色探测因此改走 DMA**）」（内容锚点 `核尾 GM 写必须排空再退出`；M113 时点 `:78`）；现象级取证见 `m15_layer_loop/ple/REAL_TABLE.md` §6.4 —— 同一 `SetValue(bid, …)` 写法在多 stage 的 body kernel 上「读回 **128 个槽位里只有 4 个核的值可见**」，改走「累计器经 UB → GM 的 `DataCopy`」后「**56/56 核全部可见**」。⇒ 元数据 / 计数 / 标志的落盘同样走 DMA，不给它们开标量直写的口子。
  - **若标量值确需落盘**（S pipe 算出、又必须进 GM），**不得**直接标量写 GM，按此依次序：**先写 UB → 用阻塞释放的 BufferID release 建立次序 → 之后才 MTE2/MTE3 搬运**。本仓的 `BufRelease<PIPE>(id)` = `RlsBufInternal<pipe,false>`（`false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」；`true` = `NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**，`true` 额外等此前同 id 的释放 ⇒ 更保守；**全项目 release 一律 `false`**；包装定义在 `m15_layer_loop/m15_moe_layer.h` 的 `BufAcquire`/`BufRelease`，**以符号为准**）。依据 = §2「跨 pipe 数据交接（如 MTE2→V）用 **BufferID release**（`RlsBufInternal<pipe,false>` + 对侧 get）即可承载数据可见性（M1 真机 10/10 PASS 实证，当时档位为 `true`）」。⚠ **覆盖面**：release 只覆盖它**之前**的写 ⇒ 标量写必须落在该 release 之前；写在 acquire **之后**的标量写不在覆盖范围内（这正是 `docs/20` 的『同族同步竞态』清单第一例 `bUb[32]` 的形态）。
  - **不得用 set_flag / wait_flag 系列**表达跨 pipe 数据可见性 —— 人类逐字禁令，即 §2 前半句原文「核内 pipeline 同步**只用 BufferID**（`get_buf`/`rls_buf`），**禁用 set_flag / wait_flag 系列**表达跨 pipe 数据可见性」。`docs/06-m0-bringup.md` §5.2 第 2 条 / §5.3 表格把「`SetFlag/WaitFlag` 事件」与 BufferID 并列为「二选一」—— **本仓只认阻塞释放（`false`=CANN `ASC_LOCK_BLOCK` 默认）BufferID 那一支**；引 `docs/06` 时不得选择性引用（M107 r2 已就此订正；该修单见 `docs/20` 的 WO-B1 段）。**有值依赖例外（M195 落库，2026-10-05）**：§2 的 PIPE_S 同步限制原文是「除**有值依赖**的场景外……**禁止把同步挂在 PIPE_S 上**」——「标量读另一个 pipe 刚算出的值」正落在该例外的定义里（典型：router 的 MTE3 GM 写排空后，标量再读回该 GM 值）。此时挂在 `PIPE_S` 上的 `SetFlag/WaitFlag<HardEvent::MTE3_S>`（或 `V_S`）是**允许的例外**，不算违反上一条禁令。允许实例（**以符号为准**，行号随代码变动）：`m13_moe_layer.asc:1082-1083` 与 `m17_moe_layer.asc:1267-1268` 的 `SetFlag/WaitFlag<MTE3_S>` —— 语义即「router 的 MTE3 GM 写排空后再做标量 GM 读」（两处源码注释逐字），标量消费的就是 MTE3 刚写出的值 ⇒ 有值依赖，**保留、不改**。判据边界：**无值依赖**（搬运与搬运之间的次序）仍一律走 BufferID；且同处 BufferID 已承载可见性时**不得**再挂事件（M195 据此**删除 7 对冗余事件**、把另外 **2 对「标量写 UB → MTE3 搬」的 `S_MTE3` 改成 BufferID 形式**（`PIPE_S` 阻塞释放 + 对侧 `BufAcq<MTE3>`），`V_S` 处经 A/B 消融实测为承载而保留；见 `m19_qsa_indexer/evidence/sync_cleanup_m195/README.md`）。
  - **`PipeBarrier` 单独不足以建立跨 pipe 依赖**：`docs/06-m0-bringup.md` §5.2 第 2 条逐字「`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，**不能保证 UB 数据路径对另一 pipe 可见**」，§5.3 的表格行直接判「**不可靠**，不要用」。⚠ **留白（必须保留）**：M107 r2 把「标量写 UB → `PipeBarrier<PIPE_ALL>` → MTE2/MTE3 搬走」这一形态的定性写成「**未建立 / 待确认**」—— **够登记、不够定罪**，即按现有依据只能记「该次序**未被建立**」，**不得**写成「已坏 / 硬件有问题」；同族站点尚未逐个核（清单与判据见 `docs/20` 的 B10 条与 WO-B4 条）。
  - **每条落盘判据必须配正对照**（`docs/17` §4 非空洞纪律在落盘通路上的延伸）：读回 0 只有在「人为让它该非零时确实非零」之后才构成证据。`ple/REAL_TABLE.md` §6.4 自带一个正对照 —— 同一个 `SetValue` 写法在 ① 的 kernel（`m15_ple_ids_kernel`）上**可用**（`dev_range_fail = 16`）⇒ 现象限定在 body kernel、不是「整个通路都不行」。修复 / 新写的判据都必须能咬住（改前应当红）。
  - **本仓正对照（该长什么样）＝ `m15_layer_loop/m15_hc_layer.h` 的 `HyperConnOp::CombineStage`**（塔已升格为**舰队级正对照**）：**纯 BufferID 的 V→MTE3 握手** —— `BufAcquire<PIPE_V>(BUF_AIV_OB)`（写侧 V 取得槽）→ VEC 计算 → **`BufRelease<PIPE_V>(BUF_AIV_OB)`（阻塞释放）** → `BufAcquire<PIPE_MTE3>(BUF_AIV_OB)` → `DataCopy` 落 GM → `BufRelease<PIPE_MTE3>(BUF_AIV_OB)`。**每一步 release 都是 `BufRelease` = `RlsBufInternal<pipe,false>`（= CANN `ASC_LOCK_BLOCK` 默认「阻塞」）**，交接**每一步都走 BufferID**、不含任何 set_flag/wait_flag，也不靠 `PipeBarrier` 承担可见性。**以符号为准**（M107 r2 口径行号：`CombineStage` 在 `20bd20d` 的 `653`、四个 BufferID 调用在 `681`/`697`/`699`/`701`；行号随代码变动）。
    - ⚠ **不作先例的两处**：`m15_attn_cache.h`（`m15_attn_cache_body`）与 `m15_attn_kv_probe.h`（`m15_attn_kv_probe_body`）里「标量写 UB → `PipeBarrier<PIPE_ALL>` → MTE3 搬走」的形态，其次序按 M107 r2 的**舰队级更正**是「**未建立**」—— **既不是反例也不是先例**，**不得**当正确形态引用（行号口径见 `docs/20` 的『形态上与「写在 acquire 之后」不同』表）。
  - **与 ⓓ 的边界（两条并列、不合并）**：ⓓ 说的是「用 `SetValue` 把**计算出的结果**写进 **UB** 属**计算落盘**、**在禁令内**」；ⓔ 说的是「落 **GM** 必须走 DMA」。ⓓ 管**核内的写落在哪一侧**，ⓔ 管**跨出核的那一跳**。
- **ⓕ 计算三分口径：矩阵乘法一律 cube（`Mmad`），其余一切数值计算一律 VF，scalar 只做控制流。** 人类原话（逐字）：「**矩阵乘法 ⇒ 必须 cube---这个是强制规则，其余的都用VF做，不要用scalar做计算，scalar只做控制流**」（tower 全仓广播 `20260927-tower-all-matmul-cube-vf-scalar-scalar.md`）。**三条并列、每条都强制**：
  - **① 矩阵乘法 ⇒ 必须 cube（`Mmad`），不分 M 大小** —— 含 **M=1 的 GEMV**、外积、卷成 GEMM 的 conv、**router 打分**。依据一：`docs/11-attn-analysis.md` 逐字「**m=1 在 mmad 被 pad 成 16**（`matmul.h:670-672` "m==1→16" —— mega kernel 同样 m 走 16 行矩阵）」（内容锚点 `m=1 在 mmad 被 pad 成 16`）⇒「M=1 所以走向量」在本工程**没有设计依据**。依据二：与 ⓑ 是**两层不同的话** —— ⓑ 说矩阵类走 memory-based **不在禁令内**（允许 `Mmad`），ⓕ① 把矩阵乘法从「允许」升为**强制**；两条不冲突、不合并。
    - **管辖边界（塔裁：免 cube，不免 VF）**：只有「**有固定的矩阵 / 权重操作数参与收缩**」的才算本条管辖的矩阵乘法（`docs/20-kernel-compliance-sweep.md` §0.1 的塔裁判据，B1/B2/B9：`M=N=1` 无权重操作数的单点积、`K=1` 外积、无固定权重矩阵的加权行和）—— **不要求 `Mmad`**，但**仍必须 VF**（见 ②）。
  - **② 其余一切数值计算 ⇒ VF**（`__VEC_SCOPE__` / `__simd_vf__` 里的 RegBase，即 ⓐ 的那条通路）：逐元素 / 广播 / 比较 / 选择 / **归约** / softmax / **门控** / **加权行和** / **递推** / **量化反量化** / 位宽转换 —— 都算。
  - **③ scalar 只做控制流**：循环 / 索引 / 分支 / 地址与偏移计算。**可操作界线（人类给的）**：**这个量是「数据」（要参与后续数值结果）⇒ VF；是「地址 / 下标 / 循环边界 / 分支条件」⇒ scalar**。⚠ 所以「顺手用**标量累加 / 标量乘**」、以及「某个**归一化系数 / 门控 / 掩码**用标量算一算」这类小活**都算计算**、不是控制流 ⇒ 一律改 VF。与 ⓓ 的边界：ⓓ 只**豁免控制路径**（索引与地址类 `GetValue`/`SetValue`）；一旦那个量参与数值结果，就回到本条 ②。
  - **反例指名（两类，勿读混）**：
    - **(a) matmul 跑了 AIV ⇒ 上 cube**：`m15_moe_layer.h` 的 `RouterStage::GemvGroupRow` / `RouterStage::SgateRows`（`x @ W[512,2560]^T` 与同一权重族的 `N=1` 行，塔裁 B8 在管辖内）与 `m15_ple.asc` / `m15_ple_wire.h` 的 `PleGemv`（`[1,2560] @ [2560,12800]`，**两份副本**）—— 即 `docs/20-kernel-compliance-sweep.md` §2.2 的 **V2 / V1（均 P1）**。修法 = 该文件的 **WO-A2 / WO-A1**（搬 cube），**不是**换一个更快的 vector API。（注：这两处在库形态是 VF RegBase 的向量 MAC —— 违的是 ①，不是 ③「标量做计算」；勿据此把 VF 判成违规。）
    - **(b) 不是 matmul，但也不能用标量 ⇒ 必须 VF（塔裁）**：`m15_hc_layer.h:GateMixStage`（系数随 `j` 逐元素变的门控归约，塔裁 B4）、`m15_ple.asc:PleGateItem`（`M=N=1`、无权重操作数的单点积，塔裁 B1）、`m15_moe_layer.h:FmaChunk` / `UnpermuteStage`（unpermute 加权行和，塔裁 B9）—— **这三项在 `docs/20-kernel-compliance-sweep.md` §2.4（N4 / N11 / N12）里被判「不是矩阵乘法」；那个判定的唯一含义是「不上 cube」，不是「可以用标量」** ⇒ 按 ② 它们**必须 VF**。

**豁免清单（不适用本节规范）**：

- **bring-up 探针 `m0/**`**——其经典 `Add`/`Adds` 正是"验证经典路径与同步原语"的设计目的；
- **取证工程 `probe_*`（`probe_sync_quirks/` 等）**——其 (a) 类是**故意保留的反例/靶子**；
- **host 侧参考实现**（各 `.asc` 的 host 段、`check_ref.py` 等）——本节只约束**设备计算路径**。
- **不许**为了"合规"去破坏证据基线或改动探针反例。

> **⚠ 案注（2026-09-26；M41 复评后补，按 tower 裁定精简；M66 更新合并状态；M67 回填内容）**：**M27 已随 `eeab864` 合入 main**（分支 `feat/l0-load-geometry-calibration` 的 tip = `cdbfa20`）⇒ 本节所引 `m16_load_geom/README.md` 以**合并版为准**，"提交前仍会变动"的窗口已关闭。
>
> **回填状态：已回填**（M67 的 commit：`git log --oneline --grep='docs(M67)'`）。M66 曾登记在案注里的四处在本次提交里逐处按合并版改写：① §6.1 表格行 ③ 的 3D 结论改为"29 组 conf 的回读窗口内未观察到写入 + `m16_load_geom/README.md` §3.6 的限度"（删去比证据更强的绝对措辞）；②「M27 标定 4 条」块的**第 1 条**按该 README §3.5 的第 2/3 条重写为**单位澄清**（不是分歧）与"`startAddr` 公式在 `m1=k1=0` 时逐项一致"，并补上 §3.5 第 4 条（`|srcStride|` 实测按有符号使用）与「分形下标 ≠ 线性字节偏移」的换算；③ 块末附带一句的 3D 结论按 §3.6 收窄（保留「BMM2 的 V 用 2D `ifTranspose=true` 即可」）；④ §6.1 举例里的 `kStep=4` 按该 README §2.3/§3.4 改述（越界组合的读数不可用于推断；`mStep=kStep=0` 是 NOP）。**内容以合并版为准**（M27 已随 `eeab864` 合入 main）。

**M27 标定的 4 条 L0 装载字段语义（2026-09-26）**——源：`m16_load_geom/README.md` §3.4 的逐字段标定表 + §3.5 的官方文档对照（**M27 标定，已随 `eeab864` 合入 main**；引用时的分支 tip 为 `de68fe3`，其后另有 `2e54a89`/`cdbfa20` 两次复评修正）。**与官方文档的对照口径 = 1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节）**（§3.5 **表内 4 行**的框定；唯一真分歧 = `srcStride`/`dstStride` 的**轴向措辞**，真差异 = 文档公式对 `|srcStride|` 取绝对值而实测按有符号用，见下方第 2 条）。完整表与逐点验证脚本 `check_ref.py geom2d`（入口与逐步跑的命令见该 README §1 的交付物表；本机 `python3` 无 numpy，需用该 README §1 指定的解释器）都在该 README §3：

- **`mStartPosition` / `kStartPosition` = 源「分形」下标**（分别沿源的行/列分形方向）——**§3.5 把它列为单位澄清，不是分歧**：文档说的 `M 轴 16 元素` / `K 轴 32 字节` 在 bf16 上**正好等于**「1 个 M 分形行」/「1 个 K 分形列」（`LoadData_2D_V2.md` 写 b16 分形为 16×16），与实测的"源分形下标"**等价**（官方样例 README 更直接写"起始**小分形位置**"）：`mStartPosition=1` = 整个 16×16 分形平移 1 个行分形，`kStartPosition=1` 在地址上平移 `srcStride` 个分形。文档的 `startAddr = srcAddr + (kStartPosition×|srcStride| + mStartPosition)×512B` 在 `m1=k1=0` 时与 §3.3 的实测模型 `u_src = (mStartPosition+m1) + (kStartPosition+k1)×srcStride`（u 的单位 = 512B）**逐项一致**（唯一细节差异 = 那个绝对值，见下一条）。**要记住的换算：分形下标 ≠ 线性字节偏移**——K 轴的起始位置在地址上要**乘 `srcStride`** 才变成分形偏移（文档公式正是这么写的）。
- **`srcStride` = 源「列方向」相邻分形在 512B 单位的间隔**（= 源带 pad 的行数/16）。API 正文写"K 方向"、官方样例 README 把 `srcStride`/`dstStride` 都写成"row 方向"——**这处轴向措辞就是 §3.5 里唯一的真分歧**；实测 `srcStride` 是**源列（K）方向**的间隔（合正文）。**按有符号使用**：文档公式写 `|srcStride|`（取绝对值），实测不取——负值把源读到 buffer 之前，属非法组合（§3.5 第 4 条）。
- **`dstStride` = 目的 L0B 的 n 分形数 = `N/16`**（实测语义：目的分形行号 = `(k/16)·dstStride + (n/16)`；官方样例的算例里 `dstStride = nAlignL0/16` 与该式数值相同，所以样例的"row 方向"措辞也能算对——这条分歧一直没暴露）。照抄别的形状（如 BMM1 的 `S2T/16`）会越界—— M27 实测该组合 device 直接报错（`507015`，通用 aicore exception 码，**不可反推"是哪条指令的语义问题"**）；两者只在 `N == S2T` 时数值恰好相同。
- **`mStep` / `kStep` = 源行 / 列方向的「分形个数」**（不是 16 元素 / 32 字节）；`mStep=0` 或 `kStep=0` 是 **NOP**。越界组合（如 `kStep=4` 超过本探针源只有 2 个列分形）的读数**不可用于推断**（§2.3）。

附带（正面结论，写 MMAD 路径时直接用）：mmad 眼里的 L0B 地址 = `16n + k`，分形行 = `(k/16)·(N/16) + (n/16)`；`ifTranspose=false` ⇒ 分形原样（`B[k][n] = src(r0+n%16, c0+k)`），`ifTranspose=true` ⇒ 分形内 16×16 转置（`B[k][n] = src(r0+k, c0+n%16)`）。⇒ **BMM2 的 V 用 2D `ifTranspose=true` 即可，不需要 3D 转置装载**：3D `LoadData3DParamsV2` → L0B 在 M27 的 29 组 conf 回读窗口内**未观察到写入**（限度见 `m16_load_geom/README.md` §3.6：回读窗口只到 L0B 行 0–3 即 mmad `n<64`、29 组里没有"已知能写"的正对照 ⇒ 严格说只能断言"未观察到写入"；文档的两条硬约束——该通路自动转置、L0B 下 `enTranspose` 无效——是"这条路不该走"的独立依据）。3D → L0A（文档承认 `enTranspose` 唯一有效的通路）M27 **未标定**（该 README §6【仍存疑】）。

### 6.2 实测硬件行为（保留；写代码必须遵守）

| 项 | 行为 | 处理 |
|---|---|---|
| **同 pipe MTE2 背靠背复用同一 buffer** | 完成序无保证；**`get_buf/rls_buf` 对同 pipe 无效**（mode 0 与 drain 均实测不保序，60/60 错 vs PipeBarrier 0/60）。`PipeBarrier<PIPE_MTE2>` 是同 pipe 内部同步的正确且唯一有效指令（用户确认）。**官方口径同**（M128 补 pin，非设备、非自指）：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Lock.md:93` 逐字「连续调用的、具有相同id与pipe的两对Lock与Unlock**不能实现单流水（参数pipe指定）内不同指令之间的同步，单流水内多个指令之间的同步请使用PipeBarrier接口**」；另一页 `asc-devkit/docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_lock.md:101` 为同一句、把 `PipeBarrier` 换成 `asc_sync_pipe`；旗标侧 `asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md:99` 逐字「相同流水、相同eventID下，连续使用SetFlag会引发未定义行为，此时再执行PipeBarrier<PIPE_ALL>会出现卡死现象」（可能与 M117 挂死相关；以官方原文口径登记，本轮不据此下根因结论） | 同 pipe buffer 复用前加 `PipeBarrier<对应PIPE>`；跨 pipe 交接用阻塞释放 BufferID（`false`=CANN `ASC_LOCK_BLOCK` 默认；实测有效） |
| **CrossCore `SetFlag/WaitFlag<mode, pipe>` 必须 pipe 类与核型匹配**（M24 实测；M17 挂死根因） | 实现只在 pipe 类与核型匹配时**发射指令**：`IsSplitVectorPipe={S,V,MTE2,MTE3}` 仅 AIV 发射、`IsSplitCubePipe={S,MTE1,MTE2,FIX,M}` 仅 AIC 发射 ⇒ 不匹配时是**静默空操作**（不报错），对侧 wait 永久挂死。**用户澄清**：这里的原始错误是用法本身错——**AIC 侧写 GM 用的是 FIXP，不可能用 MTE3**（**根因：AIC 不能「直接」访问 UB ⇒ UB→GM 对 AIC 不可路由**，见 §6.1 ⓔ 与 `probe_aic_gm_dma/README.md` §1.5；所以 AIC 上挂 `PIPE_MTE3` 从同步语义上就是错的，不是"合理但踩雷"）；pipe 类匹配规则只是让这类误用表现为静默挂死而不是编译报错 | **AIC 侧 set/wait 只能 `PIPE_S/MTE1/MTE2/FIX/M`；AIV 侧只能 `PIPE_S/V/MTE2/MTE3`**。写同步前先问"这个核型的这条 pipe 到底存不存在"；flagId 8-11 无保留冲突，pipe 类才是雷 |
| **L0B 内容会跨 kernel launch 残留**（M27 实测，2026-09-26） | **L0B 不是每次 launch 都重置的**：一个不写 L0B 的装载（静默不生效、NOP、越界读）会读到**上一次 launch 留下的 L0B**。实证：`m16_geom` 的 conf 48（`zeroFill=0` + `mStep=kStep=0`）**逐位复现前一个 conf（47）的输出**；M24「参数有效果但读数不可复现」的真正机制就在这里——**"每次 launch 只跑 1 个 conf"仍然不够**（该实证在**同一进程内**；跨进程残留在该 README §2.2 只作推断、其 §6【仍存疑】明写） | **硬约束：探针 / 调试 / 自检代码在读数之前必须显式清零目的 buffer**（M27 做法：每次 launch 前用已验证的单分形装载，从全零 L1 逐分形把 L0B 前 8 个 512B 分形槽清零；清零后良构 conf 在 5 次独立进程运行中逐位一致）。配套纪律：**越界/错参数组合的读数不可用于推断语义**（越界 conf 的内容非确定，逐对分组见 `m16_load_geom/evidence/geom2d_determinism.txt`）。源：`m16_load_geom/README.md` §2.2（同一进程内的跨 launch 残留实证）与 §2.3（清零后的确定性：5 次独立进程）（M27 标定，已随 `eeab864` 合入 main；引用时的分支 tip 为 `de68fe3`） |
| **硬规则：AIC 写 GM 必须走 `Fixpipe`，不能走 MTE3 DMA；配套：AIC 上的同步事件不得挂 `PIPE_MTE3`**（M24 + M35 + 用户裁定，2026-09-26） | **三次独立撞到同一根**：① **同步类**（M24 实测；M17 全 kernel 挂死根因）：`CrossCoreSetFlag/WaitFlag<…, PIPE_MTE3>` 在 AIC 上是**静默空操作**——`IsSplitCubePipe={S,MTE1,MTE2,FIX,M}` 不含 MTE3 ⇒ 指令根本不发射，对侧 wait 永久挂死；② **数据类**（M35 `probe_aic_blockidx.asc` 实测）：AIC 用「标量写 UB → MTE3 DMA 写 GM」**全部读回哨兵**，且**是测量手段先坏、不是被测量的对象（bid 结果）坏**（M35 的 AIC GEMM 因此**仍未收口**，如实标注在其 README §5.2「移交 tower / 后续」——M35 时点的编号写作「§5.3」，指当时 §5 列表的第 3 条）；③ **用户裁定原话**："**AIC 写 GM 用的是 FIXP，而不是 MTE3**，所以你这个同步本来就是错的" | **后果＝静默失效**：不报错、不挂死（同步类表现为对侧永久等待，数据类表现为 GM 里仍是旧值/哨兵）——属**最难查的一类错**。⇒ **AIC 侧 set/wait 只允许 `PIPE_S/MTE1/MTE2/FIX/M`，AIC 写 GM 只用 `Fixpipe`**；写任何 AIC 侧同步前先问"这条 pipe 在这个核型上到底存不存在"。方法学配套：**先确认测量手段本身成立，再解释测量结果**（原记于 M35 README §3.4(4)，该编号已随 M53 对 `m19_qsa_indexer/README.md` 的重写失效 ⇒ 现见 `docs/17` §9.1「测量手段先成立」；同族案例：`DeInterleave` 撤回）。**本条与 §6.1 的计算路径禁令是两件事，不要混进同一条**。**根因归位（M153，人类逐字 2026-10-04）**：「**AIC 就访问不了 UB，当然不能 UB->GM**」＋「**这是L1 to UB，不是直接访问UB**」——即 **AIC 不能「直接」访问 UB**，故 AIC 上的 UB→GM（MTE3）不是"配错"而是**不可路由**（AIC 侧 `L1→UB` 的 MTE1 是 pipeline 搬移、不构成反证）；raw intrinsic 编译被拒、基础 API 在 AIC 上被 `if ASCEND_IS_AIV` 门成空、AIC 标量访问 UB 直接 `507015` / plog `271` 越界，三条都是这条根因的**后果证据**（`probe_aic_gm_dma/README.md` §1/§1.5；见 §6.1 ⓔ） |
| **VF（`__simd_vf__` / `__VEC_SCOPE__`）内层循环的上界不得取外层归纳变量本身、或由它直接导出的上界变量**（M43 实测；源 `probe_vf_loop/`，已随 `7f627ce` 合入 main） | 实测 `for (uint16_t i = 0; i < rows; ++i) { for (uint16_t j = 0; j < i; ++j) {…} }`（`rows` 为运行期 `uint16_t`）**内层体只执行 `max(i−1, 0)` 次**（丢 `j = i−1`，**静默、无报错**）；`rows ∈ {0,1}` 的退化档正确。内层体**只有寄存器累加、零访存**时同样复现 ⇒ 属循环控制流层面，不是内存可见性 / 数据通路问题；`-O2` 与 `-O3` 相同。⚠ **条件与现象未对齐**（`j < i+1` 同样依赖外层归纳变量却不复现；外层归纳变量换 `uint32_t/int32_t` 时 `j < i` 也不复现）⇒ 本条只作**安全写码规则**，不是机制结论 | 内层上界写成**不依赖外层归纳变量的运行期值**（如 `j < rows`）或**编译期常量**（m18 现行做法：A 严格下三角使多算项为 0；代价 BT²/2 → BT²，m18 实测本核 +12% 向量指令）—— 这两族在全部档位都正确。**不要**用这些"看起来能规避"的写法（实测**无效或不可靠**）：把上界提到内层循环外先算（u16/u32/i32 上界变量都不行）、内层 `#pragma unroll`、把**外层**上界改成编译期常量（`ct=1/2` 恰好对、`ct=3/4/64` 仍复现，取决于编译器是否完全展开外层）。**机制未对齐 ⇒ 禁止当因果规则引用**；现象与两个反例见 §6.3 #24 |
| **硬规则：VF（`__VEC_SCOPE__` / `__simd_vf__`）循环体内不得出现任何标量指令**（M43 实测 + tower 裁定，2026-09-26） | **标量整数算术**（`cntS = cntS + 1;`）与**标量 store**（逐轮写标记 `mk[i*RW+j] = j;`）都**编译期**失败：`fatal error: error in backend: Unsupported scalar instruction in AIV loop`。**正对照**：同样"每轮一次 store"、改用**向量** `StoreAlign`（地址含 j、值不含 j）**能编过** ⇒ 被拒的是**标量指令本身**，不是"循环里做 store"。§6.3 行 d 原写的"不支持 scalar float 算术"据此收窄 | VF 内的计数 / 标记**只能用向量寄存器**，或把落盘挪到循环外。靶子与日志：`nest_o0i0b2_scal`（标量算术，`probe_vf_loop/evidence/logs/build_nest_o0i0b2_scal.log`）、`nest_o0i0b2s_scalstore`（标量算术 + 标量 store，`probe_vf_loop/evidence/logs/build_nest_o0i0b2s_scalstore.log`）、正对照 `nest_o0i0b4_marker_ctl`（**COMPILE-OK**，`probe_vf_loop/evidence/logs/build_nest_o0i0b4_marker_ctl.log`）；详见 §6.3 表行 h |

### 6.3 待复核 / 已结案（用户审核后更新，2026-09-26 第二轮）

**已结案（我方用法错误，非硬件问题）**

| # | 现象 | 结论 |
|---|---|---|
| 20 | fp32/int32 `GlobalTensor` 切片取址错误 | `GlobalTensor<T>::operator[](offset)` 是按 `PrimType*` 的原生指针算术（`kernel_tensor_impl.h:1348-1366`），**offset 单位=元素数**。原先按字节传偏移：bf16 视图（2B/元素）恰好吻合，fp32/int32（4B/元素）地址放大 4 倍越界 → 结案：用法错误 |
| 21 | 同一 `GlobalTensor` 重复 `SetGlobalBuffer` 失效 | `SetGlobalBuffer` 首次调用会把 `cacheMode_` 从 NORMAL 改为从指针位域提取的模式（sticky），二次调用对裸指针做 `L2CacheAlter` 污染地址（`kernel_tensor_impl.h:1029-1054`）→ 结案：**每对象只 Set 一次**的 API 约束 |
| 6 | `DataCopyParams` blockCount>1 落盘顺序非线性 | **文档语义**：该重载是"blockCount 个 block × blockLen + srcGap/dstGap 间隔"的块拷贝语义（`kernel_operator_data_copy_intf.h:45-52` 等），不是行主序线性拷贝——我们误当线性用才是错的。**"NZ 重排"表述删除** |
| 22 | 标量→向量跨 pipe 间歇丢写（~1/30） | **用户裁定 + 我们的错**：原实现用跨 `__VEC_SCOPE__` 抽取的 `LocalMemBar` 做同步——`LocalMemBar` 只管 SIMD VF 内部的 UB load/store，**根本覆盖不了 scalar 的 load/store**。属同步机制用错。正确原语（能覆盖 scalar↔VF 的 BufferID/事件）待复现实验确定后回填 |

**待复核（需补证；禁止当事实引用 —— 各行已归档的探针/证据在行内点名：#19/#22/#23 见 `probe_sync_quirks/`、#24 见 `probe_vf_loop/`）**

| # | 已掌握的细节 | 缺什么 / 复核计划 |
|---|---|---|
| 19 | 4 路 MrgSort（validBit=15）丢 src3/src4 胜出者 index | **最可能成因（M23 探针，2026-09-26）→ 大概率我方调错 API**：`AscendC::MrgSort4` 在 dav-3510 构建中是**静默 no-op**（dst 全 sentinel，仅一条 `[[deprecated]]` 警告、运行无错）；同参数改走 `MrgSort(dst, srcList, MrgSort4Info)`（vb=15）**结果正确**（A02/A15/A20 三个独立变体各 128 对全序、逐项等于真值归并）。**注**：当年那次调用确实用了 `MrgSort4` 的原始代码未归档，故只能给出"最可能成因"而非定论 |
| 23 | 向量循环内动态下标标量加载 → `Unsupported Inst must be hoisted` | **已定位（M23 探针 round-2，2026-09-26）**：触发条件是 **VF 内对 GM 的标量读**（`__gm__` 裸指针 `gP[j]` **或** `GlobalTensor::GetValue`；后者只是同一触发点的 API 形态——其接口只吃 `uint32_t` 偏移，**不能**用于"与 idx 类型无关"的论证）——form 8（VF 内 `__gm__ T*` 裸指针 + IdxT **原类型、无强转**）6 种真正不同的程序全部报同一句 `fatal error: error in backend: Unsupported Inst must be hoisted`，报错栈直接落在 `gP[j]` 源行（无 `GetValue` 栈帧）；而 form 0/2（UB 裸指针 `wP[j]` / `LocalTensor::GetValue`，含 int64/uint64）50×8 点全对 ⇒ **动态下标本身合法，问题在地址空间（GM vs UB）**。正确写法＝把该 GM 标量读 **hoist 到 VF 外**（form 7，50/50 PASS）。完整报错原文与复现命令已归档 `probe_sync_quirks/evidence/`（`logs/build_probe_b_idx_*_f8_gmraw.log`）；**仍存疑（只能说到这一步）**：「VF 内标量读 GM、且结果**只**进标量路径（不进任何向量操作数）」未做实验 ⇒ 目前只能断言"**标量读 GM 出现在 VF 内即报错**"。另两条 VF 硬约束（归纳变量须 `uint16_t`、`LoadAlign` 须 32B 对齐）见下方「3510 VF / store / 循环类硬约束」表 d/b 行 |
| 22 | 标量→向量跨 pipe 间歇丢写 | 维持"我方同步用错（`LocalMemBar` 覆盖不到 scalar load/store）"结论；M23 顺带量化复现：标量↔VF 通路错误率 form1 8~12/400（launch 级偶发）、form6 350/400，复现器与率**已补入本条**（`probe_sync_quirks/probe_b_vec_idx.asc`，50 launch × 8 检查点）；**对照 form 7**（同样经「标量 → UB 暂存 → VF `LoadAlign`」，但标量读的是 GM）**50/50 PASS** ⇒ 该 hazard 至少在一部分构造里与「标量读 UB / 标量结果作为向量标量操作数」相关 |
| 24 | **VF 内层循环的上界写成外层归纳变量时内层少执行一次（丢最后一项）** —— M43 探针（`probe_vf_loop/`，27 变体 × 5 次独立进程；**判据列**跨 rep 逐字符一致，数据面列在病态变体 `j<i−1` 上非确定、不影响本条）：`__simd_vf__` 内 `for (uint16_t i = 0; i < rows; ++i) { for (uint16_t j = 0; j < i; ++j) {…} }` 内层体实际执行 `max(i−1, 0)` 次（`rows ∈ {0,1,2,3,4,64}`；rows=0/1 退化档 0 次且 PASS）。读数用**向量计数器**直接读 trip count：`cnt[i] = i−1`（对照 `j < rows` 时 `cnt[i] = rows` 全对）。内层体只有寄存器累加、零访存时同样复现 ⇒ 循环控制流层面，不是内存可见性/数据通路问题。**两个反例（如实记录，禁止简化）**：`j < i+1`（同样依赖外层归纳变量）**不复现**（`cnt = [1,2,3,4]` = 期望）；外层归纳变量换 `uint32_t/int32_t` 时 `j < i` **也不复现**。另有三种"错法不同"的记录：`j < i−1`（带保护）变成 `cnt = 期望 + 255/+256`；**不带保护**的 `j < i−1` 在 `i==0` 时 uint16 下溢 ⇒ 界变 65535 ⇒ 地址远超 UB ⇒ `507035`（C 语义 foot-gun，不是被测现象）；`j+1 < i` 让编译器**段错误**（该写法在本版本不可用、无法纳入矩阵）。**可用写码规则已另立一条升 §6.2**（本行只留"观测到"的事实） | **缺一个能把全部格子对齐的单一条件**：`j<i` 错 / `j<i+1` 对 / 外层 IV 换 u32/i32 对 ⇒ **既不能说"恒少一次"，也不能说"上界依赖外层 IV 就出错"**。**归属未定**：未做 ISA 级取证（设备代码在 `.aicore_binary` fatbin 内，device side 不支持 `-S`，`msobjdump` 不是反汇编器）⇒ **不宣称**"硬件 bug"或"编译器 codegen 缺陷"。未覆盖形态：内层上界为 `rows−i`/`i/2`/非单调表达式、外层步长 ≠ 1、三层及以上嵌套、内层归纳变量非 `uint16_t`、`#pragma unroll N`、其他 CANN/bisheng 版本。**取证可复算（已归档）**：`probe_vf_loop/README.md`（`bash run_probes.sh 5`；逐 target 完整编译日志与编译选项、每个变体 5 次独立进程的完整读数 + sha256、编译不过靶子的 stderr 原文、`probe_vf_loop/tools/summarize.py verify` 三态退出码） |

**3510 VF / store / 循环类硬约束（M14 十探针实机 + M23 独立复核 + M43 VF 循环/索引探针，2026-09-26）**

> 来源：`m7_router_topk/README.md` §「3510 / CANN 9.1.0 实测 quirk」（M14，10 个最小探针 kernel，Ascend950PR 真机 / CANN 9.1.0）+ `probe_sync_quirks/`（M23 独立复现）+ `probe_vf_loop/`（M43；**已随 `7f627ce` 合入 main**）。
> **标签口径**：【已确证】＝已被隔离复现；**【仍存疑】＝未隔离复现的观测，不得当规则引用**。**`507035`/`507015` 是通用 aicore exception 码，只证明"该非法组合会异常"，不指向具体指令**（与 §6.2、M27 同一条纪律）。
> **本表行 h / i / j 与行 d 的修订由 M43 的 `probe_vf_loop/` 提供依据**；§6.3 #24 与 §6.2 新增的 VF 内层循环写码规则同源（同一份探针）。

| # | 约束 | 标签 | 复现条件 / 依据 |
|---|---|---|---|
| a | **`Reg::LocalMemBar` 必须在 `__VEC_SCOPE__` 内调用**，否则 `507035` vector core exception | 【已确证】 | M14 最小探针；与 §6.1 规则 ⓐ「`LocalMemBar` 属寄存器侧屏障、随 VF 使用」同源 |
| b | **VF 内 `LoadAlign` / `StoreAlign` 的地址必须 32B 对齐**——非 32B 对齐（4B 步长；含 `mask=1` 等小 mask 的 `StoreAlign`）→ `507035`；**32B 对齐 + `mask=8` 可用** | 【已确证】 | M14 探针 + **M23 独立复现**（`probe_b_vec_idx.asc` form 9，4B 步长 → `507035`；`evidence/logs/run_*_misalign_demo.log`）。⇒ 短向量写用「32B 对齐 + 整寄存器写 + 行级紧凑拷出」 |
| c | **`Reg::StoreUnAlign(dst, reg, ureg, n)` 是「整寄存器写」**（`n` 只是 dst 指针的 post-update 步进）⇒ 不能当短向量写用，否则**踩踏邻区** | 【已确证】 | M14 探针（真机）；规避见 `m7_router_topk.asc` |
| d | **`__VEC_SCOPE__` 内循环归纳变量必须 `uint16_t`**（否则编译期 `Induction variable must have a type uint16_t`）；**`UpdateMask` 需 `uint32_t` 左值**；**循环体内不得出现任何标量指令** —— 原写"AIV 循环内不支持 scalar float 算术"**过窄**：标量整数算术与标量 store 同样被拒（M43）⇒ 见行 h | 【已确证·措辞收窄】 | M14 编译期报错 + **M23 独立复现**（`evidence/logs/build_*_u32loop.log`）；"任何标量指令"的实测依据 = 行 h 的三个靶子 |
| e | **`Reg::Arange` 只填 64 lane**（fp32 VL = 256B / 4B）⇒ int32 128-lane 寄存器高 64 lane 为 0；索引模板须按 64-lane 分块 | 【已确证】 | M14 探针；**这不是 quirk 而是 VL 语义**（见 §6.1「SIMD VL」行，此前"Arange 只填 64 lane"的表述按此修正） |
| f | **普通 128-bf16 寄存器 `Cast<float>` 只取偶数下标元素**；全量转换用 `LoadAlign<bfloat16_t, DIST_UNPACK_B16>` + `Cast`（每步 64 个连续元素） | 【已确证】 | M14 探针；属 §6.1「位宽转换 Cast」行"bf16→fp32 需 even/odd 各一条 cast"的具体形态 |
| g | 4 路 `MrgSort`（`validBit=15`）**并不丢 src3/src4 的 index**——参数本身正确（M23 A02/A15/A20 各 128 对全序、逐项等于真值归并）；当年"丢 index"的形态来自**误用 `MrgSort4` API**（3510 静默 no-op）⇒ **我们用法选错**（详见上表 #19） | 【已确证·更正】 | `probe_sync_quirks/`；`elementLengths=256` 触发 `507035`（≤64 可用，本仓统一 32）；5 种"静默不写/截断"条件见 #19 与探针工程 README §3–§6 |
| **h（硬规则）** | **VF（`__VEC_SCOPE__` / `__simd_vf__`）循环体内不得出现任何标量指令**：标量整数算术（`cntS = cntS + 1;`）与标量 store（`mk[i*RW+j] = j;` 逐轮写标记）都**编译期**报 `fatal error: error in backend: Unsupported scalar instruction in AIV loop`。**正对照**：同样"每轮一次 store"、但用**向量** `StoreAlign`（地址含 j、值不含 j）**能编过** ⇒ 被拒的是标量指令本身，不是"循环里做 store"。⇒ VF 内计数/标记只能用向量寄存器，或把落盘挪到循环外 | 【已确证】（**tower 裁定升为硬规则**：编译期确定性的硬错误，不放在说明里） | M43 `probe_vf_loop`：靶子 `nest_o0i0b2_scal`（标量算术，`probe_vf_loop/evidence/logs/build_nest_o0i0b2_scal.log`）+ `nest_o0i0b2s_scalstore`（标量算术 + 标量 store，`probe_vf_loop/evidence/logs/build_nest_o0i0b2s_scalstore.log`）+ 正对照 `nest_o0i0b4_marker_ctl`（**COMPILE-OK**，`probe_vf_loop/evidence/logs/build_nest_o0i0b4_marker_ctl.log`）；行 d 的对应措辞按本行收窄 |
| i | **`Reg::Gather(dst, __ubuf__ base, idxReg, mask)`（vgather2）按元素索引，可寻址范围 = 本核 UB 窗口**：idx 越出源数组但仍在 UB 内 = **静默**读到该处 UB 的当前内容（不回绕、不异常）；`idx × sizeof(T) ≥ 256KB` → `507035`（通用异常码，只说明该组合异常、不指向具体指令）。实测 base 在 UB 偏移 0、fp32：`max_idx = 65535` 正常、`= 65536` 异常 | 【已确证】 | M43 `probe_vf_loop`（`ag_gather_sweep` 全档 47 档位；`start = 8191/8192/8193` 等任意非对齐起点、`stride ∈ {1,2,8,64,128}` 的源数组内 lane 全对） |
| j | **`Reg::Arange` 不支持任何无符号类型**：`static_assert((SupportType<ActualT, int8_t,int16_t,int32_t,float,half,int64_t>()))` ⇒ `Arange<uint32_t>` 编译失败（`current Arange data type is not supported on current device!`）。索引寄存器绕法 = `Arange<int32_t>` + `reinterpret_cast<RegTensor<uint32_t>&>` 交给 `Gather`（arch35/m18 同款，实测逐 lane 正确） | 【已确证】 | M43 `probe_vf_loop`（靶子 `ag_arange_u32` 原文见 `probe_vf_loop/evidence/logs/build_ag_arange_u32.log`，绕法走 `ag_gather_sweep` 正常路径）。**注**：行 e「`Arange` 只填 64 lane」是 **VL 语义**（另一件事），与本条"无符号类型不支持"**拆成两行**，不并条 |

**已撤回的语义结论登记（"我们用法待复核"）——禁止以 quirk 形式复活**

本仓**已有五次**"硬件/文档不一致"类结论最终以"**我们选错用法**"收场；凡此类结论**默认按"未隔离的观测"处理**，只有独立隔离复现过的才写成规则。以下更正记录**保留、不得静默删**：

1. **装载参数"静默不生效"** → M27 标定后订正为"参数**有效果**，只是单位/措辞与文档的写法需要换算（与官方文档 = 1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节））+ 越界/NOP 组合读到未写过的槽位"（§6.1 表格行）；
2. **`Duplicate(dst, scalar, 1)` 触发 `507015`** → M24 隔离探针推翻，真因是 **UB 目的地址 32B 对齐**（§6.1「DMA/UB 对齐」行）；
3. **`DeInterleave` 语义与文档不符** → M35 撤回，实为"**选错落盘指令**"（应用 `Cast<T, float, CAST_FP32_TO_FP16>` + `Reg::StoreAlign<T, DIST_PACK_B32>`；§6.1「位宽转换 Cast」行）；
4. **M27"与官方文档 4 处不一致"**（合并前的口径）→ M27 合并版已把它改写为 **1 处真分歧 + 2 条单位澄清 + 1 条真差异（细节）**（`m16_load_geom/README.md` §3.5 **表内 4 行**）：唯一的真分歧 = `srcStride`/`dstStride` 的**轴向措辞**（实测 `dstStride` = 目的 L0B 的 n 分形数 = `N/16`），`mStartPosition`/`kStartPosition` 的单位与 `startAddr` 公式两条是**单位等价而非分歧**，剩下一条即 `|srcStride|` 取绝对值 vs 实测按有符号用的**真差异（细节）**。⇒ 原先归入"参数语义/越界读"两类；**§6.1 表格行 ③ 与「M27 标定 4 条」块已按 §3.5 回填**；
5. **M35「AIC 标量写 UB → MTE3 落盘不成立」** → 结论方向成立，但**测量手段先坏**（§6.2「AIC 写 GM」行；M35 的 AIC GEMM 仍未收口）。**M153 根因归位**：所谓"测量手段先坏"的机制即 **AIC 不能「直接」访问 UB**（人类逐字：「这是L1 to UB，不是直接访问UB」），故「标量写 UB」这一步本身不成立（见 §6.1 ⓔ、§6.2「AIC 写 GM」行与 `probe_aic_gm_dma/README.md` §1.5）。

**M35 三条"API 不可靠"的降级【仍存疑·我们用法待复核·禁止以 quirk 复活】（2026-09-26，agent-qsaidx round-5 @ `09389b7`）**：`CompareScalar<int32_t,…>` / `UpdateMask(aiv)` / 对分数做 `EQ` —— M35 已**撤回"API 不可靠"的措辞**，改写为"**我们用法待复核**"。取证三步（① 文档定义里的 lane/mask/元素数/是否须在 `__VEC_SCOPE__` 内 → ② 试参数空间 → ③ 看官方 donor 同类位置）**尚未做完**（budget），如实报；代码已全部换成无争议的等价写法、**不再下语义结论**。⇒ 后续任何人引用这三条时**只能写成"我方用法待复核"**，不得以 quirk 或规则的形式复活（依据：§6.1 新标准 + 上列五次先例，使"我们用法不对"成为**先验概率最高的解释**；M35 曾把该取证顺序写进其 README（**M35 时点 @ `09389b7`**：§5「已知限制 / 后续」列表的第 **0** 条）并新增方法学条 **§3.4(4)**「先确认测量手段本身成立，再解释测量结果」——**这两个编号在 `m19_qsa_indexer/README.md` 被 M53 重写后已失效**（M73 实测 @ `569113d`：`grep -c '待办' m19_qsa_indexer/README.md` = 0、`grep -c '先确认' m19_qsa_indexer/README.md` = 0，范围 = 该 README 全文；§5 现为「收口状态」、§3.4 现为「单核 vs 多核的一致性论证」）⇒ **现文位置**：三步取证见该 README §5.2「移交 tower / 后续」第二条，那条方法学本身则现以 `docs/17` §9.1「测量手段先成立」的形态在仓内成立）。

**探针工程与证据**：`probe_sync_quirks/`（M23，main 合并后可见）——26 变体 MrgSort 矩阵 + 26 个 VF idx 变体 + 133 个归档证据文件（dump/编译日志/运行矩阵）+ `run_probes.sh` 一键复现。其余随附结论（`elementLengths` 单位=8B 对、`Sort32` 输出布局真值、`MrgSort4Info` 各字段静默失效条件等）见该工程 README §3–§6。

**用户对"仍存疑"条的裁定（2026-09-26）**：`ifExhaustedSuspension=true` 是另一种较复杂的行为——**4 个 source 中任何一个用完就停**；**本项目不需要该场景**（我们只用 `rep=1`、`exhausted=false`），无需继续深挖。其余存疑条（当年 src3/src4 半写原始形态、`repeatTimes>1` 语义、form5 双因子分离）维持"待需要时再看"。




## 7. 开发策略（已确认）

- **先打通 1 个层的总体功能，再细扣性能。**
- **第一个打通的单元 = MoE block 纵向切片**（router → 重排 → MXFP4 grouped GEMM×2 → SwiGLU → 反重排）：decode 带宽大头、donor 最丰富、天然覆盖"专家槽位切核 + 预取流水"核心机制；跑通后再前后扩到整层。
- **所有功能优先从现有代码库抄和改**（donor 清单见 §9）。
- 先手写、只针对这一个网络、暂不考虑泛化；但代码要**模块化**（op 级模块、清晰的 buffer/同步契约）。
- 暂不考虑 I-cache；暂不管投机解码；aclgraph 层后续再套（当前非重点；约束与开放问题见 §4.2）。
- **kernel 开发不依赖 torch**：纯 AscendC + cmake + bisheng 工具链即可开工，与 torch/vllm 环境搭建、模型下载并行。
- **golden 基准（与 `docs/17` §1 的口径统一，M49；引用口径按 M63 更新）**：逐 op 可用 **aclnn 现有算子**（CANN 自带）做**交叉校验 oracle**，整层对食用 torch CPU / numpy float 参考。**澄清**：`docs/17` §1 要求 L0 判定项的参考必须**独立**，其判法自 M63 起为**两层** —— **规则**来源独立（`docs/17` §1.2：须钉在非本设备实现的工件上并写出 `文件:符号`）＋ **输入**来源合规（`docs/17` §1.3 的三分）；旧句「不许拿设备自己的中间输出当参考」**已废止**（它把「声明输入 / 上游输出 / 被判量自身产物」混成一条）。**aclnn 是厂商 device 实现，属独立第三方 oracle（即 §1.2 的 N3 异实现交叉），不构成"拿设备自己的输出当参考"**，因此可用；但它**不建模每一步舍入、不能单独充当位级 / ≤1 ulp 的 L0 判定参考**，且与"逐句复刻参考"冲突时**以后者为准**（实例：M32 的 ±Inf 分支上，host 逐句参考比设备更接近官方）。

## 8. 建议里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M0 | 工程骨架：cmake + `<<<>>>` 直调 + mix 1:2 启动，28 AIC + 56 AIV 跑通核间/核内同步最小示例 | 各核 id/角色正确 |
| M1 | MXFP4 GEMM 单核移植（basic API，直接消费原始 nibble-packed layout），参考 asc-devkit mxfp4 样例 | 数值与 golden 一致 |
| M1.5 | **MoE block 纵向切片**：router → 重排 → grouped MXFP4 GEMM ×2 → SwiGLU → 反重排，多核（专家槽位 × K/N 切分） | m=1 与 m=4097 数值对齐 golden。**m=4097 处：自建稠密参考、与官方 QSA 固有差异**（见下注*） |
| M2 | layer 骨架：op 序列框架 + BufferID 管理 + CrossCore（mode 0/mode 2）同步框架 | 骨架可运行、同步正确 |
| M3 | 逐 op 填充（GDN 或 attention → MoE → norm/proj），整层 | m=1 与 m=4097 整层输出对齐 golden。**m=4097 处：自建稠密参考、与官方 QSA 固有差异**（见下注*） |
| M4 | 48 层循环打通（host 循环，aclgraph 后补） | 整网输出对齐 golden。**decode（context=4096）处：自建稠密参考、与官方 QSA 固有差异**（见下注*） |
| M5 | 性能细扣：跨 op 预取流水、带宽效率、负载均衡 | decode TPS 对比基线 |

> **\* 强制标注（`docs/17` §7 附则 1，2026-09-26 tower 裁决）**：凡涉及 **m=4097（prefill）或 decode-ctx4096** 的数值验收，其基准是**自建稠密 causal 参考**，**与官方 vLLM 的 QSA 输出存在固有差异**——官方 QSA 的 `indexer_budget=2048` 是 **token** 预算 ⇒ `block_topk = 2048/4 = 512` 块，而 4097 上下文的可见块有 ⌈4097/4⌉ = **1025** 个，**官方只 attend 约一半历史 token**（decode context=4096 同理）。gather 集合与 softmax 分母都不同 ⇒ 稠密路径与官方输出**逐位不可能一致**。**禁止**再写"m=4097 对齐官方输出"。依据与裁决全文见 `docs/17` §7。

## 9. 优先抄改的 donor 代码（/workspace 现有资产）

| 功能 | donor | 说明 |
|---|---|---|
| MXFP4 GEMM（basic API，原始 layout） | `asc-devkit/examples/01_simd_cpp_api/05_best_practices/01_matrix_compute/matmul_mxfp4_high_performance/`、`matmul_mxfp4_basic_api_high_performance/`（`mmad_mx.asc`） | 950PR 官方调优样例，直接消费 packed MXFP4，与目标模型最直接相关 |
| MXFP4 GEMM 生产级 tiling | `ops-nn/matmul/quant_batch_matmul_v4/`（arch35 目录） | 只抄 tiling/同步结构，计算改写为基础 API |
| MoE 全流程 | `ops-transformer/mc2/mega_moe/` | Dispatch+Linear+Act+Linear+Combine 单算子融合参考 |
| token 重排 / grouped matmul | `ops-transformer/gmm/` | 专家槽位静态分组的运行时结构 |
| 全量化 attention | `ops-transformer/experimental/attention/quant_flash_attn/`（arch35） | MXFP4 attention 内核参考 |
| 基础算子（norm/eltwise/softmax/router） | `ops-math/`、`ops-nn/activation/`、`ops-nn/norm/` | 按算子目录抄改 |
| GDN 线性注意力 | `vllm-ascend/vllm_ascend/ops/gdn.py`（逻辑参考）+ ops-transformer `mamba/` 类算子 | GDN 内核参考 |
| 同步/BufferID 用法 | 950PR 平台卡 + CANN 9.1 API 文档 | 待查 get_buf/rel_buf 确切签名、BufferID 数量上限 |

## 10. 遗留待查证事项（2026-09-26 M2 survey 后更新）

1. ~~mix 启动下 `GetBlockIdx()` 在 AIC/AIV 侧的索引语义与配对~~——**已确认**：官方 mix 样例 `matmul_leakyrelu_basic_api.asc` 实证 AIC `GetBlockIdx()`=0..numBlocks-1，AIV=0..2·numBlocks-1，配对 AIC=`AIV bid/2`；blockDim=numBlocks=AIC 数。
2. ~~BufferID 数量上限与签名~~——**已确认**：见 §4.1/§6（每核用户 0-27，get/rls_buf + Mutex 封装，mode 语义）。
3. ~~mode 0 跨类型（AIC 组→AIV 组）语义~~——**已确认**：mode 0 仅同类型；跨类型 all-to-all 用 mode2+mode0(+mode2) 组合（§2）。
4. ~~BufferID 在 CANN 9.1.0 的实际行为~~——survey 给出语义结论；**token 初始空闲 + 同 pipe 重获取仍由 M0 任务 5 例行实证**。
5. ~~自管理地址调用基础 API~~——已确认：裸指针/`LocalTensor(position,offset,size)`（asc-devkit 官方样例同款）。
6. ~~同 pipe BufferID 重获取~~——survey 结论：同 pipe rls→get 立即可得、顺序由 pipe FIFO 保证；M0 任务 5 实证后关闭。

## 11. 抄改关系登记表（复制改造的依赖管理；2026-09-26 M55 立）

**为什么需要这张表**：`docs/17` §3 第 4 条要求「复制改造的代码必须给出与上游的**逐文件差异表**，并声明**上游变更不会自动流入**」。
本项目的算件几乎全部是**从别的模块复制改造**而来（M18/M21/M22/M25/M29/M40…），而**副本与源往往同处一个仓库**——
在评审里看起来"像是一份东西"，但上游改了算件后**所有副本都不会自动跟随**。把**已知**的抄改关系登记在此，
并定一条纪律，使"源变更"这件事**可被机械地传导到依赖方**。

**纪律（本表的使用方式）**

1. 任何人**新增**一处复制改造（含脚本化抽取），须在提交时**在本表加一行**：源路径、抄改时源 commit、目标路径、同步方式、当前状态。
2. 任何人**修改表内的源文件**（或改变其算件代码），须在 review-request 里**点明"本表第 N 行受影响"** ——
   **由 tower 据本表列出依赖方，并逐一确认**处置是「重新抽取 / 手工同步 / 声明不跟随」。
3. **同步方式**分三类：`脚本抽取`（可机械重放，**必须附 `--check`**）／`手工同步`（**须给逐文件差异表**）／`声明不跟随`（**须写明理由与适用范围**）。
4. 表内「当前是否已与源同步」一栏必须**写成三元组**：**判据命令 + 判据跑在哪个 commit 上 + 当时的读数**
   （只写"已同步/未同步"而不写"在哪一支上测的"，等于把一个**有寿命的观测**写成**永久结论** —— 见纪律 6 与 §11.5）。
5. **祖先关系不能用来推断"抄的是哪一版"**（见 §11.3 的反例：分支 rebase 会把源的新进展带进祖先链，而拷贝内容仍是旧形态）——
   抄改时源 commit 必须**按内容定**，不能用 `merge-base --is-ancestor` 定。
6. **本表会自己过期，且过期窗口可以很短**：重抽类同步关系在「源已合入、目标分支尚未 rebase」的窗口里，
   状态可能在**几十秒内**翻转（M55 首轮实测：交卷后 **27 秒** M52 合入 main，把行 1 从"未同步"翻成"已同步"）。
   故：① 交卷前**重跑一次**判据并写明它跑在哪个 commit 上；② 若读者在**另一个** main 上复现，**以复现结果为准**，
   本节不主张"当时写的就一定还对"；③ 改完/合并后回填本表（见 §11.5）。

**登记表（已知 3 处）**

| # | 源文件 | 抄改时源 commit | 目标文件 | 同步方式 | 当前是否已与源同步 |
|---|---|---|---|---|---|
| 1 | `m13_moe_layer/m13_moe_layer.asc`（device 段） | `e93f5219c26a8d9d2bae78d1ccf351aa6d164e5b`（首抽）/ 现随 m13 到 `5ddaee22…` | `m15_layer_loop/m15_moe_layer.h` | **脚本抽取** | **已同步**（M52 `a06760c` 已合入；判据 `--check` rc=0） |
| 2 | `m8_permute/m8_permute.asc`（M15 的 permute/unpermute） | `8121b1d43ba0387838b38e46353ab4d104dedf51` | `m13_moe_layer.asc` S4/S9、`m17_moe_layer.asc` S4/S9、`m15_moe_layer.h` S4/S9（3 个副本） | **手工改写**（小改 A/B） | 源侧**无漂移**；副本三份改写形态各不相同 |
| 3 | `m7_router_topk/m7_router_topk.asc`（S2 router） | `b3bef2a6d9ac85a569b3f1d1f1b4e14d86149c84`（**M48 之前**的 m7 形态） | `m17_moe_real/m17_moe_layer.asc` S2 | **声明不跟随**（设计使然的分化） | **未同步（分化）**；S2 仍是 pre-M48 形态 |

### 11.1 行 1 —— m13 MoE 段 → m15_layer_loop（M40，脚本抽取）

| 项 | 值 |
|---|---|
| 源文件 | `m13_moe_layer/m13_moe_layer.asc` 的 **device 段**（**内容锚点**：首个顶层 `namespace {` 到层 kernel 入口前最后一个 `}  // namespace`；**不要用裸行号**——M32 曾给 m13 加 +454 行，行号锚点会静默错位） |
| 抄改时源 commit | `e93f5219c26a8d9d2bae78d1ccf351aa6d164e5b`（**M40 首抽时**的 M13 tip；`git merge-base --is-ancestor e93f521 c39ad01` = YES）。**M52 重抽后跟随 `5ddaee22c3d1fde6240d73af50a57e5678b333fa`**（M50） |
| 目标文件 | `m15_layer_loop/m15_moe_layer.h` |
| 抄改 commit | `c39ad01311b87df7f253a6645b2192b76ca26082`（M40 首抽）；`3623c347…`（紧凑专家槽）/`6a5740cd…`/`bf592f9f…`；**`a06760cf56d54e48cb4d539d95d4820270528230`（M52 重抽，已随 main `f0286f6` 合入）** |
| 同步方式 | **脚本抽取** `m15_layer_loop/lift_moe_segment.py`。首抽时为「7 类机械替换」（namespace `{`→`M15M`、删 `using namespace M13;`、include 改名、`M13`→`M15M`、出口 y 独立成 GM 缓冲、紧凑专家槽 Σt_e、经典 `Extract`→VF 内联）；**M52 把第 7 条由"替换"降级为断言** `assert_upstream_vf()` —— 因为 M50 已在 m13 侧把 `Extract` VF 化，那条替换规则的**目标文本已不复存在**（旧规则因此不可重放） |
| **当前是否已与源同步** | **已同步（判据可复算）**：M52 的 `a06760c` 已合入 main（`git merge-base --is-ancestor a06760c main` = YES；`git log -1 main -- m15_layer_loop/m15_moe_layer.h` = `a06760c`），副本已跟随 m13 在 `5ddaee22…`（M50）的形态。判据：`python3 m15_layer_loop/lift_moe_segment.py --check` → **rc=0**：`[lift] m15_moe_layer.h 与抽取规则一致（内容锚点：asc 行 62..2031，入口在第 2033 行）`。**「m13 已到 `5ddaee22…`」这一点在重抽后仍然成立**（重抽正是为了跟上它） |
| M52 的复核读数（`m15_layer_loop/evidence/`） | **1131 个 dump 张量 sha256 与重抽前逐位相同**（`m52_dump_sha256_ab.log`）⇒ 本轮是**纯机械搬运、0 处舍入点改动**；device 段差异 **4 个 hunk / +29−14**（`m52_relift_diff.txt`，全落在 S2 router 内）；`assert_upstream_vf()` 与 `--check` 做了 **1 条正对照 + 3 条负向对照 + 1 条匹配器对照**（`m52_lift_negative_controls.log`）；判定项 `checks=1165, guards=109, fails=0`（`accept_run_m52.log`） |
| 在办 | **已完成**：M52 `a06760c` 已合入 main（`f0286f6`）。**首轮交卷时本行写的是"未同步"—— 那是一次 27 秒的竞态**（见纪律 6） |
| 风险度（本行） | **无遗留**：可机械复核的同步关系已恢复（`--check` 现在能证明副本与源一致，且 M52 用**负向对照**证明这条断言不是恒真）。长期可跟踪点：**m13 每前进一次就要重抽一次** —— 这正是 `--check` 常驻的意义 |

### 11.2 行 2 —— M15 的 MoE permute/unpermute 副本（m8_permute → m13 / m17 / m15）

| 项 | 值 |
|---|---|
| 源文件 | `m8_permute/m8_permute.asc`（M15 产物：`MoePermuteKernel` / `MoeUnpermuteKernel`）。**注**：m8 自身是**外部 donor 的抄改**（`ops-transformer` 的 `gather_v2_simd_two_dim.h` / `moe_token_unpermute_with_routing_map_not_pad.h`）——那是外部快照、无本仓 commit，不在本表登记范围 |
| 抄改时源 commit | `8121b1d43ba0387838b38e46353ab4d104dedf51`（M15 末次改动；`git merge-base --is-ancestor 8121b1d 9e5c753` = YES） |
| 目标文件（**3 个副本**） | ① `m13_moe_layer/m13_moe_layer.asc` 的 S4 `PermuteStage` / S9 `UnpermuteStage`+`FmaChunk`；② `m17_moe_real/m17_moe_layer.asc` 的 S4/S9；③ `m15_layer_loop/m15_moe_layer.h` 的 S4/S9 |
| 抄改 commit | ① `9e5c753162a343ce60acb282be5e5eb400b0a0e4`（M13 skeleton）；② `581eb62eeb695c9c337edb0852f7a91b77712719`（M29/M17）；③ `c39ad013…`（M40，经行 1 的脚本从 m13 传递） |
| 同步方式 | **手工改写（非逐字）**：小改 A（元素计数 `DataCopy` 重载 → 显式 `DataCopyParams`/`DataCopyExtParams{1,len,0,0}`）+ 小改 B（同 pipe/同 buffer 背靠背 MTE2 复用处补 `PipeBarrier<PIPE_MTE2>`），见 `docs/12` §6 |
| **当前是否已与源同步** | **源侧无漂移**（按纪律 4 / §11.5 的三元组写法：**判据命令 + 判据跑在哪个 commit 上 + 当时的读数**）：`git log --oneline 8121b1d43ba0387838b38e46353ab4d104dedf51..569113d43d865c37c7b4773497776838eba365aa -- m8_permute/m8_permute.asc` **为空**（读数的两个 ref 都写实；上界 = 取该读数时的 `main` tip）。⇒ m8 自 `8121b1d` 起未再变更，三个副本的上游内容都没变。**要在当前 `main` 上复算**就把上界换成 `main`（与行 1 的 `git merge-base --is-ancestor a06760c main` 同型；按纪律 6，以复现结果为准）。但**三个副本的改写形态互不相同**（同一算件的三份改写）⇒ **源一旦变更，须三处一起看**，不能只同步一处 |

### 11.3 行 3 —— m7 router → m17 S2（M29/M17 的分化）

| 项 | 值 |
|---|---|
| 源文件 | `m7_router_topk/m7_router_topk.asc` 的 S2 router（GEMV → softmax(max-shift) → Sort32/归并树 → top-k → renorm） |
| **抄改时源 commit** | **`b3bef2a6d9ac85a569b3f1d1f1b4e14d86149c84`**（`m7_router_topk: router softmax topk kernel`，**M48 之前**的 m7 形态） |
| 判据（为什么是这一版） | m17 的 S2 里是**经典 memory-based** `Extract(vT, iT, mT, 2);`，与 `git show b3bef2a:m7_router_topk/m7_router_topk.asc` 里的同名调用**逐行同形**；而 m7 自 **M48 `2a07218`**（`m7 向量 API 合规改造：Extract VF 内联`）起已改为 `LoadAlign<DIST_DINTLV_B32>` 的寄存器形态（`git show 2a07218:m7_router_topk/m7_router_topk.asc` 可见） |
| ⚠ 反例（并入纪律 5） | `2a07218` **是** `581eb62` 的祖先（`git merge-base --is-ancestor` = YES）——分支 rebase 把 M48 带进了 m17 的祖先链，**但 m17 的拷贝内容仍是 pre-M48 形态** ⇒ **祖先关系不能用来推断"抄的是哪一版"**。这与 `docs/17` §9.5（两点 diff 在 rebase 后失真）是同一族问题：**rebase 之后，"历史关系"与"内容关系"必须分开看** |
| 目标文件 | `m17_moe_real/m17_moe_layer.asc` 的 S2 段 |
| 抄改 commit | `581eb62eeb695c9c337edb0852f7a91b77712719` |
| 同步方式 | **声明不跟随**（设计使然的分化）：m17 README §1 列 **10 条刻意差异**（E=4→512、TOPK≤4→10、router 全量常驻 UB→流式、单块 Sort32→16×Sort32+4 级归并树、索引生成 UB 化、k 链展开、诊断槽、`RT_RB`、`w_tk` 清零、±Inf 平价）。其中**第 10 条是一次手工同步的实例**（m13/m5 的 ±Inf 平价修复被补进 m17） |
| **当前是否已与源同步** | **未同步（分化）**：m17 S2 仍留在 m7 的 **pre-M48** 形态（经典 `Extract`）。m17 README §7 第 13 条已披露"已有模块先报不改"——改造须**另立 mission 并重跑全套验收**（当前 dump sha256 `1a0f4883…` 建立在**现有二进制**上） |

### 11.4 本表已经能说明的一件事：同一算件的多份改写

同一处 `Extract`（Sort32/归并树输出的 (value, index) 交织对拆分）现在在仓库里有 **4 个文件、2 种形态**（VF 3 处来自**三个不同来源**，经典 1 处）：

| 文件 | 形态 | 来源 |
|---|---|---|
| `m13_moe_layer.asc`（main `5ddaee2`） | **VF**（`DIST_DINTLV_B32`） | M50 内联 |
| `m15_moe_layer.h`（main `a06760c`） | **VF**（`DIST_DINTLV_B32`） | **M52 重抽后随 m13/M50 形态**（M40 那版是抽取脚本"规则 7"手写内联，已被取代 —— 见 §11.1） |
| `m7_router_topk.asc`（main） | **VF** | M48 `2a07218` 内联 |
| `m17_moe_layer.asc`（main `581eb62`） | **经典 memory-based `Extract`** | 抄自 m7 的 **pre-M48** 形态（行 3） |

⇒ 若只看"文件在不在"或"祖先关系"，这四处**完全看不出来**有差别；本表 + `docs/17` §3 第 4 条（逐文件差异表）是让它**可见**的机制。
**它不是"必须立刻统一"的待办**（M44 已裁"已有模块先报不改"），而是**源变更时必须被逐一确认的依赖清单**。

### 11.5 抄改关系的**时间窗**：这条表会自己过期（M55 首轮的真实教训）

本表首轮交卷时把行 1 写成「**未同步**（`--check` rc=1）」，**依据是我当时分支 base（`ea4c818`）上的实测**；
而 **27 秒后**，做重抽的 M52 就合入了 main（`f0286f6`）—— 评审在**新 main** 上复跑 `--check` 得 **rc=0**，
于是那句"未同步"**在交卷那一刻就已经是错的**。

**教训（并入纪律 4）**：重抽/同步这类关系，在「**源已合入、而目标分支尚未 rebase**」的窗口里，状态会**在几秒到几分钟内翻转**。
因此「当前是否已与源同步」一栏**必须写成三元组**：

> **判据命令 + 判据跑在哪个 commit 上 + 当时的读数**

只写"已同步/未同步"而不写"在哪一支上测的"，就等于把一个**有寿命的观测**写成了**永久结论**。
本表其余各行的判据也按这个格式给（命令 + `<commit>` 或"分支 tip"）。

**同一族**：这与 `docs/17` §9.5（两点 diff 在 rebase 后失真）、§8.1（**一句话原则**：活值不得以字面量形式出现在断言或打印里，须带采集时刻）
是同一个毛病 —— **把"某一刻的读数"当成"一直成立的事实"**。
