# M0 Bring-up 实录：修复、真机验证与 950 同步机制实证

> 记录日期：2026-09-26。环境：Ascend950PR（npu-smi 25.7.rc1.6，PG 降频 binning 版），CANN 9.1.0（cann-9.1.0），bisheng 15.0.5，容器内 NPU 对 `aclrtSetDevice(0)` 可见。
> 代码：`m0/m0_bringup.asc`（分支 `feat/m0-bringup-fix-and-verify`，**已随 `1003e4f` 合入 main**）。

## 1. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/m0_bringup        # 退出码 0 = 全 PASS
```

## 2. 核数探测（aclrtGetDeviceInfo）

| 属性 | 返回值 |
|---|---|
| `ACL_DEV_ATTR_AICORE_CORE_NUM` | 28 |
| `ACL_DEV_ATTR_VECTOR_CORE_NUM` | 56（= 2×28，mix 比例 1:2 吻合） |

`blockDim=28` 启动 `__mix__(1,2)` kernel：28 AIC + 56 AIV 同核组运行。核数必须由此 API 获取，禁止硬编码（满配 die 为 32 AIC + 64 AIV）。

## 3. 五项验证结果（连续 10 轮全部 ALL PASS）

```
[M0] AICORE=28 VECTOR=56 -> numBlocks=28 (expect VECTOR=2*AICORE)
[M0] kernel launch+sync: OK
[M0] PASS role probe: 56 AIVs DMA-transfer their bid-tagged slice
[M0] PASS BufferID ping-pong accumulation (8 iters, 56 AIVs)
[M0] PASS CrossCore mode0 all-AIV reduction
[M0] PASS CrossCore mode2 AIC(Mmad+FIXP)->AIV(+1) (28 pairs)
[M0] PASS same-pipe BufferID re-acquire (get/MTE2/rls/get/MTE2/rls, 2nd round data correct)
[M0] ===== ALL PASS =====
```

| # | 检查 | 内容 | 数据要点 |
|---|---|---|---|
| 1 | 角色探测 | 每个 AIV 用一次 MTE2 读 + 一次 MTE3 写，把自己 bid 对应的唯一数据切片搬运到 roleOut | 56/56 条记录逐一精确匹配；host 端按 `roleIn[bid*8+i]` 校验，bid 错位即失败 |
| 2 | BufferID 乒乓累加 | MTE2 生产 ↔ VEC 消费，`Mutex::Lock/Unlock<PING/PONG>` 乒乓协议，8 轮迭代把 8 个 4KB tile 累加进 UB | 56 AIV × 1024 float 与 golden（`Σ_it DataFormula(b,it,i)`）全部一致 |
| 3 | CrossCore mode0 汇总 | 56 AIV 各把 16 float 写入 GM 槽位，mode0 屏障后每核读回全部 56×16 并纵向求和 | 每核 `totalOut` = 全体 56 核 × 8 iter × 16 路的总和，与 host golden 一致 |
| 4 | CrossCore mode2 | AIC 做 16×16 half Mmad→FIXP 写 GM，mode2 通知配对 2 个 AIV，AIV 读回 +1.0 写回 | 28 组 AIC→2 AIV，输出与 host 端 fp32 参考（half 舍入容差 2e-3）一致 |
| 5 | 同 pipe BufferID 重获取 | 同一 buffer id 连续两轮 `get→MTE2 DataCopy→rls→get→MTE2→rls`，中间无其它 pipe 参与；第二轮 tile 序号由 `GetBlockNum()%8` 派生（顺带验证 GetBlockNum=28） | 不挂死；第二轮数据（tile4）经 MTE3 读出逐元素正确 |

## 4. 同 pipe BufferID 重获取结论（用户提出的疑问，实测回答）

用户原疑问：`get buf 0 → MTE2 → rls buf 0 → get buf 0 → MTE2 // 不知道是否会等待前面的 MTE2 结束后再启动，应该会`

**实测回答（round-2 决定性数据）：不会等待——同 pipe 背靠背两条 MTE2 DataCopy 的完成序可颠倒，两轮之间必须 `PipeBarrier<PIPE_MTE2>` 排空。** 数据：去掉排空连跑 60/60 轮有错、加排空 0/60 轮全对；出错呈 ~20% AIV-轮分布、**错元素 100% 恒等于第一轮数据**、32B burst 粒度交错；`Mutex::Lock/Unlock` 的 get/rls 用任意 mode 均不保序（**当时称作 drain 的 `true` 档**实测无效；该轮 release 侧档位为 `true`，M181/M182/M183 已统一为 `false`）。即 BufferID 只表达占有/释放互斥，**同 pipe 完成次序不受其保证**。

排错记录（两处归因，独立且都必需）：
1. **同 pipe 完成乱序**：两轮 MTE2 写同一 buffer，第二轮的完成可能先于第一轮（WAW 交错，32B burst 粒度）→ 读出内容混有/退回第一轮数据；必须 `PipeBarrier<PIPE_MTE2>`（round-2 数据：无排空 60/60 有错、有排空 0/60）。
2. **跨 pipe 读出同步**：MTE3 从 UB 读 c5buf 必须用 `MTE2_MTE3` 事件——PipeBarrier 不保证跨 pipe 数据路径可见。

正面结论（保留）：跨 pipe 交接可用 **BufferID 阻塞释放**（`false`=CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地；MTE2→V 方向 10/10 轮实证，当时档位为 `true`）**或 X_Y 事件**；两者都能表达"生产完成再消费"。本检查采用事件 + PipeBarrier 组合。

> 更正记录：本轮曾得出"FIFO 保序、无需排空"的错误结论（60 轮对照实验实为 stale binary 所致，提交 0458980 的"check5 简化"未实际进入代码），以 round-2 复审数据与本节为准。

### 4.1 官方口径 pin（非设备、非自指；M128 补）

本节 §4 的 60/60 vs 0/60 是本机实测；下列为本轮补入的**官方文档 / kernel 源码**口径，逐条给 `文件:行` 与逐字引文。来源均为仓外官方仓库（`asc-devkit`、`ops-transformer`）。

**（一）同一 id 且同一 pipe 的连续两对 Lock/Unlock 不保同 pipe 次序（官方明确）**

- C++：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Lock.md:93` 逐字——「当具有相同id与pipe的两对Lock与Unlock连续调用时，第一次调用的Lock将由参数pipe指定的流水阻塞后，第二次调用的Lock不能再次阻塞该流水。换言之，连续调用的、具有相同id与pipe的两对Lock与Unlock**不能实现单流水（参数pipe指定）内不同指令之间的同步，单流水内多个指令之间的同步请使用PipeBarrier接口**。」官方反例见同文 `:95`（两次搬运 UB 目的地址重叠），正确写法在同文 `:100-131` 的 `CopyInY`/`CopyInX` 之间插入 `PipeBarrier<PIPE_MTE2>()`（`:123`）。
- C API（同口径）：`asc-devkit/docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_lock.md:101` 逐字同一句（把 `PipeBarrier` 换成 `asc_sync_pipe`）；反例说明见 `:103`，正确写法在同文 `:105-129` 的 `CopyInY`/`CopyInX` 之间插入 `asc_sync_pipe(PIPE_MTE2)`（`:128`）。
- 旗标形态的同源风险：`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md:99` 逐字——「相同流水、相同eventID下，连续使用SetFlag会引发未定义行为，此时再执行PipeBarrier<PIPE_ALL>会出现卡死现象：」，反例代码见 `:101-109`。**本条以官方原文口径登记；它可能与 M117 挂死相关，本轮不据此替 M126 下根因结论。**

**（二）同一 BufferID 跨两个 pipe 有官方用法（本仓「跨 pipe 交接用 BufferID 阻塞释放」的官方依据）**

- 官方双缓冲指南的同步关系表里，同一 `outputMutexId` 在 **PIPE_V（写）与 PIPE_MTE3（搬）** 两 pipe 上交接——`asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md:189-190` 逐字：「…计算完成后才能搬出结果。| 使用该组的输出`mutex_id`。」与「…前一个数据块的结果搬出完成后，才能复用同组输出缓冲区。| 下一次Vector获取同一输出`mutex_id`时等待。」；该文 `:158-174` 的代码即此形态（同 `outputMutexId` 依次 `asc_lock/asc_unlock(PIPE_V, …)` 与 `asc_lock/asc_unlock(PIPE_MTE3, …)`）。
- 官方样例同形——C API `asc-devkit/examples/02_simd_c_api/02_features/01_reg_vector_compute/00_add_double_buffer/add_double_buffer.asc:151-169`（同一 `mutex_id` 依次挂 `PIPE_MTE2`→`PIPE_V`→`PIPE_MTE3`；双缓冲版在 `:230-250` 用 `input_mutex_id`/`output_mutex_id`）；C++ `asc-devkit/examples/01_simd_cpp_api/03_basic_api/05_sync_control/mutex/mutex.asc:76-100`。
- framework impl（dav_3510 = Ascend 950）同一 `bufId` 跨两 pipe——`asc-devkit/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h:832-835` 逐字：`GetBuffImpl<PIPE_V, true>(0);` / `ReleaseBuffImpl<PIPE_V, true>(0);` / `GetBuffImpl<PIPE_MTE3, false>(0);` / `ReleaseBuffImpl<PIPE_MTE3, false>(0);`（`GetBuffImpl`/`ReleaseBuffImpl` 定义在 `asc-devkit/impl/basic_api/kernel_event.h:847-865`，分别落到 `get_buf`/`rls_buf`，见同文 `:764-800`）。
- 官方 ops-transformer 把同一条 UB→GM 收尾**同时**给出 `Mutex` 支与 `SetFlag/WaitFlag` 支（`ENABLE_LOCK` 切换）——`ops-transformer/attention/common/op_kernel/init_output.h:45-78`（Mutex 支：`Mutex::Lock/Unlock<PIPE_V>(SYNC_ID)` → 写 UB → `Mutex::Lock/Unlock<PIPE_MTE3>(SYNC_ID)` → 搬 GM）。**本仓只采用 Mutex/BufferID 那一支**，§5.3 的 set/wait 禁令不变。
- **收尾 release 落在消费/搬出 pipe**：UB→GM 为 **PIPE_MTE3**（`enable_double_buffer.md:172-174`、`init_output.h:59,78`）。官方释放语义逐字——`asc-devkit/docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_unlock.md:34`：「`asc_unlock`：指定流水的前序指令执行完成后，根据`mutex_id`释放对应Mutex。」C++ 同义——`asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Unlock.md:32`：「指定流水的前序指令执行完成后，根据MutexID释放对应Mutex。」
- **窄结论（命名）**：在 `asc-devkit/docs/` 全树以 `grep -rn -i "drain" docs/`（英文）与 `grep -rn "排空" docs/`（中文）检索，**未命中名为 “drain release” 的 API**；`排空` 在该范围内只出现在 `enable_double_buffer.md:177,199`（均指 kernel 退出前的 `asc_sync_pipe(PIPE_ALL)`），英文 `drain` 无命中。⇒ 本仓 `BufRelease<PIPE>(id)`（= `RlsBufInternal<pipe,false>`，`false`=CANN `ASC_LOCK_BLOCK` 默认「阻塞」；M186 已把全项目 release 统一为 `false`，旧注释「release 延迟到 pipe drain」已同步改）是**本仓私有包装、不是官方 API 名**；官方公开面是 `asc_lock/asc_unlock` 与 `Mutex::Lock/Unlock`，`get_buf`/`rls_buf` 在该范围内无公开 API 文档页（只出现在 `asc-devkit/impl/basic_api/kernel_event.h:764-800` 等 impl 文件）。

### 4.2 已结 —— `asc_lock/asc_unlock` 的 `mode` 命名（M126 真机对照已关闭该未决项）

- **官方面**：`asc-devkit/docs/zh/api/SIMD-API/c_api/sync/intra_core_sync/asc_lock.md:59-62` 定义 `enum asc_mutex_execute_mode { ASC_LOCK_BLOCK = 0, ASC_LOCK_NON_BLOCK = 1 };`。**同一条 `mode` 语义，官方两页的措辞并不一致**：`asc_unlock.md:54` 写 `ASC_LOCK_BLOCK`＝「该指令等待`pipe`所对应的流水线中所有前置指令完成后执行」、`ASC_LOCK_NON_BLOCK`＝「该指令等待`pipe`所对应的流水线中所有前置指令完成且相同`mutex_id`的所有`asc_unlock`指令执行完成后执行」；而 `asc_lock.md:54` 写 `ASC_LOCK_BLOCK`＝「阻塞`pipe`对应流水的执行，直到代码中位于当前`asc_lock`之前且`mutex_id`相同的所有`asc_unlock`调用均已执行完成」、`ASC_LOCK_NON_BLOCK`＝「不阻塞`pipe`对应流水的执行」。底层映射（impl）：`asc-devkit/impl/c_api/reg_base_impl/sync_intf_impl.h:187-295`——`ASC_LOCK_BLOCK`→`get_buf/rls_buf(..., false)`、`ASC_LOCK_NON_BLOCK`→`(..., true)`；两参重载默认 `ASC_LOCK_BLOCK`（`:240`、`:295`）。⇒ **官方把底层 `mode=true` 命名为 “NON_BLOCK”**。
- **本仓面（M186 订正后）**：本仓 `docs/05` §2/§6 现按权威口径写：`false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」、`true` = `ASC_LOCK_NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**，`true` 额外等此前同 id 的释放（更保守）；**全项目 release 一律 `false`**。公开 `Mutex::Lock/Unlock` 把 mode 硬编码为 0（`asc-devkit/include/basic_api/kernel_common.h:141-163` 的 `GetBufInternal<pipe, 0>` / `RlsBufInternal<pipe, 0>`）——mode 0 **就是** `false`，与 `BufRelease<pipe>` = `RlsBufInternal<pipe,false>` **同一种模式**。
- ⇒ **该未决项已由 M126 真机对照结掉**：`asc_lock/asc_unlock(mode)` 的 `BLOCK/NON_BLOCK` 与底层 bool **一一对应、无反转**（`BLOCK`=`false`、`NON_BLOCK`=`true`），不是「true=drain」那种命名反转；`asc_unlock.md:54` 逐字只对 `ASC_LOCK_BLOCK` 讲「等待 pipe 对应流水线的所有前置指令完成」，与「两种模式都等本 pipe 落地、`true` 额外等此前同 id 的释放」一致。依据 = `probe_ub_bufid_tail/README.md` §0.1 第 11 条 / §8.1。本仓既有「BufferID 交接 10/10 PASS」等实测**不改写、不撤销**（当时档位为 `RlsBufInternal<pipe,true>`）；**M181/M182/M183 已把全项目 release 统一为 `false`**。

## 5. 过程中发现的问题与解决办法（按发现顺序）

### 5.1 编译期（首轮 20 个编译错误 → 0）

| 问题 | 修正 |
|---|---|
| `AscendC::PIPE_MTE2/PIPE_V/PIPE_MTE3/PIPE_FIX/PIPE_ALL` 不存在 | `pipe_t` 是**全局枚举**（cce_aicore_intrinsics.h），去掉 `AscendC::` 前缀；`EVENT_ID0..7`、`QuantMode_t`、`half` 同理 |
| `roleOutGM[i] = v` 报 `no viable overloaded '='` | `GlobalTensor::operator[]` 返回的是**偏移视图**（给 DataCopy 用），不是元素引用；元素写用 `SetValue(offset, v)` |
| `Adds<uint16_t>` / `LoadData(2D)` 静态断言失败 | 3510 只支持 `half/bf16/float/int32…`；uint16 缓冲用 `ReinterpretCast<half>()` 同址换型视图 |
| `AscendC::QuantMode_t::F322F16` | `QuantMode_t` 也是全局枚举；Fixpipe dst 的 dtype 元组只认 `half` 等（dst 需 `GlobalTensor<half>` 视图） |
| GM 目的 `DataCopyPad` 4 参重载不存在 | 3510 上 UB→GM 的 DataCopyPad 只有 `(dstGM, srcUB, DataCopyExtParams)` 三参重载 |

### 5.2 运行期（挂死 / 数据错 / 写丢失 → 全 PASS）

排查动线：先以 `M0_STAGE_LIMIT` 宏逐阶段隔离（定位到各阶段独立问题），再用最小探针二分。关键结论：

1. **`LocalTensor(pos, addr, size)` 的 `addr` 单位是字节，`size` 单位是元素**（`kernel_tensor_impl.h::CreateTensor` 中 `addr` 直接作 `bufferAddr`，按 32B 对齐检查）。原实现把整个 UB 布局表当"元素偏移"传入 → 所有缓冲区互相混叠（偏移 X 实际落在第 X/4 个 float 处），表现为"同步失灵/数据随机错乱/第 64 元素后损坏"等一系列诡异现象。**全部 UB/L1 偏移字节化后，大部分"灵异问题"同时消失。**
2. **跨 pipe 的 UB 数据交接必须走 `SetFlag/WaitFlag` 事件**（MTE2_V、V_MTE3、MTE2_MTE3、MTE3_S…）**或 BufferID 阻塞释放**（`false`=CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地；MTE2→V 方向 10/10 轮实证，当时档位为 `true`）。`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，**不能保证 UB 数据路径对另一 pipe 可见**（典型症状：host 读回的是上一轮/部分中间结果）。
3. **同 pipe 背靠背 MTE2 DataCopy 不保序**（见 §4）：完成序可颠倒（32B burst 粒度交错、错元素恒等于先发起一轮的数据），同一 buffer 连续两轮拷贝之间必须 `PipeBarrier<PIPE_MTE2>`；BufferID get/rls 任意 mode 均不保序（**当时称作 drain 的 `true` 档**实测无效；现统一为 `false`）。
4. **核尾 GM 写必须排空再退出**：最后一个 MTE3 写后加 `MTE3_S` 事件等待；否则 `aclrtSynchronizeStream` 返回成功而 host 读回丢失/滞后（标量 `SetValue` 直写 GM 同样不可靠，角色探测因此改走 DMA）。
5. **cube 基本链路**：GM→L1 用 `Nd2NzParams`；L1→L0 的 `LoadData2DParamsV2` 对 **B 矩阵必须 `ifTranspose=true`**（nz→zn），否则 Mmad 拿到错误排布、数值差固定常数；`MmadParams{16,16,16,0,false,true}`；FIXP `F322F16` 后由 FIX pipe 发 mode2 通知（发通知前 `FIX_S` 事件排空，否则 AIV 读到全 0）。
6. mode0 屏障：`CrossCoreSetFlag<0, PIPE_MTE3>` 发出前必须确认 GM 写已落盘（`MTE3_S` 事件排空后再 set），否则先到的核对等不到全部 56 份数据（实测只收到 ~20/56 份）。

### 5.3  empirically 确认的可用/不可用原语清单

| 原语 | 结论 |
|---|---|
| `Mutex::Lock/Unlock<pipe>(id)` | 可用；表达占有/释放互斥；**不保序**——同 pipe 背靠背 DMA 完成序可颠倒（见 §4，60/60 vs 0/60 数据），任意 mode（含当时称作 drain 的 `true` 档）均不保序 |
| `SetFlag/WaitFlag<HardEvent::X_Y>` 事件 | **跨 pipe 数据交接的可靠手段**（与 BufferID 阻塞释放二选一） |
| BufferID 阻塞释放（`false`=CANN `ASC_LOCK_BLOCK` 默认；两种模式都等本 pipe 已发射指令落地） | 跨 pipe 交接可用（MTE2→V 方向 10/10 轮实证，当时档位为 `true`）；但不保同 pipe 完成序 |
| `PipeBarrier`（跨 pipe 数据可见性） | **不可靠**，不要用 |
| `PipeBarrier<PIPE_MTE2>`（同 pipe 两轮 DMA 之间） | **必须**：无排空 60/60 轮有错、有排空 0/60（§4） |
| 标量 `SetValue` 写 GM | kernel 退出时可见性无保证，元数据一律走 MTE3 |
| CrossCore mode0/mode2（`__mix__(1,2)` 直调） | 可用；flagId 用 0/1（11-15 为 CANN 保留：AIC=11/AIV=12/AIC_AIV=13/AIV_ALL=14） |

## 6. 遗留与后续

- 检查 2 的累加为出位交替（accA/accB），是规避工具链敏感性时期的稳妥写法；在字节化布局修复后大概率可简化为就地累加，待回归验证。
- 事件（SetFlag/WaitFlag）为标量阻塞式同步，M0 只证正确性；layer kernel 的流水重叠需按"事件粒度"重新设计（同步尽量不挂 PIPE_S 的约束在 351x 上需要用事件而非 PipeBarrier 落地——与 M10 实证"链式同步 wait 后紧接 set 时 wait 必须挂 PIPE_S"互为补充，设计时需一起考虑）。
- 检查 5 结论：同 pipe 重获取**不会等待**前一条 MTE2，完成序可颠倒（32B burst 粒度），两轮之间必须 `PipeBarrier<PIPE_MTE2>`（60/60 vs 0/60 决定性数据）；get/rls 任意 mode 均不保序。M2 调研的"FIFO 保序"预期与实测不符，以本节实证为准。
