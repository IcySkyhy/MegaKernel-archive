# probe_sync_quirks：3510 两条待复核 quirk 的归档式复现（M23）

本目录是 docs/05 §6.3 两个"原始探针与完整报错未归档"条目的**可重跑最小复现 + 原始证据归档**：

| 条目 | 原记录 | 本目录的结论（一句话） |
|---|---|---|
| **#19** | "4 路 MrgSort：src3/src4 胜出元素的 index 写成 UB 默认值"，最终退化为 2 路归并树 | 按原记录的参数（4 路 `validBit=0b1111`、`elementLengths=[32,32,32,32]`、4 个 Sort32 产出块、`repeatTimes=1`、`exhausted=false`）**复现不出该现象，实测完全正确**；"dst 保持 UB 默认值"这种形态**只能由 `MrgSort4` API 产生——它在 3510 设备构建里是静默 no-op**（§4） |
| **#23** | "向量循环内动态下标标量加载 → backend 报 `Unsupported Inst must be hoisted`"，规避是模板展开 | 报错**已拿到原文**；触发条件是 **VF 内标量读 GM**（`__gm__` 裸指针或 `GlobalTensor::GetValue`），**与 idx 类型无关**（无 cast 的 form 8 六类型同错；int64_t/uint64_t 的 UB 动态下标则合法）（§5） |

原始证据（UB 逐元素 dump、逐变体编译/运行日志、报错原文）全部落在 `evidence/`，见 §6 索引。

---

## 1. 目录与构建

```
probe_sync_quirks/
├── CMakeLists.txt                 # 独立 CMake 工程（find_package(ASC) + --npu-arch=dav-3510，仿 m2）
├── probe_a_mrgsort4.asc           # 探针 A：MrgSort 26 变体矩阵（含 #19 原参数）
├── probe_a2_mrgsort4_api.asc      # 探针 A2：MrgSort4 API vs MrgSort(MrgSort4Info) API
├── probe_b_vec_idx.asc            # 探针 B：#23 动态下标标量加载（6 idx 类型 × 10 种写法 form 0..9，宏选择）
├── run_probes.sh                  # 一条命令重跑全部探针并重生成 evidence/
├── tools/decode_mrgsort_dump.py   # 独立解析 probe A 的 UB dump -> 逐元素 value/index 表
└── evidence/                      # 归档证据（见 §6 索引）
```

```bash
cd probe_sync_quirks
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8            # 默认 target 全部编译通过（不含预期编译失败的 form 5/8 与 u32loop 靶子）
./build/probe_a_mrgsort4                       # 只打印判读摘要
./build/probe_a_mrgsort4 <dumpdir>             # 额外落盘每变体 6144B UB 快照 .bin
./build/probe_a2_mrgsort4_api <dumpdir>        # 判读含"与真值全序逐项一致"的前缀比对（matchRun）
./build/probe_b_idx_s32_f0_raw <dumpdir> 50    # 变体名见 CMakeLists / §5；参数=重复次数
./build/probe_b_idx_s32_f9_misalign_demo       # 归档靶子：预期 aclError=507035（见 §5 硬约束②）
bash run_probes.sh 50                          # 全量重跑 + 证据重生成（约 8 分钟）
```

`MrgSort4` API（probe_a2）在 3510 上是 `[[deprecated]]` 路径，构建时会有一行告警
（`Vmrgsort4Cal<float> is deprecated: NOTICE: MrgSort4 ... is an unsupported API on current device`）——
这行告警本身就是证据，默认构建不把它当错误。

## 2. 数据契约（怎么读 dump）

* **1 对（region proposal）= 8 字节 = `{fp32 value, uint32 index}`**（小端：word0=value、word1=index）。
  `elementLengths` 的单位就是"对"（§3 的 A10/A11/A17/A25 有实验判据）。
* 输入生成：块 b ∈ [0,4)、组内序 j ∈ [0,32) → `value = 1000 - 4j - b`、`index = 1000b + j`。
  块内 value 严格递减（= 已排序队列），4 块值域**交错**（正确的 4 路归并 = k 外层 b 内层的交错全序），
  value 与 index 双射且值域不重叠 → dump 里任何"value 对不上 index"或"index 是残留值"都能直接定位来源。
* UB 布局（字节偏移，全 32B 对齐）：`UB_IN=0`(4 块×256B) / `UB_DST=1024`(2048B，输出 1024B + 1KB 越界观察窗) /
  `UB_DST2=3072` / `UB_TMP=4096`(Sort32 输出) / `UB_S32I=5120`(Sort32 输入) / `UB_END=6144`。
  每个变体把 `[0,6144)` 整段原样 dump 回 GM ⇒ **"没写的地方"也留证**。
* dst 预置：变体默认用 sentinel `(0xDEADBEEF, 0xBEEFDEAD)` 预置（`0xDEADBEEF` 是普通负数 float，
  不会引入 NaN）；`preset=zero`(A03) / `preset=none`(A04) 两个变体专门对照。
* 同步：只用 BufferID（`BUF_IN`=MTE2→V、`BUF_OUT`=V→MTE3，均阻塞释放 `false`=CANN `ASC_LOCK_BLOCK` 默认）；`LocalMemBar` 只在
  `__VEC_SCOPE__` 内下发（M15 实证 VF 外会 507035）。

> `evidence/decode_A_summary.txt` 与 `evidence/dumps_txt/variant_NN_*.txt`（**value / index 分列**的逐对表格）
> 由 `tools/decode_mrgsort_dump.py` 独立复算，kernel host 侧另有自己的判读（`evidence/logs/run_probe_a_*.log`），
> 两者一致。

---

## 3. 结论：探针 A —— #19 的 4 路 MrgSort（`probe_a_mrgsort4.asc`）

变体矩阵在 `probe_a_mrgsort4.asc:253 RunVariant()`（每个 case 一行 `DoMerge(MrgSpec{...})`），
判读表在 `probe_a_mrgsort4.asc:421 kVariants[]`，期望值复算在 `:503 ExpectedPairs()`。
两次运行逐变体输出完全一致（确定性已核）。

| 变体（case → 行号） | 参数 | 观测（dump 逐元素判读） | 结论 |
|---|---|---|---|
| A00 `:266` | 2 路 `vb=0b0011` len[32,32] | 64 对全序 PASS | 基线 |
| A01 `:269` | 2 路归并树（m7 现行实现） | 128 对全序 PASS | 基线（现行实现正确） |
| **A02 `:282`** | **4 路 `vb=0b1111` len[32,32,32,32] src=b0..b3 rep=1 exh=false（#19 原参数）** | **128 对全序 PASS，逐项一致** | **#19 记录的参数在 3510 上完全正确** |
| A03 `:285` | A02 + dst 全 0 预置 | 128 对 PASS，后 128 对仍为 0 | dst 预置不影响 |
| A04 `:288` | A02 + dst 不预置 | 128 对 PASS | 同上 |
| A05 `:291` | A02 但 dst 换 UB_DST2 | 128 对 PASS | 换地址/对齐不影响 |
| A06 `:294` | 3 路 `vb=0b0111` len[32,32,32,0] | 96 对 PASS | 3 路也对 |
| **A07 `:297`** | `vb=0b1111` 但 len=[32,32,**0,0**]、src3=src4=src1 占位 | **0 对写出（dst 整段保持 sentinel）** | **vb 与 lens 不自洽 ⇒ 静默不写** |
| A08 `:300` | `vb=0b1111` len 全 32 但 src3=src4=src1 | 128 对（块1 重复 3 份）PASS | "占位"本身没问题，问题在**零长 + vb 开位** |
| A09 `:303` | src 地址乱序 (b0,b2,b1,b3) | 128 对 PASS（与 A02 同） | 4 个 src 地址各自独立，不需要连续 |
| A10/A11/A17/A25 `:306,309,345,377` | len=16/16/4/8 各变体 | 32/64/16/32 对 PASS | **elementLengths 单位 = 对（8B）**，输出对数 = Σlen |
| A12 `:313` | `rep=2` len[32,32,32,32] | 只写 128 对（= rep=1 结果） | rep 与 len 的规模关系见 A24/A25 |
| **A13 `:315`** | A02 且 `ifExhaustedSuspension=true` | **只写 125/128 对**；`VMS4_SR=list1..4 = 32/31/31/31` | **任一队列耗尽即停**（要全序必须 false） |
| A14 `:328` | 只跑 `Sort32`（4 repeat） | 4 组各 32 对、256B 连续、组内 value 降序、`{value,index}` 逐项 PASS | **Sort32 输出布局/方向真值** |
| **A15 `:334`** | 4 路 src 直接取 Sort32 输出（#19 原场景） | 128 对 PASS | 与 A02 等价 |
| A16 `:342` | 4 个 src 全指向 b0 | 128 对（4 份重复）PASS | 4 路**都真的被读了** |
| A18/A19 `:348,352` | dst = src1 / dst = src3（原地归并，写覆盖输入） | 128 对、与规范 4 路归并**逐项一致** | dst 与 src 重叠不构成触发条件 |
| A20 `:356` | A15 去掉 Sort32→MrgSort 之间的 VEC 障碍 | 128 对 PASS | 该障碍不是正确性必需 |
| **A21 `:362`** | `rep=0` | **0 对写出** | **repeatTimes=0 = 静默失效**（易踩） |
| A22 `:365` | `vb=0b0011` 但 len 4 项全 32 | 64 对（只并前 2 路，其余 len 被忽略） | vb 是权威开关 |
| A23 `:369` | len 全 0 | 0 对写出 | 同上 |
| A24/A25 `:372,377` | len=[8,8,8,8] + rep=2 / rep=1 | rep=1：32 对（交错全序，PASS）；rep=2：64 对 = **两块各自 32 对原序拼接** | rep>1 的语义与"重复写同一 dst"不同 → 见【仍存疑】 |

### 【已确证】

1. **#19 记录的 4 路参数在 dav-3510 上完全正确**（A02/A15/A20 三个独立变体、逐元素 dump 归档、
   两次运行一致）。⇒ "4 路 MrgSort 丢 src3/src4 的 index" **不能归因于指令本身**。
2. `elementLengths` 单位 = **8 字节对**；输出对数 = Σlen[i]（A10/A11/A17/A25）。
3. **`validBit` 必须与实际非空队列数一致**：`vb=0b1111` + 任一 len=0 ⇒ **整个 merge 静默不写**
   （A07/A23）；`vb=0b0011` 时 len[2..3] 直接被忽略（A22，只出 2 路的 64 对）。
4. **`repeatTimes=0` ⇒ 静默不写**（A21）；rep=1 正确（全部主变体）。
5. `ifExhaustedSuspension=true` ⇒ **第一个队列耗尽就停**（A13 出 125 = 32+31+31+31，
   `VMS4_SR` 实测 32/31/31/31）⇒ 要全序必须 `false`（也是 API 默认）。
6. 4 个 src **地址各自独立**、允许不连续、允许与 dst 重叠、允许互为占位（A08/A09/A18/A19/A16）。
7. Sort32 输出：4 组**连续**排布、每组 32 对 = 256B、组内按 value 降序、对 = `{fp32 value, u32 index}`（A14）。
8. dst 预置/不预置、dst 位置（UB_DST/UB_DST2）都不影响结果（A02/A03/A04/A05）。

### 【仍存疑】

1. **#19 当年"只有 src3/src4 胜出元素的 index 变成 UB 默认值"的确切成因**。本探针复现不出"半写"形态；
   与"dst 保持 UB 默认值"最吻合的机制是 §4 的 `MrgSort4` API 静默 no-op（整段不写，看到的就是残留内容）。
   原始探针/原始代码未归档 ⇒ 无法定论。
2. **`repeatTimes>1` 的精确语义**：A24（rep=2,len=8）输出 64 对 = 两个输入块各自 32 对的原序拼接，
   而 A25（rep=1,len=8）输出 32 对交错全序；A12（rep=2,len=32）只写 128 对（= rep=1 结果）。
   说明 rep>1 进入了另一种（多组/流式）行为。CANN 自带 TopK 确实用 rep>1
   （`asc/impl/adv_api/detail/sort/topk/topk_v220_impl.h:64-86`，src 取自同一 buffer 的不同 offset），
   所以它另有合法用法；但我们（top-10）只需要 **rep=1**，m7 现行实现也是 rep=1。
3. A13 只测了"四路等长"一种情形；"某路更早耗尽"时的截断点数未测。

---

## 4. 结论：探针 A2 —— `MrgSort4` API 在 3510 上是静默 no-op（`probe_a2_mrgsort4_api.asc`）

| 变体 | 调用 | 观测（探针自身判读，含与真值全序的逐项前缀比对） |
|---|---|---|
| A2_00 | `MrgSort4(dst, srcs, {vb=15, len=32×4})` = `probe_a2_mrgsort4_api.asc:124` | **0 对写出、dst 128 对全 sentinel**（PASS，`evidence/logs/run_probe_a2_mrgsort4_api.log`） |
| A2_01 | `MrgSort(dst, srcs, 同参数)` = `:139` | 写出 128 对，`前缀序匹配=128/128` PASS（与真值 4 路归并逐项一致） |
| A2_02 | `MrgSort4` + `vb=0b0011` len[32,32] | 0 对写出（PASS） |
| A2_03 | `MrgSort` + `vb=0b0011` | 写出 64 对，`前缀序匹配=64/64` PASS（与真值 2 路归并逐项一致） |

（A2 的 host 判读现在与 probe A 同款：既统计 sentinel/合法/非法对数，也算 `matchRun` 前缀序匹配，
所以"全序正确"这句话由探针自身给出，不依赖外部解码。）

### 【已确证】`MrgSort4` 在 3510 设备构建里是 no-op

源码链（`$ASCEND_HOME=/usr/local/Ascend/cann-9.1.0`，`ascend-toolkit/latest` 即它）：

| 位置 | 内容 |
|---|---|
| `asc/impl/basic_api/kernel_operator_proposal_intf_impl.h:80-112`（函数体 :81-112，注释 :68-79） | `MrgSort4(dst, src, params)` → `Vmrgsort4Cal(dstPtr, addrArray, config)`（**3 参**）；config 打包：12bit lens 落在 config[11:8]/[23:20]/[35:32]/[47:44]，exhausted[59]，validBit[60]（:98-105） |
| `asc/impl/basic_api/dav_3510/kernel_operator_proposal_impl.h:41-47`（`[[deprecated]]` :42-43、函数体 :46） | 3 参 `Vmrgsort4Cal` 被 `[[deprecated("...MrgSort4 ... is an unsupported API on current device")]]` 标记，**函数体只有 `ASCENDC_REPORT_NOT_SUPPORT(false, "MrgSort4");`** |
| `asc/impl/basic_api/kernel_log.h:87-93` | 该宏在 `ASCENDC_CPU_DEBUG` 构建里 = `KERNEL_LOG + raise(SIGABRT)` |
| `asc/impl/basic_api/kernel_log.h:277` | 在**设备构建（非 CPU_DEBUG）**里该宏展开为**空** ⇒ 函数体为空、不下发任何指令、不报任何错 |
| `asc/impl/basic_api/dav_3510/kernel_operator_proposal_impl.h:68-75`（`vmrgsort4` 调用在 :73） | 4 参 `Vmrgsort4Cal` 才是真指令（`vmrgsort4(dst, addrArray, src1, config)`） |
| `asc/impl/basic_api/kernel_operator_proposal_intf_impl.h:153-198` | `MrgSort(dst, src, MrgSort4Info)` → 4 参版本；config 打包：repeatTimes[7:0]、**validBit[11:8]**、exhausted[12]、4×16bit lens 写进 src1 寄存器（:174-183） |

⇒ **两个 API 不仅实现路径不同，config 打包也不同**（validBit/exhausted/lens 的位域完全不同），
在支持 `MrgSort4` 的旧平台上也**不可互换**。判定方法：`MrgSort4` 无返回值，唯一可靠的自查是
"dst 是否被写"——本次 dump 给出的是**整段未写**。

### 【仍存疑】

* #19 当次是否调用的 `MrgSort4` API（原始代码未归档）。但"dst 里是 UB 默认值"的形态与
  no-op 完全吻合：no-op 后 dst 里看到的就是**上一阶段/上一次迭代的残留内容**，
  这也能解释"部分元素看起来对、部分像是默认值"的主观描述。
  （注：main 的 docs/05:147 已把 #19 写成"已定位 → 我方调错 API"，比现有证据强——
  缺"当年那次确实调用了 `MrgSort4`"这一环；本目录按证据只给到"最可能成因"。）
* 若换成 `ASCENDC_CPU_DEBUG` 构建，同一调用会 `raise(SIGABRT)`（宏定义如此），本目录未做该构建的实测。

---

## 5. 结论：探针 B —— #23 动态下标标量加载（`probe_b_vec_idx.asc`）

写法维度（10 种，宏 `PROBE_B_FORM`）：
`0 raw`(VF 内运行时界循环 + 裸指针 `wP[j]`，`IdxT` **原类型**，`:157-169`)、
`1 hoistub`(标量加载搬到 VF 外 + UB 暂存, `:170-190`)、`2 ctbound`(VF 内编译期界, `:191-203`)、
`3 constidx`(编译期常量下标对照, `:204-217`)、
`4 ubget`(VF 内 `LocalTensor::GetValue`, `:226-238`)、`5 gmget`(**VF 内 `GlobalTensor::GetValue`**, `:239-252`)、
`6 acc`(标量累加器 + UB `GetValue`, `:253-268`)、`8 gmraw`(**VF 内 `__gm__` 裸指针 `gP[j]`，
`IdxT` 原类型、无任何 cast**，其余与 form 0 同构, `:273-294`)、`9 misalign`(4B 步长 `LoadAlign` 归档靶子, `:296-319`)、
`7 gmhoist`(form 5 的读搬到 VF 外, `:321-344`)。
另有一个独立编译宏 `PROBE_B_LOOPVAR`（默认 `uint16_t`）只改 VF 循环归纳变量类型，用于归档硬约束①。

kernel 形参 `base`/`n` 都来自 host（运行时），保证 `idx` 非编译期常量。
**关于 idx 类型维度的举证口径（修订）**：`GlobalTensor::GetValue` / `LocalTensor::GetValue`
的接口签名只吃 `uint32_t` 偏移，所以 form 4/5/6/7 里的下标是**强转过的**
（`static_cast<uint32_t>(j)`），它们的"6 种 idx 类型报同一句错"实质是同一段程序，**不作为类型维度的证据**，
只能证明"报错点固定在 `GetValue` 内部动态寻址"。
类型维度由**不带 cast** 的 form 0（UB 裸指针）与 form 8（GM 裸指针）承载——两者都用 `IdxT` 原类型做下标记
（含 `int64_t`/`uint64_t`）。

### 编译矩阵（`evidence/logs/compile_matrix.txt`）

* `form 0/1/2/3/4/6/7/9` **全部编译通过**，含 `int64_t` / `uint64_t` 下标（form 1/7 另含 `uint32_t`）。
* `form 5`（VF 内 `GlobalTensor::GetValue`）**6/6 全部失败**，错误原文逐字相同（下方文本块）；
  该 6 份是"同一段程序 × 6 个 idx 类型宏"，只作报错点证据。
* `form 8`（VF 内 `__gm__` 裸指针、`IdxT` **原类型无 cast**）**6/6 全部失败**，错误同一句
  ——**这一组才是"与 idx 类型无关"的独立证据**（六份分别是六段真正不同的下标类型程序）。
* `probe_b_idx_s32_f0_u32loop`（form 0 + `PROBE_B_LOOPVAR=uint32_t`）**编译失败**，
  错误原文见 `evidence/logs/build_probe_b_idx_s32_f0_u32loop.log`（硬约束① 的归档）。

```
# form 5（GetValue 内部动态寻址）：evidence/logs/build_probe_b_idx_<idx>_f5_gmget.log
fatal error: error in backend: Unsupported Inst must be hoisted.
/usr/local/Ascend/cann-9.1.0/asc/include/basic_api/../../impl/basic_api/dav_3510/../kernel_tensor_impl.h:1184:20 in GetValue
probe_b_vec_idx.asc:248:29 in _ZN12_GLOBAL__N_111VecIdxProbe6RunAltEv.vector.extracted
probe_b_vec_idx.asc:241:9 in RunAlt
probe_b_vec_idx.asc:148:36 in Run
probe_b_vec_idx.asc:134:9 in Process
probe_b_vec_idx.asc:362:8 in probe_b_vec_idx_kernel

bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
2026-07-30T20:53:21+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)

# form 8（无 cast 的 __gm__ 裸指针；报错栈直接落在源行 gP[j] 上）：logs/build_probe_b_idx_<idx>_f8_gmraw.log
fatal error: error in backend: Unsupported Inst must be hoisted.
probe_b_vec_idx.asc:288:34 in _ZN12_GLOBAL__N_111VecIdxProbe6RunAltEv.vector.extracted
probe_b_vec_idx.asc:282:9 in RunAlt
probe_b_vec_idx.asc:148:36 in Run
probe_b_vec_idx.asc:134:9 in Process
probe_b_vec_idx.asc:362:8 in probe_b_vec_idx_kernel

bisheng: error: clang frontend command failed with exit code 70 (use -v to see invocation)
```
（六份 form 5 日志之间只差 gmake 规则行号；form 8 的六份同理。全量 stdout+stderr 见对应 `evidence/logs/build_*.log`。）

### 运行矩阵（`evidence/logs/run_matrix.txt`，每变体 50 次 launch × 8 检查点 = 400 点）

| 写法 | s16 | s32 | s64 | u16 | u32 | u64 |
|---|---|---|---|---|---|---|
| f0 裸指针动态下标（VF 内循环） | PASS | PASS | PASS | PASS | PASS | PASS |
| f1 标量加载 hoist 到 VF 外（UB 暂存） | PASS | PASS | 8/400 错 | — | 8/400 错 | — |
| f2 VF 内编译期界 | — | PASS | PASS | — | — | — |
| f3 编译期常量下标（对照） | — | PASS | — | — | — | — |
| f4 VF 内 `LocalTensor::GetValue` | PASS | PASS | PASS | — | — | — |
| f5 VF 内 `GlobalTensor::GetValue` | 编译失败 | 编译失败 | 编译失败 | 编译失败 | 编译失败 | 编译失败 |
| f6 标量累加器 + UB `GetValue` | — | 351/400 错 | 350/400 错 | — | — | — |
| f7 form5 搬到 VF 外 | — | PASS | PASS | — | — | — |
| f8 VF 内 `__gm__` 裸指针（IdxT 原类型） | 编译失败 | 编译失败 | 编译失败 | 编译失败 | 编译失败 | 编译失败 |
| f9 4B 步长 `LoadAlign`（归档靶子） | — | 运行 `aclError=507035` | — | — | — | — |

> f1 是**间歇性**的：本表是本次归档批次（`logs/run_matrix.txt`）；前两批次里 s16 8/400、s32 12/400、
> s64 8/400（另一批次全 PASS）也各自出现过 1 次"整次 launch 全 8 点错"——**四种 f1 类型在三批里都出现过**。
> f6 则三批都是 ~350/400 恒定错。这正是 #22 描述的"标量↔VF 通路易错"。表中 `—` = 该组合未做 target。

（"8/400 错" = 50 次 launch 里有 1 次整块 8 点全错；f1 的错点是整 launch 级的，与 #22 的"间歇丢写"形态一致。）

### 【已确证】

### 【已确证】

1. **`Unsupported Inst must be hoisted` 的触发条件是"VF 内标量读 GM"**：
   * 由**不带 cast** 的 form 8（`__gm__` 裸指针 + `IdxT` 原类型）给出——s16/s32/s64/u16/u32/u64
     **六段不同程序**报**同一句**错误（`evidence/logs/build_probe_b_idx_<idx>_f8_gmraw.log`）；
   * form 5（`GlobalTensor::GetValue`）是同一触发点的 API 形态，报错栈直指
     `kernel_tensor_impl.h:1184 GetValue`（其 6 份因 `GetValue` 只吃 `uint32_t` 而**不用于**类型论证）。
   * 判别因子是**地址空间**而非类型或"结果进向量指令"：form 8（GM 裸指针）与 form 0（UB 裸指针）
     的循环形状、下标算术（同为 `IdxT` 原类型）、向量消费方式完全相同，只差地址空间 → 前者失败、后者通过。
2. **动态下标标量加载本身在 VF 内是合法的**（用户判断正确）：`wP[j]`（裸 `__ubuf__` 指针，`IdxT` 原类型，
   含 64 位下标）50 次 × 8 点全对；`LocalTensor::GetValue` 同（其接口会把 idx 转成 `uint32_t`）。
   ⇒ 用户"64 位肯定不支持"的说法在本构造下**不成立**（`int64_t/uint64_t` 下标在 form 0/2 均通过；
   64 位只在"VF 内标量读 GM"时同样报错，而那是**与类型无关**的）。
3. **正确写法（hoist 到循环外）**：把标量读 GM 搬到 VF 外再落 UB，VF 内只按静态地址取数（f7，50/50 PASS）；
   m8 现行的"模板展开 + 编译期常量下标"（f3）也在合法集合内。
4. **附带硬约束（两条都有归档 log）**：
   * **VF 循环的归纳变量必须是 `uint16_t`**：用 `uint32_t` 得到
     `error: Induction variable must have a type uint16_t`——归档于
     `evidence/logs/build_probe_b_idx_s32_f0_u32loop.log`（靶子 `probe_b_idx_s32_f0_u32loop` = form 0 + `PROBE_B_LOOPVAR=uint32_t`）。
   * **VF 内 `LoadAlign` 的地址必须 32B 对齐**：4B 步长（想只取每块首元素）直接
     `aclError=507035`（vector core exception）——归档于
     `evidence/logs/run_probe_b_idx_s32_f9_misalign_demo.log`（靶子 `probe_b_idx_s32_f9_misalign_demo` = form 9），
     与 M9 记录的"UB 偏移非 32B 对齐异常"同一族现象，这次是在**向量 load** 侧。

### 顺带：form 1 / form 6 复现了 #22 的"标量→向量通路易错"

* f1（标量读 UB → 标量写 UB 整块 → VF 内 `LoadAlign`）**间歇性**出现"整次 launch 全 8 点错"：
  三批归档里 s16（8/400）、s32（12/400）、s64（8/400）、u32（8/400）各出现过，也有整批全 PASS；
  错值如 `0.883777`（把 UB 里别的 float 当成了结果）——量级与 docs/05 §6.3 #22 记的 "~1/30" 同阶，
  说明 `LocalMemBar` 覆盖不到 scalar 的 load/store 这一判断成立。
* f6（标量循环累加 UB `GetValue`，再 `Duplicate(acc)` 进向量）**恒定错**
  （三批均 ~350/400 点，观测到的是**某个中间值**：i=1 处得到 `301` 而期望 `2428`）——即
  "标量结果作为向量标量操作数"这条通路不可用。**这两条是 #22 的证据，不是 #23 的**，
  故不改变 §5 的结论。

### 【仍存疑】

1. m8 当年的**最小复现代码**已不可考（git 里第一版就是"模板展开"的改好版本），
   因此无法逐字确认它与 form 5/form 8 同源；能确证的是 form 5 报出的文本与原记录逐字一致，
   且触发点必须是"VF 内标量读 GM"（form 8 已用无 cast 的裸指针证明与类型无关）。
2. **"结果是否必须被向量指令消费"这一因子仍未分离**：form 8 与 form 0 都让结果进向量指令，
   只差地址空间，所以"触发条件是 GM 标量读"已确证；但"VF 内标量读 GM 且结果**只**进标量路径
   （不进任何向量操作数）是否也报错"未做实验（写一个不被 DCE 掉的纯标量消费形态需要额外设计）。
3. f6 的错值模式（恒错 vs 偶发）只测了 50 次；#22 要求的"能覆盖 scalar 的正确同步原语"仍未定。

---

## 6. 证据索引（`evidence/`）

| 路径 | 内容 |
|---|---|
| `commands.txt` | 复现用命令、CANN/编译器版本、日期、NPU、reps |
| `logs/compile_matrix.txt` | 全部 target（含 A/A2/VF 归纳变量靶子）的编译通过/失败矩阵 |
| `logs/build_<target>.log` | 每个 target 的**完整构建输出**（含 form 5 报错原文 6 份、form 8 报错原文 6 份、`*_u32loop` 归纳变量报错 1 份） |
| `logs/run_matrix.txt` | probe B 每个可运行变体的运行结论（点数/错点数） |
| `logs/run_<target>.log` | 每个可执行文件的完整运行输出（probe A / A2 的逐变体判读摘要在内） |
| `logs/decode_A.log` | 解码脚本的标准输出（= decode_A_summary.txt） |
| `decode_A_summary.txt` | probe A 26 个变体的逐变体判读（写出对数/合法性/sentinel/与期望比对） |
| `dumps/variant_NN_<name>.bin` | probe A 每个变体的 **6144B UB 快照**（原始字节） |
| `dumps/input_vfirst.bin` | probe A 的 host 输入缓冲（与 UB 布局同构，可对照复算） |
| `dumps/a2_NN_<name>.bin` | probe A2 每个变体的 2048B UB 快照 |
| `dumps/b_<target>.bin` | probe B 每个变体的 2048B 输出（8×64 lane，检查 lane 0） |
| `dumps_txt/variant_NN_<name>.txt` | **逐对两列（value / index）+ 来源标注 + 期望/是否相符**的文本表（26 份） |

两条硬约束的归档落点（§5【已确证】4 引用）：
`logs/build_probe_b_idx_s32_f0_u32loop.log`（归纳变量必须 `uint16_t`）、
`logs/run_probe_b_idx_s32_f9_misalign_demo.log`（非 32B 对齐 `LoadAlign` → `aclError=507035`）。

`dumps_txt/variant_02_A02_4way_v15.txt` 是 #19 原参数的逐元素对照表；
`dumps_txt/variant_15_A15_4way_v15_sort32src.txt` 是"src 直接取 Sort32 输出"的对照表；
`dumps_txt/variant_14_A14_sort32_layout.txt` 里有 Sort32 输出布局的真值。

## 7. 未解问题 / 后续（汇总）

1. **#19 的真正成因**仍缺"原始代码"这一环；本目录给出的是：原参数正确、`MrgSort4` API 是 no-op、
   以及 5 种"静默不写/截断"的触发条件（vb 与 lens 不自洽、len 全 0、rep=0、exhausted=true…）。
   建议后续在 docs/05 §6.3 #19 里改成"参数本身正确 + 检查是否误用了 `MrgSort4` API"。
2. `repeatTimes>1` 的语义未定（A24/A25 的现象已归档）。
3. `ifExhaustedSuspension=true` 只测了四路等长；不等长时的截断规则待测。
4. #22 需要的"能覆盖 scalar↔VF 的正确同步原语"仍未找到（本目录的 f1/f6 提供了可重跑的复现器与量化率）。
5. probe B 的"VF 内标量读 GM"已确证（form 8 无 cast 六类型同错、form 8 vs form 0 只差地址空间），
   但"结果是否必须被向量指令消费"未分离（见 §5【仍存疑】2）。
