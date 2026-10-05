# M91 · `moe_relift/` 的 M91 证据区

本目录是 M84 建立的 MoE 段临时证据区。本文件与 `check_rescale.py` / `check_stream.py` /
`m91_*.log` 属于 mission **M91**（branch `feat/m91-moe-512-device-segment-streaming-rou`）；
M84 的 `README.md` / `check_prep.py` / `m84_*.log` 原样保留、未动。

> **带时点的快照提醒（M95 追加，勿当当前态读）**：本文件 §第 2 轮那张表里 `UB_RT_END` 的读数
> （`m15_moe_resources.h:14` 的 `192544`）是 **M91 当轮的快照**。M95-#4 落地后 `UB_RT_END = 222752`
> （`m15_moe_resources.h:14` 与 §4 已同步刷成该值，`moe_relift/check_stream.py` 的 B2 在 tip 上打印
> 同一读数）；`moe_relift/check_prep.py` 的 `strip_comments` 也已在 M95 r2 换了剥离顺序
> （见 `m95_README.md` §4 的 r2 段）。本文件的其余结论不受影响。

## 本 mission 改了哪些文件

| 文件 | 改动 |
|---|---|
| `m15_layer_loop/m15_moe_resources.h` | ① §6 的 6 项每专家张量定尺 `NUM_EXPERTS * M_MAX` → 紧凑 `TOTAL_MAX`（C）；② §2 新增 2 个 AIV token（22/23）；③ §4 的 router UB 布局重排（`RT_RB` 16→8、新增 `RT_EGRP`、权重 fp32 窗从「E+1 行常驻」改「RT_EGRP 行流式窗 + 共享门 1 行」、新增 `UB_RT_SGB`/`UB_RT_SGWF`）（#3） |
| `m15_layer_loop/lift_moe_segment.py` | 新增「规则 8 = M91-#3 权重流式」及其产物断言（含 2 处删除与 3 段整块替换） |
| `m15_layer_loop/m15_moe_layer.h` | 机械生成物（2081 → 2187 行） |
| `m15_layer_loop/moe_relift/check_rescale.py` | C 的静态判据：文本/形状形态 + 布局算术（当前档 × E=512 假设档） |
| `m15_layer_loop/moe_relift/check_stream.py` | #3 的静态判据：流式形态 + UB 与 E 解耦 + IG 两档放得进 |
| `m15_layer_loop/moe_relift/m91_*.log`、本文件 | 证据（**不入 `m15_layer_loop/evidence/`**，那是别的 mission 的） |

**未动**：`m15_layer_resources.h`、`m15_layer_loop.asc`、`m15_layer_kernel.h`、`m15_attn*`、
`m15_ple*`、`CMakeLists.txt`、`slice_layer_manifest.py`、`weights_manifest.txt`、`README.md`、
`check_moe_ref.py`、`tools/golden/**`、`m17_moe_real/**`、`m13_moe_layer/**`、`probe_host_dma/**`、
`docs/**`。

## C：每专家张量按紧凑 Σt_e 上界重定尺（= M77 §2 第 9 项；**E=4 下数值 no-op**）

6 项：`SZ_AQ / SZ_AS / SZ_GU / SZ_HQ / SZ_HS / SZ_Y`，行数 `NUM_EXPERTS * M_MAX` → `TOTAL_MAX`
（= `M_MAX * TOPK_MAX`）。理由：链上的专家槽**已经是紧凑 Σt_e**（M40 的硬约束，槽起点 = S3 的
`expert_offsets[]` 前缀和、`rowBase = offGm.GetValue(slot) + mt*BASE_M`），这些缓冲只需
`Σt_e ≤ M_MAX*TOPK_MAX` 行。`SZ_LOGITS` **不缩**（router 每行必须吐 E 个 logit）。

`Σt_e ≤ TOTAL_MAX` 的依据：每个 token 恰好贡献 topk 个槽、top-k id 互异 ⇒
`Σ_e t_e = m*topk ≤ M_MAX*TOPK_MAX`。

**数值读数（`check_rescale.py` 就地求值，两档）**：

| 项 | E=4/TOPK_MAX=4（当前档） | E=512/TOPK_MAX=10（假设档） | padded 档（E=512） |
|---|---|---|---|
| `SZ_AQ` | 327,680（= padded） | **819,200** | 41,943,040 |
| `SZ_AS` | 20,480（= padded） | **51,200** | 2,621,440 |
| `SZ_GU` | 655,360（= padded） | **1,638,400** | 83,886,080 |
| `SZ_HQ` | 81,920（= padded） | **204,800** | 10,485,760 |
| `SZ_HS` | 8,192（= padded） | **20,480** | 1,048,576 |
| `SZ_Y` | 1,310,720（= padded） | **3,276,800** | 167,772,160 |
| `SZ_LOGITS` | 1,024 | **131,072** | 131,072 |
| **`WS_BYTES`** | **7,685,472** | **13,891,392** | （315,637,568，M77 §3.2） |

E=512 紧凑档的 6 项与 `WS_BYTES` **与 M77 §3.2 的独立复算逐个相等**（那张表是 M77 用另一套算式
在 wt-77 上算的），这是本项最强的外部对照。

## #3：router 权重流式（形态来源 = `m17_moe_real` 的 `RouterStage`）

**改前**：`PrecastWeights()` 把 `[NUM_EXPERTS+1][HIDDEN]` 的 bf16 权重行**全部**预转 fp32 常驻
UB（E=4 → 51,200 B；E=512 → 5.25 MB，**装不下** —— 这是 #3 要拆的规模墙）。

**改后**（`LoadWRow` / `PrecastWRow` / `PrecastSgateW` / `ComputeBlock` / `PadLogitsRow` /
`SgateRows` / `GemvGroupRow`）：

- x 行块（`RT_RB = 8` 行）一次 MTE2 载入常驻；
- router 权重按「`RT_EGRP = 8` 个专家一组」流式读入 **bf16 ping/pong** → V 预转 fp32 到
  `RT_EGRP` 行窗；组内 8 个专家的点积**共享同一份 x 载入**；
- 组内协议照抄 m17 的 `Gemv`：进第一组前 W0=w[0]、W1=w[1]；槽 `j` 预转 `w[e0+j]`（缓冲 `j%2`），
  预转后立刻把 `w[e0+j+RT_EGRP]` 装进该缓冲（最后一槽即下一组的预取）；
- 共享门权重（1 行）用**独立** UB 槽 `RT_SGB` 预转一次常驻 `RT_SGWF`（抄 m17：复用 ping/pong
  会让 MTE2 写与 V 读落在同一片 UB 上而没有 token）；
- **UB 常驻量与 `NUM_EXPERTS` 解耦**：`UB_RT_*` 16 项在两档（E=4 与 E=512）**读数完全相同**，
  `UB_RT_END = 192,544 B`（改前 193,600）≤ GDN 相位峰值 227,328 ⇒ **融合 UB 峰值未被抬高**
  （`m15_layer_resources.h` 的 `static_assert(UB_PEAK_FUSED <= UB_TOTAL_BYTES)` 原样通过）。

**算术一字未改**（这是「E=4 逐字节零回归」的机制）：同一 `CHUNKS_H` chunk 数、同一
`MulAddDst` 的 chunk 升序累加、同一 `Reduce SUM`、同一 `Interleave` 打包树 —— 8 路树的
lane 0..3 == 改前 4 路树的 `[l0,l1,l2,l3]`（已逐 lane 推过）；共享门从「第 5 个累加器」改成
独立一圈，但累加序与 Reduce 不变 ⇒ 逐位相同；对数行的 lane ≥ E 由 `PadLogitsRow` 写
`RT_NEG_BIG`，与改前 `Select(rowSeg, v4, negInf, mE)` 的产物同常量同范围。

**两处必须注意的坑**（我在写之前先确认过，否则 E=4 会坏）：

1. **写盘掩码不能省**。m17 的 `Dot8Row` 固定按 8 专家一组写 8 个 lane；E=4 时那会把
   lane 4..7 写成（重复行的）点积值、覆盖 pad。故本实现的 `GemvGroupRow` 用
   `ng = min(RT_EGRP, NUM_EXPERTS - e0)` 掩码，E=4 时只写 lane 0..3。
   即使这样，**m17 那份 RouterStage 原样搬到 E=4 仍然会坏**，另有三处（下面「未完成项」）。
2. **`BufAcquire`/`BufRelease` 一律配对使用**。复审核过 `GetBufInternal<pipe,false>` 展开到
   `get_buf(pipe, bufId, mode)`，那个 `mode` 是 ping/pong 选择位、**不是「计数」标志**，从 CANN
   头文件**无法断定它是否阻塞**（本节早先写的「buffer token 计数」这句推理未被证实，已按复审
   意见软化）。但**两种解释下本实现选的都是更保守的那一侧**：越界槽（`e0+j >= NUM_EXPERTS`）
   **不做** m17 `LoadW` 那种提前 `return`（早退与配对两种写法都合法，这里取配对），而是用
   `e % NUM_EXPERTS` 的真实行填充 —— 既保持成对、又不读越权重区；这些槽算出的值因 `ng`
   掩码不落盘。设备验收 + A/B dump 逐字节相同见证了这条路。

## 验收：E=4 零回归（硬要求）

1. **`runs=all` = 2068 判定项 + 290 guard，0 FAIL**；**`runs=chain` = 910 + 52，0 FAIL**
   —— 与改前逐项相同（`m91_stream_verify.log` §3；完整 stdout 见
   `m91_stream_accept_run_all.log` / `m91_stream_accept_chain.log`）。
2. **A/B dump 逐字节对拍**：改前二进制与加完 C+#3 的二进制各
   `M15_DUMP=1 M15_STEPS=3 … all` 一次，各落 **1307** 个文件，两个
   `sha256sum * | sort -k2` 清单 `diff` **rc=0**（逐行相同）。二进制本身 sha 已变
   （`c6ee3ca0…` → `2d42ba86…`），但**可观测输出逐字节不变**（`m91_stream_ab_dump_sha256.log`）。
3. **静态判据**：`check_stream.py` **15 条可 FAIL 判据 0 FAIL**；`check_rescale.py` **30 条 0 FAIL**；
   M84 的 `check_prep.py` 仍 **16 条 0 FAIL**；`lift_moe_segment.py --check` 回绿（两档 locale）。
4. **负向对照 6 条**（变异**只在 `/tmp/m91_nc` 副本**）：
   - `nc1` `SZ_GU` 退回 padded 算式 → `check_rescale` 3 条 FAIL（含 E=512 档 `WS_BYTES`
     13,891,392 → 96,139,072）；`nc2` routed `WS_Y` 基址退回 `slot * M_MAX * …` → 2 条 FAIL；
     `nc3` 把 routed 基址换成集外符号 `WS_ROUTED` → 2 条 FAIL（`m91_neg_controls.log`）。
   - `nc4` 去掉 `ng` 的 `min()`（E=4 会写 8 个 lane）→ A4 FAIL；`nc5` 把 `RT_EGRP` 写成
     `NUM_EXPERTS`（UB 窗重新随 E 增长）→ B1 FAIL（16 项里 15 项被标为随 E 变）；
     `nc6` 多插一处 `Sort32`（模拟“归并树已落地”的假形态）→ A10 FAIL
     （`m91_stream_neg_controls.log`）。

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <worktree>
LC_ALL=C python3 m15_layer_loop/lift_moe_segment.py          # 重生成 m15_moe_layer.h
LC_ALL=C python3 m15_layer_loop/lift_moe_segment.py --check  # 必须回绿
cmake --build m15_layer_loop/build -j4
LC_ALL=C python3 m15_layer_loop/moe_relift/check_prep.py
LC_ALL=C python3 m15_layer_loop/moe_relift/check_rescale.py
LC_ALL=C python3 m15_layer_loop/moe_relift/check_stream.py
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt chain
```

## 显式未完成项（本 mission **没有**交付的东西）

### #4 归并树（16×Sort32 + 4 级二路 MrgSort → 全局 top-64 → Extract 64 → 取前 10）

**未完成。** 卡点（做完 #3 之后能说得更具体）：

1. **对数行/中间缓冲的行距仍是写死的 64 lane**（`grep -c 'r \* 64 \* 4' m15_moe_layer.h` =
   **8 处**：LOG 4 / IDS 2 / WS 2，正文按行读是 `UB_RT_LOG`×4 + `UB_RT_IDS`×2 + `UB_RT_WS`×2）——
   这是改前 m13 的形态，`RouterStage` 的 softmax（`Reduce MAX` 覆盖 32 lane）与
   `Sort32(…, 1)`（1 块 = 32 对）都建立其上。
   E=512 下每行有 512 个 logit、需要 16 个 32 块 + 4 级归并，**必须先把这三个行距与
   `UB_RT_VAL/IDX/PAIR/OV/OI` 的容量一起重排**（M77 §3.3 给的 m17 读数 `RT_END = 222,752`
   是按 m17 的全套布局算的，不能直接套到本布局上）。
   ⇒ **#3 与 #4 动的是同一张 `m15_moe_resources.h` §4 表** ⇒ 分两个 mission 做时**第二个必然
   要再重排一次布局**。**注意这不是依赖关系**：`#3` 只动权重窗 / x 块 / 新增 token，**没有动
   行距**（本次 diff 里 `r * 64 * 4` 的 8 处一字未改）⇒ **#4 不依赖 #3 落地**（复审核过这一点，
   本文件的早先措辞「在 UB 布局层面耦合」已按复审意见改成「同表串行」）。
2. 归并树的 `MrgSort(dst, srcList, MrgSort4Info)`（`validBit=0b0011`）+ 每级
   `elementLengths=[32,32,0,0]` 的写法本仓已验证（`m17_moe_real/m17_moe_layer.asc:1184-1201`，
   2 路；`probe_sync_quirks` 的 A02/A15/A20 档案另证 `MrgSort4` 在 dav-3510 是静默 no-op），
   **照抄即可**；但 `MergeTree()` 的 16→8→4→2→1 乒乓（`RT_MA`/`RT_MB`）与 `Extract(…, 2)` 的
   经典（memory-based）形态**不在 M44 的白名单里**（m17 自己登记为「已有模块先报不改」），
   本段现有实现用的是厂商 VF 内联（`DINTLV_B32` + 掩码 `StoreAlign`）⇒ 落地时要决定用哪一份，
   并给出 docs/05 §6.1 的依据注释。
3. **在 E=4 下 #4 只能验「no-op」**（链上没有 512 专家权重，规模常量归 MoE-A）⇒ 它的
   `Sort32` 块数 = `max(1, ceil(E/32))` 在 E=4 时必须退化成「单块、不归并」才可能逐字节相同，
   而这一步的 bit-identity 论证比 #3 更绕（要证 `Extract` 64 对 vs 32 对在 lane 0..9 上同值、
   归并树多算出的 lane 不落盘）。

### #1/#2 规模常量与权重装载（**归后续 mission MoE-A**，本 mission 未动）

`NUM_EXPERTS 4→512`、`TOPK_MAX 4→10`（`m15_moe_resources.h`）、`MOE_W_STRIDE` / host arena
（`m15_layer_resources.h`、`m15_layer_loop.asc`）、`slice_layer_manifest.py` /
`weights_manifest.txt` 的切片。⇒ 因此本 mission **验不了 512 真实规模**，验收基准是
**E=4 零回归 + 静态自洽（含 E=512 假设档的算术）**，这是塔明确定下的降级形态。

⇒ 合起来：**本段的正确性上界是 `NUM_EXPERTS ≤ 32`**（= 单块 `Sort32` / 单寄存器 `Reduce MAX` /
32 对拆分的**覆盖宽度**：`RT_LANES = 32`，`Sort32(pairT, valT, idxT, 1)` 只排 32 对、
`Reduce<MAX>` 只覆盖 `m32` = 32 lane、拆分只拆 32 对）。分档如实写清（**这是 MoE-A/MoE-B 排期
的输入**，别读成「E=64 还能跑」）：

- **`NUM_EXPERTS ≤ 32`**：top-k 候选集完整 ⇒ 正确。
- **`E = 33..64`**：logits 行（64-lane 行距）**还装得下**，但 **top-k 候选集已被截断**
  （只含专家 0..31）⇒ **结果是错的**，属 #4 的墙。
- **`E ≥ 65`**：连 logits 行都放不下（行距写死 64 lane）⇒ 属 #4 的行距/UB 重排。

#3 拆掉的是「UB 常驻量随 E 线性增长」这条墙（`UB_RT_*` 16 项与 E 解耦）；**剩下的墙在 top-k 侧
（#4）**。`check_stream.py` 的 A10 文本用的是同一口径（「E <= 32 覆盖」）。

## 第 2 轮：复审 r1（`p2-3items` / fix-then-merge）的处置

复审对象 tip = `70cf4f4`（冻结）。三条 p2 全是**注释/文档口径**，无代码行为缺陷。处置如下，
逐条「命令 → 输出」见 `m91_r2_review_fixes.log`：

| 项 | 处置 | tip 上的读数 |
|---|---|---|
| **p2-1** `m15_moe_resources.h:14` 写 `UB_RT_END = 193600`（与同文件 `:45-48` 的 192544 矛盾） | 改成 **192544** | `grep -n 'UB_RT_END = 192544'` → `:14`；`grep -c '193600'` → **无匹配**（rc=1） |
| **p2-2** 抽取规则自述未同步规则 8（脚本 docstring「6 类替换」+ `PROLOGUE`→生成物头部「逐行相同 / 7 类机械替换」） | ① 脚本 docstring：改成「**8 类**替换」并写明「第 8 类是整块替换 ⇒ 对 S2 router「与 m13 逐行相同」不成立」；② `PROLOGUE`：改成「**除下面列出的 8 类替换之外**，算件代码逐行相同（第 1–7 类是重命名式/机械替换；**第 8 类是整块替换**）」，清单加第 8 条；③ **重新生成**生成物（2187 → 2197 行） | `grep -n '除下面列出的 8 类替换之外'` → 生成物 `:7`；`grep -c '7 类机械替换'` → 无匹配（rc=1）；`grep -c '6 类替换'` → 无匹配（rc=1） |
| **p2-3** `:152-153` 头条写「`NUM_EXPERTS ≤ 64` 下能跑」会误导排期 | 改成「**正确性上界 `NUM_EXPERTS ≤ 32`**」并分三档写清（≤32 正确 / **33..64 top-k 候选集已截断 ⇒ 错** / ≥65 连 logits 行都放不下）；资源表 §4 的 `UB_RT_LOG` 行距注与 `SZ_LOGITS` 注同步改成 32 口径 | `grep -n '正确性上界是'` → README `:159`、`m15_moe_resources.h:216` |

**同时按复审意见改的两处措辞**：

- **drop reason 的「耦合」讲重了** → 本文件 §#4 的卡点第 1 条改为「**#3 与 #4 动的是同一张 §4
  表** ⇒ 分两个 mission 做时第二个必然要再重排一次布局；**这不是依赖关系**：`#3` 只动权重窗 /
  x 块 / 新增 token，**没动行距**（`r * 64 * 4` 的 8 处本次 diff 一字未改）⇒ **#4 不依赖 #3**」。
- **「`r * 64 * 4` 共 6 处」实测是 8 处** → 已改成「`grep -c 'r \* 64 \* 4'` = **8 处**（LOG 4 /
  IDS 2 / WS 2）」，并附上命令。
- **「`BufAcquire` 是 buffer token 计数、不是旗语」这句推理复审未证实**（`GetBufInternal<pipe,false>`
  → `get_buf(pipe, bufId, mode)`，`mode` 是 ping/pong 选择位）⇒ 生成物与本节都软化为
  「**按配对使用**（与 m17 的早退形态不同，两种写法都合法，这里取更保守的那一侧）」，免得 #4
  接手人照着旧句做同步推理。

**第 2 轮的证据（`m91_r2_review_fixes.log`）**：

- **二进制 sha 与复审过的 tip 逐字节相同**（`2d42ba86…`）⇒ 三处改动**零 codegen 影响**；
- ⇒ **A/B dump 不重做**，理由：dump 比的是机器码的行为，而机器码逐字节相同；该二进制上一轮的
  dump 已与 base 逐字节对拍过（1307 文件、diff rc=0）；
- **但设备验收重新见证了一次**（生成物/资源表变了）：`runs=all` = **2068 + 290, 0 FAIL**、
  `runs=chain` = **910 + 52, 0 FAIL**（并发背景：NPU 0 上有另一个 `m15_layer_loop` 进程）；
- 三条静态判据两档 locale：`check_prep` 16/0、`check_rescale` 30/0、`check_stream` **15/0**；
  `lift --check` 两档 rc=0；
- 6 条负向对照在**重生成后的产物**上重跑，计数不变（3/2/2/1/1/1）。

> 本轮还修了 `check_stream.py` 的一条判据**实现在先、被自己的注释咬**的坑：A2 原先在**全文**
> 上找 `PrecastWeights` 等旧串，而生成物头部现在**合法地提到**该名（讲第 8 类替换时点名旧形态）
> ⇒ A2 改为在**剥掉注释后的代码**上判（复用 `check_prep.strip_comments`）。A2 文本里同时加了
> 提示：「落地 #4 时本判据与 A10 都须同步改，否则合法落地会报 FAIL」（复审的观察 G）。

### 复审点名、**本 mission 不修**（已按复审判定留给后续）

- `SoftmaxTopkRow` 的 `maskAllI = CreateMask<int32_t, ALL>()`（int32 ALL mask ⇒ 每行 512 B store
  对 256 B 行距）是 **m13 旧形态、非本轮引入**（复审核过 `m13_moe_layer.asc:1006` 与 base
  `m15_moe_layer.h:983` 一字不差）；但本 mission 的 `RT_RB 16→8` 使**最小触发 m 由 16 降到 8**
  —— 复审已 `TowerFinding` 登记，交后续 mission（动它要重排行距，与 #4 同批）。
- `UB_IG_CNT` 的陈旧注释（M84 已报）、`m15_layer_loop/README.md:151/1281` 的 `193600`（scope 外）
  —— 一并留给 docs/清账 mission。
