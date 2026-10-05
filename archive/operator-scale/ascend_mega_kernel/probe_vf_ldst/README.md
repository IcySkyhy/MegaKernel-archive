# probe_vf_ldst：VF load/store 分布模式（`LoadDist`/`StoreDist`）语义最小探针（M45）

本目录把 M36（hc kernel，`m20_hyperconn/`）报的两条「与 VF load/store 分布模式相关的平台行为」
**先隔离、再定性**，产出可重跑的最小复现 + 原始证据归档。
触发理由见 mission：本仓已有**四条**同类结论**全部**以「我们选错用法」收场，所以这两条在入库前必须先取证、先隔离。

## 0. 结论一览

| 现象 | 记录来源 | 本目录的结论（一句话） |
|---|---|---|
| **(a)** `StoreDist::DIST_FIRST_ELEMENT_B32` 无论掩码如何都只写 lane 0 | M36 README §4.1-1 | **官方定义就是如此 ⇒ 我方误用，不是平台 quirk，不作为 quirk 候选入 docs**。官方 StoreAlign 表3 原文：「**忽略mask**，向dst中搬运src**第一个元素**」。实测：连续掩码 {0,1,8,64} 与**非连续/交替掩码**（M4 / M3 / MaskGenWithRegTensor 真交替）**一律只写 1 个 lane**；同一地址/同一指令位点换 `DIST_NORM_B32` 掩码**严格生效**（写出 0/1/8/64、16、22、32 lane，含真交替的"全偶/全奇 32 lane"）⇒ mask 按**位**参与、机制本身可用，FIRST_ELEMENT 不吃掩码是它的定义（§4） |
| **(b)** `LoadDist::DIST_BRC_B32` 从「MTE2 刚搬进的 UB 槽」读会非确定性读到旧值 | M36 README §4.1-2 | **归因修正：问题不在 BRC、也不在 UB 槽，而在 GM 回环的排序。** ① 探针 B（23 变体 × 5 独立进程，把 mission 列的四个轴全扫完）**复现不出任何"MTE2→V 的 UB 槽交接"缺口**；② **agent-hcmix（M36）已回信确认原实现**：injW 表走的是 `MTE3: UB→GM`（`DataCopyPad` 16B）→ `MTE2: GM→UB`（`DataCopyPad` 4B→32B）→ 再 `DIST_BRC_B32`，且**两侧用了两个不同的 BufferID**（`BUF_AIV_SG=5` 释放、`BUF_AIV_IW=3` 立即取）⇒ **那次释放没有任何消费者 acquire，对"GM 写→GM 读"不构成任何排序**；③ 探针 C 把这条回环单独隔离，`rt_unordered` 与**完全复刻其形态的 `rt_twotok`** 都稳定复现（读到尚未写回的值），加上任一排序（硬件事件 / 同一 token 的阻塞释放交接）即 64/64 正确。M36 自己也已把 README §4.1 的归因改掉。⇒ **是「同步缺失（释放无人取）」，不是"BRC 读 MTE2 搬进来的槽不稳定"**（§5/§6） |

**两条都不是平台 quirk**——(a) 是 API 语义误用，(b) 是**同步缺失**（跨 pipe 依赖漏了排序，且"阻塞释放 + 对侧 acquire"必须成对）。**因此建议都不进 `docs/05` §6.2（实测硬件行为）**；(b) 建议作为 §6.1 里既有"跨 pipe 交接"规则的**适用场景扩张**记录，建议入库文本见 §7。

---

## 1. 目录与构建

```
probe_vf_ldst/
├── CMakeLists.txt                  # 独立 CMake 工程（find_package(ASC) + --npu-arch=dav-3510，仿 probe_sync_quirks）
├── probe_a_storedist.asc           # 探针 A：StoreDist 掩码语义（(a)；连续掩码 + 非连续/交替掩码两族，各 2 target）
├── probe_b_ldst_handover.asc       # 探针 B：MTE2 写 UB 槽 → BufferID 交接 → V 侧读（(b)，23 变体）
├── probe_c_gm_roundtrip.asc        # 探针 C：同核 GM 回环 MTE3→MTE2 的可见性（(b) 的真机制，7 变体）
├── run_probes.sh                   # 一条命令重跑全部变体 + 每个变体 ≥5 个独立进程 + 重生成 evidence/
│                                   # 同时产出 logs/declared_groups.txt（声明集，与实跑同源）
├── tools/summarize.py             # 逐进程汇总 + 覆盖自校验（声明集 vs 实到集）+ 对照自检 + 三态退出码
├── tools/ub_regions.py            # 判定项 dump 一致性的**分区**证据（区分判定项区与未写 scratch）
└── evidence/                       # 归档证据（见 §8 索引）
```

```bash
cd probe_vf_ldst
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
bash run_probes.sh 5            # 34 个 target → 46 个变体组 × 每个 5 个独立进程 = 230 次 launch
                                # 耗时提示（**本机观测、非证据**：单卡与其他 worker 共用，
                                #   每个独立进程约 10~60 s 的 ACL 初始化开销，量级为数十分钟到 ~1.5 小时）
#   → evidence/compile_matrix.txt / run_matrix.txt / sha256_manifest.txt / run_summary.txt
# 单变体手工复现（**完整编译选项**见 evidence/commands.txt）：
./build/probe_a_first_elem   evidence/dumps 0 8    # 参数：dumpdir procIdx mask有效lane数
./build/probe_b_rel1_brc     evidence/dumps 0
./build/probe_c_rt_unordered evidence/dumps 0
```

编译期宏（每个变体一个可执行文件）：探针 A 的 2 个（dist × 掩码族）、探针 B 的 9 个维度、
探针 C 的排序(4) × 回写形态(2)，取值与含义见 `CMakeLists.txt` 的规格串与各 `.asc` 顶部注释。
**编译选项完整归档在 `evidence/commands.txt`**（含 `-std=c++17 --npu-arch=dav-3510 -O3 -DNDEBUG
-c --asc-aicore-lang` 与 `-DNPU_ARCH_DAV_3510`；结论能否复现的关键）。

### 三态退出码与覆盖自校验（tower 2026-09-26 的脚本规则）

`run_probes.sh` / `tools/summarize.py`：`0` = 比过、**声明集全到齐**、每组进程数达标、
跨进程判读一致、对照自检符合预期；`1` = 比过但有差异（**缺组 / 进程数不足** / 判读不一致 / 对照自检不符）；
`2` = **没得比**（没有任何 `run_*.log`，或缺 `declared_groups.txt`）。

**计数与匹配器同源**：声明集 `logs/declared_groups.txt` 由 `run_probes.sh` 在**跑变体的同一组循环里**
逐条产出，summarize.py 拿它当覆盖基准 ⇒ 「声明了多少组」不可能与「实际跑了多少组」各写一份而漂移；
汇总里**声明数 / 实到数 / 进程数达标数三栏分开打印**（不做成一个"0 问题"）。

**三条负向对照已实测并归档**（`evidence/logs/negative_control_*.log`，都是按 tower 规则
"找一条已知会被漏掉的样本，确认它被报成不合格"）：

| 对照 | 做法 | 期望且实测 |
|---|---|---|
| `negative_control_nocompare.log` | 空 evidence 目录 | `RESULT: SKIPPED` + **exit 2** |
| `negative_control_partial_missing.log` | **删掉两个整组变体**（`run_probe_b_rel1_brc_p*.log`、`run_probe_b_rel0_brc_p*.log`） | `RESULT: DIFF`（缺组）+ **exit 1**（修复前是 `OK 31/31` exit 0） |
| `negative_control_fewer_procs.log` | **只留 1/5 个进程** | `RESULT: DIFF`（进程数不足）+ **exit 1**，且结论行给**实测**最小进程数（修复前宣称"每变体 5 个独立进程"） |

## 2. 数据契约（怎么读 dump）

* **探针 A**：`UB_DST` 预置 `POISON = -999.0f`（不在源值域 `[100,164)` 内），源寄存器 lane i = `100+i`。
  dump = `UB_DST` 的 256B 逐字节快照 ⇒ 「被写的 lane」直接可见（写出的值 == 源值，未触碰 == `-999`）。
* **探针 B / C**：chunk k 的期望值 = `1000.0f + k`（**每 chunk 互不相同**，所以"读到旧值"能定位来源）。
  判读分类（T1 逐字节，`docs/17` §1.1）：

  | 分类 | 含义 |
  |---|---|
  | `.` ok | `got == want` |
  | `p` stale_prev | `got == want-1`（读到**上一个** chunk 的值 = 槽复用时最自然的"旧值"） |
  | `n` ahead | `got == want+1`（读到**下一个** chunk 的值 = 生产者抢跑覆盖） |
  | `X` poison | `got == -999`（该槽**从未被写过**） |
  | `z` zero | `got == 0`（**M36 的症状签名**：`out = h` ⟺ 广播读到 0） |
  | `?` other | 其它（残留 / 第三来源） |

  判据 = **判定项**：每个 chunk 的 lane0 == `1000+k`；**报告项**：全 VL lane 不符点数、dump 跨进程 sha256。
* 同步：只用 BufferID（`GetBufInternal` / `RlsBufInternal`）；release 一律 `false`（= CANN `ASC_LOCK_BLOCK` 默认、阻塞）；探针 B 的历史 `PROBE_B_REL` 档位键已按 all-false 政策作废。

---

## 3. 官方文档取证（(a) 定性的前提）

头文件里 `LoadDist`/`StoreDist` **只有枚举、没有任何语义注释**
（`basic_api/reg_compute/kernel_reg_compute_utils.h` 的 `enum class LoadDist` / `enum class StoreDist`；
**按符号定位，不按行号**），
所以必须查文档。原文已归档 `evidence/doc_refs.txt`（含两份文档的 sha256）：

| 来源 | 表 | 原文 |
|---|---|---|
| `/workspace/cannbot-knowledge/.../reg_data_store/store_align_continuous.md`（CANN 9.1.0 随包知识库，sha256 `0ae53eba…`） | 表3 StoreDist（单搬出） | `DIST_FIRST_ELEMENT_B32` → 「**忽略mask**，向dst中搬运src第一个元素，数据类型为b32。」对齐约束 4B |
| 同上表3 | `DIST_NORM_B32` | 「正常模式，搬运数据量为VL，数据类型为b32。」对齐约束 32B（**不忽略 mask**） |
| `/workspace/cannbot-knowledge/.../reg_data_load/load_align_continuous.md`（sha256 见 `doc_refs.txt`） | 表3 LoadDist（单搬入） | `DIST_BRC_B32` → 「搬运一个b32类型的数据，并Broadcast到所有元素位置。」对齐约束 4B |

在线同源（昇腾社区 CANN 920beta1 文档，同一张表）：[StoreAlign_continuous](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/API/ascendcopapi/docs/zh/api/SIMD-API/%E5%9F%BA%E7%A1%80API/Reg%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/Reg%E6%95%B0%E6%8D%AE%E6%90%AC%E5%87%BA/StoreAlign_continuous.md)。

⇒ **(a) 的官方定义 = 「忽略 mask + 只搬第一个元素」**，所以「按 lane 掩码选流」本身就是误用，**不需要再谈"语义不符"**。
（第二条 bullet 任务里的「若需探针」按 mission 判定条件为**不必须**；本目录仍做了最小实证，把「文档说 X / 实测 Y」并列归档。）

---

## 4. 结论：探针 A —— `StoreDist` 掩码语义（(a)）

`probe_a_storedist.asc`：单 launch。MTE2 把源/毒值搬进 UB → V 侧 `LoadAlign` 全宽读源寄存器
（lane i = `100+i`）→ `StoreAlign<float, PROBE_A_DIST>`，掩码由**运行时实参**选
（一次 launch 只跑一种 mask —— M24 reviewer 的纪律）→ 256B 逐字节 dump。
两个 compile-time 轴：宏 `PROBE_A_DIST`（0 = `DIST_FIRST_ELEMENT_B32`，1 = `DIST_NORM_B32` 对照）
× 宏 `PROBE_A_ALT`（0 = 连续掩码族，1 = 非连续/交替掩码族），共 **4 个 target × 4 个 mask × 5 个独立进程**。

**两类掩码（连续族 + 非连续/交替族）**：
只测连续族时，「掩码按位参与」只能靠"低 8 lane 恰好被写"间接推断 —— 而 `UpdateMask<T>(count)`
只能表达「最低 count 个 lane 有效」这一族前缀集合，**造不出非连续集合**，故另立一族直接观测：

| 族 | target | 掩码来源与取值 |
|---|---|---|
| 连续 | `probe_a_first_elem` / `probe_a_norm_b32` | `UpdateMask<float>(count)`，count ∈ {0,1,8,64} = 全 0 / 单 lane / 低 8 lane / 全 1 |
| **非连续/交替** | `probe_a_first_elem_nc` / `probe_a_norm_b32_nc` | sel 0 = `CreateMask<float, MaskPattern::M4>`；sel 1 = `MaskPattern::M3`；sel 2/3 = `MaskGenWithRegTensor<uint32_t,0>` 从 host 给的 64bit 码型（`0x5555…` / `0xAAAA…`）造**真交替**掩码 |

> 说明（初版遗漏的原因，保留在案）：`UpdateMask<T>(count)` **只能生成「最低 count 个 lane 有效」**这一族，
> 表达不出非连续掩码；所以「掩码是按位参与还是按数量参与」这一层曾只能靠"低 8 lane"间接推断。
> 补上非连续族后，这一层变成**直接可观测量**（见下表）。

### 读数（`evidence/logs/run_matrix.txt`，`evidence/run_summary.txt`）

| dist | mask 有效 lane | 实测落盘 | 与文档一致？ |
|---|---|---|---|
| `DIST_FIRST_ELEMENT_B32` | 0（**全 0 掩码**） | 写出 **1** lane（`dst[0]=100`），其余 63 lane 保持 `-999` | ✅ 「忽略mask」 |
| `DIST_FIRST_ELEMENT_B32` | 1 | 写出 **1** lane | ✅ |
| `DIST_FIRST_ELEMENT_B32` | 8 | 写出 **1** lane | ✅ |
| `DIST_FIRST_ELEMENT_B32` | 64（全 1） | 写出 **1** lane | ✅ |
| `DIST_NORM_B32`（对照） | 0 | 写出 **0** lane | ✅「掩码生效」 |
| `DIST_NORM_B32`（对照） | 1 | 写出 **1** lane | ✅ |
| `DIST_NORM_B32`（对照） | 8 | 写出 **8** lane | ✅ |
| `DIST_NORM_B32`（对照） | 64 | 写出 **64** lane | ✅ |

**非连续/交替掩码族**（同一个 256B 落盘快照，位图逐字节读出）：

| dist | target | 掩码 | 实测落盘（位图） | 写出 lane 数 |
|---|---|---|---|---|
| `DIST_FIRST_ELEMENT_B32` | `probe_a_first_elem_nc` | M4（周期 4） | 恒只有 lane 0 被写 | **1** |
| `DIST_FIRST_ELEMENT_B32` | `probe_a_first_elem_nc` | M3（周期 3） | 恒只有 lane 0 被写 | **1** |
| `DIST_FIRST_ELEMENT_B32` | `probe_a_first_elem_nc` | 交替 `0x5555…` | 恒只有 lane 0 被写 | **1** |
| `DIST_FIRST_ELEMENT_B32` | `probe_a_first_elem_nc` | 交替 `0xAAAA…` | 恒只有 lane 0 被写 | **1** |
| `DIST_NORM_B32`（对照） | `probe_a_norm_b32_nc` | M4 | `#...#...#...`（lane 0,4,8,…,60） | **16** |
| `DIST_NORM_B32`（对照） | `probe_a_norm_b32_nc` | M3 | `#..#..#..#..`（lane 0,3,6,…,63） | **22** |
| `DIST_NORM_B32`（对照） | `probe_a_norm_b32_nc` | 交替 `0x5555…` | `#.#.#.#.#...`（**全偶 lane**） | **32** |
| `DIST_NORM_B32`（对照） | `probe_a_norm_b32_nc` | 交替 `0xAAAA…` | `.#.#.#.#.#..`（**全奇 lane**） | **32** |

跨进程判读一致、同配置的 256B 落盘 dump sha256 逐字节一致（`evidence/logs/ub_region_consistency.txt`、
`run_summary.txt`；这些 dump **全量初始化**，所以整文件一致成立）。

### 【已确证】

1. **`DIST_FIRST_ELEMENT_B32` 完全忽略 mask**：连「全 0 掩码」（`UpdateMask<float>(0)`）也照样写 `dst[0]`。
2. **它只写 lane 0**：`dst[1..63]` 一个字节都没动（预置 `-999.0f` 原样保留）。
3. **区分「定义如此」与「掩码被静默忽略」的对照成立**：**同一 UB 地址、同一指令位点**换成 `DIST_NORM_B32`，
   掩码**严格生效**（0/1/8/64 lane）⇒ mask 机制在该位点本身可用 ⇒ `FIRST_ELEMENT` 不吃掩码是它的**定义**，
   不是设备把掩码丢了。
4. **掩码按「位」参与、不是按「数量」参与**（由非连续/交替掩码直接观测）：非连续掩码下
   `DIST_NORM_B32` 写出的 lane 集合是**周期性/交替集合**而非前缀 —— M4 写 16 个（每 4 个取 1）、
   M3 写 22 个（每 3 个取 1）、真交替掩码写 32 个（**全偶 / 全奇**两个反相）。
   在同样的非连续/交替掩码下，`DIST_FIRST_ELEMENT_B32` 仍**恒只写 1 个 lane** ⇒ 「忽略 mask」在
   非连续掩码上同样成立，不是"低 N lane 恰好没暴露"。
5. ⇒ **定性：我方误用**（M36 拿它做「按 lane 掩码选流」）。M36 已采用的规避
   `Compares → Select → Reduce<SUM> → DIST_FIRST_ELEMENT_B32` 是**正确写法**（先把目标元素归约到 lane 0，再落盘）。
   `m6`/`m4`/`m12`/M24 里用它配 `VL1` 掩码取标量、存行状态，都是**符合定义**的用法，不受任何影响。

### 【仍存疑】

* 无。（此条由文档 + 实测双向闭合；剩下唯一可议的是"要不要写进 docs"，见 §7 —— 建议写成 **§6.1 API 约束**而不是 §6.2 quirk。）

---

## 5. 结论：探针 B —— MTE2 写 UB 槽 → BufferID 交接 → V 侧读（(b)）

### M36 交过来的原文（`m20_hyperconn/README.md` §4.1，逐字引用以便 reviewer 对照）

> 1. **`StoreDist::DIST_FIRST_ELEMENT_B32` 恒写 lane 0，与掩码无关**——不能拿它做「按 lane 取元素」。
>    本工程 W0 段需要把 4 个流的注入权重分别落到 4 个 32B 槽首，改为
>    `Compares(cmpS, idx, s) → Select(sel, v, zero, cmpS) → Reduce<SUM> → DIST_FIRST_ELEMENT_B32`
>    （掩码清零其它 lane 后归约到 lane 0）。原写法把 4 个槽都写成 lane 0 的值（`exact0` 档因四值相等而未暴露）。
> 2. **`LoadDist::DIST_BRC_B32` 从「MTE2 刚搬进来的 UB 槽」读会非确定性读到旧值**——
>    S1 按 item 从 GM 取 `injW[s]` 再广播时，部分 chunk 退化为 `out = h`（137/160 chunk 错，且每次跑错的位置不同）。
>    改为「W0 段用 V 自己写出全表（**V→V**），S1 再从 32B 对齐槽 BRC 广播」后稳定正确；
>    W0 的 `LoadAlign<bfloat16_t, DIST_UNPACK_B16>`（普通全宽读）不受影响。

`probe_b_ldst_handover.asc` 忠实复刻 M36 的 S1 形态：**per-item 取一个标量 → 广播 → 用**。
启动时 V 把整段槽区写成 `POISON`（确定性初值），随后每个 chunk：
`生产者（MTE2 或 V）写 chunk k 的值进 UB 槽 → 释放 BufferID（按 REL 选 mode）→ V 侧 get → 用 READ 指定的读法
把槽读进寄存器 → 全 VL 落盘`，循环后整段 UB 原样 dump 回 GM。

### 变体矩阵（23 个变体，每个 × 5 个独立进程；规格串见 `CMakeLists.txt`）

| 变体 | 轴 | 读数 |
|---|---|---|
| `rel0/rel1/rel2` × `brc/norm/getval`（9 个） | 生产者释放 **mode=false / mode=true / mode=true+PipeBarrier<MTE2>**（当时的命名见下注）× 读法 **`DIST_BRC_B32` / `LoadAlign` 全宽 / `LocalTensor::GetValue`**，整块 256B、所有 chunk 复用一个槽 | 全部 **64/64 正确** |
| `rel1_brc_membar` | BRC 前加 `LocalMemBar<VEC_STORE,VEC_LOAD>`（负对照） | 64/64 正确 |
| `rel1_brc_mid` | acquire 与读之间插一次无关 32B scratch 的 store→load | 64/64 正确 |
| `rel1_brc_g32` / `rel1_brc_g4` | 搬运粒度 32B（`DataCopy` 8 元素）/ 4B（`DataCopyPad`） | 64/64 正确 |
| `rel1_brc_perchunk` | **每 chunk 自己的槽**（无同槽复用） | 64/64 正确 |
| `rel1_brc_off4` | 读地址 = 槽首 **+4B**（4B 对齐、非 32B 对齐） | 64/64 正确 |
| `rel1_brc_vprod` | **生产者 = V**（`Duplicate`+`StoreAlign`，即 M36 的规避形式） | 64/64 正确 |
| `vrelimm` / `rel0_vrelimm` | **消费者释放改 mode=false** ⇒ 生产者可抢先覆盖同槽 | 64/64 正确 |
| `backlog_rel1` / `backlog_rel0` / `backlog_rel2` | 槽写**之前**先发一条 **16KB 大 DataCopy** ⇒ MTE2 队列里有真实在飞工作量，让 mode=false 的释放有机会被抢先 | 64/64 正确 |

> **release mode 命名口径**（权威口径，人类裁定；本表读数数值未改）：`false` = CANN `ASC_LOCK_BLOCK`（默认，「阻塞」）、`true` = `ASC_LOCK_NON_BLOCK`；**两种模式都等本 pipe 已发射指令落地**，`true` 额外等此前同 id 的释放 ⇒ `true` 更保守。本表里 `rel0`（mode=false）当时被记作「立即生效」、`rel1`（mode=true）被记作「drain」，两处命名均不准确（`false` 不是「立即生效」，`true` 也不是「drain」）；下文 §6/§7 沿用旧称处同此口径。

### 对照（证明本矩阵的 PASS 不是空洞的）

| 变体 | 设计意图 | 实测（必须如此，否则矩阵的 PASS 不可信） |
|---|---|---|
| `noprod`（`PROD=2`） | **生产者关闭**（只握手不写槽） | **64/64 读到 POISON** ⇒ 读路径确实在读这个槽，判读器不是恒报 ok ✅ |
| `prod3_selftest`（`PROD=3`） | 生产者**故意写上一 chunk 的值** | 判读器立刻报 **`ok=1 stale_prev=63`**，首个不符 chunk=1 得到 1000 期望 1001 ✅ ⇒ 分类器真的能识别"读到旧值" |

### 定性（回答 mission 的两个问题）

* **Q1「是 BRC 特有，还是整类装载读都被影响？」** —— **都不是**。在本最小形态下，
  `DIST_BRC_B32` / `LoadAlign` 全宽（`DIST_NORM`）/ `LocalTensor::GetValue` 三类读法行为**完全一致**（64/64），
  `DIST_BRC_B32` 没有表现出任何"指令类型敏感"的可见性缺口。
* **Q2「是我们少了一步 drain/acquire，还是跨 pipe 可见性对装载指令类型敏感？」** ——
  **在 MTE2→V 这一对 pipe 上两者都对不上**：连当时被记作「立即生效、不承载可见性」的
  `mode=false` 释放（`rel0`、`rel0_vrelimm`、`backlog_rel0`）都拿到正确数据。
  ⇒ 本形态下 MTE2→V 的 BufferID 交接（无论 mode）**足以承载可见性**；M36 的现象不在这条通路上。

### 【已确证】

1. 23 个变体 × 5 个独立进程 = 115 次 launch，**22 个判定项变体全部 64/64 正确**（第 23 个 `prod3_selftest`
   是判读器自检，设计上就该 FAIL 并稳定报 63 个 `stale_prev`）；**无一例跨进程非确定性**；
   **判定项 dump（探针 A 的 `a_*_mask*_p*.bin`、探针 B 的 `b_*_p*_out.bin`）跨进程逐字节一致**。

   ⚠ **口径限定**：`b_*_p*_ub.bin` 是**整段 UB 现场快照**，其中
   `UB_MID`(32768..33024) 与 `UB_BACKLOG`(33024..49408) 两块 scratch 在内核**从不写它们**的变体里
   是 UB 残留 ⇒ 跨进程内容随机，**跨进程 sha256 不稳定**（实测 **23/23** 个 B 变体的 `_ub.bin`
   整文件哈希跨进程不同；而**槽区 `[0,16384)` 与输出区 `[16384,32768)` 在 23/23 个变体上跨进程逐字节一致**）。
   **只有这两个判定项区可作一致性证据**；
   分区比对见 `evidence/logs/ub_region_consistency.txt`（`tools/ub_regions.py` 按区算哈希）。
   这与 M36 那条"整槽清零"是同一族问题：**未写内存会让 dump 不稳定，但它不是判据**。
   （`evidence/run_summary.txt` 末行：`RESULT: OK（声明 46 组全到齐…）`）
2. 读法、释放模式、搬运粒度（4B/32B/256B）、槽复用与否、中间 V 操作、读地址 4B 偏移、
   MTE2 在飞 backlog —— **都不是**触发条件。
3. 探针的判读器**能**识别"读到旧值"与"读到未写内容"（`noprod` / `prod3_selftest` 对照）。
4. ⇒ **「MTE2 写 UB 槽 → V 侧读」这条通路在最小形态下没有可见性缺口**，与读法
   （BRC / 全宽 / GetValue）、释放模式、搬运粒度都无关。

### 【仍存疑 / 未对齐】

1. 探针 B **未覆盖**的形态：跨核（CrossCore）交接、`__mix__(1,2)` 里 AIC 同时运行、
   **数据绕行 GM**、一个 launch 内多个段的先后关系。
   —— 其中「绕行 GM」由 §6 的探针 C 承担，并**已由 M36 的事实确认**（见 §6）。
2. 探针 B 自身**不**证明"M36 当时的代码没问题"，只证明这一条通路没问题。归因由 §6 给出。

---

## 6. 结论：探针 C —— 同核 GM 回环的可见性（(b) 的真机制，**已由 M36 确认**）

### 起点：先读 M36 的现役源码，再等它的回信

探针 B 扫完 mission 列的全部变量后仍不复现，于是去读 M36 的**现役源码**
（`m20_hyperconn/m20_hyperconn.asc`）：`InjwStage`（W0）里 V 算出 injW 全表后
`DataCopy(injwGm[0], tabL, …)` —— **MTE3 把整张表落 GM**；`CombineStage`（S1）读的却是
**UB 里的同表**（`iwTabUb`）⇒ 现役版本**不绕 GM**；而 `ProcessAiv` 里 **W0 与 S1 之间没有任何 barrier**。
⇒ 原设计很可能是「S1 用 MTE2 从 GM 把 injW 取回 UB 槽再 BRC」。
（引用只给**符号**，不给裸行号 —— 行号会随 M36 分支漂移。）

**agent-hcmix（M36）随后回信确认了这个假说**，并自查出第二个因子。其原话要点（不改它的代码，只引事实）：

> * **injW 表当初确实先在 GM 里**：`W0` → `DataCopyPad(injwGm[mi*INJW_STRIDE], sigL, ExtBlock1(16))`（MTE3，16B）；
>   `S1` → `DataCopyPad(iwL, injwGm[mi*INJW_STRIDE + s], ExtBlock1(4), …)`（MTE2，4B）→ 再
>   `LoadAlign<float, LoadDist::DIST_BRC_B32>(w, iwUb)`。**所以出问题的是同核 `MTE3 → GM → MTE2`，
>   不是「BRC 读 MTE2 搬进来的 UB 槽」；它 README §4.1 原来的归因是错的，已改。**
> * **两个不同的 BufferID**：生产者 `BUF_AIV_SG=5` drain 释放；消费者 `BUF_AIV_IW=3` 立即取。
>   **两者不是同一个 token ⇒ 生产者的 drain 释放没有任何消费者去 acquire，对 GM 写→GM 读不构成任何排序。**
> * 粒度已被排除：W0 侧 16B；S1 侧 4B 失败、**改 32B 仍失败**。UB 槽 32B 对齐，广播源地址 = 槽首。

### 实验与读数（7 个变体，每变体 5 个独立进程）

W0 等价：V 写 `ITERS×256B` 的表 → **MTE3 落 GM（`gmMid`，host 预置 `POISON`）**；
S1 等价：MTE2 逐 chunk 从 `gmMid` 取回 UB 槽 → V `DIST_BRC_B32` 读槽 → 落盘。
两个轴：`PROBE_C_ORDER`（两次搬运之间插什么排序）× `PROBE_C_SHAPE`（回写形态）。

| 变体 | 排序 | 回写形态 | 判读 | 首个不符 |
|---|---|---|---|---|
| `probe_c_rt_unordered` | **无任何排序** | 整块 16KB | **FAIL** `ok=63 poison=1`，5/5 进程一致 | chunk 0 读到 `-999` 期望 `1000` |
| **`probe_c_rt_twotok`** | **两个不同 BufferID**（token A drain 释放**无人取** / token B 立即取）—— **完全复刻 M36 已确认的形态** | 整块 16KB | **FAIL** `ok=63 poison=1`，5/5 进程一致 | 同上 |
| `probe_c_rt_event` | `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` | 整块 16KB | **PASS 64/64** | — |
| `probe_c_rt_bufid` | MTE3 drain 释放 → **对侧取同一个 token** | 整块 16KB | **PASS 64/64** | — |
| `probe_c_rt_small_unordered` | 无任何排序 | 逐 chunk 16B ×64 | PASS 64/64 | — |
| `probe_c_rt_small_twotok` | 两个不同 BufferID | 逐 chunk 16B ×64 | PASS 64/64 | — |
| `probe_c_rt_small_bufid` | 同 token drain 交接 | 逐 chunk 16B ×64 | PASS 64/64 | — |

所有变体的 GM 最终内容都是 `64/64` 正确（写回最终都到了），**错的只是"读者的时机"**。

### 【已确证】定性

* **这是一条真实、可复现的可见性断裂**：同核 `MTE3: UB→GM` 紧接 `MTE2: GM→UB` 读**同一地址**，
  中间无跨 pipe 排序 ⇒ MTE2 读到**尚未写回**的 GM 旧内容（本探针里是 `POISON`；
  在 M36 里对应"读到 0 ⇒ `out = h`"，正是 `z`/`X` 那一类签名）。
* **最隐蔽的形态是"看似有同步、实则没有"**：`probe_c_rt_twotok` 里生产者**确实**用了 BufferID 释放
  （`RlsBufInternal<PIPE_MTE3, true>`），但**释放的是 token A、消费者取的是 token B**
  ⇒ 这条释放对 GM 写→GM 读**不构成任何排序**。**这正是 M36 的原实现**。
  ⇒ 规则应当写成「**阻塞释放必须被对侧 `acquire` 同一个 token 才算同步**」，
  而不是"用了 BufferID release 就安全"。
  > ⚠ **M183 需重读**（`probe_vf_ldst/README.md:307`–`:311`）：该臂当时用的是 `RlsBufInternal<...,true>`；全项目已统一 `false`（= CANN `ASC_LOCK_BLOCK` 默认、阻塞），本条结论的模式依赖已变，**需重读**（读数数值未改）。
* **它是"我们少了一步同步"而不是"硬件对装载指令类型敏感"**：加上任一有效排序（硬件事件，
  或同一 token 的阻塞释放交接）立刻 64/64 正确，**与搬运粒度、与读法（BRC/全宽）都无关**。
* 因此这**同样不是 quirk**，是**跨 pipe 数据依赖漏了同步**——与 `docs/05` §6.1 已写的
  「BufferID 只覆盖核内 / 跨 pipe 交接要阻塞释放 + 对侧 get」同源，属**既有规则的适用场景扩张**，不是新规则。

### 【仍存疑】

* **失败率的量级未对齐**：本形态下错的是**最先读的那 1 个 chunk**（5/5 进程一致、位置固定），
  而 M36 报的是 **137/160**（86%，且每次位置不同）；并且「逐 chunk 小回写」形态
  （`probe_c_rt_small_*`）三种排序**全部 PASS**。⇒ **失败率取决于「回写仍在飞时读者已到」的程度**，
  本探针只覆盖到「整块 16KB 回写仍在飞」这一档。**未对齐，不硬给归因**（见 §9-1）。
* M36 原实现的具体形态**已由它自己回信确认**（`MTE3→GM→MTE2` + 两个不同 BufferID、无跨 pipe 原语）；
  但「86% 这个量级由哪一层 timing 决定」仍未对齐。

---

## 7. 建议入库文本（**只是建议**，docs 定稿是 tower 的事；本 mission 不改 `docs/**`）

1. **建议进 `docs/05 §6.1`（规格 / API 约束），不进 §6.2：**

   > **`StoreDist::DIST_FIRST_ELEMENT_B32` 按定义忽略 mask、只把 src 的第一个元素搬到 dst**
   > （官方《连续对齐搬出（StoreAlign）》表3 原文：「忽略mask，向dst中搬运src第一个元素」）。
   > 因此**不能用它做「按 lane 掩码选元素」**；需要时先 `Compares → Select → Reduce<SUM>` 把目标元素
   > 归约到 lane 0，再用 `DIST_FIRST_ELEMENT_B32` 落盘。对照实测（同一地址/同一指令位点换
   > `DIST_NORM_B32` 掩码严格生效，0/1/8/64 lane）见 `probe_vf_ldst/`（M45，探针 A）。
   > `m6`/`m4`/`m12`/M24 用它配 `VL1` 掩码取标量属**符合定义**的用法，不受影响。

2. **建议进 `docs/05 §6.1`（同一条"跨 pipe 交接"规则下的适用场景扩张），不要写成新 quirk：**

   > **同核 `MTE3: UB→GM` 紧接 `MTE2: GM→UB` 读同一地址时，两次搬运之间必须有跨 pipe 排序**，
   > 否则 MTE2 会读到**尚未写回**的旧 GM 内容。可用 `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>`，
   > 或 BufferID **阻塞释放 + 对侧 acquire 同一个 token**。
   > ⚠ **阻塞释放单独不构成同步**：若释放的是 token A、消费者取的是 token B
   > （两个不同 id），这条阻塞释放对 GM 写→GM 读**不产生任何排序**，行为与"完全无同步"一致。
   > 实测（`probe_vf_ldst/` 探针 C，M45，每变体 5 个独立进程）：无排序 / 两个不同 token ⇒
   > 稳定读到尚未写回的值（本形态为先读的 1 个 chunk，M36 现场为 137/160）；
   > 加任一有效排序（硬件事件，或同一 token 的阻塞释放交接）⇒ 64/64 正确。
   > 该形态已由 M36 确认是其原实现（`BUF_AIV_SG=5` 释放 / `BUF_AIV_IW=3` 立即取）。

3. **不建议入库**（未隔离到"平台行为"层面）：M36 的「`DIST_BRC_B32` 读 MTE2 刚搬进的 UB 槽非确定性读到旧值」。
   探针 B 的 23 变体（把读法 / 释放模式 / 搬运粒度 / 槽复用 / 中间 V 操作全扫过，每个 5 个独立进程）
   **在该形态下复现不出任何缺口**；M36 也已把该归因改掉。建议在 `docs/05 §6.3` 的"待复核"栏记一行：
   「**probe_vf_ldst（M45）在探针 B 的最小形态下未复现该归因；真机制是同核 GM 回环缺排序（见 §6.1 新增行）；
   M36 已确认其原实现为该形态**」。

4. **给 M36 的工程建议**（不改它的代码，只给建议）：M36 目前的规避（W0 用 V 写出全表、S1 从 UB 槽 BRC）
   **绕开了 GM 回环，所以是好的**。但本探针的 23 变体最小形态**未复现**其原归因
   （"BRC 读 MTE2 搬进来的槽不稳定"）；按已确认的事实，**若将来又把表落 GM 再取回，
   要补的是回环的跨 pipe 排序（且两个 token 不算交接），而不是换读法**。

## 8. 证据索引（`evidence/`）

| 路径 | 内容 |
|---|---|
| `doc_refs.txt` | 官方文档原文摘录 + 两份文档的 sha256 + 头文件枚举位置（(a) 定性的依据） |
| `commands.txt` | 复现命令、CANN/编译器版本、日期、NPU、**完整编译选项** |
| `logs/compile_matrix.txt` | 34 个 target 的编译通过/失败矩阵（34 target → 46 个变体组：探针 A 的 4 个 target 各含 4 个掩码配置，B/C 的 target 与组一一对应） |
| `logs/build_<target>.log` | 每个 target 的完整构建输出 |
| `logs/declared_groups.txt` | **声明集**（由 `run_probes.sh` 跑变体的同一组循环产出 ⇒ 与实跑同源） |
| `logs/run_<target>[_mask<M>]_p<k>.log` | 每个（变体 × 配置 × **独立进程**）的完整运行输出 |
| `logs/run_matrix.txt` | 逐（变体 × 进程）判读一行 |
| `logs/sha256_manifest.txt` | 每个 dump 文件的 sha256 |
| `logs/ub_region_consistency.txt` | **分区**一致性：判定项区（探针 A 落盘、B/C 的 `_out.bin`）逐字节一致，B 的 MID/BACKLOG scratch 不一致（`tools/ub_regions.py`） |
| `logs/negative_control_nocompare.log` | 负向对照①：空 evidence ⇒ `RESULT: SKIPPED` + exit 2 |
| `logs/negative_control_partial_missing.log` | 负向对照②：**删掉两个整组变体的日志** ⇒ `RESULT: DIFF`（缺组）+ exit 1 |
| `logs/negative_control_fewer_procs.log` | 负向对照③：**只留 1/5 个进程** ⇒ `RESULT: DIFF`（进程数不足）+ exit 1 |
| `dumps/a_<target>_mask<M>_p<k>.bin` | 探针 A 的 256B 落盘逐字节快照（全量初始化 ⇒ 整文件一致） |
| `dumps/b_<target>_p<k>_out.bin` | 探针 B 的 16KB 输出（64 chunk × 64 lane）——**判定项 dump** |
| `dumps/b_<target>_p<k>_ub.bin` | 探针 B 的整段 UB 快照：槽区 16KB + 输出区 16KB + `UB_MID` scratch 256B + `UB_BACKLOG` scratch 16KB = **49408 B**；其中两块 scratch 在内核未写它们的变体里是 UB 残留，**设计上不做确定性要求**（分区证据见 `logs/ub_region_consistency.txt`） |
| `dumps/c_<target>_p<k>_{out,mid,ub}.bin` | 探针 C 的输出 / GM 最终内容 / 整段 UB |
| `run_summary.txt` | **跨进程汇总**：覆盖三栏（声明/实到/进程数） + 每变体跨进程判读一致性与判定项 dump 一致性 + 对照自检 + 三态结论 |

## 9. 未解问题 / 后续

1. **失败率的量级仍未对齐**：M36 报 **137/160**（86%，且位置每次不同），本探针复现的是
   **固定最先读的 1/64**，且"逐 chunk 小回写"形态（`probe_c_rt_small_*`）**三种排序全 PASS**。
   ⇒ 失败率取决于「回写仍在飞时读者已到」的程度：本探针只覆盖到"**整块 16KB 回写仍在飞**"这一档。
   M36 的 S1 每 item 还要搬 H/BO 且消费者更重，其读数落在更宽的一档。
   **未对齐，不硬给归因** —— 要复现该量级需要一个把"读侧也在飞"一起建模的变体。
2. 探针 B 未覆盖 `__mix__(1,2)` 下 AIC 同时运行、跨核 CrossCore 交接对 VF 装载的影响。
3. 探针 C 的 `shape=1`（逐 chunk 16B 小回写）为什么反而 PASS，只有定性的解释
   （小写流短、且读侧被 BUF_SLOT 节流）——未做定量标定。
