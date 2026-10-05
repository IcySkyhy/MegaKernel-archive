# M126 + M133 —— UB 收尾「V 写 → 阻塞释放（false=BLOCK）→ MTE3 搬出」交接形态 + 同 pipe / release 语义探针

> 独立工程（自带 `CMakeLists.txt` + 含 `main()` 的 `probe_ub_bufid_tail.asc`，不碰仓库任何既有 CMakeLists）。
> 目标：把 M117/B3 的 `m=64` 挂死（`m25_attn_fa_core` AIV 收尾段）缩成一个**不含算法**的最小循环，
> 逐档只动一个变量，直接读「挂 / 不挂」与「搬出内容对不对」；并顺带把人类提出的
> **「`get buf/rls buf` 对同一个 pipeline 是否有效」** 用设备读数定死。
>
> **M133 增量**（本文件 §7–§9）：① 把 `skipRlsE3` 首次置位（去掉 MTE3 侧 release），看下一次同 id 的
> `Acq<PIPE_MTE3>` 会不会阻塞；② 就 `RlsBufInternal<pipe,true/false>` 与公开 `asc_lock/asc_unlock(mode)`
> 取四档真机对照读数（前文 §1–§6 仍是 M126 的原读数，未被改写）。

---

## 0. 结论速览（TL;DR）

| # | 结论 | 判据（可复现） |
|---|---|---|
| 1 | **该交接形态本身不构成挂死**：`Acq<PIPE_V>(id) → VF 写 UB → Rls<PIPE_V>(id) → Acq<PIPE_MTE3>(id) → DataCopy(UB→GM) → Rls<PIPE_MTE3>(id)` 在 `N=1/3/8/16/64` 逐档全绿 | §2.1 `base_N*`（每档 3/3，`evidence/logs/tail_base_N*.log`） |
| 2 | 复刻 M117「每工作项 3 次同 id 交接」（P落盘 1 + 收尾 2 半块）在 `N=3/8` 同样全绿 | §2.1 `multi3_N3/N8`（3/3） |
| 3 | 交接**依赖共享同一个 id**：V 与 MTE3 分成两个独立 id（构造上等于没有 V→MTE3 握手）后判据变红 | §2.1 `c_two_ids`（0/3；run=1 `mismatched=14/16`，复审独立复跑 run=1 为 `13/16` —— 该档错 tile 数随运行不同） |
| 4 | **判据能咬住挂死**：去掉 V 侧 release ⇒ 第 2 次 `Acq<PIPE_V>` 阻塞不返回、3/3 超时 124 | §2.1 `neg_skipvrel`（hang=3） |
| 5 | **同 pipe `get/rls` 不保序（人类那个点，实测回答）**：`PIPE_MTE2` 背靠背同 buffer 复用，最终内容退回**上一次 op** 的 tag；drain 与 mode0 都出现。`N=16` 第 1 批：drain `rc==0` **5/8**（即 **3/8** 次运行出现乱序）、mode0 `rc==0` **7/8**（即 **1/8** 次运行出现乱序）；第 2 批与复审复跑见 §3.2.1 | §3.2 `mte2_drain_n16` / `mte2_imm_n16`（`HIST` 行：`obs[…, 下标14]`，下标为 0-based） |
| 6 | 同一现象的**正确规避**：两次同 pipe op 之间加 `PipeBarrier<PIPE_MTE2>`（不是只加在收尾），`N=8/16` 各 8/8 次运行未再出现 | §3 `mte2_pb_between_8` / `mte2_pb_between_n16` |
| 7 | `PIPE_V` 同 pipe 复用在本档（≤16 次 × 2 AIV × 8 次运行）未观察到该现象 | §3 `v_drain_8/v_imm_8/v_n16/v_pb_between_8`（各 8/8） |
| 8 | 对 M117 的**写窄未定界**：本探针把该交接形态、`N`、copy 行数/stride、id 复用**逐维度试过**，没有一档把挂死点搬到「第 3 个工作项」，也没有复现挂死 ⇒ 该交接形态**不构成** M117 `m=64` 挂死的充分条件；**根因未定位**（不写成已定位） | §4 |

> ⚠ 未定界纪律：本 README 只写设备读数直接支持的东西；「M117 根因在跨核一侧」是**待验猜测**，
> 已在 §4 明确标成「未定位 / 下一步」，不作事实引用。

### 0.1 M133 增量速览（本轮新增；详读 §7–§9）

| # | 结论 | 判据（可复现） |
|---|---|---|
| 9 | **漏一次 MTE3 侧 release ⇒ 整框挂死**（与 `neg_skipvrel` 对称的形状成立）：去掉 `Rls<PIPE_MTE3>(idE3)` 后，单 id 档与两 id 档都 3/3 超时 124；同形但保留 release 的对照档 3/3 正常返回 | §7.1（`neg_skipse3` / `neg_skipse3_2id` / `ctrl_e3rls`） |
| 10 | 阻塞点收窄：`N=1`（无第二次 acquire）时两档都返回 ⇒ 卡的是**紧随其后的那一次同 id acquire**；两 id 档里 `ID_B` 只在 MTE3 侧出现且该档挂 ⇒ 卡的正是**下一次 `Acq<PIPE_MTE3>(ID_B)`** | §7.2（`neg_skipse3_N1` 3/3 rc==0、`neg_skipse3_2id` 3/3 rc=124） |
| 11 | 官方 `asc_unlock` 的 `BLOCK/NON_BLOCK` 与底层 bool **一一对应、无反转**（BLOCK=`false`、NON_BLOCK=`true`）；四档 release 原语在收尾交接构造（`N=32`，负向对照变红）里读数**同一结果**，本轮未读到「只有某个 mode 才建立跨 pipe 可见性」的情形 | §8.1（源码映射）+ §8.2（四档各 4/4 绿、`rel_neg_*` 0/4 红） |

> ⚠ **M183 需重读**：本 README 中 mode 相关的历史结论 —— §0.1 第 11 条（`:36`）、§3.3（`:208`）、§8.2（`:455`–`:461`）、§8.3（`:473`–`:476`）、§8.4（`:488`，及 `evidence/logs/` 下 M126 日志）—— 其读数取自 release 侧含 `true` 的时期；全项目已统一 `false`（= CANN `ASC_LOCK_BLOCK` 默认、阻塞），这些结论所依赖的模式已被改动，**需重读**（读数数值未改）。

> M133 的**写窄**：第 9/10 条的挂死是**合成档**（人为去掉一次 release），不得写成 M117 挂死的根因（§7.3）；
> 第 11 条只覆盖两个收尾交接构造，不外推到未测的 pipe/mode 组合（§8.4）。

---

## 1. 被测对象与判据

### 1.1 两个 kernel

**① `probe_tail_kernel`（收尾交接，`probe_ub_bufid_tail.asc:129`）** —— 单个 AIV 循环 `N` 次；
每个「工作项」内做 `handoffs` 次同一形态的交接（`handoffs=3` 即复刻 M117 的 P落盘 + 两个收尾半块）。
核心体（`probe_ub_bufid_tail.asc:151`–`:183`）：

```
Acq<PIPE_V>(idV);           // :151
VfFillRows(ubF, ROWS, tag); // VF 写 UB（__simd_vf__ / __VEC_SCOPE__）
RlsApi<PIPE_V>(idV, apiV);  // :154   V 侧 release（原语由 apiV 选，见 §8.1）
Acq<PIPE_MTE3>(idE3);       // :157
DataCopy(...UB→GM...);      // MTE3 落 GM（DataCopy/DMA）
RlsApi<PIPE_MTE3>(idE3, apiE3); // :183
```

`idV`/`idE3` 由 host 选：**同一个 id（ID_A=0）或两个独立 id（ID_A/ID_B）**。
输出布局：每个 `(aiv, tile)` 一段独立 GM 区，`tile i` 期望逐元素等于 `tag = aiv*4096 + 序号 + 1`；
host 逐元素比对，并给出「错元素等于哪个 op 的 tag」的直方图（`HIST` 行的下标 = **0-based** op 序号，
「下标 k」即第 `k+1` 次 op）。

**② `probe_sp_kernel`（同 pipe 重获取，`probe_ub_bufid_tail.asc:196`）** —— 同一 pipe 连续
`Acq → op → Rls` ×`N`（`PIPE_V`：VF 写同一 UB；`PIPE_MTE2`：DataCopy GM→同一 UB，源每轮不同 tag）。
收尾在生产者 pipe 上再取一次令牌 → **release**（原语由 `tailApi` 选，见 §8.3）→ 经**同一 id** 交接给 MTE3 搬出
（`probe_ub_bufid_tail.asc:227`–`:243`）。于是「MTE3 读到的内容」= 生产者 pipe 全部退休后的内容。
判据：最终内容应等于**最后一次 op** 的 tag；退回更早的 tag 即同 pipe 完成序被颠倒。
> 关键：搬出走的是**同一个 id 的跨 pipe 阻塞释放握手**，所以本实验**不掺** MTE2→MTE3 的可见性 —— 这是隔离
> 「同 pipe 保序」与「跨 pipe 可见性」两件事的要点（早期版本把搬出走 `ID_DRAIN`、无握手，读数被后者污染，
> 已把那一版作废，见 §3 的「方法修正」）。

### 1.2 与 M117 现场的对照（`文件:行`）

- M117 段体（只读参考）：`/workspace/ascend_mega_kernel/.tower/worktrees/wt-117/m15_layer_loop/m15_attn_fa_core.h`
  - AIV 收尾：`:824`–`:838`（`for h2 in 0..1: Acq<PIPE_V>(B_PC) → FacNormalizeVf → Rls<PIPE_V>(B_PC) →
    Acq<PIPE_MTE3>(B_PC) → DataCopy(outGM) → Rls<PIPE_MTE3>(B_PC)`）
  - P 落盘（同 id 的另一次交接）：`:805`–`:810`
  - 同步原语定义：`:135`–`:136`（`Acq = GetBufInternal<p,false>`、`Rls = RlsBufInternal<p,true>`）
- 故障日志：同 worktree `m25_attn_fa_core/evidence/run_m64_clean.log`（`EXIT_C=124`）、
  `evidence/probe_logs/run_m64i.log` / `run_m64j.log`（`nBlk=1`，AIV0 在 `wi=2` 的收尾处停在
  `PV_RDY ok` 之后、`datacopy done` 之前）。

---

## 2. 变量矩阵（tail：收尾交接）

命令（本目录）：`bash run_probe.sh tail '<正则>'`（脚本自带 `flock -w 300 /tmp/npu0.lock`，进锁先 `npu-smi`，
每条 `timeout 40s`）；每档 3 次独立进程。原始日志 `evidence/logs/tail_<变体>.log`，
汇总 `evidence/logs/tail_matrix.txt`。

### 2.1 结果表

| 变体 | 变量 | 读数（3 次独立进程） | 结论 |
|---|---|---|---|
| `base_N1` | N=1 | rc==0 3/3，`match=4096/4096` | 冒烟过 |
| `base_N3` | **N=3（mission 预期复现档）** | rc==0 3/3，`match=12288/12288` | **未挂死、内容全对** |
| `base_N8` | N=8 | rc==0 3/3 | 未挂死 |
| `base_N16` | N=16 | rc==0 3/3 | 未挂死 |
| `base_N64` | N=64 | rc==0 3/3 | 未挂死 |
| `multi3_N3` | 每工作项 3 次同 id 交接（M117 形态），N=3 | rc==0 3/3，`match=36864/36864` | 未挂死 |
| `multi3_N8` | 同上，N=8 | rc==0 3/3 | 未挂死 |
| `a_noVdrain` | V 侧 release 用 mode0 | rc==0 3/3 | 本档内容全对 |
| `a_noE3drain` | MTE3 侧 release 用 mode0 | rc==0 3/3 | 本档内容全对 |
| `a_bothimm` | 两侧都 mode0 | rc==0 3/3 | 本档内容全对 |
| `b_one_strided` | 单次 32 行带 `dstStride` | rc==0 3/3 | 未挂死 |
| `b_rows` | 32 行逐行 | rc==0 3/3 | 未挂死 |
| `b_one_contig` | 单次连续 | rc==0 3/3 | 未挂死 |
| `c_two_ids` | **V 与 MTE3 分两个独立 id** | rc==0 **0/3**；run=1 `mismatched=14/16`、`match=4608/32768`（复审独立复跑 run=1 为 `13/16`） | **变红：本 idiom 内该握手必需**（限定见 §2.2 (c) 与 §4） |
| `neg_tagshift` | 写出 tag 偏 1（可控变红） | rc==0 **0/3**，`mismatched=16/16` | 内容判据能变红 |
| `neg_skipvrel` | 去掉 V 侧 release（可控挂死） | **hang=3，rc=124** | 挂死判据能变红 |

### 2.2 逐条对账（mission 的 (a)–(e)）

- **(a) drain 形态 / 合并 / 分开 / 去掉**：`a_noVdrain`、`a_noE3drain`、`a_bothimm` 与基线同形，均**未挂死**；
  去掉 V 侧 release（`neg_skipvrel`）才挂死（这是刻意的可控挂死对照，说明 `get/rls` 确实成对阻塞）。
  ⇒ 在本档里 drain vs mode0 **不是**挂死开关（但见 §3：drain 也不是同 pipe 保序的保证）。
- **(b) DataCopy 行数 / stride**：`b_one_strided`（单次 32 行带 `dstStride`）、`b_rows`（逐行）、
  `b_one_contig`（连续）、基线 `copyForm=0`（两个 16 行半块带 `dstStride`）**四种都未挂死**。
- **(c) 同 id 复用 vs 两个独立 id**：同 id 全绿；两个独立 id 变红（`c_two_ids`）⇒ V 侧 release 与
  MTE3 侧 acquire 必须落在**同一个 id** 上，否则 MTE3 在 V 写完前就搬（内容 stale）。
  **限定（复审点名）**：`c_two_ids` 的两侧都做了完整的 `Acq/Rls`，只是用了两个不同 mutex id —— 构造上
  等价于「**没有 V→MTE3 依赖**」。因此它的正确含义是「**本 idiom 里跨 pipe 交接必须共用一个 id**」，
  **不等价于**「硬件禁止用两个 id + 其它显式跨 pipe 同步」。（另：该档错 tile 数随运行不同，
  run=1 为 14/16、复审独立复跑 run=1 为 13/16。）
- **(d) 同 pipe 保序**：见 §3（独立 kernel，不与本交接混淆）。
- **(e) 前置工作项 / 「第 3 次」**：在 `N=3` 与 `multi3_N3`（每个工作项 3 次交接）两档都**未复现挂死**；
  `N=1/8/16/64` 与 `N=3` 的读数没有「只有第 3 次特殊」的痕迹 ⇒ **本形态里不存在与「第 3 次」绑定的挂死**。

---

## 3. 同 pipe `get buf / rls buf` 有效吗？（人类原话那个点，实测回答）

命令：`bash run_probe.sh sp '<正则>'`；每档 8 次独立进程（负向对照 3 次）。

### 3.1 方法修正（先说清楚，否则读数会被误读）

第一版把 UB 搬出写成「释放 `ID_A` 后改用另一个 id `ID_DRAIN` 做 MTE3 交接」，**没有**把生产者 pipe 的
阻塞释放交接给 MTE3 ⇒ 那一版的失败可能全部来自 **MTE2→MTE3 可见性**，不能用来判同 pipe 保序。
现版本改成：生产者 pipe 上再取一次令牌 → `Rls<producer>`（release 一律 false=BLOCK）→ `Acq<PIPE_MTE3>(同一 id)` → 搬出，
**跨 pipe 可见性由同一 id 的阻塞释放握手兜住**，剩下的差异只能来自同 pipe 完成序。下表的读数均来自现版本。

### 3.2 结果表

口径：`rc==0 p/N` 为**通过数**（`N` 见格内，多数档 `N=8`）；`出现乱序` = 其补数（`N−p`）。`下标` 为 **0-based** op 序号（`HIST` 口径，
「下标 k」即第 `k+1` 次 op）。表内「读数」列标了批次；第 2 批与复审复跑见 §3.2.1，
`mte2_drain_8` 的 M126/M133 两批对账也在 §3.2.1。

| 变体 | pipe | 逐轮 release | 两次 op 之间排空 | 读数（批次与运行次数见格内） | 结论 |
|---|---|---|---|---|---|
| `v_drain_8` | V | drain | 无 | rc==0 8/8 | 未观察到乱序 |
| `v_imm_8` | V | mode0 | 无 | rc==0 8/8 | 未观察到乱序 |
| `v_n16` | V | drain | 无 | rc==0 8/8 | 未观察到乱序 |
| `v_pb_between_8` | V | drain | `PipeBarrier<PIPE_V>` | rc==0 8/8 | 未观察到乱序 |
| `mte2_drain_8` | MTE2 | drain | 无 | **M126 批 rc==0 8/8；M133 复核批 rc==0 2/4**（run3 `obs[6]=1152`、run4 `obs[6]=128`） | M126 批未出现；**M133 复核批出现乱序**（两批并列见 §3.2.1） |
| `mte2_imm_8` | MTE2 | mode0 | 无 | rc==0 **7/8**（第 1 批，1/8 次出现乱序；run3：stale=下标 6 的 tag，128 元素） | **出现乱序** |
| `mte2_drain_n16` | MTE2 | drain | 无 | rc==0 **5/8**（第 1 批，3/8 次出现乱序；run1/5/7：stale=下标 14，896~1024 元素） | **出现乱序** |
| `mte2_imm_n16` | MTE2 | mode0 | 无 | rc==0 **7/8**（第 1 批，1/8 次出现乱序；run6：stale=下标 14，896 元素） | **出现乱序** |
| `mte2_pb_between_8` | MTE2 | drain | `PipeBarrier<PIPE_MTE2>` | rc==0 8/8 | 未再出现 |
| `mte2_pb_between_n16` | MTE2 | drain | `PipeBarrier<PIPE_MTE2>` | rc==0 8/8 | 未再出现 |
| `mte2_pb_end_8` | MTE2 | drain | 仅**收尾**排空 | rc==0 3/3 | 本档未出现（注意：排空点在收尾，不在两次 op 之间） |
| `mte2_neg_shift` | MTE2 | drain | 无 | rc==0 **0/3**（源偏 1） | 判据能变红 |
| `v_neg_shift` | V | drain | 无 | rc==0 **0/3**（写出偏 1） | 判据能变红 |

失败签名的原始行（第 1 批 `evidence/logs/batches/batch1_sp_mte2_drain_n16.log`；第 1 批原始日志保留在该子目录，
`sp_mte2_drain_n16.log` 现为第 2 批）：

```
SUMMARY mode=sp variant=mte2_drain_n16 run=1 N=16 aivs=2 mismatched=1 ... match=3072 ... rc=1
HIST mode=sp variant=mte2_drain_n16 run=1 N=16 obs=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1024,0] other=0 (下标=第几次 op 的 tag)
```

即：16 次 `Acq/DataCopy/Rls` 之后，最终 UB 里有 1024 个元素是**下标 14（0-based，即第 15 次 op）**的值，
而不是最后一次（下标 15 = 第 16 次 op）。
每次失败都落在「上一次 op」这一档，粒度是部分 buffer（128~1024 个元素）。

### 3.2.1 各批独立读数（`N=16` 两档三批 + `N=8` 一档两批的对账；批次标清）

该现象是**间歇**的，各批出现率不同。下表把本探针第 1 批、第 2 批（本轮复审后复跑）与 r1 复审的独立复跑
**并列**，不取单一批次：

| 变体 | 第 1 批（2026-10-04 早，`batches/batch1_*.log`） | 第 2 批（2026-10-04T09:36–09:37，本 README 提交批） | r1 复审独立复跑（2026-10-04） |
|---|---|---|---|
| `mte2_drain_n16` | `rc==0` 5/8 → **3/8** 出现（run1/5/7；stale=下标 14） | `rc==0` 6/8 → **2/8** 出现（run4/6；stale=下标 14，128~896 元素） | `rc==0` 7/8 → **1/8** 出现（run8；stale=下标 14，896 元素） |
| `mte2_imm_n16` | `rc==0` 7/8 → **1/8** 出现（run6；stale=下标 14，896 元素） | `rc==0` 7/8 → **1/8** 出现（run4；stale=下标 14，1024 元素） | `rc==0` 7/8 → **1/8** 出现（run6；stale=下标 14，128 元素） |

三批的**签名一致**（都是「退回上一次 op」= 下标 14 / 0-based），**出现率不一致**（1/8 ~ 3/8）——
引用时请写「间歇、本批出现 x/8」，不要写成恒定或写成单一批次的值。第 2 批原始日志：`evidence/logs/sp_mte2_drain_n16.log`、
`evidence/logs/sp_mte2_imm_n16.log`（由 `run_probe.sh` 各 8 次独立进程生成，锁内先 `npu-smi`）。

**`mte2_drain_8`（`N=8`）的批次对账（M133 r1 复审 P2-1）**：该档在 M126 与 M133 两轮各有独立读数，
**两批并列**如下（§3.2 表里那一行的原始读数 = M126 批；M133 复核批由 `feat/m133-*` 分支的复跑产生）：

| 变体 | M126 批（原始日志留底 `batches/batch1_sp_mte2_drain_8.log`） | M133 复核批（现 `evidence/logs/sp_mte2_drain_8.log`） |
|---|---|---|
| `mte2_drain_8` | `rc==0` **8/8** → 出现 **0/8**（未出现） | `rc==0` **2/4** → 出现 **2/4**（run3 `obs[6]=1152`、run4 `obs[6]=128`） |

⇒ 同一变体、同一 `N`，两批分别读成「未出现」与「2/4 出现」，**签名同为「退回上一次 op」**。
引用时请写「间歇：M126 批 8/8 未出现、M133 复核批 2/4 出现」，不要只引一批、也不要写成恒定。
（M126 原始日志已按既有 batch 惯例从 `main` 取回归档到 `batches/`，该行恢复可复算。）

### 3.3 回答（对人类原话）

> `get buf 0 / MTE2 / rls buf 0 / get buf 0 / MTE2 … 是否会等待前面的 MTE2 结束后再启动？`

**实测：在 `PIPE_MTE2` 上，第二次 `get` 不保证等到第一次 MTE2 结束。** 同一 id 背靠背两轮 MTE2 写同一
UB buffer，完成后内容可能退回**上一轮**的值；逐轮 release 用 drain 模式（`RlsBufInternal<pipe,true>`）
与 mode0 都出现。**正确的同 pipe 内部同步是 `PipeBarrier<PIPE_MTE2>`（加在两次同 buffer 拷贝之间）** ——
本探针在 `N=8/16` 各 8 次运行里，加 PB 的那两档未再出现该现象（与 `docs/06` §4、`docs/05` §6.2 的结论同向）。
> ⚠ **M183 需重读**（`probe_ub_bufid_tail/README.md:209`–`:212`）：本段涉及当时 `RlsBufInternal<pipe,true>` 的逐轮 release；全项目已统一 `false`（= CANN `ASC_LOCK_BLOCK` 默认、阻塞），此处结论的模式依赖已变，**需重读**（数值未改）。

`PIPE_V` 在本档（≤16 次 × 2 AIV × 8 次运行）未观察到同类现象；这**不等于** V 侧不需要规则，只说明
「V 侧复用同一 UB buffer 时，本档读数未咬到」。

> 对 `docs/05` §6.2 的补充边界：该表把结论写成「同 pipe MTE2 背靠背复用同一 buffer … `PipeBarrier<PIPE_MTE2>`
> 是唯一有效指令」。本探针复现了前半句的**存在性**（失败签名 = 退回上一轮），并复现了
> 「两次 op 之间加 PB ⇒ 本档未再出现」。**但**：本档在 M126 批里还显示 `mte2_imm_8`/`mte2_drain_8`（N=8）各 8/8 次
> 未出现（M133 复核批把 `mte2_drain_8` 读成 **2/4 出现**，批次对账见 §3.2.1），
> 即该现象是**间歇的、随迭代数增大更易命中**（`N=16`：第 1 批 drain `rc==0` 5/8 ⇒ 出现 **3/8**；第 2 批出现
> **2/8**；复审复跑出现 **1/8** —— 三批并列见 §3.2.1）——引用时请带上 N 与运行次数，不要写成恒定，也不要只引单一批次。

### 3.4 与官方文档的对照（来自 M127 零设备调研，转引，未由本探针独立复读官方文件）

M127（`agent-idiomsurvey`，官方 idiom 调研）发来几条与本探针直接相关的官方落点；本探针的真机读数与它们**同向**：

1. **同一 BufferID 跨 V/MTE3 交接有官方正面用法**（对应本探针 `base_*`/`multi3_*` 全绿）：
   `asc-devkit/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md:189-190`
   的同步关系表逐字写「输出 `outputMutexId` 在 PIPE_V（写）与 PIPE_MTE3（搬）两 pipe 交接，下一次 Vector 获取同一
   mutex_id 时等待」；framework impl `asc-devkit/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h:832-835`
   同一 bufId 0 在 `GetBuffImpl<PIPE_V,true>` 与 `GetBuffImpl<PIPE_MTE3,false>` 两侧获取/释放。
2. **同 pipe `Acq→op→Rls→Acq→op` 官方明确说不保序**（对应本探针 §3.2 的 MTE2 失败签名）：
   `asc-devkit/.../intra_core_sync/Lock.md:93-95`、`.../c_api/sync/intra_core_sync/asc_lock.md:101-103`，
   要求改用 `PipeBarrier` / `asc_sync_pipe` —— 与本仓 `docs/06-m0-bringup.md:43-49` 的 60/60 vs 0/60 同向。
3. **可做 A/B 对照的官方模板**：`ops-transformer/attention/common/op_kernel/init_output.h:44-83` 把同一条
   UB→GM 收尾**同时**写成 `Mutex::Lock/Unlock` 支与 `SetFlag/WaitFlag` 支（`ENABLE_LOCK` 切换）。
4. **口径差提醒（M127 提出，本探针未测）**：公开 `Mutex::Lock/Unlock` 把 mode 硬编码为 0
   （`include/basic_api/kernel_common.h:149,161` 的 `GetBufInternal/RlsBufInternal<pipe,0>`）；
   官方 `asc_lock.md:54` / `asc_unlock.md:54` 的 `asc_mutex_execute_mode`（BLOCK/NON_BLOCK）语义描述
   与「mode=true=drain」并不直觉一致 ⇒ **本次探针只用了仓库口径的 Internal 原语**
   （`GetBufInternal<p,false>` / `RlsBufInternal<p,true|false>`），未对 `asc_lock/asc_unlock` 取读数；
   若要按公开 `Mutex::Lock/Unlock` 的 mode 语义下结论，需另开一档。（本条为 M126 的**未做项**。）
   → **M133 已补**：`asc_lock/asc_unlock(mode)` 与 `RlsBufInternal<pipe,true/false>` 的四档真机对照读数见 §8。

---

## 4. 对 M117 的含义（结论 + 写窄的未定界）

**已由设备读数支持（可引用）**：

1. 「`Acq<PIPE_V>(id) → 写 UB → Rls<PIPE_V>(id) → Acq<PIPE_MTE3>(id) → DataCopy(UB→GM) → Rls<PIPE_MTE3>(id)`」
   这一交接形态，在 `N=1..64`、每工作项 1 次或 3 次交接、四种 DataCopy 形态下**均未挂死**，内容逐元素正确。
2. 该形态的**内容正确性依赖 V 与 MTE3 共用同一个 BufferID**（两个独立 id 时变红）。**限定（复审点名）**：
   `c_two_ids` 两侧都做了完整的 `Acq/Rls`、只是用了两个不同 mutex id，构造上等价于「无 V→MTE3 依赖」；
   它的正确含义是「**本 idiom 里跨 pipe 交接必须共用一个 id**」，**不等价于**「硬件禁止用两个 id +
   其它显式跨 pipe 同步」——后者本探针未测。
3. 去掉 V 侧 release 会让下一次 `Acq<PIPE_V>` 阻塞不返回（可控挂死，3/3 超时 124）。

**未定位（不得写成已定位）**：

- 本探针**没有复现** M117 `m=64` 的挂死，也**没有**任何一档把挂死点搬到「第 3 个工作项」。
  因此：M117 的挂死不满足「由这个 AIV 内交接形态单独导致」这一假设；**其根因不在本次定死的范围内**。
- 未被本探针覆盖、因此**不能排除**的方向（只是下一步候选，均未验证）：
  ① AIC 侧把 S/PV 经 Fixpipe 直写 AIV UB 的**跨核数据面**；② `CC_S_RDY/CC_S_FREE/CC_PV_RDY/CC_PV_FREE/CC_P0/CC_P1`
  这套 CrossCore flagId 的记账在 `wi≥2` 时是否两侧守恒；③ AIV 收尾与 AIC 的 Fixpipe 之间是否存在
  「AIC 等 AIV / AIV 等 AIC」的环。**这三条都属猜测**，需要单独的最小跨核探针去定，不在本 mission 的读数里。

**给 M117 的可照抄片段**：本 mission 的结论**不是**「该交接形态不成立」，因此**不推翻** M117 收尾里
现用的握手次序；可以直接照抄的**唯一新增规则**是 §3 的同 pipe 规则 ——
**同一 pipe 里背靠背复用同一 UB buffer 时，两次 op 之间加 `PipeBarrier<该 pipe>`**；
交接本身仍按 `Acq → 阻塞释放 RlsBufInternal<p,false> → 对侧 Acq` 走同一 id（release 一律 false=BLOCK；本探针 §2 的 `base_*` 即是可照抄的最小片段，
`probe_ub_bufid_tail.asc:151`–`:183`）。

---

## 5. 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd probe_ub_bufid_tail
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
bash run_probe.sh all                 # 全部变体（脚本自带 flock + npu-smi + 每条 timeout）
REPS=8 bash run_probe.sh sp '.'       # 同 pipe 轴，8 次/档
bash run_probe.sh tail base_N3        # 单变体
bash tools/summarize.sh               # 从日志重算矩阵
```

M133 两档的复现命令（**每档各自进一次锁**，与塔的「一次 flock 只跑一条短命令」一致）：

```bash
# 变体①：去掉 MTE3 侧 release（TO=20 让挂死档及时收）
REPS=3 TO=20 bash run_probe.sh tail '^neg_skipse3$'       # 期望 rc=124 x3
REPS=3 TO=20 bash run_probe.sh tail '^neg_skipse3_2id$'   # 期望 rc=124 x3
REPS=3 bash run_probe.sh tail '^neg_skipse3_N1$'          # 对照：N=1 无第二次 acq ⇒ rc=0 x3
REPS=3 bash run_probe.sh tail '^ctrl_e3rls$'              # 对照：保留 release ⇒ rc=0 x3
# 变体②：release 原语四档（V 侧 / MTE3 侧各一组）
REPS=4 bash run_probe.sh tail '^relv_imm$'
REPS=4 bash run_probe.sh tail '^relv_unlockB$'
REPS=4 bash run_probe.sh tail '^relv_unlockNB$'
REPS=4 bash run_probe.sh tail '^rel_neg_twoids$'          # 负向：期望变红
# 变体② 第二构造（真实 MTE2 DMA 生产者）
REPS=4 bash run_probe.sh sp '^mte2_tail_imm$'
REPS=4 bash run_probe.sh sp '^mte2_tail_unlockNB$'
```

**设备槽纪律（M133 轮记录）**：每个变体**各起一次** `flock -w 300 /tmp/npu0.lock`（脚本自身 exec）；
进锁后先 `npu-smi` 复查（各批各档都打了 `npu-smi` 头）；`timeout` 在锁内（绿档 30 s、挂死档 20 s）；
写文件前查 `df -h /`（本轮 430 G 可用 / 96% 已用，未触满）。驱动输出（含每档进锁时刻、`npu-smi` 头、
逐档 `rc==0` 汇总）归档在 `evidence/logs/m133_driver_batch1.txt`、`m133_driver_batch2.txt`；
`evidence/logs/` 里本轮涉及的变体，其逐变体日志以**第 2 批**（与 `evidence/commands.txt` 的源码 sha
`e6b70d8d…` 对应）覆盖为准；M126 原有、本轮未重跑的变体日志维持原样。

**被覆写的 M126 原始日志已按 batch 惯例留底（M133 r1 复审 P2-1）**：本轮复跑覆写了三个 M126 逐变体日志，
它们从 `main` 取回归档到 `evidence/logs/batches/`（命名沿用既有 `batch1_*` 惯例）：

| 变体 | M126 原始（留底路径） | 本分支覆写后 |
|---|---|---|
| `mte2_drain_8` | `batches/batch1_sp_mte2_drain_8.log`（`rc==0` 8/8） | `sp_mte2_drain_8.log`（`rc==0` 2/4） |
| `neg_skipvrel` | `batches/batch1_tail_neg_skipvrel.log`（`rc==0` 0/3，HANG 3/3） | `tail_neg_skipvrel.log`（`rc==0` 0/2，HANG 2/2） |
| `base_N8` | `batches/batch1_tail_base_N8.log`（`rc==0` 3/3） | `tail_base_N8.log`（`rc==0` 3/3；仅因 `drainV/drainE3`→`apiV/apiE3` 改字段名而覆写） |

⇒ `mte2_drain_8` 与 `neg_skipvrel` 的 M126 读数恢复可复算；`base_N8` 两批读数一致（3/3 绿），差别只在
`[probe]` 行的字段名。`mte2_drain_8` 的批次对账写进 §3.2/§3.2.1。

**`trace=1` 取证批（M133 r1 复审 P2-2）**：为让 §7.2「设备 `printf` 不随 `timeout` 落盘」这条方法学论断可复算，
另起两档 `trace=1` 运行并单独归档（**每档各自进一次锁**、`timeout` 在锁内、进锁先 `npu-smi`）：

```bash
# 非挂死档（锁内约 10 s）
flock -w 300 /tmp/npu0.lock bash -c 'cd <探针目录>; source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  npu-smi info | sed -n "5,7p"; \
  for i in 1 2; do timeout 30 ./build/probe_ub_bufid_tail tail ctrl_e3rls $i out_tmp --trace; echo exit=$?; done' \
  > evidence/logs/batches/m133trace_ctrl_e3rls.log 2>&1

# 挂死档（锁内约 40 s）
flock -w 300 /tmp/npu0.lock bash -c 'cd <探针目录>; source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
  npu-smi info | sed -n "5,7p"; \
  for i in 1 2; do timeout 20 ./build/probe_ub_bufid_tail tail neg_skipse3 $i out_tmp --trace; echo exit=$?; done' \
  > evidence/logs/batches/m133trace_neg_skipse3.log 2>&1
```

产物：`batches/m133trace_ctrl_e3rls.log`（2 次 `rc==0`，`grep -c '\[TAIL\]'` = `128`）、
`batches/m133trace_neg_skipse3.log`（2 次 `exit=124`，同一条 `grep -c '\[TAIL\]'` 在该份上 = `0`）。
两份日志的锁内 `npu-smi` 头与 `lock_acquired=` 时刻都在文件内。

**过程自陈（纪律偏差，复审记录 2 点名补入）**：首次取数时「每档一次 flock」实际持锁约 **2–3 min**
（sp 批 3 变体 ×8 次 ≈ 175 s、tail 批 9 变体 ×3 次 ≈ 300 s），**长于塔的「单次进锁 ≤120 s」目标**。
本轮复审后的第 2 批复跑已改为**每变体各自进一次锁**
（`REPS=8 bash run_probe.sh sp 'mte2_drain_n16'` ≈ 58 s/次），与该目标一致。
**M133 轮**继续沿用「一变体一锁」，单次持锁按档位为 **约 10–60 s**（挂死档 = `reps × TO` = 3 × 20 s，
绿档 ≈ 4 × 单次运行）；可逐档从两份 `m133_driver_batch*.txt` 里相邻两行的时间戳相减核对。

## 6. 完成度与未完成项（逐条）

- [x] 最小复现工程（独立 CMake + 含 `main()` 的 `.asc`，不碰既有 CMakeLists）
- [x] `N=3 / 8 / 16` 逐档读数（另加 N=1/64），**未复现挂死**；每档有能变红的负向对照
- [x] 变量矩阵 (a) drain 形态 (b) DataCopy 行数/stride (c) 同 id vs 两个 id (d) 同 pipe 保序 (e) 前置/N 轴
- [x] 每变体设备读数（EXIT 码 + 内容读数 + 锁内 `npu-smi`）归档 `evidence/`
- [ ] **未做**：跨核（AIC Fixpipe → AIV UB + CrossCore flagId）一侧的最小复现 —— 本 mission 未覆盖，
      属 §4 列出的「下一步候选」，需另开探针
- [ ] **未做**：`PIPE_V` 同 buffer 复用在更大 N / 更多核下的是否也会乱序（本档未咬到，n 不足）
- [ ] **未做（复审记录 4 登记）**：mission (a) 的「去掉其中一个 release」只做了 **V 侧**（`neg_skipvrel`，
      可控挂死）；形参 `skipRlsE3` 存在但**无变体置位** ⇒ 「去掉 MTE3 侧 release」未测。
      → **M133 已补**：`neg_skipse3` / `neg_skipse3_2id` 等档已置位，读数见 §7。
- [ ] **未做（复审记录 3 登记）**：`asc_lock/asc_unlock` 公开 mode 语义 vs `RlsBufInternal<p,true/false>`
      未取读数（见 §3.4 第 4 条）。→ **M133 已补**：四档真机对照见 §8。
- [ ] **过程自陈（复审记录 2）**：设备槽纪律偏差见 §5 —— 首批每档持锁 2–3 min > 塔 120 s 目标；
      第 2 批已按「每变体各自进一次锁」复跑。

---

## 7. M133 变体① —— 去掉 MTE3 侧 release（`skipRlsE3` 首次置位）

被测形态与 §1.1 的 `probe_tail_kernel` 逐字同形，只把 `Rls<PIPE_MTE3>(idE3)` 这一句去掉
（`skipRlsE3=1`，形参在 M126 就存在、本轮首次有变体置位）。两档：`idSame=1`（V 与 MTE3 共用一个 id）、
`idSame=0`（V 用 `ID_A`、MTE3 用 `ID_B`）。命令见 §5。

### 7.1 读数

| 变体 | 构造 | 读数（各自进锁） | 说明 |
|---|---|---|---|
| `ctrl_e3rls` | 同形、**保留** MTE3 侧 release，`N=8` | rc==0 3/3，`match=32768/32768` | 有 release 的对照组 |
| `neg_skipse3` | 去掉 MTE3 侧 release，共用一个 id，`N=8` | **HANG 3/3（rc=124）** | 整框不返回 |
| `neg_skipse3_2id` | 去掉 MTE3 侧 release，V/E3 两个 id，`N=8` | **HANG 3/3（rc=124）** | 同上 |
| `neg_skipse3_N1` | 去掉 MTE3 侧 release，`N=1` | rc==0 3/3，`match=4096/4096` | 只有 1 次交接 ⇒ 没有「下一次 acquire」 |
| `neg_skipse3_2id_N1` | 两 id + 去掉 release，`N=1` | rc==0 2/3（run1 变红：`mismatched=1`、`zero=2048`） | 不挂；红的原因是两 id 无 V→MTE3 握手（与 §2.1 `c_two_ids` 同源） |
| `ctrl_e3rls_m3` | 保留 release + 每工作项 3 次交接（M117 形态），`N=1` | rc==0 3/3，`match=12288/12288` | 对照组 |
| `neg_skipse3_m3` | 去掉 MTE3 侧 release + 每工作项 3 次交接，`N=1` | **HANG 3/3（rc=124）** | 停在第 2 次交接 |
| `neg_skipvrel`（既有） | 去掉 V 侧 release | HANG 2/2（rc=124） | 挂死检测的既有对照，本轮复跑仍在（M126 批为 3/3，两批留底见 §5） |

### 7.2 阻塞点收到哪一次 acquire

设备 `printf` **不随 `timeout` 落盘** —— 本轮为这条方法学论断单独起了两档 `trace=1` 取证运行并归档
（各自进锁、`timeout` 在锁内、进锁先 `npu-smi`；命令见 §5）：

- **非挂死档** `evidence/logs/batches/m133trace_ctrl_e3rls.log`：2 次运行都 `rc==0`；同一份日志上
  `grep -c '\[TAIL\]'` 的输出是 `128`（每次交接的 `acqV`/`rlsV`/`acqE3`/`rlsE3` 四条）。
- **挂死档** `evidence/logs/batches/m133trace_neg_skipse3.log`：2 次运行都 `exit=124`；同一份日志上
  `grep -c '\[TAIL\]'` 的输出是 `0`，日志正文只有 host 的 `[probe]` 行与 `exit=124`
  （命令覆盖范围：该单份日志；对照份的输出见上一条）。

（日常读数批按 `trace=0` 跑，所以 `evidence/logs/tail_*.log` 里本来就不带 `[TAIL]` 行。）
所以「第几次 acq 卡住」给不出 trace 读数，只能靠逐档收窄：

1. `N=1` 两档（`neg_skipse3_N1`、`neg_skipse3_2id_N1`）都**返回**（前者 3/3 绿、后者 1/3 因两 id 无握手变红但不挂）
   ⇒ 挂死不出现在「缺 release 的那一次交接」内部，只出现在**紧随其后的那一次同 id acquire**。
2. 两 id 档里 `ID_B` 只出现在 MTE3 侧 ⇒ 能阻塞在 `ID_B` 上的只有**第 2 次交接的 `Acq<PIPE_MTE3>(ID_B)`**；
   该档 3/3 挂 ⇒ **「下一次同一 id 的 `Acq<PIPE_MTE3>` 会阻塞」在本档成立**。
3. 单 id 档里，程序序上下一次同 id acquire 是 `Acq<PIPE_V>(ID_A)`（第 2 次交接的第 1 条 acquire）；该档 3/3 挂
   ⇒ 读作「漏掉一次 release 后，该 id 的令牌未被交回，仍持有它的构造里后续同 id acquire 取不到」。
   两 id 档把这一点直接收到 `Acq<PIPE_MTE3>(ID_B)`（第 2 条），单 id 档则只能说到「共用的那个 id 未被交回」。

**形状结论（窄）**：本档合成出的形状是「**漏一次 release ⇒ 紧随其后的那一次同 id acquire 阻塞、整框不返回**」。
阻塞点必落在第一次漏 release 的**下一次** acquire 上；本探针没有把它推到更后面（结构上：过不去第一次
阻塞就发不出后续 acquire）。

### 7.3 与 M117 的边界（写窄，不得当根因）

- M117 的读数把阻塞点收到 AIV **第 9 次** `Acq<PIPE_MTE3>(B_PC)`；本档是**单个 AIC 的纯 AIV 循环**，
  不含 AIC / Fixpipe / CrossCore flagId，而且**是人为去掉一次 release 才挂的**（M117 代码里没有显式的 release 缺失）。
- ⇒ 本档**不能**写成 M117 挂死的根因。它支持的窄结论只有一句：**「release 泄漏 ⇒ 下一次同 id acquire 阻塞」
  这一形状在最小档里可合成**。
- 本档也**不支持**「第 9 次特殊」：它的阻塞点在第 2 次交接，与「第几次」这个数无关。

---

## 8. M133 变体② —— release 原语四档真机对照（跨 pipe 可见性轴）

问题：`RlsBufInternal<pipe,true/false>`（含本仓 `BufRelease<PIPE>`）与公开 `asc_lock/asc_unlock(mode)` 里，
**哪个 mode 实测才建立「前序排空后才释放」**。

### 8.1 先定「四档各落到底层哪个 intrinsic」（源码口径，非文档措辞推断）

- 本仓镜像：`asc-devkit/impl/c_api/reg_base_impl/sync_intf_impl.h:187-295` ——
  `asc_lock/asc_unlock(...,ASC_LOCK_BLOCK)` → `get_buf/rls_buf(...,false)`；`ASC_LOCK_NON_BLOCK` → `(...,true)`；
  两参重载默认 `ASC_LOCK_BLOCK`（`:240`、`:295`）。
- **本机构建实际用的已装 CANN 头**（不是上面那个镜像）：
  `/usr/local/Ascend/cann-9.1.0/asc/include/c_api/sync/sync.h:72-79`
  （`enum ascMutexExecuteMode { ASC_LOCK_BLOCK = 0, ASC_LOCK_NON_BLOCK = 1 }`；两参默认 `ASC_LOCK_BLOCK`）；
  `/usr/local/Ascend/cann-9.1.0/asc/impl/c_api/instr_impl/npu_arch_3510/sync_impl/asc_unlock_impl.h:30-33`
  逐字 `if constexpr ((mode) == ASC_LOCK_BLOCK) { rls_buf((pipe),(mutex_id),false); } else { rls_buf(...,true); }`；
  `asc_lock_impl.h:30-33` 同构走 `get_buf`。
- 本仓 `RlsBufInternal<pipe,mode>` → `rls_buf(pipe,bufId,mode)`（`asc-devkit/impl/basic_api/kernel_event.h:783-800`）。

⇒ 按底层映射：(a) `RlsBufInternal<pipe,false>` 与 (c) `asc_unlock(...,ASC_LOCK_BLOCK)` 落**同一 intrinsic 同参**；
(b) `RlsBufInternal<pipe,true>` 与 (d) `asc_unlock(...,ASC_LOCK_NON_BLOCK)` 落同一 intrinsic 同参。
即公开 `BLOCK/NON_BLOCK` 命名**没有**把底层 bool 反过来，它只是同一个 bool 的另一个名字。

> ⚠ **M183 需重读**：§8.2（`:455`–`:461`）、§8.3（`:473`–`:476`）、§8.4（`:488`）的 mode 档位读数取自 release 侧含 `true` 的时期；全项目已统一 `false`（= CANN `ASC_LOCK_BLOCK` 默认、阻塞）。§8.4 里「`RlsBufInternal<pipe,true>` = 本仓 `BufRelease<PIPE>`」的对应关系已不成立（`BufRelease` 现为 `false`）。结论所依赖的模式已被改动，**需重读**（读数数值未改）。

### 8.2 第一构造（收尾交接）逐档读数

构造与 §2 的收尾交接同形（`Acq<V> → VF 写 UB → Rls<V>(档) → Acq<MTE3> → DataCopy(UB→GM) → Rls<MTE3>(档)`），
`N=32`；每档 4 次独立进程、各自进锁。

| 档位 | V 侧 release | MTE3 侧 release | 读数 |
|---|---|---|---|
| `rel_base`（b） | `RlsBufInternal<V,true>` | `RlsBufInternal<MTE3,true>` | rc==0 4/4，`match=131072/131072` |
| `relv_imm`（a） | `RlsBufInternal<V,false>` | `true` | rc==0 4/4，`match=131072/131072` |
| `relv_unlockB`（c） | `asc_unlock(V,id,ASC_LOCK_BLOCK)` | `true` | rc==0 4/4，`match=131072/131072` |
| `relv_unlockNB`（d） | `asc_unlock(V,id,ASC_LOCK_NON_BLOCK)` | `true` | rc==0 4/4，`match=131072/131072` |
| `rele_imm`（a） | `true` | `RlsBufInternal<MTE3,false>` | rc==0 4/4，`match=131072/131072` |
| `rele_unlockB`（c） | `true` | `asc_unlock(MTE3,id,ASC_LOCK_BLOCK)` | rc==0 4/4，`match=131072/131072` |
| `rele_unlockNB`（d） | `true` | `asc_unlock(MTE3,id,ASC_LOCK_NON_BLOCK)` | rc==0 4/4，`match=131072/131072` |
| `rel_neg_twoids`（负向） | V/E3 两个 id（无握手） | — | rc==0 **0/4**；run1 `mismatched=61/64`、`match=6848` |
| `rel_neg_shift`（负向） | 写出 tag 偏 1 | — | rc==0 **0/4**；`mismatched=64`、`match=0`、`histOther=4096` |

### 8.3 第二构造（真实 MTE2 DMA 生产者）逐档读数

`probe_sp_kernel`（§1.1 ②）：同 pipe 连续 `Acq/DataCopy/Rls` ×8，收尾在生产者 pipe 上
`Acq → Rls(档) → Acq<MTE3>(同一 id)` → 搬出。收尾这次 release 的原语按档选；逐轮 release 固定 drain。
`N=8`，4 次独立进程。

| 档位 | 收尾 release | 读数 | 说明 |
|---|---|---|---|
| `mte2_drain_8`（基线） | `RlsBufInternal<MTE2,true>` | rc==0 **2/4**（M126 批同一变体为 8/8，见 §3.2.1）；run3 `obs[6]=1152`、run4 `obs[6]=128` | 基线自己就变红 |
| `mte2_tail_imm` | `RlsBufInternal<MTE2,false>` | rc==0 4/4 | — |
| `mte2_tail_unlockB` | `asc_unlock(MTE2,id,ASC_LOCK_BLOCK)` | rc==0 3/4；run1 `obs[6]=128` | — |
| `mte2_tail_unlockNB` | `asc_unlock(MTE2,id,ASC_LOCK_NON_BLOCK)` | rc==0 3/4；run4 `obs[6]=1024` | — |
| `v_tail_imm` | `RlsBufInternal<V,false>` | rc==0 4/4 | — |

（`obs` 下标为 0-based op 序号，`obs[6]` = 第 7 次 op 的 tag。）

### 8.4 回答（写成「哪个 mode 实测对应哪种行为」）

- **第一构造**（负向对照能变红）里，四档读数**是同一结果**：`N=32` 各 4 次运行的内容都逐元素等于
  各自 tile 的 tag。本轮**没有**读到「只有某个 mode 才建立跨 pipe 可见性」的情形 —— 在这个
  「写 UB → rls → 另一 pipe 读」的形状里，`false` 与 `true`（连同它们各自的 C API 名字）在**内容可见性**
  上没有可判读的差异。
- 合上 §8.1 的源码映射 ⇒ **实测对应关系**：`ASC_LOCK_BLOCK` ↔ 底层 `false` ↔ `RlsBufInternal<pipe,false>`；
  `ASC_LOCK_NON_BLOCK` ↔ 底层 `true` ↔ `RlsBufInternal<pipe,true>`（= 本仓 `BufRelease<PIPE>`）。
  这是一条**改名**关系、不是反转关系；`docs/06 §4.2` 登记的「两页措辞不一致」是**两页对同一条 mode 的
  行为描述**不一致（`asc_lock.md` 讲 lock 侧是否阻塞、`asc_unlock.md` 讲 unlock 侧等什么），
  **不是**「true 其实是 no-drain」的证据。
- **第二构造的数不作 mode 差异的证据**：基线 `mte2_drain_8` 自己就 2/4 变红（同一变体的 M126 批为 8/8，
  见 §3.2.1），且各变红档签名一致（`obs[6]` = 第 7 次 op），即 §3 已知的**同 pipe MTE2 间歇乱序**
  （§3.2.1 记录该现象在各批间 0/8 ~ 3/8 波动）。
  这一构造由同 pipe 现象主导，`mte2_tail_imm` 的 4/4 也在这个噪声带内。
- **写窄**：本节结论只覆盖两个构造（`PIPE_V→PIPE_MTE3` 与 `PIPE_MTE2→PIPE_MTE3` 的收尾交接，
  1 AIC = 2 AIV，`N ≤ 32`）。**未测**：同 pipe 复用（§3 那条，与 mode 无关）、连续多次同 id unlock 的排队行为、
  `PIPE_S`/`PIPE_FIX` 与 AIC 侧、`asc_lock` 的 mode。**不得**据此推断未测的 pipe/mode 组合。

---

## 9. M133 完成度、纪律自陈与未完成项

- [x] 变体①：`skipRlsE3` 置位 + 有 release 对照 + `N=1` / 两 id / `handoffs=3` 的收窄档；读数见 §7
- [x] 变体②：四档 release 原语 × 两个构造；负向对照 `rel_neg_twoids`、`rel_neg_shift` 变红；读数见 §8
- [x] 两个变体各自都有能变红的负向对照（未用「跑了没挂」代替判读）
- [x] 落 GM 走 DMA；BufferID / 地址静态自管（本轮未引入 `TPipe`/`TBuf`/`TQue`/`AllocTensor`）
- [x] 设备纪律：一变体一锁、进锁先 `npu-smi`、`timeout` 在锁内、写文件前查 `df -h /`（记录见 §5）
- [x] **r1 复审 P2-1**：M126 被覆写的三个原始日志按 batch 惯例留底（`batches/batch1_sp_mte2_drain_8.log`、
      `batch1_tail_neg_skipvrel.log`、`batch1_tail_base_N8.log`）；§3.2 / §3.2.1 的 `mte2_drain_8` 行改成
      「M126 批 8/8；M133 复核批 2/4」两批并列，并与 §8.3 / §8.4 互相一致（见 §5 的对账表）
- [x] **r1 复审 P2-2**：补跑并归档 `trace=1` 取证批（挂死档 + 对照档各一，各自进锁），
      使 §7.2 的方法学论断可复算：`batches/m133trace_ctrl_e3rls.log`（`[TAIL]` 128 条）、
      `batches/m133trace_neg_skipse3.log`（同一条 `grep -c '\[TAIL\]'` 在该份上为 `0`；命令与范围见 §5）
- [ ] **未做**：把阻塞点推到第一次漏 release 之后的更远处（结构上过不去第一次阻塞，未构造出这种档）
- [ ] **未做**：跨核（AIC Fixpipe → AIV UB + CrossCore flagId）一侧 —— 承 §4，仍未覆盖
- [ ] **未做**：第二构造（§8.3）被同 pipe MTE2 乱序主导，未取得能把 mode 差异读出来的档
- [ ] **过程自陈**：上面两批读数 + `trace=1` 取证批都由与 `evidence/commands.txt` 里同一份源码
      （`probe_ub_bufid_tail.asc` sha256 `e6b70d8d…`）构建的二进制产生；第 2 批 23 档、第 1 批 14 档、
      取证批 2 档都取到了读数（没有档因等锁而未取到；驱动输出归档在 `evidence/logs/m133_driver_batch1.txt`
      / `batch2.txt`，取证批各自进锁的时刻写在其日志内的 `lock_acquired=` 行）。
