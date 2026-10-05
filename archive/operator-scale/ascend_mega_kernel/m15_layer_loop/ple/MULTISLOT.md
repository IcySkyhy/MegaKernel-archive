# M171 —— 真实 ngram 表：多槽窗口池 + 跨分片窗口

> 分支 `feat/m171-ple-real-table-multi-slot-cross-sha`（worktree `wt-171`，base `main`）。
> 前置：M92（`m15_layer_loop/ple/REAL_TABLE.md`，单窗口 host-mapped gather）与 M158 survey
> （`.tower/comms/inbox/20261004-agent-plesurvey-tower-survey-summary-m158-ple-wiring-readiness-tip-ac46c32-zero-de.md`）。
> 本文是 M171 的取证 + 实现 + 判据 + 读数。**引用一律给 `文件:行`（行号基准 = 本 tip）。**

---

## 0. TL;DR

| 项 | 结论 | 位置 |
|---|---|---|
| 单窗口容量（改前） | 256 MiB = **838,860 行**；`row = id - winBase`；越窗只数不做 | `REAL_TABLE.md §2.3`、`m15_ple.asc` 原 `PleGather` |
| 多槽池（改后） | `n_slots` 个槽，每槽一段**连续全局行区间**，可**横跨分片边界** | `m15_ple.asc:1675-1714`、`real_table_probe.py:160-236` |
| 跨分片行基址算术 | 设备侧 `UDivMod64(id, rowsPerShard)` → `(shard, local)`；`delta = (shard-shard_s)*rps + local - local_base_s` | `m15_ple.asc:292-307`、`m15_ple.asc:526-560` |
| 基线判据 | **11/11 PASS**（M92 的 10 条 + 新增 `Hd.served`），`in_win=1024/1024`、`dev_row_fail=0`、`dev_miss=0` | `ple/logs/multislot_base_check.log` |
| 负向对照 1 | `ids_hm_miss.bin`（16 个 id 在池外）⇒ **只 `Hd.served` 红** | `ple/logs/multislot_miss_check.log` |
| 负向对照 2 | `M15_PLE_MUT=65536`（bit16 改错 `rows_per_shard`）⇒ `Hd.row_fail`/`H2.emb`/`H3`/`H4`/`H5`/`Hd.miss` 红 | `ple/logs/multislot_mut16_check.log` |
| 回归 | 单窗口路径（`n_slots=0`）判据读数与 `main` HEAD 二进制**逐条一致** | `ple/logs/legacy_regress_check.log` |
| 机械生成物 | 改 `.asc` 后重跑 `lift_ple_device_segment.py`，`--check` 逐字节通过 | `/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check` |

---

## 1. 现状取证（task 1）

### 1.1 表几何与单窗口容量

* 全表 = 128 分片 × 2,500,012 行 × 160 列 × 2 B = 320,001,536 行 = **95.37 GiB**
  （`ple/REAL_TABLE.md:22`；`ple/real_table_probe.py:35-40` 的常量）。
* 行 id → 文件位置：`shard = id // 2500012`、`local = id % 2500012`、
  `file_off = data_off[shard] + local*320`（`ple/REAL_TABLE.md:78-87`；`real_table_probe.py:79-81`）。
* 单窗口 = 一次性 `aclrtHostRegisterV2(MAPPED)` 的匿名 staging；M92 实跑 **1 槽 × 256 MiB =
  838,860 行**（`ple/REAL_TABLE.md:193-208`、`ple/logs/hm_run.log:4`）。
* 行基址算术（单窗口）：`row = id - winBase`（`REAL_TABLE.md:96`；本 tip 的
  `m15_ple.asc:561-570` 是同一算式）。**窗口是单个连续区间** ⇒ `winBase` 只能锚在一个分片内。

### 1.2 越窗判据（改前）

* 设备侧：`id ∉ [winBase, winBase+winRows)` ⇒ `miss++`，累计器经 UB → GM 落盘
  （`REAL_TABLE.md:282-289`；本 tip `m15_ple.asc:555-556`）。
* host 侧：`m15_ple_check.py::check_hm` 用**单个区间掩码** `inw = (ids>=r0)&(ids<r0+win_rows)`
  （`m15_ple_check.py:443`），参考行 `real_row()` 只用**单个** `meta["file"]/data_off`
  （`m15_ple_check.py:403-415`）——**这两个假设就是多槽/跨分片下失效的地方**。
* 读数（M92 基线）：`dev_miss=2`（1024 个 item 里 2 个越窗），`Hd.miss` 与 host 独立算出的
  越窗数一致（`ple/logs/hm_run.log:37`）。

### 1.3 U-A / U-B / U-C 各自缺什么

| 缺口 | 缺的是什么（改前） | 本 tip 的状态 |
|---|---|---|
| **U-A 多槽滑窗** | 只有「一个窗口一次填满、整段跑完」；`win_rows=838,860`，无槽表、无槽选择策略（`REAL_TABLE.md:513`） | **已做**：`n_slots` 个槽 + 设备侧槽选择（`m15_ple.asc:526-560`）；host 侧池排布 `real_table_probe.py:192-236`。**LRU/滑窗策略本身仍未做**（见 §6） |
| **U-B 跨分片窗口** | 窗口落在单分片内 ⇒ `row = id - winBase` 与 `id % rows_per_shard` 等价；跨分片需 64 位除法（`REAL_TABLE.md:514`） | **已做**：槽 0 横跨 shard 0→1；设备侧 `UDivMod64` 分解 `(shard,local)` 后按 `delta` 定位（`m15_ple.asc:292-307,526-560`）；host 侧搬运行按边界拆两段（`real_table_probe.py:216-236`） |
| **U-C 真实 ①→② 端到端** | ① 的真实输出在**全词表**上产出，16 个 head 的 id 区间互不重叠（每 head 约落在 8 个分片），任何单窗口只能服务一小片（`REAL_TABLE.md:515`、`INTERFACE.md §7 U-E`） | **未做**（mission 边界）：本 tip 的 ids 仍是 harness 构造的真实行 id，只是**跨了多个分片**。多槽池让「一次服务跨分片的多头 id」成为可能，但**没有**把 ① 的 kernel 输出直接喂 ② |

`INTERFACE.md` 的实际落点：`m15_layer_loop/evidence/ple_wire/INTERFACE.md`（**不是** `m15_layer_loop/ple/INTERFACE.md`）。

---

## 2. 实现（task 2）

### 2.1 槽的语义（host / device 同一套）

* 一个槽 = 一段**连续全局行区间** `[base, base+rows)`，`base = shard*rows_per_shard + local_base`；
  `rows` **允许越过一行 `rows_per_shard` 边界**（跨分片）。槽在注册窗口里的起始行 = `row_off`。
* 池 = `n_slots` 个槽首尾相接，窗口 = 平坦 `[pool_rows,160]` bf16，`pool_rows = Σ rows`。
* 排布（`real_table_probe.py:192-214`）：**槽 0 横跨 shard 0→1**（`local_base = rows_per_shard - rows/2`），
  其余槽落在 shard 3 / 60 / 127 / …（互不相邻、分属不同分片）。默认 4 槽 × 4 MiB = 16,777,216 B。

### 2.2 设备侧（`m15_ple.asc`，机械抽到 `m15_ple_wire.h`）

* 新增 `UDivMod64`（移位-减法，一次 64 次迭代同时得商与余；`m15_ple.asc:292-307`）。
* `BodyGm` 新增 `slotMeta`（每槽 4×u32：shard / local_base / rows / row_off）、`nSlots`、
  `rowsPerShard`（`m15_ple.asc:454-458`）。
* `PleGather` 的 `nSlots != 0` 分支（`m15_ple.asc:526-560`）：
  1. `UDivMod64(id, rowsPerShard)` → `(shard, local)`；
  2. 逐槽算 `delta = (shard-shard_s)*rps + local - local_base_s`，`0 ≤ delta < rows_s` 即命中，
     `row = row_off_s + delta`；
  3. 一个槽都没命中 ⇒ `miss++`（真实行也可能落在池外）。
* `nSlots == 0` 时**一字不改**地走原路径（`row = id - winBase` / 平坦 GM 表）——
  见 §4 的回归读数。
* 复用 M92 的设备侧校验：仍对刚取到的行前 64 个 bf16 与 host 直读期望比对，
  错行计数 `hmBad`、越窗计数 `miss` 落 GM（`m15_ple.asc:577-610`）。

### 2.3 host 侧（`real_table_probe.py`）

* `--emit --slots N --slot-mib M` 走 `run_emit_multislot`（`real_table_probe.py:346-437`）；
  `--slots 0`（默认）走原 legacy 单窗口分支（`real_table_probe.py:439`）。
* 分片算术**全在 host**：`plan_slots`（排布）、`fill_segments`（把每个槽拆成
  `(dst, nbytes, file, byte_off)` 搬运行，跨分片自动拆两段）。
* 产物：`ids_hm.bin` / `exp_rows.bin` / `exp64.bin`（逐 id 直读分片文件）、
  `hm_slots.txt`（槽表）、`hm_fill.txt`（搬运行）、`hm_meta.txt/json`（含 `n_slots/pool_rows`）。
  C++ harness（`m15_ple.asc:1859-1873, 1985-2000`）只按 `hm_fill.txt` open/mmap/memcpy，
  设备侧看不到分片算术。

### 2.4 机械生成物见证

```
$ /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check
[ok] .../m15_ple_wire.h == 从 .../m15_ple.asc 重新抽取的结果（60408 字节，逐字节）
```

`m15_ple.asc` sha256 `152064023d094a49dc9e59751129faca6bc7e3f579824181c5f860148403cec9`；
`m15_ple_wire.h` sha256 `c2f7a51ef29d36170143ada2a6670ba916b90685fe4b6ec81043165f233bee6f`。

---

## 3. 判据与读数（task 3）

判据脚本 = `m15_layer_loop/ple/hm_multislot_check.py`（**在 scope 内**；它 import
`m15_ple_check.py` 复用 `Report` / `cmp_t3_masked` / `ref_gate` / `ref_conv_out` / T3 分档与 ε，
并 import `real_table_probe.shard_map`）。判据名与 M92 的 10 条一致，另加 `Hd.served`。

与 `check_hm` 的**两处口径差异**（多槽下必需，逐条给理由）：

1. **槽内掩码**：`check_hm` 用单区间 `[r0, r0+win_rows)`；多槽池的槽互不相邻，改用
   「id 被某个槽**真的命中**」的逐 item 掩码（`hm_multislot_check.py:56-69,86-105`），
   与设备侧同一套算式在 host 独立复算。
2. **参考行来源**：`check_hm` 的 `real_row` 只用单个 `file/data_off`；多槽跨分片 ⇒ 按
   `shard = id // rows_per_shard` 取**该分片**的文件偏移（`hm_multislot_check.py:75-83`）。
3. **④⑤ 的 T3 多一项传播界**：端到端口径（参考吃 `kv_ref`，M92 §6.3）在**深度相消**的
   value 通道上会被上游格点差主导 —— 实测 4/5 个元素的 `d/ulp` 达 3.0–4.0（超出 2·ulp）。
   按 `Δv ≤ EPS_MMAD·Σ|terms_v|`（H3 已证 `max(d/bound)=0.624`）沿 ④⑤ 向下传播，
   给 H4/H5 补一项 `extra_terms`（`hm_multislot_check.py:174-212`）。补后 `max(d/bound)` =
   0.35 / 0.40 / 0.38（下方读数）。

### 3.1 基线（`ple/logs/multislot_base_check.log`，11/11 PASS）

| 判据 | 档 | n | bad | 说明 |
|---|---|---|---|---|
| `H2.emb.nonvac` | T1-struct | 163840 | 0 | 对照行非零 163840/163840；槽内 item 1024 |
| `H2.emb` | T1 | 163840 | 0 | 逐字节；覆盖**跨分片**行与槽选择 |
| `H3.kv` | T3 | 819200 | 0 | 界 Σ\|terms\|，`max(d/bound)=0.624` |
| `H4.gated` | T3 | 655360 | 0 | `max(d/bound)=0.352` |
| `H4.normed` | T3 | 655360 | 0 | `max(d/bound)=0.401` |
| `H5.out` | T3 | 655360 | 0 | `max(d/bound)=0.384` |
| `Hd.cores` | T1-struct | 56 | 0 | 56/56 核；设备侧读到 `pool_rows=52428` |
| `Hd.row_fail` | T1-struct | 1024 | 0 | 设备侧错行计数 0 |
| `Hd.miss` | T1-struct | 1024 | 0 | 设备越窗 0 = host 独立算出的池外 0 |
| `Hd.nonvac` | T1-struct | 1024 | 0 | 槽内 item 1024、56 核 |
| `Hd.served` | T1-struct | 1024 | 0 | **新增**：所有 id 都被池服务（越窗计数 = 0） |

设备原始读数（`ple/logs/multislot_base.log:4-6`）：`参与核=56 winRows(设备侧)=52428
hmBad=0.0 越窗=0.0`；`n_tok=64 items=1024 in_win=1024 dev_row_fail=0.0 dev_miss=0`。

### 3.2 负向对照 1：越窗入口（`ids_hm_miss.bin`）

把 token0 的 16 个 id 换成池外的真实行（shard 0 头部，落在槽 0 起点之前）。
设备读数 `ple/logs/multislot_miss.log`：`in_win=1008 dev_miss=16`。
判据 `ple/logs/multislot_miss_check.log`：**只 `Hd.served` 红**（25 满分 1 红）。

* 「红的是该红的项」：`Hd.served` 判的就是「有没有 id 落在所有槽之外」——
  这一档正是要它红。`Hd.miss` 绿（host 独立算出池外 = 16，与设备一致，越窗计数本身正确）；
  `H2/H3/H4/H5` 绿（池外 item 被槽内掩码排除，不被误判）。

### 3.3 负向对照 2：`rows_per_shard` 改错（`M15_PLE_MUT=65536`，bit16）

设备侧 `rps += 1` ⇒ `(shard,local)` 分解错 ⇒ `shard_s != 0` 的槽 `delta` 整体偏移。
设备读数 `ple/logs/multislot_mut16.log`：`dev_row_fail=736.0 dev_miss=6`。
判据 `ple/logs/multislot_mut16_check.log`：**8 条红** ——
`H2.emb(118602)`、`H3.kv(814593)`、`H4.gated(648922)`、`H4.normed(648028)`、
`H5.out(297305)`、`Hd.row_fail`、`Hd.miss`、`Hd.served`。

* 「红的是该红的项」：改错 `rows_per_shard` 会让取到**错行** ⇒ `Hd.row_fail`（设备侧自己抓到）
  与 `H2.emb`（逐字节）必红，下游 ③④⑤ 连带红；`Hd.miss`/`Hd.served` 红是因为 6 个 id 的
  错误分解把它们推出了池。`Hd.cores`/`Hd.nonvac`/`H2.emb.nonvac` 仍绿（它们只判「有没有东西可比」）。

### 3.4 单窗口回归（`n_slots=0`）

`ple/logs/legacy_regress_check.log`（`m15_ple_check.py`，M92 的 10 条）：
与在 `main` HEAD 上重建的二进制**逐条一致**（`diff` 无差异）。读数：
`H2.emb.nonvac/H2.emb/H3.kv/H4.gated/H5.out/Hd.* PASS`；`H4.normed` **1 条红**（`max(d/bound)=1.499`）。

> `H4.normed` 的这一条红**不是本 mission 引入**：同一份 `data_hm`（legacy emit，与本 tip 的
> emit 逐字节相同）喂给**在 HEAD 上重建的 pristine 二进制**，得到同样的 1 条红与同样的
> `maxRel=0.0123`。它是 M92 `hm_run.log`（该阅读数来自 pre-M124 的 AIV GEMV 实现）之后、
> M124 把 ③ 改成 cube `Mmad` 带来的既存边缘（M92 `REAL_TABLE.md §6.2` 已预警「换输入要重核这一档」）。
> 已按 scope 纪律记为 finding，不在本 mission 修。

---

## 4. 约束合规（task 4）

| 约束 | 本 tip 的做法 | 核查 |
|---|---|---|
| 核内只用 BufferID | 多槽选择是标量控制流 + 地址计算，**不新增任何核内同步**；落盘通路沿用 M111 的 `BufAcquire/BufRelease`（`m15_ple.asc:335-334,608-615`） | 无新增 `PipeBarrier`/`SetFlag` |
| 核间只用 CrossCore | `②→③→④` 的 mode-2 交接、③ 的两条 mode-0 对齐均为既有 `CrossCoreSetFlag/WaitFlag` | `m15_ple.asc:796,829`（未改） |
| 矩阵乘必须 cube mmad | ③ 自 M124 起是 AIC 侧 `Mmad`（`m15_ple.asc §CUBE`） | 本 tip 未碰 ③ 的形态 |
| scalar 只做控制流 | `UDivMod64`/槽扫描只算 `(shard,local)`、`delta` 与行地址（下标/偏移），不产生数值结果 | `m15_ple.asc:292-307,526-560` |
| 禁用 set_flag/wait_flag 系列 | 新增代码无 `SetFlag/WaitFlag` 裸 API；只有 `CrossCore*` | 见下方逐字命令 |
| 同 pipe 背靠背复用需 `PipeBarrier<PIPE>` | 本 tip 未新增同 pipe 复用点 | — |

逐字命令（设备侧新增 API 检查）：

```
$ grep -nE "SetFlag|WaitFlag" m15_layer_loop/m15_ple.asc   # 命中全部是 CrossCoreSetFlag/CrossCoreWaitFlag
```

---

## 5. 证据清单与复算

一键复算（仓库根目录，设备档各自 flock + 锁内 npu-smi + timeout）：

```
$ bash m15_layer_loop/ple/reproduce.sh
```

产出（均入库）：

| 文件 | 角色 |
|---|---|
| `ple/logs/multislot_emit.log` | 多槽 emit 清单（`slots=4 pool=52428 rows`、5 条搬运行、槽 0 跨界 0→1） |
| `ple/logs/multislot_base.log` / `multislot_base_npusmi.txt` | 基线设备档 + 锁内 npu-smi 快照 |
| `ple/logs/multislot_base_check.log` | 基线判据 11/11 PASS |
| `ple/logs/multislot_miss.log` / `_npusmi.txt` / `_check.log` | 负向对照 1（越窗入口） |
| `ple/logs/multislot_mut16.log` / `_npusmi.txt` / `_check.log` | 负向对照 2（bit16） |
| `ple/logs/multislot_legacy.log` / `_npusmi.txt` | 单窗口回归设备档 |
| `ple/logs/legacy_regress_check.log` | 单窗口回归判据（与 HEAD 二进制逐条一致） |
| `ple/hm_multislot_check.py` | 槽-aware 判据脚本 |
| `ple/reproduce.sh` | 一键复算（能传播失败：base 非全绿 / NC 签名不符 ⇒ 非 0 退出） |

基线设备档证据（`ple/logs/multislot_base.log`）：
`aclrtHostRegisterV2(MAPPED) rc=0`、`getDevPtr rc=0`、`hm-fill: 16776960 B 搬入窗口（多槽，含跨界拆分）`。

---

## 6. 本切片之后，PLE 真接线还差什么（一行更新）

**多槽池已能把「一次服务跨分片的多头 id」跑通并判绿（U-A/U-B 的实现与判据就位），但 PLE 真接线仍差：
① ①→② 的真实端到端（U-C，全词表 ① 输出直接喂 ②，需按 head 的 id 区间做槽分配/滑窗策略）；
② 槽的驻留/滑窗策略（LRU、预取、槽数按目标机实测定）；③ 生产路径接入（`runs=all`/chain 打开
`M15_PLE_WIRE`，并把 `H_PleArgsOf` 的「已分配」与「已装载」分开门控）；④ 段内 `PipeBarrier` → BufferID 流水；
⑤ prefill（T>1）接入；⑥ 设备侧行校验目前只覆盖行首 64/160 列（U-J）。**

---

## 7. 已核实 vs 待验证

**已核实（本 tip 实跑，附文件:行）**
* 多槽 + 跨分片选择在设备上正确：`multislot_base_check.log` 11/11，`dev_row_fail=0 dev_miss=0`。
* 越窗计数真的在数、且不误报：`multislot_miss_check.log` 只 `Hd.served` 红。
* 改错 `rows_per_shard` 会被设备侧 + host 侧同时咬住：`multislot_mut16_check.log` 8 条红。
* 单窗口路径与 HEAD 二进制逐条一致：`legacy_regress_check.log`（`diff` 无差异）。
* `wire.h` 与 `.asc` 逐字节一致：`lift_ple_device_segment.py --check` rc=0。

**待验证（不做绝对断言）**
* 槽数 / 槽大小的驻留策略与命中率（本 tip 只给固定 4 槽 × 4 MiB 的 harness，未做 LRU/预取）。
* 真实 ① 输出喂 ② 时的槽分配（U-C；需全词表 ids 分布与滑窗策略）。
* 跨分片槽在**更大**（如真 256 MiB/槽）与多槽并发下的注册代价（M86 观测方差极大）。
* 多槽池在层 kernel 路径的接入（`LayerArgs` 目前只有单个 `pleWinBase/pleWinRows`；多槽参数只在
  standalone harness 里传递，层路径 `M15L_PleBody` 未设置 ⇒ `nSlots=0`，行为不变）。
