# M110 Wave A —— 接口面（**写码之前**的改动清单）

> **状态（M144 定案）**：本文件是 M110 的**写码前接口面快照**，**不是权威清单**（权威 = 下方三处 + 现码；被引事实与逐条对账见文末 §4）。本文件是任务 1 的交付物，**在任何源文件改动之前写下**（它的 commit 早于同 mission 的源改动 commit，
> `git log --oneline -- m15_layer_loop/evidence/prefill_contract/interface_face.md m15_layer_loop/m15_layer_resources.h`
> 可验证顺序）。权威清单 = `docs/15-prefill-design.md` 的 `## M103 重盘（@20bd20d）` 一节（M103-2.1
> Wave A 表 + M103-2.3 Wave C + M103-3）+ `docs/19-gdn-prefill-scan-selection.md` §4.4（G2 缝函数 +
> `GdnStateHome`）+ M101 复审 F2（mode-4 / `FLAG_PER_CORE`）。
>
> 基准 commit：本 mission 的 base = `d348547`（M100 合入后的 main）。行号只作定位辅助，**符号名才是锚点**。

## 0. 判据（本 mission 的完成定义）

**B1–B5 的实现者能否只靠本 mission 交付的东西把自己的段接进来** = 资源常量 + 段接口（缝函数 /
`GdnStateHome` / 段签名）+ 挂载点字段 + **默认关闭的编译期开关**。读数次之。
本 mission **不交任何段体实现**。

## 1. 要改的每一处 `文件:符号` 与理由

### 1.1 `m15_layer_loop/m15_layer_resources.h`

| # | 符号（基准 commit 的锚点） | 改动 | 理由（权威出处） |
|---|---|---|---|
| ① | `M15L::ENTRY_GDN_PREFILL` / `ENTRY_ATTN_PREFILL`（§6，尾部两行 `const char*`） | 从「符号预留，未实现」的裸字符串 → **真入口登记**：新增 `EntryReg ENTRY_TABLE[]` 登记表（符号名 / kind / 相位数 / hc / prefill / 接线开关 / 备注），两个 prefill 项 `wired=0`；文件头 §二 的入口表同步改写 | M103-2.1 ①；`m15_layer_kernel.h` 里要出现真 `__global__` ⇒ 登记表是 host 选符号的唯一权威 |
| ②a | §4 尾部 `FLAG_PER_CORE = 16` + `FlagSeqAdjacentOk()`（`id >= FLAG_PER_CORE → false`） | **裁决 `FLAG_PER_CORE`**：保留 16 作 mode0/1/2 的每核池，新增 `FlagIdLimit(core, mode)`（AIC ∧ mode==4 ⇒ 32）；`FlagSeqAdjacentOk()` 改用它 | M101 复审 F2：`FLAG_PER_CORE=16` 结构性地拒收 AIC mode-4 的 id 16/17（`ccMM + AIV_CH`，`AIV_CH=16`）⇒ 不裁决就接不进 B3 |
| ②b | 新增 `§4b-2` mode-4 分节（**订正（M144 对账）：现码里 mode-4 分节是 `§4c`（`m15_layer_resources.h:929`），不是 `§4b-2`**） | 新增 `FLAG_SEQ_ATTN_CORE[]`：把 `m15_attn_core.h`（M101，未合入）的 **35 次调用 / 11 个 (mode,id)** 逐个登记成 `(核型, mode, id, pipe, 方向, 用途)`；新增 `FlagSeqAttnCoreOk()` | M103-1.2 N2 + M103-2.1 ②（「含 mode 4，必须逐个登记并给相邻性 static_assert」） |
| ②c | 新增 `§4c` prefill 分节（**订正（M144 对账）：现码里 prefill 的 flagId 分节是 `§4d`（`m15_layer_resources.h:1051`）；`§4c` 现为 mode-4 分节（`m15_layer_resources.h:929`）**） | `FLAG_SEQ_PREFILL_GDN[]`（**订正（M138 实核 / M142 同步）：B1 用 6 个 mode-2 号 `M15G::GP_FLAG_GO1..DONE3` = 0..5**，与 `m15_gdn_resources.h:396-401` 同源；原记的「复用 decode GDN 的 4-7 / 8-11 / 12-15」是误记）+ `FLAG_SEQ_PREFILL_ATTN_PA[]`（**订正（M144 对账）：权威名 = `FLAG_SEQ_PREFILL_ATTN[]`（`m15_layer_resources.h:1113`，6 行 = M97 的 2 个 + B3 core 的 4 个）；`_PA` 后缀名现码里不存在（`git grep -n FLAG_SEQ_PREFILL_ATTN_PA` 的命中都落在本快照文件内）**）（B2：复用 M97 的 2 个）+ 复用见证 `static_assert`（集合包含 + 入口互斥见证 `FlagSeqPrefillReuseOk()`（**订正（M144 对账）：现码拆成 `PfGdnReuseOk()`（`m15_layer_resources.h:1134`）与 `PfAttnReuseOk()`（`m15_layer_resources.h:1149`），无此单名**）） | `docs/19` §4.1 约束 2：「能否复用必须用 static_assert 序列显式见证，不能靠"反正不同时跑"的口头论证」 |
| ③ | 新增 `§2b` / `§3b` 峰值断言位 | `PF_PEAK_UNSET` 哨兵 + `PF_{UB,L1,L0C}_<段>` 槽位 + `PREFILL_WIRED` 编译期开关 + `PfPeaksFilled()` + `static_assert(!PREFILL_WIRED \|\| PfPeaksFilled(), …)` | M103-2.1 ④「预留 UB/L1/L0C 峰值断言位（数值等 Wave C 填）」。**用哨兵 + 开关合取**，否则「填 0 即通过」是空洞断言（`docs/17` §4） |
| ④ | §1 `MW_*` / `MOE_W_STRIDE` 数值区 | ① 把每一式子抽成 `MwXxxBytes(E)` 形式（本文件仍是**唯一 owner**）；② 新增 prefill 定尺常量 `MOE_E_PREFILL=512`、`MOE_TOPK_PREFILL=10` 与 **prefill 槽表** `MWP_*` / `MOE_W_PREFILL_STRIDE`；③ `static_assert` 同时钉死 decode 档（E=4）与 prefill 档（E=512）的每个数值 | M103-2.1 的 A-2（塔裁：`MW_*` / `MOE_W_STRIDE` 唯一 owner 是本文件，B4 只消费）；E=512 时 `MW_WGU_BYTES` = 838,860,800 B（= 128×） |
| ⑤ | §5 `BUF_TABLE[]` / `BUF_AIV_PEAK` / `BUF_AIC_PEAK` | 新增 prefill 相位 A 的 AIV 行窗（0..2）与 AIC 的 mmad 窗（0..6，7..10 预留）登记行 + prefill 峰值常量 | M103-2.1 ③「新增 prefill 的 BufferID 分节」；人类要求「所有 buffer id 自己管理、编译期静态分配」 |

### 1.2 `m15_layer_loop/m15_layer_kernel.h`

| # | 符号 | 改动 | 理由 |
|---|---|---|---|
| ① | `struct LayerArgs` 尾部（现在是 M100 的 PLE 字段，`pleStateSlots` 收尾） | **追加** prefill 字段（**不动任何既有字段、不重排**）：`pfKv/pfComp/pfRing/pfPack`（三套 cache + packed indices 基址）、`pfPos`（per-row 位置表，可空）/`pfPosBase`（连续请求起点）/`pfBlockTable`（页表，nullptr = 恒等表）、`pfLayerK`、`pfCounts/pfExpertOffsets`（MoE prefill）、`pfRowHit`（**行分派见证面**）、`pfStageMask/pfWired/pfMutant` | M103-3（`posArr` / `slotMapping` / `layerK` / 四个 cache 指针 / `counts` / `expertOffsets`）；塔补充 5：与 PLE 字段的相对位置必须显式说明 |
| ② | 新增 `M15L_LAYER_PREFILL_ARGS_DECL` / `M15L_LAYER_PREFILL_ARGS_FILL` | 两个宏，按既有 `M15L_LAYER_ARGS_DECL/FILL` 的写法（基础 27 实参 + prefill 追加项） | M103-2.1（`m15_layer_kernel.h` ① 的宏要求） |
| ③ | 新增 `M15L_PrefillPhaseA<KIND>()`（**段体挂载点**） | 相位 A 的行分派骨架 + 逐行 `pfRowHit[row]++` 见证；`pfWired=0` 时是「逐行 x→y 拷贝」的结构性占位（**数学未实现，显式标注**）；`pfWired=1` 时进「段体挂载点」分支，段体未落地 ⇒ **响亮失败**（写毒值 + 计错），不允许静默绿 | M103-2.7 第 3 条（每段一个默认关闭的开关）+ 任务「至少一个最小档能跑起来并给出读数」 |
| ④ | 新增两个 `__global__ __mix__(1,2)`：`m15_layer_kernel_gdn_prefill` / `m15_layer_kernel_attn_prefill` | 真入口（不是注释 + 字符串常量） | ① 的登记表必须有对应符号，否则「真入口登记」是空的 |
| ⑤ | 文件头入口清单（基准里第 68 行那句「预留，未实现」） | 改为真入口 + `wired=0` 的说明 | 文档与代码同步（`docs/17` 的 doc-ref 纪律） |

### 1.3 `m15_layer_loop/m15_layer_loop.asc`

| # | 符号 | 改动 | 理由 |
|---|---|---|---|
| ① | `struct Opts` + `H_ParseOpts` | 新增 `pfWire`（`M15_PREFILL_WIRE`，默认 **0**）、`pfM`（`M15_PREFILL_M`，默认 4097）、`pfKind`、`pfLayer`、`pfMutant` | M103-2.7 第 3 条（形态照 `M15_PLE_WIRE`） |
| ② | `struct Ctx` + `H_Alloc` | 新增 prefill 激活平面（x/y 按 `M15Loop::H_ROWS_BYTES_PF`）、`pfRowHit`、`pfPos` 平面；尺寸全部取自 `m15_loop_layout.h` 的新常量 | 任务「`m15_loop_layout.h`：激活平面按 m 定尺」的消费侧 |
| ③ | 新增 `H_RunPrefill()` | 最小档：填 x 行图案 → 起 `*_prefill` 入口 → 判据（y 与 x 逐字节 + `pfRowHit` 全 1）+ 负向对照（`M15_PREFILL_MUTANT=1` 把行循环截到 `M_MAX` ⇒ 必须变红） | 任务「`runs=` 加 prefill 档，至少一个最小档能跑起来并给出读数」+ `docs/17` §4 非空洞性 |
| ④ | `main()` 的分派 | 新增 `runs == "prefill" \|\| runs == "pf"`，**故意不进 `runs=all`**（零回归） | 零回归要求 + 既有各档不进 `all` 的先例 |
| ⑤ | include 块 | **不新增 include**（本 mission 的 scope 只有 4 个源文件，夹具头无处安放 ⇒ 夹具直接写在本 `.asc` 里） | 见 §3 的偏差说明 |

### 1.4 `m15_layer_loop/m15_loop_layout.h`

| # | 符号 | 改动 | 理由 |
|---|---|---|---|
| ① | `H_ROWS_BYTES`（基准第 96 行 = `M_MAX * HIDDEN * 2` = 327,680 B，只够 64 行） | 新增 `M_PREFILL = 4097`、`H_ROWS_BYTES_PF = 4097*2560*2 = 20,976,640`、`H_BYTES_PF`、`RESZERO_BYTES_PF`、`WS_PF_*` 派生尺 + 每项 `static_assert` 钉死数值（含 `q\|gate` 12288 列 = 100,687,872 B、`qkvzba` 16480 列 = 135,037,120 B、`opin` 6144 列 = 50,343,936 B 三条手算式） | M103-2.1 Wave A 表最后一行 + M103-3「每个按行集的平面 = 20,976,640 B」；复审 r1 的 P2-1 就是这一族手算错 |

## 2. 不做的（显式划界）

- **N1（`subOut` 的生产者）**：`docs/15` M103-2.1 原把它归 Wave A（`m15_layer_kernel.h` 那行第 ③ 条），
  **塔裁 2026-09-27 改为随 Wave C（相位 B 接线）落地**，依赖 M101 的核心 + `o_proj` 合入。
  ⇒ 本 mission **不实现**，只登记（四点见 `README.md` §5；核内登记点在 `m15_layer_kernel.h` §3d）。
  ⚠ 这条是 r1 复审（P2-2）抓出的"清单里有、交付里 0 命中、也没解释"——已在 r2 补齐登记。
- **B1–B5 段体**：`m15_gdn_prefill*` / `m15_attn_prefill*` / `m15_attn_fa_core*` / `m15_moe_prefill*` /
  `m15_hc_prefill*` 一行不写（M103-2.2 的 scope）。
- **Wave C 收口**：真实峰值数值（本 mission 只交断言位 + 哨兵门）、把段体接进融合 TU。
- **Wave D**：`m15_chain_host.h` / `m15_moe_host.h` / `m15_hc_host.h` 的 host 接线、H2D 装载与 dump 口径。
- **aclgraph / 性能**：不在本 mission。
- **`Scan_SolveG` 的裁决**：`docs/19` §10.2 自己标「请塔确认」⇒ 本 mission **写成待定**，不替塔裁决。

## 3. 需要改清单外文件的地方（**先报塔，不越界**）

A-2 的字面要求（「Wave A 把 `MW_*` 按 `NUM_EXPERTS = 512` 重算」）若**原地改**这四组数字，会立刻牵动
四个**不在本 mission scope** 的文件，并破坏 `runs=all` 零回归：

| 清单外文件 | 为什么必须一起改 | 影响 |
|---|---|---|
| `m15_layer_loop/m15_moe_resources.h` | `NUM_EXPERTS=4` / `TOPK_MAX=4` 是它的常量（M103-2.2 B4 的「不许碰」列）；不改它，`MWP_*` 与 `M15M::SZ_*` 对不上 | 全段 UB/GM 窗重算 |
| `m15_layer_loop/m15_moe_host.h` | `W.wGu.resize(M15L::MW_WGU_BYTES)` + `H_MoeWeightSourceCheck` 按 manifest 逐字节核 ⇒ 槽表一变，E=4 manifest 立刻红 | `runs=all` 的 84 条权重来源判据 |
| `m15_layer_loop/slice_layer_manifest.py` + `weights_manifest.txt` | manifest 的 tensor `bytes=` 决定槽内填充量 | 重新切片 |
| `m15_layer_loop/m15_layer_loop.asc`（`H_Alloc` 的 `moeArenaBytes = NL * MOE_W_STRIDE`） | E=512 时 **48 × 1,342,182,400 B = 64,424,755,200 B ≈ 64.4 GB** 的 **host** arena；本机 cgroup 上限 34,359,738,368 B（32 GiB）⇒ **装不下**，host 侧必须改成分片/惰性装载 | 另立 mission |

⇒ 本 mission 的选择：**同一个 owner 文件里给两套定尺**（decode 档原值不动 + prefill 档新表
`MWP_*` / `MOE_W_PREFILL_STRIDE`），两套都由**同一批式子**导出并由 `static_assert` 钉死；
真正的「把全套换成 E=512」列为**清单外依赖 mission**（已 `TowerSend` 报塔）。

## 4. M144 对账与定位（状态说明 + 被引事实 + 行号纪律）

**定位（M144 定案）**：本文件**不是权威接口面**，而是 **M110 写码前的快照**（其 commit `b079732`
早于同 mission 的源改动 commit，见 §0 的行号/符号纪律）。权威 = 本文件 §0 所列的 `docs/15`
（M103 重盘）+ `docs/19`（§4.4 缝函数 + §4.1 约束 2）+ `M101` 复审 F2 + **现码**（`m15_layer_resources.h`）。
后人若把本文件当接口真相去接段，会踩到下面的历史性漂移；以现码为准。

**被引事实**（命令：`git grep -n interface_face -- ':!m15_layer_loop/evidence/prefill_contract/interface_face.md'`，
只查**跟踪文件** ⇒ 天然排除未跟踪的 `.tower/`（在主检出上用裸 `grep -rn .` 会额外命中 `.tower/` 的 worktree
副本与 comms，故此处以 `git grep` 为准）；末一参数再排除本快照文件自身及其自指）——在源树上实测，
外部引用 **4 处**，都按「写码前清单 / 快照」引用（性质见下表），未见把它当权威接口面的位置：

| 引用处（文件:行） | 引用形态 | 性质 |
|---|---|---|
| `m15_layer_loop/evidence/prefill_contract/README.md:9` | 「**写码之前的接口面清单**在 `interface_face.md`（它的 commit 早于源改动）」 | 快照 |
| `m15_layer_loop/evidence/prefill_contract/README.md:12` | 「任务 1 的交付物：写码之前逐条列出的 `文件:符号` + 理由」 | 快照 |
| `m15_layer_loop/evidence/prefill_contract/README.md:347` | `interface_face.md` §2（N1 不做项的登记点，**符号引用**） | 登记点 |
| `m15_layer_loop/evidence/prefill_contract/README.md:365` | `interface_face.md:26`（**行号引用**，指本文件 ②c 行） | 行号锚点 |

**漂移对账（逐条拿现码核实，文件:行）**：

| 本文件（快照原文） | 现码权威 | 核实（文件:行） |
|---|---|---|
| `FLAG_SEQ_PREFILL_ATTN_PA[]`（§1.1 ②c） | `FLAG_SEQ_PREFILL_ATTN[]`（6 行 = M97 的 2 个 + B3 core 的 4 个） | `m15_layer_resources.h:1113`（定义）、`:1121`（`FLAG_SEQ_PREFILL_ATTN_N`）、`:1122`（`== 6u` 断言） |
| `§4c` = prefill 分节（§1.1 ②c） | `§4d` = prefill 的 flagId 分节 | `m15_layer_resources.h:1051`（节头） |
| `§4b-2` = mode-4 分节（§1.1 ②b） | `§4c` = mode-4 分节（attention core 家族） | `m15_layer_resources.h:929`（节头） |
| `FlagSeqPrefillReuseOk()`（§1.1 ②c，复用见证） | `PfGdnReuseOk()` / `PfAttnReuseOk()` | `m15_layer_resources.h:1134` / `:1149` |

死名核实（命令：`git grep -n -e FLAG_SEQ_PREFILL_ATTN_PA -e FlagSeqPrefillReuseOk`，只查**跟踪文件** ⇒ 天然不含未跟踪的 `.tower/`）：本 tip 上两名的命中都落在**本快照文件自身**（②c 快照行 + 本节 §4 的「快照原文/死名」两处自述）—— **除本快照文件自身外**，全仓（`git grep` 的跟踪范围）再无命中 ⇒ 二者是快照里的历史名，现码无此符号。

**边界（未动任何登记表 / 判据 / 读数）**：M144 只改本文件 §1.1 ②b/②c 两行的**名称与节号措辞**（同行加「订正」标记）与本节；§1 的其余行、§2 / §3、`m15_layer_resources.h` 的登记表与 `static_assert`、以及所有读数均未触碰。

**行号纪律（M144）**：上述订正一律**同行替换 / 同行加注**（不增删 §0–§3 的任何行）⇒ 本文件第 1–84 行**行号未移动**，`prefill_contract/README.md:365` 的 `interface_face.md:26` 锚点仍指向同一行（②c）。
