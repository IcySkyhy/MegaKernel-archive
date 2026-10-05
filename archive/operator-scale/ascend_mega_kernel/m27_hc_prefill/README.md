# m27_hc_prefill —— M119（Wave B5）**hc prefill 行分块（m-tile）段体**的独立验证路

> **被测对象** = `m15_layer_loop/m15_hc_prefill.h`（段体，设备头）+ `m15_layer_loop/m15_hc_prefill_host.h`（host 几何/校验/布局）。
> **本目录** = 自带 CMake 工程 + 带 `main()` 的 `.asc` + 独立 numpy 判据 + 证据（`docs/15 §M103-2.7` 的"独立验证路"三件套）。
> **不碰** `m15_layer_loop/CMakeLists.txt`（塔裁；归在飞的 M101）、`m15_layer_loop/m15_hc_layer.h`、
> `m15_layer_loop/m15_hc_resources.h`（lift 产物，手改会破 `lift_hc_segment.py --check`）、
> `m15_layer_loop/m15_hc_host.h`（M100 在改）。

## 1. 一句话

`M15H::M_MAX = 64` 是**编译期**定尺；`m=4097` 直接喂进 hc 段要抬包络，而抬包络必爆 UB
（`UB_IWTAB_SLOTS = M_MAX*HC` ⇒ 4097 行要 `524,416 B > 248 KB = 253,952 B`）。
本 mission 的交付 = **外挂一层 m-tile 循环**（缺省 8 行/块，可 `-DM15_HC_PF_MT=12`），
每块重新组装 `HcPtrs`（输入按行偏移、`ws` 指向该块的 arena 槽、`m` = 本块行数）后调**既有的**
`M15H::HyperConnOp`，从而在**不改 helper 语义**的前提下吃下任意 `m`。

## 2. 权威文档逐条核对（先读再写码）

权威 = `docs/15-prefill-design.md` 的 `## M103 重盘（@20bd20d）` 一节：§M103-2.2 的 **B5 行** 与
§M103-1.4（hc 段的包络分析）。逐条对当前 main 复算/实测：

| 权威说法（§M103-2.2 B5 / §M103-1.4） | 在 main 上的复算 / 现状 | 本 mission 的处置 |
|---|---|---|
| 段体头交付到 `m15_layer_loop/m15_hc_prefill.h` | 交付 ✓（本头） | 同 |
| 独立验证路 `m15_hc_prefill/**` | **本 mission 的 scope 写的是 `m27_hc_prefill/`**（塔的任务书） | 落到 `m27_hc_prefill/`；目录名与 §M103-2.6 的 `m15_hc_prefill/**` 不同，**内容与三件套一致**。这条差异在此显式登记 |
| 允许改：**方案 α（不改任何既有 hc 文件）** | 塔裁（M119 任务书第 4 条）也选 α | 取 α：`m15_hc_layer.h` / `m15_hc_resources.h` 一字未改（`git diff` 可核） |
| AIV 侧五个段的 item 网格已经是 `p.m * NCH_*` | 复读 `m15_hc_layer.h`：`InjwStage`/`CombineStage`（`p.m*NCH_D`）/`NormStage`（`p.m*HC`）/`SiluStage`（`p.m*NCH_R`）/`GateMixStage`（`p.m*NCH_H`）**全部**按 `p.m` ✓ ⇒ hc 不缺行循环，缺的是包络 | 本头只做"切块 + 重绑定"，不重写段序 |
| 墙 = `UB_IWTAB_SLOTS = M_MAX * HC`，`INJW_SLOT = 8`：64 行档 8,192 B，4097 行档 **524,416 B** | `static_assert` 逐条复算（`ENVELOPE_IWTAB_BYTES_*`）：8,192 ≤ 253,952 ✓；524,416 > 253,952 ✓ | 用**块高 ≤ M_MAX** 绕过（`static_assert(MT*HC <= UB_IWTAB_SLOTS)`） |
| 建议 m-tile ≤ 8–12 行（正文 §6.1 R3） | `MT = 8`（缺省），上限断言 `MT <= 12`；工程另编 `MT=12` 档 | 同 |
| UB 余量：`UB_PEAK = 89,472` 对 248 KB | 按 `m15_hc_resources.h` 的偏移式**逐项复算** = **89,472** ✓ | 本头在其上只加 5,120 B 行 staging（峰值 94,592 ≤ 248 KB，`static_assert`） |
| `HcBf16Gemm`（mixer 投影 down/up）已经是 mmad | 复读 `m15_hc_layer.h`：`Mmad` + `Fixpipe`，**没有** VF 收缩；本头不新增任何矩阵乘法 | 人类裁决①「矩阵乘法一律 mmad」在 hc 段**本来就满足**，本头无需换实现 |
| `BASE_M = 64`、`RunTile` 契约「`mLoop` 必须为 1（m ≤ BASE_M）」 | 每块的 `p.m = TileRows(m,t) ≤ MT ≤ 12 ≤ 64` ✓ | 不变（块内仍是单 m-tile） |
| 不许碰 `m15_layer_loop/m15_hc_host.h`（M100 在改） | 未碰（改用 `m15_hc_prefill_host.h` 承载本段的 host 几何） | 同 |
| scope 里没有 CMakeLists / Wave A 4 文件 | 未碰 | 同 |

**没有需要改清单外文件的地方**（⇒ 未向塔报"要改清单外文件"）。

## 3. 设计：块几何 + arena 不变式（三条，**机械**可核）

```
块 t 的 ws 基址 = arena + t*TILE_STRIDE      TILE_STRIDE = MT * HYPER * 2 = 163,840 B（MT=8）
WS_HCP == 0（static_assert）⇒ 块 t 的 H' 落在 arena 第 t*MT 行起、行距 HYPER
                              ⇒ 整个 arena 的前 m 行 = **H' 的平铺 [m, HYPER] 平面**
```

三条性质（都由 `m15_hc_resources.h` 的常量 + `static_assert` 推出，不靠人眼）：

1. **H' 不被自己的 scratch 踩**：`TILE_STRIDE <= WS_XN`（= 块内任何区都起于 H' 行之后）
   ⇒ 块 t 只在它自己那 MT 行 H' 上写。
2. **H' 不被别的块踩**：能被块 t' 的**任何写**覆盖到块 t 的 H' 行的充要条件是
   `t'*MT < t*MT + MT` ⇒ `t' <= t`；而 `t' = t` 的那次写就是块 t 自己的 CombineStage（块内第一步）
   ⇒ **只要块按下标递增执行，H' 平面每行的最后写者就是它的属主块**。
3. **`blk` / `injw` / `rstd` 一定会被更晚的块覆盖**：`CoveredByLater(...)` 两条 `static_assert`
   在编译期**搜出**那个更晚的块（MT=8：blk 被块 t+8 的 GATE 写窗覆盖、rstd/injw 被块 t+16 的
   H' 写窗覆盖；MT=12：t+5 / t+10）⇒ 它们必须**在块末重定位**才能当平铺平面用。

**重定位**（**块末**，每块一次：`Body` 的循环里，`ProcessAiv` 之后、进下一块之前）：
> **这条是实测改出来的**（M119 的第一版把它放在"循环结束后一次搬"，m=4097 档直接红：
> `blk` 逐位一致率 **0.015**、`rstd` maxRel **3.2e+00** —— 因为块 t 的这三个区在块 t+8/t+16
> 写下去的那一刻就被覆盖了，循环结束时槽里已经不是它的产物）。小档（m ≤ 64，块数 ≤ 8）看不出
> 来：那时没有"更晚的块"来覆盖 ⇒ **两种做法在小档同绿、在 m=257/4097 档分离**，这是判据灵敏度
> 的一个天然内部对照。
- 可见性：源由**别的 AIV** 的 MTE3 写出 ⇒ 进入 pass 前一条**全体 AIV 的 mode-0 barrier**
  （set 挂 MTE3 = 写 GM 排空 / wait 用 MTE2，即 donor 的 `BarrierAiv<PIPE_MTE2, FLAG>`）；
- 搬运：`blk`（5,120 B/行）、`injw`（128 B/行）逐行 DMA；`rstd` 只有 16 B/行 ⇒ 按**成对**（行 2j、2j+1）
  搬 32 B（`i`、`r` 同奇偶 ⇒ 两端都 32B 对齐，`MT` 为偶数是 `static_assert`）；
- 一次循环过**所有行**、按 `bid / nAiv` 分派（与 hc 段同款行网格），不引入额外的相位；
- 规则 ⓔ：全程 DMA（`BufAcquire<PIPE_MTE2>` → `DataCopy` → **阻塞释放** → 对侧 `BufAcquire<PIPE_MTE3>`），
  不写任何标量、不用 set/wait flag、不靠 `PipeBarrier` 承担可见性。

**零拷贝的两条**（`m27` 的 A2 档实测的就是这两条）：边界 #2 的 `hIn` 直接指边界 #1 的 arena
（H' 平铺面），`ij` 直接指边界 #1 的 `OH[:,320:324)`（`WS_OH + OH_INJ*2`、行距 `OH_W`、
按块推进 `TILE_STRIDE`）—— 层内 handoff **不走 host**（M58 的主张，在 prefill 尺度复验）。

## 4. 验证

### 4.1 怎么跑

```bash
bash m27_hc_prefill/reproduce.sh /tmp/m27_out          # 构建 + 设备跑 + 判据 + 两个负向对照
# 或分步：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m27_hc_prefill/build -S m27_hc_prefill -DCMAKE_BUILD_TYPE=Release && cmake --build m27_hc_prefill/build -j4
mkdir -p /tmp/m27_out && cd /tmp/m27_out
flock -w 900 /tmp/npu0.lock bash -c 'M27_DUMP=1 /workspace/ascend_mega_kernel/.tower/worktrees/wt-119/m27_hc_prefill/build/m27_hc_prefill'
/usr/local/python3.12.13/bin/python3 /workspace/.../m27_hc_prefill/check_ref.py /tmp/m27_out   # rc: 0/1/2
```
环境变量：`M27_CASES=a,b`（档过滤）`M27_CHAIN=0`（关第二边界）`M27_SYNTH=1`（合成权重）
`M27_DUMP=1`（落盘）`M27_MUTANT=rowsteal|norel`（负向对照）`M27_ARENA_MAX_BYTES`（整块 arena 落盘阈值，
缺省 32 MB）`M27_LAYER`（权重层号）。

### 4.2 档表（`m27_hc_prefill.asc` 的 `cases[]`）

| 档 | m | mode | 覆盖什么 |
|---|---|---|---|
| `m1` | 1 | COMBINE_MIX | 单行（= decode 尺寸）；1 块 |
| `m1_mix` | 1 | MIX | 无 combine 分支（H' 不物化 ⇒ hcp 项 SKIP、A2 整段 SKIP） |
| `m1_conly` | 1 | COMBINE_ONLY | 第 4 档 mode（只 W0+S1；AIC 直接返回 ⇒ **没有 mode-2 交接**） |
| `m9` | 9 | COMBINE_MIX | 2 块 + 尾块 1 行（块高边界的部分块） |
| `m33` | 33 | COMBINE_MIX | 5 块；M58 同一个 m 档（在那边是 numpy 门限越界的披露档） |
| `m64` | 64 | COMBINE_MIX | = `M_MAX`（整个包络） |
| `m257` | 257 | COMBINE_MIX | 33 块：**覆盖真的发生**（> d_gate/d_rstd）⇒ arena 差分见证在这一档生效 |
| `m4097` | 4097 | COMBINE_MIX | **验收档**：513 块 = 真实 shape；flag 用量最大的一档 |
| `m4097_mix` | 4097 | MIX | 真实 shape 的无 combine 分支 |

### 4.3 判据来源与方向（不自己造第二套数学）

- 参考 = `m20_hyperconn/check_ref.py::reference`（独立 numpy **float64**，按 `m` 参数化）；
  门限 = 同一文件的 `judge()`（归一化绝对误差 ≤1e-2 **且** 良态 bf16 ulp ≤2 **且** 良态逐位一致率 ≥99%）。
  `injw` / `rstd` 是 fp32 张量 ⇒ 单独用相对误差判据（≤1e-6 / ≤1e-4，与 m20 的 B 段同口径）。
- 判 A1（边界 #1）与 A2（边界 #2，**输入取设备自己的一档 dump** ⇒ 把两个边界解耦便于归因）。
- 判据只在 python（`.asc` 只落盘 + 三条 host 自检："HCP 已写出 / mode MIX 未写 HCP / 重定位后不再是毒值"）。

### 4.4 负向对照（"把被测对象弄坏必变红"）

**逐档的"谁保证 rc"矩阵在 `evidence/negative_control_matrix.log`**（由 `tools/negative_control_matrix.sh`
一条命令生成；每步打印命令与 rc）。下表是它的文字版，**按当前代码**逐条对齐 —— 复审 r2 的 B1
点出的正是"掩码化之后旧表述不再成立"：

| 档 | `reloc_mask` | 弄坏什么 | **当前**读数（逐项） | rc 由**谁**保证 |
|---|---|---|---|---|
| `M27_MUTANT=rowsteal` | 0xF | 三个块推进量**各少一行** ⇒ 块与块重叠一行（块 0 的推进量用不到 ⇒ **单块档 m ≤ MT 不受影响**，是判据灵敏度的一个内部对照） | 归档（设备）：**10 条红**，含 `a1.blk`/`a1.ij_handoff`/`a1.rstd`/`a2.blk`（m33）与 `hcp`（m257）；m1 档不红 | **判定项值级红** |
| `M27_MUTANT=norel`（**现形**） | **0x0** | 四个重定位目标全传 `nullptr` = 调用方**明确不要**这四个平铺面 | `blk/injw/rstd/ij_handoff` 判定项**记 SKIP**（7 条 @m33，附"reloc_mask 未当目标"）＋ 新增 guard「被判 SKIP 的平铺面**必须仍是毒值**」OK；红的是**消费那些面的下游项**（`a2.hcp`：它的 ij 输入正是未写的毒值面） | 判定项（下游）＋ 结构 guard；**对四个产物本身没有值级红**（见下面的判别力账） |
| `M27_MUTANT=sinkreloc`（**新**） | **0xF**（掩码不变 ⇒ 契约仍要求四个面有值） | 把四个重定位**落点**旁路到一块 scratch ⇒ 被消费的平面**保持毒值**（"重定位的输出没有到达该到的地方"） | **判据半（零设备已跑）**：`m33` 档 **10 条红**，含四个产物的判定项全部**值级红**（`blk` 逐位 0.0000、`injw` maxRel 4.9e+08、`rstd` 3.8e+08、`ij_handoff` 逐位 0.0000）＋ 两个"值多样"guard 红；**设备半未取得读数**（设备冻结；探锁返回忙）⇒ 登记 §7 | **判定项值级红**（判据半已复现；设备半待解冻） |
| `M27_ONLY_IJFLAT=1` | 0x8 | 只把 ij handoff 面当重定位目标（P2-1 点名的配置） | 该配置**不是**负向对照：判据把未当目标的三个面记 SKIP + 毒值见证，`a1.ij_handoff` 照判（= 修好的判别性那一半） | 判定项（ij_handoff）；SKIP 项有 guard |
| 判据链路自检（`tools/selftest_dump.py`，不占设备） | — | 合成"理想设备输出" / 弄坏任一产物 | 正向 **rc=0**（判定 9 + guard 5）；`--mutate a1.blk` **rc=1**（只红该条） | 判据本身 |
| 空目录 | — | 没有 dump | **rc=2** SKIPPED（三态） | 三态 |

**判别力账（如实，复审 r2 的问题）**：把重定位目标改成位掩码后，`norel` 从"**值级负向对照**"
退成了"**配置档**"（SKIP + 毒值见证）——
- 改前 `norel` 的 36 条红里，**四个产物的判定项占 22 条**（值级）；现形 `norel` 对这四个产物
  **没有值级红**（7 条 SKIP @m33），只剩"消费这些面的下游项"（`a2.hcp`）在红。
- **重定位路径的值级判别力由谁承担**（逐条）：① `sinkreloc`（新）—— **判据半零设备已复现 10 条值级红**
  （含四个产物），设备半待解冻；② `rowsteal`（不变）—— 10 条值级红里含 `a1.blk`/`a1.ij_handoff`/
  `a1.rstd`（**覆盖**这四个产物，但它是扰动输入、**不隔离**重定位）；③ arena 差分 guard（`m257` 档）
  —— 咬的是"**为什么必须**块末重定位"（覆盖真的发生），不是产物的值。
- ⇒ **结论（不越界）**：现形 `norel` 单档**不再**证明"重定位缺失 ⇒ 产物错"；这条现在由
  `sinkreloc`（判据半已复现 + 设备半待跑）与 `rowsteal` 共同承担。`rowsteal` 与自检的两侧读数
  都在归档里，所以"**判据能证明它会红**"这条仍然看得见。
| arena 差分见证（guard，`m257` 档） | —— | 末 d_blk 块的**阻塞** blk == 设备平铺面（逐字节：arena 里确实是段体产物、重定位忠实）；更早块的阻塞 blk ≠ 平铺面（**覆盖真的发生** ⇒ 块末重定位承担负载）；更早块的 arena OH 列 ≠ ij handoff 面（⇒ **ij 也必须块末重定位**） |

### 4.5 「同一判据」的改前 / 改后设备读数（本 mission 的两处修正各一条）

| # | 修正 | **改前**读数（同一批判据） | **改后**读数 |
|---|---|---|---|
| ① | 重定位从"循环后一次搬"改成**块末搬** | m=4097：`blk` 逐位一致率 **0.0154**、`rstd` maxRel **3.17e+00**（同档 `hcp` 逐位 0.9999 通过 ⇒ 红的就是这一环）。该次读数取自 2026-09-27 的设备日志，**日志未随 /tmp 清理保住**，数字见 commit `e9c9632` 的 message 与本表 | m=4097：`blk` 逐位 0.9938 / 良态 0.9977（只剩 ulp 那条边际，见 §7 第 0 条）、`rstd` maxRel 2.4e-05 ✓ |
| ② | A2 的 ij 源从"直接读 arena 的 OH 列"改成**块末重定位出的平铺 ij 面** | m=257：从 arena 抽出来的"OH 列"只有 **22.2% 的行**等于参考的 OH inj 列（且当时 A2 的判决全绿 ⇒ **假绿**）。同上，日志未保住，数字见 commit `e9c9632` | 新增判定项 `a1.ij_handoff` 在 m=1/9/33/64/257/4097 **全绿**（逐位 0.9883–1.0000、良态 ≥0.9955）；A2 四条判定项在 m≤257 全绿、m=4097 仅 `blk` 边际（§7 第 0 条） |

两条改前的**设备读数**用负向对照档**今天可复算**（`M27_MUTANT=norel` = 完全不做重定位；
`M27_MUTANT=ijfromoh` = 把 A2 的 ij 源换回旧路径），读数见 §9 的 mutant 日志与下表。
（"日志未保住"这条如实登记：`reproduce.sh` 能把所有档重跑一遍，所以复算路径不依赖那两份日志。）

**另外两条"设计级"反例（都是实测读数，不是推理）** —— 它们是"判据能区分两种设计"的见证：

| 反例 | 做法 | 实测读数 |
|---|---|---|
| 重定位放在"循环后一次搬" | 块 t 的 blk/rstd/injw 区早在块 t+8/t+16 就被写掉 | m=4097 档：`blk` 逐位一致率 **0.0154**、`rstd` maxRel **3.17e+00**（而同一档的 `hcp` 逐位 0.9999 通过 ⇒ 错的就是重定位那一环） |
| ij 直接取 arena 的 OH 列（不重定位） | 块 t 的 OH inj 列被块 ~t+17 的 H' 写覆盖 | m=257 档：抽出来的"OH 列"只有 **22.2% 的行**等于参考的 OH inj 列；**而 A2 的 8 条判定项当时全绿**（两边用了同一份被覆盖的数据 ⇒ 假绿）⇒ 这才有了 `a1.ij_handoff` 这条判定项 |

**结论以 `evidence/` 下的实跑日志为准**（见 §7）。

### 4.6 读数的机器来源与「未取证」声明（按塔的冻结口径与日志口径）

- **本仓自 2026-09-27 10:33 起处于「设备冻结」**（人类："所以是 ECC 问题，我换台机器"；塔广播
  `20260927-tower-all-item-2.md`）：全体**不再起新设备档**、在跑的收尾、转零设备工作。
  ⇒ 本 mission **不再补跑任何设备档**（含下面 §7 的 `ijfromoh` 负向对照与 m=4097 元数据重生成）。
- **§8 的读数来源（逐条登记，不合并口径）**：
  - 09-27 那一批（`evidence/check_main.log`、`check_mt12.log`）：同机 `Ascend950PR`，AIC=28/AIV=56；
    二进制与判据的 sha256 见 `evidence/sha256.txt`（该文件是 10-04 生成的，覆盖当日四个短档；
    09-27 那批的二进制哈希**未归档** ✗ —— 这是本 mission 的一处证据缺口，如实登记）。
  - 10-04 的四次短档（小档 m1..m257 / m4097+m4097_mix / rowsteal / norel）：都**完成并自洽**
    （小档判据 3 条 `blk` 边际未过、其余全绿；两个负向对照红）。哈希已归档 ✓。
    **机器身份与驱动/CANN 版本未核** ⇒ 按塔的冻结口径，这些只作"同一判据下改后状态"的内部证据，
    **不作跨机对比依据**。
- **本 mission 的设备档没有归档设备侧错误报告**（`errStr` / `ECC` / `aivec` / `aicore error exception` /
  `mte error info` / `retCode` …）：跑档时只落了自检行与 dump（`| tail`）。
  ⇒ 按塔的日志口径（"先做独立计数再引"）**我不能声称"本档内没有 ECC/异常行"**（未取证）；
  因此 §7 第 0 条对 `blk` 余差的"数值累积"归因只是**推断**，**不能排除设备侧异常**
  （已知的形态：`errStr: A multi-bit ECC error occurs when fixpipe reads L0C`，而本段每块都要
  `Fixpipe` 写 GM —— 正是这条通路）。**换机后的第一件验证**：同一档重跑并把设备侧错误报告
  整段归档（独立计数），再决定 `blk` 余差是数值还是设备。

### 4.7 三条强制规则的合规自查（matmul⇒cube / 其余⇒VF / scalar 只做控制流）

范围 = 本 mission 新增的两个头 + 本目录的 `.asc`（被判对象 = `m15_hc_prefill.h`，命令与范围逐字给出）：

```
$ grep -cE "Mmad|Fixpipe|LoadData|__VEC_SCOPE__|Muls\b|Mul\b|Add\b|Reduce|Cast|Exp\b" m15_layer_loop/m15_hc_prefill.h
0        # 该文件里**没有**任何矩阵/向量/标量计算 API（连一次都没有）
$ grep -cE "DataCopy" m15_layer_loop/m15_hc_prefill.h
5        # 全部是重定位的 DMA 行拷贝（搬数据，不做计算）；"DataCopyPad" 的 1 处命中在**注释**里
$ for p in Mmad Fixpipe LoadData __VEC_SCOPE__ Muls Sigoid; do … ; done  # 被调对象（donor，本 mission 未改）
Mmad 4 / Fixpipe 5 / LoadData 6 / __VEC_SCOPE__ 8 / Muls 9 / SigmoidReg 5
```

⇒ **本段体自己不做任何计算**：矩阵乘法与逐元素运算全部在**被它调用的 donor**（`m15_hc_layer.h`，
M58 已验收、本 mission 一字未改）里 —— 那里 `Mmad`+`Fixpipe` 走 cube、其余在 `__VEC_SCOPE__` 里走 VF。
本头里的标量只出现在 `TileCount` / `TileRows` / `ArenaBytes` / 平铺步长 / 重定位地址这些
**地址与循环边界**上（"数据"与"控制流"的界线按塔的口径：参与数值结果的量 → VF；地址/下标/边界 → scalar）。

### 4.8 复审 r1 的 P2-1（重定位门限漏 `ijFlat`）：修法 + 反向验证

**缺陷**（复审逐条给的位置与事实，作者复读确认）：`Body` 的"要不要搬"与 `RelocateTile` 的"能不能早退"
曾经是**两处独立写的条件**，前者漏了 `p.ijFlat` ⇒ 按 §5(b) 省掉 `blk`/`injw`/`rstd`、**只为 H2 传
`ijFlat`（`pfHcIj0`）**这种**文档允许**的配置会连 `BarrierAiv` 都不做，ij handoff 面保持毒值
且**不 Trap** ⇒ 边界 #2 拿垃圾 ij。交付档恒传非空 `dBlk1/dBlk2` ⇒ 该配置**没有任何设备档覆盖**。

**修法两层（"漂移"这一类结构上被去掉）**：
1. **调用侧只留一个入口**：`RelocatePass(p, t)`（门限 + `BarrierAiv` + 拷贝都在里面）；
   `Body` 里**不再有**任何条件 —— 原先那句 `if ((blk) || (injw) || (rstd))` 已删除 ⇒ **没有第二处可以写错**。
2. **谓词的语义被编译期钉死**：`NeedReloc(bool×4)`（`__host_aicore__`，两侧都能求值）在
   `m27_hc_prefill.asc` 里对**全部 16 种配置**逐个 `static_assert`（`M27RelocGate::Ok<MASK>()`）。

**反向验证（零设备，做了）**：把谓词改回"漏 `ijFlat`"的旧语义后重新构建 ⇒ 构建在 `Ok<0x8>()`
上报错（逐字：`error: static assertion failed due to requirement 'M27RelocGate::Ok()': **只传 ijFlat（P2-1 点名的配置）⇒ 必须进 pass**`），
改回正确谓词后构建干净 ⇒ **这组断言不是空洞的**（构建失败 = 门限漂移，而不是等设备上跑出错数）。

**覆盖该配置的设备档（已就位，本轮未跑）**：`M27_ONLY_IJFLAT=1` ⇒ `reloc_mask = 0x8`
（只把 ij handoff 面当重定位目标；其余三个传 `nullptr`），case 元数据里登记 `reloc_mask`；
判据侧按该掩码把 `blk`/`injw`/`rstd` 的判定项记 **SKIP（附"reloc_mask 未当目标"的原因）**，
同时新增一条 guard：**被判 SKIP 的平铺面必须仍是毒值**（否则"SKIP"会掩盖"面被写坏"⇒ 变空洞档）。
跑法：`flock -w 180 … M27_ONLY_IJFLAT=1 M27_CASES=m33 <bin>`（预期：`a1.ij_handoff` 绿
= 修好的判别性那一半；`blk/injw/rstd` 记 SKIP + 毒值见证）。
**本轮读数状态**：探锁返回忙（`flock -n` 未取到）⇒ 按设备纪律**未取得读数**（不写成"没复现"），
登记为解冻后的残留项（§7）。

### 4.9 复审 r1 的 P2-3（`blk` 未达被复用门限）：**举证**（零设备数值实验 + M58 先例），结论分两半

工具 = `tools/localize_blk_deviation.py`（分段链）+ `tools/ulp_envelope.py`（可达性实验），
读数 = `evidence/check_m33_stages.log`、`evidence/ulp_envelope.log`。**没有改门限**（塔的要求）。

**实验设计（零设备）**：同一批输入、同一批**文档写明的 bf16 落盘点**，算两条参考链：
`A` = `m20` 的 fp64 参考（被复用门限所依据的那条）；
`B` = 同一批落盘点 + **fp32 的 K-分块累加**（每 64 元素一块，块内 fp32、块间 fp32 相加 —— 对应设备
`Mmad(k=64)` 打进 fp32 L0C 的形态）⇒ `B` = 「**任何**按这批落盘点取整的实现」给出的下界。

| 档 | `B`（理想同位取整实现）`blk` | `D`（设备）`blk` |
|---|---|---|
| m1 a1 / a2 | 逐位 1.0000 良态 1.0000 ulpMax **0** | 逐位 1.0000 良态 1.0000 ulpMax **0**（设备在同档**逐位**一致） |
| m9 a2 | 0.9951 / 0.9971 / ulpMax **1** | 0.9842 / 0.9929 / ulpMax 2（判据过） |
| m33 a1 | 0.9992 / 0.9998 / ulpMax **1** | 0.9956 / 0.9995 / ulpMax 1（判据过） |
| m33 a2 | 0.9933 / **0.9969** / ulpMax **4** | 0.9751 / **0.9883** / ulpMax 4（**未过**） |
| m64 a2 | 0.9962 / **0.9983** / ulpMax **4** | 0.9798 / **0.9903** / ulpMax 4（**未过**） |

**结论（分两半，按证据）**：
- **`ulpMax ≤ 2` 这条子门限在这条链深 + 这批输入分布上不是「实现质量」的判据**（支持登记一次
  口径变更）：`B` 本身在 m33_a2 / m64_a2 上已到 **ulpMax 4**（良态逐位率 0.9969 / 0.9983）——
  也就是说**任何**只在这批落盘点取整的实现都会在某个良态元素上超过 2 ulp。
  ⇒ **提出**（不是单方面改）：把该子门限换成"良态 ulp 的**分位数**"（例如"良态元素里 ulp>2 的占比
  ≤1e-3"，本组合实测 B 为 `0.0000`、D 为 `0.0001`）或按链深标定 ulpMax（本组合取 4），
  并**保留**「归一化绝对误差 ≤1e-2」与「良态逐位率 ≥99%」两条；这是**判据口径的变更** ⇒ 交复审/塔复核。
- **但余差**不能**整个**归因于链深（所以这 6 条**保持未过**）：同一条 `良态逐位率 ≥99%` 在 `B` 上
  可达（0.9969 / 0.9983），而设备实测 0.9883 / 0.9903 ⇒ 设备与"理想同位取整实现"之间仍有
  **0.8–0.9 个百分点**的差距，这部分**未归因**；分段链（`xn` 0.9999 → `lora` 0.9987 → `ls` 0.9963 →
  `blk` 0.9883）显示它沿链累积，但 `gate` 在 arena 里读不到（`d_clobber=1`）。
  **限度保留**：本档案未归档设备侧错误报告 ⇒ **不能排除设备异常**；下一步要什么写在 §7。

**M58 先例（逐字引，出处 `m15_layer_loop/README.md:489-490` 的 M58-9 第 2 条）**：
> 「**m=33 档上有 11/64 条 numpy 判定项越过 m20 的保守门限 —— 未归因**。
> 读数：`maxulp` 达 3（门限 ≤2）、良态逐位率最低 0.9891（门限 ≥0.99）、逐位率最低 0.9690，
> 集中在 `A2.*`（边界 #2）与 `B整层 blk_mlp/injection`；**m=1 与 m=2 档全部逐位一致**。」

⇒ 与本次**同一门限、同一形态**（同一 `m20.judge`、越界项同样集中在**边界 #2 与 blk**、
同样 m=1/m=2 逐位一致）；M58 的**处置**是"如实披露、未归因"、**没有**重标定门限
（其归档读数见 `m15_layer_loop/evidence/m58_check_hc_ref_run.log` 的 m=33 档）。

## 5. 融合清单（`docs/15 §M103-2.7` 的六项）

### (a) 挂载点

`m15_layer_loop/m15_layer_kernel.h` 的 §3d（`M15L_PrefillBody<KIND>`，在**本分支 tip `f14a839`（base = main `caf00b8`）**
上的行号：函数体 `:593`、两个相位注释位 `:598`-`:601`、既有的 `FLAG_HC0_BOUND_AIV` 调用点 `:756`）里已留的两个相位注释位：

```cpp
//   M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>();   /* B5 的 hc prefill（行分块） */
//   M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>();   /* B4 的 MoE prefill */
```

- **相位 H1**（= `attn_hc` 边界，KIND_ATTN 层）：在**相位 A 之后、子层段之前**。
- **相位 H2**（= `mlp_hc` 边界）：在**子层段之后、相位 B（MoE）之前**。
- 两侧都必须在 `M15L_PhaseBoundaryAiv<...>()`（全体 AIV 的 mode-0 barrier + `PipeBarrier<PIPE_ALL>`）
  **之后**进入段体（段体自己以 arena 为 ws，要求上一段对本段没有任何未完成的 UB/L1/L0C 引用）。
- 消费/生产的 GM 平面：

| 边界 | 消费 | 产出 |
|---|---|---|
| H1（attn_hc） | `hcH`（[m,HYPER] 平铺）、`hcBo`（[m,HID]）、`hcIj`（ijStride）、`hcAttn*` 权重 | arena0 的 H' 平铺面（= H2 的 `hIn`，**零拷贝**）、`blk0`（= 子层段的输入）、**ij handoff 面**（= 它自己的 `OH[:,320:324)` 经块末重定位得到的平铺 `[m,IJ_STRIDE]` 面，= H2 的 `ij`）、`injw0/rstd0` 证据 |
| ⚠ **mode MIX 的例外** | 同上 | **H' 不物化**（donor 的 W0/S1 跳过）⇒ 此时 arena 的 HCP 区**不是有效平面**（它的靠后部分会被更早块的 scratch 覆盖；只有前 `WS_XN/TILE_STRIDE` 块的行没被碰过）。消费方必须取 `hcH`（= decode 的 `hcpFromWs=false` 分支同口径）；本路把"前 8 块的 HCP 区仍是零"当判别性见证用 |
| H2（mlp_hc） | `arena0`（H'）、`hcAttnOut`（bo）、**H1 的 ij handoff 面**（平铺，ijStride = `IJ_STRIDE`）、`hcMlp*` 权重 | arena1 的 H'' 平铺面（= 层出口 / 下一层 `hcH`）、`blk1`（= MoE 的输入）、`injw1/rstd1` 证据（本边界不需要再产出 ij 面） |

### (b) `LayerArgs` 需要的字段（Wave A/C 加）

| 字段 | 类型 | 必填？ | 说明 |
|---|---|---|---|
| `pfHcArena0` / `pfHcArena1` | `__gm__ uint8_t*` | 是 | 行页 arena；字节数 ≥ `M15L::HcPfH::ArenaBytes(m)`（m=4097 时 88,239,104 B ≈ 84.15 MB）。**可不新增**：host 把现成的 `hcWs0`/`hcWs1` 指向这两个 arena 即可（decode 档的 per-layer ws 与 prefill 档的 arena 互斥使用） |
| `pfHcBlk0` / `pfHcBlk1` | `__gm__ uint8_t*` | 是 | 平铺 BLK 平面 `[m, HID]`（子层段 / MoE 的输入）。若消费方愿意读**阻塞**布局，可省（把对应的 `pfHcBlk*` 传 `nullptr`，BLK 就留在 arena 的块槽里）。**注意**：四个重定位目标（blk/injw/rstd/ijFlat）**任一非空就进重定位 pass**（门限的唯一权威 = `NeedReloc`，见 §4.8）；省略某几个只是"那几块平铺面不写"，不影响 ij handoff 面——**"只传 `pfHcIj0`、其余三个留 `nullptr`"是受支持的配置**（`M27_ONLY_IJFLAT=1` 档覆盖它） |
| `pfHcIj0` | `__gm__ uint8_t*` | H2 需要 | **ij handoff 面**：H1 的 `OH[:,320:324)` 经块末重定位得到的平铺面（32 B/行、`(m+M_MAX)*32 B` ≈ 133 KB @m=4097）。H2 把它当 `ij`（`ijStride = IJ_STRIDE`）。**H2 不需要它**（没有再下一个边界） |
| `pfHcInjw0/1`、`pfHcRstd0/1` | `__gm__ uint8_t*` | 否 | 证据平面（`nullptr` = 不重定位）；验证档才需要 |
| `m`、`hcIjStride`、`hcAttnMode`/`hcMlpMode` | 已有 | — | 复用现成字段；**不需要**新增 |

⇒ **最小新增 = 3 个指针**（arenas 复用 `hcWs*`；BLK 两个 + ij handoff 面一个；证据可省）。
   ⚠ **ij 的 handoff 不能靠"直接读 arena 的 OH 列"**：那块区在 arena 里不持久（PROLOGUE ④，
   m=257 档实测只有 22.2% 的行是参考的 OH inj 列）⇒ 必须由本段重定位出一块平铺面。

### (c) 资源窗

| 资源 | 区间 / 峰值 | 新增？ |
|---|---|---|
| UB（AIV） | 既有 hc 窗 `[0, 89,472)`（逐项复算见 §2）+ **`[89,472, 94,592)` 行 staging** ⇒ 峰值 **94,592 B** ≤ 248 KB | **新增 5,120 B** |
| L1（AIC） | `[0, 303,104)`（`L1_PEAK`，沿用 hc 段） | 零新增 |
| L0C | 40,960 B / tile（`L0C_PEAK`） | 零新增 |
| GM | arena ×2 = 2×`ArenaBytes(m)`；BLK ×2 = 2×`m*5120`；ij handoff 面 `(m+M_MAX)*32`；injw `m*128`；rstd `(m+1)*16`（m=4097：168.3 MB + 42 MB + 0.13 MB + 0.5 MB + 0.07 MB） | 新平面（见 (b)） |
| UB/L1/L0C 的 `static_assert` | 全部写在 `m15_hc_prefill.h` 里（峰值 ≤ 248 KB / 512 KB / 256 KB） | — |

### (d) BufferID / flagId 清单

| 类别 | 清单 | 说明 |
|---|---|---|
| AIV BufferID | `BUF_AIV_RELOC = 17`（**新增 1 个**）；其余 `0..16` 全部沿用 hc 段既有登记 | AIV 峰值 **18 个（0..17）** ≤ 28；Wave A 登记的 prefill 窗口 `PF_BUF_AIV_* = 0..2`（它自己的行搬运）与本段**编号重叠**，但两者不在同一个 kernel 里并存（本段是 prefill 的另一条臂）⇒ Wave C 合并登记时按 max 记 |
| AIC BufferID | `0..6` 沿用（零新增） | 峰值 7 ≤ 28 |
| 核间 flagId | 核内 AIV mode-0 **新增 id 15**（`FLAG_RELOC`）做"块末重定位"的 barrier（不新造池：15 在 AIV mode-0 的池子里，现由 MoE 段在**另一个相位**用） | 相邻性（本头执行序）：块内 12 → 13 → 14 → **15** → 下一块的 12 ⇒ 相邻对 (14,15)、(15,12) 不同号 ✓（`static_assert` 钉住）；MODE_COMBINE_ONLY 档 donor 只用 12 ⇒ (12,15) ✓。融合档：（15）→ 相位边界 8 → MoE 段首个 AIV mode-0（12，见 `FLAG_SEQ` 表）✓ 亦不同号 |
| flag 用量 | 每块 AIV mode-0 用 12/13/14/15 各一次 ⇒ m=4097 档（513 块）各约 513 次 | 本仓的 `FlagMaxUse() <= 6` 预算是**无循环段**的。**m=4097 档已实跑通过**（513 块、约 1,500 次跨核 flag 使用，H' 判据逐位一致率 0.9999/良态 1.0000）⇒ 硬件那 4bit 计数器按"在飞（未配平）"计，不是"每次 kernel 累计"（见 §7 第 1 条） |

### (e) 需要的相位边界

1. 相位 A → 相位 H1（hc(attn) 前）：`M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>()`；
2. 相位 H1 → 子层段：同一条边界之后进入子层段（**H1 的产出 `blk0` 是子层段的输入**）；
3. 子层段 → 相位 H2：`M15L_PhaseBoundaryAiv<FLAG_HC1_BOUND_AIV>()`（**M138 订正**：原写
   `FLAG_HC2_BOUND_AIV`，与 `m15_layer_resources.h:592-594` 的登记表不符 —— 表把"子层段→H2"定义为
   `FLAG_HC1_BOUND_AIV = 9`，`FLAG_HC2_BOUND_AIV = 8` 是"H2→B"。若按原写法用 8，则 AIV mode-0 上
   出现 8(`FLAG_HC0_BOUND_AIV`, H1→A) → 8(子层段→H2) **相邻同号**，违反"相邻同步点不同号"）；
4. 相位 H2 → 相位 B（MoE）：`M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>()`（= 8，M40/M58 就按这条复用）。
**段体内部**不需要额外相位边界：块与块之间靠 donor 自己的 mode-0 barrier + mode-2 交接；
"块末重定位"的 barrier（AIV mode-0 id 15）由段体自带，不占用相位边界。

⚠ **已知缺口（M138 登记）：边界 #2 的 `bo` 在 GDN-prefill 路径上未追到生产者。**
本 README §6 补丁 1 把 H2 的 `bo` 取 `A.hcAttnOut`（补丁 1 的第 2 实参 = `m15_layer_kernel.h:776`
的接线）。decode（四相位）路径里 `A.hcAttnOut` 由子层段出口写入（`m15_layer_kernel.h:901`
的 `subOut = A.hcAttnOut`），那时有生产者；但 **prefill 路径的 `KIND_GDN` 子层段是 B1**
（`m15_gdn_prefill.h` 的 `M15GP`），它的出口是 `A.wsO`（`m15_layer_kernel.h:692`
的 `gp.out = A.wsO`）。核对范围 = `m15_layer_kernel.h` 里对 `hcAttnOut` 的写入点与 prefill 的
`gp.out` 赋值：该范围内对 `hcAttnOut` 的写入只读到 decode 路径的 `:901`，prefill 路径上未读到
`hcAttnOut` 的生产者。
两个平面还不同型，不能直接别名：`wsO` 是 `[48, m, 128] fp32`（每 v-head 的 o，
`m15_layer_kernel.h:263`），`hcAttnOut` 是 `[m, HID=2560] bf16`（单流子层出口，
`m15_layer_kernel.h:150`）。
**出错条件**：`pfWired != 0` ∧ `KIND_GDN` ∧ H2 的 `hcMlpMode != MODE_MIX` 时，donor 的 combine
段会读 `bo`（`m15_hc_layer.h:463` 的 `useCombine = (p.mode != MODE_MIX)`；S1 读 `boGm`，
`m15_hc_layer.h:441` / `:676`）⇒ H2 读到的是未定义/陈旧值，且**不报错**（静默错值）。
**处置**：接 Wave D 的 host 真指针前必须先给这条生产者（补 B1 出口 → `hcAttnOut` 的那一段：
o_proj / RMSNormGated / 残差），或由塔裁定 `wsO` 与 `hcAttnOut` 的规范平面关系；本段体
（`m15_hc_prefill.h`）只按 `Plan.bo` 消费，不代上层裁定来源。M138 只登记，未改挂载点
（`m15_layer_kernel.h` 归 M136）。

### (f) `m` / `pos` 语义

- `m ∈ [1, ∞)`（本段不设上界；**4097 是验收档、不是接口假设** —— 正文 §8.3 第 3 条同口径）；
  `m = 0` 时调用方应**整段跳过**（段体对 `m == 0` 直接 `Trap`，响亮失败）。
- **无位置依赖**：hc 边界是逐 token 的（无 RoPE / 无 KV / 无 causal），段体不接受也不需要 `pos`。
- `mode` 取 decode 的同一张表：`MODE_MIX`(0) / `MODE_COMBINE_MIX`(1) / `MODE_FINAL_MIX`(2) /
  `MODE_COMBINE_ONLY`(3)；**`MODE_COMBINE_ONLY` 的 AIC 直接返回**（没有任何 GEMM、没有 mode-2 交接），
  AIV 只跑 W0+S1（`m1_conly` 档实测这条路径）。
- 块高 `MT` 是**编译期**常量（缺省 8；`-DM15_HC_PF_MT=12` 另编一档），host 与设备同源（同一个头）。

## 6. 挂载点补丁文本（可直接用；**本 mission 不改 `m15_layer_kernel.h`**）

把下面两段分别贴进 `m15_layer_loop/m15_layer_kernel.h`（Wave C 做）：

```cpp
// ============================================================
// 【补丁 1】新增：hc prefill 的单边界入口（放在 `namespace PF {` 之后、`M15L_PrefillPhaseA` 之前）
//   段体签名只吃「指针 + 标量」（Wave B 共同契约第 1 条）⇒ 这里从 LayerArgs 取指针组装 Plan。
//   两个边界共用它（`which = 0/1`）；`LayerArgs` 的 5 个新字段见 m27 README §5(b)。
// ============================================================
__aicore__ inline void M15L_HcPrefillBoundary(uint32_t which, const LayerArgs& A)
{
    const bool second = (which != 0u);
    // 边界 #1：hIn = 层输入 H，bo = pending BO，ij = 独立平面，ws = arena0
    // 边界 #2：hIn = arena0（边界 #1 的 H' 平铺面，**零拷贝**），bo = 子层出口，
    //          ij = arena0 + OH 列（行距 OH_W、按块推进 TILE_STRIDE），ws = arena1
    // 边界 #1 的 ij = 层给的独立平面；边界 #2 的 ij = **边界 #1 重定位出来的平铺 ij 面**。
    // 两个边界的 ij 都是"平铺面 + ijStride = IJ_STRIDE" ⇒ 同一条代码路径（不再有 OH 列特例）。
    M15H::HcPF::Plan p = M15H::HcPF::MakePlan(
        second ? A.pfHcArena0 : A.hcH,                       // hIn
        second ? A.hcAttnOut : A.hcBo,                       // bo
        second ? A.pfHcIj0 : A.hcIj,                         // ij
        second ? A.hcMlpDown : A.hcAttnDown, second ? A.hcMlpInj : A.hcAttnInj,
        second ? A.hcMlpUp : A.hcAttnUp, second ? A.hcMlpNorm : A.hcAttnNorm,
        second ? A.pfHcArena1 : A.pfHcArena0,                // arena
        second ? A.pfHcBlk1 : A.pfHcBlk0,                    // blk（平铺；nullptr ⇒ 留在 arena 阻塞布局）
        second ? A.pfHcInjw1 : A.pfHcInjw0, second ? A.pfHcRstd1 : A.pfHcRstd0,
        second ? nullptr : A.pfHcIj0,                        // ijFlat（边界 #1 才产出）
        M15H::HcPF::FlatHInTileStride(), M15H::HcPF::FlatBoTileStride(),
        M15H::HcPF::FlatIjTileStride(M15H::IJ_STRIDE), M15H::IJ_STRIDE, A.m,
        second ? A.hcMlpMode : A.hcAttnMode);
    M15H::HcPF::Body(p);
}
```

```cpp
// ============================================================
// 【补丁 2】§3d 的 `M15L_PrefillBody<KIND>`：把两个注释位换成真调用
//   （`A.pfWired != 0` 才走；缺省 0 ⇒ decode 零回归，与 Wave A 的开关同款）
// ============================================================
    if (A.pfWired != 0u) {
        M15L_HcPrefillBoundary(0u, A);            // 相位 H1（attn_hc）
        M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>();   // = 8：H1 → 子层段
        // ... 子层段（B2/B3 或 B1/GDN）落在这里 ...
        M15L_PhaseBoundaryAiv<FLAG_HC1_BOUND_AIV>();   // = 9：子层段 → 相位 H2（M138 订正，原写 HC2/8）
        M15L_HcPrefillBoundary(1u, A);            // 相位 H2（mlp_hc）
        M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>();   // = 8：相位 H2 → 相位 B（MoE）
        // ... 相位 B（B4 的 MoE）落在这里 ...
    }
```

host 侧（`m15_layer_loop.asc`）：

```cpp
// 两个 arena（按 m 定尺；m=4097 时各 ≈84.15 MB）
size_t arenaBytes = M15L::HcPfH::ArenaBytesAligned(m);
void* arena0 = nullptr; void* arena1 = nullptr;
aclrtMalloc(&arena0, arenaBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&arena1, arenaBytes, ACL_MEM_MALLOC_HUGE_FIRST);
// BLK 平铺面（两个边界各一块；相位严格串行时也可以共用一块）+ ij handoff 面（边界 #1 产出）
size_t blkBytes = M15L::HcPfH::HidPlaneBytes(m);
size_t ijFlatBytes = M15L::HcPfH::IjFlatPlaneBytes(m);   // (m + M15H::M_MAX) * 32 B
// 校验（响亮失败，别等设备上跑出错数）
std::string err = M15L::HcPfH::ValidateGeom(M15L::HcPfH::GeomFlat(m, mode, ijStride), arenaBytes, true,
                                            blkBytes, injwBytes, rstdBytes);
```

> **arena 的 `memset` 不是必须**：本段的"未写"区域不会被当成有效值读（H' 除外 —— mode ≠ MIX 时
> 每个块都写满自己那 MT 行；mode = MIX 时不物化 H'，此时消费方本来就该用 `hcH`）。判据侧把
> "mode MIX 档 HCP 区全零"当成**判别性见证**用，所以 host 在验证档建议 `aclrtMemset(arena, 0, …)`。

## 7. 未完成 / 未验证（窄写，逐条给判据）

0. **`blk` 判定项在大 m 档有边际越门限（未归因，如实披露）**：m=4097 档 `blk` 的读数
   （a1：逐位 0.9938 / 良态 0.9977 / ulpMax **5** / 归一 maxAbs 2.98e-03；a2：0.9806 / 0.9905 /
   ulpMax **6** / 6.58e-03）—— 三个子门限里只有 **ulpMax ≤ 2** 越界（逐位率与归一化误差都在门限内，
   归一化误差甚至比门限低 3 倍），m ≤ 257 档同项在门限内（ulpMax 0–1）。
   **性质（有实测支撑的部分）**：`tools/localize_blk_deviation.py` 把**能读的** stage 从 arena 里
   抽出来与参考逐位比（m=33 档，`evidence/check_m33_stages.log`）：
   `xn` 逐位 0.9999 / 良态 1.0000（ulpMax 1）→ `lora`(OH[:,:320]) 0.9959 / 0.9987（1）→
   `ls` 0.9957 / **0.9963**（2）→ `blk` 0.9751 / **0.9883**（4）；同一批的 a1 链是
   0.9999 → 0.9960 → 0.9965 → 0.9956（ulpMax 全 ≤1，`blk` 通过）。
   ⇒ 沿链**单调累积**（逐位率逐段下降、ulpMax 1→1→2→4），不是"某一段突然大面积错"
   （寻址错的形态是后者）。**未定位到的那一步**：`gate`（up GEMM 输出）在 arena 里**读不到**
   —— 它的区被**下一块**的 OH/LS 写覆盖（`d_clobber = 1`），块内 dump 才能读（本 mission 未做）。
   同类读数在 M58 的 m=33 档已披露过
   （`m15_layer_loop/README.md` M58-9 第 2 条：11/64 项越门限、maxulp 3、良态逐位率 0.9891），
   那边的结论也是"门限是在别的输入分布上校准的、未归因"。
   **本条不宣称 `blk` 达门限**：要主张"通过"需要先把 m20 的门限在 prefill 输入分布上重标定
   （本 mission 不做，因为那会动到别的 mission 的判据口径）。
   **后续若要收口这条**：① 给段体加"块内 dump 中间量"（`gate`）的验证档，把余差定位到具体 stage；
   ② 或在 m=4097 的真实输入分布上重标定门限（需要 tower 对判据口径的裁决）。

1. **跨核 flag 用量的硬件语义：已由 m=4097 档证伪"累计"说**（判据见 §8 的 evidence）。
   本段的块循环让 `(AIV, mode0)` 的 12/13/14/15 各用约 513 次（合计 ≈1,500 次跨核 flag 使用）。
   本仓的 `FlagMaxUse() <= 6` 预算是给无循环段写的，所以这曾是一条真风险：若硬件计数器按"每次
   kernel 累计"计，513 块根本跑不完。**实测：m=4097（513 块）H' 判据通过**（逐位率 0.9999、
   良态 1.0000、ulpMax=1、归一化误差 3.7e-03）⇒ 计数器按**未配平/在飞**计。**未覆盖**：单核
   一次 kernel 内 > 某阈值的次数的上界仍未标定（本档只到 ~513 次/id）。
2. **四个待跑的设备档**（都代码就绪、本轮**未取得读数**：探锁 `flock -n` 返回忙）：
   (d) `M27_MUTANT=sinkreloc`（**重定位落点旁路**：掩码保持 `0xF`、被消费的四个平面留毒值）——
   它的**判据半**零设备已复现（`evidence/negative_control_matrix.log` 第 3 步：10 条值级红），
   **设备半**（落点真被旁路 ⇒ 平面真的留毒值）待解冻。这条是"重定位路径的值级负向对照"的新落点（见 §4.4 的判别力账）。
   (a) `M27_MUTANT=ijfromoh`（第三个负向对照：A2 的 ij 退回读 arena 的 OH 列）；
   (b) `M27_ONLY_IJFLAT=1`（P2-1 点名的配置：只把 ij handoff 面当重定位目标 ⇒ 见 §4.8 的预期读数）；
   (c) ~~`m=4097` 的逐 case 元数据重生成~~ —— **本轮已用零设备手段解决**（`tools/rebuild_case_meta.py`
   从设备写的布局文件重建 + 复判，读数与归档同值，见 §4.6/§8）；仍存的是"那批二进制 sha 未钉"。
   **旧条**：`M27_MUTANT=ijfromoh`（第三个负向对照）代码就绪但**未跑**：设备冻结（§4.6）⇒ 本 mission
   不补跑。它的作用是复算"改前"读数（A2 的 ij 直接读 arena 的 OH 列），跑法：
   `flock … M27_MUTANT=ijfromoh M27_CASES=m33,m257 <bin>`（预期：`a2.*` 四条判定项变红、
   `a1.ij_handoff` 保持绿 —— 后者是"红的是 ij 源而不是重定位本身"的判别性那一半）。
   同一趟里还应重生成 m=4097 的 `m27_case_m4097*.txt`（元数据格式在 10-04 改过）以便判据覆盖真 shape 档。
3. **没有实测的性能数字**：本路只做正确性（`m27` 不打点、不落 msprof）。分母口径的算术
   （每层 2 个边界 × 513 块 × 每块 5 段 + 2 GEMM）见 `docs/15 §6`；**本 mission 不宣称吞吐**。
4. **`M15L::HcPfH::ArenaBytes(0)` 的下溢**（复审 r1 的纯文字第 3 条）：已**加固** ——
   `m == 0` 单独返回 0，并加两条 `static_assert`（`ArenaBytes(0)==0`、`ArenaBytes(1)==WS_BYTES`）；
   `Body` / `ValidateGeom` 仍先拒 `m=0`。原形态（`TileCount(0) - 1` 在 uint32 上下溢）不再可达。
5. **`blk` 的阻塞布局路径未作为判据的另一臂**：`blk=nullptr` 时 BLK 留在 arena 的块槽里
   （阻塞布局），本路的 `norel` 负向对照只证明了"平铺平面保持毒值 ⇒ 判据对缺失重定位敏感"；
   **没有**单独写一个"按阻塞布局读 BLK 并与参考对齐"的正判据（`arena` 差分见证只覆盖
   末 d_blk 块 —— 那几块的阻塞内容确实等于参考，但那是 guard，不计入判定项）。
6. **`MODE_FINAL_MIX`(2) 档未跑**：它是"全局 mixer"（无 injection 列），prefill 里由层循环之外的
   独立入口出；本路的档表里没有它（`mode >= MODE_COUNT` 的响亮失败有 `Trap` 兜底）。
7. **`m15_hc_host.h` 的权重装配路径未复用**：为了自成工程，m27 自带一份**最小**权重装载
   （manifest 只读 + 合成两档 + `inj` 补 16 行）。融合时仍应走既有的 `M15HcW`/`H_HcWAlloc`
   （本 mission 不复制它的槽装配逻辑，只复制了"补 16 行、行 [4,16) 置零"这一条 donor 契约）。
8. **目录名与权威文档不一致**（`m27_hc_prefill/` vs §M103-2.6 的 `m15_hc_prefill/**`）：
   按塔的任务书 scope 落地，已在 §2 登记。

## 8. 实测结果（`evidence/` 下的日志为准）

**设备**：Ascend950PR，AIC=28 / AIV=56（API 取得）；块几何：MT=8、`TILE_STRIDE`=163,840、`WS_BYTES`=4,353,024、
`UB_PEAK_PF`=94,592 ≤ 248 KB。

主档 = **真实 checkpoint 权重**（layer 0 的 `attn_hc_*` 给边界 #1、`mlp_hc_*` 给边界 #2），9 个 case（含 m=4097）。

> **口径分开（复审 r1 的 P2-2 的处置）**：原本 m=4097 两行只有 09-27 的历史读数、且那份 dump
> 的逐 case 元数据缺失（`check_ref.py` 按 `m27_case_*.txt` glob ⇒ 那份 dump 判不出真 shape 档）。
> **本轮（零设备）已把它变成可复算**：`tools/rebuild_case_meta.py` 从**同一次设备运行写的**
> `m27_layout_m4097*.txt` 重建元数据（字段来源逐条打印，不发明任何读数）⇒ 对幸存的 dump 重跑判据，
> 9 个 case 的读数与 09-27 归档逐条**同值**（`evidence/check_main_1004_rebuilt_meta.log` 对
> `check_main.log`：hcp 逐位 1.0000 / blk 0.9977 ulp 5 / ij_handoff 0.9983 … 全部一致）。
> **仍存的限度**：那一批的**二进制 sha 未钉**（该构建已被后续构建覆盖；10-04 的改动只落在
> host 侧的元数据/实参装配与编译期断言上，**设备侧代码同源**——这是同值的原因，但属推断而非哈希证据）。
> 10-04 的两档（小档）与本次三档（负向对照）的源码/二进制/dump 的 sha256 已入库（`evidence/sha256.txt`）。

| 档 | m | tiles | `hcp`(H') | `blk` | `ij_handoff` | `injw` | `rstd` |
|---|---|---|---|---|---|---|---|
| m1 / m1_mix / m1_conly | 1 | 1 | **1.0000 逐位**（MIX 档不物化 ⇒ SKIP） | **1.0000 逐位**（CONLY 档不产出 ⇒ SKIP） | **1.0000 逐位**（MIX/CONLY ⇒ SKIP） | 6.3e-08（MIX ⇒ SKIP） | 6.3e-08（CONLY ⇒ SKIP） |
| m9 | 9 | 2 | 0.9999 / 良态 1.0000 | 0.9842 / 良态 0.9929（ulpMax 2，a2） | 1.0000 逐位 | 1.0e-07 | 8.0e-08 |
| m33 | 33 | 5 | 0.9999 / 1.0000 | a1 0.9956 / 0.9995 ✓；**a2 0.9751 / 0.9883 ✗** | 0.9924 / 1.0000 | 1.0e-07 | 1.5e-05 |
| m64 | 64 | 8 | 0.9999 / 1.0000 | a1 0.9955 / 0.9989 ✓；**a2 0.9798 / 0.9903 ✗** | 0.9883 / 0.9955 | 1.3e-07 | 1.9e-05 |
| m257 | 257 | 33 | 0.9999 / 1.0000 | a1 0.9916 / 0.9981 ✓；**a2 0.9809 / 0.9906 ✗** | 0.9942 / 0.9977 | 1.3e-07 | 4.5e-05 |
| **m4097** | 4097 | **513** | **0.9999 / 1.0000（a1）· 1.0000 / 1.0000（a2）** | a1 **0.9938 / 0.9977（ulpMax 5）✗**；a2 **0.9807 / 0.9905（ulpMax 9）✗** | 0.9968 / 0.9983 | 1.3e-07 | 2.4e-05 |
| m4097_mix | 4097 | 513 | 不物化 ⇒ SKIP | **0.9964 / 0.9989（ulpMax 4）✗** | — | SKIP | 1.2e-07 |

计数（主档）：**判定项 60 条**（本脚本 27 + 复用 `m20.judge` 的 33）、**guard 53 条**、SKIPPED 12 条 ⇒
**未过的 6 条全部是 `blk`**（见 §7 第 0 条：三个子门限里只有 `ulpMax ≤ 2` 越界，逐位率与归一化误差都在门限内）。

**结构性 guard 全绿**（`m257` 档的 arena 见证，逐字节）：
末 8 块的 arena 阻塞 `blk` == 设备平铺面 ✓（`d_blk=8`）；末 8 块的 arena 阻塞 `rstd` == 设备平铺面 ✓（`d_rstd=8`）；
更早块的阻塞 `blk` ≠ 设备平铺面 ✓（⇒ 覆盖真的发生、**块末重定位承担负载**）；
更早块的 arena `OH` 列 ≠ ij handoff 面 ✓（⇒ **OH 列也必须块末重定位**，`d_oh=8`）。

**MT=12 档**（`-DM15_HC_PF_MT=12`，合成权重，m=1/33/257）：除 `m257_a2.rstd`（maxRel 2.1e-04 vs 门限 1e-04，
合成权重下 `var` 偏小 ⇒ `eps` 主导、相对误差放大）外全过 —— 这是"块循环不是只对 8 调出来的"的运行期见证。

**flag 用量语义**（CLAUDE 里列为风险的那条）：m=4097 档一块用 4 个 AIV mode-0 id、共 513 块
⇒ 每 id ≈ 513 次、合计 ≈ 1,500 次跨核 flag 使用，**H' 判据仍逐位通过** ⇒ 硬件计数器按"未配平/在飞"计。

**判据链路自检**（不占设备，`tools/selftest_dump.py`）：用参考生成"理想设备输出" ⇒ 判据 rc=0
（9 判定项 + 5 guard）；分别弄坏 `a1.ij_handoff` / `a1.hcp` / `a2.blk` / `a2.rstd` ⇒ 各 rc=1。

## 9. 复现与证据

**分批跑**（设备纪律要求"一次 `flock` 只跑一条短命令 ≤120s"）：每个 case 写**自己**的元数据文件
`m27_case_<名>.txt`，判据侧 glob 它 ⇒ 分几批跑、补跑某几档都不会互相抹掉元数据
（早先用"全局清单 + 每次调用截断"的写法，分批跑会把前一批的元数据删掉 —— 实测栽过一次）。

| 文件 | 内容 |
|---|---|
| `reproduce.sh` | 一条命令：判据自检 → 构建 → 设备跑（flock 槽）→ 判据 → 两个负向对照 |
| `tools/rebuild_case_meta.py` | 从**设备写的**布局文件重建旧格式 dump 的 case 元数据（P2-2 的补救；字段来源逐条打印） |
| `tools/ulp_envelope.py` | P2-3 的举证工具：同落盘点 + fp32 分块累加的"可达性下界"（零设备） |
| `evidence/check_main_1004_rebuilt_meta.log` + `evidence/m27_case_m4097*.txt` | 上面那条重建 + 复判的读数（与 09-27 归档逐条同值） |
| `evidence/ulp_envelope.log` | P2-3 的可达性实验读数（m1/m9/m33/m64 × 两个边界） |
| `tools/fingerprint.sh` | 证据指纹：被测源码 / 判据源码（含 read-only 依赖 `m20_hyperconn/check_ref.py`）/ 二进制 / dump 产物的 sha256 |
| `evidence/sha256.txt` | 上面那条命令在**入库时**的读数（"这次读数属于哪个二进制、哪个判据"的钉子） |
| `evidence/check_mutant_rowsteal.log` / `check_mutant_norel.log` | 两个负向对照档的判据日志 + host 侧自检行 |
| `evidence/device_run_main.log` | 主档设备日志（核数、块几何、权重来源、逐 case 的 host 自检三/四条） |
| `evidence/device_side_selfcheck.log` | 上面那份的 `[m27]` 行摘要 |
| `evidence/check_main.log` | 主档判据日志（09-27，9 个 case 含 m=4097；每条判定项的逐位率 / ulp / 归一化误差 + guard + 计数 + 三态） |
| `evidence/check_main_1004_small.log` | 10-04 复跑的小档（m=1..257，7 个 case）判据日志 —— **不含 m=4097**（那批用了改元数据格式之前的二进制 ⇒ `m27_case_m4097*.txt` 未生成；真 shape 档的读数见上一行） |
| `evidence/check_mt12.log` | MT=12 档（第二个块高）判据日志 |
| `evidence/layout_m257.txt` | 布局文件（判据侧据此算设备地址；由 `.asc` 用同一批常量写出） |
| `evidence/selftest_check.log` | 判据链路自检（正向 rc=0 / 弄坏 `a1.ij_handoff` rc=1；不占设备） |
| `evidence/negative_control_empty.log` | 无 dump 时的三态对照（rc=2 SKIPPED） |
| `evidence/negative_control_matrix.log` | **逐档"谁保证 rc"矩阵**（`tools/negative_control_matrix.sh` 生成；含每步 rc、SKIP 条数与红项清单）。**M125 重生成**：该脚本第 3 节 echo 里的反引号原先未转义（shell 当命令替换 ⇒ 打印 `sinkreloc: command not found`，且该节标题缺 `sinkreloc`），已改为转义写法；除这一行输出外与修前逐字节相同，修前捕获见下一行 |
| `evidence/negative_control_matrix_before_fix.log` | **修前捕获**（M125；含 `sinkreloc: command not found` 报错行与残缺标题 `### 3)  的**判据半**…`），是上一行"重生成"的对照证据：去掉那条 stderr 报错行后，与重生成后的日志只差第 3 节标题那一行 |
| `evidence/m125_citation_check.py` | **M125（r2 复审 B1′）**：`docs/17` §9.7–§9.11 触发事件「引文 ↔ 出处」的**清单 + 复核器**（清单与判定在同一份文件里）。清单 = 每张表里对源文加了「」引文的那段文字（17 条；§9.7 的表不含「」引文，故不计）。判定按三栏：**N 逐字连续子串 / M 去 `**`、空白、`/` 后归一化吻合 / K 已显式标注省略或脱敏**（`MISMATCH` 应为 0）。在仓库根跑 `python3 m27_hc_prefill/evidence/m125_citation_check.py`：本 mission 读数 **N=9 / M=4 / K=4 / MISMATCH=0**，rc=0；源文件缺失时照 §8.3 记 `SKIPPED`、rc=2 |
| `evidence/check_mutant_rowsteal.log` | `rowsteal` 档判据日志（设备档；10 条值级红，见 §4.4） |
| `evidence/check_mutant_norel.log` | `norel` 档判据日志（设备档）—— ⚠ **改动前的历史读数**：它是**掩码引入之前**的二进制/判据产物（当时判据不按掩码 SKIP ⇒ 36 条红，其中 22 条是四个产物的判定项）。当前代码跑同一命令得到的是 §4.4 的"现形 norel"（SKIP + 毒值见证）；两者都保留、口径分别标注 |
| （附带限度）两个 mutant 档的 dump-sha | ⚠ 复审 r2 指出：`rowsteal` / `norel` 两档的 **dump 已不在磁盘**（/tmp 被清）⇒ `evidence/sha256.txt` 只覆盖 `/tmp/m27_out`；这两份**日志**的读数因此没有 dump-sha 钉子（日志本身入库、生成命令在 §4.4 与 `tools/negative_control_matrix.sh`） |
