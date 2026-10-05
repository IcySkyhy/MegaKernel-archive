# M110（Wave A）—— prefill 的**契约与挂载点**（交付清单）

> **本 mission 只交契约与挂载点，不交任何段体实现**（B1–B5 是后续并行 mission，`docs/15` M103-2.2）。
> 完成定义（塔给的判据）= **B1–B5 的实现者能否只靠这里的东西把自己的段接进来**：
> 资源常量 + 段接口 + 挂载点字段 + **默认关闭的编译期开关**。读数次之。
>
> 权威清单来源：`docs/15` 的 `## M103 重盘（@20bd20d）` 一节（M103-2.1 Wave A / M103-2.3 Wave C /
> M103-3 形状）+ `docs/19` §4.4（G2 缝函数 + `GdnStateHome`）+ M101 复审 r1 的 F2（mode-4 /
> `FLAG_PER_CORE`）。**写码之前的接口面清单**在 `interface_face.md`（它的 commit 早于源改动）。

本目录的其它文件：
- `interface_face.md` —— 任务 1 的交付物：写码之前逐条列出的 `文件:符号` + 理由。
- `run_evidence.sh` —— 全部读数的一键复跑（含设备锁与 `npu-smi` 巡检）。
- `m101_flag_call_points.log` —— M101（未合入分支）35 个 cross-core 调用点的逐个点名（命令 + 原样输出）。
- `*.log` —— 读数（文件名见 §6）；`binary_sha256.txt` 记录产生这些读数的二进制。

---

## 1. 交付物一览（4 个源文件 + 本目录）

| 文件 | 加了什么 | 权威锚点 |
|---|---|---|
| `m15_layer_loop/m15_layer_resources.h` | MoE 权重槽的**两套定尺**（decode 原值不动 + prefill E=512 新表）、prefill 的峰值断言位、**mode-4 分节 + `FLAG_PER_CORE` 裁决**、prefill 的 flagId/BufferID 分节、**入口登记表** | M103-2.1 ①–⑤、A-2；M101-F2 |
| `m15_layer_loop/m15_layer_kernel.h` | `LayerArgs` 的 **13 个 prefill 字段**（追加在 PLE 之后）、`M15L_LAYER_PREFILL_ARGS_DECL/FILL`、**相位 A 挂载点**（行分派网格）、**B1 的缝函数契约表 + `GdnStateHome`**、两个真 `__global__` 入口 | M103-2.1、M103-2.7、docs/19 §4.4 |
| `m15_layer_loop/m15_layer_loop.asc` | `runs=prefill` 档（`Opts`/`Ctx`/`H_Alloc`/`H_RunPrefill` + 分派），夹具写在 `.asc` 内 | 任务 3、M103-2.7 第 3 条 |
| `m15_layer_loop/m15_loop_layout.h` | 激活平面按 **m=4097** 定尺（4 个平面 + 手算式 + `static_assert`） | M103-2.1 Wave A 表末行、M103-3 |

## 2. 契约表 —— 资源常量（B1–B5 消费）

### 2.1 MoE 权重槽（`MW_*` / `MOE_W_STRIDE` 的**唯一 owner = `m15_layer_resources.h`**，B4 只读）

| 项 | decode 档（E=4，**数值一字未动**） | prefill 档（E=512 / topk=10） |
|---|---|---|
| 式子 | `MwXxxBytes(E)` 一族（本文件 §1a） | 同一批式子，`E = MOE_E_PREFILL = 512` |
| `WGU_BYTES`（gate_up） | 6,553,600 | **838,860,800**（= 128×，与 `docs/15` M103-2.1 的 A-2 读数一致） |
| `ROUTER_BYTES` | 20,480 | 2,621,440 |
| `SGU_BYTES` | 409,600 | 52,428,800 |
| `WDN_BYTES` | 3,276,800 | 419,430,400 |
| `SDN_BYTES` | 204,800 | 26,214,400 |
| 共享专家 4 项（`*SHD`） | 1,638,400 / 102,400 / 819,200 / 51,200 | **同值**（与 E 无关） |
| 每层槽 `stride` | **13,091,840** | **1,342,182,400** |
| 48 层合计 | 628,408,320 B（≈0.63 GB） | **64,424,755,200 B（≈64.42 GB）** |
| 相对倍数 | — | **102.52×**（注意：128× 只在 `WGU` 那一项上成立） |

- 每个数值都有 `static_assert`（decode 档 13,091,840 本仓 README 的资源表读数一致；prefill 档
  838,860,800 / 1,342,182,400 与 `docs/15` 的 A-2 逐字一致）。
- **topk=10 不进权重槽**（槽是逐专家的 `[E, …]`）⇒ 它影响的是 `M15M::TOTAL_MAX` 与 `SZ_*` 一族，
  落点在 B4 的 `m15_moe_prefill_res.h`；本文件把这条**显式登记**（§1c）并给跨文件见证
  `static_assert(M15M::RT_ROWL >= 512)`（M107 已核过一次）。
- **host arena 的账**：prefill 档 48 层 = 64.42 GB > 本机 cgroup 上限 34,359,738,368 B（32 GiB）
  ⇒ host 侧必须改（分片/惰性装载）。**本 mission 不动 host**，见 §5 的依赖清单。

### 2.2 prefill 的峰值断言位（数值由 Wave C 填）

`PF_{UB,L1,L0C}_*` 共 **8 个槽**，全部是哨兵 `PF_PEAK_UNSET = 0xFFFFFFFF`，装在
`PF_PEAK_TABLE[]` 里（`what / bytes / budget`）。判据是**合取**而不是"填 0 即通过"：

- `PfPeaksOk()`：**已填的**必须在预算内（UB 248 KB / L1 512 KB / L0C 256 KB）；
- `static_assert(!(PREFILL_WIRED != 0 && PfPeakUnfilled() != 0))`：**要把 prefill 接线打开，就必须先把
  8 个槽填齐**，填漏一个编不过。
- 当前读数：`PfPeakUnfilled() == 8`（本 mission 只交断言位，见 §6 的 `Pf.peak` guard）。

### 2.3 flagId 分节

| 分节 | 内容 | 判据 |
|---|---|---|
| §4a-1 **`FLAG_PER_CORE` 裁决** | `FLAG_PER_CORE` 保留 **16** 作 mode 0/1/2 的每核池；新增 `FlagIdLimit(core, mode)`：**仅 `AIC ∧ mode==4`** 放行到 **32**；`FlagSeqAdjacentOk()` 改用它 | `static_assert` 见证四条：mode0/2 仍拒 16（AIC 与 AIV 各一条）、mode4 的 AIC 放行 32、**mode4 的 AIV 仍是 16** |
| §4c **mode-4 分节**（attention core） | M101 的 **35 个 cross-core 调用点**逐个登记（AIC 21 / AIV 14），含 **mode-4 的 AIC 高号 16/17/21/22/23**（`ccMM + AIV_CH`）；mode2 的 9/10/11 **重号到 4/6/7** | `FlagSeqAttnCoreCalls() == 35`、`…Of("AIC") == 21`、`…Of("AIV") == 14`、id+覆盖必须在 `FlagIdLimit` 内、重号落进 `{4..7}`、不与前端 `AP_A2V_GEMM` 撞号 |
| §4d **prefill 分节** | **B1**：6 个 mode-2 同步点，id 0..5（`M15G::GP_FLAG_GO1..DONE3`，与 `m15_gdn_resources.h:396-401` 同源）；**B2/B3**：6 个（前端 2 + core 4，core 用 §4c 的重号）。**订正（M138 实核 / M142 同步）**：原写「B1 复用 decode GDN 的 **12 个**同步点」是误记 —— 那 12 个（mode2 4-7 + AIV mode0 8-11 + AIC mode0 12-15）是 **decode GDN 段**的号，B1 段体（`m15_gdn_prefill.h`）实际只用 `GP_FLAG_GO1..DONE3` 这 6 个 mode-2 号 | **复用见证**：每一个 `(核型, mode, id)` 都必须在 decode 的 `FLAG_SEQ[]` 里已登记（不新增号）—— B1 的 0..3 落在 decode 的 MoE mode2、4..5 落在 decode 的 GDN mode2，且 6 行**全部 mode2**（`FlagSeqPrefillGdnAllMode2()`）；B2/B3 的 mode2 ⊂ {4..7}、AIC mode0 落进 hc 的 0..3 或 MoE 的 8..11。**核对范围** = `m15_gdn_prefill.h` 的全部 `CcSet`/`CcWait` 调用点（`:571-585`、`:945-1069`）—— 该范围内未读到 mode-0 调用（M142 复核：这 12 个调用点都取默认模板参 `GP_CC_MODE2`、id 全为 `GP_FLAG_*`） |

**`FLAG_PER_CORE` 的两个选项与后果**（选 A，依据写在头文件里）：
- **A（采纳）**：保留 16 + mode 相关的 `FlagIdLimit`。既有全部记录的保证**一字不变**（mode0/2 限仍是 16），
  而 AIC mode-4 的 16..31 合法 ⇒ 解掉 B3 的硬阻塞。代价：检查器变 mode 相关，"16" 的含义要读成
  「mode0/1/2 每核 16；mode4 的 AIC 侧 32」。
- **B（否）**：全局抬到 32 ⇒ AIV 或 mode0/2 上的 id 16..31 也会被**接受**，而硬件池只有 0..15
  ⇒ 断言失去抓「真会挂死」那一类 bug 的能力（M2 实证过同核连续 set 不保序）。

**要 B3 自己承担的一条使用契约**（不是断言，写在 §4d 末尾）：mode-4 与 mode0/2 在 AIV 侧**共用同一个
物理池** ⇒ 用 mode4 之前必须保证 mode0/2 的 set/wait 全部 drain。B3 的进入点（相位 A 开头）恰在
全体 AIV 的 mode-0 边界之后，这条前提在那里成立；**若 B3 改成与前端交替使用 mode2/mode4，必须重做**。

### 2.4 BufferID 分节

`BUF_TABLE[]` 新增 4 行（GDN-PF / ATN-PF 各 AIV、AIC 一对），并给：`PF_BUF_AIV_{ROW0,ROW1,POS}`、
`PF_BUF_{AIV,AIC}_{LO,HI}`、`BUF_{AIV,AIC}_PEAK_PF`。断言：三个行 staging id 互异且顺排、
窗宽与峰值一致、窗宽 ≤ 每核 28。**段体自带的 BufferID 归 Wave C 打平**（本节只钉"落在同一批 0..27 里"）。

### 2.5 入口登记表（§6b，`ENTRY_TABLE[]`）

9 个入口逐项登记 `sym / kind / nPhase / hc / prefill / wired / note`；两个 prefill 项
`prefill=1 & wired=0`。断言：条数 9、符号名互异（constexpr 字符串比较）、**prefill 档 wired 缺省必须为 0**、
`PREFILL_WIRED == 0`。host 侧另有 `strcmp` 核对（见 §6 的 `Pf.entry` 两条判定项）。

## 3. 契约表 —— 挂载点与段接口（B1–B5 照它接）

### 3.1 两个真入口与开关

| 符号 | `kind` | 相位数 | `wired` 缺省 | 语义 |
|---|---|---|---|---|
| `m15_layer_kernel_gdn_prefill` | gdn | 1（相位 A） | 0 | 相位 A = **挂载点**（B1 落点） |
| `m15_layer_kernel_attn_prefill` | attn | 1（相位 A） | 0 | 相位 A = **挂载点**（B2+B3 落点） |

- 实参表 = 基础 **30** 实参（与两相位入口逐字相同）+ **13** 个 prefill 项（`M15L_LAYER_PREFILL_ARGS_*`）。
- **默认关闭**：`M15_PREFILL_WIRE=0`（与 `M15L::PREFILL_WIRED` 由一条 `static_assert` 钉在一起）。
- `wired == 0`：相位 A = **结构性占位**（逐行 x→y；**数学未实现，显式标注**）。
- `wired != 0`：进"段体挂载点"分支，**段体未落地 ⇒ 什么都不写** ⇒ 判据必红（"响亮失败"）。

### 3.2 `LayerArgs` 的 13 个 prefill 字段（追加在 PLE 字段之后，**未重排任何既有字段**）

| 字段 | 语义 | 给定方 |
|---|---|---|
| `pfKv` / `pfComp` / `pfRing` / `pfPack` | 三套 cache + packed indices 的基址（布局权威 = `m15_attn_kv.h`） | host |
| `pfPos` | per-row 位置表 `u32[m]`（可空） | host |
| `pfPosBase` | 单请求连续时的起点（`pos(r) = pfPosBase + r`） | host |
| `pfBlockTable` | 页表（**`nullptr` = 恒等表 / 单请求**） | host |
| `pfCounts` / `pfExpertOffsets` | MoE prefill 的每专家计数 / 前缀和 | host |
| `pfLayerK` | attention 层序号 k（0..11；GDN 层 = `PF_LAYER_K_NONE`） | host |
| `pfStageMask` | 段序截断（bring-up 定位；0 = 全开） | host |
| `pfWired` | 0 = 结构占位；1 = 段体挂载点 | host（来自 `M15_PREFILL_WIRE`） |
| `pfMutant` | 负向对照掩码（只在验证档用；契约档 = 0） | host |

**两条接口裁决**（写在字段注释里，B1–B5 照它做）：
1. **不引入 `slot_mapping`**（M103-3 的结论）：所有 KV 地址由 `pos` 经 `M15KV_KV_*` 算出；
   `pfBlockTable == nullptr` **就是**"单请求 / 恒等页表"的可执行表示。将来要支持 paged 多请求，
   **扩展点是 `pfBlockTable`**，不是新加一个数组。
2. **`pos` 二选一**：`pfPos != nullptr` 用表（支持将来 batch>1），否则 `pfPosBase + 行号`；两者同时给以表为准。

### 3.3 相位 A 的行分派网格（挂载点的骨架）

```
for (row = bid; row < m; row += 2 * GetBlockNum()) { M15L_PrefillRowCopy(A, row); }
```

**为什么它是本 mission 的关键交付**：老的 `m15_attn_passthrough_body`（`docs/15` M103-1.2 点名）
是 m=1 定尺 —— 每 AIV 固定 16 个 32B 块、56 个 AIV 只覆盖 896 块 = **5.6 行**，`m ≥ 6` 起**静默丢行且不报错**。
本挂载点用行网格，`m = 4097` 时 4097 行全部被覆盖（读数见 §6 `Pf.gdn` / `Pf.attn`）。

### 3.4 B1 的缝函数契约（`docs/19` §4.4）+ `GdnStateHome`

`m15_layer_kernel.h` §3e 登记 `GDNPF_SEAMS[9]`（`fn / contraction / shape / note`）：
`Scan_Kk` / `Scan_QKt` / `Scan_StateApply` / `Scan_OutInter` / `Scan_OutIntra` / `Scan_StateUpdate`
（**6 条收缩 ⇒ 人类裁决「一律 mmad」逐条命中**）、`Scan_SolveG` / `Scan_Gamma` / `Scan_Scalar`（3 条非收缩）。
断言：`GDNPF_SEAM_N == 9`、收缩数 `== 6`、非收缩数 `== 3`。

- **`GdnStateHome { GDNS_AIV_UB, GDNS_AIC_L1, GDNS_GM }` 进了接口**（`enum` + 说明）。
  理由（`docs/19` §4.4 第 2 张表）：接口只给裸指针不给"状态住哪" ⇒ 交付实现会静默退化成
  「每 chunk 一次 GM 往返」，把最大的结构收益（`docs/10:21` 记的 ~400 MB/层）还回去，**且没有判据会红**。
- **`Scan_SolveG` 的形式写成"待定"**：`docs/19` §10.2 自己标了「请塔确认」，**本 mission 不替塔裁决**。
- ⚠ **条数口径的差异（报塔项）**：塔的补充说「8 个缝函数」，而 `docs/19` §4.4 的缝表是 **9 行**
  （6 收缩 + `Scan_SolveG` + `Scan_Gamma` + `Scan_Scalar`）。本 mission **按 `docs/19` 原文的 9 行登记**，
  不静默取一个数字。若塔的原意是 8（例如把 `Scan_Gamma` 与 `Scan_Scalar` 合成一条），改 `GDNPF_SEAMS[]`
  与那条 `static_assert` 即可，**不影响其它契约**。

### 3.5 段体接进来时必须交的"融合清单"（`docs/15` M103-2.7 第 2 条，Wave A 不填）

(a) 挂载点：接进哪个相位 / 哪个入口符号、消费与生产哪些 GM 平面；
(b) `LayerArgs` 需要的字段（本 mission 已把 M103-3 列的都给了，若还缺请回报，**不要改 `LayerArgs`**）；
(c) 资源窗：UB 字节区间、L1 区、L0C tile + 各自峰值（填进 §2.2 的 8 个槽）；
(d) BufferID 清单 + flagId 清单 `(核型, mode, id, pipe)`（供 Wave C 打平）；
(e) 需要相位边界的位置；(f) `m` / `pos` 语义。

### 3.6 flagId **相邻性**的覆盖与不覆盖（r1 复审 P2-1 的答复；`m15_layer_resources.h` §4e）

r1 复审抓出的缺口成立：`§4c/§4d` 先前只有**容量 / 位置 / 复用**三类断言，**没有**「同一
(核型, mode) 的相邻同步点不得同号」这一条 —— 而任务 2 的原话要它，且 `FlagSeqAdjacentOk()`
**只遍历 decode 档的 `FLAG_SEQ[]`**，三张新表不在它的判据范围内。本轮按**塔的选项 (a)** 落地：

| 表 | 粒度 | 判据 | 读数 |
|---|---|---|---|
| `FLAG_SEQ_PREFILL_GDN[6]`（§4d） | 逻辑同步点（一行 = 一次 set 配一次 wait） | `FlagRefsAdjacentOk()`：相邻逻辑点异号 + id 上限 | 通过 |
| `FLAG_SEQ_PREFILL_ATTN[6]`（§4d） | 同上 | 同上 | 通过 |
| `FLAG_SEQ_ATTN_CORE[18]`（§4c） | **事件级**（一行 = 一次 set 或一次 wait，带 count 与 parity 覆盖 `cov`） | `FlagEventsAdjacentOk()`：**相邻两次 `set` 异号** + `id+cov` 上限；另 `FlagEventsMaxSetUse() ≤ 15` | 通过（最大 set 次数 = 4） |

- **为什么必须分两个粒度**：逻辑点级判据套到事件级表上会**假红**（`(AIC,0,8,set)` 与
  `(AIC,0,8,wait)` 是**配对**，不是相邻同步点）；而硬件真正的危险面是**相邻两次 `set` 同号**
  （M2 实证：同核连续 set 不保序）⇒ 事件表只比 set、按 (核型, mode) 分组。
- **正负两侧都在代码里**（`m15_layer_resources.h` §4e 末尾）：`PF_ADJ_{NEG,POS}` / `PF_EV_{NEG,POS}`
  四张对照表 + 四条 `static_assert`（负例必须被拒、正例必须通过）。**外加一次独立实跑**：
  把**真实表**里前端与 core 那两条相邻 AIC mode0 逻辑点人为弄成同号 ⇒ 编译期
  `static_assert(FlagRefsAdjacentOk(FLAG_SEQ_PREFILL_ATTN, …))` **FAIL、二进制不产出**（rc=2）；
  改回后 `diff` 与备份逐字节相同。命令与逐字输出见 `r1_neg_adjacency.log`。
- **一处事实订正**（复审该条的**前提**）：复审写「前端 `AP_AIC_M0_OUT` 与 core `CC_BAR` 都是 8」。
  实测**不是** —— `M15AP::AP_AIC_M0_OUT = 1`（`m15_attn_prolog.h:218`，与 hc 的 `FLAG_AC1` 同号）、
  `ATTN_CORE_M0_AIC_BAR = 8`（本文件 §4c）⇒ 相邻逻辑点是 `1 → 8`，**本来就满足**相邻性判据。
  已把这条**可执行化**：`static_assert(AP_AIC_M0_OUT != ATTN_CORE_M0_AIC_BAR)` ——
  M15AP 将来若改成 8，它会立刻编译期变红（而不是等人手推）。
- **明确不覆盖的部分**（不沉默）：
  1. **decode 档的 `FLAG_SEQ[]`** 仍由原来的 `FlagSeqAdjacentOk()` 管，本 mission **一个字没改它**
     （零回归；负例仍能抓红 —— r1 复审自己验过：往 `FLAG_SEQ[]` 塞 `AIV/mode0/id16` ⇒ 编译失败）。
  2. **没有做「sets 数 == waits 数」的 drain 见证**：mode2 的配对本来就是 **n:m** 语义
     （`m15_attn_core.h` 头注写明 `CC_AIVDONE` 是"2 set 配 1 wait"）⇒ 这条见证会**假红**。
     故只保留 §4d 末尾那一段「用 mode4 之前必须 drain mode0/2」的**使用契约**。
  3. **跨段合并成一条 48 层执行序再比一遍**（decode 档 + prefill 档 + hc/MoE 的相位插桩）不在
     本 mission —— `docs/15` M103-2.3 把「flagId 相邻性 + 每 id 用量」的收口算在 **Wave C**。


## 4. `m15_loop_layout.h`：激活平面按 m 定尺

| 常量 | 值 | 说明 |
|---|---|---|
| `M_PREFILL` | 4097 | 单序列 prefill 上界（= `M15Kv::PREFILL_M`） |
| `H_ROWS_BYTES_PF` | **20,976,640** = 4097×2560×2 | 对比 decode 档 `H_ROWS_BYTES` = 327,680（**64.02 倍**） |
| `PF_QGATE_BYTES` | **100,687,872** = 4097×12288×2 | q\|gate 平面 |
| `PF_QKVZBA_BYTES` | **135,037,120** = 4097×16480×2 | GDN in_proj 输出平面 |
| `PF_OPIN_BYTES` | **50,343,936** = 4097×6144×2 | out_proj 输入平面 |

三个平面另有"行数必须是 4097"的 `static_assert`（把"不得留缩形档"变成可编译期核对的东西）。
`ws` / `hcWs` 的 prefill 定尺**不在本 mission**（它们的尺寸来自各段自己的 `*_resources.h`，属 B4/B5/Wave C）。

## 5. 显式未完成项与依赖（本 mission **不**做的事）

| 未做 | 归属 | 依赖 |
|---|---|---|
| **N1（`subOut` 的生产者）** —— `docs/15` M103-2.1 原归 **Wave A**，**塔裁 2026-09-27 改为随 Wave C（相位 B 接线）落地** | **Wave C**（不阻塞 Wave B） | 依赖 **M101 的 attention 核心 + `o_proj` 合入**（M101 r2 在飞）；核内的登记点在 `m15_layer_kernel.h` §3d 的挂载点说明里 |
| B1 GDN prefill 段体（6 个 mmad 收缩 + `Scan_SolveG` 等） | Wave B1（新文件 `m15_gdn_prefill*`） | 缝表 + `GdnStateHome`（已给）；`Scan_SolveG` 的形式**待塔裁决** |
| B2 attention 前端 + cache 填 | Wave B2 | M103-2.2 B2 的 scope；**N1 是它的交付义务**（接相位 B 前必须先给 `subOut` 生产者） |
| B3 稠密 causal core（FA，m-tile） | Wave B3 | **mode-4 分节 + 相邻性判据（已给）** + §4d 的 drain 前提 |
| B4 MoE prefill（AIC 打平 / router 多核 / `active_num` 定尺） | Wave B4 | `MWP_*` / `MOE_W_PREFILL_STRIDE`（已给，只读） |
| B5 hc prefill（行分块） | Wave B5 | `UB_IWTAB_SLOTS` 墙的处置 |
| 峰值实数 + 打平收口 + 段体接进融合 TU | Wave C | §2.2 的 8 个槽必须填齐才能开 `PREFILL_WIRED` |
| flagId 相邻性 / 每 id 用量的**跨段合并收口**（decode 档 + prefill 档 + 相位插桩） | Wave C（`docs/15` M103-2.3 原文如此） | 本 mission 已把三张新表的**段内**相邻性 + 用量落成断言，见 §3.6 |
| host 接线（`m15_chain_host.h` / `m15_moe_host.h` / `m15_hc_host.h` 的 m 传参、H2D 装载、dump 口径） | Wave D | 本 mission 未动这三个文件 |
| **E=512 的 host arena 切换**（分片/惰性装载 + manifest 重切 + `NUM_EXPERTS/TOPK_MAX`） | **清单外的依赖 mission** | 见 §2.1 的账；**已 `TowerSend` 报塔** |
| aclgraph / 性能 | 不在本 mission | — |
| `Scan_SolveG` 是否算"矩阵乘法" | 塔 | `docs/19` §10.4 自己标了"请塔确认" |

**N1 的四点登记**（塔要求的四项，逐条）：
1. **原归属**：`docs/15` M103-2.1 的 `m15_layer_kernel.h` 那行第 ③ 条 =「处理 N1（`subOut` 的生产者）」
   ⇒ 原归 **Wave A**；本 mission 未实现它（r1 复审指出的"既没处理也没登记"= 本节的补登）。
2. **塔裁后的归属**：**随 Wave C（相位 B 接线）落地**；不在 Wave B 的阻塞面上。
3. **依赖**：**M101 的 attention 核心 + `o_proj` 合入**（M101 r2 在飞）——`subOut` 的生产者只能是它们。
4. **不处理会怎样**：`A.apW != nullptr` 时 `M15L_FusedBody` 走 attention 臂、**不再写 `subOut`**
   ⇒ 子层出口**没有生产者**，而下游 MoE 照跑、**不报错、只数值错**（M97
   `m15_layer_loop/evidence/attn_wire/README.md` §5 第 8 项记的形态）。
   **本档不构成活的错误面**（两个 prefill 入口只有相位 A，之后 kernel 就结束）；一旦 B2/B3 把
   attention 臂放进 `M15L_PrefillPhaseA<KIND_ATTN>`、Wave C 又接上相位 B，这条静默错就会复活
   ⇒ **接相位 B 的前一条 commit 必须先给 `subOut` 生产者**。


**另一个 scope 限制（先声明）**：本 mission 的 scope 只有 4 个源文件 + 本目录，**没有新头文件的位置**
⇒ 任务里那句「include 新的夹具头」实现为「夹具直接写在 `m15_layer_loop.asc` 里」（`.asc` 在 scope 内，
不新增文件 = 不越界）。若塔要独立夹具头，请扩 scope。

## 6. 读数（全部实跑；命令与输出在 `*.log`）

命令（一条不落，见 `run_evidence.sh`；`flock` 内的每一步之前都查过 `npu-smi`）：

```bash
cd <repo>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release && cmake --build m15_layer_loop/build -j16
flock -w 900 /tmp/npu0.lock bash m15_layer_loop/evidence/prefill_contract/run_evidence.sh ./m15_layer_loop/build/m15_layer_loop new
```

| 日志 | 档 / env | 读数 |
|---|---|---|
| `new_pf_default.log` | `runs=prefill`（`M15_PREFILL_WIRE=0`，m 列表 `{1, 4097}`） | 见下面 §6.1 |
| `new_pf_smallm.log` | `runs=prefill M15_PREFILL_M=64,65`（跨包络边界） | 见下面 §6.1 |
| `new_pf_mutant.log` | `runs=prefill M15_PREFILL_MUTANT=1`（负向对照） | 见下面 §6.1 |
| `new_pf_wired.log` | `runs=prefill M15_PREFILL_WIRE=1`（开关语义） | 见下面 §6.1 |
| `new_plewire_body.log` | `runs=plewire M15_PLE_WIRE=1 M15_PLE_STAGE=15` | 见 §6.2 |
| `new_plewire_full.log` | `runs=plewire M15_PLE_WIRE=1 M15_PLE_STAGE=31` | 见 §6.2 |
| `new_plewire_off.log` | `runs=plewire M15_PLE_WIRE=0 M15_PLE_STAGE=15` | 见 §6.2 |
| `new_all48.log` | `runs=all M15_DUMP=1`（与基线**同一套 env/argv**） | 见 §6.3 |

### 6.1 prefill 档（本 mission 的主读数）

二进制 sha256 = `4a22c225a36b4302c230994bce608aeb44910fcea2c6ab0ae438184a11a127d9`（`binary_sha256.txt`）。

> **M136 更新（M142 同步）**：下表是 **M110 当时**的结构占位判据（段体未落地，`Pf.gdn` = 逐行 `x→y`）；
> 这份历史读数与二进制保持原样（不改写历史）。自 **M136** 起，GDN 相位 A 的段体真产出已接上，
> 判据升级为 **段体真产出 + `m23_gdn_prefill/check_ref.py` numpy 对拍 + `Pf.gdn_mut1` 负向对照必须变红**；
> 当前读数与覆盖范围（只到 **KIND_GDN 相位 A**、q/k/v/g/β 为宿主合成输入）见
> `m15_layer_loop/evidence/m136_prefill_gdn/README.md`。

`new_pf_default.log`（`M15_PREFILL_WIRE=0`，m 列表 `{1, 4097}`）：

| arm | m | 行标记命中 | 整面逐字节相等 | 毒值行 | 判定 |
|---|---|---|---|---|---|
| `Pf.gdn` | 1 | 1/1 | 1/1 | 0 | 3 条 PASS |
| `Pf.attn` | 1 | 1/1 | 1/1 | 0 | 3 条 PASS |
| `Pf.gdn` | **4097** | **4097/4097** | **4097/4097** | 0 | 3 条 PASS |
| `Pf.attn` | **4097** | **4097/4097** | **4097/4097** | 0 | 3 条 PASS |
| `Pf.neg.mutant`（负向对照） | 4097 | 64/4097 | 64/4097 | **4033**（首个未命中行 **64**） | 3 条 PASS（= 预测值） |
| `Pf.neg.wired`（开关语义） | 4097 | 0/4097 | 0/4097 | 4097 | 3 条 PASS（= 预测值） |

合计 `checks=152, guards=58, fails=0, rc=0`（`Pf 20` 条判定项 + 84+48 条权重来源；guard 里 9 条是 Pf 的契约常量）。

- **m=1 与 m=4097 都覆盖 100%**（人类原话「prefill 和decode都要照顾到，测试m=1和4097」）。
- **负向对照恰好 4033 行**（= 4097 − `M15G::M_MAX`64）且首个未命中行 = 64 ⇒ 判据对"行循环被截断"
  这一类坏法**必红**，不是恒真断言（`docs/17` §4 的非空洞性）。
- **`new_pf_smallm.log`**（`M15_PREFILL_M=64,65`，跨 `M_MAX` 包络边界）：m=64 与 m=65 各自 100% 命中；
  同一档的 mutant 恰好 1 行毒值（65 − 64）—— 边界处同样非空洞。
- **`new_pf_wired.log`**（`M15_PREFILL_WIRE=1`）：**FAILURES PRESENT（checks=152, guards=58, fails=1），rc=1**。
  红在哪一条要说准：**红在 guard `M15L::PREFILL_WIRED == C.O.pfWire`**（编译期开关说"接线关"、
  运行期 env 说"接线开" ⇒ 契约自相矛盾），而不是数学判据 —— 因为 `wired=1` 时挂载点的**正确行为**
  本来就是"什么都不写"（段体未落地），3 条判据因此按预期通过。⇒ **想开接线必须同时做两件事**：
  ① 把 §2.2 的 8 个峰值槽填齐（`PREFILL_WIRED=1` 时 `PfPeaksOk` 那条门会拦），② 把编译期
  `PREFILL_WIRED` 置 1 **并**把段体接进来。**只翻 env 拿不到绿**（这正是塔要的"默认关闭"语义）。
  `wired` 档自身的非空洞性来自与 default 档的**对照**：同一二进制、同一 m，default 档 4097/4097、
  wired 档 0/4097 ⇒ 开关**确实改变了执行路径**（不是"内核压根没跑"）。

### 6.2 `runs=plewire` 与 M100 r3 的对账（本 mission 的改动对它零影响）

| 本 mission 的日志 | 档 | 本 mission 读数 | M100 r3 归档读数（`evidence/ple_wire/*.log`） | 对账 |
|---|---|---|---|---|
| `new_plewire_body.log` | `M15_PLE_WIRE=1 M15_PLE_STAGE=15` | checks=142, guards=49, **fails=0**, ALL PASS, rc=0 | `A_body_stage15.log`：142 / 49 / 0 ALL PASS | **逐项相同** |
| `new_plewire_full.log` | `M15_PLE_WIRE=1 M15_PLE_STAGE=31` | checks=142, guards=49, **fails=3**, FAILURES, rc=1 | `B_full_stage31.log`：142 / 49 / 3 | **逐项相同**（M100 自述的 ①→② 落盘可见性问题仍未修，见其 WITNESS.md） |
| `new_plewire_off.log` | `M15_PLE_WIRE=0 M15_PLE_STAGE=15` | checks=142, guards=49, **fails=5**, FAILURES, rc=1 | `C_wire_off.log`：142 / 49 / 5 | **逐项相同**（接线关时的负向对照） |

### 6.3 零回归

| 档 | 本 mission 读数（`new_all48.log`） | 基线读数（`base_all48.log`，base `d348547` 的**未改动**二进制） | 对账 |
|---|---|---|---|
| `runs=all M15_DUMP=1` | **ALL PASS（checks=2095, guards=303, fails=0）** rc=0 | **ALL PASS（checks=2095, guards=303, fails=0）** rc=0 | **逐位相同**（判定项 2095、guard 303、fails 0 三项全同） |

- **基线口径说明（塔要求"若基线变了以你实跑为准并写清"）**：mission 文里写的是
  「2068 + 290 / 0」，那是 **M100 当时的自己分支**读数（`evidence/ple_wire/runs_all.log`：2068/290）。
  本 mission 的 base 是 `d348547`（M100 合入**之后**又陆续合入了 M82/M98/M65 等的判据）⇒ 基线变成
  **2095 / 303 / 0**。**本 mission 的判据因此按"新二进制 vs 未改动基线二进制、同一 env/argv"对账**
  （两行同值），而不是按 mission 文里的 2068+290。
- 另有一条**不跑 dump** 的同档读数（`M15_DUMP=1` 关掉）：`checks=2095, guards=302`——差的 1 条 guard
  是 `m15_chain_host.h` 里"**M39 对拍 dump 已落盘**"那一条（只在 dump 开时计入），**与判定项无关**。
- `runs=prefill` / `runs=plewire` **都不并进 `runs=all`**（各自档跑）：前者起两个 prefill 入口、
  写按 4097 行定尺的平面，与 decode 路径零交集。

### 6.4 其它实跑读数（写文件前的环境）

| 项 | 读数 | 命令 |
|---|---|---|
| 磁盘 | `overlay 8.7T / 已用 8.4T / 可用 313G (97%)` | `df -h /` |
| 设备占用（本批每一步之前） | 空闲（`No running processes found in NPU 0`） | `npu-smi info` |
| 构建 | `cmake --build m15_layer_loop/build -j16` rc=0（17 s） | 见 §6 顶部两条 |
| dump 产物 | `runs=all M15_DUMP=1` 会在**仓库根**落 1300+ 个 `.bin`（0.16 GB）；脚本末尾已清理（`git ls-files \| grep -E '^[^/]+\.bin$'` 在 base 上是 **0**，不会碰到入库文件） | `run_evidence.sh` 末尾 |

---

## 7. r1 复审的答复（本轮：`p2-2items` + 1 nit）

复审认可的部分（入口/字段/资源窗/峰值断言位四类清单**漏项 = 0**、独立 `git archive` 重建后
二进制 sha256 与入库值相同、35 个跨核调用点独立点数一致、负例自己弄红、CANN 依据核到）
**本轮一律未重做、未改动**。下面只列本轮**新增**的三处（全部是**加法**）：

| 复审条目 | 本轮的处置 | 落点 |
|---|---|---|
| **P2-1**（§4c/§4d 没有相邻性断言、且偏离没写明）| 按塔的**选项 (a)** 做出来：两个粒度各一条判据 + 正负对照随代码入库 + 一次独立负例实跑；另**订正该条的前提**（`AP_AIC_M0_OUT=1` ≠ `CC_BAR=8`，并把它写成断言）；"不覆盖什么"逐条写出 | `m15_layer_resources.h` §4e；`README.md` §3.6；`r1_neg_adjacency.log` |
| **P2-2**（N1 / `subOut` 归属未登记）| **不实现**，按塔的四点要求**登记**（原归属 Wave A → 塔裁改 Wave C → 依赖 M101 核心 + `o_proj` → 不处理会静默错） | `README.md` §5（表首行 + 四点说明）、`interface_face.md` §2、`m15_layer_kernel.h` §3d 的挂载点注释 |
| **nit**（`64.06×` / `64.02` 混用）| 统一成 **64.02**（= 4097/64 = 64.0156，同一句里两个数已一致） | `m15_loop_layout.h:102` |

**未改任何已交付的契约、常量或判据**：本轮三处改动都是"加断言 / 加注释 / 加登记"。
可执行的核对 = **二进制 sha256 与 r1 复审时逐字节相同**（`4a22c225…`，见 §8 与
`binary_sha256.txt`）⇒ 段体实现者看到的 ABI、常量值与判据一字未变。

## 8. r2 的读数（本轮重跑，与 r1 同一套命令）

见 `r2_pf_default.log` / `r2_all48.log`（以及 `r1_neg_adjacency.log`）。要点：
- `runs=prefill`（WIRE=0）：`ALL PASS 152/58/0`，m=1 与 m=4097 两个入口都 **4097/4097** 行；
- `runs=all M15_DUMP=1`：`ALL PASS 2095/303/0`（零回归，与基线逐项相同）。

## 9. M142：三条遗留项的收口记录（只改文档/注释，无代码语义改动）

| # | 遗留项来源（会话证据） | 改动前（文件:行） | 改动后 | 权威依据 |
|---|---|---|---|---|
| 1 | M136 r2 复审非阻塞观察（`.tower/comms/reviews/review-feat-m136-wave-d-gdn-prefill-host-wiring-reviewer-m136-r2.md:47`）+ `m15_layer_loop/evidence/m136_prefill_gdn/README.md:92-94`（待同步项） | `m15_layer_loop/README.md` 未记载 prefill「验证 Pf」判据（M142 实测 `grep -n Pf m15_layer_loop/README.md` 输出为空；`git log --all -S "Pf.gdn" -- m15_layer_loop/README.md` 亦空） | 在该 README 末尾**追加** §11「验证 Pf（prefill 判据）」：段体真产出 + `m23_gdn_prefill/check_ref.py` 对拍 + `Pf.gdn_mut1` 必须变红；覆盖范围写窄（只到 KIND_GDN 相位 A） | `m15_layer_loop/evidence/m136_prefill_gdn/README.md`（设备读数 / 对拍 / 限度）；本节 §6.1 的 M110 历史读数保持原样（二进制 `4a22c225…`） |
| 2 | M138 finding（`.tower/comms/findings/20261004-agent-registry-bug-m138-fixed-4d-gdn-prefill-flag-table-12-6-two-prefill-contra.md:16,23-24`） | 本 README `:69`（「B1 复用 decode GDN 的 12 个同步点」）、`:170`（`FLAG_SEQ_PREFILL_GDN[12]`）；`interface_face.md:26`（「复用 decode GDN 的 4-7 / 8-11 / 12-15」） | 三处统一订正为 B1 真实 6 个 mode-2 号（`M15G::GP_FLAG_GO1..DONE3` = 0..5）；注明核对范围 = `m15_gdn_prefill.h` 的 `:571-585`、`:945-1069` | `m15_layer_resources.h` §4d（`:1069-1095`，含 `FLAG_SEQ_PREFILL_GDN_N == 6` 断言）；`m15_gdn_resources.h:396-401` |
| 3 | M139 survey（`.tower/comms/inbox/20261004-agent-ccaudit-tower-survey-summary-m139-crosscore-set-wait.md:64`） | `m19_qsa_indexer/m19_qsa_indexer.asc:492` 注释「全体 barrier ×HANDOFF_ROUNDS」（M142 实测 `grep -rn HANDOFF_ROUNDS`（排除 `.git`/`build`）只命中该注释自身，仓内无此符号） | 改注为「一次 `AivBarrier(FLAG_HANDOFF)`（+ AIV0 校验「整段无哨兵」）」 | `m19_qsa_indexer.asc:924-927`（`VerifiedHandoff` 只调一次 `AivBarrier(FLAG_HANDOFF)`）；改前/改后重建二进制 sha256 相同 |

**核对（M142 实跑）**：`python3 docs/scan_doc_refs.py` rc=0（`RESULT: OK`，18 files / 276 doc refs / 255 section refs / 487 path refs / 0 out-of-range）；`python3 docs/scan_quote_refs.py` rc=1（13 条 FIND，改动前后输出逐字节相同，均为既有项）。三处改动要么同行替换、要么尾部追加 ⇒ **未移动任何被引用的行号**。
