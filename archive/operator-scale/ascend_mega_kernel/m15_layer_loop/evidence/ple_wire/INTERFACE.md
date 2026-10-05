# M100 — PLE 段接进层 kernel 挂载点：接口面盘点（task 1）

> 本文件是 **task 1 的交付物**：先于写码把「挂载点需要什么」从权威处盘清。
> **引用一律用内容锚点（符号/常量名/注释首句），不钉行号** —— 本队第六变体（行号会随别的 mission
> 合入而漂；本文件里凡给出行号处都显式声明基准 commit）。
> 基准 tip（本 mission 的 worktree）：`feat/m100-ple-segment-wiring-into-the-layer-k`，其 base =
> `c9f97e3`（`Merge branch 'feat/m97-attention-phase-wiring-into-the-laye'`）。

---

## 0. 一句话结论

PLE 段在**四相位形态的相位 H1 内部**：H1 被 `LayerArgs::hcPleBreak` 拆成
`combine-only → ①ids → 相位边界 → ②③④⑤ → 相位边界 → mix-only`，PLE 直接**就地**读写 10240 宽
的多流态（`hcWs0 + WS_HCP`）。**不新增任何 flagId、UB 窗在既有相位窗内、manifest 需补 PLE role**
（本 tip 上一条都没有，见 §5）。

---

## 1. 挂载点在四相位里的确切位置（内容锚点）

| 事实 | 权威出处 |
|---|---|
| 四相位序：hc(attn) → 子层段 → hc(mlp) → MoE | `m15_layer_kernel.h` 文件头 `四相位（HC=true，M58 形态）` 段 |
| PLE 打断点把 **H1 拆成三段**，中间夹 PLE 挂载点 | `m15_layer_kernel.h` 文件头 `**M65 的 PLE 打断点**` 段 |
| 挂载点函数（M65 是 `M15L_PlePhasePlaceholder`，**M100 换成 `M15L_PleIds` / `M15L_PleBody`**） | `m15_layer_kernel.h` §3b（`// 3b. PLE 段的**挂载点**`） |
| 打断点的语义来源（先物化 combine、PLE 之后再 mix） | `m15_layer_kernel.h` §3b 引的 `if prev_block_output is not None:` 伪码（= `V-N:model.py:290-301`） |
| 两条相位边界（AIV 全体 mode-0）已登记、id 已定 | `m15_layer_resources.h::FLAG_PLE_IN_BOUND_AIV` / `FLAG_PLE_OUT_BOUND_AIV` |
| combine-only / mix-only 两档 mode 的 device 实现 | `m15_hc_resources.h::MODE_COMBINE_ONLY`（=3）/ `MODE_MIX`（=0） |
| H1 三段在 AIV/AIC 两侧的调用点 | `m15_layer_kernel.h::M15L_FusedBody` 的 `if (A.hcPleBreak != 0u)` 两处（AIV 段：`hc0.ProcessAiv()` 之后；AIC 段：`hc0.ProcessAic()` 之后） |
| 边界 #2 的输入**必须**取 `hcWs0+WS_HCP`（mix-only 不再蕴含 H'≡层输入） | `m15_layer_kernel.h::M15L_FillHcPtrs` 的 `hcpFromWs` 那一行 |
| 启动器（host 侧唯一入口） | `m15_hc_host.h::H_LaunchLayerHc`（`pleBreak` 是它的第 10 个形参） |
| 链上的调用点 | `m15_chain_host.h::H_LaunchChainLayer` → `H_LaunchLayerHc`；`pleBreak = (L == 1u) ? C.O.chainPle : 0u`（`H_ChRunOnce` 内） |
| 手工对照路径里层 1 的三段序列（combine-only → 单次 combine_and_mix 参考 → mix-only） | `m15_chain_host.h::H_ChManualLayer` 的 `else if (L == 1u)` 分支 |
| 「层 1 = 0-based 1」「PLE 只挂这一层」 | `ple/PLE_SPEC.md` §5 末（`ple_layer_ids=[2]` 是 1-based）；`slice_layer_manifest.py::PLE_LAYER` |

**插入点的精确读法**：`M15L_FusedBody` 的 AIV 段里，层 1（`hcPleBreak != 0`）依次执行
`hc0.ProcessAiv()`（MODE_COMBINE_ONLY，只 W0+S1，把 bf16 `H'` 物化到 `hcWs0+WS_HCP`）→
`PipeBarrier<PIPE_ALL>` → **①** → `FLAG_PLE_IN_BOUND_AIV` → **②③④⑤** → `PipeBarrier` →
`FLAG_PLE_OUT_BOUND_AIV` → `hcMix.ProcessAiv()`（MODE_MIX，读物化的 `H'`）→ …。AIC 段在这一段里
只做自己的 `hc0.ProcessAic()`（combine-only 立即返回）与 `hcMix.ProcessAic()`，**不参与 PLE**
（PLE 的实现只有 AIV 段，见 `m15_ple.asc::m15_ple_body_kernel` 的 `if ASCEND_IS_AIV`）。

---

## 2. `LayerArgs` 需要哪些字段（M100 新增的 17 个）

全部追加在结构体**末尾**，两相位入口（`m15_layer_kernel_{gdn,attn}`）保持 `nullptr/0`，
kernel 内以 `if (A.pleW != nullptr)` 守卫 ⇒ 既有入口一字未动。

| 字段 | 语义 | 尺寸/来源 |
|---|---|---|
| `pleW` | 权重 slab：`wcat[12800,2560]` \| `wtap[4,10240]` \| `nk` \| `nq` \| `ncw` | `M15L::PLEW::W_BYTES` = 65,679,360 B |
| `pleScr` | scratch slab：ids/emb/kv/gated/normed/sidx/fail(①/②)/exp64/hmBad/stIn/stOut | `M15L::PLEW::S_BYTES` = 28,325,120 B |
| `pleTable` | **host-mapped 注册窗口**的 device 指针（M92 的 ACL 链路） | `aclrtHostGetDevicePointer` 的返回值 |
| `pleIds` / `pleQsl` / `pleCtx` | [T] / [nReq+1] / [nReq,2] int32（① 的输入） | host 提供 |
| `pleM` / `pleSz` / `pleOf` | [3] / [16] / [16] int64（`layer_multipliers` / `vocab_sizes` / `offsets`） | checkpoint（`ple_lm`/`ple_sz`/`ple_of` role）或缩减词表 |
| `pleTok` / `pleNReq` | T / 请求数 | host |
| `pleTableRows` | 窗口行数（`SetGlobalBuffer` 的上界） | host（= 窗口行数） |
| `pleStageMask` | bit4 = 跑 ①；bit0..3 = 跑 ②③④⑤（**必须是前缀**，M85 的契约） | host |
| `pleNegMask` | 原样透传给 `M85P::*` 的变异/负向掩码（契约档 0） | host |
| `pleWinBase` / `pleWinRows` | 窗口第 0 行的**全局行 id** / 行数（0 = 关窗口模式） | host（M92 §2.3） |
| `pleStateSlots` | short-conv 状态槽数（`stBase = slot*STLEN*HYPER`） | host |

**为什么是 slab 而不是 20 多个指针**：kernel 参数表已经 ~50 个实参，再加 28 个会逼近启动参数的
尺寸上限；把「权重 5 块 / scratch 12 块」折成 2 个基址 + **host/device 同一批编译期偏移常量**
（`M15L::PLEW::W_*` / `S_*`），既省参数又正好落在人类裁定「所有 buffer 的地址自管、尽量编译期
静态分配」上。

**宏的同步**（任务要求「两个宏」）：
- `M15L_LAYER_ARGS_FILL`：新增 9 个指针置 `nullptr` + 8 个 u32 置 0（两相位入口）；
- `M15L_LAYER_HC_ARGS_DECL` / `M15L_LAYER_HC_ARGS_FILL`：末尾追加这 17 个形参/赋值。

---

## 3. `H_LaunchLayerHc` 的调用面

- 形参**不变**（不新增入参）：PLE 的 17 个实参由 `H_PleArgsOf(C, layer, pleBreak)` 在**函数内部**
  组装 —— 它读 `Ctx` 的 9 个 PLE 平面指针 + `Opts` 的 6 个开关；条件不满足（该层不是 PLE 层 /
  `M15_PLE_WIRE=0` / 平面未分配）时返回**整组空** ⇒ kernel 内空操作。
- 两个 launch 点（GDN / attention 入口）末尾都追加 `M15_PLE_LAUNCH_ARGS(pa)`。
- 负向对照的「实参错位」落在同一个函数里：`M15_PLE_MISWIRE=1` ⇒ `pa.table = pa.scr`（表基址接错）；
  `=2` ⇒ `pa.winBase + 1`（行基址错位）。

---

## 4. buffer id / cross-core id 预算表（塔要求「新增同步点前先拿预算表」）

### 4.1 cross-core flagId：**不新增任何号**

| 同步点 | 核/mode | id | 登记处 | 相邻性 |
|---|---|---|---|---|
| H1a（combine-only）出口 | AIV mode0 | 12 | `M15H::FLAG_AV0` | — |
| `FLAG_PLE_IN_BOUND_AIV`（物化 H' + ①ids 落盘 → PLE 段） | AIV mode0 | 8 | `m15_layer_resources.h`（M65 已登记） | 12→8 ✓ |
| ②→③ | AIV mode0 | 12 | `M85P::FLAG_B2` | 8→12 ✓ |
| ③→④ | AIV mode0 | 13 | `M85P::FLAG_B3` | 12→13 ✓ |
| ④→⑤ | AIV mode0 | 14 | `M85P::FLAG_B4` | 13→14 ✓ |
| `FLAG_PLE_OUT_BOUND_AIV`（PLE 段 → mix-only） | AIV mode0 | 9 | `m15_layer_resources.h`（M65 已登记） | 14→9 ✓ |
| H1b（mix-only）出口 | AIV mode0 | 12 | `M15H::FLAG_AV0` | 9→12 ✓ |

**①→② 的跨核可见性不另开号**：① 放在 `FLAG_PLE_IN_BOUND_AIV` **之前**跑，那条已登记的 mode-0
barrier 同时承担「① 的 ids 已落 GM」。

层 1 的完整 AIV mode0 执行序（逐对核过相邻性，全部两两不同）：

```
12(hc H1a) 8(PLE_IN) 12(B2) 13(B3) 14(B4) 9(PLE_OUT) 12/13/14(hc H1b)
8(H1→A) 10/8/9/11(GDN) 9(A→H2) 12/13/14(hc H2) 8(H2→B) 12/13/14/15/12(MoE)
```

每 (AIV, mode0, id) 在一次 kernel 内的使用次数估算：`8→4`、`9→3`、`12→6`、`13/14→4`
（硬件 4 bit 计数上限 15）⇒ 不需要给 `FLAG_SEQ` 加行。

**`m15_layer_kernel.h` 里的编译期断言**（把上表的相邻对钉住；`M15L::PLEW` 之后）：
`FLAG_PLE_IN_BOUND_AIV != M85P::FLAG_B2` / `B2 != B3` / `B3 != B4` / `B4 != FLAG_PLE_OUT_BOUND_AIV` /
`FLAG_PLE_OUT_BOUND_AIV != M15H::FLAG_AV0`。

**未做**：`m15_layer_resources.h::FLAG_SEQ` 的注释里没有 PLE 段内的 `12/13/14` 三行（M65 时 PLE
是空操作）。把这三行补进权威登记表要动 `m15_layer_resources.h`（**scope 外**）⇒ 列为未完成项。

### 4.2 核内 buffer id：**不新增**

PLE 段沿用 M85 的实现，核内同步**只有 `PipeBarrier<PIPE_ALL>`**，**没有任何 BufferID**
（`m15_ple.asc` 文件头 PROLOGUE 的「核内同步」两条即是权威表述）。⇒ 不存在 BufferID 预算冲突。

### 4.3 UB 预算

`M85P::UB_END` = 52,512 B < `M15L::UB_PEAK_FUSED`（hc 相位 `M15H::UB_PEAK`）。
段间同址叠放的合法性：PLE 段两侧各有一条相位边界 + `PipeBarrier<PIPE_ALL>`，与 hc/GDN/MoE
三段之间同一论证（`m15_layer_resources.h §2` 的「峰值 = max(各相位)，不是求和」）。
→ 在 `m15_layer_kernel.h` 补 `static_assert(M85P::UB_END <= UB_PEAK_FUSED)`（**scope 内**，
效果等于给 §2 表加一行 PLE）；**§2 权威表本身加行要动 `m15_layer_resources.h`** ⇒ 未完成项。

---

## 5. manifest role：**本 tip 上一条 PLE role 都没有**（塔的简报这处与事实不符）

取证（在 base `c9f97e3` 上）：

```
$ grep -c " role=ple" m15_layer_loop/weights_manifest.txt      → 0
$ grep -n ple m15_layer_loop/slice_layer_manifest.py           → 无命中
```

M100 追加（`slice_layer_manifest.py::PLE_ROLES` / `PLE_TABLE_*`，**纯追加在文件尾**）：

| role | checkpoint 张量 | 形状 | 用途 |
|---|---|---|---|
| `ple_key_proj` | `…layers.1.ple.key_proj.weight` | [10240,2560] BF16 | ③（与 value **分开**，checkpoint 无 `kv_proj`） |
| `ple_value_proj` | `…layers.1.ple.value_proj.weight` | [2560,2560] | ③ |
| `ple_conv1d` | `…layers.1.ple.conv1d.weight` | [10240,1,4] | ⑤（host 侧转成 tap-major `wtap[k][c]`） |
| `ple_norm_key` / `ple_norm_query` / `ple_norm_conv` | 同名 | [10240] | ④ 的 `(1+w)` |
| `ple_lm` / `ple_sz` / `ple_of` | `…ple.ple_embedding.{layer_multipliers,ngram_heads_vocab_sizes,ngram_heads_offsets}` | I64 [3]/[16]/[16] | ① |
| `ple_tbl_shard_000` … `ple_tbl_shard_127` | `…ple.ple_embedding.ngram_embedding.shard_k.weight` | [2500012,160] BF16 | ② 的窗口源（每片一张独立张量，**不是一片一文件**） |

**表的三个元数据用第二个来源交叉核对**（`slice_layer_manifest.py` 的 emit 块）：从 checkpoint 的
`ple_sz`/`ple_of` 直接读出，校验 `offsets` 是 `sizes` 的前缀和、`ceil(Σsize/128)×128 == 逐片累加行数`
（320,001,446 → 320,001,536 → 128 × 2,500,012，与 `ple/PLE_SPEC.md §2.1` 的 padding 公式逐值相符）。

---

## 6. 接线形态的机制选择（本 mission 的**关键决策**与它的代价）

### 6.1 硬约束（实测）

1. `<<<>>>` 要求被启动的 `__global__` 与启动代码**同一 TU** ⇒ 层循环二进制里**不能**启动
   `m15_ple_ids_kernel`/`m15_ple_body_kernel`（那两个符号在 `m15_ple.asc` 这个独立 target 里）。
2. `m15_hc_layer.h` **没有 include guard**（`#define` 起首检查：文件里没有任何 `#ifndef`）。
3. `m15_ple.asc` **没有 include guard**且自带 `main()`、匿名 namespace 里的 `struct Ctx`。

### 6.2 采用：**机械抽取 device 段** → `m15_ple_wire.h`（(C) 形态；**由实跑选出来**）

`m15_layer_loop/m15_ple_wire.h` = 从 `m15_ple.asc` **逐字抽出的 `namespace M85P { … }`**
（①`IdsOneToken` / ②`PleGather` / ③`PleGemv` / ④`PleGateItem` / ⑤`PleConvItem` 与常量/UB/flag），
**不含** `main()`、匿名 namespace、system 头、AscendC 头。生成器 = `evidence/ple_wire/lift_ple_device_segment.py`：
- `python3 …/lift_ple_device_segment.py` → 生成（`--check` → 重新抽取并与磁盘逐字节比，不一致 rc=1）；
- 脚本还会拒绝任何含预处理指令（`#include`/`#define`/`#undef`）的抽取片段 ⇒ "device only" 是可检查的。
- `m15_ple.asc` **一字未动**；它的 `m15_ple` target 与 13 条判据继续作同一段 device 逻辑的独立验证路。

**为什么不是"零副本"的借用（`#include "m15_ple.asc"`）——实测，不是偏好**：

| 形态 | `runs=all` |
|---|---|
| base `c9f97e3`（无 PLE） | **干净**（2068 + 290 / 0 FAIL） |
| base + **只**把 `m15_hc_layer.h` 那行换成 `#include "m15_ple.asc"`（不含任何接线代码） | **36 FAIL**，全部 `M.moews.L*` |
| base + 只把 `m15_hc_layer.h` 那行**往后挪几行** | 干净 |
| 我们的接线 + 借用形态（提交 `d63ef3a`） | **36 FAIL**，同样 4 字节 |
| 我们的接线 + **抽取形态**（本 tip） | **干净** |

失败形态固定：36 个 GDN 层各 **4 字节**，落在 MoE 段 ws 的 `WS_OFFSETS + 32`
（64 B 专家偏移表的尾部槽位），device 侧 0x00 / 参考侧 0x59；**逐次跑完全相同**（确定性）。
因为"只换 include 位置就翻面、只抽 device 段就干净"，定位到**借用形态引入的额外编译单元内容**
（`main()`／匿名 namespace／system 头／AscendC 头重复包含）会改变同一 TU 里别的 kernel 的产出；
本 mission 未把机理挖到底，**登记为 U-G**（见 §7）。

### 6.3 与塔三条硬要求的关系

| 塔的要求 | 本实现 |
|---|---|
| ①「两份逐字一致」的见证 | **生成式 + `--check`**（重新抽取逐字节比；抽取片段禁含预处理指令） |
| ② 逐字搬运、namespace/常量/UB 偏移不变 | 生成器**只切片**、不改写：`--check` 复跑可验 |
| ③ 把 (D) 写成显式未完成项 | 见 §7 U-C |

> 塔先用 (C)、后因我报"零副本可行"改判 (A′)。**实测后我回到 (C)**：(A′) 在 `runs=all` 上有
> 36 条 FAIL，而 (C) 干净 —— 硬门（零回归）优先，且 (C) 正是塔最初的裁决。

---

## 7. 显式未完成项（**不声称完整**）

| # | 项 | 卡在哪 |
|---|---|---|
| **U-A**（**M111 已修**，读数与命令见 `M111_LANDING_PATH.md`） | **①→② 的 ids 落盘可见性** | 原状：① 的 ids 落点是 `outG.SetValue`（**标量写 GM**，走 scalar pipe），② 在**别的核**上读它；而 `M15L_PhaseBoundaryAiv` 的 set 挂 `PIPE_MTE3`（只排 MTE3 写）⇒ mode-0 barrier 只保证「每核都执行过 set」，**覆盖不到标量落点**，于是除「自己写自己读」的那个核外，其余核读到旧值。逐 item 证据：`WITNESS.md §4`（`M15_PLE_DBG=1` 的 16 行：g=0 对、g=1..15 全同一个错值）。**M111 的处置 = 原「修法 (a)」**：① 的落点改 UB → MTE3 `DataCopy`，**核内次序按 `docs/05 §6.1 ⓔ` 的标量条款**（写侧 `BufAcquire<PIPE_S>` → 标量写 → `BufRelease<PIPE_S>` drain release → 才搬；正对照 = `m15_hc_layer.h::CombineStage`；**不是** `PipeBarrier`，`m15_attn_cache.h` 那处按 ⓔ 不作先例），同步重跑生成器、M85 的判据与 M92 的设备判据。**口径（必须连带读）**：`pleStageMask` 默认 31（① 也跑）—— 修前 **`M15_PLE_WIRE=1` 的默认档是红的**，修后这一档在 M111 的批次里转绿（`WITNESS.md §3`）；「先只接 1 层跑通」的旧口径（只在 `STAGE=15` 下成立）不再必要 |
| **U-B** | **③④⑤ 的 T3 数值对拍**（本档未写 host float64 参考链） | 本档对 `gated/normed/out` 只有「非空洞 + 状态移位逐字节 + 层入口态被改动的读数」；③④⑤ 的 T3 逐元素判据仍只在 M85 的独立档（`m15_ple_check.py` 13/13）里 |
| **U-C** | **(D) host 侧拆两次启动的形态**（塔要求写清代价） | 代价：改 `CMakeLists.txt`（把 `m15_ple.asc` 编进层循环 target）+ `LayerArgs` 加分段字段 + 重排 `M15L_FusedBody` 的相位结构（M97 刚改过它）。换来：PLE 仍以**真 kernel** 插在相位序里、① 的落盘由 stream 顺序保证（U-A 自动消失） |
| **U-D** | 其余 PLE 层 / prefill / aclgraph / 性能 | 本 mission 只接 0-based 层 1（`ple_layer_ids=[2]`）；prefill/spec 的 short-conv 独立 writeback 见 `ple/README.md §7 U3`；性能见 U4 |
| **U-E** | 单窗口装不下真实 token 的 16 个 id | 真实词表下 16 个 id 几乎必然落在不同分片 ⇒ 单窗口 `miss=16`（实测档 F）。多槽滑窗 = M92 §9 U-A，未做 |
| **U-F** | `m15_layer_resources.h` 的两处登记（FLAG_SEQ 的 PLE 段内 3 行、§2 UB 表的 PLE 行） | 该文件不在本 mission scope |
| **U-G** | **借用形态（`#include "m15_ple.asc"`）为什么会让 `runs=all` 的 `M.moews` 4 字节翻面** | 实测把机理缩到"借用形态引入的额外编译单元内容（`main()`/匿名 namespace/system 头/AscendC 头重复包含）"这一层：只换 include 位置即翻面、只抽 device 段即干净、失败点固定在 MoE 段 `WS_OFFSETS+32`。**没有挖到底**（未逐项二分 `m15_ple.asc` 的 include 与 host 段）。影响：本 mission 改用抽取形态（干净）；但如果要回到"零副本"形态，得先解掉它 |
| **U-H** | `M.moews` 判据自身的**脆弱性**（次要观察） | 它是 MoE 段 ws 的**全量逐字节**比（`M.gdnws` 有 padding 跳过，它没有），而那 4 字节落在**专家偏移表的尾部槽位**；我试过在 M1 之前把三块 ws 复位到同一个 0xCD 基线（`aclrtMemset`）——**不能**消除差异（说明两条路都真的写了那块，只是值不同）。⇒ 该判据的 PASS 是否稳健，值得后续 mission 单独查（不是本 mission 的交付项，也是我**遇到的那 36 条 FAIL 的表面所在**） |
