# M105 · S3 诊断槽 UB 标量写抢跑修复 —— 证据与口径

本目录是 mission **M105**（分支 `feat/m105-m15-moe-diagnostic-slot-race-fix`，base `main @ 20bd20d`）的证据。
交付物两件：

| 文件 | 性质 | 改动 |
|---|---|---|
| `m15_layer_loop/lift_moe_segment.py` | **生成器**（唯一手改的源文件） | 新增 [M105] 次序修复：S3 诊断槽的 UB 标量写从 `BufAcquire<PIPE_MTE3>` 之后提前到 `MutexLock<PIPE_S>` 之前；并加三条**次序/唯一性断言** |
| `m15_layer_loop/m15_moe_layer.h` | **生成物**（由上面的生成器机械重生成，非手改） | 同上那一处位移 + 注释；`--check` rc=0 |

## 0. 一句话结论

`m15_moe_layer.h` 的 `IndexGenStage::Run`（= S3 索引生成）里，诊断槽（GM `988224`，UB `UB_IG_CNT+128`）
那次 **UB 标量写**落在 `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX)` **之后** ⇒ 与紧随其后的
`DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4))`（MTE3 读同一 UB 位置）**没有 pipe 次序保证**。
该 UB 位置 `UB_IG_CNT+128 = UB_RT_XB+2176` 正好落在 router 的 **x 行 0 缓冲**里 ⇒ 抢跑时落盘的是
**同层 `x_norm` 行 0 的第 2176 字节起 4 B 残值**，不是计数器。

**因果闭环（本 mission 的价值）**：
1. 这个槽是 bring-up 的「专家 id 越界计数」诊断，本档真值是 `0`（`badIds == 0`）；
2. 抢跑使它有时落 `0x00000000`（真读到计数器）、有时落 x 残值（例如 `0x8c3f07bd`）；
3. 判据 `M.moews.L*` 比的是**融合路 ws vs moe-only 路 ws**（同一段代码两条路径），
   它 PASS 当且仅当**两条路径的抢跑落点恰好一致** —— 与「计数器对不对」无关；
4. ⇒ 当前 main 上的全绿里，那 4 B 是**假绿**（两条独立抢跑读数相等的巧合，§4③ 有逐字节落盘证据）；
   `c9f97e3` 基座上同一份源码只多一个 include 就变 36 条红（§4① 复现），红绿都不携带该槽的信息；
5. 修后两条路径都在次序 token 之内写入 ⇒ 落盘恒为计数器真值 `0`（§4③），判据这才真的在比计数器。

**第 2 轮复审（r1，`p1-1items`）后的最终形态**：复审/塔裁认定"只把写挪到 mode-0 release 的区域"**不算修好**
（`docs/05 §6.1 ⓔ` / `docs/20 §3.2 WO-B1` 选项 3），要求**写侧释放改用 drain 模式**。本轮的交付形态因此是
**两处一起**：① 写提前到写侧 acquire 之前；② 写侧释放 `MutexUnlock<PIPE_S>`（mode 0）→ **`BufRelease<PIPE_S>`
（drain）** —— 成对形态（ⓔ）「**写** → 写侧 acquire → 写侧 drain release → 搬侧 acquire → 搬」，
本次的写落在写侧 acquire **之前**（ⓔ 的覆盖面前提只要求写在该 drain release 之前）。可见性由
**规则**建立（不再只靠实测），§4 的五条判据在本轮形态上**全部重取**。

## 1. 复现（两个基座 × {未修, 修后}）

配方（**M102 的复现配方**，无任何接线代码）：把 `m15_layer_loop.asc` 的
`#include "m15_hc_layer.h"` 换成 `#define main m15_ple_standalone_main_unused / #include "m15_ple.asc" / #undef main`，
并把 `Ctx C;` 写成 `M15Run::Ctx C;`。脚本化的完整 host 侧序列见 `reproduce_host.sh`。

四档二进制与 `runs=all` 读数（读数为**本 mission 实跑**；`run_*.log` 在本目录）：

| /tmp 干净树 | 二进制 `sha256`（前 12） | `runs=all` |
|---|---|---|
| `c9f97e3` + 借 include，**未修** | `a1d431f9a283` | rc=1 `FAILURES PRESENT（checks=2068, guards=291, fails=35）`，35 条**全部** `M.moews.L*`；`run_c9_unf_dump.log` |
| `c9f97e3` + 借 include，**上一轮形态**（只挪写） | `2d279dc01fa5` | rc=0 `ALL PASS（checks=2068, guards=291, fails=0）`；`run_c9_fix_dump.log` |
| `c9f97e3` + 借 include，**本轮形态**（写提前 + drain 释放） | `8310e71df5d8` | rc=0 `ALL PASS（checks=2068, guards=291, fails=0）`；`run_c9_fix2_dump.log` |
| `c9f97e3` + 借 include，**反例**（写搬回 acquire 之后 + drain 释放） | `1c0d3a6961e7` | rc=1 `FAILURES PRESENT（checks=2068, guards=290, fails=36）`；`run_c9_racy2.excerpt.log` |
| `20bd20d` + 借 include，**未修** | `63ffd4d9e032` | rc=0 `ALL PASS（checks=2095, guards=303, fails=0）`；`run_mn_unf_dump.log` |
| `20bd20d` + 借 include，**修后** | `f61800fb35ae` | rc=0 `ALL PASS（checks=2095, guards=302, fails=0）`；`run_mn_fix.log` |

两个「未修」档的二进制 sha256 与 M102 survey 给的前 12 位**逐位相同**（`a1d431f9a283` / `63ffd4d9e032`）
⇒ 我的 /tmp 树与 M102 的是同一形态（另：两档带 `M15_DUMP=1` 时 guard 计数比不带时多 1，
`291 vs 290` / `303 vs 302`，判定项 `checks` 不变）。

**「同一 include 在 `c9f97e3` 翻面、在当前 main 不翻面」这个事实**：两个基座之间
**只有 attention kv/cache 一族文件不同**（`git diff --stat c9f97e3 20bd20d`：`m15_attn_cache.h` 新增 481 行、
`m15_attn_cache_host.h` 新增 1163 行、`m15_attn_kv*.h` 改 ~335 行、`m15_layer_loop.asc` / `m15_moe_layer.h` /
`m13_moe_layer.asc` / `lift_moe_segment.py` / `m15_moe_resources.h` **一字未变**）。
⇒ 翻不翻面**不取决于 MoE 段自身**，而取决于**整棵 TU 里多出来的 device 代码改变了调度**：
同一份 S3 代码的标量写与 MTE3 读谁先到 UB，是被整棵 TU 的指令排布决定的。
这正是「只报『借 include 会翻面』会漏掉一半事实」的含义，也说明**光靠 `runs=all` 绿**证明不了这 4 B。

一条完整红档输出（任务①要的那条）：

```
[m15]   M.moews.L01                    FAIL (4/7685472 字节不同; first off 988224 got 0x00 exp 0x8c)
```

（`run_c9_unf_dump.log`；`got` = 融合路 ws、`exp` = moe-only 路 ws。同一命令同一二进制连跑会抖：本档连跑
5 次得 3×36 + 2×35（两次 35 都少 L00），见 §4⑤(a)/`jitter_unf.txt`。）

## 2. 定位确认（三件事，file:symbol）

基准：`m15_layer_loop/m15_moe_layer.h`（**修后** tip；生成物，行号会随生成器变化，故同时给**内容锚点**）。

### ① 标量写与 `BufAcquire<PIPE_MTE3>` 的先后

| 位置 | 内容锚点（可 grep） |
|---|---|
| `M15M::IndexGenStage::Run` 开头 | `oobCountUb = badIds;   // 诊断槽` —— **修后**紧跟 `diagUb[32] = static_cast<int32_t>(oobCountUb);` |
| 函数中部 | `MutexLock<PIPE_S>(BUF_AIV_IDX);` ← 写侧 acquire（`GetBufInternal<PIPE_S,0>`）；**修后**其后的释放是 `BufRelease<PIPE_S>(BUF_AIV_IDX);`（**drain** release） |
| 函数末 | `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);` ← 搬侧 acquire；其**之后**是 `DataCopyPad(offsGm[IG_DIAG_GM_SLOT], bL[32], ExtBlock1(4));` |

- **改前（m13 原样，= base rev `20bd20d` 的库内 artifact）**：那次标量写 `bUb[32] = static_cast<int32_t>(oobCountUb);`
  就在 `DataCopyPad(… bL[32] …)` 的**上一行**，而两者都在 `BufAcquire<PIPE_MTE3>` 之后；写侧的释放是
  `MutexUnlock<PIPE_S>`（= `RlsBufInternal<PIPE_S,0>`，**mode 0**，不承载跨 pipe 数据可见性）⇒ 那次写
  与那条读之间**没有任何次序契约**。
- **改后（本 tip 的形态，两处一起）**：① 写提前到写侧 acquire **之前**；② 写侧释放由 `MutexUnlock<PIPE_S>`
  改成 **`BufRelease<PIPE_S>`（drain）** —— S pipe 排空后才释放 token，**覆盖它之前的全部标量写**（含这一次）。
  `docs/05 §6.1 ⓔ` 的"标量值确需落盘"条款逐字要求正是这个次序（内容锚点：`若标量值确需落盘` /
  `先写 UB` / `用 drain 模式的 BufferID release 建立次序` / `之后才 MTE2/MTE3 搬运`，
  同段还有一句覆盖面前提：`release 只覆盖它` **之前** `的写` ⇒ 标量写必须落在该 drain release 之前）；
  同款的**已受审**站点（在 `main` 上，我用 `git show main:m15_layer_loop/m15_ple.asc` 逐行核过，内容锚点：
  `BufAcquire<PIPE_S>(id)`（写侧成对）→ **标量写 UB** → `BufRelease<PIPE_S>(id)`（= `RlsBufInternal<PIPE_S,true>`，
  **drain release**））：`m15_ple.asc` 的 IDS/BODY 三处。
  （r1 复审信里提到的"另一处 `m15_hc_layer.h::CombineStage`"我在当前 `main` 上没核到 PIPE_S 的 drain 站点
  —— `m15_hc_layer.h` 里的 drain 站点是 `PIPE_MTE2`/`PIPE_V`；本条**不据复审的信件转述**，只记我自己核到的。）
  （机制口径的完整交代、以及上一轮"只挪写、释放仍是 mode 0"为何**不算修好**，见 §2④ 与 `mechanism_note.md`。）

`grep` 锚点（一处在改前、一处在改后；行号基准 = 本分支 tip，**以内容锚点为准**）：

```
$ grep -n "bUb\[32\] = static_cast<int32_t>(oobCountUb);" m13_moe_layer/m13_moe_layer.asc
1171:            bUb[32] = static_cast<int32_t>(oobCountUb);
$ grep -n "diagUb\[32\] = static_cast<int32_t>(oobCountUb);" m15_layer_loop/m15_moe_layer.h
1340:        diagUb[32] = static_cast<int32_t>(oobCountUb);
$ grep -n "MutexLock<PIPE_S>(BUF_AIV_IDX);" m15_layer_loop/m15_moe_layer.h
1351:        MutexLock<PIPE_S>(BUF_AIV_IDX);
$ grep -n "BufRelease<PIPE_S>(BUF_AIV_IDX);" m15_layer_loop/m15_moe_layer.h
1385:        BufRelease<PIPE_S>(BUF_AIV_IDX);        # 写侧 drain release
$ grep -n "BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);" m15_layer_loop/m15_moe_layer.h
1397:        BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);
```

（1340 < 1351 < 1385 < 1397 = 写 → 写侧 acquire → 写侧 drain release → 搬侧 acquire。**行号随生成器漂，以内容
锚点为准** —— 上面这组数字是本 tip 的实测读数。改前那个变量名 `bUb` 在产物里是否还在，不靠 grep 计数下结论：
生成器断言直接断言 `"bUb[32] = static_cast<int32_t>(oobCountUb);" not in text`，且该断言在 §3 的变体 B2 上实测报错。）

### ② 诊断槽 UB 地址与 router x 窗的重叠（常量自算）

算式与输出见 `addr_arith.md`（程序 `addr_arith.cpp` 直接 include 库内 `m15_moe_resources.h`）：

```
UB_IG_SRC   = UB_RT_XB       = 144384
UB_IG_EXP   = UB_IG_SRC + TOTAL_MAX*4 = 144384 + 1024 = 145408      (TOTAL_MAX = M_MAX*TOPK_MAX = 64*4 = 256)
UB_IG_CNT   = UB_IG_EXP + TOTAL_MAX*4 = 145408 + 1024 = 146432
诊断槽字地址 = UB_IG_CNT + 32*4 = 146432 + 128 = 146560
             146560 - UB_RT_XB(144384) = **2176**
router x 行 0 的 UB 跨度 = [UB_RT_XB, UB_RT_XB + HIDDEN*2) = [144384, 149504)   (HIDDEN = 2560)
             ⇒ 146560 ∈ [144384, 149504) ⇒ **诊断槽那 4 B 就在 router x 行 0 缓冲里**
GM 诊断槽    = WS_OFFSETS(988192) + IG_DIAG_GM_SLOT(8)*4 = **988224**
             （IG_DIAG_GM_SLOT = AlignUp((NUM_EXPERTS+1)*4+4, 32)/4 = AlignUp(24,32)/4 = 8；SZ_OFFSETS = 64）
WS_BYTES     = 7685472（= 落盘 `L*_moe_ws.bin` 的字节数，逐一相符）
```

落盘的 `moe_layout.txt`（程序自己按同一批常量算出）给出同一结论的两条旁证：
`tensor name=x_norm mode=contig ws_off=0 bytes=5120`（⇒ dump 内偏移 2176 在 x_norm 行 0 内）、
`tensor name=expert_offsets ws_off=988192 bytes=20`（⇒ 988224 恒在真 `offsets[0..E]` 之后，是那 32 B 槽的起点）。

### ③ 与 m17 先例逐字对照

逐字摘录见 `m17_precedent.txt`（基准不可变 rev `20bd20d659b11cbb5e6e6d317921ecb3c16e2629`，
m17 的注释在 1292–1295、写语句在 1297–1298）。m17 的原话是：

> `// 诊断槽的 UB 标量写必须落在 MutexLock/Unlock<PIPE_S> **之内或之前**：`
> `// MTE3 读 UB 的次序由「S 侧 unlock（drain）→ MTE3 acquire」保证，若把这次标量写`
> `// 放到 unlock 之后（m13 原样），就会与 DataCopyPad 抢跑 —— 实测 m=8 时诊断槽读到`
> `// 上一段遗留的 UB 残值（0x3ec5be6f），落盘 sha256 逐次不同。`

**采用的形态**：与 m17 相同 —— 写在 `oobCountUb = badIds;` 之后、`offUb[0] = 0;` 之前，
**早于 `MutexLock<PIPE_S>`**（m17 注释里的「之内或之前」，m17 自己选的是「之前」，我照抄「之前」）。

**与 m17 的差异（逐条）**：

| # | m17 | 本 mission（m15） | 说明 |
|---|---|---|---|
| 1 | 写语句包在 `{ … }` 里，指针名 `bUb` | 不包块，指针名 `diagUb` | 纯作用域/命名风格（m15 生成物里 `cntUb/offUb/curUb` 也不包块）；语义相同 |
| 2 | 写 `static_cast<int32_t>(badIds)` | 写 `static_cast<int32_t>(oobCountUb)` | 同一值（上一句刚 `oobCountUb = badIds;`）；m15 保留成员变量是为了 `OobCount()` 出口 |
| 3 | 槽下标是具名常量 `IG_CNT_DIAG_SLOT = 2*NUM_EXPERTS + 8`（E=512 ⇒ 1032） | 保留 m13 的字面下标 `32`（`UB_IG_CNT+128`） | **地址一字未动**（本次只动先后）；m17 的 UB 槽地址与 m15 本就不同（两段 UB 布局不同），不构成应当对齐的差 |
| 4 | m17 那次写无条件执行（它的门控在别处） | 本次把写从 m13 的 `if (subLimit >= 9) { … }` 里**提出来**，无条件执行；`DataCopyPad` 仍留在门控内 | m15 里 `subLimit` 已退化为编译期常量 `9`（`m15_moe_layer.h` 头注：「sliceMode / stageLimit / subLimit 三个 bring-up 截断开关退化为编译期常量」）⇒ 门控恒真；提出去后 `subLimit<9` 的 GM 效果不变（只多写一个没人读的 UB 字） |

### ④ 机制口径：M107-r1 更正 → r1 复审 P1 → 本轮改用 drain 释放（全文见 `mechanism_note.md`）

**事实链（三份来源都可核）**：
- CANN `basic_api/kernel_common.h:138-160`：`Mutex::Lock<pipe>` = `GetBufInternal<pipe,0>`、
  `Mutex::Unlock<pipe>` = `RlsBufInternal<pipe,0>` —— **mode 0 = 立即生效（互斥交接），不排空生产 pipe**；
- 本仓 `m15_moe_layer.h` 的两个包装：`BufAcquire<pipe>` = `GetBufInternal<pipe,false>`、
  `BufRelease<pipe>` = `RlsBufInternal<pipe,true>`（**drain**："release 延迟到 pipe drain"）；
- 仓内规则：`docs/05` 「同步约束」与 `docs/06:76` —— 跨 pipe UB 数据可见性要用 **drain 模式 release**；
  `docs/05 §6.1 ⓔ` 的"标量值确需落盘"条款（内容锚点 `若标量值确需落盘`）逐字：**先写 UB → 用 drain 模式的
  BufferID release 建立次序 → 之后才 MTE2/MTE3 搬运**，并明写本仓的 drain release 是 `BufRelease<PIPE>(id)`
  **不是**公开 `Mutex::Unlock`（mode 0）；同段还写明覆盖面：**release 只覆盖它之前的写** ——
  写在 acquire 之后的标量写不在覆盖内（正是本例的形态）。

**⇒ 上一轮形态（只把写提前、释放仍是 `MutexUnlock<PIPE_S>`）落在 ⓔ 之外**：那次写只被一个 mode-0 的
release 覆盖，MTE3 读"一定在标量写落地之后"**没有被建立**。r1 复审因此判 p1：`grep -c "BufRelease<PIPE_S>"`
在上一版 tip 上为 0。**塔接受了我关于"机制论证给不出、读数只能支持提前量"的如实交代，并据此维持原裁**
（症状消失 + 位置必要性 ≠ 次序建立；本缺陷本身是调度依赖的）。

**本轮改法**：把**写侧释放**换成 `BufRelease<PIPE_S>(BUF_AIV_IDX)`（drain）+ 保留写的提前 —— 于是
① 次序由规则建立（drain release 覆盖它之前的**全部**标量写，含诊断槽那次）；② 写仍提前（位置必要性由
D2/`c9_racy2` 实测：写在搬侧 acquire 之后时，任何 release 都覆盖不到它 ⇒ 仍 36 条红）。
写侧 acquire 保持 `MutexLock<PIPE_S>`（= `GetBufInternal<PIPE_S,0>`，与 `BufAcquire<PIPE_S>` 同一 token
获取）⇒ **成对不破**（`m15_ple.asc` 文件头 §BUF 记着"只放 release 而缺写侧 acquire 会挂死"的教训）。

**"另外 10 处"的顺序性**：本目录**从未**声称它们有序（非自指的可复跑检查 + 逐处登记表见 `mechanism_note.md` §2），
本轮也**没有**把它们升级成"已坏" —— 它们仍登记为**顺序性未确认**；不过本次改动
（drain release）**顺带把它们也纳入了同一条 drain 覆盖**（它们本来就在写侧 acquire 之后、释放之前）。

## 3. 修：改生成器、不手改产物

`m15_moe_layer.h` 是 `lift_moe_segment.py` 从 `m13_moe_layer/m13_moe_layer.asc` 机械抽取的**生成物**
（`--check` 可复核）。所以修在生成器里，改法是**调整既有规则 7 的 #5/#6 那两处替换文本 + 新增一处释放替换**：

1. `IDX_HEAD_NEW`（规则 7 的 #5「计数/前缀和」新文本）末尾追加写语句 + [M105] 注释 —— 它落在
   `oobCountUb = badIds;` 之后、写侧 acquire（`MutexLock<PIPE_S>`）之前；
2. `IDX_DIAG_NEW`（#6「诊断槽」新文本）**删掉**那两行（`bUb` 声明 + 写），只留 MTE3 那一读；
3. **新增 `IDX_RELEASE_OLD → IDX_RELEASE_NEW`**：`MutexUnlock<PIPE_S>(BUF_AIV_IDX);` →
   `BufRelease<PIPE_S>(BUF_AIV_IDX);`（**drain** release）+ [M105] 注释（r1 复审 P1 的落点）；
4. `rewrite_index_gen()` 里加**四条非空转**的断言（改一处立刻 `AssertionError`，四条各有一个反例实测报错）：
   - 写语句在产物里**唯一**（`count == 1`）；
   - 写侧 **drain release 必须存在且唯一**（`count("BufRelease<PIPE_S>(BUF_AIV_IDX);") == 1`），
     且**不得残留** `MutexUnlock<PIPE_S>(BUF_AIV_IDX);`；
   - 次序 `写 → MutexLock<PIPE_S> → BufRelease<PIPE_S> → BufAcquire<PIPE_MTE3>` 在产物里按序出现；
   - 改前的特征串 `bUb[32] = static_cast<int32_t>(oobCountUb);` 在产物里**不再出现**，且它**确实**出现在
     上游 `m13_moe_layer.asc` 的 `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);` **之后**、上游释放**确实**是
     `MutexUnlock<PIPE_S>`（正对照：上游若哪天自己修了，本规则报「空转」而不是静默变 no-op）。

A/B（逐字节，命令与读数见 `ab_generator.log`）：

```
旧生成器（`git show 20bd20d:…` 的版本，在 /tmp 重生成） → sha256 0af84032dc19…  == base rev 的库内 artifact（diff 空）
新生成器（本 mission 版，在 /tmp 重生成）               → sha256 50b12682c033…  == 库内新 artifact（diff 空）
库内 --check（tip 上实跑）: rc=0
产物差异面：只有 M105 那三处（写位移 + 释放改 drain + 注释；`git diff --stat` 口径见下）
```

**注释改动的二进制身份**（可复跑）：artifact `ee12ba635c95…`（第 2 轮送审版）与 `50b12682c033…`（r2 复审后按
建议改措辞的版本）只差注释文本，**重编后二进制逐字节相同**（`8310e71df5d8…`，2026-10-04 实编）
⇒ 本节所有设备读数对两者都成立，本轮无需重跑设备。

`--check` 在 tip 上：

```
$ cd m15_layer_loop && python3 lift_moe_segment.py --check; echo rc=$?
[lift] m15_moe_layer.h 与抽取规则一致（内容锚点：asc 行 62..2031，入口在第 2033 行）
rc=0
```

## 4. 五条判据（在本轮交付形态上全部重取）

设备纪律（本轮按复审要求收紧）：每条设备命令 = `flock -w 900 /tmp/npu0.lock /tmp/m105/inner_run.sh …`
—— `-w` 在锁文件之前；**进锁后先 `npu-smi` 复查**（脚本里 16×15s 的等待环）；设备命令一律 `timeout 900` 包住；
单进程、未并发。逐次读数（含起止时点）见 `runs_summary.txt`。

### ① 正解负向对照（翻面基座：35–36 红 → 绿）

同一基座（`c9f97e3` + 借 include），三种 artifact：

| 档 | 二进制 sha256（前 12） | `runs=all` |
|---|---|---|
| base rev `20bd20d` 的 artifact（未修） | `a1d431f9a283` | rc=1 `checks=2068, guards=290, fails=36`（×4 与 `fails=35` ×1，见 §4⑤(a)） |
| **本轮交付形态**（写提前 + 写侧 drain release） | `8310e71df5d8` | rc=0 `ALL PASS（checks=2068, guards=290, fails=0）`（`c9_fix2_dump`；另 20 次见 §4④） |
| 上一轮形态（只提前写、释放仍 mode 0）——**不算修好** | `2d279dc01fa5` | rc=0 `ALL PASS（checks=2068, guards=290, fails=0）`（保留为"症状面不足以定案"的证据） |

失败明细（未修档）：36 个 GDN 层里 35–36 条 `M.moews.L*` 报 `4/7685472 字节不同; first off 988224`；
12 个 attention 层（占位直通）从不报错。三种 artifact 的修后形态 48 条 `M.moews.L*` 全部 PASS。

### ② 零回归

**(a) 交付形态（本分支 tip 的**正常**构建，不加借 include）**：

| 档 | 命令 | 读数 |
|---|---|---|
| 本分支 tip（`20bd20d` + 本轮生成物，正常 include） | `M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 <bin> <manifest> all` | rc=0 `ALL PASS（checks=2095, guards=302, fails=0）`，`M.moews` 48/48 PASS（`run_tip3.log`；二进制 `1322c0e11d54…`） |
| **当前 main（`9296e79`）+ 本轮生成物**（正常 include；r1 复审当时用的 main 是 `a1521cb`，本读数取本轮送审时的 main `9296e79` —— 其后 main 又前移到 `30d6984`，我的读数是**不可变 rev** 上的） | 同上 | rc=0 `ALL PASS（checks=2095, guards=302, fails=0）`，`M.moews` 48/48 PASS（`run_main_now.log`；二进制 `8b78c168332d…`） |

**(b) 与 M102 同口径的基座档（当前 main 基座 + 借 include + 修后）**：

| 档 | 命令 | 读数 |
|---|---|---|
| `20bd20d` + 借 include + **修后** | `M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 <bin> <manifest> all` | rc=0 `ALL PASS（checks=2095, guards=302, fails=0）` —— 与 M98/M102 在该基座上的 `2095/302/0` 逐项相同（`run_mn_fix.log`） |

### ③ 「真的比到计数器」的落盘证据

落盘档加 `M15_DUMP=1 M15_DUMP_MOE=<层号表>`；读取脚本 `check_slot_bytes.py`（两个偏移都由库内常量自算，
不抄结论）。逐档逐层读数全文见 `slot_bytes.txt`，摘要：

| 档 | 槽 `[988224:988228]`（融合路落盘） | 同层 `x_norm` 行 0 `[2176:2180]` | 判据行 |
|---|---|---|---|
| **本轮交付形态** · 9 层（`c9_fix2_dump`，二进制 `8310e71df5d8`） | 9 层全 `00000000`（L00/L01/L02/L04/L05/L06/L08/L10/L14） | 9 层各自一个非零残值（**与上一轮逐字节相同**：L00 `593f36bc`、L01 `8c3f07bd`、L02 `9b3d2b3e`、L04 `c8bdaf3d`、L05 `68ba36be`、L06 `893d18be`、L08 `a1bdba3e`、L10 `cdbe793c`、L14 `d0bb47bb`） | 48 条 `M.moews.L*` 全 PASS ⇒ 两侧都是这个 `0` |
| 上一轮形态 · 9 层（`c9_fix_mldump`，二进制 `2d279dc01fa5`） | L00 `00000000`、L01 `00000000`、L02 `00000000`、L04 `00000000`、L05 `00000000`、L06 `00000000`、L08 `00000000`、L10 `00000000`、L14 `00000000` | 9 层各自一个非零残值（L00 `593f36bc`、L01 `8c3f07bd`、L02 `9b3d2b3e`、L04 `c8bdaf3d`、L05 `68ba36be`、L06 `893d18be`、L08 `a1bdba3e`、L10 `cdbe793c`、L14 `d0bb47bb`） | 48 条 `M.moews.L*` 全 PASS ⇒ **两侧都是这个 `0`** |
| 未修 · 9 层（`c9_unf_mldump`） | 同上 9 层全 `00000000`（这一趟融合路读到了计数器） | 同上（残值逐层相同 —— 残值内容是确定的 x 数据） | 36 条 `M.moews.L*` FAIL，`got 0x00` / `exp 0x??`，且 **`exp` 首字节逐层 == 该层 dump 的 `x_norm[2176]`**（9 层全对上，见 `slot_bytes.txt` 末表） |
| 未修 · 融合路读到残值（`c9_unf_dump` L00、`mn_unf_dump` L00/L01） | `593f36bc` / `8c3f07bd`（**== 同层 x 残值**） | 同左（相等） | **PASS** ⇒ 两路都读到残值 ⇒ 这条 PASS 与计数器无关（假绿现场） |
| 拨值探针（`c9_fix_probe_dump`） | `5a5a5a5a`（= `0 ^ 0x5A5A5A5A`） | 非零残值，≠ 槽 | 48 条 PASS |
| 拨值探针（`c9_unf_probe_dump`） | `593f36bc` / `8c3f07bd`（残值；`5a5a5a5a` 未落地） | 同左（相等） | 48 条 PASS |

五条读法：

1. **修后**：槽恒为 `00000000`（= 本档 `badIds` 真值），且**不再等于**同层 x 残值；判据 48/48 PASS
   ⇒ 融合路与 moe-only 路由同一段代码产出同一个 `0`。
2. **未修 · 融合路读到残值**：槽 == 残值 且判据 PASS ⇒ 那 4 B 是**两条路径都读了同一片 x 残值**，
   与计数器无关。这一形态在**当前 main 基座**（`mn_unf_dump`）上整层整层地出现 ⇒ main 的全绿是假绿。
3. **未修 · 融合路读到计数器**：槽 = `00000000`，而 moe-only 侧 `exp 0x??` **逐层等于**该层 x_norm
   行 0 的第 2176 字节 ⇒ 判据报的是「计数器 vs 残值」的差异，36 条红。
4. **拨值探针**（只改「写进槽的值」，其余不动，仅 /tmp）：修后槽跟着变成 `5a5a5a5a`（**值随那次标量写落盘**），
   未修槽仍是残值（`5a5a5a5a` 没落地）⇒ 那 4 B 在修后由这次标量写决定、在修前与它无关。
   它也说明判据不是「0 vs 0 空过」：本档 `badIds == 0`（槽读回 0），而把写值拨成 `0x5A5A5A5A` 之后
   槽随之变化 ⇒ 槽的内容**随标量写变**。

5. **本轮形态（drain release）**：9 层槽 `00000000`、同层残值逐字节与上一轮相同（残值内容由 x 数据决定，
   与 release mode 无关）⇒ 换了 release 形态后"槽 ≠ 残值"这条**依旧成立**；判据 48/48 PASS。
   （本轮另有一个 `/tmp` 拨值探针 `c9_fix2_probe`，见 §4⑤(b)：值随这次写落盘 ⇒ 不是"0 vs 0 空过"。）

`L*_moe_ws.bin` 是**融合路**那一侧的 ws（host 侧 `H_DumpMoeLayer` 只落这一侧）；moe-only 侧的值只在
`run.log` 的 `exp 0x??` 里露首字节 —— 第 3 条正是靠这个首字节完成了两侧的对照。

### ④ 确定性（同一二进制连跑 ≥20 次）

本轮交付形态的二进制 `8310e71df5d8…` 连跑 20 次（`d_01`…`d_20`，与 §4① 的 `c9_fix2_dump` 同一二进制）：

```
| # | tag | run.log sha256(前 16) | M.moews 行数 | PASS | FAIL | 汇总行 |
| 1 | `d_01` | `c5386aab1d5c3bfe` | 48 | 48 | 0 | `===== ALL PASS（checks=2068, guards=290, fails=0）=====` |
| 10 | `d_10` | `f0e4a71e0396d3f2` | 48 | 48 | 0 | `===== ALL PASS（checks=2068, guards=290, fails=0）=====` |
| 20 | `d_20` | `472cb5dba95357bc` | 48 | 48 | 0 | `===== ALL PASS（checks=2068, guards=290, fails=0）=====` |
```

（20 行全表见 `det20_form2.txt`；上面三行是首/中/末，其余 17 行的两个计数与汇总行同形。）

分布：**20/20 `fails=0`**。对照同一会话语境里的**未修**二进制：5 次得 **3×36 + 2×35**（§4⑤(a)），
M102 survey 在未修形态上另测到 **8×36 + 1×35**。

### ⑤ 把被测对象弄坏（只在 /tmp）

**(a) 未修形态（artifact 回到 base rev `20bd20d` 的版本）＝ 把「修复」这件事撤掉**：同一基座上
5 次跑得 **3×36 + 2×35**（失败层恒为 36 个 GDN 层；失败字节恒为 `4/7685472`，首差异偏移恒为 `988224`；
两次 35 少的那层都是 L00 —— 两条抢跑读数恰好一致 ⇒ 该层翻绿）。逐次 tag / 时点 / 失败层见 `jitter_unf.txt`。

**(b) 位置必要性（本轮形态的对照档）**：只把**那次写搬回 `BufAcquire<PIPE_MTE3>` 之后**、**保留**
写侧 drain release，其余一字不改（`c9_racy2`，二进制 `1c0d3a6961e7`）⇒ rc=1 `fails=36`，
36 条 `M.moews.L*` 全部 `4/7685472 字节不同; first off 988224 got 0x00 exp 0x??`。
⇒ 任何 release 都排在那次写**之前**，drain 也覆盖不到它 ⇒ **挪位是必要动作**（这一档与 r1 复审自己跑的 D2 是同一二进制）。
（注：把本档与 (a) 的 `c9_unf_dump` 档**按层**比 `exp` 首字节：两档**共同失败**的 35 层**逐层相同**，
差别只在 L00 的判定 —— 本档红、`c9_unf_dump` 那一次绿。**不据此推断任何机制**（例如"residue 随 release
形态改变"之类）：本 mission 没有支持该说法的读数。）

**(c) 拨值探针（本轮形态）**：把写进槽的值改成 `oobCountUb ^ 0x5A5A5A5A`（仅 /tmp，`c9_fix2_probe`，
二进制 `c487fbdf8483`）⇒ 槽 = `5a5a5a5a`（= `0 ^ 0x5A5A5A5A`），**不是**残值 ⇒ 那 4 B 是**这次标量写**的值，
不是"两侧都没写过的 0"（判据非空洞）。同一探针在上一轮形态上也是 `5a5a5a5a`（`c9_fix_probe_dump`）。

**(d) 生成器守卫的咬合力**（把守卫要守的东西弄坏）：**四个**变体各让一条断言报错，见
`generator_guard_mutation.log`（本轮版本）——
A2（删掉提前的那次写）⇒ 「写不唯一：0 处」；
B2（#6 里把改前的写加回来）⇒ 「改前的诊断槽标量写仍在」；
C2（写搬回 `BufAcquire` 之后）⇒ 次序断言报错；
D2（把写侧释放的替换做成 no-op，仍是 `MutexUnlock`）⇒ 「写侧 drain release 不唯一：0 处」。

### ⑥ 形态三档对照（"只挪写"为什么不等于修好）

同一基座（`c9f97e3` + 借 include）、同一数据集、同一命令，只差 `m15_moe_layer.h` 的两处：

| # | 写的位置 | 写侧释放 | 二进制（前 12） | 判据（`M.moews` 那 4 B） | 是否满足 `docs/05 §6.1 ⓔ` |
|---|---|---|---|---|---|
| 1 | 在搬侧 acquire **之后**（base rev 原样） | `MutexUnlock`（mode 0） | `a1d431f9a283` | **35–36 条红**（落点是 UB 残值） | 否 |
| 2 | 提前到写侧 acquire **之前** | `MutexUnlock`（mode 0） | `2d279dc01fa5` | 绿（20/20）但**可见性无契约**（复审/塔裁：不算修好） | **否**（ⓔ 选项 3） |
| 3 | 提前（同 2） | **`BufRelease<PIPE_S>`（drain）** ← **本轮交付** | `8310e71df5d8` | 绿（20/20；槽 = 计数器） | **是**（ⓔ 成对形态） |
| 4 | 在搬侧 acquire **之后**（同 1） | `BufRelease<PIPE_S>`（drain） | `1c0d3a6961e7` | **36 条红**（drain 覆盖不到它后面的写） | 否 |

读法：**第 4 行证明"挪位"必要；第 2/3 行一起证明"光挪位不够、必须换 drain 释放"** —— 第 2 行症状面与第 3 行
一样绿，所以**只靠 `M.moews` 绿无法区分**，这正是 r1 复审判"不算修好"的理由；补上 drain release 之后，
次序由**规则**（`docs/05 §6.1 ⓔ`）建立，不再只靠"提前量"。

## 5. 代价（trade-off，如实写）

- **性能**：写侧改用 drain release ⇒ `BufRelease<PIPE_S>` 要等 **S pipe 排空**才释放 token，消费端
  （MTE3）也跟着等；这削弱了"scalar 跑在指令前面"的发射重叠（`docs/05`「PIPE_S 同步限制」担心的正是这条）。
  本 mission **没有**做段级计时，**不给**耗时读数（只记这条结构性代价）；S3 的写 span 本身不长，
  真要定量需要 msprof 段级采样。
- **语义**：`BUF_AIV_IDX` 这条 token 在本段只被这一处 S→MTE3 交接使用（`grep -n "BUF_AIV_IDX" m15_moe_layer.h`
  的命中都在这一个 span 里），**没有**别处把它当纯互斥锁用 ⇒ 改 release mode **不影响其它使用者**。
- **配对**：写侧 acquire（`MutexLock<PIPE_S>`）保持不动（与 `BufAcquire<PIPE_S>` 同一 token 获取），
  所以不出现 `m15_ple.asc` 文件头 §BUF 记的"只放 release 而缺写侧 acquire ⇒ 挂死"那种形态。

## 6. 显式不做（塔裁的四条，逐条确认）

1. **未软化 `M.moews` 判据**：没有做 M102 列的 K1（跳过那 32 B）；判据文件不在本 mission 的 diff 里
   （`git status --short` 只有 `m15_layer_loop/lift_moe_segment.py`、`m15_layer_loop/m15_moe_layer.h`
   与新增的 `m15_layer_loop/evidence/moe_race/` 三项）。修完它才真的在比计数器，那才有意义。
2. **未动 `m15_layer_loop/m15_moe_host.h`**（判据住那里）。
3. **未动 `m13_moe_layer/m13_moe_layer.asc`**：上游同款缺陷本 mission 只在报告里指路
   （§6 第 1 条），不改上游、也不为此加生成器替换规则。
4. **未碰 `m15_layer_kernel.h` / `m15_layer_loop.asc` / `m15_ple*` / `m15_attn*`**（各有所属；
   /tmp 里对这些文件的改动只在「借 include 配方」范围内，且**只在 /tmp**，不入库）。

## 7. 未做 / 边界（诚实清单）

1. **上游 m13 的同一颗雷仍在**（本 mission 不越界改它）：`m13_moe_layer/m13_moe_layer.asc` 里
   同一函数的标量写仍在 `BufAcquire<PIPE_MTE3>(BUF_AIV_IDX);` 之后
   （base commit `20bd20d659b11cbb5e6e6d317921ecb3c16e2629` 上实测：`BufAcquire` 在第 1157 行、
   那次标量写在第 1171 行；行号基准 = 该不可变 rev，内容锚点见 §2①）。
   **影响面**：m13 自己的 `runs=all` 判据若也有跨路 ws 逐字节比，就有同款假绿风险；本 mission 未去核 m13 自己的判据。
   建议另立 mission（发现已按协议上报过，见 `.tower/comms/findings/` 里 M102 的两条）。
2. **`lift_moe_segment.py` 的模块 docstring 里「8 类替换」的计数未逐条重述第 9 类（M95 top-k 归并树）** ——
   该计数在本 mission 之前就不含 M95 那条。本 mission 只把自己那条挂在规则 7（M84）项下作为 `(e)`，
   **没有**去改别人的文档口径（避免越界改别人的叙述）。
3. **同一形态是否还有第二处**：我只核了 `IndexGenStage` 这一处（M102 的 survey 也只看这里）。
   「在 m15 device 代码里 grep 所有 `BufAcquire<PIPE_MTE3>` 之后才发生的标量 UB 写」这件事**未做**。
4. **抖动率**的统计只在**一个档位**（`M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 … all`、真实权重 manifest）上做；
   换档位会换调度，本结论不自动外推。
5. 修后「两侧都等于计数器」这一条用的是**本档 `badIds == 0`**（计数器真值 0）。「两侧都等于一个**非零**计数器」
   的形态，本档数据里没有脏 id ⇒ 用**拨值探针**（§4⑤）代替：把写进槽的值改成可辨常量，
   看它是否原样落盘（修后是、修前不是）。这**不是**脏 id 场景的实测，如实标注。
6. **"另外 10 处"标量写的顺序性仍登记为未确认**：本轮把写侧释放改成 drain 之后，那 10 处理论上落在
   同一条 drain 覆盖里（它们本来就在写侧 acquire 之后、释放之前），但**本 mission 没有对它们做受控 A/B**
   ⇒ 只记"被同一条 drain 覆盖"，不记"已证有序"（本目录也从未声称过；见 `mechanism_note.md` §2 的登记表）。
7. **drain 的性能代价未量化**：§5 只写结构性代价（scalar 要等 S pipe 排空），**没有**段级计时读数；
   要定量需要 msprof 段级采样，本 mission 未做。
8. **一档一形态**：本轮的设备读数只在「`M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 … all` + 真实权重
   manifest」这一档上取；换档位换调度，结论不自动外推（这也是本缺陷本身的性质：调度依赖）。

## 8. 本目录文件清单

**第 2 轮（本轮）新增/更新**：

| 文件 | 内容 |
|---|---|
| `run_c9_fix2_dump.log` | flip 基座 + **本轮形态** + 9 层落盘（**整份**）：`2068/291/0`，槽 = `00000000` |
| `run_tip3.log` | **交付形态**（本分支 tip 正常构建）（**整份**）：`2095/302/0`，`M.moews` 48/48 |
| `run_main_now.log` | **当前 main（`9296e79`）+ 本轮 artifact** 正常构建（**整份**）：`2095/302/0` |
| `run_c9_racy2.excerpt.log` | **反例**（写搬回 `BufAcquire<PIPE_MTE3>` 之后 + 保留 drain release）**节选**：36 条红 |
| `run_c9_fix2_probe.excerpt.log` | **拨值探针**（本轮形态 + 写值 `^0x5A5A5A5A`）**节选**：槽 = `5a5a5a5a` |
| `det20_form2.txt` | 本轮 §4④：同一二进制 20 次逐次读数（run.log sha256 + `M.moews` 行数/PASS 数 + 汇总行） |
| `generator_guard_mutation.log` | 生成器**四条**断言的咬合力（A2/B2/C2/D2 四个反例各自报错，本轮重取） |
| `runs_summary.txt` | §A = 本轮 25 档读数；§B = 第 1 轮 36 档读数（历史） |
| `slot_bytes.txt` / `jitter_unf.txt` | 第 2 轮段落（本轮 dump / 反例 / 探针）+ 第 1 轮原正文 |
| `ab_generator.log` | 第 2 轮 A/B（旧生成器 ⇒ base rev artifact、新生成器 ⇒ tip artifact、注释改动不改二进制、`reproduce_host.sh` 重跑读数） |
| `mechanism_note.md` | M107-r1 更正 → r1 复审 P1 → 本轮改 drain 的完整口径（含"另外 10 处"的逐行登记与代价） |

**第 1 轮（上一轮形态）留存的档案**（读数有效、形态已被本轮取代，保留作对照与历史）：

| 文件 | 内容 |
|---|---|
| `README.md` | 本文 |
| `addr_arith.cpp` / `addr_arith.md` | 地址自算程序与读数（诊断槽 UB `UB_IG_CNT+128` → x 窗内 2176；GM `988224`；x 行 0 覆盖判定） |
| `m17_precedent.txt` | m17 同款修法的逐字摘录（rev `20bd20d…`，行号 + 内容锚点） |
| `bases_diff.txt` | 两基座之间到底差了什么（只有 attention kv/cache 一族）⇒ 翻面的自变量 |
| `check_slot_bytes.py` | 从落盘 ws 里读 `988224` 与 `2176` 两处字节的读取脚本（偏移自算） |
| `reproduce_host.sh` | host 侧复现（干净树 → 借 include → 四档构建）；本轮的读数见 `ab_generator.log` §4 |
| `batch_log.txt` | 批脚本的逐条结果行（本轮 25 档全部首次进锁即取到 ⇒ 文件里只有结果行，没有重试行） |
| `run_c9_unf_dump.log` / `run_c9_fix_dump.log` | 第 1 轮：flip 基座未修（35 红）/ 上一轮形态（全绿）**整份** |
| `run_mn_unf_dump.log` | 第 1 轮：main 基座未修，`2095/303/0` 全绿而槽里是 x 残值 ⇒ **假绿现场** |
| `run_mn_fix.log` / `run_tip_normal.log` / `run_c9_fix_c.log` | 第 1 轮的零回归/交付形态档（**整份**） |
| `run_c9_D1.excerpt.log` / `run_c9_D2.excerpt.log` | 第 1 轮的两个 drain 对照档**节选**（D1 = **本轮交付形态**的同一二进制、D2 = **本轮反例**的同一二进制） |
| `run_c9_unf_mldump.excerpt.log` / `run_c9_fix_mldump.excerpt.log` | 第 1 轮的 9 层落盘档**节选** |
| `run_c9_fix_probe_dump.excerpt.log` / `run_c9_unf_probe_dump.excerpt.log` | 第 1 轮的拨值探针**节选** |
| `det20_fix.txt` | 第 1 轮的 20 次逐次读数 |
| `.gitignore` | 本目录不入库的 dump（`*.bin` / `dump_manifest.txt` / `moe_layout.txt`） |

（所有 `*.excerpt.log` 都是**节选**，不是整份 run.log；节选里每一行都是程序自己的原文，未改写。）
