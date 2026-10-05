# M26（M118 / B4）：MoE prefill 段（E=512 / topk=10，router 打分用 `Mmad`）

被测对象 = **`m15_layer_loop/m15_moe_prefill.h`**（交付到融合 TU 的同一份设备头），
本目录只是它的独立靶场（`docs/15` §M103-2.0 的 P1：Wave B 一律不碰 `m15_layer_loop/CMakeLists.txt`）。

**一句话结论**：**S2（router 打分）已在设备上按真实 shape 验证通过** ——
`x[64,2560] bf16 @ W[640,2560]^T bf16 → logits fp32` 由 **AIC 上的 `Mmad`** 算出，
判据 **32,768/32,768 逐元素在 T3 界内**（`max|Δ| = 2.4e-08`，容差占用 **0.0016**，余量 ≈ 600×），
top-k ids / weights / 共享门点积同样全绿（见 §2）。S1/S4 的段体已交付，其中 **S1+S3+S4 已在
r2 上设备**（`--stage 3` 链路臂，见 §2 档 B）；**S5–S10 只到编译**（§5 逐条写窄）。

---

## r2 修订（r1 复审 p1-1 + 4 条 p2 + 2 nit 的逐条处置）

| # | 复审条目 | 处置 |
|---|---|---|
| **P1-1** | S3（`IndexGenP`，AIV0 单核）**标量读回全量** ids/weights，而这些由**全部 AIV** 分行写出；屏障却落在 `ig.Run()` **之后** ⇒ 跨核数据竞态 | **已修**：在 `ig.Run()` **之前**补一条全体 AIV 屏障（r2 时代码里叫 `BarrierAiv<PIPE_S>(1)`，**r3 起改名/改为 `BarrierAivStep<PIPE_S>(1)`**，见 §r3 的顺手②），原 S3→S4 的屏障挪到槽 2；AIV mode0 用序重排为 `12→13→14→15→12→13`（相邻必不同），`PfRingAdjacentOk` 断言仍绿。**并补了能咬住它的设备档**：新增 `--stage 3`「链路臂」（唯一执行 `ig.Run()` 的臂）+ 新增 J5–J9（用**设备自己的 ids** 现算计数排序/permute 的参考并逐位比对）——AIV0 若读到陈旧行，这族判据会红（r2 时这只是机制论断；**r3 已补坏实现的设备直证**，见 §2 的「变异档」）。 |
| **P2-1** | `Permute/Unpermute/CombineStage` 是 `using` 直接复用，GM 视图仍是 decode 的 256 行，真 shape 需要 640；头注却写"已改成 TP_MAX" | **已修**：真复制为 `PermuteP`/`UnpermuteP<TK>`/`CombineP`（§4b），视图按 `TP_MAX=640`/`MT*TOPK=640` 定尺；头注与事实对齐。 |
| **P2-2** | 义务①（AIC 打平）实际仍是 5/28，README §5.3 低报 | **已改**：§5.3 给出三条 N 条带的**真实块数**（router 5/28、gate_up 5/28、down 10/28）+ 三条打平路径 + `E_PAD` 多算 127 列的代价。 |
| **P2-3** | 挂载补丁把 B4 插在相位 A 之后，与 kernel 自己登记的"先 B5（`FLAG_HC0_BOUND_AIV`）再 B4（`FLAG_HC2_BOUND_AIV`）"不符 | **已改**：补丁 A 改成**贴着 `FLAG_HC2_BOUND_AIV` 插**并显式带上该相位边界（`m15_moe_prefill_host.h` §4）。 |
| **P2-4** | README §3 把 `--stage 2` 的覆盖讲宽（与 §5.1 矛盾） | **已改**：§3 换成覆盖表：`2` = **只** S2/S2b；`3` = 链路臂 S1+S2/S2b+S3+S4；`99` = 只编译。 |
| nit-1 | `check_ref.py` 的 J1 标签在 `mt` 行上算 `n_safe`（m=1 时打印 "59/1 行"） | **已修**：`safe = safe_full[:m]`，标签与 `tie-risk` 报告项都只在 `[0,m)` 上算。 |
| nit-2 | 4 个负向对照只污染 logits ⇒ ids/weights 的判定未被证伪 | **已补**：`negctl` 扩到 7 个变体，新增 V5（改一行 ids）/ V6（挪一个权重）/ V7（行内两列对调）⇒ J1/J1b/J4/J5–J9 的可证伪性有设备档见证。 |
| **新规** | 「矩阵乘法⇒cube；其余用 VF；**scalar 只做控制流**」 | **已自查并改**：`IndexGenP` 原来用 scalar 做**权重的 bf16 RNE 位打包**（`(bits+0x7FFF+((bits>>16)&1))>>16`）= 用标量算数据 ⇒ **删掉整个 `w_tk` 平面与打包**，改由 `UnpermuteP` 在 **VF** 里 `Cast` 到 bf16 再展回 fp32（与 donor 的舍入点**逐位相同**）。其余 scalar 用法逐个核过：`IndexGenP` 的计数/前缀和/cursor 是**索引推导**、`PermuteP`/`UnpermuteP` 的 `GetValue` 是**地址**、`VecQuantP` 的行数来自 counts 是**控制流** ⇒ 均在允许侧（`evidence/consts.txt` 的平面表已无 `PWS_WTK`）。 |

---

## r3 修订（r2 复审 p2-2items 的处置 + 变异档直证）

| # | 条目 | 处置 |
|---|---|---|
| **P2-1** | README §1(c) 的 `PWS_BYTES` 是 r1 的旧值（13,596,480），与同表引用的 `consts.txt`（13,592,384）矛盾 | **已改**为 **13,592,384**（`m26_consts` 的机器读数；差的 4,096 B 正是 r2 删掉的 `SZ_WTK`） |
| **P2-2** | README §2 档 D 写"7 绿"，而它引用的 `check_ref_chain_m1.log` 是 6 guard | **已改**为 **6 绿**，并注明 `G7` 在 m<8 档按 `check_ref.py` 的规则降为**报告项** |
| 顺手① | `m15_moe_prefill.h:17` / `:489` 的 S3 小节仍写产出 `w_tk`（r2 已删该平面） | **已删**该词并注明平面取消 |
| 顺手②（复审残余 2） | `PfRingAdjacentOk` 只校验 4 元 ring 的**环邻**，绑不住 6 步的执行序 | **已改**：改成 `PF_AIV_BARRIER_SEQ[6]` 表 + `BarrierAivStep<pipe>(step)` —— 调用点只写**步号**（编译期常量），flagId 一律取自表；`PfSeqAdjacentOk` 对**实际长度**的序列逐对断言相邻不同，另加两条"表 ↔ ring 一致"的 static_assert。**能绑**：步号→id 的映射、相邻两步不同 id、表与 ring 一致；**不能绑**：调用点写错步号（源码级错误，仍靠复审逐点核对） |
| 顺手③（复审残余 5） | `run_probe.sh` 头注写"两个设备档"、README §3 写"4 个负向对照"、§6 的证据文件清单是 r1 的 | **已按实际改齐**（四档 / 7 个变体 / 现文件名 + 新增的两个变异档日志） |
| 顺手④（复审残余 4） | `source_sha256.txt` 的 `git_head` 是**运行时**的 HEAD（不是被审的 tip），`npu_smi` 行含表格竖线 | **写清口径**（见 §2 末注）：权威部分是那 4 个 blob 的 sha256（r3 时与 tip 逐一相等；**M188 起订正，见 §r6**）；`run_probe.sh` 起把 `npu_smi` 行裁成干净读数（`git_head` 仍如实记运行时 HEAD，不再假装是被审 tip） |
| **残余①（复审自登的限度）** | "没有做删掉这道屏障 ⇒ J5–J9 变红的直证" | **已补**：新增 `run_mutant.sh` + 两个变异档读数，见 §2「变异档」（含"只删屏障但窗口没开 ⇒ 3 次绿"的如实记录） |

---

## r4 修订（r3 复审 p2-2items：两条纯文字 + "同类复扫"）

r3 复审的两条都是**README 的陈述与归档产物不一致**，且都不碰被指纹钉住的 4 个源文件
（`m15_moe_prefill.h` / `_res.h` / `m26_moe_prefill.asc` / `check_ref.py`）⇒ 本轮**只改文档文字、不重跑设备**：
r4 时那 4 个 blob 的 sha256 与 r3 归档的 `source_sha256.txt` 逐一相等（**M188 起订正**：其中 `m15_moe_prefill.h` 因 M187 注释清扫而变动、已重钉，另 2 个 blob 自 M170/M155 起与 main 不一致 —— 见 §r6）。

| # | 复审条目 | 处置 |
|---|---|---|
| **P2-1** | 本文档曾称变异脚本的树核对含 `grep MUTANT`（"这两个检查都印在变异日志里"），而 `run_mutant.sh` 的恢复段**只印 `restored_sha256`**，归档的两份变异日志里也没有这一行 | **按归档改齐**：改成"树已恢复的证明是变异日志里的 `restored_sha256`（== 变异前的 sha256）"，并写明恢复段只印这一项、归档日志里没有 `grep` 那一行。**选择改文字而不是改脚本再重跑**：后者要动设备，且会让脚本与既有归档日志互相不一致 |
| **P2-2** | 本文档的 wide 档曾写"有效运行 3 + 另一轮 2""判据 3/3（另一轮 2/2）"，而归档只有**一轮** 3 次（`N=3` 写死、重跑覆盖同一份日志） | **删掉没有产物的"另一轮 2"**：该行改成 `3` 与 **`3 / 3`**（只保留归档在案的那一轮） |

### 同类复扫（人类口径"检查其他地方有没有同样的问题"）

扫的对象：本文档里**所有指向归档产物的陈述**（"读数见 …""…印在…日志里""另一轮 N 次""经复核"
"`…` 的 rc=0" 等），逐条与 `m26_moe_prefill/evidence/` 的实际内容对照。**共 14 条**：

- **与产物不一致 5 条**：复审点名的 2 条（上表），复扫**另发现 3 条** ——

| 扫到的陈述 | 实际产物 | 处置 |
|---|---|---|
| `source_sha256.txt` 的 `git_head` 是"塔的 WIP 快照 `332778a`" | 归档里是 **`git_head=ef2e670`**（r2 的提交；r3 的提交在这次运行之后） | 改成实际值，并保留"它是运行那一刻的 HEAD、不是被审 tip"这层意思 |
| §5.1 "`evidence/build.log` 的 rc=0" | `build.log` 以 `[100%] Built target m26_moe_prefill` 收尾，**没有 `rc=` 行**（那行由 `run_probe.sh` 印到控制台） | 改成"以 `[100%] Built target …` 收尾"，并注明 `build rc=0` 印在控制台 |
| §6 的证据清单只列到 `mutant_no_pre_ig_barrier.log` | 目录里还有 `mutant_no_pre_ig_barrier_wide.log` | 清单补上该文件 |

- **措辞无产物支撑 1 条**（不算"不一致"）：§5.1 的"`NormStage`/`MXFP4GemmItem` 两处**经复核**
  不属于该问题" —— "复核"没有产物可指 ⇒ 改成"**不属于该问题**（`MT == M_MAX`；MXFP4 是形状模板
  —— 直接由源码可核，非日志产物）"。
- **核对一致 8 条**（不改）：`consts.txt` 的 `PWS_BYTES=13592384` 与其余峰值读数；`source_sha256.txt`
  的 4 个 blob（r4 时与 tip 逐一相等；**M188 起订正，见 §r6**）；四个 `check_ref_{router,chain,router_m1,chain_m1}.log` 的判定项/guard
  计数与 `被测源 sha256`；`negctl.log` 的 V1..V7 全 rc=1；`run.log` 的四档与锁内 `npu-smi`；
  两份变异日志的 `mutant_judge_red_count`（`0 / 3` 与 `3 / 3`）与 `Σt_e`（`640` 与 `20`）；
  `npu_smi` 行的干净读数；两份变异日志的 `SCRIPT ERROR` 未出现、各有 3 条真判据输出。

### r3 复审的三条次要观察（**保持现状**，不扩大改动）

| 观察 | 为什么不改 |
|---|---|
| `run_mutant.sh` 的 `flock -w 180 …` 未检查返回码（等锁超时会让判据循环打出 `SCRIPT ERROR`，摘要行有歧义） | 归档的两份日志没有走这条路（`SCRIPT ERROR` 未出现、各有 3 条真判据）；改脚本会让脚本与既有归档日志互相不一致，而本轮纪律是只改文档文字 |
| 变异直接改被跟踪文件而非用 include 遮蔽 | 同上；`trap` 恢复 + `restored_sha256` 核对已在归档里，复审也确认这套机制有效 |
| wide 档日志标题的"只差这一处"措辞（wide 实际是两处改动：自旋 + 删屏障） | 同上（标题属于脚本文字，日志是历史产物） |

> 这三条与 r3 复审的结论一致：**不影响任何读数或结论**；本轮按塔的纪律不做扩大改动。

---

## r5 修订（M170：M85P include 修复 —— 两个 target 曾编不过）

**问题**（finding `20261004-agent-flagreg-bug-m26-moe-prefill-m26-consts-m15-layer-resources-h-m85p-m26-tu.md`，severity=medium）：
本工程两个 target 在 main 上编译失败。`m26_moe_prefill.asc` / `m26_consts.asc` 经
`m15_moe_prefill.h:66 → m15_layer_resources.h` 引入资源表，而 §4 的 `FLAG_SEQ[]`
（`m15_layer_resources.h:728-737`）直接引用 `M85P::FLAG_B2/B3/B4`。`M85P` 只由
`m15_ple_wire.h` 提供，两个 `.asc` 从不 include 它 ⇒ 报 `use of undeclared identifier 'M85P'`
及其级联错误。

**复现（M170 实跑；基线 = 本分支 base `94b2a1b`）**：

| 命令 | 修前 | 修后 |
|---|---|---|
| `cmake --build m26_moe_prefill/build -j4`（默认目标） | **rc=2**（5 处 `M85P` + 4 处级联） | **rc=0**，0 error |
| `cmake --build m26_moe_prefill/build -j4 --target m26_consts` | **rc=2**（同样 5+4 条） | **rc=0**，0 error |

修前逐条（两个 target 同形；`文件:行` 取实测编译器输出）：

- `m15_layer_resources.h:728:28`、`:729:27`、`:730:27`、`:731:28`、`:737:28` —— `error: use of undeclared identifier 'M85P'`；
- 级联：`:797:39 invalid application of 'sizeof' to an incomplete type 'const M15L::FlagStep[]'`、
  `:863:15`、`:898:15`、`:1221:15` —— `static assertion expression is not an integral constant expression`。

**修法（scope = `m26_moe_prefill/**`）**：在两个 `.asc` 的 include 块里，于 `m15_hc_layer.h`
之后、`m15_moe_prefill.h` **之前**补 `#include "m15_ple_wire.h"`（该头自身要求先有
`m15_hc_layer.h`，包含序已满足）。**未改** `m15_layer_loop/**`（在飞的 M168 占用
`m15_layer_resources.h` / `m15_layer_kernel.h`）。

**防回退自检**：`m26_moe_prefill/check_build.sh` —— configure + 两个 target 构建，任一失败即
非零退出。咬合力实测：删掉两个 `.asc` 里的 `#include "m15_ple_wire.h"` 后脚本 **rc=2**，报同一批
`M85P` 错误；恢复后 rc=0（日志 `/tmp/m170_selfcheck_{red,green}.log`）。

**回归（M170 实跑）**：

- `m26_moe_prefill` / `m26_consts`：rc=0、0 error；warning **21 条/target**，其中 **7 条/target**
  来自新引入的 `m15_ple_wire.h`（该头此前不在本 TU 的包含闭包内），均为既有的
  `-Wcce-compat`（`expected type of ops in vector loop cond is uint16_t`）。
- `m15_layer_loop` 主工程（`cmake --build m15_layer_loop/build -j4`，默认目标含 `m15_layer_loop`
  / `m15_ple` / `m15_attn_core`）：**修前/修后均 rc=0、0 error、32 条 warning**；`m15_layer_loop`
  二进制 sha256 修前 = 修后 = `d2cc02d9b82e86f29c87c3daa57c4b7c5fd676ad443cd12f04f3b068bd7cdb1a`
  （逐字节相同）。

**同类问题全仓排查**（人类口径「检查一下其他地方有没有同样的问题」）：

排查方法：把 `m15_layer_loop/` 作为 include 根，从每个 `.asc` 起对 `#include "…"` 求传递闭包，
看是否到达 `m15_layer_resources.h`、以及闭包里是否含 `m15_ple_wire.h`（即 `M85P` 是否可见）。
全仓 `.asc` 里**只有 3 个 TU 到达**该资源表；其中只有 m26 的两个闭包不含 `m15_ple_wire.h`：

| TU（CMake target） | 到达资源表 | `M85P` 可见 | 实跑构建 | 处置 |
|---|---|---|---|---|
| `m15_layer_loop/m15_layer_loop.asc`（`m15_layer_loop`） | 是（直接 `:93`） | 是（`:86` 直接 include） | rc=0 | 不动 |
| `m26_moe_prefill/m26_moe_prefill.asc`（`m26_moe_prefill`） | 是（经 `m15_moe_prefill.h:66`） | **否** | **rc=2** | 补 include（本次） |
| `m26_moe_prefill/m26_consts.asc`（`m26_consts`） | 是（同上） | **否** | **rc=2** | 补 include（本次） |
| `m15_layer_loop/m15_ple.asc`（`m15_ple`） | 否 | 是 | rc=0（同一次 `cmake --build`） | 不动 |
| `m15_layer_loop/m15_attn_core.asc`（`m15_attn_core`） | 否 | 否 | rc=0（同一次 `cmake --build`） | 不动 |
| m24 / m25 / m27 的 `.asc`（Wave B/C 靶场） | 否 | 否 | 不在枚举范围（闭包不到达资源表） | 不动 |
| m13 / m17 / m18 / m23 / m28 的 `.asc` | 否 | 否 | 不在枚举范围（闭包不到达资源表） | 不动 |

> 判定依据是**包含闭包**（不依赖"某个 grep 未命中"）：`m15_layer_resources.h` 的 quoted
> includer 全仓只有 3 处 —— `m15_layer_loop.asc:93`、`m15_moe_prefill.h:66`、
> `m15_moe_prefill_res.h:33`；而 `m15_moe_prefill.h` 又只被 `m15_layer_loop.asc:95`、
> `m26_moe_prefill.asc`、`m26_consts.asc` include（`m15_moe_prefill_host.h:25` 那份无任何
> TU include）。故受影响面 = 上表前 3 行。

**未采纳的替代修法（follow-up 建议）**：把 `m15_layer_resources.h` §4 里对
`M85P::FLAG_B2/B3/B4` 的直接引用改成本文件的字面量，并在能看到 `M85P` 的 TU
（`m15_layer_loop.asc`）里用 `static_assert` 把字面量与 `M85P` 常量钉死。它能让 m26 的 TU
不必引入 `m15_ple_wire.h`（连带去掉上面那 7 条/target 的既有 warning），但要改
`m15_layer_loop/m15_layer_resources.h` —— 该文件不在本 mission scope（M168 在飞占用），
故只作建议、本次未落。

---

## r6 修订（M188：M187 注释清扫后的源码指纹重钉）

**来源**：reviewer-m187 在 M187（release 措辞清扫）复审里报的非阻塞 TowerFinding —— sha256 覆盖注释，
M187 对注释的改动使被 committed 源指纹钉死的文件哈希变了：`evidence/source_sha256.txt:6`
（`m15_layer_loop/m15_moe_prefill.h`）。M187 只动注释、不进编译 ⇒ **二进制与全部读数不变**。
本 mission 只做「指纹 / 引用文本」的重新对齐，未改任何设备代码/注释，未改任何已提交读数。

**本工程内的重钉（M187 所致）**：

| 指纹文件:行 | 被钉文件 | 记录值（M187 前，== main `dc045ae`） | 重钉值（M187 后，tip `95671e7`） |
|---|---|---|---|
| `evidence/source_sha256.txt:6` | `m15_layer_loop/m15_moe_prefill.h` | `e7167e4b…` | **`e9223756…`** |

复算命令（任一，输出须与右列一致）：

```bash
git show feat/m187-release-mode-module-readme-comment:m15_layer_loop/m15_moe_prefill.h | sha256sum
sha256sum m15_layer_loop/m15_moe_prefill.h    # M187 合入 main 之后
```

**本文件内与 M187 无关的既有陈旧**（报塔另派，本轮未改其行值）：

| 指纹文件:行 | 被钉文件 | 记录值 | 当前实际值 | 陈旧原因 |
|---|---|---|---|---|
| `evidence/source_sha256.txt:8` | `m26_moe_prefill/m26_moe_prefill.asc` | `84fb0435…` | `e14fb7f6…` | M170 补 `#include "m15_ple_wire.h"` |
| `evidence/source_sha256.txt:9` | `m26_moe_prefill/check_ref.py` | `dc898eda…` | `3ec3bb55…` | M155 判据补强 |

> 上述两行在本指纹生成（`git_head=ef2e670`）之后被改过，而 `source_sha256.txt` 未随之重新生成；
> 按塔纪律不在本 mission 处置（避免把改动扩散到其它工程的合格证），已在 review-request 里带 文件:行 报塔。

**跨工程、由 M187 注释清扫同样影响的指纹**（不在本工程 scope，列出以便对照）：
`m29_ple_unit/evidence/r1/sha256.txt:1`（已随本 mission 在 m29 侧重钉）、
`m27_hc_prefill/evidence/sha256.txt:5`（`m15_hc_prefill.h`，在 main 上就已陈旧，另行处置）。

---

## 1. 融合清单（`docs/15` §M103-2.7 的 (a)–(f) 固定小标题）

### (a) 挂载点
- **相位 / 入口**：`m15_layer_kernel.h` 的 prefill 入口 `m15_layer_kernel_gdn_prefill` /
  `m15_layer_kernel_attn_prefill`（M110 落地）里的**相位 B**。
  M110 现在的 `M15L_PrefillBody<KIND>`（`m15_layer_kernel.h:592-604`）只跑相位 A，
  并在 `:597-600` 注明了 B5/B4 的挂载点 ⇒ B4 挂在**相位 A 之后的相位 B**。
- **消费的 GM 平面**：`xLayer`（本 tile 的 MT 行层输入 bf16）、`resZero`（全 0 bf16）、
  `gamma1/gamma2`（每 HIDDEN 一个 bf16）、MoE 的 12 个权重指针（decode 档同一批）、
  **新增 1 个**：`routerWpad`（见 (b)）。
- **生产的 GM 平面**：`yLayer`（S10 出口 bf16，层间残差流）+ 内部 ws 平面（§(c)）。
- **默认关**：整段由 `pfWired`（= `.asc` 的 `M15_PREFILL_WIRE`）门控 ⇒ `runs=all` 的 decode 零回归。

### (b) `LayerArgs` 需要的字段（Wave A 照它加）
复用 M110 已加的 13 个 prefill 字段里的 12 个；**B4 只新增 1 个**：

| 字段 | 类型 | 语义 |
|---|---|---|
| `pfMoeRouterWPad` | `__gm__ uint8_t*` | **pad 后的 router 权重平面** `[E_PAD=640, HIDDEN=2560] bf16`：行 0..511 = 512 个专家（模型原始 `mlp.gate.weight`），行 512 = 共享门（`mlp.shared_expert_gate.weight`），行 513..639 = 复制行 512（mmad 的 N 方向无尾块，须可读） |

其余 12 个（`ws / xLayer / yLayer / resZero / gamma1 / gamma2 / wGu / sGu / wDn / sDn /
wGuShd / sGuShd / wDnShd / sDnShd / m / topk`）全部沿用既有字段（decode 的 MoE 权重同一批）。
组板用 `M15PFH::BuildRouterWPad()`（`m15_moe_prefill_host.h` §2）。

### (c) 资源窗
| 资源 | 定尺 | 峰值 | 预算 |
|---|---|---|---|
| UB（**prefill MoE 相位独占**） | `M15PFR::PF_UB_PEAK` = **149,632 B** | 149,632 B | 248 KB（`M15M::UB_BYTES_TOTAL`） |
| L1（router 的 A 窗 @0 与 MXFP4 GEMM 的 A 区**同址叠放**（相位分隔）；B 窗 @128K 落在未用区，不叠放） | `M15PFR::L1R_END` = **163,840 B** | 163,840 B | 512 KB |
| L0C（router tile `MT×RB_N` fp32） | `M15PFR::MT*RB_N*4` = **32,768 B** | 32,768 B | 256 KB |
| GM ws（每 tile 一块，所有平面顺排） | `M15PFR::PWS_BYTES` = **13,592,384 B** | — | 独立缓冲 |

> **r3 更正（复审 r2 的 P2-1）**：本单元格原写 `13,596,480`（r1 的旧值），而同一张表声明
> "全部是编译期常量本身打出来的" —— 与 `evidence/consts.txt` 的 `PWS_BYTES=13592384` 矛盾
> （差的 4,096 B 正是 r2 删掉的 `SZ_WTK`）。现值取自 `m26_consts` 的**机器读数**。

> 上表**全部是编译期常量本身打出来的**（不是手算）：`./m26_moe_prefill/build/m26_consts`
> 的读数见 `evidence/consts.txt`。

> **UB 窗复用**：本段与 decode 档 MoE 段**同址叠放**（两者只在各自的入口里出现）。
> `PF_UB_PEAK` 已带断言；`PF_UB_BYTES_MOE` 槽（`m15_layer_resources.h:505`）由 Wave C 填 149,632。
> **注意一个真实的收口点**：本段的 UB 窗从 `UB_VEC = 36,864` 起，与 GDN-preflill 段（B1）
> **不同相位**，故不冲突；但若 Wave C 要把 hc/MoE 的窗**并列**在同一相位，必须先重排（本表已给峰值）。

### (d) BufferID 清单（核内）+ flagId 清单（核间）
核内 BufferID：**全部复用 decode 档 MoE 段已登记的 id**（AIC 0-6；AIV 0/1/4/5/15-18/19/20/21），
在 `m15_moe_prefill_res.h` §5 以 `PFR_BUF_*` 别名登记（**不新占编号**，峰值数不变）。

核间 flagId（`(核型, mode, id, pipe)`）：

| 序 | 段 | 核型 | mode | id | pipe | 邻近性说明 |
|---|---|---|---|---|---|---|
| 1 | AIC: router mmad 完成 → 全体 AIC 对齐 | AIC | 0 | 8 | set `PIPE_FIX` / wait `PIPE_S` | |
| 2 | AIC → 配对 AIV：logits 就绪 | AIC | 2 | 0 | set `PIPE_MTE2` | AIV 侧同 id wait（`PIPE_MTE2`） |
| 3 | AIV: S1 后全体 AIV 对齐 | AIV | 0 | 12 | set `PIPE_MTE3` / wait `PIPE_MTE2` | 与 #2 不同 id ✓ |
| 4 | **AIV: topk 后、`ig.Run()` 之前**（全体 AIV 对齐） | AIV | 0 | 13 | wait `PIPE_S` | ≠12 ✓ ／ **r1 复审 P1-1 的修复点** |
| 5 | AIV: S3（`ig.Run()`）后 → S4 | AIV | 0 | 14 | wait `PIPE_S` | ≠13 ✓ |
| 6 | AIV: permute 后 | AIV | 0 | 15 | wait `PIPE_MTE2` | ≠14 ✓ |
| 7 | AIV → AIC：A 就绪 | AIV | 2 | 1 | set `PIPE_MTE3` | ≠15 ✓ |
| 8 | AIC: S6 前对齐 | AIC | 0 | 9 | set `PIPE_MTE2` / wait `PIPE_S` | ≠8 ✓ |
| 9 | AIC: S6 后对齐 | AIC | 0 | 10 | set `PIPE_FIX` / wait `PIPE_S` | ≠9 ✓ |
| 10 | AIC → AIV：GU 就绪 | AIC | 2 | 2 | set `PIPE_MTE2` | AIV wait `PIPE_MTE2`；≠1 ✓ |
| 11 | AIV → AIC：H 就绪 | AIV | 2 | 3 | set `PIPE_MTE3` | ≠2 ✓ |
| 12 | AIC: S8 前对齐 | AIC | 0 | 11 | set `PIPE_MTE2` / wait `PIPE_S` | ≠10 ✓ |
| 13 | AIC: S8 后对齐 | AIC | 0 | 8 | set `PIPE_FIX` / wait `PIPE_S` | ring 回到 8；与 #12 不同 ✓ |
| 14 | AIC → AIV：Y 就绪 | AIC | 2 | 0 | set `PIPE_MTE2` | ring 回到 0；与 #13 不同 ✓ |
| 15 | AIV: unpermute 后 | AIV | 0 | 12 | wait `PIPE_MTE2` | ring 回 12；≠15(上轮) ✓ |
| 16 | AIV: combine 后 | AIV | 0 | 13 | wait `PIPE_MTE2` | ≠12 ✓ |

> **为什么 #4 必须在 `ig.Run()` 之前**（r1 复审 P1-1）：ids/weights 由**全部 AIV** 分行写出
> （`nAiv = 2*numBlocks` = 56），而 S3 由 AIV0 独占并**标量读回全量** `S = m*topk` 条。
> donor 里 router 与 idxGen **同在 `if (isPrimary)`**（`m15_moe_layer.h:2082-2090`）⇒ 写者与读者同核、
> 天然不需要屏障；B4 把 router 改成多核分派（交付义务②）后这是**新引入**的跨核依赖。
> 原先屏障落在 `ig.Run()` **之后** ⇒ AIV0 可能读到尚未落地的行（静默错路由，S4–S10 全错）。

- **相邻性**：AIV 序列 `12→13→14→15→(ring)→12→13…`、AIC 序列 `8→9→10→11→8…`、
  mode-2 序列 `0→1→2→3→0…`，三者在 `m15_moe_prefill_res.h` §5 有 `static_assert` 机器化见证
  （`PfRingAdjacentOk`）。
- **无跨模式复用**：AIV 的 mode-0 id ∈ {12,13,14,15}、mode-2 id ∈ {0,1,2,3}（**不相交**）；
  AIC 同理（{8..11} vs {0..3}）⇒ 不需要援引 `docs/05` §2 的"同 id 跨模式须前一模式全 drain"弱前提。
- 每 id 每相位用量 ≤ 2 ≪ 15（硬件 4 bit 计数器）。

### (e) 需要相位边界的位置
- **入口处**：`.asc` 的 `runs=prefill` 分支在调用 prefill kernel 之前保证 ws/权重已 H2D（host 侧）。
- **相位 A → 相位 B**：`PipeBarrier<PIPE_ALL>`（M110 的 `M15L_PrefillBody` 里已有）。
- **段内**：见 (d) 的 #1/#7/#8/#11/#12（全体 AIC 对齐）与 #3/#5/#14/#15（全体 AIV 对齐）。
- **相位 B → 后续段**：段尾 `PipeBarrier<PIPE_ALL>`（本段自己出）。

### (f) `m` / `pos` 语义
- 本段接受 **m ∈ [1, M_PREFILL=4097]**；**按 `MT = 64` 行分块**，每块 `curM = min(MT, m - blk*MT)`。
- 块内所有行索引张量的定尺 = `MT`（不是 `M_PREFILL`），Σt_e 的定尺 = `TP_MAX = MT*TOPK = 640`
  （= **active_num 上界**，不是 padded 的 `E*MT = 32768`）⇒ B4 交付义务③。
- **本段不消费 `pos`**：MoE 段是逐 token 语义（路由/permute/unpermute 都在 token 内），
  位置只影响 attention 段的 KV 落点 ⇒ `pfPos/pfPosBase` 由 B2/B3 消费。
- `topk` 是**运行期参数**（`docs/15` §M103-2.2 不许退化成常量）；真实档 = 10。

---

## 2. 本轮读数（设备档，真实权重 + 真实 shape）

**四个设备档**（各自一次进锁、单进程、`timeout`；读数见 `evidence/check_ref_{router,chain,router_m1,chain_m1}.log`）：

| 档 | 命令 | 跑了哪些段 | 判定项 / guard |
|---|---|---|---|
| A | `--m 64 --stage 2` | S2（AIC `Mmad`）+ S2b（AIV top-k） | **5/5 PASS**（34,702 元素 0 超界） / 6 绿 |
| B | `--m 64 --stage 3` | 链路臂：S1 + S2/S2b + **S3** + S4 | **10/10 PASS**（1,676,047 元素 0 超界） / **7 绿** |
| C | `--m 1 --stage 2` | 同 A（顺带验 `calcM` 抬到 ≥2 的 3510 quirk） | 5/5 PASS / 6 绿 |
| D | `--m 1 --stage 3` | 同 B | 10/10 PASS / **6 绿**（`G7` 在 m<8 档按 `check_ref.py` 的规则降为**报告项**，不计入 guard） |

权重与输入**全部来自真实 checkpoint**（`m17_moe_real/m17_weight_manifest.txt` →
`/workspace/Qwen3.8-Flash-Next-MXFP4` 的 layer 0 `mlp.gate.weight` / `mlp.shared_expert_gate.weight` /
`gamma1`（hc_norm 行切片）/ `embed_tokens.weight` 前 64 行）；源码指纹见 `evidence/source_sha256.txt`。

### 判定项（档 B，`m=64` 链路臂）

| 判定项 | 结果 |
|---|---|
| `J1 topk_ids`（safe 行 59/64，逐位） | **590/590 PASS** |
| `J1b topk_ids`（全部行，集合口径） | **640/640 PASS** |
| `J2 router_logits`（T3，逐元素） | **32,768/32,768 PASS** |
| `J3 sgate 裸点积`（T3） | **64/64 PASS** |
| `J4 topk_weights`（T3，解析梯度上界） | **640/640 PASS** |
| `J5 S3 counts[E]`（T1，逐位） | **512/512 PASS** |
| `J6 S3 offsets[E+1]`（T1） | **513/513 PASS** |
| `J7 S3 perm_src / perm_expert`（T1，稳定计数排序） | **1280/1280 PASS** |
| `J8 S3 inv_slot`（T1：`(t,k)` 在 permuted 序里的位置） | **640/640 PASS** |
| `J9 S4 xsorted[pos] == xnorm[perm_src[pos]]`（T1，**逐字节**） | **1,638,400/1,638,400 PASS** |

> **J5–J9 就是"能咬住 P1-1"的那一族判据**：它们只用**设备自己的 ids**（不依赖 tie-risk 的 J1）
> 现算计数排序/permute 的参考。AIV0 若在 S3 里读到尚未落地的行，算出的 counts/perm/inv
> 就是"别的 ids 的计数排序"，与用 dump 出来的 ids 现算的参考不一致 ⇒ 这五条会红。
> **r3 已把这一步从机制论断补成设备直证**：见 §2 的「变异档」（删掉这道屏障 + 人为拉宽竞态窗后，
> `J5–J9` 每次都红、`Σt_e` 从 640 掉到 20；而只删屏障不拉宽窗的 3 次是绿的 —— 说明窗口没开，不是判据不敏感）。
> 档 B 是**唯一**执行 `IndexGenP` 的臂（`G7` 同时见证覆盖成立：`nAiv = 56`、`m = 64`）。

### 报告项（档 B）

| 报告项 | 读数 |
|---|---|
| logits `max|Δ|`（vs fp64 参考） | **2.4163e-08** |
| logits 容差占用（`max |Δ|/tol`） | **0.0016**（余量 ≈ 600×） |
| weights `max|Δ|` | **1.2713e-08** |
| sgate `max|Δ|` | **1.1560e-09** |
| S3 `Σt_e` | **640** = `m*topk` ✓ |
| min 10/11 名间隔（fp64 参考） | **6.7040e-06** |
| tie-risk 行数（间隔 ≤ 4×容差） | **5 / 64**（如实计入报告项，不计入 FAIL） |

**T3 的 ε 是推导的、不是实测的**（`check_ref.py` 顶部）：
`ε_acc = γ_2559 = 2559·u/(1−2559·u)`，`u = 2^-24` —— 依据是
① bf16×bf16 的乘积在 fp32 里**精确**（8+8 ≤ 24 位尾数）；
② 40 个 K-block 的顺序累加共 2559 次 fp32 加法，最坏相对误差 ≤ γ_2559。
`J5–J9` 是**索引/计数数据**（不是算术量）⇒ 按 T1 **逐位**判，不设容差。

**guard（非空洞性）7 条全绿**（档 B）：
G1 logits 行内跨专家非常量（最小极差 `3.6e-02`）、G2 跨行非常量、
G3 ids 全在 `[0,E)` 且行内互异、G4 weights 行和为 1（`max|Σ−1| = 1.19e-07`）、
G5 权重非退化、G6 sgate 跨行非常量（极差 `4.4e-03`）、
**G7 S3 的跨核覆盖成立**（`nAiv = 56 ≥ 8` 且 `m ≥ 8` ⇒ "多写者读"这条依赖真的被跑到了）。

**负向对照 7 个变体全部 rc=1**（把判据的**被测读数在内存里**弄坏，不写任何文件）：
V1 单点挪 1% / V2 整列置零 / V3 末行复制首行 / V4 全置常数（以上污染 `logits`）；
**V5 改一行 ids / V6 挪一个权重 / V7 行内两列对调**（r1 复审 nit-2：后三个专为证伪
`J1/J1b/J4` 与 `J5–J9` 而加）。读数见 `evidence/negctl.log`。

### 变异档（r3 新增：判据"坏实现 ⇒ 红"的设备直证）

`bash m26_moe_prefill/run_mutant.sh` 有两个模式（脚本会把**变异前后只差这一处**的 `diff -U2` 全量印进日志，
跑完立刻从备份恢复并用 sha256 核对树）：

- `MUT_MODE=no_barrier`（默认）：只删 `BarrierAivStep<PIPE_S>(1)` 这一行（r2 的 P1-1 屏障），按**真实时序**跑；
- `MUT_MODE=no_barrier_wide`：再让**非 0 号 AIV** 在其 `topk.Run` 之前自旋 20,000 次 `PipeBarrier<PIPE_ALL>`
  —— **人为**拉宽竞态窗。加这一档只为把"窗口没开"与"判据不敏感"分开，不是对生产行为的声明。

| 变异 | 有效运行 | 判据红 | 判据形态 |
|---|---|---|---|
| `no_barrier`（真实时序） | 3 | **0 / 3** | 10/10 PASS、7 guard 绿（与正常档同） |
| `no_barrier_wide`（人为拉宽窗） | 3 | **3 / 3** | `J1–J4`（router/topk）**全 PASS**；`J5–J9`（S3/S4 的索引数据）**全 FAIL**；报告项 `S3 Σt_e = 20`（期望 640） |

读法（这几条要一起看）：

1. `no_barrier_wide` 的红法是**竞态机制的直接签名**：top-k 本身没错（J1–J4 绿），而 AIV0 在 S3 里
   只看到少数几行已落盘的 ids（`Σt_e = 20` 而不是 640）⇒ counts/perm/inv/xsorted 全错。
   ⇒ **这道屏障的必要性有了"坏实现 ⇒ 红"的设备直证**（复审 r2 自登的那条限度已补上）。
2. `no_barrier`（真实时序）3 次全绿：**不能**据此说屏障可以去掉 —— 它只说明**本档时序下窗口不会自然打开**
   （AIV0 的 640 次标量 GM 读比其它 AIV 的 MTE3 落盘慢得多）。同一竞态在拉宽窗口后**每次都红**。
3. 判据对"读到错 ids"的敏感性另有一条**独立**见证：负向对照 `V5`（只把一行 ids 换掉）
   就会把 `J1/J1b/J5–J9` 打红（`evidence/negctl.log`）。
4. 变异档跑完的树核对：**树已恢复的证明是变异日志里印出的 `restored_sha256`**（它 == 变异前的 sha256）。
   `run_mutant.sh` 的恢复段**只印这一项**（另有 `git status`/`git diff --stat` 两行诊断）；归档的两份
   变异日志里**没有** `grep MUTANT` 这一行 —— 变异标记只出现在 `diff` 打出的 `+ // [MUTANT] …` 与
   结论行的 `MUTANT BITE / DID NOT BITE` 里。r3 复审的 P2-1 指出的就是这处措辞，已按归档改齐。

> **`evidence/source_sha256.txt` 的口径**：归档里的是 **`git_head=ef2e670`** —— 那是**运行那一刻**的 HEAD
> （r2 的提交；r3 的提交是在这次运行之后做的），**不是**被审的 tip。**权威部分是那 4 个 blob 的 sha256**；
> 这 4 个值在 r3/r4 时与 tip 逐一相等，**M188 起已订正**（见 §r6）：① `m15_layer_loop/m15_moe_prefill.h`
> 因 M187 的注释清扫而变动，该行已重钉；② `m26_moe_prefill/m26_moe_prefill.asc` / `check_ref.py` 自
> M170 / M155 起就已与 main 不一致（既有陈旧，非 M187 所致）。`npu_smi` 行在 r3 起已裁成干净读数
> （归档里是 `npu_smi 25.7.rc1.10 Version: 25.7.rc1.10`，不再含表格竖线）。

---

## 3. 复现

```bash
# 一键：构建 → 四档设备（锁内，一次进锁一条短命令）→ 判据 → 7 个负向对照 → 归档到 evidence/
bash m26_moe_prefill/run_probe.sh 64
# 或分步
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m26_moe_prefill/build -S m26_moe_prefill -DCMAKE_BUILD_TYPE=Release
cmake --build m26_moe_prefill/build -j4
cd m26_moe_prefill/build && flock -w 900 /tmp/npu0.lock \
  ./m26_moe_prefill --manifest ../../m17_moe_real/m17_weight_manifest.txt --m 64 --stage 2 --out m26_out
/usr/local/python3.12.13/bin/python3 m26_moe_prefill/check_ref.py m26_moe_prefill/build/m26_out
```

资源读数（不碰设备、不需要锁）：

```bash
cmake --build m26_moe_prefill/build -j4 --target m26_consts && ./m26_moe_prefill/build/m26_consts
# → evidence/consts.txt（PWS_BYTES / PF_UB_PEAK / L1R_END / L0C tile 等，全部是编译期常量本身）
```

构建防回退自检（M170；只做 host 侧编译，不碰设备）：

```bash
bash m26_moe_prefill/check_build.sh   # configure + 两个 target；任一失败即非零退出
```

`--stage` 是 bring-up 旋钮（**覆盖口径按 §5.1，不夸大**）：

| `--stage` | 实际执行 | 本轮是否上设备 |
|---|---|---|
| `2` | **只**跑 `RouterMmadStage`（AIC）+ `RouterTopkStage`（AIV）：S2/S2b | ✅ 档 A |
| `3` | **链路臂**：`MoePrefillChain` 到 stageLimit=3 ⇒ S1 + S2/S2b + **S3** + S4（唯一执行 `IndexGenP` 的臂） | ✅ 档 B |
| `4` | 活性探针（AIV 只把 logits 的 32 B DMA 到 sgate 平面，不等 AIC） | 诊断用 |
| `99` | **只编译不运行**的实例化臂：把 S1–S10 整条链的代码生成纳入编译单元 | ❌（只到编译） |

`--m` ∈ [1,64]（单 tile）。**S5–S10（量化 / gate_up / down / unpermute / combine / S10）本轮仍未上设备。**

---

## 4. 器件侧的三个真实坑（本轮实测踩到并修掉，供后续段复用）

1. **UB 窗必须逐窗 32B 对齐**。`Sort32` / `StoreAlign` / `DataCopyPad` 的 UB 地址是硬要求；
   首版把 `UB_PF_VAL` 写成 `UB_PF_SG + 4`（1 个 fp32），**后续所有窗整体偏 4 B** ⇒
   AIV 每次搬运都 `Trap`，现象是**"kernel 正常返回、但 AIV 的产物一个字节都没写"**
   （三个平面全是毒值 `0xCD`）。现在 `m15_moe_prefill_res.h` 有 `PfUbAlignedOk()` 的
   **逐窗 32B 对齐 `static_assert`**，这类错误在编译期就会红。
2. **`__mix__(1,2)` 的 kernel 必须 `AscendC::InitSocState()`**（`m15_layer_kernel.h` 的每个入口都调）。
3. **`DataCopyPad` 的 GM→UB 方向要 4 参形式**（带 `DataCopyPadExtParams<T>`），
   3 参形式只在 UB→GM 方向编译得过。

诊断手法（低成本、可复制）：`--stage 4` 是一条**活性探针** —— AIV 只把 logits 的 32 B DMA 到
`sgate` 平面（不等 AIC）。`sgate` 一旦非毒 ⇒ AIV 分支在跑、DMA 可用 ⇒ 问题在段内；
仍是毒 ⇒ AIV 分支没跑到。这一步把"没跑"与"跑了但算错"一次分开。

---

## 5. 未完成项（**写窄**，不声称完成）

1. **S5–S10 的段体已交付且编译通过，但本轮仍未上设备验证。**（r2 起 **S1–S4 已上设备**，见 §2 档 B）
   它们由「复用 donor 算件 + 按 E 改写的拷贝」组成：
   - **复用**（`m15_moe_layer.h`，只读不改）：`NormStage`（S1/S10，视图按 `M_MAX=64` = 本段 `MT`，**无需改**）、
     `MXFP4GemmItem`（S6/S8，形状模板、与 E 解耦 ⇒ 视图无问题）、`PrecastGammaF32`。
   - **真复制 + 改 E / 改定尺**：`IndexGenP`（S3）、`VecQuantP`（S5/S7，另去掉 prefill 不需要的
     `PAD_SRC/PAD_DST`）、`PermuteP`/`UnpermuteP<TK>`/`CombineP`（S4/S9，见 P2-1）。
   - **视图收口已完成**：r1 复审 P2-1 指出的三处"视图说小"（256 vs 640）在 r2 已由真复制消除；
     `NormStage`/`MXFP4GemmItem` 两处不属于该问题（`MT == M_MAX`；MXFP4 是形状模板 —— 直接由源码可核，非日志产物）。
   - **编译见证**：`--stage 99` 强制实例化整条链；`evidence/build.log` **以 `[100%] Built target m26_moe_prefill` 收尾**
     （`build rc=0` 那一行是 `run_probe.sh` 印到**控制台**的，不在该文件里 —— r4 复扫时按归档改准）。
2. **S3（计数排序）仍是 AIV0 单核**（与 decode 档同形）。B4 交付义务②的"计数排序多核化"**未做**；
   已做的多核化只有 router（AIC 的 N 方向条带 + AIV 的逐行分派）。
3. **AIC 打平（交付义务①）没有达成，实测读数如下（不是"只到 N 条带"这句模糊话）**：
   设备档 `nblk = 28` 个 AIC；本段的 AIC 活数由**N 方向的块数**决定，而三条 N 条带的块数是
   `RB_NBLK = E_PAD/RB_N = 640/128 = 5`（router）、`GU_NBLK = 1280/256 = 5`（gate_up）、
   `DN_NBLK = 2560/256 = 10`（down）⇒ **router 段 5/28 个 AIC 有活、gate_up 5/28、down 10/28**。
   这与 `docs/15` §M103-2.2 B4 行交付义务①说的"现在只有 5/28 个 AIC 有活"**是同一个数**
   ⇒ **该义务未被 B4 解决**。要打平至少需要（按代价从低到高）：
   ① 降 `RB_N`（128→64/32 ⇒ router 的块数 5→10/20，B 的 L0B 占用同步下降）；
   ② 对 K 再切一刀（split-K + 跨 AIC 的 fp32 归约，需要 L0C→GM 的部分和平面）；
   ③ down 的 `DN_NBLK=10` 之外再做 M 方向的工作项切分。
   **另**：`E_PAD = 640` 相对 `E+1 = 513` 多算了 127 列（≈20% 的 N 是做无用功），
   这是"N 方向无尾块"的代价（`RB_N=128` 的整数倍）——降 `RB_N` 到 32 可把它压到 543/640。
   同一段里 S6/S8 的槽循环仍是**逐专家串行提交**（空专家靠 O(E) 标量读跳过，E=512 下每 AIC 512 次）；
   未做 worklist / 前缀和驱动的分配。
4. **m > 64 的多块循环未接**：`MoePrefillChain` 现在按**单块**（m ≤ MT）编排；
   `PF_BLK_MAX = 65`（m=4097）的逐块循环（每块重新 `Init` 并复用同一 ws）**已设计但未实现**。
   ⇒ 本轮设备档只覆盖 m=64（与"真实 shape"的 E/topk/几何无关的那一维）。
   （M155 已把**判据侧**的逐块/首块支持与命名接口做好，见 §7.4；**落盘侧仍未接线**。）
5. **精度档位**：router 的 logits 走了 **fp32 Fixpipe（`NoQuant`）**，判据按 T3（不是 T1）——
   因为 mmad 的 K 内累加次序与 fp64 参考不同（`docs/20` WO-A2 的判据①允许走 T3 + 推界）。
   `topk_ids` 在 **safe 行**上要求逐位（59/64 行；5 行因 10/11 名间隔只有 6.7e-06 而无法要求）。
6. **`m15_layer_kernel.h` / `m15_layer_loop.asc` 未改**（不在 scope）：§1 的挂载点补丁文本
   （`m15_moe_prefill_host.h` §4）**未贴**，故本段现在**还不能从融合 TU 跑到**——
   能跑的只有本目录的独立靶场（同一份头）。

---

## 6. 文件与来源

| 文件 | 角色 | 来源 |
|---|---|---|
| `../m15_layer_loop/m15_moe_prefill.h` | **设备段体**（交付到融合 TU 的头） | 新写 + 复制改造（见 §5.1 的逐件清单） |
| `../m15_layer_loop/m15_moe_prefill_res.h` | prefill 段自用资源表（UB/L1/L0/GM/BufferID/flagId + 断言） | 新写；`MWP_*` 等**只消费** `m15_layer_resources.h`（Wave A） |
| `../m15_layer_loop/m15_moe_prefill_host.h` | host 侧定尺/组板/启动参数 + **挂载点补丁文本** | 新写 |
| `m26_moe_prefill.asc` | 独立靶场（kernel + `main()`） | 新写；host 段照 `m18`/`m17` 的 manifest/pread/launch 范式 |
| `check_ref.py` | numpy float64 参考 + 判定项/报告项/guard + 负向对照 | 新写；容差口径照 `docs/17` 与 `m18_gdn_prefill/check_ref.py` |
| `CMakeLists.txt` / `.gitignore` | 独立工程 | 照 `m18_gdn_prefill/` 的模板 |
| `run_probe.sh` | 一键取证（含源码指纹） | 照 `probe_cube_fp32/run_probes.sh` 的锁内单进程范式 |
| `m26_consts.asc` | 只打印资源头的编译期读数（不碰设备） | 新写 |
| `check_build.sh` | **防回退构建自检**（configure + 两个 target；缺 `m15_ple_wire.h` 即红） | M170 新写 |
| `run_mutant.sh` | **变异档**：临时去掉 r2 的 `ig.Run()` 前屏障并重跑链路臂，期望判据变红（判据可证伪性的直证） | 新写 |
| `tools/gen_judge_fixtures.py` | **M155 合成夹具**（逐块覆盖 + 输入面消费见证），只喂判据、不涉设备 | 新写 |
| `run_judge_gap.sh` | **M155 离线自检**：归档回归 + J1c 负向对照 + 夹具演示；任一断言不符即 `exit 1`（可传播失败） | 新写 |
| `evidence/` | 读数归档（`source_sha256.txt` / `build.log` / `run.log` / `consts.txt` / `check_ref_{router,chain,router_m1,chain_m1}.log` / `negctl.log` / `mutant_no_pre_ig_barrier.log` / `mutant_no_pre_ig_barrier_wide.log` / **`judge_gap.log`**） | — |

**不触碰**：`m15_layer_loop/CMakeLists.txt`（归 M101）、`m15_moe_layer.h` / `lift_moe_segment.py`（归 M105）、
`m15_moe_resources.h`、Wave A 的四个文件、`NUM_EXPERTS` / `TOPK_MAX`（需改先报塔）。

---

## 7. M155 判据空档补强（判据侧；dump 侧未接线）

来源：M152 survey 逐条点在 `check_ref.py` 上的三个空档（① `n_safe==0` 时 `J1` 整条被跳过、
② 只对最后一块对拍、③ 判据吃的是跑完后回读的 `m26_x`）。本 mission **只改判据侧**
（`check_ref.py` + `tools/gen_judge_fixtures.py` + `run_judge_gap.sh`），**不动设备、不动 `m15_layer_loop/**`**。

### 7.1 空档的现场取证（带 文件:行）

① **`J1` 被跳过**：`m26_moe_prefill/check_ref.py:371-374` —— `safe = gap > 4×row_tol`、
`n_safe = count(safe)`、`if n_safe > 0` 才入判定项。p12 档 `tie-risk 1/1 ⇒ n_safe=0`，
`ids` 只剩 `J1b`（集合，忽略列内顺序）判。**漏放的具体错**：把某行 top-k 的第 0/1 列对调
（集合不变）。本 mission 前的哨基线：同一归档 `m1_p12/mut1` 上 `--negctl 7` 是 `RESULT: PASS`
（`J1` 被跳过、`J1b` 是集合判）；同一条命令现在被 `J1c` 判红，读数在 `evidence/judge_gap.log` 的 B 段。

② **只覆盖最后一块**：dump 侧 `m15_layer_loop/m15_layer_loop.asc:4040-4045` 的 `H_PfMoeDump26`
取 `row0=(nTiles-1)*mt`、`rows=m-row0`，只落这一块；`m26_meta.txt` 写的是这一块的 `rows`
（`:4088-4091`），**meta 里没有总行数 / 块号 / 块数** ⇒ `m>MT` 时首块与其余块没有数值判据
（m4097 档读的是 `row0=4096` 的最后 1 行）。

③ **读哪份输入不可区分**：判据用 dump 出来的 `m26_x.bin`（`.asc:4070-4073`，跑完后 D2H 回读）
现算参考；AIC 真正消费时读的那份面没有第二份证据 ⇒ FAIL 只能记成"错"，
分不出"算错"还是"读错输入"。

### 7.2 (a) J1c —— 行内 top-k 顺序/置换判据（`check_ref.py:377-401`）

与 `J1b` 的**集合**判互补：`J1c` 只在「设备集合 == 参考集合」的行上判**列内顺序**
（集合错的行交给 `J1b`，不把两种错混在一起）。判法是逐相邻对：参考降序的第 j 与 j+1 两个 id，
在设备序里的相对位置必须一致；只对**参考间隔 > 4×行容差**的相邻对判（tie 对不判）。
它比 `J1` 的「整行 safe」门细一档 ⇒ `n_safe==0` 的档里也有效。
`pairs_total==0` 时**不入判定项**，改为报告项「J1c 行内顺序判据未生效」，不拿空判据冒充通过。

### 7.3 负向对照：J1c 能变红（`--negctl 8`）

`negctl_ids` 新增 V8（`check_ref.py:194-218`）：**只在 `J1` 被跳过的行**上对调 top-k 第 0/1 列。

`safe` 只覆盖被判的 `[0,m)` 行，而 ids 有 `MT` 行；**尾块 `MT > m` 时** V8 按
`[0, min(MT, len(safe)))` 对齐，`≥ m` 的 pad 行不属于被判行、**不置换**。若被判行没有一行被
`J1` 跳过（`safe` 全 True），该负控在此档**无意义** ⇒ 判据报报告项「负向对照未生效：V8 不适用」，
**不静默空转、也不抛 `IndexError`**（r1 复审 p2-1 的修法）。

| 命令 | rc | 红的是谁 |
|---|---|---|
| `m1_p12/mut1 --negctl 8` | 1 | **只有 J1c**（其余 4 条判定项 + 6 条 guard 仍绿）⇒ 是 J1c 在抓，不是 J1 |
| `m1_p12/mut1 --negctl 7` | 1 | J1c（该档 `J1` 被跳过） |
| `m1_p8/mut1 --negctl 8` | 0 | 被判行都被 `J1` 判（`safe` 全 True）⇒ 报「负向对照未生效」，不假红 |
| `m1_p8/mut1 --negctl 7` | 1 | J1 与 J1c 都红 |
| `tail_plain --negctl 8`（`MT=8 > m=4`，被判行全 safe） | 0 | 报「负向对照未生效」（`MT > m` 分支，不崩） |
| `tail_tie --negctl 8`（`MT=8 > m=4`，被判行 tie-risk） | 1 | **只有 J1c**（pad 行未被置换）⇒ `MT > m` 的置换路径也对 |

签名行在 `evidence/judge_gap.log`：`SIGNATURE-B1` / `SIGNATURE-B4` / `SIGNATURE-B5` /
`SIGNATURE-C5` / `SIGNATURE-C6` / `SIGNATURE-C7`。
`tail_tie` / `tail_plain` 两个夹具即 r1 复审要求的 `MT > m` 用例。

### 7.4 (b) 逐块/首块：判据侧接口（dump 侧未接线）

判据侧接口（`discover_blocks` `check_ref.py:213-230`；`coverage` `:233-262`）：

- **多块**：`<dump>/m26_meta.b<i>.txt` + `m26_{x,logits,ids,w,sgate}.b<i>.bin`（`m26_wpad.bin` 共享）。
  每块 meta 新增字段：`blk=<i>`（0 基块号）、`n_tiles=<N>`（= `ceil(m_total/MT)`；缺时由 `m_total`+`MT` 现算）、
  `row0=<i*MT>`、`m_total=<M>`；`m` = 本块有效行数、`MT` = 本块缓冲行数（尾块同样按 `MT` 行落盘）。
  判据对每块用**它自己的 x** 现算参考并对拍。
- **legacy 单块**：只有 `m26_meta.txt` 时，判据报「**只覆盖 1 块**，无法从 meta 判定首/末块
  （落盘实现为最后一块），`m>MT` 时首块及其余块无判据」——不再静默只看最后一块。
- **覆盖报告恒出**：多块时给「已判块 / 共 N 块 / 缺哪些块」与「首块(blk=0) 已覆盖 / 未覆盖」。
- **bin 缺失的块**记为 `块数据缺失（SKIPPED）`；全部缺时 `RESULT: SKIPPED`（rc=2），
  不抛 traceback、也不假报。

**待 dump 侧接线后才能验证**：本 mission **没有**改 `H_PfMoeDump26`（不在 scope），
所以「首块真的被落盘、且判据读到的就是首块」这一条**未在设备上验过**；判据侧用合成夹具
（`tools/gen_judge_fixtures.py` 的 `blocks/`，块 0、2 在档、块 1 缺）演示到
「会逐块判、会报缺块、会报首块覆盖」这一层。

### 7.5 (c) W1 —— 输入面消费见证（`check_ref.py:266-309`）

证据源任选其一：`m26_x_consumed[.b<i>].bin`（消费前 x 快照）或 meta 的 `x_consumed_sha256`
（消费时校验和）。提供快照且与回读面不同时，判据用**快照**重算一份参考，看 logits 与哪一份一致：

- 与快照一致、与回读面不一致 ⇒ 报「**读错输入**」；
- 与两者都不一致 ⇒ 报「**算错**」；
- 两者都能解释、或不一致方向相反 ⇒ 如实报该情形。

无任何证据源 ⇒ 报「**未提供**（无快照、无 checksum）⇒ 不可区分」。
`W1` 是**报告项**，不改变判定（旧档 PASS/FAIL 不变）。

**待 dump 侧接线后才能验证**：消费前快照要 AIC 在 router 读 x 之前落一份（M152 的 S3 片），
本 mission 没做；合成夹具（`consumed_read/`、`consumed_calc/`、`no_witness/`）演示三种判别。

### 7.6 回归：旧结论不变（读数 `evidence/judge_gap.log`）

| 档 / arm | 旧 rc | 新 rc | 变的是什么 |
|---|---|---|---|
| m1_p8 clean/mut1/mut2 | 0 | 0 | 判定项 5→6（+J1c）；报告项 +3（J1c 覆盖 / W1 / 覆盖范围） |
| m1_p12 clean | 1 | 1 | 判定项 4（J1c 未生效 ⇒ 入报告项） |
| m1_p12 mut1/mut2 | 0 | 0 | 判定项 4→5（+J1c） |
| m1_p15 clean | 1 | 1 | 同 p12 clean |
| m1_p15 mut1 | 0 | 0 | 判定项 5→6 |
| m1_p15 mut2 | 1 | 1 | J1c 也红（原 J1 已红） |
| m4097_p15 clean/mut1/mut2 | 0（m140 README §6 表的末块） | **2（SKIPPED）** | **唯一的判定口径变化**：归档只入 meta、`*.bin` 不入 git ⇒ 洁净树上离线重跑**读不到数据**，如实记 SKIPPED；归档的末块结论仍以 m140 README §6 表为准，不被本判据改写或背书 |

逐档解释：

- p12 clean / p15 clean 的 `J1c` **未生效**（集合已经不一致 ⇒ 门 1 不满足）⇒ 新增判据**不能**
  替它们区分顺序；它们的 FAIL 仍由 `J1b`/`J2`/`J3`/`J4` + `G5` 给出。
- p15 mut2 原本 `J1` 已红（6/10），`J1c` 又红 ⇒ 结论不变，但多一条对顺序的读数。
- m4097 的 SKIPPED 是**数据缺失**，不是判据改判；旧档的 PASS 结论在洁净树上不可复算
  （bin 不入 git），故本轮**不引用它作通过**。

### 7.7 适用范围与空档（**写窄**）

- `J1c` 只在「参考间隔显著 + 集合一致」的相邻对上判 ⇒ 参考自己并列的对、以及集合错的行，
  它都**不判顺序**。
- `J1c` 用参考值排序 ⇒ 若参考本身错（x 读错），它给出的顺序也可能与设备一致而**不红**；
  它抓的是"设备算出的序与它自己那份参考序不一致"，不抓"参考整体错"。
- 块覆盖：判据能逐块判，但**逐块落盘由 dump 侧决定**；本 mission 只把判据侧与接口做好，
  **未接线即未验证首块**。
- `W1`：无快照/checksum 时**没有**判别力，只能记「不可区分」；提供时判的是
  "logits 与哪一份输入自洽"；若设备既读错又算错（两处都不一致），归**算错**（如实，不夸大成定位）。

### 7.8 复算

```bash
bash m26_moe_prefill/run_judge_gap.sh        # 零设备；证据 → evidence/judge_gap.log；断言不符 exit 1
python3.12 m26_moe_prefill/tools/gen_judge_fixtures.py /tmp/fx   # 只生成夹具（含 tail_tie/tail_plain 的 MT>m 用例）
python3.12 m26_moe_prefill/check_ref.py <dump> [--negctl N]      # 单档判据；N=8 为 (a) 的专用负向对照
```

归档 dump 由 `M140_ARCH`（默认 `../m15_layer_loop/evidence/m140_prefill_full_layer`）指定。
`evidence/judge_gap.log` 的头字段是 **`head_at_generation=`**（生成该日志时的 HEAD）——日志在
commit 之前生成，故它与本文件所在 commit 的 tip 不同；权威 provenance 锚点是同文件里的
`judge_sha256`（与本目录 committed 的 `check_ref.py` 一致）。
禁用词规范六词（逐字 pattern 见本次 review-request）在本目录被改文件上扫过，未命中。
