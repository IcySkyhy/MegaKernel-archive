# M84 · `moe_relift/` —— MoE 段「规模无关前置修复」的临时证据区

本目录属于 mission **M84**（branch `feat/m84-moe-segment-scale-independent-prep-f`）。
M84 与 M82 / M85 **并行**写同一个 `m15_layer_loop/**` 目录，靠**文件级互斥**隔开；
`m15_layer_loop/evidence/**` 是 M82 的，所以 mission 约定的设备日志/转录放在本目录
（这是并行写作期的**临时安排**，不是长期归属）。

## 本 mission 改了什么（只动 2 个文件 + 本目录）

| 文件 | 改动 |
|---|---|
| `m15_layer_loop/m15_moe_layer.h` | 机械生成物（2025 → 2081 行），由下面的脚本重跑得到 |
| `m15_layer_loop/lift_moe_segment.py` | 新增「规则 7 = M84 规模无关前置修复（#5/#6/#7/#8）」及其产物断言 |
| `m15_layer_loop/moe_relift/**` | 本目录：判据脚本 + 设备/静态证据（**不入 `evidence/`**） |

> **第 2 轮增补（复审 r1 的 3 条 P2，见文末「第 2 轮」一节）**：`check_prep.py` 的判据集合与
> 汇总口径已改（**可 FAIL 的判据 16 条 + 诊断读数 4 条**，不再是「19 条判据」）；
> `IG_DIAG_SLOT_END` 改为由 `SZ_OFFSETS` 反推。生成物行数 2075 → **2081**。

四项修复（源头 = M77 勘查报告 §2 的逐点差异表；`#1/#2` 规模常量、`#3` 流式 router、
`#4` 归并树、以及 §2 第 14 项的重定尺**都不在本 mission**）：

- **#5** S3 索引生成的 `cnt[E] / off[E+1] / cursor[E]` 由 **AIV 标量栈数组**改为 **UB 静态槽位**
  （`UB_IG_SCAL` / `UB_IG_OFF` / `UB_IG_CUR`，从 `UB_IG_END` 起、**按 NUM_EXPERTS 定尺**；
  counts 与 offsets 起始各按 32B 对齐）。E=512 时栈数组 3×2KB = **6144 B**。
- **#6** 诊断槽的 GM 目的从写死 `offs[8]` 改为 `IG_DIAG_GM_SLOT`
  = `AlignUp((NUM_EXPERTS+1)*4 + 4, 32)/4`（仓内布局反推）：**E=4 → 8（与改前同一字节）**、
  E=512 → **520**。写死 8 在 E=512 会落进真 `offsets[0..512]` 里、覆盖 `expert_offsets[8]`。
- **#7** unpermute 的 K 链（`FmaChunk`）与模板实例从 `1..3`/`<1..4>` 展开到 `1..9` + `<10>`
  （topk=10；k=0 由 `Mul` 承担，故只需 `FmaChunk<1..9>`）。
- **#8** unpermute 的 Y 视图上界显式写成紧凑 Σt_e 上界 `TOTAL_MAX`（**不是** padded 的
  `NUM_EXPERTS*M_MAX` —— 两者在 E=4 相等只是巧合；E=512 时 640 行 vs 32768 行）。

**不动**：规模常量（`m15_moe_resources.h`，本 mission 无写权限）、紧凑 Σt_e 寻址
（`offGm.GetValue(slot)` / `rowBase`，M40 的硬约束）、router 与归并树。

## 本目录文件

| 文件 | 内容 |
|---|---|
| `check_prep.py` | 四项修复的静态判据：① 生成物的**文本形态**；② 把资源表与生成物里的 `constexpr uint32_t` 表达式**就地求值**（非副本），在**当前档 E=4/topk=2/TOPK_MAX=4** 与 **E=512 假设档**（`--E 512 --topk 10`，只重算算术、不改仓库常量）各算一遍布局。**可 FAIL 的判据 16 条**（另有 4 条恒真的**诊断读数**，单列、不计入判据数），rc 非 0 即 FAIL。 |
| `m84_verify.log` | 「命令 → 输出/rc」表：`--check`（两个 locale）、重新生成、构建、`runs=all`/`runs=chain`、静态判据（两个 locale 读数一致）。 |
| `m84_check_prep.log` | `check_prep.py` 的完整读数。 |
| `m84_accept_run_all.log` | `runs=all`（48 层 × 3 token）完整 stdout：**2012 判定项 + 160 guard，0 FAIL**。 |
| `m84_accept_chain.log` | `runs=chain` 完整 stdout：**910 判定项 + 52 guard，0 FAIL**。 |
| `m84_ab_dump_sha256.log` | A/B：① 改前/改后两个构建各 dump 一次，**1307 个文件 sha256 逐位相同**（含 `dump_manifest.txt`）；② 第 2 轮 r1/r2 同样逐位相同，且二进制 sha256 相同。 |
| `m84_neg_controls.log` | 6 条负向对照（变异**只在 `/tmp/m84_nc` 副本**上做），各咬中 1..2 条判据，含 #6 使用点的方向级错误（`nc6_diag_usage`）。 |
| `m84_dump_run.log` | 落盘那一次 `runs=all` 的 stdout（guard 由 160 → 161，多出的一条是 `Ch.m39.dump`）。 |
| `m84_evidence_audit.log` | **第 3 轮**：本目录每一个「命令 → 输出」对的回扫清单 —— 在 tip 上重跑、`LC_ALL=C` 与 `LC_ALL=C.UTF-8` 两档、逐条记「一致 / 已改」。 |

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <repo>
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/build -j4
LC_ALL=C python3 m15_layer_loop/lift_moe_segment.py --check          # 必须回绿
LC_ALL=C python3 m15_layer_loop/moe_relift/check_prep.py             # 16 判据 0 FAIL（+4 诊断读数）
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all
./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt chain
```

## 第 2 轮：复审 r1 的 3 条 P2 处置

复审对象 tip `49e2e77`（`reviewer-m84` r1，`p2-3items`，`merge`）。三条都不改交付语义，
只补强静态判据与一处常量形态；**二进制 sha256 与 r1 逐字节相同**（确定性构建），落盘 1307 文件 sha256 也与 r1 全等。
措辞按第 3 轮复审的观察订正为「**未改变可观测行为**」：`IG_DIAG_SLOT_END` 其实**进了代码表达式**
（`offsGm.SetGlobalBuffer(..., NUM_EXPERTS + IG_DIAG_SLOT_END)`，E=4 时该实参 20 → 16），
只是视图长度仅作元数据、不参与寻址，故二进制/落盘无差异。

- **P2-1（判据强度，#6）**：原先只钉「`offsGm[8]` 消失」，没钉「诊断槽 GM 目的**使用** `IG_DIAG_GM_SLOT`」；
  复审用 `/tmp` 变异实测「使用点改成 `offsGm[NUM_EXPERTS + 8]`」时仍 rc=0。现加两条判据
  （使用点计数 == 1 且无写死 `offsGm[8]`；不出现 `offsGm[NUM_EXPERTS`），同一变异现报 **2 条 FAIL、rc=1**
  （`m84_neg_controls.log` 的 `nc6_diag_usage`）。**该错误在 E=4 设备侧咬不住**（诊断值 `oobCountUb=0`
  且槽区本就为 0），能咬住它的只有这条静态判据。
- **P2-2（分栏计数）**：`Report` 改为「可 FAIL 的判据」/「诊断读数」两栏，硬编码 `True` 的行不再计入判据数。
  实测口径（脚本自己的汇总行）：**可 FAIL 的判据 16 条 + 诊断读数 4 条**。
  与复审的读数（「19 条中 2 条恒真 ⇒ 17 条」）差在**计数单位**：那两条恒真的检查各打印**两档一次**
  （E=4 与 E=512 各一行）⇒ 按**打印行**算是 4 条诊断读数。改前 19 行 = **15 判据 + 4 诊断读数**；
  加 P2-1 的 2 条判据后 = **16 判据 + 4 诊断读数 = 20 行**。两栏都不再把诊断行算进判据数。
  为免歧义，`m84_verify.log` 与 `m84_check_prep.log` 里放的都是脚本的原始汇总行。
- **P2-3（惰性，沿用 donor m17）**：`IG_DIAG_SLOT_END` 由写死 `16` 改为 `SZ_OFFSETS / 4 - NUM_EXPERTS`
  ⇒ `offsGm` 视图恒等于底层 `expert_offsets` 区（E=4：16×4 = 64 B = `SZ_OFFSETS`，原来 20×4 = 80 B 会超出底层；
  E=512：528×4 = 2112，与原来相同），并补 `static_assert(IG_DIAG_GM_SLOT + 1 <= NUM_EXPERTS + IG_DIAG_SLOT_END)`。

## 第 3 轮：复审 r2 的 1 条 P2（证据读数不可复现）+ 2 条不阻塞观察

复审对象 tip `6f51587`（`reviewer-m84b` r1，`p2-1items`，`fix-then-merge`）。

- **P2（读不可复现，`m84_ab_dump_sha256.log` 的归档对照块）**：原块写
  `$ join … | awk '$2!=$3' | wc -l` → **1131**，而该管道的真值是 **1**（`1131` 是 `join` 的输出行数、
  即两个清单的**名目交集数**）；且原文写的是**仓根不存在的** `evidence/dump_sha256.txt`（真路径是
  `m15_layer_loop/evidence/dump_sha256.txt`），照抄跑不出东西。**已改成命令与读数自洽的四条**：
  不匹配数 = **1**、不匹配行 = `dump_manifest.txt`、匹配数 = **1130**、名目交集数 = **1131**（单列并注明
  「这是 join 的输出行数，不是上面那三条管道的输出」），并补注路径与 `grep -E '^[0-9a-f]{64}  '` 过滤
  （不加过滤 `join` 会在 stderr 报 `input is not in sorted order`，因为归档头两行是 `#` 注释）。
  四条读数在 `LC_ALL=C` 与 `LC_ALL=C.UTF-8` 下逐条一致。
- **观察 1（措辞）**：已按上节说明改为「未改变可观测行为」（原「未进代码路径」机制上不准确）。
- **观察 2（汇总行的档数标签）**：`Report` 原先硬编码「`diag // 2` 个代码点 × 2 档」，
  在 `--E 4 --topk 4`（只跑当前档）时「2 档」不实。现改为**跟踪实跑档数**与**诊断代码点集合**：
  默认调用打印「4 条（2 个代码点 × **2 档跑**…）」，单档调用打印「2 条（2 个代码点 × **1 档跑**…）」
  （`m84_verify.log` §7 是该用例的实跑记录）。
- **回扫**：本目录每个「命令 → 输出」对都在 tip 上重跑过（本地判据各跑 `LC_ALL=C` 与 `LC_ALL=C.UTF-8`），
  逐条清单与处置见 `m84_evidence_audit.log`。

## 一条**不在本 mission** 的、本目录顺手登记的东西

`m15_moe_resources.h` §4 的注释（`:207`）仍写着 `UB_IG_CNT` 装 `counts / offsets / 诊断`，
但 #5 之后 counts/offsets 搬到了 `UB_IG_SCAL/UB_IG_OFF`（该文件本 mission 无写权限）。
另有 m17 README §1 第 7 行登记的**诊断槽 UB 标量写与 MTE3 抢跑**（m13 原样：
`bUb[32] = oobCountUb;` 落在 `BufAcquire<PIPE_MTE3>` 之后）—— 这两条都**只登记、未改**，
已按纪律用 `TowerFinding` 报出。
