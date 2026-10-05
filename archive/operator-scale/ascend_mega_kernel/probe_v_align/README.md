# probe_v_align —— M57：逐 API 的 UB 对齐要求探针（dav-3510 / Ascend950PR）

> **一句话**：把"哪些 SIMD Vector function 的 load/store 能吃非 32B 对齐的 UB 地址、哪些必须 32B、
> 边界到底在哪一位"从"散落的经验"变成**逐 API 可重跑、逐字节可复核的实测**；
> 顺带把 M35 交接的 `Interleave`/`DeInterleave` 语义悬案做成**可判定的正例+反例**。

所有结论**只有两种来源**，全文分栏不混排：
* **[实测]** —— 本目录的探针在本机 NPU 上跑出来的日志/dump（`evidence/` 里逐条可查）；
* **[文档转述]** —— 来自 `docs/05` §6.1 / 用户 2026-09-26 澄清，**未由本探针独立验证**者一律标注（见 §7）。
  §7 是**转述栏**，不回填进 §3 的实测表。

---

## 1. 交付物与目录

| 文件 | 作用 |
|---|---|
| `probe_v_align.asc` | 主探针：对齐/长度矩阵（`kOps` **44 个条目**（44 个不同名字，`arange`/`arangef32` 共用 `OP_ARANGE`）；`list` 打印 **218 条变体**、覆盖 **43 个不同 op 名**（`arangef32` 不在矩阵里）与 **49 个条目组**（full 4 / ad 28 / len 6 / pt 8 / nc 3；复算 `grep -oE '\((full\|ad\|len\|pt\|nc):[a-z0-9_:]*' evidence/logs/matrix.txt \| sort -u \| wc -l` → `49`，分前缀 `… \| sed 's/^(//; s/:.*//' \| sort \| uniq -c` → `28 ad / 4 full / 6 len / 3 nc / 8 pt` —— **字符类只用 ASCII**，故与 locale 无关：`LC_ALL=C`、`POSIX`、`C.UTF-8` 三者读数相同）；`./build/probe_v_align list` 逐条打印），一个变体一次 launch |
| `probe_v_ildl.asc` | `Reg::Interleave`/`Reg::DeInterleave` 语义标定（8 个用例，逐 lane 解码） |
| `run_probes.sh` | **一键复现**：构建 → 逐变体运行（逐变体独立进程）→ ildl → 两个独立复核（约 15 分钟；共享卡上更久） |
| `run_ildl.sh` | 只重跑 ildl 部分 |
| `check_ref.py` | 主探针的**独立复核**：从归档 dump 用 numpy 重算逐字节模型（T1）+ 故障分类交叉核对 + 最小对齐推断 |
| `check_ref_ildl.py` | ildl 的**独立复核**：逐 lane 解码出真实映射并与候选映射比对 |
| `tools/make_tables.py` | 把 `evidence/logs/matrix.txt` 汇总成 §3 的表（规则写在脚本里，reviewer 可逐项复算；含「判定方式」列） |
| `tools/check_archive_matches_list.py` | **归档 vs 当前代码一致性守卫**（把"归档是否由当前代码产出"变成可复核守卫；`b9d468f` 评审的 P2-1）：行序逐项一致 + 每条 log 的 aclError/FNV 与矩阵列一致 + guard②；带负向对照 `--selftest` |
| `tools/diff_archives.py` | **归档差异记账**（复核用）：按 git 状态位 + **掩码后的全部非空行**把改动分成"变体 log 结果变化 / 变体 log 仅易变字段或措辞 / 其它"。**掩码按两条规则**：① 易变字段按标签（时间戳/PID/设备序号/stream_id/report_stream_id/task_id/program id/flip_num/hash/device_id/同步超时/core id）；② 地址类**按片段** —— 只掩**地址语境标签之后紧邻的那串值**（`pc start`/`current`/`sc、su、mte、vec、cube、l1 error info`/`aic error mask`/`para base`），**同一行其它位置的 `0x…` 不动**（若整行掩就会把 `DUMP_FNV` 指纹一起吃掉）。**已知边界**：若真实测量值恰好写在这些标签之后会被掩掉 —— 对现有归档的三条可达性核查（命令与逐字读数见 `tools/diff_archives.py` 的「边界」段，本行不重复）：① 地址语境标签之后的值只有 `pc start`/`current`/`sc、su、mte、vec error info`/`aic error mask`/`para base`（**各 46 条**，即规则 ② 替换的对象）外加 `subErrType`（46 条，值 `0x4`，**不在标签表里 ⇒ 参与比对**）；② 不带冒号的"标签 + `0x…`"有 `DUMP_FNV` **83 条**与 `DUMP_FNV32` **7 条**（后者在 ildl 日志里）⇒ 两者都**参与比对**（命令用 `[A-Za-z_0-9]{3,}`，见 docstring）；③ `DUMP_FNV` 落在地址语境行上的次数在该归档内为 **0**。据此该归档里没有参与比对的项落在会被掩的位置；但这是**可达性事实（只覆盖当前归档）、不是保证**；工具是 ref-vs-ref、不看工作树未提交内容；判别性的设备 `errcode`/`error code` 活在 `FAULT_MSG` 续行上，所以比的是整份文件而不是几类前缀行。口径：**只有 `RESULT:` 行不同算"措辞"**（不算结果变化）。0 项与"子树写错"分别给 `RESULT: EMPTY` / `RESULT: SKIPPED` 且 **rc=2/1**，不发合格证。带 `--selftest`（/tmp 丢弃式仓库逐字段注入：FNV、errcode、COMPARE、OUTCOME、VARIANT、HEAD32、FAULT_CODE、errorStr、纯噪声，另加两个状态控制） |
| `evidence/` | 归档：逐变体原始 log（含 `FAULT_MSG` 原文）、1024B OUT arena `.bin` dump、矩阵、两个复核输出 |

```bash
cd probe_v_align
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PY=/usr/local/python3.12.13/bin/python3   # 本机 /usr/bin/python3 **没有 numpy**（复核脚本依赖它）
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8
bash run_probes.sh                      # 全量 + 证据归档（共享设备上会被别人的 kernel 排队）
# 只跑某几个 op（**写 evidence_partial/，不动归档**）：
bash run_probes.sh 'gather|scatter'
# 单变体手工复现：
./build/probe_v_align run ls 4 0 64 /tmp/d     # 源侧 4B 非对齐 ⇒ 预期 507035
$PY check_ref.py .                              # 三态退出码
$PY check_ref_ildl.py .
$PY tools/check_archive_matches_list.py .        # 归档↔当前代码守卫（含窗口自洽）
$PY tools/diff_archives.py . <old_ref> --path=probe_v_align/evidence   # 归档差异记账（0 项时 rc=2 EMPTY）
$PY tools/diff_archives.py . <ref> --selftest                          # 注入矩阵自检
```

## 2. 方法（这部分比结论重要：reviewer 应该先看它）

### 2.1 一个变体 = 一次 launch + 一个进程
对齐违规是**设备侧异常**（`aclError=507035`）。一次 launch 里串多个 API 就无法把故障归因到某个 API，
也分不清"前一个 API 静默写坏、后一个才报错"。因此**每变体独占一次 launch**；
又因为故障/挂死可能污染上下文，**每变体独占一个进程** ⇒ 每条变体都有自己干净的 `aclError`。

### 2.2 被探地址不是编译期常量
被探地址 = `GetPhyAddr(arena) + 偏移`，**偏移是运行时 kernel 实参**（host 从 argv 传），
编译器无法常量折叠 ⇒ 探到的是硬件对**运行期地址**的行为，而不是编译器对常量地址的某种优化结果。
（`arena` 的物理基址是 32B 对齐的——这本身也被实测：偏移 0 与 32 都通过、1..24 全故障，见 §3。）

### 2.3 逐字节模型（档位声明：docs/17 §1.1 **T1**）
**档位**：本 mission 的判据对象是**字节/位模式**与**地址合规性**（什么偏移能过、能过时写出的字节是否逐位等于模型），
不涉及浮点舍入 ⇒ 按 docs/17 §1.1 走 **T1（整数域/位域逐位，无容差）**，**不启用任何容差**。
唯一例外：`Exp` 的输出含超越函数近似（属 T3 域），本探针**只对它做结构性非空洞判定**（窗口内确有内容、
窗口外未被写），**不判其数值**——这条 caveat 写在 §9.3。

**运行时文案按模型分档（`d350f08` 的订正，见 §9.1）**：每条变体打印的 `RESULT:` 行现在与它**实际做的检查**一致——
* 字节模型（`M_BYTESHIFT`/`M_CONSTF32`/`M_BRC`/`M_ADD`/`M_MUL`/`M_ARANGE_F32`/`M_F32BF16_PACK`/`M_F32BF16_NORM`/`M_BF16F32_NORM`/`M_BF16F32_UNPK`/`M_CMPSEL`/`M_RED`）→ `RESULT: OK (bitwise model match over 1024 bytes; …)`；
* 部分位判（`M_F32BF16_NORMB16`：只判偶数 16-bit lane）→ `RESULT: OK (partial bitwise: …)`；
* 结构性（`M_NONEMPTY`/`M_EXP`）→ `RESULT: OK (structural only: model=…; window NB non-empty, N non-sentinel bytes, M bytes written outside window)`；
* 控制组（`M_NONE`）→ `RESULT: OK (no model: control variant …)`。

（在此之前（`b9d468f`）所有 `cmp==0` 的变体统一打 "bitwise model match"，对结构性模型是**夸大检查范围**；
`M_RED` 也已从结构性升为真字节模型——`Reduce<SUM|MAX>` 的 lane0 值是精确可算的 2048 与 63.5。）

* `IN arena` 填 `f(i) = i + 0.5` 的字节模式（可 host 侧复算）；
* `OUT arena` 先用 **0xA5 sentinel** 填满（GM→UB 的纯搬运），op 只写它该写的窗口，最后**整段 1024B** 回 GM；
* 于是"写没写 / 写到哪 / 写超了多少"全在 dump 里可见，host 与 `check_ref.py` **各自独立**逐字节比对
  （整数/位域逐位，无容差）；**窗口外被写的字节数**单列报告项（掩码 store 是否静默写整 32B 块，就看它）。

### 2.4 自适应降序扫描 + 它的分类假设（为什么可以把故障次数从 5 降到 1）
对齐要求必是 32 的 2 幂因子（1/2/4/8/16/32）。于是对每个 API 按 **降序 {0, 16, 8, 4, 2, 1}** 试，
**首个故障点**就唯一确定最小对齐：16 故障 ⇒ 32B（8 的倍数都会通过 16，故 16 故障排除 1..16）；
16 通过、8 故障 ⇒ 16B；8 通过、4 故障 ⇒ 8B；…；1 也通过 ⇒ 1B。
`ad:` 分组在首个故障处停止；**`LoadAlign`/`StoreAlign` 的源与目的两侧、外加 `vld` 与 `gather` 做完整扫描
{0,1,2,4,8,16,24,32}**（`full:` 分组，不提前停）——这就是上述分类假设的**完整性证据**。

### 2.5 共享设备上的"无输出/同步超时"绝不写成对齐结论
本机 NPU 同时被多个 agent 的 kernel（层 kernel、别的探针）占用，**排队延迟可达分钟级**。因此：
* 探针自身的同步窗口 `SYNC_TIMEOUT_MS = 120s`；到期报 `507046` ⇒ 结论记 **`NOSYNC`**（host 侧窗口到期，
  在共享设备上可能只是排队，**不构成设备侧结论**）；设备侧真挂死以 `507034`/`107020` 报出 ⇒ 记 **`HANG`**；
* `run_probes.sh` 对 `NOSYNC`/无输出**重试 3 次**，仍无输出记 `NO-OUTPUT`；
* **`NOSYNC`/`NO-OUTPUT` 既不参与对齐推断、也不计入故障计数**（绝不能让"排队超时"变成"对齐要求"）；
* `ad:` 分组遇到 `NOSYNC`/`NO-OUTPUT` **不停组**（只有真故障才停）。

### 2.6 归档必须能证明"由当前代码产出"
`tools/check_archive_matches_list.py` 把这条变成一条命令（三态退出码）：
① `matrix.txt` 的 `(op,a,b,n,group)` 序列与当前 `./build/probe_v_align list` **逐项同序**；
② 每条执行过的变体，其 log 里的 `aclError`/`DUMP_FNV` 与矩阵对应列一致；
③ §8 的 guard②（每个 op 的偏移 0 对照点必须 OK），两个登记例外（`gatherb`/`lmbarout`，见 §9.2）显式列出。
负向对照 `--selftest`（顺序扰动 + 抹掉 1 条）必须被检出。
**为什么需要**：`FAULT_MSG` 里的 `hash=` **不能**当源码指纹（同一次归档里不同 kernel 的 hash 就不同），
所以"归档 = 当前代码"只能靠**结构性比对**（行序 + 列值）来证明 —— `b9d468f` 评审的 P2-1 正是发现了归档与当时代码
在**行序**与 `pld` 一行上不一致。本 commit 的归档起由**一次完整运行**产出，该守卫报 OK。

### 2.7 同步批处理形状（方法学对照，非平台断言）
V 段**不跨计算持有** OUT 的 BufferID（形状逐字照抄 `probe_sync_quirks/probe_b_vec_idx.asc`）。
实测对照：其余代码逐字不变、地址全对齐，仅把 `Acquire(PIPE_V,BUF_OUT)` 移到计算之前与
`Acquire(PIPE_V,BUF_IN)` 成对持有 ⇒ 单 AIV、单次 launch 下**挂死**（两次）；改成该形状后同一段计算立刻 PASS。
**一次对照、两次观测 ⇒ 只作方法学记录**，不声称它是平台通则。

## 3. [实测] API × 最小对齐 × 触发条件 × 错误码（总表）

> 本表由 `tools/make_tables.py` 从 `evidence/logs/matrix.txt` **机器生成**（`evidence/alignment_table.md` 为同一内容），
> 汇总规则写在脚本里、可逐项复算。**「判定方式」列是 `b9d468f` 评审的 P2-3 要求**：把
> **完整扫描**（偏移逐点实测）与**降序推断**（2 点 + 分类假设）**逐行分开**——
> 引用 B 类（降序推断）结论时必须保留该标注。R1 之后 `vld` 与 `gather` 已升为完整扫描。
> 本表的**落盘窗口**列来自实测（`stpack4` 在 `6256700` 里由 32B 订正为 64B）。

| API（本探针 op） | API 形态 / 底层指令 | 落盘窗口 | **实测最小 UB 偏移对齐** | 判定方式（P2-3 要求逐行可辨） | 触发条件（实测） | 错误码 | 依据日志（evidence/logs/） |
|---|---|---|---|---|---|---|---|
| `ls` | LoadAlign→StoreAlign（vlds/vsts） | 256B（1 个 VL） | **源 32B / 目的 32B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 16B ⇒ FAULT(507035); 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_ls_a*_b*_n*.log`（逐点，共 16 条） |
| `vld` | LoadAlign(reg,addr,AddrReg)（vld） | 256B | **源 32B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_vld_a*_b*_n*.log`（逐点，共 8 条） |
| `gather` | Gather（vgather2，UB base + 索引寄存器） | 256B | **源 4B** | **完整扫描 {0,1,2,4,8,16,24,32}** | 源: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_gather_a*_b*_n*.log`（逐点，共 8 条） |
| `ldbrc` | LoadAlign<…,DIST_BRC_B32>（单元素广播 load） | 256B | **源 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_ldbrc_a*_b*_n*.log`（逐点，共 5 条） |
| `stfirst` | StoreAlign<…,DIST_FIRST_ELEMENT_B32>（单元素 store） | 4B | **目的 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_stfirst_a*_b*_n*.log`（逐点，共 5 条） |
| `stpack` | Cast<bf16,float>+StoreAlign<…,DIST_PACK_B32>（pack b16） | 128B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_stpack_a*_b*_n*.log`（逐点，共 2 条） |
| `stpack4` | Cast<fp4>+StoreAlign<…,DIST_PACK4_B32> | 64B（128 个 fp4 nibble 打包；P2-4 由实测订正，原写 32B） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_stpack4_a*_b*_n*.log`（逐点，共 2 条） |
| `vldas` | LoadUnAlignPre+LoadUnAlign（vldas/vldus） | 256B | **源 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_vldas_a*_b*_n*.log`（逐点，共 6 条） |
| `vstus` | StoreUnAlign(+Post)（vstus/vstas） | 256B | **目的 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_vstus_a*_b*_n*.log`（逐点，共 6 条） |
| `load1` | Reg::Load（vldas+vldus） | 256B | **源 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_load1_a*_b*_n*.log`（逐点，共 6 条） |
| `store1` | Reg::Store（vstus+vstas） | 256B | **目的 1B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 无（1B 也接受） | -（无不合规路径：1B 也接受） | `run_store1_a*_b*_n*.log`（逐点，共 6 条） |
| `gatherb` | GatherB（vgatherb，位索引） | 未取到（见备注） | **不作结论**（偏移 0 即故障，非对齐问题；见备注） | 降序推断 {0,16,8,4,2,1}（首个故障停） | 偏移 0 ⇒ FAULT(507035) | 507035 | `run_gatherb_a0_b0_n64.log` |
| `scatter` | Scatter（vscatter） | 256B | **目的 4B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 2B ⇒ FAULT(507035) | 507035 | `run_scatter_a*_b*_n*.log`（逐点，共 5 条） |
| `pld` | LoadAlign(MaskReg,…)/StoreAlign(…,MaskReg)（plds/psts） | 32B | **源 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 源: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_pld_a*_b*_n*.log`（逐点，共 2 条） |
| `dupub` | 经典 Duplicate(UB 目的)（507015 历史现场） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_dupub_a*_b*_n*.log`（逐点，共 2 条） |
| `brcb` | 经典 Brcb(UB 目的) | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_brcb_a*_b*_n*.log`（逐点，共 2 条） |
| `dupreg` | Reg::Duplicate（寄存器目的）+ StoreAlign | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_dupreg_a*_b*_n*.log`（逐点，共 2 条） |
| `add` | Add（vadd） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_add_a*_b*_n*.log`（逐点，共 2 条） |
| `mul` | Mul（vmul） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_mul_a*_b*_n*.log`（逐点，共 2 条） |
| `exp` | Exp（vexp） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_exp_a*_b*_n*.log`（逐点，共 2 条） |
| `redsum` | Reduce<SUM>（vcadd） | 4B（lane0） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_redsum_a*_b*_n*.log`（逐点，共 2 条） |
| `redmax` | Reduce<MAX>（vcmax） | 4B（lane0） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_redmax_a*_b*_n*.log`（逐点，共 2 条） |
| `cmpsel` | Compares+Select（vcmps/vsel） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_cmpsel_a*_b*_n*.log`（逐点，共 2 条） |
| `arange` | Arange<float>（vci） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_arange_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16norm` | Cast<bf16,float> + NORM StoreAlign | 256B（值在偶数 16-bit lane） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16norm_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16nb16` | 同上但落盘用 bf16 ALL 掩码 | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16nb16_a*_b*_n*.log`（逐点，共 2 条） |
| `f32bf16pack` | Cast<bf16,float> + DIST_PACK_B32 | 128B（紧凑 64 bf16） | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_f32bf16pack_a*_b*_n*.log`（逐点，共 2 条） |
| `bf16f32norm` | LoadAlign<bf16,NORM>+Cast<float> | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_bf16f32norm_a*_b*_n*.log`（逐点，共 2 条） |
| `bf16f32unpk` | LoadAlign<bf16,UNPACK_B16>+Cast<float> | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_bf16f32unpk_a*_b*_n*.log`（逐点，共 2 条） |
| `lmbarin` | LocalMemBar 在 __VEC_SCOPE__ 内（控制组） | 256B | **目的 32B** | 降序推断 {0,16,8,4,2,1}（首个故障停） | 目的: 偏移 16B ⇒ FAULT(507035) | 507035 | `run_lmbarin_a*_b*_n*.log`（逐点，共 2 条） |
| `lmbarout` | LocalMemBar 在 __VEC_SCOPE__ 外（靶子） | 256B | **不作结论**（偏移 0 即故障，非对齐问题；见备注） | 降序推断 {0,16,8,4,2,1}（首个故障停） | 偏移 0 ⇒ FAULT(507035) | 507035 | `run_lmbarout_a0_b0_n64.log` |
| `widthf32` | Duplicate(7.0f)+StoreAlign，dtype=f32 | 256B = 64 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widthf32_a0_b0_n*.log` |
| `widthbf16` | 同上，dtype=bf16 | 256B = 128 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widthbf16_a0_b0_n*.log` |
| `widths16` | 同上，dtype=int16 | 256B = 128 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widths16_a0_b0_n*.log` |
| `widths8` | 同上，dtype=int8 | 256B = 256 元素 | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_widths8_a0_b0_n*.log` |
| `aranges32` | Arange<int32> | 256B = 64 lane | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges32_a0_b0_n*.log` |
| `aranges16` | Arange<int16> | 256B = 128 lane | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges16_a0_b0_n*.log` |
| `aranges8` | Arange<int8> | 256B = 256 lane（实测 lane i == i 全 256 个） | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_aranges8_a0_b0_n*.log` |
| `vnoop` | 控制组：V 段不做任何事 | - | **N/A（单点 32B 对齐，不扫偏移）** | 单点标定（32B 对齐） | - | -（无不合规路径：1B 也接受） | `run_vnoop_a0_b0_n*.log` |

### 负向对照定点与探测到的无效路径（`nc:` 分组；**不作对齐结论**，只作'对照是否活着'的证据）

| op | 观测 | 结论 | 日志 |
|---|---|---|---|
| `vldasnopre` | a=1 b=0 ⇒ WRONG(0) | 缺 init/post ⇒ **无故障但与逐字节模型不符**（负向对照生效）——有状态协议的 init/post 不是可选的 | `run_vldasnopre_a*_b*_n*.log` |
| `vstusnopost` | a=0 b=1 ⇒ WRONG(0) | 缺 init/post ⇒ **无故障但与逐字节模型不符**（负向对照生效）——有状态协议的 init/post 不是可选的 | `run_vstusnopost_a*_b*_n*.log` |
| `psts` | a=0 b=0 ⇒ OK(0); a=0 b=16 ⇒ FAULT(507035) | 谓词落盘（psts）**限 32B 对齐**（0 通过 / 16 故障），落盘 32B | `run_psts_a*_b*_n*.log` |

### 长度维度（`lsn`：地址固定 32B 对齐，掩码长度 n 变化）

| n（元素） | 偏移 b | 实测写出窗口 | 窗口外被写字节 | 结果 | 日志 |
|---|---|---|---|---|---|
| 1 | 0 | window=4 | outside=0 | OK | `run_lsn_a0_b0_n1.log` |
| 3 | 0 | window=12 | outside=0 | OK | `run_lsn_a0_b0_n3.log` |
| 7 | 0 | window=28 | outside=0 | OK | `run_lsn_a0_b0_n7.log` |
| 8 | 0 | window=32 | outside=0 | OK | `run_lsn_a0_b0_n8.log` |
| 63 | 0 | window=252 | outside=0 | OK | `run_lsn_a0_b0_n63.log` |
| 64 | 0 | window=256 | outside=0 | OK | `run_lsn_a0_b0_n64.log` |

## 4. [实测] 基准断言：VL 与 element 数（用户澄清 ②）

| dtype | 一个 VL 落盘字节数（`Duplicate(常量)`+全掩码 `StoreAlign`） | element 数 | 证据 |
|---|---|---|---|
| f32 | **256B** | 64 | `logs/run_widthf32_a0_b0_n64.log`、`dumps/va_widthf32_a0_b0_n64.bin` |
| bf16 | **256B** | 128 | `logs/run_widthbf16_a0_b0_n64.log` |
| int16 | **256B** | 128 | `logs/run_widths16_a0_b0_n64.log` |
| int8 | **256B** | 256 | `logs/run_widths8_a0_b0_n64.log` |

**落盘字节数逐条实测 = 恰好 256B（一个 VL），element 数 = 256 / sizeof(T)**；`dumps` 里 256 字节全被常量填满、
窗口外 0 字节（`outside=0`）。

`Arange` 的 lane 填充数（同一个道理，用 dump 逐 lane 读出来）：

| dtype | Arange 填充 lane 数 | 判据 | 证据 |
|---|---|---|---|
| float32 | 64（= 全寄存器） | `f32[i] == i`（i<64），第 64 个 lane 仍是 sentinel | `dumps/va_arange_a0_b0_n64.bin` |
| int32 | 64（= 全寄存器） | `u32[i] == i`（i<64） | `dumps/va_aranges32_a0_b0_n64.bin` |
| int16 | 128（= 全寄存器） | `u16[i] == i`（i<128） | `dumps/va_aranges16_a0_b0_n64.bin` |
| int8 | 256（= 全寄存器） | `u8[i] == i`（i<256，实测全 256 个成立） | `dumps/va_aranges8_a0_b0_n64.bin` |

⇒ 用户澄清 ②「SIMD API 固定 VL 256B，不同数据类型 element 数不同（F32/S32=64、S16=128）」
**实测成立**；而 `docs/05` §6.1 早期那句「Arange 只填 64 lane」**只在 32 位 dtype 上成立**
（那时 64 lane 就是全寄存器），作为通则不成立（docs/05 已于 2026-09-26 按同一口径修正）。

## 5. 正例 / 反例对照（每条"支持非对齐"与每条"限 32B"都必须有反例）

| # | 声明（本次要实测的） | **正例**（证据） | **反例**（证据） | 判定 |
|---|---|---|---|---|
| 1 | 多数 load/store API 限 32B 对齐 | `ls` a=0,b=0 ⇒ OK，dump 逐字节等于模型（`run_ls_a0_b0_n64`） | `ls` a∈{1,2,4,8,16,24} **逐点全部** FAULT(507035)；对照 a=32 ⇒ OK（完整 8 点扫描，源/目的两侧各一遍） | 成立（32B，且是"32 的倍数才算对齐"） |
| 2 | 单元素广播 load 支持非 32B | `ldbrc` a=16/8/4 ⇒ OK（a=16/8 非 32B） | `ldbrc` a=2 ⇒ FAULT(507035) | 成立，**最小对齐 = 4B**（元素粒度） |
| 3 | 单元素 store 支持非 32B | `stfirst` b=16/8/4 ⇒ OK，且**只写 4 字节**（实测写出区间 [b,b+4)，outside=0） | `stfirst` b=2 ⇒ FAULT(507035) | 成立，**最小对齐 = 4B** |
| 4 | unaligned load/store 支持非 32B | `vldas` a=1、`load1` a=1、`vstus` b=1、`store1` b=1 **全部** OK（1B = 最小可测非零偏移） | 无更小偏移可测；**故障路径的活性由同套件 45 条 FAULT 证明**（复算 `tail -1 evidence/logs/matrix.txt` → `… fault=45 …`）（`ls`/`ldbrc`/`gather` 都在相邻更小偏移上故障） | 成立，**1B 粒度（完全不对齐也接受）** |
| 5 | gather/scatter 支持非 32B | `gather` a=16/8/4 ⇒ OK；`scatter` b=16/8/4 ⇒ OK | `gather` a=2、`scatter` b=2 ⇒ FAULT(507035) | 成立，**最小对齐 = 4B** |
| 6 | 寄存器算子（`Duplicate`/`Add`/`Mul`/`Exp`/`Reduce`/`Compares+Select`/`Arange`/`Cast` 各变体/`LocalMemBar`）**自身没有 UB 地址要求** | 每个算子 b=0 ⇒ OK；其中由**逐字节模型**判定的是 **11 条**（`dupreg`/`add`/`mul`/`redsum`/`redmax`/`cmpsel`/`arange`/`f32bf16norm`/`f32bf16pack`/`bf16f32norm`/`bf16f32unpk`；复算 `grep -E '^OK +op=(dupreg\|add\|mul\|redsum\|redmax\|cmpsel\|arange\|f32bf16norm\|f32bf16pack\|bf16f32norm\|bf16f32unpk) ' evidence/logs/matrix.txt \| wc -l` → `11`；`exp`/`f32bf16nb16`/`lmbar*` 走结构性判定） | **同一算子、只把落盘地址挪 16B** ⇒ 全部 FAULT(507035) | 成立：要求属于**相邻 store**，不属于算子 |
| 7 | `LocalMemBar` 必须在 `__VEC_SCOPE__` 内 | `lmbarin`（VF 内）b=0 ⇒ OK | `lmbarout`（VF 外）b=0（32B 对齐！）⇒ FAULT(507035)，设备侧 errcode=**259 "The scalar instruction is abnormal"** | 成立；且**细化**：与对齐故障同码不同因。`lmbarout` 是**靶子**（偏移 0 也故障），故在 §2.6 的守卫里列为**登记例外** |
| 8 | 掩码 store 不写掩码外 | `lsn` n=1/3/7/8/63/64：实测写出区间**恰好** 4n 字节（n=1 ⇒ [0,3]），`outside=0` | 无可构造反例（否命题）——但**同族负向对照**存在：`vldasnopre`/`vstusnopost` 证明"少一步有状态协议"能被逐字节抓到 | 成立：**不做 32B 块对齐填充** |
| 9 | `Interleave`/`DeInterleave` 配对且互逆 | `inv_f32`：`DeInterleave(Interleave(a,b))` 逐 lane == `(a,b)` | `intlv_f32_neg`：二次 `Interleave` 无任何候选映射命中（不是逆操作） | 成立（详见 §6） |
| 10 | 有状态 unaligned 族**必须**按 init/post 协议用 | `vldas`/`vstus` 完整形态：1B 偏移 OK 且逐字节等于模型 | `vldasnopre`（只用 vldus 不做 init）、`vstusnopost`（只发 vstus 不补 post）⇒ **无故障但与模型不符** | 成立：init/post 不是可选的 |

## 6. [实测] `Interleave` / `DeInterleave` 精确语义（M35 交接悬案）

`Reg::Interleave/DeInterleave` 的**声明里没有任何 lane 映射说明**（只有一句 doxygen
"Interleave src0 and src1 to dst0 and dst1"），`vintlv`/`vdintlv` 是编译器内建、源码树里查不到映射，
`sizeLine` 这个符号在整个 CANN 树 0 命中。所以下面是**实测标定**（dump 里每个 lane 的值都能唯一反推来源，
见 `check_ref_ildl.py` 的解码）。

设 V = 一个寄存器的 lane 数（f32 为 64、b16 为 128）、h = V/2：

| API | 实测映射（逐 lane 解码，T1） | 证据 |
|---|---|---|
| `Interleave(d0,d1,s0,s1)` | `d0 = [s0[0],s1[0],s0[1],s1[1],…,s0[h-1],s1[h-1]]`；`d1 = [s0[h],s1[h],…,s0[V-1],s1[V-1]]` | `intlv_f32`（f32, h=32）、`intlv_b16`（b16, h=64）——两种位宽**同一形态**，d1 用的是后半段 |
| `DeInterleave(d0,d1,s0,s1)` | `d0 = [s0 的偶 lane（h 个）, s1 的偶 lane（h 个）]`；`d1 = [s0 的奇 lane, s1 的奇 lane]` | `dintlv_f32`、`dintlv_b16` |
| 互逆性（正例） | `DeInterleave(Interleave(s0,s1)) == (s0,s1)` **逐 lane 相等** | `inv_f32` ⇒ 命中 `IDENTITY(d0=s0,d1=s1)` |
| 非逆操作（反例） | 把 `Interleave` 的输出再 `Interleave` 一次 ⇒ **不是** `(s0,s1)`（实测得到一个 4 路交织的排列），且**不命中**任何候选映射 | `intlv_f32_neg` |
| 仓库惯用法 `DeInterleave(d0,d1,s,s)`（m3:281 / m22:391 / m19:114 同形） | `d0 = s 的偶 lane`、`d1 = s 的奇 lane`（两个半区各重复一次） | `same_f32` ⇒ 命中 `SAME(=src0==src1)` |

三条可判定的结论（回答 M35 的悬案）：
1. **元素级，不是 block 级**：映射按 lane（dtype 元素）定义，f32（64 lane）与 b16（128 lane）**同一份代码、
   同一形态**，只是 lane 数按位宽变 —— 没有任何 32B 分块语义。
2. **无源/目标对齐要求（结构性 N/A）**：`Reg::Interleave/DeInterleave` 的操作数**全是寄存器**，
   签名里没有 UB 地址、没有 mask、没有 count/sizeLine。实测对照 `intlv_misalign`：**算子逐字不变、
   只把落盘地址挪 4B** ⇒ FAULT(507035) ⇒ 对齐要求属于相邻 store（与 §5 #6 同一判据）。
3. **配对且逐位互逆**（正例），而"对输出再 Interleave"不还原（反例）——所以"用 Interleave 当逆操作"
   这类写法**可判定为错**。

与位宽转换的关系（用户澄清 ⑤，本次实测的落位）：`Cast<bf16,float>` 的结果在 bf16 寄存器里是
**每 32-bit lane 一个值（落在偶数 16-bit lane）**的展开形态；要落成紧凑 bf16 序列有两条**都已实测**的路：
`StoreAlign<bfloat16_t, DIST_PACK_B32>`（128B 紧凑，见 §3 `stpack`/`f32bf16pack`）或
`Interleave` 之后再按 `DIST_INTLV_*` 落盘（m2/m5 的 fp4 路径用的就是后者，本次未单独复测该配对的
逐字节结果 —— 见 §9 未做项）。

## 7. [文档转述] 转述栏 + 给 docs 的建议文本（A/B/C）

### 7.1 转述栏（只放"文档/他人口述"的条目）

> 本节**只放"文档/他人口述"的条目**，**不得**与 §3 的实测混排。最后一列是"本次是否独立验证过"，
> 其中"已实测"的判据在 §3/§4/§5/§6，不在本节。

| # | 转述内容 | 出处 | 本次验证状态 |
|---|---|---|---|
| 1 | 「UB 偏移非 32B 对齐 → 507035」 | 用户 2026-09-26 | **已实测**：run 侧 **44** 条 errcode 340（`errcode:(340) errorStr: The address for VEC to access UB is not aligned`），加 ildl 侧 1 条共 45 条 340；复算 `grep -ohE 'errcode:\([0-9]+\)' evidence/logs/run_*.log \| sort \| uniq -c` → `44 errcode:(340)` + `1 errcode:(259)` |
| 2 | 「大部分 API 限 32B，但 gather/scatter、unaligned load/store、单元素 load/store 支持非 32B」 | docs/05 §6.1（记 M9/M16 实测 + 用户裁决） | **已实测**（§3 逐 API）；并把"支持非对齐"量化到**粒度**：gather/scatter/单元素 = **4B**，unaligned（vldas/vstus/Load/Store）= **1B** |
| 3 | 「SIMD API 固定 VL 256B，element 数按位宽：F32/S32=64、S16=128」 | 用户 2026-09-26；docs/05 §6.1 | **已实测**（§4，含 int8=256） |
| 4 | 「`Arange` 只填 64 lane」（旧表述） | docs/05 §6.1 早期版本 | **实测与之不符**：Arange 按位宽填满全寄存器（f32/s32=64、int16=128、int8=256）；该旧表述在 32 位 dtype 上恰好等于"全寄存器"，作通则错。docs/05 已按同一口径改 |
| 5 | 「load unalign 和 store unalign 都是有状态的操作，硬件有 U register 做拼接 buffer，还有额外的 init/post」 | 用户 2026-09-26 | **已实测**：完整形态在 1B 偏移上逐字节正确；缺 init/post 的负向对照**无故障但结果错** |
| 6 | 「LocalMemBar 须在 `__VEC_SCOPE__` 内」 | 用户 2026-09-26；docs/05 §6.2 记 507035 | **已实测并细化**：aclError 同为 507035，但设备侧 errcode 是 **259（scalar instruction abnormal）**，与对齐的 **340** 不同 |
| 7 | 「fp32→bf16 可用 store 的 pack b16 模式（需确认）」 | 用户 2026-09-26；M35 提到官方 `DIST_PACK_B32` 用法 | **已实测**（§3 stpack / f32bf16pack）：`DIST_PACK_B32` → 64 个**连续** bf16 = 128B；NORM 落盘 → "每 32-bit lane 一个 bf16"（值在偶数 16-bit lane）的展开布局 |
| 8 | 「bf16→fp32 需 even/odd 两条 cast；要还原原序还需 deinterleave」 | 用户 2026-09-26；docs/05 §6.1 | **落位已实测**：`LoadAlign<bf16,NORM>`+`Cast<float>` **只换偶数 16-bit lane**（`bf16f32norm`）；`LoadAlign<bf16,UNPACK_B16>`+`Cast<float>` 给出连续 64 元素（`bf16f32unpk`）。"是否必须两条 cast"取决于源在 UB 的布局——见 §9 未做项 |
| 9 | 「`Reg::Gather` 的 index 单位（元素还是字节）在所有头文件里都未写明」 | docs/18 §5.3 存疑 | **本次仍未标定**（本探针把 idx 当元素序号用、结果结构正确，但**不作单位断言**）⇒ 仍存疑 |
| 10 | 「`Sort32`/`MrgSort`/`Extract`/`Transpose`/`Concat` 无 Reg 版」（`Brcb` **不在** docs/18 的清单里，归属订正为"本次穷举"） | docs/18 §6.1 + **本次穷举**（与 `6256700` 的穷举复核一致：`cann-9.1.0/asc/include/{basic_api,c_api}/reg_compute/` 26 个 `.h`，`Brcb`/`Transpose`/`Concat`/`Sort32`/`MrgSort`/`Extract` 命中 **0**） | **本次复核穷举**：`interleave`/`deinterleave` 各 2 处、上述 6 个 0 命中。因此 `Brcb` 只能用经典 API 探测（§3 的 `brcb` 行即"例外说明"） |
| 11 | 「`StoreUnAlign(dst,reg,ureg,n)` 的 n 只是 dst 指针后更新步长，整寄存器写」 | docs/05 §6.2（记 M14 实测） | **部分实测**：`vstus`（含 `StoreUnAlignPost`）落盘 256B 整寄存器 ✓；`n` 的作用本次**未单独标定** |
| 12 | 「把 `Acquire(PIPE_V,BUF_OUT)` 与输入成对持有会挂死」 | 本探针 §2.6 的方法学对照 | **非平台断言**：一次对照、两次观测（见 §2.6） |

### 7.2 给 docs/05（§6.1/§6.2）与 docs/18 的**建议文本**（`6256700` 的 A/B/C 骨架，已并入该评审的四条订正）

> 用途：**落库时可直接照抄**。分层含义：**A = 可直接依赖**（实测、可复核）；**B = 引用时必须保留
> "降序推断"标注**（§3 的「判定方式」列逐行可辨）；**C = 必须带上 caveat**。
> 本节的每一条都能在 `probe_v_align/evidence/` 里找到逐变体日志与 dump；**不要**把 B 类当 A 类引用。

**A. 可以直接依赖（实测）**

* `LoadAlign`/`StoreAlign`（`vlds`/`vsts`）**源与目的都要求 32B**：偏移 1/2/4/8/16/24 逐点全部 `507035`
  （设备侧 `errcode 340`：*The address for VEC to access UB is not aligned*），0 与 32 通过 —— 两侧各做了**完整 8 点扫描**。
* `vld`（`LoadAlign(reg, addr, AddrReg)`）**仍是 32B** —— 它是唯一与直觉分组相反的一条（看着像 unaligned 的形态），
  因此**已升为完整 8 点扫描**（详见 §3 `vld` 行的「判定方式」列）。
* 支持非 32B 的族与**粒度**：`gather`/`scatter`/单元素 load（`ldbrc`，`DIST_BRC_B32`）/单元素 store（`stfirst`，
  `DIST_FIRST_ELEMENT_B32`）= **4B**（2B 即故障；`gather` 亦已升为完整扫描）；`unaligned` 族
  `LoadUnAlign(Pre)`/`StoreUnAlign(Post)`/`Reg::Load`/`Reg::Store` = **1B，完全不对齐也接受**。
* `unaligned` 族是**有状态**协议：缺 `init`/`post` **不报错但结果错**（逐字节对照证明，见 §5 #10）。
* 寄存器算子（`Duplicate`/`Add`/`Mul`/`Exp`/`Reduce`/`Compares+Select`/`Arange`/`Cast` 各变体/`LocalMemBar`）
  **自身没有 UB 地址操作数**；对齐要求属于相邻的 store —— 同一算子只把落盘地址挪 16B 即 `507035`。
* 掩码 store **不做 32B 块填充**：`lsn` n=1/3/7/8/63/64 写出**恰好 4n 字节**、窗口外 0 字节（n=1 只写 `[0,3]`）。
* 一个 VL = **256B**；element 数 = 256/sizeof(T)：f32/s32=64、bf16/int16=128、int8=256；
  `Arange` **按位宽填满全寄存器**（不是"只填 64 lane"）。
* `Interleave`/`DeInterleave` 是**元素级**（无 32B 分块语义），f32(64 lane) 与 b16(128 lane) 同形态；
  `DeInterleave(Interleave(a,b)) == (a,b)` 逐位成立；对 `Interleave` 输出**再** `Interleave` 不还原；
  `DeInterleave(d0,d1,s,s)` ⇒ d0 = 偶 lane、d1 = 奇 lane（各重复两个半区）。签名无 UB 地址/mask/count
  ⇒ 其对齐要求**结构性 N/A**（实测：只挪落盘地址 4B 即故障）。
* fp32→bf16 落盘：`Cast<bf16,float>` + `StoreAlign<bf16, DIST_PACK_B32>` = **64 个连续 bf16（128B）**；
  `NORM` 落盘 = 每 32-bit lane 一个 bf16（值在**偶数** 16-bit lane）。bf16→fp32：`NORM` load + `Cast`
  **只换偶数 16-bit lane**；`DIST_UNPACK_B16` + `Cast` 给连续 64 元素。
* `LocalMemBar` 在 VF 外：`aclError` 与对齐故障**同为 507035**，但设备侧 `errcode` 是 **259**
  （*The scalar instruction is abnormal*）vs 对齐的 **340** ⇒ 归因要看 `FAULT_MSG` 里的 errcode。

**B. 需标注「降序推断」才能依赖（结论与 A 同强度，证据是 2 点 `{0: OK, 16: FAULT}`）**

`add`、`mul`、`exp`、`redsum`、`redmax`、`cmpsel`、`arange`、`dupreg`、`dupub`、`brcb`、`stpack`、`stpack4`、
`f32bf16norm`、`f32bf16nb16`、`f32bf16pack`、`bf16f32norm`、`bf16f32unpk`、`lmbarin` 的 32B 结论由
`{offset 0 通过, offset 16 失败}` + "对齐要求 ∈ {1,2,4,8,16,32}" 推出（`vld`、`gather` 已按该评审建议补完整扫描，
故不在本栏）。另：`stpack4` 的**落盘窗口**已由实测订正为 **64B**（128 个 fp4 nibble 打包，见 §3 该行）。
**写 kernel 时按 32B 用是安全的**；若要写进规范正文，请保留"降序推断"标注，
或按同样方式补完整扫描（`bash run_probes.sh` 后看该行「判定方式」列）。

**C. 使用时的 caveat（写进规范时必须带上）**

* `Exp` **只判结构、不判数值**（超越函数属 docs/17 的 T3 域）；本探针对其数值无结论。
  其运行日志现在打的也是 `RESULT: OK (structural only: …)`（`d350f08` 之前会错打成 "bitwise model match"）。
* `Reg::Gather` 的 **index 单位（元素 vs 字节）仍未标定** —— 依赖 gather 的代码必须先做一次值级标定，
  不要照搬 donor 参数。
* `gatherb`：**偏移 0（32B 对齐）即失败**，用法/支持性存疑 ⇒ **无对齐结论**，不要当"支持 4B"用。
* `pld`/`psts`：谓词 load/store 的**掩码位布局未标定**；`pld` 的 load 侧 32B 是实测的，
  store 侧在 `b=0` 可观测（32B）但只作落盘观测，不当语义结论。
* `4B` 这个粒度是**用 f32 元素量出来的**；**不要外推到 b16/b8**（偏移粒度未逐 dtype 复测）。
* 未测：`Interleave` + `DIST_INTLV_*` 的端到端落盘字节序（m2/m5 的 fp4 打包链用这个配对）；
  "bf16→fp32 必须两条 cast"的场景；`StoreUnAlign(dst,reg,ureg,n)` 里 `n` 的单独作用。
* 本表只出**语义/对齐**结论，**不含性能数字**（host 墙钟在共享卡上不构成证据）。

## 8. 判定项 / 报告项 / guard 分栏与计数

**判定项 / 报告项 / guard 分栏**（docs/17 §2.1 要求；PASS/FAIL 计数里只含判定项）：

| 类 | 内容 | 计数 | 复算命令 |
|---|---|---|---|
| 判定项（主探针，矩阵层） | 实际执行的变体 | **129**（`matrix.txt` 里非 STOPPED 行） | `grep -c 'op=' evidence/logs/matrix.txt` 减 STOPPED |
| 判定项（主探针，独立复核） | 逐字节 T1 比对 **57** 条（55 普通 + 2 负向对照）+ 结构性比对 **27** 条 | **84** 行，与模型不符 **0** | `$PY check_ref.py .`（`PY=/usr/local/python3.12.13/bin/python3`） |
| 判定项（ildl） | 7 条（含 1 条反例）+ 1 条预期故障 | 7/7 通过 | `$PY check_ref_ildl.py .` |
| **归档一致性守卫**（`b9d468f` 评审 P2-1 的自动化形式） | `matrix.txt` 与当前 `list` 逐项同序 + 每条 log 的 aclError/FNV 与矩阵列一致 + guard② + **窗口自洽（`VARIANT window` == `COMPARE window_bytes`）** | 218 行 / 129 条 log / 31 个 op / 84 条窗口，**零差异** | `$PY tools/check_archive_matches_list.py .`（负向对照 `--selftest`） |
| 负向对照 | `vldasnopre`/`vstusnopost`（期望 WRONG） | **2**，均生效 | 同上 |
| 故障分类 | 45 条 507035：errcode **340**（UB 未对齐）×44 + errcode **259**（标量指令异常）×1 | 45 | `grep -ohE "errcode:\([0-9]+\)" evidence/logs/run_*.log \| sort \| uniq -c` |
| 主动跳过（**非判定项**） | `ad:` 分组按降序策略在首个故障处停 | 89 | `grep -c ^STOPPED evidence/logs/matrix.txt` |
| 无数据（**非判定项**） | `b9d468f` 归档里 `pld` 落盘侧 0 字节**已随归档刷新消失**（现为可观测的 32B）；`gatherb` 属"偏移 0 即故障"，走故障分类那一栏 | **0** | `check_ref.py` 的"无数据"计数 |
| 报告项（不参与 PASS/FAIL） | 窗口外被写字节数、实测写出区间、dump FNV、`HEAD32` | 逐变体一行 | `matrix.txt` 的 `outside=`/`fnv=` 列 |
| guard 非空洞性 | ① `vnoop` 控制组：window=0、dump 全 sentinel；② 每个 op 的 32B 对齐点必须 OK（否则该 op 无法解读）——**已工具化**进 `tools/check_archive_matches_list.py`，并显式登记两个例外（`gatherb`/`lmbarout`，见 §9.2）；③ `lsn` 的 n 变化必须改变 dump（n=1/3/7/8/63/64 六个 FNV 互不相同）；④ 同一变体重复执行 FNV 一致：`ls 0 0 64` 在两个分组里各跑一次，FNV 均为 `0xc94a34646b376c33` | 4 类 | 见 `matrix.txt` 与 `evidence/check_archive_matches_list.log` |

**计数口径（防止"包含关系"误读，docs/17 §2.2）**：`check_ref.py` 的 `n_compared`（逐字节模型行数）
与 `n_struct`（结构性行数）**不相交**，二者之和 = 非故障行数（84 = 129 执行 − 45 故障）；
"通过行数"= 逐字节通过 + 结构性通过 + 负向对照按"期望不符"计通过。第一版打印把"通过行数"与
"结构性行数"并列，看起来像 104 行判定，属口径错，已改为现在的写法。

**"与模型不符 0"是怎么做到的**（不空洞性）：`check_ref.py` 从 dump **独立**用 numpy 重算期望值
（不复用 C++ 侧模型）。第一版核对器里我自己写的"把整型位模式直接赋给 float 数组"是错的，
于是 `bf16f32unpk` 报了 1 处不符；查下来是**核对器的错**（C++ 模型用 `memcpy` 构造位模式是对的），
改对后 0 处不符 —— 这正是"独立实现"该有的作用（详见 §9.1）。

## 9. 已知局限 / 未做 / 仍存疑

### 9.1 探针自身的三个缺陷 + 一次"归档跨修订拼装"（修复见 `d350f08` + `6256700`）

**0. 归档跨修订拼装（`b9d468f` 评审的 P2-1，最高优先级，已修）**：该评审指出**归档不能由当时的提交代码复现**——
归档里 `pld a=0 b=0` 仍是修 `brcb` 时那版（掩码源取 OUT arena）的旧结果（`WRONG`/0 字节），
而当前代码取 `IN+a`（写 32B ramp，`OK`）；且归档的 `matrix.txt` 行序是"旧运行 + 手工追加"，
当前 `PrintList` 产不出（`brcb` 块位置不同）。⇒ 本 commit 的归档由**一次完整运行**重新产出，
`tools/check_archive_matches_list.py` 行序/列值/guard② 全绿；§9.2 与 §10 的措辞同步订正。
（历史不被抹掉：这段拼装史就记在这里，且守卫会把再次发生的情况报出来。）

**1. 分派 bug**：`Run()` 初版按 op id 大小分段（`op_ >= OP_DUPREG` → RunArith），新增的
   `psts`(42)/`vldasnopre`(40)/`vstusnopost`(41) 掉进了 RunArith 而**静默 no-op** ⇒ 三个变体当时都是
   "无故障、0 字节落盘"。**这个失败模式正是本探针要抓的第二类风险**（"没跑"和"跑对了"在表面上一模一样），
   所以修复后重跑并把旧行从归档里删除。已改为**显式列举**分组。

**2. `brcb` 源与目的同址**：初版把源也取在 OUT arena，a==b==0 时 `Brcb` 内部
   `ASCENDC_DEBUG_ASSERT(src0.GetPhyAddr() != dst.GetPhyAddr())` 判定失败 ⇒ **无故障、0 字节落盘**。
   改为源取 IN arena 后 OK（落 256B），并测出其 32B 对齐要求。

**3. 核对器的整型↔浮点位模式 bug**（见 §8 末）——由"独立复核"暴露。

### 9.2 无数据 / 仍存疑

| 项 | 观测 | 为什么不作结论 | 建议后续 |
|---|---|---|---|
| `pld`（谓词 load→store 往返） | **load 侧**：地址挪 16B ⇒ FAULT(507035)（32B 要求，实测）；**store 侧**：当前代码在 `b=0` 写 32B 且逐字节等于模型（`OK`） | **已修复 `b9d468f` 归档的陈旧问题**：旧归档那行是掩码源还取在 OUT arena（sentinel）时的结果（活跃位太少 ⇒ 0 字节）；当前代码取 `IN+a`（§2.6 的守卫会把这类不一致报出来）。谓词 load/store 的**完整语义**（掩码位布局）仍未标定 | 若要当规范正文用，建议对谓词 load/store 另做一次掩码位布局标定 |
| `gatherb`（vgatherb） | **偏移 0（32B 对齐）即 FAULT(507035)** | 故障与非对齐无关 ⇒ 我的用法（索引来自 `Arange<int32>`）或该形态的支持性存疑；不作对齐结论 | 按"位索引"语义重新构造索引再标定 |
| `Reg::Gather` 的 index 单位 | 本探针把 idx 当**元素序号**用，结果结构正确（窗口有内容、无故障） | 未做单位判别实验（元素 vs 字节需要一次可分辨的标定） | 用 idx 值 0/1/2 对 f32 做一次"值级"判别 |
| "最小对齐"的推导方式 | 除主族 `ls`（源/目的各 8 点完整扫描）外，其余 op 用**降序 {0,16,8,4,2,1} + 首个故障停** | 依赖"对齐要求必是 32 的 2 幂因子"这一分类假设；主族的完整扫描是该假设的**完整性证据**（1/2/4/8/16/24 逐点全故障） | 若要更强，可对争议 op 补完整扫描 |

### 9.3 未做（附取舍理由）

> **`Exp` 的数值判据**：本次只判结构（非空洞），不判数值。理由：`Exp` 的核心结论是"它自身没有 UB 地址
> 操作数、对齐要求属于相邻 store"，其数值正确性属 T3 域且与 M57 的问题无关（本仓 m5/m13 的 `Exp` 用法
> 已有各自的验收档位）。若 reviewer 要求补齐数值判据，需按 docs/17 §1.1 的 T3 流程推导界。


| 未做项 | 取舍理由 |
|---|---|
| `Transpose` / `Concat` / `Sort32` / `MrgSort` / `Extract` | **寄存器侧无这些 API**（本次穷举复核：`reg_compute/` + `c_api/reg_compute/` 下 0 命中）。经典的 `Transpose`/`Concat` 存在，但不在"实际在役的 VF API"清单内，且本仓 0 使用（`docs/18` §6.1 同结论） |
| GM 侧 / L1 / L0 侧对齐 | 本 mission 的口径是 **UB 偏移对齐**；搬运类（`DataCopy` 等）属另一族（docs/18 §6.3 的 (c) 类） |
| b16/b8 的**偏移粒度**逐 dtype 复测 | 本次访存偏移扫描只用 f32（元素 4B）；b16/b8 未必同粒度（如 `ldbrc` 的 4B 是 f32 元素宽度）。**不要**把 4B 外推到 b16 |
| 掩码长度 n 与 dtype 的交叉 | `lsn` 只测了 f32；b16 掩码按 16-bit lane 计，长度语义可能不同 |
| `Interleave` + `DIST_INTLV_*` 落盘配对的逐字节复测 | m2/m5 的 fp4 打包链用的是这个配对；本次只测了算子本身的映射与 `DIST_PACK_B32` 的落盘，**没有**复测"Interleave 之后按 INTLV dist 落盘"的端到端字节序（M35 若要动那条链，需先补这一测） |
| `Cast` 的"两条 cast（even/odd）"完整通路 | 本次测了 `Cast` 的**落位**（NORM 只换偶数 lane / UNPACK 连续）与 `DIST_PACK_B32` 的紧凑落盘；"必须两条 cast"取决于源布局，未构造该场景 |
| 共享设备上的时序/性能 | 本探针只出**语义/对齐**结论，不出性能数字（host 墙钟在共享卡上只作定性，docs/17 §8.4 #8） |

## 10. 证据索引

| 路径 | 内容 | 规模 |
|---|---|---|
| `evidence/logs/run_<op>_a<a>_b<b>_n<n>.log` | 逐变体完整 stdout+stderr（`VARIANT`/`LAUNCH aclError`/`FAULT_MSG` 原文/`COMPARE`/`DUMP_FNV`/`OUTCOME`/`RESULT`） | **128 个**（= 129 条执行行：`ls 0 0 64` 同属两个分组、共用一个文件） |
| `evidence/logs/matrix.txt` | 逐变体一行的机器可读矩阵（含 89 条 STOPPED 与末尾计数块） | **218 行 + 2 行**（计数块） |
| `evidence/dumps/va_*.bin` | 每条成功变体的 **1024B OUT arena 原始 dump**（逐字节证据；sentinel=0xA5） | **83 个**（+ ildl 的 7 个 `il_*.bin` = 90 个 dump） |
| `evidence/logs/ildl_*.log` / `evidence/dumps/il_*.bin` | ildl 8 个用例的日志与 dump | 8 + 7 |
| `evidence/check_ref.log` | 主探针独立复核输出（三态退出码 0） | — |
| `evidence/check_ref_ildl.log` | ildl 独立复核输出（含逐 lane 来源解码） | — |
| `evidence/alignment_table.md` | §3 的机器生成版（`tools/make_tables.py` 产出，README 内嵌同一内容） | — |
| `evidence/commands.txt` | 本次运行的确切环境/版本/复现命令（含采集时设备上的其它进程快照） | — |
| `evidence/check_archive_matches_list.log` | 归档↔当前代码一致性守卫的输出（行序 / log 列 / guard② / 窗口自洽） | — |

**归档口径声明**（必须说明，否则"复现"会被误读）：

* 本归档**由一次完整运行产出**（当前 commit 的代码；`bash run_probes.sh` 一条命令），不再有跨修订拼装的行。
  守卫 `tools/check_archive_matches_list.py` 能证明的是**内部一致性**（行序与当前 `list` 同序、每条 log 的列值
  与矩阵一致、窗口自洽、guard②）—— 那是很强的代理，但**不是 provenance 证明**；provenance 另有两条旁证：
  ① 故障 log 里的**设备序号沿矩阵顺序严格递增且连续**（跨批拼装会留下序号回退或空洞）——
  守卫的输出里有 `[R]` 行给出该读数（`$PY tools/check_archive_matches_list.py . | grep '^\[R\]'`），
  本 commit 的读数是"45 条故障行带序号，区间 888..932，严格递增=True，连续=True"；
  ② 同一变体重复执行的 FNV 一致（§8 guard④）。守卫核对的是：
  `matrix.txt` 的 `(op,a,b,n,group)` 序列与当前 `list` 逐项同序、每条 log 的 `aclError`/`DUMP_FNV`
  与矩阵列一致、guard②（偏移 0 对照点 OK，两个登记例外）成立 ⇒ 该守卫报 `RESULT: OK`。
* **`b9d468f` 归档的历史**（据实保留）：那次归档是"一次较早的完整运行 + 修缺陷后手工追加受影响的变体"，
  于是出现了两处不一致（`pld` 一行陈旧、`brcb` 块的行序不是当前 `PrintList` 能产出的），
  而 §10 当时声称"由当前代码产生"—— 这就是 `b9d468f` 评审 P2-1 的来源，已按上面的方式修复。
  这条历史与守卫一起保留，是为了让"再次发生"能被立刻看见。
* **逐变体 FNV 逐字节可复现**（同机同版本；共享设备只影响墙钟、不影响数值）——
  同一变体在矩阵里出现两次（`ls 0 0 64` 同属 `full:ls:1` 与 `full:ls:2` 两个分组）时 FNV 相同，
  这一点被 §8 的 guard④ 用到。

**已知不完整**（与 §9.2 一致，逐条可核）：`gatherb` 在偏移 0 即故障、无对齐结论；
`pld`/`psts` 的**掩码位布局**未标定（`pld` 的 load 侧 32B 是实测的、store 侧在 `b=0` 可观测 32B）；
`Reg::Gather` 的索引单位未标定；最小对齐的推导口径见 §9.2（`ls`/`vld`/`gather` 完整扫描、
其余按分类假设 + 降序停组）；4B 粒度由 f32 量出、不向 b16/b8 外推。
