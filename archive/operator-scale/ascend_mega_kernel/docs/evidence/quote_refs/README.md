# M129 证据：`docs/scan_quote_refs.py`（docs 引文保真扫描器）与 `.tower/` 引用清单

**基准**：本分支 base = `53c2107e9b0e680a49dba51ffd60d6b1243c460d`（`main` 的 merge M105），worktree `.tower/worktrees/wt-129`。**零设备**：全部读数只用到 `python3` / `git` / `bash`。
**复跑**：`bash docs/evidence/quote_refs/repro.sh`（在仓根；一键重建本目录的 `scan.log` / `scan_pairs.log` / `selftest.log` / `tower_refs.md`）。

本目录有**两份报告**：
- **(a) 全 `docs/` 扫描报告** = 本文件 + `scan.log`（汇总）+ `scan_pairs.log`（逐对明细）。
- **(b) `.tower/` 引用清单** = `tower_refs.md`（由 `--tower-refs` 生成）。

工具本体的**覆盖范围与已知盲区**写在 `docs/scan_quote_refs.py` 的 docstring（第 5–34 行）；**本文件不复述全部判据口径**，只落读数与逐条事实。

---

## 0. 读数（事实）

命令（在仓根）：`python3 docs/scan_quote_refs.py`（语料层）；`python3 docs/scan_quote_refs.py --selftest`（工具层）。两项分别落 `scan.log` / `selftest.log`。

| 量 | 读数 | 说明 |
|---|---|---|
| `docs/*.md` 被扫份数 | **19**（read-fail 0） | `docs/` 下全部 `.md`（`docs/evidence/**` 与 `m*/README.md` 不在扫描面上） |
| 成对样本 `(引文, 文件:行)` | **31** | 与下面三栏**同一次运行、同一套匹配器**产出 |
| ├ verified | **11** | `OK` 5 + `OK-EMPH` 3 + `OK-PARTIAL` 3 |
| ├ unverified（跳过，原因分栏） | **12** | `shorthand-no-path` 8 + `bare-name` 3 + `external-prefix` 1 |
| └ finding | **8** | `FIND-NOTINFILE` 6 + `FIND-PARTIAL` 2 |
| 引文 `「…」` 总数 | **952** | 逐行匹配 921 + **源侧跨行 31** |
| ├ 成对受检 | **30** | 逐行 22 + 跨行 8 |
| └ 未成对（计入 `quote-unpaired`） | **922** | 逐行 899 + 跨行 23 |
| 源侧跨行引文（开闭不在同一行） | **31** | 其中开行左侧有出处 → 成对检查 **8**；无 → 计入 `quote-unpaired` **23** |
| 文件:行 总数 | **2001** | |
| ├ 成对受检 | **23** | |
| └ 未成对（计入 `cite-unpaired`） | **1978** | |
| `--selftest` 对照 | **14 条全部成立**（`rc=0`） | 含正/负合成、M125 修前/修后真实回归、源侧跨行引文 2 条、其它盲区 6 条 |

**退出码**（三态，`docs/17` §8.3）：主扫描当前 `rc=1`（有 finding）；`--selftest` 当前 `rc=0`。

**读法提醒**：`quote-unpaired=922` 与 `cite-unpaired=1978` 是**本工具不覆盖**的量（没有与本工具出处置信关系），**不是**「已核对通过」。

---

## 1. 任务 3：`--selftest` 的正负两侧与 M125 真实回归

`selftest.log` 逐条列出 14 条对照。要点：

| 对照 | 内容 | 当前 |
|---|---|---|
| NC-1 / NC-2 | 合成样例：引文正确 ⇒ 判绿（`OK`）；引文改错 ⇒ 判红（`FIND-NOTINFILE`） | 成立 |
| **NC-13** | 合成**源侧跨行**引文（开 `「` / 闭 `」` 分处两行）：正确 ⇒ 判绿、改错 ⇒ 判红，**各自出一条记录**（不是静默零输出） | 成立（`crossline_records=2 ok=1 find=1`） |
| **NC-14** | 合成**源侧跨行且开行无出处**的引文 ⇒ 计入 `quote-unpaired` 分母（可见，不是零输出） | 成立（`crossline=3 bound=2 quote-unpaired=1`） |
| NC-8 | **M125 B1 真实回归（修前）**：`docs/17` §9.9 末行修正前那一行（第二处引文挂 `…m123-r2.md:88`）⇒ **判红** | 成立（`findings=1`） |
| NC-9 / NC-10 | **M125 B1 真实回归（修后）**：修正后那一行 ⇒ 判绿，且第二处引文**确实被核到** `.tower/comms/reviews/review-feat-m115-b1-gdn-prefill-chunk-scan-reviewer-l0sync-r5.md:70` | 成立 |
| NC-3 / NC-4 / NC-5 / NC-6 / NC-7 / NC-11 / NC-12 | 盲区与解析登记：省略号引文按段核、裸行号、外部前缀、裸文件名、引文无紧邻出处、`docs/NN` 简写解析、`FIND-ELSEWHERE` | 成立 |

修前/修后两版文本**逐字**取自 M125 分支的 `docs/17`（`git show feat/m125-verification-governance-landing-and:docs/17-verification-standard.md`，修前版 = `git show 3534681:…`）与评审记录（`.tower` 未被 git 跟踪，故按主检出根解析，见 `docs/scan_quote_refs.py` 的 `main_worktree_root()`）。

---

## 2. 任务 4a：全 `docs/` 扫描报告（逐条）

### 2.1 finding 逐条（8 条；每条给判据与出处）

| # | 位置 | 引文（本工具抽出） | 所引 `文件:行` | 判定 | 判据（同一次运行产出） |
|---|---|---|---|---|---|
| 1 | `docs/17-verification-standard.md:672` | `` `docs/17` §9.5、§8.1 `` | `docs/05:379` | `FIND-NOTINFILE` | 该引文不是 `docs/05` 第 379 行的连续子串，也不是 `docs/05` 全文（任一处）的连续子串 |
| 2 | `docs/17-verification-standard.md:678` | 同行同引文 | `docs/05:379` | `FIND-NOTINFILE` | 同上 |
| 3 | `docs/18-vector-api-audit.md:52` | 「Reg API 必须在 vector function（`__VEC_SCOPE__`）内使用」 | `docs/05:114` | `FIND-NOTINFILE` | 引文不在 `docs/05` 第 114 行、也不在 `docs/05` 全文 |
| 4 | `docs/18-vector-api-audit.md:459` | `halfScale 非有限覆盖 = 缺失` | `docs/13:117` | `FIND-NOTINFILE` | 引文不在 `docs/13` 第 117 行、也不在 `docs/13` 全文 |
| 5 | `docs/18-vector-api-audit.md:459` | `已立 mission…修复前不要声称位级一致`（含省略号） | `docs/13:139`（裸行号 `:139`） | `FIND-PARTIAL` | 省略号切段后有所引行里找不到的段；且各段按序在被引文件的其它位置也找不到 |
| 6 | `docs/20-kernel-compliance-sweep.md:522` | `unlock 之前的 S 写对 MTE3 可见` | `docs/05:19/135/143` | `FIND-NOTINFILE` | 引文不在所引三行、也不在 `docs/05` 全文。**读法**：该 `「…」` 在原文里是**被讨论的命题**（「…未被建立」），**不是** docs/05 的引文；配对由同行紧邻规则产生，属 docstring「已知盲区」②（本工具不保证语义归属）——**不是**「docs/20 挂错出处」 |
| 7 | `docs/20-kernel-compliance-sweep.md:593` | `事件 **或** drain BufferID` | `docs/06:76` | `FIND-NOTINFILE` | 引文（含 `**` 归一后）不是 `docs/06` 第 76 行的连续子串，也不在 `docs/06` 全文 |
| 8 | `docs/20-kernel-compliance-sweep.md:673-675`（**源侧跨行**） | 「跨 pipe 的 UB 数据交接必须走 `SetFlag/WaitFlag` 事件 或 drain BufferID……」（拼接后） | `docs/06:76` | `FIND-PARTIAL` | 原文以「逐字」引出该段；拼接（去换行缩进、去续行 `> ` 标记）后，省略号切段的第一段跨过源处「事件（MTE2_V、…）或」中**被省掉的括号内容**（该处未加省略号）⇒ 该段不是连续子串 |

**三条同一文件、就近行的对照读数**（仅列事实，帮读者判读上表；命令见 §4）：

```
$ sed -n '385p;419p' docs/05-megakernel-design.md | grep -n 'docs/17'      # docs/05:379 为空行；相关文字在 419（两处 § 之间夹了括号说明 ⇒ 非连续子串）
$ sed -n '123p' docs/13-mx-quant-primitives.md                              # 含「**halfScale 非有限覆盖**」，无「 = 缺失」；第 117 行是另一行（clamp 下界）
$ sed -n '145p' docs/13-mx-quant-primitives.md                              # 含「修复前不要声称」，在第 145 行（所引 `:139` 不在该行）
```

⇒ 上表 8 条**是事实读数**：「引文与所引行不吻合」。**本报告不对"该不该改、算不算错"下判断**（性质与处置由塔/评审裁）。这 8 条都不改任何 docs 内容。

### 2.2 verified 逐条（11 条）

> **M186 重锚（2026-10-05）**：本表是该 report 时点（M129 base `53c2107`）的历史读数，§0 的计数不随本轮更新；但 `docs/20:511`/`:512` 两行已随 `docs/05` 的 release-mode 改正**重锚到现文本**（下两行，引文/所引行已更新）。

| 位置 | 引文 | 所引 | 判定 |
|---|---|---|---|
| `docs/17-verification-standard.md:450` | `与 live 一致` | `baseline_env/README.md:493-494` | `OK` |
| `docs/20-kernel-compliance-sweep.md:511`（M186 重锚） | 跨 pipe 数据交接…`RlsBufInternal<pipe,false>`… | `docs/05:19`（裸行号→`docs/05`） | `OK` |
| `docs/20-kernel-compliance-sweep.md:512`（M186 重锚） | `` **全项目 release 一律 `false`**（docs/05 §6 硬件约束表 `:136`） `` | `docs/05:136` | `OK` |
| `docs/20-kernel-compliance-sweep.md:513` | `公开 Mutex::Lock/Unlock 的 mode 硬编码为 0——就用 mode 0` | `docs/05:143` | `OK` |
| `docs/20-kernel-compliance-sweep.md:514` | 跨 pipe 的 UB 数据交接必须走 `SetFlag/WaitFlag` 事件… | `docs/06-m0-bringup.md:76` | `OK-PARTIAL` |
| `docs/20-kernel-compliance-sweep.md:47-48`（跨行） | `m=1 在 mmad 被 pad 成 16（matmul.h:670-672 "m==1→16"——mega kernel 同样 m 走 16 行矩阵）` | `docs/11-attn-analysis.md:103` | `OK-EMPH` |
| `docs/20-kernel-compliance-sweep.md:597-598`（跨行） | `` `PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，不能保证 UB 数据路径对另一 pipe 可见 `` | `docs/06:76` | `OK` |
| `docs/20-kernel-compliance-sweep.md:614-615`（跨行） | `…否则 aclrtSynchronizeStream 返回成功而 host 读回丢失/滞后（标量 SetValue 直写 GM 同样不可靠，…）` | `docs/06-m0-bringup.md:78` | `OK-PARTIAL` |
| `docs/20-kernel-compliance-sweep.md:616-617`（跨行） | `\| 标量 SetValue 写 GM \| kernel 退出时可见性无保证，元数据一律走 MTE3 \|` | `docs/06-m0-bringup.md:91` | `OK` |
| `docs/20-kernel-compliance-sweep.md:900-901`（跨行） | 同 `:614-615` 的引文 | `docs/06-m0-bringup.md:78` | `OK-PARTIAL` |
| `docs/20-kernel-compliance-sweep.md:902-903`（跨行） | 同 `:616-617` 的引文 | `docs/06-m0-bringup.md:91` | `OK` |

### 2.3 unverified 逐条（12 条，原因分栏）

| 位置 | 所引 | 原因 |
|---|---|---|
| `docs/16-vllm-ascend-qwen4exp-plan.md:27` | `docs/source/tutorials/models/Qwen3.8-Flash-Next.md:9`（带 `[asc-main]` 前缀） | `external-prefix` |
| `docs/16-vllm-ascend-qwen4exp-plan.md:92` | `README.md:82` | `bare-name` |
| `docs/16-vllm-ascend-qwen4exp-plan.md:92` | 裸行号 `:136`（同行无前置路径） | `bare-name` |
| `docs/20-kernel-compliance-sweep.md:688` | 裸行号 `:22` | `shorthand-no-path` |
| `docs/20-kernel-compliance-sweep.md:907` | 裸行号 `:891` | `shorthand-no-path` |
| `docs/20-kernel-compliance-sweep.md:1226` | 裸行号 `:662` | `shorthand-no-path` |
| `docs/21-scalar-computation-sweep.md:259` | 裸行号 `:188` | `shorthand-no-path` |
| `docs/21-scalar-computation-sweep.md:389-390`（**源侧跨行**） | `m15_moe_layer.h:1887` | `bare-name`（裸文件名不解析 —— 见下） |
| `docs/21-scalar-computation-sweep.md:981` | 裸行号 `:793` | `shorthand-no-path` |
| `docs/21-scalar-computation-sweep.md:983` | 裸行号 `:845` | `shorthand-no-path` |
| `docs/21-scalar-computation-sweep.md:984` | 裸行号 `:870` | `shorthand-no-path` |
| `docs/21-scalar-computation-sweep.md:985` | 裸行号 `:883` | `shorthand-no-path` |

**unverified ≠ 通过**（`docs/17` §9.2 第 2/3 条）：这 12 条**本工具没有核**，逐条原因如上。

**关于 `docs/21:389-390`**：该行以跨行引文逐字引 `m15_moe_layer.h:1887`。本仓里 `m15_moe_layer.h` 位于 `m15_layer_loop/m15_moe_layer.h`；因为**裸文件名不解析**（§2.4 的「跳过」一栏），这一条落在 `unverified:bare-name`，**被点出来但没有被判定**。若要判定它，需要把出处写成带 `/` 的仓内路径（或让本工具接受唯一同名解析 —— 后者与 M39 教训冲突，未采用）。

### 2.4 分类与盲区（逐类说明：finding / 跳过 / 盲区）

| 形态 | 本工具的处理 | 依据 |
|---|---|---|
| 引文含 `**` 加粗标记 | 先逐字核；不中则去掉双方 `**` 再核，记 `OK-EMPH`（verified 栏的独立一档） | §2.2 第 2/3/6 行 |
| 省略号截断（`…`/`...`） | 按省略号切段、各段按序核；全中所引行 ⇒ `OK-PARTIAL`；有所引行找不到的段 ⇒ finding | §2.1 #5/#8、§2.2 #5/#8/#10 |
| **源侧**跨行 `「…」`（开闭不在同一行） | 整份文本匹配后映射回行号；**开行左侧最近一个** `文件:行` 作出处；引文按「去换行缩进 / 折成空格」两种拼法各核一次（续行 `> ` 标记拼前去掉）。**开行有出处 ⇒ 成对检查；无 ⇒ 计入 `quote-unpaired` 分母** | §0 的 `source 跨行` 两栏；§2.1 #8、§2.3 第 8 行 |
| **目标侧**跨行/合并行（「`:18`（同 `:20`）」） | 对同一被引文件内所列各行的**合并文本**再判一次，取更优判定（`cite_kind=merged-lines`） | §2.2 #6–#11 的跨行命中 |
| 表内「（同 `:xx`）」式指代（裸行号） | 只回看**同行**最近前置 `文件:行`，其次最近的 `docs/NN` 提法；都没有 ⇒ `unverified`（**跳过**） | §2.3 的 `shorthand-no-path` 8 条 |
| 所引文件在 `.tower/**` | 若本机（主检出）存在则**照核**，并标注「目标在 `.tower/`、git 未跟踪」 | 清单见 `tower_refs.md` |
| 外部快照（`[asc-main]`/`[vLLM]`/绝对路径） | **跳过**（`unverified`） | §2.3 第 1 行 |
| 裸文件名（`README.md`、`m15_ple.asc`、`m15_moe_layer.h`） | **跳过**（`unverified`）；**不按唯一同名猜**（M39 已实测同名会拿错文件，`docs/17` §9.2 触发事件 1） | §2.3 第 2/3/8 行 |
| ASCII 引号 `"…"` 的引用 | **不覆盖**（本工具只认 `「…」` 定界）。例：`docs/18-vector-api-audit.md:436` 用 ASCII 引号引 `docs/05:114`，与 `:52`（`「」`）是同一处失配，但本工具**不列** `:436` | docstring「已知盲区」⑤ |
| 引文与出处分处**不相邻**的表格列 | **不配对**（本工具不覆盖） | docstring「已知盲区」③ |
| 引文与出处同处一行但**不紧邻**（无分隔符关系） | **不配对**（本工具不覆盖）；计入「未成对的引文」 | 同栏 ① |
| 文档 §0 **显式钉死的不可变 rev**（「本文件里出现的所有行号都以这个 rev 为准」） | **不建模**：工具按**当前**文件核引文，不按文档声明的 rev 核 ⇒ 对这类文档，「引文不是当前行的子串」类 finding **可能是**该文档按 rev 写、而工具按当前文件核所致（**须按该 rev 复核**）；**按该 rev 复核后仍不吻合的，仍是引文保真问题、不在本条之内** —— 本工具不判其归属 | M130：`docs/20`（rev `20bd20d`）、`docs/21`（rev `9296e79`）、`docs/18`（抬头 rev `55f5e18`）等 survey/审计文档的声明 |

**本工具不覆盖**（明写，不得读成通过）：各行/各单元格内多条引文与多个出处的**语义归属**（单元格级绑定是集合式）；不相邻列的引文-出处关系；跨行/跨单元格的裸行号指代；**ASCII 引号 `"…"`**；**源侧跨行引文在开行左侧没有出处**时（计入分母但不被核）；以及 `quote-unpaired=922`、`cite-unpaired=1978` 这两批量本身。另：**文档 §0 的 rev 钉死约定**不在本工具建模内 —— 对显式声明不可变 rev 的文档，「引文不是当前行子串」类 finding 属**本工具的已知盲区**（应按该文档声明的 rev 复核；**按该 rev 复核后仍不吻合的，照旧是引文保真问题**），既**不得**据此断为假阳、也**不得**据此断为真错。**「本工具无 finding」不等于「docs 的引文都没问题」**。

---

## 3. 任务 4b：`.tower/` 引用清单

**见 `tower_refs.md`（逐处表格，由 `--tower-refs` 同源产出）**。事实摘要：

- `docs/*.md` 里 `.tower/` 引用**处数 19**（**带路径 18 + 裸 `.tower/` 1**，裸的那处在 `docs/20-kernel-compliance-sweep.md:635`）；**去重 12**。
- `git ls-files .tower` 的**输出行数 = 0**。
- 按主检出根判「本机是否存在」：**存在 8 处、不存在 11 处**（不存在的都是 `.tower/worktrees/wt-25|wt-37|wt-104` 下的旧 worktree 路径）。
- 12 条去重路径**全部**「git ls-files 是否跟踪 = 否」。

**本报告不判这些引用是否「无效/悬空」，也不判交付后是否可达**——只列上列事实与复现命令；可达性由塔裁决。

---

## 4. 复现命令

```bash
# 工具层自检（14 条对照；rc=0 表示全部成立）
python3 docs/scan_quote_refs.py --selftest

# 语料层主扫描（汇总 + finding）
python3 docs/scan_quote_refs.py

# 逐对明细（report (a) 第 2 节的原始素材）
python3 docs/scan_quote_refs.py --dump-pairs

# report (b)：`.tower/` 引用清单
python3 docs/scan_quote_refs.py --tower-refs

# 一键重建本目录全部读数
bash docs/evidence/quote_refs/repro.sh
```

---

## 5. 附：对 M125 分支语料的读数（**不在本 base 的 docs 里**）

本工具还能对**未合入**的分支语料读数，用于验证它对触发事件 M125 的适用性（命令：把 `--docs-dir` 指向该分支 `docs/` 的提取副本）：

```bash
mkdir -p /tmp/m125docs && cp docs/*.md /tmp/m125docs/
git show feat/m125-verification-governance-landing-and:docs/17-verification-standard.md \
    > /tmp/m125docs/17-verification-standard.md
python3 docs/scan_quote_refs.py --docs-dir /tmp/m125docs --dump-pairs
```

读数（时点：本 base，语料 = M125 分支 `docs/17` + 本 base 其余 docs）：`pairs=39 verified=17 unverified=12 finding=10`。与 M125 新增内容相关的附加 finding 有 **4 条**，全在 `docs/17`：

- `docs/17:778` 与 `:780` —— **§9.10 触发事件表第 1 行（M116 r3）/ 第 3 行（M116 r5）**（该分支 `### 9.10` 在 `:766`、`### 9.11` 在 `:789`；§9.11 的表行 `:801-804` 无一被判红）。判据是「引文与所引行只差空白 / `**` 落位 / 拼接符」，不是「引文挂错出处」。
- `docs/17:823` 与 `:829` —— 与 §2.1 #1/#2 同形的 `` `docs/17` §9.5、§8.1 `` → `docs/05:379`（本 base 里是 `:672`/`:678`，M125 分支上因行数前移变成 `:823`/`:829`）。

**§9.9 修前/修后那一行**的判红/判绿由 `--selftest` NC-8/NC-9 复现。**这些行不与本报告 2.1 的 8 条同域**（它们在 M125 分支上，不在本 base）；是否要按逐字口径改，由塔/作者定 —— 已另报塔为 finding `20261004-agent-quotecheck-improve-m125-docs-17-9-11-2`，其**节号标错**（§9.11 → 实为 §9.10）已由复审 r1 核出并另发更正 finding `20261004-agent-quotecheck-improve-m125-docs-17-2-9-10-9-11`。

---

## 6. 任务清单对账

| mission 任务 | 落点 |
|---|---|
| 建 `docs/scan_quote_refs.py`（三态退出码 + `--selftest` + 覆盖范围交代） | `docs/scan_quote_refs.py`（docstring 第 5–34 行 = 覆盖范围与盲区，含**源侧跨行引文**与 **ASCII 引号**两条限制） |
| 各类形态逐类说明（finding / 跳过 / 盲区） | 本文件 §2.4（含「源侧跨行引文」与「ASCII 引号」） |
| `--selftest` 正负两侧 + M125 真实回归（修前判红） | 本文件 §1；`selftest.log`（14 条） |
| 报告 (a) 全 `docs/` 扫描 | 本文件 + `scan.log` + `scan_pairs.log` |
| 报告 (b) `.tower/` 引用清单 | `tower_refs.md` |
| 纪律：不改 docs 内容 / `文件:行` / 零设备 / 分块提交 / `df -h /` | `git diff --name-status <base>...HEAD` 只含 `docs/scan_quote_refs.py` 与 `docs/evidence/quote_refs/**`；零设备；本文件每条结论带 `文件:行` |
