# M114 证据：`docs/scan_doc_refs.py` 章节判定「前导 `§`」修复

**基准**：本分支 base = `0a9922374f55ba548ecab1e09ba11b0148cefcef`（main 的 `Merge branch 'feat/m112-land-m107-compliance-sweep-docs'`，即 M112 合入后），worktree `.tower/worktrees/wt-114`。**零设备**：全部命令只用到 `python3` / `git` / `bash`。
**复跑**：`bash docs/evidence/scan_doc_refs/repro.sh`（在仓根；一键重建本目录全部 log）。

## 0. 四句话结论

1. **影响面**：仓内带前导 `§` 的章节标题 **28 条，全在 `docs/20`**（`### §3.2 …` 形态，仓内唯一出处）；指向它们的 `docs/NN §X` 引用在 base `0a99223` 上是 **0 条**。
2. **这一类的性质要说准**：不是"被静默跳过"（假阴性），而是**恒判负**（假阳性）—— 引用是好的，坏的是判定。端到端 fixture：`见 docs/20 §3.2 与 docs/20 §5` 在**改前** rc=1（2 条 `SEC-REF`）、**改后** rc=0。
3. **改动面**：只放宽**目标标题那一侧**的判定（`^§?<num>`）。**匹配面（哪些文本算引用）、path 校验、退出码三态、`ALLOWLIST` 准入全未动** ⇒ 主扫描五组计数与 rc **改前改后逐字相同**（§4）。
4. **常驻判据**：`--selftest` 新增 **NC-6**（正 + 负 + 非空洞对照）；并把判定分别「退回改前 / 改宽成永真 / 改宽成永假」跑逆证，三档 NC-6 都 FAIL、rc=1（§3）。

---

## 1. 任务 1：影响面（先量后改）

命令（在仓根，`LC_ALL=C`）：

```
$ python3 docs/evidence/scan_doc_refs/measure_sec_refs.py . /tmp/m114_before/scan_doc_refs.py   # 改前
$ python3 docs/evidence/scan_doc_refs/measure_sec_refs.py . docs/scan_doc_refs.py               # 改后
```

读数落 `measure_before.log` / `measure_after.log`。两份 log **只差一行**（第 4 行「量尺 vs 模块口径一致性」：改前 `N/A（模块里还没有 _has_section）`、改后 `AGREE`；`diff` 实跑）—— 量尺的统计口径不随扫描器版本变，改后还与工具本体的判定函数对齐。

分域：`gate` = 主扫描真正判的（`docs/*.md` 去掉 `SELF`，17 个文件）、`self` = `docs/17` 自己、`outside` = 其余 tracked `*.md`（70 个）。**只有 gate 域决定 rc**。

**本目录自己的证据 `.md` 不计入语料**（量尺在 `# 本目录自己的证据 .md（**不计入语料**，n=1）` 一行里列出）：那些文字是**引用样本的转录**，让它参与会让读数随本报告的措辞漂移 —— 读数只应取决于语料。所以本报告写长写短，上表的数字都不动（`repro.sh` 复跑可核）。

### 1.1 ① 总量 与 ② 解析/跳过（markers = 仓内 `§<num>` 标记数）

| 域 | markers | `parsed`（`SECREF` 命中 ⇒ 进分母、会被判） | `gap_narrow`（第 ⑤ 栏形态） | `other`（扫描器不看） |
|---|---|---|---|---|
| gate | 1158 | 212 | 2 | 944 |
| self | 299 | 83 | 8 | 208 |
| outside | 2116 | 386 | 0 | 1730 |
| **仓内合计** | **3573** | **681** | **10** | **2882** |

### 1.2 `parsed` 那栏按标题口径 old/new 分桶（目标由 `SECREF` 唯一确定，无推断）

| 域 | 可解析（new 口径） | 其中「改前误判、改后救回」 | 仍 out-of-range | 目标文档有带前导 `§` 的标题 |
|---|---|---|---|---|
| gate | 212 | 0 | 0 | 0 |
| self | 55 | 0 | 28 | 0 |
| outside | 359 | 0 | 27 | 0 |

- **「改前误判、改后救回」= 0**：base `0a99223` 上没有任何 `docs/NN §X` 落在带 `§` 的标题上 ⇒ 这次修改**不改变当天语料的任何一条判定**。
- `self` 域那 28 条 = `docs/17` 正文里**故意引着**的负向样本（`docs/99-nothing.md`、`docs/07 §2`、`docs/11 §5.1`、`docs/15 §99.7` …）；`outside` 域那 27 条 = 各 `m*/README.md`、`baseline_env/**` 里指 `docs/17 §2.x` 这类**子项而非标题**的引用（25 条 → `docs/17`、1 条 → `docs/18`、1 条 → `docs/05`）。**两域都不在扫描面上**，不影响门禁。

### 1.3 ③ 修好之后会不会立刻新增一批 out-of-range ⇒ 多少真悬空

- **新增 = 0**：本次修改**不动匹配面**（`SECREF` 一行未改）⇒ `parsed` 集合不变 ⇒ 新增暴露面在构造上为 0；实测一致：gate 域 `parsed_bad` 改前 0 / 改后 0，五组计数逐字相同（§4）。
- **真悬空 = 0（gate 域）**：gate 域被 `parsed` 的 212 条里，改前改后都没有 out-of-range。今天唯一可能变判定的形态（引用指向带 `§` 标题）在当天语料上是 0 条。
- **埋雷规模**：`docs/20` 的 28 条 `§`-前导标题今天收到 **0 条** `parsed` 引用 ⇒ 修工具**不会牵出另一批文档工作**，可按小改动对待。

### 1.4 `skipped` 那栏（本次**未**扩，作为待裁决量交塔）

| 域 | `gap_narrow` | 其中悬空 | `other` | 该号在 0 个文档是标题 | 1 个 | ≥2 个 |
|---|---|---|---|---|---|---|
| gate | 2 | 0 | 944 | 22 | 41 | 881 |
| self | 8 | 0 | 208 | 6 | 17 | 185 |
| outside | 0 | 0 | 1730 | 218 | 44 | 1468 |

- **标定（量尺可信度的证据）**：gate 域 `gap_narrow` = **2 行**（`docs/09-moe-donor-map.md:4`、`docs/11-attn-analysis.md:4`，夹层都是 `' v1.2 '`），与 `docs/17` §8.6（M78）登记的历史读数「2 处（2 个文件的抬头行各 1 行）」**同数**，且两条都**可解析**（与 M78 的「两个目标在 `docs/05` 里都存在」一致）。注意 M78 的单位是"行"；按 `§` 标记算，每行还有第二个 `§6`（被"夹层里不含第二个 `§`"这条收紧规则排除），故按引用算每行是 2 个。
- `other` 那一栏**不推断目标**（裸 `§` 有歧义，这正是扫描器把它登记为盲区第 ④ 栏的理由），只给「该号的标题出现在几个文档里」的普查：gate 域 944 条里 **881 条**的编号在 ≥2 个文档里都有标题 ⇒ 扩到裸 `§` 会大面积误报，与既有 docstring 的判断一致；22 条在 0 个文档里（含 `§11.5` 这类明确指别处的简写）。**本 mission 未据此做任何改动。**

### 1.5 标题形态普查（"同一个问题"还有没有别的形态）

按扫描器自己的 `HEAD` 口径（`^#{1,6}\s+`；Python 的 `\s` 是 Unicode 空白 ⇒ **双 locale 读数一致**），标题文本分三桶（合计 694）：

| 形态 | 条数 | 说明 |
|---|---|---|
| `<num>…` | 412 | 其余 docs 的常规形态，`^<num>` 判得动 |
| `§<num>…` | 28 | **全在 `docs/20`**；`^<num>` 恒不匹配 ⇒ **M114 修的就是这一桶** |
| 其它（不以数字开头） | 254 | 如 `# 形态① path.ext:line …`、`### L0 — 单算子 / 单 kernel 级`、`#### M103-1.3 MoE 段…`。这一桶里也含被 `^#{1,6}\s+` 抓到的**代码块注释行**（如 `docs/14` 的 `#  ⇒ 原稿把…`），所以它**不是"章节数"** |

- 除 `§`-前导这一桶外，**没有**其它"标题形态与解引用口径对不上、且今天有引用指向它"的组合（gate 域 `parsed_bad` = 0）。
- 其余 254 条今天有 **0 条** `docs/NN §X` 引用指向。若要给它们做 `§`-号引用，得先决定 `M103-1.3` / `L0` 这类编号算不算"章节点位"——**本轮刻意不扩**（没有样本）。

---

## 2. 任务 2：改法（改动最小）

`docs/scan_doc_refs.py`：

- 新增 `_SEC_HEAD_LEAD = '§?'` 与 `_has_section(heading, sec)`（= `^§?<num>(?!\d)`）；生产路径 `scan_docrefs()` 与自检里的另外两处标题判定（NC-4 的 `sec_exists`、NC-5 (c) 的 `v_ok`）**统一改用它**（一处口径，不许两套）。
- 新增 `_has_section_legacy(heading, sec)`（= 改前的 `^<num>`），**只**给 NC-6 (c) 当"改前"对照，生产路径不用。
- **未做**（逐条对照纪律）：
  - `SECREF` / `_SEC_PATH` / `_SEC_TAIL` / `DOCREF` / `PATHREF` / `HEAD` **一行未改** ⇒ 匹配面与 path 部分未放宽；
  - 判定阈未动：`docs/NN` 必须存在、`§X` 必须在目标文档**标题**里存在、退出码三态、`ALLOWLIST` 准入规则；
  - **不凭猜扩字符集**：`§` 实测只有 U+00A7 一种（`docs/*.md` 里 1515 次，全部同一码位），**全角/半角变体 0**、`第<num>节/章` 写法 **0**、标题里 `§` 后紧跟空白的形态 **0** ⇒ 只加 `§` 本身，没有 `\s*` 之类的无样本分支；
  - 正则字符类全 ASCII（`§` 在模式里是**字面量**，不进字符类）。
- docstring：覆盖表 ② 行补一句判定侧的说明；新增「M114 放宽判定面」段（性质、不扩集的证据、读数位置）；`--selftest` 的 5 条 → 6 条（docstring、`--help`、`RESULT` 串三处同步）。

---

## 3. 任务 3：常驻判据（正 + 负）

`python3 docs/scan_doc_refs.py --selftest`（全文见 `scan_after.log`）。新增 NC-6，样本逐字：

```
sign-heading: docs/20 §3.2 and docs/20 §5 ; bad: docs/20 §99.7 and docs/20 §10 and docs/17 §99.7
```

- (a) **负向**：`docs/20 §99.7`、`docs/20 §10`、`docs/17 §99.7` **必须**被登记（改后实测 `bad_sec = ['docs/20 §99.7', 'docs/20 §10', 'docs/17 §99.7']`）。
  `§10` 特意选它：`docs/17` **有** §10 ⇒ 同时证明"容忍前导 `§`"没有变成"任意数字都算"；`docs/17 §99.7` 证明"目标标题不带 `§`"的那一半也仍然会咬。
- (b) **正向**：`docs/20 §3.2`、`docs/20 §5` **必须不**被登记，且**确实被匹配到**（改后实测命中 5 条，`secs6 = ['10', '3.2', '5', '99.7']`）—— 防"压根没匹配上"的假通过。
- (c) **非空洞**：同一份样本改用 `_has_section_legacy()` 跑，(b) 那 2 条**必须全被判负**（实测 `pre6 = ['3.2', '5']`）⇒ 证明这条对照测的是**新**判定面。
- 并当场核 `docs/20` 里确实有以 `§` 开头的标题（实测 28 条）。

**逆证三档**（`check_inverse_controls.sh` → `nc_inverse_controls.log`）：

| 改法 | NC-6 | `--selftest` rc | 顺带被带出的 FAIL |
|---|---|---|---|
| `_has_section = _has_section_legacy`（退回改前） | FAIL（正向 2 条被判负） | 1 | 无 |
| `_has_section = lambda h, s: True`（改宽成永真） | FAIL（负向一条都没登记） | 1 | NC-1、NC-4 |
| `_has_section = lambda h, s: False`（永假） | FAIL（正向全被判负） | 1 | NC-5 |

⇒ "改宽成永真"这道闸确实在咬（NC-6 (a) 与 NC-1 同时报 FAIL）。

---

## 4. 任务 4：全仓回归（改前 vs 改后）

命令：`bash docs/evidence/scan_doc_refs/repro.sh`（一键重建 `scan_before.log` / `scan_after.log`）。「改前」= base `0a99223` 上的 `docs/scan_doc_refs.py`（**钉不可移的 sha**，不钉 HEAD —— 本 mission 落库后 HEAD 上就是改后的扫描器）。

| 命令 | 改前（`LC_ALL=C` / `C.UTF-8`） | 改后（`LC_ALL=C` / `C.UTF-8`） |
|---|---|---|
| 主扫描 | doc refs=219 (unresolved=0) · section refs=212 (unresolved=0) · path refs=347 · gating=0 (distinct=0) · allowlisted=2 (distinct=2) · **rc=0** | **逐字相同** · **rc=0** |
| `--include-self` | doc refs=323 (unresolved=14) · section refs=295 (unresolved=28) · FAIL(14 doc + 28 sec + 17 path) · rc=1 | 逐字相同 · rc=1 |
| `--ignore-allowlist` | FAIL(0 doc + 0 sec + 2 path) · rc=1 | 逐字相同 · rc=1 |
| `--selftest` | `OK (5 条负向对照全部成立)` · rc=0 | `OK (6 条负向对照全部成立)` · rc=0 |

- **新增 out-of-range 逐条分类：0 条**（gate 域）。分母（`section refs=212`）也**未变** ⇒ 无需重基线。
- **端到端 fixture**（`run_e2e_fixture.sh` → `e2e_fixture.log`；fixture = `docs/` 真拷贝 + 其余顶层项与 `.git` 走 symlink，探针文档注入为 `docs/98-m114-probe.md`）：

| 探针 | 扫描器 | 读数 | rc |
|---|---|---|---|
| `见 docs/20 §3.2 与 docs/20 §5`（两条都真实存在） | 改前 | `SEC-REF ×2`（两条都被判负）；section refs=214 (unresolved=2) | **1** |
| 同上 | 改后 | 无 `SEC-REF`；section refs=214 (unresolved=0) | **0** |
| `见 docs/20 §99.7`（不存在） | 改前 | `SEC-REF ×1` | **1** |
| 同上 | 改后 | `SEC-REF ×1`（**仍红**） | **1** |

⇒ 正反两侧都在**真扫描器 + 真语料**上跑过：好引用 2 → 0 unresolved、坏引用两版都红。

---

## 5. 顺带发现（"检查别处有没有同样的坑"）—— 交塔，本 mission 不改

1. **`docs/17` §10 逐字引着 `--selftest` 的转录**（含 `# RESULT: OK (5 条负向对照全部成立)` 与 5 条对照的输出）。M114 把它变成 **6 条** ⇒ 那段转录**已过期**。`docs/17` 不在本 mission 的 scope（scope = `docs/scan_doc_refs.py` + `docs/evidence/scan_doc_refs/**`），且它默认不在扫描面上（`SELF`）⇒ 不 gate。**需另派单**。
2. **`outside` 域 27 条** `docs/NN §X` 被判 out-of-range（逐条见 `measure_after.log` §B），涉及 `m10_attn_decode/README.md:57`、`m13_moe_layer/README.md:232/236/243/350`、`m13_moe_layer/evidence/README.md:12/33`、`m15_layer_loop/README.md:574`、`m17_moe_real/README.md:20/226/230/457`、`m18_gdn_prefill/README.md:264`、`m1_mxfp4_gemm/README.md:53`、`m20_hyperconn/README.md:480/545`、`m21_layer_ref/README.md:429/468`、`m22_router512/README.md:38`、`m4_gdn_recurrent/README.md:99`、`probe_v_align/README.md:328/343`、`tools/golden/README.md:214/588`、`baseline_env/README.md:476/502`、`baseline_env/evidence/13-number-audit-script-audit.md:188`。其中 25 条指向 `docs/17 §2.x` 这类**子项**（`docs/17` 的 `## 2. 判据的写法规范` 下面的分条不是标题）。这些文件**不在扫描面上**，不影响门禁；**是否清理由塔定**（本 mission 只修工具，且不许改这些文件）。
3. **第 ⑤ 栏（版本号夹层）2 行**、**裸 `§` 盲区 944 条（gate 域）** 保持现状：本次刻意未扩匹配面，理由与普查见 §1.4。

---

## 6. 交付物与复跑

| 文件 | 内容 |
|---|---|
| `measure_sec_refs.py` | 影响面量尺（import 扫描器本体；只统计不推断；参数 `[repo_root] [scanner_path]`） |
| `check_inverse_controls.sh` | 逆证三档（退回 / 永真 / 永假） |
| `run_e2e_fixture.sh` | 端到端 fixture（改前/改后 × 好/坏探针） |
| `repro.sh` | 一键重建本目录全部 log |
| `measure_before.log` / `measure_after.log` | 影响面读数（两份只差"口径一致性"一行） |
| `scan_before.log` / `scan_after.log` | 主扫描 + `--include-self` + `--ignore-allowlist` + `--selftest`，双 locale |
| `nc_inverse_controls.log` | 三档逆证的完整输出 |
| `e2e_fixture.log` | 端到端四跑的完整输出 |
| `.gitignore` | 屏蔽本目录的字节码缓存与 fixture 临时文件 |

复跑：`bash docs/evidence/scan_doc_refs/repro.sh`（在仓根，零设备）。

---

## 7. 纪律自检

- 上面每条读数都出自本目录的 log（命令 → 输出/rc 都在 log 里），没有未实跑的断言。
- 读数一律标**基准 sha**（`0a99223`）与命令；`skipped` 栏的目标**不推断**，只给歧义普查。
- 双 `locale`（`LC_ALL=C` / `C.UTF-8`）逐条一致（`scan_*.log` 里两段并列）。
- 正则字符类全 ASCII；`§` 在模式里是字面量。
- 零设备。
