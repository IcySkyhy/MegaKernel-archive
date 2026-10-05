# M100 — PLE 段接进层 kernel 挂载点：device 见证与读数（task 3）

> 命令 → 输出/rc 的转录在 `m15_layer_loop/evidence/ple_wire/*.log`（每份日志末尾一行 `rc=`）。
> 采集脚本 = `evidence/ple_wire/run_wire_evidence.sh`（**每一步之前**查 `npu-smi info`，见塔的
> 环境硬纪律 A）。引用的都是内容锚点，不钉行号。
>
> **本文件里的每条读数都是同一份二进制**：`binary_sha256.txt` 记的 sha256 与所有 `.log` 同批。
> 复现：`bash m15_layer_loop/evidence/ple_wire/run_wire_evidence.sh`（环境：`npu-smi info` 空闲、
> `df -h /` 有余量）。

---

> ## ⚠ 本目录的 `*.log` 已在 **M111** 重采（旧读数按批次读）
>
> M111（`feat/m111-ple-gm-scalar-landing-path-fix`）修了 ① 的 ids 落盘通路（U-A），
> 因此**本目录的 `*.log`、`binary_sha256.txt`、`runs_all.log` 是本 branch 的批次**，
> 不再是 `1cf6ecad…` 那一批。M111 的完整读数、命令与消费者盘点在
> **`M111_LANDING_PATH.md`**；本文 §3 的表已按新批次更新受影响的档（B / F / G），
> 未受影响的档（A / C / D / E / F2）读数不变。M111 之前那批的读数仍可从 git 历史取出
> （`git show <M100 的 commit>:m15_layer_loop/evidence/ple_wire/<log>`）。
>
> **r2（落盘通路改 ⓔ 形态）同样整批重采**：二进制 `e5971447…` → `aa3652f1…`，
> 七个接线档的 `Pw.*` 行与 `runs=all` 的判据行/runs 末行与 r1 **逐行相同**（只差时序毫秒）。

---

## 1. 接的到底是什么（一句话）

`m15_layer_kernel.h::M15L_FusedBody` 的四相位 AIV 段里，层 1（`hcPleBreak != 0`）的 H1 现在是：

```
hc0.ProcessAiv()（MODE_COMBINE_ONLY：把 bf16 H' 物化到 hcWs0+WS_HCP）
  └─ M15L_PleIds（① n-gram id → scratch slab 的 S_IDS）
     └─ FLAG_PLE_IN_BOUND_AIV（已登记的 8；全体 AIV mode-0）
        └─ M15L_PleBody（②表行 gather → ③kv 投影 → ④门控 → ⑤膨胀卷积+残差+状态移位）
           └─ FLAG_PLE_OUT_BOUND_AIV（已登记的 9）
              └─ hcMix.ProcessAiv()（MODE_MIX：读被 PLE 就地改过的 H'）
```

`pleW == nullptr`（接线关）时逐字退化为 M65 的空操作 —— 这两条边界与 `PipeBarrier<PIPE_ALL>`
照旧执行。

---

## 2. 判据（**当前 tip 12 条**；分档按 `docs/17 §1.1`）

> **计数口径（M179 起）**：本目录 committed 的 `*.log` 是 **M111 批次**（二进制 `aa3652f1…`），
> 其 `Pw` 判定项 = **10 条**。M124（`6a8fee3`，已合入 main）在判据块里新增了 `Pw.kv.T3` /
> `Pw.kv.nonvac` 两条 ⇒ **当前 tip 的 `Pw` 判定项 = 12 条**（M124 批次转录
> `m124/M124_wire_{A,B}_kv.log`：A / B 档均 `12/12 PASS`、`checks=144`）。本文 §3 的档表里
> A / B 两行按当前口径记 12；C / D / E / F / F2 的 12 条口径读数**未归档** ⇒ 那些行只标「M111 批次」，
> 其 `delta` / `max|Δ|` / `row_fail` / `miss` 等数值一律不改。`*.log` 里打印的「10 条」是历史转录，按纪律不改。

| # | 判据 | 档 | 量 | 被测对象 |
|---|---|---|---|---|
| 1 | `Pw.wired.nonvac` | 非空洞性 | 10240 bf16 | 接线开/关的**同一层入口多流态**（`ws0+WS_HCP`）必须不同 |
| 2 | `Pw.wired.finite` | 结构 | 10240 bf16 | 接线开的那块多流态必须有限（NaN/Inf 计数） |
| 3 | `Pw.ids.T1` | **T1 逐位** | 16 int64 | ① 的 16 个 id vs **host 独立参考**（按 `ple/PLE_SPEC.md §4.1 ①` 重写，含 EOS 回退与 int64 回绕） |
| 4 | `Pw.emb.T1` | **T1 逐字节** | 16×160 bf16 | ② 取到的行 vs host 从**同一窗口**读回的字节 |
| 5 | `Pw.kv.T3` | **T3**（M124 新增） | n_tok × 12800 bf16 | ③ kv 投影的数值 vs **host-only 参考**（窗口字节 + host ids 重建 emb，× host `wcat`，double 点乘），界 `2560·2⁻²⁴·Σ\|terms\| + 1.0·ulp(out)` |
| 6 | `Pw.kv.nonvac` | 非空洞性（M124 新增） | 同 #5 | 参与行 > 0 且参照面非零 |
| 7 | `Pw.win.dev` | 设备侧判据（M92） | 128 核 × 4 槽 | 设备自己比过的**错行数** `row_fail`、**越窗行数** `miss`、非空洞核数、见窗核数 |
| 8 | `Pw.fail` | 设备侧计数 | 128 × 2 槽 | ① 的值域坏 id 计数（`dev_range_fail` 同源）+ ②③④⑤ 的坏值计数 |
| 9 | `Pw.state.evolve` | **T4** | 9×10240 bf16 | conv 状态输出 ≠ 输入（活跃行确实演化） |
| 10 | `Pw.state.shift` | **T1 逐字节** | 9×10240 bf16 | 状态移位 `new[i]=old[i+1]`（i<8）且 `new[8]=normed` |
| 11 | `Pw.planes.nonvac` | 非空洞性 | gated/normed | 两个平面都被写出且非常量 |
| 12 | `Pw.det` | 确定性 | 全部产物 | 同配置两次接线跑逐字节相同 |

**读数项（不计入判定）**：`Pw.wired.delta` = PLE 改动了多少个 bf16 元素 + `max|Δ|`。

---

## 3. 档位与读数

（下表由 `run_wire_evidence.sh` 一次采集；`rc` 是所有档共用的退出码口径：本档 `fails != 0` ⇒ rc=1。**判定项 / FAIL 列**：A / B 记当前 **12 条**口径（`m124/M124_wire_{A,B}_kv.log`）；C / D / E / F / F2 记 **M111 批次** 10 条口径（12 条口径未归档）—— 见 §2 计数口径注）

| 档 | 环境 | 判定项 / FAIL | 关键读数（`evidence/ple_wire/*.log`） |
|---|---|---|---|
| **A** `A_body_stage15` | `M15_PLE_WIRE=1 M15_PLE_STAGE=15`（②③④⑤；ids 由 host 参考灌入） | **12 / 0**，rc=0（M124 批次；M111 批次 10 / 0） | `emb bad=0 miss=0/16`；`row_fail=0 miss=0 非空洞核=56 见窗核=56`；`delta=5636/10240 max|Δ|=0.427734` |
| **B** `B_full_stage31` | `…STAGE=31`（①+②③④⑤ 全跑） | **12 / 0**，rc=0（M124 批次；M111 批次 10 / 0，修 U-A 之前是 10 / **3**，rc=1） | `emb bad=0 miss=0/16`；`row_fail=0 非空洞核=56 见窗核=56`；`delta=5636/10240 max|Δ|=0.427734`（与 A 档同值）。修前：`emb bad=15/16`、`row_fail=15`、`det` 红 ⇒ U-A（§4，M111 已修） |
| **C** `C_wire_off` | `M15_PLE_WIRE=0`（接线关） | 10 / **5**，rc=1（M111 批次） | `delta=0/10240`（PLE 没做事） |
| **D** `D_miswire_table` | `…MISWIRE=1`（表基址接到 scratch slab） | 10 / **5**，rc=1（M111 批次） | `emb bad=16/16`；`delta=0` |
| **E** `E_miswire_winbase` | `…MISWIRE=2`（行基址偏 1） | 10 / **2**，rc=1（M111 批次） | `emb bad=16/16`；`row_fail=16`；`delta=5518/10240`（本批读数；与上一批同）。⚠ 该档的 `Pw.det` **偶发红**（同一 env，**有转录可举证的 16 次观察里 1 次红（≈1/16）**；**是哪一项在抖未捕捉到**）—— 详见下节 §3.4 与 `E_det_probe_summary.txt` |
| **F** `F_real_vocab_full` | `…VOCAB_REAL=1 …STAGE=31` | 10 / **5**，rc=1（M111 批次；M111 修 U-A 之前是 10 / **3**） | `miss=16/16`、`emb bad=0`、`delta=0` ⇒ **与 F2 一致**（U-A 修好之后，① 的真实 id 真的传到 ② 了，于是「单窗口装不下 16 个真实 id」这条**已知未做**的多槽滑窗限制显形；不是回归） |
| **F2** `F2_real_vocab_body` | `…VOCAB_REAL=1 …STAGE=15` | 10 / **5**，rc=1（M111 批次） | `miss=16/16`（干净的「单窗口装不下真实 id」读数，`delta=0`） |
| **G** `runs_all` | `M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 … all`（接线默认关） | **2095 + 302 / 0**，rc=0 | 日志末行 `===== ALL PASS（checks=2095, guards=302, fails=0）=====`（§3.5；**注意**：M100 那批记的 2068+290 比它小 27+12，原因是那批不含 M98 的 attention-KV 判据，见下） |

**G 档口径更正（M111 发现）**：本目录 committed 的 `runs_all.log` 曾记 `checks=2068, guards=290`，
但那份转录出自 **M100 的分支**（提交 `3f62351`），而 M100 那个分支**不含** M98 的 attention-KV 判据
（`git merge-base --is-ancestor fc83496 3f62351` ⇒ 否；`fc83496`/`4829927` 只随 main 的合并进来）。
⇒ 在当前 main 的源码上，同一命令的计数**本来就更高**（本批 = M111 的读数，逐项与「改前同源二进制」
对照过，见 `M111_LANDING_PATH.md §3.6`）。**不要把 2068+290 当作当前 main 的基线**。

二进制（**当前批次 = M111 r2**）：`bin sha256: aa3652f120169b3e78e378586e932f9e86265b2aeed2086a81482fede51bc8c2`
（`binary_sha256.txt`），与本目录全部 `.log` 同批。**r2 只改落盘通路的核内次序**（三处改用
`docs/05 §6.1 ⓔ` 的写侧成对 drain release，见 `M111_LANDING_PATH.md §2.4`）——本目录七个接线档的
`Pw.*` 行与 r1 批次（`e5971447…`）**逐行相同**（`diff` 空），`runs=all` 亦同（§3.5）。
更早的批次：M111 r1 = `e5971447…`、M100 r2 = `1cf6ecad…`、`fbc132a3…`、`dacbc1e3f864…`。

### 3.1 主体（A / B）

- **A（②③④⑤ 独跑）：12 / 12 PASS，rc=0**（当前 tip；M111 批次 10 / 10）。`Pw.emb.T1 bad=0 miss=0`：设备取到的 16 行与 host
  从**同一注册窗口**读回的字节**逐字节相同**；`Pw.win.dev row_fail=0 miss=0 非空洞核=56
  见窗核=56`：设备**自己**把刚取到的行首与 host 直读分片文件得到的期望逐元素比过（M92 §6.2 的
  设备侧判据在这条路径上仍成立），且 56 个核确实在窗口模式下工作；`Pw.state.shift`、
  `Pw.planes.nonvac`、`Pw.det` 全绿。
- **B（①+② 全跑）：12 / 12 PASS，rc=0**（当前 tip；M111 批次 10 / 10 —— 修 U-A 之前是 7 / 10、3 条红）。`Pw.emb.T1 bad=0 miss=0/16`、
  `Pw.win.dev row_fail=0`、`Pw.det` PASS；`Pw.wired.delta = 5636/10240`，**与 A 档同值**。
  修前的读数（`emb bad=15/16`、`row_fail=15`、`Pw.det` 红）与根因见 §4 —— M111 已修

### 3.2 负向对照（「把被测对象弄坏 ⇒ 判据必须变红」，本队第五变体）

| 档 | 弄坏了什么 | 变红的判据 | 条数 |
|---|---|---|---|
| **C** | **接线关掉**（`M15_PLE_WIRE=0` ⇒ 挂载点空操作） | `Pw.wired.nonvac`、`Pw.emb.T1`、`Pw.win.dev`、`Pw.state.evolve`、`Pw.planes.nonvac` | **5 / 10**（M111 批次；见 §2 注） |
| **D** | **实参错位 1**：表基址接到 scratch slab（`MISWIRE=1`） | 同 C 的 5 条 | **5 / 10**（M111 批次） |
| **E** | **实参错位 2**：行基址偏 1（`MISWIRE=2`） | `Pw.emb.T1`、`Pw.win.dev`（偶发多一条 `Pw.det`，见 §3.4） | **2 / 10**（M111 批次） |

要点：C/D 下 `Pw.wired.delta = 0/10240`（PLE 一个元素都没改）⇒ 非空洞性判据不是恒真的；
E 只错行映射 ⇒ 只有两条**盯取行**的判据红（`state`/`planes` 仍绿，因为卷积那一段仍按错行内容
跑完）—— 这正说明这三条判据各自盯的是不同的量。

### 3.3 真实词表的诚实读数（F / F2）

用 config 的**原始** `ple_sz`/`ple_of`（不是缩减词表）时，一个真实 token 的 16 个 id 几乎必然
落在不同的表分片上，而本档的窗口是**一个连续行区间**（M92 §2.3）⇒ `miss` 应该等于 16。
F2（② 独跑，ids 由 host 参考灌入）给出这个干净读数；F（① 全跑）的 `miss` 被 U-A 的旧值污染，
故不作为读数。（多槽滑窗 = M92 §9 U-A，未做；见 `INTERFACE.md §7 U-E`。）

### 3.4 `E_miswire_winbase` 档 `Pw.det` 的 run-to-run 跳变（r2 复审唯一的 p2）

**现象**：上一批（二进制 `fbc132a3…`）的 committed `E_miswire_winbase.log` 里写着 `Pw.det FAIL`
⇒ `FAIL 3 条`；但同一二进制、同一 env 下，**r2 复审重跑 6 次全是 `Pw.det PASS`（`FAIL 2 条`）**，
r1 复审在更早的二进制（`dacbc1e3…`）上那次也是 2 红；复审把两边逐行比过：**除 `Pw.det` 那一行
（及其派生的 FAIL 计数与 rc）外全文逐字节相同**。
⇒ `Pw.det` 在这条**故意弄坏的负向对照档**上是 **run-to-run 跳变**。

**旧归因（错，已改）**：我原先在 §3 写的是"旧二进制 2 红、新二进制多出 `Pw.det`"。这站不住：
r2 相对 r1 的**唯一行为性**改动是删掉 `runs=="all"||runs=="m"` 分支里那三行 host 侧 `aclrtMemset`，
而 **`plewire` 档不进那个分支** ⇒ 两个二进制在这条路径上跑的是同一段代码。复审在**新**二进制上
6/6 得 2 红，直接证伪它。

**本轮做了什么**：
1. **重跑并替换**该 log：`E_miswire_winbase.log` 现在是与本批同二进制的**可复现读数（2 红）**；
   那份 3 红的转录**不再留在库里**（避免单读 log 的人看到不可复现的读数）。
2. **探针 6 次**（`E_det_probe_summary.txt`，设备段全程 `flock -w 900 /tmp/npu0.lock`）：
   6/6 `Pw.det PASS`、2 红 ⇒ **本次没捕捉到跳变**；**是哪一项在抖：未捕捉到**（当时 detail 只有
   "两次接线跑逐字节相同"，判不出来）——**不猜机制**。
3. **把 `Pw.det` 的 detail 改成点名差异项**（受判项与判据口径**不变**：仍是
   `ids/emb/hcp/stout/hmbad/failb` 六项逐字节；只是失败时打印 `不相等的项 = …`）⇒ 下次出现可直接定位。

**影响**：只在这条**故意弄坏**的档上（部分 id 落到 `row<0` 被当越窗跳过）；该档无论 2 红还是 3 红
都是红的、负向对照成立；主体档 A 与 D/F2 的 `Pw.det` 在作者多次跑与复审多次跑里都稳定 PASS。

### 3.5 零回归（G）

**实测（本批 = M111）：`===== ALL PASS（checks=2095, guards=302, fails=0）=====`，rc=0。**
同一命令、同一 env 在**改前源码**（`git stash` 回 main 源码重建，`bin sha256 1596284e…`）上也是
`checks=2095, guards=302, fails=0` ⇒ 两侧的 `判据分账` / `guard` 行逐字相同，本改动没有移动计数。
`runs=all` 走的是**接线默认关**的路径（`M15_PLE_WIRE` 默认 0）⇒ 挂载点逐字退化。

> ⚠ 与本文早先批次记的 `2068 + 290` 不同 —— 那是**基线不同**：那份转录出自 M100 的分支提交
> `3f62351`，它不含 M98 的 attention-KV 判据（`git merge-base --is-ancestor fc83496 3f62351` ⇒ 否）。
> 差的 `Kv 56→83`（+27）与 guards `其余 130→142`（+12）都在 attention-KV 段。
> 详见 `M111_LANDING_PATH.md §3.6`。

> ⚠ 这条读数**不是自动成立的**：中间形态（`#include "m15_ple.asc"` 的借用式）在 `runs=all` 上有
> **36 条 `M.moews.*` FAIL**（4 字节，落在 MoE 段 `WS_OFFSETS+32`），换成**机械抽取**的
> `m15_ple_wire.h` 后干净。基线对照与机理缩限见 §6 与 `INTERFACE.md §7 U-G`。

---

## 4. ①→② 的交接：一条**实跑暴露**的根因（U-A，本 mission 最重要的一条发现）

> **M111 已修（2026-09-27）**：本节描述的落点 `outG.SetValue` 已改成 **UB → MTE3 `DataCopy`**，
> B 档由 7/10 转 10/10（M111 时点的 10 条口径；M124 后判据 12 条、B 档 12/12，见 §2 计数口径注）。本节保留为**根因记录**：它是「本来就不应该用 scalar pipe 写 GM」
> 这条判断的实跑依据。修法与读数见 `M111_LANDING_PATH.md`。下面的「两条真修法」里 (a) 即 M111 选的路线。

**现象**（B 档）：① 算出的 16 个 id 读回来**逐位正确**（`Pw.ids.T1 bad=0/16`，判据读的是
launch 之后的 `S_IDS`），但 ② 取到的行里只有 1 个是对的。

**定位实验**（A 档）：把 ① 关掉、ids 由 host 参考直接 H2D 进 `S_IDS` ⇒ `Pw.emb.T1 bad=0`、
`Pw.det` 绿、`Pw.win.dev row_fail=0`。⇒ **根因在 ①→② 这一段交接**，与「窗口/取行/视图/注册」
无关（那三样在 A 档里全绿）。

**逐字证据**：同一批 id 下逐 item 打印（`M15_PLE_DBG=1`；临时诊断，未入库）——
`g=0`（由**写 ids 的那个核**处理）`got == want` 逐字节相同；`g=1..15`（由**别的核**处理）
`got` **16 个全相同且 ≠ want`。

**机理**：`m15_layer_kernel.h::M15L_PhaseBoundaryAiv` 的 set 挂 `PIPE_MTE3`（该注释自己写的是
「drain 本核写」）。① 的落点却是 `outG.SetValue`（**标量写**，走 scalar pipe），
**不在 MTE3 的 drain 覆盖内** ⇒ 除了「自己写自己读」的那个核，其余核在 barrier 之后仍读到旧值。
M92 的 `ple/REAL_TABLE.md §6.4` 已经记过同形状：*「实测 SetValue 的落点只对部分核可见…设备侧
计数一律走与其它输出同一条 DataCopy 通路」*。

**试过的修法（负结果，如实记录）**：在 ① 之后、边界之前补一条 `AscendC::PipeBarrier<PIPE_ALL>()`
（排空本核全部 pipe，含标量）。**实测无效**：`Pw.emb.T1` 仍是 `bad=15/16`、`Pw.det` 仍红，
与不加时逐项相同 ⇒ **这是写路径的问题，不是排序能修的**。该行**已撤掉**（既无效又会扰动
codegen，见 §6），只在 `m15_layer_kernel.h` 的挂载点注释里留下这条负结果。

**两条真修法（都不在本 mission scope）**：
- **(a)** 把 `IdsOneToken` 的落点改成 UB → GM 的 `DataCopy`（M92 §6.4 给设备侧计数就是这么改的）
  ⇒ 要改 `m15_ple.asc`；
- **(b)** ① 另起一次 kernel 启动（= `INTERFACE.md §7 U-C` 的 (D) 形态）⇒ 由 stream 顺序保证。

**这条同时回答了「挂载点设计是否成立」**：M65 的挂载点（两条已登记的相位边界）**承得住
MTE3 写的交接**（A 档证明：物化后的 `H'`、②③④⑤ 的产物都由 MTE3/DataCopy 落盘，跨核可见性
成立），**但承不住标量写的交接**（B 档）——「① 放在已登记的 PLE_IN 边界之前跑」这条论证的
**依据**与它的**边界**就在这里，见 `INTERFACE.md §4.1`。

---

## 5. 显式未完成项

见 `INTERFACE.md §7`（U-A ①→② 交接、U-B ③④⑤ 的 T3 数值对拍、U-C (D) 两段启动形态、
U-D 其余 PLE 层/prefill/aclgraph/性能、U-E 多槽滑窗、U-F `m15_layer_resources.h` 的两处登记、
U-G 借用形态为何让 `M.moews` 翻面、U-H `M.moews` 自身的脆弱性）。两条必须连带读的口径：

- **U-A —— M111 已修（「修法 (a)」）**：① 的 ids 原来用 `outG.SetValue`（scalar pipe）落盘，
  而 `M15L_PhaseBoundaryAiv` 的 set 挂 `PIPE_MTE3` ⇒ 覆盖不到标量落点，除「自己写自己读」的那个核外
  读到旧值（§4）。M111 把落点改成 UB → MTE3 `DataCopy`（连带重跑生成器 + M85 的判据 + M92 的设备判据，
  见 `M111_LANDING_PATH.md`）；B 档由 7/10 转 10/10（M111 时点的 10 条口径；现 12 条口径见 §2 注）。**未采用**的另一条路是 (b)「① 另起一次启动」
  （= U-C 的 (D) 形态），仍未做。与 `ple/REAL_TABLE.md §6.4` 同形（那一处 M111 也一并更正了判读）。
- **默认档口径（M111 后）**：`pleStageMask` **默认 31**（① 也跑）—— 修 U-A 之前**这一档是红的**
  （`Pw.emb.T1 bad=15/16`）；M111 之后这一档在 `run_wire_evidence.sh` 的 B 档下 **12/12 PASS**（M124 批次 `m124/M124_wire_B_kv.log`；M111 批次 10/10）。
  ⇒「先只接 1 层跑通」不再只在 `M15_PLE_STAGE=15` 下成立。
