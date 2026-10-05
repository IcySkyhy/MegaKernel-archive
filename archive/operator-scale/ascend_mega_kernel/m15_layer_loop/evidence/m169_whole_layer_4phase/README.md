# M169 —— 整层预填充：**四相位同时生效** + 两条段间链路（prolog + epilog）的新基线

> **性质与边界（先读）**
> - 本 mission 做的是**一次读数与基线的重设**：把已经合入的两条段间链路
>   （prolog = M151 S2/S3；epilog = M164 转位→S5→S6）与**四相位掩码**（H1/A/H2/B）同时打开，
>   取一份整层读数并重设基线。**attention 臂（B2/B3）未接**，`KIND_ATTN` 在 `wire=1` 下**响亮失败**
>   （见 §1）。**不得**读成本 mission 接上了 attention。
> - **激活仍由宿主合成**（hc 的 h/bo/ij 与 attention 的 q/k/v/g/β 在链路关的档里由 `H_PfHcFillHost` /
>   `H_PfGdnFillHost` 造）；**权重是真的**（checkpoint 切片）。
> - 两条既有红项**原样报告**：`m1_p12` 的进程内 `fails=3` 与 m=4097 的 `h2.blk` ulpMax（§5）。
>   本 mission 不预判因果、不抹平。
> - 行号基准 = 本分支 tip（内容锚点随行给出；行号漂移时按锚点重定位）。

## 0. 一句话

新基线 = `M15_PREFILL_PHASES=0xF`（H1|A|H2|B 四相位） **且** `M15_PREFILL_PROLOG=1` **且**
`M15_PREFILL_EPILOG=1`，m=1 与 m=4097 各一档：进程内判据 ALL PASS（checks=99, fails=0），
逐相位对拍全部复用现成参考（H1/H2=m20、相位 A=m23 NaN-aware、相位 B=m26、prolog=m151、
epilog=m164）。**m=1 各相位全绿**；**m=4097 除 `h2.blk`（ulpMax 9，沿用旧红）外全绿**。

## 1. 任务 1 —— 当前能一次跑齐的相位组合（带 文件:行）

四个开关 + attention 的失败口径，逐条落点如下（`m15_layer_loop/m15_layer_loop.asc` 记 `.asc`）：

| 开关 / 常量 | 字段 | env 解析 | 语义 |
| --- | --- | --- | --- |
| `M15_PREFILL_PHASES` | `.asc:166` | `.asc:284` | 相位掩码，位 = H1:1 / A:2 / H2:4 / B:8；0 也读成全开（与 kernel 内 `pfStageMask==0 → PF_STAGE_ALL` 对齐，`.asc:282-283`） |
| `M15_PREFILL_PROLOG` | `.asc:178` | `.asc:287` | 1 = 相位 A 的 q/k/v/g/β 由设备 S2+S3 产出；**并强制 H1 进掩码**（`.asc:4217`） |
| `M15_PREFILL_EPILOG` | `.asc:186` | `.asc:290` | 1 = 相位 A 出口 wsO + prolog z 经「转位→S5→S6」落成 H2 的 bo；**依赖 prolog**（`.asc:4374` 定义 `epilogOn = pfEpilog && prologOn`） |
| `M15_PREFILL_WIRE` | `.asc:161` | `.asc:279` | 1 = GDN 走真段体（真指针 + 段体真产出） |
| `M15_PREFILL_KIND` | `.asc:162` | `.asc:281` | 1=只 GDN / 2=只 attn / 3=both |

相位常量（`m15_layer_loop/m15_layer_kernel.h`）：`PF_STAGE_H1=1`（`:712`）、`PF_STAGE_A=2`（`:713`）、
`PF_STAGE_H2=4`（`:714`）、`PF_STAGE_B=8`（`:715`）、`PF_STAGE_ALL=15`（`:716`）。

**attention 未接时的行为（响亮失败，不给「开了却飘绿」）**：
`M15_PREFILL_WIRE=1` 且 `M15_PREFILL_KIND` 含 attn（2/3）时，`H_RunPrefill` 打
`[m15][FAIL] … 含 attention ⇒ 本轮不接（B2/B3 未落地）` 并 `H_CmpOk(C,false)`（`.asc:4800-4805`）。
⇒ 本 mission 的所有档都取 `M15_PREFILL_KIND=1`（只 GDN）。

**四相位全开时两条链路的 launch 序列**（host-orchestrated，`.asc:4375-4421`；`epilogOn` = `.asc:4374`）：

```
① PfLaunch(H1)                       .asc:4392    —— 产 BLK（prolog 的 z 源与 H2 的 hIn 都来自 H1）
② H_PfPrologRun（S2 in_proj + S3）   .asc:4396    —— 产 q/k/v/g/β 与 pfQkvzba
③ PfLaunch(aMask = rest & A)         .asc:4400    —— 相位 A 产 wsO（pfGdnODev）
④ H_PfEpilogRun（转位→S5→S6）        .asc:4404    —— wsO + z → hcAttnOut（H2 的 bo）
⑤ PfLaunch(h2bMask = rest & (H2|B))  .asc:4408    —— H2（combine 读新 bo）与相位 B（MoE）
```

`M_PREFILL = 4097`（`m15_layer_loop/m15_loop_layout.h:106`），即「m=1 + ctx=4097」这一档的 4097 口径。

**档矩阵**（`reproduce.sh` 逐档各自进一次锁；公共 env：`M15_LAYERS=1 M15_SKIP_WCHECK=1
M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1`）：

| 档 | 掩码 | prolog | epilog | m | 定位 |
| --- | --- | --- | --- | --- | --- |
| `m1_p15_links` | 0xF | 1 | 1 | 1 | **新基线（四相位 + 两条链路）** |
| `m4097_p15_links` | 0xF | 1 | 1 | 4097 | 新基线（满量级） |
| `m1_p15_nolinks` | 0xF | 0 | 0 | 1 | 四相位回归（链路关；旧基线） |
| `m4097_p15_nolinks` | 0xF | 0 | 0 | 4097 | 同上（满量级） |
| `m1_p12_iso` | 0xC | 0 | 0 | 1 | 既有红项 `m1_p12` 的隔离档（H1 关） |

## 2. 设备档读数（真权重、每档一次 `flock -w 300`、锁内 `npu-smi`、锁内 `timeout`）

逐档日志：`logs/run_*.log`（每份都含 `=== lock acquired … ===`、锁内 `npu-smi` 快照、逐字命令、
完整输出、`exit=`）。锁内 `npu-smi` 各档均为 `No running processes found in NPU 0`（未出现等锁）。
二进制 sha256 见 `binary_sha256.txt`（`d2cc02d9…`）。

| 档 | 进程内判据 | 日志 |
| --- | --- | --- |
| `m1_p15_links` | ALL PASS（checks=99, guards=9, fails=0） | `logs/run_m1_p15_links.log` |
| `m4097_p15_links` | ALL PASS（checks=99, guards=9, fails=0） | `logs/run_m4097_p15_links.log` |
| `m1_p15_nolinks` | ALL PASS（checks=44, guards=9, fails=0） | `logs/run_m1_p15_nolinks.log` |
| `m4097_p15_nolinks` | ALL PASS（checks=44, guards=9, fails=0） | `logs/run_m4097_p15_nolinks.log` |
| `m1_p12_iso` | FAILURES PRESENT（checks=23, guards=9, fails=3） | `logs/run_m1_p12_iso.log` |

四相位在**同一档**里各自留下真数值（进程内读数，取 `Pf.gdn` arm 首现）：

| 档 | 相位 A（o / ht） | H1（H'/BLK 毒值） | H2（H'/BLK 毒值） | 相位 B（y 毒值/零） | epilog 出口（hcAttnOut） |
| --- | --- | --- | --- | --- | --- |
| `m1_p15_links` | 非有限 0；max\|o\|=1.7200e-02；零 0/6144；max\|ht\|=1.2812e-01 | 0/10240、0/2560 | 0/10240、0/2560 | 0/2560、零 0 | 残渣 0/2560、非零 2560、非有限 0（输入 wsO 非有限 0） |
| `m4097_p15_links` | 非有限 0；max\|o\|=3.6037e-01；零 0/25171968；max\|ht\|=4.1437e+00 | 0/41953280、0/10488320 | 0/41953280、0/10488320 | 0/10488320、零 236 | 残渣 0/10488320、非有限 0（输入 wsO 非有限 0） |
| `m1_p15_nolinks` | 非有限 0；max\|o\|=2.1381e-02；零 0/6144；max\|ht\|=2.4824e-01 | 0/10240、0/2560 | 0/10240、0/2560 | 0/2560、零 0 | ——（未挂） |
| `m4097_p15_nolinks` | 非有限 0；max\|o\|=6.7415e-02；零 1/25171968；max\|ht\|=6.3126e-01 | 0/41953280、0/10488320 | 0/41953280、0/10488320 | 0/10488320、零 250 | ——（未挂） |

⇒ 四相位各写各的面、互不越界（未开的相位面保持 `0xCD` 毒值，`.asc:4535-4541` 的反面判据）。

## 3. 逐相位对拍（**全部复用现成参考，不自己造第二套数学**）

判据脚本与调用（锁外；失败以非零 rc 传播，`reproduce.sh` 已内建期望）：

| 相位 | 参考（现成） | 塔给的定位 |
| --- | --- | --- |
| H1 / H2 | `m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py <dump> Pf.gdn <m> 15`（参考 = `m20_hyperconn/check_ref.py::reference`） | m20/hc 参考 |
| 相位 A | `m23_gdn_prefill/check_ref.py --dir <dump> --allow-stale`（**已是 NaN-aware 版**：非有限面单列） | m23 |
| 相位 A 非有限分布 | `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py <dump>`（指数差参考 + NaN 掩码核对） | m151（M162 订正版） |
| 相位 B | `m26_moe_prefill/check_ref.py <dump>/m26_Pf.gdn` | m26 |
| prolog S2/S3 | `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_prolog.py <dump>` | m151 |
| epilog 链 | `m15_layer_loop/evidence/m164_epilog_mount/check_epilog.py <dump>` | m164 |

**结果表**（`logs/judge_*.log`；rc 由 `reproduce.sh` 核对）：

| 档 | H1/H2 | 相位 A（m23） | 相位 A NaN | 相位 B（m26） | prolog | epilog |
| --- | --- | --- | --- | --- | --- | --- |
| `m1_p15_links` | h1.hcp/h1.blk/h2.hcp/h2.blk 逐位 1.0000，ulpMax 0 ⇒ OK | o 超界 0/6144 max\|Δ\|=5.267e-09；ht 超界 0/786432 ⇒ PASS | 设备非有限 0；XOR=0 ⇒ PASS | 6/6 PASS（J1/J1b/J1c/J2/J3/J4） | S2/S3 PASS（S2 位型 0.999697，S3 五项 PASS） | 转位/z 逐位 0/6144；S5 worstRel=0；S6 tol-ratio 0.0788 ⇒ ALL PASS |
| `m4097_p15_links` | h1.hcp 逐位 1.0000 ulpMax 1；h1.blk 0.9949 ulpMax 1；**h2.blk 0.9785 ulpMax 9 ⇒ FAIL** | o 超界 0/25171968 max\|Δ\|=1.685e-07；ht 超界 0/786432 ⇒ PASS | 设备非有限 0；XOR=0 ⇒ PASS | 5/5 PASS | S2/S3 PASS（位型 0.999759） | 转位/z 逐位 0/25171968；S5 worstRel=0.00781；**S6 finite-bad 0/10488320，tol-ratio 0.493（不再是空过，见 §5.2）** |
| `m1_p15_nolinks` | 四张量逐位 1.0000 ⇒ OK | o 超界 0/6144 ⇒ PASS | 非有限 0 ⇒ PASS | 6/6 PASS | —— | —— |
| `m4097_p15_nolinks` | **h2.blk 0.9797 ulpMax 11 ⇒ FAIL**（同 M140 读数） | o 超界 0/25171968 ⇒ PASS | 非有限 0 ⇒ PASS | 6/6 PASS（含 J1） | —— | —— |
| `m1_p12_iso` | 无 H1（`phases=0xC`）⇒ 不判 | ——（A 关） | —— | 5/5 PASS | —— | —— |

## 4. 核心三问

### ① 两条链路是否都在四相位档里生效（**数据来自设备的见证，不只是 ALL PASS**）

**prolog**：同一二进制、同一掩码下切 `M15_PREFILL_PROLOG`。关档时宿主把 q/k/v/g/β H2D 上去
（`.asc:4239-4245` 的 `if (!prologOn)`），开档时这几面**只由设备 S3 写**。实测五个面的 sha256 前 16 位：

```
q    device=a35e7a4a69b3ac71   nolinks=d377afe486c7fcad   DIFFER
k    device=abf05cbebc299aa0   nolinks=0a396c406ad704cb   DIFFER
v    device=17b565f313774a27   nolinks=928590145617407a   DIFFER
g    device=467ed1a929cc3d69   nolinks=b85a642d14b6d88d   DIFFER
beta device=a8144d73f4b0468e   nolinks=a620c685e78ef542   DIFFER
```

且该档落有 S2/S3 的设备产出目录 `dumps_m1_p15_links/m151_s2_Pf.gdn/`、`…/m151_s3_Pf.gdn/`
（关档没有），`check_prolog.py` 用**独立 numpy** 从 H1.BLK + 真权重重算 S2、并 import m9 的
`check_ref.py` 判 S3 —— 两边 PASS。⇒ 相位 A 的输入确实来自设备 S2+S3。

**epilog**：开着时 H2 读的 `bo` = epilog 设备产出（`hcAttnOut`），关着时是宿主合成面。实测：

```
h2_bo  links=3de51f844f93205f   nolinks=f8398d001de4387e   DIFFER
h2_blk links=e0954ade5a73ed27   nolinks=e827954dc0893e45   DIFFER   （H2 吃了不同的 bo ⇒ 输出不同）
```

`check_epilog.py` 的判据 (a) 另给**出处**：设备 `hcAttnOut` 的 `0xCD16` 残渣 0，且与宿主合成 bo
**按面 DIFFER**（m=1 2560/2560、m=4097 10478831/10488320）。⇒ H2 的 bo 确实换成了设备产出。

**两条链路同档同时挂载**：`m1_p15_links` 一份 dump 目录里同时有 `m151_s2_Pf.gdn/`、`m151_s3_Pf.gdn/`
（prolog）与 `m164_Pf.gdn/`（epilog）三套设备产出，且 `reproduce.sh` 末段逐目录断言其存在。

### ② m=4097 的非有限是否真的消失（NaN-aware 计数）

`check_phaseA_nan.py`（指数差参考，M162 订正版）在 `m4097_p15_links` 与 `m4097_p15_nolinks` 上：

```
T=4097 o 元素 25171968  参考=指数差
  dev 非有限=0（NaN=0）
  NaN(ref_fp64)=0  NaN(ref_fp32)=0  （参考自洽要求 0）
  XOR(dev,fp64)=0  XOR(dev,fp32)=0
  对照：旧式(eg·ig) fp32 的 NaN = 8374272（links 档）
  ⇒ PASS
```

⇒ **m=4097 的相位 A `o`/`ht` 非有限计数 = 0**，与 M162/M163 登记的设备读数一致；而**旧式写法
在同一档会给出 8,374,272 个 NaN**（对照打印），说明这个 0 不是「判据看不见」而是设备真的不再溢出。
进程内读数同向：`o 非有限 0、零 0/25171968；ht 非有限 0`（`m4097_p15_links`）。
m23 的 NaN-aware 判据亦报 `devNaN=0 refNaN=0`（`logs/judge_m4097_p15_links_phaseA(m23).log`）。

### ③ 两条既有红项现在的状态（**不预判因果、不抹平**）

**(a) `m1_p12` 的进程内 `FAILURES PRESENT(fails=3)` —— 未变，仍红（且只在该隔离档）。**
本 mission 用同一二进制复取该档：`checks=23, guards=9, fails=3`（`.asc` 的判决项、逐字日志
`logs/run_m1_p12_iso.log`）。逐项定位：三条 = **每个 arm 各一条**，都是 `.asc:4530` 的
`H_CmpOk(C, a1pois * 100u < hpWords)` —— H2 的 `arena1`（H'）要求非毒值，而 p12 档 **H1 关**、
H2 的输入 `pfHcArena0` 本就是毒值 ⇒ 设备读数 `a1pois=10240/10240`（100% `0xCD`）⇒ 该条为假。
它在**四相位全开的档里不出现**：`m1_p15_links` / `m4097_p15_links` 的 H1 开 ⇒ `arena1` 非毒值
⇒ 进程内 ALL PASS。⇒ 该红项是 **p12 隔离档特有**，与两条链路、与四相位档都无关，本 mission
不改动它，只登记。

**(b) m=4097 的 `h2.blk` —— 仍红，且**移动了**（不是变绿）。**
同一二进制、同一 `check_full_layer.py`（参考 = m20）下：

| 档 | 逐位 | 良态逐位 | ulpMax | 结果 |
| --- | --- | --- | --- | --- |
| `m4097_p15_nolinks`（旧基线，链路关） | 0.9797 | 0.9903 | **11** | FAIL（与 M140 归档读数 `0.9797 / ulpMax 11` 逐字相同） |
| `m4097_p15_links`（新基线，链路开） | 0.9785 | 0.9900 | **9** | FAIL |

⇒ epilog 把 H2 的 `bo` 从宿主合成面换成设备产出后，`h2.blk` 的 ulpMax 由 11 移到 9、逐位率
0.9797→0.9785，**仍 > 2 的门限 ⇒ 仍红**。本 mission 只登记「移动」，不主张它与 epilog 的因果
（H2 的输入确实变了是事实，但「为什么 blk 仍超 2 ulp」未归因）。

**旁证（同一次移动带来的覆盖变化，不是红项）**：epilog 的 S6 数值判据在 m=4097 从**空过**变为
**有内容** —— M164 归档时 wsO 在 m>1 含非有限，S6 判据退化成 `finite-bad 0/0`；本次因 M162 让
相位 A 在 m=4097 有限，S6 变成 `finite-bad 0/10488320、worst(tol-ratio)=0.493`（< 1 ⇒ 在容差内，
但不再是空过）。见 `logs/judge_m4097_p15_links_epilog(m164).log`。

## 5. 负向对照（判据有牙）

四相位 + 两条链路的档里，`reproduce.sh` 的 `Pf.gdn_mut1`（只搅动送设备的 h0）与
`Pf.prolog_mut1/2/3`（S3 noShift / S3 noPad / S2 GEMM K−1）自动随档运行，判红由离线脚本做：

- `check_prolog.py` 认定 `Pf.prolog_mut3` 的 S2 **如预期变红**（m=1：T3 超界 12086；m=4097：46413615），
  `Pf.prolog_mut2` 的 S3 `qpad/kpad=FAIL`；`Pf.prolog_mut1` 的 S3 `state=FAIL`。
- `m23` 的 `--mutant 1` 路径由 dump 的 `m23_Pf.gdn_mut1_*` 提供（host 侧按 k 搅动输入 ⇒ 判据须变红）。
- 设备侧的 `[NEGATIVE] …如预期变红 ✓` 行见 `logs/judge_*_prolog(m151_S2_S3).log`。

## 6. 新基线 vs 旧基线的差异（写清）

| 维度 | 旧（M140 p15，两条链路关） | 新（本 mission：四相位 + prolog + epilog） |
| --- | --- | --- |
| 相位 A 输入 | 宿主合成 q/k/v/g/β | **设备 S2+S3 产出**（§4①） |
| 相位 A o/ht（m=4097） | 有限 0（宿主合成有界） | 有限 0（真权重 + M162 稳定化后也是 0） |
| H2 的 bo | 宿主合成面 | **epilog 设备产出**（§4①） |
| H2 的 blk（m=4097） | ulpMax 11 FAIL | **ulpMax 9 FAIL**（移动，仍红，§4③b） |
| epilog 的 S6（m=4097） | 空过（wsO 非有限） | **有内容**：0/10488320、tol-ratio 0.493 |
| 进程内判定项 | checks=44 | checks=99（多出 prolog/epilog 与负向臂） |
| m=1 各相位 | 全绿 | 全绿（H1/H2 逐位 1.0000；A/B/prolog/epilog PASS） |

## 7. 仍未通的（如实登记，不得读成「整层打通」）

1. **attention 臂（B2/B3）未接**：`KIND_ATTN` 在 `wire=1` 下响亮失败（`.asc:4800-4805`）。
2. **激活仍由宿主合成**（hc 的 h/bo/ij；链路关档的 q/k/v/g/β）—— 只有链路开档的 q/k/v/g/β 与
   epilog 出口是设备产出。
3. **内联未做**：prolog/epilog 是 **host-orchestrated 多 launch**（每层 4 次启动：H1 / prolog /
   A / epilog / H2|B，`.asc:4392-4408`），不是终态「每层一次 `__mix__(1,2)`、kernel 内四相位」；
   `pfQkvzba`（135 MB/层）等中间面走 GM 往返。这是 M151/M164 已登记的差距。
4. **PLE 未接进层 kernel**（PLE 走独立 target `m15_ple`，`runs=plewire` 故意不进 `runs=all`）。
5. **m=4097 的 `h2.blk` 仍红**（§4③b）；**相位 B 只对最后一块做 m26 数值对拍**（`m26_meta.txt`
   的 `m = 最后一块行数`），其余块只有非毒值/非全零级别判据。
6. **只到 `M15_LAYERS=1` 的单层**；48 层链、attention 层的 prefill 路径未验。

## 8. 复算（可传播失败）

```bash
# 全量：5 档设备 + 锁外离线对拍（失败以非零 rc 传播）
bash m15_layer_loop/evidence/m169_whole_layer_4phase/reproduce.sh
# 只跑离线对拍（用已有 dumps_*/；需先跑过设备档）
bash m15_layer_loop/evidence/m169_whole_layer_4phase/reproduce.sh --offline-only

# 构建（仓库根）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/build -j16 --target m15_layer_loop

# 单档设备（逐字；从仓库根；四相位 + 两条链路）
flock -w 300 /tmp/npu0.lock bash -c '{ npu-smi info; timeout 240 env \
  M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=15 \
  M15_PREFILL_PROLOG=1 M15_PREFILL_EPILOG=1 M15_PREFILL_M=1 M15_PREFILL_DUMPDIR=<dir> \
  ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt prefill; }'
```

`reproduce.sh` 纪律：每档各自进一次 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照写进该档
日志）、`timeout` 在锁内、一次进锁一条短命令；**等锁未拿到 ⇒ 该档记「未取得读数」并以非零退出**
（不写成「未复现」）。所有离线对拍在锁外做。本档 dump 未入库（`dumps_*/` 在 `.gitignore`；含
每臂 81 MB 的 S2 面等），需设备重跑生成；入库的是逐档日志 + 锁内 `npu-smi` + 逐字命令 + 本脚本。

**期望值语义（为什么脚本会在「有红项」时仍报 ALL PASS）**：`reproduce.sh` 对每条判据都带一个
期望 rc，并与实测比对。唯一两条期望非零的是 m=4097 的 h1/h2 判据（`judge_red_h2blk`）——它期望
**rc=1 且 `h2.blk` 的 ulpMax 恰为登记值 9（links）/ 11（nolinks）**；`ulpMax` 或 rc 一旦偏离，
脚本即打印 ✗ 并以非零退出（把「红项移动/变绿」也当作基线漂移传播）。其余判据期望 rc=0。
本机复算末尾行（`bash …/reproduce.sh --offline-only`）：`==== reproduce 汇总：ALL PASS ====`、rc=0。

## 9. 纪律自检

- **判据非空洞**：每条相位判据都带「把被测对象弄坏必变红」的对照（§5）；m=4097 的 S6 空过问题在
  本档已消除并如实写明（§4③b）。
- **参考独立于实现**：H1/H2 用 m20 的 `reference`；相位 A 用 m23 numpy fp64；相位 B 用 m26；
  prolog 的 S3 直接 import m9 的 `check_ref.py`；epilog 用 m164 的 numpy 链 —— 都由现成脚本承担。
- **不预判因果**：§4③ 只登记两项红项的状态与移动，未给机理结论。
- **禁用断言字面**：对本目录新增文本（`README.md` / `reproduce.sh` / `.gitignore`）扫塔口径的
  六条绝对断言字面，实测计数 0（逐字 pattern 与命中范围写在 review-request 正文里；此处不逐字
  复写清单，以免自命中）。
- **只改 scope 内文件**：`m15_layer_loop/evidence/m169_whole_layer_4phase/**`（新增）+
  `.gitignore`；**未改** `.asc`（本次无需源码改动即可跑齐）、**未碰** `m15_layer_resources.h` /
  `m15_layer_kernel.h`（在飞的 M168）。
- **行号基准**：本 README 的 `文件:行` 以本分支 tip 为准，并随内容锚点给出。
