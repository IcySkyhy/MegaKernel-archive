# M162 / M163 —— GDN chunk 扫描的**设备侧数值稳定化**

> 一句话：把 `m15_layer_loop/m15_gdn_prefill.h` 里 Γs/Γi/KT′ 三处的
> `exp(ĝ)·exp(−ĝ)`（`0·inf=NaN`）改成**指数差** `exp(ĝ_i−ĝ_j)` / `exp(ĝL−ĝ[t])`；设备实测
> m=4097 的 `o` 非有限 **8,374,400 → 0**、`ht` **262,144 → 0**，且与稳定化 fp64 参考最大差
> `o` 1.7e-7 / `ht` 6.6e-7。另（塔扩宽 scope）把 M151 的 `check_phaseA_nan.py` 参考也从旧式
> `eg·ig` 订正为指数差、旧式留作 `--naive-ref` 对照（§4b）。**只修这一处数值写法，不是整层打通**（见 §7）。
>
> 本文的 `文件:行` 基准 = **本任务提交的 tip**（复算：`git log -1 --format=%H -- m15_layer_loop/m15_gdn_prefill.h`）。
> 修前（base `0cd7fe5`）对应行：`m15_gdn_prefill.h:959,963,969,1047` 与 Γ 体 `:191-206`/`:208-223`。

---

## 1. 修了哪几处（文件:行）

被测路径 = `M15_PREFILL_PROLOG=1` 下的相位 A（`M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=3`）
⇒ `M15GP::GdnPrefillAiv::RunChunk` 的 chunk 扫描。修的是它内部 4 个调用点 + 3 个体：

| 位置（tip 行号） | 修前 | 修后 |
| --- | --- | --- |
| `m15_gdn_prefill.h:175` `CumSumExpVF` | 产 `gc=ĝ`、`eg=exp(ĝ)`、`ig=exp(−ĝ)` | 只产 `gc`、`eg`；**删掉 `ig`**（不再物化） |
| `m15_gdn_prefill.h:202` `GammaStrictVF`（Γs） | `out = brc(eg_r) · ig`（`Mul<ZEROING>`） | `out = exp(brc(ĝ_r) − ĝ_lane)`（`Sub` + `Exp<ZEROING>`） |
| `m15_gdn_prefill.h:232` `GammaInclVF`（Γi） | 同上 | 同上（`CMPMODE::LE`） |
| `m15_gdn_prefill.h:262` `BrcDiffExpVF`（**新增**） | （原用 `BrcMulVecVF(scKT, scIG, scL)` = `egL·ig[t]`） | `scKT[t] = exp(ĝL − ĝ[t])` |
| `m15_gdn_prefill.h:1009`（调用） | `CumSumExpVF(scG, scGC, scEG, scIG)` | `CumSumExpVF(scG, scGC, scEG)` |
| `m15_gdn_prefill.h:1013`（调用） | `BrcMulVecVF(scKT, scIG, scL)` | `BrcDiffExpVF(scKT, scGC, cv - 1)` |
| `m15_gdn_prefill.h:1019`（调用） | `GammaStrictVF(scEG, scIG, aUb)` | `GammaStrictVF(scGC, aUb)` |
| `m15_gdn_prefill.h:1097`（调用） | `GammaInclVF(scEG, scIG, aUb)` | `GammaInclVF(scGC, aUb)` |
| `m15_gdn_prefill.h:966` | 局部量 `scIG` | 删除 |

`eg=exp(ĝ)` 仍物化，只用于**单因子**处（`scSE = scale·eg`、`scSG = β⊙eg`、`scL = egL`）——这些是
`0` 的正确下溢，不与 `inf` 相乘，无 `0·inf`。

**为何是这四处**：`Γs[i,j]=exp(ĝ_i−ĝ_j)`（`j<i`）与 `Γi`、`KT'[t]=exp(ĝL−ĝ[t])` 在数学上都等于
`exp(a)·exp(−b)`，但后者分别物化 `exp(a)` 与 `exp(−b)`；当 chunk 内 `|ĝ|>88.72`（fp32 `exp` 上溢阈值）
时 `eg→0`、`ig→inf` ⇒ `0·inf=NaN`，而正确因子 `exp(ĝ_i−ĝ_j)≤1` **恒有限**。
官方 FLA 正是用指数差：`vllm/…/flash_linear_attention/ops/chunk_o.py:119-120`
（`b_A = b_A * exp(b_g[:,None]-b_g[None,:])`）、`chunk_delta_h.py:216-221`（`exp2(b_g_last-b_g)`）。

## 2. 代价（精度 / 吞吐 / 约束）

- **人类约束不变**：核内仍只用 BufferID（`BufAcquire/BufRelease`）、跨核仍只有 `CrossCoreSet/WaitFlag`
  （mode 2）、**未引入 `set_flag/wait_flag`**；这三处仍是 AIV 的 VF（`__simd_vf__`），无 scalar 计算、
  无 mmad 变动。
- **VF 指令数基本持平**：Γ 体每行 `-1 Mul +1 Sub`（原 1 次 `Mul<ZEROING>`；新 1 次 `Sub`+1 次
  `Exp<ZEROING>`，因为 `Exp` 取代了行向的 `ig` 复用）；`CumSumExpVF` **少** 1 次 `Muls`+1 次 `Exp`+1 次
  `StoreAlign`（不再写 `ig`）；`BrcDiffExpVF` 比 `BrcMulVecVF` **多** 1 次 `Sub`。合计 ≈ 持平。
- **精度**：对大 `|ĝ|` 更准（去掉 `0·inf`）；同一 chunk 内差值 `ĝ_i−ĝ_j` 的相对误差由 `ĝ` 的 fp32
  cumsum 决定，与官方同法。实测与稳定化 fp64 参考的最大差见 §4（远低于 rtol=2e-3）。
- **资源**：`GP_UB_SC_IG` 槽位保留但不再读写（该常量在 `m15_gdn_resources.h`，**不在本 mission scope**，
  未改；其旧注释 `egL·ig` 已 stale —— 已随 finding 报塔）。

## 3. 设备读数（修前 / 修后 / 负向对照）

每档各自 `flock -w 300 /tmp/npu0.lock`、进锁先 `npu-smi`（快照在该档日志头部）、`timeout` 在锁内。
真权重（`weights_manifest.txt`），`M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1
M15_PREFILL_PHASES=3 M15_PREFILL_PROLOG=1`，逐字命令见 `reproduce.sh`。

| 档 | 日志 | m | `o` 非有限 | `ht` 非有限 | `o` 零元素 | max\|o\| | exit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base（修前） | `logs/base_m1.log` | 1 | 0 | 0 | 0/6144 | 1.7200e-02 | 0 |
| base（修前） | `logs/base_m4097.log` | 4097 | **8,374,400** | **262,144** | 384/25,171,968 | 3.6037e-01 | 0 |
| fix（修后） | `logs/fix_m1.log` | 1 | 0 | 0 | 0/6144 | 1.7200e-02 | 0 |
| fix（修后） | `logs/fix_m4097.log` | 4097 | **0** | **0** | 0/25,171,968 | 3.6037e-01 | 0 |
| nc1（naive） | `logs/nc1_m4097.log` | 4097 | 8,088,064 | 262,144 | 1152/25,171,968 | 3.6037e-01 | 0 |
| nc2（错下标） | `logs/nc2_m4097.log` | 4097 | 938,752 | 32,768 | 128/25,171,968 | 2.3195e+38 | 0 |

- **m=1 修前/修后**：`max|o|`（1.7200e-02）、`max|ht|`（1.2812e-01）**标量逐位相等**，但**张量非逐位** ——
  实测 `o` **688/6144**、`ht` **78,145/786,432** 个元素不同，`max|Δ|` o=9.3e-10 / ht=7.5e-9（≤256/8192 ulp），
  量级不变（脚本 `check_m1_bitdiff.py`，读数 `logs/m1_bitdiff.log`）。差异来自指数差的舍入路径不同
  （m=1 的 KT′ 列与 Γi 对角：旧式 `exp(g)·exp(−g)` vs 新式 `exp(0)=1`）。
  `m=4097` 修后 `max|o|` 与修前相同（3.6037e-01）⇒ 本次改动**没有改变量级**，只是把 NaN 消掉。
- **注意**：设备 host 自带的判据仍打印 `===== ALL PASS（checks=59）=====`（六档全部如此）——它的
  `o 非有限` 只打印、不进判定项，且 `badO<size` 门近乎空转。这正是 M160/M151 指出的**检测缺口**，
  也是本 mission 另配 NaN-aware 判据（§4）的原因。

## 4. NaN-aware 判据（`check_stab_nan.py`）

**参考用的是哪一份**（写清）：
- **权威参考 = `check_stab_nan.py::ref_head_stable`**：fp64，逐句同 m23 `check_ref.py::ref_head` 的公式，
  但 Γ/KT′ 用**指数差**（`dgc = gc[:,:,None]-gc[:,None,:]`、`exp(ĝL−ĝ)`）。它自身非有限计数为 0。
- **机理参考 = `ref_head_naive`**：参数化 fp64/fp32，逐句复刻修前 `eg·ig` 写法，只用来复现溢出与给出
  "修前应有的 NaN 数"，**不是判据**。
- **m23 既有判据**（`m23_gdn_prefill/check_ref.py:147-149`，**本分支 base `0cd7fe5` 版**）仍是 naive 且
  对 NaN 静默 ⇒ 与本判据的差异就在"NaN 单独计数"这一层。实测：对 base 与 fix 两份 dump，m23 判据**都**报
  `o 超界 0 … PASS`（`logs/m23judge_base_m4097.log`、`logs/m23judge_fix_m4097.log`）⇒ **它分不出
  8,374,400-NaN 与 0-NaN**。
  **版本注记（r1 复审 F1）**：当前 main 已含 M157 的 NaN-aware 判据（`23f997e`）与 M165 的指数差参考
  （`600b05e`）⇒ "未合入 main / NaN 盲"**只对 base `0cd7fe5` 成立**；`m23judge_*.log` 用的就是该 base 版，
  故"检测缺口"是本分支 base 上的实证。

判据（可传播失败）：① 设备 `o`/`ht` 非有限计数必须为 0；② 稳定化参考自身非有限计数为 0；
③ `o[0,cv)` 行与 `ht` 对稳定化 fp64 参考逐元素 `|Δ| ≤ rtol·max(1,|ref|)`（rtol=2e-3，口径同 m23）。

| 档 | dev `o`/`ht` NaN | ref_stab NaN | ref_naive(fp32) `o` NaN | XOR(dev,ref_stab) | max\|Δ\| o / ht | 判据 |
| --- | --- | --- | --- | --- | --- | --- |
| base m=1 （`judge_base_m1`） | 0 / 0 | 0 | 0 | 0 | 5.3e-9 / 1.2e-8 | PASS |
| base m=4097（`judge_base_m4097`） | 8,374,400 / 262,144 | 0 | 8,374,272 | 8,374,400 | 7.2e-4 / 5.5e-7 | **FAIL** |
| fix m=1 （`judge_fix_m1`） | 0 / 0 | 0 | 0 | 0 | 5.3e-9 / 1.9e-8 | PASS |
| fix m=4097（`judge_fix_m4097`） | 0 / 0 | 0 | 8,374,272 | 0 | 1.7e-7 / 6.6e-7 | PASS |
| nc1 m=4097（`judge_nc1_m4097`） | 8,088,064 / 262,144 | 0 | 8,374,272 | 8,088,064 | 1.6e-3 / 6.6e-7 | **FAIL** |
| nc2 m=4097（`judge_nc2_m4097`） | 925,952 / 32,768 | 0 | 8,374,272 | 925,952 | 2.3e+38 / 7.6e+0 | **FAIL** |

- `fix m=4097` 的 `ref_naive(fp32) o NaN = 8,374,272` 与 M160 survey 的读数一致；设备修后在这些位置
  **一个 NaN 都没有**（XOR=0），即设备现在给出的是正确因子 `exp(ĝ_i−ĝ_j)`。
- base m=4097 的 `XOR(NaN(dev), NaN(naive_fp32)) = 128`，与 M151 复审读数逐字相同（fp32 与设备
  在 128 个元素上的溢出范围之差）。
- 判据**非空洞**的证据：base m=4097 → FAIL；nc1 / nc2 → FAIL（§5）。把被测对象弄坏，判据变红。

## 4b. M151 `check_phaseA_nan.py` 的订正（塔扩宽 scope）

塔在 M162 进行中把 `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py` 纳入 scope，
理由（塔的 note）：该文件 `:59` 复制了**旧式（naive）参考**，其"设备 NaN = fp32/fp64 精度产物"的结论
已被 M160/M165 推翻。本 mission 已订正：

- 默认参考从 naive `eg·ig` 改为**指数差**（`dgc=ĝ_i−ĝ_j`、`S = egL·S + kᵀ(d⊙exp(ĝL−ĝ))`），并加**参考自洽守卫**
  （所选参考自身非有限计数必须为 0）；
- 旧式保留为**可切换对照** `--naive-ref`，且无论用哪种都另打印"旧式 fp32 的 NaN 数"作对照；
- 判据更新为：参考自洽 + `T=1` 零非有限 + `T>1` 的 `XOR(NaN(dev), NaN(参考 fp32)) ≤ tot/1000`。

实测退出码（`reproduce.sh` 已把它列入期望）：

| 调用 | rc | 读数（`logs/m151nan_*.log`） |
| --- | --- | --- |
| `check_phaseA_nan.py dumps_fix_m1` | 0 PASS | 参考指数差 NaN=0；设备 0 非有限 |
| `check_phaseA_nan.py dumps_fix_m4097` | 0 PASS | 参考 NaN=0；设备 0 非有限；旧式 fp32 NaN=8,374,272（对照） |
| `check_phaseA_nan.py dumps_base_m4097` | 1 FAIL | 设备非有限 8,374,400；XOR(dev,指数差参考)=8,374,400 |
| `check_phaseA_nan.py dumps_fix_m4097 --naive-ref` | 1 FAIL | 参考自身 NaN：fp64=3,670,912、fp32=8,374,272（与 M160 的 naive fp64 读数一致） |

⇒ 订正后该脚本在**修前 dump 上变红**（旧版对同一 dump 报 PASS）——这正是塔要的"结论被推翻"的收口。
**局限（写窄）**：M151 的 `README.md` 里引用该脚本结论的**散文段**不在本 mission scope，未改；如需一并
订正，请塔另派（或允许扩 scope）。M151 的 `reproduce.sh` 调本脚本时用默认参数，故其 `--offline-only`
在**旧归档 dump** 上现在会 FAIL（正确反映旧 dump 是非有限的）。

## 5. 负向对照（两个，均能变红）

两个都是**编译期宏**（写在 `m15_gdn_prefill.h`，与该文件既有的 `M15GP_J1_ONLY` 等诊断宏同体例），
用 `ASCFLAGS` 注入、单独 build dir 构建，**不改 `m15_layer_loop.asc`、不改 `CMakeLists.txt`**：

| 对照 | 宏 | 语义 | m=4097 设备读数 | 判据 |
| --- | --- | --- | --- | --- |
| nc1 | `-DM15GP_STAB_NAIVE=1` | Γ 退回 `exp(ĝ_r)·exp(−ĝ_lane)` 旧乘法 | `o` 非有限 8,088,064、`ht` 262,144 | FAIL |
| nc2 | `-DM15GP_STAB_BADIDX=1` | Γ/KT′ 的指数差下标**错一位** | `o` 非有限 938,752、`ht` 32,768，`o` 有限元素超界 4,133,512/25,171,968、max\|Δ\|=2.3e+38 | FAIL |

- nc1 只回退 Γ 两体（`scKT` 仍是稳定形态），故其 NaN 数（8,088,064）与修前 base（8,374,400）不完全相等；
  两者都远超判据阈值。
- **如实写**：两个对照在 **m=1** 都 **PASS**（`logs/nc1_m1.log`、`logs/nc2_m1.log`）——nc1 在 m=1 时
  `|g|≤84.18<88.72`（M151 归档档）不溢出；nc2 在 m=1 的错一位恰好落到 padding 0 上，数值与正确一致。
  即：这两个对照的判别力**只在 m>1**（本 mission 用 m=4097）。

## 6. 同类写法全仓清单（真会溢出 vs 形式相似但安全）

**grep 命令（逐字，覆盖范围写清）**：
```
grep -rn "Exp<"                        --include=*.h --include=*.asc .   # templated Reg Exp
grep -rn "[^a-zA-Z_]Exp("              --include=*.h --include=*.asc .   # 非 templated Reg Exp
grep -rn "exp("                        --include=*.py .                  # numpy/torch
grep -rn "eg\[.*\].*ig\|eg.*\*.*ig\|egB.*ig\|egL.*ig" --include=*.py --include=*.asc --include=*.h .
```

**A. 真会溢出（同款 `exp(ĝ)·exp(−ĝ)` 配对；`|ĝ|>88.7` 时 `0·inf=NaN`）**

| 文件:行 | 形态 | 本 mission 是否改 |
| --- | --- | --- |
| `m15_layer_loop/m15_gdn_prefill.h:175,202,232,1009,1013,1019,1097` | 设备核，本次被测 | **已改** |
| `m18_gdn_prefill/m18_gdn_prefill.asc:174-198`（产 ig）、`:201-216`/`:218-233`（Γ）、`:682-690`（`d'=d⊙ig` 后 `S=egL·(S+kᵀd')`） | 设备核（另一份实现） | 否（不在 scope）→ finding |
| `m23_gdn_prefill/check_ref.py:64,66-67,77` | numpy 参考（判据用） | 否（参考侧另派）→ finding |
| `m18_gdn_prefill/check_ref.py:197-199,211,233-235,245` | m18 的 numpy 参考 | 否（不在 scope）→ finding |
| `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py:59-78` | M151 的 NaN 核对脚本自带 naive 参考 | **已改**（塔扩宽 scope：默认改指数差、旧式留 `--naive-ref` 对照，见 §4b） |

**B. 形式上相似但安全（只有单一 `exp` 因子，或分母/掩码形式，无 `0·inf` 配对）**

| 文件:行（代表） | 形态 | 为何安全 |
| --- | --- | --- |
| `m4_gdn_recurrent/m4_gdn_recurrent.asc:119-124`；`m15_layer_loop/m15_gdn_layer.h:985-993`；`m14_gdn_layer/m14_gdn_layer.asc:687-696` | decode 递推 `s = e^g·S` | 只有 `exp(g)`（`g≤0`）；`0·有限量=0`，无 `inf` |
| `m15_layer_loop/m15_prefill_prolog.h:151-161,525-535` | SiLU `acc/(1+exp(−acc))` | 分母形式；`exp→inf` 时 `有限/inf=0`，非 NaN |
| `m15_layer_loop/m15_prefill_prolog.h:182-195,551-564` | softplus `log1p(e^x)` + `Select`；`exp(A_log)` | 大 `x` 的 `inf` 被 `Select(x≤20)` 丢弃；`A_log` 有限（`exp(A_log)∈[0.028,158]`） |
| `m3_grouped_gemm/m3_grouped_gemm.asc:347-390`；`m15_attn_oproj.h:130`；`m15_hc_layer.h:210`；`m20_hyperconn/m20_hyperconn.asc:232`；`m13/m17/m22/m12/m28` 的 softmax/sigmoid | SiLU / σ / softmax | 分母或 `Sub(s,max)` 形式 |
| `m10_attn_decode/m10_attn_decode.asc:605-613`；`m15_layer_loop/m15_attn_core.h:743-751`；`m15_attn_fa_core.h:525-532` | softmax `Sub(s,max)` 后 `Exp` | 指数 ≤0 |

**C. stale 注释（不在 scope）**：`m15_gdn_resources.h:382`（`GP_UB_SC_KT … // egL·ig`）、
`:412`（`GP_SC_KT … // kᵀ·egL·ig`）——已随 finding 报塔。

## 7. 写窄 / 未做（不得读成"整层打通"）

- 本次**只改** `m15_gdn_prefill.h` 的 4 个调用点 + 3 个体（§1），另**订正 M151 `check_phaseA_nan.py`**
  的参考口径（§4b，塔扩宽 scope）；**实测 m 范围 = m=1 与 m=4097**
  （真权重、`PHASES=H1|A`、`PROLOG=1`）。
- `m15_prefill_prolog.h` 在本 mission scope 内，但复核后**无需改**：它的 `exp` 全是单因子/分母形式
  （§6-B），无 `exp(ĝ)·exp(−ĝ)` 配对。
- **未做**：`m18` 设备核与 `m18`/`m23` 的 numpy 参考（同款写法，**不在 scope**，已 finding 报塔）；
  H2/B 相位；attention（B2/B3）；epilog。（参考侧的 NaN-aware 判据 M157 已在 main 落地，非本 mission 所做。）
- 资源槽 `GP_UB_SC_IG` 保留未用（`m15_gdn_resources.h` 不在 scope）。

## 8. 复算

```
bash m15_layer_loop/evidence/m163_stab/reproduce.sh
```

- 脚本自行：`source set_env.sh`；构建 `fix`（工作树）；构建 `base`（`git archive 0cd7fe5…` 到 `/tmp`，
  **不动工作树**）；用 `ASCFLAGS` 构建 `nc1`/`nc2`（`/tmp` build dir）；逐档 `flock -w 300` + 锁内
  `npu-smi` 快照 + `timeout`；对每档跑 `check_stab_nan.py` 并**核对期望 rc**（base m=4097 / nc1 / nc2
  期望 FAIL；base m=1 / fix m=1 / fix m=4097 期望 PASS）；另跑 `check_m1_bitdiff.py`（m=1 非逐位对照，
  PASS）与 M151 订正后的 `check_phaseA_nan.py`（fix 期望 PASS / base 与 `--naive-ref` 期望 FAIL）；
  最后跑 m23 既有判据留存"检测缺口"证据。
- **期望全满足时 rc=0**；任一档等锁未取得读数记「未取得读数」、任一判据 rc 不符期望 ⇒ rc≠0 传播失败。
- 实测：`==== reproduce 汇总：ALL EXPECTATIONS MET ====`（rc=0）。
- 环境：`M163_TMP`（默认 `/tmp/m163_stab`）可改中转目录；单进程 RSS 峰值约 15.6 GiB（cgroup 32 GiB）。

## 9. 纪律自查

- **禁用词**：对本次新增文本（`README.md` / `reproduce.sh` / `check_stab_nan.py` / `check_m1_bitdiff.py`）
  扫 mission 纪律里的绝对化措辞清单，未命中；为避免自命中，此处不逐字复写清单（照 M151 的先例）。
- **复现命令不钉会移动的 ref**：`reproduce.sh` 里的 base 钉**不可变 commit** `0cd7fe5…`；本文行号
  基准声明为"本任务提交的 tip"并给出复算命令。
- **判据非空洞**：§4/§5 给出 base 与两个负向对照的实测 FAIL（判据变红），fix 为 PASS。
- **参考独立于实现**：`ref_head_stable` 是 numpy fp64 独立复算，不 import 被测设备代码。
- **r1 整改（对 tip e088afb 的复审）**：F1 已收窄 m23/M157 版本表述（§4）与 m=1 "逐位" 口径（§3，
  差异计数见 `logs/m1_bitdiff.log`）；F2 已 file finding（M151 `reproduce.sh` 期望与订正脚本不一致）；
  F3 已删死代码 `BrcMulVecVF`。设备代码未改。
