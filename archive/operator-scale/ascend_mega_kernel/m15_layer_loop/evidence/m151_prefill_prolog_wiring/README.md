# M151 —— 预填充 prolog 链路（S2 in_proj + S3 conv1d/l2norm/gating）挂进相位 A

> **覆盖范围（不得读大）**：本 mission **只接 prolog 链路** ——
> `H1.BLK → S2 in_proj → S3 conv1d/l2norm/gating → 相位 A(q/k/v/g/β)`。
> **epilog**（`wsO → hcAttnOut`，S5 RMSNormGated + S6 out_proj + S7 残差/门控）**未接**；
> **attention B2/B3 未接**。**不得**描述成「整层四相位打通」。
> 原有红项（相位 B m=1、m=4097 `h2.blk` ulpMax 11）**原样保留**，本 mission 未动。

## 0. 一句话

相位 A 的 q/k/v/g/β 从「宿主合成」改成「设备真产出」：先跑 H1 产 BLK，再跑 S2 in_proj
（`m11 bf16_gemm` donor 的 m15 lift）产 `qkvzba`，再跑 S3 prolog（m9 `gdn_prolog_mt_kernel`
的逐字抽取）产 q/k/v/g/β 与 conv_state，最后相位 A 读设备产出。宿主合成路径保留为开关。

## 1. 数据流与接线契约

逐条 `文件:行` 契约见 [`CONTRACT.md`](CONTRACT.md)。要点：

| 环节 | 契约 |
| ---- | ---- |
| BLK | H1 输出 `pfHcBlk0` `[m,HID=2560]` bf16，行距 5120 B（`m15_layer_kernel.h:270-271`、`m15_hc_prefill.h:114`） |
| S2 in_proj | A=BLK `[m,2560]`、B=wIn `[16480,2560]` bf16、C=qkvzba `[m,16480]` bf16；cube |
| S3 prolog | x=qkvzba、conv_state `[3,10240]` bf16、w `[4,10240]`、bias `[10240]`、a_log/dt_bias fp32 `[64]`；AIV |
| 相位 A 输入 | q/k `[16,m+64,128]`、v `[48,m,128]`、g/β `[48,align8(m)]` fp32（`m15_layer_kernel.h:258-262`, `:721`） |
| 权重常驻 | `H_W(C,L,W_IN_OFF/W_CONV_OFF/W_CONVB_OFF/W_ALOG_OFF/W_DTB_OFF)`（`m15_layer_loop.asc:871`；`m15_loop_layout.h:135-145`） |

**挂载形态**：S2/S3 是**独立 `__global__`**（`m15_layer_loop/m15_prefill_prolog.h` 的
`m15_pf_inproj_gemm_kernel` / `m15_pf_gdn_prolog_kernel`），由 host 在 `H_PfGdnWired` 里
按「起 H1 → 起 S2 → 起 S3 → 起其余相位」串行起动（同一条 stream）。**不引入 set_flag/wait_flag、
不新增跨核 flagId**；两段内部各自是 BufferID 同步（S2 = `OpAcq/OpRls`；S3 = m9 的
`GetBufInternal/RlsBufInternal`）。**权重不搬**：直接传 decode 权重区 `C.wDev` 的常驻 GM 指针。

**S2 的复用来源**：用 `M15OP::OProjGemm<2560,16480>`（`m15_attn_oproj.h:264`）—— 它本身就是
`m11_bf16_gemm.asc` 的 `Bf16Gemm` donor 的 lift（该文件顶部有"抄的账"）。选它而不是
`Cube::Bf16Gemm` 的原因：它自带负向对照 `GEMM_MODE_KMINUS1`（`:63`）。

## 2. 改了哪些文件

| 文件 | 改动 |
| ---- | ---- |
| `m15_layer_loop/m15_prefill_prolog.h` | **新增**。M15PM = m9 `m9_gdn_prolog.asc` 第 53..842 行的逐字抽取（device 段体 `MtGdnProlog`）；另含 S2/S3 两个入口薄壳。 |
| `m15_layer_loop/m15_layer_loop.asc` | 新增 include；`M15_PREFILL_PROLOG[_MUT|_SCALE]` 选项；`pfQkvzbaDev`/`pfConvStateDev` 两平面与分配；`H_PfPrologRun`/`H_PfPrologDump`；`H_PfGdnWired` 三档起动 + 相位 A 输入改设备产出；prolog 负向对照档。 |
| `m15_layer_loop/m15_layer_resources.h` | §3c-bis：prolog 两个 kernel 的 UB/L1 峰值与 GM 平面台账登记 + 预算 static_assert；钉死 `PfPeakUnfilled()==3` 不变。 |
| `m15_layer_loop/evidence/m151_prefill_prolog_wiring/**` | 本 evidence（契约、复算脚本、判据脚本、逐档日志）。 |

`m15_layer_loop/m15_layer_kernel.h` **未改**（相位 A 的输入契约不变；prolog 由 host 在入口之外起）。

## 3. 抽取等价性（逐字节见证）

`verify_lift.sh`：`m15_prefill_prolog.h` 的 M15PM 命名空间体 == `m9_gdn_prolog.asc` 第 53..842 行。
实测两边 sha256 = `961d6723d5dfd30fdaa19bbc23d78b87bd52102042041955fcaa39b5b36e184d`（36463 B）⇒ `LIFT OK`。
⇒ S3 的 device 逻辑与 m9 独立模块**同一份字节**，其 20 档 m=1/m>1 判据继续成立。

## 4. 设备档读数

环境：`M15_LAYERS=1`、真 checkpoint 权重、`M15_SKIP_WCHECK=1`、`M15_PREFILL_WIRE=1`、
`M15_PREFILL_KIND=1`、`M15_PREFILL_PHASES=3`（H1|A）。每档各自 `flock -w 300 /tmp/npu0.lock`、
进锁先 `npu-smi`（快照在该档日志内）、`timeout` 在锁内。

| 档 | 命令要点 | 设备判据 | 日志 |
| -- | -------- | -------- | ---- |
| `m1_clean` | `M15_PREFILL_PROLOG=1 M15_PREFILL_M=1` | ALL PASS `checks=59 fails=0` | `logs/run_m1_clean.log` |
| `m4097_clean` | `M15_PREFILL_PROLOG=1 M15_PREFILL_M=4097` | ALL PASS `checks=59 fails=0` | `logs/run_m4097_clean.log` |
| `m1_hostsynth` | `M15_PREFILL_PROLOG=0 M15_PREFILL_M=1`（回归对照） | ALL PASS `checks=32 fails=0` | `logs/run_m1_hostsynth.log` |

- **宿主合成回归**：`m1_hostsynth` 的 `checks=32/fails=0` 与 M140 `p3` 基线**逐字相同**
  （`evidence/m140_prefill_full_layer/logs/run_m1_p3.log`：`checks=32, guards=9, fails=0`）⇒ 切回旧路径读数回到已知值。
- **相位 A 的输入确实来自设备（见证）**：同一档位切开关，`m23_Pf.gdn_{q,k,v,g,beta}.bin` 五个面
  在 device 档与宿主合成档之间 **sha256 全 DIFFER**（见 `reproduce.sh` 末段）：q `a35e7a4a…` vs
  `d377afe4…` 等。⇒ 相位 A 吃的是设备产出，不是旧合成面。

## 5. 离线对拍（独立参考，锁外）

`check_prolog.py`（本次新增）：
- **S2**：`ref = bf16(a) @ bf16(b)^T`（fp32 累加 → bf16 RNE）。口径：`|Δ| ≤ 2.5e-3·Σ|terms| + 1e-6`
  （≈0.64×bf16 ULP）。**为什么不是逐位**：设备用 base-K 分块 fp32 累加、numpy 用 BLAS 累加序，
  真实激活里的近零抵消项会把 fp32 累加序差放大成若干 bf16 ULP；实测位型逐位率
  m=1 `16475/16480 = 0.999697`、m=4097 `67502321/67518560 = 0.999759`，T3 超界 0。
  （m11 自带 `check_ref.py` 在同一档给出 5 / 16239 个位型差 —— 同一现象，见 §6。）
- **S3**：**直接 import** `m9_gdn_prolog/check_ref.py::check_mt_case`（不复制判据 ⇒ 参考与实现独立；
  `check_prolog.py` 用 `importlib` 载入它）。判据：conv_state_out 位型逐位、q/k 尾部 64 行与
  g/β 行内 pad 列必须为 0、有效区 `1e-5·|exp|+1e-6`。

| 档 | S2 | S3 |
| -- | -- | -- |
| `m1_clean` | PASS（bit 0.999697，T3 0） | state/qpad/kpad/gpad/bpad PASS + q/k/v/g/β PASS（maxRelDiff ≤ 1.76e-4） |
| `m4097_clean` | PASS（bit 0.999759，T3 0） | 同上全 PASS |

**相位 A 参考判据（`m23_gdn_prefill/check_ref.py --allow-stale`；M165 已把默认参考改为指数差）**：

- `dumps_m1_clean`：m=1 设备有限，`o max|Δ|=5.27e-9` ⇒ **PASS（rc=0）**。
- `dumps_m4097_clean`：设备 `o` 有 **8,374,400** 个非有限（`ht` 262,144），而**指数差参考自身 0 非有限**
  ⇒ m23 判据 **FAIL（rc=1）**（`m23_gdn_prefill/check_ref.py:218-219` 的非有限分类：`仅一侧NaN=8,374,400`）。
  这条 FAIL 是**真判红**：设备这份 dump 是 **M163 之前**的 naive 核产物（见下）。

**为什么这条期望从「PASS」改成「预期红」（引 M160 / M162 / M165）**：

- M151 当初的期望基于**旧式**（naive）参考 `eg=exp(ĝ); ig=exp(−ĝ); Γ=eg·ig`
  （`check_phaseA_nan.py:59`@基线 `ad38d4d`；同一写法在设备侧 `m15_gdn_prefill.h:959,963,969,1047`@`ad38d4d`，
  M160 finding 的 Location 亦逐条编号）。旧式在 chunk 内 `|ĝ|>88.7`（fp32 `exp` 上溢阈值）时 `eg→0`、`ig→inf`
  ⇒ `0·inf=NaN`；设备与被测参考**同时 NaN**，于是当初报「device NaN 掩码 ≈ fp32 参考，XOR=128」的 PASS。
- **M160 survey**（finding `.tower/comms/findings/20261004-agent-nonfinite-bug-chunked-gdn-prefill-scan-evaluates-exp-exp-naively-fp32-nan.md:13,16,21-24`；
  其结论亦落在仓内 `m23_gdn_prefill/README.md:410-412,433`）：把参考改成**指数差** `exp(ĝ_i−ĝ_j)≤1`
  （官方 FLA `vllm/.../chunk_o.py:119-120`、`chunk_delta_h.py:216-221`）后，同一份 `dumps_m4097_clean` 的
  fp32 参考 NaN 由 **8,374,272 → 0**。⇒ 那 8,374,400 个非有限是**设备侧旧式写法造成的真缺陷**
  （真权重让 `|g|` 很大：M160 finding `:23-24`），**不是**「fp32/fp64 精度差」；
  当初「精度产物、不是接线/逻辑缺陷」的结论**已被 M160/M165 推翻**。
- **M162**（设备侧修复，commit `5d53c61`；`m15_layer_loop/m15_gdn_prefill.h:24-30` 的 `Γ[i,j]=exp(ĝ_i−ĝ_j)`、
  `:67-68` 的 `M15GP_STAB_NAIVE` 负向宏）：设备 Γ/KT′ 改用指数差，实测 m=4097 `o`/`ht` 非有限
  **8,374,400 / 262,144 → 0**（`m15_layer_loop/evidence/m163_stab/README.md:4-6`@`5d53c61`）。
  同一 mission（塔扩宽 scope）把 `check_phaseA_nan.py` 的**默认参考也改成指数差**、并加**参考自洽守卫**
  （`check_phaseA_nan.py:158-159`@`e088afb`：所选参考自身非有限必须为 0），旧式留作 `--naive-ref` 对照。
- **M165**（参考侧，commit `600b05e`；`m23_gdn_prefill/check_ref.py:11-15,91-96`）：m23 的 `ref_head` 默认
  也改为指数差、`--legacy-exp` 留旧式对照。

⇒ 于是**修正后的参考在这份 M151 归档 dump 上判红**。这不是判据回退、也不是新回归：是**旧期望（基于旧式参考）过期**，
且旧期望恰好把设备的**真缺陷**放过了 —— M160/M165 的原始 finding 正是为此而立的。

**核对（修正后的 `check_phaseA_nan.py`，指数差默认）**：

| 量（`dumps_m4097_clean`，`o` 共 25,171,968） | 值 |
| -- | -- |
| device `o` 非有限 | **8,374,400** |
| 指数差参考（fp64 与 fp32）非有限 | **0**（自洽要求） |
| 旧式对照 fp32 参考非有限 | **8,374,272**（⇒ 旧式参考自身会造 NaN） |
| XOR(device, 指数差 fp32) | **8,374,400** ⇒ **FAIL（rc=1）** |

实测命令与 rc：`check_phaseA_nan.py dumps_m1_clean` → rc=0；`… dumps_m4097_clean` → rc=1；
`… dumps_m4097_clean --naive-ref` → rc=1（**参考自洽守卫**咬住旧式参考：naive fp64 自身有 3,670,912 个 NaN）。
⇒「改用 `--naive-ref` 让这一项回绿」这条路**走不通**（守卫不容纳自含 NaN 的参考）；本 mission 选的是
「把期望改成与**指数差**参考一致」——即对这一档的参考判据**期望 rc=1**，由 `reproduce.sh` 的逐档 `expect_rc` 显式记录。

**残余限度（如实登记）**：本 mission 只订正**复算脚本的期望**，**未重跑设备、未改任何 dump**；
它不声称 `dumps_m4097_clean` 的相位 A 数值可信 —— 修正后的参考已判它**红**（设备当时输出确为错，M162 已按同一机理修设备）。
`dumps_m1_clean` 仍全域有限（m=1 `o max|Δ|=5.27e-9`）。

## 6. 负向对照（能变红，非空洞）

prolog 档每档自动跑三个负向臂（`M15_PREFILL_PROLOG_MUT=1/2/3`），判红由 `check_prolog.py` 做：

| 臂 | 变异 | 设备侧读数 | 离线判据 |
| -- | ---- | ---------- | -------- |
| `Pf.prolog_mut1` | S3 `noShift=1`（不交接 conv_state） | `conv_state 前进=0` | S3 `state=FAIL` ⇒ **红** ✓ |
| `Pf.prolog_mut2` | S3 `noPad=1`（不写 q/k pad 行） | `pad 全零=0`（毒值留下） | S3 `qpad/kpad=FAIL` ⇒ **红** ✓ |
| `Pf.prolog_mut3` | S2 `GEMM_MODE_KMINUS1`（K 少累加一个 base 块） | `q=7.97e-2`（干净 8.33e-2） | S2 T3 超界 12086（m=1）/ 46413615（m=4097） ⇒ **红** ✓ |

`prolog_mut2` 之前一度是**空洞**的（q/k 面被上一臂写过零，`noPad` 不留痕）⇒ 本次在 S3 前**毒化**
相位 A 的输入面（0xCD），毒值让"不写 pad"必定显形。设备侧另有 `H_CmpOk(finite)` / pad 契约两项判定。

## 7. 已知限度 / 未接（如实登记）

1. **m=4097 相位 A 的 o/ht 有非有限**（归档 dump：o NaN 8,374,400/25,171,968、ht NaN 262,144/786,432）。
   原因 = **设备侧旧式 Γ/KT′ 写法** `exp(ĝ)·exp(−ĝ)` 在 chunk 内 `|ĝ|>88.7` 时 `0·inf=NaN`（M160 finding
   `:13,16`；真权重让 `|g|` 很大，`:23-24`）—— **不是**「fp32/fp64 精度范围之差」（该结论已被 M160/M165 推翻，见 §5）。
   M162 已按**指数差**修设备（commit `5d53c61`；`m15_layer_loop/m15_gdn_prefill.h:24-30`），实测 `o`/`ht`
   非有限 **8,374,400 / 262,144 → 0**（`m15_layer_loop/evidence/m163_stab/README.md:4-6`@`5d53c61`）。
   - 本 mission **未重跑设备**：上面一行描述的是 M151 **归档 dump（M163 前）**的性质，不是当前设备。
   - 旧读数「device NaN 掩码 ≈ fp32 变体参考、只差 128 处」是**旧式参考自身也 NaN** 造成的**假吻合**；
     修正后的指数差参考在这份归档上判红（§5）。
   - 设备侧 `H_CmpOk(finite)` 那条门（`m15_layer_loop.asc:4474` 的 `badO < C.pfGdnOHost.size()`，登记注释在
     `:4175-4178`）在 m=4097 上近乎空转（只要有一个有限 `o` 就过），**唯一**意义是"没全变 NaN"，
     **不得**当作 m=4097 数值可信的判据。**m=1 仍要求零非有限**（实测 0）。
2. **epilog 未接**（`wsO → hcAttnOut`）：H2 的 `bo` 仍来自宿主合成（`H_PfHcFillHost`）。
3. **attention B2/B3 未接**：`KIND_ATTN` 在 `wire=1` 下响亮失败（`m15_layer_loop.asc`）。
4. **M140 红项原样保留**：相位 B m=1（`evidence/m140_prefill_full_layer/README.md:133-138`）、
   m=4097 `h2.blk` ulpMax 11（`:120-124`）。本 mission 未动相位 B/H2 的判据。
5. `Pf.gdn_mut2`（hcH 扰动）在 prolog 档**跳过**：它会沿 BLK→S2→S3 渗进相位 A 并把 gating 推成
   非有限，不再是"只扰 hc 相位"的干净对照；prolog 档的负向对照改用上面三个。prolog 关的档仍跑它。
6. **挂载形态 = host 编排的 4 次 launch/层，不同于终态（如实登记的差距）**。本次把 prolog 挂成
   **H1 / S2 / S3 / 其余相位**四次独立启动（`m15_layer_loop.asc:4127`（H1）、`:3839`（S2）、
   `:3867`（S3）、`:4138`（其余）），而同一个 `m15_layer_resources.h` 头文档化的**终态**是
   「**每层一次 `__mix__(1,2)` 启动、kernel 内四个相位**」（`m15_layer_resources.h:4`）⇒
   本次是**打通步骤、不是终态**：
   - **①额外 launch**：每层 4 次启动（终态 1 次），多 3 次下发/同步往返。
   - **②GM 往返**：S2 的输出 `pfQkvzbaDev` 是 **135 MB/层**（`4097×16480×2 B`；定尺
     `m15_layer_loop.asc:1860`、分配 `:1890`）⇒ 必须落 GM 再被 S3 读；`conv_state` 也每档
     H2D 进 / D2H 出（`:3832` / `:3894`）。终态形态下这些中间面本可留在 UB/L1。
   - **③无 UB/L1 复用**：S2（`M15OP::OProjGemm`，自有 L1 ping-pong）与 S3（m9 donor，自有 UB
     静态区）是独立 kernel，各自 UB/L1 在**内核之间不共享** ⇒ prolog 与紧随的相位 A 之间没有
     "同一入口内复用同一 UB/L1 窗"的机会；峰值因此只能**每 kernel 各自**登记（`m15_layer_resources.h`
     §3c-bis）。
   - **后续动作（follow-up build 项）**：把 S2/S3 **内联进 fused prefill 入口**（`M15L_PrefillBody`，
     即 docs/22 §6.1 的写法）⇒ 让 prolog 与相位 A 共享 UB/L1 窗、去掉 `pfQkvzba` 的 GM 往返。
     已知障碍：m9 的 `MtGdnProlog` 是 AIV-only 且依赖 `GetBlockNum()==AIV 数` 与自有 UB 基址，
     内联需先解决分核/UB 基址语义（这也是本次选 host 编排的原因）。

## 8. 复算命令（可传播失败）

```
# 全量（3 档设备 + 离线；锁纪律内建）——离线按"刚重生成"核对（m=4097 参考判据期望 rc=0）
bash m15_layer_loop/evidence/m151_prefill_prolog_wiring/reproduce.sh
# 只跑离线（用已有 dump）——归档是 M163 前 naive 核 ⇒ m=4097 参考判据**期望 rc=1**（§5）
bash m15_layer_loop/evidence/m151_prefill_prolog_wiring/reproduce.sh --offline-only
# 单独：相位 A 的 NaN 位置核对（M162 订正版；默认指数差参考）
/usr/local/python3.12.13/bin/python3 m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py \
  m15_layer_loop/evidence/m151_prefill_prolog_wiring/dumps_m4097_clean   # 归档 ⇒ rc=1（真缺陷）
# 单档设备（逐字；从仓库根）
flock -w 300 /tmp/npu0.lock bash -c '{ npu-smi info; timeout 240 env \
  M15_LAYERS=1 M15_SKIP_WCHECK=1 M15_PREFILL_KIND=1 M15_PREFILL_WIRE=1 M15_PREFILL_PHASES=3 \
  M15_PREFILL_PROLOG=1 M15_PREFILL_M=1 M15_PREFILL_DUMPDIR=<dumpdir> \
  ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt prefill; }'
```

`--offline-only` 对 m=4097 的两条相位 A 参考判据（`check_phaseA_nan.py`、m23 `check_ref.py`）**期望 rc=1**（理由见 §5），
对 m=1 期望 rc=0；`reproduce.sh` 用逐档 `expect_rc` 把这两组期望与实测 rc 分别记录（红/绿都不被吞）。
归档 dump 是 M163 前的 naive 核产物；用**当前（M163 后）二进制**重跑的读数见 `m15_layer_loop/evidence/m163_stab/`。

> 版本注记：本 README 与 `reproduce.sh` 的相位 A 期望按**并入 M163 后**（订正 `check_phaseA_nan.py` 的 `e088afb`，
> 已在当前 main）定义。本分支 base `ad38d4d` 早于该合入 ⇒ 单独 checkout **base** 时 `check_phaseA_nan.py` 仍是旧式，
> m=4097 那一项会因「脚本尚未订正」而与期望不符；以**并入 M163 后的树**为准（复现命令见 review-request）。

构建：`source /usr/local/Ascend/ascend-toolkit/set_env.sh && cmake --build m15_layer_loop/build -j4`。

## 9. 纪律自检

- **禁用词**：本目录新增文件对 mission 纪律清单里的绝对化措辞未命中；为免自命中，此处不逐字复写清单，
  逐字 pattern 与命令见 review-request。
- **判据非空洞**：三个负向臂各自把对应判据打红（§6）；`prolog_mut2` 的空洞已由毒化修掉。
- **参考独立于实现**：S3 判据是 m9 独立模块的 `check_ref.py`（importlib 载入，不复制）；S2 判据是
  纯 numpy 实现；相位 A 判据是 m23 的 numpy 参考（M165 起默认指数差）。
- **期望口径（M167 订正）**：相位 A 参考判据在 m=4097 上**期望 rc=1**（归档为 M163 前 naive 核，指数差参考
  正确地判红，§5）；`reproduce.sh` 用逐档 `expect_rc` 核对，期望与 rc 分开记录，红/绿都不被吞。
- **行号基准**：引用本目录/本分支文件的 `文件:行` 以本分支 base（`ad38d4d`）+ M167 两个文件为准；引用
  M160/M162/M165 的**订正物**时钉不可变 commit（`5d53c61` 设备修复、`e088afb` 订正 `check_phaseA_nan.py`、
  `600b05e` m23 参考），因为它们在本分支 base 之后才合入 —— 钉分支名会随合入而漂（塔第六变体纪律）。
