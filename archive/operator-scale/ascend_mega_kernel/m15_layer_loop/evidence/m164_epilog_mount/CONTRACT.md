# M164 接线契约 —— 相位 A 出口 `wsO` → epilog（转位 → S5 → S6）→ H2 的 `bo`(`hcAttnOut`)

> **口径（先读）**
> - 契约先写 `文件:行` 再落码。行号基准 = **包含本文件的提交**的 tip；重算：
>   `git log -1 --format=%H -- m15_layer_loop/evidence/m164_epilog_mount/CONTRACT.md`
> - 纪律：不写绝对断言；未取到的读数记「未读到 / 未取得读数」；所有代数只作定位辅助，
>   **符号名与契约注释才是锚点**（行号天然会漂，优先用可 grep 的符号名）。
> - 本 mission **只接 epilog 链路**：`相位 A wsO + prolog S2 的 z → 转位 → S5 RMSNormGated →
>   S6 out_proj → hcAttnOut(H2 的 bo)`。attention 的 B2/B3 **未接**；**不得**读成「整层四相位打通」。
> - 人类约束（逐字）：「矩阵乘法 ⇒ 必须 cube 做，其余的都用 VF 做，不要用 scalar 做计算，
>   scalar 只做控制流」；同步「不要使用 set flag wait flag 系列的同步，核内 pipeline 需要使用
>   buffer id，核间使用 set cross core 系列」。

---

## 0. 相位拓扑与挂载位置

四相位骨架 = `H1 → A → H2 → B`（`m15_layer_kernel.h:18-21`，prefill body `M15L_PrefillBody`
`m15_layer_kernel.h:830-852`）。相位掩码位 `PF_STAGE_H1=1 / A=2 / H2=4 / B=8`
（`m15_layer_kernel.h:655-659`）。

- kernel 内锚点：相位 A `m15_layer_kernel.h:855-857` → `FLAG_HC1_BOUND_AIV` 边界 `:859-861` →
  H2 `M15L_HcPrefillBoundary(1u, A)` `:861-863`；H2 的 `bo = second ? A.hcAttnOut : A.hcBo`
  （`m15_layer_kernel.h:818`）。
- ⇒ epilog 的**语义位置** = 相位 A 之后、H2 之前。
- 但本 mission 照 M151 的**host-orchestrated 形态**（`H_PfGdnWired` / `H_PfPrologRun`，
  `m15_layer_loop.asc`），把「rest 单次 launch」拆成 `A` 与 `H2|B` 两次，中间插三段 epilog：
  ```
  PfLaunch(PF_STAGE_H1); H_Sync("pfgdn_h1");     // ① H1 产 BLK（prolog 的输入）
  H_PfPrologRun(...);                            // ② S2 in_proj + S3 prolog
  PfLaunch(rest & PF_STAGE_A); H_Sync("pfgdn_a");// ③ 相位 A 产 wsO（pfGdnODev）
  H_PfEpilogRun(...);  H_Sync("pf_epilog_s6");   // ④ 转位 → S5 → S6 → hcAttnOut
  PfLaunch(rest & (H2|B)); H_Sync("pfgdn_h2b");  // ⑤ H2 的 combine 读新的 bo
  ```
- **为什么必须拆**：H2 的 combine 读 `bo = hcAttnOut`（`m15_hc_layer.h:463` `useCombine =
  (mode != MODE_MIX)`；`:676` 读 `boGm[mi*HID + jj*CHUNK]`）；A 与 H2 同一次 launch 就插不进去。
- 挂载代码：`m15_layer_loop.asc` 的 `H_PfGdnWired` → `epilogOn` 分支（见 §6）。
- `wsO` 的字段声明 `m15_layer_kernel.h:263`（`[48, m, 128] fp32`）；存储式
  `m15_gdn_prefill.h:1089-1091`（`oOff = hv*m_*GP_DV + t0*GP_DV`）。host 传
  `pfGdnODev`（`m15_layer_loop.asc` 的 `H_PfGdnWired` 里 `PfLaunch` 的实参表）。
- **拆后的相位边界重放（M163 复审非阻塞项 1；M175 补写）**：`M15L_PhaseBoundaryAiv<>`
  （`m15_layer_kernel.h:360-365`；符号锚点为准，行号随 tip 漂移）是**无条件**跑在**全体 AIV** 上的
  mode-0 屏障 —— `CrossCoreSetFlag/WaitFlag<CC_MODE0>`，set 挂 `PIPE_MTE3`、wait 挂 `PIPE_MTE2`
  （`:363-364`）。接线段体里三条边界调用（`:915` / `:921` / `:927`）都收在 `if ASCEND_IS_AIV` 内，
  但**不在** `pfStageMask` 的相位分支里（相位分支见 `:911` / `:917` / `:923` / `:929`）⇒ **每次 launch
  都会跑全三条**，与开了哪些位无关；`pfStageMask` 只决定段体跑不跑，不决定边界跑不跑。
  - **为何安全（由构造，不是靠观测）**：① mode-0 是「前一段落 GM、后一段从 GM 读」的 **all-to-all**
    屏障，**同一次 launch 内参与方固定**（`__mix__(1,2)`、`PfLaunch` 的 blockDim 固定）且**每个 AIV
    都执行**同一条 set 与同一条 wait ⇒ 每次都是良构的 N set ↔ N wait，没有核会跳（M136 的挂死形态是
    **AIC 也被拖进 mode-0**，本形态把调用收进 `if ASCEND_IS_AIV` 正是修法，注释见
    `m15_layer_kernel.h:905-909`）。② 多跑的边界在「该相位无数据流动」时是空操作：屏障只对齐到达时刻，
    不搬运也不校验数据。③ 每次 launch 是独立 kernel 调用、同一 stream 串行，flag 计数在 launch 内自洽
    ⇒ 无跨 launch 残留。
  - **计数（本 mission 的档 H1|A|H2）**：本 mission 把 rest 拆成 ① H1（`m15_layer_loop.asc:4416`）、
    ③ A（`:4424`）、⑤ H2|B（`:4432`）三次 launch，每次都跑全 3 条边界 ⇒ **每层 9 次**（单次 launch
    形态下是 3 次）；`h2bMask == 0` 时只两次 launch（6 次）。
  - **实测支撑（9 档）**：M164 的 9 档设备运行（`logs/run_*.log`，本目录；矩阵见同目录
    `README.md` §4）在该拆法下均未挂（无 timeout/挂死），且各臂的边界产出面判据为真（H1 的 BLK、相位 A 的
    `wsO`/m23 对拍、H2 的 combine 输入 `bo` 逐面 DIFFER）。**这是"没挂 + 该臂数据判据成立"的支撑，不是对
    屏障本身的独立判据** —— 相位边界不单独出读数，故本条的安全性以①的构造论证为主、实测为辅，不写成
    「屏障已被单独验证」。

---

## 1. M159 引用的逐条复核（本 mission task 1）

M159 的 survey 以 `@646c881`（M151 分支）与 main tip 混合为基准。本 mission 的 base 是
`main` **已含 M151 合入**（`git merge-base --is-ancestor 646c881 HEAD` = 0），故 M159 §0 的
「M151 未合入」前提**已不成立**。逐条核对结果（`OK` = 仍成立；`漂移` = 行号已动，符号仍在）：

| M159 引用 | 复核 | 现在的位置 / 符号 |
|---|---|---|
| `m15_layer_kernel.h:18-21` 四相位注释 | OK | 同 |
| `m15_layer_kernel.h:655-659` `PF_STAGE_*` | OK | 同 |
| `m15_layer_kernel.h:830-852` prefill body | OK | 同 |
| `m15_layer_kernel.h:855-863` A→H2 边界 + H2 调用 | OK | 同 |
| `m15_layer_kernel.h:818` `bo = A.hcAttnOut` | OK | 同 |
| `m15_layer_kernel.h:263` `wsO [48,m,128] fp32` | OK | 同 |
| `m15_layer_kernel.h:150` `hcAttnOut` | OK | 同 |
| `m15_layer_kernel.h:648/:721` `PF_GDN_QK_PAD_ROWS` / `qkStride` | OK | 同 |
| `m15_layer_kernel.h:945-946` `subIn/subOut` | **漂移** | 现在是 `:963-964`（`subIn = A.hcWs0 + M15H::WS_BLK` / `subOut = A.hcAttnOut`） |
| `m15_layer_kernel.h:1281-1307` `M15L_LAYER_PREFILL_ARGS_*` | **漂移** | `_DECL` `:1283-1307`；`_FILL` `:1309` 起（**不在** M159 引的区间内） |
| `m15_gdn_prefill.h:1089-1091` `oOff` | OK | 同 |
| `m15_gdn_resources.h:48-59`（Z_OFF/Z_DIM） | OK | `Z_DIM` `:52`、`Z_OFF = V_OFF + V_DIM` `:57` |
| `m15_gdn_resources.h:272` `SZ_QKVZBA` | OK | 同 |
| `m15_hc_resources.h:40` `HID=2560` | OK | 同 |
| `m15_hc_prefill.h:114` `ROW_HID = HID*2` | OK | 同 |
| `m15_hc_layer.h:415/:441/:463/:664-676` `bo`/`boGm`/`useCombine`/读 `boGm` | OK | 同 |
| `m28_epilog_chain.asc:899/:906/:914-921` 三个链入口 | OK | 同 |
| `m28_epilog_chain.asc:876-877` Fixpipe `dstStride=N` / `F322BF16` | OK | 同 |
| `m28_epilog_chain.asc:106-115/:766-805` BufferID / `AscendC::Mutex` | OK | 同 |
| `m28_epilog_chain.asc:241-249` S5 UB / `:714-717` S6 L1 | **部分漂移** | S5 UB OK（`:241-249`）；S6 的 **L1** 在 `:704-711`（`:714-717` 是 L0 ping-pong） |
| `m28_epilog_chain.asc:256-258/:1317-1323` S5 mutant 定义/解析 | OK | 同 |
| `m28_gdn_epilog.asc:86-90/:98/:161-164` 转位 UB / `MUT_HEADSTRIDE` | OK | 同 |
| `m15_layer_resources.h:503/:528-538` `PREFILL_WIRED` / `PF_PEAK_TABLE` | OK | 同 |
| `m15_layer_resources.h:1085-1176` `FLAG_SEQ_PREFILL_*` | **部分漂移** | 表在 `:1117-1124`（GDN）/`:1145-1152`（ATTN）；§4d 区间约 `:1083-1210` |
| `m15_attn_oproj.h:63/:287` `GEMM_MODE_KMINUS1` | OK | 同 |
| `m15_attn_core_host.h:791` K−1 驱动 | OK | 同 |
| **`m15_layer_loop.asc:1824/1842/1867`（hcAttnOut 分配/毒）** | **漂移** | 声明 `void* pfHcAttnOutDev` 在 Ctx；`H_PfSegAlloc` 的 `items[]` 与毒值块整体后移（本次改动前的 tip 上：items `1867-1879`、毒 `1903-1919`；本提交后见 §6） |
| **`m15_layer_loop.asc:1832-1844` `H_PfSegAlloc` items** | **漂移** | 同上 |
| **`m15_layer_loop.asc:2086/2095` `W_GG_OFF`/`W_OUT_OFF`** | **漂移** | 现在是权重装配的 `put(M15Loop::W_GG_OFF, …)` / `put(M15Loop::W_OUT_OFF, …)`；`H_W` 见 `H_Launch` |
| **`m15_layer_loop.asc:3751` `hcMlpMode = MODE_COMBINE_MIX`** | **漂移** | 现在 `hcMlpMode` / `hcAttnMode` 在 `H_PfGdnWired` 的 `hcAny` 块内（本次改动前 `:4014`/`:4031`） |
| **`m15_layer_loop.asc:830` `H_Sync`** | **漂移** | `static bool H_Sync(...)` 在本次改动前的 tip 上是 `:860` |
| `m15_layer_loop.asc:3816` `H_PfPrologRun` | OK | 同（本次在其后插入 `H_PfEpilogRun`） |

### 1.1 一条事实更正（对本 mission 有实质影响）

M159 §2.3 写「现在 H2 读的 `bo` 是 `0xCD` 毒值（从无生产者）」。在当前 tip 上**不成立**：
`H_PfHcH2D`（`m15_layer_loop.asc`）会把**宿主合成**的 `bo` 平面 H2D 进 `pfHcAttnOutDev`
（seed `0x7201`、scale `0.5`；见 `H_PfHcFillHost`），也就是说 H2 当前读的是**宿主合成 bo**，
不是毒值。M138 登记的生产者缺口被 M140 用「宿主合成 bo」临时补上；本 mission 把它换成
**设备产出**。因此判据 (c)（`EPILOG=0/1` 的 H2 输出必须不同）比 M159 设想的更有意义：它区分的是
「宿主合成 bo」与「设备产出 bo」，而不是「毒值」与「真值」。

---

## 2. 平面 / 指针 / 实参

### 2.1 复用（既有）

| 用途 | 指针 / 符号 | 出处 |
|---|---|---|
| epilog 输入 `wsO` | `C.pfGdnODev`（`LayerArgs.wsO`；head-major fp32 `[48,m,128]`）| `m15_layer_kernel.h:263`；`m15_gdn_prefill.h:1089-1091` |
| z 源（packed 前）| `C.pfQkvzbaDev`（`[m,16480] bf16`；z 段 `[10240,16384)`）| `m15_gdn_resources.h:52/:57`；M151 产出 |
| epilog 出口 `hcAttnOut` | `C.pfHcAttnOutDev`（`[M_MAX,2560] bf16`，行距 `5120 B`）| `m15_hc_resources.h:40`；`m15_hc_prefill.h:114` |
| gamma（S5）| `H_W(C,0,W_GG_OFF)`（`linear_attn.norm.weight [128] bf16`）| `m15_loop_layout.h` `W_GG_OFF`/`W_GG_BYTES` |
| Wout（S6）| `H_W(C,0,W_OUT_OFF)`（`[2560,6144] bf16`）| `m15_loop_layout.h` `W_OUT_OFF`/`W_OUT_BYTES` |

### 2.2 新增（GM 暂存）—— **三块，不是两块**

| 平面 | 定尺 | 字节（m=4097）|
|---|---|---|
| `pfOTokDev` | fp32 `[M_PREFILL, 6144]`：转位后 token-major `o`（S5 的 `o` 输入）| 100,687,872 |
| `pfZTokDev` | bf16 `[M_PREFILL, 6144]`：z 压实（S5 的 `z` 输入）| 50,343,936 |
| `pfYTokDev` | bf16 `[M_PREFILL, 6144]`：S5 出 = S6 的 `A` 面 | 50,343,936 |

**与 M159 §2.2 的差别（一处更正）**：M159 只列了 `oTok` / `zTok` 两块。但链是 **S5 → S6 两次
独立 launch**，S5 的出口 `y`（`m28_epilog_chain.asc` README 的中间面带也单列它为「中间 `y`
（段②出、段③入=A）」）必须有自己的 GM 生命周期。m28 donor 的 `RunChain` 同样为 `y` 单独
`aclrtMalloc`（`yRows = (m<2)?2:m`）。⇒ 本实配**三块**。三块都在 `H_PfSegAlloc` 的
`items[]` 之后**按需分配**（`epilogOn`）并毒 `0xCD`。

- **实参**：不改 `LayerArgs`、不改 `M15L_LAYER_PREFILL_ARGS_DECL/FILL`；epilog 的三个 kernel 用裸
  `__gm__ uint8_t*` 直接启动（照 `m15_prefill_prolog.h` 的 `m15_pf_inproj_gemm_kernel` 形态）。
- host 开关 `M15_PREFILL_EPILOG` / `M15_PREFILL_EPILOG_MUT`；**依赖** `M15_PREFILL_PROLOG=1`
  （z 源 `pfQkvzbaDev` 是 S2 产出）；prolog 关时本档不挂 epilog（响亮提示，见 §6）。

### 2.3 `hcAttnOut` 的生产者 = S6 的 Fixpipe

`m28_epilog_chain.asc:876-877`：`fp.dstStride = N`（`N=2560` 元素）、`fp.quantPre = F322BF16`
⇒ 输出 bf16 `[m,2560]`、行距 `2560*2 = 5120 B`，**等于** H2 的 `ROW_HID`
（`m15_hc_prefill.h:114`）⇒ S6 直接写 `hcAttnOut`，无需再落位拷贝。
M138 的型别/行距漂移（`wsO` fp32 head-major vs `hcAttnOut` bf16 行主序）由**步骤①的转位**处理掉
（元素下标 `h*m*128 + t*128 + d` → `t*6144 + h*128 + d`）。

---

## 3. 资源与同步

- **同步**：三个 epilog kernel（transpose / S5 / S6）各自是独立 `__global__`，同 `C.stream` 串行 +
  每次 `H_Sync`；段内只用 BufferID（`m28_epilog_chain.asc:106-115` 的 `BufAcquire/BufRelease`、
  `:766-805` 的 `AscendC::Mutex`/`PipeBarrier`）。⇒ **不新增 flagId、不用 set flag/wait flag**；
  `m15_layer_resources.h` 的 `FLAG_SEQ_*` 表一字未动。
- **峰值登记**（`m15_layer_resources.h` §3c-ter；数值 = 各设备头里那条 `static_assert(... <= 预算)`
  的**被断言量本身**）：
  - 转位段 UB = 196,608 B（`M15PE::Trans::UB_Z + UB_Z_BYTES`）
  - S5 UB = 50,688 B（`M15PE::Chain::UB_S5_RSTD + 256`）
  - S6 L1 = 303,104 B（`L1_B_REGION + 2*L1_B_ELEMS*2`）；S6 L0C = 40,960 B（`BASE_M*BASE_N*4`）
  - GM 台账：`PF_EPILOG_{OTOK,ZTOK,YTOK}_BYTES_PER_ROW`
  - **`PF_PEAK_TABLE` 不动**（固定 8 槽）；`static_assert(PfPeakUnfilled() == 3u)` 与
    `static_assert(PREFILL_WIRED == 0u)` **钉死**不变式。
- **跨文件机器见证**：`m15_layer_loop.asc` 顶部有 4 条 `static_assert` 把上面的峰值与设备头里的
  实际常量逐一对上（数字漂了编不过）。

---

## 4. 窗口划分（人类约束的逐条落实）

- ① 转位/落位：纯 DMA 块拷贝 + stride 寻址（既非矩阵乘法、也无需逐元素运算）⇒ AIV。
- ② S5：逐元素/规约（平方和、rsqrt、sigmoid 门）⇒ AIV 的 `__VEC_SCOPE__` 向量寄存器（VF）。
- ③ S6：`C[m,2560] = A[m,6144]·B[2560,6144]^T` ⇒ **矩阵乘法**，走 AIC 的 `Mmad`
  （`M15PE::Chain::S6::Bf16Gemm`，cube）。
- 地址/下标/循环边界/尾块 `rows`/`curM` 全部是 scalar 控制流；scalar 不做数值计算。

---

## 5. 验收判据（与 `check_epilog.py` / `reproduce.sh` 一一对应）

- **(a) 出处**：`hcAttnOut[0,m)` 行**无 `0xCD16` 残渣**、非零、无 inf/nan；且设备产出**逐面 DIFFER**
  于宿主合成 `bo`（`m164_hostsynth_bo.bin`）⇒ 证数据来自设备（不是宿主合成件的回声）。尾行
  `[m, M_PREFILL)` 必须**整片保毒**（未写区不得看起来合法）。
- **(b) 数值**：离线 numpy 链（转位 → S5 → S6，独立于 kernel）——①② 按 **T1 逐位**，
  ③④ 按 **T3**（`|Δ| ≤ 2.5e-3·Σ|terms| + 1e-6`；S5 另按 m12 口径 bf16 相对 `≤1e-2`），
  **段级 S5 / S6 分别 PASS**。
- **(c) H2 真吃到新 bo**：`EPILOG=0/1` 两档的 **H2 输出**（`m140_*_h2_blk.bin`）sha256 **必须不同**。
- **(d) 无副作用**：相位 A 的 m23 对拍（`m23_gdn_prefill/check_ref.py`）在 m=1 与 m=4097 仍 PASS。
- **(e) 负向对照（能变红）**：转位 `headstride`（只 m>1 红、m=1 退化仍相等）、S5 `nogamma`、
  S6 `K−1`（经 `M15OP::OProjGemm` 的现成 mutation）。期望红/绿由 `check_epilog.py` 按 `mut` 逐档判。

---

## 6. 本 mission 的落点（文件:符号）

- `m15_layer_loop/m15_prefill_epilog.h`：`M15PE::Trans`（m28_gdn_epilog.asc:69-220 逐字）、
  `M15PE::Chain`（m28_epilog_chain.asc:91-893 逐字）、四个入口壳
  `m15_pf_epilog_{transpose,s5,s6,s6_mut}_kernel`。lift 见证 `evidence/m164_epilog_mount/verify_lift.py`。
- `m15_layer_loop/m15_layer_loop.asc`：`Opts.pfEpilog/pfEpilogMut`（`:186-187`）、env 解析
  （`:290-291`）、`Ctx.pfOTokDev/pfZTokDev/pfYTokDev`（`:797` 起）、`H_PfSegAlloc` 的 `epilogOn`
  （`:1903`）、`H_PfEpilogDump`（`:4033`）、`H_PfEpilogRun`（`:4079`）、`H_PfGdnWired` 的 launch
  拆分（`:4374` `epilogOn`；`:4404` 调 `H_PfEpilogRun`）、跨文件峰值 `static_assert`（`:108` 起）。
- `m15_layer_loop/m15_layer_resources.h`：§3c-ter（`:599` 起；峰值常量 `:616-619`、GM 台账 `:627`）。

---

## 7. 本次不接（窄口）

- epilog 的**权重**用层 0 的 `gamma`/`Wout`（真实 shape，与本 run 的 `M15_LAYERS=1` 一致）；
  多层链路未接。
- attention 的 **B2/B3 未接**；`PREFILL_WIRED` 仍为 0。
- 既有红项（相位 B m=1 MoE router、m=4097 `h2.blk` ulpMax 11）**作为回归输入保留**，
  **不预判因果**；本 mission 只重设基线。
- epilog 的 S5/S6 未做多 stage 流水/带宽调优；B 相位（MoE）不在本档掩码内（用 `H1|A|H2=7`）。
