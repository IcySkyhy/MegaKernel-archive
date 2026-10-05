# M124 — PLE ③ kv 投影：AIV 逐列 GEMV → **cube `Mmad`**（合规返工 + 层路径 AIC 挂载点）

> 分支 `feat/m124-ple-plegemv-cube-mmad-compliance-re`（**未合入 `main`**；引用本文件的读数时按
> `docs/17 §9.6` 标注分支与 tip）。输入依据 = `docs/20-kernel-compliance-sweep.md` 的 §2.2-V1（P1）、
> §3.3 的 G1/G2/G3、§4 的 **WO-A1 / WO-B2**，加人类逐字口径「**凡是涉及矩阵乘法的操作都需要用
> mmad 实现，不管 M=1 或者是多大**」「矩阵乘法 ⇒ 必须 cube」「不要留 VF 备选支，直接搬 cube」。
> 本文件是该 mission 的正式交付物（改动清单 + 判据 + 读数 + 「没做完 / 没取到」清单）。

## 1. 一句话与结论

③（`kv = emb @ wcatᵀ`，`M=n_tok`、`K=2560`、`N=12800`）**从 AIV 的逐列向量 GEMV 换成 AIC 上的
cube `Mmad`**（28 条 AIC 按 `BASE_N=160` 的 N-tile 轮转条带，`K` 分 40 块），并把**层路径**
（`m15_layer_kernel.h` 的层 1 PLE 打断点）的 AIC 侧挂载点接上；判据侧把 ③ 的 T3 界换成
`docs/17 §1.1` 的 cube 推导式 `EPS_MMAD = 2560·2⁻²⁴`，**改前/改后同一条判据**。

**已验到的**（命令与原始日志见 §7）：

| 项 | 读数 |
|---|---|
| 独立 kernel 改前（`9296e79` 重建） | `rc=0`、`ALL PASS`（14 条判据 / 0 FAIL）、`B_kv.bin sha256=f07fadeb…` |
| 独立 kernel 改后（本 tip） | `rc=0`、`ALL PASS`（14/14）、`B_kv.bin sha256=b88e108d…` |
| 改前/改后逐平面差异 | `kv`：**9/25600 元素不同（0.035%）、最大 1 个 bf16 格点、>1 ulp 者 0 个**；`emb`/`ids`/`out` **逐字节相同**；`gated` 15/20480、`normed` 13/20480（≤2 格点）、`state_out` 3/184320（1 格点） |
| 变异矩阵（15 位） | 基线 14/14 PASS；**15 位全部按期望变红**；覆盖 14/14（`RESULT\|coverage\|14\|14\|(none)\|OK`）；其中 **bit7（③ key/value 对调，cube 档形态）⇒ `B3.kv` FAIL** —— 即 ③ 这条判据在新算法上咬得住 |
| 生成器↔产物双向交叉对拍 | 2×2 四条腿的 sha256 分别等于**对应源**的归档值（旧 8bbd5e03… / 新 55bba435…），四条腿 `--check` **rc=0**；手改产物一个 token ⇒ `--check` **rc=1** |
| 层路径 `runs=all` 零回归 | **2095 + 302 / 0 FAIL**（与当前 main 基线逐字相同）⇒ 见 §6.5 |
| 层路径 `plewire`（接线开，A/B 档） | **12/12 Pw PASS**（M124 tip 口径：A/B 均 PASS；与 M111 归档的 10 条判定行**逐行相同**，多出 `Pw.kv.T3`/`Pw.kv.nonvac` 两条），`Pw.wired.delta = 5636/10240` ⇒ 见 §6.4 |

**没做完 / 没取到读数的**：§8 逐条列（其中一条 —— 层路径 kv 的**数值**判据 —— 要落
`m15_layer_loop.asc`，本 tip 的 scope 不含该文件，已报塔待裁）。

## 2. 改了什么（逐文件）

| 文件 | 改动 |
|---|---|
| `m15_ple.asc` | 新增 `PleCubeGemv`（③ 的 AIC 段）+ 重写 `PleGemv`（双核型臂）+ §CUBE 编译期资源（L1/L0 偏移、分形常量、BufferID 复用 hc 的 0–6）+ `FLAG_CUBE_IN/OUT` + `FLAG_B2/B3` 语义从 mode-0 改 mode-2；独立 kernel 的 `m15_ple_body_kernel` 新增 **AIC 分支**（③ 的调用点）；删除只剩 ③ 用的 5 个 UB 槽常量（`UB_EMB`/`UB_WR`/`UB_ACC`/`UB_KVF`/`UB_KVB`，其中 `UB_ACC` 改前已是死常量）+ 新增 env `M15_PLE_NTOK`（1 或 2，取 decode 的 M=1 档） |
| `m15_ple_wire.h` | **机械生成物**：`python3 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py` 重新生成（`--check` rc=0），与 `.asc` 逐字节一致 |
| `m15_layer_kernel.h` | `M15L_PleBody`（AIV 侧）撤掉两条已过时的全体 AIV mode-0 barrier（`BarrierAiv<FLAG_B2>` / `<FLAG_B3>`）；新增 **`M15L_PleAic`**（AIC 侧挂载点）；`M15L_FusedBody` 的 AIC 分支在 hc(H1a) 与 hc(H1b) 之间调用它；flagId 相邻性 static_assert 按新的执行序重写 |
| `m15_layer_resources.h` | `FLAG_SEQ` 新增 **5** 行登记（PLE 的 mode2 `FLAG_B2/B3` 标 `core="both"`（AIV set/wait + **AIC wait/set**，r1 复审 P2-1 订正）、AIC mode0 `FLAG_CUBE_IN/OUT`、**AIV mode0 `FLAG_B4`**（r2 复审 P2-1 残留：它的执行位置在 `8(PLE_IN)` 与 `9(PLE_OUT)` 之间））+ 依据/跨 mode 复用注释 + 「事实」清单与 `FlagMaxUse()` 注释同步（**只做登记性最小改动**，既有行的 id/内容一字未动） |
| `m15_ple_check.py` | 新增 `EPS_MMAD = 2560·2⁻²⁴`（cube 累加界的推导式，`docs/17 §1.1` 的「mmad 的 `k·2^-24`」）；`B3.kv` 与 M92 的 `H3.kv` 的 ε 从 `EPS_REDUCE` 换成 `EPS_MMAD`（**同一条判据、同一条界**用于改前/改后）；`EPS_REDUCE` 保留给 ④ 的判据 |
| `m15_ple_mutants.py` | 变异 bit7 的**形态随算法更新**（从「逐列读对侧权重行」改为「N-tile 读取基址 ±HID」），语义仍是 D1 的顺序反；表头与本行注释同步 |
| `m15_layer_loop.asc` | **塔批复 scope 后新增**：`PleRunOut::kv` + `H_PleReadBack` 的 kv 回读 + 判据块 (3b) 的 **`Pw.kv.T3`** 与 **`Pw.kv.nonvac`**（host-only 参考 + T3 推导界 + 非空洞守卫）。**只动 PLE 判据相关的局部**：`Pw.det` 的六项、`runs=all` 的判据行、其它块一字未动 |
| `ple/README.md` | 新增 §0.2「M124 补丁」；§2 的 ε 表新增 `EPS_MMAD` 行；§1 判据表第 8 行 / §1 变异覆盖表第 7 行 / §4.1 的 ③→④ 残差段 的数字与口径同步；§7 U4 改写（③ 已不是逐列 GEMV；**仍未做**装置计时） |
| `evidence/ple_wire/m124/**` | 本文件 + 全部读数与复现脚本（`M124_*.log` / `M124_*.sh` / `M124_*.py`） |

**没有改**：`m15_layer_loop/CMakeLists.txt`（归 M101）、`m15_hc_layer.h`、`m15_gdn_layer.h`、
`ple/PLE_SPEC.md` 的语义表、③/④/⑤ 的数学口径（判据一条都没放宽或收紧 —— 只有 ③ 的 ε 按
「含 mmad 累加」换成推导式，且它比旧界**宽**、两版都 0 越界，见 §5.3）。

## 3. ③ 的算法与参数（抄改自哪里）

- **数学形态**：`C[n_tok, KVW] = A[n_tok, HE] · B[KVW, HE]ᵀ`，`A = G.emb`、`B = G.wcat`（`[12800,2560]` 行主序 `(out,in)`）。
- **donor（人类口径「优先从现有代码库抄改」）**：`m15_gdn_layer.h:Bf16Gemm`（M19/M22 的 bf16 GEMM）
  与 `m15_hc_layer.h:HcBf16Gemm`（同形，M36/M58）。四组参数与 ping-pong BufferID 形态逐条照抄：
  - `CopyInA`：`Nd2NzParams{nValue=calcM, dValue=BASE_K, srcDValue=HE, dstNzC0Stride=BASE_M, dstNzNStride=1, dstNzMatrixStride=0, srcNdMatrixStride=0}`；
  - `CopyInB`：同形，`nValue=BASE_N`、`dstNzC0Stride=BASE_N`，读 `wcat[ntRead*BASE_N*HE + kb*BASE_K]`；
  - `LoadA/LoadB`：`LoadData2DParamsV2` 的 `mStep/kStep/srcStride/dstStride` 与 donor 逐字一致；
  - `Mmad`：`mp.m=calcM, n=BASE_N, k=BASE_K, cmatrixInitVal=(kb==0)`；
  - `Fixpipe`：`FixpipeParamsArch3510<CO2Layout::ROW_MAJOR>`，`nSize=BASE_N`、`mSize=curM`、`srcStride=calcMAlign`、`dstStride=KVW`、`quantPre=QuantMode_t::F322BF16`（**RNE**，与改前 `Cast<...,castTraitB322B16>`（`CAST_RINT`）同方向）；
  - 核内次序：`MutexLock<PIPE_M>(L0C)` → 40 个 K 块（`MTE2 → MTE1 → M`）→ `MutexUnlock<PIPE_M>` → `MutexLock<PIPE_FIX>` → `Fixpipe` → `MutexUnlock<PIPE_FIX>`。
- **权重排布**：**不做任何在线权重格式转换**。`wcat` 的既有 `[12800,2560]` 行主序正是 donor 的
  `B[N,K]` 形态（`m15_layer_kernel.h` 的 `PLEW::W_CAT` 逐字「行主序 (out,in)」）⇒ B 侧只按
  `BASE_N` 行 × `BASE_K` 列的大包搬。
- **M 维**：一次 tile 吃下全部 token（`n_tok ≤ BASE_M=64`），**不按 token 拆 tile**。
  `M=1`（decode）按 donor 的 3510 契约抬到 2 行：`calcM = max(curM,2)`，
  多读的 1 行**必须可读**（独立 kernel 的 `emb` 平面按 2 token 分配、n_tok=1 时第 2 行在平面内；
  层路径的 `PLEW::S_EMB` 是 `T_MAX=64` 行的平面），Fixpipe 只写 `curM` 行。
- **分档与界**：`docs/17 §1.1` 的 T3 触发条件 ③「含 mmad / cube 累加」命中 ⇒ kv 的界 =
  `2560·2⁻²⁴·Σ|terms| + 1.0·ulp(out)`（`EPS_MMAD`）；**为什么改前也算这一条**：`EPS_MMAD` 盖住
  旧的 VF 路径界（`96·2⁻²⁴`）⇒ 用同一条界做改前/改后的比较，判据没有随算法漂移。
- **N 条带**：`NTILES = 12800/160 = 80` 个 tile，`for (nt = aBid; nt < 80; nt += nAic)`（`nAic=28`
  ⇒ 24 条核 3 个 tile、4 条核 2 个 tile）。每 tile 一趟完整 K 循环（40 块）⇒ 权重总读 = 65.5 MB
  （= `wcat` 的一遍），与改前同量级、但摊在 28 条 AIC 上而不是 56 条 AIV 的逐列 `Reduce`。

## 4. 核间同步：谁等谁、用什么 mode、号从哪来

### 4.1 协议（③ 的入口/出口都是跨核型交接）

```
AIV（② 的最后一个 `DataCopy` 写完 G.emb）
  └─ CrossCoreSetFlag<CC_MODE2, PIPE_MTE3>(FLAG_B2)        # 配对：本核 ↔ 它那条 AIC
AIC
  ├─ CrossCoreWaitFlag<CC_MODE2, PIPE_S>(FLAG_B2)          # 等到自己配对的两条 AIV 都交完 ②
  ├─ BarrierAic<PIPE_S, PIPE_MTE2, FLAG_CUBE_IN>           # 全体 AIC 到齐 ⇒ 任一 AIC 可读任一条 AIV 写的行
  ├─ PleCubeGemv（Nd2Nz → LoadData2D → Mmad → Fixpipe 落 G.kv）
  ├─ BarrierAic<PIPE_S, PIPE_FIX, FLAG_CUBE_OUT>           # FIXP 写 GM 排空 + 全体 AIC 到齐
  └─ CrossCoreSetFlag<CC_MODE2, PIPE_MTE2>(FLAG_B3)        # 通知自己配对的两条 AIV
AIV
  └─ CrossCoreWaitFlag<CC_MODE2, PIPE_MTE2>(FLAG_B3)       # 挡住随后的 ④ 对 G.kv 的 DataCopy 读
```

**mode 判定是「谁等谁」，不是「数据走不走 GM」**（塔的广播口径）：AIV 产出 → AIC 消费、配对是 mix
组内的 (1 AIC, 2 AIV) ⇒ **mode 2**；「每个 AIC 都要读**全部** AIV 写的行」（N 条带与 AIV 的 item
网格不一一对应 ⇒ all-to-all）⇒ 中间补一条**全体 AIC 的 mode 0**。逐字依据 = `docs/05` §2：
「mode 0 仅同类型（'全部 AIC 之间'或'全部 AIV 之间'二选一，不支持 AIC 组 set、AIV 组 wait 的跨类型
用法）；跨类型 all-to-all 的标准组合：**AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）→ AIC→AIVs
（mode 2）**」。形态先例 = `m15_attn_prolog_probe.h`（`AP_AIC_M0_OUT` → `AP_A2V_GEMM`）与
`m15_gdn_layer.h`（`FLAG_AIC_SEG0B` → `FLAG_BOUND_INPROJ`）；**没有**用通用 `SyncAllImpl`。

**两条编译期/硬件事实（实测踩到并留档）**：
1. `ffts_cross_core_sync` 的 compiler builtin 对**第 1 个实参（pipe）**只收 `[2,5] ∪ {10}`
   （`pipe_t`：`PIPE_S=0`, `PIPE_V=1`, `PIPE_M=2`, `PIPE_MTE1=3`, `PIPE_MTE2=4`, `PIPE_MTE3=5`, `PIPE_FIX=10`）
   ⇒ **set 侧挂 `PIPE_S` 直接编译不过**（逐字报错 `the ranges of 1st parameter must be [2, 5], [10, 10]`，
   在 `kernel_operator_sync_impl.h` 的 `NotifyEventImpl`）。wait 侧没有这条限制。
   ⇒ ③ 的入口对齐 set 取 `PIPE_MTE2`（与 `m15_hc_layer.h:530` 的 `BarrierAic<PIPE_S, PIPE_MTE2, FLAG_AC0>` 同形）。
2. **AIC 上不许挂 `PIPE_MTE3`**（`docs/05` §6.2 的硬规则：`IsSplitCubePipe = {S,MTE1,MTE2,FIX,M}`
   ⇒ 静默空操作、对侧永久挂死）—— 本段的所有 AIC 侧 set/wait 只出现在 `{S, MTE2, FIX}` 上。

### 4.2 逐核 id 用量表（**贴脚本输出，不手抄**；两种口径分开写）

读数来源：`m15_layer_loop/evidence/ple_wire/m124/M124_flagid_table.py` → 本目录 `M124_flagid_table.log`
（脚本从 `m15_layer_resources.h` 的 `FLAG_SEQ[]` **解析**常量与执行序，并**独立重实现** C++ 侧
`FlagSeqAdjacentOk()` / `FlagMaxUse()` 的语义，含 `CoreSame` 对 `"both"` 的处理，用来与编译期
`static_assert` 互为见证）。r1 复审 P2-1 与 r2 复审 P2-1 残留都指向同一类问题：**这张表早期是手抄的**
（先漏了 AIC mode2 的 12/13，后又把未登记的 `FLAG_B4` 手写进 AIV mode0 行、并把全表用量与层 1 用量混在一句里）。
本版起：`FLAG_B4` 已**登记进 `FLAG_SEQ`**，下表**逐字取自脚本输出**。

**两种口径（必须分清，不许混在一张表里）**：
- **全表口径** = `FLAG_SEQ` 原样（同一张表要覆盖「PLE 打断点层」与「非 PLE 层」两种层形态，故含 `hc(H1)` 那段）
  = **C++ 侧 `FlagSeqAdjacentOk()` / `FlagMaxUse()` 所见**；
- **层 1 口径** = 去掉只有非 PLE 层才走的 `hc(H1)` 行 ⇒ PLE 打断点层的真实执行序与用量。

**脚本输出（`M124_flagid_table.log`，逐字；`bound` = 挂载点的相位边界，见文末注）**：

```
-- AIC · mode0：22 个同步点
   执行序 id: 0(hc(H1)) → 1(hc(H1)) → 2(hc(H1)) → 3(hc(H1)) → 14(PLE) → 15(PLE) → 0(hc(H1b)) → 1(hc(H1b)) → 2(hc(H1b)) → 3(hc(H1b)) → 12(GDN) → 13(GDN) → 14(GDN) → 15(GDN) → 0(hc(H2)) → 1(hc(H2)) → 2(hc(H2)) → 3(hc(H2)) → 8(MoE) → 9(MoE) → 10(MoE) → 11(MoE)
   每 id 用量（**全表口径**）: 0×3, 1×3, 2×3, 3×3, 8×1, 9×1, 10×1, 11×1, 12×1, 13×1, 14×2, 15×2

-- AIV · mode0：25 个同步点
   执行序 id: 12(hc(H1)) → 13(hc(H1)) → 14(hc(H1)) → 12(hc(H1a)) → 8(bound) → 14(PLE) → 9(bound) → 12(hc(H1b)) → 13(hc(H1b)) → 14(hc(H1b)) → 8(bound) → 10(GDN) → 8(GDN) → 9(GDN) → 11(GDN) → 9(bound) → 12(hc(H2)) → 13(hc(H2)) → 14(hc(H2)) → 8(bound) → 12(MoE) → 13(MoE) → 14(MoE) → 15(MoE) → 12(MoE)
   每 id 用量（**全表口径**）: 8×4, 9×3, 10×1, 11×1, 12×6, 13×4, 14×5, 15×1

-- both · mode2：22 个同步点
   执行序 id: 8(hc(H1)) → 9(hc(H1)) → 10(hc(H1)) → 11(hc(H1)) → 12(PLE) → 13(PLE) → 8(hc(H1b)) → 9(hc(H1b)) → 10(hc(H1b)) → 11(hc(H1b)) → 4(GDN) → 5(GDN) → 6(GDN) → 7(GDN) → 8(hc(H2)) → 9(hc(H2)) → 10(hc(H2)) → 11(hc(H2)) → 0(MoE) → 1(MoE) → 2(MoE) → 3(MoE)
   每 id 用量（**全表口径**）: 0×1, 1×1, 2×1, 3×1, 4×1, 5×1, 6×1, 7×1, 8×3, 9×3, 10×3, 11×3, 12×1, 13×1

== 独立重算的判定 ==
FlagSeqAdjacentOk() 口径：通过（相邻对无同号）
FlagMaxUse() 口径：同一 (核型,mode,id) 的最大用量 = 6（C++ 侧 static_assert 的上限是 6，硬件 4bit 计数器上限 15）

== 层 1（PLE 打断点层）的真实执行序与用量（**层 1 口径**：去掉非 PLE 层才有的 hc(H1) 行）==
   AIV · mode2: 12(PLE) → 13(PLE) → 8(hc(H1b)) → 9(hc(H1b)) → 10(hc(H1b)) → 11(hc(H1b)) → 4(GDN) → 5(GDN) → 6(GDN) → 7(GDN) → 8(hc(H2)) → 9(hc(H2)) → 10(hc(H2)) → 11(hc(H2)) → 0(MoE) → 1(MoE) → 2(MoE) → 3(MoE)
      用量: 0×1, 1×1, 2×1, 3×1, 4×1, 5×1, 6×1, 7×1, 8×2, 9×2, 10×2, 11×2, 12×1, 13×1；相邻对无同号
   AIV · mode0: 12(hc(H1a)) → 8(bound) → 14(PLE) → 9(bound) → 12(hc(H1b)) → 13(hc(H1b)) → 14(hc(H1b)) → 8(bound) → 10(GDN) → 8(GDN) → 9(GDN) → 11(GDN) → 9(bound) → 12(hc(H2)) → 13(hc(H2)) → 14(hc(H2)) → 8(bound) → 12(MoE) → 13(MoE) → 14(MoE) → 15(MoE) → 12(MoE)
      用量: 8×4, 9×3, 10×1, 11×1, 12×5, 13×3, 14×4, 15×1；相邻对无同号
   AIC · mode2: 12(PLE) → 13(PLE) → 8(hc(H1b)) → 9(hc(H1b)) → 10(hc(H1b)) → 11(hc(H1b)) → 4(GDN) → 5(GDN) → 6(GDN) → 7(GDN) → 8(hc(H2)) → 9(hc(H2)) → 10(hc(H2)) → 11(hc(H2)) → 0(MoE) → 1(MoE) → 2(MoE) → 3(MoE)
      用量: 0×1, 1×1, 2×1, 3×1, 4×1, 5×1, 6×1, 7×1, 8×2, 9×2, 10×2, 11×2, 12×1, 13×1；相邻对无同号
   AIC · mode0: 14(PLE) → 15(PLE) → 0(hc(H1b)) → 1(hc(H1b)) → 2(hc(H1b)) → 3(hc(H1b)) → 12(GDN) → 13(GDN) → 14(GDN) → 15(GDN) → 0(hc(H2)) → 1(hc(H2)) → 2(hc(H2)) → 3(hc(H2)) → 8(MoE) → 9(MoE) → 10(MoE) → 11(MoE)
      用量: 0×2, 1×2, 2×2, 3×2, 8×1, 9×1, 10×1, 11×1, 12×1, 13×1, 14×2, 15×2；相邻对无同号
```

**层 1 口径的用量一览（同样取自上面的脚本输出）**：

| 核型 · mode | 层 1 用量（脚本） | 全表用量（脚本） | 说明 |
|---|---|---|---|
| AIV · mode0 | `8×4, 9×3, 10×1, 11×1, 12×5, 13×3, 14×4, 15×1` | `8×4, 9×3, 10×1, 11×1, 12×6, 13×4, 14×5, 15×1` | 差额来自非 PLE 层的 `hc(H1)` 三点（12/13/14 各 +1） |
| AIV · mode2 | `8..11 各 ×2, 12×1, 13×1, 0..7 各 ×1` | `8..11 各 ×3, 12×1, 13×1, 0..7 各 ×1` | 差额来自 `hc(H1)` 的 8/9/10/11 |
| AIC · mode0 | `0..3 各 ×2, 14×2, 15×2, 8..13 各 ×1` | `0..3 各 ×3, 14×2, 15×2, 8..13 各 ×1` | 差额来自 `hc(H1)` 的 0..3 |
| AIC · mode2 | `8..11 各 ×2, 12×1, 13×1, 0..7 各 ×1` | `8..11 各 ×3, 12×1, 13×1, 0..7 各 ×1` | 同上 |

- **相邻性（三处一致）**：`m15_layer_kernel.h` 的逐对 static_assert（AIV mode0：`PLE_IN(8) → B4(14) →
  PLE_OUT(9) → AV0(12)`；AIV/AIC mode2（同一对号、两侧都算）：`GATE(11) → B2(12) → B3(13) → XN(8)`；
  AIC mode0：`AC3(3) → CUBE_IN(14) → CUBE_OUT(15) → AC0(0)`）＋ `FlagSeqAdjacentOk()`（编译期）＋
  脚本（独立重算）——**相邻对无同号**。
- **`FlagMaxUse()` = 6（脚本与 C++ 两侧同值）= 上界**：最大的那个是 **AIV mode0 的 id 12**
  （全表口径 6：hc(H1) + hc(H1a) + hc(H1b) + hc(H2) + MoE×2；层 1 口径 5）。硬件 4bit 计数器上限 15。
- **号从哪来（选号判据）**：`docs/05 §6.1` 明写「每核 16 个 flagId 是 mode0/mode2 **共享**的池」
  ⇒ 选号看「本层该核上这个号有没有被另一 mode 占用」。
  · **AIC mode0 = 14/15**：4–7 是 GDN 的 AIC mode2、0–3 是 hc 的 AIC mode0、8–11 是 hc 的 AIC mode2
    与 MoE 的 AIC mode0、12–15 归 GDN 的 AIC mode0 ⇒ 只有 14/15 不与**另一 mode**同号，
    代价是与 GDN 的 AIC mode0 **同 mode 先后复用**（GDN 的相位 A 在 PLE 段之后，两段都 set/wait 配平）。
  · **mode2 = 12/13（两侧同一对号）**：AIC/AIV 的 mode2 池里 0–3 归 MoE、4–7 归 GDN、8–11 归 hc
    ⇒ 12/13 是这一池里仅剩的可用号。它带来两处**跨 mode 复用**（`docs/05` §2 第 3 条：同 id 跨模式
    复用的前提是前一 mode 的 set/wait 全部 drain）：AIV 侧 mode0 的 12/13 归 hc（H1a 是自封 barrier、
    H1b/H2 各自成对），AIC 侧 mode0 的 12/13 归 GDN（相位 A；段序 hc(H1a) → PLE → hc(H1b) → **GDN** → hc(H2) → MoE）—— 两处都在时间上
    **不与 PLE 段重叠**，且每处 set/wait 成对 ⇒ 计数在段间回到 0。
  · **`FLAG_B4`（AIV mode0、id 14）**：`reuse=true`（复用 hc 的 AV2），位置在 `8(PLE_IN)` 与 `9(PLE_OUT)`
    之间 —— 与 `m15_layer_kernel.h` 的两条 static_assert 一致；**r2 复审前它没登记进 `FLAG_SEQ`**，已补。

**登记面订正的「代码生成中性」见证**：补 `FLAG_B4` 行 + r1 的 `core="both"` 订正之后整体重建，
`m15_layer_loop` 的二进制 sha256 **仍是 `9447734a…`**（= r1/r2 复审在全新目录重建得到的值），
`m15_ple` 仍是 `33c8b0f1…`（该 target 不含 `m15_layer_resources.h`）⇒ 这轮只落在**编译期登记/注释**上，
**未进入编译产物**（因此谱系/归属表的二进制 sha 不需要更新）。

注：脚本里的 `bound` = PLE 挂载点的相位边界（`FLAG_PLE_IN/OUT_BOUND_AIV` = `FLAG_HC0/HC1_BOUND_AIV` = 8/9）；
层 1 序里出现的另外三处 `8/9/8`（`H1→A` / `A→H2` / `H2→B`）在表里同样以 `bound` 记（同号复用，
见 `m15_layer_kernel.h` 的注释）。

## 5. 判据

### 5.1 分档与界（`docs/17 §1.1`）

| 判据 | 档 | 界 | 为什么是这一档 |
|---|---|---|---|
| `B3.kv`（③ 的主判据）/ `H3.kv` | **T3** | **`EPS_MMAD·Σ\|terms\| + 1.0·ulp(out)`，`EPS_MMAD = 2560·2⁻²⁴`** | 触发条件 ③「含 mmad / cube 累加」；`k·2⁻²⁴` 是 `docs/17 §1.1` 逐字点名的 mmad 项（`k` = 归约长度 = `HE` = 2560）。**改前也算这一条**：`EPS_MMAD` 盖住旧的 VF 界 `96·2⁻²⁴` ⇒ 两版共用同一条界，判据没有随算法漂移 |
| `B4.gated` / `B4.normed` / `B5.out` / `B5.state` | T3 | 不变（`EPS_REDUCE + EPS_RSQRT + EPS_SIGMOID` 等） | ④⑤ 的运算没变；③ 的 1 格点差进 ④ 后由 `1.0·ulp(out)` 项覆盖（读数上 `max(d/bound)=0.000`，见 §6.2） |

### 5.2 判据**没有**放宽的证据

- `B3.kv` 的 ε 从 `5.72e-6` 变成 `1.526e-4` **是变宽**，但它对应的是**换了的算法**（cube 累加）：
  按 `docs/17 §1.1` 的护栏「T3 的界必须推导」逐项列出来源（`k·2⁻²⁴`，k=2560）。
  两版都在**同一条界**下 `bad=0/25600`（§6.2 的两个读数）。
  ⚠ **本 tip 的两次读数都是在 `EPS_MMAD` 下采的**（`M124_check_{base,new}.log` 都印 `eps=0.000153`）；
  **没有**在旧界 `EPS_REDUCE = 96·2⁻²⁴` 下重采过 kv 读数（r1 复审 P2-4 指出初版「旧界下也 0 越界」这句
  缺归档读数，已按此收窄）。可以确定的只有：**M85/M111 时期**（③ 还是 AIV 逐列 GEMV）的 kv 判据就是用
  `EPS_REDUCE` 判过的（`ple/logs/` 与 `M111_LANDING_PATH.md` 的归档），那两个读数不能直接搬到本 tip 的
  cube 实现上；若要「同一批 dump 在两个界下各判一次」的读数，需要另采（本 tip 未采）。
- ④⑤ 的界一字未动；`B2.emb`（②）仍是 **T1 逐字节**。

### 5.3 负向对照（咬合力）

1. **变异 bit7**（③ 的 key/value 输出块对调，cube 档形态 = N-tile 读取基址 ±HID）⇒ **`B3.kv` FAIL**：
   `RESULT|mut|7|D1|B3.kv|B3.kv|OK`。完整 15 位矩阵：`RESULT|coverage|14|14|(none)|OK`、`rc=0`
   （基线 14/14 PASS；每一位都按期望变红）。
2. **生成器↔产物双向交叉对拍**（§6.6）：2×2 四条腿 + 手改产物的负向对照。
3. **层路径「③ 真的被生产」**：见 §6.4 的 `Pw.*` 读数（`Pw.wired.nonvac` / `Pw.wired.delta` 对
   「kv 无生产者」是**响亮失败**：没有 kv ⇒ ④ 的 `gated` 全 0 ⇒ PLE 不再改动多流态 ⇒ 该判据红）。
   **kv 的数值判据**要落 `m15_layer_loop.asc`（本 tip 的 scope 不含该文件）⇒ 见 §8 第 1 条。

## 6. 读数

设备档一律：`flock -w 300 /tmp/npu0.lock` 内、**单进程**、`timeout`、**进锁后先读设备状态**。
本批全部在同一台 NPU 0（Ascend950PR）上，未观察到 `aicore exception` 一类的设备错误报告
（每档进锁后的 `npu-smi` 见对应的 `M124_device_*.log`）。

### 6.1 独立 kernel：真 shape（③ 的 `K=2560`、`N=12800`，权重 = 真 checkpoint 切片 65.5 MB）

| 档 | 二进制 | 形状 | 结果 | `B3.kv`（**同一条判据、同一条界** `EPS_MMAD=1.526e-4`） | `B_kv.bin` sha256 |
|---|---|---|---|---|---|
| **改前**（`9296e79` 源码重建） | `sha256 40e66c7d…` | `n_tok=2` | `rc=0`、`ALL PASS`（14/14） | `bad=0/25600`，`maxRel=0.00719`，`max(d/bound)=0.979` | `f07fadeb…` |
| **改后**（本 tip） | `sha256 33c8b0f1…`（＝ `M124_sha256.txt` 的归档值；r1 复审在全新 `/tmp/rev124_build` 独立重建得同一值） | `n_tok=2` | `rc=0`、`ALL PASS`（14/14） | `bad=0/25600`，`maxRel=0.00637`，`max(d/bound)=0.976` | `b88e108d…` |
| **改后**（本 tip） | 同上 | **`n_tok=1`（decode 的真实 M=1）** | `rc=0`、13/13 PASS（`B5.state_evolve` 本档**不可判**并如实 `[SKIP]`，见 §5.3/§8） | `bad=0/12800`，`maxRel=0.00571`，`max(d/bound)=0.974` | 见 `M124_check_new_ntok1.log` |

- 「同一判据」的落实：`m15_ple_check.py` 的 `B3.kv` 判据**在改前与改后是同一条**（同 numpy 参考、同分档 T3、
  同 `EPS_MMAD` 界），两侧读数见 `M124_check_{base,new}.log`。
- **M=1 档的证据链对账（r1 复审 P2-3）** —— 归档目录里有 **3** 个与 M=1 有关的日志，逐条说清：
  1. `M124_device_new_ntok1_attempt1_h2dfail.log`：**第一次**尝试（08:06:32–08:06:40）。设备段 `run_rc=1`：
     `[m15ple][FAIL] body: H2D` —— 旧 harness 把 `stoD`/`embD` 按 `n_tok` 行定尺（`n_tok=1` 时 `stoD` 比
     上传的 `stH` 小、`embD` 也小于 ③ 要读的 2 行）⇒ **加 M=1 档所暴露的 harness 缺陷**，已在
     `m15_ple.asc` 的 `RunBody` 按「设备真实读取范围」定尺修掉（另 `RunBodyMapped` 同形加固）。
     该档的 `check_rc=0` 与「4 条判据」是因为 body 没跑起来、只有 ① 的产物可判。
  2. `M124_device_new_ntok1.log`：**第二次**尝试（08:07:39–08:07:48）。`run_rc=0`、`n_tok=1`、设备侧
     `kernel-side OK`；同一行里的 **`check_rc=1` 是当时的 `m15_ple_check.py` 在 T4 行崩掉**
     （`B5.state_evolve` 里 `act` 全 null 行时是空列表、被 numpy 当成 float64 索引：
     `IndexError: arrays used as indices must be of integer (or boolean) type`）⇒ 那次只跑到
     `B2.emb`/`B3.kv` 两行（该设备日志里打印的就是这两行）。**随后**在 `m15_ple_check.py` 里修了两点
     （`act` 显式 `int64`；`act` 为空时**不判 PASS** 而是如实 `[SKIP]`「无活跃行 ⇒ 本条不可判」），
     **在同一个 dump（`/tmp/m124_out_new_m1`）上复跑** ⇒ 13 条判定项、FAIL 0、`ALL PASS`（rc=0）。
  3. `M124_check_new_ntok1.log`：就是上面那次**复跑**的完整输出（末尾逐字 `判据合计 13 条 = 判定项 13 +
     对照项 0；FAIL 0 条` + `ALL PASS`）。⇒ 报告里的「M=1 档 13/13」引的是**它**（来源唯一）。
  ⚠ 一句话口径：**M=1 的设备段读数（`run_rc=0`、`n_tok=1`、`B2.emb`/`B3.kv` PASS）取自第二次尝试**；
  **13/13 是修好判据脚本后在同一 dump 上复跑的读数**（不是设备重跑）。
- **改前的 M=1 读数取不到**：旧 harness 的 body 档把 `n_tok` 写死为 2（`const uint32_t nTok = 2u;`），
  没有 env 入口；`M15_PLE_NTOK` 是本 mission 新加的。为了不破坏「改前 = 冻结的基线」这条性质，
  **没有**给旧树补同一个 knob ⇒ 改前的 M=1 只能算「未取到」（见 §8）。
- `docs/20 §4 WO-A1` 的第 ① 条判据（「A/B dump 锁住现值、换实现后要求逐字节相同」）**不成立**：
  改前/改后的 kv 平面**不是逐字节相同** ⇒ 按同一条的降级条款走 T3（界已推导），见 §6.2。
- `m15_ple.asc` 在区间内**改过**（`09996f61…` → `a3e29fdd…`），生成器**未改**（`41ac4ba8…` 两侧相同）。

### 6.2 改前/改后逐平面差异（同一输入、同一判据脚本；`M124_plane_diff.{py,log}`）

| 平面 | 元素数 | 不同元素 | 占比 | max\|Δ\| | max 格点距 | >1 bf16 格点 |
|---|---|---|---|---|---|---|
| `B_ids_body`（①→② 的输入） | 128 | 0 | 0% | 0 | — | 0 |
| `B_emb`（②，未改） | 5120 | 0 | 0% | 0 | — | 0 |
| **`B_kv`（③，本次改的就是它）** | 25600 | **9** | **0.035%** | 6.1e-5 | **1** | **0** |
| `B_gated`（④） | 20480 | 15 | 0.073% | 7.6e-6 | 2 | 2 |
| `B_normed`（④） | 20480 | 13 | 0.063% | 2.4e-4 | 2 | 3 |
| `B_out`（⑤ 的最终输出） | 20480 | **0** | **0%** | 0 | — | 0 |
| `B_state_out`（⑤ 的状态） | 184320 | 3 | 0.002% | 4.8e-7 | 1 | 0 |

读法：**舍入点（fp32 累加 → RNE 落 bf16）没变**，变的是 fp32 累加的次序 ⇒ 只有 9/25600 个 kv 元素
翻 1 个 bf16 格点，且**最终层输出 `out` 逐字节相同**。④ 的 gated/normed 有 4 个元素翻到 2 格点，
仍在两条 T3 判据的界内（`bad=0`，`max(d/bound)=0.000`）—— 这正好把 `ple/README.md §4.1` 里
「③ 的误差会被 ④ 的参考继承」那句残差风险**量化**了（继承确实存在，量级 = 0.07% 的元素、≤2 格点）。

### 6.3 变异矩阵（咬合力；`M124_mutants.log`）

- 命令：`flock -w 300 /tmp/npu0.lock … python3.12 m15_layer_loop/m15_ple_mutants.py`（**一次锁内 16 次设备运行**，
  08:01:53 → 08:03:50，**锁持有时长 117 s**、单进程、host RSS 量级与既有 PLE 档相同）。
- 读数：基线 `14 条判据 / FAIL 0`；15 个变异位**全部**按期望变红；
  `RESULT|coverage|14|14|(none)|OK`（每一条基线判据都被某个变异演示过）；`rc=0`。
- 与 ③ 直接相关的那一位：**`RESULT|mut|7|D1|B3.kv|B3.kv|OK`**
  （③ 的 key/value 输出块对调 = cube 档的「N-tile 读取基址 ±HID」⇒ `B3.kv` 变红）。
- **判据脚本改过之后的重算**：`m15_ple_check.py` 在本批之后又加了一处 T4 的加固（`act` 空集时的
  dtype / 不可判处理），为免归档读数与脚本错位，我用**当前**脚本在既有 `ple/out_mut_*`（落盘仍在）
  上逐位重算了一遍 FAIL 集合 —— 与 `M124_mutants.log` 里逐位**完全一致**（15/15 位的 FAIL 集合逐项相同）。

### 6.4 层路径（`plewire` 档：`m15_layer_kernel.h` 的 AIC 挂载点 + AIV 握手）

| 档 | env | 结果 | `Pw.*` 读数 | rc |
|---|---|---|---|---|
| **A**：②③④⑤（ids 由 host 灌入；判据 10 条时） | `M15_PLE_WIRE=1 M15_PLE_STAGE=15` | 10/10 PASS | `Pw.wired.delta = 5636/10240`、`max\|Δ\|=0.427734`、`ALL PASS（checks=142, guards=49, fails=0）` | 0 |
| **A′**：同 A，**加守卫的二进制** | 同上 | 10/10 PASS | 与 A **逐行相同**（`diff` 空） | 0 |
| **B′**：同 A，**最终判据（12 条）** | 同上 | 12/12 PASS | 新增 **`Pw.kv.T3` PASS**：`n=12800 bad=0 maxAbs=1.76e-4 maxRel=5.31e-3`（ε=k·2⁻²⁴=1.53e-4）；`Pw.kv.nonvac` PASS（参照面非零 12800/12800） | 0 |
| **B**：①+②③④⑤（最终判据） | `M15_PLE_WIRE=1 M15_PLE_STAGE=31` | 12/12 PASS | 与 B′ 的 `Pw.kv.*` 读数逐字相同；`Pw.wired.delta = 5636/10240`、`ALL PASS（checks=144, guards=49, fails=0）` | 0 |
| **C**：接线关（负向对照） | `M15_PLE_WIRE=0 M15_PLE_STAGE=15` | 5 FAIL | 与 M111 归档的 C 档同 5 条、`Pw.wired.delta = 0/10240` | 1 |
| **D**：**③ 不跑**（`STAGE=13` = ②④⑤）—— 「**kv 无生产者**」的负向对照 | `M15_PLE_WIRE=1 M15_PLE_STAGE=13` | **4 FAIL** | **`Pw.kv.T3` FAIL**（`bad=12753/12800`、`maxAbs=8.97e-2`、参考非零而设备平面全零）；`Pw.wired.nonvac`/`Pw.planes.nonvac`/`Pw.state.evolve` FAIL；`Pw.wired.delta = 0/10240` | 1 |

- **A/B 的 `Pw.wired.delta` 与 M111 归档读数逐行相同**（`M124_wire_A_vs_M111.diff.log`）：10 条 Pw 判定行 +
  delta 读数完全一致，唯一差异是 `判据分账` 行里的 `+ Pf 0`（M110 晚于 M111 合入的**计数器**，与本改动无关）。
  ⇒ **③ 换成 cube 之后，层路径的多流态改写量一模一样**（5636/10240、max|Δ|=0.427734）。
- **D 档的意义（约束 #4 的「间接」那一半）**：`Pw.wired.delta = 0` + `Pw.wired.nonvac`/`Pw.planes.nonvac` 变红
  ⇒ 「③ 没有生产者」在层路径上是**响亮失败**，不是静默错值。
- **`Pw.kv.T3`（M124 新增，塔放宽 scope 后落进 `m15_layer_loop.asc`）** —— 层路径上 ③ 的**数值**判据：
  参考**完全取自 host**（窗口字节 `C.pleWinHost` + host 的 ids 参考 `refIds` 重建 emb；再与 host 侧
  `wcat`（`wslab` 的 `W_CAT` 段，与设备读的是同一批字节）在 double 下点乘），界 = `ε·Σ|terms| + 1.0·ulp(out)`，
  `ε = k·2⁻²⁴`（k=2560，与独立 kernel 的 `B3.kv` 同一条推导式）；只判「16 个 head 全在窗口内」的 token。
  配 `Pw.kv.nonvac`（参照面非零 + 参与行 > 0）作非空洞守卫。
  **咬合力 = D 档**（③ 不跑 ⇒ 平面全零 ⇒ `bad=12753/12800`、`maxAbs=8.97e-2` ⇒ 判据红）。
- 其余既有 `Pw.*` 的口径本 mission 未动：`Pw.wired.delta` 是**读数**不是判据；`Pw.det` 是自比
  （两次接线跑逐字节一致），**不构成正确性证据**。

### 6.5 零回归：`runs=all`（接线默认关）

| 二进制 | 命令 | 末行 | rc |
|---|---|---|---|
| 改后（本 tip，接线+挂载点） | `M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 ./build/m15_layer_loop m15_layer_loop/weights_manifest.txt all` | `===== ALL PASS（checks=2095, guards=302, fails=0）=====` | 0 |
| 改后（**再加 `Pw.kv.T3` 之后**；`M124_runs_all_after_kv.log`） | 同上 | `===== ALL PASS（checks=2095, guards=302, fails=0）=====`（判据分账行逐字不变，`Pw 0`） | 0 |

⇒ 与 `docs/20`/M111 记录的当前 main 基线 **逐字相同**（2095 + 302 / 0 FAIL）⇒ 本改动（含
`m15_layer_kernel.h` 的 AIC 分支与 `m15_layer_resources.h` 的登记）**没有移动这个计数**
（`plewire` 档之外的路径上 `A.pleW == nullptr` ⇒ 新增代码是死代码）。

**锁纪律（逐档）**：A/B/C/D 与 `runs=all` 全部在 `flock -w 300/900 /tmp/npu0.lock` 内、**单进程**、带 `timeout`，
进锁后先读 `npu-smi`（见 `M124_device_*.log` 与各档日志首行）。`runs=all` 与 A′ 合并在**同一次锁**里跑
（08:15:15 → 08:16:07，**52 s**）；变异矩阵一次锁内 16 次设备运行（117 s）；其余各档每次独占一次锁、30–60 s。

### 6.6 生成器 ↔ 产物双向交叉对拍（`M124_generator_xcheck.{sh,log}`；形态照 M105 已受审先例）

| 腿 | 源 sha256（前 8） | 生成器 | 产物 sha256 | `--check` |
|---|---|---|---|---|
| oldgen + oldsrc | `09996f61` | `41ac4ba8` | `8bbd5e03…`（= `9296e79:m15_ple_wire.h` 归档值） | rc=0 |
| **oldgen + newsrc** | `a3e29fdd` | `41ac4ba8` | `55bba435…`（= 本 tip 归档值） | rc=0 |
| **newgen + oldsrc** | `09996f61` | `41ac4ba8` | `8bbd5e03…`（= 旧归档值） | rc=0 |
| newgen + newsrc | `a3e29fdd` | `41ac4ba8` | `55bba435…`（= 新归档值） | rc=0 |
| 负向对照：手改产物一个 token | — | — | `[FAIL] … 不一致（漂移）—— 重新生成` | **rc=1** |

⇒ 产物**只是源（`namespace M85P` 那一刀）的函数**；本 mission 的产物差异完全由 `m15_ple.asc` 产生。
注：生成器在区间内**未改** ⇒ 上表的「newgen」腿与「oldgen」腿同值（这一维不可分，如实标注）；
负向对照证明「产物非手改」这一半仍然成立。另：本批最后一次 host 侧改动（`RunBody` 的缓冲定尺）
部分落在 `namespace M85P` **之外**（`RunBody` 的缓冲定尺）、部分**之内**（`PleCubeGemv` 的 `nTok==0` 早退）
⇒ 产物 sha256 相应地从 `55416764…` 变成 `55bba435…`；本表的四条腿是**以最终源**复跑的读数。
## 7. 复现命令（逐条；设备档都在锁内、单进程）

```bash
# 0. 环境与构建（仓库根）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m15_layer_loop/build -S m15_layer_loop -DCMAKE_BUILD_TYPE=Release
cmake --build m15_layer_loop/build -j4 --target m15_ple          # 独立 kernel（含 ③ 的 cube 段）
cmake --build m15_layer_loop/build -j4 --target m15_layer_loop   # 层路径（含 AIC 挂载点）

# 1. 数据（真 checkpoint 的 PLE 权重：wcat = key‖value = 12800×2560 bf16）
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/gen_ple_data.py

# 2. 生成器通路（改 .asc 之后必须重跑；--check 必须 rc=0）
python3 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py
python3 m15_layer_loop/evidence/ple_wire/lift_ple_device_segment.py --check

# 3. 独立 kernel 设备档（③ 的真 shape K=2560 / N=12800）
flock -w 300 /tmp/npu0.lock bash -c '
  npu-smi info | sed -n "/Process id/,\$p" | tail -2        # 进锁后先读设备状态
  timeout 110 env M15_PLE_DATA=m15_layer_loop/ple/data M15_PLE_OUT=/tmp/out_new ./m15_layer_loop/build/m15_ple'
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_check.py /tmp/out_new   # 14/14 PASS

# 3b. decode 的 M=1 档（本 mission 新增的 env）
flock -w 300 /tmp/npu0.lock bash -c '
  timeout 110 env M15_PLE_NTOK=1 M15_PLE_DATA=m15_layer_loop/ple/data M15_PLE_OUT=/tmp/out_m1 ./m15_layer_loop/build/m15_ple'
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_check.py /tmp/out_m1    # 13/13 PASS（T4 state_evolve 不可判 ⇒ [SKIP]）

# 3c. 改前基线（**冻结的旧源码**，不改 knob）：在 /tmp 里从基线 rev 重建
mkdir -p /tmp/m124_base && git archive 9296e79 m15_layer_loop | tar -x -C /tmp/m124_base
( cd /tmp/m124_base/m15_layer_loop && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 --target m15_ple )
flock -w 300 /tmp/npu0.lock bash -c '
  timeout 110 env M15_PLE_DATA=<worktree>/m15_layer_loop/ple/data M15_PLE_OUT=/tmp/out_base /tmp/m124_base/m15_layer_loop/build/m15_ple'
/usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_check.py /tmp/out_base  # 14/14 PASS（同一条判据/同一条界）

# 4. 变异矩阵（16 次设备运行；本批一次锁内 117 s、单进程）
flock -w 300 /tmp/npu0.lock bash -c 'timeout 280 /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_mutants.py'

# 5. 层路径（接线开 / 关；`M15_PLE_STAGE=15` = ②③④⑤，`31` = ①+②③④⑤）
flock -w 300 /tmp/npu0.lock bash -c 'timeout 150 env M15_PLE_WIRE=1 M15_PLE_STAGE=15 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt plewire'
flock -w 300 /tmp/npu0.lock bash -c 'timeout 150 env M15_PLE_WIRE=1 M15_PLE_STAGE=31 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt plewire'
flock -w 300 /tmp/npu0.lock bash -c 'timeout 150 env M15_PLE_WIRE=0 M15_PLE_STAGE=15 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt plewire'   # 负向对照

# 6. 零回归（接线默认关）
flock -w 300 /tmp/npu0.lock bash -c 'M15_LAYERS=48 M15_STEPS=3 M15_HC_LAYERS=0,1,3 ./m15_layer_loop/build/m15_layer_loop m15_layer_loop/weights_manifest.txt all'

# 7. 生成器↔产物双向交叉对拍（含负向对照）
bash m15_layer_loop/evidence/ple_wire/m124/M124_generator_xcheck.sh          # 输出即归档日志
python3.12 m15_layer_loop/evidence/ple_wire/m124/M124_plane_diff.py /tmp/out_base /tmp/out_new   # 逐平面差异
```

**本文件里的日志 → 命令的对应**：

| 证据文件 | 来源命令 |
|---|---|
| `M124_device_base.log` / `M124_device_new.log` / `M124_device_new_ntok1.log` | §7 的 3 / 3b / 3c（锁内的 `npu-smi` + kernel stdout） |
| `M124_check_base.log` / `M124_check_new.log` / `M124_check_new_ntok1.log` | §7 的 3 / 3b / 3c 之后的 `m15_ple_check.py` |
| `M124_plane_diff.log` | §7 的 7（`M124_plane_diff.py`） |
| `M124_mutants.log` | §7 的 4 |
| `M124_generator_xcheck.log` | §7 的 7（`M124_generator_xcheck.sh`） |
| `M124_sha256.txt` | 源码 / 产物 / 两个二进制 / 两个 kv 平面的 sha256 |
| `M124_wire_{A,A_final_bin,B,C}.log` | §7 的 5（判据 10 条时的那一批） |
| `M124_wire_A_kv.log` / `M124_wire_B_kv.log` / `M124_wire_stage13_nokv_kv.log` | §7 的 5（**加 `Pw.kv.T3` 之后**：A/B 档 PASS、D 档 `Pw.kv.T3` FAIL 的负向对照） |
| `M124_runs_all.log` / `M124_runs_all_after_kv.log` | §7 的 6（零回归；后者是加 `Pw.kv.T3` 之后的那次） |
| `M124_wire_A_vs_M111.diff.log` | A 档与 M111 归档读数的逐行 diff（唯一差异 = `Pf 0` 计数器） |
| `M124_flagid_table.py` / `M124_flagid_table.log` | flagId 逐 (核型, mode) 执行序与用量表（**从 `FLAG_SEQ` 解析**，非手抄；r1 复审 P2-1 的落点） |
| `M124_device_new_ntok1_attempt1_h2dfail.log` | M=1 档**第一次**尝试（设备段 `body: H2D` 失败；harness 定尺缺陷，见 §6.1 的对账） |
## 8. 没做完 / 没取到读数（逐条，不用「已全部 / 无残留 / 0 命中」这类断言收口）

| # | 项 | 卡在哪 / 为什么 |
|---|---|---|
| ~~1~~ | ~~层路径 kv 的数值判据~~ | **已完成**（塔批复了 `m15_layer_loop/m15_layer_loop.asc` 的 scope）：新增 `PleRunOut::kv` + `H_PleReadBack` 的 kv 回读 + **`Pw.kv.T3`**（host-only 参考、T3 推导界、非空洞守卫）+ 负向对照 D 档。读数见 §6.4。**遗留**：该判据只在**窗口内的 token** 上可判（`参与行=1/1`，本 tip 的档只有 1 个 token 全在窗口内）⇒ 覆盖面受窗口大小限制（与 `Pw.emb.T1` 同一限度）。 |
| 2 | **设备侧计时**（「cube 比 VF 快多少」） | 没做 msprof / 计时采集。理由：共享卡墙钟不构成证据（`docs/17 §9.1`），且本 mission 没有开 msprof 窗口。**本 tip 没有任何「快了 N 倍」的读数**（结构上的论据只是「权重读一遍 65.5 MB、摊在 28 条 AIC 上」）。 |
| 3 | **改前的 M=1 读数** | 旧 harness 的 body 档把 `n_tok` 写死为 2，没有 env 入口；为保「改前 = 冻结基线」的性质**没有**给旧树补 knob（见 §6.1）。 |
| 4 | **M 轴 > 2 / 多请求 / 多 token chunk** | 未铺（`ple/README.md §7 U5` 的老缺口）。`n_tok=1` 与 `n_tok=2` 两档都有设备读数；`BASE_M=64` 的上界只在代码里论断（同一条 `calcMAlign`/Fixpipe `mSize` 路径），**没有 64 行的设备读数**。 |
| 5 | **N 条带负载不均** | `80 个 tile / 28 条 AIC` ⇒ 24 条核 3 个 tile、4 条核 2 个 tile（未做均衡/未换 `BASE_N` 试）。`nAic` 是设备常量，没跑过别的档。 |
| 6 | **L0/L1 残留面** | 本段每个 tile 都写满 `BASE_N×BASE_K`（无 NOP/越界装载），`Pw.det` 与两次重跑逐字节一致；但**没有**专门做「L0B 残留 × 本段」的对照实验（`docs/05 §6.2` 那条实测的适用范围未经本 mission 复核）。 |
| 7 | **`m15_layer_loop.asc` 的 `Pw.*` 判据本身** | 既有 10 条一字未动；本 mission 按塔放宽 scope **新增 2 条**（`Pw.kv.T3`/`Pw.kv.nonvac`，见 §2 改动清单的 `m15_layer_loop.asc` 行）⇒ 层路径最终 **12 条** Pw 判定项（读数见 §6.4）。 |
| 8 | **M92 host-mapped 档（`ple/data_hm`）** | 本机没有那份数据 ⇒ 未复跑；只对 `RunBodyMapped` 的 `embD` 定尺做了与 `RunBody` 同形的加固（`n_tok=1` 时可读 2 行），**该加固未在设备上验证**（如实标注）。同理，变异矩阵的 M92 段（`HM_MUTANTS`）在本批**SKIP**。 |
| 9 | **`n_tok=1` 档的 `B5.state_evolve`** | 该档唯一 token 是 null 槽位 ⇒ 无活跃行 ⇒ 本条**不可判**，脚本如实 `[SKIP]`（**不判 PASS**）。该档的判据数是 13 而不是 14。 |
| 10 | **③ 的 `M≥3` 与「多 tile 的 M 分块」** | donor 原本有 `mBlock` 循环，本段为了「一个 A 大包吃全部 token」把它折掉了（`n_tok ≤ 64` 单 tile）。若将来 `pleTok > 64`，需要把 `mBlock` 循环加回来 —— 代码里只有注释契约，没有编译期断言（`BASE_M` 与 `T_MAX` 都是 64，靠 host 契约保证）。 |
| 11 | **层路径的 kv 数值判据之外的** `Pw.*` 口径 | `Pw.wired.delta` 是「元素数」读数、不是判据；`Pw.det` 是自比（两次运行逐字节一致），**不构成正确性证据**。这些既有口径本 mission 未改，如实登记。 |

## 9. 与任务书逐条对账

| 任务书条目 | 状态 | 落点 |
|---|---|---|
| 读 `docs/20` 的 §2.2-V1 / §3.3 / §4（WO-A1、WO-B2）并按判据办事 | ✅ | 本文件 §1–§5（口径逐条引用） |
| **WO-A1**：`PleGemv` 从 AIV GEMV 改成 cube `Mmad`（bf16 操作数），**两份副本都改**（改 `.asc` + 重跑生成器，`--check` rc=0） | ✅ | `m15_ple.asc` 的 `PleCubeGemv`/`PleGemv`；`m15_ple_wire.h` 由生成器重出，`--check` rc=0（§6.6） |
| WO-A1 的操作数布局：优先复用仓内既有 bf16 mmad 先例与权重既有排布；要转换就报塔 | ✅ | 复用 `m15_gdn_layer.h:Bf16Gemm` 形态；`wcat` 的 `[12800,2560]` 行主序**直接**当 donor 的 `B[N,K]`，**零在线转换**（§3） |
| **WO-B2**：PLE 的标量写 GM 改成合规落盘通路 + 处置 `dev_fail` 死分支 | ✅（**M111 已落，本 mission 只复核**） | 本 tip 的设备段**已无任何 GM 标量 `SetValue`**（`grep` 只剩 3 处 UB 写 `idsL`/`sinkL`/`failL`，都走 `BufAcquire<PIPE_S>` → 写 → `BufRelease<PIPE_S>` drain → `DataCopy`）；`dev_fail` 死分支已由 M111 换成 `PleGather` 的真计数（`B_dev.fail` + 变异 bit15 变红） |
| **N4**：`PleGateItem` 必须 VF（不上 cube） | ✅（**核查后：本来就是 VF**，无需改） | `PleGateItem` 的五个 pass 全在 `__VEC_SCOPE__` 内用 RegTensor（`Mul`/`Add`/`Reduce`/`SigmoidReg`），**没有标量算式**；本 mission 未动它（改动清单 §2 亦未列） |
| 真 shape 验证：用真实 shape、给出**修改前后同一判据**的设备读数（锁内单进程 + `timeout` + 先读设备错误报告） | ✅ | §6.1（`n_tok=2` 改前/改后 + `n_tok=1`）、§7 的第 3/3b/3c 条 |
| 判据可复现性：生成器↔产物**双向交叉对拍**（M105 先例）+ 变异体确实变红的负向对照 | ✅ | §6.6（2×2 + 手改产物负向对照）、§6.3（15 位矩阵，含 bit7） |
| 把改后源文件的 sha256 与设备档日志归档进 `evidence/ple_wire/**`，README/证据里写清「改了什么、判据是什么、读数是什么」 | ✅ | `evidence/ple_wire/m124/**`（见 §7 的清单）、`ple/README.md §0.2`、本文件 |
| `git diff --name-status main...HEAD` 全部落在 scope 内；**不得**改 `CMakeLists.txt` | ✅ | 改动 **8** 个源码/文档文件全在 scope（原 6 项 + 塔放宽的 2 个），外加 `evidence/ple_wire/m124/**` 的 29 个证据文件；`CMakeLists.txt` 未动（`git diff --name-status 9296e79..HEAD \| grep -c CMakeLists` = 0） |
| 报告里明写哪些做完 / 没做完 / 哪些读数没取到 | ✅ | §8（11 条） + §9（本表） |
| **塔放宽 scope 后的四条硬约束** | ✅ | ① 形态照先例（§4.1）；② mode 判据是「谁等谁」（§4.1）；③ AIC flagId 预算重核 + 相邻性 + 逐核用量表（§4.2）；④ **kv 的数值判据 `Pw.kv.T3` 已落**（host-only 参考、T3 推导界、非空洞守卫、③ 不跑的负向对照变红）⇒ §6.4 |
| 塔对「kv 数值判据」的五条要求 | ✅ | ① 参考取自 host（窗口字节 + host ids 参考重建 emb；`wslab` 的 wcat），不取设备产物；② 界=推导式 `k·2⁻²⁴`（k=2560），不用「逐字节」当判据；③ 负向对照 D 档（③ 不跑）⇒ `Pw.kv.T3` **FAIL**（`bad=12753/12800`）；④ 非空洞守卫 `Pw.kv.nonvac`；⑤ `m15_layer_loop.asc` 只做 PLE 判据相关的局部改动：既有 `runs=all` 判据行/`Pw.det` 的六项/其它块**一字未动**，`runs=all` 的判据分账行**逐字不变**（2095 + 302 / 0 FAIL） |
| 9/25600 的 kv 格点差按 T3 如实记（三个数一起写）+ sha256 改了就是改了 | ✅ | §6.1/§6.2（9/25600、0.035%、最大 1 个 bf16 格点、>1 ulp 者 0 个；`f07fadeb… → b88e108d…`） |
| 两条编译期发现落到证据/README | ✅ | §4.1（`ffts_cross_core_sync` 的 pipe 域 `[2,5]∪{10}` 逐字报错）、§4.2（AIC mode0 取 14/15 的共享池依据 + 相邻性/用量读数）；两处也写进了 `m15_ple.asc` / `m15_layer_kernel.h` 的注释 |
