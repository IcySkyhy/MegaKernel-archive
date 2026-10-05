# M23 —— M115 / Wave B1：GDN prefill chunk 扫描段（mmad 版）的独立验证路 + 融合清单

> **状态（2026-10-04 更新，逐条如实；历史过程见 `evidence/logs/` 与 `HANDOFF_R1.md`）**
> * 设备段 `m15_layer_loop/m15_gdn_prefill.h` **编译通过**；独立验证工程（本目录）**编译并链接通过**。
> * **内核两处缺陷已修，并已被独立复跑验证**：
>   ① `GdnPrefillAic::DoMmad` 缺 **L0C 的 `M↔FIXP` 成对交接**（buffer id `GP_BUF_L0C`，阻塞释放（`false` = CANN `ASC_LOCK_BLOCK` 默认），未用
>   `SetFlag/WaitFlag`）：修复前 `m=4097` 稳定复现 `507015` + `mte error info: 0x13d1…0202ce`（设备 `errStr` =
>   `A multi-bit ECC error occurs when fixpipe reads L0C`），修复后该档不再报错；
>   ② `GdnPrefillAiv::StoreState` 的 `TransposeVF` **R/C 与源 stride 不符**（布局 bug）：修复前 `ht`
>   超界 778,845/786,432，修复后 **0 超界**。
> * **判据（新鲜档，`check_ref.py` rc=0）**：`m1`：`o` **0/6144**（max|Δ| 7.655e-09）、`ht` **0/786432**（max|Δ| 3.313e-08）；
>   `m4097`：`o` **0/25171968**（max|Δ| 3.042e-08）、`ht` **0/786432**（max|Δ| 1.458e-07）。
> * **反向对照（判据能否咬住）已取得红读数**：`--mutate 1` ⇒ `m1_mut1` / `p4097_mut1` 双双 **FAIL**，脚本打印
>   「如预期变红 ✓」（若反而 PASS 会打印「判据没咬住 ✗」并 `rc=1`）。
> * **读数与二进制的对应**：`evidence/logs/fix2_*`（新鲜判据 PASS）来自 commit `51be119` 构建的二进制（内核两处修复、
>   判据侧未改）；`evidence/logs/r3_*`（新鲜判据 PASS + 反向对照红 + 守卫触发）来自 commit `461bfe0` 构建的二进制
>   （判据侧返工后）。当前 `build/m23_gdn_prefill` 的 `sha256` 前 24 位 = **`cf001d6af43a00c791f3dfce`**
>   （二进制不入库，故以源码 commit 为锚）。
> * **参考侧数值稳定化（M165，2026-10-04，零设备）**：判据的 fp64 参考把 Γ 从旧式
>   `exp(ĝ)·exp(−ĝ)` 改成**指数差** `exp(ĝ_i−ĝ_j)`（与官方 FLA 一致）⇒ chunk 内 `|ĝ|` 超
>   fp32/fp64 `exp` 上溢阈值时参考不再自造 `0·inf=NaN`。同一份 M151 m4097 归档上：
>   **旧式 fp32 = 8,374,272 NaN / 指数差 fp32 = 0 NaN**，设备 = 8,374,400（`o`）—— 详见 §4o。
>   旧式保留在 `--legacy-exp`；`--stab-demo` 可并排复现。**未改**任何已提交读数/日志/dump。
> * **仍未做的**（见 §5）：满量级重复统计（M=16/17 那批按「先搁置」停着）、端到端全链验收、性能优化、
>   P2-2 登记项（只登记未修）。**本段体不自行声称 clean，走复审。**

---

## 1. 被测对象与边界

| 项 | 内容 |
|---|---|
| 被测 | `m15_layer_loop/m15_gdn_prefill.h` 的 `M15GP::GdnPrefillAivEntry` / `GdnPrefillAicEntry` |
| 驱动 | 本目录 `m23_gdn_prefill.asc`（`__mix__(1,2)`，自带 `main()`；**不碰** `m15_layer_loop/CMakeLists.txt`） |
| 资源 | `m15_layer_loop/m15_gdn_resources.h` §6（**只增**：prefill 段窗；decode 档常量一字未动） |
| host 契约 | `m15_layer_loop/m15_gdn_prefill_host.h`（scratch 尺寸 / 补齐行数 / 挂载点说明） |
| 真实 shape | `m=4097`（= 64×64+1，尾 chunk 只有 1 行有效）与 `m=1`（单 chunk、cv=1），`H=48` |

## 1b. 接口契约（**塔裁 2026-09-27**；防边界漂移）

**本段的边界 = 「GDN prefill 的 chunk 扫描段」**：吃的是**已经过 prolog 的** `q/k/v/g/β`，
即 m18_gdn_prefill 的 I/O 契约（`docs/15 §M103-2.2` B1 行的 donor 就是它）。塔已明确：
任务书里「m18 参考不含 conv1d/l2norm/gating，缺的部分要自己立」是**条件句**（只有当本段含那三项时才要自建参考），
**不是**要求把它们纳进来。

| 平面 | 形状 | 谁供应（上游） | 谁消费 |
|---|---|---|---|
| `q`, `k` | `[NK=16, m, 128]` fp32 | **m9 prolog 段**（conv1d + l2norm；scale 未乘） | 本段（J1 的 `kk`/`qk` + J2 的 `q·S`） |
| `v` | `[48, m, 128]` fp32 | m9 prolog 段（conv1d） | 本段（u 的右端） |
| `g`, `β` | `[48, tp]` fp32 | m9 prolog 段（gating：`g=−exp(A_log)·softplus(a+dt_bias)`、`β=sigmoid(b)`） | 本段（cumsum / WY / 衰减） |
| `h0` | `[48,128,128]` fp32 | 上游（decode 递推或上一层链的状态；首 chunk 可全零） | 本段（状态初值） |
| `scale` | 标量 | 由调用方给（`1/√128`） | 本段（`o_part` 与 `ABM` 的 scale） |
| `out` | `[48, m, 128]` fp32 | **本段生产** | 下游 **P10 RMSNormGated** |
| `ht` | `[48,128,128]` fp32 | **本段生产**（= 状态末值） | 下游（下一 chunk / decode 递推） |

**不在本段**：P0 `in_proj`、P1 `conv1d`、P2 `l2norm`、P3 `gating`、P10 `RMSNormGated`、P11 `out_proj`。
⇒ `check_ref.py` 的参考**不含** conv1d / l2norm / gating（与 m18 的 fp64 参考同口径）；若融合时把 prolog
并进本段，**必须另立这三项的参考**。

**塔裁（`Scan_SolveG`）**：三角前代 `(I+A)⁻¹X` **留 fp32 VF** —— 它是**串行递推、不是 contraction**，
不在人类「凡矩阵乘法必须 mmad」的射程内（按 `docs/19 §10.2` 执行；`docs/19 §10.4` 那处"需塔确认"
**已由塔确认为塔裁**，本段按此落地，**不修改 `docs/19`**）。

## 2. 设计（一页）

* **全部矩阵乘法走 cube 的 `Mmad`**（人类裁决）：6 个收缩 —— `kk=k·kᵀ`、`qk=q·kᵀ`、`(w·S)ᵀ`、`q·S`、
  `(AB·d)`、`(kᵀ·DP)ᵀ`。**无 VF 收缩实现、无选型对比、无 VF 备选支**。
  非收缩项（cumsum / Γ=exp 组装 / β / 逐行缩放 / 三角前代）留 AIV 的 fp32 VF —— 依据 `docs/19`
  §10.2 的逐条映射，且 §11.2 的精度分析要求这两条链留在 fp32。
* **dtype = 全 fp32**：依据 M106（`probe_cube_fp32/README.md` §3/§6）：`Mmad(float,float,float)` 成立、
  默认模式操作数保持全 fp32、累加 fp32 宽度 ⇒ 判据形态不变（fp64 紧容差）。代价 = fp32 操作数下
  Mmad 吞吐为 bf16 的 1/15.88（≈23 TFLOPS）。本段**不调 `asc_enable_hf32()`**。
  fp32 的 cube `C0_SIZE = 8`（bf16 是 16），全部装载参数按此口径。
* **状态归属 `GdnStateHome::AivUb`**（`kGdnStateHome`）且以 **ST = Sᵀ（[DV,DK]）** 朝向存放
  —— 这个朝向让 6 个收缩的 cube 操作数**零转置**（M3 的 A 与 M4 的 B 都是 ST 本身）。
  AIV 每个 chunk 只做 2 次 VF 转置（`u→uᵀ`、`k→kᵀ`）；`KT' = kᵀ` 每列乘 `egL·ig[t]`，
  把跨 chunk 衰减折进操作数 ⇒ 状态更新退化成一趟 `ST₁ = egL·ST₀ + SDT`。
* **三段 job / mode 2 核间同步**：J1 `{kk,qk}`；J2 `{(w·S)ᵀ, q·S}`；J3 `{AB·d, (kᵀ·DP)ᵀ}`。
  每 job 一对 flag（`GO*` 由 AIV set、`DONE*` 由 AIC set），共 6 个 flagId。
* **核内一律 BufferID**（阻塞释放），**无 set_flag/wait_flag 系列**；同 pipe 背靠背复用同一 buffer 处
  加 `PipeBarrier<对应 PIPE>`；落 GM 一律 DMA（AIV 用 MTE3、AIC 用 Fixpipe），无标量直写 GM。
* **资源全静态**：UB 窗 248,064B（≤248KB）、L1 窗 491,520B（≤512KB）、L0A/L0B/L0C 各 64KB、
  BufferID（AIC 3 个 / AIV 2 个）、flagId 6 个 —— 全部编译期常量 + `static_assert`。

## 3. 融合清单（Wave C 直接照抄）

**(a) 挂载点**：`m15_layer_kernel.h` 的 **PREFILL 分支、`KIND_GDN` 层、相位 A**
（GDN 段在 attention 段之前，`docs/12` §3 的 S1–S7 里对应 S3–S4 之间那一段）。
入口符号：`M15GP::GdnPrefillAivEntry`（AIV）/ `M15GP::GdnPrefillAicEntry`（AIC）。
消费的 GM 平面：`q,k[NK,m,128]`、`v[48,m,128]`、`g,β[48,tp]`、`h0[48,128,128]`；
生产的 GM 平面：`out[48,m,128]`、`ht[48,128,128]`（in-place 语义由 Wave A 决定，本段只写不做 RMW）。

**(a2) 核间 mode 的逐 handoff 审计（塔裁 ①：逐个核）**：
本段的 6 个 flag **全部**是「AIC ↔ 它自己那 2 个 AIV」的**成对同步**（每个 AIC 只等它配对的 AIV；
前后不是两个 op、tiling 一一对应）⇒ 按人类裁决的判据属于 **mode 2**，**不属于**「上段落 GM、下段从 GM 读
（all to all）」那一类。GM 中介只改数据通路，不改"AIC 只等自己那对 AIV"这件事。
⚠ 该审计结论**不是**挂死根因的排除依据 —— 见 §4b 的探针读数（探针把三个核间假设都实测排除了）。

**(b) `LayerArgs` 需要的字段**（Wave A 照它加）：`q/k/v/g/β/h0/out/ht` 的 GM 基址、`m`、`heads(=48)`、
`tp = align8(m)`、`scale`、以及**段私有 scratch 基址**（尺寸 = `nAic × 753,664B`）。
本段**不消费 `LayerArgs` 本体、不改 Wave A 的文件**。

**(c) 资源窗与峰值**：

| 资源 | 窗 | 峰值 | 断言 |
|---|---|---|---|
| UB（AIV） | `[0, 248,064)`（段独享；与 decode 段窗互斥） | 248,064B | `static_assert(GP_UB_BYTES <= UB_BYTES_TOTAL)` |
| L1（AIC） | `[0, 491,520)`（kq 128KB + ST 128KB + W 64KB + J3 160KB） | 491,520B | `static_assert(GP_L1_BYTES <= L1_BYTES_TOTAL)` |
| L0A / L0B | 各 64KB（单缓冲） | 64KB | `static_assert(GP_DV*GP_DK <= GP_L0A/B_MAX_ELEMS)` |
| L0C | 64KB | 64KB | `static_assert(GP_DV*GP_DV <= GP_L0C_MAX_ELEMS)` |
| GM scratch | `nAic × 753,664B`（28 核 = 20.6MB） | — | host 侧 `GdnPrefillScratchBytes()` |

**(d) BufferID / flagId 清单**：
`GP_BUF_L1=0`、`GP_BUF_L0=1`、`GP_BUF_L0C=2`（AIC）；`GP_BUF_BLK=0`、`GP_BUF_UP=1`（AIV）。
flagId（**相邻同步点必不同 id**）：`GP_FLAG_GO1=0`、`DONE1=1`、`GO2=2`、`DONE2=3`、`GO3=4`、`DONE3=5`
—— 全线用 **mode 2**（AIC ↔ 其配对 2 个 AIV）；set 侧挂生产 pipe（AIV 的 `GO*` 挂 `PIPE_MTE3`
（数据已上传）/ `PIPE_V`（J1 前的放行）、AIC 的 `DONE*` 挂 `PIPE_FIX`），wait 侧挂 `PIPE_MTE2`。
6 个 id 与 decode 段的 4–15 槽**不同名、可共存**（融合后由 Wave C 统一登记相邻性）。

**(e) 需要的相位边界**：本段**自带全部核间同步**（三段 job 的 flag 对），**不需要**外层全体 AIV 的
mode-0 barrier；段入口返回时（`ht` 已落 GM）即对下游可见。若 Wave C 把本段与 attention-prefill 段
放进同一相位 A，**两者只共享 UB 窗，互斥复用**（`docs/15` §M103-2.3），需要一次全体 AIV 的
mode-0 barrier 分隔。

**(f) `m` / `pos` 语义**：`m ∈ [1, 4097]`（`GP_M_PREFILL` 是验收档、不是接口假设）；
本段**不吃 `pos`**（无绝对位置概念，衰减是 chunk 内相对的）。
越界读契约：AIC 的 `Nd2Nz` 对尾 chunk **固读 64 行** ⇒ `q/k` 平面末尾须留 ≥ 64 行可读且**置零**
（`GdnPrefillPadRows()`）。

**(g) 开关（两层门，M132 裁决）**：`M15_GDN_PREFILL_WIRE` 是**编译期宏**，与 §3(h) 的 `#if` 同一名字；
**运行期**真正的总开关是既有的 `M15_PREFILL_WIRE`。M138 订正：本小节原先给出的
`O.gdnPrefillWire = H_EnvU32("M15_GDN_PREFILL_WIRE", 0u)`（运行期 env 字段）在本仓**不存在**，
且与 §3(h) 的 `#if` 拼法自相矛盾；两个名字的分工如下（落点见 `m15_layer_kernel.h` §3d-0）：

```cpp
// m15_layer_kernel.h（Wave C 的挂载点；缺省 1 = B1 段体编进本 TU，只吃设备头）
#ifndef M15_GDN_PREFILL_WIRE
#define M15_GDN_PREFILL_WIRE 1
#endif
...
#if M15_GDN_PREFILL_WIRE
    if constexpr (KIND == KIND_GDN) { /* ... 段体挂载 ... */ }
#endif
```
- `M15_GDN_PREFILL_WIRE`（**编译期**）：B1 段体是否编译进融合 TU，缺省 **1**（段体已随 main 合入）；
- `M15_PREFILL_WIRE`（**运行期**）：host `O.pfWire` → `LayerArgs::pfWired`，缺省 **0**
  （`m15_layer_loop.asc:224`），门控 prefill 路径（含 B1/B4/B5）这一段是否执行 ⇒ `runs=all` 的
  decode 零回归在融合前后都可复跑。

**(h) 可直接用的挂载点补丁文本**（放到 `m15_layer_kernel.h` 的 PREFILL 分支里；Wave C 只做机械替换）：

```cpp
#if M15_GDN_PREFILL_WIRE
    if constexpr (KIND == M15L::KIND_GDN) {
        M15GP::GdnPrefillArgs gp{};
        gp.q = args.wsQ;   gp.k = args.wsK;   gp.v = args.wsV;
        gp.g = args.wsG;   gp.beta = args.wsBeta;
        gp.h0 = args.ssmState;  gp.out = args.wsO;  gp.ht = args.ssmState;
        gp.scratch = args.wsGdnPrefillScratch;      // Wave A 新增字段
        gp.m = args.m;  gp.heads = M15G::HEADS;  gp.tp = M15G::WS_AlignUp(args.m, 8);
        gp.scale = 0.08838834764831845f;            // 1/√128
        if ASCEND_IS_AIV { M15GP::GdnPrefillAivEntry(gp); }
        if ASCEND_IS_AIC { M15GP::GdnPrefillAicEntry(gp); }
    }
#endif
```
（`#if M15_GDN_PREFILL_WIRE` 是**编译期**门（宏由 `m15_layer_kernel.h` §3d-0 缺省定义为 1，
**不由** `.asc` 的运行期 env 传入；运行期总开关见 (g) 的 `M15_PREFILL_WIRE`）；
字段名 `args.ws*` 是**示意** —— 最终以 Wave A 的 `LayerArgs` 实际字段为准。）

## 4b. 核间握手探针读数（塔裁 ③；`m23_cc_probe.asc`，自有 target）

**变量**：`variant` + `iters`；**判据 = 返回 / 不返回**。逐字读数见 `evidence/logs/cc_probe_r1.md`。

| variant | 形态 | 读数 |
|---|---|---|
| 0 | AIV set 挂 `PIPE_MTE3`；2 AIV set(GO) → AIC **等 1 次**；AIC set(DONE) 1 次 → 2 AIV 各等 1 次 | **返回 OK** @ iters = 4/8/16/24/**200** |
| 1 | 同 0，但 2 set → AIC **等 2 次** | **不返回** @ iters = 4/8/16（每个都挂） |
| 2 | 同 0，但 AIC 真发 `Fixpipe` 且 set 挂 `PIPE_FIX`（= 本段体 AIC 侧形态） | **返回 OK** @ 8/40 |
| 3 | 同 0，但 AIV 的 set 挂 `PIPE_V`（= 本段体 GO1 形态） | **返回 OK** @ 8/40 |

**结论**：① 本段体用的那套 mode 2 形态**在 200 轮重复下成立** ⇒ 排除「每 id 4bit 计数器未配平超 15」；
② **配对必须严格 N set ↔ 1 wait**（多等一次即死锁，iters=4 就挂）—— 这条建议全舰队复用；
③ 三个核间假设（mode 选择 / 计数器 / set 挂的 pipe）**均被实测排除** ⇒ 本段体的挂死根因在**核内
BufferID 纪律**或**段体逻辑**（下一步：把三段 job 收成只留 J1、逐段加回）。

## 4c. 收窄读数（塔裁 ④；逐字见 `evidence/logs/narrow_r1.md`）

| 档 | 编译期开关 | 读数 |
|---|---|---|
| **J1-only**（只跑 J1） | `M15GP_J1_ONLY=1`（target `m23_gdn_prefill_j1`） | **返回 OK**：m=1 **0.059 ms**、m=4097 **1.930 ms**；该档已含每 head 195 次 `GO1/DONE1` 握手 + LoadState/StoreState + 4 个 MTE2 组 + 转置/cumsum/Γs/两次三角前代 + KT' 上传 |
| 全档（J1+J2+J3） | 默认 | **设备异常（不是超时）**：`aivec error exception, core id is 57`、`mte error info`/`vec error info` 均非零、`retCode=0x26 [aicore exception]` |

**判读**：① **不是同步死锁** —— 同一套核间握手 + 核内令牌纪律在 J1-only 档满量级跑通
（与 `cc_probe_r1.md` 独立排除三个核间假设互相印证）；② 异常在 **AIV** 的 **J2/J3 专属路径**
（`SubTransposeVF` / `MulRows1VF` / `ConstScale1VF` / `AddRows2VF` / `ScaleAddRowsBrc2VF`、
WT/ABD/SDT 的 MTE2 装载参数、ST/W/DT/ABM 上传握手）—— **具体是哪一处未取证**；
③ 早先两次"不返回"与本轮异常同源（设备异常经 `rtStreamSynchronize` 表现为长等待）。

## 4c2. 收窄读数 r2（`evidence/logs/narrow_r2.md`）

| 档 | m=1 | m=4097 |
|---|---|---|
| 只 J1 | OK | OK |
| J1+J2 | OK | **507015 aicore error** |
| J1+J2+J3 | OK | **507015（与 J1+J2 同签名）** |

⇒ **J3 被排除**（加回 J3 不改签名）；故障在 **AIC 侧的 J1+J2 路径**，且 **m 相关**（65 chunk 才出）
⇒ 与"同一处代码被重复执行"有关。同签名的 `mte error info`（`0x13d1…0202ce`）、`vec/cube error info = 0`
说明是**同一类形态**。另查出：**J3 的 `l1dt`/`l1ab` 按 `TPosition::A1` 声明在偏移 360448/376832 ——
若 A1 是 256KB 窗口（m11 的 A/B 分窗提示如此）则超出窗口**；以及 **q/k 平面行 stride 未进契约**
（host 用 `Tp = m+64`、kernel 用 `m`）—— 后者已修（新增 `GdnPrefillArgs.qkStride`），但不改故障签名。

## 4c0. 判据侧的两条守卫（r2 复审返工，2026-10-04）

* **反向对照（`--mutate`）已能咬住**：`MutateInputs` 只搅动**送设备的副本**（`vDev/gDev/h0Dev`），
  host 侧 dump 用干净数据 ⇒ 参考吃干净、设备吃搅动；`check_ref.py --mutant 1` 实测**变红**
  （见 `evidence/logs/r2_rework_r1.md`）。
* **新鲜度守卫**：`check_ref.py` 会比对 dump 与当前二进制的 mtime，**陈旧 dump 拒绝判定**（`exit 3`），
  `--allow-stale` 可显式越过；根目录陈旧 dump 已隔离进 `stale_dumps/`。**措辞订正（r3 复审）**：`.gitignore` 里的 `stale_dumps/` 只对**未跟踪**文件生效，
  而该目录下 **11 份 `_meta.txt` 是有意保留 tracked 的「改前读数」存档**（`git check-ignore` 对它们不命中）
  ⇒ 那一行**不代表**这些文件已被忽略；误跑风险由三重防护挡住（目录名自解释 + 守卫 mtime 拒绝 `exit 3` + 越过须显式
  `--allow-stale`），三条均已实测（见 `evidence/logs/r2_rework_r1.md` 与 r3 复审的零设备 5 例）。
  判据命令请仍按 §4 从 `build/` 跑。

## 4c3. L1 的 A1/B1 窗口语义：**取证后仍未确定**（`evidence/logs/l1_window_evidence_r1.md`）

塔要求"先取证再动手"。已查：官方文档（`asc-devkit` 的 `L1_L0A_B_memory_structure_intro.md`，CANN 9.1.0）
逐字说 **L1 总容量 512K 字节、16 个 Bank ×32K、地址编码 `L1_ADDR[18:0]={BANK,BANK_DEPTH,BG,BANK_WIDTH}`**
—— **一根扁平 512KB 空间，没有把 A1/B1 描述成两个地址窗**；CANN/bisheng 头文件里
`A1_OFFSET`/`B1_OFFSET`/`POSITION_A1`/`TSCM_A1`/`A1_BASE`/`B1_BASE` **0 命中**（我未能定位 A1/B1 的基址常量）。
本仓的"A 区 [0,256KB) / B 区 [256KB,512KB)"只是**约定**（`m11_bf16_gemm.asc`、`m15_gdn_layer.h`、`m15_hc_layer.h`），
**属间接线索**。⇒ **结论：未确定**（不能断言窗口语义成立，也不能断言 `Job3` 的 A1 声明非法）；
**未据此改 L1 布局**，也**未对外广播**。

## 4c4. 收窄读数 r3（`evidence/logs/j12nz_r1.md`）：**J2 的 Nd2Nz 跨 chunk 重复 = 触发器**

| 档 | m=1 | m=4097 |
|---|---|---|
| 只 J1 | OK | OK |
| J1+J2（Job2 的 Nd2Nz 每 chunk 跑） | OK | **507015** |
| J1+J2（**Job2 的 Nd2Nz 从第 1 chunk 起跳过**，`M15GP_J2_NZ_SKIP_FROM=1`） | OK | **OK** 2.360 ms |

⇒ 故障由 **`Job2` 里"每 chunk 重复的 MTE2→L1 写入"**触发；而 **J1 自己每 chunk 重复的 Nd2Nz（写同一批 L1 区）
在 65 chunk 下没问题** ⇒ 不是"MTE2 重复写 L1"这个笼统形态，而是 **J2 这一处**（三条里至少一条）。
下一步：按条跳过（q 重装 / ST / W）定位到具体那一条 —— 详见该证据文件（机械可执行）。

## 4c5. `Job1` vs `Job2` 静态对照（塔裁 ③：零设备；与按条跳过互为交叉验证）

见 `evidence/logs/job1_job2_static_diff.md`。**无差异**的维度：BufferID（都是 `GP_BUF_L1`）、
acquire/release 位置（同一 `DoMmad`）、是否每 chunk 重新 acquire（都是）。
**有差异的三处**：① L1 目的偏移/尺寸（`L1St(h)` 是唯一的 **64KB、128 行**Nd2Nz；`L1W(h)` 是**唯一 B1 声明**）；
② `Nd2Nz` 参数（ST 的 `dstNzC0Stride=128`）；③ **同一 chunk 内重复写同一 L1 地址**（`L1Kq(h)+32KB`
本 chunk 内已被 `Job1` 写过一次 —— 这是唯一一处）。
三条各自对应"按条跳过"的一个预测（表内已列），**这就是两条路互为交叉验证的用法**。

⚠ **性能口径警示**：`j12nz_r1.log` 的 **2.360 ms 不是** B1 的段体耗时 —— 那是跳过 `Job2` 的 L1 写入之后的
耗时，当时 q 重装/ST/W 的数据是**陈旧/错的**。**不得**引用为 B1 性能读数。

## 4c6. 收窄读数 r4（`evidence/logs/j12nz_which_r1.md`）：按条跳过 —— **三条各跳一条都仍复现、同签名**

| 跳掉的 | m=4097 | `mte error info` |
|---|---|---|
| 只 q 重装 / 只 ST / 只 W | **三档都复现 507015** | 三档逐位相同（`0x13d1…0202ce`）；`vec`/`cube` 全 0 |
| 三条全跳（r3） | **OK** | — |

**修正 r3 的推论**：三条**没有一条是单独必要的** ⇒ §4c5 那张"按条跳过 ↔ 静态差异"的一一对应表
**在本步未获确认**，**不能**据此指认某一条差异为根因。**同签名仍成立**（与 r2/r3 逐位相同 ⇒ 同一处形态）。
**下一步是补集实验**（只保留一条，其余两条跳掉），三种结果各自对应什么判读已写进该证据文件。
**方法学（建议进舰队口径）**：*逐项移除只能证明"某项不必要"，不能证明"某项是根因"*；定位"哪一项触发"
必须做"只保留一项"的补集实验。

## 4d. 本次修正（塔裁 ⑤：修前/修后读数）

| # | 缺陷 | 修前 | 修后 |
|---|---|---|---|
| 1 | `scSE` 算错：应为 `scale·eg`（逐 lane），原写成 `scale·(egL 广播)` ⇒ `o_part` 的行缩放整列用同一个 `scale·egL` | 构建 rc=0（数值错，无读数） | 改用新原语 `VecScaleConstVF(scSE, scEG, scale_)`；构建 rc=0（`evidence/logs/build_r2.log`） |
| 2 | `DoMmad` 缺 `GP_BUF_L1` 的写侧/搬侧成对 acquire（L1 大包就绪无见证） | 构建 rc=0（无读数） | 补 `BufAcquire<PIPE_MTE1>(GP_BUF_L1)` / `BufRelease<PIPE_MTE1>`；构建 rc=0 |
| 3 | `BrcScalarVF` 误用（`scSE` 分支） | 见 #1 | 见 #1 |
| 5 | `Job3` 的 `l1dt`/`l1ab` 以 `TPosition::A1` 声明在偏移 360448/376832（若 A1 是 256KB 窗口则越窗） | 该档无数值读数（507015） | **未改**：窗口语义**未确定**（见 §4c3），先取证再动手 |
| 4 | q/k 平面**行 stride 未进契约**：host 按 `Tp=m+64` 存放（每 head 尾部补齐），kernel 按 `m` 读 ⇒ 行 stride 不匹配（m=1 时被掩盖） | 构建 rc=0；m=1 档"返回 OK"但数值无判据读数 | 新增 `GdnPrefillArgs.qkStride`（host 传 `Tp`），AIC/AIV 均按它寻址；构建 rc=0；**修后 507015 仍在 ⇒ 不是本次故障的根因**（如实标注）。**数值差异读数**：修前/修后该档都跑不到判据（全档在 m=4097 抛 507015，m=1 档虽返回 OK 但 `p4097` 档无 dump ⇒ `check_ref.py` 全档无可判对象）；**唯一可说的差异**是 m=1 档 `o[0]` 从 `0.000000`（J12 收窄档）变为 `-0.061852`（全档，修后 r2 为 `-0.063054`）—— 即 stride 修正确实改变了实际算出的值，但没有合格读数可用于判断对错。 |

**与 M111（`b162590`，塔指定的对照物）的形态对齐**：本段的每个上传握手都是
「**写侧 acquire（V 或 MTE2）→ 写 → 写侧阻塞释放（`RlsBufInternal<pipe,false>`）→ 对侧 acquire（MTE3）→ 搬 → 对侧阻塞释放**」
（`m15_gdn_prefill.h` 的 `UpAcqV/UpRelV/UpAcqMte2/UpRelMte2/UpAcqMte3/UpRelMte3`），与 M111 的三处站点同形；
同 pipe 背靠背复用同一 buffer 处另有 `PipeBarrier<对应 PIPE>`。

## 4. 判据（本目录）

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4
cd build && flock -w 900 /tmp/npu0.lock ./m23_gdn_prefill          # 落 m23_<tag>_*.bin
/usr/local/python3.12.13/bin/python3 ../check_ref.py               # fp64 逐句参考（默认：指数差 Γ）
/usr/local/python3.12.13/bin/python3 ../check_ref.py --mutant 1    # 反向对照：必须变红
/usr/local/python3.12.13/bin/python3 ../check_ref.py --selftest    # 零设备自检（NaN 盲区 + 上溢档 + 负向对照，见 §4n/§4o）
/usr/local/python3.12.13/bin/python3 ../check_ref.py --dir <归档目录> --allow-stale   # M165：对归档档重跑（默认指数差）
/usr/local/python3.12.13/bin/python3 ../check_ref.py --dir <归档目录> --allow-stale --legacy-exp   # M165：旧式 Γ 对照
/usr/local/python3.12.13/bin/python3 ../check_ref.py --stab-demo --dir <归档目录>     # M165：并排打印 旧式/指数差 × fp64/fp32 的 NaN 计数
```
* 判定项（**M157 订正**）：`o` 的 `[0,cv)` 行与 `ht` 全量 —— ① **两侧皆有限**的元素逐元素
  `|dev−ref| ≤ rtol·max(1,|ref|)`（`rtol=2e-3`）；② **非有限面显式判定**（仅一侧 NaN/Inf、NaN↔Inf、
  异号 Inf ⇒ 计错；NaN↔NaN、同号 Inf↔同号 Inf 视为同类）。踩到 ① 或 ② 即 FAIL，逐条见 §4n。
* 反向对照：`--mutate 1/2/3`（v 翻号 / g 置零 / h0 加倍）—— 同一判据**必须变红**，
  否则判据是空的（`docs/17` §4）。
* **现状（2026-10-04）**：构建 + 设备档 + 判据档**都跑通** —— 新鲜档 `check_ref.py` **rc=0**（`m1`/`m4097` 的 `o` 与 `ht` 全 0 超界），
  反向对照 `--mutant 1` **变红**；逐字见文首状态与 `evidence/logs/r3_*`。

## 4n. 判据的非有限面（NaN/Inf）纳入判定 —— M157 订正（零设备、零探针）

**背景（回源取证）**：M151 复审报 P2 —— 旧判据的判定式是 `|dev−ref| > lim`，
即 **base commit `ac46c32` 的 `m23_gdn_prefill/check_ref.py:147-149`**：`:147` `do = np.abs(o_dev[hv] - o_ref)`、
`:148` `lim_o = args.rtol * np.maximum(1.0, np.abs(o_ref))`、`:149` `bad_o += int((do > lim_o).sum()); tot_o += do.size`。
该比较**对 NaN 静默为 False**：设备与参考都是 NaN 的元素既不进 `bad_o`，也不进 `max|Δ|`
（旧写法 `max(finite, nan)` 返回 `finite`）⇒ 报出的「0 超界」在那一片上**是盲的**。

**归档 dump 的 NaN 分布（本轮复现）**：对 M151 归档档
`m15_layer_loop/evidence/m151_prefill_prolog_wiring/dumps_m4097_clean/m23_Pf.gdn_*`
（`meta` sha256 `4e57f562…`、`o.bin` sha256 `be589303…`、100,687,872 B = 25,171,968 个 fp32）：

| 量 | 读数 |
|---|---|
| 设备 `o` NaN | **8,374,400** / 25,171,968 |
| fp64 参考 `o` NaN | **3,670,912**（是设备 NaN 的**子集**） |
| 仅一侧 NaN（XOR） | **4,703,488** |
| 旧判据在同一份数据上 | `o 超界 **0**/25171968 max\|Δ\|=1.892e-07 … PASS`（`rc=0`） |

⇒ 与 M151 复审的 8,374,400 / 3,670,912 / 4,703,488 一致（该复审记于
`.tower/comms/reviews/review-feat-m151-prefill-prolog-mount-into-phase-a-reviewer-m151-r1.md:28-34`）。
*〔该 dump 是 wt-151 工作区里的**未跟踪**文件（`git ls-files … | wc -l` = 0），属外部快照，
不在本目录 scope 内、本轮**只读不改**；M151 分支内 README 记录了同一组数。〕*

**判据改了哪一处（判据侧；内核与驱动一字未动）**〔M165 重排后行号已更新，现锚点见 §4o〕：
* 新增 `cat_of()`（`check_ref.py:78`）：按 **有限 / NaN / +Inf / −Inf** 归类 —— 同类视为相等
  （**NaN↔NaN、同号 Inf↔同号 Inf**），异类计为**非有限不匹配**（**仅一侧 NaN、仅一侧 Inf、NaN↔Inf、异号 Inf**）。
* `acc_pair()`（`check_ref.py:149`）里有限容差只在 `fin = (两侧皆有限)` 上比；`verdict()`（`:204`）改为
  「有限子集容差 = 0 **且** 非有限不匹配 = 0」；`render()`（`:210`）**新建第二行**逐档给出
  `o`/`ht` 各自的 `devNaN / refNaN / 仅一侧NaN / 双侧NaN / devInf / refInf / 仅一侧Inf / 异号Inf / 非有限不匹配`。
* `max|Δ|` / `max|ref|` 改为**只在有限元素上取**。

**适用范围与残余空档（写窄）**：
* **两侧都 NaN** 的位置（`双侧NaN`）**不计错** —— 但它**不是**"数值正确"：判据不声称那片可信，
  只是两侧同样非有限。**不得**把 `双侧NaN > 0` 的档读成"通过"，也**不得**由此断言"该处数值已验"。
* **仅一侧 NaN/Inf** ⇒ 一律 FAIL —— 这正是 M151 那 4,703,488 个位置的类别。
* 有限子集容差只覆盖两侧皆有限的元素，它**不**构成对非有限面的数值背书；非有限面的物理成因
  （例如 fp32 vs fp64 的 `exp` 溢出范围差）**仍需另行分析**，判据只负责把它**报出来**。
  *〔M165 已定位并修正：成因是**参考侧**旧式 Γ 的 `eg·ig` 物化相乘（`0·inf`），不是设备缺陷也不是
  "两种精度的参考本身差很大"；已改成指数差，见 §4o。设备侧同一写法仍待 M162。〕*

**负向对照（两个，零设备）**：`--selftest`（`check_ref.py:352`）在临时目录合成小 dump，走**与真判据同一条**
`judge_one / render / verdict` 路：

| 档 | 注入（`synth_dump`，`:308`） | 新判据读数 | 判定 |
|---|---|---|---|
| `syn` | 无（干净） | o/ht 非有限不匹配均 0 | **PASS**（对照非空洞） |
| `syn_refmut`（**a 参考侧**） | 只把喂参考的输入 `g[0,7]` 置 NaN（`:338`） | o `refNaN=1024 仅一侧NaN=1024`；ht `refNaN=16384 仅一侧NaN=16384` | **FAIL** |
| `syn_devmut`（**b 设备侧**） | 只把设备 dump `o[0,1,0:16]` 置 NaN（`:336`） | o `devNaN=16 仅一侧NaN=16`；ht 全 0 | **FAIL** |

*〔M165 后 `--selftest` 新增第 4 档 `syn_overflow`（chunk 内上溢）：指数差判据 PASS / 旧式判据 FAIL；
其逐字输出与本节签名见 §4o 与 `evidence/logs/m165_repro.log`。〕*

签名（**M157 时点**逐字，rc=0；M165 后 `--selftest` 新增第 4 档 `syn_overflow` 且每档打印加了「chunk 内上溢」一项，
新档逐字见 §4o 与 `evidence/logs/m165_repro.log` §1）：
```
[SELFTEST] 期望 PASS / 实得 PASS  ✓                              (syn)
[SELFTEST] 期望 FAIL / 实得 FAIL  ✓                              (syn_refmut)
[SELFTEST] 旧式 `|dev-ref|>lim` 计数 = 0（该数据非有限元素 0 个）
[SELFTEST] 期望 FAIL / 实得 FAIL  ✓                              (syn_devmut)
[SELFTEST] 旧式 `|dev-ref|>lim` 计数 = 0（该数据非有限元素 16 个）
[SELFTEST] ⇒ 复现「NaN 被静默放行」：旧判据报 0 超界，新判据报 非有限不匹配 > 0
[SELFTEST] 各档符合期望（rc=0）
```
⇒ `syn_devmut` 那两行**就是盲区的现场复现**：同一份数据，旧式比较计数 = 0（静默放行），新判据 = 16。
命令：`/usr/local/python3.12.13/bin/python3 m23_gdn_prefill/check_ref.py --selftest`（零设备）。

**回归：已有归档档逐档重跑（旧判据 = `ac46c32`、新判据 = 本轮；零设备）**

| 档 | 归档路径 | 旧 | 新 | 差异解释 |
|---|---|---|---|---|
| m23 `m1` clean | `/tmp/m115_verify` | PASS rc=0 | PASS rc=0 | 该档无 NaN/Inf ⇒ 非有限项全 0，判定不动 |
| m23 `m4097` clean | `/tmp/m115_verify` | PASS | PASS | 同上 |
| m23 `m1`/`m4097` clean | `/tmp/m130tip2/…/m23_gdn_prefill/build` | PASS | PASS | 同上（守卫触发，两版同用 `--allow-stale`） |
| m23 `m1_mut1` / `p4097_mut1` | 同上 `--mutant 1` | FAIL rc=0 | FAIL rc=0 | 有限子集已红（2948/6144、708806/786432 …）且无非有限项，判据不变 |
| m23 `p4097_mut1`（另一份归档） | `/tmp/m115_mut` `--mutant 1` | PASS→「判据没咬住」rc=1 | 同 | 该归档的变异档与干净档本身无差异（旧档缺陷，非本轮引入）；两版判据同样报 rc=1 |
| **M151 `Pf.gdn` m=4097** | `wt-151 …/dumps_m4097_clean` | **PASS rc=0** | **FAIL rc=1** | **新红**：`o 仅一侧NaN=4703488`（devNaN 8,374,400 / refNaN 3,670,912）⇒ 旧判据的盲区被新判据报出 |
| M151 `Pf.prolog_mut1` m=4097 | 同上 `--mutant 1` | **PASS rc=0** | **FAIL rc=1** | **新红**：与上一行**同一签名**（同一份非有限面） |
| M151 `Pf.gdn_mut1` m=4097 | 同上 `--mutant 1` | FAIL | FAIL | 有限子集已红；新判据再加同一非有限项，判定不变 |
| M151 `Pf.gdn` m=1 | `wt-151 …/dumps_m1_clean` | PASS | PASS | T=1 无非有限 |

**逐档解释（新变红的档）**：只有 M151 归档的 `Pf.gdn` 与 `Pf.prolog_mut1`（m=4097）两档**新变红**，
两者是**同一份非有限面**（8,374,400 个设备 NaN，其中 4,703,488 落在参考有限侧）。
**新判据看见的是旧判据没计入的那片溢出区**：M151 复审当时把它归因为「fp32 与 fp64 的 `exp(gcum)` 溢出
范围之差（fp32 变体 XOR=128），**不是设备缺陷**」——**这条归因已被 M160/M165 修正**（本节首注
`m23_gdn_prefill/README.md:347-349` 与 §4o）：修正参考侧旧式 Γ 的 `0·inf`（改指数差）后参考不再有 NaN，
设备那 8,374,400 个 `o` NaN 因此成为「仅一侧 NaN」⇒ 它们来自**设备侧同一旧式写法**（设备侧实修在
M162），**不是**两种精度的范围差；
本轮是把这件事**从"人算的"变成"判据自己能报的"**。**没有任何档由红转绿**（新判定只增加计错项）。
一处**读数口径**变化如实记下：同一档（`Pf.gdn` m=4097）的有限子集 `max|Δ|` 由旧判据的 `1.892e-07`
变为 `7.203e-04` —— 旧写法在含 NaN 的 head 上 `max(finite, nan)` 把那一 head 折掉，新写法按有限子集取，
所以**新值才是有限子集上诚实的最大偏差**（仍 `< rtol·max(1,|ref|)`，故 `o 超界` 仍为 0）。

**纪律自查（M157）**：零设备、零探针；改动集 = `m23_gdn_prefill/check_ref.py` + 本 README；
**未改**任何已提交读数/日志/dump。禁用词自查：本轮新增/改动行对 mission 的六词禁用表逐字普查，
结果 = 未命中。**六词表内容此处不逐字复列** —— mission 明列「六个词均不得出现」，若在本文件里复列，
这条自查命令/清单本身就会成为新增行里唯一的命中点。（若要复跑：按 M157 mission 的六词表串成
`grep -nE '<六词>'` 作用在 `git diff -U0 ac46c32..HEAD -- m23_gdn_prefill | grep '^+'` 上。）
**结论带 文件:行**（M165 重排后已更新）：`check_ref.py:78` 的 `cat_of`、`:149` 的 `acc_pair`、
`:204` 的 `verdict`、`:210` 的 `render`、`:336`/`:338` 的注入点、`:352` 的 `selftest`。

## 4o. 参考侧数值稳定化与官方对齐 —— M165（零设备）

**人类原口径（逐字）**：「那就说明这个golden的生成脚本或者公式或者数值范围本身就很病态，你应该看看别人类似的kernel的golden怎么生成的」。
M160 survey（只读结论）把它定位到**参考/实现的写法**：chunk 扫描把 `eg=exp(ĝ)` 与 `ig=exp(−ĝ)` 分开物化再相乘，
chunk 内 `|ĝ|` 超 `exp` 上溢阈值（fp32 88.7 / fp64 709）时 `0·inf=NaN`；而正确因子 `exp(ĝ_i−ĝ_j)≤1`
（`i≥j`，有限输入下恒有限）。官方 FLA 用的就是指数差
（`/workspace/vllm/vllm/third_party/flash_linear_attention/ops/chunk_o.py:119-120`、
`chunk_delta_h.py:216-221`）。**本 mission 修参考侧（golden），零设备；设备侧同一写法在 M162。**

### (1) 修的是哪几处（`m23_gdn_prefill/check_ref.py`）

| 处 | 旧式（修前） | 指数差（修后，默认） |
|---|---|---|
| Γ 下三角 `Gs`/`Gi` | `:118-121` `eg[:,None]*ig[None,:]` | `:114-117` `exp(ĝ_i−ĝ_j)`（上三角的 exp 输入置 0，避开无关上溢） |
| 跨 chunk 衰减 | `:139-140` `egL*(S + kᵀ(d*ig))` | `:136-138` `egL*S + kᵀ(d*exp(ĝ_last−ĝ_j))` |
| `solve` 奇异兜底 | 无（旧式溢出时 `M` 含 inf/nan ⇒ 可能抛 `LinAlgError`） | `:125-131` `try/except` 记该 chunk 非有限（指数差不会走到） |

`ref_head` 签名见 `check_ref.py:91`（`stable=True` 默认 / `stable=False` 旧式 / `dtype` 供 fp64/fp32 离线对照）。
**未改**：`eg=exp(ĝ)` 仍用于 `w` 的右端与 `o_part`（`ĝ≤0 ⇒ eg≤1`，本就不上溢），`Gs/Gi` 之外的结构一字未动。

### (2) 全仓同类写法普查（`exp(ĝ)·exp(−ĝ)` / 会 `0·inf` 的形态）

| 文件:行 | 形态 | 判定 |
|---|---|---|
| `m23_gdn_prefill/check_ref.py:118-121,139-140` | `eg*ig` 物化相乘 | **真会溢出**（本 mission 已修） |
| `m18_gdn_prefill/check_ref.py:197-199,211,233-234,245-246` | 物化 `eg=exp(gc)`、`ig=exp(-gc)`；Γ 已用 `exp(dgc)`（稳定），但**状态更新仍乘 `ig`** | **形式上同类、真数据会溢出**：其测试 g 上界 ≤0.051（chunk 和 ≤3.3）故当前档安全；喂真权重（chunk 和可达 9,856）会与 m23 同病。**未改（越界）—— 已报 TowerFinding** |
| `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py:59` | 自己复制了一份旧式 `eg*ig` 参考（未 import `ref_head`） | **同类**；它是 M151 的取证脚本，其结论（"fp64 参考 NaN 是设备 NaN 子集 ⇒ 精度产物"）已被 M160/M165 推翻。**未改 —— 已报 TowerFinding** |
| `m4_gdn_recurrent/check_ref.py:46`、`m14_gdn_layer/check_ref.py:382`、`m21_layer_ref/ref/gdn.py:50,172` | 单侧 `exp(g)` 衰减（`g≤0 ⇒ ≤1`），无 `eg*ig` 乘积 | **形式上相似但安全**（不同写法） |
| `m10_attn_decode/check_ref.py:257-259`、MoE softmax（`m13`/`m17`/`m26`/`tools/golden/*`） | `exp(m−m')` 型重标定 / 减最大值 | 安全 |
| 全仓 sigmoid/SiLU `1/(1+exp(−x))` | `exp(−x)` 可上溢为 inf，但 `1/(1+inf)=0` | 安全（不出 NaN） |

### (3) 与官方对齐（`m21_layer_ref/run_reference.py --layer 0 --input embed`，真权重，CPU）

只读跑官方脚本（不改 `m21_layer_ref/**`）：`--m 1 / 64 / 4097`，逐份取 `gdn.core_out` 与
`gdn.q/k/v/g/beta`，用同一份输入面分别跑 旧式 / 指数差 参考（脚本 `evidence/tools/m165_official_align.py`）。
逐字输出 `evidence/logs/m165_official_align.log` 与 `m165_repro.log` §3。关键读数：

| 档（m21 tag） | 官方 core_out NaN | 指数差 NaN | 旧式 fp64 NaN | chunk−seq（fp64） | chunk−core_out |
|---|---|---|---|---|---|
| `m165_layer0_m1` | 0 | 0 | 0 | 6.9e-18 | 2.705e-4（0.70 bf16 ulp） |
| `m165_layer0_m64` | 0 | 0 | 73,728 | 2.8e-16 | 1.252e-3（0.65 bf16 ulp） |
| `layer0_chunk_m64`（M160 用过的旧档） | 0 | 0 | 57,344 | 6.9e-17 | **2.881e-4**（0.64 bf16 ulp；M160 的 2.88e-4 ✓） |
| `m165_layer0_m4097` | 0 | 0 | 5,235,968 | 1.6e-14 | 8.981e-3（1.18 bf16 ulp） |

- `seq` = 官方递推逐 token 形式（m21 `ref/gdn.py::_recurrence` 的逐句等价）在**同样的 bf16 输入**上 fp64 重算。
  chunk−seq ≤ 1.6e-14 ⇒ **chunk 因子分解 == 官方递推**（只是求和顺序不同）。
- chunk−core_out 的来源是**输入 bf16 舍入**：官方 `core_out` 用 fp32 的 `qn/kn`，而 dump 出的 `gdn.q/gdn.k`
  是 bf16（`gdn.v` 本就 bf16）；上表 `seq−core_out` 与 `chunk−core_out` 同值即证。判据取
  「≤ 2 个 bf16 ulp（相对 max core_out）」，实测 0.64–1.18 ulp。
- **修前/修后对比**：旧式在这些真权重数据上**也自造 NaN**（m=64 fp64 73,728；m=4097 fp64 5,235,968、
  fp32 11,111,168），而官方数据 0 NaN ⇒ 那些 NaN 完全是旧式参考自己造的。

### (4) 归档档逐档重跑（NaN-aware 判据；这是**修正后的真实结论**）

命令见 `evidence/tools/m165_repro.py`（一键，rc 可传播失败）。关键档（零设备；外部归档只读）：

| 档 | 归档 | 判据模式 | 设备 NaN | 参考 NaN | 仅一侧 NaN | 判定 |
|---|---|---|---|---|---|---|
| M151 `Pf.gdn` m=4097 | `wt-151…/dumps_m4097_clean` | 旧式计数（M157 前） | — | — | — | **PASS（盲）** |
| M151 `Pf.gdn` m=4097 | 同上 | 旧式 Γ + NaN-aware | 8,374,400 | 3,670,912 | 4,703,488 | FAIL |
| M151 `Pf.gdn` m=4097 | 同上 | **指数差 Γ（默认）** | 8,374,400 | **0** | **8,374,400** | **FAIL** |
| M151 `Pf.gdn` m=1 | `wt-151…/dumps_m1_clean` | 指数差 | 0 | 0 | 0 | PASS |
| M151 `Pf.gdn_mut1`、`Pf.prolog_mut1` m=4097 | 同上 `--mutant 1` | 指数差 | 8,374,400 | 0 | 8,374,400 | FAIL（红 ✓） |
| m23 `m1`/`m4097` clean | `/tmp/m115_verify`、`/tmp/m130tip2/…/build` | 指数差 | 0 | 0 | 0 | PASS |
| m23 `m1_mut1`/`p4097_mut1` | `/tmp/m130tip2/…/build` | 指数差 | 0 | 0 | 0 | FAIL（有限子集红 ✓） |

**逐档解释**：修正后参考侧在溢出档上**不再有 NaN**（`refNaN=0`），因此设备那 **8,374,400** 个 `o` NaN
（`ht` 262,144）**全部**是"仅一侧 NaN" ⇒ FAIL。这**不是**"两种精度的参考本身差很大"，也**不是**新的设备缺陷：
M157 记的 `refNaN=3,670,912 / 仅一侧=4,703,488` 是**旧式参考自己造的**那一部分（`0·inf` 的位置），
改对参考后它们变成"参考有限、设备 NaN"的更清晰读数。M157 记的"多出的设备 NaN 是 fp32/fp64 `exp` 溢出范围之差、
不是设备缺陷"这句**口径被 M160/M165 修正**：设备（fp32）与参考（现在指数差）本都该有限，那 8,374,400 个 NaN
是**设备侧同一旧式写法**造成的（设备侧实修在 M162）。**无档由红转绿**。

### (5) 复现命令与证据（零设备）

```
# 自检（含合成上溢档：指数差 PASS / 旧式 FAIL）
/usr/local/python3.12.13/bin/python3 check_ref.py --selftest
# 修前/修后对照（M151 归档，只读）：旧式 fp32 8,374,272 / 指数差 fp32 0
/usr/local/python3.12.13/bin/python3 check_ref.py --stab-demo --dir <M151>/dumps_m4097_clean
# 官方对齐（需先只读跑 m21 run_reference.py --layer 0 --input embed --m 1/64/4097）
/usr/local/python3.12.13/bin/python3 evidence/tools/m165_official_align.py <refdir>...
# 一键：以上全部 + 归档重跑逐项核对（rc 可传播失败）
/usr/local/python3.12.13/bin/python3 evidence/tools/m165_repro.py --m151 <M151> --ref <refdir>...
```

证据：`evidence/logs/m165_repro.log`（一键全套，rc=0）、`m165_official_align.log`（官方对齐，rc=0）；
工具：`evidence/tools/m165_official_align.py`、`m165_repro.py`；自检签名与逐档输出在 `m165_repro.log` §1/§4。

**限度（写窄）**：① 官方对齐只到 **layer 0、只 `--input embed`**（更深层/真上游链未覆盖）；
② chunk−core_out 是**输入 bf16 舍入地板**（2.7e-4–9.0e-3），不是设备侧精度背书；chunk 分解自身的误差
用 chunk−seq ≤1.6e-14 单列；③ 修后实测 m 范围：官方数据 **m=1/64/4097 均为 0 NaN**，归档档 **m=1/4097**
（clean 与 mut）判据可跑；**未做设备验证**（本 mission 零设备）；④ `stab_demo` 的 fp32 变体只是离线对照，
不是新的判据路径。

**纪律自查（M165）**：零设备、零探针；改动集 = `m23_gdn_prefill/check_ref.py` + 本 README + `evidence/{logs,tools}/`；
**未改**任何已提交读数/日志/dump（`evidence/logs/` 里已有文件一字未动，新证据落新文件名）。
新增/改动行对 mission 六词禁用表逐字普查 = 未命中；六词表不在此复列（理由同 §4n）。
**结论带 文件:行**：`check_ref.py:91` 的 `ref_head`、`:114-117` 的指数差 Γ、`:136-138` 的指数差衰减、
`:118-121`/`:139-140` 的旧式（`--legacy-exp`）、`:259` 的 `stab_demo`、`:352` 的 `selftest`（`:391` 旧式对照）。

## 4x. 已知残留（r3 复审登记，**不抹掉**；本轮不修）

1. **P3**：守卫告警文本与行为不完全一致 —— 「部分陈旧」时打印的措辞比它实际做的（只剔除陈旧、照判新鲜）更重。
2. **P3**：`check_ref.py` 自动定位二进制的三个候选路径**都不存在**时，守卫**静默跳过**（不告警）。
3. **P3**：新鲜度比较用的是 `m23_<tag>_meta.txt` 的 mtime，**不是每个 `.bin` 各自**的 mtime。
4. **格位不足（最严对照还差一格，**保留登记**）**：反向对照（`--mutate 1`）走的是**与干净档相同的代码路径**，
   但两侧的 **salt 并不相同**（**干净档** `m1`/`m4097` `salt=0x2304`；**变异档** `m1_mut1`=`0x2301` / `p4097_mut1`=`0x2302`）
   *〔归属取证，供后来人自核〕* ① **驱动源码逐字**：case 表 `m23_gdn_prefill.asc:277-278` = `{"m1",48,1,0x2301u}` / `{"p4097",48,4097,0x2302u}`，
   而 `ONLY_M` 分支 `:315` = `one.salt = 0x2304u`；② **日志逐字**：干净档来自 `M15GP_ONLY_M=… ⇒ 单 case 运行`（`evidence/logs/r3_M1.log`），
   变异档来自 `M15GP_ONLY_P4097=(unset) ⇒ 本次跑 case [0, 2)`（`evidence/logs/r3_mut1.log`，tag 为 `m1_mut1`/`p4097_mut1`）；
   ③ **数据侧旁证**：`--mutate 1` 不动 q/k/beta，而干净档与变异档的 q/k/beta 内容不同（两组 sha256 全不同）。
   ⇒ 因此现有红读数**能**证明"判据对设备输入扰动敏感"，但**不能**排除 salt 差异带来的额外成分；
   **「同一 salt、只换一个输入」那一格仍然没有**（这正是"最严对照还差一格"的意思）。

以上均**非阻塞**，如实登记于此与 `evidence/logs/r2_rework_r1.md`。

## 5. 显式未完成项（本 mission 不做什么）

1. **融合进融合 TU**（`m15_layer_kernel.h` 的挂载 + `m15_layer_loop.asc` 的开关 + host 接线）——
   归 Wave C/D；本段只给补丁文本与清单。
2. **设备侧数值验收：已 PASS**（不再是未完成项）—— 新鲜档 `check_ref.py` **rc=0**，`m1` 与 `m4097` 的
   `o`、`ht` **全 0 超界**（`max|Δ|` 见文首状态块与 §4c0）；反向对照 `--mutant 1` **已红**。
   内核两处缺陷（L0C 的 `M↔FIXP` 成对交接、`StoreState` 的 `TransposeVF`）**已修并已被独立复跑验证**。
   *〔历史口径，仅供追溯，勿当现状〕* 修前本项的表述是"设备侧数值验收未通过/未跑通，主要未完成项"，
   定位过程（`narrow_r1.md` 的 J1-only OK / 全档 aivec、J2/J3 专属路径的追查、J1+J2-only → J1+J2+J3-only 的
   收窄路线）**已由 L0C 修复收口**，故原"下一步"作废，保留在此仅为追溯。
3. **端到端 prefill 全链验收**（P0–P11）、**decode 档回归**（m=1/ctx=4097 的链路）—— 未做。
4. **性能优化**（当前设计每 chunk-head 约 0.5MB GM 往返；没有任何调优）—— 未做。
5. **aclgraph**、**跨 48 层**、（人类裁决）**精度专项**—— 均不在本 mission。
6. `check_ref.py` 的 fp64 参考**不含** conv1d / l2norm / gating（边界所在，见 §1）。

7. **满量级重复统计**（M=16/17 那批按「先搁置」停着；M=1 的 50 次与三档对照已跑，见 `evidence/logs/m1_repeat_r1.md`）。
8. **P2-2 登记项**（AIV `if (hv >= heads_) continue` 可能让配对 AIC 永久等 GO，`m15_gdn_prefill.h:796`）—— **只登记未修**。
9. **参考侧已数值稳定化（M165）**：`check_ref.py` 默认 Γ 改为**指数差**（见 §4o）；旧式在 `--legacy-exp` 保留。
   **设备侧同一旧式写法未修**（`m15_gdn_prefill.h` 的 `CumSumExpVF`/Γ 体）—— 归 M162，本 mission 零设备。
   M151 真权重归档 `m4097` 的设备 NaN（`o` 8,374,400 / `ht` 262,144）因此在默认（指数差）判据下是"仅一侧 NaN" ⇒ FAIL。

## 6. 证据

| 文件 | 内容 |
|---|---|
| `evidence/logs/build_r1.log` | `cmake --build` 的落库读数（rc=0） |
| `evidence/logs/run_hang_r1.md` | 两次设备档未返回的记录（命令、上限、观测） |
| `evidence/logs/cc_probe_r1.md` | 核间握手探针的逐条读数（falsify 三个核间假设） |
| `evidence/logs/build_r2.log` | 修 `scSE` 真 bug（应为 `scale·eg` 逐 lane）后的构建读数 rc=0 |
| `evidence/logs/m165_repro.log` | **M165**：一键复现（`--selftest` / `--stab-demo` / 官方对齐 / 归档重跑）rc=0 |
| `evidence/logs/m165_official_align.log` | **M165**：与官方 m21 `gdn.core_out` 对齐的逐档读数（rc=0） |
| `evidence/tools/m165_official_align.py` | **M165**：官方对齐脚本（只读 m21 dump；rc 可传播失败） |
| `evidence/tools/m165_repro.py` | **M165**：一键复现 + 归档逐档核对（rc 可传播失败；外部归档缺失报 SKIP） |
