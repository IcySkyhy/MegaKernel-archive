# M18：GDN 线性注意力 prefill chunk 扫描核（m ≤ 4097）

Qwen3.8-Flash-Next GDN 线性注意力的 **prefill 路径**：把 decode（m=1）的逐 token 递推核
（`m4_gdn_recurrent`）换成 **分块扫描（chunked scan）**，支持 m 到 4097（= 64×64+1，尾块 ragged）。
每 (value head, key head) 独立做一次 BT=64 的分块扫描：chunk 内准备 WY 表示（A/KKT、Γ 衰减、
(I+A) 前代），chunk 间串行推进 fp32 状态 S。

**结论**：4 档用例（含目标档 H=48/m=4097）全部 PASS —— 与 numpy float64 逐句参考比较，
**逐段抽点**（ĝ/eg/A/u/w/d/AB/o 各段 + 每个 chunk 末的状态 + 端到端 o/state）。
档位 **T1（输入六张量逐位）+ T3（其余，ε=5e-5 推导见 §4.2）**：
**260 个判定行 / 76,432,326 个判定元素全部 0 超界**；报告项全局真值 = max|Δ| **8.799e-07**、
max|exp| **1.854**、max Σ|terms| **8.296**、**T3 最差容差占用 0.0158**（余量 63×）、
≤1ulp 比例 13.2%（最小）/ 16.0%（中位）。误差**不随 65 个 chunk 累积**（逐 chunk 状态每边界
~2e-7，与 chunk0 同量级）。m=4097 单层耗时 **15.51 ms**（48 head ∥ 48 AIV；活值，range 15.494~15.522，出处见 §4.7）。

> 口径按 `docs/17-verification-standard.md`：**只给一个 maxRel 不合格**，故本 README 的判据
> 一律按「判定项（超界数/总数，决定 PASS/FAIL）／报告项（max、比例、占用）／guard」分栏列出。

设计约束（用户定）：`mix(1,2)` 全核启动、只用基础 API、禁 TPipe/TBuf/TQue/AllocTensor、
buffer id / cross core id / 地址自己静态管理、尽量不挂 PIPE_S、功能优先从现有代码库抄改。

---

## 1. 数学（逐句伪码）

记号：`BT=64`、`DK=DV=128`、`hv` = value head、`hk = hv//3` = 对应 key head（模型 48=3×16）；
`S ∈ R^{DK×DV}`（**行 = K 维**，fla 约定）；`q,k ∈ R^{BT×DK}`、`v ∈ R^{BT×DV}`、
`g,β ∈ R^{BT}`；`scale` 为算子参数（`1/√DK`）。全部 fp32。

设 `ĝ = inclusive_cumsum(g)`（**chunk 内**前缀和，每 chunk 从 0 起，不跨 chunk 累加），
`eg = exp(ĝ)`、`ig = exp(−ĝ)`、`egL = eg[cv−1]`（cv = 本 chunk 有效行数），

```
Γs[i,j] = eg_i·ig_j  (j < i，否则 0)          # 严格下三角
Γi[i,j] = eg_i·ig_j  (j ≤ i，否则 0)          # 含对角
```

**chunk 内（只依赖本 chunk 输入，可并行）**

```
A[i,j] = β_i · Γs[i,j] · (k_i·k_j)            # KKT + 衰减 + β，严格下三角
u = (I+A)^{-1} · (β ⊙ v)                      # [BT,DV]
w = (I+A)^{-1} · (β ⊙ eg ⊙ k)                 # [BT,DK]
```

**chunk 间（串行状态扫描，S 从上一 chunk 继承）**

```
d      = u − w·S₀                                       # raw delta = fla 的 v_new
o      = ((q·scale)·S₀) ⊙_row eg  +  (Γi ⊙ ((q·scale)·kᵀ)) · d
d'     = d ⊙ ig
S₁     = egL · (S₀ + kᵀ·d')                              # 等价：S₁ = egL·S₀ + kᵀ·(d ⊙ exp(ĝ_last−ĝ_j))
```

`A/u/w` 是 WY 表示；`d` 是「本 chunk 内每个 token 的新增 delta」；`o` 的两项分别对应
「chunk 起始状态对当前 token 的贡献」与「chunk 内因果注意力项」。

> **M166 稳定化**：`S₁` 的第二式（指数差）**只**用于 `check_ref.py` 的 fp64 参考；本目录的
> `.asc` 设备侧仍用第一式（物化 `ig`），两者在 fp64 上差 ≤4.4e-16（§4.8(4)）。**为什么参考要用
> 指数差、它修了哪几处、以及 NaN/Inf 判据**见 §4.8。**§1 的伪码保留设备侧写法，不改。**

### 两个容易抄错的点（本实现踩过，已用判据定位）

1. **`w` 里带 `e^{ĝ}`，`d = u − w·S₀` 里用的是未衰减的 `S₀`。** 这是
   `/workspace/vllm-ascend/vllm_ascend/ops/triton/fla/chunk_delta_h.py:121-149` 的顺序
   （先 `b_v_new = b_v − b_w@b_h1`，之后才 `b_h1 *= exp(b_g_last)`）。fla 上游原版把衰减提到
   `w@h` 之前、且 `w` 里不带 `e^{ĝ}`，两者**等价但不可混抄**。本实现与 vllm-ascend 一致。
2. **`egL` 乘在整个括号上**：`S₁ = egL·(S₀ + kᵀd')`。写成 `S₁ = egL·S₀ + kᵀd'`（先缩放 S 再累加）
   会漏掉 `egL·kᵀd'`（实测误差 ~几十个百分点）。实现上等价的两条路：把 `egL` 折进 `d'`，或者
   **先累加 `kᵀd'` 再整体缩放 S**（本实现取后者，少一个 [BT] 向量 pass、数值也更好）。

### chunk 取值依据

| 依据 | 事实 |
|---|---|
| donor 取值 | arch35 `op_host/chunk_gated_delta_rule_tiling.cpp:120` 硬编码 `c = 64`；vllm-ascend `fla/chunk.py:51` `chunk_size = 64` |
| 上界约束 | arch35 的 cumsum VF 把整个 chunk 的 `g` 塞进**一个 fp32 寄存器**（64 lane @256B）→ `BT ≤ 64` 不需改 VF；BT≥128 要拆两寄存器 |
| 求逆代价 | arch35 的 `INVERSE_SHAPE=32` + AIC 2×2 块合并是为 BT=64 定制的；本实现用行前代，代价 O(BT²·128) 线性于 BT |
| 本实现 | `constexpr uint32_t BT = 64;`（唯一调优旋钮，改一个常量） |

**ragged 尾块**：m=4097 = 64×64+1，末块 cv=1（真用例，已验证）。处理：把 padding 行的
`g/β/k/q/v` 全部置 0 —— 则 padding lane 的 `ĝ` 是常数（`=ĝ[cv−1]`，无溢出）、`β=0` 让该行
不参与、`Γ`/`A` 的 padding 行列恒 0、`q=0` 让输出行为 0。**不需要任何掩码**，`egL` 直接取
`eg[BT−1]`（等于 `eg[cv−1]`）。

---

## 2. 布局

### 输入/输出（GM，fp32，per-head 连续）

| 张量 | 形状 | 说明 |
|---|---|---|
| q | `[NK, T, 128]` | 已 l2norm（**纯单位化**，scale 由 kernel 内乘） |
| k | `[NK, T, 128]` | 同上 |
| v | `[H, T, 128]` | conv1d + SiLU 后（本核外） |
| g | `[H, Tp]`，`Tp=ceil(T/8)·8` | log 衰减 ≤ 0；行 stride 按 32B 对齐（GM 起始地址必须 32B 对齐） |
| beta | `[H, Tp]` | sigmoid gate ∈ (0,1) |
| h0 / ht | `[H, 128, 128]` | 初始/最终状态，**行 = K 维** |
| out | `[H, T, 128]` | o |

### UB 静态布局（字节偏移，全部 32B 对齐，总 225.5KB ≤ 248KB）

| 偏移 | 内容 | 谁写→谁读 |
|---|---|---|
| 0 | `k` `[64,128]` 32KB | MTE2 → V（整 chunk 只读） |
| 32768 | `q` `[64,128]` 32KB → `q·scale` | MTE2 → V |
| 65536 | `v` → RHS_k → `w` → `o_part` → `o` 32KB | MTE2/V → V/MTE3 |
| 98304 | `u` → `d` 32KB | V |
| 131072 | `KKT` → `A` → `AB` `[64,64]` 16KB | V |
| 147456 | `Γs` / `Γi` `[64,64]` 16KB | V |
| 163840 | 状态 `S` `[128,128]` fp32 64KB（**跨 65 chunk 常驻**） | MTE2/V → V/MTE3 |
| 229376 | 标量区：`g / β / ĝ / eg / ig / β⊙eg` 各 [64] 共 1536B | MTE2/V → V |

UB 缓冲刻意**复用**（`v→w→o_part→o`、`u→d`、`KKT→A→AB`、`Γs→Γi`），靠 §3 的令牌把
「MB 读走 o」与「下一 chunk 的 MTE2 复写 v」串起来。

---

## 3. 同步表

只用 **BufferID**（`GetBufInternal<pipe,false>` = `get_buf`，`RlsBufInternal<pipe,false>`
= `rls_buf`，release 用 `false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」、`true` = `NON_BLOCK`；两种模式都等本 pipe 已发射指令落地），无 `set_flag/wait_flag`、不挂 PIPE_S。

| BufferID | 交接（生产者 → 消费者） | 保护的资源 |
|---|---|---|
| `BUF_BLK` = 0 | **MTE2 → V → MTE3 → MTE2**（一个 chunk 的 UB 工作集令牌，三方轮转） | k/q/v/g/β/UB_SC 的装载、整个 chunk 的 V 计算、o 的 MTE3 写出 |
| `BUF_ST` = 1 | MTE2（装 S）→ V | 初始状态 |
| `BUF_SO` = 2 | V（S 终值）→ MTE3（回写 S） | 最终状态 |
| `BUF_PR` = 3 | V → MTE3（抽点）→ V（等读完成） | probe 抽点（probe 跑法可慢，保证读完成才继续） |

`BUF_BLK` 之所以要三方轮转：`UB_V` 同一个 buffer 先装 `v`、后被 `o` 覆盖，
**必须等 MTE3 读完 o 才能给下一 chunk 的 MTE2 复写**；用一个令牌把「装载→计算→写出」
串成链，既保证可见性又不需要额外 flag。同 pipe 背靠背拷贝不保序（M1 实测裁决）在本核不构成
问题：每次 DataCopy 都换 buffer/地址，靠令牌交接。

**其它同步点**：VF 内 UB store→load 依赖用 `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()`
（= `mem_bar(VST_VLD)`，arch35 同款）；所有 `LoadAlign<DIST_BRC_B32>` 读 UB 标量前都先阻塞释放（等本 pipe 已发射指令落地）。
AIC 侧不发起任何访存（见 §AIC 升级路径），因此**没有 cross-core flag**，也就没有
「pipe 类必须匹配核型」那类挂死风险。

---

## 4. 判据与结果（口径按 docs/17-verification-standard.md）

> **基准性质声明（docs/17 §7）**：本节所有 m=4097 的数值验收基准是**自建 numpy float64 参考**
> （GDN 段公式逐句对齐官方 golden，见 §10），**不是「与官方 vLLM 输出对齐」**。GDN 段不涉及
> QSA 稀疏选带，故 docs/17 §7 那条「官方 QSA 只 attend 约一半历史 ⇒ 自建稠密 causal 参考与官方
> 输出存在固有差异」**不构成本节结论的偏差来源**；引用本节数字时必须同时说明「基准是自建参考」，
> 不得表述为「m=4097 对齐官方输出」。

`check_ref.py` 用 numpy **float64** 把 §1 的公式逐句重写（同一 chunk 边界、同一 padding 语义、
同一运算顺序：β/eg/Γ 的组装顺序、先累加后整体缩放等），并：
1. 用 numpy 重生成输入与 C++ dump 的输入做 **T1 逐位交叉核对**（两侧生成器用**同一套**
   float64 顺序累加 + 46 轮二分求 1/√Σx²，故应 bit-exact；见下方"0 误差"的正确写法）；
2. 逐段抽点比较：`ĝ / eg / A(缩放后·求解前) / u / w / d / AB / o`（抽 chunk 0、chunk 1、末块）
   + **每个 chunk 末的 S**（抽 head 0/1，全部 65 个 chunk）+ 端到端 `o`/`state`。

> **M166 起，参考的跨 chunk 衰减改用指数差**（§4.8），判据**显式纳入 NaN/Inf**（§4.8(2)）。
> 下面 §4.3–§4.7 的判定项/报告项/guard/性能数字是 **M34 归档读数**（参考为改前的旧式）；
> §4.8(4) 实测：两式在该归档数据面上差 ≤4.4e-16、Σ|terms| 差 ≤1.4e-15 ⇒ 归档数字保持其打印精度有效。

### 4.1 档位声明（docs/17 §1.1）

| 对象 | 档 | 理由 |
|---|---|---|
| 输入六张量 q/k/v/g/β/h0 | **T1 逐位** | kernel 不碰它们；两侧生成器同一套 double 运算 ⇒ 期望 bit-exact，容差 0 |
| 其余全部（ĝ/eg/A/u/w/d/AB/o/S/ht） | **T3** | ①累加链深度 ≥64（GEMM inner=128、行前代 64 步、状态链 65 chunk）；②含 **VF `Reg::Exp`**（官方规格「最大精度误差 1 ulp」，`reg_vector_compute/basic_arithmetic/Exp.md:77`）；③非 T2（参考无法"完整建模每一步舍入"：设备侧向量加法/FMA 的具体结合顺序不可见） |

T3 判据：`|got − exp| ≤ ε·Σ|terms| + 0.5·ulp_fp32(exp)`，**逐元素**检查。
Σ|terms| = **一层 + 传递**口径：该元素的直接项之和；若某项里的操作数是"前级算出来的量"，
则代入**它自己的一层项和**（把上游舍入向前带），但**不递归展开**——递归展开会让状态链
形成正反馈、容差指数膨胀。**该膨胀已归档可复算**：`check_ref_log.txt` 的
`[terms-口径对照]` 行给出同一份数据下两种口径的 max Σ|terms|(S) —— target 档（65 chunk）
一层+传递 = **1.294e+00**，递归展开 = **8.338e+41**（比值 6.44e+41 ⇒ 界完全失去意义），
故口径明确为「一层+传递、不递归展开」。

### 4.2 ε 推导（docs/17 §1.1 护栏 2：逐项来源与数值）

| 来源 | 依据 | 数值 |
|---|---|---|
| fp32 单位舍入 u | IEEE754 binary32（尾数 24 位） | u = 2⁻²⁴ = 5.96e-8 |
| 最长累加链 | GEMM inner = 128 步顺序 FMA ⇒ γ₁₂₈ = 128u/(1−128u) | **7.63e-6** |
| VF `Exp` | 官方 spec：最大精度误差 **1 ulp** ⇒ 相对 2⁻²³ | **1.19e-7** |
| 状态链 65 chunk | 每 chunk 1 次 `Exp`(egL) ⇒ 65·δ_exp（最坏串联） | **7.7e-6** |
| 行前代累加链 | 行前代 64 步 ⇒ γ₆₄ = 64u/(1−64u) | **3.82e-6** |
| **行前代放大** | 三角解条件数 **κ∞ = ‖(I+A)⁻¹‖∞ ≤ 3.212**（矩阵的**精确性质**，由 `audit_readme_numbers.py` 从归档 A 复算 —— 不是误差观测）。**不采用** Neumann 界 1/(1−‖A‖∞)：实测 **‖A‖∞ = 2.31 > 1**，该界不成立 | ×3.21 |
| Γ/eg 的 exp 传递 | 2×exp + 1 mul（A = β·Γ·KKT 一项里） | 3.0e-7 |
| **合计** | κ∞·(γ₁₂₈+γ₆₄) + 65·δ_exp + 3e-7 = 4.48e-5 ⇒ 向上取一位有效数字 | **ε = 5e-5** |

（ε 取整到 5e-5 而非紧贴 4.48e-5，是为了给上面未逐项列出的零散舍入留余量。
**本轮 round-3 的数字对账发现**：此处曾写"三角解条件因子 1/(1−‖A‖∞)，实测 ‖A‖∞ ≤ 0.05 ⇒ ×1.05"，
对账脚本从归档 A 重算得 ‖A‖∞ = 2.31、κ∞ = 3.21 —— 该写法既数值错、方法也错（Neumann 界要求
‖A‖∞<1）。已改为按 κ∞ 推导，ε 随之由 2e-5 提到 5e-5（判定结论不变，仍全 PASS；最差占用按 ε 同比变小）。）

**κ∞/‖A‖∞ 的归档出处（M56 修）**：这两个量本要从 `build/*_probe.bin` 复算，而 **dump 不入库**
（`.gitignore` 里 `build/`、`*.bin`），所以干净 checkout 上算不出来。已把复算结果连同三个 dump 的
`sha256` 一起归档到 `evidence/probe_geometry.txt`（`A_inf = 2.313775544`、`kappa_inf = 3.211607375`，
算法与偏移写在文件头）。`audit_readme_numbers.py` 以**归档为权威**对账；`build/*_probe.bin` 存在时
再从原始字节**独立复算**作交叉见证。**归档缺失 ⇒ 这四条断言（‖A‖∞/κ∞/ε 合计/ε 取值）报
`UNVERIFIED` 并使脚本 rc=2，绝不回退到已作废的 Neumann 口径**。

### 4.3 判定项（决定 PASS/FAIL；**只含判定项**）

| case | H | T | chunk 数 | 判定行数 | 判定元素总数 | 超界数 | verdict |
|---|---|---|---|---|---|---|---|
| `gqa3` | 3 | 257 | 5（末块 cv=1） | 42 | 650,118 | **0** | PASS |
| `one` | 8 | 64 | 1 | 18 | 517,248 | **0** | PASS |
| `all48` | 48 | 129 | 3（末块 cv=1） | 38 | 3,920,352 | **0** | PASS |
| `target` | 48 | 4097 | 65（末块 cv=1） | 162 | **71,344,608** | **0** | PASS |
| **合计** | | | | **260** | **76,432,326** | **0** | PASS |

判定行按对象逐个列出（格式 `超界/总数`，计数可由
`grep -c` + `awk` 从 `evidence/check_ref_log.txt` 逐项复算）：六张输入张量（T1，target 的
`q 0/8390656`、`k 0/8390656`…）、`o[all_heads,tok] 0/25171968`、`ht[final_state] 0/786432`、
**65 个 chunk 状态 `S[head0/1,chunk0..64]` 各 0/16384**（130 行）、以及分段抽点
`ĝ/eg/A/u/w/d/AB/o` × (chunk0 / chunk1 / 末块)。完整逐行见
`evidence/check_ref_log.txt`（605 行；含 `[判定项]/[报告项]/[guard]` 三栏）。

### 4.4 报告项（不参与 PASS/FAIL；数字口径三件套：max、绝对/相对、哪一档输入）

**全局真值（4 档全部比较对象，260 行）**：

术语（本文统一）：**占用** = max(|Δ|/tol)（≤1 才合格）；**余量** = 1/占用 = tol/max|Δ|。

| 报告项 | 值 | 出处（case / stage；260 行中取 max） |
|---|---|---|
| **max\|Δ\|（max、绝对）** | **8.799e-07** | **target** / `d(v_new)[chunk0]`，该元素 Σ\|terms\|=1.975 |
| maxRel（max、相对） | 1.133e+01 | **target** / `o[all_heads,tok]`，相消区元素（分母 ≈0），**勿读成判定** |
| **max\|exp\|（max、绝对幅值）** | **1.854** | **one** / `gcum[chunk0]` |
| **max Σ\|terms\|** | **8.296** | **gqa3** / `d(v_new)[chunk1]` |
| **T3 最差容差占用** | **0.0158**（余量 63×） | **one** / `d(v_new)[chunk0]` |
| T3 占用中位 | 4.60e-03 | 全部 260 个报告行 |
| ≤1ulp 比例（min / 中位 / max） | 13.2% / 16.0% / 100% | max 出现在 `ĝ`（单步量，误差 ~1 ulp） |
| **相对 \|out\| 口径占用（同 ε，仅作反例）** | **2.263e+05** | **target** / `o[all_heads,tok]` |

关于「相对 \|out\| 口径」这一列（round-1 用的就是它）：它把分母取成 |参考输出|，于是在
**相消元素**上分母趋 0（参考输出本身接近 0），占用可以变成 2.26e+05 —— 同一份数据、同一个 ε，
该口径**完全不可判定**。round-1 报的「4.1×」就是它在**另一个 oracle 的 |exp| 分母 + 绝对地板
1e-6** 下的取值；两组数的**绝对偏差完全一致（8.799e-07）**，差别只在分母。这正是
`docs/17` §1.1 把长 fp32 链从「≤1 ulp / 相对 \|out\|」改判为 `ε·Σ\|terms\|` 相对界的原因：
Σ\|terms\| 随相消一起变大，分母不会趋 0。
本 README 的**判定只认 T3 占用**（最差 0.0158 ⇒ 余量 63×，即实测误差比推导界小 63 倍）。

**target 档关键行**（`max|Δ| / maxRel / ≤1ulp / T3占用 / maxΣ|terms| / max|exp| / 相对|out|占用`，
**逐字取自归档** `evidence/check_ref_log.txt`，可用脚本重抽比对）：
```
o[all_heads,tok]  4.737e-08  1.133e+01  13.5%  2.344e-03  2.096e+00  6.389e-02  2.263e+05
ht[final_state]   2.553e-07  6.171e-02  15.6%  5.682e-03  1.294e+00  5.988e-01  1.233e+03
S[head0,chunk0]   2.302e-07  1.015e-03  16.0%  5.702e-03  1.006e+00  5.288e-01  2.028e+01
S[head0,chunk64]  2.038e-07  5.760e-03  15.2%  4.556e-03  1.163e+00  4.723e-01  1.151e+02
```
**逐 chunk 状态**（head0/1 的 65 个 chunk 边界，共 130 行）max|Δ| 全部落在
**1.536e-07 ~ 2.684e-07**（148 个 S 报告行跨 4 档的全局范围也是同一对端点）——
与 chunk0 同量级，即**误差不随 chunk 数累积**（状态每 chunk 乘 egL<1，误差被同因子衰减）。

### 4.5 guard（不参与 PASS/FAIL；docs/17 §4 非空洞性）

| guard | 检查什么 | 结果 |
|---|---|---|
| G1 输入非空洞 ×6 | 每个输入 max\|·\|>0 且 std>0（防"恒输出常数也能过"） | PASS（target: q max 0.186/std 0.088，v max 1.0/std 0.577） |
| G2 状态跨 chunk 演化 | 相邻 chunk 的 S 必须不相等；计数须等于 (head,chunk) 边界数 | PASS 8/8（gqa3）、4/4（all48）、128/128（target），`one` 为 N/A（nc=1） |
| G2 终态≠初态（全部 head） | ht[h] ≠ h0[h]（状态确实演化） | PASS 3/3、8/8、48/48、48/48 |
| G3 A/AB 严格上三角恒 0 | 结构性：WY 矩阵必须严格下三角 | PASS（max\|上三角\|=0.0） |
| G3 ĝ 单调不增 | g≤0 ⇒ cumsum 单调不增 | PASS（递增位置数=0） |
| G3 ragged 尾块 padding 行=0 | 末块 cv=1 ⇒ o 第 1 行起必须为 0 | PASS（max\|·\|=0.0） |
| G3 probe 未用 chunk 槽位=0 | dump 位置正确：chunk≥nc 的槽位必须未被写过 | PASS（max\|·\|=0.0） |
| G3 probe 槽位可区分 | 前 9 个 chunk 的 S 互不相同（防索引错位） | PASS |
| G4 确定性（dump sha256 对照） | 与 `evidence/dump_sha256.txt` 逐文件比对 | PASS ×4 |
| G5 确定性实测（两次独立运行） | 第二次独立运行的 40 个 dump sha256 与第一次**逐行相同** | PASS（`evidence/dump_sha256_run2.txt`） |

### 4.6 本次验证**证明什么 / 不证明什么**（docs/17 §3.5）

**证明**：
- 算法正确：4 档（含 65 chunk + ragged 尾块）的逐段抽点与端到端 o/ht 全部落在 T3 界内；
  §1 的公式（含两个易错点：`w` 带 `e^ĝ`+未衰减 S₀、`egL` 乘整个括号）与 numpy float64 参考一致。
- 结构正确：A/AB 严格下三角、ĝ 单调、ragged padding 为 0、probe 槽位可区分（G3）。
- 状态语义正确：S 跨 chunk 确实演化、终态≠初态（G2）；误差不随 chunk 累积（4.4 末段）。
- 确定性：两次独立运行的 40 个 dump **逐字节相同**（G5）。
- 输入契约：六张输入由 numpy 独立重生成后与 dump **逐位相同**（T1 判定项 0 超界）。

**不证明**：
- **不证明生产精度**：本核全程 fp32，而生产路径（donor/vllm-ascend）是 bf16 操作数 + fp32 累加。
  T3 的 ε 是按 fp32 推导的；换 bf16 需重推 ε 并重跑（见 §已知限制 4）。
- **不证明性能充分**：15.51 ms/层是纯向量域数字，与 HBM 界差 ~39×（§性能）；AIC 空转未验。
- **不证明与官方算子的逐位一致**：L0 判据只能证明"与本仓库的 float64 逐句参考一致"；
  与 `chunk_gated_delta_rule` aclnn/triton 的位级一致性属 L2 级，不在本 mission 范围
  （reviewer 用官方 golden 另写 oracle 复算过设备输出并得出一致结论，但那仍不是位级等同）。
- **不证明端到端可用性**：conv1d/l2norm/gating 在核外，本核只吃 post-conv 的 q/k/v 与算好的 g/β。

### 4.7 性能（同机实测，含口径）

**耗时是活值（跨运行会变）**，下表逐字取自 3 份归档运行日志（`evidence/run_log.txt` /
`run_log_run2.txt` / `run_log_run3.txt`，各 12 行，行号见括号）：

| 用例 | 规模 | run1 | run2 | run3 | 稳定性 |
|---|---|---|---|---|---|
| `gqa3` | 3 head × 5 chunk | 4640.499（`:2`） | 1.272（`:2`） | **80044.004**（`:2`） | **不稳定（3 次差 6e4 倍）** |
| `one` | 8 head × 1 chunk | 0.254（`:4`） | 0.260（`:4`） | 0.257（`:4`） | 稳定（±1.2%） |
| `all48` | 48 head × 3 chunk | 0.729（`:6`） | 0.732（`:6`） | 0.734（`:6`） | 稳定（±0.7%） |
| `target` | 48 head × 65 chunk | 15.514（`:8`） | 15.522（`:8`） | 15.516（`:8`） | 稳定（±0.05%） |
| `target`（probe 关） | `probeHeads=0` | 15.494（`:11`） | 15.503（`:11`） | 15.506（`:11`） | 稳定 |

单位 ms；括号内为归档行号。

- 单层 GDN chunk 扫描 **15.51 ms**（3 次运行 15.514 / 15.522 / 15.516，probe 关 15.494~15.506）；
  按 36 个 GDN 层外推 **≈559 ms/次 prefill**（36 × 15.51 ms）。
- **计时不是上界，且对机器负载极敏感**：`gqa3` 是每次进程的**首个小档**（含首次启动开销），
  3 份归档里分别是 4640.499 / 1.272 / **80044.004** ms —— 同一 commit、同一二进制差 6e4 倍。
  故**只把 `one`/`all48`/`target` 当量级参考**（三者跨 3 次运行波动 ≤1.2%），
  任何单次计时都不构成性能结论。本机 NPU 与 8+ worker 共用。
- 每 chunk（48 head 并行）≈238 µs（= 15.514 ms / 65，由归档 target 计时算出），折合
  **0.83 G 向量指令/s/AIV**；**按假定时钟 1.8GHz 折算 ≈0.46 指令/cycle**（时钟是假设，未在本
  卡上标定，属"假设值"而非实测；改时钟只影响这一行的 cycle 换算）。
- 指令数模型：每 chunk-head ≈ **2.0×10⁵** 条向量指令（1024 条 lane 的 fp32 寄存器，1 指令 = 64 lane）。
  其中行前代两次占 ~20%（§已知限制里那条 quirk 让它从 BT²/2 变 BT²）。
- 两个乘性损失各约 2×：①算法指令效率（必要 MAC 5.8M/64 = 9.1 万指令，实跑 20 万，2.2×）；
  ②流水线实际发射率 0.46 指令/cycle（受 UB 装载延迟 + BRC/Gather 依赖 + 每 chunk 128 次
  `mem_bar` 拖累）。参考上界：docs/10 §3 未融合激活流量 ~400 µs/层 —— 本核与 HBM 界差 ~39×，
  **纯向量域跑矩阵乘就是这个量级**（§AIC 升级路径 是唯一量级级别的出路）。

---

## 4.8 参考侧数值稳定化与 NaN/Inf 纳入判定 —— M166（零设备）

**背景（人类原口径，逐字）**：「那就说明这个golden的生成脚本或者公式或者数值范围本身就很病态，你应该看看别人类似的kernel的golden怎么生成的」。
M160 survey 把 m=4097 非有限定位到**写法**：chunk 扫描把 `eg=exp(ĝ)` 与 `ig=exp(−ĝ)` **分开物化再相乘**，
chunk 内 `|ĝ|` 超 `exp` 上溢阈值（fp64 709 / fp32 88.7）时 `0·inf=NaN`；而正确因子 `exp(ĝ_i−ĝ_j)≤1`
（`i≥j`，有限输入下恒有限）。官方 FLA 用的就是指数差
（`vllm/.../flash_linear_attention/ops/chunk_o.py:119-120`、`chunk_delta_h.py:216-221`）。
M165 修了 m23 的参考（commit `600b05e`），并在全仓普查里点名本文件为同型（finding 未越界改）。
**本 mission（M166）修 m18 的 golden，零设备；本目录设备侧（`m18_gdn_prefill.asc`）未改，见 (5)。**

### (1) 修了哪几处（`check_ref.py`；「改前」= base 版本行号）

| 处（改前行） | 旧式 | 默认（指数差） |
|---|---|---|
| `:199` `ig = np.exp(-gc)`（`:197` 是 `eg`，保留） | 物化 `ig=exp(−ĝ)` | 删；改 `:265` `decay = np.exp(gc[cv−1] − gc)`（`[H,BT]`，`≤1`） |
| `:233` `igd = d * ig[...]` | `d⊙ig` | `:300` `igd = d * decay[...]`（因子换成指数差） |
| `:245` `S = egL*(S + ktd)` | 先加后整体缩放 | `:314` `S = egL*S + ktd`（衰减并进 `decay`，避开 `0·inf`） |
| `:246` `tS = |egL|*(tS + t_ktd)` | 同上 | `:315` `tS = |egL|*tS + t_ktd`（Σ|terms| 同步） |
| `:242` `t_igd_rec = t_d_rec * t_ig` | 递归演示口径 | `:309` 同步换 `t_decay` |

旧式保留在 `--legacy-exp`（`:267` `decay = np.exp(-gc)`，`:319-320` 旧式 `S/tS`），供修前/修后对照。

**未改的同类点（安全）**：`eg=exp(ĝ)`（改前 `:197`，现 `:259`）仍用于 `w` 的右端（`:278` `Rk = kk*(bb*eg)`）
与 `o_part`（`:288`）：`ĝ≤0 ⇒ eg≤1`，本就不上溢 —— 与 m23/M165 的处理一致（其 §4o(1)「未改」栏同）。
**Γ（`Gs`/`Gi`）本就用 `exp(ĝ_i−ĝ_j)`（改前 `:203`/`:223`，现 `:270`/`:290`），无需改。**

M165 的判定「m18 的 Γ 已稳定、但状态更新仍乘 `ig`」**属实**：改前 `:233`（`d⊙ig`）与 `:245`
（`egL·(S+kᵀ(d⊙ig))`）就是那句；改前 `:199`/`:200` 的 `ig` 只被这两处用。

### (2) 判据：NaN/Inf 显式纳入判定（口径与 M157/M165 一致）

* 新增 `cat_of()`/`nf_stats()`（`check_ref.py:80,93`）：按 有限 / NaN / +Inf / −Inf 归类 ——
  **NaN↔NaN、同号 Inf↔同号 Inf 视为相等**（不计错，但**不**由此断言该处数值已验）；
  **仅一侧 NaN、仅一侧 Inf、NaN↔Inf、异号 Inf 计为不匹配**。判定量 = 有限子集容差超界 + 非有限不匹配。
* `Judge.chk`/`Judge.chk_exact`（`:354,390`）改为「容差只在**两侧皆有限**处比 + 非有限不匹配计入 `bad`」；
  `Judge.dump`（`:414`）新增 **[非有限面]** 栏（`:425`），逐行给
  `devNaN/refNaN/仅一侧NaN/双侧NaN/devInf/refInf/仅一侧Inf/异号Inf/nfd`；`max|Δ|` 只在两侧皆有限处取
  （旧写法 `max(finite,nan)` 会把含 NaN 的元素折掉、掩盖读数）。
* **适用范围与残余空档（写窄）**：两侧皆 NaN 的位置不计错、也**不**构成数值背书；有限子集容差不覆盖非有限面。

### (3) 负向对照（`--selftest`，零设备；三个必红 + 一个干净档）

逐字读数见 `evidence/m166_selftest.log`（rc=0）。合成 dump 走**与真判据同一条** `run_case` 路
（`synth_dump` `:601`、`selftest` `:640`）：

| 档 | 构造 | 判定 | 关键读数（归档逐字） |
|---|---|---|---|
| `syn` 干净 | 设备 = 稳定参考 | **PASS** | `ht 0/49152`，非有限栏各项为 0 |
| `syn_overflow` + 旧式 | `g[0,0] = −900` | **FAIL** | `ht refNaN=16384 仅一侧=16384 nfd=16384`（`0·inf` 现场，`m166_selftest.log:131-132`） |
| `syn_overflow` + 指数差 | 同一输入 | **PASS** | `ht 0/49152`，非有限栏各项为 0（修法见证） |
| `syn_devmut` 设备 NaN | 设备侧 `o[0,0,0:16]=NaN` | **FAIL** | `o devNaN=16 仅一侧=16 nfd=16` |
| `syn_chunkmut` 错一个 chunk 边界 | 设备取自 T−1 的参考（T=65） | **FAIL** | `o 383/24960`、`ht 49085/49152`（有限子集红） |

⇒ 同一份上溢输入：旧式参考自造 16384 个 NaN（判定红），指数差参考 0 个（判定绿）。`--legacy-exp` 可切回旧式复现。

### (4) 回归（零设备）：m18 设备 dump **未归档**，设备档不可重跑

**取证（写窄）**：`git ls-files m18_gdn_prefill` 无任何 dump；`m18_gdn_prefill/.gitignore` = `build/`、`*.bin`；
`git log --all --diff-filter=A -- 'm18_gdn_prefill/**/*.bin'` 计数为 0（从未入库）；
`find /workspace/ascend_mega_kernel -name 'm18_h*' -type f | grep -v '\.asc'` 输出为空（仓内各 worktree 均无）。
（`/tmp` 下同名 `m18_h*` 只可能是本节 `--selftest` **自己合成**的 `/tmp/m18_nan_selftest_*/m18_h*`——
脚本跑完即清，**不是**归档 dump；故取证命令按**仓根**限定范围，不按 `/`。）
与 m23「交叉」的归档（`m15_layer_loop/evidence/m151_prefill_prolog_wiring/dumps_*`）是 **m23 格式**
（`m23_Pf.*`），本目录判据读 `m18_h<H>_t<T>_*` 前缀，**读不了**；M165 已用 m23 判据重跑过那些档（其 README §4o(4)）。

因此设备侧「逐档新旧判定差异」在本 mission **给不出**（不是"没复现"：dump 未归档，且零设备不能重跑）。
可做的**参考侧**回归已做，逐字见 `evidence/m166_ref_regress.log`（rc=0）：

| case | H | T | nc | 旧式 refNaN | 指数差 refNaN | max\|Δ\|o | max\|Δ\|ht | Σ\|terms\| 相对差 to / tS |
|---|---|---|---|---|---|---|---|---|
| gqa3 | 3 | 257 | 5 | 0 | 0 | 3.469e-17 | 2.776e-16 | 7.564e-16 / 1.073e-15 |
| one | 8 | 64 | 1 | 0 | 0 | 0.000e+00 | 3.886e-16 | 0.000e+00 / 1.345e-15 |
| all48 | 48 | 129 | 3 | 0 | 0 | 3.816e-17 | 4.441e-16 | 7.779e-16 / 1.392e-15 |
| target | 48 | 4097 | 65 | 0 | 0 | 4.857e-17 | 3.886e-16 | 9.061e-16 / 1.226e-15 |

- 输入由 `gen_case` **逐位重生成**（§4.1 的 T1 判定项已证与 C++ host 侧 bit-exact），故这是**同一输入面**上的两式对照。
- 两式的 o/ht 差 ≤ **4.4e-16**、Σ|terms| 相对差 ≤ **1.4e-15**，且各档两式 refNaN 均为 0
  ⇒ 在 m18 现有测试数据（`g∈[−0.051,−0.001]`，chunk 和 ≤3.3，**远低于任何上溢阈值**）上，改法**不改判定**；
  `decay_j = exp(ĝ_L−ĝ_j)` 在实数意义下恒等于 `egL·ig_j`，故 §4.4 的归档报告项数字保持其打印精度有效，**不重定基**。
- **归档读数 `evidence/check_ref_log.txt` 逐字复查**：`grep -in 'nan\|inf' evidence/check_ref_log.txt` 输出为空
  ⇒ 该归档数据面本无非有限元素，故新判据新增项在该数据上计数为 0、旧 PASS 不翻。
- **不得**把「设备侧 dump 未归档」读成「设备档已复验」：本轮的绿只到**参考侧**与合成自检。

### (5) 还有哪些同型写法未改（写窄）

| 文件:行 | 形态 | 判定 / 归属 |
|---|---|---|
| `m15_layer_loop/evidence/m151_prefill_prolog_wiring/check_phaseA_nan.py`（**M162 订正版**） | 默认参考**已不是**旧式 `eg*ig`：`ref_head_dt`（`:48`）默认走指数差（`:78-80`、`:98`）；旧式 `eg*ig` 仅存于 `naive=True` 对照分支（`:74-76`、`:96`） | **已改**（M162 `e088afb`，已合入 main）：原先「fp64 参考 NaN 是设备 NaN 子集 ⇒ 精度产物」的结论已被 M160/M165 推翻，M162 随之把默认参考订正为指数差 |
| `m18_gdn_prefill/m18_gdn_prefill.asc`（设备侧：`:110` `UB_SC_IG` 注释、`:171` `ig = exp(−ĝ)`、`:200` Γ 注释、`:606` `CumSumExpVF` 调用） | 设备侧仍物化 `ig`（与改前的参考同型） | **未改**：本 mission 只修 golden（零设备）；设备侧改动需设备验证，属独立后续（m18 设备档的 `g` 上界使它当前不触发，生产量级会触发，见 §4.8(1) 的 m23 同病） |
| `m4_gdn_recurrent/check_ref.py:46`、`m14_gdn_layer/check_ref.py:382`、`m21_layer_ref/ref/gdn.py:50,172` | 单侧 `exp(g)` 衰减（`g≤0 ⇒ ≤1`），无 `eg*ig` 乘积 | 形式上相似但安全（M165 §4o(2) 已核） |

### (6) 复现（零设备）

```bash
bash m18_gdn_prefill/evidence/m166_repro.sh   # [1]--selftest [2]--ref-regress [3]同型写法普查；rc 可传播失败
```

产物：`evidence/m166_selftest.log`（rc=0）、`evidence/m166_ref_regress.log`（rc=0）、`evidence/m166_same_style_scan.log`、
`evidence/m166_readings.txt`（前两份读数的 machine-readable 归档，供 `audit_readme_numbers.py` 的 PART2 分类取用）。

**README 数字对账（M166 在加了本节后的复跑）**：`audit_readme_numbers.py` 走**三态、rc 可传播失败**：
正式合格证 `compared 77/77 … PART3 0 行` / **rc=0**；`--negative-control` 的 A+B 两组对照均**有效**且脚本自身 rc=0；
`--no-geometry` **rc=2、NOT CERTIFIED**。逐字读数见 `evidence/m166_audit_readings.log`，正式合格证另存
`evidence/m166_readme_number_audit.md`。**M56 那三份归档合格证（`readme_number_audit.md` /
`_negative_control.md` / `_no_geometry.md` / `readme_number_audit_readings.log`）本轮一字未动**
（其 PART2 普查行数 259 是 pre-§4.8 README 的快照）；这里的 `m166_*` 是同一脚本在**当前 README** 上的新读数。

**限度**：① 只到参考侧——设备档未归档、本 mission 零设备；② 覆盖 m18 的 4 档（含 4097）+ 合成小档，未覆盖真权重数据面；
③ 上溢合成档的 `g=−900` 是**量级构造**（复现 `0·inf` 机制），不是真权重读数。

---

## 5. 与 donor 的差异

| | donor（arch35 MIX_AIC_1_2 三段式 / vllm-ascend） | 本核 |
|---|---|---|
| 资源管理 | TPipe/TBuf/TQue + AllocTensor | 全静态地址 + BufferID（无框架资源管理） |
| 求逆 | 物化 `T=(I+A)^{-1}`：AIV 32×32 前代 + AIC 2×2 块合并（`InverseAIVVF<32>`），再 cube GEMM 求 T@RHS | **不物化 T**：直接对两个右端做行前代 `(I+A)X=RHS`（`TrilSolveVF`；`InverseAIVVF` 是它取 RHS=I 的特例）——同公式、flop 更少、免单位阵/免块合并 |
| 矩阵乘 | cube（bf16 操作数 + fp32 L0C 累加，fixpipe） | AIV 向量域 fp32（三个小 GEMM 原语 GemmAA/GemmBT/GemmTA，每 (row,t) 一次 BRC + 一次 FMA） |
| dtype | bf16 操作数 + fp32 累加（生产精度目标） | 全程 fp32（对齐 fp64 紧容差判据；见 §已知限制） |
| 结构 | 3 段（stage1/2/3）+ chunk group + SyncAll 屏障，h 走 GM 原子累加 | 单核内 fuse 全流程，**h 常驻 UB 跨 65 chunk** |
| 相似度 | — | 每 (value head, key head) 独立；BT=64；公式逐句一致（含 `w` 带 `e^ĝ`、`egL` 整体缩放两个易错点） |

---

## 6. decode 复用性：本核与 m4 递推核的关系

**代数上本核在 BT=1 时退化为 m4 的递推核**（scale 折法不同，结构一致）：

```
BT=1 ⇒ A=0、T=I ⇒ u=βv、w=β·eg·k、d = β(v − e^g·kᵀS₀)、
       o = scale·e^g·(q·S₀) + scale·(q·k)·d
m4:    S←e^g S; v←β(v−S k); S←S+k⊗v; o=S·q   ⇒   o = e^g(S q) + d·(k·q)   ✓ 同式
```

| 维度 | m4（decode m=1） | m18（prefill m≤4097） | 合并影响 |
|---|---|---|---|
| 接口 | q/k `[NK,128]`、v `[H,128]`、g/β `[H,8]`（stride-8 槽位） | q/k `[NK,T,128]`、v `[H,T,128]`、g/β `[H,Tp]` | 可统一为带 T 的形态（decode 取 T=1 段） |
| 状态 | `[H,128,128]` **行=V**，与 vllm cache `[block,NV,V,K]` 一致，**in-place GM RMW** | `[H,128,128]` **行=K**，UB 常驻、仅首尾过 GM | **布局必须统一**（差一次转置：让 cache 适配层转，或把布局做成模板参数）；形状一致 |
| 并行轴 | 1 head / AIV（H≤64），跨 head 软件流水（MTE2 预取 ∥ VEC 递推 ∥ MTE3 回写） | 1 head / AIV，chunk 串行扫描（无跨 chunk 流水） | 轴一致；decode 的预取流水在 chunk 形态下不适用 |
| 核内同步 | 7 个 BufferID（ping-pong slab + q/k/v/o） | 4 个 BufferID（chunk 工作集令牌 + S 首尾 + probe） | 机制相同（BufferID + 阻塞释放），可兼容；合并后取较大的 UB 布局（226KB < 248KB ✓） |
| UB 占用 | 138KB | 226KB | 合并取 max，仍放得下 |
| 每 token-head 指令量 | ~2.7×10³（单遍融合，状态每行只读写一次） | 2.0×10⁵/chunk ⇒ 均摊 ~3.1×10³/token；**BT=1 时**实测口径估算 ~3×10³ | chunk 形态在 BT=1 时指令量只差 ~1.1–1.3×，但状态 UB 遍历从 1 遍变 ~4 遍 |

**合并结论**：**能合成同一个 kernel，单条代码路径吃两头在指令量上可接受，但要付两笔代价。**

- 指令量论证（推翻「必须两条路径」的直觉）：m4 每 head-token ≈2.7×10³ 条向量指令
  （128 行 × 21 条/行：两载、两存、4 步融合）；本核 chunk 路径均摊 ≈3.1×10³/token，
  且 **BT=1 时 A 为空、T=I、KKT/Γ 退化为标量**，fixed cost 归零，估算 ~3×10³ —— 与 m4
  **同量级（差 1.1–1.3×）**。所以「decode 走 chunk 路径会慢百倍」是错的，不需要为性能而分裂路径。
- 代价 ①：**状态布局必须统一**。m4 用 `[V,K]`（对齐 vllm cache `[block,NV,V,K]`），本核用 `[K,V]`
  （chunk 数学最顺）；二选一，另一个付一次转置（放在 cache 适配层，每层一次，可忽略）。
- 代价 ②：BT=1 时状态 UB 遍历 ~4 遍（`d = u − wS`、`o_part = q'S` 各 1 遍、`S += kᵀd'` 1 遍
  读 + 1 遍写）vs m4 的 1 遍（每行 state 只 Load 一次、Store 一次，四步全在寄存器）。
  UB 带宽不是瓶颈，但指令与延迟成本在那里；若 decode 是主战场，可在同一 kernel 里加一个
  `BT==1` 特化直接发 m4 那种融合写法（共享布局常量/BufferID/GM 接口）。
- 代价 ③：模板化会把两份 VF 编进同一个 kernel（编译时间与指令 cache 变大）；且 decode 还需要
  conv1d/l2norm/gating prolog 与 MTP token 循环（m9/m16 的范围）在核内或段内接上。
- 因此**当前建议**：维持两个 kernel（m4 decode + m18 prefill）共用一份「布局/常量/参考公式」
  头文件；megakernel 需要单 launch 内按 `m` 切换时，把 `BT` 与状态布局做成模板参数合并
  （`BT==1` 用 m4 的融合写法，`BT==64` 走本核），而不是现在硬合。

---

## 7. AIC/cube 升级路径（本切分为何 AIC 空转）

- 本核的 chunk 工作 **100% 是 fp32 向量域**（A/KKT/Γ/前代/状态都要求 fp32 精度以对齐 fp64 判据），
  而 3510 的 cube 只能吃 bf16 操作数 + fp32 累加。**用 cube 就得把操作数降成 bf16**，判据随之
  变成「bf16 三方 cross_check」（donor 的判据就是这个），当前 fp64 紧容差路线会作废。
- 问题形状也不站在 cube 这边：48 个 value head vs 28 个 AIC —— 1 head/AIC 会剩 20 个 head
  无处放（要么 2 head/AIC 串行、要么改成 arch35 那种「chunk group + 三段式 + SyncAll 屏障」，
  把 chunk 轴也变成并行轴）。而 AIV 侧 48 个 value head 正好 1 head/AIV 用掉 48 个 AIV
  （mix(1,2) 的 56 个 AIV 线程里 **8 个空闲早退**，占用 48/56）。
- 所以本切分选择：**AIV 空满、AIC 空转**（`if ASCEND_IS_AIV` 内做全部工作），换取
  ①fp32 紧容差判据 ②零 cross-core 同步（无挂死风险）③**48/56 AIV 占用**（不是"铺满"：
  48 个 head 用满 48 个 AIV，8 个 AIV 空闲早退）。
- 若要走 donor 路线：需按 arch35 三段式重构（stage1 chunk 并行 / stage2 串行状态扫描 /
  stage3 输出），GM 工作区约 `Nv·S·(Dk+Dv+C)` bf16，跨段 `SyncAll` 屏障，并把 h 从 UB 常驻
  改成 GM 原子累加（docs/10 §3：GM 往返 ~400MB/层）。**收益是量级级别的（cube 的 MAC 吞吐
  远高于向量域），代价是 bf16 精度 + 3 段同步 + 跨核挂死风险**。建议作为独立后续 mission，
  而不是在本核上打补丁。

---

## 8. 已知限制

1. **VF 内层循环上界不能依赖外层归纳变量**（本次实测 quirk，已单独 file 到 tower）：
   `for (i) { for (j = 0; j < i; ++j) }` 在 3510 上**内层少执行一次**（静默丢 `j=i−1` 项，
   ~1% 数值误差）。本核 `TrilSolveVF` 因此把内层上界改成编译期常量 `BT`（A 严格下三角使多算的
   项恒 0），代价是行前代从 BT²/2 变 BT²。修好后 u/w 的误差与其它段同量级（见 §4.4 的
   `u[chunk0]`/`w[chunk0]` 行）；**故障态的实测数值见已 file 给 tower 的 finding
   （`3510 VF quirk：内层循环上界取外层归纳变量时少执行一次`），README 不复制那些历史数字。**
2. **单层 15.5 ms**（m=4097）：正确性优先的产物，纯向量域跑小矩阵乘；性能口径见 §性能。
3. **无跨 chunk / 跨核流水**：每个 AIV 串行走完自己 head 的 65 个 chunk；A/Kxy 的 chunk 内
   准备事实上也可并行，但本切分没把它拆到别的核上（48 head 已用掉 48 个 AIV、剩 8 个空闲，
   拆出去收益有限）。
4. **fp32 only**：与生产路径（bf16 操作数）不同，是**判据选择**而非能力限制；换 bf16 需要
   把参考改成 bf16 逐步舍入模型。
5. **conv1d / l2norm / gating 在核外**（与 m4 同约定；m9/m16 是 prolog 核）：本核吃
   post-conv 的 q/k/v 与已算好的 g/β。prefill 的 conv1d 因果 + conv_state 语义见 docs/10 §1。
6. **`egL` 取 `eg[BT−1]`**：只在 padding 行 `g=0` 时等价于 `eg[cv−1]`（本核强制清零 padding，
   故成立）。若将来允许非零 padding，要显式按 cv 取。
7. **单 head 状态 64KB 常驻 UB**，UB 总占用 225.5KB —— 没有空间再给第二份 slab 做跨 chunk
   双缓冲（预取下一 chunk 的 k/v）。
8. `probe` 抽点会开 MTE3 + 串行握手（probeHeads>0 时逐 chunk 写状态）；纯性能口径请用
   `probeHeads=0`（两次运行实测：probe 开 15.514 / 15.522 ms，probe 关 15.494 / 15.503 ms，
   差 ≤0.2%）。
9. **性能数字对机器负载敏感**：本机与 8+ worker 共用 NPU，早期一次运行观测到 65.7 ms
   （共享争用）。§4.7 的 15.5 ms 是"空载附近"的数字，不是上界。

---

## 9. 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m18_gdn_prefill/build -S m18_gdn_prefill -DCMAKE_BUILD_TYPE=Release
cmake --build m18_gdn_prefill/build -j3
cd m18_gdn_prefill/build && ./m18_gdn_prefill            # 4 档用例 + 计时（dump *.bin 到 cwd）
/usr/local/python3.12.13/bin/python3 ../check_ref.py     # T1/T3 逐段判据（退出码 0/1）
# 可选：只跑某一档
/usr/local/python3.12.13/bin/python3 ../check_ref.py --only one        # one/gqa3/all48/target
# 可选：改 T3 ε（默认 5e-5，推导见 §4.2）——ε 是唯一可调项；没有 --rel/--abs
/usr/local/python3.12.13/bin/python3 ../check_ref.py --eps 2e-5
# 可选：打印本轮 40 个 dump 的 sha256（与 evidence/dump_sha256.txt 对齐，确定性复核）
/usr/local/python3.12.13/bin/python3 ../check_ref.py --sha256
# M166 零设备（不需要 build/、不需要设备）：判据自检 + 参考侧回归 + 同型写法普查
bash ../evidence/m166_repro.sh
#   等价单项：
/usr/local/python3.12.13/bin/python3 ../check_ref.py --selftest      # 干净 PASS + 上溢/设备NaN/chunk边界 三个必红对照
/usr/local/python3.12.13/bin/python3 ../check_ref.py --ref-regress   # 4 档确定性输入的 旧式 vs 指数差 参考侧对照
/usr/local/python3.12.13/bin/python3 ../check_ref.py --legacy-exp    # 判据切回旧式参考（仅对照；上溢档会自造 NaN）
# README 数字对账（**不需要 build/、不需要 numpy**）：每个数字 grep 回出处 + 数值比对
#   注意 cwd：上一行 `cd m18_gdn_prefill/build` 之后仍在 build/，故脚本要用 `../`（或先 `cd ..`）
/usr/local/python3.12.13/bin/python3 ../audit_readme_numbers.py  # 产物 evidence/readme_number_audit.md
# 可选：负向对照 A+B（扰动一条断言 / 强制几何量归档缺失）
/usr/local/python3.12.13/bin/python3 ../audit_readme_numbers.py --negative-control
# 可选：缺输入读数（产物另写，不覆盖正式合格证）
/usr/local/python3.12.13/bin/python3 ../audit_readme_numbers.py --no-geometry
```

`audit_readme_numbers.py` 的**退出码三态**（调用方/CI 只看退出码时的契约）：

| rc | 含义 |
|---|---|
| `0` | 比过且通过：`compared > 0` 且 `failed == 0` 且本仓输入无 `UNVERIFIED` |
| `1` | 比过且**有差异**（`failed > 0`；有差异时优先报 1） |
| `2` | **没得比**：`compared == 0`，或存在本仓输入缺失导致的 `UNVERIFIED`（PART 3 有待裁定项亦为 2） |

合格证文案只在 rc=0 时打印，且**必须带实际比较计数**（`RESULT: OK (compared N/M claims, 0 failed, …)`）；
缺输入时打印 `RESULT: NOT CERTIFIED/SKIPPED` + 逐条点名缺哪个输入。三份读数（正常 / 负向对照 /
缺输入）原样归档在 `evidence/readme_number_audit_readings.log`。**未作数的"对比"不会被当成通过**。

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists。
dump 文件名 `m18_h<H>_t<T>_<tensor>.bin` + `m18_h<H>_t<T>_meta.txt`（pwd = build/）。

## 10. 与 docs/15（M33 prefill 设计 survey）的对齐

M34 曾按 mission 任务 6 把公式、chunk 取值依据、两个易错点、AIC 取舍发给 `agent-prefill`（M33）。
**`docs/15` 已入 main，本轮按 tower 指令与它逐条对账**，结果如下（`docs/15 §3.1`）：

| docs/15 §3.1 的写法 | 本核 | 判定 |
|---|---|---|
| `v_prime = k_cumdecay @ h; v_new = u − v_prime`（h **未衰减**，`k_cumdecay=T@(k_β·e^g)` 里带 `e^ĝ`） | `d = u − w·S₀`，`w = T(β⊙eg⊙k)` | **一致** ✓（含"先减后衰减"的顺序，与官方 golden `:139-140` 同） |
| `h = h·e^{g_last} + (k·e^{g_last−g})ᵀ @ v_new` | `d'=d⊙ig`；`S₁ = egL(S₀+kᵀd')` | **等价**（同一 `Σ_j e^{ĝ_L−ĝ_j} k_j⊗d_j` 的两种结合顺序） |
| `BT=64`、`(I+L)⁻¹` 用 VF 前代、h 保 fp32 | `BT=64`、行前代（不物化 T）、h 保 fp32 | **一致** ✓ |
| **`A = −(k_beta @ k^T) ⊙ decay_mask` 然后 `T = (I + A)^{-1}`** | `A = +β⊙Γs⊙(k·kᵀ)`（严格下三角），`T=(I+A)^{-1}` | **⚠ 有符号冲突**，见下 |
| **`decay_mask`（`:116`，含对角）同时用于 A 与输出项** | A 用**严格**下三角（i>j）；输出项用含对角（i≥j） | **⚠ 掩码范围不同**，见下 |
| **`o_inter = (q·e^g) @ h` + `o = o_inter + tril(QKᵀ⊙decay_mask) @ v_new`** | 输出项用**未乘 e^ĝ 的 q**（只带 scale），e^{ĝ_i} 只加在状态项上 | **⚠ 措辞会诱导重复乘 e^ĝ**，见下 |

**三处符号/措辞冲突的裁定（本核以官方 golden 为准，已逐句验证）**：权威实现是
`ops-transformer/attention/chunk_gated_delta_rule/tests/pytest/chunk_gated_delta_rule_golden.py:99-149`。
按 golden 逐句核对：`:110-114` 先定义 `mask=triu(ones, diagonal=0)`；`:117`
`attn = −((k_β@kᵀ)·decay_mask).masked_fill(mask, 0)`（**连对角一起清零 ⇒ 严格下三角**）；
`:118-121` fla 逐行递推；`:122` `attn += I`。该结果**恰等于 `(I+L)^{-1}`，其中
`L[i,j]=+β_i(k_i·k_j)e^{ĝ_i−ĝ_j}`（严格下三角）** —— 已用随机数据在 BT=2/3/5/8 上数值验证：
`max|golden_T − (I+L)^{-1}| ≤ 4.0e-15`，而 `max|golden_T − (I−L)^{-1}|` 是 O(1)（1.5~146）。
（**判据形态（M56 订正）**：4.0e-15 是**随机数据下的经验上界**，不是逐位恒等 —— 换个抽取就换个末位。
故 `audit_readme_numbers.py` 用**固定 seed 的确定性数据集**复算，按 **`计算值 ≤ 4.0e-15`（上界）
+ `计算值 ≥ 1`（判别性）** 判，不按 2% 恒等判；纯标准库实现，无 numpy 依赖。
**跨环境可复现的是判据、不是数值末位**：`random.gauss` 随 Python 版本不同（实测 py3.12.13 →
1.110e-15、py3.11.6 → 3.553e-15），两次都满足判据 —— 这也正是判据取「界/量级」而非 2% 恒等的原因。）

1. **符号**：docs/15 的 `A=−(k_βkᵀ)⊙decay` 与 `T=(I+A)^{-1}` **不能直接连用** —— 那给出的是
   `(I−L)^{-1}=I+L+L²+…`，而 golden（以及本核）要的是 `(I+L)^{-1}=I−L+L²−…`。要写 `−` 就必须
   保留 golden 的**逐行递推 + `+I`** 那两步（docs/15 的伪码把这两步省了）；或者直接改成
   `L=+(k_βkᵀ)⊙decay`（严格下三角）+ `(I+L)^{-1}`（**本核写法**）。**下游照 docs/15 字面实现会错**。
2. **掩码范围**：golden 里同一个 `decay_mask`（`:116`，含对角）在两处配**不同**的 mask ——
   A 用 `triu(diagonal=0)`（`:110-114`，连对角一起清零 → A 严格下三角），输出项用
   `triu(diagonal=1)`（`:131-134`，只清严格上三角 → 含对角）。docs/15 只写了 A 处
   「masked_fill(上三角)」，读者会漏掉对角；**A 的对角非零会直接破坏 WY**（本核按严格下三角实现，
   并由此通过 G3 guard：`max|A 上三角|=0.0`）。
3. **e^ĝ 只能乘一次**：golden 的输出项用 `q_i`（**只乘过 scale**），e^{ĝ_i} 只在 `attn_inter` 里出现；
   decay_mask 供衰减。docs/15 写成 `o_inter=(q·e^g)@h` 与 `tril(QKᵀ⊙decay_mask)` 并列，若把同一个
   `q·e^g` 复用到第二项就是 `e^{2ĝ_i−ĝ_j}` —— 本核实现时正是踩了这个坑，靠逐段抽点定位
   （故障态数值不在此复制）。

**本核的立场**：以上三处已按官方 golden 实现并验证（4 档 PASS；reviewer round-1 另用官方 golden
自建 oracle 复算设备输出、结论一致）。**这不是"以实测压文档"**：冲突点在 docs/15 的伪码符号/措辞
层面，`docs/**` 归 tower 维护，已把上表报给 tower 由其裁决是否修订 `docs/15 §3.1`；本 README 不改
docs，只记录对账结论，供下游写代码时以 golden 为准。

> ⚠ 与 `docs/17` §7 的 QSA 免责声明无冲突：§7 说的是 **attention 段的稀疏选带**导致
> "m=4097 不可能对齐官方稠密输出"；本核是 **GDN 段**，不涉及 QSA 选带，且本核的验收基准是
> **自建 numpy float64 参考（与官方 golden 公式逐句一致）**，不是"与官方 aclnn/triton 输出逐位对齐"。

---

## 11. 选型依据的指针（M104）

本核的**性能口径**（15.51 ms/层 怎么来的、它只覆盖 chunk core 的哪一段、与 `docs/15` 的
1.37 ms/层下限怎么对齐、以及"VF 路线 vs AIC mmad 路线"该怎么选）由另外一份文件给出：

> **`docs/19-gdn-prefill-scan-selection.md`**（M104 的交付物，rev 基准 `20bd20d`）

**为什么指过去而不是写在这里**：本文（m18 的 README）是 M34 的验收文档，它证明的是
**"这个核在 4 档上数值正确"**；而 `docs/19` 做的是**跨段的选型对照**（把本核的归档计时拆成
固定开销 / 必要 MAC / 算法指令膨胀 / 发射率四项，再与 `docs/15` 的 cube roofline 对齐）。
两者是**不同的问题、不同的 scope**，不合并。本文的结论（算法已证、4 档 PASS、AIC 空转、
UB 225.5 KB、每 chunk 238 µs、行前代 128 次 `mem_bar`）**一条都没有被 `docs/19` 改动或推翻**；
`docs/19` 对本文数字的用法是**引用**（逐条给命令与范围）。

**本文与 `docs/19` 一致的两条定位**（供下游引用时不要读错）：
1. 本核是**判据/参考 + VF 子原语 donor**（`CumSumExpVF` / `GammaStrictVF` / `TrilSolveVF`
   可原样复制），**不是"已完成的 GDN prefill 段"** —— 它没有接进融合 kernel
   （`grep -rn "m18" m15_layer_loop/ | wc -l` → 0，范围见 `docs/19` §1.2）。
2. 本核的 `15.51 ms/层` 是**向量的、chunk core 段单段的**实测值；**不能**与 `docs/15` §6.3 的
   整层 `1.37 ms/层` 直接比 —— 口径对齐见 `docs/19` §3.1。

**后续（r2 起）的两条指针，供下游按最新口径取用**：
- **选型已定**：人类裁决「凡是涉及矩阵乘法的操作都需要用 mmad 实现」⇒ 生产形态是 mmad；
  本核的 VF 实现是**对照/参考**，不是交付形态 —— 逐条映射见 `docs/19` §10。
- **后续头号设计风险是 mmad 的精度**（不是性能）：`docs/19` §11 给出三处数值敏感点、
  bf16 的量化后果、以及「cube 能否吃 fp32 操作数」两种分支下各自的做法（待 M106 裁决）。
