# M10：attention decode FA core（mix(1,2) 单 kernel，GQA m=1 dense FA，head_dim=256）

**状态：判定口径已定（边界感知 ulp，§6.6）；M61 把"判据的独立性"收口：Exp 容差改**官方 Reg 精度规格**（不再是设备探针实测）、
新增**由 host q/k 独立重算 P̃**的判据 C/D 与**独立 P 的 P·V**判据 E、并**正式否掉**"主判据改成以设备 P̃ 为输入"的提议（§7.1）。**
判定项（参与 PASS/FAIL）——**256 ✅ / 300 ✅**（100% ≤1 ulp）；**4096 ❌ 1/6144**（`out[0][11][217]` 绝对界超 13.6%，
成因 = 单个 P̃ 的 bf16 舍入翻转，因果闭链见 §6.2 末段 + `check_partials.py`）。
判据总账（`logs/check_ref_20260926_exp_spec.log`，一条命令可复跑）：**判定项 23/24 通过**（唯一 FAIL = 4096 的判据 A）、
**报告项 27 项单列**、**guard 9/9**、覆盖 6 条判据 × 3 档、**无抽样**。
**但 T4"确定性"未达成**：一次多档连跑里只有 `s300_out.bin` 数值不同、0/41 次复现且证据已不可核验
⇒ 按纪律记为**未隔离观测**（§8.1，含我三处不准确表述的更正与 probe mission 问题描述）⇒ **本 mission 不宣称验收完成**。
本文档如实记录已打通的部分、已实证的硬件/API 结论、以及当前卡点与复现方式（原始 dump 与日志见 `data/`、`logs/`）。
承接自已弃用的 M17（`feat/attention-decode-fa-core-kernel`，未提交工作留在 `wt-17/m10_attn_decode/`）。

## 0. 当前验证进度（真机 Ascend950PR / CANN 9.1.0 / bisect 15.0.5）

| 阶段 | 内容 | 状态（口径 docs/17 §L0） |
|---|---|---|
| 工程 | 独立 cmake ASC 工程编译 | ✅ 六个 target（`m10_attn_decode`/`m10_probe`/`m10_l0probe`/`m10_mmadprobe`/`m10_expprobe`/`m10_dup_probe`） |
| 同步骨架 | FD 收尾三类 CrossCore | ✅ `m10_probe` 14 条探针实证（见 §4） |
| BMM1 | Q·K^T（M=16,N=256,K=256 拆 2×128） | ✅ L0C 与 numpy 逐元素一致（maxErr 1.9e-6） |
| FIXP→UB | L0C→UB dualDstCtl=1 双 AIV、fp32 | ✅ |
| softmax / P | RegBase VF；行 max/行 sum/E/P(bf16) | ✅ **T3 逐元素（判据 C，M61 新增）**：`gp` **全数组**（4096 = 131072 个元素、**无抽样**）对 host 从 q/k **独立重算**的 P̃ ⇒ 越界 **0**，位差 0/1/2/≥3 格点 = 131071/1/0/0（唯一 1 处 = 1 个 bf16 格点，界占用 0.9965）；判据 D（未用槽位恒 0 + 尾 tile 无效列恒 0）PASS。**注 1**：旧 `check_gp` 只覆盖 unit0/parity0 的 0–11 行、容差 5e-3（无来源经验值）⇒ 现由判据 C 全数组覆盖、容差换成派生的 `1.0·ulp+ε_arg`（**更严**，最小界 9.8e-4），旧读数保留在同一行输出里。**注 2（ulp 约定，M61 r1 评审 P2-2；r2 改为自足表述）**：判据里的"1 ulp" = **相邻 bf16 格点间距** = `2^(e-7)`（bf16 = 1 符号位 + 8 指数位 + **7 位显式尾数** ⇒ 一个存储步长）。三条依据：① 两个**都已量化到同一 bf16 网格**的值，最大合法差 = **一个格点间距**；② **测量事实锚点**：M36 的 `real_m1` 用例里设备 `0x3dc0` 与参考 `0x3dc1` 是**相邻格点**；③ 取**半格**会把本处唯一那 1 处**合法**翻转（`|Δ| = 1.9531e-3` vs `tol = 1.9601e-3`、界占用 0.9965）误判成 FAIL 1/131072。详见 `check_ref.py::bf16_grid_ulp` 的 docstring |
| BMM2 | V^T 装载（M27 配方 2D V2 + `ifTranspose=true`） | ✅ **已修**：判据 E（**独立重算的 P̃**·V，全 unit 的 tile0；4096 覆盖 16 unit × 16 行 × 256 列 = 65536 元素）越界 **0/65536**、maxAbsErr 1.6e-3（= 已知那 1 个 P̃ 翻转的传播量）、界占用 5.4e-3；判据 F（**设备 P**·V，M59 的 S1）越界 0/65536、maxAbsErr 3.0e-6、界占用 3.3e-3；4096/4096 非零 |
| FD combine | partial 写 GM + AIV 归并（VF） | ✅ |
| **数值验收** | 三档 seq=256/300/4096，**判定项 A–F 共 8 项/档**（A = 边界感知 ulp，B = out 结构性 ×3，C/D = P̃ 全数组 + P̃ 结构性，E/F = P·V 独立 P / 设备 P；逐条咬合力见 §9） | **256 ✅ PASS**（常规 6130/边界 14 元素，0 越界；位级 98.18%）**/ 300 ✅ PASS**（常规 6086/边界 58，0 越界；位级 98.62%）**/ 4096 ❌ FAIL 1/6144**（常规 5924/边界 220：ulp 项 0 越界，**1 个边界元素的绝对界越界 13.6%**；位级 99.37%、≤1ulp 99.9837%）。该 1 个元素已被**完整归因到一个 P̃ 的 bf16 舍入翻转**（§6.2 末段 + `check_partials.py`），属判定界之外的机制 ⇒ **按 tower 要求不自行放宽、不申请例外** |

**分档声明（docs/17 §1.1 要求"用哪一档必须在 README 写明理由"）**：

| 判据项 | 档 | 理由与状态 |
|---|---|---|
| Q/K/V 输入字节 + 列 mask 契约 | **T1** | 整数/位域，逐字节：sha256 清单 + reviewer 独立生成器复现一致（含可再生成的 4096 k/v）；列 mask 契约另由判据 D 咬（尾 tile 无效列必须恒 0，实测三档全 0） |
| `gp`（P̃ 中途量化为 bf16；判据 C/D） | **T3**（三个触发条件全中：含 `Exp` 近似 + 跨 tile 在线重标定 + mmad k=256）——旧写 T2′，本轮按 docs/17 §1.1 的**触发条件改正** | 参考**无法**复刻设备 Exp 的位级结果（官方规格 1 ulp）⇒ 不能按 T1 逐位。判据 = 逐元素 `\|Δ\| ≤ 1.0·ulp(P̃_ref) + ε_arg·\|P̃_ref\|`。**为什么是 1.0·ulp 而不是 0.5**：参考与设备两侧都落在 bf16 格点上 ⇒ 按 docs/17 §1.1 的 `0.5·ulp`/`1.0·ulp` 条款（参考也被量化到 bf16 格点时取后者），格点对格点的最大合法差 = 1.0·ulp。**这里的 1 ulp 有明确数值约定（M61 r1 评审 P2-2；r2 改为自足表述）**：= 相邻 bf16 格点间距 `2^(e-7)`（bf16 = 1 符号位 + 8 指数位 + **7 位显式尾数** ⇒ 一个存储步长）；依据：① 两个都已量化到同一 bf16 网格的值，最大合法差就是**一个格点间距**；② **测量事实锚点**：M36 的 `real_m1` 用例里设备 `0x3dc0` 与参考 `0x3dc1` 是**相邻格点**；③ 取**半格**会把本处唯一那 1 处**合法**翻转（`\|Δ\|=1.9531e-3` vs `tol=1.9601e-3`）误判成 FAIL。`ε_arg=(S2T+EXP_ULP)·2^-24`（mmad k=256 的 S 累加界 + Exp 官方 1 ulp；`Muls(1/16)` 是 2 的幂乘=精确、官方规格表 Sub 为 0 ulp ⇒ 不另计）。实测 4096 全数组 **131071/131072 逐位**，唯一差异 = 1 个 bf16 ulp（`gp[unit10][slot0][row11][col252]`：设备 `0x3ee8` vs numpy `0x3ee7`） |
| `out`（判据 A） | **T3**（触发条件全中：含 `Exp` 近似 + 跨 tile/split 在线重标定 + mmad k>16） | 判据 = T3 的逐元素 `\|out−ref\| ≤ ε·Σ\|terms\| + 0.5·ulp(out)`，**并自加更严的一半**：常规元素（参考距网格点 ≥ δ）要求 **≤1 ulp**，只有边界元素用 **≤2 ulp + 绝对界**。ε 各项来源与 δ 推导见 §6.6 |
| P·V（判据 E，**独立 P**） | **T3** | 逐元素 `\|Δ\| ≤ S2T·u·Σ_k\|P̃_ref V\| + Σ_k(1.0·ulp(P̃_ref_k)+ε_arg·P̃_ref_k)\|V_kd\|`，两项都是**推导**。**分辨率声明**：第二项 ≈0.39%·Σ\|P̃V\| ⇒ 咬得住 P 的**约定级/布局级**错误与"BMM2 实际消费的 P̃ 是否就是 `gp` 那份"，咬不住比 0.39% 更细的差异（更细的元素级由判据 C 承担） |
| P·V（判据 F，**设备 P**；M59 的 **S1**） | **T3** | 逐元素 `\|Δ\| ≤ S2T·u·Σ_k\|P̃_dev V\|`（mmad k=256 累加的保守界，**推导**；旧写的 5e-3「bf16 网格」是无来源经验值，已换）。**被判量 = "给定 P 下 BMM2 的布局/转置"，P 不是本判据的被判量**（两侧同一份设备 P̃）；P 由判据 C/D 独立咬 |
| 结构性（判据 B/D + `ws*` 部分和） | **T4（所有档都要）** | 非空洞性（out 非零/非常数/每行 256 列非零）、索引与槽位可区分（判据 D 的"未用槽位恒 0"咬 unit 索引空间、`ws*` 部分和判据）、dump 非零且位置正确 —— 均 PASS；**"确定性"一项按 reviewer r3 判为**未达成**，见 §8（未隔离的观测）⇒ 本 mission 不宣称验收完成** |

**判据报法（docs/17 §1.1）**：`T? 档；ε 各项来源与数值；判定项越界数/总数；报告项（≤1ulp 比例、maxRel、位级一致率）`。
本核：ε 的四个来源见 §6.6；**判定项 A 的越界数** 256 0/6144、300 0/6144、4096 **1/6144**；
**判定项 C/D/E/F** 越界数 0（判据 C 覆盖 8192 / 9600 / 131072 个 P̃ 元素，判据 E/F 各覆盖 8192 / 16384 / 65536 个 P·V 元素）；
报告项 位级 98.1771% / 98.6165% / 99.3652%、≤1ulp 100% / 100% / 99.9837%、maxRel 7.6e-3 / 7.6e-3 / 1.0e-2（相消区元素主导）。

> **ε 中 `Exp` 项的出处（docs/17 护栏 2 要求"官方精度规格"）—— M61 已换成官方规格，不再是设备实测**：
> 官方 Reg 矢量计算精度标准（`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`
> 表 1「基础算术 / Exp」行 = `1ulp, not support denormalized numbers`，硬件与软仿**同为 1 ulp**；
> 同文档 `.../reg_vector_compute/basic_arithmetic/Exp.md` 的「最大精度误差」一节亦为 1 ulp、**约束说明：无**。
> 本核三处 Exp（`SoftmaxTileVf` 的 E 与 expdiff、`CombineWeightsVf` 的 combine 权重）都是 Reg `Exp`
> （`__VEC_SCOPE__` + RegTensor、fp32、默认 `ExpAlgo::INTRINSIC`）⇒ 该规格适用；**域条件（VL/输入域）分析见 §6.6**。
> 「1 ulp」→ 相对界的换算（**最坏情形，不是观测**）：`y = exp(x) ∈ (0,1]` 落在 binade `[2^e, 2^(e+1))` 时
> `ulp(y) = 2^(e-23)`、而 `y ≥ 2^e` ⇒ `|Δy|/y ≤ 2^-23 = 2·2^-24` ⇒ 本式（单位 2^-24）里取 **`EXP_ULP = 2`**。
> **数值上恰好等于旧值 2，出处则完全不同**：旧值是"设备探针实测 1.1~1.6 ulp 向上取 2"（docs/17 §1.1 已明禁
> "实测最大 X ulp 不是界"）—— 那是**观测**；新值是官方规格 + 最坏情形换算 —— 这是**界**。
> 另有两条与探针无关的**交叉见证**（都不再是 ε 的来源）：`m10_expprobe` 在三个相关区间给出 5 次逐字节一致的
> 1.1~1.4 fp32 ulp（§6.4）、`gp` 全数组只有 1 次 bf16 边界翻转（§6.4 的 Poisson 反推 ≈1.2 ulp）—— 两者**都 ≤ 官方 1 ulp 界**，
> 即"实测没有超出规格"。**敏感性**：Exp 项占 ε 的 ~1.5%（4096 档 272u 里的 4u），即便放大 10 倍也只改 ε ~13%、三档判定结论不变。

**判定项 / 报告项 / guard 三栏**（docs/17 §2.1；`check_ref.py` 用**同源计数**把三栏打进总账行）：

- **判定项（参与 PASS/FAIL，8 项/档 = 24 项）**：
  A. `out` 的**边界感知 ulp** 判据（与量化规格匹配的 fp64 numpy 参考比，1 项）；
  B. `out` 非全零 / 非常数 / 每 (n2,g) 行 256 列全非零（T4 结构性，3 项）；
  C. **P̃ 全数组逐元素**（T3，M61 新增）：`gp` 的每个 (unit,slot,row,col) 对 host 从 q/k 独立重算的 P̃（1 项）；
  D. **P̃ 位置/掩码结构性**（T4/T1，M61 新增）：未用槽位恒 0 + 尾 tile 无效列恒 0（1 项）；
  E. **P·V（独立 P）**（T3，M61 新增）：BMM2 的 L0C 对 `einsum(P̃_ref, V)`，覆盖全 unit 的 tile0（1 项）；
  F. **P·V（设备 P）**（S1 定位声明）：给定 P 下 BMM2 的布局/转置 + "AIC 消费的 P 与 `gp` 同一份"（1 项）。
- **报告项（27 项，不参与判定）**：位级一致率、≤1ulp/≤2ulp 比例、maxAbsErr、**maxRel**（相消区元素主导）、
  单一相对累加界的覆盖率、判据 C 的位差分布/翻转位置/覆盖元素数、判据 E/F 的覆盖率与界占用、旧 `check_gp` 窄范围读数等。
- **guard（9 项，不计入判定项数；失败即视为判据工具自身失效 ⇒ 退出码不为 0）**：host mask 契约自检 ×3、
  判据 C 的**判别力对照**（分辨率 = 1 个 bf16 格点：Δ=0 ⇒ 0 越界、+1 格点 ⇒ 0（在 1.0·ulp 允许差内）、+2 格点 ⇒ 必须被抓）×3、
  判据 C 的**口径负向对照**（把参考换成 `globalmax` 口径 ⇒ 必须报越界；**如实报限度**：seq=256 全序列只有 1 个 tile ⇒
  两口径同解、不可判）×3。
- **退出码三态（docs/17 §8.3）**：`0` 全通过且无输入缺失 / `1` 比过且有 FAIL / `2` 有输入缺失（`RESULT: SKIPPED`，**绝不发合格证**）。
  实测（`logs/check_ref_tristate_20260926.log`，**五段**各带真实 rc）：三档 `pv`（E/F 全过）⇒ **rc 0**；
  三档全量 ⇒ **rc 1**（唯一 FAIL = 4096 判据 A）；空目录 ⇒ **rc 2**（12 项"没得比"）；
  只给 seq=256 ⇒ **rc 2**（256 的 8 项照跑并打印，300 标为没得比）；
  **只有归档 dump（无 4096 k/v）⇒ rc 2**（256/300 **16/16 判定项通过**、报告项 18、guard 6/6；
  4096 的四项标为"没得比"——缺 k,v —— 并不当成通过）。

**本核判定规则（判据 A：边界感知 ulp；完整推导见 §6.6）**：
- **常规元素**（参考距最近 bf16 网格点的间距 ≥ δ·ulp(out)）：判 **≤1 ulp**（严格，不放宽）；
- **边界元素**（参考落在相邻两网格点**中点** ±`noise_ub` 内，`noise_ub = ε·Σ|acc·w|/den`）：判 **≤2 ulp**
  **且**必须同时满足绝对界 `|out−ref| ≤ noise_ub + 0.5·ulp(out)`；
- `ε = (S2T + nsplit + 2·nTile + EXP_ULP·nTile)·2^-24`，`EXP_ULP = 2`（= 官方 Reg 规格的 **1 ulp** 折成 2^-24 单位的
  **最坏情形**值）；四项 = mmad k=256 累加 + 跨 split 归并 + 在线重标定/累加 + Exp 误差（E 一次 + 重标定因子 nTile−1 次）；
  **改前/改后对照见 §6.6**（256/300 数值不变，4096 由 270u → 272u）。
- `δ = noise_ub/ulp(out)` 随相消程度增长（实测 256 的 δ ≤ 0.0038、300 达 27.28、4096 达 9.28）。
- **maxRel 是报告项**：它的分母是元素自身，在相消区（|out| 比同行量级小 1~2 个数量级）必然被放大
  （256：7.6e-3、300：7.6e-3、4096：1.0e-2，均由 |out|≈1e-3~5e-5 的相消元素主导）；判定项不用小分母。

**为什么数学上不可能逐位**（docs/17 §L0 要求写明）：① P 在流水线中途被舍入到 bf16（donor 同款设计，
本核规格的一部分 → 参考实现同样做 bf16 RNE 舍入）；② FD split-K 的 **fp32 累加顺序**与 numpy 不同
（每 tile 的 PV 在 cube 内按 k=256 累加、AIV 侧再跨 tile 在线重标定、combine 跨 split 累加）。
⇒ 末位 ulp 抖动不可避免；实测 256/300 的 ≤1ulp 比例均为 100%。
另有第三类离散事件：设备 exp（**官方规格 ≤1 ulp**，§6.6；实测 1.1~1.4 fp32 ulp，§6.4）+ fp32 S 可能把某个 P̃ 推过 bf16 量化边界
（概率 ~1e-4/元素；4096 三档实测 1/131072 ≈ 7.6e-6），它对输出的影响是 2^-8·|P̃·V|/den —— 4096 那 1 个越界元素正是这一类（§6.2 末段）。

> **§6 是历史记录**：其中的"PV 恒 0 / out 全 0 / 0.26/0.96/1.6e4"等表述均已被后续轮次修正，
> 保留仅为排查轨迹。**当前状态以上表与 §5 为准。**
> 另外：早期记的"装载系参数静默不生效"已被 M27 几何标定推翻——所有"看起来不生效"的现象归位为
> (a) 参数语义/单位与文档不符、(b) 组合越界或 NOP 从而**读了没写过的槽位**；**没有任何字段被证实配了也无效**。
> 本模块的 `dstElemOff` 修复属于"2D 装载没有目的偏移参数，必须用切片表达"（M27 §5.4 同结论），不是"参数不生效"。

> 第四轮（tower 决策后）找到并修掉了 BMM2 的真根因：**`LoadL0_2D` 没有目的偏移**——P 一直被装到 L0A 的
> 头部（0..8KB），而 mmad 从 8KB 起的槽取操作数，于是 A 操作数恒为未初始化内存、PV 恒 0。
> 补上 `dstElemOff` 后 **PV 立刻出数**（此前"换 B 侧怎么都出 0"的所有现象都由这一个 bug 解释）。
> 另外按 tower 决策 3 把 P 的落盘改成 **UB→GM→AIC→GM→L1（Nd2Nz）**，已实测 GM 里的 P 数值正确。
> 剩余问题只在 **V 的转置装载**（`dbgc2` 已有数但数值不对）：dim 半块偏移未生效且数值与任何
> dim 切分的参考都不符 —— 即 README §7 里保留的"真转置装载"待办。

已打通的链路本身是可复用的：`BMM1 → FIXP → AIV 读 S → 行 max` 全部对齐参考，
说明 L1/L0/mad/FIXP/CV 交接的**布局与同步范式已经跑通**；剩余问题集中在
softmax 的 exp/行求和与 P 回流/BMM2/combine 三处（§6）。

## 1. 语义与形状

dense GQA decode（m=1）FA core：`N1=24` q 头 / `N2=2` KV 头（`g=12`）、`head_dim=256`、bf16、
`scale=256^-0.5=1/16`，无 causal mask（decode 全量 attend seq 个 cache token）。

- **gS1-merge**（donor `flash_attn_kernel_dn.h:296-297`）：同一 KV 头的 12 个 q 头排进 cube M 维
  （12 pad 16），一个 KV tile 只读一次 → KV 带宽 ÷12。
- Q `[N2][16][256]`（行 0..11 为 12 个 q 头，12..15 为 pad 置零且不写出）；
  K/V `[N2][seqPad][256]`（`seqPad = ceil(seq/256)*256`，`seq ≤ s < seqPad` 由 host 置零）；
  out `[N2][12][256]` bf16。

### tile 形状（每 AIC 每 KV tile，docs/11 §3.3 config5 系）

| 项 | 值 |
|---|---|
| BMM1 | M=16（gS1 pad）、N=256（sInner256）、K=256 拆 2×128 kLoop |
| S / PV 中间 | L0C fp32 `[16,256]`（16KB/槽，4 槽环） |
| FIXP | L0C→UB `dualDstCtl=1`，M 拆两半各写一个 AIV（每 AIV 8 行 × 256 fp32 = 8KB） |
| softmax | 全 fp32（`Muls` 缩放、折半归约行 max、`Exp`、行 sum、online max/sum/acc） |
| P | bf16 `[16,256]` 写回 L1 3-buffer（buffer 0/1/2），AIV→AIC 经 CrossCore mode 4 |
| BMM2 | M=16、N=256 拆 2×128（N-split）、K=256；V 经 `LoadData3D enTranspose` 转置装载 |
| acc | AIV fp32 `[8,256]`（每 AIV 8 行）+ 行 m/sum |
| FD split-K | seq 按 256-token tile 切；`chunkTiles=ceil(nTiles/14)`，`nsplit=ceil(nTiles/chunkTiles)`；unit = (n2, split) 静态分派 `n2=bid&1, split=bid>>1`；≤28 unit 用满 28 AIC |

### FD 归并

每 split 的 AIV 把未归一 partial（fp32 acc `[8,256]` + 行 m + 行 sum）写 GM workspace；
全体 AIC mode 0 屏障后 AIC 放行，4 个 AIV（2 头 × 2 半）按下式归并后 bf16 写回：

```
M = max_i m_i ;  out = Σ_i acc_i·e^{m_i−M} / Σ_i sum_i·e^{m_i−M}
```

### 资源（静态自管理，编译期常量）

| 资源 | 用量 |
|---|---|
| L1（AIC） | P 3×8KB + Q 8KB + KV 2×128KB = 288KB / 512KB |
| L0A / L0B / L0C | 8KB（Q 常驻 + P 轮转）/ 64KB（K 或 V^T 半块）/ 4×16KB 环 |
| AIV UB | ≈50KB / 248KB（mmRes 2×8KB、P cast 4KB、acc 8KB、mask 8KB、广播/state ≈16KB、combine ≈6KB） |

## 2. 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m10_attn_decode/build -S m10_attn_decode -DCMAKE_BUILD_TYPE=Release
cmake --build m10_attn_decode/build -j4
cd m10_attn_decode/build && ./m10_attn_decode        # 3 档 seq：256 / 300 / 4096，dump 各 case 的 .bin 到 cwd
./m10_probe <1..14>                                  # CrossCore 同步原语探针（§4）
/usr/local/python3.12.13/bin/python3 ../check_ref.py # numpy fp64 FA 参考校验（判据 A–F；退出码三态 0/1/2）
/usr/local/python3.12.13/bin/python3 ../check_ref.py pv 4096   # 只跑 E/F 两条 P·V 判据
/usr/local/python3.12.13/bin/python3 ../check_ref.py eps       # 只做 ε 的 nTile 记账自查（**不需要 dump**）
/usr/local/python3.12.13/bin/python3 ../check_partials.py      # seq=4096 那 1 个越界元素的因果闭链
```

`m10_attn_decode` 的 dump（`m10_case_s<seq>_*`）：`q/k/v/out`（kernel 面）、
`wsm/wss/wsacc`（FD workspace：行 max / 行 sum / 未归一分子）、`dbgc`（tile0 的 L0C = 未缩放 QK^T）、
`dbgs/dbgp`（tile0 各 AIV 的 S / P）。调试开关（环境变量）：
`M10_SYNTH=1|2|3|4|5`（K=单位阵；1=Q 随机、2=Q one-hot、3=仅 Q[0][0]、4=K 全 1、**5=另加 V=单位阵**，
用于逐元素定位 L1/L0 与 BMM2 通路）、
`M10_L0CFG="qM,qK,qS,qD,qT,kM,kK,kS,kD,kT[,vM,vK,vS,vD,vT]"`（覆盖 L1→L0 装载参数，定位用）。

## 3. donor 映射

| 本核 | donor | 说明 |
|---|---|---|
| 整体数据流 / 1:2 CV 配比 | `ops-transformer/attention/flash_attn` arch35（ND、config5 系） | D=256 走 ND；gS1-merge；AIC BMM1+BMM2、AIV softmax |
| FIXP L0C→UB + 双 AIV | `flash_attn_block_cube_nd.h:314,406`（`FIXPIPE_ROW_MAJOR_UB` + `dualDstCtl=1`） | M 拆两半直写两个 AIV 的 UB |
| P 回流 L1 3-buffer | 同上（`CROSS_CORE_SYNC_FORWARD`） | AIV 写 P → L1 → AIC 当 BMM2 A 操作数 |
| FD split-K + combine | `flash_attn_metadata_aicpu.cpp:284-354` + `flash_attn_block_vec_flashdecode.h:413-513` | partial 落 GM + AIV 归并 |
| mode 4 flag 体系 | `attention/common/op_kernel/attn_buffer.h:36`（`AIV0_AIV1_OFFSET=16`） | AIC 侧 flagId / flagId+16 分别对应 AIV0/AIV1 |
| BMM1 的 L1→L0B 字段约定 | `m11_bf16_gemm`（本仓库已合并、已实证）+ `bsa_copy_l1_to_l0b_a5.hpp` | mStep=N/16、kStep=K/16、srcStride=dstStride=N/16、ifTranspose=false |
| buffer/同步范式 | `docs/05 §2/§6`（BufferID + CrossCore + 裸指针） | 无 TPipe/TBuf/TQue/高阶 Matmul API |

## 4. 同步表（**全部经本模块 `m10_probe` 真机实证**）

### 4.1 实测结论：CrossCore set/wait 的 pipe 必须匹配核型（本轮最重要的可复用发现）

CANN 9.1.0 `NotifyEventImpl<mode,pipe>` / `WaitEventImpl<mode,pipe>` 的实际发射条件是
（`impl/basic_api/kernel_event.h`）：

```
IsSplitVectorPipe = {PIPE_S, PIPE_V, PIPE_MTE2, PIPE_MTE3}     ← 只有 AIV 会发射
IsSplitCubePipe   = {PIPE_S, PIPE_MTE1, PIPE_MTE2, PIPE_FIX, PIPE_M}  ← 只有 AIC 会发射
```

⇒ **AIC 上任何 `CrossCoreSetFlag/WaitFlag(..., PIPE_MTE3)` 或 `PIPE_V` 都是静默空操作；
AIV 上用 `PIPE_FIX/PIPE_MTE1/PIPE_M` 同理。** 空操作的 set 会让对侧 wait 永久挂死
（M17 原实现的两处 AIC `PIPE_MTE3` set 就是全 kernel 挂死的根因）。探针实测矩阵：

| 探针 | 内容 | 结果 |
|---|---|---|
| 1/2 | AIV `set<0x2,MTE3>` → AIC `wait<0x2,PIPE_S/PIPE_FIX>` | 完成（不挂死） |
| 3 | AIC `set<0x2,PIPE_FIX>` → AIV `wait<0x2,MTE2>`（m0 检查 4 同款） | 完成 |
| 4 | AIC `set<0x2,PIPE_MTE1>` → AIV `wait<0x2,MTE2>` | 完成 |
| **5** | AIC `set<0x2,PIPE_MTE3>` → AIV `wait<0x2,MTE2>` | **挂死（空操作）** |
| 6/7 | mode 0 全体 AIC 屏障：set `PIPE_FIX` / `PIPE_MTE2` | 完成 |
| **8** | mode 0 全体 AIC 屏障：set `PIPE_MTE3` | **挂死（空操作）** |
| 9 | mode 0 全体 AIV 屏障：set `PIPE_MTE3`（m0 同款） | 完成 |
| 10/11 | mode 4 AIV→AIC / AIC→AIV（AIC 侧 flagId 与 +16 = 两个 AIV） | 完成 |
| 12/14 | FD 收尾骨架（AIV→AIC → AIC 屏障 → AIC→AIV → AIV 收尾），低 id 0-3 与高 id 8-11 两组 | 均完成（flagId 8-11 无保留冲突） |

> 探针 PASS/FAIL 判据以"是否挂死"为准：其完成标记 `out[]` 用裸标量写 GM，按 m0 结论
> （kernel 退出时标量写可见性无保证）并不可靠——实测只有 8~13/56 落盘。挂死信号是可靠的。

### 4.2 本 kernel 的 flagId 分配

CrossCore flagId（mode 4：AIC 侧 0-15 → AIV0、+16 → AIV1；AIV 侧用基准 id）：

| flagId | mode | 方向 / 含义 |
|---|---|---|
| 0 / 1 | 4 | mmRes ping-pong（S/PV 交接）：AIC `set(PIPE_FIX)` 数据就绪 / AIV `set(PIPE_V)` 消费完可覆写 |
| 5 / 6 / 7 | 4 | P L1 3-buffer：AIV `set(PIPE_MTE3)` P 已写回 / AIC `wait(PIPE_MTE1)` 后装载 |
| 8 | 0 | 全体 AIC barrier（FD partial 全部落 GM）——set 必须 `PIPE_FIX`（AIC 核型 pipe） |
| 9 | 2 | AIV `set(PIPE_MTE3)` partial 落 GM → 配对 AIC `wait(PIPE_S)`（2 set 配 1 wait） |
| 10 | 2 | AIC `set(PIPE_FIX)` 放行 combine → 配对 2 AIV `wait(PIPE_MTE2)` |
| 11 | 2 | AIV `set(PIPE_MTE3)` 收尾 → 配对 AIC `wait(PIPE_S)` |

核内 BufferID（一律阻塞释放：`GetBuffImpl<pipe,false>` 获取 / `ReleaseBuffImpl<pipe,false>` 释放 = CANN 默认 `ASC_LOCK_BLOCK`）：

| 侧 | BufferID | 交接 |
|---|---|---|
| AIC | B_Q(0) | MTE2 Q→L1 → MTE1 L1→L0A（跨 pipe：MTE1 侧也 get B_Q） |
| AIC | B_KV0/KV1(1/2) | MTE2 K/V→L1 → MTE1 →L0B，M 侧 mmad 持有 |
| AIC | B_L0(3) | MTE1→L0A，M 侧 mmad 持有 |
| AIC | B_C0..C3(4-7) | L0C 槽：M 侧 mmad 持有 → 阻塞释放 → FIX 侧 get（Fixpipe） |
| AIV | B_PC(0)/B_ACC(1)/B_MSK(2)/B_CIN(3)/B_COUT(4)/B_ACCIN(5) | V↔MTE3 / MTE2↔V 交接 |

约束遵守：同一 token 绝不被两个 pipe 同时持有（M17 原版在 combine 里嵌套持有 → 自死锁）；
跨 pipe 交接用阻塞释放 BufferID（`false`=CANN `ASC_LOCK_BLOCK` 默认）；同 pipe 背靠背复用由阻塞释放语义 + 单次发射保证。

## 5. 校验方法与结果

host 侧确定性生成 Q/K/V（Hash3，无 rand 状态；K/V 的 pad 段与 Q 的 pad 行置零），
H2D → launch → 同步 → D2H，就地把每 case 的输入/输出/中间 dump 成 `.bin`；
`check_ref.py` 用 numpy **fp64** 按 §1 公式算参考（含 fp32 对照，用于界定预算未被参考自身舍入吃掉），
**输入只取 host 生成的 q/k/v dump**（M59 的 N1 输入隔离），不取任何设备中间张量；判据 A–F 的逐条咬合力见 §9。

当前实测（`logs/check_ref_20260926_exp_spec.log` = **M61 改后**；三档 dump 与 `data/dump_sha256.txt`
逐文件一致，判定口径见 §0 的"本核判定规则"/§6.6）：

```
[check] seq=256,300,4096 | 判定项 23/24 通过 | 报告项 27 单列 | guard 9/9 | 输入缺失 0 项
  判定项 A（out 边界感知 ulp，T3）:
       256  PASS | 常规 6130 越界 0 / 边界 14：ulp 越界 0、绝对界越界 0
       300  PASS | 常规 6086 越界 0 / 边界 58：ulp 越界 0、绝对界越界 0
      4096 **FAIL** | 常规 5924 越界 0 / 边界 220：ulp 越界 0、**绝对界越界 1**
             （out[0][11][217]：absErr 4.21e-7 > 界 3.71e-7，超出 13.6%；位级 99.37%、≤1ulp 99.9837%）
  判定项 B（out 结构性，T4）：三档 PASS（非全零 / 非常数 / 每行 256 列非零）
  判定项 C（P̃ 全数组逐元素，T3，M61 新增）：三档 PASS
       比较 8192 / 9600 / 131072 个元素、越界 0；位差 0/1/2/≥3 格点 = 8192/0/0/0、9600/0/0/0、**131071/1/0/0**
       （唯一 1 处 = 1 个 bf16 格点，最大界占用 0.9965；翻转位置见下方 `check_partials`）
  判定项 D（P̃ 位置/掩码结构性，T4/T1，M61 新增）：三档 PASS
       （未用槽位非零 0/82、0/80、0/52；尾 tile 无效列非零 0）
  判定项 E（P·V，**独立 P**，T3，M61 新增）：三档 PASS
       8192 / 16384 / 65536 个元素、越界 0；maxAbsErr 1.8e-6 / 1.9e-6 / 1.6e-3、最大界占用 ≤5.4e-3
       （4096 的 1.6e-3 = 已知那 1 个 P̃ 翻转经 V 传播的量，见 §6.2 末段）
  判定项 F（P·V，**设备 P**，S1 定位声明）：三档 PASS
       同样覆盖、越界 0；maxAbsErr 1.8e-6 / 1.9e-6 / 3.0e-6、最大界占用 ≤3.4e-3
  guard：host mask 契约 ×3 OK；C 判别力对照 ×3 OK（Δ=0 ⇒ 0；+1 格点 ⇒ 0（在 1.0·ulp 允许差内）；+2 格点 ⇒ 被抓）
         C 口径负向对照 ×3 OK（`globalmax` 口径越界 0（256 档只有 1 个 tile ⇒ 同解、不可判）/ 1983 / 86738）
```

（历史 log 块，保留作对照：`logs/check_ref_20260926_vf.log` = **同一条命令的 M61 改前**输出，
差异只有三处 + 新增判据行：4096 档 ε 270u→272u、边界元素 219→220、常规元素 5925→5924 —— 改前/改后对照表见 §6.6。
BMM2 修复前的三档 maxRel 为 4.169e4 / 8.529e3 / 1.285e4，见下方历史块。）

```
[check] seq=  256 (seq_pad=256)  maxRel=4.169e+04  FAIL
[check] seq=  300 (seq_pad=512)  maxRel=8.529e+03  FAIL
[check] seq= 4096 (seq_pad=4096) maxRel=1.285e+04  FAIL
```

（三档均为 **max** 相对误差；`out` 已非全 0。**注**：正文早期段落里的 "0.26/0.96/1.6e4" 是**首次不匹配**的
相对误差，不是 max（对应的 max 是 4.17e4/7.66e2/1.68e4），两者不可直接比较——reviewer r1 已指出。误差绝对值大是因为 BMM2 的 V^T 装载仍不对——
见 §6 卡点 A 末段；softmax/P 段本身已逐元素精确，见下表。）

分段实测（均为与 numpy fp32 参考的逐元素对比）。
**注意**：`dbgc2`/`out` 两行是 BMM2 修复**之前**的状态（保留作排查轨迹）：`dbgc2` 的 V^T 装载已在
M27 配方下修好（现 `pv` 对拍三档 1.4~1.8e-6），`out` 三档判定见上；`dbgc`/`dbgs`/`wsm`/`wss`/`gp` 五行仍为当前状态。

| 中间量 | 结果 |
|---|---|
| `dbgc`（AIC0 tile0 L0C = 未缩放 QK^T） | ✅ maxErr **1.9e-6** |
| `dbgs`（AIV 侧 S dump，已补排水） | ✅ maxErr **1.9e-6** |
| `wsm`（FD partial 行 max） | ✅ 精确（seq=256/300 逐行相等；300 的尾 tile maxErr 6e-8） |
| `wss`（FD partial 行 sum） | ✅ 精确（256 全 tile 相等；300 split0 = 132.097/147.497/156.381/155.797 逐位相等，尾 tile maxErr 3.8e-6） |
| `gp`（P 的 GM 中转 = 未归一 E，bf16；判据 C/D） | ✅ **T3 全数组逐元素**（M61 起）：每个 (unit,slot,row,col) 对 host 从 q/k **独立重算**的 P̃ —— 比较 8192/9600/131072 个元素、越界 0；位差 0/1/2/≥3 格点 = 8192/0/0/0、9600/0/0/0、**131071/1/0/0**（唯一 1 处 = 1 个 bf16 格点：`gp[unit10][slot0][row11][col252]`，设备 `0x3ee8` vs numpy `0x3ee7`，界占用 **0.9965**）。判据 D 同时咬"未用槽位恒 0"（unit 索引空间）与"尾 tile 无效列恒 0"（列 mask 契约）。**旧读数保留**：`check_gp` 只覆盖 unit0/parity0 的 0–11 行、容差 5e-3（无来源经验值）⇒ 已由判据 C 覆盖并换成派生的 `1.0·ulp+ε_arg`（更严） |
| `dbgc2`（BMM2 L0C = PV；判据 E/F） | ✅ **已修**：判据 F（**设备 P**，S1）三档 maxAbsErr 1.8e-6 / 1.9e-6 / 3.0e-6；判据 E（**独立 P**）1.8e-6 / 1.9e-6 / 1.6e-3（4096 的 1.6e-3 是已知那 1 个 P̃ 翻转经 V 传播的量，§6.2 末段）⇒ V^T 装载正确（旧状态"有数但数值不对"已按 M27 配方修好） |
| `out`（判据 A） | ❌ 三档：256/300 **PASS**、4096 **FAIL 1/6144**（见上；该 1 个元素已归因到一个 P̃ 的 bf16 翻转） |

**归档与复现（证据口径）**：`data/` 归档 q/out/gp/ws*/dbg* 等 27 个文件 + 两个 sha256 清单
（`dump_sha256.txt` = 已归档文件、`dump_sha256_regenerable.txt` = 未归档但可确定性重跑得到的
seq=4096 的 k/v 各 4MB）；一条命令复现并校验全部：
`data/regen_4096_kv.sh /tmp/m10`（构建 + `M10_CASES=…` 重跑 + `sha256sum -c` 两个清单）。
**校正（reviewer r3 的 P2-4）**：该脚本默认 `CASES=256,300,4096`，清单按**本次真正生成的 case 过滤**
（不再出现"拿没生成的文件去 `-c` 而报一堆 open-or-read 失败"），且**任何一项失败都会 `exit` 非零**
（旧版用 `A && echo` 的写法在 `set -e` 下会漏掉失败并退 0，已修）。本轮实测：默认三档 → 25 项 + 2 项全绿；
`CASES=4096` → 7 项 + 2 项全绿；把某个 dump 篡改后 `sha256sum -c` 退出码 = 1（咬得住）。
`logs/` 归档：`run_20260926_vf.log`（三档 kernel 运行）、`check_ref_20260926_vf.log`（**M61 改前**的判定 + gp + pv 全量输出，
留作零回归对照）、`check_ref_20260926_exp_spec.log`（**M61 改后**：判定项 23/24 + 报告项 27 + guard 9/9，本 README §0/§5 引用的是这一份；
r1 评审的 nTile 修后与它**逐字节一致**，故未重发）、
`check_ref_tristate_20260926.log`（退出码三态 **五段**实测，含"只有归档 dump"那一段）、
`eps_ntile_schedule_20260926.log`（`check_ref.py eps` 的 7 档 nTile 记账自查，含窗口 seq∈[3585,3840] 的"旧式少算一档"，r1 评审 P2-1）、
`partials_20260926.log`（seq=4096 残差的 P̃ 翻转定位，`check_partials.py`）、
`dup_probe_20260926.log` / `mmadprobe_20260926.log` / `expprobe_20260926.log` / `l0probe_20260926.log` /
`m10_probe_20260926.log`（五个探针）、`run_20260926_vf_bad_s300.log`（§8.1 那条未隔离观测的原始 trace）。
判定项（判定规则见 §0/§6.6/§9）另可由 `/usr/local/python3.12.13/bin/python3 ../check_ref.py [seq…]`（退出码三态 0/1/2，
见 §0）与 `../check_partials.py` 复跑；只要 P·V 两条：`../check_ref.py pv [seq…]`；
只要 ε 的 nTile 记账（**不需要 dump**）：`../check_ref.py eps [seq…]`；
mmad 探针的辨别力诊断：`./m10_mmadprobe [1]`（加参数 1 = 块对角 A 诊断模式）。
**本轮未重生成任何 `.bin`**（本 mission 不改 kernel、不改 dump 语义 ⇒ 27 个归档文件的 sha256 与 `data/dump_sha256.txt`
逐字相同，`sha256sum -c` 全绿），故 review 里没有新旧 sha 清单需要核对；**新增的只有两个 `.log`**
（`check_ref_tristate_20260926.log` 补第 5 段、`eps_ntile_schedule_20260926.log` 新建）。

## 6. 卡点与复现（交给续跑者）

> **阅读提示**：本节按轮次追加，早期段落的"PV 恒 0 / out 全 0"等表述已被后续轮次修正。
> **当前（本轮 tip）状态一句话**：P 的 L0A 目的偏移缺失与列 mask 两个真 bug 已修（见 §6.1），
> softmax/P/partial 全段逐元素精确；**唯一未解是 V 的转置装载**，`dbgc2` 有数但与任何自然 PV 参考相关性≈0、
> 两个 N 半块逐位相同（M27 的几何映射表出来后按其修）。三档 seq 的 max 相对误差 4.17e4/8.5e3/1.3e4。

### 6.0 V^T 装载：按 M27 几何标定表修复（本 mission 最后一块拼图）

**根因**：BMM2 的 B 操作数原本走 3D `LoadData3DParamsV2` + `enTranspose=1`。
M27（`m16_load_geom`，分支 `feat/l0-load-geometry-calibration` @ `de68fe3`）实测：**3D → L0B 在本平台
完全写不进去（29 组 conf 零写入）**，且文档明确 L0B 通路"自动转置、`enTranspose` 无效"。

**修复**（M27 `README §3.2/§3.4` 逐字段表 + `§4` 真实形状配方，用本 mission 的真实 P/V dump 逐位验证过）：

```text
L1 已是 Nz（行 = token = K、列 = dim = N，dstNzC0Stride = K）
B 装载 = LoadData2DParamsV2{mStep = K/16, kStep = N半/16, srcStride = K/16, dstStride = N半/16,
                            ifTranspose = true}      // T1：分形内 16x16 转置
源码偏移 = nh*(128/16) 个列分形 × srcStride(16) × 512B = nh*32768 元素（靠 LocalTensor 切片）
mmad：单次 k=256（M27 在真实 BMM2 形状上已验证合法）、n=128
```
**★ 不可照抄 BMM1 的元组**（M27 §4.1 用 `m16_pv baddst` 给出反例：`dstStride` 必须 = `N/16`，抄 `K/16` 直接 507015）。

**验后**：`pv` 对拍三档 maxAbsErr 1.4/1.8/1.6e-6（4096/4096 非零）⇒ 与 M27 的结论一致；
seq=256/300 的 FA 判定项随即全部 PASS（修前 maxRel 4.17e4/8.5e3）。

### 6.2 4096 残差的逐级定位（tower 要求：定位到具体哪一步）

按"把在线重标定建进参考 + 逐级对拍"把链路拆成四段判：

| # | 段 | 判据与结果 |
|---|---|---|
| ① | 逐 tile 的 **P̃**（bf16） | `gp[unit][slot]` 比对：**范围仅 split0 的 tile0/tile1**（旧的 `check_gp` 只查 unit0/slot0/12 行/容差 5e-3）⇒ 该窄范围内 **bit-exact**，且 tile1 的 bit-exact 是"用运行 max"的 P̃（差 0.0）⇒ 设备用**逐 tile 在线跑 max** 的舍入，与 kernel 源码一致。**注意**：这**不是** gp 全数组逐位——全数组事实是 131071/131072（1 个翻转），见 §6.4。**M61 起**这一窄范围由**判据 C**（全数组、派生容差）覆盖，读数保留在同一行输出里；口径的外部 pin（donor 的 `Max(新,旧)`/`ExpSub`）写在 `check_ref.py` 文件头 |
| ② | 逐 tile 的 **PV** | 判据 F（设备 P）maxAbsErr 1.4~3.0e-6、判据 E（独立 P）1.4e-6~1.6e-3、4096/4096 非零 ⇒ BMM2 通路 OK（M61 起两条都覆盖全 unit 的 tile0） |
| ③ | **在线重标定因子** `e^{m_old−m_new}` | 用设备 partials + fp64 复算 combine 与设备 out 差 6.0e-5；**fp32 与 fp64 复算结果相同** ⇒ 不是 combine 的 fp32 舍入 |
| ④ | **acc/sum 的累加** | **根因**：我最初的参考把 `P̃` 写成 `bf16(exp(S − split_max))`，而 kernel 是**逐 tile、用当时的运行 max** 舍入 —— 两者差**一次 bf16 量化**（≈0.2% 相对），使 partials 偏 1e-3、最终 out 落在 91% ≤1ulp。**修正参考后 4096 从 91.0% → 99.9837% ≤1ulp（位级 65.7% → 99.37%）** |

⇒ 结论：**残差的主因是参考建模（不是 kernel 精度）**。修正后仅剩 **1/6144 元素 >1 ulp（2 ulp）**：
`out[0][11][217]` = -4.7922e-05（参考 -4.7445e-05，绝对差 **4.8e-7**；该元素 |out| 比同行量级小约 200×，是相消点）。

**第五段判（2026-09-26，第十一轮）：这 1 个元素已完整归因到「一个 P̃ 的 bf16 舍入翻转」。**
用 kernel dump 的每 split 部分和（`wsacc/wsm/wss`）逐层定位，工具 `check_partials.py`（日志
`logs/partials_20260926.log`，可复现）：

| 步骤 | 证据 |
|---|---|
| ① 逐 split 比 acc | 8 个 split 里 **7 个**的 \|Δacc\| 只有其保守界 `k·u·Σ|P̃V|` 的 ≤1.2e-4（= §6.3 的 0.13% 量级），**只有 split5 达 0.0644**（1.56e-3） |
| ② 秩一结构 | 该 split 的 Δacc 是**一条 V 行的倍数**：最佳拟合 token **2812**，Pearson **0.99999995**、残差比 4.7e-6；系数 α = **+1.953044e-3** = P̃(2812) 的 bf16 ulp 的 **+1.0000 倍** |
| ③ 因果闭链 | 把**这一个** P̃（split5, row 11, token 2812: 0.451171875 → 0.453125）注入参考重算：整档 **>1ulp 元素 1 → 0、≤1ulp 99.9837% → 100%**，且 `out[0][11][217]` 由 0xb847 变为 **0xb849 = 设备值（逐位相同）** |

机制：P̃ 在中途被量化到 bf16（本核规格），而 S 是 fp32 mmad、exp 是 **Reg `Exp`（官方规格 ≤1 ulp，实测 1.1~1.4 fp32 ulp，§6.4/§6.6）**。
当 `exp(S−m)` 距某个 bf16 量化边界小于该误差（概率 ~1e-4/元素；三档实测 1/131072）时，P̃ 的舍入方向与独立 numpy 参考不同；
一个 P̃ 翻 1 个 bf16 ulp 对输出的影响 = `2^-8·|P̃_t·V_t|·w/den`，在相消 440× 的元素上约为输出的 2 ulp。
**这是设计固有的离散量化事件，不在 `ε·Σ|acc·w|`（fp32 累加误差）的量纲内**——所以 §6.6 的绝对界覆盖不了它。
**诚实结论**：`out` 在 ≤1 ulp 意义上可达 6143/6144；剩下 1 个元素的偏差已定位、可复现、
量级与机制都给出，但**它落在既定判定界之外** ⇒ 按 tower 指示**不自行放宽、不申请例外**，交由 tower 裁定
（可选的干净出路见 §6.6 末段）。

### 6.3 mmad k 方向累加标定与"有界误差"判据的现状（tower 第十轮要求）

**已完成（可复现数字）**：
- **真实数据上的 mmad k=256 累加误差测量**（用设备 BMM2 的 L0C 对同一组 bf16 `P̃`/`V` 的 fp64 参考）：
  seq=256 maxAbsErr **1.43e-6**（Σ|A·B| 最大 71.5）、seq=4096 **1.61e-6**（Σ|A·B| 最大 69.9）
  ⇒ 只占保守解析界 `k·2^-24·Σ|A·B|` 的 **0.13~0.15%**（界覆盖、余量 ~700×）。
  复现：`check_ref.py pv`（`[pv]` 行）。
- **解析界**（报告项，非判据）：`ε_total = (k + nsplit + 2·nTile)·2^-24`，其中
  `k·2^-24` 为 mmad 累加项、`nsplit + 2·nTile` 为 VF/combine 项；判据形式 `|out−ref| ≤ ε_total·(Σ|acc·w|/den)`。

**第十二轮：探针跑通了，但**它的 B 操作数映射是错**的 ⇒ 该探针**无法**标定 mmad（明确 blocker）**

上一轮修好了 507015（根因 = L0B 单次装载 footprint 越界：B 一次装 K=256 时 `kStep=16` ⇒ 16·16·512B = **128KB > 64KB**；
改为"拆 k=128 两块" = 主核 BMM1 同款后跑通）。本轮为了**判定 d3/d4 到底是谁的错**，把分布改成有辨别力的一组
（`e1` 单格 (n0,k0)、`e2` 单格 (n0,k128)、`e3` 仅 k=0 列、`e4` 16-列块签名、`e5/e6` 仅 chunk0/仅 chunk1、
`e7` absorb、`e8` incr、`e9` rand），并加了 `./m10_mmadprobe 1` 的**块对角 A 诊断模式**（A[m][k]=1 iff k/16==m）
⇒ `C[m][0]` 直接读出"设备把哪些源 16-列块放进了它的第 m 个 k 块"。读数（日志 `logs/mmadprobe_20260926.log`）：

```
e4 块签名的 C[m][0]:  0.5, 0.25, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, ...   ← 正确应为 128, 64, 32, 16, ...
⇒ 设备实际用的 B 操作数 = 源列 {32..255} ∪ {128..159}：**源 k 块 0/1（列 0..31）根本没进 mmad，
   源块 8/9（列 128..159）被读了两次**（A 侧映射是恒等）
```

三条独立证据互相印证（都可复跑）：① `e1`(单格 (n0,k0)) 与 `e3`(仅 k=0 列) 的行和都是 **0**（该列没进）；
② `e5`/`e6` 在 A=全 1 时给出 **96 / 160**（= 128−32 / 128+32，总数 256 守恒）；③ `e4` 的行和实测 **64.7461**，
与模型 Σ_{j=2..15}16·8·2^-j + 16·8·(2^-8+2^-9) = 63.996 + 0.75 = **64.746** **逐位吻合**。
把 B 装载后的 `PipeBarrier<PIPE_MTE1>` 换成 `PipeBarrier<PIPE_ALL>` 后读数**不变** ⇒ **不是跨 pipe 可见性/时序**，
而是探针 B 通路的**确定性几何错**。（`d0/d1/d2` 当年"精确"是因为这几组分布在 16 列块内恒定 ⇒ 对这种块级
位移天然免疫；`d1` 的算术也正好抵消，可以手算验证——所以"d0~d2 精确"从来不是 B 通路正确的证据。）

**结论（如实报 blocker，不用"已归档/已复现"盖过去）**：**tower round-10/11 要的"k 方向误差 / 吸收性 /
最坏情况关系"本轮仍未给出**——用这个探针**给不出来**，因为它自己的操作数就不是它以为的那份数据。
主核路径不受影响（其 BMM1 的 `dbgc`/`dbgs` 与 numpy 一致、`out` 三档与完整参考对齐到 ≤1 ulp），
但**不能用主核的正确性去替这个探针担保**。可选出路（下一轮，二选一）：
- **(a) 修探针**：把 B 通路逐字节对齐主核已验证的 BMM1 装载（同 L1 偏移/同 `LoadL0_2D` 形态/同 mmad 序列，
  含主核那套 `Acq/Rls` 交接），修好后重跑本轮的辨别力分布（`e1..e6`）直到读数与参考一致，再做 k 扫描；
- **(b) 换路径标定**（推荐）：不靠独立探针，**在主核已验证的 BMM2 路径上**用对抗性 V 做标定——加一个
  `M10_SYNTH`/`M10_VDIST` 模式把 GM 里的 V 换成 {ones / incr / 交替相消 / 吸收 / 随机}，再拿 `dbgc2`（L0C，fp32）
  与"设备自己的 P̃（`gp`）× V"的 fp64 参考比 ⇒ 直接得到真实通路上 mmad 的 k 方向/吸收性误差表；
  `wsacc`（fp32 部分和）同时给出跨 tile 链的误差。

**已有的事实（不是最坏情况标定，别当界用）**：真实数据分布下，设备 BMM2 L0C 对同一组 bf16 P̃/V 的 fp64 参考
maxAbsErr **1.43e-6(seq=256) / 1.61e-6(4096)**，占保守界 `k·2^-24·Σ|A·B|` 的 **0.13~0.15%**（复现 `check_ref.py pv`）。
⇒ ε 里的 mmad 项目前是**纯理论保守界**（`k=256` 全额计入），**没有**对抗性分布的最坏情况支撑。

**未完成 / 如实报**：
2. **"单一相对累加界"的覆盖率本身不构成判据**：该界的量纲是"归约噪声底"，不是输出网格。
   seq=4096 只有 **328/6144** 个元素的绝对误差小于 `ε·(Σ|acc·w|/den)`——但这不是"界被违反"，
   而是其余 95% 元素的误差由**输出自身 bf16 网格**（≤0.5 ulp = 0.2%·|out|）支配，与累加噪声无关。
   把 `0.5·ulp(out)` 计入后，越界的就只剩相消区元素（边界元素 219 个中的 1 个，即 §6.2 的 P̃ 翻转）。
   ⇒ 判定口径已按 tower 第十一轮要求改成**边界感知判据**（§0 判定规则 / §6.6 推导与 δ），
   **不放宽、不申请例外**：256/300 PASS、4096 判 FAIL（1/6144，绝对界超 14%），fail 的原因如实给出。

### 6.4 VF `Exp` 精度：探针修好一半 + ε 的 Exp 项改由独立证据支撑

**⚠ 先说结论的有效范围（reviewer r3 的 P2-2）**：原表（下面"历史表"）里的"小差值 1.2 ulp / 大负 / 随机"三行
**不可用**——探针在 8KB 之后仍返回垃圾。**当前结论只由前三个区间 + 与探针无关的 gp 统计支撑**（见本节末），
**表里后三行按撤回处理**。

历史表（**前 3 行为修好后 5 次逐字节一致的可复现读数；后 3 行已撤回**）：

| 输入分布 | maxRel | 折合 fp32 ulp | >1ulp 个数 | 状态 |
|---|---|---|---|---|
| x∈[-2,0]（**重标定因子所在区间**） | 8.389e-8 | **1.4** | 19/512 | ✅ 可复现（5 次同一 md5） |
| 2 的幂边界附近 | 6.416e-8 | 1.1 | 5/512 | ✅ |
| x∈[-20,0]（softmax E） | 8.130e-8 | 1.4 | 29/1024 | ✅ |
| x∈[-1e-3,1e-3]（小差值） | 1.0 | — | — | ❌ **撤回**（区域在 8KB 之外，读数 = 垃圾） |
| x∈[-100,-20] | 1.0 | — | — | ❌ **撤回**（同上） |
| x∈[-30,0] 随机 | inf | — | — | ❌ **撤回**（同上） |

⇒ 可支撑的说法（收窄后）：**在与本核 Exp 用法相关的区间上，VF `Exp` 的相对误差 ≈1.1~1.4 fp32 ulp**
（≈1e-7 相对），比"解释 4.8e-7 输出偏差所需的 ~2.4e-5 相对"小两个数量级 ⇒ **Exp 不是缺的那一项**。
> 探针修法记录（两轮）：① 首版输出全是垃圾 → 补 V→MTE3 的 drain BufferID 后第一版读数正常；
> ② reviewer r3 复现出运行间不稳定（0.97/inf/inf）⇒ 本轮把 GM↔UB 的两条 20KB `DataCopy` 改为
> 512B×40 分块 + `PipeBarrier<PIPE_ALL>`，**前 8KB 变为 5 次逐字节一致**；8KB 之后的残留在 README §7 记为已知限制。

**ε 的 Exp 项：M61 已换成官方规格（本节此前的"假设"口径撤回）**

**M61 修订**：ε 里的 Exp 项**不再是假设**，也不再依赖本探针 —— 它现在是**官方 Reg 规格 + 最坏情形换算**
（`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md` 表 1
「基础算术 / Exp」= `1ulp, not support denormalized numbers`（硬件）/ `1ulp, support denormalized numbers`（软仿）；
`.../reg_vector_compute/basic_arithmetic/Exp.md` 的「最大精度误差」节 = 1 ulp、**约束说明：无**）。
完整的"1 ulp → `2·2^-24`"换算、以及任务要求的 **VL/输入域条件分析**，见 **§6.6**。
**此前那句"本仓库内没有该规格（已检索 `docs/`）⇒ `EXP_ULP=2` 是显式假设"撤回**：那次检索只覆盖了 `docs/`，
**没有查 CANN / asc-devkit 的官方 API 文档树**（规格就在 `asc-devkit/docs/zh/api/appendix/` 与
`/usr/local/Ascend/cann-9.1.0/` 下的对应文档里）。按 M59 的口径，旧写法属"用实测值当界"（docs/17 §1.1 明禁）；
Exp 项的**数值恰好不变**，改前/改后对照见 §6.6。

**第十二轮：探针修好了一半 + Exp 项改为不依赖探针（reviewer r3 的 P2-2）**

reviewer r3 的三次运行给出 `in=-2 → 0.97044772 / inf / inf`（声称值 `0.13533528` 从未出现）——**旧版确实
运行间不稳定，该结论我接受**。根因（本轮定位并部分修复）：kernel 里 GM→UB / UB→GM 各用**一条 20KB 的
`DataCopy`**（5120 float），改成 512B/块的 40 次分块搬运 + `PipeBarrier<PIPE_ALL>` 后，**前 8KB 稳定**：
与主核 Exp 用法直接相关的三个区间给出**连跑 5 次逐字节相同**的读数（5 次输出 md5 同一值）：

| 区间 | maxRel | 折合 fp32 ulp | >1ulp |
|---|---|---|---|
| x∈[-2,0]（**在线重标定因子**所在区间） | 8.389e-08 | **1.4** | 19/512 |
| 2 的幂边界附近 | 6.416e-08 | **1.1** | 5/512 |
| x∈[-20,0]（softmax E） | 8.130e-08 | **1.4** | 29/1024 |

**仍如实报未修好的部分**：8KB 之后的 3 组（小差值 / 大负 / 随机）仍返回垃圾（`maxRel=1.0` 或 `inf`）⇒
该探针**只在上面三个区间可用**，日志 `logs/expprobe_20260926.log` 的头部已写明这一范围限制。
**本探针从此只作交叉见证（不再是 ε 的来源）**：官方规格（≤1 ulp）见 §6.6 与新段首；下面这条独立证据
用来支撑"**实测没有超出规格**"：主核 `gp` 全数组 131072 个元素里**恰好 1 次** bf16 边界翻转
（§6.2⑤，reviewer 独立复现），反推 Exp 的相对误差上界：翻转率 ≈ 2·ε_exp/2^-8，观测 1/131072 ⇒
用 Poisson 95% 上界（≤4.7 次）得 ε_exp ≤ 4.7/131072·2^-9 ≈ **3.6e-8 ≈ 0.6 fp32 ulp**（保守取 ~1.2 ulp）
——**≤ 官方 1 ulp 界** ✓。（上式的假设：各元素到 bf16 边界的距离在 [0, 半个网格) 上近似均匀、
翻转近似独立；量级结论不依赖这两条。）

**并且回答了"完美 combine 用的 exp 是哪一侧"**：我用的是 **numpy fp64 exp**。那 3.9~6.0e-5 的差
**不是 exp** —— 真正的来源是**输出自身的 bf16 网格**（设备 `out` 是 bf16，我的复算不是）：
`0.5·2^-8·|out| ≈ 4e-5`（|out|≈2e-2）与实测 6e-5 吻合。

### 6.5 有界误差判据：为什么"累加相对界"覆盖不了（结论）

逐元素差异由三项构成：**(a) 累加噪声**（相对 `Σ|acc·w|`：mmad k·2^-24 实测占保守界 0.13~0.15%、Reg `Exp` ≤1 ulp（官方规格）、
VF/combine 若干 ulp）+ **(b) 输出 bf16 网格**（≤0.5 ulp ≈ 0.2%·|out|）+ **(c) 中途 P̃ 的 bf16 量化翻转**（离散，§6.2 末段）。
(b) **不是相对 `Σ|acc·w|` 的**，所以"单一相对累加界"在相消元素上必然不覆盖：4096 只有 328/6144 个元素的
绝对误差小于 `ε·(Σ|acc·w|/den)`——但这只说明"归约噪声底"这条尺子在相消区不再有意义（相消比可到 440×），
不是"界被违反"。由 (b) 引出的正确判据形式是**把输出网格的 0.5 ulp 显式写出来、并对"参考落在两网格点中点附近"
的元素单独定规则** ⇒ 见 §6.6（本核现行判定规则）。

### 6.6 边界感知判据：δ 的推导、取值后果与三档实测（本核判定规则；ε 的 Exp 项 M61 换成官方规格）

**判据**（实现见 `check_ref.py` 的 `check()` / `eps_terms()`）：
```
ε        = (S2T + nsplit + 2·nTile + EXP_ULP·nTile)·2^-24   # 各为一次 fp32 相对舍入（≤2^-24），除 Exp 项外
noise_ub = ε · (Σ|acc·w| / den)                        # 归约噪声的绝对上界（相对"累加量级"）
δ        = noise_ub / ulp(out)                         # 边界带宽（以输出 ulp 为单位）
元素分类：|ref − 相邻两网格点中点| ≤ noise_ub  ⇒ 边界元素，否则常规元素
判定：常规元素 |out−ref| ≤ 1 ulp（严格）
      边界元素 |out−ref| ≤ 2 ulp  且  |out−ref| ≤ noise_ub + 0.5·ulp(out)（绝对界）
```
各项含义（**M61：Exp 项的出处与次数都改了，推导见下**）：
- `S2T=256`：cube 内 k 方向累加，保守取 `k·u·Σ|A·B|`（真实数据实测只占保守界的 0.13~0.15%，此处仍**全额计入不放松**）；
- `nsplit`：跨 split 归并的乘加（num/absnum/den 各一）；
- `2·nTile`：每 tile 在线重标定 `acc·e^{Δm}` 与新 tile 累加各一次 fp32 舍入；
- `EXP_ULP·nTile`：Reg `Exp` 自身误差（**官方规格 1 ulp**）在两条路径上的次数 —— 每个元素被
  `E = exp(S−mNew)` 影响 1 次/tile，被重标定因子 `exp(mOld−mNew)` 影响 (nTile−1) 次
  （首 tile 的 `mOld = −inf/MASKV` ⇒ `exp` 精确为 0、无误差）⇒ 合计 **nTile 次 × EXP_ULP**
  （旧式只写 `+EXP_ULP` = 只算了 E 那一侧，漏掉重标定因子那一侧）。
- **`nTile` 的定义 = 实际调度里每 split 最多的 tile 数 = kernel 的 `chunkTiles`**
  （`check_ref.py::eps_terms` 直接数 `split_ranges()` 的 token 区间，并与 kernel 的
  `chunkTiles = CeilDiv(nTiles,SPLITS)` / `nsplit = CeilDiv(nTiles,chunkTiles)` 逐项断言对齐）。
  **M61 r1 评审 P2-1 修**：旧式写 `max(1, ceil(seq/S2T)//nsplit)`，在 `n_tiles = 15`
  （**seq ∈ [3585, 3840]**，如 3600）时给 **1** 而实际是 **2** ⇒ 该窗口 ε 少 `4u`
  （`2·nTile` 与 `EXP_ULP·nTile` 各少 2u、约 −1.5%，方向是**更严**、不会假 PASS）；
  三档文档化用例（256/300/4096）的 nTile 本来就是 1/1/2，**不受影响**。
  自查命令（不需要 dump）：`../check_ref.py eps [seq…]`，读数归档
  `logs/eps_ntile_schedule_20260926.log`（7 档：256/300/3584/**3585/3600/3840**/4096，
  窗口三档标出"旧式少算一档（已修）"）。

**Exp 项的三段推导（回答 docs/17 护栏 2 要的"官方精度规格"，含任务要求的 VL / 输入域条件）**：

1. **规格出处**：官方 Reg 矢量计算精度表（`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`
   表 1）「基础算术 / Exp」= `1ulp, not support denormalized numbers`（硬件）/ `1ulp, support denormalized numbers`（软仿）
   —— **两个分支同为 1 ulp**；接口文档 `.../reg_vector_compute/basic_arithmetic/Exp.md` 的「最大精度误差」节
   同样给 1 ulp，且该文档**约束说明：无**（没有 VL / repeat / mask 的附加限制）。
   **适用性（pin 对得上）**：本核三处 Exp 都是 **Reg** `Exp`（`__VEC_SCOPE__` + RegTensor、fp32、默认
   `ExpAlgo::INTRINSIC`；调用点 = `m10_attn_decode.asc` 的 `SoftmaxTileVf`（E 与 expdiff）与 `CombineWeightsVf`）
   ⇒ 用 Reg 规格，而不是经典 memory-based 的精度表。
2. **1 ulp → 相对界（最坏情形，不是观测）**：`y = exp(x) ∈ (0,1]` 落在 binade `[2^e, 2^(e+1))` 时
   `ulp(y) = 2^(e-23)`，而 `y ≥ 2^e` ⇒ `|Δy|/y ≤ 2^-23 = 2·2^-24` ⇒ 本式（单位 2^-24）取 **`EXP_ULP = 2`**。
3. **输入域条件（实测自三档 dump，可复算）**：三档的 Exp 自变量 = `S−m ∈ [−1.67, 0]`（`|S| ≤ 1.01`）、
   重标定 `mOld−mNew ≤ 0`、combine 权重 `m_i−max ≤ 0` ⇒ **结果全部落在 `[0.185, 1]`（正常数）**，
   不溢出、不产生次正规/NaN ⇒ 规格里"not support denormalized numbers"（硬件的 FTZ）**在本数据域内不触发**，
   两个分支给同一 1 ulp 界。两处"精确 0"也不是近似：mask 列的自变量 = `−3.39e38`、首 tile 重标定的自变量
   = `MASKV−mNew ≈ −3.39e38` ⇒ `exp` 精确为 0（离次正规区还有 38 个数量级）⇒ 与 FTZ 与否无关。
   **把边界写清**：若将来某档出现 `x < −87.34`（结果落进次正规区），硬件 FTZ 会直接清零（相对误差 100%），
   此时该元素对 `out` 的贡献 ≤`2^-126`（绝对），在判据 A 的绝对界 `0.5·ulp(out)` 之下、不影响判定；
   但**判据 C（P̃ 元素级）必须单独给这类元素解释** —— 本轮三档实测该情形 **0 个元素**（已复算）。
   **两条交叉见证（已不是 ε 的来源，只说明"实测没有超出规格"）**：`m10_expprobe` 在相关三区间 1.1~1.4 fp32 ulp；
   `gp` 的 1/131072 翻转率反推 ≤~1.2 fp32 ulp（两者见 §6.4）。

**M61 改前/改后对照（零回归表；`logs/check_ref_20260926_vf.log` = 改前 vs `..._exp_spec.log` = 改后）**：

| seq | ε（改前 → 改后） | 判定 A 常规元素 | 判定 A 边界元素 | max δ | 判定 A |
|---|---|---|---|---|---|
| 256 | 261u（**不变**） | 6130 → **6130** | 14 → **14** | 0.0038 | PASS → **PASS** |
| 300 | 262u（**不变**） | 6086 → **6086** | 58 → **58** | 27.28 | PASS → **PASS** |
| 4096 | 270u → **272u** | 5925 → **5924** | 219 → **220** | 9.21 → **9.28** | FAIL 1/6144 → **FAIL 1/6144** |

（改动只有 `EXP_ULP` 那一项的**次数**：`+EXP_ULP` → `+EXP_ULP·nTile`；nTile=1 的 256/300 数值逐位不变，
4096 的 nTile=2 ⇒ ε +0.74%。**这不是放宽**：补的是旧式漏掉的"重标定因子那一侧"的 Exp 误差（推导见上），
且 4096 那 1 个越界元素**仍然越界**（`absErr 4.213e-7` vs 界 `3.692e-7 → 3.710e-7`，超 14% → 13.6%）。
判定项/报告项/guard 的总账见 §0；判据 C/D/E/F 为新增，与本次 ε 改动无关（它们用自己的 `ε_arg`，见 §9）。
**r1 评审 P2-1 的 nTile 修（seq ∈ [3585,3840] 少算一档）不改变上表任何一行**：三档的 nTile 本来就是 1/1/2，
修后 `check_ref.py` 的输出与已归档的 `logs/check_ref_20260926_exp_spec.log` **逐字节一致**（本轮实测 diff 为空），
故该 log 无需重发；受影响窗口的记账读数另归档在 `logs/eps_ntile_schedule_20260926.log`。）

**为什么需要"边界元素"这一档（δ 的取值后果）**：bf16 输出只有 7 位尾数（网格 2^-8 相对）。
若参考值恰好落在两网格点**中点**附近，判定"设备应落在哪一侧"本身就不成立——设备的 fp32 归约
最后一次舍入（≤0.5 ulp）足以把它推到任一侧的相邻格点，于是 |out−ref| = 1 ulp **或 2 ulp 都可能**。
而噪声带宽度是 `noise_ub`（相对累加量级）；把两者相除得
```
δ = noise_ub/ulp(out) ≈ ε·2^8·(Σ|acc·w|/den)/|out| = ε·256·(相消比)
```
⇒ **δ 与相消比成正比**。这不只是"某个界不够宽"：δ ≫ 0.5 时（相消比大），参考值落在"中点 ±noise_ub"
这条带内的概率与"设备 vs 参考的 1 ulp 差异不可判"是同一件事 ⇒ **在深度相消元素上"≤1 ulp"在数学上不可推导**
（不是"我用不了严格口径"，而是该命题在那里没有定义）。三档实测：256 的 max δ = **0.0038**（即 ≤1 ulp 完全可推导）、
300 = **27.28**、4096 = **9.28**。
> δ 的推导口径说明：`δ = ε·Σ|acc·w|/ulp(out)` 与 tower 建议的
> `δ ≈ ε_acc·Σ|acc·w|/ulp(out)` 同式；若按"只算 VF/combine 项"（`ε=(nsplit+2·nTile)·2^-24`）则 δ 小一个量级，
> 但会漏掉 mmad 的 k 项——本核按**实际使用的 ε**（含 `k` 与 `EXP_ULP`）给出 δ，故偏保守。

**三档实测（两栏元素数量与越界数；日志 `logs/check_ref_20260926_exp_spec.log`）**：

| seq | 常规元素（判 ≤1 ulp） | 边界元素（判 ≤2 ulp + 绝对界） | max δ | 判定项 A |
|---|---|---|---|---|
| 256 | 6130 个，越界 **0** | 14 个，ulp 越界 0、绝对界越界 0 | 0.0038 | **PASS** |
| 300 | 6086 个，越界 **0** | 58 个，ulp 越界 0、绝对界越界 0 | 27.28 | **PASS** |
| 4096 | 5924 个，越界 **0** | 220 个，ulp 越界 0、**绝对界越界 1** | 9.28 | **FAIL（1/6144）** |

⇒ 256/300 = **100% ≤1 ulp**（这两档根本没有元素需要动用边界档，δ 也说明其相消程度不足以让 ≤1 ulp 失效）；
4096 = **6143/6144 ≤1 ulp**，剩下那 1 个是边界元素（`out[0][11][217]`，绝对界 3.710e-7 vs 实测 4.213e-7，超 13.6%），
已被**完整归因到单个 P̃ 的 bf16 翻转**（§6.2 末段，`check_partials.py` 因果闭链）——
即上面第 **(c)** 类机制，**不在** `noise_ub + 0.5·ulp(out)` 这个（只描述 fp32 归约 + 输出网格的）界的量纲内。
**本次不复核、不放宽**：不把 (c) 塞进判定界去凑 PASS；由 tower 裁定是接受"1/6144 且机制已定位"还是要求返工。
> **M61：把原"出路 ①"正式否掉（不采纳），理由见 §7.1**。原文建议"参考口径改为『以设备的 P̃ 为输入、
> 只对 sum/combine 段独立复算』"——那等于把 **S1（设备产物入参）** 写进**主**判据：P̃ 的舍入口径从此不可咬。
> 本档恰好有一个现成的反例证明它有多危险：4096 唯一越界元素就落在**唯一一处**设备 P̃ 与 host 参考 P̃
> 相差 1 个 bf16 格点的元素上（判据 C 量出：`gp[unit10][slot0][row11][col252]`，占用 0.9965）——
> 主判据一旦改吃设备 P̃，这个 FAIL 会被**吃掉**变成 PASS（`check_partials.py` 的注入实验正是这条闭链）。
> 现在的落法是：**保留** host 从 q/k 重算 P̃ 的主链（判据 A），**另外新增**独立的判据 C/D（P̃ 自身）
> 与判据 E（独立 P 的 P·V）来补咬合力，而不动主判据（§9）。"出路 ②"（设计上不让 P̃ 中途掉到 bf16）
> 仍留作设计侧的可选项，不在本 mission 范围。

### 6.1 本轮（reviewer r1 后）修掉的两个真 bug

| # | bug | 证据与修复 |
|---|---|---|
| 1 | **P staging 的 unit 索引空间错**：AIV 写用 `bid`（0..55），AIC 读用 `GetBlockIdx()`（0..27），只有 AIC0 对上；且 AIV≥28 时越界写坏相邻 unit | 修为 `unit = bid >> 1` 并加 `unit < SPLITS*2` 保护。复验：`gp` unit0 的 rows8-15 由全 0 变为正确 E（maxErr 1.95e-3 = bf16 网格） |
| 2 | **列 mask 被加到每个 tile**（host 只生成一份，VF 无条件 `Add`） | 修为只有尾 tile（`valid < S2T`）加 mask，非尾 tile 用 V 侧清零的全 0 mask。复验：seq=300 split0 行 sum 132.097/147.497/156.381/155.797 与参考**逐位相等**（修前只有 88 个 token：56.96/58.62/…）；尾 tile wss maxErr 3.8e-6、wsm 6e-8 |
| 3 | 附带：Combine 路径上的 **AICore exception 507015** | 改为 VF 整表计算 Combine 权重（`CombineWeightsVf`）；同时按 reviewer 要求去掉 Combine 里的经典 `Cast`（改 `CastRowToBf16Vf`）与标量写 UB。**⚠ 归因更正（reviewer r2 的隔离探针 `m10_dup_probe.asc`，本仓库已收录、日志 `logs/dup_probe_20260926.log`）**：507015 **不是**"calCount=1 的经典 V 指令"触发的（早期归因错误，勿再引用）：真因是 **UB 目的地址非 32B 对齐**。14 个 mode 的实测表见 §6.7。 |

### 6.7 507015 的隔离探针（`m10_dup_probe`）与"标量读 VF 写的 UB"的如实标注

**结论（reviewer r2 提供隔离探针，本仓库逐字收录、本轮复现）**：
`AICore exception 507015` 由 **UB 目的地址非 32B 对齐**触发，与 `calCount` 无关。

| mode | 指令 | 结果 |
|---|---|---|
| 0/1/2/3/6 | `Duplicate(ub, 1.0f, {1,2,8,64,2048})`（base，32B 对齐） | ✅ err=0 |
| 4 | `Duplicate(ub[8], 1.0f, 1)`（32B 对齐切片） | ✅ |
| 5/7 | `Duplicate(ub[1], 1.0f, {1,8})`（**4B 偏移**） | ❌ **507015** |
| 9 / 10 / 12 | `ub[2]`（8B）/ `ub[4]`（16B，fp32）/ `ubb[8]`（16B，bf16） | ❌ **507015** |
| 11 / 13 | `ub[16]`（64B）/ `ubb[16]`（32B，bf16） | ✅ |
| 8 | `Duplicate(ub,0.5f,1)` + 经典 `Exp(ub,ub,1)`（calCount=1） | ✅ err=0（out=e^0.5=1.64872） |

⇒ 措辞应为：**非 32B 对齐的 UB 目的地址会让 `Duplicate`/经典 V 指令以 AICore exception 507015 报错
（4/8/16B 偏移必挂，32B 对齐正常；与 calCount 无关）**；507015 是通用异常码，不指向具体指令。
复现：`./m10_dup_probe <0..13>`（14 次 launch，各约 8s；日志 `logs/dup_probe_20260926.log`，
源码 sha256 见该日志末行）。

**如实标注（剩余的一处 primitive 不是已验证的那一个）**：Combine 里"标量只**读** VF 写好的权重"这一方向
（`wM.GetValue(...)`/`sM.GetValue(...)`）现在用的排水 primitive 是 **`PipeBarrier<PIPE_V>`**，
**不是** docs/05 #22 验证过的那个（#22 验证的是"标量写 UB"的危险方向；本核的方向是**只读**、本身不写，
但 #22 的结论不能直接搬来给"读"背书）。10 次重复运行未复现任何异常，但这**排不掉 ~1/30 量级的事件类**——
即本核在这一处仍有一个**概率性未标定项**，如实记录（若后续要收紧，可把权重读改成 VF 侧的选择/广播，
彻底去掉标量读 UB）。

**卡点 A（已定位并修掉主因）—— P 的 L0A 目的偏移缺失（真根因），以及 V^T 装载（剩余待办）**

### 真根因与修复（第四轮，commit 见下）

`LoadL0_2D` 只接受"源偏移"，目的永远从 `l0Dst` 的 **位置 0** 开始写；而本核的 L0A 约定是
Q 常驻头部 `0..8KB`、P 进 `8KB` 起的槽（mmad 用 `aElemOff = L0A_OFF_P/2 + ks×2048` 取 P）。
⇒ P 实际写到了 L0A 头部（把 Q 覆盖了），mmad 从 8KB 处读到的**恒为未初始化内存** → PV 恒 0。
这一个 bug 解释了此前"B 侧换什么都不出数"的全部现象（消元实验 1-4 报的 0 都是同一原因）。

修复：给 `LoadL0_2D` 加 `dstElemOff` 参数，P 装载传 `L0A_OFF_P/2`。修复后：
- `dbgc2`（BMM2 L0C）**立即出数**（不再恒 0）；
- 同时按 tower 决策 3 把 P 落盘改为 **UB→GM→AIC→L1**（AIV `DataCopy` 写 GM 中转 buffer `[unit][parity][16][256]`，
  AIC 用 `Nd2Nz` 搬进 L1），GM 里的 P 数值经核对与参考 E 一致（例 row0[:4] = 0.695/0.570/0.855/0.412）；
- 该路径也顺带给出了**可信的 P dump**（`m10_case_s*_gp.bin`，不再依赖 UB 拷贝）。

### 第五轮（tower 第五次裁决：B 装载有限网格扫描，时间盒 2 轮）——两条路线均已试并排除

| 实验 | 内容 | 结果 | 结论 |
|---|---|---|---|
| EXP2 | V 走 **dim-major 落 L1** + 已验证的非转置装载形态（等价于"BMM2-ready 布局"方案，也就是 tower 提到的 AIV 预转置回退方案的等价物） | 三档相对误差 **9.7e2 / 4.0e3 / 4.9e3** | **排除**：比"token-major + 3D 转置"（0.26/0.96/1.6e4）更差 ⇒ **"预先把 V 变成 BMM2-ready 布局"这条回退路线本身不成立**，不是转置装载的问题 |
| EXP3 | token-major + 3D 转置，改由指令内 **`mStartPt/kStartPt`** 选 dim 半块 / token 半块（`l1ElemOff=0`，n×8 / ks×8） | **挂死**（无 dump） | 该参数组合非法；3D 路径的源选择语义与我的推断不符 |

**新增的确定性结论**（对后续定位最有用）：
1. **3D `LoadData3D` 的源偏移（L0B/`l0Dst` 之外那个 `l1ElemOff`）确实"传了不生效"**——判据：两个 N 半块（nh=0/1）
   拿到**完全相同**的数据（同一个 mmad 结果被 dump 两次）。这与本轮修掉的"目的偏移缺失"是同一类
   *参数静默不生效* 模式，值得写进 docs/05 作为通用经验。
2. **"把 V 预转置成 BMM2-ready 布局"不等于解决**：EXP2（host 侧 dim-major = 完美预转置）比 3D 转置更差，
   且 BMM2 的 B 形态与 BMM1 同构、参数照抄已验证组合，仍然不对 ⇒ 说明**我的"L0B 装载几何模型"本身有错**
   （不是"V^T 缺一个转置"）。下一步应先做一个**最小 2-fractal 用例**（N=16、K=16）把 L0B 装载几何本身标定出来，
   再回到 256 尺寸；继续在 256 尺寸上扫参数是低效的。
3. 时间盒（2 轮）已用尽且两条路线都被排除，按 tower 规则停止扫描；当前交付保持"token-major + 3D 转置"
   （实测：首次不匹配的相对误差 0.26/0.96/1.6e4，max 为 4.17e4/7.66e2/1.68e4；out 已非全 0）。

### 第六轮（tower 批准几何标定路线，时间盒 ≤3 轮）：新增 `m10_l0probe` 几何标定探针

**方法**（tower 裁决 + docs/05 §6.1 判据）：AIC-only 探针，源用**值唯一编码**的构造输入
（`src[i][j] = i*1000+j`，i=行/N 轴、j=列/K 轴，bf16 精确），Nd2Nz 进 L1；A 用 m0 已实证的 16×16
单位阵装载；`mmad k=16, A=I` ⇒ `L0C[m][n] = L0B[n][m]` ⇒ **L0C 直接映射出 L0B 的 (n,k) 内容**，
host 反查唯一值即可知道每个元素来自源的哪个 (i,j)，从而反推 `(mStep,kStep,srcStride,dstStride,ifTranspose)`
各控制哪个轴。一次 launch 跑 10 个 conf（基线 + 各字段 ±/翻转 + 源偏移）。运行：`./m10_l0probe`。

**两轮实测结论**：
1. ⚠️ **参数有效果但结果不稳定**：第 1 次运行 conf0（"BMM1 形态" (2,1,2,2,T0)）读出 `780640 / -2147483648`
   等**非源值**（未初始化内存），而 conf1（mStep=1）读出**结构化的源数据**（1000/1004/1008… 周期性重复）；
   第 2 次运行**所有 conf 读出完全相同的数据** ⇒ 结果不可复现。
2. **探针方法学缺陷（下轮必须先修）**：10 个 conf 共用一次 launch 且**未在 conf 之间重置 L0B**——
   某个非法/无效组合不写 L0B 时，后续 conf 会沿用前一个 conf 的 L0B 内容，于是"所有 conf 结果相同"。
   ⇒ 正确做法：**每次 launch 只跑 1 个 conf**（或 conf 之间显式清零 L0B），否则读数不可解释。
3. **对 BMM1 解释的旁证**：本尺寸（N=32,K=32）下"BMM1 形态"(2,1,2,2) 读出的是**未初始化内存**，
   而 mStep=1 反而读出真实源数据 ⇒ **"BMM1 的 (16,8,16,16) 组合只是在其 N=256/K=128 形状下恰好自洽"**
   这一怀疑得到支持（同一套"几何模型"放到 2-fractal 尺寸就不成立）。**警示全仓库的 L0 装载代码：
   务必在本形状下做"传两个值看结果是否变"的判据，不要照抄其他形状的参数组合。**
4. 时间盒（≤3 轮）已用尽且探针需先修方法学缺陷，故按 tower 规则收口；「L0 装载几何标定 + V^T 装载」
   建议列为**独立 mission**（下一步就是"每 launch 一个 conf"重跑本探针，should 一轮出表）。

### 剩余待办：V 的转置装载（§7 保留项）

修完 A 侧后 BMM2 的残余错误全部指向 V^T 装载：
- 两个 N 半块（nh=0/1）拿到**完全相同**的数据 ⇒ dim 半块的源偏移未生效；
- 数值与"只取 dims 0-127"或"只取 dims 128-255"的参考都不符 ⇒ 3D `enTranspose` 的取数本身也不对。
建议：既然 A 侧现在可靠，可用"已知 A=P（GM 里可核对）+ 扫描 B 装载"直接对 `P·V` 逐元素比对，
不必再猜；2D `ifTranspose=true` 路线此前 6 组参数全挂死，需配合 `mStartPosition/kStartPosition` 语义重新推。

### 原（前三轮的）描述保留作背景

### 本轮（tower 裁决后）做过的四组实验与结论

| # | 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | BMM2 改 K 拆 2×128（不再用单次 mmad k=256，与 BMM1 同构） | PV 仍 0 | **排除** "k=256 单次 mmad 不合法" |
| 2 | V 改 dim-major `[N2][HD][seqPad]` + 非转置装载（(mStep,kStep,src,dst)=(N/16,K/16,16,N/16)，与 BMM1 的 B 侧**同一形态**） | PV 仍 0 | **排除** "转置装载语义错" 是**唯一**原因 |
| 3 | 逐字段核对锚点：m0（已验证）里唯一一处 `ifTranspose=true` 是 16×16 单 fractal（全 1 参数，无法外推）；m11/m3/m1 全部 `ifTranspose=false` | — | 仓库内没有"多 fractal 转置装载"的已验证参数可抄，故改走实验 2 的规避路线 |
| 4 | **消元实验**：把 BMM2 的 B 操作数指向**已验证的 K tile**（同一装载形态与参数） | PV **仍 0** | **排除整个 B 侧装载**：B 已换成 BMM1 里被证明正确的那个 operand，仍出 0 |

⇒ 故障必在 **A 侧（P → L1 → L0A）** 或 **BMM2 的 L0C slot / FIXP 路径**（与 B 侧无关）。
注：V 置单位阵（`M10_SYNTH=5`）时 PV 应等于 P，同样为 0，与上述一致。

### 已排除的假设清单（供续跑者直接跳过）

1. ~~单次 mmad `k=256` 不合法~~ —— 拆成 2×128 后无变化。
2. ~~V^T 转置装载（3D enTranspose）参数错~~ —— 换成非转置的已验证形态后无变化，故它不是唯一原因。
3. ~~V 的 L1 布局（dim-major vs token-major）~~ —— 两种布局结果相同。
4. ~~B 操作数的装载方式~~ —— 直接换成已验证的 K tile 仍为 0。
5. ~~P 的数值~~ —— `dbgp` 与参考 E 逐元素一致（§5）。
6. ~~P 的 L1 布局与装载参数~~ —— 与 Q 的 NZ 布局/装载参数逐字段同构（`(col/16)*256 + row*16 + col%16` 形式、`(1,16,1,1)`），而 Q 侧 BMM1 逐元素正确 —— 但**"布局与参数同构"不等于"P 确实写进了 L1"**，这是当前第一嫌疑。

### 建议的下一步（各 1 次改动 + 1 次运行，按信息量排序）

1. **A 侧判据实验**：把 BMM2 的 A 操作数指向已验证的 **Q tile**（`LoadL0_2D(l0A, l1Q, 0, 1, 16, 1, 1)`），B 仍用 K tile，
   则 `dbgc2` 应等于 `Q·K^T[:, :128]`（可用设备自己的 Q/K 在 numpy 算）。若出数 ⇒ A=Q 路径与 BMM2 的
   slot/FIXP 都正常 ⇒ **问题就是 P 的 L1 写入**（AIV 的 `DataCopy(L1, UB, DataCopyParams{16,1,0,15})` 是否真的落盘、
   或 `Acq<PIPE_MTE3>(B_PC)` 的交接是否成立）；若仍为 0 ⇒ 查 BMM2 的 L0C slot（`L0CId(t*3+1+nh)`）与
   `FixpToGmDbg(nSize=128)` 的 dump 参数（可先用 nSize=256 + dst 偏移 0 试）。
2. **P 落盘旁证**：让 AIV 在写 L1 之前把 P 从 UB dump 到 GM（已有 `dbgp`），再让 **AIC** 在 `Acq<PIPE_FIX>` 之后
   用 Fixpipe 把 L0A 的内容... 不可行（L0 不可读）⇒ 改用"把 P 经 GM 中转"的方案（AIV→GM→AIC→L1，多一次搬运+一次 mode2 同步），
   这条能同时证伪/证实"UB→L1 DataCopy"。

### 原（已被实验 1-4 否定的）描述，保留作背景
证据（**注**：本条为第一轮的观察，其中"PV 恒 0"的成因已在第四轮修正为 P 的 L0A 目的偏移缺失；
修正后 `dbgc2` 有数但数值仍不对 ⇒ V^T 装载仍是待办）：① 当时 `dbgc2`（BMM2 的 L0C）全 0；
② 当时 `M10_SYNTH=5`（K=单位阵且 **V=单位阵**）PV 仍全 0 ⇒ 当时不是"P 数值错"而是通路没出数；
③ P 的 L1→L0A 路径与 Q 完全同构（同一 `LoadL0_2D` 参数形态、同一 NZ C0=16 布局），而 Q 侧 BMM1
逐元素正确 ⇒ A 侧嫌疑低，**B 侧 V^T 装载是主嫌疑**。
已试两条 B 侧路线：
- 3D 转置装载（`LoadData3DParamsV2` + `enTranspose=1`，现默认）：不挂死但出 0；
- 2D `ifTranspose=true`（donor `bsa_copy_l1_to_l0b_a5.hpp` 的 zN→nZ 语义）：试了 6 组
  (mStep,kStep,src,dst,T) = (16,8,16,8,T)/(16,8,16,16,T)/(8,16,16,16,T)/(16,8,8,8,T)/(16,16,16,16,T)/(8,8,16,8,T)
  **全部挂死**（`M10_L0CFG` 第 11-15 项可覆盖，见 §2）。
建议下一步：先按 donor 的 tla stride 定义逐字段**推演**出正确 (mStep,kStep,srcStride,dstStride)
（不要继续盲扫），或用 `M10_SYNTH=5` 把 B 侧换成已知 1-fractal 形状（K=16）的最小 mmad 探针。
旁证：`mp.k=256` 的单次 mmad 是否合法也值得一并核实（BMM1 用的是 k=128）。

**卡点 B（已由 RegBase VF 消除，保留记录）—— 经典 API softmax 的 E/sum 异常**
换 VF 前 `wss` 高 1.086~1.111×（`wsm` 却完全正确）、P 出现 >1 值，且 P 的 bf16 落地呈"偶位 0"交错。
换 VF 后三项全部精确 ⇒ **根因确为经典 API 的位宽转换/标量-向量交接语义**（docs/05 §6.1 与 #22），
与 tower 的裁决一致：该问题随范式切换消失，无需再查。

**卡点 C（历史记录）—— 经典 API softmax 的 E（exp）在部分 lane 未正确扣行 max**
证据链：① `wsm`（行 max）完全正确 → m 对、E 的理论上界为 1.0；
② `wss` 却高 1.086~1.111×（已排除 bf16 精度与 2^x 底数混淆，见 §5）；
③ `dbgp` 里出现 **>1 的 P 值**（1.398 / 1.484 / 1.133）——若 E 的上界是 1，出现 >1 只可能是
某些 lane 的 `exp(T − m)` 用了偏小的 m。
⇒ 怀疑 `Duplicate(Mb[r*S2T], mNew, S2T)`（每行广播）或 `Sub(Ts, Ts, Mb, 2048)` 在部分 lane 上
没拿到正确行 max。行 max 折半归约与行 sum 折半归约结构相同、max 侧完全正确，
所以**不是归约结构问题**（而且 max 幂等，sum 不幂等——这也解释了为什么只有 sum 暴露偏差）。
复现（**已修复，此段是修复前的排查记录**）：该 `wss` 偏差的根因是"列 mask 被加到每个 tile"（§6.1 表第 2 行，
修后 seq=300 split0 的 4 个行 sum 与参考**逐位相等**）。当前归档文件名为
`data/m10_case_s{256,300,4096}_wss.bin`（旧的 `m10_val_s256_wss.bin` 已不存在，勿再引用）；
比对方法：`./m10_attn_decode` 生成 dump 后跑 `check_ref.py`（其对 `wss` 的报告见 §5 分段实测表），
或用 `M10_SYNTH=1` 造已知输入把行 max 逐 lane dump 出来定位。

**卡点 2 —— P cast（fp32→bf16 `Cast`）输出 stride 可疑**
`dbgp` 呈"偶位 0、奇位有效"的规则交错（每行 256 个里 128 个非 0，行和≈127.5）。
这与 docs/05 §6.1 记录的 "fp32→bf16 需先 interleave、even/odd 各一条输出、再 select+mask merge"
以及 M10 "连续两个 Cast 后者错乱" 的既有 quirk 吻合：疑似 bf16 结果按 4B 槽交错落盘，
若如此则真正写进 L1 的 P 有一半是脏的 → BMM2 必然错。
注：`dbgp` 本身也受卡点 3（读回缺同步）影响，**先按卡点 3 补同步再判定**。
修复方向：改用 `Cast` 的 pack b16 store 模式，或按文档做 interleave + 两条 cast + select/merge；
本仓库已合并模块（m4/m5/m12）**没有**任何经典 API `Cast(` 用法——它们走 RegBase VF
（`__VEC_SCOPE__`，store 侧转换）或 Fixpipe `F322BF16` 落 bf16，可作规避路线参考。

**卡点 D —— 调试 dump 的跨 pipe 同步（已修）**
`dbgs`（贴 FIXP 直写的 UB）现在先经 V 侧 `CopySTileVf` 搬进 scratch，再走 V→MTE3 的阻塞释放
BufferID（`B_DBG`）落 GM；`dbgp` 放在 MTE3 已持有 `B_PC` 之后。修复后 `dbgs` 与参考 maxErr 1.9e-6，
dump 已可用于定位。`wsm/wss` 由 V 侧写、经 `B_ACC` 阻塞释放交给 MTE3，同样已成立。

**卡点 E —— FD combine 尚未验证**
combine 已能算出（reviewer 复核 den 通路正确），但受 BMM2 数值错误传导，`out` 三档仍 FAIL。BMM2 修好后即可用现有
`wsacc/wsm/wss` dump 逐 split 对参考比（参考分子 = Σ_k E[s,k]·V[k,d]，用设备自己的 bf16 P）。

## 7. 已知限制

- 只做 **decode m=1** dense FA core；无 causal mask、无 QSA/indexer 稀疏路径（docs/11 §0 的范围裁决）。
- 输入布局为 **连续 KV**（非 paged）；paged/block_table 契约未做（docs/11 §3.1）。
- **V 的输入布局为 token-major**（`[N2][seqPad][HD]`，与模型 KV cache 一致；tower 决策后已从早期的
  dim-major 试验回退，代码/host/§1/§6 一致）。BMM2 的 B 操作数需要 [N=dim,K=token]，即**对 V 的转置装载**——
  该装载目前仍不正确（§6 卡点 A），待 M27 的 `LoadData2DParamsV2`/3D `enTranspose` 几何映射表出来后按其修。
- 只支持 `N1=24 / N2=2 / g=12 / head_dim=256 / bf16`；核数假定 28 AIC（`numBlocks != 28` 直接报错退出），
  满配 32 AIC 需按 `GetCoreNumAic()` 重新推导 split 数。
- **P̃ 在中途被量化到 bf16**（donor 同款设计 ⇒ 本核规格；口径的外部 pin = donor 的 `Max(新,旧)`/`ExpSub(x,新max)`/
  `ExpSub(旧max,新max)`，见 `check_ref.py` 文件头）：设备 exp（**官方规格 ≤1 ulp**，§6.6；相关区间实测 1.1~1.4 fp32 ulp）
  可能把某个 P̃ 推过 bf16 量化边界（概率 ~1e-4/元素；三档实测 1/131072），该离散事件在深度相消输出元素上可达数个输出 ulp ——
  这是 seq=4096 唯一那 1 个越界元素的成因（§6.2 末段完全归因），**不在 fp32 累加误差界内**（§6.6）；
  该事件本身由**判据 C** 覆盖（逐元素 1.0·ulp 的格点允许差 + 位差计数，§9）。
- **T4"确定性"未达成**（reviewer r3 的 hold 理由）：一次三档连跑里只有 `s300_out.bin` 数值不同，
  **0/41 次复现、证据已不可核验** ⇒ 按纪律记为**未隔离观测**（§8.1），**不作为时序/硬件结论、不进 docs**；
  §8.2 给出了可交给 probe mission 的问题描述（含隔离步骤与预期读数形态）。
- **mmad 的 k 方向/吸收性/最坏情况仍未标定（明确 blocker）**：`m10_mmadprobe` 的 507015 已修（§6.3），
  但本轮用**有辨别力**的分布 + 块对角 A 诊断证实：**该探针的 B 操作数映射是错的**
  （设备实际用的是"源列 {32..255} ∪ {128..159}"，源列 0..31 根本没进 mmad；三条独立分布的读数逐位互证，
  换 `PipeBarrier<PIPE_ALL>` 不变 ⇒ 不是可见性问题）。故**该探针给不出标定**；只有"真实数据分布占保守界
  0.13~0.15%"这一条实测（§6.3）。两条出路（修探针 / 改在主核已验证的 BMM2 路径上用对抗性 V 标定）写在 §6.3 末。
- **`m10_expprobe` 只在前 8KB（三个相关区间）可用**：8KB 之后的 3 组仍返回垃圾（`maxRel=1.0`/`inf`），
  已按 §6.4 只引用可复现部分；它现在只是**交叉见证**（ε 的 Exp 项已由**官方 Reg 规格**给出，§6.6）。
- ~~**ε 的 `Exp` 项不是官方规格**~~ —— **M61 已改正**：规格已找到并引用
  （`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md` 表 1 +
  `.../reg_vector_compute/basic_arithmetic/Exp.md`），§6.6 给出三段推导（规格出处 / `1 ulp → 2·2^-24` 的最坏情形换算 /
  VL·输入域条件），`EXP_ULP` 数值不变但**出处从观测变成界**；同时补上旧式漏掉的"重标定因子那侧"的次数（`+EXP_ULP·nTile`）。
  此前"仓库内无该规格"的结论只检索了 `docs/`、漏了官方 API 文档树，已在 §6.4 撤回。
- **Combine 的"标量读 VF 写的 UB"方向用的 primitive 是 `PipeBarrier<PIPE_V>`**，不是 docs/05 #22 验证过的那个
  （#22 是"标量写"方向）；10 次运行无异常，但排不掉 ~1/30 量级事件类 ⇒ 概率性未标定项（§6.7），
  并且它是 §8.1 那条未隔离观测的首要嫌疑路径。
- `M10_CASES` 已加合法性校验（必须 1..4096，否则打印原因并 `exit 1`）：`0` 会在设备侧触发 `CeilDiv(0,0)` 除零、
  `>4096` 会越界读 K/V（P2-6）。
- `KV L1 2×128KB` 单 buffer 轮替（非 donor 的 4×64KB）；P 3-buffer 已按 donor。
- 未做性能细扣（无跨 op 预取、无 L2 hint），本阶段只求正确性。
- 调试期间为定位保留了若干 `printf` 与 4 个 dump buffer（`dbgc`=BMM1 L0C、`dbgc2`=BMM2 L0C、
  `dbgs`=AIV 侧 S、`dbgp`=P），定位完成后应移除以减少 MTE3 开销。
- AIV softmax 已全部走 RegBase VF（无经典 API `Cast`、无标量参与行状态）——这是本核的既定范式，
  后续改动请保持（经典 API 版本已删除，见 §6 卡点 B）。
- **存量的 memory-based 用法（M61 核查，非本次引入）**：`.asc` 里没有 `AscendC::` 经典向量计算 API
  （`grep -nE "AscendC::(Cast|Duplicate|Exp|Add|Sub|Mul|Muls|Max|Min|Reduce|Gather|Sort|MrgSort|Select|Compare)" m10_attn_decode.asc` = **0 命中**）；
  存量的三类是：① UB 状态初始化 `Duplicate`（5 处：`m=MASKV` / `sum=0` / `acc=0` / `mask=0` / `num=0`，性质 = 常量填充、不在计算链上）；
  ② 搬运/矩阵类（`DataCopy`/`Fixpipe`/`Nd2Nz`/`Mmad`，docs/05 §6.1 规则 ⓑ 豁免）；③ **无** `Sort32`/`MrgSort`
  （本核不排序 ⇒ 例外未被用到）。①是否计入"计算路径"由 docs/05 §6.1 的四条判据定；本 mission 不改 kernel、只如实记录。
- **host 墙钟不作为证据**：本 README 与判据里没有任何 host 计时**读数**（判据只报判定项/报告项/guard 的计数与数值）。
  可复算：`grep -cE "[0-9]+ ?(ms|us)\b" README.md` = **0**（"数字 + ms/us"形式的计时值）；
  `grep -cE "time\.|perf_counter|clock" check_ref.py` = **0**（判据脚本里没有计时调用）。
  **r1 评审 P2-3a 的订正**：此前这一条写的是「宽式 `grep -nE "[0-9]+ ?ms|[0-9]+ ?us|耗时|计时|墙钟|elapsed"` = 0 命中」——
  实测该宽式的命中**正是本段自身的字样**（它含"计时/墙钟"），即它数到了自己 ⇒「0 命中」这句不成立、已改为上两句；
  结论（页面里没有计时读数）不变。

### 7.1 M61：正式否掉「主判据改成以设备的 P̃ 为输入」的提议（**不采纳**）

原提议（§6.6 早前列为"出路 ①"）：*"参考口径改为『以设备的 P̃ 为输入、只对 sum/combine 段独立复算』
（这样 ≤1 ulp 就是关于本核自有逻辑的严格命题，P̃ 舍入属设计规格、由 gp 逐位对拍来保证）"*。

**裁定：不采纳。** 理由（按 M59 的判据代号）：

1. 那等于把 **S1（设备产物入参）写进主判据**：主判据的被判量会从"整条 attention"缩到"sum/combine 段"，
   P̃ 的舍入方向与其中每一次量化事件从此**不可咬**（两侧同一份设备 P̃）。
2. **本档就有反例证明它有多危险**：4096 那唯一的越界元素，恰好落在**唯一一处**设备 P̃ 与 host 参考 P̃
   相差 1 个 bf16 格点的元素上（判据 C 独立量出：`gp[unit10][slot0][row11][col252]`，界占用 0.9965）。
   主判据一旦改吃设备 P̃，这个 FAIL 会被**吃掉**变成 PASS（`check_partials.py` 的注入实验正是这条闭链）——
   即该提议会把"我们唯一发现的那一处 P̃ 口径分歧"从判定里**删掉**，而不是解决它。
3. **正确落法是"分开"而不是"替换"**：保留既有主链（`out` 由 host 从 q/k/v 重算，判据 A），
   另立**判据 C/D**（P̃ 自身的独立来源）与**判据 E**（独立 P 的 P·V）。于是：① 主判据仍咬全链；
   ② "给定 P 下 BMM2"这一隔离命题由判据 F 承担（已标注 S1 定位）；③ P̃ 口径本身由判据 C 逐元素咬，
   且它咬出的那 1 处已在 §6.2 末段完整归因、在 §9 的咬合力表里写明"咬得住/咬不住"。
4. 若将来确实要一条"只用核自有逻辑"的隔离判据，**可以新增、不得替换**（同一份数据上两条都跑），
   并且新增那条必须在 README 声明它**不覆盖** P̃ 口径（否则等于把 S1 制度化）。

## 8. 未隔离的观测（不构成结论；**不得进 docs**）

### 8.1 上一轮报为"时序相关的数值不稳定"的那一次运行 —— 降级为未隔离观测

**观测（一次，2026-09-26）**：一次三档连跑里 `s256`/`s4096` 的 `out.bin` 与归档**逐位相同**，
`s300_out.bin` 不同（按本核判定：常规元素越界 1011、边界元素 13、max 366 ulp、maxRel 6.0）。

**复现状态：0 次复现 / 41 次尝试**（reviewer r3：22 次单跑 seq=300 + 6 次三连跑；我：3 次单跑 + 10 次三连跑）
⇒ 观测频率 ~1/41，**未隔离**。按全项目纪律（"硬件/文档不一致"类结论默认按**未隔离观测**处理，先隔离复现再谈入库），
本节**只作线索**，且 **P1 要求的 T4"确定性"一项因此判为未达成** —— 本 mission 不宣称验收完成。

**我上一轮的三处不准确表述，逐条更正（reviewer r3 指出，全部成立）**：
1. **"trace 逐条相同"是错的**。我当时只比了**结构**（行数 1161、每 AIV 的 wait/got/done、combine 的 nsplit），
   没比 `combine … den=` 的**数值**。重新逐行 diff 后确认 **72 行 den 里有 4 行不同**，而且正是
   **每个 combine AIV 处理的第一行**（与 reviewer 的摘录一致）：

   | combine AIV (n2,half) | 首行 | 正常 den | 坏 run den |
   |---|---|---|---|
   | 0 (0,0) | row 0 | 155.095001 | **498.221527** |
   | 1 (1,0) | row 0 | 184.630371 | **236.941483** |
   | 2 (0,1) | row 8 | 164.643097 | **23.494633** |
   | 3 (1,1) | row 8 | 174.325974 | **25.034525** |

2. **坏 dump 没有归档**：`s300_out.bin` 已被随后的运行覆盖，错值**无法再检视** ⇒ 我上一轮建议的
   "把错行与 256 档逐行比"在仓库里做不到（该建议撤回）。**这是流程错误：观测当次没有留证。**
3. **贴出的 `sha256sum -c` 输出读起来像"所有文件都 unreadable"**：那段输出**不是**在本次 dump 目录里产生的
   （是并发进程的 `sha256sum -c` 混进了终端输出）；我能确认的是同一次坏运行的 dump 目录里
   `sha256sum -c data/dump_sha256.txt` **只报 `m10_case_s300_out.bin: FAILED` + "1 computed checksum did NOT match"**。
   无论哪种解释，**它都不该被当作输入一致性证据使用** —— 该表述已撤回。
4. **坏日志的 case 顺序**：主机侧 `[M10] case seq=… dumped` 标记在坏日志里是 **256 → 300 → 4096**（= 文档化默认
   `casesAll`）；"256→4096→300" 是**设备 printf 惰性 flush** 的假象（300 档的设备行出现在 4096 的 host 标记之后；
   用最近的 `AIC … enter tiles=` 可判定这些 den 属于 300 档，`tiles=2`）。**这条不算异常**，但当时没记录命令行/顺序，
   属证据记录不完整（本次已在 §8.2 里定为流程要求）。

**这条观测里唯一有用的部分（reviewer 从坏日志算出、我复核了数据）**：4 个坏 den 中，
两条 half-1 行**恰好等于尾块（split1）贡献单独一项**（`w1·s1 = 0.8809×26.672 = 23.494`、`0.9345×26.791 = 25.035`
⇒ split0 的项消失），两条 half-0 行**超出正确 partials 能产生的上界**（`w≤1` 时 n2=0/row0 ≤161.4、n2=1/row0 ≤188.8，
实测 498.2 / 236.9）⇒ **若成立，是数据完整性级的错误中间量**（不是 ulp 抖动），线索指向 **FD combine 的 ws 读 / 首行路径**。
但按纪律这仍是**未隔离观测**：本仓已有五次同类结论以"我们选错用法"收场，且 §6.7 已记 combine 的"标量读 VF 写的 UB"
用的 primitive 并非 docs/05 #22 验证过的那个。**⇒ 不作为时序/硬件结论、不写进 docs。**

### 8.2 可交给 probe mission 的问题描述（本轮不做，请 tower 决定是否开 mission）

**问题**：`Combine` 中**每个 combine AIV 处理的第一个行**读到的每-split 部分和，是否会偶发"缺失/串到超出上界"的可见性
或初始化问题？（线索 = §8.1 的 4 个 den：half-1 行像 split0 项消失，half-0 行像读了不可产生的数据。）

**要验什么**（按信息量排序）：
1. **首行特殊性**：把 combine 的行循环起点错开（先写一行 dummy、或从 row 1 起再回来补 row 0），看坏值是否只跟
   "第一个被处理的行"走。若跟着走 ⇒ 是**迭代起点/同步状态**问题，与具体数据无关。
2. **分段定位**：复现后先**只读** `wsm/wss`（`wsacc` 同理）——若首行的 partials 本身就是错的，问题在 AIV 的 ws 写/释放
   （`Rls<PIPE_V>(B_ACC)` → MTE3 接手）；若 partials 正确而 `den` 错，问题在 combine 的读路径
   （`Acq<PIPE_MTE2>(B_CIN)` → `DataCopy(mM/mS)` → `Rls` → `Acq<PIPE_V>(B_CIN)`，以及 `B_ACCIN`）与 `CombineWeightsVf`。
3. **频率上界**：固定输入（`M10_SYNTH`）+ 同一 binary，在同一进程内**连跑 N 次** seq=300（把首行信号放大 N 倍），
   统计 `combine den=` 行的差异率 ⇒ 给出可引用的频率上界（目标：0/1000 或给出定量率）。

**怎么隔离（流程要求，直接针对 §8.1 的教训）**：
- 每次运行都**归档**：完整设备 trace + 该次 `out.bin`/`ws*.bin` + **命令行与环境变量**（含 case 顺序）；
- 单档运行（`M10_CASES=300`）→ 只跑 nsplit=2 + 尾 tile（44/256 有效列）这条唯一出现该现象的路径；
- 记录"是否首行/第几个 AIV/是否 half=0 或 1"，以便区分 `m/sum` 读与 `acc` 读。

**预期读数（可判定的结论形态）**：
- 0/N 次 + 无法定向复现 ⇒ 只能归档为"**未隔离偶发**"，并写明 N 与观测条件；
- 复现且 **partials 正确** ⇒ 定位到 combine 读路径，按"跨 pipe 交接必须用 BufferID"的既有结论修（本模块已有先例）；
- 复现且 **partials 就错** ⇒ 定位到 AIV 的 ws 写/释放侧（`B_ACC`/`B_CIN` 握手）。

## 9. 判据咬合力表（逐条写清"咬得住什么 / 咬不住什么"；自指形态按 M59 的 S1/S2/S3/S4 代号标注）

**M59 的代号**（本表只用它的判决语言）：**S1** 设备产物入参（参考的输入取自设备在被判量自身或其下游产出的字节）；
**S2** 无外部权威的规则镜像（规则由设备实现/同一作者对同一规范的同一份读法转写而来，**无任何仓外工件以 `文件:符号` 钉住**）；
**S3** 反演自消；**S4** 同一实现算两遍。不算自指的：**N1** 输入隔离（只用算件的声明输入）、**N2** 外部权威 pin、
**N3** 异实现/异数值域交叉、**N4** 结构性。
**M59 的分水岭**：S1/S2 只在"被共享的那个中间量自身**没有**被一条规则外部钉住的判据覆盖"时才有害。

| # | 判据（代码符号） | 输入来源 | 规则出处 | **咬得住** | **咬不住**（如实写） |
|---|---|---|---|---|---|
| A | `out` 边界感知 ulp（`check_ref.py::check`） | **N1**：host dump 的 q/k/v；参考 = fp64 numpy 全链重算 | **N2**：P̃ 口径 pin 在 donor 的 `Max(新,旧)`/`ExpSub`（`check_ref.py` 文件头）；Exp 项 pin 在官方 Reg 精度表；判据形式 pin 在 docs/17 §1.1 | 全链数值：BMM1 / FIXP / softmax / FD 归约 / BMM2 / combine 的**任何**非 ulp 级错误；`out` 的索引与槽位错位；恒零/常数输出（配合 B） | ① 落在 bf16 输出网格**同一侧**的 ≤1 ulp 差异（按 §1.1，该命题在深度相消元素上不可推导，δ 见 §6.6）；② **单个 P̃ 的 bf16 翻转事件**（离散、在 `noise_ub` 量纲之外）—— 该事件由判据 C 覆盖 |
| B | `out` 非全零 / 非常数 / 每行 256 列非零（同 A 的函数） | 同 A | **N4**（结构性） | 恒零/常数输出、整行或整列位置丢失 | 不提供任何精度/正确性证据（T4 门槛项） |
| C | **P̃ 全数组逐元素**（`check_ref.py::check_p_full` + `ref_p_all` + `p_judge`） | **N1**：只用 host 的 q/k，**不碰任何设备字节** | **N2（部分）+ S2 残余**：逐 tile 运行 max / 每 tile 取 exp / rescale 三件事都 pin 在 donor（`vf_basic_block_aligned128_update.h` 的 `Max(...)`/`ExpSub(...)`、`vf_flashupdate_new.h::FlashUpdateBasicVF`）；**但"bf16 用 RNE"这条 donor 未逐字钉住**（该文件 bf16 分支用 `castTraitZero` = `RoundMode::UNKNOWN`，只有 fp8 分支明写 `CAST_RINT`）⇒ 记 S2 残余，靠本判据对**设备自己的 P̃**逐元素复核兜住 | P̃ 的舍入口径（用 running max 而非 split/global max —— `globalmax` 负向对照实测越界 1983/86738）、P̃ 的 unit/slot/行/列**位置**、设备 Exp 的**系统偏差 >1 个 bf16 格点**、P̃ 量化边界上的**批量**翻转 | ① 单个 1-格点差异（判据按 §1.1 的"参考已量化"条款**明写允许 1.0·ulp**，本处 1 ulp = 相邻格点间距 `2^(e-7)`，约定见下；实测 1/131072）；② 设备 Exp 的**位级**行为（只能到官方 1 ulp + 派生的翻转率上界）；③ **seq=256 档没有任何 max 口径可判**（全序列只有 1 个 tile ⇒ 各口径同解，负向对照实测 0 并如实标注） |
| D | **P̃ 位置 / 掩码结构性**（同 C 的函数） | **N1**（期望的 (unit,slot,行,列) 集合由 host 侧调度推导） | **N2**：列 mask 契约 = `m10_attn_decode.asc` 的 host `GenMaskCol` + kernel "只有尾 tile 加 mask" | unit 索引空间错位（AIV 的 `bid` vs AIC 的 `bid>>1`，历史上真出现过）、AIV≥28 越界写坏相邻 unit、尾 tile 无效列没被清零 | 不查数值（数值由 C 查）；也不查 `ws*` 的 split 归并（那部分目前只由 A 间接覆盖） |
| E | **P·V，独立 P**（`check_ref.py::check_pv_ref` + `pv_tol_independent`） | **N1**：q/k/v 全来自 host；P̃ 由判据 C 的重算给出 | **N2**：容差两项都是**推导**（mmad `k·u·Σ\|AB\|` + 判据 C 允许差经 V 的传播） | P 的**约定级/布局级**错误、V^T 的转置与布局（BMM2 的 B 侧）、**"AIC 实际消费的 P̃ 就是 `gp` 里那份"**（C 只查 AIV 写出的 `gp`） | 比 ~`0.39%·Σ\|P̃V\|` 更细的差异（分辨率由"P̃ 的 1 格点允许差"决定）；"更细"那部分由 C 在**P̃ 元素级**承担 |
| F | **P·V，设备 P**（`check_ref.py::check_pv`，M27 `m16_pv` 口径） | **S1**：`p = gp[unit][0]` 是**设备产出的 P̃** | **N2**：容差 = `k·u·Σ\|AB\|`（推导；旧写的 5e-3「bf16 网格」是无来源经验值，已换） | 给定 P 下 BMM2 的布局/转置 + "AIC 消费的 P 与 `gp` 同一份" | **P 本身**（两侧同一份设备 P̃）。按 M59 的分水岭：本核现在**有** C/D 覆盖 P̃ ⇒ F 记为 **S1-条件（条件已满足）**，不再是裸露的 S1 |
| — | 参考的 P̃ 口径（建模） | **N1**（只用 host 输入） | donor pin（同 C）；**旧版曾被重新拟合到设备**（README 自陈"最初写 `bf16(exp(S−split_max))`，后改逐 tile 运行 max"）⇒ 当时是 S2；M61 把出处改成 donor 引用 | 见 C | 见 C；"bf16 用 RNE"这一条仍是 S2 残余（donor 未逐字钉住） |

**覆盖范围（"统计工具必须交代覆盖范围"）**：判据 C 覆盖 `gp` 全部 344064 个槽位 = 每档**预期的** (unit,slot) 全查
（8192 / 9600 / 131072 个元素，**无抽样**）+ 其余未用槽位按 D 全查（必须恒 0）；判据 E/F 覆盖**每个 unit 的 tile0**
（8192 / 16384 / 65536 个元素；`dbgc2` 的 dump 条件就是 `t == 0`）；判据 A/B 覆盖 `out` 全部 6144 个元素 ×3 档。
**同源计数** = `check_ref.py` 末尾的 `[check] … 判定项 N/M 通过 | 报告项 R 单列 | guard G` 行 —— README 引用的
判定项/报告项/guard 数字都出自它；`RESULT:` 行的三态见 §0。
**负向对照（一条，判据自身非空洞的证据）**：判据 C 把参考换成 `globalmax` 口径 ⇒ 越界 **1983**（300 档）/ **86738**（4096 档）
⇒ 判据**确实会 FAIL**；seq=256 档该对照为 0 并**如实标注原因**（全序列只有 1 个 tile ⇒ 两口径同解、不可判）。
另有**判别力对照**（每次运行自动跑，guard）：`Δ=0` ⇒ 0 越界；某元素 +1 个 bf16 格点 ⇒ 0（在判据明写的 1.0·ulp 允许差之内）；
+2 个格点 ⇒ 必须被抓（实测 1 处）。
**"1 ulp" 的数值约定（M61 r1 评审 P2-2；r2 改为自足表述）：本表与全文一律 = 相邻 bf16 格点间距 `2^(e-7)`**
（bf16 = 1 符号位 + 8 指数位 + **7 位显式尾数** ⇒ 一个存储步长）。三条依据：
① 两个**都已量化到同一 bf16 网格**的值，最大合法差就是**一个**格点间距；
② **测量事实锚点**：M36 的 `real_m1` 用例里设备 `0x3dc0` 与参考 `0x3dc1` 是**相邻格点**；
③ 取**半格**会把判据 C 那唯一 1 处**合法**翻转（`|Δ|=1.9531e-3` vs `tol=1.9601e-3`、界占用 0.9965）误判成 FAIL 1/131072。
**为什么 A 的"1 ulp 允许"不是放水**：A 对**常规元素**仍判 ≤1 ulp；只有参考落在"两网格点中点 ±noise_ub"的元素才走
"≤2 ulp + 绝对界"，且 δ 是推导出来的（§6.6）。**为什么 C 的"1 ulp 允许"不是放水**：两边都已在 bf16 格点上，
格点对格点的最大合法差就是 1.0·ulp（docs/17 §1.1 的"参考本身已量化"条款）；C 仍逐元素检查、仍报位差分布与位置。
