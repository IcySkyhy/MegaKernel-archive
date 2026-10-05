# 18 — 向量 API 审计：memory-based → `__VEC_SCOPE__` / register-based

> **出处**：本文由 **M44（agent-vecaudit，只读审计）** 产出，由 **M49（agent-docssweep）** 原样落库——内容与结论**未经改动**，仅在本段补出处说明。M44 的分支本身**已合并**入 main，但本文当时是 `wt-44` 工作树里的**未跟踪文件**（落库前 `git log --all -- docs/18-vector-api-audit.md` 为空 ⇒ 从未进入任何 commit）⇒ 落库前只存在于 `wt-44` 工作树。
> **行号锚点时效**：文内对 `docs/05` 的锚点**（凡本文所引者，含 `:30`、`:114`、`:115`、`:120`、`:123`、`:127`、`:128`、`:145`、`:154`）**以落库前的 `main` `55f5e18` 为准。M49 已按 tower 对本文 §9 的裁定重写 `docs/05` §6.1（四条规范 + 豁免清单）并把 `:30` 旧白名单收窄为 `Sort32`/`MrgSort`，条文内容与本文 §9/§10 的建议一致；`docs/05` 的行号因此已漂移，引用时以章节号为准。
> **原样保真说明**：本文第 **104** 行（§3.1 `m3_grouped_gemm` 明细表内；`wt-44` 原件中的第 **100** 行）含原件中就有的**未转义裸 `|`**（代码 span `` `…(c0|(c1<<4))` `` 内），该行 markdown 渲染会多拆一格单元格，**内容不丢**。为保持与 `wt-44` 原件的逐字节可比对，**M49 按 tower 裁定不做转义**。
>
> **性质**：M44 只读 survey（不产生任何代码变更；本文件是全仓唯一新增物）。
> **裁定依据**：用户原话 —— **「你应该都用VEC_SCOPE aka simd vector function，不应该使用memory base的API」**。
> 判读条文（tower，已广播）落在 `docs/05-megakernel-design.md` §6.1 表内（**main** `docs/05:115`；本次审计所在的 `wt-44` 树 HEAD 落后于 main，树内 `docs/05` 尚无该行，只有 `docs/05:114` 的旧行）。
> **扫描对象**：main 分支上已有的 16 个 kernel 目标 —— `m0/`、`m1_mxfp4_gemm/`、`m2_mxfp4_quant/`、`m3_grouped_gemm/`、`m4_gdn_recurrent/`、`m5_swiglu_quant/`、`m6_rmsnorm/`、`m7_router_topk/`、`m8_permute/`、`m9_gdn_prolog/`、`m11_bf16_gemm/`、`m12_rmsnorm_gated/`、`m13_moe_layer/`、`m14_gdn_layer/`、`m15_layer_loop/`（含其设备头）、`probe_sync_quirks/`。
> （mission 清单里的 `m16_*` 在 main 上不存在；main 上也没有 `m10_*`（M24 的 attention kernel 在 `wt-24` 的未合入分支里，本次只作参考、不作基线）。**——以上是 M44 审计当日的** main 状态；这两个工程此后均已合入，见下条。）
> **行号口径**：全部相对 `wt-44` 树内 main 内容（`m*/*.asc` 与同目录 README/evidence）；`docs/05` 的标准条文引 **main** 的行号并注明。
> **状态补记（M66，2026-09-26）**：本文 §1.3 与 §8.3 的「未合入 main 的 worktree」是 **M44 审计当时（`wt-44` 树）**的口径。这些工程此后**均已合入 main**：`m10_attn_decode` 随 `b680b0e`、`m16_load_geom` 随 `eeab864`、`m17_moe_real` 随 `95ff3f5`、`m18_gdn_prefill` 随 `e035b0f`、`m19_qsa_indexer` 随 `a208849`、`m20_hyperconn` 随 `ea4c818`、`m21_layer_ref` 随 `7c6ee12`、`m22_router512` 随 `2d20490`、`probe_vf_loop` 随 `7f627ce`。⇒「只读参考、不作为基线」是**审计当时的范围声明**，不是今天的合并状态；M49 的「原样保真」使本文内容保持不动，合并状态以本补记为准。

---

## 0. 摘要（先看这一节）

**在全仓 16 个目标的设备代码里，「计算类却走 memory-based」的真实存量比预期小得多，而且分布极不均匀。**

| 判定 | 模块 | 依据 |
|---|---|---|
| **生产级 (a) 存量，唯一一处** | `m3_grouped_gemm` | AIV 段 SwiGLU + 量化器整段经典：`Cast` `:450`、`Silu` `:451`、标量量化器 `QuantRow` `:485-519`（`GetValue`/`SetValue` 逐元素） |
| **探针级 (a)** | `m0/m0_bringup.asc` | 10 处经典 `Add`/`Adds`（`:144-150`、`:184`/`:186`、`:213`）——bring-up 探针，不是层 kernel |
| **(a) 存疑（控制路径）** | `m8_permute.asc` | `:157-159` 标量读 GM `counts` 求和，结果只喂标量分条控制 |
| **(d) 边界：计算类但确无 Reg 等价物** | `m7_router_topk`、`m13_moe_layer`、`probe_sync_quirks` | `Sort32`（`m7:352`、`m13:974`）、`MrgSort`×15（`m7:450`，经 `MergeTree`）、`Extract`（`m7:362`、`m13:975`）。**其中 `Extract` 经核实是可转换的**（见 §6.2），真正不可转的只有 `Sort32`/`MrgSort` |
| **标量胶水（非向量 API，但属计算）** | `m13_moe_layer.asc` | `IndexGenStage` `:1063-1123`：直方图 + 前缀和 + 数据相关 scatter + 标量 bf16 RNE 打包 |
| **已完全合规（计算 100% VF）** | `m2`、`m5`、`m6`、`m12`、`m4`、`m9`、`m14`、`m15_gdn_layer.h`、`m15_attn_layer.h` | 零 (a)、零经典 `Cast`；见 §3.7 / §3.8 |
| **无向量计算可查** | `m1_mxfp4_gemm`、`m11_bf16_gemm` | 纯 cube：只有 `DataCopy`/`LoadData`/`Mmad`/`Fixpipe`（(c)） |
| **形态存疑（非违规，需裁决）** | `m4`、`m14`、`m15_gdn_layer.h` | `EgExpAll`/`GdnHeadRecurrence` 是 `__simd_vf__`，但**在 `__VEC_SCOPE__` 外被裸调用**（`m4:214`/`m4:279`、`m14:1084`/`m14:1149`） |

**一句话结论**：**「(a) 类需要改代码」的实际清单只有 `m3` 一个生产模块（5 处）+ `m0` 探针（10 处）+ `m8` 一处存疑**；其余全部已经落在 `__VEC_SCOPE__` + `RegTensor`/`AscendC::Reg::` 上。**真正需要 tower 裁决的不是改造量，而是两处口径冲突**：`Sort32`/`MrgSort` 无 VF 等价物（旧白名单 `docs/05:30` 与新标准 §6.1:115 冲突），以及 `__simd_vf__` 裸调用是否满足「VEC_SCOPE」的字面要求。

**方法论警告（本次审计的第一条教训）**：**不能按 API 名字判类。** 本仓 `m2/m5/m6/m7/m8/m9/m12/m13/m14` 全部带 `using namespace AscendC::Reg;`，在 `__VEC_SCOPE__` 内裸写 `Add(...)`/`Cast(...)`/`Select(...)` 时解析到的是 **`Reg::` 重载（`RegTensor` 参数）**，是合规的；同一个名字写在 VEC_SCOPE 外、操作 `LocalTensor` 才是违规。按名字 grep 会得到**相反**的结论（例如会把已合规的 m2/m5 判成重灾）。本审计对每个调用点都判**操作数类型 + 是否在 VEC_SCOPE 内**。

---

## 1. 分类口径

### 1.1 四类定义

| 类 | 定义 | 处置 |
|---|---|---|
| **(a) 违规** | **计算类**语义（逐元素算术/广播/归约/比较/选择/位宽转换/超越函数/gather/interleave/pack 等）在**内存**（`LocalTensor`/`__ubuf__ T*`）上用**经典 memory-based API** 完成；或由**逐元素标量循环**（`GetValue`/`SetValue`、手写 element loop）在 UB 上做本该向量化的计算 | 应改为 VF |
| **(b) 合规** | 计算在 `__VEC_SCOPE__` 内（或 `__simd_vf__` 函数体内），操作 `RegTensor`/`MaskReg`，经 `AscendC::Reg::` 重载。`Reg::LoadAlign`/`StoreAlign`/`UpdateMask`/`CreateMask`/`LocalMemBar` 属寄存器侧 load/store 与屏障，合规 | 不动 |
| **(c) 允许的 memory-based（搬运/矩阵/同步）** | GM↔UB `DataCopy`/`DataCopyPad`/`Copy`；L1→L0 `LoadData`（本仓口中的 `LoadL0_2D`）；`Mmad`/`MmadMx`；`Fixpipe`；`Mutex`/`BufAcquire`/`BufRelease`/`SetFlag`/`WaitFlag`/`CrossCore*`/`PipeBarrier`；`GetPhyAddr`/`SetGlobalBuffer`；`Nd2Nz`/`Dn2Nz` 参数结构 | 不动 |
| **(d) 边界** | 计算类却**找不到 register/VF 等价物**的 memory-based 调用 | 逐项说明可否接受；不可接受或口径冲突者报**存疑** |

### 1.2 判定的三个陷阱（都已在本仓命中）

1. **名字 ≠ 类别**：`Add`/`Cast`/`Select`/`Reduce` 在 VEC_SCOPE 内是 `Reg::` 重载（(b)），在 VEC_SCOPE 外作用于 `LocalTensor` 才是 (a)。本仓两种写法**同时存在**（如 m6 全 VF vs m3 全经典）。
2. **`__simd_vf__` 是编译器属性**（`/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_defines.h:44`：`__attribute__((cce_simd_vf))`），**不是词法作用域**。因此 `__simd_vf__` 函数体内的 Reg 算子即使外层没有 `__VEC_SCOPE__` 块，也仍是「SIMD vector function 内」——**M44 审计时它与 `docs/05:114`（`55f5e18` 旧行，见文首出处说明）的字面要求「必须在 vector function（`__VEC_SCOPE__`）内使用」存在张力**（见 §9 存疑 1）；**该旧行此后已被 §6.1 计算路径规范取代，现行文本把 `__VEC_SCOPE__` 与 `__simd_vf__` 明确定为等价**。
3. **host 代码不算**：每个 `.asc` 都含 host 段（参考实现、dump、`main`）。host 段的逐元素 C 循环不是违规。各文件的 host/device 边界见 §3。

### 1.3 范围界定（明确声明，避免被读成遗漏）

- **host 段**：不计入 (a)/(d)。
- **`probe_sync_quirks/`**：是**取证工程**（M23），不是产品算子。它的 `(a)` 命中大多是**故意保留的反例/靶子**，因此本审计对它只做**边界（(d)）与证据引用**，不建议改造。
- **未合入 main 的 worktree**（`m17_moe_real`/`m18_gdn_prefill`/`m19_qsa_indexer`/`m20_hyperconn`/`m21_layer_ref`/`m22_router512`/`probe_vf_loop`、M24 的 `m10_attn_decode`）：**只读参考，不作为基线**。参考价值见 §8.3。
- **`DataCopy` 逐条不枚举**：本仓 `DataCopy*` 约 239 处、`Mmad/LoadData/Fixpipe` 数十处，属 (c) 白名单；本文件只给**逐函数计数 + 每类 API 代表点位**（这是可读性取舍，非遗漏）。**(a) 与 (d) 是完整枚举，无遗漏。**
- **(b) 只做逐函数计数**（一个函数内部哪些算子在 VEC_SCOPE 内），不逐条列寄存器算子。

---

## 2. 结论总表（16 个目标）

| 模块 | 设备段行范围 | (a) | (d) | 经典 `Cast(` on AIV | 计算路径判定 |
|---|---|---|---|---|---|
| `m0/m0_bringup.asc` | 1–315（kernel 300–315） | **10** | 0 | 无 | **非合规**（AIV 计算全 memory-based） |
| `m1_mxfp4_gemm/m1_mxfp4_gemm.asc` | 1–334 | 0 | 0 | 无 | 合规（无向量计算，纯 cube） |
| `m2_mxfp4_quant/m2_mxfp4_quant.asc` | 1–286 | 0 | 0 | 无 | **合规（全 VF）** |
| `m3_grouped_gemm/m3_grouped_gemm.asc` | 1–569（AIC+AIV） | **5** | 0 | **1（`:450`）** | **非合规（AIV 段经典）** |
| `m4_gdn_recurrent/m4_gdn_recurrent.asc` | 48–322 | 0 | 0 | 无 | 合规（全 Reg；但调用形态存疑，§9.1） |
| `m5_swiglu_quant/m5_swiglu_quant.asc` | 1–350 | 0 | 0 | 无 | **合规（全 VF）** |
| `m6_rmsnorm/m6_rmsnorm.asc` | 1–558 | 0 | 0 | 无 | **合规（全 VF）** |
| `m7_router_topk/m7_router_topk.asc` | 52–490 | 0 | **3**（`Sort32`:352 / `Extract`:362 / `MrgSort`:450） | 无 | **混合**（softmax/gemv/renorm 全 VF；topk 排序 memory-based） |
| `m8_permute/m8_permute.asc` | 74–353 | **1 存疑**（`:157-159`） | 0 | 无 | 基本合规（compute 全 VF） |
| `m9_gdn_prolog/m9_gdn_prolog.asc` | 56–440 | 0 | 0 | 无 | **合规（全 Reg，且 VF 调用形态最规范）** |
| `m11_bf16_gemm/m11_bf16_gemm.asc` | 1–272 | 0 | 0 | 无 | 合规（无向量计算，纯 cube） |
| `m12_rmsnorm_gated/m12_rmsnorm_gated.asc` | 1–562 | 0 | 0 | 无 | **合规（全 VF）** |
| `m13_moe_layer/m13_moe_layer.asc` | 1–2070（host 2072+） | **0**（+3 标量胶水存疑） | **2**（`Sort32`:974 / `Extract`:975） | 无 | **基本合规**（15 个 VEC_SCOPE 覆盖全部段；例外：router topk + `IndexGenStage` 标量） |
| `m14_gdn_layer/m14_gdn_layer.asc` | 38–1832（host 1833+） | 0 | 0 | 无 | **合规（全 Reg）**；调用形态存疑同 m4 |
| `m15_layer_loop/m15_gdn_layer.h` | 1–1841 | 0 | 0 | 无 | **合规**（与 m14 ≤1623 行逐行同源，结论继承） |
| `m15_layer_loop/m15_attn_layer.h` | 33–67 | 0 | 0 | 无 | 合规（纯搬运占位） |
| `m15_layer_loop/m15_layer_loop.asc` | —（无设备代码） | 0 | 0 | — | 纯 host 驱动 |
| `probe_sync_quirks/*.asc` | 见 §3.6 | 1+3（探针反例） | **6**（`MrgSort`/`Sort32`/`MrgSort4`） | 无 | 探针工程（(d) 为主） |

---

## 3. 逐模块明细

> 每个模块给：host/device 边界 → **(a)/(d) 完整表**（`文件:行号` + 成本 + 位级风险）→ **逐函数 (a)/(b)/(c) 计数表** → 小结。
> 成本分级口径（§4.1 复述）：**低** = 机械搬寄存器、同数学、无布局/归约/交接变化、≲30 行；**中** = 需归约或布局重构、增删 UB staging、或触及 mask 语义、~30–150 行；**高** = 涉及标量↔向量交接（`GetValue`/`SetValue`/逐元素寻址/循环）、数据重排、改变归约次序、>150 行或跨函数。

### 3.1 `m3_grouped_gemm` — 唯一的生产级 (a) 重灾【已确证】

device 1–569；host 571–1021（含 host 侧同规范量化器 `QuantRowH:702-725`，不算违规）。

| # | file:line | enclosing function | API 调用（原文，截断） | 操作数 | 类 | 成本 | 成本理由 | 位级风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | `m3_grouped_gemm.asc:450` | `M3GroupedGemm::ProcessAiv` | `AscendC::Cast<float, bfloat16_t>(guF, guRaw, AscendC::RoundMode::CAST_NONE, GU_N)` | `LocalTensor<float>`←`LocalTensor<bfloat16_t>`，1280 elem | **(a)** | 中 | AIV 侧经典位宽转换，`docs/05:123` **点名禁用**；须 even/odd 两条 `Reg::Cast` + `LoadDist::DIST_UNPACK_B16`（m6:114-132 有现成 `CastTrait` 范本） | **低**（bf16→fp32 无损，逐位可保） |
| 2 | `m3_grouped_gemm.asc:451` | `M3GroupedGemm::ProcessAiv` | `AscendC::Silu(guF, guF, DN_K)`（原地，前 640） | `LocalTensor<float>` | **(a)** | 中 | Reg 命名空间**无 `Silu`**（已核 reg_compute 头文件）⇒ 须用 `Muls(-1)→Exp→Adds(1)→Div` 组合（官方 sigmoid 即此形，§8.2） | **高**（换实现 = 换超越函数近似，`docs/17` 按 **T3** 判；须同步改参考/判据） |
| 3 | `m3_grouped_gemm.asc:494` | `M3GroupedGemm::QuantRow`（:485-519） | `guF.GetValue(g*SCALE_GROUP+j) * guF.GetValue(DN_K + g*SCALE_GROUP+j)` | `LocalTensor<float>` 逐元素标量读 | **(a)** | **高** | 整个 `QuantRow` 是标量量化器：20 组×32 elem 的 h 乘法 / amax 归约 / fp32 位域取 scale / e2m1 最近值编码 / nibble 打包，全在标量相位；矢量化需 amax(`Reg::Max` 跨 VL) + 位域(`ShiftRights`/`And`/`Adds`) + `Pack`，>150 行且跨函数 | **中**（见 §5：整数域可保逐位，但属既有逐字节验收契约） |
| 4 | `m3_grouped_gemm.asc:505` | `M3GroupedGemm::QuantRow` | `scalB.SetValue(g, sByte)` | `LocalTensor<uint8_t>` 逐元素标量写 | **(a)** | 高（与 #3 同一次重构） | scale 字节由 fp32 位运算得出（:500-505），VF 侧须 `ShiftRights/And/Adds` 或 `Compare/Select` | 中 |
| 5 | `m3_grouped_gemm.asc:516` | `M3GroupedGemm::QuantRow` | `packB.SetValue(g*(SCALE_GROUP/2)+j/2, (uint8_t)(c0|(c1<<4)))` | `LocalTensor<uint8_t>` 逐元素标量写 | **(a)** | 高（同 #3） | nibble 打包 + `E2M1Code`（:470-482）纯标量阈值链；VF 侧等价物为 `Compare/Select` + `Pack`（`DIST_PACK4_B32` 落盘） | **中–高**（字节级位序契约 lo=even / hi=odd） |

**同一 (a) 组内的附属标量算核**（未单列成 API 站点，但属同一重构范围）：`:496-498`（`|h|` 与 amax）、`:508-515`（invScale、取绝对值、符号位）、`:470-482` `E2M1Code`、`:521-527` `ScaleByteToFloat`。

逐函数计数：

| 函数（线） | (a) | (b) | (c) | 判定 |
|---|---|---|---|---|
| `MXFP4GemmItem<K,N,S>::Run`（AIC） | 0 | 0 | 17（`Mutex`×16 + `MmadMx`:196） | 合规（矩阵/搬运） |
| `MXFP4GemmItem::CopyInA/CopyInB/LoadA/LoadB/CopyOut` | 0 | 0 | 8（`DataCopy`×4 :221/:254/:236/:269、`LoadData`:292/:316、`Fixpipe`:328） | 合规 |
| `M3GroupedGemm::ProcessAic` | 0 | 0 | 8（`CrossCore*`:389-396 + `countsGM.GetValue`:378/:400） | 合规 |
| `M3GroupedGemm::ProcessAiv` | **2**（:450/:451） | 0 | 18（`DataCopy`×3 :443/:461/:462、`Mutex`×10、`CrossCore*`:423/:465） | **计算 memory-based** |
| `M3GroupedGemm::QuantRow` | **3**（:494/:505/:516） | 0 | 0 | **完全 memory-based** |

**README/证据交叉引用**：
- `m3_grouped_gemm/README.md:76-78` 记录了 **3510 quirk（实测）**：「连续两个 `Cast<float,bfloat16_t>` 后者输出错乱（保留 UB scratch 干扰）」，现行规避是「整行一次 `Cast` + 把 `silu*up` 移进 scalar 量化器」。⇒ 这正是 `:450`、`:451`、`:494` 三处的**成因**：**经典路径特有的问题被"绕开"，而不是被修掉**。按新标准把该段整体 RegBase VF 化，该规避的前提很可能同时消失——**但必须重测，不能假定**（见 §5 与 §9.2）。
- `m3_grouped_gemm/README.md:98-100`：H 的 packed/scale 字节与 host 量化器**零失配（逐字节）**——即 `QuantRow` 的整数输出是 **T1 级判据**，重构不得破。

**小结**：m3 的 AIC 与全部搬运/同步合规；**AIV 的 SwiGLU+量化段整段经典，且全文件零 `__VEC_SCOPE__`**。这是本审计里**唯一"改了就有收益"的生产模块**（它同时踩中用户理由 ②「memory-based 中间量必须落 UB」与 ③「经典 `Cast` + 标量↔向量交接出过真 bug」）。

---

### 3.2 `m0/m0_bringup.asc` — bring-up 探针，10 处 (a)【已确证】

device 1–315（`__global__` 300–315）；host 317–562。

| # | file:line | enclosing function | API 调用 | 操作数 | 类 | 成本 | 成本理由 | 位级风险 |
|---|---|---|---|---|---|---|---|---|
| 1–7 | `m0_bringup.asc:144`–`:150` | `M0Bringup::ProcessAiv` | `AscendC::Add(accB, accA, t1, TILE_LEN)` 等 7 连 | `LocalTensor<float>`×3，1024 elem | (a) | 低 | 1024 fp32 = 16 VL 的机械循环（`LoadAlign/Add/StoreAlign`），加序不变 | 低（同序加 ⇒ 逐位一致） |
| 8 | `m0_bringup.asc:184` | `M0Bringup::ProcessAiv` | `AscendC::Add(redB, redA, partialRd[c*MODE0_CHUNK], MODE0_CHUNK)` | `LocalTensor<float>`，16 elem（<1 VL） | (a) | 低 | 需 `UpdateMask(16)`；56 项汇总主循环 | **中**（归约次序=真值次序，host 容差 1e-3；换归约树会移位） |
| 9 | `m0_bringup.asc:186` | 同上 | `AscendC::Add(redA, redB, partialRd[c*MODE0_CHUNK], MODE0_CHUNK)` | 同上 | (a) | 低 | 同 #8 | 中（同 #8） |
| 10 | `m0_bringup.asc:213` | `M0Bringup::ProcessAiv` | `AscendC::Adds<half>(outHalf, cHalfH, half(1.0f), MAT_ELEMS)` | `LocalTensor<half>`（源是 `:210` 的 `uint16` 同址换型视图） | (a) | 低–中 | 256 half = 2 VL；VF `RegTensor<half>` + `Adds` | **中**（half 域 +1 的舍入点 + uint16 视图契约；host 容差 2e-3 :519。视图来自 `docs/06-m0-bringup.md:67` 的 API 约束） |

逐函数计数：`M0Bringup::ProcessAiv` (a)=10 / (b)=0 / (c)=57（`DataCopy`×10、`DataCopyPad`×2、UB→UB `Copy`×4、`SetFlag/WaitFlag`×26、`Mutex`×11、`CrossCore*`×3、`PipeBarrier`×2）；`ProcessAic` (a)=0/(b)=0/(c)=15（`DataCopy` Nd2Nz :252-253、`LoadData` :262/:265、`Mmad` :272、`Fixpipe` :278）→ **合规**。host 不计。

**小结**：**无 `__VEC_SCOPE__`**；AIV 计算全经典。但这是 bring-up 探针（用途是验证 BufferID/mode0、uint16 视图等**经典路径与同步原语**），其 `evidence/` 日志是既有基线。**建议不改（见 §7 批次 5）**——改它只会让历史证据对不上，不产生任何性能/功能收益。

---

### 3.3 `m7_router_topk` — 混合：gemv/softmax/renorm 全 VF，topk 排序 memory-based【已确证】

device 52–490；host 492–764。`__VEC_SCOPE__` 10 处（:153/:199/:217/:328/:366 + 5 个单行 `LocalMemBar` 块 :416/:420/:424/:428/:430）。

**(d) 完整表**：

| # | file:line | enclosing function | API 调用 | 操作数 | 类 | 成本 | 成本理由 | 位级风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | `m7_router_topk.asc:352` | `SoftmaxTopkRenormRow` | `Sort32(tmpT, valT, idxT, SORT_BLK);` | `LocalTensor<float>`×2 + `LocalTensor<uint32_t>` | **(d)** | **高** | Reg 侧**无** sort/compare-exchange 网络（CANN 9.1.0 `reg_compute/` 已穷举核实）；替换 = 换算法（K 趟 `Reduce<MAX>`+mask 选择），>150 行 | 高（产出 `topk_ids` 次序与 value/index 配对） |
| 2 | `m7_router_topk.asc:362` | `SoftmaxTopkRenormRow` | `Extract(vT, iT, mT, 2);` | `LocalTensor<float>`/`uint32_t` | **(d)** | **低–中** | **可转换**：官方 donor 已在 VF 内重实现同一件事（§8.2），厂商的经典 3510 `Extract` **本身就是 VF**（`asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h:426-487` 的 `ExtractVf`）⇒ 属"把 wrapper 内联进来"，不是重写 | 低（纯 pair 拆分，无算术） |
| 3 | `m7_router_topk.asc:450` | `Merge2`（经 `MergeTree` :408-431 调用 15 次/行） | `MrgSort(dst, sl, params);` | `LocalTensor<float>` + `MrgSortSrcList<float>` | **(d)** | **高** | Reg 侧无 `MrgSort`（同 #1 的穷举结论）；`MrsSort4Info.elementLengths` 单位=8B 对（probe A 结论） | 高（改变整个 top-k 结构；平局次序不指定） |

逐函数计数（b/c 近似）：

| 函数（线） | (a) | (d) | (b) | (c) | 判定 |
|---|---|---|---|---|---|
| `RouterTopk::Process`（131-145） | 0 | 0 | — | — | 合规 |
| `BuildIndexTemplate`（149-163） | 0 | 0 | 3 | 0 | 合规 |
| `CopyInX`/`LoadW`/`CopyOut` | 0 | 0 | 0 | 1/2/3 | 合规（搬运） |
| `PrecastW`（194-210） | 0 | 0 | 4 | 0 | 合规 |
| `Dot8Row`（215-269） | 0 | 0 | ~42 | 0 | 合规 |
| `Gemv`（272-320） | 0 | 0 | 0 | 同步 | 合规 |
| `SoftmaxTopkRenormRow`（323-405） | 0 | **2**（:352/:362） | ~34 | 0 | **混合** |
| `MergeTree`（408-431） | 0 | 5（:450 的调用点 + 5 `LocalMemBar`） | 0 | 0 | **混合** |
| `Merge2`（434-451） | 0 | **1**（:450） | 0 | 0 | **混合** |

(c) 代表点位：`DataCopy` :171 / :183 / :459 / :465；`BufAcquire/Release` :168 / :274。
**经典 `Cast(` on AIV：无**（全部 `Reg::Cast`，:206/:229）。
**注意**：m7 **没有** `WholeReduceMax`/`Concat`；归约一律寄存器侧 `Reduce<SUM/MAX>`（:249-256/:337/:376）⇒ 任务书里担心的"memory-based 归约不可替代"在 m7 **不成立**。
**次级重构候选（非违规）**：`:384-397` 用**手写整数 RNE + `Reg::DeInterleave`** 落 bf16；`docs/05:123`（M35 归因修正）明确这是「多余的自造动作」，规范写法是 `Reg::Cast` + `Reg::StoreAlign<..., StoreDist::DIST_PACK_B32>`（m8:313-314 已是范本）。
**文档卫生**：`m7_router_topk.asc` **文件头注释**（原 `:36-37`）与 `m7_router_topk/README.md` 的「4-way MrgSort 丢 src3/src4 index」已被 `docs/05 §6.3 #19` + probe A/A2 取代（真因是 `MrgSort4` 在 3510 上静默 no-op）；m7 当前用的是**正确的** `MrgSort(dst, srcList, MrgSort4Info)`。

---

### 3.4 `m13_moe_layer` — 基本合规；(d) 2 处 + 标量胶水 3 处【已确证】

device 1–2070；host 2072+（`H_*` 2079、`H_RunCase` 2708、quant-inf 自测 3302+）。
`__VEC_SCOPE__` **15 处**（:418/:551/:589/:659/:688/:837/:856/:896/:954/:977/:1378/:1418/:1480/:1597/:1693），覆盖 13 个函数。

**(d) 完整表**：

| # | file:line | enclosing function | API 调用 | 操作数 | 类 | 成本 | 成本理由 | 位级风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | `m13_moe_layer.asc:974` | `RouterStage::SoftmaxTopkRow`（945-995） | `Sort32(pairT, valT, idxT, 1);` | `LocalTensor<float>`/`uint32_t` | **(d)** | 高 | 同 `m7:352`：无 Reg 等价物 | 高（`topk_ids`/`w_tk_packed`/`inv_slot` 的次序与配对都继承它） |
| 2 | `m13_moe_layer.asc:975` | 同上 | `Extract(ovT, oiT, pairT, 1);` | `LocalTensor<float>`/`uint32_t` | **(d)** | **低–中** | 同 `m7:362`：**可转换**（内联厂商 `ExtractVf` 的 Reg 实现或 donor 写法） | 低 |
| 3 | `m13_moe_layer.asc:1069-1076` | `IndexGenStage::Run`（1057-1154） | `idsGm.GetValue(s)` → `cnt[e] += 1`（逐元素标量直方图） | `GlobalTensor<int32_t>` 标量读 + 标量数组 | **(d) 存疑** | 高 | 数据相关直方图；`Reg::Histograms`（`reg_compute/kernel_reg_compute_histograms_intf.h:31`）**存在**，但 bin id 来自 GM ⇒ 须先把 ids 搬进 UB/VF，标量↔向量交接 | 中（计数为整数精确；分组语义须稳定） |
| 4 | `m13_moe_layer.asc:1078-1082` | `IndexGenStage::Run` | `off[e+1] = off[e] + cnt[e];`（E=4 前缀和） | 标量数组 | **(d) 存疑** | 高 | E=4 无向量收益，但确属 VF 外的计算型胶水 | 低（整数精确） |
| 5 | `m13_moe_layer.asc:1101-1123` | `IndexGenStage::Run` | `srcUb[pos]=t; expUb[pos]=e; invUb[s]=e*M_MAX+(pos-off[e]); …wtkUb[...]=rounded;` | `GlobalTensor` 标量读 + 裸 `__ubuf__` 写 | **(d) 存疑** | 高 | 数据相关 **scatter**（counting-sort 落位）+ **标量 bf16 RNE 打包**（:1118-1121）。scatter 侧**有** `Reg::Scatter`（`reg_compute/kernel_reg_compute_datacopy_intf.h:144`，→`vscatter`）⇒ 落位可转；但 index 计算仍是数据相关标量 | 中（RNE 位技巧与 `inv_slot` padding 编码是**逐位契约**） |

逐段/逐函数计数：AIC `MXFP4GemmItem`(139-354) (a)=0/(b)=0/(c)≈22 合规；`NormDonor`(377-570) 全 VF；`NormStage`(609-754) 全 VF；`RouterStage`(773-1025) **混合**（除 :974/:975）；`IndexGenStage`(1034-1168) **全标量**；`PermuteStage`(1184-1250) 纯搬运合规；`VecQuantStage`(1261-1514) 全 VF（m2/m5 同款，**已无 PIPE_S 量化**）；`UnpermuteStage`+`FmaChunk`(1534-1637) 全 VF；`CombineStage`(1644-1745) 全 VF；编排层(1789-2070) 纯同步。
**经典 `Cast(` on AIV：无**（22 处 `Cast<` 全是 VEC_SCOPE 内的 `Reg::Cast`）。
**继承关系**：m13 是 m6/m7/m8/m3/m5 的抄改件 ⇒ **m6/m5/m8 部分继承其合规结论，m7 部分继承其 (d) 边界**；旧的 m3 AIV 标量量化器**已被移除**（docs/12 §6 小改 C），这是 m13 比 m3 干净的原因。
**旁证（未合入 worktree，仅参考）**：`wt-40` 的 `m13_moe_layer.asc` 与本树同构（同样 15 VEC_SCOPE、同样 :974/:975、同样标量 `IndexGenStage`），另加 `m17_moe_real` 把 (d) 放大成 `Sort32` + 2 路 `MrgSort` 归并树 + `Extract`。⇒ **本审计结论可原样继承到 M40/M29**（但不得把它们当 main 现状）。

---

### 3.5 `m8_permute` — 基本合规，仅 1 处 (a) 存疑【已确证】

device 74–353；host 355–794。`__VEC_SCOPE__` 1 处（:287）。

| # | file:line | enclosing function | API 调用 | 操作数 | 类 | 成本 | 成本理由 | 位级风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | `m8_permute.asc:157-159` | `MoePermuteKernel::Process` | `for (e…) total += static_cast<uint32_t>(countsGm.GetValue(e));` | `GlobalTensor<int32_t>` 标量读（E≤512） | **(a) 存疑** | 中 | 逐元素标量循环做归约，结果**只喂标量控制**（行数/分条）。工具书口径是"做 vector math 且经 UB"；此处是标量读 **GM**，更像控制路径 | 无（整数精确和） |

逐函数计数：`MoePermuteKernel::Process`(153-175) **(a) 存疑 1**；`CopyIn`/`CopyOut` (c) 合规；`FmaChunk<K>`(215-226) (b)=6（**注意**：它是普通 `__aicore__ inline` 且接受 `RegTensor&`，被**内联进调用方 VEC_SCOPE**（:287-316）⇒ 合规，且正是 `__simd_vf__` 不能传 `RegTensor&` 时该用的形态）；`MoeUnpermuteKernel::ComputeRow`(277-318) (b)=10 + ≤9 次 `FmaChunk` ⇒ 全 VF。
(c)：`DataCopy` :183/:191/:267/:270/:324；索引用 `GetValue` :181/:269（**正确地 hoist 在 VF 之外**，符合 probe B #23）。
**m8 是本仓的"规范范本"**：bf16 落盘用 `Reg::Cast<bfloat16_t,float,…>` + `StoreAlign<..., Dist_PACK_B32>`（:313-314）；**permute 本身用逐行 `DataCopy` 完成**——即"搬运类"承担了 `Gather` 的活，**不在禁令内**（这点值得写进标准解读：`Gather` 被点名禁的是 register/UB 上的 gather 计算，不是用 `DataCopy` 做数据搬移）。

---

### 3.6 `probe_sync_quirks/` — 取证工程，(d) 为主【已确证】

device/host 边界：`probe_a_mrgsort4.asc` 1–403 / 405–690；`probe_a2_mrgsort4_api.asc` 1–164 / 166–321；`probe_b_vec_idx.asc` 1–363 / 365–492。

| # | file:line | 函数 | API 调用 | 类 | 说明 |
|---|---|---|---|---|---|
| 1 | `probe_a_mrgsort4.asc:202` | `DoMerge`（27 处调用） | `MrgSort(dst, sl, p)` | **(d)** | Reg 侧无 sort/merge（穷举核实） |
| 2–4 | `probe_a_mrgsort4.asc:330/:336/:358` | `MrgSortProbe::RunVariant`(A14/A15/A20) | `Sort32(...)` | **(d)** | 同上 |
| 5 | `probe_a2_mrgsort4_api.asc:124` | `MrgSort4ApiProbe::Process` | `MrgSort4(...)` | **(d)** | **3510 上是 `[[deprecated]]` 空函数体**（`ASCENDC_REPORT_NOT_SUPPORT` 展开为空）⇒ 根本不发射指令 |
| 6 | `probe_a2_mrgsort4_api.asc:139` | 同上 | `MrgSort(...)` | **(d)** | 正确形态 |

`probe_b_vec_idx.asc`：`__VEC_SCOPE__` **10 处**（:159/:182/:193/:206/:228/:241/:261/:282/:309/:334），计算基本都在寄存器侧；命中 (a) 的 4 处（:259 标量累加器；:178-180 / :329-331 / :306 标量 staging/广播填充）**全是探针故意保留的反例与靶子**，不建议改造。
该文件对本标准的**三条硬约束证据**（应写进落地规范）：① VF 循环归纳变量必须 `uint16_t`（:163）；② VF 内 `LoadAlign` 仍须 **32B 对齐**（:314，form 9 → `507035`）——**"非 32B 对齐"能力只在 `LoadUnAlign`/`StoreUnAlign`/`Load`/`Store`/`Gather`/`Scatter`/`AddrReg` 那一族**，不在 `LoadAlign`/`StoreAlign`（§8.1）；③ **VF 内标量读 GM 编译失败**（`Unsupported Inst must be hoisted`，:248/:288）⇒ 必须 hoist 出 VF。
**附带证据**：form 6（标量结果当向量标量操作数）**恒定错 ~350/400** ⇒ 任何 (a)→(b) 迁移方案都要避开该形态。

---

### 3.7 已完全合规组 A：`m2` / `m5` / `m6` / `m12`（量化与归一化）【已确证】

**四者 (a)=0、(d)=0、经典 `Cast(` on AIV=0，计算 100% 在 `__VEC_SCOPE__` 内。**

| 文件 | device / host | `__VEC_SCOPE__` 段 | (b) | (c) | 关键点 |
|---|---|---|---|---|---|
| `m2_mxfp4_quant.asc` | 1–286 / 289+ | (183,238)、(245,268) | 35 | 13 | `MxQuantComputeScale`(179-239) 全寄存器；`MxQuantComputeDataFP4`(242-269) 全寄存器 |
| `m5_swiglu_quant.asc` | 1–350 / 353+ | (208,232)、(246,301)、(308,331) | 47 | 15 | siLU 五元组全在 fp32 寄存器（:218-230 `UpdateMask/LoadAlign/Cast/Muls/Exp/Adds/Div/Mul/Cast/StoreAlign`） |
| `m6_rmsnorm.asc` | 1–558 / 561+ | 7 段 | 102 | 18 | 四段 donor（XAdd / square-reduce / NR rsqrt / Y）全寄存器；`Reduce<SUM>` 寄存器版（:164 等），**无** memory-based `ReduceSum/WholeReduceMax` |
| `m12_rmsnorm_gated.asc` | 1–562 / 565+ | 6 段 | 102 | 17 | sigmoid 用 `Muls/Exp/Adds/Div`（:491-494）寄存器组合——与 `docs/05:123` 要求一致 |

**"名字陷阱"命中清单**（按名字判会误判为 (a)，实为 **(b)**）：`m2:217`/`m5:280` `ReduceDataBlock<MAX>`；`m2:261`/`m5:324` `Interleave`；`m6:164/194/233/240/250/263`/`m12:174…` `Reduce<SUM>`；`m2:262-263`/`m5:325-326` fp4 `Cast`；`m6:485`/`m12:484` `LoadAlign<..., DIST_BRC_B32>`。
**m6/m12 的两条非 API 备注**（非违规，供维护者）：`NormDonor::StoreRegForDtype`（m6:136-147 / m12:146-157）是**死代码**；`CalculateSquareReduceSumLessThanVL/LessThanTwoVL` 对 HIDDEN=2560 不可达（m6 dispatcher 走 `Common<1>`）。
**m6 的代码规范 nit**：仍在用元素计数 `DataCopy` 重载（m6:442/443/536/537），而 m12 已按 `docs/05:128` 迁到显式块参数。

**M32 ±Inf 一致性修复的位置（全部在 VEC_SCOPE 内）**：m2:235 / m5:298 的 `Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult)`（常量 m2:64 / m5:69；`Duplicate` m2:205-206 / m5:268-269；m13 同位 :1412/:1440-1441/:1469）。

---

### 3.8 已完全合规组 B：`m4` / `m9` / `m14` / `m15`（GDN 系，register-based）【已确证】

| 文件 | device / host | `__VEC_SCOPE__` | `__simd_vf__` | (a) | (d) | 判定 |
|---|---|---|---|---|---|---|
| `m4_gdn_recurrent.asc` | 48–322 / 324–502 | **0** | 2（:112 `EgExpAll`、:129 `GdnHeadRecurrence`） | 0 | 0 | 全 Reg 计算；**调用形态存疑** |
| `m9_gdn_prolog.asc` | 56–440 / 442–648 | 1（:360-374） | 1（:144 `PrologBlock`） | 0 | 0 | 全 Reg；**唯一"VEC_SCOPE 包裹 + `__simd_vf__`"双保险写法** |
| `m14_gdn_layer.asc` | 38–1832 / 1833–3001 | 9（:151/:214/:235/:347/:425/:454/:834/:1268/:1601） | 3（:618/:982/:999） | 0 | 0 | 全 Reg；`EgExpAll`/`GdnHeadRecurrence` 调用形态存疑 |
| `m15_layer_loop/m15_gdn_layer.h` | 1–1841（无 host） | 同 m14（:151…:1601） | 同 m14（:618/:982/:999） | 0 | 0 | **继承 m14**（≤1623 行逐行相同） |
| `m15_layer_loop/m15_attn_layer.h` | 33–67 | 0 | 0 | 0 | 0 | 纯搬运占位（`DataCopy` :57/:61） |
| `m15_layer_loop/m15_layer_loop.asc` | 无设备代码 | 0 | 0 | 0 | 0 | 纯 host 驱动（`__global__`/`VEC_SCOPE`/`DataCopy` 命中 0） |

**零经典 memory-based 向量算件的取证方式**：对 40+ 个经典 API 名（`Transpose/Sort32/MrgSort/Concat/Extract/WholeReduce*/BlockReduce*/Gather*/Scatter/Brcb/Interleave/DeInterleave/Pack/Histogram/Where/Clamp/Relu/Sigmoid/Tanh/Reciprocal/SoftMax/…`）做词边界扫描 → **0 命中**；`GetValue`/`SetValue` 在 m14/m15 设备段 **0 命中**（无"逐元素标量冒充向量"型 (a)）；`Cast(` 全部 12 处（m14）均为 `Reg::Cast(RegTensor,…)`。
**`__simd_vf__` 调用形态清单**（本组唯一立案项）：

| helper | 定义 | 是否自带 `__VEC_SCOPE__` | 调用点 | 调用点是否在 `__VEC_SCOPE__` 内 | 判定 |
|---|---|---|---|---|---|
| `PrologBlock` | m14:618 / m9:144 / m15_gdn_layer.h:618 | 否 | m14:847（VEC_SCOPE 834-848 内） / m9:373（VEC_SCOPE 360-374 内） | **是** | 合规 |
| `EgExpAll` | m4:112 / m14:982 / m15_gdn_layer.h:982 | 否（body 全 `AscendC::Reg::`） | **m4:214** / **m14:1084** | **否**（裸 `{}` 作用域） | **存疑**（§9.1） |
| `GdnHeadRecurrence` | m4:129 / m14:999 / m15_gdn_layer.h:999 | 否（body 全 `AscendC::Reg::`） | **m4:279** / **m14:1149** | **否**（`ComputeHead` 内无 VEC_SCOPE） | **存疑**（§9.1） |
| `ComputeRstdNewtonRaphsonReg` | m14:283（`__aicore__`，`RegTensor<float>&` 入参，非 `__simd_vf__`） | 否 | m14:360 | 是（VEC_SCOPE 347-363） | 合规 |
| `ComputeRstdNewtonRaphson` | m6:351 / m12:361 / m14:343 | 是（自带 VEC_SCOPE） | dispatch 调用 | — | 合规 |

**注释与事实不符（两种裁定下都需改）**：`m14:604-606` / `m15_gdn_layer.h:604-606` 写「寄存器 VF 计算（`__simd_vf__`，**全部在 `__VEC_SCOPE__` 内调用**）」——对 `PrologBlock` 成立，对 `EgExpAll`/`GdnHeadRecurrence` **不成立**。
**m14↔m15 重复性**：归一化命名后仅 **58 行**差异，实质 3 处（`GdnLayerPtrs` 增 `yLayer` m15:1624；`norm2.Init` 出口改 `args.yLayer` m15:1654-1656；新增层循环 kernel 入口 m15:1804-1841）。**段序、同步表、各段算件逐字未动** ⇒ **m14 若因本标准改动，必须手工同步到 `m15_gdn_layer.h`**（最易漏的一步）。
**m14 里"本想用 gather 语义"的地方已被写成纯寄存器**：`PrologBlock` 的 one-hot lane 抽取（m14:687-696，`Arange` + `Compares<EQ>` + `Reduce<SUM>` + `StoreAlign DIST_FIRST_ELEMENT_B32`）——这是**标准想要的写法**，可作为"gather 的 VF 替代范式"参考。

---

### 3.9 无向量计算可查：`m1_mxfp4_gemm` / `m11_bf16_gemm`【已确证】

两者 (a)=0、(d)=0：设备段**不存在任何向量计算**，全部是 `DataCopy`(Nd2Nz/Dn2Nz) / `LoadData` / `Mmad`(`MmadMx`) / `Fixpipe` + `Mutex` + 参数视图。
代表点位：`m1:202`(Nd2Nz) / `m1:218`(Dn2Nz) / `m1:275`(LoadData MX) / `m1:173`(MmadMx) / `m1:313`(Fixpipe)；`m11:190` / `m11:221` / `m11:164`(Mmad) / `m11:254`(Fixpipe)。
`ReinterpretCast<half>()`（m1:218/:251、m11）是位视图，不是计算。⇒ **就本标准而言无可查项**（若将来把 MX scale 计算搬进来，会立刻触发新标准）。

---

## 4. (a) 类：成本估计与"已被 review 抓过同类 bug 的位置"

### 4.1 成本分级口径

| 级 | 定义 |
|---|---|
| **低** | 机械搬寄存器、同数学、无布局/归约/交接变化；结果逐位一致或只有一个明确定义的舍入点；≲30 行 |
| **中** | 需归约/布局重构、增删 UB staging、或触及 `Select`/mask 语义；~30–150 行 |
| **高** | 涉及标量↔向量交接（`GetValue`/`SetValue`、逐元素寻址/循环）、数据重排/转置、改变归约次序、>150 行或跨函数 |

### 4.2 (a) 清单（按成本排序）

| 位次 | file:line | 成本 | 是否标量↔向量交接 | 是否逐元素寻址 | 是否需重排布局 | 备注 |
|---|---|---|---|---|---|---|
| 1 | `m3:485-519` `QuantRow`（:494/:505/:516） | **高** | **是**（`GetValue`/`SetValue` 逐元素） | 是（`g*32+j`、`g*16+j/2`） | 否（同布局逐位可保） | 全仓唯一的"标量量化器"；**与 M24 的 Combine 段同类**（§4.3） |
| 2 | `m13:1101-1123` IndexGen scatter + 标量 bf16 RNE | **高** | **是**（GM `GetValue` + 裸 `__ubuf__` 写） | 是（数据相关 `cursor[e]`） | 否 | 落位侧可用 `Reg::Scatter`；index 计算仍是数据相关标量 |
| 3 | `m13:1069-1076` 标量直方图 | 高 | 是 | 是 | 是（须把 ids 从 GM 搬 UB） | `Reg::Histograms` 存在 |
| 4 | `m13:1078-1082` E=4 前缀和 | 高 | 否 | 否 | 否 | 计算量极小，成本全在"搬进 VF"的 plumbing |
| 5 | `m7:450` `MrgSort`（15 次/行） | 高 | 否 | 否 | 是 | **不可转**（§6.1），列此仅为完整 |
| 6 | `m7:352` / `m13:974` `Sort32` | 高 | 否 | 否 | 是 | **不可转**（§6.1） |
| 7 | `m3:451` 经典 `Silu` | 中 | 否 | 否 | 否 | Reg 无 `Silu`，须组合 `Muls/Exp/Adds/Div` |
| 8 | `m3:450` 经典 `Cast<float,bfloat16_t>` | 中 | 否 | 否 | 否（bf16→fp32 无损） | `docs/05:123` 点名禁用 |
| 9 | `m8:157-159` counts 求和 | 中 | 是（结果喂标量控制） | 否 | 是（须搬 UB） | 存疑：属控制路径 |
| 10 | `m7:362` / `m13:975` `Extract` | **低–中** | 否 | 否 | 否 | **可转**（内联厂商 `ExtractVf` 的 Reg 实现） |
| 11–20 | `m0:144-150`（7×`Add`）、`m0:184`/`:186`（`Add`）、`m0:213`（`Adds<half>`） | 低 | 否 | 否 | 否 | bring-up 探针；**建议不改** |

### 4.3 已被 review 抓过同类真 bug 的位置（本审计特别标注）

| 历史事件 | 记录位置 | 本仓对应的**存量同类风险点** |
|---|---|---|
| **M24**：经典 `Cast` + **标量↔向量交接**出真 bug，换 `CombineWeightsVf`/`CastRowToBf16Vf` 后消失；经典路径下 softmax sum 高 1.086–1.111×、P 出现"偶位 0/奇位 >1"交错 | `docs/05` §6.1 位宽转换行（`docs/05:123`）；`docs/05:114`（`55f5e18` 旧行） | **`m3:450`（经典 `Cast`）+ `m3:451`（经典 `Silu`）+ `m3:485-519`（把 V/经典输出用 `GetValue` 逐元素读回标量相位算 `h = gate*up`）** —— 这正是 M24 那条通路的**同型结构**（经典位宽转换 + 经典逐行计算 + 标量接管） |
| **507015 fault**：原归因"`Duplicate(dst,scalar,1)` 是硬件 quirk"，后被隔离探针推翻 ⇒ 真因是 **UB 目的地址 32B 对齐**；且出错那次 `Duplicate(…,1)` **本身就是 memory-based**，按新标准不该出现在计算路径 | tower 广播 `repo-507015-fault-...`；`docs/05` §6.3 待复核表 | `probe_sync_quirks/probe_b_vec_idx.asc:306-321`（form 9，4B 步长 `LoadAlign` → `507035`）是**同类对齐坑**的证据基线；m3 的 `SetValue` 逐元素写 `scalB`/`packB`（:505/:516）也属"标量写 UB"形态 |
| **M32**：MX 量化 ±Inf 一致性（halfScale 须覆盖 `0x7F81`） | `docs/13` §5 行 10a / §6；修复点 m2:235 / m5:298 / m13:1469 | 已修且**已在 VF 内**；但 `docs/13` 的表述与行号**已过期**（§10.1） |
| **M3 记录在案的 3510 quirk**：连续两个经典 `Cast<float,bfloat16_t>` 后者输出错乱 | `m3_grouped_gemm/README.md:76-78`、`.asc:446-447` | `m3:450-451` 的**现行规避**（整行一次 `Cast` + `silu*up` 移进标量量化器）——按新标准改 VF 时**该规避的前提需重测**（经典路径特有，见 §5.3） |
| **m13 router topk**（与 m7 同源） | `m13:974-975` | 与 m7 同型 (d)，同一裁定对象 |
| **M24 的 507015 现场曾同时改多处 ⇒ 归因未隔离** | tower 广播 | 本次审计**不新报任何"硬件 quirk"**：§6 的 `Sort32`/`MrgSort` 只说"Reg 侧无等价物"（头文件穷举，可复核），未说"硬件不支持" |

---

## 5. 风险清单：哪些 (a) 改造会**改变位级结果**

### 5.1 纯机械替换（位级不变）

| 位置 | 理由 |
|---|---|
| `m0:144-150`、`m0:184`/`:186` | 同操作数、同加序的 fp32 `Add` ⇒ 寄存器版逐位一致（前提：**保持加序**） |
| `m0:213` | `half + 1.0f` 同一条 `Adds` ⇒ 逐位一致（须保留 `:210` 的 uint16 同址视图语义） |
| `m3:450` | bf16→fp32 **无损**，任何取整模式都逐位一致（须用 `LoadDist::DIST_UNPACK_B16` + `Reg::Cast`） |
| `m7:362` / `m13:975` `Extract` | 纯 pair 拆分（DINTLV + store），无算术 ⇒ 逐位一致 |
| `m8:157-159`（若改） | 整数精确和 ⇒ 位级不变 |

### 5.2 会改变位级结果 / 需同步改参考与判据

| 位置 | 会变什么 | 判据影响 |
|---|---|---|
| **`m3:451` 经典 `Silu` → Reg 组合** | **换超越函数近似**（经典 intrinsic 的近似多项式 ≠ `Exp`+`Div` 组合） | `docs/17` §1.1 触发条件 ① ⇒ 该段判据**必须升到 T3**（推导界 + 逐元素检查）；现 README 的容差口径需重写并写明理由 |
| **`m3:485-519` 标量量化器 → VF** | 理论上**整数域逐位可保**（每步都是单次 fp32 乘 + 精确位运算 + 阈值链），但：① amax 归约从"顺序比较"变成"跨 VL `Max` 树"——`max` 幂等且精确 ⇒ **不影响**；② 打包位序 lo=even/hi=odd 是 **T1 契约** | 判据**不得降档**（T1 逐字节）；须重跑 `m3_grouped_gemm/README.md` §校验方法与结果 的 host 逐字节对拍（原 `README.md:98-100`，未写模块名易误读）。**风险中**：属"重写既有逐位契约"，而非"必然变" |
| **`m13:1101-1123` 标量 RNE bf16 打包** | 若改用 `Reg::Cast` + `DIST_PACK_B32`，**舍入模式必须与手写 RNE 逐位一致**（`m7:384-397` 的教训：M35 曾因自造 RNE 把值写成 NaN） | 属 T1/T2 边界；须逐元素对拍后再切 |
| **`m3:450` 相关的"连续 Cast 干扰"规避** | 若改 VF，经典路径特有干扰**预期消失**，但**须实测确认**；若误判而保留 scalar 接管，则等于把规避理由写进了新代码 | 需一条独立小实验（不是纯机械） |

### 5.3 需要新实验才能定方案的（高风险项）

1. **`m3` 量化段矢量化后的"标量↔向量交接"面**：现行实现把 `h` 的点积放在**标量相位**（因为经典 Cast 干扰）；改 VF 后建议**整段（Cast→Silu→Mul→amax→scale→pack）留在 VF 内**，**不要**再把中间量交给标量。若仍需要标量参与，注意 `probe_sync_quirks` 的两条实测：标量结果当向量标量操作数**恒定错**（form 6）、标量↔VF 通路间歇错（`docs/05:154`）。
2. **`m13 IndexGenStage` 的 index 生成本质是数据相关标量**：即使 scatter 用 `Reg::Scatter`，`cursor[e]` 的推进与 `counts` 的分条仍需标量或 `Arange`+mask 技巧；须先做小实验再定成本（§9.3）。
3. **`Reg::Gather` 的 index 单位（元素 vs 字节）**在所有头文件里**都未写明**，任何依赖 gather 的重构必须先标定（不要照搬 donor 参数——`docs/05:120` 的教训）。

---

## 6. 边界清单：确实没有 VF 等价物的 memory-based 调用

> 判据来源：CANN 9.1.0 `asc/include/basic_api/reg_compute/**` + `asc/impl/basic_api/reg_compute/{dav_3510,dav_l300}/**` 穷举 grep（详见 §8.1）。

### 6.1 【已确证】无 register/VF 等价物，且**不是**搬运/矩阵 ⇒ 需裁定

| API | 本仓使用位置 | 性质 | 可否接受 | 存疑 |
|---|---|---|---|---|
| **`Sort32`** | `m7:352`、`m13:974`、`probe_a:330/:336/:358` | 计算（32 lane 内降序排序） | 目前**只能** memory-based | **是**（§9.2）：`docs/05:30` 旧白名单（tower 代判、注明"可推翻"）与新标准 §6.1:115 冲突。**官方 donor 同样是 memory-based**（`ops-transformer/moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h:187`）⇒ 属"行业惯例级例外"，建议显式列入例外清单而非当作违规 |
| **`MrgSort` / `MrgSortSrcList` / `MrgSort4Info`** | `m7:450`（×15/行，经 `Merge2`/`MergeTree`）、`probe_a:202`、`probe_a2:139` | 计算（多路归并） | 同上；官方 donor 亦用（`…perf_arch35.h:225/:243`） | **是**（同上） |
| **`MrgSort4`** | `probe_a2:124` | **在 3510 上是 `[[deprecated]]` 空函数体** ⇒ 不下发指令、无计算 | 建议**从代码库清除**（Deprecated + no-op + 与 `MrgSort` 参数不可互换，是易踩陷阱），与本事裁决无关但顺手 | 否（只建议清理） |
| **`SetValue` / `GetValue`** | `m3:505/:516`（量化打包）、`m8:158/:181/:269`（index/counts）、`m13:1070/:1104/:1117`（index bookkeeping） | 无 register→scalar lane 提取通路；寄存器侧最近的是 `Reg::Store(addr,reg,count)` / `MaskPattern::VL1` 的掩码 store | **索引/地址类 `GetValue` 应视为控制路径、可接受**（m8/m13 的用法即如此，且已被 probe B #23 证明必须 hoist 出 VF）；**m3 那两处 `SetValue` 是计算落盘，属 (a)** | 部分（§9.3） |
| **`Transpose` / `Concat`** | **本仓 0 使用** | 无 VF 形式；且 3510 经典 `Concat` 是**no-op**（`asc/impl/basic_api/kernel_operator_proposal_intf_impl.h:299-302`） | N/A（无暴露） | 否 |

### 6.2 【已确证·重要修正】曾被误列进"边界"但**其实可转**的项

| API | 位置 | 为什么可转 |
|---|---|---|
| **`Extract`** | `m7:362`、`m13:975` | ① 厂商 3510 的经典 `Extract` **本身就是 VF**（`asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h:426-487` 的 `ExtractVf`，`__simd_vf__` + `Reg::LoadAlign<DIST_DINTLV_B32>` + `Reg::StoreAlign` + `Reg::DeInterleave` + `Reg::Squeeze`）；② 官方 topk donor **已不再用**经典 `Extract`，而是手写了 `ExtractKFP32Perf`（`…perf_arch35.h:309-343`，`__VEC_SCOPE__` + `Reg::LoadAlign<DIST_DINTLV_B32>` + `Reg::Squeeze`/`StoreUnAlign`）⇒ **这是"内联 wrapper"级的低–中成本改造，不是重写** |
| **归约（`ReduceSum`/`ReduceMax`/`ReduceMin`/`WholeReduce*`）** | 本仓**未使用**经典版 | 寄存器侧有 `Reg::Reduce<SUM/MAX/MIN>`（整 VL）、`ReduceDataBlock<…>`（32B 块）、`ReduceSum/Max/MinWithDataBlock`；**注意粒度差异**：`Reg` 归约一次只覆盖一个 256B VL ⇒ 跨长张量的归约要显式累加循环（这是成本来源，不是"不可转"） |
| **`Gather` / `GatherMask`** | 本仓**未使用**经典版（m8 的 permute 用 `DataCopy` 逐行搬） | `Reg::Gather`（`reg_compute/kernel_reg_compute_gather_mask_intf.h:42`）、`Reg::GatherB`、`Reg::Squeeze`/`GatherMask<STORE_REG>`（同文件 :30）、`Reg::Gather`（UB 版 `datacopy_intf.h:136`）、`Reg::Scatter`（:144）**都存在** ⇒ gather/scatter 属**寄存器侧能力** |
| **`Rsqrt` / `Reciprocal` / `Sigmoid` / `Silu`** | m6/m12/m14/m9 的 rsqrt 是自写 `Sqrt`+Newton 序列（**已合规**）；`Silu` 只在 m3:451（经典） | Reg 侧无这些名字，但厂商自己在 3510 都用 Reg 组合实现（`Reciprocal` `asc/impl/basic_api/dav_3510/kernel_operator_vec_unary_impl.h:347-360`；`Rsqrt` `:435-463`；`Sigmoid` `asc/impl/adv_api/detail/activation/sigmoid/sigmoid_3510_impl.h:36-64`）⇒ **照抄厂商组合即可** |

### 6.3 【已确证】搬运/矩阵/同步（标准明文允许，不动）

`DataCopy` / `DataCopyPad` / `DataCopyExtParams` / UB→UB `Copy` / `Nd2Nz`·`Dn2Nz` / `LoadData`(+`LoadData2DParamsV2`、`LoadData2DMxParams`) / `Mmad` / `MmadMx` / `Fixpipe`(+`FixpipeParamsArch3510`) / `Mutex::Lock/Unlock` / `GetBufInternal`·`RlsBufInternal` / `SetFlag`·`WaitFlag`·`CrossCoreSetFlag`·`CrossCoreWaitFlag` / `PipeBarrier` / `LocalMemBar`（**仅 VEC_SCOPE 内**）/ `GetPhyAddr` / `SetGlobalBuffer` / `Trap`。
**术语修正（备查）**：`LoadL0_2D`/`LoadL0_3D` 在本仓文档里是简写；CANN 9.1.0 里这两个名字**不存在**（grep 无此符号），实际是 `LoadData` 的 2D/3D 重载。写代码时不要按 `LoadL0_2D` 去查头文件。

---

## 7. 建议的改造批次（每批一个可独立验收的 mission 提案；只给建议）

> 排序依据：**收益（用户理由 ② 的 UB 压力 / ③ 的 bug 史）/ 成本 / 风险**。**不拆现有 scope**：每批只碰自己的模块目录。

| 批次 | 提案（mission 草案） | scope | 内容 | 成本/风险 | 验收要点 |
|---|---|---|---|---|---|
| **B1（最高优先）** | **m3 AIV 段 RegBase VF 化（Cast + SwiGLU + 量化器）** | `m3_grouped_gemm/**` | ① `:450` 经典 `Cast` → `Reg::Cast` + `LoadDist::DIST_UNPACK_B16`（低风险，可先做）；② `:451` 经典 `Silu` → `Muls/Exp/Adds/Div` Reg 组合；③ `:485-519` `QuantRow` → 全 VF（amax 跨 VL + 位域 scale + `Compare/Select` + `Pack`/`DIST_PACK4_B32`）；④ 顺带重测"连续经典 Cast 干扰"是否随 VF 化消失（若消失，删除该规避及其注释） | 中–高 / 中（③ 的 T1 逐字节契约） | ① host 逐字节（`m3_grouped_gemm/README.md` §校验方法与结果，原 `README.md:98-100`）保持零失配；② 该段判据按 `docs/17` 声明档位（`Silu` 段 = **T3**，推导 + 逐元素）；③ 与 m13 的 `VecQuantStage` 输出对拍（同源代码应逐位一致） |
| **B2** | **Extract 的 VF 内联（m7 + m13）** | `m7_router_topk/**`、`m13_moe_layer/**` | 用厂商 `ExtractVf` 的 Reg 实现（或 donor `ExtractKFP32Perf` 写法）替换 `m7:362`、`m13:975` 的经典 `Extract` | 低–中 / 低 | top-k 的 value/index 平面与替换前**逐位一致**（T1）；topk_ids/weights 全量对拍 |
| **B3** | **bf16 落盘归一：去掉手写整数 RNE + `DeInterleave`** | `m7_router_topk/**` | `m7:384-397` → `Reg::Cast` + `StoreAlign<..., Dist_PACK_B32>`（m8:313-314 已是范本，`docs/05:123` 亦如此要求） | 低 / 中（位级须逐位对拍） | 落盘 bf16 逐位一致；这是**简化**而非新能力，收益是可维护性 |
| **B4** | **`__simd_vf__` 调用形态统一（m4/m14/m15）** | `m4_gdn_recurrent/**`、`m14_gdn_layer/**`、`m15_layer_loop/**` | 依 tower 对 §9.1 的裁定二选一：① 每个调用点包 `__VEC_SCOPE__`（≲5 行/处，4 处）；② 采用上游 `asc_vf_call<F>` 形态（本仓 0 使用）并改 `m14:604-606` 注释 | 低 / 无（纯包裹，不动算子序列） | 数值结果逐位不变 + 编译无 warning；**m14 改动必须同步 m15_gdn_layer.h**（58 行差异是同步点） |
| **B5** | **（建议不做）m0 bring-up 探针 VF 化** | `m0/**` | 10 处经典 `Add`/`Adds` | 低 / 中 | **建议列为"证据豁免"**：m0 是 bring-up 探针，其价值在于验证经典路径与同步原语，改造会使其 `evidence/` 基线失效而**不带来任何收益**。若要统一标准，应在文档里显式豁免（注明理由），而不是改代码 |
| **B6（需先小实验）** | **m13 IndexGenStage 的标量化改造** | `m13_moe_layer/**` | 直方图/前缀和/落位/RNE 打包 → VF（`Reg::Histograms`、`Reg::Scatter`…） | **高** / 中（RNE 与 inv_slot 位级契约） | 先做 index 生成的可行性小实验（`cursor` 推进与 `counts` 分条是数据相关标量）；不成就**显式留标量并立案豁免** |
| **B7（非代码，文档）** | **例外清单落库** | tower（docs/05） | 把 §6.1 的 `Sort32`/`MrgSort` 写成**显式例外**（附"Reg 侧无等价物 + 官方 donor 同形"的证据），替换 `docs/05:30` 的旧白名单；并明确 `__simd_vf__` 调用形态（§9.1） | — | 消除 §9.2 的口径冲突；同时把 `docs/13` §5 的行号与 10a 行状态刷新（§10.1） |

---

## 8. 证据基础

### 8.1 register-based API 面（CANN 9.1.0，dav-3510）

- **结构性事实**：`AscendC::MicroAPI` 是 `AscendC::Reg` 的别名（`asc/impl/basic_api/kernel_macros.h:127`）；`__simd_vf__`/`__simd_callee__` 是编译器属性（`tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_defines.h:44-45`）⇒「SIMD vector function」与 `__VEC_SCOPE__` 是同一制度的两面。
- **存在 Reg 版**（节选，声明位于 `asc/include/basic_api/reg_compute/`）：binary `Add/Sub/Mul/Div/Max/Min/ShiftLeft(s)/ShiftRight(s)/And/Or/Xor/…`；binary-scalar `Adds/Muls/Maxs/Mins/ShiftLefts/ShiftRights/LeakyRelu`；ternary-scalar `Axpy`；unary `Abs/Relu/Exp/Sqrt/Ln/Log/Log2/Log10/Neg/Not`；`Compare/Compares/Select`；`Cast`(+`Truncate`)；`Duplicate/Interleave/DeInterleave`；reduce `Reduce<SUM|MAX|MIN>`、`ReduceDataBlock<…>`、`PairReduceSum`、`ReduceSum/Max/MinWithDataBlock`、`PrefixSum`；`Pack/UnPack`；`Squeeze/Unsqueeze/GatherMask<…>/Gather/GatherB/Scatter/GetSpr`；load/store `LoadAlign/StoreAlign`(+`LoadDist`/`StoreDist`、post-update、AddrReg、block-stride、**unaligned 族** `LoadUnAlign(Pre)/StoreUnAlign(Post)/Load/Store`、掩码 load/store)；`UpdateMask/CreateMask/Move/MoveMask`；`Arange`；`CreateAddrReg`；fused `MulsCast/AbsSub/ExpSub/MulDstAdd`；`Histograms`；`LocalMemBar`。
- **不存在 Reg 版**（穷举 grep：`include/basic_api/reg_compute`、`include/interface/reg_compute`、`include/c_api/reg_compute`、`impl/basic_api/reg_compute/{dav_3510,dav_l300,dav_l311}`）：`Sort*`、`MrgSort*`、`Extract`、`Concat`、`Transpose`、`Rsqrt`、`Reciprocal`、`Sigmoid`、`GetValue`、`SetValue`。
- **`Cast` 支持的目标类型对**（权威白名单：`asc/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h:515-536`）：含 `f32←bf16`、`bf16←f32`、`bf16←fp8_e8m0`、`fp8_e8m0←bf16`、`fp4x2_e2m1←bf16`、`f16←f32` 等；**没有** `fp32→fp4`、`half↔fp4`、`fp32/fp16→e8m0`（e8m0 只能从 bf16 来）—— `docs/13:16-22` 的结论经此复核**成立**。
- **32B 对齐的确切边界（对应用户理由 ①）**：`LoadAlign`/`StoreAlign`（`vlds`/`vsts` 族）**在 VF 内也仍要求 32B 对齐**（本仓设备证据：`probe_b_vec_idx.asc:306-321` form 9 → `507035`）。**"非 32B 对齐"能力只在** `LoadUnAlign`/`StoreUnAlign`/`Load`/`Store`（单元素/短计数）/`Gather`·`GatherB`·`Scatter`（元素粒度）/`CreateAddrReg`（任意元素寻址）/掩码 store 这一族——**且经典 basic_api 里根本没有 `LoadUnAlign`/`StoreUnAlign`**（grep 0 命中）⇒ 用户理由 ① 成立且**表述可更精确**：不是"VF 的 load/store 支持非对齐"，而是"**非对齐/单元素/gather-scatter 这族只在寄存器侧**"。
- **`ReduceDataBlock` 粒度**：块 = **32 字节**（=16×u16 或 8×f32），不是 32 元素（与 `docs/13:150` 一致）。

### 8.2 官方/donor 侧用的是哪一族？（对 m2/m5/m13 定性的关键）

| 官方文件 | 逐函数判定 |
|---|---|
| `/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/add_rms_norm_dynamic_mx_quant_common.h` | **量化路径全寄存器**：`MxQuantComputeMaxExpOCP`(:307-355，VEC_SCOPE :310)、`MxQuantComputeScaleOCP`(:357-420，VEC_SCOPE :372)、`MxQuantComputeDataFP4`(:660-768，VEC_SCOPE :664) 均为 `Reg::`；`LocalTensor` 只用于 `GetPhyAddr()`；全文件**零**经典计算调用 |
| `ASCEND_HOME/opp/.../ops_nn/ascendc/dynamic_mx_quant/arch35/dynamic_mx_quant_tail_axis.h` | **寄存器**（18 × `__VEC_SCOPE__`，全部 `Reg::…`） |
| `/workspace/ops-nn/quant/dynamic_mx_quant/.../dynamic_mx_quant_tail_axis.h`、`activation/gelu/.../gelu_dag.h`、`norm/rms_norm/.../rms_norm_regbase_common.h`、`activation/swiglu_group_quant/...`、`activation/softmax_v2/...` | **全寄存器**（抽样 5 个，均为 `__VEC_SCOPE__` + `Reg::`） |
| **`/workspace/ops-transformer/moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h`（m7 的 donor）** | **混合，且与我们的形态一致**：`InitIndex` 是 Reg VF（VEC_SCOPE :168）；但 `SortFP32Perf`(:187 `Sort32`)、`MergeSortFP32PerfBlockMerge`(:225 `MrgSort`)、`MergeSortFP32Perf2To1`(:243 `MrgSort`) 是**经典 memory-based**；`MergeSortFP32PerfCopy` 用经典 `Duplicate(LocalTensor)`(:196) + `DataCopy`；**而 `ExtractKFP32Perf`(:309-343) 已经不用经典 `Extract`，改成手写 Reg VF** |
| `/workspace/ops-transformer/moe/moe_token_permute_with_routing_map/.../gather_v2_simd_two_dim.h`（m8 的 donor） | **无 Reg 计算，纯搬运**（`DataCopyPad` :97/:112 + 标量 index 读）⇒ m8 用 `DataCopy` 做 permute 是**继承 donor 形态** |
| 语料规模 | `/workspace/ops-nn` 下 **567** 个文件、`/workspace/ops-transformer` 下 **130** 个文件含 `__VEC_SCOPE__` ⇒ 寄存器化是官方主流写法 |

**三条决定性事实**：① **排序（`Sort32`/`MrgSort`）在官方 donor 里也是 memory-based** ⇒ 没有"官方写法"可抄；② **`Extract` 官方已改为 Reg VF** ⇒ 我们那两处是"还没跟上"；③ 官方 topk 的 `Duplicate(LocalTensor)`/`DataCopy` 属填充/搬运，正落在本裁决的豁免里。

### 8.3 未合入 main 的 worktree（只读参考，非基线）

- **M24（`wt-24/m10_attn_decode`）**：`docs/05:123` 记录的"经典 `Cast` + 标量↔向量交接出真 bug → 换 VF 后消失"就是这里；它是 B1 的先例（`CombineWeightsVf`/`CastRowToBf16Vf`）。**不作基线，仅作为"改了会怎样"的证据**。
- **M34（GDN prefill，`:__simd_vf__` + `TrilSolveVF`/`GemmAA`）**、**M36（hc mixer，4 条流整段驻 UB 的理由来源）**、**M40（MoE 并层）/M29（`m17_moe_real`，把 (d) 放大为排序树）**、**M35（`m19_qsa_indexer`）**、**M42（`m22_router512`）**：均为**参考**。其中 M40/M29 的 m13 抄改件与本审计结论同构（§3.4 旁证）。
- **`probe_vf_loop`（M43）**：VF 嵌套循环 trip count 探针，与本次 §9.1 的"VF 形态"问题相关，可作后续证据来源。

---

## 9. 存疑汇总（待 tower 裁决）

| # | 存疑 | 事实 | 需要什么裁定 |
|---|---|---|---|
| **9.1** | **`__simd_vf__` 裸调用是否满足「VEC_SCOPE」** | `m4:214`/`m4:279`、`m14:1084`/`m14:1149` 在 `__VEC_SCOPE__` **外**裸调用 `__simd_vf__` 函数（m4 全文 0 个 VEC_SCOPE）；`PrologBlock`（m9:373/m14:847）则是 VEC_SCOPE **内**调用。`docs/05:114`（`55f5e18` 旧行）字面要求"Reg API 必须在 vector function（`__VEC_SCOPE__`）内使用"；但 `__simd_vf__` = `__attribute__((cce_simd_vf))`（`__clang_cce_defines.h:44`），CANN 上游用 `asc_vf_call<F>` 调用 `__simd_vf__` 函数且调用点也不在 VEC_SCOPE 内（`asc/include/basic_api/kernel_common.h:48`；官方例 `xlog1py_kernel.h:229-236`）。**本仓 `asc_vf_call` 命中 0** | ① 是否要求词法 `__VEC_SCOPE__`；② 若要求 ⇒ B4（≲5 行/处）；③ 若不要求 ⇒ 只需改 `m14:604-606`/`m15_gdn_layer.h:604-606` 的注释（现状与事实不符）。**并请一并给出"规范写法"结论（裸调用 vs `asc_vf_call`）**，因为 m4/m14/m15 都继承自 donor 的裸调用形态 |
| **9.2** | **`Sort32`/`MrgSort` 的例外** | 两者在 Reg 侧**确无等价物**（穷举核实），**且官方 donor 也用 memory-based**；但新标准 §6.1:115 的豁免只覆盖"搬运/矩阵"，`docs/05:30` 的旧白名单（tower 代判、注明"可推翻"）把 `Sort32/MrgSort/WholeReduceMax/Concat/Extract` 列为允许的指令级原语 | 是否把「排序/归并（`Sort32`/`MrgSort`）」**显式列入例外**；若不入例外，则 `m7:352`/`m7:450`/`m13:974` 无法合规、必须换算法（成本高且改变平局次序）。**建议：入例外 + 要求"仅在 topk 排序段使用、不得扩散"**。附带请裁定 `Extract`：本审计确认它**可转**（§6.2），建议**不**入例外而是列 B2 改造 |
| **9.3** | **`m13 IndexGenStage` 的标量 index 生成是否属"计算类"** | `m13:1069-1123` 是直方图 + 前缀和 + 数据相关 scatter + 标量 bf16 RNE 打包（`README §6.3` 自承"小改 E 只做了一半"）。它不是"vector API on LocalTensor"，而是**标量控制/胶水**；但确属 VF 外的计算 | ① 是否纳入本标准（若纳入 ⇒ B6，成本高、需先小实验）；② 若宽容，请在文档里明确"数据相关 index/控制路径不属禁令"，避免以后反复争论 |
| **9.4** | **`m8:157-159`（标量读 GM `counts` 求和，喂标量分条控制）** | 逐元素标量循环做归约，但走的是 GM 读、结果喂控制 | 同 9.3 的定性：控制路径可宽容，还是要求搬进 UB 用 VF 求和（成本中、无收益）。**建议宽容并写进文档** |
| **9.5** | **bring-up 探针（`m0`）与取证工程（`probe_sync_quirks`）是否豁免** | `m0` 10 处经典 `Add`/`Adds`（其**用途**就是验证经典路径/同步原语）；`probe_b` 的 (a) 是**故意保留的反例** | 是否**显式豁免**"探针/证据工程"（建议豁免并写进 docs/05 或 docs/18 的适用范围），否则会出现"为了合规而破坏证据基线"的荒谬动作 |
| **9.6** | **host 参考实现是否受约束** | 各模块 host 段有大量逐元素 C/numpy 循环（如 m2:330+、m3:702-725） | 明确"仅设备计算路径"（**建议明确豁免**，避免把 host 参考也算违规） |

---

## 10. 附：交叉引用发现（越出本文件范围，建议他人修）

### 10.1 `docs/13` §5 的行号与状态**已过期**【已确证】

`docs/13-mx-quant-primitives.md` **§5「逐句差异表：官方 vs m2/m5/m13 设备路径」**（M32 写作时的行号口径 `:104-124`，已过期；以**§号 + 列名**定位），其"我们的设备 kernel"列锚点指向 **M32 修复前**的版本：

| docs/13 锚点 | 现在实际行 | 漂移 |
|---|---|---|
| `m2:209-212`（And×2+Max+ReduceDataBlock） | **m2:214-217** | +5 |
| `m2:215` / `:216-217` / `:218` / `:219` / `:220` | **m2:220 / :221-222 / :223 / :224 / :225** | +5 |
| `m2:225`（halfScale zero 覆盖） | **m2:236** | +11 |
| `m2:251-252`（乘法） | **m2:259-260** | +8（且 **pre-fix 版本本身是 :248-249**，该锚点在其自身基线上也偏 3，疑为抄写错误） |
| `m5:273-276 / :279 / :280-281 / :282 / :283 / :284 / :287 / :288` | **m5:277-280 / :283 / :284-285 / :286 / :287 / :288 / :291 / :292** | +4 |
| `m5:289` / `:312-313` / `:314` | **m5:299 / :322-323 / :324** | +10 |
| **行 10a「halfScale 非有限覆盖 = 缺失」**（`docs/13:117`）与 §6（`:139`）「已立 mission…修复前不要声称位级一致」 | **已修复**：`m2:235`、`m5:298`（+ m13:1469），且 evidence 有 after-fix PASS 日志 | **状态需翻转** |

⇒ `docs/13` 需刷新行号 + 把 10a 改为"本核已实现"。**该文件不在本 mission 的写入范围**（本 mission 只写 `docs/18`），已另报 finding。

### 10.2 `docs/05` 的两条旧表述与新标准冲突

- `docs/05:30`（§2 指令级白名单）：`Sort32/MrgSort/WholeReduceMax/Concat/Extract` 列为"指令级原语、允许使用"（tower 代判，注明"可推翻"）⇒ 与 §6.1:115（计算类一律 VEC_SCOPE）**冲突**。另：本审计实测 `WholeReduceMax`/`Concat` 在本仓**并未使用**，`Extract` 可转、`Sort32`/`MrgSort` 不可转 ⇒ 白名单可以收窄成**只保留 `Sort32`/`MrgSort`**。
- `docs/05:114`（旧行）："Register-based API 必须在 vector function（`__VEC_SCOPE__`）内使用"——与 §9.1 的 `__simd_vf__` 裸调用问题直接相关，建议与 §6.1:115 合并表述并给出规范写法。
- m14 的若干代码注释仍引用**已被 docs 撤回**的机制（`m14:91-93`/`:774-776`"元素计数 DataCopy 读到未初始化栈垃圾"、`:801-803`/`:866-871`"blockCount>1 走 NZ 重排"），`docs/05` 早已把它们改成"API 语义/代码规范"（`docs/05:127`/`:145`）。**代码写法本身是对的，只是注释理由过期**。

---

## 11. 方法与可复核性

1. **判定按调用点，不按 API 名**：每个站点看①操作数类型（`LocalTensor`/`__ubuf__ T*` vs `RegTensor`/`MaskReg`）②是否在 `__VEC_SCOPE__` 内（或 `__simd_vf__` 体内）③是否属搬运/矩阵/同步类。
2. **数据来源**：对 16 个目标做逐行扫描（模块级扫描由只读子代理按同一份分类口径并行完成）；**争议项与高价值项由我本人复核原文**：`m3:438-519`、`m7:344-366`/`:434-451`、`m8:150-165`、`m13:965-980`/`:1055-1102`、`m0:138-152`/`:178-190`/`:205-215`、`m4:108-131`/`:205-218`/`:270-283`、`m14:618/982/999` 调用点。其中**发现并纠正了一处子代理的错误结论**（"m3 是 cube-only、无向量计算"——实为 §3.1 的 5 处 (a)）。
3. **CANN 侧结论**为**头文件存在性/穷举 grep**，未编译、未运行（只读 mission）。凡"某算子是否支持某 dtype"未逐条取证者，本文件未作断言。
4. **未做**：未编译、未跑任何 kernel；未修改任何文件；未把未合入 main 的 worktree 当作基线。
5. **已知不完整**：`(c)` 类未逐条枚举（§1.3）；`(b)` 类只给逐函数计数；`Reg` 侧 dtype 支持表只对 `Cast` 做了完整枚举。
