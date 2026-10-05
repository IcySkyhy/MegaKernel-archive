# MXFP4 量化原语与官方实现对照（昇腾 950 / dav-3510）

> 来源：M28 只读调研（agent-mxquant，2026-09-26），由 tower 整理入档。全部结论来自**静态阅读随包 CANN 头文件与 ops-nn/ops-transformer 源码**，未做真机验证——标「存疑」的条目见 §7。
> 背景裁决：用户明确「**参考官方的，和官方一致就行了**」——本文件即「官方是什么」的取证结果。

## 1. 一句话结论

**不存在"一条硬件指令替代三段式"的 MX 量化原语。** 华为官方 `DynamicMxQuant` kernel 走的就是和我们 m2/m5/m13 同构的三段式：

1. 组内**指数域** max（`And 0x7F80` ×2 → `Max` → `ReduceDataBlock<MAX>`）
2. E8M0 scale + halfScale（`Sub` → `ShiftRights 7` 等）
3. 乘 → `Cast` → `Pack4`

逐句对拍后，**15 个步骤已全部位级等价**：曾经唯一的真实差异是第 10a 项（Inf 输入的 halfScale 覆盖，原实现缺一句 `Select`），**已由 M32 补齐**（`m2:235` / `m5:298` / `m13:1469`，并补了含 ±Inf 的用例）——详见 §5/§6。

顺带更正一个此前流传的错误结论：「不存在 e8m0 cast」**是错的**——`Reg::Cast<fp8_e8m0_t, bfloat16_t>` 存在于 C++ `Reg::` 模板层（早前只查了 c_api `reg_convert.h` 就下结论，取证不完整）。但它**不构成性能理由**，见 §3。

## 2. e8m0 cast 存在性（分层回答）

| 层 | e8m0 cast | 证据 |
|---|---|---|
| **Reg**（C++ `AscendC::Reg::`） | ✅ 存在 | `asc/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h:177-186`（bf16→e8m0）、`:55-65`（e8m0→bf16） |
| c_api（`asc_simd` / `reg_convert.h`） | ❌ 不存在 | `asc/include/c_api/reg_compute/reg_convert.h` 全文件 grep `e8m0` = 0 命中 |
| basic_api（`LocalTensor` 版 `Cast`） | ❌ 不存在 | `kernel_operator_vec_vconv_impl.h:1345-1362` 的 tuple 白名单无 e8m0 |

`Reg::Cast<fp8_e8m0_t, bfloat16_t>` 的逐句实现（`kernel_reg_compute_vec_vconv_impl.h:177-186`）：

```cpp
// bf162f8e8m0 will ignore sat mode and use sat only.
vshls(srcReg_u16, srcReg_u16, SHIFT_ONE_BIT,   mask, modeValue);   // << 1
vshrs(srcReg_u16, srcReg_u16, SHIFT_EIGHT_BIT, mask, modeValue);   // >> 8
vcvt (dstReg_u8,  srcReg_u16, mask, satModeOnly, partModeValue, modeValue); // u16 -> u8
```

`<<1 >>8` 从 bf16 位 `[15:0]` 中取出偏置指数域 `E = b[14:7]`，`vcvt` 截成 1 字节。所以

> `Reg::Cast<fp8_e8m0_t, bfloat16_t>(x)` 的字节值 = `E` = `127 + floor(log2|x|)`，即 **E8M0 编码的 `2^floor(log2|x|)`**。

这是 OCP「对 amax 取 floor(log2)」语义的**未归一化单元素形式**：**没有减 emax、没有组内 max 归约、没有 clamp、没有 Inf/NaN 分支**。

## 3. 那能不能用它替掉 `ShiftRights`？

**能，但没必要。** 把 `sharedExp`（`= (E - emax) << 7`）当 bf16 喂进去，`Reg::Cast<fp8_e8m0_t, bfloat16_t>` 直接给出 `E - emax`——因 `sharedExp < 0x8000`，`<<1 >>8` 与 `>>7` **逐位等价**，即它等价于官方/我们第 ② 步最后那句 `ShiftRights(scaleValue, sharedExp, 7)`。

但它内部是 2 条 shift + 1 条 vcvt，**指令数不比 `ShiftRights` 少**，所以不构成性能理由，除非后续发现它能与其他 op 融进同一个 VF。

## 4. 官方逐句序列

权威源：`/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/add_rms_norm_dynamic_mx_quant_common.h`（与 opp 内置 `.../ops_nn/ascendc/dynamic_mx_quant/arch35/dynamic_mx_quant_tail_axis.h` 同构）。

### 4.1 scale 生成 `MxQuantComputeScaleOCP`（下表 = 该函数的 **`for` 循环体** :397-418；**函数定义自 :358 起**，行号随代码变动）

emax 选择（:362-371）：e4m3→`0x0400`、e5m2→`0x0780`、**e2m1→`FP4_E2M1_BF16_MAX_EXP = 0x0100`**、e1m2→`0x0000`。
常量（:71-98）：`NAN_CUSTOMIZATION=0x7f81`、`MAX_EXP_FOR_BF16=0x7f80`、`MAX_EXP_FOR_FP8=0x00ff`、`SPECIAL_EXP_THRESHOLD=0x0040`、`SHR_NUM_FOR_BF16=7`、`BF16_EXP_BIAS=0x7f00`。

```
401 Compare<uint16_t, CMPMODE::NE>(cmpResult,       vdMaxExp, expMask,     preMaskScale);
402 Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
403 Select <uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);           // clamp 下界
404 Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);                           // shared = maxexp - emax
405 ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);            // >>7 -> E8M0 字节
406 Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);          // 非有限 -> 0xFF
410 Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
411 Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
412 Sub(halfScale, scaleBias, sharedExp, preMaskScale);                            // 0x7F00 - shared
413 Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);               // 非有限 -> 0x7F81
414 Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);               // shared==0 -> 0
415 Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);  // shared==0x7F00 -> 0x0040
```

组内 max 阶段（官方 `MxQuantComputeMaxExpOCP`：`add_rms_norm_dynamic_mx_quant_common.h` ops-nn 现版本 :308-355、随包 CANN 同构副本自 :274 起，**行号随代码变动**）：`LoadAlign DIST_DINTLV_B16` → `And 0x7F80` ×2 → `Max` → `ReduceDataBlock<MAX>`。
`Select` 语义：**mask 置位取 src0**。

> ⚠ **锚点更正（M70，据塔内 finding `20260926-reviewer-m60-bug-docs-13-4-mxquantcomputemaxexpocp-381-401-308-356.md`）**：本节原写 `:381-401 / opp tail_axis.h:479-509`，两处在本机**都不可按符号复算** —— ① `:381-401` 落在 `MxQuantComputeScaleOCP` 的寄存器声明/序言与循环体内（`:381` 是该函数的 `Duplicate(maxExpValue, emax)`，`:397` 才是它的 `for`），**不是 max 阶段**；② `dynamic_mx_quant_tail_axis.h`（ops-nn 与随包 CANN 两份，1657/1653 行）里实测 `grep -n 'MxQuantComputeMaxExpOCP'` **两份均 0 命中**（= 该名字不在这两个文件里），它的同一步内联在 `DynamicMxQuantTailAxis::ComputeMaxExpOcpBf16`（ops-nn :469、随包 CANN :479）。⇒ 本行已改为**符号引用**，行号只作查阅提示（本仓规范：引用代码位置以符号为准）。

### 4.2 数据转换 `MxQuantComputeDataFP4` 的 **bf16 分支**（:751-758）

```
753     Mul(vdExp0, vdExp0, halfScaleForMul_asT_X, dataMask1);
754     Mul(vdExp1, vdExp1, halfScaleForMul_asT_X, dataMask1);
755     Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
756     Cast<T_Y, T_X, castTraitRM<roundMode>>(vdExp0FP4, vdExp0, dataMask1);
757     Cast<T_Y, T_X, castTraitRM<roundMode>>(vdExp1FP4, vdExp1, dataMask1);
760-765 StoreAlign<int8_t, ..., DIST_PACK4_B32>(outLocalAddr, ..., OUT_ELE_NUM_ONE_BLK, dataMask1); ×2
```

（`:695-750` 是 **half 输入**的 optimize 路径——它故意转 fp32 乘再回 bf16，与 bf16 路径不同。我们全是 bf16，走 :751-758，**是 bf16 域乘**。）

**rounding 的位置与 mode**：`castTraitRM<RM>`（`ops-nn/norm/norm_common/op_kernel/mx_quant_cast_traits.h:55-57`）

```cpp
template <AscendC::RoundMode RM>
constexpr AscendC::Reg::CastTrait castTraitRM = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RM};
```

与 m2/m5 的 `castTraitRM_Round = {ZERO, UNKNOWN, ZEROING, CAST_ROUND}`（m2:67 / m5:73）**逐字段相同**。rounding 只发生在**最后那一次 fp4 cast**；`round_mode="round"` → `CAST_ROUND`（`dynamic_mx_quant.cpp:102-113` 的 TPL_ROUND 映射）。

**OCP 关系**：`shared_exp = floor(log2(amax)) - emax`，`scale = 2^shared_exp` —— 是 **floor**，不是 ceil/round。三处独立一致：ops-nn aclnn 文档 `quant/dynamic_mx_quant/docs/aclnnDynamicMxQuant.md:35-37`、golden `tests/assets/golden.py:141-154`、cannbot 知识卡 `.../quant/dynamic_mx_quant_950.md:71`。Inf/NaN 组 scale = **0xFF**（E8M0 NaN 码），不是 0x7F。

**官方实现落点**：
- 安装态 kernel 源码：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_mx_quant/`（`dynamic_mx_quant.cpp` + `arch35/*.h`）
- python 注册：`.../ops_nn/dynamic/dynamic_mx_quant.py:193-195`，op type = `DynamicMxQuant`
- IR：`opp/built-in/op_graph/inc/quantize_ops.h:475-486`（x: FP16/BF16；y: FLOAT4_E2M1/E1M2/FLOAT6/FLOAT8；mxscale: FLOAT8_E8M0）
- 开源同源：`/workspace/ops-nn/quant/dynamic_mx_quant/`

## 5. 逐句差异表：官方 vs m2/m5/m13 设备路径

> **列口径（M49 补）**：「我们的设备 kernel」列的锚点**全部落在 `__VEC_SCOPE__` 内、只用 `AscendC::Reg::`（register-based）**——官方同一序列亦然（官方 `MxQuantComputeMaxExpOCP` / `MxQuantComputeScaleOCP` / `MxQuantComputeDataFP4` 各带 `__VEC_SCOPE__`，全文件零经典计算调用，见 `docs/18` §8.2）⇒ 本表是**同族 register 算子序列**的逐步等价，不是"经典 memory-based API 拼出同样结果"。依据 `docs/05` §6.1 规则 ⓐ。
>
> **行号刷新（M49）**：原表锚点写于 M32（±Inf parity 修复）合入**之前**，整体漂移 **+3…+11**（m2 插入 `nanRegTensor`/`Duplicate` 与注释块、m5 同步）——下表已按 main 现版本刷新；引用时以本表/`docs/18` §10.1 为准。另注：原锚点 **`m2:251-252`（乘法）在它自己的 pre-fix 基线上也偏 3**（pre-fix 实为 `:248-249`），属当时的抄写错误，不是 M32 引入的漂移。

| # | 步骤 | 官方（:397-418 / :751-758） | 我们的设备 kernel（register-based） | 位级影响 |
|---|---|---|---|---|
| 1 | 组内 max 指数域 | `And 0x7F80`×2 + `Max` + `ReduceDataBlock<MAX>` | 同（m2:214-217 / m5:277-280） | 无 |
| 2 | emax 常数 | `0x0100`（e2m1） | `Duplicate(maxExpValue, 0x0100)`（m2:197） | 无 |
| 3 | 非有限判定 | `Compare NE(vdMaxExp, 0x7F80)` :401 | 同（m2:220 / m5:283） | 无 |
| 4 | clamp 下界 | `Compare LE`+`Select` :402-403 | 同（m2:221-222 / m5:284-285） | 无 |
| 5 | `shared` | `Sub` :404 | 同（m2:223 / m5:286） | 无 |
| 6 | E8M0 字节 | `ShiftRights 7` :405 | 同（m2:224 / m5:287） | 无 |
| 7 | scale 非有限→0xFF | `Select` :406 | 同（m2:225 / m5:288） | 无 |
| 8 | zeroMask | `Compare NE(sharedExp,0)` :410 | 同（m2:228 / m5:291） | 无 |
| 9 | `halfScale` | `Sub(0x7F00, sharedExp)` :412 | 同（m2:229 / m5:292） | 无 |
| **10a** | **halfScale 非有限覆盖** | `Select(..., nanRegTensor=0x7F81, cmpResult)` **:413** | **同（m2:235 / m5:298 / m13:1469，M32 已补）** | **无**（M32 修复后含 ±Inf 用例位级一致；见 §6） |
| 10b | specialMask | `Compare EQ(sharedExp, 0x7F00)` :411 | 缺失 | 无（不可达） |
| 10c | special 覆盖 | `Select(..., 0x0040, specialDataMask)` :415 | 缺失 | 无（不可达） |
| 11 | halfScale 零覆盖 | `Select(..., 0, zeroMask)` :414 | 同（m2:236 / m5:299） | 无 |
| 12 | 乘 | **bf16×bf16** `Mul` :753-754 | 同（m2:259-260 / m5:322-323） | 无 |
| 13 | Interleave | :755 | 同（m2:261 / m5:324） | 无 |
| 14 | fp4 cast | `Cast<T_Y,T_X,castTraitRM<roundMode>>`，trait={ZERO,UNKNOWN,ZEROING,RM} | `Cast<fp4x2_e2m1_t,bfloat16_t,castTraitRM_Round>`（m2:262-263 / m5:325-326），trait 字面相同 | 无（round 模式同为 CAST_ROUND） |
| 15 | 打包写出 | `DIST_PACK4_B32`，count=`OUT_ELE_NUM_ONE_BLK` | `DIST_PACK4_B32`，count=64（m2:264-267 / m5:327-330） | 无（`VECTOR_LENGTH/2/2 = 64`） |

10b/10c 不可达的论据：`sharedExp ∈ {k*0x80, k=0..253}`，而 `0x7F00 = 254*0x80` 不在集合内。

## 6. 曾经的唯一实际位级差异：10a（**已由 M32 修复**）

组内含 ±Inf（或 NaN）时，官方 `:413` 把 halfScale 强制成 `0x7F81`；原实现没做，得到 `0x7F00 - 0x7E80 = 0x0080`：

- 官方：`Mul(±Inf, NaN) = NaN` → `Cast` → **0.0**
- 原实现：`Mul(±Inf, 2^-126) = ±Inf` → `Cast` → **±6**（饱和；边界表 `reg_vector_compute_interface_boundary_value_summary.md:1803-1804`：inf→6 / -inf→-6）

NaN 输入两边都是 0.0，所以**只有 Inf 会分叉**。

**现状（M32 已并入 main）**：该句已补齐 —— `m2_mxfp4_quant.asc` 的 `MxQuantComputeScale`、`m5_swiglu_quant.asc` 的 `MxQuantComputeScale`（`Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult)`，`nanRegTensor = Duplicate(NAN_CUSTOMIZATION = 0x7F81)`）与 `m13_moe_layer.asc` 的 `MxQuantComputeScale`（三处同源公式），并补了含 ±Inf 的用例 ⇒ **10a 不再是差异**。**我们的 host 参考模型本来就有这个分支**——`m2_mxfp4_quant/check_ref.py` 的 `ocp_numpy`、`m5_swiglu_quant/check_ref.py` 的 `ocp_numpy`、以及 .asc 内 host 参考（`m2_mxfp4_quant.asc` 的 `MxQuantComputeDataFP4` 段、`m5_swiglu_quant.asc` 的 `MxQuantComputeDataFP4` 段）都写了 `if is_inf: hs = 0x7F81`；修复后**设备、host 参考、官方三者一致**。（引用以**符号**为准；括注的 `m2:235`/`m5:298`/`m13:1469`/`m2:343`/`m5:425` 是 M32 写作时的行号口径，随代码变动。）

> ⇒ 现在**可以**主张「量化路径与官方逐句一致」（含 ±Inf 用例位级 PASS，见 `docs/17` §6 对照表）。此前那条"修复前不要声称位级一致"的限制随之解除。

## 7. 量化原语总表

| 原语 | 位置 | 语义 | 融合 scale？ | 与 OCP 关系 |
|---|---|---|---|---|
| `Reg::Cast<fp8_e8m0_t, bfloat16_t, trait>` | `asc/impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_vconv_impl.h:177-186` | 取 bf16 指数域 = `E8M0(2^floor(log2|x|))` | ❌ | OCP 的**未归一化**单元素 floor 幂次；缺 `-emax`、缺组归约、缺 clamp、缺 Inf/NaN 分支 |
| `Reg::Cast<bfloat16_t, fp8_e8m0_t, trait>` | 同上 :55-65 | E8M0→bf16（`vcvt` + `<<7`，结果=inf 时置 NaN 0x7FC0） | ❌ | 反量化侧 |
| `Reg::Cast<fp4x2_e2m1_t/e1m2_t, bfloat16_t, trait>` | 同上（`ppCondition` 分支，`vcvt_ff bf162fp4e2m1/e1m2`） | bf16→fp4x2 | ❌ | 三段式的**第③步内**，已硬件融合 |
| c_api `asc_bfloat162e2m1x2_{rd,rz,rn,ru,rna}` | `asc/include/c_api/reg_compute/reg_convert.h:783-822` | bf16→fp4x2，**5 种舍入可选** | ❌ | 同上；比 Reg 层多给 round mode 选择 |
| `asc_e2m1x22bfloat16` | 同上 :910-917 | fp4→bf16 | ❌ | — |
| `asc_reduce_max_datablock` | `asc/include/c_api/reg_compute/reg_vector.h:896-907` | 数据块 reduce max（**32 字节 = 16×u16 或 8×f32，不是 32 元素**） | ❌ | 组 max 的**半成品**；官方要配 `DIST_DINTLV_B16` + 成对 `Max` 合成 32 元素组 max |
| `Reg::ReduceDataBlock<MAX>` / `ReduceMaxWithDataBlock` | `asc/.../kernel_reg_compute_vec_reduce_impl.h` | 同上（寄存器版） | ❌ | 我们与官方都在用 |
| `QuantMode_t`（Fixpipe） | `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:165-224` | per-tensor / per-channel 量化模式全集 | ❌ | **无任何 block/group/MX 模式，无 e2m1 目标，无 e8m0** |
| `mad_mx` / `load_cbuf_to_ca_mx` / `set_mx_buf_addr` | 同上 :1465 / :1375 / :2267 | cube MX matmul，**消费**预置 scale buffer | ❌（只消费） | OCP 消费侧 |
| `asc_set_deq_scale` | `asc/include/c_api/vector_compute/vector_compute.h:63-68` | 设置 dequant scale（≤`ASC_VDEQ_SIZE` 个/通道） | ❌ 只设不产 | 非 MX |
| `AscendQuant` / `AscendQuantPerGroup`（PER_GROUP 支持 fp4x2_e2m1 目标） | `asc/include/adv_api/quantization/ascend_quant.h:637-646`；impl `ascend_quant_per_group_3510_impl.h:1372-1389,1490-1519` | 分组量化，强制 `groupSize % 32 == 0` | ❌ scale 是**入参**且 dtype 限 `half/float/bfloat16_t`（**不收 E8M0**），**从不输出 scale** | 不能当 MX scale 生产者 |
| `AscendAntiQuant`（E8M0 入参） | `ascend_antiquant.h:203,215`；impl `ascend_antiquant_3510_impl.h:59-96,1724-1779` | E8M0→bf16 解码 + per-group FP4 反量化 | ❌ | 反量化侧 |

### Cast 目标 dtype 白名单（basic_api，`kernel_operator_vec_vconv_impl.h:1345-1362`）

与 fp4/fp8 相关的全部条目：

```
Tuple<fp4x2_e1m2_t, bfloat16_t>   Tuple<fp4x2_e2m1_t, bfloat16_t>     // 量化方向
Tuple<bfloat16_t, fp4x2_e1m2_t>   Tuple<bfloat16_t, fp4x2_e2m1_t>     // 反量化方向
Tuple<float, fp8_e5m2_t>          Tuple<float, fp8_e4m3fn_t>          // + float->hifloat8_t
```

fp4 的源 dtype **只有 2 字节类型**（`sizeof(SRC_TYPE) == 2` 门限，见 `:227`/`:278`）——**没有 half↔fp4、没有 fp32↔fp4、没有任何 e8m0**。Reg 层额外多出 e8m0↔bf16 两条（§2）。c_api `reg_convert.h` 无 e8m0 转换。

**任何 `Cast` 变体都不输出 scale**：`adv_api` 的 `AscendQuant` 全系列（20+ 重载，`ascend_quant.h:58-726`）scale 一律是入参。

## 8. 对开发的直接含义

1. **不要指望找到替代三段式的单指令**——官方自己的 `DynamicMxQuant` / `swiglu_mx_quant` / `add_rms_norm_dynamic_mx_quant` 全是同一套软件序列（cannbot 的 TS 图确认该算子全部节点是 AIV_VECTOR、无 Cube/AIC）。我们与官方同构是**正确路线，不是妥协**。
2. **fp32→fp4 没有硬件 cast**，所以官方和我们一样被迫走「先乘再 cast」；且 bf16 输入必须在 **bf16 域乘**（官方 :753-754）。**切勿**为精度改成 fp32 乘再回 bf16——half 路径（:698-736）才走 fp32，改了就位级不一致。
3. **不要用 `AscendQuantPerGroup` 当 MX scale 生产者**：它不收 E8M0、也不输出 scale。
4. **10a 已由 M32 补齐**（§6）⇒ 现可主张「设备量化路径与官方逐句一致」（含 ±Inf 用例位级 PASS）。
5. **我们的那一列是 register-based**：m2/m5/m13 的量化段全部在 `__VEC_SCOPE__` 内只用 `AscendC::Reg::`，与官方同构（官方亦然，见 `docs/18` §8.2）；依据 `docs/05` §6.1 规则 ⓐ。**不要**把本表读成"经典 memory-based API 拼出同样结果"。

## 9. 存疑与未取证

- 全部结论来自静态阅读，**未在真机跑**（M28 为只读调研）。
- `torch_npu` 的 Python wrapper 源码本机不存在（未安装、无 wheel）⇒ `npu_dynamic_mx_quant` → `aclnnDynamicMxQuant` → op `DynamicMxQuant` 这条链**末端一节未取证**；只看到调用侧 `vllm-ascend/csrc/online_mxfp4_gemm/test/run_mix.py:66`。
- `npu_mx_quant` / `npu_dynamic_double_quant` / `npu_quantize_with_amax` 三个名字全库无命中（对应的应是 `DynamicDualLevelMxQuant` / `QuantMax`，未逐一取证）。
- `QuantMode_t` 在 `#if (__NPU_ARCH__==3510)` 下有枚举**值** 12/13 重复（`VQF322FP8_PRE` 与 `VSHIFTS322S16`）；只报了枚举名，未断言 3510 下的最终数值集。
