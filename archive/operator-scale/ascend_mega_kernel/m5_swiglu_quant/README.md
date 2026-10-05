# M5：SwiGLU 融合 MXFP4 量化 epilogue（纯 AIV 核）

GEMM#1 的 epilogue 融合（docs/09 §3 结论：SwiGLU 必须在量化前）：单 AIV 核将 bf16 gate_up GEMM 输出 `x [m,1280]`（gate|up 拼接）逐行做 **SwiGLU → bf16 [m,640] → 在线 MXFP4 量化**（原地接力，quant 不进 GM），输出 `qx [m,320] u8 + scale [m,20] u8(E8M0)`，量化语义与 **m2_mxfp4_quant 逐行一致**（OCP，bit-exact 对标 `torch_npu.npu_dynamic_mx_quant round_mode="round", scale_alg=0, group=32`）。

donor：`ops-nn/activation/swiglu_group_quant` arch35（`VFSwiGlu` base.h:186-194 / `VFComputeMaxExpMXFP4Vf` :281-333 / `VFComputeScaleMXFP4Vf` :350-409 / `VFComputeDataMXFP4Vf` :428-482，调度参考 `swiglu_mxfp4_quant_perf.h:117-192` 原地接力形态）+ `ops-nn/activation/swi_glu` impl.hpp:54-82 五元组；quant 三段式为 m2 kernel 的逐行移植。CPU 对照：`tools/golden/moe_block_ref.py`（bf16 RNE 工具）+ m2 校验链思路。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m5_swiglu_quant/build -S m5_swiglu_quant -DCMAKE_BUILD_TYPE=Release
cmake --build m5_swiglu_quant/build -j4
./m5_swiglu_quant/build/m5_swiglu_quant
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。host 侧 CPU 参考的 `expf` 需 libm（CMakeLists 已链接）。

## 输入输出 layout

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | `[m, 1280]` | `bfloat16` | gate_up GEMM 输出，行主序；前 640 列 = gate，后 640 列 = up；m 运行时 1..256 |
| swigluOut | `[m, 640]` | `bfloat16` | SwiGLU 结果（校验/调试输出；融入 layer kernel 时该 GM 写出可裁剪，donor 的 `outputOrigin` 同款开关语义） |
| qx | `[m, 320]` | `uint8` | 打包 fp4：`byte = lo \| (hi << 4)`，lo=偶数 k、hi=奇数 k |
| scale | `[m, 20]` | `uint8` | E8M0（bias-127），每 32 个元素 1 个 scale；kernel 内部按 uint16 对经 DataCopyPad 写出，字节序与 `[m,20]` 展一致（低字节=偶数 group） |

## 融合结构与数据通路

每行每 256-tile（8 个 quant 组）一条通路，SwiGLU 结果留 UB、quant 原地接力：

```
GM --DataCopy(MTE2)--> UB(gate|up tile 各 256)
   --SwiGLU 五元组(VEC, fp32 寄存器)--> UB(swiglu bf16 tile)      <-- 接力点：quant 直接读本 UB
   --MaxExp/Scale/DataFP4(VEC, 同 m2)--> UB(qx tile 128B + scale tile 8B)
   --DataCopy/DataCopyPad(MTE3)--> GM(swigluOut 512B / qx 128B / scale 8B)
```

- **SwiGLU 五元组**（donor `VFSwiGlu`，fp32 路径）：`silu(g)=g/(1+exp(-g))`，`y=silu(gate)*up`；bf16→fp32 后计算，RNE（CAST_RINT）回 bf16。每迭代 64 元素（fp32 寄存器宽度），256-tile = 4 迭代；640 = 10×64 整除，**无尾块、无 mask**。
- **quant 三段式**：m2 的 `MxQuantComputeScale` / `MxQuantComputeDataFP4` 逐行移植（maxexp 全程留寄存器，只写 E8M0 scale + bf16 halfScale 两个结果）。

## 量化语义（OCP，与 m2 逐行一致）

```
maxexp = max(bf16 bits & 0x7F80)          # 每 32 组的最大指数域
shared = max(maxexp, 0x0100) - 0x0100
scale  = shared >> 7                      # E8M0 字节（bf16 Inf/NaN 组 -> 0xFF）
halfSc = (组 maxexp == 0x7F80) ? 0x7F81(NaN) : (shared == 0 ? 0 : 0x7F00 - shared)
xs     = RNE_bf16(x * halfSc)             # bf16 乘法，RNE
q      = e2m1 CAST_ROUND(xs)              # 远离零，0.25/0.75/... tie 归上一档；-0 保留符号
```

组内含 ±Inf/NaN 时 `halfSc` 覆盖为 bf16 NaN `0x7F81`（官方 `:413`，**M32 之前本核缺失该句**，会对 ±Inf 饱和成 ±6），`Mul(±Inf, NaN) = NaN → Cast<fp4> = 0.0` 使该组 nibble 归零。

**-0 符号语义**（对 quant_ref.py 字面规则的唯一细化）：e2m1 符号取 `signbit(xs)` 而非 `xs<0`。硬件 `CAST_ROUND(-0)` 给 `-0`（nibble 8），`xs<0` 规则给 `+0`；m2 通过生成器只产 +0 零组避开该角落（其 README 已记录），而 m5 的 SwiGLU 乘积下溢会**自然产生 -0**（实测 kernel 与 host silu 逐位一致，含 -0 与次正规下溢），故参考侧直接对齐硬件，使全链对 -0 也逐字节一致。

## 与官方实现的逐句对照（M32）

权威源：`/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/add_rms_norm_dynamic_mx_quant_common.h`（`scale` 生成 `:397-418`，`MxQuantComputeDataFP4` 的 bf16 分支 `:751-758`；安装态同源副本在 `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_mx_quant/arch35/dynamic_mx_quant_tail_axis.h`）。本核的 quant 三段式与之逐句同构（第 1-15 步对照表见 `m2_mxfp4_quant/README.md`，本核完全一致，差别仅在本核的源 tile 由 SwiGLU 就地产生）：

| # | 步骤 | 官方（path:line） | 本核 | 位级影响 |
|---|---|---|---|---|
| 1-9 | 组 maxexp / emax / 非有限判定 / clamp / shared / E8M0 字节 / scale 0xFF / zeroMask / `halfScale` | `:401-412` | 同（`MxQuantComputeScale`） | 无 |
| **10a** | **halfScale 非有限覆盖** | `Select<uint16_t>(halfScale, halfScale, nanRegTensor=0x7F81, cmpResult)` **:413** | **M32 补上** | **有：修复前 ±Inf → ±6 饱和** |
| 10b/10c | `specialMask`(`shared==0x7F00`) + `0x0040` 覆盖 | `:411` / `:415` | 缺（**不可达**，论据同 m2：`sharedExp ∈ {k*0x80, k≤253}`，`0x7F00` 需 `vdMaxExp==0x8000` 而掩码上界 `0x7F80`） | 无 |
| 11-15 | halfScale 零覆盖 / bf16×bf16 乘 / Interleave / fp4 cast / PACK4 写出 | `:414` / `:753-758` | 同 | 无 |

SwiGLU 段（`:695-750` 是官方 **half 输入的 optimize 路径**，故意转 fp32 乘再回 bf16；本核沿用的是 `:751-758` 的 bf16 路径，**bf16×bf16 乘**）：本核 `SwigluComputeTile` 的 fp32 五元组与 donor `VFSwiGlu` 同序，结果 RNE 回 bf16 后再进 quant；**不要**改成 fp32 乘上 quant（那会与官方 bf16 路径位级不一致）。

## 非有限（±Inf/NaN）用例（M32）

`RunCase(..., infKind)` 在 gate|up 上注入（注入组固定 `grp%4==1`，其余组正常）：

| infKind | 用例（任务 ①②③） | 注入内容（gate / up） |
|---|---|---|
| 1 | 整组 amax=+Inf | 全组 `gate=+Inf, up=1.0` → SwiGLU 输出整组 +Inf |
| 2 | 组内只有部分元素 Inf | 一处 `(+Inf, 1.0) → +Inf`，一处 `(4.0, -Inf) → -Inf`，其余正常值 |
| 3 | NaN 组 | 全组 `gate=NaN(±交替), up=1.0`（一处 up=NaN） |

覆盖 `m ∈ {1, 17, 256} × seed ∈ {0,1,2} × infKind ∈ {1,2,3}` = **27 组 × 2 判据（SwiGLU + 量化）= 54 判据**。

| 运行 | 基线判据（inf=0） | 非有限判据 | 总计 | 备注 |
|---|---|---|---|---|
| 修复前（删掉 `Select(..., nanRegTensor, cmpResult)` 后重编） | 50/50 PASS | **36/54 PASS**（(a) SwiGLU 全 PASS；(b) 量化 inf=1/2 全 FAIL：各 9 条；inf=3 全 PASS：18 条。即 36 PASS / 18 FAIL） | 86/104 | inf=1/2 (b) device 出 `0x77` 饱和码；inf=3（NaN）修复前已 PASS |
| 修复后 | 50/50 PASS | **54/54 PASS** | **104/104 PASS** | 注入组 nibble 非零 0/（组数×32） |

（表格口径：每格都是**判据行**数 = `: PASS` / `: FAIL` 结尾的行数；总计 = 基线 50 + 非有限 54；修复前 50 + 36 = 86。）

SwiGLU 判据对非有限值的口径：同类即等（NaN 之间不比符号/payload、±Inf 各自同档），另加「有限/非有限归类必须一致」的显式计数（两边都为 0 才是 PASS）。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV，3510 quirk 要求），blockDim=1（单核，正确性优先）；m 运行时 1..256 逐行循环（行间无依赖）。
- **资源全静态**：UB 偏移/BufferID 编译期静态表；全部基础 API + `__ubuf__` 裸指针，无 TPipe/TBuf/TQue/AllocTensor；UB 偏移全部 32B 对齐（M9 实测 quirk：非 32B 对齐 UB→GM DataCopy 直接 exception）；只用 1D 连续 DataCopy（不触 Nd2Nz 行数=1 quirk）。
- **同步只用 BufferID**（`GetBufInternal/RlsBufInternal`，release 一律阻塞释放 `mode=false` = CANN `ASC_LOCK_BLOCK` 默认、`true`=`NON_BLOCK`，两种模式都等本 pipe 已发射指令落地，不用 set_flag/wait_flag，不挂 PIPE_S）：

| BufferID | 交接 | 含义 |
|---|---|---|
| 0 (BUF_X) | MTE2 → V | gate|up tile 就绪 |
| 1 (BUF_O) | V → MTE3 | swiglu/qx/scale tile 就绪（兼挡下一 tile 覆写） |

- **尾 tile（640 = 256+256+128）**：同 m2 的重叠回退窗口 `[384, 640)`（group 对齐，重叠区 SwiGLU/量化结果逐位一致只是重写一遍），无 mask、无越界读。
- **UB 布局**（字节偏移）：`UB_X@4096`（gate 512B + up 512B + 512B 预读 slack，供 UNPACK_B16 整寄存器加载越界预读，数据被 mask 丢弃）、`UB_SWIGLU@5632`、`UB_QX@6144`、`UB_SCALE@6272`、`UB_HALF@6304`，共 <6KB / 248KB。

## 校验方法与结果

host 确定性生成 bf16 gate|up（Hash3 公式，无 rand 状态），按输出 32-group 选模式：全零输出组（up=±0 且符号匹配 gate → 严格 +0，走 scale=0/halfScale=0 路径）、小整数档、up=精确幂（mantissa 平移，tie 统计命中）、双随机宽动态范围（组内指数窗 ≤31 档，覆盖 xs∈[4,8) 饱和区；随机模式会产生 -0/次正规下溢乘积，覆盖 -0 角落）。m ∈ {1, 2, 17, 128, 256} × seed ∈ {0..4} 共 **25 组**：

**(a) SwiGLU bf16 vs CPU 参考**（host `expf` 同次序 fp32 五元组 + RNE）：25/25 组 **maxULP=0、逐位一致**（含 -0 与次正规下溢；容差为 bf16 网格 ≤1 ULP，实测为 0）。

**(b) MXFP4 vs CPU 量化参考**（C 移植 quant_ref.py 算法 + 官方 `:413` 非有限分支，对 **kernel 自己的 bf16 输出**量化）：25/25 组 qx+scale **逐字节一致**。

**四层校验链**（复用 m2 思路，`check_ref.py` 为 numpy 转写，`f32_to_bf16` 取自 `tools/golden/moe_block_ref.py`）：

1. **NPU kernel vs C 参考**（可执行文件内建，如上 (a)(b)）：基线 25 组 × 2 判据 = **50 判据**，加非有限 27 组 × 2 判据 = **54 判据**，共 **104 判据全部 PASS**，退出码 0（日志 `evidence/run_after.log`；修复前同一套判据 **86/104**，即非有限 36/54 PASS + 基线 50/50，日志 `evidence/run_prefix_nofix.log`）。
2. **C 参考 vs numpy 算法**：`m5 dump <m> <seed> [inf]` 落盘 C 参考链，`check_ref.py` 逐字节/ULP 比对：(a1)(b1) 全 PASS —— 含非有限用例（`evidence/check_ref_inf.log`：`m=17 seed=0 inf ∈ {0..3}`）。
   M32 一并修了该脚本的非有限缺陷：① 量化/反量化改**位精确** bf16 解码（算术式解码会把 bf16 NaN `0x7F81` 解成 `+Inf`）；② `np.digitize(NaN, mids) = 7` 显式改判为 `Cast(NaN) = 0.0`；③ 回写 bf16 时不把 NaN 规范化成 `0x7FC0`（保留符号位，与设备/C 参考的 `Cast(-NaN) = -0` 一致）；④ 非有限值的 ULP 判据按「同类即等 + 归类必须一致」。
3. **NPU kernel vs numpy 算法**：`M5_DUMP=1` 落盘 device 输出，同一脚本比对：(a2)(b2) 全 PASS（含非有限用例）。
4. **独立见证（另一个 agent 的实现）**：`M5_DUMP=1` 的全部 52 个用例（量化段输入取 device 自己的 SwiGLU 输出）与 `tools/golden/moe_block_ref.py` 的 `quantize_ocp()`（M26 写的官方 ops-nn 序列逐句 numpy 转写）**逐字节一致**：52/52 PASS（日志 `evidence/ocp_independent_witness.log`；该参考函数出自 M26 分支 `6f68f8e`，合入 main 后即可复跑）。

交叉验证用法（在 build/dump 之类的临时目录运行，避免污染仓库）：

```bash
cd m5_swiglu_quant/build && mkdir -p dump && cd dump
for s in 0 1 2 3 4; do for m in 1 2 17 128 256; do ../m5_swiglu_quant dump $m $s 0; done; done
../m5_swiglu_quant dump 17 0 1        # 第 4 个参数 infKind：1=整组 +Inf / 2=部分 ±Inf / 3=NaN 组
M5_DUMP=1 ../m5_swiglu_quant          # 全部用例（含 inf=1..3）额外落盘 *_device.bin
/usr/local/python3.12.13/bin/python3 ../../check_ref.py 256 4 0    # 或任意 (m, seed, inf)
```

**实测精度备注**：在全部测试数据上，硬件向量 `Exp` 与 host `expf`/numpy `np.exp` 的 fp32 结果逐位一致，因此 numpy 全链（numpy silu→numpy quant）对 device qx 也达到 100% 字节一致（`check_ref.py` 的 info 行）。这说明在本数据域上融合链可以做全链 bit-exact 断言；(a) 的 1-ULP 容差是跨平台保险带。**例外**：inf=3（NaN 组）的 info 行只有 4080/5440 字节一致 —— 设备 `Cast<float→bf16>(NaN)` 给 `0x7FFF`（正值、丢符号），x86 host 保留 NaN 符号，故该组「零的符号位」在两条链上不同（见「已知限制」）。

## 已知限制 / 后续

- 单核正确性优先：未做多 AIV 核行切分与流水化（行间天然独立，后续可按行 stride 分核）；未做 GM 预取双缓冲；swigluOut 的 GM 写出在 layer kernel 中可裁剪（relay 语义已在 UB 内完成）。
- m > 256 未验证（任务书范围 1..256）；输入宽度固定 1280（gate|up = 2×640，模型形状）。
- 真实激活幅值 O(1) 不会触发 SwiGLU 下溢 -0，该角落由生成器随机模式覆盖验证；`Cast(-0) = -0` 与 `xs<0` 规则在**恰为 -0** 处不同（符号位），生成器已避开「`halfScale==0` 的组里含负元素」这一可达路径（M32 的非有限用例生成同样遵守）。
- **NaN 符号/payload 角落（M32 实测）**：设备 `Cast<float→bf16>(NaN)` 给 `0x7FFF`（正 NaN），x86 host 保留符号（`0xFFC0`/`0x7FC0`）；因此对 NaN 输入，设备与 host 参考在量化输出的「零的符号位」上可能差一位（设备全 `0x0`、host 的 -NaN 给 `0x8`）。量级语义（`Cast(NaN) → 0.0`）两边一致，(a) 判据按「同类即等」不计该差异。
- donor `VFComputeScaleMXFP4Vf` 对 `shared==0x7F00` 极值组有额外特判（halfScale 置 0x0040）：**不可达** —— `sharedExp ∈ {k*0x80, k≤253}`，`0x7F00` 需 `vdMaxExp == 0x8000`，而指数掩码上界 `0x7F80`；本工程与官方对本路径同为死代码（M32 论据，见 m2 README）。
