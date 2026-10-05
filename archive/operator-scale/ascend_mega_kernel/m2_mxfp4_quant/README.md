# M2：MXFP4 激活量化（纯 AIV 核）

单 AIV 核将 bf16 激活 `x [m, K]` 在线量化为 OCP MXFP4（e2m1 nibble 打包 + E8M0 scale），输出与 CPU 位精确参考（`quant_ref.py` 算法 + 官方非有限分支）**逐字节一致**（**104 组用例全部 `bit-exact`**：50 组基线（含 m=1 与 m=256、5 组随机种子）+ 54 组含 ±Inf/NaN 的非有限用例，见「非有限用例」一节）。

donor：`/workspace/vllm-ascend/csrc/online_mxfp4_gemm/src/online_mxfp4_quant.asc`（MxQuantComputeScaleFused / MxQuantComputeDataFP4，已 bit-exact 验证过 torch_npu.npu_dynamic_mx_quant）与 `test/quant_ref.py`（CPU 位精确参考，host 侧已逐行移植到 C，另经 `check_ref.py` numpy 转写交叉验证）。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m2_mxfp4_quant/build -S m2_mxfp4_quant -DCMAKE_BUILD_TYPE=Release
cmake --build m2_mxfp4_quant/build -j4
./m2_mxfp4_quant/build/m2_mxfp4_quant
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。

## 输入输出 layout

| 张量 | 形状 | 类型 | 说明 |
|---|---|---|---|
| x | `[m, K]` | `bfloat16` | 激活，行主序；K 编译期（640=down_proj / 2560=gate_up），m 运行时 1..256 |
| qx | `[m, K/2]` | `uint8` | 打包 fp4：`byte = lo \| (hi << 4)`，lo=偶数 k、hi=奇数 k |
| scale | `[m, K/32]` | `uint8` | E8M0（bias-127），每 32 个 K 元素 1 个 scale |

语义对标 `torch_npu.npu_dynamic_mx_quant(dst_type=fp4x2_e2m1, round_mode="round", scale_alg=0/OCP, group=32)`。e2m1 值表：`0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6`（CAST_ROUND 远离零舍入）。

## 量化语义（OCP，按 quant_ref.py）

```
maxexp = max(bf16 bits & 0x7F80)          # 每 32 组的 bf16 最大指数域
shared = max(maxexp, 0x0100) - 0x0100     # = (floor(log2(amax)) - 2 + 127) << 7
scale  = shared >> 7                      # E8M0 字节（bf16 Inf/NaN 组 -> 0xFF）
halfSc = (组 maxexp == 0x7F80) ? 0x7F81(NaN) : (shared == 0 ? 0 : 0x7F00 - shared)
xs     = RNE_bf16(x * halfScale)          # bf16 乘法，RNE
q      = e2m1 CAST_ROUND(xs)              # 远离零，0.25/0.75/... tie 归上一档
```

组内含 ±Inf/NaN 时（`maxexp == 0x7F80`）`halfScale` 被覆盖成 bf16 NaN `0x7F81`，于是 `Mul(±Inf, NaN) = NaN → Cast<fp4> = 0.0`，整组 nibble 归零（配合 `scale = 0xFF`）。这一句来自官方 `:413`，见下节对照表 —— **M32 之前本核缺失该句**，会对 ±Inf 元素 `Mul(±Inf, 2^-126) = ±Inf → Cast` 饱和成 ±6。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV），blockDim=1（单核，正确性优先）；m 运行时 1..256，逐行循环、行内逐 tile（TILE=256 bf16 = 8 组）。
- **数据通路**（全基础 API，无 TPipe / TBuf / TQue / AllocTensor）：
  `GM --DataCopy(MTE2)--> UB(x) --Reg::LoadAlign/Max/ReduceDataBlock/... (VEC)--> 寄存器 --Reg::StoreAlign--> UB(out/scale) --DataCopy/DataCopyPad(MTE3)--> GM`
  全部 UB buffer 用 `LocalTensor(position, offset, size)` 编译期静态分配（偏移 32B 对齐，供 DataCopy）。
- **同步只用 BufferID**（`GetBufInternal/RlsBufInternal`，= get_buf/rls_buf；不用 set_flag/wait_flag，不挂 PIPE_S）。生产者 release 一律阻塞释放（`mode=false` = CANN `ASC_LOCK_BLOCK` 默认；`true` = `NON_BLOCK`；两种模式都等本 pipe 已发射指令落地，`true` 额外等此前同 id 的释放 ⇒ 更保守），保证 UB 写对跨 pipe 读可见：

| BufferID | 交接 | 含义 |
|---|---|---|
| 0 (BUF_X) | MTE2 → V | x tile 就绪 |
| 1 (BUF_O) | V → MTE3 | out/scale 就绪（同时挡住下一 tile 的覆写） |

- **K=640 尾 tile（128 元素）**：不用 donor 的 `len/256==0` 尾块路径（该路径不执行任何计算，donor 实际只支持 K%256==0），改为**重叠回退窗口** `[K-256, K)`：与前一 tile 重叠的 128 元素 group 对齐（128%32==0），两次量化结果逐位一致，只是重写一遍。无 mask、无越界读（正好读到行尾，host 无需 padding 行）。
- **Nd2Nz 行数=1 quirk（docs/05 §6）**：本核按行 1D 连续拷贝（每行每 tile 512B），不使用 Nd2Nz/Dn2Nz，quirk 天然规避。
- **寄存器内派生 scale**：maxexp 全程留在寄存器（3510 上 reduced maxExp 经 UB 回读与后续向量 load 不相干，会读出 0，见 donor 注释），只把 E8M0 scale（DIST_PACK_B16）与 bf16 halfScale 写 UB；scale 仅 8B/ tile，MTE3 走 `DataCopyPad(Compact)` 精确写字节。
- **3510 quirk（调试实测）**：UB 偏移非 32B 对齐时 UB→GM DataCopy 直接 vector core exception（本工程 UB 布局全部 32B 对齐）。

## 与官方实现的逐句对照（M32）

权威源：`/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/add_rms_norm_dynamic_mx_quant_common.h`
（安装态同源副本：`/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/dynamic_mx_quant/arch35/dynamic_mx_quant_tail_axis.h`）。
`scale` 生成见 `:397-418`（`MxQuantComputeScaleOCP`），数据转换见 `:751-758`（`MxQuantComputeDataFP4` 的 bf16 分支）。
本核的 `MxQuantComputeScale` / `MxQuantComputeDataFP4` 与之逐句同构：

| # | 步骤 | 官方（path:line） | 本核（函数 / 语句） | 位级影响 |
|---|---|---|---|---|
| 1 | 组内 maxexp（指数域） | `And 0x7F80`×2 + `Max` + `ReduceDataBlock<MAX>` | 同 | 无 |
| 2 | emax 常数 e2m1 | `Duplicate(0x0100)`（`FP4_E2M1_BF16_MAX_EXP`） | 同 | 无 |
| 3 | 非有限判定 | `Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, 0x7F80)` **:401** | 同 | 无 |
| 4 | clamp 下界 | `Compare LE` + `Select` **:402-403** | 同 | 无 |
| 5 | `sharedExp` | `Sub` **:404** | 同 | 无 |
| 6 | E8M0 字节 | `ShiftRights(scaleValue, sharedExp, 7)` **:405** | 同 | 无 |
| 7 | scale 非有限→0xFF | `Select(scaleValue, scaleValue, fp8Nan, cmpResult)` **:406** | 同 | 无 |
| 8 | zeroMask | `Compare NE(sharedExp, 0)` **:410** | 同 | 无 |
| 9 | `halfScale` | `Sub(halfScale, 0x7F00, sharedExp)` **:412** | 同 | 无 |
| **10a** | **halfScale 非有限覆盖** | `Select<uint16_t>(halfScale, halfScale, nanRegTensor=0x7F81, cmpResult)` **:413** | **M32 补上（本核此句）** | **有：修复前 ±Inf → ±6 饱和** |
| 10b | `specialMask` | `Compare EQ(sharedExp, 0x7F00)` **:411** | 缺（不可达，见下） | 无 |
| 10c | special 覆盖 | `Select(halfScale, specialExp=0x0040, halfScale, specialMask)` **:415** | 缺（不可达） | 无 |
| 11 | halfScale 零覆盖 | `Select(halfScale, halfScale, 0, zeroMask)` **:414** | 同 | 无 |
| 12 | 乘 | bf16×bf16 `Mul` **:753-754** | 同 | 无 |
| 13 | `Interleave` | **:755** | 同 | 无 |
| 14 | fp4 cast | `Cast<T_Y,T_X,castTraitRM<roundMode>>`，trait `{ZERO, UNKNOWN, ZEROING, RM}` | `castTraitRM_Round`（round 模式同为 `CAST_ROUND`） | 无 |
| 15 | 打包写出 | `StoreAlign<..., DIST_PACK4_B32>`，count = `OUT_ELE_NUM_ONE_BLK` | 同（count=64） | 无 |

`Select` 的 mask 语义（置位取 src0）在官方同一序列中多处使用（`:403` clamp、`:406` scale 覆盖、`:413` halfScale 覆盖），本核逐句同源。

**10b/10c 在本工程不可达（可达性论据）**：`specialDataMask = Compare EQ(sharedExp, 0x7F00)`，其中 `sharedExp = vdMaxExp - 0x0100`。各变量的取值集要分清：

| 变量 | 来源 | 取值集 |
|---|---|---|
| `vdMaxExp`（`And 0x7F80` 之后、clamp 之前） | `And(vdExp, 0x7F80)` | `{k*0x80, k = 0..255}`（上界 `0x7F80`，故 `0x7F80` 即 ±Inf/NaN 组可达） |
| `vdMaxExp`（clamp 之后，即 `Sub` 的被减数） | `Select(vdMaxExp, 0x0100, vdMaxExp, invalidDataMask)` | `{k*0x80, k = 2..255}`（clamp 只抬升下界，`k = 0,1` 被抬到 `0x0100`） |
| `sharedExp` | `Sub(vdMaxExp, 0x0100)` | `{k*0x80, k = 0..253}`（上界 `0x7E80`） |

`0x7F00 = 254*0x80` 需要 `vdMaxExp == 0x8000`，而 `And 0x7F80` 的上界是 `0x7F80 < 0x8000`，故 `sharedExp == 0x7F00` **恒为假** → `:411` 的 mask 恒空、`:415` 的 `Select` 永不生效。因此本核不写这两句不产生任何位级差异（官方那两句对本路径同样是防御性死代码）。**若将来 emax 或指数掩码改变（例如 fp8 目标、或 `And` 掩码放宽到 `0x8000`），必须重新评估。**

## 非有限（±Inf/NaN）用例（M32）

`RunCase(..., infKind)` 在既有确定性生成之上，按 group 注入非有限元素（注入组固定 `grp%4==1`，其余组保持正常取值，用于验证修复不外溢）：

| infKind | 用例（任务 ①②③） | 注入内容 |
|---|---|---|
| 1 | 整组 amax=+Inf | 注入组 32 个元素全 `0x7F80` |
| 2 | 组内只有部分元素 Inf | 注入组恰有 `+Inf` 与 `-Inf` 各一个（位置随 Hash 变化），其余正常 |
| 3 | NaN 组 | 注入组全 `0x7FC0`/`0xFFC0` 交替 |

覆盖 `K ∈ {640, 2560} × m ∈ {1, 17, 256} × seed ∈ {0,1,2} × infKind ∈ {1,2,3}` = **54 组**，与 host C 参考（官方语义，含 `:413` 分支）**逐字节断言**；判据另含「注入组的 device nibble 必须全 0」（修复前是 ±6 饱和码）。

| 运行 | 基线判据（inf=0） | 非有限判据 | 总计 | 备注 |
|---|---|---|---|---|
| 修复前（删除 `Select(..., nanRegTensor, cmpResult)` 一句后重编） | 50/50 PASS | **0/54 PASS**（inf=1: 0/18、inf=2: 0/18、inf=3: 18/18） | 68/104 | inf=1/2 device 出 `0x77`/饱和码 ±6；**inf=3（NaN）修复前就 PASS**——NaN 输入两边都是 0.0，只有 Inf 分叉 |
| 修复后 | 50/50 PASS | **54/54 PASS** | **104/104 PASS** | 注入组 nibble 非零 0/（组数×32），device `scale=0xFF` 组数与参考一致 |

复跑（`evidence/` 内有全量日志）：

```bash
# 修复前（临时副本，删掉那一句 Select 后重编）
cp -r m2_mxfp4_quant /tmp/m2_prefix && sed -i '/nanRegTensor, cmpResult/d' /tmp/m2_prefix/m2_mxfp4_quant.asc
cmake -B /tmp/m2_prefix/build -S /tmp/m2_prefix -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/m2_prefix/build -j2
/tmp/m2_prefix/build/m2_mxfp4_quant            # -> 判据行 68 PASS / 36 FAIL（inf=1/2 全 FAIL）
```

## 校验方法与结果

host 侧确定性随机生成 bf16 激活（公式哈希，无 rand 状态），覆盖：全零组（+0，走 shared==0 路径）、精确 tie 组（amax=4.0 使 halfScale=1.0，元素命中 0.25/0.75/1.25/1.75 中点与 2.0 精确档，含负 tie）、随机组（组内指数窗 ≤31 档、跨组宽动态范围，覆盖 xs∈[4,8) 饱和区）。组 maxexp 指数域取 3..126：杜绝 -0/次正规乘积（见"已知限制"）。

校验链（四层全部逐字节一致）：

1. **NPU kernel vs C 参考**（`quant_ref.py` 逐行移植 + 官方 `:413` 非有限分支，内建于可执行文件）：基线 `K ∈ {640, 2560} × m ∈ {1, 2, 17, 128, 256} × seed ∈ {0..4}` **50 组** + 非有限 `K ∈ {640, 2560} × m ∈ {1, 17, 256} × seed ∈ {0,1,2} × infKind ∈ {1,2,3}` **54 组**，共 **104 组全部 PASS**，进程退出码 0（日志 `evidence/run_after.log`；修复前同一套判据 68/104，日志 `evidence/run_prefix_nofix.log`）。
2. **C 参考 vs quant_ref.py 算法**：`check_ref.py`（`quant_ref.py` 的 numpy 转写，torch 的 bf16 视图/RNE 用 `tools/golden/moe_block_ref.py` 的纯 numpy 等价物替代）对 dump 用例逐字节一致 —— 含非有限用例（`evidence/check_ref_inf.log`：`K ∈ {640,2560} × m ∈ {1,17} × inf ∈ {0..3}` 共 16 组全 True）。
   M32 一并修了该脚本的两个非有限缺陷（修复前 `evidence/check_ref_inf_prefix_numpy.log`：Inf 组算出 `0x77`、NaN 组算出 `0x7f`）：① 本地 bf16 解码器由算术式改为**位精确**（`bits << 16` 重解释；算术式会把 `0x7F81` 解成 `+Inf`）；② `np.digitize(NaN, mids) = 7`（错，会得 +6）显式改判为 `Cast(NaN) = 0.0`，符号取 `signbit`（与设备/C 参考一致）。
3. **NPU kernel vs quant_ref.py 算法**：`M2_DUMP=1` 落盘 device 输出后由同一脚本逐字节比对，104 组全部 bit-exact。
4. **独立见证（另一个 agent 的实现）**：`M2_DUMP=1` 的全部 104 个 device dump 与 `tools/golden/moe_block_ref.py` 的 `quantize_ocp()`（M26 写的官方 ops-nn 序列逐句 numpy 转写）**逐字节一致**（K=640 52/52、K=2560 52/52，日志 `evidence/ocp_independent_witness.log`；该参考函数出自 M26 分支 `6f68f8e`，合入 main 后即可复跑）。

交叉验证用法（需在 build/dump 之类的临时目录运行，避免污染仓库）：

```bash
cd m2_mxfp4_quant/build && mkdir -p dump && cd dump
../../build/m2_mxfp4_quant dump 640 17 0        # x.bin / qx.bin / scale.bin（C 参考）
../../build/m2_mxfp4_quant dump 640 17 0 1      # 第 5 个参数 infKind：1=整组 +Inf / 2=部分 ±Inf / 3=NaN 组
/usr/local/python3.12.13/bin/python3 ../../check_ref.py 640 17
M2_DUMP=1 ../../build/m2_mxfp4_quant            # 额外落盘 *_qx_device.bin / *_scale_device.bin（含 inf=1..3 用例）
```

## 已知限制 / 后续

- 单核正确性优先：未做多 AIV 核行切分与流水化（行间天然独立，后续可按行 stride 分核）；未做 GM 预取双缓冲。
- m > 256 未验证（任务书范围 1..256）；K 仅支持 128 的倍数且 ≥256（模型两形状 640/2560 已覆盖）。
- **`xs` 恰为 ±0 的符号角落（与 M32 无关的既有差异）**：`halfScale == 0` 的组里 `x * 0` 的符号由硬件保留，`Cast(-0)` 给 nibble `0x8`，而 C 参考/quant_ref 的字面规则 `xs < 0` 给 `+0`。仅当组 `maxexp ≤ 0x0100` 且组内含负元素时可达，真实激活不可达；生成器已按 m2 原有策略避开（零组全 `+0`，非零组 `maxexp ≥ 3`）——M32 的非有限用例生成同样遵守该约束（否则判据会被这个角落污染）。
- **NaN 符号/payload 角落（M32 实测，同为既有差异）**：设备 `Cast<float→bf16>(NaN)` 给 `0x7FFF`（正值、payload 全 1），丢符号；x86 host 保留 NaN 符号。对 `-NaN` 输入，设备与（保留符号的）host 参考在「零的符号位」上可能差一位（`0x0` vs `0x8`）。仅 NaN 输入可达；`Cast(NaN) → 0.0` 的量级语义两边一致。
- 与 m1 相同的 e2m1 NaN 编码（0x7/0xF）在生成侧不会出现；真实权重若含 NaN 需另行定义语义。
