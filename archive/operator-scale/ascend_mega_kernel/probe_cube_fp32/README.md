# probe_cube_fp32 —— M106：cube（AIC）吃 fp32 操作数的能力 / 精度形态 / 吞吐标定

> **裁决一句话**：`roofline_model.md:28` 的 **"CUBE Peak (FP32) ~24 TFLOPS" 是对的，且它指的就是
> 「fp32 操作数」**；`m18_gdn_prefill/README.md:371` 的 **"3510 的 cube 只能吃 bf16 操作数" 这一句
> 被证伪**（cube 能吃 fp32 操作数，编译/运行/数值全成立）。但 m18 那句话的**意图**（cube 对 bf16 友好、
> fp32 挨罚）在数量级上是对的：**fp32 操作数下 Mmad 吞吐 = bf16 的 1/15.88**（归档 3 次：15.872 / 15.879 / 15.880）。
>
> **对 `docs/19` §7 第 5 条（第三选项）**：见 §6 —— **前提成立、数字成立，但"cube 吞吐"要改口径**：
> fp32 操作数下约 **23 TFLOPS = bf16 的 1/16、且低于 VEC fp32 峰值 28 TFLOPS**。

本文件的所有读数都由本目录的探针产出，可复跑；命令与判据见 §7。

---

## 1. 两条矛盾读数的定位与逐字引用（任务 1）

### 1.1 读 A（`roofline_model.md`）—— 「CUBE 能吃 fp32 操作数」

`/workspace/cannbot-knowledge/knowledge/ops/ascendc/concepts/roofline_model.md`（rev 由该仓自身版本管理，下文按行号引用）：

```
23  | AI Vector Cores | 56 | ...
24  | VEC Peak (FP32) | ~28 TFLOPS | 56 cores × 512 FLOPS/cycle × 1GHz |
25  | VEC Peak (FP16) | ~56 TFLOPS | 2x FP32 (half-precision packing) |
26  | **CUBE Peak (FP16)** | **~373 TFLOPS** | **Measured 2026-06-12, .171 NPU1 957b, torch.matmul 8192³** |
27  | **CUBE Peak (BF16)** | **~368 TFLOPS** | **Measured (same run)** |
28  | **CUBE Peak (FP32)** | **~24 TFLOPS** | **Measured — cube is fp16/bf16-optimized, fp32 not favored** |
29  | HBM Bandwidth | 1.5 TB/s | Measured (theoretical ~2 TB/s) |
```

紧接着的上下文（`roofline_model.md:33-40`）在讲"别把 CUBE 和 VEC 的峰值搞混"：

```
33  > ⚠ **CUBE vs VEC peak — do not conflate.** matmul / FlashAttention / any
34  > op dominated by `Matmul`/`Mmad` runs on the **CUBE unit** (peak ~373 TFLOPS
35  > fp16), NOT the vector unit (~56 TFLOPS fp16). ...
39  > VEC peak. 源仓的 `roofline_eval.py` 按 `peak_cube_*` 字段 + `_peak_tflops(op_type)`
40  > 编码了这个拆分，ridge 按单元分别计算（见下）。
```

**它指的是"操作数 fp32"还是"累加器 fp32"？—— 答案是「操作数 fp32」，而且是「默认（未开 HF32）模式下的 fp32 操作数」。** 依据有三条，见 §5：
① 表格同一列的三行分别是 FP16 / BF16 / FP32，都是**操作数**精度（累加器一律是 fp32，不会单列一行）；
② 官方 `cube_k` 表：3510 上 `Mmad(float,float)` 的 `cube_k = 1`，`Mmad(bf16,bf16)` 的 `cube_k = 16`
   ⇒ `368 TFLOPS ÷ 16 = 23 TFLOPS ≈ 24`（`roofline_model.md` 那行的数字就是这么来的）；
③ 本探针实测同 shape 吞吐比 = **15.86 ~ 15.89×**（§5），与 ② 的 16 吻合。

### 1.2 读 B（`m18_gdn_prefill/README.md` §7）—— 「cube 只能吃 bf16 操作数」

`m18_gdn_prefill/README.md:368-374`（该行号钉在不可变 rev `20bd20d`）：

```
368  ## 7. AIC/cube 升级路径（本切分为何 AIC 空转）
369
370  - 本核的 chunk 工作 **100% 是 fp32 向量域**（A/KKT/Γ/前代/状态都要求 fp32 精度以对齐 fp64 判据），
371    而 3510 的 cube 只能吃 bf16 操作数 + fp32 累加。**用 cube 就得把操作数降成 bf16**，判据随之
372    变成「bf16 三方 cross_check」（donor 的判据就是这个），当前 fp64 紧容差路线会作废。
```

**这两句不可能同时为真。** 本探针裁决的结果是：**读 A 对，读 B 错**（§3/§4/§5）。

### 1.3 官方文档/头文件侧的锚点（都是本机 CANN 9.1.0 自带）

| 位置 | 内容（逐字） |
|---|---|
| `/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_aicore_functions.h:2356,2382-2389` | 3510 的 `mad` intrinsic 重载列表里有 **`MMAD(float, float, float);`**（与 `MMAD(int32_t,int8_t,int8_t)`、`MMAD(float,bfloat16_t,bfloat16_t)` 并列） |
| `/workspace/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.md:194` | 「**表5** dst、fm、filter 支持的精度类型组合（Ascend 950PR&950DT系列产品）」中一行：`\| float \| float \| float \|` |
| `/workspace/asc-devkit/docs/zh/api/appendix/cube_instruction_theoretical_perf_summary.md:67-68` | 3510：`Mmad \| float \| float \| 16 \| 16 \| **1** \| 8`（默认）与 `Mmad（开启HF32）… \| 16 \| 16 \| **8** \| 8` |
| `/workspace/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_compute/asc_enable_hf32.md:31,76` | 「开启该模式后，Mmad 计算 FP32 数据的性能将得到提升，但会带来一定的精度损失」/「开启 HF32 模式后，L0A Buffer/L0B Buffer 中的 FP32 数据将在参与 Mmad 计算之前被舍入为 HF32 格式」 |

（`/workspace/asc-devkit` 是本地 CANN 9.x 官方 AscendC 样例与文档仓，`git rev-parse HEAD` = `648a6018207d75af44c6865f96511bafadd90630`。**本 mission 未修改其中任何文件。**）

---

## 2. 实验 1：fp32×fp32→fp32 的 `Mmad` 能不能编译（任务 2）

**结论：能编译，rc=0，无诊断。**

```
$ source /usr/local/Ascend/ascend-toolkit/set_env.sh
$ cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
-- CMAKE_ASC_COMPILER: /usr/local/Ascend/cann-9.1.0/bin/bisheng
$ cmake --build build -j4
[ 50%] Building ASC object CMakeFiles/probe_cube_fp32.dir/probe_cube_fp32.asc.o
[100%] Linking ASC executable probe_cube_fp32
[100%] Built target probe_cube_fp32
build rc=0
```

编译器：`bisheng --version` → `clang version 15.0.5`；CANN `9.1.0`（`compiler/version.info` timestamp `20260730_231653901`）；`--npu-arch=dav-3510`。

### 2.1 被编译的那段代码（即「代码实现路径」的完整上下文）

- API：AscendC C++ basic API 的 `Mmad`（`#include "kernel_operator.h"`；头 `basic_api/kernel_operator_mm_intf.h`）。
- dtype：`LocalTensor<float>` × 2 → `LocalTensor<float>`（即 `Mmad(float×float→float)`）。
- 形状：`m=16, n=16, k=256`。
- layout：A 在 L0A（`TPosition::A2`）为 **Nz**、B 在 L0B（`TPosition::B2`）为 **Zn**、C 在 L0C（`TPosition::CO1`）为 **Nz**——
  与 `Mmad.md` §「表1」对 950 的规定一致；**fp32 的 `C0_SIZE = 8`**（`32B / sizeof(float)`，官方例 `mmad.asc` 的 `c0Size = 8`）。
- 完整调用链（`probe_cube_fp32.asc` 的 `pcf::prec_kernel`）：
  `DataCopy(GM→L1, Nd2NzParams)` → `LoadData(L1→L0A, LoadData2DParamsV2)` / `LoadData(L1→L0B, LoadData2DParams)` → `Mmad` → `Fixpipe(L0C→GM)`。
  关键参数：A 的 `LoadData2DParamsV2{kStep = K_DIM / 8 = 32, mStep = M_DIM/16 = 1, src/dstStride = 1}`；
  B 的 `LoadData2DParams{repeatTimes = (N_DIM/16)*(K_DIM/8) = 32, srcStride = 1, dstGap = 0}`。

### 2.2 第一次运行时**真的出过错**（逐字留档，因为它有价值）

首版用 `PipeBarrier<PIPE_ALL>()` 做同步，结果**整张表滞后一轮**（`dev[d] = ref[d-1]`），首轮读到残值：

```
dist             dev C[0][0] ref_fp64 C[0][0]       absErr       relErr      dev C[1][1]      dev C[3][3]
d0 ones          0.610267878              256    2.554e+02    9.976e-01   1.91357005e+09                0
d1 mant13                256         256.0625    6.250e-02    2.441e-04              256              256
d2 mant10           256.0625            256.5    4.375e-01    1.706e-03         256.0625         256.0625
d3 absorb              256.5         16777471    1.678e+07    1.000e+00            256.5            256.5
d4 rand             16777216       59.7598435    1.678e+07    2.807e+05         16777216         16777216
d5 cancel         59.7598228                0    5.976e+01    5.976e+01       60.0117607       63.5480423
```

判读：**这是探针的流水同步缺陷，不是硬件不支持**（`dev[d]` 精确等于 `ref[d-1]`，说明操作数与布局都是对的、
只是 `Mmad`(M 管道)→`Fixpipe`(FIX 管道) 以及 `Fixpipe`→下一轮 `Mmad` 的跨管道依赖没建立）。
改成显式事件配对（`SetFlag/WaitFlag<HardEvent::{MTE1_MTE2, MTE2_MTE1, M_MTE1, MTE1_M, M_FIX, FIX_M}>`，
跨迭代的 3 个 flag 在循环前预置一次）后，结果全部正确（§3）。

> 顺带一条 out-of-scope 观察（已另 file 给塔，见 §9）：仓库既有的 `m10_attn_decode/m10_mmadprobe.asc`
> **只用 `PipeBarrier`、0 处 `SetFlag/WaitFlag`**（事实部分已核）；其归档 log 表头自述「结论：探针 B 操作数映射错 ⇒ 给不出标定」。
> 缺事件配对是那条「映射错」结论的**候选真因（未经 m10 侧复现）**，不是已证结论 ——
> m10 那份归档读数（单个 Fixpipe 内部的行串位）与本探针这次观测到的形态并不相同。
> **本 mission 不改 m10。**

---

## 3. 实验 2：能不能跑、算得对不对、精度形态（任务 3）

**结论：能跑（rc=0），数值与 host 独立参考一致。默认模式下操作数是「全 fp32」（23 位尾数全保），
累加是「fp32 宽度」——没有降到 bf16/tf32/hf32，也没有宽累加器。**

命令：`flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 dump evidence/dumps`（7 个分布，一个 launch）。
下表为**默认模式（未开 HF32）**，逐字来自 `evidence/logs/run_dump_r1.log`：

```
dist             dev C[0][0] ref_fp64 C[0][0]       absErr       relErr      dev C[1][1]      dev C[3][3]
d0 ones                  256              256    0.000e+00    0.000e+00              256              256
d1 mant13           256.0625         256.0625    0.000e+00    0.000e+00         256.0625         256.0625
d2 mant10              256.5            256.5    0.000e+00    0.000e+00            256.5            256.5
d3 absorb           16777216         16777471    2.550e+02    1.520e-05         16777216         16777216
d4 rand           59.7598228       59.7598435    2.064e-05    3.454e-07       60.0117607       63.5480423
d5 cancel                  0                0    0.000e+00    0.000e+00                0                0
d6 mant23s        1.00000012       1.00000012    0.000e+00    0.000e+00       1.00000012       1.00000012
```

7 个分布的含义与判读（`M=N=16, K=256`；A、B 都是 fp32，host 参考是 numpy/C 的 **fp64 精确和**）：

| 分布 | 构造 | 读数 | 判读 |
|---|---|---|---|
| d0 ones | A=1, B=1 | dev = 256 = K，误差 0 | 恒真自检：搬运/布局/Mmad/Fixpipe 全对 |
| d1 mant13 | A=1+2⁻¹², B=1 | dev = **256.0625** = 精确值 | 操作数保留 **13 位**有效位 ⇒ **未**降到 hf32(10 位)/bf16(8 位) |
| d2 mant10 | A=1+2⁻⁹, B=1 | dev = 256.5 = 精确值 | 操作数保留 10 位有效位 ⇒ **未**降到 bf16 |
| d6 mant23s | A[·,0]=1+2⁻²³，其余 0；B=1（单项，无累加） | dev = `0x3F800001` = **1+2⁻²³**，精确值 | **23 位尾数全保 ⇒ 操作数是全 fp32**（这一条没有累加噪声，直接读操作数位宽） |
| d3 absorb | A=1, B[·,0]=2²⁴，其余 1 | dev = **16777216** = 2²⁴；真值 2²⁴+255 | 见下方 d3 的展开：**fp32 宽度累加**，同时是「`cube_k=1`（默认档）」与「`cube_k=8`（HF32 档 / 按 8 分块精确和）」的**判别项** |
| d4 rand | A、B 为 23 位尾数伪随机 ∈[−1,1) | dev vs **fp64** = 7.18e−5；vs bf16-参考 = 2.14e−2；vs hf32-参考 = 3.65e−3 | dev 最贴合「fp64/fp32 操作数」参考 ⇒ 操作数确为 fp32；误差量级是 fp32 累加噪声 |
| d5 cancel | A=1, B=(−1)^k | dev = 0，误差 0 | 完全相消为 0 ⇒ 无偏移/无脏项 |

**d3 的展开（M106 r1 复审的加固，比最低主张更强）**：真值 2²⁴+255 = 16777471，
`np.float32(2**24+255) = 16777472`（0x4B800000）。三个候选各给不同读数：

| 假设 | 预期 dev | 实测 |
|---|---|---|
| fp32 宽度累加（默认档 `cube_k=1`，逐项 +1 被吞） | 2²⁴ = **16777216** | ✅ 默认 run 给 16777216 |
| 按 8 元素分块精确和（每块 sum 后一次性加，也等价于 `cube_k=8` 的收缩粒度） | 2²⁴+256 = **16777472** | ✅ **HF32 run 给 16777472** |
| 宽累加器（≥25 位）后一次舍入 | 16777472 | — 与上面同值，故被"按 8 分块"这条一起排除 |

⇒ d3 不只是"排除宽累加器"，它**实际在区分 `cube_k=1` 与 `cube_k=8`**，且与官方 `cube_k` 表（§1.3）
逐条自洽。HF32 档下操作数 1 与 2²⁴ 都能被 10 位尾数精确表示 ⇒ 差异**不可能**来自操作数舍入，
只能来自累加结构，这让 d3 的归因更干净。

**d4 的逐元素误差分布（256 个元素，每个是 K=256 的内积；独立 numpy fp64 复核，`check_ref.py`）**：

```
d4 rand   : dev vs fp64  maxAbsErr = 7.175603e-05   (maxRelErr = 1.007e-06)
            dev vs bf16  maxAbsErr = 2.140994e-02   (比 fp64 参考差 298×)
            dev vs hf32  maxAbsErr = 3.654220e-03   (比 fp64 参考差 51×)
            => dev 最贴合【fp64/fp32 操作数】（残差 7.176e-05）
            dev 与 fp64 逐位相等元素数: 26/256
```

⇒ **"内部是不是真的全 fp32 乘累加"的答案**：**操作数是全 fp32**（d1/d6 二值判据 + d4 的三档比较）；
**累加是 fp32 宽度**（d3），即"fp32 操作数 + fp32 精度累加"，**不是** bf16/tf32 操作数，**也不是**分块精确和 / 宽累加器。

### 3.2 「快路」HF32 的精度形态（`asc_enable_hf32()`，对照）

同 7 个分布，但 kernel 内先调 `asc_enable_hf32()`（逐字来自 `evidence/logs/run_hf32_r1.log`）：

```
d1 mant13                256         256.0625    6.250e-02    2.441e-04    ⇒ 13 位有效位**丢了**
d2 mant10              256.5            256.5    0.000e+00    0.000e+00    ⇒ 10 位还在
d3 absorb           16777472         16777471    1.000e+00    5.960e-08    ⇒ dev 变成 16777472（见 §3 的 d3 展开）
d6 mant23s                 1       1.00000012    1.192e-07    1.192e-07    ⇒ 2^-23 位**丢了**（dev=1.0）
d4 rand           59.7588577       59.7598435    9.858e-04    1.650e-05
   dev vs fp64 = 3.652615e-03 ; dev vs hf32/tf32/fp16-参考(同一个参考，尾数同为 10 位) = 1.490628e-05
   ⇒ 最贴合 hf32 参考
```

⇒ HF32 模式下 **L0 里的 fp32 操作数先被舍到 HF32(1s+8e+**10**m) 再参与乘累加**（官方 `asc_enable_hf32.md:76` 的逐字描述）；
d4 的相对误差从 1.0e−6 掉到 1.7e−5（**差 16×**）。

**「默认模式 HF32 是关的」不是先验假设，而是被数值证据独立证实的**：默认 run 的 d1（13 位不丢）、
d2（10 位不丢）、d6（23 位不丢）如果 HF32 开着**必定塌** —— 对照 HF32 run 的同名分布：d1 → 256（丢 13 位）、
d6 → 1.0（丢 2⁻²³ 位）。所以"操作数是 fp32"这个结论不依赖"默认即关"这一先验。
（之所以仍把"B1 运行期须显式断言 HF32 关闭"列为待办，是因为**别的模块**——如 Matmul 高阶 API 的 `SetHF32`——可能打开它：§6.3 第 5 条。）
这与 HF32 表里"默认 `cube_k=1`、开启后 `cube_k=8`"的自洽性一致。

---

## 4. 实验 3：吞吐 —— fp32 vs bf16（任务 4 的实测部分）

方法：操作数一次性装进 L0，然后**连发 2000 次 `Mmad`**（`M=64, N=64, K=128`），用设备侧 `GetSystemCycle()` 量这段 cycle；
两侧（fp32 / bf16）走**完全相同的代码路径**，只有 dtype（⇒ C0_SIZE）不同。`dev C[0][0] = 128 = K` 确认这 2000 次真的执行了。

默认模式（`perf`）—— 下面这一段**逐字**来自归档的 `evidence/logs/run_perf_r1.log`（三次运行各一份，见 §4.1）：

```
[PCF] Mmad 吞吐：同一 shape M=64 N=64 K=128，连发 2000 次，SYS_CNT 计时
dtype          C0    cycles(all)      cycles/Mmad      MAC/cycle  dev C[0][0]
fp32            8        2485322          1242.66        421.908          128
bf16           16         156584            78.29       6696.572          128
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 15.872 x（MAC/cycle 比 0.063 x）
rc=0
```

HF32 模式（`perf_hf32`）—— 逐字来自归档的 `evidence/logs/run_perf_hf32_r1.log`：

```
[PCF] Mmad 吞吐：同一 shape M=64 N=64 K=128，连发 2000 次，SYS_CNT 计时
dtype          C0    cycles(all)      cycles/Mmad      MAC/cycle  dev C[0][0]
fp32            8         312512           156.26       3355.314          128
bf16           16         156396            78.20       6704.622          128
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 1.998 x（MAC/cycle 比 0.500 x）
rc=0
```

### 4.1 跨进程重复（每一条都能在归档里找到出处）

`run_probes.sh` 对 `perf` / `perf_hf32` 各跑 **3 个独立进程**，逐次落盘：

```
$ grep -h "cycle 比为" evidence/logs/run_perf_r?.log
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 15.872 x（MAC/cycle 比 0.063 x）
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 15.879 x（MAC/cycle 比 0.063 x）
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 15.880 x（MAC/cycle 比 0.063 x）
$ grep -h "cycle 比为" evidence/logs/run_perf_hf32_r?.log
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 1.998 x（MAC/cycle 比 0.500 x）
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 1.999 x（MAC/cycle 比 0.500 x）
[PCF] 同 shape 吞吐比 fp32 : bf16 的 cycle 比为 1.999 x（MAC/cycle 比 0.500 x）
```

| 口径 | 本目录归档的 3 次独立运行 | 复审独立复跑 | 官方 `cube_k`（bf16=16）预期 |
|---|---|---|---|
| fp32 默认 vs bf16 | **15.872 / 15.879 / 15.880** | 15.866 / 15.871 / 15.872 | 16 / 1 ⇒ **16×** |
| fp32 HF32 vs bf16 | **1.998 / 1.999 / 1.999** | 1.997 / 2.000 | 16 / 8 ⇒ **2×** |

⇒ 实测比值与官方 `cube_k`（`cube_instruction_theoretical_perf_summary.md:67-68`）**逐条吻合**；
跨进程散度 ≈0.05%（fp32）/ ≈0.05%（HF32）。**全部观测区间**（本目录归档 + 复审复跑）：
fp32 = **15.866 ~ 15.880**，HF32 = **1.997 ~ 2.000**。

**换算成绝对吞吐**（以 `roofline_model.md:27` 实测 bf16 = 368 TFLOPS 为基准，按实测比值折算）：

| 模式 | 相对 bf16 | 换算 TFLOPS | 备注 |
|---|---|---|---|
| bf16 | 1× | 368（官方实测） | 基准 |
| fp32 + HF32 | 1/2 | ≈ 184 | 操作数舍到 10 位尾数（§3.2） |
| **fp32 默认** | **1/15.88**（归档 3 次的均值 15.877） | **≈ 23.2** | 与 `roofline_model.md:28` 的 ~24 TFLOPS 一致；也低于 `roofline_model.md:24` 的 VEC fp32 峰值 ~28 TFLOPS（**注意**：那个 VEC fp32 峰值本身在本工作区有 3 个互不相同的口径，见 `docs/19` §7 第 2 条 —— 所以"低于 VEC 峰值"这句继承该口径的不确定性）|

**未定标项（不写绝对断言）**：本探针只给**比值**与"以 368 为基准的折算"。`SYS_CNT` 与 cube 时钟的绝对关系本 mission
**未独立标定**，故上表的 421.9 / 6696.6 MAC/cycle 这类**绝对**读数只作原始记录；其中 bf16 的 ≈6700 MAC/cycle
高于朴素 `16×16×16=4096 MAC/cycle/核`，说明 `SYS_CNT` 的计数频率与 cube 时钟不是 1:1 —— 该关系需要另做一次
已知指令数的标定（`docs/19` §7 第 3 条）。

**关于「23.2 TFLOPS 是不是循环论证」**：**不是循环，是链式**。368 是 `roofline_model.md:27` 记的
**独立实测**（2026-06-12，`torch.matmul 8192³`）bf16 峰值；`15.88` 是本探针**独立测的比值**；两者相乘得 23.2
（368 ÷ 15.877 = 23.18）。它不是用 24 反推的。而且还有第三条独立佐证：官方 `cube_k` 16:1 ⇒ 368÷16 = 23，
与 roofline 自己那条**独立实测的** "~24 TFLOPS" 一致 ⇒ **三条腿互相独立且吻合**。
但要紧的是口径：**23.2 是"折算值"，不是本探针实测的绝对值**（`SYS_CNT` 与 cube 时钟的绝对关系未标定，见上）。

---

## 5. 两处矛盾源怎么裁决的（任务 1 的收口）

| 读数 | 裁决 | 依据 |
|---|---|---|
| `roofline_model.md:28` "CUBE Peak (FP32) ~24 TFLOPS — cube is fp16/bf16-optimized, fp32 not favored" | **准确**。它指的是**fp32 操作数**，模式是**默认（未开 HF32）**。 | `cube_k`：bf16=16 / fp32=1 ⇒ 368÷16 = 23 ≈ 24；本探针实测 368÷15.88 = 23.2 |
| `m18_gdn_prefill/README.md:371` "3510 的 cube 只能吃 bf16 操作数 + fp32 累加" | **前半句证伪**（cube 能吃 fp32 操作数）；"fp32 累加"这半句在默认模式下也对（§3 的 d3）。 | ① 3510 `mad` intrinsic 有 `MMAD(float,float,float)` 重载（在 `__clang_cce_aicore_functions.h:2356` 的 arch-3510 守卫内，:`2385`）；② 官方 `Mmad.md:194` 表5 列了 `float/float/float`；③ 本探针编译 rc=0 且数值全对 |

**两条的正确合并表述**（建议给 B1 用）：*「3510 的 cube 能吃 fp32 操作数（默认模式下操作数保持全 fp32、累加 fp32 宽度），
但 fp32 操作数下的 Mmad 吞吐只有 bf16 的 1/16（约 23 TFLOPS），而 bf16 是 368 TFLOPS。」*

`m18` 那句错误**没有动摇 m18 的任何结论**：m18 §7 的整体论证是"本切分 100% 在向量域、AIC 空转"，这个选择依然成立；
错的只是它对**硬件能力**的那句描述。已按纪律**另 file 给塔**（不自行改 m18 / 不改知识库）。

---

## 6. 【直接回答】`docs/19` §7 第 5 条的「第三选项」是否成立

`docs/19:386` 的原话（rev 见 wt-104 的分支）：*"若 cube **能**吃 fp32 操作数：chunk core 在 cube-fp32 上
≈ `31.2 GFLOP ÷ 24 T = 1.30 ms/层`"*，`docs/19:1181-1185` 把它叫**分支①**。

### 6.1 逐条裁决

| 第三选项的三个组成 | 裁决 | 本探针的读数 |
|---|---|---|
| ① cube **能**吃 fp32 操作数 | **成立** | 编译 rc=0；运行 rc=0；d0/d1/d2/d6/d5 逐元素精确；d4 贴合 fp32 参考 |
| ② 保住 **fp64 紧容差判据** | **成立（在"fp32 精度"这个意义上）** | 默认模式：操作数 23 位尾数全保（d6）、13 位有效位不丢（d1）；累加 fp32 宽度（d3）；d4 单次 K=256 内积 maxRelErr = 1.0e−6 |
| ③ 拿到 **cube 吞吐 = 1.30 ms/层** | **数字成立，口径必须改** | 实测 fp32 = bf16 的 **1/15.88** ⇒ 约 **23.2 TFLOPS**；`31.2 GFLOP ÷ 23.2 T = 1.34 ms/层`（`docs/19` 用 24 T 得 1.30）|

### 6.2 结论（一句话版）

> **第三选项成立，但它不是"cube 吞吐 + 紧判据"，而是"1/16 的 cube 吞吐 + 紧判据 + 相对 VF 路线的约 11.6× 提速"。**

三个必须一起说的数字（都以 chunk core 口径）：

| 路线 | chunk core 下限 | 相对 VF 实测 | 判据形态 |
|---|---|---|---|
| VF/RegBase（m18 实测） | 15.51 ms/层（实测） | 1× | fp32 紧容差 |
| **cube-fp32（本探针路线）** | **1.34 ms/层**（31.2 GFLOP ÷ 23.2 T；m18 自己的 36.0 GFLOP 口径 = 1.55 ms） | **≈ 11.6×** | **fp32 紧容差**（同 VF 一档） |
| cube-bf16（`docs/15` §3.1 路线） | 0.085~0.125 ms/层 | ≈ 124~183× | 必须换 bf16 cross_check |

**关键提醒（否则会被误读）**：第三选项相对 VF 的 ~11.6× **不是**来自峰值 FLOPS（23.2 < VEC fp32 的 28 TFLOPS），
而是来自 **cube 的结构优势**：VF 路线实测只跑到 **0.466 指令/cycle**（`docs/19` §0 第 2 条），瓶颈在逐元素指令发射，
不在 MAC 吞吐。所以：

- 若**判据允许 bf16** ⇒ cube-bf16 仍比 cube-fp32 快 **11~16×**，第三选项没有意义；
- 若**判据要求 fp32 紧容差**（B1 的现状）⇒ 第三选项**同时**优于 VF（~11.6×）且不需要改判据形态 —— **这是它的价值所在**。
- 若有人为了再快 8× 打开 HF32 ⇒ 操作数降到 10 位尾数（§3.2 实测 d1 丢 13 位、d6 丢 2⁻²³ 位、d4 误差差 16×），
  **`docs/17` 的 ε≈5e-5 紧容差会失守**，等于退回"判据必须改"的那条路。

### 6.3 若 B1 走第三选项，**下应额外验证什么**（本 mission 不改 B1 设计文档）

本探针只量到 **Mmad 本身**的吞吐/精度；下面是它**没有**覆盖、但对 B1 的 1.34 ms 结论是**必要前提**的项：

1. **fp32 操作数的喂数带宽**（本项最关键）：我的 1.34 ms 假设操作数永远就绪。fp32 操作数字节数是 bf16 的 2×，
   而 **3510 的官方表**给 L1→L0A **256 B/cycle**、L1→L0B **256 B/cycle**
   （`cube_instruction_theoretical_perf_summary.md` 的**表5**「矩阵计算搬入类指令理论性能说明（NPU架构版本3510）」，
   标题在 :126，`LoadData（2D矩阵搬运V2）` 的 L0A 行 :130、**L0B 行 :131 = 256**。**注意不要引成表3**：
   表3 的标题（:99）是「NPU架构版本2201」= A3/910b，其中 L0B 才是 128 —— 那是别的架构的值）。
   另需注意同文件 :141 的 950 专属注脚：**「针对Ascend 950PR&950DT系列产品，LoadData（2D矩阵搬运）接口仅为兼容实现，
   内部使用了LoadData（2D矩阵搬运V2）接口实现…但需要注意，该兼容实现会造成性能损失」** —— 本探针的 B 路用的正是
   v1 `LoadData2DParams`，所以 B1 若照抄这条路径，还要把这条兼容实现的损失算进去（表5 里没有 v1 的行）。
   需要实测"UB→L1→L0 喂一套 fp32 操作数"的 cycle，并证明它 ≤ Mmad 的 cycle。**这正是 `docs/19` §7 第 4 条
   （"UB→L1 的操作数通路"）—— 它现在是第三选项的绑定约束**。
2. **每 chunk 的 Mmad 固定开销**：本探针用 `M=64,N=64,K=128`（大块）测吞吐；GDN 的 chunk 形状小得多，
   且每 head 是 **65 步串行链**。需要一次"小形状 Mmad 每 call floor"的标定（即 `docs/19` §7 第 9 条 /
   M33 §9 存疑 5 问的那条 44-69 µs/call floor 在 `LoadData`+`Mmad` 路径上是否存在）。
   本探针**没有**回答这条。
3. **48 value head vs 28 AIC 的排布**：与 dtype 无关，但会否掉"1 head/AIC"（`m18 §7` 已指出）→ 需要 2 head/AIC
   串行或 chunk 轴并行化；这会乘在 1.34 ms 上。
4. **ε 的端到端重推导**：d4 只证了单次 K=256 内积的 maxRelErr=1.0e−6；整层的 K 更长、还有 cumsum/三角求解/状态递推，
   必须按 `docs/17` §1.1 重新推 ε（本探针不能替代）。
5. **模式断言**：必须显式确认/断言 HF32 处于**关闭**状态（默认关，§3.2），并把这条写进 B1 的就绪检查；
   一旦任何一层（Matmul 高阶 API 的 `SetHF32`、或别的模块）打开它，判据形态就变了。
6. **L0C→UB/GM 的 tap 成本**：每 chunk 要把 L0C 搬出来，本探针只做了一次 Fixpipe，**未标定**这条成本。

---

## 7. 复现命令与判据

```bash
cd probe_cube_fp32
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4

# 设备一律用锁自助排队（-w 在锁文件之前；锁内只放设备那段；绝不并发）
#   prec      默认（未开 HF32）的 7 分布精度表
#   hf32      asc_enable_hf32() 后的同一张表
#   perf      同 shape 的 fp32 vs bf16 吞吐
#   perf_hf32 HF32 vs bf16 吞吐
#   dump      落盘 prec_a/b/c.bin 供独立复核
#   pref_hf32 / perf_hf32 需各跑多次（run_probes.sh 跑 3 次并逐次归档）
flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 dump evidence/dumps
flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 prec
flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 hf32
flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 perf
flock -w 900 /tmp/npu0.lock ./build/probe_cube_fp32 perf_hf32

# 独立判据：rc=0 等价于【d0/d1/d2/d3/d4/d5/d6 七条判据全部成立】，不只是自检项
/usr/local/python3.12.13/bin/python3 check_ref.py evidence/dumps

# 归档完整性自查（必须在 evidence/ 目录内跑：sha256.txt 里的路径是相对 evidence/ 的）
( cd evidence && sha256sum -c sha256.txt )

# 一键全量（构建 + prec/hf32/dump 各 1 + perf/perf_hf32 各 3 个独立进程 + 独立复核 + 两条负向对照 + 归档 + sha256）
bash run_probes.sh
```

**判据（可机械检查；每一条都为 rc 或打印值所覆盖）**：
- `build rc = 0`（§2）；
- `check_ref.py` **rc = 0**，其含义是**下列 7 条全部成立**（不是"只看自检"）：
  `d0 == K` / `d1 == 256.0625 精确` / `d2 == 256.5 精确` / `d3 == 2²⁴ 精确` / `d4 最贴合 fp64-fp32 参考且 maxRelErr < 1e-4` /
  `d5 全 0` / `d6 全 == 0x3F800001`；
- **负向对照（判据非空洞）**：`run_probes.sh` 会做两次故意破坏并断言 rc 必须变红——
  ① 把 d6 的 C 全改成 `1.0`（模拟操作数被舍成 hf32/bf16）⇒ rc 必须 = 1；
  ② 把 d3 的 C 改成 `np.float32(2²⁴+255)`（模拟变成按 8 分块精确和 / `cube_k=8`）⇒ rc 必须 = 1。
  两次读数落 `evidence/logs/negctl_{d6,d3}.log`，被破坏的输入落 `evidence/negctl_{d6,d3}/`；
- `perf` 的比值落在 `15 < r < 17`（3 次运行都查）；`perf_hf32` 的比值落在 `1.9 < r < 2.1`（3 次运行都查）。

最近一次全量运行（2026-09-27）：`build_rc=0`、`check_ref_rc=0`、`negctl_d6_rc=1`、`negctl_d3_rc=1`，
`prec_r1`/`hf32_r1`/`dump_r1` 各 1 次 + `perf_r{1,2,3}`/`perf_hf32_r{1,2,3}` 各 3 次全部 `rc=0`
（`dump_r1` 因设备被别人占用重试了 2 次才拿到锁，log 尾部有 `attempts=2`；见 `evidence/logs/matrix.txt`）。
本次 dump 的 `prec_c.bin` sha256 = `139b9860a1e5c5d827e3a5243a5bf02a88e8e33ce36e2cdff75173d5f5c82c46`
—— 与 M106 r1 复审自己新鲜复跑得到的 sha **逐字节相同**（M106 r1 复审报告 §Checks 记了同一串）。

---

## 8. 没验证什么 / 未确定（按纪律明写）

1. **未做**"fp32 操作数的 L1/L0 喂数带宽"标定（§6.3 第 1 条）。1.34 ms 是**Mmad 吞吐下限**，不是端到端预测。
2. **未做**小形状（GDN chunk 形状）的每 call 固定开销标定（§6.3 第 2 条）。
3. **未标定** `SYS_CNT` 与 cube 时钟的绝对关系（§4 末），故不写"fp32 = X TFLOPS 实测"这种绝对断言，
   只写"以 368 TFLOPS 基准按实测比值折算"。
4. **未测** `kDirectionAlign`、bias、unitFlag、GEMV(M=1) 等其它 Mmad 参数下 fp32 的行为；本次只测了
   默认 `MmadParams{cmatrixInitVal=true, cmatrixSource=false}`、`M>=16`。
5. **未测** `K` 不是 8 的倍数时的行为（doc 说 fp32 默认 K 对齐 `ceil(K/8)*8`；本次 K=256 是 8 的倍数）。
6. **未做**多 AIC 并发/整芯片的吞吐（本探针 `<<<1>>>` 单核）。
7. **未验证** HF32 的舍入模式配置（`asc_set_hf32_round_mode`）；本次只用默认 RNE。
8. 本探针是**单次 launch、单分布数组**的最小用例；它证明"能/不能"与"这一档精度形态"，
   **不构成**"任何形状/任何参数下都如此"的普遍断言（人类原话："芯片的 bug 哪有那么容易被你发现"）。

---

## 9. 纪律读数

- **设备**：全程 `flock -w 900 /tmp/npu0.lock` 排队（排队超时会重试，逐次尝试记为 `evidence/logs/run_*.log` 尾部的 `attempts=`），
  **未并发**；跑前跑后各记一次 `npu-smi`（见 `evidence/logs/commands.txt`）。
  设备上另有 `m15_layer_loop` 进程常驻（HBM ~12.9 GB），与本探针共存。**设备忙未记为 blocker。**
- **数据自造**：全部在 host 端用 `Hash()`/常量生成，**未加载任何模型权重**；变异只在 `/tmp`（`/tmp/hf32probe` 是一次编译可行性试跑）。
- **内存**：本探针 host RSS 远小于 15.6 GiB 量级（只 hold 7×16×256×4 B 的 A/B 与 7×16×16×4 B 的 C）。
- **写文件前 `df -h /`**：m10 与本探针产物合计 MB 量级。
- **绝对断言**：本文件不写"无残留/全部/0 命中"类断言；每个计数/读数都带命令与范围。
- **外部知识库**：`cannbot-knowledge` 与 `asc-devkit` 都**只读**，未改一行；发现的问题走 findings（§5）。

### 9.1 本文件在 M106 r1 复审后的改动（都只动 `probe_cube_fp32/**`，不动任何结论）

- **P2-1**：§4 的两个代码块换成**真正归档的** `run_perf_r1.log` / `run_perf_hf32_r1.log` 逐字内容；
  `run_probes.sh` 改为对 `perf`/`perf_hf32` **各跑 3 个独立进程并逐次归档**，"多次运行"现在能在
  `evidence/logs/run_perf_r{1,2,3}.log` / `run_perf_hf32_r{1,2,3}.log` 里逐条找到出处（§4.1）。
- **P2-2**：`check_ref.py` 的**退出码现在覆盖全部 7 条判据**（d0/d1/d2/d3/d4/d5/d6），不再只覆盖自检项；
  并新增**两条负向对照**（§7）：把 d6 弄坏、把 d3 弄坏 ⇒ rc **必须**变红，证据落 `evidence/logs/negctl_{d6,d3}.log`。
- **P2-3**：§6.3 第 1 条的喂数带宽改引 **3510 的表5**（`cube_k` 同表口径），L0B = **256** B/cycle（原引表3 = 2201/A3，L0B=128 是别的架构），
  并补上 950 的 v1 `LoadData` "兼容实现、有性能损失"那条注脚。
- **次要**：`.asc` 里重复的 fp16/hf32 参考行合并为一（尾数同为 10 位 ⇒ 同一参考）；`CMakeLists.txt` 注释里的模式名改对；
  §6.2 的"约 10×"改成 **11.6×**；§2 关于 m10 的 out-of-scope 观察按复审意见降级为"候选真因（未经 m10 侧复现）"并去掉现象类比；
  §3 按复审的加固补上 **d3 能区分 `cube_k=1` 与 `cube_k=8`** 一节，并把"默认关 HF32"写成**由 d1/d2/d6 数值证据证实**而非先验假设。

**一处需要点明的溯源细节**：本轮归档运行是在 `run_probes.sh` 的那一版上跑的，之后我在它里面**多了两行**
（`rm -f "$LOGS"/run_*.log …` 与 `rm -rf "$EV"/negctl_d6 "$EV"/negctl_d3`，用来清掉上一轮命名的旧日志 —— 正是 P2-1 的教训）。
所以 `evidence/logs/commands.txt` 里 `run_probes.sh` 的 sha（`22ed0b2e…`）与工作树里的现值（`02b8671f…`）**不一致**；
**决定设备读数的三件**（`probe_cube_fp32.asc` / `CMakeLists.txt` / `check_ref.py`）**sha 全部一致**。
这条差异**可机械验证**：从当前 `run_probes.sh` 里删掉上面那两行，`sha256sum` 应回到 `22ed0b2ea59995c804f7a06f65a55ac0623d38a784bbd21b16bffcb6743fabd7`（我已实测过）。
