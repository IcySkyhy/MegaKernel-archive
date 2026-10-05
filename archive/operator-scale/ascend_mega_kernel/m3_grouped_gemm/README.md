# M3：多核 grouped MXFP4 GEMM（MoE 纵向切片：GEMM×2 + SwiGLU）

`mix(1,2)` 一次启动内完成 8 个专家槽位的两段 grouped MXFP4 GEMM 与中间 SwiGLU：
MoE block 纵向切片的 GEMM 段（docs/05 §8 的 M1.5 的第一步）。GEMM 数据通路抄改自
`m1_mxfp4_gemm/`（单核链路已实测逐位一致），在其上扩展**专家槽位 × N 维静态切分**
与 **AIC↔AIV 跨核 GM 交接**。权重为真实 checkpoint 切片（`tools/weights/data/layer0_e0-8`，
8 专家 gate_up/down + scale），数值与 CPU golden（dequant + double matmul）逐位一致。

**AIV 段（SwiGLU + 重量化）已按 `docs/05 §6.1` 标准全量 VF 化**（`__VEC_SCOPE__` +
`RegTensor` + `AscendC::Reg::`，无 memory-based 向量计算；M44 审计认定的全仓唯一生产级
(a) 违规点已消除）。位级影响面与 N/A 判据见 §5。

**本 mission（M54）把 AIV 重量化收敛到 floor/OCP 权威规范**（官方 ops-nn 序列；与
`m13_moe_layer` 的 `VecQuantStage`、`tools/golden` 的 `quantize_ocp` 同一规范）：
scale 规则 `ceil -> floor`、data 取整 `最近值+平局取偶 -> CAST_ROUND`；并按官方语义把 data 的
缩放从**乘 scale** 修到**乘 halfScale = 1/scale**（收敛过程中发现的**一处方向缺陷**，见 §5.4）。
h 仍保持 fp32（唯一的遗留差异，见 §5.3）。收敛后 m3 量化器与 m13 的参考在**同一批 bf16 网格
输入上逐字节一致**（`check_m13_cross.py`）。定性（无在役消费）、重基线对照、与 m13 的逐位
对拍见 §5.3/§5.4。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m3_grouped_gemm/build -S m3_grouped_gemm -DCMAKE_BUILD_TYPE=Release
cmake --build m3_grouped_gemm/build -j4
./m3_grouped_gemm/build/m3_grouped_gemm            # 需在仓库根目录运行（读 tools/weights 相对路径）
./m3_grouped_gemm/build/m3_grouped_gemm vf-probe   # AIV VF 自检（不需要数据集，秒级）
./m3_grouped_gemm/build/m3_grouped_gemm vf-prim    # VF 原语语义实测（诊断用，打印读数）
./m3_grouped_gemm/build/m3_grouped_gemm vf-dump m3_grouped_gemm/evidence/quant_cross   # 落盘 device 量化输出
python3 m3_grouped_gemm/check_m13_cross.py m3_grouped_gemm/evidence/quant_cross        # 与 m13 / tools/golden 逐字节对拍
python3 m3_grouped_gemm/analyze_rebaseline.py m3_grouped_gemm/evidence/quant_cross     # 语义重基线量化（pre/post M54）
```

（`check_m13_cross.py` / `analyze_rebaseline.py` 需要 numpy：用
`/usr/local/python3.12.13/bin/python3`，同 `m13_moe_layer/check_ref.py`。）

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`）。blockDim = AIC 数
（`aclrtGetDeviceInfo(ACL_DEV_ATTR_AICORE_CORE_NUM)` 获取，本机 28，不硬编码）。

## 数据流（三 phase，全部产出经 GM 交接）

```
phase 1 (AIC)  grouped gate_up GEMM：8 槽位 × 5 列块 = 40 work items 静态切到 28 AIC
               GU[e] = MXFP4(A_e) @ MXFP4(Wgu_e)^T  → GM (bf16)     K=2560 N=1280
   │  CrossCore：全体 AIC mode 0 barrier → AIC→配对 AIV mode 2
   ▼
phase 2 (AIV)  SwiGLU + MXFP4 重量化：全部 token 行 round-robin 切到 56 AIV
               H[e] = quant(silu(GU[e][:, :640]) * GU[e][:, 640:])  → GM (MXFP4 + e8m0)
   │  CrossCore：AIV→配对 AIC mode 2 → 全体 AIC mode 0 barrier
   ▼
phase 3 (AIC)  grouped down GEMM：8 槽位 × 10 列块 = 80 work items
               Y[e] = MXFP4(H_e) @ MXFP4(Wdn_e)^T  → GM (bf16)      K=640  N=2560
```

每槽位 token 数 `t_e` 运行时读 `counts[e]`（GM），验收取 `t_e = max(1, m-4e)`：
m=1 时全 1（decode 最小形态），m=33 时为 33/29/25/21/17/13/9/5（覆盖奇数尾块 mask、
每核多 item 跨槽位混合）。m ∈ [1,64] 单 M tile（BASE_M=64），`curM` 尾块 mask，
m=1 沿用 m1 的 `calcM=max(curM,2)` quirk 处理（host 按 64 行分配所有 A 类 buffer）。

## 静态切分方案

- work item = (专家槽位, N 列块)；`BASE_N=256`，K 维不切（避免跨核归约）。
- 分派：`item = coreId + i*numBlocks` 轮转。gate_up 40 items → 12 核×2 + 16 核×1；
  down 80 items → 24 核×3 + 4 核×2。负载不均是本 milestone 的已知取舍（ correctness 优先）。
- 每 item 内部：m1 同款 K 循环流水（L1 ping-pong + L0 ping-pong + L0C 累加 → Fixpipe 写 GM）。
- AIV 行切分：行 r = bid, bid+56, ...（slot 前缀和由 counts 标量计算）。每行
  `DataCopy(显式 DataCopyParams) → 整行 VF（10 × 64 lane）→ DataCopy`；UB 全静态（5472B）。

## 核间同步表（CrossCore）

| flagId | 模式 | 方向 | Set 挂的 pipe | Wait 挂的 pipe | 语义 |
|---|---|---|---|---|---|
| 0 | mode 0 | 全体 AIC | PIPE_FIX | PIPE_S | gate_up 全部写 GM 后全体 AIC 对齐（set 在 FIX 排空时触发） |
| 1 | mode 2 | AIC → 配对 2×AIV | PIPE_MTE2 | PIPE_MTE2（AIV 侧） | 通知 AIV 可读 GU（AIV wait 挂窄 pipe 保留预取旁路） |
| 2 | mode 2 | AIV → 配对 AIC | PIPE_MTE3 | PIPE_S（AIC 侧） | AIV 的 H 行全部落 GM（set 在 MTE3 排空时触发） |
| 3 | mode 0 | 全体 AIC | PIPE_MTE2 | PIPE_S | 全体 AIC 再对齐 → 全部 H 就绪，down GEMM 开读 |

要点（M3 实测，已报 tower finding 建议并入 docs/05 §4.1）：

- `CrossCoreSetFlag` 是 **drain 触发**（"挂"在 pipe 上，等该 pipe 排空），不是 pipe FIFO
  指令序——所以 set 一律挂在**生产 pipe**（FIX=Fixpipe 写 GM、MTE3=DataCopy 写 GM）。
- Wait 只阻塞所挂 pipe 的后续**下发**；**链式同步里"wait 之后还要发 set"时，这个 wait
  必须挂 PIPE_S**（阻塞 scalar，CANN `SyncAllImpl` 同款用法），挂窄 pipe 的 wait 给不了
  后续 set 的次序保证（初版挂 PIPE_MTE2 导致 AIV 读到旧 GM 数据）。
- 相邻同步点 flagId 各不相同（0/1/2/3）；mode 2 双向配对：1 set(AIC) 配 2 wait(AIV)，
  2 set(AIV) 配 1 wait(AIC)。
- Set 侧全程不用 PIPE_S/PIPE_ALL；AIC/AIV 各自 BufferID 空间独立（AIC 0..6 同 m1，
  AIV 0=GM→V 行缓冲、1=V→MTE3 行缓冲）。
- **核内 UB 跨 pipe 交接一律 `BufAcquire`/`BufRelease`**（= `GetBufInternal<pipe,false>` /
  `RlsBufInternal<pipe,false>`，release **阻塞释放**（`false` = CANN `ASC_LOCK_BLOCK` 默认；`true` = `NON_BLOCK`；两种模式都等本 pipe 已发射指令落地）；逐字抄 `m12_rmsnorm_gated.asc` 的 `BufAcquire`/`BufRelease` 与 `m5_swiglu_quant.asc` 同名封装）——
  公开 `Mutex::Lock/Unlock` 的 mode 硬编码 0（= `false`，与本仓 `BufRelease` 同模式）；V 侧 UB 写对 MTE3 的可见性
  由成对的 acquire/release 建立（本次 VF 化实测记录：切到 `BufAcquire`/`BufRelease` 之前 h 落盘只可见一部分）。

## 重量化规范（AIV 与 host golden 同一规范，逐字节一致）

**M54 起本核的 AIV 重量化 = 官方 ops-nn 序列（floor/OCP）**，与 `m13_moe_layer` 的
`VecQuantStage::MxQuantComputeScale`/`MxQuantComputeDataFP4`、`tools/golden` 的
`quantize_ocp` 同一规范（M26 裁决：floor/OCP 为 default and authoritative，ceil 为 legacy）：

- group=32 沿 K（down 的 K=640 → 20 组）。**scale byte（floor）**：`byte = E - 2`，E = 组内
  最大 |h| 的 fp32 指数域；`E <= 2`（零/次正规/退化）→ `0`，`E == 255`（±Inf/NaN）→ `0xFF`。
  等价于官方 `shared = maxexp - emax`（`emax = 0x0100`）的 `shared >> 7`。
- **data（CAST_ROUND）**：`h × halfScale`，`halfScale = 2^(127-byte) = 1/scale`
  （官方 `halfScale = 0x7F00 - shared`）；byte 0 → halfScale 0、byte 0xFF → halfScale NaN。
  取整 = CAST_ROUND（平局远离零、饱和到 ±6、保留 `-0` 符号位、NaN → code 0）；
  code 7 = 6.0。
- **h 域（唯一遗留差异）**：官方量化器的输入域是 **bf16**，m13 亦然；本核的 h 是 **fp32**
  （silu 五元组与乘法都在 fp32）。本 mission **不**把 h 额外舍入到 bf16——host 镜像的 double
  silu 与 device 的 Reg `Exp/Div` 可差 ≤2 fp32 ulp，额外做 bf16 舍入会把该差异放大成逐字节
  T1 判据的失配（破坏零回归）。因此量化器本身（scale 规则 / halfScale / CAST_ROUND）与 m13
  逐位一致（§5.3 的逐位对拍证明），**输入域（fp32 vs bf16 h）是收敛后唯一的遗留差异**，
  其影响见 §5.3 报告项。
- **本 mission 定性（仓库卫生 + 消除 legacy 规范）**：先确认 m3 的量化器**没有**被任何在役
  代码消费（§5.0），再把它从 legacy 收敛到权威规范。收敛同时修掉了一处 **data 缩放方向反了**
  的实现缺陷（§5.3），该缺陷此前因验收 golden 与 device 同源而未被发现。

## 校验方法与结果

host 侧（C++，无 libm）：读真实权重 → 确定性 MXFP4 激活（Hash3，每槽位独立 salt，64 行）→
device 跑 kernel → 逐槽位比对：

1. **gate_up 输出**：dequant(A_e) @ dequant(Wgu_e)^T（double），参考值 round 到 bf16 网格，
   容差 `1e-2|expect| + 1e-3·√K·rowMaxA·rowMaxB`（m1 同款）；
2. **H 重量化字节（判定项，T1）**：与 host 同规范量化器 `QuantRowH/E2M1CodeH/QuantScaleByteH`
   （M54 起为 floor/OCP + CAST_ROUND + halfScale）逐字节比对（每行 320B nibble + 20B scale）；
3. **down 输出**：dequant(H_ref) @ dequant(Wdn_e)^T，同容差公式；
4. **自检 vs 生产（判定项，T4）**：同一次运行里用 `vf-probe` 的单 VL VF 对同一批 device GU 行
   重算 H，与生产路径落 GM 的 H 字节**逐字节**比对（`probe==production`）。

实测（Ascend950PR，CANN 9.1.0，28 AIC + 56 AIV）：

| 组 | 判定项 | 结果 |
|---|---|---|
| m=1 | 8 gate_up + 8 h_quant(T1) + 8 down + 1 probe==production | 全 PASS，maxRelDiff=0，H 字节 0 失配 |
| m=33 | 8 gate_up + 8 h_quant(T1) + 8 down + 1 probe==production | 全 PASS，maxRelDiff=0，H 字节 0 失配 |

**判据口径（可复算）**：M54 的验收日志 `evidence/accept_run_m54.log` 里判定项 = **50 条**
（结尾带 `PASS (` 的行：16 gate_up + 16 h_quant + 16 down + 2 probe==production），
`echo $?` = 0，末行 `===== ALL PASS =====`（banner 不计入判据）。

**M54 语义重基线前后（判据数 / PASS 数）对照**（同一可执行、同一数据；口径 = 结尾带 `PASS (` 的判定行，banner 不计）：

| 运行 | 判定项数 | PASS 数 | 日志 |
|---|---|---|---|
| M54 前（M47 VF 化后，ceil 语义） | 50 | 50 | `evidence/accept_run_after_vf.log` |
| M54 后（floor/OCP 语义重基线） | 50 | 50 | `evidence/accept_run_m54.log` |
| M47 VF 化前（经典实现，ceil） | 32 | 32 | `evidence/accept_run_before_vf.log` |

（M47 那两行是**位级不变**的 VF 化（§5.1）；M54 这一行是**语义重基线**（§5.4）——两者性质不同，
不可互相引用。50 条在 M54 前后不变，是因为 H 字节判据（`h_quant`）与 down 判据都由同源的 host
镜像给出，换规范后 device 与镜像一起改、逐字节仍为 0 失配。）

全部输出与 bf16 网格化的 double 参考**逐位一致**（maxRelDiff = 0.00e+00），
H 的 packed/scale 字节与 host 量化器零失配。

## 5. 位级影响面与语义重基线

> 口径：`docs/17` §1.1 分档。判据数、日志、改前/改后对照均在本 commit 内自洽。
> **M47 与 M54 的分工**：§5.1/§5.2 是 **M47**（AIV 段全量 VF 化，**位级不变**，只换实现形态）；
> §5.3/§5.4 是 **M54**（量化器收敛到 floor/OCP，**语义重基线**，H 字节会变）。两者性质不同，
> 判据对数（50 条）不变但含义不同，不可互相引用。

### 5.0 本 mission 的定性：m3 的量化器无在役消费（先查后改）

任务要求先确认 m3 的量化器有没有被任何在役路径 import/拷贝。全仓 grep（`m13_moe_layer/**`、
`m15_layer_loop/**`、`m17_moe_real/**`、`m8_permute/**`、`tools/golden/**`、`baseline_env/**`、
`docs/**`，以及 vllm 侧——本仓只有 `docs/16` 的接入方案，无 vllm 代码）结论：

| 事实 | 证据 |
|---|---|
| m3 是**独立 CMake 工程 + 自带 `main()`**，无任何文件 `#include` 它 | `m3_grouped_gemm/CMakeLists.txt`（独立 `find_package(ASC)`）；全仓 `#include "m3...` 0 命中 |
| m13/m15/m17 只 **lift 了 m3 的 AIC `MXFP4GemmItem`**（GEMM 数据通路），**没有**用它的量化器 | `m13_moe_layer.asc` / `m15_layer_loop/m15_moe_layer.h` / `m17_moe_real.asc` 的注释「m3_grouped_gemm 的 MXFP4GemmItem 逐行 lift」；其量化段是 m5 的 `VecQuantStage`（「小改 C：m3-AIV 标量量化器被替换」） |
| `tools/golden` 把 ceil 规则标为 legacy，且明写它是「被 docs/12 §6 小改 C 替换掉的 m3 标量量化器规则」 | `tools/golden/README.md`、`gen_dataset.py` docstring、`moe_block_ref.SCALE_RULE_DOC` |

⇒ 无在役消费，可以按「消除仓库内最后一处 legacy 规范」的定性来做（不存在"改 m3 会改坏别人"
的连锁风险）。**注意**：lift 的是 `MXFP4GemmItem`（AIC GEMM），本 mission 不动它。

### 5.1 M47：纯机械替换（位级不变，T1 判据不降档）

| 位置 | 改动 | 为什么位级不变 |
|---|---|---|
| `ProcessAiv` 里的经典 `AscendC::Cast<float,bfloat16_t>`（原实现） | `Reg::LoadAlign<LoadDist::DIST_UNPACK_B16>` + `Reg::Cast<float,bfloat16_t,M3_CT_B16_B32>` | bf16→fp32 **无损**（8 位指数 + 7 位尾数全部保留），任何取整模式都逐位一致 |
| `M3GroupedGemm::QuantRow` 标量量化器（原实现，20 组 × 32 元素） | 全 VF：`And 0x7FFFFFFF` 取绝对值 → 半区置 0 后整寄存器 `Reduce<MAX>` 求组 amax → fp32 位域算 scale 字节（`ShiftRights 23` / `And 0x7FFFFF` / `Compares GT 0x400000` / `Adds -2` / `Select`）→ `byte<<23` 得 invScale（2 的幂乘 = 精确）→ 7 个阈值 `Compares`+`Select`+`Add` 计数 + 3 个平局取偶修正 + 符号位 → `Cast<int32_t,float>` → `DeInterleave`+`ShiftLefts 4`+`Or` 拼 nibble → `StoreAlign<DIST_PACK4_B32>` | 每一步都在**同一数据域**做同一运算：fp32 乘 2 的幂是精确的；整数域位运算与比较链逐位等价；amax 用 `Max`（幂等、精确）替代顺序比较，结果不变；nibble 位序（低半字节=偶元素）与 host 契约一致 |
| 生产 AIV 的 `DataCopy` | 元素计数重载 → 显式 `DataCopyParams{1, 块数, 0, 0}`（`docs/12 §6 A` 点名的 m3-AIV `DataCopy(guRaw, guOutGM16[...], GU_N)` 等） | 修的是**未初始化栈垃圾**（M16 latent：元素计数重载的 blockCount 随机翻倍），字节内容不变 |
| 核内 UB 交接 | `Mutex::Lock/Unlock` → `BufAcquire/BufRelease`（阻塞释放） | 只影响可见性，不改数值 |

**T1 证据**：`quant-pack` 0/2560、`quant-scale` 0/160 字节不符（`vf-probe`，合成 amax 网格
+ 阈值/平局用例）；验收 16 条 `h_quant` 判定项在 m=1/m=33 全部 **0 字节失配**；
`probe==production`（T4）0 字节失配。

> 上表描述的是 **M47 当时**的实现（ceil + `byte<<23` 乘 scale）。M54 已把量化段收敛到
> floor/OCP + `halfScale = 2^(127-byte)` 乘 1/scale（§5.3/§5.4），M47 的 **VF 结构未回退**
> （`__VEC_SCOPE__` / 寄存器算子 / `Extract`/`Sort32` 合规状态均保留）。

### 5.2 改了舍入点的一处：经典 `AscendC::Silu` → Reg 五元组（`M3VfSwiGluQuantRow` 内）（`docs/17` **T3** 判）

经典 `AscendC::Silu` 的超越函数近似 ≠ `Muls(-1)/Exp/Adds(1)/Div` 组合，故该段按 T3 判：
**ε 逐项推导**（`M3_VF_SILU_EPS` 常量旁有同样注释）：

| 项 | 来源 | 贡献（相对） |
|---|---|---|
| ① `Exp(-g)` | 官方精度表（Ascend 950PR&950DT）：**1 ulp**（官方文档 `asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md` 的 Reg 矢量计算精度表「基础算术 / Exp」行；`.../basic_arithmetic/Exp.md` 原文「最大精度误差为1ulp」；行号随文档变动，以标题定位） | ≤ 2⁻²³ |
| ② `t = 1 + Exp_c` | 一次 IEEE 加舍入 | ≤ 2⁻²⁴ |
| ③ `s = g / t` | `Div` 亦为 **1 ulp**（同表「基础算术 / Div」行）；叠加 ② 的传递 `Δt/t ≤ 2⁻²³+2⁻²⁴` | ≤ 2.5·2⁻²³ |
| ④ `h = s * u` | 一次 IEEE 乘舍入 | ≤ +2⁻²⁴ ⇒ 设备侧 ≤ 6·2⁻²⁴ |
| ⑤ 参考侧 | `fp32(double silu) * u` 两次舍入 ≤ 2·2⁻²⁴；另加 host 参考自身的 `ExpD` 级数截断（10 项泰勒，相对 ~2⁻³⁷） | ≤ 2·2⁻²⁴ + 2⁻³⁷ |
| **合计** | 2⁻³⁷ 比 2⁻²⁴ 小 13 个数量级，不改变截断后的整数幂取值 | **ε ≤ 8·2⁻²⁴ + 2⁻³⁷ → 取 ε = 2⁻²¹ = 4.768e-7** |

判据：**逐元素** `|h_dev − h_ref| ≤ ε·Σ|terms| + 0.5·ulp(h_dev)`，其中 `Σ|terms| = |silu(gate)|·|up| = |h|`。
覆盖：16 行 × 640 元素 = **10240 个元素**（gate/up 取 8 档动态范围：|gate| ≤ 60、|up| ≤ 16），
**超界 0/10240**；报告项：实测 max |Δ| = **2.000 ulp**、T3 界最大占用 **0.414**。
**非有限与下溢角落（本判据的边界，记录在此）**：
- `Exp` 官方规格注明 "not support denormalized numbers"（FTZ）；本用例 gate ∈ [−60, 60] ⇒
  `exp(∓g) ∈ [8.8e-27, 1.1e26]` 全为正常数，**不触发 FTZ/上溢**，故逐元素判据无需额外分支。
- **NaN 不可达（可论证）**：gate/up 都是**有限 bf16**，`silu` 五元组里只有 `Exp/Div/Mul` 三种
  运算作用在有限值上（`Exp(-g)` 在 g∈[−60,60] 内有限非零 ⇒ `1+Exp` 有限非零 ⇒ `Div` 有限），
  故 `h = silu(gate)·up` **不可能是 NaN**；因此 T3 判据无需为 NaN 预留分支。
  （m13 §5.4 记录的"非有限组的符号位/零码分叉"只与非有限输入有关，本段不可达。）
- **±Inf 角落**：M54 起 device 与官方/m13 一致——±Inf/NaN 组 e8m0 byte = `0xFF`、halfScale = NaN、
  data 侧 `Cast<fp4>(NaN) = 0`（`nonfinite` 判据覆盖；此前 ceil 规则走 `code 7/15` 饱和，已随
  收敛修正）。该角落不在本段逐元素判据内（本用例不产生 Inf），由 `nonfinite` 与 §5.3 的对拍覆盖。
- `vf-probe` 的 `scale-corner` 把 `E <= 2`/次正规单列成一栏，避免混进前两条同解区间的通过率。

### 5.3 与 m13 `VecQuantStage` 的逐位对拍（M54：同一规范、逐字节一致）

M47 的结论是「m3 = ceil、m13 = floor，只差规范」。M54 把 m3 收敛到 floor/OCP 后，**两个量化器
在同一批输入上逐字节一致**。对拍方式（可离线复跑）：

- 被测对象：device（**生产同一份 VF 量化函数**，经 `vf-dump` 落盘的 `pack_dev.bin`/`scale_dev.bin`）；
- 参考 1（**m13 自己的**）：`m13_moe_layer/check_ref.py::quant_mxfp4_hw`（逐句对照 m13 的
  `VecQuantStage::MxQuantComputeScale` / `MxQuantComputeDataFP4`）；
- 参考 2（**独立见证**）：`tools/golden/moe_block_ref.py::quantize_ocp`（官方 ops-nn 序列的逐句
  numpy 转写，M26；M50 已证明它与 m13 device 逐字节一致）；
- 输入：128 行 × 640，全部 **bf16 网格值** + 逐组端点（4/5/6/6+1ulp/7/8−1ulp/8·2^k）+
  退化/非有限（0 / ±0 / 次正规 / 最小正规 / ±Inf / ±NaN / 组内混 Inf）。

结果（`evidence/m54_check_m13_cross.log`；dump 的 sha256 见 `evidence/quant_cross/sha256.txt`）：

| 对拍 | pack | scale |
|---|---|---|
| device vs m13 `quant_mxfp4_hw` | **0/40960 字节不符** | **0/2560 字节不符** |
| device vs `tools/golden.quantize_ocp` | **0/40960** | **0/2560** |
| device vs legacy `quantize_ceil_legacy`（**负向对照，必须不一致**） | 15831/40960 | 1589/2560 |

⇒ m3 与 m13 的量化器**逐位一致**（scale 规则、halfScale 方向、CAST_ROUND、±Inf/NaN/退化角落全一致）。

**剩下的差异来源：h 的输入域（fp32 vs bf16）**。m13 的 `SwigluComputeTile` 先把 fp32 的
`silu*up` `Cast<bfloat16_t>`（CAST_RINT）再量化；本核的 h 保持 fp32。因此：

- 在 **bf16 网格输入**上两者逐位一致（上表；官方 op 的输入域本就是 bf16）；
- 生产里 h 是**非 bf16 网格的 fp32**，与「先把 h RNE 到 bf16 再量化」会有少量差异。本 mission
  **不改** h 精度，理由已量化：host 镜像的 double-silu 与 device Reg `Exp/Div` 差 ≤2 fp32 ulp
  （§5.2 报告项），而 bf16 网格比 fp32 粗 2^15 ⇒ 约 `2·2^-23 / 2^-8 = 2^-15 ≈ 0.003%` 的元素会
  被 RNE 舍到不同 bf16 值（评审在真实验收数据上实测 ≈ **0.002%**），足以把 0 失配的 T1 `h_quant`
  判据变成非零失配、破坏零回归。故这是**唯一遗留差异**，属"输入域"而非"量化器规范"。若要消除，
  需先把 silu 的 device/host 逐位对齐，或改判据口径。

**同一批 amax 输入的两规范分类对拍**（M54 起 device 必须 = floor）：

- `scale-same` 97 组（ceil == floor，device == 二者）、`scale-diverge` 59 组（ceil == floor+1，
  device == floor == ceil−1）、`scale-corner` 4 组（`E <= 2` / 次正规，device == floor）、零组 2 组
  ——全部 PASS（`evidence/vf_selftest_run_m54.log`）。
- 端点实测读数（`scale-endpoint`，k = 0/−8/8/20，28 个端点全部与解析分类一致）：

  | 端点 | 实测 device | 解析 floor | legacy ceil | 分类 |
  |---|---|---|---|---|
  | `4/5/6·2^k` | 127 | 127 | 127 | 同解 |
  | `6·2^k + 1ulp` | **127** | 127 | 128 | 分歧（device 取 floor） |
  | `7·2^k` | **127** | 127 | 128 | 分歧（device 取 floor） |
  | `8·2^k − 1ulp` | **127** | 127 | 128 | 分歧（device 取 floor） |
  | `8·2^k` | 128 | 128 | 128 | 同解（下一区间左端点） |

  （k 各值读数同形；`vf-probe` 逐行打印。）

**覆盖范围与负向对照（tower 规则「测量工具必须交代覆盖范围」）**：判定项的计数与匹配器同源
（脚本从 `meta.txt` 读 rows/k/group，不硬编码；pack/scale 逐字节计数由同一次 load 得出）。
dump 的 scale 只落**有效 group 字节/行**（`scale_row_stride == k/group`）：device 行缓冲是 32 B
（20 有效），尾部 12 B padding 是陈旧 UB、**不确定**，故不落盘也不参与比较（README §7 已登记的
布局性质；这也是 `scale_dev.bin` 的 sha256 可复现的原因）。**已知会被漏掉**：dump 输入全部是
bf16 网格值，**非 bf16 网格的生产 fp32 h 不在对拍覆盖内**（正是上面那条遗留差异），其影响不在
本脚本判定项内。

**负向对照 `quantize_ceil_legacy`（legacy ceil 的权威实现）必须、且确实与 device 不一致**
（否则判据无法区分两套规范）。注意措辞要限定到 **scale 字节规则**：pre-M54 的 m3 device
**scale 规则**走 ceil，但 **data 方向与 legacy 相反**（§5.4）⇒ 它的 `pack` 与
`quantize_ceil_legacy` 差 **37895/40960 (92.5%)**，`scale` 字节也差 **220/2560**（全部落在
次正规/±Inf/±NaN 的退化行：base commit 的 `uint8` 环绕 vs golden 的 clamp；随机+端点行内
scale 字节 0/2240 不同）。即 pre-M54 的 device 在 data 上**既不等于 legacy、也不等于 floor**；
M54 收敛后 device 与 floor 逐字节一致（上表）。

### 5.4 M54：语义重基线（ceil -> floor/OCP，并修一处 data 方向缺陷）

与 §5.1/§5.2 的 M47（**位级不变**）不同，本节是**语义重基线**——H 字节会变，不是"改实现不改结果"。
三处一起改（外加一处本次发现的缺陷）：

| # | 项 | M54 前 | M54 后（权威 floor/OCP） | 依据 |
|---|---|---|---|---|
| ① | scale byte | `E-2+(M>0x400000)`（ceil） | `clip(E-2, 0, 254)`，`E==255 -> 0xFF`（floor） | 官方 `shared = maxexp - emax` 的 `shared >> 7`；m13 `MxQuantComputeScale`；`tools/golden.quantize_ocp` |
| ② | data 取整 | 最近值 + 平局取偶 | CAST_ROUND（平局远离零、饱和 ±6、保留 `-0`、NaN -> 0） | 官方 `Cast<fp4, bf16, castTraitRM_Round>`；`H_E2M1Code`/`e2m1_encode_away` |
| ③ | data 缩放方向 | `v = h × 2^(byte-127)`（**乘 scale**） | `v = h × 2^(127-byte)`（**乘 halfScale = 1/scale**） | 官方 `Mul(vdExp, vdExp, halfScale)`（`halfScale = 0x7F00 - shared`）；m13 同名序列 |
| ④ | h 输入域 | fp32 | **保持 fp32**（唯一遗留差异） | §5.3 |

**③ 的定性（本次发现的一处实现缺陷）**：M54 前 m3 把 h **乘以** `2^(byte-127)`，而 MXFP4 的
正确操作是 `code = round(h / scale)`，即乘以 `1/scale = 2^(127-byte)`；两者互为倒数。故当组
amax 落在 `[4, 6]`（byte 127 ⇒ scale = 1）之外时（多数组），H 的反量化严重偏离 h。
`analyze_rebaseline.py`（pre-M54 模型按 base commit 的**位模式**语义 `f = bitcast((uint32)b << 23)`
构造乘子，逐位忠实）在 `evidence/quant_cross/h.bin` 上量化：pre-M54 的 H 反量化相对误差
**mean 1.46e75 / p50 142× / p99 2.04e39 / max 1.74e77**，post-M54 与 e2m1/group32 的量化噪声
同量级（mean 0.665）；pack 字节 pre/post 差异 **38823/40960 (94.8%)**；且 pre-M54 与
`tools/golden.quantize_ceil_legacy`（legacy ceil 的权威实现）也差 **37895/40960 (92.5%)** ——
即它在 data 方向既不等于 legacy（它自称的规范）也不等于 floor/OCP（`evidence/m54_quant_rebaseline.log`）。

**决定性旁证**：仓库里同一套 legacy ceil 规则的**所有其它实现都做除法**——
`tools/golden.quantize_ceil_legacy` 是 `x3 / scale_f`、m13 `quantize_mxfp4_f32` 是 `g / scale_f`；
pre-M54 m3 是唯一的乘法。⇒ 这是"方向反了"的实现缺陷，不是第二套约定。

> 该缺陷此前**未被验收咬住**：m3 的 golden 与 device 同源——`h_quant` 比的是 device H 字节 vs
> 同规则 host 镜像，`down` 的参考 `H_ref` 又由同一批 H 字节反量化得到，两者都跟着 device 一起
> "错得自洽"。这正是本 mission 引入**独立参考**（m13 自己的 `quant_mxfp4_hw` + `tools/golden`
> 的 `quantize_ocp` 双重对拍）的原因：收敛前该对拍会 FAIL（负向对照已量化 15831/40960 的差异）。

**判据变化**：`vf-probe` 判定项 **8 -> 9 条**（新增 `nonfinite`：±Inf/NaN 组 -> byte `0xFF`、
code 全 0；`scale-corner` 增加 `E <= 2` 用例）；验收判据仍为 **50 条**（§「校验方法与结果」的
前后对照表）。

### 5.5 两条既有 quirk 的最终处置

| quirk | 处置 |
|---|---|
| M10/M3 README「3510 连续两个经典 `Cast<float,bfloat16_t>` 后者输出错乱」的规避（整行一次 Cast + 把 `silu*up` 移进 scalar 量化器） | **删除**。规避的前提是经典 `Cast` 共用保留 UB scratch（`__ASC_USE_RESERVED_UBUF__(3510)`）；VF 化后是寄存器 `Reg::Cast`，无 UB scratch，且同一循环内连续两条 `Cast<float,bfloat16_t>` 的结果**逐位正确**（`quant-pack`/`quant-scale`/验收 h_quant/probe==production 四处零字节失配即证据）。代码与本文档中不再保留该规避及其理由。 |
| 「m3-AIV 用 PIPE_S 串标量量化器」 | **删除**。量化已回 V pipe，AIV 段不再有 S 参与的数据路径（`docs/12 §6`「小改 C」的意图在 m3 内落地）。 |
| 元素计数 `DataCopy` 重载（未初始化栈垃圾） | 本路径全部改显式 `DataCopyParams{1,块数,0,0}`（`docs/12 §6 A` 点名的 m3-AIV `DataCopy(guRaw, guOutGM16[...], GU_N)`）。 |

### 5.6 已知未解决问题（已报 tower finding，不在本 mission 范围）

**VF 内"多次 256B 落盘"的尾部丢失**：一个 `__VEC_SCOPE__` 循环里做 10 次
「`LoadAlign(UNPACK_B16)` + `Cast` + `Store<float>(64 lane = 256B)`」时，**只有前 5 次迭代的
数据落到 UB**（后 5 次为 0/陈旧值）；换成 32B（nibble）与 2B（scale）落盘则 10 次全对，
8 次不循环的 256B 落盘也全对。最小复现器：`m3_vf_probe_kernel` 的 **mode 6**（`M3VfCastOnlyRow`，只做
`LoadAlign(UNPACK_B16)` + `Cast` + `Store<float>(64 lane)`）——`vf-probe` 的最后一条报告项即它：
**10 次迭代中前 5 次逐位可见**，后 5 次为陈旧/零值。
影响面：只影响**把中间量写回 UB 做 dump** 的场景；生产路径（h 不落 UB、直接进量化器）
与自检的字节判据（改用**一个 VF 只落一个 VL** 的 mode 9）均不受影响。

## 6. AIV VF 自检（`vf-probe`）

自检复用**生产同一份** VF 函数（`M3VfSwiGluQuantRow` / `M3VfQuantTwoGroups` /
`M3VfSiluOneVl`），`__vector__` 单核发射，行序确定。判定项（均打印 `[M3-VF] <tag> : PASS (...)`）：

| 判据 | 档 | 内容 | 实测 |
|---|---|---|---|
| `quant-pack` | T1 | 合成 amax 网格（8 行 × 20 组）device nibble 字节 vs host 镜像（floor/OCP + CAST_ROUND） | 0/2560 不符 |
| `quant-scale` | T1 | 同上 scale 字节（floor/OCP） | 0/160 不符 |
| `scale-same` | T1 | 两规范同解区间 device == floor == ceil | 97 组全对 |
| `scale-diverge` | T1 | 分歧区间 device == floor == ceil−1（M54 起取 floor） | 59 组全对 |
| `scale-corner` | T1 | clamp 角落（`E <= 2`/次正规）device == floor | 4 组全对 |
| `e2m1-threshold` | T1 | 阈值/±1ulp/平局/±0/±6.0/近零用例逐字节 + 16 个 code 全覆盖（CAST_ROUND 平局远离零） | 0 字节不符、16/16 |
| `scale-endpoint` | T1 | 区间端点（k=0/−8/8/20 × 4·2^k/5·2^k/6·2^k/6·2^k+1ulp/7·2^k/8·2^k−1ulp/8·2^k）实测读数与解析分类 | 28/28 一致 |
| `nonfinite` | T1 | ±Inf/NaN 组 -> scale byte `0xFF`、nibble 全 0（官方 :401/:406/:413） | 0/80 字节、0/1280 nibble |
| `silu-t3(VL)` | **T3** | 10240 元素逐元素满足 §5.2 的 ε 界 | 0/10240 超界 |
| 报告项 | — | silu 实测 max \|Δ\|（2.000 ulp）、T3 界最大占用（0.414）、非有限元素数、同解/分歧/角落组数、code 覆盖集、整行落盘异常（§5.6） | — |

判定项 **9 条**（M54 起；M47 时为 8 条），与 `evidence/vf_selftest_run_m54.log` 逐条对应
（`[M3-VF] <tag> : PASS (` 行）。`quant-pack`/`quant-scale` 的 host 镜像同时是 §5.3 里
「bf16 网格输入下与 m13 逐位一致」的 m3 侧实现。

`vf-prim`（诊断，不参与判定）打印 VF 原语实测语义（`DIST_DINTLV_B32` 偶奇拆分、
`MaskPattern` VL32/H/ALL 的 lane 范围、`Select` 取 src0、`Duplicate<LOWEST>` 广播、
`DIST_PACK4_B32` 的"每 32bit lane 落 1 字节"行为、`Store<uint8_t>(.,.,2)` 的 2 字节落盘）
——本文件的实现直接建立在这些实测读之上（`evidence/vf_prim_run.log`）。

## 7. 已知限制 / 后续

- **性能未优化**（正确性 milestone）：A 随 item 重复搬运（跨 nBlock 可复用同槽位 A，
  80KB+5KB 可驻留 L1）；down 权重未在 barrier 前预取（PIPE_S wait 阻塞了 scalar 预取重叠，
  后续可按 docs/05 §4 的"窄 pipe wait + 数据面旁路"重构）；AIV 行处理无流水。
  AIV VF 段每行 10 迭代（每迭代 64 元素，2 个 scale 组）。
- **量化规范**：M54 已收敛到官方 floor/OCP（§5.3/§5.4），与 m13 `VecQuantStage` 在 bf16 网格
  输入上逐位一致。**唯一遗留差异 = h 的输入域**（本核 fp32 vs 官方/m13 的 bf16），见 §5.3；
  若要消除需先把 silu 的 device/host 逐位对齐或改判据口径。
- 专家槽位数、BASE_N 等均为编译期常量；`t_e ≤ 64`（MAXM）。
- K 尾块（K % 128 ≠ 0）不支持（同 m1；模型两形状 K=640/2560 均可整除）。
- 共享专家、router/重排不在本 milestone（host 用确定性 counts 直接构造分组）。
- H 的 scale 行 GM 布局 padded 到 32B（kernel 写 32B/行，20B 有效）。
- §5.6 的 VF 落盘 loop 异常（已报 finding）：若后续要在生产路径里 dump 中间量，
  需先解掉它或改用"一个 VF 一个 VL"的形式。
- **仍把 m3 描述为 ceil 载体的文字**（本 mission 按边界**只报不改**，不在 scope；已核实
  `docs/17` 的 `ceil` 命中为 0，`tools/golden` 的 m3-ceil 引用是**正确的 legacy 标签**、无需改）：
  `m13_moe_layer/README.md` §5.3（「等价于 m3 标量量化器里 `+ (M > 0x400000)`」）与『已知限制』6、
  `m15_layer_loop/README.md`（§量规差异）、`m17_moe_real/README.md`（§量化规范参考项）。
  这三处读起来是历史的「m3 原标量量化器」，可接受；若后续要与 M54 后的 m3 对齐，需 routing 到这三个文件。

## 8. 证据归档（`m3_grouped_gemm/evidence/`）

| 文件 | 内容 |
|---|---|
| `accept_run_m54.log` | **M54** 全链验收日志（m=1/m=33 × 8 槽位；50 条判定项全 PASS，rc=0，末行 `===== ALL PASS =====`） |
| `accept_run_after_vf.log` / `accept_run_before_vf.log` | **M47** VF 化后 / 前（ceil 语义，位级不变；50 / 32 条判定项）——用于 §5.1 与重基线对照表的"M47 前"基线 |
| `vf_selftest_run_m54.log` | **M54** `vf-probe` 自检日志（9 条判定项全 PASS，与 §6 表逐条对应） |
| `m54_check_m13_cross.log` | **M54** 与 m13 `quant_mxfp4_hw` / tools/golden `quantize_ocp` 的逐字节对拍（5/5 PASS，含 legacy ceil 负向对照） |
| `m54_quant_rebaseline.log` | **M54** 语义重基线量化（scale ceil→floor、data 方向、pre/post H 反量化重建误差） |
| `quant_cross/{h,pack_dev,scale_dev}.bin` + `meta.txt` + `sha256.txt` | 对拍用的输入 dump、device 输出 dump 与 sha256。`scale_dev.bin` **只含有效 group 字节/行**（`scale_row_stride == k/group`；device 行缓冲尾部 12 B padding 是陈旧 UB，不落盘）⇒ 四个产物均**可离线复算**（连跑 3 次哈希一致，见下） |
| `m54_dump_hash_stability.log` | `vf-dump` 同一二进制/同一目录连跑 3 次的 sha256 读数（4 个产物逐次一致 ⇒ 可复现，P2-1 修复证据） |
| `vf_prim_run.log` | `vf-prim` 原语语义实测读数（§6 末） |
| `vf_selftest_run.log` | **M47** 的 `vf-probe` 归档（8 条判定项；`analyze_rebaseline.py` 用它证明"pre-M54 device == pre-M54 镜像"） |
