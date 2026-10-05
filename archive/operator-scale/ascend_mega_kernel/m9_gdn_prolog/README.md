# M9(M16)：GDN 预处理核（conv1d+SiLU / l2norm / gating 融合，纯 AIV）

Qwen3.8-Flash-Next GDN 线性注意力 **decode（m=1）递推前的全部 AIV 预处理**，
单核融合：**conv1d depthwise K=4（10240ch，+bias+SiLU，conv_state 环形原位更新）
→ q/k per-key-head l2norm（仅 q 乘 1/√128）→ gating（g/β）**。
输出直接按 m4（M11）递推核的 kernel 面契约排布，可无缝交接。
数值与 numpy float32 参考逐输出一致（组合容差 1e-5 相对 + 1e-6 绝对地板，
conv_state 写回**位精确**），5 组 blockDim 配置（56/28/16/8/4 AIV）全部 PASS。

**另有一个独立 m>1 入口** `gdn_prolog_mt_kernel`（M147）：同一 conv1d/l2norm/gating 数学，做**跨 token
窗口 + conv_state 环形交接**，输出几何对齐**相位 A（GDN prefill）的输入契约**（q/k 尾部 64 行置零、
v `[48,m,128]`、g/β `[48,align8(m)]`），并在 `qScaleOn` 上区分「相位 A 契约（默认，q 不乘）」与
「m9 独立模块语义（q ×1/√128，交叉复现用）」两种口径。详见文末
<「m>1（MTP / prefill prolog）段体（M147）」>。m=1 入口一字未动。

donor：`/workspace/ops-transformer/mamba/causal_conv1d`（arch35 环形 conv UB kernel，
ComputeConv1dUnroll 增量 tap 累加写法；`(cache, state_len, dim)` state 布局）+
`/workspace/vllm-ascend/vllm_ascend/ops/triton/fla/sigmoid_gating.py`（gating/softplus 逻辑）。

## 数学与公式（docs/10-gdn-analysis.md §1）

in_proj 输出 `x [1,16480] bf16 = [q 2048 | k 2048 | v 6144 | z 6144 | b 48 | a 48]`。
conv1d 只作用 **q|k|v 前 10240 通道**（z 是 output gate 输入，不属于本核）：

```
窗口 win[c] = [state[0][c], state[1][c], state[2][c], x[c]]      # state 行 j = 第 j 个历史样本
y[c] = bias[c] + Σ_j w[j][c]·win[j][c]        （fp32；w 行 j 与 win 同序，w[0] 配最旧样本）
silu(y) = y / (1 + e^(−y))
q_h = l2norm(silu(q_h)) · (1/√128)，  k_h = l2norm(silu(k_h))      # l2norm: x/√(Σx²+1e-6)
v_h = silu(v_h)                                                    # 不乘 scale
g[h] = −exp(A_log[h]) · softplus(a[h] + dt_bias[h])                # β=1, thr=20：
        softplus(x) = x≤20 ? log1p(e^x) : x                        # （vllm sigmoid_gating 逻辑）
β[h] = sigmoid(b[h]) = 1/(1+e^(−b[h]))
```

**语义决策（已呈塔确认）**：`1/√128` **只乘 q、不乘 k**——docs/10 §1 原文
"q,k=l2norm(eps 1e-6)，q×=1/√128"；vllm fla `chunk.py:244-245`（q/k 均 l2norm）+
`sigmoid_gating.py:277-281`（仅 `b_q = b_q * scale`）。任务书措辞 "q/k ... l2norm×1/√128"
按此模型真实语义实现；若人类本意 k 也乘，kernel 内改一行（`qScale` 条件）即可。

## 输入 layout（kernel 面，全部 bf16 除非注明）

| 张量 | 形状 | 说明 |
|---|---|---|
| x | `[16480]` | in_proj 输出：`[q2048|k2048|v6144|z6144|b48|a48]`；conv 消费前 10240，gating 消费 b/a 尾段 |
| conv_state | `[3][10240]` bf16 | **planar**（donor causal_conv1d 的 `(cache, state_len, dim)`）：行 j = 全通道第 j 个历史样本；**in-place** 更新。docs/10 的 "(10240,3)" 指 dim×stateLen 尺寸 |
| w | `[4][10240]` bf16 | conv 权重，行主序，行 j = tap j（配 win[j]，w[0] 配最旧历史） |
| bias | `[10240]` bf16 | conv bias |
| A_log | `[64]` fp32（48 有效+尾零） | gating 权重 |
| dt_bias | `[64]` fp32（48 有效+尾零） | gating 权重 |

z 段（`x[10240..16384]`）本核不读；b/a 从 x 的 bf16 段解码后与 A_log/dt_bias（fp32）
一起算 gating——与 vllm `npu_fused_qkvzba_split_gating` 消费 packed bf16 一致。

## 输出 layout（kernel 面 = m4 递推核输入契约）

| 张量 | 形状 | 说明 |
|---|---|---|
| q | `[16,128]` fp32 | 每 key head 的 q，已 l2norm×1/√128 |
| k | `[16,128]` fp32 | 每 key head 的 k，已 l2norm（**无 scale**） |
| v | `[48,128]` fp32 | 每 value head 的 v（conv+SiLU 结果） |
| g | `[48,8]` fp32，stride-8 | log-decay，**仅 `[h][0]` 有效**（GM 32B 对齐，m4 用 `{H,1,0,0}` 直读） |
| β | `[48,8]` fp32，stride-8 | 同上 |
| conv_state | `[3][10240]` bf16 | in-place：new[0]=old[1]，new[1]=old[2]，new[2]=x |

与 m4（M11 递推核）的接口：q/k/v/g/β/state 五个张量形状、dtype、stride 约定
**逐字段对齐 m4 README「输入 layout」表**；m4 的 `H=48`、`NK=16`、`GROUP=3`
（hv←hk=hv//3）分组由本核的输出布局天然满足（q/k 按 key head、v/g/β 按 value head）。
集成时递推核直接消费本核输出 GM buffer 即可（同一 layer 内 mode-0 barrier 之后）。

## state 语义

- 布局：planar `[3][10240]`，行 j 是 10240 个通道共享的第 j 个历史输入样本
  （最旧在行 0）。与 ops-transformer causal_conv1d（arch35）的 conv_states 布局一致。
- 更新（每 decode token 一次，原位）：左移一行，`new[2][c] = x[c]`。
- 写回值是输入 x/state 的**纯 bf16 搬移**（无运算），check 按位精确校验。
- w 与窗口的配对：`y[c] = bias + w[0][c]·old[0][c] + w[1][c]·old[1][c] + w[2][c]·old[2][c] + w[3][c]·x[c]`
  （与 donor InitRing/ComputeConv1dUnroll 的 ring[(t+j)%5]·w[j] 及 PyTorch causal conv1d
  左 pad K−1 语义一致）。

## 构建与运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m9_gdn_prolog/build -S m9_gdn_prolog -DCMAKE_BUILD_TYPE=Release
cmake --build m9_gdn_prolog/build -j4
cd m9_gdn_prolog/build && ./m9_gdn_prolog   # dump 各 case 的 .bin 到 cwd（build/ 已被 .gitignore）
/usr/local/python3.12.13/bin/python3 ../check_ref.py
```

独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`），不依赖仓库顶层 CMakeLists.txt。
host 运行 5 个 case：blockDim ∈ {56, 28, 16, 8, 4}（满机 56 AIV 覆盖 24×2+32×1 块，
其余覆盖每 AIV 多 v-head gating、q/k/v 混合所有权、跨 parity 流水），各用不同确定性 salt。

## Kernel 设计

- **核型**：`__vector__ __global__`（AIV-only，3510 quirk 要求），blockDim=AIV 数。
  10240ch 按 128ch（=1 head）切 80 个 block：b<16 为 q head，16..31 为 k head，
  32..79 为 v head（h_v=b−32）；block 按 `b = bid, bid+nblk, ...` 条带划分。
- **数据通路**（全静态地址 + BufferID，无 TPipe/TBuf/TQue/高阶 API）：
  GM --DataCopy(MTE2)--> UB（state 3×256B + x/w/bias 各 256B，parity 双缓冲）
  → UB --LoadAlign(UNPACK_B16)+Cast(fp32)+MulAddDst/Exp/Sqrt/Div/ReduceSum(VEC)--> 寄存器
  → UB --DataCopy(MTE3)--> GM（state 3 行单块写回 / q·k·v 512B / g·β 32B slot）。
- **UB 布局**（自管理地址，全部 32B 对齐，footprint ≈13KB ≪ 248KB）：
  每 parity 2816B：`[0,768)` state 3 行（256B/行）· `[768,1024)` x · `[1024,2048)` w 4 行 ·
  `[2048,2304)` bias · `[2304,2816)` OUT（fp32 128，conv+SiLU 结果；q/k 块原地被 l2norm 覆盖）；
  gating 区：a/b bf16 staging（+UNPACK 预读 slack）· A_log/dt_bias fp32（+slack）·
  g/β slot 各 `[48,8]` fp32 ×2 parity。
- **BufferID 仅 3 个**：`BUF_BK0/1`（block 输入区 MTE2→VEC→MTE3 全生命周期，parity 双缓冲）、
  `BUF_GB`（gating 阵列 MTE2→VEC，首 v block acquire、末 v block release）。
  acquire 立即 / release 阻塞释放（mode=false = CANN ASC_LOCK_BLOCK 默认；两种模式都等本 pipe 已发射指令落地）；g/β slot 写（VEC）→读（MTE3）
  搭 BUF_BK 的阻塞释放便车。跨 block 流水：MTE2 预取 b+nblk ∥ VEC 计算 b ∥ MTE3 写回 b。
- **gating 抽取**（3510 quirk 规避，见下）：staging 对齐整 64-lane 加载 → 全 head 向量化
  gating → `Arange`+`Compares(EQ)` 造 one-hot 掩码 → `Reduce<SUM>` 抽取本 head 的 lane →
  `StoreAlign DIST_FIRST_ELEMENT_B32`（VL1 mask，CANN 同款）落 32B slot。

## 校验方法与结果

host 确定性生成（Hash3 公式；a 每 4 head 一个 ≈20.5± 值，softplus 阈值两分支均覆盖；
A_log/dt_bias 尾零填充），H2D → launch → D2H → 就地 dump 每 case 的
x/conv_state_init/w/bias/A_log/dt_bias 与 conv_state_out/q/k/v/g/beta（g/β 导出紧凑 [48]）。
`check_ref.py` 用 numpy float32 按同公式逐输出比对：conv_state **位型精确一致**；
q/k/v/g/β 组合容差 `|got−exp| ≤ 1e-5·|exp| + 1e-6`。重复运行逐位一致，退出码 0。

实测（Ascend950PR，CANN 9.1.0，2026-09-26，blk=56 / blk=4 代表）：

| case | state | q | k | v | g | β |
|---|---|---|---|---|---|---|
| b56 | PASS（bit-exact） | 3.63e-07 | 3.65e-07 | 2.58e-07 | 1.12e-05 | 1.25e-07 |
| b28 | PASS | 3.53e-07 | 3.26e-07 | 2.18e-07 | 2.96e-06 | 1.00e-07 |
| b16 | PASS | 3.19e-07 | 2.86e-07 | 2.19e-07 | 4.19e-06 | 1.29e-07 |
| b08 | PASS | 2.94e-07 | 3.36e-07 | 2.27e-07 | 4.49e-06 | 1.13e-07 |
| b04 | PASS | 2.95e-07 | 2.35e-07 | 2.28e-07 | 3.17e-06 | 1.48e-07 |

（表中为 maxRelDiff；g 的一行 ~1e-5 出现在 |exp| 小处，maxAbsDiff ≤ 7.6e-6，
判定以组合容差为准。）**5 case × 6 输出组全部 PASS。**

## 3510 quirks（本任务实测，已报 TowerFinding）

1. **元素计数 `DataCopy(dst,src,count)` 重载**：内部 `DataCopyParams` 只填 blockLen，
   blockCount/srcGap/dstGap 是**未初始化栈垃圾**——复制块数随机翻倍、silent 数据错乱。
   **规避：一律显式 `DataCopyParams{1, blockLen, 0, 0}`**（本核全部拷贝遵守；
   m4/m6 使用该重载，存量 latent 风险已报）。
2. **`DataCopyParams` blockCount>1 走 NZ 格式重排**（bf16/256B 块实测被置换，
   m4 的 fp32/512B×128 块幸免）：**多块拆成多条 blockCount=1 单块拷贝**。
3. **`Reg::LoadAlign` 地址必须 32B 对齐**（按元素偏移加载直接 507035，同 §6
   UB→GM DataCopy 对齐 quirk 的向量加载侧）：**对齐整寄存器加载 + one-hot 掩码
   Reduce 抽 lane**（`Arange`+`Compares`+`Reduce<SUM>`，见 GatingHead）。
4. 已有规避：`__vector__ __global__` 声明；UB 偏移全 32B 对齐；无 B32→B16 Cast
   （天然避开 M10 连续 Cast bug）；bf16→fp32 用 m5 已验证的 UNPACK+Cast 全行模式。

## 已知限制 / 后续

- 本入口（`gdn_prolog_kernel`）是单 token（m=1）decode 路径；**MTP 多 token 见文末的
  `gdn_prolog_mt_kernel`（M147）**；prefill chunk 扫描本身（S4）不在本任务（docs/10 §3 为另一套）。
- block 粒度和 head 对齐（128ch）：80 块静态条带划分，负载不匀（56 AIV 时
  24 核 2 块 + 32 核 1 块）。prolog 段带宽 ~0.27MB/token·层（docs/10 §2），
  正确性优先未细扣流水（双缓冲已覆盖主要重叠）。
- A_log/dt_bias 按 `[64]`（48 有效+尾零）传入；若上游只给 `[48]`，需补齐或改
  gating 加载偏移（当前 slack 设计已容忍 64 宽对齐加载）。
- z 段 output gating（RMSNormGated，sigmoid）属后续 op，不在本核。

## 与 donor 的差异点

| | donor（ops-transformer causal_conv1d / vllm gating） | 本核 |
|---|---|---|
| 资源管理 | TPipe/TBuf/TQue + tiling-key 分发 | 全静态地址 + 3 个 BufferID |
| 同步 | SetEvent/WaitFlag（MTE2_V 等） | 只用 BufferID（get/rls_buf，阻塞释放） |
| conv 计算 | ring[(t+j)%5] UB 行 + fp32 权重预 cast | 寄存器增量 tap 累加（MulAddDst），w bf16 随路 cast |
| state 写回 | WriteBackState ring 行 + varLen 增量写 | planar 3 行单块原位写回（m=1 无 varLen） |
| gating | vllm host/torch 侧 softplus+exp | 全 head 向量化 + one-hot 抽取，fp32 寄存器内完成 |
| 通用性 | batch/varlen/cacheIndices/MTP/K∈{2,3,4} | 单 token、整层固定形状、K=4 硬编码 |

---

## m>1（MTP / prefill prolog）段体（M147）

m9 原有段体严格 m=1（文件头 `:2`；`Init` 把 x 当单 token 定尺 `:256-266`）。本目录新增**独立入口**
`gdn_prolog_mt_kernel`（类 `MtGdnProlog`），实现 **m>1 的 conv1d 跨 token 窗口 + conv_state 环形交接 +
l2norm + gating**，几何对齐**相位 A（GDN prefill chunk 扫描）的输入契约** —— 即
`hc(attn) BLK → 相位 A 输入` 那条未接链路的 prolog 段
（`docs/22-prefill-prolog-epilog-wiring.md:24`、§5.2/§5.3）。m=1 入口一字未动，原 5 档回归照跑。

### 输出几何（= 相位 A 输入契约）

| 张量 | 形状 | stride / pad | 出处 |
|---|---|---|---|
| q, k | `[16, m+64, 128]` fp32 | 每 head 行 stride = `m+64`；尾部 `[m, m+64)` 行须为 0 | `m15_layer_kernel.h:643-648`（`PF_GDN_QK_PAD_ROWS = 64`）、`:721`（`gp.qkStride = A.m + PF_GDN_QK_PAD_ROWS`）；消费者读法 `m15_gdn_prefill.h:931-934`；判据 `m23_gdn_prefill/check_ref.py:130`（`Tp = T + 64`） |
| v | `[48, m, 128]` fp32 | 无 pad | `m15_layer_kernel.h:260`；`m15_gdn_prefill.h:933-934` |
| g, β | `[48, align8(m)]` fp32 | 行对齐 8（`tp = ((m+7)/8)*8`）；行内 `[m, tp)` 列由消费者在 UB 覆盖 | `m15_layer_kernel.h:261-262`、`:717`；`m15_gdn_prefill.h:935-940` |

q/k 的 pad 行**由本段产出时置零**（`MtGdnProlog::ZeroPadRows`）；g/β 的 `[m, tp)` 列也在段内清零
（g/β 行缓冲先整片清零再逐 token 落槽）。`docs/22:264` 逐字：「q/k 的 pad 行必须由产 q/k 的段（prolog）负责置零」。

### scale 口径（塔裁 2026-10-04）

`gdn_prolog_mt_kernel` 以 `qScaleOn` 形参区分两种口径（kernel 内一个条件）：

- **默认（`qScaleOn = 0`）= 相位 A 契约**：q/k 均**不乘** 1/√128，scale 由相位 A 消费者施加：
  `m15_gdn_prefill.h:967` `VecScaleConstVF(scSE, scEG, scale_)`、`:1051` `ConstScale1VF(aUb, ..., scale_, ...)`；
  `m15_layer_kernel.h:722` `gp.scale = 0.08838834764831845f; // 1/√128`；`m23_gdn_prefill.asc:67` 逐字
  「l2norm（纯单位化；1/√DK 的 scale 由 kernel 的 scale 参数施加）」。生产者若再乘一次即**双重 scale**。
- **`qScaleOn = 1`**：切回 m9 独立模块的历史语义（仅 q ×1/√128、k 不乘；`m9_gdn_prolog.asc:75`/`:225`），
  用于 **m=1 档与 m9 现有参考做交叉复现**（本次 `m9mt_m1_b56_mut0` 即此档），不作默认。

**这是"未接线项"**：本段输出 q 未乘 scale、由相位 A 施加；m9 作为独立 decode 模块的 `QSCALE` 语义
（`:75`/`:225`）与本链路口径不同，接线时**以相位 A 为准**（`qScaleOn = 0`）。本段**未接进
`m15_layer_loop`**（挂载点补丁见 `docs/22 §6`），不主张端到端。

### 链级推理：decode 路的 prolog 与 scan 各自怎么处理 scale

（塔要求，防同类口径陷阱。）

- **decode 路自洽于"prolog 乘、下游不乘"**：`m15_gdn_layer.h:29` 的 S3 = `Prolog::GdnProlog`（m9 的拷贝，
  `m15_gdn_layer.h:704`/`:714-715` 仅 q 乘 `QSCALE`），下游 S4 = m4 递推核把 q/k 当**不透明输入**
  （`m4_gdn_recurrent/README.md:47-48`：q「已 l2norm **且乘 1/√128**」、k「不乘」，`kernel 外预计算`）
  ⇒ decode 路**不**在下游再乘 scale。
- **prefill 路（相位 A）自洽于"生产者不乘、消费者乘"**：相位 A 自己乘 `gp.scale`（上节三处出处）。
- ⇒ 两条链路的 scale 归属**方向相反**。本段按 prefill 链（相位 A）落地（默认 `qScaleOn = 0`），与 decode 链
  m9 `QSCALE` 语义**不同向** —— 如实登记，供接线时对照。

### 跨 token 语义与 conv_state 交接

对 token t：窗口 `win = [st0, st1, st2, x[t]]`（`st` 为当前 conv_state 3 行，planar `[3][10240]`），
算完把 `st` 左移一格（`st0←st1`、`st1←st2`、`st2←x[t]`）。m 个 token 跑完，
`conv_state_out = [state; x[0..m-1]] 的最后 3 行`，与 `m21_layer_ref/ref/gdn.py:115-134`
（`x_new = cat([conv_state, x])`、`new_state = x_new[..., -3:]`、`out = conv1d(x_new)[..., -T:]`）逐句同义。
m≥3 时 `conv_state_out = [x[m-3], x[m-2], x[m-1]]`；m<3 时按同一左移与初始 state 拼接
（m=1 退化为 m9 的 `[s1, s2, x0]`，本次已交叉复现）。

### 段体设计（要点）

- 80 个 128ch block 条带（q 0-15 | k 16-31 | v 32-79），每个 AIV 对自己持有的每个 block 跑 m 个 token 的
  串行 token 循环；每 block 的 W/BIAS/state 只载一次，X 按 token 预取（parity 双缓冲）。
- **计算全 VF**（`__simd_vf__`：conv/l2norm/gating 全在寄存器内），**矩阵乘法为零**（本段无收缩）；
  scalar 只做下标/循环/地址。
- **同步全 BufferID**（`get_buf`/`rls_buf`，无 SetFlag/WaitFlag）：`MT_BUF_W`（W/BIAS/AL/DT：MTE2→VEC）、
  `MT_BUF_ST`（state 3 行：MTE2→VEC→MTE3）、`MT_BUF_T0/1`（每 token 的 X/AST/BST/OUT：MTE2→VEC→MTE3）、
  `MT_BUF_GB`（g/β 行缓冲：VEC→MTE3）、`MT_BUF_Z`（pad 置零源：VEC→MTE3）。
- g/β 行内落槽用 `Store`（vstu 非对齐单元素）：`StoreAlign<...DIST_FIRST_ELEMENT_B32>` 要求 32B 对齐，
  而行内位置 t 是 4B 步长；`docs/18-vector-api-audit.md:408` 指明单元素/非对齐落盘只在
  `Store`/`StoreUnAlign`/`Gather` 一族。pad 行清零用一次 32KB `DataCopy`（blockCount=1，规避 NZ 重排）。
- UB 全静态：persistent（W/BIAS/AL/DT/state）+ 每 token 区 ×2 parity + g/β 行缓冲（`MT_MAX_TP64`）
  + 32KB pad 零源，`static_assert(MT_END <= 248KB)`。

### 校验方法与结果

判据分类口径按 `docs/22-prefill-prolog-epilog-wiring.md:333-344`：**T1 逐位**（conv_state_out 的 bf16 位型、
q/k 尾部 pad 行、g/β 行内 pad 列）与 **T3**（q/k/v/g/β 有效区，`|got-exp| <= 1e-5·|exp| + 1e-6`）。
`check_ref.py` 用 numpy fp32 逐公式参考；host 侧在设备输出面上先埋非零（`POISON = 1234.5f`），
使「未写 / pad 未清」可判。设备档逐档各自 `flock -w 300`、进锁先 `npu-smi`、锁内 `timeout`
（逐档日志见 `evidence/logs/`）。

| 档 | m | blk | 说明 | state | qpad/kpad | q/k/v | g/β |
|---|---|---|---|---|---|---|---|
| mt0 | 1 | 56 | `qScaleOn=1`，与 m9 交叉复现 | PASS(位精确) | PASS | PASS | PASS |
| mt1 | 4 | 56 | 小档 | PASS | PASS | PASS | PASS |
| mt2 | 65 | 56 | 中档 | PASS | PASS | PASS | PASS |
| mt3 | 257 | 56 | 大档 | PASS | PASS | PASS | PASS |
| mt4 | 65 | 4 | 少 AIV：每 AIV 多 block × token 循环 | PASS | PASS | PASS | PASS |
| mt5 | 65 | 16 | 条带 | PASS | PASS | PASS | PASS |

`check_ref.py` 默认档 rc=0（6 个 m>1 档 + 原 m=1 五档）；逐字读数见 `evidence/logs/check_default.log`。
原 m=1 段体 5 档（56/28/16/8/4 AIV）在新二进制上仍 PASS（`evidence/logs/m1_all_run.log`）。

### 负向对照（判据能咬住的证据）

| 档 | 注入 | 期望 | 实测 |
|---|---|---|---|
| mt6（`mut=1`, m=65） | 不交接 conv_state（每 token 用初始 state） | 变红 | state FAIL + q/k/v FAIL（g/β 与 conv 无关，仍 PASS）；`check_ref.py --mutant 1` 打印「如预期变红」 |
| mt7（`mut=2`, m=257） | 不写 q/k pad 行 | 变红 | qpad FAIL + kpad FAIL（值仍 PASS）；`check_ref.py --mutant 2` 打印「如预期变红」 |

两个反向对照都走到「变红」，说明 T1（state/pad）与 T3（值）两族判据都不是空的。逐字读数见
`evidence/logs/check_mut1.log`、`check_mut2.log`。

### 本段已知限制 / 未完成

- **m 范围**：实测到 m=257（含 m=1/4/65 各档），**未实测** m>257；`MT_MAX_M = 4097` 是相位 A 验收档
  `GP_M_PREFILL`（`m15_gdn_resources.h:317`）的静态上界，未跑。UB 静态尺寸按 4097 预留。
- **未接进 `m15_layer_loop`**：挂载点/实参（`LayerArgs` 新字段）归 Wave C/D（`docs/22 §6`）；本段不主张端到端。
- **scale 边界未接线**：默认 `qScaleOn = 0`（相位 A）；接线时须保证相位 A 侧只乘一次（见上「链级推理」）。
- **z 平面 / in_proj（S2）不在本段**：本段只吃 in_proj 之后的 x（与 m9 一致），z 由别处持有（`docs/22` G1-c）。
- 每 block 内 token 串行、block 之间亦串行；未做跨 block 流水，正确性优先。
