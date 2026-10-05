# M13：MoE block layer kernel 骨架（单一 `__mix__(1,2)` 启动，段序 S1–S10）

图纸 = `docs/12-layer-integration.md`（段序 §1/§3、资源表 §4、同步图 §5、小改清单 §6）
+ `docs/05-megakernel-design.md` §2/§5/§6（约束与 quirk 清单）。算件代码全部**复制改造**
自 main 上的 m6 / m7 / m8 / m3 / m5（原目录未改动）。

## 1. 链结构与分段

| 段 | 算子 | 来源算件 | 核 | 输入 → 输出（GM） |
|---|---|---|---|---|
| S1 | Add+RMSNorm #1 | m6_rmsnorm | AIV 行切分 | 层输入 x + 零残差 → `x_norm`(bf16) + `res1`(**fp32**) |
| S2 | router：GEMV → softmax(max-shift) → top-k 降序（`Sort32` 例外 + `Extract` VF 内联，见 §8.2）→ renorm；同段顺带算共享专家门裸点积 | m7_router_topk | AIV0 | `x_slice` + router_w → `logits`/`topk_ids`/`topk_weights`/`sgate`(fp32 裸点积) |
| S3 | routing 索引生成（计数排序）**新增 glue** | —（m7↔m8 之间的缺口） | AIV0 | `topk_ids` → `perm_src_token`/`perm_expert`/`counts`/`offsets`/`inv_slot`/`w_tk_packed` |
| S4 | permute（逐行 gather，四级流水） | m8_permute #1 | AIV 行切分 | `x_slice` ⊕ `perm_src_token` → `x_sorted`(bf16，打包槽位) |
| S5 | A 侧 MXFP4 **全 VEC** 量化（routed + shared 两次调用） | m5_swiglu_quant 的 MxQuant 三段式 | AIV 行切分 | `x_sorted`/`x_slice` → `A_qx`+`A_scale`（80B 行距）/ `A_qx_shd`+`A_scale_shd` |
| S6 | grouped gate_up GEMM（5 槽位 = 4 routed + 1 shared，K=2560 N=1280 BASE_N=256） | m3_grouped_gemm | AIC | `A_qx`+权重+counts → `GU`/`GU_shd`(bf16) |
| S7 | SwiGLU + MXFP4 **全 VEC** 量化（routed + shared） | m5_swiglu_quant | AIV 行切分 | `GU`/`GU_shd` → `H_swiglu`(bf16 抽点) + `H_qx`+`H_scale`（32B 行距，20B 有效） |
| S8 | grouped down GEMM（5 槽位，K=640 N=2560） | m3_grouped_gemm | AIC | `H_qx`+`dn 权重` → `Y`/`Y_shd`(bf16) |
| S9a | unpermute 加权折叠（topK 模板 1..4，bf16 权重位打包协议保留） | m8_permute #2 | AIV 行切分 | `Y` ⊕ `inv_slot` ⊕ `w_tk_packed` → `routed_output` |
| S9b | combine：`shared = sigmoid(sgate)·Y_shd`；`moe = routed + shared` | 新增 glue（sigmoid 走向量） | AIV 行切分 | → `shared_output` / `moe_output` |
| S10 | Add+RMSNorm #2（**残差 fp32** = S1 的 `res1`，小改 D） | m6_rmsnorm | AIV 行切分 | `moe_output` + `res1` → `y_final`(bf16) + `res2`(fp32) |

数据面：全部中间张量落在一段连续 workspace（偏移见 `m13_resources.h` §6，
32B 对齐；`WS_BYTES ≈ 7.7MB`）。权重直接消费 HF 原始 layout（uint8 nibble 打包 MXFP4 +
e8m0 分组 scale），**不做离线转换**；唯一 host 侧组板是共享专家的 gate/up **行拼接**
（checkpoint 是独立 gate_proj/up_proj，拼接后与 routed 的 merged `gate_up_proj` 同形态，
权重字节不变）。

`x_slice` = S2/S4/S5 的 MoE 段输入，由 `sliceMode` 选择：
* `sliceMode=0`（MoE slice 模式）：直接消费 golden 数据集里的 `x`（数据集定义该张量为 MoE
  段输入、即 post-norm 激活）→ 全链**逐 op 对齐 golden**；
* `sliceMode=1`（layer chain 模式）：消费 S1 产出的 `x_norm` → 真实层链（router 消费 norm 输出）。

> 为什么不把 golden 的 x 送进 S1 再用 `x_norm` 喂 router：RMSNorm 对逐 token 缩放不变，
> 任何 per-channel gamma 都无法让 `norm(x)·γ == x`（除非 γ 逐 token），因此「golden 的 x
> 既当 norm 输入又当 MoE 输入」在数学上不成立。本实现保留两条路径：golden 逐 op 对齐
> （slice）与真实层链（chain），见 §5 校验方法。

## 2. 全局资源表（`m13_resources.h`，对应 docs/12 §4）

| 资源 | 落地 |
|---|---|
| AIC BufferID | 0-1 A(+scale) ping/pong、2-3 B、4-5 L0A/L0B、6 L0C、7-10 预留（跨段权重预取）、11-13 预留（FIXP→UB） |
| AIV BufferID | 0-7 通用行窗（各段 barrier 分隔、地址叠放）、15-18 permute 四级流水、19-23 预取/gamma 预转/rstd scratch、8-14 保留给 m4 |
| flagId | 0-3 AIV 侧段边界 mode 0、4-7 AIC 侧段边界 mode 0、8-11 段内 mode-2 流式对、12-15 保留（相邻同步点必不同 id；每 id 用量 ≪ 15） |
| UB | PERSIST @0 16KB；gamma fp32 预转 @16384（20KB）；SEG-VEC 窗 @36864 —— 窗内峰值 ~153KB（router x 块 80KB + 权重 fp32 51KB，UB_RT_END=193600）；各段同址叠放，占 248KB 的 ~78% |
| L1 | A 区 @0 ×2、B 区 @262144 ×2（与 m3 一致）；@330KB 起为预取滚动窗（本 mission 未实现跨 op 预取） |
| GM workspace | 32 个中间张量，偏移/尺寸全部编译期常量（`SZ_*`/`WS_*`），总量 ~7.7MB |

## 3. 同步表

段边界跨核走 docs/12 §5 的**标准 mix 序列**：

| 方向 | 序列 |
|---|---|
| AIV → AIC | AIV `set mode2(PIPE_MTE3, drain)` → AIC `wait mode2(PIPE_S)` → AIC `set/wait mode0(PIPE_MTE2/PIPE_S)`（全体 AIC 对齐） |
| AIC → AIV | AIC `set/wait mode0(PIPE_FIX/PIPE_S)`（FIXP 写 GM 排空）→ AIC `set mode2(PIPE_MTE2)` → AIV `wait mode2(窄 pipe)` |
| AIV ↔ AIV（段内交接） | 全体 AIV `set mode0(PIPE_MTE3)` + `wait mode0(窄 pipe)`（需要标量值依赖的段挂 `PIPE_S`） |

| flagId 槽 | 用途 | 相邻性 |
|---|---|---|
| 0-3（AIV ring） | S1 后、S3 后（`PIPE_S`：perm_src 标量读）、S4 后、S9a 后、S9b 后 | 逐一不同，第 5 次复用 0（前一对已 drain 完） |
| 4-7（AIC ring） | S6 前对齐、S6 后（FIX）、S8 前对齐、S8 后（FIX） | 逐一不同 |
| 8-11（mode2 ring） | A_routed 就绪、GU 就绪、H 就绪、Y 就绪 | 逐一不同 |

规则核对：set 一律挂生产 pipe（FIX/MTE3），**从不挂 PIPE_S/ALL**；wait 用最窄 pipe，
仅标量值依赖（地址计算）的段挂 `PIPE_S`；mode 0 只在同类型核之间；**相邻同步点 flagId 必不同**。

BufferID（`GetBufInternal`/`RlsBufInternal`）语义要点（本 kernel 实测踩到的两条）：
1. **同一 UB 区域若被不同 BufferID 保护，跨 pipe 交接就会失效**——router 的 logits 行
   一度由 GEMV 块（`ROW0`）写、`MTE3` 读（`ROW1`），m=33 时所有行读到最后一行的值；
   改成「写/读共用同一 token」后正确。
2. **同一 pipe 内 `drain` 释放后再 `acquire` 同一 id 会自锁**（释放要等 pipe 排空，而
   acquire 就在该 pipe 队里）——因此每行的 V 段只在**段尾**统一释放，下一行改由另一条
   pipe（MTE3）acquire，形成「V↔MTE3 交替」的乒乓。
3. 行级 round-robin 复用同一 UB 行缓冲的循环一律补 `PipeBarrier<PIPE_MTE2>`（小改 B）。

## 4. 小改清单落实（docs/12 §6）

| 项 | 落实 |
|---|---|
| **A** 元素计数 DataCopy → 显式参数 | 22 处元素计数重载（docs/12 §6 A 点名的 m6/m5/m8/m3 点位 + m7 + 新增 combine）**逐点**改为显式 `DataCopyParams{1,len,0,0}`；`Block1(bytes)` helper 内部按 dtype 换算（**blockLen 单位 = 32B 块**：bf16 行 `Block1(HIDDEN*2)`={1,160,0,0}、fp32 resOut `Block1(HIDDEN*4)`={1,320,0,0}、`Block1(WROW_I32*4)`={1,2,0,0}）；小/非对齐传输一律 `DataCopyPad` + `DataCopyExtParams{1,bytes,0,0,0}`（`ExtBlock1`，含 4B 的 sgate 单值和 8B 的 scale 行）。AIC 侧 Nd2Nz/Dn2Nz 沿用自身显式参数结构 |
| **B** 行循环 MTE2 复用补 PipeBarrier | `NormStage`/`RouterStage`/`PermuteStage`/`VecQuantStage`/`UnpermuteStage`/`CombineStage` 的行/tile 循环入口全部 `PipeBarrier<PIPE_MTE2>` |
| **C** 量化器换 m5 全 VEC 路径 | S5/S7 用 m5 的 `SwigluComputeTile` + `MxQuantComputeScale` + `MxQuantComputeDataFP4`；**全链无 `PIPE_S` 挂载**（`grep -c "PIPE_S"` 只剩跨核 wait 与 BufferID 的标量握手） |
| **D** m6 残差输入 fp32 | `NormStage<RES_F32>` 模板：S10 消费 S1 的 fp32 `res1`（实测 `res2` 与参考**逐位一致**，`res1` 同样） |
| **E** 索引读 / w_tk_packed | `w_tk_packed`（bf16 位打包进 int32 低 16 位）协议保留；perm_src/counts/inv 仍为 m8 协议的 GM 索引（标量读），**未**改成 MTE2 批量 → 见 §6 限制 |
| **F** m>64 / 尾块 | 宿主显式校验 `m ≤ 64`、`topk ≤ 4`、`t_e ≤ 64`（BASE_M 单 tile）；prefill m-tile 路径未实现 |
| **G** m9↔m4 契约 | 本 mission 不含 GDN，未涉及 |

## 5. 校验方法与结果（Ascend950PR / CANN 9.1.0 / 28 AIC + 56 AIV）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -B m13_moe_layer/build -S m13_moe_layer -DCMAKE_BUILD_TYPE=Release
cmake --build m13_moe_layer/build -j2
./m13_moe_layer/build/m13_moe_layer tools/golden/data m1 m33     # 两 case × 两模式
./m13_moe_layer/build/m13_moe_layer quant-inf                    # 量化段自检（含 ±Inf/NaN，无需数据集；见 5.4）

# 落盘全部中间张量后，用 numpy 独立交叉校验（118 条判定项 + 32 条参考项）
mkdir -p /tmp/m13_dump && cd /tmp/m13_dump
M13_DUMP=1 <repo>/m13_moe_layer/build/m13_moe_layer <repo>/tools/golden/data m1 m33
/usr/local/python3.12.13/bin/python3 <repo>/m13_moe_layer/check_ref.py . m1 m33
```

`M13_STAGE_LIMIT` / `M13_SUB_LIMIT` 可截断段序（bring-up 定位用；S2/S3 段都有运行时门控）。

### 5.1 kernel 内判据（`evidence/accept_run_m1_m33.log`）

682 条判据 + 末尾 banner，四个 case×mode 组合：

1. **对 golden 的逐 op 严格判据**（slice 模式，直接消费 `tools/golden/data/{m1,m33}` 的输入与
   golden bin —— 这些 bin 即 `moe_block_ref.py` 的 `router_logits/topk_ids/topk_weights/
   perm_*/counts/x_sorted/routed_output/shared_output/moe_output`）：
   `topk_ids` / `perm_src_token` / `perm_expert` / `expert_token_counts` **精确相等**；
   `topk_weights` bf16 网格 ≤1 ulp；`router_logits` 2e-2（实测 maxAbs 2.4e-7 / 4.8e-7）；
   `x_sorted` **逐位一致**（m1: 5120 元素、m33: 168960 元素，0 失配）。
2. **w4a4 参考链严格判据**（两模式）：把 **device 自己产出的** MXFP4 激活
   （`A_qx/A_scale`、`H_qx/H_scale`）反量化后用 double 做 GEMM/SwiGLU/门控，逐段与 device
   比对（`GU`/`H`/`Y` 每槽位每行、`routed`/`shared` 每行、`combine` 逐元素）：全部 PASS，
   最差占容差 0.28（= 残差只含输出 bf16 舍入）；`combine` 段
   `shared = bf16(fp32(sigmoid·Y_shd))`、`moe = bf16(fp32(routed + gated_shared))`
   与参考的 bf16 位距 ≤1 ulp。
   另加**内部一致性**（两种模式都成立）：counts 与 perm_expert 一致、perm 按
   (expert, token) 分组有序、`inv_slot` 与 perm/topk_ids 自洽、`w_tk_packed` 低 16 位 =
   bf16(topk_weights)、`x_sorted` 与 `perm_src_token` 的 gather 逐位一致。
3. **与 float golden 的量化偏差预算**：`|dev − golden|` 必须被 `|w4a4 ref − golden|` 解释
   （逐元素 `ad ≤ 1.5·ar + 0.02·|gold| + 5e-3`）→ 三个输出全部 PASS，违反 0 元素。
   偏差本身作为报告打印（绝对量：m1 mode0 = 0.6045/0.6282/0.7627，m33 mode0 =
   0.9609/0.8594/1.1875，m33 mode1 = 3.3789/1.8848/3.7734，对应 routed/shared/moe）。
4. `res1/res2` **fp32 逐位一致**、`x_norm`/`y_final` bf16 网格 max rel 0.0077。

### 5.2 numpy 独立交叉校验（`check_ref.py`，`evidence/check_ref_run.log`）

`check_ref.py` 用 `tools/golden/moe_block_ref.py` 的编解码/路由语义**独立**复算，消费
`M13_DUMP=1` 的中间张量：**118 条判定项全 PASS，另有 32 条参考项（非判定，见 5.3）始终打印**
（每个 case×mode 各 8 条；参考项不计入 PASS/FAIL）。它补上 host C++ 判据覆盖不到的三件事：

* **量化器取值质量**（review P2-5 的盲区）：用 numpy 从 device 自己的 `x_sorted`/`h_swiglu`
  按 kernel 采用的 MXFP4 规范（见 5.3）独立重量化，与 device 的
  `A_qx/A_scale/A_shd_qx/A_shd_scale/H_qx/H_scale/H_shd_qx/H_shd_scale` **逐字节**比对
  → 两个 case × 两个模式 **0 字节不符**。
  —— host 的「偏差预算」判据对量化器整体变粗不敏感，本项是独立见证。
* **double 参考链**（numpy 独立实现）：GU/H/Y/routed/shared/moe 分段比对（同 5.1 第 2 项口径）。
* **golden 逐 op**（mode 0）+ **以 device 的 x_norm 为段输入的复算**（mode 1）：router
  logits/ids/weights、perm 数组、x_sorted 逐位。

另有一个子命令 `check_ref.py quant-inf [dump 目录] [--quantizer PATH]`（M50 新增，见 §5.4）：对
`M13_QS_DUMP=1 quant-inf` 落盘的 ±Inf/NaN/次正规用例 dump 做**逐字节**判据（**40 条判定项 +
2 条报告项**），并可用 `--quantizer` 把被判实现换成 base commit 的旧参考做**负向对照**（旧参考必须 FAIL）。
退出码为**三态**（tower 2026-09-26 规则）：`0` 比过且通过 / `1` 比过且有差异 / `2` 没得比或输入缺失
（用例不齐、或数据集模式缺 dump ⇒ 打印 `SKIPPED`/`PARTIAL` 并返回 2，**不发合格证**；
负向对照读数见 `evidence/m50_check_ref_tristate.log`）。

### 5.3 量化规范差异（已量化、已披露、可复核）

kernel 的**激活**量化用 m5 `MxQuant` 路径（docs/12 §6 小改 C 指定）的 e8m0 规则：
`byte = amax 的 bf16 指数域 − 2`（只用指数域 → floor 语义），data 侧 `CAST_ROUND`
（远离零、饱和到 ±6）。而 golden 的**权重**打包规范（`moe_block_ref.pack_mxfp4`）用
`scale = 2^ceil(log2(amax/6))`（ceil 语义，等价于 m3 标量量化器里 `+ (M > 0x400000)`
的尾数修正项）。两者在 amax 落在 `(6·2^k, 8·2^k]` 段时给出不同 scale（floor 规则会把
`amax/scale` 推到 6 以上并饱和）。

`check_ref.py` 把该差异作为**参考项始终打印**（`evidence/check_ref_run.log` 可逐行复核），
每个 case×mode 8 条、共 32 条；下表是 device 字节 vs「golden 权重规范」量化器的**逐字节差异占比**
（判定项用的是「device 字节 vs 硬件规范」→ 0 字节不符）：

| case×mode | A_qx(routed) | A_scale(routed) | A_shd_qx(共享) | A_shd_scale(共享) | H_qx(routed) | H_scale(routed) | H_shd_qx(共享) | H_shd_scale(共享) |
|---|---|---|---|---|---|---|---|---|
| m1 mode0 | 42.9% | 42.5% | 42.9% | 42.5% | 37.0% | 47.5% | 32.2% | 40.0% |
| m1 mode1 | 32.0% | 32.5% | 32.0% | 32.5% | 33.9% | 42.5% | 32.8% | 40.0% |
| m33 mode0 | 29.0% | 28.5% | 29.0% | 28.5% | 32.6% | 41.4% | 31.0% | 39.5% |
| m33 mode1 | 35.1% | 35.5% | 35.1% | 35.5% | 31.6% | 41.1% | 28.3% | 36.1% |

（注意：m=1 的差异占比**最大**，因为 2 行数据的 group 数少、amax 更容易落在规范分歧区间；
差异占比本身不是质量问题，两套规范的 p99 精度同级，只是 scale 取整方向不同。）

该差异不影响本 mission 的任何判定：
* 权重走 golden 规范（kernel 直接消费 checkpoint 字节，不重新量化）；
* 激活量化没有 golden（`moe_block_ref` 全程 float32，不对激活量化），故激活侧规范由
  docs/12 §6 小改 C 指定 → 本实现跟随 m5（= 硬件 `npu_dynamic_mx_quant` 语义）；
* 若将来要求与某种「量化感知 golden」逐位对齐，需先确定激活侧取哪一套规范（见 §6.6）。

### 5.4 量化段自检：含 ±Inf/NaN/次正规 的用例（M32；用例 ④ 与参考侧修复见 M50）

S5/S7 的激活量化（`VecQuantStage::MxQuantComputeScale` / `MxQuantComputeDataFP4`，quantA/quantAShd K=2560 无 SwiGLU、quantH/quantHShd K=640 带 SwiGLU）此前缺少官方
`add_rms_norm_dynamic_mx_quant_common.h:413` 的 **halfScale 非有限覆盖**：组内 `maxexp == 0x7F80`（±Inf/NaN）时官方把 `halfScale` 覆盖成 bf16 NaN `0x7F81`，使 `Mul(±Inf, NaN) = NaN → Cast<fp4> = 0.0`；缺这句时 `Mul(±Inf, 2^-126) = ±Inf → Cast` 饱和成 **±6**（E8M0 scale 的 `0xFF` 覆盖 `:406` 本来就已实现，故只有 data 侧分叉）。M32 已补上该句（注释标注官方 path:line）。

官方逐句对照表（15 步）见 `m2_mxfp4_quant/README.md`「与官方实现的逐句对照」——本核的量化段是 m2/m5 的逐字移植，15 步中 14 步位级等价，**唯一差异就是 10a**；
`:411`/`:415` 的 `shared==0x7F00` 特判（halfScale→`0x0040`）在本核**不可达**：`sharedExp ∈ {k*0x80, k ≤ 253}`，而 `0x7F00 = 254*0x80` 需要 `vdMaxExp == 0x8000`，指数掩码 `And 0x7F80` 的上界是 `0x7F80`。

> 引用口径：本节的 `:NNN` 一律指**官方冻结头文件** `add_rms_norm_dynamic_mx_quant_common.h` 的行号
> （上游产物、与 m2/m5/M32 的文档同口径）；引用本仓自己的代码时按 tower 写作规则用**符号**，行号只作
> 查阅提示并注明"随代码变动"。

自检入口（**不需要数据集**，复用同一份 `VecQuantStage` 设备代码，仅把行索引换成紧凑布局）：

```bash
cd <空目录>
<m13 build>/m13_moe_layer quant-inf            # 输出 40 条判据 + 总结行（M32 时为 32 条）
M13_QS_DUMP=1 <m13 build>/m13_moe_layer quant-inf   # 额外落盘 qs_{A,H}_i{0..4}_r{1,64}_s{0,1}_*.bin
```

| infKind | 用例（任务 ①②③ 出自 M32，④ 出自 M50） | 注入内容 |
|---|---|---|
| 1 | 整组 amax=+Inf | A 侧：注入组 32 个元素全 `+Inf`；H 侧：`gate=+Inf, up=1.0` → SwiGLU 输出整组 `+Inf` |
| 2 | 组内只有部分元素 Inf | A 侧：组内 `+Inf`/`-Inf` 各一；H 侧：`(+Inf,1.0)→+Inf` 与 `(4.0,-Inf)→-Inf` 各一 |
| 3 | NaN 组 | A 侧：`gate[j] = (j&1) ? +NaN : -NaN`（**实际落成整组 `+NaN`**，见下注）；H 侧：`gate=±NaN`（`up` 一处 NaN） |
| 4（M50） | 次正规 / 退化组 | A 侧：注入组 = bf16 **正**次正规 `0x0001..0x000F`（含 `+0`）+ 最小正规 `0x0080`；H 侧：`gate=2^-115, up=2^-14` → SwiGLU 输出 = 正次正规 `2^-130`（`0x0008`） |

> infKind 3 的行内注：A 侧的 `up` 与 `gate` 同址（`H_GenQuantSrc` 的 `rowLen` 约定），注入循环里
> `up[j] = QS_NAN_P` 会把 `gate` 的 ±NaN 交替改成整组 `+NaN`（M50 归档 dump 实测：`qs_A_i3_r1_s0_x.bin`
> 的注入组整组 `0x7FC0`）。行为保持原样未动（改它会把「负 NaN 的符号位」这个既有角落拉进判据，见 §6.10）；
> 因此 **`±NaN` 的符号位在 A 侧未被覆盖**，这是本自检的已知覆盖缺口，不是回归。

用例矩阵：`side ∈ {A(K=2560,SS=80,SwiGLU=false), H(K=640,SS=32,SwiGLU=true)} × rows ∈ {1,64} × seed ∈ {0,1} × infKind ∈ {0..4}` = **40 条判据**，每条的判据是：

* device `qx`（每行 K/2 字节）与 device `scale`（每行前 K/32 字节）对 host C 参考（官方语义，含 `:413`）**逐字节一致**；
* 行内 `K/32` 之后的 scale 字节**未被写**（预置 0 → 说明 scale 行距 `SS=32 > 20` 的 padding 不被踩）；
* H 侧另比 SwiGLU：device vs host 参考 bf16 网格 ≤1 ULP，且「有限/非有限归类」逐元素一致；
* 注入组的 device nibble **全 0**（修复前为 ±6 饱和码；kind 4 的机制不同：组 amax 指数域 `< 0x0100` ⇒ `:402-403` 夹到 `emax`、`:414` 令 `halfScale = 0` ⇒ 组内乘积恒 `±0`）。

| 运行 | 基线（inf=0） | inf=1（整组 +Inf） | inf=2（部分 Inf） | inf=3（NaN） | inf=4（次正规，M50） | 总计 |
|---|---|---|---|---|---|---|
| M32 修复前（删掉 `Select(..., nanRegTensor, cmpResult)` 后重编） | 8/8 PASS | 0/8 | 0/8 | 8/8 | 用例不存在 | **16/32 PASS** |
| M32 修复后（= M50 之前） | 8/8 | 8/8 | 8/8 | 8/8 | 用例不存在 | **32/32 PASS** |
| M50（新增 kind 4） | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | **40/40 PASS** |

**零回归**：数据集全链判据（`m1`/`m33` × mode 0/1，**682 条判据行**）在 M32 修复前后、以及 M50 的
「Extract 改造 + 参考修复」前后都给出**逐字节相同的日志**（`evidence/run_dataset_prefix_nofix.log`、
`run_dataset_after.log`、M50 复跑见 `evidence/m50_zero_regression.log`），即这些改动对不含非有限/次正规
输入的激活不产生任何位级变化。
判据行口径（避免以后混）：**判据行 = 结尾带 `PASS (` 的判定行 678 条 + `routing 内部一致性 …… PASS` 4 条 = 682**；**不含**末尾的 `[M13] ===== ALL PASS =====` banner。故 `grep -c PASS` 得 683（打印行数），按 `docs/17` §2.2 不作判据数。

**参考侧（numpy）同名判据（M50 新增）**：`check_ref.py quant-inf <dump 目录>` 直接拿 §8.1 修复后的
`quant_mxfp4_hw` 对上面同一批 device dump 做逐字节判据：**40 条判定项全 PASS**（10 个用例 × 4 项），
另有 **2 条报告项**（用例集合完整性 guard、`tools/golden` 交叉见证；按 `docs/17` §2.1 单列，
不计入 PASS/FAIL 计数）；把被判实现换成 base commit 的旧参考（`--quantizer`）同一命令
**27 PASS / 13 FAIL** —— 这就是 M50 要求的负向对照（13 条 FAIL 全落在 40 条判定项内，见 §8.1）。
日志：`evidence/m50_quant_ref_inf_after.log` 与 `evidence/m50_quant_ref_inf_prefix_nofix.log`。

**独立见证（另一个 agent 的实现）**：自检落盘的 32 个 device dump 与 `tools/golden/moe_block_ref.py` 的 `quantize_ocp()`（M26 写的官方 ops-nn 序列逐句 numpy 转写，含 `:413` 与非有限角落）**逐字节一致**：A 侧 16/16、H 侧 16/16（日志 `evidence/ocp_independent_witness.log`；该参考函数出自 M26 分支 `6f68f8e`，合入 main 后即可复跑）。M50 把该见证扩到全部 10 个用例（含 kind 4）：`quantize_ocp` vs device **0/8500 字节不符**（`evidence/m50_quant_ref_inf_after.log` 末行）。

**本自检证明什么 / 不证明什么（`docs/17` §3.5 边界声明）**：
- **证明**：`VecQuantStage` 的量化段（`MxQuantComputeScale` / `MxQuantComputeDataFP4`）在 **K=2560（无 SwiGLU）与 K=640（带 SwiGLU）两种模板实例化**、**SS=80 / SS=32 两种 scale 行距**下，对含 ±Inf/NaN 在内的激活与 host 参考（官方语义）逐字节一致；且 scale 行内 padding 区不被写。
- **不证明**：自检以 `VecQuantStage<K, SS, SWIGLU, false, false>` 实例化（`PAD_SRC = PAD_DST = false`），只覆盖**紧凑行布局**。正式 routed 链路的 `PAD_SRC/PAD_DST = true` **行映射**（按专家 slot 的 `slot*M_MAX+lr` 定位、`countsGm` 专家前缀和驱动的行数）**不在本自检范围内** —— 该部分由数据集全链判据（§5.1/§5.2，682 条判据行）覆盖。因此「量化段数值正确」由本自检证明，「量化段被路由到正确行」由数据集全链证明，二者不可互相替代。

**自检生成器的约束（避开既有 -0 角落）**：除「零组」外，各组 amax 指数域 ≥ 3（保证 `halfScale ≠ 0`，不做 `x·0` 的精确零乘积）；零组只含 `+0`（A 侧）或输出恒 `+0`（H 侧）；H 侧另保证 `|silu(gate)·up| ≥ 2^-22`（乘积不下溢到 0）。原因：`halfScale == 0` 的组里 `x·0` 的符号位被硬件保留（`Cast(-0) → nibble 0x8`），而 host 参考的字面规则 `xs < 0` 给 `+0`——这是与本修复无关的既有角落，真实激活不可达（见 §6）。

### 5.5 证据归档（`m13_moe_layer/evidence/`）

| 文件 | 内容 |
|---|---|
| `accept_run_m1_m33.log` | kernel 内 682 条判据全量日志（4 个 case×mode 组合，rc=0；含 4 行 `M13_DUMP` 落盘提示，说明日志与 `dump_manifest.md` 出自同一次运行） |
| `check_ref_run.log` | `check_ref.py` 的日志（118 条判定项 ALL PASS + 32 条参考项占比） |
| `dump_manifest.md` | **140** 个 `.bin` dump 张量的 sha256（全 64 位）+ 字节数，以及**再生成命令**（同目录另有 4 个 `*_meta.txt`，共 144 个文件） |
| `run_dataset_after.log` / `run_dataset_prefix_nofix.log` | M32：量化段自检改动后 / 修复前（删除 `:413` 那一句）的数据集全链日志，二者逐字节相同（零回归证据） |
| `quant_inf_after.log` / `quant_inf_prefix_nofix.log` | M32：`quant-inf` 子命令自检日志（修复后 32/32 PASS；修复前 16/32，inf=1/2 全 FAIL） |
| `quant_inf_dump_run.log` | 带 `M13_QS_DUMP=1` 的自检运行日志（对应下列 dump） |
| `quant_inf_dumps/` | 自检 dump（`qs_{A,H}_i{0..4}_r1_s0_*.bin`：`x`/`swiglu_device`/`qx_device`/`scale_device`/`qx_ref`/`scale_ref`），可离线复算逐字节比对 |
| `m50_quant_ref_inf_after.log` | M50：`check_ref.py quant-inf`（修复后的 numpy 参考）对上述 device dump 的逐字节判据 —— **40 判定项 / 40 PASS**，另有 2 条报告项（含 `tools/golden` 交叉见证 0/8500 字节不符） |
| `m50_quant_ref_inf_prefix_nofix.log` | M50：**负向对照**——同一判据换成 base commit 的旧参考（`--quantizer`，sha256 记在文件头）—— **40 判定项 / 27 PASS / 13 FAIL** |
| `m50_quant_inf_after.log` | M50：`quant-inf` 设备自检（含 kind 4 次正规/退化组）—— **40/40 PASS** |
| `m50_zero_regression.log` | M50：零回归证据（数据集 682 条判据日志逐字节相同、144 个 dump sha256 全同、140 张量与 `dump_manifest.md` 0 不符、`check_ref` 118 条日志逐字节相同、quant-inf 32 → 40） |
| `m50_dump_sha256.txt` | M50：复跑落下的 144 个 dump 文件（含 `*_meta.txt`）的 sha256 |
| `m50_dataset_after.log` / `m50_check_ref_after.log` | M50：复跑的**原始**数据集全链日志（682 条判据行）与 `check_ref.py` 数据集日志（118 判定 + 32 参考项）——与 M32 归档的 `run_dataset_after.log` / `check_ref_run.log` **逐字节相同**（`diff` 空；见 `m50_zero_regression.log` §6） |
| `m50_check_ref_tristate.log` | M50：审校脚本**三态退出码**的负向对照（tower 2026-09-26 规则）——(a) 正常 quant-inf ⇒ 40/40 + rc=0；(b) 空输入 quant-inf ⇒ `SKIPPED` + rc=2（报告项转 WARN，不发合格证）；(c) 空输入数据集模式 ⇒ `SKIPPED` + rc=2；(d) 正常数据集模式 ⇒ `ALL PASS` + rc=0（输出与归档逐字节相同） |

未把 dump 原始字节（~31MB/次）入库：kernel 已实测**确定性**（同二进制两次运行逐字节一致），
故用 sha256 清单 + 再生成命令替代，既保证可复核又不给仓库塞二进制（需要原始字节时按
`dump_manifest.md` 的命令在空目录复现并按 sha256 逐字节核对）。

## 6. 已知限制 / 后续

1. **AIV 段间串行、无跨段预取**：段边界一律 barrier 分隔（骨架正确性优先）；docs/12 §4
   里 AIC BufferID 7-10 的第二组 GEMM ping/pong、L1 @330KB 预取滚动窗、AIV 19-23 的
   预取 staging 都只**留位未用**——跨 op 权重预取与 A/B 双缓冲重叠是下一步的主要收益点。
2. **router/索引生成为单核（AIV0）**：m≤64 时 router 是带宽受限 GEMV，m7 实测单核
   m=33 也可接受；索引生成是同核标量计数排序（m*topk ≤ 256）。后续可按专家槽位切多 AIV。
3. **小改 E 只做了一半**：`w_tk_packed` 协议保留，但 perm_src/counts/inv_slot 仍是
   m8 协议的标量 GM 读（m8 README 的「改 UB 批量索引」未做）；`Σt_e` 也仍是标量前缀和。
4. **t_e = 0 的槽位**：GEMM item 直接跳过（不产生输出）；`A/H` 对应槽位的 padding 区不写。
5. **prefill 未覆盖**：只验收到 m=33（奇数列尾块）；m>64 需 m-tile 分段（每段重复段序同步），
   docs/12 §8 已列为遗留。
6. **量化器跨 donor 规范不一致（待用户裁决）**：激活量化用 m5 VEC 路径的 floor 指数 e8m0 规则，
   而 m3 原标量量化器（被小改 C 替换掉的那支）与 golden 的权重规范都是 ceil 规则。差异已量化
   （`check_ref.py` 参考项）且不影响本 mission 的任何判据，但若后续要与量化感知 golden 逐位
   对齐，需要先定激活侧规范。
7. **host 侧无 msprof 计时**：性能数字未测（README 的定位是正确性 + 同步纪律）。
8. **数据集缺口**：golden 不含 norm 权重（gamma），故 S1/S10 的 gamma 由 host 确定性合成
   （∈[0.5,1.5]）；若要端到端对齐真实模型，需要往 `tools/golden/data` 里加 gamma。
9. **性能未优化**：本 milestone 只保证正确性与同步纪律（kernel 时间未单独测量）。
10. **量化段非有限语义（M32 已修 + 残留角落）**：`:413` 的 halfScale 非有限覆盖已补（见 5.4），
    含 ±Inf 的激活组现在与官方一致（整组 nibble 归零 + scale `0xFF`）。仍存在的**既有角落**
    （与本修复无关、真实激活不可达，自检生成器已避开）：
    * `halfScale == 0` 的组（amax 指数域 ≤ `0x0100` 且组内含负元素）里 `x·0` 的符号位被硬件保留
      （`Cast(-0) → nibble 0x8`），而 host 参考的字面规则 `xs < 0` 给 `+0`；
    * NaN 输入的「零的符号位」：设备 `Cast<float→bf16>(NaN)` 给 `0x7FFF`（正值、丢符号），
      x86 host 保留 NaN 符号，故 `-NaN` 输入在两条链上可能差一位（`0x0` vs `0x8`）；
      量级语义（`Cast(NaN) → 0.0`）两边一致。
    两者都只在非有限/下溢输入的「符号位」上体现，若要与某种量化感知 golden 逐位对齐需先定规范（见 §6.6）。
11. **参考侧与 device 的 parity（M50 已修，遗留两条覆盖缺口）**：`check_ref.py::quant_mxfp4_hw`
    已按 device 路径重写（§8.1），非有限组与退化组逐字节对齐（负向对照：旧参考 13 FAIL）。
    仍未覆盖的两处（都不在 M50 判据范围内，属 §6.10 的既有角落）：
    * **A 侧 `±NaN` 符号位**：自检 kind 3 的 A 侧注入因 `up`/`gate` 同址而整组落成 `+NaN`，
      故「负 NaN ⇒ nibble 0x8」这条（m2/m5 的 numpy 参考有建模）在 m13 侧**未被 device 用例覆盖**；
    * **退化组的负元素**：`halfScale = 0` 时负元素的 `x·0 = -0` 是否保留符号位（device `Cast(-0) = 0x8`），
      kind 4 用例刻意只用非负值以避开 host C 参考 `xs < 0` 的字面规则；numpy 参考用 `signbit(product)`
      与 device 行为一致（与 m2/m5 同源），但 m13 的 device dump 不区分该角落。
    两者都需要一条**新的 device 用例**才能把差异钉死；若要补，建议同时统一 m13 的 host C 参考
    `H_E2M1Code`（把 `xs < 0` 换成「NaN 单独判、零看符号位」）并把 m2/m5 的建模一起对齐。

## 7. 文件

| 文件 | 说明 |
|---|---|
| `m13_resources.h` | 全局静态资源表（BufferID / flagId / UB / L1 / GM workspace 偏移 + 形状常量） |
| `m13_moe_layer.asc` | 单一 `__mix__(1,2)` kernel（段序 S1-S10 + 同步）+ host（数据集装载、w4a4 参考链、判据、dump）+ `quant-inf` 量化段自检入口（5.4，`__vector__` 自检 kernel 复用同一份 `VecQuantStage`；M50 起含 kind 4 次正规/退化组用例） |
| `check_ref.py` | numpy 独立交叉校验（消费 `M13_DUMP=1` 的 dump；量化器逐字节 + double 参考链 + golden 逐 op），118 条判定项 + 32 条参考项；另有 `quant-inf` 子命令（M50：非有限/次正规用例逐字节 **40 判定项 + 2 报告项** + `--quantizer` 负向对照，见 §5.4/§8.1） |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`，`-ffp-contract=off` 保证 host 参考与 kernel 同 IEEE 序列） |
| `evidence/` | 验收证据归档（682 条 kernel 判据日志、118 条 check_ref 判定项 + 32 条参考项日志、140 个 dump 张量的 sha256 清单 + 再生成命令、M32 量化段自检日志与 dump、M50 参考修复/Extract 改造的日志与零回归证据） |

## 8. M50：±Inf 参考修复与 Extract 改造的位级影响面

### 8.1 ±Inf/NaN/次正规 参考修复：**参考错、device 对**

**缺陷**：`check_ref.py::quant_mxfp4_hw` 是 m13 的 numpy 量化参考（也是 M40 的
`m15_layer_loop/check_moe_ref.py` 直接 `import` 的那份）。M32 给 device 的 `MxQuantComputeScale`
补了官方 `:413` 的 halfScale 非有限覆盖，**参考侧没跟着改**，于是非有限组上两侧分叉；同一函数另有
一处更早的分叉：**退化组**（组 amax 的 bf16 指数域 `< 0x0100`）。

| 用例（都拿 device 字节为准） | M50 前的参考 | device | 性质 |
|---|---|---|---|
| 组内含 `+Inf`/NaN：scale 字节 | `0xFD`（253 = 指数域 − 2） | **`0xFF`**（E8M0 NaN，官方 `:406`） | 参考漏了非有限覆盖 |
| 组内含 `+Inf`（i1）：data 码 | `0x7`（= +6.0 饱和） | **`0x0`**（`Mul(±Inf, NaN) = NaN → Cast<fp4> = 0`） | 同上 |
| 组内含 NaN（i3）：data 码 | `0x6`（= 3.0） | **`0x0`** | 同上 |
| 退化/次正规组（A 侧 `0x0080` = 2⁻¹²⁶，i4）：data 码 | `0x4`（= 2.0） | **`0x0`**（`:402-403` 夹 `maxexp`、`:414` halfScale = 0） | 参考漏了夹下界 |

**为什么"device 对"**：device 的三处行为都能逐条对上官方
`add_rms_norm_dynamic_mx_quant_common.h`（`:406` / `:413` / `:402-414`），并且
`tools/golden/moe_block_ref.quantize_ocp`（M26 的逐句转写，**独立实现**）与 device 在全部 10 个用例上
**0/8500 字节不符**（§5.4 末）。参考侧则既不对官方序列、也不对 device。

**修法**（代码：`check_ref.py` 的 `quant_mxfp4_hw`，M50 重写；查阅提示 `:101-146`，**行号随代码变动、以符号为准**）：
抄仓内同源 donor（m2/m5 的 `check_ref.py::ocp_numpy`），把
`quant_mxfp4_hw` 重写为对 device 的逐句建模：位域 `maxexp` → 非有限分支（字节 `0xFF` + halfScale
`0x7F81`）→ 退化分支（halfScale 0）→ bf16 域乘 + RNE → CAST_ROUND 取档（平局远离零、饱和 ±6）+
`signbit` 定符号、NaN 结果码 0。
**判据 T1（逐字节）**：`check_ref.py quant-inf` 对 10 个 device dump 用例 **40 条判定项全 PASS**
（10 × 4：qx / scale 合法区 / scale padding / 结构性），另有 **2 条报告项**单列
（用例集合完整性 guard、`tools/golden` 交叉见证 0/8500；按 `docs/17` §2.1 不计入 PASS/FAIL）。

**它过去可能造成的假 FAIL 面**：任何按 M32 结论补「含 ±Inf/NaN 激活」用例的后续 mission，只要复用这条
参考链，就会把**正确的 kernel 判 FAIL**。已知直接使用者是 M40 的
`m15_layer_loop/check_moe_ref.py`（`import check_ref as M13R`，mission M40 task3 明确要求复用 m13 的
参考链）；`tools/golden/**` **不**受影响（§8.4 审计）。既有数据集（`m1`/`m33`，不含非有限/次正规激活）
一条判据都没变（§8.3）。

**负向对照（读数的可复现性 + 顺序说明）**：FAIL 读数由 **base commit 的原始参考文件**产生
（`--quantizer` 指向 `git show <base>:m13_moe_layer/check_ref.py` 的副本，sha256 记在日志头），
**不是**本次改动里的开关：

| 读数 | 实现 | 结果（判定项；报告项单列） | 日志 |
|---|---|---|---|
| 修复前（动手前同日实测；当时 dump 只有 i0..i3） | base commit 的 `quant_mxfp4_hw` | A/H 两侧 inf=1/2/3 的 qx+scale 全 FAIL | —（与下行同一实现；本轮起可复跑复现） |
| 负向对照（i0..i4，归档） | 同上（base commit 原始文件） | **40 判定项：27 PASS / 13 FAIL**（+2 报告项） | `evidence/m50_quant_ref_inf_prefix_nofix.log` |
| 修复后（归档） | 本 commit 的 `quant_mxfp4_hw` | **40 判定项：40 PASS / 0 FAIL**（+2 报告项） | `evidence/m50_quant_ref_inf_after.log` |

### 8.2 Extract 改造的位级影响面

**改动**：`m13_moe_layer.asc` 的 `RouterStage::SoftmaxTopkRow`（S2 行内 top-k）里，经典
`Extract(ovT, oiT, pairT, 1)` →
**VF 内联**：`LoadAlign<float, LoadDist::DIST_DINTLV_B32>` 一次把 32 个 (value, idx) 交错对拆成两个
寄存器，再两条 32-lane 掩码 `StoreAlign` 分别写 `UB_RT_OV` / `UB_RT_OI`。抄厂商 dav_3510
`ExtractVf`（`asc/impl/basic_api/dav_3510/kernel_operator_vec_gather_mask_impl.h` 的 float 分支；
`repeatTime = 1` 即其 loopTimes=0/tail=1 路径）与官方 topk donor 同形
（`ops-transformer/moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_perf_arch35.h`
的 `ExtractKFP32Perf`；引用时给出符号名，行号仅作查阅提示）。VF 块首按 m7 的同款写法放
`LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()`
（隔开经典 `Sort32` 的 UB 写与本 VF 的 UB 读）；取掩码用 `UpdateMask<float>(32)`（前 32 lane），
对 `DIST_DINTLV_B32` 的两种可能宽度解释都成立（64 或 128 float 载入时前 32 lane 都是本行的 value/idx）。

**性质**：纯 pair 拆分、**零算术**（无 cast / 无舍入 / 无饱和），故与经典 `Extract` 逐位等价。查 `Extract`
不入 docs/05 §6.1 的例外（厂商 3510 的经典 `Extract` 本身就是 `__simd_vf__`），已改造完毕 —— 改动落在
`RouterStage::SoftmaxTopkRow`（`m13_moe_layer.asc`）：`Sort32` 调用点与其上方例外注释、以及紧随的
`__VEC_SCOPE__` 拆分块（查阅提示：`:971-997`，**行号随代码变动、以符号为准**）；
`Sort32` 保留但按 §6.1 ⓒ 在调用点加了一行依据注释（无 `Reg::` 等价物 + 官方 donor 同样 memory-based +
官方位置 `perf_arch35.h` 的 `Sort32`/`MrgSort`）；`MrgSort4` 在本文件 **0 处**（`grep -n 'MrgSort'
m13_moe_layer.asc` 的 3 处命中全在注释里），符合"3510 上是 deprecated 空函数体、不得使用"。

**位级证据（T1，本 mission 实测）**：
* 数据集全链日志与改动前**逐字节相同**（682 条判据行，`diff` 空）；
* `M13_DUMP=1` 的 **144 个文件 sha256 全部相同**（含 140 个张量）；
* 140 个张量与归档的 `dump_manifest.md` **0 不符**；
* `check_ref.py` 的 118 条判定项日志与改动前**逐字节相同**（其中 `topk_ids`/`topk_weights`/`perm_*`
  是精确/位级判据，正是 Extract 的直接下游）。
参见 `evidence/m50_zero_regression.log`。

### 8.3 判据数：改前 / 改后（M50）

| 判据块 | 改前 | 改后 | 证据 |
|---|---|---|---|
| kernel 数据集全链（`m1 m33` × mode 0/1） | **682 PASS** | **682 PASS**（日志逐字节相同） | `m50_zero_regression.log` §1 |
| kernel `quant-inf` 量化段自检 | **32 PASS** | **40 PASS**（+8 条 kind 4） | 同上 §5 |
| `check_ref.py` 数据集（118 判定 + 32 参考项） | 118 PASS / 0 FAIL | 118 PASS / 0 FAIL（日志逐字节相同） | 同上 §4 |
| `check_ref.py quant-inf`（M50 新增） | 用例不存在 | **40 判定项：40 PASS / 0 FAIL**（+2 报告项）；负向对照（旧参考）**13 FAIL**（同样 40 判定项内） | §8.1 |
| `M13_DUMP=1` dump 张量 | 144 文件 | 144 文件，sha256 全同 | 同上 §2/§3 |

零回归结论：**M50 的两处改动对既有 682 + 32 + 118 条判据不产生任何位级变化**；新增判据只来自
量化段自检新增的非有限/次正规用例与参考侧同名判据。

### 8.4 `tools/golden/**` 同源量化器审计（M50 任务 ③：只报不改）

查了哪些文件、依据是什么（下表引用**以符号为准**，行号仅作查阅提示）：

| 文件 | 同源量化器 | 结论 |
|---|---|---|
| `tools/golden/moe_block_ref.py` | `quantize_ocp()`（:286-332，官方 ops-nn 15 步逐句转写） | **无缺陷**：非有限组 → `byte 0xFF`（`FP8_NAN`）+ `HALF_SCALE_NAN 0x7F81` + `np.isnan(h) → code 0`；退化组 → `half_bits = where(shared == 0, 0, …)` 后走同一乘路径（乘积恒 ±0 ⇒ 码 0）。逐句对上官方 `:402-406/:412-414/:753-758` |
| 同上（`e8m0_bytes_floor()` :220-231） | 仅**有限** amax 的字节助手（官方 :405-406 的有限分支） | 不构成独立量化器；其唯一调用方 `selfcheck.py:166-176` 只做**规则分派一致性**检查，不用于判 device |
| `tools/golden/selfcheck.py` | 角落自检（`check 4`/`check 3` 段：`all-zero/bf16-denormal` 与 `±Inf/NaN` 两条 `check(...)`） | **已有**对应的两条角落判据：`all-zero/bf16-denormal 组 ⇒ byte 0 + 码全 0`、`±Inf/NaN 组 ⇒ 0xFF + 码全 0`（自检通过） |
| `tools/golden/qaware_ref.py` / `gen_dataset.py` | 经 `quantize_activations(rule)`（:350-367）委托到 `quantize_ocp` | 无第二份实现 |

**交叉见证（运行证据，非"读代码"推断）**：把 `quantize_ocp` 作用在**设备自己的量化输入**上，与 10 个
device dump（±Inf / NaN / 次正规）逐字节比对 → **0/8500 字节不符**
（`evidence/m50_quant_ref_inf_after.log` 末行；该判据也是 `check_ref.py quant-inf` 的一条判定项）。

⇒ **`tools/golden/**` 无 M40 finding 所指的缺陷，因此按任务 ③ 未开 finding、未改任何文件**
（该目录归 M46 评审，本 mission 只读）。
另：`m2_mxfp4_quant/check_ref.py` / `m5_swiglu_quant/check_ref.py` 的 `ocp_numpy` 也已正确建模两处
角落（`is_inf → 0xFF / 0x7F81`、`shared == 0 → 0`）——它们正是 §8.1 的**抄改来源**，本身无需修。
（M40 finding 提到的"m2/m5 的参考侧是否同型"由此结清：**同型缺陷只存在于 m13**。）

### 8.5 m13 与 `m15_layer_loop` 副本的关系（go-forward 归属）

* **m13 是 MoE 段（S1-S10）的 go-forward 源**：本 mission 的全部判据都在 m13 上跑、在 m13 上归档。
* m15 的 per-layer kernel **带一份 m13 的搬运副本**：`m15_layer_loop/m15_moe_layer.h`（由
  `m15_layer_loop/lift_moe_segment.py` 从 m13 lift 出来），同一处在
  `RouterStage::SoftmaxTopkRow` 里 —— `Sort32` / `Extract` 两个调用点相邻（查阅提示：`:946`/`:947`，
  **行号随代码变动、以符号为准**），资源常量 `UB_RT_PAIR` 在 `m15_moe_resources.h`。
  该目录归 **M40（agent-moefuse）**，按 mission 边界**本 mission 未改动 m15 一行**；M40 需要
  重新 lift 才能带上：① §8.2 的 VF Extract；② `Sort32` 例外依据注释；③（若要）量化段自检的 kind 4 用例。
* **参考侧不需要同步**：`m15_layer_loop/check_moe_ref.py` 直接 `import` m13 的 `check_ref.py`
  （`import check_ref as M13R`），§8.1 的修复对它**自动生效**（这正是 M40 finding 里担心的那条链）。
  另注意 m15 自己还有一份 `check_moe_ref.py`/`check_ref.py` 的宿主逻辑，但量化器不复制。
* 因此：M50 只对 m13 负责；m15 侧同一处（Extract 改造 + 判据口径）由 M40 处理。
