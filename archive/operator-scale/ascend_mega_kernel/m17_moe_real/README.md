# M17：MoE block **真实 MXFP4 权重**端到端（M29）

把 m13 的 MoE block 整层 kernel（段序 S1–S10，单一 `__mix__(1,2)` 启动）从
`tools/golden` 的**合成数据集**（E=4 / topk≤4 / 合成 x）换成**真实 checkpoint**：
`/workspace/Qwen3.8-Flash-Next-MXFP4` 的某一层 MoE block 全部权重（512 个专家的
packed u8 + e8m0 scale、router/gate、共享专家）字节原样装入 device GM，层输入取真实
embedding 行，两处 RMSNorm gamma 取真实 `hc_norm` 权重行切片 —— **整条链无任何合成数据**。

- 权重契约（与 m13 完全一致）：`mxfp4-pack-quantized-e8m0`，专家权重 `u8` nibble 按
  **lo-hi**（byte 低半 = K 偶数位）打包、K 方向 group = 32、scale 为 e8m0（`2^(b-127)`）；
- 所有 host 侧参考量都是 bf16 / u8 **字节切片**，**不做任何离线转换**；唯一 host 组板是
  共享专家 `gate_proj`/`up_proj` 的**行拼接**（checkpoint 是独立两张，kernel 侧要 merged
  `gate_up_proj` 同形态，权重字节不变）。

| 项 | 值 |
|---|---|
| 层形状 | hidden 2560 / moe_intermediate 640 / num_experts **512** / num_experts_per_tok **10** |
| 每层 MoE 权重量 | packed `gate_up [512,1280,1280]` 800 MiB + scale 50 MiB + `down [512,2560,320]` 400 MiB + scale 25 MiB ≈ **1.24 GiB** |
| m 包络 | m ≤ 64（decode，单 BASE_M tile）；本 mission 验收 m = 1 / 8 / 33 / 64 |
| 结果 | host 判据 **13 PASS / 0 FAIL（不变量 12 条 + cube 解包探针 1 条）**（4 个 m + 换层换 token 各一次）；numpy/double 独立参考链 **74 / 251 / 545 / 716 条判定项全 PASS**（m=1/8/33/64；其中 **M60 的量化规则见证 11 条/档**，见 §5.5），换层复核 74/74 PASS（另有 1 条非空洞性 guard，按 `docs/17` §2.1 单列为报告项、不计入判定项） |
| 逐字节证据 | host pread 1.24 GiB 与 Python 独立 sha256 全部一致；device GM D2H 回读全量 `memcmp` 0 差异；nibble 解包探针 16384/16384 与独立解包一致 |

## 1. 与 m13 的关系与逐条改动

`m17_moe_layer.asc` 是 `m13_moe_layer.asc` 的**真实形状版本**：段序、同步纪律、
S1/S10（m6 RMSNorm）、S4/S9（m8 permute/unpermute）、S5/S7（m5 SwiGLU + MxQuant 全 VEC）、
S6/S8（m3 `MXFP4GemmItem`）**逐字沿用**；只有与 E/TOPK 强相关的部分被改写。

| # | m13（E=4 / topk≤4） | m17（E=512 / topk=10） | 原因 |
|---|---|---|---|
| 1 | `NUM_EXPERTS=4`、`TOPK_MAX=4` | `512` / `10` | config.json `text_config` 真实值 |
| 2 | router 权重 `[E+1][HIDDEN]` **全量预转 fp32 常驻 UB**（51KB） | **流式**：x 行块 + 权重行 ping/pong + 8 专家一组预转窗（m7 结构） | E=512 时全量常驻需 5.25MB ≫ UB 248KB |
| 3 | top-k 用**单块 Sort32**（只覆盖 E≤32） | **Sort32（16 个 32 块）+ 4 级 2 路 `MrgSort` 归并树** → 根节点 = **全局 top-64** → Extract 64 → 取前 10（每级 `elementLengths=[32,32,0,0]` 只出 64 对，**不是** 512 对全序） | E=512 超出单块容量；取 top-10 只需各级 top-32 的归纳（`top-10 ⊆ top-32(各输入) ⊆ 本级输出 64`）。**API 选择**：走 `MrgSort`（不用 `AscendC::MrgSort4` —— 后者在 dav-3510 上是**静默 no-op**，见 `probe_sync_quirks` A02/A15/A20 与 `docs/05` §6 #19；当年"4 路丢 src3/src4 index"的真因是本方调错 API，**不是**硬件限制，同参数的 4 路 `MrgSort` 真机完全正确） |
| 4 | 索引生成的 `cnt/off/cursor` 是**栈数组**（E×3×4B） | 搬进 UB 静态槽位；offsets 槽从 `cnt[8]` 挪到 `cnt[E]` | E=512 时栈数组 6KB 压垮 AIV 标量栈；`cnt[8]` 与会 counts 重叠 |
| 5 | `UnpermuteStage` k 链展开 1..3 | 展开 1..9 + `UnpermuteStage<10>` 实例 | TOPK=10 |
| 6 | unpermute 的 Y 视图声明 `M_MAX*TOPK_MAX*HIDDEN` | `NUM_EXPERTS*M_MAX*HIDDEN` | m13 里两者相等是 E=4 的巧合（4×64），E=512 时不等 |
| 7 | 诊断槽写 `offs[8]`（覆盖真 offsets[8]） | 写 `offs[E+8]`；UB 侧标量写移进 `MutexLock<PIPE_S>` 区之前 | 修 m13 的别名 + 实测 m=8 时诊断槽与 MTE3 抢跑读到 UB 残值 |
| 8 | `RT_RB=16` 行块 | `RT_RB=8` | RB=16 时 router 段 UB 峰值超 248KB |
| 9 | 无 | 新增 `w_tk_packed` 尾部清零（M×16 槽） | 尾部内核不写，MTE3 却整段拷出 → 落盘 sha256 逐次不同 |
| 10 | 无 ±Inf 平价修复 | **补上** main 已落地的 ±Inf/NaN 平价修复（官方 `add_rms_norm_dynamic_mx_quant_common.h` 的 `NAN_CUSTOMIZATION = 0x7f81` + `Select` 覆盖 `halfScale`；行号随上游变动，以符号为准） | 本分支从合并 `7ee8444` **之前**分叉，m13/m5 的激活量化器当时还没有这条；不补会让 m17 的拷贝静默缺该行为（组内出现 ±Inf 时应量化为 0 而不是饱和成 ±6） |

未改动的共享项：BufferID 编号与语义、flagId 旋转槽位、UB/L1 偏移约定、GM workspace
32 段平面图（`m17_resources.h` §4/§5/§6）、`-ffp-contract=off`、`__mix__(1,2)` 启动。

**逐文件差异表（`docs/17` §3 第 4 条）** —— 本目录每个文件与上游的关系：

| 本目录文件 | 上游 | 差异性质 |
|---|---|---|
| `m17_moe_layer.asc` | `m13_moe_layer.asc`（并按段抄 m6/m7/m8/m5/m3 的算件） | 段序/同步/算件逐字沿用；差异 = §1 的 10 条（E/TOPK 放大、router 流式重写、topk 归并树、索引生成 UB 化、k 链展开、诊断槽、`RT_RB`、`w_tk` 清零、±Inf 平价）+ host 侧全部重写（真实 checkpoint pread/三腿核对/dump/判据/探针） |
| `m17_resources.h` | `m13_resources.h` | 差异 = §1 的 1/2/8 条（形状常量、router UB 布局、UB 尺寸）；§4/§5/§6 的偏移约定与 m13 同构 |
| `check_ref.py` | 无直接上游（随 M29 分支 `581eb62` 入库、`d88c684` 起独立编写） | 只用 `tools/golden/moe_block_ref.py` 的 6 个 codec 符号（路由语义内联）；口径见 §5.3 |
| `gen_weight_manifest.py` | `m15_layer_loop/slice_layer_manifest.py`（同为"从分片头取 offset/bytes"的做法） | 张量清单/角色命名按 MoE block 重写；新增 Python 独立 sha256 与行切片 |
| `CMakeLists.txt` / `.gitignore` | `m13_moe_layer/` 同名文件 | 工程名/注释改写，构建选项一致 |

> **上游变更不会自动流入（`docs/17` §3 第 4 条）**：上表的"逐字沿用"指的是**本 commit 的
> 拷贝**，不是"跟随上游"。§1 第 10 行就是一次实例：m13/m5 在 main 上补了 ±Inf 平价修复，
> m17 的拷贝不会自动获得，本轮已手工同步；**今后上游对这些算件的任何修改都需要从 main
> 重新同步并重跑本目录的验收**（`check_ref.py` 的量化器字节判据会抓住量化器类的不一致，
> 但 sync 语义/搬运算件的不一致需要靠重跑 device 全链发现）。
>
> **UB 保留槽**：`UB_ZEROS_B16`（m13 的 bf16 全 0 残差行）与 `UB_CB_OUT` 在本 mission
> 未被使用（残差走 host 的 `res_zero` GM 缓冲；combine 原地改写 `UB_CB_ROUTED/SHARED`），
> 保留是为了不破坏 m13 的 UB 平移/同址叠放关系 —— 已在 `m17_resources.h` 就地注明。
> **`perm_expert`** 在 kernel 链内**没有消费者**（链上只需要 `perm_src`/`counts`/`inv_slot`/
> `w_tk_packed`），它的 producer 存在是为了 host 不变量判据与 `check_ref.py` 的路由分组核对。

## 2. checkpoint 张量 → kernel 槽位的布局转换表

manifest（`m17_weight_manifest.txt`，由 `gen_weight_manifest.py` 生成）给出每个张量在
分片文件里的**绝对字节偏移 + 字节数 + sha256**；host 用 `pread` 原样取出。所有目标地址
都是**字节原样**，没有 requantize、没有 permute、没有 padding 填充。

| role（manifest） | checkpoint 张量（layer L） | 形状 → dtype | ∈ 分片 | device GM 指针 | kernel 槽位/消费点 | 转换 |
|---|---|---|---|---|---|---|
| `router_w` | `layers.L.mlp.gate.weight` | `[512,2560]` BF16 | 3 | `rwDev` | S2 router：B 侧 `[E, HIDDEN]` 逐行流式读 | 无（bf16 原样） |
| `sgate_w` | `layers.L.mlp.shared_expert_gate.weight` | `[1,2560]` BF16 | 3 | `sgDev` | S2：与 router 同段的一次点积（`[HIDDEN]` 行） | 无 |
| `gu_packed` | `layers.L.mlp.experts.gate_up_proj` | `[512,1280,1280]` U8 | 2 | `wguDev` | S6 grouped gate_up GEMM 的 **B 侧**（K=2560/N=1280），按 slot 取 `e*1280*1280` | 无（nibble lohi，group 32） |
| `gu_scale` | 同名 `.weight_scale` | `[512,1280,80]` U8 | 2 | `sguDev` | S6 B 侧 scale，行距 80B（K/32） | 无 |
| `dn_packed` | `layers.L.mlp.experts.down_proj` | `[512,2560,320]` U8 | 3 | `wdnDev` | S8 grouped down GEMM 的 B 侧（K=640/N=2560），按 slot 取 `e*2560*320` | 无 |
| `dn_scale` | 同名 `.weight_scale` | `[512,2560,20]` U8 | 3 | `sdnDev` | S8 **B 侧** scale：行距 20B（`SCALE_K = K/GROUP`），5 个 kBlock **读满 20B**，无越界；`DN_SCALE_STRIDE=32` 是 A 侧激活 `hs` 的槽宽 | 无 |
| `shd_gate` + `shd_up` | `shared_expert.gate_proj.weight` / `up_proj.weight` | `[640,1280]` U8 ×2 | 3 | `wgushdDev` | S6 第 E+1 号槽（N=1280）：`[0,640)` = gate、`[640,1280)` = up | **行拼接**（host，字节不变） |
| `shd_gate_scale` + `shd_up_scale` | 对应 `.weight_scale` | `[640,80]` U8 ×2 | 3 | `sgushdDev` | 同上 scale | 行拼接 |
| `shd_down` | `shared_expert.down_proj.weight` | `[2560,320]` U8 | 3 | `wdnshdDev` | S8 共享槽（K=640/N=2560） | 无 |
| `shd_down_scale` | 对应 `.weight_scale` | `[2560,20]` U8 | 3 | `sdnshdDev` | S8 共享槽 scale | 无 |
| `x_embed` | `embed_tokens.weight` | `[248320,2560]` BF16 | 130 | `xDev` | S1 层输入：第 `token..token+m-1` 行，其余行 0 padding 到 `M_MAX=64` | **行切片**（只有 m 行被读） |
| `gamma1` | `layers.L.mlp_hyper_connection.hc_norm.weight` 行 `[0,2560)` | `[10240]` BF16 | 3 | `g1Dev` | S1 RMSNorm gamma | 行切片 |
| `gamma2` | 同上，行 `[2560,5120)` | — | 3 | `g2Dev` | S10 RMSNorm gamma | 行切片 |

### GM workspace 布局（32 段，`m17_resources.h` §6，总 315 637 568 B ≈ 301 MB）

每个专家槽位都是 `[slot][M_MAX=64][row]` 的 padding 布局（E=512 时槽位表占主导，这是
m13 契约的直接放大，不是本 mission 引入的）：

| 段 | 形状 | 生产 → 消费 |
|---|---|---|
| `x_norm` / `res1` | `[64,2560]` bf16 / fp32 | S1 → S2/S4/S5、S10 |
| `logits` | `[64,512]` fp32（max-shift） | S2 → 判据 |
| `topk_ids` / `topk_weights` | `[64,10]` i32 / fp32 | S2 → S3/S9 |
| `perm_src` / `perm_expert` / `counts` / `offsets` / `inv_slot` / `w_tk_packed` | `[640]`/`[640]`/`[512]`/`[513]+诊断`/`[64,10]`/`[64,16]` | S3 → S4/S9 + AIC 的槽位数 |
| `sgate` | `[64]` fp32（共享门裸点积） | S2 → S9b |
| `x_sorted` | `[640,2560]` bf16 | S4 → S5 |
| `aq`/`as`、`aq_shd`/`as_shd` | `[512,64,1280]`+`[512,64,80]` U8；`[64,1280]`+`[64,80]` U8 | S5 → S6（A 侧） |
| `gu`/`gu_shd` | `[512,64,1280]` / `[64,1280]` bf16 | S6 → S7 |
| `h_swiglu`/`h_swiglu_shd` | `[640,640]` / `[64,640]` bf16 | S7（抽点）→ S8 |
| `hq`/`hs`、`hq_shd`/`hs_shd` | `[512,64,320]`+`[512,64,32]`；`[64,320]`+`[64,32]` U8 | S7 → S8（A 侧） |
| `y`/`y_shd` | `[512,64,2560]` / `[64,2560]` bf16 | S8 → S9 |
| `routed`/`shared`/`moe`/`y_final` | `[64,2560]` bf16 | S9/S10 → 出口 |
| `res2` | `[64,2560]` fp32 | S10 残差出口 |

`w_tk_packed` 协议：**一个权重一个 int32 槽**，把 `bf16(topk_weights)` 的位模式放在该 int32 的
**低 16 位**（生产 `wtkUb[t*16 + k]`、消费 `LoadAlign<int32_t, DIST_BRC_B32>` + `ShiftLefts 16`、
参考 `wtk & 0xFFFF` 三者一致）。行槽 16 个 int32 = 64B，只用前 `TOPK=10` 个，其余 6 个为 0
（内核显式清零，见 §1 第 9 行）。

`dn_scale` / `hs` 两个 32 的区别（勿混）：checkpoint 的 `down_proj.weight_scale` 是
`[512,2560,20]`，kernel 的 **B 侧**按 `SCALE_K = K/GROUP = 640/32 = 20B` 行距读、5 个 kBlock
**读满 20B**，无越界；`DN_SCALE_STRIDE = 32` 是 **A 侧激活 `hs`** 的槽宽（每行 32B，只有前
`INTER/GROUP = 20B` 有效，由量化器按 tile 写出）。

## 3. 真实输入与 gamma 的来源（需要读者留意的一处抉择）

* **x**：`embed_tokens.weight` 的第 `token..token+m-1` 行（真实 checkpoint 字节）。
  它不是"真实推理时的 MoE 输入"（那需要先跑 attention 才能得到 post-attention-norm 激活），
  但它是**真实模型张量里的真实向量**，量级/分布合理，且完全可复现（token 由命令行给定）。
* **gamma1/gamma2**：`mlp_hyper_connection.hc_norm.weight` 的行 `[0,2560)` / `[2560,5120)`。
  本模型的 MoE 输入归一化由 hyper-connection mixer（4 流 + lowrank）承担，没有独立的
  `input_layernorm`；本 mission 不建模 mixer，故取 mixer 自己的 norm 权重（`hc_count=4` ×
  2560 的第 0/1 流）作为两处 RMSNorm 的 gamma —— 是**真实字节**，但"用第 0/1 流"是本
  mission 的等价替身（见 §7 限制 1）。
* 除此之外，链上**没有任何合成量**：m13 里由 host 合成 gamma 的做法（其 README 限制 8）在
  本 mission 已消除。

## 4. 权重字节同一性的三条腿（任务书「逐字节证明与源切片一致」）

| 腿 | 做法 | 结果 |
|---|---|---|
| (a) Python 独立 | `gen_weight_manifest.py` 直接从分片文件的 `data_offsets` 区间分块读并算 sha256（**不经过** C++ 的 manifest 解析/pread 代码） | 写进 manifest 的 `sha256=` 字段 |
| (b) host pread | host 按 manifest 的 offset/bytes `pread` 出 1.24 GiB，各自复算 sha256，与 (a) 逐条比对 | `[M17][src] … == manifest` ×14，合计 **1 342 182 400 B 全一致** |
| (c) device GM | 每条上传后整段 `aclrtMemcpy` D2H（838MB 的专家权重按 64MB 分块），与 host 缓冲 `memcmp` 并流式复算 sha256 | `[M17][bytes] … D2H==HOST OK (diff 0)` ×14（每次运行），sha256 与 (a) 相同 |

### nibble 解包自检（证明 lo-hi 打包顺序 + e2m1/e8m0 语义）

`m17_weight_probe_kernel`（纯 cube `__global__ __cube__`，复用 `MXFP4GemmItem`）：
令 **A = 单位阵**的 fp4 编码（e2m1 的 1.0 = `0b0010`，lo-hi 打包；scale 全 `0x7f`=2⁰），
B = 真实 checkpoint 的 `gate_up_proj` 第 0 号专家 packed/scale 指针 —— 则
`C[i][n] = Σ_k A[i][k]·dequant(W[n][k]) = dequant(W[n][i])`（除 k=i 外全是"0×值"的精确 0，
累加为 fp32），即 device 用**硬件自己的 e2m1/e8m0 语义**把权重解出来。host 逐元素与
`e2m1[nibble] × 2^(scale_byte−127)` 的独立解包比对：

```
[M17]   解包探针 C[i][n] == host 独立解包（0/16384 不符，0 跳过） PASS
```

（±0 视为相等：`e2m1` 码字 8 = −0 会被硬件归成 +0；数值语义一致。落盘证据
`probe_c_bf16.bin` + `probe_spec.txt` 在 dump 目录，不进仓库。）

## 5. 判据与结果

### 5.1 device 内 / host 侧判据（每次运行 **12 条不变量 + 1 条 cube 解包探针**）

12 条不变量：`IG 越界诊断槽=0`、`counts 求和 = m·topk`、`offsets = counts 前缀和`、
`t_e ∈ [0,64]`、`topk_ids ∈ [0,E) 且行内唯一`、`perm 按 (expert,token) 分组有序`、
`每个 (token,expert) 都在 perm 里`、`inv_slot 与 perm 自洽`、`x_sorted = gather(x_norm, perm_src) 逐位`、
`w_tk_packed 低 16 位 = bf16(权重)`、`logits/权重/输出全体有限`、`topk_weights 行和 ≈ 1`。

第 13 条是 §4 的 **cube nibble 解包探针**（不属于不变量，`--no-probe` 时不出现），
host 的汇总行会显式分栏：

```
[M17] ================ 结果：host 判据 13 PASS / 0 FAIL（其中不变量 12 条 + cube 解包探针 1 条）================
[M17]   不变量 12 PASS / 0 FAIL；解包探针 1 PASS / 0 FAIL（--no-probe 时无探针）
```

**m=1/8/33/64 与换层换 token（layer 8/token 2000）：全部 PASS / 0 FAIL。**
边界护栏：`--m > M_MAX(64)` 与 `--m < 1` 一律**拒绝启动**（rc=2）并给出原因，不静默截断
（`--m` 直接决定各 `SZ_*` 段的分配，越界会走出分配）；日志里有一条 `--m=65` 被拒的实例。

### 5.2 numpy/double 独立参考链（`check_ref.py`）

从**同一份真实 checkpoint**（`tools/weights/safetensors_reader.py`，独立于 kernel host 的
manifest/pread 路径）出发，按真实 MoE block 语义复算 S1–S10，与 device 落盘分段抽点比对。

> **口径声明（必读）**：本表的比对是 **逐 op 隔离** 口径 —— **每一段的输入取自 device
> 上一段落盘的 bf16/fp32 中间量**（S2 取 device 的 `x_norm`、S6 取 device 的 `A_qx/A_scale`
> 字节、S7 取 device 的 `bf16 GU`、S9a 取 device 的 `bf16 Y` 与 `w_tk_packed`、S9b 取 device
> 的 `Y_shd`/`sgate`、S10 取 device 的 `res1`/`moe`），段内算术则是**独立的 double 复算**。
> 这样每段的残差只含**本段自身**的误差，段输入的正确性由**上游段的判据**见证
> （S1 从真实 x 起步；e2e 链从真实 x 起步）。代价是**整链没有在单次复算里走过** ——
> `docs/17` L0 的这条要求由 **§5.4 的不打断端到端链** 补上（判定项）。

| 段/项 | 判据 | 预算口径（T3 推导，见 §5.3） | m=1 实测（m=64 同量级） |
|---|---|---|---|
| S1 `x_norm` | RMSNorm#1 vs double | `1×spacing + EPS_FAST·\|ref\|` | max 7.8e-3，占预算 **0.330** |
| S1 `res1` | = 层输入 fp32 | **逐位** (T1) | 0 失配 |
| S2 `logits` | `x_norm·Wᵀ` max-shift，fp32 GEMV vs double | `EPS_GEMV·Σ\|a·w\|`（nulp=0：fp32 输出） | max 1.0e-6，占预算 **0.012** |
| S2 `topk_ids` | 降序 / 并列取小 id | **精确** (T1) | 0 失配（512 专家 top-10） |
| S2 `topk_weights` | renorm | `1×spacing + EPS_FAST·\|ref\|` | max 1.9e-8，占预算 **0.000** |
| S2 `sgate` | 共享门裸点积 | `EPS_GEMV·Σ\|a·w\|` | max 1.5e-8，占预算 **0.001** |
| S3 `perm_src/perm_expert/counts/inv_slot/w_tk_packed` | 计数排序 + 协议 | **精确** (T1) | 全 0 失配 |
| S4 `x_sorted` | gather 逐位 | **逐位** (T1) | 0 失配 |
| S5 `A_qx/A_scale`（routed + 共享） | vs numpy 独立量化器（硬件 floor-指数 e8m0） | **逐字节** (T1) | 0 字节不符 |
| S6 `GU`（每个非空槽） | device 的 A 字节反量化 × 真实权重 double GEMM | `1×spacing + eps_cube(2560)·Σ\|a·w\|` | max 3.8e-3，占预算 **0.499** |
| S7 `H_swiglu` | Silu·up（输入取 device bf16 GU） | `1×spacing + EPS_FAST·\|ref\|` | max 3.3e-3，占预算 **0.331** |
| S7 `H_qx/H_scale` | vs 独立量化器 | **逐字节** (T1) | 0 字节不符 |
| S8 `Y` | device 的 H 字节反量化 × 真实 `down_proj` | `1×spacing + eps_cube(640)·Σ\|a·w\|` | max 4.9e-4，占预算 **0.922** |
| S6/S7/S8 共享专家（`GU_shd/H_shd/H_qx_shd/Y_shd`） | 同 routed | 同左 | max 3.7e-3 / 1.9e-3 / 0 字节 / 5.0e-4；占预算 **0.494 / 0.322 / 逐字节 / 0.924** |
| S9a `routed_output` | Σ_k bf16(w_tk)·Y[inv] | `1×spacing + EPS_FOLD·Σ\|w·Y\|` | max 2.0e-4，占预算 **0.999（m=1/8/33）·1.000（m=64，零余量，见 §7.9）** |
| S9b `shared_output` | sigmoid(sgate)·Y_shd | `1×spacing + EPS_FAST·\|ref\|` | max 4.0e-4，占预算 **0.307** |
| S9b `moe_output` | `bf16(routed + 未取整的 gated shared)`（见下注） | `1×spacing + 2·2⁻²⁴·Σ\|加数\| + EPS_FAST·\|gated\|` | max 2.3e-4，占预算 **0.941** |
| S10 `y_final` | RMSNorm#2 over **fp32 res2** | `1×spacing + EPS_FAST·\|ref\|` | max 7.6e-3，占预算 **0.324** |
| S10 `res2` | `fp32(res1 + bf16(moe))` | **fp32 逐位** (T1) | 0 失配 |
| §5.4 e2e `moe_output` | 不打断链（输入=真实 x）vs device | `\|A−B\| + (1×spacing + EPS_FAST·\|ref\|)` | max **0（逐位）**，占预算 **0.000** |
| §5.4 e2e `y_final` | 同上 | 同上 | max **0（逐位）**，占预算 **0.000** |
| §5.4 e2e 量化字节 | 链上量化 `A_qx/A_scale/H_qx/H_scale` vs device 字节 | **逐字节** (T1) | 0/17000 字节不符 |
| §5.5 `W0` | dump 身份：`ws_*.bin` 整段 sha256 vs `evidence/dump_manifest.md` | **整段 sha**（T4 结构性） | 5/5 相符（见 §5.5） |
| §5.5 `W1` | 规则方向定点：`amax∈{0.5,1,2}` ⇒ scale 字节 124/125/126、code 6、反量化回 amax | **定点读数**（设备无关） | 两实现均 `byte=(124,125,126) code=(6,6,6)` |
| §5.5 `W3` | 角落平价：`quant_hw == tools/golden::quantize_ocp`（±Inf/NaN / 次正规 / 全零） | **逐字节**（设备无关） | 3/3 角落逐字节一致 |
| §5.5 `W4` | device 量化字节 vs `tools/golden::quantize_ocp`（A/H routed + 共享） | **逐字节** (T1，S1 输入) | 0/18700（m=1）… 0/1196800（m=64） |
| §5.5 `W5` | 设备侧方向定点：每个活动组的 `|amax|` 元素码 ∈ {6,7} | **整数计数**（S1 + 规则不敏感） | 违反 0/1100（m=1）… 0/70400（m=64） |
| §5.5 `W6` | 值域往返：`|dequant(device H) − h_ref| ≤ 2·scale + (1×spacing + EPS_FAST·\|h_ref\|)` | **推导界**（S1 + 规则不敏感） | 越界 0/7040（m=1）；最差占预算 **0.951–0.955** |
| §5.5 `自检-W*`（W2/W3n/W4n/W5n/W6n，共 5 条） | 负向对照：方向反的规则 / base 版镜像**必须** FAIL | 计数（不 FAIL 即判 FAIL） | 全 PASS（读数见 §5.5 与本表下方日志） |

判定项数：**m=1 → 74、m=8 → 251、m=33 → 545、m=64 → 716，全部 PASS**（= 分段 58/235/529/700
+ e2e 链 5 + **量化规则见证 11**，见 §5.5）；
换层复核（layer 8 / token 2000 / m=1）74/74 PASS。判据之外另有 **1 条非空洞性 guard**
（`[check_ref][guard] Σt_e == m·topk`），按 `docs/17` §2.1 与判定项分栏打印、**不计入**总数
（对照 `docs/17` §6 的 m15 口径「761 判定项 + 109 guard」）；guard 不成立时改为硬中止。
**本表的数字由 `evidence/m60_check_ref_run.log` 逐行核对生成**（§5.5 的 `W*` 行另见
`evidence/m60_quant_rule_witness.log`）
（`docs/17` §2.6 自洽；核对脚本未入库，以日志与本表为准）。M29 时代的同名日志
`evidence/check_ref_run.log`（63/240/534/705）**保留为改前存档**：M60 修改 `quant_hw` 的角落分支
后，两者的既有判定项**逐行完全相同**（回归核对的读数见 §5.5 与 `evidence/m60_quant_rule_witness.log` §5④）。

**预算口径说明**（完整分档与推导见 §5.3）：

1. **每段都报分位表**（`n / 越界数(含非有限) / mean / p50 / p90 / p99 / max / 最差占预算`），
   不只报 max；判定 = 越界元素数 0；`NaN/Inf` 显式计入越界。
2. 输出侧张量的预算是 **T3 推导界**（`tol_bf16` / `tol_l1`，见 §5.3 的表）：
   `nulp×spacing_bf16(ref) + noise`。`noise` 逐段写明来源 —— fast-math 段用
   `EPS_FAST·|ref|`（`EPS_FAST = 2·2⁻⁸`），长累加段用 `EPS_·Σ|terms|`（**逐元素算 L1，
   没有常量余项**）。每段"最差占预算"落在 **0.00–1.000**：其中 S9a `routed` 用到
   **0.999（m=1/8/33/换层）与 1.000（m=64）—— 零余量**（界与实测同量级，见 §7.9 的风险说明），
   `Y`/`Y_shd` 0.92–0.95、`moe` 0.99 同理：这些段的界不是"宽松地过"。
3. **`moe_output` 按 kernel 的真实序列复算**：`CombineStage` 里 `moe = bf16(fp32(routed) +
   Mul 的 fp32 结果)` —— 用的是 `sigmoid(sgate)·Y_shd` 的**未取整** fp32 值，不是 `shared`
   落盘的 bf16 值。
4. `GU`/`Y` 的固定分位表按**最差槽位**给出（每槽一行都在日志里）。

> 已量化、已披露的一处规范差异（与 m13 §5.3 同类）：kernel 的**激活**量化走 m5 `MxQuant`
> （硬件 floor-指数 e8m0 规则），而 `tools/golden/moe_block_ref.py` 的权重打包规范是
> `ceil(log2(amax/6))`。`check_ref.py` 把该差异作为**参考项**打印：本次真实数据下
> `A_qx` **38.1%** 字节不同、`A_scale` **40.0%** 字节不同（与 m13 在合成数据上的 29–43% 同级）。
> 它不影响任何判定（权重走 checkpoint 字节、激活侧规范由 docs/12 §6 小改 C 指定）。

### 5.3 判据口径（按 `docs/17-verification-standard.md` 逐档标注）

`docs/17` §1 要求「按数据域分档 + 用哪一档必须在 README 写明理由」。本 mission 各段的档位：

| 档 | 用在哪 | 理由 |
|---|---|---|
| **T1 整数域/位域逐位** | `perm_src`/`perm_expert`/`counts`/`offsets`/`inv_slot`/`topk_ids`/`w_tk_packed`、`res1`、`res2`、`x_sorted`（gather 逐位）、`A_qx/A_scale/h_qx/h_scale`（**逐字节**）、e2e 链重算的 `ids`/`counts` 与链上量化字节、**§5.5 的 `W1`（定点读数）/`W3`（角落字节）/`W4`（设备交叉字节）** | 这些在 device 上是整数运算/纯搬运/纯量化，**每一步都可完整建模**（量化器规范与 m5 `MxQuant` 逐句一致、M60 起另钉了官方 `文件:符号` 与第二转写，且权重字节由 §4 的探针独立见证），不存在不可推导的残差 |
| **T2 可建模舍入 → ≤1 ulp** | —（本 mission 无纯 T2 段） | MoE 链上没有"只有一次舍入"的输出：所有 bf16 落盘量都经过 fast-math 路径或长累加链 |
| **T3 长 fp32 链 / 超越函数近似 / mmad-cube 累加 → 推导界 + 逐元素检查** | `logits`、`sgate`（fp32 GEMV，2560 项）、`x_norm`/`y_final`（NR-rsqrt + exp 门）、`topk_weights`（exp）、`H_swiglu`（silu/exp）、`shared`/`moe`（sigmoid/exp）、`GU`/`Y`（cube 累加 + fixpipe 落 bf16）、`routed`（10 项加权折叠）、e2e 链的 `moe`/`y_final`、**§5.5 的 `W6`（值域往返 `2·scale + …`）** | 见下 |
| **T4 结构性判据** | `--m` 包络拒绝、`Σt_e == m·topk`、perm 分组有序、`inv_slot` 自洽、非空洞性（换层/换 token 改 top-10 集合、`aclrtMemset(0xCD)` 污染）、**§5.5 的 `W0`（dump 身份 sha256）与 `W5`（方向统计的存在性/计数）** | 见 §5.1 与 `check_ref.py` 的非空洞性护栏；`W0`/`W5` 只报"结构事实"（是不是归档字节、码在不在允许集合里），不提供正确性推导 |

**T3 的界怎么推导**（权威版在 `check_ref.py` 的 `tol_bf16` / `tol_l1` / `EPS_*` docstring）：

```
tol = nulp · spacing_bf16(ref)  +  noise
      └── 输出网格项（fp32 输出时 nulp=0）──┘  └── 非舍入项 ──┘
```

* **输出网格项** `spacing_bf16(x) = 2^(floor(log2|x|)-8)`：bf16 尾数 7 位 ⇒ `[1,2)` 上步长 2⁻⁸。
  device 侧一次 RNE 落 bf16 的偏差上界是**半个 spacing**，故 `nulp = 1` 已是 2× 余量。
  **本 mission 无 `nulp = 2` 的段**（早前版本的"级联取 2"已被下面的显式累加项取代）。
* **非舍入项 `noise`** 三种来源，逐段写明：
  | noise | 系数（推导） | 用在哪 |
  |---|---|---|
  | `EPS_FAST·\|ref\|` | `EPS_FAST = 2·2⁻⁸ ≈ 7.8e-3`：device 的 `Reg::Exp` / Newton-Raphson rsqrt 在 **bf16 域快速路径**上工作 ⇒ 每次域内舍入引入 ≤0.5·2⁻⁸ 相对误差，按 2 次计。**这是一条显式的建模假设**（依据：m5/m6 donor 的 Exp/NR 路径以 bf16 域中间量为准），**待独立探针标定 `Reg::Exp` 精度后可收紧** —— 已在 §7 记为遗留项 | `x_norm`/`y_final`/`topk_weights`/`H_swiglu`/`shared`/`moe`(门控项)/e2e `moe`,`y_final` |
  | `EPS_ACC·Σ\|terms\|`（逐元素算 L1，**无常量余项**） | `EPS_GEMV = 2·48·2⁻²⁴`（Dot8Row 的 40 次 chunk 累加 + 64-lane 归约树 ~8 级，×2 余量）；`eps_cube(K) = 2·(K + (K/128+1)·8)·2⁻²⁴`（cube 的 K 次累加 + 每 kBlock 归约，×2 余量，mmad 内部顺序未公开）；`EPS_FOLD = 2·11·2⁻²⁴`（10 项乘加 + 收尾） | `logits`、`sgate`（GEMV）；`GU`/`Y`/`GU_shd`/`Y_shd`（cube）；`routed`（折叠）；`moe` 的加法项 `2·2⁻²⁴·Σ\|加数\|` |
  | `0`（纯网格） | — | 无（本 mission 的 bf16 输出段都带至少一项非舍入噪声） |
* **逐元素、不按 |out| 缩放累加项**：`logits` 的界用 `Σ|x_norm·w_e|`（该元素的 L1）而**不是** `|logit|`
  —— 相消元素上后者会失效（`docs/17` §1.1 的立条理由）。
* **δ 与边界元素**（`docs/17` §1.1）：δ = `noise`（逐元素）。本 mission 各段的 δ 大多 ≥ spacing/2
  （非舍入噪声大于网格步长）⇒ 这些段**分类退化、全部元素记为边界**，判定直接用**绝对界**
  （网格项 + 非舍入项），正是 §1.1 要求"同时满足绝对界"的那一半；少数段（如 `Y`/`moe`/`routed`）
  δ < spacing/2，分类给出常规/边界计数。`check_ref.py` 每段打印分类结论与计数，**不做判定松弛**。
* **非有限元素算越界**：`report()` 显式把 `NaN/Inf` 计入越界数（`NaN > b` 为 False，只看
  `ad > b` 会漏判），并在每段打印「非有限 N」。
* **判定项与报告项分离**（§1 第 4 条）：判定项只含上表 T1/T3/T4 的检查；「device 字节 vs
  `moe_block_ref` 的 ceil 权重规范」的字节差异、`topk_weights` 区间、rms 摘要一律标
  `[参考]`/`[报告]`，不计入 PASS 计数。
* **五条反例纪律**（§1 第 5 条）：分位表来自**同一 commit** 的归档日志（本 README 的 §5.2
  数字由该日志逐行核对生成）；不引用不存在的文件；5 份 dump 的每段 sha256 均入库
  （`evidence/dump_manifest.md`）；算件来源与 m13/m5/m7/m8/m3 的逐条差异见 §1 表；
  「证明不了」的部分在 §7 写清（例：权重字节同一性证明的是**搬运与摆位**，**不**证明数值
  语义 —— 后者由 §4 的解包探针 + §5.2 的 e2e 链见证）。

### 5.4 端到端（不打断）链 —— 对 §5.2 逐段口径的补强

§5.2 的逐段比对把每段的输入取成 device 上一段落盘值（**逐 op 隔离**），好处是段内错必被
本段抓住，代价是 **`x → moe_output` 的整链从未在单次独立复算里走过一遍**（`docs/17` L0
「不许拿设备自己的中间输出当参考」）。因此 `check_ref.py` 另有一条**不打断**的链：

* `e2e_chain(W, x, m)`：从**真实 x** 起步，按 kernel 的段序与数据流一路复算到
  `moe_output` / `y_final` —— S1 用 double RMSNorm → 落 bf16，S2 路由，S3 计数排序，
  S4 permute，S5 量化（自己实现），S6/S7/S8（自己量化出的字节 → double GEMM → 落 bf16），
  S9 折叠/门控/combine，S10 RMSNorm#2。**全程不取任何 device 中间量当输入**；每个"落盘"
  点都按 device 的数据流做一次 bf16 RNE。
* 判据（列入**判定项**）：
  1. 链上重算的 `ids` / `counts` 与 device 逐位一致；
  2. **链上量化出的 `A_qx/A_scale/H_qx/H_scale` 字节 vs device 字节** —— 把差异源钉死在
     量化点上（实测 0 字节不符）；
  3. `moe_output` / `y_final`：用**误差预算合成**给出界 —— 再跑一条
     `B = e2e_chain(..., xnorm_override=device 的 bf16 x_norm)`，则
     `|A − dev| ≤ |A − B| + |B − dev|`：第一项是 S1 段残差在链上的**实际放大**（逐元素实测，
     不是拟合、也不是"实测 X 当界"）；第二项 `tolB` 目前只按 fast-math 项
     `1×spacing + EPS_FAST·|B|` 取（**未**并入 `EPS_GEMV` / `eps_cube(2560)` / `eps_cube(640)` /
     `EPS_FOLD`；因此该界**偏紧**，实测占用 ≤0.61）。若要严格同 §5.2 口径，应把各段 ε 逐项
     合成后再判 —— 本轮按"偏紧不会假 PASS"保留，并在日志里给出实测占用供复核。
* 实测：m=1/8/33/64 与换层换 token 全部 PASS，其中 m=1 与 m=8 的 `moe`/`y_final` 与 device
  **逐位相同**；m=33/m=64 的 `e2e moe` 实测 max 1.9e-6 / 2.4e-4（占预算 0.53），
  全部实测占用 ≤0.61。
* **这条链证明什么／不证明什么（`docs/17` §3 第 5 条）**：它证明的是
  `|A − dev| ≤ |A − B| + tolB` —— B 的输入含 device 的 `x_norm`（故该项是 slack），
  **并不等于证明了 `A ≈ dev` 作为一条独立的整链等价性**；`A − B` 已被显式计入界内，
  而 S1 段本身另有 §5.2 的独立判据。换言之：本项把"S1 之外的整链能否一步不落地复现 device"
  变成可判定命题，但不声称"A 链是 device 的逐位等价物"（m=1/8 的逐位相同是本轮数据上的
  观测结果，不是界推导的前提）。

> 与 M26 的关系：M26（`agent-qgolden`，commit `6f68f8e`）的 `tools/golden/data/*/qaware/floor`
> 是**官方 ops-nn 序列**（OCP `npu_dynamic_mx_quant(round_mode="round")`）的逐句转写，与本
> kernel 激活侧规范同源。本 mission 的「device 字节 vs ceil 规范」参考项差异正是该规范差异，
> 与 m13 §5.3 同类、同量级；M26 的 w4a4 golden 是对**合成数据集**（E=4）建的，真实权重链
> 没有对应 golden，故本 mission 不套用 `1.5·|qaware−float|` 判据，只用「设备自身量化字节 +
> 真实权重」的 double 参考链 + §5.4 的 e2e 链（口径见上表）。

**证据归档**（`m17_moe_real/evidence/`）：

| 文件 | 内容 |
|---|---|
| `accept_run_m1_m64.log` | m=1/8/33/64 + 换层（layer 8/token 2000）的完整 device 日志（含 3 条腿的字节核对、cube 解包探针、12 条不变量判据）× 5 次运行 + 确定性复核（**本次落盘 2 份** m=1 的整段 workspace sha256 完全相同 —— 批内 1 次 + 抖动块最后一次，同一 sha `1a0f4883…` 分别在 `/tmp/m17_accept/` 与 `/tmp/m17_jit/`；**历史轮次另有 3 份相同**。措辞与该日志 `########## 确定性复核：…` 那一行的括注对齐，随 M69 订正）+ `--m=65` 拒绝实例 |
| `check_ref_run.log` | **M29 存档**：`check_ref.py` 在 5 个 case 上的全量日志（63/240/534/705 条判定项）。M60 之后**不要**再引用它当"当前读数"，改用下面的 `m60_check_ref_run.log`（两者既有判定项逐行相同，本文件保留为改前记录） |
| `m60_check_ref_run.log` | **M60 复跑**：`check_ref.py` 在同样 5 个 case 上的全量日志 —— **74/251/545/716 条判定项**（分段 + e2e + §5.5 的 11 条量化规则见证），本 README §5.2/§5.5 的所有当前数字都出自它。**两轮评审后因 `[coverage]`/`[报告]` 文案修正各重生成过一次**：相对首次评审所评的 tip（`a875204`），`git diff --numstat a875204..8d4f2f1 -- m17_moe_real/evidence/m60_check_ref_run.log` = **16 insertions / 16 deletions**（= 头部 1 行 + 5 case × 2 条 `[coverage]` 行 + 5 case × 1 条 `[报告] W6/W4/W5 的自指形态…` 行；`a875204..5472bd8` 与 `a875204..53730e9` 时该数为 **11/11**，`32603f8` 的第二次重生成把它抬到 16/16），`git diff a875204..8d4f2f1 -- <该 log> \| grep -cE '^[+-]\[check_ref\] (PASS\|FAIL)'` = **0**（判定项行差异零）、`RESULT` 文案跨 tip 逐字不变 —— 这三处读数都由这两条命令在 §5.6 规则 1 下当场复算过 |
| `m60_quant_rule_witness.log` | §5.5 的原始读数：W0–W6 与 5 条负向对照在 5 个 case 上的逐条记录 + 「已知会被漏掉」的篡改对照 + base 版镜像的角落分叉 + 自指形态/覆盖范围交代。**各块的来源见生成文件头部的「来源逐块标注」**（逐块列出哪些是派生、哪些是引用/说明并注明出处；例如 §1/§2 的读数与条数、§3 的计数、§4 的角落读数、§5②、§5④ 的回归比对、§5⑥ 的改动范围与计时项计数属派生，§3 的实验描述、§4 的语义解释与「0 个角落组」、§5①/§5③/§5⑤/§5⑥ 的说明与引用属非派生） |
| `m60_tamper_experiment.log` | §3「已知会被漏掉」的负向对照的**仓内输入**：在归档 dump 上篡改空槽位 `hq[0][0][0]` 与活动行行距尾部 `hs[41][0][20]` 后 `check_ref.py` 的完整输出（`RESULT: FAIL (1/74)`、唯一 FAIL = `W0`、内容判据 73 PASS）。文件头内嵌复现命令与期望 sha —— 因此 §3 不依赖任何仓外路径 |
| `m60_dump_pin.log` | 归档 dump 身份核对：5 份 `ws_*.bin` 的整段 sha256 vs `dump_manifest.md`，以及 m=1 的 **32 段**逐段复算 |
| `m60_extractor_selftest.log` | **提取器的自检输出**（六种合成输入：正常 / 0 case / 含 FAIL / 缺文件 / 只有非判定行 / 分隔符不可解析）——实测退出码 `0/2/1/2/2/2`、每种状态的汇总行当场派生；后两态是「已知会被漏掉」在提取器上的形态（**解析不到就不发合格证**）。可复算（`--selftest` 重跑逐字节相同；输出里不打印随机临时路径） |
| `dump_manifest.md` | 5 份 dump 的**每段 sha256 + 字节数**（32 段/份）+ 再生成命令 |
| `weight_bytecheck.log` | 最后一次运行的 14 条「device D2H == host 预读」逐字节核对记录 + 合计 |

### 5.3.1 审校脚本的三态退出码与负向对照（tower 规则「审校脚本必须报 SKIPPED」）

`check_ref.py` 与 `gen_weight_manifest.py --check` 都实现了三态退出码
（`0` = 比过且通过 / `1` = 比过有差异 / `2` = 没得比或输入缺失 → `RESULT: SKIPPED`），
**OK 文案里带实际比较计数**；`check_ref.py` 还在每次运行末尾打印 `[check_ref][coverage]`
**覆盖范围交代**（计数与匹配器同源、三态分栏："全量比较 / 部分比较 / 不在本脚本范围"），
并明说"**本脚本的 OK 不等于整个 workspace 都比过了**"。
下列读数（含 M60 新增的 5 条负向对照与 1 条「已知会被漏掉」实验）都实测归档如下
（判据自己必须先会咬）：

| 脚本 | 输入 | 输出要点 | 退出码 |
|---|---|---|---|
| `check_ref.py` | 真 dump（m=1） | `RESULT: OK (74 条判定项全部比较通过：分段判据 58 + e2e 链 5 + 量化规则见证 11（含负向自检 5）；另有 1 条非空洞性 guard 单列、未计入)` | **0** |
| `check_ref.py` | 真 dump **篡改 1 字节**（`x_sorted` 首字节 XOR 0x01，整段 sha256 变 `234658cd…`） | `FAIL S4 x_sorted = gather(x_norm, perm_src): 1/25600 元素不符` → `RESULT: FAIL (n/74 条判定项越界)` | **1** |
| `check_ref.py` | dump 目录不存在 / 目录里无 `ws_*.bin` / tag 不存在（3 例） | `RESULT: SKIPPED（…未比较任何判据）` | **2** |
| `check_ref.py` | **「已知会被漏掉」负向对照（M60 重做）**：篡改**空槽位** `hq[0][0][0]` 与**活动行行距尾部** `hs[41][0][20]`（各 XOR 0x01，见 §5.5） | `RESULT: FAIL (1/74)` —— 唯一 FAIL 是 **`W0`**（整段 sha `fdb6721b…` ≠ 归档 `1a0f4883…`）；**内容判据仍全 OK** ⇒ ① 整段字节层面的篡改 M60 起**不再漏**（W0 咬住）；② 上述两处内容字节仍是**已披露的覆盖边界**（`[coverage]` 每次都打印"只比活动槽前 t_e 行 + `[:K/32]`"） | **1**（内容边界不变；W0 是新增能力） |
| `check_ref.py` | **W3 负向对照**：用 base 版镜像（`M17_BASE_REV`，默认 `f0286f6`）跑同一判据 | `PASS 自检-W3n：f0286f6:…::quant_hw 与 quantize_ocp 失配 5/85 字节（+Inf 组 scale 253/code 7 vs 官方 255/0）` —— 修复前的镜像**必须**与官方分叉，否则角落判据失明 | **0**（W3n 本身是"必须检出分叉"，故 PASS） |
| `check_ref.py` | **W4n/W5n/W6n 负向对照**：方向反的 legacy 规则（`quant_rule_reversed`）跑同三条 device 侧判据 | `W4n 88.2% 字节不符`（m=1）/`W5n 违反 1089/1100 组`/`W6n 越界 1123/7040` ⇒ 三条见证对方向级错误都是活的 | **0**（同上，PASS = 检出活体） |
| `gen_weight_manifest.py --check` | 正确 manifest | `RESULT: OK（已比较 16 条张量记录的 offset/bytes/shape/dtype/sha256…）` | **0** |
| `gen_weight_manifest.py --check` | manifest 被篡改（`offset=` 全改 1） | `RESULT: DIFF（…已比较 16 条张量记录）` | **1** |
| `gen_weight_manifest.py --check` | manifest 不存在 | `RESULT: SKIPPED（…无从对账）` | **2** |

> **M29 → M60 的一处行为变化（必须知情）**：第 2 行「篡改 1 字节」与第 4 行「篡改空槽位」在 M60 之前
> 的读数分别是 `1/63` 与 `OK`；M60 起 `W0`（dump 身份 pin）会**另行**把整段 sha 的变化报出来，
> 所以任何"在归档 dump 上改字节"的实验都会多出一条 `FAIL W0`。这不是判据退化：`W0` 是一条
> **结构性（T4）** 断言——它回答的是"本轮判的是不是归档字节"，与内容正确性判据正交；M60 之后
> 做内容篡改实验时，**要按"除 W0 外全 PASS"读**（§5.5 的 §3 节就是这么记的）。

### 5.5 量化规则的对外 pin 与设备无关见证（M60）

**改动前的状态（M59 判为 `S2 残余`）**：`quant_hw` 的 docstring 只写"m5 `MxQuant` 全 VEC 路径的
numpy 镜像"，**没有任何 `文件:符号` 外部引用**；而 S5/S7 的 `A_qx/A_scale/H_qx/H_scale` 字节判据
与 §5.4 的 e2e 字节判据**全托在它上面**。这与 m3 修复前的位置同型：镜像与 device 若共享同一条
读法错误（方向/规则级），逐字节仍然 0 失配、全 PASS（m3 实测 94.8% 字节差异却全 PASS 数月）。

#### 5.5.1 pin：规则来源与它的强度（自评）

| 层级 | `文件:符号` | 作用 |
|---|---|---|
| **官方（仓外）权威** | `add_rms_norm_dynamic_mx_quant_common.h::MxQuantComputeMaxExpOCP`（:308-356）、`::MxQuantComputeScaleOCP`（函数自 :358 起，其**缩放循环体** :397-418）、`::MxQuantComputeDataFP4` 的 bf16 分支（:751-758；函数自 :661 起），常量 :71-98 | 规则本身。源码在 `/workspace/ops-nn/norm/add_rms_norm_dynamic_mx_quant/op_kernel/arch35/`，与随包 CANN 的 `opp/.../dynamic_mx_quant/arch35/` 同构（**两份副本行号不同**；括号里给的是 ops-nn 现版本、M60 已逐行核过）。⚠ `docs/13-mx-quant-primitives.md` §4 记的是 M28 当时那次阅读的区间，其中 `MxQuantComputeMaxExpOCP` 那处（`:381-401`）与本机现版本对不上（现版本该函数在 `:308-356`，`:381-401` 落在 `ScaleOCP` 的循环体内）—— M60 的 reviewer 已就此另开 finding，`docs/**` 不在本 mission scope，故本目录只按现版本行号写 |
| **仓内第二转写** | `tools/golden/moe_block_ref.py::quantize_ocp`（:293-348，逐句带官方行号） | W3/W4 的逐字节对照方 |
| **device 符号（对照 pin，不是规则来源）** | `m17_moe_real/m17_moe_layer.asc::MxQuantComputeScale` / `::MxQuantComputeDataFP4`（`0x7f81` 的 `Select` 覆盖在 :1631/:1660） | 说明 device 侧跟的是同一条序列；只读引用 |

> 表中与本节其余处的**行号只是查阅提示**：定位一律以**符号名/常量名**为准，行号随上游版本与
> 本仓注释变动（同 §1 第 10 行与 `docs/05` §6.1 ⓒ 的口径）；只有**归档证据**里的行号按当时事实冻结。

**pin 的强度（按 M59 §1.2 第 2 条自评，必须说清"这不是两处独立"）**：`quant_hw` 与 `quantize_ocp`
是**同一工作区里两个 agent 对同一份官方头文件的两次转写**（同源转录 / *correlated
transcription*）：共用同一组常量、同一份读法、同一种"在 bf16 位域上手工搭序列"的实现风格。
因此 **W3/W4 的逐字节一致只能咬住"转录笔误 + 角落分支实现差异"**（§5.5.3 的 W3n 就是实例：
修复前的镜像在 ±Inf 角落与官方分叉，被 W3n 咬出 5/85 字节失配），**咬不住"对规范的共同误读"**。
把这两者写成"两处独立见证"是**过度声明**；真正的独立来源只有官方头文件本身（代码里写出
`文件:符号` 只是可追溯，不等于我们已经验证了那份头文件），以及**只引用"MXFP4 group32 的数学"**的
W1/W5/W6（量化实现只以"两份转写并列"的形式出现，不拿其一当基准）。

> **M60 顺带修掉的一处真实分叉**：修复前 `quant_hw` 缺官方两条覆盖——① 非有限组（官方 :406
> 给 E8M0 NaN 字节 `0xFF`、:413 给 `halfScale = 0x7F81` ⇒ 全组 code 0），旧实现给 scale `253`、
> 该组 code `7`（饱和成 6.0）；② 指数域 < 2 的退化组（官方 :402-403 把 `shared` 夹到 0、:414
> 令 `halfScale = 0` ⇒ code 只剩符号位），旧实现给 code 4。三处符号位/零的细节已与官方逐句对齐
> （`quant_hw` docstring 里逐条写明行号）。**真实数据不触发这两个角落**（5 份归档 dump 的全部
> 活动组里非有限组 0 个、指数域 < 2 的组 0 个），所以修复后 63/240/534/705 条既有判定项
> **逐行不变**（回归读数见 `evidence/m60_quant_rule_witness.log` §5④）。

#### 5.5.2 每条判据「咬得住什么 / 咬不住什么」（自指形态按 M59 的 S1/S2/S3/S4 代号标注）

| 判据 | 口径 | 咬得住 | 咬不住 | 自指形态 | m=1 实测 / m=64 实测 |
|---|---|---|---|---|---|
| `W0` dump 身份 | `ws_*.bin` 整段 sha256 vs `dump_manifest.md` | "本轮判的是不是归档字节"；任何整段篡改 | 内容正确性（它**不是**正确性判据） | **T4 结构性**（无正确性咬合力） | 相符 5/5 |
| `W1` 规则方向定点 | `amax∈{0.5,1,2}` 的定点读数：scale 字节 `124/125/126`、code `6`、反量化回 amax | **方向级**错（乘/除反了 ⇒ code 0/1/2）；e8m0 字节的 floor 规则 | floor 与 ceil 的差别（`amax` 为 2 的幂时两者同解）；上游（S1–S4）的错 | **N2 外部 pin + 规则不敏感**（不读 dump、不引用任何镜像实现） | 两实现均 `byte=(124,125,126) code=(6,6,6)` |
| `W3` 角落平价 | `quant_hw == tools/golden::quantize_ocp`（±Inf/NaN / 次正规 / 全零） | 镜像的角落分支实现差异（**同源转录**层能咬的那一半） | 共同误读；device 侧行为（这条不读 device） | **N2 + 同源转录** | 3/3 角落逐字节一致 |
| `W4` 设备交叉字节见证 | device 的 `A_qx/A_scale/H_qx/H_scale`（routed + 共享）vs `quantize_ocp` | device 与"官方序列的另一次转写"之间的任何分歧；scale 与 nibble 分开计数 | 共同误读（见 5.5.1）；空槽位/行距尾部（覆盖边界） | **S1（输入取 device 的 `x_sorted`/bf16 `GU`/`xnorm`）+ 同源转录** | 0/18700；m=64 **0/1196800** |
| `W5` 设备侧方向定点 | 每个活动组内 `|amax|` 那个元素在 device 打包字节里的码必须 ∈ `{6,7}`（= 值 4.0/6.0） | device 的**方向级**错（反向实现会把该元素压到 0/1 或顶到饱和）；统计覆盖**全部**活动组 | **大 `amax` 组**（`amax ≥ 4`，即 bf16 指数域 ≥ 129、`E = ⌊log₂amax⌋ ≥ 2`）上反向规则也会落到码 7 ⇒ 那些组不区分（推导：反向 `h = amax·2^(byte−127)`、`byte = field−2`，`E = 2` 时 `scale = 1`、`h = amax ≥ 4` ⇒ 最近码为 4.0/6.0；`E ≤ 1` 时 `h < 2` ⇒ 码 2/3/4，W5 判得出。本数据里**每个** `amax ≥ 4` 的组都不区分：m=1 **11/1100** = 1100−1089）；ceil/floor 差别 | **S1 + 规则不敏感** | 违反 0/1100；m=64 **0/70400** |
| `W6` 值域往返 | `|dequant(device H 字节) − h_ref|` ≤ `2·scale + (1×spacing + EPS_FAST·|h_ref|)`；`h_ref` = host double SwiGLU（**不由 device 字节反推**） | 规则/方向级错（反向后码被压到 0 或放大 ⇒ 越界数爆）；device 与 host 参考的值域一致 | `h_ref` 自身误差（其输入是 device bf16 `GU` ⇒ 由 S7 判据见证，本判据只把它的界并入预算）；只覆盖活动槽前 t_e 行 | **S1 + 规则不敏感** | 越界 0/7040，最差占预算 **0.953**；m=64 **0/450560，0.955** |
| `自检-W2/W3n/W4n/W5n/W6n` | 同上四条判据的**负向对照** | "判据本身会不会咬"：方向反的 `quant_rule_reversed` 与 base 版镜像**必须** FAIL | —（它们不判被测件，只判判据的活性） | 同上各自的形态 | 全部 PASS（读数见 5.5.3） |

`[报告] W6b`（只打印、不计判定）：同一口径但 `h_ref` 取 §5.4 的 **A 链**（输入 = 真实 x，全程不取
device 中间量）⇒ m=1 `|Δ|` p50 8.06e-3 / max 3.28e-1、最差占预算 0.947、越界 0/7040。它比 W6 更
接近 S1-free，但它的界还含 A 链 S6 段的残差（本轮未逐项合成），故**只报不判** —— 真正的
S1-free 锚点仍是 §5.4 的 e2e A 链本身。

#### 5.5.3 负向对照：判据必须先会咬（m=1 读数；5 个 case 全部同量级，见 `evidence/m60_quant_rule_witness.log`）

| 负向对照 | 构造 | 实测（m=1） | 结论 |
|---|---|---|---|
| `自检-W2` | 方向反的定点规则 | 定点 code `(0,1,2)`（正确规则 `(6,6,6)`），而**scale 字节与正确规则逐字节相同** | 只比 scale 字节的 vf-probe 对 data 侧翻转**完全无感**（M59 §6 第 3 步）。**平局口径已注明**：`amax=1.0` 那格反向 `h=0.25` 与 E2M1 的 `0.0`/`0.5` **精确等距**（tie）；本镜像按 `CAST_ROUND`（远离零）给 code **1**，**tie-even** 实现会给 code **0** —— 两种口径下 code 都 `≠6`，故 W2 的判据**不依赖平局口径**（`docs/17` §1.4；另两格 `amax=0.5/2.0` 给反向 `h=0.0625/1.0`，非平局，code `0`/`2`） |
| `自检-W3n` | `git show f0286f6:m17_moe_real/check_ref.py` 的 **base 版镜像** | 与官方失配 **5/85 字节**（+Inf 组 scale 253/code 7 vs 255/0；`amax=2⁻¹²⁶` 组 code 4 vs 0） | 角落判据是活的；也说明"同源转录"确实咬得住分支差异 |
| `自检-W4n` | 方向反规则 vs device 字节 | **88.2% 字节不符**（16490/18700） | 设备字节见证对方向错是活的 |
| `自检-W5n` | 方向反规则的 argmax 码统计 | 违反 **1089/1100 组** | 同上 |
| `自检-W6n` | 方向反编码 + 值域往返判据 | 越界 **1123/7040** | 同上 |
| 「已知会被漏掉」（§5.3.1 第 4 行） | 篡改空槽位 `hq` 与活动行行距尾部 `hs` | 内容判据全 OK（唯一 FAIL 是 `W0`） | 覆盖边界**已披露**（`[coverage]` 每次打印） |

**与 M59 归档标尺的对照**（同一口径）：m3 修复前 pack 字节差异 **94.8%**（M60 在真实权重上
87.8–89.3%）；m1 golden 上反向 **283/640 = 44.2%**；m3 的 `|deq−h|/|h|` p50 **142**、max **5.79e76**
（M60 的 W6n 在同一相对口径下 p50/max 都是 **1.0**）。

> **为什么 W6 用「绝对界 + 预算比」而不是相对误差作判定**：相对误差在 `h_ref → 0` 处无界，不能当
> 判据（`docs/17` §1.1 的口径）；而 m3 与本目录的相对读数之所以差 142× 与 1.0×，本质是**数据尺度**
> 不同（m3 的激活幅度大、`scale > 1` ⇒ 反向后 `deq ≈ h·scale²` 反而变大；本目录真实 MoE 激活幅度小、
> `scale < 1` ⇒ 反向后码被压进 e2m1 的 0 档，相对误差饱和在 1.0）。W6 的绝对界 `2·scale` 直接来自
> e2m1 码表的最大间距（组内 `amax/scale ∈ [4,8)`，最近码 4.0/6.0 ⇒ 误差 < `2·scale`），与数据无关。

#### 5.5.4 覆盖范围与限度（三态分栏 + 同源计数 + 已知会漏掉）

* **计数与匹配器同源**：W4/W5/W6 的字节数、组数、越界数都由 `check_ref.py` 的**同一个循环**产出，
  `[coverage]` 行里的活动槽数/行数取自同一次运行的 `counts_dev`；README 与证据日志只用正则从
  运行日志提取，不另写数字（`docs/17` §2.6）。
* **三态分栏**（`check_ref.py` 每次运行都打印）：**设备无关**= W1/W2/W3/W3n（只吃合成输入，不读
  dump）；**吃归档 dump**= W0/W4/W4n/W5/W5n/W6/W6n（覆盖 = 全部活动槽的前 t_e 行；m=1 → 18700 个
  A/H 量化字节 + 1100 个活动组；m=64 → 1196800 字节 + 70400 组）；**不在覆盖内**= 空槽位（m=1
  跳过 502 个）与各行 padding、`hs`/`as` 的行距尾部字节、`W6` 的 `h_ref` 自身误差。
* **已知会被漏掉的那一条**：§5.5.3 最后一行（篡改空槽位 `hq` + 活动行 `hs` 行距尾部）——M60 既没
  扩大也没缩小这个边界（与 M29 的 S5/S7 字节判据边界相同）。
* **哪些判据"不得计入独立端到端"**：W4/W5/W6 的**输入**含 device 中间量（`x_sorted` / bf16 `GU` /
  `xnorm`）⇒ 按 M59 §1 都是 **S1（隔离判据）**。它们不能替代 §5.4 的 e2e A 链（那条从真实 x 起步）。
  同样地，`quantize_ocp` 属**同源转录**，不得被引用为"独立第二来源"。
* **本轮不新增 device 代码**：`git diff --name-only main...HEAD` 的路径清单只在 `m17_moe_real/**`
  （`m17_moe_layer.asc` / `m17_resources.h` 零改动）⇒ **"禁 memory-based API" 不适用**（本段全是
  host 侧 numpy/字节判据；device 侧只被引用两个 pin 用符号 `MxQuantComputeScale` /
  `MxQuantComputeDataFP4`，见 5.5.1，可当场 `grep` 核行数）。
* **host 墙钟不作为证据**：W0–W6 的判定量只有字节相等（sha256/nibble/scale）、整数计数
  （失配/越界/违反组数）与推导界下的分位；**没有任何一项读时间**（对照 §7.5）。
* **验证形态**：本节的 5 个 case 全部跑在**已归档 dump** 上（整段 sha256 与 `dump_manifest.md`
  逐份核对，见 `evidence/m60_dump_pin.log`），**本轮未新跑 device** —— device 侧算件零改动，
  没有新 dump 的需求（复核时 `npu-smi` 显示 HBM 上有其它 agent 的进程在跑，也支持这一取舍）。
* **「没跑 device」成立的前提与失效条件（评审判定内行为，落成于 `5472bd8`）**：前提是
  (i) 本 mission 对 device 算件零改动（`m17_moe_layer.asc` / `m17_resources.h` 三点 diff 为零）；
  (ii) **归档 dump 就是当前 device 算件的输出** —— `m17_moe_layer.asc` 的最后一次改动
  `2a4fa0b` 正是入档 `evidence/dump_manifest.md` 的那一次；(iii) 归档身份由 `W0` 与
  `dump_manifest.md` **双向钉住**（5 份整段 sha256 逐份相符）。**失效条件**：一旦 device 算件
  （`.asc` / `resources.h`）改动，就必须**重跑 device 并重生成 `dump_manifest.md`** —— 在那之前
  `W0` 会先 FAIL 提醒（文案即「≠ 归档字节！W4/W5/W6 判的不是归档 dump」），这是设计内行为，
  不是误报；**不要**用放宽/跳过 `W0` 的方式绕过。

### 5.6 编辑与交卷自查纪律（自 `53730e9` 起；本目录的硬性交卷项）

本目录在 M60 的三轮评审里连续出现**同一形态**的三次拦项，全都是「**关于交付物自身的一段文字
没有被当场核**」，且**三次都由 reviewer 而非作者自查抓到**：

| 评审工件（行内为完整相对路径） | 那一处 | 形态 |
|---|---|---|
| `.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r1.md` | `check_ref.py` 的 `[coverage]` 打印「篡改空槽位仍报 OK」 | 打印与同 commit 事实相反（M60 起 `W0` 会 FAIL） |
| `.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r2.md` | `gen_witness_log.py` 标题「5 个 case 全 PASS」 | 手写常量冒充派生结论；0 条读数仍 exit 0 |
| `.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r3.md` | 「唯一非派生的是 §5⑤」+ 引用「前两行两个评审工件的 reviewer 数过」 | 自述过度概括 + 引用别处记录（被引的两次评审里，有一次并没重数） |

因此自 `53730e9` 起，本目录的交卷自查**必须**包含：

1. **逐句核断言**：把本轮新写的每一句断言性文字，逐句核一遍它引用的**命令 / 文件 / 计数** ——
   - 对符号/文件的用途下判断：引用声明处原文，或给当场计数（不自己概括用途）；
   - 不引用别处的措辞/行为当证据（引用者一改，这句就变成假陈述）——要写成**自足表述**，
     或写明「谁做了什么、谁没有做什么」；
   - 不用「全部 / 唯一 / 任何」这类**绝对量词**去概括证据文件的来源，改用**逐块列清单**
     （`gen_witness_log.py` 输出的头部「来源逐块标注」就是这条的模板）。
2. **块集合核对**：改文档/代码后，除核**文件尾部**与**章节序列**外，还要按空行分块比**块集合**
   （确认没有静默删失）。**本目录自 `53730e9` 起按此法自查**；作为参照，
   `.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r3.md`
   记了当时两份文件的同类比对读数（README `84→84` 块、witness log `52→51` 块，「旧块不在新块里」
   的全是标题/段落、无一是逐条读数行）。
3. **派生优于手写**：凡是工具印出的计数/结论，必须由**与匹配器同源的遍历**当场算出；
   实在派生不了的（外部归档引用、需要读 device dump 的读数）必须**逐条注明出处**并标为
   「非派生/引用」，且给出**可复算的路径**而不是「谁数过」。
4. **三态 + 分母 + 一条「已知会被漏掉」的负向对照**：审校/统计类脚本必须有
   `0`（比过且通过）/`1`（比过有差异）/`2`（没得比 → `SKIPPED`，**绝不发合格证**）三态，
   并在 `OK/SKIPPED` 文案里带**实际比较计数**；`gen_witness_log.py --selftest` 的六态自检
   是本目录这一条的范本（其输出归档为 `evidence/m60_extractor_selftest.log`）。
5. **交卷前的「绝对量词 / 来源断言」扫描 —— ★这是**报告项，不是判定器**（M60 `77897d3` 起）**

   **扫描范围（显式清单；scope 外目录不扫；= 12 个文件 = 3 个源码 + 9 个 `evidence/*.log`）**：
   `m17_moe_real/README.md`、**`m17_moe_real/check_ref.py`**、`m17_moe_real/gen_witness_log.py`、
   `m17_moe_real/evidence/*.log`（当前 9 个）。
   **源码必须在内**：`check_ref.py` 是**源码**，而 M60 的 C1 修复里有 2 处就落在它身上
   （`5472bd8` 的 `[报告]` 行、`77897d3` 的 docstring/注释）——只扫文档会漏。这条是 `32603f8`
   的清单遗漏后补进来的（当时清单里没有 `check_ref.py`；在**当时那套 5 词词表**下它有 18 条命中，
   在本节的量词词表下有 56 条 —— 换词表就要重跑，别沿用旧数）。

   **命令**（自足 heredoc；只在内存里跑、不写任何文件；输出即下表 + C1 明细）：

   ```bash
   cd <worktree 根>
   python3 - <<'PY'
   #!/usr/bin/env python3
   """M60 §5.6 的「绝对量词/来源断言」扫描（**报告项**，不是判定器；见 README §5.6 第 5 条的限度声明）"""
   import re, glob, sys, collections
   BAN = re.compile(r'全部|所有|唯一|任何|无一|不写死|一律|均|只|仅|每')          # 量词（含 所有/均/只/仅）
   SRC = re.compile(r'派生|来源|来自|出处|取自|产出|生成|结论|计数|数字|读数|统计|标注|引用|标尺')  # 来源语汇
   C1  = re.compile(rf'({BAN.pattern}).{{0,40}}({SRC.pattern})|({SRC.pattern}).{{0,40}}({BAN.pattern})')  # 行级 ∧
   C4F = {"m17_moe_real/evidence/accept_run_m1_m64.log", "m17_moe_real/evidence/check_ref_run.log",
          "m17_moe_real/evidence/weight_bytecheck.log"}                          # C4 冻结归档（docs/05 §6.1 ⓒ）
   C3  = re.compile(r'这类词|这类量词|不用「|看[不]?见什么|扫描器|扫描命令|grep -nE|分类器|BAN =|SRC =|C1  ='
                    r'|\|\s*r3\s*\||r1 – r5|被改掉的 C1|那一行（原文「|那一行（「|负向对照|注入'
                    r'|盲区|见下方限度|诚实降级|报告项|历史（r1|就是证据|不得把它当门槛')
   FILES = ["m17_moe_real/README.md", "m17_moe_real/check_ref.py", "m17_moe_real/gen_witness_log.py"
            ] + sorted(glob.glob("m17_moe_real/evidence/*.log"))
   tot = collections.Counter(); rows = []; c1 = []
   for f in FILES:
       c = collections.Counter(); fence = False
       for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
           if line.lstrip().startswith("```"):
               fence = not fence; continue
           if not BAN.search(line):
               continue
           if f in C4F: c["C4"] += 1
           elif fence or C3.search(line): c["C3"] += 1
           elif C1.search(line): c["C1"] += 1; c1.append((f, i, line.strip()[:100]))
           else: c["C2"] += 1
       tot.update(c); rows.append((f.replace("m17_moe_real/", ""), sum(c.values()), c["C1"], c["C2"], c["C3"], c["C4"]))
   print("| 文件 | 量词命中 | C1 疑似来源断言 | C2 其他（覆盖/分母/读数） | C3 引用语境/规则 | C4 冻结归档 |")
   print("|---|---|---|---|---|---|")
   for r in rows: print("| `%s` | %d | %d | %d | %d | %d |" % r)
   print("| **合计** | **%d** | **%d** | **%d** | **%d** | **%d** |" % (sum(tot.values()), tot['C1'], tot['C2'], tot['C3'], tot['C4']))
   print(f"\nC1 明细（逐条人工判读；本扫描**不是**判定器）：")
   for f, i, t in c1: print(f"  - {f}:{i}: {t}")
   PY
   ```

   **四栏**：**C1 = 疑似「来源断言」（`量词` 与 `来源语汇` 同行且相距 ≤ 40 字符）—— 逐条人工判读**；
   C2 = 覆盖/分母/读数（分母就在句子里，逐条可核）；C3 = 引用语境（规则条文、代码块里的命令与正则、
   历史复盘）；C4 = 冻结归档（M29 时期的 device/run 日志，按当时事实冻结）。

   **写本节时的实测**（`含本节自身与 §5.7 的引用语境；C1 为**人工判读对象**，见下方限度`；读数在
   **写本节时的 tip** 上取 —— 本文件的 `.md` 改动会立刻让它变，重跑即上方 heredoc）：

   ```text
   | 文件 | 量词命中 | C1 疑似来源断言 | C2 其他（覆盖/分母/读数） | C3 引用语境/规则 | C4 冻结归档 |
   |---|---|---|---|---|---|
   | `README.md` | 141 | 31 | 88 | 22 | 0 |
   | `check_ref.py` | 56 | 7 | 47 | 2 | 0 |
   | `gen_witness_log.py` | 35 | 12 | 22 | 1 | 0 |
   | `evidence/accept_run_m1_m64.log` | 37 | 0 | 0 | 0 | 37 |
   | `evidence/check_ref_run.log` | 847 | 0 | 0 | 0 | 847 |
   | `evidence/m60_check_ref_run.log` | 872 | 5 | 857 | 10 | 0 |
   | `evidence/m60_dump_pin.log` | 3 | 0 | 3 | 0 | 0 |
   | `evidence/m60_extractor_selftest.log` | 4 | 2 | 2 | 0 | 0 |
   | `evidence/m60_quant_rule_witness.log` | 34 | 6 | 21 | 7 | 0 |
   | `evidence/m60_scan_negcontrol.log` | 14 | 5 | 0 | 9 | 0 |
   | `evidence/m60_tamper_experiment.log` | 34 | 1 | 30 | 3 | 0 |
   | `evidence/weight_bytecheck.log` | 1 | 0 | 0 | 0 | 1 |
   | **合计** | **2078** | **69** | **1070** | **54** | **885** |
   ```

   **★ 它看不见什么（已实测的盲区，必须知情）**：
   ① **不含词表量词**的自然表述 —— 例：「本文件的结论**都**由提取器生成」（`都` 不在量词表）
   ⇒ **漏掉**；② **只有量词、没有来源语汇**的 —— 例：「§5⑤ 是唯一的例外」⇒ **漏掉**；
   ③ **纯表格/符号式表述**（没有实词可匹配）⇒ 漏掉。
   并且 **C1 命中里既有真断言、也有正当的覆盖语句**（如「全部 512 个专家」「分母 = 全部活动组」），
   所以扫描器**不判对错**、**不得**把它当门槛：`C1 = 0` 只意味着「这套词表 + 距离几乎没命中」，
   **不等于**「不存在来源断言」—— 证据是 `32603f8` 那版 5 词词表对 3/4 条自然措辞的来源断言是瞎的
   （读数见 `.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r5.md`
   的「唯一拦项」一节；本节第二个 heredoc 会把同样的 4 条注入重跑一遍）。

   **强制负向对照**（改这条规则的词表/距离时**必须重跑并归档**；注入只发生在 `/tmp` 副本里）：

   ```bash
   cd <worktree 根>
   python3 - <<'PY'
   #!/usr/bin/env python3
   """M60 §5.6 扫描器的**负向对照**：往 README 的 /tmp 副本注入自然措辞的来源断言，
   核「该抬的抬起来」，并把「抬不起来的」如实打出来（覆盖边界）。"""
   import pathlib, re, sys
   QL = r'全部|所有|唯一|任何|无一|不写死|一律|均|只|仅|每'
   SRC = r'派生|来源|来自|出处|取自|产出|生成|结论|计数|数字|读数|统计|标注|引用|标尺'
   C1 = re.compile(rf'({QL}).{{0,40}}({SRC})|({SRC}).{{0,40}}({QL})')
   MUST = {
    "A「本文件**全部结论均**由提取器当场派生。」": "本文件**全部结论均**由提取器当场派生。",
    "B「本文件**唯一的非派生**引用是 §5⑤ 的标尺。」": "本文件**唯一的非派生**引用是 §5⑤ 的标尺。",
    "C「本文件**所有**结论均来自提取器。」": "本文件**所有**结论均来自提取器。",
    "D「本文件的**计数与结论**均由提取器派生。」": "本文件的**计数与结论**均由提取器派生。",
   }
   LIMIT = {   # 已知抬不起来的（覆盖边界，如实记录）
    "E「本文件的结论都由提取器生成。」（`都` 不在词表）": "本文件的结论都由提取器生成。",
    "F「§5⑤ 是唯一的例外。」（无来源语汇）": "§5⑤ 是唯一的例外。",
   }
   src = pathlib.Path("/tmp/m60_scan/README.md").read_text(encoding="utf-8")
   ok = True
   print("## 方向一：该抬起来的必须抬起来（注入到 `README.md` 的 /tmp 副本）")
   print()
   print("| 注入措辞 | 期望 | 实测 | 判定 |")
   print("|---|---|---|---|")
   for k, v in MUST.items():
       p = pathlib.Path(f"/tmp/m60_scan/inj_{k[0]}.md"); p.write_text(src + "\n\n" + v + "\n", encoding="utf-8")
       got = any(C1.search(l) for l in open(p, encoding="utf-8"))
       ok &= got
       print(f"| {k} | 抬进 C1 | {'抬进 C1' if got else '**漏掉**'} | {'符合' if got else '**不符合**'} |")
   print()
   print("## 方向二：已知抬不起来的（覆盖边界；扫描器**不是**判定器）")
   print()
   print("| 措辞 | 原因 | 实测 |")
   print("|---|---|---|")
   for k, v in LIMIT.items():
       got = bool(C1.search(v))
       print(f"| {k} | 见左侧说明 | {'抬进 C1' if got else '**漏掉（已披露）**'} |")
   print()
   print(f"⇒ 负向对照结论：**{sum(1 for v in MUST.values() if C1.search(v))}/{len(MUST)} 条该抬的抬起来了**"
         f"，另 {len(LIMIT)} 条已披露的盲区仍漏 —— 因此本扫描只作**报告项**，不作为交卷门槛。")
   sys.exit(0 if ok else 1)
   PY
   ```

   实测（归档 `evidence/m60_scan_negcontrol.log`）：**4/4 该抬的抬起来了**
   （A `全部结论均` / B `唯一的非派生` / C `所有` / D `计数与结论`），**E/F 两条盲区如实漏掉** ——
   即「扫描器是活的，但它的限度是写明的」。

   **历史（同一形态的 4 个 commit；按新规引 commit 而不引「第 N 轮」）**：
   `13d601f`（`check_ref.py` 的 `[coverage]` 打印与事实相反，修于 `6a5bfa0`）→
   `5472bd8`（`gen_witness_log.py` 标题手写「5 个 case 全 PASS」，修于 `ce6822d`）→
   `ce6822d`（README 自称「唯一非派生的只有 §5⑤」，修于 `32603f8`）→
   `32603f8`（把扫描的「C1 = 0」写成通过门槛，修于 `77897d3` 的诚实降级：降为报告项 +
   写清盲区 + 强制负向对照 + 补源码进范围）。**交卷门槛仍是本节的规则 1–4（人工逐句核）**，
   扫描只是辅助发现手段。

> 工具化建议（M60 提交给 `tools/check_symbol_claims.py` 那类「把事实并排摆出来」的工具）：
> 加一条**对文档/打印文本的「量词 ∧ 来源语汇」扫描** —— 命中时打印**上下文 + 作者应引用的证据行**，
> 并**同时**打印「本扫描看不见什么」（词表外的自然措辞、纯符号表述）；**只并排事实、不判对错**，
> 也**不要**把「0 命中」写成通过条件。

### 5.7 评审/轮次引用一律「可解引用」—— 扫描范围与已知残留（M79 起）

**规则**（与 `m22_router512/README.md` §13.3 同族）：写「某版曾经如何 / 是谁在哪次评审里发现的」
时，引用必须能被**独立定位** —— **commit sha** 与**完整相对路径的评审工件名**都**可解引用**；
**裸的轮次叙述**（「第 N 轮」「rN」「round-N」，前后无 commit / 工件指针）**不可解引用**。
**归档文件的正文例外**：它记录的是**那一刻的事实**，按归档规则不改（下表类 4）。

**自查命令**（可复算；本节**不写聚合命中数** —— 它是活值，`.md` 一改就漂）：

```bash
cd <worktree 根>
python3 - <<'PY'
#!/usr/bin/env python3
"""M79 §5.7 的「轮次标签可解引用」扫描（**报告项**，不是判定器：判类要按匹配片段看 ±30 字上下文）"""
import os, re
PAT = re.compile(
    r'第[ 　]*[0-9一二三四五六七八九十]+[ 　]*轮'   # ① 第 N 轮（容许半角/全角空格 + 中文数字）
    r'|\br[0-9]+\b'                                  # ② 裸 rN
    r'|[Rr]ound[ 　]*[ _-]?[ 　]*[0-9]+'             # ③ round-N / round N / RoundN
    r'|末节'                                          # ④ 末节
    r'|\bP[0-9]+-[0-9]+\b'                           # ⑤ 优先级项标签 PN-M
    r'|\bR[0-9]+\b')                                 # ⑥ 大写 RN
EXT = ('.md', '.py', '.h', '.asc', '.txt', '.log', '.sh')
n_files = 0
for dp, _, fns in os.walk("m17_moe_real"):
    for fn in sorted(fns):
        if not (fn.endswith(EXT) or fn == '.gitignore'):
            continue
        f = os.path.join(dp, fn)
        n_files += 1
        for i, line in enumerate(open(f, encoding='utf-8', errors='replace').read().splitlines(), 1):
            for m in PAT.finditer(line):
                s = max(0, m.start()-30); e = min(len(line), m.end()+30)
                print(f"{f}:{i}: [{m.group(0)}] …{line[s:e]}…")
print(f"# 扫描文件数：{n_files}")
PY
```

（本节的扫描范围比 §5.6 第 5 条的量词扫描宽 —— 那条是 **12 个文件** = 3 个源码 + 9 个
`evidence/*.log`；本节走遍本目录，把 `.asc`/`.h`/`.txt`/manifest 也纳入，故打印的文件数不同。）

**归类按「匹配片段」而不是按行**：一行里可能同时躺着「允许的文件名引用」与「该改的裸标签」
（`m22` §13.3 记过这个坑）。判类动作：对**每个匹配**看它所在的 token 上下文（±30 字），
而不是看整行里有没有别的允许项。

**下表是在写本节之前的树上**跑上面那条命令的实测归类（行号随 `.md` 改动漂移，以「文件 + 匹配
片段 + 上下文」为准）：

| 类 | 含义 / 处置 | 实测命中（按片段） |
|---|---|---|
| 1 | **完整相对路径的评审工件名**（可解引用；在**主检出**（`.tower/` 所在的那棵树）`test -f` rc=0 —— `.tower/` 未入库，`.tower/worktrees/wt-NN` 这类 checkout 里没有它） | `README.md` :492/:493/:494/:506/:598；`gen_witness_log.py` :277/:278；`evidence/m60_extractor_selftest.log` :5；`evidence/m60_quant_rule_witness.log` :159 |
| 2 | **标识符**（局部变量名，与轮次无关） | `check_ref.py` :1005–:1007/:1014/:1015（`r32`）、:1088/:1095（`r6`） |
| 3 | **fenced code block 里的正则字面量**（模式字符串，不是叙述） | `README.md` :540/:541（§5.6 扫描器的 `C3` 模式） |
| 4 | **归档正文**（当时捕获的原文，按归档规则不改 —— 逐条见下） | `evidence/accept_run_m1_m64.log` :1（`round-2`）、`evidence/m60_quant_rule_witness.log` :160 |
| 5 | **裸标签** ⇒ 这一类就是漏网，改成引 commit / 完整相对路径 | 本节**不给清单**：这一类的成员正是要动掉的东西 |

**本节自身会被扫到**（与 §5.6 扫描器的「含本节自身的引用语境」同形）：上表的类 1–4 例子与下面
「已知残留」段**逐字引用**了那些片段，所以在**写完之后**重跑上面的命令，`README.md` 上除类 1 的
完整工件名与类 3 的正则外，还会多出这一批**引用语境**的命中 —— 判类时按同一条规则读它们的
±30 字上下文即可（它们是被引用的原文，不是新的裸标签）。

**复核动作**：跑命令 → 把输出逐片段与类 1–3 对照 → 剩下的逐个判「是不是类 4 归档正文」→
不是的按类 5 改。改完重跑，重复到落不进类 5 为止。

**已知残留（类 4 归档正文，按归档规则不改；在此登记以免被后来的扫描当成漏网）**：

- `m17_moe_real/evidence/accept_run_m1_m64.log:1` —— 抬头 `…（M29 / agent-moereal，round-2 修复后）`。
  该文件是 **M29 时期的 device 验收日志**（§8 表里按冻结归档处理），正文是当时那次运行的记述；
  `round-2` 是那次运行的名称，不是可解的评审指针。
- `m17_moe_real/evidence/m60_quant_rule_witness.log:160` —— `…-reviewer-m60-r2.md` 的**省略式回声**。
  该文件是 `gen_witness_log.py` 的**派生输出**（§5.6 扫描器把它当活文本，计入 C1–C3 栏）。
  **M79 只改生成器、未重生成归档**（`evidence/**` 的数据文件不在本 mission 的改动范围），
  故归档保留 M60 生成时的形态。**可复算的差异**（下面两条命令在本 commit 上跑过）：

  ```bash
  python3 m17_moe_real/gen_witness_log.py > /tmp/m79_gen.log
  diff /tmp/m79_gen.log m17_moe_real/evidence/m60_quant_rule_witness.log
  ```

  实测差异恰 **2 行**：`:160` = 本行（归档仍是 `…-reviewer-m60-r2.md`，
  生成器已是完整相对路径）；`:224` = §5⑥ 的 `git diff --name-only main...HEAD` 路径清单 ——
  **它本来就随分支而变**（归档是在 M60 分支上生成的，那份清单记着当时的 9 个路径），所以在任何
  commit 上重跑生成器都不会与归档逐字节相同 ⇒ 本文件只能按**当时快照**读。

**M79 在 m17 侧改掉的**：`gen_witness_log.py` §4 里那处**省略式工件名**（`…-reviewer-m60-r2.md`）
→ 完整相对路径（与上一行 r1 的写法、README `:506`/`:598` 同形；在主检出 `test -f` rc=0）。

**本扫描看不见什么**（已实测，必须知情）：① **计数 / 自指式表述**不在上面六种标签形态内 ——
实测命中 `README.md` :342「历史轮次」、:344「两轮评审」、:487「三轮评审」，以及 §1/§5.4/§5.5/§7
等节里的「本轮…」（自指本 mission 的改动；行号不列，用 `grep -nE '本轮' m17_moe_real/README.md`
自取当时的行号 —— 本节自己的文字里也含「轮次 / 轮」字样，那是自指，不在此列）。逐片段判读：
:342 的「历史轮次」指**历次验收运行**（该行自己写明「措辞与该日志那一行的括注对齐」，指的是冻结
日志的 `########## 确定性复核：…` 那行，不是评审轮次）；:344/:487 的「两轮/三轮评审」在**同一段里
给出 commit / 完整路径工件名**（`a875204`/`5472bd8`/`53730e9`/`32603f8` 与 :492–:494 的三行）；
「本轮」是**自指**（本 mission 自己的改动）。这三类都不构成「裸的评审轮次指针」，且改动它们会
连带作废 §5.6 与 §8 里与冻结日志对齐的声明（:342 那句就写着「随 M69 订正」）⇒ **只登记、不动**。
② **不在六形态里的轮次措辞**（例如「上次评审说…」）⇒ 漏掉。

## 6. 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <repo>

# 1) 生成权重 manifest（从 checkpoint 分片头取 offset/bytes + Python 独立 sha256；~3.5s）
/usr/local/python3.12.13/bin/python3 m17_moe_real/gen_weight_manifest.py --layer 0
#    （--check 只校验已生成的 manifest 与 checkpoint 现况是否一致）

# 2) 构建（并行度别拉满：cgroup 内存上限 32GB）
cmake -B m17_moe_real/build -S m17_moe_real -DCMAKE_BUILD_TYPE=Release
cmake --build m17_moe_real/build -j2

# 3) 跑 device（真实权重装载 + 字节核对 + 段序 S1-S10 + dump；每档 ~20-40s）
./m17_moe_real/build/m17_moe_real m17_moe_real/m17_weight_manifest.txt --m 1 --token 1000 --dumpdir /tmp/m17_dump
./m17_moe_real/build/m17_moe_real m17_moe_real/m17_weight_manifest.txt --m 64 --token 1000 --dumpdir /tmp/m17_dump

# 4) numpy/double 独立参考链 + 端到端链交叉校验（64 条判定项 @m=1）
/usr/local/python3.12.13/bin/python3 m17_moe_real/check_ref.py /tmp/m17_dump layer0_tok1000_m1
```

CLI：`--layer`（覆盖 manifest 的层）`--token`（embedding 行号，默认 1000）`--m`（1..64，
**越界直接拒绝**）`--stage`/`--sub`（段序/段内截断，bring-up 用，默认 8/9）`--dumpdir`
`--no-dump` `--no-probe` `--no-bytecheck`（跳过 1.24 GiB 的 D2H 字节核对，约省一半时间）。

## 7. 已知限制与后续

1. **x 与 gamma 的语义替身**（§3）：x 是真实 embedding 行，不是 post-attention-norm 激活；
   gamma 取 hyper-connection mixer 的真实 norm 权重行。**权重侧是 100% 真实**，
   激活侧的"真实推理数值"要等 attention 段接上（m15 的层循环骨架是现成入口）。
2. **m ≤ 64**：单 `BASE_M` tile 包络（`t_e ≤ 64`）。prefill（m=4097 档）需要 m-tile
   分段 + 段序重复，docs/12 §8 列为遗留，本 mission 不含。
3. **E=512 的槽位表放大**：workspace 301MB，其中 512 个槽位的 `aq/as/gu/hq/hs/y` 占绝大多数；
   m=1 时只有 10 个槽位非空。真实推理引擎会按"命中专家"做动态槽位分配或专家并行，
   本 mission 保持 m13 的静态 `[slot][M_MAX][row]` 契约以便逐 op 对齐。
4. **全量权重常驻 device**：1.24 GiB/层 × 48 层 = 60 GiB packed，48 层端口需
   权重流式换入/换出（或专家并行），本 mission 只做单层。
5. **性能未优化、未测量（且本 mission 的 host 墙钟不作为证据）**：跨段预取、A/B 双缓冲重叠
   （docs/12 §4 预留的 BufferID 7-10、L1 @330KB 窗、AIV 19-23）都没实现。
   **测量口径（tower 规则，12:45）**：host 侧"N 次连发 + 一次 sync 的墙钟"在共享 NPU 上
   **不构成证据**（实测同类测量有 24× 抖动；本目录自己的 9 次运行也落在 6–35 ms）。
   `evidence/accept_run_m1_m64.log` 里每个 case 都印了原值；日志末尾的「host 墙钟抖动块」是
   **同一二进制、同一命令连续 5 次**的实测：**1.180 / 5.284 / 5.319 / 12.594 / 18.564 ms**
   （min→max **15.7×**，静场 1.18 ms vs 拥塞 18.6 ms），而 5 次的落盘 sha256 完全相同 ——
   即"抖动与被测代码无关"在本目录也成立。这些数字**只作定性**，任何"变快/变慢"的结论都必须
   先按该规则排除测量噪声（或改用设备侧 `msprof` 口径）。本 mission 的定位是正确性，
   **不作任何性能断言**。
6. **激活量化规范跨 donor 不一致**（同 m13 §5.3 遗留）：激活走硬件 floor-指数规则、
   权重走 checkpoint 的 ceil 规则；本次真实数据上 38–40% 的字节差异已量化报告，
   若要"量化感知 golden"逐位对齐需先裁定激活侧规范。
7. **诊断槽语义**：`offsets[E+8]` 的 4B 槽是 IG 越界计数（0 = 正常），它与 `offsets[E+1]`
   数组在同一个 GM 段内（m13 原本写 `offs[8]` 会覆盖真值，本实现已挪开）。
8. **`check_ref.py` 依赖 `tools/golden/moe_block_ref.py` 的范围**：只 import 其 **MXFP4 编解码**
   （`E2M1_POS` / `e2m1_decode` / `e8m0_decode` / `f32_to_bf16_bits` / `bf16_bits_to_f32` /
   `group_scale_exp`）作为权威定义，外加 M60 起 import **`quantize_ocp`** 作为 §5.5 的
   **第二转写**（同源转录，不是"独立来源"——见 §5.5.1 的强度自评）；**路由语义是 `check_ref.py`
   内联重写的**（softmax/top-k/计数排序/折叠），不 import。**不消费** tools/golden 的任何数据集
   （`quantize_ocp` 只作函数调用，不读它的 golden bin）。e2m1/e8m0 语义另有 §4 的 device 解包探针
   作为独立见证。
9. **`Reg::Exp` / NR-rsqrt 的精度未独立标定（T3 界里唯一的假设项）**：`EPS_FAST = 2·2⁻⁸`
   是按 "device 的 Exp/NR 在 bf16 域快速路径上工作" 建模的相对误差上界，**不是**从指令手册
   推导的。它只在两处成立：§5.4 的 e2e 链与 device 逐位相同（说明真实误差远小于该界）；
   以及各段"最差占预算"实测 0.00–1.000 —— 其中 S9a `routed` 的界已用到 **0.999（m=1/8/33/
   换层）与 1.000（m=64）**，即**零余量**：说明该段的界与实测同量级而非宽松。若后续要收紧
   判决，需要一条**独立的 `Reg::Exp` 探针**（扫 exp 输入区间、与 double exp 比）把 `EPS_FAST`
   换成标定值，并复核 `EPS_FOLD`。**当前判据的边界风险（必须知情）**：`routed` 的界在 m=64
   处正好被用到 1.000，**换 token / 换层 / 微调参考序列都可能把它翻成 FAIL** —— 这是本分支
   唯一有非文档后果的余量问题。按 `docs/17` §1 的护栏，**不允许用放宽界的方式消除该风险**；
   正确做法是先做上面的探针标定（已列入 probe backlog），必要时再重新推导 `EPS_FOLD` 的
   ε（unpermute 的 10 项乘加 + 收尾，现取 `2·11·2⁻²⁴`）。
10. **±Inf/NaN 行为：device 侧仍未被真实数据触发；host 镜像侧 M60 已补注入用例并修掉一处分叉**：
   第 10 行（§1）已把 m13/m5 落地的 `NAN_CUSTOMIZATION = 0x7f81` + `Select` 平价修复搬进 m17 的
   `VecQuantStage`；本轮所有**真实**数据都是有限的，故 **device 侧该路径仍未被实际触发**
   （m5 有自己的 `InjectInf` 三类用例，m17 没有 —— 若要求"真实权重 + 非有限激活"的 device 行为
   见证，仍需在 host 侧做一次注入用例并重跑 device）。
   M60 做的是**镜像侧**的注入：§5.5 的 `W3/W3n` 用合成 ±Inf/NaN 与次正规组把 `quant_hw` 与
   `tools/golden::quantize_ocp` 逐字节对照，并咬出镜像缺的两条官方覆盖（±Inf 组 scale `253`/code `7`
   → 修成 `255`/`0`；指数域 < 2 的组 code `4` → `0`）。**真实数据不触发**这两个角落（0 个这样的组），
   所以既有 63/240/534/705 条判定项逐行不变；"device 侧角落行为"仍**只有 m13/m5 的 device 用例**见证，
   不是本目录的（这条限度必须保留在案）。
11. **e2e 链的 `w_tk` 字节项未加**：`e2e_chain` 算了 `wts` 但比对项里没加"链上 `w_tk` 字节
   == device"（值本身已由 §5.2 的 `topk_weights` 判据 + 链上 `ids`/`counts` 逐位见证）。
   若要彻底去环可补一条，代价是判定项计数 +1。**注意**：判定项计数已因 M60 变成
   **74/251/545/716**（+11 条 §5.5 见证），tower 早前预期的 63/240/534/705 是 M29 的数；
   若再补 `w_tk` 那条，README 的表与 `evidence/m60_check_ref_run.log` 需要同步再生。
12. **`perm_expert` 无 kernel 内消费者**、`UB_ZEROS_B16`/`UB_CB_OUT` 保留未用 —— 动机见 §1
    表末的两条注（前者服务 host 判据与 `check_ref.py`，后者保持 m13 的 UB 平移关系）。
13. **vector-API 标准（tower 11:52 裁决 + M44 四条裁定）在本目录的状态 —— "先报不改"**：
    * 白名单内的两处均**已按要求附注释写明依据**：`Sort32`（`m17_moe_layer.asc` S2 段）
      与 `MrgSort`（`Merge2`）—— 依据 = M44 裁定 ②（排序单元 primitive；CANN 9.1.0
      无 `Reg::` 等价物、官方 donor 同为 memory-based）。`MrgSort4` 未使用（dav-3510
      deprecated 空函数体）。
    * **不在白名单、需要改造的既有形态**：S2 里的经典 `Extract`（从 m7 拷贝）——
      已就地加 ⚠ 注释。同类问题还有本目录从 m5/m13 拷来的 S5/S7 量化与 S9b combine
      路径（属"已有的模块先报不改"，等 tower 的 vector-API 审计排序）。
    * 本轮**不动**这些计算路径：既因 M44 明确"已有模块先报不改、不要自己冲进去改"，
      也因本 mission 的验收基线（dump sha256 `1a0f4883…`）建立在当前二进制上 ——
      改造应另立 mission 并重跑全套验收。

## 8. 文件

| 文件 | 说明 |
|---|---|
| `m17_moe_layer.asc` | 单一 `__mix__(1,2)` kernel（S1–S10 + 同步）+ 解包探针 kernel（`__cube__`）+ host（manifest pread、字节核对、落盘、不变量判据） |
| `m17_resources.h` | 真实形状资源表（形状常量 / BufferID / flagId / UB / L1 / GM workspace 32 段） |
| `gen_weight_manifest.py` | 从 checkpoint 分片头生成 `m17_weight_manifest.txt`（offset/bytes/shape/sha256） |
| `m17_weight_manifest.txt` | layer 0 的 16 条张量记录（含 Python 独立 sha256），host 直接消费 |
| `check_ref.py` | numpy/double 独立参考链（S1–S10）+ §5.4 的**不打断端到端链**（`e2e_chain`，从真实 x 起步）+ §5.5 的**量化规则对外 pin 与见证**（`W0`–`W6` + 5 条负向对照：定点方向 / 角落平价 / 设备交叉字节 / 设备侧方向统计 / 值域往返）+ 分段分位表判据（T1/T3/T4 分档）+ 量化器逐字节 + 参考项 |
| `gen_witness_log.py` | §5.5 证据日志的**提取器**（`5472bd8` 入库、`ce6822d` 补三态、`53730e9` 逐块标注来源）：逐条读数由正则从输入日志取出，条数/`PASS/FAIL` 计数/§3 判定项计数与 FAIL 名/§5④ 的逐行比对/§5⑥ 改动范围与计时项计数**当场算**；**§4 的角落读数逐字取自 §2 的 `自检-W3n` 行**。输出头部有一行**「来源逐块标注」**，把 §0–§5⑥ 逐块写明哪些是派生、哪些是引用/说明（引用项逐条注明出处）—— 不以「全部/唯一」这类量词概括。三态退出码 `0`（有读数且全 PASS）/`1`（有读数但含 FAIL）/`2`（0 条读数或缺文件 → `RESULT: SKIPPED`，**绝不发合格证**）；`--selftest` 用六种合成输入自检（正常 / 0 case / 含 FAIL / 缺文件 / 只有非判定行 / 分隔符不可解析）。用法见其 docstring |
| `CMakeLists.txt` | 独立 CMake 工程（`find_package(ASC)` + `--npu-arch=dav-3510`，`-ffp-contract=off`） |
| `evidence/` | 验收证据（见 §5.3 末的证据归档表） |
