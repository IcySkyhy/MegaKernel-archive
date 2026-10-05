# docs/22 —— 预填充 prolog / epilog 段间接线设计（M143）

> **性质与边界（先读）**
> - 本文是**设计文档**：M143 只写本文件，不改任何代码、不跑设备。文中所有"计划 / 应 / 建议"都是**设计意图**，
>   不是既成事实；所有"现有 / 已 / 实测"都在文末给出 `文件:行`，并标出取证范围。
> - 事实基线：`main` @ `d316d5b`（wt-143 的 base）。**M140 分支 `feat/m140-whole-layer-gdn-prefill-hc-fields-a`
>   @ `852dfbc` 未合入 main**（`git merge-base --is-ancestor 852dfbc HEAD` 在本 worktree 返回非 0），
>   M140 复审钉死的两条未接链路与 15 个 hc 实参、`M15_PREFILL_PHASES` 都在那条分支上；本文凡涉 M140 处均显式标注。
> - 纪律：不写绝对断言（禁用"已全部 / 无残留 / 0 命中 / 绝不"类措辞）；未取到的读数记「未读到 / 未知」；
>   所有代数只作定位辅助，**符号名与契约注释才是锚点**。
> - 人类约束（M143 Context 逐字，作为设计输入）：①「我们先把 1 个层的总体功能打通了再细扣性能」；
>   ②「E=512/topk=10，打通阶段也需要使用真实 shape」；③「权重就是要常驻，只有 ngram 权重在 host」；
>   ④「矩阵乘法 ⇒ 必须 cube 做，其余都用 VF 做，不要用 scalar 做计算，scalar 只做控制流」。

---

## 0. 速览

预填充入口（`m15_layer_kernel_gdn_prefill`）当前走 `M15L_PrefillBody`，四相位里**只有相位 A 挂了段体**；
两条段间链路在预填充路上没有段体：

| # | 链路 | 需要的段 | 现有可复用资产 | 缺口（本设计要接线的东西） |
| - | ---- | -------- | -------------- | -------------------------- |
| ① | `hc(attn) BLK` → 相位 A 输入 | in_proj（S2）+ conv1d/l2norm/gating（S3） | `m11_bf16_gemm`（in_proj 形状）、`m9_gdn_prolog`（conv1d+l2norm+gating）；decode 里由 `m15_gdn_layer.h` 的 `GdnLayerChain` S2/S3 组装 | 两者都只有 **decode m=1** 形态；预填充 m>1 的 prolog 段体在本仓**未读到**；z 平面、q/k 补零行无生产者 |
| ② | 相位 A 出口 `wsO` → H2 的 `bo`（`hcAttnOut`） | RMSNormGated（S5）+ out_proj（S6）+ 残差/门控（S7） | `m12_rmsnorm_gated`、`m11_bf16_gemm`（out_proj 形状）、`M15OP::OProjGemmAic`、`M15H::HyperConnOp::CombineStage` | `wsO`（head-major fp32）与 `m12` 期望的 token-major fp32 不同布局；z 平面缺失；预填充路上无 out_proj/RMSNormGated 启动点；`hcAttnOut` 在预填充路无生产者（M138 已登记） |

**关键拓扑澄清（本文沿用，后文不再重复）**：融合 kernel 的四相位是
`H1(hc 边界 #1) → A(子层段) → H2(hc 边界 #2) → B(MoE)`（`m15_layer_kernel.h:18-21`）。
`BLK` 是 **hc 边界的 block-output = 子层段的输入**，不是 attention 的输出；`hcAttnOut` 是**子层段出口**
（= H2 的 `bo`），不是 hc 边界的产出。两者方向相反，别混。

---

## 1. 两个未接链路在融合 kernel 里的确切位置

### 1.1 两条入口路

- **预填充路**：`m15_layer_kernel_gdn_prefill` / `m15_layer_kernel_attn_prefill`
  （`m15_layer_kernel.h:1312` / `:1319`）→ `M15L_PrefillBody<KIND>`（`:816`）。
- **decode 四相位路**：`m15_layer_kernel_gdn_hc` / `m15_layer_kernel_attn_hc`
  （`m15_layer_kernel.h:1341` / `:1348`）→ `M15L_FusedBody<KIND, HC=true>`（`:914`）。

`M15L_PrefillBody` 的四相位骨架（`m15_layer_kernel.h:830-852`）：
`H1`（`:831-833`）→ 相位边界 `FLAG_HC0_BOUND_AIV`（`:835`）→ **相位 A**（`:837-839`）→
相位边界 `FLAG_HC1_BOUND_AIV`（`:841`）→ `H2`（`:843-845`）→ 相位边界 `FLAG_HC2_BOUND_AIV`（`:847`）→ `B`（`:849-851`）。

### 1.2 相位 A 的现状：只挂 B1 扫描段

`M15L_PrefillPhaseA<KIND>`（`m15_layer_kernel.h:686-740`）在 `pfWired != 0` 时：

- `KIND_GDN`：组装 `M15GP::GdnPrefillArgs gp`，把 `A.wsQ/wsK/wsV/wsG/wsBeta` 直接喂给扫描段
  （`m15_layer_kernel.h:703-728`），出口写 `gp.out = A.wsO`（`:710`）。
- `KIND_ATTN`：B2（前端）/ B3（稠密 core）**本轮不接**，该支不产出（`:731-733`）。

也就是说，**相位 A 在预填充路上只做 GDN chunk 扫描**（B1），它吃的是"已过 prolog 的 q/k/v/g/β"，
产的是扫描输出 `o`（`wsO`）。prolog 与 epilog 都**不在**这条路上——这正是两条链路的所在。

### 1.3 对照：decode 四相位路的 GDN 子层段是完整的

`M15L_FusedBody` 在 `KIND_GDN` 且 `HC=true` 时调用的是 `GdnLayerChain`（`m15_layer_kernel.h:949-969`），
即 **S1–S7 全链**（段序见 `m15_gdn_layer.h:5-13`），子层入口/出口分别是
`subIn = A.hcWs0 + M15H::WS_BLK`、`subOut = A.hcAttnOut`（`m15_layer_kernel.h:945-946`）。
即：**同一套段序在 decode 路已组装过，但只在 m=1**（`m15_gdn_layer.h:5/38/1833`，`m15_gdn_resources.h:70`
`CHAIN_M = 1`）。预填充路要复用的，是这条段序的**语义**，而不是它的 m=1 形态。

### 1.4 判定：两条链路缺的都是"段体 + 挂载点"，不是"重写数据流"

- 链路①缺：`H1 → 相位 A` 之间的 prolog 挂载点与段体（把 BLK 变成 q/k/v/g/β）。
- 链路②缺：`相位 A → H2` 之间的 epilog 挂载点与段体（把 `wsO` 变成 `hcAttnOut`）。

---

## 2. 链路①：prolog 段盘点

### 2.1 需要哪些段（按 `GdnLayerChain` 的 S 编号）

`m15_gdn_layer.h:5-13` 给出 GDN 子层的段序（decode m=1）：

| 段 | 内容 | 本链路相关？ |
| -- | ---- | ------------ |
| S1 | Add+RMSNorm#1：`x + res → x_norm` + fp32 `res1` | 归属见 §8-U4（HC 形态下 BLK 与 hc mix 的关系未定） |
| **S2** | **in_proj bf16 GEMM：K=2560, N=16480（权重 [16480,2560]）→ qkvzba** | **prolog 第一段** |
| **S3** | **conv1d(K=4)+bias+SiLU → q/k l2norm → gating(g/β)** | **prolog 第二段** |
| S4 | recurrence（chunk 扫描）| 相位 A 的 B1 已实现 |
| S5 | RMSNormGated | 见 §3（epilog） |
| S6 | out_proj bf16 GEMM：K=6144, N=2560 | 见 §3（epilog） |
| S7 | Add+RMSNorm#2 | 见 §3（残差/门控） |

任务口径里的 prolog = "in_proj + conv1d + l2norm"，对应 **S2 + S3**。

### 2.2 现有 kernel 可用性

| 段 | kernel / 段体 | 入口名 | 输入 plane（shape / stride / dtype） | 输出 plane | 出处 |
| -- | ------------- | ------ | ------------------------------------ | ---------- | ---- |
| S2 in_proj | `m11_bf16_gemm/m11_bf16_gemm.asc` | `bf16_gemm_kernel<K=2560, N=16480>` | A `[m, K=2560]` bf16 行主序（`srcDValue=K`）；B=W `[N=16480, K=2560]` bf16（B^T 消费） | C `[m, 16480]` bf16（F322BF16） | 入口 `:265`；类 `:84`；常量 `:46-49`（`BASE_M=64`、`BASE_K=64`、`BASE_N=160`）；实例 `:362-366`；量化 `:250` |
| S2 同形 | `m15_attn_oproj.h`（attention 用，形状同款 K=6144/N=2560） | `M15OP::OProjGemmAic` / `OProjGemm6144x2560` | A `t` bf16 `[m,6144]`；B `w` bf16 `[2560,6144]` | C `y` bf16 `[m,2560]` | 常量 `:50-54`；`OProjGemmAic :437`；`GateMulAiv :184` |
| S3 conv1d+l2norm+gating | `m9_gdn_prolog/m9_gdn_prolog.asc` | `gdn_prolog_kernel`（类 `GdnProlog`，`Init :253` / `Process :269`） | x bf16 宽 `INW=16480`（`q2048|k2048|v6144|z6144|b48|a48`）；`conv_state` bf16 planar `[3][10240]`；`w` bf16 `[4][10240]`；`bias` bf16 `[10240]`；`a_log`/`dt_bias` fp32 `[64]`（48 有效） | q/k fp32 `[16][128]`、v fp32 `[48][128]`、g/β fp32 `[48][8]`（slot-stride，仅 `[h][0]` 有效） | 入口 `:434-440`；常量 `:61-71`、`:256-266`；只吃 in_proj 之后的 x（`:8-9`） |
| S4 扫描（相位 A 现状） | `m15_layer_loop/m15_gdn_prefill.h` | `GdnPrefillAivEntry :1135` / `GdnPrefillAicEntry :1143` | 见 §5.2 | `out [48,m,128] fp32`（= `wsO`） | `GdnPrefillArgs :85-106` |
| decode 组装 | `m15_layer_loop/m15_gdn_layer.h` | `GdnLayerChain`（入口 `m15_gdn_layer_kernel :1809`） | 全链 decode m=1 | S7 

注：`m11` 的实例选择由 host 按 `(k,n)` 二分（`:362-366`），K=2560/N=16480 走 in_proj 实例，
K=6144/N=2560 走 out_proj 实例。

注：`m18_gdn_prefill`（入口 `m18_gdn_prefill_kernel :752`）是**扫描段**、**不含 prolog**
（其输入契约就是"已过 prolog 的 q/k/v/g/β"，`m23_gdn_prefill/check_ref.py:9-10`）；
它可作为扫描段的独立参考路，但不提供 in_proj/conv1d/l2norm 的段体。

### 2.3 缺口（链路①）

- **G1-a（m 包络）**：`m9` 是 **decode m=1** 核（文件头 `:2`；`m15_gdn_resources.h:70` `CHAIN_M=1`），
  conv1d/l2norm/gating（S3）m>1 的段体在本仓未读到。`m11`（S2/S6）的 M 方向是**运行时多 tile 循环**
  `mLoop = CeilDiv(mTotal, BASE_M)`（`m11_bf16_gemm.asc:102`、循环 `:120`、尾块 `:121`）；`BASE_M=64`
  （`:47`）是 **tile 边长**，不是单次启动的 M 上界。设备实测（M146）：m=4097 两形状 in_proj
  `K=2560,N=16480` 与 out_proj `K=6144,N=2560` **单次启动、不改代码**逐位一致
  （`m11_bf16_gemm/evidence/m4097_envelope/README.md`）；m=1/2/17/33/64 见 `m11_bf16_gemm/README.md:105-108`。
  设备已实测的 m 集合 = {1,2,17,33,64,4097}，更大 m 未逐个实测。预填充验收档 m=4097
  （`m15_gdn_resources.h:317` `GP_M_PREFILL=4097`）在 S2/S6 上已有单次启动的代码路径与设备读数。
- **G1-b（q/k 补零行）**：相位 A 按 `qkStride = m + PF_GDN_QK_PAD_ROWS`（= m+64）读 q/k
  （`m15_layer_kernel.h:721`、`:648`），要求 `[m, m+64)` 行**可读且置零**（`:643-648`）。
  prolog 若产出 q/k，必须同时产出这段置零的 pad 行。**当前无生产者**（相位 A 的 q/k 由 host 合成，
  `m15_layer_loop.asc:3315-3336` 先 `assign(nk*Tp*128, 0.0f)` 再写 `[0,m)` 行）。
- **G1-c（z 平面）**：S5（RMSNormGated）需要 `z`（`m15_gdn_resources.h:49` `Z_DIM=6144`，decode 里
  就是 in_proj 输出的 `[10240,16384)` 段，`m15_gdn_layer.h:11` 注明"S2→S5 全程存活"）。
  预填充的 `GdnPrefillArgs`（`m15_gdn_prefill.h:85-106`）只有 `q,k,v,g,beta`，**未读到 z 字段**；
  z 在预填充路上由谁持有、落在哪个平面，属未定项（§8-U3）。
- **G1-d（权重常驻）**：prolog 需要的 `wIn/convW/convBias/aLog/dtBias` 在 `LayerArgs` 里已是基础字段
  （`m15_layer_kernel.h:119-120`，FILL `:1096-1100`），但预填充 host **目前传 nil**
  （`m15_layer_loop.asc:3466` 连续 7 个 `nullptr`）。按人类约束③（权重常驻），接线时应传常驻 GM 指针，
  而不是逐层 H2D。
- **G1-e（cube/VF 划分）**：按人类约束④，S2 是矩阵乘法 ⇒ 走 cube（`m11` 的 `Nd2Nz → LoadData2D → Mmad`）；
  S3 的 conv1d/l2norm/gating 是逐元素/规约 ⇒ 走 VF（`m9` 全 AIV）。**设计上不引入 scalar 计算**。

---

## 3. 链路②：epilog 段盘点

### 3.1 需要哪些段

任务口径里的 epilog = "o_proj + RMSNormGated + 残差/门控"，对应段序是 **S5 + S6 + S7**。
注意**执行次序**与任务措辞的书写次序不同：`m15_gdn_layer.h:5-13` 的次序是
**S5 RMSNormGated → S6 out_proj → S7 Add+RMSNorm#2**（先按 value head 128 做 gated RMSNorm，
再 6144→2560 投影，最后加残差）。本文按代码次序叙述。

### 3.2 现有 kernel 可用性

| 段 | kernel / 段体 | 入口名 | 输入 plane（shape / stride / dtype） | 输出 plane | 出处 |
| -- | ------------- | ------ | ------------------------------------ | ---------- | ---- |
| S5 RMSNormGated | `m12_rmsnorm_gated/m12_rmsnorm_gated.asc` | `rmsnorm_gated_kernel`（类 `RmsNormGated`） | `o` fp32 `[m,6144]` 行主序；`z` bf16 `[m,6144]`；`gamma` bf16 `[128]` | `out` bf16 `[m,6144]` | 入口 `:557`；IO `:395-402`；常量 `:72-78`（`HEADS=48/HEAD=128/HIDDEN=6144`，`EPS=1e-6`）；语义 `:4-20` |
| S5 同形（decode 融合版） | `m15_gdn_layer.h` 的 `RmsNormGatedStage` | — | 同 m12，但按 `oGm[h*128]` 取址（**假定单 token 内 4 个头连续**） | bf16 `[HEADS, HEAD]` | `:1230`、取址 `:1259`。**m=1 专用**（见 §3.3） |
| S6 out_proj | `m11_bf16_gemm` | `bf16_gemm_kernel<K=6144, N=2560>` | A `[m,6144]` bf16；B=W `[2560,6144]` bf16 | C `[m,2560]` bf16 | 同 §2.2 |
| S6 同形 | `m15_attn_oproj.h` | `OProjGemmAic :437` | 同款 K=6144/N=2560 | 同 | `:433` |
| S7 残差/门控 | `m15_hc_layer.h`（lift 产物） | `M15H::HyperConnOp::CombineStage :653`（读 `bo`） | `bo` bf16 `[m,2560]`（row stride `HID=2560` elems = 5120 B） | 回写 HC 多流态 | `bo` 声明 `:415`，`boGm` `:441`，读 `boGm[mi*HID + jj*CHUNK]` `:676` |
| S7 同形（GDN 段自己的 add+norm） | `m15_gdn_layer.h`（`NormStage<true>`） | — | `out_proj` 输出 + S1 的 fp32 `res1` | `y_final` + `res2` | 声明 `:1794`、Run 分派 `:1735` |

**gating 的归属**：任务措辞里的"门控"落在 **S5 内**——`m12` 的公式是
`out = bf16(((o·rstd)·gamma)·sigmoid(z))`（`m12_rmsnorm_gated.asc:4-20`、`:491-498`），
即 sigmoid(z) 就是输出门。S7 的"残差"是把 block 输出加回残差流。

### 3.3 缺口（链路②）

- **G2-a（布局）**：`wsO` 是 **head-major** `[48, m, 128] fp32`（`m15_layer_kernel.h:263`；段体存储式
  `m15_gdn_prefill.h:1089-1091`：`oOff = hv*m*128 + t0*128`）。而 `m12` 的 `o`/`z` 是
  **token-major** `[m, 6144]` 行主序（`:398-401`）。m>1 时两者不是同一块内存布局 ⇒ 需要**转位/重排**（§4）。
- **G2-b（z）**：同 G1-c：预填充路未读到 z 平面；`m12` 的 sigmoid 门缺输入面。
- **G2-c（启动点）**：相位 A 的挂载点只产出 `wsO`（`m15_layer_kernel.h:710`）；预填充路上**未读到**
  RMSNormGated / out_proj 的启动点，H2 拿到的 `A.hcAttnOut`（`:800`）在预填充路无写入者。
- **G2-d（m 包络）**：decode 融合版 `RmsNormGatedStage` 的取址假定单 token 内 head 连续
  （`m15_gdn_layer.h:1259`），是 m=1 专用；`m12` 本体接运行期 `m`（`:395`），但需 token-major 输入。
- **G2-e（cube/VF 划分）**：S5/S7 走 VF；S6 矩阵乘法走 cube。

---

## 4. 平面落差与归属：`wsO` ↔ `hcAttnOut`

### 4.1 两侧几何

| 侧 | 字段/平面 | 声明 | m 维行距（token stride） | head/列 stride | dtype | 出处 |
| -- | --------- | ---- | ------------------------ | -------------- | ----- | ---- |
| 生产（相位 A 出口） | `wsO` | `[48, m, 128]` | head 内：`128` elems = 512 B | head stride = `m*128` elems = `m*512` B | fp32 | `m15_layer_kernel.h:263`；存储式 `m15_gdn_prefill.h:1089-1091` |
| 消费（H2 的 `bo`） | `hcAttnOut` | `[M_MAX, HID=2560]` | `HID*2` = 5120 B（`ROW_HID`） | 列向连续，无额外 padding | bf16 | `m15_layer_kernel.h:150`；`m15_hc_resources.h:40`；`m15_hc_prefill.h:114` |

`wsO` 的 `m` 是"每 head 内 token 维长度"，随运行期 `m` 变；`hcAttnOut` 的 m 是"行数"，行距恒定 5120 B。
两者**不同型**（fp32 vs bf16），且 `wsO` 的 head 维在最外层、`hcAttnOut` 的 hidden 维在最内层。

### 4.2 从 `wsO` 到 `hcAttnOut` 必须做的变换（设计意图）

按 S5→S6 段序，最小链路是：

1. **转位/重排（必须）**：`wsO` head-major `[48,m,128]` → token-major `[m,6144]`（列 c = `hv*128 + d`）。
   这是 S5 输入面要求的布局（`m12` 按 `[m,6144]` 取址）；m=1 时两者恰好重合（`hv*128` vs `t*128` 退化），
   所以 decode 的 m=1 段体不需要它，**m>1 才暴露**。此步是纯搬运，判据应按 T1 逐位（§7）。
2. **分组 RMSNorm + sigmoid(z) 门（必须）**：`m12_rmsnorm_gated`，输出 bf16 `[m,6144]`。
   输入 `o` 用步骤 1 的 token-major fp32（fp32→fp32，无转型），`z` bf16 `[m,6144]`（缺，见 G2-b），
   `gamma` bf16 `[128]`（= GDN `norm.weight`，切片角色 `linear_attn.norm.weight`，`slice_layer_manifest.py:63`）。
3. **out_proj（必须）**：`m11` K=6144/N=2560，A 用步骤 2 的 bf16 `[m,6144]`，B 用常驻权重
   `linear_attn.out_proj.weight` `[2560,6144]`（`slice_layer_manifest.py:59`），输出 bf16 `[m,2560]`。
4. **落位（必须）**：步骤 3 的输出行主序 `[m,2560]` 与 `hcAttnOut` 的行距 5120 B 一致 ⇒ 可直接写入
   `hcAttnOut`，或在中间平面产出后拷贝一次。**由 epilog 段拥有写权限**。

### 4.3 谁拥有暂存（设计意图）

- 需要在 GM 上新增的暂存：至少一块 **token-major fp32 `[m,6144]`**（步骤 1 的输出 / 步骤 2 的 `o` 输入），
  以及一块 **z bf16 `[m,6144]`**（G2-b）。
- 归属建议：由 epilog 段拥有这两块（读 `wsO`、写 token-major、写 z 的来源见 §8-U3），host 在
  `H_PfSegAlloc`（`m15_layer_loop.asc:1762-1815`）里分配。**本设计不改变 B1 的 `wsO` 契约**
  （改 B1 定尺会牵动 m23 的 dump/对拍契约，`m15_layer_loop.asc:3517`、`m23_gdn_prefill/check_ref.py:130-137`）。
- 另一条路（备选，需另议）：若让 B1 直接产 token-major，可省掉步骤 1，但会改 B1 的段契约与 m23 参考
  的 reshape 口径，属于口径变更，应显式登记并交复审。

### 4.4 M138 已登记缺口（引用）

- `m15_hc_prefill.h:72-76`：「边界 #2 的 `bo` 生产者缺口（M138 登记）……子层段 B1（`m15_gdn_prefill.h`）
  写的是 `LayerArgs.wsO`（`[48,m,128] fp32`），而挂载点给边界 #2 传的是 `A.hcAttnOut`（`[m,HID] bf16`）
  —— 两者不同型……未读到两者间的生产者关系」。
- `m27_hc_prefill/README.md:337-351`：「本 README §6 补丁 1 把 H2 的 `bo` 取 `A.hcAttnOut`」、
  「prefill 路径的 `KIND_GDN` 子层段是 B1」、「它的出口是 `A.wsO`」、
  「prefill 路径上未读到 `hcAttnOut` 的生产者」、「两个平面还不同型，不能直接别名」。
  **出错条件**：`pfWired != 0` ∧ `KIND_GDN` ∧ H2 的
  `hcMlpMode != MODE_MIX` 时，donor 的 combine 段会读 `bo`（`m15_hc_layer.h:463` `useCombine`、
  `:441`/`:676` 读 `boGm`）⇒ H2 读到未定义/陈旧值，且**不报错**（静默错值）。
- `docs/15-prefill-design.md:889`：N1「子层出口**没有生产者**」。

---

## 5. 契约：BLK 与相位 A 输入

### 5.1 attention 的 BLK（H1 输出）契约

- **声明**：`pfHcBlk0` / `pfHcBlk1` = 「边界 #1 / #2 的 BLK 平铺面 `[m, HID] bf16`」
  （`m15_layer_kernel.h:270-271`）。
- **行距与定尺**：`ROW_HID = HID*2 = 5120 B`（`m15_hc_prefill.h:114`），`SZ_BLK == M_MAX*ROW_HID`
  （`m15_hc_prefill.h:125`），host 端 `blkB = mMax*HID*2`（`m15_layer_loop.asc:1777`）。
- **语义**：`BLK = hc(attn) 的 block input → 子层段`（`m15_layer_kernel.h:945` 译文；
  相位图 `:18-19`）。它是**子层段的输入**，不是 attention 的输出。
- **生产者**：`M15L_HcPrefillBoundary(0u, A)`（`m15_layer_kernel.h:832`，实参 `blk = A.pfHcBlk0` `:805`），
  内部由 `M15H::HcPF::Body` 的 `RelocateTile` 把 arena 阻塞布局搬成平铺面
  （`m15_hc_prefill.h:395-402`）。
- **当前状态（预填充 bring-up）**：`m15_layer_loop.asc:3471` 硬编码 `pfStageMask = PF_STAGE_A`，
  H1 被截断 ⇒ BLK 平面被分配并毒成 `0xCD`（`:1803-1810`）但**未被写入**。
  即：BLK 契约的定义是清晰的，只是预填充档尚未跑 H1。

### 5.2 相位 A 期望的「已过 prolog 的 q/k/v/g/β」输入契约

段体契约见 `GdnPrefillArgs`（`m15_gdn_prefill.h:85-106`）与 `LayerArgs` 字段注释
（`m15_layer_kernel.h:258-264`）：

| 平面 | shape | stride / padding | dtype | 出处 |
| ---- | ----- | ---------------- | ----- | ---- |
| q | `[NK=16, m+64, 128]` | 每 head 行 stride = `qkStride = m+64`；`[m, m+64)` 行**可读且置零** | fp32 | `m15_layer_kernel.h:258`、`:721`；`m15_gdn_prefill.h:104` |
| k | `[NK=16, m+64, 128]` | 同 q | fp32 | `m15_layer_kernel.h:259` |
| v | `[48, m, 128]` | 无 pad | fp32 | `m15_layer_kernel.h:260` |
| g | `[48, align8(m)]` | 行对齐到 8（`tp = ((m+7)/8)*8`） | fp32 | `m15_layer_kernel.h:261`、`:717` |
| β | `[48, align8(m)]` | 同 g | fp32 | `m15_layer_kernel.h:262` |
| o（出口） | `[48, m, 128]` | 无 pad | fp32 | `m15_layer_kernel.h:263` |
| h0 / ht | `[48, 128, 128]` | in-place 复用 `ssmState` | fp32 | `m15_layer_kernel.h:256-257`、`:709-711` |

### 5.3 `PF_GDN_QK_PAD_ROWS` 与扫描段对 padding 的要求

- 常量：`PF_GDN_QK_PAD_ROWS = 64u`（`m15_layer_kernel.h:648`）；相位 A 把它加进 stride
  （`:721` `gp.qkStride = A.m + PF_GDN_QK_PAD_ROWS`）。
- 理由（逐字，`m15_layer_kernel.h:643-648`）：「AIC 的 `Nd2Nz` 每个 chunk 固读 `GP_BT = 64` 行，
  最后一个 chunk 会读到 m 之后的 pad 行 ⇒ 宿主必须按 `qkStride = m + 本常量` 给 q/k 定尺，
  并把 `[m, m+本常量)` 行**置零**（否则 mmad 读垃圾）」。
- 段体侧同款：`m15_gdn_prefill.h:88-89`；host 契约 `m15_gdn_prefill_host.h:21-35`
  （`kPadRows=64`、`kBT=64`、`GdnPrefillTp(m)=((m+7)/8)*8`）。
- 参考契约：`m23_gdn_prefill/check_ref.py:130-137` 取 `Tp = T + 64`，q/k reshape `(nk, Tp, DK)`
  （`:130-137`）；`m23_gdn_prefill/README.md:122-123`「`q/k` 平面末尾须留 ≥ 64 行可读且**置零**」。
  **v 无 pad**（`v.reshape(H, T, DV)`，`check_ref.py:134`），g/β 只有 8 对齐（README `:36` 附近、`check_ref.py:135-136`）。
- 对 prolog 的含义：**q/k 的 pad 行必须由产 q/k 的段（prolog）负责置零**；本仓在预填充路未读到该生产者。

---

## 6. 可执行接线方案（设计意图，不是已实现）

> 本节描述"若要接线，应改哪些文件的哪些符号、加什么"。落地属后续 mission；M140 分支
> （`852dfbc`）已把四相位实参与相位掩码做了部分准备（见 §6.4），但两条链路本体仍未接。

### 6.1 `m15_layer_kernel.h`（kernel 内挂载点与实参表）

1. **新增 `LayerArgs` 字段**（追加在 `pfHcIj0` 之后，`:272`；不动既有字段、不重排）：
   prolog/epilog 的**激活平面**指针，例如：`pfQkvzba`（in_proj 输出 bf16 `[m,16480]` 或分块）、
   `pfZ`（z bf16 `[m,6144]`）、prolog 的 conv_state / scratch 平面、epilog 的 token-major 暂存
   `pfOTok`（fp32 `[m,6144]`）。命名与粒度待段体定（§8-U5）。
   注：prolog 的**权重**已在基础字段里（`wIn/wOut/convW/convBias/aLog/dtBias/gammaG`，`:119-120`，
   FILL `:1096-1100`），无需新增字段，只需 host 传常驻指针。
2. **在 `M15L_PrefillBody` 插挂载点**（`:830-852`）：
   - prolog 调用：`H1` 之后、相位 A 之前，即 `:835` 与 `:837` 之间（消费 `A.pfHcBlk0`，产 q/k/v/g/β）。
   - epilog 调用：相位 A 之后、`H2` 之前，即 `:841` 与 `:843` 之间（消费 `A.wsO` + z，写 `A.hcAttnOut`）。
   段体签名沿用既有契约"只吃指针 + 标量"（`:699`）。
3. **实参表**：在 `M15L_LAYER_PREFILL_ARGS_DECL`（`:1259-1277`）与 `..._FILL`（`:1279-1307`）尾部**追加**
   新平面实参（顺序与 `LayerArgs` 追加字段一致）。
4. **`KIND_GDN` 的相位 A 组合**：现有 `M15L_PrefillPhaseA<KIND_GDN>` 只做扫描；接线后它应改成
   "prolog（新段）→ 扫描（B1）→ epilog（新段）"，或把 prolog/epilog 作为独立相位 A 前后段。
   具体切法属未定项（§8-U5）。
5. **负向对照保持非空洞**：`pfMutant`（`:692-696`）与 `pfStageMask`（`:830`）机制应在新增段上仍能
   把对照打红（§7.4）。

### 6.2 `m15_layer_loop.asc`（host 分配与相位驱动）

1. **新增激活平面**：在 `H_PfSegAlloc`（`:1762-1815`）里按 §6.1 的新字段分配；尺寸常量按 `m`
   定尺（现有 `qkB/vB/gbB/hB/scB` 的算法可循，`:1769-1778`）。
2. **权重不再传 nil**：`H_PfGdnWired` 的启动实参（`:3463-3479`）目前把 `wIn/convW/convBias/aLog/dtBias/gammaG`
   传 `nullptr`（`:3466`）；接线时传常驻 GM 指针（人类约束③）。
3. **相位掩码**：目前硬编码 `PF_STAGE_A`（`:3471`）；接线后需要 `PF_STAGE_H1|A|H2`（乃至 `B`）。
   **注意**：`M15_PREFILL_PHASES` 的运行期解析只在 M140 分支上（`wt-140/.../m15_layer_loop.asc`），
   main 上未读到。
4. **对拍 dump**：相位 A 已有 m23 dump（`:3517`）；epilog 落位后应增加 `hcAttnOut` 的 D2H 与对拍
   （现有 decode 路的 `H_D2H(..., C.hcAttnOutDev, ...)` 在 `m15_chain_host.h:378` 可参照）。

### 6.3 `m15_layer_resources.h`（flag / buffer / 峰值 / 登记表）

新增段若引入新的跨核同步点/UB 窗，需按现有三类登记表登记（权威 = 本文件）：

1. **flagId 序列表**：`§4c/§4d` 的 `FLAG_SEQ_PREFILL_*[]`（`:1051-1176`），每行须过
   `FlagIdRegisteredDecode`（复用 decode 号）、`FlagRefsAdjacentOk`（相邻逻辑点异号）与 id 上限。
   **不新造号**优先（`m15_layer_kernel.h:604`）。
2. **峰值槽**：`PF_PEAK_TABLE[]`（`:528-537`，固定 8 槽 `:539`）填 UB/L1/L0C 峰值；`PREFILL_WIRED`
   （`:503`）在两链路接通前保持 0，`PfPeakUnfilled` 门（`:542-558`）据此。
3. **BufferID**：`BUF_TABLE[]`（`:1323-1346`）与 prefill 行窗（`:1342-1345`）。
4. **入口表**：仅当新增 `__global__` 符号才改 `ENTRY_TABLE[]`（`:1407-1417`、`ENTRY_TABLE_N==9 :1419`）；
   若 prolog/epilog 作为现有 prefill 入口内的段，则不动。

### 6.4 M140 分支已有的准备（不属于 main）

- `M15L_LAYER_PREFILL_ARGS_DECL/FILL` 尾部追加 15 个 hc 层界/权重实参（`wt-140/.../m15_layer_kernel.h:1299-1308`，
  FILL `:1327-1342`）。
- host 解析 `M15_PREFILL_PHASES` 并驱动相位掩码。
- E=512 MoE 权重装载与 router pad 修复。
- **两条链路本体（prolog/epilog 段体与挂载）在 M140 上同样未接**（M140 复审 acceptance 段），本文的接线方案
  应与 M140 的 15 实参块一致地追加，避免 ABI 冲突。

---

## 7. 验收判据与可复用参考脚本

### 7.1 分档（照 `docs/17-verification-standard.md`）

- **T1 逐位（容差 0）**：纯数据搬移/转位/置零——`wsO → token-major` 的重排、q/k pad 行置零、
  conv_state 的 bf16 位型搬移（m9 参考已把 conv_state 定为逐位，`m9_gdn_prolog/check_ref.py:16-17`）。
- **T3（ε 公式 + 逐元素超界计数）**：含 fp32 累加链与 VF 超越函数的段（RMSNormGated、out_proj、
  conv1d+l2norm+gating）。口径见 `m18_gdn_prefill/check_ref.py:9-16`（`ε=5e-5`，
  `|got-exp| ≤ ε·Σ|terms| + 0.5·ulp`）。

### 7.2 段级对拍脚本（现成、可直接复用）

| 段 | 脚本 | 比较对象 | 需设备 dump |
| -- | ---- | -------- | ----------- |
| in_proj / out_proj | `m11_bf16_gemm/check_ref.py` | `a.bin/b.bin` vs `c_device.bin`，bf16 位型 RNE | 是 |
| conv1d+l2norm+gating | `m9_gdn_prolog/check_ref.py` | device 输出 vs numpy fp32；conv_state 逐位，q/k/v/g/β `1e-5·|exp|+1e-6` | 是 |
| RMSNormGated | `m12_rmsnorm_gated/check_ref.py` | C 参考 + device vs numpy RMSNormGated（`HEADS=48/HEAD=128/EPS=1e-6`） | 是 |
| GDN 扫描（相位 A） | `m23_gdn_prefill/check_ref.py` | device q/k/v/g/β/h0/o/ht vs numpy float64，`Tp=T+64`，rtol 2e-3；`--mutant` 必须红 | 是（`H_PfGdnWired` dump `m15_layer_loop.asc:3517`） |
| 扫描（独立参考路） | `m18_gdn_prefill/check_ref.py` | 同公式 fp64，T1 输入逐位、其余 T3 | 是 |
| attention FA core | `m25_attn_fa_core/check_ref.py` | dense causal FA vs fp64 | 是 |
| MoE prefill | `m26_moe_prefill/check_ref.py` | S2 router logits/ids/weights/sgate vs fp64 | 是（默认 `build/m26_out`） |
| hc 边界 | `m27_hc_prefill/check_ref.py`（复用 `m20_hyperconn/check_ref.py` 的 `reference`/`judge`） | hc m-tile A1/A2 | 是（`/tmp/m27_out`） |
| 层链 | `m15_layer_loop/check_ref.py`、`check_hc_ref.py`、`check_chain_ref.py`、`check_moe_ref.py` | hc 边界 / 层链 / MoE vs m20 / m39 torch / m17 参考 | 是（`M15_DUMP=1`） |

**已读到的缺口**：本仓**未读到**直接对拍"预填充 prolog（m>1）"或"预填充 epilog（wsO→hcAttnOut）"的
现成脚本——它们在 m=1 decode 形态下由 `m15_gdn_layer` 的段级对拍覆盖，但预填充 m>1 的段体本身尚未存在。

### 7.3 端到端判据（设计意图）

- **数值**：BLK（真值）→ prolog → 扫描 → epilog → `hcAttnOut`，与 golden 的逐段对拍；
  建议沿用 `m23` + `m26` 的"输入六张量 T1 逐位 + 其余 T3"分档。
- **对拍参考**：`m21_layer_ref`（`m15_layer_loop/check_chain_ref.py` 依赖的 M39 torch oracle）
  与 `m18`/`m23` 的 numpy 参考；m=1 档另可与 decode 全链 `GdnLayerChain` 的读数交叉（同一公式、不同 m）。
- **m 档**：至少 m=1（退化档，暴露转位退化）与 m=4097（尾 chunk = 1 行，`GP_M_PREFILL`）。

### 7.4 负向对照（必须能变红，非空洞性，`docs/17` §4）

- **in_proj / out_proj GEMM**：`m11_bf16_gemm` 本体**未读到**内建负向对照——入口签名 `bf16_gemm_kernel(a,b,c,m)` 无 mode 参数（`m11_bf16_gemm.asc:265`），`m11_bf16_gemm/check_ref.py` 也未读到变红分支。同型 GEMM 的现成负向对照落在 `m15_layer_loop/m15_attn_oproj.h:63` 的 `GEMM_MODE_KMINUS1`（K 方向少累加一个 base 块，`Process` 处 `:287`），由 `m15_attn_core_host.h:791`、`:867` 驱动 ⇒ 若 epilog 复用 attention 的 o-proj 段，此对照可用；若走 `m11` 本体，需自建（见本节末条）。
- `m23`：`--mutant` 必须红（`m23_gdn_prefill/check_ref.py`）。
- `m15` 预填充：`pfMutant` 把行数截到 `M_MAX`（`m15_layer_kernel.h:692-696`），该行标记必须保持毒值。
- `m27`：`reproduce.sh`（`m27_hc_prefill/reproduce.sh`）的负向对照族。
- **新增段应有自己的负向对照**：例如 prolog 不写 q/k pad 行 ⇒ 扫描段对拍必然红（证明 pad 契约不是空洞的）；
  epilog 不做转位（按 head-major 读）⇒ m>1 对拍必然红。

---

## 8. 未定项 / 风险（逐条标未知）

| # | 未定项 | 现状 | 风险 |
| - | ------ | ---- | ---- |
| U1 | **预填充 m>1 的 prolog 段体来源** | `m9` 是 decode m=1（`m15_gdn_resources.h:70`、`m9_gdn_prolog.asc:2`）；`m11` 的运行时 M-tile 循环已覆盖 m>64，且 M146 设备实测 m=4097 两形状逐位通过（`m11_bf16_gemm.asc:102`；`m11_bf16_gemm/evidence/m4097_envelope/README.md`）。本仓未读到 m>1 的 conv1d/l2norm/gating 实现 | conv1d 是跨 token 的深度卷积（K=4，`conv_state` planar）；m>1 的扫描式实现、分块与状态交接未定 |
| U2 | **in_proj 单次启动的 m 包络与激活平面定尺** | M146 设备实测：m11 的运行时 M-tile 循环（`mLoop=CeilDiv(m,64)`，`m11_bf16_gemm.asc:102`）在 m=4097 **单次启动**对两形状逐位通过（`m11_bf16_gemm/evidence/m4097_envelope/README.md`），无需改段体或分块多次启动 | 激活平面定尺仍未定：m=4097 的 `qkvzba` `[4097,16480]` bf16 ≈ 135 MB，host 分配可行性需核 |
| U3 | **z 平面的来源与持有者** | decode 里 z 在 in_proj 输出 `qkvzba` 的 `[10240,16384)` 段（`m15_gdn_resources.h:49`、`m15_gdn_layer.h:11`）；预填充 `GdnPrefillArgs` 未读到 z | z 必须跨 S2→S5 存活；预填充路谁产、落在哪个平面、是否复用 qkvzba 未定 |
| U4 | **BLK 语义与 S1/S7 的归属** | HC 形态下 `subIn = hcWs0+WS_BLK`（`m15_layer_kernel.h:945`）；decode 段仍跑 S1 Add+RMSNorm | BLK 是否已是"归一化后的 block input"、S1/S7 属子层段还是 hc 边界，未定；会影响 prolog 是否含 norm |
| U5 | **挂载点切法与段粒度** | prolog/epilog 是独立相位、还是并入相位 A 前后，未定 | 影响 `pfStageMask` 位、相位边界 flag、以及本次 bring-up 的截断粒度 |
| U6 | **token-major 暂存的归属与是否改 B1 契约** | 见 §4.3 | 改 B1 输出布局会牵动 m23 对拍口径（口径变更需登记+复审） |
| U7 | **新段的 flagId / BufferID / 峰值登记数值** | 登记表机制见 §6.3；具体号未定 | 相邻性/id 上限/4-bit 计数器约束（`m15_layer_resources.h:1051-1176`）未排 |
| U8 | **`hcAttnOut` 与 `wsO` 的规范关系由谁裁定** | M138 只说"未读到生产者关系"、`m27` README 请塔裁定（`:350-351`） | 若塔裁定 `wsO` 才是规范出口，则 H2 的 `bo` 取值点要改；本设计按"补 B1→`hcAttnOut` 的 epilog"给出 |
| U9 | **M140 分支合入时序** | 15 hc 实参 + 相位掩码在 `852dfbc`，未合入 main | 本文的实参追加顺序须与 M140 对齐，否则 ABI 冲突 |

---

## 9. 当前红项（作为计划输入；均**未解**，不写成已知根因）

### 9.1 相位 B m=1：MoE prefill router 输入面不符

- 出处：finding `.tower/comms/findings/20261004-agent-fullprefill-bug-m140-context-m-1-moe-prefill-router.md`
  （`Summary` / `Location` / `Details`），登记于 M140 README §8 缺口 #4。
- 读数（引自该 finding）：m=4097 最后一块对 m26 numpy 参考 5/5 PASS（`logits max|Δ|=7.66e-07`）；
  m=1 档 5/5 FAIL，`logits max|Δ|=6.816e8`，行内跨专家极差 `1.146e9`（≈ `0xCDCD` 毒值量级），
  `sgate max|Δ|=3.77e7`，guard G5 红；两次连跑逐字节相同 ⇒ 非随机竞态。m26 独立 m=1 路 PASS
  ⇒ 段体数学在 m=1 本身未见问题。
- 线索（finding 自述，**非结论**）：M140 融合档 m=1 时 x 平面只有 1 行有效（其余 `0xCD`），
  而 m26 独立档 x 有 MT=64 行有效；建议查 router 的 M 方向 tiling/L1 装载在 `mTotal ≪ MT` 时的行为；
  补一条 H2→B 可见性握手试过、读数变了但未修好（实验已回退）。
- 与本文的关系：该红项属**相位 B（MoE）**输入面，与本文的 prolog/epilog 不同段；但同属"融合四相位 context
  下小 m 的输入面契约"，是接线验收计划必须携带的输入。**根因未定位**，本文不代下结论。

### 9.2 m=4097 `h2.blk` ulpMax 11

- 出处：M140 分支 evidence `wt-140/m15_layer_loop/evidence/m140_prefill_full_layer/README.md`
  （summary `:19-20`、表 `:95`、注 `:96-99`）。**main 的证据目录内未读到该字符串。**
- 读数（引自该 README）：m=4097 p15 档 `h1.hcp`/`h2.hcp` 逐位一致率 1.0000；`h1.blk` 逐位 0.9949
  （ulpMax 1）；`h2.blk` 逐位 0.9797、good-bitrate 0.9903、**ulpMax 11** ⇒ 该项 FAIL
  （判据 `ulpMax > 2`）。作者登记为红项，未按 PASS 处理；与重定位/相位接线的因果关系**未归因**，
  m27 也观察到同族的 `blk` 残差。
- 与本文的关系：`h2.blk` 是 hc 边界 #2 的 BLK 平铺面（§5.1 同族契约），与 epilog 落位后的
  `hcAttnOut` 面相邻。**未解**，本文不写成已知根因，仅作为接线后的回归输入。

---

## 10. 引用索引（`文件:行`）

- 相位拓扑 / 入口：`m15_layer_kernel.h:9-21`，`:686-740`，`:816-852`，`:914-969`，`:1312`，`:1319`，`:1341`，`:1348`
- 实参表：`m15_layer_kernel.h:1081-1089`，`:119-120`，`:1096-1100`，`:1259-1277`，`:1279-1307`
- 平面字段：`m15_layer_kernel.h:150`，`:258-264`，`:270-272`；`m15_hc_resources.h:40`；`m15_hc_prefill.h:114`，`:125`
- pad / 相位位：`m15_layer_kernel.h:643-648`，`:650-659`，`:692-696`，`:721`
- BLK 来源：`m15_layer_kernel.h:795-812`，`:832`，`:945-946`；`m15_hc_prefill.h:395-402`；`m15_layer_loop.asc:1777`
- prolog 资产：`m11_bf16_gemm.asc:46-49`，`:102`，`:120-121`，`:250`，`:265`，`:362-366`；`m11_bf16_gemm/evidence/m4097_envelope/README.md`；`m9_gdn_prolog.asc:2`，`:61-71`，`:256-266`，`:434-440`
- epilog 资产：`m12_rmsnorm_gated.asc:4-20`，`:72-78`，`:395-402`，`:557`；`m15_attn_oproj.h:50-54`，`:184`，`:433-437`
- GDN 段序 / 常量：`m15_gdn_layer.h:5-13`，`:1230`，`:1259`，`:1735-1736`，`:1794`，`:1833`；`m15_gdn_resources.h:47-70`，`:312-318`
- 扫描契约：`m15_gdn_prefill.h:85-106`，`:1135`，`:1143`；`m15_gdn_prefill_host.h:21-35`
- M138 缺口：`m15_hc_prefill.h:72-76`；`m27_hc_prefill/README.md:337-351`；`docs/15-prefill-design.md:889`
- 登记表：`m15_layer_resources.h:503`，`:528-539`，`:1051-1176`，`:1323-1346`，`:1407-1419`
- host 驱动：`m15_layer_loop.asc:643`，`:677-691`，`:1762-1815`，`:3312-3336`，`:3429-3527`
- 参考脚本：`m9_gdn_prolog/check_ref.py`，`m11_bf16_gemm/check_ref.py`，`m12_rmsnorm_gated/check_ref.py`，
  `m18_gdn_prefill/check_ref.py`，`m23_gdn_prefill/check_ref.py:130-137`，`m25_attn_fa_core/check_ref.py`，
  `m26_moe_prefill/check_ref.py`，`m27_hc_prefill/check_ref.py`，`m15_layer_loop/check_ref.py`/`check_hc_ref.py`/`check_chain_ref.py`/`check_moe_ref.py`
- 红项：`.tower/comms/findings/20261004-agent-fullprefill-bug-m140-context-m-1-moe-prefill-router.md`；
  `wt-140/m15_layer_loop/evidence/m140_prefill_full_layer/README.md:19-20,95-99`
- 人类约束：M143 mission Context（逐字）。
