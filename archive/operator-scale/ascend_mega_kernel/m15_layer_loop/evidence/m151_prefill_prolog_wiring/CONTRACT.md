# M151 接线契约 —— BLK → S2(in_proj) → S3(prolog) → 相位 A 输入

> 口径：本文件先把**逐段数据流契约**钉成 `文件:行`（写代码之前），再据它实现。**行号基准 = 本文件
> 所在 commit 的 tip**（复算：`git log -1 --format=%H -- m15_layer_loop/evidence/m151_prefill_prolog_wiring/CONTRACT.md`）。
> **r1 复审（P2-2）指出**：原版把 `m15_layer_loop.asc` 的引用误写成 base-main 号（却声称按 tip）；
> 本次已**重锚到 tip**。base→tip 对照（供追溯）：`H_W` 841→871、`H_PfGdnFillHost` 3374→3427、
> 实参对账表 3320-3345→3363-3401、7 个 nullptr 权重实参 3822-3826→4103-4104、`pfgdn_q` H2D
> 3710-3715→3989、`cs_init` 886→916、KIND_ATTN 响亮失败 4199-4204→4519、`blkB` 1813→1843、
> `H_PfGdnGeom` 1786-1793→1814-1821、`H_AssembleBigW` 555→577、w 槽 H2D 2094/2105→2147/2158。
> **未变文件**（`m15_layer_kernel.h` / `m9_gdn_prolog.asc` / `m15_attn_oproj.h` / `m15_loop_layout.h` /
> `m15_hc_prefill.h`）的行号不受本次整改影响。
>
> **本 mission 只接 prolog 链路**：`H1.BLK → S2 in_proj → S3 conv1d/l2norm/gating → 相位 A(q/k/v/g/β)`。
> epilog（`wsO → hcAttnOut`，S5/S6/S7）与 attention B2/B3 **未接**。不得读成「整层四相位打通」。

## 0. 相位拓扑与挂载位置

融合 prefill 入口的四相位骨架：`H1 → A → H2 → B`（`m15_layer_kernel.h:18-21`，
`M15L_PrefillBody` `m15_layer_kernel.h:834-878`）。prolog 挂在 **H1 之后、相位 A 之前**：

| 段 | 现有位置 | 本 mission 追加 |
| -- | -------- | --------------- |
| H1（B5，hc 边界 #1） | `M15L_PrefillBody` `m15_layer_kernel.h:850` → `M15L_HcPrefillBoundary(0u,A)` `:813` | 不变（产出 BLK） |
| **S2 in_proj** | 预填充路无 | **新增**（本文件 §2） |
| **S3 prolog** | 预填充路无 | **新增**（本文件 §3） |
| 相位 A（B1 扫描） | `M15L_PrefillPhaseA` `m15_layer_kernel.h:687-740`，调用点 `:856` | 输入改为**设备产出** |

## 1. BLK（H1 输出，S2 的输入）

- **声明**：`LayerArgs::pfHcBlk0/pfHcBlk1` = 「边界 #1 / #2 的 BLK 平铺面 `[m, HID]` bf16」
  （`m15_layer_kernel.h:270-271`）；实参传入 `:1335-1336`。
- **生产者**：`M15L_HcPrefillBoundary(0u, A)`（`m15_layer_kernel.h:823` 取 `A.pfHcBlk0`）—
  由 `M15H::HcPF::Body` 的 `RelocateTile` 把 arena 阻塞布局搬成平铺面
  （`m15_hc_prefill.h:395-402`）。
- **几何**：行主序 `[m, HID=2560]` bf16，行距 `ROW_HID = HID*2 = 5,120 B`（`m15_hc_prefill.h:114`），
  平面字节 `m*ROW_HID`（`m15_hc_prefill.h:125`；host 定尺 `m15_layer_loop.asc:1843`）。
- **S2 消费**：作为 GEMM 的 A 面 `[m, K=2560]` bf16，行距 = `K*2 = 5,120 B`（§2）—— BLK 与 A 面
  **逐字节同型、行距一致**，可直接别名 `A.pfHcBlk0`。

## 2. S2 in_proj（m11 bf16_gemm）

- **donor 入口**：`bf16_gemm_kernel<K,N>(a,b,c,m)`（`m11_bf16_gemm/m11_bf16_gemm.asc:264`）；
  template = **`<K, N>`**（`m11_bf16_gemm.asc:84`），本档实例 `<2560, 16480>`（`:362-366`）。
- **m15 现有 lift**（复用对象，行号二选一）：
  - `M15OP::OProjGemm<K, N>`（`m15_layer_loop/m15_attn_oproj.h:264`）—— 通用模板，含负向对照
    `GEMM_MODE_KMINUS1`（`:63`，`Process` 内 `kLoop-1`）。本 mission 用它实例化 `<2560,16480>`。
  - `Cube::Bf16Gemm<2560,16480>`（`m15_layer_loop/m15_gdn_layer.h:1402`）—— decode 链用，
    `RunTile(nBlock)` 按 AIC 条带划分 N。
- **平面契约**（K=2560, N=16480）：

  | 面 | 参数 | shape | dtype | 行距/stride | 出处 |
  | -- | ---- | ----- | ----- | ----------- | ---- |
  | A（激活） | `a` | `[m, K=2560]` | bf16 | 行距 = `K` 元素 = 5,120 B | `m15_attn_oproj.h:279`；`m11:102` |
  | B（权重） | `b` | `[N=16480, K=2560]` 行主序（按 `B^T` 消费） | bf16 | 行距 = `K` 元素 | `m11_bf16_gemm.asc:29` |
  | C（出） | `c` | `[m, N=16480]` | bf16（Fixpipe F322BF16） | 行距 = `N` 元素 = 32,800 B | `m15_attn_oproj.h:281` |

  shape 裁决：N=16480 = `q2048 | k2048 | v6144 | z6144 | b48 | a48`；K=2560 = `HIDDEN`。
- **m=1 quirk**：`Nd2Nz` 在 `nValue==1` 退化 ⇒ 段体把计算 M 抬到 `max(curM,2)`
  （`m11_bf16_gemm.asc`；`m15_gdn_layer.h:1441`）；A 面须有 ≥2 行可读 ⇒ 预填充 A 面按 `M_PREFILL`
  定尺（`m15_layer_loop.asc:1814-1821` 几何口径），m=1 时天然满足。
- **权重常驻**：B = `wIn`，来自 decode 权重区 `H_W(C,L,M15Loop::W_IN_OFF)`（`m15_layer_loop.asc:871`，
  `m15_loop_layout.h:135`），由 `H_AssembleBigW`（`m15_layer_loop.asc:577`，角色
  `in_proj_qkv|z|b|a` 按行拼接 `[16480,2560]`）装配、H2D 落设备（`:2147/:2158`）。
- **现状**：预填充路把 `wIn` 实参传 `nullptr`（`m15_layer_loop.asc:4103`，对账表 `:3375`）。

## 3. S3 prolog（m9 gdn_prolog_mt_kernel）

- **入口**：`gdn_prolog_mt_kernel(x,convState,w,bias,aLog,dtBias,q,k,v,g,beta,m,qScaleOn,noShift,noPad)`
  （`m9_gdn_prolog/m9_gdn_prolog.asc:844`）；类 `MtGdnProlog`（`:629`），`__vector__ __global__`（AIV-only）。
- **输入平面**：

  | 参数 | shape | dtype | 语义 / 常驻位置 | 出处 |
  | ---- | ----- | ----- | --------------- | ---- |
  | `x` | `[m, INW=16480]` 行主序 | bf16 | **S2 的输出 C**；行距 `INW` 元素 | `m9:641,:716-718` |
  | `convState` | planar `[3][10240]` | bf16 | conv 历史 3 行，GM 常驻、in-place RMW | `m9:642,:702-705` |
  | `w` | `[4][10240]` 行 = tap j | bf16 | conv 权重（tap-major） | `m9:643,:687` |
  | `bias` | `[10240]` | bf16 | conv bias | `m9:644,:689` |
  | `aLog`/`dtBias` | `[64]`（48 有效） | fp32 | gating 常量 | `m9:645-646,:691-692` |
- **输出平面**（= 相位 A 输入契约，§4）：

  | 参数 | shape | dtype | pad |
  | ---- | ----- | ----- | --- |
  | `q`,`k` | `[16, qkStride=m+64, 128]` | fp32 | 行 `[m,m+64)` **置零**（`noPad=0`） |
  | `v` | `[48, m, 128]` | fp32 | 无 |
  | `g`,`β` | `[48, tp=align8(m)]` | fp32 | 列 `[m,tp)` 置零 |
- **scale**：默认 `qScaleOn=0` = 相位 A 契约（q/k 均不乘 1/√128，由消费侧施加，`m15_layer_kernel.h:722`）；
  `=1` 切回 m9 历史（仅 q 乘），用于 m=1 交叉复现。
- **负向开关**：`noShift=1`（不交接 conv_state，每个 token 用初始 state）、`noPad=1`（不写 q/k pad 行）
  （`m9:637-638`）—— 本 mission 的两个负向对照档直接复用。
- **导出几何常量**：`MT_PAD=64`（= `PF_GDN_QK_PAD_ROWS`，`m15_layer_kernel.h:648`）、
  `tp_=align8(m)`（`m9:639-640`）。

## 4. 相位 A 输入（`wsQ/wsK/wsV/wsG/wsBeta`）

- **装配点**：`M15L_PrefillPhaseA<KIND_GDN>`（`m15_layer_kernel.h:703-728`）：
  `gp.q=A.wsQ … gp.beta=A.wsBeta`；`gp.tp=(A.m+7)/8*8`（`:717`）；`gp.qkStride=A.m+PF_GDN_QK_PAD_ROWS`
  （`:721`，常量 `:648`）；`gp.scale=1/√128`（`:722`）。
- **字段声明**：`m15_layer_kernel.h:258-264`。`q/k [NK=16, m+64, 128]`、`v [48,m,128]`、
  `g/β [48, align8(m)]` fp32。
- **现状（宿主合成）**：`H_PfGdnFillHost`（`m15_layer_loop.asc:3427`）造 q/k（L2 归一）、v、g、β，
  经 `H_H2D` 落 `pfGdnQDev…pfGdnBetaDev`（`:3989`）。**本 mission 保留该路径为可切开关**。
- **本次改为设备产出**：S3 直接写同名平面（q/k/v/g/β），相位 A 读到的字节由设备产生。

## 5. 权重常驻位置（本 mission 传真指针）

预填充路当前 7 个 GDN 权重实参全传 `nullptr`（`m15_layer_loop.asc:4103-4104`；对账表 `:3363-3401`）。
本 mission 把 S3 需要的权重接成**常驻 GM 指针**（人类约束③「权重就是要常驻」）：

| prolog 参数 | 常驻来源 | shape | 出处 |
| ----------- | -------- | ----- | ---- |
| `w`（convW） | `H_W(C,L,M15Loop::W_CONV_OFF)` | `[4,10240]` bf16 | `m15_layer_loop.asc:871`；`m15_loop_layout.h:139` |
| `bias`（convBias） | `H_W(C,L,M15Loop::W_CONVB_OFF)` | `[10240]` bf16 | `m15_loop_layout.h:141` |
| `aLog` | `H_W(C,L,M15Loop::W_ALOG_OFF)` | `[64]` fp32 | `m15_loop_layout.h:143` |
| `dtBias` | `H_W(C,L,M15Loop::W_DTB_OFF)` | `[64]` fp32 | `m15_loop_layout.h:145` |
| S2 的 `b`（wIn） | `H_W(C,L,M15Loop::W_IN_OFF)` | `[16480,2560]` bf16 | `m15_loop_layout.h:135` |

conv_state：本 mission 新增 `pfConvStateDev`（`[3,10240]` bf16），初值取该层 `C.csInit`
（`m15_layer_loop.asc:916` 的起始 conv_state 口径，decode 路同源）。

## 6. 同步口径（人类约束）

「核内 pipeline 用 BufferID，核间用 set/write cross core，不得 set flag/wait flag」：

- **S2 donor**（`M15OP::OProjGemm`，`m15_attn_oproj.h:96` `OpAcq/OpRls` = `GetBuffImpl/RlsBuffImpl`
  drain release）与 **S3 donor**（`m9:121-129` `GetBufInternal/RlsBufInternal`）内部已是 BufferID。
- 本 mission 的 S2/S3 各自为**独立 `__global__`**、由同一条 stream 串行 ⇒ 段间不引入任何
  `set_flag/wait_flag`、也不新增跨核 flagId；核间次序由 stream 保证。

## 7. 本次不接（窄口）

- **epilog**：`wsO → hcAttnOut`（S5 RMSNormGated + S6 out_proj + S7 残差）未接
  （`evidence/m140_prefill_full_layer/README.md:169-174`）。
- **attention B2/B3**：`KIND_ATTN` 在 `wire=1` 下响亮失败（`m15_layer_loop.asc:4519`）。
- **已有红项原样保留**：相位 B m=1（`evidence/m140_prefill_full_layer/README.md:133-138`）、
  m=4097 `h2.blk` ulpMax 11（`:120-124`）。
