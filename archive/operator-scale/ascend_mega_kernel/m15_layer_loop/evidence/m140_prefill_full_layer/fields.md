# M140 任务 1 的交付物：要补进 prefill 入口的字段清单（逐条）

**取证先于动手**。来源：`m15_layer_loop/evidence/m136_prefill_gdn/README.md`（M136 的四档读数与限度）、
M136 立的 medium finding `.tower/comms/findings/20261004-agent-waved1-bug-m136-prefill-hc-gdn-h1-h2-b.md`、
`m27_hc_prefill/README.md` 的融合清单（§5 边界 / §6 挂载点补丁）。

## 0. 现状（M136 留下的缺口）

`M15L_LAYER_PREFILL_ARGS_DECL/FILL`（`m15_layer_kernel.h`，M136 时点）= 基础 30 实参 + M110 的 13
（`pfKv..pfMutant`）+ M132 的 13（`wsQ..pfHcIj0`）。**没有** hc 的层界/权重字段 ⇒ 入口内
`M15L_LAYER_ARGS_FILL` 把 `LayerArgs` 的 hc 字段全部置 nil ⇒ `M15L_HcPrefillBoundary`（H1/H2）会读到全 nil、
相位 B（MoE）也缺 H1/H2 的数据流。M136 因此只开相位 A（`pfStageMask = PF_STAGE_A`）。

`LayerArgs` 里这些 hc 字段**本来就存在**（M58 的四相位入口在用），本 mission 只是把它们加进 prefill 入口。

## 1. 要补的 15 条（全部来自 `LayerArgs` 的既有 hc 字段）

| # | 字段名 | 语义（形状） | 从哪取 | 归哪个相位 |
|---|---|---|---|---|
| 1 | `hcH` | 层输入多流残差 H：`[m, HYPER=10240]` bf16 | 激活（真链上 = 层输入 / 上一层残差；本档 host 合成，见 README §限度） | 边界 #1 hIn |
| 2 | `hcBo` | pending block output：`[m, HID=2560]` bf16 | 激活（真链上 = 上一层 mlp_out） | 边界 #1 bo |
| 3 | `hcIj` | pending injection logits：`[m, 16]` bf16（前 `INJ_N=4` 列有效） | 激活（真链上 = 上一层 OH 的 inj 列） | 边界 #1 ij |
| 4 | `hcAttnOut` | 子层段出口：`[m, HID]` bf16 | 激活（真链上 = 本层子层段的出口） | 边界 #2 bo |
| 5 | `hcAttnNorm` | attn_hc 的 norm 权重 `[HC, HID]` bf16 | 权重：`HC_LAYER_W_STRIDE` 槽的 `HC_ATTN_W_OFF + HW_NORM_OFF` | 边界 #1 norm |
| 6 | `hcAttnDown` | attn_hc 的 down 投影 `[LOWRANK=320, HYPER]` bf16 | 同上 + `HW_WDOWN_OFF` | 边界 #1 down |
| 7 | `hcAttnInj` | attn_hc 的 inject 权重 `[16, HYPER]` bf16（前 4 行有效） | 同上 + `HW_WINJ_OFF` | 边界 #1 inj |
| 8 | `hcAttnUp` | attn_hc 的 up 投影 `[HYPER, LOWRANK]` bf16 | 同上 + `HW_WUP_OFF` | 边界 #1 up |
| 9 | `hcMlpNorm` | mlp_hc 的 norm `[HC, HID]` bf16 | `HC_MLP_W_OFF + HW_NORM_OFF` | 边界 #2 norm |
| 10 | `hcMlpDown` | mlp_hc 的 down `[320, HYPER]` bf16 | `HC_MLP_W_OFF + HW_WDOWN_OFF` | 边界 #2 down |
| 11 | `hcMlpInj` | mlp_hc 的 inject `[16, HYPER]` bf16 | `HC_MLP_W_OFF + HW_WINJ_OFF` | 边界 #2 inj |
| 12 | `hcMlpUp` | mlp_hc 的 up `[HYPER, 320]` bf16 | `HC_MLP_W_OFF + HW_WUP_OFF` | 边界 #2 up |
| 13 | `hcIjStride` | ij 的**行距（元素）**：独立平面 = `IJ_STRIDE = 16`；OH handoff = `OH_W = 336` | 常量 `M15H::HC_IJ_STRIDE_PLANE` | 边界 #1 ij 寻址 |
| 14 | `hcAttnMode` | H1 的 mode（0=MIX / 1=COMBINE_MIX / 2=FINAL_MIX / 3=COMBINE_ONLY） | 真链上：层 0 = `MODE_MIX`，其余 = `MODE_COMBINE_MIX`。**本 harness 一律取 `MODE_COMBINE_MIX`**（`MODE_MIX` 不物化 `H'`，而边界 #2 的 `hIn` 正需要它）⇒ 真链上 layer 0 的那个 mode 本档没跑到，登记见 README §10 第 3 条 | 边界 #1 |
| 15 | `hcMlpMode` | H2 的 mode | 真链上恒 `MODE_COMBINE_MIX` | 边界 #2 |

**为什么是 15 条而不是更多**：`LayerArgs` 里另有 `hcWs0/hcWs1`（decode 四相位路径的 hc 私有 ws），
prefill 路径**不用它们** —— B5 段的 arena 由 `pfHcArena0/1`（M132 已加）承担，`M15L_HcPrefillBoundary`
的 `MakePlan` 只吃 `arena` 不吃 `hcWs*`。`hcPleBreak` 同理不传（prefill 不做 PLE 打断点）。

## 2. 相位 B（B4）所需的**额外**输入（不在 `LayerArgs`，走既有 MoE 字段）

相位 B 的段体按 `E=512 / topk=10` 编译（`m15_moe_prefill_res.h`），它吃的 12 个权重指针
（`moeWGu..moeSDnShd`、`moeGamma1/2`）在 prefill 入口上**已经是实参**（基础 30 里），但 M136 的
prefill 档传的是 **decode 的 E=4 槽**（`C.moeWDev`，13.09 MB/层）。E=512 的槽是
`MOE_W_PREFILL_STRIDE = 1.34 GB/层`（102.52×）⇒ 相位 B 打开前必须有 E=512 的装载路径。

| 项 | 语义 | 从哪取 | 归哪个相位 |
|---|---|---|---|
| `moeWGu/sGu/wDn/sDn` | `[E=512, …]` MXFP4 packed + e8m0 scale | manifest 的 `moe512_*` 段（M93 已入 manifest），槽偏移 = `MWP_*` | B（S6/S8） |
| `moeWGuShd/sGuShd/wDnShd/sDnShd` | 共享专家（与 E 无关） | manifest 的 `moe_shared_*`（复用 decode 的 role） | B |
| `moeGamma1/2` | MoE 段层级 norm（checkpoint 无此形状 ⇒ 合成） | 同 decode 口径 | B（S1/S10） |
| `pfMoeRouterWPad` | `[E_PAD=640, HIDDEN]` bf16（512 专家 + 共享门 + pad 行） | 由 `moe512_router_w` + `moe_sgate_w` host 组板 | B（S2） |

> M136 的 `H_PfGdnBuildRouterWPad` 从 **decode 的 E=4 router 槽**读 512 行（越界读槽外字节）——
> 相位 B 未开时没暴露；本 mission 改为从 E=512 的 router 平面取。

## 3. 逐条判据（"补了没有"怎么核）

- 结构：`M15L_LAYER_PREFILL_ARGS_DECL` 里出现这 15 个形参名；`FILL` 里 15 个 `A.hc* = hc*;`
  （`grep -c` 可核）。
- 行为：`M15_PREFILL_PHASES` 逐位打开时，对应相位的产出面从"全毒值"变成"非毒值/非零"
  （读数见 `README.md` §设备读数）。
- 数值：H1/H2 用 `m20_hyperconn/check_ref.py::reference` 复算（逐张量判），相位 A 用
  `m23_gdn_prefill/check_ref.py`。
