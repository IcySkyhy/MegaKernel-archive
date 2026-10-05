/**
 * m15_layer_resources.h —— **融合 per-layer kernel 的全局资源登记表**（M40 两相位 → M58 四相位）
 *
 * 融合形态（M58）：每层一次 `__mix__(1,2)` 启动，kernel 内**四个相位**（一次启动段序）
 *
 *   相位 H1：hc 边界 #1（`attn_hc`）  —— hc 段（m15_hc_layer.h，复制自 m20，见 lift_hc_segment.py）
 *   相位 A ：子层段                   —— GDN 段 S1-S7（m15_gdn_layer.h，复制自 m14）
 *                                        或 attention 占位直通（m15_attn_layer.h）
 *   相位 H2：hc 边界 #2（`mlp_hc`）    —— 同一个 hc 段实现，另一组权重/另一块 ws
 *   相位 B ：MoE 段 S1-S10             —— m15_moe_layer.h（复制自 m13）
 *
 * 层链数据流（与 vLLM 的 `Qwen4ExpDecoderLayer.forward` 逐句对应）：
 *
 *   H  ──H1(combine_and_mix | layer0 的 mix-only)──► BLK_attn ──A──► attn_out
 *      └────────────────────────────► H'(多流残差)                     │
 *   H' , attn_out ──H2(combine_and_mix)──► BLK_mlp ──B(MoE)──► yLayer（= 下一层的 pending BO）
 *                                            └──► H''（多流残差，层输出之一）
 *
 * 相位交界的同步（**三条**：H1→A、A→H2、H2→B）与 M40 的两相位版本同款：
 * `PipeBarrier<PIPE_ALL>`（核内全 pipe drain）+ 全体 AIV mode-0 barrier（跨核可见性），
 * 详见 m15_layer_kernel.h。**仍不跨 kernel**：每层仍是一次启动，只是这一次里有四个相位。
 *
 * ---------------------------------------------------------------------------
 * 一、命名空间与「mode 分节」声明（M33/discs15 §5.3 ★8 的硬约束）
 * ---------------------------------------------------------------------------
 * - GDN 段的资源常量在 `namespace M15G`（m15_gdn_resources.h，复制自 m14）；
 * - MoE 段的资源常量在 `namespace M15M`（m15_moe_resources.h，复制自 m13）；
 * - hc 段的资源常量在 `namespace M15H`（m15_hc_resources.h，复制自 m20）；
 * - **本文件（M15L）是三段合起来的登记表**：只定义「跨段共享的那些平面」（GM workspace
 *   切分、hc 与 MoE 的权重槽、以及**全局的 flagId / BufferID 命名空间图**）。
 *
 * 声明（★8 要求头文件显式写出，也是 id 可复用的前提）：
 *   **四段在本 kernel 内互斥执行、且每个 (核型, mode) 子空间的 flagId 集合由 §4 的
 *   `FLAG_SEQ[]` 按执行序逐对核对（相邻必不同）**。hc 段**两次**出现（H1/H2）并用**同一组**
 *   id（与 MoE 的 4 槽 ring 同款复用）。子空间层面的错开情况必须说准（不要笼统地写「不相交」）：
 *     · **mode 2**：hc 8-11 / GDN 4-7 / MoE 0-3 —— 三段**完全不相交**；
 *     · **AIC mode 0**：hc 0-3 / MoE 8-11 / GDN 12-15 —— 三段**完全不相交**；
 *     · **AIV mode 0**：hc 12/13/14 与 MoE 的 ring 12-15 **有意复用同一批 id**（这正是 §4 里
 *       `reuse=true` 那一族的由来；两者在四相位执行序里隔着 hc 自己的另两条 barrier 与两个相位
 *       边界）——**相邻性由 §4 的 `FlagSeqAdjacentOk()` 逐对保证，用量由 `FlagMaxUse()` 核对**。
 *   BufferID 因每核只有 28 个、三段各自需要 7/17/17 个，**无法完全不相交**：本表按 docs/12 §4
 *   的「段窗叠放」范式让 id 跨相位复用，前提是**每个相位交界处的 PipeBarrier<PIPE_ALL>**
 *   （每核的 6 条 pipe 全部 drain）——即 ★8 说的「两 mode 互斥 ⇒ id 可复用」，且这里的
 *   drain 是显式语句而不是依赖某个 flag 的语义。§5 逐 id 登记了「哪一段在用」。
 *
 * ---------------------------------------------------------------------------
 * 二、入口符号（docs/15 §5 方案 B：单 TU、入口符号由 host 选、kernel 内无 m 分叉）
 * ---------------------------------------------------------------------------
 * 单 TU = m15_layer_loop.asc（#include m15_gdn_layer.h + m15_hc_layer.h + m15_attn_layer.h
 * + m15_moe_layer.h + 本文件 + m15_layer_kernel.h，编译期资源表只有一份）。
 *
 * | 符号 | 状态 | 说明 |
 * |---|---|---|
 * | `m15_layer_kernel_gdn`   | **已实现**（decode，两相位） | 子层段 + MoE 段（M40/M25 的验收形态） |
 * | `m15_layer_kernel_attn`  | **已实现**（decode，两相位） | 同上 |
 * | `m15_layer_kernel_gdn_hc`  | **M58 已实现**（decode，**四相位**） | hc(attn) + GDN 段 + hc(mlp) + MoE 段 |
 * | `m15_layer_kernel_attn_hc` | **M58 已实现**（decode，**四相位**） | hc(attn) + attention 占位 + hc(mlp) + MoE 段 |
 * | `m15_moe_segment_kernel` | 已实现（**仅验证用**） | 只跑 MoE 段；「段间零串扰」判据的对照路 |
 * | `m15_hc_segment_kernel`  | **M58 已实现**（**仅验证用**） | 只跑**一个** hc 边界（任意 mode，含 combine-only）；对照路 |
 * | `m15_layer_kernel_gdn_prefill`    | **M110 真入口**（prefill；**段体未接，默认关**） | 相位 A 挂载点 + 行分派；B1 接进来（`M103-2.2`） |
 * | `m15_layer_kernel_attn_prefill`   | **M110 真入口**（prefill；**段体未接，默认关**） | 同上；B2/B3 接进来 |
 *
 * M110 起这两个名字**不再是注释 + 字符串常量**：它们有真 `__global__` 定义（`m15_layer_kernel.h`），
 * 并在 §6b 的 `ENTRY_TABLE[]` 里逐项登记（符号名 / kind / 相位数 / wired 开关），
 * `wired` **缺省 0**（结构占位：逐行分派 + 写见证面；数学未实现，显式标注）。
 * m 仍以**运行期参数**进 kernel 签名（★5：不要学 m15 把 m退化成常量），但 kernel 内**没有任何
 * 按 m 的分叉**：tiling 是编译期常量，m 只进 curM mask / 行循环上界。
 *
 * ---------------------------------------------------------------------------
 * 三、GM 平面切分（激活 / 状态）
 * ---------------------------------------------------------------------------
 * 每层的层内 workspace：
 *
 *   ws[0, WS_BYTES_TOTAL)                 = **每层一次启动复用**的单块融合 ws，内含两段顺排：
 *       ws[0, WS_GDN_BYTES)                   GDN 段 ws（M15G 的 WS_* 偏移表）
 *       ws[WS_MOE_OFF, …+WS_MOE_BYTES)        MoE 段 ws（M15M 的 WS_* 偏移表）
 *   hcWs[L] = HC_WS_LAYER_STRIDE × L        = **按层号编址**的 hc ws（两块 × 一个边界一块）
 *       hcWs[L] + HC_WS0_IN_LAYER             hc 边界 #1（attn_hc）的 ws
 *       hcWs[L] + HC_WS1_IN_LAYER             hc 边界 #2（mlp_hc）的 ws
 *
 * **hc 的 ws 为什么必须按层编址**：hc 边界的输出里有**两个跨层存活**的张量 —— `H'`
 * （`WS_HCP`，更新后的 4 路多流残差流 = 下一层的 `hcH`）与 `OH[:,320:324)`（injection logits
 * = 下一层的 `hcIj`，按行距 `OH_W` 直读）。它们必须活到下一层启动，故不能落在每层复用的单块
 * ws 里。代价 48 × 8.7MB = 417MB（见 README §资源表），收益是**跨层 handoff 零拷贝**。
 * **M65 已把这条接线接上**（48 层链 `runs=chain`，见 m15_chain_host.h）；层槽之后还有
 * 1 块末层全局 mixer 的 ws（`HC_WS_GMIX_SLOT`）。
 * 状态（conv_state / ssm_state）只有 GDN 段有；MoE/hc 段无跨 token 状态。
 */

#ifndef M15_LAYER_RESOURCES_H
#define M15_LAYER_RESOURCES_H

#include <cstdint>

#include "m15_gdn_resources.h"
#include "m15_hc_resources.h"
#include "m15_moe_resources.h"
#include "m15_attn_prolog.h"   // M97：attention 相位 A 的 flagId / UB / L1 / L0C 取证需要 M15AP 的常量
// M168 / M185：§4f.2 登记 B3（`m15_attn_fa_core.h`）的 **mode-2** 通道族。**引用它的常量而不是抄一份数值**
// —— 该头是这 6 个号的权威 owner（`CC_S_RDY/CC_S_FREE/CC_PV_RDY/CC_PV_FREE/CC_P0/CC_P1`），
// 本表只做登记与断言。该头自带 `m15_attn_kv.h` 的 include，
// 不依赖调用方的 include 顺序；未实例化任何模板 ⇒ 不产生设备代码（M168 实测二进制逐字节不变）。
#include "m15_attn_fa_core.h"

namespace M15L {

using namespace M15G;   // HIDDEN / M_MAX / BASE_* / UB_* / WS_*（GDN 段）

// ============================================================
// 0. 相位 / 层类型 / 实例参数
// ============================================================

constexpr uint32_t KIND_GDN = 0;
constexpr uint32_t KIND_ATTN = 1;

// MoE 段的实例参数（m13 的入口把它们当运行期开关；融合后按 m15 对 m14 的同款处理折成常量：
// **段序一字未动，只是把 bring-up 用的死分支折掉** —— 见 README 的差异表）
constexpr uint32_t MOE_SLICE_MODE = 1;    // 1 = 层链模式（router/permute/共享量化消费 S1 的 x_norm）
constexpr uint32_t MOE_STAGE_LIMIT = 8;   // 8 = 段序全开 S1→S10
constexpr uint32_t MOE_SUB_LIMIT = 9;     // 9 = router/索引生成段内全开

// hc 段的实例参数（m20 的入口把它们当运行期开关；同款处理）
constexpr uint32_t HC_STAGE_LIMIT = M15H::HC_STAGE_LIMIT;   // 7 = 段序全开 W0→S6

// 层内 workspace 三段顺排（512B 对齐：UB→GM DataCopy 32B 对齐 + 便于整槽 memcmp）
constexpr uint32_t Align512(uint32_t x) { return (x + 511u) / 512u * 512u; }
constexpr uint32_t WS_GDN_BYTES = WS_BYTES;                       // M15G::WS_BYTES
constexpr uint32_t WS_MOE_OFF = Align512(WS_GDN_BYTES);
constexpr uint32_t WS_MOE_BYTES = M15M::WS_BYTES;
constexpr uint32_t WS_BYTES_TOTAL = WS_MOE_OFF + WS_MOE_BYTES;
static_assert(WS_MOE_OFF >= WS_GDN_BYTES, "MoE 段 ws 不得与 GDN 段 ws 重叠");

// ---- hc 段的两块 ws：**按层号编址**（不是每层复用的单块）----
// 为什么 hc 的 ws 与 GDN/MoE 的单块 ws 处置不同：hc 边界的输出里有**两个跨层存活**的张量——
//   · `H'`（`WS_HCP`）= 更新后的 4 路多流残差流 ⇒ **下一层的 `hcH`**
//   · `OH[:,320:324)` = injection logits ⇒ **下一层的 `hcIj`**（下一层按行距 `OH_W` 直读）
// 若与 GDN/MoE 共用每层复用的单块 ws，这两个输出会在下一层启动时被覆盖。按层编址（48 × 8.7MB
// = 417 MB）后，**「H''/injection 传给下一层」零拷贝在结构上成立**：下一层的 hc 边界直接把
// 上一层 hc1 ws 的 `WS_HCP` / `WS_OH+OH_INJ*2` 当输入，不需要任何抽取 stage 或 host 搬运。
// （本 milestone 只交付**单层的四相位形态**与它的验证；48 层链的这条接线见 README 的未完成项。）
constexpr uint32_t HC_WS0_IN_LAYER = 0;
constexpr uint32_t HC_WS1_IN_LAYER = Align512(M15H::WS_BYTES);
constexpr uint32_t HC_WS_LAYER_STRIDE = Align512(HC_WS1_IN_LAYER + M15H::WS_BYTES);

static_assert(HC_WS1_IN_LAYER >= M15H::WS_BYTES, "hc 两个边界的 ws 不得重叠");
static_assert(HC_WS_LAYER_STRIDE >= HC_WS1_IN_LAYER + M15H::WS_BYTES, "hc 层槽必须容纳两块 ws");
// 层内 handoff 的 IJ 源 = 上一层 hc1 ws 的 `WS_OH + 320*2B`（20 个 32B 块）⇒ 必须 32B 对齐，
// 否则 InjwStage 的按行 32B 搬运（lift_hc_segment.py 替换类 4）会读到错位字节。
static_assert((M15H::WS_OH % 32u) == 0u && ((M15H::OH_INJ * 2u) % 32u) == 0u,
              "hc 的 IJ 源（WS_OH + OH_INJ）必须 32B 对齐");

// 逐段 GM 预算（README 的资源表就是这张表；数字由编译器从三段的偏移表算出，不是手抄）
struct SegBudget {
    const char* seg;
    uint32_t    bytes;
};
constexpr SegBudget GM_WS_TABLE[] = {
    {"GDN 段 ws（S1-S7 的 13 个中间张量；每层复用一块）", WS_GDN_BYTES},
    {"MoE 段 ws（S1-S10 的 32 个中间张量；每层复用一块）", WS_MOE_BYTES},
    {"hc 边界 #1（attn_hc）ws（8 个中间张量；**按层号编址**）", M15H::WS_BYTES},
    {"hc 边界 #2（mlp_hc）ws（同上，另一块）", M15H::WS_BYTES},
    {"末层全局 mixer 的 ws（M65；`WS_HCP` = multi_hidden、`WS_BLK` = sample_hidden）", M15H::WS_BYTES},
};

// ============================================================
// 1. MoE 段权重槽（每层一个槽；**48 层都有 MoE**，与 GDN 的 36 槽分开编号）
// ============================================================
// 为什么单独一个权重区而不是并进 M15Loop::W_STRIDE：attention 层**没有** linear_attn 权重
// （M25 的槽表刻意不给它们占位，省 12×116MB），但它们**有** MoE 权重 → 两套槽表按层号各自
// 编址（MoE 槽号 = 层号，0..47）。GDN 槽号仍是 GDN 序列序号（M15Loop::LayerSlot）。

constexpr uint32_t WAL = 512;
constexpr uint32_t WAlignUp(uint32_t x) { return (x + WAL - 1u) / WAL * WAL; }

// ------------------------------------------------------------
// 1a. M110（Wave A · A-2）：**逐项式子**（本文件 = 唯一 owner；B4 只消费、不得再定义第二份）
// ------------------------------------------------------------
// 为什么要写成 E 的函数：`MW_*` 原来把 `M15M::NUM_EXPERTS`（= 4，golden 缩形档）**烧进了数值**，
// 于是「换成真实档 E=512」就必须改 12 个常量值（改一个漏一个 ⇒ 槽内偏移静默错位）。
// 抽成式子后，**两档（decode E=4 / prefill E=512）由同一批式子导出**，数值各由 static_assert 钉死。
//
// 与 TOPK_MAX **无关**：权重槽是逐专家的（`[E, …]`），topk 只影响「一次算几个专家」⇒ 它落在
// `M15M` 的 `TOTAL_MAX = M_MAX*TOPK_MAX` 与 `SZ_*` 那一族（B4 的 `m15_moe_prefill_res.h` 范围）。
// 本节的 prefill 档只把 **E** 换成 512；topk=10 的落点在下面 §1c 显式登记。
constexpr uint32_t MOE_E_PREFILL = 512;      // 真实档：num_experts（docs/15 §1）
constexpr uint32_t MOE_TOPK_PREFILL = 10;    // 真实档：num_experts_per_tok（docs/15 §1）

constexpr uint32_t MwRouterBytes(uint32_t e) { return e * HIDDEN * 2; }                   // bf16[E, HIDDEN]
constexpr uint32_t MwSgateBytes() { return HIDDEN * 2; }                                  // bf16[HIDDEN]（共享门，与 E 无关）
constexpr uint32_t MwWguBytes(uint32_t e) { return e * M15M::GU_N * (HIDDEN / 2); }       // u8[E, GU_N, HIDDEN/2]
constexpr uint32_t MwSguBytes(uint32_t e) { return e * M15M::GU_N * M15M::GU_SCALE_STRIDE; }
constexpr uint32_t MwWdnBytes(uint32_t e) { return e * HIDDEN * (M15M::INTER / 2); }      // u8[E, HIDDEN, INTER/2]
constexpr uint32_t MwSdnBytes(uint32_t e) { return e * HIDDEN * (M15M::INTER / M15M::GROUP); }
// 共享专家（`*_SHD`）恒为 1 个专家 ⇒ 与 E 无关，式子不带参
constexpr uint32_t MwWguShdBytes() { return M15M::GU_N * (HIDDEN / 2); }
constexpr uint32_t MwSguShdBytes() { return M15M::GU_N * M15M::GU_SCALE_STRIDE; }
constexpr uint32_t MwWdnShdBytes() { return HIDDEN * (M15M::INTER / 2); }
constexpr uint32_t MwSdnShdBytes() { return HIDDEN * (M15M::INTER / M15M::GROUP); }
constexpr uint32_t MwG1Bytes() { return HIDDEN * 2; }                                     // MoE 段 S1 的 gamma1
constexpr uint32_t MwG2Bytes() { return HIDDEN * 2; }

// 一个槽的完整布局（每层一个槽；偏移顺排、512B 对齐 —— 对齐规则原样保留）
struct MoeWSlot {
    uint32_t routerBytes;
    uint32_t sgateBytes;
    uint32_t wguBytes;
    uint32_t sguBytes;
    uint32_t wdnBytes;
    uint32_t sdnBytes;
    uint32_t routerOff;
    uint32_t sgateOff;
    uint32_t wguOff;
    uint32_t sguOff;
    uint32_t wdnOff;
    uint32_t sdnOff;
    uint32_t wgushdOff;
    uint32_t sgushdOff;
    uint32_t wdnshdOff;
    uint32_t sdnshdOff;
    uint32_t g1Off;
    uint32_t g2Off;
    uint32_t stride;      // 每层槽字节数（512B 对齐后的整槽）
};

constexpr MoeWSlot MoeWSlotOf(uint32_t e)
{
    MoeWSlot s{};
    s.routerBytes = MwRouterBytes(e);
    s.sgateBytes = MwSgateBytes();
    s.wguBytes = MwWguBytes(e);
    s.sguBytes = MwSguBytes(e);
    s.wdnBytes = MwWdnBytes(e);
    s.sdnBytes = MwSdnBytes(e);
    s.routerOff = 0;
    s.sgateOff = s.routerOff + s.routerBytes;
    s.wguOff = WAlignUp(s.sgateOff + s.sgateBytes);
    s.sguOff = s.wguOff + s.wguBytes;
    s.wdnOff = WAlignUp(s.sguOff + s.sguBytes);
    s.sdnOff = s.wdnOff + s.wdnBytes;
    s.wgushdOff = WAlignUp(s.sdnOff + s.sdnBytes);
    s.sgushdOff = s.wgushdOff + MwWguShdBytes();
    s.wdnshdOff = WAlignUp(s.sgushdOff + MwSguShdBytes());
    s.sdnshdOff = s.wdnshdOff + MwWdnShdBytes();
    s.g1Off = WAlignUp(s.sdnshdOff + MwSdnShdBytes());
    s.g2Off = s.g1Off + MwG1Bytes();
    s.stride = WAlignUp(s.g2Off + MwG2Bytes());
    return s;
}

// ---- decode 档（E = M15M::NUM_EXPERTS = 4）：**数值一字不动**（README 的资源表 / host arena /
//      manifest 的 `moe_*` 张量字节数都按它算 —— 改它 = 破坏 `runs=all` 的零回归）----
constexpr MoeWSlot MOE_W_SLOT_DECODE = MoeWSlotOf(M15M::NUM_EXPERTS);
constexpr uint32_t MW_ROUTER_BYTES = MOE_W_SLOT_DECODE.routerBytes;                    // 20,480
constexpr uint32_t MW_SGATE_BYTES = MOE_W_SLOT_DECODE.sgateBytes;                      // 5,120
constexpr uint32_t MW_WGU_BYTES = MOE_W_SLOT_DECODE.wguBytes;                          // 6,553,600
constexpr uint32_t MW_SGU_BYTES = MOE_W_SLOT_DECODE.sguBytes;                          // 409,600
constexpr uint32_t MW_WDN_BYTES = MOE_W_SLOT_DECODE.wdnBytes;                          // 3,276,800
constexpr uint32_t MW_SDN_BYTES = MOE_W_SLOT_DECODE.sdnBytes;                          // 204,800
constexpr uint32_t MW_WGUSHD_BYTES = MwWguShdBytes();                                  // 1,638,400
constexpr uint32_t MW_SGUSHD_BYTES = MwSguShdBytes();                                  // 102,400
constexpr uint32_t MW_WDNSHD_BYTES = MwWdnShdBytes();                                  // 819,200
constexpr uint32_t MW_SDNSHD_BYTES = MwSdnShdBytes();                                  // 51,200
constexpr uint32_t MW_G1_BYTES = MwG1Bytes();                                          // 5,120
constexpr uint32_t MW_G2_BYTES = MwG2Bytes();                                          // 5,120

constexpr uint32_t MW_ROUTER_OFF = MOE_W_SLOT_DECODE.routerOff;
constexpr uint32_t MW_SGATE_OFF = MOE_W_SLOT_DECODE.sgateOff;
constexpr uint32_t MW_WGU_OFF = MOE_W_SLOT_DECODE.wguOff;
constexpr uint32_t MW_SGU_OFF = MOE_W_SLOT_DECODE.sguOff;
constexpr uint32_t MW_WDN_OFF = MOE_W_SLOT_DECODE.wdnOff;
constexpr uint32_t MW_SDN_OFF = MOE_W_SLOT_DECODE.sdnOff;
constexpr uint32_t MW_WGUSHD_OFF = MOE_W_SLOT_DECODE.wgushdOff;
constexpr uint32_t MW_SGUSHD_OFF = MOE_W_SLOT_DECODE.sgushdOff;
constexpr uint32_t MW_WDNSHD_OFF = MOE_W_SLOT_DECODE.wdnshdOff;
constexpr uint32_t MW_SDNSHD_OFF = MOE_W_SLOT_DECODE.sdnshdOff;
constexpr uint32_t MW_G1_OFF = MOE_W_SLOT_DECODE.g1Off;
constexpr uint32_t MW_G2_OFF = MOE_W_SLOT_DECODE.g2Off;
constexpr uint32_t MOE_W_STRIDE = MOE_W_SLOT_DECODE.stride;

static_assert(MW_WGU_BYTES == 6553600u, "decode 档 MW_WGU_BYTES 变了（README 资源表 / host arena 依赖它）");
static_assert(MOE_W_STRIDE == 13091840u, "decode 档 MOE_W_STRIDE 变了（README §资源表：13,091,840 B/层）");
static_assert(MW_ROUTER_BYTES == 20480u && MW_SGATE_BYTES == 5120u && MW_SGU_BYTES == 409600u &&
                  MW_WDN_BYTES == 3276800u && MW_SDN_BYTES == 204800u,
              "decode 档的某个 MW_* 数值变了（清单见 m15_layer_resources.h §1a）");
static_assert(MW_ROUTER_OFF == 0u && MW_SGATE_OFF == 20480u && MW_WGU_OFF == 25600u && MW_SGU_OFF == 6579200u &&
                  MW_WDN_OFF == 6988800u && MW_SDN_OFF == 10265600u,
              "decode 档的某个 MW_* 偏移变了（槽内顺排 + 512B 对齐的逐项读数）");

// ---- prefill 档（E = 512 / topk = 10，真实 shape；`docs/15` §1 的 checkpoint 字段）----
// 这是 Wave A 交给 B4 的**唯一一份** E=512 槽表；B4 只准引用，不得在 `m15_moe_prefill_res.h` 里再定义。
constexpr MoeWSlot MOE_W_SLOT_PREFILL = MoeWSlotOf(MOE_E_PREFILL);
constexpr uint32_t MWP_ROUTER_BYTES = MOE_W_SLOT_PREFILL.routerBytes;                   // 2,621,440
constexpr uint32_t MWP_SGATE_BYTES = MOE_W_SLOT_PREFILL.sgateBytes;                     // 5,120
constexpr uint32_t MWP_WGU_BYTES = MOE_W_SLOT_PREFILL.wguBytes;                         // 838,860,800
constexpr uint32_t MWP_SGU_BYTES = MOE_W_SLOT_PREFILL.sguBytes;                         // 52,428,800
constexpr uint32_t MWP_WDN_BYTES = MOE_W_SLOT_PREFILL.wdnBytes;                         // 419,430,400
constexpr uint32_t MWP_SDN_BYTES = MOE_W_SLOT_PREFILL.sdnBytes;                         // 26,214,400
constexpr uint32_t MWP_ROUTER_OFF = MOE_W_SLOT_PREFILL.routerOff;                       // 0
constexpr uint32_t MWP_SGATE_OFF = MOE_W_SLOT_PREFILL.sgateOff;                         // 2,621,440
constexpr uint32_t MWP_WGU_OFF = MOE_W_SLOT_PREFILL.wguOff;                             // 2,626,560
constexpr uint32_t MWP_SGU_OFF = MOE_W_SLOT_PREFILL.sguOff;                             // 841,487,360
constexpr uint32_t MWP_WDN_OFF = MOE_W_SLOT_PREFILL.wdnOff;                             // 893,916,160
constexpr uint32_t MWP_SDN_OFF = MOE_W_SLOT_PREFILL.sdnOff;                             // 1,313,346,560
constexpr uint32_t MWP_WGUSHD_OFF = MOE_W_SLOT_PREFILL.wgushdOff;                       // 1,339,560,960
constexpr uint32_t MWP_SGUSHD_OFF = MOE_W_SLOT_PREFILL.sgushdOff;                       // 1,341,199,360
constexpr uint32_t MWP_WDNSHD_OFF = MOE_W_SLOT_PREFILL.wdnshdOff;                       // 1,341,301,760
constexpr uint32_t MWP_SDNSHD_OFF = MOE_W_SLOT_PREFILL.sdnshdOff;                       // 1,342,120,960
constexpr uint32_t MWP_G1_OFF = MOE_W_SLOT_PREFILL.g1Off;                               // 1,342,172,160
constexpr uint32_t MWP_G2_OFF = MOE_W_SLOT_PREFILL.g2Off;                               // 1,342,177,280
constexpr uint32_t MOE_W_PREFILL_STRIDE = MOE_W_SLOT_PREFILL.stride;                    // 1,342,182,400

static_assert(MWP_WGU_BYTES == 838860800u, "E=512 的 MW_WGU_BYTES ≠ 838,860,800（docs/15 M103-2.1 的 A-2 读数）");
static_assert(MWP_WGU_BYTES == 128u * MW_WGU_BYTES, "E=512 的 gate_up 槽相对 E=4 不是 128×");
static_assert(MOE_W_PREFILL_STRIDE == 1342182400u, "E=512 的槽 stride ≠ 1,342,182,400 B");
static_assert(MOE_W_PREFILL_STRIDE > MOE_W_STRIDE, "prefill 槽必须比 decode 槽大");
// 每项偏移都必须是 512B 对齐（host 的 pread 落点 + 设备侧的 SetGlobalBuffer 都依赖它）
static_assert(MWP_SGATE_OFF % WAL == 0u && MWP_WGU_OFF % WAL == 0u && MWP_SGU_OFF % WAL == 0u &&
                  MWP_WDN_OFF % WAL == 0u && MWP_SDN_OFF % WAL == 0u && MWP_WGUSHD_OFF % WAL == 0u,
              "prefill 槽内某偏移不是 512B 对齐");
// 槽内各段必须顺排、不重叠（同一段内的偏序检查；越界一类错误在 Wave A 这一层就能变红）
static_assert(MWP_SGATE_OFF >= MWP_ROUTER_OFF + MWP_ROUTER_BYTES, "prefill 槽：SGATE 与 ROUTER 重叠");
static_assert(MWP_WGU_OFF >= MWP_SGATE_OFF + MWP_SGATE_BYTES, "prefill 槽：WGU 与 SGATE 重叠");
static_assert(MWP_SGU_OFF >= MWP_WGU_OFF + MWP_WGU_BYTES, "prefill 槽：SGU 与 WGU 重叠");
static_assert(MWP_WDN_OFF >= MWP_SGU_OFF + MWP_SGU_BYTES, "prefill 槽：WDN 与 SGU 重叠");
static_assert(MWP_SDN_OFF >= MWP_WDN_OFF + MWP_WDN_BYTES, "prefill 槽：SDN 与 WDN 重叠");
static_assert(MWP_SDNSHD_OFF + MW_SDNSHD_BYTES <= MWP_G1_OFF, "prefill 槽：共享 down 的 scale 越进 gamma1");
static_assert(MWP_G2_OFF + MW_G2_BYTES <= MOE_W_PREFILL_STRIDE, "prefill 槽：gamma2 越出整槽 stride");
// host arena 的账（**只给账，不改 host**：host 侧改动不在本 mission 的 scope，见 §1d）
constexpr uint64_t MOE_ARENA_DECODE_BYTES = static_cast<uint64_t>(48) * MOE_W_STRIDE;          // 628,408,320 B
constexpr uint64_t MOE_ARENA_PREFILL_BYTES = static_cast<uint64_t>(48) * MOE_W_PREFILL_STRIDE;  // 64,424,755,200 B
static_assert(MOE_ARENA_PREFILL_BYTES == 64424755200ull, "E=512 的 48 层 MoE 权重区账变了（48×1,342,182,400）");
// 比值的准确口径：**逐槽是 102.52×**（1,342,182,400 / 13,091,840），不是 128× —— 128× 只在
// `MW_WGU_BYTES` 那一项上成立（专家数 4→512）；槽里还有一堆与 E 无关的共享专家项。
static_assert(MOE_ARENA_PREFILL_BYTES > 100ull * MOE_ARENA_DECODE_BYTES &&
                  MOE_ARENA_PREFILL_BYTES < 103ull * MOE_ARENA_DECODE_BYTES,
              "prefill/decode 的 MoE 权重区比不在 (100×, 103×) 内（实算 102.52×）");

// ---- 1b. 真实 shape 的两个常量 + 与 M15M 的跨文件见证 ----
constexpr uint32_t MOE_E_REAL_CHECK = MOE_E_PREFILL;
static_assert(MOE_E_REAL_CHECK == 512u && MOE_TOPK_PREFILL == 10u,
              "真实 shape 必须是 E=512 / topk=10（docs/15 §1 的 checkpoint 字段）");
static_assert(M15M::TOPK_MAX <= 32u, "M15M 的 top-k 归并树上界（TOPK_MAX<=32）被破坏");
static_assert(M15M::RT_ROWL >= MOE_E_PREFILL,
              "E=512 装不进 top-k 的行宽 RT_ROWL（M107 已核过一次；这里是第二次见证：行宽必须覆盖全部 E 个 logit）");
static_assert(M15M::TOPK_MAX <= M15M::RT_WROW, "top-k 结果放不进 IDS/WS 的 staging 行距");

// ---- 1c. topk = 10 的落点（**不在本文件**，显式登记以免"只改了一半"）----
// 权重槽是逐专家的（`[E, …]`），**与 TOPK_MAX 无关**（§1a 开头已说明）。topk 真正改动的量是：
//   · `M15M::TOTAL_MAX = M_MAX * TOPK_MAX`（紧凑槽上界 Σt_e）
//   · `M15M::SZ_IDS / SZ_W / SZ_INV / SZ_WTK / SZ_PERM / SZ_SGATE` 一族
//   · `U_STAGE_B = (TOPK_MAX + 1) * HIDDEN * 2 + 64`（unpermute 的每 stage 字节数）
// ⇒ 真实档要改的是 **B4 的 `m15_moe_prefill_res.h`**（它自己声明"只放 prefill 段自用的常量"）
//    与 `m15_moe_resources.h` 的 `NUM_EXPERTS / TOPK_MAX`（**那个文件是 decode 档，不在本 mission
//    的 scope ⇒ 未改动**，见 §1d 的清单）。

// ---- 1d. host arena 的账 + **为什么本 mission 不原地把 decode 档换成 E=512** ----
// 只给账、不改 host：`m15_moe_host.h`（`W.wGu.resize(MW_WGU_BYTES)` + 按 manifest 逐字节核）与
// `m15_layer_loop.asc`（`moeArenaBytes = NL * MOE_W_STRIDE`）都在本 mission 的 scope 之外。
// 账（实算，见上面两条 static_assert）：
//   decode 档 48 层 = 48 ×      13,091,840 =   628,408,320 B ≈  0.63 GB（现状：manifest 按 4 专家切片）
//   prefill 档 48 层 = 48 ×   1,342,182,400 = 64,424,755,200 B ≈ 64.42 GB（**host 侧装不下**：
//                                                                  本机 cgroup 上限 34,359,738,368 B）
// ⇒ 本 mission 交付的是「**同一 owner 文件里的两套定尺**」（decode 原值不动 + prefill 新表）；
//    真正的切换（host arena 分片/惰性装载 + manifest 重切 + `NUM_EXPERTS/TOPK_MAX`）是**清单外的
//    依赖 mission**，已 `TowerSend` 报塔。**这不是把 A-2 缩形**：E=512 的每个数值都给了断言。

// 权重来源角色名（与 weights_manifest.txt 的 role 字段一一对应；host 装载按它 pread）
constexpr const char* MW_ROLE_ROUTER = "moe_router_w";      // [NUM_EXPERTS, HIDDEN] bf16 ← ckpt mlp.gate.weight 行 0..E-1
constexpr const char* MW_ROLE_SGATE = "moe_sgate_w";        // [HIDDEN] bf16 ← ckpt mlp.shared_expert_gate.weight 行 0
constexpr const char* MW_ROLE_WGU = "moe_experts_gate_up";  // [E, GU_N, HIDDEN/2] u8 ← ckpt mlp.experts.gate_up_proj 专家 0..E-1
constexpr const char* MW_ROLE_SGU = "moe_experts_gate_up_scale";
constexpr const char* MW_ROLE_WDN = "moe_experts_down";
constexpr const char* MW_ROLE_SDN = "moe_experts_down_scale";
constexpr const char* MW_ROLE_SH_GATE = "moe_shared_gate";              // [INTER, HIDDEN/2] u8
constexpr const char* MW_ROLE_SH_GATE_S = "moe_shared_gate_scale";      // [INTER, HIDDEN/32]
constexpr const char* MW_ROLE_SH_UP = "moe_shared_up";                  // [INTER, HIDDEN/2]
constexpr const char* MW_ROLE_SH_UP_S = "moe_shared_up_scale";
constexpr const char* MW_ROLE_WGUSHD = "moe_shared_gate_up";   // device 槽内角色名：host 行拼接 gate||up
constexpr const char* MW_ROLE_SGUSHD = "moe_shared_gate_up_scale";
constexpr const char* MW_ROLE_WDNSHD = "moe_shared_down";
constexpr const char* MW_ROLE_SDNSHD = "moe_shared_down_scale";

// ============================================================
// 1b. hc 段权重槽（M58 新增；每层**两个**边界：attn_hc 与 mlp_hc）
// ============================================================
// 布局与 checkpoint **逐字节一致**（不做任何离线转换；m20 的「权重表示一行不转」契约）：
//   hc_norm  [HYPER]        bf16  ← `layers.L.{attn,mlp}_hyper_connection.hc_norm.weight`
//   wdown    [LOWRANK,HYPER] bf16 ← `…input_mix_weight_down.weight`（行主序，行 = 低秩分量）
//   wup      [UP_N, UP_K]   bf16  ← `…input_mix_weight_up.weight`
//   wInj     [HC_WINJ_ROWS, HYPER] bf16 ← `…block_inject_weight.weight`（checkpoint 只有
//            INJ_N=4 行）**行 [4,16) 必须由 host 置零**（m20 §5.1 的 host 契约：cube 的
//            N 分形以 16 行为单位，最后一个 N-tile 读 16 行 ⇒ L0C 的列 [4,16) 由 0·x 精确为 0，
//            OH 的 padding 列 [324,336) 恒为 0 ⇒ 输出字节确定性成立）。
constexpr uint32_t HC_WINJ_ROWS = 16;

constexpr uint32_t HW_NORM_BYTES = M15H::HYPER * 2;                       // 20,480
constexpr uint32_t HW_WDOWN_BYTES = M15H::LOWRANK * M15H::HYPER * 2;      // 6,553,600
constexpr uint32_t HW_WINJ_BYTES = HC_WINJ_ROWS * M15H::HYPER * 2;        // 327,680
constexpr uint32_t HW_WUP_BYTES = M15H::UP_N * M15H::UP_K * 2;            // 6,553,600

constexpr uint32_t HW_NORM_OFF = 0;
constexpr uint32_t HW_WDOWN_OFF = WAlignUp(HW_NORM_OFF + HW_NORM_BYTES);
constexpr uint32_t HW_WINJ_OFF = HW_WDOWN_OFF + HW_WDOWN_BYTES;
constexpr uint32_t HW_WUP_OFF = WAlignUp(HW_WINJ_OFF + HW_WINJ_BYTES);
constexpr uint32_t HC_W_STRIDE = WAlignUp(HW_WUP_OFF + HW_WUP_BYTES);     // 一个 hc 边界的槽
// 一层两个边界：attn_hc @0、mlp_hc @HC_W_STRIDE（槽号 = 层号，0..47）
constexpr uint32_t HC_LAYER_W_STRIDE = 2 * HC_W_STRIDE;
constexpr uint32_t HC_ATTN_W_OFF = 0;                                     // 层槽内 attn_hc 的起点
constexpr uint32_t HC_MLP_W_OFF = HC_W_STRIDE;                            // 层槽内 mlp_hc 的起点

// ---- M65：末层全局 mixer（`hyper_connection_mixer`）的权重槽 ----
// **槽号 = 48 个层槽之后的第 49 个槽**，槽内布局与本表的 hc 边界槽**完全相同**（沿用 `HC_ATTN_W_OFF`
// 那一格），差别只有一处：checkpoint 的 `hyper_connection_mixer` **没有** `block_inject_weight`
// （`V-N:model.py:612` 显式把它从权重里丢弃）⇒ host 把注入区 16 行**全零**装进去；而
// `MODE_FINAL_MIX` 的 S3 只跑 lora 的 `LOWRANK/BASE_N = 2` 个 N-tile，那 16 行**从不被读**。
constexpr uint32_t HC_GMIX_LAYER_SLOT = 48;                // = M15Loop::NL（host 侧有一条 static_assert 钉住）
// **每槽 = `HC_LAYER_W_STRIDE` = 2 × HC_W_STRIDE = 26,910,720 B = 26.91 MB**（一个层槽含 2 个边界
// 子槽）；49 槽 = 48 个层槽 + 1 个全局 mixer 槽 ⇒ 权重区合计 49 × 26.91 MB = 1318.63 MB。
// 全局 mixer 槽只填 norm/down/up 共 13.13 MB 数据，其余字节保持全零（槽 stride 不变）。
constexpr uint32_t HC_W_SLOTS = HC_GMIX_LAYER_SLOT + 1;    // 49

// ---- M65：hc ws 的槽数 ----
//   0 .. 47     每层两块（边界 #1 / #2），**按层号编址**（跨层 handoff 零拷贝的前提）
//   48 .. 50    验证 H 用的 3 块 scratch（M58 原样）
//   51          末层全局 mixer 的 ws（它的两个输出 `WS_HCP`/`WS_BLK` = multi_hidden / sample_hidden）
constexpr uint32_t HC_WS_GMIX_SLOT = 48 + 3;               // 51
constexpr uint32_t HC_WS_SLOTS = HC_WS_GMIX_SLOT + 1u;     // 52

// 权重来源角色名（与 weights_manifest.txt 的 role 字段一一对应）
constexpr const char* HW_ROLE_ATTN_NORM = "attn_hc_norm";
constexpr const char* HW_ROLE_ATTN_DOWN = "attn_hc_down";
constexpr const char* HW_ROLE_ATTN_INJ = "attn_hc_inj";
constexpr const char* HW_ROLE_ATTN_UP = "attn_hc_up";
constexpr const char* HW_ROLE_MLP_NORM = "mlp_hc_norm";
constexpr const char* HW_ROLE_MLP_DOWN = "mlp_hc_down";
constexpr const char* HW_ROLE_MLP_INJ = "mlp_hc_inj";
constexpr const char* HW_ROLE_MLP_UP = "mlp_hc_up";
// M65：末层全局 mixer 的三个 checkpoint 张量（`model.language_model.hyper_connection_mixer.*`）
constexpr const char* HW_ROLE_GM_NORM = "gmixer_norm";
constexpr const char* HW_ROLE_GM_DOWN = "gmixer_down";
constexpr const char* HW_ROLE_GM_UP = "gmixer_up";

// ============================================================
// 2. UB 预算（**四相位同址叠放**，见 §3 的合法性论证）
// ============================================================
// 相位 H1/H2（hc 段）窗：PERSIST+staging+五个段窗 **并列** [0, M15H::UB_BYTES_USED=89472)
// 相位 A（GDN 段）窗：PERSIST [0, UB_SEG) + 四个段窗 [UB_SEG, UB_RC_END)
// 相位 B（MoE 段）窗：PERSIST [0, UB_VEC) + 段窗 [UB_VEC, UB_RT_END)
// 峰值 = max(四相位)，**不是求和**（M33/discs15 §5.3 第 7 条要求的正是「每 mode 独立断言峰值」）。

constexpr uint32_t UB_TOTAL_BYTES = 248 * 1024;          // GetCoreMemSize(UB)
constexpr uint32_t UB_PEAK_PHASE_H = M15H::UB_PEAK;      // hc 段（m15_hc_resources.h §2）
constexpr uint32_t UB_PEAK_PHASE_A = UB_RC_END;          // GDN 段（m15_gdn_resources.h §4）
constexpr uint32_t UB_PEAK_PHASE_B = M15M::UB_RT_END;    // MoE 段（m15_moe_resources.h §4）
constexpr uint32_t UB_PEAK_FUSED = (UB_PEAK_PHASE_A > UB_PEAK_PHASE_B)
                                       ? ((UB_PEAK_PHASE_A > UB_PEAK_PHASE_H) ? UB_PEAK_PHASE_A : UB_PEAK_PHASE_H)
                                       : ((UB_PEAK_PHASE_B > UB_PEAK_PHASE_H) ? UB_PEAK_PHASE_B : UB_PEAK_PHASE_H);

static_assert(UB_PEAK_FUSED <= UB_TOTAL_BYTES, "融合 kernel 的 UB 峰值超出 248KB");
// hc 段的整个 UB 区从 0 起算 ⇒ 它与 GDN/MoE 的 PERSIST 区**物理重叠**。合法性由段序保证：
// H1 在 GDN 相位之前、H2 在 GDN/MoE 之外（H2 之后才是 MoE 相位，MoE 的 PERSIST 在
// MoE.ProcessAiv 开头才写）——即「没有任何张量跨相位存活」，每个相位交界都有 PipeBarrier<PIPE_ALL>。
static_assert(M15H::UB_PEAK == M15H::UB_BYTES_USED, "hc 相位峰值必须等于它的 UB 段窗末尾");
static_assert(M15M::UB_VEC >= 32768u, "MoE 段 PERSIST 区必须容纳 gamma fp32 两份（20480B）");

// ============================================================
// 3. L1 / L0C 预算（四相位同址叠放；相位内各自独占）
// ============================================================
//   L1：GDN 段 A 区 [0, 16384) / B 区 [262144, 303104)；hc 段同构（A 区 8KB×2、B 区 20KB×2，
//       峰值 303104）；MoE 段 A 区 [0, 8704) / B 区 [262144, 296960)。三段区间重叠但**相位互斥**，
//       峰值 = max(三段) ≤ 512KB。
//   L0C：三段都从 offset 0 起用一整块 —— GDN 段 64×160 fp32 = 40KB、hc 段同 40KB、
//       MoE 段 64×256 fp32 = 64KB。峰值 64KB ≤ L0C 容量（CO1 256KB）。
constexpr uint32_t L1_PEAK_PHASE_H = M15H::L1_PEAK;                                    // 303,104
constexpr uint32_t L1_PEAK_PHASE_A = L1_B_REGION + 2 * L1_B_ELEMS * 2;                 // 303,104
constexpr uint32_t L1_PEAK_PHASE_B = M15M::L1_OFF_B0 + 2 * M15M::L1_B_DATA + 2 * M15M::L1_B_SCAL;  // 296,960
constexpr uint32_t L1_PEAK_FUSED = (L1_PEAK_PHASE_A > L1_PEAK_PHASE_B)
                                       ? ((L1_PEAK_PHASE_A > L1_PEAK_PHASE_H) ? L1_PEAK_PHASE_A : L1_PEAK_PHASE_H)
                                       : ((L1_PEAK_PHASE_B > L1_PEAK_PHASE_H) ? L1_PEAK_PHASE_B : L1_PEAK_PHASE_H);
static_assert(L1_PEAK_FUSED <= L1_BYTES_TOTAL, "融合 kernel 的 L1 峰值超出 512KB");

constexpr uint32_t L0C_PEAK_PHASE_H = M15H::L0C_PEAK;                         // 40,960
constexpr uint32_t L0C_PEAK_PHASE_A = BASE_M * BASE_N * 4;                    // 40,960
constexpr uint32_t L0C_PEAK_PHASE_B = M15M::BASE_M * M15M::BASE_N * 4;        // 65,536
constexpr uint32_t L0C_PEAK_FUSED = (L0C_PEAK_PHASE_A > L0C_PEAK_PHASE_B)
                                        ? ((L0C_PEAK_PHASE_A > L0C_PEAK_PHASE_H) ? L0C_PEAK_PHASE_A : L0C_PEAK_PHASE_H)
                                        : ((L0C_PEAK_PHASE_B > L0C_PEAK_PHASE_H) ? L0C_PEAK_PHASE_B : L0C_PEAK_PHASE_H);
constexpr uint32_t L0C_TOTAL_BYTES = 256 * 1024;
static_assert(L0C_PEAK_FUSED <= L0C_TOTAL_BYTES, "融合 kernel 的 L0C 峰值超出 256KB");

// ------------------------------------------------------------
// 3c. M110（Wave A）：**prefill 的 UB / L1 / L0C 峰值断言位**（数值由 Wave C 填）
// M132（Wave C，第一种 mode）：B1/B4/B5 的 5 个槽已填实数；余 3 个 = KIND_ATTN 的 UB/L1/L0C
// （B2+B3 本轮不接，见下面槽定义处的说明）。`PfPeakUnfilled() == 3`。
// ------------------------------------------------------------
// 口径（M103-2.3 Wave C 第一条 + M33/discs15 §5.3 第 7 条）：相位 A 的 UB 窗对 GDN-prefill 与
// attention-prefill 是**互斥**的（KIND_GDN / KIND_ATTN 不同层）⇒ prefill 的 UB 峰值
// = max(GDN-prefill 窗, attention-prefill 窗, hc 窗, MoE 窗)，**不是求和**；L1 / L0C 同理。
//
// **为什么用哨兵而不是填 0**：填 0 会让 `static_assert(x <= 248KB)` 恒真 —— 那是空洞断言
// （`docs/17` §4 的非空洞性纪律）。这里改成「**未填 = 0xFFFFFFFF 哨兵**，且**只有当整档被打开时
// 才要求填齐**」：`PREFILL_WIRED = 1` 而任一槽仍是哨兵 ⇒ **编译期直接 FAIL**。
// ⇒ Wave C 想把 prefill 接线打开，就**必须**先把这 8 个槽填上实数，填漏一个都编不过。
constexpr uint32_t PF_PEAK_UNSET = 0xFFFFFFFFu;
// **M132（Wave C）保持 0**：B1/B4/B5 已接线，但 **attention 模式（B2+B3）待 B3（M101）合入后再接**
// ⇒ 它的 3 个峰值槽仍未填 ⇒ 接线门（本节末尾的 static_assert）不允许开。这里必须与 `.asc` 的
// `M15_PREFILL_WIRE` 缺省一致（0 = 关）——这是"默认关闭 = decode 零回归"的编译期锚点。
constexpr uint32_t PREFILL_WIRED = 0;
// prefill 入口里"该层没有 attention 序号"的哨兵（= `M15Loop::NO_SLOT`；host 侧有断言钉住，见 .asc）
constexpr uint32_t PF_LAYER_K_NONE = 0xFFFFFFFFu;

// ---- M132（Wave C）：填 B1 / B4 / B5 三种 mode 的峰值；KIND_ATTN 的 3 槽按本轮分工留哨兵 ----
// 每个数 = 段设备头里那条 `static_assert(... <= 预算)` 的**被断言量本身**（不是旁证的手算值），
// 行尾给 `文件:行` 出处；文件:行 取本提交时的行号。`m15_layer_kernel.h` 里另有 static_assert 把
// 这三个数与各自的设备常量逐一对上（跨文件机器见证，数字漂了就编不过）。
// **为什么 ATTENTION 的 3 个槽仍留哨兵**：KIND_ATTN 的段体（B2 前端 + B3 稠密 core）本轮不接
// —— B3（M101）尚未合入，其融合清单在未合入分支上，按 mission 的硬约束不得引用、不得提前占位。
// ⇒ `PfPeakUnfilled()` 从 8 降到 3，余下的 3 个**全部且只是** attention 的 UB/L1/L0C。
constexpr uint32_t PF_UB_BYTES_GDN = 248064u;         // B1：= M15G::GP_UB_BYTES（m15_gdn_resources.h:384；m23 README §3(c):102）
constexpr uint32_t PF_UB_BYTES_ATTN = PF_PEAK_UNSET;  // B2+B3：KIND_ATTN 本轮不接（B3 未合入，见上）
constexpr uint32_t PF_UB_BYTES_HC = 94592u;           // B5：= M15H::HcPF::UB_PEAK_PF = UB_BYTES_USED(89472)+ROW_HID(5120)（m15_hc_prefill.h:152；m27 README §5(c):309）
constexpr uint32_t PF_UB_BYTES_MOE = 149632u;         // B4：= M15PFR::PF_UB_PEAK（m15_moe_prefill_res.h:200；m26 README §1(c):115/:128）
constexpr uint32_t PF_L1_BYTES_GDN = 491520u;         // B1：= M15G::GP_L1_BYTES（m15_gdn_resources.h:337；m23 README §3(c):103）
constexpr uint32_t PF_L1_BYTES_ATTN = PF_PEAK_UNSET;  // B2+B3：本轮不接
constexpr uint32_t PF_L0C_BYTES_GDN = 65536u;         // B1：= M15G::GP_L0C_BYTES（m15_gdn_resources.h:351；m23 README §3(c):105）
constexpr uint32_t PF_L0C_BYTES_ATTN = PF_PEAK_UNSET; // B2+B3：本轮不接

struct PfPeakSlot {
    const char* what;
    uint32_t    bytes;
    uint32_t    budget;
};
constexpr PfPeakSlot PF_PEAK_TABLE[] = {
    {"UB/相位A/GDN-prefill", PF_UB_BYTES_GDN, UB_TOTAL_BYTES},
    {"UB/相位A/attn-prefill", PF_UB_BYTES_ATTN, UB_TOTAL_BYTES},
    {"UB/相位H/hc-prefill", PF_UB_BYTES_HC, UB_TOTAL_BYTES},
    {"UB/相位B/MoE-prefill", PF_UB_BYTES_MOE, UB_TOTAL_BYTES},
    {"L1/GDN-prefill", PF_L1_BYTES_GDN, L1_BYTES_TOTAL},
    {"L1/attn-prefill", PF_L1_BYTES_ATTN, L1_BYTES_TOTAL},
    {"L0C/GDN-prefill", PF_L0C_BYTES_GDN, L0C_TOTAL_BYTES},
    {"L0C/attn-prefill", PF_L0C_BYTES_ATTN, L0C_TOTAL_BYTES},
};
constexpr uint32_t PF_PEAK_N = sizeof(PF_PEAK_TABLE) / sizeof(PF_PEAK_TABLE[0]);
static_assert(PF_PEAK_N == 8u, "prefill 峰值断言位的条数变了（Wave C 的收口清单按它遍历）");

// 已填的必须在预算内（未填的按哨兵跳过；**已填即受检**）
constexpr bool PfPeaksOk()
{
    for (uint32_t i = 0; i < PF_PEAK_N; ++i) {
        if (PF_PEAK_TABLE[i].bytes != PF_PEAK_UNSET && PF_PEAK_TABLE[i].bytes > PF_PEAK_TABLE[i].budget) {
            return false;
        }
    }
    return true;
}
// 未填的槽数（0 = 全填齐）
constexpr uint32_t PfPeakUnfilled()
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < PF_PEAK_N; ++i) {
        n += (PF_PEAK_TABLE[i].bytes == PF_PEAK_UNSET) ? 1u : 0u;
    }
    return n;
}
static_assert(PfPeaksOk(), "prefill 的某个峰值断言槽已填但超预算（UB 248KB / L1 512KB / L0C 256KB）");
// **接线门**：打开 prefill 之前必须把 8 个槽填齐（这是本节的判据，不是注释）
static_assert(!(PREFILL_WIRED != 0u && PfPeakUnfilled() != 0u),
              "PREFILL_WIRED=1 但 prefill 峰值槽未填齐：Wave C 必须先填 PF_{UB,L1,L0C}_* 再开接线");


// ------------------------------------------------------------
// 3c-bis. M151：prefill **prolog 链路**（S2 in_proj + S3 conv1d/l2norm/gating）的登记
// ------------------------------------------------------------
// 形态：prolog 的两个段体是**独立 `__global__`**（复用 m11/m9 donor；见 m15_prefill_prolog.h），
// 由 host 在相位 A 之前按同一条 stream 串行起动（H_PfGdnWired → H_PfPrologRun）。因此：
//   · **不新增 flagId**：段间次序由 stream 给；两段内部各自是 BufferID 同步（S2 = OProjGemm 的
//     OpAcq/OpRls；S3 = m9 的 GetBufInternal/RlsBufInternal）⇒ 不需要跨核 id，`FLAG_SEQ_*` 不动。
//   · **不动 `PF_PEAK_TABLE` / `PfPeakUnfilled()`**：本节的 8 个槽是**融合 prefill 入口内**
//     相位 A/H/B 的 UB/L1/L0C 峰值；S2/S3 是入口**之外**的独立 kernel，其资源是**每 kernel 自己的**，
//     不与入口内相位共享窗 ⇒ 语义上不属于这张表。`PfPeakUnfilled() == 3` 的不变式**不变**。
//   · 下面是 prolog 两个 kernel 自己的 UB/L1 **峰值登记**（数值 = 各自设备头里那条
//     `static_assert(... <= 预算)` 的**被断言量本身**），并补"每 kernel 消耗"的 GM 平面台账。
// S2 in_proj（M15OP::OProjGemm<2560,16480>）：L1 用 A(2×64×64×2B)+B(2×160×64×2B 起于 L1_B_REGION)
//   两段 ping-pong，峰值 = L1_OFF_B1 + L1_B_ELEMS*2（m15_attn_oproj.h:239-245）。
// S3 prolog（M15PM::MtGdnProlog）：UB 静态自管理，峰值 = MT_END（m15_prefill_prolog.h 内 MT_* 表）。
// 这两个数与设备头的**逐一对上**由 `m15_layer_loop.asc` 的跨文件 static_assert 钉住（数字漂了编不过）。
constexpr uint32_t PF_INPROJ_L1_BYTES = 303104u;    // S2：L1_OFF_B1(282624) + L1_B_ELEMS*2(20480)
constexpr uint32_t PF_PROLOG_UB_BYTES = 71680u;     // S3：MT_END = 38912 + 64*128*4
constexpr uint32_t L1_BYTES_TOTAL_M151 = 512u * 1024u;
constexpr uint32_t UB_TOTAL_BYTES_M151 = 248u * 1024u;
static_assert(PF_INPROJ_L1_BYTES <= L1_BYTES_TOTAL_M151, "M151 S2 in_proj 的 L1 峰值超 512KB");
static_assert(PF_PROLOG_UB_BYTES <= UB_TOTAL_BYTES_M151, "M151 S3 prolog 的 UB 峰值超 248KB");
// GM 平面台账（**新分配，按 M_PREFILL=4097 定尺**；见 H_PfSegAlloc）。
constexpr uint32_t PF_QKVZBA_BYTES_PER_ROW = 16480u * 2u;   // S2 输出 = S3 的 x：[m,16480] bf16
constexpr uint32_t PF_CONV_STATE_BYTES = 3u * 10240u * 2u;  // conv_state：[3,10240] bf16 = 61,440 B
// 这两式与模型常量逐一钉死（r1 复审 P2-3：原注释声称有"另一处 static_assert"，实际没有 ⇒ 在此补上）。
static_assert(PF_QKVZBA_BYTES_PER_ROW == M15G::IN_N * 2u, "M151：qkvzba 行距 != M15G::IN_N*2");
static_assert(PF_CONV_STATE_BYTES == M15G::ST * M15G::CH * 2u, "M151：conv_state 字节 != [3,10240] bf16");
// 不变式钉死：prolog 的接入不得改变 prefill 峰值表的语义（attention 的 3 槽仍是唯一未填项）。
static_assert(PfPeakUnfilled() == 3u, "M151 不得改变 prefill 峰值表的未填数（应恒为 3 = attention 的 UB/L1/L0C）");


// ------------------------------------------------------------
// 3c-ter. M164：prefill **epilog 链路**（转位/落位 + S5 RMSNormGated + S6 out_proj）的登记
// ------------------------------------------------------------
// 形态与 M151 §3c-bis 同款：epilog 的三个段体是**独立 `__global__`**（逐字复用 m28 donor，见
// `m15_prefill_epilog.h`），由 host 在相位 A 之后、H2 之前按同一条 stream 串行起动
// （`H_PfGdnWired → H_PfEpilogRun`）。因此：
//   · **不新增 flagId**：段间次序由 stream 给；各段内部各自是 BufferID / `AscendC::Mutex` 同步
//     ⇒ 不需要跨核 id，`FLAG_SEQ_*` 一字不动。
//   · **不动 `PF_PEAK_TABLE` / `PfPeakUnfilled()`**：本节的 8 个槽是**融合 prefill 入口内**相位
//     A/H/B 的峰值；epilog 三段在入口**之外**，其资源是每 kernel 自己的，不与入口内相位共享窗。
//   · 下面是三段各自的 UB/L1/L0C **峰值登记**（数值 = 各自设备头里那条 `static_assert(... <= 预算)`
//     的**被断言量本身**），并补"每 kernel 消耗"的 GM 平面台账。
// 三个数与设备头的**逐一对上**由 `m15_layer_loop.asc` 的跨文件 static_assert 钉住（数字漂了编不过）。
// 转位/落位（M15PE::Trans::GdnEpilogKernel）：UB 静态自管理，峰值 = UB_Z + UB_Z_BYTES
//   （两窗 98304 B 不重叠，m28_gdn_epilog.asc:86-90）。
// S5（M15PE::Chain::ChainS5Kernel）：UB 峰值 = UB_S5_RSTD + 256（m28_epilog_chain.asc:241-249）。
// S6（M15PE::Chain::S6::Bf16Gemm）：L1 峰值 = L1_B_REGION + 2*L1_B_ELEMS*2（:704-711）；
//   L0C = BASE_M*BASE_N floats（:759）。
constexpr uint32_t PF_EPILOG_TRANS_UB_BYTES = 196608u;   // 转位段：UB_Z(98304) + UB_Z_BYTES(98304)
constexpr uint32_t PF_EPILOG_S5_UB_BYTES = 50688u;       // S5：UB_S5_RSTD(50432) + 256
constexpr uint32_t PF_EPILOG_S6_L1_BYTES = 303104u;      // S6：L1_B_REGION(262144) + 2*10240*2(40960)
constexpr uint32_t PF_EPILOG_S6_L0C_BYTES = 40960u;      // S6：BASE_M(64)*BASE_N(160)*4
static_assert(PF_EPILOG_TRANS_UB_BYTES <= UB_TOTAL_BYTES_M151, "M164 epilog 转位段的 UB 峰值超 248KB");
static_assert(PF_EPILOG_S5_UB_BYTES <= UB_TOTAL_BYTES_M151, "M164 S5 的 UB 峰值超 248KB");
static_assert(PF_EPILOG_S6_L1_BYTES <= L1_BYTES_TOTAL_M151, "M164 S6 的 L1 峰值超 512KB");
static_assert(PF_EPILOG_S6_L0C_BYTES <= L0C_TOTAL_BYTES, "M164 S6 的 L0C 峰值超 256KB");
// GM 平面台账（**新分配，按 M_PREFILL=4097 定尺**；见 H_PfSegAlloc）。
// 注意：M159 §2.2 只列了 oTok/ztok 两块；`y`（S5 出 = S6 的 A 面，`m28_epilog_chain.asc` README:195）
// 是转位/S5 与 S5/S6 之间的**第三个** GM 中间面 —— S5 与 S6 是两次独立 launch，必须有自己的生命周期。
constexpr uint32_t PF_EPILOG_OTOK_BYTES_PER_ROW = 6144u * 4u;   // [m,6144] fp32：转位后 token-major o
constexpr uint32_t PF_EPILOG_ZTOK_BYTES_PER_ROW = 6144u * 2u;   // [m,6144] bf16：z 压实
constexpr uint32_t PF_EPILOG_YTOK_BYTES_PER_ROW = 6144u * 2u;   // [m,6144] bf16：S5 出 = S6 的 A
// 与模型常量逐一钉死（M15G::HIDDEN 是 host 侧同一常量；此处用字面量并在 .asc 侧核）。
static_assert(PF_EPILOG_OTOK_BYTES_PER_ROW == 48u * 128u * 4u, "M164：oTok 行距 != HEADS*HEAD*4");
static_assert(PF_EPILOG_ZTOK_BYTES_PER_ROW == 48u * 128u * 2u && PF_EPILOG_YTOK_BYTES_PER_ROW == 48u * 128u * 2u,
              "M164：zTok/yTok 行距 != HEADS*HEAD*2");
// 不变式钉死：epilog 的接入不得改变 prefill 峰值表的语义（attention 的 3 槽仍是唯一未填项），
// 且接线门 PREFILL_WIRED 仍为 0（B2/B3 未接 ⇒ 不得打开）。
static_assert(PfPeakUnfilled() == 3u, "M164 不得改变 prefill 峰值表的未填数（应恒为 3 = attention 的 UB/L1/L0C）");
static_assert(PREFILL_WIRED == 0u, "M164：prefill 接线门应仍为 0（attention B2/B3 未接）");


// ============================================================
// 4. CrossCore flagId 全局登记表（**唯一权威**：三段的 §3 表都必须与这张表一致）
// ============================================================
// 每核 16 个 id（docs/05 §6.1：本项目不用 SyncAll/高阶 API → 0-15 全可用）。
// **M110 修正口径**：这句话只对 **mode 0/1/2** 成立；**mode 4 的 AIC 侧扩到 0..31**（16..31 → 配对
// AIV1）。上限现在由 `FlagIdLimit(核型, mode)` 给，裁决与依据见下面的 §4a-1。
//
// 硬规则（docs/05 §2 / docs/12 §4）：**同一核上相邻的两个同步点必须用不同 flagId**
// （同一核连续多个 CrossCoreSetFlag 硬件不保证执行顺序）。于是本表**按 (核型, mode) 把
// 整个融合 kernel 的同步点排成执行序**，再用 static_assert 逐对核对相邻 id 不同。
//
// 分配（M58 四相位版；hc 段两次出现都用同一组 id —— 与 MoE 的 4 槽 ring 同款复用）：
//   hc 段   AIV mode0 = 12/13/14（原 m20 是 0/1/2，为与 MoE 的 mode2 0-3 完全错开而重编）
//           AIC mode0 = 0/1/2/3  （原 m20 是 4/5/6/7，与 GDN 的 mode2 4-7 撞号）
//           mode2     = 8/9/10/11（与 m20 相同）
//   GDN 段  AIV mode0 = 10/8/9/11、AIC mode0 = 12/13/14/15、mode2 = 4/5/6/7
//   MoE 段  AIV mode0 ring = 12/13/14/15(→12)、AIC mode0 ring = 8/9/10/11、mode2 ring = 0/1/2/3
//   相位边界（全体 AIV mode0）：H1→A = 8、A→H2 = 9、H2→B = 8（= M40 的 FLAG_L0_BOUND_AIV）
//
// **注意 hc 与 GDN/MoE 在同 id 上的复用**：表里 `reuse=true` 的行是「同 (核型, mode) 的
// 第二次使用」，相邻性由下面的 static_assert 逐对保证（不是靠区间不相交的静态说法）。

// ---- 三条相位边界的全体 AIV mode-0 barrier id（M58）----
// 全部在 AIV mode0 子空间里；相邻性由 FLAG_SEQ[] 的执行序见证：
//   H1 的 S4→S4(14) → H1→A(8) → GDN 的 10/8/9/11 → A→H2(9) → H2 的 12/13/14 → H2→B(8) → MoE 的 12
// （KIND_ATTN 层没有 GDN 的四条，于是相邻序变成 …14→8→9→12…，仍两两不同）
constexpr uint16_t FLAG_HC0_BOUND_AIV = 8;   // hc(H1) → 子层段
constexpr uint16_t FLAG_HC1_BOUND_AIV = 9;   // 子层段 → hc(H2)
constexpr uint16_t FLAG_HC2_BOUND_AIV = 8;   // hc(H2) → MoE 段（= M40 的 FLAG_L0_BOUND_AIV）
static_assert(FLAG_HC2_BOUND_AIV == M15M::FLAG_L0_BOUND_AIV,
              "hc(H2)→MoE 的相位边界必须沿用 M40 登记的 id（8）——两相位形态的入口仍用它");

// ---- M65：PLE 打断点（层 1）的两条相位边界 ----
// **不新占 flagId**：AIV 核上 0..15 已被三段的 mode2（MoE 0-3 / GDN 4-7 / hc 8-11）与 mode0
// （hc 12-14 / GDN 8-11 / MoE 12-15 / 三条 boundary 8,9）用满，故这两条边界直接复用 M40/M58
// 登记的**同一类**同步点（全体 AIV mode-0 barrier）的 id 8/9 —— 与 FLAG_HC0/1_BOUND_AIV 同类，
// 相邻性由下面的 FLAG_SEQ 执行序见证（层 1 的序：hc(H1a) 12 → 8 → 9 → hc(H1b) 12/13/14 → 8 → …）。
constexpr uint16_t FLAG_PLE_IN_BOUND_AIV = FLAG_HC0_BOUND_AIV;    // combine-only → PLE 挂载点
constexpr uint16_t FLAG_PLE_OUT_BOUND_AIV = FLAG_HC1_BOUND_AIV;   // PLE 挂载点 → mix-only

struct FlagStep {
    const char* seg;    // "hc(H1)" / "hc(H2)" / "GDN" / "MoE" / "bound"
    const char* core;   // "AIV" / "AIC" / "both"（mode2 两侧同 id）
    uint32_t    mode;   // 0 / 2
    uint32_t    id;
    bool        reuse;  // true = 与前面某个同步点同 id（有意复用，相邻性由断言保证）
    const char* use;
};
constexpr FlagStep FLAG_SEQ[] = {
    // ---- 相位 H1：hc 边界 #1（attn_hc）----
    {"hc(H1)", "both", 2, M15H::FLAG_A2C_XN,  false, "H1 AIV→AIC：XN 就绪"},
    {"hc(H1)", "both", 2, M15H::FLAG_C2A_OH,  false, "H1 AIC→AIV：OH（lora+inj）就绪"},
    {"hc(H1)", "both", 2, M15H::FLAG_A2C_LS,  false, "H1 AIV→AIC：LS 就绪"},
    {"hc(H1)", "both", 2, M15H::FLAG_C2A_GATE, false, "H1 AIC→AIV：GATE 就绪"},
    {"hc(H1)", "AIV",  0, M15H::FLAG_AV0,     false, "H1 S1→S2：H' 全量落盘"},
    {"hc(H1)", "AIV",  0, M15H::FLAG_AV1,     false, "H1 S2→AIC：XN 全量落盘"},
    {"hc(H1)", "AIV",  0, M15H::FLAG_AV2,     false, "H1 S4→AIC：LS 全量落盘"},
    {"hc(H1)", "AIC",  0, M15H::FLAG_AC0,     false, "H1 down GEMM 前对齐"},
    {"hc(H1)", "AIC",  0, M15H::FLAG_AC1,     false, "H1 down GEMM 后（FIXP 写 GM 排空）"},
    {"hc(H1)", "AIC",  0, M15H::FLAG_AC2,     false, "H1 up GEMM 前对齐"},
    {"hc(H1)", "AIC",  0, M15H::FLAG_AC3,     false, "H1 up GEMM 后（FIXP 写 GM 排空）"},
    // ---- M65：PLE 打断点（**只在该层 = 0-based 层 1** 执行；其它层是上面那组 H1 一段）----
    // 执行序（层 1）：hc(H1a) 的 AV0(12) → 边界 8 → 边界 9 → hc(H1b) 的 8/9/10/11 + 12/13/14
    // + 0..3 → 边界 8（H1→A）→ …。相邻性：14→12→8→9→12→13→14→8 两两不同（见 FlagSeqAdjacentOk）。
    // H1a（combine-only）**没有** mode2 交接（AIC 立即返回），故它只贡献一条 AIV mode0。
    {"hc(H1a)", "AIV", 0, M15H::FLAG_AV0,     true, "层 1 H1a（combine-only）跑完 W0+S1：H' 物化（复用 AV0）"},
    {"bound",   "AIV", 0, FLAG_PLE_IN_BOUND_AIV,  true, "层 1：combine 物化 → PLE 挂载点（全体 AIV mode-0）"},
    // ---- M124：PLE ③ 改走 cube（AIC 上的 mmad）之后的四条新同步点 ----
    // 执行序（层 1，②③④④⑤ 五步）：② AIV 的 gather 落盘 → 【AIV mode2 set B2 / AIC mode2 wait B2
    //   → AIC mode0 CUBE_IN（全体 AIC 对齐）→ AIC 的 mmad + Fixpipe → AIC mode0 CUBE_OUT（FIXP 排空）
    //   → AIC mode2 set B3 / AIV mode2 wait B3】→ ④ → ⑤。
    // 依据（三处都逐字）：`docs/05` §2「mode 0 仅同类型……跨类型 all-to-all 的标准组合：
    //   AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）→ AIC→AIVs（mode 2）」；
    //   `docs/05` §6.1 的「每核 16 个 flagId 是 mode0/mode2 共享的池」；
    //   形态先例 = `m15_attn_prolog_probe.h` 的 `AP_AIC_M0_OUT → AP_A2V_GEMM` 与
    //   `m15_gdn_layer.h` 的 `FLAG_AIC_SEG0B → FLAG_BOUND_INPROJ`。
    // ⚠ **这两条是 mode2 ⇒ 两侧都用同一个号**（`core="both"`，与该表其它 mode2 行的标注一致）：
    //   AIV 侧 `PleGemv` 的 AIV 臂 set `FLAG_B2` / wait `FLAG_B3`；**AIC 侧**（`m15_ple.asc` 的
    //   `if ASCEND_IS_AIC` 臂，即 `PleGemv` 的 AIC 臂）wait `FLAG_B2` / set `FLAG_B3`
    //   —— 即 `m15_ple.asc:742/:761`。⇒ 本表的这两个 AIC 侧使用**必须**在登记里体现
    //   （初版误登记为 `"AIV"`，是 M124 r1 复审 P2-1 指出的读数不实，已订正为 `"both"`）。
    // ⚠ 跨 mode 复用（`docs/05` §2 第 3 条：同 id 跨模式复用的前提是前一 mode 的 set/wait 全部 drain）：
    //   · AIV：这些号（12/13）与 hc 的 AIV **mode0** 的 AV0/AV1 同号 —— M65/M100 以来就有；
    //     hc(H1a) 的 AV0 是自封 barrier（set+wait 成对、跑完即归零），PLE 这一段跑在它之后、
    //     hc(H1b) 之前 ⇒ 时间上不重叠。
    //   · AIC：这些号与 GDN 的 AIC **mode0** 的 12/13 同号（`FLAG_SEQ` 里的 S2/S6 对齐点）——
    //     GDN 的相位 A 在 PLE 段之后（段序：hc(H1a) → PLE → hc(H1b) → **GDN** → hc(H2) → MoE，
    //     见 `m15_layer_kernel.h` 的 AIC 分支），且每处 set/wait 成对 ⇒ 计数在段间回到 0。
    {"PLE",     "both", 2, M85P::FLAG_B2,      false, "层 1 ③：AIV→AIC 的 emb 落盘就绪（配对 mode2：AIV set / AIC wait）"},
    {"PLE",     "AIC", 0, M85P::FLAG_CUBE_IN,  false, "层 1 ③ 入口：全体 AIC 到齐（此时 ② 的 emb 对任意 AIC 可见）"},
    {"PLE",     "AIC", 0, M85P::FLAG_CUBE_OUT, false, "层 1 ③ 出口：Fixpipe 写 GM 排空 + 全体 AIC 到齐"},
    {"PLE",     "both", 2, M85P::FLAG_B3,      false, "层 1 ③→④：AIC→AIV 的 kv 落盘就绪（配对 mode2：AIC set / AIV wait）"},
    // ⚠ **`FLAG_B4` 也要登记**（r2 复审 P2-1 残留的落点）：④→⑤ 是**全体 AIV 的 mode-0 barrier**
    //   （`M15H::BarrierAiv<PIPE_MTE2, FLAG_B4>`，跑在 PLE body 内部 = ③ 之后、⑤ 之前）
    //   ⇒ 它的执行位置在 `8(PLE_IN)` 与 `9(PLE_OUT)` **之间**（与 `m15_layer_kernel.h` 里
    //   「PLE_IN 边界与 ④→⑤ 相邻」「④→⑤ 与 PLE_OUT 边界相邻」两条 static_assert 一致）。
    //   初版漏登记它，导致脚本按 `FLAG_SEQ` 复现不出 AIV mode0 的真实序 —— 补上后脚本即可原样产出。
    {"PLE",     "AIV",  0, M85P::FLAG_B4,      true,  "层 1 ④→⑤：全体 AIV mode-0 barrier（id 复用 hc 的 AV2）"},
    {"bound",   "AIV", 0, FLAG_PLE_OUT_BOUND_AIV, true, "层 1：PLE 挂载点 → mix-only（全体 AIV mode-0）"},
    {"hc(H1b)", "both", 2, M15H::FLAG_A2C_XN,  true, "层 1 H1b（mix-only）AIV→AIC：XN 就绪（复用）"},
    {"hc(H1b)", "both", 2, M15H::FLAG_C2A_OH,  true, "层 1 H1b AIC→AIV：OH 就绪（复用）"},
    {"hc(H1b)", "both", 2, M15H::FLAG_A2C_LS,  true, "层 1 H1b AIV→AIC：LS 就绪（复用）"},
    {"hc(H1b)", "both", 2, M15H::FLAG_C2A_GATE, true, "层 1 H1b AIC→AIV：GATE 就绪（复用）"},
    {"hc(H1b)", "AIV",  0, M15H::FLAG_AV0,     true, "层 1 H1b 段内 barrier（S1 不跑，但该 barrier 无条件执行）"},
    {"hc(H1b)", "AIV",  0, M15H::FLAG_AV1,     true, "层 1 H1b S2→AIC：XN 全量落盘（复用）"},
    {"hc(H1b)", "AIV",  0, M15H::FLAG_AV2,     true, "层 1 H1b S4→AIC：LS 全量落盘（复用）"},
    {"hc(H1b)", "AIC",  0, M15H::FLAG_AC0,     true, "层 1 H1b down GEMM 前对齐（复用）"},
    {"hc(H1b)", "AIC",  0, M15H::FLAG_AC1,     true, "层 1 H1b down GEMM 后 drain（复用）"},
    {"hc(H1b)", "AIC",  0, M15H::FLAG_AC2,     true, "层 1 H1b up GEMM 前对齐（复用）"},
    {"hc(H1b)", "AIC",  0, M15H::FLAG_AC3,     true, "层 1 H1b up GEMM 后 drain（复用）"},
    // ---- 相位边界 H1→A ----
    {"bound", "AIV", 0, FLAG_HC0_BOUND_AIV, true, "hc(H1)→子层段：全体 AIV mode-0 barrier"},
    // ---- 相位 A：GDN 段（m15_gdn_resources.h §3 的权威表；此处按执行序重排）----
    {"GDN", "both", 2, 4, false, "AIV→AIC：in_proj 输入 x_norm 就绪（set）"},
    {"GDN", "both", 2, 5, false, "AIC→AIV：in_proj 输出 qkvzba 就绪"},
    {"GDN", "both", 2, 6, false, "AIV→AIC：RMSNormGated 输出（out_proj 输入）就绪"},
    {"GDN", "both", 2, 7, false, "AIC→AIV：out_proj 输出就绪"},
    {"GDN", "AIV",  0, 10, false, "S1 后：x_norm/res1 全体 AIV 就位"},
    {"GDN", "AIV",  0, 8,  false, "S3→S4：q/k/v/g/β 就位"},
    {"GDN", "AIV",  0, 9,  false, "S4→S5：o 就位"},
    {"GDN", "AIV",  0, 11, false, "S5→S7：opin 全体就位"},
    {"GDN", "AIC",  0, 12, false, "S2 in_proj 前对齐"},
    {"GDN", "AIC",  0, 13, false, "S2 in_proj 后 drain 对齐"},
    {"GDN", "AIC",  0, 14, false, "S6 out_proj 前对齐"},
    {"GDN", "AIC",  0, 15, false, "S6 out_proj 后 drain 对齐"},
    // ---- 相位边界 A→H2 ----
    {"bound", "AIV", 0, FLAG_HC1_BOUND_AIV, true, "子层段→hc(H2)：全体 AIV mode-0 barrier"},
    // ---- 相位 H2：hc 边界 #2（mlp_hc）----
    {"hc(H2)", "both", 2, M15H::FLAG_A2C_XN,  true, "H2 AIV→AIC：XN 就绪（复用 H1 的 id）"},
    {"hc(H2)", "both", 2, M15H::FLAG_C2A_OH,  true, "H2 AIC→AIV：OH 就绪（复用）"},
    {"hc(H2)", "both", 2, M15H::FLAG_A2C_LS,  true, "H2 AIV→AIC：LS 就绪（复用）"},
    {"hc(H2)", "both", 2, M15H::FLAG_C2A_GATE, true, "H2 AIC→AIV：GATE 就绪（复用）"},
    {"hc(H2)", "AIV",  0, M15H::FLAG_AV0,     true, "H2 S1→S2：H' 全量落盘（复用）"},
    {"hc(H2)", "AIV",  0, M15H::FLAG_AV1,     true, "H2 S2→AIC：XN 全量落盘（复用）"},
    {"hc(H2)", "AIV",  0, M15H::FLAG_AV2,     true, "H2 S4→AIC：LS 全量落盘（复用）"},
    {"hc(H2)", "AIC",  0, M15H::FLAG_AC0,     true, "H2 down GEMM 前对齐（复用）"},
    {"hc(H2)", "AIC",  0, M15H::FLAG_AC1,     true, "H2 down GEMM 后 drain（复用）"},
    {"hc(H2)", "AIC",  0, M15H::FLAG_AC2,     true, "H2 up GEMM 前对齐（复用）"},
    {"hc(H2)", "AIC",  0, M15H::FLAG_AC3,     true, "H2 up GEMM 后 drain（复用）"},
    // ---- 相位边界 H2→B ----
    {"bound", "AIV", 0, FLAG_HC2_BOUND_AIV, true, "hc(H2)→MoE 段：全体 AIV mode-0 barrier"},
    // ---- 相位 B：MoE 段（m15_moe_resources.h §3 的权威表；4 槽 ring，执行序）----
    {"MoE", "both", 2, 0,  false, "AIV→AIC：A（激活量化）就绪"},
    {"MoE", "both", 2, 1,  false, "AIC→AIV：GU 就绪"},
    {"MoE", "both", 2, 2,  false, "AIV→AIC：H 就绪"},
    {"MoE", "both", 2, 3,  false, "AIC→AIV：Y 就绪"},
    {"MoE", "AIV",  0, 12, false, "S1 后 ring slot0"},
    {"MoE", "AIV",  0, 13, false, "S3 后 ring slot1（PIPE_S：perm_src 标量读）"},
    {"MoE", "AIV",  0, 14, false, "S4 后 ring slot2"},
    {"MoE", "AIV",  0, 15, false, "S9a 后 ring slot3"},
    {"MoE", "AIV",  0, 12, true,  "S9b 后 ring slot0（复用）"},
    {"MoE", "AIC",  0, 8,  false, "GateUp 前对齐"},
    {"MoE", "AIC",  0, 9,  false, "GateUp 后 drain 对齐"},
    {"MoE", "AIC",  0, 10, false, "Down 前对齐"},
    {"MoE", "AIC",  0, 11, false, "Down 后 drain 对齐"},
};
constexpr uint32_t FLAG_PER_CORE = 16;
constexpr uint32_t FLAG_SEQ_N = sizeof(FLAG_SEQ) / sizeof(FLAG_SEQ[0]);

// ------------------------------------------------------------
// 4a-1. M110（Wave A）：**mode-4 的 id 上限裁决**（M101 复审 F2 点名的硬冲突）
// ------------------------------------------------------------
// 硬件事实（CANN 官方 spec；M101 复审 F2 引 `CrossCoreSetFlag_ISASI.md` 的「flagId 取值范围说明」）：
//   · **mode 0/1/2**：AIV 与 AIC **各自的** id 池都是 0..15（每核 16 个，`docs/05 §6.1`）；
//   · **mode 4**：**AIV 侧仍是 0..15（同一个物理池）**，**只有 AIC 侧扩到 0..31**，其中 AIC 的
//     16..31 指向配对里的 **AIV1** 那一侧（`m15_attn_core.h` 的 `AIV_CH = 16` 就是这么用的）。
// ⇒ 「每核 16 个」这句话**只对 mode 0/1/2 成立**。原来的 `FlagSeqAdjacentOk()` 无条件用
//    `FLAG_PER_CORE = 16` 卡所有记录，于是 **AIC mode-4 的 id 16/17（`ccMM + AIV_CH`）被结构性拒收**
//    —— 这是 B3（`m15_attn_fa_core*`）接不进来的根因（M101 复审 F2）。
//    **M168 收口**：该根因已由 `FlagIdLimit()` 解掉；**M185 起 B3 的通道族改用 mode 2**（§4f.2），
//    B3 的号不再需要 AIC 侧 0..31 —— 该裁决只对仍用 mode 4 的 `m15_attn_core.h` 继续成立。
//    §4f.2 是 B3 那族的落点（逐条断言上限/相邻性/配对）。
//
// **裁决（两个选项的后果，选 A）**：
//   · **A（采纳）**：`FLAG_PER_CORE` 保留 16 作 mode 0/1/2 的每核池；新增 `FlagIdLimit(core, mode)`：
//     **仅 `AIC ∧ mode == 4`** 放行到 `FLAG_PER_CORE_AIC_MODE4 = 32`。后果：全部既有记录
//     （mode 0/2）的上限仍是 16 ⇒ **既有保证一字不变**；AIC mode-4 的 16..31 合法。
//     代价：检查器变成 mode 相关，「16」的含义要读成「mode0/1/2 每核 16；mode4 的 AIC 侧 32」。
//   · **B（否）**：把 `FLAG_PER_CORE` 全局抬到 32。后果：**AIV 或 mode0/2 上的 id 16..31 也会被接受**，
//     而硬件池只有 0..15 ⇒ 断言失去抓「真会挂死」那一类 bug 的能力（M2 实证过同核连续 set 不保序）。
constexpr uint32_t MODE4 = 4;
constexpr uint32_t FLAG_PER_CORE_AIC_MODE4 = 32;

// 核型归类："AIV"→0、"AIC"→1、"both"→2（mode2 的 id 空间是 AIC 与配对 AIV 共用的）
constexpr uint32_t CoreId(const char* c) { return (c[2] == 'V') ? 0u : ((c[2] == 'C') ? 1u : 2u); }
constexpr bool CoreSame(const char* a, const char* b)
{
    const uint32_t x = CoreId(a);
    const uint32_t y = CoreId(b);
    return (x == y) || (x == 2u) || (y == 2u);
}

// 该 (核型, mode) 子空间的合法 id 上界（把「每核 16 个」放回它成立的那三个 mode 上）
constexpr uint32_t FlagIdLimit(const char* core, uint32_t mode)
{
    return ((mode == MODE4) && (CoreId(core) == 1u)) ? FLAG_PER_CORE_AIC_MODE4 : FLAG_PER_CORE;
}

// 裁决的非空洞性见证（不是"说了一句"）：
//   · mode0/2 上仍拒 16 —— 用 AIC/mode0 举一个反例，检查器必须给出 false；
//   · mode4 的 AIC 上放行 16/17 —— B3 要用的两个号必须在限内；
//   · mode4 的 **AIV** 上仍拒 16 —— AIV 侧的池没变宽（F2 的核心事实）。
static_assert(FlagIdLimit("AIC", 0u) == 16u && FlagIdLimit("AIC", 2u) == 16u &&
                  FlagIdLimit("AIV", 0u) == 16u && FlagIdLimit("AIV", 2u) == 16u,
              "mode0/2 的每核 id 池必须仍是 16（改动这条就动了既有全部保证）");
static_assert(FlagIdLimit("AIC", MODE4) == 32u && FlagIdLimit("AIV", MODE4) == 16u,
              "mode4 只有 AIC 侧扩到 32；AIV 侧仍是 16（同一物理池）");

// 相邻性核对：同一 (核型, mode) 的执行序里，相邻两步的 id 必须不同。
constexpr bool FlagSeqAdjacentOk()
{
    for (uint32_t i = 0; i < FLAG_SEQ_N; ++i) {
        if (FLAG_SEQ[i].id >= FlagIdLimit(FLAG_SEQ[i].core, FLAG_SEQ[i].mode)) {
            return false;
        }
        for (uint32_t j = i + 1; j < FLAG_SEQ_N; ++j) {
            if (FLAG_SEQ[i].mode == FLAG_SEQ[j].mode && CoreSame(FLAG_SEQ[i].core, FLAG_SEQ[j].core)) {
                if (FLAG_SEQ[i].id == FLAG_SEQ[j].id) {
                    return false;
                }
                break;   // 只与**紧邻**的同 (核型, mode) 同步点比较
            }
        }
    }
    return true;
}
static_assert(FlagSeqAdjacentOk(), "flagId 相邻性违规：同一 (核型, mode) 的相邻同步点用了同一个 id");

// 用量核对：每个 (核型, mode, id) 在**一次 kernel 内**被使用几次（硬件 4bit 计数器 ≤ 15）
constexpr uint32_t FlagMaxUse()
{
    uint32_t mx = 0;
    for (uint32_t i = 0; i < FLAG_SEQ_N; ++i) {
        uint32_t n = 0;
        for (uint32_t j = 0; j < FLAG_SEQ_N; ++j) {
            if (FLAG_SEQ[i].mode == FLAG_SEQ[j].mode && FLAG_SEQ[i].id == FLAG_SEQ[j].id && CoreSame(FLAG_SEQ[i].core, FLAG_SEQ[j].core)) {
                ++n;
            }
        }
        mx = (n > mx) ? n : mx;
    }
    return mx;
}
// M58 口径：id 12 在 AIV mode0 上被用 **4** 次（hc H1、hc H2、MoE S1 后、MoE S9b 后）——这就是
// 「四相位 + 两段各自复用」下的上界。
// M65 口径：**6** 次 —— 层 1 的 PLE 打断点把 H1 拆成 H1a/H1b，两者都用 AIV mode0 的 id 12
// （H1a 的 AV0、H1b 的 AV0），于是 id 12 的用量 = hc(H1a) + hc(H1b) + hc(H2) + MoE 两处 = **5**；
// 非 PLE 层则是 hc(H1) + hc(H2) + MoE 两处 = 4。**表里两类层都列了**（同一张表要覆盖两种层形态），
// 故按最保守的计数是 **6**。
// 仍远低于硬件 4bit 计数器上限 15（docs/05 §6.1）。
// **M124 追加（r1 复审 P2-1）**：层 1 的 PLE 段在 **AIC mode2** 上也用了 12/13（`FLAG_B2/FLAG_B3`
// 的另一半，见上面 `FLAG_SEQ` 里那两行的 `"both"` 标注）。⇒ 逐核用量要按**核型**分别看：
//   · AIV mode2 id 12/13：hc(H1a) 8/9/10/11 → **PLE 12/13** → hc(H1b) 8/9/10/11 → GDN 4–7 → MoE 0–3；
//     每个 id 各 1 次；
//   · AIC mode2 id 12/13：**PLE 12/13**（wait B2 / set B3）→ hc(H1b) 8/9/10/11 → GDN 4–7 → MoE 0–3；
//     每个 id 各 1 次；相邻对 12→13→8 两两不同；
//   · **跨 mode**：AIC 上 mode0 的 12/13 归 GDN（相位 A，在 PLE 段之后）、mode2 的 12/13 归 PLE
//     （层 1 的打断点段）⇒ 同一号在同一次 kernel 里被两个 mode 各用一次，但**时间上不重叠**
//     （PLE 段夹在 hc(H1a) 与 hc(H1b) 之间；GDN 的相位 A 在 hc(H1b) 之后、hc(H2) 之前），
//     且每处都是成对的 set/wait
//     ⇒ 计数在段间回到 0（`docs/05` §2 第 3 条的前提）。AIV 侧同理（mode0 的 12/13 归 hc）。
static_assert(FlagMaxUse() <= 6, "同一 (核型,mode,flagId) 每 kernel 用量过大（硬件上限 15）");

// ------------------------------------------------------------
// 4b. M97（W1）：attention 相位 A 的 flagId 分节
// ------------------------------------------------------------
// **预算表与逐格归属见 `evidence/attn_wire/README.md` §1**；这里只放可编译期核对的那一半。
//
// 事实（按 (核型, mode) 子空间；号由三段的 FLAG_* 常量给出，执行序由上面的 FLAG_SEQ[] 给出）：
//   · AIV mode0 用 {8,9,10,11}（GDN 10/8/9/11 + 三条边界 8,9）∪ {12,13,14}（hc）∪ {12,13,14,15}（MoE）；
//   · AIV mode2 用 {0..3}（MoE）∪ {4..7}（GDN）∪ {8..11}（hc）∪ **{12,13}（M124 的 PLE 打断点段，
//     层 1；AIV 侧 set/wait）**；
//   · AIC mode0 用 {0..3}（hc）∪ {8..11}（MoE）∪ {12..15}（GDN）∪ **{14,15}（M124 的 PLE 段）**；
//   · AIC mode2 用 {0..3}（MoE）∪ {4..7}（GDN）∪ {8..11}（hc）∪ **{12,13}（M124 的 PLE 打断点段，
//     层 1；AIC 侧 wait/set —— 与 AIV 侧同一对号，mode2 是配对的）**。
// ⇒ **把 mode0 与 mode2 合起来看，AIV/AIC 的 0..15 都已占**（docs/05 §6.1：每核 16 个 flagId
//   是 mode0/mode2 **共享**的池；docs/05 §2 第 3 条：同 id 跨模式复用须前一 mode 的 set/wait
//   全部 drain 完）。所以 attention **不新造号**，走下面的复用分节。
//
// 分节方案（依据 = 「KIND_ATTN 层上 GDN 段整段不执行」这一执行序事实）：
//   · **AIC mode0 = `M15AP::AP_AIC_M0_OUT`（=1）**：与 hc 的 `FLAG_AC1` 同号。hc(H1) 与 hc(H2)
//     各自 set/wait 该号一次，attention 的相位 A 夹在两者**之间**（H1 的 wait 已完成 ⇒ 计数归零），
//     故这里是有意复用。AIC mode0 在 KIND_ATTN 层的执行序变成
//     `0,1,2,3 → 1（attention）→ 0,1,2,3 → 8,9,10,11`，相邻两两不同。
//   · **mode2 = `M15AP::AP_A2V_GEMM`（=5）**：与 GDN 的 `FLAG_BOUND_INPROJ` 同号 —— GDN 的
//     4-7 在 KIND_ATTN 层**不执行**，故同一层内不可能同时出现。mode2 执行序变成
//     `8,9,10,11 → 5 → 8,9,10,11 → 0..3`，相邻两两不同。
//   · **AIV 侧不新增任何同步点**：那条臂只 `wait` mode2，不 `set`（`m15_attn_prolog_probe.h`
//     的 AIV 分支），所以 AIV mode0 的序（…14 → 8 → 9 → 12…）一字不变。
//
// ⚠ 本表**只覆盖 KIND_ATTN 层的相位 A**（与 FLAG_SEQ[] 只覆盖 KIND_GDN 层是同一分工）；
//   两种层形态由入口符号 `m15_layer_kernel_{gdn,attn}_hc` 在 host 侧互斥选择，同一次启动里
//   只可能有一种。下面把「与相位 A 两侧 hc 同步点相邻」这四条边界条件**显式写出来**，
//   而不是靠"号段区间不相交"的口头说法。
struct FlagStepAttn {
    const char* core;   // "AIC" / "both"（mode2 两侧同号）
    uint32_t    mode;   // 0 / 2
    uint32_t    id;
    const char* use;
};
constexpr FlagStepAttn FLAG_SEQ_ATTN_PA[] = {
    {"AIC", 0, M15AP::AP_AIC_M0_OUT,
     "attention：4 个 GEMM 的 FIXP 写 GM 排空 + 全体 AIC 对齐（set 挂 PIPE_FIX / wait 挂 PIPE_S）"},
    {"both", 2, M15AP::AP_A2V_GEMM,
     "attention：AIC→其配对 2 个 AIV「4 个 GEMM 的 GM 输出可见」（AIV 侧只 wait；AIC 挂 PIPE_MTE2）"},
};
constexpr uint32_t FLAG_SEQ_ATTN_PA_N = sizeof(FLAG_SEQ_ATTN_PA) / sizeof(FLAG_SEQ_ATTN_PA[0]);
static_assert(FLAG_SEQ_ATTN_PA_N == 2u, "attention 相位 A 的同步点应是 2 个（1 个 AIC mode0 + 1 个 mode2）");

constexpr bool FlagAttnPhaseOk()
{
    // ① 号必须在 0..15
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_PA_N; ++i) {
        if (FLAG_SEQ_ATTN_PA[i].id >= FLAG_PER_CORE) {
            return false;
        }
    }
    // ② 子序列内部：同一 (核型, mode) 的相邻两步不同号
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_PA_N; ++i) {
        for (uint32_t j = i + 1; j < FLAG_SEQ_ATTN_PA_N; ++j) {
            if (FLAG_SEQ_ATTN_PA[i].mode == FLAG_SEQ_ATTN_PA[j].mode &&
                CoreSame(FLAG_SEQ_ATTN_PA[i].core, FLAG_SEQ_ATTN_PA[j].core)) {
                if (FLAG_SEQ_ATTN_PA[i].id == FLAG_SEQ_ATTN_PA[j].id) {
                    return false;
                }
                break;
            }
        }
    }
    // ③ 与相位 A 两侧的**同一个 (核型, mode) 子空间**里的 hc 同步点相邻：
    //    上一个 AIC mode0 = hc(H1) 的最后一条 FLAG_AC3；下一个 = hc(H2) 的第一条 FLAG_AC0；
    //    上一个 mode2     = hc(H1) 的最后一条 FLAG_C2A_GATE；下一个 = hc(H2) 的第一条 FLAG_A2C_XN。
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_PA_N; ++i) {
        if (FLAG_SEQ_ATTN_PA[i].mode == 0u && CoreSame(FLAG_SEQ_ATTN_PA[i].core, "AIC")) {
            if (FLAG_SEQ_ATTN_PA[i].id == M15H::FLAG_AC3 || FLAG_SEQ_ATTN_PA[i].id == M15H::FLAG_AC0) {
                return false;
            }
        }
        if (FLAG_SEQ_ATTN_PA[i].mode == 2u) {
            if (FLAG_SEQ_ATTN_PA[i].id == M15H::FLAG_C2A_GATE ||
                FLAG_SEQ_ATTN_PA[i].id == M15H::FLAG_A2C_XN) {
                return false;
            }
        }
    }
    return true;
}
static_assert(FlagAttnPhaseOk(),
              "attention 相位 A 的 flagId 分节违规：与 hc 的相邻同步点撞号（见 evidence/attn_wire/README.md §1）");

// 相位 A 的注意力段在 KIND_ATTN 层上**取代** GDN 段 ⇒ 它用掉的号必须是「GDN 段在该层不执行」
// 的那一批或「两侧都已 drain」的那一批。下面三条把它钉死（值变了就必须重做分节）。
static_assert(M15AP::AP_A2V_GEMM == M15G::FLAG_BOUND_INPROJ,
              "attention 的 mode2 号必须落进 GDN 段在 KIND_ATTN 层不执行的那一批（4..7）");
static_assert(M15AP::AP_A2V_GEMM >= 4u && M15AP::AP_A2V_GEMM <= 7u, "attention 的 mode2 号必须在 GDN 段的 4..7 内");
static_assert(M15AP::AP_AIC_M0_OUT >= M15H::FLAG_AC0 && M15AP::AP_AIC_M0_OUT <= M15H::FLAG_AC3,
              "attention 的 AIC mode0 号必须落进 hc 相位 A 两侧都已 drain 的 0..3 内");

// 资源窗（相位内独占；峰值口径与 §2/§3 一致：取 max 而非求和）
static_assert(M15AP::UB_AP_END <= UB_TOTAL_BYTES, "attention 相位 A 的 UB 窗超出 248 KB");
static_assert(M15AP::AP_L1_B_REGION + 2u * M15AP::AP_L1_B_ELEMS * 2u <= M15G::L1_BYTES_TOTAL,
              "attention 相位 A 的 L1 占用超出 512 KB");
static_assert(M15AP::AP_BASE_M * M15AP::AP_BASE_N * 4u <= L0C_TOTAL_BYTES,
              "attention 相位 A 的 L0C tile 超出 256 KB");

// ------------------------------------------------------------
// 4c. M110（Wave A）：**mode-4 分节**（attention core 家族；35 处 cross-core 的显式登记）
// ------------------------------------------------------------
// 事实来源 = M101 的分支 `feat/m101-attention-core-segment-lift-and-o-p`（**未合入**）的生成物
// `m15_layer_loop/m15_attn_core.h` + M101 复审 r1 的 F2（含 CANN 官方 spec 对 flagId 取值范围的引用）。
// M110 在**未合入分支**上实跑的读数（命令与输出见 `evidence/prefill_contract/README.md`）：
//   · 35 次 CrossCore 调用（`AttnCoreAic` 21 次 + `AttnCoreAiv` 14 次）：
//     Set mode0/FIX ×2、Set mode2/FIX ×3、Set mode2/MTE3 ×5、Set mode4/FIX ×4、Set mode4/MTE3 ×1、
//     Set mode4/V ×4、Wait mode0/S ×2、Wait mode2/MTE2 ×2、Wait mode2/S ×4、
//     Wait mode4/FIX ×4、Wait mode4/MTE2 ×2、Wait mode4/V ×2
//   · 段内 9 个常量：`CC_MM0=0 / CC_MM1=1 / CC_P0=5 / CC_P1=6 / CC_P2=7 / CC_BAR=8 /
//     CC_AIVDONE=9 / CC_RDY=10 / CC_ALLDONE=11`，且 mode-4 的 AIC 侧还要 `+AIV_CH(=16)`。
// **它必须改造才能进融合 TU**（M101 自己的头也写了这条），两个原因：
//   ① **mode2 的 9/10/11 与 hc 段 mode2 的 8/9/10/11 在同一个 (核型, mode) 子空间里撞号**；
//   ② mode4 是一个**新的 mode**：AIV 侧与 mode0/2 共用 0..15 同一个物理池（跨模式复用须先 drain），
//      AIC 侧才扩到 0..31（16..31 → 配对 AIV1）。
// ⇒ 本节给出**重号映射**（裁决 ①）与 **mode-4 的显式登记 + 上限裁决**（裁决 ②）。
//
// **重号映射（M110 的裁决）**：mode2 的三个号落在 `{4,5,6,7}` —— 这一批在 KIND_ATTN 层上
// **GDN 段整段不执行**，与 M97 给前端 prolog 的号是同一批（`M15AP::AP_A2V_GEMM = 5`，见 §4b）。
// 4 个槽、前端用 1 个、core 用 3 个 ⇒ **恰好占满**，故下面有"不与前端撞号"的逐条断言。
constexpr uint16_t ATTN_CORE_M2_AIVDONE = 4;   // 原 CC_AIVDONE(=9) 重号
constexpr uint16_t ATTN_CORE_M2_RDY = 6;       // 原 CC_RDY(=10) 重号
constexpr uint16_t ATTN_CORE_M2_ALLDONE = 7;   // 原 CC_ALLDONE(=11) 重号
// mode0（AIC 全体对齐，原 CC_BAR=8）原样保留：它落进 hc 相位 A 两侧都已 drain 的 8；
// 同时也就是 MoE 的 AIC mode0 ring 的首号（下面有断言把这条复用钉住）。
constexpr uint16_t ATTN_CORE_M0_AIC_BAR = 8;
// mode4 的两个见证量：AIV 侧用到的最大号（0/1/5/6/7 ⇒ 7）与 AIC 侧的最小"高号"（16 = ccMM + AIV_CH）。
constexpr uint16_t ATTN_CORE_M4_AIV_MAX = 7;
constexpr uint16_t ATTN_CORE_M4_AIC_HI_LO = 16;
static_assert(M15M::FLAG_AIC_SEG_RING[0] == ATTN_CORE_M0_AIC_BAR,
              "attention core 的 AIC mode0 号（CC_BAR=8）必须落进 MoE 的 AIC mode0 ring 首号（复用见证）");
struct FlagStepM4 {
    const char* core;   // "AIV" / "AIC"
    uint32_t    mode;   // 0 / 2 / 4
    uint16_t    id;     // 该行覆盖的**最小** id
    uint16_t    cov;    // 覆盖的连续 id 个数（1 = 单号；`ccMM/ccP` 这类由 parity 选号的调用点覆盖 2/3 个）
    const char* dir;    // "set" / "wait"
    const char* pipe;
    uint32_t    count;  // **调用点**数（一个调用点由 parity 覆盖多个 id 时只计一次；口径见下面的断言）
    const char* use;
};
// **计数口径**（这条决定了 35 这个数怎么来的，必须写死）：`count` 数的是核体里的
// `CrossCoreSetFlag/WaitFlag` **调用点**，不是 (调用点 × id) 的笛卡尔积。
// `ccMM ∈ {CC_MM0, CC_MM1}` / `ccP ∈ {CC_P0, CC_P1, CC_P2}` 由 parity 在**不同迭代**里选号，
// 一个调用点在整段里先后落到多个 id 上 ⇒ 用 `cov` 记覆盖个数，用 `count` 记调用点数。
constexpr FlagStepM4 FLAG_SEQ_ATTN_CORE[] = {
    // ---- AttnCoreAic：21 个调用点 ----
    {"AIC", 0, ATTN_CORE_M0_AIC_BAR, 1, "set",  "PIPE_FIX",  2, "CC_BAR：全体 AIC 对齐（FIXP 写 GM 排空）"},
    {"AIC", 0, ATTN_CORE_M0_AIC_BAR, 1, "wait", "PIPE_S",    2, "CC_BAR：同上"},
    {"AIC", 2, ATTN_CORE_M2_RDY,     1, "set",  "PIPE_FIX",  3, "CC_RDY：放行配对 AIV 做 combine（重号 6）"},
    {"AIC", 2, ATTN_CORE_M2_AIVDONE, 1, "wait", "PIPE_S",    2, "CC_AIVDONE：配对 2 AIV 的 partial 已落 GM（重号 4）"},
    {"AIC", 2, ATTN_CORE_M2_ALLDONE, 1, "wait", "PIPE_S",    2, "CC_ALLDONE：等 AIV 收尾（重号 7）"},
    {"AIC", 4, 0,  2, "wait", "PIPE_FIX",  2, "CC_MM{0,1}：S(t) 就绪 / 已读完（AIV0 通道）"},
    {"AIC", 4, 16, 2, "wait", "PIPE_FIX",  2, "CC_MM{0,1} + AIV_CH = AIC→**AIV1** 通道（id 16/17）"},
    {"AIC", 4, 0,  2, "set",  "PIPE_FIX",  2, "PV(t) 就绪（AIV0 通道，id 0/1）"},
    {"AIC", 4, 16, 2, "set",  "PIPE_FIX",  2, "PV(t) 就绪（AIV1 通道，id 16/17）"},
    {"AIC", 4, 5,  3, "wait", "PIPE_MTE2", 1, "CC_P{0,1,2}：AIV0 的 P(t) 已写入 GM（id 5/6/7）"},
    {"AIC", 4, 21, 3, "wait", "PIPE_MTE2", 1, "CC_P{0,1,2} + AIV_CH（**AIV1** 通道，id 21/22/23）"},
    // ---- AttnCoreAiv：14 个调用点 ----
    {"AIV", 2, ATTN_CORE_M2_AIVDONE, 1, "set",  "PIPE_MTE3", 4, "CC_AIVDONE：AIV 的 ws 写完 → AIC（重号 4）"},
    {"AIV", 2, ATTN_CORE_M2_ALLDONE, 1, "set",  "PIPE_MTE3", 1, "CC_ALLDONE：AIV 收尾 → AIC（重号 7）"},
    {"AIV", 2, ATTN_CORE_M2_RDY,     1, "wait", "PIPE_MTE2", 2, "CC_RDY：等 AIC 放行 combine（重号 6）"},
    {"AIV", 4, 0, 1, "set",  "PIPE_V",    2, "S(t) 已消费 / PV 就绪（AIV 本地号；两处"},
    {"AIV", 4, 1, 1, "set",  "PIPE_V",    2, "同上（id 1；与 id 0 成一对）"},
    {"AIV", 4, 0, 1, "wait", "PIPE_V",    2, "等 AIC 的 S(t)（本地号，两处）"},
    {"AIV", 4, 5, 3, "set",  "PIPE_MTE3", 1, "CC_P{0,1,2}：P 已落 GM（id 5/6/7）"},
};
constexpr uint32_t FLAG_SEQ_ATTN_CORE_N = sizeof(FLAG_SEQ_ATTN_CORE) / sizeof(FLAG_SEQ_ATTN_CORE[0]);
// 登记表覆盖的调用点数必须与核体的实测一致（这条把"逐处点名"变成可编译期核对的量）
constexpr uint32_t FlagSeqAttnCoreCalls()
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_CORE_N; ++i) {
        n += FLAG_SEQ_ATTN_CORE[i].count;
    }
    return n;
}
static_assert(FlagSeqAttnCoreCalls() == 35u, "attention core 的 cross-core 调用点数 ≠ 35（登记表与核体不一致）");
// AIC 侧 21 / AIV 侧 14（核体实测），分开断言：只对上总数会对不上"哪一侧多了一处"
constexpr uint32_t FlagSeqAttnCoreCallsOf(const char* core)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_CORE_N; ++i) {
        n += (CoreId(FLAG_SEQ_ATTN_CORE[i].core) == CoreId(core)) ? FLAG_SEQ_ATTN_CORE[i].count : 0u;
    }
    return n;
}
static_assert(FlagSeqAttnCoreCallsOf("AIC") == 21u && FlagSeqAttnCoreCallsOf("AIV") == 14u,
              "attention core 的 AIC/AIV 调用点数 ≠ 21/14（核体实测读数）");

constexpr bool FlagSeqAttnCoreOk()
{
    // ① id（含 parity 覆盖到的末尾）必须在**该 (核型, mode) 的**上限内 ——
    //    这一条就是 F2 的解：AIC mode4 的 16/17/21/22/23 合法，AIV 的 16 仍非法
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_CORE_N; ++i) {
        const uint32_t limit = FlagIdLimit(FLAG_SEQ_ATTN_CORE[i].core, FLAG_SEQ_ATTN_CORE[i].mode);
        if (static_cast<uint32_t>(FLAG_SEQ_ATTN_CORE[i].id) + FLAG_SEQ_ATTN_CORE[i].cov > limit) {
            return false;
        }
    }
    // ② 段内独占：mode2 的三个重号必须落进 {4..7}（GDN 段在 KIND_ATTN 层不执行的那一批）
    if (ATTN_CORE_M2_AIVDONE < 4u || ATTN_CORE_M2_AIVDONE > 7u) { return false; }
    if (ATTN_CORE_M2_RDY < 4u || ATTN_CORE_M2_RDY > 7u) { return false; }
    if (ATTN_CORE_M2_ALLDONE < 4u || ATTN_CORE_M2_ALLDONE > 7u) { return false; }
    return true;
}
static_assert(FlagSeqAttnCoreOk(), "attention core 的 flagId 分节违规（mode-4 上限 / mode2 重号区间）");
// 与非收缩前端（M97 的 prolog）**在同一层**、同一 mode2 子空间：三个重号不得与它撞
static_assert(ATTN_CORE_M2_AIVDONE != M15AP::AP_A2V_GEMM && ATTN_CORE_M2_RDY != M15AP::AP_A2V_GEMM &&
                  ATTN_CORE_M2_ALLDONE != M15AP::AP_A2V_GEMM,
              "attention core 的 mode2 重号与前端的 AP_A2V_GEMM 撞号（两者在 KIND_ATTN 层的相位 A 里相邻）");
static_assert(ATTN_CORE_M2_AIVDONE != ATTN_CORE_M2_RDY && ATTN_CORE_M2_RDY != ATTN_CORE_M2_ALLDONE &&
                  ATTN_CORE_M2_AIVDONE != ATTN_CORE_M2_ALLDONE,
              "attention core 自己的 mode2 三个号必须互异");
// mode-4 的 AIV 侧号必须在 0..15（与 mode0/2 同一个物理池 ⇒ 跨模式复用须"前一 mode 全部 drain"）
static_assert(ATTN_CORE_M4_AIV_MAX < FLAG_PER_CORE,
              "mode-4 的 AIV 侧号越出 0..15（AIV 的池没有变宽）");
// mode-4 的 AIC 侧**确实用到了** 16..31：这两个号必须 >15（否则上面那条裁决没被见证）
static_assert(ATTN_CORE_M4_AIC_HI_LO > 15u && ATTN_CORE_M4_AIC_HI_LO < FLAG_PER_CORE_AIC_MODE4,
              "AIC mode-4 的高号（16..31）没有被登记表见证 —— 裁决等于没生效，或号被改小了");

// ------------------------------------------------------------
// 4d. M110（Wave A）：**prefill 的 flagId 分节**
// ------------------------------------------------------------
// 结构前提（`docs/15` §5 方案 B / M103-0 第 4 条）：prefill 走**另一个入口符号**
// （`ENTRY_GDN_PREFILL` / `ENTRY_ATTN_PREFILL`），与 decode 的入口在 host 侧互斥选择 ⇒ 同一次
// 启动里只可能有一档的同步点存在。**但"不同时跑"是口头论证** —— `docs/19` §4.1 约束 2 明确要求
// 「能否复用必须用 static_assert 序列显式见证」。下面把它变成两个可编译期核对的量：
//   · **复用见证**：prefill 用到的每一个 (核型, mode, id) 都必须在 decode 档的 `FLAG_SEQ[]` 里
//     **已登记**（= 不新增号）；
//   · **形态见证**：mode2 ⊆ {4..7}（KIND_ATTN 层上 GDN 不执行的那一批）之类的位置断言。
struct FlagIdRef {
    const char* seg;
    const char* core;
    uint32_t    mode;
    uint16_t    id;
    const char* dir;     // "set" / "wait" / "both"（B1 的具体配对由它自己的段序定，见 §1e 的边界说明）
    const char* use;
};

// B1（GDN prefill，`m15_gdn_prefill.h` 的 `M15GP` 段体）：登记的 6 个都是 mode-2 同步点
// （每 job 一对：AIV set `GO*`、AIC set `DONE*`）；本表 6 行逐行可核，没有 mode-0 行。
//
// **订正记录（M138；本条即 M132 自立的 finding「§4d GDN-prefill flag 登记与 B1 实现不符」）**：
// 本表原先登记的是 **decode GDN 段**的 12 个号（`m15_gdn_resources.h:154-165` 的
// `FLAG_BOUND_*` / `FLAG_AIV_SEG*` / `FLAG_AIC_SEG*`：mode2 4-7 + AIV mode0 8-11 + AIC mode0 12-15），
// 而 B1 段体实际用的是 `m15_gdn_resources.h:396-401` 的 `GP_FLAG_GO1..DONE3`（值 0..5，**全部 mode2**）。
// 两套号不同源：B1 的三段 job 自带全部核间同步、不依赖外层全体 AIV 的 mode-0 barrier
// （`m23_gdn_prefill/README.md` §3(d)/§3(e)）。核对范围 = `m15_gdn_prefill.h` 的全部
// `CcSet`/`CcWait` 调用点（`:571-585`、`:945-1069`）：这些点读到的都是 `GP_CC_MODE2` 与
// `GP_FLAG_*`；在该范围内未读到 mode-0 的 `CrossCoreSet/WaitFlag` 调用。
// ⇒ 本表按 B1 的真实号重写，"复用见证"（`PfGdnReuseOk()`）从此钉在它真正用的号上：
//   0..3 已在 decode 的 MoE mode2 登记、4..5 已在 decode 的 GDN mode2 登记（见上面的 `FLAG_SEQ[]`）。
// **不覆盖**：B1 的 6 个号与 prefill 另一臂 B4（MoE-prefill，mode2 0..3，`m15_moe_prefill_res.h:326`）
// 在 **id 0..3 上同名**，但两者分处相位 A / 相位 B、中间隔着 H2 段与边界号，跨表相邻性不在本判据内
// （§4e 的"不覆盖"划界）。「chunk 轴串行 + AIV→L1 操作数交接」是 B1 段内的流水，**它自己的相邻性归它**。
constexpr FlagIdRef FLAG_SEQ_PREFILL_GDN[] = {
    {"GDN-PF", "both", 2, M15G::GP_FLAG_GO1,   "set",  "AIV→AIC：chunk 输入就绪（J1 可跑）"},
    {"GDN-PF", "both", 2, M15G::GP_FLAG_DONE1, "wait", "AIC→AIV：kk / qk 就绪"},
    {"GDN-PF", "both", 2, M15G::GP_FLAG_GO2,   "set",  "AIV→AIC：ST / w 已上传（J2 可跑）"},
    {"GDN-PF", "both", 2, M15G::GP_FLAG_DONE2, "wait", "AIC→AIV：(w·S)ᵀ / q·S 就绪"},
    {"GDN-PF", "both", 2, M15G::GP_FLAG_GO3,   "set",  "AIV→AIC：KT' / ABM / DT 已上传（J3 可跑）"},
    {"GDN-PF", "both", 2, M15G::GP_FLAG_DONE3, "wait", "AIC→AIV：AB·d / (kᵀ·DP)ᵀ 就绪"},
};
constexpr uint32_t FLAG_SEQ_PREFILL_GDN_N = sizeof(FLAG_SEQ_PREFILL_GDN) / sizeof(FLAG_SEQ_PREFILL_GDN[0]);
static_assert(FLAG_SEQ_PREFILL_GDN_N == 6u,
              "GDN prefill 的同步点数变了（B1 段体为 3 个 job × 2 = 6 个 mode-2 号）");

// 形态见证（把 B1 的"只有 mode-2、没有 mode-0"从散文变成编译期判据）：本表每行都须是 mode 2。
// 若有人把 decode GDN 段的 mode-0 号（`FLAG_AIV_SEG*` / `FLAG_AIC_SEG*`）搬回来，这条会变红。
constexpr bool FlagSeqPrefillGdnAllMode2()
{
    for (uint32_t i = 0; i < FLAG_SEQ_PREFILL_GDN_N; ++i) {
        if (FLAG_SEQ_PREFILL_GDN[i].mode != 2u) {
            return false;
        }
    }
    return true;
}
static_assert(FlagSeqPrefillGdnAllMode2(),
              "GDN prefill 登记里出现 mode != 2 的行 —— B1 段体（m15_gdn_prefill.h）没有 mode-0 同步点");

// B2（attn prefill 前端）+ B3（稠密 causal core）：前端复用 M97 已登记的 2 个，
// core 用 §4c 重号后的 3 个（mode2）+ 1 个（AIC mode0）。**本表只覆盖 mode0/2 的前端与 M101 变体
// （`m15_attn_core.h` 的 35 处）；B3 核体（`m15_attn_fa_core.h`）自己的 **mode-2** 通道族是**另一套号**
// （`CC_S_RDY..CC_P1`），落在 §4f.2，不在这张 mode0/2 的表里（M168 分节、M185 改 mode 2）。
// **M197 补登记**（m24_attn_prefill/README.md §5.5 的“Wave C 要补登记”清单）：B2 前端实际使用的
// `(both,2,4)`=`AP_V2A_READY` / `(AIV,0,10)`=`FLAG_AIV_SEG_S1` / `(AIV,0,11)`=`FLAG_AIV_SEG2`
// 三条此前只在 `.asc` 的 B2 段体里调用、**没进本表** ⇒ 登记表与实际使用不一致（少 3 条）。
// 行序 = **KIND_ATTN 相位 A 的执行序**（前端 → core），相邻性由 `FlagRefsAdjacentOk` 见证。
constexpr FlagIdRef FLAG_SEQ_PREFILL_ATTN[] = {
    {"ATN-PF", "AIC",  0, M15AP::AP_AIC_M0_OUT,   "both", "前端：4 个 GEMM 的排空 + 全体 AIC 对齐（M97 已登记）"},
    {"ATN-PF", "both", 2, M15AP::AP_A2V_GEMM,     "both", "前端：AIC→配对 AIV「GM 输出可见」（M97 已登记）"},
    {"ATN-PF", "AIV",  0, M15G::FLAG_AIV_SEG_S1,  "both", "前端 B1：in 输入面（环行 + K/V 行）全体就位（M197 补登记）"},
    {"ATN-PF", "AIV",  0, M15G::FLAG_AIV_SEG2,    "both", "前端 B2：block 0 读完 in 前别的 AIV 不得开下一 chunk（M197 补登记）"},
    {"ATN-PF", "both", 2, M15AP::AP_V2A_READY,    "both", "前端：AIV→配对 AIC「y0 已消费，可覆写」（M197 起启用；无它则 AIC 下一 chunk 的 GEMM 会覆写在读的 y0）"},
    {"ATN-PF", "AIC",  0, ATTN_CORE_M0_AIC_BAR,   "both", "core：全体 AIC 对齐（CC_BAR，重号表见 §4c）"},
    {"ATN-PF", "AIC",  2, ATTN_CORE_M2_RDY,       "both", "core：放行配对 AIV 做 combine"},
    {"ATN-PF", "AIC",  2, ATTN_CORE_M2_AIVDONE,   "both", "core：AIV 的 partial 已落 GM"},
    {"ATN-PF", "AIC",  2, ATTN_CORE_M2_ALLDONE,   "both", "core：AIV 收尾"},
};
constexpr uint32_t FLAG_SEQ_PREFILL_ATTN_N = sizeof(FLAG_SEQ_PREFILL_ATTN) / sizeof(FLAG_SEQ_PREFILL_ATTN[0]);
static_assert(FLAG_SEQ_PREFILL_ATTN_N == 9u, "attn prefill 的 mode0/mode2 同步点数变了（应为 5 + 4）");

// decode 档在**同一个 (核型, mode)** 子空间里登记过这个 id 吗？（= 复用的定义）
constexpr bool FlagIdRegisteredDecode(const char* core, uint32_t mode, uint16_t id)
{
    for (uint32_t i = 0; i < FLAG_SEQ_N; ++i) {
        if (FLAG_SEQ[i].mode == mode && FLAG_SEQ[i].id == id && CoreSame(FLAG_SEQ[i].core, core)) {
            return true;
        }
    }
    return false;
}
constexpr bool PfGdnReuseOk()
{
    for (uint32_t i = 0; i < FLAG_SEQ_PREFILL_GDN_N; ++i) {
        if (!FlagIdRegisteredDecode(FLAG_SEQ_PREFILL_GDN[i].core, FLAG_SEQ_PREFILL_GDN[i].mode,
                                    FLAG_SEQ_PREFILL_GDN[i].id)) {
            return false;
        }
        if (FLAG_SEQ_PREFILL_GDN[i].id >= FlagIdLimit(FLAG_SEQ_PREFILL_GDN[i].core, FLAG_SEQ_PREFILL_GDN[i].mode)) {
            return false;
        }
    }
    return true;
}
static_assert(PfGdnReuseOk(), "GDN prefill 的某个 flagId 不在 decode 档已登记的集合里 —— 复用未经见证（新增了号）");

constexpr bool PfAttnReuseOk()
{
    for (uint32_t i = 0; i < FLAG_SEQ_PREFILL_ATTN_N; ++i) {
        const FlagIdRef& r = FLAG_SEQ_PREFILL_ATTN[i];
        // mode2 的四个号必须落进 {4..7}（KIND_ATTN 层上 GDN 段整段不执行的那一批）
        if (r.mode == 2u && (r.id < 4u || r.id > 7u)) {
            return false;
        }
        // AIC mode0 的号必须落进 hc 的 0..3（两侧都已 drain）或 MoE 的 AIC ring 8..11
        if (r.mode == 0u && CoreId(r.core) == 1u) {
            const bool inHc = (r.id >= M15H::FLAG_AC0 && r.id <= M15H::FLAG_AC3);
            const bool inMoeRing = (r.id >= M15M::FLAG_AIC_SEG_RING[0] && r.id <= M15M::FLAG_AIC_SEG_RING[3]);
            if (!inHc && !inMoeRing) {
                return false;
            }
        }
        if (r.id >= FlagIdLimit(r.core, r.mode)) {
            return false;
        }
    }
    return true;
}
static_assert(PfAttnReuseOk(), "attn prefill 的 flagId 分节违规（mode2 未落进 4..7 / AIC mode0 未落进 hc 或 MoE 的窗）");
// M197：新补登记的 3 个前端号必须在 decode 档 `FLAG_SEQ[]` 里**已登记过**（复用见证，
// 不是新占池）—— 与 `PfGdnReuseOk` 同一条"登记表 ↔ 实际使用一致"的纪律。
static_assert(FlagIdRegisteredDecode("both", 2u, M15AP::AP_V2A_READY), "AP_V2A_READY 不在 decode 档（复用未经见证）");
static_assert(FlagIdRegisteredDecode("AIV", 0u, M15G::FLAG_AIV_SEG_S1), "FLAG_AIV_SEG_S1 不在 decode 档（复用未经见证）");
static_assert(FlagIdRegisteredDecode("AIV", 0u, M15G::FLAG_AIV_SEG2), "FLAG_AIV_SEG2 不在 decode 档（复用未经见证）");

// **一条必须写下来的前提**（不是断言，是使用契约）：mode 0 与 mode 2 在**每核共用 0..15
// 同一个物理池** ⇒ B3（**M185 起用 mode 2**，不再用 mode 4）在相位 A 里与前端/hc 的
// mode0/mode2 交替使用前，必须先保证前一段的 set/wait 全部 drain（`docs/05 §2` 第 3 条）。
// B3 的进入点（相位 A 的开头）恰好是全体 AIV 的 mode-0 边界之后，这条前提在那里成立；
// ⚠ **接线时的遗留**：B3 的 mode-2 号（`CC_S_RDY..CC_P1` = 0/1/2/3/5/6）与 GDN(4..7)/MoE(0..3) 的
// mode-2 号段**有重叠** —— B3 现未接进融合 TU（`ENTRY_TABLE` 的 attn-prefill 仍 `wired=0`），
// Wave C 连线时必须把 B3 的号一并打平（或改成与共跑段不重叠的一段）。

// ------------------------------------------------------------
// 4e. M110（Wave A）：新分节的**相邻性**与**用量**收口（r1 复审 P2-1 的修法）
// ------------------------------------------------------------
// 复审抓出的缺口：§4c/§4d 只落了**容量侧**（`id + cov <= FlagIdLimit`）、**位置侧**（mode2 ⊂ {4..7}、
// AIC mode0 ⊂ hc 0..3 ∪ MoE 8..11）与**复用见证**，而**没有落**「同一 (核型, mode) 的相邻同步点
// 不得同号」这一条 —— 任务 2 的原话是「每条 id 相邻性/容量都要有 static_assert」，而
// `FlagSeqAdjacentOk()` **只遍历** decode 档的 `FLAG_SEQ[]`，三张新表不在它的判据范围内。
//
// **为什么不能"把 `FlagSeqAdjacentOk()` 套到新表上"就完事**（复审提醒、我这里给结论与事实）：
//   那条判据的粒度是「**一行 = 一个逻辑同步点**（一次 set 配一次 wait）」，而 §4c 的表是
//   **事件级**（一行 = 一次 set 或一次 wait，还带 count）⇒ 直接喂进去，`(AIC,0,8,set)` 与
//   `(AIC,0,8,wait)` 会被读成"相邻同号"**假红**。硬件真正的危险面是**相邻两次 `set` 同号**
//   （M2 实证：同核连续 set 不保序）⇒ 事件级的表要用"**只比 set、按 (核型, mode) 分组**"的判据。
// ⇒ 本节给**两个粒度**各自的判据（`FlagRefsAdjacentOk` / `FlagEventsAdjacentOk`），把三张表
//   全覆盖上，并给**正负两侧**的对照表（任一条不成立 ⇒ 编译期失败）。
//
// **一处事实订正（r1 复审 P2-1 的前提）**：复审写「前端 AIC mode0 的 `AP_AIC_M0_OUT` 与 core 的
// `ATTN_CORE_M0_AIC_BAR(CC_BAR)` **都是 8**」。实测**不是** —— `M15AP::AP_AIC_M0_OUT = 1`
// （`m15_attn_prolog.h:218`，与 hc 的 `FLAG_AC1` 同号）、`ATTN_CORE_M0_AIC_BAR = 8`（本文件 §4c）。
// 两者不同号 ⇒ AIC mode0 子空间在 KIND_ATTN 层的相邻逻辑点是 `1 → 8`，**本来就满足**相邻性判据
// （下面把它跑成断言）。把这条订正**可执行化**：`static_assert(AP_AIC_M0_OUT != ATTN_CORE_M0_AIC_BAR)`
// —— 若 M15AP 将来改成 8，它会立刻编译期变红，而不是等人手推。
//
// **覆盖与不覆盖（显式划界，不许再"没说"）**：
//   · **覆盖**：§4c 事件表（相邻两次 `set` 同号）、§4d 两张逻辑点表（相邻逻辑点同号）、
//     三张表各自的**每 (核型, mode, id) 用量**（硬件 4bit 计数器 ≤ 15）+ id 上限。
//   · **不覆盖**：decode 档的 `FLAG_SEQ[]` —— 它仍由原来的 `FlagSeqAdjacentOk()` 管，**本 mission
//     一个字没改它**（零回归）。"把两档合并成一条 48 层执行序再比一遍"需要跨段相位插桩，
//     不在本 mission（`docs/15` M103-2.3 把这一条算 Wave C 的收口）。
//   · **没有做**"sets 数 == waits 数"这种 drain 见证：mode2 的配对本来就是 **n:m** 语义
//     （`m15_attn_core.h` 的头注写明 `CC_AIVDONE` 是"2 set 配 1 wait"）⇒ 这条见证会**假红**。
//     故只保留 §4d 末尾那一段"用 mode4 之前必须 drain mode0/2"的**使用契约**。
constexpr bool FlagDirIsSet(const char* d) { return d[0] == 's'; }   // "set" → true；"wait" → false

// ① 逻辑同步点表（一行 = 一个 set/wait 配对的同步点）：相邻逻辑点不得同号
constexpr bool FlagRefsAdjacentOk(const FlagIdRef* seq, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        if (seq[i].id >= FlagIdLimit(seq[i].core, seq[i].mode)) {
            return false;
        }
        for (uint32_t j = i + 1; j < n; ++j) {
            if (seq[i].mode == seq[j].mode && CoreSame(seq[i].core, seq[j].core)) {
                if (seq[i].id == seq[j].id) {
                    return false;
                }
                break;
            }
        }
    }
    return true;
}

// ② 事件级表（一行 = 一次 set 或一次 wait，带 count）：**相邻两次 set** 不得同号
constexpr bool FlagEventsAdjacentOk(const FlagStepM4* seq, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t limit = FlagIdLimit(seq[i].core, seq[i].mode);
        if (static_cast<uint32_t>(seq[i].id) + seq[i].cov > limit) {
            return false;
        }
        if (!FlagDirIsSet(seq[i].dir)) {
            continue;   // wait 与 set 之间是**配对**关系，不是相邻同步点
        }
        for (uint32_t j = i + 1; j < n; ++j) {
            if (seq[j].mode == seq[i].mode && CoreSame(seq[j].core, seq[i].core) && FlagDirIsSet(seq[j].dir)) {
                if (seq[i].id == seq[j].id) {
                    return false;
                }
                break;
            }
        }
    }
    return true;
}

// ③ 用量：事件表里每个 (核型, mode, id) 的 **set 次数**（= 4bit 计数器真正会计的量）
constexpr uint32_t FlagEventsSetUse(const FlagStepM4* seq, uint32_t n, uint32_t i)
{
    uint32_t c = 0;
    for (uint32_t j = 0; j < n; ++j) {
        if (seq[j].mode == seq[i].mode && CoreSame(seq[j].core, seq[i].core) && seq[j].id == seq[i].id &&
            FlagDirIsSet(seq[j].dir)) {
            c += seq[j].count;
        }
    }
    return c;
}
constexpr uint32_t FlagEventsMaxSetUse(const FlagStepM4* seq, uint32_t n)
{
    uint32_t mx = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t c = FlagEventsSetUse(seq, n, i);
        mx = (c > mx) ? c : mx;
    }
    return mx;
}

static_assert(FlagRefsAdjacentOk(FLAG_SEQ_PREFILL_GDN, FLAG_SEQ_PREFILL_GDN_N),
              "GDN prefill 的 flagId 相邻性违规：同一 (核型, mode) 的相邻逻辑同步点同号");
static_assert(FlagRefsAdjacentOk(FLAG_SEQ_PREFILL_ATTN, FLAG_SEQ_PREFILL_ATTN_N),
              "attn prefill 的 flagId 相邻性违规：同一 (核型, mode) 的相邻逻辑同步点同号");
static_assert(FlagEventsAdjacentOk(FLAG_SEQ_ATTN_CORE, FLAG_SEQ_ATTN_CORE_N),
              "attention core 的 flagId 相邻性违规：同一 (核型, mode) 的相邻两次 set 同号");
static_assert(FlagEventsMaxSetUse(FLAG_SEQ_ATTN_CORE, FLAG_SEQ_ATTN_CORE_N) <= 15u,
              "attention core 的某个 (核型, mode, id) 的 set 次数超过硬件 4bit 计数器上限 15");
// 前端与 core 在 AIC mode0 子空间上是**相邻的两个逻辑同步点**：必须异号（r1 复审前提的订正，见上）
static_assert(M15AP::AP_AIC_M0_OUT != ATTN_CORE_M0_AIC_BAR,
              "前端与 core 的 AIC mode0 号相同 —— 相邻逻辑同步点同号（相邻性判据会变红）");

// ④ **判据的正负两侧**（随代码入库：任一条不成立就编译不过 ⇒ 判据不是"恒真的装饰"）
constexpr FlagIdRef PF_ADJ_NEG[] = {
    {"NEG", "AIC", 2, 4, "both", "人为构造：相邻逻辑点同号（负向对照）"},
    {"NEG", "AIC", 2, 4, "both", "人为构造：相邻逻辑点同号（负向对照）"},
};
constexpr FlagIdRef PF_ADJ_POS[] = {
    {"NEG", "AIC", 2, 4, "both", "相邻逻辑点异号（正向对照）"},
    {"NEG", "AIC", 2, 6, "both", "相邻逻辑点异号（正向对照）"},
};
static_assert(!FlagRefsAdjacentOk(PF_ADJ_NEG, 2u), "负向对照没被抓住：相邻逻辑点同号竟然通过");
static_assert(FlagRefsAdjacentOk(PF_ADJ_POS, 2u), "正向对照被误判：相邻逻辑点异号竟然不通过");
constexpr FlagStepM4 PF_EV_NEG[] = {
    {"AIC", 4, 0, 2, "set", "PIPE_FIX", 1, "人为构造：相邻两次 set 同号（负向对照）"},
    {"AIC", 4, 0, 2, "set", "PIPE_FIX", 1, "人为构造：相邻两次 set 同号（负向对照）"},
};
constexpr FlagStepM4 PF_EV_POS[] = {
    {"AIC", 4, 0, 2, "set", "PIPE_FIX", 1, "相邻两次 set 异号（正向对照）"},
    {"AIC", 4, 16, 2, "set", "PIPE_FIX", 1, "相邻两次 set 异号（正向对照）"},
};
static_assert(!FlagEventsAdjacentOk(PF_EV_NEG, 2u), "负向对照没被抓住：相邻两次 set 同号竟然通过");
static_assert(FlagEventsAdjacentOk(PF_EV_POS, 2u), "正向对照被误判：相邻两次 set 异号竟然不通过");


// ------------------------------------------------------------
// 4f. M168：两处既有缺口的合并登记（**M164 的 6/7** + **M139 finding 的 B3 六个号**）
// ------------------------------------------------------------
// §4 的 `FLAG_SEQ[]` 是 **decode 四相位**的权威表，§4c/§4d 是 **prefill 各段**的分节。M168 之前
// 有两处「已实现、已取证、却没进任意一张表」的号：
//   ① `M15L_H2ToBHandshake`（M164 的 M1 修法，prefill 相位 B 的 H2→B 跨类型握手）；
//   ② `m15_attn_fa_core.h`（B3 稠密 causal prefill core）自己的通道族 —— M139 finding
//      （`.tower/comms/findings/20261004-agent-ccaudit-improve-b3-m15-attn-fa-core-h-6-mode-4-flagid-flagid.md`）点名的缺口；
//      **M185 起该族由 mode 4 改为 mode 2**（§4f.2）。
// 塔 2026-10-04 的裁决：M163/M164 合入后派一条「登记合并件」把两者一并落进本文件（本节）。

// ---- 4f.1 M164：prefill 相位 B 的 H2→B 握手（**这两个号的唯一权威定义**）----
// 形态 = `docs/05 §2` 逐字给的「**AIVs→AIC（mode 2）→ 全体 AIC barrier（mode 0）**」跨类型
// all-to-all 标准组合（与 §4b 的 M97 前端、§4 的 M124 PLE ③ 同款）。调用点 =
// `m15_layer_kernel.h` 的 `M15L_H2ToBHandshake<M2_FLAG, AIC_BAR_FLAG>()`（定义 `:406-419`，
// 在 `M15L_PrefillPhaseB` 进块循环前调用一次 `:821`）；依赖面 = 全体 AIV 写 `pfHcBlk1`、
// 每个 AIC 的 router 读整块（`m15_moe_prefill.h:1273` 的 `ProcessAic` 开头）。
// **M168 收口**：此前这两个号是 `m15_layer_kernel.h:397-398` 的本头局部常量（M164 因
// `m15_layer_resources.h` 当时归 M163 占用而就地定义，见塔 2026-10-04 的裁决信）；现在本头是
// 唯一权威，`m15_layer_kernel.h` 只引用、不再定义 ⇒ 同一号只有一处定义。
constexpr uint16_t FLAG_H2B_A2C_M2 = 6;      // AIV → 配对 AIC：H2 的 BLK（pfHcBlk1）已写完
constexpr uint16_t FLAG_H2B_AIC_BAR = 7;     // 全体 AIC mode-0 barrier（每个 AIC 已收到配对 AIV 的信号）

constexpr FlagIdRef FLAG_SEQ_PREFILL_H2B[] = {
    {"H2B", "both", 2, FLAG_H2B_A2C_M2,  "both", "H2→B：AIV set（写完 BLK）→ 每个 AIC wait（配对 2 AIV set ↔ 1 AIC wait）"},
    {"H2B", "AIC",  0, FLAG_H2B_AIC_BAR, "both", "H2→B：全体 AIC 到齐（set 挂 MTE2 / wait 挂 PIPE_S，见调用点）"},
};
constexpr uint32_t FLAG_SEQ_PREFILL_H2B_N = sizeof(FLAG_SEQ_PREFILL_H2B) / sizeof(FLAG_SEQ_PREFILL_H2B[0]);
static_assert(FLAG_SEQ_PREFILL_H2B_N == 2u, "H2→B 握手的同步点数变了（应为 1 个 mode2 + 1 个 AIC mode0）");
static_assert(FlagRefsAdjacentOk(FLAG_SEQ_PREFILL_H2B, FLAG_SEQ_PREFILL_H2B_N),
              "H2→B 握手的 flagId 相邻性违规（同一 (核型,mode) 的相邻逻辑点同号）");
// 上限（`FlagIdLimit`：mode0/2 的每核池仍是 16）
static_assert(FLAG_H2B_A2C_M2 < FlagIdLimit("both", 2u), "H2→B 的 mode2 号越出 0..15");
static_assert(FLAG_H2B_AIC_BAR < FlagIdLimit("AIC", 0u), "H2→B 的 AIC mode0 号越出 0..15");
// 相邻性（同 (核型, mode) 的执行序：hc(H2) 段 → 6/7 → 相位 B 段）
//   mode2  ：H2 的最后一条 = `M15H::FLAG_C2A_GATE`(11) → **6** → 相位 B 的首条 = `M15M::FLAG_M2_RING[0]`(0)
//   AIC mode0：H2 的最后一条 = `M15H::FLAG_AC3`(3) → **7** → 相位 B 的首条 = `M15M::FLAG_AIC_SEG_RING[0]`(8)
// （相位 B 的 prefill 段体 `m15_moe_prefill.h:1227/1291/1293` 读的正是这两个 ring；prefill 档的
//   `M15PFR::PFR_FLAG_M2_RING`/`PFR_FLAG_AIC_RING`（`m15_moe_prefill_res.h:324-325`）与 `M15M`
//   同槽同值，本头不反向包含 `M15PFR` ⇒ 断言落在 `M15M` 的号上，值一致。）
static_assert(FLAG_H2B_A2C_M2 != M15H::FLAG_C2A_GATE && FLAG_H2B_A2C_M2 != M15M::FLAG_M2_RING[0],
              "H2→B 的 mode2 号与两侧相邻同步点（hc(H2) 尾 / 相位 B 首）同号");
static_assert(FLAG_H2B_AIC_BAR != M15H::FLAG_AC3 && FLAG_H2B_AIC_BAR != M15M::FLAG_AIC_SEG_RING[0],
              "H2→B 的 AIC mode0 号与两侧相邻同步点（hc(H2) 尾 / 相位 B 首）同号");
// 「两核同时空闲只有 {6,7}」的见证（M164 取证、塔 2026-10-04 认可）：把 prefill(KIND_GDN) 里已
// 登记的 AIV mode2 与 AIC mode0 占用区间写成判据 —— 6/7 必须落在它们之外，且正好用满这两格。
constexpr bool PfH2BFreeOk()
{
    const uint32_t m2 = FLAG_H2B_A2C_M2;
    const uint32_t m0 = FLAG_H2B_AIC_BAR;
    // AIV mode2 已占：{0..5}（B1 的 `GP_FLAG_GO1..DONE3`）∪ {8..13}（hc 的 mode2 8..11 + PLE 的 12/13）
    if (m2 <= M15G::GP_FLAG_DONE3 || (m2 >= M15H::FLAG_A2C_XN && m2 <= M85P::FLAG_B3)) {
        return false;
    }
    // AIC mode0 已占：{0..3}（hc 的 `FLAG_AC0..AC3`）∪ {8..11}（相位 B 的 AIC ring）
    if (m0 <= M15H::FLAG_AC3 || (m0 >= M15M::FLAG_AIC_SEG_RING[0] && m0 <= M15M::FLAG_AIC_SEG_RING[3])) {
        return false;
    }
    // 两核同时空闲 = {6,7}；两个号都必须落在这一对里、且互异
    return (m2 >= 6u && m2 <= 7u) && (m0 >= 6u && m0 <= 7u) && (m2 != m0);
}
static_assert(PfH2BFreeOk(),
              "H2→B 的 6/7 越出「两核同时空闲」的 {6,7}：与某段已占号撞车，或没用满这两格");

// ---- 4f.2 M139 finding + **M185 重写**：B3（`m15_attn_fa_core.h`）的 **mode-2** 通道族 ----
// 权威值 = `m15_attn_fa_core.h` 的 `CC_S_RDY/CC_S_FREE/CC_PV_RDY/CC_PV_FREE/CC_P0/CC_P1`；
// 本头只**引用**、不重定义（M168 起本文件 `#include "m15_attn_fa_core.h"`）。
// **M185 重写**（原为 mode-4 的「AIC 侧对 AIV1 通道 `id + 16`」）：attention 是 **1 AIC ↔ 2 AIV**，
// 改用 **mode 2**（`docs/05` §2 逐字：单个 AIC ↔ 其 2 个 AIV；`1 set(AIC) 配 2 wait(AIV)`，
// 或 `2 set(AIV) 配 1 wait(AIC)` 算配对）。⇒ **每个同步事件只用一个 flagId**（不再有 `+16`），
// AIV 侧 id 与 AIC 侧 id 落在**同一个 mode2 池 0..15**（`FlagIdLimit(核型, 2) == 16`）。
// parity：`q = t & 1` ⇒ 一个调用点覆盖 `base` 与 `base+1`（下表 `cov=2`），**不是固定常数**（按 tile 数走）。
// P 事件语义（M185 起）：**P(t) 已被配对 2 个 AIV 直写进 L1**（不再经 GM），AIC 侧只 `L1 -> L0A`。
// 调用点计数（核体实测）：AIV 9 + AIC 5 = **14**（AIV：Run 预置 4 + 每 tile 循环 5；AIC：每 tile 循环 5
// —— M185 删掉了原 AIC 侧 AIV1 通道的 5 个 `+16` 点）。每通道严格**单向**（一个 set 方 + 一个 wait 方）。
constexpr uint16_t FAC_CC_ID[6] = {
    M15FAC::CC_S_RDY, M15FAC::CC_S_FREE, M15FAC::CC_PV_RDY, M15FAC::CC_PV_FREE, M15FAC::CC_P0, M15FAC::CC_P1,
};
static_assert(M15FAC::CC_P1 == M15FAC::CC_P0 + 1u,
              "CC_P1 必须是 CC_P0 的 parity1 名（同一通道）—— 6 个号里只有 5 个物理通道");
// mode-2 的每核池上界见证（AIV 与 AIC 共用 0..15；**不再有** mode-4 的 AIC 侧 0..31）。
constexpr uint16_t ATTN_FA_M2_MAX = static_cast<uint16_t>(M15FAC::CC_P1 + 1u);   // = 7（最大号 CC_P1 的 parity1）
static_assert(ATTN_FA_M2_MAX < FLAG_PER_CORE,
              "B3 mode-2 的号越出每核 0..15（mode0/2 的池没有 mode4 的 AIC 侧扩展，`docs/05 §6.1`）");
static_assert(FlagIdLimit("AIC", 2u) == FLAG_PER_CORE && FlagIdLimit("AIV", 2u) == FLAG_PER_CORE,
              "mode2 的每核池是 16（B3 的 AIC/AIV 都落在这个池里）");

// 事件级登记（一行 = 一次 set 或一次 wait；`cov`/`count` 见 §4c 的口径说明）。
// ⚠ 采用 §4c 的 `FlagStepM4` 粒度（事件级）而非 §4d 的 `FlagIdRef`（逻辑点级）：mode 2 的两侧是
// `1 set ↔ 2 wait` / `2 set ↔ 1 wait` 的配对语义，逻辑点表会把两侧压成一行而失真。
constexpr FlagStepM4 FLAG_SEQ_ATTN_FA_CORE[] = {
    // ---- AIV（9 个调用点；**行序 = 核体执行序**：`m15_attn_fa_core.h` 的 `FacAiv::Run` 预置（并入
    //   下面两条 set 行的 count）→ 每 tile S_RDY wait → S_FREE set → P0 set → PV_RDY wait → PV_FREE set。
    //   行序与执行序对齐，`FlagEventsAdjacentOk` 才测的是真相邻 set 序。）----
    {"AIV", 2, M15FAC::CC_S_RDY,  2, "wait", "PIPE_V",    1, "S(t) 就绪：AIC→配对 2 AIV（基号 0，parity 0/1）"},
    {"AIV", 2, M15FAC::CC_S_FREE, 2, "set",  "PIPE_V",    3, "S 槽位空闲：Run 预置×2 + 每 tile 消费后×1（配对 2 AIV 都 set 才放行 AIC）"},
    {"AIV", 2, M15FAC::CC_P0,     2, "set",  "PIPE_MTE3", 1, "P(t) 已**直写 L1**：配对 2 AIV→AIC（`CC_P0+q` 覆盖 CC_P0/CC_P1，基号 5）"},
    {"AIV", 2, M15FAC::CC_PV_RDY, 2, "wait", "PIPE_V",    1, "PV(t) 就绪：AIC→配对 2 AIV（基号 2）"},
    {"AIV", 2, M15FAC::CC_PV_FREE, 2, "set", "PIPE_V",    3, "PV 槽位空闲：Run 预置×2 + 每 tile 消费后×1"},
    // ---- AIC（5 个调用点；mode 2 下**一个事件只有一行** —— M185 删掉了原 AIV1 通道的 5 个 `+16` 行）----
    {"AIC", 2, M15FAC::CC_S_FREE, 2, "wait", "PIPE_FIX",  1, "S[q] 槽位空闲（配对 2 AIV 都 set 后才放行，id 1/2）"},
    {"AIC", 2, M15FAC::CC_S_RDY,  2, "set",  "PIPE_FIX",  1, "S[q] 就绪 → 放行配对 2 AIV（id 0/1）"},
    {"AIC", 2, M15FAC::CC_P0,     2, "wait", "PIPE_MTE1", 1, "配对 2 AIV 的 P(t) 已直写 L1（id 5/6）"},
    {"AIC", 2, M15FAC::CC_PV_FREE, 2, "wait", "PIPE_FIX", 1, "配对 2 AIV 已消费上一轮 PV[q]（id 3/4）"},
    {"AIC", 2, M15FAC::CC_PV_RDY, 2, "set",  "PIPE_FIX",  1, "PV[q] 就绪 → 放行配对 2 AIV（id 2/3）"},
};
constexpr uint32_t FLAG_SEQ_ATTN_FA_CORE_N = sizeof(FLAG_SEQ_ATTN_FA_CORE) / sizeof(FLAG_SEQ_ATTN_FA_CORE[0]);

constexpr uint32_t FlagSeqAttnFaCoreCalls()
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_FA_CORE_N; ++i) {
        n += FLAG_SEQ_ATTN_FA_CORE[i].count;
    }
    return n;
}
constexpr uint32_t FlagSeqAttnFaCoreCallsOf(const char* core)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < FLAG_SEQ_ATTN_FA_CORE_N; ++i) {
        n += (CoreId(FLAG_SEQ_ATTN_FA_CORE[i].core) == CoreId(core)) ? FLAG_SEQ_ATTN_FA_CORE[i].count : 0u;
    }
    return n;
}
static_assert(FlagSeqAttnFaCoreCalls() == 14u, "B3 的 cross-core 调用点数 ≠ 14（登记表与核体不一致）");
static_assert(FlagSeqAttnFaCoreCallsOf("AIC") == 5u && FlagSeqAttnFaCoreCallsOf("AIV") == 9u,
              "B3 的 AIC/AIV 调用点数 ≠ 5/9（核体实测：AIV 预置 4 + 循环 5 = 9；AIC 循环 5）");
// 相邻性（相邻两次 set 不得同号）+ 上限（`id + cov <= FlagIdLimit`）
static_assert(FlagEventsAdjacentOk(FLAG_SEQ_ATTN_FA_CORE, FLAG_SEQ_ATTN_FA_CORE_N),
              "B3 的 flagId 相邻性违规：同一 (核型, mode) 的相邻两次 set 同号");
// ---- M197 重推：硬件 4bit 计数器计的是「未配平累计」，不是 set 总次数 ----
// 依据（逐字）：`docs/05-megakernel-design.md` 的 CrossCore flagId 行 = 「每（核,flagId) 一个 4bit
//   计数器（0-15），**未配平累计超 15 报错**」⇒ 约束量 = 执行中**同时未配平**（set 减已在同 id 上 wait
//   掉的）令牌数的**峰值**，与总 set 次数无关。
// B3 核体（`m15_attn_fa_core.h` 的 `FacAiv::Run` / `FacAic::RunWorkItem`）的配对是严格乒乓：
//   · `CC_S_FREE`/`CC_PV_FREE`（AIV→AIC，各 2 个号 = parity）是**槽位空闲令牌**；
//     AIV 在进 tile 循环前给两个槽各预置 1 个（共 4 个 set）⇒ 未配平峰值 = **2**（= 双缓冲槽数）。
//   · `CC_S_RDY`/`CC_PV_RDY`（AIC→AIV）与 `CC_P0/+1`（AIV→AIC）：同 parity 的下一次 RDY 必在该
//     parity 的上一轮被 wait 之后才发 ⇒ 未配平峰值 = **1**。
// ⇒ 全 (核型, mode, id) 的未配平峰值 = **2 ≤ 15**，且**与 tile 数无关**：生产档 nTiles 由
//   `ceil(min(posBase+rowBase+64, ctx)/128)` 给（4097 行 ctx 下 ≈ 33），逐 tile 的 set/wait **成对
//   回到 ≤2**，累计量 O(1)。⇒ **不需要改计数**（计数模型按未配平量本就不该随 nTiles 放大）。
// 保留下面这条 per-调用点代理断言（它不是硬件计数器本身，而是"单次调用里同一 id 的 set 调用点数"的
//   一个**保守代理**；它对 nTiles **不缩放** —— 缩放的量按上面推导不构成约束）。真实约束的推导见
//   `m15_layer_loop/evidence/m197_b2_mount/attn_fa_flag_unbalanced.md`。
static_assert(FlagEventsMaxSetUse(FLAG_SEQ_ATTN_FA_CORE, FLAG_SEQ_ATTN_FA_CORE_N) <= 15u,
              "B3 单次调用内同一 (核型, mode, id) 的 set 调用点数超过 15（保守代理；未配平峰值另见证据）");
// `FAC_CC_ID[6]` 的用法（不留在库里的死代码）：这 6 个号必须都被事件表的 **AIV 侧** (id, cov=2)
// 区间覆盖，且 6 个号互异。覆盖区间 = {0,1}∪{1,2}∪{2,3}∪{3,4}∪{5,6} = {0..6}（比 6 个号多出的
// id 4 是 `CC_PV_FREE` 的 parity1）。
constexpr bool FacCcIdsCoveredOk()
{
    for (uint32_t k = 0; k < 6u; ++k) {
        for (uint32_t m = k + 1u; m < 6u; ++m) {
            if (FAC_CC_ID[k] == FAC_CC_ID[m]) { return false; }
        }
        bool covered = false;
        for (uint32_t i = 0; i < FLAG_SEQ_ATTN_FA_CORE_N; ++i) {
            const FlagStepM4& r = FLAG_SEQ_ATTN_FA_CORE[i];
            if (CoreId(r.core) != 0u || r.mode != 2u) { continue; }   // 只看 AIV 侧登记
            if (FAC_CC_ID[k] >= r.id && static_cast<uint32_t>(FAC_CC_ID[k]) < static_cast<uint32_t>(r.id) + r.cov) {
                covered = true;
                break;
            }
        }
        if (!covered) { return false; }
    }
    return true;
}
static_assert(FacCcIdsCoveredOk(),
              "FAC_CC_ID 里的某个号没被事件表登记（或 6 个号有重复）—— 登记集合与 FAC_CC_ID 不一致");

// 「每个逻辑通道严格 1 个 set 方」的配对口径（`m15_attn_fa_core.h` §2：`*_RDY` 只由 AIC set、
// AIV wait；`*_FREE` 只由 AIV set、AIC wait）。mode 2 下**一个事件一对核**（不再是 mode-4 的
// 「AIV0 基号 + AIV1 的 `+16`」两行），配对语义是 `1 set(AIC) ↔ 2 wait(AIV)` / `2 set(AIV) ↔ 1 wait(AIC)`
// —— 那 2 个 AIV 是配对里的两个半核、共享同一个 flagId（本表一行即代表）。
// 5 个物理通道（`CC_P1` 是 `CC_P0` 的 parity1 名，已在上面钉死）：
struct FacCcChannel {
    const char* setCore;   // 谁 set（生产者核型）
    uint16_t    base;      // 本地基号
    const char* waitCore;  // 谁 wait（消费者核型）
    const char* use;
};
constexpr FacCcChannel FAC_CC_CH[] = {
    {"AIC", M15FAC::CC_S_RDY,   "AIV", "S(t) 就绪（AIC→配对 2 AIV）"},
    {"AIV", M15FAC::CC_S_FREE,  "AIC", "S(t) 槽位空闲（配对 2 AIV→AIC）"},
    {"AIC", M15FAC::CC_PV_RDY,  "AIV", "PV(t) 就绪（AIC→配对 2 AIV）"},
    {"AIV", M15FAC::CC_PV_FREE, "AIC", "PV(t) 槽位空闲（配对 2 AIV→AIC）"},
    {"AIV", M15FAC::CC_P0,      "AIC", "P(t) parity 已直写 L1（配对 2 AIV→AIC，CC_P0/CC_P1 两号）"},
};
constexpr uint32_t FAC_CC_CH_N = sizeof(FAC_CC_CH) / sizeof(FAC_CC_CH[0]);
static_assert(FAC_CC_CH_N == 5u, "B3 的 mode-2 物理通道应是 5 个（6 个号里 CC_P1 是 CC_P0 的 parity1 名）");

// 对每个通道：事件表里恰好 1 条 AIV 行 + 1 条 AIC 行（同一 base），且每行的 set/wait 角色与
// `FAC_CC_CH` 声明的生产者/消费者一致。⇒ 这同时钉住「每通道严格 1 个 set 方 + 1 个 wait 方」。
constexpr bool FacCcPairOkFor(const FacCcChannel* chs, uint32_t n)
{
    for (uint32_t c = 0; c < n; ++c) {
        const uint16_t base = chs[c].base;
        uint32_t aiv = 0, aic = 0;
        for (uint32_t i = 0; i < FLAG_SEQ_ATTN_FA_CORE_N; ++i) {
            const FlagStepM4& r = FLAG_SEQ_ATTN_FA_CORE[i];
            if (r.mode != 2u || r.id != base) { continue; }
            const bool rowIsProd = (CoreId(r.core) == CoreId(chs[c].setCore));
            if (FlagDirIsSet(r.dir) != rowIsProd) { return false; }
            if (CoreId(r.core) == 0u) { ++aiv; }
            else if (CoreId(r.core) == 1u) { ++aic; }
            else { return false; }   // mode2 的登记行只用 "AIV"/"AIC" 两种核型
        }
        if (aiv != 1u || aic != 1u) { return false; }
    }
    return true;
}
static_assert(FacCcPairOkFor(FAC_CC_CH, FAC_CC_CH_N),
              "B3 的 mode-2 通道配对违规：每通道应恰有 1 条 AIV 行 + 1 条 AIC 行，且 set/wait 单向");
// 负向对照（随代码入库：写反 set 方必须编译期变红 ⇒ 判据不是"恒真的装饰"）
constexpr FacCcChannel FAC_CC_NEG[] = {
    {"AIV", M15FAC::CC_S_RDY, "AIC", "人为构造：把 S_RDY 的 set 方写反（负向对照）"},
};
static_assert(!FacCcPairOkFor(FAC_CC_NEG, 1u),
              "负向对照没被抓住：把通道的 set 方写反竟然通过（配对判据失去抓错能力）");


// ============================================================
// 5. BufferID 登记表（每核用户可用 0..27，docs/05 §6.1）
// ============================================================

struct BufReg {
    const char* seg;
    const char* core;
    uint32_t    lo;
    uint32_t    hi;
    bool        sharedAcrossPhase;
    const char* use;
};
constexpr BufReg BUF_TABLE[] = {
    {"GDN", "AIC", 0, 6, true,  "A/B 大包 ping-pong + L0A/L0B + L0C（相位边界后交回下一段）"},
    {"MoE", "AIC", 0, 6, true,  "同上（MoE 的 mmad 复用同编号 token）"},
    {"hc",  "AIC", 0, 6, true,  "同上（hc 的两段 GEMM 复用同编号 token；相位边界全 drain）"},
    {"GDN", "AIV", 0, 4, true,  "m6 行窗 0/1 + m9 prolog 2/3/4"},
    {"MoE", "AIV", 0, 7, true,  "通用行窗 0..7（段内 barrier 分隔、地址叠放）"},
    {"hc",  "AIV", 0, 16, true, "hc 段 17 个核内 token（S1/W0/S2/S4/S6 各行窗；与 GDN/MoE 叠放）"},
    {"GDN", "AIV", 8, 17, false, "m4 递推 8-14 + m12 RMSNormGated 15-17（MoE/hc 段不用同址）"},
    {"MoE", "AIV", 15, 23, true, "m8#1 四级流水 15-18 + 预取/gamma/rstd 19-23"},
    {"GDN", "AIV", 21, 22, true, "gamma1/2 与 gammaG 的 bf16→fp32 预转（MoE 段复用同编号）"},
    // M97（W1）：attention 相位 A 的 AIV BufferID 与 GDN/MoE/hc 的 0..2 **同址叠放**
    // （相位互斥 + 相位交界 PipeBarrier<PIPE_ALL>，与 §2 的段窗叠放同一条依据）。
    // 逐 id 依据：`m15_attn_prolog_probe.h` 的 AP_BUF_IO=0 / AP_BUF_IN=1 / AP_BUF_OUT=2。
    // AIC 侧 attention 复用 `M15G::BUF_AIC_*`（0..6），与上表 GDN "AIC 0..6" 那一行同址。
    {"ATN", "AIV", 0, 2, true, "M97 attention 相位 A：抄写中转 0 + 输入面 1 + 输出头 2（与 GDN/MoE/hc 叠放）"},
    // ---- M110：prefill 的 BufferID 分节（**只有本 mission 用到的那些**；段体自带的编号归 Wave C 打平）----
    // 分工（照 M103-2.3 的 Wave C 清单）：本节登记的是 **Wave A 自己那几个挂载点**的 id
    // （相位 A 的行分派 + pos/位置表的 staging），段体（B1..B5）必须按 §5.3 的共同契约把自己的
    // BufferID 清单交出来，由 Wave C 一次性打平；本节只把"它们必须落在同一批 0..27 里"这件事钉住。
    {"GDN-PF", "AIV",  0, 2, true, "prefill 相位 A 的行分派：行 staging ping/pong 0/1 + pos 表 2"},
    {"GDN-PF", "AIC",  0, 6, true, "prefill 的 mmad 窗（A 0/1、B 2/3、L0A/L0B 4/5、L0C 6）——与 decode 同址叠放"},
    {"ATN-PF", "AIV",  0, 2, true, "attention prefill 相位 A 的挂载点窗口（与 GDN-PF 同址）"},
    {"ATN-PF", "AIC",  0, 6, true, "同上（B3 的 L1 K/V 窗另算，归它自己的 _res 头 + Wave C 打平）"},
};

// 峰值同时占用数（每核）：AIC = max(GDN 7, MoE 7, hc 7) = 7 ≤ 28；
// AIV = max(GDN 17, MoE 17, hc 17) = 17 ≤ 28
constexpr uint32_t BUF_AIC_PEAK = 7;
constexpr uint32_t BUF_AIV_PEAK = 17;
constexpr uint32_t BUF_PER_CORE_MAX = 28;
// M110：prefill 档自己的峰值（Wave A 登记的挂载点窗口；段体的占用由 Wave C 并进来）
constexpr uint32_t PF_BUF_AIV_ROW0 = 0;
constexpr uint32_t PF_BUF_AIV_ROW1 = 1;
constexpr uint32_t PF_BUF_AIV_POS = 2;
constexpr uint32_t PF_BUF_AIV_LO = 0;
constexpr uint32_t PF_BUF_AIV_HI = 2;
constexpr uint32_t BUF_AIV_PEAK_PF = 3;
constexpr uint32_t PF_BUF_AIC_LO = 0;
constexpr uint32_t PF_BUF_AIC_HI = 6;
constexpr uint32_t BUF_AIC_PEAK_PF = 7;
// 相邻性/容量（本仓惯例：每条 id 都要有断言）
static_assert(PF_BUF_AIV_ROW0 < PF_BUF_AIV_ROW1 && PF_BUF_AIV_ROW1 < PF_BUF_AIV_POS,
              "prefill 的行 staging 三个 id 必须互异且顺排（同一 BufferID 上的两笔搬运会互相踩）");
static_assert(PF_BUF_AIV_HI - PF_BUF_AIV_LO + 1u == BUF_AIV_PEAK_PF, "prefill AIV 窗的宽与峰值不一致");
static_assert(PF_BUF_AIC_HI - PF_BUF_AIC_LO + 1u == BUF_AIC_PEAK_PF, "prefill AIC 窗的宽与峰值不一致");
static_assert((PF_BUF_AIV_HI - PF_BUF_AIV_LO + 1u) <= BUF_PER_CORE_MAX, "prefill AIV 窗超出每核 28 个");
static_assert((PF_BUF_AIC_HI - PF_BUF_AIC_LO + 1u) <= BUF_PER_CORE_MAX, "prefill AIC 窗超出每核 28 个");
static_assert(M15H::BUF_AIV_IDS <= BUF_AIV_PEAK, "hc 段 AIV BufferID 用量超过了登记峰值");
static_assert(M15H::BUF_AIC_IDS <= BUF_AIC_PEAK, "hc 段 AIC BufferID 用量超过了登记峰值");
static_assert(BUF_AIC_PEAK <= BUF_PER_CORE_MAX, "AIC BufferID 超出每核 28 个");
static_assert(BUF_AIV_PEAK <= BUF_PER_CORE_MAX, "AIV BufferID 超出每核 28 个");

// ============================================================
// 6. 入口符号登记（docs/15 §5.2 硬约束 1：mode 由 host 选符号）
// ============================================================
constexpr const char* ENTRY_GDN = "m15_layer_kernel_gdn";              // 已实现（decode，两相位）
constexpr const char* ENTRY_ATTN = "m15_layer_kernel_attn";            // 已实现（decode，两相位）
constexpr const char* ENTRY_GDN_HC = "m15_layer_kernel_gdn_hc";        // M58：四相位（hc 融合）
constexpr const char* ENTRY_ATTN_HC = "m15_layer_kernel_attn_hc";      // M58：四相位（hc 融合）
constexpr const char* ENTRY_MOE_ONLY = "m15_moe_segment_kernel";       // 仅验证用（段间零串扰对照）
constexpr const char* ENTRY_HC_ONLY = "m15_hc_segment_kernel";         // 仅验证用（hc 单边界对照）
constexpr const char* ENTRY_FINAL_MIX = "m15_final_mixer_kernel";      // M65：末层之后的全局 mixer
constexpr const char* ENTRY_GDN_PREFILL = "m15_layer_kernel_gdn_prefill";    // M110：真入口（段体未接，默认关）
constexpr const char* ENTRY_ATTN_PREFILL = "m15_layer_kernel_attn_prefill";  // M110：真入口（段体未接，默认关）

// ------------------------------------------------------------
// 6b. M110：**入口登记表**（从"符号常量"变成"真入口登记"）
// ------------------------------------------------------------
// 判别方式（docs/15 §5 方案 B）：**mode 由 host 选符号**，kernel 内无 m 分叉。本表是 host 侧
// 「哪个符号存在、它是哪一档、接线开关关着还是开着」的**唯一权威** —— M110 之前两个 prefill
// 名字只是注释 + 字符串常量（连 `__global__` 都没有），"预留"这件事无法被机械核对。
//
// `wired` 的语义（**默认 0**，与 `.asc` 的 `M15_PREFILL_WIRE` 缺省一致）：
//   0 = 结构占位（相位 A 逐行分派 + 写见证面；**数学未实现**，显式标注）
//   1 = 段体挂载点（要求对应段体已落地；段体缺席时**响亮失败**，不允许静默通过）
struct EntryReg {
    const char* sym;
    const char* kind;      // "gdn" / "attn" / "moe" / "hc" / "gmix"
    uint32_t    nPhase;    // 相位数
    uint32_t    hc;        // 1 = 四相位（hc 融合）形态
    uint32_t    prefill;   // 1 = prefill 档
    uint32_t    wired;     // 段体接线开关（0 = 结构占位）
    const char* note;
};
constexpr EntryReg ENTRY_TABLE[] = {
    {"m15_layer_kernel_gdn",           "gdn",  2, 0, 0, 1, "M40 两相位（decode，已验收）"},
    {"m15_layer_kernel_attn",          "attn", 2, 0, 0, 1, "M40 两相位（decode，相位 A 为占位直通）"},
    {"m15_layer_kernel_gdn_hc",        "gdn",  4, 1, 0, 1, "M58 四相位（decode，交付形态）"},
    {"m15_layer_kernel_attn_hc",       "attn", 4, 1, 0, 1, "M58 四相位（decode，交付形态）"},
    {"m15_moe_segment_kernel",         "moe",  1, 0, 0, 1, "仅验证用（段间零串扰对照）"},
    {"m15_hc_segment_kernel",          "hc",   1, 0, 0, 1, "仅验证用（hc 单边界对照）"},
    {"m15_final_mixer_kernel",         "gmix", 1, 0, 0, 1, "M65 末层全局 mixer"},
    {ENTRY_GDN_PREFILL,                "gdn",  1, 0, 1, 0, "M110 prefill；M132 已接 B1/B4/B5（wired 仍 0）"},
    {ENTRY_ATTN_PREFILL,               "attn", 1, 0, 1, 0, "M110 prefill；attention（B2/B3）待 B3 合入后再接（wired 仍 0）"},
};
constexpr uint32_t ENTRY_TABLE_N = sizeof(ENTRY_TABLE) / sizeof(ENTRY_TABLE[0]);
static_assert(ENTRY_TABLE_N == 9u, "入口登记表的条数变了（每个 __global__ 入口都必须在这里有一行）");
// 每个符号名在表里只能出现一次（同一符号登记两次 = host 选符号时有歧义）
constexpr bool EntrySymsUnique()
{
    for (uint32_t i = 0; i < ENTRY_TABLE_N; ++i) {
        for (uint32_t j = i + 1; j < ENTRY_TABLE_N; ++j) {
            const char* a = ENTRY_TABLE[i].sym;
            const char* b = ENTRY_TABLE[j].sym;
            bool same = true;
            for (uint32_t k = 0; same; ++k) {
                same = (a[k] == b[k]);
                if (a[k] == '\0') {
                    break;
                }
            }
            if (same) {
                return false;
            }
        }
    }
    return true;
}
static_assert(EntrySymsUnique(), "入口登记表里有重复的符号名");
// prefill 档必须**默认关**（这是零回归的硬要求：`runs=all` 的 decode 路径不因本表变化）
constexpr bool EntryPrefillOffOk()
{
    for (uint32_t i = 0; i < ENTRY_TABLE_N; ++i) {
        if (ENTRY_TABLE[i].prefill != 0u && ENTRY_TABLE[i].wired != 0u) {
            return false;
        }
    }
    return true;
}
static_assert(EntryPrefillOffOk(), "prefill 入口的 wired 必须缺省为 0（M103-2.7 第 3 条：默认关闭是硬要求）");
static_assert(PREFILL_WIRED == 0u, "PREFILL_WIRED 缺省必须是 0（同 .asc 的 M15_PREFILL_WIRE）");

}  // namespace M15L

#endif  // M15_LAYER_RESOURCES_H
