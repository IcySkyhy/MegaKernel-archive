/**
 * m15_moe_prefill_res.h —— MoE **prefill** 段（B4）自用静态资源表
 *
 * 范围纪律（塔裁 2026-09-27，`docs/15` §M103-2.2 的 B4 行）：
 *   · 本文件**只放 prefill 段自用**的常量（行分块、UB/L1/L0 窗、GM 工作区、BufferID、flagId）。
 *   · `MW_*` / `MOE_W_STRIDE` 的**唯一 owner 是 `m15_layer_resources.h`** ⇒ 本文件**只消费**
 *     （`MWP_*` / `MOE_W_PREFILL_STRIDE` / `MOE_E_PREFILL` / `MOE_TOPK_PREFILL`），
 *     **不得**再定义第二份 `MW_*` / `MOE_W_STRIDE`（原文作废，见 §1a 的塔裁注）。
 *   · `NUM_EXPERTS` / `TOPK_MAX`（`m15_moe_resources.h`，decode 档）**不在本 mission 的 scope**
 *     ⇒ 本文件自带 prefill 档的 `E` / `TOPK`，并**不引用** `M15M::NUM_EXPERTS` / `M15M::TOPK_MAX`
 *     作为形状来源（只在"两档必须不同"的见证断言里读它们一次）。
 *
 * 形状（真实档，`docs/15` §1 的 checkpoint 字段）：
 *   `HIDDEN = 2560`、`INTER = 640`、`E = 512`、`TOPK = 10`、`M_PREFILL = 4097`（验收档，不是接口假设）。
 *
 * 与 decode 档（`m15_moe_resources.h`）的关系：**不是替换，是另一套定尺**。
 *   decode 档把 `M_MAX = 64 / NUM_EXPERTS = 4 / TOPK_MAX = 4` 烧进 UB 与 GM 的尺寸；
 *   prefill 档把「一次 m 行」改成「**m 按 `MT = 64` 行分块**，块内 Σt_e ≤ `MT × TOPK` = 640」，
 *   于是每一块都能用**同一套**常量：UB 窗按 64 行、GM 逐专家张量按 640 行（= active_num 上界，
 *   而不是 padded 的 `E × MT` = 32768 行 —— 后者是 51× 的浪费，且与紧凑槽寻址不自洽）。
 *   `docs/15` §M103-2.2 的 B4 交付义务③「缓冲区按 active_num 而不是 M_MAX*TOPK_MAX 定尺」即此。
 */

#ifndef M15_MOE_PREFILL_RES_H
#define M15_MOE_PREFILL_RES_H

#include <cstdint>

// 包含次序：调用方（`m15_moe_prefill.h` / 融合 TU）必须已引入 `m15_gdn_resources.h`（M15G）
// 与 `m15_attn_kv.h`，再引入本文件 —— 见 `m15_moe_prefill.h` 的 include 前缀与
// `m15_attn_kv.h` 末尾的次序注释（`m15_layer_resources.h` → `m15_attn_prolog.h` → `m15_attn_kv.h`
// → `m15_attn_cache.h` → `m15_attn_prolog_probe.h`，probe 需要 M15AP 已声明）。
#include "m15_layer_resources.h"   // MWP_* / MOE_W_PREFILL_STRIDE / MOE_E_PREFILL / MOE_TOPK_PREFILL（唯一 owner）
#include "m15_moe_resources.h"     // M15M::HIDDEN/INTER/GU_N/GROUP/... + L1/L0/BufferID 静态表（只读）

namespace M15PFR {

using namespace AscendC;

// ============================================================
// 1. 形状常量（**消费** Wave A 的档位常量；本文件不定义第二份）
// ============================================================

constexpr uint32_t E = M15L::MOE_E_PREFILL;            // 512（真实档专家数）
constexpr uint32_t TOPK = M15L::MOE_TOPK_PREFILL;      // 10（真实档 top-k）

constexpr uint32_t HIDDEN = M15M::HIDDEN;              // 2560
constexpr uint32_t INTER = M15M::INTER;                // 640
constexpr uint32_t GU_N = M15M::GU_N;                  // 1280
constexpr uint32_t GROUP = M15M::GROUP;                // 32（MXFP4 K 方向 group）
constexpr uint32_t GU_SCALE_STRIDE = M15M::GU_SCALE_STRIDE;   // 80 B/行
constexpr uint32_t DN_SCALE_STRIDE = M15M::DN_SCALE_STRIDE;   // 32 B/行

constexpr uint32_t MT = M15M::BASE_M;                  // 行分块 = 64 行/tile（= cube 的 M tile）
constexpr uint32_t M_PREFILL = 4097;                   // 验收档 m 上界（docs/15 §8.3 第 3 条：不是接口假设）
// Σt_e 上界：每个 token 恰好贡献 TOPK 个槽、top-k id 互异 ⇒ Σt_e = m_block × TOPK ≤ MT × TOPK。
constexpr uint32_t TP_MAX = MT * TOPK;                 // 640（**active_num 定尺**的根据）

// ---- 1a. 与 Wave A 的消费见证（塔裁：MW_* 只读不定义）----
// 这些断言证明本文件读到的是 **Wave A 的** E=512 槽表，而不是自己又算了一份。
static_assert(E == 512u && TOPK == 10u, "真实档必须是 E=512 / topk=10（docs/15 §1）");
static_assert(M15L::MWP_WGU_BYTES == E * GU_N * (HIDDEN / 2u),
              "gate_up 权重槽的字节数与 E=512 不自洽（应等于 Wave A 的 MWP_WGU_BYTES）");
static_assert(M15L::MWP_ROUTER_BYTES == E * HIDDEN * 2u, "router 权重槽字节数与 E=512 不自洽");
static_assert(M15L::MOE_W_PREFILL_STRIDE > M15L::MOE_W_STRIDE,
              "prefill 槽（E=512）必须大于 decode 槽（E=4）—— 若相等说明读错了档");
static_assert(M15L::MOE_W_PREFILL_STRIDE > 0u, "MOE_W_PREFILL_STRIDE 未定义");
// 两档必须**不同**：这条同时是"本文件没有把 decode 档当 prefill 档用"的见证。
static_assert(E != M15M::NUM_EXPERTS, "prefill 的 E 与 decode 档的 NUM_EXPERTS 撞了（读错档）");
static_assert(TOPK > M15M::TOPK_MAX, "prefill 的 topk 必须大于 decode 档的 TOPK_MAX（真实档 10 > 4）");

// ============================================================
// 2. router（S2/P2）：`x[MT, HIDDEN] bf16 @ W[E_PAD, HIDDEN]^T bf16 → logits[MT, E_PAD] fp32`
//    **矩阵乘法一律 Mmad**（人类裁决：「凡涉及矩阵乘法必须 mmad，不管 M=1 或者是多大」；
//    `docs/20` §4 的 WO-A2 把 AIV 逐列 GEMV 的 router 打分列为 **P1 违规**）。
//    W 按模型原始 layout [N, K] 行主序存放（与 `m15_gdn_layer.h` 的 `Cube::Bf16Gemm` 同款），
//    故 N 方向 = 专家行；共享门作为**第 E 行**（m13 donor 原本就是这个形态）。
// ============================================================

constexpr uint32_t E_PAD = 640;                        // N 方向 padded：512 专家 + 1 共享门 + pad 到 5×128
constexpr uint32_t RB_K = 64;                          // K 方向大包（bf16 → 128 B/行）
constexpr uint32_t RB_N = 128;                         // N 方向 tile
constexpr uint32_t RB_KBLK = HIDDEN / RB_K;            // 40
constexpr uint32_t RB_NBLK = E_PAD / RB_N;             // 5
constexpr uint32_t RB_MBLK = (MT + M15M::BASE_M - 1) / M15M::BASE_M;   // 1（MT = BASE_M）
static_assert(E_PAD % RB_N == 0u, "router 的 N 方向必须被 RB_N 整除（无尾块）");
static_assert(E_PAD >= E + 1u, "router 的 N 必须覆盖 512 个专家 + 1 行共享门");
static_assert(HIDDEN % RB_K == 0u, "router 的 K 方向必须被 RB_K 整除（无尾块）");
static_assert(M15M::BASE_K % RB_K == 0u || RB_K % M15M::BASE_K == 0u, "router 的 K tile 与 cube 分形不自洽");
static_assert(RB_MBLK == 1u, "MT 必须 <= BASE_M：router 的 M tile 只有一块");

// ---- 2a. L1 静态窗（字节偏移；phase 分离 ⇒ 与 MXFP4 GEMM 的 A/B 区同址叠放，见 §5）----
constexpr uint32_t L1R_A0 = 0;                                     // bf16[MT, RB_K] = 64×64×2 = 8192 B
constexpr uint32_t L1R_A1 = L1R_A0 + MT * RB_K * 2u;               // 8192
constexpr uint32_t L1R_B0 = 128u * 1024u;                          // bf16[RB_K, RB_N] = 64×128×2 = 16384 B
constexpr uint32_t L1R_B1 = L1R_B0 + RB_K * RB_N * 2u;             // 147456
constexpr uint32_t L1R_END = L1R_B1 + RB_K * RB_N * 2u;            // 163840
static_assert(L1R_END <= M15M::L1_BYTES_TOTAL, "router 的 L1 窗超出 512 KB");
// B 窗落在 [128K, 256K) 的**未用区**；A 窗 @0 与 MXFP4 GEMM 的 A 区（0..~8.5KB）重叠，
// 二者由 router→GEMM 的 phase barrier 分隔（同址叠放的先例见 m15_moe_resources.h 文件头）。

// ---- 2b. L0/L0C 静态窗 ----
constexpr uint32_t L0R_OFF0 = M15M::L0_OFF_0;                      // A2 bf16[64,64]=8KB / B2 bf16[64,128]=16KB
constexpr uint32_t L0R_OFF1 = M15M::L0_OFF_1;
constexpr uint32_t L0R_L0C = 0;                                    // L0C fp32[MT, RB_N] = 64×128×4 = 32768 B
static_assert(MT * RB_K * 2u <= M15M::L0_PP_BYTES, "router A2 tile 超出 L0A 半区");
static_assert(RB_K * RB_N * 2u <= M15M::L0_PP_BYTES, "router B2 tile 超出 L0B 半区");
static_assert(MT * RB_N * 4u <= 256u * 1024u, "router L0C tile 超出 256 KB");

// ---- 2c. AIV 侧的 top-k 行宽（沿用 decode 档的 16 块 Sort32 + 归并树形状）----
// E = 512 恰好等于 RT_ROWL（16×32）⇒ **无 pad lane**，候选集 = 全部 512 个专家。
constexpr uint32_t RT_ROWL = M15M::RT_ROWL;                        // 512 lane
constexpr uint32_t RT_WROW = M15M::RT_WROW;                        // 64（top-k 结果 staging 行距）
constexpr uint32_t RT_RB = M15M::RT_RB;                            // 8 行/块
static_assert(RT_ROWL >= E, "logits 行宽 < E：top-k 候选集会被截断");
static_assert(RT_WROW >= TOPK, "top-k 结果放不进 IDS/WS 的 staging 行距");

// ============================================================
// 3. UB 静态布局（prefill MoE 相位独占；复用 decode 档 MoE 段已登记的窗）
// ============================================================
// 复用而非新开窗的理由：prefill MoE 相位与 decode MoE 相位在**同一 kernel 的不同入口**
// （`m15_layer_kernel_*_prefill` vs `m15_layer_kernel_*`），二者不共存 ⇒ 同址复用不冲突。
// 峰值断言仍按 **prefill 自己的读数** 给（见 §3b），供 Wave C 做全局分节。
constexpr uint32_t UB_BYTES_TOTAL = M15M::UB_BYTES_TOTAL;   // 248 KB
constexpr uint32_t UB_PERSIST = M15M::UB_PERSIST;
constexpr uint32_t UB_M6_X1 = M15M::UB_M6_X1;               // S1/S10 的 m6 行 scratch
constexpr uint32_t UB_M6_RES = M15M::UB_M6_RES;
constexpr uint32_t UB_M6_XF = M15M::UB_M6_XF;
constexpr uint32_t UB_M6_Y = M15M::UB_M6_Y;
constexpr uint32_t UB_M6_TMP = M15M::UB_M6_TMP;
constexpr uint32_t UB_M6_RED = M15M::UB_M6_RED;
constexpr uint32_t UB_M6_RSTD = M15M::UB_M6_RSTD;
constexpr uint32_t UB_VEC = M15M::UB_VEC;

// ---- S2'（AIV）：logits 行 staging（mmad 的产物从 GM 读进来）----
// **单行窗**（不是 decode 档的 RT_RB=8 行块）：prefill 的行由 AIV 逐行分派
// （`r = bid; r < M; r += nAiv`），行块粒度在这里没有收益，反而让负载不均。
constexpr uint32_t UB_PF_LOG = UB_VEC + 0;                                  // fp32[1][RT_ROWL] = 2 KB
constexpr uint32_t UB_PF_IDS = UB_PF_LOG + RT_ROWL * 4u;                    // i32[RT_WROW] = 256 B
constexpr uint32_t UB_PF_WS = UB_PF_IDS + RT_WROW * 4u;                     // fp32[RT_WROW] = 256 B
constexpr uint32_t UB_PF_SG = UB_PF_WS + RT_WROW * 4u;                      // fp32[1] 共享门裸点积
// **每个窗必须 32B 对齐**：`Sort32`/`StoreAlign`/`DataCopyPad` 的 UB 地址硬要求（M9 quirk）。
// 这里踩过：`UB_PF_VAL = UB_PF_SG + 4` 让后续所有窗都偏 4 B ⇒ AIV 逐次 Trap（表现为"什么都没写"）。
constexpr uint32_t UB_PF_VAL = UB_PF_SG + 32u;                              // fp32[RT_ROWL]
constexpr uint32_t UB_PF_IDX = UB_PF_VAL + RT_ROWL * 4u;                    // i32[RT_ROWL] 索引模板
constexpr uint32_t UB_PF_PAIR = UB_PF_IDX + RT_ROWL * 4u;                   // fp32[2*RT_ROWL]
constexpr uint32_t UB_PF_MA = UB_PF_PAIR + 2u * RT_ROWL * 4u;               // 归并树乒乓 A
constexpr uint32_t UB_PF_MB = UB_PF_MA + 2u * RT_ROWL * 4u;                 // 归并树乒乓 B
constexpr uint32_t UB_PF_OV = UB_PF_MB + 2u * RT_ROWL * 4u;                 // fp32[64]
constexpr uint32_t UB_PF_OI = UB_PF_OV + 256u;                              // u32[64]
constexpr uint32_t UB_PF_END = UB_PF_OI + 256u;

// ---- S3'（AIV）：计数排序的 UB 静态槽（按 E=512 定尺；decode 档那份按 E=4 ⇒ 装不下）----
constexpr uint32_t UB_IG_SCAL = UB_PF_END;                                          // i32[E] counts
constexpr uint32_t UB_IG_OFF = UB_IG_SCAL + ((E * 4u + 31u) / 32u) * 32u;           // i32[E+1] offsets
constexpr uint32_t UB_IG_CUR = UB_IG_OFF + (((E + 1u) * 4u + 31u) / 32u) * 32u;     // i32[E] cursor
constexpr uint32_t UB_IG_SRC = UB_IG_CUR + ((E * 4u + 31u) / 32u) * 32u;            // i32[TP_MAX] perm_src
constexpr uint32_t UB_IG_EXP = UB_IG_SRC + TP_MAX * 4u;                             // i32[TP_MAX] perm_expert
constexpr uint32_t UB_IG_INV = UB_IG_EXP + TP_MAX * 4u;                             // i32[MT*TOPK] inv_slot
// w_tk 槽**已取消**：decode 档的 w_tk 是把 bf16 权重位打包进 int32（scalar 数据计算，违反新规）；
// 权重的 bf16 取整改在 `UnpermuteP` 的 VF 里做，直接用 `SZ_W` 的 fp32 平面。
constexpr uint32_t UB_IG_END = UB_IG_INV + MT * TOPK * 4u;

// ---- S4' permute（四级流水）----
constexpr uint32_t UB_PM_ROW = UB_VEC + 0;                                          // 4 × bf16[HIDDEN]

// ---- S5'/S7' 量化（MXFP4，全 VEC；与 decode 档同一组偏移）----
constexpr uint32_t UB_QT_GATE = UB_VEC + 0;
constexpr uint32_t UB_QT_UP = UB_QT_GATE + 512u;
constexpr uint32_t UB_QT_SWIGLU = UB_QT_GATE + 1536u;
constexpr uint32_t UB_QT_QX = UB_QT_SWIGLU + 512u;
constexpr uint32_t UB_QT_SCALE = UB_QT_QX + 128u;
constexpr uint32_t UB_QT_HALF = UB_QT_SCALE + 32u;

// ---- S9' unpermute（双缓冲 stage）----
constexpr uint32_t U_STAGE_B = (TOPK + 1u) * HIDDEN * 2u + 64u;
constexpr uint32_t UB_UP_STAGE0 = UB_VEC + 0;
constexpr uint32_t UB_UP_STAGE1 = UB_UP_STAGE0 + U_STAGE_B;

// ---- S9b' combine ----
constexpr uint32_t UB_CB_ROUTED = UB_VEC + 0;
constexpr uint32_t UB_CB_SHARED = UB_CB_ROUTED + HIDDEN * 2u;
constexpr uint32_t UB_CB_OUT = UB_CB_SHARED + HIDDEN * 2u;
constexpr uint32_t UB_CB_G = UB_CB_OUT + HIDDEN * 2u;

// ============================================================
// 3b. UB 峰值断言（**prefill 自己的读数**，供 Wave C 填 PF_UB_BYTES_MOE）
// ============================================================
constexpr uint32_t PfUbPeak()
{
    uint32_t p = UB_PF_END;
    p = (UB_M6_RSTD + 256u > p) ? (UB_M6_RSTD + 256u) : p;
    p = (UB_IG_END > p) ? UB_IG_END : p;
    p = (UB_PM_ROW + 4u * HIDDEN * 2u > p) ? (UB_PM_ROW + 4u * HIDDEN * 2u) : p;
    p = (UB_QT_HALF + 16u > p) ? (UB_QT_HALF + 16u) : p;
    p = (UB_UP_STAGE1 + U_STAGE_B > p) ? (UB_UP_STAGE1 + U_STAGE_B) : p;
    p = (UB_CB_G + 256u > p) ? (UB_CB_G + 256u) : p;
    return p;
}
constexpr uint32_t PF_UB_PEAK = PfUbPeak();
static_assert(PF_UB_PEAK <= UB_BYTES_TOTAL, "prefill MoE 段的 UB 峰值超出 248 KB");
// **每个 UB 窗 32B 对齐**（`Sort32`/`StoreAlign`/`DataCopyPad` 的硬要求，M9/M57）——
// 这条断言是"逐窗对齐"的机器化见证，不是人工核对。
constexpr bool PfUbAlignedOk()
{
    const uint32_t offs[] = {UB_PF_LOG, UB_PF_IDS, UB_PF_WS, UB_PF_SG, UB_PF_VAL, UB_PF_IDX, UB_PF_PAIR,
                             UB_PF_MA, UB_PF_MB, UB_PF_OV, UB_PF_OI, UB_IG_SCAL, UB_IG_OFF, UB_IG_CUR,
                             UB_IG_SRC, UB_IG_EXP, UB_IG_INV, UB_PM_ROW, UB_QT_GATE, UB_QT_UP,
                             UB_QT_SWIGLU, UB_QT_QX, UB_QT_SCALE, UB_QT_HALF, UB_UP_STAGE0, UB_UP_STAGE1,
                             UB_CB_ROUTED, UB_CB_SHARED, UB_CB_OUT, UB_CB_G, UB_M6_X1, UB_M6_RES, UB_M6_XF,
                             UB_M6_Y};
    for (uint32_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) {
        if ((offs[i] & 31u) != 0u) {
            return false;
        }
    }
    return true;
}
static_assert(PfUbAlignedOk(), "prefill MoE 段的某个 UB 窗不是 32B 对齐（AIV 会 Trap）");
static_assert(UB_PF_END <= UB_IG_SCAL, "S2' 的窗与 S3' 的 UB 静态槽重叠（布局串了）");
static_assert(UB_IG_OFF == (UB_IG_SCAL + ((E * 4u + 31u) / 32u) * 32u), "counts/offsets 槽重叠");
static_assert(UB_IG_END <= UB_BYTES_TOTAL, "S3' 的 UB 静态槽超出 248 KB");
// 与 decode 档 MoE 段（UB_RT_END = 222752）的关系：复用同一片窗 ⇒ 峰值不得超过它太多；
// 这条是**读数断言**不是预算断言，超了要重排窗（Wave C 的收口点）。
static_assert(PF_UB_PEAK <= M15M::UB_RT_END + 8192u,
              "prefill MoE 的 UB 峰值显著超过 decode 档 MoE 段（需重排窗，别默默叠上去）");

// ============================================================
// 4. GM 工作区（一个连续 ws 缓冲内；单位字节、32B 对齐）
// ============================================================
constexpr uint32_t PWS_ALIGN = 32;
constexpr uint32_t AlignUp32(uint32_t x) { return (x + PWS_ALIGN - 1u) / PWS_ALIGN * PWS_ALIGN; }

// 行数与 topk 相关的张量一律按 **MT 行 / TP_MAX 行** 定尺（active_num 上界），
// 逐专家张量**不按 E × MT padding 定尺**（那会是 51× 浪费，且与紧凑槽寻址不自洽）。
constexpr uint32_t SZ_XNORM = AlignUp32(MT * HIDDEN * 2u);
constexpr uint32_t SZ_RES1 = AlignUp32(MT * HIDDEN * 4u);
constexpr uint32_t SZ_LOGITS = AlignUp32(MT * E_PAD * 4u);          // router 的 mmad 产物（fp32）
constexpr uint32_t SZ_IDS = AlignUp32(MT * TOPK * 4u);
constexpr uint32_t SZ_W = AlignUp32(MT * TOPK * 4u);
constexpr uint32_t SZ_SGATE = AlignUp32(MT * 4u);              // 共享门**裸点积**（sigmoid 在 combine 里做）
constexpr uint32_t SZ_COUNTS = AlignUp32(E * 4u);
constexpr uint32_t SZ_OFFSETS = AlignUp32((E + 1u) * 4u + 4u) + 32u;   // +32B 诊断槽
constexpr uint32_t SZ_PERM = AlignUp32(TP_MAX * 4u);
constexpr uint32_t SZ_INV = AlignUp32(MT * TOPK * 4u);
constexpr uint32_t SZ_XSORT = AlignUp32(TP_MAX * HIDDEN * 2u);
constexpr uint32_t SZ_AQ = AlignUp32(TP_MAX * (HIDDEN / 2u));
constexpr uint32_t SZ_AS = AlignUp32(TP_MAX * GU_SCALE_STRIDE);
constexpr uint32_t SZ_AQ_SHD = AlignUp32(MT * (HIDDEN / 2u));
constexpr uint32_t SZ_AS_SHD = AlignUp32(MT * GU_SCALE_STRIDE);
constexpr uint32_t SZ_GU = AlignUp32(TP_MAX * GU_N * 2u);
constexpr uint32_t SZ_GU_SHD = AlignUp32(MT * GU_N * 2u);
constexpr uint32_t SZ_H = AlignUp32(TP_MAX * INTER * 2u);
constexpr uint32_t SZ_H_SHD = AlignUp32(MT * INTER * 2u);
constexpr uint32_t SZ_HQ = AlignUp32(TP_MAX * (INTER / 2u));
constexpr uint32_t SZ_HS = AlignUp32(TP_MAX * DN_SCALE_STRIDE);
constexpr uint32_t SZ_HQ_SHD = AlignUp32(MT * (INTER / 2u));
constexpr uint32_t SZ_HS_SHD = AlignUp32(MT * DN_SCALE_STRIDE);
constexpr uint32_t SZ_Y = AlignUp32(TP_MAX * HIDDEN * 2u);
constexpr uint32_t SZ_Y_SHD = AlignUp32(MT * HIDDEN * 2u);
constexpr uint32_t SZ_ROUTED = AlignUp32(MT * HIDDEN * 2u);
constexpr uint32_t SZ_SHARED = AlignUp32(MT * HIDDEN * 2u);
constexpr uint32_t SZ_MOE = AlignUp32(MT * HIDDEN * 2u);
constexpr uint32_t SZ_RES2 = AlignUp32(MT * HIDDEN * 4u);       // S10 的 fp32 残差出口（m6#2）

constexpr uint32_t PWS_XNORM = 0;
constexpr uint32_t PWS_RES1 = PWS_XNORM + SZ_XNORM;
constexpr uint32_t PWS_LOGITS = PWS_RES1 + SZ_RES1;
constexpr uint32_t PWS_IDS = PWS_LOGITS + SZ_LOGITS;
constexpr uint32_t PWS_W = PWS_IDS + SZ_IDS;
constexpr uint32_t PWS_SGATE = PWS_W + SZ_W;
constexpr uint32_t PWS_COUNTS = PWS_SGATE + SZ_SGATE;
constexpr uint32_t PWS_OFFSETS = PWS_COUNTS + SZ_COUNTS;
constexpr uint32_t PWS_PERM_SRC = PWS_OFFSETS + SZ_OFFSETS;
constexpr uint32_t PWS_PERM_EXP = PWS_PERM_SRC + SZ_PERM;
constexpr uint32_t PWS_INV = PWS_PERM_EXP + SZ_PERM;
constexpr uint32_t PWS_XSORT = PWS_INV + SZ_INV;
constexpr uint32_t PWS_AQ = PWS_XSORT + SZ_XSORT;
constexpr uint32_t PWS_AS = PWS_AQ + SZ_AQ;
constexpr uint32_t PWS_AQ_SHD = PWS_AS + SZ_AS;
constexpr uint32_t PWS_AS_SHD = PWS_AQ_SHD + SZ_AQ_SHD;
constexpr uint32_t PWS_GU = PWS_AS_SHD + SZ_AS_SHD;
constexpr uint32_t PWS_GU_SHD = PWS_GU + SZ_GU;
constexpr uint32_t PWS_H = PWS_GU_SHD + SZ_GU_SHD;
constexpr uint32_t PWS_H_SHD = PWS_H + SZ_H;
constexpr uint32_t PWS_HQ = PWS_H_SHD + SZ_H_SHD;
constexpr uint32_t PWS_HS = PWS_HQ + SZ_HQ;
constexpr uint32_t PWS_HQ_SHD = PWS_HS + SZ_HS;
constexpr uint32_t PWS_HS_SHD = PWS_HQ_SHD + SZ_HQ_SHD;
constexpr uint32_t PWS_Y = PWS_HS_SHD + SZ_HS_SHD;
constexpr uint32_t PWS_Y_SHD = PWS_Y + SZ_Y;
constexpr uint32_t PWS_ROUTED = PWS_Y_SHD + SZ_Y_SHD;
constexpr uint32_t PWS_SHARED = PWS_ROUTED + SZ_ROUTED;
constexpr uint32_t PWS_MOE = PWS_SHARED + SZ_SHARED;
constexpr uint32_t PWS_RES2 = PWS_MOE + SZ_MOE;
constexpr uint32_t PWS_BYTES = PWS_RES2 + SZ_RES2;

// 外部的层间残差流（S1 的 res 入口；prefill 用全 0 ⇒ 必须是 host 提供的 64 行零缓冲）
// 它**不在 ws 内**（与 decode 档的 resZero 同一协议），故不计入 PWS_BYTES。

// ============================================================
// 5. BufferID（核内）与 flagId（核间）清单
// ============================================================
// 复用 decode 档 MoE 段已登记的 id：prefill MoE 相位只出现在 `*_prefill` 入口里，
// 与该入口内的 GDN/attention 段**不同 (核型, mode) 子空间**相撞时会被 Wave C 的全局表挡住。
constexpr AscendC::MutexID PFR_BUF_A0 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_A0);
constexpr AscendC::MutexID PFR_BUF_A1 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_A1);
constexpr AscendC::MutexID PFR_BUF_B0 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_B0);
constexpr AscendC::MutexID PFR_BUF_B1 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_B1);
constexpr AscendC::MutexID PFR_BUF_L0_0 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_L00);
constexpr AscendC::MutexID PFR_BUF_L0_1 = static_cast<AscendC::MutexID>(M15M::BUF_AIC_L01);
constexpr AscendC::MutexID PFR_BUF_L0C = static_cast<AscendC::MutexID>(M15M::BUF_AIC_L0C);

constexpr AscendC::MutexID PFR_BUF_ROW0 = static_cast<AscendC::MutexID>(M15M::BUF_AIV_ROW0);
constexpr AscendC::MutexID PFR_BUF_ROW1 = static_cast<AscendC::MutexID>(M15M::BUF_AIV_ROW1);
constexpr AscendC::MutexID PFR_BUF_IDX = static_cast<AscendC::MutexID>(M15M::BUF_AIV_IDX);
constexpr AscendC::MutexID PFR_BUF_OUT = static_cast<AscendC::MutexID>(M15M::BUF_AIV_OUT);
constexpr AscendC::MutexID PFR_BUF_PERM0 = static_cast<AscendC::MutexID>(M15M::BUF_AIV_PERM0);
constexpr AscendC::MutexID PFR_BUF_STG = static_cast<AscendC::MutexID>(M15M::BUF_AIV_STG);
constexpr AscendC::MutexID PFR_BUF_GAMMA = static_cast<AscendC::MutexID>(M15M::BUF_AIV_GAMMA);
constexpr AscendC::MutexID PFR_BUF_RSTD = static_cast<AscendC::MutexID>(M15M::BUF_AIV_RSTD);

// flagId：与 decode MoE 段同一组（见 `m15_moe_resources.h` §3 的相邻性论证）。
constexpr uint16_t PFR_FLAG_AIV_RING[4] = {12, 13, 14, 15};
constexpr uint16_t PFR_FLAG_AIC_RING[4] = {8, 9, 10, 11};
constexpr uint16_t PFR_FLAG_M2_RING[4] = {0, 1, 2, 3};
constexpr uint16_t PFR_FLAG_PHASE_BOUND = M15M::FLAG_L0_BOUND_AIV;   // 8（相位边界，全体 AIV mode 0）
constexpr uint8_t PFR_CC_MODE0 = 0x0;
constexpr uint8_t PFR_CC_MODE2 = 0x2;

// 相邻性断言（本段自己的执行序）：
//   AIV mode0:  12 → 13 → 14 → 15 → 12 → 13   （6 次，见下面的 PF_AIV_BARRIER_SEQ）
//   AIC mode0:   8 →  9 → 10 → 11 →  8         （5 次）
//   mode2    :   0 →  1 →  2 →  3 →  0 → 1     （6 次；wait/set 成对）
// 同核相邻两个同步点必须不同 id（docs/05 §2）。每 id 每相位用量 ≤ 2 ≪ 15（硬件 4 bit 计数器）。
//
// ---- AIV 的全体 mode-0 屏障：**把"第几步"与"用哪个 id"集中到一张表**（r2 复审残余 2 的处置）----
// 复审指出 `PfRingAdjacentOk` 只校验 4 元 ring 的**环邻**，并不把 §5 注释里的 6 次执行序与代码绑起来
//（"若哪天把同一槽连用两次，这条断言不会红"）。⇒ r3 起改成本表 + `BarrierAivStep<pipe>(step)`：
// 调用点只写**步号**（0..5，编译期常量），id 一律从本表取，`PfSeqAdjacentOk` 对**实际长度**的序列
// 逐对断言相邻不同。
// **本表能绑住什么**：步号 → id 的映射（改表即改全部调用点）；相邻两步的 id 必须不同。
// **绑不住什么**：调用点写错步号（例如两次 step=3）仍是源码级错误 —— 那一步无法由 static_assert 抓，
// 只能靠复审逐点核对（r2 复审已按代码手数过一次）。
constexpr uint16_t PF_AIV_BARRIER_SEQ[6] = {12, 13, 14, 15, 12, 13};
constexpr uint32_t PF_AIV_BARRIER_N = 6;
constexpr bool PfSeqAdjacentOk(const uint16_t* r, uint32_t n)
{
    for (uint32_t i = 0; i + 1u < n; ++i) {
        if (r[i] == r[i + 1u]) {
            return false;
        }
    }
    return true;
}
static_assert(PfSeqAdjacentOk(PF_AIV_BARRIER_SEQ, PF_AIV_BARRIER_N),
              "AIV 全体 mode-0 屏障的相邻两步用了同一个 flagId（docs/05 §2）");
static_assert(PF_AIV_BARRIER_SEQ[0] == PFR_FLAG_AIV_RING[0] &&
                  PF_AIV_BARRIER_SEQ[1] == PFR_FLAG_AIV_RING[1] &&
                  PF_AIV_BARRIER_SEQ[2] == PFR_FLAG_AIV_RING[2] &&
                  PF_AIV_BARRIER_SEQ[3] == PFR_FLAG_AIV_RING[3],
              "屏障步序表与 PFR_FLAG_AIV_RING 的槽位定义不一致");
static_assert(PF_AIV_BARRIER_SEQ[4] == PFR_FLAG_AIV_RING[0] &&
                  PF_AIV_BARRIER_SEQ[5] == PFR_FLAG_AIV_RING[1],
              "屏障步序表的第 2 轮复用必须与 ring 一致（4 槽轮转）");
constexpr bool PfRingAdjacentOk(const uint16_t* r)
{
    for (uint32_t i = 0; i < 4u; ++i) {
        if (r[i] == r[(i + 1u) % 4u]) {
            return false;
        }
    }
    return true;
}
static_assert(PfRingAdjacentOk(PFR_FLAG_AIV_RING), "AIV mode-0 ring 的相邻 id 撞了");
static_assert(PfRingAdjacentOk(PFR_FLAG_AIC_RING), "AIC mode-0 ring 的相邻 id 撞了");
static_assert(PfRingAdjacentOk(PFR_FLAG_M2_RING), "mode-2 ring 的相邻 id 撞了");
static_assert(PFR_FLAG_PHASE_BOUND != PFR_FLAG_AIV_RING[3], "相位边界 id 与 AIV ring 尾槽相同");
static_assert(PFR_FLAG_PHASE_BOUND != PFR_FLAG_AIV_RING[0], "相位边界 id 与 AIV ring 首槽相同");

// ============================================================
// 6. `m` / `pos` 语义（供融合清单的 (f) 项）
// ============================================================
// 本段接受 **m ∈ [1, M_PREFILL]**；块内行数 curM = min(MT, m - blk*MT)。
// 位置 pos 由挂载点提供（prefill 是单请求连续 [start, start+m) 或 per-row 表），本段**不消费 pos**：
// MoE 段按行独立（路由/permute 都是 token 内语义），位移只影响 attention 段的 KV 落点。
constexpr uint32_t PF_BLK_MAX = (M_PREFILL + MT - 1u) / MT;   // 65 块（m=4097 时最后一块 1 行）
static_assert(PF_BLK_MAX == 65u, "m=4097 / MT=64 的块数变了");

}  // namespace M15PFR

#endif  // M15_MOE_PREFILL_RES_H
