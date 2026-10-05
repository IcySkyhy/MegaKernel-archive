/**
 * m17_resources.h —— MoE block **真实权重**层 kernel 全局静态资源表
 *
 * 本表是 m13_resources.h（golden 合成数据集，E=4 / topk≤4）的**真实形状版本**：
 * 所有形状常量来自 /workspace/Qwen3.8-Flash-Next-MXFP4 的 config.json
 * （text_config：hidden_size 2560 / moe_intermediate_size 640 / num_experts 512 /
 *  num_experts_per_tok 10），权重契约（packed u8 e2m1 lohi + e8m0 scale，group 32 沿 K）
 * 与 m13 完全一致 —— 因此 S4..S10 的算件（permute / 量化 / grouped GEMM / unpermute /
 * combine / RMSNorm）逐字沿用 m13，只有与 E、TOPK 强相关的资源与段被改写。
 *
 * 相对 m13 的**逐条改动**（README「与 m13 的差异」表有完整对照）：
 *   1. NUM_EXPERTS  4   → 512
 *      TOPK_MAX     4   → 10
 *      TOTAL_MAX    256 → 640
 *   2. router 段 UB 布局整段替换：m13 把 [E+1][HIDDEN] 的 router 权重**全量**预转 fp32 常驻
 *      UB（E=4 时 51KB），E=512 时需要 5.25MB → 不可能。改为 m7_router_topk 的
 *      **流式**布局：x 行块 8 行常驻（RT_RB=8）+ 权重行 ping/pong + 8 专家一组的 fp32 预转窗
 *      （EGRP×HIDDEN×4 = 80KB）。router 权重仍按 B 侧 [E, HIDDEN] 原始 layout 消费。
 *   3. router 的 top-k 由 m13 的「单块 Sort32（只覆盖 E≤32）」换成 m7 的
 *      **Sort32 + 4 级 2 路 `MrgSort` 归并树**（16×32 → 根节点 = 全局 top-64，取前 10；
 *      每级 elementLengths=[32,32,0,0] 只出 64 对，**不是** 512 对全序）。
 *      API 说明：不用 `AscendC::MrgSort4`（dav-3510 上静默 no-op，probe_sync_quirks
 *      A02/A15/A20 + docs/05 §6 #19；同参数 4 路 `MrgSort` 真机正确）。
 *   4. 索引生成段的 cnt/off/cursor 由**栈数组**（E×3×4B → 512 时 6KB，AIV 标量栈吃不消）
 *      搬到 UB 静态槽位；offsets 槽位从 `cnt[8]` 挪到 `cnt[E]`（E=512 时原位置与 counts 重叠）。
 *   5. UnpermuteStage 的 k 链由 1..3 展开到 1..9（TOPK_MAX=10）。
 *   6. unpermute 的 Y 视图尺寸声明改为 NUM_EXPERTS*M_MAX*HIDDEN（m13 里
 *      M_MAX*TOPK_MAX == NUM_EXPERTS*M_MAX 是 E=4 的巧合，E=512 时不等）。
 *   7. 补上 main 的 m5/m13 已落地的 ±Inf/NaN 平价修复（`NAN_CUSTOMIZATION=0x7f81` + `Select`，
 *      官方 `add_rms_norm_dynamic_mx_quant_common.h` 的 `NAN_CUSTOMIZATION` 覆盖路径同源；
 *      行号随上游变动、以符号为准）；本分支从合并 7ee8444
 *      **之前**分叉，m13/m5 当时还没有这条 —— 即"逐字沿用 m13"只对修复前的版本成立，
 *      上游变更**不会自动流入**（docs/17 §3 第 4 条），今后需手工同步。
 *
 * buffer 约定沿用 m13/M9：相邻同步点 flagId 必不同、set 挂生产 pipe、wait 挂最窄 pipe、
 * UB/L1 地址 32B 对齐、GM workspace 偏移全编译期常量。
 */

#ifndef M17_RESOURCES_H
#define M17_RESOURCES_H

#include <cstdint>

namespace M17 {

// ============================================================
// 0. 形状常量（Qwen3.8-Flash-Next-MXFP4 真实 MoE block）
// ============================================================

constexpr uint32_t HIDDEN = 2560;      // hidden_size（K of gate_up / N of down）
constexpr uint32_t INTER = 640;        // moe_intermediate_size
constexpr uint32_t GU_N = 2 * INTER;   // gate_up GEMM 输出宽度（gate|up 拼接）1280
constexpr uint32_t NUM_EXPERTS = 512;  // 路由专家数（真实 checkpoint）
constexpr uint32_t TOPK_MAX = 10;      // top-k 上界（config：num_experts_per_tok = 10）
constexpr uint32_t M_MAX = 64;         // decode 包络 m ≤ 64（BASE_M 单 tile）
constexpr uint32_t TOTAL_MAX = M_MAX * TOPK_MAX;  // Σt_e 上界

constexpr uint32_t GROUP = 32;         // MXFP4 K 方向 group
constexpr uint32_t GU_SCALE_STRIDE = HIDDEN / GROUP;      // A(=gate_up 输入) scale 行距 80B
constexpr uint32_t DN_SCALE_STRIDE = 32;                  // H(=down 输入) scale 行距 32B（20 有效）

// ============================================================
// 1. AIC BufferID（与 m13/m3 完全一致）
// ============================================================

constexpr uint32_t BUF_AIC_A0 = 0;     // L1 A(+scale) ping
constexpr uint32_t BUF_AIC_A1 = 1;     // L1 A(+scale) pong
constexpr uint32_t BUF_AIC_B0 = 2;     // L1 B ping
constexpr uint32_t BUF_AIC_B1 = 3;     // L1 B pong
constexpr uint32_t BUF_AIC_L00 = 4;    // L0A/L0B ping
constexpr uint32_t BUF_AIC_L01 = 5;    // L0A/L0B pong
constexpr uint32_t BUF_AIC_L0C = 6;    // L0C 累加
constexpr uint32_t BUF_AIC_RESERVED_PREFETCH0 = 7;
constexpr uint32_t BUF_AIC_RESERVED_FIX2UB0 = 11;

// ============================================================
// 2. AIV BufferID（与 m13 一致）
// ============================================================

constexpr uint32_t BUF_AIV_ROW0 = 0;
constexpr uint32_t BUF_AIV_ROW1 = 1;
constexpr uint32_t BUF_AIV_ROW2 = 2;
constexpr uint32_t BUF_AIV_ROW3 = 3;
constexpr uint32_t BUF_AIV_IDX = 4;
constexpr uint32_t BUF_AIV_OUT = 5;
constexpr uint32_t BUF_AIV_SCR = 6;
constexpr uint32_t BUF_AIV_SCR2 = 7;
constexpr uint32_t BUF_AIV_PERM0 = 15;
constexpr uint32_t BUF_AIV_PERM1 = 16;
constexpr uint32_t BUF_AIV_PERM2 = 17;
constexpr uint32_t BUF_AIV_PERM3 = 18;
constexpr uint32_t BUF_AIV_STG = 19;
constexpr uint32_t BUF_AIV_GAMMA = 20;
constexpr uint32_t BUF_AIV_RSTD = 21;
// router（S2）段专用 token（与 m13 不同：m13 复用 ROW0/ROW1/STG；本版新增 22-25，
// 因为流式 router 有 4 条独立的 MTE2→V 交接：x 块 / w ping / w pong / sgate 权重行）
constexpr uint32_t BUF_AIV_RT_X = 22;    // x 行块 MTE2 → V
constexpr uint32_t BUF_AIV_RT_W0 = 23;   // w 行 ping MTE2 → V
constexpr uint32_t BUF_AIV_RT_W1 = 24;   // w 行 pong MTE2 → V
constexpr uint32_t BUF_AIV_RT_SG = 25;   // sgate 权重行 MTE2 → V
constexpr uint32_t BUF_AIV_RT_LOG = 26;  // logits/id/weights staging V → MTE3

// ============================================================
// 3. CrossCore flagId 旋转槽位（与 m13 一致）
// ============================================================

constexpr uint16_t FLAG_AIV_SEG_RING[4] = {0, 1, 2, 3};
constexpr uint16_t FLAG_AIC_SEG_RING[4] = {4, 5, 6, 7};
constexpr uint16_t FLAG_M2_RING[4] = {8, 9, 10, 11};
constexpr uint16_t FLAG_RESERVED_HI = 12;

constexpr uint32_t NUM_SEG_SLOTS = 4;

constexpr uint8_t CC_MODE0 = 0x0;
constexpr uint8_t CC_MODE2 = 0x2;

// ============================================================
// 4. UB 静态布局（字节偏移；SEG-VEC 窗内各段同址叠放、barrier 分隔）
// ============================================================

constexpr uint32_t UB_BYTES_TOTAL = 248 * 1024;

// ---- PERSIST @0 ----
constexpr uint32_t UB_PERSIST = 0;
constexpr uint32_t UB_PERSIST_BYTES = 16384;
// UB_ZEROS_B16：m13 用它放 bf16[HIDDEN] 全 0 行（residual 占位）。本 mission 的残差由
// host 侧的 `res_zero` GM 缓冲提供（S1 的 residual 入口），故该槽位**保留未用**（与
// UB_CB_OUT 同：显式保留以免改动 m13 的 UB 平移关系，避免同址叠放区错位）
constexpr uint32_t UB_ZEROS_B16 = UB_PERSIST + 0;

// ---- gamma fp32 预转 @16384 ----
constexpr uint32_t UB_GAMMA_F32 = UB_PERSIST + UB_PERSIST_BYTES;
constexpr uint32_t UB_GAMMA1_F32 = UB_GAMMA_F32;
constexpr uint32_t UB_GAMMA2_F32 = UB_GAMMA1_F32 + HIDDEN * 4;

// ---- SEG-VEC 窗 @36864（各段同址叠放）----
constexpr uint32_t UB_VEC = UB_GAMMA2_F32 + HIDDEN * 4;            // @36864

// S1/S10  m6 残差 RMSNorm 行 scratch
constexpr uint32_t UB_M6_X1 = UB_VEC + 0;                 // bf16[HIDDEN]
constexpr uint32_t UB_M6_RES = UB_M6_X1 + HIDDEN * 2;     // bf16[HIDDEN]（fp32 残差时占 10240B）
constexpr uint32_t UB_M6_XF = UB_M6_RES + HIDDEN * 4;     // fp32[HIDDEN]
constexpr uint32_t UB_M6_Y = UB_M6_XF + HIDDEN * 4;       // bf16[HIDDEN]
constexpr uint32_t UB_M6_TMP = UB_M6_Y + HIDDEN * 2;      // fp32[64]
constexpr uint32_t UB_M6_RED = UB_M6_TMP + 256;           // fp32[64]
constexpr uint32_t UB_M6_RSTD = UB_M6_RED + 256;          // fp32[64]

// S2 router（m7 流式结构；E=512 时权重表不能常驻 UB）
//   注：RT_RB 取 8（m7 用 16）—— E=512 让 x 块与 logits 行块各放大 128 倍，
//   RB=16 时 RT_END 会超出 248KB；RB=8 时 RT_END = 222752B，留 31200B 余量。
constexpr uint32_t RT_RB = 8;                             // x 行块行数（m ≤ 64 → 最多 8 块）
constexpr uint32_t RT_EGRP = 8;                           // GEMV 专家分组（共享 x 载入；8 点积拼向量 store）
constexpr uint32_t RT_XB = UB_VEC + 0;                               // bf16[RT_RB][HIDDEN] = 40960B（RT_RB=8）
constexpr uint32_t RT_WB = RT_XB + RT_RB * HIDDEN * 2;               // bf16[2][HIDDEN] ping/pong = 10240B
constexpr uint32_t RT_WF = RT_WB + 2 * HIDDEN * 2;                   // fp32[RT_EGRP][HIDDEN] 预转窗 = 81920B
constexpr uint32_t RT_SGB = RT_WF + RT_EGRP * HIDDEN * 4;            // bf16[HIDDEN] sgate 权重行原始 bf16 = 5120B
constexpr uint32_t RT_SGWF = RT_SGB + HIDDEN * 2;                    // fp32[HIDDEN] sgate 权重行 fp32 预转 = 10240B
constexpr uint32_t RT_LOG = RT_SGWF + HIDDEN * 4;                    // fp32[RT_RB][NUM_EXPERTS] = 16384B（RT_RB=8）
constexpr uint32_t RT_VAL = RT_LOG + RT_RB * NUM_EXPERTS * 4;        // fp32[NUM_EXPERTS] e_i = 2048B
constexpr uint32_t RT_IDX = RT_VAL + NUM_EXPERTS * 4;                // u32[NUM_EXPERTS] arange 模板 = 2048B
// 以下 MA/MB/TMP 的尺寸只取决于 NUM_EXPERTS（每块 NUM_EXPERTS=512 对 = 2×512 个 fp32），与 RT_RB 无关 → 4096B 不变
constexpr uint32_t RT_TMP = RT_IDX + NUM_EXPERTS * 4;                // fp32[2*NUM_EXPERTS] Sort32 对 = 4096B
constexpr uint32_t RT_MA = RT_TMP + 2 * NUM_EXPERTS * 4;             // fp32[2*NUM_EXPERTS] 归并缓冲 A = 4096B
constexpr uint32_t RT_MB = RT_MA + 2 * NUM_EXPERTS * 4;              // fp32[2*NUM_EXPERTS] 归并缓冲 B = 4096B
constexpr uint32_t RT_OUTV = RT_MB + 2 * NUM_EXPERTS * 4;            // fp32[64] Extract 值 = 256B
constexpr uint32_t RT_OUTI = RT_OUTV + 64 * 4;                       // u32[64] Extract 索引 = 256B
constexpr uint32_t RT_IDS = RT_OUTI + 64 * 4;                        // i32[RT_RB][64] 行 staging = 2048B（RT_RB=8）
constexpr uint32_t RT_WS = RT_IDS + RT_RB * 64 * 4;                  // fp32[RT_RB][64] 行 staging = 2048B（RT_RB=8）
constexpr uint32_t RT_SG = RT_WS + RT_RB * 64 * 4;                   // fp32[RT_RB] 共享门裸点积 = 32B（RT_RB=8）
constexpr uint32_t RT_END = RT_SG + RT_RB * 4;

// S3  index generator（AIV0）：自 UB_VEC 起的独立叠放区（barrier 分隔）
//     布局：counts[E] @0、offsets[E+1] @E（32B 对齐：E*4=2048B）、
//     诊断槽 @(2E+8)（(2E+8)*4=4128B 亦 32B 对齐）、cursor[E]
// 各槽位按 32B 对齐（DataCopyPad 源要求）：TOTAL_MAX*4、NUM_EXPERTS*4、M_MAX*16*4 本身
// 都是 32B 倍数，只有 counts+offsets+诊断槽的 2E+9 个 int32 需要向上取整。
constexpr uint32_t IG_CNT_I32 = (2 * NUM_EXPERTS + 9) * 4;              // 4132B
constexpr uint32_t IG_CNT_BYTES = (IG_CNT_I32 + 31) / 32 * 32;         // 4160B
constexpr uint32_t UB_IG_SRC = UB_VEC + 0;                             // i32[TOTAL_MAX] perm_src
constexpr uint32_t UB_IG_EXP = UB_IG_SRC + TOTAL_MAX * 4;              // i32[TOTAL_MAX] perm_expert
constexpr uint32_t UB_IG_CNT = UB_IG_EXP + TOTAL_MAX * 4;              // i32[2E+9] counts/offsets/诊断
constexpr uint32_t UB_IG_CUR = UB_IG_CNT + IG_CNT_BYTES;               // i32[E] cursor
constexpr uint32_t UB_IG_INV = UB_IG_CUR + NUM_EXPERTS * 4;            // i32[M_MAX*TOPK_MAX] inv_slot
constexpr uint32_t UB_IG_WTK = UB_IG_INV + M_MAX * TOPK_MAX * 4;       // i32[M_MAX*16] w_tk_packed
constexpr uint32_t UB_IG_END = UB_IG_WTK + M_MAX * 16 * 4;
constexpr uint32_t IG_CNT_OFF_SLOT = NUM_EXPERTS;      // offsets 起始 int32 下标（counts 之后）
constexpr uint32_t IG_CNT_DIAG_SLOT = 2 * NUM_EXPERTS + 8;   // UB 诊断槽（IG 越界计数）
// GM 侧的诊断槽位置：放在 offsets[E+1] 之后（m13 放在 offs[8] 会覆盖 offsets[8]）
constexpr uint32_t IG_DIAG_GM_SLOT = NUM_EXPERTS + 8;
constexpr uint32_t IG_DIAG_SLOT_END = 16;   // offsGm 视图尾部多留的 int32 数
static_assert(UB_IG_END <= RT_END, "IG 区与 router 区同址叠放，不可越界");

// S4  permute（m8#1，四级流水）
constexpr uint32_t UB_PM_ROW = UB_VEC + 0;                       // 4 × bf16[HIDDEN] = 20480B

// S5/S7  MXFP4 量化（m2/m5 全 VEC 路径；per-256-element tile）
constexpr uint32_t UB_QT_GATE = UB_VEC + 0;
constexpr uint32_t UB_QT_UP = UB_QT_GATE + 512;
constexpr uint32_t UB_QT_SWIGLU = UB_QT_GATE + 1536;
constexpr uint32_t UB_QT_QX = UB_QT_SWIGLU + 512;
constexpr uint32_t UB_QT_SCALE = UB_QT_QX + 128;
constexpr uint32_t UB_QT_HALF = UB_QT_SCALE + 32;

// S9  unpermute（m8#2，双缓冲 stage）
constexpr uint32_t U_STAGE_B = (TOPK_MAX + 1) * HIDDEN * 2 + 64;
constexpr uint32_t UB_UP_STAGE0 = UB_VEC + 0;
constexpr uint32_t UB_UP_STAGE1 = UB_UP_STAGE0 + U_STAGE_B;
constexpr uint32_t UB_UP_END = UB_UP_STAGE1 + U_STAGE_B;

// S9b combine
constexpr uint32_t UB_CB_ROUTED = UB_VEC + 0;      // bf16[HIDDEN] routed（原地改写成 moe）
constexpr uint32_t UB_CB_SHARED = UB_CB_ROUTED + HIDDEN * 2;   // bf16[HIDDEN] shared（原地改写成 gated）
constexpr uint32_t UB_CB_OUT = UB_CB_SHARED + HIDDEN * 2;      // 保留未用（m13 原布局占位，见上）
constexpr uint32_t UB_CB_G = UB_CB_OUT + HIDDEN * 2;           // fp32[64] sgate 广播槽

static_assert(RT_END <= UB_BYTES_TOTAL, "router UB footprint exceeds UB");
static_assert(UB_UP_END <= UB_BYTES_TOTAL, "unpermute UB footprint exceeds UB");
static_assert(UB_IG_END <= UB_BYTES_TOTAL, "index-gen UB footprint exceeds UB");

// ============================================================
// 5. L1 / L0 静态布局（AIC；与 m13/m3 一致）
// ============================================================

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t BASE_M = 64;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t BASE_N = 256;

constexpr uint32_t L1_A_DATA = BASE_M * (BASE_K / 2);
constexpr uint32_t L1_A_SCAL = BASE_M * (BASE_K / GROUP);
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_A_DATA;
constexpr uint32_t L1_OFF_AS0 = 2 * L1_A_DATA;
constexpr uint32_t L1_OFF_AS1 = L1_OFF_AS0 + L1_A_SCAL;
constexpr uint32_t L1_B_DATA = BASE_K * BASE_N / 2;
constexpr uint32_t L1_B_SCAL = BASE_N * (BASE_K / GROUP);
constexpr uint32_t L1_B_REGION = 256 * 1024;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_DATA;
constexpr uint32_t L1_OFF_BS0 = L1_B_REGION + 2 * L1_B_DATA;
constexpr uint32_t L1_OFF_BS1 = L1_OFF_BS0 + L1_B_SCAL;
constexpr uint32_t L1_PREFETCH_BASE = 330 * 1024;
constexpr uint32_t L1_BYTES_TOTAL = 512 * 1024;

constexpr uint32_t L0_PP_BYTES = 32 * 1024;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;

// ============================================================
// 6. GM workspace 偏移（单位字节，32B 对齐）
// ============================================================
// 尺寸全部按真实形状展开（E=512 / TOPK=10 / M_MAX=64）：total ≈ 322MB。
// 每个专家槽位都是 [slot][M_MAX][row] 的 padding 布局，故 E=512 时槽位表占主导；
// 这是 m13 契约的直接放大（m13 的解释见其 §6 与 README「GM 平面图」）。

constexpr uint32_t WS_ALIGN = 32;
constexpr uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

constexpr uint32_t SZ_XNORM = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_RES1 = AlignUp(M_MAX * HIDDEN * 4, WS_ALIGN);
constexpr uint32_t SZ_LOGITS = AlignUp(M_MAX * NUM_EXPERTS * 4, WS_ALIGN);
constexpr uint32_t SZ_IDS = AlignUp(M_MAX * TOPK_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_W = AlignUp(M_MAX * TOPK_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_PERM = AlignUp(TOTAL_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_COUNTS = AlignUp(NUM_EXPERTS * 4, WS_ALIGN);
constexpr uint32_t SZ_OFFSETS = AlignUp((NUM_EXPERTS + 1) * 4 + 4, WS_ALIGN) + 32;
constexpr uint32_t SZ_INV = AlignUp(M_MAX * TOPK_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_WTK = AlignUp(M_MAX * 16 * 4, WS_ALIGN);
constexpr uint32_t SZ_SGATE = AlignUp(M_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_XSORT = AlignUp(TOTAL_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_AQ = AlignUp(NUM_EXPERTS * M_MAX * (HIDDEN / 2), WS_ALIGN);
constexpr uint32_t SZ_AS = AlignUp(NUM_EXPERTS * M_MAX * GU_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_AQ_SHD = AlignUp(M_MAX * (HIDDEN / 2), WS_ALIGN);
constexpr uint32_t SZ_AS_SHD = AlignUp(M_MAX * GU_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_GU = AlignUp(NUM_EXPERTS * M_MAX * GU_N * 2, WS_ALIGN);
constexpr uint32_t SZ_GU_SHD = AlignUp(M_MAX * GU_N * 2, WS_ALIGN);
constexpr uint32_t SZ_H = AlignUp(TOTAL_MAX * INTER * 2, WS_ALIGN);
constexpr uint32_t SZ_H_SHD = AlignUp(M_MAX * INTER * 2, WS_ALIGN);
constexpr uint32_t SZ_HQ = AlignUp(NUM_EXPERTS * M_MAX * (INTER / 2), WS_ALIGN);
constexpr uint32_t SZ_HS = AlignUp(NUM_EXPERTS * M_MAX * DN_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_HQ_SHD = AlignUp(M_MAX * (INTER / 2), WS_ALIGN);
constexpr uint32_t SZ_HS_SHD = AlignUp(M_MAX * DN_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_Y = AlignUp(NUM_EXPERTS * M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_Y_SHD = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_ROUTED = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_SHARED = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_MOE = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_YFINAL = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_RES2 = AlignUp(M_MAX * HIDDEN * 4, WS_ALIGN);

constexpr uint32_t WS_XNORM = 0;
constexpr uint32_t WS_RES1 = WS_XNORM + SZ_XNORM;
constexpr uint32_t WS_LOGITS = WS_RES1 + SZ_RES1;
constexpr uint32_t WS_IDS = WS_LOGITS + SZ_LOGITS;
constexpr uint32_t WS_W = WS_IDS + SZ_IDS;
constexpr uint32_t WS_PERM_SRC = WS_W + SZ_W;
constexpr uint32_t WS_PERM_EXP = WS_PERM_SRC + SZ_PERM;
constexpr uint32_t WS_COUNTS = WS_PERM_EXP + SZ_PERM;
constexpr uint32_t WS_OFFSETS = WS_COUNTS + SZ_COUNTS;
constexpr uint32_t WS_INV = WS_OFFSETS + SZ_OFFSETS;
constexpr uint32_t WS_WTK = WS_INV + SZ_INV;
constexpr uint32_t WS_SGATE = WS_WTK + SZ_WTK;
constexpr uint32_t WS_XSORT = WS_SGATE + SZ_SGATE;
constexpr uint32_t WS_AQ = WS_XSORT + SZ_XSORT;
constexpr uint32_t WS_AS = WS_AQ + SZ_AQ;
constexpr uint32_t WS_AQ_SHD = WS_AS + SZ_AS;
constexpr uint32_t WS_AS_SHD = WS_AQ_SHD + SZ_AQ_SHD;
constexpr uint32_t WS_GU = WS_AS_SHD + SZ_AS_SHD;
constexpr uint32_t WS_GU_SHD = WS_GU + SZ_GU;
constexpr uint32_t WS_H = WS_GU_SHD + SZ_GU_SHD;
constexpr uint32_t WS_H_SHD = WS_H + SZ_H;
constexpr uint32_t WS_HQ = WS_H_SHD + SZ_H_SHD;
constexpr uint32_t WS_HS = WS_HQ + SZ_HQ;
constexpr uint32_t WS_HQ_SHD = WS_HS + SZ_HS;
constexpr uint32_t WS_HS_SHD = WS_HQ_SHD + SZ_HQ_SHD;
constexpr uint32_t WS_Y = WS_HS_SHD + SZ_HS_SHD;
constexpr uint32_t WS_Y_SHD = WS_Y + SZ_Y;
constexpr uint32_t WS_ROUTED = WS_Y_SHD + SZ_Y_SHD;
constexpr uint32_t WS_SHARED = WS_ROUTED + SZ_ROUTED;
constexpr uint32_t WS_MOE = WS_SHARED + SZ_SHARED;
constexpr uint32_t WS_YFINAL = WS_MOE + SZ_MOE;
constexpr uint32_t WS_RES2 = WS_YFINAL + SZ_YFINAL;
constexpr uint32_t WS_BYTES = WS_RES2 + SZ_RES2;

}  // namespace M17

#endif  // M17_RESOURCES_H
