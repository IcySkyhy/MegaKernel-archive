/**
 * m13_resources.h —— MoE block layer kernel 全局静态资源表（docs/12-layer-integration.md §4）
 *
 * 单一全局编译期资源表（docs/05 §5.2）：BufferID / CrossCore flagId / UB 偏移 /
 * L1 偏移 / GM workspace 偏移 全部编译期静态常量，模块只声明 footprint、不自持有内存。
 * 所有地址单位为**字节**，且 32B 对齐（UB→GM DataCopy 的硬要求，M9 quirk）。
 *
 * 与 docs/12 §4 的对应关系（逐条）：
 *   - AIC BufferID 0-10 + 11-13 预留：§4「0-1=A(+scale)、2-3=B、4-5=L0A/L0B、
 *     6=L0C、7-10=第二组 GEMM、11-13=FIXP→UB 交接预留」。本 mission 的 MoE 段只用
 *     到 0-6（与 m3 兼容）；7-13 保留给跨段权重预取（§4）与后续 GDN/CV 段。
 *   - AIV BufferID 0-7 = 通用行 ping-pong 窗 4 对（m2/m5/m6/m8#2/m9/m3-AIV 段间
 *     barrier 分隔，地址可叠放 → 段内局部 id 全叠到 0-7）；15-18 = m8#1 四级流水；
 *     19-23 = 预取 staging / gamma fp32 预转 / rstd scratch。8-14 为 m4 递推专用
 *     （本 mission 不含 GDN，保留不动）。
 *   - CrossCore flagId 旋转槽位制：0-3 = AIV 侧段边界 mode 0（每段边界换一个槽，
 *     相邻必不同 id）；4-7 = AIC 侧段边界 mode 0；8-11 = 段内 mode-2 流式对；
 *     12-15 保留。（§4 原文「0-3=MoE 段边界 / 4-7=GDN / 8-11=段内 mode-2」——本
 *     mission 把 AIC 侧段边界放在 4-7、AIV/AIC 的槽位编号完全错开（避免同一 flagId
 *     在同一核上跨模式复用），相邻同步点不同 id 的硬规则与「每 id 计数 ≪15」均满足，
 *     README 有完整同步表。）
 *   - UB：PERSIST @0（16KB，§4 名义 ~12KB，本实现多留 4KB 常量 staging）；
 *     gamma fp32 预转 20KB 紧随其后（@16384）；SEG-VEC 窗 @36864（§4 @16384 ~112KB）——
 *     本实现窗内峰值占用 ~153KB（router 的 x 块 80KB + 权重 fp32 51KB 为最大消费者，
 *     见 UB_RT_END = 193600），超出 §4 的 ~112KB 名义值；各段仍按 barrier 分隔同址叠放，
 *     UB 总量 248KB 内留有 ~55KB 余量（不含 GDN 段）。
 *   - L1：A 区 @0 ×2 组、B 区 @262144 ×2 组（与 §4 一致）；@330KB 起为权重预取
 *     滚动窗（本 mission 未实现跨 op 预取，仅留出边界）。
 */

#ifndef M13_RESOURCES_H
#define M13_RESOURCES_H

#include <cstdint>

namespace M13 {

// ============================================================
// 0. 形状常量（Qwen3.8-Flash-Next-MXFP4 MoE block；golden 数据集缩形）
// ============================================================

constexpr uint32_t HIDDEN = 2560;      // hidden_size（K of gate_up / N of down）
constexpr uint32_t INTER = 640;        // moe_intermediate_size
constexpr uint32_t GU_N = 2 * INTER;   // gate_up GEMM 输出宽度（gate|up 拼接）1280
constexpr uint32_t NUM_EXPERTS = 4;    // 专家数（golden 数据集：512 → 4）
constexpr uint32_t TOPK_MAX = 4;       // top-k 模板上界（golden 数据集：10 → 2）
constexpr uint32_t M_MAX = 64;         // decode 包络 m ≤ 64（BASE_M 单 tile）
constexpr uint32_t TOTAL_MAX = M_MAX * TOPK_MAX;  // Σt_e 上界

constexpr uint32_t GROUP = 32;         // MXFP4 K 方向 group
constexpr uint32_t GU_SCALE_STRIDE = HIDDEN / GROUP;      // A(=gate_up 输入) scale 行距 80B
constexpr uint32_t DN_SCALE_STRIDE = 32;                  // H(=down 输入) scale 行距 32B（20 有效）

// ============================================================
// 1. AIC BufferID（0-10 用，11-13 预留，14-27 扩展）
// ============================================================

constexpr uint32_t BUF_AIC_A0 = 0;     // L1 A(+scale) ping
constexpr uint32_t BUF_AIC_A1 = 1;     // L1 A(+scale) pong
constexpr uint32_t BUF_AIC_B0 = 2;     // L1 B ping
constexpr uint32_t BUF_AIC_B1 = 3;     // L1 B pong
constexpr uint32_t BUF_AIC_L00 = 4;    // L0A/L0B ping
constexpr uint32_t BUF_AIC_L01 = 5;    // L0A/L0B pong
constexpr uint32_t BUF_AIC_L0C = 6;    // L0C 累加
// 7-10  §4 第二组 GEMM ping/pong（跨段权重预取）——本 mission 未使用
// 11-13 §4 FIXP→UB 直写交接预留——本 mission 未使用
constexpr uint32_t BUF_AIC_RESERVED_PREFETCH0 = 7;
constexpr uint32_t BUF_AIC_RESERVED_FIX2UB0 = 11;

// ============================================================
// 2. AIV BufferID（0-7 通用行窗；15-18 permute 四级流水；19-23 scratch）
// ============================================================

constexpr uint32_t BUF_AIV_ROW0 = 0;   // 通用行 ping（各 AIV 段 barrier 分隔，可叠放）
constexpr uint32_t BUF_AIV_ROW1 = 1;   // 通用行 pong
constexpr uint32_t BUF_AIV_ROW2 = 2;
constexpr uint32_t BUF_AIV_ROW3 = 3;
constexpr uint32_t BUF_AIV_IDX = 4;    // 索引/小控制 staging（MTE2→S / S→MTE3）
constexpr uint32_t BUF_AIV_OUT = 5;    // V→MTE3 行输出
constexpr uint32_t BUF_AIV_SCR = 6;    // 核内 scratch 交接
constexpr uint32_t BUF_AIV_SCR2 = 7;
// 8-14 m4 递推专用（GDN 段）——保留
constexpr uint32_t BUF_AIV_PERM0 = 15; // m8#1 四级流水（§4 15-18）
constexpr uint32_t BUF_AIV_PERM1 = 16;
constexpr uint32_t BUF_AIV_PERM2 = 17;
constexpr uint32_t BUF_AIV_PERM3 = 18;
constexpr uint32_t BUF_AIV_STG = 19;   // 预取/权重 staging（§4 19-23）
constexpr uint32_t BUF_AIV_GAMMA = 20; // gamma fp32 预转
constexpr uint32_t BUF_AIV_RSTD = 21;  // rstd scratch

// ============================================================
// 3. CrossCore flagId 旋转槽位
// ============================================================
// 相邻同步点必须不同 id（M2 实证：同核连续 set 不保序）；每 id 每层用量 ≪ 15。
// 每次「全体 AIV / 全体 AIC mode 0」按槽位轮转，槽位满 4 轮流复用（复用前前一对
// set/wait 已完全 drain，因为每个同步点本身是 barrier）。

constexpr uint16_t FLAG_AIV_SEG_RING[4] = {0, 1, 2, 3};   // AIV 侧段内/段边界 mode 0（全体 AIV）
constexpr uint16_t FLAG_AIC_SEG_RING[4] = {4, 5, 6, 7};   // AIC 侧段边界 mode 0（全体 AIC）
constexpr uint16_t FLAG_M2_RING[4] = {8, 9, 10, 11};      // 段内 mode 2 流式对（AIC↔配对 AIV）
constexpr uint16_t FLAG_RESERVED_HI = 12;                 // 12-15 保留

constexpr uint32_t NUM_SEG_SLOTS = 4;                  // 旋转槽位数

// CrossCore 模式（magic number 与 m0/m3 一致）
constexpr uint8_t CC_MODE0 = 0x0;   // 同类型 all-to-all（全体 AIV 或全体 AIC）
constexpr uint8_t CC_MODE2 = 0x2;   // AIC ↔ 其配对 2 个 AIV

// ============================================================
// 4. UB 静态布局（字节偏移；SEG-VEC 窗内各段同址叠放）
// ============================================================

constexpr uint32_t UB_BYTES_TOTAL = 248 * 1024;   // GetCoreMemSize(UB) 可用量（编译期上限）

// ---- PERSIST @0（§4：~12KB）----
constexpr uint32_t UB_PERSIST = 0;                     // 16KB 常量 staging（zeros/常量行）
constexpr uint32_t UB_PERSIST_BYTES = 16384;
constexpr uint32_t UB_ZEROS_B16 = UB_PERSIST + 0;                  // bf16[HIDDEN] 全 0（residual 占位）

// ---- gamma fp32 预转（§4 AIV 19-23 语义；放 SEG-VEC 之前）----
constexpr uint32_t UB_GAMMA_F32 = UB_PERSIST + UB_PERSIST_BYTES;   // @16384
constexpr uint32_t UB_GAMMA1_F32 = UB_GAMMA_F32;                   // fp32[HIDDEN]
constexpr uint32_t UB_GAMMA2_F32 = UB_GAMMA1_F32 + HIDDEN * 4;     // fp32[HIDDEN]

// ---- SEG-VEC 窗（§4 @16384 ~112KB；本实现 @36864 起，窗内 <64KB）----
constexpr uint32_t UB_VEC = UB_GAMMA2_F32 + HIDDEN * 4;            // @36864

// S1/S10  m6 残差 RMSNorm 行 scratch（m6 原布局平移进窗）
constexpr uint32_t UB_M6_X1 = UB_VEC + 0;                 // bf16[HIDDEN] 5120B
constexpr uint32_t UB_M6_RES = UB_M6_X1 + HIDDEN * 2;     // bf16[HIDDEN] 5120B（fp32 残差时占 10240B）
constexpr uint32_t UB_M6_XF = UB_M6_RES + HIDDEN * 4;     // fp32[HIDDEN] 10240B
constexpr uint32_t UB_M6_Y = UB_M6_XF + HIDDEN * 4;       // bf16[HIDDEN] 5120B
constexpr uint32_t UB_M6_TMP = UB_M6_Y + HIDDEN * 2;      // fp32[64]
constexpr uint32_t UB_M6_RED = UB_M6_TMP + 256;           // fp32[64]
constexpr uint32_t UB_M6_RSTD = UB_M6_RED + 256;          // fp32[64]

// S2  router（AIV0；m7 同款块结构：16 行一块 → x 块一次载入 → 逐行 V 计算写 staging）
constexpr uint32_t RT_RB = 16;                                   // row-block 行数
constexpr uint32_t UB_RT_WF = UB_VEC + 0;                        // fp32[NUM_EXPERTS+1][HIDDEN] = 51200B
constexpr uint32_t UB_RT_WB0 = UB_RT_WF + (NUM_EXPERTS + 1) * HIDDEN * 4;   // bf16[HIDDEN] 权重行 ping
constexpr uint32_t UB_RT_WB1 = UB_RT_WB0 + HIDDEN * 2;           // bf16[HIDDEN] pong
constexpr uint32_t UB_RT_XB = UB_RT_WB1 + HIDDEN * 2;            // bf16[RT_RB][HIDDEN] = 81920B（x 块）
constexpr uint32_t UB_RT_LOG = UB_RT_XB + RT_RB * HIDDEN * 2;    // fp32[RT_RB][64] = 4096B
constexpr uint32_t UB_RT_IDS = UB_RT_LOG + RT_RB * 64 * 4;       // i32[RT_RB][64]  = 4096B
constexpr uint32_t UB_RT_WS = UB_RT_IDS + RT_RB * 64 * 4;        // fp32[RT_RB][64] = 4096B
constexpr uint32_t UB_RT_SG = UB_RT_WS + RT_RB * 64 * 4;         // fp32[RT_RB]      = 64B
constexpr uint32_t UB_RT_VAL = UB_RT_SG + RT_RB * 4;             // fp32[32] 每行 e_i
constexpr uint32_t UB_RT_IDX = UB_RT_VAL + 128;                  // u32[32] 索引模板
constexpr uint32_t UB_RT_PAIR = UB_RT_IDX + 128;                 // fp32[64] Sort32 对
constexpr uint32_t UB_RT_OV = UB_RT_PAIR + 256;                  // fp32[64] Extract 值
constexpr uint32_t UB_RT_OI = UB_RT_OV + 256;                    // u32[64] Extract 索引
constexpr uint32_t UB_RT_END = UB_RT_OI + 256;

// S3  index generator（AIV0）：复用 router 的 x 块区（前一段已结束，barrier 分隔）
//     布局：counts[E] 与 offsets[E+1] 各占一个 32B 对齐槽（DataCopyPad 源必须 32B 对齐）
constexpr uint32_t UB_IG_SRC = UB_RT_XB;                         // i32[TOTAL_MAX] perm_src
constexpr uint32_t UB_IG_EXP = UB_IG_SRC + TOTAL_MAX * 4;        // i32[TOTAL_MAX] perm_expert
constexpr uint32_t UB_IG_CNT = UB_IG_EXP + TOTAL_MAX * 4;        // i32[E] @+0 / offsets[E+1] @+32 / 诊断 @+128
constexpr uint32_t UB_IG_INV = UB_IG_CNT + 256;                  // i32[M_MAX*TOPK_MAX] inv_slot
constexpr uint32_t UB_IG_WTK = UB_IG_INV + M_MAX * TOPK_MAX * 4; // i32[M_MAX*16] w_tk_packed
constexpr uint32_t UB_IG_END = UB_IG_WTK + M_MAX * 16 * 4;
static_assert(UB_IG_END <= UB_RT_XB + RT_RB * HIDDEN * 2, "IG 区超出 router x 块区");
static_assert(UB_RT_END <= UB_BYTES_TOTAL, "router UB footprint exceeds UB");

// S4  permute（m8#1，四级流水）
constexpr uint32_t UB_PM_ROW = UB_VEC + 0;                       // 4 × bf16[HIDDEN] = 20480B

// S5/S7  MXFP4 量化（m2/m5 全 VEC 路径；per-256-element tile）
constexpr uint32_t UB_QT_GATE = UB_VEC + 0;                      // bf16[TILE] 512B（同 m5 UB_X）
constexpr uint32_t UB_QT_UP = UB_QT_GATE + 512;                  // bf16[TILE] 512B
constexpr uint32_t UB_QT_SWIGLU = UB_QT_GATE + 1536;             // bf16[TILE] 512B（m5 UB_SWIGLU）
constexpr uint32_t UB_QT_QX = UB_QT_SWIGLU + 512;                // int8[TILE/2] 128B
constexpr uint32_t UB_QT_SCALE = UB_QT_QX + 128;                 // u16[TILE/GROUP/2] 8B
constexpr uint32_t UB_QT_HALF = UB_QT_SCALE + 32;                // u16[TILE/GROUP] 16B

// S9  unpermute（m8#2，双缓冲 stage）
constexpr uint32_t U_STAGE_B = (TOPK_MAX + 1) * HIDDEN * 2 + 64; // 每 stage 字节数
constexpr uint32_t UB_UP_STAGE0 = UB_VEC + 0;                    // stage0
constexpr uint32_t UB_UP_STAGE1 = UB_UP_STAGE0 + U_STAGE_B;      // stage1

// S9b combine（routed + sigmoid(sgate)*shared）
constexpr uint32_t UB_CB_ROUTED = UB_VEC + 0;                    // bf16[HIDDEN]
constexpr uint32_t UB_CB_SHARED = UB_CB_ROUTED + HIDDEN * 2;     // bf16[HIDDEN]
constexpr uint32_t UB_CB_OUT = UB_CB_SHARED + HIDDEN * 2;        // bf16[HIDDEN]
constexpr uint32_t UB_CB_G = UB_CB_OUT + HIDDEN * 2;             // fp32[64]（sigmoid 中间量）

// ============================================================
// 5. L1 / L0 静态布局（AIC；§4 A 区 @0、B 区 @262144、@330KB 起预取窗）
// ============================================================

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t BASE_M = 64;        // M 方向 tile（m ∈ [1,64] 单 tile，curM mask）
constexpr uint32_t BASE_K = 128;       // K 方向 tile
constexpr uint32_t BASE_N = 256;       // N 方向 tile

constexpr uint32_t L1_A_DATA = BASE_M * (BASE_K / 2);                      // 4KB（fp4 大包）
constexpr uint32_t L1_A_SCAL = BASE_M * (BASE_K / GROUP);                  // 256B
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_A_DATA;
constexpr uint32_t L1_OFF_AS0 = 2 * L1_A_DATA;
constexpr uint32_t L1_OFF_AS1 = L1_OFF_AS0 + L1_A_SCAL;
constexpr uint32_t L1_B_DATA = BASE_K * BASE_N / 2;                        // 16KB
constexpr uint32_t L1_B_SCAL = BASE_N * (BASE_K / GROUP);                  // 1KB
constexpr uint32_t L1_B_REGION = 256 * 1024;
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_DATA;
constexpr uint32_t L1_OFF_BS0 = L1_B_REGION + 2 * L1_B_DATA;
constexpr uint32_t L1_OFF_BS1 = L1_OFF_BS0 + L1_B_SCAL;
constexpr uint32_t L1_PREFETCH_BASE = 330 * 1024;   // §4 权重预取滚动窗边界（本 mission 未用）
constexpr uint32_t L1_BYTES_TOTAL = 512 * 1024;

constexpr uint32_t L0_PP_BYTES = 32 * 1024;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;

// ============================================================
// 6. GM workspace 偏移（一个连续 workspace 缓冲内，单位字节，32B 对齐）
// ============================================================
// 交叉核对表（tensor / 形状 / 生产者 → 消费者）见 README「GM 平面图」一节。

constexpr uint32_t WS_ALIGN = 32;
constexpr uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

constexpr uint32_t SZ_XNORM = AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_RES1 = AlignUp(M_MAX * HIDDEN * 4, WS_ALIGN);
constexpr uint32_t SZ_LOGITS = AlignUp(M_MAX * NUM_EXPERTS * 4, WS_ALIGN);
constexpr uint32_t SZ_IDS = AlignUp(M_MAX * TOPK_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_W = AlignUp(M_MAX * TOPK_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_PERM = AlignUp(TOTAL_MAX * 4, WS_ALIGN);
constexpr uint32_t SZ_COUNTS = AlignUp(NUM_EXPERTS * 4, WS_ALIGN);
// +32B：offsets[NUM_EXPERTS+1] 之后留一个 32B 对齐诊断槽（IG 越界计数）
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

}  // namespace M13

#endif  // M13_RESOURCES_H
