/**
 * m14_resources.h —— GDN layer kernel 全局静态资源表（docs/12-layer-integration.md §4）
 *
 * 单一全局编译期资源表（docs/05 §5.2）：BufferID / CrossCore flagId / UB 偏移 /
 * L1 偏移 / GM workspace 偏移 全部编译期静态常量；模块只声明 footprint，不自持有内存。
 * 所有地址单位为**字节**，且 32B 对齐（UB↔GM DataCopy 的硬要求，M9 quirk）。
 *
 * 与 docs/12 §4 的对应关系（逐条）：
 *   - AIC BufferID：0-1 = A ping/pong、2-3 = B ping/pong、4-5 = L0A/L0B ping/pong、
 *     6 = L0C（与 m1/m3/m11 编号一致）；7-13 保留（§4 第二组 GEMM ping/pong 与
 *     FIXP→UB 交接预留，本 mission 未用）。
 *   - AIV BufferID：0-1 = m6 行窗（S1/S7）、2-4 = m9 prolog（block ping/pong + gating）、
 *     8-14 = m4 递推专用（slab×2 / IV×2 / OV×2 / g-β，§4 原样）、15-17 = m12
 *     RMSNormGated（o/z 行、out 行、gamma 预转）、21-22 = gamma fp32 预转。
 *     §4 的 0-7 通用行窗含义在本实现里按「段内局部 id 全叠放到低编号」落实：
 *     GDN 段只有 m6 用 0-1，其余段各自持有互不重叠的 id（见 README 同步表）。
 *   - CrossCore flagId 旋转槽位制（相邻同步点必不同 id；每 id 每层用量 ≪ 15）：
 *     4-7 = **GDN 段边界**（docs/12 §4「4-7=GDN/其它段边界」）：S1 后 in_proj 输入就绪、
 *     in_proj 后、S5 后 out_proj 输入就绪、out_proj 后，四次 AIC↔AIV 交接各占一槽；
 *     8-10 = AIV 侧段内 mode-0 barrier（S1 后、S3→S4、S4→S5）；11 = AIV 侧 S5 后 barrier；
 *     12-15 = AIC 侧段内 mode-0 对齐（in_proj 前/后、out_proj 前/后，前与后各占不同槽）。
 *     **本 mission 实际占用 4-15 全部槽位，只剩 0-3 空**（0-3 留给 MoE/attention 段）。
 *   - UB：PERSIST @0（gamma bf16 staging + 三份 gamma fp32 预转，末址 @26112）；
 *     SEG-GDN 窗 @32768（§4 「SEG-GDN @16384 ~138KB」，本实现为留出 gamma 预转区
 *     挪到 @32768；窗内峰值 = m4 的 141312B = 138KB，与 §4 数值一致）。
 *     四个段窗 **互不重叠**（m6 / m9 / m12 / m4 依次顺排，见下方 UB_M6/UB_PL/UB_NG/UB_RC）：
 *     §4 允许的「同址叠放」以段间 barrier 分隔为前提，本实现选择不重叠以构造性消除
 *     m13 记录的两类跨 pipe UB 交接坑（同一 UB 区被不同 BufferID 保护会失效），
 *     代价是窗内余量从 ~100KB 降到 26624B（见 UB_RC_END 与 UB_BYTES_TOTAL）。
 *   - L1：A 区 @0 ×2 组、B 区 @262144 ×2 组（与 §4 一致，编号沿用 m11）；
 *     §4 的 @330KB 权重预取滚动窗本 mission 未实现（段间串行，正确性优先）。
 *   - GM workspace：13 个张量，偏移/尺寸全部编译期常量（SZ_ / WS_ 前缀），
 *     总量 WS_BYTES = 5258240B（~5.0MB，按 M_MAX=64 行上界分配）。
 */

#ifndef M14_RESOURCES_H
#define M14_RESOURCES_H

#include <cstdint>

namespace M14 {

// ============================================================
// 0. 形状常量（Qwen3.8-Flash-Next GDN 层，decode m=1；docs/10 §1）
// ============================================================

constexpr uint32_t HIDDEN = 2560;        // 层宽度（in_proj K / out_proj N / m6 norm 宽度）
constexpr uint32_t IN_N = 16480;         // in_proj 输出宽度 = q2048|k2048|v6144|z6144|b48|a48
constexpr uint32_t Q_DIM = 2048;         // q 段（16 key head × 128）
constexpr uint32_t K_DIM = 2048;         // k 段
constexpr uint32_t V_DIM = 6144;         // v 段 = 48 value head × 128（= GDN 层输出宽度）
constexpr uint32_t Z_DIM = 6144;         // z 段（output gate 输入，S2→S5 生命周期）
constexpr uint32_t GATE_DIM = 48;        // b / a 段（每 value head 一个标量）
constexpr uint32_t Q_OFF = 0;            // x_qkvzba 内 q 段起始元素
constexpr uint32_t K_OFF = Q_OFF + Q_DIM;
constexpr uint32_t V_OFF = K_OFF + K_DIM;
constexpr uint32_t Z_OFF = V_OFF + V_DIM;
constexpr uint32_t B_OFF = Z_OFF + Z_DIM;      // 16384
constexpr uint32_t A_OFF = B_OFF + GATE_DIM;   // 16432

constexpr uint32_t CH = 10240;           // conv1d 通道数 = q|k|v
constexpr uint32_t KW = 4;               // conv1d 宽度 K=4
constexpr uint32_t ST = 3;               // conv_state 历史行数（K−1）
constexpr uint32_t HEADS = 48;           // value head 数（NV）
constexpr uint32_t NK = 16;              // key head 数（q/k 段 2048/128）
constexpr uint32_t HEAD_D = 128;         // head 维度（DK = DV = 128）
constexpr uint32_t VGROUP = 3;           // NV = 3×NK：value head hv ← key head hv/3

constexpr uint32_t M_MAX = 64;           // GEMM 包络（BASE_M 单 tile；m6 行窗上界）
constexpr uint32_t CHAIN_M = 1;          // 层链只覆盖 decode m=1（m9/m4 为 m=1 核）

// in_proj / out_proj 两档真实形状
constexpr uint32_t IN_K = HIDDEN;        // in_proj K
constexpr uint32_t IN_OUT = IN_N;        // in_proj N
constexpr uint32_t OUT_K = V_DIM;        // out_proj K（6144）
constexpr uint32_t OUT_N = HIDDEN;       // out_proj N（2560）

static_assert(Q_DIM + K_DIM + V_DIM == CH, "conv 通道 = q|k|v");
static_assert(Q_DIM + K_DIM + V_DIM + Z_DIM + 2 * GATE_DIM == IN_N, "in_proj 输出宽度分解");
static_assert(HEADS * HEAD_D == V_DIM, "v 段 = HEADS × HEAD_D");
static_assert(NK * HEAD_D == Q_DIM, "q 段 = NK × HEAD_D");

// ============================================================
// 1. AIC 侧：tile 常量 / L1 布局 / BufferID（编号与 m1/m3/m11 一致）
// ============================================================

constexpr uint32_t CUBE_BLOCK = 16;      // cube 分形边长（bf16: 16 元素 = 32B）
constexpr uint32_t BASE_M = 64;          // M 方向 tile（m ∈ [1,64] 单 tile，curM mask）
constexpr uint32_t BASE_K = 64;          // K 方向 tile（in/out_proj 均整除）
constexpr uint32_t BASE_N = 160;         // N 方向 tile（gcd(16480, 2560) = 160）

constexpr uint32_t L1_A_ELEMS = BASE_M * BASE_K;   // A 大包 64×64 bf16 = 8KB
constexpr uint32_t L1_B_ELEMS = BASE_N * BASE_K;   // B 大包 160×64 bf16 = 20KB
constexpr uint32_t L1_OFF_A0 = 0;
constexpr uint32_t L1_OFF_A1 = L1_OFF_A0 + L1_A_ELEMS * 2;
constexpr uint32_t L1_B_REGION = 256 * 1024;       // §4：B 区 @262144
constexpr uint32_t L1_OFF_B0 = L1_B_REGION;
constexpr uint32_t L1_OFF_B1 = L1_B_REGION + L1_B_ELEMS * 2;
constexpr uint32_t L1_PREFETCH_BASE = 330 * 1024;  // §4 权重预取滚动窗边界（本 mission 未用）
constexpr uint32_t L1_BYTES_TOTAL = 512 * 1024;

constexpr uint32_t L0_PP_BYTES = 32 * 1024;
constexpr uint32_t L0_OFF_0 = 0;
constexpr uint32_t L0_OFF_1 = L0_PP_BYTES;

constexpr uint32_t BUF_AIC_A0 = 0;       // A1 ping: MTE2 → MTE1
constexpr uint32_t BUF_AIC_A1 = 1;       // A1 pong
constexpr uint32_t BUF_AIC_B0 = 2;       // B1 ping
constexpr uint32_t BUF_AIC_B1 = 3;       // B1 pong
constexpr uint32_t BUF_AIC_L00 = 4;      // L0A/L0B ping: MTE1 → M
constexpr uint32_t BUF_AIC_L01 = 5;      // L0A/L0B pong
constexpr uint32_t BUF_AIC_L0C = 6;      // L0C: M → FIXP
// 7-13 §4 预留（第二组 GEMM ping/pong + FIXP→UB 交接），本 mission 未用
constexpr uint32_t BUF_AIC_RESERVED_PREFETCH0 = 7;
constexpr uint32_t BUF_AIC_RESERVED_FIX2UB0 = 11;

static_assert(L1_B_REGION + 2 * L1_B_ELEMS * 2 <= L1_BYTES_TOTAL, "A/B ping-pong L1 footprint exceeds 512KB");
static_assert(BASE_M * BASE_K * 2 <= L0_PP_BYTES, "A2 tile exceeds L0A half");
static_assert(BASE_K * BASE_N * 2 <= L0_PP_BYTES, "B2 tile exceeds L0B half");

constexpr uint32_t IN_PROJ_NTILES = IN_OUT / BASE_N;    // 103
constexpr uint32_t OUT_PROJ_NTILES = OUT_N / BASE_N;    // 16
constexpr uint32_t IN_PROJ_KSTEPS = IN_K / BASE_K;      // 40
constexpr uint32_t OUT_PROJ_KSTEPS = OUT_K / BASE_K;    // 96
static_assert(IN_OUT % BASE_N == 0 && OUT_N % BASE_N == 0, "N 必须被 BASE_N 整除");
static_assert(IN_K % BASE_K == 0 && OUT_K % BASE_K == 0, "K 必须被 BASE_K 整除");

// ============================================================
// 2. AIV 侧 BufferID（每核用户可用 0..27，静态分配）
// ============================================================

constexpr uint32_t BUF_NORM_ROW = 0;     // m6 行输入: MTE2 → V（S1/S7 共用，段内行循环乒乓）
constexpr uint32_t BUF_NORM_OUT = 1;     // m6 行输出: V → MTE3
constexpr uint32_t BUF_PL_BK0 = 2;       // m9 prolog block 输入/输出 parity0: MTE2 → V → MTE3
constexpr uint32_t BUF_PL_BK1 = 3;       // m9 parity1
constexpr uint32_t BUF_PL_GB = 4;        // m9 gating 阵列: MTE2 → V
constexpr uint32_t BUF_RC_ST0 = 8;       // m4 state slab ping: MTE2 → V → MTE3（§4 8-14）
constexpr uint32_t BUF_RC_ST1 = 9;
constexpr uint32_t BUF_RC_IV0 = 10;      // m4 q/k/v ping
constexpr uint32_t BUF_RC_IV1 = 11;
constexpr uint32_t BUF_RC_OV0 = 12;      // m4 o
constexpr uint32_t BUF_RC_OV1 = 13;
constexpr uint32_t BUF_RC_GB = 14;       // m4 g/β 阵列
constexpr uint32_t BUF_NG_X = 15;        // m12 o/z head: MTE2 → V
constexpr uint32_t BUF_NG_OUT = 16;      // m12 out head: V → MTE3
constexpr uint32_t BUF_NG_G = 17;        // m12 gamma: MTE2 → V（每核一次）
constexpr uint32_t BUF_PRECAST1 = 21;    // gamma1/gamma2 bf16→fp32 预转（每核一次）
constexpr uint32_t BUF_PRECAST_G = 22;   // m12 gamma_g 预转

// ============================================================
// 3. CrossCore flagId（GDN 段用 4-7 槽；见文件头映射说明）
// ============================================================

constexpr uint16_t FLAG_BOUND_IN = 4;          // AIV→AIC：in_proj 输入（x_norm）就绪
constexpr uint16_t FLAG_BOUND_INPROJ = 5;      // AIC→AIV：in_proj 输出（qkvzba）就绪
constexpr uint16_t FLAG_BOUND_S5 = 6;          // AIV→AIC：RMSNormGated 输出（out_proj 输入）就绪
constexpr uint16_t FLAG_BOUND_OUTPROJ = 7;     // AIC→AIV：out_proj 输出就绪
constexpr uint16_t FLAG_AIV_SEG_S1 = 10;       // AIV 段内 mode-0 barrier（S1 后：x_norm 全体就位）
constexpr uint16_t FLAG_AIV_SEG2 = 11;         // AIV 段内 mode-0 barrier（S5 后：opin 全体就位）
constexpr uint16_t FLAG_AIV_SEG0 = 8;          // AIV 段内 mode-0 barrier（S3→S4：q/k/v/g/β 就位）
constexpr uint16_t FLAG_AIV_SEG1 = 9;          // AIV 段内 mode-0 barrier（S4→S5：o 就位）
constexpr uint16_t FLAG_AIC_SEG0A = 12;        // AIC in_proj 前对齐（mode-0）
constexpr uint16_t FLAG_AIC_SEG0B = 13;        // AIC in_proj 后 drain 对齐（mode-0）
constexpr uint16_t FLAG_AIC_SEG1A = 14;        // AIC out_proj 前对齐（mode-0）
constexpr uint16_t FLAG_AIC_SEG1B = 15;        // AIC out_proj 后 drain 对齐（mode-0）

constexpr uint8_t CC_MODE0 = 0x0;   // 同类型 all-to-all（全体 AIV 或全体 AIC）
constexpr uint8_t CC_MODE2 = 0x2;   // AIC ↔ 其配对 2 个 AIV

// ============================================================
// 4. UB 静态布局（字节偏移；四个段窗互不重叠，见下）
// ============================================================

constexpr uint32_t UB_BYTES_TOTAL = 248 * 1024;

// ---- PERSIST @0 ----
constexpr uint32_t UB_PERSIST = 0;                            // bf16[HIDDEN] staging（gamma 预转）
constexpr uint32_t UB_PERSIST_BYTES = HIDDEN * 2;             // 5120B
constexpr uint32_t UB_GAMMA1_F32 = UB_PERSIST + UB_PERSIST_BYTES;   // @5120, fp32[HIDDEN]
constexpr uint32_t UB_GAMMA2_F32 = UB_GAMMA1_F32 + HIDDEN * 4;      // @15360
constexpr uint32_t UB_GAMMA_G_F32 = UB_GAMMA2_F32 + HIDDEN * 4;     // @25600, fp32[HEAD_D] = 512B
constexpr uint32_t UB_SEG = 32768;                                  // 段窗起点（32KB，留 6.6KB 余量）
static_assert(UB_SEG >= UB_GAMMA_G_F32 + HEAD_D * 4, "UB_SEG 与 PERSIST 区重叠");
static_assert(UB_SEG % 512 == 0, "UB_SEG must be well aligned");

// ---- 四个互不重叠的段窗（顺排；不采用 docs/12 §4 的同址叠放，见文件头说明）----
constexpr uint32_t UB_M6 = UB_SEG;            // 段窗 1/4：m6 NormStage（需求 31488B）
constexpr uint32_t UB_PL = UB_M6 + 32768;     // 段窗 2/4：m9 prolog（需求 13312B）
constexpr uint32_t UB_NG = UB_PL + 16384;     // 段窗 3/4：m12 RMSNormGated（需求 2560B）
constexpr uint32_t UB_RC = UB_NG + 4096;      // 段窗 4/4：m4 递推（需求 141312B = docs/12 §4 的 138KB 峰值）

// ---- 段窗 1/4：m6 NormStage（S1/S7；行切分，m=1 时仅 AIV0 有活）----
constexpr uint32_t UB_M6_X1 = UB_M6 + 0;                      // bf16[HIDDEN] 5120B
constexpr uint32_t UB_M6_RES = UB_M6_X1 + HIDDEN * 2;         // bf16/fp32 残差（占 fp32 大小）10240B
constexpr uint32_t UB_M6_XF = UB_M6_RES + HIDDEN * 4;         // fp32[HIDDEN] 10240B（残差和；resOut 出口）
constexpr uint32_t UB_M6_Y = UB_M6_XF + HIDDEN * 4;           // bf16[HIDDEN] 5120B
constexpr uint32_t UB_M6_TMP = UB_M6_Y + HIDDEN * 2;          // fp32[64] 256B
constexpr uint32_t UB_M6_RED = UB_M6_TMP + 256;               // fp32[64]
constexpr uint32_t UB_M6_RSTD = UB_M6_RED + 256;              // fp32[64]
constexpr uint32_t UB_M6_END = UB_M6_RSTD + 256;

// ---- 段窗 2/4：m9 prolog（S3；block 粒度为 128ch，80 块条带划分）----
constexpr uint32_t BN = 128;                                  // 每 block 通道数（= HEAD_D）
constexpr uint32_t NBLK_PL = CH / BN;                         // 80
constexpr uint32_t PAR_B = 2816;                              // 每 parity block 区字节数
constexpr uint32_t UB_PL_BK0 = UB_PL + 0;
constexpr uint32_t UB_PL_BK1 = UB_PL_BK0 + PAR_B;
constexpr uint32_t UB_PL_AST = UB_PL_BK1 + PAR_B;             // a staging bf16[64]+slack 256B
constexpr uint32_t UB_PL_BST = UB_PL_AST + 256;               // b staging 256B
constexpr uint32_t UB_PL_AL = UB_PL_BST + 256;                // A_log fp32[64]+slack 512B
constexpr uint32_t UB_PL_DT = UB_PL_AL + 512;                 // dt_bias fp32[64]+slack 512B
constexpr uint32_t UB_PL_GS0 = UB_PL_DT + 512;                // g slot [48,8] fp32 1536B
constexpr uint32_t UB_PL_BS0 = UB_PL_GS0 + 1536;
constexpr uint32_t UB_PL_GS1 = UB_PL_BS0 + 1536;
constexpr uint32_t UB_PL_BS1 = UB_PL_GS1 + 1536;
constexpr uint32_t UB_PL_END = UB_PL_BS1 + 1536;

// ---- 段窗 4/4：m4 递推（S4；每 head 128×128 fp32 slab 64KB ×2 = §4 8-14 专用）----
constexpr uint32_t RC_ROWS = HEAD_D;
constexpr uint32_t RC_COLS = HEAD_D;
constexpr uint32_t RC_ST_ELEMS = RC_ROWS * RC_COLS;
constexpr uint32_t RC_ST_BYTES = RC_ST_ELEMS * 4;             // 64KB
constexpr uint32_t UB_RC_ST0 = UB_RC + 0;
constexpr uint32_t UB_RC_ST1 = UB_RC_ST0 + RC_ST_BYTES;       // @+64KB
constexpr uint32_t UB_RC_Q0 = UB_RC_ST1 + RC_ST_BYTES;        // @+128KB（每 parity q/k/v/o 各 512B）
constexpr uint32_t UB_RC_K0 = UB_RC_Q0 + 512;
constexpr uint32_t UB_RC_V0 = UB_RC_K0 + 512;
constexpr uint32_t UB_RC_O0 = UB_RC_V0 + 512;
constexpr uint32_t UB_RC_Q1 = UB_RC_O0 + 512;
constexpr uint32_t UB_RC_K1 = UB_RC_Q1 + 512;
constexpr uint32_t UB_RC_V1 = UB_RC_K1 + 512;
constexpr uint32_t UB_RC_O1 = UB_RC_V1 + 512;
constexpr uint32_t UB_RC_G = UB_RC_O1 + 512;                  // g [64][8] fp32（stride-8）
constexpr uint32_t UB_RC_EG = UB_RC_G + M_MAX * 32;           // e^g
constexpr uint32_t UB_RC_BETA = UB_RC_EG + M_MAX * 32;        // β
constexpr uint32_t UB_RC_END = UB_RC_BETA + M_MAX * 32;

// ---- 段窗 3/4：m12 RMSNormGated（S5；head 切分：每 AIV 一个 head）----
constexpr uint32_t UB_NG_O = UB_NG + 0;                       // fp32[128] 512B
constexpr uint32_t UB_NG_Z = UB_NG_O + HEAD_D * 4;            // bf16[128] 256B
constexpr uint32_t UB_NG_Y = UB_NG_Z + HEAD_D * 2;            // bf16[128] 256B
// 注：gamma 预转落在 PERSIST 区 UB_GAMMA_G_F32（kernel 开头一次性，48 head 共享），
//     故本段窗内不留 gamma 原始/预转 buffer。
constexpr uint32_t UB_NG_TMP = UB_NG_Y + HEAD_D * 2;          // fp32[64] 256B（reduce 二分 partials，备用）
constexpr uint32_t UB_NG_RED = UB_NG_TMP + 256;
constexpr uint32_t UB_NG_RSTD = UB_NG_RED + 256;
constexpr uint32_t UB_NG_END = UB_NG_RSTD + 256;

static_assert(UB_M6_END <= UB_PL, "m6 段窗与 m9 段窗重叠");
static_assert(UB_PL_END <= UB_NG, "m9 段窗与 m12 段窗重叠");
static_assert(UB_NG_END <= UB_RC, "m12 段窗与 m4 段窗重叠");
static_assert(UB_RC_END <= UB_BYTES_TOTAL, "段窗总占用超出 248KB UB");
static_assert(UB_RC_END <= UB_BYTES_TOTAL, "GDN 段窗 footprint exceeds UB");

// ============================================================
// 5. GM workspace 偏移（一个连续 workspace 缓冲内，单位字节，32B 对齐）
// ============================================================
// 张量生命周期（docs/12 §3 + mission「z 段在 S2→S5 间生命周期管理」）：
//   x_norm  : S1 产 → S2 消费（in_proj A）
//   res1    : S1 产 → S7 消费（m6#2 残差 fp32，小改 D）
//   qkvzba  : S2 产 → S3 消费 q|k|v|b|a、S5 消费 z 段（元素 [Z_OFF, Z_OFF+Z_DIM)）
//             —— z 段无独立 buffer，随 qkvzba 一起存活，S3/S4 不触碰该区间
//   q/k/v/g/β : S3 产 → S4 消费；o: S4 产 → S5 消费
//   opin    : S5 产 → S6 消费（out_proj A）；opout: S6 产 → S7 消费
//   y_final / res2 : S7 出口

constexpr uint32_t WS_ALIGN = 32;
constexpr uint32_t WS_AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }

constexpr uint32_t SZ_XNORM = WS_AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_RES1 = WS_AlignUp(M_MAX * HIDDEN * 4, WS_ALIGN);
constexpr uint32_t SZ_QKVZBA = WS_AlignUp(M_MAX * IN_N * 2, WS_ALIGN);
constexpr uint32_t SZ_Q = WS_AlignUp(NK * HEAD_D * 4, WS_ALIGN);
constexpr uint32_t SZ_K = SZ_Q;
constexpr uint32_t SZ_V = WS_AlignUp(HEADS * HEAD_D * 4, WS_ALIGN);
constexpr uint32_t SZ_G = WS_AlignUp(HEADS * 8 * 4, WS_ALIGN);
constexpr uint32_t SZ_BETA = SZ_G;
constexpr uint32_t SZ_O = WS_AlignUp(HEADS * HEAD_D * 4, WS_ALIGN);
constexpr uint32_t SZ_OPIN = WS_AlignUp(M_MAX * V_DIM * 2, WS_ALIGN);
constexpr uint32_t SZ_OPOUT = WS_AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_YFINAL = WS_AlignUp(M_MAX * HIDDEN * 2, WS_ALIGN);
constexpr uint32_t SZ_RES2 = WS_AlignUp(M_MAX * HIDDEN * 4, WS_ALIGN);

constexpr uint32_t WS_XNORM = 0;
constexpr uint32_t WS_RES1 = WS_XNORM + SZ_XNORM;
constexpr uint32_t WS_QKVZBA = WS_RES1 + SZ_RES1;
constexpr uint32_t WS_Q = WS_QKVZBA + SZ_QKVZBA;
constexpr uint32_t WS_K = WS_Q + SZ_Q;
constexpr uint32_t WS_V = WS_K + SZ_K;
constexpr uint32_t WS_G = WS_V + SZ_V;
constexpr uint32_t WS_BETA = WS_G + SZ_G;
constexpr uint32_t WS_O = WS_BETA + SZ_BETA;
constexpr uint32_t WS_OPIN = WS_O + SZ_O;
constexpr uint32_t WS_OPOUT = WS_OPIN + SZ_OPIN;
constexpr uint32_t WS_YFINAL = WS_OPOUT + SZ_OPOUT;
constexpr uint32_t WS_RES2 = WS_YFINAL + SZ_YFINAL;
constexpr uint32_t WS_BYTES = WS_RES2 + SZ_RES2;

}  // namespace M14

#endif  // M14_RESOURCES_H
