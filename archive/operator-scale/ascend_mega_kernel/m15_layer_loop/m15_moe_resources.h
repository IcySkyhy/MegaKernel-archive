/**
 * m15_moe_resources.h —— MoE 段（S1-S10）静态资源表，**融合进 m15 per-layer kernel** 后的版本
 *
 * 来源：m13_moe_layer/m13_resources.h（复制改造，逐行对照见 README「与 donor 的差异表」）。
 * 相对 m13 版的差异**有三处**（其余常量、UB/L1/GM 布局一字未动）：
 *   0. §6 的 6 项每专家张量尺寸由 padded `NUM_EXPERTS * M_MAX` 改为紧凑 Σt_e 上界 `TOTAL_MAX`
 *      （M91，= M77 §2 第 9 项）——**E=4/TOPK_MAX=4 下两者同为 256 行 ⇒ 数值 no-op**；
 *   1. namespace M13 → M15M；
 *   2. §3 的 CrossCore flagId 槽位重编：融合后 GDN 段（m15_gdn_resources.h §3，占 AIV m2 4-7 /
 *      AIV m0 8-11 / AIC m0 12-15 / AIC m2 4-7）与本段共存于**同一个 kernel**，故本段按
 *      「(核型, mode) 子空间与 GDN 段完全错开」重排 → 见 §3 的槽位表与相邻性核对。
 *
 * **UB 与 GDN 段窗的重叠是有意的、且是安全的**：本表 §4 的段窗（@36864 起，峰值
 * UB_RT_END = 222752）与 m15_gdn_resources.h §4 的四段窗（@32768 起，峰值 UB_RC_END =
 * 227328）在地址上重叠，但**两个段在本 kernel 内互斥执行**（中间有 PipeBarrier<PIPE_ALL> +
 * 全体 AIV mode-0 barrier，见 m15_layer_kernel.h），且没有任何一个张量跨相位存活：
 *   - GDN 段的 gamma1/gamma2/gammaG fp32 只在 GDN 相位内被读；
 *   - 本段的 gamma1/gamma2 fp32（§4 UB_GAMMA1_F32/UB_GAMMA2_F32，@16384/@20480）只在 MoE
 *     相位内被写与读 —— 它会覆盖 GDN 段的 gamma2_f32(@15360) 区，但那时 GDN 相位已结束。
 * 这就是 docs/12 §4 的「段窗同址叠放、以段间 barrier 为前提」范式；docs/17 §4「dump 非零且
 * 位置正确」与本 kernel 的 dump 判据共同见证它不是「看起来对」。
 *
 * 原始文件头（m13 的资源表说明，编号对应关系保留原文以便逐条回溯）：
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
 *     本实现窗内峰值占用见 `UB_RT_END = 222752 B`，**与 NUM_EXPERTS 无关**（M91-#3 把 router 的
 *     权重 fp32 窗从「NUM_EXPERTS+1 行常驻」改成「RT_EGRP 行流式窗 + 共享门 1 行」；M95-#4 又把
 *     top-k 的行宽/排序缓冲按**固定的** 16 块 + 归并树定尺 ⇒ `RT_ROWL`/`RT_SORTLN`/`RT_WROW` 皆为
 *     常量、**两档读数同值**，由 `moe_relift/check_stream.py` 的 B1 逐项核）；`222752 ≤ 227328`
 *     （GDN 相位峰值 UB_RC_END）⇒ 融合 UB 峰值不被抬高；各段仍按 barrier 分隔同址叠放，UB 总量
 *     248KB 内留有余量（不含 GDN 段）。**余量仅 4576 B**（对照 UB_RC_END）：后续若要在同一窗内
 *     再加张量，须先重算这里（可行方向见 `m95_README.md` §8）。
 *   - L1：A 区 @0 ×2 组、B 区 @262144 ×2 组（与 §4 一致）；@330KB 起为权重预取
 *     滚动窗（本 mission 未实现跨 op 预取，仅留出边界）。
 */

#ifndef M15_MOE_RESOURCES_H
#define M15_MOE_RESOURCES_H

#include <cstdint>

namespace M15M {

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
// [M91-#3] router 权重流式的两个 bf16 行缓冲 ping/pong 的 token（22/23 是 §2 已登记的
// 「MoE AIV 15..23」范围内原本未用的两个编号 ⇒ 峰值占用数 17 不变，无需改 m15_layer_resources.h）
constexpr uint32_t BUF_AIV_WST0 = 22;
constexpr uint32_t BUF_AIV_WST1 = 23;

// ============================================================
// 3. CrossCore flagId 旋转槽位
// ============================================================
// 相邻同步点必须不同 id（M2 实证：同核连续 set 不保序）；每 id 每层用量 ≪ 15。
// 每次「全体 AIV / 全体 AIC mode 0」按槽位轮转，槽位满 4 轮流复用（复用前前一对
// set/wait 已完全 drain，因为每个同步点本身是 barrier）。

// ---- M40（融合）槽位重编：按 (核型, mode) 与 GDN 段**完全错开**，无跨模式复用 ----
//
// m13 原表是「MoE 独占整层」的编号（AIV 0-3 / AIC 4-7 / M2 8-11），融合后必须与 GDN 段
// （m15_gdn_resources.h §3：AIV m0 = 8-11、AIV m2 = 4-7、AIC m0 = 12-15、AIC m2 = 4-7）
// 共存于同一个 kernel。下面按 **flagId 命名空间按 mode 分节登记**（M33/discs15 §5.3 ★8）重编：
//
//   | 段  | AIV mode-0      | AIC mode-0      | mode-2（AIC↔配对 AIV） |
//   |-----|-----------------|-----------------|------------------------|
//   | GDN | 8,9,10,11       | 12,13,14,15     | 4,5,6,7                |
//   | MoE | 12,13,14,15 (ring) + 8（相位边界，见表下注释） | 8,9,10,11 | 0,1,2,3 |
//
// 即 **两个段在每一个 (核型, mode) 子空间里占的都是互不相交的 id 集合**：GDM 段用不到的
// AIV 12-15 给 MoE 的 mode-0 ring、AIC 用不到的 8-11 给 MoE 的 mode-0 ring、两侧都用不到的
// 0-3 给 MoE 的 mode-2 ring。**全程不存在「同一 flagId 在同一核上跨模式复用」**（这是 m13
// 原实现自行遵守的保守规则，本 mission 继续遵守），因此不需要援引 docs/05 §2「同 flagId
// 跨模式复用须前一模式全部 drain」这条较弱的前提。
//
// 相邻性（docs/05 §2：同核相邻同步点必须用不同 id）在相位交界处逐条核对：
//   AIV:  GDN 最后一次自己发起的同步 = set(11, mode0) → MoE 相位边界 set(8, mode0)  ≠11 ✓
//         MoE 相位边界(8) → MoE S1 后 ring slot0(12)                                  ≠8  ✓
//   AIC:  GDN 最后一次自己发起的同步 = set(7, mode2) → MoE 第一次 = wait(0, mode2)     ≠7  ✓
//         GDN 最后一次 mode-0 = set(15) → MoE 第一次 mode-0 = set(8)                   ≠15 ✓
// MoE 段内 ring 复用与 m13 原实现一致（4 槽轮转，相邻必不同；每 id 每 kernel 用量 ≤2 ≪ 15）。

constexpr uint16_t FLAG_AIV_SEG_RING[4] = {12, 13, 14, 15};   // AIV 侧段内/段边界 mode 0（全体 AIV）
constexpr uint16_t FLAG_AIC_SEG_RING[4] = {8, 9, 10, 11};     // AIC 侧段边界 mode 0（全体 AIC）
constexpr uint16_t FLAG_M2_RING[4] = {0, 1, 2, 3};            // 段内 mode 2 流式对（AIC↔配对 AIV）
// 相位边界：GDN 段 → MoE 段的**全体 AIV mode-0 barrier**（见 m15_layer_kernel.h 的段序注释）。
// 它复用 GDN 段 S3→S4 barrier 的 id 8（同 mode、同核型）；该 id 的最后一次配对发生在本 kernel
// 的 GDN 相位内、且其后还隔着 GDN 的 S4→S5(9) / S5→S7(11) 两次 barrier —— 与 m13 段内
// 「4 槽轮转复用」是同一模式（本表自己的 ring 也在复用），故安全。
constexpr uint16_t FLAG_L0_BOUND_AIV = 8;
// 已被 GDN 段占用的 id（本段不得触碰）：AIV m2 = 4-7、AIC m0 = 12-15、AIC m2 = 4-7。
constexpr uint16_t FLAG_GDN_RESERVED = 4;                 // 4-7（两侧 mode-2，GDN 段）

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

// S2  router（AIV0；m7/m17 同款块结构：RT_RB 行一块 → x 块一次载入 → 逐行 V 计算写 staging）
//   **M91-#3 权重流式**：改前把 `[NUM_EXPERTS+1][HIDDEN]` 全部预转 fp32 常驻 UB（E=4 → 51,200 B；
//   E=512 → 5.25 MB，装不下）。现只保留 **RT_EGRP 行**的 fp32 窗常驻 + 2 个 bf16 行 ping/pong
//   （+ 共享门 1 行 bf16 + 1 行 fp32），UB 常驻量与 NUM_EXPERTS **解耦**。
constexpr uint32_t RT_RB = 8;                                    // row-block 行数（x 块；m17 同款）
constexpr uint32_t RT_EGRP = 8;                                  // 权重流式的“每专家组”行数
constexpr uint32_t UB_RT_WF = UB_VEC + 0;                        // fp32[RT_EGRP][HIDDEN] 权重行流式窗
constexpr uint32_t UB_RT_WB0 = UB_RT_WF + RT_EGRP * HIDDEN * 4;   // bf16[HIDDEN] 权重行 ping
constexpr uint32_t UB_RT_WB1 = UB_RT_WB0 + HIDDEN * 2;           // bf16[HIDDEN] pong
constexpr uint32_t UB_RT_SGB = UB_RT_WB1 + HIDDEN * 2;           // bf16[HIDDEN] 共享门权重行（独立槽）
constexpr uint32_t UB_RT_SGWF = UB_RT_SGB + HIDDEN * 2;          // fp32[HIDDEN] 共享门权重常驻
// ---- [M95-#4] top-k 归并树的**行宽与排序容量**（这是 §4 里唯一随 NUM_EXPERTS 定尺的一族）----
// 改前（M91 及以前）：行距写死 64 lane、`Sort32(..., 1)` 只排 32 对、`Reduce<MAX>` 只覆盖
// 32 lane、拆分只拆 32 对 ⇒ **正确性上界 NUM_EXPERTS ≤ 32**。现在按「每 32 个专家一块
// Sort32 + 二路 MrgSort 归并树」的形状定尺（形状来源 = `m17_moe_real/m17_moe_layer.asc`
// 的 `MergeTree`/`Merge2`，见 `moe_relift/m95_README.md`）。
// **为什么写成固定的 16 块 / 512 lane 而不随 E 缩**：① m17 的已验收形态就是硬编码的 16 块
// （E=512）；② E=4 下这棵树是**逐字节的退化档**（行内 lane ≥ E 全是 `RT_NEG_BIG` ⇒ exp = 0，
// 排序落在尾部，top-TOPK 与单块 Sort32 同集合同次序 —— 见 `m15_moe_layer.h` 的论证）；
// ③ 固定行宽让 `UB_RT_*` **整体与 E 解耦**（M91-#3 买到的性质不被 #4 退回去），UB 读数不随
// 档位漂移。代价是 E=4 下多排 15 次空归并（缩形档的性能，不是正确性）。
constexpr uint32_t RT_SORT_NBLK = 16;                        // Sort32 的 32 块个数（固定 16 块）
constexpr uint32_t RT_SORTLN = RT_SORT_NBLK * 32;            // 排序覆盖宽度（lane）= 16 × 32 = 512
constexpr uint32_t RT_ROWL = RT_SORTLN;                      // 对数行距（lane）
constexpr uint32_t RT_NCHUNK_W = RT_ROWL / 64;               // 行内 64-lane chunk 数 = 8
// top-k **结果**的 staging 行距：只放 ≤ 64 个 top-k 结果，**与 E 无关**（m17 亦如此：IDS 16 /
// WS 64）⇒ `UB_RT_IDS/WS` 不随 E 增长；`TOPK_MAX ≤ 64` 由下面的 static_assert 守住。
constexpr uint32_t RT_WROW = 64;
// Extract 的重复次数：目标 64 对 = 2×32 对（VF 内联的重复数）
constexpr uint32_t RT_EXTRACT_REP = 2;

constexpr uint32_t UB_RT_XB = UB_RT_SGWF + HIDDEN * 4;           // bf16[RT_RB][HIDDEN]（x 块）
constexpr uint32_t UB_RT_LOG = UB_RT_XB + RT_RB * HIDDEN * 2;    // fp32[RT_RB][RT_ROWL]（每行 E 个 logit + pad）
constexpr uint32_t UB_RT_IDS = UB_RT_LOG + RT_RB * RT_ROWL * 4;  // i32[RT_RB][RT_WROW]（top-k 结果 staging）
constexpr uint32_t UB_RT_WS = UB_RT_IDS + RT_RB * RT_WROW * 4;   // fp32[RT_RB][RT_WROW]
constexpr uint32_t UB_RT_SG = UB_RT_WS + RT_RB * RT_WROW * 4;    // fp32[RT_RB]
constexpr uint32_t UB_RT_VAL = UB_RT_SG + RT_RB * 4;             // fp32[RT_ROWL] 本行 e_i（Sort32 输入值）
constexpr uint32_t UB_RT_IDX = UB_RT_VAL + RT_ROWL * 4;          // u32[RT_ROWL] 索引模板（全局 expert id）
constexpr uint32_t UB_RT_PAIR = UB_RT_IDX + RT_ROWL * 4;         // fp32[2*RT_ROWL] Sort32 输出（每块 32 对）
constexpr uint32_t UB_RT_MA = UB_RT_PAIR + 2 * RT_ROWL * 4;      // fp32[2*RT_ROWL] 归并树乒乓 A
constexpr uint32_t UB_RT_MB = UB_RT_MA + 2 * RT_ROWL * 4;        // fp32[2*RT_ROWL] 归并树乒乓 B
constexpr uint32_t UB_RT_OV = UB_RT_MB + 2 * RT_ROWL * 4;        // fp32[64] Extract 前 64 对的值
constexpr uint32_t UB_RT_OI = UB_RT_OV + 256;                    // u32[64] Extract 前 64 对的索引
constexpr uint32_t UB_RT_END = UB_RT_OI + 256;
// 行宽/排序宽度必须覆盖**全部** E 个 logit（否则 top-k 候选集被截断）；归并树的根前 32 对
// = 全局 top-32（逐级归纳，证明见 `m15_moe_layer.h` 的 `SoftmaxTopkRow` 头注）⇒ top-k 上界 32。
static_assert(RT_ROWL == RT_SORTLN, "行距与排序覆盖宽度必须一致（Sort32 按 32 lane 连续取块）");
static_assert(RT_ROWL % 64 == 0, "行距必须是 64 lane（一个 fp32 chunk）的整数倍");
static_assert(RT_ROWL >= NUM_EXPERTS, "对数行距 < NUM_EXPERTS：装不下全部专家 logit");
static_assert(RT_SORTLN >= NUM_EXPERTS, "排序覆盖宽度 < NUM_EXPERTS：top-k 候选集会被截断");
static_assert(TOPK_MAX <= 32, "top-k 上界超过归并树根的 top-32 保证");
static_assert(TOPK_MAX <= RT_WROW, "top-k 结果放不进 IDS/WS 的 staging 行距");
// 注：`UB_RT_*` **全部与 NUM_EXPERTS 无关**（`RT_ROWL`/`RT_SORTLN` 是固定 16 块⇒512 lane，
// `RT_RB`/`RT_EGRP`/`RT_WROW` 也是常量）—— M91-#3 买到的「权重窗与 E 解耦」与 M95-#4 买到的
// 「行宽覆盖 E ≤ 512」在读数上都不随档位漂移，由 `moe_relift/check_stream.py` 逐项核。
static_assert(RT_RB * HIDDEN * 2 >= 2 * TOTAL_MAX * 4 + 256 + M_MAX * TOPK_MAX * 4 + M_MAX * 16 * 4,
              "IG 区放不进 router x 块区（RT_RB 太小）");

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
// router 必须每行吐出 **E 个 logit** ⇒ 本项**随 E 增长、不能缩**（E=4 → 1,024 B；E=512 → 131,072 B，
// 与 M77 §3.2 的参考数一致）。它是 top-k 排序的候选集来源；M95-#4 之后本段 top-k 的**覆盖宽度**
// 与它对齐（`RT_SORTLN ≥ NUM_EXPERTS`，见 §4 的 `RT_ROWL`/`RT_SORT_NBLK` 与三条 static_assert）。
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
// ---- M91 (§2 第 9 项)：每专家张量的行数 = **紧凑 Σt_e 上界** `TOTAL_MAX = M_MAX*TOPK_MAX` ----
// 原实现按 padded「每专家固定 M_MAX 行」定尺（`NUM_EXPERTS * M_MAX`）。链上的专家槽**已经是紧凑
// Σt_e**（M40 的硬约束：槽起点 = S3 产出的 `expert_offsets[]` 前缀和，`rowBase = offGm.GetValue(slot)
// + mt*BASE_M`）⇒ 这些缓冲只需 `Σt_e ≤ M_MAX*TOPK_MAX` 行，padded 定尺是 41× 的浪费，且与紧凑
// 寻址的声明不自洽。Σt_e ≤ TOTAL_MAX 的依据：每个 token 恰好贡献 topk 个槽、top-k id 互异 ⇒
// `Σ_e t_e = m*topk ≤ M_MAX*TOPK_MAX`。**E=4/TOPK_MAX=4 下 `NUM_EXPERTS*M_MAX == TOTAL_MAX == 256`
// ⇒ 本节 6 项在当前档数值不变（no-op）**；真实档 E=512/topk=10 下 padded 32768 行 → 紧凑 640 行。
// 共享专家的 `*_SHD` 项不受影响（它恒为 m ≤ M_MAX 行，与 E 无关）。
constexpr uint32_t SZ_AQ = AlignUp(TOTAL_MAX * (HIDDEN / 2), WS_ALIGN);
constexpr uint32_t SZ_AS = AlignUp(TOTAL_MAX * GU_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_AQ_SHD = AlignUp(M_MAX * (HIDDEN / 2), WS_ALIGN);
constexpr uint32_t SZ_AS_SHD = AlignUp(M_MAX * GU_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_GU = AlignUp(TOTAL_MAX * GU_N * 2, WS_ALIGN);
constexpr uint32_t SZ_GU_SHD = AlignUp(M_MAX * GU_N * 2, WS_ALIGN);
constexpr uint32_t SZ_H = AlignUp(TOTAL_MAX * INTER * 2, WS_ALIGN);
constexpr uint32_t SZ_H_SHD = AlignUp(M_MAX * INTER * 2, WS_ALIGN);
constexpr uint32_t SZ_HQ = AlignUp(TOTAL_MAX * (INTER / 2), WS_ALIGN);
constexpr uint32_t SZ_HS = AlignUp(TOTAL_MAX * DN_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_HQ_SHD = AlignUp(M_MAX * (INTER / 2), WS_ALIGN);
constexpr uint32_t SZ_HS_SHD = AlignUp(M_MAX * DN_SCALE_STRIDE, WS_ALIGN);
constexpr uint32_t SZ_Y = AlignUp(TOTAL_MAX * HIDDEN * 2, WS_ALIGN);
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

}  // namespace M15M

#endif  // M15_MOE_RESOURCES_H
