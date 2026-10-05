/**
 * m22_resources.h —— M22 真实规模 MoE router（512 experts / top-10）全局静态资源表
 *
 * 设计约束（docs/05 §2/§5.2）：BufferID / UB 地址偏移全部编译期静态常量，
 * 不使用 TPipe / TBuf / TQue / TBufPool / AllocTensor；基础 API 一律用裸指针或
 * LocalTensor(pos, byteOffset, elemCount) 取址；同步只用 BufferID（禁 set/wait flag
 * 表达跨 pipe 数据可见性）。
 *
 * 与 m7_router_topk 的差异（逐条见 README「移植边界」表）：
 *   - RB 16 -> 8：为 M40 契约输出段（counts/base/group_list/perm）腾出 UB 余量；
 *   - 新增 BUF_RT：路由计数段的「PIPE_S 标量写 UB -> PIPE_MTE3 出 GM」握手；
 *   - 新增 UB_CNT/UB_BASE/UB_CUR/UB_SCL/UB_G64 五块（契约输出段专用）；
 *   - MERGE_MODE 编译期可选 2 路（m7 现行）/ 4 路归并树。
 *
 * UB 预算（RB=8）：182848 B = 178.6 KB，可用 248KB（docs/05 §6）→ 余量 69.4KB。
 * 逐块明细见文件末尾 UB_BUDGET 表。
 */

#ifndef M22_RESOURCES_H
#define M22_RESOURCES_H

#include <cstdint>

#ifndef M22_MERGE_MODE
#define M22_MERGE_MODE 2
#endif

namespace M22 {

// ============================================================
// 0. 形状常量（真实规模：Qwen3.8-Flash-Next-MXFP4 第 0 层 mlp.gate.weight）
// ============================================================

constexpr uint32_t HIDDEN = 2560;   // router K = hidden_size
constexpr uint32_t E = 512;         // 专家数（真实 checkpoint [512,2560]）
constexpr uint32_t KTOPK = 10;      // top-10（num_experts_per_tok）

// ============================================================
// 1. AIV 向量参数
// ============================================================

constexpr uint32_t VLF = 64;                    // 256B 向量寄存器 fp32 lane 数（SIMD VL 固定 256B）
constexpr uint32_t KCHUNK = 64;                 // GEMV K 分步：UNPACK 载入 + Cast = 64 连续 fp32
constexpr uint32_t NCHUNK = HIDDEN / KCHUNK;    // 40 步
constexpr uint32_t EGRP = 8;                    // GEMV 专家分组（共享 x 载入；8 点积拼成一个寄存器写）
constexpr uint32_t SORT_BLK = E / 32;           // 16 个 32 对有序块（Sort32 一趟）
constexpr uint32_t RB = 8;                      // row-block 行数（m7 为 16）
constexpr uint32_t MERGE_MODE = M22_MERGE_MODE; // 2 = m7 现行 2 路树；4 = 4 路树

constexpr uint32_t UB_BYTES_TOTAL = 248 * 1024; // AIV UB 可用（docs/05 §6：256KB 物理 / 248KB 可用）

// ============================================================
// 2. BufferID（每核用户 0-27；AIV 与 AIC 空间独立）
// ============================================================

constexpr uint32_t BUF_X = 0;     // x row-block:   MTE2 -> V
constexpr uint32_t BUF_W0 = 1;    // w 行 ping:     MTE2 -> V
constexpr uint32_t BUF_W1 = 2;    // w 行 pong:     MTE2 -> V
constexpr uint32_t BUF_LOG = 3;   // logits/scores: V -> MTE3
constexpr uint32_t BUF_TOP = 4;   // ids/weights:   V -> MTE3，阻塞释放后再由 PIPE_S 取（值依赖）
constexpr uint32_t BUF_RT = 5;    // 计数段 staging: PIPE_S 标量写 UB -> MTE3 出 GM

// ============================================================
// 3. UB 静态布局（字节偏移，全部 32B 对齐）
// ============================================================

// ---- GEMV / topk 段（与 m7 同构，仅 RB 改变）----
constexpr uint32_t UB_XB = 0;                             // bf16 [RB][HIDDEN]       40960
constexpr uint32_t UB_WB = UB_XB + RB * HIDDEN * 2;       // bf16 [2][HIDDEN] ping   10240
constexpr uint32_t UB_WF = UB_WB + 2 * HIDDEN * 2;        // fp32 [EGRP][HIDDEN]     81920
constexpr uint32_t UB_LOG = UB_WF + EGRP * HIDDEN * 4;    // fp32 [RB][E]            16384
constexpr uint32_t UB_VAL = UB_LOG + RB * E * 4;          // fp32 [E] 行内 e_i        2048
constexpr uint32_t UB_IDX = UB_VAL + E * 4;               // u32  [E] arange 模板     2048
constexpr uint32_t UB_TMP = UB_IDX + E * 4;               // fp32 [2E] Sort32 输出    4096
constexpr uint32_t UB_MA = UB_TMP + 2 * E * 4;            // fp32 [2E] 归并缓冲 A     4096
constexpr uint32_t UB_MB = UB_MA + 2 * E * 4;             // fp32 [2E] 归并缓冲 B     4096
constexpr uint32_t UB_OUTV = UB_MB + 2 * E * 4;           // fp32 [64] Extract 值      256
constexpr uint32_t UB_OUTI = UB_OUTV + 64 * 4;            // u32  [64] Extract 索引    256
constexpr uint32_t UB_IDS = UB_OUTI + 64 * 4;             // i32  [RB][128] 行 staging 4096
constexpr uint32_t UB_WS = UB_IDS + RB * 128 * 4;         // bf16 [RB][128] 行 staging 2048

// ---- 契约输出段（M40 MoE 段输入契约）----
constexpr uint32_t UB_CNT = UB_WS + RB * 128 * 2;         // i32 [E] 每专家 token 计数（count 模式）
constexpr uint32_t UB_BASE = UB_CNT + E * 4;              // i32 [E] 独占前缀和（紧凑槽起点）
constexpr uint32_t UB_CUR = UB_BASE + E * 4;              // i32 [E] 落位游标（UB_BASE 的副本，写回时自增）
constexpr uint32_t UB_SCL = UB_CUR + E * 4;               // i32 [16] 标量槽：active_num / n_active
constexpr uint32_t UB_G64 = UB_SCL + 64;                  // i64 [E] count 模式 group_list（官方编码投影）
constexpr uint32_t UB_END = UB_G64 + E * 8;

static_assert(UB_END <= UB_BYTES_TOTAL, "M22 UB footprint exceeds 248KB");

// ============================================================
// 4. UB 预算表（逐块，供 README / reviewer 复算）
// ============================================================
//
//  #  块       字节    说明
//  0  UB_XB   40960   x row-block 常驻（RB=8 行 bf16）
//  1  UB_WB   10240   w 行 ping-pong（每行 5120B bf16，掩盖 MTE2）
//  2  UB_WF   81920   8 专家行 fp32 预转窗（**唯一**的权重解包工作集）
//  3  UB_LOG  16384   logits（逐行 max-shift 后就地覆盖）
//  4  UB_VAL   2048   行内 e_i = exp(l - max)
//  5  UB_IDX   2048   arange 索引模板（512，建一次）
//  6  UB_TMP   4096   Sort32 输出（16 块 × 32 对）
//  7  UB_MA    4096   归并缓冲 A
//  8  UB_MB    4096   归并缓冲 B
//  9  UB_OUTV   256   Extract 值（64 对）
// 10  UB_OUTI   256   Extract 索引（64 对）
// 11  UB_IDS   4096   ids 行 staging（每行 128 槽位，仅前 10 有效）
// 12  UB_WS    2048   weights 行 staging（bf16 位打包）
// 13  UB_CNT   2048   每专家计数（count 模式 group_list 的载荷）
// 14  UB_BASE  2048   紧凑槽起点（独占前缀和）
// 15  UB_CUR   2048   落位游标
// 16  UB_SCL     64   active_num / n_active（2 × i32，32B 对齐槽）
// 17  UB_G64   4096   i64[512] 官方 count 模式编码投影
//     ----------------
//     UB_END 182848 = 178.6 KB  / 248KB 可用 → 余量 69.4 KB
//
// 对比 m13 的路由器布局（UB_RT_WF = (NUM_EXPERTS+1)*HIDDEN*4）：512 专家下
// (512+1)*2560*4 = 5.25MB ≫ 248KB，**必须**按行 staging；本表的 #2 即该问题的解
// （固定 8 行窗口 = 80KB，与专家总数无关）。

}  // namespace M22

#endif  // M22_RESOURCES_H
