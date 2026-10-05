// ============================================================
// m15_attn_core.h —— **机械生成物，勿手改**（M101）
// ============================================================
// 由 `m15_layer_loop/evidence/attn_core/lift_attn_core_segment.py` 从**只读 donor**
// `m10_attn_decode/m10_attn_decode.asc` 的 device 段抽出，做下表列的 8 类机械替换 **+ 第 9 类整块替换**。
// `--check` 重新抽取并与本文件逐字节比 ⇒ donor 一改，`--check` 立刻红（不存在静默漂移）。
//
//   · 内容锚点：donor asc 行 58 .. 1005（顶格 `namespace m10 {` .. 顶格 `}  // namespace m10`）
//   · donor 文件 sha256 = e68354eb408b1d91900020cea173f00f9a437e8f1582048f7532bc477b8de4ba
//   · 抽取形态：**include guard + 只含 inline body + 无 `main()` + 无 `#include`**
//     （donor 的 device 段本来就没有 `#include`；3 对 `#if/#else/#endif` 是调试开关）
//
// ---- 8 类机械替换（生成器逐类计数断言；重命名式，无算法改动）----
//   1. `namespace m10 {`           → `namespace M15AC {`
//   2. `}  // namespace m10`       → `}  // namespace M15AC`
//   3. `M10Aic`                    → `AttnCoreAic`
//   4. `M10Aiv`                    → `AttnCoreAiv`
//   5. `M10_DBG_STAGE0`            → `M15AC_DBG_STAGE0`
//   6. `M10_DEBUG_SKIP_AIC_FD`     → `M15AC_DEBUG_SKIP_AIC_FD`
//   7. `M10_DEBUG_SKIP_COMBINE`    → `M15AC_DEBUG_SKIP_COMBINE`
//   8. printf 标签 `[M10K]`         → `[M15AC]`（纯文案，不动数值/判据/同步）
//
// ---- 第 9 类：**整块替换**（**抽取物不再是"逐字 + 纯重命名"**，塔 2026-10-04 裁决）----
//   把 `AttnCoreAiv::Combine()` 的 FD 归并内循环里两处**经典 memory-based 向量 API** 换成 RegBase VF：
//     改前：`Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`（donor asc:977-978 同款）
//     改后：`AccFmaRowVf(numUb, accUb, w)` = `__VEC_SCOPE__` 内的 `LoadAlign + Muls + Add + StoreAlign`
//     依据：人类逐字规则「……都应该都用 VEC_SCOPE aka simd vector function，**不应该使用 memory base 的 API**」
//     **等价性**：`Mul`/`Add`/`Muls` 在官方 Reg 精度表里都是 **0 ulp**，且 `nsplit` 累加次序一字未动
//     ⇒ 预期逐位不变；由「与 `m10_attn_decode/data` 归档 dump **逐字节**对拍 + donor `check_ref.py`
//     三档」见证（读数见 `evidence/attn_core/README.md` §3.2/§3.4）。**若对拍不成立即停手报塔。**
//
// ============================================================
// 段接口（供 M102 在顶层接线与**打平分配**；人类要求"暴露依赖哪些 pipeline、输出哪些 pipeline、
// 需要哪些 buffer id / cross-core id"）
// ============================================================
// 入口：`M15AC::AttnCoreBody(q, k, v, out, seq, wsAcc, wsM, wsS, maskCol, dbgS, dbgP, dbgC, cfg, dbgC2, gP)`
//   调度形态 `__mix__(1, 2)`；**blockDim = AIC 数**，本段内部假定 28 个 unit（14 split × 2 KV 头，
//   `n2 = bid & 1`、`split = bid >> 1`）。AIV 数 = 2 × AIC 数。
//   形状（全 bf16）：Q `[2][16][256]`（行 0..11 为 12 个 q 头，12..15 pad）、
//   K/V `[2][seqPad][256]`、out `[2][12][256]`；`seq` 是 GM 上的运行期标量（`seqGM.GetValue(0)`）。
//
// **依赖（消费）的管道** —— 语义 = "谁必须先跑完":
//   · 跨段: **无**。段起点是自包含的：调用方只需把 q/k/v/seq/maskCol/workspace/dbg 备好并对全体
//     核可见（前一个相位边界已给）。本段不 wait 任何段外 flag。
//   · 段内: AIC ↔ AIV 经 cross-core（下表），核内经 BufferID（下表）。
// **输出（生产）的管道**:
//   · GM `out`（`[2][12][256]` bf16，由 `Combine()` 按行写出 —— FD 归并的最终结果）。
//   · GM workspace `wsAcc/wsM/wsS`（未归一 partial；`WritePartialsAndSignal()` 写）。
//   · GM 调试面 `dbgS/dbgP/dbgC/dbgC2`（**首 tile 无条件写**，见下"调试面"）。
//   · 跨段: **无 set**。段尾由调用方补 `PipeBarrier<PIPE_ALL>`（与 `m15_attn_passthrough_body` 同款）。
//
// **cross-core flagId：35 次调用，本段段内独占；融合后必须由顶层按 `m15_layer_resources.h` §4 重号。**
//   两条必须重号的理由（M102 逐条核）：
//   ① **mode-2 的 9/10/11 与 hc 段 mode-2 的 8/9/10/11 撞号**：同一 (核型, mode) 子空间内不许有
//      同一 id 的两个同步点。本段这三个号在顶层应取"已 drain 的空洞"或另分节。
//   ② **mode-4 不是"新子空间"**（复审 r1-F2 更正；官方口径 = 本机 CANN 文档
//      `asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md`
//      的「flagId取值范围说明」）：模式 0/1/2 每核 16 个（0-15）；**模式 4 时 AIV 仍是同一批 0-15
//      （池没有变宽），只有 AIC 侧放宽到 0-31**，且「AIC 发起 flagId 16-31 ↔ AIV1 的 wait 0-15」。
//      同一核上同一 flagId 跨模式复用是**合法**的，前提是"模式切换前前一个 mode 的所有
//      set/wait 都已执行完（drain）"。⇒ 本段的 AIV mode-4 号 {0,1,5,6,7} 与顶层 AIV 上
//      mode0+mode2 已占满的 0-15 **是同一个物理池**，打平表必须按"drain 后复用"登记；
//      本段的 AIC mode-4 高号（`ccMM + AIV_CH` = 16/17）落在 AIC 放宽后的 0-31 里 ——
//      **顶层登记表必须为 `AIC ∧ mode==4` 单独放行到 32**，否则 16/17 会被 `FLAG_PER_CORE = 16`
//      判成越界。截至本提交，主线 `m15_layer_resources.h` 已落 `FlagIdLimit(core, mode)` 与
//      `FLAG_PER_CORE_AIC_MODE4 = 32`（以及 `ATTN_CORE_M4_AIV_MAX` / `ATTN_CORE_M4_AIC_HI_LO`
//      两个见证量）—— 以该文件为准，本行只记"本段需要什么"。
//
// **资源足迹（人类口径"资源分配要全部打平考虑"；复审 r1-F5 要求写进本头）**：
//   · L1：段内 `L1_OFF_P0/P1/P2`(3×8KB) + `L1_OFF_Q`(8KB) + `L1_OFF_KV0/KV1`(2×128KB)
//     ⇒ 末端 = `L1_OFF_KV1 + L1_KV_BYTES` = 32768 + 131072 + 131072 = **294,912 B = 288 KB / 512 KB**。
//   · L0A 64KB（Q 8KB 常驻 + P 8KB 轮转）、L0B 64KB 单 buffer 轮转、L0C 4×16KB 环 = 64KB / 256KB。
//   · AIV UB：段内自报的两个峰值常数 `UB_TOTAL` = 50,208 B、`UB_TOTAL2` = 64,512 B（combine 路径）
//     ⇒ 峰值按 **64,512 B ≈ 63 KB / 248 KB** 记账。
//   · GM：`out` 12,288 B、workspace `wsAcc/wsM/wsS` = 458,752 + 1,792 + 1,792 B、
//     调试面 `dbgC/dbgC2/dbgS/dbgP` = 458,752+458,752+458,752+229,376 B、P 中转 `gP` 688,128 B。
//   ⇒ 以上都是**段内自管理**的编译期常量（没有用 AscendC 的资源管理函数）；融合后与其它段
//     同址叠放时要按相位互斥重核（L1/L0C/UB 的峰值取 max 不是求和）。
//
// **prefill 边界（复审 r1-F5 要求写进本头）**：本段是 **decode 形状** —— `blockDim` = AIC 数、
//   段内假定 28 个 unit（`SPLITS=14` × 2 KV 头）、`n2 = bid & 1`、`split = bid >> 1`、
//   KV tile = `S2T=256` token。**它不能直接吃 prefill `m=4097`**：AIC 的 N-块分派与 AIV 的按头
//   分派都要 per-row 重排（与 M97 §5 第 7 项同一件事，本 mission 未做，见
//   `evidence/attn_core/README.md` §5 第 3 项）。
//   段内常量（由生成器实测 donor 得出）：
//   · CC_MM0      = 0
//   · CC_MM1      = 1
//   · CC_P0       = 5
//   · CC_P1       = 6
//   · CC_P2       = 7
//   · CC_BAR      = 8
//   · CC_AIVDONE  = 9
//   · CC_RDY      = 10
//   · CC_ALLDONE  = 11
//   调用分布（生成器实测）：
//   · Set  mode 0x0 PIPE_FIX  × 2
//   · Set  mode 0x2 PIPE_FIX  × 3
//   · Set  mode 0x2 PIPE_MTE3 × 5
//   · Set  mode 0x4 PIPE_FIX  × 4
//   · Set  mode 0x4 PIPE_MTE3 × 1
//   · Set  mode 0x4 PIPE_V    × 4
//   · Wait mode 0x0 PIPE_S    × 2
//   · Wait mode 0x2 PIPE_MTE2 × 2
//   · Wait mode 0x2 PIPE_S    × 4
//   · Wait mode 0x4 PIPE_FIX  × 4
//   · Wait mode 0x4 PIPE_MTE2 × 2
//   · Wait mode 0x4 PIPE_V    × 2
//
// **MutexID（核内局部编号）**：融合后必须与同核其它段打平重编；本段不假定全局编号。
//   段内常量：
//   · B_Q     = 0
//   · B_KV0   = 1
//   · B_KV1   = 2
//   · B_L0    = 3
//   · B_C0    = 4
//   · B_C1    = 5
//   · B_C2    = 6
//   · B_C3    = 7
//   · B_PL1   = 8
//   · B_PC    = 0
//   · B_ACC   = 1
//   · B_MSK   = 2
//   · B_CIN   = 3
//   · B_COUT  = 4
//   · B_ACCIN = 5
//   · B_DBG   = 6
//   调用分布（生成器实测）：
//   · Acq M         B_KV0   × 1
//   · Acq M         B_KV1   × 1
//   · Acq M         B_L0    × 2
//   · Acq MTE1      B_KV0   × 1
//   · Acq MTE1      B_KV1   × 1
//   · Acq MTE1      B_L0    × 2
//   · Acq MTE1      B_PL1   × 1
//   · Acq MTE1      B_Q     × 1
//   · Acq MTE2      B_ACCIN × 1
//   · Acq MTE2      B_CIN   × 1
//   · Acq MTE2      B_KV0   × 2
//   · Acq MTE2      B_KV1   × 1
//   · Acq MTE2      B_MSK   × 1
//   · Acq MTE2      B_PL1   × 1
//   · Acq MTE2      B_Q     × 1
//   · Acq MTE3      B_ACC   × 1
//   · Acq MTE3      B_COUT  × 1
//   · Acq MTE3      B_DBG   × 1
//   · Acq MTE3      B_PC    × 1
//   · Acq V         B_ACC   × 1
//   · Acq V         B_ACCIN × 1
//   · Acq V         B_CIN   × 1
//   · Acq V         B_COUT  × 1
//   · Acq V         B_DBG   × 1
//   · Acq V         B_MSK   × 1
//   · Acq V         B_PC    × 1
//   · Rls M         B_KV0   × 1
//   · Rls M         B_KV1   × 1
//   · Rls M         B_L0    × 2
//   · Rls MTE1      B_KV0   × 1
//   · Rls MTE1      B_KV1   × 1
//   · Rls MTE1      B_L0    × 2
//   · Rls MTE1      B_PL1   × 1
//   · Rls MTE1      B_Q     × 1
//   · Rls MTE2      B_ACCIN × 1
//   · Rls MTE2      B_CIN   × 1
//   · Rls MTE2      B_KV0   × 2
//   · Rls MTE2      B_KV1   × 1
//   · Rls MTE2      B_MSK   × 1
//   · Rls MTE2      B_PL1   × 1
//   · Rls MTE2      B_Q     × 1
//   · Rls MTE3      B_ACC   × 1
//   · Rls MTE3      B_COUT  × 1
//   · Rls MTE3      B_DBG   × 1
//   · Rls MTE3      B_PC    × 1
//   · Rls V         B_ACC   × 1
//   · Rls V         B_ACCIN × 1
//   · Rls V         B_CIN   × 1
//   · Rls V         B_COUT  × 1
//   · Rls V         B_DBG   × 1
//   · Rls V         B_PC    × 1
//
// 量测面（生成器实测）：`__VEC_SCOPE__` × 8、`Duplicate(` × 5
//   （`Duplicate` 是 UB 常量填充，不在计算链上 —— donor README §7 的存量说明）。
//
// **调试面（M102 接线注意）**：`dbgC`（`[28][16][256]` fp32 = 458,752 B）、`dbgC2`（同尺寸）、
//   `dbgS`（`[56][8][256]` fp32 = 458,752 B）、`dbgP`（`[56][8][256]` bf16 = 229,376 B）
//   是**首 tile 无条件写**的（donor 的定位遗留）。融合形态若要省掉这 4 次 dump 与
//   `printf`，需要改段内代码 —— 本 mission 按"机械抽取"口径**原样保留**，逐条记在
//   `evidence/attn_core/README.md` §5「未完成项」。
//
// ---- 包含前提 ----
// 本文件**不自带任何 include**（不引入系统头/AscendC 头）；依赖 include 它的 TU 已经包含
// `kernel_operator.h`（AscendC 基础 API + `AscendC::Reg`）。donor 的 device 段亦如此。
// ============================================================
#ifndef M15_ATTN_CORE_H
#define M15_ATTN_CORE_H

namespace M15AC {
using namespace AscendC;

// ---------------- 形状 / schedule 常量 ----------------
constexpr uint32_t GQA_N2  = 2;          // KV 头数
constexpr uint32_t GQA_G   = 12;         // q 头 / KV 头
constexpr uint32_t M_PAD   = 16;         // gS1-merge 后 cube M（12 pad 16）
constexpr uint32_t HD      = 256;        // head_dim
constexpr uint32_t S2T     = 256;        // KV seq tile（config5 sInner256）
constexpr uint32_t SPLITS  = 14;         // FD split 上限（28 AIC / 2 KV 头）
constexpr uint32_t AIV_ROWS = M_PAD / 2; // 每个 AIV 经 dualDstCtl 分得 8 行
constexpr float    SCALE   = 0.0625f;    // 256^-0.5 = 1/16
constexpr float    MASKV   = -3.38953139e38f;  // 列 mask 屏蔽值（exp 下溢为 0）

__aicore__ __inline__ constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
__aicore__ __inline__ constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }
__aicore__ __inline__ constexpr uint32_t MinU(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---------------- BufferID（阻塞释放封装；docs/05 M1 精炼）----------------
template <pipe_t p> __aicore__ __inline__ void Acq(MutexID id) { GetBuffImpl<p, false>(id); }
template <pipe_t p> __aicore__ __inline__ void Rls(MutexID id) { ReleaseBuffImpl<p, false>(id); }

// ---------------- CrossCore flagId ----------------
constexpr uint16_t CC_MM0 = 0;   // mode4 mmRes buf0（AIC 侧 +16 → AIV1 通道）
constexpr uint16_t CC_MM1 = 1;   // mode4 mmRes buf1
constexpr uint16_t CC_P0  = 5;   // mode4 P L1 buf0
constexpr uint16_t CC_P1  = 6;
constexpr uint16_t CC_P2  = 7;
constexpr uint16_t CC_BAR = 8;   // mode0 全体 AIC barrier
constexpr uint16_t CC_AIVDONE = 9;  // mode2 AIV ws 写完 -> AIC（2 set 配 1 wait）
constexpr uint16_t CC_RDY     = 10; // mode2 AIC 放行 combine -> AIV（1 set 配 2 wait）
constexpr uint16_t CC_ALLDONE = 11; // mode2 AIV 收尾 -> AIC（2 set 配 1 wait）
constexpr uint32_t AIV_CH = 16;     // mode 4 下 AIC 侧 AIV1 通道偏移

// ---------------- AIC L1/L0 静态布局（字节）----------------
constexpr uint32_t L1_P_BYTES  = M_PAD * S2T * 2;    // 8KB
constexpr uint32_t L1_OFF_P0   = 0;
constexpr uint32_t L1_OFF_P1   = L1_P_BYTES;
constexpr uint32_t L1_OFF_P2   = 2 * L1_P_BYTES;
constexpr uint32_t L1_OFF_Q    = 3 * L1_P_BYTES;     // 8KB [16,256] NZ
constexpr uint32_t L1_KV_BYTES = S2T * HD * 2;       // 128KB
constexpr uint32_t L1_OFF_KV0  = L1_OFF_Q + M_PAD * HD * 2;  // 32KB
constexpr uint32_t L1_OFF_KV1  = L1_OFF_KV0 + L1_KV_BYTES;   // 160KB
static_assert(L1_OFF_KV1 + L1_KV_BYTES <= 512 * 1024, "L1 overflow");

constexpr uint32_t L0A_BYTES   = 64 * 1024;
constexpr uint32_t L0A_OFF_Q   = 0;                  // Q [16,256] bf16 常驻
constexpr uint32_t L0A_OFF_P   = M_PAD * HD * 2;     // 8KB：每 tile P [16,256]
constexpr uint32_t L0B_BYTES   = 64 * 1024;
constexpr uint32_t L0C_BYTES   = 256 * 1024;
constexpr uint32_t L0C_SLOT    = M_PAD * S2T * 4;    // 16KB fp32 [16,256]
static_assert(4 * L0C_SLOT <= L0C_BYTES, "L0C overflow");

// AIC BufferID
constexpr MutexID B_Q   = 0;   // Q L1（MTE2->MTE1）
constexpr MutexID B_KV0 = 1;   // KV0 L1（K tile；MTE2->MTE1->M 全生命周期）
constexpr MutexID B_KV1 = 2;   // KV1 L1（V tile）
constexpr MutexID B_L0  = 3;   // L0A 的 MTE1->M 交接（Q 常驻 + P 轮转）
constexpr MutexID B_C0  = 4;   // L0C slot0..3（M->FIX）
constexpr MutexID B_C1  = 5;
constexpr MutexID B_C2  = 6;
constexpr MutexID B_C3  = 7;
constexpr MutexID B_PL1 = 8;   // AIC：P 在 GM->L1 的跨 pipe 交接（MTE2 生产 -> MTE1 消费）

// ---------------- AIV UB 静态布局（字节）----------------
constexpr uint32_t UB_MM0   = 0;                    // [8,256] fp32 8KB（mmRes ping）
constexpr uint32_t UB_MM1   = 8192;                 // 8KB
constexpr uint32_t UB_PC    = 16384;                // [8,256] bf16 4KB（P cast staging）
constexpr uint32_t UB_ACC   = 20480;                // [8,256] fp32 8KB
constexpr uint32_t UB_MASK  = 28672;                // [8,256] fp32 8KB（tail 列 mask 驻留）
constexpr uint32_t UB_MB    = 36864;                // [8,256] fp32 8KB（RegBase 化后空出：改作 S 调试 scratch）
constexpr uint32_t UB_MASKZ = 53248;                // [8,256] fp32 8KB：全 0 mask（非尾 tile 用；见 docs/05 §6.1）
constexpr uint32_t UB_MVEC  = 45056;                // [8] fp32
constexpr uint32_t UB_SVEC  = 45088;                // [8] fp32
constexpr uint32_t UB_TVEC  = 45152;                // [8] fp32 scratch（expdiff）
constexpr uint32_t UB_WMAT  = 45216;                // [8,16] fp32 512B（combine 权重）
constexpr uint32_t UB_MMAT  = 45728;                // [14,8] fp32 448B（combine m）
constexpr uint32_t UB_SMAT  = 46176;                // [14,8] fp32 448B（combine sum）
constexpr uint32_t UB_ACCR  = 46624;                // [256] fp32 1KB（combine 行加载）
constexpr uint32_t UB_TMPR  = 47648;                // [256] fp32 1KB
constexpr uint32_t UB_NUMR  = 48672;                // [256] fp32 1KB
constexpr uint32_t UB_OUTR  = 49696;                // [256] bf16 512B
constexpr uint32_t UB_CVF_M = 61440;                // combine：每 split 的行 m（VF 整表计算）
constexpr uint32_t UB_CVF_S = 62464;                // combine：每 split 的行 sum
constexpr uint32_t UB_CVF_W = 63488;                // combine：权重 w=exp(m-M)
constexpr uint32_t UB_TOTAL2 = 63488 + 1024;        // ≈63KB « 248KB
constexpr uint32_t UB_TOTAL = 50208;                // ~49KB « 248KB

// AIV BufferID（独立命名空间）
constexpr MutexID B_PC   = 0;  // pCast staging：V(cast) -> MTE3(P->L1)
constexpr MutexID B_ACC  = 1;  // acc：V(FlashUpdate) -> MTE3(ws 写)
constexpr MutexID B_MSK  = 2;  // tail 列 mask：MTE2(GM) -> V(softmax)
constexpr MutexID B_CIN  = 3;  // combine：MTE2(ws m/sum 读) -> scalar 读
constexpr MutexID B_COUT = 4;  // combine：V(cast) -> MTE3(out 写)
constexpr MutexID B_ACCIN = 5; // combine：MTE2(ws acc 读) -> V（同一 token 不可同时被两 pipe 持有）
constexpr MutexID B_DBG  = 6;  // 调试 dump：V(搬 S 到 scratch) -> MTE3（跨 pipe 排水交接）

// ---------------- ND2NZ GM->L1（bf16 行主序 -> NZ）----------------
template <typename T>
__aicore__ __inline__ void GmToL1Nz(const LocalTensor<uint8_t>& dstL1, const GlobalTensor<T>& srcGm,
                                    uint32_t gmElemOff, uint32_t rows, uint32_t cols, uint32_t gmRowPitch)
{
    Nd2NzParams par;
    par.ndNum = 1;
    par.nValue = rows;
    par.dValue = cols;
    par.srcNdMatrixStride = 0;
    par.srcDValue = gmRowPitch;
    par.dstNzC0Stride = rows;
    par.dstNzNStride = 1;
    par.dstNzMatrixStride = 0;
    DataCopy(dstL1.template ReinterpretCast<T>(), srcGm[gmElemOff], par);
}

// ---------------- L1->L0 2D 装载（m11 已实证的字段约定 + 调试可覆盖）----------------
template <typename T>
__aicore__ __inline__ void LoadL0_2D(const LocalTensor<uint8_t>& dstL0, const LocalTensor<uint8_t>& srcL1,
                                     uint32_t l1ElemOff, uint32_t mStep, uint32_t kStep, uint32_t srcStride,
                                     uint32_t dstStride, bool ifTranspose = false, uint32_t dstElemOff = 0)
{
    LocalTensor<T> src = srcL1.template ReinterpretCast<T>()[l1ElemOff];
    LocalTensor<T> dst = dstL0.template ReinterpretCast<T>()[dstElemOff];
    LoadData2DParamsV2 lp;
    lp.mStartPosition = 0;
    lp.kStartPosition = 0;
    lp.mStep = (uint16_t)mStep;
    lp.kStep = (uint16_t)kStep;
    lp.srcStride = (int32_t)srcStride;
    lp.dstStride = (uint16_t)dstStride;
    lp.ifTranspose = ifTranspose;
    lp.sid = 0;
    LoadData(dst, src, lp);
}

// ---------------- V^T 3D 转置装载（donor LoadDataToL0B KN 分支同款）----------------
template <typename T>
__aicore__ __inline__ void LoadL0B_Transpose(const LocalTensor<uint8_t>& dstL0, const LocalTensor<uint8_t>& srcL1,
                                             uint32_t l1ElemOff, uint32_t kSize, uint32_t nSize)
{
    LoadData3DParamsV2<T> p;
    p.l1H = kSize / 16;      // 源 height（16 行一组）
    p.l1W = 16;
    p.padList[0] = 0;
    p.padList[1] = 0;
    p.padList[2] = 0;
    p.padList[3] = 255;
    p.mExtension = kSize;    // 目的 height 传输长度
    p.kExtension = nSize;    // 目的 width
    p.mStartPt = 0;
    p.kStartPt = 0;
    p.strideW = 1;
    p.strideH = 1;
    p.filterW = 1;
    p.filterSizeW = false;
    p.filterH = 1;
    p.filterSizeH = false;
    p.dilationFilterW = 1;
    p.dilationFilterH = 1;
    p.enTranspose = 1;
    p.fMatrixCtrl = 0;
    p.channelSize = nSize;
    LocalTensor<T> src = srcL1.template ReinterpretCast<T>()[l1ElemOff];
    LocalTensor<T> dst = dstL0.template ReinterpretCast<T>();
    static constexpr IsResetLoad3dConfig kCfg(true, true);   // donor LOAD3DV2_CONFIG：isSetFMatrix/isSetPadding
    LoadData<T, kCfg>(dst, src, p);
}

// ---------------- FIXP L0C-->UB（dualDstCtl=1，M 拆两半写两个 AIV）----------------
__aicore__ __inline__ void FixpToUb(const LocalTensor<uint8_t>& dstUb, const LocalTensor<uint8_t>& srcL0C,
                                    uint32_t nSize, uint32_t dstElemOff, uint32_t dstStride)
{
    // M16/M19 加固：FixpipeParamsArch3510 的 reluScalar/vectorRelu/deqScalar 三个成员无 NSDMI，
    // 必须显式清零，否则 FIXP 拿到栈上垃圾（症状：mmRes 数值明显不随输入变化/量级失控）。
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
    fp.nSize = (uint16_t)nSize;
    fp.mSize = M_PAD;                 // 16（偶数，dualDstCtl 前提）
    fp.srcStride = M_PAD;
    fp.dstStride = dstStride;
    fp.dualDstCtl = 1;
    fp.reluScalar = 0;
    fp.vectorRelu = 0;
    fp.deqScalar = 0;
    fp.params.ndNum = 1;
    fp.params.srcNdStride = 0;
    fp.params.dstNdStride = 0;
    LocalTensor<float> src = srcL0C.template ReinterpretCast<float>();
    LocalTensor<float> dst = dstUb.template ReinterpretCast<float>()[dstElemOff];
    static constexpr FixpipeConfig kFixUb(CO2Layout::ROW_MAJOR, true);   // donor FIXPIPE_ROW_MAJOR_UB
    Fixpipe<float, float, kFixUb>(dst, src, fp);
}

// ---------------- FIXP L0C-->GM（调试归档：绕过 UB 路径直接落盘 L0C）----------------
__aicore__ __inline__ void FixpToGmDbg(GlobalTensor<float> dstGm, const LocalTensor<uint8_t>& srcL0C,
                                       uint32_t nSize, uint32_t dstStride)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
    fp.nSize = (uint16_t)nSize;
    fp.mSize = M_PAD;
    fp.srcStride = M_PAD;
    fp.dstStride = dstStride;
    fp.reluScalar = 0;
    fp.vectorRelu = 0;
    fp.deqScalar = 0;
    static constexpr FixpipeConfig kFixGm(CO2Layout::ROW_MAJOR, false);
    Fixpipe<float, float, kFixGm>(dstGm, srcL0C.template ReinterpretCast<float>(), fp);
}

// ---------------- mmad（bf16，C=A·B^T，B 为 [N,K] NZ）----------------
__aicore__ __inline__ void MmBf16(const LocalTensor<uint8_t>& cL0C, const LocalTensor<uint8_t>& aL0,
                                  const LocalTensor<uint8_t>& bL0, uint32_t aElemOff, uint32_t m, uint32_t n,
                                  uint32_t k, bool init)
{
    MmadParams mp;
    mp.m = (uint16_t)m;
    mp.n = (uint16_t)n;
    mp.k = (uint16_t)k;
    mp.cmatrixInitVal = init;
    mp.cmatrixSource = false;
    Mmad(cL0C.template ReinterpretCast<float>(), aL0.template ReinterpretCast<bfloat16_t>()[aElemOff],
         bL0.template ReinterpretCast<bfloat16_t>(), mp);
}

// ============================================================
// AIC
// ============================================================
class AttnCoreAic {
public:
    __aicore__ inline AttnCoreAic() {}

    __aicore__ inline void Init(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v, __gm__ uint8_t* out,
                                __gm__ uint32_t* seq, __gm__ uint8_t* wsAcc, __gm__ uint8_t* wsM, __gm__ uint8_t* wsS,
                                __gm__ uint8_t* dbgC, __gm__ uint32_t* cfg, __gm__ uint8_t* dbgC2,
                                __gm__ uint8_t* gP)
    {
        qGM.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        kGM.SetGlobalBuffer((__gm__ bfloat16_t*)k);
        vGM.SetGlobalBuffer((__gm__ bfloat16_t*)v);
        seqGM.SetGlobalBuffer(seq);
        dbgCGM.SetGlobalBuffer((__gm__ float*)dbgC);
        dbgC2GM.SetGlobalBuffer((__gm__ float*)dbgC2);
        pGmGM.SetGlobalBuffer((__gm__ bfloat16_t*)gP);
        cfgGM.SetGlobalBuffer(cfg);
        (void)out; (void)wsAcc; (void)wsM; (void)wsS;
    }

    __aicore__ inline void Run()
    {
        const uint32_t bid = GetBlockIdx();
        const uint32_t seq = seqGM.GetValue(0);
        const uint32_t seqPad = AlignUp(seq, S2T);
        const uint32_t nTiles = seqPad / S2T;
        const uint32_t chunkTiles = CeilDiv(nTiles, SPLITS);
        const uint32_t nsplit = CeilDiv(nTiles, chunkTiles);
        const uint32_t n2 = bid & 1u;
        const uint32_t split = bid >> 1;
        const bool hasWork = (bid < 2 * nsplit);
        uint32_t tiles = 0;
        if (hasWork) {
            tiles = MinU(chunkTiles, nTiles - split * chunkTiles);
        }

#if defined(M15AC_DBG_STAGE0)
        // 阶段 0：只验证 FD 收尾同步骨架（无 cube / 无 DMA / 无 printf 密集）
        CrossCoreWaitFlag<0x2, PIPE_S>(CC_AIVDONE);
        CrossCoreSetFlag<0x0, PIPE_FIX>(CC_BAR);
        CrossCoreWaitFlag<0x0, PIPE_S>(CC_BAR);
        CrossCoreSetFlag<0x2, PIPE_FIX>(CC_RDY);
        CrossCoreWaitFlag<0x2, PIPE_S>(CC_ALLDONE);
        (void)hasWork; (void)tiles;
#else
        if (hasWork) {
            printf("[M15AC] AIC %u enter tiles=%u\n", bid, tiles);
            RunTiles(n2, split, tiles, chunkTiles, seqPad);
            printf("[M15AC] AIC %u tile loop done\n", bid);
        }

        // ---- FD 收尾同步（全体 AIC 参与 mode 0 barrier）----
        printf("[M15AC] AIC %u reach FD sync\n", bid);
#ifdef M15AC_DEBUG_SKIP_AIC_FD
        // 调试：跳过 AIC 侧的 FD 等待，让 kernel 能跑到结尾以便 flush 全部 printf 与 dump
        CrossCoreSetFlag<0x2, PIPE_FIX>(CC_RDY);
#else
        CrossCoreWaitFlag<0x2, PIPE_S>(CC_AIVDONE);   // 配对 2 AIV 的 partial 已落 GM（链式 wait->set 挂 S）
        CrossCoreSetFlag<0x0, PIPE_FIX>(CC_BAR);      // 全体 AIC 对齐
        CrossCoreWaitFlag<0x0, PIPE_S>(CC_BAR);
        CrossCoreSetFlag<0x2, PIPE_FIX>(CC_RDY);      // 放行配对 AIV 做 combine
        CrossCoreWaitFlag<0x2, PIPE_S>(CC_ALLDONE);   // 等 AIV 收尾
#endif
#endif
    }

private:
    __aicore__ inline LocalTensor<uint8_t> UbMmRes(uint32_t parity)
    {
        return LocalTensor<uint8_t>(TPosition::VECIN, parity ? UB_MM1 : UB_MM0, 8192);
    }

    __aicore__ inline MutexID L0CId(uint32_t slot)
    {
        switch (slot & 3) {
            case 0: return B_C0;
            case 1: return B_C1;
            case 2: return B_C2;
            default: return B_C3;
        }
    }

    __aicore__ inline LocalTensor<uint8_t> Slot(LocalTensor<uint8_t>& l0C, MutexID id)
    {
        uint32_t off = 0;
        switch (id) {
            case B_C1: off = L0C_SLOT; break;
            case B_C2: off = 2 * L0C_SLOT; break;
            case B_C3: off = 3 * L0C_SLOT; break;
            default: off = 0; break;
        }
        return l0C.template ReinterpretCast<uint8_t>()[off];
    }

    // BMM1：S(t)=Q·K^T（K=256 拆 2x128 kLoop），FIXP 直写 mmRes[t&1]
    __aicore__ inline void RunBmm1(uint32_t t, LocalTensor<uint8_t>& l0A, LocalTensor<uint8_t>& l0B,
                                   LocalTensor<uint8_t>& l0C, LocalTensor<uint8_t>& l1KV0, MutexID bC)
    {
        const uint32_t parity = t & 1;
        const uint16_t ccMM = parity ? CC_MM1 : CC_MM0;
        // 等 AIV 把 mmRes[parity] 的上一轮占用释放（t<2 时由 AIV init-set 放行）
        CrossCoreWaitFlag<0x4, PIPE_FIX>(ccMM);
        CrossCoreWaitFlag<0x4, PIPE_FIX>(ccMM + AIV_CH);
        for (uint32_t ks = 0; ks < 2; ++ks) {
            const uint32_t kOff = ks * (128 / 16) * S2T * 16;   // K L1 NZ 视图内 128 列半块偏移（元素）
            Acq<PIPE_MTE1>(B_KV0);
            if (cfgGM.GetValue(6) != 0) {
                LoadL0_2D<bfloat16_t>(l0B, l1KV0, kOff, cfgGM.GetValue(7), cfgGM.GetValue(8), cfgGM.GetValue(9),
                                      cfgGM.GetValue(10), cfgGM.GetValue(11) != 0);
            } else {
                LoadL0_2D<bfloat16_t>(l0B, l1KV0, kOff, S2T / 16, 128 / 16, S2T / 16, S2T / 16);
            }
            Rls<PIPE_MTE1>(B_KV0);
            Acq<PIPE_M>(B_L0);
            Acq<PIPE_M>(B_KV0);
            if (ks == 0) { Acq<PIPE_M>(bC); }   // L0C slot 由 M 侧持有（写），FIX 侧 get 后才可见
            MmBf16(Slot(l0C, bC), l0A, l0B, ks * (M_PAD * 128), M_PAD, S2T, 128, ks == 0);
            Rls<PIPE_M>(B_KV0);
            Rls<PIPE_M>(B_L0);
        }
        Rls<PIPE_M>(bC);   // drain：L0C 累加结果就绪（跨 pipe 交接给 FIX）
        Acq<PIPE_FIX>(bC);
        if (t == 0) {
            // 调试归档：t=0 的 L0C（16x256 fp32）经 Fixpipe 直写 GM（绕过 UB，判别 mmad 本身是否出数）
            FixpToGmDbg(dbgCGM[(uint64_t)GetBlockIdx() * M_PAD * S2T], Slot(l0C, bC), S2T, S2T);
        }
        FixpToUb(UbMmRes(parity), Slot(l0C, bC), S2T, 0, S2T);
        Rls<PIPE_FIX>(bC);
        CrossCoreSetFlag<0x4, PIPE_FIX>(ccMM);        // S(t) 就绪（两通道）
        CrossCoreSetFlag<0x4, PIPE_FIX>(ccMM + AIV_CH);
    }

    __aicore__ inline void RunTiles(uint32_t n2, uint32_t split, uint32_t tiles, uint32_t chunkTiles, uint32_t seqPad)
    {
        const uint32_t kvElemPitch = seqPad * HD;   // K/V GM 每头元素跨度
        const uint32_t qBase = n2 * M_PAD * HD;     // Q GM 头基址（元素）
        const uint32_t kvBase = n2 * kvElemPitch;
        const uint32_t s2Base0 = split * chunkTiles * S2T;

        LocalTensor<uint8_t> l1Q(TPosition::A1, L1_OFF_Q, M_PAD * HD * 2);
        LocalTensor<uint8_t> l1KV0(TPosition::B1, L1_OFF_KV0, L1_KV_BYTES);
        LocalTensor<uint8_t> l1KV1(TPosition::B1, L1_OFF_KV1, L1_KV_BYTES);
        LocalTensor<uint8_t> l0A(TPosition::A2, 0, L0A_BYTES);
        LocalTensor<uint8_t> l0B(TPosition::B2, 0, L0B_BYTES);
        LocalTensor<uint8_t> l0C(TPosition::CO1, 0, L0C_BYTES);
        LocalTensor<uint8_t> l1P[3] = {LocalTensor<uint8_t>(TPosition::A1, L1_OFF_P0, L1_P_BYTES),
                                       LocalTensor<uint8_t>(TPosition::A1, L1_OFF_P1, L1_P_BYTES),
                                       LocalTensor<uint8_t>(TPosition::A1, L1_OFF_P2, L1_P_BYTES)};

        // prologue：Q 常驻 L1 + L0A
        Acq<PIPE_MTE2>(B_Q);
        GmToL1Nz<bfloat16_t>(l1Q, qGM, qBase, M_PAD, HD, HD);
        Rls<PIPE_MTE2>(B_Q);
        Acq<PIPE_MTE1>(B_L0);
        Acq<PIPE_MTE1>(B_Q);   // 跨 pipe 交接：MTE2 阻塞释放后 MTE1 才可读 l1Q（m11 同款）
        {
            const uint32_t c0 = cfgGM.GetValue(0);
            if (c0 != 0) {
                LoadL0_2D<bfloat16_t>(l0A, l1Q, 0, cfgGM.GetValue(1), cfgGM.GetValue(2), cfgGM.GetValue(3),
                                      cfgGM.GetValue(4), cfgGM.GetValue(5) != 0);
            } else {
                LoadL0_2D<bfloat16_t>(l0A, l1Q, 0, M_PAD / 16, HD / 16, M_PAD / 16, M_PAD / 16);  // Q [16,256]
            }
        }
        Rls<PIPE_MTE1>(B_Q);
        Rls<PIPE_MTE1>(B_L0);

        // prologue：BMM1(t=0)
        Acq<PIPE_MTE2>(B_KV0);
        GmToL1Nz<bfloat16_t>(l1KV0, kGM, kvBase + s2Base0 * HD, S2T, HD, HD);
        Rls<PIPE_MTE2>(B_KV0);
        RunBmm1(0, l0A, l0B, l0C, l1KV0, L0CId(0));

        // 主循环：stage1 = BMM1(t+1)，stage2 = BMM2(t)
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t s2Base = (split * chunkTiles + t) * S2T;
            if (t + 1 < tiles) {
                const uint32_t nb = s2Base + S2T;
                Acq<PIPE_MTE2>(B_KV0);   // 等 BMM1(t) 的 M 侧释放 KV0
                GmToL1Nz<bfloat16_t>(l1KV0, kGM, kvBase + nb * HD, S2T, HD, HD);
                Rls<PIPE_MTE2>(B_KV0);
                RunBmm1(t + 1, l0A, l0B, l0C, l1KV0, L0CId((t + 1) * 3));
            }

            // stage2：BMM2(t)——先发射 V(t) 的 GM 预取（与 P 等待重叠）
            // V 的 GM 布局为 token-major：[N2][seqPad][HD]（与模型 KV cache 一致，tower 决策 1 回退）。
            // 注意：BMM2 的 B 操作数需要 [N=dim, K=token]，即对 V 的**转置装载**——
            // 当前用 3D enTranspose（不挂死但出 0），真正的转置装载参数待办，见 README §7。
            Acq<PIPE_MTE2>(B_KV1);
            GmToL1Nz<bfloat16_t>(l1KV1, vGM, kvBase + s2Base * HD, S2T, HD, HD);
            Rls<PIPE_MTE2>(B_KV1);

            const uint32_t pBuf = t % 3;
            const uint16_t ccP = (pBuf == 0) ? CC_P0 : (pBuf == 1) ? CC_P1 : CC_P2;
            CrossCoreWaitFlag<0x4, PIPE_MTE2>(ccP);          // AIV0 的 P(t) 已写入 GM
            CrossCoreWaitFlag<0x4, PIPE_MTE2>(ccP + AIV_CH); // AIV1 的 P(t) 已写入 GM
            const uint16_t ccMM = (t & 1) ? CC_MM1 : CC_MM0;
            CrossCoreWaitFlag<0x4, PIPE_FIX>(ccMM);          // AIV0 已读完 S(t)
            CrossCoreWaitFlag<0x4, PIPE_FIX>(ccMM + AIV_CH); // AIV1 已读完 S(t)

            // P(t) L1 -> L0A
            // P 从 GM 中转（UB->L1 直写在本平台不可靠，tower 决策 3）：AIV 写 GM，AIC Nd2Nz GM->L1，
            // 再把 L1 的 P 装 L0A。三段交接各用已验证原语（MTE3 写 GM / Nd2Nz / drain BufferID）。
            Acq<PIPE_MTE2>(B_PL1);
            GmToL1Nz<bfloat16_t>(l1P[pBuf], pGmGM, ((uint64_t)GetBlockIdx() * 3 + pBuf) * M_PAD * S2T, M_PAD, S2T, S2T);
            Rls<PIPE_MTE2>(B_PL1);
            Acq<PIPE_MTE1>(B_L0);
            Acq<PIPE_MTE1>(B_PL1);   // MTE2 drain 后 L1 数据可见
            // 目的偏移必须给 L0A_OFF_P/2（Q 常驻 L0A 头部 0..8KB，P 进 8KB 起的槽）
            LoadL0_2D<bfloat16_t>(l0A, l1P[pBuf], 0, M_PAD / 16, S2T / 16, M_PAD / 16, M_PAD / 16, false,
                                  L0A_OFF_P / 2);                                                       // P [16,256]
            Rls<PIPE_MTE1>(B_PL1);
            Rls<PIPE_MTE1>(B_L0);

            // V(t)^T 两个 128 列半块 -> L0B，两次 mmad N=128 K=256。
            // B 操作数需要 [N=dim, K=token]（转置读），故走 ifTranspose=true 的 zN->nZ 路径
            // （donor bsa_copy_l1_to_l0b_a5.hpp：mStep=L0K/16、kStep=L0N/16、ifTranspose=true）。
            // BMM2：N=256 拆 2×128（nh）；单次 mmad k=256（M27 在真实 BMM2 形状上已验证合法）。
            // B 操作数 = V^T，按 M27 几何标定表（m16_load_geom/README §3.2/§3.4 + §4 配方）：
            //   L1 V 已是 Nz（行 = token = K、列 = dim = N，dstNzC0Stride = K），
            //   要 mmad 的 B[k][n] = V[k][n] 必须 **ifTranspose=true**（T1：分形内 16x16 转置）；
            //   字段：mStep = 源行分形数 = K/16；kStep = 源列分形数 = N半/16；
            //         srcStride = 源列分形间隔（512B 单位）= K/16；dstStride = 目的 n 分形数 = N半/16。
            //   ★ 不可照抄 BMM1 的元组（M27 §4.1：dstStride 必须是 N/16，抄 K/16 会 507015）。
            for (uint32_t nh = 0; nh < 2; ++nh) {
                const MutexID bC = L0CId(t * 3 + 1 + nh);
                // dim 半块的源偏移：kStartPosition = nh*(128/16) 个列分形 × srcStride(16) × 512B
                //                 = nh*8*16*256 元素 = nh*32768
                const uint32_t kOff = nh * (128 / 16) * (S2T / 16) * 256;
                Acq<PIPE_MTE1>(B_KV1);
                if (cfgGM.GetValue(12) != 0) {
                    LoadL0_2D<bfloat16_t>(l0B, l1KV1, kOff, cfgGM.GetValue(13), cfgGM.GetValue(14),
                                          cfgGM.GetValue(15), cfgGM.GetValue(16), cfgGM.GetValue(17) != 0);
                } else {
                    LoadL0_2D<bfloat16_t>(l0B, l1KV1, kOff, S2T / 16, 128 / 16, S2T / 16, 128 / 16, true);
                }
                Rls<PIPE_MTE1>(B_KV1);
                Acq<PIPE_M>(B_L0);
                Acq<PIPE_M>(B_KV1);
                Acq<PIPE_M>(bC);   // L0C slot 由 M 侧持有（写）
                MmBf16(Slot(l0C, bC), l0A, l0B, L0A_OFF_P / 2, M_PAD, 128, S2T, true);
                Rls<PIPE_M>(B_KV1);
                Rls<PIPE_M>(B_L0);
                Rls<PIPE_M>(bC);
                Acq<PIPE_FIX>(bC);
                if (t == 0) {
                    FixpToGmDbg(dbgC2GM[(uint64_t)GetBlockIdx() * M_PAD * S2T + nh * 128], Slot(l0C, bC), 128, S2T);
                }
                FixpToUb(UbMmRes(t & 1), Slot(l0C, bC), 128, nh * 128, S2T);
                Rls<PIPE_FIX>(bC);
            }
            CrossCoreSetFlag<0x4, PIPE_FIX>(ccMM);         // PV(t) 就绪（两通道）
            CrossCoreSetFlag<0x4, PIPE_FIX>(ccMM + AIV_CH);
        }
    }

private:
    GlobalTensor<bfloat16_t> qGM;
    GlobalTensor<bfloat16_t> kGM;
    GlobalTensor<bfloat16_t> vGM;
    GlobalTensor<uint32_t> seqGM;
    GlobalTensor<float> dbgCGM;    // 调试归档：t=0 的 L0C（fp32）
    GlobalTensor<float> dbgC2GM;   // 调试归档：t=0 的 BMM2 L0C（PV）
    GlobalTensor<bfloat16_t> pGmGM;  // P 的 GM 中转缓冲：[28 unit][3 parity][16][256] bf16
    GlobalTensor<uint32_t> cfgGM;  // 调试：L1->L0 装载参数覆盖（[0..4]=Q, [5..9]=K；0 表示用默认）
};


// ============================================================
// AIV softmax：RegBase VF（m4 递推核 / m5 SwiGLU / m12 RMSNormGated 已验证范式）
// 经典 API 的 fp32->bf16 Cast 与标量<->向量 UB 交接在本平台不可靠（docs/05 §6.1 / #22），
// 故 max/sum/exp/P 生成与 P 的 bf16 落盘全部走 __VEC_SCOPE__ + RegTensor：
//   LoadAlign/StoreAlign + Reduce + Cast(castTrait) + StoreAlign<DIST_PACK_B32>；
// VF 内 store->load 用 LocalMemBar<VEC_STORE, VEC_LOAD>（m12 同款）。
// ============================================================
constexpr uint32_t VL_F32 = 64;             // 256B / 4B：fp32 向量寄存器 lane 数
constexpr uint32_t ROW_VL = S2T / VL_F32;   // 每行 4 个向量寄存器

constexpr Reg::CastTrait castTraitB322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// Vec1：整 tile（AIV_ROWS 行 × S2T）在线 softmax——
//   缩放 + 列 mask → 行 max（折半 + Reduce MAX）→ E = exp(S − mNew) → 行 sum 累加
//   → P = bf16 落 UB（pack b32 连续）；expdiff = exp(mOld − mNew) 落 UB 供 Vec2 用。
// 行状态 m/sum 只由 V 侧读写（DIST_FIRST_ELEMENT_B32 存 / DIST_BRC_B32 广播取），无标量参与。
__simd_vf__ inline void SoftmaxTileVf(__ubuf__ float* sUb, __ubuf__ bfloat16_t* pUb, __ubuf__ float* maskUb,
                                      __ubuf__ float* mUb, __ubuf__ float* sumUb, __ubuf__ float* edUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> s0, s1, s2, s3, t, mn, mo, ed, sg, msk;
        RegTensor<bfloat16_t> pb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        MaskReg oneM = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t r = 0; r < AIV_ROWS; ++r) {
            const uint32_t o = r * S2T;
            LoadAlign(s0, sUb + o);
            LoadAlign(s1, sUb + o + VL_F32);
            LoadAlign(s2, sUb + o + 2 * VL_F32);
            LoadAlign(s3, sUb + o + 3 * VL_F32);
            Muls(s0, s0, SCALE, fullM);
            Muls(s1, s1, SCALE, fullM);
            Muls(s2, s2, SCALE, fullM);
            Muls(s3, s3, SCALE, fullM);
            LoadAlign(msk, maskUb + o);                 Add(s0, s0, msk, fullM);
            LoadAlign(msk, maskUb + o + VL_F32);        Add(s1, s1, msk, fullM);
            LoadAlign(msk, maskUb + o + 2 * VL_F32);    Add(s2, s2, msk, fullM);
            LoadAlign(msk, maskUb + o + 3 * VL_F32);    Add(s3, s3, msk, fullM);
            // 行 max：4 个 VL 折半后再 Reduce（lane0 = 行 max）
            Max(t, s0, s1, fullM);
            Max(mn, s2, s3, fullM);
            Max(t, t, mn, fullM);
            Reduce<ReduceType::MAX>(mn, t, fullM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mo, mUb + r);              // mOld 广播
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(mUb + r, mn, oneM);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mn, mUb + r);              // 行 max 广播
            Max(mn, mn, mo, fullM);                                            // mNew = max(mOld, rowMax)
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(mUb + r, mn, oneM);
            // expdiff 与 sum 衰减
            Sub(ed, mo, mn, fullM);
            Exp(ed, ed, fullM);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(edUb + r, ed, oneM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumUb + r);
            Mul(sg, sg, ed, fullM);
            // E = exp(S − mNew)
            Sub(s0, s0, mn, fullM); Exp(s0, s0, fullM);
            Sub(s1, s1, mn, fullM); Exp(s1, s1, fullM);
            Sub(s2, s2, mn, fullM); Exp(s2, s2, fullM);
            Sub(s3, s3, mn, fullM); Exp(s3, s3, fullM);
            // 行 sum 累加（每组 Reduce SUM 后累到 lane0）
            Reduce<ReduceType::SUM>(t, s0, fullM); Add(sg, sg, t, fullM);
            Reduce<ReduceType::SUM>(t, s1, fullM); Add(sg, sg, t, fullM);
            Reduce<ReduceType::SUM>(t, s2, fullM); Add(sg, sg, t, fullM);
            Reduce<ReduceType::SUM>(t, s3, fullM); Add(sg, sg, t, fullM);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumUb + r, sg, oneM);
            // P = bf16（RNE），pack b32 连续落 UB
            Cast<bfloat16_t, float, castTraitB322B16>(pb, s0, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o, pb, fullM);
            Cast<bfloat16_t, float, castTraitB322B16>(pb, s1, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o + VL_F32, pb, fullM);
            Cast<bfloat16_t, float, castTraitB322B16>(pb, s2, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o + 2 * VL_F32, pb, fullM);
            Cast<bfloat16_t, float, castTraitB322B16>(pb, s3, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o + 3 * VL_F32, pb, fullM);
        }
    }
}

// Vec2：acc = acc·expdiff + PV（整 tile，PV 来自 AIC BMM2 的 FIXP 直写）
__simd_vf__ inline void AccUpdateTileVf(__ubuf__ float* accUb, __ubuf__ float* pvUb, __ubuf__ float* edUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> acc, pv, ed;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0; r < AIV_ROWS; ++r) {
            const uint32_t o = r * S2T;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ed, edUb + r);
            for (uint16_t g = 0; g < (uint16_t)ROW_VL; ++g) {
                LoadAlign(acc, accUb + o + g * VL_F32);
                LoadAlign(pv, pvUb + o + g * VL_F32);
                Mul(acc, acc, ed, fullM);
                Add(acc, acc, pv, fullM);
                StoreAlign(accUb + o + g * VL_F32, acc, fullM);
            }
        }
    }
}

// Combine 权重：w_{i,r} = exp(m_{i,r} − max_i m_{i,r})。整表一次算完（8 lane = 8 行），
// 全程 VF：既不用标量写 UB（docs/05 §6.1/#22），也不用经典 Cast。
__simd_vf__ inline void CombineWeightsVf(__ubuf__ float* mUb, __ubuf__ float* wUb, uint16_t nsplit)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> v, macc, w;
        uint32_t n8 = AIV_ROWS;
        MaskReg m8 = UpdateMask<float>(n8);   // 8 lane = 8 行
        for (uint16_t i = 0; i < nsplit; ++i) {
            LoadAlign(v, mUb + (uint32_t)i * AIV_ROWS);
            if (i == 0) {
                Max(macc, v, v, m8);                // macc = v（无 Copy 原语时的等价初值）
            } else {
                Max(macc, macc, v, m8);
            }
        }
        for (uint16_t i = 0; i < nsplit; ++i) {
            LoadAlign(v, mUb + (uint32_t)i * AIV_ROWS);
            Sub(w, v, macc, m8);
            Exp(w, w, m8);
            StoreAlign<float, StoreDist::DIST_NORM>(wUb + (uint32_t)i * AIV_ROWS, w, m8);
        }
    }
}

// Combine 收尾：fp32 -> bf16 行落盘（RegBase VF；不用经典 API Cast，见 docs/05 §6.1）
__simd_vf__ inline void CastRowToBf16Vf(__ubuf__ float* srcUb, __ubuf__ bfloat16_t* dstUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> v;
        RegTensor<bfloat16_t> pb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t g = 0; g < (uint16_t)ROW_VL; ++g) {
            LoadAlign(v, srcUb + g * VL_F32);
            Cast<bfloat16_t, float, castTraitB322B16>(pb, v, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstUb + g * VL_F32, pb, fullM);
        }
    }
}

// 调试归档用：把 S tile 原样搬到 scratch（V 侧读一次，使 FIXP 直写对后续 MTE3 可见）
__simd_vf__ inline void CopySTileVf(__ubuf__ float* sUb, __ubuf__ float* dstUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> v;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t g = 0; g < (uint16_t)(AIV_ROWS * ROW_VL); ++g) {
            LoadAlign(v, sUb + g * VL_F32);
            StoreAlign(dstUb + g * VL_F32, v, fullM);
        }
    }
}

// M101 r2 · **第 9 类整块替换**新增：FD combine 的归并累加走 RegBase VF。
//   替换前（donor / 本抽取件 r1 逐字形态）：`Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`
//   —— 经典 memory-based API，人类逐字禁：「…不应该使用 memory base 的 API」。
//   替换后：LoadAlign + Muls + Add + StoreAlign（全在 __VEC_SCOPE__ 内），与 `AccUpdateTileVf`
//   同款。**累加次序（nsplit 循环、逐 i 先乘后加）一字未动**；官方 Reg 规格里 Mul/Add/Muls 都是
//   0 ulp ⇒ 预期与替换前逐位相同，由"与 m10_attn_decode/data 归档 dump 逐字节对拍"见证。
__simd_vf__ inline void AccFmaRowVf(__ubuf__ float* numUb, __ubuf__ float* accUb, float w)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> acc, num;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t g = 0; g < (uint16_t)ROW_VL; ++g) {
            LoadAlign(acc, accUb + g * VL_F32);
            LoadAlign(num, numUb + g * VL_F32);
            Muls(acc, acc, w, fullM);
            Add(num, num, acc, fullM);
            StoreAlign(numUb + g * VL_F32, num, fullM);
        }
    }
}

// ============================================================
// AIV
// ============================================================
class AttnCoreAiv {
public:
    __aicore__ inline AttnCoreAiv() {}

    __aicore__ inline void Init(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v, __gm__ uint8_t* out,
                                __gm__ uint32_t* seq, __gm__ uint8_t* wsAcc, __gm__ uint8_t* wsM, __gm__ uint8_t* wsS,
                                __gm__ float* maskCol, __gm__ uint8_t* dbgS, __gm__ uint8_t* dbgP,
                                __gm__ uint8_t* gP)
    {
        outGM.SetGlobalBuffer((__gm__ bfloat16_t*)out);
        seqGM.SetGlobalBuffer(seq);
        wsAccGM.SetGlobalBuffer((__gm__ float*)wsAcc);
        wsMGM.SetGlobalBuffer((__gm__ float*)wsM);
        wsSGM.SetGlobalBuffer((__gm__ float*)wsS);
        maskColGM.SetGlobalBuffer(maskCol);
        dbgSGM.SetGlobalBuffer((__gm__ float*)dbgS);
        dbgPGM.SetGlobalBuffer((__gm__ uint8_t*)dbgP);
        pGmGM.SetGlobalBuffer((__gm__ bfloat16_t*)gP);
        (void)q; (void)k; (void)v;
    }

    __aicore__ inline void Run()
    {
        const uint32_t bid = GetBlockIdx();   // AIV 0..2*numBlocks-1
        const uint32_t aic = bid >> 1;
        const uint32_t half = bid & 1;        // dualDstCtl 的 M 半块（本核分得的 8 行）
        const uint32_t seq = seqGM.GetValue(0);
        const uint32_t seqPad = AlignUp(seq, S2T);
        const uint32_t nTiles = seqPad / S2T;
        const uint32_t chunkTiles = CeilDiv(nTiles, SPLITS);
        const uint32_t nsplit = CeilDiv(nTiles, chunkTiles);
        const uint32_t n2 = aic & 1u;
        const uint32_t split = aic >> 1;
        const bool hasWork = (aic < 2 * nsplit);
        uint32_t tiles = 0;
        if (hasWork) {
            tiles = MinU(chunkTiles, nTiles - split * chunkTiles);
        }

#if defined(M15AC_DBG_STAGE0)
        // 阶段 0：只验证 FD 收尾同步骨架
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CC_AIVDONE);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(CC_RDY);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CC_ALLDONE);
        (void)n2; (void)split; (void)half; (void)seq; (void)tiles;
#else
        printf("[M15AC] AIV %u enter aic=%u half=%u hasWork=%u tiles=%u\n", bid, aic, half, (uint32_t)hasWork, tiles);
        if (hasWork) {
            RunTiles(n2, split, tiles, chunkTiles, seq, half);
            printf("[M15AC] AIV %u vec done\n", bid);
            WritePartialsAndSignal(n2, split, half);
            printf("[M15AC] AIV %u partial done\n", bid);
        } else {
            CrossCoreSetFlag<0x2, PIPE_MTE3>(CC_AIVDONE);
            printf("[M15AC] AIV %u idle signalled\n", bid);
        }

        // ---- FD combine 阶段 ----
        printf("[M15AC] AIV %u wait RDY\n", bid);
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(CC_RDY);   // 全体 AIC barrier 完成、ws 可读
        printf("[M15AC] AIV %u got RDY\n", bid);
#if !defined(M15AC_DEBUG_SKIP_COMBINE)
        if (bid < 4) {
            Combine(nsplit, bid & 1, (bid >> 1) & 1);
        }
#endif
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CC_ALLDONE);
        printf("[M15AC] AIV %u done\n", bid);
#endif
    }

private:
    // 行内折半归约（经典 API 版本）已由上面的 RegBase VF 取代。

    __aicore__ inline void RunTiles(uint32_t n2, uint32_t split, uint32_t tiles, uint32_t chunkTiles, uint32_t seq,
                                    uint32_t half)
    {
        (void)n2;
        (void)chunkTiles;
        LocalTensor<uint8_t> mm0(TPosition::VECIN, UB_MM0, 8192);
        LocalTensor<uint8_t> mm1(TPosition::VECIN, UB_MM1, 8192);
        LocalTensor<uint8_t> pcastU(TPosition::VECCALC, UB_PC, 4096);
        LocalTensor<uint8_t> accU(TPosition::VECCALC, UB_ACC, 8192);
        LocalTensor<uint8_t> maskU(TPosition::VECCALC, UB_MASK, 8192);
        LocalTensor<uint8_t> scratchU(TPosition::VECCALC, UB_MB, 8192);
        LocalTensor<uint8_t> mvecU(TPosition::VECCALC, UB_MVEC, 32);
        LocalTensor<uint8_t> svecU(TPosition::VECCALC, UB_SVEC, 32);
        LocalTensor<uint8_t> tvecU(TPosition::VECCALC, UB_TVEC, 32);
        LocalTensor<bfloat16_t> Pc16 = pcastU.template ReinterpretCast<bfloat16_t>();

        const uint32_t bidD = GetBlockIdx();
        __ubuf__ bfloat16_t* pcB16 = reinterpret_cast<__ubuf__ bfloat16_t*>(pcastU.GetPhyAddr());
        __ubuf__ float* maskF = reinterpret_cast<__ubuf__ float*>(maskU.GetPhyAddr());
        __ubuf__ float* scratchF = reinterpret_cast<__ubuf__ float*>(scratchU.GetPhyAddr());
        __ubuf__ float* accF = reinterpret_cast<__ubuf__ float*>(accU.GetPhyAddr());
        __ubuf__ float* mF = reinterpret_cast<__ubuf__ float*>(mvecU.GetPhyAddr());
        __ubuf__ float* sumF = reinterpret_cast<__ubuf__ float*>(svecU.GetPhyAddr());
        __ubuf__ float* edF = reinterpret_cast<__ubuf__ float*>(tvecU.GetPhyAddr());

        // 行状态初值：m = MASKV（exp(mOld−mNew)=0，等价 −inf）、sum = 0；acc = 0
        Duplicate(mvecU.template ReinterpretCast<float>(), MASKV, 8);
        Duplicate(svecU.template ReinterpretCast<float>(), 0.0f, 8);

        // acc token：V 侧常驻持有，loop 结束后阻塞释放给 MTE3（WritePartials）
        Acq<PIPE_V>(B_ACC);
        Duplicate(accU.template ReinterpretCast<float>(), 0.0f, 2048);

        // 列 mask 载入 + 一份全 0 mask。**只有尾 tile（valid < S2T）才加 mask**：
        // 非尾 tile 加 mask 会把整段 S 打成 -3.39e38（host 的 mask 是"尾 tile 内有效列=0、其余=MASKV"）。
        LocalTensor<uint8_t> maskzU(TPosition::VECCALC, UB_MASKZ, 8192);
        Acq<PIPE_MTE2>(B_MSK);
        for (uint32_t r = 0; r < AIV_ROWS; ++r) {
            DataCopy(maskU.template ReinterpretCast<float>()[r * S2T], maskColGM[0],
                     DataCopyParams{1, S2T * 4 / 32, 0, 0});
        }
        Rls<PIPE_MTE2>(B_MSK);
        Acq<PIPE_V>(B_MSK);   // MTE2 drain 后数据对 V 可见；持有至 kernel 结束
        __ubuf__ float* maskzF = reinterpret_cast<__ubuf__ float*>(maskzU.GetPhyAddr());
        Duplicate(maskzU.template ReinterpretCast<float>(), 0.0f, AIV_ROWS * S2T);   // 全 0 mask（V 写 UB）

        // init-set：两个 mmRes buffer 均空闲
        CrossCoreSetFlag<0x4, PIPE_V>(CC_MM0);
        CrossCoreSetFlag<0x4, PIPE_V>(CC_MM1);

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t parity = t & 1;
            const uint16_t ccMM = parity ? CC_MM1 : CC_MM0;
            LocalTensor<uint8_t> mmB = parity ? mm1 : mm0;
            __ubuf__ float* sf = reinterpret_cast<__ubuf__ float*>(mmB.GetPhyAddr());
            const uint32_t pBuf = t % 3;
            const uint32_t s2Base = (split * chunkTiles + t) * S2T;
            const uint32_t valid = MinU(S2T, seq - s2Base);   // 尾 tile 判定（只有尾 tile 加列 mask）

            // ---- Vec1：等 S(t) 就绪（mode 4：AIC FIXP 直写本 AIV 的 UB）----
            CrossCoreWaitFlag<0x4, PIPE_V>(ccMM);
            if (t == 0) {
                // 调试归档：S 先经 V 侧搬进 scratch（保证 FIXP 直写已可见），再由 MTE3 落 GM
                CopySTileVf(sf, scratchF);
                Acq<PIPE_V>(B_DBG);
                Rls<PIPE_V>(B_DBG);
                Acq<PIPE_MTE3>(B_DBG);
                DataCopy(dbgSGM[(uint64_t)bidD * 2048], scratchU.template ReinterpretCast<float>(),
                         DataCopyParams{8, 32, 0, 0});
                Rls<PIPE_MTE3>(B_DBG);
            }
            Acq<PIPE_V>(B_PC);
            SoftmaxTileVf(sf, pcB16, (valid < S2T) ? maskF : maskzF, mF, sumF, edF);
            Rls<PIPE_V>(B_PC);                     // drain：E 已读完 → 放行 mmRes 复用与 P 的 MTE3
            CrossCoreSetFlag<0x4, PIPE_V>(ccMM);   // S(t) 已消费

            // P 写回 L1（每行 16 个 32B 块，dstGap=15 落到 NZ 列分形）
            Acq<PIPE_MTE3>(B_PC);
            if (t == 0) {
                DataCopy(dbgPGM[(uint64_t)bidD * 4096], pcastU, DataCopyParams{8, 16, 0, 0});
            }
            // P 落 GM（行主序 [28 unit][3 parity][16][256] bf16）：本核的 8 行连续 4KB 一次写完；
            // 由 AIC 侧用 Nd2Nz 搬进 L1（GM->L1 是本仓库验证过的路径）。
            // unit 必须用 **AIC 的 blockIdx 空间**（= AIV bid >> 1），与 AIC 读取侧一致；
            // 用 AIV bid 会让只有 AIC0 的对上，且 AIV≥28 时越界写坏相邻 unit。
            const uint32_t unit = bidD >> 1;
            if (unit < SPLITS * 2) {
                DataCopy(pGmGM[((uint64_t)unit * 3 + pBuf) * M_PAD * S2T + half * AIV_ROWS * S2T], Pc16,
                         DataCopyParams{8, 16, 0, 0});
            }
            Rls<PIPE_MTE3>(B_PC);
            CrossCoreSetFlag<0x4, PIPE_MTE3>((pBuf == 0) ? CC_P0 : (pBuf == 1) ? CC_P1 : CC_P2);  // P 已落 GM

            // ---- Vec2：等 PV(t)，acc = acc·expdiff + PV ----
            CrossCoreWaitFlag<0x4, PIPE_V>(ccMM);
            AccUpdateTileVf(accF, sf, edF);
            CrossCoreSetFlag<0x4, PIPE_V>(ccMM);
        }
        Rls<PIPE_V>(B_ACC);   // drain：acc/行状态写完后 ws 才可写
    }

    __aicore__ inline void WritePartialsAndSignal(uint32_t n2, uint32_t split, uint32_t half)
    {
        LocalTensor<uint8_t> accU(TPosition::VECCALC, UB_ACC, 8192);
        LocalTensor<uint8_t> mvecU(TPosition::VECCALC, UB_MVEC, 32);
        LocalTensor<uint8_t> svecU(TPosition::VECCALC, UB_SVEC, 32);
        LocalTensor<float> acc = accU.template ReinterpretCast<float>();
        LocalTensor<float> mV = mvecU.template ReinterpretCast<float>();
        LocalTensor<float> sV = svecU.template ReinterpretCast<float>();

        const uint32_t rowBase = half * AIV_ROWS;
        Acq<PIPE_MTE3>(B_ACC);
        for (uint32_t r = 0; r < AIV_ROWS; ++r) {
            const uint32_t row = rowBase + r;
            const uint64_t off = (((uint64_t)n2 * SPLITS + split) * M_PAD + row) * S2T;
            DataCopy(wsAccGM[off], acc[r * S2T], DataCopyParams{1, S2T * 4 / 32, 0, 0});
        }
        const uint64_t mOff = ((uint64_t)n2 * SPLITS + split) * M_PAD + rowBase;
        DataCopy(wsMGM[mOff], mV, DataCopyParams{1, AIV_ROWS * 4 / 32, 0, 0});
        DataCopy(wsSGM[mOff], sV, DataCopyParams{1, AIV_ROWS * 4 / 32, 0, 0});
        Rls<PIPE_MTE3>(B_ACC);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CC_AIVDONE);
    }

    // FD combine：bid<4 的 AIV。n2=bid&1，rowHalf=(bid>>1)&1。
    __aicore__ inline void Combine(uint32_t nsplit, uint32_t n2, uint32_t rowHalf)
    {
        LocalTensor<uint8_t> mmatU(TPosition::VECCALC, UB_CVF_M, 1024);
        LocalTensor<uint8_t> smatU(TPosition::VECCALC, UB_CVF_S, 1024);
        LocalTensor<uint8_t> wmatU(TPosition::VECCALC, UB_CVF_W, 1024);
        LocalTensor<uint8_t> accrU(TPosition::VECCALC, UB_ACCR, 1024);
        LocalTensor<uint8_t> tmprU(TPosition::VECCALC, UB_TMPR, 1024);
        LocalTensor<uint8_t> numrU(TPosition::VECCALC, UB_NUMR, 1024);
        LocalTensor<uint8_t> outrU(TPosition::VECCALC, UB_OUTR, 512);
        LocalTensor<float> mM = mmatU.template ReinterpretCast<float>();
        LocalTensor<float> sM = smatU.template ReinterpretCast<float>();
        LocalTensor<float> wM = wmatU.template ReinterpretCast<float>();
        LocalTensor<float> accR = accrU.template ReinterpretCast<float>();
        LocalTensor<float> tmpR = tmprU.template ReinterpretCast<float>();
        LocalTensor<float> numR = numrU.template ReinterpretCast<float>();
        LocalTensor<bfloat16_t> outR = outrU.template ReinterpretCast<bfloat16_t>();

        // 每 split 的 m/sum：[nsplit][8]（本半的行 rowHalf*8..+8）
        const uint64_t msOff = ((uint64_t)n2 * SPLITS) * M_PAD + rowHalf * AIV_ROWS;
        printf("[M15AC] combine enter n2=%u half=%u nsplit=%u\n", n2, rowHalf, nsplit);
        Acq<PIPE_MTE2>(B_CIN);
        printf("[M15AC] combine got CIN mte2\n");
        for (uint32_t i = 0; i < nsplit; ++i) {
            DataCopy(mM[i * AIV_ROWS], wsMGM[msOff + (uint64_t)i * M_PAD], DataCopyParams{1, AIV_ROWS * 4 / 32, 0, 0});
            DataCopy(sM[i * AIV_ROWS], wsSGM[msOff + (uint64_t)i * M_PAD], DataCopyParams{1, AIV_ROWS * 4 / 32, 0, 0});
        }
        Rls<PIPE_MTE2>(B_CIN);
        printf("[M15AC] combine m/s copied\n");
        Acq<PIPE_V>(B_CIN);
        printf("[M15AC] combine got CIN v\n");

        // 权重 w_{i,r} = exp(m_{i,r} − max_i m_{i,r})：整表 VF 计算（8 lane = 8 行），
        // 之后标量只**读** VF 写好的权重（标量读 V 写 UB 是安全的；危险的是标量写）
        CombineWeightsVf(reinterpret_cast<__ubuf__ float*>(mM.GetPhyAddr()),
                         reinterpret_cast<__ubuf__ float*>(wM.GetPhyAddr()), (uint16_t)nsplit);
        PipeBarrier<PIPE_V>();
        printf("[M15AC] combine weights done\n");

        for (uint32_t lr = 0; lr < AIV_ROWS; ++lr) {
            const uint32_t row = rowHalf * AIV_ROWS + lr;
            if (row >= GQA_G) {
                continue;   // pad 行不写出
            }
            float den = 0.0f;
            for (uint32_t i = 0; i < nsplit; ++i) {
                den += wM.GetValue(i * AIV_ROWS + lr) * sM.GetValue(i * AIV_ROWS + lr);
            }
            if (den == 0.0f) {
                continue;
            }
            const float rcp = 1.0f / den;
            Duplicate(numR, 0.0f, S2T);
            for (uint32_t i = 0; i < nsplit; ++i) {
                const uint64_t off = (((uint64_t)n2 * SPLITS + i) * M_PAD + row) * S2T;
                // 归一化因子折进标量权重：numR 只做 Muls/Add（scalar 作源操作数），
                // 末段就没有 rcp 标量参与，可整体交给 RegBase VF 完成 bf16 落盘。
                const float w = wM.GetValue(i * AIV_ROWS + lr) * rcp;
                Acq<PIPE_MTE2>(B_ACCIN);
                DataCopy(accR, wsAccGM[off], DataCopyParams{1, S2T * 4 / 32, 0, 0});
                Rls<PIPE_MTE2>(B_ACCIN);
                Acq<PIPE_V>(B_ACCIN);
                // [第 9 类整块替换] 原为经典 memory-based 的
                //   `Muls(tmpR, accR, w, S2T); Add(numR, numR, tmpR, S2T);`
                // 依人类逐字规则（「…不应该使用 memory base 的 API」）改走 RegBase VF；
                // nsplit 累加次序未动。tmpR（UB_TMPR 槽）在替换后不再被引用 —— 声明保留，
                // 以免动段内那张编译期静态 UB 地址表（不占运行期资源）。
                (void)tmpR;
                AccFmaRowVf(reinterpret_cast<__ubuf__ float*>(numR.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ float*>(accR.GetPhyAddr()), w);
                Rls<PIPE_V>(B_ACCIN);
            }
            Acq<PIPE_V>(B_COUT);
            CastRowToBf16Vf(reinterpret_cast<__ubuf__ float*>(numR.GetPhyAddr()),
                            reinterpret_cast<__ubuf__ bfloat16_t*>(outR.GetPhyAddr()));
            Rls<PIPE_V>(B_COUT);
            Acq<PIPE_MTE3>(B_COUT);
            DataCopy(outGM[((uint64_t)n2 * GQA_G + row) * S2T], outR, DataCopyParams{1, S2T * 2 / 32, 0, 0});
            Rls<PIPE_MTE3>(B_COUT);
            printf("[M15AC] combine n2=%u half=%u row=%u den=%f\n", n2, rowHalf, row, den);
        }
        Rls<PIPE_V>(B_CIN);
        printf("[M15AC] combine done n2=%u half=%u\n", n2, rowHalf);
    }

private:
    GlobalTensor<bfloat16_t> outGM;
    GlobalTensor<uint32_t> seqGM;
    GlobalTensor<float> wsAccGM;
    GlobalTensor<float> wsMGM;
    GlobalTensor<float> wsSGM;
    GlobalTensor<float> maskColGM;
    GlobalTensor<float> dbgSGM;     // 调试归档：t=0 的 S（fp32）
    GlobalTensor<uint8_t> dbgPGM;   // 调试归档：t=0 的 P（bf16）
    GlobalTensor<bfloat16_t> pGmGM;  // P 的 GM 中转缓冲（与 AIC 同一 buffer）
};
}  // namespace M15AC

// ============================================================
// 段入口（M101 生成）：融合 kernel 与本 mission 的独立验证路共用同一个 body。
// 参数表 = donor `__global__` 入口的参数表（AIC/AIV 两侧 `Init()` 的并集），
// 只是把"分核调度"搬进来 —— donor 原本写在 `__global__` 入口里，那属于 host 侧不抽取的部分。
// ============================================================
namespace M15AC {
__aicore__ inline void AttnCoreBody(__gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* v,
                                    __gm__ uint8_t* out, __gm__ uint32_t* seq, __gm__ uint8_t* wsAcc,
                                    __gm__ uint8_t* wsM, __gm__ uint8_t* wsS, __gm__ float* maskCol,
                                    __gm__ uint8_t* dbgS, __gm__ uint8_t* dbgP, __gm__ uint8_t* dbgC,
                                    __gm__ uint32_t* cfg, __gm__ uint8_t* dbgC2, __gm__ uint8_t* gP)
{
    if ASCEND_IS_AIV {
        AttnCoreAiv op;
        op.Init(q, k, v, out, seq, wsAcc, wsM, wsS, maskCol, dbgS, dbgP, gP);
        op.Run();
    }
    if ASCEND_IS_AIC {
        AttnCoreAic op;
        op.Init(q, k, v, out, seq, wsAcc, wsM, wsS, dbgC, cfg, dbgC2, gP);
        op.Run();
    }
}

// ---- 段间契约的编译期钉子：id 一旦重编号，这里先红（提示同步顶层打平表）----
static_assert(CC_MM0 == 0 && CC_MM1 == 1 && CC_P0 == 5 && CC_P1 == 6 && CC_P2 == 7 && CC_BAR == 8 &&
              CC_AIVDONE == 9 && CC_RDY == 10 && CC_ALLDONE == 11 && AIV_CH == 16,
              "cross-core flagId 清单变了 —— m15_layer_resources.h §4 的顶层打平表必须同步");
static_assert(B_Q == 0 && B_KV0 == 1 && B_KV1 == 2 && B_L0 == 3 && B_C0 == 4 && B_C1 == 5 && B_C2 == 6 &&
              B_C3 == 7 && B_PL1 == 8, "AIC 侧 MutexID 清单变了");
static_assert(B_PC == 0 && B_ACC == 1 && B_MSK == 2 && B_CIN == 3 && B_COUT == 4 && B_ACCIN == 5 &&
              B_DBG == 6, "AIV 侧 MutexID 清单变了");
}  // namespace M15AC

#endif  // M15_ATTN_CORE_H
