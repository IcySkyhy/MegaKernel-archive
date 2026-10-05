// ============================================================
// m15_attn_fa_core.h —— M117 / Wave-B3：**稠密 causal prefill attention core**（m-tile FA）
// ============================================================
// ██ 状态（M189 更新）：**挂死已解、数值已收敛（四档 PASS）** ██
//   · **M185 之前**：`m=64 / 16 / 96 / 128` 全部在设备上挂死（`timeout` 被 kill、无输出文件）。
//   · **M185 起**：核间同步由 **mode 4（AIC↔单个 AIV，带 `+16` 双通道）改成 mode 2**
//     （1 AIC ↔ 配对的 2 AIV），并删掉 `FAC_AIV_CH` 那套偏移 ⇒ **四个 m 档全部跑完（EXIT=0），
//     不再挂死，拿到 PASS/FAIL 读数**（`m25_attn_fa_core/evidence/M185_readings.txt`）。
//   · **M185 交付时数值 FAIL**（`maxratio=195.58`），M185 把失败点缩到 cube 数据面但**未定位**。
//   · **M189 定位到并修复根因**：`L1 -> L0B` 的两个循环里，**生产侧（MTE1）的 BufferID
//     `Acq/Rls` 被放在 `ks`（BMM1 的 2×128 k 半块）/ `nh`（BMM2 的 2×128 维半块）循环之外**，
//     于是 M 侧两条 `Mmad` 都读到**循环最后一次 L0B 装载**的结果（stale）：
//       – BMM1：两个 k 半块都用了第二个半块的 K ⇒ `S = Q·Kᵀ` 系统性错（M185 看到的 maxdiff≈43）；
//       – BMM2：两个 nh 半块都用了第二个半块的 V ⇒ PV 也系统性错（`out[0] ≠ V[0]`）。
//     修法 = 把 `Acq<PIPE_MTE1>(B_L0/B_K)` / `Rls<PIPE_MTE1>(...)` 挪进循环、每个半块
//     装载后立即 release（与 donor `m10_attn_decode` BMM1 的 per-half 交接同款）。**只改同步，
//     不改任何装载几何/数值参数**。
//   · **M189 设备读数**：`check_ref.py` 四档 **core 全 PASS（over=0，maxratio=0.2962）**；
//     负向对照 `negmask/negshift/negstart` **全 FAIL**；同一 `m` 三跑 `sha256` 一致。
//     证据见 `m25_attn_fa_core/evidence/M189_readings.txt`。
//   ⇒ 离线判据自证仍在：`check_ref.py` 回灌 fp64 参考 ⇒ PASS（maxratio 0.2203）、
//     ×1.05 ⇒ FAIL（over=117926/227328, maxratio 3.0760），见 `evidence/check_ref_selftest.txt`。
//   ⇒ **默认惰性**：`M25FA_*` 系列编译期开关**全部默认关闭**，默认路径行为与不编这些开关时一致。
// ============================================================
// 交付形态（`docs/15` §M103-2.7 第 1 条）：
//   · include guard；**无 `main()`**（`main()` 只在本 wave 自己的 `m25_attn_fa_core/*.asc` 里）；
//   · **具名 `namespace M15FAC`**（不用匿名 namespace —— 融合 TU 里已有多段同名工具）；
//   · 设备代码只依赖设备头（`kernel_operator.h`）+ **KV 布局唯一权威** `m15_attn_kv.h`
//     （塔契约 #6：KV/cache 寻址只用 `M15KV_KV_*` 宏，**不自带第二份数字**）；
//   · 资源全部编译期静态（BufferID / flagId / UB 窗 / L1 区 / L0C），并带峰值 `static_assert`。
//
// ---------------------------------------------------------------------------
// 【本段做什么】单请求、恒等 `block_table` 下的**稠密 causal** FlashAttention：
//     out[i, h, :] = Σ_{j ≤ pos(i)} softmax(Q[i,h,:]·K[j, n2(h), :]ᵀ · scale + mask) · V[j, n2(h), :]
//   其中 `pos(i) = posBase + i`、`n2(h) = h / FAC_G`（GQA：24 q 头 / 2 kv 头）。
//   **所有矩阵乘法一律 `Mmad`（cube）**：QKᵀ 与 PV 各为 cube 上的 mmad；AIV 只做非矩阵乘的
//   向量/标量运算（缩放 / 三角掩码 / online softmax / 归一）。
//   —— 人类裁决（逐字）：「凡是涉及矩阵乘法的操作都需要用 mmad 实现，不管 M=1 或者是多大」、
//      「只要是 mmad，就要用 cube 去做，没必要浪费时间去对比」。
//
// 【不许拿官方 QSA 输出当判据】本段自带**三角掩码 + `s2` 上界**，走的是**稠密 causal**；
//   官方 QSA 的 `indexer_budget=2048` 只覆盖约一半历史（`docs/17` §7 / M103-4），两者在任何长度下
//   都不可能一致 ⇒ 判据只能是**自建稠密 causal 参考**（`m25_attn_fa_core/check_ref.py`，numpy fp64）。
//   本文件与 README **不写**「m=4097 对齐官方输出」这类表述。
//
// 【同步纪律】核内 pipeline 只用 **BufferID**（`Acq/Rls` = `GetBufInternal/RlsBufInternal`
//   的封装，acquire/release 一律 mode=false，**不用 SetFlag/WaitFlag 系列**）；核间只用
//   **`CrossCoreSetFlag/WaitFlag`**（**mode 2**：1 AIC ↔ 它配对的那 2 个 AIV；配对口径见 §2）。
//   除值依赖外不用 `PIPE_S`。落 GM 只经 **DMA（MTE3）**，且走「UB 写 → 写出侧阻塞释放
//   （mode=false）→ 才搬」（`docs/05` §6.1 规则 ⓔ）—— **P 不落 GM**，走 UB→L1 硬通道（§5b）。
//
// ---------------------------------------------------------------------------
// 【融合清单 —— **待收敛后启用的提案**】(`docs/15` §M103-2.7 第 2 条的六项；同内容随 README 入库)
//   ⚠ 设备端未收敛（见文件头横幅）⇒ 下列挂载点/资源窗/开关**均未在设备上验证可用**，
//     只能当"收敛后照此接入"的提案，**不得读成"已可用"**。
//
//  (a) 挂载点
//      · 入口符号（**11 参**，逐字；`witScratch` 必填、无默认值）：
//        `M15FAC::FaCoreBody<Causal>(qGm, qStride, kvGm, outGm, pGm, witScratch, m, posBase, ctx, lane, nBlk)`
//      · 相位：**attention 相位的核心步**。上游 = B2 的 prolog（产出 Q 平面）+ 主 KV cache 填
//        （产出 `M15KV_KV_*` 平面）；下游 = o_proj / ×sigmoid(gate)。
//      · 消费的 GM 平面：① Q 平面（bf16，见 (f)）；② 主 KV 平面（`m15_attn_kv.h` 编址，**只读**）。
//      · 生产的 GM 平面：attention 输出平面（bf16 `[m][FAC_NH*FAC_HD]`，行距 = 6144）。
//      · 跨段：**段起点不 wait 任何段外 flag**；段尾由调用方补 `PipeBarrier<PIPE_ALL>()`。
//
//  (b) `LayerArgs` 需要的字段（Wave A 照它加）—— **5 个 `__gm__` 指针 + 6 个标量**
//      `qGm, kvGm, outGm, pGm, witnessGm`（`__gm__ uint8_t*`；**注意是 5 个指针**）
//      + `qStride, m, posBase, ctx, lane, nBlk`（标量）。
//      `witnessGm` = **跨核记账见证区**（只被编译期开关 `M25FA_WIT` 使用；**默认惰性**，默认路径不读不写）：
//        需要 `(28 + 2*nBlk) × 32(id) × 2(set/wait) × 4 B` 字节；`nBlk = 28` 时 = **21,504 B**（32 B 对齐）。
//        ⚠ 本档测试 host 的 `kWitBytes` 目前按 **56 槽** 分配（`nBlk ≤ 14` 时够用）——
//          这是探针构建的一个**已知不足**，如实记在 `evidence/exclusions.md` §8。
//      `m ∈ [1, M15Kv::PREFILL_M]`；`4097` 是**验收档、不是接口假设**（B2 共同契约 #5）。
//
//  (c) 资源窗（字节区间 + 峰值；供 Wave C 做「每 mode 独立断言」）
//      UB（AIV，峰值 `FAC_UB_TOTAL` = 148,096 B = **144.6 KB** « 248 KB）：
//        [0,16,384) UB_S0 fp32[32,128]；[16,384,32,768) UB_S1；
//        [32,768,65,536) UB_PV0 fp32[32,256]；[65,536,98,304) UB_PV1；
//        [98,304,114,688) UB_PC bf16[32,256] 16 KB（P 落盘只用前 8 KB；归一化用满）；
//        [114,688,147,456) UB_ACC fp32[32,256]；[147,456,148,096) UB_THR[64]/UB_M/UB_SUM/UB_ED fp32[32]×3
//      L1（AIC，峰值 `FAC_L1_TOTAL` = 196,608 B = **192 KB** « 512 KB）：
//        [0,32,768) Q NZ[64,256]；[32,768,98,304) K NZ[128,256]；[98,304,163,840) V NZ[128,256]；
//        [163,840,196,608) P NZ[64,128] ×2 parity
//      L0A 64 KB（Q[64,256] @0 = 32 KB，P[64,128] @32 KB = 16 KB）；
//      L0B 64 KB（用 32 KB）；L0C **4 × 32 KB = 128 KB « 256 KB**。
//      GM scratch（本段自用）：**P 中转已不再使用**（`pScratch` 实参保留签名）——
//      P 改走 **UB→L1 硬通道**（M121 真机实测：本平台 3510 可用，`docs/evidence/ub_to_l1`；
//      AIV 逐列分形 burst 写 L1 的 NZ 槽 → AIC 直接 `L1 -> L0A`），全程不落 GM。
//
//  (d) BufferID 清单（核内）与 flagId 清单（核间 CrossCore）
//      AIC MutexID：`B_Q=0, B_K=1, B_V=2, B_L0=3, B_C0=4, B_C1=5, B_C2=6, B_C3=7, B_PL1=8`
//      AIV MutexID：`B_S0=0, B_S1=1, B_PV0=2, B_PV1=3, B_PC=4`（`B_ACC=5` 仅登记 UB 窗名）
//      CrossCore flagId（**mode 2**；每核池 0..15，且**每个同步事件只用一个 id** —— 不再有
//      mode-4 那套「AIC 侧对 AIV1 通道 +16」）：
//        `CC_S_RDY=0, CC_S_FREE=1`     —— S(t) parity0 就绪（AIC→{2 AIV}）/ 已消费（{2 AIV}→AIC）
//        `CC_PV_RDY=2, CC_PV_FREE=3`   —— PV(t) parity0 就绪 / 已消费
//        （parity1 用 +1：S_FREE+1 / PV_FREE+1；**每个逻辑通道只有一个 set 方**，见下面常量的说明）
//        `CC_P0=5,  CC_P1=6`  —— P(t) parity **已直写进 L1**（{2 AIV}→AIC）
//      配对口径（mode 2，B1/B2/M121 实测）：AIC 侧 `set` 一次 → 配对 2 个 AIV 各 `wait` 一次；
//      **配对 2 个 AIV 都 `set`** 后 AIC 的 `wait` 才放行（严格 2 set ↔ 1 wait；只让一半 AIV set
//      会让 AIC 永久等 —— M121 `kill=5` 实测 rc=124）。
//      相邻性：0/1/2/3 与 5/6 两簇不共享物理槽；同一 (核型, mode) 子空间内同时「在飞」的 id
//      最多 2 个（一 RDY 一 FREE）。每 id 的 set 次数 ≈ 该核处理到的 tile 数 / 2
//      （**不是固定常数**）⇒ Wave C 必须按档断言 ≤ 15（见 README 未完成项）。
//
//  (e) 需要相位边界的位置
//      · 段尾：调用方补 `PipeBarrier<PIPE_ALL>()`。
//      · 段内**不需要**全体 AIV 的 mode-0 barrier：AIC↔AIV 全部走 mode-2 的成对 flag。
//
//  (f) `m` / `pos` / Q 平面语义
//      · Q 平面：bf16，**行距 = `qStride` 元素**，头 `h` 占列 `[h*FAC_HD, h*FAC_HD+FAC_HD)`。
//        这正是 `m15_attn_prolog.h` 的 `out` 平面（`AP_OUT_N = 13952`、`OUT_Q = 0`）—— Q 平面对齐
//        prolog 产物，**不另造第二套布局**。
//        ⚠ 调用方必须把 Q 平面**按 `FAC_P` 行向上圆整分配**（尾 tile 整块读入）。
//      · 输出平面：bf16 `[m][FAC_NH*FAC_HD]`，行距 6144，头 h 占列 `[h*FAC_HD, +FAC_HD)`。
//        ⚠ 调用方必须把 Q 平面**与输出平面**都按 `FAC_P` 行向上圆整分配（尾 tile 写满 64 行）。
//      · 主 KV 平面：`M15KV_KV_BYTE_OFF_CONTIG(pos, n2, KV_LANE_K|V, dim)` 编址。本段**只读**。
//        ⚠ 契约（M103-6 第 8 条要求显式写出）：**只支持单请求 / 恒等 `block_table`**
//          （`physical_block == logical_block`）；batch>1 或非恒等表会静默错。
//        容量：`BlocksFor(ctx)` 页即可，**不需要按 tile 圆整**（可见页数逐 tile 收紧）。
//      · `posBase` = Q 行 0 的**位置**（chunked prefill 的起点）；行 i 位置 `= posBase + i`，
//        只 attend KV 列 `0 .. posBase+i`。decode 档 = `m=1, posBase=ctx-1`。
//
// ---------------------------------------------------------------------------
// 【真实 shape】`FAC_P=64`（donor config4 的 sOuter）、`FAC_SIN=128`（config4 的 sInner）、
//   `FAC_HD=FAC_DV=256`（config4 的 D/DV，也是 checkpoint 的 head_dim）。
//   验收档：prefill `m=4097, posBase=0, ctx=4097`；decode 同档回归 `m=1, posBase=4096, ctx=4097`。
// ============================================================
#ifndef M15_ATTN_FA_CORE_H
#define M15_ATTN_FA_CORE_H

#include "m15_attn_kv.h"

namespace M15FAC {

using namespace AscendC;
using namespace M15Kv;

// ============================================================
// 0. 形状（来自 config / `m15_attn_kv.h` 的权威常量，不自带第二份数字）
// ============================================================
constexpr uint32_t FAC_NH = 24;                       // q 头数（checkpoint `num_attention_heads`）
constexpr uint32_t FAC_NKV = KV_HEADS;                // 2
constexpr uint32_t FAC_G = FAC_NH / FAC_NKV;          // 12（GQA 组大小）
constexpr uint32_t FAC_HD = KV_HEAD_DIM;              // 256
constexpr uint32_t FAC_DV = KV_HEAD_DIM_V;            // 256
constexpr uint32_t FAC_P = 64;                        // q 行 / tile（donor config4 sOuter）
constexpr uint32_t FAC_SIN = 128;                     // kv 列 / tile（donor config4 sInner）
constexpr uint32_t FAC_AIV_ROWS = FAC_P / 2;          // 32：dualDstCtl 把 M 拆给两个 AIV
constexpr uint32_t FAC_BLK_TOK = KV_BLOCK_TOKENS;     // 16
constexpr uint32_t FAC_NBLK = FAC_SIN / FAC_BLK_TOK;  // 8：一个 kv tile 恰好 8 页（页对齐）
constexpr uint32_t FAC_OUT_STRIDE = FAC_NH * FAC_HD;  // 6144：输出平面行距（元素）
constexpr float FAC_SCALE = 0.0625f;                  // 256^-0.5
constexpr float FAC_MASKV = -3.38953139e38f;          // 掩码值：exp 下溢为 0

static_assert(FAC_HD == FAC_DV, "官方 head_size_v = head_dim（`m15_attn_kv.h` static_assert 同口径）");
static_assert(FAC_NH % FAC_NKV == 0u && FAC_G * FAC_NKV == FAC_NH, "GQA 分组整除");
static_assert(FAC_P % 16u == 0u && FAC_P % 2u == 0u, "Mmad M 与 dualDstCtl 分半的前提");
static_assert(FAC_SIN % 16u == 0u && FAC_SIN % FAC_BLK_TOK == 0u, "kv tile 是 16 的倍数且页对齐");
static_assert(FAC_NBLK * FAC_BLK_TOK == FAC_SIN && FAC_SIN == 128u, "一个 kv tile = 8 页");
static_assert(FAC_HD % 128u == 0u, "head_dim 拆成 2×128 k-loop（L0B 容量约束）");

// ============================================================
// 1. 同步原语（BufferID 封装；`docs/05` §6.1 / M101 同款）
// ============================================================
// 与仓库既有封装（`m15_gdn_layer.h` 的 `BufAcquire`/`BufRelease`）**同一组原语与模板实参**：
// acquire = `GetBufInternal<pipe, false>`，release = `RlsBufInternal<pipe, false>`
// （`false` = CANN `ASC_LOCK_BLOCK` 默认「阻塞」模式，与 acquire 侧同模式；M181 统一）。
// ⚠ 不用 `GetBuffImpl/ReleaseBuffImpl`：那两个符号在本 CANN 的 `kernel_tpipe.h` 里只作内部宏使用，
//   实测把它们当作 acquire/release 用会在 `Acq<PIPE_MTE3>` 处挂死（M117 实跑读数见 evidence/）。
template <pipe_t p> __aicore__ __inline__ void Acq(MutexID id) { GetBufInternal<p, false>(id); }
template <pipe_t p> __aicore__ __inline__ void Rls(MutexID id) { RlsBufInternal<p, false>(id); }

// ============================================================
// 跨核记账的**可数见证**（只在 `M25FA_WIT` 编译期开着时生效；生产路径零开销）
// 每个核每调一次 mode-2 的 set/wait，就把 `witGm[(coreSlot*32 + id)*2 + kind]` 自增 1
// （kind: 0=set, 1=wait）。coreSlot：AIC = GetBlockIdx()，AIV = 28 + GetBlockIdx()。
// 用途：kernel 挂死时由 host 在**另一条 stream** 上把这段读回来，判「两侧配平 / 不配平」。
// 计数是控制/记账、不是数据计算；且只在探针构建里编进去。
// ============================================================
#if defined(M25FA_WIT)
__aicore__ __inline__ void FacWitCount(GlobalTensor<int32_t>& g, uint32_t slot, uint32_t id, uint32_t kind)
{
    const uint64_t idx = ((uint64_t)slot * 32u + id) * 2u + kind;
    const int32_t v = g.GetValue(idx);
    g.SetValue(idx, v + 1);
}
#endif

template <pipe_t p>
__aicore__ __inline__ void FacCcSet(GlobalTensor<int32_t>& g, uint32_t slot, uint16_t id)
{
    CrossCoreSetFlag<0x2, p>(id);
#if defined(M25FA_WIT)
    FacWitCount(g, slot, id, 0u);
#else
    (void)g; (void)slot;
#endif
}

template <pipe_t p>
__aicore__ __inline__ void FacCcWait(GlobalTensor<int32_t>& g, uint32_t slot, uint16_t id)
{
    CrossCoreWaitFlag<0x2, p>(id);
#if defined(M25FA_WIT)
    FacWitCount(g, slot, id, 1u);
#else
    (void)g; (void)slot;
#endif
}

__aicore__ __inline__ constexpr uint32_t FacCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
__aicore__ __inline__ constexpr uint32_t FacMinU(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ============================================================
// 2. 核间 flagId（mode 2）+ 核内 BufferID
// ============================================================
// **单向旗标（每个逻辑通道只有一个 set 方）** —— 这是刻意的设计，不是省事：
// 早先版本让同一 id 双向走「令牌」（AIC 与 AIV 都在同一个计数器上 set/wait）。那样两侧都可能
// 消费掉**对方**或**自己**留下的令牌，AIC 会在 AIV 还没读完 S[q] 时就覆写它（真竞态），
// 且实测出现「加 printf 能跑、去掉 printf 挂死」的时序敏感挂死（M117 实跑读数见 evidence/）。
// ⇒ 改成：`*_RDY` 只由 AIC set、AIV wait；`*_FREE` 只由 AIV set、AIC wait。计数器恒守恒。
//
// **mode 2 的配对口径**（`docs/05` §2；B1/B2/M164/M121 实测）：
//   · **AIC → AIV**：AIC `set` 一次 ⇒ **配对的那 2 个 AIV** 各 `wait` 一次；
//   · **AIV → AIC**：**配对的那 2 个 AIV 都 `set`** ⇒ AIC `wait` 一次（严格 `2 set ↔ 1 wait`；
//     只让一半 AIV set，AIC 会永久等 —— M121 `kill=5` 实测 rc=124）。
// ⇒ 一个同步事件**只用一个 flagId**（删掉 mode-4 时代「AIC 侧对 AIV1 通道用 `id + 16`」那套），
//   语义 = 「本 AIC 与它那两个 AIV 的全集」。
constexpr uint16_t CC_S_RDY = 0;    // AIC -> {2 AIV}：S(t) parity0 就绪
constexpr uint16_t CC_S_FREE = 1;   // {2 AIV} -> AIC：S(t) parity0 已消费（槽位可复用）
constexpr uint16_t CC_PV_RDY = 2;   // AIC -> {2 AIV}：PV(t) parity0 就绪
constexpr uint16_t CC_PV_FREE = 3;  // {2 AIV} -> AIC：PV(t) parity0 已消费
constexpr uint16_t CC_P0 = 5;       // {2 AIV} -> AIC：P(t) parity0 已直写 L1
constexpr uint16_t CC_P1 = 6;       // {2 AIV} -> AIC：P(t) parity1 已直写 L1

static_assert(CC_S_RDY == 0u && CC_S_FREE == 1u && CC_PV_RDY == 2u && CC_PV_FREE == 3u && CC_P0 == 5u &&
                  CC_P1 == 6u,
              "cross-core flagId 清单变了 —— 融合清单 (d) 与 `m15_layer_resources.h` §4f.2 的登记必须同步");
static_assert(CC_P1 == CC_P0 + 1u, "CC_P1 是 CC_P0 的 parity1 名（同一逻辑通道）");
static_assert(CC_P1 < 16u, "mode 2 的每核 flagId 池是 0..15（不是 mode 4 的 AIC 侧 0..31）");

constexpr MutexID B_Q = 0;     // AIC：Q tile L1（MTE2 -> MTE1）
constexpr MutexID B_K = 1;     // AIC：K tile L1
constexpr MutexID B_V = 2;     // AIC：V tile L1
constexpr MutexID B_L0 = 3;    // AIC：L0A/L0B（MTE1 -> M）
constexpr MutexID B_C0 = 4;    // AIC：L0C slot0（M -> FIX）
constexpr MutexID B_C1 = 5;    // AIC：L0C slot1
constexpr MutexID B_C2 = 6;    // AIC：L0C slot2
constexpr MutexID B_C3 = 7;    // AIC：L0C slot3
constexpr MutexID B_PL1 = 8;   // AIC：P tile L1 的 MTE1 读窗（写侧已改为 AIV 跨核直写，见 §5b）

static_assert(B_Q == 0 && B_K == 1 && B_V == 2 && B_L0 == 3 && B_C0 == 4 && B_C1 == 5 && B_C2 == 6 &&
                  B_C3 == 7 && B_PL1 == 8,
              "AIC 侧 MutexID 清单变了");

constexpr MutexID B_S0 = 0;    // AIV：S(t) parity0 UB
constexpr MutexID B_S1 = 1;    // AIV：S(t) parity1 UB
constexpr MutexID B_PV0 = 2;   // AIV：PV(t) parity0 UB
constexpr MutexID B_PV1 = 3;   // AIV：PV(t) parity1 UB
constexpr MutexID B_PC = 4;    // AIV：P 落盘 + 归一化 staging 的 V -> MTE3 交接
constexpr MutexID B_ACC = 5;   // AIV：UB_ACC 窗名（**V 自己读写、无跨 pipe 消费者 ⇒ 不进令牌**）

static_assert(B_S0 == 0 && B_S1 == 1 && B_PV0 == 2 && B_PV1 == 3 && B_PC == 4 && B_ACC == 5,
              "AIV 侧 MutexID 清单变了");

// ============================================================
// 3. L1 / L0A / L0B / L0C 静态布局（AIC）
// ============================================================
constexpr uint32_t FAC_L1_Q_BYTES = FAC_P * FAC_HD * 2u;      // 32 KB
constexpr uint32_t FAC_L1_K_BYTES = FAC_SIN * FAC_HD * 2u;    // 64 KB
constexpr uint32_t FAC_L1_V_BYTES = FAC_SIN * FAC_DV * 2u;    // 64 KB
constexpr uint32_t FAC_L1_P_BYTES = FAC_P * FAC_SIN * 2u;     // 16 KB
constexpr uint32_t FAC_L1_Q_OFF = 0;
constexpr uint32_t FAC_L1_K_OFF = FAC_L1_Q_OFF + FAC_L1_Q_BYTES;
constexpr uint32_t FAC_L1_V_OFF = FAC_L1_K_OFF + FAC_L1_K_BYTES;
constexpr uint32_t FAC_L1_P_OFF = FAC_L1_V_OFF + FAC_L1_V_BYTES;
constexpr uint32_t FAC_L1_TOTAL = FAC_L1_P_OFF + 2u * FAC_L1_P_BYTES;
static_assert(FAC_L1_TOTAL == 196608u, "L1 峰值 = 192 KB");
static_assert(FAC_L1_TOTAL <= 512u * 1024u, "L1 overflow");

constexpr uint32_t FAC_L0C_SLOT_BYTES = FAC_P * FAC_SIN * 4u;  // 32 KB（fp32 [64,128]）
static_assert(4u * FAC_L0C_SLOT_BYTES <= 256u * 1024u, "L0C overflow（4 slot = 128 KB）");
// A 操作数在 L0A 里的**元素**偏移：Q [64,256] NZ = 16 k-fractal × 64 行 × 16 = 16384 元素 = 32 KB
constexpr uint32_t FAC_L0A_Q_ELEMS = 0;
constexpr uint32_t FAC_L0A_P_ELEMS = FAC_HD / 16u * FAC_P * 16u;   // 16384
static_assert(FAC_L0A_P_ELEMS == 16384u, "P 在 L0A 的落点 = Q 之后（32 KB 处）");
static_assert((FAC_L0A_P_ELEMS + FAC_SIN / 16u * FAC_P * 16u) * 2u <= 64u * 1024u, "L0A overflow");

// ============================================================
// 4. UB 静态布局（AIV）。全部 32 B 对齐。
// ============================================================
constexpr uint32_t FAC_UB_S_BYTES = FAC_AIV_ROWS * FAC_SIN * 4u;    // 16 KB
constexpr uint32_t FAC_UB_PV_BYTES = FAC_AIV_ROWS * FAC_DV * 4u;    // 32 KB
constexpr uint32_t FAC_UB_S0_OFF = 0;
constexpr uint32_t FAC_UB_S1_OFF = FAC_UB_S0_OFF + FAC_UB_S_BYTES;
constexpr uint32_t FAC_UB_PV0_OFF = FAC_UB_S1_OFF + FAC_UB_S_BYTES;
constexpr uint32_t FAC_UB_PV1_OFF = FAC_UB_PV0_OFF + FAC_UB_PV_BYTES;
constexpr uint32_t FAC_UB_PC_OFF = FAC_UB_PV1_OFF + FAC_UB_PV_BYTES;
constexpr uint32_t FAC_UB_PC_BYTES = FAC_AIV_ROWS * FAC_DV * 2u;    // 16 KB（P 落盘只用前 8 KB）
constexpr uint32_t FAC_UB_ACC_OFF = FAC_UB_PC_OFF + FAC_UB_PC_BYTES;
constexpr uint32_t FAC_UB_ACC_BYTES = FAC_AIV_ROWS * FAC_DV * 4u;   // 32 KB
// 归一化的 bf16 staging **复用 UB_PC**（8 KB = 16 行 × 256 bf16）：那条 `V 写 -> release(mode=false) ->
// MTE3 搬` 的路在 P 落盘里每 tile 跑一次，是本档实测**跑通**的那条（时序敏感挂死的记录见 evidence/）。
constexpr uint32_t FAC_UB_THR_OFF = FAC_UB_ACC_OFF + FAC_UB_ACC_BYTES;
constexpr uint32_t FAC_UB_THR_BYTES = 256u;                         // 64 lane fp32：逐行掩码阈值向量
constexpr uint32_t FAC_UB_M_OFF = FAC_UB_THR_OFF + FAC_UB_THR_BYTES;
constexpr uint32_t FAC_UB_SUM_OFF = FAC_UB_M_OFF + FAC_AIV_ROWS * 4u;
constexpr uint32_t FAC_UB_ED_OFF = FAC_UB_SUM_OFF + FAC_AIV_ROWS * 4u;
constexpr uint32_t FAC_UB_TOTAL = FAC_UB_ED_OFF + FAC_AIV_ROWS * 4u;
static_assert(FAC_UB_S1_OFF % 32u == 0u && FAC_UB_PV0_OFF % 32u == 0u && FAC_UB_PV1_OFF % 32u == 0u &&
                  FAC_UB_PC_OFF % 32u == 0u && FAC_UB_ACC_OFF % 32u == 0u &&
                  FAC_UB_THR_OFF % 32u == 0u && FAC_UB_M_OFF % 32u == 0u && FAC_UB_SUM_OFF % 32u == 0u &&
                  FAC_UB_ED_OFF % 32u == 0u,
              "UB 窗必须 32 B 对齐（DataCopy / Fixpipe 的硬要求）");
static_assert(FAC_UB_TOTAL == 148096u, "UB 峰值 = 144.6 KB");
static_assert(FAC_UB_TOTAL < 248u * 1024u, "UB overflow");

// ============================================================
// 5. GM -> L1 的 Nd2Nz（行主序 -> NZ）；`ndNum` 个矩阵沿 N 方向拼接
//    三处用法：Q tile（4 个 16 行矩阵）、K/V tile（8 个页矩阵）、P tile（4 个 16 行矩阵）
// ============================================================
// 逐矩阵版：`ndNum = 1` 反复调（M101 只有这一种形态，仓库已实证）；
// 用于**装载几何**那一档的对照实验（`FacGmToL1Nz` 的多矩阵形态是本档自己拼的）。
// 目标 NZ 布局 = [K/16][N][16]；矩阵 m 占 N 方向 [16m, 16m+16) ⇒ 元素偏移 256*m。
template <typename T>
__aicore__ __inline__ void FacGmToL1NzOneMatrix(const LocalTensor<uint8_t>& dstL1, const GlobalTensor<T>& srcGm,
                                                uint64_t gmElemOff, uint32_t cols, uint32_t srcRowPitch,
                                                uint32_t dstRowsTotal, uint32_t dstElemOff)
{
    Nd2NzParams par;
    par.ndNum = 1;
    par.nValue = 16;
    par.dValue = cols;
    par.srcNdMatrixStride = 0;
    par.srcDValue = (uint64_t)srcRowPitch;
    par.dstNzC0Stride = (uint16_t)dstRowsTotal;
    par.dstNzNStride = 1;
    par.dstNzMatrixStride = 0;
    DataCopy(dstL1.template ReinterpretCast<T>()[dstElemOff], srcGm[gmElemOff], par);
}

template <typename T>
__aicore__ __inline__ void FacGmToL1Nz(const LocalTensor<uint8_t>& dstL1, const GlobalTensor<T>& srcGm,
                                       uint64_t gmElemOff, uint32_t ndNum, uint32_t rowsPerMatrix, uint32_t cols,
                                       uint64_t srcMatStride, uint32_t srcRowPitch, uint32_t dstRowsTotal)
{
    Nd2NzParams par;
    par.ndNum = (uint16_t)ndNum;
    par.nValue = (uint16_t)rowsPerMatrix;
    par.dValue = cols;
    par.srcNdMatrixStride = srcMatStride;
    par.srcDValue = (uint64_t)srcRowPitch;
    par.dstNzC0Stride = (uint16_t)dstRowsTotal;
    par.dstNzNStride = 1;
    par.dstNzMatrixStride = (uint32_t)(16u * 16u);
    DataCopy(dstL1.template ReinterpretCast<T>(), srcGm[gmElemOff], par);
}

// ============================================================
// 6. L1 -> L0 的 2D 装载。字段表取自 donor `attention/common/op_kernel/matmul.h` 的
//    `LoadDataToL0A/LoadDataToL0B`（非转置分支 / 转置分支），m11 已实证同一组字段语义。
// ============================================================
template <typename T>
__aicore__ __inline__ void FacLoadL0_2D(const LocalTensor<uint8_t>& dstL0, const LocalTensor<uint8_t>& srcL1,
                                        uint64_t l1ElemOff, uint32_t mStep, uint32_t kStep, uint32_t srcStride,
                                        uint32_t dstStride, bool ifTranspose, uint64_t dstElemOff)
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

// ============================================================
// 6b. AIV：把 P 半块从 UB（行主序 [FAC_AIV_ROWS, FAC_SIN] bf16）**直写 L1** 的 NZ 槽
//     —— 目标布局与 AIC 的 `FacLoadL0_2D`（§6）逐字对齐，因此 AIC 侧那条装载**一字不改**。
//
// L1P 目标 = `[FAC_P, FAC_SIN]` 的**标准 NZ**：元素 (R, c) 落在 512B 单元
//   `((R/16) + (FAC_P/16)*(c/16))`，单元内 `(R%16)*16 + (c%16)`。
//   （与旧 GM 路径 `FacGmToL1Nz` 产出的完全同构 —— 只是搬运方从 AIC 的 MTE2 改成 AIV 直写。）
// 本 AIV 只负责 `R ∈ [half*AIV_ROWS, +AIV_ROWS)`（dualDstCtl 的 M 半块）：对每个列分形 c1，
// 把 AIV_ROWS 个 32B（源行距 FAC_SIN 元素）**连续**写到单元 `(half*(AIV_ROWS/16) + (FAC_P/16)*c1)`
// 起（该 AIV 的两个行分形相邻 ⇒ 目的连续 1024B）。
//
// 通路 = **UB→L1 硬件通道**（本平台 3510 真机实测可用；`DataCopy(dstL1, srcUb, DataCopyParams)`
// 在 `__mix__(1,2)` 下走 `copy_ubuf_to_cbuf`，不经 GM、不需 Matmul 注册）—— 依据 M121 的
// `probe_ub2l1`（`docs/evidence/ub_to_l1/README.md` §3/§5）。**随路 ND2NZ 硬件不支持**
// （同 README §5.3）⇒ ND→NZ 由本函数在搬运时用「逐列分形的 burst 拷贝」表达。
// ============================================================
__aicore__ __inline__ void FacUbToL1P(const LocalTensor<uint8_t>& dstL1, const LocalTensor<bfloat16_t>& srcUb,
                                      uint32_t half)
{
    LocalTensor<bfloat16_t> d = dstL1.template ReinterpretCast<bfloat16_t>();
    // burst 的单位是 32B（= 16 个 bf16）；源行距 FAC_SIN 元素 = FAC_SIN*2/32 个 32B，gap = 减 1
    const uint16_t srcGap = (uint16_t)(FAC_SIN * 2u / 32u - 1u);
    for (uint32_t c1 = 0; c1 < FAC_SIN / 16u; ++c1) {
        const uint32_t dstElem = (half * (FAC_AIV_ROWS / 16u) + (FAC_P / 16u) * c1) * 256u;
        DataCopy(d[dstElem], srcUb[c1 * 16u], DataCopyParams{(uint16_t)FAC_AIV_ROWS, 1u, srcGap, 0u});
    }
    PipeBarrier<PIPE_MTE3>();
}

// ============================================================
// 7. Mmad（bf16 操作数 + fp32 累加；C = A·B，B 为 NZ [N,K]）
// ============================================================
__aicore__ __inline__ void FacMmadBf16(const LocalTensor<uint8_t>& cL0C, const LocalTensor<uint8_t>& aL0,
                                       const LocalTensor<uint8_t>& bL0, uint64_t aElemOff, uint32_t m, uint32_t n,
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
// 8. Fixpipe L0C -> UB（dualDstCtl=1：M 拆两半，各写一个 AIV 的 UB）
//    reluScalar / vectorRelu / deqScalar 无 NSDMI，必须显式清零（M101 的 M16/M19 加固）
// ============================================================
__aicore__ __inline__ void FacFixpToUb(const LocalTensor<uint8_t>& dstUb, const LocalTensor<uint8_t>& srcL0C,
                                       uint32_t nSize, uint32_t dstElemOff, uint32_t dstStride)
{
    FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp;
    fp.nSize = (uint16_t)nSize;
    fp.mSize = (uint16_t)FAC_P;
    fp.srcStride = (uint16_t)FAC_P;
    fp.dstStride = (uint16_t)dstStride;
    fp.dualDstCtl = 1;
    fp.reluScalar = 0;
    fp.vectorRelu = 0;
    fp.deqScalar = 0;
    fp.params.ndNum = 1;
    fp.params.srcNdStride = 0;
    fp.params.dstNdStride = 0;
    static constexpr FixpipeConfig kFixUb(CO2Layout::ROW_MAJOR, true);
    LocalTensor<float> src = srcL0C.template ReinterpretCast<float>();
    LocalTensor<float> dst = dstUb.template ReinterpretCast<float>()[dstElemOff];
    Fixpipe<float, float, kFixUb>(dst, src, fp);
}

// ============================================================
// 9. 工作项切分（AIC/AIV 两侧必须逐字一致）
//    工作项 wi = (h, qTile)：`h = wi / nQTile`、`qTile = wi % nQTile`；
//    每个 block 处理 `wi = bid, bid+nBlk, ...`（AIV 用 `bid>>1`）。
// ============================================================
struct FacWork {
    uint32_t nQTile;      // ceil(m / P)
    uint32_t nWork;
    uint32_t rowBase;     // qTile * P
    uint32_t h;           // q 头
    uint32_t n2;          // kv 头 = h / G
    uint32_t nTiles;      // 本工作项的 kv tile 数（causal 上界，被 ctx 夹住）
};

__aicore__ __inline__ void FacMakeWork(uint32_t wi, uint32_t m, uint32_t posBase, uint32_t ctx, FacWork& w)
{
    w.nQTile = FacCeilDiv(m, FAC_P);
    w.nWork = FAC_NH * w.nQTile;
    w.h = wi / w.nQTile;
    const uint32_t qTile = wi % w.nQTile;
    w.rowBase = qTile * FAC_P;
    w.n2 = w.h / FAC_G;
    // causal 上界：本工作项**最后一行**的位置决定要扫到哪个 kv tile；同时被 ctx 夹住
    // （不许读 KV 平面之外）。前面行位置更小 ⇒ 由逐行掩码负责。
    uint32_t hi = posBase + w.rowBase + FAC_P;
    if (hi > ctx) { hi = ctx; }
    w.nTiles = FacCeilDiv(hi, FAC_SIN);
}

// 本 tile 内**已装入 L1** 的页数（16-token 粒度）。掩码一律夹在这里：未装入的列在 S 上被掩成
// MASKV，在 PV 里根本不参与（BMM2 的 K 轴 = `colLimit`）。
__aicore__ __inline__ uint32_t FacBlocksInTile(uint32_t colBase, uint32_t ctx)
{
    if (ctx <= colBase) { return 0u; }
    const uint32_t nb = FacCeilDiv(ctx - colBase, FAC_BLK_TOK);
    return (nb < FAC_NBLK) ? nb : FAC_NBLK;
}

// ============================================================
// 10. AIV 侧 VF（RegBase；m4 / m5 / m12 / M101 同一范式）
// ============================================================
constexpr uint32_t FAC_VL_F32 = 64;                       // 256 B / 4 B
constexpr uint32_t FAC_SIN_VL = FAC_SIN / FAC_VL_F32;     // 2
constexpr uint32_t FAC_DV_VL = FAC_DV / FAC_VL_F32;       // 4
static_assert(FAC_SIN_VL * FAC_VL_F32 == FAC_SIN && FAC_DV_VL * FAC_VL_F32 == FAC_DV, "VF 分组整除");

constexpr Reg::CastTrait FAC_CAST_B322B16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// 逐行**自带三角掩码**（不靠 host 预置 mask、不靠 packed indices）：
//   行 r 的位置 p = `posLo + r`；阈值 `thr_r = min(p − colBase, colLimit−1)`
//   ⇒ 有效列 = `colBase + lane ≤ thr_r`（chunk1 即 `lane ≤ thr_r − colBase − 64`）。
//   两个**编译器限制**决定了它的写法（M117 实测，bisheng 15.0.5 / dav-3510）：
//     ① `__VEC_SCOPE__` 的**循环内不允许标量算术**（`(float)r + base` 一类会报
//        `Unsupported scalar instruction in AIV loop`）⇒ 阈值必须先在向量域里做好；
//     ② `Compares<>` 的第二操作数**只能是标量字面量**（传 RegTensor 会报 `vcmps_le` 无匹配）
//        ⇒ 用 `Sub(d, Arange, thrBcast)` 把比较转成「`d ≤ 0` / `d ≤ −64`」两个字面量比较。
//   于是：`thrV = min(Arange + thrBase, colHiF)` 一次算好落 UB，循环内只做 `DIST_BRC_B32` 广播。
// `Causal=false` **只用于负向对照**（把被测对象的掩码弄坏 ⇒ 判据必须变红），生产路径只用默认 `true`。
// 掩码阈值：**按塔裁 2026-10-04 从严** —— `posLo/colBase/colLimit` 三个**下标量原样传入**，
// 阈值的组合**全部在 VF 里用向量指令做**（VF 外不做任何算术；VF 内只有 index→float 的取数转换）。
// 与旧写法的语义**等价**：
//   旧： valid ⟺ `lane ≤ min(posLo + r − colBase, colLimit−1)`
//   新： `nthr[j] = colBase − posLo − j`（= −(posLo+j−colBase)）向量化造出，
//        逐行 `d = lane + nthr[r]` ⇒ `d ≤ 0` 就是 `lane ≤ posLo+r−colBase`；
//        再把行无关的 `lane < colLimit` 作为**独立的行无关掩码**叠上去（第二层 Select），
//        两者相与 ⟺ 旧式的 min 夹取（`colLimit` 是 16 的倍数 ⇒ 用 `<` 严格小于即可）。
template <bool Causal>
__simd_vf__ inline void FacSoftmaxTileVf(__ubuf__ float* sUb, __ubuf__ bfloat16_t* pUb, __ubuf__ float* thrUb,
                                         __ubuf__ float* mUb, __ubuf__ float* sumUb, __ubuf__ float* edUb,
                                         uint32_t posLo, uint32_t colBase, uint32_t colLimit)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> s0, s1, t, mn, mo, ed, sg, idx, thrV, thr, d, msk;
        RegTensor<bfloat16_t> pb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        MaskReg oneM = CreateMask<float, MaskPattern::VL1>();
        MaskReg cmp = CreateMask<float, MaskPattern::ALL>();
        MaskReg lim0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg lim1 = CreateMask<float, MaskPattern::ALL>();
        if constexpr (Causal) {
            Duplicate(msk, FAC_MASKV, fullM);
            // nthr[j] = colBase − posLo − j（全程向量指令；只有两个 index→float 取数转换）
            Arange(thrV, 0.0f);
            Adds(thrV, thrV, (float)posLo, fullM);      // j + posLo
            Muls(thrV, thrV, -1.0f, fullM);             // −(j + posLo)
            Adds(thrV, thrV, (float)colBase, fullM);    // colBase − j − posLo
            StoreAlign<float>(thrUb, thrV, fullM);
            // 行无关的「已装入列」掩码：lane < colLimit / lane + 64 < colLimit（colLimit 是 16 的倍数）
            Arange(idx, 0.0f);
            Compares<float, CMPMODE::LT>(lim0, idx, (float)colLimit, fullM);
            Adds(idx, idx, 64.0f, fullM);
            Compares<float, CMPMODE::LT>(lim1, idx, (float)colLimit, fullM);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t r = 0; r < (uint16_t)FAC_AIV_ROWS; ++r) {
            const uint32_t o = (uint32_t)r * FAC_SIN;
            LoadAlign(s0, sUb + o);
            LoadAlign(s1, sUb + o + FAC_VL_F32);
            Muls(s0, s0, FAC_SCALE, fullM);
            Muls(s1, s1, FAC_SCALE, fullM);
            if constexpr (Causal) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(thr, thrUb + r);  // −thr_r
                Arange(idx, 0.0f);
                Add(d, idx, thr, fullM);                                  // d = lane − thr_r
                Compares<float, CMPMODE::LE>(cmp, d, 0.0f, fullM);        // lane ≤ thr_r
                Select(s0, s0, msk, cmp);
                Select(s0, s0, msk, lim0);                                // ∧ lane < colLimit
                Compares<float, CMPMODE::LE>(cmp, d, -64.0f, fullM);      // lane+64 ≤ thr_r
                Select(s1, s1, msk, cmp);
                Select(s1, s1, msk, lim1);
            }
            // ---- online softmax：mNew = max(mOld, rowMax)；ed = exp(mOld−mNew)；P = exp(S−mNew) ----
            Max(t, s0, s1, fullM);
            Reduce<ReduceType::MAX>(mn, t, fullM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mo, mUb + r);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(mUb + r, mn, oneM);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mn, mUb + r);
            Max(mn, mn, mo, fullM);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(mUb + r, mn, oneM);
            Sub(ed, mo, mn, fullM);
            Exp(ed, ed, fullM);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(edUb + r, ed, oneM);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(sg, sumUb + r);
            Mul(sg, sg, ed, fullM);
            Sub(s0, s0, mn, fullM);
            Exp(s0, s0, fullM);
            Sub(s1, s1, mn, fullM);
            Exp(s1, s1, fullM);
            Reduce<ReduceType::SUM>(t, s0, fullM);
            Add(sg, sg, t, fullM);
            Reduce<ReduceType::SUM>(t, s1, fullM);
            Add(sg, sg, t, fullM);
            StoreAlign<float, StoreDist::DIST_FIRST_ELEMENT_B32>(sumUb + r, sg, oneM);
            Cast<bfloat16_t, float, FAC_CAST_B322B16>(pb, s0, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o, pb, fullM);
            Cast<bfloat16_t, float, FAC_CAST_B322B16>(pb, s1, fullM);
            StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(pUb + o + FAC_VL_F32, pb, fullM);
        }
    }
}

// acc = acc·ed + PV（整 tile；PV 来自 AIC BMM2 的 Fixpipe 直写）
__simd_vf__ inline void FacAccUpdateVf(__ubuf__ float* accUb, __ubuf__ float* pvUb, __ubuf__ float* edUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> acc, pv, ed;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0; r < (uint16_t)FAC_AIV_ROWS; ++r) {
            const uint32_t o = (uint32_t)r * FAC_DV;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ed, edUb + r);
            for (uint16_t g = 0; g < (uint16_t)FAC_DV_VL; ++g) {
                LoadAlign(acc, accUb + o + (uint32_t)g * FAC_VL_F32);
                LoadAlign(pv, pvUb + o + (uint32_t)g * FAC_VL_F32);
                Mul(acc, acc, ed, fullM);
                Add(acc, acc, pv, fullM);
                StoreAlign(accUb + o + (uint32_t)g * FAC_VL_F32, acc, fullM);
            }
        }
    }
}

// 行状态初值：m = MASKV（exp(mOld−mNew)=0，等价 −inf）、sum = 0、acc = 0。
// **整段走 RegBase**（`docs/05` §6.1：经典 API 的标量↔向量 UB 交接在本平台不可靠；M101 用的是
// 经典 `Duplicate`，本段刻意不跟）。
__simd_vf__ inline void FacInitStateVf(__ubuf__ float* mUb, __ubuf__ float* sumUb, __ubuf__ float* accUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> v, z;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        MaskReg m32 = CreateMask<float, MaskPattern::VL32>();
        Duplicate(v, FAC_MASKV, fullM);
        StoreAlign<float>(mUb, v, m32);
        Duplicate(z, 0.0f, fullM);
        StoreAlign<float>(sumUb, z, m32);
        for (uint16_t g = 0; g < (uint16_t)(FAC_AIV_ROWS * FAC_DV / FAC_VL_F32); ++g) {
            StoreAlign<float>(accUb + (uint32_t)g * FAC_VL_F32, z, fullM);
        }
    }
}

// out = acc / sum（fp32 → bf16 staging）。整 32 行一次写入 16 KB 的 UB_PC。
__simd_vf__ inline void FacNormalizeVf(__ubuf__ float* accUb, __ubuf__ bfloat16_t* outUb, __ubuf__ float* sumUb)
{
    using namespace AscendC::Reg;
    __VEC_SCOPE__
    {
        RegTensor<float> a, s;
        RegTensor<bfloat16_t> pb;
        MaskReg fullM = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t r = 0; r < (uint16_t)FAC_AIV_ROWS; ++r) {
            const uint32_t o = (uint32_t)r * FAC_DV;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(s, sumUb + r);
            for (uint16_t g = 0; g < (uint16_t)FAC_DV_VL; ++g) {
                LoadAlign(a, accUb + o + (uint32_t)g * FAC_VL_F32);
                Div(a, a, s, fullM);
                Cast<bfloat16_t, float, FAC_CAST_B322B16>(pb, a, fullM);
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(outUb + o + (uint32_t)g * FAC_VL_F32, pb, fullM);
            }
        }
    }
}

// ============================================================
// 11. AIC：两次 mmad（QKᵀ / PV）
//     每 tile 的两步流水（tile t 的 parity q = t&1）：
//       (A) 等 CC_S[q] 令牌（配对 2 AIV 都已消费）-> K 装载 -> Mmad S -> Fixpipe UB_S[q] -> set CC_S[q]
//       (B) 等 CC_P[q]（配对 2 AIV 都已把 P(t) 直写进 L1P(q)）+ 等 CC_PV[q] 令牌
//           -> P（L1 -> L0A，不落 GM）/ V 装载 -> Mmad PV -> Fixpipe UB_PV[q] -> set CC_PV[q]
//     核间是 **mode 2**：本核的每个 set 放行配对 2 个 AIV 的 wait；反之两个 AIV 都 set 后本核的
//     wait 才放行（严格 2 set ↔ 1 wait，见 §2）。S/PV 令牌的 init 由 AIV 在跑第一 tile 前预置
//     （**两个 AIV 各置一份**）⇒ 两侧计数器守恒。
//
//     ⚠ **M189 根因（L1→L0B 的 BufferID 必须逐半块交接）**：BMM1 的 `ks` 循环与 BMM2 的 `nh`
//     循环里，`FacLoadL0_2D`（MTE1 写 L0B）与 `Mmad`（M 读 L0B）**必须每个半块成对
//     `Acq/Rls<PIPE_MTE1>`**：生产侧 MTE1 装载完**立即释放**，消费侧 M 才 acquire（= donor
//     `m10_attn_decode` BMM1 的写法）。**把 MTE1 的 `Acq/Rls` 提到循环外是一次未定位的系统性
//     数值错**（M185 只看到 S 错、PV 错，没找到这里）：M 侧两条 `Mmad` 会都等到循环结束后的
//     释放、从而都读**最后一次** L0B ── BMM1 两个 k 半块都用第二个半块的 K，BMM2 两个维半块
//     都用第二个半块的 V。修好后 `check_ref.py` 四档 `over=0`（见 `evidence/M189_readings.txt`）。
// ============================================================
class FacAic {
public:
    __aicore__ inline FacAic() {}

    __aicore__ inline void Init(__gm__ uint8_t* q, uint32_t qStride, __gm__ uint8_t* kv, __gm__ uint8_t* out,
                                __gm__ uint8_t* pScratch, __gm__ uint8_t* witScratch)
    {
        witGm_.SetGlobalBuffer((__gm__ int32_t*)witScratch);
        qGM_.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        kvGM_.SetGlobalBuffer((__gm__ bfloat16_t*)kv);
        pGM_.SetGlobalBuffer((__gm__ bfloat16_t*)pScratch);
        qStride_ = qStride;
        (void)out;
    }

    __aicore__ inline void Run(uint32_t m, uint32_t posBase, uint32_t ctx, uint32_t lane, uint32_t nBlk)
    {
        const uint32_t bid = GetBlockIdx();
        const uint32_t nQTile = FacCeilDiv(m, FAC_P);
        const uint32_t nWork = FAC_NH * nQTile;
#if defined(M25FA_AICDMA)
        // 【最小探针】AIC（cube 核）到底能不能做 UB→GM 的 DMA？
        // 这段只在 AIC 目标里编译（`FacAic::Run` 由 `FaCoreBody` 的 `if ASCEND_IS_AIC` 分支调用）
        // ⇒ 编译通过本身就是一个读数；落盘与否由 host 经另一条 stream 读回核对。
        {
            LocalTensor<uint8_t> ubProbe(TPosition::VECCALC, 0x8000u, 1024u);
            DataCopy(ubProbe, pGM_.template ReinterpretCast<uint8_t>(), DataCopyParams{1, 32, 0, 0});
            PipeBarrier<PIPE_MTE2>();
            DataCopy(witGm_.template ReinterpretCast<uint8_t>()[4096], ubProbe, DataCopyParams{1, 32, 0, 0});
            PipeBarrier<PIPE_MTE3>();
        }
#endif
        FacWork w;
        for (uint32_t wi = bid; wi < nWork; wi += nBlk) {
            FacMakeWork(wi, m, posBase, ctx, w);
            RunWorkItem(w, ctx, lane);
        }
    }

private:
    __aicore__ inline MutexID L0CId(uint32_t slot)
    {
        switch (slot & 3u) {
            case 0: return B_C0;
            case 1: return B_C1;
            case 2: return B_C2;
            default: return B_C3;
        }
    }
    __aicore__ inline LocalTensor<uint8_t> Slot(const LocalTensor<uint8_t>& l0C, MutexID id)
    {
        uint32_t off = 0;
        switch (id) {
            case B_C1: off = FAC_L0C_SLOT_BYTES; break;
            case B_C2: off = 2u * FAC_L0C_SLOT_BYTES; break;
            case B_C3: off = 3u * FAC_L0C_SLOT_BYTES; break;
            default: off = 0; break;
        }
        return l0C[off];
    }
    __aicore__ inline LocalTensor<uint8_t> UbS(uint32_t q)
    {
        return LocalTensor<uint8_t>(TPosition::VECIN, q ? FAC_UB_S1_OFF : FAC_UB_S0_OFF, FAC_UB_S_BYTES);
    }
    __aicore__ inline LocalTensor<uint8_t> UbPV(uint32_t q)
    {
        return LocalTensor<uint8_t>(TPosition::VECIN, q ? FAC_UB_PV1_OFF : FAC_UB_PV0_OFF, FAC_UB_PV_BYTES);
    }
    // P tile 的 L1 双缓冲：q 选择 parity（避免非 const 引用在三元里丢失 left-value 语义）
    __aicore__ inline LocalTensor<uint8_t> L1P(uint32_t q) const
    {
        return LocalTensor<uint8_t>(TPosition::A1, q ? (FAC_L1_P_OFF + FAC_L1_P_BYTES) : FAC_L1_P_OFF,
                                    FAC_L1_P_BYTES);
    }

    __aicore__ inline void RunWorkItem(const FacWork& w, uint32_t ctx, uint32_t lane)
    {
        if (w.nTiles == 0u) { return; }

        LocalTensor<uint8_t> l1Q(TPosition::A1, FAC_L1_Q_OFF, FAC_L1_Q_BYTES);
        LocalTensor<uint8_t> l1K(TPosition::A1, FAC_L1_K_OFF, FAC_L1_K_BYTES);
        LocalTensor<uint8_t> l1V(TPosition::A1, FAC_L1_V_OFF, FAC_L1_V_BYTES);
        LocalTensor<uint8_t> l0A(TPosition::A2, 0, 64u * 1024u);
        LocalTensor<uint8_t> l0B(TPosition::B2, 0, 64u * 1024u);
        LocalTensor<uint8_t> l0C(TPosition::CO1, 0, 4u * FAC_L0C_SLOT_BYTES);

        // 层基址：**只用 `m15_attn_kv.h` 的权威常量**（`KvLayerOffset()` 是 host-only inline，
        // 设备侧不能调；算式 = `k * KV_LAYER_STRIDE`，不是第二份数字）
        const uint64_t kvLayerElem = (uint64_t)lane * M15Kv::KV_LAYER_STRIDE / M15Kv::KV_ELEM_BYTES;

        // ---- prologue：Q tile 常驻 L1 + L0A ----
        // 同 pipe 背靠背复用同一 buffer 不保证完成序（M126 真机读数：`PIPE_MTE2` 上第二次 get 不保证
        // 第一次 op 已结束；官方 `Lock.md` 同向）⇒ 在两次同 buffer op 之间排空该 pipe。
        PipeBarrier<PIPE_MTE2>();
        Acq<PIPE_MTE2>(B_Q);
        FacGmToL1Nz<bfloat16_t>(l1Q, qGM_, (uint64_t)w.rowBase * qStride_ + (uint64_t)w.h * FAC_HD, FAC_P / 16u, 16u,
                                FAC_HD, (uint64_t)16u * qStride_, qStride_, FAC_P);
        Rls<PIPE_MTE2>(B_Q);
        Acq<PIPE_MTE1>(B_L0);
        Acq<PIPE_MTE1>(B_Q);
        // Q [64,256] -> L0A：mStep = P/16、kStep = HD/16、srcStride = dstStride = mStep（非转置分支）
        FacLoadL0_2D<bfloat16_t>(l0A, l1Q, 0, FAC_P / 16u, FAC_HD / 16u, FAC_P / 16u, FAC_P / 16u, false,
                                 FAC_L0A_Q_ELEMS);
        Rls<PIPE_MTE1>(B_Q);
        Rls<PIPE_MTE1>(B_L0);

        for (uint32_t t = 0; t < w.nTiles; ++t) {
            const uint32_t q = t & 1u;
            const uint32_t colBase = t * FAC_SIN;
            const uint32_t nb = FacBlocksInTile(colBase, ctx);
            const uint32_t colLimit = nb * FAC_BLK_TOK;

            // ================= (A) S(t) = Q · Kᵀ =================
            // mode 2：配对 2 个 AIV 都 set 后本 wait 才放行（2 set ↔ 1 wait）
            FacCcWait<PIPE_FIX>(witGm_, GetBlockIdx(), CC_S_FREE + q);   // S[q] 槽位空闲（配对 2 AIV 都已消费）
            {
                PipeBarrier<PIPE_MTE2>();
                Acq<PIPE_MTE2>(B_K);
                const uint64_t kOffElem =
                    kvLayerElem + M15KV_KV_BYTE_OFF_CONTIG(colBase, w.n2, M15Kv::KV_LANE_K, 0u) /
                                      M15Kv::KV_ELEM_BYTES;
                // K tile [128 行, 256 维]：nb 个页矩阵；页 stride = KV_BLOCK_BYTES，行距 = KV_TOKEN_STRIDE
#if defined(M25FA_GEO_ONEMAT)
                for (uint32_t b = 0; b < nb; ++b) {
                    FacGmToL1NzOneMatrix<bfloat16_t>(
                        l1K, kvGM_,
                        kOffElem + (uint64_t)b * (M15Kv::KV_BLOCK_BYTES / M15Kv::KV_ELEM_BYTES),
                        FAC_HD, (uint32_t)(M15Kv::KV_TOKEN_STRIDE / M15Kv::KV_ELEM_BYTES), FAC_SIN, 256u * b);
                }
#else
                FacGmToL1Nz<bfloat16_t>(l1K, kvGM_, kOffElem, nb, FAC_BLK_TOK, FAC_HD,
                                        M15Kv::KV_BLOCK_BYTES / M15Kv::KV_ELEM_BYTES,
                                        (uint32_t)(M15Kv::KV_TOKEN_STRIDE / M15Kv::KV_ELEM_BYTES), FAC_SIN);
#endif
                Rls<PIPE_MTE2>(B_K);
                const MutexID bC = L0CId(2u * q);
#if defined(M25FA_K1MAD)
                // 【k 拆分对照档】BMM1 的 k 由「2×128 两次 mmad」改成「1×256 一次 mmad」
                PipeBarrier<PIPE_MTE1>();
                Acq<PIPE_MTE1>(B_L0);
                Acq<PIPE_MTE1>(B_K);
                FacLoadL0_2D<bfloat16_t>(l0B, l1K, 0, FAC_SIN / 16u, FAC_HD / 16u, FAC_SIN / 16u, FAC_SIN / 16u,
                                         false, 0);
                Rls<PIPE_MTE1>(B_K);
                Rls<PIPE_MTE1>(B_L0);
                Acq<PIPE_M>(B_L0);
                Acq<PIPE_M>(B_K);
                Acq<PIPE_M>(bC);
                FacMmadBf16(Slot(l0C, bC), l0A, l0B, 0, FAC_P, FAC_SIN, FAC_HD, true);
                Rls<PIPE_M>(B_K);
                Rls<PIPE_M>(B_L0);
#else
                for (uint32_t ks = 0; ks < 2u; ++ks) {
                    PipeBarrier<PIPE_M>();   // ks=1 会紧接着重新 get 同一 B_L0/B_K（同 pipe 同 id 背靠背）
                    // ⚠ M189：MTE1 侧每个 k 半块**装载后立即 release**，M 侧才 acquire ⇒ 两条 Mmad
                    //    各自读自己的 L0B。把这两对 Acq/Rls 提到循环外会让两条 Mmad 都读到 ks=1 的 L0B。
                    PipeBarrier<PIPE_MTE1>();
                    Acq<PIPE_MTE1>(B_L0);
                    Acq<PIPE_MTE1>(B_K);
                    // B 操作数 = K [N=128 行, K=128 维半块]（非转置分支）
                    FacLoadL0_2D<bfloat16_t>(l0B, l1K, (uint64_t)ks * (128u / 16u) * FAC_SIN * 16u,
                                             FAC_SIN / 16u, 128u / 16u, FAC_SIN / 16u, FAC_SIN / 16u, false, 0);
                    Rls<PIPE_MTE1>(B_K);
                    Rls<PIPE_MTE1>(B_L0);
                    Acq<PIPE_M>(B_L0);
                    Acq<PIPE_M>(B_K);
                    if (ks == 0u) { Acq<PIPE_M>(bC); }
                    FacMmadBf16(Slot(l0C, bC), l0A, l0B, (uint64_t)ks * FAC_P * 128u, FAC_P, FAC_SIN, 128u,
                                ks == 0u);
                    Rls<PIPE_M>(B_K);
                    Rls<PIPE_M>(B_L0);
                }
#endif
                Rls<PIPE_M>(bC);       // L0C 累加结果就绪，跨 pipe 交给 FIX（release mode=false）
                PipeBarrier<PIPE_FIX>();
                Acq<PIPE_FIX>(bC);
                FacFixpToUb(UbS(q), Slot(l0C, bC), FAC_SIN, 0u, FAC_SIN);
                Rls<PIPE_FIX>(bC);
            }
            FacCcSet<PIPE_FIX>(witGm_, GetBlockIdx(), CC_S_RDY + q);   // S[q] 就绪（放行配对 2 AIV）

            // ================= (B) PV(t) = P · V =================
            FacCcWait<PIPE_MTE1>(witGm_, GetBlockIdx(), CC_P0 + q);       // 配对 2 AIV 都已把 P(t) 直写进 L1P(q)
            FacCcWait<PIPE_FIX>(witGm_, GetBlockIdx(), CC_PV_FREE + q);   // 配对 2 AIV 已消费上一轮 PV[q]
            {
                // P(t)：**AIV 已直写 L1P(q)（[P,SIN] NZ）** ⇒ 本核只做 L1 -> L0A，全程不落 GM。
                PipeBarrier<PIPE_MTE1>();
                Acq<PIPE_MTE1>(B_L0);
                Acq<PIPE_MTE1>(B_PL1);
                FacLoadL0_2D<bfloat16_t>(l0A, L1P(q), 0, FAC_P / 16u, FAC_SIN / 16u, FAC_P / 16u, FAC_P / 16u,
                                         false, FAC_L0A_P_ELEMS);
                Rls<PIPE_MTE1>(B_PL1);
                Rls<PIPE_MTE1>(B_L0);

                // V(t)：GM -> L1（转置读进 L0B）
                PipeBarrier<PIPE_MTE2>();
                Acq<PIPE_MTE2>(B_V);
                const uint64_t vOffElem =
                    kvLayerElem + M15KV_KV_BYTE_OFF_CONTIG(colBase, w.n2, M15Kv::KV_LANE_V, 0u) /
                                      M15Kv::KV_ELEM_BYTES;
#if defined(M25FA_GEO_ONEMAT)
                for (uint32_t b = 0; b < nb; ++b) {
                    FacGmToL1NzOneMatrix<bfloat16_t>(
                        l1V, kvGM_,
                        vOffElem + (uint64_t)b * (M15Kv::KV_BLOCK_BYTES / M15Kv::KV_ELEM_BYTES),
                        FAC_DV, (uint32_t)(M15Kv::KV_TOKEN_STRIDE / M15Kv::KV_ELEM_BYTES), FAC_SIN, 256u * b);
                }
#else
                FacGmToL1Nz<bfloat16_t>(l1V, kvGM_, vOffElem, nb, FAC_BLK_TOK, FAC_DV,
                                        M15Kv::KV_BLOCK_BYTES / M15Kv::KV_ELEM_BYTES,
                                        (uint32_t)(M15Kv::KV_TOKEN_STRIDE / M15Kv::KV_ELEM_BYTES), FAC_SIN);
#endif
                Rls<PIPE_MTE2>(B_V);
                for (uint32_t nh = 0; nh < 2u; ++nh) {
                    PipeBarrier<PIPE_M>();   // nh=1 会紧接着重新 get 同一 B_L0/B_V/bC（同 pipe 同 id 背靠背）
                    // ⚠ M189：同 BMM1 —— MTE1 每装完一个维半块的 Vᵀ 就 release，两条 Mmad 才各读自己的 L0B。
                    PipeBarrier<PIPE_MTE1>();
                    Acq<PIPE_MTE1>(B_L0);
                    Acq<PIPE_MTE1>(B_V);
                    const MutexID bC = L0CId(2u * q + 1u);
                    // B 操作数 = Vᵀ [N=128 维半块, K=colLimit 行]（转置分支：
                    // mStep = colLimit/16（源行分形数）、kStep = N/16、srcStride = 源行数/16、dstStride = N/16）
                    FacLoadL0_2D<bfloat16_t>(l0B, l1V, (uint64_t)nh * (128u / 16u) * (FAC_SIN / 16u) * 256u,
                                             colLimit / 16u, 128u / 16u, FAC_SIN / 16u, 128u / 16u, true, 0);
                    Rls<PIPE_MTE1>(B_V);
                    Rls<PIPE_MTE1>(B_L0);
                    Acq<PIPE_M>(B_L0);
                    Acq<PIPE_M>(B_V);
                    Acq<PIPE_M>(bC);
                    FacMmadBf16(Slot(l0C, bC), l0A, l0B, FAC_L0A_P_ELEMS, FAC_P, 128u, colLimit, true);
                    Rls<PIPE_M>(B_V);
                    Rls<PIPE_M>(B_L0);
                    Rls<PIPE_M>(bC);
                    PipeBarrier<PIPE_FIX>();
                    Acq<PIPE_FIX>(bC);
                    FacFixpToUb(UbPV(q), Slot(l0C, bC), 128u, nh * 128u, FAC_DV);
                    Rls<PIPE_FIX>(bC);
                }
            }
            FacCcSet<PIPE_FIX>(witGm_, GetBlockIdx(), CC_PV_RDY + q);   // PV[q] 就绪（放行配对 2 AIV）
        }
    }

private:
    GlobalTensor<bfloat16_t> qGM_;
    GlobalTensor<bfloat16_t> kvGM_;
    GlobalTensor<bfloat16_t> pGM_;
    GlobalTensor<int32_t> witGm_;
    uint32_t qStride_ = 13952u;
};

// ============================================================
// 12. AIV：三角掩码 + online softmax + PV 累加 + 归一化落盘
// ============================================================
template <bool Causal>
class FacAiv {
public:
    __aicore__ inline FacAiv() {}

    __aicore__ inline void Init(__gm__ uint8_t* out, __gm__ uint8_t* pScratch, __gm__ uint8_t* witScratch)
    {
        witGm_.SetGlobalBuffer((__gm__ int32_t*)witScratch);
        outGM_.SetGlobalBuffer((__gm__ bfloat16_t*)out);
        // `pScratch`（P 的 GM 中转）**已不再使用**：P 改走 UB→L1 硬通道（§5b）。
        // 保留形参与成员只为 `FaCoreBody` 的对外签名稳定（融合清单 (b)/README §3(b)）。
        pGM_.SetGlobalBuffer((__gm__ bfloat16_t*)pScratch);
    }

    __aicore__ inline void Run(uint32_t m, uint32_t posBase, uint32_t ctx, uint32_t nBlk)
    {
        const uint32_t bid = GetBlockIdx();
        const uint32_t half = bid & 1u;
        const uint32_t aic = bid >> 1;
        const uint32_t nQTile = FacCeilDiv(m, FAC_P);
        const uint32_t nWork = FAC_NH * nQTile;

        // 令牌预置：S 槽位空闲 ×2、PV 槽位空闲 ×2
        // （mode 2：**配对 2 个 AIV 各自置一份**；AIC 侧的 wait 一次即代表两份都到）
        FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_S_FREE);
        FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_S_FREE + 1u);
        FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_PV_FREE);
        FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_PV_FREE + 1u);

        FacWork w;
        for (uint32_t wi = aic; wi < nWork; wi += nBlk) {
            FacMakeWork(wi, m, posBase, ctx, w);
            RunWorkItem(w, m, posBase, ctx, half);
        }
    }

private:
    __aicore__ inline void RunWorkItem(const FacWork& w, uint32_t m, uint32_t posBase, uint32_t ctx, uint32_t half)
    {
        if (w.nTiles == 0u) { return; }
        // S/PV 由 AIC 的 Fixpipe（dualDstCtl）直写本 AIV 的 UB，故与 AIC 侧同为 VECIN（M101 同款）
        LocalTensor<uint8_t> s0U(TPosition::VECIN, FAC_UB_S0_OFF, FAC_UB_S_BYTES);
        LocalTensor<uint8_t> s1U(TPosition::VECIN, FAC_UB_S1_OFF, FAC_UB_S_BYTES);
        LocalTensor<uint8_t> pv0U(TPosition::VECIN, FAC_UB_PV0_OFF, FAC_UB_PV_BYTES);
        LocalTensor<uint8_t> pv1U(TPosition::VECIN, FAC_UB_PV1_OFF, FAC_UB_PV_BYTES);
        LocalTensor<uint8_t> pcU(TPosition::VECCALC, FAC_UB_PC_OFF, FAC_UB_PC_BYTES);
        LocalTensor<uint8_t> accU(TPosition::VECCALC, FAC_UB_ACC_OFF, FAC_UB_ACC_BYTES);
        LocalTensor<uint8_t> thrU(TPosition::VECCALC, FAC_UB_THR_OFF, FAC_UB_THR_BYTES);
        LocalTensor<uint8_t> mU(TPosition::VECCALC, FAC_UB_M_OFF, FAC_AIV_ROWS * 4u);
        LocalTensor<uint8_t> sumU(TPosition::VECCALC, FAC_UB_SUM_OFF, FAC_AIV_ROWS * 4u);
        LocalTensor<uint8_t> edU(TPosition::VECCALC, FAC_UB_ED_OFF, FAC_AIV_ROWS * 4u);

        __ubuf__ float* sF[2] = {reinterpret_cast<__ubuf__ float*>(s0U.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(s1U.GetPhyAddr())};
        __ubuf__ float* pvF[2] = {reinterpret_cast<__ubuf__ float*>(pv0U.GetPhyAddr()),
                                  reinterpret_cast<__ubuf__ float*>(pv1U.GetPhyAddr())};
        __ubuf__ bfloat16_t* pcB = reinterpret_cast<__ubuf__ bfloat16_t*>(pcU.GetPhyAddr());
        __ubuf__ float* accF = reinterpret_cast<__ubuf__ float*>(accU.GetPhyAddr());
        __ubuf__ float* thrF = reinterpret_cast<__ubuf__ float*>(thrU.GetPhyAddr());
        __ubuf__ float* mF = reinterpret_cast<__ubuf__ float*>(mU.GetPhyAddr());
        __ubuf__ float* sumF = reinterpret_cast<__ubuf__ float*>(sumU.GetPhyAddr());
        __ubuf__ float* edF = reinterpret_cast<__ubuf__ float*>(edU.GetPhyAddr());

        LocalTensor<bfloat16_t> pcT = pcU.template ReinterpretCast<bfloat16_t>();

        // 行状态初值（RegBase；UB_ACC 只被 V 访问 ⇒ 不进 BufferID 令牌，`B_ACC` 仅登记窗名）
        FacInitStateVf(mF, sumF, accF);

        const uint32_t posLo = posBase + w.rowBase + half * FAC_AIV_ROWS;

        for (uint32_t t = 0; t < w.nTiles; ++t) {
            const uint32_t q = t & 1u;
            const uint32_t colBase = t * FAC_SIN;
            const uint32_t nb = FacBlocksInTile(colBase, ctx);
            const uint32_t colLimit = nb * FAC_BLK_TOK;

            // ---- Vec1：S(t) 就绪 -> 掩码 + online softmax + P(bf16) ----
            FacCcWait<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_S_RDY + q);
            PipeBarrier<PIPE_V>();
            Acq<PIPE_V>(B_PC);
            // 三个下标量**原样传入**（VF 内做全部组合；VF 外无算术）—— 见 VF 头注（按塔裁从严）
            FacSoftmaxTileVf<Causal>(sF[q], pcB, thrF, mF, sumF, edF, posLo, colBase, colLimit);
            Rls<PIPE_V>(B_PC);                       // E/S 已读完 -> 放行 S[q] 复用与 P 的 MTE3（release mode=false）
            FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_S_FREE + q);   // S[q] 已消费（回 AIC）

            // ---- P 直写 L1（UB→L1 硬通道；**不落 GM**）：本 AIV 写 L1P(q) 的 M 半块 ----
            PipeBarrier<PIPE_MTE3>();
            Acq<PIPE_MTE3>(B_PC);
            {
                LocalTensor<uint8_t> l1pU(TPosition::A1, q ? (FAC_L1_P_OFF + FAC_L1_P_BYTES) : FAC_L1_P_OFF,
                                          FAC_L1_P_BYTES);
                FacUbToL1P(l1pU, pcT, half);
            }
            Rls<PIPE_MTE3>(B_PC);
            FacCcSet<PIPE_MTE3>(witGm_, (28u + GetBlockIdx()), CC_P0 + q);   // P 已直写 L1（2 AIV 都 set 后 AIC 放行）

            // ---- Vec2：PV(t) 就绪 -> acc = acc·ed + PV ----
            FacCcWait<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_PV_RDY + q);
            FacAccUpdateVf(accF, pvF[q], edF);
            FacCcSet<PIPE_V>(witGm_, (28u + GetBlockIdx()), CC_PV_FREE + q);   // PV[q] 已消费（回 AIC）
        }

        // ---- 归一化 + 落 GM（一轮搬完 32 行）----
        // ⚠ out 平面必须按 `FAC_P` 行向上圆整分配（尾 q tile 会写满 64 行；行号 ≥ m 的走 padding）
        PipeBarrier<PIPE_V>();                       // 同 pipe 背靠背复用 B_PC 前后排空（M126 §3）
        Acq<PIPE_V>(B_PC);
        FacNormalizeVf(accF, pcB, sumF);
        Rls<PIPE_V>(B_PC);                           // 写出侧 release(mode=false)
        PipeBarrier<PIPE_MTE3>();
        Acq<PIPE_MTE3>(B_PC);
        const uint64_t dst0 = (uint64_t)(w.rowBase + half * FAC_AIV_ROWS) * FAC_OUT_STRIDE +
                              (uint64_t)w.h * FAC_HD;
        DataCopy(outGM_[dst0], pcT[0],
                 DataCopyParams{(uint16_t)FAC_AIV_ROWS, (uint16_t)(FAC_DV * 2u / 32u), 0,
                                (uint16_t)((FAC_OUT_STRIDE - FAC_DV) * 2u / 32u)});
        Rls<PIPE_MTE3>(B_PC);
    }

private:
    GlobalTensor<bfloat16_t> outGM_;
    GlobalTensor<bfloat16_t> pGM_;
    GlobalTensor<int32_t> witGm_;
};

// ============================================================
// 13. 段入口（供融合 TU 的 `__global__` 壳调用；`Causal=false` 只给负向对照）
// ============================================================
template <bool Causal = true>
__aicore__ __inline__ void FaCoreBody(__gm__ uint8_t* q, uint32_t qStride, __gm__ uint8_t* kv, __gm__ uint8_t* out,
                                      __gm__ uint8_t* pScratch, __gm__ uint8_t* witScratch, uint32_t m,
                                      uint32_t posBase, uint32_t ctx, uint32_t lane, uint32_t nBlk)
{
    if ASCEND_IS_AIV {
        FacAiv<Causal> op;
        op.Init(out, pScratch, witScratch);
        op.Run(m, posBase, ctx, nBlk);
    }
    if ASCEND_IS_AIC {
        FacAic op;
        op.Init(q, qStride, kv, out, pScratch, witScratch);
        op.Run(m, posBase, ctx, lane, nBlk);
    }
}

}  // namespace M15FAC

#endif  // M15_ATTN_FA_CORE_H
