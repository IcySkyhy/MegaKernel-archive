// ============================================================
// m15_attn_prefill.h —— M116（Wave B2）：attention 前端的 **prefill 段体** + cache 填的 chunk 尺度驱动
//
// 权威清单：`docs/15-prefill-design.md` 的 `## M103 重盘` → `M103-2.2` 的 **B2 行**
//   · 段体头（交付到 `m15_layer_loop/`）= **本文件**
//   · 允许改（独占）= `m15_attn_cache.h`（探针尺度 → chunk 尺度）；M103 的 B2 行还把
//     `m15_attn_prolog.h`（`AP_*` 形状常量）划给 B2，但**本 mission 的 scope 里没有它**
//     ⇒ 需求写进融合清单（README §5），见下面「未做」第 1 条
//   · 不许碰 = M88 的探针实现 `m15_attn_prolog_probe.h`（**只读复用**它的 `AivNormRope`/`AivCopy`）、
//     `m15_attn_kv.h`（布局权威，只读）、Wave A 的 4 个文件、`m15_layer_loop/CMakeLists.txt`
//
// 【本文件在 TU 里的位置】由 `m15_attn_cache.h` 的**末尾** include（那一步已经在融合 TU 的
// include 链里：`.asc` → `m15_attn_kv.h` → `m15_attn_cache.h` → 本文件）⇒ **融合 TU 一字不用改**。
// 沿用 `m15_attn_kv.h` 末尾 include `m15_attn_cache.h` 的同一手法（`m15_attn_kv.h` 的 §6 有说明）。
// ⚠ 本文件与 `m15_attn_cache.h` 都依赖 `M15H::ExtBlock1`（在 `m15_hc_layer.h` 里）——
//   那是 `m15_attn_cache.h` **既有**的依赖（M98 起），融合 TU 与独立验证路都按 `.asc` 的
//   include 顺序（先 `m15_gdn_layer.h` + `m15_hc_layer.h`）满足它。
//
// ---------------------------------------------------------------------------
// 【本文件相对 M88 探针改了什么（逐条 = M103-2.2 的 B2 行 + 任务书任务 2）】
//   ① **`(row × head)` 二维工作项**：M88 的 AIV 臂是 `bid → head` 的一维硬分派（`bid < 24` → q 头
//      `bid`、`24/25` → k、`26` → v、`27..30` → idx q、`31` → raw k），**一趟只处理一行**，且只有
//      32 个 AIV 有活（本机 56 个 AIV ⇒ 24 个空转；行数一变大也不会回绕）。本文件改成
//      `for (i = bid; i < rows*NITEMS; i += nAiv)`、`row = i / NITEMS`、`item = i % NITEMS`
//      —— 行维与头维都进工作项，**AIV 数不参与形状**。
//   ② **AIC 的 N 块分派加 m-tile**：M88 的 `for (g = bid; g < GEMM_NBLOCKS; g += nCores)` 只有 N 维，
//      且 `AttnGemm` 把 `CALC_M = 2` 写死（m=1 的 3510 Nd2Nz quirk）。本文件的 `PfAttnGemm` 带
//      **m-tile 循环**：一个 chunk（= `M15AC::AC_MAX_ROWS` = 128 行）**跨 2 个 m-tile**
//      （`PF_M_TILE = AP_BASE_M` = 64），工作项 = `(mTile, nBlock)` 展平后按 AIC 条带划分，
//      尾块用 `curM` mask（`calcM = max(curM, 2)` 的 3510 契约原样保留）。
//   ③ **per-row 位置**：M88 的入口只有一个 `apPos` 标量（`LayerArgs::apPos` 也是单个 `uint32`）。
//      本文件按 Wave A 的 prefill 契约接受 **per-row 位置**（`pfPos` 表 / `pfPosBase + 行号`）：
//      `PfPosRef::At(绝对行号)`。
//   ④ **矩阵乘法一律 `Mmad`**（人类裁决）：4 个 GEMM 全走 `Mmad`（`PfAttnGemm`），
//      **没有** VF 版本、**没有**选型对比。
//   ⑤ **cache 填进 chunk 尺度**：`m15_attn_cache.h` 的 `AC_MAX_ROWS` 8 → 128、`AC_MAX_GROUPS` 2 → 32；
//      主 KV 的落页由 `M15AC::MainKvFillChunk`（同一份 `M15KV_KV_*` 宏）按 chunk 逐行做。
//
// ---------------------------------------------------------------------------
// 【与 M88 探针的**语义差异（有意，且写在判据上）**】
//   M88 的 `out` 平面有 6 段（q / gate / k / v / idx q / raw k），其中 k/v/raw k 是**原样抄写**。
//   本文件**不再把 k/v/raw k 落到 `out`**：k 的 norm+rope 结果直接落进 **cache 的输入面**
//   （主 KV 的 (head,K) 槽），v 与 raw k 同样直接落进 cache 的输入面（(head,V) 槽 / 环行）。
//   理由：主 KV 存的**就是** norm+rope 之后的 k 与 v（`m15_attn_kv.h` 主 KV 段的官方依据：
//   `qwen4_exp/nvidia/qsa.py:434-442` 的 `FullAttentionSpec` + `flash_attn.py:1525-1540` 的
//   `reshape_and_cache_flash`），而 `out.k/v/rawk` 在融合路径里没有消费者（核心读主 KV 与
//   `out.q`/`out.gate`）⇒ 落两份等于把同一段字节写两遍 GM。**代价**：M88 的
//   `Ap.out.k` / `Ap.out.v` / `Ap.out.kraw` 三条**逐字节**判据在融合路径上不再适用，其覆盖由
//   cache 侧判据接管（`Ac.ring.row` 覆盖 raw k + 位置尾、`Ac.mainkv.kv` 覆盖 k/v）。
//
// 【本文件不做的事（边界）】
//   · 不做 indexer 的打分 / topk / packed 的**内容**（本文件只把 packed 的 seed 行原样抄写，
//     与 M98 探针同一条腿）；不做 attention 核心（稠密 causal FA = Wave B3 的 `m15_attn_fa_core*`）；
//     不做 `o_proj`；不做 N1（`subOut` 的生产者，塔裁归 Wave C）。
//   · **不引入 `slot_mapping`**、**只支持恒等 `block_table`（单请求）**（Wave A 的字段契约 + M82 的 D2）。
//   · **不吃 `LayerArgs`**（Wave B 共同契约第 1 条）：入口只吃指针 + 标量（`PfChunkArgs` 是纯 POD）。
// ============================================================
#ifndef M15_ATTN_PREFILL_H
#define M15_ATTN_PREFILL_H

#include "m15_attn_kv.h"
#include "m15_attn_prolog.h"
#include "m15_attn_prolog_probe.h"   // 只读复用：AivNormRope / AivCopy / AP_BUF_* / GEMM_NBLOCKS
#include "m15_attn_cache.h"          // M116 的 chunk 尺度 cache 填 + MainKvFillChunk

namespace M15PF {

using namespace AscendC;
using namespace M15Kv;
using namespace M15AP;
using namespace M15AC;

// ============================================================
// 1. 规模（**全部派生自既有常量，不写第二份数字**）
// ============================================================
constexpr uint32_t PF_ROWS = M15AC::AC_MAX_ROWS;              // 128：一次 launch 的一个 chunk
constexpr uint32_t PF_M_TILE = AP_BASE_M;                     // 64：AIC 的 m-tile（= M15G::BASE_M）
constexpr uint32_t PF_M_TILES = PF_ROWS / PF_M_TILE;          // 2：一个 chunk 跨 2 个 m-tile
constexpr uint32_t PF_NBLOCKS = GEMM_NBLOCKS;                 // 109 = 96+4+4+5（M88 的 §B 表）
constexpr uint32_t PF_GEMM_ITEMS = PF_M_TILES * PF_NBLOCKS;   // 218 个 (mTile, nBlock) 工作项
constexpr uint32_t PF_Y0_STRIDE = Y0_N;                       // 13952：y0 平面的行距（元素）
constexpr uint32_t PF_OUT_STRIDE = AP_OUT_N;                  // 13952：out 平面的行距（元素）

static_assert(PF_M_TILES * PF_M_TILE == PF_ROWS, "chunk 必须是整数个 m-tile（否则尾块语义要重推）");
static_assert(PF_Y0_STRIDE == PF_OUT_STRIDE, "y0 与 out 的行距同源（M88 §2 的两张表逐项相等）");

// ---- (row × item) 的 item 维：56 项（24 q 头 + 24 gate 头 + 2 k 头 + 1 v + 4 idx q 头 + 1 raw k）----
constexpr uint32_t IT_Q = 0;                                  // [0,24)   q 头 h
constexpr uint32_t IT_GATE = IT_Q + NH;                       // [24,48)  gate 头 h（原样抄写）
constexpr uint32_t IT_K = IT_GATE + NH;                       // [48,50)  k 头 h（norm+rope → 主 KV 的 K 槽）
constexpr uint32_t IT_V = IT_K + NKV;                         // [50]     v（2 头 → 主 KV 的 V 槽）
constexpr uint32_t IT_IDXQ = IT_V + 1u;                        // [51,55)  indexer q 头 h
constexpr uint32_t IT_RAWK = IT_IDXQ + IDX_NH;                // [55]     raw k（→ 环行）
constexpr uint32_t PF_ITEMS = IT_RAWK + 1u;                   // 56
static_assert(PF_ITEMS == NH + NH + NKV + 1u + IDX_NH + 1u, "(row × item) 的 item 维条数变了");
static_assert(IT_RAWK == 55u && PF_ITEMS == 56u, "item 下标表（判据按常量名取，不按字面量取）");

// ============================================================
// 2. 同步清单（**核内 = buffer id；核间 = CrossCore，不新造号**）
// ============================================================
// 逐条与 `m15_layer_resources.h` §4d 的 prefill 分节对齐（`(核型, mode, id)` 三元组）：
//   · AIC mode-0 `AP_AIC_M0_OUT`(1)：4 个 GEMM 的 FIXP 写 GM 全部排空 + 全体 AIC 对齐
//     —— 与 M97 登记的同一个号（§4d 表第 1 行）；"相邻逻辑点 1 → 8"的订正见 §4e
//   · mode-2 `AP_A2V_GEMM`(5)：AIC → 配对 2 个 AIV「本 chunk 的 y0 全量可见」
//   · mode-2 `AP_V2A_READY`(4)：**本 mission 起启用**（M88 的 `m15_attn_prolog.h` 注释写着
//     "仅在交替形态下用；当前未启用"，本 mission 就是那个"交替形态"）——
//     AIV → 配对 AIC「本 chunk 的 y0 已消费，可以覆写」。**没有它就有真竞态**：AIC 会直接跑
//     下一个 chunk 的 GEMM，把 y0 覆写在 AIV 还在读的行上。配对语义 = **2 set(AIV) 配 1 wait(AIC)**
//     （`docs/05 §2` 的 mode-2 配对规则之一；`m10_attn_decode` 的 `CC_AIVDONE` 同款）。
//   · AIV mode-0 `FLAG_AIV_SEG_S1`(10) / `FLAG_AIV_SEG2`(11)：chunk 内的两条全体 AIV barrier
//     —— B1 = 「(row × item) 的产物（`in` 输入面）全体就位」；B2 = 「block 0 的 cache 填读完
//     `in` 输入面」。**B2 不可省**：没有它，先跑完 item 的 AIV 会在下一个 chunk 里覆写 block 0
//     还在读的 `in` 输入面（`in` 是**每 chunk 覆写**的 scratch）。
//     两个 id 都取自 decode 档 `FLAG_SEQ[]` 已登记的 (AIV, 0, 10/11)（GDN 段的段内 barrier）
//     ⇒「复用已有号」成立；**把它们补登记进 `FLAG_SEQ_PREFILL_ATTN[]` 是 Wave C 的动作**（§README §5）。
constexpr uint16_t PF_AIC_M0_BAR = AP_AIC_M0_OUT;             // 1  （AIC mode-0）
constexpr uint16_t PF_A2V_TILE = AP_A2V_GEMM;                 // 5  （mode-2：AIC→AIV）
constexpr uint16_t PF_V2A_DONE = AP_V2A_READY;                // 4  （mode-2：AIV→AIC）
constexpr uint16_t PF_AIV_BAR_ITEMS = M15G::FLAG_AIV_SEG_S1;  // 10 （AIV mode-0）
constexpr uint16_t PF_AIV_BAR_CACHE = M15G::FLAG_AIV_SEG2;    // 11 （AIV mode-0）
static_assert(PF_A2V_TILE != PF_V2A_DONE, "两个 mode-2 同步点不得同号（同核相邻 set 不保序）");
static_assert(PF_AIV_BAR_ITEMS != PF_AIV_BAR_CACHE, "两条 AIV barrier 不得同号");
static_assert(PF_AIV_BAR_ITEMS < 16u && PF_AIV_BAR_CACHE < 16u, "mode0/2 的每核池是 0..15");

// 核内 buffer id（前端自己的两个；M15AP 用 0/1/2、本段 cache 用 10/11/12/13）：
//   PF_BUF_ROW : MTE2（raw k 256 B 读入）→ S（位置尾 24 B 标量写）→ MTE3（280 B 落 GM）
//   PF_BUF_KV  : MTE2（本行 K/V）→ MTE3（主 KV 落页）—— `MainKvFillChunk` 的暂存槽
constexpr AscendC::MutexID PF_BUF_ROW = static_cast<AscendC::MutexID>(14);
constexpr AscendC::MutexID PF_BUF_KV = static_cast<AscendC::MutexID>(15);

// ============================================================
// 3. UB 窗（前端自己的；**与 M15AP / M15AC 的窗都不重叠**）
// ============================================================
// 排布依据：M15AP 的窗是 [0, UB_AP_END = 14336)（`m15_attn_prolog.h` §7），M15AC 的窗从
// `AC_UB_BASE = 32768` 起 ⇒ 本文件用中间的 **[14336, 32768)** 这一空档（18 KB）：
//   PF_UB_ROW：环行组装窗 —— [raw k 128 bf16 = 256 B][位置尾 3×int64 = 24 B]（280 B 用，288 B 留量）
//   PF_UB_KV ：主 KV 一行（2 head × (K256‖V256) bf16 = 2048 B）
constexpr uint32_t PF_UB_ROW = 14336u;                        // 288 B（用 280 B）
constexpr uint32_t PF_UB_KV = 14720u;                         // 2048 B
constexpr uint32_t PF_UB_END = PF_UB_KV + AC_IN_KV_ROW_BYTES; // 16768
static_assert(PF_UB_ROW % 32u == 0u && PF_UB_KV % 32u == 0u, "UB 段起点 32 B 对齐");
static_assert(PF_UB_ROW + RING_ROW_BYTES + 8u <= PF_UB_KV, "环行窗不与主 KV 窗重叠");
static_assert(PF_UB_END <= AC_UB_BASE, "前端 UB 窗不得侵进本段 cache 的窗（起点 32768）");
static_assert(PF_UB_ROW >= UB_AP_END, "前端 UB 窗不得侵进 M15AP 的窗（终点 14336）");
static_assert(PF_UB_END <= 248u * 1024u, "前端 UB 窗必须落在 248 KB 硬限内");

// ============================================================
// 4. 档 / 变异（**判据的负向对照**；契约档 = 0）
// ============================================================
// 前端的四个方向级变异**直接复用 M88 的 `AP_MODE_*`**（同一份实现、同一套语义，低 8 位），
// 另加四条"把被测对象弄坏"（docs/17 §4 的非空洞性 + 变体五纪律）：
constexpr uint32_t PF_MUT_NO_QIDX = 1u << 8;      // 变体五：不跑 indexer q 头 ⇒ `Pf.out.qidx` 必须变红
constexpr uint32_t PF_MUT_NO_KV = 1u << 9;        // 变体五：k/v 不落 cache 输入面 ⇒ `Ac.mainkv.kv` 必须变红
constexpr uint32_t PF_MUT_NO_RAWK = 1u << 10;     // 变体五：raw k 不落环行 ⇒ `Ac.ring.row` 必须变红
constexpr uint32_t PF_MUT_NO_GEMM = 1u << 11;     // 变体五：AIC 不跑 GEMM ⇒ `Pf.y0.*` 必须变红
// prolog 的档（= M88 的 `AP_MODE_*`，低 8 位）与 cache 的档（= M98 的 `AC_MODE_*`）是**两个不同的
// 编号域**（两边的 1 号含义不同）⇒ `PfChunkArgs` 里分成两个字段，**不做位域混装**。
__aicore__ inline uint32_t PfApMode(uint32_t mut) { return mut & 0xFFu; }

// ---- 资源峰值（**给 Wave C 填 `m15_layer_resources.h` §3c 的 `PF_{UB,L1,L0C}_*` 槽用**）----
// 三个量都是本文件 / `m15_attn_cache.h` / `m15_attn_prolog.h` 的 `static_assert` 已经钉住的：
//   · UB：相位 A 的三段窗**不在同一时刻**（AIV 项 → cache 填），故峰值 = 三段窗顶的最大值。
//     M15AP 的窗 [0, 14336)、前端窗 [14336, 16768)、cache 窗 [32768, 117312) ⇒ **117,312 B**
//   · L1：4 个 GEMM 的 A/B ping-pong（`m15_attn_prolog.h` §6 的 L1_A0/A1/B0/B1）⇒ B 区最高
//   · L0C：单个 m-tile 的 64×128 fp32 tile
constexpr uint32_t PF_UB_PEAK_BYTES = M15AC::AC_UB_END;                        // 117,312
constexpr uint32_t PF_L1_PEAK_BYTES = M15AP::L1_B1 + M15AP::AP_L1_B_ELEMS * 2u; // 294,912
constexpr uint32_t PF_L0C_PEAK_BYTES = M15AP::L0C_BYTES;                        // 32,768
static_assert(PF_UB_PEAK_BYTES <= 248u * 1024u, "attention-prefill 的 UB 峰值必须 ≤ 248 KB");
static_assert(PF_L1_PEAK_BYTES <= 512u * 1024u, "attention-prefill 的 L1 峰值必须 ≤ 512 KB");
static_assert(PF_L0C_PEAK_BYTES <= 256u * 1024u, "attention-prefill 的 L0C 峰值必须 ≤ 256 KB");

// ---- 变异掩码的**单字段打包约定**（Wave C 只用一个 `pfMutant` 就够了，不必新增字段）----
// 位 [0,12) = 前端掩码（低 8 位 = `AP_MODE_*`）；位 [12,16) = cache 的 `AC_MODE_*`。
// 独立验证路（`m24_attn_prefill/`）可以直接分别传 `PfChunkArgs::mut` / `::acMode`（两条路等价）。
constexpr uint32_t PF_AC_MODE_SHIFT = 12u;
constexpr uint32_t PF_MUT_MASK = (1u << PF_AC_MODE_SHIFT) - 1u;
constexpr uint32_t PfPackMutant(uint32_t mut, uint32_t acMode)
{
    return (mut & PF_MUT_MASK) | ((acMode & 0xFu) << PF_AC_MODE_SHIFT);
}
constexpr uint32_t PfUnpackMut(uint32_t packed) { return packed & PF_MUT_MASK; }
constexpr uint32_t PfUnpackAcMode(uint32_t packed) { return (packed >> PF_AC_MODE_SHIFT) & 0xFu; }

// ============================================================
// 5. per-row 位置（Wave A 的 `pfPos` / `pfPosBase` 契约）
// ============================================================
//   · `tbl != nullptr`：按**绝对行号**取表（支持将来 batch>1 的形态）
//   · 否则：`base + 绝对行号`（单请求连续；与官方 `query_start_loc` 的语义相容）
//   · 两者同时给时以表为准（与 Wave A 的字段注释一字一致）
// ⚠ **chunk 内位置必须连续**：cache 侧（`m15_attn_cache_body`）的门控/组号是由
//    `(chunkStart, chunkRows)` 推出来的（`pos = chunkStart + r`），**它不逐行读位置表**。
//    多请求 / 非连续行的形态**不在 B2 的范围**（显式写窄，见 README §未完成第 2 条）。
//    ⚠ 非连续时**必须响亮失败**：本文件在 block 0 上逐行复核位置（下文的 `PfPosContiguous`），
//      不连续 ⇒ **跳过 cache 填**并把 `AC_FLAG_FE_CONTRACT` 置 1（cache 保持毒值 ⇒ 判据变红）。
//      这条不是"绝对断言"，而是"不静默错"的可执行形式。
struct PfPosRef {
    __gm__ uint32_t* tbl;   // 可空
    uint32_t base;

    __aicore__ inline uint32_t At(uint32_t absRow) const
    {
        if (tbl == nullptr) {
            return base + absRow;
        }
        AscendC::GlobalTensor<uint32_t> g;
        g.SetGlobalBuffer(tbl, PREFILL_M);   // 表长上界 = prefill 容量（调用方保证行长覆盖）
        return g.GetValue(static_cast<uint64_t>(absRow));
    }
};

// ============================================================
// 6. 入口实参（**纯 POD：指针 + 标量，不吃 `LayerArgs`**）
// ============================================================
struct PfChunkArgs {
    // ---- 输入 ----
    __gm__ uint8_t* x;         // [seqRows, HIDDEN] bf16（**按本次要跑的行数**给足；尾部 1 行余量）
    __gm__ uint8_t* w;         // attention 的 8 个 role 权重平面（`M15AP` §3 的偏移表）
    __gm__ uint8_t* cs;        // cos/sin 表（[PREFILL_M][CS_ROW_ELEMS] bf16）
    __gm__ uint8_t* posTbl;    // per-row 位置表 u32（**可空**；按绝对行号索引）
    // ---- 本 chunk 的 scratch（每 chunk 覆写；调用方保证 ≥ 一个 chunk 的容量）----
    __gm__ uint8_t* y0;        // [PF_ROWS, PF_Y0_STRIDE] bf16
    __gm__ uint8_t* out;       // [PF_ROWS, PF_OUT_STRIDE] bf16
    __gm__ uint8_t* in;        // [PF_ROWS] 的 cache 输入面（`M15AC` §1a 的布局与行距）
    // ---- 四套 cache / packed（**序列**的平面，跨 chunk 存活）----
    __gm__ uint8_t* kv;        // 主 KV（`KV_PLANE_BYTES`）
    __gm__ uint8_t* comp;      // compressed key
    __gm__ uint8_t* ring;      // raw key ring
    __gm__ uint8_t* pack;      // packed indices（本段只抄 seed 行）
    __gm__ uint8_t* packSeed;  // packed 的 seed 行（Wave A / B3 的产物；本段原样抄写）
    __gm__ uint8_t* pooled;    // 池化的 scratch（每 chunk ≤ AC_MAX_GROUPS 行；诊断面）
    __gm__ uint8_t* flag;      // 门控标志（AC_FLAG_LANES 个 i32；诊断面）
    // ---- 标量 ----
    // ⚠ **每次调用只处理一个 chunk**（段体对 `nChunks > 1` `Trap` 硬拦）：
    uint32_t rowStart;         // **本 chunk 的绝对首行号**（x 的行偏移）
    uint32_t seqRows;          // 本次要处理的行数 = 本 chunk 的行数（≤ PF_ROWS；尾 chunk 可短）
    uint32_t chunkRows;        // 同上（逐 chunk 形态下两者相等；保留两个字段是为了将来多 chunk 形态）
    uint32_t nChunks;          // **必须为 1**（0 按 1 处理）；> 1 ⇒ Trap
    uint32_t posBase;          // 单请求连续时的位置起点（`pos = posBase + 绝对行号`）
    uint32_t mut;              // 前端变异掩码（契约档 = 0；低 8 位 = AP_MODE_*）
    uint32_t acMode;           // cache 侧的档（= `M15AC::AC_MODE_*`；契约档 = 0）
};

// ============================================================
// 7. AIC：4 个 GEMM（`Mmad`），工作项 = **(mTile, nBlock)**
// ============================================================
// donor：`m15_attn_prolog_probe.h` 的 `M15AP::AttnGemm`（N 块分派 + `AP_BASE_N = 128` 的整除推导，
// 见 `m15_attn_prolog.h` §6）+ `m15_gdn_layer.h` 的 `Cube::Bf16Gemm`（**m-tile 循环**与
// `curM / calcM / calcMAlign` 的尾块处理）。两处都是"既有代码抄和改"。
// 与 donor 的**唯一**差别：C 的行距是 `CM_STRIDE`（= y0 平面的行距 13952），不是 GEMM 自己的 N
// —— 4 个 GEMM 的输出是 y0 **同一行的 4 个段**（Y0_QG / Y0_K / Y0_V / Y0_IDX），不是各自独立的矩阵。
// M88 探针里 `dstStride = N` 之所以没暴露这一点：它 `mSize` 恒 1（m=1 探针），行距不参与寻址。
template <uint32_t K, uint32_t N, uint32_t CM_STRIDE>
class PfAttnGemm {
    static_assert(K % AP_BASE_K == 0u, "K 必须被 AP_BASE_K=64 整除");
    static_assert(N % AP_BASE_N == 0u, "N 必须被 AP_BASE_N=128 整除（无尾块）");

public:
    static constexpr uint32_t Blocks() { return N / AP_BASE_N; }

    __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* cCol)
    {
        aGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a));
        bGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b));
        cGMOri.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cCol));
    }

    // mTile：第几个 m-tile；curM：本 tile 的有效行数（≤ AP_BASE_M，尾块用）；nBlock：N 块号
    __aicore__ inline void RunTile(uint32_t mTile, uint32_t curM, uint32_t nBlock)
    {
        // 3510 实测 quirk（M88/GDN 同一契约）：Nd2Nz 行数为 1 时不切 NZ ⇒ 计算侧统一 ≥2 行，
        // 多读的那 1 行由 host 保证可读（x 平面尾部留 1 行余量），写回仍用 curM mask。
        const uint32_t calcM = (curM < 2u) ? 2u : curM;
        const uint32_t calcMAlign = AP_CUBE_BLOCK * ((calcM + AP_CUBE_BLOCK - 1u) / AP_CUBE_BLOCK);
        constexpr uint32_t kLoop = K / AP_BASE_K;

        LocalTensor<bfloat16_t> a1P(TPosition::A1, L1_A0, AP_L1_A_ELEMS);
        LocalTensor<bfloat16_t> a1G(TPosition::A1, L1_A1, AP_L1_A_ELEMS);
        LocalTensor<bfloat16_t> b1P(TPosition::B1, L1_B0, AP_L1_B_ELEMS);
        LocalTensor<bfloat16_t> b1G(TPosition::B1, L1_B1, AP_L1_B_ELEMS);
        LocalTensor<bfloat16_t> a2P(TPosition::A2, L0_A0, AP_BASE_M * AP_BASE_K);
        LocalTensor<bfloat16_t> a2G(TPosition::A2, L0_A1, AP_BASE_M * AP_BASE_K);
        LocalTensor<bfloat16_t> b2P(TPosition::B2, L0_B0, AP_BASE_K * AP_BASE_N);
        LocalTensor<bfloat16_t> b2G(TPosition::B2, L0_B1, AP_BASE_K * AP_BASE_N);
        LocalTensor<float> cL0C(TPosition::CO1, 0, AP_BASE_M * AP_BASE_N);

        Mutex::Lock<PIPE_M>(M15G::BUF_AIC_L0C);
        for (uint32_t kBlock = 0u; kBlock < kLoop; ++kBlock) {
            const uint32_t p = kBlock & 1u;
            const MutexID bufA = p ? M15G::BUF_AIC_A1 : M15G::BUF_AIC_A0;
            const MutexID bufB = p ? M15G::BUF_AIC_B1 : M15G::BUF_AIC_B0;
            const MutexID bufL0 = p ? M15G::BUF_AIC_L01 : M15G::BUF_AIC_L00;
            LocalTensor<bfloat16_t> a1 = p ? a1G : a1P;
            LocalTensor<bfloat16_t> b1 = p ? b1G : b1P;
            LocalTensor<bfloat16_t> a2 = p ? a2G : a2P;
            LocalTensor<bfloat16_t> b2 = p ? b2G : b2P;

            Mutex::Lock<PIPE_MTE2>(bufA);
            CopyInA(a1, kBlock, mTile, calcM);
            Mutex::Unlock<PIPE_MTE2>(bufA);
            Mutex::Lock<PIPE_MTE2>(bufB);
            CopyInB(b1, kBlock, nBlock);
            Mutex::Unlock<PIPE_MTE2>(bufB);

            Mutex::Lock<PIPE_MTE1>(bufL0);
            Mutex::Lock<PIPE_MTE1>(bufA);
            Mutex::Lock<PIPE_MTE1>(bufB);
            LoadA(a1, a2, calcMAlign);
            LoadB(b1, b2);
            Mutex::Unlock<PIPE_MTE1>(bufA);
            Mutex::Unlock<PIPE_MTE1>(bufB);
            Mutex::Unlock<PIPE_MTE1>(bufL0);

            Mutex::Lock<PIPE_M>(bufL0);
            MmadParams mp = {};
            mp.m = calcM;
            mp.n = AP_BASE_N;
            mp.k = AP_BASE_K;
            mp.cmatrixInitVal = (kBlock == 0u);
            Mmad(cL0C, a2, b2, mp);
            Mutex::Unlock<PIPE_M>(bufL0);
        }
        Mutex::Unlock<PIPE_M>(M15G::BUF_AIC_L0C);
        Mutex::Lock<PIPE_FIX>(M15G::BUF_AIC_L0C);
        CopyOut(cL0C, mTile, nBlock, curM, calcMAlign);
        Mutex::Unlock<PIPE_FIX>(M15G::BUF_AIC_L0C);
    }

private:
    __aicore__ inline void CopyInA(LocalTensor<bfloat16_t>& a1, uint32_t kBlock, uint32_t mTile, uint32_t calcM)
    {
        Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = calcM;                 // 行数（≥2；3510 行数=1 退化的 quirk）
        par.dValue = AP_BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;                  // A = [m, K] 行主序
        par.dstNzC0Stride = AP_BASE_M;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        DataCopy(a1, aGMOri[static_cast<uint64_t>(mTile) * AP_BASE_M * K + kBlock * AP_BASE_K], par);
    }

    __aicore__ inline void CopyInB(LocalTensor<bfloat16_t>& b1, uint32_t kBlock, uint32_t nBlock)
    {
        Nd2NzParams par = {};
        par.ndNum = 1;
        par.nValue = AP_BASE_N;
        par.dValue = AP_BASE_K;
        par.srcNdMatrixStride = 0;
        par.srcDValue = K;                  // B = [N, K] 行主序（权重原样 (out,in)）
        par.dstNzC0Stride = AP_BASE_N;
        par.dstNzNStride = 1;
        par.dstNzMatrixStride = 0;
        DataCopy(b1, bGMOri[kBlock * AP_BASE_K + static_cast<uint64_t>(nBlock) * AP_BASE_N * K], par);
    }

    __aicore__ inline void LoadA(LocalTensor<bfloat16_t>& a1, LocalTensor<bfloat16_t>& a2, uint32_t calcMAlign)
    {
        LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = calcMAlign / AP_CUBE_BLOCK;
        lp.kStep = AP_BASE_K / AP_CUBE_BLOCK;
        lp.srcStride = AP_BASE_M / AP_CUBE_BLOCK;
        lp.dstStride = calcMAlign / AP_CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        LoadData(a2, a1[0], lp);
    }

    __aicore__ inline void LoadB(LocalTensor<bfloat16_t>& b1, LocalTensor<bfloat16_t>& b2)
    {
        LoadData2DParamsV2 lp = {};
        lp.mStartPosition = 0;
        lp.kStartPosition = 0;
        lp.mStep = AP_BASE_N / AP_CUBE_BLOCK;
        lp.kStep = AP_BASE_K / AP_CUBE_BLOCK;
        lp.srcStride = AP_BASE_N / AP_CUBE_BLOCK;
        lp.dstStride = AP_BASE_N / AP_CUBE_BLOCK;
        lp.sid = 0;
        lp.ifTranspose = false;
        LoadData(b2, b1[0], lp);
    }

    __aicore__ inline void CopyOut(LocalTensor<float>& cL0C, uint32_t mTile, uint32_t nBlock, uint32_t curM,
                                   uint32_t calcMAlign)
    {
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fp = {};
        fp.nSize = AP_BASE_N;
        fp.mSize = curM;                    // 尾块 mask：只写有效行
        fp.srcStride = calcMAlign;
        fp.dstStride = CM_STRIDE;           // **y0 平面的行距**（不是 GEMM 自己的 N）
        fp.quantPre = QuantMode_t::F322BF16;
        fp.reluScalar = 0;
        fp.vectorRelu = 0;
        fp.deqScalar = 0;
        Fixpipe(cGMOri[static_cast<uint64_t>(mTile) * AP_BASE_M * CM_STRIDE + nBlock * AP_BASE_N], cL0C, fp);
    }

    GlobalTensor<bfloat16_t> aGMOri;
    GlobalTensor<bfloat16_t> bGMOri;
    GlobalTensor<bfloat16_t> cGMOri;
};

// ============================================================
// 8. AIV：环行组装（raw k 256 B + 3×int64 位置尾 24 B → 280 B）
// ============================================================
// **为什么位置尾是本段产的**：环行 = `raw k[128] ‖ 3×int64 MRoPE 位置`（`RING_HEAD_SIZE = 140`）。
// M98 把整行当**声明输入**（它的 host 面直接生成），prefill 的融合路径必须自己把它组出来。
// 位置值取文本路径的三轴恒等形（`pos, pos, pos`）——依据 `m15_attn_prolog.h` 的 R4/R5
// （Qwen4Exp 的 position id 无条件是三轴恒等 ⇒ MRoPE 的置换在数值上恒等）。**本工程不做其它轴**。
// ⚠ **落盘通路按 `docs/05 §6.1 ⓔ`**：位置尾是**标量算出来的值**，落 GM 必须
//   「标量写 UB → 写出侧阻塞释放 BufferID release（`BufRelease<PIPE_S>`）→ 才 DMA 搬」，
//   **不得**标量直写 GM、**不得**只靠 `PipeBarrier`。写侧 `BufAcquire<PIPE_S>` 与
//   `BufRelease<PIPE_S>` **成对**（只换 release 不配 acquire ⇒ 挂死；M111 在**未合入**分支
//   `feat/m111-ple-scalar-gm-drain-release` 上实测过一次，本处的等价形态由 m24 的设备档复跑见证）。
__aicore__ inline void PfRingRowAssemble(__gm__ uint8_t* inPlane, __gm__ bfloat16_t* rawkSrc, uint32_t localRow,
                                         uint32_t pos, uint32_t mut)
{
    LocalTensor<bfloat16_t> rowL(TPosition::VECCALC, PF_UB_ROW, AC_IN_RAWK_STRIDE);
    GlobalTensor<bfloat16_t> src;
    src.SetGlobalBuffer(rawkSrc, RING_KEY_DIM);
    GlobalTensor<bfloat16_t> dst;
    // 目标 = 输入面里本行的环行槽（行距 AC_IN_RAWK_STRIDE，32 B 对齐）
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(inPlane) + AC_IN_RAWK_OFF / ELEM_BYTES +
                            static_cast<uint64_t>(localRow) * AC_IN_RAWK_STRIDE,
                        AC_IN_RAWK_STRIDE);

    if ((mut & PF_MUT_NO_RAWK) == 0u) {
        BufAcquire<PIPE_MTE2>(PF_BUF_ROW);
        DataCopy(rowL, src, Block1(RING_KEY_DIM * ELEM_BYTES));
        BufRelease<PIPE_MTE2>(PF_BUF_ROW);
    }
    {
        // 位置尾：3 个 int64，值都是 pos（低 32 位 = pos、高 32 位 = 0）⇒ UB 上就是 6 个 i32
        LocalTensor<int32_t> tailL(TPosition::VECCALC, PF_UB_ROW + RING_KEY_DIM * ELEM_BYTES, RING_TAIL_INT64 * 2u);
        BufAcquire<PIPE_S>(PF_BUF_ROW);     // 写侧成对（等 MTE2 的阻塞释放）
        for (uint32_t i = 0u; i < RING_TAIL_INT64; ++i) {
            tailL.SetValue(static_cast<int32_t>(2u * i), static_cast<int32_t>(pos));
            tailL.SetValue(static_cast<int32_t>(2u * i + 1u), 0);
        }
        BufRelease<PIPE_S>(PF_BUF_ROW);     // 阻塞释放：S pipe 排空（含上面 6 次标量写）
    }
    BufAcquire<PIPE_MTE3>(PF_BUF_ROW);      // 等 S pipe 的阻塞释放
    DataCopyPad(dst[0], rowL, M15H::ExtBlock1(RING_ROW_BYTES));
    BufRelease<PIPE_MTE3>(PF_BUF_ROW);
}

// ============================================================
// 9. AIV：(row × item) 的工作项
// ============================================================
__aicore__ inline void PfRunItem(const PfChunkArgs& a, uint32_t localRow, uint32_t pos, uint32_t item,
                                 __gm__ bfloat16_t* y0, __gm__ bfloat16_t* out, __gm__ bfloat16_t* in)
{
    __gm__ bfloat16_t* w = reinterpret_cast<__gm__ bfloat16_t*>(a.w);
    __gm__ bfloat16_t* csRow =
        reinterpret_cast<__gm__ bfloat16_t*>(a.cs) + static_cast<uint64_t>(pos) * CS_ROW_ELEMS;
    const uint32_t apMode = PfApMode(a.mut);
    __gm__ bfloat16_t* y0Row = y0 + static_cast<uint64_t>(localRow) * PF_Y0_STRIDE;
    __gm__ bfloat16_t* outRow = out + static_cast<uint64_t>(localRow) * PF_OUT_STRIDE;
    // cache 输入面的两段（行距只用 `m15_attn_cache.h` §1a 的常量）
    __gm__ bfloat16_t* kvRow =
        in + AC_IN_KV_OFF / ELEM_BYTES + static_cast<uint64_t>(localRow) * (AC_IN_KV_ROW_BYTES / ELEM_BYTES);

    if (item < IT_GATE) {                                   // ---- q 头 h：GemmaRMSNorm(256)+RoPE(64) → out.q
        const uint32_t h = item - IT_Q;
        AivNormRope<HD> nr;
        nr.Init(y0Row + Y0_QG + static_cast<uint64_t>(h) * 2u * HD, w + W_QN_OFF / 2u, csRow,
                outRow + OUT_Q + static_cast<uint64_t>(h) * HD, apMode);
        nr.Run();
    } else if (item < IT_K) {                               // ---- gate 头 h：原样抄写 → out.gate
        const uint32_t h = item - IT_GATE;
        AivCopy(y0Row + Y0_QG + static_cast<uint64_t>(h) * 2u * HD + HD, outRow + OUT_GATE + h * HD, HD);
    } else if (item < IT_V) {                               // ---- k 头 h：norm+rope → 主 KV 的 (h,K) 槽
        if ((a.mut & PF_MUT_NO_KV) == 0u) {
            const uint32_t h = item - IT_K;
            // 主 KV 的 (head, K/V) 相邻（`m15_attn_kv.h` 的 `M15KV_KV_IN_BLOCK_OFF`：
            // head 平面 + slot*KV_TOKEN_STRIDE + lane*KV_HEAD_CONTENT_BYTES）⇒ 输入面里同一个 head 的
            // K 槽与 V 槽就是 [K 512 B ‖ V 512 B]。**槽内偏移用宏算**（`KV_HEAD_DIM` 与
            // `KV_HEAD_CONTENT_BYTES` 是同一个量的两个名字，`AC_KV_SLOT_BYTES` 的 static_assert 钉着）。
            AivNormRope<HD> nr;
            nr.Init(y0Row + Y0_K + static_cast<uint64_t>(h) * HD, w + W_KN_OFF / 2u, csRow,
                    kvRow + static_cast<uint64_t>(h) * (2u * KV_HEAD_DIM), apMode);
            nr.Run();
        }
    } else if (item == IT_V) {                              // ---- v（2 头）→ 主 KV 的 (h,V) 槽
        if ((a.mut & PF_MUT_NO_KV) == 0u) {
            for (uint32_t h = 0u; h < KV_HEADS; ++h) {
                AivCopy(y0Row + Y0_V + static_cast<uint64_t>(h) * HD,
                        kvRow + static_cast<uint64_t>(h) * (2u * KV_HEAD_DIM) + KV_HEAD_DIM, HD);
            }
        }
    } else if (item < IT_RAWK) {                            // ---- indexer q 头 h：GemmaRMSNorm(128)+RoPE(64)
        if ((a.mut & PF_MUT_NO_QIDX) == 0u) {
            const uint32_t h = item - IT_IDXQ;
            AivNormRope<IDX_D> nr;
            nr.Init(y0Row + Y0_IDX + static_cast<uint64_t>(h) * IDX_D, w + W_IQN_OFF / 2u, csRow,
                    outRow + OUT_QIDX + static_cast<uint64_t>(h) * IDX_D, apMode);
            nr.Run();
        }
    } else {                                                // ---- raw k → 环行（含位置尾）
        PfRingRowAssemble(a.in, y0Row + Y0_IDX + IDX_Q, localRow, pos, a.mut);
    }
}

// ============================================================
// 10. 相位 A 的段体：一个 chunk 的 **(mTile × nBlock) AIC** + **(row × item) AIV** + cache 填
// ============================================================
// 同步序（**每个 chunk 一遍**；核内 buffer id、核间 CrossCore）：
//   AIC： [4 个 GEMM × (mTile,nBlock) 条带] → mode-0 全体 AIC 对齐（FIXP drain）
//          → mode-2 `PF_A2V_TILE` 放行配对 AIV → mode-2 `PF_V2A_DONE` 等 AIV 消费完 y0
//   AIV： mode-2 等 `PF_A2V_TILE` → (row × item) 全部项 → AIV mode-0 B1
//          → mode-2 `PF_V2A_DONE`（2 set 配 1 wait：AIC 侧的等待因此蕴含"全体 AIV 已过 B1"）
//          → block 0：cache 填（`m15_attn_cache_body` + `MainKvFillChunk` 逐行）
//          → AIV mode-0 B2（**保 `in` 输入面**：block 0 读完之前别的 AIV 不得开下一个 chunk）
// 为什么 `PF_V2A_DONE` 放在 B1 之后、cache 填之前：AIC 的下一个 chunk GEMM 只覆写 `y0`，
//   而 `y0` 在 B1 之前就已被全体 AIV 消费完 ⇒ 放行得越早、重叠越好；`in` 输入面的保护由 B2 承担。
__aicore__ inline void AttnPrefillChunkAic(const PfChunkArgs& a, uint32_t absRowStart, uint32_t rows)
{
    if ASCEND_IS_AIC {
        if ((a.mut & PF_MUT_NO_GEMM) == 0u) {
            PfAttnGemm<AP_HIDDEN, QG_W, PF_Y0_STRIDE> gq;
            PfAttnGemm<AP_HIDDEN, NKV * HD, PF_Y0_STRIDE> gk;
            PfAttnGemm<AP_HIDDEN, NKV * HD, PF_Y0_STRIDE> gv;
            PfAttnGemm<AP_HIDDEN, IDX_W, PF_Y0_STRIDE> gi;
            __gm__ uint8_t* xChunk = a.x + static_cast<uint64_t>(absRowStart) * AP_HIDDEN * ELEM_BYTES;
            gq.Init(xChunk, a.w + W_Q_OFF, a.y0 + Y0_QG * ELEM_BYTES);
            gk.Init(xChunk, a.w + W_K_OFF, a.y0 + Y0_K * ELEM_BYTES);
            gv.Init(xChunk, a.w + W_V_OFF, a.y0 + Y0_V * ELEM_BYTES);
            gi.Init(xChunk, a.w + W_IDX_OFF, a.y0 + Y0_IDX * ELEM_BYTES);

            const uint32_t bid = GetBlockIdx();
            const uint32_t nCores = GetBlockNum();          // mix(1,2) 的 AIC 视角 = blockDim
            // 工作项 = (mTile, nBlock) 展平：`g = mTile + nBlock * PF_M_TILES`
            for (uint32_t g = bid; g < PF_GEMM_ITEMS; g += nCores) {
                const uint32_t mTile = g % PF_M_TILES;
                const uint32_t nBlock = g / PF_M_TILES;
                const uint32_t rowBase = mTile * PF_M_TILE;
                if (rowBase >= rows) {
                    continue;                               // 尾 chunk 可能只填满第 0 个 tile
                }
                const uint32_t curM = ((rows - rowBase) < PF_M_TILE) ? (rows - rowBase) : PF_M_TILE;
                constexpr uint32_t BQ = PfAttnGemm<AP_HIDDEN, QG_W, PF_Y0_STRIDE>::Blocks();
                constexpr uint32_t BK = PfAttnGemm<AP_HIDDEN, NKV * HD, PF_Y0_STRIDE>::Blocks();
                if (nBlock < BQ) {
                    gq.RunTile(mTile, curM, nBlock);
                } else if (nBlock < BQ + BK) {
                    gk.RunTile(mTile, curM, nBlock - BQ);
                } else if (nBlock < BQ + 2u * BK) {
                    gv.RunTile(mTile, curM, nBlock - BQ - BK);
                } else {
                    gi.RunTile(mTile, curM, nBlock - BQ - 2u * BK);
                }
            }
        }
        // 全体 AIC 对齐：FIXP 写 GM 排空（= y0 全量可见）+ 所有核到齐
        CrossCoreSetFlag<AP_CC_M0, PIPE_FIX>(PF_AIC_M0_BAR);
        CrossCoreWaitFlag<AP_CC_M0, PIPE_S>(PF_AIC_M0_BAR);
        // **每 chunk 都要走同一套 set/wait**（含最后一个）：一侧少走一步就是 M88 r2 的死锁形态
        CrossCoreSetFlag<AP_CC_M2, PIPE_MTE2>(PF_A2V_TILE);   // 放行配对 AIV
        CrossCoreWaitFlag<AP_CC_M2, PIPE_S>(PF_V2A_DONE);     // 等 AIV 消费完 y0
    }
}

// chunk 内位置连续性的**可执行复核**（只在给表时做；不连续 ⇒ 不填 cache + 置契约标志）
__aicore__ inline bool PfPosContiguous(const PfPosRef& posRef, uint32_t absRowStart, uint32_t rows,
                                       uint32_t chunkStart)
{
    if (posRef.tbl == nullptr) {
        return true;        // `base + 行号` 按定义连续
    }
    for (uint32_t r = 0u; r < rows; ++r) {
        if (posRef.At(absRowStart + r) != chunkStart + r) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void AttnPrefillChunkAiv(const PfChunkArgs& a, uint32_t absRowStart, uint32_t rows)
{
    if ASCEND_IS_AIV {
        const uint32_t bid = GetBlockIdx();                 // 0 .. 2*numBlocks-1
        const uint32_t nAiv = GetBlockNum() * 2u;           // AIV 视角的 GetBlockNum() = AIC 数
        PfPosRef posRef;
        posRef.tbl = reinterpret_cast<__gm__ uint32_t*>(a.posTbl);
        posRef.base = a.posBase;

        CrossCoreWaitFlag<AP_CC_M2, PIPE_MTE2>(PF_A2V_TILE);   // 等本 chunk 的 y0 全量可见

        __gm__ bfloat16_t* y0 = reinterpret_cast<__gm__ bfloat16_t*>(a.y0);
        __gm__ bfloat16_t* out = reinterpret_cast<__gm__ bfloat16_t*>(a.out);
        __gm__ bfloat16_t* in = reinterpret_cast<__gm__ bfloat16_t*>(a.in);

        const uint32_t nItems = rows * PF_ITEMS;
        for (uint32_t i = bid; i < nItems; i += nAiv) {      // **(row × item) 二维工作项**
            const uint32_t localRow = i / PF_ITEMS;
            const uint32_t item = i - localRow * PF_ITEMS;
            PfRunItem(a, localRow, posRef.At(absRowStart + localRow), item, y0, out, in);
        }

        // B1：`in` 输入面（本 chunk 全部行的环行 + K/V 行）全体就位
        CrossCoreSetFlag<AP_CC_M0, PIPE_MTE3>(PF_AIV_BAR_ITEMS);
        CrossCoreWaitFlag<AP_CC_M0, PIPE_MTE2>(PF_AIV_BAR_ITEMS);

        // y0 已消费完 ⇒ 放行配对 AIC 跑下一个 chunk 的 GEMM（mode-2：2 set 配 1 wait）
        CrossCoreSetFlag<AP_CC_M2, PIPE_MTE3>(PF_V2A_DONE);

        if (bid == 0u) {
            const uint32_t chunkStart = posRef.At(absRowStart);
            const bool contig = PfPosContiguous(posRef, absRowStart, rows, chunkStart);
            if (contig) {
                // 输入面里的 `attn_idx_k_norm` 权重（128 bf16）—— cache 填按 M98 的输入面契约读它
                {
                    GlobalTensor<bfloat16_t> wSrc;
                    wSrc.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a.w) + W_IKN_OFF / ELEM_BYTES,
                                         RING_KEY_DIM);
                    GlobalTensor<bfloat16_t> wDst;
                    wDst.SetGlobalBuffer(in + AC_IN_W_OFF / ELEM_BYTES, RING_KEY_DIM);
                    LocalTensor<bfloat16_t> wL(TPosition::VECCALC, PF_UB_ROW, RING_KEY_DIM);
                    BufAcquire<PIPE_MTE2>(PF_BUF_ROW);
                    DataCopy(wL, wSrc, Block1(RING_KEY_DIM * ELEM_BYTES));
                    BufRelease<PIPE_MTE2>(PF_BUF_ROW);
                    BufAcquire<PIPE_MTE3>(PF_BUF_ROW);
                    DataCopy(wDst, wL, Block1(RING_KEY_DIM * ELEM_BYTES));
                    BufRelease<PIPE_MTE3>(PF_BUF_ROW);
                }
                M15AC::m15_attn_cache_body(a.in, a.cs, a.pooled, a.ring, a.comp, a.pack, a.packSeed, a.kv,
                                           a.flag, chunkStart, rows, a.acMode);
                // 主 KV：**整个 chunk 逐行落页**（M98 的探针档只落 1 行；两种用同一份宏、同一条代码）
                LocalTensor<bfloat16_t> kvL(TPosition::VECCALC, PF_UB_KV, AC_IN_KV_ROW_BYTES / ELEM_BYTES);
                const AscendC::DataCopyPadExtParams<bfloat16_t> padB{false, 0, 0, 0};
                M15AC::MainKvFillChunk(a.kv, in + AC_IN_KV_OFF / ELEM_BYTES,
                                       AC_IN_KV_ROW_BYTES / ELEM_BYTES, chunkStart, rows, kvL, PF_BUF_KV, padB);
            }
            // **契约标志**（`AC_FLAG_FE_CONTRACT`）：1 = 本 chunk 的位置不连续 ⇒ cache **没填**。
            // 它是"响亮失败"的落点（cache 保持毒值 ⇒ 判据变红），**不是**绝对断言。
            // 落盘通路同 §8 的标量条款（UB → S 阻塞释放 → MTE3）。
            {
                GlobalTensor<int32_t> fG;
                fG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(a.flag), AC_FLAG_LANES);
                LocalTensor<int32_t> oneL(TPosition::VECCALC, PF_UB_ROW, 8);
                BufAcquire<PIPE_S>(PF_BUF_ROW);
                oneL.SetValue(0, contig ? 0 : 1);
                BufRelease<PIPE_S>(PF_BUF_ROW);
                BufAcquire<PIPE_MTE3>(PF_BUF_ROW);
                DataCopyPad(fG[AC_FLAG_FE_CONTRACT], oneL, M15H::ExtBlock1(4));
                BufRelease<PIPE_MTE3>(PF_BUF_ROW);
            }
        }

        // B2：block 0 的 cache 填读完 `in` 输入面之前，任何 AIV 都不得开下一个 chunk 的项
        CrossCoreSetFlag<AP_CC_M0, PIPE_MTE3>(PF_AIV_BAR_CACHE);
        CrossCoreWaitFlag<AP_CC_M0, PIPE_MTE2>(PF_AIV_BAR_CACHE);
    }
}

// 相位 A 的入口：**一次调用只跑一个 chunk**（`a.nChunks == 1`；`nChunks == 0` 按 1 处理）。
//
// ⚠ **`nChunks > 1` 在本 mission 是"响亮失败"（`Trap`），不是"能用但慢"**：多 chunk 单次发射时
//    AIC 臂与 AIV 臂在若干轮之后失步（M116 设备读数：终态 y0 整行是毒值、门控 8 lane 全是毒值，
//    而 packed 的 seed 行却是对的 —— 形态是同步失步，不是数据通路错）。同一个段体的**逐 chunk 形态**
//    （`nChunks = 1`）在 4097 行的 33 次启动上全绿 ⇒ **只交付并验证单 chunk**。Wave C 要按 chunk 启动
//    `ceil(m / PF_ROWS)` 次（挂载点补丁见 `m24_attn_prefill/README.md` §5.8）。
//    为什么用 `Trap` 而不是"静默只跑第一个 chunk"或"跑完但结果错"：后两者会让调用方拿到**看起来正常
//    的输出**（找不到错在哪）；`Trap` 是核内防呆的既有手法（同 `M15G::Block1()` 对非 32 B 倍数的处理）。
//    **未取到的读数**：这条硬拦是复审后新加的，加拦之后**没有设备复跑读数**（设备档冻结）；
//    它只在 `nChunks > 1` 时触发 ⇒ 上面那些 `nChunks = 1` 的已验证路径**一个字节都没变**。
//    修好"多 chunk 单次发射"那条之后（建议：chunk 循环里的 mode-0 barrier 与 mode-2 信号改成
//    4 槽 id 轮转，照 MoE 段 `FLAG_*_RING[4]`），把这道 `Trap` 拿掉即可。
__aicore__ inline void AttnPrefillPhaseA(const PfChunkArgs& a)
{
    if (a.nChunks > 1u) {
        AscendC::Trap();     // 见上面的说明：多 chunk 单次发射未通过验证 ⇒ 响亮失败
    }
    const uint32_t nChunks = (a.nChunks == 0u) ? 1u : a.nChunks;
    for (uint32_t c = 0u; c < nChunks; ++c) {
        const uint32_t rowsOff = c * a.chunkRows;
        if (rowsOff >= a.seqRows) {
            break;
        }
        uint32_t rows = a.chunkRows;
        if (rows > (a.seqRows - rowsOff)) {
            rows = a.seqRows - rowsOff;                     // 尾 chunk（4097 = 32×128 + 1）
        }
        const uint32_t absRowStart = a.rowStart + rowsOff;
        // AIC 与 AIV 都进各自的臂（**两边都进**才保证 set/wait 配平 —— M88 r2 的死锁就是这个形态）
        AttnPrefillChunkAic(a, absRowStart, rows);
        AttnPrefillChunkAiv(a, absRowStart, rows);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace M15PF

#endif  // M15_ATTN_PREFILL_H
