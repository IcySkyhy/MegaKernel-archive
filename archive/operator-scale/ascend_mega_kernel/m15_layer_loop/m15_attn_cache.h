// ============================================================
// m15_attn_cache.h —— M98：attention **cache 填数学**（M82 显式 descope 的第二项）
//
// M82（`m15_attn_kv.h`，已合入 `10db7fa`）把三套 cache + packed indices 的**物理字节布局**冻死了，
// 但它自己写明 descope 了「填 cache 的数学」。M88 补上了 prolog（q/k/qidx 的 norm+rope）。
// 本文件是第三块：**把 prolog 的产物按布局写进 cache**，含 raw ring 的压缩器池化数学。
//
// ---------------------------------------------------------------------------
// 【1. 几何（每条给权威处；本文件**不新增**任何几何权威）】
//
// A. raw key ring（`M15KV_RING_*`，`m15_attn_kv.h::RING_HEAD_SIZE` 等 @fc83496:134-153）
//    行号基准 = 本 mission tip `fc83496`（不可变 commit）；符号名（内容锚点）优先于行号
//    · 行 = 128 bf16 raw k ‖ 3×int64 MRoPE 位置尾 = 140 bf16 = **280 B**
//      官方：`QSAKeyStateCache.__init__`（`vllm/models/qwen4_exp/common/qsa_cache.py:814-826`）
//      `storage_head_size = ceil(128/4)*4 + 3*4 = 140`；
//      `bind_kv_cache`（同文件 :828-836）把 `qsa_cache[..., 128:].view(torch.int64)` 当位置尾。
//    · 容量 = `4*ceil((4+num_spec)/4)` = 4 行 = **1,120 B/请求/层**
//      官方：`get_kv_cache_spec`（同文件 :838-856）`span = ratio + num_spec; capacity = ratio*ceil(span/ratio)`。
//    · 槽 = `physical_block*capacity + position % capacity`（单请求恒 1 物理块 ⇒ `pos % 4`）
//      官方：`circular_qsa_slot_mapping`（同文件 :108-156）与 metadata kernel（同文件 :279-294）。
//    · **写入门控（易漏的一条）**：环**只写 query chunk 末尾 capacity 行**
//      官方：`qsa_cache.py:144-149` 的 `keep = rows + compressor_state_size >= request_ends`；
//      metadata kernel `qsa_cache.py:283` 的 `token_idx + circular_buffer_size >= query_end`。
//    · **对齐**：280 B % 32 = **24** ⇒ 单行传输只能 `DataCopyPad`
//      （`m15_attn_kv.h::static_assert(RING_ROW_BYTES % 32u != 0u…)` @fc83496:145-151；
//       `M15G::Block1()` 对非 32 B 倍数直接 `Trap`，
//      见 `m15_gdn_layer.h:91-97`）。不用 Pad 会怎样：见 host 的 `Ac.neg.ring.block1`。
//
// B. compressed key cache（`M15KV_COMP_*`，`m15_attn_kv.h::COMP_ROW_BYTES` @fc83496:155-165）
//    · 一行 256 B = 128 bf16；页 = 4 行 = 1,024 B（`MLAAttentionSpec.block_size=16` +
//      `tokens_per_state=4` ⇒ `num_states = 16/4 = 4`，`page_size_bytes = 1*4*(128+0)*2 = 1,024`；
//      `vllm/v1/kv_cache_interface.py:506-528`）。
//    · 槽 = `physical_block*4 + (group % 4)`，组 = `pos / 4`
//      官方：`compressed_qsa_slot_mapping`（`qsa_cache.py:159-187`）与 metadata kernel `:295-317`。
//    · **门控（off-by-one）**：只有 `(pos+1) % 4 == 0` 才产生一行
//      官方：`qsa_cache.py:179-182`、`:301`；`M15KV_COMP_ROW_WRITTEN`（`m15_attn_kv.h::M15KV_COMP_ROW_WRITTEN` @fc83496:260）。
//
// C. **填数学本身**（本文件的主体）
//    · 池化：`pooled[c] = (Σ_{i=0..3} raw_k[成员 i][c]) / 4`，fp32 累加、**结果落 bf16**
//      官方：`_compress_qsa_groups_kernel`（`qwen4_exp/nvidia/ops/qsa.py:397-433`，`accumulator`
//      为 fp32、`componentwise / COMPRESS_RATIO` 存进 `pooled`，而 `pooled` 的 dtype = `raw_keys.dtype`
//      = bf16，见 `ops/qsa.py:1001-1005`）。
//    · 组成员来源：位置 ≥ chunk 起点的成员取**本 step 的 raw k 行**，否则取**环**里的旧成员
//      官方：`ops/qsa.py:401-427` 的 `use_raw = position >= chunk_start_position`，
//      环地址 `compressor_state_cache_ptr + block*stride + (position % SIZE)*stride_token`（`:415-426`）。
//    · 顺序：**先池化（读环）再写环**
//      官方：`indexer_qsa.py:342-353`（compress，读环）在 `:373-383`（`qsa_store_cache_rows`，写环）之前。
//    · norm+rope：`k_comp = gemma_rmsnorm(pooled, k_layernorm, eps)` →
//      `apply_qsa_rope(..., pos = 组首位置)`（`indexer_qsa.py:354-367`）；组首位置 = `pos - 4 + 1`
//      （`ops/qsa.py:436`）。**这两步不重写**：直接复用 M88 已合入的
//      `M15AP::AivNormRope<IDX_D>`（`m15_attn_prolog_probe.h:233-419`；GemmaRMSNorm 乘 `(1+w)` 见 R3）。
//
// D. 主 KV（`M15KV_KV_*`）—— **⚠ M98 已按塔裁 A 更正**（旧值是页 16,384 B/K-only/head 外层）
//    官方：`FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=256, head_size_v=256)`
//    （`qwen4_exp/nvidia/qsa.py:434-442`）⇒ 页 = `2*16*(256+256)*2` = **32,768 B**，
//    K/V 同页相邻（`off_V = off_K + 512`，`flash_attn.py:1525-1526` 的
//    `kv_cache.transpose(1,2).split(head_size,-1)` + `reshape_and_cache_flash`）。
//    **塔裁 A 已授权在本 mission 内更正 `m15_attn_kv.h` 的主 KV 几何**（四路依据 + 独立交叉校验写在
//    那个头文件里），本文件因此**直接引用** `KV_BLOCK_BYTES / KV_HEAD_PLANE_BYTES / KV_TOKEN_STRIDE /
//    KV_HEAD_CONTENT_BYTES / M15KV_KV_BYTE_OFF_PHYS`，**不再自带第二份数字**。
//    raw ring / compressed / packed 三套本来就与官方一致，未受影响。
//
// ---------------------------------------------------------------------------
// 【2. 输入溯源三分（docs/17 §1.3）】
//   ① 声明输入（合法，host 按 seed 生成、字节落盘可核）：`acInDev` 的三段
//      （本 chunk 的 raw k 行 = **环行形态**「raw k 128 bf16 ‖ 3×int64 位置尾」、
//       `attn_idx_k_norm` 权重 128 bf16、主 KV 的 k/v 行）、`acCsDev` 的 cos/sin 表、
//       `acPackSeedDev` 的 packed 行内容、`acRingDev` 的**环初始状态**（= 上一 chunk 的写入；
//       「写环」这条腿本身由本段的 T1 判据 `Ac.ring.row.*` 覆盖）。
//   ② 上游输出：本段**不**吃任何设备上游产物 —— prolog 的输出被当作①的声明输入由 host 端生成
//      （它们在 M88 里已由 `Ap.*` 判据覆盖，但本段不依赖那一步的运行时产物）。
//   ③ 被判量自身的产物：**无**。判据的期望值一律由 host 侧独立实现从①算出。
// ============================================================
#ifndef M15_ATTN_CACHE_H
#define M15_ATTN_CACHE_H

#include "m15_attn_kv.h"
#include "m15_attn_prolog.h"
#include "m15_attn_prolog_probe.h"   // M15AP::AivNormRope / AP_*_TO_* / AP_MODE_* / VL

namespace M15AC {

using namespace AscendC;
using namespace AscendC::Reg;
using namespace M15Kv;
using namespace M15AP;

// ============================================================
// 1. 一次 launch 的规模 = **一个 chunk**
// ============================================================
// 【M116：探针尺度 → chunk 尺度】
//   M98 首版按**探针**定尺（8 行 / 2 条压缩行），它自己的 README §5 第 3 项写明「真实规模档未跑」。
//   M116（Wave B2）把它抬到 chunk 尺度，理由与上界逐条如下：
//     · **上界来自 UB**：本文件的 UB 窗（§3）随行数线性涨 —— 128 行时窗顶 = 117,312 B，仍在 248 KB
//       硬限内（`static_assert(AC_UB_END <= 248*1024)`）；而整段 prefill（4097 行）的 raw k 行窗
//       就要 4097×288 B ≈ 1.2 MB ⇒ **整段驻 UB 不可能**，chunk 分次是结构性的。
//     · **128 = 2 × `M15AP::AP_BASE_M`（= 64）** ⇒ 一个 chunk 内前端的 AIC GEMM **正好跨 2 个 m-tile**
//       （`m15_attn_prefill.h` 的 `PF_M_TILE`）。「AIC 的 N 块分派加 m-tile」这条要求因此有真活可跑，
//       m-tile 不是一个恒 1 的空循环。
//     · **4097 = 32×128 + 1**（尾块 1 行）⇒ 「m=1 与 m=4097」两档在同一条序列里都被走到；
//       m=1 的**独立**档（pos 4096，fresh cache）另跑一次。
//   **与 M98 探针的相容性**：探针档（`m15_attn_cache_host.h`）传的 chunkStart/chunkRows 不变
//   （22 / 8 行），函数体只是把「行数上限」放大 ⇒ 读数与判据**一字未变**（本文件的入口签名、
//   `in` 输入面布局、`AC_MODE_*` 语义全部保持）。唯一被改的**行为**是主 KV 的落行数（见
//   `MainKvFillChunk` 的说明：探针档仍落 1 行，chunk 路径落 chunkRows 行）。
constexpr uint32_t AC_MAX_ROWS = 128;        // 一次最多处理的本 chunk 行数
// 权威式（组 = 4 token）；**不写第二份数字**
constexpr uint32_t AC_MAX_GROUPS = AC_MAX_ROWS / COMP_TOKENS_PER_STATE;   // 32 条
static_assert(AC_MAX_ROWS % COMP_TOKENS_PER_STATE == 0u,
              "chunk 行数必须是组长的整数倍（否则「chunk 内的组」会跨 chunk 边界，门控要重推）");
static_assert(AC_MAX_ROWS / M15AP::AP_BASE_M == 2u,
              "chunk 必须是 2 个 m-tile（PF_M_TILE = AP_BASE_M）：m-tile 循环要有真活");
constexpr uint32_t AC_RING_PHYS_BLK = 0;     // 单请求：环的物理块号恒 0（`qsa_cache.py:129` 取 block_table[req][0]）

// ---- 主 KV：**直接用 `m15_attn_kv.h` 的（M98 已更正过的）官方几何**，本文件不再自带第二份 ----
// （M98 首版曾把官方几何写成独立的 `AC_OFFICIAL_KV_*` 并标注"待塔裁"；塔裁 A 已授权更正权威头，
//  故这里改为直接引用 `M15KV_KV_*`，**消除第二份数字**。）
constexpr uint32_t AC_MAIN_KV_BYTES = KV_LAYER_STRIDE;                     // 257 页 × 32,768 B = 8,421,376
static_assert(AC_MAIN_KV_BYTES == PREFILL_BLOCKS * KV_BLOCK_BYTES, "主 KV 平面 = 页数 × 页字节");

// ---- 输入面 `acInDev`（① 声明输入；各段起点 32 B 对齐）----
constexpr uint32_t AC_IN_RAWK_OFF = 0;                                       // [AC_MAX_ROWS][140] bf16（环行形态）
constexpr uint32_t AC_IN_RAWK_STRIDE = RING_HEAD_SIZE + 4u;                  // 144 bf16 = 288 B（行距 32 B 对齐）
// ⚠ 上面是**元素**（bf16）步长；host 侧对 `uint8_t*` 做指针算术时**必须**用下面这个字节步长
constexpr uint32_t AC_IN_RAWK_STRIDE_BYTES = AC_IN_RAWK_STRIDE * ELEM_BYTES;   // 288 B
constexpr uint32_t AC_IN_RAWK_ROW_BYTES = RING_ROW_BYTES;                    // 280 B/行（140 bf16）
constexpr uint32_t AC_IN_RAWK_BYTES = AC_MAX_ROWS * AC_IN_RAWK_STRIDE * ELEM_BYTES;   // **派生式**（不写活值字面量）
constexpr uint32_t AC_IN_W_OFF = AC_IN_RAWK_OFF + AC_IN_RAWK_BYTES;          // attn_idx_k_norm [128] bf16
constexpr uint32_t AC_IN_W_BYTES = RING_KEY_DIM * ELEM_BYTES;                // 256
constexpr uint32_t AC_IN_KV_OFF = AC_IN_W_OFF + AC_IN_W_BYTES;               // [行][h0K|h0V|h1K|h1V]
// ⚠ 这里的 512 是 **head_dim×2 B**（一个 head 的 K 或 V 的字节数），**不是** `COMP_ROW_BYTES`（128 dims）
constexpr uint32_t AC_KV_SLOT_BYTES = KV_HEAD_DIM * ELEM_BYTES;              // 512 B
constexpr uint32_t AC_IN_KV_ROW_BYTES = KV_HEADS * KV_LANES * AC_KV_SLOT_BYTES;   // 2,048 B/行（2 head × (K512+V512)）
constexpr uint32_t AC_IN_KV_BYTES = AC_MAX_ROWS * AC_IN_KV_ROW_BYTES;        // **派生式**（不写活值字面量）
constexpr uint32_t AC_IN_BYTES = AC_IN_KV_OFF + AC_IN_KV_BYTES;              // 输入面总字节（派生）
static_assert(AC_IN_KV_ROW_BYTES == 2u * KV_TOKEN_STRIDE,
              "一行 = 2 head × (K 512 + V 512) = 2,048 B = 2 × 页内 token 步长（1,024 B）");
static_assert(AC_KV_SLOT_BYTES == KV_HEAD_DIM * ELEM_BYTES && AC_KV_SLOT_BYTES == KV_HEAD_CONTENT_BYTES &&
                  AC_KV_SLOT_BYTES != COMP_ROW_BYTES,
              "KV 槽 512 B（256 dims）与 compressed 行 256 B（128 dims）不是同一个量");
static_assert(AC_IN_KV_ROW_BYTES % 32u == 0u, "主 KV 输入行 2,048 B 可按块搬");
static_assert(AC_IN_RAWK_STRIDE * ELEM_BYTES % 32u == 0u, "raw k 输入行距 288 B 是 32 B 整数倍");
static_assert(AC_IN_RAWK_ROW_BYTES % 32u == 24u, "环行 280 B ⇒ 24 (mod 32)（与 RING_ROW_BYTES 同源）");
static_assert(AC_IN_RAWK_OFF % 32u == 0u && AC_IN_W_OFF % 32u == 0u && AC_IN_KV_OFF % 32u == 0u &&
                  AC_IN_BYTES % 32u == 0u,
              "输入面各段起点 32 B 对齐");

// ---- 其他 host 侧缓冲尺寸 ----
constexpr uint32_t AC_CS_BYTES = CS_NPOS * CS_ROW_BYTES;                     // 4,097 × 128 = 524,416（同 M88 的表布局）
constexpr uint32_t AC_PACK_ROWS = 2;                                         // 本段写 2 个 packed 行
constexpr uint32_t AC_PACK_SEED_BYTES = AC_PACK_ROWS * PACK_ROW_BYTES;       // 16,416
constexpr uint32_t AC_RING_BYTES = RING_LAYER_BYTES;                         // 1,120（1 层）
constexpr uint32_t AC_COMP_BYTES = COMP_LAYER_STRIDE;                        // 263,168（1 层：1,028 行 × 256 B）
constexpr uint32_t AC_PACK_BYTES = AC_PACK_ROWS * PACK_ROW_BYTES;            // 16,416
constexpr uint32_t AC_POOLED_BYTES = AC_MAX_GROUPS * RING_KEY_DIM * ELEM_BYTES;   // **派生式**（不写活值字面量）

// ============================================================
// 2. 档（`m15_attn_cache_host.h` 逐档跑一次；方向级负向对照 + 「把被测对象弄坏」）
// ============================================================
constexpr uint32_t AC_MODE_CONTRACT = 0u;       // 契约档 = 官方语义
constexpr uint32_t AC_MODE_SUM_NOT_MEAN = 1u;   // 方向：池化用 Σ 不除 4（`ops/qsa.py:431` 反向）
constexpr uint32_t AC_MODE_PLAIN_NORM = 2u;     // 方向：norm 乘 `w` 而非 `(1+w)`（R3 反向）
constexpr uint32_t AC_MODE_ROPE_LAST_POS = 3u;  // 方向：RoPE 用组**末**位置而非组首（`ops/qsa.py:436` 反向）
constexpr uint32_t AC_MODE_NO_RING_READ = 4u;   // 方向：跨 chunk 成员不从环读（按 0 算）
constexpr uint32_t AC_MODE_NO_RING_STORE = 5u;  // **变体五**：不写环 ⇒ 环判据必须变红
constexpr uint32_t AC_MODE_NO_COMP_STORE = 6u;  // **变体五**：不写压缩行 ⇒ 压缩判据必须变红
constexpr uint32_t AC_MODE_OFFBYONE = 7u;       // 负向：边界判据写成 `pos % 4 == 0`（差一位）
constexpr uint32_t AC_MODE_RING_SPLIT = 8u;     // 正对照：单行 280 B 拆成 256 B + 24 B，结果必须逐字节相同
constexpr uint32_t AC_MODE_RING_BLOCK1 = 9u;    // **不用 DataCopyPad 会怎样**：`Block1(280)` ⇒ `Trap`
constexpr uint32_t AC_MODE_N = 10u;

// ============================================================
// 3. UB 窗（编译期静态分配；起点全部 32 B 对齐；与 M15AP 的窗 [0, 14336) 不重叠）
// ============================================================
constexpr uint32_t AC_UB_ROW_ELEMS = AC_IN_RAWK_STRIDE;                 // 每行在 UB 里占 144 bf16（288 B）
constexpr uint32_t AC_UB_BASE = 32768u;
constexpr uint32_t AC_UB_ROWS = AC_UB_BASE;                             // **派生式**：AC_MAX_ROWS × 288 B（本 chunk 的 raw k 行）
constexpr uint32_t AC_UB_RINGR = AC_UB_ROWS + AC_MAX_ROWS * AC_UB_ROW_ELEMS * ELEM_BYTES;   // **派生式**：AC_MAX_GROUPS×4 × 288 B（环成员）
constexpr uint32_t AC_UB_POOL = AC_UB_RINGR + AC_MAX_GROUPS * 4u * AC_UB_ROW_ELEMS * ELEM_BYTES;   // 256 B 池化行
constexpr uint32_t AC_UB_ZERO = AC_UB_POOL + RING_KEY_DIM * ELEM_BYTES;                      // 256 B 零窗（方向档用）
constexpr uint32_t AC_UB_KV = AC_UB_ZERO + RING_KEY_DIM * ELEM_BYTES;                        // 2,048 B 主 KV 暂存
constexpr uint32_t AC_UB_PACK = AC_UB_KV + AC_IN_KV_ROW_BYTES;                               // 8,208 B packed 行
// ⚠ PACK_ROW_BYTES ≡ 16 (mod 32) ⇒ FLAG 段必须**上取整对齐**，否则它的起点不是 32 B 倍数
constexpr uint32_t AC_UB_FLAG = (AC_UB_PACK + PACK_ROW_BYTES + 31u) / 32u * 32u;             // 32 B 门控标志
constexpr uint32_t AC_UB_END = AC_UB_FLAG + 32u;
static_assert(AC_UB_ROWS % 32u == 0u && AC_UB_RINGR % 32u == 0u && AC_UB_POOL % 32u == 0u &&
                  AC_UB_ZERO % 32u == 0u &&
                  AC_UB_KV % 32u == 0u && AC_UB_PACK % 32u == 0u && AC_UB_FLAG % 32u == 0u,
              "UB 各段起点 32 B 对齐（DataCopyPad 的 UB 侧要求）");
static_assert(AC_UB_END <= 248u * 1024u, "本段 UB 用量必须落在 248 KB 硬限内");
static_assert(AC_UB_BASE >= 16384u, "与 M15AP 的 UB 窗 [0, UB_AP_END=14,336) 不重叠");

// 核内 buffer id（**本段独占**；避开 M15AP 用的 0/1/2）：
//   AC_BUF_IN  : MTE2（行/环成员）→ PIPE_V 与 PIPE_MTE3 的读侧
//   AC_BUF_OUT : PIPE_V（池化行）→ MTE3（写 GM scratch）
//   AC_BUF_PACK: MTE2（packed 行）→ MTE3（写 GM）
//   AC_BUF_FLAG: **S（标量写门控 lane）→ MTE3**（M116 新增）：`docs/05 §6.1 ⓔ` 的标量落盘
//                「写侧 acquire → 写 → 阻塞释放（mode=false）→ 才搬」需要一个**专属** token，
//                否则与 AC_BUF_IN 的行搬运互相阻塞（S 侧的释放会把整条 MTE2 流水也等上）
constexpr AscendC::MutexID AC_BUF_IN = static_cast<AscendC::MutexID>(10);
constexpr AscendC::MutexID AC_BUF_OUT = static_cast<AscendC::MutexID>(11);
constexpr AscendC::MutexID AC_BUF_PACK = static_cast<AscendC::MutexID>(12);
constexpr AscendC::MutexID AC_BUF_FLAG = static_cast<AscendC::MutexID>(13);

// 门控标志的 lane 语义（host 按同一套读）
constexpr uint32_t AC_FLAG_RING_WRITTEN = 0;    // 本档实际写了几行环
constexpr uint32_t AC_FLAG_COMP_WRITTEN = 1;    // 本档实际写了几行压缩
constexpr uint32_t AC_FLAG_CHUNK_START = 2;
constexpr uint32_t AC_FLAG_CHUNK_ROWS = 3;
constexpr uint32_t AC_FLAG_MODE = 4;
constexpr uint32_t AC_FLAG_RING_READ = 5;       // 从环读了几个成员
constexpr uint32_t AC_FLAG_COMP_MASK = 6;       // 写过的组（bit = 组序）
// **M116：lane 7 归前端（B2）用**——「本 chunk 的位置表不连续 ⇒ cache **没填**」的契约标志。
// 本函数每次 launch 都把它清 0（下面 §4 的第 8 次 SetValue）；前端在 cache 填**之后**覆盖它。
// 语义 = 0：cache 填正常（本 lane 之外 7 个 lane 才有效）；1：cache 被**跳过**（其余 lane 是旧值）。
constexpr uint32_t AC_FLAG_FE_CONTRACT = 7;
constexpr uint32_t AC_FLAG_LANES = 8;

// ============================================================
// 3c. M116：主 KV 的落页 —— **chunk 尺度**，一行一落
// ============================================================
// 为什么单列成参数化函数 ①寻址单一来源 ②两条调用路共用同一份公式：
//   · M98 的探针档（`m15_attn_cache_host.h`，**不在 M116 的 scope**）对主 KV 的期望是
//     「只落 chunk 起点的 **1** 行、其余仍是毒值」（判据 `Ac.mainkv.kv`）⇒ 探针路必须传 `rows = 1`；
//   · chunk 路径（`m15_attn_prefill.h`）要落**整个 chunk** 的每一行 ⇒ 传 `chunkRows`。
//   同一份代码、同一批宏，`rows = 1` 时与 M98 的读数**逐字节等价**。
// ⚠ **每一行的页号/槽号都必须经 `M15KV_KV_BLOCK_OF` / `M15KV_KV_SLOT_OF` 算**（这里直接用
//   `M15KV_KV_BYTE_OFF_CONTIG`，它的内部就是这两个宏）。手写 `/16`、`%16` 在 M98 更正过主 KV 几何
//   之后属于**第二份数字**：旧几何算出的 stride 恰好是一半，而且是"写错地方也不越界"的**静默错**。
// ⚠ **只支持恒等 block_table（单请求）**：本式 = 连续布局 = 恒等表的特例（`m15_attn_kv.h` D2）。
//   `pfBlockTable != nullptr` 的 paged 多请求形态**不在 B2 的范围**（显式写窄，见 README §未完成）。
__aicore__ inline void MainKvFillChunk(__gm__ uint8_t* mkvPlane, __gm__ bfloat16_t* srcBase,
                                       uint32_t srcStrideElems, uint32_t posStart, uint32_t rows,
                                       AscendC::LocalTensor<bfloat16_t>& stage, AscendC::MutexID buf,
                                       const AscendC::DataCopyPadExtParams<bfloat16_t>& padB)
{
    AscendC::GlobalTensor<bfloat16_t> mkvG;
    mkvG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(mkvPlane), AC_MAIN_KV_BYTES / ELEM_BYTES);
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint32_t pos = posStart + r;
        AscendC::GlobalTensor<bfloat16_t> src;
        src.SetGlobalBuffer(srcBase + static_cast<uint64_t>(r) * srcStrideElems, srcStrideElems);
        BufAcquire<PIPE_MTE2>(buf);
        DataCopyPad(stage, src, AscendC::DataCopyExtParams{1, AC_IN_KV_ROW_BYTES, 0, 0, 0}, padB);
        BufRelease<PIPE_MTE2>(buf);
        BufAcquire<PIPE_MTE3>(buf);
        for (uint32_t h = 0u; h < KV_HEADS; ++h) {
            for (uint32_t kv = 0u; kv < 2u; ++kv) {
                const uint64_t off = M15KV_KV_BYTE_OFF_CONTIG(pos, h, kv, 0u) / ELEM_BYTES;
                DataCopyPad(mkvG[off],
                            stage[static_cast<uint64_t>(h) * 2u * KV_HEAD_DIM +
                                  static_cast<uint64_t>(kv) * KV_HEAD_DIM],
                            M15H::ExtBlock1(KV_HEAD_DIM * ELEM_BYTES));
            }
        }
        BufRelease<PIPE_MTE3>(buf);
    }
}

// ============================================================
// 4. 设备体：本 chunk 的 raw k 行 → 环；组完成处 → 压缩行；packed；主 KV（官方几何）
//    ⚠ 单一核（AIV block 0）串行做完所有段：本档判的是寻址/几何/填数学，**不是吞吐**。
//      chunk 尺度（128 行）下这个限制仍然在（B2 交的是**填对的 cache**；多核切分见 README 的未完成项）。
//      不引入任何跨核同步（前端那两条在 `m15_attn_prefill.h`）⇒ 本函数里没有 PIPE_S 的跨核语义；
//      核内值依赖用 buffer id + `PipeBarrier<PIPE_ALL>` 表达（**例外**：标量落 GM 的通路见 §4 末）。
// ============================================================
__aicore__ inline void m15_attn_cache_body(__gm__ uint8_t* in, __gm__ uint8_t* cs, __gm__ uint8_t* pooledGm,
                                           __gm__ uint8_t* ringGm, __gm__ uint8_t* compGm, __gm__ uint8_t* packGm,
                                           __gm__ uint8_t* packSeedGm, __gm__ uint8_t* mainKvGm,
                                           __gm__ uint8_t* flagGm, uint32_t chunkStart, uint32_t chunkRows,
                                           uint32_t mode)
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0u) {
            return;
        }
        if (chunkRows > AC_MAX_ROWS) {
            chunkRows = AC_MAX_ROWS;
        }
        const uint32_t chunkEnd = chunkStart + chunkRows;

        GlobalTensor<bfloat16_t> inG;
        inG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(in), AC_IN_BYTES / ELEM_BYTES);
        GlobalTensor<bfloat16_t> csG;
        csG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cs), AC_CS_BYTES / ELEM_BYTES);
        GlobalTensor<bfloat16_t> poolG;
        poolG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(pooledGm), AC_POOLED_BYTES / ELEM_BYTES);
        GlobalTensor<bfloat16_t> ringG;
        ringG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(ringGm), AC_RING_BYTES / ELEM_BYTES);
        GlobalTensor<bfloat16_t> compG;
        compG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(compGm), AC_COMP_BYTES / ELEM_BYTES);
        GlobalTensor<int32_t> packG;
        packG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packGm),
                              static_cast<uint64_t>(AC_PACK_ROWS) * PACK_COLS);
        GlobalTensor<int32_t> packSeedG;
        packSeedG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(packSeedGm),
                                  static_cast<uint64_t>(AC_PACK_ROWS) * PACK_COLS);
        GlobalTensor<bfloat16_t> mkvG;
        mkvG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(mainKvGm), AC_MAIN_KV_BYTES / ELEM_BYTES);
        GlobalTensor<int32_t> flagG;
        flagG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(flagGm), AC_FLAG_LANES);

        LocalTensor<bfloat16_t> rowL(TPosition::VECCALC, AC_UB_ROWS, AC_MAX_ROWS * AC_UB_ROW_ELEMS);
        LocalTensor<bfloat16_t> rngL(TPosition::VECCALC, AC_UB_RINGR, AC_MAX_GROUPS * 4u * AC_UB_ROW_ELEMS);
        LocalTensor<bfloat16_t> poolL(TPosition::VECCALC, AC_UB_POOL, RING_KEY_DIM);
        LocalTensor<bfloat16_t> zeroL(TPosition::VECCALC, AC_UB_ZERO, RING_KEY_DIM);
        LocalTensor<bfloat16_t> kvL(TPosition::VECCALC, AC_UB_KV, AC_IN_KV_ROW_BYTES / ELEM_BYTES);
        LocalTensor<int32_t> packL(TPosition::VECCALC, AC_UB_PACK, PACK_COLS);
        LocalTensor<int32_t> flagL(TPosition::VECCALC, AC_UB_FLAG, AC_FLAG_LANES);
        const AscendC::DataCopyPadExtParams<bfloat16_t> padB{false, 0, 0, 0};
        const AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};

        // ---- (0) 候选组：每个候选 = 一条"要写压缩行"的**组完成位置** p，成员 = p-3..p ----
        //  契约档：组序从 chunk 起点推，p 必须落在本 chunk 内**且**过 `COMP_ROW_WRITTEN` 门控
        //          （官方 `qsa_cache.py:179-182`）。
        //  OFFBYONE：把边界判据写成 `p % 4 == 0`（差一位的经典形态）—— 候选改由"本 chunk 里
        //          `p % 4 == 0` 的行"给出，且**不过**边界门控。
        uint32_t candP[AC_MAX_GROUPS];
        uint32_t nCand = 0u;
        if (mode == AC_MODE_OFFBYONE) {
            for (uint32_t r = 0u; r < chunkRows && nCand < AC_MAX_GROUPS; ++r) {
                const uint32_t q = chunkStart + r;
                if ((q % COMP_TOKENS_PER_STATE) == 0u) {
                    candP[nCand++] = q;
                }
            }
        } else {
            for (uint32_t g = 0u; g < AC_MAX_GROUPS; ++g) {
                const uint32_t p = (chunkStart / COMP_TOKENS_PER_STATE + g + 1u) * COMP_TOKENS_PER_STATE - 1u;
                if (p >= chunkStart && p < chunkEnd && M15KV_COMP_ROW_WRITTEN(p)) {
                    candP[nCand++] = p;
                }
            }
        }

        uint32_t ringRead = 0u;
        for (uint32_t c = 0u; c < nCand; ++c) {
            for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
                if (candP[c] - (COMP_TOKENS_PER_STATE - 1u) + i < chunkStart) {
                    ringRead++;
                }
            }
        }

        // ---- (1) MTE2：本 chunk 的 raw k 行（环行形态 280 B/行）+ 跨 chunk 成员的环行 ----
        //  **先读环**（官方顺序：compress 读环 `indexer_qsa.py:342-353` → store 写环 `:373-383`），
        //  否则本 chunk 末尾那几行（槽号会绕回）会把还要用的旧成员盖掉。
        BufAcquire<PIPE_MTE2>(AC_BUF_IN);
        for (uint32_t r = 0u; r < chunkRows; ++r) {
            DataCopyPad(rowL[r * AC_UB_ROW_ELEMS], inG[static_cast<uint64_t>(r) * AC_IN_RAWK_STRIDE],
                        AscendC::DataCopyExtParams{1, AC_IN_RAWK_ROW_BYTES, 0, 0, 0}, padB);
        }
        for (uint32_t c = 0u; c < nCand; ++c) {
            for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
                const uint32_t q = candP[c] - (COMP_TOKENS_PER_STATE - 1u) + i;
                if (q >= chunkStart) {
                    continue;   // 本 step 的成员：直接用 rowL
                }
                DataCopyPad(rngL[(c * COMP_TOKENS_PER_STATE + i) * AC_UB_ROW_ELEMS],
                            ringG[static_cast<uint64_t>(M15KV_RING_SLOT(q)) * RING_HEAD_SIZE],
                            AscendC::DataCopyExtParams{1, AC_IN_RAWK_ROW_BYTES, 0, 0, 0}, padB);
            }
        }
        BufRelease<PIPE_MTE2>(AC_BUF_IN);

        __ubuf__ bfloat16_t* rowUb = reinterpret_cast<__ubuf__ bfloat16_t*>(rowL.GetPhyAddr());
        __ubuf__ bfloat16_t* rngUb = reinterpret_cast<__ubuf__ bfloat16_t*>(rngL.GetPhyAddr());
        __ubuf__ bfloat16_t* poolUb = reinterpret_cast<__ubuf__ bfloat16_t*>(poolL.GetPhyAddr());

        // ---- (2)+(3) 池化（VF）→ 压缩行（GemmaRMSNorm + RoPE@组首，复用 M88 的 AivNormRope）----
        uint32_t compWritten = 0u;
        uint32_t compMask = 0u;
        BufAcquire<PIPE_V>(AC_BUF_IN);   // 行/环成员就位（覆盖全部候选）
        __ubuf__ bfloat16_t* zeroUb = reinterpret_cast<__ubuf__ bfloat16_t*>(zeroL.GetPhyAddr());
        // zero 窗（方向档 NO_RING_READ 用）：**在 VF 之外**把它清零，从而池化循环里**没有任何分支**
        // （bisheng 对"向量循环内的标量分支/continue"会崩在 SimplifyCFG 的 removeUnreachableBlocks）
        __VEC_SCOPE__
        {
            RegTensor<float> zf;
            RegTensor<bfloat16_t> zb;
            MaskReg all = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zf, 0.0f, all);
            Cast<bfloat16_t, float, AP_F32_TO_BF16>(zb, zf, all);
            for (uint16_t ch = 0u; ch < static_cast<uint16_t>(RING_KEY_DIM / VL); ++ch) {
                StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(zeroUb + ch * VL, zb, all);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        // 池化的缩放：SUM_NOT_MEAN 档不除 4（`ops/qsa.py:431` 的方向反写）。**提到 VF 外**当标量。
        const float poolScale =
            (mode == AC_MODE_SUM_NOT_MEAN) ? 1.0f : (1.0f / static_cast<float>(COMP_TOKENS_PER_STATE));
        for (uint32_t c = 0u; c < nCand; ++c) {
            const uint32_t p = candP[c];
            const uint32_t first = p - (COMP_TOKENS_PER_STATE - 1u);
            // 4 个成员的 UB 源指针（标量算好；跨 chunk 成员指向环窗，方向档指向 zero 窗）——
            // **不在 VF 内做分支/取指针**，只用现成的标量。
            __ubuf__ bfloat16_t* mp[COMP_TOKENS_PER_STATE];
            for (uint32_t i = 0u; i < COMP_TOKENS_PER_STATE; ++i) {
                const uint32_t q = first + i;
                if (q >= chunkStart) {
                    mp[i] = rowUb + (q - chunkStart) * AC_UB_ROW_ELEMS;
                } else if (mode == AC_MODE_NO_RING_READ) {
                    mp[i] = zeroUb;
                } else {
                    mp[i] = rngUb + (c * COMP_TOKENS_PER_STATE + i) * AC_UB_ROW_ELEMS;
                }
            }
            BufAcquire<PIPE_V>(AC_BUF_OUT);   // 等上一轮 MTE3 读完 pooled 窗
            __VEC_SCOPE__
            {
                RegTensor<float> acc;
                RegTensor<float> t;
                RegTensor<bfloat16_t> sb;
                RegTensor<bfloat16_t> yb;
                MaskReg all = CreateMask<float, MaskPattern::ALL>();
                for (uint16_t ch = 0u; ch < static_cast<uint16_t>(RING_KEY_DIM / VL); ++ch) {
                    Duplicate(acc, 0.0f, all);
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, mp[0] + ch * VL);
                    Cast<float, bfloat16_t, AP_BF16_TO_F32>(t, sb, all);
                    Add(acc, acc, t, all);
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, mp[1] + ch * VL);
                    Cast<float, bfloat16_t, AP_BF16_TO_F32>(t, sb, all);
                    Add(acc, acc, t, all);
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, mp[2] + ch * VL);
                    Cast<float, bfloat16_t, AP_BF16_TO_F32>(t, sb, all);
                    Add(acc, acc, t, all);
                    LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(sb, mp[3] + ch * VL);
                    Cast<float, bfloat16_t, AP_BF16_TO_F32>(t, sb, all);
                    Add(acc, acc, t, all);
                    Muls(acc, acc, poolScale, all);
                    Cast<bfloat16_t, float, AP_F32_TO_BF16>(yb, acc, all);
                    StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(poolUb + ch * VL, yb, all);
                }
            }
            BufRelease<PIPE_V>(AC_BUF_OUT);   // pooled 窗写完（release mode=false）
            AscendC::PipeBarrier<PIPE_ALL>();
            if (mode != AC_MODE_NO_COMP_STORE) {
                BufAcquire<PIPE_MTE3>(AC_BUF_OUT);   // 等 PIPE_V 的 release
                DataCopyPad(poolG[static_cast<uint64_t>(c) * RING_KEY_DIM], poolL,
                            M15H::ExtBlock1(RING_KEY_DIM * ELEM_BYTES));
                BufRelease<PIPE_MTE3>(AC_BUF_OUT);
                // pooled 的 MTE3 写 GM → 下面 AivNormRope 的 MTE2 读同一地址：核内 drain
                AscendC::PipeBarrier<PIPE_ALL>();
                const uint32_t grp = M15KV_COMP_GROUP_OF(p);
                const uint32_t firstPos = grp * COMP_TOKENS_PER_STATE;   // 组首位置（`ops/qsa.py:436`）
                const uint32_t ropePos = (mode == AC_MODE_ROPE_LAST_POS) ? p : firstPos;
                const uint64_t compOff = M15KV_COMP_ROW_OFF_PHYS(grp / COMP_ROWS_PER_BLOCK, grp);
                const uint32_t nmode = (mode == AC_MODE_PLAIN_NORM) ? AP_MODE_PLAIN_NORM : AP_MODE_CONTRACT;
                AivNormRope<IDX_D> nr;
                nr.Init(reinterpret_cast<__gm__ bfloat16_t*>(pooledGm) + static_cast<uint64_t>(c) * RING_KEY_DIM,
                        reinterpret_cast<__gm__ bfloat16_t*>(in) + AC_IN_W_OFF / ELEM_BYTES,
                        reinterpret_cast<__gm__ bfloat16_t*>(cs) + static_cast<uint64_t>(ropePos) * CS_ROW_ELEMS,
                        reinterpret_cast<__gm__ bfloat16_t*>(compGm) + compOff / ELEM_BYTES, nmode);
                nr.Run();
                compWritten++;
                compMask |= (1u << (grp % COMP_ROWS_PER_BLOCK));
            }
        }
        BufRelease<PIPE_V>(AC_BUF_IN);

        // ---- (4) 写环（**单行 280 B 必须 DataCopyPad**；含首/末行与槽绕回）----
        //  门控（官方 `qsa_cache.py:144-149` / `:283`）：只写"本 chunk 末尾 capacity 行"。
        //  三个档：CONTRACT（一次 280 B Pad）、RING_SPLIT（256 B 块 + 24 B Pad，正对照）、
        //          RING_BLOCK1（`Block1(280)`：**非 32 B 倍数 ⇒ Trap**）。
        uint32_t ringWritten = 0u;
        BufAcquire<PIPE_MTE3>(AC_BUF_IN);   // 行窗的读侧（等 MTE2 写完 + PIPE_V 消费完）
        for (uint32_t r = 0u; r < chunkRows; ++r) {
            const uint32_t q = chunkStart + r;
            if (q + RING_ROWS_PER_BLOCK < chunkEnd) {
                continue;   // 非末尾 capacity 行：本 chunk 不写（槽里是上一 chunk 的值）
            }
            if (mode == AC_MODE_NO_RING_STORE) {
                continue;   // **变体五**：把"填环"这段整个短路
            }
            const uint64_t rowOff = M15KV_RING_ROW_OFF(AC_RING_PHYS_BLK, q) / ELEM_BYTES;
            if (mode == AC_MODE_RING_SPLIT) {
                DataCopyPad(ringG[rowOff], rowL[r * AC_UB_ROW_ELEMS], M15H::ExtBlock1(RING_KEY_DIM * ELEM_BYTES));
                DataCopyPad(ringG[rowOff + RING_KEY_DIM], rowL[r * AC_UB_ROW_ELEMS + RING_KEY_DIM],
                            M15H::ExtBlock1(RING_TAIL_BYTES));
            } else if (mode == AC_MODE_RING_BLOCK1) {
                DataCopy(ringG[rowOff], rowL[r * AC_UB_ROW_ELEMS], M15H::Block1(RING_ROW_BYTES));
            } else {
                DataCopyPad(ringG[rowOff], rowL[r * AC_UB_ROW_ELEMS], M15H::ExtBlock1(RING_ROW_BYTES));
            }
            ringWritten++;
        }

        BufRelease<PIPE_MTE3>(AC_BUF_IN);   // 收 (4) 的读侧

        // ---- (5) packed 行（8,208 B 行距 ≡ 16 (mod 32) ⇒ 单行必须 DataCopyPad）----
        //  中转窗 packL 是 MTE2 写 → MTE3 读：用**独立 id** 把这两个 pipe 排上
        for (uint32_t row = 0u; row < AC_PACK_ROWS; ++row) {
            BufAcquire<PIPE_MTE2>(AC_BUF_PACK);
            DataCopyPad(packL, packSeedG[static_cast<uint64_t>(row) * PACK_COLS],
                        AscendC::DataCopyExtParams{1, PACK_ROW_BYTES, 0, 0, 0}, padI);
            BufRelease<PIPE_MTE2>(AC_BUF_PACK);
            BufAcquire<PIPE_MTE3>(AC_BUF_PACK);
            DataCopyPad(packG[static_cast<uint64_t>(row) * PACK_COLS], packL, M15H::ExtBlock1(PACK_ROW_BYTES));
            BufRelease<PIPE_MTE3>(AC_BUF_PACK);
        }

        // ---- (6) 主 KV：**官方几何**（⚠ 见文件头 §D；恒等 block_table）----
        //  ⚠ **本探针档固定落 1 行**（= chunk 起点的行）：M98 的 host 判据 `Ac.mainkv.kv` 期望的
        //    就是「chunk 起点那 1 行被写、其余仍毒值」，而 `m15_attn_cache_host.h` **不在 M116 的
        //    scope** ⇒ 这条行为一字不动。chunk 尺度（整个 chunk 逐行落页）由同一个
        //    `MainKvFillChunk` 提供，调用方是前端（`m15_attn_prefill.h`）—— 寻址公式只有一份。
        MainKvFillChunk(mainKvGm, reinterpret_cast<__gm__ bfloat16_t*>(in) + AC_IN_KV_OFF / ELEM_BYTES,
                        AC_IN_KV_ROW_BYTES / ELEM_BYTES, chunkStart, 1u, kvL, AC_BUF_IN, padB);

        // ---- (7) 门控标志（host 读它，不靠猜）----
        //  ⚠ **顺序形态按 `docs/05 §6.1 ⓔ` 的标量条款**（M116 修，M107 把它登记为 docs/20 §5-B10
        //    的「未建立」同族项）：标量值落 GM 必须 **写侧 `BufAcquire<PIPE_S>` → 标量写 UB →
        //    `BufRelease<PIPE_S>`（= `RlsBufInternal<PIPE_S,false>`，`false`=CANN `ASC_LOCK_BLOCK`
        //    默认；M181 起与 acquire 侧同模式）→ 才由 MTE3 搬**。
        //    原先的「标量写 → `PipeBarrier<PIPE_ALL>` → `BufAcquire<MTE3>`」按 `docs/06:76` 逐字
        //    「`PipeBarrier<PIPE_X>` 只能阻塞标量等 pipe 指令退休，**不能保证 UB 数据路径对另一
        //    pipe 可见**」⇒ **不构成可见性契约**（docs/20 §3.4 的表里本处就填「未建立」）。
        //  ⚠ **成对纪律（M111 r1 在真机上踩过）**：写侧的 `BufAcquire<PIPE_S>` 与
        //    `BufRelease<PIPE_S>` **必须成对** —— 只把 barrier 换成 `BufRelease<PIPE_S>` 而**不配**
        //    写侧 acquire ⇒ **挂死**。（M111 在**未合入**分支 `feat/m111-ple-scalar-gm-drain-release` 上
        //    的读数；本处的等价形态由 M116 自己的 m24 设备档复跑见证。）
        BufAcquire<PIPE_S>(AC_BUF_FLAG);
        flagL.SetValue(AC_FLAG_RING_WRITTEN, static_cast<int32_t>(ringWritten));
        flagL.SetValue(AC_FLAG_COMP_WRITTEN, static_cast<int32_t>(compWritten));
        flagL.SetValue(AC_FLAG_CHUNK_START, static_cast<int32_t>(chunkStart));
        flagL.SetValue(AC_FLAG_CHUNK_ROWS, static_cast<int32_t>(chunkRows));
        flagL.SetValue(AC_FLAG_MODE, static_cast<int32_t>(mode));
        flagL.SetValue(AC_FLAG_RING_READ, static_cast<int32_t>(ringRead));
        flagL.SetValue(AC_FLAG_COMP_MASK, static_cast<int32_t>(compMask));
        flagL.SetValue(AC_FLAG_FE_CONTRACT, 0);   // 前端契约 lane：本函数每跑一次清 0（前端随后可能置 1）
        BufRelease<PIPE_S>(AC_BUF_FLAG);    // mode=false（CANN ASC_LOCK_BLOCK 默认）：与写侧 acquire 成对
        BufAcquire<PIPE_MTE3>(AC_BUF_FLAG);
        DataCopyPad(flagG[0], flagL, M15H::ExtBlock1(AC_FLAG_LANES * 4u));
        BufRelease<PIPE_MTE3>(AC_BUF_FLAG);
    }
}

__global__ __mix__(1, 2) void m15_attn_cache_kernel(__gm__ uint8_t* in, __gm__ uint8_t* cs, __gm__ uint8_t* pooledGm,
                                                    __gm__ uint8_t* ringGm, __gm__ uint8_t* compGm,
                                                    __gm__ uint8_t* packGm, __gm__ uint8_t* packSeedGm,
                                                    __gm__ uint8_t* mainKvGm, __gm__ uint8_t* flagGm,
                                                    uint32_t chunkStart, uint32_t chunkRows, uint32_t mode)
{
    AscendC::InitSocState();
    m15_attn_cache_body(in, cs, pooledGm, ringGm, compGm, packGm, packSeedGm, mainKvGm, flagGm, chunkStart,
                        chunkRows, mode);
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace M15AC

#endif  // M15_ATTN_CACHE_H

// ============================================================
// M116：prefill 前端的段体（本 mission 新建，落 `m15_attn_prefill.h`）
//
// **为什么由本头在末尾 include**：与 `m15_attn_kv.h` 末尾 include 本文件是同一手法 ——
// `.asc` 的 include 清单（`m15_layer_loop/m15_layer_loop.asc`）归 M97/在飞 mission，**Wave B 不碰**；
// 而 `m15_attn_kv.h` 已经在 `.asc:63` 被引入 ⇒ 只要本文件末尾 include 前端，**融合 TU 一字不改**
// 就把 B2 的段体带进去了。位置也在 `namespace M15Run {` 之前（设备代码必须落在文件作用域）。
// 顺序：kv → cache → prefill（prefill 需要 `M15AC::m15_attn_cache_body` / `MainKvFillChunk`，
// 而它们是上面刚定义完的）——`#include` 卫哨保证互相 include 不会递归。
#include "m15_attn_prefill.h"
