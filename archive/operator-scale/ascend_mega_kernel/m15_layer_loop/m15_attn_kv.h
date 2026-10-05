// ============================================================
// m15_attn_kv.h —— M82：attention 侧 **三套 cache + packed indices 的物理字节布局**（唯一权威头文件）
//
// 本头文件把「主 KV / raw key ring / compressed key cache / packed indices」四者的**物理字节布局**
// 冻结成编译期常量 + 映射函数。它是 M33 清单 ★13 与 docs/15 §5.3 ★13 的落点：融合 kernel 的
// attention 相位（相位 A）与将来所有消费者都必须**只**经过这里的函数算地址，不许各自手写偏移。
//
// 规范来源（逐条可核）：
//   · 三套 cache 的权威表  = `docs/14-hyperconnection-ple-indexer-spec.md:616-627`（§7.3 两侧 paged cache）
//   · raw ring 宽度更正    = `docs/14:664-666`（纠正 `docs/11-attn-analysis.md:25` 的 `[tokens,128]`）
//   · paged 契约"现在就定死" = `docs/15-prefill-design.md:528-531`（★13）
//   · 压缩行 off-by-one    = `docs/17-verification-standard.md:286`、`docs/11-attn-analysis.md:158`
//   · 行为权威（实现级）    = `m19_qsa_indexer/README.md:90-123`（§1.4/§1.5）
//   · m19 实现（只读引用）  = `m19_qsa_indexer/m19_qsa_indexer.asc:1494-1495`（closes/groupIdx）
//
// ---------------------------------------------------------------------------
// 【设计决定，带时点】2026-09-27（M82，本分支首次落盘）
//
// D1. raw key ring 行宽取**规范宽 `head_size = 140`**（= 128 raw k + 12 个 bf16 = 3 个 int64
//     MRoPE 位置尾，280 B/行）。**这不是"官方一定要求 140"的恒真断言**，而是在下列事实下的取舍：
//       · **推导链已逐环核对（源码 + config，2026-09-27 M82 直接读 doner 仓库）**：
//         `QSAKeyStateCache(head_size=k, cache_rope_positions=cache_rope_positions)`
//         （`/workspace/vllm/vllm/models/qwen4_exp/common/qsa_cache.py:808-826`）里
//           `storage_head_size = ceil(k/4)*4 + (cache_rope_positions ? 3*4 : 0)`
//         而 `cache_rope_positions = vllm_config.model_config.uses_mrope`
//         （`qwen4_exp/nvidia/indexer_qsa.py:167`、`qwen4_exp/amd/indexer_qsa.py:129`）；
//         `uses_mrope = _mrope_section(config) is not None`
//         （`/workspace/vllm/vllm/transformers_utils/config.py:676-713`），它扫
//         `rope_parameters.{mrope_section,xdrope_section}`；
//         **本 checkpoint 的 config.json 里 `text_config.rope_parameters.mrope_section = [11, 11, 10]`
//         （present）** ⇒ `uses_mrope = True` ⇒ `cache_rope_positions = True` ⇒
//         `storage_head_size = 128 + 3*4 = 140`。
//         （顺带一致：`mrope_section` 的和 = 32 = 组数 3 个轴 × partial rope；`partial_rotary_factor=0.25`
//          ⇒ 主 attention 的 rope 维 = 256×0.25 = 64，与 specs 的 RoPE(64) 吻合。）
//       · 但**运行期见证未做**：本 mission 没有跑 vLLM 去观测那一块 cache 的实际分配 shape
//         （无 device/dump 见证）⇒ 上面是**源码推导**，不是运行期取证。
//       · 文本-only 下若**不**要位置尾（`cache_rope_positions = False`），行宽就是
//         `ceil(128/4)*4 = 128`（**注意不是 128 的巧合：doner 会按 4 元素向上圆整**）。这正是 m19
//         现在的形态（`ring[4,128]`）：组首位置恒等于 `p-3`，3 个 MRoPE 位置**可由位置号推出**。
//       · 140 > 128 ⇒ 只存 128 的实现**不需要改 stride**，仍落在同一块已分配的行里（前 256 B/行）；
//         反向（先按 128 分配、后改 140）需要重分配（环是跨层长寿状态 ⇒ 返工面大）。
//     ⇒ 默认 140；`RING_HEAD_SIZE_ALT = 128` 仅用于把 m19 形态与本布局对拍时**按元素比较**
//       （见 `RingRowBytesAlt()`）；**任何分配/编址都不使用 128**。
//       【仍未取证的一条】doner 侧 `cache_rope_positions` 是模型侧开关；本工程若将来要跑
//       **文本-only 且明确不要位置尾**的形态，必须显式记一次"退回 128"的决定（改 `RING_HEAD_SIZE`
//       与两处 stride，不涉及主 KV/compressed 的布局）。
//
// D2. paged 契约**现在就按 paged 写死**（`docs/15:528-531`）：单请求下 `block_table` 为**恒等**
//     （`block_table[r][b] == b`），但所有地址都经 `KvByteOffsetPaged()` 算 —— 首期连续布局只是
//     "恒等 block_table"这一特例，**不是**另一套布局。判据（T-KV-PAGED）：对同一 token 位置，
//     连续地址与经 `block_table` 映射后的 paged 地址读到的 head 维向量**逐字节相同**；
//     负向对照 = 给一个非恒等 `block_table`，判据必须 FAIL。
//     ⚠ 该负向对照由 **host 的表驱动**（`m15_attn_kv_host.h::H_KvNegPaged`：非恒等表 + `MODE_CONTRACT`
//       启动探针），**不占探针的 mode 号**。M98 r2 起 `MODE_BROKEN_OLDGEOM = 2u` 是**主 KV 旧几何**
//       那条负向对照（详见本文件主 KV 段与 `Kv.neg.oldgeom(.dev)`），与 D2 无关。
//       （r1 复审 F4c：原文写「`m15_attn_kv_probe.h` 的 mode=2」在新档加入后指向了另一条对照。）
//
// D3. 首期验收基准 = **自建稠密 causal 参考**。**本基准不是官方行为**：官方 QSA 的
//     `indexer_budget=2048` 是 token 预算 ⇒ 每 token 只 attend 约一半历史（`docs/17:281`），
//     稠密 causal 与官方输出在任何长度下都不可能一致（gather 集合与 softmax 分母都不同）。
//     ⇒ 本分支**任何地方都不写「m=4097 对齐官方输出」**。
// ---------------------------------------------------------------------------
//
// 【对齐约束（实测算术，全部有 static_assert 钉住）】
//   · 主 KV 页 32,768 B、head 平面 16,384 B、页内 token 步长 1,024 B（= K512+V512）、compressed 行 256 B
//     —— 三者都是 32 B 的整数倍，可直接 `DataCopy`/`Block1()`。
//   · **raw key ring 单行 280 B ⇒ `280 % 32 == 24`：行是 8 B 对齐但*不是* 32 B 对齐**。
//     ⇒ 单行搬运必须走 `DataCopyPad`/`ExtBlock1`（或按行内两段 256 B + 24 B 拆分），
//       绝不能对单行用 `Block1(280)`（m15 的 `Block1` 会把它折成 8.75 个 32 B 块）。
//       （整环 1,120 B = 35×32 B 是 32 B 对齐的，所以"整环一次拷"是合法的、但语义不是我们要的。）
//   · **packed indices 单行 8,208 B ⇒ `8208 % 32 == 16`：行是 16 B 对齐但*不是* 32 B 对齐**。
//     ⇒ 单行搬运同样必须走 `DataCopyPad`；`[m,2052]` 整块（m 行连续）是 32 B 对齐的。
//   · 层 stride：主 KV `257×32,768`、compressed `1,028×256`、ring `1,120` 全部 32 B 对齐。
// ============================================================
#ifndef M15_ATTN_KV_H
#define M15_ATTN_KV_H

#include "m15_loop_layout.h"

namespace M15Kv {

using namespace M15Loop;
using namespace M15G;

// ============================================================
// 0. dtype / 形状常量（三套 cache 共用的"物理视图"语言）
// ============================================================
// 层类型 3:1（每 4 层 1 个 full attention）—— attention 层在 **attention 序列**里的序号 k
// 就是它在本布局里的层号（`m15_loop_layout.h` 的 `AttnSlot()`），k = 0..N_ATTN-1。
constexpr uint32_t ELEM_BYTES = 2;                        // bf16

// ---- 主 KV（paged，官方 `[blocks, H=2, N=16, C=512]` bf16；C = **K(256) ‖ V(256)**）----
// ⚠ **本节在 M98 被更正过**（塔裁 A，2026-09-27）。旧值（页 16,384 B / token 步长 512 B /
//   head 平面 8,192 B / 轴序 head 外层 / **整个头文件没有 V 的去处**）与官方不符；四路依据：
//     · spec：`FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=256, **head_size_v=256**)`
//       `/workspace/vllm/vllm/models/qwen4_exp/nvidia/qsa.py:434-442`
//     · 物理形状与 K‖V 拼接（**决定性**：主 KV 不可能没有 V 通道）——`do_kv_cache_update` 里
//       `# (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))` +
//       `key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` +
//       `reshape_and_cache_flash`
//       `/workspace/vllm/vllm/v1/attention/backends/flash_attn.py:1525-1540`（被 `nvidia/qsa.py:480`
//       调用；attention 读侧同式 `nvidia/qsa.py:215`）
//     · 页字节算式：`page_size_bytes = num_heads * num_states * state_content_size_bytes`
//       `/workspace/vllm/vllm/v1/kv_cache_interface.py:506-528` ⇒ `2 × (16/1) × ((256+256)×2)` = 32,768
//     · 独立交叉校验（不依赖上面三条）：经典 vLLM 布局 `2(K,V) × 16 token × 2 kv_head × 256 dim × 2 B`
//       = 32,768 B/页
//   ⇒ 页内（**同一 head 内**）：`off_K = slot*1024 + dim*2`、`off_V = off_K + 512`；head 步长 16,384 B。
//   旧几何作为**负向对照**保留（`m15_attn_kv_host.h` 的 `Kv.neg.oldgeom(.dev)` 与探针的
//   `MODE_BROKEN_OLDGEOM`）—— 它的作用不再是被使用，而是证明"新的判据能咬住这个方向上的错"。
//   M82 的教训（塔记）：**"布局唯一权威"不能只靠断言**，必须对着官方**分配器的算式**交叉校验。
constexpr uint32_t KV_BLOCK_TOKENS = 16;                  // vLLM block_size（docs/11:100）
constexpr uint32_t KV_HEADS = 2;                          // N2（GQA 的 KV 头数）
constexpr uint32_t KV_HEAD_DIM = 256;                     // K 的 head_dim
constexpr uint32_t KV_HEAD_DIM_V = 256;                   // V 的 head_dim（官方 `head_size_v = head_dim`）
constexpr uint32_t KV_ELEM_BYTES = ELEM_BYTES;
constexpr uint32_t KV_HEAD_CONTENT_BYTES = KV_HEAD_DIM * KV_ELEM_BYTES;       // 512 B：一个 head 槽里的 K（或 V）
constexpr uint32_t KV_HEAD_CONTENT_ELEMS = KV_HEAD_DIM + KV_HEAD_DIM_V;       // 512 元素：一个 head 槽里的 K‖V
constexpr uint32_t KV_TOKEN_ELEMS = KV_HEAD_CONTENT_ELEMS;                    // 512（**一个 head 槽内**的 K‖V 元素数）
constexpr uint32_t KV_TOKEN_BYTES = KV_TOKEN_ELEMS * KV_ELEM_BYTES;           // 1,024 B（一个 head 槽内的 K‖V）
constexpr uint32_t KV_TOKEN_STRIDE = KV_TOKEN_BYTES;                          // 1,024 B：**页内同一 head 内**的 token 步长
constexpr uint32_t KV_HEAD_PLANE_BYTES = KV_BLOCK_TOKENS * KV_TOKEN_STRIDE;   // 16,384 B（一个 head 的整页）
constexpr uint32_t KV_BLOCK_BYTES = KV_HEADS * KV_HEAD_PLANE_BYTES;           // 32,768 B/页
constexpr uint32_t KV_TOKEN_ALL_ELEMS = KV_HEADS * KV_TOKEN_ELEMS;            // 1,024：一个 token 在 2 头上合计
constexpr uint32_t KV_TOKEN_ALL_BYTES = KV_TOKEN_ALL_ELEMS * KV_ELEM_BYTES;   // 2,048 B/token（2 head × (K512+V512)）
constexpr uint32_t KV_LANE_K = 0;                                             // 槽内 K 的 lane
constexpr uint32_t KV_LANE_V = 1;                                             // 槽内 V 的 lane（= K + 512 B）
constexpr uint32_t KV_LANES = 2;
static_assert(KV_BLOCK_BYTES == KV_BLOCK_TOKENS * KV_TOKEN_ALL_BYTES, "页字节 = 16 token × 2,048 B/token");
static_assert(KV_BLOCK_BYTES == 32768u,
              "官方 `page_size_bytes`：2 head × 16 token × (256+256) × 2 B = 32,768 B"
              "（`v1/kv_cache_interface.py:506-528` + `qwen4_exp/nvidia/qsa.py:434-442`）");
static_assert(KV_TOKEN_STRIDE == 1024u && KV_HEAD_PLANE_BYTES == 16384u,
              "token 步长 1,024 B（= K512+V512）、head 步长 16,384 B（= N*C*2）");
static_assert(KV_HEAD_DIM_V == KV_HEAD_DIM, "官方 head_size_v = head_dim（`qwen4_exp/nvidia/qsa.py:437-439`）");
static_assert(KV_BLOCK_BYTES % 32u == 0u && KV_HEAD_PLANE_BYTES % 32u == 0u &&
              KV_TOKEN_STRIDE % 32u == 0u && KV_HEAD_CONTENT_BYTES % 32u == 0u, "主 KV 可直接 DataCopy");

// ---- raw key ring（压缩器状态环；每请求一个环、终生 1 个物理块）----
constexpr uint32_t RING_KEY_DIM = 128;                    // raw k 的维度（indexer 的 KV 头维度）
constexpr uint32_t RING_TAIL_INT64 = 3;                   // 3 个 int64 MRoPE 位置
constexpr uint32_t RING_TAIL_BYTES = RING_TAIL_INT64 * 8u;                              // 24 B
constexpr uint32_t RING_TAIL_BF16 = RING_TAIL_BYTES / ELEM_BYTES;                       // 12
constexpr uint32_t RING_HEAD_SIZE = RING_KEY_DIM + RING_TAIL_BF16;                      // 140（规范宽）
constexpr uint32_t RING_HEAD_SIZE_ALT = RING_KEY_DIM;                                   // 128（m19 实现形态，仅用于对拍）
constexpr uint32_t RING_ROW_BYTES = RING_HEAD_SIZE * ELEM_BYTES;                        // 280 B/行
constexpr uint32_t RING_ROWS_PER_BLOCK = 4;               // = 4*ceil((4+num_spec)/4)，num_spec=0 ⇒ 4
constexpr uint32_t RING_NUM_SPEC = 0;                     // 本工程不做投机采样
constexpr uint32_t RING_LAYER_BYTES = RING_ROWS_PER_BLOCK * RING_ROW_BYTES;             // 1,120 B/请求/层
static_assert(RING_ROWS_PER_BLOCK == 4u * ((4u + RING_NUM_SPEC + 3u) / 4u),
              "容量规则 4*ceil((4+num_spec)/4)（docs/14:622）");
static_assert(RING_ROW_BYTES == 280u && RING_LAYER_BYTES == 1120u, "280 B/行、1,120 B/请求/层");
static_assert(RING_ROW_BYTES % 8u == 0u, "行按 8 B 对齐（int64 位置尾的要求）");
// **对齐约束**：单行不是 32 B 整数倍 ⇒ 单行搬运必须 DataCopyPad
static_assert(RING_ROW_BYTES % 32u != 0u && RING_ROW_BYTES % 32u == 24u,
              "raw ring 单行 280 B：32 B 余 24 ⇒ 单行 DataCopy 非法，必须 DataCopyPad");
static_assert(RING_LAYER_BYTES % 32u == 0u, "整环 1,120 B = 35×32 B，是 32 B 对齐的");
constexpr uint32_t RingRowBytesAlt() { return RING_HEAD_SIZE_ALT * ELEM_BYTES; }        // 256 B

// ---- compressed key cache（`[blocks, block_size/4, 1, 128]` bf16）----
constexpr uint32_t COMP_TOKENS_PER_STATE = 4;             // indexer_compress_ratio
constexpr uint32_t COMP_HEAD_DIM = 128;                   // = RING_KEY_DIM
constexpr uint32_t COMP_ROW_BYTES = COMP_HEAD_DIM * ELEM_BYTES;                          // 256 B/行
constexpr uint32_t COMP_ROWS_PER_BLOCK = KV_BLOCK_TOKENS / COMP_TOKENS_PER_STATE;        // 4 行/页
constexpr uint32_t COMP_BLOCK_BYTES = COMP_ROWS_PER_BLOCK * COMP_ROW_BYTES;              // 1,024 B/页
constexpr uint32_t COMP_BYTES_PER_CTX_TOKEN = COMP_ROW_BYTES / COMP_TOKENS_PER_STATE;    // 64 B/token-of-context
static_assert(COMP_ROW_BYTES == 256u && COMP_BYTES_PER_CTX_TOKEN == 64u &&
              COMP_ROWS_PER_BLOCK == 4u, "docs/14:623 的 256 B/行、64 B/token-of-context");
static_assert(COMP_ROW_BYTES % 32u == 0u && COMP_BLOCK_BYTES % 32u == 0u, "compressed 可直接 DataCopy");

// ---- packed indices（NVIDIA 契约 `[m, 2052] int32`，末列 = 有效数）----
constexpr uint32_t PACK_COLS = 2052;                      // = 4*512 + (tail 2) + 2
constexpr uint32_t PACK_IDX_ELEM_BYTES = 4;               // int32
constexpr uint32_t PACK_ROW_BYTES = PACK_COLS * PACK_IDX_ELEM_BYTES;                     // 8,208 B/行
constexpr uint32_t PACK_BLOCK_COLS = 2048;                // [0, 2048) = 4·min(V,512) 个 token 位置
constexpr uint32_t PACK_BLOCK_TOPK = 512;                 // indexer_budget 2048 / 4
constexpr uint32_t PACK_TAIL_COL = 2048;                  // [2048, 2048+tail) = open group 的 token
constexpr uint32_t PACK_TAIL_MAX = 2;                     // 默认 4 档里 tail 最多 2（pos≡1 mod 4 时 1 个）
constexpr uint32_t PACK_COUNT_COL = PACK_COLS - 1u;       // 2051 = 有效数（不是块数）
constexpr uint32_t PACK_FILL = 0xFFFFFFFFu;               // 无效位填 -1
static_assert(PACK_ROW_BYTES == 8208u, "docs/15:528-531 的 8,208 B/token");
static_assert(PACK_COUNT_COL == PACK_COLS - 1u && PACK_TAIL_COL + PACK_TAIL_MAX + 1u == PACK_COUNT_COL,
              "2048 + 2 + 1 = 2051 = 末列（有效数）；col 2050 是 -1 填充");
// **对齐约束**：单行 8,208 B 不是 32 B 整数倍（余 16）⇒ 单行搬运必须 DataCopyPad
static_assert(PACK_ROW_BYTES % 32u != 0u && PACK_ROW_BYTES % 32u == 16u,
              "packed 单行 8,208 B：32 B 余 16 ⇒ 单行 DataCopy 非法，必须 DataCopyPad");
static_assert(PACK_ROW_BYTES % 16u == 0u, "行按 16 B 对齐");

// ============================================================
// 1. 容量（按 **prefill m=4097 单序列** 定尺；decode ctx 同取 4097 —— M184 对齐）
// ============================================================
constexpr uint32_t PREFILL_M = 4097;                      // 单序列 prefill 的最大 token 数
constexpr uint32_t DECODE_CTX = 4097;                     // decode 档的上下文长度（= PREFILL_M；M184）

// 块数 / 行数都按「页圆整」算（`docs/14:623`：行数 = cdiv(max_len, block_size)*block_size/4）
constexpr uint32_t BlocksFor(uint32_t m) { return (m + KV_BLOCK_TOKENS - 1u) / KV_BLOCK_TOKENS; }
constexpr uint32_t CompRowsFor(uint32_t m)
{
    return BlocksFor(m) * KV_BLOCK_TOKENS / COMP_TOKENS_PER_STATE;
}
constexpr uint32_t PREFILL_BLOCKS = BlocksFor(PREFILL_M);              // 257
constexpr uint32_t PREFILL_COMP_ROWS = CompRowsFor(PREFILL_M);          // 1,028
constexpr uint32_t DECODE_BLOCKS = BlocksFor(DECODE_CTX);              // 257（M184：原 256 ⇒"少 1 页"）
constexpr uint32_t DECODE_COMP_ROWS = CompRowsFor(DECODE_CTX);          // 1,028（M184：原 1,024）
// **M184（2026-10-04）：`DECODE_CTX` 4096 → 4097**。推导链：`PREFILL_M = 4097` 的 prefill 覆盖位置
// 0..4096（共 4097 个 token）；decode 要读全部 ⇒ token 4096 落在 `BlocksFor` 的第 `4096/16 = 256`
// 页（0-based）、slot 0 ⇒ 需 257 页。旧 `DECODE_CTX = 4096` 只给 256 页 ⇒ **少 1 页**。
// 容量定尺**不受影响、也不变小**：分配的层 stride 由 `PREFILL_BLOCKS`（主 KV）与 `PREFILL_COMP_ROWS`
// （compressed）决定（见下），本次只改这两个 decode 派生量；`DECODE_CTX == PREFILL_M` 恰为其正确口径。

// 单层 stride（= 分配 stride，**12 个 attention 层按 k 顺序静态编址**）
constexpr uint32_t KV_LAYER_STRIDE = PREFILL_BLOCKS * KV_BLOCK_BYTES;        // 8,421,376 B
constexpr uint32_t COMP_LAYER_STRIDE = PREFILL_COMP_ROWS * COMP_ROW_BYTES;   // 263,168 B
constexpr uint32_t RING_LAYER_STRIDE = RING_LAYER_BYTES;                     // 1,120 B
constexpr uint32_t KV_PLANE_BYTES = N_ATTN * KV_LAYER_STRIDE;                // 101,056,512 B
constexpr uint32_t COMP_PLANE_BYTES = N_ATTN * COMP_LAYER_STRIDE;            // 3,158,016 B
constexpr uint32_t RING_PLANE_BYTES = N_ATTN * RING_LAYER_STRIDE;            // 13,440 B
// packed indices **层间复用一块**：12 层在流内串行、host 逐层消费（M76 §2.2）
constexpr uint32_t PACK_PLANE_BYTES = PREFILL_M * PACK_ROW_BYTES;            // 33,628,176 B

static_assert(PREFILL_BLOCKS == 257u && DECODE_BLOCKS == 257u,
              "ceil(4097/16)=257、ceil(4097/16)=257（M184：decode 与 prefill 同页数，旧 256 少 1 页）");
static_assert(PREFILL_COMP_ROWS == 1028u && DECODE_COMP_ROWS == 1028u,
              "prefill 1,028 行（写 1,024 行 + open group 的 4 槽尾页 + padding）；decode 1,028 行（M184：同容量）");
static_assert(KV_LAYER_STRIDE == 8421376u, "257 × 32,768");
static_assert(COMP_LAYER_STRIDE == 263168u, "1,028 × 256");
static_assert(RING_LAYER_STRIDE == 1120u, "4 × 280");
static_assert(PACK_PLANE_BYTES == 33628176u, "4,097 × 8,208");
static_assert(KV_PLANE_BYTES == 101056512u, "12 × 8,421,376 = 96.38 MiB");
static_assert(COMP_PLANE_BYTES == 3158016u, "12 × 263,168 = 3.01 MiB");
static_assert(RING_PLANE_BYTES == 13440u, "12 × 1,120");
static_assert(KV_LAYER_STRIDE % 32u == 0u && COMP_LAYER_STRIDE % 32u == 0u &&
              RING_LAYER_STRIDE % 32u == 0u, "层 stride 全部 32 B 对齐");

// 层基址：k = attention 序列序号（0..11，对应 0-based 层号 3,7,…,47）
inline uint64_t KvLayerOffset(uint32_t k) { return static_cast<uint64_t>(k) * KV_LAYER_STRIDE; }
inline uint64_t CompLayerOffset(uint32_t k) { return static_cast<uint64_t>(k) * COMP_LAYER_STRIDE; }
inline uint64_t RingLayerOffset(uint32_t k) { return static_cast<uint64_t>(k) * RING_LAYER_STRIDE; }

// ============================================================
// 1b. 算式文本（**单一权威**；宿主与设备共用）
// ============================================================
// 为什么是宏而不是 inline 函数：ASC 工具链里 `__aicore__` 函数**不能**被 host 代码调用，而未标注
// 的函数**不能**被 device 代码调用（实测报错 "call to __host__ function from __aicore__ function"，
// 见 README §「M82」的证据行）。要把同一份算式同时给两个编译空间用，宏是唯一不用复制代码的形式。
// ⇒ **所有寻址算式只在本节出现一次**；下面的 host 包装与 probe 的 device 代码都只是它的调用点。
#define M15KV_KV_BLOCK_OF(pos) ((pos) / M15Kv::KV_BLOCK_TOKENS)
#define M15KV_KV_SLOT_OF(pos) ((pos) % M15Kv::KV_BLOCK_TOKENS)
// `kv` = lane（`M15Kv::KV_LANE_K` / `KV_LANE_V`）—— 官方 K/V 在**同一 (slot, head) 槽内相邻**
#define M15KV_KV_IN_BLOCK_OFF(slot, head, kv, dim)                              \
    (static_cast<uint64_t>(head) * M15Kv::KV_HEAD_PLANE_BYTES +                 \
     static_cast<uint64_t>(slot) * M15Kv::KV_TOKEN_STRIDE +                     \
     static_cast<uint64_t>(kv) * M15Kv::KV_HEAD_CONTENT_BYTES +                 \
     static_cast<uint64_t>(dim) * M15Kv::KV_ELEM_BYTES)
#define M15KV_KV_BYTE_OFF_PHYS(physBlk, slot, head, kv, dim)                    \
    (static_cast<uint64_t>(physBlk) * M15Kv::KV_BLOCK_BYTES +                   \
     M15KV_KV_IN_BLOCK_OFF(slot, head, kv, dim))
#define M15KV_KV_BYTE_OFF_CONTIG(pos, head, kv, dim)                            \
    (static_cast<uint64_t>(M15KV_KV_BLOCK_OF(pos)) * M15Kv::KV_BLOCK_BYTES +    \
     M15KV_KV_IN_BLOCK_OFF(M15KV_KV_SLOT_OF(pos), head, kv, dim))
#define M15KV_TBL_IDX(req, tblStride, blk) (static_cast<uint64_t>(req) * (tblStride) + (blk))
#define M15KV_RING_SLOT(pos) ((pos) % M15Kv::RING_ROWS_PER_BLOCK)
#define M15KV_RING_ROW_OFF(physBlk, pos)                                        \
    (static_cast<uint64_t>(physBlk) * M15Kv::RING_LAYER_BYTES +                 \
     static_cast<uint64_t>(M15KV_RING_SLOT(pos)) * M15Kv::RING_ROW_BYTES)
// 逻辑组号 g = position / 4；组 g 的组首位置 = 4g
#define M15KV_COMP_GROUP_OF(pos) ((pos) / M15Kv::COMP_TOKENS_PER_STATE)
// **命根子判据（off-by-one）**：只有组边界才产生压缩行。
// `(pos+1) % 4 == 0` ⟺ pos ≡ 3 (mod 4) ⟹ pos ≥ 3 ⇒ 文档里的「且 p ≥ 3」是**被蕴含的条件**，
// 不是额外限制（uint32 无下溢路径）。pos 4096 ⇒ (4096+1)%4 = 1 ⇒ 不写（"开放组"）。
#define M15KV_COMP_ROW_WRITTEN(pos) ((((pos) + 1u) % M15Kv::COMP_TOKENS_PER_STATE) == 0u)
#define M15KV_COMP_ROW_OFF_PHYS(physBlk, g)                                     \
    (static_cast<uint64_t>(physBlk) * M15Kv::COMP_BLOCK_BYTES +                 \
     static_cast<uint64_t>((g) % M15Kv::COMP_ROWS_PER_BLOCK) * M15Kv::COMP_ROW_BYTES)
#define M15KV_COMP_ROW_OFF_CONTIG(g) (static_cast<uint64_t>(g) * M15Kv::COMP_ROW_BYTES)
#define M15KV_PACK_ROW_OFF(row) (static_cast<uint64_t>(row) * M15Kv::PACK_ROW_BYTES)

// ============================================================
// 2. paged 契约（D2）：**所有地址都经这里算**，连续布局 = 恒等 block_table 的特例
// ============================================================
// 下面这些是 **host 侧**包装（一行一调用 §1b 的宏，**不含第二份算式**）；device 侧算同一个地址
// 时直接调 §1b 的宏（见 `m15_attn_kv_probe.h`）。
inline uint32_t KvBlockOf(uint32_t pos) { return M15KV_KV_BLOCK_OF(pos); }
inline uint32_t KvSlotInBlock(uint32_t pos) { return M15KV_KV_SLOT_OF(pos); }
inline uint64_t KvInBlockOffset(uint32_t slot, uint32_t head, uint32_t kv, uint32_t dim)
{
    return M15KV_KV_IN_BLOCK_OFF(slot, head, kv, dim);
}
// 连续布局（**block_table 恒等**时的等价式；仍走同一个页内式子）
inline uint64_t KvByteOffsetContig(uint32_t pos, uint32_t head, uint32_t kv, uint32_t dim)
{
    return M15KV_KV_BYTE_OFF_CONTIG(pos, head, kv, dim);
}
// 物理页号已定时的 paged 地址（device 侧也用这一式：页号由 `GlobalTensor::GetValue` 取到）
inline uint64_t KvByteOffsetPhys(uint32_t physBlk, uint32_t slot, uint32_t head, uint32_t kv, uint32_t dim)
{
    return M15KV_KV_BYTE_OFF_PHYS(physBlk, slot, head, kv, dim);
}
// paged 布局：块号经 `block_table[req][logical_block]` 查表。
// **device 侧**没有 `int32_t*` 这种数据（GM 句柄是 `GlobalTensor<int32_t>`）⇒ 表项读取由调用方
// 完成（device: `GetValue(M15KV_TBL_IDX(...))`；host: 直接下标），下标算式与字节偏移仍出自本头文件。
inline uint64_t KvTableIndex(uint32_t req, uint32_t tblStride, uint32_t logicalBlock)
{
    return M15KV_TBL_IDX(req, tblStride, logicalBlock);
}

inline uint32_t KvPhysBlock(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t logicalBlock)
{
    if (table == nullptr) {
        return logicalBlock;   // debug 用；正式路径必须传真表（probe 的负向对照会红）
    }
    return static_cast<uint32_t>(table[KvTableIndex(req, tblStride, logicalBlock)]);
}

inline uint64_t KvByteOffsetPaged(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t pos,
                                  uint32_t head, uint32_t kv, uint32_t dim)
{
    const uint32_t phys = KvPhysBlock(table, req, tblStride, M15KV_KV_BLOCK_OF(pos));
    return M15KV_KV_BYTE_OFF_PHYS(phys, M15KV_KV_SLOT_OF(pos), head, kv, dim);
}

// 判据 T-KV-PAGED 的可执行定义（host 侧；device 侧的等价见证见 probe：paged 写 vs 连续读）
// 「对同一 token 位置，连续地址与经 block_table 映射后的 paged 地址读到的向量逐字节相同」
inline bool KvPagedContractHolds(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t pos,
                                 uint32_t head, uint32_t kv, uint32_t dims)
{
    for (uint32_t d = 0; d < dims; ++d) {
        if (KvByteOffsetContig(pos, head, kv, d) != KvByteOffsetPaged(table, req, tblStride, pos, head, kv, d)) {
            return false;
        }
    }
    return true;
}

// ---- **旧几何**（M82 原值；塔裁 A 后只作为**负向对照**的比对面，任何分配/编址都不使用）----
// 旧 = 页 16,384 B / token 步长 512 B / head 平面 8,192 B / 轴序 head 外层 / 无 V 通道。
// `KvOldGeomByteOffset()` 把它写成可执行式，供 `Kv.neg.oldgeom` 判"新判据能咬住这个方向"。
constexpr uint32_t KV_OLD_BLOCK_BYTES = 16384u;
constexpr uint32_t KV_OLD_TOKEN_STRIDE = 512u;
constexpr uint32_t KV_OLD_HEAD_PLANE_BYTES = 8192u;
static_assert(KV_OLD_BLOCK_BYTES != KV_BLOCK_BYTES, "旧几何已与权威不同（这正是它作负向对照的用途）");
inline uint64_t KvOldGeomByteOffset(uint32_t pos, uint32_t head, uint32_t kv, uint32_t dim)
{
    return static_cast<uint64_t>(pos / KV_BLOCK_TOKENS) * KV_OLD_BLOCK_BYTES +
           static_cast<uint64_t>(head) * KV_OLD_HEAD_PLANE_BYTES +
           static_cast<uint64_t>(pos % KV_BLOCK_TOKENS) * KV_OLD_TOKEN_STRIDE +
           static_cast<uint64_t>(dim) * KV_ELEM_BYTES + static_cast<uint64_t>(kv) * KV_HEAD_CONTENT_BYTES;
}

// 单请求恒等表：`block_table[r][b] == b`（docs/15:528-531 的默认；`--check` 用）
inline bool KvBlockTableIdentity(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t nBlocks)
{
    for (uint32_t b = 0; b < nBlocks; ++b) {
        if (static_cast<uint32_t>(table[M15KV_TBL_IDX(req, tblStride, b)]) != b) {
            return false;
        }
    }
    return true;
}

// ============================================================
// 3. raw key ring / compressed 的写入门控与寻址（off-by-one 契约的**可执行定义**）
// ============================================================
// 槽位 = physical_block*RING_ROWS_PER_BLOCK + (position % RING_ROWS_PER_BLOCK)；单请求下物理块恒 0
// ⇒ 槽位 = pos % 4（`m19 README:101-102`）。**它不是全历史 cache**，只服务"凑齐 4 个组成一行"。
inline uint32_t RingSlot(uint32_t pos) { return M15KV_RING_SLOT(pos); }
inline uint64_t RingRowOffset(uint32_t physBlock, uint32_t pos)
{
    return M15KV_RING_ROW_OFF(physBlock, pos);
}

inline uint32_t CompGroupOf(uint32_t pos) { return M15KV_COMP_GROUP_OF(pos); }
inline uint32_t CompGroupFirstPos(uint32_t g) { return g * COMP_TOKENS_PER_STATE; }
inline bool CompRowWritten(uint32_t pos) { return M15KV_COMP_ROW_WRITTEN(pos); }

// 开放组（正在累积、**尚未**产生压缩行）的组首 = 4·floor(pos/4)。pos ≡ 3 (mod 4) 时它与压缩行的
// 组首 `pos-3` 相同 —— 这正是「pos 4096 属开放组」的代数形式（4096 mod 4 = 0 ⇒ 组首 4096 本身）。
inline uint32_t OpenGroupFirstPos(uint32_t pos) { return pos - (pos % COMP_TOKENS_PER_STATE); }

// 压缩行地址：槽 = physical_block*COMP_ROWS_PER_BLOCK + (g % COMP_ROWS_PER_BLOCK)
inline uint32_t CompPhysBlock(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t g)
{
    return KvPhysBlock(table, req, tblStride, g / COMP_ROWS_PER_BLOCK);
}
// **物理页号已定**时的压缩行偏移（device 侧用这个：页号由 `GlobalTensor::GetValue` 取到）
inline uint64_t CompByteOffsetPhys(uint32_t physBlock, uint32_t g)
{
    return M15KV_COMP_ROW_OFF_PHYS(physBlock, g);
}
inline uint64_t CompRowOffset(const int32_t* table, uint32_t req, uint32_t tblStride, uint32_t g)
{
    return CompByteOffsetPhys(CompPhysBlock(table, req, tblStride, g), g);
}
// 连续等价式（恒等表）：compressed 行号 = g，行距 256 B
inline uint64_t CompRowOffsetContig(uint32_t g) { return M15KV_COMP_ROW_OFF_CONTIG(g); }

// 写入门控 + 落点（一步到位；m19 的 `closes`/`groupIdx` 就是它的两个分量）
inline bool CompFill(uint32_t pos, uint32_t& groupOut, uint64_t& byteOffsetContigOut)
{
    if (!CompRowWritten(pos)) {
        return false;
    }
    groupOut = CompGroupOf(pos);
    byteOffsetContigOut = CompRowOffsetContig(groupOut);
    return true;
}

// 打分上界：可见压缩列数（**不含**正在累积的 open group）（`m19 README:121-123`、`V-C:qsa_cache.py:268-275`）
inline uint32_t VisibleBlocks(uint32_t p, uint32_t seq)
{
    const uint32_t byPos = (p + 1u) / COMP_TOKENS_PER_STATE;
    const uint32_t bySeq = seq / COMP_TOKENS_PER_STATE;
    const uint32_t v = (byPos < bySeq) ? byPos : bySeq;
    return v;   // 两个分量都 ≥0 ⇒ max(0, min(...)) 的 max 分支恒取 min
}

// ============================================================
// 4. packed indices 的列语义
// ============================================================
inline uint64_t PackRowOffset(uint32_t row) { return M15KV_PACK_ROW_OFF(row); }
// 块 b 的第 s 个子槽（s ∈ [0,4)）在 packed 行里的列号与它代表的**请求内绝对 token 位置**
inline uint32_t PackColOfSub(uint32_t b, uint32_t s) { return b * COMP_TOKENS_PER_STATE + s; }
inline uint32_t PackTokenOfSub(uint32_t b, uint32_t s) { return b * COMP_TOKENS_PER_STATE + s; }
inline uint32_t PackTailColOf(uint32_t t) { return PACK_TAIL_COL + t; }   // t ∈ [0, PACK_TAIL_MAX)

// ============================================================
// 5. 容量自检要打印/复算的账（一处定义，host 与 README 共用）
// ============================================================
// 4097 上下文下 **实际写入**的压缩行数 = |{pos ∈ [0,4096] : pos ≡ 3 (mod 4)}| = |{3,7,…,4095}|
constexpr uint32_t PREFILL_COMP_ROWS_WRITTEN = (PREFILL_M - 1u) / COMP_TOKENS_PER_STATE;   // 1,024
static_assert(PREFILL_COMP_ROWS_WRITTEN == 1024u, "3,7,…,4095 共 1,024 行（docs/17:286）");
static_assert(PREFILL_COMP_ROWS_WRITTEN < PREFILL_COMP_ROWS,
              "写入行数 1,024 < 容量行数 1,028：差额 = open group 的 4 槽尾页 + padding");

}  // namespace M15Kv

// ============================================================
// 6. M98：cache 填数学（本 mission 新建，落 `m15_attn_cache.h`）
//
// **为什么由本权威头在末尾 include**：`.asc` 的 include 清单由 M97 独占（本 mission 不得改），
// 而设备代码必须出现在文件作用域（`namespace M15Run` 之外）。本头文件在 `.asc:63` 被引入，
// 位置在 `namespace M15Run {`（`:76`）之前、且在 `m15_gdn_layer.h`/`m15_hc_layer.h` 之后
// ⇒ 这里 include 即可让设备代码落在正确的作用域与先后次序上，且 `.asc` 一字不动。
// 该文件自带 `m15_attn_prolog.h` / `m15_attn_prolog_probe.h` 的 include，故不依赖 `.asc` 里
// 第 65/66 行的位置（唯一注意：本 TU 里 m15_attn_kv.h 必须先于 m15_attn_prolog.h 被首次引入，
// 现状如此 —— 见 `.asc:63` 与 `:65`）。
// ============================================================
#include "m15_attn_cache.h"

#endif  // M15_ATTN_KV_H
