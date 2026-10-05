// ============================================================
// m15_loop_layout.h —— 48 层循环的 GM 平面图（host 侧编译期常量）
//
// 层类型模式来源 = 真实 checkpoint 的 config.json（用户模型）：
//   /workspace/Qwen3.8-Flash-Next-MXFP4/config.json
//     text_config.num_hidden_layers   = 48
//     text_config.full_attention_interval = 4
//     text_config.layer_types = [linear_attention ×3, full_attention] × 12
//   即层号 L 满足 (L+1) % 4 == 0 → full_attention（3/7/11/…/47，共 12 层），其余 36 层为
//   linear_attention（GDN）。host 启动时会用 weights manifest 里的 layer_kinds 逐层核对
//   （见 slice_layer_manifest.py：它直接读 config.json 写出该字段）。
//
// GM 平面（层循环独占，host 一次分配 / 跨所有层与 token 复用）：
//
//   [权重区]  GDN 层各占一个 W_STRIDE 槽（行主序原始 layout，不做离线转换）
//             ┌ in_proj [IN_N=16480, HIDDEN=2560] bf16   84.4MB   ← ckpt: qkv|z|b|a 行拼接
//             │ out_proj [OUT_N=2560, OUT_K=6144] bf16   31.5MB
//             │ conv1d [KW=4, CH=10240] bf16            80KB     ← ckpt [CH,1,4] 转置
//             │ conv_bias [CH] bf16                      20KB     ← ckpt 无此张量 → 全零
//             │ A_log / dt_bias [64] fp32                1KB      ← ckpt bf16[48] → fp32 + 尾零
//             │ gamma1 / gamma2 [HIDDEN] bf16            10KB     ← ckpt 无层级 norm → 合成（见 README）
//             └ gammaG [HEAD_D] bf16                    256B     ← ckpt linear_attn.norm.weight
//
//   [状态区]  GDN 层各占一个 ST_STRIDE 槽（**跨 token 常驻**，kernel 内 in-place RMW）
//             ┌ conv_state [ST=3, CH=10240] bf16         60KB
//             └ ssm_state  [HEADS=48, 128, 128] fp32     3.0MB
//
//   [激活区]  双缓冲残差流 h[2]（各 M_MAX 行，**只用行 0** = 本 token），零残差缓冲 resZero，
//             层 workspace ws（每层一次启动复用同一块，流内串行 → 不重叠）。
//
//   [attention 长寿 cache 区]（M82 新建；attention 层按"第 k 个 attention 层"编址，k=0..11）
//             主 KV paged `[257 页/层, 2, 16, 512]` bf16  = 8,421,376 B/层（= 257 页 × 32,768 B；×12 = 101,056,512 B = 96.38 MiB）
//             compressed key `[257 页/层, 4, 1, 128]` bf16 =   263,168 B/层（×12 =  3.01 MiB）
//             raw key ring  `[cap=4, 140]` bf16           =     1,120 B/层（×12 = 13,440 B）
//             packed indices `[4097, 2052]` int32         = 33,628,176 B（**12 层复用一块**）
//             物理字节布局与寻址函数全部在 `m15_attn_kv.h`（唯一权威）。
//
// 权重/状态槽按层号**静态**编址：GDN 层第 k 个（k = 该层在 GDN 层序列里的序号）占第 k 槽，
// 这样 attention 层不占槽（省 12×116MB ≈ 1.4GB），槽号↔层号的映射由 LayerSlot() 给出。
// ============================================================
#ifndef M15_LOOP_LAYOUT_H
#define M15_LOOP_LAYOUT_H

namespace M15Loop {

using namespace M15G;

// ---- 层类型 ----
constexpr uint32_t NL = 48;                // config: num_hidden_layers
constexpr uint32_t ATTN_INTERVAL = 4;      // config: full_attention_interval
constexpr uint32_t KIND_GDN = 0;
constexpr uint32_t KIND_ATTN = 1;

static_assert(NL % ATTN_INTERVAL == 0, "48 必须被 full_attention_interval 整除");

inline uint32_t KindOf(uint32_t layer)
{
    return ((layer + 1) % ATTN_INTERVAL == 0) ? KIND_ATTN : KIND_GDN;
}

// GDN 层在 GDN 序列里的序号（= 权重/状态槽号）；attention 层返回 NO_SLOT
constexpr uint32_t NO_SLOT = 0xFFFFFFFFu;
inline uint32_t LayerSlot(uint32_t layer)
{
    if (KindOf(layer) != KIND_GDN) {
        return NO_SLOT;
    }
    return layer - layer / ATTN_INTERVAL;   // 每个 attention 层之前有 layer/ATTN_INTERVAL 个 attention 层
}

// **attention 层**在 attention 序列里的序号 k（= 0..N_ATTN-1，对应 0-based 层号 3,7,…,47）；
// GDN 层返回 NO_SLOT。它是 M82 新建的三套 cache（主 KV / raw key ring / compressed key cache）
// 的**槽号**：第 k 个 attention 层占第 k 槽，与 GDN 的 `LayerSlot` 同族（静态编址、host 一次分配）。
//   为什么单独一个函数而不是复用 LayerSlot：两条序列的长度不同（36 vs 12），混用会静默错位；
//   层号 → 序号的两套映射必须各自显式（M65 的 `Ch.fp.L*` 指纹判据就是防这类错位的）。
inline uint32_t AttnSlot(uint32_t layer)
{
    if (KindOf(layer) != KIND_ATTN) {
        return NO_SLOT;
    }
    return layer / ATTN_INTERVAL;
}

constexpr uint32_t N_GDN = NL - NL / ATTN_INTERVAL;    // 36
constexpr uint32_t N_ATTN = NL / ATTN_INTERVAL;        // 12

// ---- attention 侧长寿 cache 的**平面尺寸**（M82）----
// 定义在 `m15_attn_kv.h`（三套 cache + packed indices 的唯一权威头文件）：
//   KV_LAYER_STRIDE / COMP_LAYER_STRIDE / RING_LAYER_STRIDE（每层槽）
//   KV_PLANE_BYTES / COMP_PLANE_BYTES / RING_PLANE_BYTES（12 层合计）
//   PACK_PLANE_BYTES（packed indices，**层间复用一块**）
// 本文件只提供槽号映射（AttnSlot），不再重复尺寸常量 —— 单一权威（改这里不会漂）。

// ---- 激活 ----
constexpr uint32_t H_BYTES = HIDDEN * 2;                // 残差流一个 token（bf16[1,2560]）5120B
constexpr uint32_t H_ROWS_BYTES = M_MAX * HIDDEN * 2;   // 双缓冲残差流按 M_MAX 行分配（只用行 0）

// ============================================================
// M110（Wave A）：激活平面按 **prefill 的 m** 定尺
// ============================================================
// 为什么单列一节：上面那一族的 `M_MAX = 64` 包络是 **decode 档**（每层只用行 0）。prefill 的
// 验收档是 m = 4097，而 4097 行的平面比 64 行大 **64.02×**（= 4097/64 = 64.0156，按字节算
// 20,976,640 / 327,680 = 64.0156，同一个数）—— `docs/15` M103-2.1 的 Wave A 表最后一行点名的
// 就是这一族。下面每一项都带**手算式**并在 `static_assert` 里钉死数值：M103 复审 r1 的 P2-1
// 就是这一族的手算错（初稿把 20,976,640 写成 20,979,200，多算 2,560 B = 半行）。
constexpr uint32_t M_PREFILL = 4097;                    // = M15Kv::PREFILL_M（单序列 prefill 上界）

// 残差流（bf16[HIDDEN] / 行）：4097 × 2560 × 2 = 20,976,640 B（对比 decode 档 327,680 B）
constexpr uint32_t H_ROWS_BYTES_PF = M_PREFILL * HIDDEN * 2;
// 同族的三个平面（列宽取自 `docs/15` M103-3，同书的手算式逐条列出）：
//   q|gate   12288 列 ⇒ 4097 × 12288 × 2 = 100,687,872 B（注意：**不是** 文档 §1 的 12288 之外还有别的宽度）
//   qkvzba   16480 列 ⇒ 4097 × 16480 × 2 = 135,037,120 B（M15G::IN_N）
//   opin      6144 列 ⇒ 4097 ×  6144 × 2 =  50,343,936 B（M15G::V_DIM）
// 这三条是 attention 前端 / GDN 中间量 / out_proj 输入的按行平面，故放在同一节（同族手算）。
constexpr uint32_t PF_COLS_QGATE = 12288;               // attn_output_gate=True：q|gate 各 6144
constexpr uint32_t PF_QGATE_BYTES = M_PREFILL * PF_COLS_QGATE * 2;
constexpr uint32_t PF_QKVZBA_BYTES = M_PREFILL * IN_N * 2;
constexpr uint32_t PF_OPIN_BYTES = M_PREFILL * V_DIM * 2;

static_assert(H_ROWS_BYTES_PF == 20976640u, "prefill 残差流平面字节数变了（4097×2560×2）");
static_assert(M_PREFILL * HIDDEN * 2 > H_ROWS_BYTES * 64u,
              "prefill 平面必须比 decode 包络（M_MAX 行）大 64 倍以上——否则定尺没生效");
static_assert(PF_QGATE_BYTES == 100687872u, "q|gate 平面字节数变了（4097×12288×2）");
static_assert(PF_QKVZBA_BYTES == 135037120u, "qkvzba 平面字节数变了（4097×16480×2）");
static_assert(PF_OPIN_BYTES == 50343936u, "opin 平面字节数变了（4097×6144×2）");
// 三条平面都必须容纳 4097 行（行宽 = 上面那些列宽 × 2B）；这是「不得留缩形档」的可执行版。
static_assert(PF_QGATE_BYTES / (PF_COLS_QGATE * 2u) == M_PREFILL, "q|gate 平面行数不是 4097");
static_assert(PF_QKVZBA_BYTES / (IN_N * 2u) == M_PREFILL, "qkvzba 平面行数不是 4097");
static_assert(PF_OPIN_BYTES / (V_DIM * 2u) == M_PREFILL, "opin 平面行数不是 4097");

// ---- 权重槽（512B 对齐）----
constexpr uint32_t AL = 512;
constexpr uint32_t AlignUp(uint32_t x) { return (x + AL - 1u) / AL * AL; }

constexpr uint32_t W_IN_OFF = 0;
constexpr uint32_t W_IN_BYTES = IN_N * HIDDEN * 2;            // 84,377,600
constexpr uint32_t W_OUT_OFF = W_IN_OFF + W_IN_BYTES;
constexpr uint32_t W_OUT_BYTES = OUT_N * OUT_K * 2;           // 31,457,280
constexpr uint32_t W_CONV_OFF = AlignUp(W_OUT_OFF + W_OUT_BYTES);
constexpr uint32_t W_CONV_BYTES = KW * CH * 2;                // 81,920
constexpr uint32_t W_CONVB_OFF = W_CONV_OFF + W_CONV_BYTES;
constexpr uint32_t W_CONVB_BYTES = CH * 2;                    // 20,480
constexpr uint32_t W_ALOG_OFF = AlignUp(W_CONVB_OFF + W_CONVB_BYTES);
constexpr uint32_t W_ALOG_BYTES = 64 * 4;                     // 256（48 有效 + 16 尾零）
constexpr uint32_t W_DTB_OFF = W_ALOG_OFF + W_ALOG_BYTES;
constexpr uint32_t W_DTB_BYTES = 64 * 4;
constexpr uint32_t W_G1_OFF = AlignUp(W_DTB_OFF + W_DTB_BYTES);
constexpr uint32_t W_G1_BYTES = HIDDEN * 2;                   // 5,120
constexpr uint32_t W_G2_OFF = W_G1_OFF + W_G1_BYTES;
constexpr uint32_t W_G2_BYTES = HIDDEN * 2;
constexpr uint32_t W_GG_OFF = AlignUp(W_G2_OFF + W_G2_BYTES);
constexpr uint32_t W_GG_BYTES = HEAD_D * 2;                   // 256
constexpr uint32_t W_STRIDE = AlignUp(W_GG_OFF + W_GG_BYTES);

// ---- 状态槽 ----
constexpr uint32_t CS_BYTES = ST * CH * 2;                    // 61,440
constexpr uint32_t SSM_BYTES = HEADS * HEAD_D * HEAD_D * 4;   // 3,145,728
constexpr uint32_t ST_STRIDE = AlignUp(CS_BYTES + SSM_BYTES);
constexpr uint32_t SSM_OFF = CS_BYTES;                        // 状态槽内 ssm_state 偏移

// ---- 真机规模（文档/预算用）----
constexpr double W_TOTAL_MB = static_cast<double>(N_GDN) * static_cast<double>(W_STRIDE) / 1e6;
constexpr double ST_TOTAL_MB = static_cast<double>(N_GDN) * static_cast<double>(ST_STRIDE) / 1e6;

}  // namespace M15Loop

#endif  // M15_LOOP_LAYOUT_H
