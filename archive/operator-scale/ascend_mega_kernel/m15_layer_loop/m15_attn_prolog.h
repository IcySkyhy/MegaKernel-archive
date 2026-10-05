// ============================================================
// m15_attn_prolog.h —— M88：attention 前端 prolog（主干 q/k 路 + indexer 路）的**设备实现**
//
// 本头文件是 M82 显式 descope 的「attention 前端 prolog」（`m15_layer_loop/README.md` Part D · M82-6
// 第 1 项）的落点。它的**规则来源**逐条钉在 vLLM 官方源码上（N2 外部权威，docs/17 §1.2），
// 与 host 侧参考 `evidence/attn_prolog/oracle_attn_prolog.py` 的 R 表逐行对应：
//
//   R1  qkv 输出宽度 = nH*(1+gate)*hd + 2*nKV*hd = 12288 + 512 + 512
//       `vllm/model_executor/models/qwen3_next.py:427-434`
//   R2  split 顺序 [q|gate], k, v；q 与 gate **按头交织**（头 h 占 **512** 列 = q[256] ‖ gate[256]（`QG_W=12288`、`NH=24` ⇒ 12288/24 = 512））
//       `qwen3_next.py:428-434`、`/workspace/vllm/vllm/models/qwen4_exp/nvidia/qsa.py:505-506`
//   R3  q/k norm = **GemmaRMSNorm**：`y = x * rsqrt(mean(x^2)+eps) * (1+w)`
//       `qwen4_exp/nvidia/qsa.py:350-351`；语义 `vllm/model_executor/layers/layernorm.py:140-168`
//       （`weight = self.weight + 1.0`）；别名 `qwen3_next.py:31`
//       ⚠ **不是**朴素 RMSNorm（乘 `w`）。实测 layer 3 的 `q_norm.weight` 全 256 个非零、mean 0.2833
//       ⇒ 乘 `w` 的有效 scale ≈ 0.28、乘 `(1+w)` ≈ 1.28，**差 4.5×**。负向对照 AP_MODE_PLAIN_NORM 打的就是这条。
//   R4  Qwen4Exp 的 position id **无条件**是三个恒等轴（`qwen4_exp/nvidia/model.py:846-852`）
//       —— 注意措辞：**不是**「模型是文本模型」（config.json 有 vision_config）；
//       本条只说明**文本路径**上 MRoPE 不产生差异
//   R5  故 MRoPE 的三轴置换在数值上恒等（`rotary_embedding/mrope.py:236-247`：`x[0]==x[1]==x[2]` 时返回 `x[0]`）
//       ⇒ 退化为**普通 NeoX partial RoPE**
//   R6  类别 = `MRotaryEmbedding`（不是 `MRotaryEmbeddingInterleaved`；后者只在 `openpangu` 分支）
//       `rotary_embedding/__init__.py:110-121`、`:333`
//   R7  rotary_dim = int(256 × partial_rotary_factor 0.25) = **64**（`rotary_embedding/__init__.py:68-71`）
//   R8  cos/sin 表公式 `base^(2j/rotary_dim)`；表按 query dtype 落 **bf16**（`base.py:80-99,105-125`）
//   R9  NeoX 配对 `(j, j+32)`：`o1 = x1·cos − x2·sin`、`o2 = x2·cos + x1·sin`；`[64,256)` 直通
//       `rotary_embedding/common.py:134-173`
//   R10 indexer `index_qk_proj` N = (4+1)×128 = 640（`qwen4_exp/nvidia/indexer_qsa.py:131-137`）
//   R11 indexer q norm = `GemmaRMSNorm(128)`（`indexer_qsa.py:138-145`；fused 实证
//       `nvidia/ops/qsa_pre_indexer.py:69` 的 `weight = load(...) + 1.0`）
//   R12 indexer 复用**同一张** cos/sin 表，只旋 128 维的**前 64 维**
//       （`qsa_pre_indexer.py:180` 的 `cos_sin_stride = D//2` + `:29-30` 的 `HALF/QUARTER` + `:72-79` 的配对）
//   R13 indexer raw k **不 norm 不 rope**，原样进 raw ring（`qsa_pre_indexer.py:367-372`）
//
// ---------------------------------------------------------------------------
// 【cos/sin 表由 host 预生成并 H2D（**设备不算超越函数**）】
//   理由：vLLM 本身就是**预生成一张 cache** 再按 dtype 落地（R8）。设备侧算 `cos/sin` 会引入
//   `__cosf/__sinf` 一类近似，那是**参考里没有的**一项误差；预生成后判据 T1 还能对**表字节**生效。
//   表布局 = `[nPos][ROT]` bf16，行 = `[cos(32) | sin(32)]`（与 vLLM 的 `cat((cos, sin), -1)` 同序）。
//   ⚠ `vllm` 的 `MRotaryEmbedding` 会把 `max_position_embeddings` 放大 4 倍再建表
//   （`mrope.py:346-357` 的 `cache_max_position_num`）；本工程只用到 `pos < PREFILL_M = 4097`，
//   故表长取 4097 行即可 —— **这是本工程的口径，不是 vLLM 的分配口径**（vLLM 表更长，值相同）。
//
// 【本文件不做的事（边界）】
//   · 不做 indexer 的打分 / topk / expand（|packed| 的选择语义）；不做 attention 核心；不做 `o_proj`。
//   · 不做 raw ring / compressed 的**填充**（M82-6 第 2 项）；本文件只产出 raw k 与 indexer q 两个向量。
//   · **不碰** `m15_layer_kernel.h` 的 `LayerArgs`：本 mission 的接线范围见 README 的「未完成项」。
// ============================================================
#ifndef M15_ATTN_PROLOG_H
#define M15_ATTN_PROLOG_H

#include "m15_attn_kv.h"

namespace M15AP {



// ============================================================
// 1. 形状（与 oracle_attn_prolog.py 的常量逐项一致；改一处必须改两处）
// ============================================================
constexpr uint32_t AP_HIDDEN = 2560;                        // config.text_config.hidden_size
constexpr uint32_t NH = 24;                              // num_attention_heads
constexpr uint32_t HD = 256;                             // head_dim
constexpr uint32_t NKV = 2;                              // num_key_value_heads
constexpr uint32_t IDX_NH = 4;                           // indexer_n_heads
constexpr uint32_t IDX_D = 128;                          // indexer_head_dim
constexpr uint32_t ROT = 64;                             // int(HD * partial_rotary_factor 0.25)
constexpr uint32_t HALF = ROT / 2;                       // 32：NeoX 配对 (j, j+32)
constexpr uint32_t IDX_Q = IDX_NH * IDX_D;               // 512
constexpr uint32_t QW = NH * HD;                         // 6144
constexpr uint32_t QG_W = QW * 2;                        // 12288 = q ‖ gate（按头交织）
constexpr uint32_t IDX_W = IDX_Q + IDX_D;                // 640

static_assert(ROT == HD / 4u && ROT == IDX_D / 2u, "R7/R12：主干 256→64、indexer 128→64，同一张表");
static_assert(HALF == 32u, "mrope_section [11,11,10] 的和 = 32 = ROT/2");
static_assert(QG_W + NKV * HD + NKV * HD == 13312u, "q|gate|k|v 合计");
static_assert(IDX_W == 640u, "R10");

// ============================================================
// 2. 段布局（bf16 元素下标；probe 的 y0 / out 两块读回平面共用这套偏移）
// ============================================================
// y0 = GEMM 的原始输出（**未经任何 prolog 变换**）
constexpr uint32_t Y0_QG = 0;                            // [0, 12288)   q‖gate（AIC 的 q_proj 输出）
constexpr uint32_t Y0_K = Y0_QG + QG_W;                  // [12288, 12800)  k_proj
constexpr uint32_t Y0_V = Y0_K + NKV * HD;               // [12800, 13312)  v_proj
constexpr uint32_t Y0_IDX = Y0_V + NKV * HD;             // [13312, 13952)  index_qk_proj（前 512 = idx q，后 128 = raw k）
constexpr uint32_t Y0_N = Y0_IDX + IDX_W;                // 13952

// out = prolog 的输出
constexpr uint32_t OUT_Q = 0;                            // [0, 6144)      q（norm+rope 后）
constexpr uint32_t OUT_GATE = OUT_Q + QW;                // [6144, 12288)  gate（**原样抄写**，不 norm 不 rope）
constexpr uint32_t AP_OUT_K = OUT_GATE + QW;                // [12288, 12800) k（norm+rope 后）
constexpr uint32_t OUT_V = AP_OUT_K + NKV * HD;             // [12800, 13312) v（**原样抄写**）
constexpr uint32_t OUT_QIDX = OUT_V + NKV * HD;          // [13312, 13824) idx q（GemmaRMSNorm(128)+RoPE(64)）
constexpr uint32_t OUT_KRAW = OUT_QIDX + IDX_Q;          // [13824, 13952) raw k（**原样抄写**）
constexpr uint32_t AP_OUT_N = OUT_KRAW + IDX_D;             // 13952

static_assert(Y0_N == 13952u && AP_OUT_N == 13952u, "与 oracle 的 Y0_N/AP_OUT_N 一致");
// **防呆（M88 r2 复审 P1 的直接产物）**：判据按下标取段，**写错一个常量就会「瞄错区段而假绿」**。
// 这几条把「判据用到的段偏移」钉死；配套的裸符号审计 = `evidence/attn_prolog/audit_bare_names.py`。
static_assert(OUT_Q == 0u && OUT_GATE == 6144u && AP_OUT_K == 12288u && OUT_V == 12800u &&
              OUT_QIDX == 13312u && OUT_KRAW == 13824u && AP_OUT_N == 13952u,
              "out 段的绝对下标（判据按这些下标取段；改布局必须同步改判据）");
static_assert(Y0_QG == 0u && Y0_K == 12288u && Y0_V == 12800u && Y0_IDX == 13312u && Y0_N == 13952u,
              "y0 段的绝对下标");
static_assert(AP_OUT_K != M15G::OUT_K,
              "**这就是 r2 假绿的形态**：AP_OUT_K(12288) 与 M15G::OUT_K(6144) 必须不同；"
              "若哪天相等，裸写 OUT_K 的判据就会静默比错区段");


// ============================================================
// 3. 权重平面（layer 3 的 8 个 attention role；host 逐 role pread 后 H2D）
//    每个 role 段 512 B 对齐（`m15_loop_layout.h:99-120` 的 AlignUp(...,512) 同款）
// ============================================================
constexpr uint32_t W_ALIGN = 512;
constexpr uint32_t WAlignUp(uint32_t x) { return (x + W_ALIGN - 1u) / W_ALIGN * W_ALIGN; }

constexpr uint32_t W_Q_OFF = 0;                                       // attn_q_proj      [12288, 2560]
constexpr uint32_t W_Q_BYTES = QG_W * AP_HIDDEN * 2u;                    // 62,914,560
constexpr uint32_t W_K_OFF = WAlignUp(W_Q_OFF + W_Q_BYTES);           // attn_k_proj      [512, 2560]
constexpr uint32_t W_K_BYTES = NKV * HD * AP_HIDDEN * 2u;
constexpr uint32_t W_V_OFF = WAlignUp(W_K_OFF + W_K_BYTES);           // attn_v_proj      [512, 2560]
constexpr uint32_t W_V_BYTES = NKV * HD * AP_HIDDEN * 2u;
constexpr uint32_t W_IDX_OFF = WAlignUp(W_V_OFF + W_V_BYTES);         // attn_idx_qk_proj [640, 2560]
constexpr uint32_t W_IDX_BYTES = IDX_W * AP_HIDDEN * 2u;
constexpr uint32_t W_QN_OFF = WAlignUp(W_IDX_OFF + W_IDX_BYTES);      // attn_q_norm      [256]
constexpr uint32_t W_QN_BYTES = HD * 2u;
constexpr uint32_t W_KN_OFF = W_QN_OFF + W_QN_BYTES;                  // attn_k_norm      [256]
constexpr uint32_t W_KN_BYTES = HD * 2u;
constexpr uint32_t W_IQN_OFF = WAlignUp(W_KN_OFF + W_KN_BYTES);       // attn_idx_q_norm  [128]
constexpr uint32_t W_IQN_BYTES = IDX_D * 2u;
constexpr uint32_t W_IKN_OFF = W_IQN_OFF + W_IQN_BYTES;               // attn_idx_k_norm  [128]
constexpr uint32_t W_IKN_BYTES = IDX_D * 2u;
constexpr uint32_t W_BYTES = WAlignUp(W_IKN_OFF + W_IKN_BYTES);       // 平面总字节
static_assert(W_BYTES == 71435776u, "8 个 role 合计 = 71,435,776 B ≈ 71.4 MB");

// 段与段之间不重叠（编译期可见）
static_assert(W_K_OFF >= W_Q_OFF + W_Q_BYTES && W_V_OFF >= W_K_OFF + W_K_BYTES &&
              W_IDX_OFF >= W_V_OFF + W_V_BYTES && W_QN_OFF >= W_IDX_OFF + W_IDX_BYTES &&
              W_KN_OFF >= W_QN_OFF + W_QN_BYTES && W_IQN_OFF >= W_KN_OFF + W_KN_BYTES &&
              W_IKN_OFF >= W_IQN_OFF + W_IQN_BYTES, "role 段重叠");

// ============================================================
// 4. cos/sin 表（host 预生成，bf16，`[nPos][ROT]`，行 = [cos(32) | sin(32)]）
// ============================================================
constexpr uint32_t CS_NPOS = M15Kv::PREFILL_M;           // 4097
constexpr uint32_t CS_ROW_ELEMS = ROT;                   // 64
constexpr uint32_t CS_ROW_BYTES = CS_ROW_ELEMS * 2u;     // 128 B（32 B 整数倍 ⇒ 可直接 DataCopy）
constexpr uint32_t AP_CS_BYTES = CS_NPOS * CS_ROW_BYTES;    // 524,416
static_assert(CS_ROW_BYTES % 32u == 0u, "cos/sin 单行可直接 DataCopy");

// ============================================================
// 5. 负向对照的 mode（**方向级**改错；docs/17 §4 要求至少一条且必须有判别力）
// ============================================================
constexpr uint32_t AP_MODE_CONTRACT = 0u;        // 正确
constexpr uint32_t AP_MODE_PLAIN_NORM = 1u;      // R3 反向：乘 `w` 而不乘 `(1+w)`
constexpr uint32_t AP_MODE_ROPE_SIGN = 2u;       // R9 方向反向：`o1 = x1·c + x2·s`、`o2 = x2·c − x1·s`
constexpr uint32_t AP_MODE_NO_ROPE = 3u;         // R9 删除：完全不旋
// **变异对照（塔的纪律：把你的被测对象弄坏，判据必须变红）**——不参与"正确性"档，只用来证明判据有咬合力
constexpr uint32_t AP_MODE_NO_COPY = 4u;      // AIV 不写 gate/v/raw k 三个抄写段 ⇒ `Ap.copy.*` 必须 FAIL
constexpr uint32_t AP_MODE_NO_GEMM = 5u;      // AIC 不跑 4 个 GEMM ⇒ `Ap.y0.*` 必须 FAIL
constexpr uint32_t AP_MODE_N = 6u;

// ============================================================
// 6. GEMM tiling（**只在本 kernel 启动里用**；与 GDN/MoE/hc 的 L1/L0/BufferID 不共存，
//    但取值仍走 M15G 的资源表常量，避免第三份数字）
// ============================================================
// AP_BASE_N = 128 的选法（推导，不是凑）：三个 N 全都 16 对齐，取 gcd(12288, 512, 640) = 128
//   ⇒ 12288/128 = 96、512/128 = 4、640/128 = 5，**无尾块**；128 也是 cube 分形 16 的整数倍。
//   L0B 单 tile = 64×128×2 = 16 KB ≤ 32 KB（ping/pong 各半）✓
// AP_BASE_K = 64：K = 2560 = 64×40 整除 ✓，L0A 单 tile = 64×64×2 = 8 KB ✓
constexpr uint32_t AP_BASE_M = M15G::BASE_M;                // 64（m=1 时按 calcM≥2 处理，见 GDN 的 3510 quirk）
constexpr uint32_t AP_BASE_K = M15G::BASE_K;                // 64
constexpr uint32_t AP_BASE_N = 128;
constexpr uint32_t AP_CUBE_BLOCK = M15G::CUBE_BLOCK;        // 16
constexpr uint32_t AP_L1_A_ELEMS = AP_BASE_M * AP_BASE_K;         // 4096 元素 = 8 KB
constexpr uint32_t AP_L1_B_ELEMS = AP_BASE_K * AP_BASE_N;         // 8192 元素 = 16 KB
constexpr uint32_t L1_A0 = 0;
constexpr uint32_t L1_A1 = L1_A0 + AP_L1_A_ELEMS * 2;       // 8192
constexpr uint32_t AP_L1_B_REGION = 256u * 1024u;           // 与 m1/donor 的 A/B 分区一致
constexpr uint32_t L1_B0 = AP_L1_B_REGION;
constexpr uint32_t L1_B1 = L1_B0 + AP_L1_B_ELEMS * 2;       // 262144 + 16384
static_assert(L1_B1 + AP_L1_B_ELEMS * 2 <= 512u * 1024u, "prolog 的 L1 占用 ≤ 512 KB");
static_assert(L1_A1 + AP_L1_A_ELEMS * 2 <= AP_L1_B_REGION, "A 区不侵 B 区");

constexpr uint32_t L0_A0 = M15G::L0_OFF_0;               // 0
constexpr uint32_t L0_A1 = M15G::L0_OFF_1;               // 32768
constexpr uint32_t L0_B0 = M15G::L0_OFF_0;
constexpr uint32_t L0_B1 = M15G::L0_OFF_1;
static_assert(AP_BASE_M * AP_BASE_K * 2 <= M15G::L0_PP_BYTES, "A2 tile ≤ L0A 半区");
static_assert(AP_BASE_K * AP_BASE_N * 2 <= M15G::L0_PP_BYTES, "B2 tile ≤ L0B 半区");
constexpr uint32_t L0C_BYTES = AP_BASE_M * AP_BASE_N * 4u;     // 32 KB ≤ 256 KB

// ============================================================
// 7. UB 布局（AIV 侧；探针是**独立启动**，可用自己的窗）
// ============================================================
// 序：IN(待处理头) → WB(权重 bf16) → CS(cos/sin 行) → RED(归约/rstd) → OUT(输出头) → CPY(抄写中转)
// 每段都 ≥ 32 B 且不与下一段重叠；`CS` 段故意留 64 bf16 的过读余量（见 probe 里 `LoadAlign` 的说明）。
constexpr uint32_t UB_AP_IN = 0;                         // HD bf16 原始头（512 B）
constexpr uint32_t UB_AP_WB = 1024;                      // HD bf16 norm 权重（512 B）
constexpr uint32_t UB_AP_CS = 2048;                      // 128 bf16 = [cos32|sin32] + 过读余量（256 B）
constexpr uint32_t UB_AP_RED = 2304;                     // VL fp32 归约/rstd（256 B）
constexpr uint32_t UB_AP_OUT = 2560;                     // HD bf16 输出头（512 B）
constexpr uint32_t UB_AP_CPY = 3072;                     // 1024 B 抄写中转（gate/v 各 2 头）
constexpr uint32_t UB_AP_X = 4096;                       // A 操作数 2 行 × AP_HIDDEN bf16（10240 B）——仅 AIC 用
constexpr uint32_t UB_AP_END = UB_AP_X + 2u * AP_HIDDEN * 2u;   // 14336
static_assert(UB_AP_IN + HD * 2u <= UB_AP_WB && UB_AP_WB + HD * 2u <= UB_AP_CS &&
              UB_AP_CS + 256u <= UB_AP_RED && UB_AP_RED + 256u <= UB_AP_OUT &&
              UB_AP_OUT + HD * 2u <= UB_AP_CPY && UB_AP_CPY + 1024u <= UB_AP_X,
              "prolog UB 段重叠");
static_assert(UB_AP_END <= 32768u, "prolog UB 占用（含 A 操作数 staging）≤ 32 KB");

// ============================================================
// 8. flagId（**本 kernel 独占**；GDN/hc/MoE 的号段只在与它们同一次启动里才冲突）
// ============================================================
// AIC mode-0：全体 AIC 对齐（4 个 GEMM 前的 A 载入对齐 + 全部 GEMM 后的 drain 对齐）
constexpr uint16_t AP_AIC_M0_IN = 0;
constexpr uint16_t AP_AIC_M0_OUT = 1;
// mode-2（AIC ↔ 其配对 2 个 AIV）：AIC→AIV「4 个 GEMM 的 GM 输出全部可见」。
// ✅ **M88 r2 已定案：真根因是 AIV 那条臂曾被整块写在 `if ASCEND_IS_AIC {` 里**（见
//    `m15_attn_prolog_probe.h`）⇒ AIV 核上整段被编译掉、AIC 却会 `CrossCoreWaitFlag` 等一个
//    **只有 AIV 能置的** flag ⇒ 死锁。**与 flag 语义/配对/编号无关。**
//    ⚠ r1 曾在 `evidence/attn_prolog/README.md §4` 把根因写成「AIC 的 mode-2 set 在配对 AIV 不
//    消费时会阻塞 / 配对关系不符」，**那是错的**（r1 复审用 dbg=2 档 rc=0 就否证了「set 阻塞」：
//    该档会发 flag 而 AIV 不消费，却不挂）。r1 那组 rc 读数本身可复现，**当时的解释作废**。
//    更正后重跑：`runs=prolog` 由 rc=124 变 rc=1（不再挂死），判据开始出 PASS/FAIL。
//    排查期间试过的三种替代形态（AIC 单方面 set / GDN 式严格交替 / flagId 换号）**本来就都得挂**
//    （AIV 是死代码），与 flagId 无关 —— 那三次尝试现已被真因解释掉，不再作为「已排除原因」。
constexpr uint16_t AP_V2A_READY = 4;          // AIV→AIC（仅在交替形态下用；当前未启用）
constexpr uint16_t AP_A2V_GEMM = 5;           // AIC→AIV
constexpr uint8_t AP_CC_M0 = 0x0;
constexpr uint8_t AP_CC_M2 = 0x2;

}  // namespace M15AP

#endif  // M15_ATTN_PROLOG_H
