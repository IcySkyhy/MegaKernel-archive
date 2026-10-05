/**
 * m15_layer_kernel.h —— **融合 per-layer kernel** 的入口符号与段序
 *
 * 每层一次 `__mix__(1,2)` 启动（blockDim = AIC 数）。kernel 模板参数 `HC` 决定**两相位**还是
 * **四相位**形态：
 *
 *   两相位（HC=false，M40 形态；`m15_layer_kernel_gdn` / `_attn`）：
 *   ┌─ 相位 A（子层段）───────────────────────────────────────────────────────┐
 *   │  KIND_GDN : GdnLayerChain S1-S7（= m14/m15 已验收的 GDN 段，一字未动）   │
 *   │  KIND_ATTN: m15_attn_passthrough_body  y = x（占位，待 QSA）            │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *                        ↓ PipeBarrier<PIPE_ALL> + 全体 AIV mode-0 barrier
 *   ┌─ 相位 B（MoE 段）───────────────────────────────────────────────────────┐
 *   │  M15M::MoeLayerChain S1-S10（= m13 已验收的 MoE 段，一字未动）          │
 *   └─────────────────────────────────────────────────────────────────────────┘
 *
 *   四相位（HC=true，M58 形态；`m15_layer_kernel_gdn_hc` / `_attn_hc`）：
 *   ┌─ 相位 H1（hc 边界 #1 = attn_hc） ┐ hc(attn): H,BO,IJ ─► BLK_attn / H' / IJ'
 *   ├─ 相位 A （子层段）──────────────┤ 消费 BLK_attn → attn_out
 *   ├─ 相位 H2（hc 边界 #2 = mlp_hc）─┤ hc(mlp): H'(或 H), attn_out, IJ' ─► BLK_mlp / H'' / IJ''
 *   └─ 相位 B （MoE 段）──────────────┘ 消费 BLK_mlp → yLayer（= 下一层的 pending BO）
 *
 *   **M65 的 PLE 打断点**（`LayerArgs::hcPleBreak != 0`，只用于 0-based 层 1 = `ple_layer_ids`
 *   那一层）：相位 H1 拆成三段，中间夹 PLE 段的挂载点（PLE 本体未实现，见 §3b）：
 *   ┌─ H1a（combine-only）──┐ 只 W0+S1：H' = bf16(H + BO·injW) 就地物化（AIC 不跑 GEMM）
 *   ├─ 相位边界 ───────────┤
 *   ├─ PLE 段（**占位**）──┤ 空操作；PLE 将来在这里直接改 10240 宽的多流态
 *   ├─ 相位边界 ───────────┤
 *   ├─ H1b（mix-only）─────┤ 读物化好的 bf16 H' → 基准化/down/silu/up/gate → BLK_attn / IJ'
 *   └─ 相位边界 ───────────┘
 *   语义依据 = `V-N:model.py:290-301`（PLE 之前先物化 pending combine、之后才 mix）。
 *
 * 四相位形态与 vLLM 的 `Qwen4ExpDecoderLayer.forward` 逐句对应：
 *   `hidden, block_input, injection = attn_hc.combine_and_mix(hidden, prev_block_output, prev_injection)`
 *   `attn_out = attention(block_input)`
 *   `hidden, mlp_block_input, injection = mlp_hc.combine_and_mix(hidden, attn_out, injection)`
 *   `mlp_out  = mlp(mlp_block_input)`（= 本层的 `yLayer` = 下一层的 `prev_block_output`）
 * 层 1 则是 `attn_hc.combine(...)` → `ple(...)` → `attn_hc.combine_and_mix(..., None, None)`。
 *
 * 48 层的**层链**（M65）：每层一次四相位启动，层间只搬三条长寿状态
 * （`H [m,10240]` bf16、pending `BO [m,2560]` bf16、`IJ [m,4]` bf16），全部**零拷贝**：
 * H'' 直接落在该层 hc1 ws 的 `WS_HCP`、IJ'' 落在 `WS_OH+320*2B`（下一层按行距 `OH_W` 直读），
 * 只有 pending BO 需要一块独立平面（host 侧双缓冲）。末层之后接 `m15_final_mixer_kernel`。
 *
 * **层链数据流（两相位）**：MoE 段的输入取 `yLayer`（相位 A 的出口），残差取 `resZero`（全零）
 * ——即 m13 的 `sliceMode=1`「层链模式」。**四相位形态**下 MoE 段的输入取 `hcWs1 + WS_BLK`
 * （hc(mlp) 的 block input），出口仍是 `yLayer`（做下一层的 pending BO）。
 *
 * **为什么相位交界的同步是这两条**（M40 的论证，M58 原样适用于三条边界）：
 *   1. `PipeBarrier<PIPE_ALL>()`：核内 6 条 pipe 全部 drain。三段的 UB 段窗、L1/L0C 区、
 *      BufferID 编号都是**同址复用**（见 m15_layer_resources.h §2/§5），相位之间必须有一个
 *      全 drain —— BufferID 是核内 token，跨相位复用只有在「上一相位所有 pipe 都不再持有
 *      该 UB 区」时才是安全的；这是 docs/12 §4「段窗叠放以段间 barrier 为前提」的显式版。
 *   2. `CrossCoreSetFlag/WaitFlag<CC_MODE0>`（AIV 全体）：每个相位的出口张量都是**全体 AIV
 *      协作写**的，而下一位相的消费者用**另一套行切分**读它 —— 跨核可见性必须用 CrossCore
 *      （BufferID 管不了跨核，docs/05 §6.1）。set 挂 PIPE_MTE3（drain 本核写）、wait 挂最窄的
 *      PIPE_MTE2（挡本核后续读），于是「本核写完成 → 全体到齐 → 本核才开始读」三条同时成立。
 *      AIC 侧不需要相位边界同步：AIC 在相位之间没有数据依赖（各段的 A 操作数由 AIV 产出，
 *      走各段自己的 mode-2 交接），核内只需第 1 条的 PipeBarrier。
 *      **例外（M164/M1）**：H2 → 相位 B 的 **MoE router 输入** `pfHcBlk1` 是本段的**段输入**
 *      （全体 AIV 写、每个 AIC 读 = 跨类型 all-to-all），不经过 MoE 段自带的 mode-2 交接 ⇒
 *      该边界对 AIC 补一条 mode2 + 全体 AIC mode0 的握手（见 §1b `M15L_H2ToBHandshake`）。
 *
 * 入口符号（docs/15 §5 方案 B：单 TU、mode 由 host 选符号、kernel 内无 m 分叉）：
 *   m15_layer_kernel_gdn / m15_layer_kernel_attn          —— 两相位（M40 交付形态）
 *   m15_layer_kernel_gdn_hc / m15_layer_kernel_attn_hc    —— 四相位（**M58/M65 交付形态**；
 *                                                            `hcPleBreak` 可让层 1 走 PLE 打断点）
 *   m15_final_mixer_kernel                                —— **M65**：末层之后的全局 mixer（MODE_FINAL_MIX）
 *   m15_moe_segment_kernel                                —— 仅验证用（MoE 段对照路）
 *   m15_hc_segment_kernel                                 —— 仅验证用（hc 单边界对照路，含 combine-only）
 *   m15_layer_kernel_{gdn,attn}_prefill                   —— **M110 真入口**（prefill）；
 *                                                            M132（Wave C）已接 B1(GDN)/B4(MoE)/B5(hc)，
 *                                                            B2/B3(attention) 未接（B3/M101 未合入）
 *
 * **prefill 的形态边界（M110 交挂载点 → M132 接段体）**：相位拓扑 = H1（hc）→ 相位 A（子层段）→
 * H2（hc）→ 相位 B（MoE）。`pfWired == 0`（缺省，host 的 `M15_PREFILL_WIRE`）时相位 A 是**结构性
 * 占位**（逐行 x→y，数学未实现，显式标注）；`pfWired != 0` 时才进段体：GDN 层跑 B1、hc 跑 B5、
 * MoE 跑 B4；**attention 层（B2/B3）本轮没接** ⇒ 它的相位 A 不产出任何东西（响亮失败，不静默通过）。
 * 段体实参（权重/激活平面）目前由 host 传 nil ⇒ `wire=1` 的实跑判据归 **Wave D**；本轮只保证
 * 「入口存在、能编译、`wire=0` 的既有路径一字未变」。
 *
 * `m` 以运行期参数进签名（M33/discs15 §5.3 ★5），但**kernel 内没有任何按 m 的分叉**：
 * tiling（BASE_M/BASE_K/BASE_N）是编译期常量，m 只进 curM mask 与行循环上界。
 *
 * **M65 的范围边界（未接线项，README 有显式清单）**：PLE **本体**未实现（表 95.37 GiB 落不下，
 * docs/14 §10 第 1 条是阻塞项）——本 mission 交付的是**打断点的结构**（combine 先物化 → PLE 挂载
 * 点 → mix）与它的判据；48 层链的层间 handoff（H/BO/IJ 三条长寿状态）已接线并验证。
 */

#ifndef M15_LAYER_KERNEL_H
#define M15_LAYER_KERNEL_H

namespace M15L {

// ============================================================
// 0pre. M132（Wave C）：prefill 峰值槽 ↔ 段设备常量的**跨文件机器见证**
// ============================================================
// `m15_layer_resources.h` §3c 的槽值是字面量（那节**不能**反向包含 `M15PFR`：`m15_moe_prefill_res.h`
// 已经包含它，构成环）⇒ 在这里把每个数逐一钉回各自的设备常量：任一处漂了都编不过。
// 前提：本 TU 已先包含 `m15_gdn_prefill.h` / `m15_hc_prefill.h` / `m15_moe_prefill.h`（见 `.asc`）。
static_assert(PF_UB_BYTES_GDN == M15G::GP_UB_BYTES, "PF_UB_BYTES_GDN 与 M15G::GP_UB_BYTES 漂了");
static_assert(PF_L1_BYTES_GDN == M15G::GP_L1_BYTES, "PF_L1_BYTES_GDN 与 M15G::GP_L1_BYTES 漂了");
static_assert(PF_L0C_BYTES_GDN == M15G::GP_L0C_BYTES, "PF_L0C_BYTES_GDN 与 M15G::GP_L0C_BYTES 漂了");
static_assert(PF_UB_BYTES_HC == M15H::HcPF::UB_PEAK_PF, "PF_UB_BYTES_HC 与 M15H::HcPF::UB_PEAK_PF 漂了");
static_assert(PF_UB_BYTES_MOE == M15PFR::PF_UB_PEAK, "PF_UB_BYTES_MOE 与 M15PFR::PF_UB_PEAK 漂了");

// ---- M197（B2 挂载）：attention 前端 5 个 scratch 平面的**尺寸见证**（W1 的静态校验）----
// 与 `m24_attn_prefill/README.md` §5.2 的字节表逐项钉死：宿主定尺漂了编不过。
// 前提同上面 5 行：本 TU 已先包含 `m15_attn_kv.h`（→ `m15_attn_cache.h` → `m15_attn_prefill.h`）。
static_assert(M15PF::PF_ROWS * M15PF::PF_Y0_STRIDE * 2u == 3571712u, "pfY0 定尺 != PF_ROWS*PF_Y0_STRIDE*2");
static_assert(M15PF::PF_ROWS * M15PF::PF_OUT_STRIDE * 2u == 3571712u, "pfOut 定尺 != PF_ROWS*PF_OUT_STRIDE*2");
static_assert(M15AC::AC_IN_BYTES == 299264u, "pfIn 定尺 != AC_IN_BYTES");
static_assert(M15AC::AC_POOLED_BYTES == 8192u, "pfPooled 定尺 != AC_POOLED_BYTES");
static_assert(M15AC::AC_FLAG_LANES * 4u == 32u, "pfFlag 定尺 != AC_FLAG_LANES*4");
static_assert(M15PF::PF_Y0_STRIDE == M15PF::PF_OUT_STRIDE, "y0 与 out 的行距必须同源（README §5.2 同值）");

// ============================================================
// 0. 融合 kernel 的入参
// ============================================================

struct LayerArgs {
    __gm__ uint8_t* ws;          // 融合 ws 基址（GDN 段 @0，MoE 段 @WS_MOE_OFF）
    __gm__ uint8_t* xLayer;      // 层输入 bf16 [m, HIDDEN]（行 0 = 本 token）
    __gm__ uint8_t* resLayer;    // GDN 段 S1 的残差（层循环里 = 全零缓冲）
    __gm__ uint8_t* yLayer;      // 层输出 bf16 [m, HIDDEN]（两相位：相位 A 出口 = 相位 B 入口 = 层出口）
    __gm__ uint8_t* resZero;     // 全零残差（MoE 段 S1 的残差入口）

    // ---- GDN 段权重/状态（KIND_ATTN 时全部为 nullptr，不被触碰）----
    __gm__ uint8_t* gamma1;
    __gm__ uint8_t* gamma2;
    __gm__ uint8_t* gammaG;
    __gm__ uint8_t* wIn;
    __gm__ uint8_t* wOut;
    __gm__ uint8_t* convW;
    __gm__ uint8_t* convBias;
    __gm__ uint8_t* aLog;
    __gm__ uint8_t* dtBias;
    __gm__ uint8_t* convState;
    __gm__ uint8_t* ssmState;

    // ---- MoE 段权重（各段共用一套槽内布局，见 m15_layer_resources.h §1）----
    __gm__ uint8_t* moeRouter;
    __gm__ uint8_t* moeSgate;
    __gm__ uint8_t* moeWGu;
    __gm__ uint8_t* moeSGu;
    __gm__ uint8_t* moeWDn;
    __gm__ uint8_t* moeSDn;
    __gm__ uint8_t* moeWGuShd;
    __gm__ uint8_t* moeSGuShd;
    __gm__ uint8_t* moeWDnShd;
    __gm__ uint8_t* moeSDnShd;
    __gm__ uint8_t* moeGamma1;
    __gm__ uint8_t* moeGamma2;

    uint32_t m;      // 运行期 token 行数（本 mission 恒 1；★5 要求保留为运行期参数）
    uint32_t topk;   // 运行期 top-k（本 mission = golden/缩形档的 2）

    // ---- M58：hc 段的层边界张量与权重（**只有四相位入口填这些字段**；两相位入口全部为
    //      nullptr/0，kernel 内 `if constexpr (HC)` 保证不被触碰）----
    __gm__ uint8_t* hcH;         // [M_MAX, HYPER] bf16 层输入（4 路多流残差 H）
    __gm__ uint8_t* hcBo;        // [M_MAX, HID]   bf16 pending block output（上一层 mlp_out）
    __gm__ uint8_t* hcIj;        // [M_MAX, hcIjStride] bf16 pending injection logits（前 4 列有效）
    __gm__ uint8_t* hcAttnOut;   // [M_MAX, HID]   bf16 子层段出口（= hc(mlp) 的 BO）
    __gm__ uint8_t* hcWs0;       // hc 边界 #1（attn_hc）的 ws（host 按层号编址：见 §1b）
    __gm__ uint8_t* hcWs1;       // hc 边界 #2（mlp_hc）的 ws
    __gm__ uint8_t* hcAttnNorm;
    __gm__ uint8_t* hcAttnDown;
    __gm__ uint8_t* hcAttnInj;
    __gm__ uint8_t* hcAttnUp;
    __gm__ uint8_t* hcMlpNorm;
    __gm__ uint8_t* hcMlpDown;
    __gm__ uint8_t* hcMlpInj;
    __gm__ uint8_t* hcMlpUp;
    uint32_t hcIjStride;   // IJ 源行距（元素）：独立 [m,16] 平面 = M15H::IJ_STRIDE
    uint32_t hcAttnMode;   // H1 的 mode：层 0 的 attn 边界 = MODE_MIX，其余 = MODE_COMBINE_MIX
    uint32_t hcMlpMode;    // H2 的 mode：恒 MODE_COMBINE_MIX（attn_out 是 pending BO）

    // ---- M65：PLE 打断点（**只在 0-based 层 1 = ple_layer_ids 那一层非 0**）----
    // != 0 ⇒ 相位 H1 拆成三段：combine-only（`MODE_COMBINE_ONLY`，把 pending combine 物化成
    // bf16 的 `H'`）→ PLE 段（**本 mission 未实现**，见 `M15L_PlePhasePlaceholder`）→ mix-only
    // （`MODE_MIX`，按 `V-N:model.py:293-297` 在 PLE 之后才做 norm→…→BLK）。
    // 语义来源：PLE 直接往 10240 宽的多流态上写，所以 combine 必须**先物化**、mix 必须**后跑**。
    uint32_t hcPleBreak;

    // ---- M97：attention 相位 A（**只有 KIND_ATTN 的四相位入口填这些字段**；两相位入口与 GDN 的
    //      四相位入口全部为缺省 nullptr/0，kernel 内 `if constexpr (KIND == KIND_ATTN)` 保证不被触碰）----
    // 语义：把 M88 的前端 prolog（`m15_attn_prolog_probe.h`）接进融合 kernel 的**相位 A**，
    // 取代原来的 `m15_attn_passthrough_body` 占位直通。W5/W6 的落点。
    // 形状/段布局/UB/flag 的权威登记表 = `m15_attn_prolog.h`（本节不重复它的数字）。
    // `apX` 按 **2 行**给（AIC 侧 Nd2Nz 的 m=1 quirk：`AttnGemm::CALC_M = 2`，结果只写第 0 行）。
    __gm__ uint8_t* apW;         // M15AP::W_BYTES：8 个 role 的权重平面（512 B 对齐）
    __gm__ uint8_t* apX;         // A 操作数：2 行 × M15AP::AP_HIDDEN bf16（行 0 = 本 token）
    __gm__ uint8_t* apCs;        // M15AP::AP_CS_BYTES：host 预生成的 cos/sin 表（设备不算超越函数）
    __gm__ uint8_t* apY0;        // M15AP::Y0_N bf16：4 个 GEMM 的原始输出（段布局见其 §2）
    __gm__ uint8_t* apOut;       // M15AP::AP_OUT_N bf16：prolog 的输出
    uint32_t apPos;              // 旋转用的位置（= cos/sin 表的行号）
    uint32_t apMode;             // 负向/变异对照档（M15AP::AP_MODE_*）；契约档 = 0
    uint32_t layer;              // **层号（0-based）**：现接口此前不带层号（M76 对 README 那句的证伪）

    // ---- M100：PLE 段的真接线（**只在 pleW != nullptr 的那一层**；两相位入口与其余层全部
    //      nullptr/0，kernel 内 `if (A.pleW != nullptr)` 保证不被触碰）----
    // 语义权威 = `ple/PLE_SPEC.md`（§1 输入 / §2 权重 / §3 输出 / §4.1 五步）。
    // device 实现 = `m15_ple_wire.h` 里的 `M85P::PleGather/PleGemv/PleGateItem/PleConvItem`，
    // 而该头是**从 `m15_ple.asc` 机械抽取的生成物**（生成器
    // `evidence/ple_wire/lift_ple_device_segment.py`，`--check` 保证与 `m15_ple.asc` 逐字节一致）
    // ⇒ **改了 `m15_ple.asc` 必须重跑生成器**，不会自动生效。
    //
    // 段内平面布局 = `M15L::PLEW::*`（下节，host 与 device 同一批常量）。
    // `pleHid`/`pleOut` **都指向 `hcWs0 + WS_HCP`**（O1「就地写回 I1 的缓冲」）⇒ 不额外物化。
    __gm__ uint8_t* pleW;        // 权重 slab：wcat[12800,2560] | wtap[4,10240] | nk | nq | ncw
    __gm__ uint8_t* pleScr;      // scratch slab：ids/emb/kv/gated/normed/sidx/fail/exp64/hmBad/st
    __gm__ uint8_t* pleTable;    // host-mapped 注册窗口的 **device 指针**（M92 的 ACL 链路）
    __gm__ uint8_t* pleIds;      // [T] int32：本 step 的 token id（① 的输入）
    __gm__ uint8_t* pleQsl;      // [nReq+1] int32：每请求的 chunk 起始偏移
    __gm__ uint8_t* pleCtx;      // [nReq, NC] int32：每请求最近 NC=2 个已算 token（不足填 EOS）
    __gm__ uint8_t* pleM;        // [3] int64：layer_multipliers
    __gm__ uint8_t* pleSz;       // [NG=16] int64：ngram_heads_vocab_sizes
    __gm__ uint8_t* pleOf;       // [NG=16] int64：ngram_heads_offsets
    uint32_t pleTok;             // T（本 step 的 token 数）
    uint32_t pleNReq;            // 请求数
    uint32_t pleTableRows;       // 窗口行数（= PLEW 的 winRows；表基址的 SetGlobalBuffer 上界）
    uint32_t pleStageMask;       // bit0..3 = ②③④⑤（必须是前缀）、bit4 = ① 也跑；0 = 整段跳过
    uint32_t pleNegMask;         // 负向/变异对照掩码（原样透传给 M85P 的实现；契约档 = 0）
    uint32_t pleWinBase;         // 窗口第 0 行对应的全局行 id（`row = id - winBase`，M92 §2.3）
    uint32_t pleWinRows;         // 窗口行数（0 = 关窗口模式，走平坦 GM 表）
    uint32_t pleStateSlots;      // short-conv 状态槽**分配上界**（`H_PleArgsOf` 填 `PLEW::T_MAX`）；
                                 // **活槽数 = `pleNReq`**（`ple/PLE_SPEC.md §2 I6` 每请求一行），
                                 // 段体按 `min(pleNReq, pleStateSlots)` 定尺（M191）

    // ============================================================
    // M110（Wave A）：**prefill 的挂载点字段**
    // ============================================================
    // **与 PLE 字段的相对位置**（塔的补充第 5 条要求显式说明，不许默默重排）：
    //   M100 的 PLE 字段是**尾部收尾**（`pleStateSlots` 是上面最后一项）。本节全部字段**追加在它
    //   之后**，`M15L_LAYER_ARGS_DECL/FILL`（两相位）与 `M15L_LAYER_HC_ARGS_DECL/FILL`（四相位）
    //   的**实参顺序与内容一字未动** ⇒ 两个已验收入口（含 M100 的 PLE 接线）的 ABI 与读数零变化。
    //   prefill 用它自己的 `M15L_LAYER_PREFILL_ARGS_DECL/FILL`（= 基础 30 实参 + 下面这批 13 项）。
    //
    // 语义（契约，B1–B5 按它接）：
    //   · **`pos`**：二选一 —— `pfPos != nullptr` 时按行取表（支持 batch>1 的将来形态）；
    //     否则 `pos(row) = pfPosBase + row`（**单请求连续**，与官方 `query_start_loc` 语义相容，
    //     也是 m=4097 验收档的形态）。两者**同时给**时以 `pfPos` 为准（host 侧有 assert）。
    //   · **`slot_mapping`：本契约不引入**（M103-3 的结论 + M82 的 D2）。所有 KV 地址由 `pos` 经
    //     `M15Kv::M15KV_KV_*` 宏算出；`pfBlockTable == nullptr` **就是**"恒等页表/单请求"的
    //     可执行表示。若将来要支持 paged 多请求，扩展点是 `pfBlockTable`，不是新加一个数组。
    //   · **`layerK`**：attention 层序号（0..11，= `M15Loop::AttnSlot`）；GDN 层填
    //     `PF_LAYER_K_NONE`（= `M15Loop::NO_SLOT` 的同一个值）。
    __gm__ uint8_t* pfKv;            // 主 KV paged 区基址（M15Kv::KV_PLANE_BYTES）
    __gm__ uint8_t* pfComp;          // compressed key 区（M15Kv::COMP_PLANE_BYTES）
    __gm__ uint8_t* pfRing;          // raw key ring 区（M15Kv::RING_PLANE_BYTES）
    __gm__ uint8_t* pfPack;          // packed indices 区（M15Kv::PACK_PLANE_BYTES）
    __gm__ uint8_t* pfPos;           // per-row 位置表 u32[m]（可空，见上面的契约）
    __gm__ uint8_t* pfBlockTable;    // 页表 u32[m]（nullptr = 恒等表 = 单请求）
    __gm__ uint8_t* pfCounts;        // MoE prefill：每专家计数 i32[E]（B4 的入口）
    __gm__ uint8_t* pfExpertOffsets; // MoE prefill：前缀和 i32[E+1]（紧凑槽寻址的起点）
    uint32_t pfPosBase;              // 单请求连续时的位置起点
    uint32_t pfLayerK;               // attention 层序号 k（GDN 层 = NO_SLOT）
    uint32_t pfStageMask;            // 段序截断（bring-up 定位；0 = 全开）
    uint32_t pfWired;                // 0 = 结构占位（数学未实现）；1 = 段体挂载点（段体缺席 ⇒ 响亮失败）
    uint32_t pfMutant;               // 负向对照掩码（只在验证档用；契约档 = 0）

    // ============================================================
    // M132（Wave C）：**三条 prefill 段体（B1 / B4 / B5）的实参字段**
    // ============================================================
    // 全部**追加**在上面的 M110 prefill 块之后；`M15L_LAYER_ARGS_*`（两相位）与
    // `M15L_LAYER_HC_ARGS_*`（四相位）两个已验收入口的实参表**一字未动** ⇒ 它们的 ABI 零变化。
    // 只有 prefill 入口的 `M15L_LAYER_PREFILL_ARGS_*` 追加对应实参（见 §5）。
    // 形状/越界 `static_assert` 一律在各自的段头里（本文件不放宽任何一条）。
    //
    // ---- B1（GDN prefill，M115）：q/k/v/g/β/h0/out/ht 的 GM 平面 + 段私有 scratch ----
    // `h0`（初始状态）与 `ht`（出口状态）**复用**上面的 `ssmState`（GDN 的状态平面，prefill 里
    // in-place 语义由 Wave A 决定）⇒ 不新增第二个 h 字段。字段名 `wsQ..` 照 B1 README §3(b)/§3(h)。
    __gm__ uint8_t* wsQ;                 // [NK=16, m, 128] fp32（q/k 尾部须留 ≥64 行可读且置零，README §3(f)）
    __gm__ uint8_t* wsK;                 // [NK=16, m, 128] fp32
    __gm__ uint8_t* wsV;                 // [48, m, 128] fp32
    __gm__ uint8_t* wsG;                 // [48, tp] fp32（tp = align8(m)）
    __gm__ uint8_t* wsBeta;              // [48, tp] fp32
    __gm__ uint8_t* wsO;                 // [48, m, 128] fp32（段出口）
    __gm__ uint8_t* wsGdnPrefillScratch; // [nAic, 753,664 B] 段私有 GM 工作区（GdnPrefillScratchBytes）
    // ---- B4（MoE prefill，M118）：只新增 router 的 pad 后权重平面；其余 12 个复用 decode 的 MoE 权重 ----
    __gm__ uint8_t* pfMoeRouterWPad;     // [E_PAD=640, HIDDEN=2560] bf16（0..511 专家 / 512 共享门 / 513..639 复制 512）
    // ---- B5（hc prefill，M119）：最小 3 指针；arena 复用 hcWs0/1，证据平面（injw/rstd）不新增 ----
    __gm__ uint8_t* pfHcArena0;          // 边界 #1 的 arena（≥ HcPfH::ArenaBytes(m)）
    __gm__ uint8_t* pfHcArena1;          // 边界 #2 的 arena
    __gm__ uint8_t* pfHcBlk0;            // 边界 #1 的 BLK 平铺面 [m, HID] bf16（可空 ⇒ 留在 arena 阻塞布局）
    __gm__ uint8_t* pfHcBlk1;            // 边界 #2 的 BLK 平铺面（可空）
    __gm__ uint8_t* pfHcIj0;             // 边界 #1 产出的 ij handoff 平铺面 [(m+M_MAX)*32 B]
    // ---- M197（B2 挂载）：attention 前端（prefill 相位 A）的 5 个 scratch 平面 ----
    // 规格 = `m24_attn_prefill/README.md` §5.2（"在 LayerArgs 追加 5 个字段"那一支）。
    // 每 chunk 覆写、容量 = 一个 chunk（PF_ROWS = 128 行）；KIND_GDN 层不用（传 nullptr）。
    __gm__ uint8_t* pfY0;                // [PF_ROWS, PF_Y0_STRIDE=13952] bf16 = 3,571,712 B
    __gm__ uint8_t* pfOut;               // [PF_ROWS, PF_OUT_STRIDE=13952] bf16 = 3,571,712 B
    __gm__ uint8_t* pfIn;                // AC_IN_BYTES = 299,264 B（cache 输入面）
    __gm__ uint8_t* pfPooled;            // AC_POOLED_BYTES = 8,192 B（诊断面）
    __gm__ uint8_t* pfFlag;              // AC_FLAG_LANES*4 = 32 B（门控 lane）
};

// ------------------------------------------------------------
// 3a. M100：PLE 段的段内平面布局（**host 与 device 同一批常量**）
// ------------------------------------------------------------
// 两块 slab 由 host 一次分配、一次 H2D（权重）/清零（scratch），device 只按这些编译期偏移取址
// ——「所有 buffer 的地址自管、尽量编译期静态分配」这条人类裁定在这里的落点。
// **T 无关**：所有平面按 `M_MAX` 定尺（decode 档 T=1；host 侧也按同一常量算字节数）。
// M191：short-conv 状态的**活槽数** —— host 搬运（`m15_layer_loop.asc::H_PleCarryState`）与
// device 定尺（`M15L_PleBody` 的 `SetGlobalBuffer`）**共用这一个表达式**，避免两处边界分叉。
// = 活请求数 `nreq`，且不超过分配上界 `stateSlots`（防越界）；`nreq == 0`（未配置档）退回
// `stateSlots`，与改前「整段按分配上界处理」的缺省语义一致。依据 `ple/PLE_SPEC.md §2 I6`
// （状态每请求一行 ⇒ 活槽数 = 活请求数 `pleNReq`）。
// 用**宏**而不是 inline/constexpr 函数：bisheng 的设备 pass 不为头里的非 `__aicore__` 函数
// 发射定义（实测 `constexpr` 版本在 `m15_layer_loop.asc` 上 ld.lld 报 undefined symbol），
// 而宏在 host 与 device 两侧都展开。
#define M15L_ACT_STATE_SLOTS(nreq, stateSlots) \
    (((nreq) == 0u || (nreq) > (stateSlots)) ? (stateSlots) : (nreq))

namespace PLEW {
using M85P::HID;
using M85P::HYPER;
using M85P::KVW;
using M85P::NG;
using M85P::VL;
using M85P::STLEN;
using M85P::HDIM;
using M85P::FAIL_SLOTS;
constexpr uint32_t T_MAX = M15H::M_MAX;          // 64
constexpr uint32_t NC_MAX = M15H::M_MAX;         // 请求数上界（= 行数上界）

// ---- 权重 slab：wcat | wtap | nk | nq | ncw ----
constexpr uint32_t W_CAT = 0;                                  // [KVW, HID] bf16（行主序 (out,in)）
constexpr uint32_t W_TAP = W_CAT + KVW * HID * 2u;             // [KCONV, HYPER]：wtap[k*HYPER+c]
constexpr uint32_t W_NK = W_TAP + 4u * HYPER * 2u;             // [HYPER] bf16
constexpr uint32_t W_NQ = W_NK + HYPER * 2u;
constexpr uint32_t W_NCW = W_NQ + HYPER * 2u;
constexpr uint32_t W_BYTES = W_NCW + HYPER * 2u;               // 65,740,320

// ---- scratch slab（全部 32B 对齐）----
constexpr uint32_t S_IDS = 0;                                  // [T_MAX, NG] int64
constexpr uint32_t S_EMB = S_IDS + T_MAX * NG * 8u;            // [T_MAX, HID] bf16
constexpr uint32_t S_KV = S_EMB + T_MAX * HID * 2u;            // [T_MAX, KVW] bf16
constexpr uint32_t S_GATED = S_KV + T_MAX * KVW * 2u;          // [T_MAX, HYPER] bf16
constexpr uint32_t S_NORMED = S_GATED + T_MAX * HYPER * 2u;    // [T_MAX, HYPER] bf16
constexpr uint32_t S_SIDX = S_NORMED + T_MAX * HYPER * 2u;     // [T_MAX] int32（槽位索引）
constexpr uint32_t S_FAILI = S_SIDX + T_MAX * 4u;              // [FAIL_SLOTS] u32：① 的坏 id 计数
constexpr uint32_t S_FAILB = S_FAILI + FAIL_SLOTS * 4u;        // [FAIL_SLOTS] u32：②③④⑤ 的坏值计数
constexpr uint32_t S_EXP64 = S_FAILB + FAIL_SLOTS * 4u;        // [T_MAX*NG*VL] bf16：设备侧期望行首
constexpr uint32_t S_HMBAD = S_EXP64 + T_MAX * NG * VL * 2u;   // [FAIL_SLOTS*8] fp32：设备侧行校验
constexpr uint32_t S_STIN = S_HMBAD + FAIL_SLOTS * 8u * 4u;    // [slots, STLEN, HYPER] bf16 旧状态
constexpr uint32_t S_STOUT = S_STIN + T_MAX * STLEN * HYPER * 2u;   // 同形的新状态
constexpr uint32_t S_BYTES = S_STOUT + T_MAX * STLEN * HYPER * 2u;

// stageMask 的位（① 与 ②③④⑤ 分开；② 的 4 位必须是前缀，M85 的契约）
constexpr uint32_t ST_IDS = 16u;
constexpr uint32_t ST_BODY = 15u;
}  // namespace PLEW

// ---- M100 预算核对（编译期）----
// UB：PLE 段的 UB 窗（`M85P::UB_END`）必须 ≤ 融合 kernel 的相位峰值 —— 段间同址叠放的合法性
// 由调用点的两条相位边界 + `PipeBarrier<PIPE_ALL>` 保证（与 hc/GDN/MoE 三段同一论证）。
static_assert(M85P::UB_END <= UB_PEAK_FUSED,
              "PLE 段的 UB 峰值超出融合 kernel 的相位峰值（m15_layer_resources.h §2）");
// 权重 slab 的字节数（改布局常量即在此 FAIL；host 侧按同一常量分配）
static_assert(PLEW::W_BYTES == 65679360u, "PLE 权重 slab 字节数变了（PLEW::W_*）");
static_assert(PLEW::S_BYTES == 28325120u, "PLE scratch slab 字节数变了（PLEW::S_*）");
// flagId **相邻性**（**M124 起 ②→③→④ 是跨核型交接，登记见下**）。
// 执行序（层 1 = PLE 打断点层）：
//   AIV mode0： hc(H1a) AV0(12) → PLE_IN(8) → **B4(14)** → PLE_OUT(9) → hc(H1b) AV0(12) …
//   AIV mode2： hc(H1a) 8/9/10/11 → **B2(12) → B3(13)** → hc(H1b) 8/9/10/11 → GDN 4–7 → MoE 0–3
//   AIC mode0： hc(H1a)（combine-only，无 AIC 同步）→ **CUBE_IN(14) → CUBE_OUT(15)** →
//              hc(H1b) 0/1/2/3 → GDN 12–15 → MoE 8–11
// （②→③ 与 ③→④ 改走 mode2 之后，AIV mode0 上原来 B2/B3 那两个点**不再存在** —— 所以下面
//   比的是新序里的相邻对；`M85P::FLAG_B2/B3/B4` 的登记见 `m15_layer_resources.h` 的 `FLAG_SEQ`。）
static_assert(FLAG_PLE_IN_BOUND_AIV != M85P::FLAG_B4, "PLE_IN 边界与 ④→⑤ 相邻且同号（AIV mode0）");
static_assert(M85P::FLAG_B4 != FLAG_PLE_OUT_BOUND_AIV, "④→⑤ 与 PLE_OUT 边界相邻且同号（AIV mode0）");
static_assert(FLAG_PLE_OUT_BOUND_AIV != M15H::FLAG_AV0, "PLE_OUT 边界与 hc(H1b) 的 AV0 相邻且同号");
static_assert(M15H::FLAG_C2A_GATE != M85P::FLAG_B2,
              "hc(H1a) 的最后一条 mode2 与 ②→③ 相邻且同号（AIV/AIC mode2）");
static_assert(M85P::FLAG_B2 != M85P::FLAG_B3, "②→③ 与 ③→④ 相邻且同号（mode2）");
static_assert(M85P::FLAG_B3 != M15H::FLAG_A2C_XN,
              "③→④ 与 hc(H1b) 的第一条 mode2 相邻且同号（AIV/AIC mode2）");
static_assert(M15H::FLAG_AC3 != M85P::FLAG_CUBE_IN,
              "hc(H1) 的最后一条 AIC mode0 与 ③ 入口相邻且同号（AIC mode0）");
static_assert(M85P::FLAG_CUBE_IN != M85P::FLAG_CUBE_OUT, "③ 入口与出口相邻且同号（AIC mode0）");
static_assert(M85P::FLAG_CUBE_OUT != M15H::FLAG_AC0,
              "③ 出口与 hc(H1b) 的第一条 AIC mode0 相邻且同号（AIC mode0）");

// ============================================================
// 1. 相位边界（全体 AIV 的 mode-0 barrier + 本核 MTE3→MTE2 drain）
// ============================================================
// set 挂 MTE3（排空本核写 = 本相位出口）、wait 挂 MTE2（挡住本核后续的 MTE2 读 = 下相位载入）。
// 不挂 PIPE_S：这里的 wait 后面**不紧接** CrossCoreSetFlag，不触发 docs/05 §2 的
// 「链式 wait→set 必须挂 PIPE_S」例外；用最窄 pipe 才符合「wait 用尽可能窄的 pipe」。
template <uint16_t FLAG>
__aicore__ inline void M15L_PhaseBoundaryAiv()
{
    AscendC::CrossCoreSetFlag<M15M::CC_MODE0, PIPE_MTE3>(FLAG);
    AscendC::CrossCoreWaitFlag<M15M::CC_MODE0, PIPE_MTE2>(FLAG);
}

// ============================================================
// 1b. M164：H2 → 相位 B 的 **跨类型 all-to-all** 握手（AIV→AIC）
// ============================================================
// 为什么必须有（M1，M152 的主候选机理）：H2 的产出 `pfHcBlk1` 由**全体 AIV** 在
// `M15H::HcPF::RelocatePass` 里以 MTE3 写 GM；相位 B 的 AIC router（`M15MP::MoePrefillChain::
// ProcessAic` 开头的 `router.RunTile`，其 x = `mp.xLayer` = `A.pfHcBlk1`）读**整块** x
// （全部 m 行）⇒ 依赖是 **all-AIV → 每个 AIC**（all-to-all）。而相位边界
// `M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>` 是**只含 AIV 的 mode-0 屏障**，AIC 不参与
// ⇒ AIC 可在 AIV 尚未写下 `pfHcBlk1` 时读到 host memset 的 0xCD 毒值。
// （§1 顶部「AIC 侧不需要相位边界同步」那句只对**段内自带**的 mode-2 交接成立：MoE 段自带的
//   AIV→AIC 交接从 S5（quantA → GEMM）起；router 的输入是**段输入**，不经过它。）
//
// 形态（人类口径 + `docs/05 §2` 硬规矩，逐字）：
//   「mode 0 仅同类型……跨类型 all-to-all 的标准组合：**AIVs→AIC（mode 2）→ 全体 AIC barrier
//     （mode 0）**」。故：
//     · AIV：写完 `pfHcBlk1`（相位边界已把本核 MTE3 写排空）→ set mode2；
//     · AIC：先 mode2 wait（每个 AIC 等它**配对的 2 个 AIV**：2 set ↔ 1 wait）→ 再全体 AIC
//       mode0 barrier（保证所有 AIC 都收到各自配对 AIV 的信号 ⇒ 全体 AIV 都写完）。
//   只用 `CrossCoreSet/WaitFlag`（"set cross core" 系列），**不用 SetFlag/WaitFlag**；核内生命周期
//   一律 BufferID（本函数不碰）。
//
// flagId（**权威定义在 `m15_layer_resources.h` §4f.1**；M168 收口 —— 本头只**引用**，不再定义，
// 保证「6/7 只有一处权威定义」）：
//   6 = mode2 握手；7 = 全体 AIC mode0 barrier。
//   相邻性（同核相邻同步点必须不同号）：
//     AIV mode2：H2 的 10/11（m15_hc_layer.h:506-507）→ **6** → MoE 的 0（m15_moe_prefill.h:1227）
//     AIC mode2：H2 的 10/11 → **6** → MoE 的 0（m15_moe_prefill.h:1293）
//     AIC mode0：H2 的 0/1/2/3（m15_hc_layer.h:530-547）→ **7** → MoE 的 8（m15_moe_prefill.h:1291）
//   空闲性：prefill 入口（KIND_GDN，H1/A/H2/B）里 AIV 占 {0-5,8-15}、AIC 占 {0-5,8-11}
//   ⇒ 两核同时空闲的只有 {6,7}；6/7 现仅被 **decode** GDN 用（m15_gdn_resources.h:156-157）。
//   ⚠ 若 attn-prefill 的 B3 接线（其核体 `m15_attn_fa_core.h` 的 mode-4 通道族已按
//     `m15_layer_resources.h` §4f.2 登记；`m15_attn_core.h` 变体的 AIC mode2 会用 4/6/7），到那时需重审。
// 上限 / 相邻性 / 「两核同时空闲={6,7}」的可编译期判据都在 `m15_layer_resources.h` §4f.1
// （`FlagIdLimit` + `PfH2BFreeOk()` 等）；本头保留下面这组与**相位边界 8/9** 的异号断言
// （相位边界是 `M15L` 的常量，§4f.1 不重复这一对）。
static_assert(FLAG_H2B_A2C_M2 < 16u && FLAG_H2B_AIC_BAR < 16u, "flagId 池是 0..15");
static_assert(FLAG_H2B_A2C_M2 != FLAG_H2B_AIC_BAR, "mode2 握手与 AIC barrier 不得同号");
static_assert(FLAG_H2B_A2C_M2 != FLAG_HC0_BOUND_AIV && FLAG_H2B_A2C_M2 != FLAG_HC1_BOUND_AIV &&
                  FLAG_H2B_A2C_M2 != FLAG_HC2_BOUND_AIV && FLAG_H2B_AIC_BAR != FLAG_HC2_BOUND_AIV &&
                  FLAG_H2B_AIC_BAR != FLAG_HC0_BOUND_AIV && FLAG_H2B_AIC_BAR != FLAG_HC1_BOUND_AIV,
              "H2→B 握手与相邻的相位边界（AIV mode0 8/9）不得同号（同核相邻）");

template <uint16_t M2_FLAG, uint16_t AIC_BAR_FLAG>
__aicore__ inline void M15L_H2ToBHandshake()
{
    if ASCEND_IS_AIV {
        AscendC::CrossCoreSetFlag<M15M::CC_MODE2, PIPE_MTE3>(M2_FLAG);
    }
    if ASCEND_IS_AIC {
        // wait 挂 PIPE_S：其后紧接 `CrossCoreSetFlag`（本次接入后紧接的是 AIC 的 MoE 段 S2 set），
        // 触发 docs/05 §2 的「链式 wait→set 必须挂 PIPE_S」例外（与 m15_moe_prefill.h:1292 同款）。
        AscendC::CrossCoreWaitFlag<M15M::CC_MODE2, PIPE_S>(M2_FLAG);
        AscendC::CrossCoreSetFlag<M15M::CC_MODE0, PIPE_MTE2>(AIC_BAR_FLAG);
        AscendC::CrossCoreWaitFlag<M15M::CC_MODE0, PIPE_S>(AIC_BAR_FLAG);
    }
}

// ============================================================
// 2. 相位 B（MoE 段）的指针组装 —— 融合路径与「MoE-only 对照」共用
// ============================================================
// moeWsBase：MoE 段 ws 的基址。融合路径传 ws + WS_MOE_OFF；对照路径由 host 直接传 MoE 段基址。
__aicore__ inline void M15L_FillMoePtrs(M15M::MoeLayerPtrs& mp, __gm__ uint8_t* moeWsBase, __gm__ uint8_t* xLayer,
                                        __gm__ uint8_t* resZero, const LayerArgs& A)
{
    mp.ws = moeWsBase;
    mp.xLayer = xLayer;      // 层链模式：融合路径 = 相位 A 的出口（两相位）或 hc(mlp) 的 BLK（四相位）
    mp.yLayer = A.yLayer;    // S10 出口 → 层间残差流缓冲（M40 接口改造，见 lift_moe_segment.py 规则 5）
    mp.resZero = resZero;
    mp.gamma1 = A.moeGamma1;
    mp.gamma2 = A.moeGamma2;
    mp.routerW = A.moeRouter;
    mp.sgateW = A.moeSgate;
    mp.wGu = A.moeWGu;
    mp.sGu = A.moeSGu;
    mp.wDn = A.moeWDn;
    mp.sDn = A.moeSDn;
    mp.wGuShd = A.moeWGuShd;
    mp.sGuShd = A.moeSGuShd;
    mp.wDnShd = A.moeWDnShd;
    mp.sDnShd = A.moeSDnShd;
    mp.m = A.m;
    mp.topk = A.topk;
    // bring-up 截断开关折成编译期常量（**段序语义与 m13 的全开档完全一致**，只是折掉死分支）
    mp.sliceMode = MOE_SLICE_MODE;
    mp.stageLimit = MOE_STAGE_LIMIT;
    mp.subLimit = MOE_SUB_LIMIT;
}

// ============================================================
// 2b. hc 段边界的指针组装（M58）
// ============================================================
// which = 0 → 边界 #1（attn_hc）；1 → 边界 #2（mlp_hc）。
//
// **边界 #2 的三个输入源是层内的 handoff**（不额外物化张量）：
//   hIn      = hcWs0 + WS_HCP            （边界 #1 的 H'；mode 0 时 H' ≡ H，故取 A.hcH）
//   bo       = A.hcAttnOut               （子层段的出口）
//   ij       = hcWs0 + WS_OH + 320*2B，行距 OH_W（边界 #1 的 `OH[:,320:324)` = injection logits）
// 这三处正是 README「融合后层内 handoff 不必经 host」的落点（m20 的独立 kernel 形态里，中间
// 那一次 IJ/BO 提取由 host 物化成各自的 GM 平面）。
//
// **`hcPleBreak` 那一层（层 1）是唯一的例外**：它的 H1 拆成 combine-only（写 `HC_WS0+WS_HCP`）
// 与 mix-only 两段，`MODE_MIX` 只表示「mix 段不做 combine」，**不再蕴含 H' ≡ 层输入**（mix 段的
// 输入是 combine 物化出来的那块平面）⇒ 边界 #2 的 hIn 仍必须取 `HC_WS0+WS_HCP`。
__aicore__ inline void M15L_FillHcPtrs(M15H::HcPtrs& p, uint32_t which, const LayerArgs& A)
{
    const bool second = (which == 1u);
    p.ws = second ? A.hcWs1 : A.hcWs0;
    const bool hcpFromWs = (A.hcPleBreak != 0u) || (A.hcAttnMode != M15H::MODE_MIX);
    p.hIn = second ? (hcpFromWs ? (A.hcWs0 + M15H::WS_HCP) : A.hcH) : A.hcH;
    p.bo = second ? A.hcAttnOut : A.hcBo;
    p.ij = second ? (A.hcWs0 + M15H::WS_OH + M15H::OH_INJ * 2u) : A.hcIj;
    p.ijStride = second ? M15H::OH_W : A.hcIjStride;
    p.wDown = second ? A.hcMlpDown : A.hcAttnDown;
    p.wInj = second ? A.hcMlpInj : A.hcAttnInj;
    p.wUp = second ? A.hcMlpUp : A.hcAttnUp;
    p.hcNorm = second ? A.hcMlpNorm : A.hcAttnNorm;
    p.m = A.m;
    p.mode = second ? A.hcMlpMode : A.hcAttnMode;
    p.stageLimit = HC_STAGE_LIMIT;
}

// ============================================================
// 3b. PLE 段的**挂载点**（M65 交付；PLE 本体未实现）
// ============================================================
// vLLM 的 `Qwen4ExpDecoderLayer.forward`（`V-N:model.py:290-301`）在 `layer_idx + 1 in ple_layer_ids`
// 的层上，把 PLE **直接加到 10240 宽的多流残差态**上，且它**前面**要先物化 pending combine：
//
//     if prev_block_output is not None:                       # 实际总是成立（PLE 在层 1，不是层 0）
//         hidden = attn_hc.combine(hidden, prev_block_output, prev_injection)
//         prev_block_output = prev_injection = None
//     hidden = ple(hidden, input_ids, query_start_loc, ngram_context)
//     ... attn_hc.combine_and_mix(hidden, None, None)         # 此后才是 mix（无 combine）
//
// ⇒ 这个边界上 **combine 与 mix 必须分成两段**，中间夹 PLE。本函数就是那个「中间」的挂载点：
// 它的输入 = `hc0`（`MODE_COMBINE_ONLY`）物化在 `HC_WS0+WS_HCP` 的 bf16 `H'`；输出 = `hcMix`
// （`MODE_MIX`）将要读的那块平面。
//
// **M100 起这里不再是空操作**：`LayerArgs::pleW != nullptr` 时，PLE 本体（① n-gram id →
// ② 表行 gather → ③ key/value 投影 → ④ 门控 → ⑤ 膨胀卷积+残差+状态）就在这两条相位边界
// **之间**跑，直接读写 10240 宽的多流态（`hcWs0 + WS_HCP`，就地写回）。
// device 实现来自 `m15_ple_wire.h` 的 `M85P::*` —— 它是 `m15_ple.asc` 的 device 段的
// **机械抽取生成物**（生成器 + `--check` 见 `evidence/ple_wire/lift_ple_device_segment.py`），
// 故 M85 的 13 条判据与 M92 的 `Hd.row_fail`/`Hd.miss` 设备侧计数在**这条路径上**同样成立。
//
// **`pleW == nullptr` 时逐字退化为 M65 的空操作**（两条边界照旧、`PipeBarrier` 照旧）⇒
// 既有的 `runs=all` 读数（2068 + 290 / 0 FAIL）不受影响（见 evidence/ple_wire/）。
// ---- ① n-gram id（int64 位运算 + 取模 + 偏移）----
// 落点 = scratch slab 的 S_IDS（[T,NG] int64），② 直接从这里读（同一段内，跨核可见性由
// **调用点夹在 ①② 之间的** `FLAG_PLE_IN_BOUND_AIV`（全体 AIV mode-0）保证）。
__aicore__ inline void M15L_PleIds(const LayerArgs& A)
{
    using namespace M85P;
    if ((A.pleStageMask & PLEW::ST_IDS) != 0u) {
        AscendC::GlobalTensor<int32_t> idsG;
        AscendC::GlobalTensor<int32_t> qslG;
        AscendC::GlobalTensor<int32_t> ctxG;
        AscendC::GlobalTensor<int64_t> mG;
        AscendC::GlobalTensor<int64_t> szG;
        AscendC::GlobalTensor<int64_t> ofG;
        AscendC::GlobalTensor<int64_t> outG;
        AscendC::GlobalTensor<uint32_t> failG;
        idsG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(A.pleIds), A.pleTok);
        qslG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(A.pleQsl), A.pleNReq + 1u);
        ctxG.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(A.pleCtx), A.pleNReq * NC);
        mG.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(A.pleM), 3);
        szG.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(A.pleSz), NG);
        ofG.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(A.pleOf), NG);
        outG.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(A.pleScr + PLEW::S_IDS),
                             static_cast<uint64_t>(A.pleTok) * NG);
        failG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(A.pleScr + PLEW::S_FAILI), FAIL_SLOTS);
        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum() * 2u;
        uint32_t bad = 0;
        for (uint32_t t = bid; t < A.pleTok; t += nblk) {
            IdsOneToken(t, idsG, qslG, ctxG, mG, szG, ofG, outG, A.pleNReq, A.pleNegMask, &bad);
        }
        if (bad > 0u) {
            failG.SetValue(bid, bad);
        }
    }
}

// ---- ②③④⑤：表行 gather → kv 投影 → 门控 → 膨胀卷积 + 残差 + 状态移位 ----
__aicore__ inline void M15L_PleBody(const LayerArgs& A)
{
    using namespace M85P;
    if ((A.pleStageMask & PLEW::ST_BODY) != 0u) {
        BodyGm G;
        const uint64_t nTok = A.pleTok;
        G.ids.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(A.pleScr + PLEW::S_IDS), nTok * NG);
        G.table.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleTable),
                                static_cast<uint64_t>(A.pleTableRows) * HDIM);
        G.wcat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_CAT),
                               static_cast<uint64_t>(KVW) * HID);
        G.wtap.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_TAP),
                               static_cast<uint64_t>(4u) * HYPER);
        G.nk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_NK), HYPER);
        G.nq.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_NQ), HYPER);
        G.ncw.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_NCW), HYPER);
        // `hid` 与 `out` **同一块**（O1 就地写回）：PLE 把它当 query（④）与外层残差（⑤）
        G.hid.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.hcWs0 + M15H::WS_HCP), nTok * HYPER);
        G.out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.hcWs0 + M15H::WS_HCP), nTok * HYPER);
        // M191：状态平面按**活槽数**定尺（`M15L_ACT_STATE_SLOTS(A.pleNReq, A.pleStateSlots)`；
        // host 搬运用同一个表达式）。`A.pleStateSlots` 是 host 的**分配上界**（`H_PleArgsOf`
        // 填 `PLEW::T_MAX`），不是活槽数 ⇒ 使 SetGlobalBuffer 的界与真实状态一致、
        // 跨 step 搬运只覆盖活槽（搬运见 `m15_layer_loop.asc::H_PleCarryState`）。
        const uint32_t actStSlots = M15L_ACT_STATE_SLOTS(A.pleNReq, A.pleStateSlots);
        G.stIn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_STIN),
                               static_cast<uint64_t>(actStSlots) * STLEN * HYPER);
        G.stOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_STOUT),
                                static_cast<uint64_t>(actStSlots) * STLEN * HYPER);
        G.sidx.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(A.pleScr + PLEW::S_SIDX), nTok);
        G.emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_EMB), nTok * HID);
        G.kv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_KV), nTok * KVW);
        G.gated.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_GATED),
                                nTok * HYPER);
        G.normed.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_NORMED),
                                 nTok * HYPER);
        G.fail.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(A.pleScr + PLEW::S_FAILB), FAIL_SLOTS);
        G.exp64.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_EXP64),
                                nTok * NG * VL);
        G.hmBad.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A.pleScr + PLEW::S_HMBAD),
                                static_cast<uint64_t>(FAIL_SLOTS) * 8u);
        G.winBase = A.pleWinBase;
        G.winRows = A.pleWinRows;

        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t nblk = AscendC::GetBlockNum() * 2u;
        uint32_t bad = 0;
        const uint32_t mask = A.pleStageMask & PLEW::ST_BODY;
        // **M124：②→③→④ 是跨核型交接**，AIV 侧原来那两条「全体 AIV 的自封 mode-0 barrier」
        //   （`BarrierAiv<PIPE_MTE2, FLAG_B2/FLAG_B3>`）**已撤掉** —— ③ 现在是 AIC 上的 mmad，
        //   交接按「AIVs→AIC（mode 2）→ 全体 AIC 的 mode-0 对齐 → AIC→AIVs（mode 2）」，
        //   由 `PleGemv` 的 AIV 臂（set `FLAG_B2` / wait `FLAG_B3`，都挂在配对 mode-2 上）
        //   与 AIC 臂（本文的 `M15L_PleAic`，见 `M15L_FusedBody` 的 AIC 分支）分担。
        //   `FLAG_B4`（④→⑤）仍是原来的全体 AIV mode-0 barrier，一字未动。
        if ((mask & 1u) != 0u) {
            PleGather(bid, nblk, G, A.pleTok, A.pleNegMask);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if ((mask & 2u) != 0u) {
            PleGemv(bid, nblk, G, A.pleTok, A.pleNegMask);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        if ((mask & 4u) != 0u) {
            const uint32_t nItems = A.pleTok * HC;
            for (uint32_t i = bid; i < nItems; i += nblk) {
                PleGateItem(i / HC, i % HC, G, &bad, A.pleNegMask);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            M15H::BarrierAiv<PIPE_MTE2, FLAG_B4>();
        }
        if ((mask & 8u) != 0u) {
            for (uint32_t t = 0; t < A.pleTok; ++t) {
                PleConvItem(t, G, &bad, A.pleNegMask);
            }
        }
        if (bad > 0u) {
            G.fail.SetValue(bid, bad);
        }
    }
}

// ---- M124：③（**AIC 侧**）的挂载点 = AIC 臂的 `PleGemv` ----
// 为什么必须有它：③ 改走 cube 后，mmad 只能由 AIC 发起，而 AIV 侧的 `M15L_PleBody` 只
//   set `FLAG_B2` / wait `FLAG_B3`（配对 mode-2）⇒ **没有这一段，层路径的 kv 就没有生产者**
//   （AIV 会等一条永远不来的 flag ⇒ 挂死；这是"要么接全、要么别接"的结构）。
// 挂载位置（`M15L_FusedBody` 的 AIC 分支）：hc(H1a) 之后、hc(H1b) 之前 —— 与 AIV 侧
//   （`FLAG_PLE_IN_BOUND_AIV` 与 `FLAG_PLE_OUT_BOUND_AIV` 之间）逐字对齐。死锁自由性论证：
//   AIV 的 ② 在 PLE_IN 之后、不依赖 AIC；AIC 的 ③ 只等 AIV 的 ②；AIV 的 ④⑤ 只等 AIC 的 ③；
//   AIC 的 hcMix 只等 AIV 的 H1b（在 PLE_OUT 之后）⇒ 依赖序单调，无环。
__aicore__ inline void M15L_PleAic(const LayerArgs& A)
{
    using namespace M85P;
    if ((A.pleStageMask & PLEW::ST_BODY & 2u) != 0u) {
        BodyGm G;
        const uint64_t nTok = A.pleTok;
        // AIC 只吃 ③ 的三个平面（A = emb、B = wcat、C = kv）—— 取址与 `M15L_PleBody` 的 AIV 侧同源
        G.emb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_EMB), nTok * HID);
        G.wcat.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleW + PLEW::W_CAT),
                               static_cast<uint64_t>(KVW) * HID);
        G.kv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.pleScr + PLEW::S_KV), nTok * KVW);
        // `PleGemv` 的 AIC 臂自己按 `GetBlockIdx()/GetBlockNum()` 取 N 条带（形参在 AIC 侧不用）
        PleGemv(AscendC::GetBlockIdx(), AscendC::GetBlockNum(), G, A.pleTok, A.pleNegMask);
    }
}

// ============================================================
// 3d. M110（Wave A）交挂载点 → M132（Wave C）接段体：**prefill 的相位结构**
// ============================================================
// 挂载点的形状（B1–B5 按它接；M132 已按它接 B1/B4/B5）：
//
//   M15L_PrefillBody<KIND>(A)
//     ├─ `pfWired != 0`（段体）：相位 H1（B5）→ 边界 → 相位 A（B1 / B2+B3）→ 边界 →
//     │    相位 H2（B5）→ 边界 → 相位 B（B4）。三个边界号见 resources §4（8 / 9 / 8）。
//     │    · **B1（KIND_GDN 支）已接**：`M15L_PrefillPhaseA` 的 `pfWired != 0` 支里调
//     │      `M15GP::GdnPrefillAivEntry/AicEntry`（编译期门 `M15_GDN_PREFILL_WIRE`，见 §3d-0）。
//     │    · **B2（attn 前端）已接（M197）**：`M15L_PrefillPhaseA<KIND_ATTN>` 逐 chunk 调
//     │      `M15PF::AttnPrefillPhaseA`（补丁 = m24 README §5.8）；**B3（稠密 core）仍未接** ⇒
//     │      整条 attention 臂仍**响亮失败**（host 侧按 B3 缺席判红，不是"默默部分产出"）。
//     │    · **B4（相位 B）已接**：`M15L_PrefillPhaseB` 调 `M15MP::MoePrefillChain`（单块）。
//     │    · **B5（相位 H1/H2）已接**：`M15L_HcPrefillBoundary` 调 `M15H::HcPF::MakePlan/Body`。
//     ├─ `pfWired == 0`（缺省）：**结构性占位** —— 逐行分派 + 逐行 x→y 搬运，
//     │    **数学未实现**（写死在函数名与判据上：host 侧判「逐行标记」+「整面逐字节」）。
//     └─ **不新造任何 flagId**：段体用的号都已在 `m15_layer_resources.h` §4d/§4 登记（有复用见证）。
//
// **段体实参的来源**：B1/B4/B5 的权重与激活平面（`LayerArgs` 的 13 个新字段）目前由 host 传 nil
// ⇒ `wire=1` 的实跑判据归 **Wave D**（m=4097 的真权重组装）；本轮 B1/B4/B5 的交付形态是
// 「编译进融合 TU、`wire=0` 时路径一字未变、编译期门与峰值槽收口」。
//
// **为什么行分派网格是本 mission 的关键交付**：老的 `m15_attn_passthrough_body`（M103-1.2 点名）
// 是 m=1 定尺 —— 每 AIV 固定 16 个 32B 块、56 个 AIV 只覆盖 896 块 = **5.6 行**，m ≥ 6 起
// 静默丢行且不报错。prefill 的挂载点必须是 `for (row = bid; row < m; row += nAiv)` 的**行网格**。
//
// **N1（`subOut` 的生产者）的归属（M103-2.1 点名归 Wave A；塔裁 2026-09-27 改为 Wave C）**：
//   N1 = `A.apW != nullptr` 时子层出口没有生产者（`M15L_FusedBody` 走 attention 臂后不再写
//   `subOut`）⇒ 下游 MoE 照跑、**不报错、只数值错**（M97 `evidence/attn_wire/README.md` §5 第 8 项）。
//   它的修法要有 **attention 核心 + `o_proj`**，而 M101 那份核心还没合入 ⇒ M132 **仍不实现**，只登记。
//   M132 已把相位 B（B4）接上，但 attention 子层段（B2/B3）没接 ⇒ 在 KIND_ATTN 的 prefill 入口上
//   相位 A 不产出、下游 MoE 即便跑也读毒值 —— 这条静默错**由"attention 臂缺席"挡住**（不构成活的
//   错误面）；**B2/B3 接进来的同一条 commit 必须先给 `subOut` 生产者**，否则该静默错复活。
//   详见 `evidence/prefill_contract/README.md` §5 的第一行。
//
// ============================================================
// 3d-0. M132（Wave C）：B1（GDN prefill）接线开关 —— **唯一权威拼法**
// ============================================================
// **哪个是权威**：`M15_GDN_PREFILL_WIRE` —— 它是 B1 段自己的融合规范（`m23_gdn_prefill/README.md`
// §3(g)/§3(h)）里的名字，host 侧契约头 `m15_gdn_prefill_host.h` 也照它统一（原先那处写成
// `M15_PREFILL_GDN_WIRE`，在本 mission 一并改掉）。`m15_layer_loop/` 下的宏定义与 `#if` 用法里
// 只剩这一种拼法（本注释提到旧拼法是为了记下这次统一）。
// 默认 1：B1 的段体（`m15_gdn_prefill.h`）已随 main 合入 ⇒ 编译进融合 TU；**运行时**仍由
// `LayerArgs::pfWired`（host 的 `M15_PREFILL_WIRE`，缺省 0）门控 ⇒ decode 零回归。
// （注意区分：`M15_PREFILL_WIRE` 是 host 的运行期总开关，`M15_GDN_PREFILL_WIRE` 是"B1 段是否
//  编译进本 TU"的编译期门 —— 两者不是同一个东西。）
#ifndef M15_GDN_PREFILL_WIRE
#define M15_GDN_PREFILL_WIRE 1
#endif
namespace PF {
constexpr uint32_t ROW_BYTES = HIDDEN * 2u;                 // 一行残差流 = bf16[HIDDEN] = 5120 B
constexpr uint32_t UB_ROW = M15G::UB_SEG;                   // 相位 A 的 UB 窗（与 decode 段窗同址）
constexpr AscendC::MutexID BUF_ROW = PF_BUF_AIV_ROW0;       // 行搬运的 BufferID（登记见 §5）
}  // namespace PF

// ---- M136（Wave D）：GDN prefill 的 q/k 尾部补齐行数（host 与 device 共用这一个数）----
// AIC 的 `Nd2Nz` 每个 chunk 固读 `GP_BT = 64` 行，最后一个 chunk 会读到 m 之后的 pad 行 ⇒
// 宿主必须按 `qkStride = m + 本常量` 给 q/k 定尺，并把 [m, m+本常量) 行**置零**（否则 mmad 读垃圾）。
// 权威：`m15_gdn_prefill.h` 的 BT=64（`M15GP::GdnPrefillAicEntry` 的 Nd2Nz 行数）与 m23 README §3(f)。
// prefill 入口的 `M15L_PrefillPhaseA<KIND_GDN>` 把它加进 `gp.qkStride`（见下）。
constexpr uint32_t PF_GDN_QK_PAD_ROWS = 64u;

// ---- M136（Wave D）：prefill 相位的 bring-up 截断位（`LayerArgs::pfStageMask`）----
// 0 = 全开（H1 → A → H2 → B）。宿主在真指针未齐全时只开能跑的相位（见 `.asc` 的 H_PfGdnWired）：
// KIND_GDN 的整层四相位需要 hc 的层界/权重字段，而 prefill 入口的实参表**没有**这些字段
// （`M15L_LAYER_PREFILL_ARGS_DECL` 只有两相位基础 30 + 13 个 M110 prefill + 13 个 M132 段体项）
// ⇒ 本轮 GDN 验收只开相位 A（`pfStageMask = PF_STAGE_A`）。这是一条**如实的能力边界**，不是静默跳过。
constexpr uint32_t PF_STAGE_H1 = 1u;
constexpr uint32_t PF_STAGE_A = 2u;
constexpr uint32_t PF_STAGE_H2 = 4u;
constexpr uint32_t PF_STAGE_B = 8u;
constexpr uint32_t PF_STAGE_ALL = 15u;

// 一行 x→y（GM→UB→GM；32B 块粒度 —— 与 `m15_attn_passthrough_body` 同一套 DataCopy 用法）
__aicore__ inline void M15L_PrefillRowCopy(const LayerArgs& A, uint32_t row)
{
    if ASCEND_IS_AIV {
        AscendC::GlobalTensor<bfloat16_t> xG;
        AscendC::GlobalTensor<bfloat16_t> yG;
        xG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.xLayer),
                           static_cast<uint64_t>(A.m) * HIDDEN);
        yG.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A.yLayer),
                           static_cast<uint64_t>(A.m) * HIDDEN);
        AscendC::LocalTensor<bfloat16_t> buf(TPosition::VECCALC, PF::UB_ROW, HIDDEN);
        const uint64_t off = static_cast<uint64_t>(row) * HIDDEN;

        BufAcquire<PIPE_MTE2>(PF::BUF_ROW);
        DataCopy(buf, xG[off], Block1(PF::ROW_BYTES));
        BufRelease<PIPE_MTE2>(PF::BUF_ROW);     // drain：MTE2 排空后 token 才归 MTE3

        BufAcquire<PIPE_MTE3>(PF::BUF_ROW);
        DataCopy(yG[off], buf, Block1(PF::ROW_BYTES));
        BufRelease<PIPE_MTE3>(PF::BUF_ROW);
    }
}

// 相位 A：**行分派网格**（`row = bid; row < m; row += nAiv`，与 hc 段的 item 网格同款）。
// 段体落地点：`pfWired != 0` 的那一支 —— 目前是显式的"未实现"，不产出任何东西。
template <uint32_t KIND>
__aicore__ inline void M15L_PrefillPhaseA(const LayerArgs& A)
{
    const uint32_t nAiv = AscendC::GetBlockNum() * 2u;
    const uint32_t bid = AscendC::GetBlockIdx();
    uint32_t mRows = A.m;
    if (A.pfMutant != 0u) {
        // 负向对照（只在验证档用）：把行数截到 decode 包络 M_MAX ⇒ 该行的标记必须保持毒值。
        // 它存在的意义 = 证明"逐行标记"判据**不是空洞的**（docs/17 §4 的非空洞性纪律）。
        mRows = (mRows < M15G::M_MAX) ? mRows : M15G::M_MAX;
    }
    if (A.pfWired != 0u) {
        // ---- 段体挂载点（B1 已接；B2 已接（M197）；B3 仍未接）----
        // 段体签名只吃"指针 + 标量"（Wave B 共同契约第 1 条）⇒ 这里从 LayerArgs 组装段参。
#if M15_GDN_PREFILL_WIRE
        if constexpr (KIND == KIND_GDN) {
            // B1（M115）：机械照抄 m23 README §3(h) 的挂载点补丁（字段名以本仓 LayerArgs 实际为准）。
            M15GP::GdnPrefillArgs gp{};
            gp.q = A.wsQ;
            gp.k = A.wsK;
            gp.v = A.wsV;
            gp.g = A.wsG;
            gp.beta = A.wsBeta;
            gp.h0 = A.ssmState;          // 初始状态（复用 M110 之前的 ssmState）
            gp.out = A.wsO;
            gp.ht = A.ssmState;          // in-place：出口状态写回同一平面
            gp.scratch = A.wsGdnPrefillScratch;
            gp.m = A.m;
            gp.heads = M15G::HEADS;
            // align8(m)：不调 `M15G::WS_AlignUp` —— 它未标 `__aicore__`，设备侧会留一个外部符号
            // （链接期 undefined symbol 实测）。等价算术留在本行。
            gp.tp = (A.m + 7u) / 8u * 8u;
            // q/k 每 head 的行 stride：M136（Wave D）起 = m + PF_GDN_QK_PAD_ROWS —— AIC 的 Nd2Nz
            // 每 chunk 固读 64 行，尾部 chunk 会越过 m 读到 pad 行；宿主按同一 stride 定尺并把
            // pad 行置零（`PF_GDN_QK_PAD_ROWS` 的顶上注释是契约出处）。
            gp.qkStride = A.m + PF_GDN_QK_PAD_ROWS;
            gp.scale = 0.08838834764831845f;   // 1/√128
            if ASCEND_IS_AIV {
                M15GP::GdnPrefillAivEntry(gp);
            }
            if ASCEND_IS_AIC {
                M15GP::GdnPrefillAicEntry(gp);
            }
        }
#endif
        // KIND_ATTN：**B2（attention 前端）已挂载（M197）**；B3（稠密 causal core）仍未接。
        // ⇒ 本支产出 B2 的 `y0` / `out` / cache，但整条 attention 臂仍**响亮失败**
        //   （host 侧按 B3 缺席判红：`.asc` 的 `Pf.attn` 支 + `H_RunPrefill` 的 wire+attn 硬失败）。
        // 挂载补丁逐行照抄 `m24_attn_prefill/README.md` §5.8，**唯一偏差**：KV/cache 的层偏移
        // 用算术式而非 `M15Kv::KvLayerOffset` —— 后者未标 `__aicore__`，device 侧调用会报
        // "call to __host__ function from __aicore__ function"（`m15_attn_kv.h:242` 已写明；
        // GDN 挂载点的 `align8` 也是同一处规避）。数值与 `KvLayerOffset` 同源（同一 stride 常量）。
        if constexpr (KIND == KIND_ATTN) {
            // ⚠ **逐 chunk 调用**：段体一次只跑**一个** chunk（`pfa.nChunks = 1`）；
            //    `nChunks > 1` 在段体里是 `Trap` 硬拦（`m15_attn_prefill.h` 的 `AttnPrefillPhaseA`）。
            const uint32_t k = A.pfLayerK;             // 0..11（= M15Loop::AttnSlot）
            for (uint32_t off = 0u; off < A.m; off += M15PF::PF_ROWS) {
                const uint32_t rows = ((A.m - off) < M15PF::PF_ROWS) ? (A.m - off) : M15PF::PF_ROWS;
                M15PF::PfChunkArgs pfa{};
                pfa.x = A.xLayer;                      // **平面基址**（[m, HIDDEN] bf16；尾部留 1 行余量）
                pfa.w = A.apW;                         // attention 的 8 role 权重平面
                pfa.cs = A.apCs;                       // cos/sin 表
                pfa.posTbl = A.pfPos;                  // 可空（按绝对行号取表）
                pfa.posBase = A.pfPosBase;             // 两者都给时以表为准
                pfa.kv = A.pfKv + static_cast<uint64_t>(k) * M15Kv::KV_LAYER_STRIDE;
                pfa.comp = A.pfComp + static_cast<uint64_t>(k) * M15Kv::COMP_LAYER_STRIDE;
                pfa.ring = A.pfRing + static_cast<uint64_t>(k) * M15Kv::RING_LAYER_STRIDE;
                pfa.pack = A.pfPack;
                pfa.packSeed = A.pfPack;               // 本段只抄 seed 行（生产者是 B3）
                pfa.y0 = A.pfY0;                       // 3,571,712 B（README §5.2）
                pfa.out = A.pfOut;                     // 3,571,712 B
                pfa.in = A.pfIn;                       //   299,264 B
                pfa.pooled = A.pfPooled;               //     8,192 B
                pfa.flag = A.pfFlag;                   //        32 B
                pfa.rowStart = off;                    // **本 chunk 的绝对首行号**（x 的行偏移）
                pfa.seqRows = rows;                    // 本次要处理的行数 = 本 chunk 的行数
                pfa.chunkRows = rows;                  // 逐 chunk：chunkRows == 本 chunk 行数
                pfa.nChunks = 1u;                      // 一次调用只跑一个 chunk（>1 会被段体 Trap）
                // 同上面的偏移：`PfUnpackMut/AcMode` 是未标 `__aicore__` 的 constexpr（host 侧用），
                // device 运行时调用会留外部符号 ⇒ 在此展开等价算术（README §5.2 的位约定）。
                pfa.mut = A.pfMutant & M15PF::PF_MUT_MASK;
                pfa.acMode = (A.pfMutant >> M15PF::PF_AC_MODE_SHIFT) & 0xFu;
                M15PF::AttnPrefillPhaseA(pfa);
            }
        }
        return;
    }
    if ASCEND_IS_AIV {
        for (uint32_t row = bid; row < mRows; row += nAiv) {
            M15L_PrefillRowCopy(A, row);
        }
    }
}

// 相位 B（B4，M118）：MoE prefill 段。挂载点 = 相位 A 之后、贴着 `FLAG_HC2_BOUND_AIV`
// （m26 README §4 补丁 A 的次序要求；B5 未接线时它成为相位 A 之后的第一段，但仍保留这条边界）。
// 段体只吃"指针 + 标量"。**单块**：`MoePrefillChain` 现按单块编排（m ≤ MT）；m > MT 的按块循环是
// Wave D 的 host 组装项（m26 README §5.4），本函数不代它做主。
template <uint32_t KIND>
__aicore__ inline void M15L_PrefillPhaseB(const LayerArgs& A)
{
    if (A.pfWired == 0u) {
        return;   // 默认关：decode 的 runs=all 零回归（M103-2.7 的硬要求）
    }
    // M140（数据流）：四相位形态下相位 B 的**层输入 = hc(mlp) 的 BLK 平铺面**（相位 H2 的产出）。
    // decode 的四相位路也是这么接的（`M15L_FusedBody`：`M15L_FillMoePtrs(mp, ws, A.hcWs1+WS_BLK, …)`）。
    // prefill 的 H2 产出落在 `pfHcBlk1`（`[m, HID]` bf16 平铺面，行距 `ROW_HID = HID*2`）⇒ 类型/形状
    // 与 MoE 的 `xLayer`（`[MT, HIDDEN]` bf16）一致，可直接别名。
    // **H2 未开时退回 `A.xLayer`**（M132 原行为，不改既有读数）：`pfHcBlk1` 在那种配置下可能是毒值。
    const uint32_t stP = (A.pfStageMask == 0u) ? PF_STAGE_ALL : A.pfStageMask;
    const bool h2On = (stP & PF_STAGE_H2) != 0u;
    __gm__ uint8_t* const moeIn = (h2On && A.pfHcBlk1 != nullptr) ? A.pfHcBlk1 : A.xLayer;
    // M164（M1 修法）：H2→B 的相位边界只同步 AIV，AIC 的 router 会读 `moeIn`（H2 的产出，
    // 全体 AIV 写、每个 AIC 读 = 跨类型 all-to-all）⇒ 必须先握手再让 AIC 进 `ProcessAic`。
    // 放在块循环**之前**：整个 `pfHcBlk1` 在上面 `M15L_HcPrefillBoundary(1u, A)` 里已写全，
    // 这里只需一次（H2 未开时 `moeIn` 不依赖 AIV，握手仍按两核配平照做、无副作用）。
    M15L_H2ToBHandshake<FLAG_H2B_A2C_M2, FLAG_H2B_AIC_BAR>();
    // M140：`MoePrefillChain` 按 `MT = 64` 行单块编排 ⇒ m > MT 时在这里按块循环（m26 README §5.4 的
    // 「m > MT 的按块循环」由本挂载点承担）。每块行指针按 `MT*HIDDEN*2` 推进、`m` = 本块行数；
    // 块间一条 `PipeBarrier<PIPE_ALL>` 排空本核两 pipe（段体自己的核间 set/wait 在一个块内配平）。
    const uint32_t mt = M15MP::MT;
    const uint32_t nTiles = (A.m + mt - 1u) / mt;
    for (uint32_t t = 0; t < nTiles; ++t) {
        const uint32_t row0 = t * mt;
        const uint32_t rest = A.m - row0;
        const uint32_t rows = (rest < mt) ? rest : mt;
        M15MP::MoePrefillPtrs mp = {};
        // M136（Wave D）：MoE prefill 段的 ws 从**融合 ws 的 MoE 区**起算（与 decode 路 `:929` 同源）。
        // prefill 入口的 `A.ws` 语义 = "融合 ws 基址（GDN 段 @0、MoE 段 @WS_MOE_OFF）" ⇒ 这里必须加偏移。
        // **宿主契约**：给 prefill 入口的 `ws` 传一个 ≥ `WS_MOE_OFF + M15PFH::WsBytes()` 的缓冲的基址
        // （`.asc` 的 H_PfGdnWired 用 `pfMoeWsDev` 落实；decode 的 `WS_MOE_BYTES` 装不下 prefill 的
        // `PWS_BYTES`，故不能复用 `wsFusedDev`）。
        mp.ws = A.ws + WS_MOE_OFF;
        mp.xLayer = moeIn + static_cast<uint64_t>(row0) * HIDDEN * 2u;
        mp.yLayer = A.yLayer + static_cast<uint64_t>(row0) * HIDDEN * 2u;
        mp.resZero = A.resZero;          // 全零平面，逐块复用（无行偏移语义）
        mp.gamma1 = A.moeGamma1;
        mp.gamma2 = A.moeGamma2;
        mp.routerWpad = A.pfMoeRouterWPad;   // B4 唯一新增字段（pad 后的 router 平面）
        mp.wGu = A.moeWGu;
        mp.sGu = A.moeSGu;
        mp.wDn = A.moeWDn;
        mp.sDn = A.moeSDn;
        mp.wGuShd = A.moeWGuShd;
    mp.sGuShd = A.moeSGuShd;
    mp.wDnShd = A.moeWDnShd;
    mp.sDnShd = A.moeSDnShd;
    mp.m = rows;
    mp.topk = A.topk;
    mp.stageLimit = 8u;   // 8 = 全链（bring-up 截断开关；契约档全开）
    M15MP::MoePrefillChain moe;
    moe.Init(mp);
    if ASCEND_IS_AIV {
        moe.ProcessAiv();
    }
    if ASCEND_IS_AIC {
        moe.ProcessAic();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    }
}

// ============================================================
// 3d-B5. M132（Wave C）：hc prefill 的单边界入口（B5，M119）
// ============================================================
// 照 m27 README §6 补丁 1：段体签名只吃"指针 + 标量" ⇒ 这里从 LayerArgs 组装 Plan。两个边界
// 共用它（which = 0/1）。本 mission 只新增 3 个指针（arena 复用 hcWs*、BLK×2、ij handoff）；
// 证据面 injw/rstd 不新增字段 ⇒ 按 README §5(b) 的"最小新增"传 nullptr（门限的唯一权威 =
// RelocatePass 内部的 NeedReloc，省略证据面只意味着"那几块平铺面不写"，不影响 ij handoff 面）。
// 调用方（Body）负责两侧都进入：核间 mode-2 交接要求 AIC 与配对的 2 个 AIV 按同一块序前进。
__aicore__ inline void M15L_HcPrefillBoundary(uint32_t which, const LayerArgs& A)
{
    const bool second = (which != 0u);
    M15H::HcPF::Plan p = M15H::HcPF::MakePlan(
        second ? A.pfHcArena0 : A.hcH,                       // hIn：边界 #2 取边界 #1 的 H' 平铺面（零拷贝）
        second ? A.hcAttnOut : A.hcBo,                       // bo：边界 #2 取子层段出口
        second ? A.pfHcIj0 : A.hcIj,                         // ij：边界 #2 取边界 #1 重定位出的平铺 ij 面
        second ? A.hcMlpDown : A.hcAttnDown, second ? A.hcMlpInj : A.hcAttnInj,
        second ? A.hcMlpUp : A.hcAttnUp, second ? A.hcMlpNorm : A.hcAttnNorm,
        second ? A.pfHcArena1 : A.pfHcArena0,                // arena
        second ? A.pfHcBlk1 : A.pfHcBlk0,                    // blk（平铺；nullptr ⇒ 留在 arena 阻塞布局）
        nullptr, nullptr,                                    // injw / rstd 证据面（本 mission 不新增字段）
        second ? nullptr : A.pfHcIj0,                        // ijFlat（只有边界 #1 产出）
        M15H::HcPF::FlatHInTileStride(), M15H::HcPF::FlatBoTileStride(),
        M15H::HcPF::FlatIjTileStride(M15H::IJ_STRIDE), M15H::IJ_STRIDE, A.m,
        second ? A.hcMlpMode : A.hcAttnMode);
    M15H::HcPF::Body(p);
}

// 占位主体：相位 A 的挂载点 + 其余相位的**显式未接**说明。
template <uint32_t KIND>
__aicore__ inline void M15L_PrefillBody(const LayerArgs& A)
{
    if (A.pfWired != 0u) {
        // ---- 段体接线（M132）：H1（B5）→ 相位 A（B1）→ H2（B5）→ 相位 B（B4）----
        // 边界号取 `m15_layer_resources.h` §4 的登记：H1→A = FLAG_HC0_BOUND_AIV(8)、
        // A→H2 = FLAG_HC1_BOUND_AIV(9)、H2→B = FLAG_HC2_BOUND_AIV(8)。
        // （m27 README §6 补丁 2 把 A→H2 也写成 FLAG_HC2_BOUND_AIV；resources §4 的登记表是
        //  flagId 的唯一权威，这里按它取 9 —— 差异已立 finding 报塔。）
        //
        // M136（Wave D）：`pfStageMask` 是 bring-up 截断开关（0 = 全开）。相位边界**只由 AIV 发**
        // （`M15L_PhaseBoundaryAiv` 是全体 AIV 的 mode-0 屏障，decode 路径一律包在 `if ASCEND_IS_AIV`
        // 里；M132 的 prefill 支曾无条件调它 ⇒ AIC 也走 mode-0，实测在设备上挂死）。这里按 decode
        // 的用法收进 AIV 分支；被截断的相位只是不调它的段体。GDN 整层四相位所缺的 hc 层界/权重字段
        // 见 `PF_STAGE_A` 的顶上注释（如实的能力边界）。
        const uint32_t st = (A.pfStageMask == 0u) ? PF_STAGE_ALL : A.pfStageMask;
        if ((st & PF_STAGE_H1) != 0u) {
            M15L_HcPrefillBoundary(0u, A);                // B5：hc 边界 #1（attn_hc）
        }
        if ASCEND_IS_AIV {
            M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>();  // 边界：H1 → 相位 A
        }
        if ((st & PF_STAGE_A) != 0u) {
            M15L_PrefillPhaseA<KIND>(A);                  // B1（GDN）/ B2（attn 前端，M197 已接；B3 未接）
        }
        if ASCEND_IS_AIV {
            M15L_PhaseBoundaryAiv<FLAG_HC1_BOUND_AIV>();  // 边界：相位 A → H2
        }
        if ((st & PF_STAGE_H2) != 0u) {
            M15L_HcPrefillBoundary(1u, A);                // B5：hc 边界 #2（mlp_hc）
        }
        if ASCEND_IS_AIV {
            M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>();  // 边界：H2 → 相位 B
        }
        if ((st & PF_STAGE_B) != 0u) {
            M15L_PrefillPhaseB<KIND>(A);                  // B4：MoE prefill
        }
        return;
    }
    // ---- 结构性占位（pfWired == 0，缺省）：逐行 x→y，数学未实现 ----
    M15L_PrefillPhaseA<KIND>(A);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIC {
        // AIC 侧相位 A：mmad 段体（B1 的 6 个收缩 / B3 的 FA core）落在这里；占位形态下 AIC 无活。
    }
}

// ============================================================
// 3e. M110：B1（GDN prefill）的**段接口契约**（`docs/19` §4.4 的逐条登记）
// ============================================================
// Wave A 只登记**契约**：缝函数的实现归 B1（`m15_gdn_prefill.h`），Wave C 在 §3d 的挂载点上接。
// 段体签名只吃「指针 + 标量」（Wave B 共同契约第 1 条），**不吃 `LayerArgs`**。
//
// 为什么要进接口（`docs/19` §4.4 第 2 张表的结论）：**S（h / 状态）的归属必须是接口的一部分**。
// 接口里只给裸指针、不给"它住哪"的枚举 ⇒ 交付实现会静默退化成「每 chunk 一次 GM 往返」，
// 把 (a) 支的最大结构收益（`docs/10:21` 记的 ~400 MB/层）还回去，而**没有任何判据会红**。
enum GdnStateHome { GDNS_AIV_UB = 0, GDNS_AIC_L1 = 1, GDNS_GM = 2 };
constexpr uint32_t GDNS_HOME_N = 3;

// 缝函数登记表（逐行 = `docs/19` §4.4 的缝表；形状口径 = **每 head-pair 每 chunk**）
struct GdnPrefillSeam {
    const char* fn;           // B1 应提供的函数名（`docs/19` §4.4 的名字）
    uint32_t    contraction;   // 1 = 矩阵乘法 ⇒ 人类裁决「一律 mmad」命中；0 = 非收缩（允许 VF）
    const char* shape;
    const char* note;
};
constexpr GdnPrefillSeam GDNPF_SEAMS[] = {
    {"Scan_Kk", 1, "64x64x128", "kk = k·kᵀ（核体 :612 GemmBT）"},
    {"Scan_QKt", 1, "64x64x128", "AB = q′·kᵀ（核体 :667 GemmBT；M104 复审 P2-3 补的那一条）"},
    {"Scan_StateApply", 1, "64x128x128", "d = u − w·S（:651/:652）"},
    {"Scan_OutInter", 1, "64x128x128", "o_part = q′·S（:657/:658）"},
    {"Scan_OutIntra", 1, "64x64x128", "o += AB·d（:672/:673）"},
    {"Scan_StateUpdate", 1, "128x64x128", "S += kᵀ·d′（:687/:688）"},
    {"Scan_SolveG", 0, "BT 行前代", "非收缩（三角求解）；形式按 docs/15 §3.1，**裁决待塔确认**（本 mission 不替塔裁）"},
    {"Scan_Gamma", 0, "64x64", "Γ 的 exp 组装（strict / incl）；VF fp32"},
    {"Scan_Scalar", 0, "BT 级", "cumsum / eg·ig / β / egL / 逐行缩放 / 归零；VF fp32"},
};
constexpr uint32_t GDNPF_SEAM_N = sizeof(GDNPF_SEAMS) / sizeof(GDNPF_SEAMS[0]);
// `docs/19 §4.4` 的缝表是 **9 行**（6 个逻辑收缩 + `Scan_SolveG` + `Scan_Gamma` + `Scan_Scalar`）。
// （塔的 M110 补充里写的是「8 个缝函数」；本 mission 按 `docs/19` 原文的 9 行登记，并把这条
//   差异写在 evidence/prefill_contract/README.md 里报塔，**不静默取一个**。）
static_assert(GDNPF_SEAM_N == 9u, "B1 的缝表条数变了（docs/19 §4.4 是 9 行）");
constexpr uint32_t GdnSeamContractions()
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < GDNPF_SEAM_N; ++i) {
        n += (GDNPF_SEAMS[i].contraction != 0u) ? 1u : 0u;
    }
    return n;
}
// 「一律 mmad」的裁决按收缩逐条命中：6 条（`docs/19` §10.2 的映射表）
static_assert(GdnSeamContractions() == 6u, "收缩缝不是 6 条（docs/19 §10.2：6 个收缩 + SolveG 的下左块）");
static_assert(GDNPF_SEAM_N - GdnSeamContractions() == 3u, "非收缩缝不是 3 条（SolveG / Gamma / Scalar）");

// ============================================================
// 3c. 融合 kernel 主体
// ============================================================

template <uint32_t KIND, bool HC>
__aicore__ inline void M15L_FusedBody(const LayerArgs& A)
{
    // 子层段的入口/出口：两相位 = 层输入/层输出；四相位 = hc(attn) 的 BLK / 子层段自己的出口
    __gm__ uint8_t* subIn = A.xLayer;
    __gm__ uint8_t* subOut = A.yLayer;

    GdnLayerChain gdn;
    M15H::HyperConnOp hc0;
    M15H::HyperConnOp hc1;
    M15H::HyperConnOp hcMix;   // **只**在 PLE 断点层（hcPleBreak）用：H1 的 mix-only 第二段
    if constexpr (HC) {
        M15H::HcPtrs hp0;
        M15L_FillHcPtrs(hp0, 0u, A);
        if (A.hcPleBreak != 0u) {
            // 第一段 = combine-only：只跑 W0+S1（H' = bf16(H + BO·injW)），AIC 不跑 GEMM。
            // `p.hIn` 仍是层输入 H（`FillHcPtrs` 的 which=0 分支），输出落在 `hcWs0+WS_HCP`。
            hp0.mode = M15H::MODE_COMBINE_ONLY;
        }
        hc0.Init(hp0);
        if (A.hcPleBreak != 0u) {
            // 第二段 = mix-only：hIn = 上一步物化的 H'（与 `hc0` **同一块 ws**，故 HCP 就地被读），
            // 权重仍是 attn_hc 那一组（`attn_hc.combine_and_mix(hidden, None, None)`）。
            M15H::HcPtrs hpM;
            M15L_FillHcPtrs(hpM, 0u, A);
            hpM.mode = M15H::MODE_MIX;
            hpM.hIn = A.hcWs0 + M15H::WS_HCP;
            hcMix.Init(hpM);
        }
        M15H::HcPtrs hp1;
        M15L_FillHcPtrs(hp1, 1u, A);
        hc1.Init(hp1);
        subIn = A.hcWs0 + M15H::WS_BLK;    // hc(attn) 的 block input → 子层段
        subOut = A.hcAttnOut;              // 子层段出口 → hc(mlp) 的 BO
    }

    if constexpr (KIND == KIND_GDN) {
        GdnLayerPtrs g;
        g.ws = A.ws;                  // GDN 段 ws @0
        g.xLayer = subIn;
        g.resLayer = A.resLayer;
        g.yLayer = subOut;
        g.gamma1 = A.gamma1;
        g.gamma2 = A.gamma2;
        g.gammaG = A.gammaG;
        g.wIn = A.wIn;
        g.wOut = A.wOut;
        g.convW = A.convW;
        g.convBias = A.convBias;
        g.aLog = A.aLog;
        g.dtBias = A.dtBias;
        g.convState = A.convState;
        g.ssmState = A.ssmState;
        g.xQkvzba = nullptr;          // 层循环不需要 slice 模式的 host 直供 qkvzba
        g.m = A.m;
        g.sliceMode = 0u;             // 全链 S1→S7
        g.stageLimit = 7u;
        gdn.Init(g);
    }

    M15M::MoeLayerPtrs mp;
    M15L_FillMoePtrs(mp, A.ws + WS_MOE_OFF, HC ? (A.hcWs1 + M15H::WS_BLK) : A.yLayer, A.resZero, A);
    M15M::MoeLayerChain moe;
    moe.Init(mp);
    moe.PrepareUnpermute(mp);

    if ASCEND_IS_AIV {
        if constexpr (HC) {
            hc0.ProcessAiv();                        // 相位 H1（hc 边界 #1；PLE 层 = combine-only）
            AscendC::PipeBarrier<PIPE_ALL>();
            if (A.hcPleBreak != 0u) {
                // ---- PLE 打断点的两段（层 1）：combine 物化 → PLE → mix ----
                // 两条相位边界的必要性同 §1 的论证：combine 的输出由**全体 AIV**协作写、mix 的
                // 消费者用**另一套行切分**读 ⇒ 跨核可见性只能靠 CrossCore mode-0 barrier。
                //
                // **M100 的接线点在下面三行之间**：①（n-gram id）在 `FLAG_PLE_IN_BOUND_AIV`
                // **之前**跑 ⇒ 那条已登记的全体 AIV mode-0 barrier 同时承担「① 的 ids 已落 GM」，
                // ② 才读它 ⇒ **不新增任何 flagId**（预算表见 evidence/ple_wire/INTERFACE.md §4）。
                // `A.pleW == nullptr`（接线关）时这里逐字退化为 M65 的空操作。
                if (A.pleW != nullptr) {
                    M15L_PleIds(A);                  // ① n-gram id → S_IDS
                    // ① 的 ids 是**标量写 GM**（`outG.SetValue`），② 在**别的核**上读它；而
                    // `M15L_PhaseBoundaryAiv` 的 set 挂 MTE3（只排 MTE3 写）。
                    // **试过**在 ① 与边界之间补一条 `AscendC::PipeBarrier<PIPE_ALL>()`（排空本核
                    // 标量 pipe）—— **实测无效**：`Pw.emb.T1` 仍是 `bad=15/16`、`Pw.det` 仍红，
                    // 与不加时逐项相同 ⇒ 根因是**写路径**（标量 `SetValue` 的落点跨核不可见），
                    // 不是排序能修的（见 evidence/ple_wire/WITNESS.md §4）。故**该行已撤掉**，
                    // 不是为了掩盖失败，而是它既无效又会扰动 codegen（见 WITNESS.md §6 的
                    // `M.moews` 脆弱性证据）。修法见 U-A（INTERFACE.md §7）。
                }
                M15L_PhaseBoundaryAiv<FLAG_PLE_IN_BOUND_AIV>();
                if (A.pleW != nullptr) {
                    M15L_PleBody(A);                 // ②③④⑤（表行 gather → kv → 门控 → 卷积）
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                M15L_PhaseBoundaryAiv<FLAG_PLE_OUT_BOUND_AIV>();
                hcMix.ProcessAiv();                  // 相位 H1b（mix-only）
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>();
        }
        if constexpr (KIND == KIND_GDN) {
            gdn.ProcessAiv();                        // 相位 A（GDN 段）
        } else if (A.apW != nullptr) {
            // ---- 相位 A（**真 attention 前端**，M97 W5）----
            // 调 M88 的 prolog body（它的 AIC/AIV 两段写在同一个函数里、用 `ASCEND_IS_AIV` /
            // `ASCEND_IS_AIC` 分核）。AIV 核在这里只执行它的 AIV 段：先等 AIC 的 mode-2 通知
            // 「4 个 GEMM 的 GM 输出可见」，再按头做 GemmaRMSNorm(256/128) + partial NeoX RoPE(64)，
            // 以及 gate / v / raw k 的原样抄写（R2/R13）。**AIV 侧不 set 任何 flag**。
            // 核内同步全部走 buffer id（`AP_BUF_IO/IN/OUT`），跨核走 CrossCore mode0/mode2
            // （分节与相邻性证明见 `m15_layer_resources.h` §4b）。
            M15AP::m15_attn_prolog_probe_body(A.apW, A.apX, A.apCs, A.apY0, A.apOut, A.apPos, A.apMode, 0u);
        } else {
            m15_attn_passthrough_body(subIn, subOut, HIDDEN * 2u * A.m);   // 相位 A（占位直通）
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (HC) {
            M15L_PhaseBoundaryAiv<FLAG_HC1_BOUND_AIV>();
            hc1.ProcessAiv();                        // 相位 H2（hc 边界 #2）
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>(); // 相位 A→B 的边界（两相位形态也用它）
        moe.ProcessAiv();                            // 相位 B（MoE 段）
    }
    if ASCEND_IS_AIC {
        if constexpr (HC) {
            hc0.ProcessAic();                        // H1 的 down / up GEMM（combine-only 时立即返回）
            AscendC::PipeBarrier<PIPE_ALL>();
            if (A.hcPleBreak != 0u) {
                // ---- M124：PLE ③（cube mmad）的 AIC 挂载点 ----
                // 位置与 AIV 侧逐字对齐（AIV 的挂载点夹在 `FLAG_PLE_IN_BOUND_AIV` 与
                // `FLAG_PLE_OUT_BOUND_AIV` 之间；AIC 侧夹在 hc(H1a) 与 hc(H1b) 之间）。
                // 依赖序：AIV ②（不依赖 AIC）→ AIC ③（等 ②）→ AIV ④⑤（等 ③）→ AIV H1b → AIC hcMix
                //   ⇒ 单调、无环（`M15L_PleAic` 的注释里有逐条论证）。
                if (A.pleW != nullptr) {
                    M15L_PleAic(A);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                hcMix.ProcessAic();                  // H1b 的 down / up GEMM
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        if constexpr (KIND == KIND_GDN) {
            gdn.ProcessAic();                        // GDN 的 in_proj / out_proj
        } else if (A.apW != nullptr) {
            // ---- 相位 A（**真 attention 前端**，M97 W6）----
            // `M15L_FusedBody` 的 AIC 分支此前**对 attention 完全缺席**（M76 的勘查结论）。这里补上：
            // 同一个 prolog body 在 AIC 核上只执行它的 AIC 段 —— 4 个 bf16 GEMM（q_proj / k_proj /
            // v_proj / index_qk_proj，按 N-块跨 AIC 步进），随后 ① mode-0 全体 AIC 对齐（FIXP 写 GM
            // 排空 + 所有核到齐）、② mode-2 通知本核配对的 2 个 AIV。两条缺一不可：只有 ② 的话
            // AIV 只知道自己那对 AIC 好了（同 `m15_gdn_layer.h` 的 `FLAG_AIC_SEG0B → FLAG_BOUND_INPROJ`）。
            M15AP::m15_attn_prolog_probe_body(A.apW, A.apX, A.apCs, A.apY0, A.apOut, A.apPos, A.apMode, 0u);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (HC) {
            hc1.ProcessAic();                        // H2 的 down / up GEMM
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        moe.ProcessAic();                            // MoE 的 gate_up / down GEMM
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ============================================================
// 4. 入口符号
// ============================================================
// 参数顺序与 LayerArgs 字段顺序一致（host 侧 H_Launch 逐项对应）。

#define M15L_LAYER_ARGS_DECL                                                                                    \
    __gm__ uint8_t* ws, __gm__ uint8_t* xLayer, __gm__ uint8_t* resLayer, __gm__ uint8_t* yLayer,                \
        __gm__ uint8_t* resZero, __gm__ uint8_t* gamma1, __gm__ uint8_t* gamma2, __gm__ uint8_t* gammaG,          \
        __gm__ uint8_t* wIn, __gm__ uint8_t* wOut, __gm__ uint8_t* convW, __gm__ uint8_t* convBias,               \
        __gm__ uint8_t* aLog, __gm__ uint8_t* dtBias, __gm__ uint8_t* convState, __gm__ uint8_t* ssmState,        \
        __gm__ uint8_t* moeRouter, __gm__ uint8_t* moeSgate, __gm__ uint8_t* moeWGu, __gm__ uint8_t* moeSGu,      \
        __gm__ uint8_t* moeWDn, __gm__ uint8_t* moeSDn, __gm__ uint8_t* moeWGuShd, __gm__ uint8_t* moeSGuShd,     \
        __gm__ uint8_t* moeWDnShd, __gm__ uint8_t* moeSDnShd, __gm__ uint8_t* moeGamma1,                          \
        __gm__ uint8_t* moeGamma2, uint32_t m, uint32_t topk

#define M15L_LAYER_ARGS_FILL                                                                                  \
    LayerArgs A;                                                                                              \
    A.ws = ws;                                                                                                \
    A.xLayer = xLayer;                                                                                        \
    A.resLayer = resLayer;                                                                                    \
    A.yLayer = yLayer;                                                                                        \
    A.resZero = resZero;                                                                                      \
    A.gamma1 = gamma1;                                                                                        \
    A.gamma2 = gamma2;                                                                                        \
    A.gammaG = gammaG;                                                                                        \
    A.wIn = wIn;                                                                                              \
    A.wOut = wOut;                                                                                            \
    A.convW = convW;                                                                                          \
    A.convBias = convBias;                                                                                    \
    A.aLog = aLog;                                                                                            \
    A.dtBias = dtBias;                                                                                        \
    A.convState = convState;                                                                                  \
    A.ssmState = ssmState;                                                                                    \
    A.moeRouter = moeRouter;                                                                                  \
    A.moeSgate = moeSgate;                                                                                    \
    A.moeWGu = moeWGu;                                                                                        \
    A.moeSGu = moeSGu;                                                                                        \
    A.moeWDn = moeWDn;                                                                                        \
    A.moeSDn = moeSDn;                                                                                        \
    A.moeWGuShd = moeWGuShd;                                                                                  \
    A.moeSGuShd = moeSGuShd;                                                                                  \
    A.moeWDnShd = moeWDnShd;                                                                                  \
    A.moeSDnShd = moeSDnShd;                                                                                  \
    A.moeGamma1 = moeGamma1;                                                                                  \
    A.moeGamma2 = moeGamma2;                                                                                  \
    A.m = m;                                                                                                  \
    A.topk = topk;                                                                                            \
    /* 两相位入口：hc 字段全部缺省（kernel 内 `if constexpr (HC)` 保证不被触碰）*/                             \
    A.hcH = nullptr;                                                                                          \
    A.hcBo = nullptr;                                                                                         \
    A.hcIj = nullptr;                                                                                         \
    A.hcAttnOut = nullptr;                                                                                    \
    A.hcWs0 = nullptr;                                                                                        \
    A.hcWs1 = nullptr;                                                                                        \
    A.hcAttnNorm = nullptr;                                                                                   \
    A.hcAttnDown = nullptr;                                                                                   \
    A.hcAttnInj = nullptr;                                                                                    \
    A.hcAttnUp = nullptr;                                                                                     \
    A.hcMlpNorm = nullptr;                                                                                    \
    A.hcMlpDown = nullptr;                                                                                    \
    A.hcMlpInj = nullptr;                                                                                     \
    A.hcMlpUp = nullptr;                                                                                      \
    A.hcIjStride = M15H::HC_IJ_STRIDE_PLANE;                                                                  \
    A.hcAttnMode = M15H::MODE_MIX;                                                                            \
    A.hcMlpMode = M15H::MODE_COMBINE_MIX;                                                                     \
    A.hcPleBreak = 0u;                                                                                        \
    /* M97：attention 相位 A 的字段 —— 两相位入口与 GDN 入口全部缺省 */                                          \
    A.apW = nullptr;                                                                                          \
    A.apX = nullptr;                                                                                          \
    A.apCs = nullptr;                                                                                         \
    A.apY0 = nullptr;                                                                                         \
    A.apOut = nullptr;                                                                                        \
    A.apPos = 0u;                                                                                             \
    A.apMode = 0u;                                                                                            \
    A.layer = 0u;                                                                                             \
    /* M100：两相位入口不做 PLE（PLE 只走四相位入口的打断点层），字段全部缺省 */                                \
    A.pleW = nullptr;                                                                                         \
    A.pleScr = nullptr;                                                                                       \
    A.pleTable = nullptr;                                                                                     \
    A.pleIds = nullptr;                                                                                       \
    A.pleQsl = nullptr;                                                                                       \
    A.pleCtx = nullptr;                                                                                       \
    A.pleM = nullptr;                                                                                         \
    A.pleSz = nullptr;                                                                                        \
    A.pleOf = nullptr;                                                                                        \
    A.pleTok = 0u;                                                                                            \
    A.pleNReq = 0u;                                                                                           \
    A.pleTableRows = 0u;                                                                                      \
    A.pleStageMask = 0u;                                                                                      \
    A.pleNegMask = 0u;                                                                                        \
    A.pleWinBase = 0u;                                                                                        \
    A.pleWinRows = 0u;                                                                                        \
    A.pleStateSlots = 0u;                                                                                     \
    /* M110：prefill 的字段 —— 既有入口全部缺省（它们不引用这些字段；prefill 入口用下面的 FILL 覆写）*/           \
    A.pfKv = nullptr;                                                                                         \
    A.pfComp = nullptr;                                                                                       \
    A.pfRing = nullptr;                                                                                       \
    A.pfPack = nullptr;                                                                                       \
    A.pfPos = nullptr;                                                                                        \
    A.pfBlockTable = nullptr;                                                                                 \
    A.pfCounts = nullptr;                                                                                     \
    A.pfExpertOffsets = nullptr;                                                                              \
    A.pfPosBase = 0u;                                                                                         \
    A.pfLayerK = PF_LAYER_K_NONE;                                                                             \
    A.pfStageMask = 0u;                                                                                       \
    A.pfWired = 0u;                                                                                           \
    A.pfMutant = 0u;                                                                                          \
    /* M197：attention 前端的 5 个 scratch 平面（两相位/HC 入口不引用，全部缺省 nil）*/                          \
    A.pfY0 = nullptr;                                                                                         \
    A.pfOut = nullptr;                                                                                        \
    A.pfIn = nullptr;                                                                                         \
    A.pfPooled = nullptr;                                                                                     \
    A.pfFlag = nullptr

// 四相位入口的参数表 = 两相位参数表 + hc 的 17 个参数（追加在 `m/topk` 之前，与结构体顺序一致）
#define M15L_LAYER_HC_ARGS_DECL                                                                                 \
    __gm__ uint8_t* ws, __gm__ uint8_t* xLayer, __gm__ uint8_t* resLayer, __gm__ uint8_t* yLayer,                \
        __gm__ uint8_t* resZero, __gm__ uint8_t* gamma1, __gm__ uint8_t* gamma2, __gm__ uint8_t* gammaG,          \
        __gm__ uint8_t* wIn, __gm__ uint8_t* wOut, __gm__ uint8_t* convW, __gm__ uint8_t* convBias,               \
        __gm__ uint8_t* aLog, __gm__ uint8_t* dtBias, __gm__ uint8_t* convState, __gm__ uint8_t* ssmState,        \
        __gm__ uint8_t* moeRouter, __gm__ uint8_t* moeSgate, __gm__ uint8_t* moeWGu, __gm__ uint8_t* moeSGu,      \
        __gm__ uint8_t* moeWDn, __gm__ uint8_t* moeSDn, __gm__ uint8_t* moeWGuShd, __gm__ uint8_t* moeSGuShd,     \
        __gm__ uint8_t* moeWDnShd, __gm__ uint8_t* moeSDnShd, __gm__ uint8_t* moeGamma1,                          \
        __gm__ uint8_t* moeGamma2, __gm__ uint8_t* hcH, __gm__ uint8_t* hcBo, __gm__ uint8_t* hcIj,               \
        __gm__ uint8_t* hcAttnOut, __gm__ uint8_t* hcWs0, __gm__ uint8_t* hcWs1, __gm__ uint8_t* hcAttnNorm,      \
        __gm__ uint8_t* hcAttnDown, __gm__ uint8_t* hcAttnInj, __gm__ uint8_t* hcAttnUp, __gm__ uint8_t* hcMlpNorm, \
        __gm__ uint8_t* hcMlpDown, __gm__ uint8_t* hcMlpInj, __gm__ uint8_t* hcMlpUp, uint32_t hcIjStride,        \
        uint32_t hcAttnMode, uint32_t hcMlpMode, uint32_t hcPleBreak, __gm__ uint8_t* apW,                       \
        __gm__ uint8_t* apX, __gm__ uint8_t* apCs, __gm__ uint8_t* apY0, __gm__ uint8_t* apOut,                  \
        uint32_t apPos, uint32_t apMode, uint32_t layer, uint32_t m, uint32_t topk,                              \
        /* ---- M100：PLE 段的 17 个实参（追加在末尾；两相位入口不带它们）---- */                                  \
        __gm__ uint8_t* pleW, __gm__ uint8_t* pleScr, __gm__ uint8_t* pleTable, __gm__ uint8_t* pleIds,           \
        __gm__ uint8_t* pleQsl, __gm__ uint8_t* pleCtx, __gm__ uint8_t* pleM, __gm__ uint8_t* pleSz,              \
        __gm__ uint8_t* pleOf, uint32_t pleTok, uint32_t pleNReq, uint32_t pleTableRows,                          \
        uint32_t pleStageMask, uint32_t pleNegMask, uint32_t pleWinBase, uint32_t pleWinRows,                     \
        uint32_t pleStateSlots

#define M15L_LAYER_HC_ARGS_FILL                                                                               \
    M15L_LAYER_ARGS_FILL;                                                                                     \
    A.hcH = hcH;                                                                                              \
    A.hcBo = hcBo;                                                                                            \
    A.hcIj = hcIj;                                                                                            \
    A.hcAttnOut = hcAttnOut;                                                                                  \
    A.hcWs0 = hcWs0;                                                                                          \
    A.hcWs1 = hcWs1;                                                                                          \
    A.hcAttnNorm = hcAttnNorm;                                                                                \
    A.hcAttnDown = hcAttnDown;                                                                                \
    A.hcAttnInj = hcAttnInj;                                                                                  \
    A.hcAttnUp = hcAttnUp;                                                                                    \
    A.hcMlpNorm = hcMlpNorm;                                                                                  \
    A.hcMlpDown = hcMlpDown;                                                                                  \
    A.hcMlpInj = hcMlpInj;                                                                                    \
    A.hcMlpUp = hcMlpUp;                                                                                      \
    A.hcIjStride = hcIjStride;                                                                                \
    A.hcAttnMode = hcAttnMode;                                                                                \
    A.hcMlpMode = hcMlpMode;                                                                                  \
    A.hcPleBreak = hcPleBreak;                                                                                \
    A.apW = apW;                                                                                              \
    A.apX = apX;                                                                                              \
    A.apCs = apCs;                                                                                            \
    A.apY0 = apY0;                                                                                            \
    A.apOut = apOut;                                                                                          \
    A.apPos = apPos;                                                                                          \
    A.apMode = apMode;                                                                                        \
    A.layer = layer;                                                                                          \
    A.pleW = pleW;                                                                                            \
    A.pleScr = pleScr;                                                                                        \
    A.pleTable = pleTable;                                                                                    \
    A.pleIds = pleIds;                                                                                        \
    A.pleQsl = pleQsl;                                                                                        \
    A.pleCtx = pleCtx;                                                                                        \
    A.pleM = pleM;                                                                                            \
    A.pleSz = pleSz;                                                                                          \
    A.pleOf = pleOf;                                                                                          \
    A.pleTok = pleTok;                                                                                        \
    A.pleNReq = pleNReq;                                                                                      \
    A.pleTableRows = pleTableRows;                                                                            \
    A.pleStageMask = pleStageMask;                                                                            \
    A.pleNegMask = pleNegMask;                                                                                \
    A.pleWinBase = pleWinBase;                                                                                \
    A.pleWinRows = pleWinRows;                                                                                \
    A.pleStateSlots = pleStateSlots

// ============================================================
// 5. M110：**prefill 入口**的实参表与两个 `__global__` 符号
// ============================================================
// 实参表 = 基础 30 实参（与两相位入口**逐字相同**，故 `M15L_LAYER_ARGS_FILL` 可直接复用）
//          + 13 个 prefill 项（M110）+ 13 个段体项（M132：B1 的 7 + B4 的 1 + B5 的 5）
//          + 15 个 hc 层界/权重项（M140：H1/H2/B 的段体实参）。
// **四相位入口（HC）的实参表一字未动**（PLE 接线保持原样）。
//
// M140 的 15 项 = `LayerArgs` 里**早已存在**的 hc 字段（M58 的 `hcH/hcBo/hcIj/hcAttnOut/
// hcAttn*/hcMlp*/hcIjStride/hcAttnMode/hcMlpMode`），此前 prefill 入口的实参表没带它们
// ⇒ 入口内 `M15L_LAYER_ARGS_FILL` 把它们全置 nil，`pfStageMask` 只能开相位 A（M136 的能力边界，
// finding `20261004-agent-waved1-bug-m136-prefill-hc-gdn-h1-h2-b.md`）。这里把它们逐项加入入口。
#define M15L_LAYER_PREFILL_ARGS_DECL                                                                          \
    __gm__ uint8_t* ws, __gm__ uint8_t* xLayer, __gm__ uint8_t* resLayer, __gm__ uint8_t* yLayer,              \
        __gm__ uint8_t* resZero, __gm__ uint8_t* gamma1, __gm__ uint8_t* gamma2, __gm__ uint8_t* gammaG,        \
        __gm__ uint8_t* wIn, __gm__ uint8_t* wOut, __gm__ uint8_t* convW, __gm__ uint8_t* convBias,             \
        __gm__ uint8_t* aLog, __gm__ uint8_t* dtBias, __gm__ uint8_t* convState, __gm__ uint8_t* ssmState,      \
        __gm__ uint8_t* moeRouter, __gm__ uint8_t* moeSgate, __gm__ uint8_t* moeWGu, __gm__ uint8_t* moeSGu,    \
        __gm__ uint8_t* moeWDn, __gm__ uint8_t* moeSDn, __gm__ uint8_t* moeWGuShd, __gm__ uint8_t* moeSGuShd,   \
        __gm__ uint8_t* moeWDnShd, __gm__ uint8_t* moeSDnShd, __gm__ uint8_t* moeGamma1,                        \
        __gm__ uint8_t* moeGamma2, uint32_t m, uint32_t topk,                                                   \
        /* ---- M110：prefill 的 13 个实参（顺序与 LayerArgs 尾部字段一致）---- */                                \
        __gm__ uint8_t* pfKv, __gm__ uint8_t* pfComp, __gm__ uint8_t* pfRing, __gm__ uint8_t* pfPack,           \
        __gm__ uint8_t* pfPos, __gm__ uint8_t* pfBlockTable, __gm__ uint8_t* pfCounts,                          \
        __gm__ uint8_t* pfExpertOffsets, uint32_t pfPosBase, uint32_t pfLayerK, uint32_t pfStageMask,           \
        uint32_t pfWired, uint32_t pfMutant,                                                                    \
        /* ---- M132：B1/B4/B5 段体的 13 个实参（顺序与 LayerArgs 追加字段一致）---- */                           \
        __gm__ uint8_t* wsQ, __gm__ uint8_t* wsK, __gm__ uint8_t* wsV, __gm__ uint8_t* wsG,                      \
        __gm__ uint8_t* wsBeta, __gm__ uint8_t* wsO, __gm__ uint8_t* wsGdnPrefillScratch,                       \
        __gm__ uint8_t* pfMoeRouterWPad, __gm__ uint8_t* pfHcArena0, __gm__ uint8_t* pfHcArena1,                \
        __gm__ uint8_t* pfHcBlk0, __gm__ uint8_t* pfHcBlk1, __gm__ uint8_t* pfHcIj0,                            \
        /* ---- M140：hc 层界/权重与 H1/H2/B 的 15 个实参（顺序与 LayerArgs 的 hc 字段一致）---- */              \
        __gm__ uint8_t* hcH, __gm__ uint8_t* hcBo, __gm__ uint8_t* hcIj, __gm__ uint8_t* hcAttnOut,            \
        __gm__ uint8_t* hcAttnNorm, __gm__ uint8_t* hcAttnDown, __gm__ uint8_t* hcAttnInj,                     \
        __gm__ uint8_t* hcAttnUp, __gm__ uint8_t* hcMlpNorm, __gm__ uint8_t* hcMlpDown,                        \
        __gm__ uint8_t* hcMlpInj, __gm__ uint8_t* hcMlpUp, uint32_t hcIjStride, uint32_t hcAttnMode,           \
        uint32_t hcMlpMode,                                                                                     \
        /* ---- M197（B2 挂载）：attention 前端的 5 个 scratch 平面（追加在末尾；顺序 = LayerArgs 字段序）---- */  \
        __gm__ uint8_t* pfY0, __gm__ uint8_t* pfOut, __gm__ uint8_t* pfIn,                                       \
        __gm__ uint8_t* pfPooled, __gm__ uint8_t* pfFlag,                                                        \
        /* M197 附带：§5.8 补丁引用 `A.apW`/`A.apCs`，而本实参表此前**不带**它们（只有 HC 入口带）         \
           ⇒ B2 挂载起必须补上这两个输入平面（否则挂载点取到的 w/cs 恒 nullptr）。*/                          \
        __gm__ uint8_t* apW, __gm__ uint8_t* apCs

#define M15L_LAYER_PREFILL_ARGS_FILL                       \
    M15L_LAYER_ARGS_FILL;                                  \
    A.pfKv = pfKv;                                         \
    A.pfComp = pfComp;                                     \
    A.pfRing = pfRing;                                     \
    A.pfPack = pfPack;                                     \
    A.pfPos = pfPos;                                       \
    A.pfBlockTable = pfBlockTable;                         \
    A.pfCounts = pfCounts;                                 \
    A.pfExpertOffsets = pfExpertOffsets;                   \
    A.pfPosBase = pfPosBase;                               \
    A.pfLayerK = pfLayerK;                                 \
    A.pfStageMask = pfStageMask;                           \
    A.pfWired = pfWired;                                   \
    A.pfMutant = pfMutant;                                 \
    /* ---- M132：B1/B4/B5 段体字段 ---- */                 \
    A.wsQ = wsQ;                                           \
    A.wsK = wsK;                                           \
    A.wsV = wsV;                                           \
    A.wsG = wsG;                                           \
    A.wsBeta = wsBeta;                                     \
    A.wsO = wsO;                                           \
    A.wsGdnPrefillScratch = wsGdnPrefillScratch;           \
    A.pfMoeRouterWPad = pfMoeRouterWPad;                   \
    A.pfHcArena0 = pfHcArena0;                             \
    A.pfHcArena1 = pfHcArena1;                             \
    A.pfHcBlk0 = pfHcBlk0;                                 \
    A.pfHcBlk1 = pfHcBlk1;                                 \
    A.pfHcIj0 = pfHcIj0;                                   \
    /* ---- M140：hc 层界/权重字段（覆盖 M15L_LAYER_ARGS_FILL 的缺省 nil）---- */                            \
    A.hcH = hcH;                                           \
    A.hcBo = hcBo;                                         \
    A.hcIj = hcIj;                                         \
    A.hcAttnOut = hcAttnOut;                               \
    A.hcAttnNorm = hcAttnNorm;                             \
    A.hcAttnDown = hcAttnDown;                             \
    A.hcAttnInj = hcAttnInj;                               \
    A.hcAttnUp = hcAttnUp;                                 \
    A.hcMlpNorm = hcMlpNorm;                               \
    A.hcMlpDown = hcMlpDown;                               \
    A.hcMlpInj = hcMlpInj;                                 \
    A.hcMlpUp = hcMlpUp;                                   \
    A.hcIjStride = hcIjStride;                             \
    A.hcAttnMode = hcAttnMode;                             \
    A.hcMlpMode = hcMlpMode;                               \
    /* ---- M197（B2 挂载）：attention 前端的 5 个 scratch 平面 ---- */  \
    A.pfY0 = pfY0;                                         \
    A.pfOut = pfOut;                                       \
    A.pfIn = pfIn;                                         \
    A.pfPooled = pfPooled;                                 \
    A.pfFlag = pfFlag;                                     \
    /* M197 附带：B2 的权重 / cos-sin 输入平面（§5.8 补丁引用 A.apW/A.apCs）*/  \
    A.apW = apW;                                           \
    A.apCs = apCs

// ---- M110 交付形态：**prefill 的两个真入口**（M103-2.1 ①：从"符号预留"变成真入口登记）----
// 每个 kernel 只跑**一层**（人类裁定）；本 mission 的形态 = 相位 A 的挂载点（其余相位见 §3d）。
// host 侧按 `m15_layer_resources.h` §6b 的 `ENTRY_TABLE[]` 选符号。
__global__ __mix__(1, 2) void m15_layer_kernel_gdn_prefill(M15L_LAYER_PREFILL_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_PREFILL_ARGS_FILL;
    M15L_PrefillBody<KIND_GDN>(A);
}

__global__ __mix__(1, 2) void m15_layer_kernel_attn_prefill(M15L_LAYER_PREFILL_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_PREFILL_ARGS_FILL;
    M15L_PrefillBody<KIND_ATTN>(A);
}

__global__ __mix__(1, 2) void m15_layer_kernel_gdn(M15L_LAYER_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_ARGS_FILL;
    M15L_FusedBody<KIND_GDN, false>(A);
}

__global__ __mix__(1, 2) void m15_layer_kernel_attn(M15L_LAYER_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_ARGS_FILL;
    M15L_FusedBody<KIND_ATTN, false>(A);
}

// ---- M58 交付形态：四相位（hc(attn) → 子层段 → hc(mlp) → MoE 段），每层一次启动 ----
__global__ __mix__(1, 2) void m15_layer_kernel_gdn_hc(M15L_LAYER_HC_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_HC_ARGS_FILL;
    M15L_FusedBody<KIND_GDN, true>(A);
}

__global__ __mix__(1, 2) void m15_layer_kernel_attn_hc(M15L_LAYER_HC_ARGS_DECL)
{
    AscendC::InitSocState();
    M15L_LAYER_HC_ARGS_FILL;
    M15L_FusedBody<KIND_ATTN, true>(A);
}

// ---- 仅验证用：只跑 MoE 段（相位 A 缺席）----
// 用途（README §验证 M1「段间零串扰」）：把它与融合 kernel 打同一个输入（融合路传相位 A 的
// 出口 y），逐字节比对 MoE 段 ws —— 若融合引入了跨相位串扰（UB/资源叠放错、BufferID 复用
// 未 drain、flagId 冲突），这条判据会直接红。它**不是**交付形态，不算性能基线。
// `ws` 传 **MoE 段 ws 基址**（host 侧 = 融合 ws + WS_MOE_OFF），故本条目与融合路的 MoE 段
// 落在完全相同的地址上。
__global__ __mix__(1, 2) void m15_moe_segment_kernel(
    __gm__ uint8_t* ws, __gm__ uint8_t* xLayer, __gm__ uint8_t* yOut, __gm__ uint8_t* resZero,
    __gm__ uint8_t* gamma1,
    __gm__ uint8_t* gamma2, __gm__ uint8_t* routerW, __gm__ uint8_t* sgateW, __gm__ uint8_t* wGu, __gm__ uint8_t* sGu,
    __gm__ uint8_t* wDn, __gm__ uint8_t* sDn, __gm__ uint8_t* wGuShd, __gm__ uint8_t* sGuShd, __gm__ uint8_t* wDnShd,
    __gm__ uint8_t* sDnShd, uint32_t m, uint32_t topk)
{
    AscendC::InitSocState();
    LayerArgs A;
    A.ws = ws;
    A.yLayer = yOut;
    A.resZero = resZero;
    A.moeGamma1 = gamma1;
    A.moeGamma2 = gamma2;
    A.moeRouter = routerW;
    A.moeSgate = sgateW;
    A.moeWGu = wGu;
    A.moeSGu = sGu;
    A.moeWDn = wDn;
    A.moeSDn = sDn;
    A.moeWGuShd = wGuShd;
    A.moeSGuShd = sGuShd;
    A.moeWDnShd = wDnShd;
    A.moeSDnShd = sDnShd;
    A.m = m;
    A.topk = topk;

    M15M::MoeLayerPtrs mp;
    M15L_FillMoePtrs(mp, ws, xLayer, resZero, A);
    M15M::MoeLayerChain moe;
    moe.Init(mp);
    moe.PrepareUnpermute(mp);
    if ASCEND_IS_AIV {
        moe.ProcessAiv();
    }
    if ASCEND_IS_AIC {
        moe.ProcessAic();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ---- 仅验证用：只跑**一个** hc 边界（`m15_hc_segment_kernel`）----
// 用途（三条）：
//   ① 四相位形态的「段间零串扰」对照路：与融合 kernel 打同一组输入（含**层内 handoff 的两个
//      上游张量**：H1 的 H' 与 IJ），逐字节比对 hc 边界的整块 ws；
//   ② **combine-only 档的可用入口**（`mode = M15H::MODE_COMBINE_ONLY`）：m20 的三档 mode 都
//      不提供这一档，而层 0→1 被 PLE 打断时要用（docs/14 §4.1）。**M65 已把它接进四相位入口**
//      （`LayerArgs::hcPleBreak`），本入口继续作为它的对照路；
//   ③ bring-up 定位（逐段截断由 `stageLimit` 控制，恒传 HC_STAGE_LIMIT=7 全开）。
__global__ __mix__(1, 2) void m15_hc_segment_kernel(
    __gm__ uint8_t* hIn, __gm__ uint8_t* bo, __gm__ uint8_t* ij, uint32_t ijStride, __gm__ uint8_t* wDown,
    __gm__ uint8_t* wInj, __gm__ uint8_t* wUp, __gm__ uint8_t* hcNorm, __gm__ uint8_t* ws, uint32_t m, uint32_t mode)
{
    AscendC::InitSocState();
    M15H::HcPtrs p;
    p.hIn = hIn;
    p.bo = bo;
    p.ij = ij;
    p.ijStride = ijStride;
    p.wDown = wDown;
    p.wInj = wInj;
    p.wUp = wUp;
    p.hcNorm = hcNorm;
    p.ws = ws;
    p.m = m;
    p.mode = mode;
    p.stageLimit = M15H::HC_STAGE_LIMIT;

    M15H::HyperConnOp op;
    op.Init(p);
    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
    if ASCEND_IS_AIC {
        op.ProcessAic();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ---- 交付形态：**末层之后的全局 mixer**（`hyper_connection_mixer`，M65）----
// vLLM：`multi_hidden, sample_hidden, _ = final_mixer.combine_and_mix(hidden, block_output, injection)`
// （`V-N:model.py:581-583`，`use_combine=False`）。它**仍然做 combine**（`hc_combine_norm` 无条件
// 执行，`V-N:hyperconnection.py:166-173`），只是**不产出新的 injection**（down 只跑 lora 的
// `LOWRANK/BASE_N = 2` 个 N-tile，`OH[:,320:324)` 不再被写）——这正是 `MODE_FINAL_MIX` 的语义。
//
// 输出落点（与 m20 的 mode 2 表一致，**不额外搬运**）：
//   `multi_hidden [m,10240]` = `ws + WS_HCP`（combine 后的多流态，MTP 用）
//   `sample_hidden [m,2560]` = `ws + WS_BLK`（gate mix 的门控均值，直接进 lm_head）
// checkpoint 的 `hyper_connection_mixer` **没有** `block_inject_weight`（`V-N:model.py:612` 显式丢弃）
// ⇒ host 侧给 `wInj` 传该槽里**全零**的 16 行区（S3 的 2 个 N-tile 都不读它）。
//
// **与 `m15_hc_segment_kernel(mode=MODE_FINAL_MIX)` 的关系**：同一段 device 实现、同一档 mode。
// 单独给符号是为了让设备侧分解能**按符号**把「末层 mixer」与其它内核分开（README 的段级表）。
__global__ __mix__(1, 2) void m15_final_mixer_kernel(
    __gm__ uint8_t* hIn, __gm__ uint8_t* bo, __gm__ uint8_t* ij, uint32_t ijStride, __gm__ uint8_t* wDown,
    __gm__ uint8_t* wInj, __gm__ uint8_t* wUp, __gm__ uint8_t* hcNorm, __gm__ uint8_t* ws, uint32_t m)
{
    AscendC::InitSocState();
    M15H::HcPtrs p;
    p.hIn = hIn;
    p.bo = bo;
    p.ij = ij;
    p.ijStride = ijStride;
    p.wDown = wDown;
    p.wInj = wInj;
    p.wUp = wUp;
    p.hcNorm = hcNorm;
    p.ws = ws;
    p.m = m;
    p.mode = M15H::MODE_FINAL_MIX;
    p.stageLimit = M15H::HC_STAGE_LIMIT;

    M15H::HyperConnOp op;
    op.Init(p);
    if ASCEND_IS_AIV {
        op.ProcessAiv();
    }
    if ASCEND_IS_AIC {
        op.ProcessAic();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace M15L

#endif  // M15_LAYER_KERNEL_H
