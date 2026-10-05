/**
 * m15_hc_prefill.h —— **hc（hyper-connection）边界的 prefill 行分块（m-tile）循环**（M119 / Wave B5）
 *
 * ## 它解决什么
 * hc 段的 AIV 侧**已经按运行期 `m` 参数化**（五个段的 item 网格都是 `p.m * NCH_*`），
 * 所以缺的不是行循环，而是**包络**：`m15_hc_resources.h` 的 `M15H::M_MAX = 64` 是**编译期**定尺
 * （UB 的 `UB_IWTAB_SLOTS = M_MAX * HC` 槽表、`SZ_*` 的 GM 平面、`BASE_M = M_MAX` 的单 m-tile）。
 * `m = 4097` 直接喂进去 ⇒ 必须把 `M_MAX` 抬到 4097，而那时
 * `UB_IWTAB_SLOTS * INJW_SLOT * 4 = 4097*4*8*4 = 524,416 B > 248 KB`（UB 总容量）⇒ **不可能**。
 * （这条墙由下面的 `ENVELOPE_*` 两条 `static_assert` 机械钉住，不是散文。）
 *
 * 本头的做法 = **塔裁的 m-tile（方案 α，纯新文件）**：把 `m` 切成 `MT` 行一块（缺省 8，≤ 12），
 * 每块**重新组装一次 `HcPtrs`**（输入指针按行偏移、`ws` 指向该块的 arena 槽、`m` = 本块行数）后
 * 调**现有的** `M15H::HyperConnOp::ProcessAiv/ProcessAic`。
 * `m15_hc_layer.h` / `m15_hc_resources.h` 的**既有语义一字未动**（它们是 `lift_hc_segment.py`
 * 的机械产物，手改会破 `--check`）；本头只"外挂"一层循环。
 *
 * ## arena 的布局（**行页 / tile-chain**，`WS_HCP == 0` 是关键）
 * 块 t 的 `ws` = `arena + t * TILE_STRIDE`，`TILE_STRIDE = MT * HYPER * 2`（= MT 行多流残差流）。
 * 因为 `WS_HCP == 0`（`static_assert` 钉住），块 t 的 **H'** 恰好落在 arena 的第 `t*MT` 行起、
 * 行距 `HYPER` ⇒ **H' 在整个 arena 的前 `m` 行上是一块平铺的 `[m, HYPER]` 平面**（零拷贝，
 * 且直接就是下一个 hc 边界 / 下一层的 `hIn`：`MakeTilePtrs` 只按 `MT*HYPER*2` 推进指针）。
 *
 * 块的其余区（`WS_XN` … `WS_BLK`）是 scratch，落在同一块槽的尾部。三条"不互相踩"的性质都由
 * `m15_hc_resources.h` 的常量**编译期**推出（不靠人眼）：
 *   ① `WS_XN >= TILE_STRIDE`：本块的 scratch **起于**本块 H' 行之后 ⇒ 本块自己的后期段
 *      （S2/S5/S6）不会踩本块自己的 H'；块内**没有任何**区落在 `[0, MT*ROW_HP)` ⇒ H' 由本块写定；
 *   ② `SZ_HCP/(HYPER*2) = M_MAX >= MT`：块 t 的 scratch 伸进的是**更晚**块（t+ceil(M_MAX/MT) 起）
 *      的 H' 行 —— 更晚的块后写、后写者胜 ⇒ 只要块**按下标递增**执行，H' 平铺面的每一行
 *      最终都由它的属主块写定（t' 能覆盖块 t 的 H' 行的充要条件是 `t'*MT < t*MT + MT`
 *      ⇒ `t' <= t`，即只有属主自己）；
 *   ③ 反过来：`WS_BLK` / `WS_INJW` / `WS_RSTD` 三个区**一定**会被更晚的块覆盖 —— 由
 *      `CoveredByLater()` 在编译期**在全部写窗上取最小**搜出那个**最早**的覆盖者：
 *      · 块 t 的 BLK 内容（`WS_BLK` 起、`MT*ROW_HID` 长）落在块 **t+8** 的 GATE 写窗内
 *        （MT=8：`WS_BLK = 4,025,344`、`TILE_STRIDE = 163,840`、`WS_GATE = 2,714,624` ⇒ d = 8）；
 *      · 块 t 的 rstd/injw 内容落在块 **t+8** 的 **XN 写窗**内（`WS_XN = 1,310,720` ⇒
 *        d·TILE_STRIDE + WS_XN 恰好 = `WS_RSTD = 2,621,440` ⇒ d = 8）—— 注意**不是** d=16：
 *        只看 H' 窗会算出 d=16（`WS_RSTD` 也是 TILE_STRIDE 的整数倍），而 XN 窗**更早**覆盖它
 *        （判据侧 `check_ref.py::min_d` 在全部写窗上取最小、`check_main.log` 的 `d_rstd=8` 同此；
 *        复审 r1 的纯文字第 1 条点出的就是这处代数）。
 *      ⇒ 这四个张量**必须逐块重定位**（见 §4）：**块末**搬走，因为等到循环结束它们的槽
 *      早就被更晚的块写掉了（M119 实测：m=4097 档"循环后一次搬"读到的是覆盖者的数据 ——
 *      `blk` 逐位一致率 0.015、`rstd` maxRel 3.2e+00；改成块末搬后见 m27 的 evidence）。
 *
 * ## 包含前提（与 `m15_ple_wire.h` 同款，理由一样）
 * `m15_hc_layer.h` 是 lift 产物、**没有 include guard** ⇒ 本头**不自带**对它的 include
 * （二次包含会重定义）。要求 include 它的 TU **先**包含 `m15_hc_layer.h`（提供
 * `M15H::HcPtrs` / `HyperConnOp` / `BarrierAiv` / `BufAcquire` / `BufRelease` / `Block1` /
 * `MinU32`）与 AscendC 头。本头自带 guard、无 `main()`、用具名 namespace。
 *
 * ## 资源（融合清单的一份；见 `m27_hc_prefill/README.md` §融合清单）
 *   · UB：只在既有 `UB_BYTES_USED`(89,472) **之上**加一块 5,120 B 的行 staging
 *     （`UB_RELOC .. UB_RELOC + ROW_HID`，峰值 `UB_PEAK_PF = 94,592` ≤ 248 KB）；
 *   · L1 / L0C：**零新增**（全部沿用 hc 段的既有窗）；
 *   · BufferID：AIV 新增 1 个（`BUF_AIV_RELOC = 17`），AIC 零新增；
 *   · flagId：核内（AIV mode-0）复用 `FLAG_AV0 = 12` 做循环后的重定位 barrier —— **不新造号**。
 *
 * ## 接口约定（host 侧：为什么本头不提供"host 造 Plan"的函数）
 * `Plan` 的指针字段是 `__gm__ uint8_t*`（地址空间限定符）。实测：host 的 `void*` **不能**
 * `static_cast`/`reinterpret_cast` 成它（两种都报错）。本仓的既有约定是
 * 「**裸指针当 kernel 实参，由设备入口组装 struct**」（`m15_layer_kernel.h` 的
 * `M15L_LAYER_*_ARGS_FILL` 就是这个形态）⇒ 本头沿用它：设备入口调 `MakePlan(...)`（本头提供），
 * host 侧只算**整数几何**（`m15_hc_prefill_host.h`）。
 *
 * ## 未完成 / 未验证（显式披露，见 m27 的 README）
 *   · 本头只跑**一个边界**（H1 或 H2）；四相位/整层接线是 Wave C 的活（挂载点补丁文本见 README）。
 *   · 块的**跨核 flag 用量 = 每块一次**：现有预算表（`FlagMaxUse() <= 6`）是**无循环段**的预算。
 *     若硬件的 4bit 计数器是"累计"而不是"在飞"，513 块的档会撞上限（本头不改号：真实读数由
 *     `m27_hc_prefill` 的 m=4097 档给出，见 README §未完成项第 1 条）。
 *   · 未做：多核 AIC 打平（沿用 donor 的 N-tile 条带）、donor 里 S1「GM 取 injW」非确定性一节的
 *     归因（本头不碰那段逻辑，逐字沿用）。
 *   · **边界 #2 的 `bo` 生产者缺口（M138 登记）**：本头按 `Plan.bo` 消费"子层段出口"
 *     （`MakeTilePtrs` 的 `hp.bo`）。在 GDN-prefill 路径上，子层段 B1（`m15_gdn_prefill.h`）写的是
 *     `LayerArgs.wsO`（`[48,m,128] fp32`），而挂载点给边界 #2 传的是 `A.hcAttnOut`（`[m,HID] bf16`）
 *     —— 两者不同型，本 mission 在 prefill 路径上未读到两者间的生产者关系。完整登记与出错条件见
 *     `m27_hc_prefill/README.md` §5(e) 的"已知缺口"段（本头不代上层裁定来源）。
 */

#ifndef M15_HC_PREFILL_H
#define M15_HC_PREFILL_H

#include <cstdint>

#include "m15_hc_resources.h"   // M15H 的资源表（自带 guard，可安全二次包含）

// 「双 pass 可见」标记：`ascc` 预处理器把 `[aicore]` 的函数只编进设备侧、无标记的只编进 host 侧
// （实测：host 侧看不到 `__aicore__` 函数、设备侧看不到普通函数 —— 两个方向都报 "no matching
// function"）。本头的几何式（TileCount / TileRows / ArenaBytes / 各平面字节数）**两侧都要在
// 运行期调**（host 用来分配、设备用来切块）⇒ 用 `[host, aicore]` 标它。
// 普通 C++ 编译器（没有 ascc 预处理器）下这个宏未定义 ⇒ 退化为空，本头仍可被解析。
#ifndef __host_aicore__
#define __host_aicore__
#endif

namespace M15H {
namespace HcPF {

// ============================================================
// §0 块几何（编译期静态）
// ============================================================
// 块高：正文 §6.1 的 R3 建议 ≤ 8–12 行。上限由 **UB_IWTAB_SLOTS 的定尺**给出：块高必须 ≤ M_MAX。
// 可编译期覆盖（`-DM15_HC_PF_MT=<n>`）：m27 的第二个 target 用 MT=12 复跑同一批档。
#ifndef M15_HC_PF_MT
#define M15_HC_PF_MT 8u
#endif
constexpr uint32_t MT = M15_HC_PF_MT;
static_assert(MT >= 1u, "块高必须 ≥ 1 行");
static_assert(MT % 2u == 0u, "块高必须是偶数：rstd 按 32B 成对搬（见 §4 的对齐论证）");
static_assert(MT <= M_MAX, "块高必须 ≤ M_MAX：UB_IWTAB_SLOTS = M_MAX*HC 按 M_MAX 定尺（算术见下）");
static_assert(MT * HC <= UB_IWTAB_SLOTS, "块内的 injW 表槽数必须装进 UB_IWTAB_SLOTS");
static_assert(MT <= 12u, "块高超过 R3 建议的 ≤12 行（docs/15 §6.1）——若确有必要请另附依据");

constexpr uint32_t ROW_HP = HYPER * 2u;    // 一行多流残差流 = 20,480 B
constexpr uint32_t ROW_HID = HID * 2u;     // 一行单流残差流 = 5,120 B
constexpr uint32_t ROW_RSTD = HC * 4u;     // 一行 rstd = 16 B（4 个 fp32）
constexpr uint32_t ROW_INJW = HC * INJW_SLOT * 4u;   // 一行 injW = 128 B（4 个 32B 槽）
constexpr uint32_t TILE_STRIDE = MT * ROW_HP;
static_assert(TILE_STRIDE % 32u == 0u, "块跨距必须 32B 对齐（H' 平铺面的行距就是它）");
static_assert(TILE_STRIDE <= WS_XN, "块的 scratch 必须起于本块 H' 行之后（否则本块的 S2/S5/S6 会踩自己的 H'）");
static_assert(SZ_HCP / ROW_HP >= MT, "SZ_HCP 的块内行数必须 ≥ MT：若更小，块自己的 scratch 会踩自己的 H'");
static_assert(WS_HCP == 0u, "本设计依赖 WS_HCP == 0：块 t 的 ws 基址 = H' 平铺面的第 t*MT 行");
static_assert(WS_BLK + SZ_BLK == WS_BYTES, "WS_BLK 必须是最后一个区（arena 的块跨距按它定尾）");
static_assert(SZ_RSTD == M_MAX * ROW_RSTD, "rstd 区的定尺变了（重定位的槽宽按 HC*4 算）");
static_assert(SZ_INJW == M_MAX * ROW_INJW, "injw 区的定尺变了（重定位的槽宽按 HC*INJW_SLOT*4 算）");
static_assert(SZ_BLK == M_MAX * ROW_HID, "blk 区的定尺变了（重定位的行宽按 HID*2 算）");
// 「块 t 的这个区**一定**会被更晚的块覆盖」——机械见证（PROLOGUE §arena 的 ③ 的出处）。
// 判据：存在 `d >= 1`，使块 t 的区内容 `[off, off+bytes)` 落在块 t+d 的某个**写窗**
// `[d*TILE_STRIDE + winOff, + MT*ROW_HP)` 内（写窗 = 那一段实际会写满的 MT 行）。
__host_aicore__ inline constexpr bool CoveredByLater(uint32_t off, uint32_t bytes, uint32_t winOff)
{
    for (uint32_t d = 1u; d <= 64u; ++d) {
        const uint64_t s = static_cast<uint64_t>(d) * TILE_STRIDE + winOff;
        if (s <= off && (static_cast<uint64_t>(off) + bytes) <= (s + static_cast<uint64_t>(MT) * ROW_HP)) {
            return true;
        }
    }
    return false;
}
static_assert(CoveredByLater(WS_BLK, MT * ROW_HID, WS_GATE),
              "blk 区必须落在更晚块的 GATE 写窗内（否则'必须重定位'这条论证不成立）");
static_assert(CoveredByLater(WS_RSTD, (WS_INJW + MT * ROW_INJW) - WS_RSTD, WS_HCP),
              "rstd/injw 区必须落在更晚块的 H' 写窗内（同上）");
static_assert(WS_INJW == WS_RSTD + 1024u, "injw 区必须紧接在 rstd 之后（同一覆盖关系，1024 = rstd 区对齐后的大小）");

// 「抬包络」这条路的**存在性**断言：把 M_MAX 抬到 prefill 的真实 m（4097）时 UB 装不下。
// 它不是我们需要的性质，而是**mission 前提**的机械见证（docs/15 §M103-1.4 的算术逐字复算）。
constexpr uint32_t PREFILL_M_ACCEPT = 4097u;   // 验收档（不是接口上界）
constexpr uint32_t ENVELOPE_IWTAB_BYTES_MAX = M_MAX * HC * INJW_SLOT * 4u;                  // 64 行档 = 8,192 B
constexpr uint32_t ENVELOPE_IWTAB_BYTES_PREFILL = PREFILL_M_ACCEPT * HC * INJW_SLOT * 4u;   // 4097 档 = 524,416 B
static_assert(ENVELOPE_IWTAB_BYTES_MAX <= UB_BYTES_TOTAL, "64 行档必须装得下（否则既有语义已破）");
static_assert(ENVELOPE_IWTAB_BYTES_PREFILL > UB_BYTES_TOTAL,
              "4097 行档必须装不下：这条断言就是「必须 m-tile」的算术出处（524,416 B > 248 KB）");

// ---- 重定位 pass 的资源（只在本头新增的窗）----
constexpr uint32_t UB_RELOC = UB_BYTES_USED;             // 行 staging（靠既有窗之后）
constexpr uint32_t UB_RELOC_BYTES = ROW_HID;             // 5,120 B（按最长的一行：blk 行）
constexpr uint32_t UB_PEAK_PF = UB_RELOC + UB_RELOC_BYTES;
static_assert(UB_RELOC % 32u == 0u, "重定位窗必须 32B 对齐");
static_assert(UB_PEAK_PF <= UB_BYTES_TOTAL, "hc prefill 的 UB 峰值超过 248 KB");

constexpr uint32_t BUF_AIV_RELOC = BUF_AIV_IDS;          // = 17（沿用 0..16 之后的下一个；AIC 零新增）
static_assert(BUF_AIV_RELOC < 28u, "AIV BufferID 超出每核 28 个（MutexID 有效范围 0..27）");
static_assert(BUF_AIV_RELOC != BUF_AIV_H && BUF_AIV_RELOC != BUF_AIV_GB && BUF_AIV_RELOC != BUF_AIV_OB2,
              "重定位的 BufferID 不得与 hc 段的既有 id 撞号");

// 逐块重定位 barrier：AIV mode-0，取 **id 15**（不新造号：15 在 AIV mode-0 的池子里，
// 现由 MoE 段在**另一个相位**用；两者之间隔着相位边界 8，不相邻）。
// 相邻性（docs/05 §6.1：同一 (核型, mode) 的相邻同步点必须不同号）在本头的执行序上是：
//   每块内部 donor 的 AIV mode-0 = 12(FLAG_AV0) → 13(FLAG_AV1) → 14(FLAG_AV2)（stageLimit=7 全开）
//   → **15（本头的重定位 barrier）** → 下一块的 12 …
// ⇒ 相邻对 (14,15)、(15,12) 都不同号 ✓；MODE_COMBINE_ONLY 档 donor 只用 12 ⇒ (12,15) ✓。
// 融合档的相邻性（Wave C 复核）：本头的 15 → 相位边界 `FLAG_HC2_BOUND_AIV`(=8) → MoE 段的
// 首个 AIV mode-0（=12，见 `m15_layer_resources.h` 的 `FLAG_SEQ` 表）✓ 亦不同号。
constexpr uint16_t FLAG_RELOC = 15;
static_assert(FLAG_RELOC != FLAG_AV0 && FLAG_RELOC != FLAG_AV1 && FLAG_RELOC != FLAG_AV2,
              "重定位 barrier 与块内 donor 的 AIV mode-0 同步点相邻，必须不同号");
static_assert(FLAG_RELOC < 16u, "AIV mode-0 的 id 池是 0..15");
static_assert(FLAG_RELOC != FLAG_AV0, "重定位 barrier 与下一块的第一个同步点相邻，必须不同号");
static_assert(HC_STAGE_LIMIT >= 5u, "本头按 stageLimit = HC_STAGE_LIMIT 全开运行，最后一段（S5/S6）必须在内");

// ============================================================
// §1 Plan：段体的入参（**只吃「指针 + 标量」**，不吃 LayerArgs —— Wave B 共同契约第 1 条）
// ============================================================
// 平面约定（host 侧必须照此分配；字节单位）：
//   hIn / bo / ij  : 行平铺平面，`*TileStride` = 每块推进的字节数
//                    （平铺平面 = `MT * 行字节`；若 ij 取自上一个边界的 OH 列，则取 `TILE_STRIDE`）
//   ij 的**行距**  : `ijStride` 元素（独立 [m,16] 平面 = IJ_STRIDE；OH handoff = OH_W）
//   ij 的**过读**  : donor 的 `InjwStage` 恒读 M_MAX 行（不随 p.m 变）⇒ ij 平面必须能读到
//                    `块首行 + M_MAX` 行（host 侧补 M_MAX 行；见 host 头的 `IjPlaneBytes`）
//   arena          : 见 §0 的行页布局；必须 32B 对齐、长度 ≥ `ArenaBytes(m)`
//   blk / injw / rstd : **平铺**平面（行距 ROW_HID / ROW_INJW / ROW_RSTD）；可为 nullptr
//                    （nullptr = 跳过该张量的重定位 ⇒ 消费方须按 arena 的**阻塞**布局寻址）
//   ijFlat         : **平铺** [m, IJ_STRIDE] 平面（32 B/行，前 INJ_N 列有效）；可为 nullptr
//                    （只给"会产出 injection logits 的边界"用：mode != MIX/FINAL_MIX 时）
struct Plan {
    __gm__ uint8_t* hIn;
    __gm__ uint8_t* bo;
    __gm__ uint8_t* ij;
    __gm__ uint8_t* wDown;
    __gm__ uint8_t* wInj;
    __gm__ uint8_t* wUp;
    __gm__ uint8_t* hcNorm;
    __gm__ uint8_t* arena;
    __gm__ uint8_t* blk;
    __gm__ uint8_t* injw;
    __gm__ uint8_t* rstd;
    __gm__ uint8_t* ijFlat;   // 本边界把 `OH[:,OH_INJ:+INJ_N)` 重定位到这里（平铺 [m, IJ_STRIDE]，可 nullptr）
                              // ⇒ 下一个边界的 ij 直接取它（不再直接读 arena 的 OH 区，理由见 PROLOGUE ④）
    uint32_t hInTileStride;   // 字节/块
    uint32_t boTileStride;    // 字节/块
    uint32_t ijTileStride;    // 字节/块
    uint32_t ijStride;        // 元素/行
    uint32_t m;
    uint32_t mode;            // MODE_MIX / MODE_COMBINE_MIX / MODE_COMBINE_ONLY（MODE_FINAL_MIX 亦可用）
};

// ============================================================
// §2 几何（device 与 host 同源；host 侧在 m15_hc_prefill_host.h 里 **转发** 到这里）
// ============================================================
__host_aicore__ inline constexpr uint32_t TileCount(uint32_t m) { return (m + MT - 1u) / MT; }

__host_aicore__ inline constexpr uint32_t TileRow0(uint32_t t) { return t * MT; }

// 块 t 的行数（最后一块可以是 1..MT 行）
__host_aicore__ inline constexpr uint32_t TileRows(uint32_t m, uint32_t t)
{
    const uint32_t used = TileRow0(t);
    const uint32_t rest = m - used;   // 不调 M15H::MinU32：它是 `__aicore__`（host 侧看不到）
    return (m > used) ? ((rest < MT) ? rest : MT) : 0u;
}

// arena 的**最小**字节数：最后一块的槽 tail = WS_BYTES，此前每块 TILE_STRIDE。
// `m == 0` 单独返回 0：否则 `TileCount(0) - 1` 会在 **uint32** 上下溢成 4,294,967,295 ⇒
// 算出一个天文数字（本 mission 的路径碰不到 —— `Body` / `ValidateGeom` 都先拒 m=0 —— 但
// host 样例是"先算尺寸再校验"的形态，属潜在脚枪；复审 r1 的纯文字第 3 条点出，此处加固）。
__host_aicore__ inline constexpr uint64_t ArenaBytes(uint32_t m)
{
    if (m == 0u) {
        return 0u;
    }
    return static_cast<uint64_t>(TileCount(m) - 1u) * TILE_STRIDE + WS_BYTES;
}
static_assert(ArenaBytes(0u) == 0u, "m=0 的 arena 尺寸必须是 0（不是下溢后的巨值）");
static_assert(ArenaBytes(1u) == WS_BYTES, "m=1 的 arena 尺寸必须是单块槽（WS_BYTES）");

// 平铺平面的字节数（host 侧分配用）
__host_aicore__ inline constexpr uint64_t HpPlaneBytes(uint32_t m) { return static_cast<uint64_t>(m) * ROW_HP; }
__host_aicore__ inline constexpr uint64_t HidPlaneBytes(uint32_t m) { return static_cast<uint64_t>(m) * ROW_HID; }
__host_aicore__ inline constexpr uint64_t InjwPlaneBytes(uint32_t m)
{
    return static_cast<uint64_t>(m) * ROW_INJW;
}
// rstd 平面：**多一行**（`m+1`）—— §4 按 32B 成对搬，m 为奇数时最后一行会多写 16 B 到第 m 行。
__host_aicore__ inline constexpr uint64_t RstdPlaneBytes(uint32_t m)
{
    return (static_cast<uint64_t>(m) + 1u) * ROW_RSTD;
}

// ============================================================
// §3 Plan 组装（设备入口调；host 侧不构造 Plan —— 见 PROLOGUE §接口约定）
// ============================================================
__aicore__ inline Plan MakePlan(__gm__ uint8_t* hIn, __gm__ uint8_t* bo, __gm__ uint8_t* ij,
                                __gm__ uint8_t* wDown, __gm__ uint8_t* wInj, __gm__ uint8_t* wUp,
                                __gm__ uint8_t* hcNorm, __gm__ uint8_t* arena, __gm__ uint8_t* blk,
                                __gm__ uint8_t* injw, __gm__ uint8_t* rstd, __gm__ uint8_t* ijFlat,
                                uint32_t hInTileStride, uint32_t boTileStride, uint32_t ijTileStride,
                                uint32_t ijStride, uint32_t m, uint32_t mode)
{
    Plan p;
    p.hIn = hIn;
    p.bo = bo;
    p.ij = ij;
    p.wDown = wDown;
    p.wInj = wInj;
    p.wUp = wUp;
    p.hcNorm = hcNorm;
    p.arena = arena;
    p.blk = blk;
    p.injw = injw;
    p.rstd = rstd;
    p.ijFlat = ijFlat;
    p.hInTileStride = hInTileStride;
    p.boTileStride = boTileStride;
    p.ijTileStride = ijTileStride;
    p.ijStride = ijStride;
    p.m = m;
    p.mode = mode;
    return p;
}

// 平铺输入的缺省步长（host 与设备同源：由 MT 与行宽推出，不写第二份数字）
__host_aicore__ inline constexpr uint32_t FlatHInTileStride() { return MT * ROW_HP; }
__host_aicore__ inline constexpr uint32_t FlatBoTileStride() { return MT * ROW_HID; }
__host_aicore__ inline constexpr uint32_t FlatIjTileStride(uint32_t ijStride) { return MT * ijStride * 2u; }
// 块内 OH 区里 injection 列（前 INJ_N 个 bf16 = 8 B）的字节偏移：`WS_OH + OH_INJ*2`。
// 它是**重定位的源**（`RelocateTile` 把 32 B/行搬进 `ijFlat`），不是给消费方的地址。
__host_aicore__ inline constexpr uint32_t OhInjOff() { return WS_OH + OH_INJ * 2u; }
// ij 平铺面的行字节：IJ_STRIDE 个 bf16 = 32 B（32B 对齐，源同样 32B 对齐 ⇒ 可直接 DataCopy）
__host_aicore__ inline constexpr uint32_t IjRowBytes() { return IJ_STRIDE * 2u; }
static_assert((IJ_STRIDE * 2u) == 32u, "ij 平铺面按 32 B/行定尺（OH 列重定位按 32B 块搬）");
static_assert(((WS_OH + OH_INJ * 2u) % 32u) == 0u, "OH 的 inj 列必须 32B 对齐（重定位按 32B 块搬）");
static_assert(((OH_W * 2u) % 32u) == 0u, "OH 的行距必须 32B 对齐（重定位按 32B 块搬）");

// 块 t 的 HcPtrs（**本头的核心**：既有的段函数只认 HcPtrs，这里把"行偏移 + 块槽 + 块行数"填进去）
__aicore__ inline HcPtrs MakeTilePtrs(const Plan& p, uint32_t t)
{
    HcPtrs hp;
    hp.hIn = p.hIn + static_cast<uint64_t>(t) * p.hInTileStride;
    hp.bo = p.bo + static_cast<uint64_t>(t) * p.boTileStride;
    hp.ij = p.ij + static_cast<uint64_t>(t) * p.ijTileStride;
    hp.wDown = p.wDown;
    hp.wInj = p.wInj;
    hp.wUp = p.wUp;
    hp.hcNorm = p.hcNorm;
    hp.ws = p.arena + static_cast<uint64_t>(t) * TILE_STRIDE;
    hp.m = TileRows(p.m, t);
    hp.ijStride = p.ijStride;
    hp.mode = p.mode;
    hp.stageLimit = HC_STAGE_LIMIT;
    return hp;
}

// ============================================================
// §4 重定位 pass（循环后一次性；DMA 行拷贝 GM→UB→GM）
// ============================================================
// 为什么必须重定位：`blk` / `injw` / `rstd` 三个区落在块槽的尾部，会被**更晚的块**覆盖
// （代数见 PROLOGUE §arena 的 ③：`CoveredByLater` 搜出的 d 是 8 / 16）。
// **为什么是"块末搬"而不是"循环后一次搬"**：块 t 的这三个区在块 t+8（BLK）/ t+16（rstd、injw）
// 写下去的那一刻就被覆盖了 ⇒ 循环结束时它们的槽里已经**不是**块 t 的产物（M119 实测读数见
// PROLOGUE §arena 的 ③）⇒ 只能在自己的块里、趁自己还没被覆盖时搬走（这一档的判据 = m27 的
// `m4097` 档 + `m257` 档：两者都能红/绿区分两种做法）。
// 可见性：源数据由**别的 AIV** 的 MTE3 写出 ⇒ 进入本 pass 前需要一个**全体 AIV** 的 mode-0
// barrier（set 挂 MTE3 = 写 GM 排空；wait 用 MTE2），这正是 donor 的 `BarrierAiv<PIPE_MTE2, FLAG>`。
// 对齐：blk 行 5,120 B（32B 整数倍 ✓）、injw 行 128 B ✓ → 逐行 `DataCopy`（blockLen 单位 = 32B）；
// rstd 行只有 16 B ⇒ 按**成对**（行 2j、2j+1）搬 32 B：源 `WS_RSTD + i*16` 与目的 `r*16` 在
// `i`、`r` 同奇偶时都 32B 对齐，而 `r = t*MT + i` 且 `MT` 为偶数（§0 的 static_assert）⇒
// 「r 为偶数 ⟺ i 为偶数」恒成立 ⇒ 只有**偶数**行起始的 32B 拷贝是合法的（本 pass 就只走这些）。
// m 为奇数时最后一行（下标 m-1，仍是偶数）的 32B 拷贝会多写 16 B 到第 m 行 ⇒ rstd 平面按 m+1
// 行分配（见 `RstdPlaneBytes`）。
// 规则 ⓔ：**落 GM 一律 DMA**；本 pass 不写任何标量（标量证据同样经 UB + 阻塞释放交接）。
__aicore__ inline void DmaBytes(__gm__ uint8_t* dst, AscendC::LocalTensor<uint8_t>& buf, __gm__ uint8_t* src,
                                uint32_t bytes)
{
    AscendC::GlobalTensor<uint8_t> srcG;
    AscendC::GlobalTensor<uint8_t> dstG;
    srcG.SetGlobalBuffer(src, static_cast<uint64_t>(bytes));
    dstG.SetGlobalBuffer(dst, static_cast<uint64_t>(bytes));

    PipeBarrier<PIPE_MTE2>();   // 同 pipe 复用同一 UB 行缓冲：任何模式的 BufferID 都不保序
    BufAcquire<PIPE_MTE2>(BUF_AIV_RELOC);
    AscendC::DataCopy(buf, srcG[0], Block1(bytes));
    BufRelease<PIPE_MTE2>(BUF_AIV_RELOC);   // 阻塞释放：MTE2 排空后 token 才归 MTE3

    BufAcquire<PIPE_MTE3>(BUF_AIV_RELOC);
    AscendC::DataCopy(dstG[0], buf, Block1(bytes));
    BufRelease<PIPE_MTE3>(BUF_AIV_RELOC);
}

// ------------------------------------------------------------
// **重定位门限的唯一权威**（P2-1 的根因：`Body` 的"要不要搬"与 `RelocateTile` 的"能不能早退"
// 曾经是**两处独立写的条件**，且已经漂移 —— 前者漏了 `p.ijFlat` ⇒ "只传 ij handoff 面、
// 其余三个留 nullptr"这种**文档允许**的配置会连 `BarrierAiv` 都不做、ij 面保持毒值且不 Trap）。
// 处置分两层，**"漂移"这一类结构上被去掉**：
//   · 调用侧只有一个入口 `RelocatePass(p, t)`（它内部**无条件**检查门限）—— `Body` 里**不再有**
//     任何条件（原先的 `if (blk || injw || rstd)` 已删）⇒ 没有第二处可以写错的地方；
//   · 谓词本身收 4 个 bool（host 与设备两侧都能编译期求值），`m27_hc_prefill.asc` 把**全部 16 种
//     配置**钉成 `static_assert`（零设备、编译期覆盖）。**该断言组做过反向验证**：把谓词改回
//     "漏 ijFlat" 的旧语义 ⇒ 构建在 `Ok<0x8>()` 上报错（读数见 m27 的 README §4.8）。
// ------------------------------------------------------------
constexpr uint32_t RELOC_TARGET_N = 4u;   // blk / injw / rstd / ijFlat

__host_aicore__ inline constexpr bool NeedReloc(bool hasBlk, bool hasInjw, bool hasRstd, bool hasIjFlat)
{
    return hasBlk || hasInjw || hasRstd || hasIjFlat;
}

__aicore__ inline bool NeedReloc(const Plan& p)
{
    return NeedReloc(p.blk != nullptr, p.injw != nullptr, p.rstd != nullptr, p.ijFlat != nullptr);
}

// 搬**一块**（块 t）的四个量：`blk` / `injw` / `rstd` / `ijFlat`（OH inj 列）。行对（块内 0,1 / 2,3 / …）按 `bid / nAiv` 分派；
// 块内行号 `i` 恒为偶起步（`MT` 为偶数 ⇒ 全局行 `t*MT+i` 与 `i` 同奇偶）。
__aicore__ inline void RelocateTile(const Plan& p, uint32_t t)
{
    AscendC::LocalTensor<uint8_t> buf(TPosition::VECCALC, UB_RELOC, UB_RELOC_BYTES);
    const uint32_t bid = AscendC::GetBlockIdx();
    const uint32_t nAiv = AscendC::GetBlockNum() * 2u;
    const uint32_t mT = TileRows(p.m, t);
    const uint32_t base = t * MT;                        // 本块第 0 行的全局行号
    const uint64_t tb = static_cast<uint64_t>(t) * TILE_STRIDE;
    const uint32_t nPair = (mT + 1u) / 2u;
    for (uint32_t j = bid; j < nPair; j += nAiv) {
        const uint32_t i0 = 2u * j;                      // 块内偶数行
        if (p.blk != nullptr) {
            for (uint32_t k = 0; k < 2u; ++k) {
                if (i0 + k >= mT) {
                    break;
                }
                DmaBytes(p.blk + static_cast<uint64_t>(base + i0 + k) * ROW_HID, buf,
                         p.arena + tb + WS_BLK + static_cast<uint64_t>(i0 + k) * ROW_HID, ROW_HID);
            }
        }
        if (p.injw != nullptr) {
            for (uint32_t k = 0; k < 2u; ++k) {
                if (i0 + k >= mT) {
                    break;
                }
                DmaBytes(p.injw + static_cast<uint64_t>(base + i0 + k) * ROW_INJW, buf,
                         p.arena + tb + WS_INJW + static_cast<uint64_t>(i0 + k) * ROW_INJW, ROW_INJW);
            }
        }
        if (p.ijFlat != nullptr) {
            // 本块的 OH inj 列（32 B/行：4 个有效 bf16 + 12 个 padding bf16，donor 只用前 4 个）
            // → 平铺 ij 面。源 `WS_OH + OH_INJ*2 + i*OH_W*2` 与目的 `(base+i)*32` 都 32B 对齐
            // （三条 static_assert 钉住）⇒ 直接 DataCopy，无需 DataCopyPad。
            for (uint32_t k = 0; k < 2u; ++k) {
                if (i0 + k >= mT) {
                    break;
                }
                DmaBytes(p.ijFlat + static_cast<uint64_t>(base + i0 + k) * IjRowBytes(), buf,
                         p.arena + tb + OhInjOff() + static_cast<uint64_t>(i0 + k) * (OH_W * 2u),
                         IjRowBytes());
            }
        }
        if (p.rstd != nullptr) {
            // 成对搬 32 B（`i0`、`base+i0` 都偶数 ⇒ 两端 32B 对齐；见 §4 的论证）。
            // mT 为奇数（只可能是最后一块）时，这一搬会多读/多写 16 B：
            // 读 = 本块槽的下一行（arena 内有值、无害），写 = rstd 平面的第 base+mT 行
            // （该行归下一块写；最后一块则落在 `RstdPlaneBytes` 多给的那一行里）。
            DmaBytes(p.rstd + static_cast<uint64_t>(base + i0) * ROW_RSTD, buf,
                     p.arena + tb + WS_RSTD + static_cast<uint64_t>(i0) * ROW_RSTD, 2u * ROW_RSTD);
        }
    }
}

// ------------------------------------------------------------
// **唯一的调用入口**（P2-1）：门限 + 全体 AIV barrier + 拷贝都在这里，调用方（`Body`）无条件调它。
// 于是"忘记考虑某个目标"这件事**没有第二处可以发生**。
// ------------------------------------------------------------
__aicore__ inline void RelocatePass(const Plan& p, uint32_t t)
{
    if (!NeedReloc(p)) {
        return;   // 四个目标都空 = 调用方**明确**不要任何重定位（判据侧把"没搬"记 SKIP + 毒值见证）
    }
    // **块末**搬：全体 AIV 到齐（set 挂 MTE3 ⇒ 本块所有 GM 写排空）后立刻搬走，
    // 否则本块这四个区会被更晚的块覆盖（PROLOGUE §arena 的 ③④）。
    BarrierAiv<PIPE_MTE2, FLAG_RELOC>();
    RelocateTile(p, t);
}

// ============================================================
// §5 段体：m-tile 主循环（**这是 B5 的交付物**）
// ============================================================
// 调用形态：设备入口填好 `Plan` 后调 `Body(p)`（入口自己 `if ASCEND_IS_AIV/AIC` 分核）。
// **两侧必须都进入**：核间 mode-2 交接要求 AIC 与配对的 2 个 AIV 按同一块序前进
// （本循环对两侧对称，块序由 `p.m` 唯一决定）。
__aicore__ inline void Body(const Plan& p)
{
    if ((reinterpret_cast<uint64_t>(p.arena) & 31u) != 0u) {
        AscendC::Trap();   // arena 必须 32B 对齐（H' 平铺面的行距是 32B 的倍数）
    }
    if (p.m == 0u) {
        AscendC::Trap();   // 本段不接受 m = 0（调用方在 m=0 时应整段跳过）
    }
    if (p.mode >= MODE_COUNT) {
        AscendC::Trap();   // mode 越界（host 侧另有同口径的 ValidateGeom）
    }
    const uint32_t nT = TileCount(p.m);
    HyperConnOp op;
    for (uint32_t t = 0; t < nT; ++t) {
        const HcPtrs hp = MakeTilePtrs(p, t);
        op.Init(hp);   // 每块重新绑定：输入按行偏移、ws 指向本块槽、m = 本块行数
        if ASCEND_IS_AIV {
            op.ProcessAiv();
        }
        if ASCEND_IS_AIC {
            op.ProcessAic();
        }
        if ASCEND_IS_AIV {
            // **无条件**调（门限在 RelocatePass 内部）⇒ Body 里不存在第二个条件（P2-1 的结构性修法）
            RelocatePass(p, t);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

}  // namespace HcPF
}  // namespace M15H

#endif  // M15_HC_PREFILL_H
