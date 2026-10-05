/**
 * m15_moe_prefill_host.h —— MoE **prefill** 段（B4）的 host 侧接线面
 *
 * 范围：**host 侧**的定尺、组板、启动参数打包与「融合清单」/挂载点补丁文本。
 * 本文件不定义任何设备段体（那些在 `m15_moe_prefill.h`），也不含 `main()`。
 *
 * 融合（挂进 `m15_layer_loop.asc` 的 host 段）时，本文件的**唯一必需**动作是：
 *   ① 在 `Ctx` 里按 `M15PFH::WsBytes()` 分配 prefill MoE 的 ws（或复用 `.asc` 现有的 moe ws 平面）；
 *   ② 按 `M15PFH::BuildRouterWPad()` 组板 router 权重（**一次，可缓存**）；
 *   ③ 把 13 个指针 + 3 个标量填进 `M15PFR` 的启动参数（见 §3 的 `Launch`）；
 *   ④ 在 prefill 入口的 `.asc` 分支里调用 `m26` 式的 launch（或用 Wave C 加进 `.asc` 的开关）。
 *
 * 「融合清单」（`docs/15` §M103-2.7 的 (a)–(f) 固定小标题）在
 * `m26_moe_prefill/README.md` §1；本节只给**可直接用的挂载点补丁文本**（§4）。
 */

#ifndef M15_MOE_PREFILL_HOST_H
#define M15_MOE_PREFILL_HOST_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "m15_moe_prefill.h"   // 设备头（同一 TU）：MoePrefillPtrs / M15PFR 的资源常量

namespace M15PFH {

// ============================================================
// 1. 定尺（host 侧唯一来源；与设备侧 constexpr 同源，不另算一份）
// ============================================================
constexpr size_t WsBytes() { return static_cast<size_t>(M15PFR::PWS_BYTES); }
constexpr uint32_t RowBytes() { return M15PFR::HIDDEN * 2u; }                 // bf16 一行
constexpr uint32_t RouterWPadBytes() { return M15PFR::E_PAD * M15PFR::HIDDEN * 2u; }
constexpr uint32_t RouterWPadRowBytes() { return RowBytes(); }

// ============================================================
// 2. router 权重组板：`[E, HIDDEN] bf16` 专家 + `[1, HIDDEN] bf16` 共享门
//    → `[E_PAD, HIDDEN] bf16`（**第 E 行 = 共享门**；其余 pad 行**复制共享门行**）
//    为什么 pad 也要填真字节：mmad 的 N 方向按 `RB_N=128` 无尾块 ⇒ 会读到行 513..639；
//    那些列的结果本段**不使用**（AIV 只取列 0..E），但必须保证读的是**已分配且可读**的内存。
inline bool BuildRouterWPad(const uint8_t* expertW, size_t expertBytes, const uint8_t* sgateW,
                            size_t sgateBytes, std::vector<uint8_t>& out)
{
    const size_t row = RouterWPadRowBytes();
    if (expertBytes != static_cast<size_t>(M15PFR::E) * row || sgateBytes != row) {
        return false;
    }
    out.assign(static_cast<size_t>(M15PFR::E_PAD) * row, 0u);
    std::memcpy(out.data(), expertW, expertBytes);
    std::memcpy(out.data() + static_cast<size_t>(M15PFR::E) * row, sgateW, row);
    for (uint32_t r = M15PFR::E + 1u; r < M15PFR::E_PAD; ++r) {
        std::memcpy(out.data() + static_cast<size_t>(r) * row, sgateW, row);
    }
    return true;
}

// ============================================================
// 3. 启动参数（与设备侧 `M15MP::MoePrefillPtrs` 逐字段同序）
// ============================================================
struct Launch {
    void* ws = nullptr;
    void* xLayer = nullptr;     // [MT, HIDDEN] bf16
    void* yLayer = nullptr;     // [MT, HIDDEN] bf16
    void* resZero = nullptr;    // [MT, HIDDEN] bf16 全 0
    void* gamma1 = nullptr;     // [HIDDEN] bf16
    void* gamma2 = nullptr;     // [HIDDEN] bf16
    void* routerWpad = nullptr; // [E_PAD, HIDDEN] bf16
    void* wGu = nullptr;
    void* sGu = nullptr;
    void* wDn = nullptr;
    void* sDn = nullptr;
    void* wGuShd = nullptr;
    void* sGuShd = nullptr;
    void* wDnShd = nullptr;
    void* sDnShd = nullptr;
    uint32_t m = 1;             // 本 tile 行数（1..MT）
    uint32_t topk = M15PFR::TOPK;
    uint32_t stageLimit = 8u;   // 8 = 全链
};

// 打包成设备侧 POD（逐字段赋值，**不用 memcpy 避免布局漂移**）
inline M15MP::MoePrefillPtrs Pack(const Launch& a)
{
    M15MP::MoePrefillPtrs p = {};
    p.ws = static_cast<__gm__ uint8_t*>(a.ws);
    p.xLayer = static_cast<__gm__ uint8_t*>(a.xLayer);
    p.yLayer = static_cast<__gm__ uint8_t*>(a.yLayer);
    p.resZero = static_cast<__gm__ uint8_t*>(a.resZero);
    p.gamma1 = static_cast<__gm__ uint8_t*>(a.gamma1);
    p.gamma2 = static_cast<__gm__ uint8_t*>(a.gamma2);
    p.routerWpad = static_cast<__gm__ uint8_t*>(a.routerWpad);
    p.wGu = static_cast<__gm__ uint8_t*>(a.wGu);
    p.sGu = static_cast<__gm__ uint8_t*>(a.sGu);
    p.wDn = static_cast<__gm__ uint8_t*>(a.wDn);
    p.sDn = static_cast<__gm__ uint8_t*>(a.sDn);
    p.wGuShd = static_cast<__gm__ uint8_t*>(a.wGuShd);
    p.sGuShd = static_cast<__gm__ uint8_t*>(a.sGuShd);
    p.wDnShd = static_cast<__gm__ uint8_t*>(a.wDnShd);
    p.sDnShd = static_cast<__gm__ uint8_t*>(a.sDnShd);
    p.m = a.m;
    p.topk = a.topk;
    p.stageLimit = a.stageLimit;
    return p;
}

}  // namespace M15PFH

// ============================================================
// 4. 可直接用的挂载点补丁文本（给 Wave C / the fused TU 的接线者）
//    **本 mission 不改 `m15_layer_kernel.h` / `m15_layer_loop.asc`**（不在 scope），
//    下面两段是逐字可贴的补丁；贴完后 prefill MoE 段即接进融合 kernel。
// ============================================================
//
// ---- 补丁 A：`m15_layer_kernel.h` 的 prefill 体（插入点 = **B5 之后、贴着 `FLAG_HC2_BOUND_AIV`**）----
// **r1 复审 P2-3 订正**：kernel 自己在 `m15_layer_kernel.h:598-599` 登记的顺序是
//   `M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>();  /* B5 的 hc prefill */`
//   `M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>();  /* B4 的 MoE prefill */`
// 与两相位形态的 `:778`（`M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>(); moe.ProcessAiv();`）一致
// ⇒ **B4 不能直接接在相位 A 之后**（那会插到 B5 前面），必须贴着 `FLAG_HC2_BOUND_AIV` 插。
// 若 B5 尚未接线，本段实际成为相位 A 之后的第一段，但**仍应保留这条相位边界**（它是 AIV 的
// 全体 mode-0 barrier，与 hc 是否接线无关；`FLAG_HC2_BOUND_AIV == M15M::FLAG_L0_BOUND_AIV`，
// 已由 `m15_layer_resources.h:583` 的 static_assert 钉住）。
//
//   在 `M15L_PrefillBody<KIND>` 里、**B5 的 `M15L_PhaseBoundaryAiv<FLAG_HC0_BOUND_AIV>()` 之后**加：
//
//   M15L_PhaseBoundaryAiv<FLAG_HC2_BOUND_AIV>();   // 相位边界（B5 → B4）
//   M15L_PrefillPhaseB<KIND>(A);                   // B4：MoE prefill 段
//
//   template <uint32_t KIND>
//   __aicore__ inline void M15L_PrefillPhaseB(const M15L_LayerArgs& A)
//   {
//       if (A.pfWired == 0u) {
//           return;                      // 默认关：decode 的 runs=all 零回归（M103-2.7 的硬要求）
//       }
//       M15MP::MoePrefillPtrs mp = {};
//       mp.ws         = A.ws;
//       mp.xLayer     = A.xLayer;        // 层输入（本 tile 的 MT 行）
//       mp.yLayer     = A.yLayer;
//       mp.resZero    = A.resZero;
//       mp.gamma1     = A.moeGamma1;
//       mp.gamma2     = A.moeGamma2;
//       mp.routerWpad = A.pfMoeRouterWPad;   // ← Wave A 需在 LayerArgs 里加这个字段（见 README §1(b)）
//       mp.wGu        = A.moeWGu;   mp.sGu = A.moeSGu;   mp.wDn = A.moeWDn;   mp.sDn = A.moeSDn;
//       mp.wGuShd     = A.moeWGuShd; mp.sGuShd = A.moeSGuShd;
//       mp.wDnShd     = A.moeWDnShd; mp.sDnShd = A.moeSDnShd;
//       mp.m          = A.m;
//       mp.topk       = A.topk;
//       mp.stageLimit = 8u;
//       M15MP::MoePrefillChain moe;
//       moe.Init(mp);
//       if ASCEND_IS_AIV { moe.ProcessAiv(); }
//       if ASCEND_IS_AIC { moe.ProcessAic(); }
//       AscendC::PipeBarrier<PIPE_ALL>();
//   }
//
//   **多块（m > MT）**：`MoePrefillChain` 现按**单块**编排；m > 64 时调用方要在本函数里
//   按 `blk = 0..ceil(m/MT)-1` 循环（每块 `curM = min(MT, m - blk*MT)`，x/y 指针按 blk*MT 行偏移，
//   重新 `Init` 后 `ProcessAiv/ProcessAic`，两侧块数必须一致）——见 README §5.4。
//
// ---- 补丁 B：`m15_layer_loop.asc` 的 host 段（在 `runs=prefill` 分支里补齐 MoE 的权重与组板）----
//
//   // router 权重组板（一次，可缓存到 Ctx）
//   std::vector<uint8_t> rwPad;
//   M15PFH::BuildRouterWPad(/*expertW=*/C.moeRouterW, /*expertBytes=*/E*HIDDEN*2,
//                           /*sgateW=*/C.moeSgateW,  /*sgateBytes=*/HIDDEN*2, rwPad);
//   void* rwPadDev = nullptr;
//   aclrtMalloc(&rwPadDev, rwPad.size(), ACL_MEM_MALLOC_HUGE_FIRST);
//   aclrtMemcpy(rwPadDev, rwPad.size(), rwPad.data(), rwPad.size(), ACL_MEMCPY_HOST_TO_DEVICE);
//   // 挂载点字段：LayerArgs.pfMoeRouterWPad = rwPadDev;（Wave A 加字段；见 README §1(b)）
//
//    注：prefill 档的 13 个 `LayerArgs` 字段（M110 已加）里，B4 **需要新增 1 个**：
//    `pfMoeRouterWPad`（pad 后的 router 权重平面）。其余 12 个沿用 decode 的 MoE 权重指针。

#endif  // M15_MOE_PREFILL_HOST_H
