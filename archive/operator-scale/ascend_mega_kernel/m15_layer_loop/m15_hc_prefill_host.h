// ============================================================
// m15_hc_prefill_host.h —— M119（Wave B5）hc prefill 的 **host 侧几何 / 校验 / 布局落盘**
//
// 本头只在 host 侧用（被 `m27_hc_prefill/m27_hc_prefill.asc` 与将来融合 TU 的 host 段 include）。
// 它做三件事，且**不复制第二份数字**（全部转发 `m15_hc_prefill.h` 的 constexpr）：
//   1. 平面/arena 的字节数（分配用）+ 三个"块推进量"（`Plan` 的标量字段）；
//   2. `GrowGeom()` 家族 + `ValidateGeom()`：**整数口径**的响亮失败（对齐 / 行数 / 步长 / mode 逐条查）；
//   3. `DumpLayout()`：把 arena 的行页几何 + 全部 WS 偏移写成 `key=value` 文本，
//      供独立判据（`m27_hc_prefill/check_ref.py`）**算出设备地址**，从而不复写第二份布局。
//
// **为什么本头不构造 `M15H::HcPF::Plan`**：`Plan` 的指针字段是 `__gm__ uint8_t*`（地址空间
// 限定符），host 的 `void*` **不能** `static_cast`/`reinterpret_cast` 成它（两条都实测报错）。
// 本仓的既有约定是「**裸指针当 kernel 实参，由设备入口组装 struct**」（`m15_layer_kernel.h` 的
// `M15L_LAYER_*_ARGS_FILL` 即此形态）⇒ 本头只给**标量**，设备入口调 `M15H::HcPF::MakePlan(...)`。
//
// **为什么不把权重装载放进来**：hc 段的权重槽装配/来源校验已由 `m15_layer_loop/m15_hc_host.h`
// 提供（`M15HcW` + `H_HcWAlloc` + `H_GenHcW`），本头不复制；独立验证路（m27）自带一份**最小**
// 装载器（manifest 只读 + 合成两档），见 m27 的 `.asc`。
//
// **包含顺序**：本头 include `m15_hc_prefill.h`（后者要求 TU 已先包含 `m15_hc_layer.h`，因为它是
// lift 产物、没有 include guard）。
// ============================================================

#ifndef M15_HC_PREFILL_HOST_H
#define M15_HC_PREFILL_HOST_H

#include <cstdint>
#include <cstdio>
#include <string>

#include "m15_hc_prefill.h"   // 布局数学的**唯一源**（host 侧只转发）

namespace M15L {
namespace HcPfH {

// ---- 转发：形状与块几何 ----
constexpr uint32_t MT = M15H::HcPF::MT;
constexpr uint32_t ROW_HP = M15H::HcPF::ROW_HP;
constexpr uint32_t ROW_HID = M15H::HcPF::ROW_HID;
constexpr uint32_t ROW_RSTD = M15H::HcPF::ROW_RSTD;
constexpr uint32_t ROW_INJW = M15H::HcPF::ROW_INJW;
constexpr uint32_t TILE_STRIDE = M15H::HcPF::TILE_STRIDE;

// 512 对齐（host 侧分配对齐；hc 的 ws 常量在 m15_layer_resources.h 里也是 512 对齐惯例）
constexpr uint64_t Align512(uint64_t x) { return (x + 511u) / 512u * 512u; }

inline uint64_t ArenaBytes(uint32_t m) { return M15H::HcPF::ArenaBytes(m); }
inline uint64_t ArenaBytesAligned(uint32_t m) { return Align512(ArenaBytes(m)); }
inline uint32_t TileCount(uint32_t m) { return M15H::HcPF::TileCount(m); }
// ij 平面的最小字节数：**必须含 donor 恒读 M_MAX 行的过读余量**（`InjwStage` 的行循环上界是
// `M_MAX`，不随 `p.m` 变）⇒ 分配 `m + M_MAX` 行。少分配会读到平面外的内存。
inline uint64_t IjPlaneBytes(uint32_t m, uint32_t ijStride)
{
    return (static_cast<uint64_t>(m) + M15H::M_MAX) * ijStride * 2u;
}
inline uint64_t HpPlaneBytes(uint32_t m) { return M15H::HcPF::HpPlaneBytes(m); }
inline uint64_t HidPlaneBytes(uint32_t m) { return M15H::HcPF::HidPlaneBytes(m); }
inline uint64_t InjwPlaneBytes(uint32_t m) { return M15H::HcPF::InjwPlaneBytes(m); }
inline uint64_t RstdPlaneBytes(uint32_t m) { return M15H::HcPF::RstdPlaneBytes(m); }
// OH handoff 形态的 ij 平铺面（= 上一个边界重定位出来的 `[m, IJ_STRIDE]` 面）：
// 字节数与"独立 ij 平面"同式（`IjPlaneBytes(m, IJ_STRIDE)`，含 donor 恒读 M_MAX 行的过读余量）。
inline uint64_t IjFlatPlaneBytes(uint32_t m) { return IjPlaneBytes(m, M15H::IJ_STRIDE); }

// ============================================================
// 几何（= Plan 的标量部分；host 与设备读同一批 constexpr）
// ============================================================
struct Geom {
    uint32_t m = 1;
    uint32_t mode = M15H::MODE_COMBINE_MIX;
    uint32_t hInTileStride = 0;
    uint32_t boTileStride = 0;
    uint32_t ijTileStride = 0;
    uint32_t ijStride = M15H::IJ_STRIDE;
};

// case A：hIn / bo / ij 都是**行平铺**平面（ij 是独立 [m,16] 平面）
inline Geom GeomFlat(uint32_t m, uint32_t mode, uint32_t ijStride = M15H::IJ_STRIDE)
{
    Geom g;
    g.m = m;
    g.mode = mode;
    g.ijStride = ijStride;
    g.hInTileStride = M15H::HcPF::FlatHInTileStride();
    g.boTileStride = M15H::HcPF::FlatBoTileStride();
    g.ijTileStride = M15H::HcPF::FlatIjTileStride(ijStride);
    return g;
}

// ============================================================
// 响亮失败：几何自检（在 host 侧一眼看出"分配/步长写错了"，而不是等设备上跑出错数）
// ============================================================
// 一块设备缓冲的"分配事实"（host 侧只知道字节数 + 基址对齐；指针本身不进本头，见 §接口约定）
struct Alloc {
    uint64_t bytes = 0;
    bool aligned = false;   // 基址是否 32B 对齐（rstd 的成对搬要求 32B）
};

// `legacyIj`：**只给负向对照档**（`M27_MUTANT=ijfromoh`）放行"ij 直接读 arena 的 OH 列"
// （`ijStride = OH_W` + `ijTileStride = TILE_STRIDE`）这条**已撤回**的旧路径。正常路径必须
// `ijStride = IJ_STRIDE`（= 平铺重定位面）—— 见 `m15_hc_prefill.h` 的 PROLOGUE ④。
inline std::string ValidateGeom(const Geom& g, const Alloc& arena, const Alloc& blk, const Alloc& injw,
                                const Alloc& rstd, bool legacyIj = false)
{
    auto hex = [](uint64_t v) {
        char b[32];
        std::snprintf(b, sizeof(b), "0x%llx", static_cast<unsigned long long>(v));
        return std::string(b);
    };
    if (g.m == 0u) return "m = 0：本段不接受（调用方在 m=0 时应整段跳过）";
    if (g.mode >= M15H::MODE_COUNT) return "mode 越界（>= MODE_COUNT）";
    if (g.hInTileStride == 0u || g.boTileStride == 0u || g.ijTileStride == 0u) return "块推进量不得为 0";
    if ((g.hInTileStride % 32u) != 0u || (g.boTileStride % 32u) != 0u || (g.ijTileStride % 32u) != 0u)
        return "三个块推进量必须是 32B 的整数倍";
    if ((g.ijStride * 2u) % 32u != 0u) return "ij 的行距必须是 32B 的整数倍（donor 按行 32B 搬运）";
    if (g.ijStride != M15H::IJ_STRIDE && !(legacyIj && g.ijStride == M15H::OH_W))
        return "ij 行距只支持 IJ_STRIDE（= 16）：OH 列的 handoff 也走「重定位到平铺 ij 面」这条路"
               "（理由：arena 的 OH 区不持久，见 m15_hc_prefill.h 的 PROLOGUE ④）";
    if (!arena.aligned) return "arena 必须 32B 对齐";
    if (arena.bytes < M15H::HcPF::ArenaBytes(g.m))
        return "arena 分配不足：需要 " +
               std::to_string(static_cast<unsigned long long>(M15H::HcPF::ArenaBytes(g.m))) + " 字节，实际 " +
               hex(arena.bytes);
    // 三个**平铺**输出平面（只在传了重定位目标时查；bytes == 0 = 不重定位）
    if (blk.bytes != 0u && (!blk.aligned || blk.bytes < M15H::HcPF::HidPlaneBytes(g.m)))
        return "blk 平面必须 32B 对齐且 ≥ m*ROW_HID";
    if (injw.bytes != 0u && (!injw.aligned || injw.bytes < M15H::HcPF::InjwPlaneBytes(g.m)))
        return "injw 平面必须 32B 对齐且 ≥ m*ROW_INJW";
    // rstd：按 32B 成对搬 ⇒ m 为奇数时最后一行会多写 16 B 到第 m 行 ⇒ 必须按 m+1 行分配
    if (rstd.bytes != 0u && (!rstd.aligned || rstd.bytes < M15H::HcPF::RstdPlaneBytes(g.m)))
        return "rstd 平面必须 32B 对齐且按 m+1 行分配（成对搬的 16B 溢出，见 m15_hc_prefill.h §4）";
    return std::string();
}

// ============================================================
// 布局落盘：arena 的行页几何 + 全部 WS 偏移（判据脚本按它算设备地址，不复制第二份数字）
// ============================================================
inline void DumpLayout(FILE* f, const char* tag, const Geom& g, bool arenaDumped, bool chain)
{
    const uint32_t m = g.m;
    std::fprintf(f, "# %s：M119 hc prefill 的 arena 行页布局（由 m15_hc_resources.h 的常量算出）\n", tag);
    std::fprintf(f, "mt=%u tile_stride=%u row_hp=%u row_hid=%u row_rstd=%u row_injw=%u\n", MT, TILE_STRIDE,
                 ROW_HP, ROW_HID, ROW_RSTD, ROW_INJW);
    std::fprintf(f, "m=%u tiles=%u arena_bytes=%llu arena_dumped=%u\n", m, TileCount(m),
                 static_cast<unsigned long long>(ArenaBytes(m)), arenaDumped ? 1u : 0u);
    const struct { const char* n; uint32_t off; uint32_t bytes; } t[] = {
        {"hcp", M15H::WS_HCP, M15H::SZ_HCP},    {"xn", M15H::WS_XN, M15H::SZ_XN},
        {"rstd", M15H::WS_RSTD, M15H::SZ_RSTD}, {"injw", M15H::WS_INJW, M15H::SZ_INJW},
        {"oh", M15H::WS_OH, M15H::SZ_OH},       {"ls", M15H::WS_LS, M15H::SZ_LS},
        {"gate", M15H::WS_GATE, M15H::SZ_GATE}, {"blk", M15H::WS_BLK, M15H::SZ_BLK},
    };
    for (const auto& e : t) {
        std::fprintf(f, "off_%s=%u sz_%s=%u\n", e.n, e.off, e.n, e.bytes);
    }
    std::fprintf(f, "ws_bytes=%u hyper=%u hid=%u lowrank=%u hc=%u inj_n=%u oh_w=%u oh_inj=%u inj_stride=%u "
                    "injw_slot=%u up_n=%u up_k=%u m_max=%u\n",
                 M15H::WS_BYTES, M15H::HYPER, M15H::HID, M15H::LOWRANK, M15H::HC, M15H::INJ_N, M15H::OH_W,
                 M15H::OH_INJ, M15H::IJ_STRIDE, M15H::INJW_SLOT, M15H::UP_N, M15H::UP_K, M15H::M_MAX);
    std::fprintf(f, "mode=%u ij_stride=%u ij_tile_stride=%u h_in_tile_stride=%u bo_tile_stride=%u\n", g.mode,
                 g.ijStride, g.ijTileStride, g.hInTileStride, g.boTileStride);
    std::fprintf(f, "chain=%u oh_inj_off=%u ij_row_bytes=%u\n", chain ? 1u : 0u, M15H::HcPF::OhInjOff(),
                 M15H::HcPF::IjRowBytes());
}

}  // namespace HcPfH
}  // namespace M15L

#endif  // M15_HC_PREFILL_HOST_H
