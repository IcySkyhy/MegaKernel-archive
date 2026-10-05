#ifndef M15_GDN_PREFILL_HOST_H
#define M15_GDN_PREFILL_HOST_H
/**
 * m15_layer_loop/m15_gdn_prefill_host.h —— GDN prefill 段（M115 / Wave B1）的 **host 侧契约**
 *
 * 只放 host 需要的量：工作区尺寸、输入补齐契约、以及给融合 TU 的**挂载点补丁**所依赖的符号。
 * 设备段在 `m15_gdn_prefill.h`，本头**不包含**它（host 与设备编译单元分开，避免设备头进 host TU）。
 *
 * 融合清单里与本头对应的三项：
 *  ① `GdnPrefillScratchBytes(nAic)` —— 段私有 GM 工作区字节数（每 AICore 一个 slot）；
 *  ② `GdnPrefillPadRows()`        —— q/k 平面末尾必须可读且置零的行数（AIC 的 Nd2Nz 固读 64 行）；
 *  ③ `GdnPrefillWorkspaceFits(...)` —— 设备侧峰值断言在编译期已做，这里只给 host 侧的分配口径。
 */

#include <cstdint>

namespace M15GP_H {

// 与 m15_gdn_resources.h 的 GP_* 常量一致（host 侧不引设备头，故在此复述口径；
// 若设备侧改了 slot 尺寸，本文件必须同步 —— 由 README 的融合清单登记）。
constexpr uint32_t kSlotBytes = 753664;      // = GP_SLOT_BYTES = 2 × GP_SC_H_BYTES(376832)
constexpr uint32_t kPadRows = 64;            // q/k 平面尾部补齐行数（Nd2Nz 固读 64 行）
constexpr uint32_t kHeads = 48;
constexpr uint32_t kNK = 16;
constexpr uint32_t kBT = 64;
constexpr uint32_t kCtxMax = 4097;

/** 段私有 GM 工作区字节数（scratch）。 */
inline size_t GdnPrefillScratchBytes(uint32_t nAic) { return size_t(nAic) * kSlotBytes; }

/** q/k 平面尾部补齐行数（**必须置零**：多算出的行/列必须确定，docs/05 §5.3）。 */
inline uint32_t GdnPrefillPadRows() { return kPadRows; }

/** tp = g/β 的行 stride（32B 对齐）。 */
inline uint32_t GdnPrefillTp(uint32_t m) { return ((m + 7) / 8) * 8; }

/**
 * 融合 TU 的挂载点（Wave C 用；可执行补丁文本见本段 README 的「融合清单」节）：
 *
 *   // m15_layer_kernel.h 的 PREFILL 分支里（相位 A，KIND_GDN 层）
 *   #if M15_GDN_PREFILL_WIRE                       // **唯一权威拼法**（见下）
 *   if constexpr (KIND == KIND_GDN) {
 *       M15GP::GdnPrefillArgs gp{...};            // 由 Wave A 的 LayerArgs 填
 *       if ASCEND_IS_AIV { M15GP::GdnPrefillAivEntry(gp); }
 *       if ASCEND_IS_AIC { M15GP::GdnPrefillAicEntry(gp); }
 *   }
 *   #endif
 *
 * **开关拼法（M132 的统一裁决）**：以 `M15_GDN_PREFILL_WIRE` 为准 —— 它是 B1 段自己的融合规范
 * （`m23_gdn_prefill/README.md` §3(g)/§3(h)）里的名字 ⇒ 本头与 `m15_layer_kernel.h` 都照它。
 * 本头此前那种写法（`M15_PREFILL_GDN_WIRE`）已改掉：在 `m15_layer_loop/` 下的宏定义与 `#if` 用法里
 * 不再出现（本注释是为了记下这次统一而提到它，不是第二种拼法的使用点）。
 *
 * 需要的 `LayerArgs` 字段（Wave A 照此加）：q/k/v/g/β/h0/out/ht 的 GM 基址、m、heads、tp、scale、
 * 以及段私有 scratch 基址。本段**不消费 `LayerArgs` 本体**（Wave B 不得改 Wave A 的文件）。
 */
inline void GdnPrefillMountNote() {}

}  // namespace M15GP_H

#endif  // M15_GDN_PREFILL_HOST_H
