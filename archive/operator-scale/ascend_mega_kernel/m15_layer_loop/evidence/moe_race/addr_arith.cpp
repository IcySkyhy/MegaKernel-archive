#include <cstdio>
#include "m15_moe_resources.h"
using namespace M15M;
// 生成物 m15_moe_layer.h 里定义的派生常量（源码行 1264-1276 与 lift_moe_segment.py 的
// SCALAR_SLOT_CONSTS）；此处按同一算式独立写出，用于自算地址。
constexpr uint32_t UB_IG_SCAL_ = UB_IG_END;
constexpr uint32_t UB_IG_OFF_ = UB_IG_SCAL_ + AlignUp(NUM_EXPERTS * 4, 32);
constexpr uint32_t UB_IG_CUR_ = UB_IG_OFF_ + AlignUp((NUM_EXPERTS + 1) * 4, 32);
constexpr uint32_t UB_IG_SCAL_END_ = UB_IG_CUR_ + AlignUp(NUM_EXPERTS * 4, 32);
constexpr uint32_t IG_DIAG_GM_SLOT_ = AlignUp((NUM_EXPERTS + 1) * 4 + 4, WS_ALIGN) / 4;
int main() {
    printf("HIDDEN=%u INTER=%u NUM_EXPERTS=%u TOPK_MAX=%u M_MAX=%u TOTAL_MAX=%u\n",
           HIDDEN, INTER, NUM_EXPERTS, TOPK_MAX, M_MAX, TOTAL_MAX);
    printf("UB_RT_XB=%u UB_IG_SRC=%u UB_IG_EXP=%u UB_IG_CNT=%u UB_IG_INV=%u UB_IG_WTK=%u UB_IG_END=%u\n",
           UB_RT_XB, UB_IG_SRC, UB_IG_EXP, UB_IG_CNT, UB_IG_INV, UB_IG_WTK, UB_IG_END);
    printf("UB_IG_SCAL=%u UB_IG_OFF=%u UB_IG_CUR=%u UB_IG_SCAL_END=%u\n",
           UB_IG_SCAL_, UB_IG_OFF_, UB_IG_CUR_, UB_IG_SCAL_END_);
    printf("diag_ub_byte = UB_IG_CNT+32*4 = %u ; minus UB_RT_XB = %u\n", UB_IG_CNT + 32 * 4, UB_IG_CNT + 128 - UB_RT_XB);
    printf("xrow0_span = [%u, %u)  -> diag inside x row0 = %d\n", UB_RT_XB, UB_RT_XB + HIDDEN * 2,
           (UB_IG_CNT + 128 < UB_RT_XB + HIDDEN * 2));
    printf("ROWCTX: RT_RB=%u RT_RB*HIDDEN*2=%u\n", RT_RB, RT_RB * HIDDEN * 2);
    printf("SZ_OFFSETS=%u WS_OFFSETS=%u IG_DIAG_GM_SLOT=%u IG_DIAG_GM_SLOT*4=%u WS_OFFSETS+SZ_OFFSETS/2=%u\n",
           SZ_OFFSETS, WS_OFFSETS, IG_DIAG_GM_SLOT_, IG_DIAG_GM_SLOT_ * 4, WS_OFFSETS + 32);
    printf("WS_XNORM=%u WS_BYTES=%u WS_MOE=%u SZ_XNORM=%u\n", WS_XNORM, WS_BYTES, WS_MOE, SZ_XNORM);
    printf("SZ_COUNTS=%u SZ_OFFSETS tail check: AlignUp((E+1)*4+4,32)=%u\n", SZ_COUNTS, AlignUp((NUM_EXPERTS + 1) * 4 + 4, WS_ALIGN));
    return 0;
}
