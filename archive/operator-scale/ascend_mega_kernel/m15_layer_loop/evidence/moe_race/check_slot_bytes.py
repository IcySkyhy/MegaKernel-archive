#!/usr/bin/env python3
"""M105：从一个 `M15_DUMP=1` 的落盘目录里读出「S3 诊断槽」与「router x 行 0 残值」两处字节。

用法：
    python3 check_slot_bytes.py <dump_dir> [L00 L01 ...]

两处地址都由**库内常量自算**（见 addr_arith.md / addr_arith.cpp，同一算式）：
    UB_IG_CNT   = UB_RT_XB + 2 * TOTAL_MAX * 4        = 144384 + 2048 = 146432
    诊断槽字地址 = UB_IG_CNT + 32*4                   = 146560
    它在 router x 窗内的字节偏移 = 146560 - UB_RT_XB   = **2176**   （x 行 0 = [0, HIDDEN*2) → 落在行 0 里）
    GM 诊断槽    = WS_OFFSETS + IG_DIAG_GM_SLOT*4     = 988192 + 32 = **988224**
    dump 偏移 0 = WS_XNORM = 0（`moe_layout.txt` 的 `name=x_norm ws_off=0`）⇒ dump 内偏移与 ws 内偏移同值。

判据口径（本目录各 run 的 `L*_moe_ws.bin` 是**融合路**那一侧的 ws；moe-only 侧不落盘，
它的值只在 `run.log` 的 `M.moews.L*` 行里以 `exp 0x??` 露出首字节）：
  · 修后：`slot[988224:988228] == 00000000` 且 **!=** `x_norm_row0[2176:2180]`
    （0 = 本档 `badIds`，即「专家 id 越界计数」的真值；见 run.log 的 `M.moews` 全 PASS）。
  · 修前：`slot[988224:988228]` 可能等于 x 残值（两条抢跑读数恰好相等 ⇒ 假绿），
    也可能等于 0（这一侧真读到了计数器、另一侧读到残值 ⇒ 判据红）。
"""
import pathlib
import sys

SLOT_OFF = 988224   # = WS_OFFSETS(988192) + IG_DIAG_GM_SLOT(8)*4
XN_OFF = 2176       # = (UB_IG_CNT + 32*4) - UB_RT_XB


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = pathlib.Path(sys.argv[1])
    layers = sys.argv[2:] or ["L00", "L01"]
    for L in layers:
        p = d / f"{L}_moe_ws.bin"
        if not p.exists():
            print(f"{L}: 没有 {p.name}")
            continue
        b = p.read_bytes()
        slot = b[SLOT_OFF:SLOT_OFF + 4]
        xn = b[XN_OFF:XN_OFF + 4]
        verdict = "slot==x残值（抢跑）" if slot == xn else f"slot!=x残值（slot={slot.hex()} xn={xn.hex()}）"
        print(f"{L}: size={len(b)} slot[{SLOT_OFF}:{SLOT_OFF + 4}]={slot.hex()} "
              f"x_norm_row0[{XN_OFF}:{XN_OFF + 4}]={xn.hex()}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
