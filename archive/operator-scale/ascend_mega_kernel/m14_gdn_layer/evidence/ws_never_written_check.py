#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""m14 workspace dump 诊断：解释 ``*_ws.bin`` 为什么不可逐字节复现。

``H_DumpStep``（``m14_gdn_layer.asc`` 的 ``static void H_DumpStep(...)``）落盘的是**整段**
5,258,240 B workspace，而本 kernel 在 m=1 的各个 case 里只写其中很小一部分（case 列表 = 本文件的
``CASES``，**别在文字里另写一个数字**——M62 r1 P2-5 就是文字比 ``CASES`` 少一个）。本脚本按
``m14_resources.h`` 的编译期常量把 workspace 的每个字节分成三类：

  * ``written``  —— 内核语义写过的字节（M_MAX 尺寸张量的第 0 行 + 按实际尺寸分配的
                    q/k/v/o 全部 + g/β 每 32B 槽的 ``[h][0]``）；
  * ``slotpad``  —— g/β 的 32B 槽尾（``[h][1..7]``）：S3 用 32B 单块
                    ``DataCopy(gGm_[h*8], gL, cpSlot{1,1,0,0})`` 写回，把 UB 槽里
                    未初始化的 28 B 一起写出去 ⇒ 内容随 UB 残留漂移；
  * ``unused``   —— 内核从不触碰的字节（M_MAX 张量第 1..M_MAX-1 行等）。

用法（真机，按 ``dump_manifest.md`` §1(c) 落盘若干次后）：

    /usr/local/python3.12.13/bin/python3 evidence/ws_never_written_check.py DIR [DIR ...]

输出：三类字节数（应与 ``dump_manifest.md`` §6 的数字一致）+ 每个 case 的跨目录差异
分类。**期望结果：差异 100% 落在 ``slotpad``，0 字节落在 ``written``**——即
``*_ws.bin`` 的 sha256 不是内核输出的性质，归档记录无法事后复现。

退出码（三态；调用方/CI 只看退出码也能分辨"没比"和"比过通过"）：

    0 = 比过且通过（末行 ``RESULT: OK (N/<len(CASES)> cases compared; ...)``，分母由 ``CASES`` 推出）
    1 = 比过且有差异（末行 ``RESULT: UNEXPECTED (...)``）
    2 = 没得比 / 输入缺失（末行 ``RESULT: SKIPPED (...)``，例如目录里没有 >=2 份同名
        ``*_ws.bin``）

负向对照（本脚本已实测，勿删）：``python3 ws_never_written_check.py empty_dir missing_dir``
⇒ 所有 case 全部 SKIPPED、末行 ``RESULT: SKIPPED``、退出码 **2**（**不发合格证**）。
"""
import os
import sys

import numpy as np

# ---- 与 m14_resources.h 同步的编译期常量（WS_* / SZ_*）----
M_MAX, HIDDEN, IN_N, NK, HEADS, HEAD_D, V_DIM, ST, CH = 64, 2560, 16480, 16, 48, 128, 6144, 3, 10240
CASES = ["chain", "slice", "chain2_s0", "chain2_s1", "hostchain"]


def align32(x):
    return (x + 31) // 32 * 32


def layout():
    rows = {
        "XNORM": (M_MAX, HIDDEN, 2),
        "RES1": (M_MAX, HIDDEN, 4),
        "QKVZBA": (M_MAX, IN_N, 2),
        "Q": (NK, HEAD_D, 4),
        "K": (NK, HEAD_D, 4),
        "V": (HEADS, HEAD_D, 4),
        "G": (HEADS, 8, 4),
        "BETA": (HEADS, 8, 4),
        "O": (HEADS, HEAD_D, 4),
        "OPIN": (M_MAX, V_DIM, 2),
        "OPOUT": (M_MAX, HIDDEN, 2),
        "YFINAL": (M_MAX, HIDDEN, 2),
        "RES2": (M_MAX, HIDDEN, 4),
    }
    size = {k: align32(r * c * e) for k, (r, c, e) in rows.items()}
    off, total = {}, 0
    for k in rows:
        off[k] = total
        total += size[k]
    return rows, size, off, total


def classify(rows, size, off, total):
    written = np.zeros(total, dtype=bool)
    slotpad = np.zeros(total, dtype=bool)
    for k, (r, c, e) in rows.items():
        if r == M_MAX:                       # 只写第 0 行（m=1 的各 case，列表见 CASES）
            written[off[k]:off[k] + c * e] = True
        elif k not in ("G", "BETA"):         # q/k/v/o 按实际尺寸分配 ⇒ 全写
            written[off[k]:off[k] + size[k]] = True
    for k in ("G", "BETA"):                  # 每 head 一个 32B 槽，仅 [h][0] 有效
        for h in range(HEADS):
            written[off[k] + h * 32:off[k] + h * 32 + 4] = True
            slotpad[off[k] + h * 32 + 4:off[k] + h * 32 + 32] = True
    return written, slotpad, ~written & ~slotpad


def main() -> int:
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 2
    rows, size, off, total = layout()
    written, slotpad, unused = classify(rows, size, off, total)
    # 覆盖范围自述：case 数与名单**由 CASES 推出**（同一份遍历/同一份常量，不手写数字）
    print(f"cases checked: {len(CASES)} ({', '.join(CASES)})")
    print(f"workspace {total} B: written {int(written.sum())} ({100*written.sum()/total:.2f}%) | "
          f"g/beta 32B-slot pad {int(slotpad.sum())} B | never-written {int(unused.sum())} "
          f"({100*unused.sum()/total:.2f}%)")
    for k in sorted(off, key=lambda x: off[x]):
        print(f"  {k:8s} @{off[k]:8d} size {size[k]:8d}  " +
              ("row0 only" if rows[k][0] == M_MAX else "full"))

    bad = 0
    compared = 0
    for tag in CASES:
        arrs = []
        for d in dirs:
            p = os.path.join(d, f"{tag}_ws.bin")
            if not os.path.exists(p):
                continue
            a = np.fromfile(p, dtype=np.uint8)
            if len(a) != total:
                print(f"  [{tag}] {p}: size {len(a)} != {total} (skip)")
                continue
            arrs.append((d, a))
        if len(arrs) < 2:
            print(f"{tag:11s}: SKIPPED (need >=2 dumps, got {len(arrs)})")
            continue
        compared += 1
        stack = np.stack([a for _, a in arrs])
        var = (stack != stack[0]).any(axis=0)
        nw, npad, nun = int((var & written).sum()), int((var & slotpad).sum()), int((var & unused).sum())
        bad += nw + nun
        print(f"{tag:11s} dumps={len(arrs)} varying {int(var.sum()):6d} B -> "
              f"written {nw} | slotpad {npad} | never-written {nun}")
    if compared == 0:
        print("RESULT: SKIPPED (no case had >=2 *_ws.bin dumps — nothing was compared)")
        return 2
    print("RESULT:", ("OK " + f"({compared}/{len(CASES)} cases compared; "
                      "all variation confined to the g/beta slot pad)") if bad == 0
          else f"UNEXPECTED ({bad} bytes vary outside the slot pad; {compared}/{len(CASES)} cases compared)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
