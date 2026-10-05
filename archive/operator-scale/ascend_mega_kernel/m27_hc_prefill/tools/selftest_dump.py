#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m27_hc_prefill/tools/selftest_dump.py —— 判据链路的**自检**（不是证据）

它**不跑设备**：按 `m27_hc_prefill.asc` 完全相同的确定性公式生成输入，
用独立参考 `m20_hyperconn/check_ref.py::reference` 算出"理想设备输出"，
再按 `m27_hc_prefill.asc` 的落盘格式写成一个 dump 目录。两个用途：

    A. **正向**：`check_ref.py` 对这份 dump 必须 **rc=0** —— 否则是判据链路的 bug
       （文件名 / 形状 / meta / 参考调用签名 / injw 槽取法 / rstd 形状 任何一处错都会在这里
       先暴露，而不是白占一次设备槽）；
    B. **负向**：`--mutate <tag>.<suffix>` 把某个输出弄坏 ⇒ 必须 **rc=1**
       （证明判据对"输出错了"敏感）。

用法：
    python3.12 m27_hc_prefill/tools/selftest_dump.py /tmp/m27_self m33 33 1
    python3.12 m27_hc_prefill/check_ref.py /tmp/m27_self          # 期望 rc=0
    python3.12 m27_hc_prefill/tools/selftest_dump.py /tmp/m27_mut m33 33 1 --mutate a1.blk
    python3.12 m27_hc_prefill/check_ref.py /tmp/m27_mut           # 期望 rc=1

**明确**：本脚本生成的 dump 一律带 `m27_selftest.txt` 标记，**不得**当作 mission 的验证证据
（证据只认设备实跑的 dump，见 README §8）。
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20   # noqa: E402

HC, HID, HYPER, LOWRANK, INJ_N, INJW_SLOT = 4, 2560, 10240, 320, 4, 8
MODE_MIX, MODE_COMBINE_MIX, MODE_COMBINE_ONLY = 0, 1, 3
WS_BYTES = 4353024   # m15_hc_resources.h 的 WS_BYTES（自检只写进 case 行，判据不据此算地址）


def gen_bf16(rows, cols, salt, amp):
    """与 `.asc` 的 `GenBf16` 逐式相同（含 Hash3 的 32 位截断语义）。"""
    i = np.arange(rows, dtype=np.int64)[:, None]
    j = np.arange(cols, dtype=np.int64)[None, :]
    h = ((i * 2654435761 + j * 40503 + salt * 97 + 0x9E3779B9) & 0xFFFFFFFF).astype(np.uint32)
    h = (h ^ (h >> np.uint32(16))).astype(np.uint32)
    h = ((h * np.uint32(2246822519)) & np.uint32(0xFFFFFFFF)).astype(np.uint32)
    h = (h ^ (h >> np.uint32(13))).astype(np.uint32)
    q = (h % np.uint32(2001)).astype(np.int64) - 1000
    v = (amp * q.astype(np.float64) / 1000.0).astype(np.float32)
    return m20.f32_to_bf16_bits(v.reshape(-1)).reshape(rows, cols).astype(np.uint16)


def wr(dump, name, arr):
    """按 .asc 的格式落盘：<name>.bin + <name>.bin.meta（`<name> <dims...> <dtype>`）。"""
    a = np.ascontiguousarray(arr)
    a.tofile(os.path.join(dump, name + ".bin"))
    dt = {np.dtype(np.uint16): "BF16", np.dtype(np.float32): "F32", np.dtype(np.uint8): "U8",
          np.dtype(np.float64): "F64"}[a.dtype]
    with open(os.path.join(dump, name + ".bin.meta"), "w") as f:
        f.write("%s %s %s\n" % (name, " ".join(str(x) for x in a.shape), dt))


def mutate_bf16(a, tags, tag, suffix):
    if tag != "%s.%s" % (tags, suffix):
        return a
    a = np.array(a, copy=True)
    a.reshape(-1)[::3] ^= np.uint16(0x0040)   # 翻 bf16 的尾数位 ⇒ ulp 键差 ≫2、逐位率骤降
    return a


def mutate_f32(a, tags, tag, suffix):
    if tag != "%s.%s" % (tags, suffix):
        return a
    return (np.array(a, dtype=np.float32, copy=True) * 1.5).astype(np.float32)


def write_outputs(dump, case, tag, ref, mode, mut, delta):
    # H'：mode MIX 不物化（设备不写 ⇒ host 的 arena 是 0）
    mrows = ref["hc"].shape[0]
    if mode == MODE_MIX:
        hcp = np.zeros((mrows, HYPER), dtype=np.uint16)
    else:
        hcp = m20.b16(ref["hc"]).reshape(mrows, HYPER)
    wr(dump, "%s_%s_hcp" % (case, tag), mutate_bf16(hcp, tag, mut, "hcp"))
    # BLK：COMBINE_ONLY 只跑 W0+S1 ⇒ 不产出（设备不写 ⇒ 毒值；自检写 0 并在判据侧 SKIP）
    blk = np.zeros((ref["blk"].shape[0], HID), dtype=np.uint16) if mode == MODE_COMBINE_ONLY \
        else m20.b16(ref["blk"])
    wr(dump, "%s_%s_blk" % (case, tag), mutate_bf16(blk, tag, mut, "blk"))
    # injW：32B 槽（首元素 = 有效值，其余无效位）。mode MIX 不跑 W0 ⇒ 不产出。
    injw = np.zeros((ref["injw"].shape[0], HC * INJW_SLOT), dtype=np.float32)
    if mode != MODE_MIX:
        for mi in range(ref["injw"].shape[0]):
            for s in range(HC):
                injw[mi, s * INJW_SLOT] = ref["injw"][mi, s]
    wr(dump, "%s_%s_injw" % (case, tag), mutate_f32(injw, tag, mut, "injw"))
    # rstd：COMBINE_ONLY 不跑 S2 ⇒ 不产出
    rstd = np.zeros_like(ref["rstd"], dtype=np.float32) if mode == MODE_COMBINE_ONLY \
        else np.asarray(ref["rstd"], dtype=np.float32)
    wr(dump, "%s_%s_rstd" % (case, tag), mutate_f32(rstd, tag, mut, "rstd"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("case")
    ap.add_argument("m", type=int)
    ap.add_argument("mode", type=int)
    ap.add_argument("--mutate", default="", help="形如 a1.blk / a2.injw（空 = 正向自检）")
    ap.add_argument("--reloc-mask", type=int, default=15,
                    help="写进元数据的重定位目标掩码（0xF=常规；0x0=忠实复现 `M27_MUTANT=norel` 的配置）")
    ap.add_argument("--poison-final", action="store_true",
                    help="把四个重定位产物（blk/injw/rstd/ij_handoff）写成毒值 0xCD（= 判据侧那一半："
                         "「契约要求有值、而平面是毒值」时必须值级变红；配合 --reloc-mask 0xF 即 sinkreloc 的判据半）")
    a = ap.parse_args()
    m, mode = a.m, a.mode
    os.makedirs(a.dump, exist_ok=True)

    hin = gen_bf16(m, HYPER, 0x51, 1.0)
    bo = gen_bf16(m, HID, 0x52, 1.0)
    ij = gen_bf16(m + 64, 16, 0x53, 1.0)
    bo2 = gen_bf16(m, HID, 0x54, 1.1)
    w1 = {"norm": gen_bf16(1, HYPER, 112, 0.5), "down": gen_bf16(LOWRANK, HYPER, 122, 0.03),
          "up": gen_bf16(HYPER, LOWRANK, 132, 0.03)}
    w2 = {"norm": gen_bf16(1, HYPER, 213, 0.5), "down": gen_bf16(LOWRANK, HYPER, 223, 0.03),
          "up": gen_bf16(HYPER, LOWRANK, 233, 0.03)}
    slot1 = np.zeros((16, HYPER), dtype=np.uint16)
    slot1[:INJ_N] = gen_bf16(INJ_N, HYPER, 142, 0.5)
    slot2 = np.zeros((16, HYPER), dtype=np.uint16)
    slot2[:INJ_N] = gen_bf16(INJ_N, HYPER, 243, 0.5)
    for tg, W, slot in (("a1", w1, slot1), ("a2", w2, slot2)):
        for k, v in W.items():
            wr(a.dump, "w_%s_%s" % (tg, k), v)
        wr(a.dump, "w_%s_inj" % tg, slot)

    ref1 = m20.reference(m, mode, hin, bo, ij, w1["down"], slot1, w1["up"], w1["norm"])
    wr(a.dump, "%s_a1_hin" % a.case, hin)
    wr(a.dump, "%s_a1_bo" % a.case, bo)
    wr(a.dump, "%s_a1_ij" % a.case, ij[:m].astype(np.uint16).reshape(m, 16))
    write_outputs(a.dump, a.case, "a1", ref1, mode, a.mutate, 0.0)

    chain = mode not in (MODE_MIX, MODE_COMBINE_ONLY)
    if chain:
        hcp1 = m20.b16(ref1["hc"]).reshape(m, HYPER).astype(np.uint16)
        # 边界 #1 的 **ij handoff 面** = 它的 `OH[:,320:324)` 重定位结果（设备侧由 RelocateTile 产出）
        ij_handoff = m20.b16(ref1["lora"][:, LOWRANK:LOWRANK + INJ_N]).reshape(m, INJ_N).astype(np.uint16)
        wp = np.zeros((m, 16), dtype=np.uint16)
        wp[:, :INJ_N] = ij_handoff
        ref2 = m20.reference(m, mode, hcp1, bo2, wp, w2["down"], slot2, w2["up"], w2["norm"])
        # 与设备同样的格式：**32 B/行**的平铺面（前 INJ_N 列有效，其余 padding）
        plane = np.zeros((m, 16), dtype=np.uint16)
        plane[:, :INJ_N] = mutate_bf16(ij_handoff, "a1", a.mutate, "ij_handoff")
        wr(a.dump, "%s_a1_ij_handoff" % a.case, plane)
        wr(a.dump, "%s_a2_hin" % a.case, hcp1)
        wr(a.dump, "%s_a2_bo" % a.case, bo2)
        write_outputs(a.dump, a.case, "a2", ref2, mode, a.mutate, 0.0)
    else:
        # 本档不产出 ij handoff（判据侧对应项 SKIP）；仍写一份零面，避免判据把它当"缺文件"（rc=2）
        wr(a.dump, "%s_a1_ij_handoff" % a.case, np.zeros((m, 16), dtype=np.uint16))

    tiles = (m + 7) // 8
    if a.poison_final:
        for tag2 in (["a1", "a2"] if chain else ["a1"]):
            for nm, shape in (("blk", (m, HID)), ("injw", (m, HC * INJW_SLOT)), ("rstd", (m, HC))):
                pth = os.path.join(a.dump, "%s_%s_%s.bin" % (a.case, tag2, nm))
                if os.path.exists(pth):
                    with open(pth, "wb") as f:
                        f.write(b"\xCD" * (shape[0] * shape[1] * (4 if nm != "blk" else 2)))
        pth = os.path.join(a.dump, "%s_a1_ij_handoff.bin" % a.case)
        if os.path.exists(pth):
            with open(pth, "wb") as f:
                f.write(b"\xCD" * (m * 16 * 2))
        print("[selftest] --poison-final：四个重定位产物已写成毒值（判据侧那一半）")

    with open(os.path.join(a.dump, "m27_case_%s.txt" % a.case), "w") as f:
        f.write("# SELFTEST dump（判据链路自检，非设备证据；见 tools/selftest_dump.py）\n")
        f.write("case %s m %u mode %u mt 8 tiles %u arena_bytes %d real_w 0 chain %u mutant none "
                "ub_peak_pf 94592 arena_dumped 0 reloc_mask %u\n"
                % (a.case, m, mode, tiles, (tiles - 1) * (8 * HYPER * 2) + WS_BYTES, 1 if chain else 0,
                   a.reloc_mask))
    # 布局文件：与 m15_hc_resources.h 的偏移式同源（设备档里这份文件由 .asc 用同一批常量写出；
    # 自检里它的唯一用途 = 让"解析布局"这条代码路径也被跑到）
    offs = {"off_hcp": 0, "sz_hcp": 1310720, "off_xn": 1310720, "sz_xn": 1310720, "off_rstd": 2621440,
            "sz_rstd": 1024, "off_injw": 2622464, "sz_injw": 8192, "off_oh": 2630656, "sz_oh": 43008,
            "off_ls": 2673664, "sz_ls": 40960, "off_gate": 2714624, "sz_gate": 1310720,
            "off_blk": 4025344, "sz_blk": 327680, "ws_bytes": WS_BYTES, "tile_stride": 8 * HYPER * 2,
            "row_hp": HYPER * 2, "row_hid": HID * 2, "row_rstd": HC * 4, "row_injw": HC * INJW_SLOT * 4,
            "oh_inj_off": 2630656 + LOWRANK * 2, "ij_row_bytes": 32}
    with open(os.path.join(a.dump, "m27_layout_%s.txt" % a.case), "w") as f:
        f.write("# SELFTEST：布局常量（与 m15_hc_resources.h 的偏移式同源）\n")
        for k, v in offs.items():
            f.write("%s=%d\n" % (k, v))
        f.write("m=%d mode=%d mt=8 tiles=%d arena_dumped=0\n" % (m, mode, tiles))
    with open(os.path.join(a.dump, "m27_selftest.txt"), "w") as f:
        f.write("本目录由 m27_hc_prefill/tools/selftest_dump.py 生成：判据链路自检，**不是**设备证据。\n")
    print("[selftest] dump=%s case=%s m=%u mode=%u%s" %
          (a.dump, a.case, m, mode, "" if not a.mutate else "  已弄坏 %s" % a.mutate))
    print("[selftest] 下一步：python3.12 m27_hc_prefill/check_ref.py %s（期望 rc=%d）"
          % (a.dump, 1 if a.mutate else 0))


if __name__ == "__main__":
    main()
