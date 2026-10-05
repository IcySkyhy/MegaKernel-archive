#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent cross-check for m28_gdn_epilog: NPU device dumps vs a numpy reference.

被测对象：`m28_gdn_epilog.asc` 的转位/落位段（`wsO` head-major -> token-major `o`；z 段压实）。

参考（本文件 numpy，独立于 kernel 与 harness 内建 C 参考）：
    o[t, h*128 + d] = wsO[h, t, d]          # head-major [48,m,128] -> token-major [m,6144]
    z[t, j]         = qkvzba[t, 10240 + j]  # z 段压实 [m,6144]

判定：位级（fp32 视作 uint32、bf16 视作 uint16），逐元素，T1（容差 0）。

模式：
    pos（默认）：对每个 `m28_m<M>_*` 档，device 必须逐位等于 numpy 参考（绿）。
    mut       ：对每个 `mut_m28_m<M>_*` 档，device 必须**不等于**正确参考（红）——
                这是「判据非空洞」的负向对照：被测对象坏掉时判据必须变红。

用法：
    <python> check_ref.py <dump_dir>            # 正向档
    <python> check_ref.py <dump_dir> --mode mut # 负向对照档
退出码：0 = 全部满足预期；1 = 有反例；2 = 没得比（无 dump）。
"""

import argparse
import glob
import os
import sys

import numpy as np

HEADS = 48
HEAD = 128
HIDDEN = HEADS * HEAD   # 6144
IN_N = 16480
Z_OFF = 10240
Z_DIM = 6144


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = int(p[1])
    return meta


def ref_o(wsO, m):
    """wsO [HEADS,m,HEAD] -> ref o [m,HIDDEN]（列 c = h*HEAD + d）。"""
    return np.transpose(wsO, (1, 0, 2)).reshape(m, HIDDEN)


def ref_z(qkv):
    return qkv[:, Z_OFF:Z_OFF + Z_DIM]


def bits(arr, dtype):
    return arr.view(dtype)


def check_case(prefix, expect_equal):
    meta_path = f"{prefix}_meta.txt"
    meta = read_meta(meta_path)
    m = meta["m"]
    assert meta["heads"] == HEADS and meta["hidden"] == HIDDEN

    wsO = np.fromfile(f"{prefix}_wsO.bin", dtype=np.float32).reshape(HEADS, m, HEAD)
    qkv = np.fromfile(f"{prefix}_qkvzba.bin", dtype=np.uint16).reshape(m, IN_N)
    o_dev = np.fromfile(f"{prefix}_o_device.bin", dtype=np.float32).reshape(m, HIDDEN)
    z_dev = np.fromfile(f"{prefix}_z_device.bin", dtype=np.uint16).reshape(m, Z_DIM)

    ro = ref_o(wsO, m)
    rz = ref_z(qkv)

    o_eq = bool(np.array_equal(bits(o_dev, np.uint32), bits(ro, np.uint32)))
    z_eq = bool(np.array_equal(z_dev, rz))

    if expect_equal:
        o_ok, z_ok = o_eq, z_eq
        verdict = "PASS" if (o_eq and z_eq) else "FAIL"
        print(f"[{prefix}] m={m}: o bit-exact={o_eq} z bit-exact={z_eq} -> {verdict}")
    else:
        # 负向对照（MUT_HEADSTRIDE 只扰动 o 路）：o 必须与正确参考不同（m=1 退化档两者
        # 重合，故只对 m>1 校验）；z 路未被扰动，仍必须逐位等于参考。
        o_ok = (not o_eq) if m > 1 else True
        z_ok = z_eq
        verdict = "PASS" if (o_ok and z_ok) else "FAIL"
        print(f"[{prefix}] m={m}: o bit-exact={o_eq} (want False) z bit-exact={z_eq} (want True) -> {verdict}")
    if not o_eq:
        bad = np.argwhere(bits(o_dev, np.uint32) != bits(ro, np.uint32))
        i, j = bad[0]
        print(f"    o first mismatch [{i}][{j}] (total {len(bad)})")
    if not z_eq:
        bad = np.argwhere(z_dev != rz)
        i, j = bad[0]
        print(f"    z first mismatch [{i}][{j}] (total {len(bad)})")
    return o_ok and z_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default=".")
    ap.add_argument("--mode", choices=["pos", "mut"], default="pos")
    args = ap.parse_args()

    pattern = "m28_m*_meta.txt" if args.mode == "pos" else "mut_m28_m*_meta.txt"
    metas = sorted(glob.glob(os.path.join(args.dump_dir, pattern)))
    if not metas:
        print(f"RESULT: SKIPPED (no {pattern} under {args.dump_dir})")
        return 2

    ok = True
    for mp in metas:
        prefix = mp[: -len("_meta.txt")]
        ok &= check_case(prefix, expect_equal=(args.mode == "pos"))

    print(f"===== mode={args.mode}: {'PASS' if ok else 'FAIL'} =====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
