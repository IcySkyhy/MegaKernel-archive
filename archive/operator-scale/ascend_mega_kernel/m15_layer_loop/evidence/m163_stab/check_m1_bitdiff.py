#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M163：m=1 修前/修后 **非逐位** 的差异计数（F1 要求写进证据）。

背景：修前/修后在 m=1 的 `max|o|`/`max|ht|` **标量**逐位相等，但**张量并非逐位相同** ——
指数差的舍入路径与 `exp(ĝ)·exp(−ĝ)` 不同（例如 m=1 的 KT′ 列：旧式 `exp(g0)·exp(−g0)` 与
新式 `exp(0)=1`；Γi 对角旧式 `exp(g)·exp(−g)` 与新式 `exp(0)=1`）。本脚本量化该差异。

判据（可传播失败）：两者均有限、且逐元素 `max|Δ|` 不超过给定上界（默认 1e-7，见 m=1 量级）。
用法：`<python> check_m1_bitdiff.py <base_dir> <fix_dir>`（各自含 `m23_Pf.gdn_{o,ht,meta}.bin`）。
"""
import sys

import numpy as np

H, T, DV, DK = 48, 1, 128, 128


def load(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape)


def report(name, a, b, bound):
    ia = a.view(np.uint32).ravel()
    ib = b.view(np.uint32).ravel()
    ndiff = int((ia != ib).sum())
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mx = float(d.max()) if d.size else 0.0
    # ulp 距离（同号有限值；用 float32 位型差近似）
    finite = np.isfinite(a) & np.isfinite(b)
    ok = (finite & ((a >= 0) == (b >= 0))).ravel()
    ulp = int(np.abs(ia.astype(np.int64) - ib.astype(np.int64))[ok].max()) if ok.any() else 0
    print(f"  {name}: 不同元素 {ndiff}/{a.size}；max|Δ|={mx:.6e}；max ulp≈{ulp}")
    return ndiff, mx, ulp


def main():
    base_dir, fix_dir = sys.argv[1], sys.argv[2]
    rc = 0
    any_diff = 0
    for name, shape in (("o", (H, T, DV)), ("ht", (H, DK, DV))):
        a = load(f"{base_dir}/m23_Pf.gdn_{name}.bin", shape)
        b = load(f"{fix_dir}/m23_Pf.gdn_{name}.bin", shape)
        ndiff, mx, ulp = report(name, a, b, 1e-7)
        any_diff += ndiff
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            print(f"  ⇒ FAIL（{name} 含非有限）")
            rc = 1
        if mx > 1e-7:
            print(f"  ⇒ FAIL（{name} max|Δ|={mx:.3e} > 1e-7）")
            rc = 1
    print("  ⇒ {}（m=1 max 标量逐位相等；张量有 {} 个元素不同，量级不变）".format(
        "PASS" if rc == 0 else "FAIL", any_diff))
    return rc


if __name__ == "__main__":
    sys.exit(main())
