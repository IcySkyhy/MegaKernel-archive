#!/usr/bin/env python3.12
"""M124：改前（AIV GEMV）/ 改后（cube mmad）两组设备 dump 的逐平面差异读数。
用法: python3.12 /tmp/m124_plane_diff.py <base_out> <new_out>"""
import pathlib, sys
import numpy as np
def bf16(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)
def bf_ulp(v):
    a = np.abs(np.asarray(v, dtype=np.float64)); a = np.where(a == 0.0, np.finfo(np.float32).tiny, a)
    return np.power(2.0, np.floor(np.log2(a)) - 7.0)
base, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
print("%-16s %-9s %-7s %-8s %-11s %-9s %s" % ("平面","元素数","不同","占比","maxAbs","maxUlp",">1ulp"))
for f in ("B_ids_body.bin","B_emb.bin","B_kv.bin","B_gated.bin","B_normed.bin","B_out.bin","B_state_out.bin"):
    b, n = bf16((base/f).read_bytes()).astype(np.float64), bf16((new/f).read_bytes()).astype(np.float64)
    d = np.abs(b - n); nd = int((b != n).sum()); g = np.minimum(bf_ulp(b), bf_ulp(n))
    mu = "%.0f" % np.max(d / g) if d.max() > 0 else "-"
    print("%-16s %-9d %-7d %7.3f%% %-11.3g %-9s %d" % (f, b.size, nd, 100*nd/b.size, d.max(), mu, int((d > 1.5*g).sum())))
