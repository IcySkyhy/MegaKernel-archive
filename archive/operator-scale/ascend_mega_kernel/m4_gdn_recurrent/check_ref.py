# SPDX-License-Identifier: Apache-2.0
"""M4 GDN decode 递推核数值校验：device 输出 vs numpy float32 参考（同公式，逐 head）。

递推（docs/10-gdn-analysis.md §1，每 value head hv，key head hk = hv//3，全 fp32）：
    S <- e^g * S
    v <- beta * (v - S @ k)
    S <- S + k outer v        # S[v,k] += k[k]*v[v]
    o = S @ q

用法（在 m4_gdn_recurrent 可执行文件 dump 出的目录运行，即 build/run 或等价目录）：
    /usr/local/python3.12.13/bin/python3 check_ref.py

容差（任务书）：相对容差 1e-5；另加 1e-6 绝对地板吸收 |expect|≈0 处的相消。
state 更新与输出 o 都校验，逐 head 报告最大相对/绝对偏差。
"""
import sys

import numpy as np

CASES = [(4, 4), (8, 8), (8, 4), (4, 8), (48, 24)]  # 与 host main() 保持一致
RTOL = 1e-5
ATOL = 1e-6


def load_case(prefix):
    f32 = np.float32
    q = np.fromfile(f"{prefix}q.bin", dtype=f32)
    k = np.fromfile(f"{prefix}k.bin", dtype=f32)
    v = np.fromfile(f"{prefix}v.bin", dtype=f32).reshape(-1, 128)
    g = np.fromfile(f"{prefix}g.bin", dtype=f32)
    beta = np.fromfile(f"{prefix}beta.bin", dtype=f32)
    h = v.shape[0]
    nk = q.size // 128
    q = q.reshape(nk, 128)
    k = k.reshape(nk, 128)
    s0 = np.fromfile(f"{prefix}state_init.bin", dtype=f32).reshape(h, 128, 128)
    s_dev = np.fromfile(f"{prefix}state_out.bin", dtype=f32).reshape(h, 128, 128)
    o_dev = np.fromfile(f"{prefix}out.bin", dtype=f32).reshape(h, 128)
    return q, k, v, g, beta, s0, s_dev, o_dev


def reference(q, k, v, g, beta, s0):
    """numpy float32 逐公式实现（与 kernel 同序：decay -> delta -> outer -> matvec）。"""
    h = v.shape[0]
    hk = np.arange(h, dtype=np.int64) // 3  # value head hv <- key head hv//3
    eg = np.exp(g).astype(np.float32)[:, None, None]  # [H,1,1] fp32
    s = (s0 * eg).astype(np.float32)  # decay: S <- e^g * S
    w = np.einsum("hvk,hk->hv", s, k[hk]).astype(np.float32)  # w = (e^g S) @ k
    delta = (beta.astype(np.float32)[:, None] * (v - w)).astype(np.float32)  # v <- beta*(v-w)
    s = (s + delta[:, :, None] * k[hk][:, None, :]).astype(np.float32)  # S <- S + k outer v
    o = np.einsum("hvk,hk->hv", s, q[hk]).astype(np.float32)  # o = S_new @ q
    return s, o


def compare(name, got, expect):
    adiff = np.abs(got.astype(np.float32) - expect.astype(np.float32))
    ok = np.all(adiff <= RTOL * np.abs(expect) + ATOL)
    rdiff = adiff / (np.abs(expect) + ATOL)
    idx = np.unravel_index(np.argmax(rdiff), rdiff.shape)
    return ok, float(rdiff.max()), float(adiff.max()), idx, float(got[idx]), float(expect[idx])


def main():
    all_ok = True
    for h, blk in CASES:
        prefix = f"m4_case_h{h:02d}b{blk:02d}_"
        try:
            q, k, v, g, beta, s0, s_dev, o_dev = load_case(prefix)
        except FileNotFoundError as e:
            print(f"[{prefix}] MISSING FILE: {e}（请先运行 m4_gdn_recurrent 生成 dump）")
            all_ok = False
            continue
        s_ref, o_ref = reference(q, k, v, g, beta, s0)
        ok_s, rmax_s, amax_s, is_s, gs_s, es_s = compare("state", s_dev, s_ref)
        ok_o, rmax_o, amax_o, io_o, go_o, eo_o = compare("out", o_dev, o_ref)
        case_ok = ok_s and ok_o
        all_ok &= case_ok
        print(f"[{prefix}] state: {'PASS' if ok_s else 'FAIL'} maxRelDiff={rmax_s:.3e} maxAbsDiff={amax_s:.3e}; "
              f"out: {'PASS' if ok_o else 'FAIL'} maxRelDiff={rmax_o:.3e} maxAbsDiff={amax_o:.3e}")
        if not ok_s:
            print(f"    state worst at {is_s}: got {gs_s:.8e} expect {es_s:.8e}")
        if not ok_o:
            print(f"    out   worst at {io_o}: got {go_o:.8e} expect {eo_o:.8e}")
    print(f"[M4] numpy fp32 reference check (rtol {RTOL:g} + atol {ATOL:g}): "
          f"{'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
