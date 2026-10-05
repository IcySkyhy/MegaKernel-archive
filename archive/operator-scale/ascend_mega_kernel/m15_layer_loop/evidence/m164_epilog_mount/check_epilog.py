#!/usr/bin/env python3.12
# M164: offline judge for the prefill epilog chain (transpose -> S5 RMSNormGated -> S6 out_proj).
#
# Reads the device dumps written by H_PfEpilogDump (m15_layer_loop.asc) and re-implements the chain
# in numpy **independently** of the kernel:
#   (1) transpose  o[t, h*128+d] = wsO[h, t, d]                 -> T1 bit-exact vs device o
#   (2) z pack     z[t, j]       = qkvzba[t, 10240+j]           -> T1 bit-exact vs device z
#   (3) S5         y = bf16(((o*rstd)*gamma)*sigmoid(z))        -> T3 (bf16 rel <= 1e-2) vs device y
#   (4) S6         out = A[m,6144] @ Wout[2560,6144]^T          -> T3 (2.5e-3*Sum|terms|+1e-6) vs device
#   (a) provenance: device hcAttnOut[0,m) has no 0xCD16 residue, and DIFFERS face-wide from the
#       host-synthesised `bo` plane that H2 used to read before this mount (m164_hostsynth_bo.bin).
#
# Usage:  check_epilog.py <dumpdir> [<dumpdir> ...]
#   each <dumpdir> is either a `m164_<tag>/` directory itself, or a parent that contains one.
# Exit 0 = every arm's expected outcome held; nonzero otherwise (failures propagate).
import glob
import os
import sys

import numpy as np

HEADS = 48
HEAD = 128
HIDDEN = 6144
IN_N = 16480
Z_OFF = 10240
Z_DIM = 6144
OUT_N = 2560
EPS = 1e-6
S5_REL_TOL = 1e-2


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float32)


def f32_to_bf16(f):
    a = np.asarray(f, dtype=np.float32)
    u = np.ascontiguousarray(a).view(np.uint32).astype(np.uint64)
    bias = ((u >> 16) & 1) + 0x7FFF
    out = ((u + bias) >> 16).astype(np.uint16)
    # NaN/inf must stay non-finite: the RNE bias above can overflow a max-payload NaN into the sign
    # bit (0x7FFFFFFF -> 0x8000 = -0.0). Canonicalise non-finite lanes explicitly.
    nonfin = ~np.isfinite(a)
    if nonfin.any():
        sign = ((u >> 31) & 1).astype(np.uint16)
        isinf = np.isinf(a)
        infb = (np.uint16(0x7F80) | sign)
        nanb = (np.uint16(0x7FC0) | sign)
        out = np.where(nonfin, np.where(isinf, infb, nanb), out)
    return out


def read_meta(sub):
    meta = {}
    with open(os.path.join(sub, "m164_meta.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = int(p[1])
    return meta


def resolve_dirs(argv):
    out = []
    for a in argv:
        if os.path.isfile(os.path.join(a, "m164_meta.txt")):
            out.append(a)
        else:
            for meta in sorted(glob.glob(os.path.join(a, "m164_*", "m164_meta.txt"))):
                out.append(os.path.dirname(meta))
    return out


def f32_bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def run_dir(sub):
    meta = read_meta(sub)
    m = meta["m"]
    mut = meta["mut"]
    hid = m * HIDDEN
    ctx = {"m": m, "mut": mut}
    wsO = np.fromfile(os.path.join(sub, "m164_wsO.bin"), dtype="<f4", count=hid).reshape(HEADS, m, HEAD)
    qkv = np.fromfile(os.path.join(sub, "m164_qkvzba.bin"), dtype="<u2", count=m * IN_N).reshape(m, IN_N)
    gam_u = np.fromfile(os.path.join(sub, "m164_gamma.bin"), dtype="<u2", count=HEAD)
    wout_u = np.fromfile(os.path.join(sub, "m164_wout.bin"), dtype="<u2", count=OUT_N * HIDDEN).reshape(OUT_N, HIDDEN)
    o_u = f32_bits(np.fromfile(os.path.join(sub, "m164_o_device.bin"), dtype="<f4", count=hid))
    z_u = np.fromfile(os.path.join(sub, "m164_z_device.bin"), dtype="<u2", count=hid)
    y_u = np.fromfile(os.path.join(sub, "m164_y_device.bin"), dtype="<u2", count=hid)
    out_u = np.fromfile(os.path.join(sub, "m164_hcattnout_device.bin"), dtype="<u2", count=m * OUT_N).reshape(m, OUT_N)
    hs_u = np.fromfile(os.path.join(sub, "m164_hostsynth_bo.bin"), dtype="<u2", count=m * OUT_N).reshape(m, OUT_N)

    # (1) transpose reference (fp32, bit-exact)
    o_ref = np.ascontiguousarray(np.transpose(wsO, (1, 0, 2)).reshape(m, HIDDEN)).view(np.uint32).reshape(-1)
    o_bad = int((o_u != o_ref).sum())
    # (2) z pack reference (bf16, bit-exact)
    z_ref = qkv[:, Z_OFF:Z_OFF + Z_DIM].reshape(-1)
    z_bad = int((z_u != z_ref).sum())

    # (3) S5 reference from device o/z
    o_dev = o_u.view(np.float32).reshape(m, HEADS, HEAD).astype(np.float32)
    z_dev = bf16_to_f32(z_u).reshape(m, HEADS, HEAD)
    gam = bf16_to_f32(gam_u).astype(np.float32)  # (128,)
    mean = (o_dev.astype(np.float64) ** 2).sum(axis=2, keepdims=True) / HEAD
    rstd = (1.0 / np.sqrt(mean + EPS)).astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-z_dev.astype(np.float64)))
    y_ref = f32_to_bf16((o_dev * rstd) * gam[None, None, :] * sig.astype(np.float32)).reshape(-1)
    y_dev = y_u.reshape(-1)
    yg = bf16_to_f32(y_dev).astype(np.float64)
    yr = bf16_to_f32(y_ref).astype(np.float64)
    # m>1 的相位 A 出口 wsO 本身含非有限（M151 §5 登记，独立 fp64 参考复现同一 pattern）⇒ 这里把
    # 非有限位置按"pattern 必须一致"判，有限位置按 T3 相对容差判。两类分开报告。
    yg_f = np.isfinite(yg)
    yr_f = np.isfinite(yr)
    y_nan_mismatch = int((yg_f != yr_f).sum())
    both = yg_f & yr_f
    denom = np.abs(yr)
    rel = np.where(both, np.abs(yg - yr) / np.maximum(denom, np.finfo(np.float64).tiny), 0.0)
    rel = np.where(both & (denom == 0), (yg != 0).astype(np.float64), rel)
    y_bad = int((rel > S5_REL_TOL).sum())
    y_worst = float(rel.max()) if rel.size else 0.0

    # (4) S6 reference (BLAS fp32) from device A (y) and dumped Wout; T3 per element.
    A = bf16_to_f32(y_dev).reshape(m, HIDDEN).astype(np.float32)
    B = bf16_to_f32(wout_u.reshape(-1)).reshape(OUT_N, HIDDEN).astype(np.float32)
    Cref = A @ B.T
    Absum = np.abs(A) @ np.abs(B).T
    Cdev = bf16_to_f32(out_u.reshape(-1)).reshape(m, OUT_N)
    tol = 2.5e-3 * Absum.astype(np.float64) + 1e-6
    cf = np.isfinite(Cdev)
    rf = np.isfinite(Cref)
    s6_nan_mismatch = int((cf != rf).sum())
    bothc = cf & rf
    diff = np.where(bothc, np.abs(Cdev.astype(np.float64) - Cref.astype(np.float64)), 0.0)
    s6_bad = int((diff > tol).sum())
    s6_worst = float((diff / np.maximum(tol, 1e-30)).max())
    s6_finite = int(bothc.sum())

    # (a) provenance
    pois = int((out_u.reshape(-1) == 0xCDCD).sum())
    diff_hostsynth = int((out_u != hs_u).sum())

    print("[M164] == %s ==  m=%d mut=%d" % (os.path.basename(sub), m, mut))
    print("[M164]   (1) transpose o  bit-diff %d/%d" % (o_bad, hid))
    print("[M164]   (2) z pack      bit-diff %d/%d" % (z_bad, hid))
    print("[M164]   (3) S5 y        finite-bad %d/%d worstRel=%.3g (tol %.0e) nan-pattern-mismatch %d"
          % (y_bad, int(both.sum()), y_worst, S5_REL_TOL, y_nan_mismatch))
    print("[M164]   (4) S6 out      finite-bad %d/%d worst(tol-ratio)=%.3g nan-pattern-mismatch %d"
          % (s6_bad, s6_finite, s6_worst, s6_nan_mismatch))
    print("[M164]   (a) hcAttnOut   0xCD16 residue %d, differs-from-hostsynth %d/%d" % (pois, diff_hostsynth, m * OUT_N))

    ok = True
    # provenance / non-vacuity: these hold on every arm (clean and negative)
    ok &= pois == 0
    ok &= diff_hostsynth > 0
    # expected outcome per mutant
    if mut == 0:
        ok &= o_bad == 0
        ok &= z_bad == 0
        ok &= y_bad == 0 and y_nan_mismatch == 0
        ok &= s6_bad == 0 and s6_nan_mismatch == 0
        exp = "clean: o/z bit-exact, S5/S6 finite-T3 + nan-pattern pass"
    elif mut == 1:  # transpose head-stride mutant: red only when m > 1
        if m > 1:
            ok &= o_bad > 0
            exp = "transpose mutant: o must mismatch (m>1): %s" % ("MISMATCHED" if o_bad > 0 else "STILL EQUAL -> FAIL")
        else:
            ok &= o_bad == 0
            exp = "transpose mutant: m=1 degenerate -> o still equal (separation)"
    elif mut == 2:  # S5 gamma not effective
        ok &= (y_bad > 0) or (y_nan_mismatch > 0)
        exp = "S5 nogamma: y must mismatch"
    elif mut == 3:  # S5 all heads read head0 z
        ok &= (y_bad > 0) or (y_nan_mismatch > 0)
        exp = "S5 zhead0: y must mismatch"
    elif mut == 4:  # S6 K-1
        ok &= (s6_bad > 0) or (s6_nan_mismatch > 0)
        exp = "S6 K-1: out must mismatch"
    else:
        exp = "unknown mutant"
        ok = False
    print("[M164]   expectation: %s -> %s" % (exp, "OK" if ok else "FAIL"))
    ctx["ok"] = ok
    return ok


def main(argv):
    dirs = resolve_dirs(argv)
    if not dirs:
        print("[M164] no m164_meta.txt found under: %s" % " ".join(argv))
        return 2
    all_ok = True
    for d in dirs:
        all_ok &= run_dir(d)
    print("[M164] == check_epilog: %s ==" % ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
