#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""h2_blk_recalib.py —— M174 对 `h2.blk` 判据门限的**离线复算 + 新旧口径对比**。

只读设备 dump（M140/M169 四相位档），用 `m20_hyperconn/check_ref.py::reference` 复算参考，
逐张量打印**新旧两种口径**的读数，并按新口径给判定（失败以 rc=1 传播）。

## 复算的张量（与 M140/M169 台架判据同源，`m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py`）
  · H1 边界：输入 = dump 的 `h1_hin/h1_bo/h1_ij` + 真权重 `w_a1_*`；判定 `h1.hcp` / `h1.blk`。
  · H2 边界：输入 = 设备的 `h1_hcp`（层内 handoff）+ `h2_bo` + `h1_ij_handoff` + 真权重 `w_a2_*`；
    判定 `h2.hcp` / `h2.blk`。

## 两种口径（同一份 `judge` 数学，只换 max 与分位的那条子条件）
  · 旧（本仓此前的独立保守门限）：`norm_abs ≤ 1e-2` 且 **`良态 ulpMax ≤ 2`** 且 `良态逐位率 ≥ 0.99`。
  · 新（M174 登记的门限口径修正）：`norm_abs ≤ 1e-2` 且 **`良态 ulp>2 占比 ≤ 1e-3`** 且 `良态逐位率 ≥ 0.99`。
    `ulpMax` 仍逐张量打印（报告项，不删不隐藏）。

## 用法
    python3.12 m20_hyperconn/evidence/h2_blk_recalib.py <dumpdir> <tag> <m>

`<dumpdir>` 需含 `m140_<tag>_{h1_*,h2_*,w_a2_*}.bin`（两边界权重面齐全；m4097 档不随仓库入库，
由 `m15_layer_loop/evidence/m169_whole_layer_4phase/reproduce.sh` 在设备上再生成）。

退出码：0 = 新口径全绿；1 = 新口径有红（传播失败）；2 = 缺输入 / 档不合适。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20  # noqa: E402

HC, HID, HYPER, LOWRANK, INJ_N = 4, 2560, 10240, 320, 4
IJ_STRIDE, M_MAX = 16, 64
MODE_COMBINE_MIX = 1
OVER2_MAX = 1e-3          # 新口径：良态元素里 ulp>2 的占比上限
SFRAC_MIN, NORM_ABS_MAX = 0.99, 1e-2


def rd(dump, name, elems):
    path = os.path.join(dump, name + ".bin")
    if not os.path.exists(path):
        raise m20.MissingInput(path)
    a = np.fromfile(path, dtype=np.uint16)
    if a.size != elems:
        raise SystemExit("FAIL: %s short (%d != %d)" % (path, a.size, elems))
    return a


def load_w(dump, wtag):
    names = ["w_%s_down" % wtag, "w_%s_inj" % wtag, "w_%s_up" % wtag, "w_%s_norm" % wtag]
    if not all(os.path.exists(os.path.join(dump, "m140_Pf.gdn_%s.bin" % nm)) for nm in names):
        raise m20.MissingInput("缺权重面 w_%s_*（%s）" % (wtag, names))
    wdown = rd(dump, "m140_Pf.gdn_w_%s_down" % wtag, LOWRANK * HYPER).reshape(LOWRANK, HYPER)
    winj = rd(dump, "m140_Pf.gdn_w_%s_inj" % wtag, 16 * HYPER).reshape(16, HYPER)
    wup = rd(dump, "m140_Pf.gdn_w_%s_up" % wtag, HYPER * LOWRANK).reshape(HYPER, LOWRANK)
    norm = rd(dump, "m140_Pf.gdn_w_%s_norm" % wtag, HC * HID).reshape(HC, HID)
    return wdown, winj, wup, norm


def metrics(name, got_u16, exp_f64):
    """与 check_ref.judge 同口径的读数；逐条给出旧/新子条件。"""
    got_b = np.asarray(got_u16, dtype=np.uint16).reshape(-1)
    exp = np.asarray(exp_f64, dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(exp))) if exp.size else 0.0
    exp_b = m20.b16(exp).reshape(-1)
    d = np.abs(m20.ulp_key(got_b).astype(np.int64) - m20.ulp_key(exp_b).astype(np.int64))
    sig = np.abs(exp) > 0.05 * scale
    got = m20.f32(got_b).astype(np.float64)
    norm_abs = float(np.max(np.abs(got - exp)) / scale) if scale > 0 else 0.0
    frac_all = float(np.mean(d == 0))
    n_sig = int(np.count_nonzero(sig))
    sfrac = float(np.mean(d[sig] == 0)) if n_sig else 1.0
    maxulp = int(np.max(d[sig])) if n_sig else 0
    over2 = float(np.mean(d[sig] > 2)) if n_sig else 0.0
    old_ok = norm_abs <= NORM_ABS_MAX and maxulp <= 2 and sfrac >= SFRAC_MIN
    new_ok = norm_abs <= NORM_ABS_MAX and over2 <= OVER2_MAX and sfrac >= SFRAC_MIN
    print("[recalib] %-8s n=%-9d 良态n=%-9d 逐位=%.4f 良态逐位=%.4f "
          "ulpMax=%-3d 良态ulp>2占比=%.3e 归一maxAbs=%.3e | 旧=%s 新=%s"
          % (name, got_b.size, n_sig, frac_all, sfrac, maxulp, over2, norm_abs,
             "PASS" if old_ok else "FAIL", "PASS" if new_ok else "FAIL"))
    return dict(name=name, n=got_b.size, n_sig=n_sig, frac_all=frac_all, sfrac=sfrac,
                maxulp=maxulp, over2=over2, norm_abs=norm_abs, old_ok=old_ok, new_ok=new_ok,
                d=d, sig=sig, exp=exp, scale=scale)


def tail(name, mo):
    """打印该张量良态元素里 ulp>2 的尾部（个数 / 最大 ulp / |ref|/scale 区间）。"""
    d, sig, exp, scale = mo["d"], mo["sig"], mo["exp"], mo["scale"]
    bad = sig & (d > 2)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        print("[recalib]   %s 良态 ulp>2: 0 个" % name)
        return
    r = np.abs(exp[bad]) / scale
    print("[recalib]   %s 良态 ulp>2: %d/%d (%.3e)，最大 ulp=%d，|ref|/scale ∈ [%.4f, %.4f]"
          % (name, n_bad, mo["n_sig"], mo["over2"], int(np.max(d[bad])), float(r.min()), float(r.max())))


def main():
    argv = sys.argv[1:]
    inject = "none"
    if "--inject" in argv:
        i = argv.index("--inject")
        inject = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) != 3:
        print(__doc__)
        return 2
    dump, tag, m = argv[0], argv[1], int(argv[2])
    if m <= 0:
        print("[recalib] SKIPPED（m=%d 不合适）" % m)
        return 2

    try:
        hin = rd(dump, "m140_%s_h1_hin" % tag, m * HYPER).reshape(m, HYPER)
        bo1 = rd(dump, "m140_%s_h1_bo" % tag, m * HID).reshape(m, HID)
        ij1 = rd(dump, "m140_%s_h1_ij" % tag, m * IJ_STRIDE).reshape(m, IJ_STRIDE)
        hcp1 = rd(dump, "m140_%s_h1_hcp" % tag, m * HYPER).reshape(m, HYPER)
        blk1 = rd(dump, "m140_%s_h1_blk" % tag, m * HID).reshape(m, HID)
        h2bo = rd(dump, "m140_%s_h2_bo" % tag, m * HID).reshape(m, HID)
        hcp2 = rd(dump, "m140_%s_h2_hcp" % tag, m * HYPER).reshape(m, HYPER)
        blk2 = rd(dump, "m140_%s_h2_blk" % tag, m * HID).reshape(m, HID)
        ijh = rd(dump, "m140_%s_h1_ij_handoff" % tag, (m + M_MAX) * IJ_STRIDE)[:m * IJ_STRIDE]
        ijh = ijh.reshape(m, IJ_STRIDE)
        w1 = load_w(dump, "a1")
        w2 = load_w(dump, "a2")
    except m20.MissingInput as e:
        print("[recalib] SKIPPED（缺输入 %s）" % e)
        return 2

    print("[recalib] dump=%s tag=%s m=%d（参考 = m20/check_ref.py::reference，fp64）" % (dump, tag, m))
    ref1 = m20.reference(m, MODE_COMBINE_MIX, hin, bo1, ij1, w1[0], w1[1], w1[2], w1[3])
    ref2 = m20.reference(m, MODE_COMBINE_MIX, hcp1, h2bo, ijh, w2[0], w2[1], w2[2], w2[3])

    devs = [("h1.hcp", hcp1, ref1["hc"]), ("h1.blk", blk1, ref1["blk"]),
            ("h2.hcp", hcp2, ref2["hc"]), ("h2.blk", blk2, ref2["blk"])]

    if inject != "none":
        # 负向对照：把一个 64 行块的 h2.blk 输出整体破坏（真实量级的缺陷）
        blkn = blk2.copy().astype(np.uint16)
        rows = min(64, m)
        if inject == "zero":
            blkn[:rows] = 0
        elif inject == "scale":
            f = m20.b16(m20.f32(blkn[:rows]).astype(np.float64) * 1.02)
            blkn[:rows] = f.reshape(rows, HID)
        elif inject == "shift":
            amax = float(np.max(np.abs(m20.f32(blkn))))
            sh = np.float32(1e-2 * amax)
            f = m20.b16(m20.f32(blkn[:rows]).astype(np.float64) + sh)
            blkn[:rows] = f.reshape(rows, HID)
        else:
            raise SystemExit("未知 --inject %s（可选 zero/scale/shift）" % inject)
        print("[recalib] **负向对照**：h2.blk 前 %d 行注入 %s 缺陷" % (rows, inject))
        devs[3] = ("h2.blk+inj", blkn, ref2["blk"])

    res = []
    for name, got, exp in devs:
        mo = metrics(name, got.reshape(-1), exp.reshape(-1))
        res.append(mo)
        tail(name, mo)

    # 直接调 shipped judge（新口径）再判一次 h2.blk，证明走的确实是发布代码路径
    if inject == "none":
        m20.FAILS.clear()
        m20.JUDGE[0] = 0
        print("[recalib] shipped judge() 复判 h2.blk（新口径）：")
        shipped = m20.judge("h2.blk", blk2.reshape(-1), ref2["blk"].reshape(-1), {})
        print("[recalib] shipped judge -> %s" % ("PASS" if shipped else "FAIL"))

    new_bad = [mo["name"] for mo in res if not mo["new_ok"]]
    print("[recalib] 新口径：判定项 %d 条，红 %d 条%s"
          % (len(res), len(new_bad), ("：" + ",".join(new_bad)) if new_bad else ""))
    return 1 if new_bad else 0


if __name__ == "__main__":
    sys.exit(main())
