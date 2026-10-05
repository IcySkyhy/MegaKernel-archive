#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""negctl_over2.py —— M174 门限口径修正的**孤立负向对照**（零设备、零 dump）。

目的：证明新子条件「良态元素 ulp>2 占比 ≤1e-3」**有牙** —— 构造一个只有该子条件红的输入，
`check_ref.judge()` 必须判 FAIL；并把门限边界（0.05% 过 / 0.2% 红）实测出来。
同时证明**另外两条子条件**（归一 maxAbs ≤1e-2、良态逐位率 ≥0.99）在该输入上都是 PASS
⇒ 红的是 over2 这一条，不是别的。

用法：python3.12 m20_hyperconn/evidence/negctl_over2.py
退出码：0 = 三条断言全部成立；1 = 断言不符（传播失败）。
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20  # noqa: E402

N = 200000
PERTURB_FRAC = 0.002      # 0.2% 的良态元素被推 >2 ulp
SEED = 174


def build(perturb_frac):
    """造 exp（在 bf16 格点上）与 got：把 `perturb_frac` 比例的良态元素幅值 +3 ulp。

    刻度 = 16.0（少数锚点），主体幅值 ∈ [1,4] ⇒ 良态门限 0.05·scale = 0.8，全部元素良态。
    """
    rng = np.random.default_rng(SEED)
    mag = rng.uniform(1.0, 4.0, N)
    anchor = rng.choice(N, 8, replace=False)
    mag[anchor] = 16.0
    sign = rng.choice([-1, 1], N)
    exp_b = m20.b16((sign * mag).astype(np.float64))           # exp 落在 bf16 格点上
    exp = m20.f32(exp_b).astype(np.float64)
    got = exp_b.copy()
    n_p = int(round(perturb_frac * N))
    pool = np.setdiff1d(np.arange(N), anchor, assume_unique=True)
    idx = rng.choice(pool, n_p, replace=False)
    low = got[idx] & np.uint16(0x7FFF)                          # 幅值位；+3 步 = 推 3 个格点
    got[idx] = (got[idx] & np.uint16(0x8000)) | ((low + 3) & np.uint16(0x7FFF))
    return got.reshape(-1), exp.reshape(-1)


def subconds(got, exp):
    scale = float(np.max(np.abs(exp)))
    exp_b = m20.b16(exp).reshape(-1)
    d = np.abs(m20.ulp_key(got).astype(np.int64) - m20.ulp_key(exp_b).astype(np.int64))
    sig = np.abs(exp) > 0.05 * scale
    norm_abs = float(np.max(np.abs(m20.f32(got).astype(np.float64) - exp)) / scale)
    sfrac = float(np.mean(d[sig] == 0))
    maxulp = int(np.max(d[sig]))
    over2 = float(np.mean(d[sig] > 2))
    return dict(scale=scale, n_sig=int(sig.sum()), norm_abs=norm_abs, sfrac=sfrac,
                maxulp=maxulp, over2=over2,
                ok_norm=norm_abs <= 1e-2, ok_sfrac=sfrac >= 0.99, ok_over2=over2 <= 1e-3)


def run(label, frac, expect_judge):
    got, exp = build(frac)
    c = subconds(got, exp)
    print("[negctl] %-14s over2=%.3e (n_sig=%d) sfrac=%.4f norm_abs=%.3e ulpMax=%d "
          "| 子条件 norm=%s sfrac=%s over2=%s"
          % (label, c["over2"], c["n_sig"], c["sfrac"], c["norm_abs"], c["maxulp"],
             c["ok_norm"], c["ok_sfrac"], c["ok_over2"]))
    m20.FAILS.clear()
    m20.JUDGE[0] = 0
    verdict = m20.judge(label, got, exp, {})
    print("[negctl]   shipped judge() -> %s（期望 %s）"
          % ("PASS" if verdict else "FAIL", "PASS" if expect_judge else "FAIL"))
    return verdict == expect_judge, verdict, c


def main():
    ok = True
    # ① 全对 ⇒ 必绿
    r, v, c = run("all-correct", 0.0, True)
    ok &= r
    # ② 0.05% 越 2 ulp（over2=5e-4 < 1e-3）⇒ 仍绿（门限之上）
    r, v, c = run("over2@5.0e-4", 0.0005, True)
    ok &= r
    # ③ 0.2% 越 2 ulp（over2=2e-3 > 1e-3）⇒ 必红；且只有 over2 这条红
    r, v, c = run("over2@2.0e-3", PERTURB_FRAC, False)
    ok &= r
    ok &= (not v) and c["ok_norm"] and c["ok_sfrac"] and (not c["ok_over2"])
    if not c["ok_norm"] or not c["ok_sfrac"]:
        print("[negctl] 断言失败：期望该输入上 norm 与 sfrac 两条**都过**，实际 "
              "norm=%s sfrac=%s" % (c["ok_norm"], c["ok_sfrac"]))
    print("[negctl] RESULT: %s（新子条件 over2 在 %s 上把判据打红，另两条子条件均为 PASS）"
          % ("OK" if ok else "FAILED", "over2@2.0e-3"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
