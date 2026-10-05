#!/usr/bin/env python3
# ============================================================
# check_ref.py —— M117 / Wave-B3 的 **自建稠密 causal 参考**判据（numpy float64）
#
# **不许拿官方 QSA 输出当判据**（`docs/17` §7 / M103-4）：官方 QSA 的 `indexer_budget=2048`
# 只覆盖约一半历史，与稠密 causal 在任何长度下都不可能一致。本脚本自己算 fp64 参考。
#
# 判据口径（`docs/17` §1.1 的 T3 形态：`|got−exp| ≤ ε·Σ|terms| + 0.5·ulp(out)`）：
#   out[i,h,:] = Σ_{j ≤ posBase+i} p_ij · V[j, n2(h), :]
#   `scale_elem[i,h,d] = Σ_j p_ij·|V[j,n2(h),d]|`（该元素的 Σ|terms|）
#   `tol = EPS·(scale_elem + |ref|) + 0.5·ulp_bf16(ref)`
#   ε 的来历（**推导，不是调参**）：
#     · P 落 bf16 ⇒ 逐元素相对量化上界 `2^-9`；
#     · 分子与分母同时被扰动 ⇒ `|Δout| ≤ 2^-9·(Σp|v|/Σp + |out|)`；
#     · 再乘 SAFETY=4 吸收 fp32 累加（≤33 个 tile）、设备 `Exp` 与 fp32 行规约的松弛。
#   全部数字口径：**max 相对该 tol**（不是「首次不匹配」）。
#
# 用法：
#   python3 check_ref.py <dir>          # 单档：打印报告并给 VERDICT
#   python3 check_ref.py <dir> --quiet
# 返回码：0 = PASS，1 = FAIL，2 = 用法/数据错。
# ============================================================
import os
import sys
import json
import hashlib

import numpy as np

EPS_P = 2.0 ** -9      # P 落 bf16 的逐元素相对量化上界
SAFETY = 4.0           # fp32 累加 / 设备 exp / 行规约的松弛
EPS = EPS_P * SAFETY
BF16_ULP_REL = 2.0 ** -8   # bf16 的 ulp 相对量级（尾数 7+1 位）


def bf16_to_f64(raw_u16):
    """把 bf16 的位型（uint16 数组）解成 float64（无舍入，bf16 是 fp32 的高 16 位）。"""
    u32 = raw_u16.astype(np.uint32) << 16
    return u32.view(np.float32).astype(np.float64)


def load_u16(path, count):
    a = np.fromfile(path, dtype='<u2')
    if a.size != count:
        raise SystemExit(f"[check_ref] {path}: 元素数 {a.size} != 期望 {count}")
    return a


def parse_params(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            d[k.strip()] = int(v.strip())
    return d


def reference(q, k, v, posBase, ctx, scale):
    """稠密 causal 参考（全程 float64）。q:[m,NH,HD] k/v:[ctx,NKV,HD] -> ref:[m,NH,HD], scale_elem"""
    m, NH, HD = q.shape
    NKV = k.shape[1]
    G = NH // NKV
    ref = np.empty((m, NH, HD), dtype=np.float64)
    scal = np.empty((m, NH, HD), dtype=np.float64)
    idx = np.arange(ctx, dtype=np.int64)[None, :]                  # [1, ctx]
    pos = posBase + np.arange(m, dtype=np.int64)[:, None]          # [m, 1]
    mask = idx <= pos                                              # 因果：只见 j ≤ pos(i)
    for h in range(NH):
        n2 = h // G
        s = (q[:, h, :] @ k[:, n2, :].T) * scale                   # [m, ctx]
        s = np.where(mask, s, -np.inf)
        mx = s.max(axis=1, keepdims=True)
        w = np.exp(s - mx)
        p = w / w.sum(axis=1, keepdims=True)                       # [m, ctx]
        vk = v[:, n2, :]                                           # [ctx, HD]
        ref[:, h, :] = p @ vk
        scal[:, h, :] = (p @ np.abs(vk))
    return ref, scal


def main(argv):
    if len(argv) < 2:
        print("用法: check_ref.py <dir> [--quiet]")
        return 2
    d = argv[1]
    quiet = '--quiet' in argv
    try:
        p = parse_params(os.path.join(d, 'params.txt'))
    except OSError as e:
        print(f"[check_ref][FAIL] 读不到 params.txt: {e}")
        return 2
    m, ctx, posBase, NH, HD = p['m'], p['ctx'], p['posBase'], p['nh'], p['hd']
    NKV = 2
    q = bf16_to_f64(load_u16(os.path.join(d, 'q.bin'), m * NH * HD)).reshape(m, NH, HD)
    k = bf16_to_f64(load_u16(os.path.join(d, 'k.bin'), ctx * NKV * HD)).reshape(ctx, NKV, HD)
    v = bf16_to_f64(load_u16(os.path.join(d, 'v.bin'), ctx * NKV * HD)).reshape(ctx, NKV, HD)
    got = bf16_to_f64(load_u16(os.path.join(d, 'out.bin'), m * NH * HD)).reshape(m, NH, HD)

    scale = HD ** -0.5
    ref, scal = reference(q, k, v, posBase, ctx, scale)
    err = np.abs(got - ref)
    tol = EPS * (scal + np.abs(ref)) + 0.5 * BF16_ULP_REL * np.abs(ref)
    ratio = err / tol
    n_over = int(np.count_nonzero(ratio > 1.0))
    # 报告项（不参与 PASS/FAIL）
    rng = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(err / rng))
    ulp = BF16_ULP_REL * np.abs(ref)
    frac_1ulp = float(np.mean(err <= ulp))
    # 非空洞 guard（报告项）：参考本身必须有分布、输出必须对输入敏感
    ref_std = float(np.std(ref))
    got_std = float(np.std(got))
    sha = hashlib.sha256(open(os.path.join(d, 'out.bin'), 'rb').read()).hexdigest()[:16]

    verdict = 'PASS' if n_over == 0 else 'FAIL'
    if not quiet:
        print(f"[check_ref] dir={d}")
        print(f"[check_ref] m={m} posBase={posBase} ctx={ctx} nh={NH} hd={HD} eps={EPS:.3e}")
        print(f"[check_ref] 判定项 1 条：逐元素 |got-ref| <= tol（共 {ref.size} 元素）")
        print(f"[check_ref]   超界元素 = {n_over} / {ref.size}")
        print(f"[check_ref]   口径：max 相对 tol 的比值 = {float(ratio.max()):.4f}（<1 即全部覆盖）")
        print(f"[check_ref] 报告项：maxRel = {max_rel:.3e}；≤1bf16ulp 比例 = {frac_1ulp*100:.3f}%")
        print(f"[check_ref] guard（报告项，不计判定）：std(ref)={ref_std:.6f} std(got)={got_std:.6f} "
              f"out_sha256_16={sha}")
        if n_over:
            bad = np.argwhere(ratio > 1.0)
            print(f"[check_ref] 前 5 个超界下标(展平): {[tuple(x) for x in bad[:5]]}")
    print(f"VERDICT={verdict} dir={d} over={n_over}/{ref.size} maxratio={float(ratio.max()):.4f}")
    return 0 if n_over == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
