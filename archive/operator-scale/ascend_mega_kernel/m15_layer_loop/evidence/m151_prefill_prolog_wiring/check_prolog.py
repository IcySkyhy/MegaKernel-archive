# SPDX-License-Identifier: Apache-2.0
"""M151：预填充 prolog 链路（S2 in_proj + S3 conv1d/l2norm/gating）的离线对拍。

数据来源 = m15 `prefill` 档（`M15_PREFILL_PROLOG=1`）落下的 dump：
  · `m151_s2_<tag>/{a.bin,b.bin,c_device.bin}` —— m11 `check_ref.py` 同布局；
  · `m151_s3_<tag>/m9mt_m{m}_b{blk}_mut{mut}_*` —— m9 `check_ref.py` 同布局。

判据（独立于 m15 实现；S3 直接 import m9 的 `check_ref`，不复制）：
  · S2：`ref = bf16(a) @ bf16(b)^T`（fp32 累加）→ bf16 RNE；因设备与 numpy 的 fp32 累加序不同，
    判据用 **T3**（`|Δ| ≤ 2.5e-3·Σ|terms| + 1e-6`，≈0.64×bf16 ULP；见 `check_s2()`），
    并另报 bf16 **位型逐位率**作透明度（不是判据）；
  · S3：`m9_gdn_prolog/check_ref.py::check_mt_case`（conv_state_out 位型逐位；q/k 尾部 64 行
    与 g/β 行内 pad 列必须为 0；有效区 |got-exp| ≤ 1e-5·|exp| + 1e-6）。

用法：
    /usr/local/python3.12.13/bin/python3 check_prolog.py <dumpdir> [<dumpdir> ...]
退出码：0 = 干净档全过且负向档全红；非 0 = 任一判据不符（可传播失败）。
`--tag-filter` 可只跑某前缀的档（如 `Pf.gdn$`）。
"""
import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent          # 仓库根（<repo>/m15_layer_loop/evidence/<this>）

# m11/m9 的纯 numpy bf16 helper（参考与实现独立）
sys.path.insert(0, str(ROOT / "tools" / "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # noqa: E402


def _load_m9_check_ref():
    p = ROOT / "m9_gdn_prolog" / "check_ref.py"
    spec = importlib.util.spec_from_file_location("m9_check_ref", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_s2(dirpath):
    a = np.fromfile(os.path.join(dirpath, "a.bin"), dtype=np.uint16)
    b = np.fromfile(os.path.join(dirpath, "b.bin"), dtype=np.uint16)
    c = np.fromfile(os.path.join(dirpath, "c_device.bin"), dtype=np.uint16)
    k = 2560
    a2 = a.reshape(-1, k)
    b2 = b.reshape(-1, k)
    m = c.size // 16480
    c2 = c.reshape(m, 16480)
    a32 = bf16_bits_to_f32(a2)[:m].astype(np.float32)
    b32 = bf16_bits_to_f32(b2).astype(np.float32)
    ref = f32_to_bf16_bits((a32 @ b32.T).astype(np.float32))
    bits_same = int((ref == c2).sum())
    # 口径：差异按 T3（`|Δ| ≤ ε·Σ|terms| + 1e-6`，ε=2.5e-3）判 —— 因设备用 base-K 分块 fp32 累加、
    # 参考用 BLAS 累加，**对近零抵消项**（真实激活里常见）相对差会放大；ε·Σ|terms| ≈ 0.64×bf16 ULP
    # （bf16 ULP = Σ·2^-8），故对非抵消项该界 ≤1 个 bf16 ULP。位型逐位率作为透明度一并报出。
    got = bf16_bits_to_f32(c2).astype(np.float32)
    exp = bf16_bits_to_f32(ref).astype(np.float32)
    sumabs = (np.abs(a32) @ np.abs(b32).T).astype(np.float32)
    adiff = np.abs(got - exp)
    tol = np.float32(2.5e-3) * sumabs + np.float32(1e-6)
    nbad = int((adiff > tol).sum())
    ok = nbad == 0
    print(f"  [S2 {os.path.basename(dirpath)}] a@b^T: {'PASS' if ok else 'FAIL'} "
          f"(bit-exact {bits_same}/{c2.size} = {bits_same / c2.size:.6f}, T3 超界 {nbad}, "
          f"max|Δ|={float(adiff.max()):.3e})")
    return ok


def check_s3cases(dirpath, m9):
    metas = sorted(glob.glob(os.path.join(dirpath, "m9mt_*_meta.txt")))
    results = []
    for mt in metas:
        prefix = mt[: -len("meta.txt")]
        mut = int(m9.read_meta(mt)["mut"])
        case_ok, _ = m9.check_mt_case(prefix)
        results.append((os.path.basename(prefix), mut, case_ok))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tag-filter", default="")
    args = ap.parse_args()
    m9 = _load_m9_check_ref()

    all_ok = True
    for d in args.dirs:
        s2dirs = sorted(glob.glob(os.path.join(d, "m151_s2_*")))
        s3dirs = sorted(glob.glob(os.path.join(d, "m151_s3_*")))
        if args.tag_filter:
            import re
            s2dirs = [p for p in s2dirs if re.search(args.tag_filter, p)]
            s3dirs = [p for p in s3dirs if re.search(args.tag_filter, p)]
        print(f"== {d} ==")
        for sd in s2dirs:
            # 负向档（prolog_mut3 = S2 GEMM K-1）预期红；干净档预期绿。
            want_red = "prolog_mut3" in sd
            ok = check_s2(sd)
            if want_red:
                if ok:
                    print(f"    [NEGATIVE] S2 {os.path.basename(sd)} 期望变红却没变红 ⇒ FAIL")
                    all_ok = False
            else:
                all_ok &= ok
        # S3：干净档（mut=0 且 tag 不以 mut 结尾）预期全绿；prolog_mut1/2 预期红。
        for sd in s3dirs:
            want_red = ("prolog_mut1" in sd) or ("prolog_mut2" in sd)
            res = check_s3cases(sd, m9)
            if not res:
                continue
            if want_red:
                if all(r for _, _, r in res):
                    print(f"    [NEGATIVE] S3 {os.path.basename(sd)} 期望变红却没变红 ⇒ FAIL")
                    all_ok = False
                else:
                    print(f"    [NEGATIVE] S3 {os.path.basename(sd)} 如预期变红 ✓")
            else:
                all_ok &= all(r for _, _, r in res)
    print(f"== check_prolog: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ==")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
