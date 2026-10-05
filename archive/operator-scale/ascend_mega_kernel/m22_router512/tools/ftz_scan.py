# SPDX-License-Identifier: Apache-2.0
"""设备 softmax 下溢口径的实测扫描（M42 报告项 R0 的复现器）。

背景：设备 softmax 分数落在 fp32 次正规区时被硬件 FTZ flush 成 0（docs/05 §6.1）；
golden `moe_block_ref.router_topk` 保留次正规值。本脚本在**同一份设备 dump** 上，用不同
的参考口径复算 top-10，统计与设备的 ids 差异，从而钉出设备到底是哪一种口径。

用法：
    python3.12 ftz_scan.py <dumpdir> --w <router_weight.bin|onehot> [--x-seed N]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/workspace/ascend_mega_kernel/tools/golden")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from moe_block_ref import router_topk  # noqa: E402
from check_ref import bf16_view, gen_activations_uniform, onehot_weight  # noqa: E402

E = 512
KTOPK = 10
FP32_MIN_NORMAL = 1.1754944e-38


def select(scores: np.ndarray, k: int = KTOPK) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumpdir")
    ap.add_argument("--w", required=True)
    ap.add_argument("--x-seed", type=int, default=None)
    args = ap.parse_args()

    dev_ids = np.fromfile(f"{args.dumpdir}/topk_ids.bin", dtype=np.int32).reshape(-1, KTOPK)
    m = dev_ids.shape[0]
    try:
        x = bf16_view(np.fromfile(f"{args.dumpdir}/x.bin", dtype=np.uint16)).reshape(m, 2560)
        src = f"{args.dumpdir}/x.bin"
    except FileNotFoundError:
        assert args.x_seed is not None, "无 x.bin 时必须给 --x-seed"
        x = bf16_view(gen_activations_uniform(m, args.x_seed)).reshape(m, 2560)
        src = f"重生成 seed={args.x_seed}"
    if args.w == "onehot":
        w = bf16_view(onehot_weight()).reshape(E, 2560)
    else:
        w = bf16_view(np.fromfile(args.w, dtype=np.uint16).reshape(E, 2560))

    logits, _, _ = router_topk(x, w, KTOPK)   # 只用 golden 的 logits（与口径无关）
    sc = np.exp(logits)
    print(f"# FTZ / 下溢口径扫描   m={m}   x={src}   w={args.w}")
    print(f"# logits(max-shifted) 范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"{'参考口径':<44} {'ids 差异槽位':>12} {'差异行':>8}")
    print(f"{'golden 原样（保留次正规）':<44} {int((select(sc) != dev_ids).sum()):>12} "
          f"{int((select(sc) != dev_ids).any(1).sum()):>8}")
    for thr in (FP32_MIN_NORMAL, 1e-35, 1e-30, 1e-25, 1e-20, 1e-15):
        s2 = sc.copy()
        s2[s2 < thr] = np.float32(0.0)
        o = select(s2)
        print(f"{'scores < ' + f'{thr:.4e}' + ' → 0 (FTZ 建模)':<44} "
              f"{int((o != dev_ids).sum()):>12} {int((o != dev_ids).any(1).sum()):>8}")
    for lo in (-64, -70, -80, -87, -100):
        o = select(np.where(logits < lo, np.float32(0.0), sc))
        print(f"{'logit < ' + str(lo) + ' → score 0 (输入钳位)':<44} "
              f"{int((o != dev_ids).sum()):>12} {int((o != dev_ids).any(1).sum()):>8}")
    s2 = sc.copy()
    s2[s2 < FP32_MIN_NORMAL] = np.float32(0.0)
    ftz_diff = int((select(s2) != dev_ids).sum())
    naive_diff = int((select(sc) != dev_ids).sum())
    print()
    if naive_diff == 0 and ftz_diff == 0:
        print("结论：该档参考分数未触及次正规区，所有口径都吻合 ⇒ 本档对口径**无区分度**"
              "（用于钉口径请用有下溢的大 m 档，如 real_m4097）。")
    else:
        print(f"结论：只有「scores < 2^-126（fp32 最小正规数）→ 0」这一条与设备逐元素吻合"
              f"（该口径差异 {ftz_diff} 槽，原样 golden 差异 {naive_diff} 槽）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
