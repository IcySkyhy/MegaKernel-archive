#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M95 独立锤点（E=512 / top-10）的数值对拍：设备 dump vs 独立银点。

对拍对象 = `tools/golden/moe_block_ref.py::router_topk`（M4/M26 独立编写、**已建模设备 FTZ**
的 CPU 银点；本脚本只 import，不重写它的语义）。

**为什么需要这一档**（M95 的 mission 与 M91 的 r1 复审都点名）：
  本 mission 改的是 S2 router 的 **top-k 排序实现**（单块 Sort32 → 16 块 Sort32 + 4 级二路
  MrgSort 归并树）。M91 的 r1 复审实测过：把归并树末级换序的变异**仍能过全部 2068 判定项 +
  291 guard + 三条静态判据**，只有数值对拍能咬住。⇒ 「E=4 逐字节零回归」是**回归**证据，
  **不是**新实现正确的证据 —— 本脚本提供后者：在 E=512（> 单块 Sort32 的 32 上限）上把设备
  输出钉在独立银点上。

判定项（任一 FAIL ⇒ rc=1）：
  J1  topk_ids      与银点**逐元素相等**（int32）。这是归并树的**主判据**：级序错 / Extract
                    起点错 / 块数错都会改 id。
  J2  topk_weights  ≤ 16·2⁻²⁴（在**设备 logits** 上按银点规则复算的权重，隔离掉 GEMV 误差后
                    只剩 Exp/Div/求和序的舍入；读数里同时打印实测 ulp）。
  J3  router_logits 逐元素 ≤ T3 界 `ε·Σ|terms| + 0.5·ulp(out)`，ε = (40+6+2560)·2⁻²⁴
                    （设备 = 40 深 lane 链 + 64 lane 树归约；银点 = 2560 次顺序相加）。
硬闸门（不满足即 FAIL，不继续）：
  X1  `x.bin` 必须与 `--x-seed` 重建的确定性激活**逐字节相同**（输入溯源可重建）。
  X2  给 `--w <文件>` 时 `w.bin` 必须与它**逐字节相同**；给 `--w onehot` 时必须等于
      「W[e][k] = (k == e)」的 bf16 构造（合成权重档的溯源）。
报告项（不参与 PASS/FAIL）：
  R1 logits max abs / max rel    R2 每行 topk 权重行和（应 = 1）
  R3 银点端到端权重 vs 设备权重（含 GEMV 链误差的最大偏离，量级参考）
guard：
  G1 logits 每行 max == 0（max-shift 语义）  G2 ids 值域合法且行内互异
退出码：0 = 比过且通过；1 = 比过且有差异；2 = 没得比（输入缺失，**不是通过**）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, "tools", "golden"))
if not os.path.isdir(_GOLDEN):
    _GOLDEN = "/workspace/ascend_mega_kernel/tools/golden"
sys.path.insert(0, _GOLDEN)
from moe_block_ref import router_topk  # noqa: E402

HIDDEN = 2560
E_TIER = 512
KTOPK = 10
EPS_T3 = (40.0 + 6.0 + 2560.0) * 2.0 ** -24
EPS_W = 16.0 * 2.0 ** -24


def gen_activations_uniform(m: int, seed: int) -> np.ndarray:
    """C++ `GenActivationsUniform`（Hash3）的 numpy 逐位复刻（与 m7/m22 同一生成器）。"""
    i = np.arange(m, dtype=np.uint64)[:, None]
    j = np.arange(HIDDEN, dtype=np.uint64)[None, :]
    h = (i * np.uint64(2654435761) + j * np.uint64(40503) + np.uint64(seed * 97)
         + np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(16))) & np.uint64(0xFFFFFFFF)
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) & np.uint64(0xFFFFFFFF)
    exp = (np.uint64(122) + (h % np.uint64(10))) & np.uint64(0xFFFF)
    v = (((h >> np.uint64(8)) & np.uint64(0x8000)) | (exp << np.uint64(7))
         | ((h >> np.uint64(9)) & np.uint64(0x7F)))
    return v.astype(np.uint16)


def onehot_weight() -> np.ndarray:
    w = np.zeros((E_TIER, HIDDEN), dtype=np.uint16)
    w[np.arange(E_TIER), np.arange(E_TIER)] = np.uint16(0x3F80)   # bf16(1.0)
    return w


def bf16_view(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumpdir")
    ap.add_argument("--w", required=True, help="router_weight.bin 路径，或 'onehot'")
    ap.add_argument("--x-seed", type=int, required=True, help="重建 x 的 seed（X1 硬闸门）")
    ap.add_argument("--E", type=int, default=E_TIER)
    ap.add_argument("--topk", type=int, default=KTOPK)
    args = ap.parse_args() if len(sys.argv) > 1 else None
    if args is None:                                    # pragma: no cover
        ap.print_usage()
        return 2
    d = args.dumpdir
    try:
        xbits = np.fromfile(f"{d}/x.bin", dtype=np.uint16).reshape(-1, HIDDEN)
        wbits = np.fromfile(f"{d}/w.bin", dtype=np.uint16).reshape(args.E, HIDDEN)
        log_dev = np.fromfile(f"{d}/router_logits.bin", dtype=np.float32).reshape(-1, args.E)
        ids_dev = np.fromfile(f"{d}/topk_ids.bin", dtype=np.int32).reshape(-1, args.topk)
        w_dev = np.fromfile(f"{d}/topk_weights.bin", dtype=np.float32).reshape(-1, args.topk)
    except OSError as exc:
        print(f"[check_ref] RESULT: SKIPPED (输入缺失 — {exc}；这不是通过)")
        return 2

    m = ids_dev.shape[0]
    if xbits.shape[0] != m or wbits.shape[0] != args.E:
        print(f"[check_ref] RESULT: SKIPPED (形状不匹配：x {xbits.shape}, w {wbits.shape}, m {m})")
        return 2

    # ---- 硬闸门 X1 / X2 ----
    gx = gen_activations_uniform(m, args.x_seed)
    if not bool((gx == xbits).all()):
        print(f"[check_ref] FAIL X1: x.bin 与 --x-seed {args.x_seed} 重建的激活不逐字节一致"
              f"（输入溯源断裂 ⇒ 本档判定无效）")
        return 1
    if args.w == "onehot":
        gonehot = onehot_weight()
        if not bool((gonehot == wbits).all()):
            print("[check_ref] FAIL X2: w.bin 与合成 one-hot 权重不逐字节一致")
            return 1
        w_src = "合成 one-hot（W[e][k] = (k == e) bf16）"
    else:
        ref = np.fromfile(args.w, dtype=np.uint16)
        if ref.size != wbits.size or not bool((ref == wbits.reshape(-1)).all()):
            print(f"[check_ref] FAIL X2: w.bin 与 --w {args.w} 不逐字节一致")
            return 1
        w_src = f"{args.w}（sha256={hashlib.sha256(ref.tobytes()).hexdigest()[:16]}…）"

    x = bf16_view(xbits).reshape(m, HIDDEN)
    w = bf16_view(wbits).reshape(args.E, HIDDEN)

    judged: list[tuple[str, bool, str]] = []
    report: list[str] = []
    guards: list[tuple[str, bool, str]] = []

    # ---- 银点 ----
    log_ref, ids_ref, w_ref = router_topk(x, w, args.topk)

    # J1：ids 逐元素相等
    nbad_id = int((ids_dev != ids_ref).sum())
    rows_bad = int((ids_dev != ids_ref).any(axis=1).sum())
    judged.append(("J1", nbad_id == 0,
                   f"topk_ids 逐元素相等（{ids_dev.size} 项；差异 {nbad_id} 项 / {rows_bad} 行）"))

    # J2：在**设备 logits** 上按银点规则复算权重（隔离 GEMV 误差）
    sc_dev = np.exp(log_dev)
    sc_dev[sc_dev < np.float32(1.1754944e-38)] = np.float32(0.0)
    rows = np.arange(m)[:, None]
    w_on_dev = sc_dev[rows, ids_ref]
    w_on_dev = (w_on_dev / w_on_dev.sum(axis=1, keepdims=True)).astype(np.float32)
    dw = np.abs(w_dev - w_on_dev)
    ulp_w = np.abs(w_dev.view(np.int32) - w_on_dev.view(np.int32))
    judged.append(("J2", bool((dw <= EPS_W).all()),
                   f"topk_weights ≤ {EPS_W:.2e}（在设备 logits 上复算；max abs {float(dw.max()):.3e}，"
                   f"max ulp {int(ulp_w.max())}）"))

    # J3：logits T3 界
    rmax = np.abs(x).max(axis=1)
    wsum = np.abs(w).sum(axis=1)
    ulp = np.ldexp(1.0, np.frexp(np.abs(log_ref))[1] - 24)
    bound = np.maximum(EPS_T3 * rmax[:, None] * wsum[None, :] + 0.5 * ulp, 1e-30)
    diff = np.abs(log_dev - log_ref)
    nbad_lg = int((diff > bound).sum())
    judged.append(("J3", nbad_lg == 0,
                   f"router_logits 逐元素 ≤ T3 界（ε={EPS_T3:.3e}；越界 {nbad_lg} 元素；"
                   f"max abs {float(diff.max()):.3e}，界占用 {float((diff / bound).max()) * 100:.2f}%）"))

    # ---- 报告项 ----
    rel = np.abs(log_dev - log_ref) / (np.abs(log_ref) + 1e-6)
    report.append(f"R1 logits: max abs {float(diff.max()):.3e} / max rel {float(rel.max()):.3e}")
    rowsum = w_dev.sum(axis=1)
    report.append(f"R2 每行 topk 权重行和: min {rowsum.min():.6f} max {rowsum.max():.6f}")
    dw_e2e = np.abs(w_dev - w_ref)
    report.append(f"R3 银点端到端权重 vs 设备权重：max abs {float(dw_e2e.max()):.3e}"
                  f"（含 GEMV 链误差；J2 已在设备 logits 上隔离它）")
    report.append(f"输入 x：确定性重建（seed={args.x_seed}，sha256={hashlib.sha256(xbits.tobytes()).hexdigest()[:16]}…）；"
                  f"权重 w：{w_src}")

    # ---- guard ----
    guards.append(("G1", bool(np.allclose(log_dev.max(axis=1), 0.0, atol=1e-6)),
                   "logits 每行 max == 0（max-shift 语义）"))
    ok_ids = bool((ids_dev >= 0).all() and (ids_dev < args.E).all()
                  and all(len(set(r.tolist())) == args.topk for r in ids_dev))
    guards.append(("G2", ok_ids, f"ids 值域 ∈ [0,{args.E}) 且行内互异（{m} 行）"))

    for name, ok, desc in judged:
        print(f"[check_ref] {name:<3} {'PASS' if ok else 'FAIL'}  {desc}")
    print("[check_ref] ---- 报告项（不参与判定）----")
    for line in report:
        print(f"[check_ref] {line}")
    for name, ok, desc in guards:
        print(f"[check_ref] guard {name:<3} {'ok' if ok else 'NG'}  {desc}")

    npass = sum(1 for _, ok, _ in judged if ok)
    all_ok = npass == len(judged)
    gu = sum(1 for _, ok, _ in guards if ok)
    print(f"[check_ref] m={m} E={args.E} topk={args.topk} : {'PASS' if all_ok else 'FAIL'} "
          f"| 判定 {npass}/{len(judged)} | guard {gu}/{len(guards)}")
    if all_ok:
        print(f"[check_ref] RESULT: OK ({npass}/{len(judged)} 判定项通过，guard {gu}/{len(guards)})")
        return 0
    print(f"[check_ref] RESULT: FAIL ({len(judged) - npass}/{len(judged)} 判定项失败)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
