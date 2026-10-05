#!/usr/bin/env python3
"""M46 取证：把设备 router dump 与 golden `router_topk` 直接对拍，独立复现 M42 finding。

对拍三方（全部由本脚本自己算，不采信任何既有日志）：

  * device        —— dump 里的 `topk_ids.bin`（真机 m22_router512 输出）
  * golden(旧)    —— 不建模 FTZ 的 router_topk（M46 修正前的主干语义，本脚本内联重建）
  * golden(新)    —— 建模 fp32 次正规 FTZ 的 router_topk（M46 修正后）

并打印 FTZ 阈值扫描表，用来证明"只有 2^-126（最小正规数）这一口径与设备吻合"。

用法（只读；不写任何 dump）：
  python3.12 tools/golden/evidence/router_ftz_device_repro.py \
      --dump .tower/worktrees/wt-42/m22_router512/evidence/mode2/real_m4097 \
      --w    .tower/worktrees/wt-42/m22_router512/data/router_weight.bin \
      --x-seed 5
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import moe_block_ref as ref  # noqa: E402

HIDDEN = ref.HIDDEN
FTZ_MIN_NORMAL = np.float32(1.1754944e-38)


def gen_activations_uniform(m: int, seed: int) -> np.ndarray:
    """C++ GenActivationsUniform 的 numpy 逐位复刻（见 m22_router512/check_ref.py）。"""
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


def bf16_view(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def golden_old(x: np.ndarray, w: np.ndarray, top_k: int) -> np.ndarray:
    """M46 修正前的 router_topk ids（不建模 FTZ；内联重建旧语义，不改库）。"""
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    scores = np.exp(logits)
    scores = scores / scores.sum(axis=1, keepdims=True)
    return np.argsort(-scores, axis=1, kind="stable")[:, :top_k].astype(np.int32)


def golden_new(x: np.ndarray, w: np.ndarray, top_k: int) -> np.ndarray:
    return ref.router_topk(x, w, top_k)[1]


FTZ = np.float32(1.1754944e-38)


def _weights_variants(x, w, top_k, renorm=True):
    """各 FTZ 落点口径下的 (ids, weights)，权重按 bf16 RNE 折叠以便与设备比 ulp。

    A: 只 flush exp 输出（排序口径，M42 验证过的那条）
    B: A + flush softmax 除法（scores/sum）结果
    C: A + flush top-k renorm 除法结果
    D: B + C
    """
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    out = {}
    for tag, flush_div, flush_renorm in (("A", False, False), ("B", True, False),
                                         ("C", False, True), ("D", True, True)):
        sc = np.where(e < FTZ, np.float32(0.0), e)
        sc = sc / sc.sum(axis=1, keepdims=True)
        if flush_div:
            sc = np.where(sc < FTZ, np.float32(0.0), sc)
        ids = np.argsort(-sc, axis=1, kind="stable")[:, :top_k].astype(np.int32)
        rows = np.arange(x.shape[0])[:, None]
        wt = sc[rows, ids]
        if renorm:
            with np.errstate(invalid="ignore"):
                wt = wt / wt.sum(axis=1, keepdims=True)
            if flush_renorm:
                wt = np.where(wt < FTZ, np.float32(0.0), wt)
        out[tag] = (ids, wt.astype(np.float32))
    return out


def row_scan(x, w, dev_ids, top_k, thresholds):
    """按给定期望口径各算一遍 ids，报差异（槽/行）。"""
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    out = []
    for name, thr in thresholds:
        if thr is None:
            sc = e / e.sum(axis=1, keepdims=True)
        else:
            sc = np.where(e < np.float32(thr), np.float32(0.0), e)
        ids = np.argsort(-sc, axis=1, kind="stable")[:, :top_k].astype(np.int32)
        slot = int((ids != dev_ids).sum())
        out.append((name, slot, int((ids != dev_ids).any(axis=1).sum())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--w", type=Path, required=True)
    ap.add_argument("--x-seed", type=int, default=5)
    ap.add_argument("--x-sha256", default=None)
    args = ap.parse_args()

    ids_dev = np.fromfile(args.dump / "topk_ids.bin", dtype=np.int32).reshape(-1, 10)
    m = ids_dev.shape[0]
    w = bf16_view(np.fromfile(args.w, dtype=np.uint16).reshape(512, HIDDEN))
    xbits = gen_activations_uniform(m, args.x_seed)
    digest = hashlib.sha256(xbits.tobytes()).hexdigest()
    print(f"# 设备 dump : {args.dump}  m={m}")
    print(f"# 权重       : {args.w}")
    print(f"# x 重生成   : m={m} seed={args.x_seed} sha256={digest}")
    if args.x_sha256:
        print(f"# x sha256   : {'== 记录' if digest == args.x_sha256 else '!= 记录'}"
              f" ({args.x_sha256})")
    x = bf16_view(xbits).reshape(m, HIDDEN)

    old = golden_old(x, w, 10)
    new = golden_new(x, w, 10)
    rows_old = int((old != ids_dev).any(axis=1).sum())
    rows_new = int((new != ids_dev).any(axis=1).sum())
    print(f"\n# 与设备 topk_ids 对拍（判定 = top-10 集合不同）")
    print("# 说明：第一行是脚本内联重建的『不建模 FTZ』旧口径；第二行是本 module 的 "
          "router_topk。")
    print("#       在 M46 修正后的 module 上第二行 = 建模 FTZ 口径（应为 0）；在 M46 修正前"
          "的 module 上（对照日志 *_pre_m46.log）第二行与第一行相同（都是 20）。")
    print(f"内联旧口径（不建模 FTZ）     : {int((old != ids_dev).sum()):5d} 槽 / {rows_old} 行"
          f"  ({rows_old}/{m})")
    print(f"本 module router_topk        : {int((new != ids_dev).sum()):5d} 槽 / {rows_new} 行"
          f"  ({rows_new}/{m})")

    # 库函数（修正后的 moe_block_ref.router_topk）权重侧对拍
    wbits_dev = np.fromfile(args.dump / "topk_weights.bin", dtype=np.uint16).reshape(-1, 10)
    _, ids_lib, wt_lib = ref.router_topk(x, w, 10)
    ulp_lib = np.abs(wbits_dev.astype(np.int32) - ref.f32_to_bf16_bits(wt_lib).astype(np.int32))
    rowsum = ref.bf16_bits_to_f32(
        np.asarray(ref.f32_to_bf16_bits(wt_lib), dtype=np.uint16)).sum(axis=1)
    print(f"golden 新 topk_weights  : bf16 |ulp|>1 {int((ulp_lib > 1).sum())} 槽, "
          f"max ulp {int(ulp_lib.max())}, 行和 ∈ [{rowsum.min():.6f}, {rowsum.max():.6f}]")

    print("\n# 期望口径扫描（flush 阈值 → 与设备的差异）")
    thresholds = [
        ("原样（保留次正规）", None),
        ("< 2^-126 = 1.1755e-38（最小正规数）", 1.1754944e-38),
        ("< 1e-35", 1e-35), ("< 1e-30", 1e-30), ("< 1e-25", 1e-25),
        ("< 1e-20", 1e-20), ("< 1e-15", 1e-15),
    ]
    print(f"{'参考口径':<44}{'槽':>8}{'行':>8}")
    for name, slot, rows in row_scan(x, w, ids_dev, 10, thresholds):
        print(f"{name:<44}{slot:>8}{rows:>8}")

    # ---- FTZ 落点逐一对拍（权重侧）：决定 flush 该插在哪几条语句后 ----
    wbits_dev = np.fromfile(args.dump / "topk_weights.bin", dtype=np.uint16).reshape(-1, 10)
    print("\n# FTZ 落点口径 vs 设备 topk_weights（bf16 ulp）")
    print(f"{'落点口径':<46}{'ids行差':>8}{'ulp>1槽':>9}{'max ulp':>9}")
    for tag, (ids, wt) in _weights_variants(x, w, 10).items():
        wb = ref.f32_to_bf16_bits(wt)
        ulp = np.abs(wbits_dev.astype(np.int32) - wb.astype(np.int32))
        print(f"{tag:<46}{int((ids != ids_dev).any(axis=1).sum()):>8}"
              f"{int((ulp > 1).sum()):>9}{int(ulp.max()):>9}")

    # 次正规带可达性：被选中的 exp 分数落在 [2^-126, sum*2^-126) 即"除法后 flush"可观测
    logits = x @ w.T
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits).astype(np.float32)
    ids_ref = np.argsort(-np.where(e < FTZ, np.float32(0.0), e), axis=1,
                         kind="stable")[:, :10]
    rows = np.arange(m)[:, None]
    esel = e[rows, ids_ref]
    band = (esel >= FTZ) & (esel < esel.sum(axis=1, keepdims=True) * FTZ)
    print(f"\n# 选中 exp ∈ [2^-126, sum·2^-126)（除法-FTZ 可观测的槽）: {int(band.sum())}")
    print(f"# 选中 exp 已被 flush 成 0 的槽: {int((esel == 0).sum())}")
    nz_logit = np.abs(logits[logits != 0])
    print(f"# max-shift 后非零 logits 的 min|x| = {float(nz_logit.min()):.3e}"
          f"（< 2^-126 即 logits 侧也需 FTZ）")

    return 0 if rows_new == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
