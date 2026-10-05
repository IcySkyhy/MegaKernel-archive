#!/usr/bin/env python3.12
"""m19_qsa_indexer/check_select.py —— 选择段（离散段）的**暴力 oracle 逐块/逐元素**判据

为什么单独有这个脚本（M53）：选择段是 T1 离散判据（docs/17 §1.1），"计数对上"不等于"集合对"
—— M35 就是停在"计数差 4 但集合错 84 个 block"的盲区里。本脚本用**全排序 + 平局规则**的暴力
oracle 逐块比对，并把差异集逐块打印出来。

判据的对象是**本核（选择核）实际用于选择的那一份分数向量**（kernel 自己 dump 的 logits[0..V)）：
选择段的正确性 = 「在它拿到的分数上，选中集合恰为 top-k」。分数本身对 double 参考的保真度由
`check_ref.py` 的 T3 判据负责（四档 0 违反）。两者合起来才是完整证明链：
   T3（分数 vs 参考有界） + 本脚本（选择 vs 自身分数的暴力 top-k 精确）
⇒ 不存在"用 T3 的模糊性掩盖选择错误"的空间（本脚本是精确判据，任何置换都会被抓到）。

用法：
  python3.12 m19_qsa_indexer/check_select.py <dump 目录> [case 名...]
  python3.12 m19_qsa_indexer/check_select.py --cross <dump目录A> <dump目录B> [...]   # 多核一致性
"""

import pathlib
import sys

import numpy as np

RATIO = 4


def read_meta(d: pathlib.Path, name: str) -> dict:
    meta = {}
    for line in (d / f"{name}_meta.txt").read_text().splitlines():
        k, _, v = line.partition(" ")
        meta[k] = v
    return meta


def oracle_topk(scores: np.ndarray, k: int) -> set:
    """暴力 oracle：分数降序；平局按 block 索引升序；取前 k 个 block 索引。"""
    # stable argsort 按 (-score) 排序 ⇒ 同分保持原索引升序
    order = np.argsort(-scores, kind="stable")[:k]
    return set(int(i) for i in order)


def check_one(d: pathlib.Path, name: str, verbose: bool = True) -> bool:
    meta = read_meta(d, name)
    V = int(meta["vblocks"])
    tail_start = int(meta["tail_start"])
    tail_count = int(meta["tail_count"])
    # blk_topk：新 dump 直接带；老 dump 退回 budget/ratio
    blk_topk = int(meta.get("blk_topk", int(meta["budget"]) // RATIO))
    k = min(V, blk_topk)
    # packed 布局：块区 [0, 4k)、tail 区固定在**模型列** [budget, budget+tail_count)、count 列 2051。
    # （M19_BUDGET 覆盖只改选多少块，不改这套固定列；见 README §1.6/§4。）
    tail_col0 = int(meta["budget"])

    logits = np.fromfile(d / f"{name}_logits.bin", dtype=np.float32)[:V].astype(np.float64)
    out = np.fromfile(d / f"{name}_out.bin", dtype=np.int32)

    ok = True
    if np.isnan(logits).any():
        print(f"[{name}] ✗ logits 里有 NaN（{int(np.isnan(logits).sum())} 个）= 有列没写 ⇒ 交接失败")
        ok = False

    # ① 选中集合 vs 暴力 oracle（逐块）
    ideal = oracle_topk(logits, k)
    expanded = out[: blk_topk * RATIO]
    got = {}
    bad_tok = 0
    for slot, t in enumerate(expanded):
        if t < 0:
            continue
        b, r = int(t) // RATIO, int(t) % RATIO
        got.setdefault(b, set()).add(r)
        if r < 0 or r >= RATIO:
            bad_tok += 1
    sel = set(got)
    missing = sorted(ideal - sel)
    extra = sorted(sel - ideal)
    dup = sorted(b for b, rs in got.items() if len(rs) != RATIO)
    if verbose:
        print(f"[{name}] V={V} blk_topk={blk_topk} : 选中 {len(sel)} 个 block（oracle {k}）；"
              f"与 oracle 差集 {len(sel ^ ideal)}；非 4 元组 block {len(dup)}")
    if missing or extra or dup or bad_tok:
        print(f"       ✗ 漏选（逐块）{missing[:10]}{'...' if len(missing) > 10 else ''} 共 {len(missing)}")
        print(f"       ✗ 误选（逐块）{extra[:10]}{'...' if len(extra) > 10 else ''} 共 {len(extra)}")
        if dup:
            print(f"       ✗ token 数不为 4 的 block {dup[:10]}")
        if bad_tok:
            print(f"       ✗ 越界 token {bad_tok}")
        ok = False

    # ② 逐元素：每个 token 必须 = 4b+r，且该 block 的每个 r 恰出现一次（已由 ① 覆盖）
    cnt_col = int(out[2051])
    exp_cnt = k * RATIO + tail_count
    if cnt_col != exp_cnt:
        print(f"       ✗ count 列 {cnt_col} != 期望 {exp_cnt}")
        ok = False

    # ③ tail 与 padding（逐元素）
    tail = [int(t) for t in out[tail_col0 : tail_col0 + tail_count]]
    exp_tail = [tail_start + i for i in range(tail_count)]
    if tail != exp_tail:
        print(f"       ✗ tail（固定列 {tail_col0}）{tail} != {exp_tail}")
        ok = False
    gap = out[k * RATIO : tail_col0]                 # 块区与固定 tail 区之间必须全 -1
    if not (gap == -1).all():
        print(f"       ✗ 块区与 tail 区之间的间隙有非 -1：{sorted(set(map(int, gap)))[:5]}（{int((gap != -1).sum())} 个）")
        ok = False
    pad = out[tail_col0 + tail_count : 2051]
    if not (pad == -1).all():
        print(f"       ✗ padding 列有非 -1：{sorted(set(map(int, pad)))[:5]}（{int((pad != -1).sum())} 个）")
        ok = False
    if verbose:
        print(f"       count 列 {cnt_col}；tail {tail}；padding OK；每块 4 token OK")
    return ok


def sel_set(d: pathlib.Path, name: str):
    meta = read_meta(d, name)
    V = int(meta["vblocks"])
    blk_topk = int(meta.get("blk_topk", int(meta["budget"]) // RATIO))
    logits = np.fromfile(d / f"{name}_logits.bin", dtype=np.float32)[:V]
    out = np.fromfile(d / f"{name}_out.bin", dtype=np.int32)
    sel = {int(t) // RATIO for t in out[: blk_topk * RATIO] if t >= 0}
    return logits, sel, meta


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--cross":
        dirs = [pathlib.Path(p) for p in args[1:]]
        if len(dirs) < 2:
            print("[check_select] --cross 需要 ≥2 个 dump 目录")
            return 2
        for d in dirs:
            n = len(list(d.glob("*_meta.txt")))
            print(f"[cross] {d}: *_meta.txt 数量 {n}")
        names = sorted(p.name[: -len("_meta.txt")] for p in dirs[0].glob("*_meta.txt"))
        if not names:
            print("[check_select] ✗ dirs[0] 下没有任何 *_meta.txt ⇒ 没有可比对的输入（exit 2）")
            return 2
        ok = True
        for n in names:
            base_lg, base_sel, base_meta = sel_set(dirs[0], n)
            for d in dirs[1:]:
                try:
                    lg, s, m = sel_set(d, n)
                except FileNotFoundError:
                    print(f"[cross] {n}: ✗ {d} 缺少 dump（不能把「两边都空」当成一致）")
                    ok = False
                    continue
                same_lg = lg.shape == base_lg.shape and bool((lg.view(np.uint32) == base_lg.view(np.uint32)).all())
                same_sel = s == base_sel
                cores_a = base_meta.get("cores", "?")
                cores_b = m.get("cores", "?")
                print(f"[cross] {n}: cores {cores_a} vs {cores_b} ⇒ logits 位同 {same_lg}、选中集合同 {same_sel}"
                      f"（|sel| {len(base_sel)} vs {len(s)}）")
                if not (same_lg and same_sel):
                    print(f"        ✗ 差集 {sorted(base_sel ^ s)[:10]}")
                    ok = False
        print(f"[check_select] ===== {'CROSS-CONSISTENT' if ok else 'CROSS-MISMATCH'} =====")
        return 0 if ok else 1

    d = pathlib.Path(args[0])
    names = args[1:] or [p.name[: -len("_meta.txt")] for p in sorted(d.glob("*_meta.txt"))]
    if not names:
        print(f"[check_select] {d} 下没有 *_meta.txt")
        return 2
    all_ok = True
    for n in names:
        try:
            all_ok &= check_one(d, n)
        except Exception as exc:  # noqa: BLE001
            print(f"[{n}] 检查异常: {exc}")
            all_ok = False
    print(f"[check_select] ===== {'ALL PASS' if all_ok else 'FAILURES PRESENT'} =====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
