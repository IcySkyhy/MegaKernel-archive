#!/usr/bin/env python3
"""M91-C 紧凑重定尺（M77 §2 第 9 项）的静态判据。

改的是什么：`m15_moe_resources.h` §6 里**每专家张量的行数**从 padded「每专家固定 M_MAX 行」
（`NUM_EXPERTS * M_MAX`）改成链上已经在用的**紧凑 Σt_e 上界** `TOTAL_MAX = M_MAX * TOPK_MAX`。
涉及的 6 项 = `SZ_AQ / SZ_AS / SZ_GU / SZ_HQ / SZ_HS / SZ_Y`（`SZ_H` / `SZ_XSORT` / `SZ_PERM` /
`SZ_INV` / `SZ_WTK` 本来就是紧凑定尺，未动）；`SZ_LOGITS` **不缩**（router 每行必须吐 E 个 logit）。

两类检查：

  A. **文本/形状形态**：6 项的 RHS 已是 `TOTAL_MAX` 基、生成物里用紧凑行号 `rowBase` 寻址的
     WS 符号集合与这 6 项**闭合**（即没有第 7 个被紧凑行号写的 routed 张量仍是 padded 定尺）。
  B. **布局算术**：把 `m15_moe_resources.h` 与 `m15_moe_layer.h` 的 `constexpr uint32_t` 就地
     求值（同一份源码，不做副本），在当前档（E=4/TOPK_MAX=4）与 **E=512 假设档**
     （`--E 512 --topk 10`，只重算算术、不改仓库常量）各核对一遍：
       · 6 项 == `AlignUp(TOTAL_MAX * 单位, 32)`；
       · 6 项在**当前档**等于 padded 算式（⇒ 当前档数值不变，这是「no-op」的算术形态）；
       · `WS_BYTES` 在两档分别等于归档基线 `7,685,472`（当前档）与 M77 §3.2 的独立复算
         `13,891,392`（E=512 紧凑档）；
       · `SZ_LOGITS == AlignUp(M_MAX*NUM_EXPERTS*4, 32)`（随 E 增长、不缩）。

用法：
    python3 check_rescale.py                       # 当前档 + E=512 假设档
    python3 check_rescale.py --E 512 --topk 10
    python3 check_rescale.py --res /tmp/副本.h      # 变异副本（负向对照）
rc：全部 PASS ⇒ 0；任一 FAIL ⇒ 1。
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
M15 = HERE.parent
sys.path.insert(0, str(HERE))

from check_prep import Report, collect, evaluate, _align   # noqa: E402  （同一套就地求值机制）

# 被重定尺的 6 项 → 每行的“单位”表达式（与该行在资源表里的字面量一致）
COMPACT_ITEMS = {
    "SZ_AQ": "HIDDEN / 2",          # A 量化字节（fp4 半字节）
    "SZ_AS": "GU_SCALE_STRIDE",     # A scale（每 32 个 K 元素 1 字节）
    "SZ_GU": "GU_N * 2",            # gate_up GEMM 输出（bf16）
    "SZ_HQ": "INTER / 2",           # H 量化字节
    "SZ_HS": "DN_SCALE_STRIDE",     # H scale
    "SZ_Y": "HIDDEN * 2",           # down GEMM 输出（bf16）
}
# 生成物里用紧凑行号 `rowBase` 寻址的 routed 张量（两侧都登记：routed / shared）
ROUTED_SYMS = ["WS_AQ", "WS_AS", "WS_GU", "WS_HQ", "WS_HS", "WS_Y"]
SHARED_SYMS = [s + "_SHD" for s in ROUTED_SYMS]


def _u(env, unit_expr):
    return int(eval(unit_expr.replace("/", "//"), {"__builtins__": {}}, env))


def shape_checks(rep: Report, header: str, exprs: dict, env_now: dict):
    # ---- A1：6 项的 RHS 已是 TOTAL_MAX 基、不再含 NUM_EXPERTS ----
    bad = []
    for name in COMPACT_ITEMS:
        rhs = exprs.get(name, "<未找到>")
        if "TOTAL_MAX" not in rhs or "NUM_EXPERTS" in rhs:
            bad.append(f"{name} = {rhs}")
    rep.check("A1 6 项每专家张量的 RHS 已是 TOTAL_MAX 基且不含 NUM_EXPERTS",
              not bad, f"不符项 = {bad if bad else '（6 项无一项落在 NUM_EXPERTS 上）'}")

    # ---- A2：生成物里 `rowBase *` 的符号集合与这 6 项（+ 共享 6 项）闭合 ----
    routed, shared, others = [], [], []
    for m in re.finditer(r"\b(WS_[A-Z_]+)\s*\+\s*rowBase\s*\*", header):
        sym = m.group(1)
        (shared if sym.endswith("_SHD") else routed if sym in ROUTED_SYMS else others).append(sym)
    rep.check("A2 routed 侧用紧凑行号 rowBase 寻址的符号集 = 6 项",
              sorted(set(routed)) == sorted(ROUTED_SYMS) and not others,
              f"routed = {sorted(set(routed))}；集外符号 = {sorted(set(others))}")
    rep.check("A2 shared 侧用行号寻址的符号集 = 6 项 + _SHD",
              sorted(set(shared)) == sorted(SHARED_SYMS),
              f"shared = {sorted(set(shared))}")
    # 闭合性：A2 的 routed 集合 == 被重定尺的 6 项集合（**关系式**，不是“0 命中”式断言）
    #   符号名互为映射：SZ_<X>（资源表尺寸）↔ WS_<X>（同一张量的 ws 偏移）
    want_syms = {s.replace("SZ_", "WS_") for s in COMPACT_ITEMS}
    rep.check("A2 紧凑寻址符号集与被重定尺的 6 项**同一集合**",
              set(routed) == want_syms,
              f"routed ∩ 6 项（按 SZ_↔WS_ 映射）= {sorted(set(routed) & want_syms)}；"
              f"routed − 6 项 = {sorted(set(routed) - want_syms)}；"
              f"6 项 − routed = {sorted(want_syms - set(routed))}")

    # ---- A3：routed 行号表达式必须是紧凑前缀和 `offGm.GetValue(slot) + mt * BASE_M` ----
    n_rb = len(re.findall(r"rowBase = shd \? \(mt \* BASE_M\) : \(offGm\.GetValue\(slot\) \+ mt \* BASE_M\);",
                          header))
    rep.check("A3 routed rowBase = offGm.GetValue(slot) + mt*BASE_M（两组 GEMM 各 1 处）",
              n_rb == 2, f"命中 = {n_rb} 处")

    # ---- A4：量化器的 routed 行数由 Σt_e（startOff[NUM_EXPERTS]）驱动，shared 恒为 m ----
    n_rows = len(re.findall(r"rows = startOff\[NUM_EXPERTS\];", header))
    rep.check("A4 量化器 routed 行数 = Σt_e（startOff[NUM_EXPERTS]）",
              n_rows == 1, f"命中 = {n_rows} 处")

    # ---- 诊断读数：6 项的紧凑 vs padded（当前档）----
    for name, unit in COMPACT_ITEMS.items():
        rows_c = env_now["TOTAL_MAX"]
        rows_p = env_now["NUM_EXPERTS"] * env_now["M_MAX"]
        rep.check(f"当前档 {name} 的紧凑/padded 行数（诊断读数）", True,
                  f"{name}: 紧凑 {rows_c} 行 vs padded {rows_p} 行"
                  f"（{'同值' if rows_c == rows_p else f'压缩 {rows_p // max(rows_c, 1)}×'}）",
                  judged=False, site=f"C 行数 {name}")
    return rep


def layout_checks(rep: Report, exprs: dict, tag: str, e: int, topk_max: int):
    rep.scales += 1
    env = evaluate(exprs, {"NUM_EXPERTS": e, "TOPK_MAX": topk_max,
                           "TOTAL_MAX": evaluate(exprs, {})["M_MAX"] * topk_max})
    for name, unit in COMPACT_ITEMS.items():
        u = _u(env, unit)
        want = _align(env["TOTAL_MAX"] * u, env["WS_ALIGN"])
        rep.check(f"[{tag}] {name} == AlignUp(TOTAL_MAX * ({unit}), WS_ALIGN)",
                  env[name] == want,
                  f"NUM_EXPERTS={e}/TOPK_MAX={topk_max}: TOTAL_MAX = {env['TOTAL_MAX']} 行；"
                  f"单位 = {u} B/行；读数 = {env[name]}（期望 {want}）")
    lg = env["SZ_LOGITS"]
    rep.check(f"[{tag}] SZ_LOGITS == AlignUp(M_MAX*NUM_EXPERTS*4, WS_ALIGN)（随 E 增长、不缩）",
              lg == _align(env["M_MAX"] * e * 4, env["WS_ALIGN"]) and lg >= env["M_MAX"] * e * 4,
              f"NUM_EXPERTS={e}: SZ_LOGITS = {lg} B == AlignUp({env['M_MAX'] * e * 4}, 32)；"
              f"≥ M_MAX*E*4 = {env['M_MAX'] * e * 4} B")
    # 充分性：真实档 Σt_e 上界 = M_MAX*min(TOPK_MAX, ...) = M_MAX*TOPK_MAX；6 项行数必须 ≥ 它
    rep.check(f"[{tag}] 6 项行数 ≥ Σt_e 上界（= m*topk ≤ M_MAX*TOPK_MAX = TOTAL_MAX）",
              all(env[n] >= _align(env["TOTAL_MAX"] * _u(env, u), env["WS_ALIGN"])
                  for n, u in COMPACT_ITEMS.items()),
              f"NUM_EXPERTS={e}/TOPK_MAX={topk_max}: Σt_e 上界 = M_MAX*TOPK_MAX = {env['TOTAL_MAX']} 行；"
              f"6 项均按该上界定尺")
    rep.check(f"[{tag}] WS_BYTES（诊断读数）", True,
              f"NUM_EXPERTS={e}/TOPK_MAX={topk_max}: WS_BYTES = {env['WS_BYTES']} B",
              judged=False, site="C WS_BYTES")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(M15 / "m15_moe_resources.h"))
    ap.add_argument("--header", default=str(M15 / "m15_moe_layer.h"))
    ap.add_argument("--E", type=int, default=512, help="假设档的 NUM_EXPERTS（默认 512）")
    ap.add_argument("--topk", type=int, default=10, help="假设档的 TOPK_MAX（默认 10）")
    args = ap.parse_args()

    res_text = pathlib.Path(args.res).read_text()   # 读进来即验路径可读（内容经 exprs 使用）
    header = pathlib.Path(args.header).read_text()
    exprs = collect([args.res, args.header])
    e_now = evaluate(exprs, {})["NUM_EXPERTS"]
    tk_now = evaluate(exprs, {})["TOPK_MAX"]
    env_now = evaluate(exprs, {})

    rep = Report()
    shape_checks(rep, header, exprs, env_now)
    assert res_text, "资源表读到空内容"
    print(f"[check] ---- 布局算术：当前档 NUM_EXPERTS={e_now} / TOPK_MAX={tk_now} ----")
    layout_checks(rep, exprs, f"E={e_now}", e_now, tk_now)

    # ---- 关键恒等式：当前档必须与**改前归档值**逐项同值（这就是「no-op」的算术形态）----
    rep.check("当前档 WS_BYTES == 改前归档值 7,685,472（当前档数值不变）",
              env_now["WS_BYTES"] == 7685472,
              f"WS_BYTES = {env_now['WS_BYTES']}（归档/M77 §3.2「今天」= 7,685,472）")
    for name, unit in COMPACT_ITEMS.items():
        u = _u(env_now, unit)
        want_padded = _align(e_now * env_now["M_MAX"] * u, env_now["WS_ALIGN"])
        rep.check(f"当前档 {name} == padded 算式 AlignUp(NUM_EXPERTS*M_MAX*({unit}), WS_ALIGN)（no-op）",
                  env_now[name] == want_padded,
                  f"紧凑 = {env_now[name]}；padded = {want_padded}")

    if args.E != e_now or args.topk != tk_now:
        print(f"[check] ---- 布局算术：E={args.E} 假设档（只重算算术；仓库常量未改）----")
        env_big = layout_checks(rep, exprs, f"E={args.E}", args.E, args.topk)
        rep.check(f"E={args.E} 档 WS_BYTES == M77 §3.2 的独立复算 13,891,392（紧凑档）",
                  env_big["WS_BYTES"] == 13891392,
                  f"NUM_EXPERTS={args.E}/TOPK_MAX={args.topk}: WS_BYTES = {env_big['WS_BYTES']} B"
                  f"（M77 §3.2「真实·紧凑」= 13,891,392）")
    return rep.done()


if __name__ == "__main__":
    sys.exit(main())
