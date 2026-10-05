#!/usr/bin/env python3
"""M84 规模无关前置修复（#5/#6/#7/#8）的静态判据。

对 `m15_layer_loop/m15_moe_layer.h`（+ `m15_moe_resources.h`）做两类检查：

  A. **文本形态**：四项修复的形态是否在生成物里（标量栈数组已消失、诊断槽不再是写死的
     `offsGm[8]`、unpermute 的 K 链/实例/分发覆盖 topk=10、Y 视图上界写成 `TOTAL_MAX`）。
  B. **布局算术**：把 `m15_moe_resources.h` 与 `m15_moe_layer.h` 里的 `constexpr uint32_t`
     表达式就地求值（同一份源码，不做副本），在当前档（E=4/topk=2/TOPK_MAX=4）与
     **E=512 假设档**（`--E 512 --topk 10`，只重算算术、不改仓库常量）各算一遍，
     核对标量槽仍落在 router-x-块窗内、诊断槽恒在真 offsets 之后且 32B 对齐。

用法：
    python3 check_prep.py                       # 当前档 + E=512 假设档
    python3 check_prep.py --E 512 --topk 10
    python3 check_prep.py --header /tmp/副本.h   # 变异副本（负向对照）
rc：全部 PASS ⇒ 0；任一 FAIL ⇒ 1。
"""

import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
M15 = HERE.parent
REPO = M15.parent


def strip_comments(text: str) -> str:
    """剥掉注释后的**代码视图**（**单趟字符扫描器**；r3 起不再用两趟正则）。

    为什么必须是单趟（r1/r2/r3 三轮复审各挖出一个方向的洞，均是「注释能让判据假绿」）：

    * **r2 的洞（吞代码）**：旧写法「先 `/*...*/` 正则、后按行去 `//`」。本仓 MoE 生成物里有一行
      **行注释**内含 `reg_compute/**`（讲「Reg 侧无 Sort32/MrgSort 等价物」时的路径写法），那个
      `/*` 会被当成**块注释起点**；此后文件里只要再出现一个 `*/`，正则就从那一行**一路吞到那个
      `*/`**，中间的 `constexpr` 全部消失 ⇒ `collect()` 缺符号、`evaluate()` `KeyError`
      （复现：`m95_negctl.sh` 的 `nc10_benign`）。
    * **r2 换成「先 `//` 后 `/*...*/`」后出现的反向洞（注释文本残留）**：块注释的 `*/` 若与 `//`
      **同行**（如 `close // */`），行注释那一趟会把 `*/` 一起吃掉 ⇒ 留下**未闭合的 `/*`**；
      其后若无 `*/`，块注释那一趟**什么都不删** ⇒ **注释文本留在代码视图里**，于是「破坏实现 +
      追加一条注释」又能拿到 PASS（复现：`m95_negctl.sh` 的 `nc11_survivor`）。

    单趟扫描一次遍历同时按**最先出现的**开符号处理 `/*` 与 `//`（并跳过字符串/字符字面量、保留
    换行以维持跨行正则的可用性），两个方向的洞同时消失。**后置自检**：剥完若仍有未闭合的
    `/*`（`state == block`）、未闭合的字面量、或代码视图里仍残留 `/*` / `*/` ⇒ **直接 `raise`**
    —— 剥注释器自己出问题时不静默给假绿（`collect()` 的调用者会看到异常而不是「0 命中」）。

    局限（如实标注）：`'` 一律按字符字面量起点处理，不支持 C++14 的数字分隔符 `1'000'000`
    （本仓受检头文件里没有这种写法；若将来出现，自检会先炸，不会静默错算）。
    """
    out: list[str] = []
    i = 0
    n = len(text)
    state = "code"          # code | line | block | str | chr
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "*":
                state = "block"
                out.append(" ")          # 用空格占位，避免 `a/*x*/b` 粘成 `ab`
                i += 2
                continue
            if ch == "/" and nxt == "/":
                state = "line"
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "chr"
            out.append(ch)
            i += 1
            continue
        if state == "line":
            if ch == "\n":
                state = "code"
                out.append("\n")
            i += 1
            continue
        if state == "block":
            if ch == "*" and nxt == "/":
                state = "code"
                out.append(" ")
                i += 2
                continue
            if ch == "\n":
                out.append("\n")         # 保留行结构：A5/A12 等判据的模式里有 `\n`
            i += 1
            continue
        # str / chr：转义跳过一个字符；只在未转义时收尾
        if ch == "\\" and nxt:
            out.append(ch)
            out.append(nxt)
            i += 2
            continue
        if (state == "str" and ch == '"') or (state == "chr" and ch == "'"):
            state = "code"
        out.append(ch)
        i += 1
    code = "".join(out)
    if state == "block":
        raise ValueError("strip_comments: 输入含未闭合的块注释（/* 之后没有 */）")
    if state in ("str", "chr"):
        raise ValueError(f"strip_comments: 输入含未闭合的 {'字符串' if state == 'str' else '字符'}字面量")
    if "/*" in code or "*/" in code:
        raise ValueError("strip_comments: 剥完仍残留 /* 或 */ —— 剥注释器与输入不自洽（拒绝给假绿）")
    return code


CONST_RE = re.compile(r"constexpr\s+uint32_t\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")


def collect(paths):
    exprs = {}
    for p in paths:
        for name, rhs in CONST_RE.findall(strip_comments(pathlib.Path(p).read_text())):
            exprs[name] = " ".join(rhs.split())
    return exprs


def _align(x, a):
    return (x + a - 1) // a * a


def evaluate(exprs, overrides):
    env = dict(overrides)
    py = {k: re.sub(r"\bAlignUp\s*\(", "_align(", v).replace("/", "//")
          for k, v in exprs.items()}
    for _ in range(24):
        changed = False
        for k, v in py.items():
            if k in env:
                continue
            try:
                env[k] = int(eval(v, {"__builtins__": {}}, {**env, "_align": _align}))
                changed = True
            except Exception:
                pass
        if not changed:
            break
    return env


class Report:
    """判据与诊断读数**分栏计数**（docs/17 §1.1/§4 的分栏口径）：

    - `check(..., judged=True)`（默认）计入「可 FAIL 的判据」；
    - `check(..., judged=False)` 只作**诊断读数**（恒真、不参与 rc），不计入判据数 ——
      避免「硬编码 True 的行」把咬合力计数抬高。
    """

    def __init__(self):
        self.judged = 0
        self.diag = 0
        self.diag_sites = set()
        self.scales = 0
        self.fails = 0

    def check(self, name, ok, reading, judged=True, site=None):
        assert judged or ok, "诊断读数（judged=False）必须是恒真的"
        if judged:
            self.judged += 1
        else:
            assert site, "诊断读数必须给 site（代码点标识），否则「N 个代码点 × M 档」的读数无从复算"
            self.diag += 1
            self.diag_sites.add(site)
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.fails += 1
        print(f"[check] {tag}  {name}: {reading}")

    def done(self):
        print(f"[check] ===== 可 FAIL 的判据 {self.judged} 条（FAIL {self.fails} 条）"
              f" + 诊断读数 {self.diag} 条（{len(self.diag_sites)} 个代码点 × {self.scales} 档跑，"
              f"恒真、不计入判据数）=====")
        return 0 if self.fails == 0 else 1


def scan_checks(rep: Report, header: str, e_now: int, e_big: int, topk_big: int):
    # ---- #5：标量栈数组已消失、UB 槽位已接上 ----
    stacks = [s for s in ("uint32_t cnt[NUM_EXPERTS]", "uint32_t off[NUM_EXPERTS + 1]",
                          "uint32_t cursor[NUM_EXPERTS]") if s in header]
    rep.check("#5 标量栈数组已移除", not stacks,
              f"残留 = {stacks if stacks else '无（cnt/off/cursor 三处均未见）'}")
    ub = [s for s in ("UB_IG_SCAL", "UB_IG_OFF", "UB_IG_CUR") if s in header]
    rep.check("#5 标量数组已指向 UB 槽位", len(ub) == 3, f"命中的槽位符号 = {ub}")
    # 正对照：把检查器指向**改前**的形态必须报 FAIL（由负向对照副本触发，见 m84_neg_controls.log）

    # ---- #6：诊断槽 GM 目的必须**使用** IG_DIAG_GM_SLOT，且不再写死 offsGm[8] ----
    n_slot = header.count("offsGm[IG_DIAG_GM_SLOT]")
    n_lit = header.count("offsGm[8]")
    rep.check("#6 诊断槽 GM 目的使用 IG_DIAG_GM_SLOT（1 处）且无写死的 offsGm[8]",
              n_slot == 1 and n_lit == 0,
              f"offsGm[IG_DIAG_GM_SLOT] = {n_slot} 处；offsGm[8] = {n_lit} 处")
    n_num = header.count("offsGm[NUM_EXPERTS")
    rep.check("#6 未退回 offsGm[NUM_EXPERTS + 8] 的字面形态", n_num == 0,
              f"offsGm[NUM_EXPERTS... = {n_num} 处")

    # ---- #7：K 链 / switch / 实例 / Init 覆盖 topk=10 ----
    fma = sorted(int(m) for m in re.findall(r"FmaChunk<(\d+)>\(", header))
    rep.check("#7 FmaChunk K 链 = 1..9（k=0 由 Mul 承担 ⇒ 覆盖 TK=10）",
              fma == list(range(1, 10)), f"FmaChunk 实参 = {fma}")
    rep.check("#7 FmaChunk 无 <10>（TK=10 只需 1..9）",
              10 not in fma, f"含 <10> = {10 in fma}")
    cases = sorted(int(m) for m in re.findall(r"case (\d+): unperm\d+\.Run", header))
    rep.check("#7 topk 分发 case = 1..9 + default→10",
              cases == list(range(1, 10)) and "default: unperm10.Run" in header,
              f"case = {cases}；default→10 = {'default: unperm10.Run' in header}")
    members = sorted(int(m) for m in re.findall(r"UnpermuteStage<(\d+)> unperm\d+;", header))
    rep.check("#7 模板实例 = <1..10>", members == list(range(1, 11)), f"UnpermuteStage 实参 = {members}")
    inits = sorted(int(m) for m in re.findall(r"unperm(\d+)\.Init\(", header))
    rep.check("#7 Init 覆盖 1..10", inits == list(range(1, 11)), f"Init 实参 = {inits}")

    # ---- #8：Y 视图上界写成紧凑上界 TOTAL_MAX ----
    yview = re.search(r"yGm\.SetGlobalBuffer\(reinterpret_cast<__gm__ bfloat16_t\*>\(ySorted\)[^;]*\);", header)
    rep.check("#8 Y 视图上界 = TOTAL_MAX（紧凑 Σt_e）",
              bool(yview) and "static_cast<uint64_t>(TOTAL_MAX) * HIDDEN" in yview.group(0)
              and "M_MAX * TOPK_MAX" not in yview.group(0)
              and "NUM_EXPERTS * M_MAX" not in yview.group(0),
              f"UnpermuteStage 的 yGm 视图行 = {yview.group(0) if yview else '未找到'}")
    return rep


def layout_checks(rep: Report, exprs: dict, tag: str, e: int, topk_max: int):
    rep.scales += 1
    m_max = evaluate(exprs, {})["M_MAX"]
    env = evaluate(exprs, {"NUM_EXPERTS": e, "TOPK_MAX": topk_max, "TOTAL_MAX": m_max * topk_max})
    win = env["UB_RT_XB"] + env["RT_RB"] * env["HIDDEN"] * 2
    scal = env["UB_IG_SCAL_END"] - env["UB_IG_SCAL"]
    rep.check(f"[{tag}] #5 标量槽随 E 定尺且落在 router-x-块窗内",
              env["UB_IG_SCAL_END"] <= win,
              f"NUM_EXPERTS={e}/TOPK_MAX={topk_max}: 槽 = {scal} B；窗 = UB_IG_SCAL.."
              f"{win}（窗宽 = {win - env['UB_RT_XB']} B）；UB_IG_SCAL_END = {env['UB_IG_SCAL_END']}")
    rep.check(f"[{tag}] #5 栈数组字节数（诊断读数）", True,
              f"3 × {e} × 4 B = {3 * e * 4} B（AIV 标量栈容量在仓内无登记值，不据此判 PASS/FAIL）",
              judged=False, site="#5 栈字节数")
    slot = env["IG_DIAG_GM_SLOT"]
    rep.check(f"[{tag}] #6 诊断槽在真 offsets[0..E] 之后",
              slot >= e + 1, f"NUM_EXPERTS={e}: IG_DIAG_GM_SLOT = {slot} ≥ E+1 = {e + 1}")
    rep.check(f"[{tag}] #6 诊断槽 32B 对齐且在 SZ_OFFSETS 内",
              slot * 4 % 32 == 0 and (slot + 1) * 4 <= env["SZ_OFFSETS"],
              f"NUM_EXPERTS={e}: 字节 = {slot * 4}（%32 = {slot * 4 % 32}）；SZ_OFFSETS = {env['SZ_OFFSETS']}")
    rep.check(f"[{tag}] #8 Y 行数上界 TOTAL_MAX vs padded E*M_MAX（诊断读数）", True,
              f"NUM_EXPERTS={e}: TOTAL_MAX = {env['TOTAL_MAX']} 行；padded = {e * env['M_MAX']} 行",
              judged=False, site="#8 Y 行数")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default=str(M15 / "m15_moe_layer.h"))
    ap.add_argument("--res", default=str(M15 / "m15_moe_resources.h"))
    ap.add_argument("--E", type=int, default=512, help="假设档的 NUM_EXPERTS（默认 512）")
    ap.add_argument("--topk", type=int, default=10, help="假设档的 TOPK_MAX（默认 10）")
    args = ap.parse_args()

    header = pathlib.Path(args.header).read_text()
    exprs = collect([args.res, args.header])
    e_now = evaluate(exprs, {})["NUM_EXPERTS"]
    tk_now = evaluate(exprs, {})["TOPK_MAX"]

    rep = Report()
    scan_checks(rep, header, e_now, args.E, args.topk)
    print(f"[check] ---- 布局算术：当前档 NUM_EXPERTS={e_now} / TOPK_MAX={tk_now} ----")
    layout_checks(rep, exprs, f"E={e_now}", e_now, tk_now)
    if args.E != e_now or args.topk != tk_now:
        print(f"[check] ---- 布局算术：E={args.E} 假设档（只重算算术；仓库常量未改）----")
        layout_checks(rep, exprs, f"E={args.E}", args.E, args.topk)
    return rep.done()


if __name__ == "__main__":
    sys.exit(main())
