#!/usr/bin/env python3
"""M101 r2（复审 r1-F3 闭环）：**有判别力**的纪律面计数，替换原来那条空洞的 grep。

原读数（复审 F3 的批评）：README §7① 写「经典 memory-based 向量 API 0 处」，用的模式是
`AscendC::(Cast|Duplicate|Exp|…)\\(`。但抽出的段里有 `using namespace AscendC;`（头文件里就有），
向量/算术调用**都是非限定的** ⇒ 那个模式**永远匹配不到**，"0 处"是被模式保证的，不是代码的性质。

本脚本改成三条**能区分**的测量（都在同一个"正文切片"上做，切片口径与 README §7① 一致）：
  A. **结构**：`__VEC_SCOPE__` 块数、`__simd_vf__` 函数数、`RegTensor` 出现数。
  B. **判别计数（本脚本的核心）**：把 `AscendC::Reg` 命名空间下的向量/算术 API 名
     （Cast/Duplicate/Exp/Add/Sub/Mul/Muls/Max/Min/Reduce/Select/Compare/Reciprocal/Sqrt/Ln/Div/
     LoadAlign/StoreAlign/…）在**每个 `__VEC_SCOPE__` 块内**与**块外**分别计数。
     · 块内 = RegBase SIMD（人类要求的那条路）；
     · 块外 = 唯一的合法情形是搬运/常量填充（`DataCopy` 一族、`Duplicate` 常量填充）与
       矩阵/搬运原语（`Mmad`/`Fixpipe`/`Nd2Nz`/`LoadData`）。
     **这正是"memory-based API 有没有被用来做计算"的判别问题** —— 若块外出现 `Exp/Mul/Add/…`
     就是违规，反之块外只有搬运/填充就是合规。
  C. **资源管理函数**：`TPipe` / `TBuf` / `TQue` / `AllocTensor` / `TBufPool` / `Queue` /
     `Matmul` / `GetTPipePtr` / `TSCM` 的出现数（人类禁止的那一类）。
  D. **旁证**：`LocalTensor` 出现数与 donor 逐字比（抽取件应当完全相同）。

用法：
    python3 audit_discipline.py                 # 默认比对 m15_attn_core.h 与 donor
    python3 audit_discipline.py <a.h> <b.asc>
退出码：0 = 全部读数已列出；1 = 块外出现了"计算类"向量 API（即违规）
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M15 = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(M15, ".."))
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(M15, "m15_attn_core.h")
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "m10_attn_decode", "m10_attn_decode.asc")

# 「计算类」= 若出现在 VEC_SCOPE 之外就是违规（搬运/填充类不在此列）
COMPUTE_OPS = ["Exp", "ExpSub", "Add", "Adds", "Sub", "Mul", "Muls", "MulAddDst", "Max", "Min",
               "Maxs", "Mins", "Div", "Reduce", "Select", "Compare", "Reciprocal", "Sqrt", "Rsqrt",
               "Ln", "Log", "Sigmoid", "Abs", "Neg"]
# 搬运/填充/矩阵类：块外出现是**预期**的（本仓口径豁免，见 m10 README §7 的存量说明）
MOVE_OPS = ["DataCopy", "DataCopyPad", "Duplicate", "Fixpipe", "Mmad", "LoadData", "LoadData2D",
            "LoadData3D", "Nd2Nz", "Cast"]     # Cast 单列（经典 Cast 与 Reg Cast 同名，见下方报告）
RESOURCE_APIS = ["TPipe", "TBuf", "TQue", "AllocTensor", "TBufPool", "Queue", "Matmul",
                 "GetTPipePtr", "TSCM", "TBuf<"]


def slice_body(path):
    """与 README §7① 同一个正文切片口径：`namespace <N> {` 到 `}  // namespace <N>`。"""
    if not path:
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    for ns in ("M15AC", "M15OP", "m10"):
        i = text.find("\nnamespace %s {\n" % ns)
        if i >= 0:
            j = text.find("\n}  // namespace %s\n" % ns)
            if j > i:
                return text[i:j + len(ns) + 18]
    return text


def strip_comments(body):
    """去掉 `//` 行尾注释（本脚本的计数只应看到**代码**）。
    起因（M101 r2 实测）：第 9 类整块替换的注释里引用了被替换掉的原文
    （`Muls(...)` / `Add(...)`），旧版把注释也数进去 ⇒ 报出 2 处"块外计算类调用"的假命中。
    **注释不是代码**，所以先剥注释再计数；这也让"计数有判别力"这条站得住。
    （不处理字符串字面量里的 `//`：本切片里没有这种情形，若将来有了，本函数会切早 —— 属已知限度。）"""
    out = []
    for line in body.splitlines():
        k = line.find("//")
        out.append(line[:k] if k >= 0 else line)
    return "\n".join(out)


def vec_scope_spans(body):
    """返回每个 `__VEC_SCOPE__` 后面那个 `{...}` 的 (start, end) 字符区间（花括号配对）。"""
    spans = []
    for mm in re.finditer(r"__VEC_SCOPE__", body):
        i = body.find("{", mm.end())
        if i < 0:
            continue
        depth = 0
        for k in range(i, len(body)):
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, k + 1))
                    break
    return spans


def count_calls(body, ops):
    """按「词边界 + 可选模板实参 + 左括号」计数（非限定名也数，`Reduce<...>(` 也算）。
    注意：若不做这一步，`StoreAlign<float, StoreDist::…>(…)` 这类**带模板实参**的 RegBase 调用
    会被漏掉 —— 第一版就漏了（StoreAlign 只数到 2，实际更多），是复审 F3 那条"计数要有判别力"的延伸。"""
    out = {}
    for op in ops:
        out[op] = len(re.findall(
            r"(?<![A-Za-z0-9_:])" + re.escape(op) + r"\s*(?:<[^(){}]*>)?\s*\(", body))
    return out


def main():
    pair = ("--pair" in sys.argv)
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    a_path = argv[0] if argv else os.path.join(M15, "m15_attn_core.h")
    b_path = argv[1] if len(argv) > 1 else os.path.join(REPO, "m10_attn_decode", "m10_attn_decode.asc")
    body_a = strip_comments(slice_body(a_path))
    body_b = strip_comments(slice_body(b_path)) if pair else None
    print(f"[audit] A = {a_path}（正文切片、**已剥注释** {len(body_a)} 字符）")
    if pair:
        print(f"[audit] B = {b_path}（正文切片 {len(body_b)} 字符）  —— 逐字对拍的旁证")
    print()

    spans = vec_scope_spans(body_a)
    print("[A 结构]")
    sa = (f"  __VEC_SCOPE__   A={body_a.count('__VEC_SCOPE__')}（配对成功的块 {len(spans)}）"
          f"  __simd_vf__ A={body_a.count('__simd_vf__')}"
          f"  RegTensor A={len(re.findall(r'(?<![A-Za-z0-9_])RegTensor', body_a))}")
    print(sa)
    if pair:
        print(f"                 B={body_b.count('__VEC_SCOPE__')} / {body_b.count('__simd_vf__')} / "
              f"{len(re.findall(r'(?<![A-Za-z0-9_])RegTensor', body_b))}（依次同上）")

    chars = list(body_a)
    for s_, e_ in spans:
        for k in range(s_, e_):
            if chars[k] != "\n":
                chars[k] = " "
    outside = "".join(chars)
    inside = "".join(body_a[s_:e_] for s_, e_ in spans)

    print("\n[B 判别计数：向量/算术 API 在 VEC_SCOPE 块内 vs 块外]")
    allops = COMPUTE_OPS + MOVE_OPS + ["LoadAlign", "StoreAlign", "LocalMemBar"]
    ci = count_calls(inside, allops)
    co = count_calls(outside, allops)
    print(f"  {'op':<12} {'块内':>6} {'块外':>6}   类别")
    for op in allops:
        kind = ("计算类（块外出现即违规）" if op in COMPUTE_OPS else
                "搬运/填充/矩阵（块外属预期）" if op in MOVE_OPS else "Reg 专用（只应出现在块内）")
        print(f"  {op:<12} {ci[op]:>6} {co[op]:>6}   {kind}")

    print("\n[C 资源管理函数（人类禁止的那一类）]")
    for api in RESOURCE_APIS:
        n = len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(api.rstrip("<")) + r"(?![A-Za-z0-9_])",
                           body_a))
        extra = ""
        if pair:
            nb = len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(api.rstrip("<")) +
                                r"(?![A-Za-z0-9_])", body_b))
            extra = f"  B={nb}"
        print(f"  {api:<14} A={n}{extra}")

    print("\n[D 旁证]")
    la = len(re.findall(r"(?<![A-Za-z0-9_])LocalTensor(?![A-Za-z0-9_])", body_a))
    line = f"  LocalTensor 出现数（本切片）  A={la}"
    if pair:
        lb = len(re.findall(r"(?<![A-Za-z0-9_])LocalTensor(?![A-Za-z0-9_])", body_b))
        line += f"  B={lb}  {'相同' if la == lb else '**不同**'}"
    print(line)

    bad = [(op, co[op]) for op in COMPUTE_OPS if co[op]]
    print("\n[结论]")
    n_in = sum(ci[op] for op in COMPUTE_OPS)
    n_move_out = sum(co[op] for op in MOVE_OPS)
    n_reg_out = co["LoadAlign"] + co["StoreAlign"] + co["LocalMemBar"]
    print(f"  块内计算类调用 {n_in} 处；块外计算类调用 {sum(n for _, n in bad)} 处；"
          f"块外搬运用 {n_move_out} 处；块外 Reg 专用原语 {n_reg_out} 处")
    if bad:
        print(f"  ⇒ **块外出现了计算类调用**：{bad}。逐处定位：")
        for op, _n in bad:
            for mm in re.finditer(r"(?<![A-Za-z0-9_:])" + re.escape(op) +
                                  r"\s*(?:<[^(){}]*>)?\s*\(", outside):
                ln = outside.count("\n", 0, mm.start()) + 1
                print(f"     {op:<6} 切片第 {ln:>4} 行  {body_a.splitlines()[ln - 1].strip()[:110]}")
        print("  ⇒ 这一条不是「计数方法」的问题，是**真的违规**（详见 README §7① 与已发塔的 clarify）。")
        return 1
    print("  ⇒ '计算走 RegBase VF、memory-based API 只用于搬运/常量填充' 这条口径在本切片上"
          "**有判别力地**成立（不是靠模式匹配不到）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
