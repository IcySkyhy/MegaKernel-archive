#!/usr/bin/env python3.12
"""M124：从 `m15_layer_resources.h` 的 `FLAG_SEQ[]` **解析**出逐 (核型, mode) 的 id 执行序与用量表。

为什么要有它（r1 复审 P2-1 的教训）：塔的硬约束 #3 要的是「**跑相邻性检查 + 贴逐核 id 用量表**」，
而初版的表是**手抄**的 ⇒ 抄漏了 AIC mode2 对 12/13 的使用（登记表当时把这两行误标成 `"AIV"`）。
本脚本把那张表**从代码里算出来**，并**独立重实现**一遍 C++ 侧的判定（`FlagSeqAdjacentOk()` /
`FlagMaxUse()` 的语义，含 `CoreSame` 对 `"both"` 的处理），用来与编译期 static_assert 互为见证。

用法（仓库根）：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/evidence/ple_wire/m124/M124_flagid_table.py
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent.parent.parent / "m15_layer_resources.h"      # m15_layer_loop/m15_layer_resources.h
if not RES.exists():
    RES = HERE.parent.parent.parent.parent / "m15_layer_loop" / "m15_layer_resources.h"

# FLAG_SEQ[] 的行形如： {"seg", "core", mode, ID, reuse, "use"}
# ID 可能是 M15H::FLAG_XX / M85P::FLAG_YY / 数字 / 别名常量（如
# `constexpr uint16_t FLAG_PLE_IN_BOUND_AIV = FLAG_HC0_BOUND_AIV;`）—— 三种都要解出来
CONST_PAT = re.compile(r"^\s*constexpr\s+uint16_t\s+(\w+)\s*=\s*([^;]+);", re.M)


def load_consts():
    vals = {}
    raw = {}
    for name in ("m15_hc_resources.h", "m15_moe_resources.h", "m15_gdn_resources.h", "m15_ple.asc",
                 "m15_layer_resources.h"):
        p = RES.parent / name
        if not p.exists():
            continue
        for m in CONST_PAT.finditer(p.read_text()):
            raw.setdefault(m.group(1), m.group(2).strip())

    def resolve(name, depth=0):
        if depth > 8:
            raise SystemExit("[FAIL] 常量别名链太深：%s" % name)
        if name in vals:
            return vals[name]
        if name not in raw:
            raise SystemExit("[FAIL] 常量解析不到：%s" % name)
        expr = raw[name]
        expr = expr.split("::")[-1] if expr.startswith("M15") else expr
        # 处理 "M15H::FLAG_AC0" / "FLAG_HC0_BOUND_AIV" / "8"
        tok = expr.split("::")[-1].strip()
        if tok.isdigit():
            vals[name] = int(tok)
        elif tok in raw:
            vals[name] = resolve(tok, depth + 1)
        else:
            raise SystemExit("[FAIL] 常量表达式的形式不认识：%s = %s" % (name, expr))
        return vals[name]

    for name in list(raw.keys()):
        resolve(name)
    return vals


def load_rows():
    txt = RES.read_text()
    body = txt.split("constexpr FlagStep FLAG_SEQ[] = {", 1)[1].split("};", 1)[0]
    rows = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        # 去掉注释后按顶层逗号切
        line = line.rstrip(",")
        if "//" in line:
            line = line[: line.index("//")]
        parts = [p.strip() for p in line.strip("{}").split(",")]
        if len(parts) < 5:
            continue
        seg, core, mode, ident, reuse = parts[0], parts[1], parts[2], parts[3], parts[4]
        use = parts[5] if len(parts) > 5 else ""
        rows.append(
            dict(seg=seg.strip('"'), core=core.strip('"'), mode=int(mode), ident=ident.strip(),
                 reuse=reuse.strip() == "true", use=use.strip().strip('"'))
        )
    return rows


def core_id(c):
    return 0 if c == "AIV" else (1 if c == "AIC" else 2)


def core_same(a, b):
    x, y = core_id(a), core_id(b)
    return x == y or x == 2 or y == 2


def main():
    consts = load_consts()
    rows = load_rows()
    if not rows:
        print("[FAIL] 没解析到 FLAG_SEQ 行")
        return 2

    def resolve(ident):
        if ident.isdigit():
            return int(ident)
        name = ident.split("::")[-1]
        if name not in consts:
            raise SystemExit("[FAIL] 常量解析不到：%s" % ident)
        return consts[name]

    for r in rows:
        r["id"] = resolve(r["ident"])

    print("== 解析到的 FLAG_SEQ 行：%d 条 ==" % len(rows))
    print("（本脚本的语义与 `m15_layer_resources.h` 的 `FlagSeqAdjacentOk()`/`FlagMaxUse()` 同口径：")
    print("  相邻 = 数组序里下一条**同 mode 且 CoreSame** 的行；`both` 与任何核型都 CoreSame）\n")

    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["core"], r["mode"]), []).append(i)

    bad = 0
    for (core, mode), idxs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        seq = [(rows[i]["id"], rows[i]["seg"], rows[i]["ident"]) for i in idxs]
        ids = [s[0] for s in seq]
        print("-- %s · mode%d：%d 个同步点" % (core, mode, len(seq)))
        print("   执行序 id: " + " → ".join("%d(%s)" % (s[0], s[1]) for s in seq))
        # 相邻性（同口径）
        prev = None
        for k, i in enumerate(idxs):
            nxt = None
            for j in idxs[k + 1:]:
                if core_same(rows[i]["core"], rows[j]["core"]):
                    nxt = rows[j]
                    break
            if nxt is not None and nxt["id"] == rows[i]["id"]:
                print("   [BAD] 相邻同号: %s(%d) 与 %s(%d)" % (rows[i]["seg"], rows[i]["id"], nxt["seg"], nxt["id"]))
                bad += 1
        cnt = {}
        for s in seq:
            cnt[s[0]] = cnt.get(s[0], 0) + 1
        print("   每 id 用量（**全表口径**：含只有非 PLE 层才走的 `hc(H1)` 行；＝ C++ 侧检查所见）: "
              + ", ".join("%d×%d" % (k, v) for k, v in sorted(cnt.items())))
        print()

    mx = 0
    for (core, mode), idxs in groups.items():
        ids = [rows[i]["id"] for i in idxs]
        for iid in set(ids):
            n = sum(1 for i in idxs if core_same(rows[i]["core"], core) and rows[i]["id"] == iid)
            if n > mx:
                mx = n
    print("== 独立重算的判定 ==")
    print("FlagSeqAdjacentOk() 口径：%s" % ("通过（相邻对无同号）" if bad == 0 else "**违规 %d 处**" % bad))
    print("FlagMaxUse() 口径：同一 (核型,mode,id) 的最大用量 = %d（C++ 侧 static_assert 的上限是 6，硬件 4bit 计数器上限 15）" % mx)

    # ---- AIC / AIV 两侧在**层 1** 的真实执行序与用量（去掉非 PLE 形态的 hc(H1) 行）----
    #   ⚠ 两种口径必须分清（r1/r2 复审两次点名的同一类问题）：
    #     · **全表口径** = `FLAG_SEQ` 原样（同一张表要覆盖「PLE 层」与「非 PLE 层」两种层形态，
    #       所以含 `hc(H1)` 那一段）＝ **C++ 侧 `FlagSeqAdjacentOk()` / `FlagMaxUse()` 所见**；
    #     · **层 1 口径** = 去掉只有非 PLE 层才走的 `hc(H1)` 行 ⇒ PLE 打断点层的真实执行序。
    print("\n== 层 1（PLE 打断点层）的真实执行序与用量（**层 1 口径**：去掉非 PLE 层才有的 hc(H1) 行）==")
    for core in ("AIV", "AIC"):
        for mode in (2, 0):
            seq = [r for r in rows if r["core"] in (core, "both") and r["mode"] == mode
                   and r["seg"] not in ("hc(H1)",)]
            if not seq:
                continue
            cnt = {}
            for r in seq:
                cnt[r["id"]] = cnt.get(r["id"], 0) + 1
            adj = all(seq[k]["id"] != seq[k + 1]["id"] for k in range(len(seq) - 1))
            print("   %s · mode%d: %s" % (core, mode, " → ".join("%d(%s)" % (r["id"], r["seg"]) for r in seq)))
            print("      用量: " + ", ".join("%d×%d" % (k, v) for k, v in sorted(cnt.items()))
                  + "；相邻对%s" % ("无同号" if adj else "**有同号（BAD）**"))
            if not adj:
                bad += 1
    print("\n注：`bound` = PLE 挂载点的两条相位边界（`FLAG_PLE_IN/OUT_BOUND_AIV` = `FLAG_HC0/HC1_BOUND_AIV`"
          " = 8/9）；`h1→a` / `a→h2` / `h2→b` 三条在表里同样以 `bound` 记（同号复用，见 `m15_layer_kernel.h` 的注释）。")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
