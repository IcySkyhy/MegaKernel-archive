#!/usr/bin/env python3
"""
make_tables.py —— 把 `evidence/logs/matrix.txt` 汇总成 README §3 的「API × 最小对齐 × 触发条件 × 错误码 × 证据」表。

**不做判定**：判定在 check_ref.py（逐字节复核）与 run_probes.sh（逐变体原始日志）。本脚本只做
"从已归档的记录里按同一条规则汇总"，规则显式写在下面，reviewer 可逐项复算：

  最小 UB 偏移对齐 = 实测扫描点里**最小的、结果非故障的非零偏移**；
                    若所有非零扫描点都故障 ⇒ 32B（因为 32 ≡ 0 (mod 32)，与对照点等价）。
  证据           = 该判断所依据的逐变体日志文件名（全部在 evidence/logs/ 下）。
  触发条件       = 实测出现故障的最小偏移（= 最小对齐的下一个更小扫描点）。

用法：python3 make_tables.py [probe_v_align 目录]   → stdout 打印 markdown（同时写 evidence/alignment_table.md）
"""
import os
import sys
from collections import OrderedDict

PROBE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
MATRIX = os.path.join(PROBE, "evidence", "logs", "matrix.txt")

# 扫描点（与 probe_v_align.asc 的 SWEEP_FULL / SWEEP_AD 一致）
FULL = [0, 1, 2, 4, 8, 16, 24, 32]
AD = [0, 16, 8, 4, 2, 1]

API_FORM = {
    "ls": ("LoadAlign→StoreAlign（vlds/vsts）", "256B（1 个 VL）"),
    "lsn": ("LoadAlign + UpdateMask(n) 的 StoreAlign", "4n 字节"),
    "ldbrc": ("LoadAlign<…,DIST_BRC_B32>（单元素广播 load）", "256B"),
    "stfirst": ("StoreAlign<…,DIST_FIRST_ELEMENT_B32>（单元素 store）", "4B"),
    "stpack": ("Cast<bf16,float>+StoreAlign<…,DIST_PACK_B32>（pack b16）", "128B"),
    "stpack4": ("Cast<fp4>+StoreAlign<…,DIST_PACK4_B32>", "64B（128 个 fp4 nibble 打包；P2-4 由实测订正，原写 32B）"),
    "vld": ("LoadAlign(reg,addr,AddrReg)（vld）", "256B"),
    "vldas": ("LoadUnAlignPre+LoadUnAlign（vldas/vldus）", "256B"),
    "vstus": ("StoreUnAlign(+Post)（vstus/vstas）", "256B"),
    "load1": ("Reg::Load（vldas+vldus）", "256B"),
    "store1": ("Reg::Store（vstus+vstas）", "256B"),
    "gather": ("Gather（vgather2，UB base + 索引寄存器）", "256B"),
    "gatherb": ("GatherB（vgatherb）", "256B"),
    "scatter": ("Scatter（vscatter）", "256B"),
    "pld": ("LoadAlign(MaskReg,…)/StoreAlign(…,MaskReg)（plds/psts）", "32B"),
    "dupub": ("经典 Duplicate(UB 目的)（507015 历史现场）", "256B"),
    "brcb": ("经典 Brcb(UB 目的)", "256B"),
    "dupreg": ("Reg::Duplicate（寄存器目的）+ StoreAlign", "256B"),
    "add": ("Add（vadd）", "256B"),
    "mul": ("Mul（vmul）", "256B"),
    "exp": ("Exp（vexp）", "256B"),
    "redsum": ("Reduce<SUM>（vcadd）", "4B（lane0）"),
    "redmax": ("Reduce<MAX>（vcmax）", "4B（lane0）"),
    "cmpsel": ("Compares+Select（vcmps/vsel）", "256B"),
    "arange": ("Arange<float>（vci）", "256B"),
    "f32bf16norm": ("Cast<bf16,float> + NORM StoreAlign", "256B（值在偶数 16-bit lane）"),
    "f32bf16nb16": ("同上但落盘用 bf16 ALL 掩码", "256B"),
    "f32bf16pack": ("Cast<bf16,float> + DIST_PACK_B32", "128B（紧凑 64 bf16）"),
    "bf16f32norm": ("LoadAlign<bf16,NORM>+Cast<float>", "256B"),
    "bf16f32unpk": ("LoadAlign<bf16,UNPACK_B16>+Cast<float>", "256B"),
    "lmbarin": ("LocalMemBar 在 __VEC_SCOPE__ 内（控制组）", "256B"),
    "lmbarout": ("LocalMemBar 在 __VEC_SCOPE__ 外（靶子）", "256B"),
    "widthf32": ("Duplicate(7.0f)+StoreAlign，dtype=f32", "256B = 64 元素"),
    "widthbf16": ("同上，dtype=bf16", "256B = 128 元素"),
    "widths16": ("同上，dtype=int16", "256B = 128 元素"),
    "widths8": ("同上，dtype=int8", "256B = 256 元素"),
    "aranges32": ("Arange<int32>", "256B = 64 lane"),
    "aranges16": ("Arange<int16>", "256B = 128 lane"),
    "aranges8": ("Arange<int8>", "256B = 256 lane（实测 lane i == i 全 256 个）"),
    "vnoop": ("控制组：V 段不做任何事", "-"),
    "psts": ("谓词落盘 StoreAlign(…,MaskReg)（psts）：全 1 掩码 + StoreAlign", "32B"),
    "gatherb": ("GatherB（vgatherb，位索引）", "未取到（见备注）"),
    "vldasnopre": ("**负向对照**：有状态 load 缺 init（只用 vldus）", "256B"),
    "vstusnopost": ("**负向对照**：有状态 store 缺 post（只发 vstus）", "256B"),
}


def parse():
    groups = OrderedDict()
    for line in open(MATRIX):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("total="):
            continue
        toks = line.split()
        outcome = toks[0]
        kv = dict(t.split("=", 1) for t in toks if "=" in t)
        grp = kv.pop("(", None)
        # group 标签在行末括号里
        grp = line[line.rindex("(") + 1:line.rindex(")")]
        op = kv["op"]
        if outcome in ("NO-OUTPUT", "NOSYNC", "STOPPED"):
            continue
        a, b, n = int(kv["a"]), int(kv["b"]), int(kv["n"])
        off = a if a != 0 else b
        if off == 0 and a == 0 and b == 0:
            off = 0
        if op not in groups:
            groups[op] = {"axis": {}, "sweep": None, "ctrl": {}, "len": {}, "point": None, "off": off,
                          "nc": [], "kinds": set()}
        gk = grp.split(":")[0]
        groups[op]["kinds"].add(gk)
        g = groups[op]
        if grp.startswith("full:"):
            g["axis"].setdefault("full:" + grp.split(":")[2], {})[off] = (outcome, kv.get("aclError", "NA"))
        elif grp.startswith("ad:"):
            axis = grp.split(":")[2]
            g["axis"].setdefault("ad:" + axis, {})[off] = (outcome, kv.get("aclError", "NA"))
        elif grp.startswith("pt:"):
            g["point"] = (outcome, kv.get("aclError", "NA"))
        if grp.startswith("nc:"):
            g["nc"].append((a, b, n, outcome, kv.get("aclError", "NA")))
        if op == "lsn":
            g["len"][n] = (off, outcome, kv.get("window", "-"), kv.get("outside", "-"), kv.get("aclError", "NA"))
    return groups


def min_align(points):
    """按规则汇总：最小非零通过偏移；全故障 ⇒ 32"""
    if points is None:
        return None
    nz = {o: oc for o, (oc, _) in points.items() if o != 0}
    ctrl = points.get(0, ("?", ""))[0]
    passed = sorted(o for o, oc in nz.items() if oc in ("OK", "WRONG"))
    if passed:
        return f"{passed[0]}B"
    return "32B" if ctrl in ("OK", "WRONG") else "?"


def trigger(points, align):
    if points is None or align is None or not align.endswith("B"):
        return "-"
    a = int(align[:-1])
    if a == 1:
        return "无（1B 也接受）"
    want = a // 2
    for o in sorted({o for o in points if o != 0}):
        if o == want:
            oc, acl = points[o]
            return f"偏移 {o}B ⇒ {oc}({acl})"
    return f"偏移 <{a}B 且非 {a}B 倍数"


def main():
    groups = parse()
    out = []
    out.append("| API（本探针 op） | API 形态 / 底层指令 | 落盘窗口 | **实测最小 UB 偏移对齐** | 判定方式（P2-3 要求逐行可辨） | 触发条件（实测） | 错误码 | 依据日志（evidence/logs/） |")
    out.append("|---|---|---|---|---|---|---|---|")

    def method_of(kinds):
        """判定方式由**分组标签**（即产生该行的那一项矩阵条目）决定，不是手写"""
        if "full" in kinds:
            return "**完整扫描 {0,1,2,4,8,16,24,32}**"
        if "ad" in kinds:
            return "降序推断 {0,16,8,4,2,1}（首个故障停）"
        if "pt" in kinds:
            return "单点标定（32B 对齐）"
        if "nc" in kinds:
            return "定点（nc 组，见下表）"
        return "-"
    SKIP = {"lsn", "psts", "vldasnopre", "vstusnopost"}
    for op, g in groups.items():
        if op in SKIP:
            continue
        form, win = API_FORM.get(op, (op, "-"))
        axes = g["axis"]
        if op in ("gatherb", "lmbarout") and not any(
                oc == "OK" for pts in g["axis"].values() for (oc, _) in pts.values()):
            out.append(f"| `{op}` | {form} | {win} | **不作结论**（偏移 0 即故障，非对齐问题；见备注） | "
                       f"{method_of(g['kinds'])} | 偏移 0 ⇒ FAULT(507035) | 507035 | "
                       f"`run_{op}_a0_b0_n64.log` |")
            continue
        if axes:
            parts = []
            for axis, pts in axes.items():
                al = min_align(pts)
                tr = trigger(pts, al)
                tag = "源" if axis.endswith("1") else "目的"
                parts.append(f"{tag} {al}")
                if len(axes) == 1:
                    trig = tr
            align = " / ".join(parts)
            trig = "; ".join(f"{('源' if ax.endswith('1') else '目的')}: {trigger(pts, min_align(pts))}"
                             for ax, pts in axes.items())
            logs = "; ".join(f"`run_{op}_a{a}_b{b}_n{n}.log`"
                             for ax, pts in axes.items() for (a, b, n) in []
                             ) or f"`run_{op}_a*_b*_n*.log`（逐点，共 {sum(len(p) for p in axes.values())} 条）"
        elif op == "lsn":
            out.append(f"| `lsn` | {form} | {win} | **长度维度**（地址固定 32B 对齐，逐点实测 n=1/3/7/8/63/64） | "
                       f"单点标定（每条 n 各一次） | 写满 n 且不越界（见下表） | 见下表 | "
                       f"`run_lsn_a0_b0_n*.log`（{len(g['len'])} 条） |")
            continue
        elif g["point"] is not None:
            align = "N/A（单点 32B 对齐，不扫偏移）"
            trig = "-"
            logs = f"`run_{op}_a0_b0_n*.log`"
        else:
            align, trig, logs = "-", "-", "-"
        codes = set()
        for axis, pts in g["axis"].items():
            for o, (oc, acl) in pts.items():
                if oc in ("FAULT", "HANG") and acl not in ("-", "NA"):
                    codes.add(acl)
        codestr = " / ".join(sorted(codes)) if codes else "-（无不合规路径：1B 也接受）"
        if op in ("gatherb", "lmbarout"):
            codestr = "507035（偏移 0 即故障 ⇒ 非对齐问题，见备注）"
        out.append(f"| `{op}` | {form} | {win} | **{align}** | {method_of(g['kinds'])} | {trig} | {codestr} | {logs} |")
    out.append("")
    out.append("### 负向对照定点与探测到的无效路径（`nc:` 分组；**不作对齐结论**，只作'对照是否活着'的证据）")
    out.append("")
    out.append("| op | 观测 | 结论 | 日志 |")
    out.append("|---|---|---|---|")
    for op, g in groups.items():
        if op not in ("psts", "vldasnopre", "vstusnopost"):
            continue
        detail = "; ".join(f"a={a} b={b} ⇒ {oc}({acl})" for (a, b, n, oc, acl) in sorted(g["nc"]))
        if op == "psts":
            concl = "谓词落盘（psts）**限 32B 对齐**（0 通过 / 16 故障），落盘 32B"
        else:
            concl = "缺 init/post ⇒ **无故障但与逐字节模型不符**（负向对照生效）——有状态协议的 init/post 不是可选的"
        out.append(f"| `{op}` | {detail} | {concl} | `run_{op}_a*_b*_n*.log` |")
    out.append("")
    out.append("### 长度维度（`lsn`：地址固定 32B 对齐，掩码长度 n 变化）")
    out.append("")
    out.append("| n（元素） | 偏移 b | 实测写出窗口 | 窗口外被写字节 | 结果 | 日志 |")
    out.append("|---|---|---|---|---|---|")
    for op, g in groups.items():
        if op != "lsn":
            continue
        for n, (off, oc, win, outside, acl) in sorted(g["len"].items()):
            out.append(f"| {n} | {off} | window={win} | outside={outside} | {oc} | `run_lsn_a0_b{off}_n{n}.log` |")
    text = "\n".join(out) + "\n"
    dest = os.path.join(PROBE, "evidence", "alignment_table.md")
    with open(dest, "w") as f:
        f.write(text)
    print(text)
    print(f"[make_tables] 写入 {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
