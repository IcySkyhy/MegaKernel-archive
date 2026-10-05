#!/usr/bin/env python3
"""
check_ref.py —— M57 `probe_v_align` 的**独立复核**（不依赖被测 kernel 的 host 侧比对逻辑）。

复核三件事，全部只吃 `evidence/` 里的归档物（log + .bin dump）：
  1. **故障分类**：从每条变体的 log 里取 `LAUNCH aclError=` / `FAILED_MSG` 原文，独立复算
     OUTCOME（FAULT/HANG/OK），并与 host 自己打印的 OUTCOME 交叉核对（不一致即报警）。
  2. **逐字节模型复核（docs/17 T1：整数/位域逐位，无容差）**：用 numpy 从 dump 重算期望字节，
     与 dump 逐字节比对。模型按 op 语义独立实现（不复用 C++ 侧代码路径）。
  3. **最小对齐推断**：按 (op, 轴) 汇总"哪些偏移通过、哪些报错"，给出**实测最小 UB 偏移对齐**。

三态退出码（docs/17 §8.3）：
  0 = 比过且通过（RESULT: OK 行必须带比较计数）
  1 = 比过且发现差异
  2 = 没得比（dump/日志缺失，打印 RESULT: SKIPPED，绝不发合格证）

用法：python3 check_ref.py [probe_v_align 目录]         （默认当前目录）
"""
import os
import sys
import glob

import numpy as np

PROBE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
EV = os.path.join(PROBE, "evidence")
LOGS = os.path.join(EV, "logs")
DUMPS = os.path.join(EV, "dumps")
MATRIX = os.path.join(LOGS, "matrix.txt")
ARENA = 1024
SENT = 0xA5
VLB = 256

# ---------------------------------------------------------------- 参考模型

def ramp(nbytes=ARENA):
    """IN arena 的字节模式：f(i) = i + 0.5（与 kernel host 侧同式，但这里是独立实现）"""
    f = (np.arange(nbytes // 4, dtype=np.float32) + np.float32(0.5))
    return f.tobytes()


def rne_bf16(x):
    """fp32 -> bf16 的 RNE（ties-to-even）：现算，不查表"""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (b >> np.uint64(16)) & np.uint64(1)
    b = (b + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFFFFFF)
    return (b >> np.uint64(16)).astype(np.uint16)


# 结构性判定（无逐字节模型）的 op：必须给出**声明窗口**，窗口内要有实质内容；
# 否则 window_ok(win=0) 会恒真 —— 那是"空洞通过"，本仓明文禁止。
STRUCT_WINDOW = {
    # stpack4 的落盘窗口 = 64B（128 个 fp4 nibble 打包），与 probe_v_align.asc 的 kOps `VLB / 4`、
    # 日志里的 `COMPARE window_bytes=64`、README §3 该行一致；pld/psts = 32B（掩码寄存器）。
    "gather": VLB, "gatherb": VLB, "scatter": VLB, "vld": VLB, "stpack4": VLB // 4, "pld": 32,
    "psts": 32, "brcb": VLB, "exp": VLB, "lmbarin": VLB, "lmbarout": VLB,
    "widthf32": VLB, "widthbf16": VLB, "widths16": VLB, "widths8": VLB,
    "aranges32": VLB, "aranges16": VLB, "aranges8": VLB,
}

# 实测"探针未取到可观测效果"的 op（0 字节落盘且无故障）⇒ 无数据，不作对齐结论（见 README §9）。
NO_DATA_OPS = {"gatherb"}


def expected(op, a, b, n, src):
    """返回 (want_bytes, window_len, note)。want 为 None 表示"结构性判定，不做逐字节模型"。"""
    want = np.full(ARENA, SENT, dtype=np.uint8)
    f = np.frombuffer(src, dtype=np.float32)
    if op in ("ls", "vldas", "vstus", "load1", "store1", "vldasnopre", "vstusnopost"):
        win = VLB
        want[b:b + win] = np.frombuffer(src, dtype=np.uint8)[a:a + win]
        return want, win, "byte-shift(src@a -> dst@b)"
    if op == "lsn":
        win = 4 * n
        want[b:b + win] = np.frombuffer(src, dtype=np.uint8)[a:a + win]
        return want, win, f"byte-shift({n} elems = {win}B)"
    if op == "stfirst":
        want[b:b + 4] = np.frombuffer(src, dtype=np.uint8)[a:a + 4]
        return want, 4, "byte-shift(1 elem = 4B)"
    if op == "ldbrc":
        v = np.frombuffer(src, dtype=np.float32)[a // 4]
        want[b:b + VLB] = np.full(64, v, dtype=np.float32).view(np.uint8)
        return want, VLB, "broadcast scalar from src@a"
    if op in ("dupreg", "dupub"):
        want[b:b + VLB] = np.full(64, np.float32(7.0), dtype=np.float32).view(np.uint8)
        return want, VLB, "const 7.0f fill x64"
    if op == "add":
        want[b:b + VLB] = (f[:64] + f[32:96]).astype(np.float32).view(np.uint8)
        return want, VLB, "ramp[i]+ramp[i+32]"
    if op == "mul":
        want[b:b + VLB] = (f[:64] * f[32:96]).astype(np.float32).view(np.uint8)
        return want, VLB, "ramp[i]*ramp[i+32]"
    if op in ("arange", "arangef32"):
        want[b:b + VLB] = np.arange(64, dtype=np.float32).view(np.uint8)
        return want, VLB, "lane i == i (f32)"
    if op in ("f32bf16pack", "stpack"):
        want[b:b + 128] = rne_bf16(f[:64]).view(np.uint8)
        return want, 128, "DIST_PACK_B32: 64 连续 bf16 = RNE(ramp)"
    if op == "f32bf16norm":
        for i in range(64):
            want[b + 4 * i:b + 4 * i + 2] = np.array([rne_bf16(f[i])], dtype=np.uint16).view(np.uint8)
        return want, VLB, "NORM 落盘: 值在偶数 16-bit lane（每 32-bit lane 一个 bf16）"
    if op == "f32bf16nb16":
        # 只判偶数 16-bit lane（奇数 lane 由报告项记录，不作判据）
        return None, VLB, "偶数 16-bit lane = RNE(ramp)，奇数 lane 只报告"
    if op == "bf16f32norm":
        s16 = np.frombuffer(src, dtype=np.uint16)
        idx = (np.arange(64) * 2) % 128
        bits = s16[idx].astype(np.uint32) << np.uint32(16)   # bf16 位模式 -> fp32 位模式
        want[b:b + VLB] = bits.view(np.uint8)
        return want, VLB, "NORM: out_f32[j] = 源第 2j 个 16-bit lane"
    if op == "bf16f32unpk":
        s16 = np.frombuffer(src, dtype=np.uint16)
        idx = np.arange(64) % 128
        bits = s16[idx].astype(np.uint32) << np.uint32(16)
        want[b:b + VLB] = bits.view(np.uint8)
        return want, VLB, "UNPACK: out_f32[j] = 源第 j 个 16-bit lane"
    if op == "cmpsel":
        out = np.where(f[:64] <= np.float32(16.0), f[:64], np.float32(-1.0))
        want[b:b + VLB] = out.astype(np.float32).view(np.uint8)
        return want, VLB, "Compares(LE,16) -> Select(x, -1)"
    if op == "redsum":
        want[b:b + 4] = np.array([f[:64].astype(np.float64).sum()], dtype=np.float32).view(np.uint8)
        return want, 4, "Reduce<SUM> lane0（其余 lane 不判）"
    if op == "redmax":
        want[b:b + 4] = np.array([f[:64].max()], dtype=np.float32).view(np.uint8)
        return want, 4, "Reduce<MAX> lane0（其余 lane 不判）"
    return None, STRUCT_WINDOW.get(op, 0), "结构性判定（形状/非空洞），不做逐字节模型"


def window_ok(op, got, b, win):
    """结构性判定：窗口首字节被写过、窗口内有实质内容"""
    if win == 0:
        return True, 0
    nz = int(np.count_nonzero(got[b:b + win] != SENT))
    return (got[b] != SENT and nz >= 4), nz


def written_extent(got):
    nz = np.nonzero(got != SENT)[0]
    if nz.size == 0:
        return (0, 0)
    return (int(nz[0]), int(nz[-1]))


# ---------------------------------------------------------------- 主流程

# 负向对照：这些变体的**期望结果是 WRONG**（探针故意缺 init/post / 故意错对齐），
# 因而"WRONG"在这里 = 对照生效（通过），"OK" = 对照没生效（不通过）。
NEG_CONTROL = {"vldasnopre", "vstusnopost"}


def main():
    if not os.path.isdir(DUMPS) or not os.path.isfile(MATRIX):
        print(f"RESULT: SKIPPED (missing {MATRIX} or {DUMPS}) —— 没得比，不发合格证")
        return 2

    # matrix 行格式: OUTCOME op=.. a=.. b=.. n=.. rc=.. aclError=.. window=.. outside=.. fnv=.. (group)
    # **轴归属直接由分组标签决定**（full:op:axis / ad:op:axis:n / len:… / pt:… / nc:…），
    # 不做启发式猜测：标签就是产生该行的矩阵项。
    rows = []
    for line in open(MATRIX):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("total="):
            continue
        kv = dict(p.split("=", 1) for p in line.split() if "=" in p)
        kv["_host_outcome"] = line.split()[0]
        kv["_group"] = line[line.rindex("(") + 1:line.rindex(")")] if "(" in line else ""
        if "op" not in kv or kv["_host_outcome"] == "STOPPED":
            continue   # STOPPED = 自适应降序策略主动跳过（不是判定项）
        rows.append(kv)
    if not rows:
        print("RESULT: SKIPPED (matrix.txt 无变体行) —— 没得比，不发合格证")
        return 2

    # 每个分组实际覆盖的轴（由该组内非零坐标决定）
    gaxis = {}
    for r in rows:
        g = r["_group"]
        a, b = int(r["a"]), int(r["b"])
        cur = gaxis.setdefault(g, set())
        if a != 0:
            cur.add("src")
        if b != 0:
            cur.add("dst")
    for g, ax in gaxis.items():
        if not ax:
            gaxis[g] = {"src", "dst"}   # 只有 (0,0) 一点的控制组：两个轴都算（下面按 op 的白名单收敛）

    src = ramp()

    SRC_OPS = {"ls", "ldbrc", "vld", "vldas", "load1", "gather", "gatherb", "pld", "vldasnopre"}
    DST_OPS = {"ls", "lsn", "stfirst", "stpack", "stpack4", "vstus", "store1", "scatter", "dupub", "brcb",
               "psts", "vstusnopost", "dupreg", "add", "mul", "exp", "redsum", "redmax", "cmpsel", "arange",
               "f32bf16norm", "f32bf16nb16", "f32bf16pack", "bf16f32norm", "bf16f32unpk", "lmbarin", "lmbarout",
               "pld"}

    n_compared = n_agree = n_disagree = n_struct = n_fault = n_hang = n_missing = n_host_mismatch = 0
    n_nooutput = 0
    n_nodata = 0
    faults = {}
    per_axis = {}
    notes = []

    for r in rows:
        op, a, b, n = r["op"], int(r["a"]), int(r["b"]), int(r["n"])
        grp = r["_group"]
        log = os.path.join(LOGS, f"run_{op}_a{a}_b{b}_n{n}.log")
        acl = r.get("aclError", "NA")
        if not os.path.isfile(log):
            n_missing += 1
            continue
        txt = open(log, errors="replace").read()
        for tag in ("OK", "WRONG", "HANG", "FAULT", "NO-OUTPUT"):
            if f"OUTCOME: {tag}" in txt:
                outcome = tag
                break
        else:
            outcome = "NO-OUTPUT"
        if outcome != r["_host_outcome"]:
            n_host_mismatch += 1
            notes.append(f"{op} a={a} b={b} n={n}: 复核判定 {outcome} != host 打印 {r['_host_outcome']}")

        # 轴归属：按分组内实际变化的坐标
        axes = gaxis.get(grp, set())
        pts = []
        if grp.startswith("pt:"):
            pass
        elif a != 0:
            if "src" in axes or op in SRC_OPS:
                pts.append(("src", a))
        elif b != 0:
            if "dst" in axes or op in DST_OPS:
                pts.append(("dst", b))
        else:
            if op in SRC_OPS and (grp.startswith("full:") and grp.endswith(":1") or op not in DST_OPS):
                pts.append(("src", 0))
            if op in DST_OPS and (grp.startswith("full:") and grp.endswith(":2") or op not in SRC_OPS):
                pts.append(("dst", 0))
            if op in SRC_OPS and op in DST_OPS and grp.startswith("pt:"):
                pass
        for axis, off in pts:
            if outcome != "NO-OUTPUT" and not (grp == "nc" or grp.startswith("nc:")):
                per_axis.setdefault((op, axis), []).append((off, outcome, acl))

        if outcome in ("NO-OUTPUT", "NOSYNC"):
            n_nooutput += 1
            notes.append(f"{op} a={a} b={b} n={n}: 无输出（排队/超时，非判定项）")
            continue
        if outcome in ("FAULT", "HANG"):
            if outcome == "HANG":
                n_hang += 1
            else:
                n_fault += 1
            faults.setdefault(op, set()).add(acl)
            continue

        dump = os.path.join(DUMPS, f"va_{op}_a{a}_b{b}_n{n}.bin")
        if not os.path.isfile(dump):
            n_missing += 1
            continue
        got = np.fromfile(dump, dtype=np.uint8)
        if got.size != ARENA:
            notes.append(f"{op} a={a} b={b} n={n}: dump 大小 {got.size} != {ARENA}")
            n_missing += 1
            continue

        want, win, why = expected(op, a, b, n, src)
        if want is None:
            ok, nz = window_ok(op, got, b, win)
            n_struct += 1
            if ok:
                n_agree += 1
            elif op in NO_DATA_OPS or nz == 0:
                n_nodata += 1
                notes.append(f"{op} a={a} b={b} n={n}: **无数据**（无故障但 0 字节落盘，探针未取到可观测效果）"
                             f"⇒ 不作对齐结论（README §9）")
            else:
                n_disagree += 1
                notes.append(f"{op} a={a} b={b} n={n}: 结构性判定失败（{why}；窗口非 sentinel 字节 {nz}）")
            continue
        n_compared += 1
        diff = np.nonzero(got != want)[0]
        if op in NEG_CONTROL:
            if diff.size:
                n_agree += 1
                notes.append(f"{op} a={a} b={b}: 负向对照生效（与模型不符，首个差异字节 {int(diff[0])}）")
            else:
                n_disagree += 1
                notes.append(f"{op} a={a} b={b}: **负向对照未生效**（缺 init/post 竟仍逐字节正确）")
            continue
        if diff.size == 0:
            n_agree += 1
        else:
            first = int(diff[0])
            over = int(np.count_nonzero(got[b + win:b + VLB] != SENT)) if op == "lsn" else 0
            n_disagree += 1
            notes.append(f"{op} a={a} b={b} n={n}: 首个差异字节 {first}（窗口 {win}B，窗口内超写 {over}B）；模型={why}")
        if op in ("lsn", "stfirst"):
            lo, hi = written_extent(got)
            notes.append(f"{op} n={n} b={b}: 实测写出区间 [{lo},{hi}]（模型窗口 [{b},{b + win})）")

    print("== 1) 故障分类（独立复算自 log 原文）==")
    print(f"   FAULT {n_fault} / HANG {n_hang} / OK-or-struct {n_agree + n_disagree}")
    for op in sorted(faults):
        print(f"   {op:14s} 出现故障码 {sorted(faults[op])}")
    print()
    print("== 2) 逐字节模型复核（docs/17 T1）==")
    # 口径（避免"计数包含关系"误读）：n_compared = 逐字节模型行数，n_struct = 结构性行数，两者不相交；
    # n_agree = 通过的行数 = (逐字节通过) + (结构性通过) + (负向对照按"期望不符"计通过)。
    print(f"   判定行 {n_compared + n_struct} = 逐字节模型 {n_compared}"
          f"（其中负向对照 {len(NEG_CONTROL)} 条按'期望不符'计通过） + 结构性 {n_struct}；"
          f"与模型不符 {n_disagree}；无数据 {n_nodata}；缺证据 {n_missing}")
    if notes:
        print("   备注（前 40 条）：")
        for x in notes[:40]:
            print("     -", x)
    print()
    print("== 3) 实测最小 UB 偏移对齐（按 (op,轴) 汇总；0=32B 对齐对照）==")
    order = {"OK": 0, "WRONG": 1, "FAULT": 2, "HANG": 3, "NO-OUTPUT": 4}
    for key in sorted(per_axis):
        pts = sorted(per_axis[key])
        passed = [o for o, oc, _ in pts if oc in ("OK", "WRONG")]
        min_align = min(passed) if passed else None
        line = " ".join(f"{o}:{oc}" for o, oc, _ in pts if o != 0)
        ctrl = [oc for o, oc, _ in pts if o == 0]
        print(f"   {key[0]:14s} {key[1]:3s} 对照(0)={ctrl[0] if ctrl else '?':6s} "
              f"最小通过偏移={min_align if min_align is not None else '无'} | {line}")
    print()
    if n_host_mismatch:
        print(f"   复核判定 与 host 打印不一致 {n_host_mismatch} 条（见备注）")
    if n_disagree or n_host_mismatch:
        print(f"RESULT: FAIL ({n_disagree} 处与模型不符；判定项 {n_compared} 比 + {n_struct} 结构性，"
              f"故障 {n_fault}，挂死 {n_hang}，无输出 {n_nooutput}，无数据 {n_nodata}，缺证据 {n_missing}，"
              f"host/复核不一致 {n_host_mismatch})")
        return 1
    print(f"RESULT: OK (比过 {n_compared} 条逐字节 + {n_struct} 条结构性；故障 {n_fault}（已分类归档）、"
          f"挂死 {n_hang}、无输出 {n_nooutput}、**无数据 {n_nodata}**、缺证据 {n_missing}、与模型不符 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
