#!/usr/bin/env python3
"""
check_ref_ildl.py —— `probe_v_ildl` 的**独立复核 + lane 映射解码**（M35 交接悬案的判定件）

做法：dump 里的每个 lane 值都能**唯一反推**它的来源 (源寄存器, 源 lane)——因为 host 侧填的
      s0/s1 是互不重叠的唯一值（f32: 0.5..127.5；b16: 0..255）。于是：
        1. **解码**：逐 lane 打印/判定真实映射（这是我实测出来的，不是文档转述）；
        2. **判定**：把解码结果与三种候选映射（Interleave 式 / DeInterleave 式 / 直接拼接式）比对；
        3. **正例/反例**：`inv_f32` 必须还原 (s0,s1)（正例）；`intlv_f32_neg` 必须**不**还原（反例）。

三态退出码（docs/17 §8.3）：0=比过通过 / 1=比过有差异 / 2=没得比（dump 缺失）

用法：python3 check_ref_ildl.py [probe_v_align 目录]
"""
import os
import sys

import numpy as np

PROBE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
LOGS = os.path.join(PROBE, "evidence", "logs")
DUMPS = os.path.join(PROBE, "evidence", "dumps")
ARENA = 1024
SENT = 0xA5
VLB = 256

CASES = ["intlv_f32", "dintlv_f32", "inv_f32", "intlv_f32_neg", "same_f32",
         "intlv_b16", "dintlv_b16", "intlv_misalign"]
B16 = {"intlv_b16", "dintlv_b16"}


def src_arrays(is_b16):
    """与 kernel host 侧同式的输入（独立实现）"""
    if is_b16:
        h = np.arange(ARENA // 2, dtype=np.uint16)
        return h[:128].astype(np.uint16), h[128:256].astype(np.uint16)
    f = (np.arange(ARENA // 4, dtype=np.float32) + np.float32(0.5))
    return f[:64], f[64:128]


def decode(vals, s0, s1):
    """把一串输出 lane 值解码成 [(src, lane)]；解不出记 None"""
    out = []
    for i, v in enumerate(vals):
        m0 = np.nonzero(s0 == v)[0]
        m1 = np.nonzero(s1 == v)[0]
        if m0.size == 1:
            out.append((0, int(m0[0])))
        elif m1.size == 1:
            out.append((1, int(m1[0])))
        else:
            out.append(None)
    return out


def canon_intlv(V):
    """Interleave 假设：d0 = 两源前 V/2 逐元素交替；d1 = 后 V/2 逐元素交替"""
    h = V // 2
    d0 = [x for k in range(h) for x in ((0, k), (1, k))]
    d1 = [x for k in range(h) for x in ((0, h + k), (1, h + k))]
    return d0, d1


def canon_dintlv(V):
    """DeInterleave 假设：d0 = [s0 偶 lane, s1 偶 lane]；d1 = [s0 奇 lane, s1 奇 lane]"""
    h = V // 2
    d0 = [(0, 2 * k) for k in range(h)] + [(1, 2 * k) for k in range(h)]
    d1 = [(0, 2 * k + 1) for k in range(h)] + [(1, 2 * k + 1) for k in range(h)]
    return d0, d1


def canon_same(V):
    """源寄存器同一个（src0 == src1 == s0）时的 DeInterleave：偶/奇拆分各占两个半区"""
    h = V // 2
    d0 = [(0, 2 * k) for k in range(h)] * 2
    d1 = [(0, 2 * k + 1) for k in range(h)] * 2
    return d0, d1


def canon_ident(V):
    return [(0, k) for k in range(V)], [(1, k) for k in range(V)]


CANDIDATES = [
    ("INTLV", canon_intlv),
    ("DEINTLV", canon_dintlv),
    ("SAME(=src0==src1)", canon_same),
    ("IDENTITY(d0=s0,d1=s1)", canon_ident),
]


def classify(m0, m1, V):
    """返回命中的候选映射名；都不命中的返回 None"""
    for name, fn in CANDIDATES:
        d0, d1 = fn(V)
        if m0 == d0 and m1 == d1:
            return name
    return None


def partial_name(m, V):
    """未命中完整候选时，给出可读的部分刻画"""
    h = V // 2
    if all(x is not None for x in m):
        if all(m[k] == (0, 2 * k) for k in range(h)):
            return "s0 偶 lane 起"
        if all(m[k] == (0, 2 * k + 1) for k in range(h)):
            return "s0 奇 lane 起"
        if all(m[k] == (0, k) for k in range(h)):
            return "s0 原序起"
    return "其它"


def fmt_map(m):
    return " ".join(("-" if x is None else f"s{x[0]}[{x[1]}]") for x in m)


def main():
    if not os.path.isdir(DUMPS):
        print(f"RESULT: SKIPPED (no {DUMPS}) —— 没得比，不发合格证")
        return 2
    n_ok = n_fail = n_skip = 0
    rows = []
    for case in CASES:
        dump = os.path.join(DUMPS, f"il_{case}.bin")
        log = os.path.join(LOGS, f"ildl_{case}.log")
        if not os.path.isfile(log):
            print(f"[{case}] 缺日志 {log}")
            n_skip += 1
            continue
        txt = open(log, errors="replace").read()
        if "OUTCOME: OK" not in txt and "OUTCOME: FAULT" not in txt:
            # 运行失败（含 507035）：只有 intlv_misalign 允许/预期故障
            acl = "?"
            for line in txt.splitlines():
                if line.startswith("LAUNCH aclError="):
                    acl = line.split("=")[1].split()[0]
            if case == "intlv_misalign":
                print(f"[{case}] 预期故障：aclError={acl} ⇒ 对齐要求属于相邻 store，Interleave 本体无 UB 地址操作数 ✔")
                rows.append((case, "FAULT", f"aclError={acl}"))
                n_ok += 1
            else:
                print(f"[{case}] 非预期故障 aclError={acl}")
                n_fail += 1
            continue
        if not os.path.isfile(dump):
            print(f"[{case}] 缺 dump")
            n_skip += 1
            continue

        is_b16 = case in B16
        got = np.fromfile(dump, dtype=np.uint8)
        if is_b16:
            allv = got.view("<u2")          # 512 个 16-bit lane
        else:
            allv = got.view("<f4")          # 256 个 fp32 lane
        d0 = allv[0:128 if is_b16 else 64]
        d1 = allv[128:256] if is_b16 else allv[64:128]
        s0, s1 = src_arrays(is_b16)
        half = 64 if is_b16 else 32

        m0, m1 = decode(d0, s0, s1), decode(d1, s0, s1)
        V = 128 if is_b16 else 64
        name = classify(m0, m1, V)
        print(f"[{case}] d0/d1 命中候选映射: {name if name else partial_name(m0, V) + ' / ' + partial_name(m1, V)}")
        print(f"[{case}] d0 逐 lane 来源: {fmt_map(m0[:16])} …（前 16 lane）")
        print(f"[{case}] d1 逐 lane 来源: {fmt_map(m1[:16])} …（前 16 lane）")
        rows.append((case, name or "（无候选命中）", ""))

        want = {"intlv_f32": "INTLV", "dintlv_f32": "DEINTLV", "inv_f32": "IDENTITY(d0=s0,d1=s1)",
                "same_f32": "SAME(=src0==src1)", "intlv_b16": "INTLV", "dintlv_b16": "DEINTLV"}
        if case == "intlv_f32_neg":
            # **反例**：二次 Interleave 必须**不**等于 (s0,s1)（否则"配对/逆"的判定就是空的）
            ok = (name is None)
        else:
            ok = (name == want.get(case))
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            print(f"[{case}] ^^ 判定不通过（期望 {want.get(case) if case != 'intlv_f32_neg' else '无候选命中'}）")

    print()
    print("== 逐用例映射汇总 ==")
    for r in rows:
        print(f"   {r[0]:16s} {r[1]}")
    print()
    if n_skip and n_ok == 0 and n_fail == 0:
        print(f"RESULT: SKIPPED ({n_skip} 用例缺 dump/日志) —— 没得比，不发合格证")
        return 2
    if n_fail:
        print(f"RESULT: FAIL (判定项 {n_ok + n_fail}：通过 {n_ok} / 不通过 {n_fail}；缺证据 {n_skip})")
        return 1
    print(f"RESULT: OK (判定项 {n_ok + n_fail}：通过 {n_ok} / 不通过 0；缺证据 {n_skip})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
