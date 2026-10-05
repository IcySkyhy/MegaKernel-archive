#!/usr/bin/env python3
"""analyze_snap.py —— M64 探针证据解析（设备侧快照逐词归因 + 时间戳交叉验证）

用法：
  python3 analyze_snap.py snap <snap_xxx_run1.bin>            # 快照逐词归因
  python3 analyze_snap.py trace <log_with_TRACE_lines>        # 时间戳交叉验证（覆盖窗口证明）
  python3 analyze_snap.py matrix <logdir|logfile> [...]       # 汇总 SUMMARY 行 -> 配方 × 收敛次数/9 表

设计要点（避免自证）：本脚本只做**机械归因**，每个数字都可从原始 4B 词复算：
  - 快照里每个词都能判出「它属于谁、是第几轮」的 ticket：TICKET(T,i) = MAGIC16<<16 | T<<8 | i
  - 归因分类：match(本轮) / future(T' > 本轮 ⇒ 被"下一轮/更后的写"覆盖) / past(T' < 本轮 ⇒ 读到陈旧值)
              / zero(从未被写) / other(既非 ticket 也非 0)
  - future 类非零 = **读窗口未闭合**的直接证据（读的时候别人已经把后面的轮次写进去了）
"""
import collections
import glob
import os
import re
import struct
import sys

MAGIC16 = 0x5153


def ticket(T, rank):
    return (MAGIC16 << 16) | ((T & 0xFF) << 8) | (rank & 0xFF)


def classify(word, T_cur, rank_of_slot):
    """返回 (kind, T_obs)。kind ∈ match/future/past/zero/other"""
    if word == 0:
        return "zero", 0
    if (word >> 16) & 0xFFFF != MAGIC16:
        return "other", 0
    if (word & 0xFF) != rank_of_slot:
        return "other", 0
    T_obs = (word >> 8) & 0xFF
    if T_obs == T_cur:
        return "match", T_obs
    return ("future" if T_obs > T_cur else "past"), T_obs


def cmd_snap(path):
    with open(path, "rb") as f:
        hdr = struct.unpack("<8i", f.read(32))
        slmax, rndmax, naiv, rounds, st, wl, words, magic = hdr
        assert magic == 0x5A5A5A5A, "bad snapshot magic"
        raw = f.read()
    data = struct.unpack("<%di" % (len(raw) // 4), raw)
    print(f"# snapshot {os.path.basename(path)}: naiv={naiv} rounds={rounds} st={st} wl={wl} words={words}")
    seqwords = wl // 4
    # 每 (round, core) 的 head/词级归因
    print("# ---- 逐轮：head 命中分布 + 词级归因（对所有核求和）----")
    print("round  cores_headOK=naiv  headOK_min  word_match  word_future  word_past  word_zero  word_other  "
          "future_rounds(前3)")
    all_future_examples = []
    for r in range(rounds):
        T_cur = r + 1
        cnt = collections.Counter()
        fr = collections.Counter()
        headok = []
        for c in range(naiv):
            base = (c * rndmax + r) * words
            reg = data[base:base + words]
            h = sum(1 for i in range(naiv) if reg[i * st] == ticket(T_cur, i))
            headok.append(h)
            for i in range(naiv):
                for j in range(seqwords):
                    k, T_obs = classify(reg[i * st + j], T_cur, i)
                    cnt[k] += 1
                    if k == "future":
                        fr[T_obs] += 1
                        if len(all_future_examples) < 12:
                            all_future_examples.append((r, c, i, T_obs))
        top = ",".join(f"T{T}:{n}" for T, n in fr.most_common(3)) or "-"
        print(f"{r:5d}  {sum(1 for h in headok if h == naiv):5d}/{naiv:<9}  {min(headok):9d}  "
              f"{cnt['match']:10d}  {cnt['future']:11d}  {cnt['past']:9d}  {cnt['zero']:9d}  {cnt['other']:10d}  {top}")
    print()
    print("# ---- 被污染读的样例 (round, core, 槽, 观测到的轮次) ----")
    for e in all_future_examples:
        print(f"  round={e[0]} core={e[1]} slot={e[2]} 观测到 T={e[3]}（本轮应为 T={e[0]+1}）")
    print()
    print("# ---- 逐 (round, core) headOK 分布直方 ----")
    for r in range(rounds):
        T_cur = r + 1
        hist = collections.Counter()
        for c in range(naiv):
            base = (c * rndmax + r) * words
            reg = data[base:base + words]
            h = sum(1 for i in range(naiv) if reg[i * st] == ticket(T_cur, i))
            hist[h] += 1
        print(f"  round={r:2d} " + " ".join(f"{k}:{v}" for k, v in sorted(hist.items())))


def cmd_trace(path):
    """从 TRACE 行验证「读窗口未闭合」：违规核的 tRd(r) 是否晚于快核的 tWr(r+1)"""
    rows = collections.defaultdict(dict)   # (core, round) -> dict
    with open(path) as f:
        for line in f:
            if not line.startswith("TRACE "):
                continue
            kv = dict(kv.split("=") for kv in line.split()[1:])
            rows[(int(kv["core"]), int(kv["round"]))] = {
                "T": int(kv["T"]), "headOK": int(kv["headOK"]), "tWr": int(kv["tWr"], ),
                "tRel": int(kv["tRel"]), "tRd": int(kv["tRd"]), "naiv": None,
            }
    if not rows:
        print("no TRACE lines")
        return
    cores = sorted({c for c, _ in rows})
    rounds = sorted({r for _, r in rows})

    def d32(a, b):
        return (a - b) & 0xFFFFFFFF

    print("# ---- 逐轮：违规核 vs 快核的时序（设备侧 SYS_CNT，模 2^32 差）----")
    print("round  slowest_core(tRd-T)  its_headOK   earliest_next_write(tWr(r+1)-T)  slow_read_after_fast_write")
    naiv = max(len(cores), 56)
    for r in rounds:
        base = min(rows[(c, r)]["tRd"] for c in cores if (c, r) in rows)
        slow = max((c for c in cores if (c, r) in rows), key=lambda c: d32(rows[(c, r)]["tRd"], base))
        slow_tRd = d32(rows[(slow, r)]["tRd"], base)
        if r + 1 in rounds:
            nxt = min((c for c in cores if (c, r + 1) in rows),
                      key=lambda c: d32(rows[(c, r + 1)]["tWr"], base))
            fast_wr = d32(rows[(nxt, r + 1)]["tWr"], base)
            after = "YES" if slow_tRd > fast_wr else "no"
        else:
            fast_wr, after = -1, "n/a(末轮)"
        print(f"{r:5d}  core{slow:02d} {slow_tRd:14d}  {rows[(slow, r)]['headOK']:10d}  {fast_wr:27d}  {after}")


def cmd_matrix(paths):
    """汇总 SUMMARY 行：按变体聚合 9 次运行的 rc / viol / rounds_clean"""
    lines = []
    for p in paths:
        if os.path.isdir(p):
            files = sorted(f for f in glob.glob(os.path.join(p, "*.log"))
                           if not os.path.basename(f).startswith('session'))
        else:
            files = [p]
        for fp in files:
            with open(fp, errors="replace") as f:
                for line in f:
                    if line.startswith("SUMMARY "):
                        lines.append(line.strip())
    agg = collections.defaultdict(lambda: {"n": 0, "rc0": 0, "viol": [], "rc": [], "cores_clean": []})
    for line in lines:
        kv = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        a = agg[kv["variant"]]
        a["n"] += 1
        rcm = re.match(r"-?\d+", kv.get("rc", "2"))
        rc = int(rcm.group()) if rcm else 2
        a["rc"].append(rc)
        if rc == 0:
            a["rc0"] += 1
        vm = re.match(r"-?\d+", kv.get("viol", "-1"))
        a["viol"].append(int(vm.group()) if vm else -1)
        a["cores_clean"].append(kv.get("cores_clean", "?"))
    print(f"{'variant':26s} {'runs':>4s} {'rc==0':>5s} {'viol min..max':>16s}  cores_clean(逐次)")
    for name in sorted(agg):
        a = agg[name]
        print(f"{name:26s} {a['n']:4d} {a['rc0']:5d} {min(a['viol']):7d}..{max(a['viol']):<7d}  "
              f"{' '.join(a['cores_clean'])}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "snap":
        cmd_snap(sys.argv[2])
    elif cmd == "trace":
        cmd_trace(sys.argv[2])
    elif cmd == "matrix":
        cmd_matrix(sys.argv[2:])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
