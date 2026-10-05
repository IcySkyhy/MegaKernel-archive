#!/usr/bin/env python3
"""probe_vf_ldst/tools/summarize.py —— 逐进程日志汇总 + 覆盖自校验 + 三态结论。

退出码（tower 规则：'我没比' / '我比过但有差异' / '我比过且通过' 必须能区分）：
    0 = 比过、**声明集全部到齐**、每个变体进程数达标、跨进程判读一致、对照自检符合预期
    1 = 比过、但缺组 / 进程数不足 / 跨进程判读不一致 / 对照自检不符
    2 = **没得比**（没有任何 run_*.log，或**没有声明清单** —— 不给合格证）

覆盖自校验（tower 规则「计数与匹配器同源 / 三态分栏 / 已知会被漏掉」）：
  * 声明集 `logs/declared_groups.txt` 由 `run_probes.sh` 在**跑变体的同一组循环里**逐条 echo 出来
    ⇒ 「声明的组」与「实际跑的组」同源，不会各写一份而漂移；
  * 本脚本把**声明集**、**实到集**、**逐组进程数**三栏分开打印，不做成一个"0 问题"；
  * 缺组、多出未声明的组、进程数不足，都进 DIFF（exit 1）。

判定项
    每个（变体 × 配置）跨**独立进程**的判读必须一致、等于该变体的设计预期、且进程数 == PROCS。
报告项
    判定项 dump（`a_*_mask*_p*.bin`、`b_*_p*_out.bin`、`c_*_p*_out.bin`）跨进程 sha256 是否一致。
    现场快照 `b_*_p*_ub.bin` **不做**一致性要求 —— 其中 `UB_MID`/`UB_BACKLOG` 两块 scratch
    在内核未写它们的变体里是 UB 残留（见 README §5 与 §8）。

对照自检（PASS 不是空洞的）：任一不符即 DIFF（exit 1），因为那会让整个矩阵的 PASS 不可信：
    probe_b_noprod           生产者关闭       ⇒ 必须 64/64 poison
    probe_b_prod3_selftest   生产者故意写旧值 ⇒ 判读器必须报 stale_prev
    probe_c_rt_*             GM 回环排序三态
    probe_a_*_nc sel=0/1     非连续掩码 M4/M3 ⇒ NORM 必须写出周期集合、FIRST_ELEMENT 必须仍只写 lane 0

用法：python3 tools/summarize.py <evidence_dir> <procs>
"""

import hashlib
import os
import re
import sys
from collections import OrderedDict

EV = sys.argv[1] if len(sys.argv) > 1 else "evidence"
PROCS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LOGS = os.path.join(EV, "logs")
DUMPS = os.path.join(EV, "dumps")
DECLARED = os.path.join(LOGS, "declared_groups.txt")

A_RE = re.compile(r"^run_(probe_a_[a-z0-9_]+)_mask(\d+)_p(\d+)\.log$")
B_RE = re.compile(r"^run_(probe_b_[a-z0-9_]+)_p(\d+)\.log$")
C_RE = re.compile(r"^run_(probe_c_[a-z0-9_]+)_p(\d+)\.log$")

# 设计上就该 FAIL 的变体（阳性/负对照，不是缺陷）：
#   prod3_selftest   判读器自检：生产者故意写旧值
#   c_rt_unordered   回环无任何排序
#   c_rt_twotok      两个不同 BufferID（MTE3 drain 释放无人取）——**复刻 M36 已确认的事实形态**
EXPECT_FAIL = {"probe_b_prod3_selftest", "probe_c_rt_unordered", "probe_c_rt_twotok"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_log(path):
    with open(path, "r", errors="replace") as f:
        return f.read()


def verdict_of(text):
    if "RUN PASS" in text:
        return "PASS"
    if "RUN FAIL" in text:
        return "FAIL"
    return "ERROR"


def collect():
    """实到集：由目录里真正存在的 run_*.log 推出（与匹配器同一份正则）。"""
    a, b, c = OrderedDict(), OrderedDict(), OrderedDict()
    if not os.path.isdir(LOGS):
        return a, b, c
    for name in sorted(os.listdir(LOGS)):
        m = A_RE.match(name)
        if m:
            a.setdefault("%s mask=%s" % (m.group(1), m.group(2)), []).append(
                (int(m.group(3)), os.path.join(LOGS, name)))
            continue
        m = B_RE.match(name)
        if m:
            b.setdefault(m.group(1), []).append((int(m.group(2)), os.path.join(LOGS, name)))
            continue
        m = C_RE.match(name)
        if m:
            c.setdefault(m.group(1), []).append((int(m.group(2)), os.path.join(LOGS, name)))
    return a, b, c


def groups(a, b, c):
    out = []
    for name, runs in a.items():
        out.append((name, "PASS", runs, "a_" + name.split(" mask=")[0] + "_mask" +
                    name.split(" mask=")[1]))
    for tgt, runs in b.items():
        out.append((tgt, "FAIL" if tgt in EXPECT_FAIL else "PASS", runs, tgt))
    for tgt, runs in c.items():
        out.append((tgt, "FAIL" if tgt in EXPECT_FAIL else "PASS", runs, tgt))
    return out


def cls_of(text):
    m = re.search(r"chunk: ok=(\d+) stale_prev=(\d+) ahead=(\d+) poison=(\d+) zero=(\d+) other=(\d+)",
                  text)
    if not m:
        return None
    return dict(zip(("ok", "stale_prev", "ahead", "poison", "zero", "other"),
                    (int(x) for x in m.groups())))


def dump_sha(kind, tag, p, suffix):
    fp = os.path.join(DUMPS, "%s_%s_p%d_%s" % (kind, tag, p, suffix))
    return sha256(fp) if os.path.exists(fp) else None


def control_check(gs, gmap):
    """对照自检：返回 (通过数, 总数, [说明])"""
    def one(name):
        return next((g for g in gs if g[0] == name), None)

    checks = [
        ("probe_b_noprod", lambda v, t, g: v == "PASS" and (cls_of(t) or {}).get("poison") == 64,
         "生产者关闭 ⇒ 64/64 poison"),
        ("probe_b_prod3_selftest",
         lambda v, t, g: v == "FAIL" and (cls_of(t) or {}).get("stale_prev", 0) >= 60,
         "生产者写旧值 ⇒ 判读器报 stale_prev"),
        ("probe_c_rt_unordered",
         lambda v, t, g: v == "FAIL" and (cls_of(t) or {}).get("poison", 0) >= 1,
         "GM 回环无排序 ⇒ FAIL 且读到尚未写回的值"),
        ("probe_c_rt_event", lambda v, t, g: v == "PASS", "GM 回环 + SetFlag/WaitFlag ⇒ PASS"),
        ("probe_c_rt_bufid", lambda v, t, g: v == "PASS", "GM 回环 + 同一 token drain 交接 ⇒ PASS"),
        ("probe_c_rt_twotok",
         lambda v, t, g: v == "FAIL" and (cls_of(t) or {}).get("poison", 0) >= 1,
         "两个不同 BufferID（drain 释放无人取）= M36 事实形态 ⇒ FAIL"),
        ("probe_c_rt_small_bufid", lambda v, t, g: v == "PASS",
         "逐 chunk 小回写 + 有排序 ⇒ PASS"),
        ("probe_a_norm_b32_nc mask=0",
         lambda v, t, g: v == "PASS" and "写出 16 lane" in t,
         "非连续掩码 M4（周期 4）⇒ NORM 只写 16 个 lane（非前缀）"),
        ("probe_a_norm_b32_nc mask=1",
         lambda v, t, g: v == "PASS" and "写出 22 lane" in t,
         "非连续掩码 M3（周期 3）⇒ NORM 只写 22 个 lane（非前缀）"),
        ("probe_a_norm_b32_nc mask=2",
         lambda v, t, g: v == "PASS" and "交替掩码被严格遵守" in t,
         "交替掩码（MaskGenWithRegTensor）⇒ NORM 只写同奇偶的 32 lane"),
        ("probe_a_norm_b32_nc mask=3",
         lambda v, t, g: v == "PASS" and "交替掩码被严格遵守" in t,
         "交替掩码反相 ⇒ NORM 只写同奇偶的 32 lane"),
        ("probe_a_first_elem_nc mask=0",
         lambda v, t, g: v == "PASS" and "写出 1 lane" in t,
         "FIRST_ELEMENT 遇非连续掩码 M4 ⇒ 仍只写 1 lane"),
        ("probe_a_first_elem_nc mask=1",
         lambda v, t, g: v == "PASS" and "写出 1 lane" in t,
         "FIRST_ELEMENT 遇非连续掩码 M3 ⇒ 仍只写 1 lane"),
        ("probe_a_first_elem_nc mask=2",
         lambda v, t, g: v == "PASS" and "写出 1 lane" in t,
         "FIRST_ELEMENT 遇交替掩码 ⇒ 仍只写 1 lane"),
        ("probe_a_first_elem_nc mask=3",
         lambda v, t, g: v == "PASS" and "写出 1 lane" in t,
         "FIRST_ELEMENT 遇交替掩码反相 ⇒ 仍只写 1 lane"),
    ]
    ok, notes = 0, []
    for name, pred, desc in checks:
        g = one(name)
        if g is None:
            notes.append("MISSING  %-30s %s" % (name, desc))
            continue
        runs = sorted(g[2])
        t = read_log(runs[0][1])
        v = verdict_of(t)
        if pred(v, t, g):
            ok += 1
            notes.append("OK       %-30s [%s] %s" % (name, v, desc))
        else:
            notes.append("BAD      %-30s [%s] %s" % (name, v, desc))
    return ok, len(checks), notes


def main():
    a, b, c = collect()
    gs = groups(a, b, c)
    arrived = set(g[0] for g in gs)

    print("=" * 92)
    print("probe_vf_ldst 跨进程汇总（每变体应有 %d 个独立进程）" % PROCS)
    print("=" * 92)

    if not os.path.exists(DECLARED):
        print("\nRESULT: SKIPPED（缺 logs/declared_groups.txt —— 没有声明清单就没有比对的基准，"
              "不给合格证；请用 run_probes.sh 生成证据）")
        return 2
    with open(DECLARED, "r", errors="replace") as f:
        declared = [ln.strip() for ln in f.read().splitlines()
                    if ln.strip() and not ln.startswith("#")]
    if not arrived:
        print("\nRESULT: SKIPPED（evidence/logs 下没有任何 run_*.log —— 没有可比较的输入，不给合格证）")
        return 2

    # ---- 覆盖三栏（声明 / 实到 / 达标），计数与匹配器同源 ----
    missing = [g for g in declared if g not in arrived]
    unexpected = sorted(arrived - set(declared))
    print("\n## 覆盖（三态分栏）")
    print("   声明集(declared_groups.txt) : %d 组" % len(declared))
    print("   实到集(存在 run_*.log)      : %d 组" % len(arrived))
    print("   缺组(声明了但没跑)          : %d 组 %s" % (len(missing), missing[:8] if missing else ""))
    print("   多出(跑了但没声明)          : %d 组 %s" % (len(unexpected), unexpected[:8] if unexpected else ""))
    n_undersupplied = sum(1 for g in gs if len(g[2]) != PROCS)
    print("   进程数不足的组              : %d 组" % n_undersupplied)

    def table(title, rows, sha_kind, extra_idx=None):
        """判定项 dump 的跨进程 sha256 —— 探针 A 用落盘快照，B/C 用 `_out.bin`（不含 scratch）。"""
        if title:
            print("\n## %s" % title)
        print("   %-32s %-8s %-9s %-10s %-9s %s" %
              ("组", "判读", "进程", "判定项 dump sha", "期待", "读数"))
        for name, expect, runs, tag in rows:
            runs = sorted(runs)
            vs = [verdict_of(read_log(p)) for _, p in runs]
            uniq = sorted(set(vs))
            shas = set()
            for k, _ in runs:
                fp = (os.path.join(DUMPS, "%s_p%d.bin" % (tag, k)) if sha_kind == "a"
                      else os.path.join(DUMPS, "%s_%s_p%d_out.bin" % (sha_kind, tag, k)))
                if os.path.exists(fp):
                    shas.add(sha256(fp))
            cl = cls_of(read_log(runs[0][1])) or {}
            if not cl:
                m = re.search(r"写出 (\d+) lane（期望 \d+）、未触碰 (\d+) lane", read_log(runs[0][1]))
                if m:
                    cl = {"written": m.group(1), "untouched": m.group(2)}
                else:
                    # 真实日志句形如「…交替掩码被严格遵守（32 个 lane、全偶；其余全为 POISON）」，
                    # 所以这里**不能**在 `全.` 后面要求紧跟 `）`
                    # （否则本组在汇总表的读数列会退化成 `-`，即匹配器看不见却仍算通过）
                    m = re.search(r"交替掩码被严格遵守（(\d+) 个 lane、全([偶奇])", read_log(runs[0][1]))
                    cl = {"written": m.group(1), "parity": m.group(2)} if m else {}
            print("   %-32s %-8s %-9s %-10s %-9s %s" %
                  (name.replace("probe_a_", "").replace("probe_b_", "").replace("probe_c_", ""),
                   uniq[0] if len(uniq) == 1 else "不一致!",
                   "%d/%d" % (len(runs), PROCS),
                   ("一致" if len(shas) == 1 else ("不一致! %d 种" % len(shas))) if shas else "无 dump",
                   expect, " ".join("%s=%s" % (k, v) for k, v in cl.items()) or "-"))

    print("\n## 探针 A —— StoreDist 掩码语义（含非连续/交替掩码族）")
    table("", [g for g in gs if g[0].startswith("probe_a_")], "a")
    print("\n## 探针 B —— MTE2→V 交接可见性")
    print("   分类：ok / stale_prev(上一 chunk) / ahead(下一 chunk) / poison(未被写) / zero(读到 0) / other")
    table("", [g for g in gs if g[0].startswith("probe_b_")], "b")
    print("\n## 探针 C —— 同核 GM 回环（MTE3: UB→GM 紧接 MTE2: GM→UB）")
    table("", [g for g in gs if g[0].startswith("probe_c_")], "c")

    # ---- 判定项 ----
    n_cmp, n_consistent, n_expected, bad = 0, 0, 0, []
    for name, expect, runs, tag in gs:
        n_cmp += 1
        if len(runs) != PROCS:
            bad.append("%s：只有 %d/%d 个独立进程" % (name, len(runs), PROCS))
            continue
        vs = [verdict_of(read_log(p)) for _, p in runs]
        if len(set(vs)) != 1:
            bad.append("%s：跨进程判读不一致 %s" % (name, sorted(set(vs))))
            continue
        n_consistent += 1
        if vs[0] != expect:
            bad.append("%s：判读 %s，设计预期 %s" % (name, vs[0], expect))
        else:
            n_expected += 1

    cok, ctot, cnotes = control_check(gs, None)
    print("\n## 对照自检（PASS 不是空洞的）")
    for n in cnotes:
        print("   " + n)
    print("   对照自检: %d/%d" % (cok, ctot))

    minruns = min((len(g[2]) for g in gs), default=0)
    if missing:
        bad.append("缺组 %d 个（声明了但没跑）：%s" % (len(missing), missing[:8]))
    if unexpected:
        bad.append("多出 %d 个未声明的组：%s" % (len(unexpected), unexpected[:8]))

    print("\n## 结论口径")
    print("   * 判定项 = 声明集全到齐 + 每组 %d 个独立进程 + 跨进程判读一致 + 等于设计预期 + 对照自检。" % PROCS)
    print("   * 报告项 = 判定项 dump 跨进程 sha256（`a_*_p*.bin` / `*_out.bin`）；")
    print("     `b_*_ub.bin` 中的 `UB_MID`/`UB_BACKLOG` scratch 是 UB 残留，设计上不做一致性要求。")

    if bad or cok != ctot:
        for x in bad:
            print("   DIFF  " + x)
        print("\nRESULT: DIFF（比过 %d/%d 个组，其中 %d 个跨进程一致且符合预期、实测最小进程数 %d/%d、"
              "对照自检 %d/%d）" % (n_expected, len(declared), n_consistent, minruns, PROCS, cok, ctot))
        return 1
    print("\nRESULT: OK（声明 %d 组全到齐；%d/%d 组跨进程判读一致且符合设计预期；实测最小进程数 %d/%d；"
          "对照自检 %d/%d）" % (len(declared), n_expected, n_cmp, minruns, PROCS, cok, ctot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
