#!/usr/bin/env python3
"""probe_vf_loop/tools/summarize.py —— 从 evidence/logs 的原始日志复算矩阵表 / 核验一致性 / 重建索引。

用法（本目录下）：
  python3 tools/summarize.py                      # 打印 nest 矩阵 + gather 扫描表（回放 rep1）
  python3 tools/summarize.py matrix               # 同上
  python3 tools/summarize.py verify               # **核验**：跨 rep 一致性 + 判据列独立复算 + 覆盖范围自述
  python3 tools/summarize.py verify --logs DIR     # 对别的日志目录核验（selftest 用）
  python3 tools/summarize.py reindex              # 从原始日志**重建** run_matrix.txt 与 sha256.txt
  python3 tools/summarize.py reindex --logs DIR    # 对别的目录重建（默认写回同一目录）
  python3 tools/summarize.py selftest             # 负向对照：在 /tmp 副本里改字节，验证核验能报差异
  python3 tools/summarize.py nest|ag              # 只要其中一张表
  python3 tools/summarize.py det <tag>            # 打印某 target 的 rep1 全文

退出码（verify / selftest 用；矩阵/回放模式恒 0）：
  0 = 比过，全部通过（判据列跨 rep 一致 + 独立复算无差异）
  1 = 比过，**发现差异**（判据列跨 rep 不一致，或独立复算与日志不符）
  2 = **没得比**（日志缺失 / 可比对象不足 / 输入目录不存在）——与 1 区分，避免"没跑"被当成"通过"

口径声明（重要）：
  * **判据列**（judgement）= `cnt` / `expect` / `tri` / `MISMATCH` / `JSET-ODD` / `lane_mismatch` /
    `tri_odd` / `rows=N verdict=` 这些行；它们决定本探针的结论。**跨 rep 必须逐字符一致**。
  * **数据面列**（data-face）= `acc_maxdiff ...` / `hash=0x...` 这些行；在**读越界 UB 的病态变体**里
    天然非确定（读到的是 UB 里上一次 launch 的残留），**不作为确定性判据**，只在报告里单列。
  * 本工具**只回放/核验已有日志**，不做真机判据；真机读数的来源是 `run_*.log` 本身。
  * 已知**会漏掉**的东西（负向对照见 `selftest`）：判据列相同但数据面列不同时，本工具只会在
    "数据面非确定"一节报出，不会判 FAIL —— 这是刻意划的边界（见 README §4.2）。
"""

import hashlib
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOGS = os.path.join(ROOT, "evidence", "logs")

HDR = re.compile(r"\[probe vf nest\] tag=(\S+) outer=(\d+) inner=(\d+) body=(\d+) ct_rows=(\d+)")
SUM = re.compile(r"\[probe vf nest\] tag=(\S+) (RUN PASS|RUN FAIL)")
# 注意：一律用 [ \t] 而不是 \s —— 日志里 rows=0 的 cnt/expect/tri 行是**空值**，
# \s 会跨行匹配（把下一行的 "expect :" 当成 cnt 的值），踩过一次。
ROWS = re.compile(r"^[ \t]*rows=(\d+)[ \t]+verdict=(PASS|FAIL)", re.M)
CNT = re.compile(r"^[ \t]*cnt[ \t]*:[ \t]*(.*)$", re.M)
EXP = re.compile(r"^[ \t]*expect[ \t]*:[ \t]*(.*)$", re.M)
TRI = re.compile(r"^[ \t]*tri[ \t]*:[ \t]*(.*)$", re.M)
AGHDR = re.compile(
    r"\[probe ag\] tag=(\S+) start=(\d+)\s+stride=(\d+)\s+max_idx=(\d+)\s+"
    r"in_rng_bad=(\d+) oob_lanes=(\d+) oob_zero=(\d+) oob_first=(\S+)"
)
AGERR = re.compile(r"\[probe ag\] tag=(\S+) RUN FAIL：aclError=(\d+) \(start=(\d+) stride=(\d+)\)")

# 内层上界语义（对应 probe_vf_nest.asc 的 PROBE_INNER）：
#   0/4/5/6 = `j < i`（含把 i 提出到循环外、含 u16/u32/i32 上界变量）⇒ 期望 cnt = i
#   1/7     = `j < i-1`（带保护 / j+1<i）        ⇒ 期望 cnt = max(i-1, 0)
#   2       = `j < i+1`                          ⇒ 期望 cnt = i+1
#   3       = `j < rows`                         ⇒ 期望 cnt = rows
EXPECT_EQ_I = (0, 4, 5, 6)
EXPECT_IM1 = (1, 7)
EXPECT_IP1 = (2,)
EXPECT_ROWS = (3,)


def expcnt(inner, i, rows):
    if inner in EXPECT_EQ_I:
        return i
    if inner in EXPECT_IM1:
        return i - 1 if i >= 1 else 0
    if inner in EXPECT_IP1:
        return i + 1
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def logs_of(logs, tag):
    """返回 [(rep, path)]，按 rep 升序。"""
    out = []
    for name in os.listdir(logs):
        m = re.fullmatch(r"run_%s_rep(\d+)\.log" % re.escape(tag), name)
        if m:
            out.append((int(m.group(1)), os.path.join(logs, name)))
    return sorted(out)


def ag_logs(logs):
    out = []
    for name in sorted(os.listdir(logs)):
        m = re.fullmatch(r"run_ag_gather_sweep_s(\d+)_x(\d+)\.log", name)
        if m:
            out.append((int(m.group(2)), int(m.group(1)), os.path.join(logs, name)))
    return out


def all_nest_tags(logs):
    tags = set()
    for name in os.listdir(logs):
        m = re.fullmatch(r"run_(\S+)_rep\d+\.log", name)
        if m:
            tags.add(m.group(1))
    return sorted(tags)


# ---------------------------------------------------------------- 列分组

ACC_RE = re.compile(r"acc_maxdiff[^\n]*")
HASH_RE = re.compile(r"hash=0x[0-9a-fA-F]+")


def judge_lines(txt):
    """判据列：结论所依赖的那些行（去掉会随 UB 残留变化的 hash=）。"""
    keep = []
    for ln in txt.splitlines():
        s = ln.rstrip()
        st = s.strip()
        if HASH_RE.search(st):
            s = HASH_RE.sub("hash=<strip>", s)
        if re.match(r"^(cnt|expect|tri|sumj)\s+:", st) or re.match(r"^rows=\d+ verdict=", st):
            keep.append(s)
        elif st.startswith("MISMATCH") or st.startswith("JSET-ODD") or st.startswith("SUMJ-ODD") \
                or st.startswith("TRI-ODD") or st.startswith("MK-") or "lane_mismatch=" in st \
                or "少一次的行数=" in st or st.startswith("rows=") and "verdict=" in st:
            keep.append(s)
        elif re.match(r"^\[probe vf nest\] tag=\S+ RUN (PASS|FAIL)", st):
            keep.append(s)
    return "\n".join(keep)


def dataface_lines(txt):
    return "\n".join(ln.rstrip() for ln in txt.splitlines() if ACC_RE.search(ln) or HASH_RE.search(ln))


def sig(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 核验

def verify(logs, quiet=False):
    """返回 (exit_code, report_lines)。三态：0=通过 / 1=有差异 / 2=没得比。"""
    rep = []
    if not os.path.isdir(logs):
        return 2, ["LOGS 目录不存在：%s" % logs]
    tags = all_nest_tags(logs)
    ags = ag_logs(logs)
    if not tags and not ags:
        return 2, ["没有可比对象：%s 下既没有 run_<tag>_rep*.log 也没有 run_ag_gather_sweep_*.log" % logs]

    compared = []
    skipped = []
    diffs = []
    datadiffs = []
    indep = []
    n_rep_total = 0

    for tag in tags:
        runs = logs_of(logs, tag)
        txt1 = read(runs[0][1])
        m = HDR.search(txt1)
        if not m:
            skipped.append((tag, "无 [probe vf nest] 表头（非本探针的日志？）"))
            continue
        _, outer, inner, body, ctrows = m.groups()
        outer, inner, body = int(outer), int(inner), int(body)
        n_rep_total += len(runs)
        compared.append((tag, len(runs)))

        # (a) 跨 rep：判据列必须逐字符一致；数据面列差异只单列
        j0 = judge_lines(txt1)
        sig0 = sig(j0)
        d0 = sig(dataface_lines(txt1))
        for rk, path in runs[1:]:
            t = read(path)
            if sig(judge_lines(t)) != sig0:
                diffs.append("%s rep%d：**判据列**与 rep%d 不一致" % (tag, rk, runs[0][0]))
            if sig(dataface_lines(t)) != d0:
                datadiffs.append("%s rep%d：数据面列（acc_maxdiff/hash）与 rep%d 不同" % (tag, rk, runs[0][0]))
        if len(runs) < 2:
            skipped.append((tag, "只有 %d 个 rep（<2），无法做跨 rep 比较" % len(runs)))

        # (b) 判据列独立复算：用 inner 的语义在工具侧重算 expect，并核 cnt/verdict 自洽
        for rk, path in runs:
            t = read(path)
            blocks = t.split("\n  rows=")[1:]
            if not blocks:
                indep.append("%s rep%d：没有 rows 块可复算" % (tag, rk))
                continue
            for blk in blocks:
                rnum = int(blk.split("\n", 1)[0].split()[0])
                cm, em, tm = CNT.search(blk), EXP.search(blk), TRI.search(blk)
                vm = ROWS.search(blk)
                if not (cm and em and vm):
                    indep.append("%s rep%d rows=%d：缺 cnt/expect/verdict 行" % (tag, rk, rnum))
                    continue
                cnt = [int(x) for x in cm.group(1).split()]
                lexp = [int(x) for x in em.group(1).split()]
                want = [expcnt(inner, i, rnum) for i in range(rnum)]
                if lexp != want:
                    indep.append("%s rep%d rows=%d：**日志里的 expect** %s ≠ 工具独立复算 %s"
                                 % (tag, rk, rnum, lexp, want))
                verdict = vm.group(2)
                mism = sum(1 for i in range(rnum) if cnt[i] != want[i])
                want_verdict = "PASS" if mism == 0 else "FAIL"
                if verdict != want_verdict:
                    indep.append("%s rep%d rows=%d：日志 verdict=%s 但工具复算为 %s（cnt=%s expect=%s）"
                                 % (tag, rk, rnum, verdict, want_verdict, cnt, want))
                # tri 自洽：tri[i] 应 = t(t+1)/2（t = 该行执行次数）
                if tm:
                    tri = [int(x) for x in tm.group(1).split()]
                    for i in range(min(len(tri), rnum)):
                        if tri[i] != cnt[i] * (cnt[i] + 1) // 2:
                            indep.append("%s rep%d rows=%d i=%d：tri=%d 与 cnt=%d 不自洽（应为 %d）"
                                         % (tag, rk, rnum, i, tri[i], cnt[i], cnt[i] * (cnt[i] + 1) // 2))

    n_ag = len(ags)
    for stride, start, path in ags:
        t = read(path)
        if not (AGHDR.search(t) or AGERR.search(t)):
            skipped.append(("ag s=%d x=%d" % (start, stride), "无 [probe ag] 结果行"))
        elif not AGHDR.search(t) and not AGERR.search(t):
            skipped.append(("ag s=%d x=%d" % (start, stride), "无结果行"))

    code = 2
    if compared or n_ag:
        code = 0
    if diffs or indep:
        code = 1

    rep.append("## summarize.py verify —— 覆盖范围自述")
    rep.append("")
    rep.append("- LOGS 目录：`%s`" % logs)
    rep.append("- nest target：找到 **%d** 个，比对 **%d** 个 rep 日志（每个 target 期望 ≥5）"
               % (len(compared), n_rep_total))
    rep.append("- gather 档位日志：**%d** 个（仅做存在性检查，判据在 README §5② 表里）" % n_ag)
    rep.append("- 跳过/无法比对：**%d** 条%s" % (len(skipped), ("（见下）" if skipped else "")))
    for tag, why in skipped:
        rep.append("    - %s：%s" % (tag, why))
    rep.append("")
    rep.append("## 判据列跨 rep 一致性（期望 0 条差异）")
    rep.append("")
    if diffs:
        for d in diffs:
            rep.append("- ❌ " + d)
    else:
        rep.append("- ✅ 全部 target 的判据列（cnt/expect/tri/MISMATCH/lane_mismatch/verdict）在找到的 rep 之间逐字符一致")
    rep.append("")
    rep.append("## 数据面列跨 rep 差异（**不是判据**；读越界 UB 的病态变体会天然不同）")
    rep.append("")
    if datadiffs:
        for d in sorted(set(datadiffs)):
            rep.append("- ⚠ " + d)
    else:
        rep.append("- （无：所有 target 的数据面列也一致）")
    rep.append("")
    rep.append("## 判据列独立复算（工具侧重算 expect / verdict / tri 自洽）")
    rep.append("")
    if indep:
        for d in sorted(set(indep)):
            rep.append("- ❌ " + d)
    else:
        rep.append("- ✅ 全部可比对的 rows 块都通过（expect 与工具复算相同、verdict 与复算一致、tri = t(t+1)/2）")
    rep.append("")
    rep.append("退出码：**%d**（0=通过 / 1=有差异 / 2=没得比）" % code)

    if not quiet:
        for ln in rep:
            print(ln)
    return code, rep


# ---------------------------------------------------------------- 回放/矩阵表

def matrix_nest(logs):
    print("| target | OUTER | INNER | BODY | rows 档位 | 结果 | 失败档位 | 关键读数（rows=4，cnt vs expect） |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in all_nest_tags(logs):
        runs = logs_of(logs, tag)
        if not runs:
            continue
        txt = read(runs[0][1])
        m = HDR.search(txt)
        if not m:
            print("| %s | — | — | — | — | (无 header) | | |" % tag)
            continue
        _, o, i, b, ct = m.groups()
        sm = SUM.search(txt)
        verdict = sm.group(2) if sm else "?"
        blocks = txt.split("\n  rows=")[1:]
        failrows, detail = [], ""
        for blk in blocks:
            rnum = blk.split("\n", 1)[0].split()[0]
            if "verdict=FAIL" in blk:
                failrows.append(rnum)
            if rnum == "4":
                cm, em = CNT.search(blk), EXP.search(blk)
                if cm and em:
                    detail = "cnt=[%s] / expect=[%s]" % (cm.group(1).strip(), em.group(1).strip())
        if not detail:
            for blk in blocks:
                if blk.split("\n", 1)[0].split()[0] == "64":
                    cm = CNT.search(blk)
                    if cm:
                        detail = "rows=64 cnt=[%s]" % cm.group(1).strip()
        print("| %s | %s | %s | %s | %d | %s | %s | %s |"
              % (tag, o, i, b, len(blocks), verdict, ",".join(failrows) if failrows else "—", detail or "—"))


def matrix_ag(logs):
    rows = []
    for stride, start, path in ag_logs(logs):
        txt = read(path)
        m = AGHDR.search(txt)
        if m:
            rows.append((int(m.group(3)), int(m.group(2)), int(m.group(4)), int(m.group(5)),
                         int(m.group(6)), int(m.group(7))))
        else:
            e = AGERR.search(txt)
            rows.append((stride, start, 0, -1, 0, 0))
    print("| stride | start | max_idx = start+63·stride | 数组内 lane 不符 | 越界 lane（未异常） | 结果 |")
    print("|---|---|---|---|---|---|")
    for stride, start, maxidx, bad, oob, _z in sorted(rows):
        if bad < 0:
            print("| %d | %d | - | - | - | **运行失败**（aclError=507035） |" % (stride, start))
        else:
            print("| %d | %d | %s | %d | %d | %s |"
                  % (stride, start, maxidx if maxidx else "-", bad, oob, "PASS" if bad == 0 else "FAIL"))


# ---------------------------------------------------------------- reindex

def reindex(logs):
    """从 run_*.log 重建 run_matrix.txt 与 sha256.txt（带 target 标签 + FAIL 计数 + 判据列签名）。"""
    rm = []
    rm.append("# probe_vf_loop 运行结果矩阵（由 tools/summarize.py reindex 从 run_*.log 重建）")
    rm.append("# 每个 target 三行：RUN PASS/FAIL / rows 档位 N，FAIL M，判据列跨 rep 一致|不一致 / 数据面列跨 rep 一致|不一致")
    sh = []
    sh.append("# probe_vf_loop 逐 (target, rep) 摘要（由 tools/summarize.py reindex 从 run_*.log 重建）")
    sh.append("# 列：<日志 sha256>  <target>  rep<k>  exit=<rc>  judge=<判据列 sha16>  <JUDGE-FIRST|JUDGE-SAME|JUDGE-DIFF>")
    for tag in all_nest_tags(logs):
        runs = logs_of(logs, tag)
        if not runs:
            continue
        txt = read(runs[0][1])
        sm = SUM.search(txt)
        verdict = sm.group(2) if sm else "?"
        nrows = len(txt.split("\n  rows=")[1:])
        nfail = len(re.findall(r"^[ \t]*rows=\d+[ \t]+verdict=FAIL", txt, re.M))
        first = None
        judge_diff = False
        data_first = None
        data_diff = False
        for rk, path in runs:
            t = read(path)
            j = sig(judge_lines(t))
            d = sig(dataface_lines(t))
            if data_first is None:
                data_first = d
            elif d != data_first:
                data_diff = True
            state = "JUDGE-FIRST"
            if first is None:
                first = j
            else:
                state = "JUDGE-SAME" if j == first else "JUDGE-DIFF"
            if state == "JUDGE-DIFF":
                judge_diff = True
            # exit：host 进程退出码（本探针语义：全部 rows 档 PASS → 0，否则 1）
            exit_code = 1 if "verdict=FAIL" in t else 0
            sh.append("%s  %s  rep%d  exit=%d  judge=%s  %s"
                      % (sha256(path), tag, rk, exit_code, j[:16], state))
        rm.append("%-24s %s" % (tag, "[probe vf nest] tag=%s %s" % (tag, verdict)))
        rm.append("%-24s   rows 档位 %d，FAIL %d，判据列跨 rep %s"
                  % (tag, nrows, nfail, "**不一致**" if judge_diff else "一致"))
        rm.append("%-24s   数据面列(acc_maxdiff/hash)跨 rep %s"
                  % (tag, "**不一致**（读越界 UB，非判据；见 summarize_verify.txt）" if data_diff else "一致"))
    for stride, start, path in ag_logs(logs):
        txt = read(path)
        m = AGHDR.search(txt)
        if m:
            note = ("in_rng_bad=%s oob_lanes=%s oob_zero=%s oob_first=%s"
                    % (m.group(5), m.group(6), m.group(7), m.group(8)))
        else:
            e = AGERR.search(txt)
            note = "aclError=%s" % (e.group(2) if e else "?")
        rm.append("%-42s %s" % ("ag_gather_sweep s=%d x=%d" % (start, stride), note))
        sh.append("%s  ag_gather_sweep_s%d_x%d  rep1  exit=%d  judge=n/a  AG"
                  % (sha256(path), start, stride, 1 if "aclError" in note else 0))
    with open(os.path.join(logs, "run_matrix.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rm) + "\n")
    with open(os.path.join(logs, "sha256.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sh) + "\n")
    print("reindex：写入 %s 与 %s（nest %d 行 / ag %d 行）"
          % (os.path.join(logs, "run_matrix.txt"), os.path.join(logs, "sha256.txt"),
             len(rm), len(ag_logs(logs))))


# ---------------------------------------------------------------- selftest（负向对照）

def selftest():
    """在 /tmp 副本里做三种变异，验证核验行为符合声明（不改动工作区任何文件）。"""
    ok = True
    tmp = tempfile.mkdtemp(prefix="vfprobe_selftest_")
    try:
        src = LOGS
        dst = os.path.join(tmp, "logs")
        shutil.copytree(src, dst)
        print("## selftest（副本目录 %s，工作区只读）" % dst)

        # 0) 基线：未变异 ⇒ 期望 exit 0
        code0, _ = verify(dst, quiet=True)
        print("- 基线（未变异）：exit=%d %s" % (code0, "✅" if code0 == 0 else "❌ 期望 0"))
        ok = ok and code0 == 0

        # 1) 变异**判据列**：把一个确定性 target 的 rep2 的 cnt 行改一个数字 ⇒ 期望 exit 1 且点名该 target
        tgt = "nest_o0i3b1" if logs_of(dst, "nest_o0i3b1") else all_nest_tags(dst)[0]
        p = logs_of(dst, tgt)[1][1]
        s = read(p)
        s2 = re.sub(r"(?m)^(\s*cnt\s+:.*?)4\b", r"\g<1>3", s, count=1)
        assert s2 != s, "变异未生效"
        with open(p, "w", encoding="utf-8") as f:
            f.write(s2)
        code1, rep1 = verify(dst, quiet=True)
        hit = any(tgt in d for d in rep1 if "❌" in d)
        print("- 变异判据列（%s rep2 的 cnt 行）：exit=%d（期望 1）%s，点名 target=%s"
              % (tgt, code1, "✅" if code1 == 1 else "❌", "✅" if hit else "❌"))
        ok = ok and code1 == 1 and hit
        with open(p, "w", encoding="utf-8") as f:
            f.write(s)

        # 2) 负向对照（**已知会被漏掉**）：只改数据面 acc_maxdiff 行 ⇒ 判据列仍一致
        #    ⇒ 期望 exit 0，但"数据面列差异"一节必须报出（不是静默 OK）
        tgt2 = "nest_o0i0b1" if logs_of(dst, "nest_o0i0b1") else tgt
        p2 = logs_of(dst, tgt2)[1][1]
        s = read(p2)
        s2 = re.sub(r"(?m)^(\s*acc_maxdiff[^\n]*?)0\.0", r"\g<1>1.0", s, count=1)
        assert s2 != s, "数据面变异未生效"
        with open(p2, "w", encoding="utf-8") as f:
            f.write(s2)
        code2, rep2 = verify(dst, quiet=True)
        noted = any(("⚠" in d and tgt2 in d) for d in rep2)
        print("- 负向对照（只改数据面 %s rep2 的 acc_maxdiff）：exit=%d（期望 0 = 判据列确实没变）%s，"
              "数据面差异被单列出=%s ✅" % (tgt2, code2, "✅" if code2 == 0 else "❌",
                                       "是" if noted else "否 ❌"))
        ok = ok and code2 == 0 and noted
        with open(p2, "w", encoding="utf-8") as f:
            f.write(s)

        # 3) 缺输入 ⇒ 期望 exit 2（与"通过"区分）
        code3, _ = verify(os.path.join(tmp, "nonexistent"), quiet=True)
        print("- 缺输入（不存在的目录）：exit=%d（期望 2）%s" % (code3, "✅" if code3 == 2 else "❌"))
        ok = ok and code3 == 2

        print("- selftest 结论：%s" % ("✅ 全部符合声明" if ok else "❌ 有不符合项"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    logs = LOGS
    if "--logs" in args:
        k = args.index("--logs")
        logs = os.path.abspath(args[k + 1])
        del args[k:k + 2]
    what = args[0] if args else "all"

    if what in ("all", "matrix", "nest", "ag"):
        if what in ("all", "matrix", "nest"):
            print("## nest 矩阵（回放 rep1；判据一致性用 `verify`）\n")
            matrix_nest(logs)
            print()
        if what in ("all", "matrix", "ag"):
            print("## gather 扫描（回放）\n")
            matrix_ag(logs)
    elif what == "verify":
        code, _ = verify(logs)
        sys.exit(code)
    elif what == "reindex":
        reindex(logs)
    elif what == "selftest":
        sys.exit(selftest())
    elif what == "det":
        runs = logs_of(logs, args[1])
        print(read(runs[0][1]) if runs else "no log for %s" % args[1])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
