#!/usr/bin/env python3
"""
check_archive_matches_list.py —— **归档是否真的由当前代码产出**（一条命令的可复核守卫）

为什么需要它：归档与提交代码之间的一致性必须能被**一条命令**复核，否则"归档由当前代码产出"
这句话只能靠人工把两份 (op,a,b,n) 序列对一遍。历史上确实漂移过：归档里一行来自更早的实现、
且 `matrix.txt` 的行序是当时的 `PrintList` 产不出的（那次靠人工比对才发现）。
另外 `FAULT_MSG` 里的 `hash=` **不能**当源码指纹（同一次归档里不同 kernel 的 hash 就不同），
所以一致性只能靠**结构性比对**。

四条检查（都只看归档 + 当前二进制，不碰设备）：
  1. **行序与内容**：`matrix.txt` 里**每一行**的 (op,a,b,n,group) 序列必须与当前
     `./build/probe_v_align list` 的**顺序逐项相同**（顺序也查 —— P2-1 症状 B 正是顺序不一致）。
  2. **逐变体 log 与矩阵列一致**：每条执行过的变体，其 log 里的 `LAUNCH aclError`/`DUMP_FNV`
     必须与矩阵对应列相同；log 里必须有 `OUTCOME:` 行。
  3. **guard ②（§8）**：每个 op 的**偏移 0 对照点**（32B 对齐）必须 `OK`——否则该 op 的结论无法解读。
  4. **窗口自洽**：同一份日志里 `VARIANT … window=X` 必须等于 `COMPARE window_bytes=X`
     （声明值 vs 模型值不一致过：`redsum`/`redmax` 曾一个日志里同时出现 256 与 4；根因是两处各写一次，
     现已由 `WindowOfImpl` 收敛为单一来源，本检查是防它再分叉的回归项）。

**负向对照**（规则要求：工具必须证明自己会咬）：`--selftest` 有两个正对照——
① 喂一份被人为扰动的 list（顺序扰动 + 抹掉一条），要求报 FAIL；
② 在 /tmp 复制一份归档并把某条 log 的 `VARIANT … window=` 改坏，要求第 4 项检查报 FAIL。
读数为 `SELFTEST: OK (两个负向对照都被检出…)`。

三态退出码（docs/17 §8.3）：0=比过且通过 / 1=比过且有差异 / 2=没得比（缺 list 或 matrix）

用法：python3 tools/check_archive_matches_list.py <probe_v_align 目录> [--selftest]
"""
import os
import re
import subprocess
import sys

ARENA = 1024


def load_matrix(probe):
    rows = []
    path = os.path.join(probe, "evidence", "logs", "matrix.txt")
    if not os.path.isfile(path):
        return None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("total="):
            continue
        kv = dict(p.split("=", 1) for p in line.split() if "=" in p)
        if "op" not in kv:
            continue
        grp = line[line.rindex("(") + 1:line.rindex(")")] if "(" in line else ""
        rows.append({
            "outcome": line.split()[0], "op": kv["op"], "a": kv["a"], "b": kv["b"], "n": kv["n"],
            "group": grp, "aclError": kv.get("aclError", "NA"), "fnv": kv.get("fnv", "-"),
        })
    return rows


def load_list(probe, explicit=None):
    if explicit is not None:
        return explicit
    exe = os.path.join(probe, "build", "probe_v_align")
    if not os.access(exe, os.X_OK):
        return None
    out = subprocess.run([exe, "list"], capture_output=True, text=True, cwd=probe).stdout
    rows = []
    for line in out.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        op, a, b, n, grp = line.split()
        rows.append((op, a, b, n, grp))
    return rows


def check(probe, list_override=None, verbose=True):
    matrix = load_matrix(probe)
    if matrix is None:
        print("RESULT: SKIPPED (evidence/logs/matrix.txt 缺失) —— 没得比，不发合格证")
        return 2
    listing = load_list(probe, list_override)
    if listing is None:
        print("RESULT: SKIPPED (build/probe_v_align 不存在或不可执行) —— 没得比，不发合格证")
        return 2

    n_fail = 0
    # --- 1) 行序与内容 ---
    m_seq = [(r["op"], r["a"], r["b"], r["n"], r["group"].split("（")[0]) for r in matrix]
    l_seq = [(o, a, b, n, g) for (o, a, b, n, g) in listing]
    # matrix 里 STOPPED 行的 group 带中文括注，已在上一步剥掉
    if len(m_seq) != len(l_seq):
        n_fail += 1
        print(f"[1] 行数不一致：matrix {len(m_seq)} vs list {len(l_seq)}")
    n_bad = 0
    for i, (m, l) in enumerate(zip(m_seq, l_seq)):
        if m != l:
            n_bad += 1
            if n_bad <= 3:
                print(f"[1] 第 {i + 1} 行不一致：matrix={m} list={l}")
            n_fail += 1
    print(f"[1] 行序核对：比过 {min(len(m_seq), len(l_seq))} 行，位置不一致 {n_bad} 行")

    # --- 2) log 与矩阵列一致 ---
    n_logchk = 0
    for r in matrix:
        if r["outcome"] == "STOPPED":
            continue
        log = os.path.join(probe, "evidence", "logs", f"run_{r['op']}_a{r['a']}_b{r['b']}_n{r['n']}.log")
        if not os.path.isfile(log):
            print(f"[2] 缺对应 log：{os.path.basename(log)}")
            n_fail += 1
            continue
        txt = open(log, errors="replace").read()
        n_logchk += 1
        if "OUTCOME:" not in txt:
            print(f"[2] log 无 OUTCOME 行：{os.path.basename(log)}")
            n_fail += 1
        m = re.search(r"DUMP_FNV (0x[0-9a-f]+)", txt)
        got_fnv = m.group(1) if m else "-"
        if r["fnv"] not in ("-", "") and got_fnv != r["fnv"]:
            print(f"[2] FNV 不一致：{os.path.basename(log)} log={got_fnv} matrix={r['fnv']}")
            n_fail += 1
        m2 = re.search(r"LAUNCH aclError=(\d+)", txt)
        got_acl = m2.group(1) if m2 else "NA"
        if r["aclError"] not in ("-", "NA", "") and got_acl != r["aclError"]:
            print(f"[2] aclError 不一致：{os.path.basename(log)} log={got_acl} matrix={r['aclError']}")
            n_fail += 1
    print(f"[2] 逐变体 log 核对：{n_logchk} 条（aclError/DUMP_FNV 与矩阵列一致）")

    # --- 3) guard ②：偏移 0 对照点必须 OK ---
    # 登记例外（README §9.2 已披露，不是"通过"）：
    #   gatherb —— 偏移 0 即故障（用法/支持性存疑），不作对齐结论
    #   lmbarout —— 靶子本身就是要故障（LocalMemBar 在 VF 外）
    GUARD_EXEMPT = {"gatherb", "lmbarout"}
    ctrl = {}
    for r in matrix:
        if r["a"] == "0" and r["b"] == "0" and not r["group"].startswith(("nc", "pt", "len")):
            ctrl.setdefault(r["op"], []).append(r["outcome"])
    n_ctrl = 0
    n_exempt = 0
    for op, ocs in sorted(ctrl.items()):
        n_ctrl += 1
        if any(o != "OK" for o in ocs):
            if op in GUARD_EXEMPT:
                n_exempt += 1
                print(f"[3] guard② 登记例外：op={op} 偏移 0 = {ocs}（README §9.2 已披露，不作对齐结论）")
                continue
            print(f"[3] guard② 违反：op={op} 偏移 0 对照点结果为 {ocs}（应为 OK）")
            n_fail += 1
    print(f"[3] guard②（偏移 0 对照点 = OK）：核对 {n_ctrl} 个 op，登记例外 {n_exempt} 个")

    # --- 4) 窗口自洽：VARIANT window == COMPARE window_bytes ---
    n_winchk = 0
    n_winbad = 0
    for r in matrix:
        if r["outcome"] == "STOPPED":
            continue
        log = os.path.join(probe, "evidence", "logs", f"run_{r['op']}_a{r['a']}_b{r['b']}_n{r['n']}.log")
        if not os.path.isfile(log):
            continue
        txt = open(log, errors="replace").read()
        m_var = re.search(r"VARIANT .*?window=(\d+)", txt)
        m_cmp = re.search(r"COMPARE window_bytes=(\d+)", txt)
        if m_var is None:
            print(f"[4] log 缺 VARIANT window 字段：{os.path.basename(log)}")
            n_fail += 1
            n_winbad += 1
            continue
        if m_cmp is None:
            continue   # 故障变体没有 COMPARE 行（无窗口可对）
        n_winchk += 1
        if m_var.group(1) != m_cmp.group(1):
            print(f"[4] 窗口自相矛盾：{os.path.basename(log)} VARIANT window={m_var.group(1)} "
                  f"vs COMPARE window_bytes={m_cmp.group(1)}")
            n_fail += 1
            n_winbad += 1
    print(f"[4] 窗口自洽（VARIANT window == COMPARE window_bytes）：核对 {n_winchk} 条，冲突 {n_winbad} 条")

    # --- 报告项（**不作判定**）：设备序号沿矩阵顺序是否严格递增（单次完整运行的旁证之一；
    #     跨批拼装会留下回退或空洞。只有故障 log 带序号，故只在故障行上出现。）---
    serials = []
    for r in matrix:
        if r["outcome"] != "FAULT":
            continue
        log = os.path.join(probe, "evidence", "logs", f"run_{r['op']}_a{r['a']}_b{r['b']}_n{r['n']}.log")
        if not os.path.isfile(log):
            continue
        m = re.search(r"serial number is (\d+)", open(log, errors="replace").read())
        if m:
            serials.append((int(m.group(1)), r["op"], r["a"], r["b"], r["n"]))
    inc = all(serials[i][0] < serials[i + 1][0] for i in range(len(serials) - 1))
    cont = (len(serials) > 1 and serials[-1][0] - serials[0][0] == len(serials) - 1)
    print(f"[R] 设备序号（报告项，不参与判定）：{len(serials)} 条故障行带序号，"
          f"区间 {serials[0][0] if serials else '-'}..{serials[-1][0] if serials else '-'}，"
          f"沿矩阵顺序严格递增={inc}，连续={cont}")

    if n_fail:
        print(f"RESULT: FAIL ({n_fail} 处不一致；行序 {min(len(m_seq), len(l_seq))} 行 + log {n_logchk} 条 + "
              f"对照点 {n_ctrl} 个 + 窗口自洽 {n_winchk} 条)")
        return 1
    print(f"RESULT: OK (行序 {len(m_seq)} 行逐项一致 + log {n_logchk} 条列值一致 + guard② {n_ctrl} 个 op + "
          f"窗口自洽 {n_winchk} 条；与当前代码零差异)")
    return 0


def selftest(probe):
    """两个负向对照：① list 被扰动（顺序 + 删一条）；② 归档里某条 log 的 VARIANT window 被改坏"""
    base = load_list(probe)
    if base is None:
        print("SELFTEST: SKIPPED (无 list)")
        return 2
    ok = True
    pert = list(base)
    pert[3], pert[4] = pert[4], pert[3]          # 顺序扰动
    pert.pop()                                    # 抹掉一条
    rc = check(probe, list_override=pert, verbose=False)
    if rc == 1:
        print(f"SELFTEST ①（list 顺序扰动 + 抹掉 1 条）：被检出 → rc=1（基准 {len(base)} 条）")
    else:
        print(f"SELFTEST ①：**未被检出**（rc={rc}）")
        ok = False

    # ② 在 /tmp 造一份"窗口自相矛盾"的归档副本
    import shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="m57_win_selftest_")
    dst = os.path.join(tmp, "probe_copy")
    shutil.copytree(os.path.join(probe, "evidence"), os.path.join(dst, "evidence"))
    src_log = None
    for r in load_matrix(dst) or []:
        if r["outcome"] not in ("OK", "WRONG"):
            continue
        p = os.path.join(dst, "evidence", "logs", f"run_{r['op']}_a{r['a']}_b{r['b']}_n{r['n']}.log")
        if os.path.isfile(p) and "COMPARE window_bytes=" in open(p, errors="replace").read():
            src_log = p
            break
    if src_log is None:
        print("SELFTEST ②：SKIPPED（找不到可改的 log）")
        return 2 if not ok else 0
    t = open(src_log, errors="replace").read()
    t2 = re.sub(r"(VARIANT .*?window=)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 8), t, count=1)
    open(src_log, "w").write(t2)
    rc2 = check(dst, list_override=base, verbose=False)
    if rc2 == 1:
        print("SELFTEST ②（归档里 VARIANT window 被改坏）：被第 4 项检查检出 → rc=1")
    else:
        print(f"SELFTEST ②：**未被检出**（rc={rc2}）")
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    if ok:
        print("SELFTEST: OK（两个负向对照都被检出）")
        return 0
    print("SELFTEST: FAIL（有用例未被检出）")
    return 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    probe = os.path.abspath(args[0]) if args else os.getcwd()
    if "--selftest" in sys.argv:
        sys.exit(selftest(probe))
    sys.exit(check(probe))
