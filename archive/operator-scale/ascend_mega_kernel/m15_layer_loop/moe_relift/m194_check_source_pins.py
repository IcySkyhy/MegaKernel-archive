#!/usr/bin/env python3
"""M194: 离线复算 mission scope 内「源码 sha256 指纹 pin」的枚举与陈旧判定。

口径：一行里同时出现一个 64-hex 记号与一个 `.h/.asc/.py`（本脚本另扩 `.sh`）路径。
对每条命中：把记录值反查到它所属的 rev（对该路径的历史逐 rev 复算内容 sha256），
再与当前工作树的内容 sha256 对比，判 OK / STALE / UNVERIFIABLE。

内置期望表 EXPECT；实际结果与期望不符即非零退出（1）。本脚本**离线**运行，不需要设备。

退出码：0 = 与内置期望一致；1 = 有偏差（枚举集或任一条判定不符）。

复算命令：
    python3 m15_layer_loop/moe_relift/m194_check_source_pins.py
"""
import hashlib
import os
import re
import subprocess
import sys

HEX = re.compile(r"(?<![0-9a-fA-F])([0-9a-f]{64})(?![0-9a-fA-F])")
PATHRE = re.compile(r"([A-Za-z0-9_./-]+\.(?:h|asc|py|sh))")

SCOPE = [
    "baseline_env/evidence",
    "m15_layer_loop/moe_relift",
    "m15_layer_loop/evidence/m58_lift_diff.txt",
]

# log 里的 tier 表只印 basename，需按上下文把裸名解析到真实仓库路径。
RESOLVE = {
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 10): "m15_layer_loop/m15_moe_layer.h",
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 11): "m15_layer_loop/moe_relift/m95_e512/build/m15_moe_layer.h",
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 12): "m15_layer_loop/m15_moe_resources.h",
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 13): "m15_layer_loop/moe_relift/m95_e512/build/m15_moe_resources.h",
}

# 这些是被测副本（build 产物，未入库），无法离线复算。
BUILD_ARTIFACTS = {
    "m15_layer_loop/moe_relift/m95_e512/build/m15_moe_layer.h",
    "m15_layer_loop/moe_relift/m95_e512/build/m15_moe_resources.h",
}

# 历史日志正文必须逐字节不动（M190/M194 红线）。
FROZEN_LOGS = [
    "m15_layer_loop/moe_relift/m95_r3_fixes.log",
    "m15_layer_loop/moe_relift/m95_e512/run_commands.log",
]

# 期望表：key = (相对仓库根的命中文件, 行号, 记录值)
# 值 = (状态, 当前实际 sha256 或 None, 记录值最后所属 rev 前缀, 该值失效的 rev 前缀或 None)
# rev 以前缀比对。UNVERIFIABLE 条目 rev 全为 None。
EXPECT = {
    ("baseline_env/evidence/13-number-audit-script-audit.md", 145, "a1ec1700f91a03f128d3d0b82961d1a05189e4c482d066774feec2ecbebd8ee4"): (
        "OK", "a1ec1700f91a03f128d3d0b82961d1a05189e4c482d066774feec2ecbebd8ee4", "eb915c469a", None),
    ("baseline_env/evidence/13-number-audit-script-audit.md", 146, "9c5759c7aa9475b4e777f0369ce5a9fd211886507bfd141642390cef2f211838"): (
        "OK", "9c5759c7aa9475b4e777f0369ce5a9fd211886507bfd141642390cef2f211838", "eb915c469a", None),
    ("baseline_env/evidence/13-number-audit-script-audit.md", 147, "89b4699257893852a0ddb84a50507d3a435392a7b1fe44d446c91cfa13909759"): (
        "STALE", "e22f71e9c6d3af7a9b4da54efb1707f99a13ac959736c0b591b3e97659f2b736", "eb915c469a", "0c163b5b88"),
    ("baseline_env/evidence/13-number-audit-script-audit.md", 148, "94a4a706ad601e24ec26de54f740d9b11bddc91eb805d5fb2533ef1f8cd05f56"): (
        "OK", "94a4a706ad601e24ec26de54f740d9b11bddc91eb805d5fb2533ef1f8cd05f56", "699e00f815", None),
    ("baseline_env/evidence/13-number-audit-script-audit.md", 149, "878831f8dd72dc1aed181afb9caeb57715b9c13510e5bcf258608a2db08f9f3c"): (
        "OK", "878831f8dd72dc1aed181afb9caeb57715b9c13510e5bcf258608a2db08f9f3c", "699e00f815", None),
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 10, "0af84032dc194e8a0350caf822e91ae005833dae8df52b0ce98ea65c577b2a64"): (
        "STALE", "36d00d187a6c6e500cb269b7275e1e84bc4cd08e82ec9c569f947cb1d1f02c62", "9935c77052", "186d97ccbe"),
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 11, "0af84032dc194e8a0350caf822e91ae005833dae8df52b0ce98ea65c577b2a64"): (
        "UNVERIFIABLE", None, None, None),
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 12, "971dd3b02342fcdcbf47b4815bd1e14ec6bcfe982c2b5ac0b7629a8cd5b41d5b"): (
        "STALE", "ba39888b4ca4b1e75f622a91c816f63616e9e25c666002b0072f8dc1a648dbf7", "9935c77052", "8337542d0c"),
    ("m15_layer_loop/moe_relift/m95_e512/run_commands.log", 13, "02553fbe3f9a738046473c001e75ec9749c8114b31a0d599be85210eae8b7605"): (
        "UNVERIFIABLE", None, None, None),
    ("m15_layer_loop/moe_relift/m95_r3_fixes.log", 65, "0af84032dc194e8a0350caf822e91ae005833dae8df52b0ce98ea65c577b2a64"): (
        "STALE", "36d00d187a6c6e500cb269b7275e1e84bc4cd08e82ec9c569f947cb1d1f02c62", "9935c77052", "186d97ccbe"),
}


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True).stdout


def repo_root():
    out = git("rev-parse", "--show-toplevel").strip()
    return out or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def content_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def rev_content_sha(rev, path):
    r = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def history(base, path):
    return git("log", "--format=%H", base, "--", path).split()


def last_valid_and_change(base, path, recorded):
    """(最后仍等于 recorded 的 rev, 使 recorded 失效的 rev)。"""
    revs = history(base, path)
    for i, rev in enumerate(revs):
        if rev_content_sha(rev, path) == recorded:
            return rev, (revs[i - 1] if i > 0 else None)
    return None, None


def enumerate_hits(root):
    files = []
    for rel in SCOPE:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            files.append(rel)
        else:
            for dp, _, fns in os.walk(p):
                for fn in fns:
                    files.append(os.path.relpath(os.path.join(dp, fn), root))
    hits = []
    for rel in sorted(set(files)):
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        for i, bl in enumerate(data.split(b"\n"), 1):
            line = bl.decode("utf-8", "replace")
            for m in HEX.finditer(line):
                for tok in PATHRE.findall(line):
                    hits.append((rel, i, m.group(1), tok))
    return hits


def resolve_target(rel, lineno, token):
    if (rel, lineno) in RESOLVE:
        return RESOLVE[(rel, lineno)]
    if os.path.exists(os.path.join(repo_root(), token)):
        return token
    return token


def main():
    root = repo_root()
    base = os.environ.get("M194_REV", "main")
    if not git("rev-parse", "--verify", base).strip():
        print("[M194][FAIL] 找不到基准 rev: %s" % base)
        return 1

    hits = enumerate_hits(root)
    print("== 枚举（scope=%s）==" % ", ".join(SCOPE))
    problems = []

    seen_keys = set()
    for rel, lineno, recorded, token in hits:
        target = resolve_target(rel, lineno, token)
        key = (rel, lineno, recorded)
        seen_keys.add(key)
        tgt_abs = os.path.join(root, target)
        if target in BUILD_ARTIFACTS:
            status, cur, valid, change = "UNVERIFIABLE", None, None, None
        else:
            cur = content_sha(tgt_abs)
            valid, change = last_valid_and_change(base, target, recorded)
            status = "OK" if cur == recorded else "STALE"
        print("  %s:%d  -> %s\n      recorded=%s  %s%s" % (
            rel, lineno, target, recorded, status,
            ("\n      current=" + cur) if cur else ""))
        exp = EXPECT.get(key)
        if exp is None:
            problems.append("枚举到未列入期望的命中: %s:%d %s" % (rel, lineno, recorded))
            continue
        e_status, e_cur, e_valid, e_change = exp
        if status != e_status:
            problems.append("%s:%d 状态 %s != 期望 %s" % (rel, lineno, status, e_status))
        if cur != e_cur:
            problems.append("%s:%d 当前值 %s != 期望 %s" % (rel, lineno, cur, e_cur))
        if e_valid is None:
            if valid is not None:
                problems.append("%s:%d 期望 UNVERIFIABLE 但有历史 rev" % (rel, lineno))
        else:
            if valid is None or not valid.startswith(e_valid):
                problems.append("%s:%d 最后所属 rev %s 不以 %s 开头" % (rel, lineno, valid, e_valid))
            if e_change is None:
                if change is not None:
                    problems.append("%s:%d 期望无失效 rev，实得 %s" % (rel, lineno, change))
            elif change is None or not change.startswith(e_change):
                problems.append("%s:%d 失效 rev %s 不以 %s 开头" % (rel, lineno, change, e_change))

    missing = set(EXPECT) - seen_keys
    for k in sorted(missing):
        problems.append("期望表中的命中未被枚举到: %s:%d %s" % (k[0], k[1], k[2]))

    print("\n== 历史日志正文冻结检查 ==")
    for log in FROZEN_LOGS:
        wt = git("status", "--porcelain", "--", log).strip()
        idx = git("diff", "--name-only", "--", log).strip()
        branch = git("diff", "--name-only", base + "...HEAD", "--", log).strip()
        ok = not wt and not idx and not branch
        print("  %s: %s" % (log, "未改动" if ok else "已改动"))
        if not ok:
            problems.append("冻结日志被改动: %s" % log)

    print()
    if problems:
        print("RESULT: FAIL (%d 条偏差)" % len(problems))
        for p in problems:
            print("  - " + p)
        return 1
    print("RESULT: OK (%d 条 pin：%d 陈旧 / %d 未变 / %d 无法离线复算)" % (
        len(EXPECT),
        sum(1 for v in EXPECT.values() if v[0] == "STALE"),
        sum(1 for v in EXPECT.values() if v[0] == "OK"),
        sum(1 for v in EXPECT.values() if v[0] == "UNVERIFIABLE")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
