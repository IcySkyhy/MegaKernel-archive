#!/usr/bin/env python3
"""
diff_archives.py —— **两次归档之间的差异记账**（可复核、可复算、且**看得见它该看见的东西**）

为什么需要它：口头报"12 added / 71 modified / 1 rename，其中 60 文本 + 5 有值 + 6 非 log"
这种账别人无法复算，而且很容易把**新增**折进**修改**、把**非 log** 数错。
本脚本把记账规则写成代码。

## 分类口径（全部由脚本产出，不手写数字）

* `A` 新增 / `M` 修改 / `R` 重命名 / `D` 删除 —— 直接来自 `git diff --name-status -M [-- <path>]`；
* 修改的文件再分三栏：**变体 log**（`evidence/logs/run_*.log` / `ildl_*.log`）里"结果有差异"与
  "仅易变字段"；**其它**（dump / `matrix.txt` / 生成的总表 / 核对输出 / `commands.txt`）。
* 比对**整份文件的所有非空行**（掩码后），而不是只比几类前缀行 —— 因为判别性文字
  （设备 errcode `340`、`error code = 340`、`errorStr: …is not aligned`）就活在
  `FAULT_MSG` 的**续行**上，只比前缀行会把"同码不同因"这一类差异整类看漏。
* **分类规则（显式写出，避免"口径漂移"）**：把两边的行按前缀分成两组——
  `RESULT:` 行（**结论措辞**）与**其余行**（内容）：
  · 其余行有差异 ⇒ 记为"**结果有差异**"（并在字段里附 `RESULT(措辞)` 若措辞也变了）；
  · **只有** `RESULT:` 行不同 ⇒ 记为"仅易变字段/措辞"（措辞变更不是结果变更；这条口径与
    `b9d468f → 6256700` 的独立复算一致：**5 结果 + 60 文案**；若不排除 `RESULT:` 会算成 26 结果 + 39）。

## 掩码：规则与已知边界（这份自述**不含**"绝不掩测量值"这类保证）

* **规则 ①（易变字段，按标签）**：时间戳 / `PID:` / `serial number is N` / `stream_id` /
  `report_stream_id` / `task_id` / `program id` / `flip_num` / `hash=` / `device_id` /
  同步超时毫秒数 / `core id`。
* **规则 ②（地址，按片段）**：只掩**地址语境标签之后紧邻的那一串值** ——
  `pc start` / `current` / `sc|su|mte|vec|cube|l1 error info` / `aic error mask` / `para base`
  （连写的多个值如 `su error info: 0x…,0x…` 一起掩）。**同一行其它位置的 `0x…` 不动。**
* **边界（这是关键的一条，别读成保证）**：若某个**真实测量值**恰好写在这些标签之后，它会被掩掉。
  对本仓归档的可达性核查 —— **下面三条命令都在 `probe_v_align/` 目录里跑，读数就是它们的逐字输出**：
  · 地址语境标签之后的值：
    `grep -ohE '[A-Za-z_][A-Za-z_ ]{2,}: ?0x[0-9a-fA-F]+' evidence/logs/*.log | sed 's/:.*//' | sort | uniq -c`
    → 8 类标签**各 46 条**：`pc start` / `current` / `sc error info` / `su error info` / `mte error info` /
    `vec error info` / `aic error mask` / `para base`（这些正是规则 ② 会替换掉的值）；同一命令还会列出
    `subErrType` **46 条**（值 `0x4`）——它**不在标签表里**，所以**参与比对**；
  · 不带冒号的"标签 + `0x…`"：
    `grep -ohE '[A-Za-z_0-9]+[[:space:]]+0x[0-9a-fA-F]+' evidence/logs/*.log | sed -E 's/[[:space:]]+0x.*//' | sort | uniq -c`
    → `DUMP_FNV` **83 条** + `DUMP_FNV32` **7 条**（后者出现在 `ildl_*.log` 里）——两者都**参与比对**
    （正则须含数字：`[A-Za-z_]{3,}` 会漏掉 `DUMP_FNV32`，那是 7 条参与比对的项）；
  · `DUMP_FNV`/`DUMP_FNV32` 落在地址语境行上的次数（`grep -hE 'DUMP_FNV'` 同时匹配到 `DUMP_FNV32`）：
    `grep -hE 'DUMP_FNV' evidence/logs/*.log | grep -cE 'pc start|current|error info|aic error mask|para base'`
    → `0`（**该归档内** 0 次；计数为 0 时 `grep -c` 自身返回 rc=1，属正常）。
  ⇒ 按这三条读数，该归档里没有一个**参与比对**的项落在规则 ② 会替换的位置上；
  但这是**可达性事实**（只覆盖当前归档），不是实现上的排除保证。
* **其它已知边界**：① 本工具是 **ref-vs-ref**，不看工作树未提交内容，`--path` 只按 git 跟踪的文件匹配；
  ② `field_of` 的字段名取 `^[A-Z_]{3,}`，因此 `HEAD32` 会显示成 `HEAD`（只是标签可读性，不影响判定）；
  ③ **命令的 locale 敏感性**：上面三条命令的字符类都只用 ASCII（`[A-Za-z_]`、`[A-Za-z_0-9]`、`[0-9a-fA-F]`）；
  只有第二条用了 POSIX 类 `[[:space:]]` —— 它对**本仓的 ASCII 日志**在 `LC_ALL=C` / `POSIX` / `C.UTF-8` 下
  读数一致（实测 `83 DUMP_FNV + 7 DUMP_FNV32`），故保留。**反例**：若把字符类写成含多字节字符的形式
  （例如用 `[^)）]` 去截停组标签后的中文注解），`LC_ALL=C` 与 `C.UTF-8` 会给出**不同**读数 —— 本仓已经
  踩过一次，故此处只用 ASCII 类。

## 三态退出码（docs/17 §8.3）

`0` = 比过（有差异也照样 0，本工具是**记账**不是判据）；
`1` = 用法/参数错（子树不存在等）；
`2` = **没得比**（0 项：ref 与目标相同或无变更；或子树不存在）⇒ 打印 `RESULT: EMPTY` / `RESULT: SKIPPED`，
**绝不发"OK"**（"0 项"与"比过通过"必须能区分）。

`--selftest` 在 **/tmp 建丢弃式 git 仓库**逐字段注入（覆盖 FNV / errcode / error code / errorStr /
地址语境行上的指纹 token / COMPARE / OUTCOME / LAUNCH / VARIANT / HEAD32 / FAULT_CODE），
要求工具把每一类真变化都报出来、把纯环境噪声判为"仅易变字段"，并要求"子树写错"与"0 项"给出可区分状态。

`wt` 可以是 worktree 根 / probe 目录 / 任意子目录：git 操作自动锚到仓库根，`--path` 按仓库相对解析
（从 probe 目录里写 `--path=probe_v_align/evidence` 也能工作；本仓的正确口径就是它）。

用法：
    python3 tools/diff_archives.py <worktree> <old_ref> [new_ref=HEAD] [--path=probe_v_align/evidence]
    python3 tools/diff_archives.py <worktree> <ref> --selftest
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---- 环境噪声（非地址）----
VOLATILE = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[-T][\d:.]+"), "<TS>"),
    (re.compile(r"PID:\s*\d+"), "PID:<N>"),
    (re.compile(r"serial number is \d+"), "serial number is <N>"),
    (re.compile(r"\b(stream_id|report_stream_id|task_id|flip_num|device_id)=\d+"), r"\1=<N>"),
    (re.compile(r"program id=\d+"), "program id=<N>"),
    (re.compile(r"hash=\d+"), "hash=<N>"),
    (re.compile(r"core id is \d+"), "core id is <N>"),
    (re.compile(r"\(sync timeout=\d+ms\)"), "(sync timeout=<N>ms)"),
]

# ---- 地址：**按片段**掩 —— 只掩"地址语境标签之后紧邻的那一串值" ----
# 注意口径：**不是**"整行凡 0x… 都掩"（按行掩会把同一行上别处的指纹一并吃掉），
# **也不是**"绝不掩测量值"（若某个真实测量值恰好写在这些标签之后，它会被掩掉 —— 见文件头的边界说明）。
ADDR_LABELS = (r"(?:pc start|current|sc error info|su error info|mte error info|"
               r"vec error info|cube error info|l1 error info|aic error mask|para base)")
ADDR_VALUES = r"0x[0-9a-fA-F]+(?:\s*,\s*0x[0-9a-fA-F]+)*"
ADDR_AFTER_LABEL = re.compile(r"(" + ADDR_LABELS + r")(:\s*)(" + ADDR_VALUES + r")")


def _mask_addr_values(m):
    return m.group(1) + m.group(2) + re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", m.group(3))


def mask(txt):
    for pat, sub in VOLATILE:
        txt = pat.sub(sub, txt)
    return ADDR_AFTER_LABEL.sub(_mask_addr_values, txt)


def content_lines(txt):
    """被比较的内容 = 掩码后的所有非空行"""
    return [l for l in mask(txt).splitlines() if l.strip()]


def field_of(line):
    """给差异行一个可读字段名（便于定点抽查）"""
    m = re.match(r"^([A-Z_]{3,})", line.strip())
    if m:
        return m.group(1)
    if "errcode:(" in line or "error code =" in line or "subErrType" in line:
        return "fault_errcode"
    return "fault_detail"


def repo_root(wt):
    """`wt` 可以是 worktree 根、probe 目录或任意子目录：git 操作一律锚到仓库根，
    而 `--path` 按**仓库相对**解析 ⇒ 从 probe 目录里跑同样的命令也能工作。"""
    r = subprocess.run(["git", "-C", wt, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else os.path.abspath(wt)


def git(wt, *args):
    return subprocess.run(["git", "-C", wt] + list(args), capture_output=True, text=True)


def show(wt, ref, path):
    r = git(wt, "show", f"{ref}:{path}")
    return r.stdout if r.returncode == 0 else None


def resolve_path(root, ref, path):
    """`--path` 先按仓库相对解析；解析不到再试"相对 probe 目录"的写法（`evidence` → `probe_v_align/evidence`）。
    两者都不匹配才算"子树写错"（⇒ 报 SKIPPED + rc=1，绝不静默给 0 项 + OK）。"""
    cands = [path]
    if not path.startswith("probe_v_align"):
        cands.append(os.path.join("probe_v_align", path))   # 允许简写 `--path=evidence`
    for cand in cands:
        if git(root, "ls-tree", "-r", "--name-only", ref, "--", cand).stdout.strip():
            return cand
    for cand in cands:
        if os.path.exists(os.path.join(root, cand)):
            return cand
    return None


def is_variant_log(path):
    base = os.path.basename(path)
    return path.endswith(".log") and (base.startswith("run_") or base.startswith("ildl_"))


def classify(wt, old_ref, new_ref, path=None):
    args = ["diff", "--name-status", "-M", old_ref, new_ref]
    if path:
        args += ["--", path]
    st = git(wt, *args).stdout
    added, modified, renamed, deleted = [], [], [], []
    for line in st.splitlines():
        f = line.split("\t")
        if f[0] == "A":
            added.append(f[1])
        elif f[0] == "M":
            modified.append(f[1])
        elif f[0].startswith("R"):
            renamed.append((f[1], f[2]))
        elif f[0] == "D":
            deleted.append(f[1])
    text_only, result_changed, other = [], [], []
    for p in modified:
        if is_variant_log(p):
            a, b = show(wt, old_ref, p), show(wt, new_ref, p)
            if a is None or b is None:
                other.append(p)
                continue
            la, lb = content_lines(a), content_lines(b)
            pairs = list(zip(la, lb))
            if len(la) != len(lb):
                pairs += [(x, "") for x in la[len(lb):]] + [("", y) for y in lb[len(la):]]
            diff_pairs = [(x, y) for x, y in pairs if x != y]
            if not diff_pairs:
                text_only.append(p)
                continue
            def is_wording(line):
                # `RESULT:` 行是结论措辞；与"该行只在一边存在（另一边为空）"配对时也算措辞
                return line.strip() == "" or line.strip().startswith("RESULT")
            content_diff = [(x, y) for x, y in diff_pairs if not (is_wording(x) and is_wording(y))]
            wording_diff = [(x, y) for x, y in diff_pairs if is_wording(x) and is_wording(y)]
            if not content_diff:
                text_only.append(p)   # 只有结论措辞变了
                continue
            fields = sorted({field_of(y) for x, y in content_diff} | {field_of(x) for x, y in content_diff})
            if wording_diff:
                fields.append("RESULT(措辞)")
            if len(la) != len(lb) and not fields:
                fields = ["行数变化"]
            result_changed.append((p, fields))
        else:
            other.append(p)
    return dict(added=added, modified=modified, modified_text=text_only, modified_result=result_changed,
                modified_other=other, renamed=renamed, deleted=deleted)


def report(wt, old_ref, new_ref, path=None):
    if git(wt, "rev-parse", "--verify", old_ref).returncode != 0 or \
       git(wt, "rev-parse", "--verify", new_ref).returncode != 0:
        print(f"RESULT: SKIPPED (ref 取不到：{old_ref} / {new_ref})")
        return 2
    if path:
        resolved = resolve_path(wt, new_ref, path)
        if resolved is None:
            print(f"RESULT: SKIPPED (子树不存在：{path} —— 没得比，不发合格证；"
                  f"本仓的正确口径是 `--path=probe_v_align/evidence`)")
            return 1
        path = resolved
    c = classify(wt, old_ref, new_ref, path)
    tot = len(c["added"]) + len(c["modified"]) + len(c["renamed"]) + len(c["deleted"])
    scope = f"（限定子树 {path}）" if path else "（整仓）"
    print(f"== 归档差异记账  {old_ref} → {new_ref} {scope}（repo={wt}）==")
    print(f"A 新增 {len(c['added'])} / M 修改 {len(c['modified'])} / R 重命名 {len(c['renamed'])} / "
          f"D 删除 {len(c['deleted'])}  （合计 {tot}）")
    print(f"  修改的 {len(c['modified'])} 个再分（变体 log 结果有差异 {len(c['modified_result'])} + "
          f"变体 log 仅易变字段 {len(c['modified_text'])} + 其它（dump/生成物/核对输出）{len(c['modified_other'])} "
          f"= {len(c['modified_result']) + len(c['modified_text']) + len(c['modified_other'])}）")
    for tag, items in (("A", c["added"]), ("D", c["deleted"])):
        for p in items:
            print(f"    {tag} {p}")
    for a, b in c["renamed"]:
        print(f"    R {a} -> {b}")
    print("  结果有差异的修改（逐条给字段）：")
    for p, fields in c["modified_result"]:
        print(f"    M {p}  变化字段={fields}")
    print("  其它（非变体 log）的修改：")
    for p in c["modified_other"]:
        print("    M", p)
    if tot == 0:
        print("RESULT: EMPTY (0 项 —— ref 与目标相同、或该子树无变更；"
              "这**不是**'比过通过'，故不发合格证)")
        return 2
    print(f"RESULT: OK (记账 {tot} 项 = A {len(c['added'])} + M {len(c['modified'])} + "
          f"R {len(c['renamed'])} + D {len(c['deleted'])}；"
          f"修改再分 {len(c['modified_result'])} 结果 + {len(c['modified_text'])} 仅易变 + "
          f"{len(c['modified_other'])} 其它)")
    return 0


# ---------------------------------------------------------------- selftest

SAMPLES = ["run_ls_a0_b0_n64.log", "run_redsum_a0_b0_n64.log", "run_lmbarout_a0_b0_n64.log",
           "run_ldbrc_a0_b0_n64.log", "run_ls_a1_b0_n64.log"]   # 末条是 errcode 340 的对齐故障样本

# (名字, 正则, 替换, 是否**必须**被报成"结果有差异")
INJECTIONS = [
    ("COMPARE window", r"(COMPARE window_bytes=)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 1), True),
    ("OUTCOME", r"(OUTCOME: )OK", r"\1WRONG", True),
    ("LAUNCH aclError", r"(LAUNCH aclError=)\d+", r"\g<1>507035", True),
    ("VARIANT window", r"(VARIANT .*?window=)(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 8), True),
    ("HEAD32", r"(HEAD32 )([0-9a-f]{8})", lambda m: m.group(1) + ("deadbeef" if m.group(2) != "deadbeef" else "00000000"), True),
    ("FAULT_CODE", r"(FAULT_CODE )\d+", r"\g<1>999999", True),
    ("DUMP_FNV", r"(DUMP_FNV 0x)([0-9a-f]{15})([0-9a-f])",
     lambda m: m.group(1) + m.group(2) + ("0" if m.group(3) != "0" else "1"), True),
    ("errcode 值改 1", r"(errcode:\()(\d+)(\))",
     lambda m: m.group(1) + str(int(m.group(2)) + 1) + m.group(3), True),
    ("error code 值改 1", r"(error code = )(\d+)",
     lambda m: m.group(1) + str(int(m.group(2)) + 1), True),
    ("errorStr", r"(errorStr: )[^\[]*", r"\g<1>REPLACED", True),
    # 地址语境行上的**非标签位置**指纹：先就位（这一步本身也是真变化），下一步只改它的值 ——
    # 两版都在同一行、只差指纹 ⇒ 若掩码是"按行"的就会把它吃掉（评审实测过的那种形态）
    ("地址语境行上植入指纹 token", r"(error code = \d+, )", r"\g<1>fp=0x0123456789abcdef, ", True),
    ("地址语境行上的指纹值改 1", r"(fp=0x)0123456789abcdef", lambda m: m.group(1) + "fedcba9876543210", True),
    ("纯环境噪声 serial", r"(serial number is )(\d+)", lambda m: m.group(1) + str(max(0, int(m.group(2)) - 1000)), False),
    ("纯环境噪声 PID/时间戳", r"(EZ9999\[PID: )\d+", r"\g<1>1", False),
]


def _src_logs_dir(wt):
    """样本 log 目录：`wt` 既可以是 worktree 根，也可以是 probe_v_align 本体"""
    for cand in (os.path.join(wt, "probe_v_align", "evidence", "logs"),
                 os.path.join(wt, "evidence", "logs")):
        if os.path.isdir(cand):
            return cand
    return None


def _mk_repo(wt, tmp):
    repo = os.path.join(tmp, "repo")
    logs = os.path.join(repo, "probe_v_align", "evidence", "logs")
    os.makedirs(logs)
    src_dir = _src_logs_dir(wt)
    for s in SAMPLES:
        src = os.path.join(src_dir, s) if src_dir else ""
        if src and os.path.isfile(src):
            shutil.copy(src, os.path.join(logs, s))
    env = ["-c", "user.name=selftest", "-c", "user.email=selftest@local"]
    git(repo, "init", "-q")
    git(repo, *env, "add", "-A")
    git(repo, *env, "commit", "-q", "-m", "base")
    git(repo, "rev-parse", "HEAD")
    base = git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, base, logs, env


def _commit_mutation(repo, logs, env, name, pat, sub):
    for fn in os.listdir(logs):
        p = os.path.join(logs, fn)
        t = open(p, errors="replace").read()
        new = re.sub(pat, sub, t, count=1)
        if new != t:
            open(p, "w").write(new)
            git(repo, *env, "add", "-A")
            git(repo, *env, "commit", "-q", "-m", f"inject {name}")
            return git(repo, "rev-parse", "HEAD").stdout.strip(), fn
    return None, None


def selftest(wt):
    tmp = tempfile.mkdtemp(prefix="m57_diff_selftest_")
    repo, base, logs, env = _mk_repo(wt, tmp)
    if not os.listdir(logs):
        print("SELFTEST: SKIPPED (找不到样本 log)")
        return 2
    ok = True
    print(f"SELFTEST：丢弃式仓库 {repo}（样本 {len(os.listdir(logs))} 个 log）")
    # ① 无注入：必须 0 项
    git(repo, *env, "add", "-A")
    git(repo, *env, "commit", "-q", "--allow-empty", "-m", "noop")
    noop = git(repo, "rev-parse", "HEAD").stdout.strip()
    c = classify(repo, base, noop)
    if sum(len(v) for v in c.values()) == 0:
        print("SELFTEST ①（同一内容再提交）：0 项 ✓")
    else:
        print("SELFTEST ①：**竟报出差异** ✗"); ok = False
    # ② 逐项注入
    for name, pat, sub, must_report in INJECTIONS:
        ref, fn = _commit_mutation(repo, logs, env, name, pat, sub)
        if ref is None:
            print(f"SELFTEST ②[{name}]：SKIPPED（样本里没有可注入的位置）")
            continue
        c = classify(repo, ref + "~1", ref)
        got = bool(c["modified_result"])
        mark = "✓" if got == must_report else "✗"
        if got != must_report:
            ok = False
        kind = "必须被报出" if must_report else "必须被掩成'仅易变'"
        fields = c["modified_result"][0][1] if got else []
        print(f"SELFTEST ②[{name}]（{kind}）：{'被报出' if got else '被掩掉'} {fields} {mark}")
    # ③ 子树写错必须可区分
    rc_bad = report(repo, base, noop, path="probe_v_align/evidence_WRONG")
    print(f"SELFTEST ③（子树写错）：rc={rc_bad} " + ("✓" if rc_bad == 1 else "✗"))
    if rc_bad != 1:
        ok = False
    # ④ 空差异必须不是 OK
    rc_empty = report(repo, base, base, path="probe_v_align/evidence/logs")
    print(f"SELFTEST ④（空差异）：rc={rc_empty} " + ("✓" if rc_empty == 2 else "✗"))
    if rc_empty != 2:
        ok = False
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST: OK（注入矩阵全部符合预期 + 两个状态控制）" if ok else "SELFTEST: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        sys.exit(2)
    wt = repo_root(os.path.abspath(argv[0]))   # 允许从 probe 目录里跑：git 操作锚到仓库根
    ref = argv[1]
    if "--selftest" in argv:
        sys.exit(selftest(wt))
    rest = [a for a in argv[2:] if not a.startswith("--")]
    new_ref = rest[0] if rest else "HEAD"
    path = None
    for a in argv:
        if a.startswith("--path="):
            path = a.split("=", 1)[1]
    sys.exit(report(wt, ref, new_ref, path))
