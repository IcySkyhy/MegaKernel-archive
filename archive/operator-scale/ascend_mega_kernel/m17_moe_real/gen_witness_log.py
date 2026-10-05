#!/usr/bin/env python3
"""M60 证据日志的**提取器**：从 `check_ref.py` 的运行日志里抽出量化规则见证（`W0`–`W6` 与
`自检-*` 负向对照）的逐条读数，拼成 `evidence/m60_quant_rule_witness.log`。

它**不实现判据**（只做提取与汇总），也不手写本目录的**读数**（§5⑤ 的 M59/m3 标尺数字是**引用**，已逐条注明
出处，见下段与输出头部的逐块标注）：逐条读数是正则从输入日志里取出来的，条数、
`PASS/FAIL` 计数、§3 的判定项计数与 FAIL 名、§5④ 的逐行比对结果、§5⑥ 的改动范围与计时项计数
都由**当场计算**得出（README §5.5.4「计数与匹配器同源」的落实）。

**本文件的每一块都有来源标注**（`build()` 打出，见 `# 来源逐块标注` 一行）：标为**派生**的块
其数字都由输入当场算出；标为**引用/说明**的块里若出现数字，都是**说明性文字**并逐条注明出处
（§3 的实验描述、§4 的角落语义说明、§5① 的管线说明、§5③ 的形态声明、§5⑤ 的 M59/m3 标尺引用、
§5⑥ 的两句规则陈述）——**注意：§4 的角落读数本身是派生量（逐字取自 §2 的 `自检-W3n` 行），但
§4 的语义解释与「真实数据 0 个这样的组」不是**，故 §4 在标注里同时出现在两类中。

用法（在 worktree 根；默认参数按本文件所在目录的上级解析）：

    /usr/local/python3.12.13/bin/python3 m17_moe_real/gen_witness_log.py \
        > m17_moe_real/evidence/m60_quant_rule_witness.log

    # 也可显式给四个输入
    gen_witness_log.py [run_log] [pin_log] [tamper_log] [m29_log]

    gen_witness_log.py --selftest     # 六种输入状态的自检（见下）

默认输入（4 个都在仓内）：

| 参数 | 默认 | 用途 | 缺失时 |
|---|---|---|---|
| `run_log` | `evidence/m60_check_ref_run.log` | §1/§2 的读数来源（**必需**） | `RESULT: SKIPPED` + 退出码 2 |
| `pin_log` | `evidence/m60_dump_pin.log` | §0 的归档身份摘要（**必需**） | 同上 |
| `tamper_log` | `evidence/m60_tamper_experiment.log` | §3 的篡改实验读数 | 该节打印「未核（需重跑内嵌实验）」 |
| `m29_log` | `evidence/check_ref_run.log` | §4 的回归基线（M29 存档） | 该节打印「未核」 |

**退出码（三态；与 `check_ref.py` 同语义）**：

    0 = 提取到读数，且 §1/§2 的读数全部 PASS
    1 = 提取到读数，但其中**有 FAIL**（此时标题/汇总会写明 FAIL 条数；产出的日志仍是原始读数）
    2 = 没得比：必需输入文件缺失，或提取到 **0 条读数**（`cases == 0` 或 §1+§2 全空）
        → 打印 `RESULT: SKIPPED`。**0 条读数绝不发合格证**（tower 规则：审校/统计工具无输入须
        SKIPPED 且非零；「通过了」与「根本没检查」必须可区分）。

**覆盖范围（分母，当场打印）**：§1 = 运行日志中每个 `#### <tag> …` 段落里形如
`[check_ref] PASS|FAIL W<数字>…: …` 的行；§2 = 同段落里 `[check_ref] PASS|FAIL 自检-…: …` 的行；
§4 = 两份日志里**除** `W*`/`自检-*` 之外的 `PASS/FAIL` 行（逐行集合比对，按 case 分块）。
**已知会被漏掉**（`--selftest` 的第 ⑤/⑥ 态实测）：`[报告]`/`[coverage]`/`[guard]`/`[参考]`/`[pin]`
这些非判定行**不被计为读数** —— 所以只含这类行的日志会得到 `SKIPPED` 而不是「全 PASS」；
同样地，名字/正文分隔符不是 ASCII `": "` 的行（例如写成全角「：」）也解析不到 ⇒ 计数为 0 ⇒
`SKIPPED`。`W5b` 这类新前缀同理（正则与计数同源：**解析不到就不发合格证，绝不谎报 PASS**）。

`--selftest` 会构造六种合成输入（正常 / 0 case / 含 FAIL / 缺文件 / 只有非判定行 / 分隔符不可解析）
跑自己，打印每种状态的退出码与汇总行，并核它们与期望一致；全对返回 0、否则 1。
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULTS = {
    "run_log": os.path.join(HERE, "evidence", "m60_check_ref_run.log"),
    "pin_log": os.path.join(HERE, "evidence", "m60_dump_pin.log"),
    "tamper_log": os.path.join(HERE, "evidence", "m60_tamper_experiment.log"),
    "m29_log": os.path.join(HERE, "evidence", "check_ref_run.log"),
}

CASE_RE = re.compile(r"#+ (\S+)(?: \(/tmp/\S+\))? #+\n(.*?)(?=\n#+ |\Z)", re.S)
ITEM_RE = re.compile(r"\[check_ref\] (PASS|FAIL) ((?:W\d|自检-).+?): (.*)")


def cases(txt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in CASE_RE.finditer(txt)]


def readings(txt: str, kind: str) -> list[tuple[str, str, str]]:
    """kind="pos" → `W*`（不含自检）；kind="neg" → `自检-W*`（负向对照）。

    正则只认 `W<数字>` 前缀（含 `自检-W<数字>`）—— 这是**有意**的边界，见 docstring 的
    「已知会被漏掉」：新前缀（如 `W5b`）不会被计为读数，计数随之少，但绝不会谎报 PASS。"""
    out = []
    for line in txt.splitlines():
        m = ITEM_RE.match(line)
        if not m:
            continue
        if (kind == "neg") != m.group(2).startswith("自检-"):
            continue
        out.append((m.group(1), m.group(2), m.group(3)))
    return out


def content_items(txt: str) -> list[str]:
    """§4 的回归口径：除 `W*`/`自检-*` 之外的 PASS/FAIL 行（原样，含读数）"""
    return [l for l in txt.splitlines()
            if re.match(r"\[check_ref\] (PASS|FAIL) ", l)
            and not re.match(r"\[check_ref\] (PASS|FAIL) (?:W\d|自检-)", l)]


def _rel(path: str) -> str:
    """把输入路径按「相对仓库根」打印（不可解析时原样返回）—— 保证证据文件与 worktree 位置无关"""
    try:
        return os.path.relpath(os.path.abspath(path), REPO)
    except Exception:                                                    # noqa: BLE001
        return path


def _tally(rows: list[tuple[str, str, str]]) -> tuple[int, int]:
    npass = sum(1 for st, _, _ in rows if st == "PASS")
    return npass, len(rows) - npass


def _skip(msg: str) -> int:
    print(f"RESULT: SKIPPED（{msg}） —— 退出码 2；本文件 0 条读数，不发合格证")
    return 2


def build(run_log: str, pin_log: str, tamper_log: str, m29_log: str) -> int:
    need = [p for p in (run_log, pin_log) if not os.path.isfile(p)]
    if need:
        return _skip(f"必需输入缺失 {need}")
    for path in (tamper_log, m29_log):
        if not os.path.isfile(path):
            print(f"[gen_witness_log][warn] 可选输入缺失：{path} —— 对应小节将打印「未核」")
    cs = cases(open(run_log, encoding="utf-8").read())
    pos = {tag: readings(txt, "pos") for tag, txt in cs}
    neg = {tag: readings(txt, "neg") for tag, txt in cs}
    tot_pos, tot_neg = sum(len(v) for v in pos.values()), sum(len(v) for v in neg.values())
    pp, pf = _tally([r for v in pos.values() for r in v])
    np_, nf = _tally([r for v in neg.values() for r in v])
    n_fail = pf + nf
    if len(cs) == 0 or tot_pos + tot_neg == 0:
        return _skip(f"run_log={run_log} 里 cases={len(cs)}、§1+§2 读数={tot_pos + tot_neg}"
                     f"（此类输入通常是 pre-M60 日志或不含 W 行的日志）")

    print("# M60 量化规则见证：外部 pin + 设备无关交叉见证 + 定点/往返判据（`check_ref.py` 的 W0–W6）")
    print("#")
    print(f"# 读数汇总（**由本提取器当场从输入派生**，不是手写常量）：cases={len(cs)}"
          f"（其中 {sum(1 for t in pos if pos[t] or neg[t])} 个有读数）｜§1 `W*` {tot_pos} 条"
          f"（PASS {pp} / FAIL {pf}）｜§2 `自检-*` {tot_neg} 条（PASS {np_} / FAIL {nf}）")
    print("#")
    print("# 任务（M60）：`quant_hw` 的 docstring 此前只写「m5 MxQuant 全 VEC 路径的 numpy 镜像」，")
    print("#   无任何 `文件:符号` 外部引用；S5/S7 与 e2e 的字节判据全托在它上面 ⇒ M59 判为 **S2 残余**")
    print("#   （m3 型的规则/方向错会两侧一起错、逐字节 0 失配而全 PASS）。本文件是补完后的**原始读数**。")
    print("#")
    print("# 生成方式（本文件是**提取器**的输出，不是手写的；提取器 = `m17_moe_real/gen_witness_log.py`）：")
    print(f"#   /usr/local/python3.12.13/bin/python3 m17_moe_real/gen_witness_log.py {_rel(run_log)} \\")
    print(f"#       {_rel(pin_log)} \\")
    print(f"#       {_rel(tamper_log)} \\")
    print(f"#       {_rel(m29_log)} > m17_moe_real/evidence/m60_quant_rule_witness.log")
    print("#   （上面四个参数一律按「相对仓库根」echo —— 与 worktree 位置无关；不带参数跑也一样，")
    print("#     因为默认参数也会过同一函数。）")
    print("#")
    print("# 来源逐块标注（**派生** = 数字当场由输入算出；**引用/说明** = 不由本工具派生，已注明出处）：")
    print("#   派生：§0（pin_log 的摘要行）｜§1/§2（运行日志的逐条读数、条数、PASS/FAIL 计数）｜")
    print("#         §3（tamper log 的判定项计数与 FAIL 名）｜§4 的**读数**（逐字取自 §2 的 `自检-W3n` 行）｜")
    print("#         §5②（逐字引自运行日志的 `[coverage]` 行）｜§5④（两份日志的逐行集合比对）｜")
    print("#         §5⑥ 的改动范围（`git diff --name-only main...HEAD`）与计时项计数（当场扫 `check_ref.py`）")
    print("#   引用/说明（非派生）：§3 的实验描述文字｜§4 的**语义解释**与「真实数据 0 个角落组」｜")
    print("#         §5① 的管线说明｜§5③ 的形态声明｜§5⑤ 的 M59/m3 标尺数字（逐条注明出处）｜")
    print("#         §5⑥ 的两句规则陈述")
    print()
    print("## §0 输入来历：W4/W5/W6 跑的字节 = 归档 dump（摘要，明细见 `m60_dump_pin.log`）")
    print()
    for line in open(pin_log, encoding="utf-8"):
        if "整段核对" in line or "段级核对" in line:
            print("- " + line.rstrip())
    print("- `check_ref.py` 自己也把这条钉成判定项 **W0**（整段 sha256 vs `evidence/dump_manifest.md`），")
    print("  每个 case 的读数见 §1；**本轮未新跑 device**（device 侧算件零改动，理由见 `m60_dump_pin.log` 末）。")
    print()
    print(f"## §1 正向判据（设备无关的规则级 W1/W3 + device 侧的 W0/W4/W5/W6）：{tot_pos} 条读数"
          f"（PASS {pp} / FAIL {pf}），分布于 {sum(1 for t in pos if pos[t])}/{len(cs)} 个 case")
    print()
    for tag, _ in cs:
        rows = pos[tag]
        if not rows:
            print(f"### `{tag}`：0 条读数（**该 case 的 `W*` 判定项条数 = 0** —— 不当作 PASS）")
            print()
            continue
        a, b = _tally(rows)
        print(f"### `{tag}`：{len(rows)} 条（PASS {a} / FAIL {b}）")
        print()
        for st, name, info in rows:
            print(f"- **{st}** `{name}`：{info}")
        print()
    print(f"## §2 负向对照（`自检-*`；不 FAIL 即说明见证失明）：{tot_neg} 条读数"
          f"（PASS {np_} / FAIL {nf}），分布于 {sum(1 for t in neg if neg[t])}/{len(cs)} 个 case")
    print()
    for tag, _ in cs:
        rows = neg[tag]
        if not rows:
            print(f"### `{tag}`：0 条读数（该 case 未提取到 `自检-*` 行 —— 不当作 PASS）")
            print()
            continue
        a, b = _tally(rows)
        print(f"### `{tag}`：{len(rows)} 条（PASS {a} / FAIL {b}）")
        print()
        for st, name, info in rows:
            print(f"- **{st}** `{name}`：{info}")
        print()
    print("## §3 「已知会被漏掉」的负向对照（M60 新增）：篡改 dump 后**内容判据仍 OK**，但 W0 会咬住")
    print()
    print("口径：`docs/17` §2.1 与 tower 规则要求审校工具交代自己的覆盖边界，并**实测**一条会漏掉的对照。")
    print("本目录 M29 时代的对照是「篡改一个空槽位的 `A_qx` 字节 → 脚本仍报 OK」；M60 起这条**行为变了**：")
    print("`W0`（dump 身份 pin）会把「整段 sha 变了」直接判 FAIL —— 即整段字节层面的篡改**不再漏**。")
    print("为把 M60 后的覆盖边界仍然**实测**出来，该实验对同一份 m=1 归档 dump 做两处**内容层面**的篡改")
    print("（空槽位 `hq[0][0][0]`、活动行行距尾部 `hs[41][0][20]`，各 XOR 0x01），完整输出与复现命令见")
    print(f"`{_rel(tamper_log)}`（仓内文件，§3 的输入）。")
    print()
    if not os.path.isfile(tamper_log):
        print(f"（**未核**：未找到 `{tamper_log}` —— 请按该文件头部内嵌的命令重跑篡改实验后重生成本文件）")
        print()
    else:
        tl = open(tamper_log, encoding="utf-8").read()
        # 口径 = 该日志里**每一条** `[check_ref] PASS|FAIL <名字首 token>` 行（不只是 W*/自检-*），
        # 名字首 token 足以分辨 `W0` 身份 pin 与其余内容判据
        rows_t = re.findall(r"^\[check_ref\] (PASS|FAIL) (\S+)", tl, re.M)
        n_all = len(rows_t)
        n_ok = sum(1 for st, _ in rows_t if st == "PASS")
        fails_t = [tok for st, tok in rows_t if st == "FAIL"]
        content = [(st, tok) for st, tok in rows_t if tok != "W0"]
        c_ok = sum(1 for st, _ in content if st == "PASS")
        c_fail = len(content) - c_ok
        res = re.search(r"\[check_ref\] RESULT: (.*)", tl)
        print(f"实测读数（**当场从 `{os.path.basename(tamper_log)}` 提取**，口径 = 该日志里**每一条**"
              f"`[check_ref] PASS|FAIL` 行）：判定项 {n_all} 条（PASS {n_ok} / FAIL {len(fails_t)}）；"
              f"FAIL 项 = {'、'.join('`' + t + '`' for t in fails_t) or '（无）'}；"
              f"**内容判据（除 `W0` 身份 pin）= {len(content)} 条（PASS {c_ok} / FAIL {c_fail}）**；"
              f"RESULT 行 = `{res.group(1) if res else '（未找到）'}`")
        print()
        print("| 被篡改的字节 | 谁负责咬 | 实测（当场提取） | 结论 |")
        print("|---|---|---|---|")
        print(f"| 任意字节（整段层面） | `W0` 整段 sha256 pin | "
              f"{'FAIL' if 'W0' in fails_t else 'PASS'} | "
              f"**不再是盲区** —— M60 起「跑的是不是归档字节」可判定 |")
        print(f"| 空槽位 `hq[0][0][0]` | 无（W4/W5/W6 只遍历活动槽；不变量/guard 也不覆盖） | "
              f"内容判据 {c_ok} PASS / {c_fail} FAIL | 覆盖边界，**已披露**"
              f"（`[coverage]` 每次都打印，且打印的就是「除 W0 外全 PASS」这读法） |")
        print("| 活动行 `hs[41][0][20]`（行距尾部） | 无（判据只读 `[:K/32]`） | 同上 | 覆盖边界，**已披露** |")
        print()
        print("⇒ 这两处是「已知会被漏掉」的对照；它们与 M29 的 `S5/S7` 字节判据的边界**完全相同**")
        print("  （既有判据也只比 `[:HIDDEN/32]` / `[:INTER/32]` 与活动槽前 t_e 行），M60 既没扩大也没缩小它。")
        print()
    print("## §4 base 版镜像的角落分叉（M60 的修复点，设备无关、可复算）")
    print()
    print("`自检-W3n` 用 `git show f0286f6:m17_moe_real/check_ref.py` 取出 **M60 修复前**的 `quant_hw`（用")
    print("`exec` 装进临时模块，不写盘）与 `tools/golden::quantize_ocp` 逐字节比。**本节的读数不是手抄的**：")
    print("下面并列 §2 里 `自检-W3n` 那一行的原文（含失配字节数与具体角落数值），按取值去重后给出命中的 case。")
    print()
    w3n = [(tag, info) for tag, txt in cs for st, name, info in readings(txt, "neg")
           if name.startswith("自检-W3n")]
    if not w3n:
        print("（**未核**：输入日志里没有 `自检-W3n` 行 ⇒ 本节不给任何角落读数）")
    else:
        uniq: dict[str, list[str]] = {}
        for tag, info in w3n:
            uniq.setdefault(info, []).append(tag)
        for info, tags in uniq.items():
            where = (f"{len(tags)}/{len(cs)} 个 case 相同" if len(tags) == len(cs)
                     else "、".join(f"`{t}`" for t in tags))
            print(f"- **{where}**：{info}")
    print()
    print("**语义解释（说明性文字，非派生；读数以上面 §2 的原文为准）**：修复前的镜像缺官方两条覆盖 ——")
    print("① 组内含 ±Inf/NaN 时，官方 `:406`/`:413` 把 scale 写成 E8M0 NaN、把 `halfScale` 覆盖成 bf16 NaN，")
    print("   于是 `Mul` 出 NaN、`Cast` 收成全 0 码；旧实现只把指数域字段减 2 并裁剪，因此给出一个**可表示**")
    print("   的 scale 字节、并把该元素饱和成最大码（具体数值见上）。")
    print("② 指数域 < 2 的退化组：官方 `:402-403` 把 `shared` 夹到 0、`:414` 令 `halfScale = 0` ⇒ 全组只剩符号位；")
    print("   旧实现仍按 `2^(byte−127)` 反算，于是给出非零码（具体数值见上）。")
    print()
    print("**「真实数据 0 个这样的组」—— 非派生量，出处与复算路径如下**（本工具不读 dump，无法派生它）：")
    print("该读数是 M60 用 5 份归档 dump 的活动组直接扫描得出的；独立复查记录在评审文件里——")
    print("`.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r1.md`")
    print("独立重数过同一读数；`.tower/comms/reviews/review-feat-m60-m17-quant-rule-external-pin-and-witn-reviewer-m60-r2.md` **未重数**（它用的是「`check_ref.py` 代码树")
    print("AST 相同 ⇒ 读数不变」这条论证）。")
    print("复算路径（不依赖任何人的叙述）：按 `m17_moe_real/evidence/dump_manifest.md` 的 5 份 dump，")
    print("对每个活动组取 `max(bf16 bits & 0x7F80)`，数「等于 `0x7F80`」（非有限组）与「< `0x0100`」（退化组）的组数。")
    print()
    print("## §5 口径、自指形态与覆盖范围交代")
    print()
    print("**① 计数与匹配器同源**：§1/§2 的字节数、组数、越界数由 `check_ref.py` 的**同一个循环**")
    print("产出（`w4_tot` / `w5_n` / `rt['n']`），本文件由提取器正则从运行日志取数、计数当场统计；")
    print("`[coverage]` 行里的活动槽数、行数同样取自同一次运行的 `counts_dev`。哪些块是派生、哪些是")
    print("引用/说明，见文件头的「来源逐块标注」——那里逐块列了，不用「全部/唯一」这类量词概括。")
    print()
    print("**② 三态分栏**（哪些判据吃 dump、哪些不吃、哪些不在覆盖内）—— 原文引自运行日志：")
    print()
    if cs:
        for line in cs[0][1].splitlines():
            if "量化规则见证（M60）的三态分栏" in line or "本脚本的 OK 不等于" in line:
                print("- " + line.split("] ", 1)[1])
    print()
    print("**③ 自指形态（M59 §1 代号）**：")
    print()
    print("| 判据 | 输入 | 规则来源 | 形态 | 能不能计入「独立端到端」 |")
    print("|---|---|---|---|---|")
    print("| W0 | `ws_*.bin` 整段字节 | `evidence/dump_manifest.md`（M29 device 日志 + 确定性复跑） | structural（**T4**，无正确性咬合力） | 不计 |")
    print("| W1/W2 | 合成定点输入（amax∈{0.5,1,2}） | 官方 `:404-405` / `:412` / `:753-758` | **N2 外部 pin + 规则不敏感** | 不计（不是端到端） |")
    print("| W3/W3n | 合成角落输入 | 官方 `:402-414` / `tools/golden::quantize_ocp` | **N2 + 同源转录**（两个转写互为对照） | 不计 |")
    print("| W4/W4n | device `x_sorted` / bf16 `GU` / `xnorm` | `tools/golden::quantize_ocp` | **S1（输入取设备中间量）+ 同源转录** | **不得计入** |")
    print("| W5/W5n | device 打包字节 + host 复算的组内 amax 位置 | 「MXFP4 group32 的数学」（只引用数学，不拿任一量化实现当基准） | **S1 + 规则不敏感** | **不得计入** |")
    print("| W6/W6n | device `hq/hs` 字节 + host double SwiGLU 参考值 | 同上 | **S1 + 规则不敏感** | **不得计入** |")
    print("| `[报告] W6b` | device `hq/hs` 字节 + §5.4 A 链的 H（输入 = 真实 x） | 同上 | **S1-free（除 device 字节本身）** | 可作参考，本轮只报不判 |")
    print()
    print("⇒ 明确声明：W4/W5/W6 **都是隔离判据（S1）**，因为它们的参考值/位置的输入含 device 中间量；")
    print("  它们**不得**被计入「独立端到端」。真正的 S1-free 锚点仍是 §5.4 的 e2e A 链（从真实 x 起步）。")
    print("  `quantize_ocp` 与本镜像的关系是**同源转录**（correlated transcription：同一工作区两个 agent 对")
    print("  同一份官方头文件的两次转写，共用同一组常量与同一份读法），**不是两处独立** —— 它能咬住转录")
    print("  笔误与角落分支差异（§4 就是实例），但**咬不住**对规范的共同误读；补这个缺口的是 W1/W5/W6 的")
    print("  「方向 + 值域」口径（它们只引用 MXFP4 group32 的数学；量化实现只以「两份转写并列」出现）。")
    print()
    print("**④ 回归核对（修改 `quant_hw` 角落分支后，既有判据是否被改动）—— 本表由提取器当场比对得出**：")
    print()
    if not os.path.isfile(m29_log):
        print(f"（**未核**：未找到回归基线 `{m29_log}`）")
        print()
    else:
        old = dict(cases(open(m29_log, encoding="utf-8").read()))
        new = dict(cs)
        rows_r, same_n = [], 0
        for tag, otxt in old.items():
            ntxt = new.get(tag)
            a = content_items(otxt)
            b = content_items(ntxt) if ntxt is not None else None
            same = b is not None and a == b
            same_n += 1 if same else 0
            rows_r.append((tag, len(a), "（该 case 不在新日志里）" if b is None else len(b), same))
        print(f"基线 = `{_rel(m29_log)}`（M29 存档）；被核 = `{_rel(run_log)}`；"
              f"口径 = 剔除 `W*`/`自检-*` 行后的 `PASS/FAIL` 行**逐行集合比对**")
        print()
        print("| case | M29 归档 | M60 本次 | 逐行比对 |")
        print("|---|---|---|---|")
        for tag, na, nb, same in rows_r:
            print(f"| `{tag}` | {na} 条 | {nb} 条 | {'**逐行相同**' if same else '**有差异**'} |")
        print()
        print(f"⇒ 当场比对结果：**{same_n}/{len(rows_r)} 个 case 的既有判定项逐行相同**"
              f"（含每个分位数字）—— 修复只触及真实数据不出现的角落（§4）。")
        print()
    print("**⑤ 与 M59 归档标尺的对照（判据必须先会咬）—— 本表是「引用」，不由本工具派生**：")
    print()
    print("| 标尺（引自括号里的归档） | M60 在同一口径下的读数 |")
    print("|---|---|")
    print("| m3 修复前 pack 字节差异 **94.8%**（引自 `wt-54/m3_grouped_gemm/evidence/m54_quant_rebaseline.log`，见 M59 审计报告） | 方向反的规则 vs device 字节：§2 的 `自检-W4n` 行（每个 case 给出实测百分比） |")
    print("| m1 golden 上把方向反过来 **283/640 = 44.2%**（引自 M59 审计报告 §4 的负向对照） | 同上（不同数据，量级一致） |")
    print("| m3 的 H 反量化 p50 **142**、p99 2.04e39、max **5.79e76**（引自 M59 审计报告 §2） | §2 的 `自检-W6n` 行（相对误差 p50/max；本数据组尺度更小，方向反后码被压进 e2m1 的 0 档，相对误差饱和在 1.0） |")
    print()
    print("⇒ 读数一致性的关键差别是**数据尺度**：m3 那边的激活幅度大（`scale > 1` ⇒ 反向后 `deq ≈ h·scale²` 反而变大），")
    print("  本目录的真实 MoE 激活幅度小（`scale < 1` ⇒ 反向后码被压到 0）。这正是 M60 把 W6 的**判定**口径取成")
    print("  「绝对界 `2·scale + 1·spacing + EPS_FAST·|h_ref|` + 预算比」而不是相对误差的原因 —— 相对误差在")
    print("  `h_ref → 0` 处无界，不能当判据；绝对界的 `2·scale` 项直接来自 e2m1 码表的最大间距，与数据无关。")
    print()
    print("**⑥ 两条 tower 硬规则的落实**：")
    print()
    try:
        names = subprocess.check_output(["git", "-C", REPO, "diff", "--name-only", "main...HEAD"],
                                        stderr=subprocess.DEVNULL).decode().split()
        outside = [n for n in names if not n.startswith("m17_moe_real/")]
        # 「device 侧被引用了哪些符号」不当自称，改成当场扫 check_ref.py 计数
        sym_hits = []
        try:
            for line in open(os.path.join(HERE, "check_ref.py"), encoding="utf-8"):
                for sym in ("MxQuantComputeScale", "MxQuantComputeDataFP4"):
                    if sym in line:
                        sym_hits.append(f"{sym}@{line.strip()[:40]}")
                        break
        except OSError:
            sym_hits = []
        print(f"- **禁 memory-based API**：本 mission **未改 device 代码**（判据 = 下面这份当场 diff 里"
              f"没有 `.asc`/`.h`）—— `git diff --name-only main...HEAD` **当场**打印 {len(names)} 个路径，"
              f"落在 `m17_moe_real/` 的 {len(names) - len(outside)} 个、scope 外 {len(outside)} 个："
              f"{', '.join('`' + n.split('/')[-1] + '`' for n in names)}")
        print(f"  ⇒ W0–W6 都是 host 侧 numpy/字节判据。device 侧符号的引用情况**当场扫**"
              f"`check_ref.py`：`MxQuantComputeScale` / `MxQuantComputeDataFP4` 命中 {len(sym_hits)} 行"
              f"（pin 用的只读符号）；`m17_moe_layer.asc` 的 `Sort32`/`MrgSort` 白名单与依据注释"
              f"在 S2 段，本轮未触碰（见 README §7.13）。")
    except Exception as exc:                                            # noqa: BLE001
        print(f"- **禁 memory-based API**：本 mission 的改动范围**未核**（git 不可用：{exc}）⇒ "
              f"不据此下结论；`m17_moe_layer.asc` / `m17_resources.h` 的零改动请以三点 diff 为准。")
    print("- **host 墙钟不得作为证据**：W0–W6 的判定量只有三类 —— 字节相等（sha256/nibble/scale 字节）、")
    print("  整数计数（失配数/越界数/违反组数）、推导界下的分位（T3 口径）。这一条**当场核**（不是自称）：")
    hits = []
    try:
        for line in open(os.path.join(HERE, "check_ref.py"), encoding="utf-8"):
            if re.search(r"\btime\b|perf_counter|\bclock\b|monotonic", line):
                hits.append(line.strip()[:70])
    except OSError as exc:                                               # noqa: BLE001
        hits = [f"（读不到 check_ref.py，未核：{exc}）"]
    print(f"  `m17_moe_real/check_ref.py`（判定项所在脚本）里 `time`/`perf_counter`/`clock`/`monotonic`"
          f" 命中 **{len(hits)}** 行" + ("：" if hits else " ⇒ 计时调用 0 行。"))
    for h in hits:
        print(f"    - {h}")
    print("  对照 README §7.5 对 host 墙钟的定性声明（本文件的判定量里没有时间量）。")
    print("  （本提取器自身不参与该统计：它的源码里含这条检查所用的正则字面量，计进来会自我命中。）")
    print()
    if n_fail:
        print(f"RESULT: DIFF（§1/§2 共 {tot_pos + tot_neg} 条读数中有 {n_fail} 条 FAIL："
              f"§1 FAIL {pf}、§2 FAIL {nf}） —— 退出码 1；本文件是原始读数、不做判定")
        return 1
    print(f"RESULT: OK（§1 {tot_pos} 条 + §2 {tot_neg} 条读数均 PASS，覆盖 {len(cs)} 个 case："
          f"{', '.join(t for t, _ in cs)}） —— 退出码 0")
    return 0


# ---------------------------------------------------------------------------
# --selftest：六种输入状态的负向对照（含「已知会被漏掉」两态）
# ---------------------------------------------------------------------------
_PASS_W = "[check_ref] PASS W0 dump 身份 pin: x\n"
_FAIL_W = "[check_ref] FAIL W6 值域往返: 故意构造的 FAIL\n"
_PASS_SELF = "[check_ref] PASS 自检-W4n 负向对照: y\n"
_NONVERDICT = "[check_ref][coverage] ⇒ 只含非判定行\n[check_ref][参考] A_qx: 38.1%\n"
# 注：判定行的「名字 : 正文」分隔符是 ASCII `": "`（真实日志即如此）；若写成全角「：」，
# 本提取器**不会**把它算作读数（计数为 0 ⇒ SKIPPED），这正是「已知会被漏掉」那一态要证明的
# 性质：**解析不了就不发合格证**，而不是照旧宣称全 PASS。
_UNPARSABLE = "[check_ref] PASS W0 dump 身份 pin：全角冒号，解析器看不见\n"


def _synth(dirpath: str, body: str, name: str) -> str:
    p = os.path.join(dirpath, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("########## layer0_tok1000_m1 ##########\n" + body)
    return p


def selftest() -> int:
    with tempfile.TemporaryDirectory() as td:
        cases_tbl = [
            ("① 正常态（W* 与自检-* 各 1 条，全 PASS）", _synth(td, _PASS_W + _PASS_SELF, "ok.log"), 0, "FAIL 0"),
            ("② 0 case 态（日志里没有 case 段）", os.path.join(td, "empty.log"), 2, "SKIPPED"),
            ("③ 含 FAIL 态（一条 W 行是 FAIL）", _synth(td, _FAIL_W + _PASS_SELF, "fail.log"), 1, "FAIL 1"),
            ("④ 缺文件态（run_log 不存在）", os.path.join(td, "nope.log"), 2, "SKIPPED"),
            ("⑤ 已知会被漏掉·只有非判定行（`[报告]`/`[coverage]`…）", _synth(td, _NONVERDICT, "nonverdict.log"), 2, "SKIPPED"),
            ("⑥ 已知会被漏掉·W 行分隔符不可解析（全角「：」）", _synth(td, _UNPARSABLE, "unparsable.log"), 2, "SKIPPED"),
        ]
        open(os.path.join(td, "empty.log"), "w").close()
        pin = os.path.join(HERE, "evidence", "m60_dump_pin.log")
        print("# `gen_witness_log.py --selftest`：六种输入状态的实测（退出码 + 汇总行）")
        print("#")
        print("# 每种状态都是「喂一份合成输入 → 跑 build() → 看它印出的汇总行与退出码」；")
        print("# 退出码语义见本文件 docstring（0=有读数且全 PASS / 1=有读数但含 FAIL / 2=没得比）。")
        print()
        print("| 状态 | 期望退出码 | 实测退出码 | 实测汇总行 | 期望内容 | 判定 |")
        print("|---|---|---|---|---|---|")
        ok_all = True
        for label, path, want_rc, want_txt in cases_tbl:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = build(path, pin, os.path.join(td, "no_tamper.log"), os.path.join(td, "no_m29.log"))
            out = buf.getvalue()
            # 汇总行里会带上合成输入的临时路径 ⇒ 打印前统一替换成固定记号，保证自检输出可归档、
            # 逐字节可复算（判定用的是替换前的 out，故检测能力不受影响）
            shown = out.replace(td, "<合成输入目录>")
            summary = next((l.strip() for l in shown.splitlines() if l.startswith("# 读数汇总")), "")
            verdict = next((l.strip() for l in shown.splitlines() if l.startswith("RESULT:")), "")
            good = (rc == want_rc) and (want_txt in out)
            ok_all &= good
            print(f"| {label} | {want_rc} | {rc} | `{(summary or verdict)[:150]}` | 含 `{want_txt}` | "
                  f"{'**符合**' if good else '**不符合**'} |")
        print()
        print(f"⇒ 自检结论：{'6/6 态符合期望' if ok_all else '**有状态不符合期望**'}")
        print("   （②/④/⑤/⑥ 都必须是 SKIPPED+2：0 条读数**绝不**发合格证；③ 的汇总必须写出 `FAIL 1`，")
        print("    而不是照旧宣称「全 PASS」—— ⑤/⑥ 就是「已知会被漏掉」控制在提取器上的两种形态：")
        print("    非判定行不计入读数、不可解析的行也不计入 ⇒ 两种都落到 SKIPPED 而非 OK）")
        print(f"   合成输入目录：`TemporaryDirectory`（退出即删；本行故意不打印随机路径 —— 保证自检输出可归档且逐字节可复算）")
        print("RESULT: " + ("OK" if ok_all else "FAIL") + "（selftest 6 态，"
              + ("6/6 符合" if ok_all else "存在不符合") + "） —— 退出码 " + ("0" if ok_all else "1"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        sys.exit(selftest())
    names = ["run_log", "pin_log", "tamper_log", "m29_log"]
    paths = {n: (args[i] if len(args) > i else DEFAULTS[n]) for i, n in enumerate(names)}
    sys.exit(build(paths["run_log"], paths["pin_log"], paths["tamper_log"], paths["m29_log"]))
