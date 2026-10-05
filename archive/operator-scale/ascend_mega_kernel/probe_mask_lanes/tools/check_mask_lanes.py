#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_mask_lanes.py —— M94 的**独立判据脚本**（设备读数 vs 位编码模型）

## 为什么判据在这个脚本里，而不在探针二进制里
探针（`probe_mask_lanes.asc`）**只报事实**：arena 里哪些字节与哨兵 0xA5 不同（`WRANGES`）、
整段 FNV、launch 返回码。**期望值全部在本文件里**，且模型的来源是官方文档的位编码表
（`/workspace/asc-devkit/docs/zh/api/SIMD-API/c_api/reg_compute/reg_mask/asc_create_mask.md`
的「位宽模式说明」），不是探针自己的代码 ⇒ 判据与被判物不共享代码路径（docs/17 §1.2 反自指）。

## 模型（doc 的位编码表）
`CreateMask<MD, PAT>` 生成的 mask 是 256-bit 谓词寄存器，**元素↔bit 的映射随位宽模式变**：
  * b8 模式：1 bit / 元素，共 **256** 元素
  * b16 模式：2 bit / 元素，共 **128** 元素
  * b32 模式：4 bit / 元素，共 **64** 元素
pattern（PAT）按「mask 自己的位宽」的元素下标定义（ALL/VLk/H/Q/M3/M4）。
一次带 mask 的 `StoreAlign` 只覆盖**一个矢量寄存器 = VL = 256B**（doc：`asc_storealign`
「单次搬出量为 VL（256 字节）」），落盘元素数 = 256B / sizeof(dtype)。
⇒ 对 store 的每个元素 e，取它在 mask 位图里的那一组 bit：**该组全 1 ⇒ 元素参与**（`all1` 假设）
或**该组有任一 1 ⇒ 元素参与**（`any1` 假设）。两种假设都算，由读数裁决（本脚本报两者各自的
匹配情况；两种假设重合时标注）。

## 判据（每条都非空洞：期望量必须 > 0，且负向对照必须被检出）
  L 组（lane-own / lane-cross / lane-enc / masktype）——**门**是两条：
    C-L1 单次落盘的所有被写字节必须落在两个落盘窗口内 ⇒ 单次落盘 <= 1 VL (=256B)；
    C-L2 本位宽 ALL 的落盘必须把窗口**写满**（256B）。
    **报告项（不是门）**：观测集合与 doc 先验位扩展模型的逐字节比对（见 §5.1；交叉位宽下 10/62 不同）。
  C 组（row / 行距，四条）：
    C1 行槽包含：所有被写字节 ⊆ ∪(区,行) 的行槽（宽度 = **行距**）⇒ 抓「落盘超出本行槽」；
    C2 不越 VL：所有被写字节 ⊆ ∪[槽起点, 槽起点+VL) ⇒ 抓「单次落盘 > 1 VL」；
    C3 写入者归属：窗口内每个字节的值必须等于该窗口的 (区,行) 常数图案值（0x11 + 区*16 + 行）
       ⇒ 抓「别的行的 store 写进来了」这类**自愈后仍可指认**的越界；
    C4 覆盖：每行的 VL 窗口必须被写满 ⇒ 抓「写得比一个 VL 少」。
  负向对照（`row_nc_*`）：**C1** 的越界字节数必须 > 0（检测器必须能咬住越界；对照的造法是把
    **行距**缩到 < VL，mask 不变）；
  稀疏正对照（`row_sparse4_p512`）：C1/C2 越界必须 = 0 且窗口之间的空洞必须保持哨兵（并集判据非空洞）。

用法：
  python3 tools/check_mask_lanes.py <probe_dir>            # 读 evidence/（归档模式）
  python3 tools/check_mask_lanes.py <probe_dir> --partial  # 读 evidence_partial/
退出码：0 = **全部门成立且常量绑定通过**；1 = 有门不成立（含常量绑定不一致，列表里给出首个反例）；
        2 = 没得比（日志/归档缺失）。**所有门都反映到 rc**（按 rc 接入的调用方不会静默通过）。
"""

import os
import re
import sys

SENTINEL = 0xA5
ARENA = 12288
IDS_OFF = 0
WS_OFF = 4096
PAT_LANE_A = 8192
PAT_LANE_B = 8448
PAT_SLOT = 256
LANE_WIN_B = [0, 1024]          # 探针里两个 lane 图案落盘窗口的基址
VL = 256

SD_SIZE = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2, "f32": 4, "s32": 4, "u32": 4}
MW_ELEMS = {"m8": (1, 256), "m16": (2, 128), "m32": (4, 64)}   # 位宽模式 -> (bit/元素, 元素数)


def pat_A(i):
    return (0x5A ^ ((i + 1) & 0xFF)) & 0xFF


def pat_B(i):
    return (0xC3 ^ ((i + 2) & 0xFF)) & 0xFF


LANE_PAT = {"A": (PAT_LANE_A, pat_A), "B": (PAT_LANE_B, pat_B)}


def mask_active_elements(mw, pat):
    """按 doc 的位编码表返回 (bit/元素, 元素数, 该位宽下被置位的元素集合, 位图集合)。"""
    bits_per, nelem = MW_ELEMS[mw]
    if pat == "all":
        elems = set(range(nelem))
    elif pat.startswith("vl"):
        k = int(pat[2:])
        elems = set(range(min(k, nelem)))
    elif pat == "h":
        elems = set(range(nelem // 2))
    elif pat == "q":
        elems = set(range(nelem // 4))
    elif pat == "m3":
        elems = {e for e in range(nelem) if e % 3 == 0}
    elif pat == "m4":
        elems = {e for e in range(nelem) if e % 4 == 0}
    else:
        raise ValueError("unknown pattern %r" % pat)
    bitmap = set()
    for e in elems:
        for b in range(bits_per):
            bitmap.add(e * bits_per + b)
    return bits_per, nelem, elems, bitmap


def expected_store_bytes(sd, mw, pat, mode):
    """doc 先验模型：一次单寄存器带 mask 落盘的**应写字节集合**（相对窗口基址）。

    doc 的位编码表把「一个被选元素占几位」按 **mask 的位宽**给出（b8=1/b16=2/b32=4 bit），
    但一个 store 元素在谓词寄存器里占几位由 **store 的位宽**决定（b32 store 的元素占 4 bit）。
    ⇒ 本函数：mask 的位图按 doc 表展开（bitmap），store 元素 e 取 `[e*sizeof(sd), (e+1)*sizeof(sd))`
    这一段 bit；`mode='all1'` 要求整段全 1，`mode='any1'` 要求任一位为 1。
    """
    sz = SD_SIZE[sd]
    nelem_store = VL // sz
    _, _, _, bitmap = mask_active_elements(mw, pat)
    out = set()
    for e in range(nelem_store):
        grp = [e * sz + b for b in range(sz)]
        hit = all(b in bitmap for b in grp) if mode == "all1" else any(b in bitmap for b in grp)
        if hit:
            for j in range(sz):
                out.add(e * sz + j)
    return out


def visible(expected, base, pattern_byte):
    """把「应写字节」变成「能与哨兵区分开的字节」：图案值恰等于哨兵的字节是**盲点**。"""
    return {o for o in expected if pattern_byte((o - base) % 256) != SENTINEL}


# ---------------------------------------------------------------- 日志解析
RX_VARIANT = re.compile(r"^VARIANT name=(\S+) kind=(\d+) group=(\S+) sd=(\S+) md=(\S+) pat=(\S+) rows=(\d+) "
                        r"pitchI=(\d+) pitchW=(\d+) im=(\d+)")
RX_LAUNCH = re.compile(r"^LAUNCH aclError=(-?\d+)")
RX_RANGES = re.compile(r"^WRANGES (.*)$")
RX_NZ = re.compile(r"^NZ bytes_written=(\d+) first_off=(\d+) last_off=(\d+)")
RX_FNV = re.compile(r"^FNV (0x[0-9a-f]+)")


def parse_ranges(txt):
    if txt.strip() == "(none)":
        return set()
    out = set()
    for part in txt.split(","):
        a, b = part.split("-")
        out.update(range(int(a), int(b)))
    return out


class Variant(object):
    def __init__(self, log):
        self.op = None
        self.kind = None
        self.group = None
        self.sd = None
        self.md = None
        self.pat = None
        self.rows = None
        self.pitchI = None
        self.pitchW = None
        self.im = None
        self.acl_error = None
        self.written = None
        self.fnv = None
        self.fault = None
        for line in log.splitlines():
            m = RX_VARIANT.match(line)
            if m:
                (self.op, self.kind, self.group, self.sd, self.md, self.pat, self.rows, self.pitchI, self.pitchW,
                 self.im) = (m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5), m.group(6),
                             int(m.group(7)), int(m.group(8)), int(m.group(9)), int(m.group(10)))
                continue
            m = RX_LAUNCH.match(line)
            if m:
                self.acl_error = int(m.group(1))
                continue
            m = RX_RANGES.match(line)
            if m:
                self.written = parse_ranges(m.group(1))
                continue
            m = RX_FNV.match(line)
            if m:
                self.fnv = m.group(1)
                continue
            if line.startswith("OUTCOME: "):
                self.fault = line.split(": ", 1)[1]
        self.rep = 0

    @property
    def ok(self):
        return self.acl_error == 0 and self.written is not None


def load_logs(logdir):
    """返回 {op: [Variant(...), ...]}（同 op 多次运行按 rep 顺序）。"""
    out = {}
    for name in sorted(os.listdir(logdir)):
        if not (name.startswith("run_") and name.endswith(".log")):
            continue
        base = name[4:-4]
        m = re.match(r"^(.*)_r(\d+)$", base)
        op, rep = (m.group(1), int(m.group(2))) if m else (base, 1)
        with open(os.path.join(logdir, name), "r", encoding="utf-8", errors="replace") as f:
            v = Variant(f.read())
        if v.op is None:
            v.op = op
        v.rep = rep
        out.setdefault(op, []).append(v)
    for k in out:
        out[k].sort(key=lambda x: x.rep)
    return out


# ---------------------------------------------------------------- 判据
def check_lane(v, row, pattern_byte, tag):
    """L 组三条：
       C-L1（**门**）单次落盘的字节必须落在两个窗口内 ⇒ 单次落盘 <= 1 VL = 256B（本 mission 的核心判据）；
       C-L2（**门**）本位宽 ALL 的落盘必须把窗口**写满**（256B）⇒ 每种 dtype 的 ALL 落盘量；
       C-L3（**报告**，非门）doc 位映射先验的逐字节比对 —— 交叉位宽下与设备实测存在分歧（见 README §5.1），
              这是对官方文档位扩展方式的**证伪**读数，逐条列出，但不作为本 mission 结论的门。
    """
    # ---- C-L1：窗口包含（≤ 1 VL）----
    in_win = set()
    for base in LANE_WIN_B:
        in_win.update(range(base, base + VL))
    out_of_win = sorted(v.written - in_win)

    # ---- C-L2：本位宽 ALL 必须写满窗口 ----
    window_full = True
    missing_n = 0
    if v.group == "lane-own":
        for base in LANE_WIN_B:
            _, pfn = LANE_PAT["A" if base == 0 else "B"]
            for o in range(base, base + VL):
                if pfn((o - base) % 256) == SENTINEL:
                    continue                      # 盲点：写了也与哨兵同值（不参与判定）
                if o not in v.written:
                    missing_n += 1
                    window_full = False

    # ---- C-L3：doc 先验模型（all1 / any1 两种组语义）----
    exp_all1, exp_any1 = set(), set()
    exp_written = set()
    for base in LANE_WIN_B:
        pat_kind = "A" if base == 0 else "B"
        _, pfn = LANE_PAT[pat_kind]
        exp_written.update(o + base for o in expected_store_bytes(v.sd, v.md, v.pat, "all1"))
        for mode, acc in (("all1", exp_all1), ("any1", exp_any1)):
            acc.update(o + base for o in visible(expected_store_bytes(v.sd, v.md, v.pat, mode), base, pfn))
    doc_match = (v.written == exp_any1) or (v.written == exp_all1)

    row["expected_n"] = len(exp_all1)
    row["observed_n"] = len(v.written)
    row["expected_written_n"] = len(exp_written)
    row["blind_n"] = len(exp_written) - len(exp_all1)
    row["outside_n"] = len(out_of_win)
    row["outside_first"] = out_of_win[0] if out_of_win else None
    row["docprior"] = "match" if doc_match else "diverge"

    gates_ok = (len(out_of_win) == 0) and window_full
    row["verdict"] = "OK" if gates_ok else "WRONG"
    row["detail"] = ("C-L1 窗口外=%d（首 %s，单次落盘须 <= VL=%d）C-L2 未写满=%d | doc先验=%s"
                     "（模型应写 %d B / 可见 %d B，实测可见 %d B）"
                     % (len(out_of_win), out_of_win[0] if out_of_win else "-", VL, missing_n,
                        row["docprior"], len(exp_written), len(exp_all1), len(v.written)))
    return gates_ok



def row_slots(rows, pitchI, pitchW):
    """行槽（宽度 = 行距）与「一次落盘窗口」（宽度 = VL=256B）两种几何。
    返回 (slots, vols, val_of)：slots = 行槽区间（含归属值），vols = 每行一次落盘的窗口。"""
    slots, vols = [], []
    for r in range(rows):
        val = 0x11 + 0 * 16 + r
        s = IDS_OFF + r * pitchI
        slots.append((s, s + pitchI, val))
        vols.append((s, s + VL, val))
    for r in range(rows):
        val = 0x11 + 1 * 16 + r
        s = WS_OFF + r * pitchW
        slots.append((s, s + pitchW, val))
        vols.append((s, s + VL, val))
    return slots, vols


def in_any(regs, o):
    for (s, e, _) in regs:
        if s <= o < e:
            return (s, e, regs is not None)
    return None


def check_row(v, dump, row):
    """行距组的四条判据：
       C1 行槽包含：被写字节 ⊆ ∪ 行槽（宽度 = 行距）—— 抓「落盘超出本行槽」
       C2 不越过 VL：被写字节 ⊆ ∪ [槽起点, 槽起点+VL) —— 抓「单次落盘 > 1 VL」
       C3 写入者归属：槽内字节值必须 = 该 (区,行) 的常数图案值 —— 抓「别的行写进来了」
       C4 覆盖：每行的 VL 窗口必须被写满 —— 抓「写得比一个 VL 少」
    """
    rows, pi, pw = v.rows, v.pitchI, v.pitchW
    slots, vols = row_slots(rows, pi, pw)

    def outside(regs):
        bad = [o for o in sorted(v.written) if not any(s <= o < e for (s, e, _) in regs)]
        return bad

    c1 = outside(slots)
    c2 = outside(vols)
    c3 = []
    if dump is not None:
        for (s, e, val) in vols:
            for o in range(s, e):
                if dump[o] != SENTINEL and dump[o] != val:
                    c3.append((o, dump[o], val))
    c4 = []
    for (s, e, _) in vols:
        miss = [o for o in range(s, e) if not (dump is not None and dump[o] != SENTINEL)]
        if miss:
            c4.append((s, miss[0], len(miss)))

    row["expected_n"] = sum(e - s for (s, e, _) in vols)
    row["observed_n"] = len(v.written)
    row["c1_outside_n"] = len(c1)
    row["c1_first"] = c1[0] if c1 else None
    row["c2_outside_n"] = len(c2)
    row["c2_first"] = c2[0] if c2 else None
    row["c3_n"] = len(c3)
    row["c3_first"] = c3[0] if c3 else None
    row["c4_missing_rows"] = len(c4)
    row["c4_first"] = c4[0] if c4 else None
    row["hole_bytes"] = sum(1 for (s, e, _) in slots for _ in range(s, e)) - sum(e - s for (s, e, _) in vols)
    if v.op.startswith("row_nc_"):
        # 负向对照：行距 < VL ⇒ 落盘必然溢出本行槽 ⇒ C1 必须咬住
        ok = len(c1) > 0
        row["verdict"] = "OK" if ok else "WRONG"
        row["detail"] = ("负向对照（行距 %dB < VL %dB）：C1 检出越出本行槽 %d B，首坏偏移 %s"
                         "（≈ 落盘 %dB − 行距 %dB，末行）" % (pi, VL, len(c1), c1[0] if c1 else "-",
                                                             VL, pi))
        row["expect"] = "DETECT"
    else:
        ok = (len(c1) == 0 and len(c2) == 0 and len(c3) == 0 and len(c4) == 0)
        row["verdict"] = "OK" if ok else "WRONG"
        row["detail"] = ("C1 越行槽=%d（首 %s）C2 越 VL 窗=%d（首 %s）C3 归属错=%d（首 %s）C4 未写满行=%d（首 %s）"
                         % (len(c1), c1[0] if c1 else "-", len(c2), c2[0] if c2 else "-",
                            len(c3), (c3[0][0] if c3 else "-"), len(c4), (c4[0][0] if c4 else "-")))
        row["expect"] = "CLEAN"
    return row["verdict"] == "OK"


def check_constants_binding(probe):
    """**显式绑定**：本脚本的几何常量（ARENA/IDS_OFF/WS_OFF/图案偏移/sentinel/VL）必须与探针源码
    `probe_mask_lanes.asc` 里的同名常量逐字节一致 —— 否则判据用的几何与设备实际用的几何不是同一套
    （本队纪律：判据里引用偏移/常量必须显式限定命名空间，不得让两个文件里的同名裸常量静默各说各话）。
    返回 (ok, lines)。"""
    src = os.path.join(probe, "probe_mask_lanes.asc")
    want = {"ARENA": ARENA, "IDS_OFF": IDS_OFF, "WS_OFF": WS_OFF, "PAT_LANE_A": PAT_LANE_A,
            "PAT_LANE_B": PAT_LANE_B, "PAT_SLOT": PAT_SLOT}
    out = []
    if not os.path.isfile(src):
        return False, ["  源码缺失：%s" % src]
    txt = ""
    with open(src, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(65536)          # 分段读，避免一次性读大文件
            if not chunk:
                break
            txt += chunk
    ok = True
    for name, val in sorted(want.items()):
        m = re.search(r"^constexpr\s+\w+\s+%s\s*=\s*([0-9x]+)\s*;" % re.escape(name), txt, re.M)
        got = int(m.group(1), 0) if m else None
        good = (got == val)
        ok = ok and good
        out.append("  %-12s probe=%-8s checker=%-8s %s" % (name, got, val, "OK" if good else "**MISMATCH**"))
    ms = re.search(r"^constexpr\s+\w+\s+SENTINEL\s*=\s*(0x[0-9A-Fa-f]+)\s*;", txt, re.M)
    got_s = int(ms.group(1), 16) if ms else None
    ok = ok and (got_s == SENTINEL)
    out.append("  %-12s probe=%-8s checker=%-8s %s" % ("SENTINEL", hex(got_s) if got_s else None, hex(SENTINEL),
                                                       "OK" if got_s == SENTINEL else "**MISMATCH**"))
    out.append("  注：VL=256 不是探针里的常量（探针不按 VL 做判定）；判据侧的 VL 由**实测**支撑，")
    out.append("      即本脚本 C-L2「本位宽 ALL 必须把窗口写满 256B」——若 VL 不是 256，该判据必然变红。")
    return ok, out


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    probe = os.path.abspath(sys.argv[1])
    partial = "--partial" in sys.argv
    ev = os.path.join(probe, "evidence_partial" if partial else "evidence")
    logs = os.path.join(ev, "logs")
    dumps = os.path.join(ev, "dumps")
    if not os.path.isdir(logs):
        sys.stderr.write("FAIL: 没有 %s\n" % logs)
        return 2

    runs = load_logs(logs)
    if not runs:
        sys.stderr.write("FAIL: %s 下没有 run_*.log\n" % logs)
        return 2

    print("# M94 判据报告（设备读数 vs 位编码模型）")
    print("# 模型来源：asc-devkit 官方文档 doc:docs/zh/api/SIMD-API/c_api/reg_compute/reg_mask/"
          "asc_create_mask.md 的位宽模式表（b8=1bit/元素/256 元素；b16=2bit/128；b32=4bit/64）")
    print("#           + doc:docs/zh/api/SIMD-API/c_api/reg_compute/store/asc_storealign.md"
          "（单次搬出量 = VL = 256B）")
    print()
    print("== 常量绑定（判据的几何 vs 探针源码的几何；不得让同名裸常量静默各说各话）==")
    bind_ok, bind_lines = check_constants_binding(probe)
    for line in bind_lines:
        print(line)
    print("  绑定结果：%s" % ("PASS（判据几何 = 探针几何）" if bind_ok else "FAIL（几何不一致，下面的判据不可信）"))
    print()

    rows = []
    n_ok = n_wrong = n_skip = 0
    n_lane_visible_bytes = 0
    for op in sorted(runs):
        reps = runs[op]
        v0 = reps[0]
        r = {"op": op, "group": v0.group, "sd": v0.sd, "md": v0.md, "pat": v0.pat, "kind": v0.kind,
             "rep": len(reps), "fnv": [x.fnv for x in reps], "acl": [x.acl_error for x in reps]}
        if not all(x.ok for x in reps):
            r["verdict"] = "SKIP"
            r["detail"] = "有 rep 未取到读数（aclError=%s fault=%s）" % (v0.acl_error, v0.fault)
            n_skip += 1
            rows.append(r)
            continue
        # 稳定性：跨进程 rep 的 FNV 必须逐位一致
        r["stable"] = len(set(r["fnv"])) == 1
        if v0.kind == 0 or v0.kind == 1:      # lane / masktype：模型逐字节判
            dump = None
            if v0.kind == 1:
                # masktype 组：VARIANT 的 md 列记的是 tA/tB（哪个 C++ 类型），真正决定位宽模式的是
                # 「与 store 同位宽」—— 该组两列都是同宽不同型（half/bf16、int16/uint16、float/int32、
                # uint8/int8）⇒ 模型掩码位宽 = store 的位宽。
                v0.md = {"s8": "m8", "u8": "m8", "s16": "m16", "u16": "m16", "f16": "m16", "bf16": "m16",
                         "f32": "m32", "s32": "m32", "u32": "m32"}[v0.sd]
            ok = check_lane(v0, r, None, op)
            n_lane_visible_bytes += r["expected_n"]
        elif v0.kind == 2:                     # dual：报告 2×VL 事实（>VL 落盘的唯一形态）
            span = (max(v0.written) + 1 - min(v0.written)) if v0.written else 0
            r["verdict"] = "OK" if span == 2 * VL else "WRONG"
            r["expected_n"] = 2 * VL
            r["observed_n"] = len(v0.written)
            r["detail"] = "dual（intlv）落盘跨度 = %d B（模型 2×VL = %d B）⇒ >VL 的单次落盘**只有**这一形态" % (
                span, 2 * VL)
            r["expect"] = "REPORT"
        else:                                  # row
            cand = [os.path.join(dumps, "ml_%s_r%d.bin" % (op, v0.rep)), os.path.join(dumps, "ml_%s.bin" % op)]
            dump = None
            for dpath in cand:
                if os.path.isfile(dpath):
                    with open(dpath, "rb") as f:
                        dump = f.read()
                    break
            ok = check_row(v0, dump, r)
        if r["verdict"] == "OK":
            n_ok += 1
        else:
            n_wrong += 1
        rows.append(r)

    hdr = "%-20s %-11s %-6s %-5s %-5s %-5s %-7s %-8s %-9s %-5s %s"
    print(hdr % ("op", "group", "sd", "mask", "pat", "rep", "exp_n", "obs_n", "verdict", "stable", "detail"))
    for r in rows:
        md = r.get("md", "-")
        if r.get("kind") == 1:
            md = "tA" if md == "m8" else "tB"
        print(hdr % (r["op"], r.get("group", "-"), r.get("sd", "-"), md, r.get("pat", "-"), r["rep"],
                     r.get("expected_n", "-"), r.get("observed_n", "-"), r["verdict"],
                     ("y" if r.get("stable") else "n"), r["detail"]))

    # ---- 负向对照 / 非空洞守卫 ----
    print()
    print("== 守卫（判据不得空过）==")
    nc = [r for r in rows if r["op"].startswith("row_nc_")]
    print("  负向对照数 = %d（期望 > 0，且 C1 必须咬住越界）: %s" % (len(nc), ", ".join(
        "%s→C1=%s" % (r["op"], r.get("c1_outside_n")) for r in nc)))
    sp = [r for r in rows if r["op"].startswith("row_sparse")]
    print("  稀疏正对照（行距 > VL ⇒ 槽内空洞必须保持哨兵；空洞字节应 > 0，越界=0）: %s" % ", ".join(
        "%s→slot_hole_bytes=%s C1=%s C2=%s" % (r["op"], r.get("hole_bytes"), r.get("c1_outside_n"),
                                              r.get("c2_outside_n")) for r in sp))
    cov = [r for r in rows if r.get("group") == "row" and not r["op"].startswith("row_nc_")]
    print("  覆盖判据 C4（每行 VL 窗口必须写满）: %s" % ", ".join(
        "%s→未写满行=%s（应写 %s B）" % (r["op"], r.get("c4_missing_rows"), r.get("expected_n")) for r in cov))
    print("  L 组模型期望可见字节总数 = %d（期望 > 0；=0 说明在比 0 vs 0）" % n_lane_visible_bytes)
    div = [r for r in rows if r.get("docprior") == "diverge"]
    print("  doc 位映射先验：与实测 **不一致** 的变体数 = %d / %d（这些**不是**判据门：它们是"
          "「官方文档的交叉位宽扩展方式 ≠ 设备实测」的证伪读数，见 README §5.1；逐条已在 detail 列标出）"
          % (len(div), len([r for r in rows if "docprior" in r])))
    if div:
        print("    分歧清单（按组计数）: %s" % ", ".join(
            "%s×%d" % (g, sum(1 for r in div if r.get("group") == g))
            for g in sorted({r.get("group") for r in div})))
    print("  L 组模型盲点字节总数 = %d（可解释：图案值恰等于哨兵的字节，两种图案互补覆盖）"
          % sum(r.get("blind_n", 0) for r in rows if r.get("group", "").startswith("lane")))
    print("  可变性守卫：L 组模型期望可见字节数取值集合 = %s（须 > 1 个取值，否则判据不区分形态）"
          % sorted({r.get("expected_n") for r in rows if r.get("group", "").startswith("lane")}))

    print()
    print("== 汇总 ==")
    print("variants=%d ok=%d wrong=%d skip=%d" % (len(rows), n_ok, n_wrong, n_skip))
    bad = [r for r in rows if r["verdict"] == "WRONG"]
    if bad:
        print("首条不成立的判据：%s — %s" % (bad[0]["op"], bad[0]["detail"]))
    ok_all = (n_wrong == 0 and n_skip == 0 and bind_ok)
    print("RESULT: %s%s" % ("PASS" if ok_all else "FAIL",
                            "" if bind_ok else "（常量绑定未通过：几何不一致）"))
    # **退出码契约**：所有门（含常量绑定）都必须反映到 rc —— 否则按 rc 接入的调用方会
    # 在「判据几何 ≠ 探针几何」时**静默通过**（那正是这条守卫要防的失效模式）。
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
