#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m27_hc_prefill/check_ref.py —— M119（B5）hc **prefill 行分块（m-tile）段体**的独立 numpy 交叉校验

用法（先跑设备落盘）：
    flock -w 900 /tmp/npu0.lock bash -c 'cd /tmp/m27_out && M27_DUMP=1 <build>/m27_hc_prefill'
    python3.12 m27_hc_prefill/check_ref.py /tmp/m27_out

## 判据的来源（不自己造第二套数学）
参考 = `m20_hyperconn/check_ref.py::reference`（独立 numpy **float64**，逐句实现 hc 的伪码；
它按 `m` 参数化 —— `M15_HC_M` 只是 m15 那层 host 入口的环境变量），判据门限也直接复用它的
`judge()`（张量尺度归一化绝对误差 ≤1e-2 + 良态元素 bf16 ulp ≤2 且逐位一致率 ≥99%）。
⇒ 本脚本**不含**任何第二份 hc 数学，只做"把设备的 dump 摆成参考要的形状 + 逐张量判"。

## 退出码（三态，tower 规则）
    0 = 比过且通过（文案带实际比较条数）
    1 = 比过且有差异（列出判定项名）
    2 = 没得比 / 输入缺失（SKIPPED）

## 判什么
    A1（边界 #1）：hcp(=H') / blk / injw / rstd  vs  独立参考
    A2（边界 #2，**层内 handoff**）：输入 = 设备自己的一档 dump（hcp1 / ij_oh / attn_out），
       输出 hcp2 / blk2 / injw2 / rstd2 vs 独立参考（两个边界解耦，便于归因）
    guard（不计入判定项）：
      · 非空洞性：输入与输出都不是常量 / 全零，且 H' ≠ H（mode MIX 档反过来判"HCP 未被写"）；
      · **arena 阻塞布局的差分见证**（只在整块 arena 落盘的档）：
        (a) 末 `d_safe` 块的 arena 阻塞 blk/rstd == 参考（arena 里**确实**是段体的产物）；
        (b) 更早块的 arena 阻塞 blk ≠ 参考（"更晚的块会覆盖"真的发生 ⇒ 循环后的重定位**承担负载**）；
      · `ij_oh` 的独立复核（host 抽的那份 vs 按布局从 arena 自己抽的，逐字节）。

## 明确不判的
    · mode 0（MIX）：H' 不物化（设备侧未写 HCP 区），hcp 项 SKIP 并单列；A2 整段 SKIP
      （mode MIX 的 H' 为空 ⇒ 链上第二边界没有真实输入）；
    · OH 的 padding 列 [324,336)、injw 槽的 [1,8)（无效位）—— 与 m20 同口径不比较；
    · 本段的跨核 flag 用量语义（在飞 vs 累计）：那由设备档的真实读数说话，不由本脚本判。
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20   # noqa: E402  （独立参考 + 判据门限 + 三态工具）

HC = 4
HID = 2560
HYPER = HC * HID
LOWRANK = 320
INJ_N = 4
INJW_SLOT = 8
OH_W = 336
MODE_MIX, MODE_COMBINE_MIX, MODE_FINAL_MIX, MODE_COMBINE_ONLY = 0, 1, 2, 3
MODE_NAME = {0: "MIX", 1: "COMBINE_MIX", 2: "FINAL_MIX", 3: "COMBINE_ONLY"}

JUDGE_REL = [0]          # 本脚本自己的相对误差判据（injw / rstd）
SKIP = [0]
GUARD = [0]
GUARD_FAILS = []
REL_FAILS = []
NOTES = []


def rd(dump, name, dtype):
    """读 <name>.bin（形状取自 <name>.bin.meta 的第 2..n-1 列）。缺文件 = 没得比（rc=2）。"""
    path = os.path.join(dump, name + ".bin")
    if not os.path.exists(path):
        raise m20.MissingInput(path)
    shape = None
    meta = path + ".meta"
    if os.path.exists(meta):
        with open(meta) as f:
            parts = f.read().split()
        if len(parts) >= 3:
            shape = tuple(int(x) for x in parts[1:-1])
    a = np.fromfile(path, dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def read_kv(path):
    """读 `key=value` 文本。**一行里可以有多个 key=value**（布局文件把短项并列写在一行，
    例如 `m=33 mode=1 mt=8 tiles=5 arena_dumped=1`）⇒ 用正则逐对抽，不能按 `=` split 一次。"""
    kv = {}
    with open(path) as f:
        text = f.read()
    for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+)", text):
        kv[k] = v
    return kv


def read_cases(dump):
    """逐 case 读 `m27_case_<名>.txt`（**不用**全局清单：那种"每次调用截断"的写法会让分批/补跑互相抹掉
    元数据 —— M119 实测栽过一次）。行格式：`case <名> <key> <val> <key> <val> ...`（一行可以有多个对）。"""
    files = sorted(f for f in os.listdir(dump) if f.startswith("m27_case_") and f.endswith(".txt"))
    if not files:
        # **旧格式回退**（只读）：元数据格式在 10-04 从"全局 `m27_cases.txt`"改成"逐 case 一份"。
        # 早先批次留下的 dump 只有旧格式 ⇒ 这里回退读它，让那份 dump 仍然**可被判**（复审/复算友好）。
        legacy = os.path.join(dump, "m27_cases.txt")
        if os.path.exists(legacy):
            print("[chk] 注记：只找到旧格式 %s（逐 case 元数据缺失）⇒ 按旧格式回退解析" % legacy)
            files = [os.path.basename(legacy)]
        else:
            raise m20.MissingInput(os.path.join(dump, "m27_case_*.txt"))
    out = []
    for fn in files:
        with open(os.path.join(dump, fn)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if p[0] != "case" or len(p) < 3:
                    continue
                d = dict(zip(p[2::2], p[3::2]))
                d["case"] = p[1]
                out.append(d)
    return out


def distinct_frac(a):
    flat = np.asarray(a).reshape(-1)
    return float(len(np.unique(flat)) / flat.size) if flat.size else 0.0


def nuniq(a):
    """取值个数（非空洞性用；bf16 的取值域天然有限 ⇒ 判"够不够多样"，不判比例）。"""
    return int(len(np.unique(np.asarray(a).reshape(-1))))


def guard(name, ok, extra=""):
    GUARD[0] += 1
    if not ok:
        GUARD_FAILS.append(name)
    print("[chk] guard %-38s %s %s" % (name, "OK" if ok else "FAIL", extra))
    return ok


def judge_bf16(name, got_u16, exp_f64):
    return m20.judge(name, got_u16, exp_f64, {})    # 计数与 FAILS 由 m20 同源产出


def judge_rel(name, got, exp, tol):
    """fp32 张量的相对误差判据（injw / rstd；与 m20 check_ref.py 的 B 段同口径）。"""
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    exp = np.asarray(exp, dtype=np.float64).reshape(-1)
    rel = float(np.max(np.abs(got - exp) / np.maximum(np.abs(exp), 1e-30)))
    JUDGE_REL[0] += 1
    ok = rel <= tol
    if not ok:
        REL_FAILS.append(name)
    print("[chk] 判定 %-24s maxRel=%.3e (tol %.0e) n=%d  %s" % (name, rel, tol, got.size,
                                                               "PASS" if ok else "FAIL"))
    return ok


def blocked_plane(raw, base, row_bytes, m, mt, ts, rows=None):
    """把 arena 里的**阻塞**平面按行抽成 (rows, row_bytes) 的字节矩阵（向量化）。"""
    rows = m if rows is None else rows
    r = np.arange(rows, dtype=np.int64)
    idx = (r // mt) * ts + base + (r % mt) * row_bytes
    cols = np.arange(row_bytes, dtype=np.int64)
    return raw[idx[:, None] + cols[None, :]]


def region_windows(lay, mt):
    """块内**所有会被写满的区**的 (起始偏移, 内容字节数) —— 块 d 的写窗 = d*ts + 这两个量。

    内容字节数 = mt × 该区的**行字节**（`blk`/`rstd` 等平铺区的行字节不是 HYPER 行宽）。
    """
    return [
        (lay["off_hcp"], mt * lay["row_hp"]),
        (lay["off_xn"], mt * lay["row_hp"]),
        (lay["off_rstd"], mt * lay["row_rstd"]),
        (lay["off_injw"], mt * lay["row_injw"]),
        (lay["off_oh"], mt * lay["oh_w"] * 2),
        (lay["off_ls"], mt * lay["lowrank"] * 2),
        (lay["off_gate"], mt * lay["up_n"] * 2),
        (lay["off_blk"], mt * lay["row_hid"]),
    ]


def min_d(off, nbytes, wins, ts):
    """「块 t 的这个区**最早**被哪个更晚的块覆盖」的间隔 d（= m15_hc_prefill.h 的 `CoveredByLater`）。

    必须在**全部**写窗上取最小：只看 HCP 窗会漏掉更早的覆盖者 —— M119 实测就栽在这：
    rstd 区被块 t+16 的 HCP 窗覆盖，但**块 t+8 的 XN 窗更早覆盖它**
    （m=257 档：真正被覆盖的是块 0..24，与 d=8 一致；只看 HCP 会算出 d=16、误判块 17..24 是安全的）。
    返回 None = 没有任何更晚的块覆盖它（该区在 arena 里持久）。
    """
    best = None
    for (win_off, win_len) in wins:
        for d in range(1, 65):
            st = d * ts + win_off
            if st <= off and off + nbytes <= st + win_len:
                best = d if best is None else min(best, d)
                break
    return best


def judge_boundary(dump, tag, m, mode, wtag, ij_file, hin_file, bo_file, skip_hcp, reloc_mask=15):
    """`*_file` 是**完整的文件名前缀**（不含 .bin）：边界 #1 的 ij 是它自己的输入平面，
    边界 #2 的 ij 是**边界 #1 产出的平铺 ij 面**（`<case>_a1_ij_handoff`）—— 见 m27 的 PROLOGUE。"""
    hin = rd(dump, hin_file, np.uint16)
    bo = rd(dump, bo_file, np.uint16)
    ij = rd(dump, ij_file, np.uint16)
    wdown = rd(dump, "w_%s_down" % wtag, np.uint16)
    winj = rd(dump, "w_%s_inj" % wtag, np.uint16)
    wup = rd(dump, "w_%s_up" % wtag, np.uint16)
    norm = rd(dump, "w_%s_norm" % wtag, np.uint16).reshape(-1)
    dev = {
        "hcp": rd(dump, "%s_hcp" % tag, np.uint16),
        "blk": rd(dump, "%s_blk" % tag, np.uint16),
        "injw": rd(dump, "%s_injw" % tag, np.float32),
        "rstd": rd(dump, "%s_rstd" % tag, np.float32),
    }
    print("\n== %s：端到端（设备 vs 独立 numpy float64 参考 = m20_hyperconn/check_ref.py）==" % tag)
    print("   m=%u mode=%u(%s) 权重档=%s ij=%s(%u 列) bo=%s" %
          (m, mode, MODE_NAME.get(mode, "?"), wtag, ij_file, ij.shape[1], bo_file))
    ref = m20.reference(m, mode, hin, bo, ij, wdown, winj, wup, norm)
    # 哪些张量在**这一档 mode** 下本来就该被写出（由 donor 的段序决定；与 m20 check_ref.py 同口径）
    # `want` = "**这一档本该产出**"（由 mode 决定）× "**重定位目标里有它**"（由 reloc_mask 决定）：
    # 调用方把某个面传 nullptr = 明确不要它 ⇒ 判据侧记 SKIP（而不是把"没搬"误判成红）。
    # 这正是 P2-1 那个配置（只传 ijFlat）的判据口径。
    want = {
        "hcp": (mode != MODE_MIX),               # MIX 不跑 W0/combine ⇒ H' 不物化
        "injw": (mode != MODE_MIX) and ((reloc_mask & 0x2) != 0),   # injw 由 W0 产出 → 由重定位搬出
        "blk": (mode != MODE_COMBINE_ONLY) and ((reloc_mask & 0x1) != 0),   # 同上（blk 由 S6 产出）
        "rstd": (mode != MODE_COMBINE_ONLY) and ((reloc_mask & 0x4) != 0),  # 同上（rstd 由 S2 产出）
    }
    want_ij = (mode not in (MODE_MIX, MODE_COMBINE_ONLY)) and ((reloc_mask & 0x8) != 0)
    if not want["hcp"]:
        SKIP[0] += 1
        print("[chk] 判定 %-24s SKIPPED（mode MIX 不物化 H'，设备侧亦未写 HCP 区）" % (tag + ".hcp"))
    else:
        judge_bf16("%s.hcp (H')" % tag, dev["hcp"][:m], ref["hc"])
    if want["blk"]:
        judge_bf16("%s.blk" % tag, dev["blk"][:m], ref["blk"])
    else:
        SKIP[0] += 1
        why = "COMBINE_ONLY 不跑 S4–S6 ⇒ blk 不产出" if mode == MODE_COMBINE_ONLY \
            else "reloc_mask 未把 blk 当重定位目标（调用方传 nullptr）⇒ 平铺面按约定不写"
        print("[chk] 判定 %-24s SKIPPED（%s）" % (tag + ".blk", why))
    if want["injw"]:
        # injw：fp32，按 (token,stream) 取 32B 槽的首元素（槽的 [1,8) 是无效位，不比较）
        ijt = dev["injw"].reshape(-1, INJW_SLOT)
        judge_rel("%s.injw (fp32)" % tag, ijt[: m * HC, 0].astype(np.float64), ref["injw"].reshape(-1), 1e-6)
    else:
        SKIP[0] += 1
        why = "mode MIX 不跑 W0 ⇒ injw 不产出" if mode == MODE_MIX \
            else "reloc_mask 未把 injw 当重定位目标（调用方传 nullptr）"
        print("[chk] 判定 %-24s SKIPPED（%s）" % (tag + ".injw", why))
    if want["rstd"]:
        # rstd：fp32，(m, HC) 行主序（token 主、stream 次；与 donor 的 g = mi*HC+s 一致）
        judge_rel("%s.rstd (fp32)" % tag, dev["rstd"][:m].astype(np.float64), ref["rstd"], 1e-4)
    else:
        SKIP[0] += 1
        why = "COMBINE_ONLY 不跑 S2 ⇒ rstd 不产出" if mode == MODE_COMBINE_ONLY \
            else "reloc_mask 未把 rstd 当重定位目标（调用方传 nullptr）"
        print("[chk] 判定 %-24s SKIPPED（%s）" % (tag + ".rstd", why))
    return ref, {"hin": hin, "bo": bo, "ij": ij, "dev": dev, "want": want, "want_ij": want_ij}


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        cases = read_cases(dump)
    except m20.MissingInput as e:
        print("[chk] RESULT: SKIPPED (缺 %s：先在设备上跑 M27_DUMP=1 再复核)" % e)
        return 2
    if not cases:
        print("[chk] RESULT: SKIPPED (没有 m27_case_*.txt)")
        return 2
    print("[chk] m27 hc prefill 交叉校验：dump 目录 = %s；case 数 = %d" % (dump, len(cases)))

    for c in cases:
        case = c["case"]
        m = int(c["m"])
        mode = int(c["mode"])
        mt = int(c["mt"])
        tiles = int(c["tiles"])
        arena_dumped = int(c.get("arena_dumped", 0))
        chain = int(c.get("chain", 0))
        mutant = c.get("mutant", "none")
        reloc_mask = int(c.get("reloc_mask", 15))   # bit0=blk / bit1=injw / bit2=rstd / bit3=ijFlat
        print("\n===== case %s：m=%u mode=%u(%s) MT=%u tiles=%u arena_dumped=%u chain=%u mutant=%s "
              "reloc_mask=0x%X =====" % (case, m, mode, MODE_NAME.get(mode, "?"), mt, tiles, arena_dumped,
                                         chain, mutant, reloc_mask))
        if mutant != "none":
            NOTES.append("case %s 是**负向对照**档（mutant=%s）：本档期望判据变红" % (case, mutant))
        lay = None
        if arena_dumped:
            try:
                lay = {k: int(v) for k, v in read_kv(os.path.join(dump, "m27_layout_%s.txt" % case)).items()}
            except (OSError, ValueError):
                # 布局文件缺失/不可解析 ⇒ 只用得到它的那几条 guard 降级为 SKIP（判定项不受影响）
                SKIP[0] += 1
                print("[chk] 注记：缺 m27_layout_%s.txt ⇒ arena 差分见证跳过" % case)

        ref1, res1 = judge_boundary(dump, "%s_a1" % case, m, mode, "a1", "%s_a1_ij" % case,
                                    "%s_a1_hin" % case, "%s_a1_bo" % case, skip_hcp=(mode == MODE_MIX),
                                    reloc_mask=reloc_mask)
        dev1 = res1["dev"]
        # ---- 新增判定项：边界 #1 的 **ij handoff 面**（= 它的 OH[:,320:324) 重定位结果）----
        # 这一项是必需的：A2 的 ij 输入取自这块面 ⇒ 若不单独判它，A2 的判据会"两边都用同一份
        # 被覆盖的数据"而假绿（M119 第一版就是这样：arena 的 OH 区不持久，m=257 档只有 22.2%
        # 的行是参考的 OH inj 列，而 A2 的八条判定项当时全绿）。
        if res1["want_ij"]:
            # 平铺 ij 面的行距是 32 B（IJ_STRIDE 元素）⇒ 只比前 INJ_N 列
            ijh = rd(dump, "%s_a1_ij_handoff" % case, np.uint16)
            if ijh.shape[1] < INJ_N:
                ijh = ijh.reshape(m, -1)
            judge_bf16("%s_a1.ij_handoff (OH 列)" % case, ijh[:m, :INJ_N],
                       ref1["lora"][:, LOWRANK:LOWRANK + INJ_N])
        else:
            SKIP[0] += 1
            why = "本档不产出 injection logits" if mode in (MODE_MIX, MODE_COMBINE_ONLY) \
                else "reloc_mask 未把 ijFlat 当重定位目标（调用方传 nullptr）"
            print("[chk] 判定 %-24s SKIPPED（%s）" % (case + "_a1.ij_handoff", why))

        ref2 = None
        if chain and mode not in (MODE_MIX, MODE_COMBINE_ONLY):
            ref2, res2 = judge_boundary(dump, "%s_a2" % case, m, mode, "a2", "%s_a1_ij_handoff" % case,
                                        "%s_a2_hin" % case, "%s_a2_bo" % case, skip_hcp=False,
                                        reloc_mask=reloc_mask)
        elif chain:
            SKIP[0] += 1
            print("\n[chk] A2 SKIPPED（mode %s：H' / OH 列不完整 ⇒ 链上第二边界没有真实输入）" %
                  MODE_NAME.get(mode, "?"))

        # ---- P2-1 配置的非空洞性见证：被判 SKIP 的平铺面必须仍是"毒值"（= 调用方确实没搬）----
        # 否则"SKIP"会掩盖"面被写坏了"这件事（把该配置变成一个空洞档）。
        for bit, nm, ok_mode in ((0, "blk", mode != MODE_COMBINE_ONLY), (1, "injw", mode != MODE_MIX),
                                 (2, "rstd", mode != MODE_COMBINE_ONLY)):
            if (reloc_mask & (1 << bit)) == 0 and ok_mode:
                rawp = os.path.join(dump, "%s_a1_%s.bin" % (case, nm))
                if os.path.exists(rawp):
                    with open(rawp, "rb") as f:
                        bytes_ = f.read()
                    guard("a1.%s 未当重定位目标 ⇒ 平面仍是毒值" % nm,
                          bool(np.all(np.frombuffer(bytes_, dtype=np.uint8) == 0xCD)),
                          "（reloc_mask=0x%X 的第 %d 位为 0；这条让该配置不是空洞档）" % (reloc_mask, bit))

        # ---- 非空洞性 guard ----
        print("\n-- guard（非空洞性 / 结构性见证；不计入判定项）--")
        head = slice(0, min(m, 64))
        guard("a1.hin 值多样（非空洞）", nuniq(res1["hin"][head]) >= 32,
              "distinct=%d/%d" % (nuniq(res1["hin"][head]), res1["hin"][head].size))
        if res1["want"]["blk"]:
            guard("a1.blk 值多样（非空洞）", nuniq(dev1["blk"][head]) >= 32,
                  "distinct=%d/%d" % (nuniq(dev1["blk"][head]), dev1["blk"][head].size))
        if res1["want"]["injw"]:
            # 有效值在每 32B 槽的首元素（列 s*INJW_SLOT）⇒ 与判定项用同一种取法
            iw = dev1["injw"].reshape(-1, INJW_SLOT)[: m * HC, 0]
            guard("a1.injw 值多样（非空洞）", nuniq(iw) >= min(8, m * HC),
                  "distinct=%d/%d" % (nuniq(iw), m * HC))
        if mode != MODE_MIX:
            guard("H' != H（段体确实改了残差流）", not np.array_equal(dev1["hcp"][:m], res1["hin"][:m]))
        else:
            # mode MIX 不物化 H'：**只有"没有更早块 scratch 覆盖"的那几行**必须仍是零
            # （块 t 的 HCP 行会被块 ≤ t-WS_XN/TILE_STRIDE 的 scratch 覆盖 —— arena 行页布局的必然；
            #  mode MIX 的消费方用 hIn，与 decode 的 hcpFromWs=false 分支同口径）
            safe_rows = min(m, (lay["off_xn"] // lay["tile_stride"]) * mt) if lay else min(m, 64)
            guard("mode MIX 未写 HCP 区（前 %d 行）" % safe_rows,
                  bool(np.all(dev1["hcp"][:safe_rows] == 0)))
        if ref2 is not None:
            guard("a2.hcp != a1.hcp（第二边界确实在动）", not np.array_equal(res2["dev"]["hcp"][:m],
                                                                             dev1["hcp"][:m]))

        # ---- arena 阻塞布局的差分见证 ----
        if arena_dumped and res1["want"]["blk"] and lay is not None:
            raw = rd(dump, "%s_arena1" % case, np.uint8).reshape(-1)
            ts, row_hid, row_hp = lay["tile_stride"], lay["row_hid"], lay["row_hp"]
            row_rstd, row_injw, row_ij = lay["row_rstd"], lay["row_injw"], lay["ij_row_bytes"]
            rows_blk = blocked_plane(raw, lay["off_blk"], row_hid, m, mt, ts)
            # 与**设备自己的平铺面**逐字节比（"arena 里确实是段体产物 + 重定位忠实搬运"这件事的见证）。
            # 与参考比的是**判定项**（有门限）；这里是结构性 guard（必须逐字节）。
            dev_blk_bytes = np.ascontiguousarray(dev1["blk"][:m]).view(np.uint8).reshape(m, HID * 2)
            rows_rstd = blocked_plane(raw, lay["off_rstd"], row_rstd, m, mt, ts)
            dev_rstd_bytes = np.ascontiguousarray(dev1["rstd"][:m]).view(np.uint8).reshape(m, row_rstd)
            wins = region_windows(lay, mt)
            d_blk = min_d(lay["off_blk"], mt * row_hid, wins, ts)
            d_rstd = min_d(lay["off_rstd"], mt * row_rstd, wins, ts)
            d_safe = min(tiles, d_blk) if d_blk else tiles
            tail = slice((tiles - d_safe) * mt, m)
            guard(("arena 末 %d 块的阻塞 blk == 设备平铺面" % d_safe) if d_safe < tiles
                  else ("arena 全部 %d 块的阻塞 blk == 设备平铺面" % tiles),
                  bool(np.array_equal(rows_blk[tail], dev_blk_bytes[tail])),
                  "（d_blk=%s 块后才被覆盖 ⇒ 更早的块必须靠块末重定位" % d_blk)
            d_rs = min(tiles, d_rstd or tiles)
            tailr = slice((tiles - d_rs) * mt, m)
            guard("arena 末 %d 块的阻塞 rstd == 设备平铺面" % d_rs,
                  bool(np.array_equal(rows_rstd[tailr], dev_rstd_bytes[tailr])), "（d_rstd=%s）" % d_rstd)
            if d_blk is not None and tiles > d_blk:
                early = slice(0, (tiles - d_blk) * mt)
                guard("更早块的阻塞 blk 已被覆盖（⇒ 块末重定位承担负载）",
                      not np.array_equal(rows_blk[early], dev_blk_bytes[early]),
                      "（被块 t+%d 的 GATE 写覆盖）" % d_blk)
            if res1["want_ij"]:
                # OH 区（ij 的源）不持久：更早块的 OH 列必须**已经不等于**设备产出的平铺 ij 面
                safe_oh = min(tiles, max(1, lay["oh_inj_off"] // ts))
                d_oh = min_d(lay["oh_inj_off"], mt * row_ij, wins, ts)
                if d_oh is not None and tiles > d_oh:
                    oh_blk = blocked_plane(raw, lay["oh_inj_off"], row_ij, m, mt, ts)
                    dev_ij = np.ascontiguousarray(rd(dump, "%s_a1_ij_handoff" % case, np.uint16)
                                                  .reshape(m, -1)).view(np.uint8).reshape(m, row_ij)
                    early = slice(0, (tiles - d_oh) * mt)
                    guard("更早块的 arena OH 列已被覆盖（⇒ ij 也必须块末重定位）",
                          not np.array_equal(oh_blk[early], dev_ij[early]),
                          "（被块 t+%d 的 H' 写覆盖；safe_oh=%d 块）" % (d_oh, safe_oh))

    print("\n[chk] 计数：本脚本判定项 %d；m20.judge 判定项 %d；guard %d；SKIPPED %d" %
          (JUDGE_REL[0], m20.JUDGE[0], GUARD[0], SKIP[0]))
    for n in NOTES:
        print("[chk] 注记：%s" % n)
    bad = list(m20.FAILS) + REL_FAILS + GUARD_FAILS
    if bad:
        print("[chk] 未过：%s" % ", ".join(bad))
    total = JUDGE_REL[0] + m20.JUDGE[0]
    if not bad and total > 0:
        print("[chk] RESULT: OK（判定项 %d + guard %d；SKIPPED %d）" % (total, GUARD[0], SKIP[0]))
        return 0
    if not bad and total == 0:
        print("[chk] RESULT: SKIPPED（一条判据也没跑起来）")
        return 2
    print("[chk] RESULT: FAILED（%d 条未过）" % len(bad))
    return 1


if __name__ == "__main__":
    sys.exit(main())
