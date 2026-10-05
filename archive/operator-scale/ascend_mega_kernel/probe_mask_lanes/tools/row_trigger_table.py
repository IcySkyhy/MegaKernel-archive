#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""row_trigger_table.py —— M94 任务 3：ids 行距越界的**触发条件表**与链上可达性

这个脚本**不碰设备**，只做算术：给定「一次 ids 落盘的字节数」`store_bytes`、ids 行距 `pitch`、
块内行数 `rows`，判「最后一行（r = rows-1）的 ids 落盘是否越出 ids 行槽、越到哪里」。

## 触发条件的推导（逐字对应 m15 的代码，路径与常量由本脚本从 tip 文件里读，不是手抄）
`m15_layer_loop/m15_moe_layer.h` 的 `RouterStage::Run`：
    nblk = ceil(M / RT_RB) ; rows = min(M - b*RT_RB, RT_RB)
`SoftmaxTopkRow(r)` 落两笔（同一迭代、r 升序）：
    ids: StoreAlign(UB_RT_IDS + r*64*4, idxs, maskAllI)     # 行距 64*4 = 256B
    ws : StoreAlign(UB_RT_WS  + r*64*4, vals, maskAll)      # 行距 256B
`m15_layer_loop/m15_moe_resources.h`：`UB_RT_WS = UB_RT_IDS + RT_RB*64*4`

若单次 ids 落盘写 `store_bytes` 字节，则第 r 行的落盘覆盖
`[r*pitch, r*pitch + store_bytes)`：
  * `store_bytes <= pitch` ⇒ 本行内，安全；
  * `store_bytes == 2*pitch`（= m17 注释声称的 512B）⇒ 覆盖第 r、r+1 两行：
      - r+1 < rows ⇒ 第 r+1 行的 ids 槽被覆盖，**但 r 升序 ⇒ 随后被第 r+1 行自己的落盘盖回**（自愈）；
      - r+1 == rows（只能是**块内最后一行**）⇒ 溢到 `UB_RT_IDS + rows*256`：
          * rows < RT_RB ⇒ 落在**未被使用**的 ids 槽内（ids 区共 RT_RB 行）⇒ 输出不可见；
          * rows == RT_RB ⇒ 落在 `UB_RT_IDS + RT_RB*256 = UB_RT_WS` ⇒ **ws 区第 0 行的 top-k 权重**
            被覆盖，而它是 r=0 时写下、之后再也不写 ⇒ **真损坏**。
⇒ 触发条件 = `rows == RT_RB`（等价地 `M % RT_RB == 0` 且 M>0），首坏字节 = `UB_RT_WS + 0`
（= ids 区尾后紧接着的 256B）；损坏内容 = ws 第 0 行的 64 个 top-k 权重。

用法：python3 tools/row_trigger_table.py [probe_dir] [--m-max 64]
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
REPO = os.path.dirname(PROBE)      # 工作树根（= probe_mask_lanes 的父目录）

# 旧形态（`maskAllI = CreateMask<int32_t, ALL>()` 配 64-lane 行距）所在的**不可变 rev**：
# M95 并入 main（`40dd6de`）之前的最后一版 main。触发条件表按这一版的几何给出。
OLD_FORM_REV = "e37909e"


def read_tip_constants(repo, rev=None):
    """读出 RT_RB / 行距 / 两区布局的**引用行**（只读）。

    `rev=None` ⇒ 读工作树文件；`rev="<sha>"` ⇒ 用 `git show <rev>:<path>` 读该**不可变 rev**。
    为什么要有 rev 模式：这些锚点（尤其「旧形态」的 `maskAllI = CreateMask<int32_t, ALL>()`）会随
    产品代码重写而消失（M95 并入 main `40dd6de` 后，`main` 上已无该行）⇒ 只钉分支/`main` 这类
    moving ref 的引用会「命令不再复现它写的输出」。pin rev 后输出与后续 main 移动无关。
    每条锚点单独报命中数：**命中数为 0 的会显式打印**（该 rev 无此形态），而不是让命中列表悄悄变短。
    """
    out = {"evidence": [], "hits": {}, "rev": rev}
    names = ["m15_layer_loop/m15_moe_layer.h", "m15_layer_loop/m15_moe_resources.h",
             "m17_moe_real/m17_moe_layer.asc", "m13_moe_layer/m13_moe_layer.asc"]
    text = {}
    for path in names:
        tag = path.split("/")[-1]
        if rev is None:
            full = os.path.join(repo, path)
            if not os.path.isfile(full):
                text[tag] = None
                continue
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                text[tag] = f.read()
        else:
            p = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (rev, path)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            text[tag] = p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None
    rx_rb = re.compile(r"static constexpr uint32_t RT_RB = (\d+)|^constexpr uint32_t RT_RB = (\d+)", re.M)
    rx_ids = re.compile(r"^constexpr uint32_t UB_RT_IDS = UB_RT_LOG \+ RT_RB \* (\w+) \* 4", re.M)
    rx_ws = re.compile(r"^constexpr uint32_t UB_RT_WS = UB_RT_IDS \+ RT_RB \* (\w+) \* 4", re.M)
    rx_wrow = re.compile(r"^constexpr uint32_t RT_WROW = (\d+)", re.M)
    rx_rows = re.compile(r"const uint32_t rows = \(M - b0 < RT_RB\) \? \(M - b0\) : RT_RB")
    rx_nblk = re.compile(r"const uint32_t nblk = \(M \+ RT_RB - 1\) / RT_RB")
    # 旧形态锚点（M95 之前）
    rx_store = re.compile(r"StoreAlign\(idsUb, idxs, maskAllI\)")
    rx_pitch = re.compile(r"UB_RT_IDS \+ r \* 64 \* 4")
    rx_maskall = re.compile(r"MaskReg maskAllI = CreateMask<int32_t, MaskPattern::ALL>")
    # 新形态锚点（M95 起）
    rx_store_new = re.compile(r"StoreAlign\(idsUb, idxs, mwi\)")
    rx_pitch_new = re.compile(r"UB_RT_IDS \+ r \* RT_WROW \* 4")
    rx_mwi = re.compile(r"MaskReg mwi = UpdateMask<int32_t>\(nw\)")
    rx_comment128 = re.compile(r"int32 的 ALL mask 是 128 lane")
    rx_upd64 = re.compile(r"MaskReg m64i = UpdateMask<int32_t>\(n64\)")
    rx_m13 = re.compile(r"StoreAlign\(idsUb, idxs, maskAllI\)")
    anchors = {
        "m15_moe_layer.h": [("RT_RB(static)", rx_rb), ("nblk=", rx_nblk), ("rows=", rx_rows),
                            ("ids_pitch_old", rx_pitch), ("maskAllI_old", rx_maskall), ("ids_store_old", rx_store),
                            ("ids_pitch_new", rx_pitch_new), ("mwi", rx_mwi), ("ids_store_new", rx_store_new)],
        "m15_moe_resources.h": [("RT_RB", rx_rb), ("UB_RT_IDS", rx_ids), ("UB_RT_WS", rx_ws), ("RT_WROW", rx_wrow)],
        "m17_moe_layer.asc": [("m17_comment_128", rx_comment128), ("m17_upd64", rx_upd64)],
        "m13_moe_layer.asc": [("m13_maskAllI", rx_maskall), ("m13_ids_store", rx_m13)],
    }
    for tag, rxlist in anchors.items():
        body = text.get(tag)
        if body is None:
            out["evidence"].append(("%s:ALL" % tag, "MISSING", "(该 rev 无此文件或无此 rev)"))
            continue
        lines = body.splitlines()
        for name, rx in rxlist:
            hit = 0
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    out["evidence"].append(("%s:%s" % (tag, name), i, line.strip()))
                    hit += 1
                    if name == "RT_RB":
                        g = rx.search(line).group(1) or rx.search(line).group(2)
                        out.setdefault("RT_RB", int(g))
                    if name == "RT_WROW":
                        out.setdefault("RT_WROW", int(rx.search(line).group(1)))
                    if name in ("UB_RT_IDS", "UB_RT_WS"):
                        out.setdefault(name + "_stride", rx.search(line).group(1))
            if hit == 0:
                out["evidence"].append(("%s:%s" % (tag, name), "-", "**命中数为 0**（该 rev 无此形态/此锚点）"))
            out["hits"][("%s:%s" % (tag, name))] = hit
    # 行距 = 该 rev 里 ids 行的宽度（元素数 × 4B）：旧形态 64 个 int32；新形态 RT_WROW（=64）
    if out["hits"].get("m15_moe_layer.h:ids_pitch_old", 0) > 0:
        out["pitch"] = 64 * 4
        out["pitch_src"] = "旧形态：`UB_RT_IDS + r * 64 * 4`（64 个 int32 = 256B）"
    elif out["hits"].get("m15_moe_layer.h:ids_pitch_new", 0) > 0:
        w = out.get("RT_WROW")
        out["pitch"] = (w if w else 64) * 4
        out["pitch_src"] = "新形态：`UB_RT_IDS + r * RT_WROW * 4`（RT_WROW=%s ⇒ %sB）" % (w, out["pitch"])
    else:
        out["pitch"] = 64 * 4
        out["pitch_src"] = "**两形态锚点的命中数都是 0** ⇒ 退回 64*4 的假设，需人工核"
    return out


def build_arg_parser():
    """**真参数解析**：带取值的选项必须吃掉它的取值（旧版按 `--` 前缀滤 token ⇒ `--rev <sha>` 的 sha
    被当成位置参数 `repo`，`git show` 全失败、锚点全「未命中」而 rc 仍为 0 —— 第 3 轮复审 P2-1）。"""
    ap = argparse.ArgumentParser(
        description="M94 任务 3：ids 行距越界的触发条件表（纯算术；常量从指定 rev 读出）")
    ap.add_argument("repo", nargs="?", default=REPO,
                    help="仓库根（默认 = 本脚本所在工作树的根）")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rev", default=None,
                   help="钉一个**不可变** rev 读锚点（默认 %s；`--rev=<sha>` 连写也支持）" % OLD_FORM_REV)
    g.add_argument("--tip", action="store_true", help="读工作树（看当前形态，而不是钉 rev）")
    ap.add_argument("--m-max", type=int, default=64, help="枚举的 M 上界（默认 64）")
    return ap


def main():
    args = build_arg_parser().parse_args()
    m_max = args.m_max
    repo = args.repo
    # 默认钉**不可变 rev**（= M95 并入 main 之前的最后一版，旧形态与 RT_RB=8 几何都在这里）；
    # `--rev <sha>` 换 rev；`--tip` 读工作树（看新形态用）。
    rev = None if args.tip else (args.rev if args.rev else OLD_FORM_REV)

    c = read_tip_constants(repo, rev)
    hits = c["hits"]
    # ---- 锚点齐备性（rc 契约）：显式指定的 rev/tip 必须**真的读到该形态的锚点** ----
    old_core = ["m15_moe_layer.h:ids_pitch_old", "m15_moe_layer.h:maskAllI_old", "m15_moe_layer.h:ids_store_old"]
    new_core = ["m15_moe_layer.h:ids_pitch_new", "m15_moe_layer.h:mwi", "m15_moe_layer.h:ids_store_new"]
    n_old = sum(1 for k in old_core if hits.get(k, 0) > 0)
    n_new = sum(1 for k in new_core if hits.get(k, 0) > 0)
    n_m13 = hits.get("m13_moe_layer.asc:m13_ids_store", 0)
    rb_ok = c.get("RT_RB") is not None
    anchors_ok = (rb_ok and n_m13 > 0 and (n_old == len(old_core) or n_new == len(new_core)))

    print("# M94 任务 3：ids 行距越界的触发条件表（纯算术；常量从指定 rev 读出）")
    print("# repo = %s ; rev = %s" % (repo, rev if rev else "(工作树 / --tip)"))
    print("# 默认 rev 是**不可变**的 %s：M95 并入 main 后，`main` 上旧形态已不存在，" % OLD_FORM_REV)
    print("#   钉 moving ref 会让「命令 → 输出」随 main 移动而失效（第 2 轮复审 P2）。")
    print("# 参数：repo（位置，可省）| --rev <sha> / --rev=<sha> | --tip | --m-max <N>")
    print("")
    print("== 锚点齐备性（rc 契约：读不到锚点 ⇒ rc≠0，不静默出降级表）==")
    print("  旧形态核心锚点命中 %d/%d；新形态核心锚点命中 %d/%d；m13 内容锚点命中 %d；该 rev 的 RT_RB = %s"
          % (n_old, len(old_core), n_new, len(new_core), n_m13, c.get("RT_RB")))
    print("  判定：%s" % ("PASS（至少一形态的锚点齐备）" if anchors_ok else
                        "FAIL（两形态核心锚点都不齐备或 RT_RB 读不到 ⇒ 下表是**降级表**，不可用）"))
    print()
    print("== 常量与引用（逐条打印，便于核对）==")
    found = [e for e in c["evidence"] if not (e[1] == "-" or e[1] == "MISSING")]
    zeros = [e for e in c["evidence"] if e[1] == "-" or e[1] == "MISSING"]
    for tag, ln, s in found:
        print("  命中   %-40s L%-5s %s" % (tag, ln, s[:120]))
    if zeros:
        print("  未命中（该 rev 无此锚点；它们属另一形态，逐条列出以免「命中列表悄悄变短」）：")
        for tag, _, _ in zeros:
            print("         %s" % tag)
    print()
    pitch = c["pitch"]
    rb_tip = c.get("RT_RB")
    print("  行距 pitch = %s B（来源：%s）" % (pitch, c.get("pitch_src")))
    print("  该 rev 的 RT_RB = %s" % (rb_tip if rb_tip is not None else "-"))
    print()

    print("== 触发判定（按 store_bytes 假设 x RT_RB x M 枚举）==")
    print("  判据：rows = RT_RB（等价 M %% RT_RB == 0）⇒ 末行 ids 落盘溢到 ws 第 0 行（首坏 = UB_RT_WS+0）")
    print("        rows < RT_RB ⇒ 溢到 ids 区内未使用的槽 ⇒ 输出不可见（自愈/无害）")
    print("        store_bytes <= pitch ⇒ 任何 M 都不越界")
    print()
    print("  %-10s %-7s %-8s %-10s %-24s %s" % ("store_B", "RT_RB", "M", "rows", "覆盖区间(相对 ids 区)", "判定"))
    hits = {}
    for sb in (512, 256):
        for rb in sorted({x for x in (rb_tip, 8, 16) if x}):
            for M in range(1, m_max + 1):
                nblk = (M + rb - 1) // rb
                for b in range(nblk):
                    b0 = b * rb
                    rows = min(M - b0, rb)
                    r = rows - 1
                    lo, hi = r * pitch, r * pitch + sb
                    region_end = rb * pitch
                    if hi <= pitch * rows:
                        verd = "本行内 OK"
                    elif hi <= region_end:
                        verd = "溢到 ids 第 %d 行槽（是否越区: 否）" % rows
                        verd = "溢到未使用 ids 槽（无外部可见影响）"
                    else:
                        verd = "**越区** 首坏 = UB_RT_WS+0（ws 第 0 行权重）"
                    key = (sb, rb, verd, rows == rb)
                    if key not in hits:
                        hits[key] = (M, rows, lo, hi)
    for (sb, rb, verd, is_trig) in sorted(hits, key=lambda k: (-k[0], k[1], k[3])):
        if not is_trig and "越区" in verd:
            continue
        M, rows, lo, hi = hits[(sb, rb, verd, is_trig)]
        print("  %-10s %-7s %-8s %-10s [%d,%d)%-9s %s" % (sb, rb, ("%d..%d" % (min(x[0] for x in [hits[
            (sb, rb, verd, is_trig)]]), m_max)) if is_trig else str(M), rows, lo, hi, "", verd))

    print()
    print("== 触发集合（M 值；RT_RB 取该 rev 的实测值、以及 M91 之前的值 16）==")
    for sb in (512, 256):
        for rb in sorted({x for x in (rb_tip, 8, 16) if x}):
            trig = [M for M in range(1, m_max + 1) if M % rb == 0] if sb > pitch else []
            print("  store_B=%-4d RT_RB=%-3d 触发 M = %s%s" % (
                sb, rb, trig, "" if sb > pitch else "（store_B <= 行距 ⇒ 无触发 M）"))

    print()
    print("== 读数 vs 假设 ==")
    print("  本表给出的是**条件式**结论：它成立的前提是「单次 ids 落盘写 512B」。")
    print("  实测（probe_mask_lanes 的 row_* 变体 + lane_* 变体）给出的 store_bytes = 256B = 1 VL")
    print("  ⇒ 本表 store_B=256 那一列才是真实几何：任何 M 都不越界。")
    print("  结论与证据的绑定见 README.md §5/§6。")
    # rc 契约：锚点不齐备（读不到该形态）时**不得**返回 0 —— 否则「显式 rev 读不到 ⇒ 降级表」
    # 会被按 rc 接入的调用方当成正常结果（第 3 轮复审 P2-1）。
    print("RESULT: %s" % ("PASS（锚点齐备；表与该 rev 的几何一致）" if anchors_ok else
                          "FAIL（锚点不齐备 ⇒ 上表是降级表，勿采信）"))
    return 0 if anchors_ok else 1


if __name__ == "__main__":
    sys.exit(main())
