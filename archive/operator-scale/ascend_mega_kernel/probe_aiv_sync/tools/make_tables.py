#!/usr/bin/env python3
"""make_tables.py —— 从 evidence/logs 的原始 SUMMARY 行生成 README 的两张总表（机械汇总，不改数字）

用法：
  python3 tools/make_tables.py <probe_aiv_sync 目录>
产物：
  stdout        —— §2 主表（markdown，可直接贴进 README）
  evidence/tables/visibility.md、aic_aiv.md —— §7 / AIC 对照表
"""
import collections
import glob
import os
import re
import sys

FAILFORM = {
    # 失败形态（依据 §3 的逐词归因与 §3.3 的时序交叉验证给出；数字全部来自日志 SUMMARY/ROUND）
    "A1_poll_nobarrier": "轮询追一个正在被覆写的目标 ⇒ 0 核收敛",
    "A2_open_only": "读到 **T+1 轮** 的值（读窗口未关）",
    "A4_open_only_waits": "同 A2（wait 挂 PIPE_S 不改变窗口未关这件事）",
    "A6_open_close_waitV": "读早于放行（wait<PIPE_V> 挡不住 MTE2 读）",
    "A7_open_close_waitE3": "读早于放行（wait<PIPE_MTE3> 挡不住 MTE2 读）",
    "A14_m1_close": "mode1 只同步同一 AI Core 内 2 个 AIV ⇒ 跨核不受保护",
    "A0_none_readonce": "无任何会合（对照：证明表灵敏）",
    "B0_geom_dense4": "同 A3 配方但 4B 密排几何 ⇒ 脆弱性 ~4.7×",
    "B1_geom_align4": "同 A3 配方，32B 独占/4B 写",
    "B2_geom_align32": "同 A3 配方，32B 独占写满",
    "B4_dense4_openonly": "M53/m19 形态：4B 密排 + 只有发布会合（最差组合）",
    "B5_dense4_28aiv": "同上，28 AIV",
    "C0_halfset": "只让一半核 set ⇒ 会合永不满（死锁）",
    "A15_m2_close": "mode2 的 set 记在 AIC 侧 ⇒ AIV 自己的计数器永不满（死锁）",
    "A22_close_closepipeE2": "收尾 wait 挂 MTE2 ⇒ 挡不住下一轮 MTE3 写（读窗口仍开）",
    "A23_close_closesetE3": "收尾 set 挂 MTE3：本实验里也 0 违规（形式上侥幸，见 §8）",
    "A26_fid_same_id": "**无违规**（同一 id 兼做发布/收尾）",
    "A27_fid_same_id_sep": "**无违规**（同一 id + 独立窗口）",
    "N0_negctrl_selfonly": "**0 违规（假 PASS）**：只校验自己那格 ⇒ 无同步也过判",
    "B3_geom_align64": "无违规",
}
RECIPE = {
    "A0_none_readonce": "无屏障（写完即读）",
    "A1_poll_nobarrier": "无屏障 + 纯轮询 4096",
    "A2_open_only": "m0 型：set<MTE3>/wait<MTE2> 仅发布（**现行**）",
    "A3_open_close": "发布 + 收尾 `set<MTE2>/wait<MTE3>`（**候选 R1**）",
    "A4_open_only_waits": "仅发布，wait<PIPE_S>",
    "A5_close_waits": "发布(wait<S>) + 收尾（候选 R1'）",
    "A6_open_close_waitV": "发布 wait<PIPE_V> + 收尾（**错挂**）",
    "A7_open_close_waitE3": "发布 wait<PIPE_MTE3> + 收尾（**错挂**）",
    "A8_setV_close": "发布 set<PIPE_V> + 收尾（**错挂**）",
    "A9_setE2_close": "发布 set<PIPE_MTE2> + 收尾（**错挂**）",
    "A10_nodrain_close": "发布（不排空）+ 收尾",
    "A11_pipebar_close": "发布（PipeBarrier<MTE3>）+ 收尾",
    "A12_dcci_close": "发布（MTE3_S+DCCI+DSB）+ 收尾",
    "A13_dcci_close_rd": "A12 + 读侧 DCCI",
    "A14_m1_close": "**mode1** + 收尾",
    "A15_m2_close": "**mode2（AIV 自 set/自 wait）** + 收尾",
    "A16_fid_fixed_close": "发布+收尾，flagId 固定一对（16 轮复用）",
    "A17_fid_rot7_close": "发布+收尾，7 对 id 轮换",
    "A18_noskew_close": "发布+收尾，无偏斜",
    "A19_poll_hybrid_close": "发布+轮询混合+收尾",
    "A20_sepwins": "仅发布 + **每轮独立窗口**（**候选 R2**）",
    "A21_28aiv_close": "发布+收尾，**28 AIV**",
    "A22_close_closepipeE2": "收尾 wait 挂 MTE2（**错挂**）",
    "A23_close_closesetE3": "收尾 set 挂 MTE3（**错挂**）",
    "A24_poll_sepwins": "无会合 + 独立窗口 + 纯轮询",
    "A25_sepwins_dense4": "仅发布 + 独立窗口 + **4B 密排几何**",
    "B0_geom_dense4": "发布+收尾，几何 **st=1,wl=4（M53 形态）**",
    "B1_geom_align4": "发布+收尾，几何 st=8,wl=4",
    "B2_geom_align32": "发布+收尾，几何 st=8,wl=32",
    "B3_geom_align64": "发布+收尾，几何 st=16,wl=64",
    "B4_dense4_openonly": "**仅发布**，几何 st=1,wl=4（M53 逐项复刻）",
    "B5_dense4_28aiv": "发布+收尾，几何 st=1,wl=4，28 AIV",
    "C0_halfset": "发布+收尾，**只让一半核参与**",
    "N0_negctrl_selfonly": "无屏障，**只校验自己那格**（负向对照）",
    "A26_fid_same_id": "发布+收尾**共用同一个 flagId（1）**，16 轮反复用",
    "A27_fid_same_id_sep": "同 A26 + 每轮独立窗口",
}


def parse_summary(line):
    kv = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def collect(paths):
    agg = collections.defaultdict(list)
    for p in paths:
        for fp in (sorted(glob.glob(os.path.join(p, "*.log"))) if os.path.isdir(p) else [p]):
            if os.path.basename(fp).startswith('session'):
                continue   # session2_*.log 是跨会话补充证据，不进变体汇总（见 README §2 表下注）
            with open(fp, errors="replace") as f:
                for line in f:
                    if line.startswith("SUMMARY ") and "variant=" in line:
                        kv = parse_summary(line)
                        agg[kv["variant"]].append(kv)
    return agg


def main_table(agg, reps):
    out = []
    out.append("| 变体 | 配方 / 参数 | 收敛次数/9 | 违规 (核,轮) 区间 | 干净的轮数 | 失败形态（§3 归因） |")
    out.append("|---|---|---|---|---|---|")
    def key(n):
        m = re.match(r"([A-Z])(\d+)", n)
        return (m.group(1), int(m.group(2))) if m else (n, 0)

    for name in sorted(agg, key=key):
        rows = agg[name]
        n0 = sum(1 for r in rows if r.get("rc") == "0")
        hangs = sum(1 for r in rows if r.get("rc") == "124" or r.get("HANG") == "1")
        viols = [int(r["viol"]) for r in rows if "viol" in r]
        cc = sorted({r.get("cores_clean", "?") for r in rows})
        col_v = f"{min(viols)}~{max(viols)}" if viols and len(set(viols)) > 1 else (
            str(viols[0]) if viols else "-")
        raw_r = {r.get("rounds_clean", "?") for r in rows}
        kvals = []
        tot = "?"
        for x in raw_r:
            if "/" in x:
                a, b = x.split("/")
                if a.isdigit():
                    kvals.append(int(a))
                tot = b
            elif x.isdigit():
                kvals.append(int(x))
        ks = sorted(set(kvals))
        if not ks:
            col_r = "—"
        elif len(ks) <= 2:
            col_r = "/".join(str(v) for v in ks) + f" /{tot}"
        else:
            col_r = f"{ks[0]}~{ks[-1]} /{tot}"
        form = FAILFORM.get(name, "")
        # 这些行的"0/9 违规"需要带注解（否则会被默认文案顶掉）
        ANNOTATED = ("N0_negctrl_selfonly", "A23_close_closesetE3", "A26_fid_same_id", "A27_fid_same_id_sep")
        if name in ANNOTATED:
            form = FAILFORM.get(name, form)
        elif n0 == len(rows) and not hangs:
            form = "**无违规**"
        elif name == "B0_geom_dense4" and n0 == 0:
            form = form or "同 A3 配方但 4B 密排几何 ⇒ 脆弱性 ~4×"
        col_cc = ",".join(cc) if len(cc) <= 3 else f"{cc[0]}..{cc[-1]}"
        out.append(f"| `{name}` | {RECIPE.get(name, '')} | **{n0}/{len(rows)}**"
                   f"{'（挂死 %d）' % hangs if hangs else ''} | {col_v} | {col_r} | {form} |")
    return "\n".join(out)


def vis_table(agg):
    out = ["| 变体 | 收敛次数/9 | host 终态 |", "|---|---|---|"]
    for name in sorted(agg):
        rows = agg[name]
        n0 = sum(1 for r in rows if r.get("rc") == "0")
        hh = sorted({r.get("host_head_ok", "?") for r in rows})
        it = sorted({r.get("iter_max", "?") for r in rows})
        out.append(f"| `{name}` | **{n0}/{len(rows)}** | host_head_ok={','.join(hh)} iter_max={','.join(it)} |")
    return "\n".join(out)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    logs = os.path.join(root, "evidence", "logs")
    os.makedirs(os.path.join(root, "evidence", "tables"), exist_ok=True)
    agg = collect([logs])
    main = [k for k in agg if re.match(r"^[ABCN]\d", k)]
    vis = [k for k in agg if k.startswith("v")]
    aic = [k for k in agg if k.startswith("v") and k in ("v0_chain_mix_standard",)]
    print("<!-- MAIN TABLE -->")
    print(main_table({k: agg[k] for k in main}, 9))
    with open(os.path.join(root, "evidence", "tables", "visibility.md"), "w") as f:
        f.write(vis_table({k: agg[k] for k in vis}) + "\n")
    with open(os.path.join(root, "evidence", "tables", "aic_aiv.md"), "w") as f:
        f.write(vis_table({k: agg[k] for k in aic}) + "\n")
