#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m15_layer_loop/parse_msprof_chain.py —— M65：**48 层链**的段级设备侧分解汇总

输入 = `msprof --task-time=l1 --ai-core=on` 产出的 `op_summary_*.csv`（一份或几份，带标签）。
输出 = 每个符号的 Count / Min / Avg / Max / Total（**设备侧** µs），以及三条派生量：

  ① **每 step（48 层 + 末层 mixer）**：36×`gdn_hc` + 12×`attn_hc` + 1×`final_mixer`（3:1 层模式）；
  ② **PLE 打断点的开关 A/B**（`M15_CHAIN_PLE=0/1`）：同一符号的 Avg 差 ⇒「打断点真的接上了」的
     唯一设备侧见证（分段与单次路径按设计逐字节等价，输出判据抓不住标志被忽略 —— 见 README M65-3.2）；
  ③ **「1 次启动/层」vs「4 段独立启动/层」**：链的符号和 vs 手工路径的四个段符号和
     ⇒ 每层摊薄的**启动/收尾 + 层内 handoff**成本（同一进程、同一采集里两组符号同时出现）。
  ④ `--table`：把「多份采集的每列范围 + **该列取值的来源采集**」直接输出成 README 可贴的 markdown 表
     （**README 的表由本工具产出，不手抄** —— 手抄会让「由归档 CSV 现算」这句话不成立）。

纪律（与 README 一致）：
  * **host 墙钟不作为证据**；只用设备侧 `Task Duration(us)`。
  * 共享卡 ⇒ 同时给 **Avg 与 Min** 两个口径（Avg 会被别人的进程污染，Min 相对稳）。
  * 只依赖 python 标准库。

用法：

    /usr/local/python3.12.13/bin/python3 m15_layer_loop/parse_msprof_chain.py \
        a1=evidence/m65_msprof_a1.csv b1=evidence/m65_msprof_b1.csv c1=evidence/m65_msprof_c1.csv
    # README 的 §1 表就用这条（表由工具产出，列为：Count/次、Min、中位、稳健 Avg、原始 Avg、Max；
    # 每格形如「值（来源采集）」）：
    /usr/local/python3.12.13/bin/python3 m15_layer_loop/parse_msprof_chain.py --table \
        a1=… a2=… a3=… b1=… b2=… b3=… c1=…
"""
import argparse
import csv
import glob
import os
import statistics
import sys

# 符号 → 短名（按需扩展；未列出的符号原样打印）
SHORT = {
    "m15_layer_kernel_gdn_hc": "链·GDN 层（四相位）",
    "m15_layer_kernel_attn_hc": "链·attention 层（四相位）",
    "m15_final_mixer_kernel": "末层全局 mixer",
    "m15_layer_kernel_gdn": "两相位·GDN 层（M40 形态）",
    "m15_layer_kernel_attn": "两相位·attention 层（M40 形态）",
    "m15_hc_segment_kernel": "手工·单边界 hc",
    "m15_gdn_layer_kernel": "手工·GDN 段",
    "m15_moe_segment_kernel": "手工·MoE 段",
    "m15_attn_placeholder_kernel": "手工·attention 占位",
}


def find_csv(path):
    cands = [path] + glob.glob(os.path.join(path, "PROF_*")) + \
        glob.glob(os.path.join(path, "PROF_*", "mindstudio_profiler_output"))
    for c in cands:
        hit = glob.glob(os.path.join(c, "op_summary_*.csv"))
        if hit:
            return sorted(hit)[-1]
    return path if path.endswith(".csv") else None


def read_csv(path):
    """→ {symbol: [durations us]}（只取 Task Type 含 AIC/AIV 的 kernel 行）"""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("OP Type") or row.get("Op Name") or ""
            try:
                dur = float(row.get("Task Duration(us)") or 0.0)
            except ValueError:
                continue
            if dur <= 0:
                continue
            out.setdefault(name, []).append(dur)
    return out


def short(name):
    for k, v in SHORT.items():
        if k in name:
            return v
    return name[:40]


def pick(syms, key):
    for n, v in syms.items():
        if key in n:
            return stat(v)
    return None


def stat(v):
    """设备侧聚合。**必须给中位/稳健量**：共享卡上单次启动被别人的进程挤到 100 ms~100 s 量级是
    实测过的（M58 记录过三次现场），Avg 会被这类样本虚高，中位数与「≤3×中位」的稳健和不受影响。"""
    med = statistics.median(v)
    keep = [x for x in v if x <= 3.0 * med] if med > 0 else list(v)
    return dict(n=len(v), mn=min(v), avg=statistics.fmean(v), med=med, mx=max(v), tot=sum(v),
                nkeep=len(keep), tot_keep=sum(keep),
                keep_avg=(statistics.fmean(keep) if keep else 0.0))


def print_table(runs):
    """把多份采集的每列范围 + 每列取值的来源采集打成 markdown 表（README 直接贴，避免手抄）。

    「来源采集」是关键：本仓的教训是「声明由归档 CSV 现算的表里掺了一个表外/别处的值」——
    把 argmin/argmax 的 tag 一起打出来，读者就能当场核出每个格子出自哪份 CSV。
    """
    print("\n===== markdown 表（每列 = 该列在 %d 份采集里的范围；括号里是取到该值的采集 tag）====="
          % len(runs))
    print("| 符号 | Count/次 | Min | 中位 | 稳健 Avg（≤3×中位） | 原始 Avg | Max |")
    print("|---|---|---|---|---|---|---|")
    names = []
    for syms in runs.values():
        for n in syms:
            if n not in names:
                names.append(n)
    for name in sorted(names, key=lambda n: -max(stat(runs[t][n])["tot_keep"] if n in runs[t] else 0
                                                 for t in runs)):
        per = {t: stat(runs[t][name]) for t in runs if name in runs[t]}
        if not per:
            continue
        nset = sorted({v["n"] for v in per.values()})
        ncell = str(nset[0]) if len(nset) == 1 else "/".join(str(x) for x in nset)

        def cell(key, lo=True):
            best = (min if lo else max)(per.items(), key=lambda kv: kv[1][key])
            return "%.2f（%s）" % (best[1][key], best[0])

        mins, mins_ = cell("mn", True), cell("mn", False)
        meds, meds_ = cell("med", True), cell("med", False)
        ravg, ravg_ = cell("keep_avg", True), cell("keep_avg", False)
        mx = cell("mx", False)
        polluted = sorted(t for t in runs if name in runs[t] and stat(runs[t][name])["nkeep"]
                          < stat(runs[t][name])["n"])
        clean = sorted(t for t in per if t not in polluted)
        if not polluted:
            raw = "同稳健"
        else:
            best = min(clean, key=lambda t: per[t]["avg"]) if clean else None
            raw = "干净 %s（%s）；含污染样本的采集 %s" % (
                ("%.2f" % per[best]["avg"]) if best else "—",
                best if best else "—",
                "、".join("%s=%.0f" % (t, per[t]["avg"]) for t in polluted))
        rng = lambda a, b: a if a.split("（")[0] == b.split("（")[0] else "%s – %s" % (a, b)
        print("| `%s` | %s | **%s** | %s | %s | %s | %s |" %
              (short(name), ncell, rng(mins, mins_), rng(meds, meds_), rng(ravg, ravg_), raw, mx))
    # 每 step（48 层 + 末层 mixer）的三种口径，逐采集
    print("\n| 采集 | 每 step 稳健 Avg（µs） | 每 step 中位（µs） | 每 step Min（µs） |")
    print("|---|---|---|---|")
    for t, syms in runs.items():
        g, a, gm = pick(syms, "m15_layer_kernel_gdn_hc"), pick(syms, "m15_layer_kernel_attn_hc"), \
            pick(syms, "m15_final_mixer_kernel")
        if not (g and a):
            continue
        z = lambda d, k: 36 * d[k] + 12 * a[k] + (gm[k] if gm else 0.0)
        print("| %s | %.0f | %.0f | %.0f |" % (t, z(g, "keep_avg"), z(g, "med"), z(g, "mn")))
    print("\n（口径：本表只统计上面列出的这批 CSV；若要把别的系列（如 2 层放大 `p*`、`runs=all` 的 `t*`）"
          "并进来，必须一并重出本表 —— **不要手抄其它系列的值**。）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="tag=path 或 path（可给 msprof 输出目录）")
    ap.add_argument("--table", action="store_true",
                    help="额外产出 markdown 表：每列的范围 + 该列取值的来源采集（供 README 直接贴）")
    args = ap.parse_args()

    runs = {}
    for spec in args.csvs:
        tag, _, path = spec.partition("=")
        if not path:
            tag, path = os.path.basename(spec.rstrip("/")), spec
        p = find_csv(path)
        if p is None:
            print("[FAIL] 找不到 op_summary_*.csv：%s" % spec)
            return 2
        runs[tag] = read_csv(p)
        print("[prof] %-8s ← %s" % (tag, p))

    for tag, syms in runs.items():
        print("\n===== 采集 %s（设备侧 Task Duration，µs）=====" % tag)
        print("%-26s %5s %9s %9s %9s %9s %9s %11s %11s" %
              ("符号", "Count", "Min", "中位", "Avg", "Max", "稳健Avg", "稳健Total", "Total"))
        for name in sorted(syms, key=lambda n: -stat(syms[n])["tot_keep"]):
            s = stat(syms[name])
            print("%-26s %5d %9.2f %9.2f %9.2f %9.2f %9.2f %11.1f %11.1f" %
                  (short(name), s["n"], s["mn"], s["med"], s["avg"], s["mx"], s["keep_avg"], s["tot_keep"],
                   s["tot"]))
            if s["nkeep"] < s["n"]:
                print("%-26s       ↑ 剔除 %d 个被共享卡污染（>3×中位）的样本后：Min=%.2f 中位=%.2f"
                      % ("", s["n"] - s["nkeep"], s["mn"], s["med"]))

    # ① 每 step（48 层 = 36 GDN + 12 attention，3:1）+ 末层 mixer
    print("\n===== 派生量 =====")
    for tag, syms in runs.items():
        g, a, gm = pick(syms, "m15_layer_kernel_gdn_hc"), pick(syms, "m15_layer_kernel_attn_hc"), \
            pick(syms, "m15_final_mixer_kernel")
        if g and a:
            step_avg = 36 * g["keep_avg"] + 12 * a["keep_avg"] + (gm["keep_avg"] if gm else 0.0)
            step_min = 36 * g["mn"] + 12 * a["mn"] + (gm["mn"] if gm else 0.0)
            print("[%s] ① 每 step（4 相位链）= 36×%.2f + 12×%.2f + mixer %.2f = **稳健Avg %.0f µs / "
                  "Min %.0f µs**（稳健Avg = 只用 ≤3×中位的样本）"
                  % (tag, g["keep_avg"], a["keep_avg"], (gm["keep_avg"] if gm else 0.0), step_avg, step_min))
            print("        （gdn_hc Count=%d ⇒ 本次跑了 %d 条链；attn_hc Count=%d）"
                  % (g["n"], g["n"] // 36 if g["n"] % 36 == 0 else -1, a["n"]))
        # ③ 链 vs 手工路径
        m1, m2, m3, m4 = (pick(syms, "m15_hc_segment_kernel"), pick(syms, "m15_gdn_layer_kernel"),
                          pick(syms, "m15_moe_segment_kernel"), pick(syms, "m15_attn_placeholder_kernel"))
        if g and a and m2 and m3:
            chain_tot = g["tot_keep"] + a["tot_keep"] + (gm["tot_keep"] if gm else 0.0)
            manual = sum(x["tot_keep"] for x in (m1, m2, m3, m4) if x)
            nlayers = g["n"] + a["n"]
            print("[%s] ③ 链（%d 层，1 次启动/层）稳健设备合计 %.1f µs vs 手工路径（hc#1+子层段+hc#2+"
                  "MoE，≥4 次启动/层）稳健合计 %.1f µs ⇒ **每层摊薄差 %.1f µs**（正数 = 手工更贵）" %
                  (tag, nlayers, chain_tot, manual, (manual - chain_tot) / max(nlayers, 1)))
            if m1:
                print("        手工·单边界 hc：Count=%d Min=%.2f Avg=%.2f Max=%.2f（Min 列对应 "
                      "combine-only 档、Max 列对应全链档 —— 两种档共用一个符号）"
                      % (m1["n"], m1["mn"], m1["avg"], m1["mx"]))

    if args.table:
        print_table(runs)

    tags = list(runs)
    if len(tags) >= 2:
        print("\n===== ② PLE 打断点开关 A/B（同名符号的 Avg 差）=====")
        base = runs[tags[0]]
        for tag in tags[1:]:
            g0, g1 = pick(base, "m15_layer_kernel_gdn_hc"), pick(runs[tag], "m15_layer_kernel_gdn_hc")
            if g0 and g1:
                print("[%s] vs [%s]：gdn_hc 中位 %.2f → %.2f µs（Δ=%.2f µs/次启动）；稳健Avg %.2f → "
                      "%.2f（Δ=%.2f）；Min %.2f → %.2f；样本 n=%d/%d（剔除污染 %d/%d）" %
                      (tags[0], tag, g0["med"], g1["med"], g1["med"] - g0["med"], g0["keep_avg"],
                       g1["keep_avg"], g1["keep_avg"] - g0["keep_avg"], g0["mn"], g1["mn"], g0["n"],
                       g1["n"], g0["n"] - g0["nkeep"], g1["n"] - g1["nkeep"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
