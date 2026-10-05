#!/usr/bin/env python3
"""M25 验证 C：从 msprof 的 op_summary CSV 汇总「48 次启动的设备执行总耗时」。

msprof 采集（不进 kernel，纯设备侧真值）：
    mkdir -p /tmp/m15_prof && cd /tmp/m15_prof
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    M15_REPS=3 M15_LAYERS=48 msprof --output=prof --task-time=l1 --ai-core=on \
        --ascendcl=off --runtime-api=off \
        <repo>/m15_layer_loop/build/m15_layer_loop <repo>/m15_layer_loop/weights_manifest.txt c
    /usr/local/python3.12.13/bin/python3.12 <repo>/m15_layer_loop/parse_msprof.py prof/PROF_* --host-step-ms 5.102

它做三件事：
  1. 两个 kernel 的 Count / Total / Min / Avg / Max 设备耗时；
  2. **按 3:1 层类型模式识别「一个 decode step 的 48 个 kernel」窗口**（同一窗口内的 op 序列必须
     逐层等于 config 的 layer_types → 顺带独立见证「循环按层类型派发内核」），给出每个 step 的
     设备总耗时（中位/最小/最大）与按层类型的分解；
  3. 关键 AIV/AIC pipe 占比（GDN 层是否带宽受限）。

只依赖 python 标准库。
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

GDN_TAG = "m15_gdn_layer_kernel"
ATTN_TAG = "m15_attn_placeholder_kernel"


def find_output_dir(path):
    cands = [path]
    cands += glob.glob(os.path.join(path, "PROF_*"))
    cands += glob.glob(os.path.join(path, "PROF_*", "mindstudio_profiler_output"))
    for c in cands:
        hit = glob.glob(os.path.join(c, "op_summary_*.csv"))
        if hit:
            return os.path.dirname(hit[0]), hit[0]
    raise SystemExit(f"[FAIL] 在 {path} 下找不到 op_summary_*.csv")


def expected_pattern(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    return ["ATTN" if k == "full_attention" else "GDN" for k in cfg["layer_types"]]


def kind_of(op_name):
    if GDN_TAG in op_name:
        return "GDN"
    if ATTN_TAG in op_name:
        return "ATTN"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir", help="msprof --output 目录（或 PROF_* 子目录）")
    ap.add_argument("--model-dir", default="/workspace/Qwen3.8-Flash-Next-MXFP4")
    ap.add_argument("--host-step-ms", type=float, default=None,
                    help="host 侧测到的 step wall（ms），用于与设备侧对照")
    args = ap.parse_args()

    out_dir, op_summary = find_output_dir(args.prof_dir)
    rows = []
    with open(op_summary, newline="") as f:
        for r in csv.DictReader(f):
            t = r.get("Task Duration(us)") or ""
            if not t.strip():
                continue
            rows.append({
                "op": r["Op Name"],
                "kind": kind_of(r["Op Name"]),
                "dur": float(t),
                "start": float(r["Task Start Time(us)"].strip()),
                "aiv_mte2": float(r.get("aiv_mte2_ratio") or 0.0),
                "aiv_vec": float(r.get("aiv_vec_ratio") or 0.0),
                "aiv_mte3": float(r.get("aiv_mte3_ratio") or 0.0),
                "aic_mac": float(r.get("aic_mac_ratio") or 0.0),
                "aic_mte2": float(r.get("aic_mte2_ratio") or 0.0),
                "aic_mte1": float(r.get("aic_mte1_ratio") or 0.0),
                "aic_fix": float(r.get("aic_fixpipe_ratio") or 0.0),
            })
    rows.sort(key=lambda x: x["start"])
    print(f"[msprof] {op_summary}")
    print(f"[msprof] kernel task 数 {len(rows)}（{'?' if any(r['kind'] == '?' for r in rows) else '全部识别'}）")

    # 1) 按 kernel 汇总
    print("\n-- 按 kernel 汇总（设备侧）--")
    for kind in ("GDN", "ATTN"):
        d = [r["dur"] for r in rows if r["kind"] == kind]
        if not d:
            continue
        print(f"  {kind:4s} {len(d):4d} 次  合计 {sum(d) / 1000:8.3f} ms  "
              f"min {min(d):7.3f} avg {statistics.mean(d):7.3f} max {max(d):7.3f} us")

    # 2) 识别 48 层 step 窗口
    pat = expected_pattern(args.model_dir)
    n = len(pat)
    kinds = [r["kind"] for r in rows]
    steps = []
    i = 0
    while i + n <= len(kinds):
        if kinds[i:i + n] == pat:
            steps.append(rows[i:i + n])
            i += n
        else:
            i += 1
    print(f"\n-- 识别到的完整 {n} 层 step 窗口：{len(steps)} 个（模式逐层等于 config.layer_types）--")
    mismatch = sum(1 for k in kinds if k == "?")
    if mismatch:
        print(f"  [WARN] 有 {mismatch} 个 task 无法识别 kernel 名")
    if not steps:
        print("  [FAIL] 未找到完整 step 窗口")
        return 1
    totals = [sum(r["dur"] for r in s) for s in steps]
    gdn_tot = [sum(r["dur"] for r in s if r["kind"] == "GDN") for s in steps]
    attn_tot = [sum(r["dur"] for r in s if r["kind"] == "ATTN") for s in steps]
    print(f"  每 step 设备总耗时：中位 {statistics.median(totals) / 1000:.3f} ms  "
          f"min {min(totals) / 1000:.3f}  max {max(totals) / 1000:.3f} ms  （{len(steps)} 次）")
    print(f"    其中 GDN 层合计 中位 {statistics.median(gdn_tot) / 1000:.3f} ms "
          f"（{n - pat.count('ATTN')} 层 × {statistics.median(gdn_tot) / (n - pat.count('ATTN')):.1f} us）")
    print(f"    其中 attention 占位 中位 {statistics.median(attn_tot) / 1000:.3f} ms "
          f"（{pat.count('ATTN')} 层 × {statistics.median(attn_tot) / pat.count('ATTN'):.1f} us）")

    gdn_rows = [r for s in steps for r in s if r["kind"] == "GDN"]
    if gdn_rows:
        def avg(k):
            return statistics.mean(r[k] for r in gdn_rows) * 100.0
        print(f"\n-- GDN 层 pipe 占比（均值，占比 = 该 pipe 忙碌时间 / 该核时间）--")
        print(f"  AIV: MTE2 {avg('aiv_mte2'):5.1f}%  VEC {avg('aiv_vec'):5.1f}%  MTE3 {avg('aiv_mte3'):5.1f}%")
        print(f"  AIC: MTE2 {avg('aic_mte2'):5.1f}%  MAC  {avg('aic_mac'):5.1f}%  "
              f"MTE1 {avg('aic_mte1'):5.1f}%  FIXP {avg('aic_fix'):5.1f}%")
        print("  （AIC MTE2 = 权重 HBM→L1 搬运：该占比高即说明层 kernel 受权重带宽支配）")

    if args.host_step_ms is not None:
        dev = statistics.median(totals) / 1000.0
        print(f"\n-- 对照（mission task 4：48 次启动的下发 vs 执行）--")
        print(f"  host step wall（含下发 + 尾部 sync）{args.host_step_ms:.3f} ms")
        print(f"  设备执行总耗时（48 个 kernel 之和）  {dev:.3f} ms")
        print(f"  host/设备 = {args.host_step_ms / dev:.3f}（差别即下发与同步的净开销）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
