#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R0 的非空洞性对照（M56）。

WHY：M46（`e330796`）修好 `tools/golden/moe_block_ref.py::router_topk` 的设备 FTZ 建模后，
     `m22_router512` 的报告项 **R0 从 `125 槽 / 20 行`（m=4097 档）变成 `0 槽 / 0 行`（全档）**。
     "变成 0" 有两种可能：① R0 的语义变了（预期）② R0 的匹配器坏了、什么都看不见（危险）。
     本脚本把两者区分开：**同一份设备档数据上，同时算「当前 golden」与「把 FTZ 建模退回的
     golden 行为」两个 R0**。若前者 0/0、后者 125/20，则 R0 仍有咬合力，0/0 是语义变化的预期值。

不做的事：不改 golden、不改归档；只复算 R0 的两个口径并打印。

用法：
    /usr/local/python3.12.13/bin/python3 tools/r0_semantics_check.py
    # 可选：--mode mode2|mode4  --m 4097
退出码：0 = 两个口径都算出来了（且与 M46 commit message 的量级一致）；2 = 输入缺失**或环境缺 numpy**
（本仓约定解释器 `/usr/local/python3.12.13/bin/python3` 自带 numpy；换解释器时**不会**以 rc=1 抛 traceback）
"""
import argparse
import importlib.util
import os
import sys

try:
    import numpy as np
except ImportError as exc:                                      # noqa: F401
    print("RESULT: SKIPPED (本机解释器缺 numpy —— %s；本脚本需要 numpy 与 golden 对拍，"
          "请用 /usr/local/python3.12.13/bin/python3。这不是通过)" % exc)
    sys.exit(2)

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, os.pardir, "tools", "golden"))


def load_check_ref():
    spec = importlib.util.spec_from_file_location("cr", os.path.join(PKG, "check_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mode2", choices=("mode2", "mode4"))
    ap.add_argument("--m", type=int, default=4097)
    args = ap.parse_args()

    dump = os.path.join(PKG, "evidence", args.mode, "real_m%d" % args.m)
    wpath = os.path.join(PKG, "data", "router_weight.bin")
    if not os.path.isdir(dump) or not os.path.exists(wpath):
        print("RESULT: SKIPPED (输入缺失 — %s 或 %s)" % (dump, wpath))
        return 2

    cr = load_check_ref()
    from moe_block_ref import router_topk

    m = args.m
    if m == 4097:
        # 该档 x.bin 未入库（21MB）⇒ 用归档 seed 重建（与 check_ref.py 同一条路径）
        xbits = cr.gen_activations_uniform(m, 5)
    else:
        xbits = cr.load_bf16(os.path.join(dump, "x.bin"), (m, cr.HIDDEN))
    x = cr.bf16_view(xbits).reshape(m, cr.HIDDEN)
    w = cr.bf16_view(cr.load_bf16(wpath, (cr.E, cr.HIDDEN))).reshape(cr.E, cr.HIDDEN)

    log_ref, ids_current, _ = router_topk(x, w, cr.KTOPK)          # 当前 golden（已建模 FTZ）
    sc = np.exp(log_ref)
    sc[sc < cr.FTZ_MIN_NORMAL] = np.float32(0.0)
    ids_ref = np.argsort(-sc, axis=1, kind="stable")[:, :cr.KTOPK].astype(np.int32)

    # 回归探测：把 FTZ 建模退回 = pre-M46 的 golden 行为
    ids_native = np.argsort(-np.exp(log_ref), axis=1, kind="stable")[:, :cr.KTOPK].astype(np.int32)

    def r0(a, b):
        return int((a != b).sum()), int((a != b).any(axis=1).sum())

    s1, r1 = r0(ids_current, ids_ref)
    s2, r2 = r0(ids_native, ids_ref)
    print("档 = evidence/%s/real_m%d（真实 checkpoint 权重）" % (args.mode, m))
    print("  R0（当前 golden，已建模 FTZ）            : %d 槽 / %d 行  ← 期望 0/0（两套 FTZ 实现一致）" % (s1, r1))
    print("  R0（回归探测：golden 退回不建模 FTZ）    : %d 槽 / %d 行  ← 期望 >0，证明 R0 仍有咬合力" % (s2, r2))
    if m == 4097:
        want = (s1 == 0 and r1 == 0 and s2 == 125 and r2 == 20)
        print("  与 M46 commit message 的量级一致（125 槽 / 20 行）：%s" % ("是" if want else "否"))
        print("RESULT: %s (两个口径都算出来了；R0 非空洞)" % ("OK" if want else "MISMATCH"))
        return 0 if want else 1
    print("RESULT: OK (两个口径都算出来了；R0 非空洞)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
