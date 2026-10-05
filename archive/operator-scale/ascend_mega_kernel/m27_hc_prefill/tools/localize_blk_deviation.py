#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m27_hc_prefill/tools/localize_blk_deviation.py —— 把 `blk` 的余差**定位到段**（离线，不占设备）

背景：主档在 m ≥ 33 的 `a2` 链、以及 m=4097 的 `blk` 上越"良态 bf16 ulp ≤2 / 良态逐位率 ≥99%"这条
保守门限（`maxulp` 4–9、良态逐位率 0.988–0.991；而**归一化绝对误差** ≤6.6e-03，比门限 1e-2 还低三倍）。
本脚本回答一个更窄的问题：**这段余差最早出现在哪一 stage**（不是靠猜）。

做法（只用**已有**的 dump，不需要重跑设备）：
  1. 用设备自己的输入（hin/bo/ij + 权重）复算独立参考（`m20_hyperconn/check_ref.py::reference`）；
  2. 从 **arena 的阻塞布局**里按 `m27_layout_<case>.txt` 把设备自己的中间量抽出来
     （`xn` = 平铺面 + `oh[:,:320]`/`ij` / `ls` / `gate` —— 它们的地址由布局常量算，不写第二份）；
  3. 逐 stage 报：与参考的（bf16 位模式）**逐位一致率**、`ulpMax`、以及"参考的该 stage 值在
     bf16 网格上的**最近邻距离**"分布 —— 若某 stage 的差异只出现在"参考值几乎落在两个 bf16
     值正中间"的元素上，就说明这是**舍入边界翻转**（数值性质），不是寻址/算件错。

用法：
    python3.12 m27_hc_prefill/tools/localize_blk_deviation.py /tmp/m27_out m257
    python3.12 m27_hc_prefill/tools/localize_blk_deviation.py /tmp/m27_out m257 --tag a2

退出码：0 = 报告产出（本脚本**不是**判定项，只出读数）；2 = 缺输入。
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20   # noqa: E402

HC, HID, HYPER, LOWRANK, INJ_N, INJW_SLOT = 4, 2560, 10240, 320, 4, 8
MODE_MIX, MODE_COMBINE_MIX, MODE_COMBINE_ONLY = 0, 1, 3


def load_checker():
    path = os.path.join(REPO, "m27_hc_prefill", "check_ref.py")
    spec = importlib.util.spec_from_file_location("c27loc", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["c27loc"] = mod
    spec.loader.exec_module(mod)
    return mod


def stats(name, got_u16, exp_f64):
    """与 m20.judge 同口径的三件套（逐位率 / ulpMax（良态）/ 归一化绝对误差）。"""
    got_b = np.asarray(got_u16, dtype=np.uint16).reshape(-1)
    exp = np.asarray(exp_f64, dtype=np.float64).reshape(-1)
    exp_b = m20.b16(exp).reshape(-1)
    d = np.abs(m20.ulp_key(got_b).astype(np.int64) - m20.ulp_key(exp_b).astype(np.int64))
    scale = float(np.max(np.abs(exp))) if exp.size else 0.0
    sig = np.abs(exp) > 0.05 * scale
    got_f = m20.f32(got_b).astype(np.float64)
    norm_abs = float(np.max(np.abs(got_f - exp)) / scale) if scale > 0 else 0.0
    print("[loc] %-10s n=%-8d 逐位=%.4f 良态逐位=%.4f ulpMax(良态)=%-3d 归一maxAbs=%.3e  最大差元素占比=%.4f"
          % (name, got_b.size, float(np.mean(d == 0)), (float(np.mean(d[sig] == 0)) if np.any(sig) else 1.0),
             (int(np.max(d[sig])) if np.any(sig) else 0), norm_abs, float(np.mean(d != 0))))
    # 「参考值离最近 bf16 网格点的距离」在"差了的元素"与"没差的元素"上的分布（舍入边界诊断）
    exp32 = exp.astype(np.float32)
    lo = m20.f32(exp_b).astype(np.float64)
    hi = m20.f32(m20.b16(exp + np.spacing(np.abs(exp) + 1e-30) * 4.0)).astype(np.float64)
    near = np.minimum(np.abs(exp - lo), np.abs(hi - exp))
    span = np.maximum(np.abs(exp - lo), np.abs(hi - exp))
    frac = np.where(span > 0, near / np.maximum(span, 1e-300), 0.0)   # 0 = 正好在网格点上
    bad = d != 0
    if np.any(bad) and np.any(~bad):
        print("           舍入边距（0=在网格点上，1=恰好两值正中间）：差的元素 %.3f / 没差的元素 %.3f"
              % (float(np.median(frac[bad])), float(np.median(frac[~bad]))))
    (void := (exp32,))   # 仅为可读性保留
    return float(np.mean(d == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("case")
    ap.add_argument("--tag", default="a1", choices=["a1", "a2"])
    a = ap.parse_args()
    c27 = load_checker()
    D, case, tag = a.dump, a.case, a.tag
    lay = {k: int(v) for k, v in c27.read_kv(os.path.join(D, "m27_layout_%s.txt" % case)).items()}
    meta = dict(zip(open(os.path.join(D, "m27_case_%s.txt" % case)).read().split()[2::2],
                    open(os.path.join(D, "m27_case_%s.txt" % case)).read().split()[3::2]))
    m, mt = int(meta["m"]), lay["mt"]
    if int(meta["mode"]) in (MODE_MIX, MODE_COMBINE_ONLY) or int(meta["arena_dumped"]) == 0:
        print("[loc] SKIPPED：本档（mode=%s / arena_dumped=%s）没有可比的分段链"
              % (meta["mode"], meta["arena_dumped"]))
        return 2
    hin_file = "%s_a1_hin" % case if tag == "a1" else "%s_a2_hin" % case
    ij_file = "%s_a1_ij" % case if tag == "a1" else "%s_a1_ij_handoff" % case
    wtag = "a1" if tag == "a1" else "a2"
    try:
        hin = c27.rd(D, hin_file, np.uint16)
        bo = c27.rd(D, "%s_%s_bo" % (case, tag), np.uint16)
        ij = c27.rd(D, ij_file, np.uint16)
        wdown = c27.rd(D, "w_%s_down" % wtag, np.uint16)
        winj = c27.rd(D, "w_%s_inj" % wtag, np.uint16)
        wup = c27.rd(D, "w_%s_up" % wtag, np.uint16)
        norm = c27.rd(D, "w_%s_norm" % wtag, np.uint16).reshape(-1)
        # 边界 #2 的段内中间量在 **arena2** 里（两个边界各有自己的 arena）
        arena = c27.rd(D, "%s_arena%d" % (case, 1 if tag == "a1" else 2), np.uint8).reshape(-1)
    except m20.MissingInput as e:
        print("[loc] SKIPPED（缺 %s）" % e)
        return 2
    mode = int(meta["mode"])
    ref = m20.reference(m, mode, hin, bo, ij, wdown, winj, wup, norm)

    ts, rhp = lay["tile_stride"], lay["row_hp"]
    # 设备侧中间量（arena 的阻塞布局；地址全部由布局文件算出）
    def blocked(base, row_bytes, rows_per_tile):
        r = np.arange(m, dtype=np.int64)
        idx = (r // mt) * ts + base + (r % mt) * row_bytes
        cols = np.arange(row_bytes, dtype=np.int64)
        return arena[idx[:, None] + cols[None, :]]

    print("== 分段链定位（case=%s tag=%s m=%d mode=%d；参考 = m20_hyperconn/check_ref.py）==" %
          (case, tag, m, mode))
    print("   ⚠ 本脚本只对小 m 档有效（m ≤ 64 ⇒ 块数 ≤ d_clobber）：段内中间量（xn/oh/ls/gate）是"
          "**瞬态**的，\n     arena 里更早块的这些区会被更晚块的写覆盖（见段体头 PROLOGUE ③④）"
          "⇒ m ≥ 257 档的 arena 里已经不是它们的产物了（本脚本会给出 ~0 的逐位率，那是"
          "\"数据早被覆盖\"而不是\"算错\"）。")
    xn_dev = np.ascontiguousarray(blocked(lay["off_xn"], rhp, mt)).view(np.uint16).reshape(m, HYPER)
    oh_dev = np.ascontiguousarray(blocked(lay["off_oh"], lay["oh_w"] * 2, mt)).view(np.uint16).reshape(
        m, lay["oh_w"])
    ls_dev = np.ascontiguousarray(blocked(lay["off_ls"], LOWRANK * 2, mt)).view(np.uint16).reshape(
        m, LOWRANK)
    gate_dev = np.ascontiguousarray(blocked(lay["off_gate"], HYPER * 2, mt)).view(np.uint16).reshape(
        m, HYPER)
    # 可读的 stage：`xn`（d_clobber=8）、`oh/lora`（16）、`ls`（16）—— m ≤ 64 时它们的块内内容还在。
    # `gate` **不可读**：它的内容（`[WS_GATE, +mt*ROW_HP)`）会被**下一块**的 OH/LS 写覆盖（d=1）
    # ⇒ 循环跑完后 arena 里已经不是它的产物（本脚本照抽会给出 ~0 的逐位率与巨大的归一化误差，
    # 那是"数据早被覆盖"，不是"算错"）。要把它当判据，只能在块内 dump（本 mission 未做）。
    stats("xn", xn_dev, ref["xn"])
    stats("lora", oh_dev[:, :LOWRANK], ref["lora"][:, :LOWRANK])
    stats("ls", ls_dev, ref["ls"])
    print("[loc] gate       —— **不可读**（d_clobber=1：它的区被下一块的 OH/LS 写覆盖；"
          "块内 dump 才能读，本 mission 未做）")
    blk_dev = c27.rd(D, "%s_%s_blk" % (case, tag), np.uint16)
    stats("blk", blk_dev[:m], ref["blk"])
    print("[loc] 结论口径（不越界）：本工具能给的 = **`xn`/`lora`/`ls` 三个 stage 与参考的位级距离**；\n"
          "      `blk` 的余差**没有**在本工具里定位到 stage（`gate` 不可读，见上）。")
    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
