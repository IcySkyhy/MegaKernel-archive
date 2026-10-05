#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m15_layer_loop/check_hc_ref.py —— M58 融合 hc 段的**独立 numpy 交叉校验**

用 **m20 的独立参考**（`m20_hyperconn/check_ref.py` 的 `reference()`，纯 numpy float64 逐句实现
docs/14 §3.2 / vLLM 的 `hc.py` 语义 + `judge()` 的保守门限）复算 m15 **四相位融合 kernel** 里
两个 hc 边界的每一个中间张量。**不另造一套参考**（与 `check_moe_ref.py` 复用 m13 的参考链同款）：
本脚本 `import` m20 的模块，`judge()` / 门限 / 计数全部由那一份代码产出。

用法（在 dump 目录里跑；先用 M15_DUMP=1 落盘）：

    mkdir -p /tmp/m58_dump && cd /tmp/m58_dump
    M15_DUMP=1 M15_LAYERS=4 M15_HC_LAYERS=0,1,3 <repo>/m15_layer_loop/build/m15_layer_loop \
        <repo>/m15_layer_loop/weights_manifest.txt h
    /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_hc_ref.py

## 两档判据（task5 的「逐段 + 整层两条都要」）

    A. **逐段**（device-anchored）：以**设备自己的上游 dump** 为输入逐段复算 —— 把端到端残差
       按段归因（边界 #1 的 combine/norm/down/silu/up/gate-mix；边界 #2 同上）。
    B. **整层**（reference-chained）：从**层输入**（H, BO_prev, IJ_prev）出发，用参考算出边界 #1
       的 H' 与 injection，再把它们喂给边界 #2 的参考 —— 即「整层的 hc 数据通路」由**参考自己**
       串起来，与设备的 `H''` / `BLK_mlp` / injection 输出对拍。唯一的设备锚点是子层段出口
       （attention/GDN 段的正确性属于 M25/M40 的链，不在 hc 参考的模型里）。
    C. **combine-only 档**：设备 `M15H::MODE_COMBINE_ONLY` 入口的 H' 与参考的 combine 输出对拍
       （m20 的三档 mode 都不提供这一档，见 README 的披露项）。
    D. **非空洞性 / 可区分性**（guard）：H''/blk 非恒等且非常量、跨层 blk 指纹两两互异。

## 退出码（三态，tower 规则）

    0 = 比过且通过（OK，文案里带实际比较条数）
    1 = 比过且有差异（FAILED，列出判定项名）
    2 = 没得比 / 输入缺失（SKIPPED）——「我没比」与「我比过通过」必须能区分

## 覆盖范围交代（tower 规则：统计工具必须交代自己真正检查了什么）

* 覆盖 = `M15_HC_LAYERS` 列出的层（默认 0,1,3）× 两个边界 × 该边界在 dump 里有的张量；
  实际比较条数由**每个判据调用点自增**同一个计数器产出（不手写数字）。
* **逐段 A** 的输入锚点 = 设备 dump（因此它能抓住「某一段算错」而不是「误差传递」）；
  **整层 B** 的输入锚点 = 参考链（因此它不依赖设备边界 #1 的正确性）。
* **本脚本不覆盖**：① 子层段（GDN/attention）与 MoE 段的数值 —— 它们由 M25 的 `check_ref.py`
  与 M40 的 `check_moe_ref.py` 覆盖；② 三相位/四相位的**资源叠放与同步**正确性 —— 它由
  kernel 内的 `H.h1ws`/`H.h2ws`（与独立启动逐字节）覆盖，不在本脚本的模型里；
  ③ 跨层 handoff（H''/injection 传给下一层）—— 本 milestone 未接线（README 披露项）。

**负向对照**（必须做：喂它空输入，确认它不发合格证）：

    mkdir -p /tmp/m58_empty && cd /tmp/m58_empty
    /usr/local/python3.12.13/bin/python3 <repo>/m15_layer_loop/check_hc_ref.py; echo "rc=$?"
    # 期望：RESULT: SKIPPED（缺 hc_L00_hin.bin）且 rc=2
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m20_hyperconn"))

import check_ref as m20  # noqa: E402  —— m20 的独立参考（reference/judge/阈值/计数）

HC = m20.HC
HID = m20.HID
HYPER = m20.HYPER
LOWRANK = m20.LOWRANK
M_MAX = m20.M_MAX
INJ_N = m20.INJ_N
IJ_STRIDE = m20.IJ_STRIDE
INJW_SLOT = m20.INJW_SLOT
MODE_MIX, MODE_COMBINE_MIX = m20.MODE_MIX, m20.MODE_COMBINE_MIX

COUNT = {"judge": 0, "guard": 0, "skipped": 0}
FAILS = []


class MissingInput(Exception):
    pass


def rd(name, dtype, shape):
    path = name + ".bin"
    if not os.path.exists(path):
        raise MissingInput(path)
    a = np.fromfile(path, dtype=dtype)
    if a.size != int(np.prod(shape)):
        raise MissingInput("%s 字节数 %d != 期望 %d（shape %s）"
                           % (path, a.size, int(np.prod(shape)), shape))
    return a.reshape(shape)


def bf16(name, shape):
    return rd(name, np.uint16, shape)


def f32b(name, shape):
    return rd(name, np.float32, shape)


def J(name, got_u16, exp_f64):
    """bf16 张量的判定项 —— 门限与计数全部来自 m20 的 judge()（同一份代码）。"""
    ok = m20.judge(name, got_u16, exp_f64, {})
    COUNT["judge"] += 1
    if not ok:
        FAILS.append(name)
    return ok


def JR(name, got_f32, exp_f64, tol):
    """fp32 中间量（rstd / injw）的相对误差判据（与 m20 的 tol 同口径）。"""
    rel = float(np.max(np.abs(got_f32.astype(np.float64) - exp_f64) /
                       np.maximum(np.abs(exp_f64), 1e-30)))
    COUNT["judge"] += 1
    ok = rel <= tol
    if not ok:
        FAILS.append(name)
    print("[chk] 判定 %-32s maxRel=%.3e (tol %.0e)  %s" % (name, rel, tol, "PASS" if ok else "FAIL"))
    return ok


def G(name, cond, detail=""):
    """guard（结构性前提 / 非空洞性）：失败仍计入 FAILS，但单列计数。"""
    COUNT["guard"] += 1
    if not cond:
        FAILS.append(name)
    print("[chk] guard %-31s %s  %s" % (name, detail, "OK" if cond else "DEGENERATE"))
    return cond


def load_layout(path):
    """hc_layout.txt 的每行是 `k=v` 或一行多个 `k=v`（用空格分隔）；只取数值项。"""
    if not os.path.exists(path):
        raise MissingInput(path)
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)", line):
                d[k] = int(v)
    return d


def slice_ws(tag, lay, m, cfg):
    """从一个边界的 8 个中间张量里读 m 行（dump 时按各自行距截了 m 行，故此处直接 reshape）。"""
    hyb = m * HYPER
    return {
        "hcp": bf16("%s_hcp" % tag, (m, HYPER)),
        "xn": bf16("%s_xn" % tag, (m, HYPER)),
        "rstd": f32b("%s_rstd" % tag, (m, HC)),
        "injw": f32b("%s_injw" % tag, (m, HC, INJW_SLOT)),
        "oh": bf16("%s_oh" % tag, (m, m20.OH_W)),
        "ls": bf16("%s_ls" % tag, (m, LOWRANK)),
        "gate": bf16("%s_gate" % tag, (m, HYPER)),
        "blk": bf16("%s_blk" % tag, (m, HID)),
    }


def judge_segment(pref, dev, ref, mode, extra=""):
    """逐段判据（与 m20 的 A/B 段同一口径与同一张量集合）。"""
    if mode != MODE_MIX:
        J("%sH'<-ij,hin,bo%s" % (pref, extra), dev["hcp"], ref["hc"])
        ijw_dev = dev["injw"][:, :, 0].reshape(-1)
        JR("%sinjW<-ij%s" % (pref, extra), ijw_dev, ref["injw"].reshape(-1), 1e-6)
    else:
        # mode 0 不做 combine ⇒ H' ≡ H、injW 不参与（显式 SKIPPED，不静默）
        print("[chk] 判定 %-32s mode 0（mix-only）不做 combine → SKIPPED（unverified）" % (pref + "H'"))
        COUNT["skipped"] += 1
    JR("%srstd<-hin/h'%s" % (pref, extra), dev["rstd"], ref["rstd"], 1e-4)
    J("%sxn<-h'/%s hc_norm%s" % (pref, "" if mode != MODE_MIX else "H,", extra), dev["xn"], ref["xn"])
    J("%slora<-xn,wdown%s" % (pref, extra), dev["oh"][:, :LOWRANK], ref["lora"][:, :LOWRANK])
    J("%sinject<-xn,winj%s" % (pref, extra), dev["oh"][:, LOWRANK:LOWRANK + INJ_N],
      ref["lora"][:, LOWRANK:])
    J("%sls<-lora%s" % (pref, extra), dev["ls"], ref["ls"])
    J("%sgate<-ls,wup%s" % (pref, extra), dev["gate"], ref["gate"])
    J("%sblk<-gate,xn%s" % (pref, extra), dev["blk"], ref["blk"])


def ref_ij_plane(ij_src, m):
    """把「参考的 injection 列」做成 reference() 需要的 (m,IJ_STRIDE) 平面（前 4 列有效）。"""
    pl = np.zeros((m, IJ_STRIDE), dtype=np.float64)
    pl[:, :INJ_N] = ij_src
    return m20.b16(pl).astype(np.uint16)


def main():
    try:
        return _main()
    except MissingInput as e:
        print("[chk] RESULT: SKIPPED (缺输入文件：%s)\n"
              "[chk] 先在空目录跑一次 `M15_DUMP=1 M15_LAYERS=4 M15_HC_LAYERS=0,1,3 <bin> "
              "<manifest> h` 再复核。" % e)
        return 2


def _main():
    layout = load_layout("hc_layout.txt")
    m = int(os.environ.get("M15_HC_M", "1"))
    layers_env = os.environ.get("M15_HC_LAYERS", "0,1,3")
    layers = [int(x) for x in re.split(r"[,\s]+", layers_env.strip()) if x != ""]

    print("[chk] m15 hc 独立交叉校验（参考 = m20_hyperconn/check_ref.py 的 reference()）")
    print("[chk] 层表 = %s；m = %d；ws 布局来自 hc_layout.txt（由 C++ 常量算出，与 kernel 同源）"
          % (layers, m))
    print("[chk] 布局键：%s" % ", ".join("%s=%d" % (k, v) for k, v in sorted(layout.items())))

    fp = []   # 跨层 blk 指纹（可区分性 guard）

    for L in layers:
        base = "hc_L%02d" % L
        attn_mode = MODE_MIX if L == 0 else MODE_COMBINE_MIX
        print("\n===== 层 %02d（attn 边界 mode=%d，mlp 边界 mode=%d）====="
              % (L, attn_mode, MODE_COMBINE_MIX))

        hin = bf16(base + "_hin", (m, HYPER))
        bo = bf16(base + "_bo", (m, HID))
        ij = bf16(base + "_ij", (m, IJ_STRIDE))
        attnout = bf16(base + "_attnout", (m, HID))

        WA = {k: bf16("%s_attn_%s" % (base, k), sh) for k, sh in
              (("norm", (HYPER,)), ("wdown", (LOWRANK, HYPER)), ("winj", (16, HYPER)),
               ("wup", (HYPER, LOWRANK)))}
        WM = {k: bf16("%s_mlp_%s" % (base, k), sh) for k, sh in
              (("norm", (HYPER,)), ("wdown", (LOWRANK, HYPER)), ("winj", (16, HYPER)),
               ("wup", (HYPER, LOWRANK)))}

        d1 = slice_ws(base + "_b1", L, m, layout)
        d2 = slice_ws(base + "_b2", L, m, layout)

        # ---------- A. 逐段（device-anchored）----------
        print("\n-- A1. 边界 #1 逐段（输入 = 设备自己的 dump）--")
        r1 = m20.reference(m, attn_mode, hin, bo, ij, WA["wdown"], WA["winj"], WA["wup"], WA["norm"])
        judge_segment("A1.", d1, r1, attn_mode)

        print("\n-- A2. 边界 #2 逐段（输入 = 设备边界 #1 的 dump + 子层段出口）--")
        hin2 = d1["hcp"] if attn_mode != MODE_MIX else hin
        ij2 = ref_ij_plane(m20.f32(d1["oh"][:, LOWRANK:LOWRANK + INJ_N]), m)
        r2 = m20.reference(m, MODE_COMBINE_MIX, hin2, attnout, ij2,
                           WM["wdown"], WM["winj"], WM["wup"], WM["norm"])
        judge_segment("A2.", d2, r2, MODE_COMBINE_MIX)

        # ---------- B. 整层（reference-chained）----------
        # 从层输入出发：参考的 H' / injection 喂给边界 #2；唯一设备锚点 = 子层段出口。
        print("\n-- B. 整层链（参考自己串 hc 的两个边界；锚点 = 层输入 + 子层段出口）--")
        ij2r = ref_ij_plane(r1["lora"][:, LOWRANK:], m)
        hc1_ref = m20.b16(r1["hc"]).astype(np.uint16)
        r2c = m20.reference(m, MODE_COMBINE_MIX, hc1_ref, attnout, ij2r,
                            WM["wdown"], WM["winj"], WM["wup"], WM["norm"])
        print("[chk]   （整层：H'' = 参考链边界 #2 的 h'；BLK_mlp = 它的 blk；injection = 它的 lora[:,320:]）")
        J("B整层H''<-层输入链", d2["hcp"], r2c["hc"])
        J("B整层blk_mlp<-层输入链", d2["blk"], r2c["blk"])
        J("B整层injection<-层输入链", d2["oh"][:, LOWRANK:LOWRANK + INJ_N], r2c["lora"][:, LOWRANK:])
        # 层出口（dump 的 hout 就是 hc1 ws 的 WS_HCP 的 m 行）。
        # **报告项，不计入判定项/guard**：它与 b2_hcp 是同一块缓冲、同一批字节，因此这条只
        # 自检「层出口住在 hc 边界 #2 的 H' 平面里」这个布局契约；数值正确性由 B 段的三条判据覆盖。
        hout = bf16(base + "_hout", (m, HYPER))
        print("[chk] 报告 %-32s hout(=WS_HCP) 与 b2_hcp 逐字节一致 = %s（布局契约自检，不计入计数）"
              % ("B层出口==H''", bool(np.array_equal(hout.view(np.uint8), d2["hcp"].view(np.uint8)))))

        # ---------- C. combine-only 档（第 4 档 mode）----------
        print("\n-- C. combine-only 档（MODE_COMBINE_ONLY；m20 三档都不提供）--")
        rc = m20.reference(m, MODE_COMBINE_MIX, hin, bo, ij, WA["wdown"], WA["winj"], WA["wup"],
                           WA["norm"])
        conly = bf16(base + "_conly_hcp", (m, HYPER))
        J("C combine-only H'<-H,BO,IJ", conly, rc["hc"])

        # ---------- D. 非空洞性 / 可区分性 ----------
        print("\n-- D. 非空洞性 / 可区分性（guard）--")
        G("%d H'' 非常量" % L, int(np.unique(d2["hcp"]).size) > 1,
          "取值数=%d" % int(np.unique(d2["hcp"]).size))
        G("%d BLK_mlp 非常量" % L, int(np.unique(d2["blk"]).size) > 1,
          "取值数=%d" % int(np.unique(d2["blk"]).size))
        G("%d H''≠层输入 H" % L, not np.array_equal(d2["hcp"].view(np.uint8),
                                                   hin[:m].view(np.uint8)),
          "整层不是恒等映射")
        G("%d 两个边界 blk 不同" % L, not np.array_equal(d1["blk"].view(np.uint8),
                                                        d2["blk"].view(np.uint8)),
          "attn 边界与 mlp 边界的 block input 不同")
        fp.append((L, bytes(d2["blk"].view(np.uint8))))

    # 跨层可区分性（索引/槽位不错位）
    print("\n-- D2. 跨层可区分性 --")
    for i in range(len(fp)):
        for j in range(i + 1, len(fp)):
            G("跨层 blk 指纹 L%02d≠L%02d" % (fp[i][0], fp[j][0]), fp[i][1] != fp[j][1],
              "两层权重槽没有错位到同一个摘要")

    # ---------- 汇总（三态 + 覆盖计数同源）----------
    compared = COUNT["judge"] + COUNT["guard"]
    print("\n[chk] 覆盖：判定项 %d 条 + guard %d 条（共比较 %d 条）；unverified/SKIPPED %d 条"
          % (COUNT["judge"], COUNT["guard"], compared, COUNT["skipped"]))
    print("[chk] 其中由 m20 的 judge() 直接产出的条数 = %d（同一份门限代码）" % m20.JUDGE[0])
    if FAILS:
        print("[chk] RESULT: FAILED (%d 条不符：%s)" % (len(FAILS), FAILS))
        return 1
    if compared == 0:
        print("[chk] RESULT: SKIPPED (没有可比对的输入)")
        return 2
    print("[chk] RESULT: OK (%d/%d 条比较过：判定项 %d + guard %d；SKIPPED %d)"
          % (compared, compared, COUNT["judge"], COUNT["guard"], COUNT["skipped"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
