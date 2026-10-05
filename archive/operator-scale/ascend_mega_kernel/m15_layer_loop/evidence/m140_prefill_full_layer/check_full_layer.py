#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M140 whole-layer GDN prefill —— hc 四相位（H1/H2）的独立 numpy 交叉校验。

用法（先跑设备落盘，见 `reproduce.sh`）：
    python3.12 m15_layer_loop/evidence/m140_prefill_full_layer/check_full_layer.py <dumpdir> <tag> <m>

## 判据来源（不自己造第二套数学）
参考 = `m20_hyperconn/check_ref.py::reference`（独立 numpy float64，逐句实现 hc 的伪码；
`reference(m, mode, hin, bo, ij, wdown, winj, wup, norm)`）；门限 = 同一文件的 `judge()`
（张量尺度归一化绝对误差 ≤1e-2 + 良态元素 bf16 ulp ≤2 且逐位一致率 ≥99%）。本脚本不含第二份 hc 数学。

## 判什么
  · H1（边界 #1）：输入 = dump 的 `h1_hin/h1_bo/h1_ij`（host 合成激活）+ 真权重 `w_a1_*`；
    输出 = 设备的 `h1_hcp`（H' 平铺面）/ `h1_blk`（重定位出的 BLK 平铺面）。
  · H2（边界 #2）：输入 = 设备的 `h1_hcp`（层内 handoff，零拷贝）+ `h2_bo`（子层段出口）+ `h1_ij_handoff`；
    输出 = 设备的 `h2_hcp` / `h2_blk`。
  · 两个边界都取 `MODE_COMBINE_MIX`（见 `.asc` 的注释：H1 必须物化 H' 才能给 H2 当 hIn）。

## 明确不判的
  · `injw` / `rstd` 两个证据面：挂载点按 m27 README §5(b) 的"最小新增"传 nullptr ⇒ 不重定位成平铺面，
    本脚本不比较它们（覆盖范围 = hcp + blk，它们已经覆盖 injW→combine→norm→down/inj→silu→up→gate mix 全链）。
  · 相位 B（MoE）的数值：本脚本不判（段体数学由 `m26_moe_prefill` 独立验证）。

## 退出码（三态，tower 规则）
  0 = 比过且通过；1 = 比过且有差异；2 = 没得比 / 输入缺失（SKIPPED）
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "m20_hyperconn"))
import check_ref as m20  # noqa: E402

HC = 4
HID = 2560
HYPER = HC * HID
LOWRANK = 320
INJ_N = 4
IJ_STRIDE = 16
M_MAX = 64
MODE_COMBINE_MIX = 1


def rd(dump, name, dtype, elems):
    path = os.path.join(dump, name + ".bin")
    if not os.path.exists(path):
        raise m20.MissingInput(path)
    a = np.fromfile(path, dtype=dtype)
    if a.size != elems:
        raise SystemExit("FAIL: %s short (%d != %d)" % (path, a.size, elems))
    return a


# ---- hc 权重的**来源兜底**（M140 复审 r1 P2-2）----
# 权重平面（每边界 ≈ 13.5 MB × 2）不入 git（源 = checkpoint，可重建）⇒ 当 dump 里没有它们时，
# 直接从 `m15_layer_loop/weights_manifest.txt` + checkpoint 取（**与设备同一条来源**：host 也是这么
# 读进槽的；`inj` 在 checkpoint 里只有 4 行，host 把 [4,16) 行置零，这里照同一口径补）。
MANIFEST = os.path.join(REPO, "m15_layer_loop", "weights_manifest.txt")
_HC_ROLE = {"a1": "attn_hc", "a2": "mlp_hc"}


def _manifest_tensor(role, layer=0):
    model_dir = None
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line.startswith("model_dir="):
                model_dir = line.split("=", 1)[1]
                continue
            if not line.startswith("tensor "):
                continue
            kv = {}
            for tok in line[7:].split(" "):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            if kv.get("role") == role and int(kv.get("layer", -1)) == layer:
                return model_dir, kv
    return None, None


def _read_tensor(role, nbytes, layer=0):
    model_dir, kv = _manifest_tensor(role, layer)
    if kv is None:
        raise m20.MissingInput("manifest 缺 role=%s layer=%d" % (role, layer))
    if int(kv["bytes"]) != nbytes:
        raise SystemExit("FAIL: role=%s 字节数不符：manifest %s，期望 %d" % (role, kv["bytes"], nbytes))
    path = os.path.join(model_dir, kv["file"])
    if not os.path.exists(path):
        raise m20.MissingInput(path)
    with open(path, "rb") as f:
        f.seek(int(kv["offset"]))
        buf = f.read(nbytes)
    if len(buf) != nbytes:
        raise SystemExit("FAIL: role=%s 短读 %d/%d" % (role, len(buf), nbytes))
    return np.frombuffer(buf, dtype="<u2")


def load_w(dump, tag, wtag):
    """优先读 dump 的权重面；缺失则从 manifest + checkpoint 重建（口径同 host 的装载路径）。"""
    names = ["w_%s_down" % wtag, "w_%s_inj" % wtag, "w_%s_up" % wtag, "w_%s_norm" % wtag]
    if all(os.path.exists(os.path.join(dump, "m140_%s_%s.bin" % (tag, nm))) for nm in names):
        wdown = rd(dump, "m140_%s_w_%s_down" % (tag, wtag), np.uint16, LOWRANK * HYPER)
        winj = rd(dump, "m140_%s_w_%s_inj" % (tag, wtag), np.uint16, 16 * HYPER)
        wup = rd(dump, "m140_%s_w_%s_up" % (tag, wtag), np.uint16, HYPER * LOWRANK)
        norm = rd(dump, "m140_%s_w_%s_norm" % (tag, wtag), np.uint16, HC * HID)
        return wdown.reshape(LOWRANK, HYPER), winj.reshape(16, HYPER), wup.reshape(HYPER, LOWRANK), \
            norm.reshape(HC, HID)
    pre = _HC_ROLE[wtag]
    wdown = _read_tensor("%s_down" % pre, LOWRANK * HYPER * 2)
    wup = _read_tensor("%s_up" % pre, HYPER * LOWRANK * 2)
    norm = _read_tensor("%s_norm" % pre, HYPER * 2)
    inj4 = _read_tensor("%s_inj" % pre, HYPER * 2 * INJ_N)
    winj = np.zeros(16 * HYPER, dtype=np.uint16)   # [4,16) 行置零（与 host 的 H_LoadHcW 同口径）
    winj[: INJ_N * HYPER] = inj4
    return wdown.reshape(LOWRANK, HYPER), winj.reshape(16, HYPER), wup.reshape(HYPER, LOWRANK), \
        norm.reshape(HC, HID)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    dump = sys.argv[1]
    tag = sys.argv[2]
    m = int(sys.argv[3])
    phases = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    if not (phases & 1) and not (phases & 4):
        print("[chk] SKIPPED：phases=0x%X 没有 H1/H2（例如只开相位 A/B 的隔离档）" % phases)
        return 2
    try:
        if phases & 1:
            hin = rd(dump, "m140_%s_h1_hin" % tag, np.uint16, m * HYPER).reshape(m, HYPER)
            bo = rd(dump, "m140_%s_h1_bo" % tag, np.uint16, m * HID).reshape(m, HID)
            ij = rd(dump, "m140_%s_h1_ij" % tag, np.uint16, m * IJ_STRIDE).reshape(m, IJ_STRIDE)
            hcp1 = rd(dump, "m140_%s_h1_hcp" % tag, np.uint16, m * HYPER).reshape(m, HYPER)
            blk1 = rd(dump, "m140_%s_h1_blk" % tag, np.uint16, m * HID).reshape(m, HID)
            ijh = rd(dump, "m140_%s_h1_ij_handoff" % tag, np.uint16, (m + M_MAX) * IJ_STRIDE)[: m * IJ_STRIDE]
            ijh = ijh.reshape(m, IJ_STRIDE)
            w1 = load_w(dump, tag, "a1")
        if phases & 4:
            hcp1 = rd(dump, "m140_%s_h1_hcp" % tag, np.uint16, m * HYPER).reshape(m, HYPER)
            h2bo = rd(dump, "m140_%s_h2_bo" % tag, np.uint16, m * HID).reshape(m, HID)
            hcp2 = rd(dump, "m140_%s_h2_hcp" % tag, np.uint16, m * HYPER).reshape(m, HYPER)
            blk2 = rd(dump, "m140_%s_h2_blk" % tag, np.uint16, m * HID).reshape(m, HID)
            ijh = rd(dump, "m140_%s_h1_ij_handoff" % tag, np.uint16, (m + M_MAX) * IJ_STRIDE)[: m * IJ_STRIDE]
            ijh = ijh.reshape(m, IJ_STRIDE)
            w2 = load_w(dump, tag, "a2")
    except m20.MissingInput as e:
        print("[chk] SKIPPED：缺输入 %s（该 tag 没有四相位 dump ⇒ 相位未开）" % e)
        return 2

    print("[chk] M140 hc 四相位交叉校验：dump=%s tag=%s m=%d phases=0x%X（参考 = m20 独立 numpy float64）"
          % (dump, tag, m, phases))
    if phases & 1:
        ref1 = m20.reference(m, MODE_COMBINE_MIX, hin, bo, ij, w1[0], w1[1], w1[2], w1[3])
        m20.judge("h1.hcp", hcp1.reshape(-1), ref1["hc"].reshape(-1), {})
        m20.judge("h1.blk", blk1.reshape(-1), ref1["blk"].reshape(-1), {})
    if phases & 4:
        # H2 的 hIn = 设备自己的 H1 产物（层内 handoff）；bo = 子层段出口（见 evidence 的生产者缺口）
        ref2 = m20.reference(m, MODE_COMBINE_MIX, hcp1, h2bo, ijh, w2[0], w2[1], w2[2], w2[3])
        m20.judge("h2.hcp", hcp2.reshape(-1), ref2["hc"].reshape(-1), {})
        m20.judge("h2.blk", blk2.reshape(-1), ref2["blk"].reshape(-1), {})

    # ---- guard（非空洞性，不计入判定项）----
    g = []
    if phases & 1:
        g.append(("h1_hin 非常量", np.ptp(m20.f32(hin.reshape(-1))) > 0.0))
        g.append(("h1_hcp 非常量", np.ptp(m20.f32(hcp1.reshape(-1))) > 0.0))
        g.append(("h1_hcp != h1_hin", not np.array_equal(hcp1, hin)))
        g.append(("h1_blk 非常量", np.ptp(m20.f32(blk1.reshape(-1))) > 0.0))
    if phases & 4:
        g.append(("h2_blk 非常量", np.ptp(m20.f32(blk2.reshape(-1))) > 0.0))
    for name, ok in g:
        print("[chk] guard %-22s %s" % (name, "OK" if ok else "FAIL"))
    bad = [n for n, ok in g if not ok]

    fails = list(m20.FAILS)
    print("[chk] 判定项 %d 条，FAIL %d 条%s；guard 失败 %d 条" %
          (m20.JUDGE[0], len(fails), ("：" + ",".join(fails)) if fails else "", len(bad)))
    if fails or bad:
        print("RESULT: FAILED")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
