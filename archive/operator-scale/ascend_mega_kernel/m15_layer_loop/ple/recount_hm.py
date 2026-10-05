#!/usr/bin/env python3.12
"""recount_hm.py —— M92：**重算** `ple/REAL_TABLE.md` §6.2 / §6.3 里的每一个数字（只读，不下 PASS/FAIL）

本脚本**不是判据**（不产出 `RESULT|` 行、不改退出码语义）：判据在 `m15_ple_check.py` 的 `check_hm`。
它的作用只有一个 —— 让 §6.2 / §6.3 的读数**有一条可复现的命令**（口径与 `cmp_t3_masked` 对齐：
同样的 mask、同样的 `bf_ulp`）：
  · §6.2 ①：`H3.kv` 在「代理界 `max(|out|,|ref|)`」与「`Σ|terms|`」下的 bad / maxRel / max(d/bound)，
    以及被代理界打红的那几个元素的 `|out|/Σ|terms|` 与 `d/(ε·Σ|terms|)`（证明它们在相消区）；
  · §6.2 ②：`H4.gated` / `H4.normed` / `H5.out` 的「逐位不同元素数」「`d/ulp` 分布」
    以及 `ulp_mult ∈ {1.0, 2.0}` 下的 bad / max(d/bound)；
  · §6.3：④⑤ 的参考链吃 `kv_ref`（现行判据）vs 吃设备 `H_kv.bin`（诊断口径）的对照
    —— 后者不参与任何 PASS/FAIL，只为说明"参考吃设备自己的上游产物会让上游错误共模"。

用法（仓库根目录；可对基线或负向的 outdir 各跑一次）：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/recount_hm.py m15_layer_loop/ple/out_hm
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/recount_hm.py m15_layer_loop/ple/out_hm_neg
"""
import argparse
import importlib.util
import pathlib
import sys
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # m15_layer_loop/ple/
CHECK = HERE.parent / "m15_ple_check.py"


def load_check():
    """复用判据脚本的**常量与 helper**（`bf_ulp` / `bf16_rne` / `ref_gate` / `ref_conv_out` /
    `real_row` / `hm_meta` 与 `EPS_*`）—— 口径与判据同源，避免"重算脚本自己又写一套口径"。"""
    spec = importlib.util.spec_from_file_location("m15_ple_check", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ulp_dist(d_over_ulp, top=8):
    c = Counter(np.round(d_over_ulp[d_over_ulp > 0], 10).tolist())
    items = sorted(c.items(), key=lambda kv: -kv[1])
    return items[:top], len(items)


def diag(m, name, got, ref, eps, mask, extra=None):
    sub = got[mask].astype(np.float64)
    rsub = ref[mask].astype(np.float64)
    grid = np.maximum(np.abs(sub), np.abs(rsub))
    scale = "max(|out|,|ref|) 代理"
    if extra is not None:
        grid = np.maximum(grid, np.asarray(extra, dtype=np.float64)[mask])
        scale = "Σ|terms|"
    ulp = m.bf_ulp(sub)
    d = np.abs(sub - rsub)
    top, ndistinct = ulp_dist(d / ulp)
    print("[%s] n=%d  界尺度=%s" % (name, sub.size, scale))
    print("        与参考逐位不同的元素 = %d" % int((d > 0).sum()))
    print("        d/ulp>0 分布(按个数取前 %d，共 %d 种取值) = %s" % (len(top), ndistinct, top))
    for um in (1.0, 2.0):
        b = eps * grid + um * ulp
        print("        ulp_mult=%.1f ⇒ bad=%d  max(d/bound)=%.6f  maxRel=%.4g"
              % (um, int((d > b).sum()), float(np.max(d / np.maximum(b, 1e-300))),
                 float(np.max(d / np.maximum(np.abs(rsub), 1e-30)))))
    return sub, rsub, d, ulp, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    args = ap.parse_args()
    out = pathlib.Path(args.outdir)
    m = load_check()
    HM, W, HE, HDIM, KVW, HC, HID, KCONV, STLEN = (m.HM, m.W, m.HE, m.HDIM, m.KVW, m.HC, m.HID,
                                                   m.KCONV, m.STLEN)
    meta = m.hm_meta()
    hm = dict(l.split(" ", 1) for l in (out / "H_hm_meta.txt").read_text().strip().split("\n")
              if " " in l)
    ng, T = int(hm["ng"]), int(hm["n_tok"])
    items = T * ng
    win_base, win_rows = int(hm["r0"]), int(hm["win_rows"])
    ids = np.fromfile(out / "H_ids.bin", dtype=np.int64).reshape(T, ng)
    inw = (ids >= win_base) & (ids < win_base + win_rows)
    tok_in = inw.all(axis=1)
    print("outdir=%s  n_tok=%d  items=%d  win_rows=%d  窗内 item=%d  参与 token=%d"
          % (out, T, items, win_rows, int(inw.sum()), int(tok_in.sum())))

    # ---- 参考链（纯 host 侧）：直读分片文件 → 行 → emb → kv → gated/normed → out ----
    rows_ref = np.zeros((items, HDIM), dtype=np.float32)
    for t in range(T):
        for g in range(ng):
            if inw[t, g]:
                rows_ref[t * ng + g] = m.bf16_to_f32(m.real_row(meta, int(ids[t, g])))
    emb_ref = np.zeros((T, HE), dtype=np.float32)
    for t in range(T):
        for g in range(ng):
            emb_ref[t, g * HDIM:(g + 1) * HDIM] = rows_ref[t * ng + g]

    emb_got = m.bf16_to_f32((out / "H_emb.bin").read_bytes()).reshape(T, HE)
    sel = np.zeros((T, HE), dtype=bool)
    for g in range(ng):
        sel[:, g * HDIM:(g + 1) * HDIM] = inw[:, g][:, None]
    print("\n== H2.emb（T1 逐字节）==")
    print("  对照行非零元素 = %d / %d；设备与对照全等 = %s；不同 = %d"
          % (int((emb_ref[sel] != 0).sum()), int(sel.sum()), bool((emb_got[sel] == emb_ref[sel]).all()),
             int((emb_got[sel] != emb_ref[sel]).sum())))

    wd = m.PLE / "data"
    nk = m.bf16_to_f32((wd / "w_norm_key.bin").read_bytes())
    nq = m.bf16_to_f32((wd / "w_norm_query.bin").read_bytes())
    ncw = m.bf16_to_f32((wd / "w_norm_conv.bin").read_bytes())
    wconv = m.bf16_to_f32((wd / "w_conv1d_sq.bin").read_bytes()).reshape(W, KCONV)
    wkey = m.bf16_to_f32((wd / "w_key_proj.bin").read_bytes()).reshape(W, HE)
    wval = m.bf16_to_f32((wd / "w_value_proj.bin").read_bytes()).reshape(HE, HE)
    hidden = m.bf16_to_f32((HM / "hidden_hm.bin").read_bytes()).reshape(T, W)
    state = m.bf16_to_f32((HM / "conv_state_hm.bin").read_bytes()).reshape(-1, STLEN, W)
    sidx = np.fromfile(HM / "state_idx_hm.bin", dtype=np.int32)

    wcat = np.concatenate([wkey, wval], axis=0)
    awcat = np.abs(wcat.astype(np.float64))
    kv_got = m.bf16_to_f32((out / "H_kv.bin").read_bytes()).reshape(T, KVW)
    kv_ref = np.zeros((T, KVW), dtype=np.float32)
    kv_sumabs = np.zeros((T, KVW), dtype=np.float64)
    for t in range(T):
        e = emb_ref[t].astype(np.float64)
        kv_ref[t] = m.bf16_rne(wcat.astype(np.float64) @ e)
        kv_sumabs[t] = awcat @ np.abs(e)

    print("\n== §6.2 ①  H3.kv：代理界 vs Σ|terms| ==")
    sub, rsub, d, ulp, gridp = diag(m, "H3.kv 代理界", kv_got, kv_ref, m.EPS_REDUCE, tok_in, None)
    diag(m, "H3.kv Σ|terms| 界", kv_got, kv_ref, m.EPS_REDUCE, tok_in, kv_sumabs)
    sa = kv_sumabs[tok_in]
    badp = d > (m.EPS_REDUCE * gridp + 1.0 * ulp)
    if badp.any():
        print("        代理界打红的 %d 个元素（占掩码 %.4g%%）：|out|/Σ|terms| ∈ [%.3g, %.3g]；"
              "d/(ε·Σ|terms|) max = %.3g"
              % (int(badp.sum()), 100.0 * badp.sum() / sub.size,
                 float(np.min(np.abs(sub)[badp] / sa[badp])),
                 float(np.max(np.abs(sub)[badp] / sa[badp])),
                 float(np.max(d[badp] / (m.EPS_REDUCE * sa[badp])))))

    eps4 = m.EPS_REDUCE + m.EPS_RSQRT + m.EPS_SIGMOID
    got_g = m.bf16_to_f32((out / "H_gated.bin").read_bytes()).reshape(T, W)
    got_n = m.bf16_to_f32((out / "H_normed.bin").read_bytes()).reshape(T, W)
    got_o = m.bf16_to_f32((out / "H_out.bin").read_bytes()).reshape(T, W)

    def chain(kv_src):
        k, v = kv_src[:, :W], kv_src[:, W:]
        gr = np.zeros((T, W), dtype=np.float32)
        nr = np.zeros((T, W), dtype=np.float32)
        for t in range(T):
            for s in range(HC):
                gg, nn = m.ref_gate(k[t, s * HID:(s + 1) * HID], v[t], hidden[t, s * HID:(s + 1) * HID],
                                    nk[s * HID:(s + 1) * HID], nq[s * HID:(s + 1) * HID],
                                    ncw[s * HID:(s + 1) * HID])
                gr[t, s * HID:(s + 1) * HID] = gg
                nr[t, s * HID:(s + 1) * HID] = nn
        orf = np.zeros((T, W), dtype=np.float32)
        for t in range(T):
            if sidx[t] < 0:
                po = m.bf16_rne(gr[t].astype(np.float32) + np.float32(0.0))
                orf[t] = m.bf16_rne(hidden[t].astype(np.float32) + po)
            else:
                po, _ = m.ref_conv_out(gr[t], nr[t], state[sidx[t]], wconv)
                orf[t] = m.bf16_rne(hidden[t].astype(np.float32) + po)
        return gr, nr, orf

    gr_ref, nr_ref, or_ref = chain(kv_ref)
    print("\n== §6.2 ②  ④⑤ 的格点项（参考链吃 kv_ref = 现行判据口径）==")
    diag(m, "H4.gated", got_g, gr_ref, eps4, tok_in, None)
    diag(m, "H4.normed", got_n, nr_ref, eps4, tok_in, None)
    diag(m, "H5.out", got_o, or_ref, eps4, tok_in, None)

    print("\n== §6.3  参考链吃 kv_ref（现行）vs 吃设备 H_kv.bin（**诊断**，不参与 PASS/FAIL）==")
    gr_dev, nr_dev, or_dev = chain(kv_got)
    for tag, refs in (("吃 kv_ref(现行判据)", (gr_ref, nr_ref, or_ref)),
                      ("吃 kv_dev(诊断)", (gr_dev, nr_dev, or_dev))):
        for nm, got, ref in (("H4.gated", got_g, refs[0]), ("H4.normed", got_n, refs[1]),
                             ("H5.out", got_o, refs[2])):
            s2 = got[tok_in].astype(np.float64)
            r2 = ref[tok_in].astype(np.float64)
            d2 = np.abs(s2 - r2)
            b2 = eps4 * np.maximum(np.abs(s2), np.abs(r2)) + 1.0 * m.bf_ulp(s2)
            print("  %-16s %-9s 逐位不同=%d  1.0ulp 界 bad=%d  max(d/bound)=%.4f"
                  % (tag, nm, int((d2 > 0).sum()), int((d2 > b2).sum()),
                     float(np.max(d2 / np.maximum(b2, 1e-300)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
