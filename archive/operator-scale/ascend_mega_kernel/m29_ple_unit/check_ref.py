#!/usr/bin/env python3.12
"""m29_ple_unit/check_ref.py —— M161 PLE 单元的 host float64 参考对拍（docs/17 §1.1 分档）

用法：
  python3 m29_ple_unit/check_ref.py <outdir> <datadir>

退出码（三态，tower 规则）：
  0 = 比过且通过；1 = 比过有差异；2 = 没得比 / 输入缺失

分档：
  · ids[T,16]                          → T1 逐位
  · emb[T,2560]（表行 gather）          → T1 逐字节（真实档：与**直读分片文件**比对，独立于窗口）
  · kv / gated / normed / out / state  → T3（ε 来源见 m29_common.py；含 mmad 累加 + Rsqrt/Sigmoid）

负向对照（分片跑）：把设备档用 M29_NEG_TABLE / M29_NEG_STATE 弄坏后，本判据**必须** rc=1。
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m29_common as C  # noqa: E402

BRIEF = False
FAILS = []


def add(name, tier, n, nbad, detail):
    st = "PASS" if nbad == 0 else "FAIL"
    if nbad:
        FAILS.append(name)
    print("RESULT|%s|%s|%d|%d|%s" % (name, tier, n, nbad, st))
    if not BRIEF:
        print("[%s] %-26s tier=%-3s n=%-8d bad=%-6d %s" % (st, name, tier, n, nbad, detail))


def cmp_bits(name, got, ref, tier, detail):
    refc = np.asarray(ref).astype(got.dtype)
    nbad = int((got != refc).sum())
    add(name, tier, int(got.size), nbad, detail)


def cmp_t3(name, got_bf16, ref_f32, eps, detail, extra_terms=None):
    out = got_bf16.astype(np.float64)
    ref = ref_f32.astype(np.float64)
    # docs/17 §1.1 的 T3 原式用 Σ|terms|；不给 extra_terms 时退回 max(|out|,|ref|) 代理
    # （m15_ple_check.py:500-502 警告过：真实表行 + 真实权重 GEMV 会相消 ⇒ 代理偏紧）。
    grid = np.maximum(np.abs(out), np.abs(ref))
    kind = "proxy=max(|out|,|ref|)"
    if extra_terms is not None:
        et = np.asarray(extra_terms, dtype=np.float64)
        ratio = et / np.maximum(grid, 1e-300)
        kind = "Sum|terms|(median/max ratio=%.3g/%.3g)" % (float(np.median(ratio)), float(np.max(ratio)))
        grid = np.maximum(grid, et)
    bound = eps * grid + 1.0 * C.bf_ulp(out)
    d = np.abs(out - ref)
    nbad = int((d > bound).sum())
    marg = float(np.max(d / np.maximum(bound, 1e-300))) if d.size else 0.0
    add(name, "T3", int(out.size), nbad, "%s eps=%.3g max(d/bound)=%.3f grid=%s" % (detail, eps, marg, kind))



def ld(p):
    return np.fromfile(p, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("datadir")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    global BRIEF
    BRIEF = args.quiet
    out = pathlib.Path(args.outdir)
    d = pathlib.Path(args.datadir)

    if not (d / "unit_meta.txt").exists() or not (out / "s0_meta.txt").exists():
        print("RESULT|SKIPPED|na|0|0|SKIPPED")
        print("[SKIP] 缺 unit_meta.txt 或 s0_meta.txt")
        return 2
    meta = {}
    for l in (d / "unit_meta.txt").read_text().strip().split("\n"):
        p = l.split()
        if p:
            meta[p[0]] = p[1:]
    mode = meta["mode"][0]
    n_tok = int(meta["n_tok"][0])
    n_req = int(meta["n_req"][0])
    steps = int(meta["n_steps"][0])

    m = np.fromfile(d / "m.bin", dtype=np.int64)
    sizes = np.fromfile(d / "sizes.bin", dtype=np.int64)
    offs = np.fromfile(d / "offsets.bin", dtype=np.int64)
    nk = C.bf16_to_f32((d / "w_norm_key.bin").read_bytes())
    nq = C.bf16_to_f32((d / "w_norm_query.bin").read_bytes())
    ncw = C.bf16_to_f32((d / "w_norm_conv.bin").read_bytes())
    wkey = C.bf16_to_f32((d / "w_key_proj.bin").read_bytes()).reshape(C.W, C.HE)
    wval = C.bf16_to_f32((d / "w_value_proj.bin").read_bytes()).reshape(C.HE, C.HE)
    wcat = np.concatenate([wkey, wval], axis=0)              # [12800,2560]
    raw_conv = (d / "w_conv1d.bin").read_bytes()
    sh = (C.W, 1, C.KCONV)
    wconv = C.bf16_to_f32(raw_conv).reshape(sh[0], sh[2])     # [10240,4]
    table_flat = None
    smap = None
    slots = []
    if mode == "flat":
        table_flat = C.bf16_to_f32((d / meta["table_file"][0]).read_bytes()).reshape(-1, C.HDIM)
    else:
        smap = C.shard_map()
        for l in (d / "unit_meta.txt").read_text().strip().split("\n"):
            p = l.split()
            if p and p[0] == "slot":
                # slot <i> shard <sh> local0 <l0> rows <r> win <w> file <f>
                slots.append((int(p[3]), int(p[5]), int(p[7])))

    def table_row(gid):
        gid = int(gid)
        if mode == "flat":
            return table_flat[gid] if 0 <= gid < table_flat.shape[0] else None
        s, loc = divmod(gid, C.ROWS_PER_SHARD)
        for (ss, l0, r) in slots:
            if ss == s and l0 <= loc < l0 + r:
                return C.bf16_to_f32(C.direct_row(smap, gid))
        return None

    # ---- 逐 step ----
    state = np.zeros((64, C.STLEN, C.W), dtype=np.float32)     # state_0（slot 0..1 用 0）
    for k in range(steps):
        tag = "s%d" % k
        dev_meta = {}
        for l in (out / (tag + "_meta.txt")).read_text().strip().split("\n"):
            p = l.split()
            if p:
                dev_meta[p[0]] = p[1]
        dev_oob = int(dev_meta["dev_oob"])
        dev_miss = int(dev_meta["dev_miss"])
        in_ids = np.fromfile(d / (tag + "_ids.bin"), dtype=np.int32)
        qsl = np.fromfile(d / (tag + "_qsl.bin"), dtype=np.int32)
        ctx = np.fromfile(d / (tag + "_ctx.bin"), dtype=np.int32).reshape(-1, C.NC)
        hidden = C.bf16_to_f32((d / (tag + "_hidden.bin")).read_bytes()).reshape(n_tok, C.W)
        sidx = np.fromfile(d / (tag + "_sidx.bin"), dtype=np.int32)

        # ① ids（T1）
        ids_ref = C.ref_ids(in_ids, qsl, ctx, m, sizes, offs)
        ids_dev = np.fromfile(out / (tag + "_ids.bin"), dtype=np.int64).reshape(n_tok, C.NG)
        cmp_bits("B1.ids.%s" % tag, ids_dev, ids_ref, "T1", "n-gram id 逐位")

        # ② emb（T1，covered 部分）
        emb_dev = C.bf16_to_f32((out / (tag + "_emb.bin")).read_bytes()).reshape(n_tok, C.HE)
        covered = np.zeros((n_tok, C.NG), dtype=bool)
        emb_ref = np.zeros((n_tok, C.HE), dtype=np.float32)
        miss_exp = 0
        for t in range(n_tok):
            for g in range(C.NG):
                r = table_row(ids_ref[t, g])
                if r is None:
                    miss_exp += 1
                else:
                    covered[t, g] = True
                    emb_ref[t, g * C.HDIM:(g + 1) * C.HDIM] = r
        cov_el = np.repeat(covered, C.HDIM, axis=1)
        cmp_bits("B2.emb.%s" % tag, emb_dev[cov_el], emb_ref[cov_el], "T1",
                 "表行逐字节（covered %d/%d；真实档与直读分片比）" % (int(covered.sum()), covered.size))
        add("B2.miss.%s" % tag, "T1-struct", n_tok * C.NG, 0 if dev_miss == miss_exp else 1,
            "设备越窗计数 %d vs 参考未覆盖 %d" % (dev_miss, miss_exp))
        if dev_oob != 0:
            add("B2.oob.%s" % tag, "T1-struct", n_tok * C.NG, 1, "设备词表外 id 计数=%d" % dev_oob)
        if int(covered.sum()) < covered.size:
            # 越窗档：只判 covered 行 + miss 计数；跳过后级 T3（状态参考不推进本档）
            print("[m29-ref] step %d covered %d/%d ⇒ 跳过后级 T3（越窗档）"
                  % (k, int(covered.sum()), covered.size))
            continue

        # ③ kv（T3）：界用 docs/17 §1.1 的 Σ|terms|（真实表行 + 真实权重会相消，代理偏紧；
        #   与 m15_ple_check.py:492-503 的 cmp_t3_masked(extra_terms=Σ|terms|) 同口径）
        awcat = np.abs(wcat.astype(np.float64))
        kv_ref = np.zeros((n_tok, C.KVW), dtype=np.float32)
        kv_sumabs = np.zeros((n_tok, C.KVW), dtype=np.float64)
        for t in range(n_tok):
            e = emb_ref[t].astype(np.float64)
            kv_ref[t] = C.bf16_rne(np.float64(wcat) @ e)
            kv_sumabs[t] = awcat @ np.abs(e)
        kv_dev = C.bf16_to_f32((out / (tag + "_kv.bin")).read_bytes()).reshape(n_tok, C.KVW)
        cmp_t3("B3.kv.%s" % tag, kv_dev, kv_ref, C.EPS_MMAD, "cube mmad K=2560", extra_terms=kv_sumabs)
        key = kv_dev[:, :C.W]
        value = kv_dev[:, C.W:]

        # ④ gate（T3）
        gat_ref = np.zeros((n_tok, C.W), dtype=np.float32)
        nor_ref = np.zeros((n_tok, C.W), dtype=np.float32)
        for t in range(n_tok):
            for s in range(C.HC):
                gg, nn = C.ref_gate(key[t, s * C.HID:(s + 1) * C.HID], value[t],
                                    hidden[t, s * C.HID:(s + 1) * C.HID], nk[s * C.HID:(s + 1) * C.HID],
                                    nq[s * C.HID:(s + 1) * C.HID], ncw[s * C.HID:(s + 1) * C.HID])
                gat_ref[t, s * C.HID:(s + 1) * C.HID] = gg
                nor_ref[t, s * C.HID:(s + 1) * C.HID] = nn
        gat_dev = C.bf16_to_f32((out / (tag + "_gated.bin")).read_bytes()).reshape(n_tok, C.W)
        nor_dev = C.bf16_to_f32((out / (tag + "_normed.bin")).read_bytes()).reshape(n_tok, C.W)
        cmp_t3("B4.gated.%s" % tag, gat_dev, gat_ref, C.EPS_RSQRT + C.EPS_SIGMOID, "gate g·v")
        cmp_t3("B4.normed.%s" % tag, nor_dev, nor_ref, C.EPS_RSQRT + C.EPS_SIGMOID, "normed")

        # ⑤ conv + 残差 + 状态（T3）；参考逐 token 演化状态
        out_ref = np.zeros((n_tok, C.W), dtype=np.float32)
        nst = np.zeros((64, C.STLEN, C.W), dtype=np.float32)
        nst[:, :, :] = state
        for t in range(n_tok):
            sl = int(sidx[t])
            po, _ = C.ref_conv_out(gat_ref[t], nor_ref[t], state[sl], wconv)
            out_ref[t] = C.bf16_rne(hidden[t].astype(np.float32) + po)
            nst[sl] = np.vstack([state[sl][1:], nor_ref[t][None, :]])
        outp = out / (tag + "_out.bin")
        if outp.exists():
            out_dev = C.bf16_to_f32(outp.read_bytes()).reshape(n_tok, C.W)
            cmp_t3("B5.out.%s" % tag, out_dev, out_ref, C.EPS_RSQRT + C.EPS_SIGMOID,
                   "conv+SiLU+残差")
        stp = out / (tag + "_state_out.bin")
        if stp.exists():
            sto_dev = C.bf16_to_f32(stp.read_bytes()).reshape(64, C.STLEN, C.W)
            act = sorted(set(int(i) for i in sidx))
            cmp_t3("B5.state.%s" % tag, sto_dev[act].ravel(), nst[act].ravel(), 0.0,
                   "活跃 slot [%d] 状态演化" % len(act))
        state = nst

    verdict = "FAILED" if FAILS else "OK"
    print("VERDICT=%s steps=%d n_fail=%d" % (verdict, steps, len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
