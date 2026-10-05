#!/usr/bin/env python3.12
"""hm_multislot_check.py —— M171：真实表**多槽窗口池 + 跨分片**的判据（slot-aware）

M92 的 `m15_ple_check.py::check_hm` 把窗口模型钉成「一个连续区间 [r0, r0+win_rows)」：
  · `inw = (ids >= r0) & (ids < r0+win_rows)`（单个区间掩码）
  · 参考行 `real_row(meta, gid)` 只用 `meta["file"]/data_off`（**单个分片**的文件）
多槽池**两个假设都不成立**：ids 可落在池里几个**互不相邻**的槽（区间掩码会误判），
且一个槽可以横跨分片边界（单个 `file` 不够）。本脚本把这些口径换成槽-aware 的版本，
**其余判据（T1/T3 的分档、Σ|terms| 界、格点系数、设备侧计数器）逐字复用 `m15_ple_check`**。

判据名与 M92 的 10 条一致（`H2.emb.nonvac/H2.emb/H3.kv/H4.gated/H4.normed/H5.out/Hd.cores/
Hd.row_fail/Hd.miss/Hd.nonvac`），另加一条 `Hd.served`（**所有 id 都必须被池服务**）——
它正是 `ids_hm_miss.bin` 越窗对照要打红的项。

用法：
  python3.12 m15_layer_loop/ple/hm_multislot_check.py m15_layer_loop/ple/out_ms_base
  ... <outdir> --brief        # 只打 RESULT 行（供负向对照驱动判读）
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent          # m15_layer_loop/ple/
M15 = HERE.parent                                        # m15_layer_loop/
sys.path.insert(0, str(M15))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load(M15 / "m15_ple_check.py", "m15_ple_check")        # 复用 Report / T3 / 参考链
rtp = _load(HERE / "real_table_probe.py", "real_table_probe")  # 复用 shard_map

HDIM = mp.HDIM
HE = mp.HE
W = mp.W
KVW = mp.KVW
STLEN = mp.STLEN
HC = mp.HC


def hm_meta():
    return mp.json.loads((mp.HM / "hm_meta.json").read_text())


def slot_of(gid, slots, rps):
    """与设备端 `PleGather` 的 `nSlots != 0` 分支同一套选择：命中返回 `row`，否则 None。"""
    if gid < 0:
        return None
    shard, local = divmod(int(gid), rps)
    for s in slots:
        if shard < s["shard"]:
            continue
        base = (shard - s["shard"]) * rps + local
        if base < s["local_base"]:
            continue
        delta = base - s["local_base"]
        if delta < s["rows"]:
            return s["row_off"] + delta
    return None


def real_row_shard(smap, rps, gid):
    """按 `shard = id // rows_per_shard` 取**该分片**的文件偏移直读 320 B（跨分片也成立）。"""
    shard, local = divmod(int(gid), rps)
    info = smap[shard]
    with open(info["file"], "rb") as fh:
        fh.seek(info["data_off"] + local * 320)
        return fh.read(320)


def check(outdir, brief=False):
    rep = mp.Report()
    if brief:
        mp.BRIEF = True
    meta = hm_meta()
    d = mp.HM
    rps = int(meta["rows_per_shard"])
    slots = meta["slots"]
    T = int(meta["tokens"])
    ng = int(meta["ng"])
    items = T * ng
    pool_rows = int(meta["pool_rows"])
    ids = np.fromfile(outdir / "H_ids.bin", dtype=np.int64).reshape(T, ng)

    served = np.zeros((T, ng), dtype=bool)
    rows_ref = np.zeros((items, HDIM), dtype=np.float32)
    smap = rtp.shard_map()
    nz = 0
    for t in range(T):
        for g in range(ng):
            i = t * ng + g
            if slot_of(int(ids[t, g]), slots, rps) is None:
                continue
            served[t, g] = True
            raw = real_row_shard(smap, rps, int(ids[t, g]))
            v = mp.bf16_to_f32(raw)
            rows_ref[i] = v
            nz += int((v != 0).sum())
    n_in = int(served.sum())
    n_out = items - n_in

    rep.add("H2.emb.nonvac", "T1-struct", n_in * HDIM, 0 if nz > 0 else 1,
            "对照行（直读分片文件）的非零元素 = %d / %d；槽内 item = %d" % (nz, n_in * HDIM, n_in),
            "整片为 0 ⇒ 逐字节判据是空过（比较双方同为常量）")

    emb_got = mp.bf16_to_f32((outdir / "H_emb.bin").read_bytes()).reshape(T, HE)
    emb_ref = np.zeros((T, HE), dtype=np.float32)
    for t in range(T):
        for g in range(ng):
            emb_ref[t, g * HDIM:(g + 1) * HDIM] = rows_ref[t * ng + g]
    m_emb = np.zeros((T, HE), dtype=bool)
    for g in range(ng):
        m_emb[:, g * HDIM:(g + 1) * HDIM] = served[:, g][:, None]
    mp.cmp_bits_masked("H2.emb", rep, emb_got, emb_ref, m_emb, "T1",
                       "槽选择/(shard,local) 分解错、跨分片行偏移错、head 落点错、行宽错")

    tok_in = served.all(axis=1)
    wd = mp.PLE / "data"
    nk = mp.bf16_to_f32((wd / "w_norm_key.bin").read_bytes())
    nq = mp.bf16_to_f32((wd / "w_norm_query.bin").read_bytes())
    ncw = mp.bf16_to_f32((wd / "w_norm_conv.bin").read_bytes())
    wconv = mp.bf16_to_f32((wd / "w_conv1d_sq.bin").read_bytes()).reshape(W, mp.KCONV)
    wkey = mp.bf16_to_f32((wd / "w_key_proj.bin").read_bytes()).reshape(W, HE)
    wval = mp.bf16_to_f32((wd / "w_value_proj.bin").read_bytes()).reshape(HE, HE)
    hidden = mp.bf16_to_f32((d / "hidden_hm.bin").read_bytes()).reshape(T, W)
    state = mp.bf16_to_f32((d / "conv_state_hm.bin").read_bytes()).reshape(-1, STLEN, W)
    sidx = np.fromfile(d / "state_idx_hm.bin", dtype=np.int32)

    kv_got = mp.bf16_to_f32((outdir / "H_kv.bin").read_bytes()).reshape(T, KVW)
    kv_ref = np.zeros((T, KVW), dtype=np.float32)
    kv_sumabs = np.zeros((T, KVW), dtype=np.float64)
    wcat = np.concatenate([wkey, wval], axis=0)
    awcat = np.abs(wcat.astype(np.float64))
    for t in range(T):
        e = emb_ref[t].astype(np.float64)
        kv_ref[t] = mp.bf16_rne(wcat.astype(np.float64) @ e)
        kv_sumabs[t] = awcat @ np.abs(e)
    mp.cmp_t3_masked("H3.kv", rep, kv_got, kv_ref, tok_in, mp.EPS_MMAD,
                     "key/value 顺序反、GEMM 转置错；本条同时是②的正确性下游证据",
                     extra_terms=kv_sumabs)

    gated_got = mp.bf16_to_f32((outdir / "H_gated.bin").read_bytes()).reshape(T, W)
    nor_got = mp.bf16_to_f32((outdir / "H_normed.bin").read_bytes()).reshape(T, W)
    gated_ref = np.zeros((T, W), dtype=np.float32)
    normed_ref = np.zeros((T, W), dtype=np.float32)
    HID = W // HC
    key = kv_ref[:, :W]
    value = kv_ref[:, W:]
    for t in range(T):
        for s in range(HC):
            gg, nn = mp.ref_gate(key[t, s * HID:(s + 1) * HID], value[t],
                                 hidden[t, s * HID:(s + 1) * HID],
                                 nk[s * HID:(s + 1) * HID],
                                 nq[s * HID:(s + 1) * HID],
                                 ncw[s * HID:(s + 1) * HID])
            gated_ref[t, s * HID:(s + 1) * HID] = gg
            normed_ref[t, s * HID:(s + 1) * HID] = nn

    # ---- 上游噪声的**传播界**（M171：④⑤ 的 2.0·ulp 之外再加一项）----
    # 为什么需要：④⑤ 的参考吃 `kv_ref`（M92 的端到端口径），而设备吃 `kv_dev`；当 value 投影
    #   在某个通道**深度相消**（|value| ≪ Σ|terms|）时，`|Δvalue|` 虽被 H3 的 Σ|terms| 界罩住，
    #   但 `Δgated ≈ g·Δvalue` 的绝对量会**大于 gated 自己的 ulp**（gated 也≈0）⇒ 只用 2·ulp 会误红。
    #   正确做法是给 H4/H5 也补上「上游项的尺度」：`Δv ≤ EPS_MMAD·Σ|terms_v|`（H3 已证 max(d/bound)
    #   < 1），再沿 ④⑤ 的算式向下传播：
    #     gated  : Δg ≤ |g'|·Δv        |g'|≤1  ⇒ extra_g   = dval
    #     normed : Δn ≤ rstd·(1+ncw)·Δg          ⇒ extra_n = dval·rstd·(1+ncw)
    #     out    : Δo ≤ Δg + |wconv[:,3]|·Δn      ⇒ extra_o = extra_g + |wconv[:,3]|·extra_n
    #                       （只有 k=3 的 tap 是当前 conv_in=normed，其余 tap 是 host 给的状态）
    #   这些项除以 e4 后作为 `extra_terms` 传入（`cmp_t3_masked` 会乘回 e4）。
    e4 = mp.EPS_REDUCE + mp.EPS_RSQRT + mp.EPS_SIGMOID
    dval = kv_sumabs[:, W:] * mp.EPS_MMAD                    # [T,HID]：Δv 的上界（Σ|terms_v|·ε_mmad）
    xg = np.zeros((T, W))
    xn = np.zeros((T, W))
    xo = np.zeros((T, W))
    for t in range(T):
        for s in range(HC):
            seg = slice(s * HID, (s + 1) * HID)
            gf = gated_ref[t, seg].astype(np.float64)
            rstd = 1.0 / np.sqrt((gf * gf).mean() + mp.EPS)
            xg[t, seg] = dval[t]
            xn[t, seg] = dval[t] * rstd * (1.0 + ncw[seg].astype(np.float64))
            xo[t, seg] = xg[t, seg] + np.abs(wconv[seg, 3].astype(np.float64)) * xn[t, seg]
    mp.cmp_t3_masked("H4.gated", rep, gated_got, gated_ref, tok_in, e4,
                     "query 用 pending BO、4 流共享 value 错、`(1+w)` 漏 1", extra_terms=xg / e4,
                     ulp_mult=2.0)
    mp.cmp_t3_masked("H4.normed", rep, nor_got, normed_ref, tok_in, e4,
                     "同上 + normed 的分组归约", extra_terms=xn / e4, ulp_mult=2.0)

    out_got = mp.bf16_to_f32((outdir / "H_out.bin").read_bytes()).reshape(T, W)
    out_ref = np.zeros((T, W), dtype=np.float32)
    for t in range(T):
        sl = int(sidx[t])
        if sl < 0:
            po = mp.bf16_rne(gated_ref[t].astype(np.float32) + np.float32(0.0))
        else:
            po, _ = mp.ref_conv_out(gated_ref[t], normed_ref[t], state[sl], wconv)
        out_ref[t] = mp.bf16_rne(hidden[t].astype(np.float32) + po)
    mp.cmp_t3_masked("H5.out", rep, out_got, out_ref, tok_in, e4,
                     "taps lag 9/6/3/0 错、SiLU 漏、残差顺序反（②的错会一路传到这里）",
                     extra_terms=xo / e4, ulp_mult=2.0)

    devf = np.frombuffer((outdir / "H_hm_dev.bin").read_bytes(), dtype=np.float32).reshape(-1, 8)
    n_cores = int((devf[:, 2] != 0.0).sum())
    n_core_exp = 2 * 28
    dev_row_fail = float(devf[:, 0].sum())
    dev_miss = int(devf[:, 1].sum())
    win_rows_dev = sorted(set(devf[:, 3][devf[:, 2] != 0.0].tolist()))
    rep.add("Hd.cores", "T1-struct", n_core_exp, 0 if n_cores == n_core_exp else 1,
            "设备侧自报参与核 = %d / %d；设备侧读到的 pool_rows = %s（host 期望 %d）"
            % (n_cores, n_core_exp, win_rows_dev, pool_rows),
            "核没跑②或 pool_rows 参数没传到核里")
    rep.add("Hd.row_fail", "T1-struct", n_in, 0 if dev_row_fail == 0.0 else 1,
            "设备侧「行首 64 元素 vs host 直读期望」错行计数 = %.0f（槽内 item %d）"
            % (dev_row_fail, n_in),
            "该计数 > 0 ⇔ 设备取到的行 ≠ 该 id 应有的行（槽选择/(shard,local) 分解错、窗口内容错）")
    rep.add("Hd.miss", "T1-struct", items, 0 if dev_miss == n_out else 1,
            "设备侧越窗行计数 = %d；host 独立算出的池外 item = %d" % (dev_miss, n_out),
            "越窗判定的槽范围参数错（shard/local_base/rows 与注册长度不符）")
    rep.add("Hd.nonvac", "T1-struct", items, 0 if (n_in > 0 and n_cores == n_core_exp) else 1,
            "槽内 item = %d（>0 才算比过）；设备侧参与核 = %d" % (n_in, n_cores),
            "槽内没有 item 或没有核跑② ⇒ Hd.row_fail 是空过")
    # ★ M171 新增：多槽池必须**真的把每个 id 都服务到**（越窗对照 ids_hm_miss.bin 打红的就是它）
    rep.add("Hd.served", "T1-struct", items, 0 if dev_miss == 0 else 1,
            "设备侧越窗计数 = %d（%d 个 id 未被池服务）" % (dev_miss, dev_miss),
            "有 id 落在所有槽之外（池没覆盖到）⇒ 该 id 的 emb 不会被写、下游全错")

    n_judge = len(rep.rows)
    if not brief:
        print("\n===== 多槽判据合计 %d 条；FAIL %d 条 =====" % (n_judge, rep.fails))
    print("===== %s =====" % ("ALL PASS" if rep.fails == 0 else "FAILURES PRESENT"))
    return 0 if rep.fails == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()
    return check(pathlib.Path(args.outdir), args.brief)


if __name__ == "__main__":
    sys.exit(main())
