#!/usr/bin/env python3
"""生成 check_ref.py 的**合成**判据夹具（只涉判据脚本，不涉设备）。

用途：把 (b) 逐块/首块对拍、(c) 输入面消费见证、以及 (a) 的 `--negctl 8` **尾块（MT > m）**
形态在没有设备的情况下演示到「能报出该报的话 / 缺 dump 时如实报缺 / 不崩」这一层。
合成数据的数学与 `m26_moe_prefill.asc` 的 S2 同源（x·Wᵀ → softmax → top-k → renorm；
sgate 为裸点积），但**与任何设备档无关**，不得当成设备读数引用。

用法：
    python3.12 m26_moe_prefill/tools/gen_judge_fixtures.py <outdir>

产出：
    <outdir>/blocks/         多块 dump，只落 b0 与 b2（n_tiles=3）⇒ 覆盖报告应报「缺 b1」且首块已覆盖
    <outdir>/consumed_read/  单块 dump + 消费快照：logits 由**消费快照**算出、与回读面不同 ⇒ 判「读错输入」
    <outdir>/consumed_calc/  单块 dump + 消费快照：logits 与快照、回读面**都不**一致 ⇒ 判「算错」
    <outdir>/no_witness/     单块 dump（无快照、无 checksum）⇒ 判「未提供、不可区分」
    <outdir>/tail_tie/       **尾块 MT=8 > m=4**，被判行是 tie-risk（J1 跳过）⇒ 正常 PASS；
                             `--negctl 8` 置换成 **只有 J1c 能抓**（且 pad 行不被置换、不崩）
    <outdir>/tail_plain/     **尾块 MT=8 > m=4**，被判行都被 J1 判（safe）⇒ `--negctl 8` **不适用**，
                             判据报"负向对照未生效"而不是空转或崩
"""
import hashlib
import os
import sys

import numpy as np

H = 2560
E = 512
E_PAD = 640
TOPK = 10
MT = 4


def f32_to_bf16_bits(f):
    u = np.asarray(f, dtype=np.float32).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return (r & 0xFFFF).astype(np.uint16)


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def make_weight(rng, e_pad=E_PAD):
    w = (rng.standard_normal((e_pad, H)) * 0.05).astype(np.float32)
    return w, f32_to_bf16_bits(w)


def s2_from_bits(x_u, w_u, rows, e=E, e_pad=E_PAD):
    """与 check_ref.py 同源的参考：解码 bf16 → fp64 matmul → softmax/top-k/renorm。"""
    x = bf16_bits_to_f32(x_u).astype(np.float64).reshape(rows, H)
    w = bf16_bits_to_f32(w_u).astype(np.float64).reshape(e_pad, H)
    logits = x @ w[:e].T
    order = np.argsort(-logits, axis=1, kind="stable")
    ids = order[:, :TOPK]
    l_top = np.take_along_axis(logits, ids, axis=1)
    v = np.exp(l_top - l_top[:, :1])
    wr = v / v.sum(axis=1, keepdims=True)
    sg = x @ w[e]
    return logits, ids.astype(np.int32), wr.astype(np.float32), sg.astype(np.float32)


def write_block(d, suffix, meta_lines, x_u, w_u, logits, ids, w, sg, e_pad=E_PAD):
    os.makedirs(d, exist_ok=True)
    x_u.tofile(os.path.join(d, "m26_x%s.bin" % suffix))
    w_u.tofile(os.path.join(d, "m26_wpad.bin"))
    # dump 的 logits 平面是 MT×E_PAD（前 E 列有效，尾列只占位）——判据按 E_PAD 读
    lf = np.zeros((logits.shape[0], e_pad), dtype=np.float64)
    lf[:, :logits.shape[1]] = logits
    lf.astype("<f4").tofile(os.path.join(d, "m26_logits%s.bin" % suffix))
    ids.astype("<i4").tofile(os.path.join(d, "m26_ids%s.bin" % suffix))
    w.astype("<f4").tofile(os.path.join(d, "m26_w%s.bin" % suffix))
    sg.astype("<f4").tofile(os.path.join(d, "m26_sgate%s.bin" % suffix))
    with open(os.path.join(d, "m26_meta%s.txt" % suffix), "w") as fp:
        fp.write("".join("%s\n" % ln for ln in meta_lines))


def base_meta(rows, mt, extra, e=E, e_pad=E_PAD):
    lines = ["m=%d" % rows, "stage=8", "topk=%d" % TOPK, "E=%d" % e, "E_PAD=%d" % e_pad,
             "MT=%d" % mt, "TP_MAX=%d" % (mt * TOPK), "hidden=%d" % H, "nblk=28"]
    lines += extra
    return lines


def gen_blocks(out):
    d = os.path.join(out, "blocks")
    rng = np.random.default_rng(20261004)
    _, w_u = make_weight(rng)
    w = bf16_bits_to_f32(w_u).astype(np.float64).reshape(E_PAD, H)
    # 三块一致的权重，各自不同的 x；只落 b0、b2（b1 缺失）
    for i in [0, 2]:
        x = (rng.standard_normal((MT, H)) * 0.5).astype(np.float32)
        x_u = f32_to_bf16_bits(x)
        logits, ids, wr, sg = s2_from_bits(x_u, w_u, MT)
        meta = base_meta(MT, MT, ["blk=%d" % i, "n_tiles=3", "row0=%d" % (i * MT), "m_total=%d" % (3 * MT)])
        write_block(d, ".b%d" % i, meta, x_u, w_u, logits, ids, wr, sg)


def gen_consumed(out, wrong):
    """wrong=True：logits 与快照/回读面都不一致（算错）；False：logits 由快照算出（读错输入）。"""
    d = os.path.join(out, "consumed_read" if not wrong else "consumed_calc")
    rng = np.random.default_rng(7 if not wrong else 8)
    _, w_u = make_weight(rng)
    rows = MT
    x_read = (rng.standard_normal((rows, H)) * 0.5).astype(np.float32)
    x_snap = (rng.standard_normal((rows, H)) * 0.5).astype(np.float32)   # 消费面 != 回读面
    xr_u, xs_u = f32_to_bf16_bits(x_read), f32_to_bf16_bits(x_snap)
    if wrong:
        logits = (rng.standard_normal((rows, E)) * 5.0).astype(np.float64)   # 与两者都无关
    else:
        logits, _, _, _ = s2_from_bits(xs_u, w_u, rows)                      # 由**消费快照**算出
    _, ids, wr, sg = s2_from_bits(xr_u, w_u, rows)                           # ids/weights 取自回读面
    meta = base_meta(rows, rows, ["blk=0", "n_tiles=1", "row0=0", "m_total=%d" % rows])
    write_block(d, "", meta, xr_u, w_u, logits, ids, wr, sg)
    xs_u.tofile(os.path.join(d, "m26_x_consumed.bin"))


def gen_no_witness(out):
    src = os.path.join(out, "consumed_read")
    dst = os.path.join(out, "no_witness")
    os.makedirs(dst, exist_ok=True)
    for nm in os.listdir(src):
        if nm == "m26_x_consumed.bin":
            continue
        with open(os.path.join(src, nm), "rb") as a, open(os.path.join(dst, nm), "wb") as b:
            b.write(a.read())


def gen_tail(out, name, desired, seed, mt=8, m=4):
    """尾块夹具：缓冲 MT=mt 行、被判 m 行。

    x 按「每个专家占一段互不重叠的列块」构造 ⇒ 可把 12 个专家的 logits 设成**指定值**
    （x_块j · W[j]_块j = desired[j]），于是能确定性地造出：
      · desired 里第 10/11 名间隔很小 ⇒ 该行 tie-risk（J1 跳过）；
      · top-k 内部的相邻间隔很大 ⇒ J1c 有显著相邻对可判。
    """
    d = os.path.join(out, name)
    rng = np.random.default_rng(seed)
    e2, ep2 = 12, 16
    nb = e2 + 1                      # 专家 0..e2-1 + 共享门行（w[e2]）
    bs = H // nb                     # 2560 // 13 = 196；13*196 = 2548 ≤ 2560
    W = np.zeros((ep2, H), dtype=np.float32)
    for j in range(nb):
        W[j, j * bs:(j + 1) * bs] = (rng.standard_normal(bs) * 0.05).astype(np.float32)
    x = np.zeros((mt, H), dtype=np.float32)
    for r in range(m):
        # 每行整体缩放一个因子 ⇒ 跨行 logits 不同（G2），而 gap/row_tol 之比不变（tie 判定不变）
        scale = 1.0 + 0.3 * r
        for j in range(e2):
            blk = W[j, j * bs:(j + 1) * bs]
            nrm = float(np.dot(blk, blk))
            if nrm > 0.0:
                x[r, j * bs:(j + 1) * bs] = (desired[j] * scale / nrm) * blk
        # 共享门块：随机值让 sgate 跨行变化（G6）
        x[r, e2 * bs:(e2 + 1) * bs] = (rng.standard_normal(bs) * 0.5).astype(np.float32)
    x_u = f32_to_bf16_bits(x)
    w_u = f32_to_bf16_bits(W)
    logits, ids, wr, sg = s2_from_bits(x_u, w_u, mt, e2, ep2)
    meta = base_meta(m, mt, [], e=e2, e_pad=ep2)   # m=被判行数、MT=缓冲行数 ⇒ MT > m
    write_block(d, "", meta, x_u, w_u, logits, ids, wr, sg, e_pad=ep2)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build/judge_fixtures"
    os.makedirs(out, exist_ok=True)
    gen_blocks(out)
    gen_consumed(out, wrong=False)
    gen_consumed(out, wrong=True)
    gen_no_witness(out)
    # 9 个大值 + 第 10/11 名并列（间隔很小 ⇒ tie-risk）+ 1 个小值
    #   row_tol ≈ EPS_ACC·terms ≈ 1.5e-4·10 = 1.5e-3，4×row_tol ≈ 6e-3（见 check_ref.py 的 J1 门）
    #   ⇒ 10/11 名间隔取 2e-3，留 3× 余量。
    gen_tail(out, "tail_tie", [10, 9, 8, 7, 6, 5, 4, 3, 2, 0.5, 0.502, -1.0], seed=101)
    # 10 个大值（第 10/11 名间隔很大 ⇒ safe）⇒ V8 不适用
    gen_tail(out, "tail_plain", [10, 9, 8, 7, 6, 5, 4, 3, 2, 1.0, -1.0, -2.0], seed=102)
    print("fixtures → %s" % os.path.abspath(out))
    for root, _, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, "rb") as fp:
                print("  %-46s %8d  %s" % (os.path.relpath(p, out), os.path.getsize(p),
                                           hashlib.sha256(fp.read()).hexdigest()[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
