#!/usr/bin/env python3.12
"""m19_qsa_indexer/check_ref.py —— QSA indexer 前端的 numpy/double 参考 + 判据

参考实现严格按 /workspace/vllm/vllm/models/qwen4_exp/ 的语义（见 README §1 契约）：

  1. proj   = bf16_rne( fp64(x[1,2560]) @ W[640,2560]^T )            （AIC GEMM 段）
  2. q      = rope_neox_partial( gemma_norm(bf16(proj[:, :512]).reshape(4,128), wq),
                                 pos=p, 只旋 dims[0,64)，neox 配对 (j, j+32) )
  3. k_raw  = bf16(proj[:, 512:640])                                  （raw key，不 norm 不 rope）
  4. 若 p ≡ 3 (mod 4)：pooled = bf16_rne( mean_fp32( k_raw(p-3..p) ) )
                        ck     = rope_neox_partial( gemma_norm(pooled, wk), pos=4g )
  5. score[b] = Σ_h max( dot_fp32(ck[b], q[h]), 0 )     （无 scale；只对 b < V 计）
  6. 选中集合 = top-512（block_topk = budget/ratio），离散判据见 §判据
  7. packed 输出 [2052] int32：列 [0,4*min(V,512)) = 选中 block 的 token（4b+{0..3}），
     紧随 tail（open group 的 token），其余 -1；末列 2051 = 4*min(V,512) + tail_count

用法：
  /usr/local/python3.12.13/bin/python3.12 m19_qsa_indexer/check_ref.py [dump 目录，默认 ./m19_out] [case 名...]
"""

import math
import pathlib
import sys

import numpy as np

HIDDEN = 2560
NH = 4
D = 128
QW = NH * D
RATIO = 4
BUDGET = 2048
BLK_TOPK = BUDGET // RATIO
OUT_WIDTH = BUDGET + RATIO - 1
PACKED_WIDTH = OUT_WIDTH + 1
ROT = 64
HALF = ROT // 2
EPS = 1e-6

HERE = pathlib.Path(__file__).resolve().parent

# ---- T3 的界（docs/17 §1.1）：eps 逐项推导见 README §3.2 ----
#   eps_cos    = 2^-9   cos/sin 表按 vLLM 同精度取整 bf16
#   eps_bf16   = 2^-9   每个 norm/rope 结果落盘一次 bf16 量化
#   eps_domain = 2*2^-9 运筹域差异：融合核在 bf16 上旋转 vs 本实现 fp32 旋转、单次取整
EPS_T3 = 2.0 ** -9 + 2.0 ** -9 + 2.0 * (2.0 ** -9)   # ≈ 1.5e-3


def ulp_f32(x: np.ndarray) -> np.ndarray:
    """fp32 的 ulp(x)（以 2 的幂近似上界：2^-23 * 2^floor(log2|x|)）。"""
    x = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log2(np.maximum(x, np.finfo(np.float32).tiny)))
    return (2.0 ** -23) * (2.0 ** e)


def t3_check(tag: str, out: np.ndarray, ref: np.ndarray, terms: np.ndarray):
    """T3 逐元素判据；返回 (判定项, 违反元素数, maxRel, <=1ulp 比例)。"""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    bound = EPS_T3 * terms + 0.5 * ulp_f32(ref)
    bad = np.abs(out - ref) > bound
    nbad = int(bad.sum())
    rel = np.abs(out - ref) / np.maximum(np.abs(ref), 1e-30)
    ulp = np.abs(out - ref) / np.maximum(ulp_f32(ref), 1e-45)
    frac = float((ulp <= 1.0).mean())
    print(f"[{tag}] T3: 违反元素 {nbad}/{out.size}；maxRel(report) {rel.max():.3e}；"
          f"≤1ulp 比例(report) {frac * 100:.2f}%")
    if nbad:
        idx = np.argwhere(bad).ravel()[:5]
        for i in idx:
            print(f"     超界元素 #{i}: out {out.flat[i]:.6g} ref {ref.flat[i]:.6g} "
                  f"Σ|terms| {terms.flat[i]:.6g} 界 {bound.flat[i]:.6g}")
    return (nbad == 0), nbad, float(rel.max()), frac


# ---------------------------------------------------------------- 基础算子
def bf16_round(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16(RNE) -> fp32（与 kernel 的整型 RNE 公式逐位一致）。"""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    return (r << 16).view(np.float32)


def load_bf16(path: pathlib.Path, shape=None) -> np.ndarray:
    a = np.fromfile(path, dtype=np.uint16)
    if shape:
        a = a.reshape(shape)
    return (a.astype(np.uint32) << 16).view(np.float32)


def gemma_norm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """GemmaRMSNorm: y = (x / sqrt(mean(x^2)+eps)) * (1 + w)，fp32 计算。"""
    x = np.asarray(x, dtype=np.float64)
    rstd = 1.0 / np.sqrt((x * x).mean() + EPS)
    return (x * rstd * (1.0 + w.astype(np.float64))).astype(np.float32)


def rope_neox_partial(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """只旋 dims [0,64)：neox 配对 (j, j+32)，cos/sin 为 32 个频率对。"""
    out = np.array(x, dtype=np.float64, copy=True)
    a = x[:HALF].astype(np.float64)
    b = x[HALF:ROT].astype(np.float64)
    c = cos.astype(np.float64)
    s = sin.astype(np.float64)
    out[:HALF] = a * c - b * s
    out[HALF:ROT] = b * c + a * s
    return out


# ---------------------------------------------------------------- 参考流水线
class RefCase:
    def __init__(self, dump: pathlib.Path, name: str):
        self.dir = dump
        self.name = name
        meta = {}
        for line in (dump / f"{name}_meta.txt").read_text().splitlines():
            k, _, v = line.partition(" ")
            meta[k] = v
        self.pos = int(meta["pos"])
        self.V = int(meta["vblocks"])
        self.closes = int(meta["closes"])
        self.gidx = int(meta["group_index"])
        self.tail_start = int(meta["tail_start"])
        self.tail_count = int(meta["tail_count"])
        # 有效 blk_topk（budget/ratio）；缺省回退到本模型常量 512
        self.blk_topk = int(meta.get("blk_topk", BLK_TOPK))
        self.ring = load_bf16(dump / f"{name}_ring.bin", (RATIO, D))
        self.comp_in = load_bf16(dump / f"{name}_comp.bin", (self.V, D))
        self.cosq = load_bf16(dump / f"{name}_cosq.bin")
        self.sinq = load_bf16(dump / f"{name}_sinq.bin")
        self.cosg = load_bf16(dump / f"{name}_cosg.bin")
        self.sing = load_bf16(dump / f"{name}_sing.bin")

    def reference_math(self):
        x = load_bf16(self.dir / f"{self.name}_x.bin", (2, HIDDEN))[0].astype(np.float64)
        W = load_bf16(HERE / "data" / "layer3_index_qk_proj.bin", (640, HIDDEN)).astype(np.float64)
        wq = load_bf16(HERE / "data" / "layer3_q_layernorm.bin")
        wk = load_bf16(HERE / "data" / "layer3_k_layernorm.bin")
        proj = bf16_round((x @ W.T).astype(np.float32))          # GEMM 输出（bf16）
        q_raw = proj[:QW].reshape(NH, D)
        k_raw = bf16_round(proj[QW:QW + D])
        q = np.stack([
            bf16_round(rope_neox_partial(gemma_norm(q_raw[h], wq), self.cosq, self.sinq))
            for h in range(NH)
        ])
        ck = None
        if self.closes:
            acc = np.zeros(D, dtype=np.float64)
            for i in range(RATIO):                                # 位置序左结合
                tok = self.pos - (RATIO - 1) + i
                row = k_raw if i == RATIO - 1 else self.ring[tok % RATIO].astype(np.float64)
                acc = acc + row.astype(np.float64)
            pooled = bf16_round((acc / RATIO).astype(np.float32))
            ck = bf16_round(rope_neox_partial(gemma_norm(pooled, wk), self.cosg, self.sing))
        # 打分：内核覆盖了 comp[gidx]（= 本步新压缩行）
        comp = self.comp_in.astype(np.float64).copy()
        if self.closes:
            comp[self.gidx] = ck.astype(np.float64)
        q64 = q.astype(np.float64)
        score = np.zeros(self.V, dtype=np.float64)
        for b in range(self.V):
            acc = 0.0
            for h in range(NH):
                d = float(comp[b] @ q64[h])
                if d > 0.0:
                    acc += d
            score[b] = acc
        return q, k_raw, ck, score


# ---------------------------------------------------------------- 判据
def check(dump: pathlib.Path, name: str) -> bool:
    c = RefCase(dump, name)
    q_ref, k_ref, ck_ref, score_ref = c.reference_math()
    ok = True

    # 判据 1：index_qk_proj（内核 bf16 输出 vs 参考的双精度→bf16）
    qk = load_bf16(dump / f"{name}_qk.bin", (640,))
    x = load_bf16(dump / f"{name}_x.bin", (2, HIDDEN))[0].astype(np.float64)
    W = load_bf16(HERE / "data" / "layer3_index_qk_proj.bin", (640, HIDDEN)).astype(np.float64)
    proj_ref = bf16_round((x @ W.T).astype(np.float32))
    ulp = np.abs(qk.view(np.uint16).astype(np.int32) - proj_ref.view(np.uint16).astype(np.int32))
    print(f"[{name}] proj   : 与参考 bf16 最大位差 {ulp.max()} （0 = 逐位一致）")
    ok &= bool(ulp.max() == 0)

    # 判据 2/3：q 与压缩行 —— T3 逐元素（|out-ref| ≤ eps·Σ|terms| + 0.5·ulp(out)）
    qkern = np.fromfile(dump / f"{name}_qout.bin", dtype=np.float32).reshape(NH, D)
    qr = q_ref.astype(np.float64)
    wq = load_bf16(HERE / "data" / "layer3_q_layernorm.bin")
    # Σ|terms|：逐元素取**真实项和**——norm 部分 |x·rstd·(1+w)|，rope 部分 |a·c| + |b·s|
    xraw = proj_ref[:QW].reshape(NH, D).astype(np.float64)   # bf16 投影（与 kernel 逐位一致）
    rstd = 1.0 / np.sqrt((xraw * xraw).mean(axis=-1, keepdims=True) + EPS)
    y = np.abs(xraw * rstd * (1.0 + wq.astype(np.float64)))
    terms_q = np.array(y, copy=True)
    for h in range(NH):
        for j in range(HALF):
            terms_q[h, j] = y[h, j] * np.abs(c.cosq[j]) + y[h, HALF + j] * np.abs(c.sinq[j])
            terms_q[h, HALF + j] = y[h, HALF + j] * np.abs(c.cosq[j]) + y[h, j] * np.abs(c.sinq[j])
    okq, nq, rq, fq = t3_check(f"{name} q", qkern.astype(np.float64), qr, terms_q)
    ok &= okq
    if ck_ref is not None:
        ck = load_bf16(dump / f"{name}_ckout.bin", (D,)).astype(np.float64)
        ckr = ck_ref.astype(np.float64)
        wk = load_bf16(HERE / "data" / "layer3_k_layernorm.bin")
        acc = np.zeros(D)
        for i in range(RATIO):
            tok = c.pos - (RATIO - 1) + i
            row = (load_bf16(dump / f"{name}_kout.bin", (D,)).astype(np.float64) if i == RATIO - 1
                   else c.ring[tok % RATIO].astype(np.float64))
            acc = acc + row
        pooled = bf16_round((acc / RATIO).astype(np.float32)).astype(np.float64)
        rstdk = 1.0 / np.sqrt((pooled * pooled).mean() + EPS)
        yk = np.abs(pooled * rstdk * (1.0 + wk.astype(np.float64)))
        tck = np.array(yk, copy=True)
        for j in range(HALF):
            tck[j] = yk[j] * np.abs(c.cosg[j]) + yk[HALF + j] * np.abs(c.sing[j])
            tck[HALF + j] = yk[HALF + j] * np.abs(c.cosg[j]) + yk[j] * np.abs(c.sing[j])
        okc, nc, rc, fc = t3_check(f"{name} ck", ck, ckr, tck)
        ok &= okc

    # 判据 4：分数 —— T3 逐元素（Σ|terms| = Σ_h Σ_d |q_hd·k_bd|）
    logits = np.fromfile(dump / f"{name}_logits.bin", dtype=np.float32)
    comp = c.comp_in.astype(np.float64).copy()
    if c.closes:
        comp[c.gidx] = ck_ref.astype(np.float64)
    q64 = q_ref.astype(np.float64)
    terms = np.zeros(c.V)
    for b in range(c.V):
        acc = 0.0
        for h in range(NH):
            acc += float(np.abs(comp[b] * q64[h]).sum())
        terms[b] = acc
    oks, ns, rs, fs = t3_check(f"{name} scores", logits[:c.V].astype(np.float64), score_ref, terms)
    ok &= oks

    # 判据 5（离散，核心）：选中集合
    #
    # **为什么这里要用两条互补的判据**（M53 结论，见 README §3.4）：
    #   (i)  **精确判据（判定项）——在 kernel 自己 dump 的分数向量上**做「全排序 + 平局按索引升序」
    #        的暴力 top-k，要求逐块相等。选择段的正确性就是这件事（它只能基于自己拿到的分数选），
    #        这条是精确的、任何置换都会被抓到。
    #   (ii) **条带判据（判定项）——对 double 参考分数**：分数只有 T3 保真度（|out−ref| ≤ B(b)，
    #        见判据 3 已 0 违反），而参考分数的第 k 名与邻居的间隙常小于 B ⇒ 边界块的取舍在
    #        **契约层面本来就不可判**。可证的判据是（K=参考第 k 大，Bmax=max_b B(b)）：
    #          分数 > K + 2·Bmax 的 block **必须**选中；< K − 2·Bmax 的 **必须**不选中；
    #          落在 ±2·Bmax 内的**不可判**（只报数，不判）。
    #        证明：|K_out − K| ≤ Bmax（第 k 位序统计量对最大范数扰动 1-Lipschitz），故
    #        s(b) > K+2Bmax ⇒ s_out(b) > K+Bmax ≥ K_out ⇒ 必被任何正确的 top-k 选中（反之同理）。
    #   ⇒ 旧版「miss/extra 必须为 0」在 T3 分数下**不可能满足**（实测 A 档 ±0.05 内就有 12 个 block），
    #     本轮把它降级为**报告项**，换成上面两条判定项；判据强度没有下降（多了一条精确判据）。
    out = np.fromfile(dump / f"{name}_out.bin", dtype=np.int32)
    BK = c.blk_topk
    expanded = out[: BK * RATIO]
    blocks = set()
    block_tokens = {}
    bad = 0
    for t in expanded:
        if t < 0:
            continue
        b, r = int(t) // RATIO, int(t) % RATIO
        if r != 0 and r != 1 and r != 2 and r != 3:
            bad += 1
        blocks.add(b)
        block_tokens.setdefault(b, set()).add(r)
    complete = min(c.V, BK)
    cnt_col = int(out[OUT_WIDTH])
    expect_cnt = complete * RATIO + c.tail_count
    from collections import Counter
    cnt_occ = Counter(t // RATIO for t in expanded if t >= 0)
    dup = [b for b, n in cnt_occ.items() if n != RATIO]
    # 每块必须恰好出现 r = {0,1,2,3}（只数次数会漏过 {0,0,1,2} 这类）
    dup += sorted(b for b, rs in block_tokens.items() if rs != set(range(RATIO)))
    dup = sorted(set(dup))
    print(f"[{name}] blocks : 选中 {len(blocks)} 个（期望 {complete}），出现次数异常的 block {len(dup)}，"
          f"count 列 {cnt_col}（期望 {expect_cnt}）")
    ok &= bool(len(blocks) == complete and not dup and cnt_col == expect_cnt and bad == 0)
    if dup:
        print(f"        ✗ 出现次数 != {RATIO} 的 block：{dup[:10]}")

    # 判据 5(i)：在 kernel 自己的分数上做暴力 top-k（精确）
    logits = np.fromfile(dump / f"{name}_logits.bin", dtype=np.float32)[: c.V].astype(np.float64)
    n_nan = int(np.isnan(logits).sum())
    order = np.argsort(-logits, kind="stable")[:complete]
    ideal = set(int(i) for i in order)
    exact_missing = sorted(ideal - blocks)
    exact_extra = sorted(blocks - ideal)
    print(f"[{name}] 离散(i): 与「kernel 自身分数」的暴力 top-{complete} 逐块差集 {len(ideal ^ blocks)}"
          f"（漏 {len(exact_missing)} / 误 {len(exact_extra)}）；logits NaN {n_nan}")
    ok &= bool(len(ideal ^ blocks) == 0 and n_nan == 0)
    if ideal ^ blocks:
        print(f"        ✗ 漏选 {exact_missing[:10]}；误选 {exact_extra[:10]}")

    # 判据 5(ii)：对参考分数的可证条带判据
    if c.V > BK:
        terms_b = np.zeros(c.V)
        for b in range(c.V):
            acc = 0.0
            for h in range(NH):
                acc += float(np.abs(comp[b] * q64[h]).sum())
            terms_b[b] = acc
        bound = EPS_T3 * terms_b + 0.5 * ulp_f32(score_ref)
        bmax = float(bound.max())
        kth = np.sort(score_ref)[-BK]
        band = 2.0 * bmax
        n_gt = int((score_ref > kth).sum())
        raw_miss = [b for b in np.where(score_ref > kth)[0] if b not in blocks]
        raw_extra = [b for b in blocks if score_ref[b] < kth]
        must_sel = np.where(score_ref > kth + band)[0]
        must_not = np.where(score_ref < kth - band)[0]
        n_und = int((np.abs(score_ref - kth) <= band).sum())
        ms = [b for b in must_sel if b not in blocks]
        mn = [b for b in must_not if b in blocks]
        print(f"[{name}] 离散(ii): 第 {BK} 大 score = {kth:.6g}，>K 的 block {n_gt}；参考带宽 ±{band:.4g}"
              f"（Bmax {bmax:.4g}）⇒ 必选 {len(must_sel)} / 必不选 {len(must_not)} / 不可判 {n_und}；"
              f"必选漏 {len(ms)}，必不选误 {len(mn)}")
        print(f"                报告项（不做判定，见判据 5 说明）：旧口径 miss {len(raw_miss)} / extra {len(raw_extra)}")
        ok &= bool(not ms and not mn)
        if ms or mn:
            print(f"        ✗ 必选漏 {ms[:10]}；必不选误 {mn[:10]}")

    # 判据 6：尾部与 padding
    # tail 在**模型固定列** [BUDGET, BUDGET+tail_count)（M19_BUDGET 覆盖不改布局，见 README §1.6/§4）
    tail_col0 = BUDGET
    tail = out[tail_col0:tail_col0 + c.tail_count]
    tail_ok = all(int(t) == c.tail_start + i for i, t in enumerate(tail))
    gap_ok = all(int(t) == -1 for t in out[min(c.V, BK) * RATIO:tail_col0])
    pad_ok = gap_ok and all(int(t) == -1 for t in out[tail_col0 + c.tail_count:OUT_WIDTH])
    print(f"[{name}] tail   : {list(map(int, tail))} 期望 {[c.tail_start + i for i in range(c.tail_count)]}；padding {'OK' if pad_ok else 'BAD'}")
    ok &= bool(tail_ok and pad_ok)
    return ok


def main() -> int:
    dump = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./m19_out")
    names = sys.argv[2:] or [
        p.name[: -len("_meta.txt")] for p in sorted(dump.glob("*_meta.txt"))
    ]
    if not names:
        print(f"[check_ref] {dump} 下没有 *_meta.txt（先运行 m19_qsa_indexer 生成 dump）")
        return 2
    all_ok = True
    for n in names:
        try:
            all_ok &= check(dump, n)
        except Exception as exc:  # noqa: BLE001
            print(f"[{n}] 检查异常: {exc}")
            all_ok = False
    print(f"[check_ref] ===== {'ALL PASS' if all_ok else 'FAILURES PRESENT'} =====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
