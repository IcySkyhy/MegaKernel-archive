#!/usr/bin/env python3
"""m26_moe_prefill 的独立判据（numpy float64 参考 + 非空洞性 + 负向对照）

被测对象 = `m15_layer_loop/m15_moe_prefill.h` 的 **S2（router）**：
    logits[MT, E_PAD] = x[MT, HIDDEN] bf16 @ W_pad[E_PAD, HIDDEN]^T bf16   （AIC 上的 Mmad，fp32 累加）
    ids[MT, TOPK] / weights[MT, TOPK] = softmax(max-shift) + top-TOPK + renorm   （AIV）
    sgate[MT] = x @ W_pad[E]^T                                              （共享门裸点积）

档位（`docs/17-verification-standard.md`）：
  · **T3**（含 mmad / cube 累加）—— 判据 `|got - exp| ≤ ε·Σ|terms| + 0.5·ulp_fp32(exp)`，
    ε **推导**而非实测（推导见 §容差）。
  · `ids` 是**选择**运算（比较），不是算术 ⇒ 用「参考 top-TOPK 逐元素相等」判，
    并配一条 **tie 风险 guard**：只有参考的 10/11 名间隔 ≫ 容差的那些行才要求逐位相等
    （间隔小的行如实记为 tie-risk，不计入 FAIL，但**计入报告项**）。

三类输出（`docs/17` §2）：
  · **判定项**（决定 rc）  · **报告项**（max / 间隔 / 占比）  · **guard**（非空洞性，红了也是 FAIL）
rc：0 = 全绿；1 = 有判定项/guard 红；2 = 数据缺失（SKIPPED）。

## M155 新增的三条判据位（判据侧；dump 侧接线约定见 README §7）

  · **J1c（行内 top-k 顺序/置换）**：与 `J1b` 互补 —— `J1b` 只判**集合**（忽略列内顺序），
    `J1c` 只在「设备集合 == 参考集合」的行上判**列内顺序**。它补的是 `J1` 的空档：
    `n_safe == 0`（tie-risk 门把 `J1` 整条跳过，check_ref.py 的 J1 门控）时，一个把 top-k
    两列对调的行内置换不会被 `J1b`（集合不变）抓到；`J1c` 只对**参考的相邻列间隔 > 4×行容差**
    的相邻对判（tie 对不判），故它比 `J1` 的「整行 safe」门细一档。配套负向对照 `--negctl 8`
    **只在 `J1` 跳过的行**上置换：尾块（`MT > m`）时 `[m, MT)` 的 pad 行不属于被判行、不置换；
    若被判行全被 `J1` 判则报「负向对照未生效（V8 不适用）」，不空转、不抛异常。
  · **块覆盖（逐块/首块）**：dump 侧提供逐块文件（`m26_meta.b<i>.txt` + `m26_*.b<i>.bin`）时
    **逐块**用各自那一块的 x 现算参考并对拍；只有 legacy 单块时**明确报**「本档判据只覆盖 1 块，
    无法判定它是首块还是末块」，不把「只看最后一块」说成覆盖了首块。
  · **W1 输入面消费见证**：dump 侧提供**消费前快照**（`m26_x_consumed[.b<i>].bin`）或
    **消费时 checksum**（meta 的 `x_consumed_sha256`）时，用它区分「算错」与「读错输入」；
    未提供时**明确报**「未提供、不可区分」。W1 是**报告项**：它不改变旧档的 PASS/FAIL。

## 判据的适用范围与空档（**写窄**）
  · `J1`/`J1c` 都只在「参考的间隔显著」处判顺序 ⇒ 参考本身**并列**（tie）的相邻对无法判；
    参考容差 `tol_logit` 越大，可判的相邻对越少。
  · `J1c` 只在**集合正确**的行上生效 ⇒ 设备选错集合时它不判顺序（那已由 `J1b` 判红）。
  · 块覆盖：判据侧能逐块比，但**逐块落盘由 dump 侧决定**；本档只报「覆盖了哪些块 / 缺哪些块」，
    对未落盘的块没有数值判据（首块是竞争下风险最高的一块，尤其如此）。
  · W1 只有在 dump 侧提供快照/checksum 时才有判别力；提供时它区分「算错 / 读错输入」，
    不提供时只能如实记「不可区分」。
"""

import argparse
import glob
import hashlib
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DUMP = os.path.join(_HERE, "build", "m26_out")


class DataMissing(SystemExit):
    """某个 block 的 bin 缺失/短读。继承 SystemExit ⇒ 未捕获时行为与旧版一致（rc=1 + 消息）。"""


class NegctlInapplicable(Exception):
    """负向对照在该档没有可造的错（例：V8 找不到 J1 跳过的行）⇒ 报"不适用"，不崩、不空转。"""


def _short(path, got, want):
    return DataMissing("FAIL: %s short (%d != %d)" % (path, got, want))

U_F32 = 2.0 ** -24.0          # fp32 单位舍入
K_ACC = 2560                  # mmad 的 K（2560 个 bf16 乘积，fp32 累加）
# ---- ε 的推导（不是实测）----
# 每个乘积 bf16×bf16 在 fp32 里**精确**（8+8 ≤ 24 位尾数）；40 个 64 项块的顺序累加共 K-1 = 2559 次加法，
# 最坏相对误差 ≤ γ_{K-1} = (K-1)u / (1-(K-1)u)。⇒ ε_acc = γ_2559。
EPS_ACC = (K_ACC - 1) * U_F32 / (1.0 - (K_ACC - 1) * U_F32)
# exp/div/Reduce 三个 fp32 步的相对误差（10 项和 → γ_9；exp 1 次 ReadOnly；div 1 次）：
EPS_W = 9 * U_F32 + 2 * U_F32


def bf16_to_f32(u16):
    u = u16.astype(np.uint32) << 16
    return u.view(np.float32) if u.dtype == np.uint32 else u.view(np.float32)


def read_u16(path, elems):
    try:
        a = np.fromfile(path, dtype="<u2", count=elems)
    except OSError as e:
        raise DataMissing("FAIL: %s 缺文件/不可读（%s）" % (path, e))
    if a.size != elems:
        raise _short(path, a.size, elems)
    return a


def read_f32(path, elems):
    try:
        a = np.fromfile(path, dtype="<f4", count=elems)
    except OSError as e:
        raise DataMissing("FAIL: %s 缺文件/不可读（%s）" % (path, e))
    if a.size != elems:
        raise _short(path, a.size, elems)
    return a


def read_i32(path, elems):
    try:
        a = np.fromfile(path, dtype="<i4", count=elems)
    except OSError as e:
        raise DataMissing("FAIL: %s 缺文件/不可读（%s）" % (path, e))
    if a.size != elems:
        raise _short(path, a.size, elems)
    return a


def read_meta_file(path):
    """meta 的键一律尝试转 int（现有字段都是整数）；转不了就原样留字符串（如 checksum）。"""
    meta = {}
    with open(path, "r") as fp:
        for line in fp:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    meta[k] = int(v)
                except ValueError:
                    meta[k] = v
    return meta


class Judge(object):
    def __init__(self):
        self.items = []       # 判定项
        self.reports = []
        self.guards = []

    def chk(self, name, got, exp, terms, eps):
        tol = eps * np.abs(terms) + 0.5 * np.spacing(np.abs(exp).astype(np.float32)).astype(np.float64)
        bad = int(np.count_nonzero(np.abs(got - exp) > tol))
        self.items.append((name, bad, got.size))
        return bad

    def chk_exact(self, name, got, exp):
        bad = int(np.count_nonzero(got != exp))
        self.items.append((name, bad, got.size))
        return bad

    def chk_count(self, name, bad, total):
        """按"元素个数"计的判定项（J1c 这种逐对比较用）。"""
        self.items.append((name, int(bad), int(total)))
        return int(bad)

    def rep(self, name, value):
        self.reports.append((name, value))

    def gd(self, name, ok, detail=""):
        self.guards.append((name, bool(ok), detail))
        return bool(ok)

    def dump(self):
        nb = sum(b for _, b, _ in self.items)
        nt = sum(t for _, _, t in self.items)
        print("== 判定项 %d 条：%d PASS / %d FAIL（判定元素 %d，超界 %d）" %
              (len(self.items), len(self.items) - sum(1 for _, b, _ in self.items if b),
               sum(1 for _, b, _ in self.items if b), nt, nb))
        for name, bad, tot in self.items:
            print("   [%s] %s (%d/%d)" % ("PASS" if bad == 0 else "FAIL", name, tot - bad, tot))
        print("== 报告项 %d 条" % len(self.reports))
        for name, v in self.reports:
            print("   %-52s %s" % (name, v))
        gbad = [n for n, ok, _ in self.guards if not ok]
        print("== guard %d 条：%s" % (len(self.guards), "全绿" if not gbad else "红: " + ", ".join(gbad)))
        for name, ok, detail in self.guards:
            print("   [%s] %s %s" % ("OK " if ok else "RED", name, detail))
        return nb, len(gbad)


def negctl(variant, arr):
    """负向对照：在**内存里**把被测读数弄坏（不写任何文件），判据必须变红。"""
    a = arr.copy()
    if variant == 1:            # V1：把第 0 行第 0 个 logit 挪 1%（判据应抓到）
        a[0, 0] = a[0, 0] + 0.01 * max(1.0, abs(float(a[0, 0])))
    elif variant == 2:          # V2：整列置零（跨行相关性被破坏）
        a[:, 5] = 0.0
    elif variant == 3:          # V3：把最后一行整行按另一个行的值覆盖（非空洞性应抓到）
        a[-1, :] = a[0, :]
    elif variant == 4:          # V4：全部置常数（最粗暴的空洞）
        a[:] = 0.5
    else:
        raise SystemExit("unknown negctl variant %d" % variant)
    return a


def negctl_ids(variant, ids, w, safe=None):
    """负向对照（ids/weights 的可证伪性）：只改**内存里的副本**，不写任何文件。

    r1 复审的 nit：原来 4 个变体只污染 logits ⇒ `J1/J1b`（ids）与 `J4`（weights）从未被演示变红。
    `safe` = J1 的 tie-risk 门（True = 该行 J1 会判），只覆盖被判的 `[0,m)` 行；
    V8 用它把置换**只加在 J1 跳过的行**上。
    """
    i = ids.copy()
    ww = w.copy()
    if variant == 5:            # V5：把行 1 的 top-k 换成行 0 的 ⇒ ids 族与 S3 派生量必须红
        i[1, :] = i[0, :]
    elif variant == 6:          # V6：把第 0 行第一个权重挪 50%（破坏 renorm 行和）⇒ J4/G4 必须红
        ww[0, 0] = ww[0, 0] * 0.5
    elif variant == 7:          # V7：行内把第 0/1 列对调（集合不变、顺序与 S3 派生量必须红）
        i[:, 0], i[:, 1] = i[:, 1].copy(), i[:, 0].copy()
    elif variant == 8:          # V8：**只在 J1 被跳过的行**上把第 0/1 列对调 ⇒ 只有 J1c 能抓
        # `safe` 只覆盖被判的 [0,m) 行，而 ids 有 MT 行（尾块 MT > m）⇒ 必须按
        # [0, min(MT, len(safe))) 对齐；多出的 pad 行（≥ m）不属于被判行，**不置换**。
        # 若没有任何被判行被 J1 跳过（safe 全 True），该负控在此档**无意义** ⇒ 明确报"不适用"，
        # 不静默空转、也不崩（IndexError）。
        n = i.shape[0]
        if safe is None:
            gate = np.ones(n, dtype=bool)
        else:
            gate = np.zeros(n, dtype=bool)
            k = min(n, int(np.asarray(safe).size))
            gate[:k] = ~np.asarray(safe, dtype=bool)[:k]
        if not gate.any():
            mrows = 0 if safe is None else int(np.asarray(safe).size)
            raise NegctlInapplicable(
                "V8 不适用：被判的 %d 行（MT=%d）没有一行被 J1 跳过 ⇒ 该档没有「J1 抓不到的行内置换」可造"
                % (mrows, n))
        i[gate, 0], i[gate, 1] = ids[gate, 1].copy(), ids[gate, 0].copy()
    else:
        raise SystemExit("unknown negctl variant %d" % variant)
    return i, ww


# ---------------------------------------------------------------------------
# 逐块驱动（判据侧接口；见 README §7）
# ---------------------------------------------------------------------------
def discover_blocks(dump):
    """返回 [(suffix, meta), ...]。

    多块约定（dump 侧接线后生效）：`m26_meta.b<i>.txt` + `m26_*.b<i>.bin`，suffix = `.b<i>`。
    无 `m26_meta.b*.txt` ⇒ legacy 单块（suffix = ""，仅 `m26_meta.txt`）。
    """
    out = []
    for p in sorted(glob.glob(os.path.join(dump, "m26_meta.b*.txt"))):
        m = re.search(r"m26_meta\.b(\d+)\.txt$", os.path.basename(p))
        if m is None:
            continue
        i = int(m.group(1))
        meta = read_meta_file(p)
        meta["_blk"] = int(meta.get("blk", i))
        out.append((".b%d" % i, meta))
    if out:
        return out
    return [("", read_meta_file(os.path.join(dump, "m26_meta.txt")))]


def coverage(J, dump, blocks):
    """块覆盖报告：多块逐块判；legacy 单块**明确报**只覆盖 1 块。"""
    blockaware = blocks[0][0] != ""
    if not blockaware:
        J.rep("覆盖：档位块范围",
              "legacy 单块 dump（meta 无 blk/m_total/n_tiles）⇒ 本档判据**只覆盖 1 块**；"
              "无法从 meta 判定它是首块还是末块（落盘实现固定为**最后一块**）"
              "⇒ m>MT 时**首块及其余块无判据**")
        return
    covered = sorted(int(meta.get("blk", -1)) for _, meta in blocks)
    covered = [c for c in covered if c >= 0]
    ntiles = set()
    for _, meta in blocks:
        if "n_tiles" in meta:
            ntiles.add(int(meta["n_tiles"]))
        if "m_total" in meta and "MT" in meta:
            mt = int(meta["MT"])
            if mt > 0:
                ntiles.add((int(meta["m_total"]) + mt - 1) // mt)
    if len(ntiles) == 1:
        N = sorted(ntiles)[0]
        missing = [i for i in range(N) if i not in covered]
        J.rep("覆盖：档位块范围",
              "已判块 %s / 共 %d 块%s" % (covered, N,
                                          "" if not missing else "；**缺块 %s（本档无判据）**" % missing))
        J.rep("覆盖：首块(blk=0)",
              "已覆盖" if 0 in covered else "**未覆盖**（首块是这次竞争下风险最高的一块，本档无判据）")
    else:
        J.rep("覆盖：档位块范围",
              "已判块 %s；meta 的 n_tiles/m_total 缺失或互不一致 %s ⇒ 无法判定缺哪些块" %
              (covered, sorted(ntiles)))


def witness_input(J, dump, suffix, nm, x_u, w, E, m, mt, H, logits_got, logits_ref, tol_logit, meta):
    """W1：区分「算错」与「读错输入」。

    证据源（任选其一，dump 侧接线后提供）：
      · `m26_x_consumed[.b<i>].bin`：**消费前**的 x 快照；与回读面 `m26_x[.b<i>].bin` 比。
      · meta 的 `x_consumed_sha256=<hex>`：消费时对 x 算的校验和；与回读面的 sha256 比。
    两者都没有 ⇒ 如实报「未提供、不可区分」。W1 是报告项，不改变旧档 PASS/FAIL。
    """
    key = nm("W1 输入面消费见证")
    base_path = os.path.join(dump, "m26_x%s.bin" % suffix)
    snap_path = os.path.join(dump, "m26_x_consumed%s.bin" % suffix)
    sha = meta.get("x_consumed_sha256")
    if os.path.exists(snap_path):
        snap_u = read_u16(snap_path, mt * H)
        if np.array_equal(snap_u, x_u):
            J.rep(key, "消费快照 == 回读面（逐字节）⇒ 回读的 x 就是设备消费的输入")
            return
        nd = int(np.count_nonzero(snap_u != x_u))
        xs = bf16_to_f32(snap_u).astype(np.float64).reshape(mt, H)
        lr_s = xs @ w[:E].T
        t_s = np.abs(xs) @ np.abs(w[:E]).T
        tol_s = EPS_ACC * t_s + 0.5 * np.spacing(np.abs(lr_s).astype(np.float32)).astype(np.float64)
        bad_s = int(np.count_nonzero(np.abs(logits_got[:m, :E] - lr_s[:m]) > tol_s[:m]))
        bad_b = int(np.count_nonzero(np.abs(logits_got[:m, :E] - logits_ref[:m]) > tol_logit[:m]))
        if bad_s == 0 and bad_b > 0:
            v = "**读错输入**：logits 与消费快照一致（超界 0），与回读面不一致（超界 %d）" % bad_b
        elif bad_s > 0 and bad_b == 0:
            v = "异常：logits 与回读面一致（超界 0）而与消费快照不一致（超界 %d）⇒ 快照侧记录可疑" % bad_s
        elif bad_s > 0 and bad_b > 0:
            v = "**算错**：logits 与消费快照、回读面都不一致（超界 %d / %d）" % (bad_s, bad_b)
        else:
            v = "两者都能解释 logits（数值上区分不出）"
        J.rep(key, "消费快照 != 回读面（%d/%d 元素不同）⇒ %s" % (nd, snap_u.size, v))
        return
    if sha:
        with open(base_path, "rb") as fp:
            h = hashlib.sha256(fp.read()).hexdigest()
        same = (h == str(sha))
        J.rep(key, "无快照文件；meta 报消费时 checksum=%s，回读面 checksum=%s ⇒ %s" %
              (str(sha)[:16], h[:16],
               "一致（回读面 = 消费输入）" if same else "**不一致 ⇒ 回读面不是设备消费的输入（读错输入）**"))
        return
    J.rep(key, "**未提供**（无 m26_x_consumed%s.bin、meta 无 x_consumed_sha256）"
               "⇒ 本档**不可区分**「算错」与「读错输入」" % suffix)


def judge_one(J, dump, suffix, meta, args, blockaware):
    """判一块。legacy（suffix=="") 的读数与旧版**逐条相同**（只是新增 J1c/W1）。"""
    pfx = ("[b%d] " % int(meta.get("blk", -1))) if blockaware else ""

    def nm(s):
        return pfx + s

    def path(base):
        return os.path.join(dump, "m26_%s%s.bin" % (base, suffix))

    m = meta["m"]
    mt = meta["MT"]
    E = meta["E"]
    TP_MAX = meta["TP_MAX"] if "TP_MAX" in meta else mt * meta["topk"]
    E_PAD = meta["E_PAD"]
    TOPK = meta["topk"]
    H = meta["hidden"]

    x_u = read_u16(path("x"), mt * H)
    w_u = read_u16(os.path.join(dump, "m26_wpad.bin"), E_PAD * H)
    logits_got = read_f32(path("logits"), mt * E_PAD).reshape(mt, E_PAD)
    ids_got = read_i32(path("ids"), mt * TOPK).reshape(mt, TOPK)
    w_got = read_f32(path("w"), mt * TOPK).reshape(mt, TOPK)
    sg_got = read_f32(path("sgate"), mt).reshape(mt)

    x = bf16_to_f32(x_u).astype(np.float64).reshape(mt, H)
    w = bf16_to_f32(w_u).astype(np.float64).reshape(E_PAD, H)

    # ---- 参考：fp64 ----
    logits_ref = x @ w[:E].T                                # [mt, E]
    terms = np.abs(x) @ np.abs(w[:E]).T                     # Σ|terms|（容差的项和）
    sg_ref = x @ w[E]                                       # [mt]

    # 选 top-TOPK（降序；并列时小 id 在前 —— 与设备 Sort32/归并树的索引模板同序）
    order = np.argsort(-logits_ref, axis=1, kind="stable")
    ids_ref = order[:, :TOPK]
    l_top = np.take_along_axis(logits_ref, ids_ref, axis=1)
    # 10/11 名间隔（tie 风险 guard 用）
    gap = logits_ref[np.arange(mt), order[:, TOPK - 1]] - logits_ref[np.arange(mt), order[:, TOPK]]
    # 参考权重（fp64）
    v = np.exp(l_top - l_top[:, :1])
    w_ref = v / np.sum(v, axis=1, keepdims=True)
    # 权重对 logits 的梯度上界：Σ_j |∂w_k/∂l_j| |l_j| = w_k(|l_k| + Σ_j w_j|l_j|)
    lw = np.sum(w_ref * np.abs(l_top), axis=1, keepdims=True)
    w_terms = w_ref * (np.abs(l_top) + lw)

    tol_logit = EPS_ACC * terms + 0.5 * np.spacing(np.abs(logits_ref).astype(np.float32)).astype(np.float64)
    # ---- J1 ids：只在"参考的 10/11 名间隔 > 4×本行最大 logit 容差"的行上要求逐位相等 ----
    row_tol = np.max(tol_logit, axis=1)
    safe_full = gap > 4.0 * row_tol          # 在**全部** mt 行上算（gap/tol 都是 mt 长的向量）
    safe = safe_full[:m]                     # 判据只覆盖 [0, m) 行
    n_safe = int(np.count_nonzero(safe))

    # ---- 负向对照（在内存副本上做，判据必须变红）----
    if args.negctl in (1, 2, 3, 4):
        logits_got = negctl(args.negctl, logits_got)
    elif args.negctl in (5, 6, 7, 8):
        try:
            ids_got, w_got = negctl_ids(args.negctl, ids_got, w_got, safe)
        except NegctlInapplicable as e:
            J.rep(nm("负向对照未生效"), str(e))

    if n_safe > 0:
        J.chk_exact(nm("J1 topk_ids（safe 行；%d/%d 行）" % (n_safe, m)),
                    ids_got[:m][safe], ids_ref[:m][safe])
    J.chk_exact(nm("J1b topk_ids 的**集合**（全部行：忽略列内顺序）"),
                np.sort(ids_got[:m], axis=1), np.sort(ids_ref[:m], axis=1))

    # ---- J1c：行内 top-k 顺序/置换（与 J1b 集合判互补；补 J1 在 n_safe==0 时被跳过的空档）----
    # 门 1：只在"设备集合 == 参考集合"的行上判（集合错的行由 J1b 负责，避免把两种错混在一起）。
    # 门 2：只在**参考的相邻列间隔 > 4×行容差**的相邻对上判（tie 对不判）。
    set_row = np.all(np.sort(ids_got[:m], axis=1) == np.sort(ids_ref[:m], axis=1), axis=1)
    pairs_total = 0
    viol_total = 0
    if TOPK >= 2:
        agap = l_top[:m, :-1] - l_top[:m, 1:]                    # [m, TOPK-1]：参考序的相邻间隔
        sig = agap > (4.0 * row_tol[:m])[:, None]                # [m, TOPK-1]：显著的相邻对
        rows_ok = np.where(set_row)[0]
        for r in rows_ok:
            pos = {int(v): j for j, v in enumerate(ids_got[r])}
            pj = np.array([pos[int(v)] for v in ids_ref[r]])     # 参考序每个 id 在设备序里的位置
            s = sig[r]
            pairs_total += int(np.count_nonzero(s))
            viol_total += int(np.count_nonzero((pj[:-1] > pj[1:]) & s))
    if pairs_total > 0:
        J.chk_count(nm("J1c topk_ids 行内顺序（set 一致行；显著相邻对 %d 个）" % pairs_total),
                    viol_total, pairs_total)
        J.rep(nm("J1c 覆盖"), "set 一致行 %d/%d；显著相邻对 %d；逆序 %d"
              % (int(np.count_nonzero(set_row)), m, pairs_total, viol_total))
    else:
        J.rep(nm("J1c 行内顺序判据未生效"),
              "本档无「集合一致 + 参考间隔显著」的相邻对 ⇒ 行内顺序**未被本判据覆盖**"
              "（集合错由 J1b 判；tie 对参考自己也分不出）")
    # ---- J2 logits（T3，逐元素）----
    J.chk(nm("J2 router_logits（T3）"), logits_got[:m, :E], logits_ref[:m, :], terms[:m, :], EPS_ACC)
    # ---- J3 共享门裸点积（T3）----
    J.chk(nm("J3 sgate 裸点积（T3）"), sg_got[:m], sg_ref[:m],
          np.abs(x[:m]) @ np.abs(w[E]), EPS_ACC)
    # ---- J4 权重（T3；项和取解析梯度上界）----
    J.chk(nm("J4 topk_weights（T3，梯度上界）"), w_got[:m], w_ref[:m], w_terms[:m], EPS_ACC + EPS_W)

    # ---- J5..J9：S3（计数排序）+ S4（permute）的**索引数据**逐位判据（T1）----
    # 为什么这是 P1-1 的"能咬住的判据"：这一族只看**设备自己的 ids**（不依赖 tie-risk 的 J1）
    # 推导出来的 counts/offsets/perm/inv/xsorted。若 AIV0 在 S3 里读到**尚未落地**的 ids
    # （= r1 复审 P1-1 的竞态），它算出的 counts/perm 会是"别的 ids 的计数排序"，
    # 与用 dump 出来的（正确的）ids 现算的参考**必然不一致** ⇒ J5..J9 变红。
    p_counts = path("counts")
    if os.path.exists(p_counts):
        counts_got = read_i32(p_counts, E)
        offsets_got = read_i32(path("offsets"), (E + 1) + 8)[:E + 1]
        psrc_got = read_i32(path("perm_src"), TP_MAX)[:m * TOPK]
        pexp_got = read_i32(path("perm_exp"), TP_MAX)[:m * TOPK]
        inv_got = read_i32(path("inv"), mt * TOPK)[:m * TOPK]
        # 逐字节比较 ⇒ 读**原始 bf16 位**（不做 fp32 转换，避免把比较变成浮点比较）
        xn_u = read_u16(path("xnorm"), mt * H).reshape(mt, H)
        xs_u = read_u16(path("xsorted"), TP_MAX * H).reshape(TP_MAX, H)

        flat = ids_got[:m].reshape(-1)                       # 设备自己的 ids（m*TOPK 条）
        S = flat.size
        order_s3 = np.argsort(flat, kind="stable")            # 稳定计数排序（同键保序）
        pos_of = np.empty(S, dtype=np.int64)
        pos_of[order_s3] = np.arange(S)
        counts_ref = np.bincount(flat, minlength=E)[:E]
        offsets_ref = np.concatenate([[0], np.cumsum(counts_ref)])
        J.chk_exact(nm("J5 S3 counts[E]"), counts_got, counts_ref)
        J.chk_exact(nm("J6 S3 offsets[E+1]"), offsets_got, offsets_ref)
        J.chk_exact(nm("J7 S3 perm_src/perm_expert（稳定计数排序）"),
                    np.concatenate([psrc_got, pexp_got]),
                    np.concatenate([(order_s3 // TOPK).astype(np.int32), flat[order_s3].astype(np.int32)]))
        J.chk_exact(nm("J8 S3 inv_slot[t,k] = 该 (t,k) 在 permuted 序里的位置"),
                    inv_got, pos_of.astype(np.int32))
        # J9 permute = 纯 gather：xsorted[pos] 必须**逐字节**等于 xnorm[perm_src[pos]]
        src_rows = (order_s3 // TOPK)[:S]                     # = perm_src
        J.chk_exact(nm("J9 S4 xsorted[pos] == xnorm[perm_src[pos]]（逐字节）"),
                    xs_u[:S], xn_u[src_rows])
        J.rep(nm("S3 Σt_e"), "%d (期望 m*TOPK = %d)" % (int(offsets_got[E]), m * TOPK))
        # G7 是**覆盖见证**：只有在 m ≥ 8 的档上"多写者读"这条依赖才真的可能存在，
        # 才把它当判定项；小 m 档如实记为报告项（不假装它成立，也不用它去"通过"判据）。
        if m >= 8:
            J.gd(nm("G7 S3 的跨核覆盖成立（多写者 + m 足够）"),
                 mt > 1 and meta["nblk"] * 2 >= 8 and m >= 8,
                 "(nAiv=%d, m=%d)" % (meta["nblk"] * 2, m))
        else:
            J.rep(nm("G7 小 m 档（m=%d）：S3 的多写者路径本档不覆盖" % m),
                  "nAiv=%d；多写者覆盖见 m=64 档的 G7" % (meta["nblk"] * 2))

    # ---- (c) W1：区分「算错」与「读错输入」----
    witness_input(J, dump, suffix, nm, x_u, w, E, m, mt, H, logits_got, logits_ref, tol_logit, meta)

    # ---- 报告项 ----
    d = np.abs(logits_got[:m, :E] - logits_ref[:m])
    J.rep(nm("logits max|Δ|"), "%.4e" % float(d.max()))
    J.rep(nm("logits 容差占用（max |Δ|/tol）"), "%.4f" % float(np.max(d / (tol_logit[:m] + 1e-30))))
    J.rep(nm("min 10/11 名间隔（reference）"), "%.4e" % float(gap[:m].min()))
    J.rep(nm("tie-risk 行数（gap ≤ 4×tol）"), "%d / %d" % (m - n_safe, m))
    J.rep(nm("weights max|Δ|"), "%.4e" % float(np.abs(w_got[:m] - w_ref[:m]).max()))
    J.rep(nm("sgate max|Δ|"), "%.4e" % float(np.abs(sg_got[:m] - sg_ref[:m]).max()))
    J.rep(nm("logits 行内跨专家极差（min over rows）"),
          "%.4e" % float((logits_got[:m, :E].max(axis=1) - logits_got[:m, :E].min(axis=1)).min()))
    J.rep(nm("sgate 跨行极差"), "%.4e" % float(sg_got[:m].max() - sg_got[:m].min()))

    # ---- guard（非空洞性）----
    J.gd(nm("G1 logits 行内跨专家非常量（极差 > 1e-3）"),
         float((logits_got[:m, :E].max(axis=1) - logits_got[:m, :E].min(axis=1)).min()) > 1e-3)
    J.gd(nm("G2 logits 跨行非常量（行 0 与行 m-1 不相等）"),
         m < 2 or not np.array_equal(logits_got[0, :E], logits_got[m - 1, :E]))
    J.gd(nm("G3 ids 全在 [0, E) 且行内互异"),
         bool(np.all((ids_got[:m] >= 0) & (ids_got[:m] < E))) and
         all(len(set(ids_got[r].tolist())) == TOPK for r in range(m)))
    J.gd(nm("G4 weights 行和为 1（±1e-5）"),
         bool(np.all(np.abs(w_got[:m].sum(axis=1) - 1.0) < 1e-5)))
    J.gd(nm("G5 权重非退化（行内不是单一 1.0 全给第一名）"),
         bool(np.all((w_got[:m] > 0.0).sum(axis=1) >= 2)))
    J.gd(nm("G6 sgate 跨行非常量"),
         float(sg_got[:m].max() - sg_got[:m].min()) > 1e-6 or m == 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default=DEFAULT_DUMP)
    ap.add_argument("--negctl", type=int, default=0,
                    help="1..4 弄坏 logits；5..8 弄坏 ids/weights（8 = 只在 J1 跳过的行上置换，专测 J1c）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    blocks = discover_blocks(args.dump)
    blockaware = blocks[0][0] != ""
    J = Judge()
    skipped = []
    for suffix, meta in blocks:
        try:
            judge_one(J, args.dump, suffix, meta, args, blockaware)
        except DataMissing as e:
            label = ("[b%d] " % int(meta.get("blk", -1))) if blockaware else ""
            J.rep(label + "块数据缺失（SKIPPED）", str(e))
            skipped.append(suffix)
    coverage(J, args.dump, blocks)

    nb, gbad = J.dump()
    if not args.quiet:
        src = os.path.join(_HERE, os.pardir, "m15_layer_loop", "m15_moe_prefill.h")
        if os.path.exists(src):
            with open(src, "rb") as fp:
                print("== 被测源 sha256: %s" % hashlib.sha256(fp.read()).hexdigest()[:16])
    if len(skipped) == len(blocks):
        print("RESULT: SKIPPED（%d 块全部缺数据，无可比）" % len(skipped))
        return 2
    print("RESULT: %s" % ("PASS" if (nb == 0 and gbad == 0) else "FAIL"))
    return 0 if (nb == 0 and gbad == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
