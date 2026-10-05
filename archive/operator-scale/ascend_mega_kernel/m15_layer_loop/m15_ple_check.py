#!/usr/bin/env python3.12
"""m15_ple_check.py —— M85 PLE 独立 kernel 的对拍判据（按 docs/17 §1.1 分档）

规则来源（§1.2「规则来源独立」）：`/workspace/vllm/vllm/models/qwen4_exp/nvidia/ops/ple.py`
（① `_ple_ngram_ids_kernel:25-113`、④ `_ple_gate_kernel:185-281`、⑤ `_ple_conv_kernel:319-485`）
与 `nvidia/ple_layer.py:394-429`（③ 的 [key;value] 切分）。本文件把这三处**独立重写**为
numpy/float64 参考（不是调用上游、不是读设备中间量）。

输入来源（§1.3）：全部是本 harness 显式声明的输入（`ple/data/*.bin`，由 `gen_ple_data.py` 生成；
权重取自真实 checkpoint）。

分档（写进 README 的理由见 `ple/README.md`）：
  · `ids[T,16]` int64（含 xor / floor-mod / offset）      → **T1 逐位**
  · `emb[T,2560]`（表行 gather）                          → **T1 逐字节**
  · `kv[T,12800]`（GEMM，k=2560 mmad 累加）               → **T3**（ε 推导见下）
  · `gated/normed/out/conv_state`（Rsqrt/Sigmoid + 归约）  → **T3**

用法：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/m15_ple_check.py m15_layer_loop/ple/out
  ... <outdir> --neg          # 负向对照档（M15_PLE_NEG=1 的产物）：判据**必须** FAIL
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PLE = HERE / "ple"

HID = 2560
HC = 4
W = HID * HC
HE = 2560
P = 8
NGR = 3
NG = 16
HDIM = 160
NC = 2
KVW = HE + W
KCONV = 4
DIL = 3
STLEN = 9
EPS = 1e-6
EOS = 248044
SQ2560 = math.sqrt(2560.0)

# ---- T3 的 ε 推导（docs/17 §1.1「T3 的界必须推导」）----
# 每一项都写明来源，不用"实测最大 X ulp"当界。
# **M124：③ 从 AIV 逐列 GEMV 改成 cube `Mmad` ⇒ kv 的界换成 `mmad 的 k·2^-24` 那一项**
#   （`docs/17` §1.1 的 T3 触发条件 ③「含 mmad / cube 累加」+ 同节「ε 的来源（mmad 的 `k·2^-24`、
#   VF/combine 的若干 ulp……）」）。K = HE = 2560 ⇒ 2560 · 2^-24 = 1.526e-4。
#   这一项**盖住旧的 VF 路径界**（40 个 chunk 的 Mul+Add + 64 lane 归约 ≈ 96·2^-24 = 5.72e-6）
#   ⇒ **改前/改后是同一条界、同一条判据**（`EPS_REDUCE` 保留给 ④ 用），见
#   evidence/ple_wire/M124_CUBE_MMAD.md 的 §读数（两版都用这一条判）。
EPS_MMAD = 2560.0 * 2.0 ** -24   # ③：L0C 里 K=2560 次 fp32 乘加的累加界 = k·2^-24（k=2560）
EPS_REDUCE = 96.0 * 2.0 ** -24   # ④ 的 fp32 逐 chunk 累加：每 lane 每次 Mul+Add 各 1 次舍入
                                 #   （-ffp-contract=off），40 个 chunk ⇒ ~80 次；归约 64 lane ⇒ ~6 次
EPS_RSQRT = 4.0 * 2.0 ** -24     # NR rsqrt（m15_hc_layer.h:147-203）：1 Div + 1 Sqrt + 2 步 NR
EPS_SIGMOID = 2.0 * 2.0 ** -24   # Exp 官方规格 ≈1 ulp（m5_swiglu_quant/README.md:130 实测逐位一致）+ 1 Div


def bf16_rne(x):
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    out = (rounded & np.uint64(0xFFFF0000)).astype(np.uint32)
    return out.view(np.float32)


def bf_ulp(v):
    """bf16 的 1 个格点间距 = 2^(e-7)，e = binade 指数（docs/17 §1.1 的 BfUlp 口径）。"""
    a = np.abs(np.asarray(v, dtype=np.float64))
    a = np.where(a == 0.0, np.finfo(np.float32).tiny, a)
    e = np.floor(np.log2(a))
    return np.power(2.0, e - 7.0)


def ld(path):
    return np.fromfile(path, dtype=np.uint8)


def as_bf16(b):
    return b.view(np.uint16).astype(np.uint32).astype(np.uint32).view(np.uint32)


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def i64(x):
    x = int(x) & ((1 << 64) - 1)
    return x - (1 << 64) if x >= (1 << 63) else x


# ============================================================ 参考实现
def ref_ids(input_ids, qsl, ctx, m, sizes, offsets):
    T = len(input_ids)
    R = len(qsl) - 1
    out = np.zeros((T, NG), dtype=np.int64)
    for t in range(T):
        r = 0
        for i in range(R):
            if qsl[i] <= t:
                r = i
        c = t - qsl[r]
        cur = int(input_ids[t])
        # ctx 列 = NC - shift + c（上游 triton `ops/ple.py:81`）；★ 依赖 c：c=1 时 lag-2 取 ctx[1]
        p1 = int(input_ids[t - 1]) if c >= 1 else int(ctx[r][NC - 1 + c])
        p2raw = int(input_ids[t - 2]) if c >= 2 else int(ctx[r][NC - 2 + c])
        crossed = (p1 == EOS)
        p2 = EOS if crossed else p2raw
        base = i64(cur * int(m[0]))
        t1 = i64(p1 * int(m[1]))
        t2 = i64(p2 * int(m[2]))
        for g in range(NG):
            order = g // P + 2
            mixed = base
            if order > 1:
                mixed = i64(mixed ^ t1)
            if order > 2:
                mixed = i64(mixed ^ t2)
            sz = int(sizes[g])
            r0 = mixed % sz                      # Python 的 % 对正除数是 floor-mod，与 torch.remainder 同
            out[t, g] = r0 + int(offsets[g])
    return out


def ref_norm_gemma(x, w, eps=EPS):
    """分组 Gemma-RMSNorm：组宽 = 末维长度（PLE 里 = 2560）。x/w 为 bf16 值（float32 表示）。"""
    x = x.astype(np.float64)
    var = (x * x).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = x * rstd * (1.0 + w.astype(np.float64))
    return bf16_rne(y)


def ref_gate(key, value, hidden, nk, nq, ncw):
    """④：key/hidden 为 [W]，value 为 [HID]；返回 (gated, normed) 各 [W]。"""
    k = key.astype(np.float64)
    q = hidden.astype(np.float64)
    kn = bf16_rne(k * (1.0 / np.sqrt((k * k).mean() + EPS)) * (1.0 + nk.astype(np.float64)))
    qn = bf16_rne(q * (1.0 / np.sqrt((q * q).mean() + EPS)) * (1.0 + nq.astype(np.float64)))
    prod = bf16_rne(kn.astype(np.float64) * qn.astype(np.float64))
    dot = float(bf16_rne(np.float32(np.float32(prod.astype(np.float32).astype(np.float64)).sum(dtype=np.float64))))
    d = bf16_rne(np.array([np.float32(dot / SQ2560)], dtype=np.float32))[0]
    sign = -1.0 if d < 0 else (1.0 if d > 0 else 0.0)
    mag = bf16_rne(np.array([np.sqrt(max(abs(float(d)), 1e-6))], dtype=np.float32))[0]
    g = bf16_rne(np.array([1.0 / (1.0 + math.exp(-sign * float(mag)))], dtype=np.float32))[0]
    gated = bf16_rne(np.float32(g) * value.astype(np.float32))
    gf = gated.astype(np.float64)
    normed = bf16_rne(gf * (1.0 / np.sqrt((gf * gf).mean() + EPS)) * (1.0 + ncw.astype(np.float64)))
    return gated, normed


def ref_conv_out(gated_row, conv_in, state_rows, wconv):
    """⑤（decode，单 token）：state_rows = 旧状态 [9,W]（行 8 最新）；
    taps lag 9/6/3/0 → state_rows[0]/[3]/[6] 与当前输入。返回 (ple_output[W], conv_output[W])。"""
    taps = [state_rows[0], state_rows[3], state_rows[6], conv_in]        # k=0..3
    acc = np.zeros(W, dtype=np.float64)
    for k in range(KCONV):
        acc += wconv[:, k].astype(np.float64) * taps[k].astype(np.float64)
    conv = bf16_rne(acc)
    y = conv.astype(np.float64) * (1.0 / (1.0 + np.exp(-conv.astype(np.float64))))
    co = bf16_rne(y)
    po = bf16_rne(gated_row.astype(np.float32) + co)
    return po, co


# ============================================================ 判据
class Report:
    def __init__(self):
        self.rows = []
        self.fails = 0

    def add(self, name, tier, n, nbad, detail, bite):
        st = "PASS" if nbad == 0 else "FAIL"
        if nbad:
            self.fails += 1
        self.rows.append((name, tier, n, nbad, st, detail, bite))
        brief = globals().get("BRIEF", False)
        if not brief:
            print("[%s] %-28s tier=%-3s n=%-8d bad=%-6d %s" % (st, name, tier, n, nbad, detail))
            if bite:
                print("       └ 能 FAIL 掉的错误类别: %s" % bite)
        # 机器可读行（变异驱动脚本按它判读；避免解析人读格式）
        print("RESULT|%s|%s|%d|%d|%s" % (name, tier, n, nbad, st))


def cmp_bits(name, rep, got, ref, tier, bite):
    """逐位/逐字节。`n` 与 `bad` 都按**元素**计（字节差另列在 detail 里；M85 r1 的 P3 单位混淆已修）。"""
    refc = np.asarray(ref).astype(got.dtype)
    nbad_e = int((got != refc).sum())
    nbad_b = int((got.view(np.uint8) != refc.view(np.uint8)).sum())
    rep.add(name, tier, int(got.size), nbad_e, "元素逐位（差 %d 元素 / %d 字节）" % (nbad_e, nbad_b), bite)


def cmp_t3(name, rep, got_bf16, ref_f32, eps, bite, extra_terms=None):
    """T3：|out − ref| ≤ ε·Σ|terms| + 1.0·ulp(out)（参考已量化到 bf16 格点 ⇒ 取 1.0·ulp）。"""
    out = got_bf16.astype(np.float64)
    ref = ref_f32.astype(np.float64)
    grid = np.maximum(np.abs(out), np.abs(ref))                      # Σ|terms| 的代理上界：结果量级
    if extra_terms is not None:
        grid = np.maximum(grid, np.asarray(extra_terms, dtype=np.float64))
    bound = eps * grid + 1.0 * bf_ulp(out)
    d = np.abs(out - ref)
    nbad = int((d > bound).sum())
    maxr = float(np.max(d / np.maximum(np.abs(ref), 1e-30)))
    # 裕度报告项：max(d/bound) ≈ 1.0 表示「恰好差 1 个 bf16 格点」（参考本身已量化 ⇒ 合法且由 1.0·ulp 项覆盖）
    marg = float(np.max(d / np.maximum(bound, 1e-300)))
    rep.add(name, "T3", int(out.size), nbad,
            "eps=%.3g maxRel=%.3g max(d/bound)=%.3f (报告项)" % (eps, maxr, marg), bite)


# ============================================================ 主流程
def check_ids(rep, outdir, case, neg):
    meta = (outdir / ("%s_ids_meta.txt" % case)).read_text()
    fields = dict(l.split(" ", 1) for l in meta.strip().split("\n") if " " in l)
    d = PLE / "data"
    ids = np.fromfile(d / ("ids_%s.bin" % case), dtype=np.int32)
    qsl = np.fromfile(d / ("qsl_%s.bin" % case), dtype=np.int32)
    ctx = np.fromfile(d / ("ctx_%s.bin" % case), dtype=np.int32).reshape(-1, NC)
    tag = "full" if case == "A" else "red"
    m = np.fromfile(d / ("m_%s.bin" % tag), dtype=np.int64)
    sizes = np.fromfile(d / ("sizes_%s.bin" % tag), dtype=np.int64)
    offs = np.fromfile(d / ("offsets_%s.bin" % tag), dtype=np.int64)
    got = np.fromfile(outdir / ("%s_ids.bin" % case), dtype=np.int64).reshape(-1, NG)
    ref = ref_ids(ids, qsl, ctx, m, sizes, offs)
    if neg:
        # 负向对照（docs/17 §4）：① 的取模方向被反转（kernel 侧 negMod=1）
        #   判据 1：设备输出 **必须** 与**正确**参考逐位不符（证明主判据 B1 有咬合力）
        #   判据 2：设备输出 **必须** 与「方向反转后的参考」逐位相同（证明变异确实按预期生效）
        n_diff = int((got != ref).sum())
        refl = np.zeros_like(ref)
        for g in range(NG):
            col = ref[:, g] - offs[g]
            refl[:, g] = ((sizes[g] - col) % sizes[g]) + offs[g]
        n_eq_flip = int((got == refl).sum())
        rep.add("N1_neg.ids.%s.bites" % case, "T1-neg", int(got.size), 0 if n_diff > 0 else 1,
                "方向反转后与正确参考不符 %d/%d（须 >0）" % (n_diff, got.size),
                "无（对照项：证明 B1 能抓住 mod 方向错）")
        rep.add("N1_neg.ids.%s.applied" % case, "T1-neg", int(got.size), 0 if n_eq_flip == got.size else 1,
                "与「反转后参考」相符 %d/%d（须全等）" % (n_eq_flip, got.size), "无（对照项）")
        return
    cmp_bits("B1.ids.%s" % case, rep, got, ref, "T1",
             "mod 方向、乘子下标、xor 归 head、EOS 回退、chunk 边界取 ctx、offset 前缀和任一错")
    rng = np.asarray([0, 320001536], dtype=np.int64)
    n_oob = int(((got < rng[0]) | (got >= rng[1])).sum())
    rep.add("B1.ids.%s.range" % case, "T1-struct", int(got.size), n_oob, "值域 [0,320001536)",
            "offset 缺失、mod 未做、负余数未修正")


def check_body(rep, outdir):
    d = PLE / "data"
    meta = json.loads((d / "meta.json").read_text())
    table = bf16_to_f32((d / "table_red.bin").read_bytes()).reshape(-1, HDIM)
    nk = bf16_to_f32((d / "w_norm_key.bin").read_bytes())
    nq = bf16_to_f32((d / "w_norm_query.bin").read_bytes())
    ncw = bf16_to_f32((d / "w_norm_conv.bin").read_bytes())
    wconv = bf16_to_f32((d / "w_conv1d_sq.bin").read_bytes()).reshape(W, KCONV)
    wkey = bf16_to_f32((d / "w_key_proj.bin").read_bytes()).reshape(W, HE)
    wval = bf16_to_f32((d / "w_value_proj.bin").read_bytes()).reshape(HE, HE)
    hidden = bf16_to_f32((d / "hidden.bin").read_bytes()).reshape(-1, W)
    state = bf16_to_f32((d / "conv_state.bin").read_bytes()).reshape(-1, STLEN, W)
    # body 链的 ids 取 `B_ids_body.bin`（body 内部以 negMask=0 复算的 ids），
    # 而**被判的 ① 产物**是 `B_ids.bin`（判据 B1.ids.B 读它）—— 两者刻意分开，见 README §4
    idsB = np.fromfile(outdir / "B_ids_body.bin", dtype=np.int64).reshape(-1, NG)
    T = idsB.shape[0]

    # ② gather（T1 逐字节）
    emb_ref = np.zeros((T, HE), dtype=np.float32)
    for t in range(T):
        for g in range(NG):
            emb_ref[t, g * HDIM:(g + 1) * HDIM] = table[idsB[t, g]]
    p = outdir / "B_emb.bin"
    if p.exists():
        emb = bf16_to_f32(p.read_bytes()).reshape(T, HE)
        cmp_bits("B2.emb", rep, emb, emb_ref, "T1",
                 "行 id 用错（片号/全局 id 混用）、head 顺序反、160/2560 展平顺序错")
    else:
        print("[SKIP] B_emb.bin 不存在（body kernel 未跑）")
        return False

    # ③ kv（T3）
    kv_ref = np.zeros((T, KVW), dtype=np.float32)
    for t in range(T):
        wcat = np.concatenate([wkey, wval], axis=0)          # [12800,2560]，key 在前 value 在后
        kv_ref[t] = bf16_rne(np.float64(wcat) @ emb[t].astype(np.float64))
    kv = bf16_to_f32((outdir / "B_kv.bin").read_bytes()).reshape(T, KVW)
    cmp_t3("B3.kv", rep, kv, kv_ref, EPS_MMAD,
           "key/value 顺序反（docs/14 §6.2 D1）、GEMM 转置错、权重行主序错")
    key = kv[:, :W]
    value = kv[:, W:]

    # ④ gate（T3）+ ⑤ 卷积（T3）
    gated_ref = np.zeros((T, W), dtype=np.float32)
    normed_ref = np.zeros((T, W), dtype=np.float32)
    for t in range(T):
        for s in range(HC):
            gg, nn = ref_gate(key[t, s * HID:(s + 1) * HID], value[t], hidden[t, s * HID:(s + 1) * HID],
                              nk[s * HID:(s + 1) * HID], nq[s * HID:(s + 1) * HID],
                              ncw[s * HID:(s + 1) * HID])
            gated_ref[t, s * HID:(s + 1) * HID] = gg
            normed_ref[t, s * HID:(s + 1) * HID] = nn
    for nm, ref in (("B4.gated", gated_ref), ("B4.normed", normed_ref)):
        p = outdir / ("B_%s.bin" % nm.split(".")[1])
        if p.exists():
            cmp_t3(nm, rep, bf16_to_f32(p.read_bytes()).reshape(T, W), ref, EPS_RSQRT + EPS_SIGMOID,
                   "query 用了 pending BO（未物化）、4 流共享 value 错、`(1+w)` 漏 1、组宽错（2560 vs 10240）")
        else:
            print("[SKIP] %s 缺失" % p.name)
            return False

    # ⑤ conv + 残差 + 状态；**null 槽位行**走上游的 NULL_STATE_ID 分支
    sidx = np.fromfile(d / "state_idx.bin", dtype=np.int32)
    null_rows = set(int(i) for i in range(T) if sidx[i] < 0)
    out_ref = np.zeros((T, W), dtype=np.float32)
    state_ref = np.zeros((T, STLEN, W), dtype=np.float32)
    for t in range(T):
        if t in null_rows:
            # 上游 `ops/ple.py:391-405,438-462`：out_ok=false ⇒ conv_output=0，但仍写
            #   out = bf16( h + bf16( gated + 0 ) )；状态行不写（state_ok=false）
            po = bf16_rne(gated_ref[t].astype(np.float32) + np.float32(0.0))
            out_ref[t] = bf16_rne(hidden[t].astype(np.float32) + po)
            state_ref[t] = state[t]                       # 保持不变
        else:
            po, _ = ref_conv_out(gated_ref[t], normed_ref[t], state[t], wconv)
            out_ref[t] = bf16_rne(hidden[t].astype(np.float32) + po)
            state_ref[t] = np.vstack([state[t][1:], normed_ref[t][None, :]])
    if null_rows:
        # B5.null（T1）：null 行的 out 是纯 bf16 加链（无超越函数）⇒ 可按 T1 逐位判；
        #   同时 null 行的状态必须**逐字节不变**（设备侧不写 + host 预置）
        got = bf16_to_f32((outdir / "B_out.bin").read_bytes()).reshape(T, W)
        rows = np.array(sorted(null_rows))
        cmp_bits("B5.null.out", rep, got[rows].ravel(), out_ref[rows].ravel(), "T1",
                 "null 行整行跳过（漏写）/ 把 null 行当普通行做卷积")
        st_dev = bf16_to_f32((outdir / "B_state_out.bin").read_bytes()).reshape(T, STLEN, W)
        cmp_bits("B5.null.state", rep, st_dev[rows].ravel(), state[rows].ravel(), "T1",
                 "null 行的状态被移位（state_ok 未生效）")
    p = outdir / "B_out.bin"
    if p.exists():
        cmp_t3("B5.out", rep, bf16_to_f32(p.read_bytes()).reshape(T, W), out_ref,
               EPS_RSQRT + EPS_SIGMOID,
               "taps lag 9/6/3/0 错、SiLU 漏、先加卷积再加外层残差的顺序反、外层残差用了 pending BO")
    else:
        print("[SKIP] B_out.bin 缺失")
        return False
    p = outdir / "B_state_out.bin"
    if p.exists():
        st = bf16_to_f32(p.read_bytes()).reshape(T, STLEN, W)
        cmp_t3("B5.state", rep, st, state_ref, 0.0,
               "移位方向反（新→旧）、最新样本未写入行 8、状态未演化")
    else:
        print("[SKIP] B_state_out.bin 缺失")
        return False
    # 非空洞性（T4）：**设备产物** state_out 必须真的不同于**设备输入** state_in
    #   （M85 r1 的 P3：旧版只比参考侧的 ref_out vs 输入，设备侧没有同口径判据 —— 已改为设备对设备）
    # M124：两点加固 —— ① `act` 显式取 int64（全是 null 行时空列表会被 numpy 当成 float64，
    #   索引直接抛 `IndexError: arrays used as indices must be of integer ...`，M=1 档实测踩到）；
    #   ② `act` 为空时**本档没有活跃行** ⇒ 这一条**不可判**，按 `[SKIP]` 如实跳过（不判 PASS ⇒ 不给空判据）。
    act = (np.array([i for i in range(T) if i not in null_rows], dtype=np.int64)
           if null_rows else np.arange(T, dtype=np.int64))
    if act.size == 0:
        print("[SKIP] B5.state_evolve：本档 %d 行全是 null 槽位（无活跃行）⇒ 本条不可判"
              "（不判 PASS，避免空判据）" % T)
    else:
        dsz_dev = int((st[act] != state[act]).sum())
        dsz_ref = int((state_ref[act] != state[act]).sum())
        rep.add("B5.state_evolve", "T4", int(st[act].size), 0 if (dsz_dev > 0 and dsz_ref > 0) else 1,
                "活跃行 state_out != state_in 的元素 = %d；参考侧 = %d（null 行 %d 行不计）"
                % (dsz_dev, dsz_ref, len(null_rows)),
                "状态没写回 / 写回与读同一行 / 设备输出整片退化")

    # ---- M111：② 自己的**设备侧计数**（`PleGather` 数「词表外的 id」）----
    # 为什么必须有一条判据读它：`B_body_meta.txt` 的 `dev_fail` 以前读的是 ④⑤ 的 `bad`，
    # 而那个 `bad` 从不自增（`(void)bad;`）⇒ 恒 0 ⇒ **永不可能失败的判据**（REAL_TABLE.md §6.4 末段）。
    # M111 把计数搬进 `PleGather`（真的数），并让落盘走 UB → MTE3 `DataCopyPad`；这一条把它判起来，
    # 咬合力由变异 **bit15** 演示（`id += VOCAB_ROWS` ⇒ 该计数 = 被跳过 item 数）。
    p = outdir / "B_body_meta.txt"
    if p.exists():
        f = dict(l.split(" ", 1) for l in p.read_text().strip().split("\n") if " " in l)
        dev_fail = int(f["dev_fail"])
        rep.add("B_dev.fail", "T1-struct", T * NG, 0 if dev_fail == 0 else 1,
                "设备侧 ② 的词表外 id 计数 = %d（host 只做求和；落盘 = UB → MTE3 DataCopyPad）" % dev_fail,
                "① 的值域错、② 的输入被污染（id 越出词表 [0,%d)）" % 320001536)
    else:
        print("[SKIP] B_body_meta.txt 缺失（body kernel 未跑）⇒ B_dev.fail 不判")
    return True


# ============================================================ M92：host-mapped 真实表
HM = PLE / "data_hm"


def hm_meta():
    return json.loads((HM / "hm_meta.json").read_text())


def real_row(meta, gid):
    """**直接读该分片文件**：行 id → (shard, local) → `data_off + local*320` 读 320 B。

    这是被验方（设备侧 host-mapped 窗口）的**独立对照源**：设备读的是注册窗口，
    host 在这里按规格算式从原始分片文件取同一行 —— 两条路径不共用任何中间量。
    """
    shard, local = divmod(int(gid), int(meta["rows_per_shard"]))
    off = int(meta["data_off"]) + local * int(meta["row_bytes"])
    with open(meta["file"], "rb") as fh:
        fh.seek(off)
        b = fh.read(int(meta["row_bytes"]))
    assert len(b) == int(meta["row_bytes"]), (gid, len(b))
    return b


def cmp_bits_masked(name, rep, got, ref, mask, tier, bite):
    """逐位/逐字节，只在 `mask`（元素级 bool，与 got 同形状）选中的元素上比（M92 HM 判据用）。"""
    refc = np.asarray(ref).astype(got.dtype)
    sel = np.broadcast_to(mask, got.shape)
    n = int(sel.sum())
    nbad_e = int((got[sel] != refc[sel]).sum())
    gb = got.view(np.uint8).reshape(got.shape + (-1,))[sel]
    rb = refc.view(np.uint8).reshape(refc.shape + (-1,))[sel]
    nbad_b = int((gb != rb).sum())
    rep.add(name, tier, n, nbad_e,
            "元素逐位（差 %d 元素 / %d 字节；n=%d 在 mask 内）" % (nbad_e, nbad_b, n), bite)


def check_hm(rep, outdir):
    """M92：host-mapped 真实表的判据（①/② 的 T1 + ③④⑤ 的 T3 + 设备侧计数 + 非空洞守卫）。"""
    meta = hm_meta()
    d = HM
    hm = dict(l.split(" ", 1) for l in (outdir / "H_hm_meta.txt").read_text().strip().split("\n")
              if " " in l)
    ng = int(hm["ng"])
    T = int(hm["n_tok"])
    items = T * ng
    win_rows = int(hm["win_rows"])
    win_base = int(hm["r0"])
    ids = np.fromfile(outdir / "H_ids.bin", dtype=np.int64).reshape(T, ng)
    inw = (ids >= win_base) & (ids < win_base + win_rows)          # item 级窗口掩码
    n_in_item = int(inw.sum())
    n_out_item = items - n_in_item

    # ---- 期望行：**直读分片文件**（独立对照源）----
    rows_ref = np.zeros((items, HDIM), dtype=np.float32)
    nz_elems = 0
    for t in range(T):
        for g in range(ng):
            i = t * ng + g
            if inw[t, g]:
                raw = real_row(meta, int(ids[t, g]))
                v = bf16_to_f32(raw)
                rows_ref[i] = v
                nz_elems += int((v != 0).sum())
    # 非空洞守卫：比较双方都不是常量/全零（M85 教训：0 vs 0 的空过比没有判据更糟）
    rep.add("H2.emb.nonvac", "T1-struct", n_in_item * HDIM, 0 if (nz_elems > 0) else 1,
            "对照行（直读分片文件）的非零元素 = %d / %d；窗口内 item = %d" % (nz_elems, n_in_item * HDIM,
                                                                        n_in_item),
            "整片为 0 ⇒ 逐字节判据是空过（比较双方同为常量）")

    # ---- ② emb（T1 逐字节，仅窗口内 item）----
    emb_got = bf16_to_f32((outdir / "H_emb.bin").read_bytes()).reshape(T, HE)
    emb_ref = np.zeros((T, HE), dtype=np.float32)
    for t in range(T):
        for g in range(ng):
            emb_ref[t, g * HDIM:(g + 1) * HDIM] = rows_ref[t * ng + g]
    m_emb = np.zeros((T, HE), dtype=bool)
    for g in range(ng):
        m_emb[:, g * HDIM:(g + 1) * HDIM] = inw[:, g][:, None]
    cmp_bits_masked("H2.emb", rep, emb_got, emb_ref, m_emb, "T1",
                    "id→窗口行号换算错（方向/基址）、head 落点错、行宽错、窗口未注册而读失败")

    # ---- 下游 ③④⑤：只在「该 token 的 16 个 head 全在窗口内」的行上比 ----
    tok_in = inw.all(axis=1)
    n_tok_in = int(tok_in.sum())
    wd = PLE / "data"          # PLE 的真实小张量（与缩减表案同一份：gen_ple_data.py 是唯一来源）
    nk = bf16_to_f32((wd / "w_norm_key.bin").read_bytes())
    nq = bf16_to_f32((wd / "w_norm_query.bin").read_bytes())
    ncw = bf16_to_f32((wd / "w_norm_conv.bin").read_bytes())
    wconv = bf16_to_f32((wd / "w_conv1d_sq.bin").read_bytes()).reshape(W, KCONV)
    wkey = bf16_to_f32((wd / "w_key_proj.bin").read_bytes()).reshape(W, HE)
    wval = bf16_to_f32((wd / "w_value_proj.bin").read_bytes()).reshape(HE, HE)
    hidden = bf16_to_f32((d / "hidden_hm.bin").read_bytes()).reshape(T, W)
    state = bf16_to_f32((d / "conv_state_hm.bin").read_bytes()).reshape(-1, STLEN, W)
    sidx = np.fromfile(d / "state_idx_hm.bin", dtype=np.int32)

    kv_got = bf16_to_f32((outdir / "H_kv.bin").read_bytes()).reshape(T, KVW)
    kv_ref = np.zeros((T, KVW), dtype=np.float32)
    kv_sumabs = np.zeros((T, KVW), dtype=np.float64)     # Σ_k|w·emb|：T3 界的正确尺度（见下）
    wcat = np.concatenate([wkey, wval], axis=0)
    awcat = np.abs(wcat.astype(np.float64))
    for t in range(T):
        e = emb_ref[t].astype(np.float64)
        kv_ref[t] = bf16_rne(wcat.astype(np.float64) @ e)
        kv_sumabs[t] = awcat @ np.abs(e)
    # 只在 token 级窗口内的行比（其余 token 的 emb 里有未落盘的 head，属 harness 的已知不覆盖范围）
    # ★ 界用 **Σ|terms|**（docs/17 §1.1 的 T3 原式），不用 M85 的 max(|out|,|ref|) 代理：
    #   真实表行 + 真实权重下 GEMV 会出现**相消**（|Σ terms| ≪ Σ|terms|），代理会让界偏紧
    #   （实测：代理界下 6/806400 元素越界、maxRel=2.82；换成 Σ|terms| 后见读数）。
    cmp_t3_masked("H3.kv", rep, kv_got, kv_ref, tok_in, EPS_MMAD,
                  "key/value 顺序反、GEMM 转置错；本条同时是 ② 的正确性下游证据（emb 错 ⇒ kv 错）",
                  extra_terms=kv_sumabs)

    gated_got = bf16_to_f32((outdir / "H_gated.bin").read_bytes()).reshape(T, W)
    nor_got = bf16_to_f32((outdir / "H_normed.bin").read_bytes()).reshape(T, W)
    gated_ref = np.zeros((T, W), dtype=np.float32)
    normed_ref = np.zeros((T, W), dtype=np.float32)
    key = kv_ref[:, :W]
    value = kv_ref[:, W:]
    for t in range(T):
        for s in range(HC):
            gg, nn = ref_gate(key[t, s * HID:(s + 1) * HID], value[t],
                              hidden[t, s * HID:(s + 1) * HID], nk[s * HID:(s + 1) * HID],
                              nq[s * HID:(s + 1) * HID], ncw[s * HID:(s + 1) * HID])
            gated_ref[t, s * HID:(s + 1) * HID] = gg
            normed_ref[t, s * HID:(s + 1) * HID] = nn
    cmp_t3_masked("H4.gated", rep, gated_got, gated_ref, tok_in, EPS_REDUCE + EPS_RSQRT + EPS_SIGMOID,
                  "query 用了 pending BO、4 流共享 value 错、`(1+w)` 漏 1", ulp_mult=2.0)
    cmp_t3_masked("H4.normed", rep, nor_got, normed_ref, tok_in, EPS_REDUCE + EPS_RSQRT + EPS_SIGMOID,
                  "同上 + normed 的分组归约", ulp_mult=2.0)

    out_got = bf16_to_f32((outdir / "H_out.bin").read_bytes()).reshape(T, W)
    slots = np.array([int(v) for v in sidx], dtype=np.int64)
    out_ref = np.zeros((T, W), dtype=np.float32)
    for t in range(T):
        if slots[t] < 0:
            # null 槽位（NULL_STATE_ID）：conv_output = 0，但仍写 out = bf16(h + bf16(gated + 0))
            po = bf16_rne(gated_ref[t].astype(np.float32) + np.float32(0.0))
            out_ref[t] = bf16_rne(hidden[t].astype(np.float32) + po)
        else:
            po, _ = ref_conv_out(gated_ref[t], normed_ref[t], state[slots[t]], wconv)
            out_ref[t] = bf16_rne(hidden[t].astype(np.float32) + po)
    cmp_t3_masked("H5.out", rep, out_got, out_ref, tok_in, EPS_REDUCE + EPS_RSQRT + EPS_SIGMOID,
                  "taps lag 9/6/3/0 错、SiLU 漏、残差顺序反（② 的错会一路传到这里）", ulp_mult=2.0)

    # ---- 设备侧判据（覆盖 M85 §7 U6）：host 只读 GM，不做任何设备侧判定 ----
    devf = np.frombuffer((outdir / "H_hm_dev.bin").read_bytes(), dtype=np.float32).reshape(-1, 8)
    n_cores = int((devf[:, 2] != 0.0).sum())
    n_core_exp = 2 * 28
    dev_row_fail = float(devf[:, 0].sum())
    dev_miss = int(devf[:, 1].sum())
    win_rows_dev = sorted(set(devf[:, 3][devf[:, 2] != 0.0].tolist()))
    rep.add("Hd.cores", "T1-struct", n_core_exp, 0 if n_cores == n_core_exp else 1,
            "设备侧自报参与核 = %d / %d；设备侧读到的 winRows = %s" % (n_cores, n_core_exp, win_rows_dev),
            "核没跑 ②（判据落点为空）或 winRows 参数没传到核里")
    rep.add("Hd.row_fail", "T1-struct", n_in_item, 0 if dev_row_fail == 0.0 else 1,
            "设备侧「行首 64 元素 vs host 直读期望」错行计数 = %.0f（窗口内 item %d）"
            % (dev_row_fail, n_in_item),
            "该计数 > 0 ⇔ 设备取到的行 ≠ 该 id 应有的行（id→偏移换算错、窗口内容错、head 落点错）")
    rep.add("Hd.miss", "T1-struct", items, 0 if dev_miss == n_out_item else 1,
            "设备侧越窗行计数 = %d；host 独立算出的越窗 item = %d" % (dev_miss, n_out_item),
            "越窗判定的窗口范围参数错（winBase/winRows 与注册长度不符）")
    # 非空洞守卫：本条判据必须**真的有东西可比**（窗口内有 item）且**真的比过**
    rep.add("Hd.nonvac", "T1-struct", items, 0 if (n_in_item > 0 and n_cores == n_core_exp) else 1,
            "窗口内 item = %d（>0 才算比过）；设备侧参与核 = %d（=2×AIC 才算真的跑了 ②）"
            % (n_in_item, n_cores),
            "窗口内没有 item 或没有核跑 ② ⇒ Hd.row_fail 是空过")
    return True


def cmp_t3_masked(name, rep, got_bf16, ref_f32, tok_mask, eps, bite, extra_terms=None,
                  ulp_mult=1.0):
    """T3，按 token 行选择（HM 案里只有「16 个 head 全在窗口内」的 token 才可判）。

    `extra_terms` = Σ|terms|（正确尺度；不给就退回 max(|out|,|ref|) 代理，那在相消时偏紧）。
    `ulp_mult`：格点项系数。1.0 = 「参考已量化到 bf16 格点，最多差 1 格点」；
    2.0 用于**输出本身是某个 bf16 中间量的乘/加**的场合：若那个中间量（如门控标量 `g`）
    被允许差 1 个 bf16 格点，则 `g·v` 与 `g_ref·v` 相差 ≈ `ulp(out)`，再经输出自己的
    bf16 量化最坏可到 **2 个格点**（推导：`out=bf16(g·v)`，`|Δg| = ulp(g)` ⇒ `|Δ(g·v)| = ulp(g)|v| ≈ ulp(out)`，
    量化后 ≤ `ulp(out)` 的 2 倍）。
    """
    sub = got_bf16[tok_mask].astype(np.float64)
    rsub = ref_f32[tok_mask].astype(np.float64)
    grid = np.maximum(np.abs(sub), np.abs(rsub))
    if extra_terms is not None:
        grid = np.maximum(grid, np.asarray(extra_terms, dtype=np.float64)[tok_mask])
    bound = eps * grid + ulp_mult * bf_ulp(sub)
    d = np.abs(sub - rsub)
    nbad = int((d > bound).sum())
    maxr = float(np.max(d / np.maximum(np.abs(rsub), 1e-30))) if d.size else 0.0
    marg = float(np.max(d / np.maximum(bound, 1e-300))) if d.size else 0.0
    rep.add(name, "T3", int(sub.size), nbad,
            "eps=%.3g ulp系数=%.1f maxRel=%.3g max(d/bound)=%.3f（%d 行 token 参与；界尺度=%s）"
            % (eps, ulp_mult, maxr, marg, int(tok_mask.sum()),
               "Σ|terms|" if extra_terms is not None else "max(|out|,|ref|) 代理"), bite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--neg", action="store_true")
    ap.add_argument("--brief", action="store_true", help="只打机器可读的 RESULT 行（供变异驱动脚本判读）")
    args = ap.parse_args()
    globals()["BRIEF"] = bool(args.brief)
    outdir = pathlib.Path(args.outdir)
    rep = Report()
    for c in ("A", "B"):
        if (outdir / ("%s_ids.bin" % c)).exists():
            check_ids(rep, outdir, c, args.neg)
        else:
            print("[SKIP] %s_ids.bin 缺失（kernel 未跑该矩阵）" % c)
    if not args.neg and (outdir / "B_emb.bin").exists():
        check_body(rep, outdir)
    # M92：host-mapped 真实表案（H_*.bin；同一脚本、同一分档规则）
    if (outdir / "H_emb.bin").exists():
        check_hm(rep, outdir)
    n_judge = len([r for r in rep.rows if "neg" not in r[0]])
    n_ctrl = len(rep.rows) - n_judge
    if not args.brief:
        # ★ 本脚本**不声明**"有咬合力 N 条"：那是一个**被验证过的计数**，只能由变异矩阵给出
        #   （`m15_ple_mutants.py` 的 `RESULT|coverage|` 行）。M85 r2 复审 P3：旧版把
        #   "非取反行条数" 印成 "有咬合力"，是标签冒充计数。
        print("\n===== 判据合计 %d 条 = 判定项 %d + 对照项 %d；FAIL %d 条 =====" %
              (len(rep.rows), n_judge, n_ctrl, rep.fails))
        print("       注：'咬合力被演示'的条数见 m15_ple_mutants.py 的 RESULT|coverage 行（本脚本不声明）")
    if args.neg:
        print("===== 负向对照档：上表 PASS = 对照成立（主判据在变异输入上必 FAIL，见 README §判据）=====")
        return 0 if rep.fails == 0 else 1
    print("===== %s =====" % ("ALL PASS" if rep.fails == 0 else "FAILURES PRESENT"))
    return 0 if rep.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
