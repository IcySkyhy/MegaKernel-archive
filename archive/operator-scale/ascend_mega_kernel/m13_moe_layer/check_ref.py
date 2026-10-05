#!/usr/bin/env python3
"""M13（MoE block layer kernel）的 numpy 独立交叉校验。

复用 `tools/golden/moe_block_ref.py`（权威 golden：MXFP4 打包/解包、bf16 RNE、
路由语义）与 `M13_DUMP=1` 落盘的 device 中间张量，做三件 host 侧 C++ 判据**没有**覆盖的事：

1. **量化器取值质量**（review P2-5 的盲区）：用 numpy 从 device 自己的 `x_sorted` /
   `h_swiglu` 按 moe_block_ref 的 MXFP4 规范独立重量化，与 device 产出的
   `a_qx/a_scale`、`h_qx/h_scale` **逐字节**比对（scale 字节与 nibble 分开报）。
   —— host C++ 的「量化偏差预算」判据对量化器整体变粗不敏感，本项是独立见证。
2. **double 参考链**：numpy 独立实现 grouped MXFP4 GEMM / SwiGLU / 加权折叠 / sigmoid 门控，
   与 device 的 GU/H/Y/routed/shared/moe 分段比对（同样只用 device 自己的量化激活）。
3. **golden 逐 op**（mode 0）：router logits/ids/weights、perm 数组、x_sorted 对 golden bin；
   mode 1 则以 device 的 `x_norm`（已被 host 判据验证过）为段输入复算 router/perm。

`quant-inf` 子命令（M50）：对 `M13_QS_DUMP=1 <build>/m13_moe_layer quant-inf` 落盘的
±Inf/NaN/次正规用例 dump 做**逐字节**判据（T1，README §5.4 末 / §8.1）：**40 条判定项
（10 用例 × 4）+ 2 条报告项**（用例集合完整性 guard、`tools/golden` 交叉见证，按 `docs/17`
§2.1 单列、不计入 PASS/FAIL）。`--quantizer` 可换成任意实现复跑 —— 负向对照
（M32 前的参考必须 FAIL）即由此产生。

用法：
    M13_DUMP=1 ./m13_moe_layer <dataRoot> m1 m33      # 在 dump 目录里跑，产出 <case>_mode<N>_*_device.bin
    /usr/local/python3.12.13/bin/python3 m13_moe_layer/check_ref.py . m1 m33

    M13_QS_DUMP=1 <build>/m13_moe_layer quant-inf     # 在 dump 目录里跑，产出 qs_*_{x,qx_device,...}.bin
    /usr/local/python3.12.13/bin/python3 m13_moe_layer/check_ref.py quant-inf [dump 目录]
    # 负向对照（必须 FAIL）：用 M32 修复前的参考（base commit 的这份文件）跑同一判据
    git show <base>:m13_moe_layer/check_ref.py > /tmp/check_ref_prefix_nofix.py
    /usr/local/python3.12.13/bin/python3 m13_moe_layer/check_ref.py quant-inf <dump 目录> \
        --quantizer /tmp/check_ref_prefix_nofix.py

判据口径与 README §5 一致：精确 / 逐位 / bf16 网格 / w4a4-ref 容差（输出 bf16 舍入）。

退出码（三态，tower 2026-09-26「审校脚本不得在无输入可比时发合格证」）：
`0` = 比过且通过；`1` = 比过且有差异；`2` = 没得比 / 输入缺失（`quant-inf` 用例不齐、
或数据集模式任一 case×mode 缺 dump ⇒ 打印 `SKIPPED`/`PARTIAL` 并返回 2，绝不发合格证）。
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import (  # noqa: E402
    E2M1_POS,
    bf16_bits_to_f32,
    e2m1_decode,
    e8m0_decode,
    f32_to_bf16_bits,
    group_scale_exp,
    quantize_ocp,
    read_bin,
    unpack_mxfp4,
)

HIDDEN, INTER, GROUP = 2560, 640, 32
GUN = 2 * INTER
MM = 64          # M_MAX（A/H 的槽位 padding 行数）
TOPK_MAX = 4
E2M1_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def read(path, dtype):
    return np.fromfile(path, dtype=dtype)


def bf16(path, shape=None):
    a = bf16_bits_to_f32(read(path, np.uint16))
    return a.reshape(shape) if shape is not None else a


def quant_mxfp4_f32(x):
    """numpy 独立量化：group=32、scale=2^ceil(log2(amax/6))（moe_block_ref 规范）、
    e2m1 最近值编码（平局取偶）。x: float32 [rows, K] → (packed uint8 [rows,K/2], scale uint8 [rows,K/32])"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, k = x.shape
    assert k % GROUP == 0
    g = x.reshape(rows, k // GROUP, GROUP)
    exps = group_scale_exp(np.max(np.abs(g), axis=2))          # 权威 e8m0 指数规则
    scale_bytes = np.clip(exps + 127, 0, 254).astype(np.uint8)
    scale_f = np.exp2(exps.astype(np.float32))[:, :, None]
    v = np.abs(g / scale_f)
    idx = np.searchsorted(E2M1_POS, v, side="left")
    idx_hi = np.clip(idx, 0, 7)
    idx_lo = np.clip(idx - 1, 0, 7)
    d_hi = np.abs(E2M1_POS[idx_hi] - v)
    d_lo = np.abs(v - E2M1_POS[idx_lo])
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & ((idx_hi & 1) == 0))
    code = np.where(pick_hi, idx_hi, idx_lo).astype(np.uint8)
    code |= np.where(np.signbit(g), 8, 0).astype(np.uint8)
    code = code.reshape(rows, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_bytes


def quant_mxfp4_hw(x):
    """与 kernel 消费的 m5 `MxQuant` 路径同规范（OCP/hardware）的独立实现。

    **逐句对照设备代码** `MxQuantComputeScale` / `MxQuantComputeDataFP4`
    （`m13_moe_layer.asc`，M2/M5/M32 同源；官方 `add_rms_norm_dynamic_mx_quant_common.h`）：

    (1) `maxexp` = 组内 `(bf16 bits & 0x7F80)` 的**逐位 max**（指数域，不是 abs 值 max）；
    (2) 非有限组（`maxexp == 0x7F80`，组内含 ±Inf 或 NaN，含负 NaN）：scale 字节 = `0xFF`
        （E8M0 NaN，官方 :406）、halfScale = `0x7F81`（bf16 NaN，官方 :413）⇒ `Mul` 得 NaN
        ⇒ `Cast<fp4>(NaN) = 0` ⇒ 该组 data 码全 0。**（M50 修复点：M32 前的这份参考缺此分支，
        把 +Inf 量成 code 7 = |6.0|、把 scale 字节给成 0xFD，与 device 分叉。）**
    (3) 退化组（`maxexp < 0x0100`，即组 amax 为 bf16 次正规/最小正规以下：官方 :402-403 把
        `shared` 夹到 0、:414 令 halfScale = 0）⇒ 组内乘积恒 ±0 ⇒ data 码 0（负元素保留符号位
        = nibble 0x8，与 device `Cast(-0)` 一致，见 README §6.10）。**（同为 M50 修复点：旧参考
        走 `|x| / 2^-127` 除法，会给次正规组非零码。）**
    (4) 其余：字节 = 指数域 − 2；data = `cast_fp4(x · 2^(129−指数域))`，乘后先 RNE 回 bf16
        （`MxQuantComputeDataFP4` 的 bf16 域 `Mul`），再按 CAST_ROUND 取最近档
        （平局远离零、饱和到 ±6）。

    x: float32 [rows, K]（元素已是 bf16 网格值）→ (packed uint8 [rows,K/2], scale uint8 [rows,K/32])"""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, k = x.shape
    assert k % GROUP == 0
    xb = f32_to_bf16_bits(x)                       # 设备量化的是 bf16 数据（RNE；NaN 规范化为 0x7FC0）
    xg = bf16_bits_to_f32(xb).reshape(rows, k // GROUP, GROUP)
    maxexp = ((xb.reshape(rows, k // GROUP, GROUP).astype(np.uint32) & np.uint32(0x7F80))
              .max(axis=2).astype(np.int32))       # (1)
    nonfinite = maxexp == 0x7F80                   # (2) 官方 :401
    shared = np.maximum(maxexp, 0x0100) - 0x0100   # (3) 官方 :402-405（夹下界）
    byte = np.where(nonfinite, 0xFF, shared >> 7).astype(np.uint8)                  # 官方 :405-406
    half_bits = (0x7F00 - shared).astype(np.uint16)
    half_bits = np.where(shared == 0, np.uint16(0), half_bits).astype(np.uint16)    # 官方 :414
    half_bits = np.where(nonfinite, np.uint16(0x7F81), half_bits).astype(np.uint16)  # 官方 :413
    # (4) bf16 域 Mul（:753-754）：halfScale 是 2 的幂，乘积仍精确可表示；NaN 经 Cast 得 0
    with np.errstate(invalid="ignore"):
        prod = bf16_bits_to_f32(f32_to_bf16_bits(
            xg * bf16_bits_to_f32(half_bits)[:, :, None]))
    idx = np.digitize(np.abs(prod), E2M1_MIDS)     # 最近档、平局远离零、≥5.0 饱和到 code 7（=6.0）
    sign = np.where(np.signbit(prod), 8, 0).astype(np.uint8)   # device Cast(-0) = nibble 0x8
    code = (idx.astype(np.uint8) | sign)
    code = np.where(np.isnan(prod), np.uint8(0), code)         # Cast<fp4>(NaN) = 0（含 ±Inf/NaN 组）
    code = code.reshape(rows, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return packed, byte


def dequant_device(packed, scale, rows, packed_row_stride, scale_row_stride, k):
    """按 device 的行距布局反量化（packed [rows, k/2]、scale [rows, k/32]）"""
    codes = np.empty((rows, k), dtype=np.uint8)
    p = packed.reshape(-1, packed_row_stride)[:rows, : k // 2]
    codes[:, 0::2] = p & 0x0F
    codes[:, 1::2] = p >> 4
    vals = e2m1_decode(codes).reshape(rows, k // GROUP, GROUP)
    sc = e8m0_decode(scale.reshape(-1, scale_row_stride)[:rows, : k // GROUP]).astype(np.float64)
    return (vals * sc[:, :, None]).reshape(rows, k)


def rel_check(tag, got, exp, tol, results):
    """got/exp: float32/float64 数组；容差 = tol*|exp| + 半 bf16 ulp(exp)"""
    ad = np.abs(got.astype(np.float64) - exp.astype(np.float64))
    e = np.abs(exp.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ulp = np.where(e > 0, 2.0 ** (np.floor(np.log2(np.where(e > 0, e, 1.0))) - 9.0), 0.0)
    t = tol * e + ulp + 1e-9
    bad = int((ad > t).sum())
    results.append((tag, bad == 0, f"越界 {bad}/{ad.size}, maxAbs {ad.max():.3e}, 最差占容差 "
                                   f"{float((ad / t).max()):.2f}"))
    return bad == 0


# ---------------------------------------------------------------------------
# 单个 case × mode 的校验
# ---------------------------------------------------------------------------
def check_case_mode(d, data_dir, case, mode, results, refs):
    pre = f"{case}_mode{mode}"
    meta = {}
    for line in open(os.path.join(d, f"{pre}_meta.txt")):
        k, v = line.strip().split("=")
        meta[k] = int(v) if v.isdigit() else v
    m, topk, E, total = meta["m"], meta["topk"], meta["E"], meta["total"]
    def dev(name):
        # dump 命名：<case>_mode<N>_<tensor>_device.bin（M13_DUMP=1 落在同一目录）
        return os.path.join(d, f"{pre}_{name}_device.bin")

    # ---- 输入与 golden（数据集） ----
    x = bf16(os.path.join(data_dir, "x.bin"), (-1, HIDDEN))[:m]
    w_router = bf16(os.path.join(data_dir, "router_weight.bin"), (E, HIDDEN))
    counts = read(os.path.join(data_dir, "expert_token_counts.bin"), np.int32)   # golden（mode0 判据用）
    counts_dev = read(dev("expert_token_counts"), np.int32)[:E]                  # device 实际槽位分布（两种模式）
    ids_g = read(os.path.join(data_dir, "topk_ids.bin"), np.int32).reshape(m, topk)
    logits_g = read(os.path.join(data_dir, "router_logits.bin"), np.float32).reshape(m, E)
    w_g = read(os.path.join(data_dir, "topk_weights.bin"), np.float32).reshape(m, topk)
    psrc_g = read(os.path.join(data_dir, "perm_src_token.bin"), np.int32)
    pexp_g = read(os.path.join(data_dir, "perm_expert.bin"), np.int32)
    xsorted_g = bf16(os.path.join(data_dir, "x_sorted.bin"), (total, HIDDEN))

    # ---- device 的段输入：mode0 = golden x；mode1 = device 的 x_norm ----
    src = x if mode == 0 else bf16(dev("xnorm"), (MM, HIDDEN))[:m]

    # ============ 1. router / perm / x_sorted ============
    lg = src.astype(np.float64) @ w_router.astype(np.float64).T
    lg = lg - lg.max(axis=1, keepdims=True)
    dev_logits = read(dev("logits"), np.float32).reshape(MM, E)[:m]
    results.append((f"{pre} router_logits vs 复算",
                    bool(np.allclose(dev_logits, lg, atol=2e-2)),
                    f"maxAbs {np.abs(dev_logits - lg).max():.3e}"))
    dev_ids = read(dev("ids"), np.int32)[: m * topk].reshape(m, topk)
    exp_ids = np.argsort(-np.exp(lg), axis=1, kind="stable")[:, :topk]
    ok_ids = bool(np.array_equal(dev_ids, exp_ids))
    results.append((f"{pre} router ids vs 复算", ok_ids, f"mismatch {int((dev_ids != exp_ids).sum())}/{dev_ids.size}"))
    if mode == 0:
        results.append((f"{pre} topk_ids vs golden", bool(np.array_equal(dev_ids, ids_g)),
                        f"mismatch {int((dev_ids != ids_g).sum())}"))
        dev_log_g = read(dev("logits"), np.float32).reshape(MM, E)[:m]
        results.append((f"{pre} router_logits vs golden", bool(np.allclose(dev_log_g, logits_g, atol=2e-2)),
                        f"maxAbs {np.abs(dev_log_g - logits_g).max():.3e}"))
        srt = np.argsort(-np.exp(logits_g), axis=1, kind="stable")[:, :topk]
        dev_wb = read(dev("topk_weights"), np.float32)[: m * topk].reshape(m, topk)
        exp_wb = np.exp(logits_g)[np.arange(m)[:, None], srt]
        exp_wb = exp_wb / exp_wb.sum(axis=1, keepdims=True)
        ulp = np.abs(f32_to_bf16_bits(dev_wb).astype(np.int32) - f32_to_bf16_bits(exp_wb).astype(np.int32))
        results.append((f"{pre} topk_weights vs golden", bool((ulp <= 1).all()), f"max {int(ulp.max())} ulp"))
        psrc_d = read(dev("perm_src_token"), np.int32)[:total]
        pexp_d = read(dev("perm_expert"), np.int32)[:total]
        cnt_d = read(dev("expert_token_counts"), np.int32)[:E]
        results.append((f"{pre} perm/计数 vs golden",
                        bool(np.array_equal(psrc_d, psrc_g) and np.array_equal(pexp_d, pexp_g)
                             and np.array_equal(cnt_d, counts)),
                        f"src/exp/counts mismatch {int((psrc_d != psrc_g).sum())}/"
                        f"{int((pexp_d != pexp_g).sum())}/{int((cnt_d != counts).sum())}"))
        xs_d = read(dev("x_sorted"), np.uint16)[: total * HIDDEN].reshape(total, HIDDEN)
        xs_g_bits = read(os.path.join(data_dir, "x_sorted.bin"), np.uint16)[: total * HIDDEN].reshape(total, HIDDEN)
        results.append((f"{pre} x_sorted vs golden 逐位", bool(np.array_equal(xs_d, xs_g_bits)),
                        f"mismatch {int((xs_d != xs_g_bits).sum())}/{xs_d.size}"))
    # gather 逐位（两种模式都成立：源为 device 自己的段输入）
    psrc_d = read(dev("perm_src_token"), np.int32)[:total]
    xs_d = read(dev("x_sorted"), np.uint16)[: total * HIDDEN].reshape(total, HIDDEN)
    src_bits = f32_to_bf16_bits(src.astype(np.float32))   # bf16 值 → 位模式（两模式同一路径）
    results.append((f"{pre} x_sorted = gather(段输入, perm)", bool(np.array_equal(xs_d, src_bits[psrc_d])),
                    f"mismatch {int((xs_d != src_bits[psrc_d]).sum())}"))

    # ============ 2. 激活量化器逐字节（独立 numpy 量化） ============
    aqx = read(dev("a_qx"), np.uint8).reshape(E, MM, HIDDEN // 2)
    asc = read(dev("a_scale"), np.uint8).reshape(E, MM, HIDDEN // GROUP)
    xs_f = bf16_bits_to_f32(xs_d).astype(np.float32)
    off = np.concatenate([[0], np.cumsum(counts_dev)])
    qx_bad = sc_bad = qx_n = sc_n = 0
    # 参考项统计：device 字节 vs「golden 权重打包规范」量化器的逐字节差异
    # （kernel 激活侧用的是硬件规范 = quant_mxfp4_hw，见 README §5.3；此项只报告规范差异占比）
    ref_stat = {}

    def ref_acc(tag, dev_bytes, golden_rule_bytes):
        bad, tot = ref_stat.get(tag, (0, 0))
        ref_stat[tag] = (bad + int((dev_bytes != golden_rule_bytes).sum()), tot + dev_bytes.size)
    for e in range(E):
        t = int(counts_dev[e])
        if t == 0:
            continue
        rows = xs_f[off[e]:off[e] + t]
        pq, sq = quant_mxfp4_hw(rows)
        qx_bad += int((pq != aqx[e, :t]).sum())
        sc_bad += int((sq != asc[e, :t]).sum())
        qx_n += pq.size
        sc_n += sq.size
        # 参考项：golden 权重打包规范（ceil(log2(amax/6))）与本 kernel 采用的硬件规范差异
        pq_g, sq_g = quant_mxfp4_f32(rows)
        ref_acc("A_qx(routed)", aqx[e, :t], pq_g)
        ref_acc("A_scale(routed)", asc[e, :t], sq_g)
    results.append((f"{pre} A_qx 逐字节 vs numpy(硬件规范)", qx_bad == 0, f"{qx_bad}/{qx_n} 字节不符"))
    results.append((f"{pre} A_scale 逐字节 vs numpy(硬件规范)", sc_bad == 0, f"{sc_bad}/{sc_n} 字节不符"))

    # 共享专家 A（紧凑布局）
    aqx_s = read(dev("a_qx_shd"), np.uint8).reshape(MM, HIDDEN // 2)
    asc_s = read(dev("a_scale_shd"), np.uint8).reshape(MM, HIDDEN // GROUP)
    pq, sq = quant_mxfp4_hw(bf16_bits_to_f32(src_bits).astype(np.float32))   # 共享专家 A（紧凑行）
    results.append((f"{pre} A_shd_qx 逐字节 vs numpy(硬件规范)", bool(np.array_equal(pq, aqx_s[:m])),
                    f"{int((pq != aqx_s[:m]).sum())}/{pq.size}"))
    results.append((f"{pre} A_shd_scale 逐字节 vs numpy(硬件规范)", bool(np.array_equal(sq, asc_s[:m])),
                    f"{int((sq != asc_s[:m]).sum())}/{sq.size}"))
    pq_g, sq_g = quant_mxfp4_f32(bf16_bits_to_f32(src_bits).astype(np.float32))
    ref_acc("A_shd_qx(共享)", aqx_s[:m], pq_g)
    ref_acc("A_shd_scale(共享)", asc_s[:m], sq_g)

    # ============ 3. double 参考链（只用 device 的量化激活） ============
    wgu = read(os.path.join(data_dir, "experts.gate_up_proj.bin"), np.uint8).reshape(E, GUN, HIDDEN // 2)
    sgu = read(os.path.join(data_dir, "experts.gate_up_proj.weight_scale.bin"), np.uint8).reshape(E, GUN, HIDDEN // GROUP)
    wdn = read(os.path.join(data_dir, "experts.down_proj.bin"), np.uint8).reshape(E, HIDDEN, INTER // 2)
    sdn = read(os.path.join(data_dir, "experts.down_proj.weight_scale.bin"), np.uint8).reshape(E, HIDDEN, INTER // GROUP)
    gu_dev = bf16_bits_to_f32(read(dev("gu"), np.uint16)[: E * MM * GUN]).reshape(E, MM, GUN)
    h_dev = bf16_bits_to_f32(read(dev("h_swiglu"), np.uint16)[: total * INTER]).reshape(total, INTER)
    hqx = read(dev("h_qx"), np.uint8).reshape(E, MM, INTER // 2)
    hs = read(dev("h_scale"), np.uint8).reshape(E, MM, 32)
    y_dev = bf16_bits_to_f32(read(dev("y_sorted"), np.uint16)[: E * MM * HIDDEN]).reshape(E, MM, HIDDEN)
    routed_dev = bf16_bits_to_f32(read(dev("routed_output"), np.uint16)[: m * HIDDEN]).reshape(m, HIDDEN)
    shared_dev = bf16_bits_to_f32(read(dev("shared_output"), np.uint16)[: m * HIDDEN]).reshape(m, HIDDEN)
    moe_dev = bf16_bits_to_f32(read(dev("moe_output"), np.uint16)[: m * HIDDEN]).reshape(m, HIDDEN)

    hqx_bad = hs_bad = hqz_n = hsz_n = 0
    y_ref = np.zeros((E, MM, HIDDEN), dtype=np.float64)
    for e in range(E):
        t = int(counts_dev[e])
        if t == 0:
            continue
        adq = dequant_device(aqx[e].ravel(), asc[e].ravel(), t, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
        w = dequant_device(wgu[e].ravel(), sgu[e].ravel(), GUN, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
        gu_ref = adq @ w.T
        rel_check(f"{pre} GU slot{e}", gu_dev[e, :t], gu_ref, 1e-2, results)
        # silu 的输入取 device 自己的 bf16 GU（逐 op 口径：每段残差只含本段）
        gd = gu_dev[e, :t, :INTER].astype(np.float64)
        ud = gu_dev[e, :t, INTER:].astype(np.float64)
        h_ref = (gd / (1.0 + np.exp(-gd))) * ud
        rel_check(f"{pre} H slot{e}", h_dev[off[e]:off[e] + t], h_ref, 1e-2, results)
        # 量化器逐字节（device 的 h_swiglu → 独立 numpy 量化 → 与 device 的 h_qx/h_scale 比）
        h_rows = np.ascontiguousarray(h_dev[off[e]:off[e] + t].astype(np.float32))
        pq, sq = quant_mxfp4_hw(h_rows)
        hqx_bad += int((pq != hqx[e, :t]).sum())
        hs_bad += int((sq != hs[e, :t, : INTER // GROUP]).sum())
        hqz_n += pq.size
        hsz_n += sq.size
        pq_g, sq_g = quant_mxfp4_f32(h_rows)
        ref_acc("H_qx(routed)", hqx[e, :t], pq_g)
        ref_acc("H_scale(routed)", hs[e, :t, : INTER // GROUP], sq_g)
        hd = dequant_device(hqx[e].ravel(), hs[e].ravel(), t, INTER // 2, 32, INTER)
        wd = dequant_device(wdn[e].ravel(), sdn[e].ravel(), HIDDEN, INTER // 2, INTER // GROUP, INTER)
        y_ref[e, :t] = hd @ wd.T
        rel_check(f"{pre} Y slot{e}", y_dev[e, :t], y_ref[e, :t], 1e-2, results)
    results.append((f"{pre} H_qx 逐字节 vs numpy(硬件规范)", hqx_bad == 0, f"{hqx_bad}/{hqz_n} 字节不符"))
    results.append((f"{pre} H_scale 逐字节 vs numpy(硬件规范)", hs_bad == 0, f"{hs_bad}/{hsz_n} 字节不符"))

    # routed 折叠（device 的 bf16 Y + device 的 w_tk_packed）
    inv = read(dev("inv_slot"), np.int32)[: m * topk].reshape(m, topk)
    wtk = read(dev("w_tk_packed"), np.int32).reshape(-1, 16)[:m, :topk]
    y_flat = y_dev.reshape(E * MM, HIDDEN)
    routed_ref = np.zeros((m, HIDDEN), dtype=np.float64)
    for t in range(m):
        for k in range(topk):
            wv = bf16_bits_to_f32(np.array([wtk[t, k] & 0xFFFF], dtype=np.uint16))[0]
            routed_ref[t] += wv * y_flat[inv[t, k]]
    rel_check(f"{pre} routed_output", routed_dev, routed_ref, 1e-2, results)

    # 共享专家（device 的激活量化 + device 的 Y_shd）
    wgs = read(os.path.join(data_dir, "shared_expert.gate_proj.bin"), np.uint8).reshape(INTER, HIDDEN // 2)
    wus = read(os.path.join(data_dir, "shared_expert.up_proj.bin"), np.uint8).reshape(INTER, HIDDEN // 2)
    wgn = read(os.path.join(data_dir, "shared_expert.gate_proj.weight_scale.bin"), np.uint8).reshape(INTER, HIDDEN // GROUP)
    wun = read(os.path.join(data_dir, "shared_expert.up_proj.weight_scale.bin"), np.uint8).reshape(INTER, HIDDEN // GROUP)
    wds = read(os.path.join(data_dir, "shared_expert.down_proj.bin"), np.uint8).reshape(HIDDEN, INTER // 2)
    wdn_s = read(os.path.join(data_dir, "shared_expert.down_proj.weight_scale.bin"), np.uint8).reshape(HIDDEN, INTER // GROUP)
    adq = dequant_device(aqx_s.ravel(), asc_s.ravel(), m, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
    wg = dequant_device(wgs.ravel(), wgn.ravel(), INTER, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
    wu = dequant_device(wus.ravel(), wun.ravel(), INTER, HIDDEN // 2, HIDDEN // GROUP, HIDDEN)
    gu_s_ref = adq @ np.concatenate([wg, wu], axis=0).T
    gu_shd_dev = bf16_bits_to_f32(read(dev("gu_shd"), np.uint16)[: MM * GUN]).reshape(MM, GUN)[:m]
    rel_check(f"{pre} GU_shd", gu_shd_dev, gu_s_ref, 1e-2, results)
    # silu 输入取 device 自己的 bf16 GU_shd（逐 op 口径，与 host C++ 判据一致）
    g = gu_shd_dev[:, :INTER].astype(np.float64)
    u = gu_shd_dev[:, INTER:].astype(np.float64)
    h_s_ref = (g / (1.0 + np.exp(-g))) * u
    h_shd_dev = bf16_bits_to_f32(read(dev("h_swiglu_shd"), np.uint16)[: MM * INTER]).reshape(MM, INTER)[:m]
    rel_check(f"{pre} H_shd", h_shd_dev, h_s_ref, 1e-2, results)
    pq, sq = quant_mxfp4_hw(np.ascontiguousarray(h_shd_dev.astype(np.float32)))
    hqx_s = read(dev("h_qx_shd"), np.uint8).reshape(MM, INTER // 2)
    hs_s = read(dev("h_scale_shd"), np.uint8).reshape(MM, 32)
    results.append((f"{pre} H_shd_qx 逐字节 vs numpy(硬件规范)", bool(np.array_equal(pq, hqx_s[:m])),
                    f"{int((pq != hqx_s[:m]).sum())}/{pq.size}"))
    results.append((f"{pre} H_shd_scale 逐字节 vs numpy(硬件规范)", bool(np.array_equal(sq, hs_s[:m, : INTER // GROUP])),
                    f"{int((sq != hs_s[:m, : INTER // GROUP]).sum())}/{sq.size}"))
    pq_g, sq_g = quant_mxfp4_f32(np.ascontiguousarray(h_shd_dev.astype(np.float32)))
    ref_acc("H_shd_qx(共享)", hqx_s[:m], pq_g)
    ref_acc("H_shd_scale(共享)", hs_s[:m, : INTER // GROUP], sq_g)
    hd = dequant_device(hqx_s.ravel(), hs_s.ravel(), m, INTER // 2, 32, INTER)
    wd = dequant_device(wds.ravel(), wdn_s.ravel(), HIDDEN, INTER // 2, INTER // GROUP, INTER)
    y_shd_ref = hd @ wd.T
    y_shd_dev = bf16_bits_to_f32(read(dev("y_shd"), np.uint16)[: MM * HIDDEN]).reshape(MM, HIDDEN)[:m]
    rel_check(f"{pre} Y_shd", y_shd_dev, y_shd_ref, 1e-2, results)
    # sigmoid 门（用 device 的 sgate 裸点积 + device 的 bf16 Y_shd）
    sg = read(dev("sgate"), np.float32)[:m]
    gv = (1.0 / (1.0 + np.exp(-sg.astype(np.float64))))
    shared_ref = gv[:, None] * y_shd_dev.astype(np.float64)
    rel_check(f"{pre} shared_output", shared_dev, shared_ref, 1e-2, results)
    moe_ref = routed_dev.astype(np.float64) + shared_ref
    rel_check(f"{pre} moe_output", moe_dev, moe_ref, 1e-2, results)

    # 参考项（非判定）：kernel 激活量化规范（硬件 floor 指数）与 golden 权重打包规范
    # （ceil(log2(amax/6))）的逐字节差异占比 —— 始终打印，便于从归档日志复核 README §5.3
    for tag in ("A_qx(routed)", "A_scale(routed)", "A_shd_qx(共享)", "A_shd_scale(共享)", "H_qx(routed)",
                "H_scale(routed)", "H_shd_qx(共享)", "H_shd_scale(共享)"):
        if tag in ref_stat:
            bad, tot = ref_stat[tag]
            refs.append((f"{pre} {tag}", f"{bad}/{tot} 字节不同 = {100.0 * bad / tot:.1f}%"))
    return results


def golden_selfcheck(data_dir, results):
    """数据集自洽：由 bin 重算 golden 并与 golden bin 逐位/容差比对（权威参考可信性）"""
    m = len(read(os.path.join(data_dir, "x.bin"), np.uint16)) // HIDDEN
    cnt = read(os.path.join(data_dir, "expert_token_counts.bin"), np.int32)
    ids = read(os.path.join(data_dir, "topk_ids.bin"), np.int32)
    x = bf16(os.path.join(data_dir, "x.bin"), (-1, HIDDEN))
    w = bf16(os.path.join(data_dir, "router_weight.bin"), (-1, HIDDEN))
    logits = read(os.path.join(data_dir, "router_logits.bin"), np.float32).reshape(x.shape[0], w.shape[0])
    lg = x.astype(np.float64) @ w.astype(np.float64).T
    lg = (lg - lg.max(axis=1, keepdims=True)).astype(np.float32)
    results.append((f"{os.path.basename(data_dir)} golden 自洽 (logits)", bool(np.allclose(lg, logits, atol=2e-2)),
                    f"maxAbs {np.abs(lg - logits).max():.3e}"))
    # 权重 pack/unpack 定点性
    p = read(os.path.join(data_dir, "experts.down_proj.bin"), np.uint8).reshape(-1, INTER // 2)
    s = read(os.path.join(data_dir, "experts.down_proj.weight_scale.bin"), np.uint8).reshape(-1, INTER // GROUP)
    p, s = p[:64], s[:64]
    v = unpack_mxfp4(p, s)
    p2, s2 = quant_mxfp4_f32(v.astype(np.float32))
    v2 = unpack_mxfp4(p2, s2)
    results.append((f"{os.path.basename(data_dir)} MXFP4 反量化定点性", bool(np.allclose(v, v2, atol=0, rtol=0)),
                    f"maxAbs {np.abs(v - v2).max():.3e}"))


def load_quantizer(spec):
    """量化器选择：默认本文件的 `quant_mxfp4_hw`（M50 修复后 = device 语义）；
    传 `"PATH[:FUNC]"` 则从该文件加载（负向对照用 base commit 的实现）。

    返回 (函数, 人类可读名)。"""
    if spec is None:
        return quant_mxfp4_hw, "check_ref.py:quant_mxfp4_hw"
    path, _, fn = spec.partition(":")
    fn = fn or "quant_mxfp4_hw"
    mspec = importlib.util.spec_from_file_location("m13_ext_quantizer", os.path.abspath(path))
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    return getattr(mod, fn), f"{path}:{fn}"


# ---------------------------------------------------------------------------
# quant-inf 子命令（M50）：量化段非有限/退化用例的 device 逐字节判据（T1）
# ---------------------------------------------------------------------------
QS_KINDS = {0: "基线（无注入）", 1: "整组 +Inf", 2: "组内部分 ±Inf", 3: "NaN 组（+NaN）",
            4: "次正规/退化组（指数域 < 0x0100）"}
QS_SIDES = {"A": (HIDDEN, 80), "H": (INTER, 32)}     # side -> (K, scale 行距 SS)
QS_DUMP_ROWS, QS_DUMP_SEED = 1, 0                    # 归档 dump 的用例档（r1/s0）


def quant_inf_main(ddir, quant_spec):
    """对 `M13_QS_DUMP=1 ... quant-inf` 的 dump 逐字节判据。

    A 侧量化输入 = `qs_A_i{k}_r1_s0_x.bin`（K=2560）；H 侧 = `qs_H_i{k}_r1_s0_swiglu_device.bin`
    （K=640，与 .asc 自检同口径：参考作用在**设备自己的**量化输入上）。每个用例 4 条判定项：

    1. `qx` 与 device **逐字节**一致（T1）；
    2. scale 行内合法区（前 K/32 字节）**逐字节**一致（T1）；
    3. scale 行内 padding 区（K/32 之后到 SS 行距）**未被写**（T1）；
    4. **结构性（只用 device 字节与输入位型，不经参考）**：非有限组（scale 字节 0xFF）/ 退化组
       （输入指数域 < 0x0100 且非全零）确实存在，且这些组的 device data 码全 0 —— 即 M32 修复
       后的「非有限 ⇒ 组内 nibble 归零」，修复前是 ±6 饱和码 0x7/0xF。

    另有两项**报告项**（`docs/17` §2.1 判定项/报告项分栏，**不计入** PASS/FAIL 计数）：
    ① 用例集合完整性（非空洞性 guard：只查 5 kinds × 2 侧用例到齐，不检验被判量化器）；
    ② `tools/golden` 交叉见证（`moe_block_ref.quantize_ocp` vs device 逐字节，任务 ③ 的独立实现）。

    合计 **40 条判定项 + 2 条报告项**（10 用例 × 4）。"""
    quant_fn, quant_name = load_quantizer(quant_spec)
    results = []
    print(f"[check_ref] quant-inf：dump 目录 {os.path.abspath(ddir)}")
    print(f"[check_ref] quant-inf：被判量化器 {quant_name}")
    ncase = 0
    ocp_bad = ocp_n = 0
    for side, (k, ss) in QS_SIDES.items():
        for kind in sorted(QS_KINDS):
            pre = f"qs_{side}_i{kind}_r{QS_DUMP_ROWS}_s{QS_DUMP_SEED}"
            p = {n: os.path.join(ddir, f"{pre}_{n}.bin") for n in
                 ("x", "swiglu_device", "qx_device", "scale_device")}
            if not os.path.exists(p["qx_device"]):
                continue
            ncase += 1
            ng = k // GROUP
            src = read(p["swiglu_device"] if side == "H" else p["x"], np.uint16)
            qx = read(p["qx_device"], np.uint8)
            sc = read(p["scale_device"], np.uint8)
            srcf = bf16_bits_to_f32(src).astype(np.float32).reshape(1, k)
            srcg = src.reshape(-1, GROUP)
            fields = (srcg.astype(np.uint32) & np.uint32(0x7F80)).max(axis=1)      # 输入组指数域
            nf_g = int((fields == 0x7F80).sum())                                   # 非有限组数
            dg_g = int(((fields < 0x0100) & (srcg != 0).any(axis=1)).sum())        # 退化组数（非全零）
            pq, sq = quant_fn(srcf)
            dq = int((pq.ravel() != qx).sum())
            ds = int((sq.ravel()[:ng] != sc[:ng]).sum())
            tag = f"quant-inf {side} i{kind}"
            results.append((f"{tag} qx 逐字节", dq == 0, f"{dq}/{qx.size} 字节不符（{QS_KINDS[kind]}）"))
            results.append((f"{tag} scale 合法区逐字节", ds == 0, f"{ds}/{ng} 字节不符"))
            pad = sc[ng:]
            results.append((f"{tag} scale 行内 padding 未被写", bool((pad == 0).all()),
                            f"SS={ss} K/32={ng} padding {pad.size} 字节，非零 {int((pad != 0).sum())}"))
            # 4) 结构性：只用 device 字节 + 输入位型（不经被判参考）
            nf_dev = int((sc[:ng] == 0xFF).sum())
            nf_dev_bad = 0
            for g in np.where(sc[:ng] == 0xFF)[0]:
                nf_dev_bad += int((qx[g * GROUP // 2:(g + 1) * GROUP // 2] != 0).sum())
            dg_dev_bad = 0
            dg_idx = np.where((fields < 0x0100) & (srcg != 0).any(axis=1))[0]
            for g in dg_idx:
                dg_dev_bad += int((qx[g * GROUP // 2:(g + 1) * GROUP // 2] != 0).sum())
            if kind == 0:
                ok4, info4 = (nf_g == 0 and nf_dev == 0), f"基线无注入：输入非有限组 {nf_g}、device 0xFF 组 {nf_dev}（应全 0）"
            elif kind == 4:
                ok4 = (dg_g > 0 and dg_dev_bad == 0)
                info4 = (f"退化组 输入 {dg_g} 个、device 这 {dg_idx.size} 组 nibble 非零 {dg_dev_bad}；"
                         f"（输入非有限组 {nf_g}）")
            else:
                ok4 = (nf_g > 0 and nf_dev > 0 and nf_dev_bad == 0)
                info4 = (f"非有限组 输入 {nf_g} 个、device scale=0xFF 组 {nf_dev} 个、"
                         f"其 data 码非零 {nf_dev_bad}（修复前为 ±6 饱和码）")
            results.append((f"{tag} 结构性（非有限/退化组 nibble 归零）", ok4, info4))
            #（末项）tools/golden 交叉见证：任务 ③ 的「黄金参考是否同型缺陷」（只报不改）
            oq, os_ = quantize_ocp(srcf)
            ocp_bad += int((oq.ravel() != qx).sum()) + int((os_.ravel()[:ng] != sc[:ng]).sum())
            ocp_n += qx.size + ng
    # 报告项（非判定项，docs/17 §2.1「判定项/报告项分栏」）：① 用例集合完整性是**非空洞性 guard**
    # （只查用例到齐，不检验被判量化器）；② `tools/golden` 交叉见证是**另一个实现**的见证，
    # 不参与本判据的 PASS/FAIL 计数。两者单列打印，不计入判定项。
    reports = [
        (f"quant-inf 用例集合完整（{len(QS_SIDES)} 侧 × {len(QS_KINDS)} kinds，r{QS_DUMP_ROWS}/s{QS_DUMP_SEED}）",
         ncase == len(QS_SIDES) * len(QS_KINDS), f"实到 {ncase}"),
        ("tools/golden 交叉见证（moe_block_ref.quantize_ocp vs device 逐字节）",
         ocp_bad == 0 and ocp_n > 0, f"{ocp_bad}/{ocp_n} 字节不符，{ncase} 个用例"),
    ]
    print()
    nbad = 0
    for t, ok, info in results:
        if not ok:
            nbad += 1
            print(f"[check_ref] FAIL {t}: {info}")
    want = len(QS_SIDES) * len(QS_KINDS)
    print(f"[check_ref] quant-inf 判定项：{len(results)} 条，{len(results) - nbad} PASS / {nbad} FAIL")
    for t, ok, info in reports:
        print(f"[check_ref] quant-inf 报告项：{'OK  ' if ok else 'WARN'} {t}: {info}")
    # 三态结论（tower 2026-09-26 规则：审校脚本不得在「没有可比较输入」时发合格证）。
    # 退出码：0 = 比过且通过；1 = 比过且有差异；2 = 没得比 / 输入缺失。
    if ncase == 0:
        print(f"[check_ref] quant-inf ===== SKIPPED（{os.path.abspath(ddir)} 下没有 "
              f"qs_*_r{QS_DUMP_ROWS}_s{QS_DUMP_SEED}_*_device.bin：未比较任何用例，退出码 2）=====")
        return 2
    if nbad != 0:
        print("[check_ref] quant-inf ===== FAILURES PRESENT（退出码 1）=====")
        return 1
    if ncase < want:
        print(f"[check_ref] quant-inf ===== PARTIAL（用例 {ncase}/{want}，输入缺失；已比较部分全 PASS，"
              f"但不得据此发合格证，退出码 2）=====")
        return 2
    print("[check_ref] quant-inf ===== ALL PASS =====")
    return 0


def main():
    args = sys.argv[1:]
    if args and args[0] == "quant-inf":
        rest = args[1:]
        spec = None
        if "--quantizer" in rest:
            i = rest.index("--quantizer")
            spec = rest[i + 1]
            del rest[i:i + 2]
        d = rest[0] if rest else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "evidence", "quant_inf_dumps")
        return quant_inf_main(d, spec)
    d = args[0] if args else "."
    cases = args[1:] or ["m1", "m33"]
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden", "data")
    results = []
    refs = []
    ncomp = nskipped = 0
    for case in cases:
        data_dir = os.path.abspath(os.path.join(data_root, case))
        print(f"===== {case} =====")
        golden_selfcheck(data_dir, results)
        for mode in (0, 1):
            if not os.path.exists(os.path.join(d, f"{case}_mode{mode}_meta.txt")):
                nskipped += 1
                print(f"  [skip] {case} mode{mode}: 无 dump（先跑 M13_DUMP=1）")
                continue
            ncomp += 1
            check_case_mode(d, data_dir, case, mode, results, refs)
    print()
    nbad = 0
    for tag, ok, info in results:
        if not ok:
            nbad += 1
            print(f"[check_ref] FAIL {tag}: {info}")
    print(f"[check_ref] 判定项：{len(results)} 条，{len(results) - nbad} PASS / {nbad} FAIL")
    print(f"[check_ref] 参考项（非判定）：{len(refs)} 条 —— kernel 激活量化规范（硬件 floor 指数 e8m0）"
          f"vs golden 权重打包规范（ceil(log2(amax/6))）的逐字节差异占比")
    for tag, info in refs:
        print(f"[check_ref][参考] {tag}: {info}")
    # 三态结论（tower 2026-09-26 规则：审校脚本不得在「没有可比较输入」时发合格证）
    #   0 = 比过且通过；1 = 比过且有差异；2 = 没得比 / 输入缺失
    if ncomp == 0:
        print("[check_ref] ===== SKIPPED（没有可比较的 case×mode dump：未比较任何 device 张量；"
              "退出码 2）=====")
        return 2
    if nbad != 0:
        print(f"[check_ref] ===== FAILURES PRESENT （判定 {len(results)} 项 / 参考 {len(refs)} 项，"
              f"退出码 1）=====")
        return 1
    if nskipped:
        print(f"[check_ref] 输入缺失：只比较了 {ncomp}/{len(cases) * 2} 个 case×mode（跳过 {nskipped} 个），"
              f"已比较部分全 PASS，但**不得据此发合格证**")
        print("[check_ref] ===== PARTIAL（退出码 2）=====")
        return 2
    print(f"[check_ref] ===== ALL PASS （判定 {len(results)} 项 / 参考 {len(refs)} 项）=====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
