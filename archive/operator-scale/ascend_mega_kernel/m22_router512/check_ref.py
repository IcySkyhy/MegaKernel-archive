# SPDX-License-Identifier: Apache-2.0
"""m22_router512 的独立交叉校验（docs/17 L0 口径：判定项 / 报告项 / guard 分栏）。

对拍对象：`tools/golden/moe_block_ref.py`（M26 量化感知改版后的权威 CPU golden）
  * `router_topk(x, w, 10)`  —— softmax(512) → 降序 top-10（并列取小 id）→ renorm
  * `moe_permute(ids, 512)`  —— 每专家计数 / CSR offsets / (expert, token) 稳定序 perm

**参考口径的必要修正（本轮实证，属参考建模而非 kernel 问题）**：设备 softmax 分数落在
fp32 **次正规区**时被硬件 **FTZ flush 成 0**（docs/05 §6.1 明确「次正规 FTZ 是硬件模式，
不在规避范围」），而 golden 的 `np.exp` 保留 1e-38..1e-45 的次正规值并据此排序 ⇒ 深尾
logits（|l| ≳ 88）下两者 top-k 的**并列次序**不同。本脚本因此取 golden 的 logits + 设备
FTZ 口径复算 top-k（并列仍取小 id，与 golden 同规则），并把「golden 与本参考的 ids 差」列为报告项 R0。

**R0 的语义在 M46（`e330796`）之后变了（M56 记）**：
  * M46 之前：`moe_block_ref.router_topk` **不建模 FTZ** ⇒ R0 量的是**设备 FTZ 的真实影响**
    —— m=4097、真实 checkpoint 权重下 **125 槽 / 20 行**，其余 9 档 0/0。
  * M46 之后：golden 自己按设备口径建模了 FTZ（`ftz_f32` + 3 个点位）⇒ R0 变成**两套 FTZ
    实现的一致性**，**全档 0 槽 / 0 行**（含 m=4097）。**0/0 是预期结果，不是信号丢失**：
    它现在只在两侧 FTZ 建模不一致时才会非零，仍是一条回归护栏。
  * FTZ 影响的**原始证据**冻结在 `evidence/ftz_modeling_scan.txt`（扫描表首行 125 槽 / 20 行），
    不随本脚本重跑改变。R0 是**报告项**，不参与 PASS/FAIL。

用法：
    # real_* 用例（真实 checkpoint mlp.gate.weight 切片）
    python3.12 check_ref.py <dumpdir> --w m22_router512/data/router_weight.bin
    # nu_* 用例（受控 one-hot 合成权重，脚本自行重建，无需 dump 权重）
    python3.12 check_ref.py <dumpdir> --w onehot
    # 大 m 用例没有 x.bin（21MB，不入库）→ 用 --x-seed 重生成并核对 sha256 记录
    python3.12 check_ref.py <dumpdir> --w ... --x-seed 5 --x-sha256 <hex>

判定项（任一 FAIL 则脚本退出码 1）：
  J1  topk_ids                与 golden 逐元素相等（int32）—— T1（整数/索引域）
  J2  topk_weights            与 golden fp32→bf16(RNE) 的 |ulp| ≤ 1 —— T3 推导界（≡ ≤1 ulp）
  J3  router_logits           逐元素 ≤ T3 界 `ε·Σ|terms| + 0.5·ulp(out)`，ε = (40+6+2560)·2⁻²⁴
                              （docs/17 §1.1 触发②：K=2560 被切成 40 段跨 split 累加 + 64 lane
                              树归约 + 本参考 2560 次顺序相加；Σ|terms| 用 Rmax[t]·Wsum[e] 上界）
  J4  expert_counts[512]      == golden expert_token_counts（count 模式 group_list 载荷）—— T1
  J5  expert_slot_base[512]   == golden expert_offsets[:512]（紧凑槽起点）—— T1
  J6  perm_src_token[Σt_e]    == golden perm_src_token —— T1
  J7  perm_expert[Σt_e]       == golden perm_expert —— T1
  J8  group_list_i64[512]     == int64(expert_counts)（官方 count 模式编码投影）—— T1
  J9  route_scalars[0]/[3]    == m*topk（active_num / Σt_e）—— T1
  J10 Σ expert_counts         == m*topk —— T1
硬闸门（先于上述判定项，不满足即 FAIL + 退出码 1，不继续）：
  X1  走 --x-seed 重建路径时 **必须**给 --x-sha256，且重建 x 的 sha256 必须与之相等
      （tower 条件 ①：归档必须足以确定性重建 x，校验必须是强制的）
  X2  同时给 --x-seed 与 x.bin 时，重建的 x 必须与 dump 的 x 逐字节一致
报告项（不参与 PASS/FAIL）：
  R0 golden 与本参考的 ids 差 —— **M46 后 = 两套 FTZ 实现的一致性，全档 0/0**；
     M46 前 = 未建模 FTZ 的差异（m=4097 实测 125 槽 / 20 行）。语义改变详见文件头。
  R1 logits max abs / max rel 误差（对 numpy golden）   R2 实到的非均匀分布画像
  R3 每行 topk 权重的行和（应 = 1）
guard（非空洞性，恒真检查，单列）：
  G1 logits 每行 max == 0（max-shift 语义）   G2 perm 与 counts 自洽 + 值域合法
  G3 route_scalars[2] == 0（无脏 id）

退出码（三态，"比过了" 与 "没得比" 必须能分辨 —— tower 规则 2026-09-26）：
  0  比过且通过（打印 `RESULT: OK (<n> 判定项 ...)`）
  1  比过且有差异（判定项 FAIL，或硬闸门 X1/X2 不过）
  2  没得比/输入缺失（dump 文件缺失、判定项列表为空 —— 打印 `RESULT: SKIPPED`，
     绝不发"合格证"）
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

# 优先用**本仓相对**的 tools/golden（worktree / 主 checkout 都适用）；只在不存在的旧布局下
# 才回落到绝对路径。此前硬编码绝对路径 ⇒ 在别的 checkout 上会 ImportError。
_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, os.pardir, "tools", "golden")
if not os.path.isdir(_GOLDEN):
    _GOLDEN = "/workspace/ascend_mega_kernel/tools/golden"
sys.path.insert(0, _GOLDEN)
from moe_block_ref import f32_to_bf16_bits, router_topk, moe_permute  # noqa: E402

HIDDEN = 2560
E = 512
KTOPK = 10
# fp32 最小正规数（2^-126）：设备 softmax 分数低于它即被硬件 FTZ flush 成 0（docs/05 §6.1）
FTZ_MIN_NORMAL = np.float32(1.1754944e-38)


def gen_activations_uniform(m: int, seed: int) -> np.ndarray:
    """C++ GenActivationsUniform 的 numpy 逐位复刻（Hash3 全整数运算，uint64 中间量保精度）。"""
    i = np.arange(m, dtype=np.uint64)[:, None]
    j = np.arange(HIDDEN, dtype=np.uint64)[None, :]
    h = (i * np.uint64(2654435761) + j * np.uint64(40503) + np.uint64(seed * 97)
         + np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(16))) & np.uint64(0xFFFFFFFF)
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) & np.uint64(0xFFFFFFFF)
    exp = (np.uint64(122) + (h % np.uint64(10))) & np.uint64(0xFFFF)
    v = (((h >> np.uint64(8)) & np.uint64(0x8000)) | (exp << np.uint64(7))
         | ((h >> np.uint64(9)) & np.uint64(0x7F)))
    return v.astype(np.uint16)


def onehot_weight() -> np.ndarray:
    """受控档的合成 gate 权重：W[e][k] = 1 当 k == e，否则 0（bf16 精确）。"""
    w = np.zeros((E, HIDDEN), dtype=np.uint16)
    w[np.arange(E), np.arange(E)] = f32_to_bf16_bits(np.float32(1.0))
    return w


def bf16_view(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def load_bf16(path, shape):
    return np.fromfile(path, dtype=np.uint16).reshape(shape)


def load_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def load_i32(path, shape):
    return np.fromfile(path, dtype=np.int32).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumpdir")
    ap.add_argument("--w", required=True, help="router_weight.bin 路径，或 'onehot'")
    ap.add_argument("--x-seed", type=int, default=None,
                    help="无 x.bin 时用该 seed 重新生成 uniform 输入")
    ap.add_argument("--x-sha256", default=None, help="与重生成的 x 比对的 sha256（evidence 记录）")
    ap.add_argument("--no-logits", action="store_true",
                    help="该档的 router_logits dump 未入库（m=4097 为 8.4MB）→ J3 转为报告项；"
                         "机内 J3 已由 m22_router512 自检 + run 日志完成，sha256 见 evidence/dump_sha256.txt")
    args = ap.parse_args()

    d = args.dumpdir
    try:
        return _run(args)
    except OSError as exc:
        print(f"[check_ref] RESULT: SKIPPED (输入缺失 — {exc}；这不是通过)")
        return 2


def _run(args) -> int:
    d = args.dumpdir
    ids = load_i32(f"{d}/topk_ids.bin", (-1, KTOPK))
    m = ids.shape[0]
    wbits_dev = load_bf16(f"{d}/topk_weights.bin", (m, KTOPK))
    log_dev = None if args.no_logits else load_f32(f"{d}/router_logits.bin", (m, E))
    cnt_dev = load_i32(f"{d}/expert_counts.bin", (E,))
    base_dev = load_i32(f"{d}/expert_slot_base.bin", (E,))
    ps_dev = load_i32(f"{d}/perm_src_token.bin", (-1,))
    pe_dev = load_i32(f"{d}/perm_expert.bin", (-1,))
    g64_dev = np.fromfile(f"{d}/group_list_i64.bin", dtype=np.int64)
    scl_dev = load_i32(f"{d}/route_scalars.bin", (4,))

    judged: list[tuple[str, bool, str]] = []
    report: list[str] = []
    guards: list[tuple[str, bool, str]] = []

    # ---- 输入 ----
    # tower 条件 ①：归档必须足以**确定性重建** x，且校验是**强制**的 —— 走 --x-seed 重建路径时
    # `--x-sha256` 缺失或不符一律 **FAIL（退出码 1）**，不是警告；这是 m=4097 唯一的离线判定链，
    # 下游（M40/M7/M13）会照抄这条命令。
    x_path = f"{d}/x.bin"
    try:
        xbits = load_bf16(x_path, (m, HIDDEN))
        x = bf16_view(xbits).reshape(m, HIDDEN)
        if args.x_seed is not None:
            gx = gen_activations_uniform(m, args.x_seed)
            if not bool((gx == xbits).all()):
                print(f"[check_ref] FAIL: --x-seed {args.x_seed} 重生成的 x 与 {x_path} 不逐字节一致"
                      f"（x 生成器的 python 复刻不可信）")
                return 1
            if args.x_sha256 is not None:
                gd = hashlib.sha256(gx.tobytes()).hexdigest()
                if gd != args.x_sha256:
                    print(f"[check_ref] FAIL: 重生成 x 的 sha256 {gd} != --x-sha256 {args.x_sha256}")
                    return 1
            guards.append(("G4x", True, "重生成 x == dump x（硬闸门，不符即 exit 1）"))
        report.append(f"x 来源: {x_path}（dump）")
    except FileNotFoundError:
        if args.x_seed is None:
            print(f"[check_ref] FAIL: {x_path} 不存在且未给 --x-seed")
            return 1
        if args.x_sha256 is None:
            print("[check_ref] FAIL: 走 --x-seed 重建路径时 --x-sha256 是**强制**的"
                  "（否则无法证明重建的 x 与归档一致；tower 条件 ①）")
            return 1
        xbits = gen_activations_uniform(m, args.x_seed)
        x = bf16_view(xbits).reshape(m, HIDDEN)
        digest = hashlib.sha256(xbits.tobytes()).hexdigest()
        if digest != args.x_sha256:
            print(f"[check_ref] FAIL: 重建 x 的 sha256 {digest} != --x-sha256 {args.x_sha256}"
                  f"（x 与归档不一致 ⇒ 本档所有判定无效）")
            return 1
        report.append(f"x 来源: 重建(m={m}, seed={args.x_seed}) sha256={digest}（已与归档记录核对一致）")
    if args.w == "onehot":
        w = bf16_view(onehot_weight()).reshape(E, HIDDEN)
    else:
        w = bf16_view(load_bf16(args.w, (E, HIDDEN))).reshape(E, HIDDEN)

    # ---- golden（含设备 FTZ 口径建模）----
    # 设备 softmax 分数在 fp32 次正规区被硬件 FTZ flush 成 0（docs/05 §6.1）。**M46 之前**
    # `moe_block_ref.router_topk` 直接在 fp32 下算 np.exp 并保留次正规值 ⇒ 深尾 logits（|l|≳88）下
    # 两者的 top-k 并列次序不同（m=4097 + 真实权重档：20/4097 行的 top-10 集合不同）。**M46 起
    # golden 已按设备口径建模 FTZ**（`ftz_f32` + 3 个点位），所以下面这串"取 golden 的 logits 再
    # 自己 flush 一次"在语义上已与 golden 内部一致 —— 本脚本仍保留它，是为了让 R0（golden 的 ids
    #  vs 本参考的 ids）在两侧**任一** FTZ 建模被改动时立刻非零，充当回归护栏（详见文件头 R0 说明）。
    log_ref, ids_naive, _ = router_topk(x, w, KTOPK)
    sc = np.exp(log_ref)
    sc[sc < FTZ_MIN_NORMAL] = np.float32(0.0)
    order = np.argsort(-sc, axis=1, kind="stable")[:, :KTOPK]
    ids_ref = order.astype(np.int32)
    rows = np.arange(m)[:, None]
    w_ref = sc[rows, ids_ref]
    w_ref = (w_ref / w_ref.sum(axis=1, keepdims=True)).astype(np.float32)
    perm_ref = moe_permute(ids_ref, E)
    S = m * KTOPK
    # 报告项 R0。**标签必须与同 commit 的事实一致**（tower 规则）：
    # M46 起 `moe_block_ref.router_topk` 已建模设备 FTZ ⇒ 本项现在是「两套 FTZ 实现是否一致」，
    # 不再是「未建模 FTZ 的差异」。0/0 = 一致（预期值），非零 = 两侧 FTZ 建模分歧。
    report.append(f"R0 golden(已建模 FTZ) 与本参考的 ids 差（0 槽/行 = 两套 FTZ 实现一致）："
                  f"{int((ids_naive != ids_ref).sum())} 槽 / {int((ids_naive != ids_ref).any(axis=1).sum())} 行")

    # J1 ids
    ok = np.array_equal(ids, ids_ref)
    judged.append(("J1", bool(ok), f"topk_ids 逐元素相等（{ids.size} 项）"))
    # J2 weights ≤1 ulp
    wref_b = f32_to_bf16_bits(w_ref)
    ulp = np.abs(wbits_dev.astype(np.int32) - wref_b.astype(np.int32))
    judged.append(("J2", bool((ulp <= 1).all()), f"topk_weights |ulp| ≤ 1（max {int(ulp.max())}）"))
    # J3 logits —— T3（docs/17 §1.1：长 fp32 累加链）
    #   |dev - ref| ≤ ε·Σ|terms| + 0.5·ulp(out)，逐元素检查。
    #   ε 来源逐项：设备 GEMV = 40 深 lane 链（40×2^-24）+ 64 lane 树归约（6×2^-24）；
    #              本参考 = 2560 次顺序相加（2560×2^-24，链更长故必须计入）。
    #   Σ|x_k·w_e,k| 用上界 Rmax[t]·Wsum[e]（≤ max_k|x_k| · Σ_k|w_ek|）。
    EPS_T3 = (40.0 + 6.0 + 2560.0) * 2.0 ** -24
    if log_dev is None:
        dmax = None
        report.append("R1 logits max abs: N/A（--no-logits：该档 logits dump 未入库；"
                      "机内 J3 见 evidence/run_mode2.log 的 T3 界占用列）")
    else:
        dmax = float(np.abs(log_dev - log_ref).max())
        rmax = np.abs(x).max(axis=1)
        wsum = np.abs(w).sum(axis=1)
        ulp = np.ldexp(1.0, np.frexp(np.abs(log_ref))[1] - 24)
        bound = np.maximum(EPS_T3 * rmax[:, None] * wsum[None, :] + 0.5 * ulp, 1e-30)
        diff = np.abs(log_dev - log_ref)
        nbad = int((diff > bound).sum())
        util = float((diff / bound).max())
        judged.append(("J3", nbad == 0,
                       f"router_logits 逐元素 ≤ T3 界（ε={EPS_T3:.3e}；越界 {nbad} 元素；"
                       f"max abs {dmax:.3e}，界占用 {util * 100:.2f}%）"))
    # J4..J7
    judged.append(("J4", bool(np.array_equal(cnt_dev, perm_ref["expert_token_counts"])),
                   "expert_counts == golden expert_token_counts"))
    judged.append(("J5", bool(np.array_equal(base_dev, perm_ref["expert_offsets"][:E])),
                   "expert_slot_base == golden expert_offsets[:512]"))
    judged.append(("J6", bool(np.array_equal(ps_dev, perm_ref["perm_src_token"])),
                   f"perm_src_token == golden（{ps_dev.size} 项）"))
    judged.append(("J7", bool(np.array_equal(pe_dev, perm_ref["perm_expert"])),
                   f"perm_expert == golden（{pe_dev.size} 项）"))
    judged.append(("J8", bool(np.array_equal(g64_dev, cnt_dev.astype(np.int64))),
                   "group_list_i64 == int64(expert_counts)"))
    judged.append(("J9", bool(scl_dev[0] == S and scl_dev[3] == S),
                   f"route_scalars[0]/[3] == m*topk = {S}（实到 {scl_dev[0]}/{scl_dev[3]}）"))
    judged.append(("J10", bool(cnt_dev.sum() == S), f"Σ counts == {S}（实到 {int(cnt_dev.sum())}）"))

    # ---- report ----
    if log_dev is not None:
        rel = np.abs(log_dev - log_ref) / (np.abs(log_ref) + 1e-6)
        report.append(f"R1 logits: max abs {dmax:.3e} / max rel {float(rel.max()):.3e}")
    n_active = int((cnt_dev > 0).sum())
    report.append(f"R2 分布画像: 活跃专家 {n_active}/512, 空专家 {512 - n_active}, "
                  f"单 token 专家 {int((cnt_dev == 1).sum())}, 最大槽位 {int(cnt_dev.max())}")
    rowsum = bf16_view(wbits_dev).sum(axis=1)
    report.append(f"R3 每行 topk 权重行和: min {rowsum.min():.6f} max {rowsum.max():.6f}")

    # ---- guards ----
    if log_dev is not None:
        guards.append(("G1", bool(np.allclose(log_dev.max(axis=1), 0.0, atol=1e-6)),
                       "logits 每行 max == 0（max-shift）"))
    cov = np.bincount(pe_dev, minlength=E)
    guards.append(("G2", bool((ps_dev >= 0).all() and (ps_dev < m).all() and (pe_dev >= 0).all()
                              and (pe_dev < E).all() and np.array_equal(cov, cnt_dev)),
                   "perm 值域合法且与 counts 自洽"))
    guards.append(("G3", bool(scl_dev[2] == 0), "route_scalars[2]（脏 id 计数）== 0"))

    # ---- 输出 ----
    npass = sum(1 for _, ok, _ in judged if ok)
    for name, ok, desc in judged:
        print(f"[check_ref] {name:<4} {'PASS' if ok else 'FAIL'}  {desc}")
    print("[check_ref] ---- 报告项（不参与判定）----")
    for line in report:
        print(f"[check_ref] {line}")
    for name, ok, desc in guards:
        print(f"[check_ref] guard {name:<4} {'ok' if ok else 'NG'}  {desc}")

    all_ok = all(ok for _, ok, _ in judged)
    if not judged:
        print(f"[check_ref] RESULT: SKIPPED (0 判定项被比较 | guard "
              f"{sum(1 for _, ok, _ in guards if ok)}/{len(guards)}；这不是通过)")
        return 2
    print(f"[check_ref] m={m} : {'PASS' if all_ok else 'FAIL'} | 判定 {npass}/{len(judged)} "
          f"| guard {sum(1 for _, ok, _ in guards if ok)}/{len(guards)}")
    if all_ok:
        print(f"[check_ref] RESULT: OK ({npass}/{len(judged)} 判定项通过，"
              f"guard {sum(1 for _, ok, _ in guards if ok)}/{len(guards)})")
        return 0
    print(f"[check_ref] RESULT: FAIL ({len(judged) - npass}/{len(judged)} 判定项失败)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
