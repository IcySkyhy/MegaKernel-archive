#!/usr/bin/env python3
"""M46 排查：`tools/golden` 参考链上所有 fp32 乘法/相加/归约路径的 FTZ 可达性审计。

结论表（"会碰到次正规 / 已建模 / 处置"）见 README「FTZ 建模（M46）」一节；本脚本是那张
表的可复现取证，分三段：

  A. 数据实测：对 m1/m33 数据集的 fp32 中间量逐一致数（`0 < |v| < 2^-126` 的个数 +
     最小非零 |v|），覆盖 router logits/exp/softmax/权重、专家 GEMM 的乘积与中间量、
     SwiGLU 输入输出的 exp 与 1+exp、层链 norm 的 xAdd/平方和/输出。
  B. 构造探针：对"数学上可达但结果不可观测"的路径给出直接证明——
     silu/sigmoid（次正规项加到 1.0 上）、量化器 fp4 cast（次正规值编码到 0 码）、
     RMSNorm 的 eps（次正规平方和被 1e-6 吞掉）。
  C. 不可建模项的量级论证：GEMM/combine 的 fp32 累加链（设备 cube + vector），
     说明为什么"部分和落进次正规区"与本参考已建模的其它不确定度相比不可观测。

用法：
  python3.12 tools/golden/evidence/ftz_reachability_audit.py [--data tools/golden/data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import moe_block_ref as ref  # noqa: E402

FTZ = ref.FTZ_MIN_NORMAL


def sub_stats(tag: str, v: np.ndarray) -> None:
    """统计一个张量里落在 fp32 次正规区的元素（0 不计入次正规）。"""
    a = np.abs(np.asarray(v, dtype=np.float32))
    nz = a[a > 0.0]
    n_sub = int(((a > 0.0) & (a < FTZ)).sum())
    mn = float(nz.min()) if nz.size else 0.0
    n_zero = int((a == 0.0).sum())
    print(f"  {tag:<34} elems={a.size:>8}  次正规 {n_sub:>6}  |v|=0 {n_zero:>8}  "
          f"min|v|≠0 {mn:.3e}")


def audit_dataset(name: str, data_root: Path) -> None:
    outdir = data_root / name
    if not (outdir / "manifest.json").exists():
        print(f"  [{name}] 无数据集，跳过")
        return
    ds = ref.load_dataset(outdir)
    t, man = ds["tensors"], ds["manifest"]
    x = t["x.bin"]
    w = t["router_weight.bin"]
    top_k = man["top_k"]
    print(f"  [{name}] m={man['m']} E={man['num_experts']} top_k={top_k}")

    # --- router 段 ---------------------------------------------------------
    logits = x @ w.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted).astype(np.float32)
    scores = e / e.sum(axis=1, keepdims=True)
    _, ids, wt = ref.router_topk(x, w, top_k)
    sub_stats("router logits (pre max-shift)", logits)
    sub_stats("router logits (max-shifted)", shifted)
    sub_stats("exp(max-shifted logits)", e)
    sub_stats("softmax scores (ref /sum)", scores)
    sub_stats("topk_weights (renormed)", wt)

    # --- routed experts：GEMM 乘积 / gate_up / SwiGLU / down ----------------
    perm = ref.moe_permute(ids, man["num_experts"])
    xs = x[perm["perm_src_token"]]
    for eidx in range(man["num_experts"]):
        sel = perm["perm_expert"] == eidx
        if not np.any(sel):
            continue
        w_gu = ref.unpack_mxfp4(t["experts.gate_up_proj.bin"][eidx],
                                t["experts.gate_up_proj.weight_scale.bin"][eidx])
        w_dn = ref.unpack_mxfp4(t["experts.down_proj.bin"][eidx],
                                t["experts.down_proj.weight_scale.bin"][eidx])
        x_e = xs[sel]
        sub_stats(f"dequant w_gu (expert {eidx})", w_gu)
        g = x_e @ w_gu.T
        sub_stats(f"gate_up = x@w_gu.T (expert {eidx})", g)
        inter = w_dn.shape[1]
        sub_stats(f"SwiGLU exp(-gate) (expert {eidx})", np.exp(-g[:, :inter]))
        sub_stats(f"SwiGLU 1+exp (expert {eidx})", 1.0 + np.exp(-g[:, :inter]))
        h = (g[:, :inter] / (1.0 + np.exp(-g[:, :inter]))) * g[:, inter:]
        sub_stats(f"SwiGLU output (expert {eidx})", h)
        sub_stats(f"down = h@w_dn.T (expert {eidx})", h @ w_dn.T)
        break  # 一档一个专家足以说明量级（各专家同分布）

    # --- 共享专家 / 层链 ---------------------------------------------------
    s_gate = x @ t["shared_expert_gate_weight.bin"].T
    sub_stats("shared sigmoid 1+exp(-v)", 1.0 + np.exp(-s_gate))
    sub_stats("shared gate (sigmoid+MLP)", t["shared_output.bin"])
    sub_stats("routed_output", t["routed_output.bin"])
    sub_stats("moe_output", t["moe_output.bin"])
    x_add = t["res1.bin"]
    sub_stats("m6 xAdd (res1)", x_add)
    sub_stats("m6 mean(xAdd^2)", np.mean(x_add * x_add, axis=1))
    sub_stats("m6 y = bf16((xAdd*rstd)*gamma)", t["y_final.bin"])

    # --- 量化器：h = x(bf16) * halfScale（bf16 域乘，官方序列第 12 步） --------
    xw = ref.f32_to_bf16(x)
    x3 = xw.reshape(xw.shape[0], -1, 32)
    maxexp = np.max(ref.f32_to_bf16_bits(x3) & np.uint16(0x7F80), axis=2).astype(np.int32)
    shared = np.maximum(maxexp, ref.FP4_E2M1_BF16_MAX_EXP) - ref.FP4_E2M1_BF16_MAX_EXP
    half = ref.bf16_bits_to_f32((ref.BF16_EXP_BIAS - shared).astype(np.uint16))[:, :, None]
    sub_stats("quantizer h = x(bf16)*halfScale", x3 * half)


def probes() -> None:
    """B 段：构造探针，证明"数学可达但结果不可观测"的三条路径。"""
    print("\n[B] 构造探针（次正规可达但结果不可观测）")
    # 1) silu / sigmoid：exp(-v) 落次正规时，1+exp(-v) 在 fp32 下仍是 1.0
    v = np.array([88.8, 90.0, 100.0, 110.0], dtype=np.float32)
    ex = np.exp(-v).astype(np.float32)
    ok = bool((1.0 + ex == np.float32(1.0)).all())
    print(f"  silu/sigmoid: v={v.tolist()} -> exp(-v)={ex.tolist()}")
    print(f"    1+exp(-v) == 1.0 逐元素: {ok}  ⇒ v/(1+exp) == v/(1+0)（FTZ 不可观测）")
    # 2) fp4 cast：|h| < 2^-126 与 0 编码到同一 code（0 码）
    h = np.array([0.0, 1e-40, -1e-40, 1e-38, 5e-39], dtype=np.float32)
    codes = ref.e2m1_encode_away(h)
    print(f"  fp4 cast: h={h.tolist()} -> codes={codes.tolist()}"
          f"  ⇒ 次正规值全部落到 0 码（±0，FTZ 不可观测）")
    # 3) RMSNorm eps：次正规平方和被 1e-6 吞掉
    ss = np.array([0.0, 1e-40, 1e-38], dtype=np.float64)
    r1 = np.float32(1.0) / np.sqrt(np.float32(ss / 2560.0) + ref.RMSNORM_EPS)
    r0 = np.float32(1.0) / np.sqrt(np.float32(0.0) + ref.RMSNORM_EPS)
    print(f"  RMSNorm: sqrt(ss/2560+1e-6) = {r1.tolist()} vs 零次正规 {r0}")
    print(f"    ⇒ ss 在次正规区时 rstd 与 ss=0 逐位相同（eps=1e-6 主导）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE.parent / "data")
    ap.add_argument("--groups", default="m1,m33")
    args = ap.parse_args()

    print("# FP32 FTZ 可达性审计（threshold = 2^-126 = 1.1754944e-38）")
    print("# 次正规计数 = 0 < |v| < 2^-126 的元素数（0 不计）\n")
    print("[A] 数据实测（已入库的缩小档；真实规模 router 见 router_ftz_device_repro.log）")
    for name in [g for g in args.groups.split(",") if g]:
        audit_dataset(name, args.data)
        print()
    probes()
    print("\n[C] 不可建模项（GEMM/combine 的 fp32 累加链）")
    print("  设备 cube fp32 累加 + vector 折叠 + 本参考的 numpy fp32 求和：累加次序不同，")
    print("  逐级 FTZ 无法在本参考里表达。判据是【部分和落进次正规区】需要全部已累加项")
    print("  相互抵消到 < 2^-126 —— 而同一链上 fp32 舍入噪声下限为 0.5·ulp(最大部分和)，")
    print("  量级 ~1e-7（m33 实测 min|v|≠0 见上表）≫ 2^-126，故该事件在本数据族不可达；")
    print("  且该链的不确定度早已被 T3 界 / 量化预算 / 1.5×oracle 容差覆盖（README 偏差预算）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
