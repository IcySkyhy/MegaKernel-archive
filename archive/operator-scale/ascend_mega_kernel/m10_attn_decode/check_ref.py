#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""m10_attn_decode 数值校验（口径按 docs/17 §L0；自指形态按 M59 的 S1/S2/S3/S4 代号标注）。

参考实现：**独立 numpy**，输入只取 host 生成的 dump（q/k/v = 算件的**声明输入**，即 M59 的 N1 输入隔离），
不取任何设备中间张量：
    S[s] = Σ_d q·k · scale（bf16 输入解量化后 fp64 累加；scale=256^-0.5=1/16；无 causal，仅 s<seq 有效）
    m    = **逐 tile 的运行 max**（与 donor 的在线 softmax 同构，pin 见下）
    E    = exp(S − m) ；P̃ = bf16_RNE(E)     ← 本核规格的一部分（donor 同款：exp 之后立刻降到 P 的低精度）
    out  = bf16_RNE( Σ_s P̃·V / Σ_s P̃ )     （fp64 累加，避免参考自身舍入进入判据）

**P̃ 口径的外部 pin**（M59 finding 要求把口径出处从"拟合设备"改成 donor 引用；都是 `文件:符号`）：
  · 运行 max（新 = max(旧, 本 tile)）：donor `attention/common/op_kernel/arch35/vf/vf_basic_block_aligned128_update.h`
    的 `Max(vreg_max_new, vreg_input_max, vreg_in_max, preg_all)`（该行注释即"计算新、旧max的最大值"）；
  · 重标定因子 = exp(旧 max − 新 max)：同文件 `ExpSub(vreg_rowmax_p, vreg_input_max, vreg_max_new, preg_all)`；
  · 每 tile 用**当时的**运行 max 做 exp：同文件 `ExpSub(vreg_exp_even, vreg_input_x, vreg_max, preg_all)`；
  · 在线累加式 dst = pre·expMax + cur：donor `vf_flashupdate_new.h` 的 `FlashUpdateBasicVF`；
  · exp 之后立刻 cast 到 P 的低精度（RINT 族）：donor `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 的
    `Cast<T2, T, castTraitRint*>`。**限度（写清不掩盖）**：该文件 T2=bfloat16_t 分支用的是
    `castTraitZero`（`RoundMode::UNKNOWN`，见 donor `block_epilogue_arch35_utils.hpp` 的 `castTraitZero`），
    只有 fp8 分支明写 `CAST_RINT` ⇒ "bf16 用 RNE" 这一条**没有**被 donor 逐字钉住；本仓按 RNE 实现
    （`f32_to_bf16_bits`），并由判据 C 逐元素对**设备自己的 P̃ dump** 独立复核（4096 全数组逐位，
    唯一 1 处差 1 个 bf16 格点）。⇒ 这一条属 **S2 残余（口径无外部逐字 pin）**，靠判据 C 的可判性兜住。

**判据（判定项 / 报告项 / guard 三栏，docs/17 §2.1）**：
  判定项：
    A. `out` 的**边界感知 ulp** 判据（T3；ε 推导见 README §6.6）；
    B. `out` 非全零 / 非常数 / 每 (n2,g) 行 256 列全非零（T4 结构性，3 项）；
    C. **P̃ 全数组**判据（T3，本轮新增）：从 q/k dump 独立重算**全部** (unit,slot,row,col) 的 P̃，
       与设备 `gp` dump 逐元素比 —— 这是"P 自身"的独立来源；
    D. **P̃ 位置/掩码结构性**判据（T4/T1，本轮新增）：未用槽位必须恒 0（咬 unit 索引空间）、
       尾 tile 的无效列必须恒 0（咬列 mask 契约）；
    E. **P·V（独立 P）**判据（T3，本轮新增）：BMM2 的 L0C 对 `einsum(P̃_ref, V)`（P̃_ref 由 q/k 独立重算）；
    F. **P·V（设备 P）**判据（M59 的 **S1**）：输入 = 设备产出的 `gp[unit][0]`。它的被judged量
       **是**"给定 P 下 BMM2 的布局/转置"，**不是** P 本身 —— P 由判据 C/D 独立咬。
  报告项（不参与 PASS/FAIL）：位级一致率 / ≤1ulp / ≤2ulp / maxAbsErr / maxRel（相消区主导）/
       单一相对累加界的覆盖率 / P̃ 的位差分布与翻转位置 / 容差占用（越界元素的最大"界占用"）。
  guard（不参与判定项计数，但失败即视为**判据工具自身失效**，退出码不为 0）：
       host mask 自检；**判别力对照**（把参考自身当设备 ⇒ 0 越界；人为 +1/+2 格点 ⇒ 必须被抓）；
       **口径负向对照**（把 P̃ 口径换成"全序列 global max"⇒ 必须报越界，除非该档只有一个 tile ⇒ 同解不可判）。

**退出码（docs/17 §8.3 三态）**：0 = 全部判定项通过且**无输入缺失**；1 = 比过且判定项/guard 有 FAIL；
2 = 有输入缺失（`RESULT: SKIPPED`，**绝不发合格证**）。

用法（在 dump 所在目录运行；dump 由 `M10_CASES=256,300,4096 ./m10_attn_decode` 生成）：
    /usr/local/python3.12.13/bin/python3 ../check_ref.py [seq ...]
    /usr/local/python3.12.13/bin/python3 ../check_ref.py pv [seq]     # 只做 P·V 两条判据
    /usr/local/python3.12.13/bin/python3 ../check_ref.py eps [seq]    # 只做 ε 的 nTile 记账自查（**不需要 dump**）

**"1 bf16 ulp" 的数值约定（全文一致，见 `bf16_grid_ulp` 的 docstring）**：一律指**相邻 bf16 格点间距**
= `2^(e-7)`（值落在 binade `[2^e, 2^(e+1))` 时；bf16 = 1 符号位 + 8 指数位 + **7 位显式尾数**），
即 bf16 的一个**存储步长**。三条依据（**自足表述，不引用其它文档的措辞**，M61 r2 评审要求）：
① 两个都已量化到同一 bf16 网格的值，最大合法差就是**一个格点间距**；
② **测量事实锚点**：M36 的 `real_m1` 用例里设备 `0x3dc0` 与参考 `0x3dc1` 是**相邻格点**；
③ 取**半格**会把判据 C 那唯一 1 处**合法**翻转判成越界（该处 `|Δ| = 1.9531e-3`、界占用 0.9965）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # noqa: E402

N2 = 2
G = 12           # 每 KV 头的 q 头数（gS1-merge 行数）
M_PAD = 16       # cube M（12 pad 16）
HD = 256
S2T = 256        # KV seq tile
SPLITS = 14
SCALE = np.float32(1.0 / 16.0)
U = 2.0 ** -24   # fp32 单位舍入（RNE）；本文件里所有"u"都指它

# ── Exp 项的出处：**官方规格**，不是设备探针实测（M61 修订）──────────────────────────
# 官方 Reg 矢量计算精度标准（`asc-devkit/docs/zh/api/appendix/reg_vector_compute_interface_precision_standard_summary.md`
# 表 1「基础算术 / Exp」行 = "1ulp, not support denormalized numbers"，硬件与软仿同为 1 ulp；
# 同文档 `.../reg_vector_compute/basic_arithmetic/Exp.md` 的「最大精度误差」一节亦为 1 ulp（约束说明：无）。
# 本核的三处 Exp（`SoftmaxTileVf` 的 E 与 expdiff、`CombineWeightsVf` 的 combine 权重）都是 Reg `Exp`
# （`__VEC_SCOPE__` + RegTensor，fp32，默认 `ExpAlgo::INTRINSIC`）⇒ 该规格适用（README §6.6 有域条件分析）。
# 「1 ulp」→ 相对界的换算（**最坏情形，不是观测**）：y=exp(x) 落在 binade [2^e, 2^(e+1)) 时
# ulp(y)=2^(e-23)，y ≥ 2^e ⇒ |Δy|/y ≤ 2^-23 = 2·2^-24 ⇒ 本式的单位（2^-24）里取 EXP_ULP = 2。
EXP_ULP = 2.0


def bf16_ulp(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """两个 bf16 位型之间跨过多少个 bf16 可表示值（同号 = |Δbits|，异号 = 经 0 相加）。"""
    a = a_bits.astype(np.int32)
    b = b_bits.astype(np.int32)
    same_sign = (a & 0x8000) == (b & 0x8000)
    return np.where(same_sign, np.abs(a - b), (a & 0x7FFF) + (b & 0x7FFF) + 1)


def bf16_grid_ulp(bits: np.ndarray) -> np.ndarray:
    """每个 bf16 值**到更远离 0 的相邻格点**的绝对间距 = 该值处的 **1 bf16 ulp（本文件的数值约定）**。

    **约定（M61 r1 评审 P2-2 要求写明）**：本文件与 README 里的"1 bf16 ulp"一律指
    **相邻 bf16 格点之间的距离** = `2^(e-7)`（值落在 binade `[2^e, 2^(e+1))` 时），
    因为 bf16 = 1 符号位 + 8 指数位 + **7 位显式尾数** ⇒ 格点间距就是 `2^(e-7)`，即一个**存储步长**。

    · 三条依据（**自足表述，不引用其它文档的数值定义** —— M61 r2 评审要求：
      凡是"相对另一份文档的定义相差 N 倍"这类陈述，都会在那份文档修订后变成假陈述，故本处不写）：
      ① 判据里这一项要回答的问题是"两个**都已被量化到同一 bf16 网格**的值，最大合法差是多少" ——
         答案是**一个格点间距**，不是半个；
      ② **测量事实锚点**：M36 的 `real_m1` 用例里，设备 `0x3dc0` 与参考 `0x3dc1` 是**相邻格点**
         （恰差 1 个格点间距）；
      ③ 若取**半格**，会把**合法的 1 格翻转**判成越界 —— 判据 C 唯一那 1 处
         （`|Δ| = 1.9531e-3` vs `tol = 1.9601e-3`、占用 0.9965）就会被误判成 FAIL 1/131072。
    （配套的 `bf16_ulp()` 是**位距**度量"跨过多少个可表示值"，与本约定自洽：1 格翻转 ⇒ 位距 1。）

    实现：取该值的同号邻居（正数 +1 位型、负数 −1 位型）并取绝对值差。
    P̃ ≥ 0（exp 的输出），但仍按符号分两侧取邻居，避免 −0 / 次正规角落出错。
    """
    b = bits.astype(np.int32) & 0xFFFF
    pos = (b & 0x8000) == 0
    nxt = np.where(pos, np.minimum(b + 1, 0x7F7F), np.maximum(b - 1, 0x8001)).astype(np.uint16)
    return np.abs(bf16_bits_to_f32(nxt).astype(np.float64) - bf16_bits_to_f32(bits).astype(np.float64))


def load_case(seq: int):
    pre = f"m10_case_s{seq}_"
    seq_pad = (seq + S2T - 1) // S2T * S2T
    q = np.fromfile(pre + "q.bin", dtype=np.uint16).reshape(N2, M_PAD, HD)
    k = np.fromfile(pre + "k.bin", dtype=np.uint16).reshape(N2, seq_pad, HD)
    v = np.fromfile(pre + "v.bin", dtype=np.uint16).reshape(N2, seq_pad, HD)
    out = np.fromfile(pre + "out.bin", dtype=np.uint16).reshape(N2, G, HD)
    return q, k, v, out, seq_pad


def split_ranges(seq):
    """复刻 kernel 的 FD 切分：chunkTiles=ceil(nTiles/SPLITS)，nsplit=ceil(nTiles/chunkTiles)。"""
    n_tiles = (seq + S2T - 1) // S2T
    chunk = (n_tiles + SPLITS - 1) // SPLITS
    nsplit = (n_tiles + chunk - 1) // chunk
    return [(i * chunk * S2T, min(seq, (i * chunk + chunk) * S2T)) for i in range(nsplit)]


def eps_terms(seq):
    """ε 的各项（单位 = 2^-24），逐项可复算；返回 (ε, terms, nTile)。

    · S2T       ：cube（mmad）在 k=256 上的累加，保守取 k·u·Σ|A·B|（真实数据实测只占该界的 0.13~0.15%，
                  README §6.3；此处仍全额计入、不放松）
    · nsplit    ：跨 split 归并的一次乘加（num/absnum/den 各一）
    · 2·nTile   ：每 tile 的在线重标定（Mul）+ 新 tile 累加（Add）各一次 fp32 舍入
    · EXP_ULP·nTile：Reg `Exp` 自身误差（**官方规格 1 ulp**，见文件头）出现的次数 —— 每个元素被
                  E = exp(S−mNew) 影响 1 次（每 tile 一次）+ 被重标定因子 exp(mOld−mNew) 影响
                  (nTile−1) 次（首 tile 的 mOld = −inf/MASKV ⇒ exp 精确为 0，无误差）⇒ 合计 nTile 次。
                  **M61 修订**：旧式只计了 1 次（`+EXP_ULP`），漏掉重标定因子的 Exp 误差；补全后
                  nTile=1 的档数值不变，只有 4096（2 tile/split）从 270u 变 272u（+0.74%）。
    `nTile` 的定义 = **实际调度里每 split 最多的 tile 数**（= kernel 的 `chunkTiles`）。旧式写成
    `max(1, ceil(seq/S2T)//nsplit)`，在 `n_tiles = 15`（seq ∈ [3585, 3840]）时给 **1** 而实际是 **2**
    ⇒ 该窗口的 ε 少 4u（`2·nTile` 与 `EXP_ULP·nTile` 各少 2u、约 −1.5%，方向是**更严**、不会假 PASS）。
    **M61 r1 评审 P2-1 修**：nTile 改为直接数实际调度，并用紧随其后的断言与 kernel 的切分式逐项对齐，
    使这一类少算不再能静默发生（自查命令：`check_ref.py eps [seq…]` 打印逐档对照表）。
    """
    rngs = split_ranges(seq)
    nsplit = len(rngs)
    n_tile = max(len(range(lo, hi, S2T)) for lo, hi in rngs)   # 实际调度：每 split 的最大 tile 数
    # 独立复算（与 m10_attn_decode.asc 的 AIC/AIV 同式：chunkTiles = CeilDiv(nTiles, SPLITS)、
    # nsplit = CeilDiv(nTiles, chunkTiles)、tiles = MinU(chunkTiles, nTiles − split·chunkTiles)）
    n_tiles_all = (seq + S2T - 1) // S2T
    chunk_tiles = -(-n_tiles_all // SPLITS)
    nsplit_k = -(-n_tiles_all // chunk_tiles)
    assert n_tile == chunk_tiles and nsplit == nsplit_k, (
        f"eps_terms 调度记账不一致: seq={seq} 实际(nTile={n_tile}, nsplit={nsplit}) "
        f"vs kernel 式(chunkTiles={chunk_tiles}, nsplit={nsplit_k})")
    terms = {"S2T": float(S2T), "nsplit": float(nsplit), "2·nTile": 2.0 * n_tile,
             "EXP_ULP·nTile": EXP_ULP * n_tile}
    return sum(terms.values()) * U, terms, n_tile


def eps_schedule(seq):
    """ε 的调度事实（`check_ref.py eps` 与 README 的对照表用；**不需要任何 dump**）。"""
    rngs = split_ranges(seq)
    per_split = [len(range(lo, hi, S2T)) for lo, hi in rngs]
    n_tiles_all = (seq + S2T - 1) // S2T
    chunk_tiles = -(-n_tiles_all // SPLITS)
    return {"seq": seq, "n_tiles_all": n_tiles_all, "nsplit": len(rngs), "per_split": per_split,
            "nTile": max(per_split), "nTile_old": max(1, -(-seq // S2T) // len(rngs)),
            "chunkTiles_kernel": chunk_tiles, "nsplit_kernel": -(-n_tiles_all // chunk_tiles)}


def eps_main(cases):
    """`check_ref.py eps [seq…]`：只做 ε 的 nTile 记账自查（无需 dump）。退出码 0/1（同 §8.3 的两态）。"""
    print("[eps] ε 的 nTile 记账自查：nTile = 实际调度里每 split 最多的 tile 数 = kernel 的 chunkTiles")
    bad, mismatch = 0, 0
    for seq in cases:
        s = eps_schedule(seq)
        old_e = S2T + s["nsplit"] + 2 * s["nTile_old"] + EXP_ULP * s["nTile_old"]
        new_e = S2T + s["nsplit"] + 2 * s["nTile"] + EXP_ULP * s["nTile"]
        ok = (s["nTile"] == s["chunkTiles_kernel"]) and (s["nsplit"] == s["nsplit_kernel"])
        bad += 0 if ok else 1
        diff = "" if s["nTile_old"] == s["nTile"] else " **旧式少算一档**（已修）"
        if s["nTile_old"] != s["nTile"]:
            mismatch += 1
        print(f"[eps] seq={seq:5d} | nTiles={s['n_tiles_all']:3d} nsplit={s['nsplit']:2d}"
              f" | 每 split tile 数 {s['per_split']} | 实际 nTile={s['nTile']} / kernel chunkTiles="
              f"{s['chunkTiles_kernel']} {'OK' if ok else '**MISMATCH**'}"
              f" | 旧式 nTile={s['nTile_old']}{diff} | ε {old_e}g·2^-24 → {new_e}g·2^-24"
              f"（{(new_e / old_e - 1) * 100:+.2f}%）")
    if bad:
        print(f"RESULT: FAIL ({bad}/{len(cases)} 档的 nTile 与 kernel 切分式不一致)")
        return 1
    print(f"RESULT: OK ({len(cases)} 档：新式 nTile 与「实际调度」及 kernel 切分式逐档一致；"
          f"其中 {mismatch} 档旧式少算一档——窗口 seq ∈ [3585, 3840]，本轮已修）")
    return 0


def ref_fa(q, k, v, seq, dtype=np.float64, quant_p=True):
    """FD split-K + **逐 tile 在线重标定**口径的 FA 参考（与 kernel 的每一步逐项对齐）：

    每个 split 内按 S2T 逐 tile：
        m_new = max(m_run, rowMax(tile))            # 运行 max
        E     = exp(S_tile − m_new)                 # 用**新** max
        P̃    = bf16_RNE(E)                         # ← 关键：bf16 舍入发生在**每个 tile**、
                                                    #    用的是**当时的运行 max**，不是整个 split 的 max
        acc   = acc·e^{m_run − m_new} + Σ P̃·V(tile)
        sum   = sum·e^{m_run − m_new} + Σ P̃
        m_run = m_new
    跨 split 归并（与 Combine() 同式）：
        M = max_i m_run_i；out = Σ acc_i·e^{m_run_i−M} / Σ sum_i·e^{m_run_i−M}

    ★ 为什么必须这样建模（M24 第八/九轮实测）：若把 P̃ 写成 bf16(exp(S − split_max))，就与 kernel 的
    "逐 tile、运行 max 下舍入" 差一次 bf16 量化（≈0.2% 相对），partials 会偏 1e-3 量级、
    最终 out 会偏 ~1 ulp 以上——这是**参考建模问题**，不是 kernel 精度问题（实测 P̃ 逐位一致）。
    该口径的外部 pin（donor 的 `Max(新,旧)` / `ExpSub(x, 新max)` / `ExpSub(旧max, 新max)`）见文件头。
    """
    seq_pad = k.shape[1]
    qf = bf16_bits_to_f32(q)[:, :G, :].astype(dtype)
    kf = bf16_bits_to_f32(k).astype(dtype)
    vf = bf16_bits_to_f32(v).astype(dtype)
    s_all = (np.einsum("ngd,nkd->ngk", qf, kf) * dtype(SCALE)).astype(dtype)
    n2n = q.shape[0]
    num = np.zeros((n2n, G, HD), dtype=dtype)
    absnum = np.zeros((n2n, G, HD), dtype=dtype)   # Σ_i |acc_i·w_i|：累加量级（有界误差判据用）
    den = np.zeros((n2n, G, 1), dtype=dtype)
    m_glob = np.full((n2n, G, 1), -np.inf, dtype=dtype)
    p_first = None
    m_first = None
    for (lo, hi) in split_ranges(seq):
        m_run = np.full((n2n, G, 1), -np.inf, dtype=dtype)
        acc = np.zeros((n2n, G, HD), dtype=dtype)
        ssum = np.zeros((n2n, G, 1), dtype=dtype)
        for a in range(lo, hi, S2T):
            b = min(a + S2T, hi)
            s = s_all[:, :, a:b]
            m_t = s.max(axis=2, keepdims=True)
            m_new = np.maximum(m_run, m_t)
            ed = np.exp((m_run - m_new).astype(dtype))        # m_run=-inf ⇒ ed=0
            e = np.exp((s - m_new).astype(dtype))
            p = bf16_bits_to_f32(f32_to_bf16_bits(e.astype(np.float32))).astype(dtype) if quant_p else e
            acc = acc * ed + np.einsum("ngk,nkd->ngd", p, vf[:, a:b, :])
            ssum = ssum * ed + p.sum(axis=2, keepdims=True)
            m_run = m_new
            if p_first is None:
                p_first, m_first = p, m_t
        mg = np.maximum(m_glob, m_run)
        num = num * np.exp((m_glob - mg).astype(dtype)) + acc * np.exp((m_run - mg).astype(dtype))
        absnum = absnum * np.exp((m_glob - mg).astype(dtype)) + np.abs(acc) * np.exp((m_run - mg).astype(dtype))
        den = den * np.exp((m_glob - mg).astype(dtype)) + ssum * np.exp((m_run - mg).astype(dtype))
        m_glob = mg
    # 归并量级：Σ_i |acc_i·w_i| / den —— 有界误差判据的分母（相对"累加量级"而非输出元素，
    # 否则在相消区 |out|→0 会把相对误差荒谬放大）
    scale = absnum / np.maximum(np.abs(den), dtype(1e-38))
    return (num / den).astype(dtype), p_first, m_first, scale


def ref_p_all(q, k, seq, mode="running"):
    """从 host dump 的 q/k **独立重算全数组 P̃**（bf16 位型），复刻 kernel 的 P 调度。

    返回 (slots, absarg, meta)：
      slots[(unit, slot)]  : uint16[16,256]，该 tile 的 P̃（列 ≥ 该 tile 有效列数处为 0）
      absarg[(unit,slot)]  : float64[16,256]，Σ_d|q·k|·scale（mmad k 项保守界的底 ⇒ exp 自变量误差的底）
      meta[(unit,slot)]    : (split, n2, tile, base_token, ncol)

    调度（与 `m10_attn_decode.asc` 的 `SoftmaxTileVf` 调用点 + "P 落 GM" 一致）：
      unit = 2*split + n2（= AIC 的 blockIdx 空间）、slot = t % 3（L1/P 3-buffer 轮转）、
      行 0..15 = cube M 维（12 个 q 头 + 4 个 pad 行，pad 行的 q=0 ⇒ P̃=1）、列 = 该 tile 的 256 个 token。
    mode="running"：kernel 口径（逐 tile 运行 max）；mode="globalmax"：**负向对照**用的错口径
      （全序列行 max，即经典非在线 softmax）——两口径只在序列多于一个 tile 时可区分。
    """
    qf = bf16_bits_to_f32(q).astype(np.float64)
    kf = bf16_bits_to_f32(k).astype(np.float64)
    s_all = np.einsum("ngd,nkd->ngk", qf, kf) * np.float64(SCALE)
    absarg_all = np.einsum("ngd,nkd->ngk", np.abs(qf), np.abs(kf)) * np.float64(SCALE)
    gmax = s_all.max(axis=2, keepdims=True)
    slots, absarg, meta = {}, {}, {}
    for si, (lo, hi) in enumerate(split_ranges(seq)):
        n_tile = (hi - lo + S2T - 1) // S2T
        m_run = np.full((q.shape[0], M_PAD, 1), -np.inf)
        for t in range(n_tile):
            base = lo + t * S2T
            ncol = min(base + S2T, seq) - base
            st = s_all[:, :, base:base + ncol]
            m_new = np.maximum(m_run, st.max(axis=2, keepdims=True))
            m_use = m_new if mode == "running" else gmax
            p = np.zeros((q.shape[0], M_PAD, S2T), dtype=np.uint16)
            p[:, :, :ncol] = f32_to_bf16_bits(np.exp(st - m_use).astype(np.float32))
            ab = np.zeros((q.shape[0], M_PAD, S2T), dtype=np.float64)
            ab[:, :, :ncol] = absarg_all[:, :, base:base + ncol]
            for n2 in range(q.shape[0]):
                slots[(2 * si + n2, t % 3)] = p[n2]
                absarg[(2 * si + n2, t % 3)] = ab[n2]
                meta[(2 * si + n2, t % 3)] = (si, n2, t, base, ncol)
            m_run = m_new
    return slots, absarg, meta


def p_judge(dev_bits, ref_bits, ncol, eps_arg):
    """判据 C 的逐元素核心：返回 (越界掩码, |Δ|, 容差, 有效列掩码)。

    档位 **T3**（docs/17 §1.1 三个触发条件全中：Reg `Exp` 近似 + 跨 tile 在线重标定 + mmad k>16）。
    判据（T3 形式的逐元素检查）：
        |P̃_dev − P̃_ref| ≤ 1.0·ulp(P̃_ref) + ε_arg·|P̃_ref|
    · 1.0·ulp 项：**参考与设备两侧都已被量化到 bf16 格点** ⇒ 按 docs/17 §1.1 的"参考本身已量化"
      条款取 1.0·ulp（不是 0.5·ulp）：两个相邻格点各由就近的实值舍入而来时，格点对格点的最大合法差 = 1 ulp。
    · ε_arg·|P̃_ref| 项：exp 的自变量误差传播（exp' = exp ⇒ 结果的相对误差 = 自变量的绝对误差）：
        ε_arg = (S2T + EXP_ULP)·u = mmad k=256 的 S 累加界 + Exp 自身 1 ulp（官方规格）；
        S 的 Muls(1/16) 是 2 的幂乘（精确），Sub 是 0 ulp（官方规格表 "Add/Sub 0ulp"）⇒ 不另计。
    逐元素检查；超界元素必须逐个有解释。
    """
    dev = bf16_bits_to_f32(dev_bits).astype(np.float64)
    ref = bf16_bits_to_f32(ref_bits).astype(np.float64)
    tol = 1.0 * bf16_grid_ulp(ref_bits) + eps_arg * np.abs(ref)
    sel = np.zeros(ref_bits.shape, dtype=bool)
    sel[:, :ncol] = True
    d = np.abs(dev - ref)
    return (sel & (d > tol)), d, tol, sel


def pv_tol_independent(ref_bits, vf, eps_arg):
    """判据 E（P·V，独立 P）的逐元素容差：把"参考 P̃ 与设备 P̃ 允许的差"传播过 PV 求和。

    tol[row,d] = S2T·u·Σ_k|P̃_ref·V|                        ← mmad k=256 累加（保守界）
               + Σ_k (1.0·ulp(P̃_ref_k) + ε_arg·P̃_ref_k)·|V_kd|   ← 参考 P̃ 与设备 P̃ 的允许差
    两项都是**推导**出来的（不是实测拟合）：第一项是 cube 累加的单位舍入界，第二项是判据 C 的
    逐元素允许差（参考已被量化到 bf16 格点 ⇒ 1.0·ulp）经 V 传播。**分辨率声明**：第二项占主导
    （≈2^-8 = 0.39%·Σ|P̃V|）⇒ 本判据咬得住 P 的**约定级/布局级**错误，咬不住比 0.39% 更细的差异
    （更细的由判据 C 在元素级承担：那里的格点项同样是 1 ulp，但比较对象是 P̃ 本身）。
    """
    p = bf16_bits_to_f32(ref_bits).astype(np.float64)
    du = (bf16_grid_ulp(ref_bits) + eps_arg * np.abs(p)) @ np.abs(vf)
    base = np.abs(p) @ np.abs(vf)
    return S2T * U * base + du, base


class Tally:
    """判定项 / 报告项 / guard 三栏计数（docs/17 §2.1：只有判定项进 PASS/FAIL 计数）。"""

    def __init__(self):
        self.items = []          # (kind, name, payload)

    def judge(self, name, ok, note=""):
        self.items.append(("judge", name, bool(ok)))
        if note:
            print(f"     判定项[{name}]: {'PASS' if ok else 'FAIL'} {note}")
        return bool(ok)

    def report(self, name, text):
        self.items.append(("report", name, text))

    def guard(self, name, ok, note=""):
        self.items.append(("guard", name, bool(ok)))
        print(f"     guard[{name}]: {'OK' if ok else '**FAIL**'} {note}")
        return bool(ok)

    def skip(self, name, why):
        self.items.append(("skip", name, why))

    def count(self, kind):
        return sum(1 for it in self.items if it[0] == kind)

    def judge_fails(self):
        return [it[1] for it in self.items if it[0] == "judge" and not it[2]]

    def guard_fails(self):
        return [it[1] for it in self.items if it[0] == "guard" and not it[2]]

    def skips(self):
        return [(it[1], it[2]) for it in self.items if it[0] == "skip"]


def need(seq, names, T, who):
    """缺输入 ⇒ 记 skip（绝不当成 PASS）。返回缺失文件名列表。"""
    miss = [n for n in names if not os.path.exists(f"m10_case_s{seq}_{n}.bin")]
    if miss:
        T.skip(who, ",".join(miss))
        print(f"[skip] seq={seq} {who}: 缺 {','.join(miss)}（**不比较、不发合格证**）")
    return miss


def check(seq: int, T: Tally) -> None:
    if need(seq, ["q", "k", "v", "out"], T, "A/B out 判据"):
        return
    q, k, v, out, seq_pad = load_case(seq)
    ref64, p64, _, scale = ref_fa(q, k, v, seq, np.float64)   # 判定用参考（量化匹配 + fp64 累加）
    ref32, _, _, _ = ref_fa(q, k, v, seq, np.float32)         # 报告用：fp32 累加
    exp_bits = f32_to_bf16_bits(ref64.astype(np.float32)).astype(np.uint16)
    exp_bits32 = f32_to_bf16_bits(ref32.astype(np.float32)).astype(np.uint16)

    ulp = bf16_ulp(out, exp_bits)
    bit_exact = float((ulp == 0).mean())
    le1 = float((ulp <= 1).mean())
    le2 = float((ulp <= 2).mean())
    got_f = bf16_bits_to_f32(out).astype(np.float64)
    ref32f = bf16_bits_to_f32(exp_bits32).astype(np.float64)
    abs_err = np.abs(got_f - ref64)
    max_abs = float(abs_err.max())
    rel = np.abs(got_f - ref32f) / np.maximum(np.abs(ref32f), 1e-30)
    i_rel = tuple(int(x) for x in np.unravel_index(int(np.argmax(rel)), rel.shape))

    # ── 判定项 A：边界感知 ulp（docs/17 §L0；推导见 README §6.6）───────────────────
    # ① 常规元素（参考距**最近 bf16 网格点**的间距 ≥ δ·ulp(out)）：判 ≤1 ulp（严格）。
    # ② 边界元素（参考落在相邻两网格点的**中点** ±noise_ub 内）：fp32 归约的最后一次舍入
    #    （≤0.5 ulp）足以把它推到任一侧的相邻格点 ⇒ 与正确格点可差 2 ulp。故判 ≤2 ulp，
    #    且必须同时满足**绝对界**  |out − ref| ≤ ε·(Σ|acc·w|/den) + 0.5·ulp(out)。
    # 为什么单一相对累加界在边界元素上必然不覆盖：该界量化的是**归约噪声**，而相消区
    #    |out| 可比同行量级小两个数量级 ⇒ noise_ub/ulp(out) 随相消程度线性增长（δ 的定义），
    #    可以 ≫0.5，此时"≤1 ulp"不可判。
    eps, terms, n_tile = eps_terms(seq)
    noise_ub = eps * scale                      # 相对"累加量级" Σ|acc·w|/den 的噪声上界
    ob = exp_bits.astype(np.int32)
    g0 = bf16_bits_to_f32(exp_bits).astype(np.float64)
    nb = ((ob + np.where(ref64 >= g0, 1, -1)) & 0xFFFF).astype(np.uint16)  # 背离 ref 的那侧格点
    gn = bf16_bits_to_f32(nb).astype(np.float64)
    mid = 0.5 * (g0 + gn)
    ulp_out = np.abs(gn - g0)                   # 本地 bf16 网格间距（即 1 ulp 的绝对值）
    is_bnd = np.abs(ref64 - mid) <= noise_ub
    delta = noise_ub[is_bnd] / np.maximum(ulp_out[is_bnd], 1e-300)
    dmax = float(delta.max()) if delta.size else 0.0
    ok_inst = ulp <= np.where(is_bnd, 2.0, 1.0)
    ok_bnd_abs = np.where(is_bnd, abs_err <= noise_ub + 0.5 * ulp_out, True)
    ok_ulp = bool(ok_inst.all() and ok_bnd_abs.all())
    n_norm, n_bnd = int((~is_bnd).sum()), int(is_bnd.sum())
    n_norm_bad = int((~is_bnd & ~ok_inst).sum())
    n_bnd_bad = int((is_bnd & ~ok_inst).sum())
    n_bnd_abs_bad = int((is_bnd & ~ok_bnd_abs).sum())
    n_over = int((abs_err > noise_ub).sum())     # 单一相对累加界的覆盖率（报告项）
    ok_nonzero = bool(np.count_nonzero(out) > 0)
    ok_not_const = bool(np.unique(out).size > 1)
    nz_cols = np.count_nonzero(out.reshape(N2, G, HD), axis=2)

    print(f"[check] seq={seq:5d} (seq_pad={seq_pad})")
    print(f"  判定项 A（out 边界感知 ulp，T3）: {'PASS' if ok_ulp else 'FAIL'}"
          f" | 常规元素 {n_norm} 个（判 ≤1 ulp）：越界 {n_norm_bad}"
          f" | 边界元素 {n_bnd} 个（判 ≤2 ulp + 绝对界）：ulp 越界 {n_bnd_bad}、绝对界越界 {n_bnd_abs_bad}"
          f" | max {int(ulp.max())} ulp")
    print(f"    ε 各项（单位 2^-24；Exp 项 = **官方 Reg 精度规格 1 ulp**，不是设备实测）:"
          f" S2T {terms['S2T']:g} + nsplit {terms['nsplit']:g} + 2·nTile {terms['2·nTile']:g}"
          f" + EXP_ULP·nTile {terms['EXP_ULP·nTile']:g}（nTile={n_tile}）"
          f" = {eps / U:g}·2^-24，ε = {eps:.3e}"
          f" ⇒ 边界元素 δ = noise_ub/ulp(out) 最大 {dmax:.4f}（无边界元素时 δ=0）")
    print(f"    绝对界（仅边界元素）: ε·(Σ|acc·w|/den) + 0.5·ulp(out)，"
          f"最大 {float((noise_ub + 0.5 * ulp_out).max()):.3e}")
    T.judge("A. out 边界感知 ulp", ok_ulp,
            f"常规 {n_norm}(越界 {n_norm_bad}) / 边界 {n_bnd}(ulp 越界 {n_bnd_bad}、绝对界越界 {n_bnd_abs_bad})")
    # ── 判定项 B：out 结构性（T4）────────────────────────────────────────────────
    print(f"  判定项 B（out 结构性，T4）: 非全零 {'PASS' if ok_nonzero else 'FAIL'}"
          f" | 非常数 {'PASS' if ok_not_const else 'FAIL'}"
          f" | 每行 256 列非零 {'PASS' if nz_cols.min() == HD else 'FAIL'} (min {int(nz_cols.min())})")
    T.judge("B. out 非全零", ok_nonzero)
    T.judge("B. out 非常数", ok_not_const)
    T.judge("B. out 每行 256 列非零", nz_cols.min() == HD, f"min {int(nz_cols.min())}")
    # ── 报告项 ──────────────────────────────────────────────────────────────────
    print(f"  报告项: 位级一致率 {bit_exact:.4%} | ≤1ulp {le1:.4%} | ≤2ulp {le2:.4%} | maxAbsErr {max_abs:.3e}")
    print(f"  报告项: maxRel {float(rel.max()):.3e}"
          f" —— 由相消区元素主导 out[{i_rel[0]}][{i_rel[1]}][{i_rel[2]}]:"
          f" |ref|={abs(float(ref32f[i_rel])):.3e}、|dev−ref|={float(abs(got_f[i_rel] - ref32f[i_rel])):.3e}、"
          f"该行量级 {float(np.abs(ref32f[i_rel[0], i_rel[1]]).max()):.3e}"
          f"（相消 {float(np.abs(ref32f[i_rel[0], i_rel[1]]).max() / max(abs(float(ref32f[i_rel])), 1e-30)):.0f}×）；"
          f"报告项，小分母不由判定项承担")
    print(f"  报告项: 单一相对累加界 ε·(Σ|acc·w|/den) 覆盖 {abs_err.size - n_over}/{abs_err.size} 处"
          f"（不足 {n_over} 处中的 {n_bnd} 个即边界元素/相消区；见 README §6.6）")
    T.report("A. out 位级/≤1ulp/≤2ulp/maxAbsErr",
             f"{bit_exact:.4%}/{le1:.4%}/{le2:.4%}/{max_abs:.3e}")
    T.report("A. out maxRel（相消区主导）", f"{float(rel.max()):.3e} @ out{i_rel}")
    T.report("A. 单一相对累加界覆盖率", f"{abs_err.size - n_over}/{abs_err.size}")
    if not ok_ulp:
        bad = np.argwhere(~(ok_inst & ok_bnd_abs))
        n2, g, d = bad[0]
        print(f"  判定 A 首个越界位置 out[{n2}][{g}][{d}]: got {out[n2, g, d]:#06x} expect {exp_bits[n2, g, d]:#06x}"
              f" ({ulp[n2, g, d]} ulp, 边界元素={bool(is_bnd[n2, g, d])},"
              f" absErr {abs_err[n2, g, d]:.3e} vs 界 {noise_ub[n2, g, d] + 0.5 * ulp_out[n2, g, d]:.3e})"
              f", 共 {len(bad)} 处")

    # host 侧一致性自检（与 kernel 的"只有尾 tile 加 mask"契约配套）
    valid_tail = seq - (seq // 256) * 256
    print(f"  guard(host mask 契约): tail={valid_tail} "
          f"{'全 0（无尾 tile）' if valid_tail == 0 else f'{256 - valid_tail} 列被屏蔽'}")
    T.guard("host mask 契约（自检，恒真）", True, f"tail={valid_tail}")


def check_p_full(seq: int, T: Tally) -> None:
    """判据 C（P̃ 全数组逐元素）+ 判据 D（P̃ 位置/掩码结构性）+ 两条 guard。

    这是"**P 自身**"的独立来源：P̃_ref 只由 host 的 q/k dump（算件的声明输入 = M59 的 N1）算出，
    不碰任何设备字节；比较对象是设备 `gp`（P 的 GM 中转 dump）。
    档位与逐元素判据见 `p_judge` 的 docstring（T3 + 1.0·ulp 格点项 + ε_arg）。
    """
    if need(seq, ["q", "k", "gp"], T, "C/D P̃ 全数组判据"):
        return
    q, k, v, out, seq_pad = load_case(seq)
    gp = np.fromfile(f"m10_case_s{seq}_gp.bin", dtype=np.uint16).reshape(28, 3, M_PAD, S2T)
    slots, absarg, meta = ref_p_all(q, k, seq)
    eps_arg = (S2T + EXP_ULP) * U
    print(f"[p] seq={seq}: 判据 C/D —— 设备 `gp` 全数组 vs host 从 q/k **独立重算**的 P̃（N1 输入隔离）"
          f" | 档 T3（Exp + 在线重标定 + mmad k=256）| 逐元素 |Δ| ≤ 1.0·ulp(P̃_ref) + ε_arg·|P̃_ref|，"
          f"ε_arg=(S2T {S2T} + Exp {EXP_ULP:g})·2^-24 = {eps_arg:.3e}")
    n_cmp = n_bad = 0
    occ = 0.0
    hist = np.zeros(4, dtype=np.int64)           # 位差 0/1/2/≥3 的计数（只看有效列）
    flips = []
    first_bad = None
    n_mask_nz = 0
    for unit, slot in sorted(slots):
        si, n2, t, base, ncol = meta[(unit, slot)]
        ref = slots[(unit, slot)]
        bad, d, tol, sel = p_judge(gp[unit, slot], ref, ncol, eps_arg)
        u = bf16_ulp(gp[unit, slot], ref)
        n_cmp += int(sel.sum())
        n_bad += int(bad.sum())
        hist += np.bincount(np.minimum(u[sel], 3), minlength=4)
        if sel.any():
            occ = max(occ, float((d[sel] / np.maximum(tol[sel], 1e-300)).max()))
        for r, c in np.argwhere(sel & (u > 0)):
            flips.append((int(unit), int(si), int(n2), int(t), int(base + c), int(r), int(u[r, c])))
        if bad.any() and first_bad is None:
            r, c = np.argwhere(bad)[0]
            first_bad = (int(unit), int(si), int(base + c), int(r), float(d[r, c]), float(tol[r, c]))
        n_mask_nz += int((gp[unit, slot][:, ncol:] != 0).sum())      # 判据 D：尾 tile 无效列必须 0
    used = set(slots)
    n_unused_nz = int(sum((gp[u, s] != 0).sum() for u in range(28) for s in range(3) if (u, s) not in used))
    n_unused = 28 * 3 - len(slots)
    n_gp = 28 * 3 * M_PAD * S2T
    print(f"    判据 C（逐元素，T3）: 比较 {n_cmp} 个元素 / 覆盖 {len(slots)} 个 (unit,slot) × 16 行"
          f"（含 4 个 q pad 行；pad 行的 q=0 ⇒ P̃=1）| 越界 {n_bad}"
          f" | 位差 0/1/2/≥3 格点 = {hist[0]}/{hist[1]}/{hist[2]}/{hist[3]}"
          f" | 最大界占用 {occ:.4f}")
    print(f"    判据 D（结构性，T4/T1）: 未用槽位非零 {n_unused_nz}/{n_unused}（必须 0，咬 unit 索引空间）"
          f" | 尾 tile 无效列非零 {n_mask_nz}（必须 0，咬列 mask 契约）")
    if flips:
        for f in flips[:6]:
            print(f"    P̃ 位差（≤1 格点，判据 C 内）: unit{f[0]}(split{f[1]},n2{f[2]}) slot{f[3]}"
                  f" row{f[5]} col{f[4] - meta[(f[0], f[3])][3]} token{f[4]} → {f[6]} 个格点")
    if first_bad:
        print(f"    判据 C 首个越界: unit{first_bad[0]} token{first_bad[2]} row{first_bad[3]}"
              f" |Δ|={first_bad[4]:.3e} > 界 {first_bad[5]:.3e}")
    print(f"    覆盖范围: 设备 buffer 共 {n_gp} 个槽位 = {len(slots)} 个在用一个(unit,slot)（{n_cmp} 元素逐元素比）"
          f" + {n_unused} 个未用槽位（按判据 D 必须恒 0，已全查）；无抽样")

    # guard：判别力对照（人为注入必须被抓）
    u0, s0 = sorted(slots)[0]
    ref0 = slots[(u0, s0)]
    ncol0 = meta[(u0, s0)][4]
    dev_same = ref0.copy()
    b_same = int(p_judge(dev_same, ref0, ncol0, eps_arg)[0].sum())
    dev_p1 = ref0.copy()
    r0, c0 = 0, 0
    dev_p1[r0, c0] = (int(ref0[r0, c0]) + 1) & 0xFFFF          # +1 个 bf16 格点
    b_p1 = int(p_judge(dev_p1, ref0, ncol0, eps_arg)[0].sum())
    dev_p2 = ref0.copy()
    dev_p2[r0, c0] = (int(ref0[r0, c0]) + 2) & 0xFFFF          # +2 个格点
    b_p2 = int(p_judge(dev_p2, ref0, ncol0, eps_arg)[0].sum())
    T.guard("C 判别力对照（分辨率 = 1 个 bf16 格点）", b_same == 0 and b_p1 == 0 and b_p2 == 1,
            f"参考自身当设备 ⇒ 越界 {b_same}（须 0）；某元素 +1 格点 ⇒ {b_p1}（**须 0**：1.0·ulp 是判据明写的允许差，"
            f"docs/17 的格点条款）；+2 格点 ⇒ {b_p2}（须 1，即 >1 ulp 必须被抓）")

    # guard：口径负向对照（用错口径当参考必须报越界）
    slots_w, _, meta_w = ref_p_all(q, k, seq, mode="globalmax")
    n_bad_w = 0
    for unit, slot in sorted(slots_w):
        ref = slots_w[(unit, slot)]
        ncol = meta_w[(unit, slot)][4]
        n_bad_w += int(p_judge(gp[unit, slot], ref, ncol, eps_arg)[0].sum())
    n_tiles = max(1, -(-seq // S2T))
    ok_neg = (n_bad_w > 0) or (n_tiles == 1)
    T.guard("C 口径负向对照（global max 口径必须被咬）", ok_neg,
            f"换用『全序列行 max』口径后越界 {n_bad_w}"
            + ("（该档全序列只有 1 个 tile ⇒ 两口径**同解、不可判**，如实报）" if n_tiles == 1 else "（>0 即判据非恒真）"))

    # 旧 `check_gp` 的窄范围读数（保持历史可比；容差已换成派生的 1.0·ulp + ε_arg）
    narrow = [(u, s) for (u, s) in slots if u == 0 and s == 0]
    if narrow:
        ref = slots[(0, 0)]
        d = np.abs(bf16_bits_to_f32(gp[0, 0][:G]).astype(np.float64)
                   - bf16_bits_to_f32(ref[:G]).astype(np.float64))
        tol = (1.0 * bf16_grid_ulp(ref) + eps_arg * np.abs(bf16_bits_to_f32(ref).astype(np.float64)))[:G]
        print(f"    旧 `check_gp` 窄范围（unit0/slot0、前 12 行、单 tile）在本判据内一并覆盖:"
              f" maxAbsErr {float(d.max()):.3e}（旧容差 5e-3 → 现为派生的 1.0·ulp+ε_arg，"
              f"最小 {float(tol.min()):.3e}，**更严**）")
    T.judge("C. P̃ 全数组逐元素（T3）", n_bad == 0,
            f"比较 {n_cmp} 元素、越界 {n_bad}、最大界占用 {occ:.4f}")
    T.judge("D. P̃ 位置/掩码结构性（T4/T1）", (n_unused_nz == 0) and (n_mask_nz == 0),
            f"未用槽位非零 {n_unused_nz}/{n_unused}、尾 tile 无效列非零 {n_mask_nz}")
    T.report("C. P̃ 位差分布 0/1/2/≥3", f"{hist[0]}/{hist[1]}/{hist[2]}/{hist[3]}")
    T.report("C. P̃ 覆盖元素数", f"{n_cmp}（{len(slots)} 个 (unit,slot)，无抽样）")
    T.report("C. P̃ 位差位置", "无" if not flips else "; ".join(
        f"unit{f[0]}/slot{f[3]}/row{f[5]}/token{f[4]}={f[6]}格点" for f in flips))
    T.report("C. 口径负向对照（global max）读数", f"越界 {n_bad_w}")


def check_pv_ref(seq: int, T: Tally) -> None:
    """判据 E：BMM2 的 L0C（设备 `dbgc2`）vs `einsum(P̃_ref, V)` —— **独立 P 的 P·V**（本轮新增）。

    覆盖：`dbgc2` 的 dump 条件是 AIC 的 `t == 0`（`FixpToGmDbg(dbgC2GM[bid*M_PAD*S2T + nh*128], …, 128, S2T)`），
    故每个 unit（= 2*split+n2）都有 tile0 的 [16,256] L0C（两个 N 半块 nh=0/1 各 128 列）。
    ⇒ 本判据覆盖 **所有 unit 的 tile0**（4096 档 = 16 个 unit × 16 行 × 256 列 = 65536 个元素，
    seq=256=2 个 unit、seq=300=4 个 unit），比旧的 `check_pv`（只比 unit0）覆盖面大一个数量级。
    容差见 `pv_tol_independent`（两项都是推导：mmad 累加界 + 参考 P̃ 的格点允许差经 V 传播）。
    """
    if need(seq, ["q", "k", "v", "gp", "dbgc2"], T, "E P·V（独立 P）判据"):
        return
    q, k, v, out, seq_pad = load_case(seq)
    gp = np.fromfile(f"m10_case_s{seq}_gp.bin", dtype=np.uint16).reshape(28, 3, M_PAD, S2T)
    pv = np.fromfile(f"m10_case_s{seq}_dbgc2.bin", dtype=np.float32).reshape(28, M_PAD, S2T)
    slots, absarg, meta = ref_p_all(q, k, seq)
    vf_all = bf16_bits_to_f32(v).astype(np.float64)
    eps_arg = (S2T + EXP_ULP) * U
    units = sorted({u for (u, s) in slots if s == 0})
    n_cmp = 0
    worst = (0.0, None)          # (|Δ|, 位置)
    occ = 0.0
    n_bad = 0
    worst_u0 = 0.0
    for unit in units:
        si, n2, t, base, ncol = meta[(unit, 0)]
        ref_bits = slots[(unit, 0)]
        vv = vf_all[n2, base:base + ncol]
        tol, base_abs = pv_tol_independent(ref_bits[:, :ncol], vv, eps_arg)
        ref = bf16_bits_to_f32(ref_bits[:, :ncol]).astype(np.float64) @ vv
        dev = pv[unit].astype(np.float64)
        d = np.abs(dev - ref)
        n_cmp += int(ref.size)
        n_bad += int((d > tol).sum())
        occ = max(occ, float((d / np.maximum(tol, 1e-300)).max()))
        if d.max() > worst[0]:
            rr, cc = np.unravel_index(int(d.argmax()), d.shape)
            worst = (float(d.max()), (unit, si, n2, int(rr), int(cc), float(ref[rr, cc])))
        if unit == 0:
            worst_u0 = float(d.max())
    print(f"[pv-ref] seq={seq}: 判据 E —— BMM2 L0C（`dbgc2`）vs **独立重算 P̃**·V（T3）"
          f" | tol = S2T·u·Σ|P̃V| + Σ_k(1.0·ulp(P̃_k)+ε_arg·P̃_k)|V_kd|（两项都是推导）"
          f" | 覆盖 {len(units)} 个 unit 的 tile0、{n_cmp} 个元素 | 越界 {n_bad}"
          f" | maxAbsErr {worst[0]:.3e} @ unit{worst[1][0]} (row{worst[1][3]},col{worst[1][4]})"
          f" | 最大界占用 {occ:.2e} | unit0 maxAbsErr {worst_u0:.3e}")
    T.judge("E. P·V（独立 P，全 unit 的 tile0）", n_bad == 0,
            f"越界 {n_bad}/{n_cmp}，maxAbsErr {worst[0]:.3e}，界占用 {occ:.2e}")
    T.report("E. P·V（独立 P）覆盖率/占用", f"{len(units)} unit × {n_cmp} 元素，占用 {occ:.2e}，unit0 {worst_u0:.3e}")


def check_pv(seq: int, T: Tally) -> None:
    """判据 F：**S1**（输入 = 设备产出的 P）—— M27 `m16_pv` 口径的 P·V 对拍。

    `p = gp[unit][0]` 是**设备自己产出的 P̃**（D2H 落盘后 host 读回），拿它当 `einsum("sk,kd->sd", p, v)`
    的输入，再与设备 `dbgc2` 比。**被判量是"给定 P 下 BMM2 的布局/转置"，P 不是本判据的被判量**
    （两侧同一份 P ⇒ 对 P 自身的任何错都无感；P 由判据 C/D 用 host 独立来源咬）。
    本判据另有一条 bite：`gp`（AIV 写的 P）与 AIC 实际装进 L0A 的那份 P 是否同一份 —— 若 AIC 读了
    陈旧/错位的 P，判据 E（用参考 P̃）与 C（只查 gp）都可能过，而本判据会失败。
    容差 = S2T·u·Σ|P̃_dev·V|（mmad k=256 累加的保守界，**推导**；旧写的 5e-3「bf16 网格」是无来源的经验值）。
    """
    if need(seq, ["v", "gp", "dbgc2"], T, "F P·V（设备 P）判据"):
        return
    v = np.fromfile(f"m10_case_s{seq}_v.bin", dtype=np.uint16).reshape(N2, -1, HD)
    gp = bf16_bits_to_f32(np.fromfile(f"m10_case_s{seq}_gp.bin", dtype=np.uint16).reshape(28, 3, M_PAD, S2T))
    pv = np.fromfile(f"m10_case_s{seq}_dbgc2.bin", dtype=np.float32).reshape(28, M_PAD, S2T)
    vf_all = bf16_bits_to_f32(v).astype(np.float64)
    nsplit = len(split_ranges(seq))
    chunk = len(range(split_ranges(seq)[0][0], split_ranges(seq)[0][1], S2T))
    units = [2 * si + n2 for si in range(nsplit) for n2 in range(N2)]
    n_cmp = n_bad = 0
    worst = (0.0, None)
    occ = 0.0
    worst_u0 = 0.0
    for unit in units:
        si, n2 = unit // 2, unit % 2
        base = si * chunk * S2T
        ncol = min(base + S2T, seq) - base
        vv = vf_all[n2, base:base + ncol]
        p = gp[unit, 0][:, :ncol]
        base_abs = np.abs(p) @ np.abs(vv)
        tol = S2T * U * base_abs
        ref = p @ vv
        dev = pv[unit].astype(np.float64)
        d = np.abs(dev - ref)
        n_cmp += int(ref.size)
        n_bad += int((d > tol).sum())
        occ = max(occ, float((d / np.maximum(tol, 1e-300)).max()))
        if d.max() > worst[0]:
            rr, cc = np.unravel_index(int(d.argmax()), d.shape)
            worst = (float(d.max()), (unit, int(rr), int(cc)))
        if unit == 0:
            worst_u0 = float(d.max())
    print(f"[pv-dev] seq={seq}: 判据 F（**S1**：输入 = 设备产出的 P̃）—— BMM2 L0C vs P̃_dev·V"
          f" | tol = S2T·u·Σ|P̃V|（推导）| 覆盖 {len(units)} 个 unit 的 tile0、{n_cmp} 个元素"
          f" | 越界 {n_bad} | maxAbsErr {worst[0]:.3e} @ unit{worst[1][0]}"
          f" | 最大界占用 {occ:.2e} | unit0 maxAbsErr {worst_u0:.3e}")
    print("     定位声明：本判据的**被判量**是「给定 P 下 BMM2 的布局/转置」（咬 BMM2 ↔ `gp` 的同一性）；"
          "P 本身**不是**本判据的被判量（两侧同一份设备 P̃），P 由判据 C/D 咬。")
    T.judge("F. P·V（设备 P，S1 定位声明）", n_bad == 0,
            f"越界 {n_bad}/{n_cmp}，maxAbsErr {worst[0]:.3e}，界占用 {occ:.2e}")
    T.report("F. P·V（设备 P）覆盖率/占用", f"{len(units)} unit × {n_cmp} 元素，占用 {occ:.2e}，unit0 {worst_u0:.3e}")


def run_cases(cases, only_pv=False):
    """跑判据并汇总三态退出码（docs/17 §8.3）。only_pv=True = 只跑 E/F 两条 P·V 判据。"""
    tallies = {}
    for seq in cases:
        T = Tally()
        if only_pv:
            check_pv_ref(seq, T)
            check_pv(seq, T)
        else:
            check(seq, T)
            check_p_full(seq, T)
            check_pv_ref(seq, T)
            check_pv(seq, T)
        tallies[seq] = T
    n_j = sum(T.count("judge") for T in tallies.values())
    n_j_ok = sum(sum(1 for it in T.items if it[0] == "judge" and it[2]) for T in tallies.values())
    n_r = sum(T.count("report") for T in tallies.values())
    n_g = sum(T.count("guard") for T in tallies.values())
    n_g_ok = sum(sum(1 for it in T.items if it[0] == "guard" and it[2]) for T in tallies.values())
    jf = [(s, n) for s, T in tallies.items() for n in T.judge_fails()]
    gf = [(s, n) for s, T in tallies.items() for n in T.guard_fails()]
    sk = [(s, n, w) for s, T in tallies.items() for n, w in T.skips()]
    print(f"[check] seq={','.join(str(c) for c in cases)} | 判定项 {n_j_ok}/{n_j} 通过"
          f" | 报告项 {n_r} 单列 | guard {n_g_ok}/{n_g}"
          f" | 输入缺失 {len(sk)} 项")
    for s, n, w in sk:
        print(f"        缺输入: seq={s} {n} —— {w}")
    for s, n in jf:
        print(f"        判定项 FAIL: seq={s} {n}")
    for s, n in gf:
        print(f"        guard FAIL: seq={s} {n}")
    if jf or gf:
        print(f"RESULT: FAIL (判定项 {n_j_ok}/{n_j} 通过；判定 FAIL {len(jf)} 项、guard FAIL {len(gf)} 项；"
              f"报告项 {n_r} 单列)")
        return 1
    if sk:
        print(f"RESULT: SKIPPED (判定项 {n_j_ok}/{n_j} 通过，但 {len(sk)} 项**没得比** —— "
              f"缺输入不构成合格证；缺项清单见上列 '缺输入' 行)")
        return 2
    print(f"RESULT: OK (判定项 {n_j}/{n_j} 全通过；报告项 {n_r} 单列；guard {n_g}/{n_g}；"
          f"比较了 seq={','.join(str(c) for c in cases)} 的全部判据输入，无抽样)")
    return 0


def main():
    args = sys.argv[1:]
    if args and args[0] == "pv":
        cases = [int(a) for a in args[1:]] or [256]
        sys.exit(run_cases(cases, only_pv=True))
    if args and args[0] == "eps":
        cases = [int(a) for a in args[1:]] or [256, 300, 3584, 3585, 3600, 3840, 4096]
        sys.exit(eps_main(cases))
    cases = [int(a) for a in args] or [256, 300, 4096]
    sys.exit(run_cases(cases))


if __name__ == "__main__":
    main()
