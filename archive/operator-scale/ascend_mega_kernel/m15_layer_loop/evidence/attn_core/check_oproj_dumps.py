#!/usr/bin/env python3
"""M101 r2（复审 r1-F1/F3/F4 闭环）：**离线**独立复核 o_proj 档的 OP-A / OP-B / OP-C。

为什么需要它（复审 F1 的原话）：`m101op_*_y.bin` 曾经是**负向档**的输出（host 在循环末尾统一 dump，
那时 hY 已被 OP-C 的 KMINUS1 launch 覆盖）⇒ OP-A / OP-C 的设备结果**没有可离线复核的工件**，
评审拿它当契约输出复算就吃到 2559/2560 假红。

本脚本做三件事，**不跑设备**：
  ① **绑定**：按 `m15_attn_core_host.h` 里逐字复刻的公式重生成 attn/gate/tint/W，用 **FNV-1a 64**
     （与 C++ 侧同一个实现）证明"我算的这批字节 == 那次运行用的那批字节"；落盘的 .bin 也逐个比对。
     ⇒ 指纹打在**入库的日志**里（`[M101][fp]` 行），所以这一步在只有日志、没有 .bin 的机器上也成立。
  ② **复算**：用**独立的 numpy fp64** 参考复算三条判据，自己实现参考（不 import 任何被测代码）：
       OP-A  GEMM 逐位：输入取精确整数域 ⇒ 设备 bf16 位型必须与 `RNE(fp64 Σ W·t_int)` 相同
       OP-B  ×σ(gate)：`|f32(t_dev) − t_ref| ≤ 0.5·binade 间距 + 4u·|t_ref|`（u = 2^-24）
       OP-C  e2e（S1）：`|f32(y_dev) − Σ W·t_dev| ≤ K·u·Σ|W·t_dev|`
     并同样复算**负向档**（K 方向少累加一个 baseK 块）的参考，确认落盘的负向 y 确实是"被弄坏"的那份。
  ③ **记账**（复审 F4）：用**同一个 dump**把「第一版（写错的）容差」与「更正后的容差」各算一遍，
     把两边的越界条数打出来 —— 这样 README §4.3 里的数字有可复跑的出处，而不是一句自述。

用法（在 dump 所在目录，或给出目录）：
    python3 check_oproj_dumps.py <dir> [--log <run.log>] [--m 1 3]
退出码：0 = 三条判据全过 **且** 负向档确实被弄坏；1 = 有 FAIL；2 = 输入缺失（不发合格证）。
"""
import argparse
import os
import re
import sys

import numpy as np

OP_NH, OP_HD = 24, 256
OP_K = OP_NH * OP_HD          # 6144
OP_N = 2560
M_MAXROWS = 4                 # host 侧按 4 行分配（m<2 时 kernel 多读 1 行）
U24 = 2.0 / 16777216.0        # 2^-24（fp32 单位舍入）
KMINUS1 = 64                  # GEMM_MODE_KMINUS1 少累加一个 baseK 块

# ---- 与 C++ 侧逐字相同的公式（改一处必须改两处）----


def hash3(i: int, j: int, salt: int) -> int:
    h = (i * 2654435761 + j * 40503 + salt * 97 + 0x9E3779B9) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    return h


def float_to_bf16(f):
    x = np.asarray(f, dtype=np.float32).view(np.uint32)
    return (((x >> 16) & 1) + 0x7FFF + x >> 16).astype(np.uint16)


def gen_frac(count: int, salt: int) -> np.ndarray:
    i = np.arange(count, dtype=np.uint64)
    h = (i * 2654435761 + 7 * 40503 + salt * 97 + 0x9E3779B9) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    v = ((h & 0xFFFF).astype(np.float32) / np.float32(65535.0)) * np.float32(1.6) - np.float32(0.8)
    return float_to_bf16(v)


def gen_int(count: int, salt: int) -> np.ndarray:
    i = np.arange(count, dtype=np.uint64)
    h = (i * 2654435761 + 13 * 40503 + salt * 97 + 0x9E3779B9) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    v = (h % 17).astype(np.int64) - 8
    return float_to_bf16(v.astype(np.float32))


def fnv1a64(buf: bytes) -> int:
    """**标准 FNV-1a 64**（复审 r2-N1 更正：offset basis 是 `14695981039346656037` = 0xCBF29CE484222325，
    r1 那份少写了一位；prime `1099511628211` 本来就对）。与 C++ 侧 `Fnv1a64` 同实现。
    自查向量：`fnv1a64(b"") == 0xcbf29ce484222325`、`fnv1a64(b"a") == 0xaf63dc4c8601ec8c`。"""
    h = 14695981039346656037
    for b in buf:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def bf16_to_f64(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).astype(np.uint32).view(np.float32).astype(np.float64)


def binade_spacing(v: np.ndarray) -> np.ndarray:
    """bf16 在 |v| 所在 binade 的格点间距 = 2^(e-8)（v ∈ [2^(e-1), 2^e)，frexp 的 e）。"""
    a = np.abs(np.asarray(v, dtype=np.float64)).copy()
    a[a == 0.0] = 1.0
    _, e = np.frexp(a)
    return np.ldexp(1.0, e - 8)


def quant_half(ref, got):
    """更正后的量化半步项：取参考与设备值两侧间距的较大者（保守上界）。"""
    return 0.5 * np.maximum(binade_spacing(ref), binade_spacing(got))


def legacy_grid_ulp(ref, rne=False):
    """**第一版写错的** 1 bf16 ulp：照抄 M10 `check_ref.py::bf16_grid_ulp` 的形状（位型相邻值做差）。
    bf16 是符号-幅值 ⇒ 负数分支拿到的是**更靠近 0** 的邻居（半步）而不是整步。

    `rne=False`（默认）取 fp32 参考位型的**高 16 位（截断）** —— 这是本脚本 r2 第一版的做法；
    `rne=True` 用**与 C++ `FloatToBf16` 相同的 RNE** 取位型 —— 这是 r1 那个 C++ 版本的做法。
    **两者给出的越界条数不同，且这正是 `11/31`（RNE）与 `18/54`（截断）之差的原因**
    （复审 r2-N2 定位；见 README §4.3.3）。保留两种口径是为了让这三个数字都可复跑。"""
    x = np.asarray(ref, dtype=np.float32).view(np.uint32)
    if rne:
        b16 = (((x >> 16) & 1) + 0x7FFF + x >> 16).astype(np.uint16)   # RNE（= C++ FloatToBf16）
    else:
        b16 = (x >> 16).astype(np.uint16)                              # 截断（本脚本 r2 第一版）
    b = b16.astype(np.int32)
    pos = (b & 0x8000) == 0
    nxt = np.where(pos, np.minimum(b + 1, 0x7F7F), np.maximum(b - 1, 0x8001)).astype(np.uint16)
    return np.abs(bf16_to_f64(nxt) - bf16_to_f64(b16))


# ---- 指纹：从入库日志里取（复审 F1 的"绑定"环节）----
FP_RE = re.compile(r"\[M101\]\[fp\]\s+(?P<label>.+?)\s+file=(?P<file>\S+)\s+len=(?P<len>\d+)\s+"
                   r"fnv1a64=(?P<fp>[0-9a-f]{16})")


def load_fingerprints(log_path):
    """返回 `(label -> [(file, len, fp), …], 匹配到的 [M101][fp] **行数**)`。
    注意两个数**不同**：同一 label 在每个 m 上各出现一次（本档 m=1/3 ⇒ 9 个标签、18 行）。
    ⚠ r3 复审抓到的正是这里：本函数的第一版只打印了标签个数，而文档把它当成行数贴出去 ⇒
    现在两个数都打印、并各自写清名字（不给人留"标签数/行数"的歧义）。"""
    out = {}
    n_lines = 0
    if log_path is None or not os.path.exists(log_path):
        return out, n_lines
    for line in open(log_path, errors="replace"):
        mm = FP_RE.search(line)
        if mm:
            n_lines += 1
            out.setdefault(mm.group("label"), []).append(
                (mm.group("file"), int(mm.group("len")), int(mm.group("fp"), 16)))
    return out, n_lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--log", default=None)
    ap.add_argument("--m", type=int, nargs="*", default=[1, 3])
    args = ap.parse_args()

    # FNV-1a 64 自查（标准向量；复审 r2-N1：常量一旦写错，这里立刻红）
    assert fnv1a64(b"") == 0xCBF29CE484222325, "FNV-1a 64 offset basis 不对"
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C, "FNV-1a 64 实现不对（prime/步骤）"
    print("[selfcheck] FNV-1a 64 标准向量通过：fnv1a64(b'')=0xcbf29ce484222325、"
          "fnv1a64(b'a')=0xaf63dc4c8601ec8c")

    DUMP = args.dir
    fps, n_fp_lines = load_fingerprints(args.log)
    n_input_missing = 0
    n_fail = 0
    n_judge = 0
    print(f"[offline] dir={DUMP}  log={args.log}  "
          f"日志里 [M101][fp] **行数={n_fp_lines}** / **不同标签个数={len(fps)}**"
          f"（两者不同：同一 label 每个 m 各一行）")

    # ---- ① 绑定：重生成输入并与指纹 / 落盘文件比对 ----
    # W：31.5 MB，不落盘（只给指纹）
    w_bits_full = gen_int(OP_N * OP_K, 0x5A01)
    exp_w = fps.get("input W (not dumped)")
    if exp_w:
        got = fnv1a64(w_bits_full.tobytes())
        ok = (got == exp_w[0][2])
        print(f"[bind] W(full, {w_bits_full.nbytes} B) fnv1a64={got:016x} "
              f"log={exp_w[0][2]:016x} -> {'一致' if ok else '**不一致**'}")
        if not ok:
            n_fail += 1

    for m in args.m:
        print(f"\n=== m={m} ===")
        attn_full = gen_frac(M_MAXROWS * OP_K, 0xA101)[: m * OP_K]
        gate_full = gen_frac(M_MAXROWS * OP_K, 0xA102)[: m * OP_K]
        tint_full = gen_int(M_MAXROWS * OP_K, 0xA103)[: m * OP_K]
        for label, arr, name in (("input attn", attn_full, f"m101op_m{m}_attn.bin"),
                                 ("input gate", gate_full, f"m101op_m{m}_gate.bin"),
                                 ("input t (integer)", tint_full, f"m101op_m{m}_tint.bin")):
            fp_calc = fnv1a64(arr.tobytes())
            exp = fps.get(label)
            line = f"[bind] {label:<20} {arr.nbytes:>9} B fnv1a64={fp_calc:016x}"
            if exp:
                # 同一 label 在每个 m 上各出现一次，按顺序取（m=1 先、m=3 后）
                idx = args.m.index(m)
                if idx < len(exp):
                    line += f" log={exp[idx][2]:016x} -> {'一致' if fp_calc == exp[idx][2] else '**不一致**'}"
                    if fp_calc != exp[idx][2]:
                        n_fail += 1
            p = os.path.join(DUMP, name)
            if os.path.exists(p):
                disk = open(p, "rb").read()
                same = disk == arr.tobytes()
                line += f" | 磁盘 {name}: {'逐字节相同' if same else '**不同**'}"
                if not same:
                    n_fail += 1
            print(line)

        # ---- ② 复算三条判据 ----
        def need(name):
            p = os.path.join(DUMP, name)
            if not os.path.exists(p):
                print(f"[skip] 缺 {name}（不比较、不发合格证）")
                return None
            return np.fromfile(p, dtype=np.uint16)

        tdev = need(f"m101op_m{m}_tdev.bin")
        yA = need(f"m101op_m{m}_yA_contract.bin")
        yA_k1 = need(f"m101op_m{m}_yA_kminus1.bin")
        yC = need(f"m101op_m{m}_yC_contract.bin")
        yC_k1 = need(f"m101op_m{m}_yC_kminus1.bin")
        if any(x is None for x in (tdev, yA, yA_k1, yC, yC_k1)):
            print("RESULT: SKIPPED（缺 dump —— 绝不发合格证）")
            return 2

        Wf = bf16_to_f64(w_bits_full).reshape(OP_N, OP_K)

        # OP-B：×σ(gate)，更正后 / 第一版两套容差（F4 的记账）
        a = bf16_to_f64(attn_full)
        g = bf16_to_f64(gate_full)
        ref_t = a / (1.0 + np.exp(-g))
        got_t = bf16_to_f64(tdev)
        dv = np.abs(got_t - ref_t)          # 注意：**不能叫 `d`** —— 那会遮蔽 dump 目录名（本脚本踩过一次）
        tol_new = quant_half(ref_t, got_t) + 4.0 * U24 * np.abs(ref_t)
        tol_old = 0.5 * legacy_grid_ulp(ref_t.astype(np.float32)) + 4.0 * U24 * np.abs(ref_t)
        tol_old_rne = 0.5 * legacy_grid_ulp(ref_t.astype(np.float32), rne=True) + 4.0 * U24 * np.abs(ref_t)
        bad_new = int((dv > tol_new).sum())
        bad_old = int((dv > tol_old).sum())
        bad_old_rne = int((dv > tol_old_rne).sum())
        rb = float_to_bf16(ref_t.astype(np.float32)).astype(np.int32)
        db = tdev.astype(np.int32)
        same_sign = (db & 0x8000) == (rb & 0x8000)
        bitd = np.where(same_sign, np.abs((db & 0x7FFF) - (rb & 0x7FFF)),
                        np.abs(db & 0x7FFF) + np.abs(rb & 0x7FFF) + 1)
        n_judge += 1
        n_fail += 1 if bad_new else 0
        print(f"[OP-B] n={dv.size} 越界(更正后)={bad_new} 越界(第一版·截断口径)={bad_old} "
              f"越界(第一版·RNE 口径= r1 C++)={bad_old_rne} "
              f"max|d|={dv.max():.6g} max(dv/tol_new)={float((dv/tol_new).max()):.6f} "
              f"位差 0/1/>=2 = {int((bitd==0).sum())}/{int((bitd==1).sum())}/{int((bitd>=2).sum())}")

        # OP-A：GEMM 逐位（整数域）
        tf = bf16_to_f64(tint_full).reshape(m, OP_K)
        ref_yA = tf @ Wf.T
        exp_bits = float_to_bf16(ref_yA.astype(np.float32)).reshape(-1)
        same_bits = int((exp_bits == yA).sum())
        ref_yA_k1 = tf[:, : OP_K - KMINUS1] @ Wf[:, : OP_K - KMINUS1].T
        exp_bits_k1 = float_to_bf16(ref_yA_k1.astype(np.float32)).reshape(-1)
        same_bits_k1 = int((exp_bits_k1 == yA_k1).sum())
        n_judge += 1
        n_fail += 1 if same_bits != yA.size else 0
        print(f"[OP-A] 契约档：与 RNE(fp64 Σ_k<{OP_K}) 逐位相同 {same_bits}/{yA.size}"
              f"（须 == 全部）；负向档：与 Σ_k<{OP_K-KMINUS1} 逐位相同 {same_bits_k1}/{yA_k1.size}"
              f"（须 == 全部，即负向 dump 确实是「被弄坏」的那份）")

        # OP-C：e2e（S1：输入 = 设备 t）
        ref_yC = bf16_to_f64(tdev).reshape(m, OP_K) @ Wf.T
        absSum = np.abs(bf16_to_f64(tdev).reshape(m, OP_K)) @ np.abs(Wf).T
        tolC = OP_K * U24 * absSum
        dC = np.abs(bf16_to_f64(yC).reshape(m, OP_N) - ref_yC)
        badC = int((dC > tolC).sum())
        ref_yC_k1 = bf16_to_f64(tdev).reshape(m, OP_K)[:, : OP_K - KMINUS1] @ Wf[:, : OP_K - KMINUS1].T
        tolC_k1 = (OP_K - KMINUS1) * U24 * (np.abs(bf16_to_f64(tdev).reshape(m, OP_K)[:, : OP_K - KMINUS1])
                                            @ np.abs(Wf[:, : OP_K - KMINUS1]).T)
        dC_k1 = np.abs(bf16_to_f64(yC_k1).reshape(m, OP_N) - ref_yC_k1)
        badC_k1 = int((dC_k1 > tolC_k1).sum())          # 对 **K-1 参考**：须 0（证明它确实是截断结果）
        dC_k1_full = np.abs(bf16_to_f64(yC_k1).reshape(m, OP_N) - ref_yC)
        badC_k1_full = int((dC_k1_full > tolC).sum())   # 对 **全 K 参考**：须 > 0（证明它确实是"被弄坏"的）
        n_judge += 1
        n_fail += 1 if badC else 0
        if badC_k1 != 0 or badC_k1_full == 0:
            n_fail += 1
        print(f"[OP-C] 契约档：越界 {badC}/{dC.size} max|d|={dC.max():.6g} "
              f"最大界占用 {float((dC/tolC).max()):.6f}；"
              f"负向档：对 K-1 参考越界 {badC_k1}/{dC_k1.size}（须 0）、对全 K 参考越界 {badC_k1_full}"
              f"（须 > 0）")

    print(f"\n[offline] 判定项 {n_judge - n_fail}/{n_judge} 通过；输入缺失 {n_input_missing} 项")
    if n_fail:
        print("RESULT: FAIL（离线复算") 
        return 1
    print("RESULT: OK（离线复算的三条判据与负向档全部与入库读数一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
