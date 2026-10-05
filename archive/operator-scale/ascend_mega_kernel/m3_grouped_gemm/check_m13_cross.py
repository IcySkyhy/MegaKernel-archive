#!/usr/bin/env python3
"""M54: m3 量化器（device，floor/OCP）与 **m13 自己的参考** 及 tools/golden 的逐字节对拍。

被判对象：`./m3_grouped_gemm/build/m3_grouped_gemm vf-dump <dir>` 落盘的 device 输出
（`pack_dev.bin` / `scale_dev.bin`，由**生产同一份 VF 量化函数**产生），输入 `h.bin`。

参考（同一份输入、逐字节比对）：
  1. `m13_moe_layer/check_ref.py::quant_mxfp4_hw` —— **m13 自己的**独立 numpy 参考
     （逐句对照 m13 的 `VecQuantStage::MxQuantComputeScale` / `MxQuantComputeDataFP4`），
     即「与 m13 逐位对拍」的直接对象。
  2. `tools/golden/moe_block_ref.py::quantize_ocp` —— 官方 ops-nn 序列的逐句转写（M26），
     是与 m13 **独立**写成、且被 m13 设备逐字节见证过的权威实现；作为交叉见证。

负向对照（必须 FAIL，证明本判据非恒真）：`moe_block_ref.quantize_ceil_legacy`
（M26 判定的 legacy ceil 规则 = 收敛前的 m3 规则）——若它与 device 也逐字节一致，说明本用例
区分不出两套规范，判据无意义，脚本以退出码 1 报出。

三态退出码（tower 2026-09-26 规则）：
  0 = 比过且全部判定项逐字节一致（负向对照也如期不一致）
  1 = 比过且有差异（或负向对照没 FAIL）
  2 = 没得比 / 输入缺失（dump 不齐，不发合格证）

覆盖范围（与匹配器同源）：
  · 覆盖 dump 的 rows 行 × K 列（脚本从 `meta.txt` 读，不硬编码）；
  · 判定项 = 每行 pack(K/2 字节) + scale(K/32 字节) 的**逐字节**比较，计数由同一次 load 得出；
  · dump 的 scale 只落**有效 group 字节/行**（`scale_row_stride == k/group`）：device 行缓冲是
    32 B（20 有效），尾部 12 B padding 是陈旧 UB、**不确定**，故不落盘也不参与比较（README §7
    已登记的布局性质）——这正是 `scale_dev.bin` 的 sha256 可复现的原因。
  · **已知会被漏掉**：dump 输入全部是 bf16 网格值（官方量化器输入域）；m3 的 h 在生产里是
    fp32（silu 与 up 相乘在 fp32），**非 bf16 网格的 fp32 h 不在本对拍覆盖内**——那正是
    README 记录的「唯一遗留差异（h 精度）」，由 §5.3 的报告项量化，不由本脚本判定。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "m13_moe_layer"))
sys.path.insert(0, os.path.join(ROOT, "tools", "golden"))

try:
    import numpy as np
    import check_ref  # m13_moe_layer/check_ref.py
    from moe_block_ref import quantize_ocp, quantize_ceil_legacy
except Exception as exc:  # noqa: BLE001
    print(f"[M54-cross] SKIPPED: 依赖/参考模块不可导入（{exc}）；需要 numpy")
    sys.exit(2)


def read_meta(d):
    meta = {}
    with open(os.path.join(d, "meta.txt")) as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = int(v)
    return meta


def load(d):
    meta = read_meta(d)
    rows, k, group = meta["rows"], meta["k"], meta["group"]
    h = np.fromfile(os.path.join(d, "h.bin"), dtype=np.float32).reshape(rows, k)
    pack = np.fromfile(os.path.join(d, "pack_dev.bin"), dtype=np.uint8).reshape(rows, k // 2)
    scal = np.fromfile(os.path.join(d, "scale_dev.bin"), dtype=np.uint8).reshape(rows, meta["scale_row_stride"])
    scal = scal[:, : k // group]
    return rows, k, group, h, pack, scal


def cmp_bytes(tag, dev, ref, results):
    if dev.shape != ref.shape:
        results.append((tag, False, f"shape 不同 {dev.shape} vs {ref.shape}"))
        return False
    diff = int(np.count_nonzero(dev != ref))
    results.append((tag, diff == 0, f"{diff}/{dev.size} 字节不符"))
    return diff == 0


def main(argv):
    d = argv[1] if len(argv) >= 2 else os.path.join(HERE, "evidence", "quant_cross")
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "h.bin")):
        print(f"[M54-cross] SKIPPED: dump 目录不齐（{d}）；先跑 "
              f"`./m3_grouped_gemm/build/m3_grouped_gemm vf-dump {d}`")
        return 2

    rows, k, group, h, pack_dev, scal_dev = load(d)
    print(f"[M54-cross] dump={d} rows={rows} k={k} group={group} 覆盖 "
          f"{rows * (k // 2)} pack 字节 + {rows * (k // group)} scale 字节")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pack_m13, scal_m13 = check_ref.quant_mxfp4_hw(h)
        pack_ocp, scal_ocp = quantize_ocp(h)
        pack_ceil, scal_ceil = quantize_ceil_legacy(h)

    results = []
    ok_m13 = cmp_bytes("pack vs m13 quant_mxfp4_hw", pack_dev, pack_m13, results)
    ok_m13 &= cmp_bytes("scale vs m13 quant_mxfp4_hw", scal_dev, scal_m13, results)
    ok_ocp = cmp_bytes("pack vs tools/golden quantize_ocp", pack_dev, pack_ocp, results)
    ok_ocp &= cmp_bytes("scale vs tools/golden quantize_ocp", scal_dev, scal_ocp, results)
    # 负向对照：legacy ceil 规则必须与 device 不一致（count != 0 -> ref_ok False）
    ceil_pack_diff = int(np.count_nonzero(pack_dev != pack_ceil))
    ceil_scal_diff = int(np.count_nonzero(scal_dev != scal_ceil))
    neg_ok = (ceil_pack_diff > 0 or ceil_scal_diff > 0)
    results.append(("负向对照 vs legacy ceil（必须不一致）", neg_ok,
                    f"pack {ceil_pack_diff}/{pack_dev.size}、scale {ceil_scal_diff}/{scal_dev.size} 字节不同"))

    for tag, passed, detail in results:
        print(f"[M54-cross] {tag:44s}: {'PASS' if passed else 'FAIL'} ({detail})")

    all_ok = ok_m13 and ok_ocp and neg_ok
    print(f"[M54-cross] ===== {sum(1 for _, p, _ in results if p)}/{len(results)} 判定项 PASS "
          f"(m13 逐位对拍 {'PASS' if ok_m13 else 'FAIL'}；独立见证 {'PASS' if ok_ocp else 'FAIL'}；"
          f"负向对照 {'OK' if neg_ok else 'BAD'}) =====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
