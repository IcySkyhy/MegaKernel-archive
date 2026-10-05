# SPDX-License-Identifier: Apache-2.0
"""Cross-check for m8_permute: NPU device outputs vs the numpy permute/unpermute
reference (same formulas as RefUnpermute in m8_permute.asc).

numpy reference (golden bins from tools/golden/data/<case>):
    permute   : x_sorted[i, :] = x[perm_src_token[i], :]
                （校验 device 输出 vs golden x_sorted.bin，逐位一致）
    unpermute : out[t, :] = bf16_RNE( Σ_k fp32(bf16 w[t,k]) * fp32(y[inv[t,k], :]) )
                w[t,k]   = bf16_RNE(topk_weights[t,k])          （RNE）
                inv[t,k] = token t 的第 k 个 topk 专家在 sorted 序列中的槽位
                           （由 perm_src_token/perm_expert/topk_ids 重建）
                变体 A：y = golden x_sorted；变体 B：y = <case>_y_synth.bin（host 合成）

Usage（在 M8_DUMP=1 落盘输出的目录运行）：
    /usr/local/python3.12.13/bin/python3 check_ref.py m1 [dump_dir]
    /usr/local/python3.12.13/bin/python3 check_ref.py m33 [dump_dir]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden"))
from moe_block_ref import bf16_bits_to_f32, f32_to_bf16_bits  # pure-numpy bf16 helpers (RNE)

HIDDEN = 2560
WROW = 16  # 与 m8_permute.asc WROW_I32 一致（权重行打包行距）


def load_case(data_dir):
    x = np.fromfile(f"{data_dir}/x.bin", dtype=np.uint16)
    m = x.size // HIDDEN
    pst = np.fromfile(f"{data_dir}/perm_src_token.bin", dtype=np.int32)
    pe = np.fromfile(f"{data_dir}/perm_expert.bin", dtype=np.int32)
    tids = np.fromfile(f"{data_dir}/topk_ids.bin", dtype=np.int32).reshape(m, -1)
    tw = np.fromfile(f"{data_dir}/topk_weights.bin", dtype=np.float32).reshape(m, -1)
    x_sorted = np.fromfile(f"{data_dir}/x_sorted.bin", dtype=np.uint16)
    total = pst.size
    x = x.reshape(m, HIDDEN)
    x_sorted = x_sorted.reshape(total, HIDDEN)
    return m, pst, pe, tids, tw, x, x_sorted


def build_routing(pst, pe, tids, tw, m, topk):
    """与 host BuildRouting 同款：inv_slot [m,topk] + bf16 权重位 [m,topk]。"""
    inv = np.zeros((m, topk), dtype=np.int64)
    w_bits = np.zeros((m, topk), dtype=np.uint16)
    tw_bf = f32_to_bf16_bits(tw)
    for i in range(pst.size):
        t, e = int(pst[i]), int(pe[i])
        j = int(np.nonzero(tids[t] == e)[0][0])
        inv[t, j] = i
        w_bits[t, j] = tw_bf[t, j]
    return inv, w_bits


def unpermute_ref(y_bits, inv, w_bits, m, topk):
    """out[t] = bf16_RNE( Σ_k fp32(w_k) * fp32(y[inv[t,k]]) )，fp32 逐运算舍入。"""
    y = bf16_bits_to_f32(y_bits).astype(np.float32)
    w = bf16_bits_to_f32(w_bits).astype(np.float32)
    acc = np.zeros((m, HIDDEN), dtype=np.float32)
    for k in range(topk):
        yk = y[inv[:, k]].astype(np.float32)
        acc = (acc + w[:, k][:, None] * yk).astype(np.float32)
    return f32_to_bf16_bits(acc)


def bitexact(name, got, exp):
    ok = got.shape == exp.shape and bool(np.array_equal(got, exp))
    print(f"{name}: bit-exact {ok} (shape {got.shape})")
    if not ok and got.shape == exp.shape:
        bad = np.argwhere(got != exp)
        i, j = bad[0]
        print(f"  first mismatch [{i}][{j}]: got 0x{got[i, j]:04x} expect 0x{exp[i, j]:04x}, "
              f"total {len(bad)}")
    return ok


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "m1"
    dump_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "golden", "data", case)

    m, pst, pe, tids, tw, x, x_sorted = load_case(data_dir)
    topk = tids.shape[1]
    inv, w_bits = build_routing(pst, pe, tids, tw, m, topk)

    ok = True
    perm_dev = os.path.join(dump_dir, f"{case}_perm_device.bin")
    if os.path.exists(perm_dev):
        got = np.fromfile(perm_dev, dtype=np.uint16).reshape(-1, HIDDEN)
        # numpy 参考即 golden 语义：x_sorted = x[pst]
        ref = x[pst]
        ok &= bitexact(f"[{case}] permute: NPU vs numpy(x[pst])", got, ref)
        ok &= bitexact(f"[{case}] permute: golden x_sorted vs numpy(x[pst])", x_sorted, ref)
    else:
        print(f"skip permute check ({perm_dev} not present; run with M8_DUMP=1 first)")

    for variant, y_path in (("goldy", None), ("synth", os.path.join(dump_dir, f"{case}_y_synth.bin"))):
        dev_path = os.path.join(dump_dir, f"{case}_unperm_{variant}_device.bin")
        if not os.path.exists(dev_path):
            print(f"skip unpermute-{variant} check ({dev_path} not present)")
            continue
        if y_path is None:
            y_bits = x_sorted
        elif os.path.exists(y_path):
            y_bits = np.fromfile(y_path, dtype=np.uint16).reshape(-1, HIDDEN)
        else:
            print(f"skip unpermute-{variant} check ({y_path} not present)")
            continue
        got = np.fromfile(dev_path, dtype=np.uint16).reshape(m, HIDDEN)
        ref = unpermute_ref(y_bits, inv, w_bits, m, topk)
        ok &= bitexact(f"[{case}] unpermute-{variant}: NPU vs numpy(ref)", got, ref)

    print(f"[{case}] ===== {'PASS' if ok else 'FAIL'} =====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
