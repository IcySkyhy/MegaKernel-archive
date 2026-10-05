#!/usr/bin/env python3.12
"""pick_observable_seed.py —— 为「只在部分输入上可观测」的分歧选一个可观测的数据种子

**为什么需要它**：`PLE_SPEC.md §4.1 ④` 钉的是 `dot = fp32(bf16(Σ …))`（上游 `ops/ple.py:218-219`
的 `tl.sum(...).to(dtype).to(tl.float32)`）。若实现漏掉这次 bf16 物化，**它是否改变输出取决于
`d = dot/√2560` 是否跨过 bf16 格点** —— 在随机数据上约 2%/gate 项（M85 r1 review 的独立测算）。
也就是说：这条分歧的判据**在实测数据上可能咬不住**，而"判据咬不住"不能靠声明掩盖，只能靠**选一个
可观测的数据**。本脚本就是那个"选择"的过程，选择结果写进 `gen_ple_data.py --hidden-seed`。

**输入**：`ple/data/B_kv.bin`（**设备产物**，即 ④ 的真实输入 key/value）、`ple/data/table_red.bin`
不用。q 只由 `hidden` 决定，而 `kv` 与 `hidden` 无关（kv = wcat·emb；emb 只由表 + ids 决定）
⇒ 可以用设备 kv + 候选 hidden 在脚本里穷举，**不需要反复跑设备**。

**判据**（可执行）：对每个候选种子 s，用 `gen_ple_data.py` 同一分布抽 hidden，
逐 (t, stream) 比较「SPEC 版 gate 标量 g」与「漏掉 dot 和 bf16 物化的 g」，
输出**第一个 g 不相同**的种子（≥1 个 gate 项不同即可让 B4.gated 判据产生差异）。

用法：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/pick_observable_seed.py [--out plE/out] [--max 400]
"""
import argparse
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

HID = 2560
HC = 4
W = HID * HC
HE = 2560
KVW = W + HID
EPS = 1e-6
SQ = math.sqrt(2560.0)


def bf16_rne(x):
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    out = (rounded & np.uint64(0xFFFF0000)).astype(np.uint32)
    return out.view(np.float32)


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def gate_scalar(key, hidden_slice, nk, nq, sum_bf16):
    k = key.astype(np.float64)
    q = hidden_slice.astype(np.float64)
    kn = bf16_rne(k * (1.0 / np.sqrt((k * k).mean() + EPS)) * (1.0 + nk.astype(np.float64)))
    qn = bf16_rne(q * (1.0 / np.sqrt((q * q).mean() + EPS)) * (1.0 + nq.astype(np.float64)))
    prod = bf16_rne(kn.astype(np.float64) * qn.astype(np.float64)).astype(np.float32)
    tot = float(np.float32(prod.astype(np.float64).sum(dtype=np.float64)))
    if sum_bf16:
        tot = float(bf16_rne(np.array([np.float32(tot)], dtype=np.float32))[0])
    d = float(bf16_rne(np.array([np.float32(tot / SQ)], dtype=np.float32))[0])
    sign = -1.0 if d < 0 else (1.0 if d > 0 else 0.0)
    mag = float(bf16_rne(np.array([np.sqrt(max(abs(d), 1e-6))], dtype=np.float32))[0])
    return float(bf16_rne(np.array([1.0 / (1.0 + math.exp(-sign * mag))], dtype=np.float32))[0]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--max", type=int, default=400)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    data = pathlib.Path(args.data)
    kv = bf16_to_f32((out / "B_kv.bin").read_bytes()).reshape(-1, KVW)
    nk = bf16_to_f32((data / "w_norm_key.bin").read_bytes())
    nq = bf16_to_f32((data / "w_norm_query.bin").read_bytes())
    T = kv.shape[0]
    print("[pick] 设备 kv: %s；gate 项数 T*HC = %d" % (kv.shape, T * HC))

    hits = []
    for s in range(args.max):
        rng = np.random.default_rng(s)
        hid = bf16_rne(rng.normal(0.0, 0.5, size=(T, W)).astype(np.float32))
        for t in range(T):
            for st in range(HC):
                sl = slice(st * HID, (st + 1) * HID)
                g1, d1 = gate_scalar(kv[t, sl], hid[t, sl], nk[sl], nq[sl], True)
                g0, d0 = gate_scalar(kv[t, sl], hid[t, sl], nk[sl], nq[sl], False)
                if g1 != g0:
                    hits.append((s, t, st, d1, d0, g1, g0))
        if hits:
            break
    if not hits:
        print("[pick] %d 个种子内未找到可观测样本（判据将无法咬住该分歧）" % args.max)
        return 1
    s = hits[0][0]
    print("[pick] 可观测种子 = %d，%d/%d 个 gate 项的 g 不同：" % (s, len(hits), T * HC))
    for h in hits[:8]:
        print("       (seed=%d t=%d s=%d) d: spec=%.9g vs no-round=%.9g | g: spec=%.9g vs no-round=%.9g"
              % h)
    print("[pick] 用法：gen_ple_data.py --hidden-seed %d" % s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
