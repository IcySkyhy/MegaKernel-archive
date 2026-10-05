#!/usr/bin/env python3.12
"""m29_common.py —— M161 PLE 单元的共享参考件（数据生成 + 判据都用它）

参考函数的来源与独立性（写准）：`ref_ids` / `ref_gate` / `ref_conv_out` 是
`m15_layer_loop/m15_ple_check.py` 对应函数的**同源转录（correlated transcription）** —— 逐字相同，
仅注释/docstring 有差；规则本身钉在仓外 `/workspace/vllm/vllm/models/qwen4_exp/nvidia/ops/ple.py`
的三个 kernel 符号上（① `_ple_ngram_ids_kernel:25-113`、④ `_ple_gate_kernel:185-281`、
⑤ `_ple_conv_kernel:319-485`）与 `nvidia/ple_layer.py:394-429` 的 [key;value] 切分。
**风险**：同一处转写错误会两边一起错。覆盖它的独立证据见 README §5（真实档 `emb` 直读分片、
`docs/17` §1.2 的非空洞见证、出处符号级对照）。checkpoint 读取与真实分片定位抄改自
`m15_layer_loop/ple/{gen_ple_data.py,real_table_probe.py}`。

分档（docs/17 §1.1）：
  · ids[T,16] int64                              → T1 逐位
  · emb[T,2560]（表行 gather）                    → T1 逐字节
  · kv[T,12800]（K=2560 mmad 累加）               → T3（ε_MMAD = 2560·2^-24；界用 Σ|terms|）
  · gated/normed/out/state（Rsqrt/Sigmoid + 归约）→ T3（ε_RSQRT+ε_SIGMOID；界用 max 代理）
"""
import json
import math
import pathlib
import struct

import numpy as np

CKPT = pathlib.Path("/workspace/Qwen3.8-Flash-Next-MXFP4")
PLE_PREFIX = "model.language_model.layers.1.ple."

HID = 2560
HC = 4
W = HID * HC
HE = 2560
P = 8
NGR = 3
NG = (NGR - 1) * P
HDIM = HE // NG
NC = 2
KVW = HE + W
KCONV = 4
DIL = NGR
STLEN = (KCONV - 1) * DIL
EPS = 1e-6
EOS = 248044
SQ2560 = math.sqrt(2560.0)

# 真实 ngram 表几何（REAL_TABLE.md §2.2）
NSHARDS = 128
ROWS_PER_SHARD = 2500012
ROW_BYTES = HDIM * 2
TOTAL_ROWS = NSHARDS * ROWS_PER_SHARD

# ---- T3 的 ε 推导（docs/17 §1.1；与 m15_ple_check.py 同值）----
EPS_MMAD = 2560.0 * 2.0 ** -24
EPS_REDUCE = 96.0 * 2.0 ** -24
EPS_RSQRT = 4.0 * 2.0 ** -24
EPS_SIGMOID = 2.0 * 2.0 ** -24

SMALL_TENSORS = {
    "key_proj": PLE_PREFIX + "key_proj.weight",
    "value_proj": PLE_PREFIX + "value_proj.weight",
    "conv1d": PLE_PREFIX + "conv1d.weight",
    "norm_key": PLE_PREFIX + "norm_key.weight",
    "norm_query": PLE_PREFIX + "norm_query.weight",
    "norm_conv": PLE_PREFIX + "norm_conv.weight",
    "layer_multipliers": PLE_PREFIX + "ple_embedding.layer_multipliers",
    "ngram_heads_vocab_sizes": PLE_PREFIX + "ple_embedding.ngram_heads_vocab_sizes",
    "ngram_heads_offsets": PLE_PREFIX + "ple_embedding.ngram_heads_offsets",
}

# 合成缩减表的 16 个小素数（与 m15 同）
RED_SIZES = [131, 127, 113, 109, 107, 103, 101, 97, 89, 83, 79, 73, 71, 67, 61, 59]


# ------------------------------------------------------------------ bf16
def bf16_rne(x):
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    out = (rounded & np.uint64(0xFFFF0000)).astype(np.uint32)
    return out.view(np.float32)


def bf16_bytes(x):
    return bf16_rne(x).view(np.uint32).astype(np.uint32).ravel()


def pack_bf16(x):
    """float32 → bf16 字节（取高 16 位；低 16 位已清零）。"""
    return (bf16_rne(x).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def bf_ulp(v):
    a = np.abs(np.asarray(v, dtype=np.float64))
    a = np.where(a == 0.0, np.finfo(np.float32).tiny, a)
    e = np.floor(np.log2(a))
    return np.power(2.0, e - 7.0)


def i64(x):
    x = int(x) & ((1 << 64) - 1)
    return x - (1 << 64) if x >= (1 << 63) else x


# ------------------------------------------------------------------ checkpoint 读取
def st_headers():
    hdrs = {}
    for f in sorted(CKPT.glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        hdrs[f] = (8 + n, hdr)
    return hdrs


def read_tensor(hdrs, name):
    for f, (base, hdr) in hdrs.items():
        if name in hdr:
            v = hdr[name]
            o0, o1 = v["data_offsets"]
            with open(f, "rb") as fh:
                fh.seek(base + o0)
                raw = fh.read(o1 - o0)
            return v["dtype"], tuple(v["shape"]), raw
    raise KeyError(name)


def shard_map():
    """shard id -> {file, data_off, path}（real_table_probe.py:54-76 抄改）。"""
    out = {}
    for f in sorted(CKPT.glob("model-*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
        for i in range(NSHARDS):
            name = PLE_PREFIX + "ple_embedding.ngram_embedding.shard_%d.weight" % i
            if name in hdr:
                v = hdr[name]
                o0, o1 = v["data_offsets"]
                assert v["dtype"] == "BF16" and tuple(v["shape"]) == (ROWS_PER_SHARD, HDIM), \
                    (name, v["dtype"], v["shape"])
                assert (o1 - o0) == ROWS_PER_SHARD * ROW_BYTES
                out[i] = {"file": f, "data_off": base + o0, "path": str(f)}
    assert len(out) == NSHARDS, "only %d/128 shards found" % len(out)
    return out


def direct_row(smap, gid):
    """独立路径：直接 seek 分片文件读一行（不经窗口）。"""
    sh, local = divmod(int(gid), ROWS_PER_SHARD)
    ent = smap[sh]
    with open(ent["file"], "rb") as fh:
        fh.seek(ent["data_off"] + local * ROW_BYTES)
        return fh.read(ROW_BYTES)


# ------------------------------------------------------------------ 参考数学
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
            r0 = mixed % sz
            out[t, g] = r0 + int(offsets[g])
    return out


def ref_norm_gemma(x, w, eps=EPS):
    x = x.astype(np.float64)
    var = (x * x).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = x * rstd * (1.0 + w.astype(np.float64))
    return bf16_rne(y)


def ref_gate(key, value, hidden, nk, nq, ncw):
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
    """state_rows = 旧状态 [9,W]（行 8 最新）；wconv [W,4] tap-major（列 k=0..3）。"""
    taps = [state_rows[0], state_rows[3], state_rows[6], conv_in]
    acc = np.zeros(W, dtype=np.float64)
    for k in range(KCONV):
        acc += wconv[:, k].astype(np.float64) * taps[k].astype(np.float64)
    conv = bf16_rne(acc)
    y = conv.astype(np.float64) * (1.0 / (1.0 + np.exp(-conv.astype(np.float64))))
    co = bf16_rne(y)
    po = bf16_rne(gated_row.astype(np.float32) + co)
    return po, co


if __name__ == "__main__":
    h = st_headers()
    print("ckpt headers: %d files" % len(h))
