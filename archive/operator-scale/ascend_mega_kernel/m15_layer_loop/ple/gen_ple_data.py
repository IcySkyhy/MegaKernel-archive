#!/usr/bin/env python3.12
"""m15_ple/gen_ple_data.py —— M85 PLE 独立 kernel 的输入/权重数据生成

产出的每个文件都进 `m15_layer_loop/ple/data/`（`*.bin` 被 m15_layer_loop/.gitignore 忽略，不入库）。

设计（为什么这么切）：
  * **权重**：`key_proj` / `value_proj` / `conv1d` / 三个 norm 直接用 **真实 checkpoint** 的 BF16 张量
    （`model.language_model.layers.1.ple.*`，共 62.64 MiB）—— 比合成权重更能暴露值域/舍入问题。
  * **ngram 表**：真实表 95.37 GiB 放不下（docs/14 §10 第 1 条），本 harness 用 **缩减词表** 的合成表：
    16 个 head 的 `size[h]` 取 16 个小素数、`offset[h]` 取前缀和、表行数 = Σsize。
    这样 ①「id 算术 + 表查找」这条链路仍然**逐元素真实**，只是词表规模缩小。
  * **同时**产出 `*_full.bin`（`multipliers` 用真实值、`sizes`/`offsets` 用真实值）：
    矩阵 A 只跑 ① 的 id 算术，与参考**逐位**比（T1），词表规模取真实常量。
  * 三个 int64 buffer 的**真实值**从 checkpoint 直接读出，并与 `docs/14 §6.4` 的公式复算值
    **逐值比对**（不一致就报错退出）—— 这既是生成的前提，也是 §6.4 结论的一次独立复算。

用法：
  /usr/local/python3.12.13/bin/python3.12 m15_layer_loop/ple/gen_ple_data.py [--mode M85] [--tag M85]
"""
import argparse
import json
import os
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CKPT = pathlib.Path("/workspace/Qwen3.8-Flash-Next-MXFP4")

# ---- 模型常量（config.json#text_config，见 PLE_SPEC.md §7 命令 A）----
HID = 2560          # hidden_size
HC = 4              # hc_count
W = HID * HC        # 10240
HE = 2560           # ple_embed_dim
P = 8               # heads_per_ngram
NGR = 3             # ngram_size
G = (NGR - 1) * P   # 16
HEAD_DIM = HE // G  # 160
KCONV = 4
DIL = NGR           # 3
STATE_LEN = (KCONV - 1) * DIL  # 9
EPS = 1e-6
EOS = 248044
SEED = 1234

# 缩减词表的 16 个小素数（保持"两两互素"的结构，便于暴露 mod 方向错）
RED_SIZES = [131, 127, 113, 109, 107, 103, 101, 97, 89, 83, 79, 73, 71, 67, 61, 59]

PLE_PREFIX = "model.language_model.layers.1.ple."
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


# ------------------------------------------------------------------ 工具
def bf16_rne(x):
    """float32/float64 -> bf16（round-to-nearest-even），返回 float32（可精确表示 bf16）。"""
    a = np.asarray(x, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    rounded = u + 0x7FFF + lsb
    out = (rounded & np.uint64(0xFFFF0000)).astype(np.uint32)
    return out.view(np.float32)


def splitmix64(x):
    M = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & M
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    return z ^ (z >> 31)


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def nth_prime_after(base, n):
    """第 n 个 > base 的素数（n 从 1 起）。"""
    out, c = [], base
    while len(out) < n:
        c += 1
        if is_prime(c):
            out.append(c)
    return out


def gen_reference_buffers():
    """按 docs/14 §6.4 的公式复算 m / size / offset。"""
    max_mult = ((1 << 63) - 1) // 248320
    half_bound = max_mult // 2
    m = [2 * (splitmix64(SEED + 10007 * 0 + 0x9E3779B97F4A7C15 * (i + 1)) % half_bound) + 1
         for i in range(3)]
    sizes = nth_prime_after(20000000 - 1, 16)
    offsets = []
    acc = 0
    for s in sizes:
        offsets.append(acc)
        acc += s
    return m, sizes, offsets, acc


# ------------------------------------------------------------ safetensors 读
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


def read_table_shards(hdrs, limit=None):
    """把 shard_0..127 按序拼成一条 [rows,160] BF16。limit=只读前 limit 片（本 harness 不用）。"""
    rows = []
    n = 0
    for i in range(128):
        name = PLE_PREFIX + "ple_embedding.ngram_embedding.shard_%d.weight" % i
        dt, sh, raw = read_tensor(hdrs, name)
        assert dt == "BF16" and sh == (2500012, 160), (name, dt, sh)
        if limit is not None and i >= limit:
            break
        rows.append(raw)
        n += sh[0]
    return b"".join(rows), n


def dtype_np(dt):
    return {"BF16": np.uint16, "I64": np.int64, "I32": np.int32}.get(dt) and {
        "BF16": np.uint16, "I64": np.int64, "I32": np.int32}[dt]


def bf16_bytes(x):
    """float32 数组 -> bf16 字节（先 RNE 到 bf16，再取高 16 位）。"""
    return bf16_rne(x).view(np.uint32).astype(np.uint32).ravel()


def pack_bf16(x):
    """float32 数组 → bf16 字节。注意必须取**高 16 位**（>>16）：bf16_rne 已把低 16 位清零，
    直接 astype(uint16) 会得到全 0（M85 实测踩过：合成数据全 0 ⇒ 判据"空过"，见 ple/README.md 教训）。"""
    return (bf16_rne(x).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def rand_bf16(shape, rng, scale):
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=20260927)
    # hidden 的独立种子：`pick_observable_seed.py` 搜出来的值 —— 使 P2（dot 和的 bf16 物化）
    # 这类"只在部分输入上可观测"的分歧在随包数据上**可被判据抓住**（见 ple/DIVERGENCES.md D4）
    ap.add_argument("--hidden-seed", type=int, default=4)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    hdrs = st_headers()

    # ---- 0. 复算 m/size/offset，并与 checkpoint 逐值对账 ----
    m_ref, sizes_ref, offsets_ref, total_ref = gen_reference_buffers()
    dt, sh, raw = read_tensor(hdrs, SMALL_TENSORS["layer_multipliers"])
    m_ck = list(struct.unpack("<3q", raw))
    dt2, sh2, raw2 = read_tensor(hdrs, SMALL_TENSORS["ngram_heads_vocab_sizes"])
    sizes_ck = list(struct.unpack("<16q", raw2))
    dt3, sh3, raw3 = read_tensor(hdrs, SMALL_TENSORS["ngram_heads_offsets"])
    offsets_ck = list(struct.unpack("<16q", raw3))
    assert m_ck == m_ref, ("multipliers mismatch", m_ck, m_ref)
    assert sizes_ck == sizes_ref, ("vocab_sizes mismatch", sizes_ck, sizes_ref)
    assert offsets_ck == offsets_ref, ("offsets mismatch", offsets_ck, offsets_ref)
    print("[gen] int64 buffers: ckpt == docs/14 §6.4 recomputation (3/3) ; total=%d padded=%d"
          % (total_ref, ((total_ref + 127) // 128) * 128))

    # ---- 1. 真实 PLE 小张量 ----
    meta = {"HID": HID, "HC": HC, "W": W, "HE": HE, "P": P, "NGR": NGR, "G": G,
            "HEAD_DIM": HEAD_DIM, "KCONV": KCONV, "DIL": DIL, "STATE_LEN": STATE_LEN,
            "EPS": EPS, "EOS": EOS}
    for key in ["key_proj", "value_proj", "conv1d", "norm_key", "norm_query", "norm_conv"]:
        dt, sh, raw = read_tensor(hdrs, SMALL_TENSORS[key])
        assert dt == "BF16", (key, dt)
        (out / ("w_%s.bin" % key)).write_bytes(raw)
        meta["shape_" + key] = list(sh)
        print("[gen] w_%-11s %-16s %s bytes=%d" % (key, dt, sh, len(raw)))

    # conv1d 额外落一份 squeeze(1) 后的 [10240,4]（kernel 直接用）
    dt, sh, raw = read_tensor(hdrs, SMALL_TENSORS["conv1d"])
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(sh[0], 1, sh[2])
    (out / "w_conv1d_sq.bin").write_bytes(arr.reshape(sh[0], sh[2]).tobytes())
    # 再落一份 **tap-major [4,10240]**（同 GDN 的 `w[j][c]` 布局，m15_gdn_layer.h:557-563）：
    # kernel 侧每个通道块按 tap 连续搬运，无需带 stride 的 DataCopy
    (out / "w_conv1d_tap.bin").write_bytes(np.ascontiguousarray(arr.reshape(sh[0], sh[2]).T).tobytes())

    # ---- 2. multipliers / sizes / offsets（缩减档 + 真实档）----
    (out / "m_red.bin").write_bytes(struct.pack("<3q", *m_ref))          # 乘子用真实值
    (out / "sizes_red.bin").write_bytes(struct.pack("<16q", *RED_SIZES))
    off_red, acc = [], 0
    for s in RED_SIZES:
        off_red.append(acc)
        acc += s
    (out / "offsets_red.bin").write_bytes(struct.pack("<16q", *off_red))
    rows_red = acc
    (out / "m_full.bin").write_bytes(struct.pack("<3q", *m_ref))
    (out / "sizes_full.bin").write_bytes(struct.pack("<16q", *sizes_ref))
    (out / "offsets_full.bin").write_bytes(struct.pack("<16q", *offsets_ref))
    meta.update({"rows_red": rows_red, "rows_full": total_ref,
                 "sizes_red": RED_SIZES, "offsets_red": off_red,
                 "m": m_ref, "sizes_full": sizes_ref, "offsets_full": offsets_ref})
    print("[gen] reduced vocab: rows=%d ; full vocab rows=%d" % (rows_red, total_ref))

    # ---- 3. 合成 ngram 表（缩减档）----
    table = rand_bf16((rows_red, HEAD_DIM), rng, 0.05)
    (out / "table_red.bin").write_bytes(pack_bf16(table))
    print("[gen] table_red %s bf16 bytes=%d" % ((rows_red, HEAD_DIM), rows_red * HEAD_DIM * 2))

    # ---- 4. 矩阵 A（只跑 ①）：T=10，2 请求（6+4 token），含 EOS 回退 ----
    ids_A = [101, 102, 103, EOS, 105, 106, 201, 202, 203, 204]
    qsl_A = [0, 6, 10]
    ctx_A = [[77, 78], [88, 89]]
    (out / "ids_A.bin").write_bytes(struct.pack("<10i", *ids_A))
    (out / "qsl_A.bin").write_bytes(struct.pack("<3i", *qsl_A))
    (out / "ctx_A.bin").write_bytes(struct.pack("<4i", *[v for r in ctx_A for v in r]))
    meta.update({"T_A": len(ids_A), "R_A": len(qsl_A) - 1, "ids_A": ids_A,
                 "qsl_A": qsl_A, "ctx_A": ctx_A})

    # ---- 5. 矩阵 B（跑 ①②③④⑤）：T=2 = 2 请求各 1 token（decode）----
    ids_B = [4096, 12345]
    qsl_B = [0, 1, 2]
    ctx_B = [[31, 32], [41, 42]]
    (out / "ids_B.bin").write_bytes(struct.pack("<2i", *ids_B))
    (out / "qsl_B.bin").write_bytes(struct.pack("<3i", *qsl_B))
    (out / "ctx_B.bin").write_bytes(struct.pack("<4i", *[v for r in ctx_B for v in r]))
    meta.update({"T_B": 2, "R_B": 2, "ids_B": ids_B, "qsl_B": qsl_B, "ctx_B": ctx_B})

    # H'（物化后的多流态）与 conv 状态
    hidden = rand_bf16((2, W), np.random.default_rng(args.hidden_seed), 0.5)
    (out / "hidden.bin").write_bytes(pack_bf16(hidden))
    # conv 状态用 **planar [slot][h][c]** 布局（同 GDN 的 csGm_[j*CH + c] 惯例）：
    # kernel 侧 tap 按通道块连续搬运，无需带 stride 的 DataCopy。
    state = rand_bf16((2, STATE_LEN, W), rng, 0.3)
    state[0] = 0.0                       # slot 0 用全零（无历史），slot 1 用随机（有历史）
    (out / "conv_state.bin").write_bytes(pack_bf16(state))
    # state_idx：**含一个 null 槽位（-1）** —— 上游 `ops/ple.py:20,391-405` 的 NULL_STATE_ID 路径
    #（null 行 out_ok=false ⇒ conv_output=0，但仍写 out = bf16(hidden+bf16(gated+0))，
    # 且**不写**状态行）。判据 B5.null 专门咬这条路径。
    (out / "state_idx.bin").write_bytes(struct.pack("<2i", -1, 1))
    meta.update({"hidden_scale": 0.5, "hidden_seed": args.hidden_seed, "state_scale": 0.3,
                 "state_slot0_zero": True, "state_idx": [-1, 1], "null_rows": [0]})

    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    # `docs/17 §1.3` 要求 ① 类（声明输入）在 README 里给「路径 + sha256」：
    # 这里把本次生成的全部输入文件的 sha256 落成 SHA256SUMS，供 README 引用（可随数据重生成）
    import hashlib
    lines = []
    for f in sorted(out.iterdir()):
        if f.name == "SHA256SUMS":
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append("%s  %s" % (h, f.name))
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print("[gen] 写了 SHA256SUMS（%d 个输入文件；docs/17 §1.3 的 ① 类要求）" % len(lines))
    print("[gen] wrote %d files to %s" % (len(list(out.iterdir())), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
