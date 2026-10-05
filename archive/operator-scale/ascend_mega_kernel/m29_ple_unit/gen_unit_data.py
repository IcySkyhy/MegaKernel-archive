#!/usr/bin/env python3.12
"""m29_ple_unit/gen_unit_data.py —— M161 PLE 单元的数据生成

产出（默认写到 `--out` 目录；`*.bin` 被 .gitignore 忽略，不入库）：
  权重（真实 checkpoint，layers.1.ple.*）：w_key_proj.bin / w_value_proj.bin / w_conv1d_tap.bin /
    w_norm_key.bin / w_norm_query.bin / w_norm_conv.bin
  id 常量：m.bin / sizes.bin / offsets.bin（真实 checkpoint 值 —— 两档都用真实算术常量）
  每 step 输入：s{k}_ids.bin(int32 输入 token) / s{k}_qsl.bin / s{k}_ctx.bin / s{k}_hidden.bin /
    s{k}_sidx.bin
  初始状态：state_0.bin（[9,10240] bf16，slot 0 的初始 short-conv 状态；其余 slot 由 host 补零）
  表 + 元数据：unit_meta.txt
    · `--mode synth`：`table_flat.bin`（缩减词表的合成表 [sum(RED_SIZES),160]），n_slots=0
    · `--mode real` ：`slot_{i}.bin`（从**真实分片文件**抽出的行段）+ 多槽元数据，n_slots>0
      `--miss` 只 stage 单个分片的槽 ⇒ 别的分片的 id 会「越窗」（设备侧 miss 计数）

用法：
  python3 m29_ple_unit/gen_unit_data.py --mode synth --out m29_ple_unit/data_synth
  python3 m29_ple_unit/gen_unit_data.py --mode real  --out m29_ple_unit/data_real
  python3 m29_ple_unit/gen_unit_data.py --mode real  --miss --out m29_ple_unit/data_real_miss
"""
import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m29_common as C  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
SEED = 20261004
N_TOK = 2
N_REQ = 2
N_STEPS = 2
SLOT_BLOCK = 4     # 真实档：每个 (shard, local//SLOT_BLOCK) 块一个 4 行槽


def write(path, b):
    pathlib.Path(path).write_bytes(b)
    return len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], required=True)
    ap.add_argument("--miss", action="store_true", help="real 档：只 stage 一个分片（越窗档）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-seed", type=int, default=7)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    hdrs = C.st_headers()

    # ---- 真实权重 ----
    for key in ["key_proj", "value_proj", "conv1d", "norm_key", "norm_query", "norm_conv"]:
        dt, sh, raw = C.read_tensor(hdrs, C.SMALL_TENSORS[key])
        assert dt == "BF16", (key, dt)
        write(out / ("w_%s.bin" % key), raw)
    dt, sh, raw = C.read_tensor(hdrs, C.SMALL_TENSORS["conv1d"])
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(sh[0], 1, sh[2]).reshape(sh[0], sh[2])
    write(out / "w_conv1d_tap.bin", np.ascontiguousarray(arr.T).tobytes())   # [4,10240] tap-major
    print("[gen] weights: 6 tensors + conv1d_tap %s" % (sh,))

    # ---- id 常量（真实）----
    dt, sh, raw = C.read_tensor(hdrs, C.SMALL_TENSORS["layer_multipliers"])
    m = np.frombuffer(raw, dtype=np.int64)
    dt, sh, raw = C.read_tensor(hdrs, C.SMALL_TENSORS["ngram_heads_vocab_sizes"])
    sizes_real = np.frombuffer(raw, dtype=np.int64)
    dt, sh, raw = C.read_tensor(hdrs, C.SMALL_TENSORS["ngram_heads_offsets"])
    offs_real = np.frombuffer(raw, dtype=np.int64)
    write(out / "m.bin", m.tobytes())

    # ---- 输入 token / qsl / ctx / hidden / sidx ----
    qsl = np.array([0, 1, 2], dtype=np.int32)     # N_REQ=2，每请求 1 token
    ctx = np.array([[C.EOS, C.EOS], [C.EOS, C.EOS]], dtype=np.int32)
    sidx = np.array([0, 1], dtype=np.int32)       # 2 个 state slot（两请求各一）
    write(out / "state_0.bin", np.zeros((C.STLEN, C.W), dtype=np.uint16).tobytes())

    if args.mode == "synth":
        sizes = np.array(C.RED_SIZES, dtype=np.int64)
        offs = np.cumsum(np.concatenate([[0], sizes[:-1]])).astype(np.int64)
        table_rows = int(sizes.sum())
        # 合成表（随机 bf16），与真实档同为 bf16 行
        tbl = C.pack_bf16(rng.normal(0.0, 0.02, size=(table_rows, C.HDIM)).astype(np.float32))
        write(out / "table_flat.bin", tbl)
        n_slots = 0
    else:
        sizes = sizes_real
        offs = offs_real
        n_slots = None

    # 输入 token（两档都取一串较小的 id，避免 EOS；真实档也 < 分片行数）
    toks = np.array([123456, 654321], dtype=np.int32)
    steps_ids = []
    for k in range(N_STEPS):
        ids_k = np.array([toks[0] + 7 * k, toks[1] + 13 * k], dtype=np.int32)
        write(out / ("s%d_ids.bin" % k), ids_k.tobytes())
        write(out / ("s%d_qsl.bin" % k), qsl.tobytes())
        write(out / ("s%d_ctx.bin" % k), ctx.ravel().tobytes())
        hid = C.pack_bf16(rng.normal(0.0, 0.5, size=(N_TOK, C.W)).astype(np.float32))
        write(out / ("s%d_hidden.bin" % k), hid)
        write(out / ("s%d_sidx.bin" % k), sidx.tobytes())
        steps_ids.append(ids_k)
    write(out / "sizes.bin", sizes.tobytes())
    write(out / "offsets.bin", offs.tobytes())

    # ---- 表 & 元数据 ----
    lines = []
    if args.mode == "synth":
        lines += ["mode flat", "table_rows %d" % table_rows, "n_slots 0", "table_file table_flat.bin"]
        print("[gen] synth table rows=%d (16 primes)" % table_rows)
    else:
        smap = C.shard_map()
        # 计算每 step 的 ids，收集需要的 (shard, local//BLOCK)
        need = {}
        for k in range(N_STEPS):
            ids = C.ref_ids(steps_ids[k], qsl, ctx, m, sizes, offs)
            for gid in ids.ravel():
                sh, local = divmod(int(gid), C.ROWS_PER_SHARD)
                blk = local // SLOT_BLOCK
                need[(sh, blk)] = local
        keys = sorted(need.keys())
        if args.miss:
            keep_shard = keys[0][0]
            keys = [kkey for kkey in keys if kkey[0] == keep_shard]
        slots = []
        for (sh, blk) in keys:
            l0 = blk * SLOT_BLOCK
            rows = min(SLOT_BLOCK, C.ROWS_PER_SHARD - l0)
            ent = smap[sh]
            with open(ent["file"], "rb") as fh:
                fh.seek(ent["data_off"] + l0 * C.ROW_BYTES)
                blob = fh.read(rows * C.ROW_BYTES)
            slots.append((sh, l0, rows, blob))
        total_rows = sum(s[2] for s in slots)
        for i, (sh, l0, rows, blob) in enumerate(slots):
            write(out / ("slot_%d.bin" % i), blob)
        lines += ["mode multi", "table_rows %d" % total_rows, "n_slots %d" % len(slots)]
        win = 0
        for i, (sh, l0, rows, blob) in enumerate(slots):
            lines.append("slot %d shard %d local0 %d rows %d win %d file slot_%d.bin" % (i, sh, l0, rows, win, i))
            win += rows
        print("[gen] real multi-slot: %d slots, %d rows, shards=%s"
              % (len(slots), total_rows, sorted(set(s[0] for s in slots))))

    lines = ["n_tok %d" % N_TOK, "n_steps %d" % N_STEPS, "n_req %d" % N_REQ] + lines
    write(out / "unit_meta.txt", ("\n".join(lines) + "\n").encode())
    print("[gen] out=%s" % out)


if __name__ == "__main__":
    main()
