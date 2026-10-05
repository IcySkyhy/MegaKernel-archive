#!/usr/bin/env python3
"""M30: byte-level inventory of the Qwen3.8-Flash-Next-MXFP4 checkpoint.

Reads only the safetensors *headers* (JSON at the start of every shard), so it is cheap
and needs no torch. It answers the question the TPS baseline depends on: how much of the
checkpoint must live in HBM, and how much the design expects to keep in host memory.

Usage: python3 baseline_env/scripts/checkpoint_memory_inventory.py [--json]
"""

import collections
import glob
import json
import os
import struct
import sys

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/Qwen3.8-Flash-Next-MXFP4")

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "I4": 0.5, "U4": 0.5,
}


def category(name: str, dtype: str = "") -> str:
    n = name.lower()
    if "ngram" in n or "next_gram" in n or "nextgram" in n:
        return "ngram embedding table"
    if "experts" in n or "shared_expert" in n:
        if "scale" in n:
            return "MoE experts (e8m0 scale)"
        if dtype == "U8":
            return "MoE experts (mxfp4 packed u8)"
        return "MoE experts (unquantised bf16)"
    if "linear_attn" in n or "conv1d" in n or "gdn" in n or ".a_log" in n or "dt_bias" in n:
        return "GDN / linear attention"
    if "self_attn" in n or "indexer" in n or "qsa" in n:
        return "full attention / indexer"
    if "embed_tokens" in n or "lm_head" in n:
        return "embed / lm_head"
    if "visual" in n or "vision" in n:
        return "vision tower"
    if "mtp" in n:
        return "MTP head"
    if "norm" in n:
        return "norms"
    if "router" in n or "gate" in n or "hc" in n or "hyper" in n or "ple" in n:
        return "router / hyper-connection / ple"
    return "other"


def main():
    shards = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
    if not shards:
        print(f"no shards under {MODEL_DIR}", file=sys.stderr)
        return 1

    per_cat = collections.Counter()
    per_cat_tensors = collections.Counter()
    per_dtype = collections.Counter()
    total = 0
    n_tensors = 0
    biggest = []

    for path in shards:
        with open(path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]
            nbytes = end - start
            total += nbytes
            n_tensors += 1
            per_dtype[dtype] += nbytes
            cat = category(name, dtype)
            per_cat[cat] += nbytes
            per_cat_tensors[cat] += 1
            biggest.append((nbytes, name, dtype))

    gib = lambda b: b / 1024**3  # noqa: E731
    print(f"model dir        : {MODEL_DIR}")
    print(f"shards           : {len(shards)}")
    print(f"tensors          : {n_tensors}")
    print(f"total checkpoint : {gib(total):.2f} GiB ({total / 1e9:.2f} GB)")
    print()
    print("== by dtype ==")
    for dtype, nbytes in per_dtype.most_common():
        print(f"  {dtype:8s} {gib(nbytes):9.2f} GiB  ({100 * nbytes / total:5.1f}%)")
    print()
    print("== by role ==")
    for cat, nbytes in per_cat.most_common():
        print(f"  {cat:34s} {gib(nbytes):9.2f} GiB  {per_cat_tensors[cat]:5d} tensors"
              f"  ({100 * nbytes / total:5.1f}%)")
    print()
    if "--json" in sys.argv:
        print(json.dumps({"total_bytes": total, "by_category": dict(per_cat),
                          "by_dtype": dict(per_dtype)}))
    print("== 10 largest tensors ==")
    for nbytes, name, dtype in sorted(biggest, reverse=True)[:10]:
        print(f"  {gib(nbytes):9.3f} GiB  {dtype:6s} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
