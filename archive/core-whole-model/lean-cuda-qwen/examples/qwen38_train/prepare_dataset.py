#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Tokenize a text corpus into packed little-endian UInt32 token streams for qwen38_train.

Run through uv so the pinned tokenizer wheel is provisioned on demand:

    uv run --no-project --with 'tokenizers==0.23.1' python prepare_dataset.py \
      --model-dir "$QWEN_MODEL_DIR" --input corpus.txt \
      --train-out train.bin --val-out val.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

VOCABULARY = 248320


def encode_corpus(tokenizer, input_path: Path) -> list[int]:
    ids: list[int] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_ids = tokenizer.encode(line, add_special_tokens=False).ids
            for token in line_ids:
                if token >= VOCABULARY:
                    raise ValueError(
                        f"line {line_number}: token id {token} is outside the Qwen3.8 vocabulary"
                    )
            ids.extend(line_ids)
    return ids


def write_tokens(path: Path, ids: list[int]) -> None:
    with path.open("wb") as handle:
        for start in range(0, len(ids), 1 << 20):
            chunk = ids[start : start + (1 << 20)]
            handle.write(struct.pack(f"<{len(chunk)}I", *chunk))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", required=True, type=Path)
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=65536,
        help="tokens reserved from the end of the corpus for validation",
    )
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        print(f"error: missing tokenizer at {tokenizer_path}", file=sys.stderr)
        return 2
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    ids = encode_corpus(tokenizer, args.input)
    if len(ids) <= 2 * args.val_tokens:
        print(
            f"error: corpus has {len(ids)} tokens; need more than {2 * args.val_tokens} "
            "for the requested validation split",
            file=sys.stderr,
        )
        return 2
    train_ids = ids[: len(ids) - args.val_tokens]
    val_ids = ids[len(ids) - args.val_tokens :]
    write_tokens(args.train_out, train_ids)
    write_tokens(args.val_out, val_ids)
    print(f"train: {len(train_ids)} tokens -> {args.train_out}")
    print(f"val: {len(val_ids)} tokens -> {args.val_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
