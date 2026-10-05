#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Prepare fixed-shape chosen/rejected records for qwen38_train DPO.

Each input JSONL object has either a plain ``prompt`` or a chat ``messages`` array, plus
``chosen`` and ``rejected`` response strings. The output stores complete token sequences and
assistant-only masks in a self-describing LCQDPO1 binary.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAGIC = b"LCQDPO1\0"
VERSION = 1
VOCABULARY = 248320
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
TURN_END = "<|im_end|>\n"


@dataclass(frozen=True)
class EncodedSequence:
    tokens: tuple[int, ...]
    mask: tuple[int, ...]


@dataclass(frozen=True)
class EncodedPair:
    chosen: EncodedSequence
    rejected: EncodedSequence


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def render_history(record: dict[str, Any]) -> str:
    """Render the same text-only Qwen chat boundary used by qwen38_chat."""
    pieces: list[str] = []
    if "messages" in record:
        messages = record["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty array")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}].role is invalid")
            content = _require_text(message.get("content"), f"messages[{index}].content")
            pieces.append(f"<|im_start|>{role}\n{content}{TURN_END}")
    else:
        prompt = _require_text(record.get("prompt"), "prompt")
        system = record.get("system", "")
        if not isinstance(system, str):
            raise ValueError("system must be text")
        if system.strip():
            pieces.append(f"<|im_start|>system\n{system.strip()}{TURN_END}")
        pieces.append(f"<|im_start|>user\n{prompt}{TURN_END}")
    return "".join(pieces) + ASSISTANT_PREFIX


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    for token in ids:
        if not isinstance(token, int) or token < 0 or token >= VOCABULARY:
            raise ValueError(f"token id {token!r} is outside the Qwen3.8 vocabulary")
    return ids


def encode_response(
    tokenizer: Any,
    prefix_ids: list[int],
    response: str,
    sequence_length: int,
    pad_token: int,
) -> EncodedSequence:
    # Tokenizing prompt and completion separately guarantees identical prompt tokens for the pair.
    response_ids = _encode(tokenizer, response.strip() + TURN_END)
    if not response_ids:
        raise ValueError("response produced no tokens")
    actual = prefix_ids + response_ids
    required = len(actual) - 1
    if required > sequence_length:
        raise ValueError(
            f"prompt plus response needs {required} rows, exceeding sequence length "
            f"{sequence_length}"
        )
    if len(prefix_ids) == 0:
        raise ValueError("prompt produced no tokens")
    padded = actual + [pad_token] * (sequence_length + 1 - len(actual))
    mask = [
        int(len(prefix_ids) <= target_index < len(actual))
        for target_index in range(1, sequence_length + 1)
    ]
    if not any(mask):
        raise ValueError("response has no trainable target tokens")
    return EncodedSequence(tuple(padded), tuple(mask))


def encode_record(
    tokenizer: Any, record: dict[str, Any], sequence_length: int, pad_token: int
) -> EncodedPair:
    prefix_ids = _encode(tokenizer, render_history(record))
    chosen = encode_response(
        tokenizer,
        prefix_ids,
        _require_text(record.get("chosen"), "chosen"),
        sequence_length,
        pad_token,
    )
    rejected = encode_response(
        tokenizer,
        prefix_ids,
        _require_text(record.get("rejected"), "rejected"),
        sequence_length,
        pad_token,
    )
    return EncodedPair(chosen, rejected)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError("preference dataset is empty")
    return records


def _write_u32s(handle: Any, values: Iterable[int]) -> None:
    chunk: list[int] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == 1 << 20:
            handle.write(struct.pack(f"<{len(chunk)}I", *chunk))
            chunk.clear()
    if chunk:
        handle.write(struct.pack(f"<{len(chunk)}I", *chunk))


def write_dataset(path: Path, sequence_length: int, pairs: list[EncodedPair]) -> None:
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<III", VERSION, sequence_length, len(pairs)))
        for pair in pairs:
            for sequence in (pair.chosen, pair.rejected):
                _write_u32s(handle, sequence.tokens)
                _write_u32s(handle, sequence.mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="preference JSONL")
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", type=Path)
    parser.add_argument("--val-records", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3638)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.val_records < 0:
        parser.error("--val-records must be nonnegative")
    if (args.val_out is None) != (args.val_records == 0):
        parser.error("--val-out and a positive --val-records must be provided together")

    tokenizer_path = args.model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        print(f"error: missing tokenizer: {tokenizer_path}", file=sys.stderr)
        return 2

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token = tokenizer.token_to_id("<|im_end|>")
    if pad_token is None:
        print("error: tokenizer has no <|im_end|> token", file=sys.stderr)
        return 2

    try:
        source = read_jsonl(args.input)
        if args.val_records >= len(source):
            raise ValueError(
                f"validation split {args.val_records} must be smaller than {len(source)} records"
            )
        random.Random(args.seed).shuffle(source)
        encoded = [
            encode_record(tokenizer, record, args.sequence_length, pad_token)
            for record in source
        ]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    split = len(encoded) - args.val_records
    write_dataset(args.train_out, args.sequence_length, encoded[:split])
    print(f"train: {split} preference pairs -> {args.train_out}")
    if args.val_out is not None:
        write_dataset(args.val_out, args.sequence_length, encoded[split:])
        print(f"val: {args.val_records} preference pairs -> {args.val_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
