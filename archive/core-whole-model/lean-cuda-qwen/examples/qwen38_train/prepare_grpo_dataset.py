#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.

"""Prepare fixed-shape grouped rollout records for qwen38_train outcome-GRPO.

Each JSONL record contains one common prompt or messages history and a rollouts array.
Every rollout has a finite reward, behavior-policy log probabilities for each trainable token,
and either a response string or ordered segments with explicit train booleans. Segments let tool
observations remain context while supervising only assistant actions. Every group also identifies
the behavior checkpoint with a SHA-256 digest. The output is a self-describing LCQGRP2 binary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import prepare_preference_dataset as preference

MAGIC = b"LCQGRP2\0"
VERSION = 2
VOCABULARY = 248320


@dataclass(frozen=True)
class EncodedRollout:
    tokens: tuple[int, ...]
    mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    reward: float


@dataclass(frozen=True)
class EncodedGroup:
    behavior_policy_sha256: bytes
    rollouts: tuple[EncodedRollout, ...]


def _require_reward(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError(f"{label} must be finite")
    if not math.isfinite(struct.unpack("<f", struct.pack("<f", reward))[0]):
        raise ValueError(f"{label} is outside the Float32 range")
    return reward


def _require_policy_sha256(value: Any) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("behavior_policy_sha256 must contain exactly 64 hexadecimal characters")
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("behavior_policy_sha256 must be hexadecimal") from error
    if len(digest) != 32:
        raise ValueError("behavior_policy_sha256 must encode a 32-byte SHA-256 digest")
    return digest


def _require_behavior_logprobs(value: Any, count: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(
            f"{label}.behavior_logprobs must contain exactly {count} trainable-token values"
        )
    result: list[float] = []
    for index, raw in enumerate(value):
        probability = _require_reward(raw, f"{label}.behavior_logprobs[{index}]")
        if probability > 0:
            raise ValueError(f"{label}.behavior_logprobs[{index}] must be at most zero")
        result.append(probability)
    return result


def _encode_segments(
    tokenizer: Any, rollout: dict[str, Any], label: str
) -> tuple[list[int], list[int]]:
    if "response" in rollout:
        response = preference._require_text(rollout.get("response"), f"{label}.response")
        ids = preference._encode(tokenizer, response + preference.TURN_END)
        return ids, [1] * len(ids)
    segments = rollout.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{label} must contain response text or nonempty segments")
    ids: list[int] = []
    flags: list[int] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"{label}.segments[{index}] must be an object")
        text = segment.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{label}.segments[{index}].text must be nonempty text")
        train = segment.get("train")
        if not isinstance(train, bool):
            raise ValueError(f"{label}.segments[{index}].train must be boolean")
        segment_ids = preference._encode(tokenizer, text)
        ids.extend(segment_ids)
        flags.extend([int(train)] * len(segment_ids))
    if not any(flags):
        raise ValueError(f"{label} has no trainable assistant tokens")
    return ids, flags


def encode_rollout(
    tokenizer: Any,
    prefix_ids: list[int],
    rollout: dict[str, Any],
    sequence_length: int,
    pad_token: int,
    label: str,
) -> EncodedRollout:
    if not isinstance(rollout, dict):
        raise ValueError(f"{label} must be an object")
    generated_ids, generated_flags = _encode_segments(tokenizer, rollout, label)
    if not generated_ids:
        raise ValueError(f"{label} produced no tokens")
    actual = prefix_ids + generated_ids
    required = len(actual) - 1
    if required > sequence_length:
        raise ValueError(
            f"{label} needs {required} rows, exceeding sequence length {sequence_length}"
        )
    token_flags = [0] * len(prefix_ids) + generated_flags
    trainable_logprobs = iter(
        _require_behavior_logprobs(
            rollout.get("behavior_logprobs"), sum(generated_flags), label
        )
    )
    generated_logprobs = [next(trainable_logprobs) if flag else 0.0 for flag in generated_flags]
    token_logprobs = [0.0] * len(prefix_ids) + generated_logprobs
    padded = actual + [pad_token] * (sequence_length + 1 - len(actual))
    mask = token_flags[1:] + [0] * (sequence_length - required)
    behavior_logprobs = token_logprobs[1:] + [0.0] * (sequence_length - required)
    if (
        len(padded) != sequence_length + 1
        or len(mask) != sequence_length
        or len(behavior_logprobs) != sequence_length
    ):
        raise AssertionError("internal fixed-shape rollout encoding mismatch")
    if not any(mask):
        raise ValueError(f"{label} has no trainable assistant target tokens")
    return EncodedRollout(
        tuple(padded),
        tuple(mask),
        tuple(behavior_logprobs),
        _require_reward(rollout.get("reward"), f"{label}.reward"),
    )


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    sequence_length: int,
    group_size: int,
    pad_token: int,
) -> EncodedGroup:
    prefix_ids = preference._encode(tokenizer, preference.render_history(record))
    if not prefix_ids:
        raise ValueError("prompt produced no tokens")
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != group_size:
        raise ValueError(f"rollouts must contain exactly {group_size} members")
    encoded = tuple(
        encode_rollout(
            tokenizer,
            prefix_ids,
            rollout,
            sequence_length,
            pad_token,
            f"rollouts[{index}]",
        )
        for index, rollout in enumerate(rollouts)
    )
    return EncodedGroup(_require_policy_sha256(record.get("behavior_policy_sha256")), encoded)


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
        raise ValueError("GRPO rollout dataset is empty")
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


def write_dataset(
    path: Path, sequence_length: int, group_size: int, groups: list[EncodedGroup]
) -> None:
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<IIII", VERSION, sequence_length, group_size, len(groups)))
        for group in groups:
            if len(group.rollouts) != group_size:
                raise ValueError("encoded group size does not match dataset header")
            handle.write(group.behavior_policy_sha256)
            for rollout in group.rollouts:
                _write_u32s(handle, rollout.tokens)
                _write_u32s(handle, rollout.mask)
                handle.write(struct.pack(f"<{sequence_length}f", *rollout.behavior_logprobs))
                handle.write(struct.pack("<f", rollout.reward))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="grouped rollout JSONL")
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", type=Path)
    parser.add_argument("--val-records", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3638)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.group_size < 2:
        parser.error("--group-size must be at least two")
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
            encode_record(
                tokenizer, record, args.sequence_length, args.group_size, pad_token
            )
            for record in source
        ]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    split = len(encoded) - args.val_records
    write_dataset(args.train_out, args.sequence_length, args.group_size, encoded[:split])
    print(f"train: {split} GRPO groups -> {args.train_out}")
    if args.val_out is not None:
        write_dataset(args.val_out, args.sequence_length, args.group_size, encoded[split:])
        print(f"val: {args.val_records} GRPO groups -> {args.val_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
