#!/usr/bin/env python3
# Copyright (c) 2026 Ranvier Systems. All rights reserved.
# Released under Apache 2.0 license as described in the file LICENSE.
# Authors: Christian Pehle
"""Deterministic NumPy oracle for mixed-batch Qwen3.6 LoRA projection gates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SEED = 0xC0FFEE
BATCH = 4
SEQUENCE = 2
ROWS = BATCH * SEQUENCE
ADAPTERS = 2
RANK = 16
INPUT = 16
OUTPUT = 16
ALPHA = np.float32(16.0)
SCALE = np.float32(ALPHA / np.float32(RANK))
NO_ADAPTER = np.uint32(0xFFFFFFFF)


def projection(
    x: np.ndarray,
    base: np.ndarray,
    adapter_a: np.ndarray,
    adapter_b: np.ndarray,
    adapter_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank_activation = np.zeros((x.shape[0], RANK), dtype=np.float32)
    output = (x @ base.T).astype(np.float32)
    for row in range(x.shape[0]):
        adapter = int(adapter_ids[row // SEQUENCE])
        if adapter < ADAPTERS:
            rank_activation[row] = x[row] @ adapter_a[adapter].T
            output[row] += SCALE * (rank_activation[row] @ adapter_b[adapter].T)
    return rank_activation.astype(np.float32), output.astype(np.float32)


def vjp(
    x: np.ndarray,
    base: np.ndarray,
    adapter_a: np.ndarray,
    adapter_b: np.ndarray,
    adapter_ids: np.ndarray,
    rank_activation: np.ndarray,
    output_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_gradient = (output_gradient @ base).astype(np.float32)
    a_gradient = np.zeros_like(adapter_a)
    b_gradient = np.zeros_like(adapter_b)
    for row in range(x.shape[0]):
        adapter = int(adapter_ids[row // SEQUENCE])
        if adapter < ADAPTERS:
            rank_gradient = (SCALE * (output_gradient[row] @ adapter_b[adapter])).astype(
                np.float32
            )
            input_gradient[row] += rank_gradient @ adapter_a[adapter]
            a_gradient[adapter] += np.outer(rank_gradient, x[row]).astype(np.float32)
            b_gradient[adapter] += (
                SCALE * np.outer(output_gradient[row], rank_activation[row])
            ).astype(np.float32)
    return (
        input_gradient.astype(np.float32),
        a_gradient.astype(np.float32),
        b_gradient.astype(np.float32),
    )


def adamw(
    parameter: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    learning_rate = np.float32(2.0e-3)
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    epsilon = np.float32(1.0e-8)
    weight_decay = np.float32(0.01)
    next_first = (beta1 * first + (np.float32(1.0) - beta1) * gradient).astype(np.float32)
    next_second = (
        beta2 * second + (np.float32(1.0) - beta2) * gradient * gradient
    ).astype(np.float32)
    corrected_first = (next_first * np.float32(10.0)).astype(np.float32)
    corrected_second = (next_second * np.float32(1000.0)).astype(np.float32)
    normalized = (
        corrected_first / (np.sqrt(corrected_second).astype(np.float32) + epsilon)
    ).astype(np.float32)
    updated = (
        parameter - learning_rate * (normalized + weight_decay * parameter)
    ).astype(np.float32)
    return updated, next_first, next_second


def build() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 0.15, (ROWS, INPUT)).astype(np.float32)
    base = rng.normal(0.0, 0.12, (OUTPUT, INPUT)).astype(np.float32)
    adapter_a = rng.normal(0.0, 0.08, (ADAPTERS, RANK, INPUT)).astype(np.float32)
    adapter_b = rng.normal(0.0, 0.08, (ADAPTERS, OUTPUT, RANK)).astype(np.float32)
    output_gradient = rng.normal(0.0, 0.1, (ROWS, OUTPUT)).astype(np.float32)
    adapter_ids = np.asarray([0, 1, 0, 1], dtype=np.uint32)

    rank_activation, output = projection(x, base, adapter_a, adapter_b, adapter_ids)
    input_gradient, a_gradient, b_gradient = vjp(
        x, base, adapter_a, adapter_b, adapter_ids, rank_activation, output_gradient
    )

    masked_output_gradient = output_gradient.copy()
    for batch_row, adapter in enumerate(adapter_ids):
        if adapter == 1:
            start = batch_row * SEQUENCE
            masked_output_gradient[start : start + SEQUENCE] = 0
    masked_input_gradient, masked_a_gradient, masked_b_gradient = vjp(
        x,
        base,
        adapter_a,
        adapter_b,
        adapter_ids,
        rank_activation,
        masked_output_gradient,
    )

    a_first = rng.normal(0.0, 0.01, adapter_a.shape).astype(np.float32)
    a_second = np.abs(rng.normal(0.0, 0.005, adapter_a.shape)).astype(np.float32)
    b_first = rng.normal(0.0, 0.01, adapter_b.shape).astype(np.float32)
    b_second = np.abs(rng.normal(0.0, 0.005, adapter_b.shape)).astype(np.float32)
    updated_a = adapter_a.copy()
    updated_b = adapter_b.copy()
    updated_a_first = a_first.copy()
    updated_a_second = a_second.copy()
    updated_b_first = b_first.copy()
    updated_b_second = b_second.copy()
    (
        updated_a[0],
        updated_a_first[0],
        updated_a_second[0],
    ) = adamw(adapter_a[0], masked_a_gradient[0], a_first[0], a_second[0])
    (
        updated_b[0],
        updated_b_first[0],
        updated_b_second[0],
    ) = adamw(adapter_b[0], masked_b_gradient[0], b_first[0], b_second[0])

    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    row_permutation = np.concatenate(
        [np.arange(item * SEQUENCE, (item + 1) * SEQUENCE) for item in permutation]
    )
    permuted_x = x[row_permutation]
    permuted_gradient = output_gradient[row_permutation]
    permuted_ids = adapter_ids[permutation]
    permuted_rank, permuted_output = projection(
        permuted_x, base, adapter_a, adapter_b, permuted_ids
    )
    permuted_dx, permuted_da, permuted_db = vjp(
        permuted_x,
        base,
        adapter_a,
        adapter_b,
        permuted_ids,
        permuted_rank,
        permuted_gradient,
    )

    sentinel_ids = np.asarray([NO_ADAPTER, 1, 0, NO_ADAPTER], dtype=np.uint32)
    sentinel_rank, sentinel_output = projection(x, base, adapter_a, adapter_b, sentinel_ids)

    return {
        "input": x,
        "base": base,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "adapter_ids": adapter_ids,
        "rank_activation": rank_activation,
        "output": output,
        "output_gradient": output_gradient,
        "input_gradient": input_gradient,
        "adapter_a_gradient": a_gradient,
        "adapter_b_gradient": b_gradient,
        "masked_output_gradient": masked_output_gradient,
        "masked_input_gradient": masked_input_gradient,
        "masked_adapter_a_gradient": masked_a_gradient,
        "masked_adapter_b_gradient": masked_b_gradient,
        "adapter_a_first": a_first,
        "adapter_a_second": a_second,
        "adapter_b_first": b_first,
        "adapter_b_second": b_second,
        "update_mask": np.asarray([1, 0], dtype=np.uint32),
        "updated_adapter_a": updated_a,
        "updated_adapter_b": updated_b,
        "updated_adapter_a_first": updated_a_first,
        "updated_adapter_a_second": updated_a_second,
        "updated_adapter_b_first": updated_b_first,
        "updated_adapter_b_second": updated_b_second,
        "permuted_input": permuted_x,
        "permuted_output_gradient": permuted_gradient,
        "permuted_adapter_ids": permuted_ids,
        "permuted_rank_activation": permuted_rank,
        "permuted_output": permuted_output,
        "permuted_input_gradient": permuted_dx,
        "permuted_adapter_a_gradient": permuted_da,
        "permuted_adapter_b_gradient": permuted_db,
        "sentinel_adapter_ids": sentinel_ids,
        "sentinel_rank_activation": sentinel_rank,
        "sentinel_output": sentinel_output,
    }


def filename(name: str, value: np.ndarray) -> str:
    dims = "x".join(str(size) for size in value.shape)
    suffix = "u32" if value.dtype == np.uint32 else "f32"
    return f"lora.{name}.{dims}.{suffix}.bin"


def dump(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for name, value in build().items():
        normalized = np.asarray(
            value, dtype="<u4" if value.dtype == np.uint32 else "<f4"
        )
        path = output / filename(name, normalized)
        payload = normalized.tobytes(order="C")
        path.write_bytes(payload)
        manifest[name] = {
            "shape": list(normalized.shape),
            "dtype": "u32" if normalized.dtype == np.dtype("<u4") else "f32",
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compare(left: Path, right: Path) -> None:
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    if left_files != right_files:
        raise SystemExit("LoRA fixture file sets differ")
    for relative in left_files:
        if (left / relative).read_bytes() != (right / relative).read_bytes():
            raise SystemExit(f"LoRA fixture differs: {relative}")


def selftest() -> None:
    fixtures = build()
    adapter_one_a = fixtures["updated_adapter_a"][1]
    adapter_one_b = fixtures["updated_adapter_b"][1]
    np.testing.assert_array_equal(adapter_one_a, fixtures["adapter_a"][1])
    np.testing.assert_array_equal(adapter_one_b, fixtures["adapter_b"][1])
    np.testing.assert_array_equal(
        fixtures["sentinel_rank_activation"][:SEQUENCE], np.zeros((SEQUENCE, RANK), np.float32)
    )
    np.testing.assert_allclose(
        fixtures["sentinel_output"][:SEQUENCE],
        fixtures["input"][:SEQUENCE] @ fixtures["base"].T,
        rtol=0,
        atol=2e-7,
    )
    print("Qwen3.6 mixed-adapter LoRA NumPy oracle self-test passed")


def main(argv: list[str]) -> None:
    if len(argv) == 2 and argv[0] == "dump":
        dump(Path(argv[1]))
    elif len(argv) == 3 and argv[0] == "compare":
        compare(Path(argv[1]), Path(argv[2]))
    elif len(argv) == 1 and argv[0] == "selftest":
        selftest()
    else:
        raise SystemExit("usage: lora_oracle.py {selftest|dump OUT|compare LEFT RIGHT}")


if __name__ == "__main__":
    main(sys.argv[1:])
