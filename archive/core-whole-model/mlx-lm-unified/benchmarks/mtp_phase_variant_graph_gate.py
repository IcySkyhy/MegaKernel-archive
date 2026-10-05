#!/usr/bin/env python3
"""Real-MLX graph gate for phase-separated speculative variant arenas.

This is deliberately a component gate, not a model throughput claim.  It uses
three compiled functions with identical input signatures but different output
contracts, alternates them repeatedly, and requires exact eager parity plus one
stable build per phase followed only by registry replays.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from mlx_lm.speculative_prototypes import (
    MTPPhase,
    PhaseSeparatedVariantRegistry,
    PhaseVariantKey,
)


def _draft_output(token, state):
    next_state = state + token + mx.array(1.0, dtype=state.dtype)
    return next_state * mx.array(2.0, dtype=state.dtype), next_state


def _catchup_no_output(token, state):
    return state + token + mx.array(3.0, dtype=state.dtype)


def _target_verify(token, state):
    next_state = state + token + mx.array(5.0, dtype=state.dtype)
    logits = mx.stack((next_state, -next_state), axis=-1)
    return logits, next_state


EAGER = {
    MTPPhase.DRAFT_OUTPUT: _draft_output,
    MTPPhase.CATCHUP_NO_OUTPUT: _catchup_no_output,
    MTPPhase.TARGET_VERIFY: _target_verify,
}


def _key(phase: MTPPhase, width: int) -> PhaseVariantKey:
    return PhaseVariantKey(
        phase=phase,
        model_identity="component:explicit-state",
        input_shape=(width,),
        dtype="float32",
        cache_fingerprint=("explicit-vector", width),
        state_fingerprint=("phase-output-contract", phase.value),
    )


def _arrays(value):
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, tuple):
        out = []
        for item in value:
            out.extend(_arrays(item))
        return out
    raise TypeError(type(value))


def _next_state(phase: MTPPhase, result):
    return result if phase is MTPPhase.CATCHUP_NO_OUTPUT else result[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 2 or args.width < 1:
        raise ValueError("cycles >= 2 and width >= 1 are required")

    registry = PhaseSeparatedVariantRegistry(max_variants_per_phase=1)
    trace_builds = []

    def build(key):
        trace_builds.append(key.phase.value)
        return mx.compile(EAGER[key.phase])

    compiled_state = mx.zeros((args.width,), dtype=mx.float32)
    eager_state = mx.zeros((args.width,), dtype=mx.float32)
    token = mx.arange(args.width, dtype=mx.float32) / max(args.width, 1)
    phases = (
        MTPPhase.DRAFT_OUTPUT,
        MTPPhase.CATCHUP_NO_OUTPUT,
        MTPPhase.TARGET_VERIFY,
    )
    max_abs_error = 0.0
    started = time.perf_counter()
    for _cycle in range(args.cycles):
        for phase in phases:
            resolution = registry.resolve(_key(phase, args.width), build)
            compiled = resolution.value(token, compiled_state)
            eager = EAGER[phase](token, eager_state)
            compiled_arrays = _arrays(compiled)
            eager_arrays = _arrays(eager)
            mx.eval(*compiled_arrays, *eager_arrays)
            for actual, expected in zip(compiled_arrays, eager_arrays):
                error = float(mx.max(mx.abs(actual - expected)).item())
                max_abs_error = max(max_abs_error, error)
            compiled_state = _next_state(phase, compiled)
            eager_state = _next_state(phase, eager)
    elapsed = time.perf_counter() - started
    mx.eval(compiled_state, eager_state)

    receipt = registry.snapshot()
    if max_abs_error != 0.0:
        raise RuntimeError(f"compiled/eager drift: {max_abs_error}")
    for phase in phases:
        row = receipt["phases"][phase.value]
        if row["builds"] != 1 or row["replays"] != args.cycles - 1:
            raise RuntimeError(f"unstable {phase.value} arena: {row}")
        if row["evictions"] or row["build_failures"]:
            raise RuntimeError(f"failed {phase.value} arena: {row}")

    payload = {
        "schema": "mlx-uag.mtp-phase-variant-graph-gate/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence_level": "real_mlx_component_graph",
        "model_graph_qualified": False,
        "mlx_version": importlib.metadata.version("mlx"),
        "cycles": args.cycles,
        "width": args.width,
        "elapsed_s": elapsed,
        "phase_calls_per_second": args.cycles * len(phases) / elapsed,
        "max_abs_error": max_abs_error,
        "trace_build_order": trace_builds,
        "registry": receipt,
        "integration_note": (
            "Current unified self-MTP invokes model.mtp_step as an output-producing "
            "draft step and folds teacher-forcing catch-up into the next draft input; "
            "there is no separate no-output model graph to bind yet."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
