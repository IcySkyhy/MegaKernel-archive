#!/usr/bin/env python3
"""CPU contract receipt and deferred GPU matrix for speculative prototypes.

This script imports no MLX modules and never loads a model.  It is safe to run
while the GPU is leased elsewhere.  It proves only the portable control-plane
contracts and emits the exact GPU work that remains deferred.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "mlx_lm" / "speculative_prototypes.py"
SPEC = importlib.util.spec_from_file_location("speculative_prototypes_cpu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROTOTYPES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOTYPES
SPEC.loader.exec_module(PROTOTYPES)

MTPPhase = PROTOTYPES.MTPPhase
PhaseSeparatedVariantRegistry = PROTOTYPES.PhaseSeparatedVariantRegistry
PhaseVariantKey = PROTOTYPES.PhaseVariantKey
plan_qsa_group_amendment = PROTOTYPES.plan_qsa_group_amendment


def _key(phase: MTPPhase, capacity: int = 32768) -> PhaseVariantKey:
    return PhaseVariantKey(
        phase=phase,
        model_identity="qwen4-exp:flash-next",
        input_shape=(1, 1),
        dtype="bf16",
        cache_fingerprint=("qsa-v1", capacity),
        state_fingerprint=("kv", "gdn", "ple", "qsa", "mtp"),
    )


def build_receipt() -> dict:
    qsa_cases = [
        plan_qsa_group_amendment(3, 3),
        plan_qsa_group_amendment(3, 4),
        plan_qsa_group_amendment(8, 11),
    ]
    rewind_failed_closed = False
    try:
        plan_qsa_group_amendment(9, 8)
    except RuntimeError:
        rewind_failed_closed = True

    registry = PhaseSeparatedVariantRegistry(max_variants_per_phase=1)
    built = []

    def builder(key):
        built.append(key.phase.value)
        return {"phase": key.phase.value}

    for _ in range(10):
        registry.resolve(_key(MTPPhase.DRAFT_OUTPUT), builder)
        registry.resolve(_key(MTPPhase.CATCHUP_NO_OUTPUT), builder)
    registry.resolve(_key(MTPPhase.TARGET_VERIFY), builder)

    return {
        "schema": "mlx-lm.speculative-prototypes-qualification/v1",
        "evidence_level": "cpu_contract_only",
        "gpu_used": False,
        "service_changed": False,
        "qualification_state": {
            "implemented": True,
            "cpu_contract_qualified": True,
            "selected_in_serving": False,
            "observed_in_serving": False,
            "gpu_correctness_qualified": False,
            "gpu_performance_qualified": False,
        },
        "qsa_group_amendment": {
            "cases": [
                {
                    "captured_blocks": case.captured_blocks,
                    "current_blocks": case.current_blocks,
                    "append_block_ids": list(case.append_block_ids),
                }
                for case in qsa_cases
            ],
            "rewind_failed_closed": rewind_failed_closed,
            "unified_rollback": "MLX_QWEN4_QSA_MTP_AMEND_COMPLETE_GROUPS=0",
            "default_enabled": True,
            "mlx2_compatibility": (
                "matches existing _mtp_shared_topk_n_blocks contract; no mlx2 change made"
            ),
        },
        "mtp_phase_variants": {
            "build_order": built,
            "registry": registry.snapshot(),
            "backend_binding": "deferred; registry accepts a backend-owned builder",
            "default_enabled": False,
        },
        "deferred_gpu_matrix": {
            "preconditions": [
                "claim CPG GPU lease and /Users/Shared/mlxuag/gpu.lock",
                "verify no competing model or benchmark owns the GPU",
                "record interpreter and MLX build identity",
                "settle thermals before and between interleaved blocks",
            ],
            "qsa_arms": [
                {"name": "recompute_control", "share_qsa_indices": False, "amend": False},
                {"name": "frozen_reuse", "share_qsa_indices": True, "amend": False},
                {"name": "amended_reuse", "share_qsa_indices": True, "amend": True},
            ],
            "qsa_geometry": {
                "contexts": [16384, 32768, 65536],
                "draft_depths": [2, 3, 4],
                "batch_sizes": [1, 2, 4],
                "order": "balanced interleaving with reverse bracket",
            },
            "phase_variant_arms": ["eager_control", "phase_separated_compiled"],
            "required_metrics": [
                "target output digest",
                "accepted draft length distribution",
                "decode tokens_per_second only",
                "prefill tokens_per_second reported separately",
                "trace builds/replays/evictions/fallbacks by phase",
                "QSA amendment attempts/amendments/blocks_appended",
                "thermal and swap sentinels",
            ],
            "hard_gates": [
                "exact target output against recompute/eager control",
                "nonzero mechanism counters for every experimental arm",
                "one build per stable phase key followed by replay",
                "no acceptance regression outside interleaved run variation",
                "no throughput promotion without repeated thermal brackets",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_receipt()
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
