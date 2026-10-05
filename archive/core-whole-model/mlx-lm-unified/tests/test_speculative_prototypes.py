"""CPU-only contract tests: this file must remain importable without MLX."""

import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "mlx_lm" / "speculative_prototypes.py"
SPEC = importlib.util.spec_from_file_location("speculative_prototypes_cpu", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROTOTYPES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOTYPES
SPEC.loader.exec_module(PROTOTYPES)

MTPPhase = PROTOTYPES.MTPPhase
PhaseSeparatedVariantRegistry = PROTOTYPES.PhaseSeparatedVariantRegistry
PhaseVariantKey = PROTOTYPES.PhaseVariantKey
plan_qsa_group_amendment = PROTOTYPES.plan_qsa_group_amendment


def _key(phase, *, capacity=4096):
    return PhaseVariantKey(
        phase=phase,
        model_identity="qwen4-exp:test",
        input_shape=(1, 1),
        dtype="bf16",
        cache_fingerprint=("qsa-v1", capacity),
        state_fingerprint=("kv", "gdn", "ple", "qsa", "mtp"),
        config_epoch=0,
    )


def test_qsa_amendment_names_only_groups_closed_after_capture():
    unchanged = plan_qsa_group_amendment(17, 17)
    assert not unchanged.changed
    assert unchanged.append_block_ids == ()

    amended = plan_qsa_group_amendment(17, 20)
    assert amended.changed
    assert amended.append_block_ids == (17, 18, 19)


@pytest.mark.parametrize("captured,current", [(-1, 0), (0, -1), (True, 1)])
def test_qsa_amendment_rejects_invalid_geometry(captured, current):
    with pytest.raises(ValueError):
        plan_qsa_group_amendment(captured, current)


def test_qsa_amendment_fails_closed_after_rewind():
    with pytest.raises(RuntimeError, match="rewind"):
        plan_qsa_group_amendment(8, 7)


def test_alternating_mtp_phases_do_not_thrash_one_variant_slot():
    registry = PhaseSeparatedVariantRegistry(max_variants_per_phase=1)
    builds = []

    def build(key):
        builds.append(key.phase)
        return f"compiled:{key.phase.value}"

    phases = [MTPPhase.DRAFT_OUTPUT, MTPPhase.CATCHUP_NO_OUTPUT] * 10
    resolutions = [registry.resolve(_key(phase), build) for phase in phases]

    assert [item.built for item in resolutions[:2]] == [True, True]
    assert not any(item.built for item in resolutions[2:])
    assert builds == [MTPPhase.DRAFT_OUTPUT, MTPPhase.CATCHUP_NO_OUTPUT]
    snapshot = registry.snapshot()
    assert snapshot["phases"]["draft_output"] == {
        "live_variants": 1,
        "builds": 1,
        "replays": 9,
        "evictions": 0,
        "build_failures": 0,
    }
    assert snapshot["phases"]["catchup_no_output"] == {
        "live_variants": 1,
        "builds": 1,
        "replays": 9,
        "evictions": 0,
        "build_failures": 0,
    }


def test_eviction_is_local_to_one_phase_arena():
    registry = PhaseSeparatedVariantRegistry(max_variants_per_phase=1)
    builder = lambda key: (key.phase.value, key.cache_fingerprint)

    registry.resolve(_key(MTPPhase.CATCHUP_NO_OUTPUT), builder)
    registry.resolve(_key(MTPPhase.DRAFT_OUTPUT, capacity=4096), builder)
    registry.resolve(_key(MTPPhase.DRAFT_OUTPUT, capacity=8192), builder)

    snapshot = registry.snapshot()["phases"]
    assert snapshot["draft_output"]["evictions"] == 1
    assert snapshot["catchup_no_output"]["evictions"] == 0
    assert snapshot["catchup_no_output"]["live_variants"] == 1


def test_build_failure_and_fallback_are_receipted_without_caching():
    registry = PhaseSeparatedVariantRegistry()
    key = _key(MTPPhase.TARGET_VERIFY)

    def fail(_key):
        raise RuntimeError("trace failed")

    with pytest.raises(RuntimeError, match="trace failed"):
        registry.resolve(key, fail)
    registry.note_fallback(MTPPhase.TARGET_VERIFY, "unsupported_state_layout")
    snapshot = registry.snapshot()
    assert snapshot["phases"]["target_verify"]["live_variants"] == 0
    assert snapshot["phases"]["target_verify"]["build_failures"] == 1
    assert snapshot["fallbacks"] == {
        "target_verify:unsupported_state_layout": 1
    }


def test_unified_qsa_integration_is_default_on_and_lifecycle_bound():
    root = Path(__file__).parents[1]
    qwen = (root / "mlx_lm" / "models" / "qwen4_exp.py").read_text()
    assert "_QSA_MTP_AMEND_COMPLETE_GROUPS = _env_flag(" in qwen
    assert '"MLX_QWEN4_QSA_MTP_AMEND_COMPLETE_GROUPS", default=True' in qwen
    # Definition plus the shared-suffix and ordinary QSA consumers.
    assert qwen.count("_amend_mtp_shared_topk(cache, shared_topk, n_blocks)") == 3
    assert '("_mtp_shared_topk_n_blocks", None)' in qwen
    for relative in (
        "mlx_lm/qsa_shared_suffix.py",
        "mlx_lm/segmented_batch_cache.py",
        "mlx_lm/spomin_qwen4_surgery.py",
        "mlx_lm/models/qwen4_megakernel_runtime.py",
    ):
        assert "_mtp_shared_topk_n_blocks" in (root / relative).read_text()


def test_variant_key_rejects_non_integer_epoch():
    with pytest.raises(ValueError, match="config_epoch"):
        PhaseVariantKey(
            phase=MTPPhase.DRAFT_OUTPUT,
            model_identity="qwen4-exp:test",
            input_shape=(1, 1),
            dtype="bf16",
            cache_fingerprint="cache",
            state_fingerprint="state",
            config_epoch="zero",
        )
