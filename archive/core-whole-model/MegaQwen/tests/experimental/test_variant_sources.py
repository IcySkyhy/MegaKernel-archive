"""Tests for decode experiment source transforms."""

from __future__ import annotations

import pytest

from experimental.variant_sources import (
    build_variant_source,
    list_variants,
    load_base_source,
)


def test_baseline_matches_source() -> None:
    base = load_base_source()
    assert build_variant_source("baseline") == base


def test_fastmath_injects_helpers_and_replaces_expf() -> None:
    src = build_variant_source("fastmath")
    assert "ptx_exp2" in src
    assert "ptx_rcp" in src
    assert "fast_exp(" in src
    assert "expf(" not in src


def test_blocks128_updates_decode_grid_constant() -> None:
    src = build_variant_source("blocks128")
    assert "constexpr int LDG_NUM_BLOCKS = 128;" in src


def test_new_variants_present() -> None:
    variants = list_variants()
    assert "prefetch" in variants
    assert "uint4" in variants
    assert "uint4_prefetch" in variants
    assert "qwen_persistent" in variants
    assert "qwen_persistent_tuned" in variants


def test_prefetch_variant_updates_attention_signature() -> None:
    src = build_variant_source("prefetch")
    assert "const __nv_bfloat16* __restrict__ o_weight" in src
    assert "const __nv_bfloat16* __restrict__ gate_weight" in src
    assert "LDG_PREFETCH_BYTES_PER_IDLE_BLOCK" in src


def test_uint4_variant_uses_uint4_loads() -> None:
    src = build_variant_source("uint4")
    assert "uint4 w_u4" in src
    assert "lane_id * 8" in src


def test_qwen_variant_has_persistent_kernel() -> None:
    src = build_variant_source("qwen_persistent")
    assert "ldg_decode_kernel_persistent" in src
    assert "AtomicGridSync" in src


def test_qwen_tuned_variant_rewrites_launch_macros() -> None:
    src = build_variant_source("qwen_persistent_tuned")
    assert "#define LDG_NUM_BLOCKS 112" in src
    assert "#define LDG_BLOCK_SIZE 256" in src


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant_source("does_not_exist")
