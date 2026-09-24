"""Tests for chat decode-kernel selection policy."""

from __future__ import annotations

import pytest

from chat import (
    AUTO_KERNEL_VARIANT,
    SHORT_DECODE_TOKEN_THRESHOLD,
    SUPPORTED_DECODE_KERNEL_VARIANTS,
    select_optimal_kernel_variant,
)


def test_short_decode_prefers_fastmath() -> None:
    assert select_optimal_kernel_variant(1) == "fastmath"
    assert select_optimal_kernel_variant(SHORT_DECODE_TOKEN_THRESHOLD) == "fastmath"


def test_medium_and_long_decode_prefers_blocks128_fastmath() -> None:
    assert select_optimal_kernel_variant(SHORT_DECODE_TOKEN_THRESHOLD + 1) == "blocks128_fastmath"
    assert select_optimal_kernel_variant(500) == "blocks128_fastmath"


def test_kernel_selection_rejects_non_positive_decode_length() -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        select_optimal_kernel_variant(0)

    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        select_optimal_kernel_variant(-1)


def test_supported_kernel_variants_include_auto_and_tuned_options() -> None:
    assert AUTO_KERNEL_VARIANT == "auto"
    assert "baseline" in SUPPORTED_DECODE_KERNEL_VARIANTS
    assert "fastmath" in SUPPORTED_DECODE_KERNEL_VARIANTS
    assert "blocks128_fastmath" in SUPPORTED_DECODE_KERNEL_VARIANTS
