#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test cast codegen - common functions for Kirin9030 and KirinX90
"""

import numpy as np
import pytest
import torch

from kirin.common import check_nan, compare_cos
import pypto
from pypto import CastMode, SaturationMode

_all_f16_uint16 = np.arange(0x0000, 0x10000, dtype=np.uint16)
_all_f16_values = _all_f16_uint16.view(np.float16)
finite_mask = ~np.isnan(_all_f16_values)
_all_f16_finite = _all_f16_values[finite_mask]
_all_f16_sorted = np.sort(_all_f16_finite)


def _round_rint(low, high, dist_low, dist_high, x):
    if dist_low == dist_high:
        low_int = low.view(np.uint16)
        _high_int = high.view(np.uint16)
        return low if (low_int & 1) == 0 else high
    return low if dist_low < dist_high else high


def _round_away(low, high, dist_low, dist_high, x):
    if dist_low == dist_high:
        return high if abs(high) >= abs(low) else low
    return low if dist_low < dist_high else high


def _round_floor(low, high, dist_low, dist_high, x):
    return low


def _round_ceil(low, high, dist_low, dist_high, x):
    return high


def _round_trunc(low, high, dist_low, dist_high, x):
    if x >= 0:
        return low if low >= 0 else high
    return high if high <= 0 else low


def _round_odd(low, high, dist_low, dist_high, x):
    if dist_low == dist_high:
        low_int = low.view(np.uint16)
        _high_int = high.view(np.uint16)
        return low if (low_int & 1) == 1 else high
    return low if dist_low < dist_high else high


_FP16_ROUND_DISPATCH = {
    "CAST_NONE": _round_rint,
    "CAST_RINT": _round_rint,
    "CAST_ROUND": _round_away,
    "CAST_FLOOR": _round_floor,
    "CAST_CEIL": _round_ceil,
    "CAST_TRUNC": _round_trunc,
    "CAST_ODD": _round_odd,
}


def _resolve_nearest_fp16(x):
    pos = np.searchsorted(_all_f16_sorted, x)
    if pos == 0:
        return _all_f16_sorted[0], _all_f16_sorted[0]
    if pos == len(_all_f16_sorted):
        return _all_f16_sorted[-1], _all_f16_sorted[-1]
    return _all_f16_sorted[pos - 1], _all_f16_sorted[pos]


def round_fp32_to_fp16(x, mode):
    x = np.asarray(x, dtype=np.float64)
    if np.isnan(x) or np.isinf(x):
        return np.float16(x).astype(np.float64)

    low, high = _resolve_nearest_fp16(x)
    if low == x or high == x:
        return x

    dist_low = x - low
    dist_high = high - x

    round_fn = _FP16_ROUND_DISPATCH.get(mode)
    if round_fn is None:
        raise ValueError(f"Unsupported mode {mode}")
    return round_fn(low, high, dist_low, dist_high, x)


def round_half_away_from_zero(x):
    abs_x = np.abs(x)
    sign = np.sign(x)
    fractional = abs_x - np.floor(abs_x)
    rounded_abs = np.where(fractional >= 0.5, np.ceil(abs_x), np.floor(abs_x))
    return sign * rounded_abs


def cast_odd_round(x):
    x = np.asarray(x, dtype=np.float64)
    frac = x - np.floor(x)
    is_half = np.abs(frac - 0.5) < 1e-9
    floor_val = np.floor(x)
    result = np.where(is_half, np.where((floor_val.astype(np.int64) % 2) != 0, floor_val, floor_val + 1), np.round(x))
    return result


_DTYPE_NUMPY_MAP = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "uint8": np.uint8,
}

_INT_CAST_DISPATCH = {
    "CAST_NONE": lambda x: x,
    "CAST_RINT": np.rint,
    "CAST_ROUND": round_half_away_from_zero,
    "CAST_FLOOR": np.floor,
    "CAST_CEIL": np.ceil,
    "CAST_TRUNC": np.trunc,
    "CAST_ODD": cast_odd_round,
}


def dtype_str_to_numpy(dtype_str):
    if dtype_str not in _DTYPE_NUMPY_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(_DTYPE_NUMPY_MAP.keys())}")
    return _DTYPE_NUMPY_MAP[dtype_str]


def _resolve_cast_none_mode(in_dtype, out_np_dtype):
    if np.issubdtype(in_dtype, np.floating) and np.issubdtype(out_np_dtype, np.integer):
        return "CAST_TRUNC"
    if in_dtype == np.float32 and out_np_dtype == np.float16:
        return "CAST_RINT"
    return "CAST_NONE"


def _check_cast_odd_valid(in_dtype, out_np_dtype):
    if not (in_dtype == np.float32 and out_np_dtype == np.float16):
        raise ValueError(f"CAST_ODD only supports float32 -> float16, got {in_dtype} -> {out_np_dtype}")


def _cast_to_integer(float_arr, effective_mode, out_np_dtype, saturation_mode):
    cast_fn = _INT_CAST_DISPATCH.get(effective_mode)
    if cast_fn is None:
        raise ValueError(f"Unknown mode: {effective_mode}")
    converted = cast_fn(float_arr)
    if saturation_mode == "ON" and np.issubdtype(out_np_dtype, np.integer):
        info = np.iinfo(out_np_dtype)
        converted = np.clip(converted, info.min, info.max)
    return converted


def apply_cast_mode(arr, mode, out_dtype_str, saturation_mode):
    in_dtype = arr.dtype
    out_np_dtype = dtype_str_to_numpy(out_dtype_str)

    if mode == "CAST_ODD":
        _check_cast_odd_valid(in_dtype, out_np_dtype)

    effective_mode = _resolve_cast_none_mode(in_dtype, out_np_dtype) if mode == "CAST_NONE" else mode

    if out_np_dtype == np.float16:
        float_arr = arr.astype(np.float64)
        vround = np.vectorize(lambda v: round_fp32_to_fp16(v, effective_mode), otypes=[np.float64])
        converted = vround(float_arr)
    elif np.issubdtype(out_np_dtype, np.floating):
        converted = arr.astype(np.float64)
    else:
        float_arr = arr.astype(np.float64)
        converted = _cast_to_integer(float_arr, effective_mode, out_np_dtype, saturation_mode)

    return converted.astype(out_np_dtype)


def torch_cast_with_mode(input_tensor, out_dtype, cast_mode, sat_mode):
    input_np = input_tensor.detach().cpu().numpy()
    mode_str = cast_mode.name
    sat_str = sat_mode.name
    out_dtype_str = out_dtype.name.lower().replace("dt_", "")
    output_np = apply_cast_mode(input_np, mode_str, out_dtype_str, sat_str)
    return torch.from_numpy(output_np).to(input_tensor.device)


def generate_input(shape, dtype_str, device="cpu"):
    if dtype_str == "FP16":
        return torch.rand(shape, dtype=torch.float16, device=device) * 10 - 5
    elif dtype_str == "FP32":
        return torch.rand(shape, dtype=torch.float32, device=device) * 10 - 5
    elif dtype_str == "INT8":
        return torch.randint(-100, 100, shape, dtype=torch.int8, device=device)
    elif dtype_str == "UINT8":
        return torch.randint(0, 200, shape, dtype=torch.uint8, device=device)
    elif dtype_str == "INT16":
        return torch.randint(-10000, 10000, shape, dtype=torch.int16, device=device)
    elif dtype_str == "INT32":
        return torch.randint(-100000, 100000, shape, dtype=torch.int32, device=device)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


_CAST_MODE_MAP = {
    "CAST_NONE": CastMode.CAST_NONE,
    "CAST_RINT": CastMode.CAST_RINT,
    "CAST_ROUND": CastMode.CAST_ROUND,
    "CAST_FLOOR": CastMode.CAST_FLOOR,
    "CAST_CEIL": CastMode.CAST_CEIL,
    "CAST_TRUNC": CastMode.CAST_TRUNC,
    "CAST_ODD": CastMode.CAST_ODD,
}

_SAT_MODE_MAP = {
    "ON": SaturationMode.ON,
    "OFF": SaturationMode.OFF,
}

_DTYPE_MAP = {
    "FP16": (pypto.DT_FP16, torch.float16),
    "FP32": (pypto.DT_FP32, torch.float32),
    "INT8": (pypto.DT_INT8, torch.int8),
    "INT16": (pypto.DT_INT16, torch.int16),
    "INT32": (pypto.DT_INT32, torch.int32),
    "UINT8": (pypto.DT_UINT8, torch.uint8),
}


def make_cast_kernel(soc_version, name, in_dtype_str, out_dtype_str, tile_shapes, cast_mode_str, sat_mode_str):
    in_dtype_pypto, in_dtype_torch = _DTYPE_MAP[in_dtype_str]
    out_dtype_pypto, out_dtype_torch = _DTYPE_MAP[out_dtype_str]
    cast_mode = _CAST_MODE_MAP[cast_mode_str]
    sat_mode = _SAT_MODE_MAP[sat_mode_str]

    @pypto.frontend.jit(codegen_options={"soc_version": soc_version}, runtime_options={"run_mode": pypto.RunMode.SIM})
    def kernel(
        a: pypto.Tensor([...], in_dtype_pypto),
        out: pypto.Tensor([...], out_dtype_pypto),
    ):
        pypto.set_vec_tile_shapes(*tile_shapes)
        out[:] = pypto.cast(a, out_dtype_pypto, cast_mode, sat_mode)

    kernel.__name__ = name
    return kernel


TEST_CASES = [
    # kernel_name: name of the kernel
    # in_dtype_str: source data type string
    # out_dtype_str: destination data type string
    # tile_shapes: tile shape for pypto kernel
    # shape: tensor shape
    # cast_mode: rounding mode
    # sat_mode: saturation mode
    # marks: pytest marks
    pytest.param("cast_kernel_001", "FP16", "FP32", (50,), (112,), "CAST_NONE", "OFF", marks=[], id="001"),
    pytest.param(
        "cast_kernel_002", "INT32", "FP32", (100,), (100,), "CAST_RINT", "ON", marks=[pytest.mark.skip()], id="002"
    ),
    pytest.param(
        "cast_kernel_003", "INT16", "FP32", (2, 32), (4, 128), "CAST_ROUND", "ON", marks=[pytest.mark.skip()], id="003"
    ),
    pytest.param(
        "cast_kernel_004", "FP32", "FP16", (1, 130), (4, 130), "CAST_ODD", "OFF", marks=[pytest.mark.skip()], id="004"
    ),
    pytest.param(
        "cast_kernel_005",
        "INT8",
        "FP16",
        (1, 2, 32),
        (2, 4, 160),
        "CAST_CEIL",
        "OFF",
        marks=[pytest.mark.skip()],
        id="005",
    ),
    pytest.param(
        "cast_kernel_006",
        "UINT8",
        "FP16",
        (1, 2, 140),
        (2, 4, 140),
        "CAST_TRUNC",
        "OFF",
        marks=[pytest.mark.skip()],
        id="006",
    ),
    pytest.param(
        "cast_kernel_007",
        "INT16",
        "FP16",
        (1, 5, 32),
        (2, 5, 152),
        "CAST_FLOOR",
        "OFF",
        marks=[pytest.mark.skip()],
        id="007",
    ),
    pytest.param(
        "cast_kernel_008",
        "FP32",
        "INT16",
        (1, 3, 170),
        (2, 3, 170),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="008",
    ),
    pytest.param(
        "cast_kernel_009",
        "FP32",
        "INT32",
        (2, 1, 2, 16),
        (5, 2, 4, 176),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="009",
    ),
    pytest.param(
        "cast_kernel_010",
        "FP16",
        "INT8",
        (1, 1, 1, 130),
        (5, 2, 4, 130),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="010",
    ),
    pytest.param(
        "cast_kernel_011",
        "FP16",
        "UINT8",
        (1, 1, 5, 32),
        (2, 3, 5, 134),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="011",
    ),
    pytest.param(
        "cast_kernel_012",
        "FP16",
        "INT16",
        (2, 2, 3, 32),
        (4, 2, 6, 135),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="012",
    ),
    pytest.param(
        "cast_kernel_013",
        "FP16",
        "INT32",
        (1, 1, 4, 130),
        (6, 2, 4, 130),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="013",
    ),
    pytest.param(
        "cast_kernel_014",
        "FP16",
        "FP32",
        (1, 2, 1, 139),
        (3, 2, 3, 139),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="014",
    ),
    pytest.param(
        "cast_kernel_015",
        "FP16",
        "FP32",
        (3, 3, 5, 32),
        (6, 3, 5, 141),
        "CAST_NONE",
        "OFF",
        marks=[pytest.mark.skip()],
        id="015",
    ),
    pytest.param(
        "cast_kernel_016",
        "FP32",
        "INT16",
        (1, 3, 32),
        (2, 3, 160),
        "CAST_ROUND",
        "OFF",
        marks=[pytest.mark.skip()],
        id="016",
    ),
    pytest.param(
        "cast_kernel_017",
        "FP32",
        "INT16",
        (2, 2, 64),
        (3, 4, 128),
        "CAST_FLOOR",
        "OFF",
        marks=[pytest.mark.skip()],
        id="017",
    ),
    pytest.param(
        "cast_kernel_018",
        "FP32",
        "INT16",
        (1, 5, 144),
        (2, 5, 144),
        "CAST_CEIL",
        "OFF",
        marks=[pytest.mark.skip()],
        id="018",
    ),
    pytest.param(
        "cast_kernel_019",
        "FP32",
        "FP16",
        (1, 2, 160),
        (2, 4, 160),
        "CAST_ROUND",
        "OFF",
        marks=[pytest.mark.skip()],
        id="019",
    ),
    pytest.param(
        "cast_kernel_020",
        "FP32",
        "FP16",
        (2, 5, 136),
        (3, 5, 136),
        "CAST_CEIL",
        "OFF",
        marks=[pytest.mark.skip()],
        id="020",
    ),
    pytest.param(
        "cast_kernel_021",
        "FP32",
        "FP16",
        (2, 3, 148),
        (4, 6, 148),
        "CAST_ODD",
        "OFF",
        marks=[pytest.mark.skip()],
        id="021",
    ),
]


def run_cast_test(kernels, kernel_name, in_dtype_str, out_dtype_str, shape, cast_mode_str, sat_mode_str):
    """Run a single cast kernel test with given kernels dict."""
    in_dtype_pypto, in_dtype_torch = _DTYPE_MAP[in_dtype_str]
    out_dtype_pypto, out_dtype_torch = _DTYPE_MAP[out_dtype_str]
    cast_mode = _CAST_MODE_MAP[cast_mode_str]
    sat_mode = _SAT_MODE_MAP[sat_mode_str]

    a = generate_input(shape, in_dtype_str)
    out = torch.empty(shape, dtype=out_dtype_torch)

    kernels[kernel_name](a, out)

    golden = torch_cast_with_mode(a, out_dtype_pypto, cast_mode, sat_mode)

    check_nan(out, name=kernel_name)
    if out_dtype_torch in (torch.int8, torch.int16, torch.int32, torch.uint8):
        np.testing.assert_array_equal(out.cpu(), golden.cpu())
    else:
        cos_val = abs(compare_cos(out.numpy(), golden.numpy()))
        if cos_val < 0.9999:
            raise AssertionError(f"{kernel_name}: cos_val {cos_val} < 0.9999")


def create_test_cast_module(soc_version):
    """Create a test module for cast with specified soc_version."""
    kernels = {
        p.values[0]: make_cast_kernel(
            soc_version, p.values[0], p.values[1], p.values[2], p.values[3], p.values[5], p.values[6]
        )
        for p in TEST_CASES
    }
    return kernels, lambda: run_cast_test(kernels, None, None, None, None, None, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
