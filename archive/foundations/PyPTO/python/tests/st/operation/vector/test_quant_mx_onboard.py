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
import math
import os
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

from framework.tests.st.operation.python.vector_operator_golden import (
    _compute_scalings_from_exponents,
    _compute_scalings_from_exponents_math,
    _compute_shared_exponents_floor,
    _compute_shared_exponents_nv,
    _encode_e2m1_vectorized,
    _encode_e4m3_fn_vectorized,
    _pack_fp4_e2m1x2_low_first,
)
import pypto

pytestmark = pytest.mark.soc("950")

_FP8_E4M3_TARGET_MAX_POW2 = 8
_FP4_E2M1_MAX_POS = 6.0
_QUANT_MX_GROUP_COLS = 32
_QUANT_MX_SCALE_GROUP_COLS = 64


def _to_bfloat16_float32(values: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).to(torch.float32).numpy()


class _QuantMXGroups(NamedTuple):
    x: np.ndarray
    x_grouped: np.ndarray
    rows: int
    cols: int
    group_cols: int
    scale_group_cols: int
    padded_cols: int


class _QuantMXOnboardCase(NamedTuple):
    input_data: torch.Tensor
    quant_output: torch.Tensor
    scale_output: torch.Tensor
    input_shape: list
    view_shape: list
    tile_shape: list
    scale_shape: list
    input_dtype: Any
    quant_dtype: Any
    round_mode: Any


def _prepare_quant_mx_groups(input_tensor: torch.Tensor) -> _QuantMXGroups:
    x = input_tensor.cpu().numpy().astype(np.float32, copy=False)
    cols = x.shape[-1]
    rows = x.size // cols
    group_cols = (cols + _QUANT_MX_GROUP_COLS - 1) // _QUANT_MX_GROUP_COLS
    scale_group_cols = (cols + _QUANT_MX_SCALE_GROUP_COLS - 1) // _QUANT_MX_SCALE_GROUP_COLS
    x_flat = x.reshape(rows, cols)
    padded_cols = group_cols * _QUANT_MX_GROUP_COLS
    x_padded = np.zeros((rows, padded_cols), dtype=np.float32)
    x_padded[:, :cols] = x_flat
    x_grouped = x_padded.reshape(rows, group_cols, _QUANT_MX_GROUP_COLS)
    return _QuantMXGroups(
        x=x,
        x_grouped=x_grouped,
        rows=rows,
        cols=cols,
        group_cols=group_cols,
        scale_group_cols=scale_group_cols,
        padded_cols=padded_cols,
    )


def _restore_quant_shape(
    quant_grouped: np.ndarray, x: np.ndarray, rows: int, cols: int, padded_cols: int
) -> np.ndarray:
    return quant_grouped.reshape(rows, padded_cols)[:, :cols].reshape(x.shape)


def _build_scale_bytes(
    x: np.ndarray, e8m0: np.ndarray, rows: int, group_cols: int, scale_group_cols: int
) -> torch.Tensor:
    scale_shape = list(x.shape[:-1]) + [scale_group_cols, 2]
    scale = np.zeros(scale_shape, dtype=np.uint8)
    scale.reshape(rows, scale_group_cols * 2)[:, :group_cols] = e8m0.reshape(rows, group_cols)
    return torch.from_numpy(scale.copy())


def _quant_mx_golden_bytes(input_tensor: torch.Tensor):
    groups = _prepare_quant_mx_groups(input_tensor)

    max_abs = np.max(np.abs(groups.x_grouped), axis=2).astype(np.float32)
    e8m0 = _compute_shared_exponents_floor(max_abs, _FP8_E4M3_TARGET_MAX_POW2)
    group_scaling = _compute_scalings_from_exponents(e8m0)
    quant_grouped = _encode_e4m3_fn_vectorized(groups.x_grouped * group_scaling[:, :, np.newaxis])

    quant = _restore_quant_shape(quant_grouped, groups.x, groups.rows, groups.cols, groups.padded_cols)
    return torch.from_numpy(quant.copy()), _build_scale_bytes(
        groups.x, e8m0, groups.rows, groups.group_cols, groups.scale_group_cols
    )


def _quant_mx_e2m1_nv_golden_bytes(input_tensor: torch.Tensor):
    groups = _prepare_quant_mx_groups(input_tensor)

    max_source = np.abs(groups.x_grouped.astype(np.float16)).astype(np.float32)
    max_abs = np.max(max_source, axis=2).astype(np.float32)
    e8m0 = _compute_shared_exponents_nv(max_abs, _FP4_E2M1_MAX_POS)
    group_scaling = _compute_scalings_from_exponents_math(e8m0)
    scaling_bf16 = _to_bfloat16_float32(group_scaling)
    scaled = groups.x_grouped * scaling_bf16[:, :, np.newaxis]
    quant_grouped = _encode_e2m1_vectorized(scaled)

    quant = _restore_quant_shape(quant_grouped, groups.x, groups.rows, groups.cols, groups.padded_cols)
    quant = _pack_fp4_e2m1x2_low_first(quant)
    return torch.from_numpy(quant.copy()), _build_scale_bytes(
        groups.x, e8m0, groups.rows, groups.group_cols, groups.scale_group_cols
    )


def _run_quant_mx_onboard(case: _QuantMXOnboardCase) -> None:
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)

    input_tensor = pypto.tensor(case.input_shape, case.input_dtype, "PTO_TENSOR_input")
    quant_tensor = pypto.tensor(case.input_shape, case.quant_dtype, "PTO_TENSOR_quant")
    scale_tensor = pypto.tensor(case.scale_shape, pypto.DT_FP8E8M0, "PTO_TENSOR_scale")

    pypto.runtime._device_init()
    try:
        with pypto.function("MAIN", input_tensor, quant_tensor, scale_tensor):
            for row_idx in pypto.loop(case.input_shape[0] // case.view_shape[0], name="LOOP_ROW", idx_name="row_idx"):
                pypto.set_vec_tile_shapes(*case.tile_shape)
                input_offset = [row_idx * case.view_shape[0], 0]
                scale_offset = [row_idx * case.view_shape[0], 0, 0]
                input_view = pypto.view(input_tensor, case.view_shape, input_offset)
                quant_view, scale_view = pypto.quant_mx(
                    input_view,
                    case.quant_dtype,
                    case.round_mode,
                    -1,
                    True,
                )
                pypto.assemble(quant_view, input_offset, quant_tensor)
                pypto.assemble(scale_view, scale_offset, scale_tensor)

        pto_input = pypto.from_torch(case.input_data, "input")
        pto_quant = pypto.from_torch(case.quant_output, "quant")
        pto_scale = pypto.from_torch(case.scale_output, "scale")
        pypto.runtime._device_run_once_data_from_host(pto_input, pto_quant, pto_scale)
    finally:
        pypto.runtime._device_fini()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_quant_mx_fp32_2d_onboard():
    input_shape = [4, 256]
    view_shape = [2, 256]
    tile_shape = [1, 256]
    scale_shape = [4, 4, 2]

    input_data = torch.linspace(-7.5, 7.5, steps=math.prod(input_shape), dtype=torch.float32).reshape(input_shape)
    quant_output = torch.zeros(input_shape, dtype=torch.float8_e4m3fn)
    scale_output = torch.zeros(scale_shape, dtype=torch.float8_e8m0fnu)
    golden_quant_bytes, golden_scale_bytes = _quant_mx_golden_bytes(input_data)

    _run_quant_mx_onboard(
        _QuantMXOnboardCase(
            input_data=input_data,
            quant_output=quant_output,
            scale_output=scale_output,
            input_shape=input_shape,
            view_shape=view_shape,
            tile_shape=tile_shape,
            scale_shape=scale_shape,
            input_dtype=pypto.DT_FP32,
            quant_dtype=pypto.DT_FP8E4M3,
            round_mode=pypto.ROUND_DOWN,
        )
    )

    assert torch.equal(quant_output.view(torch.uint8), golden_quant_bytes)
    assert torch.equal(scale_output.view(torch.uint8), golden_scale_bytes)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_quant_mx_e2m1_nv_fp16_2d_onboard():
    input_shape = [1, 128]
    view_shape = [1, 128]
    tile_shape = [1, 128]
    scale_shape = [1, 2, 2]

    input_data = torch.linspace(-6.0, 6.0, steps=math.prod(input_shape), dtype=torch.float16).reshape(input_shape)
    quant_output = torch.zeros(input_shape, dtype=torch.float4_e2m1fn_x2)
    scale_output = torch.zeros(scale_shape, dtype=torch.float8_e8m0fnu)
    golden_quant_bytes, golden_scale_bytes = _quant_mx_e2m1_nv_golden_bytes(input_data)

    _run_quant_mx_onboard(
        _QuantMXOnboardCase(
            input_data=input_data,
            quant_output=quant_output,
            scale_output=scale_output,
            input_shape=input_shape,
            view_shape=view_shape,
            tile_shape=tile_shape,
            scale_shape=scale_shape,
            input_dtype=pypto.DT_FP16,
            quant_dtype=pypto.DT_FP4_E2M1X2,
            round_mode=pypto.ROUND_UP,
        )
    )

    actual_quant_bytes = (
        quant_output.view(torch.uint8).flatten()[:golden_quant_bytes.numel()].reshape(golden_quant_bytes.shape)
    )
    assert torch.equal(actual_quant_bytes, golden_quant_bytes)
    assert torch.equal(scale_output.view(torch.uint8), golden_scale_bytes)
