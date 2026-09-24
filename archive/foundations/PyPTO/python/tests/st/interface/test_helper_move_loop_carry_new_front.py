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
"""ST: new-IR loop-carry when MOVE happens inside a helper on a caller view.

Regression for gdr_fwd-style kernels:

    buf = pypto.view(states, ...)
    for c in pypto.loop(...):
        helper(buf)          # helper body: state[:] = s_next  (MOVE, not Python store)

Old IR relied on mutable Tensor handles across loop iterations. New IR lowers
loops as SSA ``iter_args``; ``Tensor.move`` inside a nested helper must still
mark the caller alias (e.g. ``buf``) as loop-carried, and if/else arms that
both call the helper must not corrupt the incoming carry between branches.
"""

import os

import torch
import torch_npu

import pypto

_N = 8
_LOOP = 4


def _helper_accumulate(state: pypto.Tensor) -> None:
    """Nested helper: in-place MOVE carry, same object identity as caller ``buf``."""
    pypto.set_vec_tile_shapes(_N, _N)
    nxt = state + 1.0
    state[:] = nxt


def _helper_one_step(state: pypto.Tensor, _is_tail: bool) -> None:
    """Thin wrapper so the loop body is a function call (like gdr_fwd chunk step)."""
    _helper_accumulate(state)


@pypto.frontend.jit(
    new_ir=True,
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def helper_move_view_loop_carry_kernel(
    x: pypto.Tensor([_N, _N], pypto.DT_FP32),
    out: pypto.Tensor([_N, _N], pypto.DT_FP32),
):
    """Carry a view of ``x`` across ``pypto.loop`` via helper MOVE."""
    pypto.set_vec_tile_shapes(_N, _N)
    buf = pypto.view(x, [_N, _N], [0, 0])
    for _i in pypto.loop(_LOOP, name="chunk", idx_name="i"):
        _helper_accumulate(buf)
    pypto.assemble(buf, [0, 0], out)


@pypto.frontend.jit(
    new_ir=True,
    runtime_options={"run_mode": pypto.RunMode.NPU},
)
def helper_move_view_loop_carry_ifelse_kernel(
    x: pypto.Tensor([_N, _N], pypto.DT_FP32),
    out: pypto.Tensor([_N, _N], pypto.DT_FP32),
):
    """gdr_fwd-like: tail vs full chunk both delegate to the same helper."""
    pypto.set_vec_tile_shapes(_N, _N)
    buf = pypto.view(x, [_N, _N], [0, 0])
    for i in pypto.loop(_LOOP, name="chunk", idx_name="i"):
        if pypto.cond(pypto.is_loop_end(i)):
            _helper_one_step(buf, True)
        else:
            _helper_one_step(buf, False)
    pypto.assemble(buf, [0, 0], out)


def _run_and_check(kernel, device_id: int) -> None:
    x = torch.zeros(_N, _N, dtype=torch.float32, device=f"npu:{device_id}")
    out = torch.empty_like(x)
    kernel(x, out)
    torch_npu.npu.synchronize()
    expected = float(_LOOP)
    got = out[0, 0].item()
    assert abs(got - expected) < 1e-4, (
        f"loop-carry broken: expected all {expected}, got out[0,0]={got}. "
        "Without helper MOVE carry, every iteration reads the initial view (0)."
    )
    assert torch.allclose(
        out.cpu(),
        torch.full((_N, _N), expected, dtype=torch.float32),
        atol=1e-4,
        rtol=1e-4,
    )


def test_helper_move_view_loop_carry_new_front():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    _run_and_check(helper_move_view_loop_carry_kernel, device_id)


def test_helper_move_view_loop_carry_ifelse_new_front():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    _run_and_check(helper_move_view_loop_carry_ifelse_kernel, device_id)


if __name__ == "__main__":
    test_helper_move_view_loop_carry_new_front()
    test_helper_move_view_loop_carry_ifelse_new_front()
