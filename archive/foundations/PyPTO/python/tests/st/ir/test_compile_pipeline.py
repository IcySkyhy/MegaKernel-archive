# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""System tests for new IR compile pipeline JIT execution."""

import os

import torch

import pypto


@pypto.frontend.jit(host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH})
def assemble_kernel(
    a: pypto.Tensor([pypto.DYN, 128]),
    out: pypto.Tensor([pypto.DYN, 128]),
):
    assert "a" == a.name
    assert "out" == out.name
    pypto.set_vec_tile_shapes(64, 128)
    aux_mat = pypto.tensor([129, 128], pypto.DT_FP32, name="aux_mat")
    pypto.assemble(pypto.full([129, 128], 0.0, pypto.DT_FP32), [0, 0], aux_mat)
    a[:] = pypto.view(a, [129, 128], [0, 0])
    pypto.assemble(a, [0, 0], aux_mat)
    pypto.assemble(aux_mat + a, [0, 0], out)


@pypto.options(pass_options={"enable_slice": True})
def test_assemble_jit_compile():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    a_torch = torch.empty([129, 128], dtype=torch.float32, device=device)
    out_torch = torch.empty([129, 128], dtype=torch.float32, device=device)

    assemble_kernel(a_torch, out_torch)


@pypto.frontend.jit(
    verify_options={"enable_pass_verify": True, "pass_verify_save_tensor": True},
    host_options={"compile_stage": pypto.CompStage.TENSOR_GRAPH},
)
def foo_kernel(
    x: pypto.Tensor([pypto.DYN, 32]),
    y: pypto.Tensor([pypto.DYN, 32]),
    z: pypto.Tensor([pypto.DYN, 32]),
):
    pypto.set_vec_tile_shapes(16, 16)
    x_view = pypto.view(x, [16, 32], [0, 0])
    y_view = pypto.view(y, [16, 32], [0, 0])
    for i in pypto.loop(2):
        if i == 0:
            tmp = x_view + y_view
        else:
            tmp = x_view - y_view
        pypto.assemble(tmp, [i * 16, 0], z)


def generate_f3_golden(a, b):
    z = torch.zeros([32, 32], dtype=torch.float32)
    z[0:16, :] = a[0:16, :] + b[0:16, :]
    z[16:32, :] = a[0:16, :] - b[0:16, :]
    return z


@pypto.options(pass_options={"enable_slice": True})
def test_verify_jit_compile():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    shape = [32, 32]
    a = torch.rand(shape, dtype=torch.float32, device=device)
    b = torch.rand(shape, dtype=torch.float32, device=device)
    c = torch.rand(shape, dtype=torch.float32, device=device)
    golden = generate_f3_golden(a.cpu(), b.cpu())

    pypto.set_verify_golden_data(goldens=[None, None, golden])
    foo_kernel(a, b, c)


if __name__ == "__main__":
    test_assemble_jit_compile()
    test_verify_jit_compile()
