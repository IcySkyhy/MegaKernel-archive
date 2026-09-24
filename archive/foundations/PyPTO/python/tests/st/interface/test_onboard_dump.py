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
上板 tensor dump 功能测试。

通过 subprocess 在独立进程中执行 dump 逻辑，确保 PYPTO_DATADUMP_ENABLE 环境变量
（C++ 侧 IsPtoDataDumpEnabled 的 static const 缓存，进程内只读一次）不影响批跑时
其它测试用例；子进程退出即彻底清理。
"""

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

import pypto

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """\
    import os

    import numpy as np
    import torch
    import torch_npu

    import pypto

    os.environ["PYPTO_DATADUMP_ENABLE"] = "true"

    verify_options = {
        "enable_pass_verify": True,
        "pass_verify_save_tensor": True,
    }


    @pypto.frontend.jit(
        runtime_options={"run_mode": pypto.RunMode.NPU},
        verify_options=verify_options,
    )
    def add_kernel(
        x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
        out: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP16),
    ):
        first_dim, second_dim = x.shape
        view_shape, tile_shape = (64, 64), (32, 32)
        first_view_shape, second_view_shape = view_shape
        for b_idx in pypto.loop(int(np.ceil(first_dim / view_shape[0])), name="LOOP_L0", idx_name="b_idx"):
            for s_idx in pypto.loop(int(np.ceil(second_dim / view_shape[1])), name="LOOP_L1", idx_name="s_idx"):
                tile_tensor_0 = pypto.view(
                    x, view_shape,
                    [b_idx * first_view_shape, s_idx * second_view_shape]
                )
                tile_tensor_1 = pypto.view(
                    y, view_shape,
                    [b_idx * first_view_shape, s_idx * second_view_shape]
                )
                pypto.set_vec_tile_shapes(*tile_shape)
                res = tile_tensor_0 + tile_tensor_1
                pypto.assemble(
                    res,
                    [b_idx * first_view_shape, s_idx * second_view_shape],
                    out,
                )
                del res, tile_tensor_0, tile_tensor_1


    shape = [64, 64]
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    device = f'npu:{device_id}'
    torch.npu.set_device(device_id)

    a = torch.rand(shape, dtype=torch.float16, device=device)
    b = torch.rand(shape, dtype=torch.float16, device=device)
    output_data = torch.zeros(shape, dtype=torch.float16, device=device)
    golden = (a.float() + b.float()).half()

    add_kernel(a, b, output_data)
    torch_npu.npu.synchronize()

    assert torch.allclose(output_data, golden), "onboard dump kernel result mismatch"
    print("SUBPROCESS_OK")
    """
)


@pypto.options(pass_options={"enable_slice": True})
def test_onboard_dump():
    env = os.environ.copy()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_SUBPROCESS_SCRIPT)
        script_path = f.name
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as err_f:
        err_path = err_f.name
    try:
        with open(err_path, "w") as err_file:
            result = subprocess.run(
                [sys.executable, script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                timeout=300,
            )
        if result.returncode != 0:
            with open(err_path) as ef:
                err_tail = ef.read()[-8000:]
            raise AssertionError(
                f"subprocess failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr(tail):\n{err_tail}"
            )
    finally:
        os.unlink(script_path)
        os.unlink(err_path)
    assert "SUBPROCESS_OK" in result.stdout, f"unexpected output:\n{result.stdout}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-p", "no:cacheprovider"]))
