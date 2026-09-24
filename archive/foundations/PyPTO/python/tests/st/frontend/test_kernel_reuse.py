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
"""Test pypto.frontend.jit kernel reuse and recompile behavior."""

import logging
import os
import time

import torch

import pypto

logging.basicConfig(level=logging.INFO, format="", force=True)
DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def kernel_with_dynamic(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC]),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC]),
):
    pypto.set_vec_tile_shapes(16, 16)
    for idx in pypto.loop(a.shape[0], name="LOOP", idx_name="k"):
        temp = a[idx:idx + 1, :]
        out[idx:idx + 1, :] = temp + 1


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU}, new_ir=True)
def kernel_with_inplace_reshape(
    scale: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
    out: pypto.Tensor([pypto.STATIC, pypto.STATIC]),
):
    pypto.set_vec_tile_shapes(1, 8)
    scale = pypto.reshape(scale, [1, scale.shape[0]], inplace=True)
    for idx in pypto.loop(out.shape[0], name="LOOP", idx_name="k"):
        temp = scale[idx:idx + 1, :]
        out[idx:idx + 1, :] = temp + 1


def test_kernel_reuse():
    """DYNAMIC axis: second call skips compilation, should be faster."""
    torch.npu.set_device(DEVICE_ID)
    dev = f"npu:{DEVICE_ID}"

    a = torch.ones(1, 8, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 8, dtype=torch.float32, device=dev)
    t1 = time.perf_counter()
    kernel_with_dynamic(a, out)
    t1 = time.perf_counter() - t1
    assert torch.allclose(out.cpu(), (a + 1).cpu())

    a = torch.ones(2, 8, dtype=torch.float32, device=dev)
    out = torch.zeros(2, 8, dtype=torch.float32, device=dev)
    t2 = time.perf_counter()
    kernel_with_dynamic(a, out)
    t2 = time.perf_counter() - t2
    assert torch.allclose(out.cpu(), (a + 1).cpu())

    ratio = t2 / t1
    logging.info(f"First: {t1:.4f}s, second: {t2:.4f}s, ratio: {ratio:.2f}")
    assert ratio < 0.1, f"Second not faster ({t2:.4f}s vs {t1:.4f}s)"
    logging.info(f"✓ Cache reused, speedup {t1 / t2:.1f}x")


@pypto.options(pass_options={"enable_slice": False})
def test_kernel_recompile():
    """STATIC axis change triggers recompile, both calls take similar time."""
    torch.npu.set_device(DEVICE_ID)
    dev = f"npu:{DEVICE_ID}"

    a = torch.ones(1, 6, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 6, dtype=torch.float32, device=dev)
    t1 = time.perf_counter()
    kernel_with_dynamic(a, out)
    t1 = time.perf_counter() - t1
    assert torch.allclose(out.cpu(), (a + 1).cpu())

    a = torch.ones(1, 4, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 4, dtype=torch.float32, device=dev)
    t2 = time.perf_counter()
    kernel_with_dynamic(a, out)
    t2 = time.perf_counter() - t2
    assert torch.allclose(out.cpu(), (a + 1).cpu())

    ratio = t2 / t1
    logging.info(f"First: {t1:.4f}s, second: {t2:.4f}s, ratio: {ratio:.2f}")
    assert 0.5 < ratio < 2, f"Recompile expected, ratio {ratio:.2f} out of range"
    logging.info("✓ Recompile on static axis change")


def test_new_ir_inplace_reshape_cache_reuse():
    """new_ir cache should use entry shapes, not shapes mutated by inplace reshape."""
    torch.npu.set_device(DEVICE_ID)
    dev = f"npu:{DEVICE_ID}"

    scale = torch.ones(8, 1, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 8, dtype=torch.float32, device=dev)
    t1 = time.perf_counter()
    kernel_with_inplace_reshape(scale, out)
    t1 = time.perf_counter() - t1
    assert torch.allclose(out.cpu(), torch.full((1, 8), 2, dtype=torch.float32))

    scale = torch.ones(8, 1, dtype=torch.float32, device=dev)
    out = torch.zeros(1, 8, dtype=torch.float32, device=dev)
    t2 = time.perf_counter()
    kernel_with_inplace_reshape(scale, out)
    t2 = time.perf_counter() - t2
    assert torch.allclose(out.cpu(), torch.full((1, 8), 2, dtype=torch.float32))

    ratio = t2 / t1
    logging.info(f"new_ir inplace reshape first: {t1:.4f}s, second: {t2:.4f}s, ratio: {ratio:.2f}")
    assert ratio < 0.1, f"Second not faster after inplace reshape ({t2:.4f}s vs {t1:.4f}s)"
    logging.info(f"✓ new_ir inplace reshape cache reused, speedup {t1 / t2:.1f}x")


if __name__ == "__main__":
    test_kernel_reuse()
    test_kernel_recompile()
    test_new_ir_inplace_reshape_cache_reuse()
