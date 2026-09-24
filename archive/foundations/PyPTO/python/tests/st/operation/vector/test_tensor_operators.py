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
Test tensor operator overloads with frontend.jit
Operators: __neg__, __pos__, __rsub__, __rmul__, __rtruediv__,
           __floordiv__, __mod__, __pow__, __lt__, __le__, __ge__, __eq__, __ne__
"""

import logging
import os

from numpy.testing import assert_allclose
import torch
import torch_npu

import pypto

# =============================================================================
# Test 1: Unary operators (-, +)
# =============================================================================


@pypto.frontend.jit()
def neg_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(-x)


def test_tensor_neg():
    """Test -tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    neg_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = -x_data.cpu()
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def pos_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(+x + 1.0)


def test_tensor_pos():
    """Test +tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    pos_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = +x_data.cpu() + 1.0
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


# =============================================================================
# Test 2: Reverse operators (__rsub__, __rmul__, __rtruediv__)
# =============================================================================


@pypto.frontend.jit()
def rsub_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(10.0 - x)


def test_tensor_rsub():
    """Test scalar - tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    rsub_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = 10.0 - x_data.cpu()
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def rmul_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(5.0 * x)


def test_tensor_rmul():
    """Test scalar * tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    rmul_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = 5.0 * x_data.cpu()
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def rtruediv_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(100.0 / x)


def test_tensor_rtruediv():
    """Test scalar / tensor"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}') + 10.0
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    rtruediv_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = 100.0 / x_data.cpu()
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


# =============================================================================
# Test 3: Other binary operators (__floordiv__, __mod__, __pow__)
# =============================================================================


@pypto.frontend.jit()
def floordiv_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x // 3.0)


def test_tensor_floordiv():
    """Test tensor // scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}') * 10
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    floordiv_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = torch.floor(x_data.cpu() / 3.0)
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def mod_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x % 7.0)


def test_tensor_mod():
    """Test tensor % scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}') * 10
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    mod_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = torch.remainder(x_data.cpu(), 7.0)
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def pow_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x**2.0)


def test_tensor_pow():
    """Test tensor ** scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    pow_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = torch.pow(x_data.cpu(), 2.0)
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


# =============================================================================
# Test 4: Comparison operators (<, <=, >=, ==, !=)
# =============================================================================


@pypto.frontend.jit()
def lt_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BOOL),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x < 0.5)


@pypto.options(pass_options={"enable_slice": False})
def test_tensor_lt():
    """Test tensor < scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.bool, device=f'npu:{device_id}')

    lt_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = x_data.cpu() < 0.5
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def le_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BOOL),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x <= 0.5)


def test_tensor_le():
    """Test tensor <= scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.bool, device=f'npu:{device_id}')

    le_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = x_data.cpu() <= 0.5
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def ge_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BOOL),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x >= 0.5)


def test_tensor_ge():
    """Test tensor >= scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.bool, device=f'npu:{device_id}')

    ge_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = x_data.cpu() >= 0.5
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def eq_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BOOL),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x == 0.5)


@pypto.options(pass_options={"enable_slice": False})
def test_tensor_eq():
    """Test tensor == scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    x_data[0, 0] = 0.5
    y_data = torch.zeros(shape, dtype=torch.bool, device=f'npu:{device_id}')

    eq_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = x_data.cpu() == 0.5
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


@pypto.frontend.jit()
def ne_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BOOL),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(x != 0.5)


def test_tensor_ne():
    """Test tensor != scalar"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    x_data[0, 0] = 0.5
    y_data = torch.zeros(shape, dtype=torch.bool, device=f'npu:{device_id}')

    ne_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = x_data.cpu() != 0.5
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)


# =============================================================================
# Test 5: Complex expressions (tanh-like, combined arithmetic)
# =============================================================================


@pypto.frontend.jit()
def tanh_like_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    exp_pos = pypto.exp(x)
    exp_neg = pypto.exp(-x)
    y.move((exp_pos - exp_neg) / (exp_pos + exp_neg))


def test_tanh_like_expression():
    """Test tanh-like expression using - operator"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    tanh_like_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = torch.tanh(x_data.cpu())
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


@pypto.frontend.jit()
def combined_arithmetic_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    z: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(2.0 * x - z + (x / 3.0))


def test_combined_arithmetic():
    """Test combined arithmetic using operators"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    z_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    combined_arithmetic_kernel(x_data, z_data, y_data)

    torch_npu.npu.synchronize()

    expected = 2.0 * x_data.cpu() - z_data.cpu() + (x_data.cpu() / 3.0)
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


@pypto.frontend.jit()
def complex_expression_kernel(
    x: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    y: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(32, 32)
    y.move(100.0 - x**2.0 % 10.0)


def test_complex_expression():
    """Test complex expression"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    tiling = 32
    n, m = tiling * 2, tiling * 2
    shape = (n, m)

    x_data = torch.randn(shape, dtype=torch.float32, device=f'npu:{device_id}')
    y_data = torch.zeros(shape, dtype=torch.float32, device=f'npu:{device_id}')

    complex_expression_kernel(x_data, y_data)

    torch_npu.npu.synchronize()

    expected = 100.0 - torch.remainder(torch.pow(x_data.cpu(), 2.0), 10.0)
    actual = y_data.cpu()
    assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_functions = [
        ("test_tensor_neg", test_tensor_neg),
        ("test_tensor_pos", test_tensor_pos),
        ("test_tensor_rsub", test_tensor_rsub),
        ("test_tensor_rmul", test_tensor_rmul),
        ("test_tensor_rtruediv", test_tensor_rtruediv),
        ("test_tensor_floordiv", test_tensor_floordiv),
        ("test_tensor_mod", test_tensor_mod),
        ("test_tensor_pow", test_tensor_pow),
        ("test_tensor_lt", test_tensor_lt),
        ("test_tensor_le", test_tensor_le),
        ("test_tensor_ge", test_tensor_ge),
        ("test_tensor_eq", test_tensor_eq),
        ("test_tensor_ne", test_tensor_ne),
        ("test_tanh_like_expression", test_tanh_like_expression),
        ("test_combined_arithmetic", test_combined_arithmetic),
        ("test_complex_expression", test_complex_expression),
    ]

    passed = 0
    failed = 0

    for name, func in test_functions:
        try:
            logging.info(f"\n>>> Running {name}...")
            func()
            logging.info(f"    ✓ {name} PASSED")
            passed += 1
        except Exception as e:
            logging.info(f"    ✗ {name} FAILED: {e}")
            failed += 1

    logging.info("\n" + "=" * 80)
    logging.info(f"Test Summary: {passed} passed, {failed} failed")
    logging.info("=" * 80)

    if failed > 0:
        exit(1)
