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
""" """

import math
import os

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

import pypto


def normal_numpy_golden(shape, key, counter, alg, dtype):
    """
    Numpy implementation of normal based on TensorFlow's Philox algorithm.
    Replicates TensorFlow's behavior using the Philox 4x32_10 algorithm with Box-Muller transform.

    Args:
        shape: Output shape (list of integers)
        key: Random seed key (list with one uint64 value)
        counter: Counter value (list with two uint64 values)
        alg: Algorithm identifier (list with one int, 1=Philox)
        dtype: Data type (pypto.DT_FP32, pypto.DT_FP16, pypto.DT_BF16)

    Returns:
        Random normal array with values from standard normal distribution (mean=0, std=1)
    """
    if dtype == pypto.DT_FP16:
        np_dtype = np.float16
    elif dtype == pypto.DT_BF16:
        np_dtype = np.float32
    else:
        np_dtype = np.float32

    def uint64_to_uint32_pair(val):
        arr = np.array([val], dtype=np.uint64)
        return arr.view(np.uint32).copy()

    key_arr = uint64_to_uint32_pair(np.uint64(key[0]))
    # Hardcode counter[0] to 0 to match pypto.normal behavior (aligned with CANN ACLNN)
    counter_arr = np.concatenate([uint64_to_uint32_pair(np.uint64(0)), uint64_to_uint32_pair(np.uint64(counter[1]))])

    philox_w32a = 0x9E3779B9
    philox_w32b = 0xBB67AE85
    philox_m4x32a = 0xD2511F53
    philox_m4x32b = 0xCD9E8D57

    def multiply_high_low(a, b):
        product = int(a) * int(b)
        return (product & 0xFFFFFFFF), ((product >> 32) & 0xFFFFFFFF)

    def compute_single_round(counter, key):
        lo0, hi0 = multiply_high_low(philox_m4x32a, int(counter[0]))
        lo1, hi1 = multiply_high_low(philox_m4x32b, int(counter[2]))

        result = np.zeros(4, dtype=np.uint32)
        result[0] = np.uint32(hi1 ^ int(counter[1]) ^ int(key[0]))
        result[1] = np.uint32(lo1)
        result[2] = np.uint32(hi0 ^ int(counter[3]) ^ int(key[1]))
        result[3] = np.uint32(lo0)
        return result

    def raise_key(key):
        key[0] = np.uint32((int(key[0]) + philox_w32a) & 0xFFFFFFFF)
        key[1] = np.uint32((int(key[1]) + philox_w32b) & 0xFFFFFFFF)

    def philox_next(counter, key):
        c = counter.copy()
        k = key.copy()

        for _ in range(10):
            c = compute_single_round(c, k)
            raise_key(k)

        counter[0] = np.uint32((int(counter[0]) + 1) & 0xFFFFFFFF)
        if counter[0] == 0:
            counter[1] = np.uint32((int(counter[1]) + 1) & 0xFFFFFFFF)
            if counter[1] == 0:
                counter[2] = np.uint32((int(counter[2]) + 1) & 0xFFFFFFFF)
                if counter[2] == 0:
                    counter[3] = np.uint32((int(counter[3]) + 1) & 0xFFFFFFFF)

        return c

    def uint32_to_float(uint_val):
        man = int(uint_val) & 0x7FFFFF
        exp = 127
        val = (exp << 23) | man
        result = np.frombuffer(np.array([val], dtype=np.uint32).tobytes(), dtype=np.float32)[0]
        return float(result - 1.0)

    def box_muller_float(x0, x1):
        epsilon = 1.0e-7
        u1 = uint32_to_float(x0)
        if u1 < epsilon:
            u1 = epsilon

        v1 = 2.0 * math.pi * uint32_to_float(x1)
        u2 = math.sqrt(-2.0 * math.log(u1))

        f0 = u2 * math.sin(v1)
        f1 = u2 * math.cos(v1)

        return f0, f1

    def convert_normal_pair(uint0, uint1):
        f0, f1 = box_muller_float(uint0, uint1)

        if dtype == pypto.DT_FP16:
            return float(np.float16(f0)), float(np.float16(f1))
        elif dtype == pypto.DT_BF16:
            return f0, f1
        else:
            return f0, f1

    total_elements = np.prod(shape)
    num_pairs = (total_elements + 1) // 2
    num_rounds = (num_pairs + 1) // 2

    counter_state = counter_arr.copy()
    key_state = key_arr.copy()

    all_random_vals = []
    for _ in range(num_rounds):
        random_uints = philox_next(counter_state, key_state)
        for i in range(0, len(random_uints), 2):
            if i + 1 < len(random_uints):
                f0, f1 = convert_normal_pair(random_uints[i], random_uints[i + 1])
                all_random_vals.extend([f0, f1])

    result = np.array(all_random_vals[:total_elements], dtype=np_dtype)
    return result.reshape(shape)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_normal_fp32():
    """Test whether the output of FP32 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    view_shape = [32]
    tile_shape = [32]

    shape = [32]
    key = [1234]
    counter = [0, 1]
    alg = [1]
    dtype = pypto.DT_FP32
    output = pypto.tensor(shape, dtype)

    loop_num = math.ceil(shape[0] / view_shape[0])
    with pypto.function("NORMAL_CONTENT_FP32", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            _valid_shape = pypto.min(pypto.symbolic_scalar(shape[0]) - offset, pypto.symbolic_scalar(view_shape[0]))
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.normal(shape, key, counter, alg, dtype)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    out_data = np.zeros(shape, dtype=np.float32)
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    golden = normal_numpy_golden(shape, key, counter, alg, dtype)
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-4, atol=1e-4)

    pypto.runtime._device_fini()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_normal_fp16():
    """Test whether the output of FP16 is correct"""
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()

    view_shape = [32]
    tile_shape = [32]

    shape = [32]
    key = [1234]
    counter = [0, 1]
    alg = [1]
    dtype = pypto.DT_FP16
    output = pypto.tensor(shape, dtype)

    loop_num = math.ceil(shape[0] / view_shape[0])
    with pypto.function("NORMAL_CONTENT_FP16", output):
        for idx in pypto.loop(loop_num, name="loop0", idx_name="idx"):
            offset = idx * view_shape[0]
            _valid_shape = pypto.min(pypto.symbolic_scalar(shape[0]) - offset, pypto.symbolic_scalar(view_shape[0]))
            pypto.set_vec_tile_shapes(tile_shape[0])
            res = pypto.normal(shape, key, counter, alg, dtype)
            pypto.assemble(res, [offset], output)

    assert isinstance(output, pypto.tensor)
    out_data = np.zeros(shape, dtype=np.float16)
    pto_out = pypto.from_torch(torch.from_numpy(out_data), "PTO_TENSOR_output")
    pypto.runtime._device_run_once_data_from_host(pto_out)
    golden = normal_numpy_golden(shape, key, counter, alg, dtype)
    assert_allclose(out_data.flatten(), golden.flatten(), rtol=1e-3, atol=1e-3)

    pypto.runtime._device_fini()
