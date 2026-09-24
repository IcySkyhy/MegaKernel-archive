#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """

import inspect

import pypto
from pypto.experimental import get_operation_options, set_operation_options


def test_print_options():
    pypto.set_print_options(edgeitems=1, precision=2, threshold=3, linewidth=4)


def test_pass_option():
    # 校验 get_pass_options 返回的 key 集合与 set_pass_options 参数集合一致
    pypto.reset_options()
    set_params = set(inspect.signature(pypto.set_pass_options).parameters)
    pass_option = pypto.get_pass_options()
    assert set(pass_option.keys()) == set_params, (
        f"get_pass_options keys {set(pass_option.keys())} != set_pass_options params {set_params}"
    )
    assert pass_option["enable_slice"] is True
    # tuple
    pypto.set_pass_options(sg_set_scope=48)
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (48, False, False)
    # map
    pypto.set_pass_options(cube_nbuffer_setting={3: 4})
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_setting"] == {3: 4}
    # bool
    pypto.set_pass_options(enable_slice=True)
    pass_option = pypto.get_pass_options()
    assert pass_option["enable_slice"] is True
    pypto.set_pass_options(enable_slice=False)
    pass_option = pypto.get_pass_options()
    assert pass_option["enable_slice"] is False


def test_host_option():
    pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.EXECUTE_GRAPH.value
    pypto.set_host_options(compile_monitor_enable=0)
    host_option = pypto.get_host_options()
    assert host_option["compile_monitor_enable"] == 0
    pypto.set_host_options(compile_monitor_print_interval=123)
    host_option = pypto.get_host_options()
    assert host_option["compile_monitor_print_interval"] == 123
    pypto.set_host_options(compile_timeout_stage=50)
    host_option = pypto.get_host_options()
    assert host_option["compile_timeout_stage"] == 50
    pypto.set_host_options(compile_timeout=1000)
    host_option = pypto.get_host_options()
    assert host_option["compile_timeout"] == 1000


def test_reset_option():
    pypto.set_host_options(compile_stage=pypto.CompStage.EXECUTE_GRAPH)
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.EXECUTE_GRAPH.value
    pypto.reset_options()
    host_option = pypto.get_host_options()
    assert host_option["compile_stage"] == pypto.CompStage.ALL_COMPLETE.value


def test_operation_option():
    set_operation_options(combine_axis=True)
    option = get_operation_options()
    assert option["combine_axis"]


def test_global_option():
    res = pypto.get_global_config("platform.enable_cost_model")
    assert not res
    pypto.set_global_config("platform.enable_cost_model", True)
    res = pypto.get_global_config("platform.enable_cost_model")
    assert res

    pypto.set_global_config("codegen.parallel_compile", 10)
    res = pypto.get_global_config("codegen.parallel_compile")
    assert res == 10


def test_option_map():
    pass_option = pypto.get_pass_options()
    assert pass_option["cube_nbuffer_setting"] == {-1: 1}


def test_sg_set_scope_new_format():
    pypto.set_pass_options(sg_set_scope=(1, True, True))
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (1, True, True)

    pypto.set_pass_options(sg_set_scope=48)
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (48, False, False)

    pypto.reset_options()
    pass_option = pypto.get_pass_options()
    assert pass_option["sg_set_scope"] == (-1, False, False)
    try:
        pypto.set_pass_options(sg_set_scope=(1, True))  # 元素不足
        assert False, "Should raise FeError"
    except pypto.error.FeError as e:
        assert "Expected 3" in str(e)

    try:
        pypto.set_pass_options(sg_set_scope=(1, "True", True))  # 类型错误
        assert False, "Should raise FeError"
    except pypto.error.FeError as e:
        assert "Expected bool" in str(e)


def test_vf_options():
    a = pypto.tensor([32, 32], pypto.DT_FP32, "a")
    b = pypto.tensor([32, 32], pypto.DT_FP32, "b")
    vf_options = "-mllvm -cce-vf-fusion-max-candidate-set-threshold=64"
    with pypto.options("main"):
        old_npuarch = pypto.platform.npuarch
        pypto.platform.npuarch = "DAV_3510"
        pypto.set_codegen_options(vf_options=vf_options)
        with pypto.function("main", a, b):
            for _ in pypto.loop(1):
                pypto.set_vec_tile_shapes(32, 32)
                b[:] = a + 1
        pypto.platform.npuarch = old_npuarch
        assert pypto.get_codegen_options()["vf_options"] == vf_options


if __name__ == "__main__":
    test_option_map()
    test_vf_options()
    test_sg_set_scope_new_format()
