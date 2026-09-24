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
Test scatternd codegen for Kirin9030
"""

from kirin.common_scatternd import (
    TEST_CASES,
    create_scatter_kernels,
    run_scatter_test,
)
import pytest


def create_test_module(soc_version):
    kernels = {"index_put_kernel": create_scatter_kernels(soc_version)}

    @pytest.mark.parametrize(
        "data_shape,indices_shape,update_shape,vec_tile_shape,"
        "torch_data_dtype,torch_indices_dtype,pypto_data_dtype,"
        "pypto_indices_dtype,accumulate",
        TEST_CASES,
    )
    def _test_scatternd(
        data_shape,
        indices_shape,
        update_shape,
        vec_tile_shape,
        torch_data_dtype,
        torch_indices_dtype,
        pypto_data_dtype,
        pypto_indices_dtype,
        accumulate,
    ):
        run_scatter_test(
            kernels,
            data_shape,
            indices_shape,
            update_shape,
            vec_tile_shape,
            torch_data_dtype,
            torch_indices_dtype,
            pypto_data_dtype,
            pypto_indices_dtype,
            accumulate,
        )

    return _test_scatternd


test_scatternd = create_test_module("Kirin9030")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
