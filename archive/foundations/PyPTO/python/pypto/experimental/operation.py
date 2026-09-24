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

from typing import Dict, List, Optional, Tuple, Union

from .. import pypto_impl
from .._op_wrapper import op_wrapper
from ..config import get_current_scope, set_options
from ..tensor import Tensor


@op_wrapper
def transposed_batchmatmul(tensor_a: Tensor, tensor_b: Tensor, out_dtype) -> Tensor:
    """
    Performs a transposed batch matrix multiplication.

    This operator computes:
        1. Transpose tensor_a from shape (M, B, K) to (B, M, K).
        2. Perform a batch matrix multiplication between the transposed tensor_a
           (B, M, K) and tensor_b (B, K, N), yielding an intermediate result of
           shape (B, M, N).
        3. Transpose the intermediate result back to shape (M, B, N).

    Parameters
    ----------
    tensor_a : Tensor
        The left-hand input tensor with shape (M, B, K).
        Supported data types: DT_FP16, DT_BF16.

    tensor_b : Tensor
        The right-hand input tensor with shape (B, K, N).
        Supported data types: DT_FP16, DT_BF16.

    out_dtype : dtype
        The data type for the output tensor.

    Returns
    -------
    Tensor
        The output tensor of shape (M, B, N).

    Examples
    --------
    a = pypto.tensor((16, 2, 32), pypto.DT_FP16, "tensor_a")
    b = pypto.tensor((2, 32, 64), pypto.DT_FP16, "tensor_b")
    c = pypto.experimental.transposed_batchmatmul(a, b, pypto.DT_FP16)
    """
    return pypto_impl.TransposedBatchMatmul(out_dtype, tensor_a, tensor_b)


def set_operation_options(*, combine_axis: Optional[bool] = None):
    """
    Set operation options.

    Parameters
    ---------
    combine_axis : bool
        Codegen forced axis fusion optimization.
    """

    options_dict = {k: v for k, v in locals().items() if v is not None}
    set_options(operation_options=options_dict)


def get_operation_options() -> Dict[str, Union[str, int, List[int], Dict[int, int]]]:
    """
    Get operation options.

    Returns
    -------
    Dict[str, Union[str, int, List[int], Dict[int, int]]]
        All operation options
    """

    scope = get_current_scope()
    return scope.get_operation_options()


@op_wrapper
def online_softmax(scores: Tensor, scale: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns axis-0 online softmax local statistics."""
    return pypto_impl.OnlineSoftmax(scores, scale)


@op_wrapper
def online_softmax_update(
    previous_max: Tensor,
    previous_sum: Tensor,
    previous_output: Tensor,
    current_max: Tensor,
    current_sum: Tensor,
    current_output: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Updates online softmax state for a new block."""
    return pypto_impl.OnlineSoftmaxUpdate(
        previous_max, previous_sum, previous_output, current_max, current_sum, current_output
    )


@op_wrapper
def nop(in_tensors: List[Tensor]) -> Tensor:
    return pypto_impl.Nop(in_tensors)
