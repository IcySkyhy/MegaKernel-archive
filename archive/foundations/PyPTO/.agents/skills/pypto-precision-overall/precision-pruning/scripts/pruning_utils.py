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
"""
PyPTO 用例剪枝 — 核心计算函数。
"""


def compute_pruned_dim(original_dim: int, tile_size: int,
                        max_unroll: int = 1) -> int:
    """剪枝后的维度大小。保持尾块行为不变：有尾块→tile+tail，无尾块→tile。

    >>> compute_pruned_dim(1024, 16)    # tail=0 → 16
    16
    >>> compute_pruned_dim(1030, 16)    # tail=6 → 22
    22
    >>> compute_pruned_dim(15, 16)      # 已1轮 → 不变
    15
    >>> compute_pruned_dim(1040, 16, max_unroll=4)   # tail=16 → 80
    80
    >>> compute_pruned_dim(65, 16, max_unroll=4)     # 已2轮 → 不变
    65
    """
    effective_tile = tile_size * max_unroll
    unroll_iters = (original_dim + effective_tile - 1) // effective_tile
    if unroll_iters <= 1:
        return original_dim

    tail = original_dim % effective_tile
    if tail == 0:
        new_dim = effective_tile
    else:
        new_dim = effective_tile + tail

    return original_dim if new_dim == original_dim else new_dim


def compute_loop_count(dim: int, tile: int, max_unroll: int = 1) -> int:
    """原始循环轮数。

    >>> compute_loop_count(17, 16)
    2
    >>> compute_loop_count(1024, 16, max_unroll=4)
    16
    """
    return (dim + tile * max_unroll - 1) // (tile * max_unroll)


def compute_hardcoded_loop_count(dim: int, tile: int) -> int:
    """硬编码 pypto.loop(N) 剪枝后的 N。

    >>> compute_hardcoded_loop_count(1024, 16)   # dim>>tile, tail=0 → 1
    1
    >>> compute_hardcoded_loop_count(1030, 16)   # dim>>tile, tail=6 → 2
    2
    >>> compute_hardcoded_loop_count(7, 16)      # dim<tile → 1
    1
    """
    if dim <= tile:
        return 1
    return 1 if dim % tile == 0 else 2
