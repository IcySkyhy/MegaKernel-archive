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

from dataclasses import dataclass
import os
from typing import Any, List, Tuple, Union

from numpy.testing import assert_allclose
import torch

import pypto

# ----------------- Config -----------------

TensorLike1D = Union[torch.Tensor, List[Any]]
TensorLike2D = Union[torch.Tensor, List[Any]]


# ----------------- 配置：改为直接用 torch.dtype -----------------
@dataclass
class PageAttentionTestConfig:
    """
    C++ 模板等价版本（偏 torch 风格）：
      template <typename IndexT, typename DataT>
      struct PageAttentionTestConfig { ... };

    这里直接存 torch 的 dtype，方便后面建 tensor。
    """

    topk_count: int  # topk 的 k 值：选出的 token 个数
    num_logical_blocks: int  # 逻辑块个数（page_table 长度）
    num_buffer_tokens: int  # buffer 第一维长度：物理 token 容量
    hidden_dim: int  # buffer 第二维长度：隐藏维度大小
    block_size: int  # 每个块里有多少个 token

    index_dtype: torch.dtype = torch.int32  # 类似 int32_t
    data_dtype: torch.dtype = torch.float16  # 类似 float16


# ----------------- 基础打印工具（支持 list / tensor） -----------------
def _to_list_1d(v: TensorLike1D) -> List[Any]:
    if isinstance(v, torch.Tensor):
        return v.flatten().tolist()
    return list(v)


# ----------------- 参数合法性检查 -----------------
def validate_config(cfg: PageAttentionTestConfig) -> Tuple[bool, str]:
    err = ""

    # 每个逻辑块有 block_size 个 token，
    # 总逻辑 token 数 = num_logical_blocks * block_size
    total_logical_tokens = cfg.num_logical_blocks * cfg.block_size

    # 强制 topk 的 k 不超过逻辑 token 总数
    if cfg.topk_count > total_logical_tokens:
        err = "topk_count 必须 <= num_logical_blocks * block_size（topk 的 k 不能超过逻辑 token 总数）"
        return False, err

    # 物理块总数 = num_buffer_tokens / block_size
    if cfg.num_buffer_tokens < cfg.block_size:
        err = "num_buffer_tokens 必须至少 >= block_size,才能容纳一个物理块"
        return False, err

    num_physical_blocks = cfg.num_buffer_tokens // cfg.block_size
    if num_physical_blocks <= 0:
        err = "num_buffer_tokens / block_size 必须 > 0"
        return False, err

    return True, ""


# ----------------- 构造 buffer[num_buffer_tokens, hidden_dim] (torch.Tensor) -----------------
def make_buffer(cfg: PageAttentionTestConfig) -> torch.Tensor:
    buffer = torch.empty(
        (cfg.num_buffer_tokens, cfg.hidden_dim),
        dtype=cfg.data_dtype,
    )

    for token_index in range(cfg.num_buffer_tokens):
        for h in range(cfg.hidden_dim):
            buffer[token_index, h] = 10.0 * token_index + h

    return buffer.to(cfg.data_dtype)


# ----------------- 构造 page_table[1, num_logical_blocks] (torch.Tensor) -----------------
def make_page_table(cfg: PageAttentionTestConfig, seed: int = 42) -> torch.Tensor:
    """
    返回 shape = [1, num_logical_blocks]
    """
    num_physical_blocks = cfg.num_buffer_tokens // cfg.block_size

    g = torch.Generator()
    g.manual_seed(seed)
    page_table = torch.randint(
        low=0,
        high=num_physical_blocks,
        size=(1, cfg.num_logical_blocks),  # 加上 batch 维
        generator=g,
        dtype=cfg.index_dtype,
    )
    return page_table


# ----------------- 构造 topk_indices[1, topk_count] (torch.Tensor) -----------------
def make_topk_indices(cfg: PageAttentionTestConfig, seed: int = 123) -> torch.Tensor:
    """
    返回 shape = [1, topk_count]
    """
    total_logical_tokens = cfg.num_logical_blocks * cfg.block_size
    g = torch.Generator()
    g.manual_seed(seed)
    topk_indices = torch.randint(
        low=0,
        high=total_logical_tokens,
        size=(1, cfg.topk_count),  # 加上 batch 维
        generator=g,
        dtype=cfg.index_dtype,
    )
    return topk_indices


# ----------------- 逻辑 index -> 物理 index 的核心函数 -----------------


def compute_physical_index(
    logical_index: Union[int, torch.Tensor],
    page_table: torch.Tensor,
    cfg: PageAttentionTestConfig,
) -> int:
    """
    输入逻辑 token index（可以是 python int 或 0-dim tensor），
    输出物理 token index（python int），便于后面做索引。
    """
    if isinstance(logical_index, torch.Tensor):
        logical_index = int(logical_index.item())
    else:
        logical_index = int(logical_index)

    logical_block_id = logical_index // cfg.block_size
    physical_block_id = int(page_table[logical_block_id].item())
    block_offset = logical_index % cfg.block_size
    physical_index = physical_block_id * cfg.block_size + block_offset
    return physical_index


# ----------------- 根据 pageattention 逻辑进行 gather (torch.Tensor) -----------------
def gather_golden(
    topk_indices: torch.Tensor,
    page_table: torch.Tensor,
    buffer: torch.Tensor,
    cfg: PageAttentionTestConfig,
) -> torch.Tensor:
    # 支持输入 [1, k]，自动 flatten
    if topk_indices.dim() == 2:
        assert topk_indices.shape[0] == 1
        topk_indices = topk_indices.flatten()

    if page_table.dim() == 2:
        assert page_table.shape[0] == 1
        page_table = page_table.flatten()

    if topk_indices.numel() != cfg.topk_count:
        raise RuntimeError("topk_indices.size() != topk_count")

    if page_table.numel() != cfg.num_logical_blocks:
        raise RuntimeError("page_table.size() != num_logical_blocks")

    if buffer.numel() != cfg.num_buffer_tokens * cfg.hidden_dim:
        raise RuntimeError("buffer.size() != num_buffer_tokens * hidden_dim")

    if buffer.dim() != 2 or buffer.shape != (cfg.num_buffer_tokens, cfg.hidden_dim):
        raise RuntimeError("buffer shape mismatch")

    result = torch.empty((cfg.topk_count, cfg.hidden_dim), dtype=cfg.data_dtype)

    total_logical_tokens = cfg.num_logical_blocks * cfg.block_size

    for j in range(cfg.topk_count):
        logical_index = int(topk_indices[j])

        if logical_index < 0 or logical_index >= total_logical_tokens:
            raise RuntimeError(f"logical_index 越界: {logical_index}")

        physical_index = compute_physical_index(logical_index, page_table, cfg)

        if physical_index < 0 or physical_index >= cfg.num_buffer_tokens:
            raise RuntimeError(f"physical_index 越界: {physical_index}")

        result[j, :] = buffer[physical_index, :]

    return result


@pypto.options(pass_options={"enable_slice": True})
def test_vector_operator_gatherinub():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    pypto.runtime._device_init()
    cfg = PageAttentionTestConfig(
        topk_count=8,
        num_logical_blocks=3,
        num_buffer_tokens=32,
        hidden_dim=4,
        block_size=4,
        index_dtype=torch.int32,
        data_dtype=torch.float16,
    )
    ok, err = validate_config(cfg)
    if not ok:
        raise RuntimeError(f"Config invalid: {err}")
    buffer = make_buffer(cfg)
    page_table = make_page_table(cfg, seed=42)
    topk_indices = make_topk_indices(cfg, seed=123)
    golden = gather_golden(topk_indices, page_table, buffer, cfg)
    src_shapes = [cfg.num_buffer_tokens, cfg.hidden_dim]
    offsets_shapes = [1, cfg.topk_count]
    page_table_shapes = [1, cfg.num_logical_blocks]
    dst_shapes = [cfg.topk_count, cfg.hidden_dim]
    src = pypto.tensor(src_shapes, pypto.DT_FP16, "src")
    offsets = pypto.tensor(offsets_shapes, pypto.DT_INT32, "offsets")
    pto_page_table = pypto.tensor(page_table_shapes, pypto.DT_INT32, "page_table")
    dst = pypto.tensor(dst_shapes, pypto.DT_FP16, "dst")
    with pypto.function("MAIN", src, offsets, pto_page_table, dst):
        for _ in pypto.loop(1, name="b0", idx_name="bidx"):
            pypto.set_vec_tile_shapes(32, 64)
            dyn_src = pypto.view(src, src_shapes, [0, 0], valid_shape=src_shapes)
            dyn_offsets = pypto.view(offsets, offsets_shapes, [0, 0], valid_shape=offsets_shapes)
            tmp = pypto.experimental.gather_in_ub(dyn_src, dyn_offsets, pto_page_table, cfg.block_size, -2)
            pypto.assemble(tmp, [0, 0], dst)
            del dyn_src, dyn_offsets
    result = torch.zeros(dst_shapes, dtype=torch.float16)
    pto_a_tensor = pypto.from_torch(buffer, "buffer")
    pto_b_tensor = pypto.from_torch(topk_indices, "topk_indices")
    pto_c_tensor = pypto.from_torch(page_table, "page_table")
    pto_d_tensor = pypto.from_torch(result, "result")
    pypto.runtime._device_run_once_data_from_host(pto_a_tensor, pto_b_tensor, pto_c_tensor, pto_d_tensor)
    assert_allclose(result.flatten(), golden.flatten(), rtol=3e-3, atol=3e-3)
    pypto.runtime._device_fini()
