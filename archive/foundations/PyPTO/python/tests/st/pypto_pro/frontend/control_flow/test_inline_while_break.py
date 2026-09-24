# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def advance_block_traversal(
    block_count,
    batch_index,
    sequence_index,
    start_position_tensor,
    sequence_used_tensor,
    cumulative_lengths_tensor,
    compression_ratio_value,
    batch_size_value,
    layout_value,
):
    skipped = 0
    while skipped < block_count:
        start_position_value = pl.getval(start_position_tensor, batch_index)
        if layout_value == 1:
            sequence_used_value = pl.getval(sequence_used_tensor, batch_index)
        else:
            sequence_used_value = pl.getval(cumulative_lengths_tensor, batch_index + 1) - pl.getval(
                cumulative_lengths_tensor, batch_index
            )
        compression_limit = (
            (start_position_value + sequence_used_value) // compression_ratio_value * compression_ratio_value
        )
        blocks_in_this_batch = (compression_limit - start_position_value) // compression_ratio_value
        blocks_remaining_this_batch = blocks_in_this_batch - sequence_index

        if blocks_remaining_this_batch > 0:
            taken = pl.min(block_count - skipped, blocks_remaining_this_batch)
            skipped = skipped + taken
            sequence_index = sequence_index + taken

        if sequence_index >= blocks_in_this_batch:
            sequence_index = 0
            batch_index = batch_index + 1

        if batch_index >= batch_size_value:
            break
    return batch_index, sequence_index


@pl.jit(auto_mutex=True)
def advance_block_traversal_kernel(
    start_pos: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    seq_used: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    cum_lengths: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    block_count = 4
    batch_idx = 0
    seq_idx = 0
    comp_ratio = 2
    batch_sz = 4
    layout = 1

    result_batch, result_seq = advance_block_traversal(
        block_count,
        batch_idx,
        seq_idx,
        start_pos,
        seq_used,
        cum_lengths,
        comp_ratio,
        batch_sz,
        layout,
    )

    tile_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        tile_a = a_db.next()
        pl.load(tile_a, start_pos, [0, result_batch])
        pl.store(out, tile_a, [0, 0])


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_advance_block_traversal():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)

    shape = [16, 16]
    start_pos = torch.zeros(shape, device=device, dtype=torch.int32)
    start_pos[0, 0] = 0
    start_pos[0, 1] = 10
    start_pos[0, 2] = 20
    start_pos[0, 3] = 30

    seq_used = torch.zeros(shape, device=device, dtype=torch.int32)
    seq_used[0, 0] = 8
    seq_used[0, 1] = 6
    seq_used[0, 2] = 4
    seq_used[0, 3] = 2

    cum_lengths = torch.zeros(shape, device=device, dtype=torch.int32)
    cum_lengths[0, 0] = 0
    cum_lengths[0, 1] = 8
    cum_lengths[0, 2] = 14
    cum_lengths[0, 3] = 18

    out = torch.zeros(shape, device=device, dtype=torch.int32)

    advance_block_traversal_kernel(start_pos, seq_used, cum_lengths, out)
    torch.npu.synchronize()

    # Reference computation: out[i, j] = start_pos[i, j+1] for i in 0..7, j in 0..7
    out_ref = torch.zeros(shape, device=device, dtype=torch.int32)
    out_ref[:8, :8] = start_pos[:8, 1:9]

    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)
    logging.info("test_advance_block_traversal passed! shape=%s", shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_advance_block_traversal()
