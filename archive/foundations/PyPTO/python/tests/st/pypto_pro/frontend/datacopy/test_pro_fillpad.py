# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime validation for fillpad-family manual ops on CCE.

CCE block.load lowers to a bare ``TLOAD(tile, tensor)``. For ND row-major vec
tiles, that means we must first load the full physical tile and only then
narrow the runtime valid-shape before the fillpad-family operation.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _print_case_header(name: str) -> None:
    logging.info("------------%s--------------", name)


def _print_tensor_block(name: str, tensor: torch.Tensor) -> None:
    _print_case_header(name)
    logging.info("%s %s", tensor.shape, tensor.dtype)
    logging.info("%s", tensor)


def _make_fillpad_inputs(device: str, output_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.full((8, 8), -99, device=device, dtype=torch.int32)
    x[:5, :7] = torch.arange(35, device=device, dtype=torch.int32).reshape(5, 7)
    z = torch.empty(output_shape, device=device, dtype=torch.int32)
    z_ref = torch.zeros(output_shape, device=device, dtype=torch.int32)
    z_ref[:5, :7] = x[:5, :7]
    return x, z, z_ref


def _run_fillpad_cce_case(
    test_name: str, kernel, output_shape: tuple[int, int], output_label: str, device: str
) -> None:
    torch.npu.set_device(device)
    x, z, z_ref = _make_fillpad_inputs(device, output_shape)

    kernel(x, z)
    torch.npu.synchronize()

    _print_tensor_block(f"{test_name}_{output_label}", z)
    _print_tensor_block(f"{test_name}_golden", z_ref)

    torch.testing.assert_close(z, z_ref)
    logging.info("result equal!")


@pl.jit()
def fillpad_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(
        shape=[8, 8],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst_type = pl.TileType(
        shape=[8, 8],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )
    src = pl.make_tile(src_type, addr=0x0000, size=256)
    dst = pl.make_tile(dst_type, addr=0x0100, size=256)

    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.dump_data(src)
        pl.set_validshape(src, [5, 7])

        pl.fillpad(dst, src)

        pl.dump_data(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, dst, [0, 0])
        pl.system.bar_all()


@pl.jit()
def fillpad_inplace_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(
        shape=[8, 8],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
        valid_shape=[-1, -1],
    )
    dst_type = pl.TileType(
        shape=[8, 8],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )
    src = pl.make_tile(src_type, addr=0x0000, size=256)
    dst = pl.make_tile(dst_type, addr=0x0000, size=256)

    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.dump_data(src)
        pl.set_validshape(src, [5, 7])

        pl.fillpad(dst, src, mode=pl.FillPadMode.INPLACE)

        pl.dump_data(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, dst, [0, 0])
        pl.system.bar_all()


@pl.jit()
def fillpad_expand_dynamic_cce_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(
        shape=[8, 8],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    dst_type = pl.TileType(
        shape=[8, 16],
        dtype=pl.DT_INT32,
        target_memory=pl.MemorySpace.Vec,
        pad=pl.TilePad.zero,
    )
    src = pl.make_tile(src_type, addr=0x0000, size=256)
    dst = pl.make_tile(dst_type, addr=0x0100, size=512)

    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        pl.dump_data(src)
        pl.set_validshape(src, [5, 7])

        pl.fillpad(dst, src, mode=pl.FillPadMode.EXPAND)

        pl.dump_data(dst)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(z, dst, [0, 0])
        pl.system.bar_all()


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad_dynamic_cce():
    _print_case_header("test_fillpad_dynamic_cce")
    _run_fillpad_cce_case("fillpad_dynamic_cce", fillpad_dynamic_cce_kernel, (8, 8), "output", ST_DEVICE)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad_inplace_dynamic_cce():
    _print_case_header("test_fillpad_inplace_dynamic_cce")
    _run_fillpad_cce_case(
        "fillpad_inplace_dynamic_cce",
        fillpad_inplace_dynamic_cce_kernel,
        (8, 8),
        "output",
        ST_DEVICE,
    )


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_fillpad_expand_dynamic_cce():
    _print_case_header("test_fillpad_expand_dynamic_cce")
    _run_fillpad_cce_case("fillpad_expand_dynamic_cce", fillpad_expand_dynamic_cce_kernel, (8, 16), "output", ST_DEVICE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cases = [test_fillpad_dynamic_cce, test_fillpad_inplace_dynamic_cce, test_fillpad_expand_dynamic_cce]
    for case in cases:
        case()
    logging.info("\nAll tests passed!")
