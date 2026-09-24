# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for ``pl.struct_array`` array field (list-valued field) scenarios.

Covers array field write/read on struct_array slots, control-flow interaction
(for/if/while/break), aliasing, reference passing, multi-struct_array
interaction, and user ``run_info`` loop scenario simulation.
"""

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


# =============================================================================
# 3.2 struct_array array field element write
# =============================================================================

@pl.jit()
def struct_array_arr_field_write_const_idx_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(2, "Saw1", x=0, arr=[0, 0, 0, 0])
    with pl.section_vector():
        arr[0].arr[1] = 99
        pl.setval(out, 0, arr[0].arr[1])


@pytest.mark.soc("950")
def test_struct_array_arr_field_write_const_idx():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_write_const_idx_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([99], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_array_arr_field_write_loop_idx_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Saw2", arr=[0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 3):
            arr[i].arr[0] = i * 100
        for i in pl.range(0, 3):
            pl.setval(out, i, arr[i].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_write_loop_idx():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_write_loop_idx_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 100, 200], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_array_arr_field_write_dynamic_idx_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "Saw3", arr=[0, 0])
    with pl.section_vector():
        for i in pl.range(0, 8):
            arr[i % 4].arr[0] = i
        for i in pl.range(0, 4):
            pl.setval(out, i, arr[i].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_write_dynamic_idx():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_write_dynamic_idx_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([4, 5, 6, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_array_arr_field_cross_slot_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(2, "Saw4", data=[0, 0, 0])
    with pl.section_vector():
        arr[0].data[0] = 10
        arr[1].data[0] = arr[0].data[0] + 5
        pl.setval(out, 0, arr[1].data[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_cross_slot():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_cross_slot_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_array_arr_field_all_slots_all_elems_kernel(
    out: pl.Tensor[[9], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Saw5", arr=[0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 3):
            for j in pl.range(0, 3):
                arr[i].arr[j] = i * 3 + j
        for i in pl.range(0, 3):
            for j in pl.range(0, 3):
                pl.setval(out, i * 3 + j, arr[i].arr[j])


@pytest.mark.soc("950")
def test_struct_array_arr_field_all_slots_all_elems():
    _check_npu()
    out = torch.zeros(9, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_all_slots_all_elems_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 4.1 for loop + struct_array array field
# =============================================================================

@pl.jit()
def struct_array_arr_field_for_slot_fill_kernel(
    out: pl.Tensor[[8], pl.DT_INT32],
):
    arr = pl.struct_array(2, "Saf1", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 2):
            for j in pl.range(0, 4):
                arr[i].arr[j] = i * 4 + j
        for i in pl.range(0, 2):
            for j in pl.range(0, 4):
                pl.setval(out, i * 4 + j, arr[i].arr[j])


@pytest.mark.soc("950")
def test_struct_array_arr_field_for_slot_fill():
    _check_npu()
    out = torch.zeros(8, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_for_slot_fill_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 4.2 if/else + struct_array array field
# =============================================================================

@pl.jit()
def struct_array_arr_field_if_branch_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "Sai1", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            if i < 2:
                arr[i].arr[0] = 100
            else:
                arr[i].arr[0] = 200
        for i in pl.range(0, 4):
            pl.setval(out, i, arr[i].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_if_branch():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_if_branch_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100, 100, 200, 200], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 4.3 while loop + struct_array array field
# =============================================================================

@pl.jit()
def struct_array_arr_field_while_ring_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Saw1", arr=[0, 0, 0])
    with pl.section_vector():
        i = 0
        while i < 10:
            arr[i % 3].arr[0] = i
            i = i + 1
        for j in pl.range(0, 3):
            pl.setval(out, j, arr[j].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_while_ring():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_while_ring_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([9, 7, 8], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 4.4 break + struct_array array field
# =============================================================================

@pl.jit()
def struct_array_arr_field_break_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(4, "Sab1", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 100):
            arr[i % 4].arr[0] = i
            if i >= 6:
                break
        for j in pl.range(0, 4):
            pl.setval(out, j, arr[j].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_break():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_break_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([4, 5, 6, 3], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 5.2 struct_array alias + array field
# =============================================================================

@pl.jit()
def struct_array_arr_field_slot_alias_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(3, "Saa1", arr=[0, 0, 0])
    with pl.section_vector():
        slot = arr[1]
        alias = slot
        alias.arr[0] = 50
        pl.setval(out, 0, arr[1].arr[0])


@pytest.mark.soc("950")
def test_struct_array_arr_field_slot_alias():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_slot_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([50], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_array_arr_field_chain_alias_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(2, "Saa2", arr=[0, 0, 0])
    with pl.section_vector():
        slot = arr[0]
        handle = slot
        alias = handle
        alias.arr[1] = 77
        pl.setval(out, 0, arr[0].arr[1])


@pytest.mark.soc("950")
def test_struct_array_arr_field_chain_alias():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([77], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 5.3 Function pass-by-reference + struct_array array field
# =============================================================================

def _fill_arr_slot(arr, i, j, val):
    slot = arr[i]
    slot.arr[j] = val


@pl.jit()
def struct_array_arr_field_pass_by_ref_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    arr = pl.struct_array(2, "Sar1", arr=[0, 0])
    with pl.section_vector():
        _fill_arr_slot(arr, 0, 0, 11)
        _fill_arr_slot(arr, 0, 1, 22)
        _fill_arr_slot(arr, 1, 0, 33)
        _fill_arr_slot(arr, 1, 1, 44)
        pl.setval(out, 0, arr[0].arr[0])
        pl.setval(out, 1, arr[0].arr[1])
        pl.setval(out, 2, arr[1].arr[0])
        pl.setval(out, 3, arr[1].arr[1])


@pytest.mark.soc("950")
def test_struct_array_arr_field_pass_by_ref():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_array_arr_field_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([11, 22, 33, 44], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 6.2 Multi struct_array interaction — array field copy
# =============================================================================

@pl.jit()
def struct_array_to_struct_array_field_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    src = pl.struct_array(3, "Src", arr=[0, 0])
    dst = pl.struct_array(3, "Dst", arr=[0, 0])
    with pl.section_vector():
        for i in pl.range(0, 3):
            src[i].arr[0] = i * 10 + 1
            src[i].arr[1] = i * 10 + 2
        for i in pl.range(0, 3):
            dst[i].arr[0] = src[i].arr[0]
            dst[i].arr[1] = src[i].arr[1]
        for i in pl.range(0, 3):
            pl.setval(out, i * 2, dst[i].arr[0])
            pl.setval(out, i * 2 + 1, dst[i].arr[1])


@pytest.mark.soc("950")
def test_struct_array_to_struct_array_field():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    struct_array_to_struct_array_field_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 2, 11, 12, 21, 22], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_aggregate_from_array_field_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    arr = pl.struct_array(4, "Sag", arr=[0, 0])
    agg = pl.struct("Agg", total=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            arr[i].arr[0] = i + 1
            arr[i].arr[1] = (i + 1) * 10
        for i in pl.range(0, 4):
            agg.total = agg.total + arr[i].arr[0] + arr[i].arr[1]
        pl.setval(out, 0, agg.total)


@pytest.mark.soc("950")
def test_struct_aggregate_from_array_field():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_aggregate_from_array_field_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([110], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 6.3 User run_info loop scenario simulation
# =============================================================================

@pl.jit()
def run_info_loop_scenario_kernel(
    out: pl.Tensor[[8], pl.DT_INT32],
):
    run_infos = pl.struct_array(
        2,
        "run_info_loop",
        batch_id=0,
        innerS1Realsize=[0, 0, 0, 0],
        kv_inner_offset=[0, 0, 0, 0],
    )
    with pl.section_vector():
        for i in pl.range(0, 2):
            run_infos[i].batch_id = i
            for j in pl.range(0, 4):
                run_infos[i].innerS1Realsize[j] = i * 10 + j
            for j in pl.range(0, 4):
                run_infos[i].kv_inner_offset[j] = i * 100 + j * 10
        for j in pl.range(0, 4):
            pl.setval(out, j, run_infos[0].innerS1Realsize[j])
        for j in pl.range(0, 4):
            pl.setval(out, 4 + j, run_infos[1].kv_inner_offset[j])


@pytest.mark.soc("950")
def test_run_info_loop_scenario():
    _check_npu()
    out = torch.zeros(8, device=ST_DEVICE, dtype=torch.int32)
    run_info_loop_scenario_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 1, 2, 3, 100, 110, 120, 130], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_array_arr_field_write_const_idx,
        test_struct_array_arr_field_write_loop_idx,
        test_struct_array_arr_field_write_dynamic_idx,
        test_struct_array_arr_field_cross_slot,
        test_struct_array_arr_field_all_slots_all_elems,
        test_struct_array_arr_field_for_slot_fill,
        test_struct_array_arr_field_if_branch,
        test_struct_array_arr_field_while_ring,
        test_struct_array_arr_field_break,
        test_struct_array_arr_field_slot_alias,
        test_struct_array_arr_field_chain_alias,
        test_struct_array_arr_field_pass_by_ref,
        test_struct_array_to_struct_array_field,
        test_struct_aggregate_from_array_field,
        test_run_info_loop_scenario,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All struct_array array field NPU tests passed!")
