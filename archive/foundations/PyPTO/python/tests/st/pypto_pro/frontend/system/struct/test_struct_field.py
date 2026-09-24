# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU coverage for ``pl.struct`` array field (list-valued field) scenarios.

Covers array field write/read, control-flow interaction (for/if/while),
aliasing, reference passing, multi-field combinations, whole-assign,
dtype coverage (FP32 / BOOL), and a user ``run_info`` scenario simulation.

Merged from the former ``test_struct_field_dtype.py`` (FP16/BF16 clones removed;
FP32 and BOOL retained for dtype coverage).
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
# 1. Element write / read-modify-write
# =============================================================================

@pl.jit()
def struct_arr_field_write_all_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sw2", arr=[0, 0, 0, 0])
    with pl.section_vector():
        s.arr[0] = 10
        s.arr[1] = 20
        s.arr[2] = 30
        s.arr[3] = 40
        for i in pl.range(0, 4):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_array_field_write_all():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_write_all_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 20, 30, 40], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_write_read_modify_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("Sw4", arr=[0, 0, 0, 0])
    with pl.section_vector():
        s.arr[0] = 5
        s.arr[0] = s.arr[0] + 10
        pl.setval(out, 0, s.arr[0])


@pytest.mark.soc("950")
def test_struct_array_field_write_read_modify():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_write_read_modify_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_write_in_loop_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sw3", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            s.arr[i] = i * 10
        for i in pl.range(0, 4):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_array_field_write_in_loop():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_write_in_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 10, 20, 30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 2. for loop + array field
# =============================================================================

@pl.jit()
def struct_arr_field_for_sum_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("Sf1", arr=[10, 20, 30, 40])
    with pl.section_vector():
        total = 0
        for i in pl.range(0, 4):
            total = total + s.arr[i]
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
def test_struct_arr_field_for_sum():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_for_sum_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_for_fill_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sf2", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            s.arr[i] = i * i
        for i in pl.range(0, 4):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_arr_field_for_fill():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_for_fill_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 1, 4, 9], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_for_accum_from_scalar_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sf3", base=100, acc=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            s.acc[i] = s.acc[i] + s.base + i
        for i in pl.range(0, 4):
            pl.setval(out, i, s.acc[i])


@pytest.mark.soc("950")
def test_struct_arr_field_for_accum_from_scalar():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_for_accum_from_scalar_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100, 101, 102, 103], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 3. if/else + array field
# =============================================================================

@pl.jit()
def struct_arr_field_if_branch_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("Si1", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            if i < 2:
                s.arr[0] = 10
            else:
                s.arr[0] = 20
        pl.setval(out, 0, s.arr[0])


@pytest.mark.soc("950")
def test_struct_arr_field_if_branch():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_if_branch_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([20], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_if_else_diff_idx_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("Si2", arr=[0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            if i < 2:
                s.arr[0] = s.arr[0] + 1
            else:
                s.arr[1] = s.arr[1] + 1
        pl.setval(out, 0, s.arr[0])
        pl.setval(out, 1, s.arr[1])


@pytest.mark.soc("950")
def test_struct_arr_field_if_else_diff_idx():
    _check_npu()
    out = torch.zeros(2, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_if_else_diff_idx_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([2, 2], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 4. struct alias + array field
# =============================================================================

@pl.jit()
def struct_arr_field_alias_loop_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sa2", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            t = s
            t.arr[i] = i * 10
        for i in pl.range(0, 4):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_arr_field_alias_loop():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_alias_loop_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 10, 20, 30], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_chain_alias_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    s = pl.struct("Sa3", arr=[0, 0, 0, 0])
    with pl.section_vector():
        ref1 = s
        ref2 = ref1
        ref2.arr[0] = 99
        pl.setval(out, 0, s.arr[0])


@pytest.mark.soc("950")
def test_struct_arr_field_chain_alias():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_chain_alias_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([99], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 5. Function pass-by-reference + array field
# =============================================================================

def _fill_arr_field(s, idx, val):
    s.arr[idx] = val


@pl.jit()
def struct_arr_field_pass_by_ref_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    s = pl.struct("Sr1", arr=[0, 0, 0])
    with pl.section_vector():
        _fill_arr_field(s, 0, 100)
        _fill_arr_field(s, 1, 200)
        _fill_arr_field(s, 2, 300)
        for i in pl.range(0, 3):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_arr_field_pass_by_ref():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_pass_by_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100, 200, 300], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 6. Mixed scalar + array field
# =============================================================================

@pl.jit()
def struct_mixed_scalar_array_ops_kernel(
    out: pl.Tensor[[5], pl.DT_INT32],
):
    s = pl.struct("Sm1", n=0, offsets=[0, 0, 0, 0], scale=0)
    with pl.section_vector():
        s.n = 5
        s.scale = 10
        for i in pl.range(0, 4):
            s.offsets[i] = s.n + i * s.scale
        pl.setval(out, 0, s.n)
        pl.setval(out, 1, s.scale)
        pl.setval(out, 2, s.offsets[0])
        pl.setval(out, 3, s.offsets[1])
        pl.setval(out, 4, s.offsets[3])


@pytest.mark.soc("950")
def test_struct_mixed_scalar_array_ops():
    _check_npu()
    out = torch.zeros(5, device=ST_DEVICE, dtype=torch.int32)
    struct_mixed_scalar_array_ops_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([5, 10, 5, 15, 35], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_multi_array_cross_ref_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
):
    s = pl.struct("Sm2", a=[0, 0, 0], b=[10, 20, 30])
    with pl.section_vector():
        for i in pl.range(0, 3):
            s.a[i] = s.b[i] + i
        for i in pl.range(0, 3):
            pl.setval(out, i, s.a[i])


@pytest.mark.soc("950")
def test_struct_multi_array_cross_ref():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.int32)
    struct_multi_array_cross_ref_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 21, 32], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 7. Two struct array field copy + run_info scenario
# =============================================================================

@pl.jit()
def two_struct_array_field_copy_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    a = pl.struct("Sa", arr=[10, 20, 30, 40])
    b = pl.struct("Sb", arr=[0, 0, 0, 0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            b.arr[i] = a.arr[i]
        for i in pl.range(0, 4):
            pl.setval(out, i, b.arr[i])


@pytest.mark.soc("950")
def test_two_struct_array_field_copy():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    two_struct_array_field_copy_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10, 20, 30, 40], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def run_info_scenario_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
):
    run_infos = pl.struct(
        "run_info",
        batch_id=0,
        s1_idx=0,
        s2_idx=0,
        bo_idx=0,
        n2o_idx=0,
        go_idx=0,
        s2o_idx=0,
        s1o_idx=0,
        s2_cv_begin=0,
        sl_real_size=0,
        s2_real_size=0,
        inner_s1_loop_num=0,
        inner_s2_loop_num=0,
        innerS1Realsize=[0, 0, 0, 0],
        kv_inner_offset=[0, 0, 0, 0],
        maxsum_offset=0,
        query_offset=0,
    )
    with pl.section_vector():
        run_infos.batch_id = 7
        run_infos.sl_real_size = 128
        run_infos.innerS1Realsize[0] = 32
        run_infos.innerS1Realsize[1] = 64
        run_infos.innerS1Realsize[2] = 96
        run_infos.innerS1Realsize[3] = 128
        run_infos.kv_inner_offset[0] = 10
        run_infos.kv_inner_offset[1] = 20
        run_infos.kv_inner_offset[2] = 30
        run_infos.kv_inner_offset[3] = 40
        run_infos.maxsum_offset = 512
        run_infos.query_offset = 1024
        pl.setval(out, 0, run_infos.batch_id)
        pl.setval(out, 1, run_infos.sl_real_size)
        pl.setval(out, 2, run_infos.innerS1Realsize[0] + run_infos.innerS1Realsize[3])
        pl.setval(out, 3, run_infos.innerS1Realsize[1] + run_infos.innerS1Realsize[2])
        pl.setval(out, 4, run_infos.kv_inner_offset[0] + run_infos.kv_inner_offset[3])
        pl.setval(out, 5, run_infos.kv_inner_offset[1] + run_infos.kv_inner_offset[2])


@pytest.mark.soc("950")
def test_run_info_scenario():
    _check_npu()
    out = torch.zeros(6, device=ST_DEVICE, dtype=torch.int32)
    run_info_scenario_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([7, 128, 160, 160, 50, 50], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 8. Whole array field assignment (s.arr = [v0, v1, ...])
# =============================================================================

@pl.jit()
def struct_arr_field_whole_assign_overwrite_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sw7", arr=[1, 2, 3, 4])
    with pl.section_vector():
        s.arr = [100, 200, 300, 400]
        for i in pl.range(0, 4):
            pl.setval(out, i, s.arr[i])


@pytest.mark.soc("950")
def test_struct_array_field_whole_assign_overwrite():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_whole_assign_overwrite_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([100, 200, 300, 400], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_whole_assign_fp32_kernel(
    out: pl.Tensor[[3], pl.DT_FP32],
):
    s = pl.struct("Sw8", farr=[0.0, 0.0, 0.0])
    with pl.section_vector():
        s.farr = [1.5, 2.5, 3.5]
        for i in pl.range(0, 3):
            pl.setval(out, i, s.farr[i])


@pytest.mark.soc("950")
def test_struct_array_field_whole_assign_fp32():
    _check_npu()
    out = torch.zeros(3, device=ST_DEVICE, dtype=torch.float32)
    struct_arr_field_whole_assign_fp32_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1.5, 2.5, 3.5], device=ST_DEVICE, dtype=torch.float32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_arr_field_whole_assign_mixed_kernel(
    out: pl.Tensor[[5], pl.DT_INT32],
):
    s = pl.struct("Sw9", n=0, arr=[0, 0, 0, 0])
    with pl.section_vector():
        s.n = 7
        s.arr = [10, 20, 30, 40]
        total = 0
        for i in pl.range(0, 4):
            total = total + s.arr[i]
        pl.setval(out, 0, s.n)
        pl.setval(out, 1, total)
        pl.setval(out, 2, s.arr[0])
        pl.setval(out, 3, s.arr[2])
        pl.setval(out, 4, s.arr[3])


@pytest.mark.soc("950")
def test_struct_array_field_whole_assign_mixed():
    _check_npu()
    out = torch.zeros(5, device=ST_DEVICE, dtype=torch.int32)
    struct_arr_field_whole_assign_mixed_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([7, 100, 10, 30, 40], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 9. Dtype coverage: FP32 + BOOL (merged from test_struct_field_dtype.py)
#    FP16/BF16 clones removed — same code path as FP32, only precision differs.
# =============================================================================

@pl.jit()
def struct_fp32_arr_write_read_kernel(
    out: pl.Tensor[[4], pl.DT_FP32],
):
    s = pl.struct("Sfp32w", x=0, farr=[0.0, 0.0, 0.0, 0.0])
    with pl.section_vector():
        s.farr[0] = 1.5
        s.farr[1] = 2.5
        s.farr[2] = 3.5
        s.farr[3] = 4.5
        for i in pl.range(0, 4):
            pl.setval(out, i, s.farr[i])


@pytest.mark.soc("950")
def test_struct_fp32_arr_write_read():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.float32)
    struct_fp32_arr_write_read_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1.5, 2.5, 3.5, 4.5], device=ST_DEVICE, dtype=torch.float32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_fp32_arr_loop_fill_kernel(
    out: pl.Tensor[[4], pl.DT_FP32],
):
    s = pl.struct("Sfp32l", farr=[0.0, 0.0, 0.0, 0.0])
    with pl.section_vector():
        for i in pl.range(0, 4):
            s.farr[i] = 1.0 * i + 0.5
        for i in pl.range(0, 4):
            pl.setval(out, i, s.farr[i])


@pytest.mark.soc("950")
def test_struct_fp32_arr_loop_fill():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.float32)
    struct_fp32_arr_loop_fill_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0.5, 1.5, 2.5, 3.5], device=ST_DEVICE, dtype=torch.float32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_fp32_arr_accum_kernel(
    out: pl.Tensor[[1], pl.DT_FP32],
):
    s = pl.struct("Sfp32a", farr=[1.0, 2.0, 3.0, 4.0])
    with pl.section_vector():
        total = 0.0
        for i in pl.range(0, 4):
            total = total + s.farr[i]
        pl.setval(out, 0, total)


@pytest.mark.soc("950")
def test_struct_fp32_arr_accum():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.float32)
    struct_fp32_arr_accum_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([10.0], device=ST_DEVICE, dtype=torch.float32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_fp32_arr_modify_kernel(
    out: pl.Tensor[[1], pl.DT_FP32],
):
    s = pl.struct("Sfp32m", farr=[0.0, 0.0, 0.0, 0.0])
    with pl.section_vector():
        s.farr[0] = 10.0
        s.farr[0] = s.farr[0] + 5.5
        pl.setval(out, 0, s.farr[0])


@pytest.mark.soc("950")
def test_struct_fp32_arr_modify():
    _check_npu()
    out = torch.zeros(1, device=ST_DEVICE, dtype=torch.float32)
    struct_fp32_arr_modify_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([15.5], device=ST_DEVICE, dtype=torch.float32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# --- BOOL array field (unique: bool-as-array-element with if-consumption) ---

@pl.jit()
def struct_bool_arr_read_init_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sboolr", flags=[True, False, True, False])
    with pl.section_vector():
        for i in pl.range(0, 4):
            if s.flags[i]:
                pl.setval(out, i, 1)
            else:
                pl.setval(out, i, 0)


@pytest.mark.soc("950")
def test_struct_bool_arr_read_init():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_bool_arr_read_init_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 0, 1, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_bool_arr_write_read_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sboolw", flags=[True, False, True, False])
    with pl.section_vector():
        s.flags[0] = False
        s.flags[1] = True
        for i in pl.range(0, 4):
            if s.flags[i]:
                pl.setval(out, i, 1)
            else:
                pl.setval(out, i, 0)


@pytest.mark.soc("950")
def test_struct_bool_arr_write_read():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_bool_arr_write_read_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 1, 1, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


@pl.jit()
def struct_bool_arr_loop_cond_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sbooll", flags=[False, False, False, False])
    with pl.section_vector():
        for i in pl.range(0, 4):
            if i % 2 == 0:
                s.flags[i] = True
            else:
                s.flags[i] = False
        for i in pl.range(0, 4):
            if s.flags[i]:
                pl.setval(out, i, 1)
            else:
                pl.setval(out, i, 0)


@pytest.mark.soc("950")
def test_struct_bool_arr_loop_cond():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_bool_arr_loop_cond_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([1, 0, 1, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# 10. External Python list as array field (closure variable, e.g. [0] * 100)
# =============================================================================

ZEROS_100 = [0] * 100


@pl.jit()
def struct_ext_list_arr_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct("Sext", arr=ZEROS_100)
    with pl.section_vector():
        s.arr[0] = 42
        s.arr[50] = 50
        s.arr[99] = 99
        pl.setval(out, 0, s.arr[0])
        pl.setval(out, 1, s.arr[50])
        pl.setval(out, 2, s.arr[99])
        pl.setval(out, 3, s.arr[1])


@pytest.mark.soc("950")
def test_struct_ext_list_arr():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_ext_list_arr_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([42, 50, 99, 0], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# 11. Two 100-length array fields: external zeros list + in-kernel literal ones
# =============================================================================
# Note: in-kernel list repetition (``[1] * 100``) is not supported — ``*`` is
# parsed as scalar multiply; a literal list of 100 ones is used instead.
# ``ZEROS_100`` (100 zeros) is shared with section 10.


@pl.jit()
def struct_two_list_fields_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
):
    s = pl.struct(
        "TwoList",
        arr0=ZEROS_100,
        arr1=[
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
    )
    with pl.section_vector():
        pl.setval(out, 0, s.arr0[0])
        pl.setval(out, 1, s.arr0[99])
        pl.setval(out, 2, s.arr1[0])
        pl.setval(out, 3, s.arr1[99])


@pytest.mark.soc("950")
def test_struct_two_list_fields():
    _check_npu()
    out = torch.zeros(4, device=ST_DEVICE, dtype=torch.int32)
    struct_two_list_fields_kernel(out)
    torch.npu.synchronize()
    expected = torch.tensor([0, 0, 1, 1], device=ST_DEVICE, dtype=torch.int32)
    assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"


# =============================================================================
# Standalone runner
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = [
        test_struct_array_field_write_all,
        test_struct_array_field_write_read_modify,
        test_struct_array_field_write_in_loop,
        test_struct_arr_field_for_sum,
        test_struct_arr_field_for_fill,
        test_struct_arr_field_for_accum_from_scalar,
        test_struct_arr_field_if_branch,
        test_struct_arr_field_if_else_diff_idx,
        test_struct_arr_field_alias_loop,
        test_struct_arr_field_chain_alias,
        test_struct_arr_field_pass_by_ref,
        test_struct_mixed_scalar_array_ops,
        test_struct_multi_array_cross_ref,
        test_two_struct_array_field_copy,
        test_run_info_scenario,
        test_struct_array_field_whole_assign_overwrite,
        test_struct_array_field_whole_assign_fp32,
        test_struct_array_field_whole_assign_mixed,
        test_struct_fp32_arr_write_read,
        test_struct_fp32_arr_loop_fill,
        test_struct_fp32_arr_accum,
        test_struct_fp32_arr_modify,
        test_struct_bool_arr_read_init,
        test_struct_bool_arr_write_read,
        test_struct_bool_arr_loop_cond,
        test_struct_ext_list_arr,
        test_struct_two_list_fields,
    ]
    for t in tests:
        t()
        logging.info("%s passed!", t.__name__)
    logging.info("All struct array field NPU tests passed!")
