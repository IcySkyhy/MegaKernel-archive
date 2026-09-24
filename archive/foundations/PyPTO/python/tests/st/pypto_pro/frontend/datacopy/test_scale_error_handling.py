# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""P5: 参数互斥与错误处理测试

验证 scale 参数的错误处理和互斥校验：
1. per-channel 互斥校验（与 relu/phase/dual-mode 的互斥，scale=Tile）
2. scale 类型错误（非法类型）
3. scale=Tensor 拒绝（Tensor 自动路径已移除）
4. per-channel 维度错误（1D/3D/0D tile、[N,1]、col 未对齐）
5. 旧参数废弃验证（pre_quant_scalar/fp_tile）

错误在 kernel 首次调用（lazy 编译、AST 解析）时触发。hook/builder 抛出的
ValueError/TypeError 会被 parse_target_program 包装为 ParserSyntaxError，
因此这里统一断言 ParserSyntaxError 并匹配其 message 原文。
"""

import pypto_pro.language as pl
from pypto_pro.language.parser.diagnostics._exceptions import ParserSyntaxError, ParserTypeError
import pytest
import torch

import pypto


def _mkl() -> "pl.TileType":
    return pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)


def _make_acc():
    return pl.make_tile(_mkl(), addr=0x0000, size=16384)


def _make_qk() -> tuple[torch.Tensor, torch.Tensor]:
    device = "cpu"
    q = torch.randn(64, 64, dtype=torch.float32, device=device)
    k = torch.eye(64, dtype=torch.float32, device=device)
    return q, k


# ============================================================================
# 1. per-channel 互斥校验（scale=Tile）
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_with_relu():
    """per-channel scale 不能与 relu_pre_mode 同时使用"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile, relu_pre_mode=pl.ReluPreMode.NormalRelu)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="cannot be used together with relu_pre_mode"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_with_phase():
    """per-channel scale 不能与 phase 同时使用"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile, phase=pl.STPhase.Partial)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="cannot be combined with phase"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_tensor_rejected():
    """scale 传 GM Tensor 应在解析期被拒绝（Tensor 自动路径已移除）"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        scale_tensor: pl.Tensor[[1, pl.DYNAMIC], pl.DT_INT64],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=scale_tensor)

    q, k = _make_qk()
    scale_tensor = torch.zeros(1, 64, dtype=torch.int64)
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="scale Tensor is not supported for per-channel quantization"):
        kernel(q, k, scale_tensor, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_uint8_output():
    """scale 量化到 UINT8 输出不被硬件支持，应在解析期被拒绝（而非设备 507015）"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=2.0)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.uint8)
    with pytest.raises(ParserSyntaxError, match="UINT8"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_fp32_to_bf16():
    """FP32→BF16 带 scale 不被硬件支持（仅支持无 scale 截断），应在解析期拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=2.0)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.bfloat16)
    with pytest.raises(ParserSyntaxError, match="BF16"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_int32_to_bf16():
    """INT32→BF16 带 scale 不被硬件支持（硬件只支持 INT32→FP16 反量化），应在解析期拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    ):
        with pl.section_cube():
            acc_type = pl.TileType(
                shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024
            )
            acc = pl.make_tile(acc_type, addr=0x0000, size=16384)
            pl.store(out, acc, [0, 0], scale=0.01)

    q = torch.randint(-8, 8, (64, 64), dtype=torch.int8)
    k = torch.eye(64, dtype=torch.int8)
    out = torch.zeros(64, 64, dtype=torch.bfloat16)
    with pytest.raises(ParserSyntaxError, match="BF16"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_tile_not_scaling():
    """scale 传非 Scaling 空间的 Tile 应在解析期被拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            mat_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Mat, layout=pl.ND)
            mat_tile = pl.make_tile(mat_type, addr=0x8000, size=512)
            pl.store(out, acc, [0, 0], scale=mat_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="MemorySpace.Scaling"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_move_dual_mode():
    """per-channel scale 在 move 时只支持 single-mode"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        vec_type = pl.TileType(shape=[64, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec)
        vec_tile = pl.make_tile(vec_type, addr=0x0000, size=4096)
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[1, 64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.move(vec_tile, acc, scale=fp_tile, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="only supports single-mode"):
        kernel(q, k, out)


# ============================================================================
# 2. scale 类型错误
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_string():
    """scale 不能是字符串"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale="invalid")

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="scale must be"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_list():
    """scale 不能是列表"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=[1.0, 2.0])

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="scale must be"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_dict():
    """scale 不能是字典"""
    # 注意：dict 字面量在 AST 解析阶段就抛 ParserTypeError早于
    # _resolve_scale_param 的类型检查），故此处断言 UnsupportedFeatureError。

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale={"value": 2.0})

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserTypeError, match="Unsupported closure variable type: dict"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_none_type():
    """scale 不能是自定义对象"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=object())


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_scale_unsupported_scalar_dtype():
    """运行时标量只支持 FP32 / INT32 / INT64，FP16/BF16 等其余 dtype 应在解析期拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        scale_val: pl.DT_FP16,
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], scale=scale_val)

    q, k = _make_qk()
    scale_val = torch.tensor(2.0, dtype=torch.float16)
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="scale runtime scalar dtype fp16 is not supported"):
        kernel(q, k, scale_val, out)


# ============================================================================
# 3. per-channel 维度错误（scale=Tile）
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_1d_tile():
    """per-channel scale tile 必须是 2D，不能是 1D"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[64], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="TileType only supports rank-2 shape"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_3d_tile():
    """per-channel scale tile 必须是 2D，不能是 3D"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[1, 64, 1], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="TileType only supports rank-2 shape"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_0d_tile():
    """per-channel scale tile 必须是 2D，不能是 0D scalar"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="TileType only supports rank-2 shape"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_n1_tile():
    """[N, 1] 逐行 scale 不被硬件 FixPipe 支持，应在解析期被拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[64, 1], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match=r"shape \[1, N\].*row == 1"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_per_channel_col_not_aligned():
    """[1, N] 的 N 必须为 16 的倍数（128B 对齐），否则应在解析期被拒绝"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            fp_type = pl.TileType(shape=[1, 8], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Scaling)
            fp_tile = pl.make_tile(fp_type, addr=0x0000, size=512)
            pl.store(out, acc, [0, 0], scale=fp_tile)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="multiple of 16"):
        kernel(q, k, out)


# ============================================================================
# 4. 旧参数废弃验证
# ============================================================================


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_legacy_pre_quant_scalar():
    """旧参数 pre_quant_scalar 已废弃，应抛出 TypeError"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], pre_quant_scalar=0x40000000)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="unexpected keyword argument.*pre_quant_scalar"):
        kernel(q, k, out)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_err_legacy_fp_tile():
    """旧参数 fp_tile 已废弃，应抛出 TypeError（per-channel 统一走 scale= 入口）"""

    @pl.jit()
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT8],
    ):
        with pl.section_cube():
            acc = _make_acc()
            pl.store(out, acc, [0, 0], fp_tile=acc)

    q, k = _make_qk()
    out = torch.zeros(64, 64, dtype=torch.int8)
    with pytest.raises(ParserSyntaxError, match="unexpected keyword argument.*fp_tile"):
        kernel(q, k, out)
