"""Warp-level max/sum reduction helpers used by the MLA online softmax."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm


@dsl_user_op
def fmax(
    a,
    b,
    c=None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    if cutlass.const_expr(cutlass.CUDA_VERSION.major == 12):
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )
    return Float32(
        nvvm.fmax(
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val=None,
    arch: cutlass.Constexpr[int] = 100,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return (
            local_max[0]
            if const_expr(init_val is None)
            else fmax(local_max[0], init_val)
        )
    else:
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i + 0], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val=None,
    arch: cutlass.Constexpr[int] = 100,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
    else:
        res = cute.make_fragment(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [
            local_sum_0,
            (res[2], res[3]),
            (res[4], res[5]),
            (res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(
                local_sum[0], (res[i + 0], res[i + 1])
            )
            local_sum[1] = cute.arch.add_packed_f32x2(
                local_sum[1], (res[i + 2], res[i + 3])
            )
            local_sum[2] = cute.arch.add_packed_f32x2(
                local_sum[2], (res[i + 4], res[i + 5])
            )
            local_sum[3] = cute.arch.add_packed_f32x2(
                local_sum[3], (res[i + 6], res[i + 7])
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]

