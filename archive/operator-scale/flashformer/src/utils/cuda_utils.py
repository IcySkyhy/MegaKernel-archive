import cheetah.api as ch
from cheetah.api import ty
from .constants import N_THREADS_PER_WARP

from typing import TypeAlias, Union


def get_shared_ptr(ptr: ch.Expr) -> ch.Expr:
    return ch.raw_expr("uint32_t(__cvta_generic_to_shared($ptr))", ty.u32, ptr=ptr)


def printf(fmt: str, *args: ch.Expr | str, **kwargs: ch.Expr | str):
    arg_strs = []
    if len(args) == 0:
        ch.raw_stmt(f'printf("{fmt}\\n");', **kwargs)
    else:
        for i in range(len(args)):
            if type(args[i]) is ch.Expr:
                # assert type is int or float
                assert (
                    args[i].type() == ty.f32
                    or args[i].type() == ty.i32
                    or args[i].type() == ty.u32
                ), f"printf needs int or float, got {args[i].type()}"
                arg_strs.append(f"$val{i}")
                kwargs[f"val{i}"] = args[i]
            else:
                arg_strs.append(str(args[i]))
        ch.raw_stmt(
            f'printf("{fmt}\\n", {", ".join([str(arg) for arg in arg_strs])});',
            **kwargs,
        )


def ceil_div(x: ch.Expr, y: ch.Expr) -> ch.Expr:
    return (x + y - 1) // y


def WARP_SHFL_DOWN_NATIVE(
    value: ch.Expr,
    lane_mask: ch.Expr,
    width: ch.Expr | None = None,
    mask: ch.Expr | None = None,
) -> ch.Expr:
    if width is None:
        width = ch.const(N_THREADS_PER_WARP, ty.i32)
    if mask is None:
        mask = ch.const(0xFFFFFFFF, ty.u32)
    return ch.raw_expr(
        "__shfl_down_sync($mask, $value, $lane_mask, $width)",
        value.type(),
        mask=mask,
        value=value,
        lane_mask=lane_mask,
        width=width,
    )


def warp_reduce_old(val: ch.Expr) -> ch.Expr:
    offset = N_THREADS_PER_WARP // 2
    while offset > 0:
        val += WARP_SHFL_DOWN_NATIVE(
            val,
            lane_mask=ch.const(offset, ty.i32),
            width=ch.const(N_THREADS_PER_WARP, ty.i32),
        )
        offset //= 2
    return val


def warp_reduce(val: ch.Expr) -> ch.Expr:
    return group_reduce(val, "tile32")


def group_reduce(val: ch.Expr, group: str, mode: str = "sum") -> ch.Expr:
    assert "tile" in group or "block" in group
    if mode == "sum":
        op = "plus"
    elif mode == "max":
        op = "greater"
    else:
        raise NotImplementedError(f"reduce mode {mode}")

    elem_ty = val.type()
    # TODO  don't directly copy tile32, assert threads is a valid number
    ch.raw_stmt(f"{group}.sync();")

    match elem_ty:
        # required in python 3.10 with pattern matching
        case typ if typ == ty.ptr_mut(ty.bf16):
            val.val = ch.raw_expr(
                f"cooperative_groups::reduce({group}, $val, cooperative_groups::{op}<__nv_bfloat16>())",
                ty.bf16,
                val=val.val,
            )
        case typ if typ == ty.ptr_mut(ty.f32):
            val.val = ch.raw_expr(
                f"cooperative_groups::reduce({group}, $val, cooperative_groups::{op}<float>())",
                ty.f32,
                val=val.val,
            )
        case _:
            raise NotImplementedError(f"group_reduce: {elem_ty}")
    return val


def init_block_atomic(name: str = "block_atomic", type: ch.Type = ty.f32) -> str:
    if type == ty.f32:
        ch.raw_stmt(
            f"__shared__  cuda::atomic<float, cuda::thread_scope_block> {name};"
        )
    elif type == ty.bf16:
        ch.raw_stmt(
            f"__shared__  cuda::atomic<__nv_bfloat16, cuda::thread_scope_block> {name};"
        )
    # ch.raw_stmt(f"__shared__  cuda::atomic<float, cuda::thread_scope_block> {name};")
    return name


def mul_gelu(in_val: ch.Expr, gate_val: ch.Expr) -> ch.Expr:
    # can optimize with tanh approximation
    # debug without glu for now
    # gate_val.val = ch.raw_expr("fmaxf(0.0, $fval)",gate_val.val.type(), fval=gate_val.val)
    # ch.raw_stmt('printf("%f\\n", $fval);', fval=gate_val.val)
    gate_val.val = ch.raw_expr(
        "$fval * 0.5 * (1.0 + erff($fval * 0.70710678))",
        gate_val.val.type(),
        fval=gate_val.val,
    )
    # ch.raw_stmt('printf("%f\\n", $fval);', fval=gate_val.val)
    return in_val.val * gate_val.val


def mul_gelu_bf16(in_val: ch.Expr, gate_val: ch.Expr) -> ch.Expr:
    gate_val_f32 = ch.alloc(ty.f32)
    gate_val_f32.val = gate_val.val.cast(ty.f32)
    gate_val_f32.val = ch.raw_expr(
        "$fval * 0.5 * (1.0 + erff($fval * 0.70710678))",
        gate_val_f32.val.type(),
        fval=gate_val_f32.val,
    )
    return in_val.val * gate_val_f32.val.cast(ty.bf16)
    # inner = ch.alloc(ty.f32, 0.0)
    # inner.val = ch.raw_expr(f"tanhf({bf16c(0.79788456)} * ($x + {bf16c(0.044715)} * $x * $x * $x))", inner.val.type(), x=gate_val.val)
    # gate_val.val = ch.raw_expr(f"$x * {bf16c(0.5)} * ({bf16c(1.0)} + __float2bfloat16($inner))",
    #                               gate_val.val.type(), x=gate_val.val, inner=inner.val)
    return in_val.val * gate_val.val


def mul_swiglu(in_val: ch.Expr, gate_val: ch.Expr) -> ch.Expr:

    gate_val.val = ch.raw_expr(
        "$fval / (1.0 + __expf(-$fval))",
        ty.f32,
        fval=gate_val.val,
    )
    return in_val.val * gate_val.val


def mul_swiglu_bf16(in_val: ch.Expr, gate_val: ch.Expr) -> ch.Expr:
    gate_val_f32 = ch.alloc(ty.f32)
    gate_val_f32.val = gate_val.val.cast(ty.f32)
    gate_val_f32.val = ch.raw_expr(
        "$fval / (1.0 + __expf(-$fval))",
        ty.f32,
        fval=gate_val_f32.val,
    )
    return in_val.val * gate_val_f32.val.cast(ty.bf16)


# alias Tile to a string
Tile: TypeAlias = str
Grid: TypeAlias = str
Block: TypeAlias = str

Group = Union[Tile, Grid, Block]


# TODO: change this to use custom types
def init_groups(tile_sizes: list[int] = [N_THREADS_PER_WARP]) -> dict[str, Group]:
    # make sure tile_sizes has unique values
    tile_sizes = list(set(tile_sizes))
    ch.raw_stmt("cooperative_groups::grid_group g = cooperative_groups::this_grid();")
    # ch.raw_stmt("cuda::barrier<cuda::thread_scope_device>::arrival_token tok;")
    ch.raw_stmt(
        "cooperative_groups::thread_block block = cooperative_groups::this_thread_block();"
    )
    for tile_size in tile_sizes:
        ch.raw_stmt(
            f"cooperative_groups::thread_block_tile<{tile_size}> tile{tile_size} = cooperative_groups::tiled_partition<{tile_size}>(block);"
        )
    # ch.raw_stmt("thread_block_tile<32> tile32 = tiled_partition<32>(block);")
    groups = {"grid": "g", "block": "block"}
    for tile_size in tile_sizes:
        groups[f"tile{tile_size}"] = f"tile{tile_size}"
    return groups


def check_tile_size(group: Group, size: int):
    assert group == f"tile{size}", f"group: {group}, size: {size}"


def get_elem_size(ptr_or_type: ch.Expr | ch.Type) -> int:
    if type(ptr_or_type) is ch.Expr:
        elem_ty = ptr_or_type.type()
    elif type(ptr_or_type) is ch.Type:
        elem_ty = ptr_or_type
    else:
        raise ValueError(f"Invalid type: {type(ptr_or_type)}")
    # check if elem_ty is a pointer type
    if elem_ty.try_to_ptr() is not None:
        elem_ty = elem_ty.try_to_ptr()[0]
    else:
        elem_ty = elem_ty
    match (elem_ty):
        case ty.f32:
            return 4
        case ty.bf16:
            return 2
        case ty.bf16x2:
            return 4
        case ty.u8:
            return 1
        case ty.u16:
            return 2
        case ty.f32x2:
            return 8
        case ty.i32:
            return 4
        case ty.i64:
            return 8
        case ty.ix4:
            return 16
        case _:
            raise NotImplementedError(f"get_elem_size: {elem_ty}")


def get_cast_dtype(size: int) -> ch.Type:
    if size > 16:
        print(
            "trying to cast to type with size > 16. You should probably have a loop here"
        )
    match (size):
        case 4:
            return ty.i32  # "int"
        case 8:
            return ty.i64  # "int2"
        case 16:
            return ty.ix4  # "int4"
        case _:
            raise NotImplementedError(f"get_cast_dtype: {size}")


def sync(group: str):
    ch.raw_stmt(f"{group}.sync();")


def get_default_elems_per_transaction(ptr_type: ch.Expr | ch.Type, memtype: str) -> int:
    elem_size = get_elem_size(ptr_type)
    if memtype == "global":
        # vectorized loads / stores
        if elem_size == 4:
            return 4
        elif elem_size == 2:
            return 8
        else:
            raise ValueError(f"No default for global and {elem_size} bit elements")
    elif memtype == "shared":
        # 32 bit with vector loads banks
        if elem_size == 4:
            return 4
        elif elem_size == 2:
            return 8
        else:
            raise ValueError(f"No default for shared and {elem_size} byte elements")
    else:
        raise ValueError(f"No defaults for memtype {memtype}")


def c_exp(x: ch.Expr) -> ch.Expr:
    val = ch.raw_expr("expf($x)", ty.f32, x=x.cast(ty.f32))
    return val
