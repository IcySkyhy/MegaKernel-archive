# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""inline helper 多 return 的上板测试。

`return` 在 inline helper 里被下降成 `while True: ... break` 加一个合流槽,因此返回值的
C++ 形态由合流变量的类型决定。本文件按**返回值类型**和**控制流形状**两个维度上板验证:

  1. 返回值类型:scalar、同构 tuple(背景数组)、具名 tuple(叶子槽)、struct(整体对象)、
     struct 数组(struct 的背景数组)、struct 嵌在异构 tuple 里(struct 元素是叶子)、
     异构 tuple(聚合叶子槽)、tile(整体拷贝)、以及无返回值的提前退出。
  2. 控制流形状:for 套 if return、if 套 for return、仅 then 分支 return、
     仅 else 分支 return、两个分支都 return。
  3. 分支条件一律取自运行期的 kernel 标量参数,并对 True/False 各跑一次——编译期常量会把
     分支折掉,合流槽根本不会生成,那样测的就不是合流路径了。
  4. 每个 kernel 在调用之后都先把返回值落到一个变量,再在后续语句里用它参与运算,而不是
     直接塞给唯一的消费者——合流槽必须能被读第二次。

`python/tests/ut/pypto_pro/codegen/test_cce_inline_return.py` 钉住生成的 C++ 形态,
本文件负责证明这些 C++ 真的能编译并跑出正确数值。
"""

from __future__ import annotations

import logging
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE_M = 64
TILE_N = 64


def _check_npu():
    try:
        torch.npu.set_device(ST_DEVICE)
        return True
    except RuntimeError as exc:
        pytest.skip(f"NPU unavailable: {exc}")
        return False


def _run(kernel, out_size, *args, dtype=torch.int32):
    out = torch.zeros(out_size, device=ST_DEVICE, dtype=dtype)
    kernel(out, *args)
    torch.npu.synchronize()
    return out.tolist()


# =============================================================================
# 返回值类型
# =============================================================================


# 标量返回值的各个 return 点不必是同一个 dtype,C++ 的隐式转换会兜住(见下面的控制流用例,
# 那里 `index * 10` 与 `limit` 就不同型)。但复合返回值的 C++ 表示是从类型推出来的——同构
# tuple 落成数组、struct 落成一个具名类——所以它们的每个 return 点必须给出相同的元素类型,
# 否则两条边会各自选到不同的表示。这里统一写成 `base + <常量>` 就是为此。


def _ret_pair(flag, base):
    """同构 tuple:两个元素类型相同,合流槽是一个 C++ 背景数组。"""
    if flag:
        return base + 0, base + 1
    return base + 10, base + 11


def _ret_named_tuple(flag, base):
    """具名 tuple:带字段名但没有 C++ 结构体名,合流槽平铺成每字段一个叶子槽。"""
    if flag:
        return pl.make_tuple(lo=base + 0, hi=base + 1)
    return pl.make_tuple(lo=base + 10, hi=base + 11)


def _ret_struct(flag, base):
    """struct:有自己的 C++ 类型名,合流槽是一个整体对象,整体赋值。"""
    if flag:
        return pl.struct("RetCtx", v=base + 0, w=base + 1)
    return pl.struct("RetCtx", v=base + 10, w=base + 11)


def _ret_struct_array(flag, base):
    """struct 数组:两个元素同型,合流槽是一个 struct 的背景数组,逐元素拷贝。"""
    if flag:
        return pl.struct("ArrCtx", v=base + 0, w=base + 1), pl.struct("ArrCtx", v=base + 2, w=base + 3)
    return pl.struct("ArrCtx", v=base + 10, w=base + 11), pl.struct("ArrCtx", v=base + 12, w=base + 13)


def _ret_struct_in_tuple(flag, base):
    """struct 嵌在异构 tuple 里:tuple 平铺成叶子槽,但 struct 元素本身是叶子,不再往下拆。"""
    if flag:
        return pl.struct("MixCtx", v=base + 0, w=base + 1), True
    return pl.struct("MixCtx", v=base + 10, w=base + 11), False


def _ret_aggregate(flag, base):
    """异构 tuple:元素类型不同,既不能是数组也不是结构体,平铺成叶子槽。"""
    if flag:
        return (base + 0, base + 1), True
    return (base + 10, base + 11), False


def _ret_tile(flag, tile_a, tile_b):
    """tile:合流槽只按 tile 类型声明,随后被整体拷贝覆盖(valid shape 随对象带过来)。"""
    if flag:
        return tile_a
    return tile_b


def _ret_nothing(out, base):
    """全部是 bare return:合流槽始终没有类型,不应该被物化出来。"""
    pl.setval(out, 0, base)
    if base > 0:
        return
    pl.setval(out, 0, base + 100)
    return


@pl.jit()
def inline_return_pair_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        lo, hi = _ret_pair(flag, base)
        total = lo + hi
        pl.setval(out, 0, lo)
        pl.setval(out, 1, hi)
        pl.setval(out, 2, total)


@pl.jit()
def inline_return_named_tuple_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        bounds = _ret_named_tuple(flag, base)
        total = bounds.lo + bounds.hi
        pl.setval(out, 0, bounds.lo)
        pl.setval(out, 1, bounds.hi)
        pl.setval(out, 2, total)


@pl.jit()
def inline_return_struct_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        ctx = _ret_struct(flag, base)
        total = ctx.v + ctx.w
        pl.setval(out, 0, ctx.v)
        pl.setval(out, 1, ctx.w)
        pl.setval(out, 2, total)


@pl.jit()
def inline_return_struct_array_kernel(
    out: pl.Tensor[[4], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        first, second = _ret_struct_array(flag, base)
        total = first.v + second.w
        pl.setval(out, 0, first.v)
        pl.setval(out, 1, first.w)
        pl.setval(out, 2, second.v)
        pl.setval(out, 3, total)


@pl.jit()
def inline_return_struct_in_tuple_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        ctx, taken = _ret_struct_in_tuple(flag, base)
        total = ctx.v + ctx.w
        pl.setval(out, 0, ctx.v)
        pl.setval(out, 1, total)
        if taken:
            pl.setval(out, 2, ctx.w)
        else:
            pl.setval(out, 2, 0)


@pl.jit()
def inline_return_aggregate_kernel(
    out: pl.Tensor[[3], pl.DT_INT32],
    flag: pl.DT_BOOL,
    base: pl.DT_INT32,
):
    with pl.section_vector():
        values, taken = _ret_aggregate(flag, base)
        total = values[0] + values[1]
        pl.setval(out, 0, values[0])
        pl.setval(out, 1, values[1])
        if taken:
            pl.setval(out, 2, total)
        else:
            pl.setval(out, 2, 0)


@pl.jit()
def inline_return_nothing_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
    base: pl.DT_INT32,
):
    with pl.section_vector():
        _ret_nothing(out, base)


@pl.jit(auto_mutex=True)
def inline_return_tile_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    flag: pl.DT_BOOL,
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[4, 5])
    with pl.section_vector():
        tile_a = a_db.next()
        tile_b = b_db.next()
        tile_c = c_db.next()
        pl.load_tile(tile_a, x, [0, 0])
        pl.load_tile(tile_b, y, [0, 0])
        picked = _ret_tile(flag, tile_a, tile_b)
        # 合流出来的 tile 先作为计算的输入被用一次,再由结果落盘 —— 它不能只是 store 的直传。
        pl.add(tile_c, picked, picked)
        pl.store_tile(z, tile_c, [0, 0])


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 6, 11]), (False, [15, 16, 31])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_pair(flag, expected):
    """同构 tuple 返回值经背景数组合流后,两个元素都写回正确。"""
    _check_npu()
    assert _run(inline_return_pair_kernel, 3, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 6, 11]), (False, [15, 16, 31])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_named_tuple(flag, expected):
    """具名 tuple 返回值平铺成叶子槽后,.lo / .hi 仍读到本分支写入的值。"""
    _check_npu()
    assert _run(inline_return_named_tuple_kernel, 3, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 6, 11]), (False, [15, 16, 31])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_struct(flag, expected):
    """struct 返回值作为整体对象合流:两个分支各建一个 struct,整体赋值给同一个槽。"""
    _check_npu()
    assert _run(inline_return_struct_kernel, 3, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 6, 7, 13]), (False, [15, 16, 17, 33])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_struct_array(flag, expected):
    """struct 数组:同构 tuple 走背景数组,两个 struct 元素逐个拷贝进同一个数组槽。"""
    _check_npu()
    assert _run(inline_return_struct_array_kernel, 4, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 11, 6]), (False, [15, 31, 0])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_struct_in_tuple(flag, expected):
    """struct 嵌在异构 tuple 里:struct 元素作为一个整体叶子合流,字段读回仍走合流后的对象。"""
    _check_npu()
    assert _run(inline_return_struct_in_tuple_kernel, 3, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag, expected", [(True, [5, 6, 11]), (False, [15, 16, 0])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_aggregate(flag, expected):
    """异构 tuple:嵌套的同构部分走数组,bool 走独立叶子槽,两者都随分支写回。"""
    _check_npu()
    assert _run(inline_return_aggregate_kernel, 3, flag, 5) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("base, expected", [(7, [7]), (0, [100])])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_nothing(base, expected):
    """全 bare return 的 helper:提前退出跳过后续写入,且不产生任何返回值槽。"""
    _check_npu()
    assert _run(inline_return_nothing_kernel, 1, base) == expected


@pytest.mark.soc("950")
@pytest.mark.parametrize("flag", [True, False])
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_tile(flag):
    """tile 返回值:合流槽被整体拷贝覆盖,存出的必须是被选中那个 tile 的内容。

    两条边都要对,考的是 auto_mutex 的 mutex id 有没有跟着合流:helper 的返回槽先被
    `slot = None` 种下,该种子带一份"没有 buffer"的 mutex meta,于是
    _coemit_tile_mutexid_companion 在包装循环之前就给 `__inline_0_return_val__mutexid`
    落了定义,两条边再各自写它。少了这个种子,companion 只在循环内首次定义、带不出循环,
    store 就会锁到 then 边那块 buffer,else 边的 store 与它自己那次 load 之间没有序。
    """
    _check_npu()
    shape = [TILE_M, TILE_N]
    x = torch.rand(shape, device=ST_DEVICE, dtype=torch.float16)
    y = torch.rand(shape, device=ST_DEVICE, dtype=torch.float16)
    z = torch.zeros(shape, device=ST_DEVICE, dtype=torch.float16)

    inline_return_tile_kernel(x, y, z, flag)
    torch.npu.synchronize()

    expected = (x if flag else y) * 2
    assert torch.equal(z, expected), f"flag={flag}: got {z[0, :4].tolist()}, expected {expected[0, :4].tolist()}"


# =============================================================================
# 控制流形状
# =============================================================================


def _for_if_return(limit):
    """for 套 if return:break 只跳出 for,循环后要再判一次 returned 才跳出 helper。"""
    for index in pl.range(0, limit, 1):
        if index >= 2:
            return index * 10
    return limit


def _if_for_return(flag, limit):
    """if 套 for return:传播守卫落在分支体内,而不是被提到函数作用域。"""
    if flag:
        for index in pl.range(0, limit, 1):
            if index >= 3:
                return index * 100
        return 0
    return limit


def _then_only_return(value):
    """只有 then 分支 return,另一条边落到 if 之后的 return。"""
    if value > 4:
        return value + 1
    return value


def _else_only_return(value):
    """只有 else 分支 return:then 边把合流槽留在 `None` 哨兵上,phi 的类型只能从 else 边取。"""
    if value > 4:
        value = value + 1
    else:
        return 0
    return value


def _both_branches_return(value):
    """两个分支都 return:if 合流出来的值没有任何可达的使用者。"""
    if value > 4:
        return value + 1
    else:
        return value - 1


@pl.jit()
def inline_return_shapes_kernel(
    out: pl.Tensor[[6], pl.DT_INT32],
    flag: pl.DT_BOOL,
    limit: pl.DT_INT32,
    value: pl.DT_INT32,
):
    with pl.section_vector():
        for_if = _for_if_return(limit)
        if_for = _if_for_return(flag, limit)
        then_only = _then_only_return(value)
        both = _both_branches_return(value)
        else_only = _else_only_return(value)
        # 五个合流槽同时存活,再一起参与运算:任何一个被下一次展开覆盖掉都会在 out[5] 上暴露。
        total = for_if + if_for + then_only + both + else_only
        pl.setval(out, 0, for_if)
        pl.setval(out, 1, if_for)
        pl.setval(out, 2, then_only)
        pl.setval(out, 3, both)
        pl.setval(out, 4, else_only)
        pl.setval(out, 5, total)


@pytest.mark.soc("950")
@pytest.mark.parametrize(
    "flag, limit, value, expected",
    [
        # limit=6 -> for 在 index==2 提前 return 20;if 分支进 for,在 index==3 return 300。
        # value=9 -> then-only 返回 10,both 返回 10,else-only 走 then 边(不 return)返回 10。
        (True, 6, 9, [20, 300, 10, 10, 10, 350]),
        # flag=False -> if 套 for 走 else 边,直接返回 limit。
        # value=2 -> then-only 落到 if 之后的 return,both 走 else 边,else-only 提前 return 0。
        (False, 6, 2, [20, 6, 2, 1, 0, 29]),
        # limit=1 -> for 跑完都没 return,落到循环后的 return limit。
        (True, 1, 9, [1, 0, 10, 10, 10, 31]),
    ],
)
@pypto.options(pass_options={"enable_slice": False})
def test_inline_return_control_flow_shapes(flag, limit, value, expected):
    """五种控制流形状在同一个 kernel 里各自合流,互不干扰。"""
    _check_npu()
    assert _run(inline_return_shapes_kernel, 6, flag, limit, value) == expected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_inline_return_pair(True, [5, 6, 11])
    test_inline_return_named_tuple(True, [5, 6, 11])
    test_inline_return_struct(True, [5, 6, 11])
    test_inline_return_struct_array(True, [5, 6, 7, 13])
    test_inline_return_struct_in_tuple(True, [5, 11, 6])
    test_inline_return_aggregate(True, [5, 6, 11])
    test_inline_return_nothing(7, [7])
    test_inline_return_tile(True)
    test_inline_return_control_flow_shapes(True, 6, 9, [20, 300, 10, 10, 10, 350])
    logging.info("test_inline_return_kinds passed!")
