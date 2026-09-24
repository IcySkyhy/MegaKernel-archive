# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Megakernel customOps. Exposed under `dl.ascend.*` (e.g. dl.ascend.wait_deps,
# dl.ascend.set_cond, dl.ascend.query_ts_condition).
#
# All megakernel-specific customOps live here:
#   - wait_deps / notify_tile_done: scoreboard synchronization
#   - query_ts_condition: AICPU task scheduler condition (DATA_MAIN_BASE)
#   - set_cond: AICore→AICPU completion signal
#   - global_block_idx_mix: mix-mode global block index
#   - dcci: cacheline invalidation
#
# C++ side:
#   - td:           include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td
#   - builder:      python/src/ir.cc (TritonOpBuilder.create_*)
#   - lowering:     lib/Conversion/TritonDistributedToHIVM/ASCEND/DistributedOpToHIVM.cpp
#   - device impl:  3rdparty/AscendNPU-IR/.../Mega.cpp (linked as bitcode)
from triton.language import core as tlc
from triton.language.core import builtin


@builtin
def wait_deps(depend_info_ptr, _semantic=None):
    assert not depend_info_ptr.type.is_block() and depend_info_ptr.type.is_ptr(), "only support scalar pointer"
    _semantic.builder.create_wait_deps(depend_info_ptr.handle)


@builtin
def notify_tile_done(scoreboard_ptr, tile_id, _semantic=None):
    assert not scoreboard_ptr.type.is_block() and scoreboard_ptr.type.is_ptr(), "only support scalar pointer"
    tile_id = _semantic.to_tensor(tile_id)
    _semantic.builder.create_notify_tile_done(scoreboard_ptr.handle, tile_id.handle)


@builtin
def query_ts_condition(_semantic=None):
    return tlc.tensor(_semantic.builder.create_query_ts_condition(), tlc.int64)


@builtin
def set_cond(cond, _semantic=None):
    cond = _semantic.to_tensor(cond)
    _semantic.builder.create_set_cond(cond.handle)


@builtin
def global_block_idx_mix(_semantic=None):
    return tlc.tensor(_semantic.builder.create_global_block_idx_mix(), tlc.int32)


@builtin
def dcci(ptr, mode, data_cache_kind, _semantic=None):
    assert not ptr.type.is_block() and ptr.type.is_ptr(), "only support scalar pointer"
    mode = _semantic.to_tensor(mode)
    data_cache_kind = _semantic.to_tensor(data_cache_kind)
    _semantic.builder.create_dcci(ptr.handle, mode.handle, data_cache_kind.handle)


@builtin
def st_io(ptr, val, _semantic=None):
    ptr = _semantic.to_tensor(ptr)
    val = _semantic.to_tensor(val)
    _semantic.builder.create_st_io(ptr.handle, val.handle)

@builtin
def set_status(addr, val, _semantic=None):
    addr = _semantic.to_tensor(addr)
    val = _semantic.to_tensor(val)
    _semantic.builder.create_set_status(addr.handle, val.handle)

__all__ = [
    "wait_deps",
    "notify_tile_done",
    "query_ts_condition",
    "set_cond",
    "global_block_idx_mix",
    "dcci",
    "st_io",
    "set_status",
]
