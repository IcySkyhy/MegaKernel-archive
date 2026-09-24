# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade megakernel context aggregates (Context / Scoreboard / TaskBaseInfo / OpContext).
# Stages and layouts follow aicore.cpp LocalContext / ring / handshake.

import triton
import triton.language as tl
import triton_dist.language as dl

# Stage constants (match aicore.cpp)
STAGE_START = tl.constexpr(1)
STAGE_AICPU_HS = tl.constexpr(2)
STAGE_READY = tl.constexpr(3)
STAGE_START_LOOP = tl.constexpr(4)
STAGE_QUIT = tl.constexpr(5)
STAGE_QUIT_ACK = tl.constexpr(6)

QUIT_OPCODE = tl.constexpr(0xFFFF)
TASK_RING_SIZE = tl.constexpr(128)

@triton.jit
def global_block_idx_mix():
    # TODO: implementation of customop
    return dl.ascend.global_block_idx_mix()

@triton.jit
def global_block_idx():
    return tl.program_id(axis=0)

@triton.jit
def query_ts_condition():
    # TODO: implementation of customop
    return dl.ascend.query_ts_condition()

@triton.jit
def wait_ts_ready():
    """wait taskSchedueler"""
    while query_ts_condition() == 0:
        pass

@triton.jit
def dcci(ptr, mode, data_cache_kind):
    # TODO: implementation of hivm::dcciop
    return dl.ascend.dcci(ptr, mode, data_cache_kind)

@tl.core._aggregate
class TaskBaseInfo:
    v_ref: tl.tensor #uint8

    TASK_BASE_INFO_SIZE: tl.constexpr
    OP_ID_OFFSET: tl.constexpr
    TILE_ID_OFFSET: tl.constexpr
    TILE_COUNT_OFFSET: tl.constexpr
    TASK_MODE_OFFSET: tl.constexpr
    SCOREBOARD_PTR_OFFSET: tl.constexpr
    DATA_PTR_OFFSET: tl.constexpr
    DEPEND_INFO_OFFSET: tl.constexpr

    @triton.constexpr_function
    def __init__(self, v_ref):
        self.v_ref = v_ref
        self.TASK_BASE_INFO_SIZE = tl.constexpr(64)
        self.OP_ID_OFFSET = tl.constexpr(0)
        self.TILE_ID_OFFSET = tl.constexpr(2)
        self.TILE_COUNT_OFFSET = tl.constexpr(4)
        self.TASK_MODE_OFFSET = tl.constexpr(6)
        self.SCOREBOARD_PTR_OFFSET = tl.constexpr(8)
        self.DATA_PTR_OFFSET = tl.constexpr(12)
        self.DEPEND_INFO_OFFSET = tl.constexpr(16)

    @triton.jit
    def get_opId(self):
        return tl.load(
            (self.v_ref.to(tl.uint64) + self.OP_ID_OFFSET).to(tl.pointer_type(tl.uint16))
        ).to(tl.int32)

    @triton.jit
    def get_tileid(self):
        return tl.load(
            (self.v_ref.to(tl.uint64) + self.TILE_ID_OFFSET).to(tl.pointer_type(tl.uint16))
        ).to(tl.int32)

    @triton.jit
    def get_tilecnt(self):
        return tl.load(
            (self.v_ref.to(tl.uint64) + self.TILE_COUNT_OFFSET).to(tl.pointer_type(tl.uint16))
        ).to(tl.int32)

    @triton.jit
    def get_data(self):
        # return tl.uint64 instead of tl.pointer_type(tl.int64) because the value is used for condition
        return tl.load(
            (self.v_ref.to(tl.uint64) + self.DATA_PTR_OFFSET).to(tl.pointer_type(tl.uint64))
        )

    @triton.jit
    def scoreboard_ref(self):
        ptr = tl.load(
            (self.v_ref + self.SCOREBOARD_PTR_OFFSET).to(tl.pointer_type(tl.uint64))
        ).to(tl.pointer_type(tl.uint8))
        return Scoreboard(ptr)

@tl.core._aggregate
class Scoreboard:
    """LocalContext over AicLocal: status, fetch_task, handshake (aicore.cpp)."""
    v_ref: tl.tensor # uint8

    TASK_BASE_INFO_SIZE: tl.constexpr
    LAST_INFO_OFFSET: tl.constexpr
    BLOCK_ID_OFFSET: tl.constexpr
    CORE_ID_OFFSET: tl.constexpr
    SUB_BLOCK_ID_OFFSET: tl.constexpr
    GLOBAL_BLOCK_ID_OFFSET: tl.constexpr
    DFX_STATUS_OFFSET: tl.constexpr
    TASK_BASE_INFO_OFFSET : tl.constexpr

    @triton.constexpr_function
    def __init__(self, v_ref):
        self.v_ref = v_ref
        self.TASK_BASE_INFO_SIZE = tl.constexpr(64)
        self.LAST_INFO_OFFSET = tl.constexpr(0)
        self.BLOCK_ID_OFFSET = tl.constexpr(8)
        self.CORE_ID_OFFSET = tl.constexpr(10)
        self.SUB_BLOCK_ID_OFFSET = tl.constexpr(12)
        self.GLOBAL_BLOCK_ID_OFFSET = tl.constexpr(14)
        self.DFX_STATUS_OFFSET = tl.constexpr(16)
        self.TASK_BASE_INFO_OFFSET = tl.constexpr(64) # __attribute__(aligned(64))

    @triton.jit
    def task_base_info_ref(self, task_idx):
        return TaskBaseInfo(self.v_ref + self.TASK_BASE_INFO_OFFSET + task_idx * self.TASK_BASE_INFO_SIZE)

    @triton.jit
    def get_block_id(self):
        return tl.load((self.v_ref.to(tl.uint64) + self.BLOCK_ID_OFFSET).to(tl.pointer_type(tl.uint16))).to(tl.uint16)

    @triton.jit
    def set_block_id(self, val):
        dl.ascend.st_io(self.v_ref.to(tl.uint64) + self.BLOCK_ID_OFFSET, tl.cast(val, tl.uint16))

    @triton.jit
    def get_core_id(self):
        return tl.load((self.v_ref.to(tl.uint64) + self.CORE_ID_OFFSET).to(tl.pointer_type(tl.uint16))).to(tl.uint16)

    @triton.jit
    def set_core_id(self, val):
        dl.ascend.st_io(self.v_ref.to(tl.uint64) + self.CORE_ID_OFFSET, tl.cast(val, tl.uint16))

    @triton.jit
    def get_sub_block_id(self):
        return tl.load((self.v_ref.to(tl.uint64) + self.SUB_BLOCK_ID_OFFSET).to(tl.pointer_type(tl.uint16))).to(tl.uint16)

    @triton.jit
    def set_sub_block_id(self, val):
        dl.ascend.st_io(self.v_ref.to(tl.uint64) + self.SUB_BLOCK_ID_OFFSET, tl.cast(val, tl.uint16))

    @triton.jit
    def get_global_block_id(self):
        return tl.load((self.v_ref.to(tl.uint64) + self.GLOBAL_BLOCK_ID_OFFSET).to(tl.pointer_type(tl.uint16))).to(tl.uint16)

    @triton.jit
    def set_global_block_id(self, val):
        dl.ascend.st_io(self.v_ref.to(tl.uint64) + self.GLOBAL_BLOCK_ID_OFFSET, tl.cast(val, tl.uint16))

    @triton.jit
    def get_status(self):
        dl.ascend.dcci(self.v_ref + self.DFX_STATUS_OFFSET, tl.constexpr(1), tl.constexpr(0))
        return tl.load((self.v_ref.to(tl.uint64) + self.DFX_STATUS_OFFSET).to(tl.pointer_type(tl.uint32)))

    @triton.jit
    def set_status(self, val):
        dl.ascend.set_status(self.v_ref.to(tl.uint64) + self.DFX_STATUS_OFFSET, tl.cast(val, tl.uint32))

    @triton.jit
    def reset(self):
        # reset register: set_cond(0)
        # TODO: implementation of customop
        dl.ascend.dcci(self.v_ref, tl.constexpr(1), tl.constexpr(0))
        dl.ascend.set_cond(tl.cast(0, tl.uint64))

    @triton.jit
    def wait(self):
        while (self.get_status() != STAGE_AICPU_HS):
            pass

    @triton.jit
    def fetch_task(self, idx):
        ring = self.task_base_info_ref(idx)
        # TODO: implementation of customop
        dl.ascend.dcci(ring.v_ref, tl.constexpr(1), tl.constexpr(0))
        return ring

    @triton.jit
    def wait_deps(self, depend_info):
        """Wait on depend_info via Ascend custom op."""
        # TODO: implementation of customop
        dl.ascend.wait_deps(depend_info)

    @triton.jit
    def release_tile(self, scoreboard_update, tile_id):
        """Notify tile done via Ascend custom op."""
        # TODO: implementation of customop
        dl.ascend.notify_tile_done(scoreboard_update, tile_id)

    @triton.jit
    def quit(self):
        """退出握手。"""
        self.set_status(STAGE_QUIT)
        while self.get_status() != STAGE_QUIT_ACK:
            pass

    @triton.jit
    def set_lastInfo(self, val):
        dl.ascend.st_io(self.v_ref.to(tl.uint64) + self.LAST_INFO_OFFSET, tl.cast(val, tl.uint64))

@tl.core._aggregate
class OpContext:
    """Abstraction of OpCtx.
    """
    ptr: tl.tensor  # __gm__ OpCtx*

    ADDR_OFFSET: tl.constexpr
    MIX_AIV_OFFSET: tl.constexpr
    ARGS_OFFSET: tl.constexpr
    TILING_OFFSET: tl.constexpr
    TYPE_OFFSET: tl.constexpr
    KERNEL_TYPE_OFFSET: tl.constexpr
    END_POS_OFFSET: tl.constexpr

    @triton.constexpr_function
    def __init__(self, ptr):
        self.ptr = ptr
        self.ADDR_OFFSET = tl.constexpr(0)
        self.MIX_AIV_OFFSET = tl.constexpr(8)
        self.ARGS_OFFSET = tl.constexpr(16)
        self.TILING_OFFSET = tl.constexpr(20)
        self.TYPE_OFFSET = tl.constexpr(24)
        self.KERNEL_TYPE_OFFSET = tl.constexpr(28)
        self.END_POS_OFFSET = tl.constexpr(32)

    @triton.jit
    def addr(self):
        # return tl.uint64 instead of tl.pointer_type(tl.uint8) 
        # because the value is recast as uint64 in set_lastInfo
        return tl.load((self.ptr + self.ADDR_OFFSET).to(tl.pointer_type(tl.uint64)))

    @triton.jit
    def args_offset(self):
        return tl.load((self.ptr + self.ARGS_OFFSET).to(tl.pointer_type(tl.uint32)))

    @triton.jit
    def tiling_offset(self):
        return tl.load((self.ptr + self.TILING_OFFSET).to(tl.pointer_type(tl.uint32)))

@tl.core._aggregate
class Context:
    """Abstraction of AicCtx.
    """
    ctx_ptr: tl.tensor  # __gm__ AicCtx*

    LOCAL_OFFSET: tl.constexpr  # __gm__ LocalContext*
    CBHASH_OFFSET: tl.constexpr
    OPLEN_OFFSET: tl.constexpr
    BLOCKDIM_OFFSET: tl.constexpr
    AICCNT_OFFSET: tl.constexpr
    AIVCNT_OFFSET: tl.constexpr
    DEVID_OFFSET: tl.constexpr
    FFTSADDR_OFFSET: tl.constexpr
    IOCAPACITY_OFFSET: tl.constexpr
    WORKSPACE_OFFSET: tl.constexpr
    OPCTX_OFFSET: tl.constexpr

    LOCAL_CONTEXT_SIZE: tl.constexpr
    OPCTX_SIZE: tl.constexpr

    @triton.constexpr_function
    def __init__(self, ctx_ptr):
        self.ctx_ptr = ctx_ptr
        self.LOCAL_OFFSET = tl.constexpr(0)
        self.CBHASH_OFFSET = tl.constexpr(8)
        self.OPLEN_OFFSET = tl.constexpr(16)
        self.BLOCKDIM_OFFSET = tl.constexpr(18)
        self.AICCNT_OFFSET = tl.constexpr(20)
        self.AIVCNT_OFFSET = tl.constexpr(22)
        self.DEVID_OFFSET = tl.constexpr(24)
        self.FFTSADDR_OFFSET = tl.constexpr(32)
        self.IOCAPACITY_OFFSET = tl.constexpr(40)
        self.WORKSPACE_OFFSET = tl.constexpr(48)
        self.OPCTX_OFFSET = tl.constexpr(56)
        # AicLocal size: 64 (header) + 128 (ring) * 64 (TaskEle aligned 64)
        self.LOCAL_CONTEXT_SIZE = tl.constexpr(8256)
        self.OPCTX_SIZE = tl.constexpr(64)

    @triton.jit
    def local_block_ref(self):
        # AicCtx.local is a __gm__ AicLocal*. Load as uint64 then cast to
        # uint8* (references default to uint8*).
        return tl.load((self.ctx_ptr + self.LOCAL_OFFSET).to(tl.pointer_type(tl.uint64))).to(tl.pointer_type(tl.uint8))

    @triton.jit
    def scoreboard_ref(self, block_idx):
        # Index by block_idx * LOCAL_CONTEXT_SIZE to pick the slot.
        return Scoreboard(self.local_block_ref() + block_idx * self.LOCAL_CONTEXT_SIZE)

    @triton.jit
    def opcontext_ref(self, op_idx):
        # opCtx[] array starts at OPCTX_OFFSET; each OpCtx is OPCTX_SIZE bytes.
        return OpContext(self.ctx_ptr + self.OPCTX_OFFSET + op_idx * self.OPCTX_SIZE)

