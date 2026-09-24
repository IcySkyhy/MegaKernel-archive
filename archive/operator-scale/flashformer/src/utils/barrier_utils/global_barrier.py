import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import Dims, DimName, Indices
from ..cuda_utils import printf
from ..safe_data_index_ptr_utils import SafeDataIndexPtr
from .local_barrier import LocalBarrier


class GlobalBarrierManager:
    def __init__(
        self,
        dims: Dims,
        global_barrier_idx: DimName,
        block_idx: DimName,
        thread_idx: DimName,
    ):
        self.dims = dims
        self.global_barrier_idx = global_barrier_idx
        self.global_barriers = [-1] * self.dims.size(global_barrier_idx)
        self.block_idx = block_idx
        self.thread_idx = thread_idx
        self.barrier_scope = dims.new_scope("barrier")
        self.arrive_count = 0
        self.wait_count = 0

    def arrive(
        self,
        ix: Indices,
        barrier_id: int,
        local_barrier: LocalBarrier,
    ) -> ch.Expr:
        debug = False
        self.arrive_count += 1
        self.global_barrier_buffer.check_memtype("global")
        assert self.global_barriers[barrier_id] == -1
        check_val = ch.alloc(ty.i32, 0)
        value_to_add = ch.alloc(ty.i32, 0)
        with ix.scope(self.barrier_scope):
            ix.set_index(self.global_barrier_idx, barrier_id)
            ch.raw_stmt("__threadfence();")
            local_barrier.wait()
            with ch.if_(ix[self.thread_idx] == 0):
                with ch.if_else(ix[self.block_idx] == 0) as branch:
                    with branch.then():
                        value_to_add.val = ch.const(-2147483647, ty.i32) - ch.const(
                            ix.size(self.block_idx), ty.i32
                        )
                    with branch.else_():
                        value_to_add.val = 1
                check_val.val = ch.raw_expr(
                    "atomicAdd($addr, $val)",
                    ty.i32,
                    addr=self.global_barrier_buffer.idx_offset(),
                    val=value_to_add.val,
                )
                if debug:
                    printf(
                        f"arrive count {self.arrive_count} block %d value_to_add %d check_val %d",
                        ix[self.block_idx],
                        value_to_add.val,
                        check_val.val,
                    )
        self.global_barriers[barrier_id] = 0
        return check_val

    def set_buffer(self, ix, global_barrier_buffer: SafeDataIndexPtr | ch.Expr):
        if isinstance(global_barrier_buffer, SafeDataIndexPtr):
            global_barrier_buffer = global_barrier_buffer.check_convert(
                self.global_barrier_idx, "global", ty.ptr_mut(ty.i32)
            )
        elif isinstance(global_barrier_buffer, ch.Expr):
            global_barrier_buffer = SafeDataIndexPtr(
                ix,
                global_barrier_buffer,
                self.global_barrier_idx,
                "global",
                "global_barrier_buffer",
            )
        else:
            raise ValueError(
                f"Invalid type: for global_barrier_buffer: {type(global_barrier_buffer)}"
            )
        self.global_barrier_buffer = global_barrier_buffer

    def wait(
        self,
        ix: Indices,
        barrier_id: int,
        check_val: ch.Expr,
        local_barrier: LocalBarrier,
    ):
        debug = False
        self.wait_count += 1
        assert self.global_barriers[barrier_id] == 0
        with ix.scope(self.barrier_scope):
            ix.set_index(self.global_barrier_idx, barrier_id)
            with ch.if_(ix[self.thread_idx] == 0):
                new_val = ch.alloc(ty.i32, 0)

                ch.raw_stmt(
                    """
                    volatile int32_t* lock_ptr = (volatile int32_t*)($addr);
                    do{
                        *$new_val = *lock_ptr;
                    }
                    while ((*$new_val ^ *$check_val) > -1);
                    """,
                    addr=self.global_barrier_buffer.idx_offset(),
                    new_val=new_val,
                    check_val=check_val,
                )
                if debug:
                    printf(
                        f"wait count {self.wait_count} block %d new_val %d check_val %d new_val^check_val %d",
                        self.block_idx,
                        new_val.val,
                        check_val.val,
                        ch.raw_expr(
                            "($new_val ^ $check_val)",
                            ty.i32,
                            new_val=new_val.val,
                            check_val=check_val.val,
                        ),
                    )
        # ch.raw_stmt("__threadfence();")
        self.global_barriers[barrier_id] = -1
        local_barrier.wait()

    def arrive_and_wait(
        self,
        ix: Indices,
        barrier_id: int,
        local_barrier: LocalBarrier,
    ):
        check_val = self.arrive(
            ix, barrier_id, local_barrier
        )
        self.wait(ix, barrier_id, check_val, local_barrier)
