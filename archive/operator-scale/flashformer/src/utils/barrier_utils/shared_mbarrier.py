import cheetah.api as ch
from cheetah.api import ty
from cheetah.index_tools import Dims, Indices
from ..cuda_utils import get_shared_ptr


class SharedMBarrierToken:
    def __init__(self, state: ch.Expr):
        self.state = state


class SharedMBarrier:
    def __init__(self, ptr: ch.Expr):
        assert ptr.type() == ty.ptr_mut(ty.u64)
        self.generic_ptr = ptr
        self.ptr = get_shared_ptr(ptr)
        ch.set_name("mbar_shmem", self.ptr)

    def init(self, thread_count: int):
        ch.asm(
            f"mbarrier.init.shared.b64 [$ptr], {thread_count};",
            ptr=self.ptr,
        )

    def arrive(self) -> SharedMBarrierToken:
        state = ch.asm_volatile(
            "mbarrier.arrive.shared.b64 $state, [$ptr];",
            state=ty.u64,
            ptr=self.ptr,
        )["state"]
        return SharedMBarrierToken(state)

    def wait(self, token: SharedMBarrierToken):
        token_ptr = ch.alloc(token.state.type())
        token_ptr.val = token.state
        ch.raw_stmt(
            """
            reinterpret_cast<cuda::barrier<cuda::thread_scope_block> *>($generic_ptr)
                ->wait(std::move(
                    *reinterpret_cast<cuda::barrier<cuda::thread_scope_block>::arrival_token *>($token)));""",
            generic_ptr=self.generic_ptr,
            token=token_ptr,
        )


class SharedMBarrierArray:
    def __init__(self, dims: Dims, ptr: ch.Expr, num_barriers: int):
        assert ptr.type() == ty.ptr_mut(ty.u64)
        self.dims = dims
        self.generic_ptr = ptr
        self.ptr = get_shared_ptr(ptr)
        ch.set_name("mbar_shmem_array", self.ptr)
        self.num_barriers = num_barriers
        self.barriers = [
            SharedMBarrier(self.generic_ptr.offset(i)) for i in range(num_barriers)
        ]

    def __getitem__(self, idx: int | ch.Expr) -> SharedMBarrier:
        return SharedMBarrier(self.generic_ptr.offset(idx))

    def init(self, ix: Indices, thread_count: int):
        for barrier in self.barriers:
            barrier.init(thread_count)

    def arrive(self, barrier_idx: int | ch.Expr) -> SharedMBarrierToken:
        state = ch.asm_volatile(
            "mbarrier.arrive.shared.b64 $state, [$ptr];",
            state=ty.u64,
            ptr=get_shared_ptr(self.generic_ptr.offset(barrier_idx)),
        )["state"]
        return SharedMBarrierToken(state)

    def wait(self, barrier_idx: int | ch.Expr, token: SharedMBarrierToken):
        token_ptr = ch.alloc(token.state.type())
        token_ptr.val = token.state
        ch.raw_stmt(
            """
            reinterpret_cast<cuda::barrier<cuda::thread_scope_block> *>($generic_ptr)
                ->wait(std::move(
                    *reinterpret_cast<cuda::barrier<cuda::thread_scope_block>::arrival_token *>($token)));""",
            generic_ptr=self.generic_ptr.offset(barrier_idx),
            token=token_ptr,
        )
