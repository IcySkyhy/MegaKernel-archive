import cheetah.api as ch
from cheetah.api import ty

import cheetah.index_tools as ixt

from flashtransformer.components.memory import (
    AsyncBarrierGTSCopyFn,
    AsyncBulkGroupSTGCopyFn,
)
from flashtransformer.utils import SDIPConfig, SharedMBarrier, LocalBarrier
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
)

n_elem = 8192
n_thread = 1024


def gen_test_kernel():
    dims = ixt.Dims()
    thread_idx = dims.new_dim("thread_idx", n_thread)
    elem_idx = dims.new_dim("elem_idx", n_elem)

    warp_idx = dims.new_dim("warp_idx")
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    src_cfg = SDIPConfig(dims, elem_idx, ty.ptr_const(ty.f32), "global", "src")
    dst_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "global", "dst")

    @ch.kernel(ch.Params(x=ty.ptr_const(ty.f32), y=ty.ptr_mut(ty.f32)))
    def async_copy_kernel(x, y):
        shared_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "shared", "shared")
        # Initialize ScopeFns before dims.init
        async_gts_copy_fn = AsyncBarrierGTSCopyFn(
            dims,
            src_cfg,
            shared_cfg,
            thread_idx,
            elem_idx,
        )

        async_stg_copy_fn = AsyncBulkGroupSTGCopyFn(
            dims,
            shared_cfg,
            dst_cfg,
            thread_idx,
            elem_idx,
        )
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        # Create pointers using wrap_ptr from SDIPConfig
        src_ptr = src_cfg.wrap_ptr(ix, x)
        shared_ptr = shared_cfg.alloc_static_shared(ix)
        dst_ptr = dst_cfg.wrap_ptr(ix, y)

        # Allocate space for mbarrier
        barrier_ptr = ch.alloc_shared(ty.u64)
        shared_barrier = SharedMBarrier(barrier_ptr)
        local_barrier = LocalBarrier(1, n_thread)

        # Initialize the barrier
        with ch.if_(ix[thread_idx] == 0):
            shared_barrier.init(n_thread)
        ch.raw_stmt("__syncthreads();")

        # Perform async copy from global to shared
        async_gts_copy_fn(ix, src_ptr, shared_ptr, shared_barrier)

        # Wait for copy to complete
        token = shared_barrier.arrive()
        shared_barrier.wait(token)

        # Perform async copy from shared to global
        async_stg_copy_fn(ix, shared_ptr, dst_ptr)

        # # Wait for all async operations to complete
        async_stg_copy_fn.wait(ix, local_barrier)
        ch.raw_stmt("__syncthreads();")

    @ch.fn(ch.Params(x=ty.ptr_const(ty.f32), y=ty.ptr_mut(ty.f32)))
    def async_copy_launch(x, y):
        kernel = async_copy_kernel.void_ptr()
        arg_stack = ch.alloc_array(ty.ptr_mut(None), 2)
        arg_exprs = [x, y]
        args = ch.alloc_array(ty.ptr_mut(None), len(arg_exprs))
        for i, arg_expr in enumerate(arg_exprs):
            arg_stack = ch.alloc(arg_expr.type(), arg_expr)
            args[i] = arg_stack.cast(ty.ptr_mut(None))

        ch.raw_stmt(
            "cudaLaunchCooperativeKernel($kernel, 1, $n_threads, $args, 0);",
            kernel=kernel,
            n_threads=ch.const(n_thread, ty.u32),
            args=args,
        )

    @ch.fn(ch.Params(x=ty.tensor_const(ty.f32), y=ty.tensor_mut(ty.f32)))
    def async_copy_torch(x, y):
        async_copy_launch(x.cast(ty.ptr_const(ty.f32)), y.cast(ty.ptr_mut(ty.f32)))

    return async_copy_torch


import torch


def async_copy_ref(x, y):
    y[:] = x[:]


if __name__ == "__main__":
    run_test_codegen(
        gen_test_kernel(),
        "async_copy_kernel_test",
        headers=standard_headers,
        bind=True,
        dir="tests",
    )
    copy_kernel = import_codegen("async_copy_kernel_test", dir="tests")

    src = torch.ones(n_elem)
    dst_kernel = torch.zeros(n_elem)

    copy_kernel(src, dst_kernel)
    print(dst_kernel)

    dst_ref = torch.zeros(n_elem)
    async_copy_ref(src, dst_ref)
    print(dst_ref)

    assert torch.allclose(dst_kernel, dst_ref)
