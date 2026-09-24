import cheetah.api as ch
from cheetah.api import ty

import cheetah.index_tools as ixt

from flashtransformer.components.memory import AsyncBulkGroupSTGRedAddFn
from flashtransformer.utils import SDIPConfig, SharedMBarrier, LocalBarrier
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
)

n_elem = 1024
n_thread = 256
n_blocks = 2


def gen_test_kernel():
    dims = ixt.Dims()
    block_idx = dims.new_dim("block_idx", n_blocks)
    thread_idx = dims.new_dim("thread_idx", n_thread)
    elem_idx = dims.new_dim("elem_idx", n_elem)

    warp_idx = dims.new_dim("warp_idx")
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))
    fill_scope = dims.new_scope("fill_scope")
    with dims.scope(fill_scope):
        fill_iter = dims.new_dim("fill_iter")
        dims.eq(elem_idx, (fill_iter, thread_idx))

    src_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "shared", "src")
    dst_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "global", "dst")

    @ch.kernel(ch.Params(x=ty.ptr_mut(ty.f32)))
    def async_red_kernel(x):
        # Initialize ScopeFns before dims.init
        async_red_fn = AsyncBulkGroupSTGRedAddFn(
            dims,
            src_cfg,
            dst_cfg,
            thread_idx,
            elem_idx,
        )

        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        # Get block index
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Create pointers
        src_ptr = src_cfg.alloc_static_shared(ix)
        dst_ptr = dst_cfg.wrap_ptr(ix, x)

        # Create a local barrier for sync
        local_barrier = LocalBarrier(1, n_thread)

        # Initialize shared memory with values based on block index
        with ix.scope(fill_scope):
            with ix.loop(fill_iter):
                src_ptr.idx_set(ix[block_idx].cast(ty.f32) + 1.0)

        # Make sure shared memory is initialized
        ch.raw_stmt("__syncthreads();")

        # Perform async reduction from shared to global
        async_red_fn(ix, src_ptr, dst_ptr)

        # Wait for all async operations to complete
        async_red_fn.wait(ix, local_barrier)

    @ch.fn(ch.Params(x=ty.ptr_mut(ty.f32)))
    def async_red_launch(x):
        kernel = async_red_kernel.void_ptr()
        arg_stack = ch.alloc_array(ty.ptr_mut(None), 1)
        arg_exprs = [x]
        args = ch.alloc_array(ty.ptr_mut(None), len(arg_exprs))
        for i, arg_expr in enumerate(arg_exprs):
            arg_stack = ch.alloc(arg_expr.type(), arg_expr)
            args[i] = arg_stack.cast(ty.ptr_mut(None))

        ch.raw_stmt(
            "cudaLaunchCooperativeKernel($kernel, $n_blocks, $n_threads, $args, 0);",
            kernel=kernel,
            n_blocks=ch.const(n_blocks, ty.u32),
            n_threads=ch.const(n_thread, ty.u32),
            args=args,
        )

    @ch.fn(ch.Params(x=ty.tensor_mut(ty.f32)))
    def async_red_torch(x):
        async_red_launch(x.cast(ty.ptr_mut(ty.f32)))

    return async_red_torch


import torch


def async_red_ref(x):
    # Each element in y will be the sum of corresponding element in x plus 1.0 and 2.0
    # Since we have 2 blocks, one adding 1.0 and another adding 2.0
    x[:] = 3.0


if __name__ == "__main__":
    run_test_codegen(
        gen_test_kernel(),
        "async_red_kernel_test",
        headers=standard_headers,
        bind=True,
        dir="tests",
    )
    red_kernel = import_codegen("async_red_kernel_test", dir="tests")

    # Initialize input data - not actually used in this test
    # src = torch.zeros(n_elem, dtype=torch.float32)

    # Initialize output with zeros
    dst_kernel = torch.zeros(n_elem, dtype=torch.float32)

    # Run the kernel
    red_kernel(dst_kernel)
    print("Kernel output:", dst_kernel[:10])  # Print first 10 elements

    # Reference implementation
    dst_ref = torch.zeros(n_elem, dtype=torch.float32)
    async_red_ref(dst_ref)
    print("Reference output:", dst_ref[:10])  # Print first 10 elements

    # Verify results
    assert torch.allclose(dst_kernel, dst_ref)
