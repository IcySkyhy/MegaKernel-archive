import cheetah.api as ch
from cheetah.api import ty

import cheetah.index_tools as ixt

from flashtransformer.components.memory import WGSyncCopyFn
from flashtransformer.utils import SafeDataIndexPtr, SDIPConfig, LocalBarrierManager
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_args,
    launch_helper,
    shmem_helper,
)


def gen_test_kernel(n_elem, n_thread):
    dims = ixt.Dims()
    thread_idx = dims.new_dim("thread_idx", n_thread)
    elem_idx = dims.new_dim("elem_idx", n_elem)

    warp_idx = dims.new_dim("warp_idx")
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))
    src_cfg = SDIPConfig(dims, elem_idx, ty.ptr_const(ty.f32), "global", "src")
    dst_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "global", "dst")

    shared_cfg = SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "shared", "cache")
    sync_copy_out_fn = WGSyncCopyFn(
        dims,
        shared_cfg,
        dst_cfg,
        thread_idx,
        elem_idx,
    )

    @ch.kernel(ch.Params(x=ty.ptr_const(ty.f32), y=ty.ptr_mut(ty.f32)))
    def sync_copy_kernel(x, y):

        local_barrier_manager = LocalBarrierManager()
        local_barrier = local_barrier_manager.get_local_barrier(n_thread)
        sync_copy_in_fn = WGSyncCopyFn(
            dims,
            src_cfg,
            shared_cfg,
            thread_idx,
            elem_idx,
        )

        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        global_x_ptr = SafeDataIndexPtr(ix, x, elem_idx, "global")
        global_y_ptr = SafeDataIndexPtr(ix, y, elem_idx, "global")

        shared_ptr = SafeDataIndexPtr.from_shared_static(ix, elem_idx, ty.f32, "src")

        ch.raw_stmt("__syncthreads();")
        sync_copy_in_fn(ix, global_x_ptr, shared_ptr, local_barrier)
        sync_copy_out_fn(ix, shared_ptr, global_y_ptr, local_barrier)

    @ch.fn(ch.Params(x=ty.ptr_const(ty.f32), y=ty.ptr_mut(ty.f32)))
    def sync_copy_launch(x, y):
        sdips = [SDIPConfig(dims, elem_idx, ty.ptr_mut(ty.f32), "shared", "cache")]
        shared_mem = shmem_helper(sdips)
        arg_exprs = [x, y]
        launch_helper(sync_copy_kernel, 1, n_thread, arg_exprs, shared_mem)

    @ch.fn(ch.Params(x=ty.tensor_const(ty.f32), y=ty.tensor_mut(ty.f32)))
    def sync_copy_torch(x, y):
        sync_copy_launch(x.cast(ty.ptr_const(ty.f32)), y.cast(ty.ptr_mut(ty.f32)))

    return sync_copy_torch


import torch


def sync_copy_ref(x, y):
    y[:] = x[:]


if __name__ == "__main__":
    parser = launch_args(return_parser=True)
    parser.add_argument("--n-elem", type=int, default=512)
    parser.add_argument("--n-thread", type=int, default=1024)
    args = parser.parse_args()

    if not args.no_codegen:
        run_test_codegen(
            gen_test_kernel(args.n_elem, args.n_thread),
            f"sync_copy_kernel_test_{args.n_elem}_{args.n_thread}",
            headers=standard_headers,
            bind=True,
        )

    if args.test:
        copy_kernel = import_codegen(
            f"sync_copy_kernel_test_{args.n_elem}_{args.n_thread}"
        )

        src = torch.ones(args.n_elem)
        dst_ref = torch.zeros(args.n_elem)
        dst_kernel = torch.zeros(args.n_elem)

        copy_kernel(src, dst_kernel)
        sync_copy_ref(src, dst_ref)

        assert torch.allclose(dst_kernel, dst_ref)
        print("Test passed!")
