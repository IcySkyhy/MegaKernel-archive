import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch
import math
import numpy as np

from flashtransformer.components.attention.sync_attention import SyncCoreAttnOutFn
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.components.memory.sync_copy import WGSyncCopyFn
from flashtransformer.utils.cuda_utils import init_groups, check_tile_size, printf
from flashtransformer.utils.barrier_utils import (
    LocalBarrierManager,
    GlobalBarrierManager,
)
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
    shmem_helper,
    safe_extern_shared,
)

# Test configuration/
n_kv_heads = 3
query_per_kv = 4  # Queries per KV head
per_kv_block = 2  # Number of blocks per KV head
n_blocks = (
    n_kv_heads * per_kv_block
)  # Each block handles one (kv_head, per_kv_block) pair
n_warps = 8
n_threads = n_warps * 32  # 32 threads per warp
n_heads = n_kv_heads * query_per_kv
d_head = 128
debug = True


def gen_test_kernel(dtype=ty.bf16, accum_dtype=ty.f32):
    """Generate test kernel for sync core attention output."""

    # Create dimensions
    dims = ixt.Dims()
    block_idx = dims.new_dim("block_idx", n_blocks)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    thread_idx = dims.new_dim_with_eq("thread_idx", (warp_idx, lane_idx))

    # Model dimensions
    kv_head_idx = dims.new_dim("kv_head_idx", n_kv_heads)
    query_per_kv_head_idx = dims.new_dim("query_per_kv_head_idx", query_per_kv)
    d_head_idx = dims.new_dim("d_head_idx", d_head)
    per_kv_block_idx = dims.new_dim("per_kv_block_idx", per_kv_block)

    # Define configurations for SyncCoreAttnOutFn
    core_attention_write_idx = dims.new_dim_with_eq(
        "core_attention_write_idx",
        (query_per_kv_head_idx, d_head_idx),
    )

    global_o_idx = dims.new_dim_with_eq(
        "global_o_idx",
        (kv_head_idx, query_per_kv_head_idx, d_head_idx),
    )

    shared_out_cfg = SDIPConfig(
        dims,
        core_attention_write_idx,
        ty.ptr_mut(accum_dtype),
        "shared",
        "shared_out_cfg",
    )

    global_o_cfg = SDIPConfig(
        dims,
        global_o_idx,
        ty.ptr_mut(dtype),
        "global",
        "global_o_cfg",
    )

    # Create the sync core attention function
    sync_fn = SyncCoreAttnOutFn(
        dims=dims,
        shared_out_cfg=shared_out_cfg,
        global_o_cfg=global_o_cfg,
        block_idx=block_idx,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        kv_head_idx=kv_head_idx,
        query_per_kv_head_idx=query_per_kv_head_idx,
        d_head_idx=d_head_idx,
    )

    # Define kernel parameter configs for global memory buffers
    global_o_buffer_idx = dims.new_dim_with_eq(
        "global_o_buffer_idx",
        (kv_head_idx, query_per_kv_head_idx, d_head_idx),
    )

    kernel_global_max_denom_cfg = SDIPConfig(
        dims,
        sync_fn.global_max_denom_idx,
        ty.ptr_mut(accum_dtype),
        "global",
        "kernel_global_max_denom",
    )

    kernel_global_o_buffer_cfg = SDIPConfig(
        dims,
        global_o_buffer_idx,
        ty.ptr_mut(dtype),
        "global",
        "kernel_global_o_buffer",
    )

    to_load_md_idx = dims.new_dim_with_eq(
        "to_load_md_idx",
        (block_idx, sync_fn.shared_block_md_idx),
    )
    to_load_md_cfg = SDIPConfig(
        dims,
        to_load_md_idx,
        ty.ptr_mut(accum_dtype),
        "global",
        "to_load_md_cfg",
    )

    to_load_out_idx = dims.new_dim_with_eq(
        "to_load_out_idx",
        (block_idx, core_attention_write_idx),
    )

    to_load_out_cfg = SDIPConfig(
        dims,
        to_load_out_idx,
        ty.ptr_mut(accum_dtype),
        "global",
        "to_load_out_cfg",
    )

    md_copy_fn = WGSyncCopyFn(
        dims,
        to_load_md_cfg,
        sync_fn.shared_block_md_cfg,
        thread_idx,
        sync_fn.shared_block_md_idx,
    )

    out_copy_fn = WGSyncCopyFn(
        dims,
        to_load_out_cfg,
        sync_fn.shared_out_cfg,
        thread_idx,
        core_attention_write_idx,
    )

    # Create global barrier index
    global_barrier_idx = dims.new_dim("global_barrier_idx", 1)

    # Create barrier managers
    local_barrier_manager = LocalBarrierManager()
    global_barrier_manager = GlobalBarrierManager(
        dims, global_barrier_idx, block_idx, thread_idx
    )

    @ch.kernel(
        ch.Params(
            to_load_out=ty.ptr_mut(accum_dtype),
            to_load_max_denom=ty.ptr_mut(accum_dtype),
            global_max_denom=ty.ptr_mut(accum_dtype),
            global_o_buffer=ty.ptr_mut(dtype),
            global_barrier_buffer=ty.ptr_mut(ty.i32),
        )
    )
    def test_kernel(
        to_load_out,
        to_load_max_denom,
        global_max_denom,
        global_o_buffer,
        global_barrier_buffer,
    ):

        # Initialize cheetah indices

        ix = dims.init()

        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))
        shared_ptr = safe_extern_shared(ix, ty.bf16, thread_idx)
        global_barrier_manager.set_buffer(ix, global_barrier_buffer)

        # Setup shared memory for sync function's internal use

        # Allocate shared memory for the attention output (per block)
        shared_out_ptr, shared_ptr = shared_out_cfg.alloc_dynamic_shared(ix, shared_ptr)

        shared_block_max_denom_ptr, shared_ptr = (
            sync_fn.shared_block_md_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )

        to_load_out = to_load_out_cfg.wrap_ptr(ix, to_load_out)
        to_load_max_denom = to_load_md_cfg.wrap_ptr(ix, to_load_max_denom)

        # Initialize groups and tile
        groups = init_groups([32])
        tile = groups[f"tile32"]

        # Initialize local barrier
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        debug_print = make_cond_print(
            ch.and_(ix[block_idx] == 0, ix[thread_idx] == 0),
            debug,
            local_barrier,
        )
        # Wrap global memory pointers
        global_max_denom_ptr = kernel_global_max_denom_cfg.wrap_ptr(
            ix, global_max_denom
        )

        global_o_buffer_ptr = kernel_global_o_buffer_cfg.wrap_ptr(ix, global_o_buffer)
        md_copy_fn(ix, to_load_max_denom, shared_block_max_denom_ptr, local_barrier)
        shared_ptr = sync_fn.setup_shmem(ix, shared_ptr, shared_block_max_denom_ptr)
        out_copy_fn(ix, to_load_out, shared_out_ptr, local_barrier)
        debug_print("to_load_max_denom[0]: %f", to_load_max_denom.raw_idx(0))
        debug_print(
            "shared_block_max_denom[0]: %f", shared_block_max_denom_ptr.raw_idx(0)
        )
        debug_print("shared_out_ptr[0]: %f", shared_out_ptr.raw_idx(0))

        sync_fn(
            ix,
            shared_out_ptr=shared_out_ptr,
            global_o_buffer=global_o_buffer_ptr,
            global_max_denom_ptr=global_max_denom_ptr,
            global_barrier_manager=global_barrier_manager,
            local_barrier=local_barrier,
            tile=tile,
        )
        debug_print(
            "global_o_buffer_ptr[0]: %f", global_o_buffer_ptr.raw_idx(0).cast(ty.f32)
        )

    @ch.fn(
        ch.Params(
            to_load_out=ty.ptr_mut(accum_dtype),
            to_load_max_denom=ty.ptr_mut(accum_dtype),
            global_max_denom=ty.ptr_mut(accum_dtype),
            global_o_buffer=ty.ptr_mut(dtype),
            global_barrier_buffer=ty.ptr_mut(ty.i32),
        )
    )
    def test_kernel_launch(
        to_load_out,
        to_load_max_denom,
        global_max_denom,
        global_o_buffer,
        global_barrier_buffer,
    ):
        arg_exprs = [
            to_load_out,
            to_load_max_denom,
            global_max_denom,
            global_o_buffer,
            global_barrier_buffer,
        ]
        shared_mem = shmem_helper([shared_out_cfg] + list(sync_fn.get_shmem_configs()))
        launch_helper(test_kernel, n_blocks, n_threads, arg_exprs, shared_mem)

    @ch.fn(
        ch.Params(
            to_load_out=ty.tensor_mut(accum_dtype),
            to_load_max_denom=ty.tensor_mut(accum_dtype),
            global_o_buffer=ty.tensor_mut(dtype),
        )
    )
    def test_kernel_torch(
        to_load_out,
        to_load_max_denom,
        global_o_buffer,
    ):

        global_barrier_buffer = ch.alloc_tensor(ty.i32, 1)
        global_md_buffer = ch.alloc_tensor(accum_dtype, n_blocks * query_per_kv * 2)

        test_kernel_launch(
            to_load_out.cast(ty.ptr_mut(accum_dtype)),
            to_load_max_denom.cast(ty.ptr_mut(accum_dtype)),
            global_md_buffer.cast(ty.ptr_mut(accum_dtype)),
            global_o_buffer.cast(ty.ptr_mut(dtype)),
            global_barrier_buffer.cast(ty.ptr_mut(ty.i32)),
        )

    return test_kernel_torch


def ref_kernel(to_load_out, to_load_max_denom, global_o_buffer):
    """Reference implementation for sync core attention output."""

    # for each kv head, compute new max_denom
    print("to_load_max_denom.flatten()[0:3]:", to_load_max_denom.flatten()[0:3])
    maxs = to_load_max_denom[:, :, 0].view(n_kv_heads, per_kv_block, query_per_kv)
    denom = to_load_max_denom[:, :, 1].view(n_kv_heads, per_kv_block, query_per_kv)

    per_kv_max = maxs.max(dim=1, keepdim=True).values
    kv_denoms = torch.sum(torch.exp(maxs - per_kv_max) * denom, dim=1, keepdim=True)

    to_load_out = to_load_out.view(n_kv_heads, per_kv_block, query_per_kv, d_head)
    out = (
        to_load_out
        * torch.exp(maxs - per_kv_max).view(n_kv_heads, per_kv_block, query_per_kv, 1)
        * denom.view(n_kv_heads, per_kv_block, query_per_kv, 1)
        / kv_denoms.view(n_kv_heads, 1, query_per_kv, 1)
    ).sum(dim=1)
    global_o_buffer.copy_(out.view(n_kv_heads, query_per_kv, d_head))


if __name__ == "__main__":

    args = launch_args()
    print("Codegen started")
    dtype = ty.bf16
    accum_dtype = ty.f32
    torch_dtype = torch.bfloat16
    torch_accum_dtype = torch.float32
    if not args.no_codegen:
        run_test_codegen(
            gen_test_kernel(dtype, accum_dtype),
            "sync_core_attn_out_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )

        print("Codegen finished, importing kernel")
    else:
        print("Codegen disabled, importing kernel")
    sync_core_attn_out_test = import_codegen("sync_core_attn_out_test", dir="tests")
    print("Kernel imported")

    to_load_out = (
        torch.tensor([1, 2], dtype=torch_accum_dtype)
        .repeat(128, query_per_kv, 3)
        .transpose(0, 2)
    ).contiguous()
    # TODO: setup automated testing to handle these kinds of things automatically
    # init_fns, contiguous, etc.
    to_load_out = args.init_fn(
        (n_blocks, query_per_kv, d_head), dtype=torch_accum_dtype
    ).cuda()
    to_load_max_denom = (
        args.init_fn((query_per_kv, 2), dtype=torch_accum_dtype).repeat(n_blocks, 1, 1)
    ).cuda()
    # to_load_max_denom = (
    #    torch.tensor([1, 2], dtype=torch_accum_dtype)
    #    .repeat(2, query_per_kv, 3)
    #    .transpose(0, 2)
    # ).contiguous()
    # print(to_load_max_denom[0, 0], to_load_max_denom[1, 0])
    # to_load_max_denom = torch.abs(
    #     args.init_fn((n_blocks, query_per_kv, 2), dtype=torch_accum_dtype)
    # )
    global_o_buffer_ref = torch.zeros(
        (n_kv_heads, query_per_kv, d_head), dtype=torch_dtype
    )
    global_o_buffer_ker = torch.zeros(
        (n_kv_heads, query_per_kv, d_head), dtype=torch_dtype, device="cuda"
    )

    ref_kernel(to_load_out, to_load_max_denom, global_o_buffer_ref)
    print("Ref kernel executed")
    sync_core_attn_out_test(to_load_out, to_load_max_denom, global_o_buffer_ker)
    print("Kernel executed")

    ker_cpu = global_o_buffer_ker.cpu()
    print("ref", global_o_buffer_ref)
    print("ker", ker_cpu)
    print(torch.sum(torch.pow(global_o_buffer_ref - ker_cpu, 2)))
