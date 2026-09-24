import cheetah.api as ch
from cheetah.api import ty

import cheetah.index_tools as ixt
import torch

from flashtransformer.components.mat_ops import WarpGroupMatVecFn
from flashtransformer.components.memory import WGSyncCopyFn
from flashtransformer.utils import SafeDataIndexPtr, SDIPConfig, LocalBarrierManager
from flashtransformer.utils.cuda_utils import init_groups
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
)

# Define test matrix dimensions
D_IN = 512
D_OUT = 32

n_warps = 8
n_threads = n_warps * 32  # 32 threads per warp


def gen_test_kernel():
    """Generate a GEMV (General Matrix-Vector Multiplication) kernel."""
    dims = ixt.Dims()

    # Define thread indices
    thread_idx = dims.new_dim("thread_idx", n_threads)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define matrix dimensions
    d_in_idx = dims.new_dim("d_in_idx", D_IN)
    d_out_idx = dims.new_dim("d_out_idx", D_OUT)

    # Define matrix and vector indices
    weight_idx = dims.new_dim_with_eq("weight_idx", (d_out_idx, d_in_idx))
    in_idx = dims.new_dim_with_eq("in_idx", d_in_idx)
    out_idx = dims.new_dim_with_eq("out_idx", d_out_idx)

    # Define SafeDataIndexPtr configs for global memory
    weight_cfg = SDIPConfig(dims, weight_idx, ty.ptr_const(ty.bf16), "global", "weight")
    in_cfg = SDIPConfig(dims, in_idx, ty.ptr_const(ty.bf16), "global", "in_vec")
    out_cfg = SDIPConfig(dims, out_idx, ty.ptr_mut(ty.f32), "global", "out_vec")

    # Define SafeDataIndexPtr configs for shared memory
    shared_weight_cfg = SDIPConfig(
        dims, weight_idx, ty.ptr_mut(ty.bf16), "shared", "shared_weight"
    )
    shared_in_cfg = SDIPConfig(
        dims, in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_in_vec"
    )
    shared_out_cfg = SDIPConfig(
        dims, out_idx, ty.ptr_mut(ty.f32), "shared", "shared_out_vec"
    )

    @ch.kernel(
        ch.Params(
            weight=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def gemv_kernel(weight, in_vec, out_vec):
        # Create GEMV function
        gemv_fn = WarpGroupMatVecFn(
            dims,
            shared_weight_cfg,
            shared_in_cfg,
            shared_out_cfg,
            warp_idx,
            lane_idx,
            d_in_idx,
            d_out_idx,
        )

        # Create copy functions for input matrices and vectors
        copy_weight = WGSyncCopyFn(
            dims,
            weight_cfg,
            shared_weight_cfg,
            thread_idx,
            weight_idx,
        )

        copy_in = WGSyncCopyFn(
            dims,
            in_cfg,
            shared_in_cfg,
            thread_idx,
            in_idx,
        )

        # Copy output from shared memory back to global memory
        copy_out = WGSyncCopyFn(
            dims,
            shared_out_cfg,
            out_cfg,
            thread_idx,
            out_idx,
        )

        # Create local barrier for synchronization
        local_barrier_manager = LocalBarrierManager()

        # Initialize cheetah indices
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        # Get local barrier
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Create SafeDataIndexPtr instances for global memory
        global_weight_ptr = weight_cfg.wrap_ptr(ix, weight)
        global_in_ptr = in_cfg.wrap_ptr(ix, in_vec)
        global_out_ptr = out_cfg.wrap_ptr(ix, out_vec)

        # Allocate shared memory
        shared_weight_ptr = shared_weight_cfg.alloc_static_shared(ix)
        shared_in_ptr = shared_in_cfg.alloc_static_shared(ix)
        shared_out_ptr = shared_out_cfg.alloc_static_shared(ix)

        groups = init_groups([gemv_fn.get_tile_size()])
        tile = groups[f"tile{gemv_fn.get_tile_size()}"]
        # Synchronize threads before computation
        ch.raw_stmt("__syncthreads();")

        # Copy inputs from global to shared memory
        copy_weight(ix, global_weight_ptr, shared_weight_ptr, local_barrier)
        copy_in(ix, global_in_ptr, shared_in_ptr, local_barrier)

        # Create a dummy tile for the GEMV operation
        # Zero out the output vector before computation
        with ix.loop(out_idx):
            shared_out_ptr.idx_set(0.0)

        # Perform GEMV computation
        gemv_fn(
            ix, shared_weight_ptr, shared_in_ptr, shared_out_ptr, tile, local_barrier
        )

        # Copy result back to global memory
        copy_out(ix, shared_out_ptr, global_out_ptr, local_barrier)

    @ch.fn(
        ch.Params(
            weight=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def gemv_launch(weight, in_vec, out_vec):
        kernel = gemv_kernel.void_ptr()
        arg_exprs = [weight, in_vec, out_vec]
        args = ch.alloc_array(ty.ptr_mut(None), len(arg_exprs))

        for i, arg_expr in enumerate(arg_exprs):
            arg_stack = ch.alloc(arg_expr.type(), arg_expr)
            args[i] = arg_stack.cast(ty.ptr_mut(None))

        ch.raw_stmt(
            "cudaLaunchCooperativeKernel($kernel, 1, $n_threads, $args, 0);",
            kernel=kernel,
            n_threads=ch.const(n_threads, ty.u32),
            args=args,
        )

    @ch.fn(
        ch.Params(
            weight=ty.tensor_const(ty.bf16),
            in_vec=ty.tensor_const(ty.bf16),
            out_vec=ty.tensor_mut(ty.f32),
        )
    )
    def gemv_torch(weight, in_vec, out_vec):
        gemv_launch(
            weight.cast(ty.ptr_const(ty.bf16)),
            in_vec.cast(ty.ptr_const(ty.bf16)),
            out_vec.cast(ty.ptr_mut(ty.f32)),
        )

    return gemv_torch


def gemv_ref(weight, in_vec, out_vec):
    """Reference implementation for matrix-vector multiplication."""
    # Convert to float32 for computation
    # Perform matrix-vector multiplication
    result = torch.matmul(weight, in_vec).to(torch.float32)

    # Copy result to output
    out_vec.copy_(result)

    return out_vec


if __name__ == "__main__":
    # Generate kernel
    print("Codegen started")
    run_test_codegen(
        gen_test_kernel(),
        "gemv_kernel_test",
        headers=standard_headers,
        bind=True,
        dir="tests",
    )
    print("Codegen finished, importing kernel")
    gemv_kernel = import_codegen("gemv_kernel_test", dir="tests")
    print("Kernel compiled and imported, executing")

    # Create test data
    weight = torch.randn(D_OUT, D_IN, dtype=torch.bfloat16)
    in_vec = torch.randn(D_IN, dtype=torch.bfloat16)

    # Test matrix-vector multiplication
    out_ref = torch.zeros(D_OUT, dtype=torch.float32)
    out_kernel = torch.zeros(D_OUT, dtype=torch.float32)

    # Run reference implementation
    gemv_ref(weight, in_vec, out_ref)

    # Run kernel implementation
    gemv_kernel(weight, in_vec, out_kernel)

    # Check results
    print("Matrix-vector multiplication result:")
    print("Reference output:")
    print(out_ref)
    print("Kernel output:")
    print(out_kernel)
    print("Max absolute difference:", torch.max(torch.abs(out_ref - out_kernel)).item())

    # Allow some numerical difference due to different computation methods
    assert torch.allclose(
        out_kernel, out_ref, rtol=1e-2, atol=torch.max(out_ref) * 0.01
    ), "Outputs don't match!"

    print("\nAll tests passed!")
