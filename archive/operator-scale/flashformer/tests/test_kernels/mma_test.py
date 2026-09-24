import cheetah.api as ch
from cheetah.api import ty

import cheetah.index_tools as ixt
import torch

from flashtransformer.components.mat_ops import MMAMatmulFn
from flashtransformer.components.memory import WGSyncCopyFn
from flashtransformer.utils import SafeDataIndexPtr, SDIPConfig, LocalBarrierManager
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
)

# Define test matrix dimensions
M = 8  # Must be <= 8 as per MMAMatmulFn assertion
K = 128
N = 128

n_warps = 4
n_threads = n_warps * 32  # 32 threads per warp

transpose = False


def gen_test_kernel(accumulate=False):
    """Generate a matmul kernel with accumulate as a generation-time parameter."""
    dims = ixt.Dims()

    # Define thread indices
    thread_idx = dims.new_dim("thread_idx", n_threads)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define matrix dimensions
    m_idx = dims.new_dim("m_idx", M)
    k_idx = dims.new_dim("k_idx", K)
    n_idx = dims.new_dim("n_idx", N)

    # Define matrix indices
    mat1_idx = dims.new_dim_with_eq("mat1_idx", (m_idx, k_idx))
    mat2_idx = dims.new_dim_with_eq("mat2_idx", (n_idx, k_idx))
    out_idx = dims.new_dim_with_eq("out_idx", (m_idx, n_idx))

    # Define SafeDataIndexPtr configs - correctly specify dims, index, pointer type, memory type, and name
    mat1_cfg = SDIPConfig(dims, mat1_idx, ty.ptr_const(ty.bf16), "global", "mat1")
    mat2_cfg = SDIPConfig(dims, mat2_idx, ty.ptr_const(ty.bf16), "global", "mat2")
    out_cfg = SDIPConfig(dims, out_idx, ty.ptr_mut(ty.f32), "global", "out")

    shared_mat1_cfg = SDIPConfig(
        dims, mat1_idx, ty.ptr_mut(ty.bf16), "shared", "shared_mat1"
    )
    shared_mat2_cfg = SDIPConfig(
        dims, mat2_idx, ty.ptr_mut(ty.bf16), "shared", "shared_mat2"
    )
    shared_out_cfg = SDIPConfig(
        dims, out_idx, ty.ptr_mut(ty.f32), "shared", "shared_out"
    )

    @ch.kernel(
        ch.Params(
            mat1=ty.ptr_const(ty.bf16),
            mat2=ty.ptr_const(ty.bf16),
            out=ty.ptr_mut(ty.f32),
        )
    )
    def matmul_kernel(mat1, mat2, out):
        # pre Index init: create functions
        mma_fn = MMAMatmulFn(
            dims,
            shared_mat1_cfg,
            shared_mat2_cfg,
            shared_out_cfg,
            warp_idx,
            lane_idx,
            m_idx,
            k_idx,
            n_idx,
            accumulate=accumulate,  # Set at generation time
            mat1_k_major=True,
            mat2_k_major=transpose,
        )

        # Create copy functions for input matrices

        # Copy input matrices to shared memory
        copy_mat1 = WGSyncCopyFn(
            dims,
            mat1_cfg,
            shared_mat1_cfg,
            thread_idx,
            mat1_idx,
        )

        copy_mat2 = WGSyncCopyFn(
            dims,
            mat2_cfg,
            shared_mat2_cfg,
            thread_idx,
            mat2_idx,
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

        # post Index init: get barriers, wrap pointers, and run functions
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Create SafeDataIndexPtr instances using wrap_ptr to avoid duplicating information
        global_mat1_ptr = mat1_cfg.wrap_ptr(ix, mat1)
        global_mat2_ptr = mat2_cfg.wrap_ptr(ix, mat2)
        global_out_ptr = out_cfg.wrap_ptr(ix, out)

        # Allocate shared memory using the alloc_static_shared method
        shared_mat1_ptr = shared_mat1_cfg.alloc_static_shared(ix)
        shared_mat2_ptr = shared_mat2_cfg.alloc_static_shared(ix)
        shared_out_ptr = shared_out_cfg.alloc_static_shared(ix)

        # Create MMA function with accumulate set at generation time

        # Synchronize threads before computation
        ch.raw_stmt("__syncthreads();")

        # Zero out output matrix if not accumulating - handled by MMA implementation internally
        if not accumulate:
            mma_fn.zero_out_accumulator(ix, shared_out_ptr, local_barrier)

        # Copy input matrices from global to shared memory
        copy_mat1(ix, global_mat1_ptr, shared_mat1_ptr, local_barrier)
        copy_mat2(ix, global_mat2_ptr, shared_mat2_ptr, local_barrier)

        # Create a dummy tile for the matmul operation
        tile = ch.alloc(ty.ptr_mut(ty.i32), name="tile")

        # Perform matrix multiplication
        mma_fn(
            ix, shared_mat1_ptr, shared_mat2_ptr, shared_out_ptr, tile, local_barrier
        )

        # Copy result back to global memory
        copy_out(ix, shared_out_ptr, global_out_ptr, local_barrier)

    @ch.fn(
        ch.Params(
            mat1=ty.ptr_const(ty.bf16),
            mat2=ty.ptr_const(ty.bf16),
            out=ty.ptr_mut(ty.f32),
        )
    )
    def matmul_launch(mat1, mat2, out):
        kernel = matmul_kernel.void_ptr()
        arg_exprs = [mat1, mat2, out]
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
            mat1=ty.tensor_const(ty.bf16),
            mat2=ty.tensor_const(ty.bf16),
            out=ty.tensor_mut(ty.f32),
        )
    )
    def matmul_torch(mat1, mat2, out):
        matmul_launch(
            mat1.cast(ty.ptr_const(ty.bf16)),
            mat2.cast(ty.ptr_const(ty.bf16)),
            out.cast(ty.ptr_mut(ty.f32)),
        )

    return matmul_torch


def matmul_ref(mat1, mat2, out):
    """Reference implementation for matrix multiplication."""
    # Convert to float32 for computation
    mat1_f32 = mat1.to(torch.float32)
    mat2_f32 = mat2.to(torch.float32)
    if transpose:
        mat2_f32 = mat2_f32.transpose(0, 1)

    # Perform matrix multiplication
    result = torch.matmul(mat1_f32, mat2_f32)

    # Copy result to output
    out.copy_(result)

    return out


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Codegen started")
        run_test_codegen(
            gen_test_kernel(accumulate=False),
            "mma_kernel_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Codegen disabled, importing kernel")

    mma_kernel = import_codegen("mma_kernel_test", dir="tests", args=args)
    print("Kernel compiled and imported")

    torch_dtype = torch.bfloat16
    torch_accum_dtype = torch.float32

    # Create test data
    mat1 = args.init_fn((M, K), dtype=torch_dtype)
    mat2 = args.init_fn((N, K), dtype=torch_dtype)
    # mat1 = torch.eye(M, K, dtype=torch_dtype)
    # mat2 = torch.arange(N * K, dtype=torch_dtype).reshape(K, N)

    # Outputs
    out_ref = torch.zeros(M, N, dtype=torch_accum_dtype)
    out_kernel = torch.zeros(M, N, dtype=torch_accum_dtype)

    if args.test:
        print("Running MMA test...")
        # Run reference implementation
        matmul_ref(mat1, mat2, out_ref)

        # Run kernel implementation
        mma_kernel(mat1, mat2, out_kernel)

        print("out_ref:")
        print(out_ref)
        print("out_kernel:")
        print(out_kernel)
        # Check results
        print(
            "Max absolute difference:",
            torch.max(torch.abs(out_ref - out_kernel)).item(),
        )
        if not torch.allclose(out_kernel, out_ref, rtol=1e-2, atol=1e-2):
            print("Outputs don't match!")
            with open("mma_debug.txt", "w") as f:
                for m in range(M):
                    for n in range(N):
                        f.write(
                            f"m{m}, n{n}: {out_ref[m, n]:.6f} {out_kernel[m, n]:.6f}\n"
                        )
        else:
            print("TEST PASSED")

    if args.benchmark:
        import time

        print("Benchmarking MMA kernel...")
        # Warmup
        for _ in range(args.n_warmup):
            mma_kernel(mat1, mat2, out_kernel)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(args.n_runs):
            mma_kernel(mat1, mat2, out_kernel)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        avg_ms = (end - start) * 1000.0 / max(1, args.n_runs)
        print(f"Kernel avg time: {avg_ms:.3f} ms")

        # Reference timing
        start = time.time()
        for _ in range(args.n_runs):
            matmul_ref(mat1, mat2, out_ref)
        end = time.time()
        ref_ms = (end - start) * 1000.0 / max(1, args.n_runs)
        print(f"Reference avg time: {ref_ms:.3f} ms")
        if avg_ms > 0:
            print(f"Speedup: {ref_ms / avg_ms:.2f}x")
