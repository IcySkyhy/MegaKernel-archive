import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch

from flashtransformer.components.glu.glu_in_fns import WarpGroupGluInFn
from flashtransformer.components.memory import AsyncBarrierGTSCopyFn, WGSyncCopyFn
from flashtransformer.utils import SafeDataIndexPtr, SDIPConfig, LocalBarrierManager
from flashtransformer.utils.cuda_utils import init_groups
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
)
from flashtransformer.utils.barrier_utils import SharedMBarrier

# Define test matrix dimensions

n_warps = 8
n_threads = n_warps * 32  # 32 threads per warp
n_blocks = 128

D_IN = 4096
D_OUT = 8 * n_blocks
D_OUT_PER_BLOCK = D_OUT // n_blocks  # Each block computes a portion of D_OUT
debug = False


def gen_test_kernel():
    """Generate a GLU (Gated Linear Unit) kernel."""
    dims = ixt.Dims()

    # Define block and thread indices
    block_idx = dims.new_dim("block_idx", n_blocks)
    thread_idx = dims.new_dim("thread_idx", n_threads)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define matrix dimensions
    d_in_idx = dims.new_dim("d_in_idx", D_IN)
    d_out_idx_block = dims.new_dim("d_out_idx_block", D_OUT_PER_BLOCK)

    # Define the global output index based on block and local output indices
    d_out_idx = dims.new_dim_with_eq("d_out_idx", (block_idx, d_out_idx_block))

    # Define matrix and vector indices
    global_weight_idx = dims.new_dim_with_eq("weight_in_idx", (d_out_idx, d_in_idx))
    shared_weight_idx = dims.new_dim_with_eq(
        "shared_weight_in_idx", (d_out_idx_block, d_in_idx)
    )
    in_idx = dims.new_dim_with_eq("in_idx", d_in_idx)
    out_idx = dims.new_dim_with_eq("out_idx", d_out_idx)

    # Define block-local output index
    out_idx_block = dims.new_dim_with_eq("out_idx_block", d_out_idx_block)

    # Define SafeDataIndexPtr configs for global memory
    weight_in_cfg = SDIPConfig(
        dims, global_weight_idx, ty.ptr_const(ty.bf16), "global", "weight_in"
    )
    weight_gate_cfg = SDIPConfig(
        dims, global_weight_idx, ty.ptr_const(ty.bf16), "global", "weight_gate"
    )
    in_cfg = SDIPConfig(dims, in_idx, ty.ptr_const(ty.bf16), "global", "in_vec")
    out_cfg = SDIPConfig(dims, out_idx, ty.ptr_mut(ty.f32), "global", "out_vec")

    # Define SafeDataIndexPtr configs for shared memory - using block-local dimensions
    shared_glu_weight_in_cfg = SDIPConfig(
        dims,
        shared_weight_idx,
        ty.ptr_mut(ty.bf16),
        "shared",
        "shared_weight_in",
    )
    shared_glu_weight_gate_cfg = SDIPConfig(
        dims,
        shared_weight_idx,
        ty.ptr_mut(ty.bf16),
        "shared",
        "shared_weight_gate",
    )
    shared_in_cfg = SDIPConfig(
        dims, in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_in_vec"
    )
    shared_out_cfg = SDIPConfig(
        dims, out_idx_block, ty.ptr_mut(ty.f32), "shared", "shared_out_vec"
    )

    @ch.kernel(
        ch.Params(
            weight_in=ty.ptr_const(ty.bf16),
            weight_gate=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def glu_in_kernel(weight_in, weight_gate, in_vec, out_vec):
        # Create GLU function
        glu_fn = WarpGroupGluInFn(
            dims,
            shared_glu_weight_in_cfg,
            shared_in_cfg,
            shared_out_cfg,
            warp_idx,
            lane_idx,
            d_in_idx,
            d_out_idx_block,  # Use the block-local output dimension
        )

        # Create async copy functions for input matrices and vectors
        copy_weight_in = AsyncBarrierGTSCopyFn(
            dims,
            weight_in_cfg,
            shared_glu_weight_in_cfg,
            thread_idx,
            shared_weight_idx,
        )

        copy_weight_gate = AsyncBarrierGTSCopyFn(
            dims,
            weight_gate_cfg,
            shared_glu_weight_gate_cfg,
            thread_idx,
            shared_weight_idx,
        )

        copy_in = AsyncBarrierGTSCopyFn(
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
            out_idx_block,
        )

        # Create local barrier for synchronization
        local_barrier_manager = LocalBarrierManager()

        # Initialize cheetah indices
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        debug_print = make_cond_print(
            ch.and_(ix[block_idx] == 0, ix[warp_idx] == 0, ix[lane_idx] == 0), debug
        )
        debug_print("Starting GLU computation in block", ix[block_idx])

        # Get local barrier (used for sync copy and general sync)
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Allocate and initialize SharedMBarrier for async copies
        barrier_ptr = ch.alloc_shared(ty.u64)
        shared_mbarrier = SharedMBarrier(barrier_ptr)
        with ch.if_(ix[thread_idx] == 0):
            shared_mbarrier.init(n_threads)
        ch.raw_stmt("__syncthreads();")  # Ensure barrier is initialized for all threads

        # Create SafeDataIndexPtr instances for global memory
        global_weight_in_ptr = weight_in_cfg.wrap_ptr(ix, weight_in)
        global_weight_gate_ptr = weight_gate_cfg.wrap_ptr(ix, weight_gate)
        global_in_ptr = in_cfg.wrap_ptr(ix, in_vec)
        global_out_ptr = out_cfg.wrap_ptr(ix, out_vec)

        # Allocate shared memory
        shared_ptr = ch.alloc_extern_shared(ty.bf16)

        # Allocate shared memory for the portion of weights relevant to this block
        shared_weight_in_ptr, shared_ptr = (
            shared_glu_weight_in_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )
        shared_weight_gate_ptr, shared_ptr = (
            shared_glu_weight_gate_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )
        shared_in_ptr, shared_ptr = shared_in_cfg.alloc_dynamic_shared(ix, shared_ptr)
        shared_out_ptr, shared_ptr = shared_out_cfg.alloc_dynamic_shared(ix, shared_ptr)

        groups = init_groups([glu_fn.get_tile_size()])
        tile = groups[f"tile{glu_fn.get_tile_size()}"]

        # Start async copies from global to shared memory using SharedMBarrier
        # Only copy the portion of weights needed for this block
        copy_weight_in(ix, global_weight_in_ptr, shared_weight_in_ptr, shared_mbarrier)
        copy_weight_gate(
            ix, global_weight_gate_ptr, shared_weight_gate_ptr, shared_mbarrier
        )
        copy_in(ix, global_in_ptr, shared_in_ptr, shared_mbarrier)

        # Arrive and wait for all async copies to complete
        token = shared_mbarrier.arrive()
        shared_mbarrier.wait(token)
        debug_print("weight_in[0]: %f", shared_weight_in_ptr.raw_idx(0).cast(ty.f32))
        debug_print(
            "weight_gate[0]: %f", shared_weight_gate_ptr.raw_idx(0).cast(ty.f32)
        )
        debug_print("in[0]: %f", shared_in_ptr.raw_idx(0).cast(ty.f32))

        # Zero out the output vector before computation
        with ix.loop(out_idx_block):
            shared_out_ptr.idx_set(0.0)

        debug_print("Copied inputs to shared memory in block", ix[block_idx])

        # Perform GLU computation on this block's portion
        glu_fn(
            ix,
            shared_weight_in_ptr,
            shared_weight_gate_ptr,
            shared_in_ptr,
            shared_out_ptr,
            tile,
            local_barrier,  # GLU function might need a standard local barrier
        )
        debug_print("shared_out_ptr[0]: %f", shared_out_ptr.raw_idx(0).cast(ty.f32))

        # Copy result back to global memory (synchronous)
        copy_out(ix, shared_out_ptr, global_out_ptr, local_barrier)
        debug_print("out_vec[0]: %f", global_out_ptr.raw_idx(0).cast(ty.f32))
        debug_print("Copied result back to global memory from block", ix[block_idx])

    @ch.fn(
        ch.Params(
            weight_in=ty.ptr_const(ty.bf16),
            weight_gate=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def glu_in_launch(weight_in, weight_gate, in_vec, out_vec):
        # Calculate shared memory size for each block
        # Each block only needs memory for its portion of D_OUT
        shared_mem = (D_IN * D_OUT_PER_BLOCK + D_IN + D_OUT_PER_BLOCK) * 4

        launch_helper(
            glu_in_kernel,
            n_blocks,  # Launch n_blocks blocks
            n_threads,
            [weight_in, weight_gate, in_vec, out_vec],
            shared_mem,
        )

    @ch.fn(
        ch.Params(
            weight_in=ty.tensor_const(ty.bf16),
            weight_gate=ty.tensor_const(ty.bf16),
            in_vec=ty.tensor_const(ty.bf16),
            out_vec=ty.tensor_mut(ty.f32),
        )
    )
    def glu_in_torch(weight_in, weight_gate, in_vec, out_vec):
        glu_in_launch(
            weight_in.cast(ty.ptr_const(ty.bf16)),
            weight_gate.cast(ty.ptr_const(ty.bf16)),
            in_vec.cast(ty.ptr_const(ty.bf16)),
            out_vec.cast(ty.ptr_mut(ty.f32)),
        )

    return glu_in_torch


def glu_in_ref(weight_in, weight_gate, in_vec, out_vec):
    """Reference implementation for GLU (Gated Linear Unit)."""
    # Convert to float32 for computation
    weight_in_f32 = weight_in.to(torch.float32)
    weight_gate_f32 = weight_gate.to(torch.float32)
    in_vec_f32 = in_vec.to(torch.float32)

    # Compute the linear and gate paths
    linear_out = torch.matmul(weight_in_f32, in_vec_f32)
    gate_out = torch.matmul(weight_gate_f32, in_vec_f32)

    # Apply sigmoid to gate path
    gate_sigmoid = torch.nn.functional.silu(gate_out)

    # Multiply linear and gate paths
    result = linear_out * gate_sigmoid

    # Copy result to output
    out_vec.copy_(result)

    return out_vec


if __name__ == "__main__":
    args = launch_args()
    if not args.no_codegen:
        # Generate kernel
        print("Codegen started")
        run_test_codegen(
            gen_test_kernel(),
            "glu_in_kernel_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")
    glu_in_kernel = import_codegen("glu_in_kernel_test", dir="tests", args=args)
    print("Kernel compiled and imported, executing")

    # Create test data
    weight_in = torch.randn(D_OUT, D_IN, dtype=torch.bfloat16)
    weight_gate = torch.randn(D_OUT, D_IN, dtype=torch.bfloat16)
    in_vec = torch.randn(D_IN, dtype=torch.bfloat16)

    # Test GLU operation
    out_ref = torch.zeros(D_OUT, dtype=torch.float32)
    out_kernel = torch.zeros(D_OUT, dtype=torch.float32)

    # Run reference implementation
    glu_in_ref(weight_in, weight_gate, in_vec, out_ref)

    # Run kernel implementation
    glu_in_kernel(weight_in, weight_gate, in_vec, out_kernel)
    torch.cuda.synchronize()

    # Check results
    if args.test:
        print("GLU operation result:")
        print("Reference output:")
        print(out_ref)
        print("Kernel output:")
        print(out_kernel)
        print(
            "Max absolute difference:",
            torch.max(torch.abs(out_ref - out_kernel)).item(),
        )

        # Allow some numerical difference due to different computation methods
        assert torch.allclose(
            out_kernel, out_ref, rtol=1e-2, atol=torch.max(out_ref) * 0.01
        ), "Outputs don't match!"

        print("\nAll tests passed!")

    if args.benchmark:
        print("Benchmarking GLU operation")
        weight_in_bench = torch.randn(args.n_loop, D_OUT, D_IN, dtype=torch.bfloat16)
        weight_gate_bench = torch.randn(args.n_loop, D_OUT, D_IN, dtype=torch.bfloat16)
        in_vec_bench = torch.randn(args.n_loop, D_IN, dtype=torch.bfloat16)
        out_ref_bench = torch.zeros(args.n_loop, D_OUT, dtype=torch.float32)
        out_kernel_bench = torch.zeros(args.n_loop, D_OUT, dtype=torch.float32)

        times_ref = []
        times_kernel = []
        for i in range(args.n_warmup + args.n_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(args.n_loop):
                out = glu_in_ref(
                    weight_in_bench[j],
                    weight_gate_bench[j],
                    in_vec_bench[j],
                    out_ref_bench[j],
                )
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_ref.append(end_time - start_time)

            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(args.n_loop):
                glu_in_kernel(
                    weight_in_bench[j],
                    weight_gate_bench[j],
                    in_vec_bench[j],
                    out_kernel_bench[j],
                )
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_kernel.append(end_time - start_time)

        print(f"GLU operation:")
        print(f"Reference: {sum(times_ref) / len(times_ref) / args.n_loop}")
        print(f"Kernel: {sum(times_kernel) / len(times_kernel) / args.n_loop}")
