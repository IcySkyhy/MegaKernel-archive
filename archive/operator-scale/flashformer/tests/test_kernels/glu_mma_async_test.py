import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch

from flashtransformer.components.glu.glu_in_fns import WarpGroupGluInFn
from flashtransformer.components.mat_ops.mma import MMAMatmulFn
from flashtransformer.components.memory import (
    AsyncBarrierGTSCopyFn,
    AsyncBulkGroupSTGRedAddFn,
    WGSyncCopyFn,
)
from flashtransformer.components.memory.typecast_copy import WGTypeCastCopyFn
from flashtransformer.utils import (
    SafeDataIndexPtr,
    SDIPConfig,
    LocalBarrierManager,
)
from flashtransformer.utils.cuda_utils import init_groups
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
    shmem_helper,
)
from flashtransformer.utils.barrier_utils import SharedMBarrier

# Define test matrix dimensions
n_warps = 8
n_threads = n_warps * 32  # 32 threads per warp
n_blocks = 1  # Reduced from 16 to 1 for debugging

# Input dimensions
D_IN = 4096
# GLU output dimensions
D_GLU_OUT = 8 * n_blocks
D_GLU_OUT_PER_BLOCK = D_GLU_OUT // n_blocks

# MMA projection matrix dimensions
M = 1  # Output dimension size for MMA
K = D_GLU_OUT_PER_BLOCK  # Input dimension size for MMA must match GLU output
N = D_IN  # Number of output features per block

# Final output dimensions
D_OUT = N * n_blocks
D_OUT_PER_BLOCK = D_OUT // n_blocks

debug = True  # Enable debug output


def gen_test_kernel():
    """Generate a GLU kernel with MMA projection and async reduce."""
    dims = ixt.Dims()

    # Define block and thread indices
    block_idx = dims.new_dim("block_idx", n_blocks)
    thread_idx = dims.new_dim("thread_idx", n_threads)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define matrix dimensions for GLU
    d_in_idx = dims.new_dim("d_in_idx", D_IN)
    d_glu_out_idx_block = dims.new_dim("d_glu_out_idx_block", D_GLU_OUT_PER_BLOCK)
    d_glu_out_idx = dims.new_dim_with_eq(
        "d_glu_out_idx", (block_idx, d_glu_out_idx_block)
    )

    # Define matrix dimensions for MMA
    m_idx = dims.new_dim("m_idx", M)
    k_idx = dims.new_dim_with_eq("k_idx", d_glu_out_idx_block)  # Same as GLU output
    n_idx = dims.new_dim("n_idx", N)
    d_out_idx_block = dims.new_dim_with_eq("d_out_idx_block", n_idx)
    d_out_idx = dims.new_dim_with_eq("d_out_idx", (block_idx, d_out_idx_block))

    # Define matrix and vector indices for GLU
    global_weight_in_idx = dims.new_dim_with_eq(
        "weight_in_idx", (d_glu_out_idx, d_in_idx)
    )
    shared_weight_in_idx = dims.new_dim_with_eq(
        "shared_weight_in_idx", (d_glu_out_idx_block, d_in_idx)
    )
    global_weight_gate_idx = dims.new_dim_with_eq(
        "weight_gate_idx", (d_glu_out_idx, d_in_idx)
    )
    shared_weight_gate_idx = dims.new_dim_with_eq(
        "shared_weight_gate_idx", (d_glu_out_idx_block, d_in_idx)
    )
    in_idx = dims.new_dim_with_eq("in_idx", d_in_idx)

    # Matrix indices for MMA projection
    mma_in_idx = dims.new_dim_with_eq("mma_in_idx", (m_idx, k_idx))
    proj_weight_idx = dims.new_dim_with_eq("proj_weight_idx", (n_idx, k_idx))
    mma_out_idx = dims.new_dim_with_eq("mma_out_idx", (m_idx, n_idx))

    # Define SafeDataIndexPtr configs for global memory - GLU
    weight_in_cfg = SDIPConfig(
        dims, global_weight_in_idx, ty.ptr_const(ty.bf16), "global", "weight_in"
    )
    weight_gate_cfg = SDIPConfig(
        dims, global_weight_gate_idx, ty.ptr_const(ty.bf16), "global", "weight_gate"
    )
    in_cfg = SDIPConfig(dims, in_idx, ty.ptr_const(ty.bf16), "global", "in_vec")

    # Define SafeDataIndexPtr configs for global memory - MMA Projection
    proj_weight_cfg = SDIPConfig(
        dims, proj_weight_idx, ty.ptr_const(ty.bf16), "global", "proj_weight"
    )
    out_cfg = SDIPConfig(dims, d_out_idx, ty.ptr_mut(ty.f32), "global", "out_vec")

    # Define SafeDataIndexPtr configs for shared memory - GLU
    shared_glu_weight_in_cfg = SDIPConfig(
        dims, shared_weight_in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_weight_in"
    )
    shared_glu_weight_gate_cfg = SDIPConfig(
        dims,
        shared_weight_gate_idx,
        ty.ptr_mut(ty.bf16),
        "shared",
        "shared_weight_gate",
    )
    shared_in_cfg = SDIPConfig(
        dims, in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_in_vec"
    )

    # Define shared memory configs that will be used by both GLU and MMA
    shared_glu_out_cfg = SDIPConfig(
        dims, d_glu_out_idx_block, ty.ptr_mut(ty.f32), "shared", "shared_glu_out_vec"
    )

    # Define MMA configs - note that MMA requires bf16 inputs
    shared_mma_in_cfg = SDIPConfig(
        dims, mma_in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_mma_in"
    )
    shared_proj_weight_cfg = SDIPConfig(
        dims, proj_weight_idx, ty.ptr_mut(ty.bf16), "shared", "shared_proj_weight"
    )
    shared_mma_out_cfg = SDIPConfig(
        dims, mma_out_idx, ty.ptr_mut(ty.f32), "shared", "shared_mma_out"
    )

    @ch.kernel(
        ch.Params(
            weight_in=ty.ptr_const(ty.bf16),
            weight_gate=ty.ptr_const(ty.bf16),
            proj_weight=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def glu_mma_kernel(weight_in, weight_gate, proj_weight, in_vec, out_vec):
        # Create GLU function
        glu_fn = WarpGroupGluInFn(
            dims,
            shared_glu_weight_in_cfg,
            shared_in_cfg,
            shared_glu_out_cfg,
            warp_idx,
            lane_idx,
            d_in_idx,
            d_glu_out_idx_block,
        )

        # Create MMA function
        mma_fn = MMAMatmulFn(
            dims,
            shared_mma_in_cfg,
            shared_proj_weight_cfg,
            shared_mma_out_cfg,
            warp_idx,
            lane_idx,
            m_idx,
            k_idx,
            n_idx,
            accumulate=False,
            mat1_k_major=True,
            mat2_k_major=True,
        )

        # Create async copy functions for input matrices and vectors
        copy_weight_in = AsyncBarrierGTSCopyFn(
            dims,
            weight_in_cfg,
            shared_glu_weight_in_cfg,
            thread_idx,
            shared_weight_in_idx,
        )

        copy_weight_gate = AsyncBarrierGTSCopyFn(
            dims,
            weight_gate_cfg,
            shared_glu_weight_gate_cfg,
            thread_idx,
            shared_weight_gate_idx,
        )

        copy_in = AsyncBarrierGTSCopyFn(dims, in_cfg, shared_in_cfg, thread_idx, in_idx)

        copy_proj_weight = AsyncBarrierGTSCopyFn(
            dims, proj_weight_cfg, shared_proj_weight_cfg, thread_idx, proj_weight_idx
        )

        # Create typecast copy function to convert GLU output (f32) to MMA input (bf16)
        typecast_copy = WGTypeCastCopyFn(
            dims, shared_glu_out_cfg, shared_mma_in_cfg, thread_idx, mma_in_idx
        )

        # Create async reduce function for final output
        async_red_fn = AsyncBulkGroupSTGRedAddFn(
            dims, shared_mma_out_cfg, out_cfg, thread_idx, d_out_idx_block
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
        debug_print("Starting GLU+MMA computation in block", ix[block_idx])

        # Get local barrier (used for sync copy and general sync)
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Allocate and initialize SharedMBarrier for async copies
        barrier_ptr = ch.alloc_shared(ty.u64)
        shared_mbarrier = SharedMBarrier(barrier_ptr)
        with ch.if_(ix[thread_idx] == 0):
            shared_mbarrier.init(n_threads)
        ch.raw_stmt("__syncthreads();")  # Ensure barrier is initialized for all threads
        
        # Create a more detailed debug printer for runtime values
        runtime_debug = make_cond_print(ch.and_(ix[block_idx] == 0, ix[thread_idx] == 0), debug, local_barrier)

        # Create SafeDataIndexPtr instances for global memory
        global_weight_in_ptr = weight_in_cfg.wrap_ptr(ix, weight_in)
        global_weight_gate_ptr = weight_gate_cfg.wrap_ptr(ix, weight_gate)
        global_proj_weight_ptr = proj_weight_cfg.wrap_ptr(ix, proj_weight)
        global_in_ptr = in_cfg.wrap_ptr(ix, in_vec)
        global_out_ptr = out_cfg.wrap_ptr(ix, out_vec)

        # Allocate shared memory
        shared_ptr = ch.alloc_extern_shared(ty.bf16)

        # Allocate shared memory for GLU inputs
        shared_weight_in_ptr, shared_ptr = (
            shared_glu_weight_in_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )
        shared_weight_gate_ptr, shared_ptr = (
            shared_glu_weight_gate_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )
        shared_in_ptr, shared_ptr = shared_in_cfg.alloc_dynamic_shared(ix, shared_ptr)

        # Allocate shared memory for GLU output / MMA input
        shared_glu_out_ptr, shared_ptr = shared_glu_out_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )

        # Allocate shared memory for projection weights and MMA output
        shared_proj_weight_ptr, shared_ptr = (
            shared_proj_weight_cfg.alloc_dynamic_shared(ix, shared_ptr)
        )
        shared_mma_out_ptr, shared_ptr = shared_mma_out_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )

        # Allocate shared memory for MMA input (bf16 type)
        shared_mma_in_ptr, shared_ptr = shared_mma_in_cfg.alloc_dynamic_shared(
            ix, shared_ptr
        )

        # Create groups for CUDA kernel organization
        glu_tile_size = glu_fn.get_tile_size()
        mma_tile_size = (
            mma_fn.get_tile_size()
            if hasattr(mma_fn, "get_tile_size")
            else glu_tile_size
        )
        tile_sizes = [glu_tile_size, mma_tile_size]
        groups = init_groups(tile_sizes)

        # Get tile for GLU
        glu_tile = groups[f"tile{glu_tile_size}"]

        # Get tile for MMA (or reuse GLU tile if appropriate)
        mma_tile = (
            groups[f"tile{mma_tile_size}"]
            if mma_tile_size != glu_tile_size
            else glu_tile
        )

        # Start async copies from global to shared memory for all inputs
        copy_weight_in(ix, global_weight_in_ptr, shared_weight_in_ptr, shared_mbarrier)
        copy_weight_gate(
            ix, global_weight_gate_ptr, shared_weight_gate_ptr, shared_mbarrier
        )
        copy_in(ix, global_in_ptr, shared_in_ptr, shared_mbarrier)
        copy_proj_weight(
            ix, global_proj_weight_ptr, shared_proj_weight_ptr, shared_mbarrier
        )

        # Arrive and wait for all async copies to complete
        token = shared_mbarrier.arrive()
        shared_mbarrier.wait(token)

        # Zero out the GLU output before computation
        with ix.loop(d_glu_out_idx_block):
            shared_glu_out_ptr.idx_set(0.0)

        # Zero out the MMA output before computation
        with ix.loop(mma_out_idx):
            shared_mma_out_ptr.idx_set(0.0)

        debug_print("Copied inputs to shared memory in block", ix[block_idx])

        # Print input samples
        runtime_debug("Input vector sample: %f", shared_in_ptr.raw_idx(0).cast(ty.f32))
        runtime_debug("Weight_in sample: %f", shared_weight_in_ptr.raw_idx(0).cast(ty.f32))
        runtime_debug("Weight_gate sample: %f", shared_weight_gate_ptr.raw_idx(0).cast(ty.f32))

        # Perform GLU computation
        glu_fn(
            ix,
            shared_weight_in_ptr,
            shared_weight_gate_ptr,
            shared_in_ptr,
            shared_glu_out_ptr,
            glu_tile,
            local_barrier,
        )
        debug_print("GLU computation complete")
        
        # Print GLU output samples
        for i in range(3):
            runtime_debug(f"GLU output sample [{i}]: %f", shared_glu_out_ptr.raw_idx(i))

        # Typecast and copy from GLU output (f32) to MMA input (bf16)
        # Note: GLUInFn already waits on the local barrier, so no extra wait needed
        typecast_copy(ix, shared_glu_out_ptr, shared_mma_in_ptr, local_barrier)
        debug_print("Completed typecast copy from GLU output to MMA input")
        
        # Print MMA input samples after typecast
        for i in range(3):
            runtime_debug(f"MMA input sample [{i}]: %f", shared_mma_in_ptr.raw_idx(i).cast(ty.f32))
        runtime_debug("Proj weight sample: %f", shared_proj_weight_ptr.raw_idx(0).cast(ty.f32))

        # Perform MMA operation on the converted input
        mma_fn(
            ix,
            shared_mma_in_ptr,  # Using GLU output as MMA input
            shared_proj_weight_ptr,
            shared_mma_out_ptr,
            mma_tile,  # Using proper tile from init_groups
            local_barrier,
        )
        debug_print("MMA computation complete")
        
        # Print MMA output samples
        for i in range(3):
            runtime_debug(f"MMA output sample [{i}]: %f", shared_mma_out_ptr.raw_idx(i))

        # Perform async reduction to global memory
        async_red_fn(ix, shared_mma_out_ptr, global_out_ptr)

        # Wait for async reduction to complete
        async_red_fn.wait(ix, local_barrier)
        debug_print("Async reduction complete")
        
        # Print final output samples
        for i in range(3):
            runtime_debug(f"Final output sample [{i}]: %f", global_out_ptr.raw_idx(i))

    @ch.fn(
        ch.Params(
            weight_in=ty.ptr_const(ty.bf16),
            weight_gate=ty.ptr_const(ty.bf16),
            proj_weight=ty.ptr_const(ty.bf16),
            in_vec=ty.ptr_const(ty.bf16),
            out_vec=ty.ptr_mut(ty.f32),
        )
    )
    def glu_mma_launch(weight_in, weight_gate, proj_weight, in_vec, out_vec):
        # Use shmem_helper to calculate shared memory size
        # We need to pass the actual SDIPConfig objects used in the kernel
        shared_mem = shmem_helper([
            shared_glu_weight_in_cfg,
            shared_glu_weight_gate_cfg,
            shared_in_cfg,
            shared_glu_out_cfg,
            shared_proj_weight_cfg,
            shared_mma_out_cfg,
            shared_mma_in_cfg
        ])
        print(f"Shared memory size: {shared_mem} bytes")

        launch_helper(
            glu_mma_kernel,
            n_blocks,
            n_threads,
            [weight_in, weight_gate, proj_weight, in_vec, out_vec],
            shared_mem,
        )

    @ch.fn(
        ch.Params(
            weight_in=ty.tensor_const(ty.bf16),
            weight_gate=ty.tensor_const(ty.bf16),
            proj_weight=ty.tensor_const(ty.bf16),
            in_vec=ty.tensor_const(ty.bf16),
            out_vec=ty.tensor_mut(ty.f32),
        )
    )
    def glu_mma_torch(weight_in, weight_gate, proj_weight, in_vec, out_vec):
        glu_mma_launch(
            weight_in.cast(ty.ptr_const(ty.bf16)),
            weight_gate.cast(ty.ptr_const(ty.bf16)),
            proj_weight.cast(ty.ptr_const(ty.bf16)),
            in_vec.cast(ty.ptr_const(ty.bf16)),
            out_vec.cast(ty.ptr_mut(ty.f32)),
        )

    return glu_mma_torch


def glu_mma_ref(weight_in, weight_gate, proj_weight, in_vec, out_vec):
    """Reference implementation for GLU with MMA projection and reduction."""
    if debug:
        print("\nReference implementation debug:")
        
    # Convert to float32 for computation
    weight_in_f32 = weight_in.to(torch.float32)
    weight_gate_f32 = weight_gate.to(torch.float32)
    proj_weight_f32 = proj_weight.to(torch.float32)
    in_vec_f32 = in_vec.to(torch.float32)
    
    if debug:
        print(f"  Input vector sample: {in_vec_f32[0]}")
        print(f"  Weight_in sample: {weight_in_f32[0][0]}")
        print(f"  Weight_gate sample: {weight_gate_f32[0][0]}")

    # Compute the linear and gate paths for GLU
    linear_out = torch.matmul(weight_in_f32, in_vec_f32)
    gate_out = torch.matmul(weight_gate_f32, in_vec_f32)

    # Apply sigmoid to gate path
    gate_sigmoid = torch.nn.functional.silu(gate_out)

    # Multiply linear and gate paths
    glu_result = linear_out * gate_sigmoid
    
    if debug:
        print(f"  Linear output (first 3): {linear_out[:3]}")
        print(f"  Gate output (first 3): {gate_out[:3]}")
        print(f"  Gate sigmoid (first 3): {gate_sigmoid[:3]}")
        print(f"  GLU result (first 3): {glu_result[:3]}")
    
    if debug:
        print(f"  Proj weight original shape: {proj_weight_f32.shape}")
        print(f"  GLU result shape: {glu_result.shape}")
        print(f"  Proj weight sample: {proj_weight_f32[0, 0]}")
    
    # Perform matrix multiplication with the projection weight
    # Proj weight shape: [D_IN, D_GLU_OUT]
    # GLU result shape: [D_GLU_OUT]
    # Expected output shape: [D_IN]
    result = torch.matmul(proj_weight_f32, glu_result)
    
    if debug:
        print(f"  MMA output (first 3): {result[:3]}")

    # Copy result to output
    out_vec.copy_(result)
    
    if debug:
        print(f"  Final output (first 3): {out_vec[:3]}")

    return out_vec


if __name__ == "__main__":
    args = launch_args()
    if not args.no_codegen:
        # Generate kernel
        print("Codegen started")
        run_test_codegen(
            gen_test_kernel(),
            "glu_mma_async_kernel_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")
    glu_mma_kernel = import_codegen("glu_mma_async_kernel_test", dir="tests", args=args)
    print("Kernel compiled and imported, executing")

    # Create test data
    weight_in = torch.randn(D_GLU_OUT, D_IN, dtype=torch.bfloat16)
    weight_gate = torch.randn(D_GLU_OUT, D_IN, dtype=torch.bfloat16)
    # Changed shape from [M, K, N] to [N, K] to match kernel's expected layout
    proj_weight = torch.randn(D_IN, D_GLU_OUT, dtype=torch.bfloat16)
    in_vec = torch.randn(D_IN, dtype=torch.bfloat16)

    # Test combined GLU and MMA operation
    out_ref = torch.zeros(D_OUT, dtype=torch.float32)
    out_kernel = torch.zeros(D_OUT, dtype=torch.float32)

    # Run reference implementation
    glu_mma_ref(weight_in, weight_gate, proj_weight, in_vec, out_ref)

    # Run kernel implementation
    glu_mma_kernel(weight_in, weight_gate, proj_weight, in_vec, out_kernel)
    torch.cuda.synchronize()

    # Check results
    if args.test:
        print("GLU + MMA + Async Reduce operation result:")
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
        print("Benchmarking GLU + MMA + Async Reduce operation")
        weight_in_bench = torch.randn(
            args.n_loop, D_GLU_OUT, D_IN, dtype=torch.bfloat16
        )
        weight_gate_bench = torch.randn(
            args.n_loop, D_GLU_OUT, D_IN, dtype=torch.bfloat16
        )
        proj_weight_bench = torch.randn(
            args.n_loop, D_IN, D_GLU_OUT, dtype=torch.bfloat16
        )
        in_vec_bench = torch.randn(args.n_loop, D_IN, dtype=torch.bfloat16)
        out_ref_bench = torch.zeros(args.n_loop, D_OUT, dtype=torch.float32)
        out_kernel_bench = torch.zeros(args.n_loop, D_OUT, dtype=torch.float32)

        times_ref = []
        times_kernel = []
        for i in range(args.n_warmup + args.n_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(args.n_loop):
                out = glu_mma_ref(
                    weight_in_bench[j],
                    weight_gate_bench[j],
                    proj_weight_bench[j],
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
                glu_mma_kernel(
                    weight_in_bench[j],
                    weight_gate_bench[j],
                    proj_weight_bench[j],
                    in_vec_bench[j],
                    out_kernel_bench[j],
                )
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_kernel.append(end_time - start_time)

        print(f"GLU + MMA + Async Reduce operation:")
        print(f"Reference: {sum(times_ref) / len(times_ref) / args.n_loop}")
        print(f"Kernel: {sum(times_kernel) / len(times_kernel) / args.n_loop}")
