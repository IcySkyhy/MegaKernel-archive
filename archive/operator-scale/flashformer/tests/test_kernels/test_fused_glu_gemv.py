#!/usr/bin/env python3

import torch
import cheetah.api as ch
from cheetah.api import ty
import cheetah.index_tools as ixt

from flashtransformer.components.glu.fused_v_glu import FusedGLUGEMV
from flashtransformer.utils import (
    SDIPConfig,
    MemoryPipeline,
    get_shared_ptr,
    init_groups,
    LocalBarrierManager,
)
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    launch_args,
    launch_helper,
    standard_headers,
)

# Test configuration
D_MODEL = 512
D_HIDDEN = 2048
N_BLOCKS = 4
N_CONSUMER_WARPS = 4
N_PRODUCER_WARPS = 4
N_WARPS = N_CONSUMER_WARPS + N_PRODUCER_WARPS
N_THREADS = N_WARPS * 32
PIPELINE_STAGES = 2
CHUNK_SIZE = 16384  # Size of data chunk per pipeline stage


def gen_test_kernel():
    """Generate test kernel for FusedGLUGEMV."""

    # Create dimensions
    dims = ixt.Dims()

    # Basic dimensions
    block_idx = dims.new_dim("block_idx", N_BLOCKS)
    d_in_idx = dims.new_dim("d_in_idx", D_HIDDEN)  # GLU hidden dimension
    d_out_idx = dims.new_dim("d_out_idx", D_MODEL)  # Output dimension

    # Thread dimensions
    thread_idx = dims.new_dim("thread_idx", N_THREADS)
    warp_idx = dims.new_dim("warp_idx", N_WARPS)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Consumer and producer warp indices
    consumer_warp_idx = dims.new_dim("consumer_warp_idx", N_CONSUMER_WARPS)
    producer_warp_idx = dims.new_dim("producer_warp_idx", N_PRODUCER_WARPS)

    # Create SDIPConfigs with correct pointer types
    in_cfg = SDIPConfig(
        dims, d_in_idx, ty.ptr_const(ty.bf16), "global", "glu_activation"
    )
    out_cfg = SDIPConfig(dims, d_out_idx, ty.ptr_mut(ty.bf16), "global", "output")
    w_out_cfg = SDIPConfig(
        dims,
        dims.new_dim_with_eq("w_out_idx", (d_out_idx, d_in_idx)),
        ty.ptr_const(ty.bf16),
        "global",
        "w_out",
    )

    # Create memory pipeline and GLU function outside kernel for shmem calculation
    local_barrier_manager = LocalBarrierManager()
    pipeline = MemoryPipeline(
        dims,
        pipeline_stages=PIPELINE_STAGES,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        consumer_warp_idx=consumer_warp_idx,
        producer_warp_idx=producer_warp_idx,
        local_barrier_manager=local_barrier_manager,
        chunk_size=CHUNK_SIZE,
        ptr_type=ty.bf16,
        block_idx=block_idx,
    )

    # Create FusedGLUGEMV function
    fused_glu_fn = FusedGLUGEMV(
        dims=dims,
        in_cfg=in_cfg,
        out_cfg=out_cfg,
        w_out_cfg=w_out_cfg,
        mem_pipeline=pipeline,
        block_idx=block_idx,
        d_in_idx=d_in_idx,
        d_out_idx=d_out_idx,
    )

    @ch.kernel(
        ch.Params(
            glu_activation=ty.ptr_const(ty.bf16),
            output=ty.ptr_mut(ty.bf16),
            w_out=ty.ptr_const(ty.bf16),
        )
    )
    def fused_glu_gemv_kernel(glu_activation, output, w_out):

        # Initialize dimensions
        ix = dims.init()

        # Set indices
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Get shared memory pointer
        shared_ptr = ch.alloc_extern_shared(ty.bf16)

        # Initialize pipeline
        pipeline_buffer, shared_ptr = pipeline.initialize(ix, shared_ptr)

        # Setup shared memory for GLU function
        shared_ptr = fused_glu_fn.setup_shmem(ix, shared_ptr)

        # Sync before starting
        ch.raw_stmt("__syncthreads();")

        # Wrap pointers
        glu_activation_ptr = in_cfg.wrap_ptr(ix, glu_activation)
        output_ptr = out_cfg.wrap_ptr(ix, output)
        w_out_ptr = w_out_cfg.wrap_ptr(ix, w_out)

        # Initialize groups
        groups = init_groups([fused_glu_fn.glu_out_gemv_fn.get_tile_size()])

        # Split workload between consumer and producer warps
        with ch.if_else(ix[warp_idx] < N_CONSUMER_WARPS) as branch:
            with branch.then():
                # Consumer warps
                ix.set_index(consumer_warp_idx, ix[warp_idx])

                # Initialize pipeline for consumers
                pipeline.consumer_pipeline_ready(ix)

                # Execute consumer computation
                fused_glu_fn.consume(
                    ix,
                    glu_activation_ptr,
                    output_ptr,
                    groups,
                )

            with branch.else_():
                # Producer warps
                ix.set_index(
                    producer_warp_idx, ix[warp_idx] - ix.size(consumer_warp_idx)
                )

                # Execute producer computation
                fused_glu_fn.produce(
                    ix,
                    w_out_ptr,
                )

    @ch.fn(
        ch.Params(
            glu_activation=ty.ptr_const(ty.bf16),
            output=ty.ptr_mut(ty.bf16),
            w_out=ty.ptr_const(ty.bf16),
        )
    )
    def launch(glu_activation, output, w_out):
        # Calculate shared memory size using the get_shared_size methods
        shmem_size = pipeline.get_shared_size() + fused_glu_fn.get_shmem_size()

        # Launch kernel
        launch_helper(
            fused_glu_gemv_kernel,
            N_BLOCKS,
            N_THREADS,
            [glu_activation, output, w_out],
            shmem_size,
        )

    @ch.fn(
        ch.Params(
            glu_activation=ty.tensor_const(ty.bf16),
            output=ty.tensor_mut(ty.bf16),
            w_out=ty.tensor_const(ty.bf16),
        )
    )
    def launch_torch(glu_activation, output, w_out):
        launch(
            glu_activation.cast(ty.ptr_const(ty.bf16)),
            output.cast(ty.ptr_mut(ty.bf16)),
            w_out.cast(ty.ptr_const(ty.bf16)),
        )

    return launch_torch


def reference_fused_glu_gemv(glu_activation, w_out, output):
    """Reference implementation: output += w_out @ glu_activation"""
    # Convert to float32 for computation
    glu_act_f32 = glu_activation.to(torch.float32)
    w_out_f32 = w_out.to(torch.float32)
    output_f32 = output.to(torch.float32)

    # Compute GEMV and add to output
    result = output_f32 + torch.matmul(w_out_f32, glu_act_f32)

    # Convert back to bf16
    return result.to(torch.bfloat16)


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Generating FusedGLUGEMV kernel...")
        run_test_codegen(
            gen_test_kernel(),
            "fused_glu_gemv_kernel",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    fused_glu_kernel = import_codegen("fused_glu_gemv_kernel", dir="tests", args=args)
    print("Kernel compiled and imported")

    # Create test data
    torch.manual_seed(42)
    glu_activation = torch.randn(D_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    w_out = torch.randn(D_MODEL, D_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    output_init = torch.randn(D_MODEL, dtype=torch.bfloat16, device="cuda") * 0.1

    if args.test:
        print("\nRunning correctness test...")

        # Reference computation
        ref_output = reference_fused_glu_gemv(
            glu_activation, w_out, output_init.clone()
        )

        # Kernel computation
        kernel_output = output_init.clone()
        fused_glu_kernel(glu_activation, kernel_output, w_out)
        torch.cuda.synchronize()

        # Calculate errors
        abs_diff = torch.abs(kernel_output - ref_output)
        max_abs_diff = torch.max(abs_diff).item()
        rel_error = abs_diff / (torch.abs(ref_output) + 1e-10)
        max_rel_error = torch.max(rel_error).item()

        print(f"\nResults:")
        print(f"  Max absolute error: {max_abs_diff:.6e}")
        print(f"  Max relative error: {max_rel_error*100:.4f}%")

        # Test with torch.allclose (rtol=1% or atol=1e-4)
        passed = torch.allclose(kernel_output, ref_output, rtol=0.01, atol=1e-4)

        if passed:
            print(
                "\n✓ Test PASSED - Output within tolerance (1% relative OR 1e-4 absolute)"
            )

            # Show sample values
            print(f"\nSample values (first 5 elements):")
            for i in range(min(5, D_MODEL)):
                print(
                    f"  [{i}] Ref: {ref_output[i]:.6f}, Kernel: {kernel_output[i]:.6f}, Diff: {abs_diff[i]:.6e}"
                )
        else:
            print("\n✗ Test FAILED - Output exceeds tolerance")

            # Show where the largest error occurs
            max_idx = torch.argmax(abs_diff).item()
            print(f"\nLargest error at index {max_idx}:")
            print(f"  Reference: {ref_output[max_idx]:.6f}")
            print(f"  Kernel:    {kernel_output[max_idx]:.6f}")
            print(f"  Diff:      {abs_diff[max_idx]:.6e}")

            assert False, "Test failed!"

        print("\nAll tests passed!")

    if args.benchmark:
        print("\nBenchmarking FusedGLUGEMV...")
        # Benchmark implementation would go here
        print("Benchmark not yet implemented")
