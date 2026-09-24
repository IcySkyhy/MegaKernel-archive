import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch

from flashtransformer.components.attention.attention_out import AttentionOutFn
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.utils.memory_pipeline import MemoryPipeline
from flashtransformer.utils.cuda_utils import init_groups, check_tile_size
from flashtransformer.utils.barrier_utils import LocalBarrierManager
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
)

# Test configuration
n_blocks = 128
n_warps = 8
n_threads = n_warps * 32
pipeline_stages = 3

# Model dimensions
d_in = 4096
d_head = 128
n_heads = 32
n_kv_heads = 8
query_per_kv = n_heads // n_kv_heads
hidden_dim = 4096  # Increased hidden dimension size
debug = True


def gen_attn_out_test_kernel():
    """Generate an attention output test kernel using MemoryPipeline."""
    dims = ixt.Dims()

    # Thread and block dimensions
    block_idx = dims.new_dim("block_idx", n_blocks)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    thread_idx = dims.new_dim_with_eq("thread_idx", (warp_idx, lane_idx))

    # Consumer and producer dimensions (split warps)
    consumer_warp_idx = dims.new_dim("consumer_warp_idx", n_warps // 2)
    producer_warp_idx = dims.new_dim("producer_warp_idx", n_warps // 2)

    inter_buffer_idx = dims.new_dim("inter_buffer_idx", pipeline_stages)
    intra_buffer_idx = dims.new_dim("intra_buffer_idx", 8192)  # Reduced buffer size

    # Model dimensions
    d_head_idx = dims.new_dim("d_head_idx", d_head)
    head_idx = dims.new_dim("head_idx", n_heads)
    d_in_idx = dims.new_dim_with_eq("d_in_idx", (head_idx, d_head_idx))
    d_out_idx = dims.new_dim("d_out_idx", hidden_dim)

    # Create SDIPConfig instances
    in_cfg = SDIPConfig(dims, d_in_idx, ty.ptr_mut(ty.bf16), "global", "attn_output")
    out_cfg = SDIPConfig(dims, d_out_idx, ty.ptr_mut(ty.bf16), "global", "final_output")
    w_o_cfg = SDIPConfig(
        dims,
        (d_out_idx, head_idx, d_head_idx),
        ty.ptr_mut(ty.bf16),
        "global",
        "out_weights",
    )

    # Create Memory Pipeline for producer-consumer pattern
    local_barrier_manager = LocalBarrierManager()
    pipeline = MemoryPipeline(
        dims,
        pipeline_stages,
        warp_idx,
        lane_idx,
        consumer_warp_idx,
        producer_warp_idx,
        local_barrier_manager,
        8192,
        ty.bf16,
        block_idx=block_idx,
    )

    # Create attention output function
    attn_out_fn = AttentionOutFn(
        dims=dims,
        in_cfg=in_cfg,
        out_cfg=out_cfg,
        w_o_cfg=w_o_cfg,
        mem_pipeline=pipeline,
        block_idx=block_idx,
        head_idx=head_idx,
        d_in_idx=d_in_idx,
        d_out_idx=d_out_idx,
        quantized=False,
        d_head_qcfg=None,
    )

    @ch.kernel(
        ch.Params(
            attn_output=ty.ptr_mut(ty.bf16),
            out_weights=ty.ptr_mut(ty.bf16),
            final_output=ty.ptr_mut(ty.bf16),
        )
    )
    def attn_out_kernel(
        attn_output,
        out_weights,
        final_output,
    ):
        ix = dims.init()
        debug_print = make_cond_print(
            ch.and_(ch.block_idx_x() == 0, ch.thread_idx_x() == 0), debug
        )
        debug_print("Starting Attention Output computation")

        # Initialize barriers for synchronization
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Initialize cooperative groups
        groups = init_groups([32, 64])
        shared_ptr = ch.alloc_extern_shared(ty.bf16)

        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Create SafeDataIndexPtr instances
        attn_output_ptr = in_cfg.wrap_ptr(ix, attn_output)
        out_weights_ptr = w_o_cfg.wrap_ptr(ix, out_weights)
        final_output_ptr = out_cfg.wrap_ptr(ix, final_output)

        # Initialize pipeline and attention output function memory
        debug_print("Initializing shared memory")
        pipeline_buffer, next_shared_ptr = pipeline.initialize(ix, shared_ptr)
        next_shared_ptr = attn_out_fn.setup_shmem(ix, next_shared_ptr)

        # Synchronize threads before execution
        ch.raw_stmt("__syncthreads();")

        # Split workload between consumer and producer warps
        with ch.if_else(ix[warp_idx] < n_warps // 2) as branch:
            with branch.then():
                # Consumer warps
                ix.set_index(consumer_warp_idx, ix[warp_idx])
                debug_print("Starting consumer warps")

                # Initialize pipeline for consumers
                pipeline.consumer_pipeline_ready(ix)

                # Execute consumer computation
                attn_out_fn.consume(
                    ix,
                    attn_output_ptr,
                    final_output_ptr,
                    groups,
                )

            with branch.else_():
                # Producer warps
                ix.set_index(
                    producer_warp_idx, ix[warp_idx] - ix.size(consumer_warp_idx)
                )
                debug_print("Starting producer warps")

                # Execute producer tasks
                attn_out_fn.produce(ix, out_weights_ptr)

        debug_print("Attention Output computation completed")

    @ch.fn(
        ch.Params(
            attn_output=ty.ptr_mut(ty.bf16),
            out_weights=ty.ptr_mut(ty.bf16),
            final_output=ty.ptr_mut(ty.bf16),
        )
    )
    def attn_out_launch(
        attn_output,
        out_weights,
        final_output,
    ):
        # Calculate shared memory size from pipeline and attention output component sizes
        pipeline_size = pipeline.get_shared_size()
        attn_out_size = attn_out_fn.get_shmem_size()

        # Add barrier size (8 bytes per barrier in pipeline stages)
        barrier_size = pipeline_stages * 8

        # Total shared memory size
        shared_mem_size = pipeline_size + attn_out_size + barrier_size

        launch_helper(
            attn_out_kernel,
            n_blocks,
            n_threads,
            [
                attn_output,
                out_weights,
                final_output,
            ],
            shared_mem_size,
        )

    @ch.fn(
        ch.Params(
            attn_output=ty.tensor_mut(ty.bf16),
            out_weights=ty.tensor_mut(ty.bf16),
            final_output=ty.tensor_mut(ty.bf16),
        )
    )
    def attn_out_torch(
        attn_output,
        out_weights,
        final_output,
    ):
        attn_out_launch(
            attn_output.cast(ty.ptr_mut(ty.bf16)),
            out_weights.cast(ty.ptr_mut(ty.bf16)),
            final_output.cast(ty.ptr_mut(ty.bf16)),
        )

    return attn_out_torch


def attn_out_ref(attn_output, out_weights, final_output):
    """Reference implementation for attention output computation using PyTorch."""
    # Reshape inputs for matmul
    attn_output_reshaped = attn_output.view(n_heads * d_head)  # [n_heads, d_head]

    # Reshape weights for matmul
    out_weights_reshaped = out_weights.view(hidden_dim, n_heads * d_head)

    final_output_reshaped = torch.matmul(out_weights_reshaped, attn_output_reshaped)
    final_output.copy_(final_output_reshaped.view(hidden_dim))

    print(f"Reference implementation completed:")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Output weights shape: {out_weights.shape}")
    print(f"Final output shape: {final_output.shape}")


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Generating attention output kernel code")
        kernel = gen_attn_out_test_kernel()
        run_test_codegen(
            kernel,
            "attn_out_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    # Import kernel
    attn_out_kernel = import_codegen("attn_out_test", dir="tests", args=args)
    print("Kernel imported, preparing for execution")

    # Create input and output tensors
    if args.init == "ones":
        attn_output = torch.ones((n_heads, d_head), dtype=torch.bfloat16, device="cuda")
        out_weights = torch.ones(
            (n_heads, hidden_dim, d_head), dtype=torch.bfloat16, device="cuda"
        )
        print("Initialized tensors with ones")
    else:
        # Random initialization
        attn_output = torch.randn(
            (n_heads, d_head), dtype=torch.bfloat16, device="cuda"
        )
        out_weights = torch.randn(
            (n_heads, hidden_dim, d_head), dtype=torch.bfloat16, device="cuda"
        )
        print("Initialized tensors with random values")

    # Output tensor
    final_output = torch.zeros(hidden_dim, dtype=torch.bfloat16, device="cuda")

    # Clone for reference implementation
    attn_output_ref = attn_output.clone()
    out_weights_ref = out_weights.clone()
    final_output_ref = torch.zeros_like(final_output)

    # Run tests
    if args.test:
        print("Running attention output compute test...")

        # Run reference implementation
        attn_out_ref(
            attn_output_ref,
            out_weights_ref,
            final_output_ref,
        )

        # Run kernel implementation
        attn_out_kernel(
            attn_output,
            out_weights,
            final_output,
        )
        torch.cuda.synchronize()

        # Check results with tolerance
        rtol = 0.1  # 10% relative tolerance for bfloat16
        atol = 1  # Absolute tolerance

        output_match = torch.allclose(
            final_output_ref.float(), final_output.float(), rtol=rtol, atol=atol
        )

        print(f"Output match: {output_match}")

        # Overall test result
        if output_match:
            print("TEST PASSED: Results match reference with acceptable tolerance!")
        else:
            print("Output mismatch:")
            print(
                "Reference output (first 10 elements):", final_output_ref[:10].float()
            )
            print("Kernel output (first 10 elements):", final_output[:10].float())
            print("TEST FAILED: Results don't match reference!")

    # Run benchmark
    if args.benchmark:
        print("Benchmarking attention output compute kernel")

        # Prepare benchmark parameters
        iterations = args.n_runs
        warmup = args.n_warmup

        # Warmup
        for _ in range(warmup):
            attn_out_kernel(
                attn_output,
                out_weights,
                final_output,
            )
            # Slight variation for each run
            attn_output.add_(torch.randn_like(attn_output) * 0.01)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            attn_output.add_(torch.randn_like(attn_output) * 0.01)
            attn_out_kernel(
                attn_output,
                out_weights,
                final_output,
            )

        torch.cuda.synchronize()
        end = time.time()

        # Calculate and report performance
        avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average kernel execution time: {avg_time_ms:.3f} ms")

        # Run reference for comparison if needed
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            attn_output_ref.add_(torch.randn_like(attn_output_ref) * 0.01)
            attn_out_ref(
                attn_output_ref,
                out_weights_ref,
                final_output_ref,
            )

        torch.cuda.synchronize()
        end = time.time()

        ref_avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average reference time: {ref_avg_time_ms:.3f} ms")

        if ref_avg_time_ms > 0:
            speedup = ref_avg_time_ms / avg_time_ms
            print(f"Speedup: {speedup:.2f}x")
