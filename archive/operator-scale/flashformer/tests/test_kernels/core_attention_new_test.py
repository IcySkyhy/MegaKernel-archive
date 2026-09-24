import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch
import math
import numpy as np

from flashtransformer.components.attention.core_attention_new import (
    CoreAttentionScopeFn,
)
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.utils.memory_pipeline import MemoryPipeline
from flashtransformer.utils.cuda_utils import init_groups, check_tile_size
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

# Test configuration
n_kv_heads = 6
query_per_kv = 1  # Queries per KV head
per_kv_block = 1  # Number of blocks per KV head
n_blocks = n_kv_heads * per_kv_block
n_warps = 12
n_threads = n_warps * 32
n_heads = n_kv_heads * query_per_kv
d_head = 128
seq_len = 256
pipeline_stages = 3
debug = False


def gen_test_kernel(dtype=ty.bf16, accum_dtype=ty.f32, chunk_size: int = 16384):
    """Generate test kernel for core attention new."""

    # Create dimensions
    dims = ixt.Dims()

    # Thread and block dimensions
    block_idx = dims.new_dim("block_idx", n_blocks)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    thread_idx = dims.new_dim_with_eq("thread_idx", (warp_idx, lane_idx))

    # Consumer and producer dimensions (split warps)
    consumer_warp_idx = dims.new_dim("consumer_warp_idx", 8)
    producer_warp_idx = dims.new_dim("producer_warp_idx", 4)

    # Model dimensions
    kv_head_idx = dims.new_dim("kv_head_idx", n_kv_heads)
    per_kv_block_idx = dims.new_dim("per_kv_block_idx", per_kv_block)
    query_per_kv_head_idx = dims.new_dim("query_per_kv_head_idx", query_per_kv)
    head_idx = dims.new_dim("head_idx", n_heads)  # n_heads = n_kv_heads * query_per_kv
    d_head_idx = dims.new_dim("d_head_idx", d_head)
    seq_len_idx = dims.new_dim("seq_len_idx", seq_len)

    # weight_scope = dims.new_scope("weight_scope")
    # Global buffer indices
    global_qo_idx = dims.new_dim_with_eq("global_qo_idx", (head_idx, d_head_idx))
    global_max_denom_idx = dims.new_dim_with_eq(
        "global_max_denom_idx",
        (kv_head_idx, per_kv_block_idx, query_per_kv_head_idx, 2),
    )
    global_attn_out_idx = dims.new_dim_with_eq(
        "global_attn_out_idx", (head_idx, d_head_idx)
    )
    kv_cache_idx = dims.new_dim_with_eq(
        "kv_cache_idx", (kv_head_idx, seq_len_idx, d_head_idx)
    )

    # Create SDIPConfigs
    global_qo_cfg = SDIPConfig(
        dims, global_qo_idx, ty.ptr_mut(dtype), "global", "global_qo"
    )
    global_max_denom_cfg = SDIPConfig(
        dims,
        global_max_denom_idx,
        ty.ptr_mut(accum_dtype),
        "global",
        "global_max_denom",
    )
    global_attn_out_cfg = SDIPConfig(
        dims, global_attn_out_idx, ty.ptr_mut(dtype), "global", "global_attn_out"
    )
    kv_cache_cfg = SDIPConfig(
        dims, kv_cache_idx, ty.ptr_mut(dtype), "global", "kv_cache"
    )

    # Create barrier managers
    local_barrier_manager = LocalBarrierManager()

    # Create Memory Pipeline (updated signature)
    pipeline = MemoryPipeline(
        dims,
        pipeline_stages,
        warp_idx,
        lane_idx,
        consumer_warp_idx,
        producer_warp_idx,
        local_barrier_manager,
        chunk_size,  # chunk_size
        dtype,  # ptr_type
        block_idx=block_idx,
    )

    # Create barrier managers
    global_barrier_idx = dims.new_dim("global_barrier_idx", 1)
    global_barrier_manager = GlobalBarrierManager(
        dims, global_barrier_idx, block_idx, thread_idx
    )

    # Create core attention function
    core_attn_fn = CoreAttentionScopeFn(
        dims=dims,
        global_qo_cfg=global_qo_cfg,
        global_max_denom_cfg=global_max_denom_cfg,
        global_attn_out_cfg=global_attn_out_cfg,
        kv_cache_cfg=kv_cache_cfg,
        mem_pipeline=pipeline,
        head_idx=head_idx,
        kv_head_idx=kv_head_idx,
        d_head_idx=d_head_idx,
        seq_len_idx=seq_len_idx,
        dynamic=False,
    )

    @ch.kernel(
        ch.Params(
            query_input=ty.ptr_mut(dtype),
            k_cache=ty.ptr_mut(dtype),
            v_cache=ty.ptr_mut(dtype),
            attn_output=ty.ptr_mut(dtype),
            global_barrier_buffer=ty.ptr_mut(ty.i32),
            global_max_denom_buffer=ty.ptr_mut(accum_dtype),
            seq_len=ty.i32,
        )
    )
    def core_attn_kernel(
        query_input,
        k_cache,
        v_cache,
        attn_output,
        global_barrier_buffer,
        global_max_denom_buffer,
        seq_len,
    ):
        # Initialize indices
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Initialize shared memory
        shared_ptr = safe_extern_shared(ix, dtype, thread_idx)

        # Initialize barriers
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)
        global_barrier_manager.set_buffer(ix, global_barrier_buffer)

        # Initialize groups
        groups = init_groups([32, 64])

        debug_print = make_cond_print(
            ch.and_(ix[block_idx] == 0, ix[thread_idx] == 0),
            debug,
            local_barrier,
        )

        # Wrap input pointers
        query_ptr = global_qo_cfg.wrap_ptr(ix, query_input)
        k_cache_ptr = kv_cache_cfg.wrap_ptr(ix, k_cache)
        v_cache_ptr = kv_cache_cfg.wrap_ptr(ix, v_cache)
        attn_output_ptr = global_attn_out_cfg.wrap_ptr(ix, attn_output)

        # Initialize pipeline and core attention shared memory
        pipeline_buffer, next_shared_ptr = pipeline.initialize(ix, shared_ptr)
        next_shared_ptr = core_attn_fn.setup_shmem(ix, next_shared_ptr)

        global_max_denom_ptr = global_max_denom_cfg.wrap_ptr(
            ix, global_max_denom_buffer.cast(ty.ptr_mut(accum_dtype))
        )

        # Synchronize before execution
        ch.raw_stmt("__syncthreads();")

        debug_print("Starting core attention computation")

        pipeline.pipeline_ready(ix)

        # Use pc_generate to handle producer/consumer split
        core_attn_fn.pc_generate(
            ix,
            warp_idx,
            query_ptr,
            k_cache_ptr,
            v_cache_ptr,
            global_max_denom_ptr,
            attn_output_ptr,
            global_barrier_manager,
            seq_len,
            groups,
            # reg_split=(224, 64),
        )

        debug_print("Core attention computation completed")

    @ch.fn(
        ch.Params(
            query_input=ty.ptr_mut(dtype),
            k_cache=ty.ptr_mut(dtype),
            v_cache=ty.ptr_mut(dtype),
            attn_output=ty.ptr_mut(dtype),
            global_barrier_buffer=ty.ptr_mut(ty.i32),
            global_max_denom_buffer=ty.ptr_mut(accum_dtype),
            seq_len=ty.i32,
        )
    )
    def core_attn_launch(
        query_input,
        k_cache,
        v_cache,
        attn_output,
        global_barrier_buffer,
        global_max_denom_buffer,
        seq_len,
    ):
        # Calculate shared memory size
        pipeline_size = pipeline.get_shared_size()
        core_attn_size = core_attn_fn.get_shmem_size()
        barrier_size = pipeline_stages * 8
        print(f"Pipeline size: {pipeline_size}")
        print(f"Core attention size: {core_attn_size}")
        print(f"Barrier size: {barrier_size}")

        shared_mem_size = pipeline_size + core_attn_size + barrier_size

        launch_helper(
            core_attn_kernel,
            n_blocks,
            n_threads,
            [
                query_input,
                k_cache,
                v_cache,
                attn_output,
                global_barrier_buffer,
                global_max_denom_buffer,
                seq_len,
            ],
            shared_mem_size,
        )

    @ch.fn(
        ch.Params(
            query_input=ty.tensor_mut(dtype),
            k_cache=ty.tensor_mut(dtype),
            v_cache=ty.tensor_mut(dtype),
            attn_output=ty.tensor_mut(dtype),
            seq_len=ty.i32,
        )
    )
    def core_attn_torch(
        query_input,
        k_cache,
        v_cache,
        attn_output,
        seq_len,
    ):
        # Initialize global barrier buffer to zero (required by GlobalBarrierManager)
        global_barrier_buffer = ch.alloc_tensor(ty.i32, 1)
        # Create global max/denom buffer
        global_max_denom_buffer = ch.alloc_tensor(
            accum_dtype, n_blocks * query_per_kv * 2
        )

        core_attn_launch(
            query_input.cast(ty.ptr_mut(dtype)),
            k_cache.cast(ty.ptr_mut(dtype)),
            v_cache.cast(ty.ptr_mut(dtype)),
            attn_output.cast(ty.ptr_mut(dtype)),
            global_barrier_buffer.cast(ty.ptr_mut(ty.i32)),
            global_max_denom_buffer.cast(ty.ptr_mut(accum_dtype)),
            seq_len,
        )

    return core_attn_torch


def ref_attention(query_input, k_cache, v_cache, attn_output, seq_len):
    """Reference implementation for core attention using PyTorch."""

    # Reshape inputs for attention computation
    q = query_input.view(n_kv_heads, query_per_kv, d_head)
    k = k_cache.view(n_kv_heads, seq_len, d_head)
    v = v_cache.view(n_kv_heads, seq_len, d_head)

    # Compute attention for each KV head
    outputs = []
    for kv_head in range(n_kv_heads):
        q_head = q[kv_head]  # [query_per_kv, d_head]
        k_head = k[kv_head]  # [seq_len, d_head]
        v_head = v[kv_head]  # [seq_len, d_head]

        # Compute attention scores: Q @ K^T
        scores = torch.matmul(
            q_head, k_head.transpose(-2, -1)
        )  # [query_per_kv, seq_len]
        scores = scores / math.sqrt(d_head)  # Scale

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [query_per_kv, seq_len]

        # Compute output: P @ V
        output = torch.matmul(attn_weights, v_head)  # [query_per_kv, d_head]
        outputs.append(output)

    if debug:
        print("Ref Scores: ", scores)
        print("Ref Attn weights: ", attn_weights)
        print("Ref Output: ", output[0])

    # Concatenate outputs
    final_output = torch.stack(outputs, dim=0)  # [n_kv_heads, query_per_kv, d_head]
    attn_output.copy_(final_output)

    print(f"Reference implementation completed:")
    print(f"Query shape: {query_input.shape}")
    print(f"K cache shape: {k_cache.shape}")
    print(f"V cache shape: {v_cache.shape}")
    print(f"Attention output shape: {attn_output.shape}")


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Generating core attention new kernel code")
        kernel = gen_test_kernel()
        run_test_codegen(
            kernel,
            "core_attention_new_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    # Import kernel
    core_attn_kernel = import_codegen("core_attention_new_test", dir="tests", args=args)
    print("Kernel imported, preparing for execution")

    # Create input and output tensors
    torch_dtype = torch.bfloat16

    if args.init == "ones":
        query_input = torch.ones(
            (n_kv_heads, query_per_kv, d_head), dtype=torch_dtype, device="cuda"
        )
        k_cache = torch.ones(
            (n_kv_heads, seq_len, d_head), dtype=torch_dtype, device="cuda"
        )
        v_cache = torch.ones(
            (n_kv_heads, seq_len, d_head), dtype=torch_dtype, device="cuda"
        )
        print("Initialized tensors with ones")
    else:
        # Random initialization
        query_input = (
            torch.ones(n_kv_heads, query_per_kv, d_head).to(torch_dtype).to("cuda")
        )
        # query_input = (
        #     torch.randn(query_per_kv)
        #     .reshape(1, query_per_kv, 1)
        #     .repeat(n_kv_heads, 1, d_head)
        #     .to(torch_dtype)
        #     .to("cuda")
        # )
        # query_input = torch.randn(
        #     (n_kv_heads, query_per_kv, d_head), dtype=torch_dtype, device="cuda"
        # )
        k_cache = torch.randn(
            (n_kv_heads, seq_len, d_head), dtype=torch_dtype, device="cuda"
        )
        v_cache = torch.ones(
            (n_kv_heads, seq_len, d_head), dtype=torch_dtype, device="cuda"
        )
        print("Initialized tensors with random values")

    # Output tensor
    attn_output = torch.zeros(
        (n_kv_heads, query_per_kv, d_head), dtype=torch_dtype, device="cuda"
    )

    # Clone for reference implementation
    query_input_ref = query_input.clone()
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    attn_output_ref = torch.zeros_like(attn_output)

    # Run tests
    if args.test:
        print("Running core attention new test...")

        # Run reference implementation
        ref_attention(
            query_input_ref,
            k_cache_ref,
            v_cache_ref,
            attn_output_ref,
            seq_len,
        )

        # Run kernel implementation
        core_attn_kernel(
            query_input,
            k_cache,
            v_cache,
            attn_output,
            seq_len,
        )
        torch.cuda.synchronize()

        # Check results with tolerance
        rtol = 0.1  # 10% relative tolerance for bfloat16
        atol = 1e-3  # Absolute tolerance

        output_match = torch.allclose(
            attn_output_ref.float(), attn_output.float(), rtol=rtol, atol=atol
        )

        print(f"Output match: {output_match}")

        # Overall test result
        if output_match:
            print("TEST PASSED: Results match reference with acceptable tolerance!")
        else:
            print("Output mismatch:")
            print(
                "Reference output (first 5 elements):",
                attn_output_ref.flatten()[:5].float(),
            )
            print(
                "Kernel output (first 5 elements):", attn_output.flatten()[:5].float()
            )
            with open("attn_debug.txt", "w") as f:
                for r, k in zip(attn_output_ref.flatten(), attn_output.flatten()):
                    f.write(f"{r.item():.3f} {k.item():.3f}\n")

            print(
                "Difference:",
                (attn_output_ref.float() - attn_output.float()).abs().max(),
            )
            print("TEST FAILED: Results don't match reference!")

    # Run benchmark
    if args.benchmark:
        print("Benchmarking core attention new kernel")

        # Prepare benchmark parameters
        iterations = args.n_runs
        warmup = args.n_warmup

        # Warmup
        for _ in range(warmup):
            core_attn_kernel(
                query_input,
                k_cache,
                v_cache,
                attn_output,
                seq_len,
            )
            # Slight variation for each run
            query_input.add_(torch.randn_like(query_input) * 0.01)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            query_input.add_(torch.randn_like(query_input) * 0.01)
            core_attn_kernel(
                query_input,
                k_cache,
                v_cache,
                attn_output,
                seq_len,
            )

        torch.cuda.synchronize()
        end = time.time()

        # Calculate and report performance
        avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average kernel execution time: {avg_time_ms:.3f} ms")

        # Run reference for comparison
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            query_input_ref.add_(torch.randn_like(query_input_ref) * 0.01)
            ref_attention(
                query_input_ref,
                k_cache_ref,
                v_cache_ref,
                attn_output_ref,
                seq_len,
            )

        torch.cuda.synchronize()
        end = time.time()

        ref_avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average reference time: {ref_avg_time_ms:.3f} ms")

        if ref_avg_time_ms > 0:
            speedup = ref_avg_time_ms / avg_time_ms
            print(f"Speedup: {speedup:.2f}x")
