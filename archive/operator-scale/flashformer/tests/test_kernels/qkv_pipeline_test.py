import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch

from flashtransformer.components.attention import QKV_ComputeFn
from flashtransformer.utils import SafeDataIndexPtr, LocalBarrierManager
from flashtransformer.utils.safe_data_index_ptr_utils import SDIPConfig
from flashtransformer.utils.memory_pipeline import MemoryPipeline
from flashtransformer.utils.cuda_utils import init_groups, check_tile_size
from flashtransformer.components.memory.async_copy import AsyncBarrierGTSCopyFn
from flashtransformer.components.memory.sync_copy import WGSyncCopyFn
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
d_in = 2048
d_head = 128
n_heads = 16
n_kv_heads = 4
query_per_kv = n_heads // n_kv_heads
max_seq_len = 128
cur_seq_len = 64
debug = True


def gen_qkv_pipeline_test_kernel():
    """Generate a QKV compute test kernel using MemoryPipeline for producer-consumer pattern."""
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
    intra_buffer_idx = dims.new_dim("intra_buffer_idx", 8192)

    # Model dimensions
    d_in_idx = dims.new_dim("d_in_idx", d_in)
    d_head_idx = dims.new_dim("d_head_idx", d_head)
    head_idx = dims.new_dim("head_idx", n_heads)
    kv_head_idx = dims.new_dim("kv_head_idx", n_kv_heads)
    query_per_kv_head_idx = dims.new_dim("query_per_kv_head_idx", query_per_kv)
    seq_len_idx = dims.new_dim("seq_len_idx", max_seq_len)

    # Create SDIPConfig instances
    input_cfg = SDIPConfig(
        dims, d_in_idx, ty.ptr_mut(ty.bf16), "global", "input_tensor"
    )
    shared_inp_cfg = SDIPConfig(
        dims, d_in_idx, ty.ptr_mut(ty.bf16), "shared", "shared_input_tensor"
    )
    q_out_cfg = SDIPConfig(
        dims, (head_idx, d_head_idx), ty.ptr_mut(ty.bf16), "global", "q_output"
    )
    kv_cache_cfg = SDIPConfig(
        dims,
        (kv_head_idx, seq_len_idx, d_head_idx),
        ty.ptr_mut(ty.bf16),
        "global",
        "kv_cache",
    )
    q_weight_cfg = SDIPConfig(
        dims,
        (head_idx, d_head_idx, d_in_idx),
        ty.ptr_mut(ty.bf16),
        "global",
        "q_weights",
    )
    k_weight_cfg = SDIPConfig(
        dims,
        (kv_head_idx, d_head_idx, d_in_idx),
        ty.ptr_mut(ty.bf16),
        "global",
        "k_weights",
    )
    v_weight_cfg = SDIPConfig(
        dims,
        (kv_head_idx, d_head_idx, d_in_idx),
        ty.ptr_mut(ty.bf16),
        "global",
        "v_weights",
    )
    # Set up debug print function

    # Setup phase: Create all objects before initializing dimensions

    # Local barrier manager and Memory Pipeline for producer-consumer pattern
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

    copy_inp_fn = WGSyncCopyFn(
        dims,
        input_cfg,
        shared_inp_cfg,
        thread_idx,
        d_in_idx,
    )

    # Create QKV compute function using explicit keyword arguments, omitting lane_idx
    qkv_compute_fn = QKV_ComputeFn(
        dims=dims,
        in_cfg=shared_inp_cfg,
        q_out_cfg=q_out_cfg,
        kv_cache_cfg=kv_cache_cfg,
        q_weight_cfg=q_weight_cfg,
        kv_weight_cfg=k_weight_cfg,
        mem_pipeline=pipeline,
        block_idx=block_idx,
        head_idx=head_idx,
        kv_head_idx=kv_head_idx,
        query_per_kv_head_idx=query_per_kv_head_idx,
        d_head_idx=d_head_idx,
        seq_len_idx=seq_len_idx,
        quantized=False,
        d_in_qcfg=None,
    )

    # Use extern shared memory allocated by the launch wrapper

    @ch.kernel(
        ch.Params(
            input_tensor=ty.ptr_mut(ty.bf16),
            q_weights=ty.ptr_mut(ty.bf16),
            k_weights=ty.ptr_mut(ty.bf16),
            v_weights=ty.ptr_mut(ty.bf16),
            q_output=ty.ptr_mut(ty.bf16),
            k_cache=ty.ptr_mut(ty.bf16),
            v_cache=ty.ptr_mut(ty.bf16),
            seq_len=ty.i32,
        )
    )
    def qkv_pipeline_kernel(
        input_tensor,
        q_weights,
        k_weights,
        v_weights,
        q_output,
        k_cache,
        v_cache,
        seq_len,
    ):

        ix = dims.init()
        debug_print = make_cond_print(
            ch.and_(ch.block_idx_x() == 0, ch.thread_idx_x() == 0), debug
        )
        debug_print("Starting QKV compute with Memory Pipeline")
        # Initialize barriers for synchronization
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)
        # Initialize cooperative groups
        groups = init_groups([32])
        shared_ptr = ch.alloc_extern_shared(ty.bf16)
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Create SafeDataIndexPtr instances
        input_ptr = input_cfg.wrap_ptr(ix, input_tensor)
        q_weights_ptr = q_weight_cfg.wrap_ptr(ix, q_weights)
        k_weights_ptr = k_weight_cfg.wrap_ptr(ix, k_weights)
        v_weights_ptr = v_weight_cfg.wrap_ptr(ix, v_weights)
        q_output_ptr = q_out_cfg.wrap_ptr(ix, q_output)
        k_cache_ptr = kv_cache_cfg.wrap_ptr(ix, k_cache)
        v_cache_ptr = kv_cache_cfg.wrap_ptr(ix, v_cache)

        # Check tile size
        tile = groups["tile32"]
        check_tile_size(tile, ix.size(lane_idx))

        # Initialize pipeline and QKV function memory
        debug_print("Initializing shared memory")
        pipeline_buffer, next_shared_ptr = pipeline.initialize(ix, shared_ptr)
        shared_inp, next_shared_ptr = shared_inp_cfg.alloc_dynamic_shared(
            ix, next_shared_ptr
        )
        next_shared_ptr = qkv_compute_fn.setup_shmem(ix, next_shared_ptr)
        copy_inp_fn(ix, input_ptr, shared_inp, local_barrier)

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
                qkv_compute_fn.consume(
                    ix,
                    shared_inp,
                    next_shared_ptr,
                    q_output_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    seq_len,
                    groups,
                )

            with branch.else_():
                # Producer warps
                ix.set_index(
                    producer_warp_idx, ix[warp_idx] - ix.size(consumer_warp_idx)
                )
                debug_print("Starting producer warps")

                # Execute producer tasks
                qkv_compute_fn.produce(ix, q_weights_ptr, k_weights_ptr, v_weights_ptr)

        debug_print("QKV computation completed")

    @ch.fn(
        ch.Params(
            input_tensor=ty.ptr_mut(ty.bf16),
            q_weights=ty.ptr_mut(ty.bf16),
            k_weights=ty.ptr_mut(ty.bf16),
            v_weights=ty.ptr_mut(ty.bf16),
            q_output=ty.ptr_mut(ty.bf16),
            k_cache=ty.ptr_mut(ty.bf16),
            v_cache=ty.ptr_mut(ty.bf16),
            seq_len=ty.i32,
        )
    )
    def qkv_pipeline_launch(
        input_tensor,
        q_weights,
        k_weights,
        v_weights,
        q_output,
        k_cache,
        v_cache,
        seq_len,
    ):
        # Calculate shared memory size from pipeline and QKV component sizes
        pipeline_size = pipeline.get_shared_size()
        qkv_size = qkv_compute_fn.get_shmem_size()
        inp_size = shared_inp_cfg.get_size()

        # Add barrier size (8 bytes per barrier in pipeline stages)
        barrier_size = pipeline_stages * 8

        # Total shared memory size
        shared_mem_size = pipeline_size + qkv_size + barrier_size + inp_size

        launch_helper(
            qkv_pipeline_kernel,
            n_blocks,
            n_threads,
            [
                input_tensor,
                q_weights,
                k_weights,
                v_weights,
                q_output,
                k_cache,
                v_cache,
                seq_len,
            ],
            shared_mem_size,
        )

    @ch.fn(
        ch.Params(
            input_tensor=ty.tensor_mut(ty.bf16),
            q_weights=ty.tensor_mut(ty.bf16),
            k_weights=ty.tensor_mut(ty.bf16),
            v_weights=ty.tensor_mut(ty.bf16),
            q_output=ty.tensor_mut(ty.bf16),
            k_cache=ty.tensor_mut(ty.bf16),
            v_cache=ty.tensor_mut(ty.bf16),
            seq_len=ty.i32,
        )
    )
    def qkv_pipeline_torch(
        input_tensor,
        q_weights,
        k_weights,
        v_weights,
        q_output,
        k_cache,
        v_cache,
        seq_len,
    ):
        qkv_pipeline_launch(
            input_tensor.cast(ty.ptr_mut(ty.bf16)),
            q_weights.cast(ty.ptr_mut(ty.bf16)),
            k_weights.cast(ty.ptr_mut(ty.bf16)),
            v_weights.cast(ty.ptr_mut(ty.bf16)),
            q_output.cast(ty.ptr_mut(ty.bf16)),
            k_cache.cast(ty.ptr_mut(ty.bf16)),
            v_cache.cast(ty.ptr_mut(ty.bf16)),
            seq_len,
        )

    return qkv_pipeline_torch


def qkv_compute_ref(
    input_tensor, q_weights, k_weights, v_weights, q_output, k_cache, v_cache, seq_len
):
    """Reference implementation for QKV computation using PyTorch."""
    # Reshape inputs for matmul
    input_reshaped = input_tensor.unsqueeze(0)  # [1, d_in]

    # Reshape weights for matmul
    q_weights_reshaped = q_weights.view(n_heads * d_head, d_in).transpose(0, 1)
    k_weights_reshaped = k_weights.view(n_kv_heads * d_head, d_in).transpose(0, 1)
    v_weights_reshaped = v_weights.view(n_kv_heads * d_head, d_in).transpose(0, 1)

    # Perform matrix multiplications
    q_result = torch.matmul(input_reshaped, q_weights_reshaped)  # [1, n_heads*d_head]
    k_result = torch.matmul(
        input_reshaped, k_weights_reshaped
    )  # [1, n_kv_heads*d_head]
    v_result = torch.matmul(
        input_reshaped, v_weights_reshaped
    )  # [1, n_kv_heads*d_head]

    # Reshape Q output
    q_reshaped = q_result.view(n_heads, d_head)
    q_output.copy_(q_reshaped)

    # Reshape and store K and V in cache
    k_reshaped = k_result.view(n_kv_heads, d_head)
    v_reshaped = v_result.view(n_kv_heads, d_head)

    # Update KV cache (all positions up to seq_len)
    k_cache[:, seq_len, :].copy_(k_reshaped)
    v_cache[:, seq_len, :].copy_(v_reshaped)

    print(f"Reference implementation completed:")
    print(f"Q output shape: {q_output.shape}")
    print(f"K cache shape: {k_cache.shape}")
    print(f"V cache shape: {v_cache.shape}")
    print(f"Current sequence length: {seq_len}")


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Generating QKV pipeline kernel code")
        kernel = gen_qkv_pipeline_test_kernel()
        run_test_codegen(
            kernel,
            "qkv_pipeline_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    # Import kernel
    pipeline_kernel = import_codegen("qkv_pipeline_test", dir="tests", args=args)
    print("Kernel imported, preparing for execution")

    # Create input and output tensors
    if args.init == "ones":
        input_tensor = torch.ones((d_in,), dtype=torch.bfloat16, device="cuda")
        q_weights = torch.ones(
            (n_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        k_weights = torch.ones(
            (n_kv_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        v_weights = torch.ones(
            (n_kv_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        print("Initialized tensors with ones")
    else:
        # Random initialization
        input_tensor = torch.randn((d_in,), dtype=torch.bfloat16, device="cuda")
        q_weights = torch.randn(
            (n_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        k_weights = torch.randn(
            (n_kv_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        v_weights = torch.randn(
            (n_kv_heads, d_head, d_in), dtype=torch.bfloat16, device="cuda"
        )
        print("Initialized tensors with random values")

    # Output tensors
    q_output = torch.zeros((n_heads, d_head), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros(
        (n_kv_heads, max_seq_len, d_head), dtype=torch.bfloat16, device="cuda"
    )
    v_cache = torch.zeros(
        (n_kv_heads, max_seq_len, d_head), dtype=torch.bfloat16, device="cuda"
    )

    # Clone for reference implementation
    input_tensor_ref = input_tensor.clone()
    q_weights_ref = q_weights.clone()
    k_weights_ref = k_weights.clone()
    v_weights_ref = v_weights.clone()
    q_output_ref = torch.zeros_like(q_output)
    k_cache_ref = torch.zeros_like(k_cache)
    v_cache_ref = torch.zeros_like(v_cache)

    # Run tests
    if args.test:
        print("Running QKV pipeline compute test...")

        # Run reference implementation
        qkv_compute_ref(
            input_tensor_ref,
            q_weights_ref,
            k_weights_ref,
            v_weights_ref,
            q_output_ref,
            k_cache_ref,
            v_cache_ref,
            cur_seq_len,
        )

        # Run kernel implementation
        pipeline_kernel(
            input_tensor,
            q_weights,
            k_weights,
            v_weights,
            q_output,
            k_cache,
            v_cache,
            cur_seq_len,
        )
        torch.cuda.synchronize()

        # Check results with tolerance
        rtol = 0.1  # 1% relative tolerance
        atol = 1  # Absolute tolerance

        q_match = torch.allclose(
            q_output_ref.float(), q_output.float(), rtol=rtol, atol=atol
        )
        k_match = torch.allclose(
            k_cache_ref[:, :cur_seq_len, :].float(),
            k_cache[:, :cur_seq_len, :].float(),
            rtol=rtol,
            atol=atol,
        )
        v_match = torch.allclose(
            v_cache_ref[:, :cur_seq_len, :].float(),
            v_cache[:, :cur_seq_len, :].float(),
            rtol=rtol,
            atol=atol,
        )

        print(f"Q output match: {q_match}")
        print(f"K cache match: {k_match}")
        print(f"V cache match: {v_match}")

        # Overall test result
        if q_match and k_match and v_match:
            print("TEST PASSED: Results match reference with acceptable tolerance!")
        else:
            if not q_match:
                print("Q output mismatch:")
                print(q_output_ref.float())
                print(q_output.float())
                float_ref = (
                    q_weights_ref.reshape(n_heads * d_head, d_in).float()
                    @ input_tensor_ref.float()
                ).reshape(n_heads, d_head)
                print(
                    "ref to fp32 ref:", (q_output_ref.float() - float_ref).pow(2).sum()
                )
                print(
                    "kernel to fp32 ref:", (q_output.float() - float_ref).pow(2).sum()
                )
            if not k_match:
                print("K cache mismatch:")
                print(k_cache_ref[:, :cur_seq_len, :].float())
                print(k_cache[:, :cur_seq_len, :].float())
            if not v_match:
                print("V cache mismatch:")
                print(v_cache_ref[:, :cur_seq_len, :].float())
                print(v_cache[:, :cur_seq_len, :].float())
            print("TEST FAILED: Results don't match reference!")

    # Run benchmark
    if args.benchmark:
        print("Benchmarking QKV pipeline compute kernel")

        # Prepare benchmark parameters
        iterations = args.n_runs
        warmup = args.n_warmup

        # Warmup
        for _ in range(warmup):
            pipeline_kernel(
                input_tensor,
                q_weights,
                k_weights,
                v_weights,
                q_output,
                k_cache,
                v_cache,
                cur_seq_len,
            )
            # Slight variation for each run
            input_tensor.add_(torch.randn_like(input_tensor) * 0.01)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            input_tensor.add_(torch.randn_like(input_tensor) * 0.01)
            pipeline_kernel(
                input_tensor,
                q_weights,
                k_weights,
                v_weights,
                q_output,
                k_cache,
                v_cache,
                cur_seq_len,
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
            input_tensor_ref.add_(torch.randn_like(input_tensor_ref) * 0.01)
            qkv_compute_ref(
                input_tensor_ref,
                q_weights_ref,
                k_weights_ref,
                v_weights_ref,
                q_output_ref,
                k_cache_ref,
                v_cache_ref,
                cur_seq_len,
            )

        torch.cuda.synchronize()
        end = time.time()

        ref_avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average reference time: {ref_avg_time_ms:.3f} ms")

        if ref_avg_time_ms > 0:
            speedup = ref_avg_time_ms / avg_time_ms
            print(f"Speedup: {speedup:.2f}x")
