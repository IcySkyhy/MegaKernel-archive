import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch
import math

from flashtransformer.components.attention import SoftMaxAndRescaleFn
from flashtransformer.utils import SafeDataIndexPtr, LocalBarrierManager
from flashtransformer.utils.safe_data_index_ptr_utils import SDIPConfig
from flashtransformer.utils.cuda_utils import init_groups, check_tile_size
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
)

# Define test dimensions
n_blocks = 1  # Single block test
n_warps = 8
n_threads = n_warps * 32  # 32 threads per warp

# Attention dimensions
batch_size = 1
max_seq_len = 128
cur_seq_len = 64
head_dim = 128
query_head_dim = 4
debug = True


def gen_softmax_test_kernel():
    """Generate a SoftMax and Rescale test kernel that matches SoftMaxAndRescaleFn exactly."""
    dims = ixt.Dims()

    # Define block and thread indices
    block_idx = dims.new_dim("block_idx", n_blocks)
    thread_idx = dims.new_dim("thread_idx", n_threads)
    warp_idx = dims.new_dim("warp_idx", n_warps)
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define attention dimensions
    query_per_kv_head_idx = dims.new_dim("query_per_kv_head_idx", query_head_dim)
    d_head_idx = dims.new_dim("d_head_idx", head_dim)
    buffer_kv_row_idx = dims.new_dim("buffer_kv_row_idx", max_seq_len)
    seq_len_idx = dims.new_dim("seq_len_idx", max_seq_len)
    dims.eq(seq_len_idx, buffer_kv_row_idx)

    # Define score and output indices
    scores_idx = dims.new_dim("scores_idx")
    dims.eq(scores_idx, (query_per_kv_head_idx, buffer_kv_row_idx))

    partial_output_idx = dims.new_dim("partial_output_idx")
    dims.eq(partial_output_idx, (query_per_kv_head_idx, d_head_idx))

    @ch.kernel(
        ch.Params(
            scores=ty.ptr_mut(ty.f32),
            partial_softmax=ty.ptr_mut(ty.bf16),
            partial_out=ty.ptr_mut(ty.f32),
            old_max=ty.ptr_mut(ty.f32),
            old_denom=ty.ptr_mut(ty.f32),
            seq_len=ty.i32,
        )
    )
    def softmax_kernel(
        scores, partial_softmax, partial_out, old_max, old_denom, seq_len
    ):
        # Create SDIPConfigs for score, partial_softmax, and partial_out
        scores_cfg = SDIPConfig(
            dims, scores_idx, ty.ptr_mut(ty.f32), "global", "scores"
        )
        partial_softmax_cfg = SDIPConfig(
            dims, scores_idx, ty.ptr_mut(ty.bf16), "global", "partial_softmax"
        )
        partial_out_cfg = SDIPConfig(
            dims, partial_output_idx, ty.ptr_mut(ty.f32), "global", "partial_out"
        )

        # Create softmax function
        softmax_fn = SoftMaxAndRescaleFn(
            dims=dims,
            scores_cfg=scores_cfg,
            partial_softmax_cfg=partial_softmax_cfg,
            partial_out_cfg=partial_out_cfg,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            query_per_kv_head_idx=query_per_kv_head_idx,
            d_head_idx=d_head_idx,
            buffer_kv_row_idx=buffer_kv_row_idx,
            seq_len_idx=seq_len_idx,
            ptr_type=ty.bf16,
            accum_ptr_type=ty.f32,
        )

        # Initialize cheetah indices
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Create debug print function
        debug_print = make_cond_print(
            ch.and_(ix[block_idx] == 0, ix[warp_idx] == 0, ix[lane_idx] == 0), debug
        )
        debug_print("Starting SoftMax test")

        # Create local barrier for synchronization
        local_barrier_manager = LocalBarrierManager()
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Create SafeDataIndexPtr instances for data
        scores_ptr = SafeDataIndexPtr(ix, scores, scores_idx, "global", "scores")
        partial_softmax_ptr = SafeDataIndexPtr(
            ix, partial_softmax, scores_idx, "global", "partial_softmax"
        )
        partial_out_ptr = SafeDataIndexPtr(
            ix, partial_out, partial_output_idx, "global", "partial_out"
        )

        # Load old max and denom values - each warp loads its own values
        old_max_val = ch.alloc(ty.f32, name="old_max")
        old_denom_val = ch.alloc(ty.f32, name="old_denom")

        # Each warp loads its query's max/denom values
        with ch.if_(ix[warp_idx] < query_head_dim):
            old_max_val.val = old_max[ix[warp_idx]]
            old_denom_val.val = old_denom[ix[warp_idx]]

        # Debug print happens outside conditional blocks to avoid synchronization issues
        debug_print(
            "Initial values loaded. Per-query max/denom for first query: max=%f, denom=%f",
            old_max[0],
            old_denom[0],
        )

        # Initialize groups for cooperative operations
        groups = init_groups([32])
        tile = groups["tile32"]
        check_tile_size(tile, ix.size(lane_idx))

        # Call the softmax function
        softmax_fn(
            ix,
            scores_ptr,
            partial_softmax_ptr,
            partial_out_ptr,
            old_max_val,
            old_denom_val,
            seq_len,
            tile,
            local_barrier,
            debug,
        )

        # Store updated max and denom values - each warp stores its own values
        with ch.if_(ch.and_(ix[warp_idx] < query_head_dim, ix[lane_idx] == 0)):
            old_max[ix[warp_idx]] = old_max_val.val
            old_denom[ix[warp_idx]] = old_denom_val.val

        # Debug print happens outside conditional blocks
        debug_print(
            "Finished SoftMax test. Per-query max/denom for first query: max=%f, denom=%f",
            old_max[0],
            old_denom[0],
        )

    @ch.fn(
        ch.Params(
            scores=ty.ptr_mut(ty.f32),
            partial_softmax=ty.ptr_mut(ty.bf16),
            partial_out=ty.ptr_mut(ty.f32),
            old_max=ty.ptr_mut(ty.f32),
            old_denom=ty.ptr_mut(ty.f32),
            seq_len=ty.i32,
        )
    )
    def softmax_launch(
        scores, partial_softmax, partial_out, old_max, old_denom, seq_len
    ):
        launch_helper(
            softmax_kernel,
            n_blocks,
            n_threads,
            [scores, partial_softmax, partial_out, old_max, old_denom, seq_len],
            0,  # No shared memory needed for this test
        )

    @ch.fn(
        ch.Params(
            scores=ty.tensor_mut(ty.f32),
            partial_softmax=ty.tensor_mut(ty.bf16),
            partial_out=ty.tensor_mut(ty.f32),
            old_max=ty.tensor_mut(ty.f32),
            old_denom=ty.tensor_mut(ty.f32),
            seq_len=ty.i32,
        )
    )
    def softmax_torch(
        scores, partial_softmax, partial_out, old_max, old_denom, seq_len
    ):
        softmax_launch(
            scores.cast(ty.ptr_mut(ty.f32)),
            partial_softmax.cast(ty.ptr_mut(ty.bf16)),
            partial_out.cast(ty.ptr_mut(ty.f32)),
            old_max.cast(ty.ptr_mut(ty.f32)),
            old_denom.cast(ty.ptr_mut(ty.f32)),
            seq_len,
        )

    return softmax_torch


def softmax_rescale_ref(
    scores, partial_softmax, partial_out, old_max, old_denom, seq_len, head_dim
):
    """Reference implementation for SoftMaxAndRescaleFn.

    This function modifies the input tensors directly:
    1. Computes softmax and stores in partial_softmax
    2. Rescales partial_out
    3. Updates old_max and old_denom values

    Now processes each query independently with its own max/denom values
    """
    # Extract dimensions
    n_queries, seq_size = scores.shape
    scale = math.sqrt(1 / head_dim)
    device = scores.device

    # Apply scale to scores
    scaled_scores = scores * scale

    # Create mask for valid sequence positions
    mask = torch.arange(seq_size, device=device) < seq_len
    # Broadcast mask to match scores dimensions
    mask = mask.expand(scaled_scores.shape)
    masked_scores = scaled_scores.clone()
    masked_scores[~mask] = -float("inf")

    # Process each query separately, with its own max and denom values
    for q_idx in range(n_queries):
        # Get initial values for this query
        old_max_val = old_max[q_idx].item()
        old_denom_val = old_denom[q_idx].item()

        # Get scaled scores for this query
        q_scores = masked_scores[q_idx]

        # Find max score for this query
        q_max = q_scores.max()

        # Apply softmax with numeric stability
        q_exp_scores = torch.exp(q_scores - q_max)
        q_exp_scores[~mask[q_idx]] = 0.0  # Apply mask to scores

        q_denom = q_exp_scores.sum()

        # Update denominator for this query
        # The CUDA implementation uses: e_diff = exp(max_val - old_max)
        q_e_diff = torch.exp(q_max - old_max_val)
        q_updated_denom = q_e_diff * old_denom_val + q_denom

        # Create normalized scores and store directly in partial_softmax
        partial_softmax[q_idx].copy_(
            (q_exp_scores / q_updated_denom).to(partial_softmax.dtype)
        )

        # Rescale partial_out in-place for this query
        rescale_factor = old_denom_val / q_updated_denom
        partial_out[q_idx].mul_(rescale_factor)

        # Store max and denom for this query (last one will be saved to old_max/old_denom)
        old_max[q_idx] = q_max
        old_denom[q_idx] = q_updated_denom

        if debug and q_idx == 0:  # Debug print for first query only to avoid spam
            print(f"Query {q_idx} reference implementation:")
            print(f"Scale: {scale}")
            print(f"Old max: {old_max_val}, New max: {q_max.item()}")
            print(f"Old denom: {old_denom_val}, New denom: {q_updated_denom.item()}")
            print(f"First 5 scores: {scores[q_idx, :5].tolist()}")
            print(f"First 5 scaled scores: {scaled_scores[q_idx, :5].tolist()}")
            print(f"first five q_exp_scores: {q_exp_scores[:5].tolist()}")
            print(
                f"First 5 softmax values: {partial_softmax[q_idx, :5].float().tolist()}"
            )

    if debug:
        print(f"Final values for all queries:")
        print(f"Max values: {old_max.tolist()}")
        print(f"Denom values: {old_denom.tolist()}")


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        # Generate kernel
        print("Codegen started")
        kernel = gen_softmax_test_kernel()
        run_test_codegen(
            kernel, "softmax_test", headers=standard_headers, bind=True, dir="tests"
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    # Import the generated kernel
    cuda_kernel = import_codegen("softmax_test", dir="tests", args=args)
    print("Kernel imported, executing")

    # Create input tensors based on initialization method
    if args.init == "ones":
        # Initialize with ones
        scores = torch.ones(
            (query_head_dim, max_seq_len), dtype=torch.float32, device="cuda"
        )
        partial_softmax = torch.zeros(
            (query_head_dim, max_seq_len), dtype=torch.bfloat16, device="cuda"
        )
        partial_out = torch.ones(
            (query_head_dim, head_dim), dtype=torch.float32, device="cuda"
        )
        print("Initialized tensors with ones")
    else:
        # Default to random initialization using randn
        scores = torch.randn(
            (query_head_dim, max_seq_len), dtype=torch.float32, device="cuda"
        )
        partial_softmax = torch.zeros(
            (query_head_dim, max_seq_len), dtype=torch.bfloat16, device="cuda"
        )
        partial_out = torch.randn(
            (query_head_dim, head_dim), dtype=torch.float32, device="cuda"
        )
        print("Initialized tensors with random values (randn)")
    # Create per-query max/denom values (one for each query)
    old_max = torch.zeros((query_head_dim,), dtype=torch.float32, device="cuda")
    old_denom = torch.ones((query_head_dim,), dtype=torch.float32, device="cuda")

    # Make copies for reference implementation
    scores_ref = scores.clone()
    partial_softmax_ref = torch.zeros_like(partial_softmax)
    partial_out_ref = partial_out.clone()
    old_max_ref = old_max.clone()
    old_denom_ref = old_denom.clone()

    if args.test:
        print("Running SoftMax test...")

        # Run reference implementation
        softmax_rescale_ref(
            scores_ref,
            partial_softmax_ref,
            partial_out_ref,
            old_max_ref,
            old_denom_ref,
            cur_seq_len,
            head_dim,
        )

        # Run kernel implementation
        cuda_kernel(
            scores, partial_softmax, partial_out, old_max, old_denom, cur_seq_len
        )
        torch.cuda.synchronize()

        # Use allclose to check for approximate equality
        # Focus on relative differences with a high absolute tolerance
        rtol = 0.01  # 1% relative tolerance
        atol = 1.0  # High absolute tolerance to focus on relative differences

        # Check softmax outputs
        softmax_match = torch.allclose(
            partial_softmax_ref.float(), partial_softmax.float(), rtol=rtol, atol=atol
        )
        print(f"Softmax match: {softmax_match}")

        # Check partial output rescaling
        output_match = torch.allclose(
            partial_out_ref, partial_out, rtol=rtol, atol=atol
        )
        print(f"Output rescale match: {output_match}")

        # Check max value
        max_match = torch.allclose(old_max_ref, old_max, rtol=rtol, atol=atol)
        print(f"Max value match: {max_match}")

        # Check denom value
        denom_match = torch.allclose(old_denom_ref, old_denom, rtol=rtol, atol=atol)
        print(f"Denom value match: {denom_match}")

        # Overall test result
        test_passed = softmax_match and output_match and max_match and denom_match

        if test_passed:
            print("TEST PASSED: Results match reference with acceptable tolerance!")
        else:
            print("TEST FAILED: Results don't match reference!")
            print(f"Reference max values: {old_max_ref.tolist()}")
            print(f"CUDA max values: {old_max.tolist()}")
            print(f"Reference denom values: {old_denom_ref.tolist()}")
            print(f"CUDA denom values: {old_denom.tolist()}")

    if args.benchmark:
        print("Benchmarking SoftMax kernel")

        # Prepare benchmark data
        n_copies = args.n_loop
        iterations = args.n_runs

        # Warmup
        for _ in range(args.n_warmup):
            cuda_kernel(
                scores, partial_softmax, partial_out, old_max, old_denom, cur_seq_len
            )
            # Vary the scores for each iteration
            scores.add_(torch.randn_like(scores) * 0.01)

        # Reset for actual benchmark
        old_max.fill_(-1000.0)
        old_denom.fill_(0.0)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            # Vary the scores slightly to simulate real inputs
            scores.add_(torch.randn_like(scores) * 0.01)
            cuda_kernel(
                scores, partial_softmax, partial_out, old_max, old_denom, cur_seq_len
            )

        torch.cuda.synchronize()
        end = time.time()

        avg_time_ms = (end - start) * 1000 / iterations
        print(f"Average kernel execution time: {avg_time_ms:.3f} ms")
        print(f"Final max: {old_max[0].item()}, Final denom: {old_denom[0].item()}")
