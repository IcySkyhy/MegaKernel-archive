import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch
import torch.nn.functional as F

from flashtransformer.components.mat_ops.rmsnorm import RMSNormFn
from flashtransformer.components.memory import WGSyncCopyFn
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

# Define test dimensions
n_threads = 256
hidden_size = 4096  # Common hidden size for transformer models

debug = False


def gen_test_kernel():
    """Generate a kernel that uses RMSNorm for testing."""
    dims = ixt.Dims()

    # Define thread indices
    thread_idx = dims.new_dim("thread_idx", n_threads)

    # Define hidden dimension for RMSNorm
    hidden_idx = dims.new_dim("hidden_idx", hidden_size)

    # Define SafeDataIndexPtr configs for global memory
    in_cfg = SDIPConfig(dims, hidden_idx, ty.ptr_const(ty.f32), "global", "in_data")
    weight_cfg = SDIPConfig(dims, hidden_idx, ty.ptr_const(ty.f32), "global", "weight")
    out_cfg = SDIPConfig(dims, hidden_idx, ty.ptr_mut(ty.f32), "global", "out_data")

    # Define SafeDataIndexPtr configs for shared memory
    shared_in_cfg = SDIPConfig(
        dims, hidden_idx, ty.ptr_mut(ty.f32), "shared", "shared_in"
    )
    shared_weight_cfg = SDIPConfig(
        dims, hidden_idx, ty.ptr_mut(ty.f32), "shared", "shared_weight"
    )
    shared_out_cfg = SDIPConfig(
        dims, hidden_idx, ty.ptr_mut(ty.f32), "shared", "shared_out"
    )
    dummy_idx = dims.new_dim("dummy_idx", 1)
    rms_shared_cfg = SDIPConfig(
        dims, dummy_idx, ty.ptr_mut(ty.f32), "shared", "rms_shared"
    )

    @ch.kernel(
        ch.Params(
            in_data=ty.ptr_const(ty.f32),
            weight=ty.ptr_const(ty.f32),
            out_data=ty.ptr_mut(ty.f32),
            eps=ty.f32,
        )
    )
    def rmsnorm_kernel(in_data, weight, out_data, eps):
        # Create RMSNorm function
        rmsnorm_fn = RMSNormFn(
            dims,
            shared_in_cfg,
            shared_out_cfg,
            shared_weight_cfg,
            thread_idx,
            hidden_idx,
            eps=eps,
        )

        # Create copy functions for input, weight, and output
        copy_in = WGSyncCopyFn(
            dims,
            in_cfg,
            shared_in_cfg,
            thread_idx,
            hidden_idx,
        )

        copy_weight = WGSyncCopyFn(
            dims,
            weight_cfg,
            shared_weight_cfg,
            thread_idx,
            hidden_idx,
        )

        copy_out = WGSyncCopyFn(
            dims,
            shared_out_cfg,
            out_cfg,
            thread_idx,
            hidden_idx,
        )

        # Create local barrier for synchronization
        local_barrier_manager = LocalBarrierManager()
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # Initialize cheetah indices
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        # Set up debug printing
        debug_print = make_cond_print(
            ch.and_(ix[thread_idx] == 0), debug, local_barrier
        )
        debug_print("Starting RMSNorm test...")

        # Create SafeDataIndexPtr instances for global memory
        global_in_ptr = in_cfg.wrap_ptr(ix, in_data)
        global_weight_ptr = weight_cfg.wrap_ptr(ix, weight)
        global_out_ptr = out_cfg.wrap_ptr(ix, out_data)

        # Allocate shared memory for input, weights, and output
        shared_in_ptr = shared_in_cfg.alloc_static_shared(ix)
        shared_weight_ptr = shared_weight_cfg.alloc_static_shared(ix)
        shared_out_ptr = shared_out_cfg.alloc_static_shared(ix)

        # Allocate shared memory for RMS calculations (one per thread)
        rms_shared_ptr = rms_shared_cfg.alloc_static_shared(ix)

        # Copy input and weights to shared memory
        copy_in(ix, global_in_ptr, shared_in_ptr, local_barrier)
        copy_weight(ix, global_weight_ptr, shared_weight_ptr, local_barrier)

        debug_print("Copied inputs to shared memory")

        # Sample input/weight values for debugging
        debug_print("Input[0]: {}", shared_in_ptr.raw_idx(0).cast(ty.f32))
        debug_print("Weight[0]: {}", shared_weight_ptr.raw_idx(0).cast(ty.f32))

        groups = init_groups([n_threads])
        tile = groups[f"tile{n_threads}"]

        # Run RMSNorm
        rmsnorm_fn(
            ix,
            shared_in_ptr,
            shared_out_ptr,
            shared_weight_ptr,
            rms_shared_ptr,
            tile,
            local_barrier,
            debug,
        )

        debug_print("Completed RMSNorm computation")

        # Sample output value for debugging
        debug_print("Output[0]: {}", shared_out_ptr.raw_idx(0).cast(ty.f32))

        # Copy result back to global memory
        copy_out(ix, shared_out_ptr, global_out_ptr, local_barrier)

        debug_print("Copied results back to global memory")

    @ch.fn(
        ch.Params(
            in_data=ty.ptr_const(ty.f32),
            weight=ty.ptr_const(ty.f32),
            out_data=ty.ptr_mut(ty.f32),
            eps=ty.f32,
        )
    )
    def rmsnorm_launch(in_data, weight, out_data, eps):
        # Calculate shared memory size (input + weight + output + rms cache)
        shared_mem = hidden_size * 3 * 4 + n_threads * 4  # 4 bytes per float

        launch_helper(
            rmsnorm_kernel,
            1,  # 1 block is sufficient for one RMSNorm operation
            n_threads,
            [in_data, weight, out_data, eps],
            shared_mem,
        )

    @ch.fn(
        ch.Params(
            in_data=ty.tensor_const(ty.f32),
            weight=ty.tensor_const(ty.f32),
            out_data=ty.tensor_mut(ty.f32),
            eps=ty.f32,
        )
    )
    def rmsnorm_torch(in_data, weight, out_data, eps):
        rmsnorm_launch(
            in_data.cast(ty.ptr_const(ty.f32)),
            weight.cast(ty.ptr_const(ty.f32)),
            out_data.cast(ty.ptr_mut(ty.f32)),
            eps,
        )

    return rmsnorm_torch


def rmsnorm_ref_debug(input_data, weight, output, eps=1e-5):
    """
    Reference implementation using PyTorch's RMS normalization.

    RMSNorm normalizes the activations by the RMS (root mean square) and applies
    a learnable scale (weight) parameter.
    """
    # Manually implement RMS normalization since there might be
    # issues with the PyTorch implementation
    input_dtype = input_data.dtype

    # Calculate sum of squares
    sum_squares = torch.sum(input_data**2)

    # Calculate mean squares
    mean_squares = sum_squares / input_data.numel()

    # Calculate RMS and add epsilon for numerical stability
    rms = torch.sqrt(mean_squares) + eps

    # Normalize and scale
    normalized = (input_data / rms) * weight

    # Print debug information
    if debug:
        print("Reference implementation:")
        print(f"Input shape: {input_data.shape}, weight shape: {weight.shape}")
        print(f"Sum squares: {sum_squares.item()}")
        print(f"Mean squares: {mean_squares.item()}")
        print(f"RMS value: {rms.item()}")
        print(f"Input[0:5]: {input_data[:5].tolist()}")
        print(f"Weight[0:5]: {weight[:5].tolist()}")
        print(f"First 5 normalized values: {normalized[:5].tolist()}")

        # Let's also print the first few values that would be calculated by our CUDA code
        # This will help us confirm if the formulas match
        cuda_rms = torch.sqrt(mean_squares) + eps
        cuda_inv_rms = 1.0 / cuda_rms
        print(f"Calculated inverse RMS value: {cuda_inv_rms.item()}")
        print(
            f"First 5 values with direct formula: {(input_data[:5] * cuda_inv_rms * weight[:5]).tolist()}"
        )

    # Copy to output tensor
    output.copy_(normalized)

    return output


def rmsnorm_ref(input_data, weight, output, eps=1e-5):
    if debug:
        rmsnorm_ref_debug(input_data, weight, output, eps)
    else:
        out = torch.nn.functional.rms_norm(input_data, (hidden_size,), weight, eps)
        output.copy_(out)


if __name__ == "__main__":
    args = launch_args()
    eps = 1e-5

    if not args.no_codegen:
        # Generate kernel
        print("Codegen started")
        run_test_codegen(
            gen_test_kernel(),
            "rmsnorm_kernel_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    rmsnorm_kernel = import_codegen("rmsnorm_kernel_test", dir="tests", args=args)
    print("Kernel compiled and imported, executing")

    # Create test data based on initialization method
    if args.init == "ones":
        input_data = torch.ones(hidden_size, dtype=torch.float32)
        weight = torch.ones(hidden_size, dtype=torch.float32)
        print("Using ones initialization for easier debugging")
    else:
        input_data = torch.randn(hidden_size, dtype=torch.float32)
        weight = torch.randn(hidden_size, dtype=torch.float32)

    # Test RMSNorm operation
    out_ref = torch.zeros(hidden_size, dtype=torch.float32)
    out_kernel = torch.zeros(hidden_size, dtype=torch.float32)

    # Run reference implementation
    rmsnorm_ref(input_data, weight, out_ref, eps)

    # Run kernel implementation
    rmsnorm_kernel(input_data, weight, out_kernel, torch.tensor(eps))
    torch.cuda.synchronize()

    # Check results
    if args.test:
        print("RMSNorm operation result:")
        print("Reference output (first 10 elements):")
        print(out_ref[:10])
        print("Kernel output (first 10 elements):")
        print(out_kernel[:10])
        print(
            "Max absolute difference:",
            torch.max(torch.abs(out_ref - out_kernel)).item(),
        )

        # Allow some numerical difference due to different computation methods
        assert torch.allclose(
            out_kernel, out_ref, rtol=1e-4, atol=1e-5
        ), "Outputs don't match!"

        print("\nAll tests passed!")

    if args.benchmark:
        print("Benchmarking RMSNorm operation")

        # Prepare benchmark data
        # Use multiple copies for looping during benchmark
        n_copies = args.n_loop
        input_bench = torch.randn(n_copies, hidden_size, dtype=torch.float32)
        weight_bench = torch.randn(n_copies, hidden_size, dtype=torch.float32)
        out_ref_bench = torch.zeros(n_copies, hidden_size, dtype=torch.float32)
        out_kernel_bench = torch.zeros(n_copies, hidden_size, dtype=torch.float32)

        times_ref = []
        times_kernel = []

        # Run benchmarks
        for i in range(args.n_warmup + args.n_runs):
            # Benchmark reference implementation
            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(n_copies):
                rmsnorm_ref(input_bench[j], weight_bench[j], out_ref_bench[j], eps)
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_ref.append(end_time - start_time)

            # Benchmark kernel implementation
            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(n_copies):
                rmsnorm_kernel(
                    input_bench[j],
                    weight_bench[j],
                    out_kernel_bench[j],
                    torch.tensor(eps),
                )
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_kernel.append(end_time - start_time)

        # Report benchmark results
        avg_time_ref = sum(times_ref) / len(times_ref) / n_copies
        avg_time_kernel = sum(times_kernel) / len(times_kernel) / n_copies

        print(f"RMSNorm performance (average time per operation):")
        print(f"Reference: {avg_time_ref:.6f} seconds")
        print(f"Kernel:    {avg_time_kernel:.6f} seconds")
        print(f"Speedup:   {avg_time_ref / avg_time_kernel:.2f}x")
