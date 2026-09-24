import cheetah.api as ch
from cheetah.api import ty
import time
import cheetah.index_tools as ixt
import torch

from flashtransformer.utils.memory_pipeline import MemoryPipeline
from flashtransformer.components.memory import WGSyncCopyFn, AsyncBarrierGTSCopyFn
from flashtransformer.utils.barrier_utils import LocalBarrierManager
from flashtransformer.utils.safe_data_index_ptr_utils import (
    SafeDataIndexPtr,
    SDIPConfig,
)
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
)

# Matrix dimensions
M = 512  # Matrix height
N = 512  # Matrix width
CHUNK_SIZE = 1024  # Size of each chunk to copy

# Thread layout
N_THREADS = 512
N_BLOCKS = 1
PIPELINE_STAGES = 4
debug = False


def gen_test_kernel():
    """Generate a kernel that uses memory pipeline to copy a matrix in chunks."""
    dims = ixt.Dims()

    # Define thread dimensions
    thread_idx = dims.new_dim("thread_idx", N_THREADS)
    warp_idx = dims.new_dim("warp_idx")
    lane_idx = dims.new_dim("lane_idx", 32)
    dims.eq(thread_idx, (warp_idx, lane_idx))

    # Define matrix dimensions
    mat_idx = dims.new_dim("mat_idx", M * N)

    # Define pipeline stage index

    # Configuration for data pointers
    src_cfg = SDIPConfig(dims, mat_idx, ty.ptr_const(ty.f32), "global", "src_matrix")
    dst_cfg = SDIPConfig(dims, mat_idx, ty.ptr_mut(ty.f32), "global", "dst_matrix")

    @ch.kernel(
        ch.Params(
            src=ty.ptr_const(ty.f32),
            dst=ty.ptr_mut(ty.f32),
        )
    )
    def pipelined_matrix_copy_kernel(src, dst):

        iter_idx = dims.new_dim("iter_idx")

        local_barrier_manager = LocalBarrierManager()
        consumer_barrier = local_barrier_manager.get_local_barrier(N_THREADS // 2)

        # Define consumer thread index (first 256 threads)
        consumer_warp_idx = dims.new_dim("consumer_warp_idx", N_THREADS // 64)
        producer_warp_idx = dims.new_dim("producer_warp_idx", N_THREADS // 64)
        # Create memory pipeline manager
        pipeline = MemoryPipeline(
            dims,
            pipeline_stages=PIPELINE_STAGES,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            consumer_warp_idx=consumer_warp_idx,
            producer_warp_idx=producer_warp_idx,
            local_barrier_manager=local_barrier_manager,
            chunk_size=CHUNK_SIZE,
            ptr_type=ty.f32,
        )

        chunk_idx = pipeline.intra_buffer_idx

        dims.eq(mat_idx, (iter_idx, chunk_idx))
        # Create copy functions for the pipeline
        copy_to_shared_fn = AsyncBarrierGTSCopyFn(
            dims,
            src_cfg,
            pipeline.get_total_buffer_cfg(),
            pipeline.producer_thread_idx,
            chunk_idx,
        )

        copy_from_shared_fn = WGSyncCopyFn(
            dims,
            pipeline.get_single_buffer_cfg(),  # Source: shared memory chunk
            dst_cfg,  # Destination: global memory chunk
            pipeline.consumer_thread_idx,
            chunk_idx,
        )

        # Initialize indices after all dimensions are created
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))

        debug_print = make_cond_print(
            ch.and_(ix[warp_idx] == 0, ix[lane_idx] == 0), debug
        )
        debug_print("Starting pipelined matrix copy")

        src = src_cfg.wrap_ptr(ix, src)
        dst = dst_cfg.wrap_ptr(ix, dst)

        # Allocate shared memory for the pipeline
        shared_ptr = ch.alloc_extern_shared(ty.i8)
        buffer, _ = pipeline.initialize(ix, shared_ptr)

        debug_print(
            "Pipeline initialized with %d stages", ch.const(PIPELINE_STAGES, ty.i32)
        )
        ch.raw_stmt("__syncthreads();")

        # Copy chunks through the pipeline
        with ch.if_else(
            ix[thread_idx] < ix.size(pipeline.consumer_thread_idx)
        ) as branch:
            with branch.then():
                ix.set_index(pipeline.consumer_warp_idx, ix[warp_idx])
                # consumer_debug_print = make_cond_print(
                #     ix[consumer_thread_idx] == 0, debug, consumer_barrier
                # )
                # consumer_debug_print(f"consumer readying")
                pipeline.consumer_pipeline_ready(ix)
                # consumer_debug_print(f"consumer ready")
                with ix.loop(iter_idx):
                    chunk_buffer, stage = pipeline.consumer_get_shared_buffer(
                        f"chunk_{iter_idx}"
                    )
                    copy_from_shared_fn(ix, chunk_buffer, dst, consumer_barrier)
                    pipeline.consumer_release_shared_buffer(stage)
            with branch.else_():
                ix.set_index(
                    pipeline.producer_warp_idx,
                    ix[warp_idx] - ix.size(pipeline.consumer_warp_idx),
                )
                # producer_debug_print = make_cond_print(
                #     ix[producer_thread_idx] == 0, debug
                # )
                with ix.loop(iter_idx):
                    # producer_debug_print(f"producer iter idx %d", ix[iter_idx])
                    pipeline.producer_copy(src, copy_to_shared_fn)

    @ch.fn(ch.Params(src=ty.ptr_const(ty.f32), dst=ty.ptr_mut(ty.f32)))
    def pipelined_matrix_copy_launch(src, dst):
        # Calculate shared memory size
        # Need memory for pipeline buffer plus barriers
        pipeline_buffer_size = PIPELINE_STAGES * CHUNK_SIZE * 4  # f32 = 4 bytes
        mbarrier_size = PIPELINE_STAGES * 8  # Size for SharedMBarrierArray
        shared_mem = pipeline_buffer_size + mbarrier_size

        launch_helper(
            pipelined_matrix_copy_kernel,
            N_BLOCKS,
            N_THREADS,
            [src, dst],
            shared_mem,
        )

    @ch.fn(ch.Params(src=ty.tensor_const(ty.f32), dst=ty.tensor_mut(ty.f32)))
    def pipelined_matrix_copy_torch(src, dst):
        pipelined_matrix_copy_launch(
            src.cast(ty.ptr_const(ty.f32)), dst.cast(ty.ptr_mut(ty.f32))
        )

    return pipelined_matrix_copy_torch


def reference_copy(src, dst):
    """Reference implementation of matrix copy"""
    dst.copy_(src)


if __name__ == "__main__":
    args = launch_args()

    if not args.no_codegen:
        print("Generating kernel code")
        run_test_codegen(
            gen_test_kernel(),
            "pipelined_matrix_copy_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    copy_kernel = import_codegen("pipelined_matrix_copy_test", dir="tests", args=args)
    print("Kernel compiled and imported, executing")

    # Create input and output matrices
    src_matrix = torch.rand(M * N, dtype=torch.float32)
    dst_kernel = torch.zeros(M * N, dtype=torch.float32)

    # Test pipeline copy
    if args.test:
        print("Testing pipelined matrix copy")
        # Run kernel implementation
        copy_kernel(src_matrix, dst_kernel)
        torch.cuda.synchronize()

        # Run reference implementation
        dst_ref = torch.zeros(M * N, dtype=torch.float32)
        reference_copy(src_matrix, dst_ref)

        # Compare results
        print(f"src_matrix: {src_matrix}")
        print(f"dst_kernel: {dst_kernel}")
        print(f"dst_ref: {dst_ref}")
        print(
            f"Maximum absolute difference: {torch.max(torch.abs(dst_kernel - dst_ref)).item()}"
        )
        assert torch.allclose(
            dst_kernel, dst_ref, rtol=1e-5, atol=1e-5
        ), "Outputs don't match!"
        print("Test passed!")

    # Benchmark pipeline copy
    if args.benchmark:
        print("Benchmarking pipelined matrix copy")
        times_ref = []
        times_kernel = []

        # Create test matrices
        n_matrices = args.n_loop
        src_matrices = [
            torch.rand(M * N, dtype=torch.float32) for _ in range(n_matrices)
        ]
        dst_kernel_matrices = [
            torch.zeros(M * N, dtype=torch.float32) for _ in range(n_matrices)
        ]
        dst_ref_matrices = [
            torch.zeros(M * N, dtype=torch.float32) for _ in range(n_matrices)
        ]

        # Warmup and benchmark
        for i in range(args.n_warmup + args.n_runs):
            torch.cuda.synchronize()

            # Benchmark reference implementation
            start_time = time.time()
            for j in range(n_matrices):
                reference_copy(src_matrices[j], dst_ref_matrices[j])
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_ref.append(end_time - start_time)

            # Benchmark kernel implementation
            torch.cuda.synchronize()
            start_time = time.time()
            for j in range(n_matrices):
                copy_kernel(src_matrices[j], dst_kernel_matrices[j])
            torch.cuda.synchronize()
            end_time = time.time()
            if i >= args.n_warmup:
                times_kernel.append(end_time - start_time)

        # Print benchmark results
        avg_time_ref = sum(times_ref) / len(times_ref) / n_matrices
        avg_time_kernel = sum(times_kernel) / len(times_kernel) / n_matrices

        print(f"Matrix size: {M}x{N}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Pipeline stages: {PIPELINE_STAGES}")
        print(f"Average reference time: {avg_time_ref:.6f}s per matrix")
        print(f"Average kernel time: {avg_time_kernel:.6f}s per matrix")
        print(f"Speedup: {avg_time_ref/avg_time_kernel:.2f}x")
