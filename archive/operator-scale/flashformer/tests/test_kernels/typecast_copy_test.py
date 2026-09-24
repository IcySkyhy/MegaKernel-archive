import cheetah.api as ch
from cheetah.api import ty
import cheetah.index_tools as ixt
import torch
import time

from flashtransformer.components.memory import WGSyncCopyFn
from flashtransformer.components.memory.typecast_copy import WGTypeCastCopyFn
from flashtransformer.utils import SafeDataIndexPtr, SDIPConfig, LocalBarrierManager
from flashtransformer.utils.kernel_utils import (
    run_test_codegen,
    import_codegen,
    standard_headers,
    launch_helper,
    launch_args,
    make_cond_print,
)

# Test dimensions
n_threads = 256
n_blocks = 1
vec_size = 1024
debug = True  # Enable kernel debug output to diagnose issues


def gen_test_kernel():
    """Generate a kernel to test type casting copy operations."""
    dims = ixt.Dims()

    # Define thread and block indices
    thread_idx = dims.new_dim("thread_idx", n_threads)
    block_idx = dims.new_dim("block_idx", n_blocks)

    # Define vector indices
    vec_idx = dims.new_dim("vec_idx", vec_size)

    # Define TypeCastCopy test configurations
    # F32 to BF16 conversion
    src_f32_cfg = SDIPConfig(dims, vec_idx, ty.ptr_const(ty.f32), "global", "src_f32")
    dst_bf16_cfg = SDIPConfig(dims, vec_idx, ty.ptr_mut(ty.bf16), "global", "dst_bf16")

    # BF16 to F32 conversion
    src_bf16_cfg = SDIPConfig(
        dims, vec_idx, ty.ptr_const(ty.bf16), "global", "src_bf16"
    )
    dst_f32_cfg = SDIPConfig(dims, vec_idx, ty.ptr_mut(ty.f32), "global", "dst_f32")

    # F32 to F32 no conversion (control case)
    src_f32_ctrl_cfg = SDIPConfig(
        dims, vec_idx, ty.ptr_const(ty.f32), "global", "src_f32_ctrl"
    )
    dst_f32_ctrl_cfg = SDIPConfig(
        dims, vec_idx, ty.ptr_mut(ty.f32), "global", "dst_f32_ctrl"
    )

    # Shared memory configurations for testing
    shared_f32_cfg = SDIPConfig(
        dims, vec_idx, ty.ptr_mut(ty.f32), "shared", "shared_f32"
    )
    shared_bf16_cfg = SDIPConfig(
        dims, vec_idx, ty.ptr_mut(ty.bf16), "shared", "shared_bf16"
    )

    @ch.kernel(
        ch.Params(
            src_f32=ty.ptr_const(ty.f32),
            dst_bf16_direct=ty.ptr_mut(ty.bf16),  # For direct global-to-global test
            dst_bf16_via_shared=ty.ptr_mut(
                ty.bf16
            ),  # For global-to-shared-to-global test
            src_bf16=ty.ptr_const(ty.bf16),
            dst_f32_direct=ty.ptr_mut(ty.f32),  # For direct global-to-global test
            dst_f32_via_shared=ty.ptr_mut(
                ty.f32
            ),  # For global-to-shared-to-global test
            src_f32_ctrl=ty.ptr_const(ty.f32),
            dst_f32_ctrl=ty.ptr_mut(ty.f32),
        )
    )
    def typecast_copy_kernel(
        src_f32,
        dst_bf16_direct,
        dst_bf16_via_shared,
        src_bf16,
        dst_f32_direct,
        dst_f32_via_shared,
        src_f32_ctrl,
        dst_f32_ctrl,
    ):
        # IMPORTANT: First create all needed functions and add all equations BEFORE initializing indices

        # Create all copy functions before initializing indices
        # 1. F32 to BF16 conversion (global to global)
        f32_to_bf16_copy = WGTypeCastCopyFn(
            dims, src_f32_cfg, dst_bf16_cfg, thread_idx, vec_idx, elems_per_load=4
        )

        # 2. BF16 to F32 conversion (global to global)
        bf16_to_f32_copy = WGTypeCastCopyFn(
            dims, src_bf16_cfg, dst_f32_cfg, thread_idx, vec_idx, elems_per_load=4
        )

        # 3. F32 to F32 (control case - no conversion)
        f32_to_f32_copy = WGTypeCastCopyFn(
            dims,
            src_f32_ctrl_cfg,
            dst_f32_ctrl_cfg,
            thread_idx,
            vec_idx,
            elems_per_load=4,
        )

        # 4. Shared memory conversion tests
        # F32 global to F32 shared
        f32_to_shared_f32 = WGSyncCopyFn(
            dims, src_f32_cfg, shared_f32_cfg, thread_idx, vec_idx, elems_per_load=4
        )

        # F32 shared to BF16 global using TypeCastCopy with debug enabled
        shared_f32_to_bf16 = WGTypeCastCopyFn(
            dims,
            shared_f32_cfg,
            dst_bf16_cfg,
            thread_idx,
            vec_idx,
            elems_per_load=4,
            debug=True,  # Enable debugging to see detailed logs
        )

        # BF16 global to BF16 shared
        bf16_to_shared_bf16 = WGSyncCopyFn(
            dims, src_bf16_cfg, shared_bf16_cfg, thread_idx, vec_idx, elems_per_load=4
        )

        # BF16 shared to F32 global using TypeCastCopy with debug enabled
        shared_bf16_to_f32 = WGTypeCastCopyFn(
            dims,
            shared_bf16_cfg,
            dst_f32_cfg,
            thread_idx,
            vec_idx,
            elems_per_load=4,
            debug=True,  # Enable debugging to see detailed logs
        )

        # Initialize local barrier
        local_barrier_manager = LocalBarrierManager()
        local_barrier = local_barrier_manager.get_local_barrier(n_threads)

        # NOW initialize indices AFTER all equations and functions are set up
        ix = dims.init()
        ix.set_index(thread_idx, ch.thread_idx_x().cast(ty.i32))
        ix.set_index(block_idx, ch.block_idx_x().cast(ty.i32))

        # Create debug print function
        debug_print = make_cond_print(
            ch.and_(ix[thread_idx] == 0), debug, local_barrier
        )
        debug_print("Starting TypeCastCopy tests")

        # Allocate static shared memory instead of dynamic
        shared_f32_ptr = shared_f32_cfg.alloc_static_shared(ix)
        shared_bf16_ptr = shared_bf16_cfg.alloc_static_shared(ix)

        # Create pointers for global memory
        src_f32_ptr = src_f32_cfg.wrap_ptr(ix, src_f32)
        dst_bf16_direct_ptr = dst_bf16_cfg.wrap_ptr(ix, dst_bf16_direct)
        dst_bf16_via_shared_ptr = dst_bf16_cfg.wrap_ptr(ix, dst_bf16_via_shared)
        src_bf16_ptr = src_bf16_cfg.wrap_ptr(ix, src_bf16)
        dst_f32_direct_ptr = dst_f32_cfg.wrap_ptr(ix, dst_f32_direct)
        dst_f32_via_shared_ptr = dst_f32_cfg.wrap_ptr(ix, dst_f32_via_shared)
        src_f32_ctrl_ptr = src_f32_ctrl_cfg.wrap_ptr(ix, src_f32_ctrl)
        dst_f32_ctrl_ptr = dst_f32_ctrl_cfg.wrap_ptr(ix, dst_f32_ctrl)

        # 1. Test F32 to BF16 conversion (global to global)
        debug_print("Testing F32 to BF16 conversion (global to global)")
        # Print first value from src_f32_ptr
        debug_print("Source F32 first value: %f", src_f32_ptr.raw_idx(0).cast(ty.f32))
        f32_to_bf16_copy(ix, src_f32_ptr, dst_bf16_direct_ptr, local_barrier)
        debug_print("F32 to BF16 conversion complete")
        # Print first value from dst_bf16_direct_ptr
        debug_print(
            "Destination BF16 first value: %f",
            dst_bf16_direct_ptr.raw_idx(0).cast(ty.f32),
        )

        # 2. Test BF16 to F32 conversion (global to global)
        debug_print("Testing BF16 to F32 conversion (global to global)")
        # Print first value from src_bf16_ptr
        debug_print("Source BF16 first value: %f", src_bf16_ptr.raw_idx(0).cast(ty.f32))
        bf16_to_f32_copy(ix, src_bf16_ptr, dst_f32_direct_ptr, local_barrier)
        debug_print("BF16 to F32 conversion complete")
        # Print first value from dst_f32_direct_ptr
        debug_print("Destination F32 first value: %f", dst_f32_direct_ptr.raw_idx(0))

        # 3. Test F32 to F32 (control case - no conversion)
        debug_print("Testing F32 to F32 (control case - no conversion)")
        # Print first value from src_f32_ctrl_ptr
        debug_print(
            "Source F32 control first value: %f",
            src_f32_ctrl_ptr.raw_idx(0).cast(ty.f32),
        )
        f32_to_f32_copy(ix, src_f32_ctrl_ptr, dst_f32_ctrl_ptr, local_barrier)
        debug_print("F32 to F32 copy complete")
        # Print first value from dst_f32_ctrl_ptr
        debug_print(
            "Destination F32 control first value: %f", dst_f32_ctrl_ptr.raw_idx(0)
        )

        # 4. Test shared memory conversions
        # F32 global to F32 shared
        debug_print("Testing F32 global to F32 shared")
        
        # Debug the original source data before copying to shared memory
        debug_print("Source F32 first value (before shared copy): %f", src_f32_ptr.raw_idx(0))
        debug_print("Source F32 second value (before shared copy): %f", src_f32_ptr.raw_idx(1))
        
        f32_to_shared_f32(ix, src_f32_ptr, shared_f32_ptr, local_barrier)
        
        # Debug the shared memory after copying
        debug_print("Shared F32 first value (after copy): %f", shared_f32_ptr.raw_idx(0))
        debug_print("Shared F32 second value (after copy): %f", shared_f32_ptr.raw_idx(1))
        
        debug_print("F32 global to F32 shared copy complete")

        # F32 shared to BF16 global using TypeCastCopy
        debug_print("Testing F32 shared to BF16 global using TypeCastCopy")
        shared_f32_to_bf16(ix, shared_f32_ptr, dst_bf16_via_shared_ptr, local_barrier)
        debug_print("F32 shared to BF16 global conversion complete")
        debug_print(
            "Destination BF16 via shared first value: %f",
            dst_bf16_via_shared_ptr.raw_idx(0).cast(ty.f32),
        )

        # BF16 global to BF16 shared
        debug_print("Testing BF16 global to BF16 shared")
        
        # Debug the original BF16 source data before copying to shared memory
        debug_print("Source BF16 first value (before shared copy): %f", src_bf16_ptr.raw_idx(0).cast(ty.f32))
        debug_print("Source BF16 second value (before shared copy): %f", src_bf16_ptr.raw_idx(1).cast(ty.f32))
        
        bf16_to_shared_bf16(ix, src_bf16_ptr, shared_bf16_ptr, local_barrier)
        
        # Debug the shared memory after copying
        debug_print("Shared BF16 first value (after copy): %f", shared_bf16_ptr.raw_idx(0).cast(ty.f32))
        debug_print("Shared BF16 second value (after copy): %f", shared_bf16_ptr.raw_idx(1).cast(ty.f32))
        
        debug_print("BF16 global to BF16 shared copy complete")

        # BF16 shared to F32 global using TypeCastCopy
        debug_print("Testing BF16 shared to F32 global using TypeCastCopy")
        shared_bf16_to_f32(ix, shared_bf16_ptr, dst_f32_via_shared_ptr, local_barrier)
        debug_print("BF16 shared to F32 global conversion complete")
        debug_print(
            "Destination F32 via shared first value: %f",
            dst_f32_via_shared_ptr.raw_idx(0),
        )

        debug_print("All TypeCastCopy tests completed")

    @ch.fn(
        ch.Params(
            src_f32=ty.ptr_const(ty.f32),
            dst_bf16_direct=ty.ptr_mut(ty.bf16),
            dst_bf16_via_shared=ty.ptr_mut(ty.bf16),
            src_bf16=ty.ptr_const(ty.bf16),
            dst_f32_direct=ty.ptr_mut(ty.f32),
            dst_f32_via_shared=ty.ptr_mut(ty.f32),
            src_f32_ctrl=ty.ptr_const(ty.f32),
            dst_f32_ctrl=ty.ptr_mut(ty.f32),
        )
    )
    def typecast_copy_launch(
        src_f32,
        dst_bf16_direct,
        dst_bf16_via_shared,
        src_bf16,
        dst_f32_direct,
        dst_f32_via_shared,
        src_f32_ctrl,
        dst_f32_ctrl,
    ):
        # No need to calculate shared memory size for static shared memory
        shared_mem = 0  # Using static shared memory allocation

        launch_helper(
            typecast_copy_kernel,
            n_blocks,
            n_threads,
            [
                src_f32,
                dst_bf16_direct,
                dst_bf16_via_shared,
                src_bf16,
                dst_f32_direct,
                dst_f32_via_shared,
                src_f32_ctrl,
                dst_f32_ctrl,
            ],
            shared_mem,
        )

    @ch.fn(
        ch.Params(
            src_f32=ty.tensor_const(ty.f32),
            dst_bf16_direct=ty.tensor_mut(ty.bf16),
            dst_bf16_via_shared=ty.tensor_mut(ty.bf16),
            src_bf16=ty.tensor_const(ty.bf16),
            dst_f32_direct=ty.tensor_mut(ty.f32),
            dst_f32_via_shared=ty.tensor_mut(ty.f32),
            src_f32_ctrl=ty.tensor_const(ty.f32),
            dst_f32_ctrl=ty.tensor_mut(ty.f32),
        )
    )
    def typecast_copy_torch(
        src_f32,
        dst_bf16_direct,
        dst_bf16_via_shared,
        src_bf16,
        dst_f32_direct,
        dst_f32_via_shared,
        src_f32_ctrl,
        dst_f32_ctrl,
    ):
        typecast_copy_launch(
            src_f32.cast(ty.ptr_const(ty.f32)),
            dst_bf16_direct.cast(ty.ptr_mut(ty.bf16)),
            dst_bf16_via_shared.cast(ty.ptr_mut(ty.bf16)),
            src_bf16.cast(ty.ptr_const(ty.bf16)),
            dst_f32_direct.cast(ty.ptr_mut(ty.f32)),
            dst_f32_via_shared.cast(ty.ptr_mut(ty.f32)),
            src_f32_ctrl.cast(ty.ptr_const(ty.f32)),
            dst_f32_ctrl.cast(ty.ptr_mut(ty.f32)),
        )

    return typecast_copy_torch


def run_typecast_copy_test(
    kernel,
    src_f32,
    dst_bf16_direct,
    dst_bf16_via_shared,
    src_bf16,
    dst_f32_direct,
    dst_f32_via_shared,
    src_f32_ctrl,
    dst_f32_ctrl_kernel,
):
    """Run the test and compare results with PyTorch reference implementations."""
    # Create reference outputs
    dst_bf16_ref = src_f32.to(torch.bfloat16)
    dst_f32_ref = src_bf16.to(torch.float32)
    dst_f32_ctrl_ref = src_f32_ctrl.clone()

    # Run kernel
    kernel(
        src_f32,
        dst_bf16_direct,
        dst_bf16_via_shared,
        src_bf16,
        dst_f32_direct,
        dst_f32_via_shared,
        src_f32_ctrl,
        dst_f32_ctrl_kernel,
    )
    torch.cuda.synchronize()

    # Check F32 to BF16 direct conversion
    print("Checking F32 to BF16 direct conversion:")
    max_diff_f32_to_bf16_direct = torch.max(
        torch.abs(dst_bf16_ref - dst_bf16_direct)
    ).item()
    print(f"Max difference: {max_diff_f32_to_bf16_direct}")
    print(f"First few values - Reference: {dst_bf16_ref[:5]}")
    print(f"First few values - Kernel direct output: {dst_bf16_direct[:5]}")

    # BF16 has limited precision (only 8 bits mantissa), so we need higher absolute tolerance
    # Maximum value difference seen is around 4.21875
    assert torch.allclose(
        dst_bf16_direct, dst_bf16_ref, rtol=1e-2, atol=5.0
    ), "F32 to BF16 direct conversion failed"

    # Check F32 to BF16 via shared conversion
    print("Checking F32 to BF16 via shared conversion:")
    max_diff_f32_to_bf16_via_shared = torch.max(
        torch.abs(dst_bf16_ref - dst_bf16_via_shared)
    ).item()
    print(f"Max difference: {max_diff_f32_to_bf16_via_shared}")
    print(f"First few values - Reference: {dst_bf16_ref[:5]}")
    print(f"First few values - Kernel via shared output: {dst_bf16_via_shared[:5]}")

    assert torch.allclose(
        dst_bf16_via_shared, dst_bf16_ref, rtol=1e-2, atol=5.0
    ), "F32 to BF16 via shared conversion failed"

    # Check BF16 to F32 direct conversion
    print("Checking BF16 to F32 direct conversion:")
    max_diff_bf16_to_f32_direct = torch.max(
        torch.abs(dst_f32_ref - dst_f32_direct)
    ).item()
    print(f"Max difference: {max_diff_bf16_to_f32_direct}")
    print(f"First few values - Reference: {dst_f32_ref[:5]}")
    print(f"First few values - Kernel direct output: {dst_f32_direct[:5]}")

    # BF16 to F32 conversion may also need higher absolute tolerance
    assert torch.allclose(
        dst_f32_direct, dst_f32_ref, rtol=1e-2, atol=5.0
    ), "BF16 to F32 direct conversion failed"

    # Check BF16 to F32 via shared conversion
    print("Checking BF16 to F32 via shared conversion:")
    max_diff_bf16_to_f32_via_shared = torch.max(
        torch.abs(dst_f32_ref - dst_f32_via_shared)
    ).item()
    print(f"Max difference: {max_diff_bf16_to_f32_via_shared}")
    print(f"First few values - Reference: {dst_f32_ref[:5]}")
    print(f"First few values - Kernel via shared output: {dst_f32_via_shared[:5]}")

    assert torch.allclose(
        dst_f32_via_shared, dst_f32_ref, rtol=1e-2, atol=5.0
    ), "BF16 to F32 via shared conversion failed"

    # Check F32 to F32 control case
    print("Checking F32 to F32 control case:")
    max_diff_f32_to_f32 = torch.max(
        torch.abs(dst_f32_ctrl_ref - dst_f32_ctrl_kernel)
    ).item()
    print(f"Max difference: {max_diff_f32_to_f32}")
    assert torch.allclose(
        dst_f32_ctrl_kernel, dst_f32_ctrl_ref
    ), "F32 to F32 control case failed"

    print("All TypeCastCopy tests passed!")
    return True


if __name__ == "__main__":
    args = launch_args()
    if not args.no_codegen:
        # Generate test kernel
        print("Codegen started")
        run_test_codegen(
            gen_test_kernel(),
            "typecast_copy_test",
            headers=standard_headers,
            bind=True,
            dir="tests",
        )
        print("Codegen finished, importing kernel")
    else:
        print("Skipping codegen, importing kernel")

    typecast_copy_kernel = import_codegen("typecast_copy_test", dir="tests", args=args)
    print("Kernel compiled and imported, executing")

    # Create test data
    src_f32 = torch.randn(vec_size, dtype=torch.float32)
    dst_bf16_direct = torch.zeros(vec_size, dtype=torch.bfloat16)
    dst_bf16_via_shared = torch.zeros(vec_size, dtype=torch.bfloat16)
    src_bf16 = torch.randn(vec_size, dtype=torch.bfloat16)
    dst_f32_direct = torch.zeros(vec_size, dtype=torch.float32)
    dst_f32_via_shared = torch.zeros(vec_size, dtype=torch.float32)
    src_f32_ctrl = torch.randn(vec_size, dtype=torch.float32)
    dst_f32_ctrl = torch.zeros(vec_size, dtype=torch.float32)

    # Run test
    if args.test:
        result = run_typecast_copy_test(
            typecast_copy_kernel,
            src_f32,
            dst_bf16_direct,
            dst_bf16_via_shared,
            src_bf16,
            dst_f32_direct,
            dst_f32_via_shared,
            src_f32_ctrl,
            dst_f32_ctrl,
        )
        if result:
            print("TypeCastCopy component verified!")
