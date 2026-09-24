# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Ascend-specific overrides for mega_triton_kernel/kernels/task_context.py.
# The package __init__ loads the original (cuda) task_context module and apply patches.
import triton
import triton.language as tl
from triton_dist.mega_triton_kernel.kernels.task_context import TaskBaseInfo


# TODO: Fix the cast issue and use the impl of TensorDesc in mega_triton_kernel directly.
@triton.jit
def TensorDesc_data_ptr(self, dtype):
    # base_ptr is uint32*. ConvertTritonIRToLinalgIR pass rejects a 32->64-bit pointer cast.
    # The error is "Casting pointers with unmatched bitwidth"
    # So load the two halves and combine into a uint64, then cast to the pointer type.
    low = tl.load(self.base_ptr).to(tl.uint64)
    high = tl.load(self.base_ptr + 1).to(tl.uint64)
    data_ptr_val = low | (high << 32)
    data_ptr = data_ptr_val.to(tl.pointer_type(dtype))
    data_ptr = tl.multiple_of(data_ptr, 16)
    return data_ptr


@triton.jit
def Scoreboard_wait_deps(self, task_base_info: TaskBaseInfo):
    # Spin on each required signal slot until ready.
    entry_start = task_base_info.depend_entry_start
    entry_end = task_base_info.depend_entry_end
    for t in range(entry_start, entry_end):
        l = tl.load(self.task_deps_ptr + t * self.INT_PER_DEPS + 0)
        r = tl.load(self.task_deps_ptr + t * self.INT_PER_DEPS + 1)
        num_signals = r - l
        sb_wait_base_ptr = self.scoreboard_table + l
        for i in range(num_signals):
            while tl.load(sb_wait_base_ptr + i) != self.TILE_READY_SIGNAL:
                pass


@triton.jit
def Scoreboard_release_tile(self, task_base_info: TaskBaseInfo, tile_id):
    sb_set_base_ptr = self.task_scoredboard_start(task_base_info)
    # Ensure all previous computations are completed before notifying subsequent tasks.
    tl.debug_barrier()
    tl.store(sb_set_base_ptr + tile_id, self.TILE_READY_SIGNAL)


# Load the original task_context and swap in the ascend overrides. Called from
# __init__.py after the original module is available.
def _apply_patches(orig_tc):
    orig_tc.TensorDesc.data_ptr = TensorDesc_data_ptr
    orig_tc.Scoreboard.wait_deps = Scoreboard_wait_deps
    orig_tc.Scoreboard.release_tile = Scoreboard_release_tile

    # TODO: Fix the hash_attrs setting in triton-ascend and can remove these code.
    # triton-ascend's @tl.core._aggregate puts the plain-Python __init__ into
    # hash_attrs; DependenciesFinder rejects non-JIT callables referenced from
    # a kernel ("Unsupported function referenced").
    from triton.runtime.jit import JITCallable
    for _agg in (orig_tc.Scoreboard, orig_tc.TaskBaseInfo, orig_tc.TensorDesc):
        _agg.hash_attrs = [a for a in _agg.hash_attrs if isinstance(a, JITCallable)]
    return orig_tc
