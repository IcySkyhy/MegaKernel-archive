# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Ascend port of mega_triton_kernel ModelBuilder (scoreboard / TaskBase path).
# Stubs CUDA/nvshmem at import time and subclasses the upstream builder to
# override device touchpoints: __init__, compile, run. Single-process only.
#
# Distinct from ``models_blade.ModelBuilder``: that route has no TaskBase /
# scoreboard; it registers named compute and emits op_id if/elif megakernel
# source for the Ascend blade/customOp pipeline.
import importlib.util
import os
import sys
import types


# ----------------------------------------------------------------------------
# import-time stubs (same technique as dump_qwen3_dsl.py): meta-path finders
# make nvshmem + triton_dist.models importable as no-op mocks so the real
# ModelBuilder module can be imported on NPU.
# ----------------------------------------------------------------------------
class _Mock:
    def __getattr__(self, _): return _Mock()
    def __call__(self, *a, **k): return _Mock()


class _StubLoader:
    def __init__(self, mod): self._mod = mod
    def create_module(self, spec): return self._mod
    def exec_module(self, module): return None


class _AttrMockModule(types.ModuleType):
    def __getattr__(self, name): return _Mock()


class _NVMetaFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "nvshmem" or fullname.startswith("nvshmem."):
            mod = _AttrMockModule(fullname)
            mod.__path__ = []
            if fullname == "nvshmem":
                class _Teams: TEAM_NODE = 0
                class _Core: Teams = _Teams
                class _Bindings:
                    @staticmethod
                    def mc_ptr(*a, **k): return 0
                mod.core = _Core()
                mod.bindings = _Bindings()
            spec = importlib.util.spec_from_loader(fullname, loader=None)
            spec.loader = _StubLoader(mod)
            return spec
        return None


_STUB_PREFIXES = (
    "triton_dist.models",
    "triton_dist.layers.nvidia",
    "triton_dist.layers.amd",
    "triton_dist.kernels.nvidia",
    "triton_dist.kernels.amd",
    "triton_dist.kernels.common_ops",
)


class _StubFinder:
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in _STUB_PREFIXES):
            mod = _AttrMockModule(fullname)
            mod.__path__ = []
            spec = importlib.util.spec_from_loader(fullname, loader=None)
            spec.loader = _StubLoader(mod)
            return spec
        return None


# nvshmem_create_tensor / nvshmem_free_tensor_sync come from triton_dist.utils.
# triton_dist.utils exports nvshmem_* names; the single-process path never
# calls them, but the `from ... import nvshmem_*` line must resolve. Inject mocks.
import triton_dist.utils as _utils  # noqa: E402
if not hasattr(_utils, "nvshmem_create_tensor"):
    _utils.nvshmem_create_tensor = _Mock()
if not hasattr(_utils, "nvshmem_free_tensor_sync"):
    _utils.nvshmem_free_tensor_sync = _Mock()

# install finders before importing the upstream model_builder module
sys.meta_path.insert(0, _NVMetaFinder())
sys.meta_path.insert(0, _StubFinder())

import torch  # noqa: E402

# scheduler.py / model_builder.py call torch.cuda.current_device() etc. at runtime;
# point them at NPU so the unmodified scheduler resolves device tensors to NPU.
if not hasattr(torch.cuda, "_ascend_patched"):
    torch.cuda.is_available = lambda: False
    torch.cuda.current_device = lambda: torch.device("npu")
    torch.cuda.get_device_properties = lambda d: torch.npu.get_device_properties(0)
    torch.cuda.synchronize = lambda *a, **k: torch.npu.synchronize()
    torch.cuda._ascend_patched = True

# triton-ascend puts constexpr_function on `triton`, not `tl`/`tl.core`. Alias it
# onto both so the unmodified kernels/utils.py imports cleanly.
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
if not hasattr(tl, "constexpr_function"):
    tl.constexpr_function = triton.constexpr_function
if not hasattr(tl.core, "constexpr_function"):
    tl.core.constexpr_function = triton.constexpr_function

# triton_dist/jit.py installs a shmem post-compile hook that is incompatible with
# the ascend backend (uses knobs not imported on ascend + NPUOptions lacks an
# attribute the hook reads). The single-process path never uses shmem, so no-op
# the hook and return an empty extern-lib dict on ascend.
import triton_dist  # noqa: E402  (ensure parent package loaded)
import triton_dist.jit  # noqa: E402,F401  (force module into sys.modules)
_tdist_jit_mod = sys.modules["triton_dist.jit"]
_tdist_jit_mod._install_triton_dist_hook = lambda: None
_orig_get_shmem_extern_lib = _tdist_jit_mod.get_shmem_extern_lib
def _ascend_get_shmem_extern_lib():
    from triton_dist.utils import is_ascend as _is_asc
    if _is_asc():
        return {}
    return _orig_get_shmem_extern_lib()
_tdist_jit_mod.get_shmem_extern_lib = _ascend_get_shmem_extern_lib
from triton_dist.mega_triton_kernel.models.model_builder import (  # noqa: E402
    ModelBuilder as _CudaModelBuilder,
)
from triton_dist.mega_triton_kernel.core.scheduler import enque_tasks  # noqa: E402
from triton_dist.mega_triton_kernel.core.task_base import DeviceProp, MAX_NUM_TENSOR_DIMS  # noqa: E402


def _npu_device():
    """Current NPU device tensor allocator shortcut."""
    return torch.npu.current_device()


class ModelBuilder(_CudaModelBuilder):
    """NPU ModelBuilder: overrides __init__ / compile / run / finalize."""

    def __init__(self, rank=0, world_size=1, local_world_size=1, num_warps=4,
                 enable_profiling=False, enable_dep_opt=True,
                 enable_runtime_scheduler=False):
        # Reproduce the upstream __init__ but swap NUM_SMS to the NPU property
        # (cube_core_num). super().__init__() can't be used: the upstream __init__
        # calls torch.cuda.get_device_properties("cuda").
        self.reset()
        from triton_dist.mega_triton_kernel.core.registry import registry as _reg
        from triton_dist.mega_triton_kernel.core.code_generator import CodeGenerator, CodeGenOptions
        from triton_dist.mega_triton_kernel.core.graph import Graph
        from triton_dist.mega_triton_kernel.core.task_base import TaskDependency
        self._registry = _reg
        self._code_generator = CodeGenerator()
        self._max_tensor_dim = MAX_NUM_TENSOR_DIMS
        props = torch.npu.get_device_properties(0)
        NUM_SMS = getattr(props, "cube_core_num", 8)
        self.device_prop = DeviceProp(NUM_SMS=NUM_SMS)
        self.megakernel_tasks = []
        self.scoreboard = None
        self.wq_tensor = None
        self.num_task_tensor = None
        self.MAX_NUM_TILES_PER_OP = 1
        self.last_dependency = TaskDependency()
        self.max_layer_id = 0
        self.max_task_id = 0
        self.num_warps = num_warps
        self._metrics = {"memory": 0}
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.rank = rank
        self.local_rank = self.rank % self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.world_size > 0 and self.local_world_size > 0
        self.all_symm_tensors = []
        # world_size==1 (single-process test) never touches nvshmem symm tensors
        self.barrier_all_intra_node_buf = None
        self.logger = _Mock()  # triton_dist.models.utils.logger is stubbed; silence logs
        self._enable_profiling = enable_profiling
        self._enable_dep_opt = enable_dep_opt
        self._enable_runtime_scheduler = enable_runtime_scheduler
        self._codegen_options = CodeGenOptions(enable_profiling=enable_profiling,
                                               enable_runtime_scheduler=enable_runtime_scheduler)
        self.task_types_to_str = None
        self._graph = Graph()
        self._dsl_src = None

    def compile(self):
        self.logger.log(f"num_total_tasks = {len(self.megakernel_tasks)}")
        megakernel_tasks = (self._graph.to_tasks() if self._enable_dep_opt
                            else self.megakernel_tasks)
        num_sms = 1 if self._enable_runtime_scheduler else self.device_prop.NUM_SMS
        # enque_tasks hardcodes torch.cuda.current_device(); the module-top patch
        # points it at NPU, so work-queue tensors land on NPU.
        self.wq_tensor, self.num_tasks_tensor, self.scoreboard, self.task_deps_tensor = enque_tasks(
            num_sms, megakernel_tasks, "round_robin",
            enable_dependency_opt=not self._enable_runtime_scheduler)
        self.scoreboard = torch.zeros(
            (self.max_layer_id + 1, self.max_task_id + 1, self.MAX_NUM_TILES_PER_OP),
            dtype=torch.int32, device=_npu_device())
        src, task_types_to_str = self._code_generator.generate_code(
            self.megakernel_tasks, self._codegen_options)
        self.task_types_to_str = task_types_to_str
        self._dsl_src = src
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(src.encode('utf-8'))
            tmp_path = tmp.name
        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._gen_kernel = module.MEGA_TRITON_KERNEL

    def run(self):
        grid = lambda META: (self.device_prop.NUM_SMS,)
        work_queue_start = torch.empty((1,), dtype=torch.int32, device=_npu_device())
        if self._enable_runtime_scheduler:
            work_queue_start.fill_(0)
        self._gen_kernel[grid](
            work_queue_start,
            self.wq_tensor,
            self.num_tasks_tensor,
            self.scoreboard,
            self.task_deps_tensor,
            INT_PER_DEPS=self.task_deps_tensor.shape[1],
            INT_PER_TASK=self.wq_tensor.shape[2],
            MAX_TASK_ID=self.scoreboard.shape[1],
            MAX_NUM_TILES_PER_OP=self.scoreboard.shape[2],
            MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
            NUM_SMS=self.device_prop.NUM_SMS,
            num_warps=self.num_warps,
            inject_block_all=True,
            multibuffer=False
        )
        self.scoreboard.zero_()

    def finalize(self):
        # nvshmem symm tensors never allocated in the single-process path
        self.all_symm_tensors = []
