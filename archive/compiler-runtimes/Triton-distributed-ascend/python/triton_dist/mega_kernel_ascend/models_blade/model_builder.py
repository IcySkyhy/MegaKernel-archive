# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade ModelBuilder: TD-compatible make_* surface without TaskBase/scoreboard.
#
# Relationship to ``mega_kernel_ascend.models.ModelBuilder``:
# - ``models``: scoreboard / TaskBase NPU port.
# - ``models_blade`` (this file): make_* appends a Graph node (op_id = node
#   index) and maps it to a named compute; compile() emits if/elif megakernel
#   source. Shares Graph / OpGraph helpers only.
#
# User scripts can call the same make_* / compile / run sequence; tensors are
# auto-declared for OpGraph unless explicitly declare_tensor()'d.
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set
import os

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
import tempfile
import importlib.util
from types import SimpleNamespace
from triton.compiler import ASTSource

from . import (
        SchemaBuilder, dim, all,
        ShapeDimExpr,
        launchMegakernel,
    )

from triton_dist.mega_kernel_ascend.kernels.task_context_blade import (
    Context, Scoreboard,
    STAGE_START, STAGE_AICPU_HS, STAGE_READY, STAGE_START_LOOP,
    STAGE_QUIT, STAGE_QUIT_ACK,
    QUIT_OPCODE, TASK_RING_SIZE,
)
from triton_dist.mega_kernel_ascend.core.code_generator import CodeGenerator
from triton_dist.mega_kernel_ascend.models_blade.op_registry import OP_REGISTRY, get_op_spec
from triton_dist.mega_triton_kernel.core.graph import Graph
from triton_dist.mega_triton_kernel.core.op_graph import MemoryPool
from .compiler import mega_target


TRITON_FLAG = 0x100

_BLADE_KERNEL_TYPE = {"aiv": 1, "mix": 2, "aic": 0}

_KERNEL_TYPE_MAP = {
    "add": 1 | TRITON_FLAG,
    "matmul": 0 | TRITON_FLAG,
    "linear": 0 | TRITON_FLAG,
    "rmsnorm": 1 | TRITON_FLAG,
    "silu_mul_up": 1 | TRITON_FLAG,
    "flash_attn": 0 | TRITON_FLAG,
}


class ModelBuilder:
    """Registers ops via make_*; compile() builds op_id -> compute dispatch.

    Constructor accepts the same keyword args as the scoreboard builder; blade
    ignores scheduling/profiling flags that do not apply here.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        local_world_size: int = 1,
        num_warps: int = 4,
        enable_profiling: bool = False,
        enable_dep_opt: bool = True,
        enable_runtime_scheduler: bool = False,
        auto_declare: bool = True,
        platform: Optional[str] = None,
    ):
        del rank, world_size, local_world_size
        del enable_profiling, enable_dep_opt, enable_runtime_scheduler
        self.num_warps = num_warps
        self._auto_declare = auto_declare
        # Optional default Matmul wire name override (e.g. "950" -> matmulA5).
        self._platform = platform
        self._code_generator = CodeGenerator()
        self._op_table: Dict[int, Callable] = {}
        self._dsl_src: Optional[str] = None
        self._graph = Graph()
        self._declared: Set[int] = set()

    # ------------------------------------------------------------------ helpers
    def declare_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        pool_hint: int = 0,
        external_addr: int = 0,
    ) -> None:
        """Pin OpGraph wire metadata for ``tensor`` (name / pool / addr)."""
        self._graph.declare_tensor(
            tensor, name, pool_hint=pool_hint, external_addr=external_addr
        )
        self._declared.add(id(tensor))

    def _ensure_declared(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        pool_hint: int,
    ) -> None:
        if not self._auto_declare:
            return
        if id(tensor) in self._declared:
            return
        self.declare_tensor(tensor, name, pool_hint=pool_hint, external_addr=0)

    def _declare_io(
        self,
        op_type: str,
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        *,
        weight_input_indices: Optional[Set[int]] = None,
    ) -> None:
        weight_input_indices = weight_input_indices or set()
        for i, t in enumerate(inputs):
            pool = (
                int(MemoryPool.Persistent)
                if i in weight_input_indices
                else int(MemoryPool.Recyclable)
            )
            self._ensure_declared(t, name=f"{op_type}.in{i}", pool_hint=pool)
        for i, t in enumerate(outputs):
            self._ensure_declared(
                t, name=f"{op_type}.out{i}", pool_hint=int(MemoryPool.Recyclable)
            )

    def _convert_op(
        self,
        op_type: str,
        io_tensors,
        *,
        wire_op_name: Optional[str] = None,
        weight_input_indices: Optional[Set[int]] = None,
    ) -> int:
        """Append a Graph node and map its index to the registered compute."""
        spec = get_op_spec(op_type)
        graph_op_type = wire_op_name or spec.wire_op_name or op_type
        inputs, outputs = io_tensors
        self._declare_io(
            op_type,
            list(inputs),
            list(outputs),
            weight_input_indices=weight_input_indices,
        )
        self._graph.new_node(
            tasks=[],
            op_type=graph_op_type,
            io_tensors=io_tensors,
            extra_params={
                "opgraph_op_name": graph_op_type,
                "opgraph_omit_attrs": spec.opgraph_omit_attrs,
                "opgraph_omit_deps": spec.opgraph_omit_deps,
            },
        )
        op_id = len(self._graph._nodes) - 1
        self._op_table[op_id] = spec.compute
        return op_id

    def _default_matmul_wire_name(self, op_name: Optional[str] = None) -> str:
        if op_name is not None:
            return op_name
        if self._platform in ("950", "A5"):
            return "matmulA5"
        return "Matmul"

    # ---------------------------------------------------------------- make_* API
    def make_add(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        output: torch.Tensor,
        layer_id: int = 0,
        block: int = 256,
        op_name: str = "add",
    ) -> int:
        del layer_id, block
        assert lhs.shape == rhs.shape == output.shape
        return self._convert_op(
            "add",
            [[lhs, rhs], [output]],
            wire_op_name=op_name,
        )

    def make_matmul(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        layer_id: int = 0,
        op_name: Optional[str] = None,
    ) -> int:
        del layer_id
        assert a.ndim == 2 and b.ndim == 2 and output.ndim == 2
        assert a.shape[1] == b.shape[0]
        assert output.shape[0] == a.shape[0] and output.shape[1] == b.shape[1]
        return self._convert_op(
            "Matmul",
            [[a, b], [output]],
            wire_op_name=self._default_matmul_wire_name(op_name),
            weight_input_indices={1},
        )

    def make_linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        layer_id: int = 0,
    ) -> int:
        del layer_id
        assert input.ndim == 2 and weight.ndim == 2 and output.ndim == 2
        M, K = input.shape
        N, wK = weight.shape
        assert K == wK and output.shape == (M, N)
        return self._register_op(
            "linear",
            [[input, weight], [output]],
            wire_op_name=self._default_matmul_wire_name(),
            weight_input_indices={1},
        )

    def _make_fc(
        self,
        op_type: str,
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        layer_id: int = 0,
    ) -> int:
        del layer_id
        assert input.ndim == 2 and weight.ndim == 2 and output.ndim == 2
        M, K = input.shape
        N, wK = weight.shape
        assert K == wK and output.shape == (M, N)
        return self._register_op(
            op_type,
            [[input, weight], [output]],
            wire_op_name=self._default_matmul_wire_name(),
            weight_input_indices={1},
        )

    def make_fc1(self, input, weight, output, layer_id: int = 0) -> int:
        return self._make_fc("mlp_fc1", input, weight, output, layer_id)

    def make_fc2(self, input, weight, output, layer_id: int = 0) -> int:
        return self._make_fc("mlp_fc2", input, weight, output, layer_id)

    def make_qkv_proj(self, input, weight, output, layer_id: int = 0) -> int:
        return self._make_fc("qkv_proj", input, weight, output, layer_id)

    def make_o_proj(self, input, weight, output, layer_id: int = 0) -> int:
        return self._make_fc("o_proj", input, weight, output, layer_id)

    def make_silu_mul_up(self, fc1_out, act_out, layer_id: int = 0) -> int:
        del layer_id
        assert fc1_out.ndim == 2 and act_out.ndim == 2
        M, N = fc1_out.shape
        assert act_out.shape[0] == M and act_out.shape[1] * 2 == N
        return self._register_op("silu_mul_up", [[fc1_out], [act_out]])

    def make_add_rms_norm(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        gamma: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        rstd: torch.Tensor,
        layer_id: int = 0,
        op_name: str = "add_rms_norm",
    ) -> int:
        """Fused add + rms_norm matching Ascend ``StaticOpResult<3>``.

        Outputs (must match Ascend ``ops::add_rms_norm`` port names/shapes):
          [0] ``y``    — rms_norm(x1+x2), same shape as x1  ``[M, H]``
          [1] ``x``    — add result x1+x2, same shape as x1 ``[M, H]``
          [2] ``rstd`` — ``[M, 1]`` (not ``[M]``)
        """
        del layer_id
        assert x1.shape == x2.shape == y.shape == x.shape
        assert gamma.ndim == 1 and gamma.shape[0] == x1.shape[-1]
        assert rstd.ndim == 2 and rstd.shape[0] == x1.shape[0] and rstd.shape[1] == 1
        return self._register_op(
            "add_rms_norm",
            [[x1, x2, gamma], [y, x, rstd]],
            wire_op_name=op_name,
            weight_input_indices={2},
        )

    def make_matmul_lmhead(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
        layer_id: int = 0,
        op_name: str = "matmul_lmhead",
    ) -> int:
        """LM head matmul: ``output = a @ b`` with ``b`` layout ``[H, V/ranks]``.

        Matches Ascend ``ops::matmul_lmhead(normOutput, lmHeadWeight)``.
        """
        del layer_id
        assert a.ndim == 2 and b.ndim == 2 and output.ndim == 2
        assert a.shape[1] == b.shape[0]
        assert output.shape == (a.shape[0], b.shape[1])
        return self._register_op(
            "matmul_lmhead",
            [[a, b], [output]],
            wire_op_name=op_name,
            weight_input_indices={1},
        )

    def build_lm_head(
        self,
        mlp_output: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        rstd: torch.Tensor,
        logits: torch.Tensor,
        *,
        ranks: int = 1,
    ) -> tuple[int, int]:
        """Register Ascend-shaped final_norm + lm_head subgraph (ranks==1).

        ``add_rms_norm`` exports Ascend ``StaticOpResult<3>`` as ``y``, ``x``,
        ``rstd`` (non-empty shapes). ``matmul_lmhead`` consumes ``y``
        (= ``outputs[0]``). AllGather for ``ranks>1`` stays on Ascend after merge.
        """
        if ranks != 1:
            raise NotImplementedError(
                "blade OpGraph export for lm_head AllGather (ranks>1) is not ready"
            )
        id0 = self.make_add_rms_norm(
            mlp_output, residual, norm_weight, y, x, rstd
        )
        id1 = self.make_matmul_lmhead(y, lm_head_weight, logits)
        return id0, id1

    def make_flash_decode(self, *args, **kwargs):
        return self._unsupported("flash_decode")

    def make_qkv_pack_flash_attn(self, *args, **kwargs):
        return self._unsupported("qkv_pack_flash_attn")

    def make_flash_attn(self, *args, **kwargs):
        return self._unsupported("flash_attn")

    def make_qk_norm_rope_update_kvcache(self, *args, **kwargs):
        return self._unsupported("qk_norm_rope_update_kvcache")

    def make_qkv_pack_qk_norm_rope_split_v(self, *args, **kwargs):
        return self._unsupported("qkv_pack_qk_norm_rope_split_v")

    def make_rms_norm(self, *args, **kwargs):
        return self._unsupported("rms_norm")

    def make_barrier_all_intra_node(self, *args, **kwargs):
        return self._unsupported("barrier_all_intra_node")

    def make_allreduce(self, *args, **kwargs):
        return self._unsupported("allreduce")

    def make_prefetch(self, *args, **kwargs):
        return self._unsupported("prefetch")

    def _unsupported(self, name: str):
        supported = ", ".join(sorted(OP_REGISTRY))
        raise NotImplementedError(
            f"blade backend has no compute for {name!r}; supported: {supported}"
        )

    # ---------------------------------------------------------- compile / run
    def build_opgraph(self) -> bytes:
        from triton_dist.mega_triton_kernel.core.op_graph import (
            build_opgraph_from_graph,
            serialize_opgraph,
        )
        return serialize_opgraph(build_opgraph_from_graph(self._graph))

    def save_opgraph(self, path: str) -> None:
        from triton_dist.mega_triton_kernel.core.op_graph import (
            build_opgraph_from_graph,
            save_opgraph_to_file,
        )
        save_opgraph_to_file(build_opgraph_from_graph(self._graph), path)


    def compile(self):
        # 1. Generate + compile ONE megakernel
        self._dsl_src = self._code_generator.generate_code(self._op_table)
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(self._dsl_src.encode('utf-8'))
            tmp_path = tmp.name

        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for fn in self._op_table.values():
            setattr(module, fn.__name__, fn)

        kernel = module.kernel
        sig = {"ctxData": "i64"}
        compiled = triton.compile(ASTSource(fn=kernel, signature=sig),
                                  target=mega_target(),
                                  options={'inject_barrier_all': True})

        o_path = next(p for k, p in compiled.metadata_group.items() if k.endswith(".npubin"))
        print(f"[DEBUG] megakernel .o: {o_path}")
        self._megakernel_o_path = o_path
        self._blade_kernel_type = _BLADE_KERNEL_TYPE.get(compiled.metadata.mix_mode, 1)

    def run(self, callback_path=None, block_dim=24, aicpu_block_dim=1, device_id=0):
        mega_home = os.environ.get("MEGA_KERNEL_HOME", "")
        if callback_path is None:
            callback_path = os.path.join(mega_home, "libprint_callback.so") if mega_home \
                else "./libprint_callback.so"

        megakernel_o_path = getattr(self, '_megakernel_o_path', None)

        # Dump OpGraph bytes to a temp file (PR101 wire format).
        fd, graph_path = tempfile.mkstemp(suffix="_opgraph.bin")
        os.close(fd)
        self.save_opgraph(graph_path)
        print(f"[DEBUG] opgraph .bin: {graph_path}")

        output_id = 0  # TODO: fix once launchMegakernel reads graph_path
        print(f"[DEBUG] calling launchMegakernel(output_id={output_id}, "
              f"megakernel_o={megakernel_o_path}, graph_path={graph_path})", flush=True)

        return launchMegakernel(
            output_id, callback_path, device_id, block_dim, aicpu_block_dim,
            object_path=megakernel_o_path, graph_path=graph_path
        )
