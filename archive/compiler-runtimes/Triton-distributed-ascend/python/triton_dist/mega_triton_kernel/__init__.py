################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
# CUDA ModelBuilder / tasks pull nvshmem + triton_dist.models.utils (CUDA/HIP).
# Keep them lazy so Ascend blade can import core.graph / core.op_graph.
from .core.tile_graph import (  # noqa: F401
    TileGraph,
    TileNode,
    TileEdge,
    EdgeOrigin,
    build_tilegraph_from_tasks,
    serialize_tilegraph,
    serialize_tilegraph_to_dict,
    deserialize_tilegraph,
    save_tilegraph_to_file,
    load_tilegraph_from_file,
)
from .core.op_graph import (  # noqa: F401
    OpGraph,
    OpNode,
    OpAttributes,
    Tensor as OpTensor,
    TensorDesc,
    DType,
    LayoutTag,
    MemoryPool,
    build_opgraph_from_graph,
    serialize_opgraph,
    serialize_opgraph_to_dict,
    deserialize_opgraph,
    canonicalize_opgraph_bytes,
    save_opgraph_to_file,
    load_opgraph_from_file,
)


def __getattr__(name):
    if name == "ModelBuilder":
        # Task builders register into core.registry only when tasks/*.py are
        # imported. Eager package-level ``from . import tasks`` would pull
        # CUDA/HIP deps via TaskBuilderBase -> models.utils; keep that lazy,
        # but always load tasks before ModelBuilder so make_* works.
        from . import tasks as _tasks  # noqa: F401
        from .models.model_builder import ModelBuilder
        return ModelBuilder
    if name == "tasks":
        from . import tasks
        return tasks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
