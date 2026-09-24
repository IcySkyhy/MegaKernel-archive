# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Blade-based megakernel package.
#
# Blade provides two Python modules:
#   - ascendmk_dp: SchemaBuilder, AxisMap, TensorHandle (operator schema definition)
#   - ascendmk_rt: graph building, tiling registration, launch (runtime)
#
# Compute kernels are written as @triton.jit, compiled to .o by Triton,
# then registered with Blade via aclrt_registry_triton_op.
# Users build graphs via aclrt_add_tensor / aclrt_operator_tensor,
# and launch via aclrt_run (which calls aclrtLaunchMega internally).
#
# Usage:
#   PYTHONPATH=/path/to/megakernel/output/lib:$PYTHONPATH
#   from triton_dist.megakernel import *
#
#   # 1. Register a Triton-compiled .o with Blade
#   schema = SchemaBuilder("add").inputs("X1","X2").outputs("Y")
#   schema.outputTile("Y").tileDims("d0").reads("X1",[dim("d0")]).reads("X2",[dim("d0")])
#   aclrt_registry_triton_op("add", "tiled_add", "add", "./add.o", schema, 1, 1)
#
#   # 2. Build graph
#   a = aclrt_add_tensor("A", torch_tensor)
#   b = aclrt_add_tensor("B", torch_tensor)
#   out = aclrt_operator_tensor("add", a, b, numel)
#
#   # 3. Launch (aclrtLaunchMega inside)
#   result = aclrt_run(out, "./libdefault_callback.so", 0, 24, 1)

import ascendmk_dp
import ascendmk_rt

from ascendmk_dp import (
    SchemaBuilder, OutputTileBuilder, TensorHandle,
    AxisMap, AxisMapKind, AffineExpr, AffineExprKind,
    ShapeDimExpr, Kind, MemoryPool,
    dim, all, slice, offset, affine, dataDep,
)

from ascendmk_rt import (
    DType, TensorDesc,
    SessionOpDesc, SessionDesc, Session, create_session,
    aclrt_add_tensor, aclrt_operator_tensor,
    aclrt_registry_tile, aclrt_registry_binary,
    aclrt_registry_triton_op, aclrt_registry_op_tiling,
    launchMegakernel,
)

from .compiler import MEGA_BACKEND, MegaKernelBackend, MegaKernelDriver, mega_target


def _register_mega_backend():
    """Make MEGA_BACKEND resolvable by triton's make_backend().

    Runtime registration rather than a setuptools entry point: it needs no
    changes to setup.py and copies nothing back into the triton-ascend tree.
    """
    from triton.backends import backends, Backend
    backends.setdefault(MEGA_BACKEND, Backend(MegaKernelBackend, MegaKernelDriver))


_register_mega_backend()

from .model_builder import ModelBuilder

__all__ = [
    # ascendmk_dp (schema layer)
    "SchemaBuilder", "OutputTileBuilder", "TensorHandle",
    "AxisMap", "AxisMapKind", "AffineExpr", "AffineExprKind",
    "ShapeDimExpr", "Kind", "MemoryPool",
    "dim", "all", "slice", "offset", "affine", "dataDep",
    # ascendmk_rt (runtime layer)
    "DType", "TensorDesc",
    "SessionOpDesc", "SessionDesc", "Session", "create_session",
    "aclrt_add_tensor", "aclrt_operator_tensor",
    "aclrt_registry_tile", "aclrt_registry_binary",
    "aclrt_registry_triton_op", "aclrt_registry_op_tiling",
    "aclrt_run",
    # megakernel compile backend
    "MEGA_BACKEND", "MegaKernelBackend",
    # ModelBuilder
    "ModelBuilder",
]
