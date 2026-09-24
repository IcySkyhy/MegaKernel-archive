# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Maps logical op_type (make_* registry key) -> blade compute + OpGraph wire meta.
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from triton_dist.mega_kernel_ascend.kernels.elementwise import (
    add_compute,
    add_rms_norm_compute,
    silu_mul_up_compute,
)
from triton_dist.mega_kernel_ascend.kernels.matmul import (
    matmul_compute,
    matmul_lmhead_compute,
)


@dataclass(frozen=True)
class OpSpec:
    """One blade-capable op."""

    compute: Callable
    # Wire op type/name written into OpGraph; None keeps the registry key.
    wire_op_name: Optional[str] = None
    opgraph_omit_attrs: bool = True
    opgraph_omit_deps: bool = True


# Keys match mega_triton_kernel ModelBuilder._convert_op / make_* op_type strings.
# Ascend docking names (add_rms_norm / matmul_lmhead) match ops:: on that side.
OP_REGISTRY: Dict[str, OpSpec] = {
    "add": OpSpec(compute=add_compute, wire_op_name="Add"),
    "Matmul": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "matmulA5": OpSpec(compute=matmul_compute, wire_op_name="matmulA5"),
    "linear": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "mlp_fc1": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "mlp_fc2": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "qkv_proj": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "o_proj": OpSpec(compute=matmul_compute, wire_op_name="Matmul"),
    "silu_mul_up": OpSpec(compute=silu_mul_up_compute, wire_op_name="SiluMulUp"),
    "add_rms_norm": OpSpec(compute=add_rms_norm_compute, wire_op_name="add_rms_norm"),
    "matmul_lmhead": OpSpec(compute=matmul_lmhead_compute, wire_op_name="matmul_lmhead"),
}


def get_op_spec(op_type: str) -> OpSpec:
    spec = OP_REGISTRY.get(op_type)
    if spec is None:
        supported = ", ".join(sorted(OP_REGISTRY))
        raise NotImplementedError(
            f"blade backend has no compute for op_type={op_type!r}; "
            f"supported: {supported}"
        )
    return spec
