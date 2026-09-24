# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Package-level ModelBuilder entry: default blade backend so existing make_* /
# compile / run scripts can target Ascend megakernel with minimal changes.
# Scoreboard / TaskBase path remains available via backend="scoreboard".
from __future__ import annotations

from typing import Any


def _ensure_scoreboard_patches() -> None:
    """Install scoreboard NPU import stubs + task_context patches (idempotent)."""
    from triton_dist.mega_kernel_ascend.models import model_builder as _mb  # noqa: F401
    from triton_dist.mega_kernel_ascend.kernels import task_context as _ascend_overrides
    import triton_dist.mega_triton_kernel.kernels.task_context as _orig_tc
    _ascend_overrides._apply_patches(_orig_tc)


def ModelBuilder(*args: Any, backend: str = "blade", **kwargs: Any):
    """Construct a ModelBuilder.

    Args:
        backend: ``"blade"`` (default) or ``"scoreboard"``.
        *args / **kwargs: forwarded to the selected implementation. Blade
            accepts the scoreboard constructor kwargs and ignores unused ones.
    """
    if backend == "blade":
        from triton_dist.mega_kernel_ascend.models_blade.model_builder import (
            ModelBuilder as BladeModelBuilder,
        )
        return BladeModelBuilder(*args, **kwargs)
    if backend == "scoreboard":
        return ScoreboardModelBuilder(*args, **kwargs)
    raise ValueError(
        f"unknown ModelBuilder backend={backend!r}; expected 'blade' or 'scoreboard'"
    )


def ScoreboardModelBuilder(*args: Any, **kwargs: Any):
    """Explicit scoreboard / TaskBase NPU ModelBuilder."""
    _ensure_scoreboard_patches()
    from triton_dist.mega_kernel_ascend.models.model_builder import (
        ModelBuilder as _Scoreboard,
    )
    return _Scoreboard(*args, **kwargs)
