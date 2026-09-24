# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Package entry. Default ``ModelBuilder`` is the blade backend (TD-compatible
# make_* -> op_id codegen). Scoreboard / TaskBase path:
#   ModelBuilder(backend="scoreboard") or ScoreboardModelBuilder(...).
from .facade import ModelBuilder, ScoreboardModelBuilder  # noqa: E402,F401
from . import kernels  # noqa: F401,E402

__all__ = ["ModelBuilder", "ScoreboardModelBuilder", "kernels"]
