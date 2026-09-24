#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pipeline configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for preload pipeline transformation.

    Args:
        preload: Number of times the first stage pre-fires before steady-state
                 alternation begins. E.g. preload=2 means C1 C1 C1 C2 C1 C2 ...
                 The actual ctx ring-buffer depth is computed from the stage
                 delays (max_delay + 1), not from preload directly.
        sync_only: If True, do NOT transform into a preload pipeline; only
                 auto-insert cross-core sync around each stage in the original
                 serial loop. Lets users validate their serial kernel is correct
                 before enabling the full pipeline. Default False (full pipeline).

    The transformed source is always written to the build directory as
    ``pipeline_generated.py`` for inspection.
    """

    preload: int = 2
    sync_only: bool = False
