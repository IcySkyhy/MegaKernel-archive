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

"""Stage decorator for pipeline functions."""

PIPELINE_STAGE_ATTR = "pipeline_stage"


def stage(fn):
    """Mark a function as a pipeline stage.

    This is a transparent decorator that tags the function with
    ``pipeline_stage = True`` for the pipeline framework to identify.
    It does not change the function's behavior.
    """
    setattr(fn, PIPELINE_STAGE_ATTR, True)
    return fn


def is_pipeline_stage(fn) -> bool:
    """Return whether a callable was marked by ``@stage``."""
    return callable(fn) and getattr(fn, PIPELINE_STAGE_ATTR, False)
