#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Helpers mirroring framework/tests/.../test_dynamic_ops.cpp golden + EXPECT_NO_VERIFY_FAILED."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Sequence

import torch

import pypto


def set_verify_goldens(goldens: Sequence[Optional[torch.Tensor]]) -> None:
    """Register user goldens (None = skip that tensor arg), like ProgramData::AppendGoldens."""
    pypto.set_verify_golden_data(goldens=list(goldens))


_VERIFY_FAIL_RE = re.compile(r"error|fail", re.IGNORECASE)


def assert_pass_verify_ok() -> None:
    """Fail if pass_verify reported a golden mismatch (C++ EXPECT_NO_VERIFY_FAILED equivalent)."""
    log_path = Path(pypto.pypto_impl.LogTopFolder()) / "verify" / "interpreter.log"
    if not log_path.is_file():
        return
    text = log_path.read_text(errors="ignore")
    if "result FAILED" in text or "[VERIFY:FAIL]" in text:
        raise AssertionError(
            f"pass_verify golden check failed (see {log_path}). "
            "Interpreter reported result FAILED / [VERIFY:FAIL]."
        )
    match = _VERIFY_FAIL_RE.search(text)
    if match:
        raise AssertionError(
            f"pass_verify check failed (see {log_path}). "
            f"Interpreter log contains error/fail keyword: {match.group()!r}."
        )
