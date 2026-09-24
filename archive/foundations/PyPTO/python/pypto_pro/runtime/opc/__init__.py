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

"""PyPTO binary-delivery compile leaf for the CANN asc_opc flow.

The whole asc_opc driver (op-store, ``SingleOpCompile`` with ``build_config`` + ``op_context``,
``SingleOpPostCompile``) is reused verbatim from the CANN ``asc_opc_tool`` / ``asc_op_compile_base``
packages — nothing is migrated here. The only PyPTO-specific piece is :func:`pypto_compile_op`, a drop-in
replacement for ``asc_op_compiler.compile_op``'s leaf: it takes a PyPTO DSL ``.py`` and does per-tilingkey
codegen + bisheng compile (reusing the asc_op_compile_base backend), producing the same flat ``kernel_meta``
artifacts. The generated per-op wrapper (ops-transformer ``ascendc_impl_build.py``) calls it in place of
``compile_op``.
"""

__all__ = ["pypto_compile_op"]

from .pypto_compile import pypto_compile_op
