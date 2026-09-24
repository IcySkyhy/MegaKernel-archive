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

from . import ir


class TypeConverter:
    def ConvertPipeType(self, pipe: ir.PipeType) -> str: ...
    def ConvertEventId(self, event_id: int) -> str: ...
    def GenerateShapeType(self, dims: list[int]) -> str: ...
    def GenerateStrideType(self, shape: list[int]) -> str: ...


class CCECodegen:
    def __init__(self, target: ir.SectionKind) -> None: ...

    def generate_single(
        self,
        program: ir.Program,
        arch: str,
    ) -> str: ...

    def get_tiling_headers(self) -> dict[str, str]: ...
