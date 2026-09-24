# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bootstrap the native PyPTO extension for Python package imports."""

import sys

from pypto import pypto_impl as _pypto_impl

# pypto_impl is a C extension whose submodules must be registered before
# downstream Python packages can import them through dotted module paths.
for _submodule_name in ("ir", "backend", "codegen", "logging"):
    if hasattr(_pypto_impl, _submodule_name):
        sys.modules.setdefault(f"pypto.pypto_impl.{_submodule_name}", getattr(_pypto_impl, _submodule_name))

DataType = _pypto_impl.ir.DataType
InternalError = _pypto_impl.InternalError
codegen = _pypto_impl.codegen
