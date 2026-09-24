/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

namespace npu::tile_fwk {

class Function;

/**
 * Post-build finalization for dynamic functions lowered from new IR via RootFunctionBuilder.
 * Analogous to the dynFunc tail of RecordFunc::EndFunction (recorder.cpp).
 */
void FinalizeDynamicFunction(Function* dynFunc);

} // namespace npu::tile_fwk
