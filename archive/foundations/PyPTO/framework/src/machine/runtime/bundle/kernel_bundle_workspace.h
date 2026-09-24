/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kernel_bundle_workspace.h
 * \brief Load-side dynamic-workspace evaluation for the kernel bundle.
 *
 * Rebuilds an Evaluator from the SymbolMeta TLV and computes the workspace size for the real input shapes.
 * The matching pack-side serialization lives in pack/kernel_bundle_pack.h (a different target; see that header).
 */

#pragma once

#include <cstdint>
#include <vector>

namespace npu::tile_fwk {

namespace dynamic {
class DeviceTensorData;
}

namespace bundle {

struct LoadedBundle;

// Evaluate the workspace size for the given tensor shapes and patch the bundle's in-memory devProgram memBudget
// in place. Falls back to the static memBudget.Total() when the bundle has no SymbolMeta segment (old bundles)
// or on parse failure.
uint64_t EvalWorkspaceForShapes(LoadedBundle& bundle, const std::vector<dynamic::DeviceTensorData>& inputs,
                                const std::vector<dynamic::DeviceTensorData>& outputs);

} // namespace bundle
} // namespace npu::tile_fwk
