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
 * \file kernel_bundle_launch.h
 * \brief Self-contained offline launch for a packed kernel bundle.
 *
 * Lives outside device_launcher so that file needs no bundle code. Reads topology from the packed
 * DevAscendProgram (no Function) and reuses only public DeviceLauncher helpers.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "adapter/api/runtime_define.h"                     // RtStream
#include "machine/runtime/launcher/device_launcher_types.h" // DeviceTensorData, DeviceLauncherConfig
#include "machine/utils/dynamic/dev_encode_tensor.h"        // DevDynamicCellMatchStridePatch

namespace npu::tile_fwk::dynamic {

// Launch a packed bundle once. `devProgBinary` stays base-0 (device relocates); `binHandle` is a pre-registered
// RtBinHandle; `cacheKey` (= LoadedBundle::bundleKey, a content digest -- NOT hashKey, which collides across
// shape variants of one op) keys the process-wide ctrl-flow-cache device buffer. `tensorList` is the unified
// operand list ([nTensors, 0], outputs folded in). `cellMatchStridePatches` are the per-shape patches from
// EvalWorkspaceForShapes (empty when no cell-match table), which must already have patched `devProgBinary`'s
// memBudget for the launch shape. Returns 0 on success.
int LaunchBundleKernelOnce(const std::vector<uint8_t>& devProgBinary, void* binHandle, uint64_t cacheKey,
                           const std::vector<DeviceTensorData>& tensorList,
                           const std::vector<uint8_t>& hostCtrlFlowCache,
                           const std::vector<DevDynamicCellMatchStridePatch>& cellMatchStridePatches,
                           void* workspaceAddr, RtStream aicoreStream, bool streamSynchronize,
                           const DeviceLauncherConfig& config = DeviceLauncherConfig());

} // namespace npu::tile_fwk::dynamic
