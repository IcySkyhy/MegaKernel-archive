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
 * \file device_launcher_context.h
 * \brief
 */

#pragma once

#include "interface/configs/config_manager.h"
#include "interface/interpreter/raw_tensor_data.h"
#include "interface/program/program.h"

namespace npu::tile_fwk {
class DeviceLauncherContext {
public:
    static DeviceLauncherContext& Get();
    void Initialize()
    {
        // 使能 Aihac 后端
        oriEnableAihacBackend = config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true);
#ifdef ENABLE_STEST_BINARY_CACHE
        // BinaryCache
        oriEnableBinaryCache = config::GetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, true);
#endif
#ifdef ENABLE_STEST_DUMP_JSON
        oriEnableDumpJson = config::GetPassConfig(KEY_PRINT_GRAPH, oriEnableDumpJson);
        config::SetPassConfig(KEY_PRINT_GRAPH, true);
#endif
        // Reset Program
        Program::GetInstance().Reset();
        ProgramData::GetInstance().Reset();
    }

    void Finalize()
    {
        config::SetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, oriEnableAihacBackend);
#ifdef ENABLE_STEST_BINARY_CACHE
        config::SetPassGlobalConfig(KEY_ENABLE_BINARY_CACHE, oriEnableBinaryCache);
#endif
#ifdef ENABLE_STEST_DUMP_JSON
        config::SetHostConfig(KEY_PRINT_GRAPH, oriEnableDumpJson);
#endif
    }

    void SetCaptureMode(const bool captureMode);

    bool IsCaptureMode() const { return captureMode_; }

protected:
    bool oriEnableAihacBackend = false;
#ifdef ENABLE_STEST_BINARY_CACHE
    bool oriEnableBinaryCache = false;
#endif
#ifdef ENABLE_STEST_DUMP_JSON
    bool oriEnableDumpJson = false;
#endif
    bool captureMode_ = false;
};
} // namespace npu::tile_fwk
