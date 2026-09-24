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
 * \file log_module_manager.h
 * \brief
 */

#pragma once

#include <array>
#include "tilefwk/tilefwk_log.h"

namespace npu::tile_fwk {
class LogModuleManager {
public:
    static LogModuleManager& Instance();
    int32_t GetModuleLogLevel(const LogModule logModule) const;
    int32_t GetLowestLogLevel() const;

private:
    LogModuleManager();
    ~LogModuleManager();

    std::array<int32_t, static_cast<size_t>(LogModule::BOTTOM)> moduleLogLevel_;
};
} // namespace npu::tile_fwk
