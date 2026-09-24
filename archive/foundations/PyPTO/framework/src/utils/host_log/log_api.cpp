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
 * \file log_api.cpp
 * \brief
 */

#include <cstdarg>
#include <array>

#include "tilefwk/tilefwk_log.h"
#include "host_log/log_manager.h"
#include "host_log/dlog_handler.h"
#include "host_log/log_module_manager.h"

namespace npu::tile_fwk {
int32_t TilefwkCheckLogLevel(int32_t moduleId, int32_t logLevel, LogModule logModule)
{
    (void)moduleId;
    int32_t moduleLevel = LogModuleManager::Instance().GetModuleLogLevel(logModule);
    if (moduleLevel >= 0) {
        return logLevel >= moduleLevel ? 1 : 0;
    }
    return LogManager::Instance().CheckLevel(static_cast<LogLevel>(logLevel)) ? 1 : 0;
}

void TilefwkLogRecord(int32_t moduleId, int32_t logLevel, const char* fmt, ...)
{
    (void)moduleId;
    va_list list;
    va_start(list, fmt);
    LogManager::Instance().Record(static_cast<LogLevel>(logLevel), fmt, list);
    va_end(list);
}

void TilefwkSetLogAttr(bool isDevice)
{
    if (isDevice) {
        LogManager::Instance().EnableDeviceLog();
    } else {
        LogManager::Instance().EnableHostLog();
    }
}

#ifndef __DEVICE__
LogFuncInfo& LogFuncInfo::Instance()
{
    static LogFuncInfo instance;
    return instance;
}
LogFuncInfo::LogFuncInfo()
{
    checkLevel = TilefwkCheckLogLevel;
    record = TilefwkLogRecord;
    pyptoRecord = TilefwkLogRecord;
    setAttr = TilefwkSetLogAttr;
}

LogFuncInfo::~LogFuncInfo()
{
    checkLevel = nullptr;
    record = nullptr;
    pyptoRecord = nullptr;
    setAttr = nullptr;
}
#endif
} // namespace npu::tile_fwk
