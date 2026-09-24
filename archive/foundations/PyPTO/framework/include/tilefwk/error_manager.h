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
 * \file error_manager.h
 * \brief
 */

#pragma once

#include <string>
#include <mutex>
#include <queue>
#include <atomic>

#define REPORT_ERROR_MSG(errCode, fmt, ...)                                                                       \
    do {                                                                                                          \
        npu::tile_fwk::ErrorManager::Instance().ReportErrorMessage(                                               \
            "ErrCode: F%05X! Enum: %s. " fmt, static_cast<uint32_t>(errCode) & 0xFFFFF, #errCode, ##__VA_ARGS__); \
    } while (0)

namespace npu::tile_fwk {
class ErrorManager {
public:
    static ErrorManager& Instance();
    void ReportErrorMessage(const char* fmt, ...);
    void OutputErrorMessage(const bool outputAll = false);
    bool GetFirstErrorMessage(std::string& errMsg);
    ErrorManager(const ErrorManager&) = delete;
    ErrorManager& operator=(const ErrorManager&) = delete;
    ErrorManager(ErrorManager&&) = delete;
    ErrorManager& operator=(ErrorManager&&) = delete;

private:
    ErrorManager();
    ~ErrorManager();
    void ReportInnerErrMsg(const char* fmt, va_list list);
    void SaveErrMsg(const std::string& errMsg);
    bool IsAlive() const { return isInit_; }
    std::atomic<bool> isInit_;
    std::queue<std::string> errorMsgQueue_;
    std::mutex reportMutex_;
};
} // namespace npu::tile_fwk
