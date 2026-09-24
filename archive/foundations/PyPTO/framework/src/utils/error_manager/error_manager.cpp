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
 * \file error_manager.cpp
 * \brief
 */

#include "tilefwk/error_manager.h"

#include <cstdarg>
#include <iostream>
#include <vector>
#include "securec.h"

namespace npu::tile_fwk {
namespace {
constexpr size_t MAX_MSG_LENGTH = 1024;
}
ErrorManager::ErrorManager() : isInit_(true) {}

ErrorManager::~ErrorManager() { isInit_ = false; }

ErrorManager& ErrorManager::Instance()
{
    static ErrorManager instance;
    return instance;
}

void ErrorManager::ReportErrorMessage(const char* fmt, ...)
{
    if (!IsAlive()) {
        return;
    }
    va_list list;
    va_start(list, fmt);
    ReportInnerErrMsg(fmt, list);
    va_end(list);
}

void ErrorManager::ReportInnerErrMsg(const char* fmt, va_list list)
{
    std::vector<char> msgbuf(MAX_MSG_LENGTH, '\0');
    int ret = vsprintf_s(msgbuf.data(), MAX_MSG_LENGTH, fmt, list);
    if (ret < 0) {
        std::cerr << "Construct error message failed: " << ret << std::endl;
        return;
    }
    SaveErrMsg(std::string(msgbuf.data()));
}

void ErrorManager::SaveErrMsg(const std::string& errMsg)
{
    if (errMsg.empty()) {
        return;
    }
    const std::lock_guard<std::mutex> lockGuard(reportMutex_);
    errorMsgQueue_.push(errMsg);
}

void ErrorManager::OutputErrorMessage(const bool outputAll)
{
    if (!IsAlive()) {
        return;
    }
    const std::lock_guard<std::mutex> lockGuard(reportMutex_);
    if (outputAll) {
        while (!errorMsgQueue_.empty()) {
            std::cerr << errorMsgQueue_.front() << std::endl;
            errorMsgQueue_.pop();
        }
    } else {
        if (!errorMsgQueue_.empty()) {
            std::cerr << errorMsgQueue_.front() << std::endl;
            errorMsgQueue_.pop();
        }
    }
}

bool ErrorManager::GetFirstErrorMessage(std::string& errMsg)
{
    const std::lock_guard<std::mutex> lockGuard(reportMutex_);
    if (errorMsgQueue_.empty()) {
        return false;
    }
    errMsg = errorMsgQueue_.front();
    return true;
}
} // namespace npu::tile_fwk
