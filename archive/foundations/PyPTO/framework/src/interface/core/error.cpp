/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "core/error.h"

#include <sstream>
#include <string>

namespace pypto {
namespace ir {

std::string Error::GetFormattedStackTrace() const { return Backtrace::FormatStackTrace(stackTrace_); }

std::string Error::GetFullMessage() const
{
    std::ostringstream oss;

    oss << what();

    // Append C++ stack trace
    std::string stackTrace = GetFormattedStackTrace();
    if (!stackTrace.empty()) {
        oss << "\n\nC++ Traceback (most recent call last):\n";
        oss << stackTrace;
    } else {
        oss << "\n\nNo stack trace available. \n"
               "(Tip: Build with CMake in Debug or RelWithDebInfo mode to enable stack trace support.)";
    }

    return oss.str();
}

} // namespace ir
} // namespace pypto
