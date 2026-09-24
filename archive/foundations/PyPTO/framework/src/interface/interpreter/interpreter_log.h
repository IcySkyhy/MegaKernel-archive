/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file interpreter_log.h
 * \brief Interpreter logging helpers and macros.
 */

#pragma once

#include <cstdarg>
#include <cstdint>
#include <string>

namespace npu::tile_fwk::interpreter {
enum class LogLevel : uint8_t { kDebug = 0, kInfo, kWarn, kEvent, kError };

const std::string& LogFilePath();
void SetLogFilePath(const std::string& path);
void Log(LogLevel level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
} // namespace npu::tile_fwk::interpreter

#define INTERPRETER_LOGD(...) npu::tile_fwk::interpreter::Log(npu::tile_fwk::interpreter::LogLevel::kDebug, __VA_ARGS__)
#define INTERPRETER_LOGI(...) npu::tile_fwk::interpreter::Log(npu::tile_fwk::interpreter::LogLevel::kInfo, __VA_ARGS__)
#define INTERPRETER_LOGW(...) npu::tile_fwk::interpreter::Log(npu::tile_fwk::interpreter::LogLevel::kWarn, __VA_ARGS__)
#define INTERPRETER_EVENT(...) \
    npu::tile_fwk::interpreter::Log(npu::tile_fwk::interpreter::LogLevel::kEvent, __VA_ARGS__)

#define INTERPRETER_LOGE(errCode, fmt, ...)                                                                        \
    npu::tile_fwk::interpreter::Log(npu::tile_fwk::interpreter::LogLevel::kError, "ErrCode: F%05X! Enum: %s " fmt, \
                                    static_cast<uint32_t>(errCode) & 0xFFFFF, #errCode, ##__VA_ARGS__)

#define INTERPRETER_LOGE_FULL(errCode, fmt, ...) INTERPRETER_LOGE(errCode, fmt, ##__VA_ARGS__)
