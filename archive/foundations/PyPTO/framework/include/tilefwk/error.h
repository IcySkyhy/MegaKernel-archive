/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file error.h
 * \brief
 */

#pragma once

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

#include "error_code.h"
#include "lazy.h"

namespace npu::tile_fwk {
using Backtrace = std::shared_ptr<LazyValue<std::string>>;

Backtrace GetBacktrace(size_t skipFrames, size_t maxFrames);

struct ErrorMessage {
public:
    std::string Message() { return ss.str(); }

    template <typename T>
    ErrorMessage& operator<<(const T& value)
    {
        ss << value;
        return *this;
    }

    template <typename T>
    ErrorMessage& operator<<(const std::vector<T>& vec)
    {
        ss << "[";
        for (auto iter = vec.begin(); iter != vec.end(); ++iter) {
            if (iter != vec.begin()) {
                ss << ", ";
            }
            ss << *iter;
        }
        ss << "]";
        return *this;
    }

    // support std::endl etc.
    ErrorMessage& operator<<(std::ostream& (*manipulator)(std::ostream&))
    {
        ss << manipulator;
        return *this;
    }

    std::stringstream ss;
};

class Error : public std::exception {
public:
    Error(const char* func, const char* file, size_t line, const std::string& msg, Backtrace backtrace)
        : func_(func), file_(file), line_(line), msg_(msg), backtrace_(backtrace)
    {}

    Error(const char* func, const char* file, size_t line, Backtrace backtrace = nullptr)
        : func_(func), file_(file), line_(line), backtrace_(backtrace)
    {}

    const char* what() const noexcept override;

    [[nodiscard]] std::string DiagnosticWithBacktrace() const;

    int operator=(ErrorMessage& msg);

private:
    const char* func_;
    const char* file_;
    size_t line_;
    std::string msg_;
    Backtrace backtrace_;
    mutable LazyShared<std::string> what_;
};

class AssertInfo {
public:
    [[noreturn]] int operator=(ErrorMessage& msg)
    {
        (void)fprintf(stderr, "%s\n", msg.Message().c_str());
        abort();
    }
};

#ifndef __DEVICE__
#define ASSERT_WITH_CODE(errcode, cond)                                                                             \
    (cond) ? 0 :                                                                                                    \
             npu::tile_fwk::Error(__func__, __FILE__, __LINE__,                                                     \
                                  npu::tile_fwk::GetBacktrace(                                                      \
                                      0, /* 64 is maxFrames */ 64)) = npu::tile_fwk::ErrorMessage()                 \
                                                                      << "ASSERT FAILED: "                          \
                                                                      << "ErrCode: F" << std::uppercase << std::hex \
                                                                      << std::setw(5) << std::setfill('0')          \
                                                                      << (static_cast<unsigned>(errcode) & 0xFFFFF) \
                                                                      << std::dec << "! Enum: " << #errcode << "\n"

#define CHECK_WITH_CODE(errcode, cond)                                                                              \
    (cond) ? 0 :                                                                                                    \
             npu::tile_fwk::Error(__func__, __FILE__, __LINE__,                                                     \
                                  npu::tile_fwk::GetBacktrace(                                                      \
                                      0, /* 64 is maxFrames */ 64)) = npu::tile_fwk::ErrorMessage()                 \
                                                                      << "CHECK FAILED: "                           \
                                                                      << "ErrCode: F" << std::uppercase << std::hex \
                                                                      << std::setw(5) << std::setfill('0')          \
                                                                      << (static_cast<unsigned>(errcode) & 0xFFFFF) \
                                                                      << std::dec << "! Enum: " << #errcode << "\n"

#define TILEFWK_ERROR()                                \
    npu::tile_fwk::Error(__func__, __FILE__, __LINE__, \
                         npu::tile_fwk::GetBacktrace(0, /* 64 is maxFrames */ 64)) = npu::tile_fwk::ErrorMessage()
#else
#define ASSERT_WITH_CODE(errcode, cond)                                                                         \
    (cond) ? 0 :                                                                                                \
             AssertInfo() = npu::tile_fwk::ErrorMessage()                                                       \
                            << "ASSERT FAILED: "                                                                \
                            << "ErrCode: F" << std::uppercase << std::hex << std::setw(5) << std::setfill('0')  \
                            << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "! Enum: " << #errcode \
                            << "\n"

#define CHECK_WITH_CODE(errcode, cond)                                                                          \
    (cond) ? 0 :                                                                                                \
             AssertInfo() = npu::tile_fwk::ErrorMessage()                                                       \
                            << "CHECK FAILED: "                                                                 \
                            << "ErrCode: F" << std::uppercase << std::hex << std::setw(5) << std::setfill('0')  \
                            << (static_cast<unsigned>(errcode) & 0xFFFFF) << std::dec << "! Enum: " << #errcode \
                            << "\n"
#endif

#define ASSERT_DEFAULT_SELECT(_1, _2, NAME, ...) NAME
#define ASSERT_DEFAULT_WITHOUT_ERR_CODE(default_errcode, cond) ASSERT_WITH_CODE(default_errcode, cond)
#define ASSERT_DEFAULT_WITH_ERR_CODE(default_errcode, errcode, cond) ASSERT_WITH_CODE(errcode, cond)
#define ASSERT_DEFAULT(default_errcode, ...)                                                          \
    ASSERT_DEFAULT_SELECT(__VA_ARGS__, ASSERT_DEFAULT_WITH_ERR_CODE, ASSERT_DEFAULT_WITHOUT_ERR_CODE) \
    (default_errcode, __VA_ARGS__)

#define CHECK_DEFAULT_SELECT(_1, _2, NAME, ...) NAME
#define CHECK_DEFAULT_WITHOUT_ERR_CODE(default_errcode, cond) CHECK_WITH_CODE(default_errcode, cond)
#define CHECK_DEFAULT_WITH_ERR_CODE(default_errcode, errcode, cond) CHECK_WITH_CODE(errcode, cond)
#define CHECK_DEFAULT(default_errcode, ...)                                                        \
    CHECK_DEFAULT_SELECT(__VA_ARGS__, CHECK_DEFAULT_WITH_ERR_CODE, CHECK_DEFAULT_WITHOUT_ERR_CODE) \
    (default_errcode, __VA_ARGS__)

#define ASSERT(...) ASSERT_DEFAULT(npu::tile_fwk::InternalError::COMMON_INNER_ERROR, __VA_ARGS__)
#define CHECK(...) CHECK_DEFAULT(npu::tile_fwk::ExternalError::COMMON_EXTERNAL_ERROR, __VA_ARGS__)

#ifndef FE_ASSERT
#define FE_ASSERT(...) ASSERT_DEFAULT(npu::tile_fwk::InternalError::FE_INNER_ERROR, __VA_ARGS__)
#endif

#ifndef PASS_ASSERT
#define PASS_ASSERT(...) ASSERT_DEFAULT(npu::tile_fwk::InternalError::PASS_INNER_ERROR, __VA_ARGS__)
#endif

#ifndef MACHINE_ASSERT
#define MACHINE_ASSERT(...) ASSERT_DEFAULT(npu::tile_fwk::InternalError::MACHINE_INNER_ERROR, __VA_ARGS__)
#endif

#ifndef SIM_ASSERT
#define SIM_ASSERT(...) ASSERT_DEFAULT(npu::tile_fwk::InternalError::SIM_INNER_ERROR, __VA_ARGS__)
#endif
} // namespace npu::tile_fwk
