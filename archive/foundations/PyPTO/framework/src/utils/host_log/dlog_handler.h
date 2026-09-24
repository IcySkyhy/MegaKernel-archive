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
 * \file dlog_handler.h
 * \brief
 */

#pragma once

#include <cstdint>

namespace npu::tile_fwk {
class DLogHandler {
public:
    static DLogHandler& Instance();
    int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel) const;
    int32_t GetLogLevel(int32_t moduleId, int32_t* enableEvent) const;
    int32_t SetLogLevel(int32_t moduleId, int32_t logLevel, int32_t enableEvent) const;
    bool IsAvailable() const { return checkLevelFunc_ != nullptr && logRecordFunc_ != nullptr; }
    void (*logRecordFunc_)(int32_t, int32_t, const char*, ...);

private:
    DLogHandler();
    ~DLogHandler();
    void CloseHandle();
    void* handle_{nullptr};
    int32_t (*checkLevelFunc_)(int32_t, int32_t);
    int32_t (*getLevelFunc_)(int32_t, int32_t*);
    int32_t (*setLevelFunc_)(int32_t, int32_t, int32_t);
};
} // namespace npu::tile_fwk
