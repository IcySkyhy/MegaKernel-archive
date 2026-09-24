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
 * \file cann_host_runtime.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>

namespace npu {
namespace tile_fwk {
using GetSocVerFunc = int (*)(char*, const uint32_t);
using GetSocSpecFunc = int (*)(const char*, const char*, char*, const uint32_t);
using GetDeviceInfoFunc = int (*)(uint32_t, uint32_t, int64_t*);
using GetDeviceIdFunc = int (*)(int32_t*);

class CannHostRuntime {
public:
    static CannHostRuntime& Instance();
    bool GetSocVersion(std::string& socVersion);
    bool GetSocSpec(const std::string& column, const std::string& key, std::string& val);
    bool GetAICPUCnt(size_t& aiCpuCnt);

    CannHostRuntime(const CannHostRuntime&) = delete;
    CannHostRuntime& operator=(const CannHostRuntime&) = delete;

private:
    CannHostRuntime();
    ~CannHostRuntime();
    void* GetSymbol(const std::string& sym);

    GetSocVerFunc socVerFunc_ = nullptr;
    GetSocSpecFunc socSpecFunc_ = nullptr;
    GetDeviceInfoFunc deviceInfoFunc_ = nullptr;
    GetDeviceIdFunc getDeviceIdFunc_ = nullptr;
    void* handleDep_ = nullptr;
    void* handle_ = nullptr;
    void* aclHandle_ = nullptr;
};
} // namespace tile_fwk
} // namespace npu
