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
 * \file cann_host_runtime.cpp
 * \brief
 */

#include "tilefwk/cann_host_runtime.h"
#include "utils/file_utils.h"
#include "tilefwk/pypto_fwk_log.h"
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace npu {
namespace tile_fwk {
const uint32_t kMaxLength = 50;
const std::string socVerFuncName = "rtGetSocVersion";
const std::string socSpecFuncName = "rtGetSocSpec";
const std::string deviceInfoFuncName = "aclrtGetDeviceInfo";
const std::string getDeviceFuncName = "aclrtGetDevice";

void* CannHostRuntime::GetSymbol(const std::string& sym)
{
#ifdef BUILD_WITH_CANN
    if (handleDep_ != nullptr && handle_ != nullptr) {
        return dlsym(handle_, sym.c_str());
    }
#endif
    (void)sym;
    return nullptr;
}

CannHostRuntime::CannHostRuntime()
{
#ifdef BUILD_WITH_CANN
    const char* ascendCannPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendCannPath == nullptr || std::strlen(ascendCannPath) == 0) {
        FE_LOGW("Environment variable ASCEND_HOME_PATH is not set or empty.");
        return;
    }
    std::string LibPathDir = std::string(ascendCannPath) + "/lib64/";
    std::string soDepPath = RealPath(LibPathDir + "libprofapi.so");
    FE_LOGW("soDepPath = %s", soDepPath.c_str());
    handleDep_ = dlopen(soDepPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    std::string soPath = RealPath(LibPathDir + "libruntime.so");
    FE_LOGW("soPath = %s", soPath.c_str());
    handle_ = dlopen(soPath.c_str(), RTLD_LAZY);
    if (handleDep_ != nullptr && handle_ != nullptr) {
        socVerFunc_ = (GetSocVerFunc)GetSymbol(socVerFuncName);
        socSpecFunc_ = (GetSocSpecFunc)GetSymbol(socSpecFuncName);
    }
    std::string aclSoPath = RealPath(LibPathDir + "libascendcl.so");
    FE_LOGW("aclSoPath = %s", aclSoPath.c_str());
    aclHandle_ = dlopen(aclSoPath.c_str(), RTLD_LAZY);
    if (aclHandle_ != nullptr) {
        deviceInfoFunc_ = (GetDeviceInfoFunc)dlsym(aclHandle_, deviceInfoFuncName.c_str());
        getDeviceIdFunc_ = (GetDeviceIdFunc)dlsym(aclHandle_, getDeviceFuncName.c_str());
    }
#endif
    if (handleDep_ == nullptr || handle_ == nullptr) {
        FE_LOGW("Cannot obtain so file through dlopen.");
    }
}

CannHostRuntime::~CannHostRuntime()
{
    if (handle_ != nullptr) {
        dlclose(handle_);
    }
    if (handleDep_ != nullptr) {
        dlclose(handleDep_);
    }
    if (aclHandle_ != nullptr) {
        dlclose(aclHandle_);
    }
}

CannHostRuntime& CannHostRuntime::Instance()
{
    static CannHostRuntime instance;
    return instance;
}

bool CannHostRuntime::GetSocVersion(std::string& socVersion)
{
#ifdef BUILD_WITH_CANN
    int ret = 1;
    char socVer[kMaxLength] = {0x00};
    if (socVerFunc_ != nullptr) {
        ret = socVerFunc_(socVer, kMaxLength);
        socVer[kMaxLength - 1] = '\0';
    }
    if (ret == 0) {
        socVersion = std::string(socVer);
        return true;
    }
#endif
    socVersion.clear();
    return false;
}

bool CannHostRuntime::GetSocSpec(const std::string& column, const std::string& key, std::string& val)
{
#ifdef BUILD_WITH_CANN
    int ret = 1;
    char charVal[kMaxLength] = {0};
    if (socSpecFunc_ != nullptr) {
        ret = socSpecFunc_(column.c_str(), key.c_str(), charVal, kMaxLength);
    }
    if (ret == 0) {
        charVal[kMaxLength - 1] = '\0';
        val = std::string(charVal);
        return true;
    }
#endif
    (void)column;
    (void)key;
    (void)val;
    return false;
}

bool CannHostRuntime::GetAICPUCnt(size_t& aiCpuCnt)
{
#ifdef BUILD_WITH_CANN
    if (deviceInfoFunc_ != nullptr && getDeviceIdFunc_ != nullptr) {
        int32_t deviceId = 0;
        if (getDeviceIdFunc_(&deviceId) != 0) {
            FE_LOGW("aclrtGetDevice failed.");
            return false;
        }
        int64_t aicpuCnt = 0;
        const uint32_t aclDevAttrAicpuCoreNum = 1;
        if (deviceInfoFunc_(static_cast<uint32_t>(deviceId), aclDevAttrAicpuCoreNum, &aicpuCnt) == 0) {
            aiCpuCnt = static_cast<size_t>(aicpuCnt);
            return true;
        }
    }
    FE_LOGW("aclrtGetDeviceInfo failed.");
#endif
    (void)aiCpuCnt;
    return false;
}
} // namespace tile_fwk
} // namespace npu
