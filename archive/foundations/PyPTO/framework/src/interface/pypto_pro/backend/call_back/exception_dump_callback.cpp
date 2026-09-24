/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "core/logging.h"

static constexpr int32_t MAX_KERNEL_BUF_LEN = 1024;
static constexpr int32_t MAX_TENSOR_NUM = 128;
static constexpr int32_t MAX_DEBUG_CMD_LEN = 8192;

enum class AdxTensorType : int32_t { INPUT, OUTPUT, WORKSPACE };
enum class AdxAddressType : int32_t { TRADITIONAL, NOTILING, RAW };
enum class AdxExceptionDumpMode : uint32_t {
    ADX_DUMP_MODE_NONE = 0,
    ADX_DUMP_MODE_OVERWRITE = 1,
    ADX_DUMP_MODE_ADDITIONAL = 2,
};

struct AdxTensorInfo {
    AdxTensorType type;
    size_t tensorSize;
    int32_t format;
    int32_t dataType;
    int64_t* tensorAddr;
    AdxAddressType addrType;
    int32_t placement;
    uint32_t argsOffSet;
    std::vector<int64_t> shape;
    std::vector<int64_t> originShape;
};

struct AdxExceptionDumpInfo {
    uint32_t coreId;
    int32_t coreType;
    uint32_t argssize;
    void* argAddr;
    void* bin;
    char kernelName[MAX_KERNEL_BUF_LEN];
    char kernelDisplayName[MAX_KERNEL_BUF_LEN];
    uint32_t extraTensorNum;
    AdxTensorInfo tensorInfo[MAX_TENSOR_NUM];
};

using AdumpExceptionDumpCallback = int32_t (*)(void*, AdxExceptionDumpInfo*, uint32_t, uint32_t*,
                                               AdxExceptionDumpMode*);
using AdumpRegExceptionDumpCallBack = int32_t (*)(AdumpExceptionDumpCallback);

static std::vector<AdxTensorInfo> g_cachedTensors;
static char g_kernelName[MAX_KERNEL_BUF_LEN] = {0};
static char g_debugCmd[MAX_DEBUG_CMD_LEN] = {0};
static bool g_debugCmdExecuted = false;
static bool g_registered = false;
static void* g_adumpHandle = nullptr;
static AdumpRegExceptionDumpCallBack g_regFunc = nullptr;

static AdumpRegExceptionDumpCallBack LoadAdumpRegFunc()
{
    if (g_regFunc != nullptr) {
        return g_regFunc;
    }
    const char* cannPath = std::getenv("ASCEND_CANN_PACKAGE_PATH");
    if (cannPath == nullptr) {
        cannPath = std::getenv("ASCEND_HOME_PATH");
    }
    if (cannPath != nullptr) {
        std::string soPath = std::string(cannPath) + "/lib64/libascend_dump.so";
        g_adumpHandle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    if (g_adumpHandle == nullptr) {
        g_adumpHandle = dlopen("libascend_dump.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (g_adumpHandle == nullptr) {
        return nullptr;
    }
    g_regFunc = reinterpret_cast<AdumpRegExceptionDumpCallBack>(
        dlsym(g_adumpHandle,
              "_ZN3Adx29AdumpRegExceptionDumpCallbackEPFjPvPNS_17ExceptionDumpInfoEjPjPNS_17ExceptionDumpModeEE"));
    return g_regFunc;
}

static int32_t ProExceptionDumpCallback(void* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                                        uint32_t exceptionDumpSize, uint32_t* exceptionDumpRealSize,
                                        AdxExceptionDumpMode* mode)
{
    if (exceptionDumpInfo == nullptr || exceptionDumpRealSize == nullptr || mode == nullptr) {
        return 1;
    }
    (void)exceptionInfo;

    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE;

    for (uint32_t i = 0U; i < exceptionDumpSize; i++) {
        exceptionDumpInfo[i].coreId = 0U;
        exceptionDumpInfo[i].coreType = 0;
        exceptionDumpInfo[i].argAddr = nullptr;
        exceptionDumpInfo[i].argssize = 0U;
        exceptionDumpInfo[i].bin = nullptr;
        snprintf(exceptionDumpInfo[i].kernelName, MAX_KERNEL_BUF_LEN, "%s", g_kernelName);
        snprintf(exceptionDumpInfo[i].kernelDisplayName, MAX_KERNEL_BUF_LEN, "%s", g_kernelName);

        uint32_t idx = 0U;
        for (const auto& cached : g_cachedTensors) {
            if (idx >= MAX_TENSOR_NUM) {
                break;
            }
            exceptionDumpInfo[i].tensorInfo[idx] = cached;
            idx++;
        }
        exceptionDumpInfo[i].extraTensorNum = idx;
    }

    *exceptionDumpRealSize = exceptionDumpSize;

    if (g_debugCmd[0] != '\0' && !g_debugCmdExecuted) {
        g_debugCmdExecuted = true;
        int ret = std::system(g_debugCmd);
        if (ret != 0) {
            IR_LOGW() << "ProExceptionDumpCallback: debug compile command failed (ret=" << ret << ")";
        }
    }

    return 0;
}

extern "C" {

int32_t pro_register_exception_dump_callback()
{
    if (g_registered) {
        return 0;
    }
    auto regFunc = LoadAdumpRegFunc();
    if (regFunc == nullptr) {
        return -1;
    }
    int32_t ret = regFunc(ProExceptionDumpCallback);
    if (ret == 0) {
        g_registered = true;
    }
    return ret;
}

void pro_set_dump_info(const char* kernelName, int32_t numTensors, const int32_t* types, const size_t* tensorSizes,
                       const int32_t* dataTypes, const void** tensorAddrs, const int64_t* flatShapes,
                       const int32_t* shapeCounts, int32_t maxDims)
{
    g_cachedTensors.clear();
    g_cachedTensors.reserve(static_cast<size_t>(numTensors));
    for (int32_t i = 0; i < numTensors; i++) {
        AdxTensorInfo info;
        info.type = static_cast<AdxTensorType>(types[i]);
        info.tensorSize = tensorSizes[i];
        info.format = 0;
        info.dataType = dataTypes[i];
        info.tensorAddr = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(tensorAddrs[i]));
        info.addrType = AdxAddressType::TRADITIONAL;
        info.placement = static_cast<int32_t>(0);
        info.argsOffSet = 0U;
        for (int32_t j = 0; j < shapeCounts[i] && j < maxDims; j++) {
            info.shape.push_back(
                flatShapes[static_cast<size_t>(i) * static_cast<size_t>(maxDims) + static_cast<size_t>(j)]);
            info.originShape.push_back(
                flatShapes[static_cast<size_t>(i) * static_cast<size_t>(maxDims) + static_cast<size_t>(j)]);
        }
        g_cachedTensors.push_back(std::move(info));
    }
    snprintf(g_kernelName, MAX_KERNEL_BUF_LEN, "%s", kernelName);
    g_debugCmdExecuted = false;
}

void pro_clear_dump_info()
{
    g_cachedTensors.clear();
    g_kernelName[0] = '\0';
    g_debugCmd[0] = '\0';
    g_debugCmdExecuted = false;
}

void pro_set_debug_cmd(const char* cmd)
{
    if (cmd == nullptr || cmd[0] == '\0') {
        g_debugCmd[0] = '\0';
        return;
    }
    std::snprintf(g_debugCmd, MAX_DEBUG_CMD_LEN, "%s", cmd);
    g_debugCmdExecuted = false;
}

#ifdef ENABLE_TESTS
extern "C" int32_t pro_test_exception_dump_callback(uint32_t dumpSize, uint32_t* realSize, uint32_t* mode,
                                                    char* outKernelName, uint32_t kernelNameBufSize,
                                                    uint32_t* outExtraTensorNum)
{
    AdxExceptionDumpInfo dumpInfo = {};
    AdxExceptionDumpMode dumpMode = AdxExceptionDumpMode::ADX_DUMP_MODE_NONE;
    uint32_t realDumpSize = 0;
    int32_t ret = ProExceptionDumpCallback(nullptr, &dumpInfo, dumpSize, &realDumpSize, &dumpMode);
    if (realSize != nullptr) {
        *realSize = realDumpSize;
    }
    if (mode != nullptr) {
        *mode = static_cast<uint32_t>(dumpMode);
    }
    if (outKernelName != nullptr && kernelNameBufSize > 0) {
        snprintf(outKernelName, kernelNameBufSize, "%s", dumpInfo.kernelName);
    }
    if (outExtraTensorNum != nullptr) {
        *outExtraTensorNum = dumpInfo.extraTensorNum;
    }
    return ret;
}

extern "C" int32_t pro_test_exception_dump_callback_nullptr()
{
    return ProExceptionDumpCallback(nullptr, nullptr, 1, nullptr, nullptr);
}
#endif
}
