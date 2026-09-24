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
 * \file runtime_stubs.cpp
 * \brief
 */

#include "adapter/stubs/runtime_stubs.h"
#include "tilefwk/pypto_fwk_log.h"
#include "securec.h"

namespace npu::tile_fwk {
constexpr uint64_t MAX_MALLOC_SIZE = 1024 * 1024 * 1024;
RtError StubMalloc(void** devPtr, uint64_t size, RtMemType type, const uint16_t moduleId)
{
    ADAPTER_LOGD("Enter stub function of Malloc.");
    (void)type;
    (void)moduleId;
    if (devPtr == nullptr || size == 0 || size > MAX_MALLOC_SIZE) {
        ADAPTER_LOGW("devPtr is null or size can not be support");
        return RT_SUCCESS;
    }
    *devPtr = malloc(size);
    if (*devPtr == nullptr) {
        ADAPTER_LOGW("StubMalloc failed: malloc returned nullptr");
        return static_cast<RtError>(-1);
    }
    return RT_SUCCESS;
}

RtError StubMemset(void* devPtr, uint64_t destMax, uint32_t val, uint64_t cnt)
{
    ADAPTER_LOGD("Enter stub function of Memset.");
    (void)devPtr;
    (void)destMax;
    (void)val;
    (void)cnt;
    return RT_SUCCESS;
}

RtError StubMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind)
{
    ADAPTER_LOGD("Enter stub function of Memcpy.");
    (void)kind;
    if (dst != nullptr && src != nullptr && cnt <= destMax) {
        errno_t ret = memcpy_s(dst, destMax, src, cnt);
        if (ret != 0) {
            ADAPTER_LOGW("StubMemcpy failed: memcpy_s returned %d", ret);
            return static_cast<RtError>(-1);
        }
    }
    return RT_SUCCESS;
}

RtError StubMemcpyAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind, RtStream stm)
{
    ADAPTER_LOGD("Enter stub function of MemcpyAsync.");
    (void)dst;
    (void)destMax;
    (void)src;
    (void)cnt;
    (void)kind;
    (void)stm;
    return RT_SUCCESS;
}

RtError StubFree(void* devPtr)
{
    ADAPTER_LOGD("Enter stub function of Free.");
    if (devPtr != nullptr) {
        free(devPtr);
    }
    return RT_SUCCESS;
}

RtError StubSetDevice(int32_t devId)
{
    ADAPTER_LOGD("Enter stub function of SetDevice.");
    (void)devId;
    return RT_SUCCESS;
}

RtError StubGetDevice(int32_t* devId)
{
    ADAPTER_LOGD("Enter stub function of GetDevice.");
    (void)devId;
    return RT_SUCCESS;
}

RtError StubGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
    ADAPTER_LOGD("Enter stub function of GetSocSpec.");
    (void)label;
    (void)key;
    (void)val;
    (void)maxLen;
    return RT_SUCCESS;
}

RtError StubGetSocVersion(char_t* ver, const uint32_t maxLen)
{
    ADAPTER_LOGD("Enter stub function of GetSocVersion.");
    (void)ver;
    (void)maxLen;
    return RT_SUCCESS;
}

RtError StubGetAiCpuCount(uint32_t* aiCpuCnt)
{
    ADAPTER_LOGD("Enter stub function of GetAiCpuCount.");
    (void)aiCpuCnt;
    return RT_SUCCESS;
}

RtError StubGetL2CacheOffset(uint32_t deviceId, uint64_t* offset)
{
    ADAPTER_LOGD("Enter stub function of GetL2CacheOffset.");
    (void)deviceId;
    (void)offset;
    return RT_SUCCESS;
}

RtError StubGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t* const logicDevId)
{
    ADAPTER_LOGD("Enter stub function of GetLogicDevIdByUserDevId.");
    (void)userDevId;
    (void)logicDevId;
    return RT_SUCCESS;
}

RtError StubFuncGetByName(const RtBinHandle binHandle, const char_t* kernelName, RtFuncHandle* funcHandle)
{
    (void)binHandle;
    (void)kernelName;
    (void)funcHandle;
    return RT_SUCCESS;
}

RtError StubBinaryLoadFromFile(const char_t* const binPath, const RtLoadBinaryConfig* const optionalCfg,
                               RtBinHandle* handle)
{
    (void)binPath;
    (void)optionalCfg;
    (void)handle;
    return RT_SUCCESS;
}

RtError StubStreamCreate(RtStream* stm, int32_t priority)
{
    ADAPTER_LOGD("Enter stub function of StreamCreate.");
    (void)stm;
    (void)priority;
    return RT_SUCCESS;
}

RtError StubStreamDestroy(RtStream stm)
{
    ADAPTER_LOGD("Enter stub function of StreamDestroy.");
    (void)stm;
    return RT_SUCCESS;
}

RtError StubStreamAddToModel(RtStream stm, RtModel captureMdl)
{
    ADAPTER_LOGD("Enter stub function of StreamAddToModel.");
    (void)stm;
    (void)captureMdl;
    return RT_SUCCESS;
}

RtError StubStreamSynchronize(RtStream stm)
{
    ADAPTER_LOGD("Enter stub function of StreamSynchronize.");
    (void)stm;
    return RT_SUCCESS;
}

RtError StubDevBinaryUnRegister(void* handle)
{
    ADAPTER_LOGD("Enter stub function of DevBinaryUnRegister.");
    (void)handle;
    return RT_SUCCESS;
}

RtError StubRegisterAllKernel(const RtDevBinary* bin, void** hdl)
{
    ADAPTER_LOGD("Enter stub function of RegisterAllKernel.");
    (void)bin;
    (void)hdl;
    return RT_SUCCESS;
}

RtError StubDevBinaryRegister(const RtDevBinary* bin, void** hdl)
{
    ADAPTER_LOGD("Enter stub function of DevBinaryRegister.");
    (void)bin;
    (void)hdl;
    return RT_SUCCESS;
}

RtError StubFunctionRegister(void* binHandle, const void* stubFunc, const char_t* stubName, const void* kernelInfoExt,
                             uint32_t funcMode)
{
    ADAPTER_LOGD("Enter stub function of FunctionRegister.");
    (void)binHandle;
    (void)stubFunc;
    (void)stubName;
    (void)kernelInfoExt;
    (void)funcMode;
    return RT_SUCCESS;
}

RtError StubKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args, uint32_t argsSize, RtSmDesc* smDesc,
                         RtStream stm)
{
    ADAPTER_LOGD("Enter stub function of KernelLaunch.");
    (void)stubFunc;
    (void)blockDim;
    (void)args;
    (void)argsSize;
    (void)smDesc;
    (void)stm;
    return RT_SUCCESS;
}

RtError StubLaunchCpuKernel(const RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                            const RtKernelLaunchCfg* cfg, RtCpuKernelArgs* argsInfo)
{
    ADAPTER_LOGD("Enter stub function of LaunchCpuKernel.");
    (void)funcHandle;
    (void)numBlocks;
    (void)stm;
    (void)cfg;
    (void)argsInfo;
    return RT_SUCCESS;
}

RtError StubKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t numBlocks, RtArgsEx* argsInfo,
                                     RtSmDesc* smDesc, RtStream stm, const RtTaskCfgInfo* cfgInfo)
{
    ADAPTER_LOGD("Enter stub function of KernelLaunchWithHandleV2.");
    (void)hdl;
    (void)tilingKey;
    (void)numBlocks;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)cfgInfo;
    return RT_SUCCESS;
}

RtError StubAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t* const opName, const uint32_t numBlocks,
                                        const RtAicpuArgsEx* argsInfo, RtSmDesc* const smDesc, const RtStream stm,
                                        const uint32_t flags)
{
    ADAPTER_LOGD("Enter stub function of AicpuKernelLaunchExWithArgs.");
    (void)kernelType;
    (void)opName;
    (void)numBlocks;
    (void)argsInfo;
    (void)smDesc;
    (void)stm;
    (void)flags;
    return RT_SUCCESS;
}

RtError StubLaunchKernelWithHostArgs(RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm, RtKernelLaunchCfg* cfg,
                                     void* hostArgs, uint32_t argsSize, RtHostInputInfo* placeHolderArray,
                                     uint32_t placeHolderNum)
{
    ADAPTER_LOGD("Enter stub function of LaunchKernelWithHostArgs.");
    (void)funcHandle;
    (void)numBlocks;
    (void)stm;
    (void)cfg;
    (void)hostArgs;
    (void)argsSize;
    (void)placeHolderArray;
    (void)placeHolderNum;
    return RT_SUCCESS;
}
} // namespace npu::tile_fwk
