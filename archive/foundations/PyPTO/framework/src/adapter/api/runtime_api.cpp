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
 * \file runtime_api.cpp
 * \brief
 */

#include "adapter/api/runtime_api.h"

#include "adapter/api/acl_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#include <type_traits>
#include "runtime/base.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "runtime/rts/rts_kernel.h"
#endif
#include "adapter/stubs/runtime_stubs.h"

namespace npu::tile_fwk {
RtError RuntimeMalloc(void** devPtr, uint64_t size, RtMemType type, const uint16_t moduleId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Malloc);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            void**, uint64_t, rtMemType_t,
            const uint16_t) = reinterpret_cast<rtError_t (*)(void**, uint64_t, rtMemType_t, const uint16_t)>(func);
        return runtimeFunc(devPtr, size, static_cast<rtMemType_t>(type), moduleId);
    }
#endif
    return StubMalloc(devPtr, size, type, moduleId);
}

RtError RuntimeMemset(void* devPtr, uint64_t destMax, uint32_t val, uint64_t cnt)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::rtMemset);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(void*, uint64_t, uint32_t,
                                 uint64_t) = reinterpret_cast<rtError_t (*)(void*, uint64_t, uint32_t, uint64_t)>(func);
        return runtimeFunc(devPtr, destMax, val, cnt);
    }
#endif
    return StubMemset(devPtr, destMax, val, cnt);
}

RtError RuntimeMemcpyDirect(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopy);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(void*, uint64_t, const void*, uint64_t,
                                 rtMemcpyKind_t) = reinterpret_cast<rtError_t (*)(void*, uint64_t, const void*,
                                                                                  uint64_t, rtMemcpyKind_t)>(func);
        return runtimeFunc(dst, destMax, src, cnt, static_cast<rtMemcpyKind_t>(kind));
    }
#endif
    return StubMemcpy(dst, destMax, src, cnt, kind);
}

RtError RuntimeMemcpyDirectAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind,
                                 RtStream stm)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::MemCopyAsync);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(void*, uint64_t, const void*, uint64_t, rtMemcpyKind_t,
                                 rtStream_t) = reinterpret_cast<rtError_t (*)(void*, uint64_t, const void*, uint64_t,
                                                                              rtMemcpyKind_t, rtStream_t)>(func);
        return runtimeFunc(dst, destMax, src, cnt, static_cast<rtMemcpyKind_t>(kind), stm);
    }
#endif
    return StubMemcpyAsync(dst, destMax, src, cnt, kind, stm);
}

RtError RuntimeFree(void* devPtr)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::Free);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(void*) = reinterpret_cast<rtError_t (*)(void*)>(func);
        return runtimeFunc(devPtr);
    }
#endif
    return StubFree(devPtr);
}

RtError RuntimeSetDevice(int32_t devId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::SetDevice);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(int32_t) = reinterpret_cast<rtError_t (*)(int32_t)>(func);
        return runtimeFunc(devId);
    }
#endif
    return StubSetDevice(devId);
}

RtError RuntimeGetDevice(int32_t* devId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetDevice);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(int32_t*) = reinterpret_cast<rtError_t (*)(int32_t*)>(func);
        return runtimeFunc(devId);
    }
#endif
    return StubGetDevice(devId);
}

RtError RuntimeGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocSpec);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            const char*, const char*, char*,
            const uint32_t) = reinterpret_cast<rtError_t (*)(const char*, const char*, char*, const uint32_t)>(func);
        return runtimeFunc(label, key, val, maxLen);
    }
#endif
    return StubGetSocSpec(label, key, val, maxLen);
}

RtError RuntimeGetSocVersion(char_t* ver, const uint32_t maxLen)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetSocVersion);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(char_t*,
                                 const uint32_t) = reinterpret_cast<rtError_t (*)(char_t*, const uint32_t)>(func);
        return runtimeFunc(ver, maxLen);
    }
#endif
    return StubGetSocVersion(ver, maxLen);
}

RtError RuntimeGetAiCpuCount(uint32_t* aiCpuCnt)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetAiCpuCount);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(uint32_t*) = reinterpret_cast<rtError_t (*)(uint32_t*)>(func);
        return runtimeFunc(aiCpuCnt);
    }
#endif
    return StubGetAiCpuCount(aiCpuCnt);
}

RtError RuntimeGetL2CacheOffset(uint32_t deviceId, uint64_t* offset)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetL2CacheOffset);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(uint32_t, uint64_t*) = reinterpret_cast<rtError_t (*)(uint32_t, uint64_t*)>(func);
        return runtimeFunc(deviceId, offset);
    }
#endif
    return StubGetL2CacheOffset(deviceId, offset);
}

RtError RuntimeGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t* const logicDevId)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetLogicDevIdByUserDevId);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(const int32_t,
                                 int32_t* const) = reinterpret_cast<rtError_t (*)(const int32_t, int32_t* const)>(func);
        return runtimeFunc(userDevId, logicDevId);
    }
#endif
    return StubGetLogicDevIdByUserDevId(userDevId, logicDevId);
}

RtError RuntimeFuncGetByName(const RtBinHandle binHandle, const char_t* kernelName, RtFuncHandle* funcHandle)
{
#ifdef BUILD_WITH_CANN
    return AclRtBinaryGetFunction(reinterpret_cast<AclRtBinHandle>(binHandle), kernelName,
                                  reinterpret_cast<AclRtFuncHandle*>(funcHandle));
#endif
    return StubFuncGetByName(binHandle, kernelName, funcHandle);
}

RtError RuntimeBinaryLoadFromFile(const char_t* const binPath, const RtLoadBinaryConfig* const optionalCfg,
                                  RtBinHandle* handle)
{
#ifdef BUILD_WITH_CANN
    return AclRtBinaryLoadFromFile(
        binPath, reinterpret_cast<AclRtBinaryLoadOptions*>(const_cast<RtLoadBinaryConfig*>(optionalCfg)),
        reinterpret_cast<AclRtBinHandle*>(handle));
#endif
    return StubBinaryLoadFromFile(binPath, optionalCfg, handle);
}

RtError RuntimeStreamCreate(RtStream* stm, int32_t priority)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamCreate);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(rtStream_t*, int32_t) = reinterpret_cast<rtError_t (*)(rtStream_t*, int32_t)>(func);
        return runtimeFunc(stm, priority);
    }
#endif
    return StubStreamCreate(stm, priority);
}

RtError RuntimeStreamDestroy(RtStream stm)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamDestroy);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(rtStream_t) = reinterpret_cast<rtError_t (*)(rtStream_t)>(func);
        return runtimeFunc(stm);
    }
#endif
    return StubStreamDestroy(stm);
}

RtError RuntimeStreamAddToModel(RtStream stm, RtModel captureMdl)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamAddToModel);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(rtStream_t, rtModel_t) = reinterpret_cast<rtError_t (*)(rtStream_t, rtModel_t)>(func);
        return runtimeFunc(stm, captureMdl);
    }
#endif
    return StubStreamAddToModel(stm, captureMdl);
}

RtError RuntimeStreamSynchronize(RtStream stm)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::StreamSynchronize);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(rtStream_t) = reinterpret_cast<rtError_t (*)(rtStream_t)>(func);
        return runtimeFunc(stm);
    }
#endif
    return StubStreamSynchronize(stm);
}

RtError RuntimeDevBinaryUnRegister(void* handle)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryUnRegister);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(void*) = reinterpret_cast<rtError_t (*)(void*)>(func);
        return runtimeFunc(handle);
    }
#endif
    return StubDevBinaryUnRegister(handle);
}

RtError RuntimeRegisterAllKernel(const RtDevBinary* bin, void** hdl)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::RegisterAllKernel);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(const rtDevBinary_t*,
                                 void**) = reinterpret_cast<rtError_t (*)(const rtDevBinary_t*, void**)>(func);
        return runtimeFunc(reinterpret_cast<const rtDevBinary_t*>(bin), hdl);
    }
#endif
    return StubRegisterAllKernel(bin, hdl);
}

RtError RuntimeDevBinaryRegister(const RtDevBinary* bin, void** hdl)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::DevBinaryRegister);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(const rtDevBinary_t*,
                                 void**) = reinterpret_cast<rtError_t (*)(const rtDevBinary_t*, void**)>(func);
        return runtimeFunc(reinterpret_cast<const rtDevBinary_t*>(bin), hdl);
    }
#endif
    return StubDevBinaryRegister(bin, hdl);
}

RtError RuntimeFunctionRegister(void* binHandle, const void* stubFunc, const char_t* stubName,
                                const void* kernelInfoExt, uint32_t funcMode)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::FunctionRegister);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            void*, const void*, const char_t*, const void*,
            uint32_t) = reinterpret_cast<rtError_t (*)(void*, const void*, const char_t*, const void*, uint32_t)>(func);
        return runtimeFunc(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
    }
#endif
    return StubFunctionRegister(binHandle, stubFunc, stubName, kernelInfoExt, funcMode);
}

RtError RuntimeKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args, uint32_t argsSize, RtSmDesc* smDesc,
                            RtStream stm)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunch);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(const void*, uint32_t, void*, uint32_t, rtSmDesc_t*,
                                 rtStream_t) = reinterpret_cast<rtError_t (*)(const void*, uint32_t, void*, uint32_t,
                                                                              rtSmDesc_t*, rtStream_t)>(func);
        return runtimeFunc(stubFunc, blockDim, args, argsSize, reinterpret_cast<rtSmDesc_t*>(smDesc), stm);
    }
#endif
    return StubKernelLaunch(stubFunc, blockDim, args, argsSize, smDesc, stm);
}

RtError RuntimeLaunchCpuKernel(const RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                               const RtKernelLaunchCfg* cfg, RtCpuKernelArgs* argsInfo)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::LaunchCpuKernel);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            const rtFuncHandle, uint32_t, rtStream_t, const rtKernelLaunchCfg_t*,
            rtCpuKernelArgs_t*) = reinterpret_cast<rtError_t (*)(const rtFuncHandle, uint32_t, rtStream_t,
                                                                 const rtKernelLaunchCfg_t*, rtCpuKernelArgs_t*)>(func);
        return runtimeFunc(funcHandle, numBlocks, stm, reinterpret_cast<const rtKernelLaunchCfg_t*>(cfg),
                           reinterpret_cast<rtCpuKernelArgs_t*>(argsInfo));
    }
#endif
    return StubLaunchCpuKernel(funcHandle, numBlocks, stm, cfg, argsInfo);
}

RtError RuntimeKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t numBlocks, RtArgsEx* argsInfo,
                                        RtSmDesc* smDesc, RtStream stm, const RtTaskCfgInfo* cfgInfo)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::KernelLaunchWithHandleV2);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            void*, const uint64_t, uint32_t, rtArgsEx_t*, rtSmDesc_t*, rtStream_t,
            const rtTaskCfgInfo_t*) = reinterpret_cast<rtError_t (*)(void*, const uint64_t, uint32_t, rtArgsEx_t*,
                                                                     rtSmDesc_t*, rtStream_t, const rtTaskCfgInfo_t*)>(
            func);
        return runtimeFunc(hdl, tilingKey, numBlocks, reinterpret_cast<rtArgsEx_t*>(argsInfo),
                           reinterpret_cast<rtSmDesc_t*>(smDesc), stm,
                           reinterpret_cast<const rtTaskCfgInfo_t*>(cfgInfo));
    }
#endif
    return StubKernelLaunchWithHandleV2(hdl, tilingKey, numBlocks, argsInfo, smDesc, stm, cfgInfo);
}

RtError RuntimeAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t* const opName,
                                           const uint32_t numBlocks, const RtAicpuArgsEx* argsInfo,
                                           RtSmDesc* const smDesc, const RtStream stm, const uint32_t flags)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::AicpuKernelLaunchExWithArgs);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            const uint32_t, const char_t* const, const uint32_t, const rtAicpuArgsEx_t*, rtSmDesc_t* const,
            const rtStream_t,
            const uint32_t) = reinterpret_cast<rtError_t (*)(const uint32_t, const char_t* const, const uint32_t,
                                                             const rtAicpuArgsEx_t*, rtSmDesc_t* const,
                                                             const rtStream_t, const uint32_t)>(func);
        return runtimeFunc(kernelType, opName, numBlocks, reinterpret_cast<const rtAicpuArgsEx_t*>(argsInfo),
                           reinterpret_cast<rtSmDesc_t*>(smDesc), stm, flags);
    }
#endif
    return StubAicpuKernelLaunchExWithArgs(kernelType, opName, numBlocks, argsInfo, smDesc, stm, flags);
}

RtError RuntimeLaunchKernelWithHostArgs(RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                                        RtKernelLaunchCfg* cfg, void* hostArgs, uint32_t argsSize,
                                        RtHostInputInfo* placeHolderArray, uint32_t placeHolderNum)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::LaunchKernelWithHostArgs);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            rtFuncHandle, uint32_t, rtStream_t, rtKernelLaunchCfg_t*, void*, uint32_t, rtPlaceHolderInfo_t*,
            uint32_t) = reinterpret_cast<rtError_t (*)(rtFuncHandle, uint32_t, rtStream_t, rtKernelLaunchCfg_t*, void*,
                                                       uint32_t, rtPlaceHolderInfo_t*, uint32_t)>(func);
        return runtimeFunc(funcHandle, numBlocks, stm, reinterpret_cast<rtKernelLaunchCfg_t*>(cfg), hostArgs, argsSize,
                           reinterpret_cast<rtPlaceHolderInfo_t*>(placeHolderArray), placeHolderNum);
    }
#endif
    return StubLaunchKernelWithHostArgs(funcHandle, numBlocks, stm, cfg, hostArgs, argsSize, placeHolderArray,
                                        placeHolderNum);
}

RtError RuntimeGeExceptionRegInfo(RtExceptionInfo* exceptionInfo, RtExceptionRegInfo* execptionReg)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetRuntimeAdapter().GetFunction(RuntimeFunc::GetExceptionRegInfo);
    if (func != nullptr) {
        rtError_t (*runtimeFunc)(
            RtExceptionInfo*, RtExceptionErrRegInfo**,
            uint32_t*) = reinterpret_cast<rtError_t (*)(RtExceptionInfo*, RtExceptionErrRegInfo**, uint32_t*)>(func);
        return runtimeFunc(exceptionInfo, &(execptionReg->errRegInfo), &execptionReg->coreNum);
    }
#else
    (void)exceptionInfo;
    if (execptionReg != nullptr) {
        static RtExceptionErrRegInfo errRegInfo = {1U, RtCoreType::RT_CORE_TYPE_AIC, 0U, 0U, {}};
        execptionReg->coreNum = 1U;
        execptionReg->errRegInfo = &errRegInfo;
    }
#endif
    return RT_SUCCESS;
}

#ifdef BUILD_WITH_CANN
static_assert(std::is_same<RtError, rtError_t>::value);
static_assert(std::is_same<RtMemType, rtMemType_t>::value);
static_assert(std::is_same<RtStream, rtStream_t>::value);
static_assert(std::is_same<RtModel, rtModel_t>::value);
static_assert(std::is_same<RtFuncHandle, rtFuncHandle>::value);
static_assert(std::is_same<RtBinHandle, rtBinHandle>::value);
static_assert(RT_SUCCESS == RT_ERROR_NONE);
static_assert(sizeof(RtDevBinary) == sizeof(rtDevBinary_t));
static_assert(sizeof(RtHostInputInfo) == sizeof(rtHostInputInfo_t));
static_assert(sizeof(RtHostInputInfo) == sizeof(rtPlaceHolderInfo_t));
static_assert(sizeof(RtArgsEx) == sizeof(rtArgsEx_t));
static_assert(sizeof(RtSmData) == sizeof(rtSmData_t));
static_assert(sizeof(RtSmDesc) == sizeof(rtSmDesc_t));
static_assert(sizeof(RtTaskCfgInfo) == sizeof(rtTaskCfgInfo_t));
static_assert(sizeof(RtAicpuArgsEx) == sizeof(rtAicpuArgsEx_t));
static_assert(sizeof(RtCpuKernelArgs) == sizeof(rtCpuKernelArgs_t));
static_assert(sizeof(RtTimeoutUs) == sizeof(rtTimeoutUs));
static_assert(sizeof(RtLaunchKernelAttrVal) == sizeof(rtLaunchKernelAttrVal_t));
static_assert(sizeof(RtKernelLaunchCfg) == sizeof(rtKernelLaunchCfg_t));
static_assert(sizeof(RtLoadBinaryOptionValue) == sizeof(rtLoadBinaryOptionValue_t));
static_assert(sizeof(RtLoadBinaryOption) == sizeof(rtLoadBinaryOption_t));
static_assert(sizeof(RtLoadBinaryConfig) == sizeof(rtLoadBinaryConfig_t));
static_assert(sizeof(RtArgsSizeInfo) == sizeof(rtArgsSizeInfo_t));
#endif
} // namespace npu::tile_fwk
