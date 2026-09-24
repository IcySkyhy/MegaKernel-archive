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
 * \file runtime_api.h
 * \brief
 */

#pragma once

#include "adapter/api/runtime_define.h"
#include "adapter/api/acl_define.h"

namespace npu::tile_fwk {
RtError RuntimeMalloc(void** devPtr, uint64_t size, RtMemType type, const uint16_t moduleId);
RtError RuntimeMemset(void* devPtr, uint64_t destMax, uint32_t val, uint64_t cnt);
RtError RuntimeMemcpyDirect(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind);
RtError RuntimeMemcpyDirectAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind,
                                 RtStream stm);

RtError RuntimeFree(void* devPtr);

RtError RuntimeSetDevice(int32_t devId);
RtError RuntimeGetDevice(int32_t* devId);
RtError RuntimeGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
RtError RuntimeGetSocVersion(char_t* ver, const uint32_t maxLen);
RtError RuntimeGetAiCpuCount(uint32_t* aiCpuCnt);
RtError RuntimeGetL2CacheOffset(uint32_t deviceId, uint64_t* offset);
RtError RuntimeGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t* const logicDevId);

RtError RuntimeFuncGetByName(const RtBinHandle binHandle, const char_t* kernelName, RtFuncHandle* funcHandle);

RtError RuntimeBinaryLoadFromFile(const char_t* const binPath, const RtLoadBinaryConfig* const optionalCfg,
                                  RtBinHandle* handle);

RtError RuntimeStreamCreate(RtStream* stm, int32_t priority);
RtError RuntimeStreamDestroy(RtStream stm);
RtError RuntimeStreamAddToModel(RtStream stm, RtModel captureMdl);
RtError RuntimeStreamSynchronize(RtStream stm);

RtError RuntimeDevBinaryUnRegister(void* handle);
RtError RuntimeRegisterAllKernel(const RtDevBinary* bin, void** hdl);

RtError RuntimeKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t numBlocks, RtArgsEx* argsInfo,
                                        RtSmDesc* smDesc, RtStream stm, const RtTaskCfgInfo* cfgInfo);

RtError RuntimeLaunchCpuKernel(const RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                               const RtKernelLaunchCfg* cfg, RtCpuKernelArgs* argsInfo);

RtError RuntimeAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t* const opName,
                                           const uint32_t numBlocks, const RtAicpuArgsEx* argsInfo,
                                           RtSmDesc* const smDesc, const RtStream stm, const uint32_t flags);

RtError RuntimeLaunchKernelWithHostArgs(RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                                        RtKernelLaunchCfg* cfg, void* hostArgs, uint32_t argsSize,
                                        RtHostInputInfo* placeHolderArray, uint32_t placeHolderNum);
RtError RuntimeDevBinaryRegister(const RtDevBinary* bin, void** hdl);
RtError RuntimeFunctionRegister(void* binHandle, const void* stubFunc, const char_t* stubName,
                                const void* kernelInfoExt, uint32_t funcMode);
RtError RuntimeKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args, uint32_t argsSize, RtSmDesc* smDesc,
                            RtStream stm);
RtError RuntimeGeExceptionRegInfo(RtExceptionInfo* exceptionInfo, RtExceptionRegInfo* execptionReg);
} // namespace npu::tile_fwk
