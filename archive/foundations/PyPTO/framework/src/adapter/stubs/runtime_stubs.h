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
 * \file runtime_stubs.h
 * \brief
 */

#pragma once

#include "adapter/api/runtime_define.h"

namespace npu::tile_fwk {
RtError StubMalloc(void** devPtr, uint64_t size, RtMemType type, const uint16_t moduleId);
RtError StubMemset(void* devPtr, uint64_t destMax, uint32_t val, uint64_t cnt);
RtError StubMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind);
RtError StubMemcpyAsync(void* dst, uint64_t destMax, const void* src, uint64_t cnt, RtMemcpyKind kind, RtStream stm);
RtError StubFree(void* devPtr);

RtError StubSetDevice(int32_t devId);
RtError StubGetDevice(int32_t* devId);
RtError StubGetSocSpec(const char* label, const char* key, char* val, const uint32_t maxLen);
RtError StubGetSocVersion(char_t* ver, const uint32_t maxLen);
RtError StubGetAiCpuCount(uint32_t* aiCpuCnt);
RtError StubGetL2CacheOffset(uint32_t deviceId, uint64_t* offset);
RtError StubGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t* const logicDevId);

RtError StubFuncGetByName(const RtBinHandle binHandle, const char_t* kernelName, RtFuncHandle* funcHandle);

RtError StubBinaryLoadFromFile(const char_t* const binPath, const RtLoadBinaryConfig* const optionalCfg,
                               RtBinHandle* handle);

RtError StubStreamCreate(RtStream* stm, int32_t priority);
RtError StubStreamDestroy(RtStream stm);
RtError StubStreamAddToModel(RtStream stm, RtModel captureMdl);
RtError StubStreamSynchronize(RtStream stm);

RtError StubDevBinaryUnRegister(void* handle);
RtError StubRegisterAllKernel(const RtDevBinary* bin, void** hdl);
RtError StubDevBinaryRegister(const RtDevBinary* bin, void** hdl);
RtError StubFunctionRegister(void* binHandle, const void* stubFunc, const char_t* stubName, const void* kernelInfoExt,
                             uint32_t funcMode);
RtError StubKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args, uint32_t argsSize, RtSmDesc* smDesc,
                         RtStream stm);

RtError StubLaunchCpuKernel(const RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm,
                            const RtKernelLaunchCfg* cfg, RtCpuKernelArgs* argsInfo);

RtError StubKernelLaunchWithHandleV2(void* hdl, const uint64_t tilingKey, uint32_t numBlocks, RtArgsEx* argsInfo,
                                     RtSmDesc* smDesc, RtStream stm, const RtTaskCfgInfo* cfgInfo);

RtError StubAicpuKernelLaunchExWithArgs(const uint32_t kernelType, const char_t* const opName, const uint32_t numBlocks,
                                        const RtAicpuArgsEx* argsInfo, RtSmDesc* const smDesc, const RtStream stm,
                                        const uint32_t flags);

RtError StubLaunchKernelWithHostArgs(RtFuncHandle funcHandle, uint32_t numBlocks, RtStream stm, RtKernelLaunchCfg* cfg,
                                     void* hostArgs, uint32_t argsSize, RtHostInputInfo* placeHolderArray,
                                     uint32_t placeHolderNum);
} // namespace npu::tile_fwk
