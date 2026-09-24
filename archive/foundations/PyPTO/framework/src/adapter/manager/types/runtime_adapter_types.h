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
 * \file runtime_adapter_types.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class RuntimeFunc {
    Malloc = 0,
    rtMemset,
    MemCopy,
    MemCopyAsync,
    Free,
    SetDevice,
    GetDevice,
    GetSocSpec,
    GetSocVersion,
    GetAiCpuCount,
    GetL2CacheOffset,
    GetLogicDevIdByUserDevId,
    StreamCreate,
    StreamDestroy,
    StreamAddToModel,
    StreamSynchronize,
    DevBinaryUnRegister,
    RegisterAllKernel,
    LaunchCpuKernel,
    KernelLaunchWithHandleV2,
    AicpuKernelLaunchExWithArgs,
    DevBinaryRegister,
    FunctionRegister,
    KernelLaunch,
    LaunchKernelWithHostArgs,
    GetExceptionRegInfo,
    Bottom
};
const std::string kRuntimeLibName = "libruntime.so";
const std::map<RuntimeFunc, std::string> kRuntimeFuncStrMap{
    {RuntimeFunc::Malloc, "rtMalloc"},
    {RuntimeFunc::rtMemset, "rtMemset"},
    {RuntimeFunc::MemCopy, "rtMemcpy"},
    {RuntimeFunc::MemCopyAsync, "rtMemcpyAsync"},
    {RuntimeFunc::Free, "rtFree"},
    {RuntimeFunc::SetDevice, "rtSetDevice"},
    {RuntimeFunc::GetDevice, "rtGetDevice"},
    {RuntimeFunc::GetSocSpec, "rtGetSocSpec"},
    {RuntimeFunc::GetSocVersion, "rtGetSocVersion"},
    {RuntimeFunc::GetAiCpuCount, "rtGetAiCpuCount"},
    {RuntimeFunc::GetL2CacheOffset, "rtGetL2CacheOffset"},
    {RuntimeFunc::GetLogicDevIdByUserDevId, "rtGetLogicDevIdByUserDevId"},
    {RuntimeFunc::StreamCreate, "rtStreamCreate"},
    {RuntimeFunc::StreamDestroy, "rtStreamDestroy"},
    {RuntimeFunc::StreamAddToModel, "rtStreamAddToModel"},
    {RuntimeFunc::StreamSynchronize, "rtStreamSynchronize"},
    {RuntimeFunc::DevBinaryUnRegister, "rtDevBinaryUnRegister"},
    {RuntimeFunc::RegisterAllKernel, "rtRegisterAllKernel"},
    {RuntimeFunc::LaunchCpuKernel, "rtsLaunchCpuKernel"},
    {RuntimeFunc::KernelLaunchWithHandleV2, "rtKernelLaunchWithHandleV2"},
    {RuntimeFunc::AicpuKernelLaunchExWithArgs, "rtAicpuKernelLaunchExWithArgs"},
    {RuntimeFunc::DevBinaryRegister, "rtDevBinaryRegister"},
    {RuntimeFunc::FunctionRegister, "rtFunctionRegister"},
    {RuntimeFunc::KernelLaunch, "rtKernelLaunch"},
    {RuntimeFunc::LaunchKernelWithHostArgs, "rtsLaunchKernelWithHostArgs"},
    {RuntimeFunc::GetExceptionRegInfo, "rtGetExceptionRegInfo"},
};
} // namespace npu::tile_fwk
