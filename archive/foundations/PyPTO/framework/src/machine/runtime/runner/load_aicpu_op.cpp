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
 * \file load_aicpu_op.cpp
 * \brief
 */

#include "machine/runtime/runner/load_aicpu_op.h"

#include <fstream>
#include <limits.h>
#include "tilefwk/pypto_fwk_log.h"
#include "utils/file_utils.h"
#include "interface/utils/op_info_manager.h"
#include "machine/runtime/runner/runtime_utils.h"
#include "tilefwk/error_code.h"

namespace {
const std::string ControlFlowLaunchKernelName = "batchLoadsoFrombuf";
const std::string ControlFlowKernelSoName = "libcontrol_flow.so";
constexpr int BuiltInOpNum = 2;
std::string BuiltInFunName[BuiltInOpNum] = {"PyptoInit", "PyptoRun"};
} // namespace

namespace npu::tile_fwk {
namespace {
RtAicpuArgsEx BuildRtAicpuArgs(const AicpuLaunchDesc& desc)
{
    RtAicpuArgsEx rt{};
    rt.args = desc.args;
    rt.argsSize = desc.argsSize;
    rt.hostInputInfoPtr = reinterpret_cast<RtHostInputInfo*>(desc.hostInputs);
    rt.hostInputInfoNum = desc.hostInputNum;
    rt.timeout = desc.timeout;
    rt.kernelNameAddrOffset = desc.kernelNameOffset;
    rt.soNameAddrOffset = desc.soNameOffset;
    return rt;
}
} // namespace

void LoadAicpuOp::CustomAiCpuSoLoad()
{
#ifdef BUILD_WITH_NEW_CANN
    RtLoadBinaryConfig optionCfg;
    auto loadBinOptions = std::make_unique<RtLoadBinaryOption>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RtLoadBinaryOptionType::CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 1;
    optionCfg.numOpt = 1;
    std::string customOpJsonPath = OpInfoManager::GetInstance().GetCustomOpJsonPath();
    if (RealPath(customOpJsonPath).empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "Custom op json path is empty");
        return;
    }
    customBinHandle_ = OpInfoManager::GetInstance().GetControlBinHandle(customOpJsonPath);
    if (customBinHandle_ != nullptr) {
        return;
    }
    auto ret = RuntimeBinaryLoadFromFile(customOpJsonPath.c_str(), &optionCfg,
                                         reinterpret_cast<void**>(&customBinHandle_));
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_LOAD_FAILED, "Load aicpu json failed ret is %d", ret);
    }
    OpInfoManager::GetInstance().SetControlBinHandle(customBinHandle_);
#endif
}

int LoadAicpuOp::AicpuKernelLaunch(void* funcHandle, const AicpuLaunchDesc& desc)
{
    RtFuncHandle aicpuFuncHandle = static_cast<RtFuncHandle>(funcHandle);

    RtCpuKernelArgs argInfo;
    memset_s(&argInfo, sizeof(argInfo), 0, sizeof(argInfo));
    argInfo.baseArgs = BuildRtAicpuArgs(desc);
    RtLaunchKernelAttr launchKernelAttr{};
    RtKernelLaunchCfg kernelLaunchCfg = {&launchKernelAttr, 0U};
    return RuntimeLaunchCpuKernel(aicpuFuncHandle, desc.blockDim, desc.stream, &kernelLaunchCfg, &argInfo);
}

int LoadAicpuOp::LaunchWithHostArgs(void* funcHandle, const AicpuLaunchDesc& desc)
{
    RtFuncHandle aicpuFuncHandle = static_cast<RtFuncHandle>(funcHandle);
    RtLaunchKernelAttr launchKernelAttr{};
    RtKernelLaunchCfg kernelLaunchCfg = {&launchKernelAttr, 0U};
    if (desc.timeout > 0) {
        launchKernelAttr.id = RtLaunchKernelAttrId::TIMEOUT;
        launchKernelAttr.value.timeout = desc.timeout;
        kernelLaunchCfg.numAttrs = 1U;
    }
    return RuntimeLaunchKernelWithHostArgs(aicpuFuncHandle, desc.blockDim, desc.stream, &kernelLaunchCfg, desc.args,
                                           desc.argsSize, reinterpret_cast<RtHostInputInfo*>(desc.hostInputs),
                                           desc.hostInputNum);
}

int LoadAicpuOp::LaunchCustomOp([[maybe_unused]] const AicpuLaunchDesc& desc,
                                [[maybe_unused]] std::string& OpType) const
{
#ifdef BUILD_WITH_NEW_CANN
    ASSERT(DevCommonErr::PARAM_INVALID, customBinHandle_ != nullptr) << "customBinHandle cannot be null";
    RtFuncHandle custFuncHandle;
    auto ret = RuntimeFuncGetByName(customBinHandle_, OpType.c_str(), &custFuncHandle);
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "Get OpType[%s] funcHandle failed ret[%d]", OpType.c_str(), ret);
        return ret;
    }
    return AicpuKernelLaunch(custFuncHandle, desc);
#else
    return 0;
#endif
}

int LoadAicpuOp::LaunchAicpuServerInit(int64_t* devArgsAddr)
{
    auto aicpuStream = GetStreamContext().GetScheStream();
    DeviceKernelArgs kArgs;
    AicpuLaunchDesc launchDesc;
    kArgs.cfgdata = devArgsAddr;
    launchDesc.stream = aicpuStream;
    launchDesc.args = reinterpret_cast<AiCpuArgs*>(&kArgs);
    launchDesc.argsSize = sizeof(DeviceKernelArgs);
    launchDesc.blockDim = 1U;
    auto ret = LaunchBuiltInOpWithHostArgs(launchDesc, "PyptoInit");
    if (ret != 0) {
        MACHINE_LOGE(0, "kernel_launch init is failed");
        return ret;
    }
    return RuntimeStreamSynchronize(aicpuStream);
}

int LoadAicpuOp::GetBuiltInOpBinHandle(int64_t* devArgsAddr)
{
    builtInOpJsonPath_ = GetPyptoLibPath() + "/pypto_op_info.json";
    if (RealPath(builtInOpJsonPath_).empty()) {
        MACHINE_LOGE(DevCommonErr::FILE_ERROR, "JsonPath[%s] is empty", builtInOpJsonPath_.c_str());
        return static_cast<int>(DevCommonErr::FILE_ERROR);
    }
    RtLoadBinaryConfig optionCfg;
    auto loadBinOptions = std::make_unique<RtLoadBinaryOption>();

    optionCfg.options = loadBinOptions.get();
    optionCfg.options->optionId = RtLoadBinaryOptionType::CPU_KERNEL_MODE;
    optionCfg.options->value.cpuKernelMode = 0;
    optionCfg.numOpt = 1;
    void* binHandle;
    auto ret = RuntimeBinaryLoadFromFile(builtInOpJsonPath_.c_str(), &optionCfg, reinterpret_cast<void**>(&binHandle));
    if (ret != 0) {
        MACHINE_LOGE(RtErr::RT_LOAD_FAILED, "Get built in bin handle failed");
        return static_cast<int>(RtErr::RT_LOAD_FAILED);
    }

    for (int i = 0; i < BuiltInOpNum; i++) {
        RtFuncHandle funcHandle;
        ret = RuntimeFuncGetByName(binHandle, BuiltInFunName[i].c_str(), &funcHandle);
        if (ret != 0) {
            MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "Get BuiltIn FuncName[%s] funcHandle failed ret[%d]",
                         BuiltInFunName[i].c_str(), ret);
            return ret;
        }
        builtInFuncMap_[BuiltInFunName[i]] = funcHandle;
    }
    return LaunchAicpuServerInit(devArgsAddr);
}

int LoadAicpuOp::LaunchBuiltInOpWithHostArgs(const AicpuLaunchDesc& desc, const std::string& funcName) const
{
    RtFuncHandle funcHandle;
    auto it = builtInFuncMap_.find(funcName);
    if (it != builtInFuncMap_.end()) {
        funcHandle = it->second;
    } else {
        MACHINE_LOGE(RtErr::RT_GET_FUNC_FAILED, "The func name[%s] is invalid", funcName.c_str());
        return static_cast<int>(RtErr::RT_GET_FUNC_FAILED);
    }
    return LaunchWithHostArgs(funcHandle, desc);
}
} // namespace npu::tile_fwk
