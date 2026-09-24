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
 * \file device_exception_dump.cpp
 * \brief
 */
#include <string>
#include <vector>
#include "device_exception_dump.h"
#include "tilefwk/pypto_fwk_log.h"
#include "adapter/api/runtime_api.h"
#include "interface/machine/device/tilefwk/aikernel_data.h"
#include "tilefwk/data_type.h"
#include "interface/utils/common.h"
#include "interface/program/program.h"
#include "runtime_utils.h"
#include "interface/configs/config_manager.h"
#include "utils/file_utils.h"
#include "machine/utils/dynamic/dev_encode_program.h"
#include "machine/runtime/bundle/pack/kernel_bundle_packer.h"
#include "machine/runtime/bundle/pack/kernel_bundle_pack.h"
#include "machine/utils/dynamic/dev_encode_function_param.h"
#include "machine/compile/aicore_compiler.h"
#include "interface/function/function.h"

using namespace npu::tile_fwk;
namespace npu::tile_fwk::dynamic {
constexpr int32_t MAX_AICPU_ARG_NUM = 7;

static bool PackAndDumpBundle(const std::vector<uint8_t>& kernelBinary, const DyndevFunctionAttribute* dynAttr,
                              const char* kernelName, const char* suffix)
{
    if (kernelBinary.empty() || dynAttr == nullptr || kernelName == nullptr) {
        return false;
    }
    bundle::KernelBundlePacker packer;
    packer.SetAicoreKernel(kernelBinary);
    packer.SetAicpuSo(ReadFile(GetPyptoLibPath() + "/libtilefwk_backend_server.so"));
    packer.SetDevProgram(dynAttr->devProgBinary);
    packer.SetSymbolMeta(bundle::SerializeWorkspaceSymbols(*dynAttr));

    std::string path = RealPath(config::LogTopFolder()) + "/" + kernelName + suffix + ".pyptokb";
    packer.Pack(path);
    MACHINE_LOGI("Dump pypto bundled kernel to file, file: %s.", path.c_str());
    return true;
}

void DumpBundledKernel(const DyndevFunctionAttribute* dynAttr, const char* kernelName)
{
    if (dynAttr == nullptr || kernelName == nullptr || dynAttr->devProgBinary.empty()) {
        return;
    }
    // DumpFile() needs host-absolute pointers, but .pyptokb must keep devProgBinary base-0.
    // Relocate only a copy so the bundle does not capture process-local host addresses.
    auto devProgForDump = dynAttr->devProgBinary;
    auto devProg = reinterpret_cast<DevAscendProgram*>(devProgForDump.data());
    devProg->RelocProgram(0, reinterpret_cast<uint64_t>(devProg));
    auto progtxtPath = RealPath(config::LogTopFolder()) + "/" + kernelName + "_program.tifwkbintxt";
    devProg->DumpFile(progtxtPath);
    MACHINE_LOGI("Dump pypto device program to file, file: %s", progtxtPath.c_str());

    if (dynAttr->kernelBinary.empty()) {
        MACHINE_LOGI("Skip bundled kernel dump: kernelBinary is empty for %s.", kernelName);
        return;
    }
    PackAndDumpBundle(dynAttr->kernelBinary, dynAttr, kernelName, "");
}

void DumpNoSubFuncBundledKernel(const DyndevFunctionAttribute* dynAttr, Function* func, const char* kernelName)
{
    if (dynAttr == nullptr || func == nullptr || kernelName == nullptr) {
        return;
    }
    if (dynAttr->devProgBinary.empty()) {
        MACHINE_LOGW("Skip no-subfunc bundled kernel dump: devProgBinary is empty for %s.", kernelName);
        return;
    }

    // 1. 恢复编译输入
    std::map<uint64_t, Function*> leafDict;
    for (auto leaf : dynAttr->funcGroup.devLeafList) {
        if (leaf == nullptr) {
            continue;
        }
        leafDict[leaf->ComputeHash().GetHash()] = leaf;
    }
    // 只重建编译必需的 calleeHashIndexDict（{hash -> leafIndex}），
    // 其余 param 字段（symbolTable/slot/outcast 等）仅链接器需要，编译流程不使用。
    dynamic::EncodeDevAscendFunctionParam param;
    for (const auto& leafIndex2Hash : dynAttr->devLeafIndex2Hash) {
        param.calleeHashIndexDict[leafIndex2Hash.second] = leafIndex2Hash.first;
    }

    const std::string ccePath = RealPath(config::GetEmitPath("kernel_aicore")) + "/";
    const std::string funcHash = std::to_string(func->GetFunctionHash().GetHash());
    const std::string funcRawName = func->GetOriginalRawName();

    // 2. 重新编译 __HAS_SUB_FUNC__ 未定义（enableSubFunc=false）的 kernel
    std::string kernelPath;
    int ret = CompileAICoreKernel(leafDict, param, ccePath, funcHash, funcRawName, kernelPath, false);
    if (ret != 0 || RealPath(kernelPath).empty()) {
        MACHINE_LOGW("Skip no-subfunc bundled kernel dump: recompile kernel failed for %s.", kernelName);
        return;
    }
    bool readOk = false;
    std::vector<uint8_t> noSubFuncKernel = ReadFile(kernelPath, &readOk);
    if (!readOk || noSubFuncKernel.empty()) {
        MACHINE_LOGW("Skip no-subfunc bundled kernel dump: failed to read recompiled kernel %s.", kernelPath.c_str());
        return;
    }

    // 3. 打包并 dump
    PackAndDumpBundle(noSubFuncKernel, dynAttr, kernelName, "_nosubfunc");
}

void GetTensorInfo(uint32_t inputSize, DevTensorData* tensorData, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    auto func = Program::GetInstance().GetLastFunction();
    if (func == nullptr) {
        MACHINE_LOGW("Function is nullptr not support to dump exception info");
        return;
    }
    auto dynAttr = func->GetDyndevAttribute();
    if (dynAttr == nullptr) {
        MACHINE_LOGW("dynAttr is nullptr not support to dump exception info");
        return;
    }
    auto& disableL2List = dynAttr->disableL2List;
    auto l2Offset = GetRuntimeL2Offset();
    if (inputSize > MAX_TENSOR_NUM) {
        inputSize = MAX_TENSOR_NUM;
        MACHINE_LOGW("Current funciton input is larger than %d", MAX_TENSOR_NUM);
    }
    for (uint32_t i = 0; i < inputSize; i++) {
        if (tensorData[i].address == 0) {
            MACHINE_LOGW("GetTensorInfo tensorData[%u].address is nullptr", i);
            continue;
        }
        exceptionDumpInfo->tensorInfo[i].tensorAddr = reinterpret_cast<int64_t*>(tensorData[i].address);
        // 双页tensor地址，需要进行还原
        if (disableL2List.size() && disableL2List[i]) {
            exceptionDumpInfo->tensorInfo[i].tensorAddr -= l2Offset;
        }
        exceptionDumpInfo->tensorInfo[i].dataType = DataType2CannType(static_cast<DataType>(tensorData[i].dataType));
        exceptionDumpInfo->tensorInfo[i].tensorSize = 1;
        for (int shapeIdx = 0; shapeIdx < tensorData[i].shape.dimSize; shapeIdx++) {
            exceptionDumpInfo->tensorInfo[i].shape.emplace_back(tensorData[i].shape.dim[shapeIdx]);
            exceptionDumpInfo->tensorInfo[i].tensorSize *= tensorData[i].shape.dim[shapeIdx];
        }
        exceptionDumpInfo->tensorInfo[i].tensorSize *= BitsOf(static_cast<DataType>(tensorData[i].dataType)) / 8;
    }
    exceptionDumpInfo->extraTensorNum = inputSize;
    DumpBundledKernel(dynAttr.get(), exceptionDumpInfo->kernelName);
    // 触发 __HAS_SUB_FUNC__ 未定义的二进制编译并打包 dump（失败仅 WARNING 降级）
    DumpNoSubFuncBundledKernel(dynAttr.get(), func, exceptionDumpInfo->kernelName);
}

int32_t GetAicoreExceptionDumpInfo(std::vector<void*> kernelArg, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    int64_t* tensor = static_cast<int64_t*>(kernelArg[4]);
    uint32_t tensorSize = tensor[0] - tensor[1];
    MACHINE_LOGD("GetAicoreExceptionDumpInfo: tensorSize=%u, outputTensorSize:[%ld]", tensorSize, tensor[1]);
    auto tensorData = (DevTensorData*)kernelArg[6];
    GetTensorInfo(tensorSize, tensorData, exceptionDumpInfo);
    return 0;
}

int32_t GetDeviceExceptionDumpInfo(RtAicoreExDetailInfo& aicoreExceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    auto kernelArgAddr = aicoreExceptionInfo.exceptionArgs.argAddr;
    auto argsSize = aicoreExceptionInfo.exceptionArgs.argsize;

    if (kernelArgAddr == nullptr) {
        MACHINE_LOGW("GetDeviceExceptionDumpInfo failed: kernelArgAddr is nullptr");
        return static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX);
    }

    auto aicoreArgsize = sizeof(void*) * MAX_AICPU_ARG_NUM;
    // Check it maybe pto
    if (argsSize != aicoreArgsize) {
        MACHINE_LOGI("GetDeviceExceptionDumpInfo failed: argsize not from pto info");
        return 0;
    }

    // memcpy D2H
    std::vector<void*> kernelArg(MAX_AICPU_ARG_NUM, nullptr);
    int rc = RuntimeMemcpyDirect(kernelArg.data(), argsSize, kernelArgAddr, argsSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (rc != 0) {
        MACHINE_LOGW("GetDeviceExceptionDumpInfo D2H memcpy failed: ret=%d", rc);
        return rc;
    }
    // kernel launch kernalArg 0: kernelName; 4 inputSize; 6 tensorData
    char* kernelName = static_cast<char*>(kernelArg[0]);
    // only support handle pto exception info
    if (kernelName != nullptr && strncmp(kernelName, "PyPTO", 5) != 0) {
        MACHINE_LOGI("Current exception info not PyPTO, which kernelName is[%s]", kernelName);
        return 0;
    }
    exceptionDumpInfo->argAddr = kernelArgAddr;
    exceptionDumpInfo->argssize = argsSize;
    auto exceptionKernelInfo = aicoreExceptionInfo.exceptionArgs.exceptionKernelInfo;
    MACHINE_LOGD("GetDeviceExceptionDumpInfo: kernelArgAddr=%p, argsSize=%u, binSize=%u, kernelName=%s", kernelArgAddr,
                 argsSize, exceptionKernelInfo.binSize,
                 exceptionKernelInfo.kernelName ? exceptionKernelInfo.kernelName : "(null)");
    return GetAicoreExceptionDumpInfo(kernelArg, exceptionDumpInfo);
}

void FillExceptionKernelName(const char* kernelName, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    if (kernelName != nullptr) {
        auto ret = strcpy_s(exceptionDumpInfo->kernelName, MAX_KERNEL_BUF_LEN, kernelName);
        if (ret != 0) {
            MACHINE_LOGW("Mem cpy KernelName from exceptionKernelInfo failed");
        }
        ret = strcpy_s(exceptionDumpInfo->kernelDisplayName, MAX_KERNEL_BUF_LEN, kernelName);
        if (ret != 0) {
            MACHINE_LOGW("Mem cpy kernelDisplayName from exceptionKernelInfo failed");
        }
    }
}

int32_t FillCoreExceptionInfo(RtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                              uint32_t exceptionDumpSize)
{
    RtExceptionRegInfo exceptionRegInfo = {0, nullptr};
    auto ret = RuntimeGeExceptionRegInfo(exceptionInfo, &exceptionRegInfo);
    if (ret == 0 && exceptionRegInfo.errRegInfo != nullptr && (exceptionDumpSize == exceptionRegInfo.coreNum)) {
        auto aicoreExceptionInfo = exceptionInfo->expandInfo.u.aicoreInfo;
        auto exceptionKernelInfo = aicoreExceptionInfo.exceptionArgs.exceptionKernelInfo;
        auto aicoreBin = exceptionKernelInfo.bin;
        for (uint32_t i = 0; i < exceptionDumpSize; i++) {
            exceptionDumpInfo[i].coreId = exceptionRegInfo.errRegInfo[i].coreId;
            exceptionDumpInfo[i].coreType = exceptionRegInfo.errRegInfo[i].coreType;
            exceptionDumpInfo[i].bin = aicoreBin;
            FillExceptionKernelName(exceptionKernelInfo.kernelName, &exceptionDumpInfo[i]);
            MACHINE_LOGD("Current No[%u] exception from %s coreId: %u", i,
                         exceptionDumpInfo->coreType == RtCoreType::RT_CORE_TYPE_AIC ? "AIC" : "AIV",
                         exceptionDumpInfo->coreId);
        }
        return 0;
    }
    MACHINE_LOGW("Cannot Get ExceptionRegInfo, which CoreType coreId would not support");
    return static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX);
}

int32_t GetAicpuExceptionDumpInfo(RtAicpuExDetailInfo& aicpuExcepitionInfo, AdxExceptionDumpInfo* exceptionDumpInfo)
{
    auto kernelArgAddr = aicpuExcepitionInfo.argAddr;
    auto argSize = aicpuExcepitionInfo.argsize;

    if (kernelArgAddr == nullptr) {
        MACHINE_LOGW("GetAicpuExceptionDumpInfo failed: kernelArgAddr is nullptr");
        return static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX);
    }

    if (argSize == 0) {
        MACHINE_LOGD("Aicpu kernelArgs is not suitable pypto");
        return 0;
    }
    // aicpu Op whose functionName == kerneName
    if (aicpuExcepitionInfo.functionName != nullptr &&
        strncmp(aicpuExcepitionInfo.functionName, "DynTileFwkKernelServer", strlen("DynTileFwkKernelServer")) != 0) {
        MACHINE_LOGI("Current exception info is not PyPTO");
        return 0;
    }

    std::string aicpuPyptoName = "PyPTO_Aicpu_" + Program::GetInstance().GetLastFunction()->GetOriginalRawName();
    FillExceptionKernelName(aicpuPyptoName.c_str(), exceptionDumpInfo);
    MACHINE_LOGI("Current argAddr is %p, argSize: %u", kernelArgAddr, argSize);
    std::vector<uint8_t> kernelArg(argSize);
    int rc = RuntimeMemcpyDirect(kernelArg.data(), argSize, kernelArgAddr, argSize, RtMemcpyKind::DEVICE_TO_HOST);
    if (rc != 0) {
        MACHINE_LOGE(npu::tile_fwk::MachineError::DUMP_DFX, "Aicpu exception info D2H memcpy failed: ret=%d", rc);
        return rc;
    }

    npu::tile_fwk::AiCpuArgs* aicpuArgs = (AiCpuArgs*)kernelArg.data();
    // device kernelArgs
    [[maybe_unused]] DeviceKernelArgs deviceKernelArgs = aicpuArgs->kArgs;
    // tensor info
    int64_t* tensorInfo = (int64_t*)(aicpuArgs + 1);
    int64_t inputSize = tensorInfo[0];
    auto tensorData = (DevTensorData*)(tensorInfo + 2);
    GetTensorInfo(inputSize, tensorData, exceptionDumpInfo);
    return 0;
}

int32_t DeviceExceptionDumpCallBack(RtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                                    uint32_t exceptionDumpSize)
{
    auto expandInfo = exceptionInfo->expandInfo;
    MACHINE_LOGD("DeviceExceptionDumpCallBack: expandInfo.type=%d", static_cast<int>(expandInfo.type));
    if (expandInfo.type == RtExceptionExpandType::AICORE) {
        auto ret = FillCoreExceptionInfo(exceptionInfo, exceptionDumpInfo, exceptionDumpSize);
        if (ret != 0) {
            return ret;
        }
        return GetDeviceExceptionDumpInfo(expandInfo.u.aicoreInfo, &exceptionDumpInfo[0]);
    }
    if (expandInfo.type == RtExceptionExpandType::AICPU) {
        return GetAicpuExceptionDumpInfo(expandInfo.u.aicpuInfo, &exceptionDumpInfo[0]);
    }
    return 0;
}

int32_t ExceptionDumpCallBack(AclRtExceptionInfo* exceptionInfo, AdxExceptionDumpInfo* exceptionDumpInfo,
                              uint32_t exceptionDumpSize, uint32_t* exceptionDumpRealSize, AdxExceptionDumpMode* mode)
{
    MACHINE_LOGI("ExceptionDumpCallBack enter: exceptionInfo");
    if (exceptionInfo == nullptr || exceptionDumpInfo == nullptr || exceptionDumpRealSize == nullptr ||
        mode == nullptr) {
        MACHINE_LOGW("DeviceExceptionDumpCallBack failed: the input params is invalid [%p, %p, %p, %p]",
                     (void*)exceptionInfo, (void*)exceptionDumpInfo, (void*)exceptionDumpRealSize, (void*)(mode));
        return static_cast<int32_t>(npu::tile_fwk::MachineError::DUMP_DFX);
    }
    *mode = AdxExceptionDumpMode::ADX_DUMP_MODE_OVERWRITE;
    *exceptionDumpRealSize = exceptionDumpSize;
    return DeviceExceptionDumpCallBack(exceptionInfo, exceptionDumpInfo, exceptionDumpSize);
}

int32_t AdumpRegExceptionDump() { return AdumpRegExceptionDumpCallBack(ExceptionDumpCallBack); }
} // namespace npu::tile_fwk::dynamic
