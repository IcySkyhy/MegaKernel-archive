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
 * \file adump_api.cpp
 * \brief
 */

#include "adapter/api/adump_api.h"

#ifdef BUILD_WITH_CANN
#include "adapter/manager/adapter_manager.h"
#include "acl/acl_base_rt.h"
#include "dump/adump_pub.h"
#endif
#include "adapter/stubs/adump_stubs.h"

namespace npu::tile_fwk {
uint64_t AdxDumpGetDumpSwitch(const AdxDumpType dumpType)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::GetDumpSwitch);
    if (func != nullptr) {
        uint64_t (*adumpFunc)(const Adx::DumpType) = reinterpret_cast<uint64_t (*)(const Adx::DumpType)>(func);
        return adumpFunc(static_cast<Adx::DumpType>(dumpType));
    }
#endif
    return StubDumpGetDumpSwitch(dumpType);
}

int32_t AdumpRegExceptionDumpCallBack(AdumpExceptionDumpCallback callback)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpFailTaskExceptionCallBack);
    if (func != nullptr) {
        int32_t (*adumpFailRegFunc)(
            AdumpExceptionDumpCallback) = reinterpret_cast<int32_t (*)(AdumpExceptionDumpCallback)>(func);
        return adumpFailRegFunc(callback);
    }
#endif
    (void)callback;
    return 0;
}

int32_t AdxDumpDataDumpServerInit()
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DataDumpServerInit);
    if (func != nullptr) {
        int32_t (*adumpFunc)() = reinterpret_cast<int32_t (*)()>(func);
        return adumpFunc();
    }
#endif
    return StubDumpDataDumpServerInit();
}

int32_t AdxDumpDataDumpServerUnInit()
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DataDumpServerUnInit);
    if (func != nullptr) {
        int32_t (*adumpFunc)() = reinterpret_cast<int32_t (*)()>(func);
        return adumpFunc();
    }
#endif
    return StubDumpDataDumpServerUnInit();
}

#ifdef BUILD_WITH_CANN
void ConvertTensorInfos(const std::vector<AdxTensorInfoV2>& tensors, std::vector<Adx::TensorInfoV2>& adxTensors)
{
    for (const AdxTensorInfoV2& tensor : tensors) {
        Adx::TensorInfoV2 adxTensorInfo;
        adxTensorInfo.type = static_cast<Adx::TensorType>(tensor.type);
        adxTensorInfo.tensorSize = tensor.tensorSize;
        adxTensorInfo.format = tensor.format;
        adxTensorInfo.dataType = tensor.dataType;
        adxTensorInfo.tensorAddr = tensor.tensorAddr;
        adxTensorInfo.addrType = static_cast<Adx::AddressType>(tensor.addrType);
        adxTensorInfo.placement = tensor.placement;
        adxTensorInfo.argsOffSet = tensor.argsOffSet;
        adxTensorInfo.shape = tensor.shape;
        adxTensorInfo.originShape = tensor.originShape;
        adxTensors.push_back(adxTensorInfo);
    }
}
#endif

int32_t AdxDumpDumpTensorV2(const std::string& opType, const std::string& opName,
                            const std::vector<AdxTensorInfoV2>& tensors, AclRtStream stream)
{
#ifdef BUILD_WITH_CANN
    void* func = AdapterManager::Instance().GetAdumpAdapter().GetFunction(AdumpFunc::DumpTensorV2);
    if (func != nullptr) {
        int32_t (*adumpFunc)(
            const std::string&, const std::string&, const std::vector<Adx::TensorInfoV2>&,
            aclrtStream) = reinterpret_cast<int32_t (*)(const std::string&, const std::string&,
                                                        const std::vector<Adx::TensorInfoV2>&, aclrtStream)>(func);
        std::vector<Adx::TensorInfoV2> adxTensors;
        ConvertTensorInfos(tensors, adxTensors);
        return adumpFunc(opType, opName, adxTensors, stream);
    }
#endif
    return StubDumpDumpTensorV2(opType, opName, tensors, stream);
}

#ifdef BUILD_WITH_CANN
static_assert(static_cast<int32_t>(AdxDumpType::OPERATOR) == static_cast<int32_t>(Adx::DumpType::OPERATOR));
static_assert(static_cast<int32_t>(AdxTensorType::INPUT) == static_cast<int32_t>(Adx::TensorType::INPUT));
static_assert(static_cast<int32_t>(AdxAddressType::TRADITIONAL) == static_cast<int32_t>(Adx::AddressType::TRADITIONAL));
static_assert(static_cast<int32_t>(AdxTensorPlacement::kOnDeviceHbm) ==
              static_cast<int32_t>(Adx::TensorPlacement::kOnDeviceHbm));
static_assert(sizeof(AdxDumpType) == sizeof(Adx::DumpType));
static_assert(sizeof(AdxTensorType) == sizeof(Adx::TensorType));
static_assert(sizeof(AdxAddressType) == sizeof(Adx::AddressType));
static_assert(sizeof(AdxTensorPlacement) == sizeof(Adx::TensorPlacement));
static_assert(sizeof(AdxTensorInfoV2) == sizeof(Adx::TensorInfoV2));
#endif
} // namespace npu::tile_fwk
