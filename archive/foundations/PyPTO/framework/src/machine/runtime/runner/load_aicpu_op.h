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
 * \file load_aicpu_op.h
 * \brief
 */

#ifndef LOAD_AICPU_OP_H
#define LOAD_AICPU_OP_H
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "adapter/api/runtime_define.h"
#include "machine/utils/machine_ws_intf.h"

namespace npu::tile_fwk {
struct AicpuHostInput {
    // device 上待刷新的位置， launch时。rt
    // 会申请Device内存并将该device内存地址刷新到hostArgs中，该参数描述就是这个待刷新的位置的偏移
    uint32_t addrOffset = 0;
    // 数据偏移区，指向的数据去需要拷贝到device侧， 该参数用于指定数据区基于hostArgs的地址偏移(主要描述input 信息)
    uint32_t dataOffset = 0;
};

struct AicpuLaunchDesc {
    void* stream = nullptr;
    // args 表示的host侧 DeviceKernelArgs + input info 等组装的info
    AiCpuArgs* args = nullptr;
    uint32_t argsSize = 0;
    AicpuHostInput* hostInputs = nullptr;
    uint16_t hostInputNum = 0;
    uint16_t timeout = 0;
    uint32_t blockDim = 1;
    uint32_t kernelNameOffset = 0;
    uint32_t soNameOffset = 0;
};

class LoadAicpuOp {
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp() {}
    static int AicpuKernelLaunch(void* funcHandle, const AicpuLaunchDesc& desc);
    static int LaunchWithHostArgs(void* funcHandle, const AicpuLaunchDesc& desc);
    int LaunchBuiltInOp(const AicpuLaunchDesc& desc, const std::string& funcName) const;
    int LaunchBuiltInOpWithHostArgs(const AicpuLaunchDesc& desc, const std::string& funcName) const;
    int GetBuiltInOpBinHandle(int64_t* devArgsAddr);
    int LaunchCustomOp([[maybe_unused]] const AicpuLaunchDesc& desc, [[maybe_unused]] std::string& OpType) const;
    int LaunchAicpuServerInit(int64_t* devArgsAddr);
    void CustomAiCpuSoLoad();
    void GenBuiltInOpInfo();
    static LoadAicpuOp& GetInstance()
    {
        static LoadAicpuOp loadCustomAicpuOp;
        return loadCustomAicpuOp;
    }

private:
    void* customBinHandle_ = nullptr;
    bool isPyptoNullLaunched_ = false;
    std::string builtInOpJsonPath_;
    std::unordered_map<std::string, RtFuncHandle> builtInFuncMap_;
};
} // namespace npu::tile_fwk
#endif
