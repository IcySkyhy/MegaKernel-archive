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
 * \file kernel_arg.h
 * \brief
 */

#ifndef __ASCENDTENSOR_KERNEL_ARG_H__
#define __ASCENDTENSOR_KERNEL_ARG_H__

#include <string>
#include <memory>
#include <map>
#include <iostream>
#include "interface/tensor/logical_tensor.h"

namespace npu::tile_fwk {

struct ArgInfo {
    std::string name;
    std::vector<int> offset;
    std::shared_ptr<RawTensor> tensor;
};

class KernelArg {
public:
    bool IsOpArgs(std::shared_ptr<RawTensor> tensor)
    {
        // if (op_args_.empty()) {
        //     return true;
        // }
        return op_args_.find(tensor->rawmagic) != op_args_.end();
    }

    std::vector<std::string> GetOpArgName()
    {
        std::vector<std::string> res;
        for (const auto& ele : op_args_) {
            res.emplace_back(ele.second.name);
        }
        return res;
    }

    void SetOpArgs(std::vector<std::shared_ptr<LogicalTensor>> tensors)
    {
        for (auto tensor : tensors) {
            int magic = tensor->tensor->rawmagic;
            if (IsOpArgs(tensor->tensor)) {
                auto& arg = op_args_[magic];
                assert(tensor.GetStorage()->Symbol() == arg.name);
                assert(tensor->offset == arg.offset);
            } else {
                std::cout << "Add args " << magic << " " << tensor.GetStorage()->Symbol() << std::endl;
                op_args_[magic] = {tensor.GetStorage()->Symbol(), tensor->offset, tensor->tensor};
            }
        }
    }

    std::string GetOpArgName(std::shared_ptr<RawTensor> tensor)
    {
        for ([[maybe_unused]] const auto& ele : op_args_) {
            if (op_args_.find(tensor->rawmagic) != op_args_.end()) {
                return op_args_[tensor->rawmagic].name;
            }
        }
        return "";
    }

    void clear() { op_args_.clear(); }

private:
    std::map<int, ArgInfo> op_args_; // op input/output args
};

inline KernelArg& GetKernelArg()
{
    static KernelArg ka;
    return ka;
}
} // namespace npu::tile_fwk

#endif
