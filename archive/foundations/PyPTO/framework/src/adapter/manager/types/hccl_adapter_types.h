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
 * \file hccl_adapter_types.h
 * \brief
 */

#pragma once

#include <string>
#include <map>

namespace npu::tile_fwk {
enum class HcclFunc {
    GetCommName = 0,
    GetL0TopoTypeEx,
    GetCommHandleByGroup,
    GetRootInfo,
    CommInitRootInfo,
    CommDestroy,
    AllocComResourceByTiling,
    Bottom
};

const std::string kHcclLibName = "libhccl.so";
const std::map<HcclFunc, std::string> kHcclFuncStrMap{
    {HcclFunc::GetCommName, "HcclGetCommName"},
    {HcclFunc::GetL0TopoTypeEx, "HcomGetL0TopoTypeEx"},
    {HcclFunc::GetCommHandleByGroup, "HcomGetCommHandleByGroup"},
    {HcclFunc::GetRootInfo, "HcclGetRootInfo"},
    {HcclFunc::CommDestroy, "HcclCommDestroy"},
    {HcclFunc::CommInitRootInfo, "HcclCommInitRootInfo"},
    {HcclFunc::AllocComResourceByTiling, "HcclAllocComResourceByTiling"}};
} // namespace npu::tile_fwk
