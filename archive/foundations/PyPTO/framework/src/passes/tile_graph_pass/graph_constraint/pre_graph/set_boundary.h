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
 * \file set_boundary.h
 * \brief
 */

#ifndef PASS_SET_BOUNDARY_H
#define PASS_SET_BOUNDARY_H
#include "pre_graph_common.h"

namespace npu::tile_fwk {
class SetBoundary {
public:
    SetBoundary() {}
    ~SetBoundary() = default;

    void InsertTemporaryCopyIn(Function& function, Operation& op) const;
    void SetTensorBoundary(Function& function) const;
};
} // namespace npu::tile_fwk
#endif // PASS_SET_BOUNDARY_H
