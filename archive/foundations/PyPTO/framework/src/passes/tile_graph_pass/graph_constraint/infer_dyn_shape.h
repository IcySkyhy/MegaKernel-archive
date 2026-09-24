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
 * \file infer_dyn_shape.h
 * \brief
 */

#ifndef INFER_DYN_SHAPE_PASS_H_
#define INFER_DYN_SHAPE_PASS_H_
#include "interface/operation/op_infer_shape_impl.h"
#include "passes/pass_interface/pass.h"
#include "interface/function/function.h"
namespace npu {
namespace tile_fwk {
class InferDynShape : public Pass {
public:
    InferDynShape() : Pass("InferDynShape") {}
    ~InferDynShape() override {}
    Status RunOnFunction(Function& function) override;
    Status PostCheck(Function& function) override;
    void RecordStaticValidShapeOnL0CCopyUB(Function& function);
};
} // namespace tile_fwk
} // namespace npu
#endif
