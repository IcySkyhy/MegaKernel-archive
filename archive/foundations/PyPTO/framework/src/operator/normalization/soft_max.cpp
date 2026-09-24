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
 * \file soft_max.cpp
 * \brief
 */

#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
using namespace npu::tile_fwk;

namespace npu::tile_fwk {
Tensor Softmax(const Tensor& operand)
{
    auto tRowmax = RowMaxExpand(operand);
    auto tSub = Sub(operand, tRowmax);
    auto tExp = Exp(tSub);
    auto tEsum = RowSumExpand(tExp);
    auto tSoftmax = Div(tExp, tEsum);

    return tSoftmax;
}

Tensor SoftmaxNew(const Tensor& operand)
{
    // 获取输入数据类型
    auto inputDtype = operand.GetStorage()->Datatype();
    Tensor castOperand = operand;
    // 如果输入数据类型不是FP32，则将其转换为FP32
    if (inputDtype != DataType::DT_FP32) {
        castOperand = Cast(operand, DataType::DT_FP32);
    }
    // 描述计算逻辑
    // M=rowMax(xi)
    auto rowmax = Amax(castOperand, -1, true);
    // S=rowSum(exp(xi-M))
    auto sub = Sub(castOperand, rowmax);
    auto exp = Exp(sub);
    auto esum = Sum(exp, -1, true);
    // softmax(zi)=exp(xi-M)/S
    auto softmax = Div(exp, esum);
    // 如果输出数据类型与输入不同，则进行类型转换
    if (inputDtype != softmax.GetStorage()->Datatype()) {
        softmax = Cast(softmax, inputDtype);
    }
    return softmax;
}

void SoftmaxDynamicCompute(Tensor& input, Tensor& output)
{
    // 获取输入形状信息[b, n1, n2, dim], batch轴动态
    SymbolicScalar b = GetInputShape(input, 0);
    int n1 = input.GetShape()[1];
    int n2 = input.GetShape()[2];
    int dim = input.GetShape()[3];
    // 设置Loop处理的batch大小及循环次数
    int tileB = 1;
    SymbolicScalar bLoop = b / tileB;
    // 定义循环，用于处理每个batch块，每个batch块为(1, 32, 1, 256)
    LOOP("SOFTMAX_LOOP_L0_bIdx", FunctionType::DYNAMIC_LOOP, bIdx, LoopRange(0, bLoop, 1), {}, true)
    {
        // 计算偏移量
        SymbolicScalar bOffset = bIdx * tileB;
        std::vector<SymbolicScalar> outOffset = {bOffset, 0, 0, 0};
        // 对每个batch块进行tile切分，切分大小为(1, 4, 1, 64)
        TileShape::Current().SetVecTile({1, 4, 1, 64});
        // 创建输入视图
        auto inputView = View(input, {tileB, n1, n2, dim}, {bOffset, 0, 0, 0});
        // 调用Softmax算子函数
        auto outputView = SoftmaxNew(inputView);
        // 将输出结果组装到输出Tensor中
        Assemble(outputView, outOffset, output);
    }
}

void SoftmaxDynamic(Tensor& input, Tensor& output)
{
    FUNCTION("SOFTMAX_DYNAMIC_EXAMPLE", {input}, {output}) { SoftmaxDynamicCompute(input, output); }
}
} // namespace npu::tile_fwk
