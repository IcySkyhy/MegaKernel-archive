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
 * \file derivation_tile_shape.h
 * \brief
 */

#ifndef PASS_DERIVATION_TILE_SHAPE_H_
#define PASS_DERIVATION_TILE_SHAPE_H_

#include <vector>
#include <unordered_map>

#include "passes/pass_interface/pass.h"
#include "interface/inner/tilefwk.h"
#include "interface/tensor/logical_tensor.h"

namespace npu {
namespace tile_fwk {
/* 轴变形操作，0:不涉及变形；1:涉及拆轴；2：涉及合轴；3：其他变形 */
enum AxisTransformType : int32_t { AXIS_KEEP = 0, AXIS_SPLIT = 1, AXIS_MERGE = 2, AXIS_RESHAPE = 3 };

/* input shape->align shape 和 align shape->output shape轴的状态结构体, input shape只可能拆轴，align shape只可能合轴 */
struct ShapeStatus {
    int64_t size;
    int64_t tileSize;
    AxisTransformType axisType;
    /* 拆轴/合轴后轴的下标集合 */
    std::vector<size_t> transformAxisIndex;
    int64_t stride;

    ShapeStatus() : size(0), tileSize(0), axisType(AXIS_KEEP), transformAxisIndex(), stride(0) {}
    ShapeStatus(int64_t s, int64_t ts = 0, AxisTransformType type = AXIS_KEEP, const std::vector<size_t>& indices = {},
                int64_t st = 0)
        : size(s), tileSize(ts), axisType(type), transformAxisIndex(indices), stride(st)
    {}
};

struct AlignContext {
    size_t i = 0UL;    // 记录输入shape的下标
    size_t o = 0UL;    // 记录输出shape的下标
    size_t a = 0UL;    // 记录aligned shape的下标
    int64_t iprod = 1; // 记录输入较大时的整除结果
    int64_t oprod = 1; // 记录输出较大时的整除结果
};

class DerivationTileShape {
public:
    DerivationTileShape() = default;
    ~DerivationTileShape() = default;

    /*
     * 输入参数：op为需要推导的op， inShape、inTileShape表示传入Tensor的shape和tileshape，
     * outShape为需要推导tileshape的tensor shape 输出参数：返回值Status和outTileShape
     * 返回值为SUCCESS时，outTileShape正常赋值； 返回值为FAILED时未赋值
     */
    Status DerivationReshapeTileShape(Operation* op, const Shape& inShape, const Shape& outShape,
                                      const std::vector<int64_t>& inTileShape, std::vector<int64_t>& outTileShape);
};
} // namespace tile_fwk
} // namespace npu
#endif // PASS_DERIVATION_TILE_SHAPE_H_
