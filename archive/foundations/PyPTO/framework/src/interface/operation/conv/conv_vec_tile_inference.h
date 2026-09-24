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
 * \file conv_vec_tile_inference.h
 * \brief Infer VecTileShape for fmap/weight/out/reshape from ConvTileShape.
 */

#ifndef FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_VEC_TILE_INFERENCE_H
#define FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_VEC_TILE_INFERENCE_H

#include "tilefwk/tile_shape.h"
#include "tilefwk/tilefwk_op.h"
#include "tilefwk/data_type.h"

namespace npu {
namespace tile_fwk {

class Operation;

namespace Conv {

struct ConvVecTileShapes {
    VecTile fmapVecTile;
    VecTile weightVecTile;
    VecTile outVecTile;
};

ConvVecTileShapes InferConvVecTileShapes(const Operation& convOp, DataType dtype);

ConvVecTileShapes InferConvVecTileShapes(const ConvTile& convTile, DataType dtype,
                                         const std::vector<int64_t>& oriFmapShape,
                                         const std::vector<int64_t>& oriWeightShape, bool isConv3D, bool isConv1D,
                                         int64_t groups);

VecTile SelectConvVecTile(const ConvVecTileShapes& vecTiles, TileOpFormat targetFormat);

VecTile GetReshapeVecTile(const VecTile& srcVecTile, bool isConv1D);

} // namespace Conv
} // namespace tile_fwk
} // namespace npu

#endif // FRAMEWORK_SRC_INTERFACE_OPERATION_CONV_VEC_TILE_INFERENCE_H
