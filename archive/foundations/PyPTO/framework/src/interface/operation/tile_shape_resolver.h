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
 * \file tile_shape_resolver.h
 * \brief Resolve the per-input / per-output tile shape of an op from its opcode and op-level tile shape.
 */

#pragma once

#include "operation.h"
#include "tilefwk/tile_shape.h"

namespace npu::tile_fwk {

// Singleton method class that derives the tile shape of a given input or output tensor
// from the opcode and the op-level tile shape. Per-opcode derivation rules are filled in
// GetInputTileShape / GetOutputTileShape; the default returns the op-level tile shape
// clamped to the operand's own shape (elementwise).
class TileShapeResolver {
public:
    static TileShapeResolver& Instance();

    // Given the op and the input operand index, return the tile shape describing
    // how that input tensor is actually tiled. The op contributes its opcode,
    // op-level tile shape and input tensor shapes, so the result reflects the
    // real per-axis tile size (e.g. clamped to each input's own shape).
    TileShape GetInputTileShape(const Operation& op, int index) const;

    // Given the op and the output operand index, return the tile shape describing
    // how that output tensor is actually tiled (i.e. the first-cut output tile shape
    // produced by the op's TileFunc during expansion). The default rule clamps the
    // op-level VecTile to the output's own shape per axis; cube matmul and a few
    // layout-transform ops override this.
    TileShape GetOutputTileShape(const Operation& op, int index) const;

private:
    TileShapeResolver() = default;
    TileShapeResolver(const TileShapeResolver&) = delete;
    TileShapeResolver& operator=(const TileShapeResolver&) = delete;
};

} // namespace npu::tile_fwk
