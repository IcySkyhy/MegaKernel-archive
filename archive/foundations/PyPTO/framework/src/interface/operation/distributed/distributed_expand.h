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
 * \file distributed_expand.h
 * \brief
 */

#ifndef DISTRIBUTED_EXPAND_H
#define DISTRIBUTED_EXPAND_H

#include <cstdint>
#include <vector>
#include <memory>
#include "interface/configs/config_manager.h"
#include "interface/function/function.h"
#include "interface/operation/operation.h"
#include "tilefwk/error_code.h"

namespace npu::tile_fwk {
namespace Distributed {
inline void CreateTileOp(const TileShape& tileShape,
                         const std::function<void(int32_t, int32_t, int32_t, int32_t, int32_t)>& callback)
{
    const auto& tileRow = tileShape.GetDistTileRow();
    const auto& tileCol = tileShape.GetDistTileCol();
    int32_t rowCount = tileRow[1] + (tileRow[2] == 0 ? 0 : 1);
    int32_t colCount = tileCol[1] + (tileCol[2] == 0 ? 0 : 1);
    CHECK(DistributedErrorCode::INVALID_TILE_DIM, tileRow[0] > 0)
        << "Invalid tiling strategy of the row axis: the first number must be greater than 0, but got " << tileRow[0];
    CHECK(DistributedErrorCode::INVALID_TILE_DIM, (tileCol[0] > 0))
        << "Invalid tiling strategy of the col axis: the first number must be greater than 0, but got " << tileCol[0];

    int32_t tileIndex = 0;
    for (int32_t rowIndex = 0; rowIndex < rowCount; rowIndex++) {
        int32_t rowShape = ((tileRow[2] != 0) && (rowIndex == rowCount - 1)) ? tileRow[2] : tileRow[0];
        for (int32_t colIndex = 0; colIndex < colCount; colIndex++) {
            int32_t colShape = ((tileCol[2] != 0) && (colIndex == colCount - 1)) ? tileCol[2] : tileCol[0];
            callback(tileIndex, rowIndex * tileRow[0], colIndex * tileCol[0], rowShape, colShape);
            tileIndex++;
        }
    }
}

void TiledShmemPut(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemStore(Function& function, const TileShape& tileShape,
                     const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                     const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemGet(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemLoad(Function& function, const TileShape& tileShape,
                    const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                    const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemSignal(Function& function, const TileShape& tileShape,
                      const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                      const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemWaitUntil(Function& function, const TileShape& tileShape,
                         const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                         const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemSet(Function& function, const TileShape& tileShape,
                   const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                   const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);
void TiledShmemBindTensor(Function& function, const TileShape& tileShape,
                          const std::vector<std::shared_ptr<LogicalTensor>>& iOperand,
                          const std::vector<std::shared_ptr<LogicalTensor>>& oOperand, const Operation& op);

} // namespace Distributed
} // namespace npu::tile_fwk

#endif // DISTRIBUTED_EXPAND_H
