/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COSTMODEL_ST_BINARY_VEC_TEST_CONTEXT_HPP
#define PTO_COSTMODEL_ST_BINARY_VEC_TEST_CONTEXT_HPP

#include <cstdint>
#include <vector>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

namespace pto::test {

template <typename T, int Row, int ValidRow, int Col, int ValidCol, PadValue Pad>
struct BinaryVecTestContext {
    using TileData = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, Pad>;

    TileData src0Tile;
    TileData src1Tile;
    TileData dstTile;
    std::vector<T> src0Buf;
    std::vector<T> src1Buf;
    std::vector<T> dstBuf;

    BinaryVecTestContext()
        : src0Tile(ValidRow, ValidCol),
          src1Tile(ValidRow, ValidCol),
          dstTile(ValidRow, ValidCol),
          src0Buf(Row * Col, T{1}),
          src1Buf(Row * Col, T{1}),
          dstBuf(Row * Col, T{0})
    {
        TASSIGN(src0Tile, reinterpret_cast<std::uintptr_t>(src0Buf.data()));
        TASSIGN(src1Tile, reinterpret_cast<std::uintptr_t>(src1Buf.data()));
        TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));
    }
};

} // namespace pto::test

#endif // PTO_COSTMODEL_ST_BINARY_VEC_TEST_CONTEXT_HPP
