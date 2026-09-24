/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file block_ops/sort.cpp
 * \brief Block sorting operations with explicit output tiles: sort32, mrgsort (format1), mrgsort2 (format2).
 */

#include <any>
#include <string>
#include <utility>
#include <vector>

#include "core/logging.h"
#include "ir/kind_traits.h"
#include "ir/op_registry.h"
#include "ir/type.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// sort32
// ---------------------------------------------------------------------------

REGISTER_OP("block.sort32")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output sort fixed-size 32-element blocks with index mapping. "
                     "Sorts each 32-element block and produces sorted values and permutation indices.")
    .add_argument("dst", "Output tile for sorted values (TileType)")
    .add_argument("src", "Input tile (TileType)")
    .add_argument("idx", "Input/output tile for permutation indices (TileType, UINT32)")
    .add_argument("tmp", "Optional scratch tile (TileType)")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 3 || args.size() == 4)
            << "block.sort32 requires 3 or 4 arguments (dst, src, idx[, tmp]), but got " << args.size();
        auto dst_type = As<TileType>(args[0]->GetType());
        CHECK(dst_type) << "block.sort32: dst must be TileType, but got " << args[0]->GetType()->TypeName();
        return dst_type;
    });

// ---------------------------------------------------------------------------
// mrgsort (format1): single tile, block-wise merge
// ---------------------------------------------------------------------------

REGISTER_OP("block.mrgsort")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output merge sort on sorted lists (format1). "
                     "Performs merge sort with specified block length.")
    .add_argument("dst", "Output tile (TileType)")
    .add_argument("src", "Input tile (TileType)")
    .set_attr<int>("block_len")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 2) << "block.mrgsort requires 2 arguments (dst, src) for format1, but got " << args.size();
        auto dst_type = As<TileType>(args[0]->GetType());
        CHECK(dst_type) << "block.mrgsort: dst must be TileType, but got " << args[0]->GetType()->TypeName();
        return dst_type;
    });

// ---------------------------------------------------------------------------
// mrgsort2 (format2): multi-list top-K merge (2, 3, or 4 sources)
// ---------------------------------------------------------------------------

REGISTER_OP("block.mrgsort2")
    .set_op_category("BlockOp")
    .set_description("Block explicit-output merge sort on multiple sorted lists (format2). "
                     "Merges 2, 3, or 4 pre-sorted source tiles into dst using tmp buffer.")
    .add_argument("dst", "Output tile (TileType)")
    .add_argument("src0", "First input tile (TileType)")
    .add_argument("tmp", "Temporary buffer tile (TileType, same shape as dst)")
    .add_argument("src1", "Second input tile (TileType)")
    .set_attr<bool>("exhausted")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() >= 4 && args.size() <= 6)
            << "block.mrgsort2 requires 4-6 arguments (dst, src0, tmp, src1, [src2, src3]), but got " << args.size();
        auto dst_type = As<TileType>(args[0]->GetType());
        CHECK(dst_type) << "block.mrgsort2: dst must be TileType, but got " << args[0]->GetType()->TypeName();
        return dst_type;
    });

REGISTER_OP("block.histogram")
    .set_op_category("BlockOp")
    .set_description("Radix sort histogram accumulation for 256-bin histograms. "
                     "Only supported on A5. Builds histogram for radix sort preprocessing.")
    .add_argument("dst", "Destination histogram tile (TileType, dtype=UINT32, cols=256)")
    .add_argument("src", "Source tile (TileType, dtype=UINT16 or UINT32)")
    .add_argument("idx", "Index tile for LSB filtering (TileType, dtype=UINT8)")
    .set_attr<bool>("is_msb")
    .f_deduce_type([]([[maybe_unused]] const std::vector<ExprPtr>& args,
                      [[maybe_unused]] const std::vector<std::pair<std::string, std::any>>& kwargs) {
        CHECK(args.size() == 3) << "block.histogram requires 3 arguments (dst, src, idx), but got " << args.size();
        auto dst_type = As<TileType>(args[0]->GetType());
        CHECK(dst_type) << "block.histogram: dst must be TileType, but got " << args[0]->GetType()->TypeName();
        return dst_type;
    });

} // namespace ir
} // namespace pypto
