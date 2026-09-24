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
 * @file test_simt_ops.cpp
 * \brief Type-deduction tests for the SIMT operation registry.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dtype.h"
#include "ir/kind_traits.h"
#include "ir/op_attr_types.h"
#include "ir/op_registry.h"
#include "ir/scalar_expr.h"
#include "ir/type.h"
#include "test_op_helpers.h"
#include "tilefwk/error.h"

namespace pypto {
namespace ir {

using namespace test_helpers;

using Kwargs = std::vector<std::pair<std::string, std::any>>;

TEST(SimtOpsTest, ContextAndBuiltinScalarTypes)
{
    auto& registry = OpRegistry::GetInstance();
    const std::vector<std::string> context_ops = {"simt.thread_idx", "simt.block_dim", "simt.block_idx",
                                                  "simt.grid_dim"};

    for (std::size_t axis = 0; axis < context_ops.size(); ++axis) {
        auto call = registry.Create(context_ops[axis], {}, Kwargs{{"axis", static_cast<int>(axis % 3)}}, Sp());
        auto type = As<ScalarType>(call->GetType());
        ASSERT_NE(type, nullptr);
        EXPECT_EQ(type->dtype_, DataType::UINT32);
    }

    auto linear = registry.Create("simt.linear_thread_idx", {}, Sp());
    auto linear_type = As<ScalarType>(linear->GetType());
    ASSERT_NE(linear_type, nullptr);
    EXPECT_EQ(linear_type->dtype_, DataType::UINT32);

    auto warp = registry.Create("simt.warp_size", {}, Sp());
    auto warp_type = As<ScalarType>(warp->GetType());
    ASSERT_NE(warp_type, nullptr);
    EXPECT_EQ(warp_type->dtype_, DataType::INT32);
}

TEST(SimtOpsTest, LaunchType)
{
    auto& registry = OpRegistry::GetInstance();
    auto scalar = MakeScalarVar("count", DataType::UINT32);

    std::vector<ExprPtr> launch_args = {
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(4, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(8, DataType::INT64, Sp()),
        scalar,
    };
    auto launch = registry.Create("simt.launch", launch_args,
                                  Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 256}}, Sp());

    EXPECT_EQ(launch->GetType(), GetUnknownType());
}

TEST(SimtOpsTest, CastTypeDeductionContracts)
{
    auto& registry = OpRegistry::GetInstance();
    auto make_cast = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        return registry.Create("simt.cast", {source},
                               Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp());
    };

    auto rounded = make_cast(DataType::FP32, DataType::FP16, RoundMode::CAST_RINT);
    EXPECT_EQ(rounded->name_, "simt.cast");
    EXPECT_EQ(rounded->GetKwarg<DataType>("target_type"), DataType::FP16);
    EXPECT_EQ(rounded->GetKwarg<int>("mode"), static_cast<int>(RoundMode::CAST_RINT));
    auto rounded_type = As<ScalarType>(rounded->GetType());
    ASSERT_NE(rounded_type, nullptr);
    EXPECT_EQ(rounded_type->dtype_, DataType::FP16);

    struct CastCase {
        DataType source;
        DataType target;
        RoundMode mode;
    };
    const std::vector<CastCase> representative_cases = {
        {DataType::INT32, DataType::UINT64, RoundMode::CAST_NONE},
        {DataType::FP32, DataType::FP32, RoundMode::CAST_NONE},
        {DataType::BF16, DataType::FP32, RoundMode::CAST_NONE},
        {DataType::FP32, DataType::BF16, RoundMode::CAST_ROUND},
        {DataType::FP32, DataType::UINT64, RoundMode::CAST_FLOOR},
        {DataType::INT64, DataType::FP32, RoundMode::CAST_TRUNC},
        {DataType::FP32, DataType::FP16, RoundMode::CAST_ODD},
    };
    for (const auto& cast_case : representative_cases) {
        auto result = make_cast(cast_case.source, cast_case.target, cast_case.mode);
        auto result_type = As<ScalarType>(result->GetType());
        ASSERT_NE(result_type, nullptr);
        EXPECT_EQ(result_type->dtype_, cast_case.target);
    }
}

TEST(SimtOpsTest, CastRejectsUnsupportedDtypeAndModeCombinations)
{
    auto& registry = OpRegistry::GetInstance();
    auto expect_rejected = [&registry](DataType source_dtype, DataType target_dtype, RoundMode mode) {
        auto source = MakeScalarVar("value", source_dtype);
        EXPECT_THROW(
            (void)registry.Create("simt.cast", {source},
                                  Kwargs{{"target_type", target_dtype}, {"mode", static_cast<int>(mode)}}, Sp()),
            npu::tile_fwk::Error);
    };

    expect_rejected(DataType::FP16, DataType::INT32, RoundMode::CAST_NONE);
    expect_rejected(DataType::FP32, DataType::BF16, RoundMode::CAST_ODD);
    expect_rejected(DataType::INT32, DataType::INT64, RoundMode::CAST_RINT);
}

TEST(SimtOpsTest, RejectsInvalidContextAxisAndLaunchBounds)
{
    auto& registry = OpRegistry::GetInstance();
    EXPECT_THROW((void)registry.Create("simt.thread_idx", {}, Kwargs{{"axis", 3}}, Sp()), npu::tile_fwk::Error);

    std::vector<ExprPtr> too_many_threads = {
        std::make_shared<ConstInt>(2048, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(2, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
    };
    EXPECT_THROW((void)registry.Create("simt.launch", too_many_threads,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 2048}}, Sp()),
                 npu::tile_fwk::Error);
}

TEST(SimtOpsTest, RejectsInvalidLaunchArgumentAbi)
{
    auto& registry = OpRegistry::GetInstance();
    auto invalid_tensor = MakeTensorVar("tensor", {256}, DataType::FP32);
    std::vector<ExprPtr> tensor_args = {
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        std::make_shared<ConstInt>(1, DataType::INT64, Sp()),
        invalid_tensor,
    };
    EXPECT_THROW((void)registry.Create("simt.launch", tensor_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);

    auto invalid_tile = MakeTileVar("tile", {1, 256}, DataType::FP32);
    std::vector<ExprPtr> tile_args = {tensor_args[0], tensor_args[1], tensor_args[2], invalid_tile};
    EXPECT_THROW((void)registry.Create("simt.launch", tile_args,
                                       Kwargs{{"callee", std::string("simt_entry")}, {"max_threads", 1}}, Sp()),
                 npu::tile_fwk::Error);
}
} // namespace ir
} // namespace pypto
