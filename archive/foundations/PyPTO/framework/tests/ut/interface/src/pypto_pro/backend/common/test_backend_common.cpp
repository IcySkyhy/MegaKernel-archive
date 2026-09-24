/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <any>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "backend/common/backend_registry.h"
#include "backend/common/backend_utils.h"
#include "backend/common/soc.h"
#include "core/error.h"
#include "ir/expr.h"
#include "ir/scalar_expr.h"
#include "ir/span.h"
#include "ir/type.h"

namespace pypto {
namespace backend {

TEST(BackendCommonTest, RegistersCCEAsSingletonBackend)
{
    auto& registry = BackendRegistry::Instance();
    EXPECT_TRUE(registry.IsRegistered("CCE"));
    EXPECT_FALSE(registry.IsRegistered("910B_CCE"));
    EXPECT_THROW(registry.Create("CCE", nullptr), ir::ValueError);
    EXPECT_THROW(CreateBackendFromRegistry("CCE", nullptr), ir::ValueError);
}

TEST(BackendCommonTest, CreatesSingletonSoC)
{
    const SoC& first = CreateSoC();
    const SoC& second = CreateSoC();

    EXPECT_EQ(&first, &second);
    EXPECT_GT(first.TotalDieCount(), 0);
    EXPECT_GT(first.TotalCoreCount(), 0);
}

TEST(BackendCommonTest, GetsMutexIdCandidatesFromExplicitConstantsOrMaxId)
{
    using Kwargs = std::vector<std::pair<std::string, std::any>>;
    const auto span = ir::Span::Unknown();
    auto dynamic_id = std::make_shared<const ir::Var>(
        "mutex_id", std::make_shared<const ir::ScalarType>(ir::DataType::INDEX), span);

    auto explicit_ids = std::make_shared<const ir::Call>(
        "system.mutex_lock_dyn", std::vector<ir::ExprPtr>{dynamic_id},
        Kwargs{{"pipe", 1}, {"max_mutex_id", 2}, {"mutex_ids", std::vector<int>{8, 9}}}, span);
    EXPECT_EQ(mutex_id::GetMutexIds(explicit_ids), (std::vector<int>{8, 9}));

    auto constant_ids = std::make_shared<const ir::Call>(
        "system.mutex_lock_dyn",
        std::vector<ir::ExprPtr>{
            std::make_shared<const ir::ConstInt>(8, ir::DataType::INT64, span),
            std::make_shared<const ir::ConstInt>(3, ir::DataType::INT64, span),
        },
        Kwargs{{"pipe", 1}}, span);
    EXPECT_EQ(mutex_id::GetMutexIds(constant_ids), (std::vector<int>{8, 3}));

    auto max_id_fallback = std::make_shared<const ir::Call>(
        "system.mutex_lock_dyn",
        std::vector<ir::ExprPtr>{
            std::make_shared<const ir::ConstInt>(1, ir::DataType::INT64, span),
            dynamic_id,
        },
        Kwargs{{"pipe", 1}, {"max_mutex_id", 3}}, span);
    EXPECT_EQ(mutex_id::GetMutexIds(max_id_fallback), (std::vector<int>{0, 1, 2}));
}

TEST(BackendCommonTest, SupportsPointerPrintfConversion)
{
    EXPECT_TRUE(debug_printf::IsSupportedPrintfConversion('p'));
    EXPECT_EQ(debug_printf::FindPrintfConversionIndex("address=%p"), 9);

    auto segments = debug_printf::ParsePrintfSegments("address=%p");
    ASSERT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].format_segment, "address=%p");
    EXPECT_EQ(segments[0].conversion, 'p');
}

} // namespace backend
} // namespace pypto
